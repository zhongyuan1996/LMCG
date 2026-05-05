# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
2D VAE for CXR (Chest X-Ray) grayscale images.
Input shape: [B, 1, H, W] (grayscale)
Output latents: [B, z_dim, H', W'] (spatially downsampled)
"""
import logging
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

__all__ = [
    'CxrAutoencoder2D',
]


# ============================================================================
# Building blocks for 2D VAE (adapted from WanVAE 3D)
# ============================================================================

class RMS_norm_2d(nn.Module):
    """RMS normalization for 2D images."""
    def __init__(self, dim, bias=False):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1)) if bias else 0.

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class Upsample2d(nn.Upsample):
    """Fix bfloat16 support for nearest neighbor interpolation."""
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class ResidualBlock2d(nn.Module):
    """2D residual block with Conv2d."""
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.residual = nn.Sequential(
            RMS_norm_2d(in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            RMS_norm_2d(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock2d(nn.Module):
    """2D self-attention block (spatial attention)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.norm = RMS_norm_2d(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        # Zero out the last layer params
        nn.init.zeros_(self.proj.weight)
    
    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        
        x = self.norm(x)
        # Compute query, key, value
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        q = q.flatten(2).transpose(1, 2)  # [B, H*W, C]
        k = k.flatten(2).transpose(1, 2)  # [B, H*W, C]
        v = v.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Apply attention
        x = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(b, c, h, w)
        
        # Output
        x = self.proj(x)
        return x + identity


class Resample2d(nn.Module):
    """2D resampling (downsample/upsample) block."""
    def __init__(self, dim, mode):
        assert mode in ('downsample2d', 'upsample2d', 'none')
        super().__init__()
        self.dim = dim
        self.mode = mode
        
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample2d(scale_factor=2.0, mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=2)
            )
        else:
            self.resample = nn.Identity()
    
    def forward(self, x):
        return self.resample(x)


class Encoder2d(nn.Module):
    """
    2D encoder for grayscale images.
    
    Assumes input CXR images are resized to 256×256.
    Encodes to 16×16 latent grid (256 tokens total).
    Architecture: 4 downsampling stages (256 → 128 → 64 → 32 → 16).
    """
    def __init__(self,
                 dim=128,
                 z_dim=4,  # This will be z_dim * 2 from the wrapper
                 dim_mult=[1, 2, 4, 8, 8],  # 5 stages total: initial + 4 downsampling
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 in_channels=1):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim  # This is already z_dim * 2
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        
        # Dimensions: [dim, dim*1, dim*2, dim*4, dim*8, dim*8]
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        
        # Init block
        self.conv1 = nn.Conv2d(in_channels, dims[0], 3, padding=1)
        
        # Downsample blocks - need 4 downsampling stages for 256 → 16
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock2d(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock2d(out_dim))
                in_dim = out_dim
            
            # Downsample block - always downsample except at the last stage
            # We need exactly 4 downsample stages: 256→128→64→32→16
            if i < len(dim_mult) - 1:
                downsamples.append(Resample2d(out_dim, mode='downsample2d'))
                scale /= 2.0
        
        self.downsamples = nn.Sequential(*downsamples)
        
        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock2d(out_dim, out_dim, dropout),
            AttentionBlock2d(out_dim),
            ResidualBlock2d(out_dim, out_dim, dropout)
        )
        
        # Output blocks - z_dim here is already z_dim * 2, so output z_dim channels
        self.head = nn.Sequential(
            RMS_norm_2d(out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, z_dim, 3, padding=1)  # Output z_dim (which is z_dim*2 from wrapper)
        )
    
    def forward(self, x):
        # Input: [B, 1, 256, 256]
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        x = self.head(x)
        # Output: [B, z_dim * 2, 16, 16]
        assert x.shape[-2:] == (16, 16), f"Expected 16x16 latents, got {x.shape[-2:]}"
        return x


class Decoder2d(nn.Module):
    """
    2D decoder for grayscale images.
    
    Mirrors Encoder2d architecture.
    Decodes from 16×16 latent grid to 256×256 reconstruction.
    Architecture: 4 upsampling stages (16 → 32 → 64 → 128 → 256).
    """
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 8, 8],  # Must match Encoder2d
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 out_channels=1):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        
        # Dimensions (reversed for decoder)
        # Encoder dims: [dim, dim*1, dim*2, dim*4, dim*8, dim*8]
        # Decoder dims: [dim*8, dim*8, dim*4, dim*2, dim*1, dim]
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 1)  # Start at 1/16 scale
        
        # Init block
        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)
        
        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock2d(dims[0], dims[0], dropout),
            AttentionBlock2d(dims[0]),
            ResidualBlock2d(dims[0], dims[0], dropout)
        )
        
        # Upsample blocks - need 4 upsampling stages for 16 → 256
        # After each upsample2d, channels are halved, so we need to account for that
        upsamples = []
        current_dim = dims[0]  # Start with first dimension
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            
            # Residual (+attention) blocks
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock2d(current_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock2d(out_dim))
                current_dim = out_dim
            
            # Upsample block - always upsample except at the last stage
            # We need exactly 4 upsample stages: 16→32→64→128→256
            if i < len(dim_mult) - 1:
                upsamples.append(Resample2d(out_dim, mode='upsample2d'))
                current_dim = out_dim // 2  # Resample2d halves channels
                scale *= 2.0
        
        self.upsamples = nn.Sequential(*upsamples)
        
        # Output blocks - current_dim after all upsamples (last stage, no halving)
        self.head = nn.Sequential(
            RMS_norm_2d(current_dim),
            nn.SiLU(),
            nn.Conv2d(current_dim, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        # Input: [B, z_dim, 16, 16]
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.head(x)
        # Output: [B, 1, 256, 256]
        assert x.shape[-2:] == (256, 256), f"Expected 256x256 recon, got {x.shape[-2:]}"
        return x


class CxrAutoencoder2D_(nn.Module):
    """
    2D VAE for grayscale CXR images.
    
    Architecture:
    - Encoder: 2D convolutional encoder (no temporal dimension)
    - Decoder: 2D convolutional decoder (no temporal dimension)
    - Input: [B, 1, 256, 256] grayscale images (assumes input CXR images are resized to 256×256)
    - Latents: [B, z_dim, 16, 16] = 256 tokens per image
    
    The encoder performs 4 downsampling stages: 256 → 128 → 64 → 32 → 16
    The decoder performs 4 upsampling stages: 16 → 32 → 64 → 128 → 256
    """
    
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 8, 8],  # 5 stages: initial + 4 downsampling stages
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 in_channels=1,  # Grayscale
                 out_channels=1,  # Grayscale
                 resolution=768):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        
        # Encoder: 2D conv layers (no temporal)
        self.encoder = Encoder2d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, dropout, in_channels)
        
        # Decoder: 2D conv layers (no temporal)
        self.decoder = Decoder2d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, dropout, out_channels)
        
        # Head convs for mu and log_var (encoder outputs z_dim*2, we split it)
        self.conv_mu = nn.Conv2d(z_dim * 2, z_dim, 1)
        self.conv_logvar = nn.Conv2d(z_dim * 2, z_dim, 1)
    
    def encode(self, x, scale):
        """
        Encode grayscale images to latents.
        
        Args:
            x: [B, 1, H, W] grayscale images
            scale: [mean, std] for normalization
                  - During training: use scale=[0, 1] (no normalization)
                  - During inference: use scale=[latent_mean, 1/latent_std] (normalization)
        
        Returns:
            mu: [B, z_dim, H', W'] mean
            log_var: [B, z_dim, H', W'] log variance
            features: intermediate features (optional)
        """
        out = self.encoder(x)  # [B, z_dim*2, H', W']
        mu = self.conv_mu(out)  # [B, z_dim, H', W']
        log_var = self.conv_logvar(out)  # [B, z_dim, H', W']
        
        # Apply scale normalization only if scale != [0, 1] (inference time)
        # During training, scale=[0, 1] means no normalization
        # At inference: scale=[mean, 1/std], so normalize: mu_norm = (mu - mean) * (1/std)
        if isinstance(scale[0], torch.Tensor):
            mean = scale[0]
            inv_std = scale[1]  # This is 1/std
            # Check if this is the no-op case [0, 1]
            if not (torch.allclose(mean, torch.zeros_like(mean), atol=1e-6) and 
                    torch.allclose(inv_std, torch.ones_like(inv_std), atol=1e-6)):
                mu = (mu - mean.view(1, self.z_dim, 1, 1)) * inv_std.view(1, self.z_dim, 1, 1)
        else:
            if abs(scale[0]) > 1e-6 or abs(scale[1] - 1.0) > 1e-6:
                mu = (mu - scale[0]) * scale[1]
        
        return mu, log_var, out
    
    def decode(self, z, scale):
        """
        Decode latents back to grayscale images.
        
        Args:
            z: [B, z_dim, H', W'] latents
            scale: [mean, std] for denormalization
                  - During training: use scale=[0, 1] (no denormalization)
                  - During inference: use scale=[latent_mean, 1/latent_std] (denormalization)
        
        Returns:
            x_recon: [B, 1, H, W] reconstructed grayscale images
        """
        # Apply inverse scale denormalization only if scale != [0, 1] (inference time)
        # During training, scale=[0, 1] means no denormalization
        # At inference: scale=[mean, 1/std], so denormalize: z = z * std + mean
        # Since scale[1] = 1/std, we need: z = z * (1/scale[1]) + scale[0] = z * std + mean
        if isinstance(scale[0], torch.Tensor):
            mean = scale[0]
            inv_std = scale[1]  # This is 1/std
            # Check if this is the no-op case [0, 1]
            if not (torch.allclose(mean, torch.zeros_like(mean), atol=1e-6) and 
                    torch.allclose(inv_std, torch.ones_like(inv_std), atol=1e-6)):
                # Denormalize: z = z * std + mean = z * (1/inv_std) + mean
                std = 1.0 / inv_std
                z = z * std.view(1, self.z_dim, 1, 1) + mean.view(1, self.z_dim, 1, 1)
        else:
            if abs(scale[0]) > 1e-6 or abs(scale[1] - 1.0) > 1e-6:
                # Denormalize: z = z * std + mean = z * (1/scale[1]) + scale[0]
                std = 1.0 / scale[1]
                z = z * std + scale[0]
        
        x_recon = self.decoder(z)  # [B, 1, H, W]
        return x_recon
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def sample(self, images, scale, deterministic=False, return_features=False):
        """
        Sample latents from images.
        
        Args:
            images: [B, 1, H, W] grayscale images
            scale: [mean, std] for normalization
            deterministic: if True, return mu without sampling
            return_features: if True, return intermediate features
        
        Returns:
            latents: [B, z_dim, H', W'] sampled latents
            features: (optional) intermediate features
        """
        mu, log_var, feats = self.encode(images, scale)
        if deterministic:
            return mu if not return_features else (mu, feats)
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        z = mu + std * torch.randn_like(std)
        return (z, feats) if return_features else z
    
    def forward(self, x):
        """
        Forward pass for training.
        
        NOTE: During training, NO normalization is applied.
        Always uses scale=[0, 1] which means no normalization.
        """
        mu, log_var, _ = self.encode(x, scale=[0.0, 1.0])
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, scale=[0.0, 1.0])
        return x_recon, mu, log_var


def _cxr_vae_2d(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Factory function to create CxrAutoencoder2D_ instance.
    
    Args:
        pretrained_path: Path to pretrained weights (not implemented yet)
        z_dim: Latent dimension
        device: Device to load model on
        **kwargs: Additional config parameters
    
    Returns:
        CxrAutoencoder2D_ model
    """
    # Default config for CXR
    cfg = dict(
        dim=128,                  # was 96 → more capacity
        z_dim=z_dim or 16,        # was 8 → more latent channels
        dim_mult=[1, 2, 4, 8, 8],  # 5 stages: initial + 4 downsampling stages
        num_res_blocks=2,
        attn_scales=[0.25, 0.125],  # NEW: add attention at 64×64 and 32×32
        dropout=0.0,
        in_channels=1,  # Grayscale
        out_channels=1,  # Grayscale
        resolution=256,  # Input resolution (256×256)
    )
    cfg.update(**kwargs)
    
    # Init model (no meta device for smaller models, can instantiate directly)
    model = CxrAutoencoder2D_(**cfg)
    
    # Load checkpoint if provided
    if pretrained_path:
        logging.info(f'Loading CXR VAE from {pretrained_path}')
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Handle both checkpoint formats:
        # 1. Direct state dict
        # 2. Dictionary with 'model_state_dict' key (from training script)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, assign=True)
    else:
        logging.info("CxrAutoencoder2D_: No pretrained path provided, using random initialization")
    
    return model


class CxrAutoencoder2D:
    """
    Wrapper class for CxrAutoencoder2D_ with same API as WanVAE.
    
    Methods:
        - sample(): Encode images to latents [B, 1, H, W] -> [B, z_dim, H', W']
        - batch_decode(): Decode latents to images [B, z_dim, H', W'] -> [B, 1, H, W]
        - per_decode(): Decode single samples
        - per_sample(): Encode single samples
    """
    
    def __init__(self,
                 z_dim=16,          # was 8
                 vae_pth='',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device
        
        # Normalization parameters (computed post-training from dataset statistics)
        # Load from file if provided, otherwise use zeros/ones (no normalization)
        # During training: stats not available, use scale=[0, 1] (no normalization)
        # During inference: load stats file to enable normalization
        stats_loaded = False
        if vae_pth:
            vae_path = Path(vae_pth)
            # Try to load stats from multiple locations:
            # 1. Same directory as checkpoint
            # 2. Path from global config
            try:
                from configs.paths import paths
                config_stats_path = paths.CXR_LATENT_STATS_PATH
            except ImportError:
                # Fallback if config not available
                config_stats_path = Path("${REPO_ROOT}/outputs/cxr_latent_stats.pth")
            
            stats_paths = [
                vae_path.parent / "cxr_latent_stats.pth",
                config_stats_path,
            ]
            
            for stats_path in stats_paths:
                if stats_path.exists():
                    try:
                        stats = torch.load(stats_path, map_location=device)
                        loaded_mean = stats["mean"].to(dtype).to(device)
                        loaded_std = stats["std"].to(dtype).to(device)
                        
                        # Check if stats dimensions match expected z_dim
                        if loaded_mean.shape[0] != z_dim or loaded_std.shape[0] != z_dim:
                            logging.warning(
                                f"Stats file has z_dim={loaded_mean.shape[0]} but model expects z_dim={z_dim}. "
                                f"Skipping stats file and using no normalization."
                            )
                            continue
                        
                        self.mean = loaded_mean
                        self.std = loaded_std
                        stats_loaded = True
                        logging.info(f"Loaded latent statistics from {stats_path}")
                        break
                    except Exception as e:
                        logging.warning(f"Failed to load stats from {stats_path}: {e}")
                        continue
        
        if not stats_loaded:
            # No stats file - use zeros/ones (no normalization)
            # This is the default during training
            self.mean = torch.zeros(z_dim, dtype=dtype, device=device)
            self.std = torch.ones(z_dim, dtype=dtype, device=device)
            logging.info("No latent statistics found - using scale=[0, 1] (no normalization)")
        
        # Scale format: [mean, 1/std]
        # For encode: mu_normalized = (mu - mean) * (1/std)
        # For decode: z_denormalized = z * std + mean = z * (1/(1/std)) + mean
        self.scale = [self.mean, 1.0 / self.std]
        
        # Init model
        self.model = _cxr_vae_2d(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)
    
    def batch_decode(self, zs):
        """
        Decode batch of latents to images.
        
        Args:
            zs: [B, z_dim, H', W'] latents
        
        Returns:
            images: [B, 1, H, W] grayscale images
        """
        device_type = 'cuda' if (isinstance(self.device, torch.device) and self.device.type == 'cuda') or (isinstance(self.device, str) and 'cuda' in self.device.lower()) else 'cpu'
        with torch.amp.autocast(device_type, dtype=self.dtype):
            return self.model.decode(zs, self.scale).float().clamp_(-1, 1)
    
    def sample(self, images, deterministic=False, return_features=False):
        """
        Encode images to latents.
        
        Args:
            images: [B, 1, H, W] grayscale images
            deterministic: if True, return mean without sampling
            return_features: if True, return intermediate features
        
        Returns:
            latents: [B, z_dim, H', W'] encoded latents
            features: (optional) intermediate features
        """
        device_type = 'cuda' if (isinstance(self.device, torch.device) and self.device.type == 'cuda') or (isinstance(self.device, str) and 'cuda' in self.device.lower()) else 'cpu'
        with torch.amp.autocast(device_type, dtype=self.dtype):
            if return_features:
                out, feats = self.model.sample(images, self.scale, 
                                               deterministic=deterministic,
                                               return_features=return_features)
                return out.float(), feats.float()
            else:
                return self.model.sample(images, self.scale, 
                                        deterministic=deterministic).float()
    
    def per_decode(self, zs):
        """Decode single samples (one at a time)."""
        outputs = []
        device_type = 'cuda' if (isinstance(self.device, torch.device) and self.device.type == 'cuda') or (isinstance(self.device, str) and 'cuda' in self.device.lower()) else 'cpu'
        with torch.amp.autocast(device_type, dtype=self.dtype):
            for i in range(zs.shape[0]):
                outputs.append(self.model.decode(zs[i][None], self.scale).float().clamp_(-1, 1))
        return torch.cat(outputs, dim=0)
    
    def per_sample(self, images, deterministic=False, return_features=False):
        """Encode single samples (one at a time)."""
        device_type = 'cuda' if (isinstance(self.device, torch.device) and self.device.type == 'cuda') or (isinstance(self.device, str) and 'cuda' in self.device.lower()) else 'cpu'
        with torch.amp.autocast(device_type, dtype=self.dtype):
            if return_features:
                outputs = []
                features = []
                for i in range(images.shape[0]):
                    out, feats = self.model.sample(images[i][None], self.scale, 
                                                  deterministic=deterministic,
                                                  return_features=return_features)
                    outputs.append(out)
                    features.append(feats)
                outputs = torch.cat(outputs, dim=0)
                features = torch.cat(features, dim=0)
                return outputs.float(), features.float()
            else:
                outputs = []
                for i in range(images.shape[0]):
                    out = self.model.sample(images[i][None], self.scale, 
                                           deterministic=deterministic).float()
                    outputs.append(out)
                outputs = torch.cat(outputs, dim=0)
                return outputs.float()


if __name__ == '__main__':
    """
    Shape check: Verify 256×256 input -> 16×16 latents (256 tokens) -> 256×256 output.
    
    Note: In practice, CXR images should be resized/center-cropped to 256×256 
    before encoding in the dataloader/preprocessing step.
    """
    print("=" * 80)
    print("CXR VAE Shape Verification Test")
    print("=" * 80)
    
    # Test instantiation
    model = CxrAutoencoder2D(vae_pth="", z_dim=8, device="cpu")
    print("✓ Model initialized (no normalization - scale=[0, 1])")
    
    # Test with 256×256 input (as specified)
    cxr_image = torch.randn(2, 1, 256, 256)
    print(f"\nInput CXR shape: {cxr_image.shape}")
    assert cxr_image.shape[-2:] == (256, 256), "Input must be 256×256"
    
    # Encode
    latents = model.sample(cxr_image, deterministic=True)
    print(f"Encoded latents shape: {latents.shape}")
    
    # Verify latent grid is 16×16 (256 tokens per image)
    assert latents.shape[-2:] == (16, 16), f"Expected 16×16 latents, got {latents.shape[-2:]}"
    num_tokens = latents.shape[-2] * latents.shape[-1]
    print(f"Number of latent tokens per image: {num_tokens} (should be 256)")
    assert num_tokens == 256, f"Expected 256 tokens, got {num_tokens}"
    
    # Decode
    reconstructed = model.batch_decode(latents)
    print(f"Reconstructed CXR shape: {reconstructed.shape}")
    
    # Verify reconstruction is 256×256
    assert reconstructed.shape[-2:] == (256, 256), f"Expected 256×256 recon, got {reconstructed.shape[-2:]}"
    assert reconstructed.shape == cxr_image.shape, f"Shape mismatch: {reconstructed.shape} != {cxr_image.shape}"
    
    print("\n" + "=" * 80)
    print("✓ All shape checks passed!")
    print("=" * 80)
    print(f"  Input: {cxr_image.shape}")
    print(f"  Latents: {latents.shape} ({num_tokens} tokens per image)")
    print(f"  Output: {reconstructed.shape}")
    print("\n✓ Architecture verified: 256×256 → 16×16 → 256×256")







