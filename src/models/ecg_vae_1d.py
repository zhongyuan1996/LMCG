# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
1D VAE for ECG (Electrocardiogram) multi-lead signals.
Input shape: [B, L, T] where L = number of leads, T = time steps
Output latents: [B, z_dim, T'] where T' < T (temporally downsampled)
"""
import logging
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'EcgAutoencoder1D',
]


# ============================================================================
# Building blocks for 1D VAE (adapted from WanVAE 3D)
# ============================================================================

class RMS_norm_1d(nn.Module):
    """RMS normalization for 1D signals."""
    def __init__(self, dim, bias=False):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1)) if bias else 0.

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class Upsample1d(nn.Upsample):
    """Fix bfloat16 support for nearest neighbor interpolation."""
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class ResidualBlock1d(nn.Module):
    """1D residual block with Conv1d."""
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.residual = nn.Sequential(
            RMS_norm_1d(in_dim),
            nn.SiLU(),
            nn.Conv1d(in_dim, out_dim, 3, padding=1),
            RMS_norm_1d(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim, out_dim, 3, padding=1)
        )
        self.shortcut = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock1d(nn.Module):
    """1D self-attention block (temporal attention)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.norm = RMS_norm_1d(dim)
        self.to_qkv = nn.Conv1d(dim, dim * 3, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        
        # Zero out the last layer params
        nn.init.zeros_(self.proj.weight)
    
    def forward(self, x):
        identity = x
        b, c, t = x.shape
        
        x = self.norm(x)
        # Compute query, key, value
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention: [B, C, T] -> [B, C, T] -> [B, T, C]
        q = q.transpose(1, 2)  # [B, T, C]
        k = k.transpose(1, 2)  # [B, T, C]
        v = v.transpose(1, 2)  # [B, T, C]
        
        # Apply attention
        x = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape back: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        
        # Output
        x = self.proj(x)
        return x + identity


class Resample1d(nn.Module):
    """1D resampling (downsample/upsample) block."""
    def __init__(self, dim, mode):
        assert mode in ('downsample1d', 'upsample1d', 'none')
        super().__init__()
        self.dim = dim
        self.mode = mode
        
        if mode == 'upsample1d':
            self.resample = nn.Sequential(
                Upsample1d(scale_factor=2.0, mode='nearest'),
                nn.Conv1d(dim, dim // 2, 3, padding=1)
            )
        elif mode == 'downsample1d':
            self.resample = nn.Sequential(
                nn.ZeroPad1d((0, 1)),
                nn.Conv1d(dim, dim, 3, stride=2)
            )
        else:
            self.resample = nn.Identity()
    
    def forward(self, x):
        return self.resample(x)


class Encoder1d(nn.Module):
    """1D encoder for multi-lead ECG signals."""
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 in_channels=12):  # num_leads
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        
        # Dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        
        # Init block
        self.conv1 = nn.Conv1d(in_channels, dims[0], 3, padding=1)
        
        # Downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock1d(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock1d(out_dim))
                in_dim = out_dim
            
            # Downsample block
            if i != len(dim_mult) - 1:
                downsamples.append(Resample1d(out_dim, mode='downsample1d'))
                scale /= 2.0
        
        self.downsamples = nn.Sequential(*downsamples)
        
        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock1d(out_dim, out_dim, dropout),
            AttentionBlock1d(out_dim),
            ResidualBlock1d(out_dim, out_dim, dropout)
        )
        
        # Output blocks - z_dim here is already z_dim * 2, so output z_dim channels
        self.head = nn.Sequential(
            RMS_norm_1d(out_dim),
            nn.SiLU(),
            nn.Conv1d(out_dim, z_dim, 3, padding=1)  # Output z_dim (which is z_dim*2 from wrapper)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        x = self.head(x)
        return x


class Decoder1d(nn.Module):
    """1D decoder for multi-lead ECG signals."""
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 out_channels=12):  # num_leads
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        
        # Dimensions (reversed for decoder)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        
        # Init block
        self.conv1 = nn.Conv1d(z_dim, dims[0], 3, padding=1)
        
        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock1d(dims[0], dims[0], dropout),
            AttentionBlock1d(dims[0]),
            ResidualBlock1d(dims[0], dims[0], dropout)
        )
        
        # Upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Adjust in_dim for decoder (similar to 3D version)
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            
            # Residual (+attention) blocks
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock1d(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock1d(out_dim))
                in_dim = out_dim
            
            # Upsample block
            if i != len(dim_mult) - 1:
                upsamples.append(Resample1d(out_dim, mode='upsample1d'))
                scale *= 2.0
        
        self.upsamples = nn.Sequential(*upsamples)
        
        # Output blocks
        self.head = nn.Sequential(
            RMS_norm_1d(out_dim),
            nn.SiLU(),
            nn.Conv1d(out_dim, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.head(x)
        return x


class EcgAutoencoder1D_(nn.Module):
    """
    1D VAE for multi-lead ECG signals.
    
    Architecture:
    - Encoder: 1D convolutional encoder (temporal dimension only)
    - Decoder: 1D convolutional decoder (temporal dimension only)
    - Input: [B, L, T] where L = leads (e.g., 12), T = time steps
    - Latents: [B, z_dim, T'] where T' < T (temporally compressed)
    """
    
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 dropout=0.0,
                 num_leads=12,  # Standard 12-lead ECG
                 seq_len=5000):  # Typical ECG length
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.num_leads = num_leads
        self.seq_len = seq_len
        
        # Encoder: 1D conv layers (temporal only)
        self.encoder = Encoder1d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, dropout, num_leads)
        
        # Decoder: 1D conv layers (temporal only)
        self.decoder = Decoder1d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, dropout, num_leads)
        
        # Head convs for mu and log_var (encoder outputs z_dim*2, we split it)
        self.conv_mu = nn.Conv1d(z_dim * 2, z_dim, 1)
        self.conv_logvar = nn.Conv1d(z_dim * 2, z_dim, 1)
    
    def encode(self, x, scale):
        """
        Encode multi-lead ECG signals to latents.
        
        Args:
            x: [B, L, T] multi-lead ECG signals
            scale: [mean, std] for normalization
        
        Returns:
            mu: [B, z_dim, T'] mean
            log_var: [B, z_dim, T'] log variance
            features: intermediate features (optional)
        """
        out = self.encoder(x)  # [B, z_dim*2, T']
        mu = self.conv_mu(out)  # [B, z_dim, T']
        log_var = self.conv_logvar(out)  # [B, z_dim, T']
        
        # Apply scale normalization if provided
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1)) * scale[1].view(1, self.z_dim, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        
        return mu, log_var, out
    
    def decode(self, z, scale):
        """
        Decode latents back to multi-lead ECG signals.
        
        Args:
            z: [B, z_dim, T'] latents
            scale: [mean, std] for denormalization
        
        Returns:
            x_recon: [B, L, T] reconstructed ECG signals
        """
        # Apply inverse scale normalization
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1) + scale[0].view(1, self.z_dim, 1)
        else:
            z = z / scale[1] + scale[0]
        
        x_recon = self.decoder(z)  # [B, L, T]
        return x_recon
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def sample(self, ecg_signals, scale, deterministic=False, return_features=False):
        """
        Sample latents from ECG signals.
        
        Args:
            ecg_signals: [B, L, T] multi-lead ECG signals
            scale: [mean, std] for normalization
            deterministic: if True, return mu without sampling
            return_features: if True, return intermediate features
        
        Returns:
            latents: [B, z_dim, T'] sampled latents
            features: (optional) intermediate features
        """
        mu, log_var, feats = self.encode(ecg_signals, scale)
        if deterministic:
            return mu if not return_features else (mu, feats)
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        z = mu + std * torch.randn_like(std)
        return (z, feats) if return_features else z
    
    def forward(self, x):
        """Forward pass for training."""
        mu, log_var, _ = self.encode(x, scale=[0.0, 1.0])
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, scale=[0.0, 1.0])
        return x_recon, mu, log_var


def _ecg_vae_1d(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Factory function to create EcgAutoencoder1D_ instance.
    
    Args:
        pretrained_path: Path to pretrained weights (not implemented yet)
        z_dim: Latent dimension
        device: Device to load model on
        **kwargs: Additional config parameters
    
    Returns:
        EcgAutoencoder1D_ model
    """
    # Default config for ECG
    cfg = dict(
        dim=96,
        z_dim=z_dim or 8,  # ECG may need different z_dim
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],  # Can add attention at specific scales
        dropout=0.0,
        num_leads=12,  # Standard 12-lead ECG
        seq_len=5000,  # Typical ECG length
    )
    cfg.update(**kwargs)
    
    # Init model (no meta device for smaller models, can instantiate directly)
    model = EcgAutoencoder1D_(**cfg)
    
    # Load checkpoint if provided
    if pretrained_path:
        logging.info(f'Loading ECG VAE from {pretrained_path}')
        model.load_state_dict(
            torch.load(pretrained_path, map_location=device), assign=True)
    else:
        logging.info("EcgAutoencoder1D_: No pretrained path provided, using random initialization")
    
    return model


class EcgAutoencoder1D:
    """
    Wrapper class for EcgAutoencoder1D_ with same API as WanVAE.
    
    Methods:
        - sample(): Encode ECG to latents [B, L, T] -> [B, z_dim, T']
        - batch_decode(): Decode latents to ECG [B, z_dim, T'] -> [B, L, T]
        - per_decode(): Decode single samples
        - per_sample(): Encode single samples
    """
    
    def __init__(self,
                 z_dim=8,
                 vae_pth='',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device
        
        # Normalization parameters (not implemented yet - need to compute from ECG dataset)
        # TODO: Compute mean/std from ECG training data
        self.mean = torch.zeros(z_dim, dtype=dtype, device=device)
        self.std = torch.ones(z_dim, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]
        
        # Init model
        self.model = _ecg_vae_1d(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)
    
    def batch_decode(self, zs):
        """
        Decode batch of latents to ECG signals.
        
        Args:
            zs: [B, z_dim, T'] latents
        
        Returns:
            ecg_signals: [B, L, T] multi-lead ECG signals
        """
        with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', dtype=self.dtype):
            return self.model.decode(zs, self.scale).float()
    
    def sample(self, ecg_signals, deterministic=False, return_features=False):
        """
        Encode ECG signals to latents.
        
        Args:
            ecg_signals: [B, L, T] multi-lead ECG signals
            deterministic: if True, return mean without sampling
            return_features: if True, return intermediate features
        
        Returns:
            latents: [B, z_dim, T'] encoded latents
            features: (optional) intermediate features
        """
        with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', dtype=self.dtype):
            if return_features:
                out, feats = self.model.sample(ecg_signals, self.scale,
                                               deterministic=deterministic,
                                               return_features=return_features)
                return out.float(), feats.float()
            else:
                return self.model.sample(ecg_signals, self.scale,
                                        deterministic=deterministic).float()
    
    def per_decode(self, zs):
        """Decode single samples (one at a time)."""
        outputs = []
        with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', dtype=self.dtype):
            for i in range(zs.shape[0]):
                outputs.append(self.model.decode(zs[i][None], self.scale).float())
        return torch.cat(outputs, dim=0)
    
    def per_sample(self, ecg_signals, deterministic=False, return_features=False):
        """Encode single samples (one at a time)."""
        with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', dtype=self.dtype):
            if return_features:
                outputs = []
                features = []
                for i in range(ecg_signals.shape[0]):
                    out, feats = self.model.sample(ecg_signals[i][None], self.scale,
                                                  deterministic=deterministic,
                                                  return_features=return_features)
                    outputs.append(out)
                    features.append(feats)
                outputs = torch.cat(outputs, dim=0)
                features = torch.cat(features, dim=0)
                return outputs.float(), features.float()
            else:
                outputs = []
                for i in range(ecg_signals.shape[0]):
                    out = self.model.sample(ecg_signals[i][None], self.scale,
                                           deterministic=deterministic).float()
                    outputs.append(out)
                outputs = torch.cat(outputs, dim=0)
                return outputs.float()


if __name__ == '__main__':
    # Test instantiation
    model = EcgAutoencoder1D(vae_pth="", z_dim=8, device="cpu")
    
    # Test with dummy ECG signal [B, L, T]
    ecg_signal = torch.randn(2, 12, 5000)  # 2 samples, 12 leads, 5000 time steps
    print(f"Input ECG shape: {ecg_signal.shape}")
    
    # Encode
    latents = model.sample(ecg_signal)
    print(f"Encoded latents shape: {latents.shape}")
    
    # Decode
    reconstructed = model.batch_decode(latents)
    print(f"Reconstructed ECG shape: {reconstructed.shape}")







