#!/usr/bin/env python3
"""
ICD Token Embedding Initialization

Initialize ICD special token embeddings from their descriptions instead of random.
This gives the model semantic grounding for ICD codes.

Usage:
    from training.icd_embedding_init import initialize_icd_embeddings
    initialize_icd_embeddings(model, tokenizer, device='cuda')
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


def get_icd_embeddings_cache_path(icd_descriptions_path: str) -> Path:
    """Get the cache file path for pre-computed ICD embeddings."""
    desc_path = Path(icd_descriptions_path)
    return desc_path.parent / "icd_embeddings_cache.pt"


def save_icd_embeddings_cache(
    embed_layer,
    tokenizer,
    icd_descriptions_path: str,
    verbose: bool = True,
):
    """Save initialized ICD embeddings to cache file for fast reloading."""
    cache_path = get_icd_embeddings_cache_path(icd_descriptions_path)
    
    # Load mappings to get token names
    with open(icd_descriptions_path, 'r') as f:
        mappings = json.load(f)
    
    # Extract ICD embeddings
    icd_embeddings = {}
    for token_name in mappings.keys():
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        if token_id != tokenizer.unk_token_id:
            icd_embeddings[token_name] = embed_layer.weight.data[token_id].cpu().clone()
    
    # Save to cache
    torch.save({
        'embeddings': icd_embeddings,
        'vocab_size': len(tokenizer),
    }, cache_path)
    
    if verbose:
        print(f"Saved {len(icd_embeddings)} ICD embeddings to cache: {cache_path}")
    
    return cache_path


def load_icd_embeddings_cache(
    model,
    tokenizer,
    icd_descriptions_path: str,
    verbose: bool = True,
) -> bool:
    """
    Load pre-computed ICD embeddings from cache.
    
    Returns True if cache was loaded successfully, False otherwise.
    """
    cache_path = get_icd_embeddings_cache_path(icd_descriptions_path)
    
    if not cache_path.exists():
        if verbose:
            print(f"No ICD embeddings cache found at {cache_path}")
        return False
    
    try:
        cache = torch.load(cache_path, map_location='cpu')
        icd_embeddings = cache['embeddings']
        cached_vocab_size = cache.get('vocab_size', 0)
        
        # Verify vocab size matches
        if cached_vocab_size != len(tokenizer):
            if verbose:
                print(f"Cache vocab size mismatch ({cached_vocab_size} vs {len(tokenizer)}), regenerating...")
            return False
        
        # Get embedding layer
        if hasattr(model, 'backbone'):
            embed_layer = model.backbone.get_input_embeddings()
        else:
            embed_layer = model.get_input_embeddings()
        
        # Load embeddings directly (no normalization needed since we use input embeddings)
        # The cached embeddings are already in the correct scale
        loaded_count = 0
        for token_name, embedding in icd_embeddings.items():
            token_id = tokenizer.convert_tokens_to_ids(token_name)
            if token_id != tokenizer.unk_token_id:
                embed_layer.weight.data[token_id] = embedding.to(embed_layer.weight.device)
                loaded_count += 1
        
        if verbose:
            print(f"Loaded {loaded_count} ICD embeddings from cache: {cache_path}")
        
        return True
    
    except Exception as e:
        if verbose:
            print(f"Failed to load ICD embeddings cache: {e}")
        return False


def initialize_icd_embeddings(
    model,
    tokenizer,
    icd_descriptions_path: str = "${REPO_ROOT}/outputs/icd_category_descriptions.json",
    device: str = 'cuda',
    verbose: bool = True,
    use_cache: bool = True,
):
    """
    Initialize ICD token embeddings from their descriptions.
    
    For each ICD token (e.g., <ICD10_E66>), we:
    1. Get its description (e.g., "Overweight and obesity")
    2. Encode the description through the model
    3. Average pool the hidden states
    4. Set this as the token's embedding
    
    For fallback cases (no direct description), we average embeddings of all children.
    
    Args:
        model: The Stage-1 model with backbone
        tokenizer: Tokenizer with ICD tokens added
        icd_descriptions_path: Path to the JSON mapping file
        device: Device to run on
        verbose: Print progress
        use_cache: If True, try to load from cache first; save to cache after computing
    """
    # Try to load from cache first
    if use_cache:
        if load_icd_embeddings_cache(model, tokenizer, icd_descriptions_path, verbose):
            return {'loaded_from_cache': True}
    
    if verbose:
        print("\n" + "="*60)
        print("Initializing ICD Token Embeddings from Descriptions")
        print("="*60)
    
    # Load description mappings
    with open(icd_descriptions_path, 'r') as f:
        mappings = json.load(f)
    
    if verbose:
        print(f"Loaded {len(mappings)} ICD token mappings")
    
    # Get the embedding layer
    # Handle both regular and PEFT-wrapped models
    if hasattr(model, 'backbone'):
        embed_layer = model.backbone.get_input_embeddings()
    else:
        embed_layer = model.get_input_embeddings()
    
    embed_dim = embed_layer.weight.shape[1]
    
    # Statistics
    stats = {
        'direct': 0,
        'unspecified_variant': 0,
        'common_keywords': 0,
        'fallback_embed_average': 0,
        'skipped': 0,
    }
    
    # Process each ICD token
    iterator = tqdm(mappings.items(), desc="Initializing ICD embeddings") if verbose else mappings.items()
    
    for token_name, info in iterator:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        
        # Skip if token not in vocabulary
        if token_id == tokenizer.unk_token_id:
            stats['skipped'] += 1
            continue
        
        method = info['method']
        
        if method in ['direct', 'unspecified_variant', 'common_keywords']:
            # Use description text
            description = info['description']
            if not description:
                stats['skipped'] += 1
                continue
            
            # Encode description and get embedding
            new_embedding = _encode_description(
                model, tokenizer, description, device, embed_dim
            )
            
            if new_embedding is not None:
                embed_layer.weight.data[token_id] = new_embedding
                stats[method] = stats.get(method, 0) + 1
            else:
                stats['skipped'] += 1
        
        elif method == 'fallback_embed_average':
            # Average embeddings of all children
            children = info.get('children', [])
            if not children:
                stats['skipped'] += 1
                continue
            
            child_embeddings = []
            for child in children:
                child_desc = child.get('desc', '')
                if child_desc:
                    child_emb = _encode_description(
                        model, tokenizer, child_desc, device, embed_dim
                    )
                    if child_emb is not None:
                        child_embeddings.append(child_emb)
            
            if child_embeddings:
                avg_embedding = torch.stack(child_embeddings).mean(dim=0)
                embed_layer.weight.data[token_id] = avg_embedding
                stats['fallback_embed_average'] += 1
            else:
                stats['skipped'] += 1
        
        elif method == 'not_found':
            # Use the generic description as fallback
            description = info.get('description', '')
            if description:
                new_embedding = _encode_description(
                    model, tokenizer, description, device, embed_dim
                )
                if new_embedding is not None:
                    embed_layer.weight.data[token_id] = new_embedding
            stats['skipped'] += 1
    
    if verbose:
        print("\n" + "-"*40)
        print("ICD Embedding Initialization Complete:")
        print(f"  Direct descriptions:     {stats['direct']}")
        print(f"  Unspecified variants:    {stats['unspecified_variant']}")
        print(f"  Common keywords:         {stats['common_keywords']}")
        print(f"  Averaged children:       {stats['fallback_embed_average']}")
        print(f"  Skipped:                 {stats['skipped']}")
        print(f"  Total initialized:       {sum(v for k, v in stats.items() if k != 'skipped')}")
        print("-"*40 + "\n")
    
    # Save to cache for future runs
    if use_cache:
        save_icd_embeddings_cache(embed_layer, tokenizer, icd_descriptions_path, verbose)
    
    return stats


def _encode_description(
    model,
    tokenizer,
    description: str,
    device: str,
    embed_dim: int,
) -> Optional[torch.Tensor]:
    """
    Encode a description text and return averaged INPUT embedding.
    
    Uses the model's input embedding layer directly (NOT hidden states).
    This ensures the output is in the same scale as the original embeddings.
    """
    try:
        # Tokenize
        inputs = tokenizer(
            description,
            return_tensors='pt',
            truncation=True,
            max_length=64,  # Descriptions are short
            padding=False,
        ).to(device)
        
        # Get embedding layer (handle PEFT wrapping)
        if hasattr(model, 'backbone'):
            embed_layer = model.backbone.get_input_embeddings()
        else:
            embed_layer = model.get_input_embeddings()
        
        # Get input embeddings directly (NOT hidden states)
        # This is the first layer, so output is in the same scale as original embeddings
        with torch.no_grad():
            # input_ids: [1, seq_len]
            # token_embeds: [1, seq_len, hidden_dim]
            token_embeds = embed_layer(inputs['input_ids'])
            
            # Average pool over sequence length
            pooled = token_embeds.mean(dim=1)  # [1, hidden_dim]
            
            return pooled.squeeze(0)
    
    except Exception as e:
        print(f"Warning: Failed to encode '{description[:50]}...': {e}")
        return None


def test_icd_embeddings(model, tokenizer, device='cuda'):
    """
    Test that ICD embeddings are semantically meaningful.
    
    Compare similarity between related ICD codes and unrelated ones.
    """
    print("\n" + "="*60)
    print("Testing ICD Embedding Quality")
    print("="*60)
    
    # Get embedding layer
    if hasattr(model, 'backbone'):
        embed_layer = model.backbone.get_input_embeddings()
    else:
        embed_layer = model.get_input_embeddings()
    
    # Test pairs: (code1, code2, expected_relation)
    test_pairs = [
        # Related pairs (should be similar)
        ("<ICD10_E66>", "<ICD10_E65>", "related"),  # Obesity, Localized adiposity
        ("<ICD10_E11>", "<ICD10_E10>", "related"),  # Type 2 diabetes, Type 1 diabetes
        ("<ICD10_I10>", "<ICD10_I11>", "related"),  # Hypertension, Hypertensive heart
        # Unrelated pairs (should be dissimilar)
        ("<ICD10_E66>", "<ICD10_S72>", "unrelated"),  # Obesity vs Fracture
        ("<ICD10_E11>", "<ICD10_J18>", "unrelated"),  # Diabetes vs Pneumonia
    ]
    
    print("\nCosine similarities between ICD codes:")
    print("-"*60)
    
    for token1, token2, relation in test_pairs:
        id1 = tokenizer.convert_tokens_to_ids(token1)
        id2 = tokenizer.convert_tokens_to_ids(token2)
        
        if id1 == tokenizer.unk_token_id or id2 == tokenizer.unk_token_id:
            print(f"  {token1} <-> {token2}: SKIPPED (unknown token)")
            continue
        
        emb1 = embed_layer.weight[id1]
        emb2 = embed_layer.weight[id2]
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()
        
        expected = "HIGH" if relation == "related" else "LOW"
        actual = "HIGH" if cos_sim > 0.5 else "LOW"
        status = "✓" if expected == actual else "?"
        
        print(f"  {token1} <-> {token2}: {cos_sim:.4f} (expect {expected}) {status}")


# NOTE: a standalone __main__ test block originally lived here; it imported
# helper modules from an older project layout that are not part of this
# release. The functions exposed above (initialize_icd_embeddings,
# test_icd_embeddings) are the public API and are exercised by
# pipeline/stage1/train.py.
