#!/usr/bin/env python3
"""
Build 3-digit ICD code → description mapping for embedding initialization.

Strategy:
1. ICD-10: Direct 3-char category description (nearly complete coverage)
2. ICD-9:
   - First: Try *9 (unspecified) variant's description
   - Second: Extract common keywords from all children
   - Fallback: Mark for embedding averaging (done at init time)

Output: JSON file with mappings and a report for manual verification.
"""

import gzip
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_icd_descriptions(filepath: str) -> Dict[Tuple[str, int], str]:
    """Load all ICD code descriptions from MIMIC d_icd_diagnoses.csv.gz"""
    descriptions = {}
    
    with gzip.open(filepath, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['icd_code'].strip().upper()
            version = int(row['icd_version'])
            title = row['long_title'].strip()
            descriptions[(code, version)] = title
    
    print(f"Loaded {len(descriptions)} ICD descriptions")
    return descriptions


def truncate_icd_code(code: str, version: int) -> str:
    """Truncate ICD code to category level (matching our tokenizer)."""
    code = code.strip().upper()
    
    if version == 9:
        # ICD-9: E/V codes keep 4 chars, others keep 3
        if code.startswith('E') or code.startswith('V'):
            return code[:4]
        else:
            return code[:3]
    else:
        # ICD-10: First 3 characters
        return code[:3]


def extract_common_keywords(descriptions: List[str]) -> Optional[str]:
    """
    Extract common keywords from multiple descriptions.
    
    Example:
        ["Malignant essential hypertension", 
         "Benign essential hypertension",
         "Unspecified essential hypertension"]
        → "essential hypertension"
    """
    if not descriptions:
        return None
    
    if len(descriptions) == 1:
        return descriptions[0]
    
    # Tokenize and lowercase
    word_sets = []
    for desc in descriptions:
        # Remove common suffixes and noise
        desc_clean = desc.lower()
        desc_clean = re.sub(r'\bunspecified\b', '', desc_clean)
        desc_clean = re.sub(r'\bother\b', '', desc_clean)
        desc_clean = re.sub(r'\bnot elsewhere classified\b', '', desc_clean)
        desc_clean = re.sub(r'\bnec\b', '', desc_clean)
        words = set(re.findall(r'\b[a-z]+\b', desc_clean))
        words = {w for w in words if len(w) > 2}  # Filter short words
        word_sets.append(words)
    
    # Find intersection
    if not word_sets:
        return None
    
    common = word_sets[0]
    for ws in word_sets[1:]:
        common = common & ws
    
    if not common:
        return None
    
    # Reconstruct phrase from first description, keeping common words
    first_desc = descriptions[0].lower()
    words_in_order = re.findall(r'\b[a-z]+\b', first_desc)
    common_in_order = [w for w in words_in_order if w in common]
    
    if common_in_order:
        return ' '.join(common_in_order)
    
    return None


def build_category_descriptions(
    all_descriptions: Dict[Tuple[str, int], str],
    our_codes_9: List[str],
    our_codes_10: List[str],
) -> Tuple[Dict[str, dict], dict]:
    """
    Build 3-digit category → description mapping.
    
    Returns:
        mappings: Dict[token_name, {code, version, description, method}]
        stats: Summary statistics
    """
    mappings = {}
    stats = {
        'icd10_direct': 0,
        'icd9_direct': 0,
        'icd9_unspecified': 0,
        'icd9_common': 0,
        'icd9_fallback': 0,
        'not_found': 0,
    }
    
    # Group all codes by their 3-digit category
    category_children = defaultdict(list)  # (category, version) -> [(full_code, description)]
    
    for (code, version), desc in all_descriptions.items():
        cat = truncate_icd_code(code, version)
        category_children[(cat, version)].append((code, desc))
    
    # Process ICD-10 codes
    for cat in our_codes_10:
        token_name = f"<ICD10_{cat}>"
        
        # Try direct 3-char description
        if (cat, 10) in all_descriptions:
            mappings[token_name] = {
                'code': cat,
                'version': 10,
                'description': all_descriptions[(cat, 10)],
                'method': 'direct',
            }
            stats['icd10_direct'] += 1
        else:
            # Fallback: use children
            children = category_children.get((cat, 10), [])
            if children:
                mappings[token_name] = {
                    'code': cat,
                    'version': 10,
                    'description': None,
                    'method': 'fallback_embed_average',
                    'children': [{'code': c, 'desc': d} for c, d in children],
                }
                stats['icd9_fallback'] += 1  # Reuse counter
            else:
                mappings[token_name] = {
                    'code': cat,
                    'version': 10,
                    'description': f"ICD-10 category {cat}",
                    'method': 'not_found',
                }
                stats['not_found'] += 1
    
    # Process ICD-9 codes
    for cat in our_codes_9:
        token_name = f"<ICD9_{cat}>"
        children = category_children.get((cat, 9), [])
        
        # Method 1: Direct 3-digit description
        if (cat, 9) in all_descriptions:
            mappings[token_name] = {
                'code': cat,
                'version': 9,
                'description': all_descriptions[(cat, 9)],
                'method': 'direct',
            }
            stats['icd9_direct'] += 1
            continue
        
        if not children:
            mappings[token_name] = {
                'code': cat,
                'version': 9,
                'description': f"ICD-9 category {cat}",
                'method': 'not_found',
            }
            stats['not_found'] += 1
            continue
        
        # Method 2: Try *9 (unspecified) variant
        unspec_desc = None
        for code, desc in children:
            # Check for unspecified variants: ends in 9 or 0, or contains "unspecified"
            if code.endswith('9') or code.endswith('0') or 'unspecified' in desc.lower():
                unspec_desc = desc
                # Clean up the description
                unspec_desc = re.sub(r'\bunspecified\b', '', unspec_desc, flags=re.IGNORECASE).strip()
                unspec_desc = re.sub(r'^,\s*', '', unspec_desc).strip()
                unspec_desc = re.sub(r',\s*$', '', unspec_desc).strip()
                if unspec_desc:
                    break
        
        if unspec_desc and len(unspec_desc) > 3:
            mappings[token_name] = {
                'code': cat,
                'version': 9,
                'description': unspec_desc,
                'method': 'unspecified_variant',
            }
            stats['icd9_unspecified'] += 1
            continue
        
        # Method 3: Extract common keywords
        child_descs = [d for _, d in children]
        common = extract_common_keywords(child_descs)
        
        # Validate common keywords result
        is_valid_common = False
        if common and len(common) > 5:  # Must be reasonably long
            # Check it's not too generic or grammatically incomplete
            generic_words = {'status', 'other', 'unspecified', 'the', 'for', 'of'}
            common_words = set(common.lower().split())
            non_generic = common_words - generic_words
            
            # Must have meaningful medical content (at least 2 non-generic words)
            # And shouldn't end with articles/prepositions
            ends_badly = common.split()[-1].lower() in {'the', 'for', 'of', 'a', 'an', 'to', 'in'}
            
            if len(non_generic) >= 2 and not ends_badly:
                is_valid_common = True
        
        if is_valid_common:
            mappings[token_name] = {
                'code': cat,
                'version': 9,
                'description': common,
                'method': 'common_keywords',
            }
            stats['icd9_common'] += 1
            continue
        
        # Method 4: Fallback - mark for embedding averaging
        mappings[token_name] = {
            'code': cat,
            'version': 9,
            'description': None,
            'method': 'fallback_embed_average',
            'children': [{'code': c, 'desc': d} for c, d in children],
        }
        stats['icd9_fallback'] += 1
    
    return mappings, stats
