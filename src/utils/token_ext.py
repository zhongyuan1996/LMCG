import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

"""
Helpers to extend the base tokenizer with EHR-specific special tokens and
autoregressive DX code tokens without disturbing existing IDs.
"""

# Structural and modality markers we need for EHR multimodal packing.
DEFAULT_EHR_SPECIAL_TOKENS: List[str] = [
    "[PAT_BOS]",
    "[PAT_EOS]",
    "[VIS_BOS]",
    "[VIS_EOS]",
    "[CTX_CONTINUED]",
    "[GEN_TARGET]",
    "[MOD_DEMO_BOS]",
    "[MOD_DEMO_EOS]",
    "[MOD_SUM_BOS]",
    "[MOD_SUM_EOS]",
    "[MOD_NOTE_BOS]",
    "[MOD_NOTE_EOS]",
    "[MOD_DX_BOS]",
    "[MOD_DX_EOS]",
    "[MOD_CXR_BOS]",
    "[MOD_CXR_EOS]",
    "[MOD_ECG_BOS]",
    "[MOD_ECG_EOS]",
    "[DX_SYS_ICD9]",
    "[DX_SYS_ICD10]",
    "[CXR_LAT]",
    "[ECG_LAT]",
]


def sanitize_dx_code(code: str) -> str:
    """Keep alnum plus dot, strip spaces; fallback to raw code if empty."""
    cleaned = re.sub(r"[^A-Za-z0-9.]", "", code.strip())
    return cleaned or code.strip()


def make_dx_token(code: str, system: str) -> str:
    """
    Token string for a single DX code with system prefix.
    system: "icd9" or "icd10"
    """
    prefix = "DX9" if system.lower() == "icd9" else "DX10"
    return f"[{prefix}_{sanitize_dx_code(code)}]"


def extend_tokenizer_with_ehr_tokens(
    tokenizer,
    dx9_codes: Iterable[str],
    dx10_codes: Iterable[str],
    extra_special_tokens: Iterable[str] = (),
) -> Dict[str, int]:
    """
    Extend the tokenizer vocabulary with EHR markers and DX code tokens.

    Returns a mapping of useful token ids for downstream packing.
    """
    specials = list(DEFAULT_EHR_SPECIAL_TOKENS) + list(extra_special_tokens or [])
    tokenizer.add_special_tokens({"additional_special_tokens": specials})

    # Build and add DX tokens
    dx_tokens = [make_dx_token(code, "icd9") for code in dx9_codes]
    dx_tokens += [make_dx_token(code, "icd10") for code in dx10_codes]
    tokenizer.add_tokens(dx_tokens)

    # Collect ids
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in specials}
    for tok in dx_tokens:
        token_ids[tok] = tokenizer.convert_tokens_to_ids(tok)

    return token_ids


def build_dx_token_map(dx9_codes: Iterable[str], dx10_codes: Iterable[str]) -> Dict[str, str]:
    """
    For callers that already added tokens, build a map:
        (system, code) -> token string
    """
    m: Dict[str, str] = {}
    for code in dx9_codes:
        sc = sanitize_dx_code(code)
        m[f"icd9:{sc}"] = make_dx_token(code, "icd9")
    for code in dx10_codes:
        sc = sanitize_dx_code(code)
        m[f"icd10:{sc}"] = make_dx_token(code, "icd10")
    return m


def load_dx_vocab_json(vocab_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load dx vocab JSON produced by build_dx_vocab.py.
    Returns (icd9_codes, icd10_codes).
    """
    with Path(vocab_path).open() as f:
        js = json.load(f)
    return js.get("icd9_codes", []), js.get("icd10_codes", [])

