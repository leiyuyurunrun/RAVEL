from __future__ import annotations

import re
from typing import Any

CANONICAL_ATTACK_LABELS = {
    "access_control",
    "insufficient_validation",
    "price_manipulation",
    "market_manipulation",
    "reentrancy",
    "token_semantic_exploitation",
    "protocol_accounting_exploitation",
    "flashloans",
}

_ATTACK_LABEL_ALIASES = {
    "accesscontrol": "access_control",
    "access_control": "access_control",
    "insufficientvalidation": "insufficient_validation",
    "insufficient_validation": "insufficient_validation",
    "insufficient_validationy": "insufficient_validation",
    "insufficientvalidationy": "insufficient_validation",
    "price_manipulation": "price_manipulation",
    "pricemanipulation": "price_manipulation",
    "market_manipulation": "market_manipulation",
    "marketmanipulation": "market_manipulation",
    "reentrancy": "reentrancy",
    "reentrant": "reentrancy",
    "protocol_accounting_exploitation": "protocol_accounting_exploitation",
    "protocolaccountingexploitation": "protocol_accounting_exploitation",
    "token_semantic_exploitation": "token_semantic_exploitation",
    "tokensemanticexploitation": "token_semantic_exploitation",
    "token_smantic_exploitation": "token_semantic_exploitation",
    "tokensmanticexploitation": "token_semantic_exploitation",
    "flashloans": "flashloans",
    "flashloan": "flashloans",
    "flashloand": "flashloans",
    "flashloan_assisted_exploitation": "flashloans",
    "flashloanassistedexploitation": "flashloans",
}


def normalize_attack_label(label: Any, default: str = "attack") -> str:
    """Return the canonical internal attack label key.

    Normalizes casing and punctuation, then resolves known aliases/typos.
    Unknown labels are returned in normalized snake_case form so custom labels
    remain readable and stable.
    """
    raw = str(label or "").strip().lower()
    if not raw:
        return default
    compact = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    collapsed = compact.replace("_", "")
    return _ATTACK_LABEL_ALIASES.get(compact) or _ATTACK_LABEL_ALIASES.get(collapsed) or compact or default


def display_attack_label(label: Any, default: str = "attack") -> str:
    """Human-readable label for reports/prompts."""
    canonical = normalize_attack_label(label, default=default)
    return canonical.replace("_", " ")


def label_slug(label: Any, default: str = "attack") -> str:
    """Filesystem-safe canonical label slug."""
    return normalize_attack_label(label, default=default)


def label_text(label: Any, default: str = "attack") -> str:
    """Space-separated canonical label for substring matching."""
    return normalize_attack_label(label, default=default).replace("_", " ")
