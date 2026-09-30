from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import (
    EvidencePlan,
    EvolvingRule,
    JudgeStep,
    RuleCondition,
)
from evotx.runtime.view_catalog import PACKET_VIEW_CATALOG, packet_view_render_policy


# ---------------------------------------------------------------------------
# Packet view groups: semantic group -> packet view names
# ---------------------------------------------------------------------------

_PACKET_VIEW_GROUPS: Dict[str, List[str]] = {
    "fund": [
        "participant_net_delta_view",
        "value_release_view",
        "contribution_vs_payout_view",
        "transfer_event_view",
        "external_fundflow_view",
        "profit_loss_view",
    ],
    "trace": [
        "critical_call_view",
        "trace_view",
        "unknown_selector_view",
    ],
    "reentrancy": [
        "critical_call_view",
        "reentrancy_state_order_summary_view",
        "value_release_view",
        "trace_outline_view",
        "event_view",
        "transfer_event_view",
        "semantic_state_delta_view",
        "reentrancy_state_order_view",
    ],
    "state": [
        "semantic_state_delta_view",
        "protocol_accounting_outcome_view",
        "market_mechanism_profile_view",
        "price_relevant_state_view",
        "state_change_view",
        "amm_reserve_transition_view",
    ],
    "event": ["event_view", "transfer_event_view"],
    "address": ["address_labels", "tx_card"],
    "adequacy": ["evidence_adequacy_view"],
    "atomic_capital": [
        "flash_or_atomic_capital_view",
        "external_fundflow_view",
        "transfer_event_view",
        "critical_call_view",
    ],
}

_GROUP_KEYWORDS: Dict[str, List[str]] = {
    "fund": [
        "fund", "flow", "token", "asset", "profit", "loss", "balance",
        "transfer", "withdraw", "deposit", "collection", "mint", "reward",
        "value", "revenue", "payment", "drain", "payout",
        "contribution", "output", "input", "release", "recipient",
        "extraction", "extract", "outflow",
    ],
    "trace": [
        "call", "trace", "execution", "function", "invoke", "reentrancy",
        "repeated", "pattern", "contract", "delegatecall", "caller",
        "callee", "subtree", "callback", "internal", "nested", "entry",
    ],
    "reentrancy": [
        "reentrancy", "reentrant", "re-enter", "reentry", "callback",
        "nested", "recursive", "stale", "stale-state", "state order",
        "state ordering", "read before write", "write after", "sload",
        "sstore", "guard", "lock", "unlock", "nonreentrant",
        "non-reentrant", "repeated release", "repeated payout",
    ],
    "state": [
        "state", "storage", "slot", "accounting", "reserve", "diff",
        "invariant", "storage_write", "stale", "oracle", "share",
        "collateral", "debt", "liquidity", "tick", "exchange-rate",
        "exchange rate", "price", "market", "pool balance", "slippage",
        "skim", "sync", "valuation", "settlement",
    ],
    "event": ["event", "log", "emit", "approval", "log_entry"],
    "address": [
        "address", "role", "attacker", "victim", "admin", "router",
        "protocol", "sender", "receiver", "multisig", "governance",
        "oracle", "keeper",
    ],
    "adequacy": [
        "missing", "unknown", "decode", "coverage", "truncated", "selector",
    ],
    "atomic_capital": [
        "flash", "flashloan", "flash loan", "flash swap", "atomic",
        "borrow", "repay", "unwind", "restore", "callback",
    ],
}

_DEFAULT_PACKET_REFS = [
    "operation_summary_view",
    "evidence_adequacy_view",
    "classification_digest_view",
    "trace_outline_view",
    "critical_call_view",
    "event_view",
    "tx_card",
]

_COMMON_PREFIX_VIEWS = [
    "operation_summary_view",
    "evidence_adequacy_view",
]

ORDINARY_MAX_DEFAULT_VIEWS = 4
SPECIAL_MAX_DEFAULT_VIEWS = 5
MAX_DEFAULT_LARGE_VIEWS = 1
ORDINARY_MAX_FOLLOWUP_VIEWS = 5
SOURCE_MAX_FOLLOWUP_VIEWS = 7
STATE_ORDER_MAX_FOLLOWUP_VIEWS = 6
LABEL_AWARE_PLAN_BUDGETS: Dict[str, Dict[str, int]] = {
    "reentrancy": {
        "max_default_views": 5,
        "max_followup_views": 6,
        "max_large_views": 2,
    },
    "access_control": {
        "max_default_views": 6,
        "max_followup_views": 6,
        "max_large_views": 3,
    },
    "insufficient_validation": {
        "max_default_views": 6,
        "max_followup_views": 8,
        "max_large_views": 3,
    },
    "market_manipulation": {
        "max_default_views": 5,
        "max_followup_views": 8,
        "max_large_views": 2,
    },
    "flashloans": {
        "max_default_views": 6,
        "max_followup_views": 8,
        "max_large_views": 2,
    },
    "protocol_accounting_exploitation": {
        "max_default_views": 6,
        "max_followup_views": 8,
        "max_large_views": 2,
    },
}
_FIRST_PASS_MAX_DEFAULT_VIEWS = ORDINARY_MAX_DEFAULT_VIEWS
_FIRST_PASS_MAX_LARGE_VIEWS = MAX_DEFAULT_LARGE_VIEWS
_FIRST_PASS_ALWAYS_KEEP = tuple(_COMMON_PREFIX_VIEWS)

_SOURCE_DEPENDENT_KEYWORDS = [
    "access control",
    "authorization",
    "authorisation",
    "permission",
    "privilege",
    "privileged",
    "owner",
    "admin",
    "role",
    "onlyowner",
    "only owner",
    "modifier",
    "initializer",
    "initialize",
    "uninitialized",
    "proxy",
    "upgrade",
    "implementation",
    "msg.sender",
    "caller validation",
    "validation logic",
    "visibility",
    "public function",
    "internal function",
    "source code",
    "selector",
    "decode",
    "function semantics",
    "storage-slot semantics",
    "accounting formula",
    "price formula",
    "formula",
]

_SOURCE_REQUIRED_KEYWORDS = [
    "onlyowner",
    "only owner",
    "modifier",
    "msg.sender",
    "caller validation",
    "role check",
    "owner check",
    "admin check",
    "whitelist check",
    "allowlist check",
    "permission check",
    "authorization check",
    "validation logic",
    "callback sender validation",
    "callback initiator validation",
    "selector meaning",
    "unknown selector semantics",
    "storage-slot semantics",
    "storage slot semantics",
    "source-level formula",
    "accounting formula",
    "price formula",
    "function visibility",
    "public function",
    "internal function",
]

_SOURCE_HELPFUL_KEYWORDS = [
    "validation",
    "input",
    "state consumed",
    "consumes user",
    "consumed data",
    "accounting",
    "state order",
    "invariant",
    "authorization",
    "role",
    "admin",
    "price",
    "oracle",
    "exchange rate",
    "exchange-rate",
    "formula",
]

_SOURCE_FOLLOWUP_VIEWS = [
    "trace_outline_view",
    "critical_call_view",
    "unknown_selector_view",
    "trace_view",
    "semantic_state_delta_view",
    "state_change_view",
    "value_release_view",
    "beneficiary_controller_view",
    "address_labels",
    "event_view",
    "transfer_event_view",
    "participant_net_delta_view",
    "contribution_vs_payout_view",
]

_STATE_ORDER_REENTRANCY_KEYWORDS = [
    "reentrancy",
    "reentrant",
    "re-enter",
    "reentry",
    "nested",
    "callback",
]

_STATE_ORDER_KEYWORDS = [
    "stale",
    "state order",
    "state ordering",
    "execution-ordering",
    "execution ordering",
    "read before write",
    "write after",
    "applied only after",
    "update only after",
    "state update after",
    "pre-update state",
    "pre update state",
    "before the outer",
    "outer function updated",
    "incomplete state",
    "incomplete accounting",
    "accounting order",
    "sload",
    "sstore",
]

_MARKET_STATE_KEYWORDS = [
    "market",
    "amm",
    "reserve",
    "reserves",
    "oracle",
    "price",
    "pricing",
    "valuation",
    "settlement",
    "slippage",
    "swap",
    "arbitrage",
    "dex",
    "sandwich",
    "mev",
    "pool balance",
    "skim",
    "sync",
    "kLast".lower(),
    "sqrtprice",
    "tick",
    "liquidity",
    "virtual price",
    "exchange rate",
]

_MARKET_DISTORTION_KEYWORDS = [
    "distort",
    "perturb",
    "manipulat",
    "misuse",
    "donation",
    "imbalance",
    "stale",
    "consume",
    "relies upon",
    "relies on",
    "downstream",
    "value-sensitive",
    "price-sensitive",
    "reserve-sensitive",
]

_VALUE_EXTRACTION_KEYWORDS = [
    "value extraction",
    "extract",
    "profit",
    "loss",
    "beneficiary",
    "gain",
    "payout",
    "release",
    "recipient",
    "withdraw",
    "redeem",
    "borrow",
    "claim",
    "harvest",
    "reward",
    "protocol loss",
    "user loss",
    "disproportionate",
]

_ACCOUNTING_KEYWORDS = [
    "accounting",
    "bookkeeping",
    "share",
    "shares",
    "reward",
    "yield",
    "staking",
    "collateral",
    "debt",
    "vault",
    "strategy",
    "claim eligibility",
    "contribution",
    "payout",
    "stale balance",
    "stale debt",
]

_PROTOCOL_ACCOUNTING_LABEL_KEYWORDS = [
    "protocol accounting",
    "accounting exploitation",
]

_ACCOUNTING_STATE_ROOT_KEYWORDS = [
    "accounting state",
    "internal accounting",
    "bookkeeping",
    "stale",
    "state order",
    "state ordering",
    "update order",
    "wrong order",
    "before",
    "after",
    "index",
    "reward index",
    "debt index",
    "exchange rate",
    "share price",
    "collateralization",
    "invariant",
    "global accounting",
    "total reserves",
    "total debt",
    "total supply",
]

_ACCOUNTING_FORMULA_KEYWORDS = [
    "formula",
    "calculation",
    "calculate",
    "mathematical",
    "reward calculation",
    "share calculation",
    "debt calculation",
    "collateral calculation",
    "claim eligibility",
    "eligibility",
    "rate variable",
    "reward rate",
    "accounting rule",
    "business invariant",
    "bookkeeping invariant",
    "source-level invariant",
]

_TOKEN_SEMANTIC_KEYWORDS = [
    "token semantic",
    "fee-on-transfer",
    "fee on transfer",
    "deflationary",
    "tax",
    "rebase",
    "reflection",
    "transfer hook",
    "self-transfer",
    "self transfer",
    "balance/event mismatch",
    "event amount",
    "actual balance",
    "locked token",
    "excluded",
    "reflected",
]

_INSUFFICIENT_VALIDATION_LABEL_KEYWORDS = [
    "insufficient validation",
    "insufficientvalidation",
]

_VALIDATION_INPUT_KEYWORDS = [
    "input",
    "parameter",
    "calldata",
    "callback data",
    "callback-data",
    "return value",
    "return-value",
    "token address",
    "market address",
    "swap path",
    "path",
    "amount",
    "signature",
    "selector",
    "unknown selector",
    "externally sourced",
    "external state",
    "oracle value",
    "price value",
    "business state",
    "state assumption",
]

_VALIDATION_GAP_KEYWORDS = [
    "validation",
    "validate",
    "unchecked",
    "unvalidated",
    "malformed",
    "fake",
    "invalid",
    "inconsistent",
    "adversarial",
    "boundary",
    "range",
    "min",
    "max",
    "precision",
    "rounding",
    "freshness",
    "deviation",
    "whitelist",
    "allowlist",
    "trusted",
    "trust",
    "entitlement",
    "balance sufficiency",
    "business invariant",
    "invariant",
    "guard",
    "check",
]

_AUTHORIZATION_BOUNDARY_KEYWORDS = [
    "authorization",
    "authorisation",
    "permission",
    "access control",
    "owner",
    "role",
    "admin",
    "whitelist",
    "allowlist",
    "privileged",
    "callback sender",
    "callback caller",
    "initiator",
]

# ---------------------------------------------------------------------------
# View inference from condition description keywords
# ---------------------------------------------------------------------------

def infer_packet_views(condition: RuleCondition) -> List[str]:
    """Infer relevant packet views from condition description keywords."""
    desc = condition.description.lower()
    views: List[str] = []
    seen: set = set()
    for group, keywords in _GROUP_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            for packet_view in _PACKET_VIEW_GROUPS.get(group, []):
                if packet_view not in seen:
                    views.append(packet_view)
                    seen.add(packet_view)
    return views or list(_DEFAULT_PACKET_REFS)


def infer_packet_views_for_rule(
    condition: RuleCondition,
    *,
    attack_label: str = "",
) -> List[str]:
    policy = infer_followup_policy_for_rule(condition, attack_label=attack_label)
    if policy.get("default_evidence_refs"):
        return list(policy["default_evidence_refs"])
    return infer_packet_views(condition)


def _infer_matched_groups(condition: RuleCondition) -> List[str]:
    """Return which semantic groups match this condition's description."""
    desc = condition.description.lower()
    matched: List[str] = []
    for group, keywords in _GROUP_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            matched.append(group)
    return matched or ["fund", "trace", "state", "event"]


def _views_for_groups(groups: List[str]) -> List[str]:
    """Collect deduplicated views from semantic groups, preserving order."""
    views: List[str] = []
    seen: set = set()
    for group in groups:
        for v in _PACKET_VIEW_GROUPS.get(group, []):
            if v not in seen:
                views.append(v)
                seen.add(v)
    return views


def _normalized_condition_text(condition: RuleCondition) -> str:
    return str(condition.description or "").lower().replace("_", " ").replace("-", " ")


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _normalized_attack_label(attack_label: str = "") -> str:
    return normalize_attack_label(attack_label, default="")


def _question_needs_identity_default(
    condition: RuleCondition,
    *,
    attack_label: str = "",
) -> bool:
    label_text = str(attack_label or "").lower().replace("_", " ").replace("-", " ")
    if "access control" not in label_text and "accesscontrol" not in label_text:
        return False
    text = _normalized_condition_text(condition)
    keywords = [
        "caller",
        "owner",
        "role",
        "admin",
        "whitelist",
        "allowlist",
        "authorized",
        "authorization",
        "permission",
        "callback sender",
        "callback caller",
        "beneficiary",
        "keeper",
    ]
    return _contains_any(text, keywords)


def _is_selector_semantics_text(text: str) -> bool:
    normalized = str(text or "").lower().replace("_", " ").replace("-", " ")
    return any(
        keyword in normalized
        for keyword in (
            "selector",
            "unknown function",
            "undecoded",
            "function semantics",
        )
    )


def _access_control_identity_default_views(
    condition: RuleCondition,
    *,
    attack_label: str = "",
) -> List[str]:
    if not _question_needs_identity_default(condition, attack_label=attack_label):
        return []
    views = [
        "operation_summary_view",
        "evidence_adequacy_view",
        "address_labels",
        "beneficiary_controller_view",
        "source_unavailable_auth_view",
        "critical_call_view",
        "trace_outline_view",
    ]
    if _is_selector_semantics_text(condition.description):
        views.append("unknown_selector_view")
    return [view for view in views if view in PACKET_VIEW_CATALOG]


def _condition_needs_value_release_default(condition: RuleCondition) -> bool:
    text = _normalized_condition_text(condition)
    keywords = [
        "payout",
        "release",
        "withdraw",
        "claim",
        "borrow",
        "mint",
        "redeem",
        "asset",
        "value",
        "profit",
        "loss",
    ]
    return _contains_any(text, keywords)


def _access_control_core_plan_policy(condition: RuleCondition) -> Optional[Dict[str, object]]:
    condition_id = str(condition.id or "").strip().upper()
    packet_tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
    ]
    local_tools = [
        *packet_tools,
        "get_local_call_context",
    ]
    authorization_tools = [
        *local_tools,
        "read_function_chunk",
    ]
    if condition_id == "C1":
        return {
            "default_evidence_refs": [
                "tx_card",
                "operation_summary_view",
                "critical_call_argument_view",
                "critical_call_view",
                "trace_outline_view",
                "value_release_view",
            ],
            "allowed_followup_views": [
                "critical_call_argument_view",
                "unknown_selector_view",
                "semantic_state_delta_view",
                "value_release_view",
                "beneficiary_controller_view",
                "state_change_view",
                "contribution_vs_payout_view",
                "reentrancy_state_order_summary_view",
            ],
            "allowed_tools": list(packet_tools),
            "max_followups": 2,
        }
    if condition_id == "C2":
        return {
            "default_evidence_refs": [
                "address_labels",
                "critical_call_view",
                "critical_call_argument_view",
                "source_unavailable_auth_view",
                "beneficiary_controller_view",
                "value_release_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                "unknown_selector_view",
                "source_unavailable_auth_view",
                "critical_call_argument_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "semantic_state_delta_view",
                "state_change_view",
                "beneficiary_controller_view",
                "reentrancy_state_order_summary_view",
            ],
            "allowed_tools": list(authorization_tools),
            "max_followups": 2,
        }
    if condition_id == "C3":
        return {
            "default_evidence_refs": [
                "tx_card",
                "value_release_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                "critical_call_argument_view",
                "state_change_view",
                "external_fundflow_view",
                "transfer_event_view",
                "beneficiary_controller_view",
                "value_release_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
            ],
            "allowed_tools": list(local_tools),
            "max_followups": 2,
        }
    if condition_id == "E1":
        return {
            "default_evidence_refs": [
                "tx_card",
                "address_labels",
                "beneficiary_controller_view",
                "critical_call_view",
                "critical_call_argument_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                "unknown_selector_view",
                "semantic_state_delta_view",
                "value_release_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_by_id",
                "read_evidence_context",
                "get_local_call_context",
            ],
            "max_followups": 1,
        }
    if condition_id == "E2":
        return {
            "default_evidence_refs": [
                "tx_card",
                "operation_summary_view",
                "critical_call_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "value_release_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                "critical_call_argument_view",
                "semantic_state_delta_view",
                "unknown_selector_view",
                "state_change_view",
                "beneficiary_controller_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_by_id",
                "read_evidence_context",
                "get_local_call_context",
            ],
            "max_followups": 1,
        }
    return None


def _price_manipulation_core_plan_policy(
    condition: RuleCondition,
) -> Optional[Dict[str, object]]:
    """Return price-source-specific evidence routes.

    Price manipulation is intentionally separate from the broader market
    manipulation label. In particular, skim/donate/MEV profiles are not price
    evidence unless a distorted price or reserve is subsequently consumed.
    """
    condition_id = str(condition.id or "").strip().upper()
    common_tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ]
    if condition_id == "C1":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
            ],
            "allowed_followup_views": [
                "event_view",
                "semantic_state_delta_view",
                "critical_call_view",
                "critical_call_argument_view",
                "trace_outline_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    if condition_id == "C2":
        return {
            "default_evidence_refs": [
                "market_mechanism_profile_view",
                "price_relevant_state_view",
                "critical_call_view",
                "critical_call_argument_view",
            ],
            "allowed_followup_views": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "amm_reserve_transition_view",
                "event_view",
                "semantic_state_delta_view",
                "state_change_view",
                "trace_outline_view",
                "value_release_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 2,
        }
    if condition_id == "C3":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "value_release_view",
                "contribution_vs_payout_view",
            ],
            "allowed_followup_views": [
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "external_fundflow_view",
                "participant_net_delta_view",
                "beneficiary_controller_view",
                "critical_call_view",
                "trace_outline_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 2,
        }
    if condition_id == "E1":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
            ],
            "allowed_followup_views": [
                "critical_call_view",
                "critical_call_argument_view",
                "event_view",
                "trace_outline_view",
                "value_release_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    if condition_id == "E2":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "contribution_vs_payout_view",
                "beneficiary_controller_view",
            ],
            "allowed_followup_views": [
                "event_view",
                "semantic_state_delta_view",
                "critical_call_view",
                "price_relevant_state_view",
                "external_fundflow_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    return None


def _market_manipulation_core_plan_policy(
    condition: RuleCondition,
) -> Optional[Dict[str, object]]:
    condition_id = str(condition.id or "").strip().upper()
    common_tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ]
    market_defaults = [
        "operation_summary_view",
        "evidence_adequacy_view",
        "market_mechanism_profile_view",
        "price_relevant_state_view",
        "amm_reserve_transition_view",
    ]
    market_followups = [
        "market_mechanism_profile_view",
        "critical_call_view",
        "event_view",
        "semantic_state_delta_view",
        "state_change_view",
        "transfer_event_view",
        "external_fundflow_view",
        "beneficiary_controller_view",
        "trace_outline_view",
    ]
    token_views = [
        "token_semantic_delta_summary_view",
        "token_accounting_origin_view",
    ]
    if condition_id == "C1":
        return {
            "default_evidence_refs": [
                *market_defaults,
                "critical_call_view",
            ],
            "allowed_followup_views": [
                *market_followups,
                *token_views,
                "value_release_view",
                "participant_net_delta_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    if condition_id == "C2":
        return {
            "default_evidence_refs": [
                "market_mechanism_profile_view",
                "price_relevant_state_view",
                "critical_call_view",
                "critical_call_argument_view",
            ],
            "allowed_followup_views": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "amm_reserve_transition_view",
                "event_view",
                "semantic_state_delta_view",
                "state_change_view",
                "external_fundflow_view",
                "transfer_event_view",
                "contribution_vs_payout_view",
                "beneficiary_controller_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    if condition_id == "C3":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "value_release_view",
                "contribution_vs_payout_view",
            ],
            "allowed_followup_views": [
                "market_mechanism_profile_view",
                "external_fundflow_view",
                "beneficiary_controller_view",
                "profit_loss_view",
                "transfer_event_view",
                "critical_call_view",
                "trace_outline_view",
                "semantic_state_delta_view",
                "amm_reserve_transition_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 2,
        }
    if condition_id == "E1":
        return {
            "default_evidence_refs": [
                *market_defaults,
                "critical_call_view",
            ],
            "allowed_followup_views": [
                *market_followups,
                "value_release_view",
                "contribution_vs_payout_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    if condition_id == "E2":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
                "amm_reserve_transition_view",
            ],
            "allowed_followup_views": [
                "transfer_event_view",
                "semantic_state_delta_view",
                "state_change_view",
                "event_view",
                "critical_call_view",
                "price_relevant_state_view",
                "contribution_vs_payout_view",
                "external_fundflow_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    return None


def _protocol_accounting_core_plan_policy(
    condition: RuleCondition,
) -> Optional[Dict[str, object]]:
    condition_id = str(condition.id or "").strip().upper()
    common_tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ]
    source_tools = _dedupe_step_ids([*common_tools, "read_function_chunk"])
    accounting_locators = [
        "operation_summary_view",
        "evidence_adequacy_view",
        "semantic_state_delta_view",
        "trace_outline_view",
        "critical_call_view",
        "reentrancy_state_order_summary_view",
    ]
    accounting_detail = [
        "state_change_view",
        "reentrancy_state_order_view",
        "critical_call_argument_view",
        "event_view",
        "unknown_selector_view",
        "protocol_accounting_outcome_view",
        "value_release_view",
        "participant_net_delta_view",
        "contribution_vs_payout_view",
    ]
    outcome_defaults = [
        "operation_summary_view",
        "evidence_adequacy_view",
        "protocol_accounting_outcome_view",
        "value_release_view",
        "contribution_vs_payout_view",
        "participant_net_delta_view",
        "semantic_state_delta_view",
    ]
    outcome_followups = [
        "state_change_view",
        "critical_call_view",
        "critical_call_argument_view",
        "event_view",
        "external_fundflow_view",
        "beneficiary_controller_view",
        "profit_loss_view",
        "trace_outline_view",
    ]
    if condition_id == "C1":
        return {
            "default_evidence_refs": list(accounting_locators),
            "allowed_followup_views": list(accounting_detail),
            "allowed_tools": list(source_tools),
            "max_followups": 2,
        }
    if condition_id == "C2":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
            ],
            "allowed_followup_views": [
                "state_change_view",
                "value_release_view",
                "protocol_accounting_outcome_view",
                "participant_net_delta_view",
                "event_view",
                "trace_outline_view",
                "unknown_selector_view",
                "external_fundflow_view",
                "beneficiary_controller_view",
            ],
            "allowed_tools": list(source_tools),
            "max_followups": 2,
        }
    if condition_id == "C3":
        return {
            "default_evidence_refs": list(outcome_defaults),
            "allowed_followup_views": list(outcome_followups),
            "allowed_tools": list(common_tools),
            "max_followups": 2,
        }
    if condition_id == "E1":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "protocol_accounting_outcome_view",
                "contribution_vs_payout_view",
                "semantic_state_delta_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "value_release_view",
                "participant_net_delta_view",
                "state_change_view",
                "event_view",
                "critical_call_argument_view",
                "trace_outline_view",
                "beneficiary_controller_view",
            ],
            "allowed_tools": list(source_tools),
            "max_followups": 2,
        }
    if condition_id == "E2":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
                "reentrancy_state_order_summary_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "flash_or_atomic_capital_view",
                "semantic_state_delta_view",
                "state_change_view",
                "protocol_accounting_outcome_view",
                "value_release_view",
                "participant_net_delta_view",
                "external_fundflow_view",
            ],
            "allowed_tools": list(source_tools),
            "max_followups": 2,
        }
    return None


def _flashloans_core_plan_policy(
    condition: RuleCondition,
) -> Optional[Dict[str, object]]:
    condition_id = str(condition.id or "").strip().upper()
    common_tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ]
    capital_defaults = [
        "operation_summary_view",
        "evidence_adequacy_view",
        "flash_or_atomic_capital_view",
        "trace_outline_view",
        "critical_call_view",
    ]
    capital_followups = [
        "external_fundflow_view",
        "transfer_event_view",
        "event_view",
        "critical_call_argument_view",
        "value_release_view",
        "participant_net_delta_view",
        "contribution_vs_payout_view",
        "profit_loss_view",
    ]
    mechanism_followups = [
        "semantic_state_delta_view",
        "state_change_view",
        "price_relevant_state_view",
        "amm_reserve_transition_view",
        "event_view",
        "unknown_selector_view",
        "critical_call_argument_view",
    ]
    outcome_defaults = [
        "operation_summary_view",
        "evidence_adequacy_view",
        "value_release_view",
        "contribution_vs_payout_view",
        "participant_net_delta_view",
    ]
    if condition_id == "C1":
        return {
            "default_evidence_refs": list(capital_defaults),
            "allowed_followup_views": list(capital_followups),
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    if condition_id == "C2":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "trace_outline_view",
                "critical_call_view",
                "semantic_state_delta_view",
            ],
            "allowed_followup_views": [
                "critical_call_argument_view",
                "state_change_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "event_view",
                "unknown_selector_view",
                "external_fundflow_view",
            ],
            "allowed_tools": _dedupe_step_ids([
                *common_tools,
                "read_function_chunk",
            ]),
            "max_followups": 2,
        }
    if condition_id == "C3":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
            ],
            "allowed_followup_views": [
                "external_fundflow_view",
                "beneficiary_controller_view",
                "profit_loss_view",
                "transfer_event_view",
                "semantic_state_delta_view",
                "trace_outline_view",
                "critical_call_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 2,
        }
    if condition_id == "E1":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "external_fundflow_view",
                "contribution_vs_payout_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                "critical_call_view",
                "value_release_view",
                "participant_net_delta_view",
                "profit_loss_view",
                "event_view",
                "transfer_event_view",
            ],
            "allowed_tools": list(common_tools),
            "max_followups": 1,
        }
    if condition_id == "E2":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "semantic_state_delta_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "reentrancy_state_order_summary_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "state_change_view",
                "semantic_state_delta_view",
                "critical_call_argument_view",
                "unknown_selector_view",
            ],
            "allowed_tools": _dedupe_step_ids([
                *common_tools,
                "read_function_chunk",
            ]),
            "max_followups": 2,
        }
    return None


def _reentrancy_core_plan_policy(
    condition: RuleCondition,
) -> Optional[Dict[str, object]]:
    """Stable baseline view profile for reusable reentrancy plans."""
    text = str(condition.description or "").lower().replace("_", " ").replace("-", " ")
    state_order = is_reentrant_state_order_text(
        condition.description,
        attack_label="reentrancy",
    )
    value_effect = _condition_needs_value_release_default(condition)
    structural_reentry = any(
        keyword in text for keyword in _STATE_ORDER_REENTRANCY_KEYWORDS
    )
    tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ]
    common_defaults = [
        "operation_summary_view",
        "evidence_adequacy_view",
    ]
    state_detail_followups = [
        "reentrancy_state_order_view",
        "state_change_view",
        "semantic_state_delta_view",
    ]
    if state_order:
        return {
            "default_evidence_refs": [
                *common_defaults,
                "reentrancy_candidate_catalog_view",
                "reentrancy_state_order_summary_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                *state_detail_followups,
                "value_release_view",
                "critical_call_argument_view",
            ],
            "allowed_tools": list(tools),
            "max_followups": 2,
        }
    if value_effect:
        return {
            "default_evidence_refs": [
                *common_defaults,
                "value_release_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "reentrancy_state_order_summary_view",
                "trace_outline_view",
                *state_detail_followups,
                "critical_call_argument_view",
                "contribution_vs_payout_view",
            ],
            "allowed_tools": list(tools),
            "max_followups": 1,
        }
    if structural_reentry:
        return {
            "default_evidence_refs": [
                *common_defaults,
                "reentrancy_candidate_catalog_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "reentrancy_state_order_summary_view",
                "trace_outline_view",
                *state_detail_followups,
                "value_release_view",
                "critical_call_argument_view",
            ],
            "allowed_tools": list(tools),
            "max_followups": 1,
        }
    return None


def _iv_question_mentions_state_order_text(text: str) -> bool:
    normalized = str(text or "").lower().replace("_", " ").replace("-", " ")
    return any(
        keyword in normalized
        for keyword in (
            "reentrancy",
            "reentrant",
            "nested execution",
            "stale nested",
            "state-order",
            "state order",
            "callback order",
        )
    )


def _infer_insufficient_validation_condition_role(
    condition_id: str,
    text: str,
) -> str:
    cid = str(condition_id or "").strip().upper()
    normalized = str(text or "").lower().replace("_", " ").replace("-", " ")
    if cid == "C1":
        return "path_consumption"
    if cid == "C2":
        return "validation_gap"
    if cid == "C3":
        return "incorrect_outcome"
    if cid.startswith("E") and _contains_any(
        normalized,
        [
            "access",
            "caller",
            "owner",
            "role",
            "whitelist",
            "allowlist",
            "authorization",
            "authorisation",
            "callback sender",
            "callback-sender",
        ],
    ):
        return "access_control_exclusion"
    if cid.startswith("E") and _contains_any(
        normalized,
        [
            "market",
            "oracle",
            "reentrancy",
            "reentrant",
            "token",
            "accounting",
            "price",
        ],
    ):
        return "other_mechanism_exclusion"
    if _contains_any(
        normalized,
        [
            "consume",
            "input",
            "callback data",
            "return value",
            "oracle",
            "external state",
            "business assumption",
            "business-state",
        ],
    ):
        return "path_consumption"
    if _contains_any(
        normalized,
        [
            "invalid",
            "stale",
            "inconsistent",
            "boundary",
            "precision",
            "malformed",
            "fake",
            "validation",
            "validate",
            "check",
            "verify",
            "guard",
        ],
    ):
        return "validation_gap"
    if _contains_any(
        normalized,
        [
            "outcome",
            "payout",
            "release",
            "loss",
            "corruption",
            "invariant",
            "protocol loss",
            "user loss",
            "accounting",
            "state",
        ],
    ):
        return "incorrect_outcome"
    return "unknown"


def _insufficient_validation_core_plan_policy(
    condition: RuleCondition,
) -> Optional[Dict[str, object]]:
    condition_id = str(condition.id or "").strip().upper()
    role = _infer_insufficient_validation_condition_role(condition_id, condition.description)
    base_tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ]
    if role == "path_consumption":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "semantic_state_delta_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                "unknown_selector_view",
                "semantic_state_delta_view",
                "event_view",
                "state_change_view",
                "value_release_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
            ],
            "allowed_tools": list(base_tools),
            "max_followups": 1,
            "insufficient_validation_role": role,
        }
    if role == "validation_gap":
        followups = [
            "trace_outline_view",
            "unknown_selector_view",
            "semantic_state_delta_view",
            "event_view",
            "state_change_view",
            "value_release_view",
            "contribution_vs_payout_view",
            "participant_net_delta_view",
        ]
        if _iv_question_mentions_state_order_text(condition.description):
            followups.insert(0, "reentrancy_state_order_summary_view")
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "semantic_state_delta_view",
            ],
            "allowed_followup_views": followups,
            "allowed_tools": [*base_tools, "read_function_chunk"],
            "max_followups": 2,
            "insufficient_validation_role": role,
        }
    if role == "incorrect_outcome":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "value_release_view",
            ],
            "allowed_followup_views": [
                "critical_call_view",
                "trace_outline_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
                "unknown_selector_view",
                "state_change_view",
                "beneficiary_controller_view",
            ],
            "allowed_tools": list(base_tools),
            "max_followups": 1,
            "insufficient_validation_role": role,
        }
    if role == "access_control_exclusion":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "source_unavailable_auth_view",
                "beneficiary_controller_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                "tx_card",
                "address_labels",
                "beneficiary_controller_view",
                "unknown_selector_view",
                "value_release_view",
                "semantic_state_delta_view",
                "trace_view",
            ],
            "allowed_tools": [*base_tools, "read_function_chunk"],
            "max_followups": 2,
            "insufficient_validation_role": role,
        }
    if role == "other_mechanism_exclusion":
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "unknown_selector_view",
                "semantic_state_delta_view",
                "event_view",
                "state_change_view",
                "value_release_view",
                "beneficiary_controller_view",
                "trace_view",
            ],
            "allowed_tools": list(base_tools),
            "max_followups": 1,
            "insufficient_validation_role": role,
        }
    return None


def _semantic_route_policy(
    condition: RuleCondition,
    *,
    attack_label: str = "",
) -> Optional[Dict[str, object]]:
    """Route packet views by condition semantics, without using condition ids."""
    text = _normalized_condition_text(condition)
    label_text = str(attack_label or "").lower().replace("_", " ").replace("-", " ")

    market_state = _contains_any(text, _MARKET_STATE_KEYWORDS)
    market_distortion = _contains_any(text, _MARKET_DISTORTION_KEYWORDS)
    value_extraction = _contains_any(text, _VALUE_EXTRACTION_KEYWORDS)
    accounting = _contains_any(text, _ACCOUNTING_KEYWORDS)
    protocol_accounting_label = _contains_any(
        label_text,
        _PROTOCOL_ACCOUNTING_LABEL_KEYWORDS,
    )
    insufficient_validation_label = _contains_any(
        label_text,
        _INSUFFICIENT_VALIDATION_LABEL_KEYWORDS,
    )
    accounting_root = _contains_any(text, _ACCOUNTING_STATE_ROOT_KEYWORDS)
    accounting_formula = _contains_any(text, _ACCOUNTING_FORMULA_KEYWORDS)
    token_semantic = _contains_any(text, _TOKEN_SEMANTIC_KEYWORDS)
    validation_input = _contains_any(text, _VALIDATION_INPUT_KEYWORDS)
    validation_gap = _contains_any(text, _VALIDATION_GAP_KEYWORDS)
    authorization_boundary = _contains_any(text, _AUTHORIZATION_BOUNDARY_KEYWORDS)

    if insufficient_validation_label and authorization_boundary and (
        "exclusion" in text
        or "better explained" in text
        or "primary" in text
        or "pure" in text
    ):
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "tx_card",
                "address_labels",
                "beneficiary_controller_view",
                "unknown_selector_view",
                "value_release_view",
                "semantic_state_delta_view",
                "trace_view",
                "state_change_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_context",
                "get_local_call_context",
                "read_function_chunk",
            ],
            "max_followups": 1,
        }

    if insufficient_validation_label and (
        validation_input or validation_gap
    ) and not value_extraction:
        tools = [
            "read_packet_view",
            "read_evidence_context",
            "get_local_call_context",
        ]
        if validation_gap and "read_function_chunk" not in tools:
            tools.append("read_function_chunk")
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "unknown_selector_view",
                "semantic_state_delta_view",
                "event_view",
                "state_change_view",
                "value_release_view",
                "beneficiary_controller_view",
                "trace_view",
            ],
            "allowed_tools": tools,
            "max_followups": 1,
        }

    if insufficient_validation_label and value_extraction and (
        validation_input or validation_gap or accounting or "invariant" in text
    ):
        tools = [
            "read_packet_view",
            "read_evidence_context",
            "get_local_call_context",
        ]
        if validation_gap and "read_function_chunk" not in tools:
            tools.append("read_function_chunk")
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "value_release_view",
            ],
            "allowed_followup_views": [
                "critical_call_view",
                "trace_outline_view",
                "unknown_selector_view",
                "state_change_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
                "beneficiary_controller_view",
                "event_view",
                "external_fundflow_view",
            ],
            "allowed_tools": tools,
            "max_followups": 1,
        }

    if market_state and any(
        keyword in text
        for keyword in (
            "consume",
            "relies upon",
            "relies on",
            "downstream",
            "subsequent",
            "value-sensitive",
            "price-sensitive",
        )
    ):
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "critical_call_view",
            ],
            "allowed_followup_views": [
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "event_view",
                "semantic_state_delta_view",
                "trace_view",
                "unknown_selector_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_context",
                "get_local_call_context",
            ],
            "max_followups": 1,
        }

    if market_state and market_distortion and not (
        value_extraction and "root cause" not in text and "primary" not in text
    ):
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
            ],
            "allowed_followup_views": [
                "trace_outline_view",
                "event_view",
                "critical_call_view",
                "semantic_state_delta_view",
                "state_change_view",
                "trace_view",
                "participant_net_delta_view",
                "value_release_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_context",
                "get_local_call_context",
            ],
            "max_followups": 1,
        }

    if token_semantic:
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
            ],
            "allowed_followup_views": [
                "semantic_state_delta_view",
                "transfer_event_view",
                "state_change_view",
                "participant_net_delta_view",
                "event_view",
                "contribution_vs_payout_view",
                "external_fundflow_view",
                "critical_call_view",
                "value_release_view",
                "amm_reserve_transition_view",
                "price_relevant_state_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_context",
                "get_local_call_context",
                "read_evidence_by_id",
            ],
            "max_followups": 1,
        }

    if accounting and (protocol_accounting_label or accounting_root or accounting_formula):
        if value_extraction and accounting_formula:
            return {
                "default_evidence_refs": [
                    "operation_summary_view",
                    "evidence_adequacy_view",
                    "semantic_state_delta_view",
                    "contribution_vs_payout_view",
                ],
                "allowed_followup_views": [
                    "state_change_view",
                    "value_release_view",
                    "participant_net_delta_view",
                    "beneficiary_controller_view",
                    "external_fundflow_view",
                    "profit_loss_view",
                    "trace_outline_view",
                    "critical_call_view",
                    "event_view",
                    "unknown_selector_view",
                ],
                "allowed_tools": [
                    "read_packet_view",
                    "read_evidence_context",
                    "get_local_call_context",
                    "read_function_chunk",
                ],
                "max_followups": 1,
            }
        if value_extraction and not (accounting_root or accounting_formula):
            return {
                "default_evidence_refs": [
                    "operation_summary_view",
                    "evidence_adequacy_view",
                    "contribution_vs_payout_view",
                    "value_release_view",
                ],
                "allowed_followup_views": [
                    "semantic_state_delta_view",
                    "state_change_view",
                    "participant_net_delta_view",
                    "beneficiary_controller_view",
                    "external_fundflow_view",
                    "profit_loss_view",
                    "critical_call_view",
                    "trace_outline_view",
                    "event_view",
                ],
                "allowed_tools": [
                    "read_packet_view",
                    "read_evidence_context",
                    "get_local_call_context",
                ],
                "max_followups": 1,
            }
        tools = [
            "read_packet_view",
            "read_evidence_context",
            "get_local_call_context",
        ]
        if accounting_formula and "read_function_chunk" not in tools:
            tools.append("read_function_chunk")
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "trace_outline_view",
            ],
            "allowed_followup_views": [
                "state_change_view",
                "critical_call_view",
                "event_view",
                "value_release_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
                "trace_view",
                "unknown_selector_view",
                "beneficiary_controller_view",
            ],
            "allowed_tools": tools,
            "max_followups": 1,
        }

    if accounting and not value_extraction:
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
            ],
            "allowed_followup_views": [
                "value_release_view",
                "participant_net_delta_view",
                "trace_outline_view",
                "state_change_view",
                "event_view",
                "trace_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_context",
                "get_local_call_context",
            ],
            "max_followups": 1,
        }

    if value_extraction:
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "value_release_view",
                "participant_net_delta_view",
            ],
            "allowed_followup_views": [
                "contribution_vs_payout_view",
                "beneficiary_controller_view",
                "external_fundflow_view",
                "profit_loss_view",
                "transfer_event_view",
                "semantic_state_delta_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_context",
                "get_local_call_context",
            ],
            "max_followups": 1,
        }

    return None


def _infer_dynamic_followup_policy(
    condition: RuleCondition,
    *,
    attack_label: str = "",
) -> Dict[str, object]:
    """Build a follow-up policy dynamically from condition description keywords."""
    if _normalized_attack_label(attack_label) == "access_control":
        access_control_policy = _access_control_core_plan_policy(condition)
        if access_control_policy:
            return access_control_policy
    if _normalized_attack_label(attack_label) == "insufficient_validation":
        insufficient_validation_policy = _insufficient_validation_core_plan_policy(condition)
        if insufficient_validation_policy:
            return insufficient_validation_policy
    if _normalized_attack_label(attack_label) == "price_manipulation":
        price_policy = _price_manipulation_core_plan_policy(condition)
        if price_policy:
            return price_policy
    if _normalized_attack_label(attack_label) == "market_manipulation":
        market_policy = _market_manipulation_core_plan_policy(condition)
        if market_policy:
            return market_policy
    if _normalized_attack_label(attack_label) == "protocol_accounting_exploitation":
        protocol_accounting_policy = _protocol_accounting_core_plan_policy(condition)
        if protocol_accounting_policy:
            return protocol_accounting_policy
    if _normalized_attack_label(attack_label) == "flashloans":
        flashloans_policy = _flashloans_core_plan_policy(condition)
        if flashloans_policy:
            return flashloans_policy
    if _normalized_attack_label(attack_label) == "reentrancy":
        reentrancy_policy = _reentrancy_core_plan_policy(condition)
        if reentrancy_policy:
            return reentrancy_policy

    if is_reentrant_state_order_text(
        condition.description,
        attack_label=attack_label,
    ):
        return {
            "default_evidence_refs": [
                "operation_summary_view",
                "evidence_adequacy_view",
                "reentrancy_state_order_summary_view",
                "critical_call_view",
            ] + (["value_release_view"] if _condition_needs_value_release_default(condition) else []),
            "allowed_followup_views": [
                "trace_outline_view",
                "reentrancy_state_order_view",
                "semantic_state_delta_view",
                "state_change_view",
                "unknown_selector_view",
                "trace_view",
                "event_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
                "value_release_view",
            ],
            "allowed_tools": [
                "read_packet_view",
                "read_evidence_by_id",
                "read_evidence_context",
                "get_local_call_context",
            ],
            "max_followups": 1,
        }

    semantic_policy = _semantic_route_policy(condition, attack_label=attack_label)
    if semantic_policy:
        return semantic_policy

    matched_groups = _infer_matched_groups(condition)
    core_views = _views_for_groups(matched_groups)
    source_level = source_dependency_level(condition.description, attack_label)
    source_required = source_level == "required"
    source_helpful = source_level == "helpful"
    followup_candidates: List[str] = []

    identity_defaults = _access_control_identity_default_views(
        condition,
        attack_label=attack_label,
    )
    default_refs = list(identity_defaults or _COMMON_PREFIX_VIEWS)
    for v in core_views:
        if v not in default_refs:
            default_refs.append(v)

    if not any(v in default_refs for v in ("trace_outline_view",)):
        if matched_groups and any(g in matched_groups for g in ("trace", "state", "atomic_capital")):
            default_refs.append("trace_outline_view")
    if source_required or source_helpful:
        source_locators = [
            "classification_digest_view",
            "trace_outline_view",
            "critical_call_view",
        ]
        if _is_selector_semantics_text(condition.description):
            source_locators.append("unknown_selector_view")
        for view in source_locators:
            if view not in default_refs:
                default_refs.append(view)
        for view in (
            "value_release_view",
            "state_change_view",
            "beneficiary_controller_view",
            "semantic_state_delta_view",
            "address_labels",
            "event_view",
            "transfer_event_view",
            "participant_net_delta_view",
            "contribution_vs_payout_view",
        ):
            if view not in followup_candidates:
                followup_candidates.append(view)

    all_groups = list(_PACKET_VIEW_GROUPS.keys())
    for group in all_groups:
        if group not in matched_groups:
            for v in _PACKET_VIEW_GROUPS.get(group, []):
                if v not in default_refs and v not in followup_candidates:
                    followup_candidates.append(v)
    followup_views = followup_candidates[:5]

    tools = ["read_packet_view", "read_evidence_by_id", "read_evidence_context", "get_local_call_context"]
    if source_required:
        for view in _SOURCE_FOLLOWUP_VIEWS:
            if view not in followup_views:
                followup_views.append(view)
        tools.append("read_function_chunk")
    elif source_helpful:
        for view in (
            "critical_call_view",
            "unknown_selector_view",
            "semantic_state_delta_view",
            "state_change_view",
            "event_view",
        ):
            if view not in followup_views:
                followup_views.append(view)

    return {
        "default_evidence_refs": default_refs,
        "allowed_followup_views": followup_views,
        "allowed_tools": tools,
        "max_followups": 2 if source_required else 1,
    }


def source_dependency_level(text: str, attack_label: str = "") -> str:
    """Classify whether a local question merely benefits from source or requires it."""
    label_text = str(attack_label or "").lower().replace("_", " ").replace("-", " ")
    text_only = str(text or "").lower().replace("_", " ").replace("-", " ")
    normalized_label = _normalized_attack_label(attack_label)
    value_only = _contains_any(text_only, _VALUE_EXTRACTION_KEYWORDS) and not _contains_any(
        text_only,
        [
            "validation",
            "validate",
            "check",
            "authorization",
            "role",
            "owner",
            "admin",
            "formula",
            "selector",
            "storage",
            "source",
            "msg.sender",
            "callback sender",
        ],
    )
    if value_only:
        return "none"
    if normalized_label == "access_control":
        ac_required_keywords = [
            "onlyowner",
            "only owner",
            "modifier",
            "msg.sender",
            "role mapping",
            "signature verification formula",
            "callback sender validation implementation",
            "callback caller validation implementation",
            "source-level",
            "source code",
            "implementation",
            "function visibility",
            "storage-slot semantics",
            "storage slot semantics",
        ]
        if _contains_any(text_only, ac_required_keywords):
            return "required"
        ac_behavioral_keywords = [
            "sensitive by effect",
            "authorization-setting",
            "permission-modifying",
            "role/permission",
            "self-granted",
            "self granted",
            "unprotected initializer",
            "unprotected initialization",
            "uninitialized",
            "critical state initialization",
            "protocol-critical state",
            "callback handler",
            "non-standard path",
            "non standard path",
            "public function",
            "protected state mutation",
            "privilege/approval/authorization",
            "protocol-controlled",
            "shared assets",
            "victim",
            "privileged assets",
            "beyond entitlement",
            "legitimate contribution",
            "behavioral evidence",
            "source-unavailable",
            "source unavailable",
            "missing/bypassed authorization",
            "missing authorization",
            "bypassed authorization",
        ]
        if _contains_any(text_only, ac_behavioral_keywords):
            return "helpful"
    if _contains_any(text_only, _SOURCE_REQUIRED_KEYWORDS):
        return "required"
    if any(keyword in text_only for keyword in ("require ", "require(", "require:")) and _contains_any(
        text_only,
        [
            "validation",
            "caller",
            "callback",
            "role",
            "owner",
            "admin",
            "whitelist",
            "allowlist",
            "authorization",
            "permission",
        ],
    ):
        return "required"
    if " check" in text_only and _contains_any(
        text_only,
        [
            "validation",
            "caller",
            "callback",
            "role",
            "owner",
            "admin",
            "whitelist",
            "allowlist",
            "authorization",
            "permission",
        ],
    ):
        return "required"
    if _contains_any(label_text, _PROTOCOL_ACCOUNTING_LABEL_KEYWORDS) and (
        _contains_any(text_only, ["source-level formula", "accounting formula", "price formula"])
    ):
        return "required"
    if _contains_any(label_text, _INSUFFICIENT_VALIDATION_LABEL_KEYWORDS) and (
        _contains_any(
            text_only,
            [
                "validation logic",
                "validate ",
                "validate(",
                "validate:",
                "validated by",
                "callback sender",
                "callback initiator",
                "return-value check",
                "return value check",
                "whitelist check",
                "allowlist check",
                "range check",
                "freshness check",
            ],
        )
    ):
        return "required"
    if _contains_any(label_text, _PROTOCOL_ACCOUNTING_LABEL_KEYWORDS) and (
        _contains_any(text_only, _ACCOUNTING_FORMULA_KEYWORDS)
        or _contains_any(text_only, _ACCOUNTING_STATE_ROOT_KEYWORDS)
    ):
        return "helpful"
    if _contains_any(label_text, _INSUFFICIENT_VALIDATION_LABEL_KEYWORDS) and (
        _contains_any(text_only, _VALIDATION_GAP_KEYWORDS)
        or _contains_any(text_only, _VALIDATION_INPUT_KEYWORDS)
        or _contains_any(text_only, _AUTHORIZATION_BOUNDARY_KEYWORDS)
    ):
        return "helpful"
    if _contains_any(text_only, _SOURCE_HELPFUL_KEYWORDS):
        return "helpful"
    if any(keyword in text_only for keyword in _SOURCE_DEPENDENT_KEYWORDS):
        return "helpful"
    return "none"


def is_source_dependent_text(text: str, attack_label: str = "") -> bool:
    return source_dependency_level(text, attack_label) in {"helpful", "required"}


def is_source_required_text(text: str, attack_label: str = "") -> bool:
    return source_dependency_level(text, attack_label) == "required"


def is_reentrant_state_order_text(
    text: str,
    attack_label: str = "",
) -> bool:
    haystack = str(text or "").lower().replace("_", " ").replace("-", " ")
    reentrancy_context = (
        _normalized_attack_label(attack_label) == "reentrancy"
        or any(keyword in haystack for keyword in _STATE_ORDER_REENTRANCY_KEYWORDS)
    )
    return reentrancy_context and any(
        keyword in haystack for keyword in _STATE_ORDER_KEYWORDS
    )


def followup_round_policy(
    condition_or_text: RuleCondition | JudgeStep | str,
    *,
    attack_label: str = "",
    condition_id: str = "",
) -> Dict[str, object]:
    """Return the shared semantic cap for Judge follow-up rounds.

    Two rounds are reserved for evidence paths where the second request can
    depend on the first observation. Tool names alone do not establish that
    dependency.
    """
    text = (
        condition_or_text.description
        if isinstance(condition_or_text, RuleCondition)
        else condition_or_text.question
        if isinstance(condition_or_text, JudgeStep)
        else str(condition_or_text or "")
    )
    resolved_condition_id = str(
        condition_id
        or getattr(condition_or_text, "condition_id", "")
        or getattr(condition_or_text, "id", "")
        or ""
    ).strip().upper()
    label = _normalized_attack_label(attack_label)
    source_level = source_dependency_level(text, label)
    positive_state_order = is_reentrant_state_order_text(
        text,
        attack_label=label,
    )
    structured_state_order = False
    if isinstance(condition_or_text, JudgeStep):
        required_state_inputs = {
            "reentrancy_candidate",
            "reentrancy_value_effect_summary",
        }
        structured_state_order = (
            label == "reentrancy"
            and resolved_condition_id == "C3"
            and condition_or_text.state_prompt_role == "re_state_order_causality"
            and condition_or_text.produces_state_key
            == "reentrancy_causal_order_summary"
            and required_state_inputs.issubset(
                set(condition_or_text.consumes_state_keys or [])
            )
            and len(set(condition_or_text.depends_on or [])) >= 2
        )
    label_core_sequence = (
        label in {"access_control", "insufficient_validation"}
        and resolved_condition_id.startswith("C")
    )
    reasons: List[str] = []
    if source_level == "required":
        reasons.append("source_required_locate_then_read")
    if structured_state_order:
        reasons.append("reentrancy_state_contract_then_causality")
    elif positive_state_order:
        reasons.append("state_order_locate_then_context")
    if label_core_sequence:
        reasons.append(f"{label}_core_precise_row_then_semantics")
    if label == "price_manipulation" and resolved_condition_id == "C2":
        reasons.append("price_consumption_locate_then_decode")
    if label == "price_manipulation" and resolved_condition_id == "C3":
        reasons.append("price_outcome_effect_then_causal_attribution")
    if label == "market_manipulation" and resolved_condition_id == "C3":
        reasons.append("market_outcome_multi_view_extraction_check")
    if label == "protocol_accounting_exploitation" and resolved_condition_id in {
        "C1",
        "C2",
        "C3",
        "E1",
        "E2",
    }:
        reasons.append("protocol_accounting_candidate_consumption_outcome_check")
    if label == "flashloans" and resolved_condition_id in {"C2", "C3", "E2"}:
        reasons.append("flash_capital_chain_mechanism_then_outcome")
    sequential = bool(reasons)
    minimum_followups = (
        2
        if label == "price_manipulation" and resolved_condition_id == "C2"
        else 0
    )
    return {
        "max_followups": 2 if sequential else 1,
        "minimum_followups": minimum_followups,
        "sequential_followup": sequential,
        "reasons": reasons,
        "source_dependency_level": source_level,
        "condition_id": resolved_condition_id,
        "attack_label": label,
    }


def condition_view_budget(condition_or_text: RuleCondition | JudgeStep | str, attack_label: str = "") -> Dict[str, int]:
    text = (
        condition_or_text.description
        if isinstance(condition_or_text, RuleCondition)
        else condition_or_text.question
        if isinstance(condition_or_text, JudgeStep)
        else str(condition_or_text or "")
    )
    label_budget = LABEL_AWARE_PLAN_BUDGETS.get(_normalized_attack_label(attack_label))
    if label_budget:
        return dict(label_budget)
    source_level = source_dependency_level(text, attack_label)
    if source_level == "required":
        return {
            "max_default_views": SPECIAL_MAX_DEFAULT_VIEWS,
            "max_followup_views": SOURCE_MAX_FOLLOWUP_VIEWS,
            "max_large_views": MAX_DEFAULT_LARGE_VIEWS,
        }
    if is_reentrant_state_order_text(text, attack_label=attack_label):
        return {
            "max_default_views": SPECIAL_MAX_DEFAULT_VIEWS,
            "max_followup_views": STATE_ORDER_MAX_FOLLOWUP_VIEWS,
            "max_large_views": MAX_DEFAULT_LARGE_VIEWS,
        }
    return {
        "max_default_views": ORDINARY_MAX_DEFAULT_VIEWS,
        "max_followup_views": ORDINARY_MAX_FOLLOWUP_VIEWS,
        "max_large_views": MAX_DEFAULT_LARGE_VIEWS,
    }


def resolve_judge_step_view_budget(step: JudgeStep) -> Dict[str, int]:
    """Read the budget fixed by the plan, preserving legacy plan selections."""
    explicit = dict(step.view_budget or {})
    required = {
        "max_default_views",
        "max_followup_views",
        "max_large_views",
    }
    if required.issubset(explicit):
        return {
            "max_default_views": max(1, int(explicit["max_default_views"])),
            "max_followup_views": max(0, int(explicit["max_followup_views"])),
            "max_large_views": max(0, int(explicit["max_large_views"])),
        }

    defaults = list(step.default_evidence_refs or step.evidence_refs)
    followups = list(step.allowed_followup_views or [])
    large_count = sum(
        1
        for view in defaults
        if str(packet_view_render_policy(view).get("cost", "")).lower() == "large"
    )
    return {
        "max_default_views": max(1, len(defaults)),
        "max_followup_views": max(1, len(followups) + len(defaults)),
        "max_large_views": max(1, large_count),
    }


def _trim_view_refs(views: List[str], limit: int) -> List[str]:
    return _dedupe_known_views(views)[: max(0, int(limit or 0))]


def infer_followup_policy_for_rule(
    condition: RuleCondition,
    *,
    attack_label: str = "",
) -> Dict[str, object]:
    policy = _infer_dynamic_followup_policy(condition, attack_label=attack_label)
    source_level = source_dependency_level(condition.description, attack_label)
    source_required = source_level == "required"
    source_helpful = source_level == "helpful"
    label = _normalized_attack_label(attack_label)
    condition_id = str(condition.id or "").strip().upper()
    iv_role = str(policy.get("insufficient_validation_role") or "")
    tools = list(policy.get("allowed_tools", []))
    views = list(policy.get("allowed_followup_views", []))
    iv_policy_handles_source = bool(label == "insufficient_validation" and iv_role)
    if iv_policy_handles_source:
        policy["max_followups"] = int(policy.get("max_followups", 1) or 0)
    elif source_required:
        if "read_evidence_by_id" not in tools:
            tools.append("read_evidence_by_id")
        if "read_evidence_context" not in tools:
            tools.append("read_evidence_context")
        if "read_function_chunk" not in tools:
            tools.append("read_function_chunk")
        for view in _SOURCE_FOLLOWUP_VIEWS:
            if view not in views:
                views.append(view)
        policy["max_followups"] = 2
    elif source_helpful:
        if "read_evidence_by_id" not in tools:
            tools.append("read_evidence_by_id")
        if "read_evidence_context" not in tools:
            tools.append("read_evidence_context")
        policy["max_followups"] = min(1, int(policy.get("max_followups", 1) or 0))
    else:
        policy["max_followups"] = min(1, int(policy.get("max_followups", 1) or 0))
    if (
        label == "insufficient_validation"
        and condition_id.startswith("C")
        and iv_role == "validation_gap"
    ):
        for tool in (
            "read_packet_view",
            "read_evidence_by_id",
            "read_evidence_context",
            "get_local_call_context",
            "read_function_chunk",
        ):
            if tool not in tools:
                tools.append(tool)
        policy["max_followups"] = max(2, int(policy.get("max_followups", 0) or 0))
    elif label == "access_control" and condition_id.startswith("C"):
        role_tools = {
            "C1": (
                "read_packet_view",
                "read_evidence_by_id",
                "read_evidence_context",
            ),
            "C2": (
                "read_packet_view",
                "read_evidence_by_id",
                "read_evidence_context",
                "get_local_call_context",
                "read_function_chunk",
            ),
            "C3": (
                "read_packet_view",
                "read_evidence_by_id",
                "read_evidence_context",
                "get_local_call_context",
            ),
        }.get(condition_id, tuple(tools))
        tools = [tool for tool in tools if tool in set(role_tools)]
        for tool in role_tools:
            if tool not in tools:
                tools.append(tool)
        policy["max_followups"] = max(2, int(policy.get("max_followups", 0) or 0))
    elif label == "reentrancy" and is_reentrant_state_order_text(
        condition.description,
        attack_label=attack_label,
    ):
        policy["max_followups"] = max(
            2,
            int(policy.get("max_followups", 0) or 0),
        )
    elif label == "price_manipulation" and condition_id in {"C2", "C3"}:
        policy["max_followups"] = max(2, int(policy.get("max_followups", 0) or 0))
    elif label == "market_manipulation" and condition_id == "C3":
        policy["max_followups"] = max(2, int(policy.get("max_followups", 0) or 0))
    elif label == "protocol_accounting_exploitation" and condition_id in {
        "C1",
        "C2",
        "C3",
        "E1",
        "E2",
    }:
        policy["max_followups"] = max(2, int(policy.get("max_followups", 0) or 0))
    elif label == "flashloans" and condition_id in {"C2", "C3", "E2"}:
        policy["max_followups"] = max(2, int(policy.get("max_followups", 0) or 0))
    policy["allowed_tools"] = tools
    policy["allowed_followup_views"] = views
    identity_defaults = _access_control_identity_default_views(
        condition,
        attack_label=attack_label,
    )
    if identity_defaults and not (
        label == "access_control"
        and condition_id in {"C1", "C2", "C3", "E1", "E2"}
    ):
        existing_defaults = list(policy.get("default_evidence_refs", []) or [])
        policy["default_evidence_refs"] = [
            *identity_defaults,
            *[view for view in existing_defaults if view not in identity_defaults],
        ]
        identity_followups = [
            "trace_outline_view",
            "critical_call_view",
            "unknown_selector_view",
            "value_release_view",
            "semantic_state_delta_view",
        ]
        existing_followups = list(policy.get("allowed_followup_views", []) or [])
        policy["allowed_followup_views"] = [
            *identity_followups,
            *[view for view in existing_followups if view not in identity_followups],
        ]
        views = list(policy["allowed_followup_views"])
    budget = condition_view_budget(condition, attack_label=attack_label)
    defaults, followups, deferred = constrain_first_pass_view_refs(
        list(policy.get("default_evidence_refs", []) or []),
        list(policy.get("allowed_followup_views", []) or []),
        max_default_views=int(budget["max_default_views"]),
        max_large_views=int(budget["max_large_views"]),
        max_followup_views=int(budget["max_followup_views"]),
    )
    policy["default_evidence_refs"] = defaults
    policy["allowed_followup_views"] = followups
    policy["view_budget"] = dict(budget)
    round_policy = followup_round_policy(
        condition,
        attack_label=attack_label,
        condition_id=condition_id,
    )
    policy["max_followups"] = max(
        int(round_policy.get("minimum_followups", 0) or 0),
        min(
            int(round_policy["max_followups"]),
            int(policy.get("max_followups", 0) or 0),
        ),
    )
    policy["followup_round_policy"] = round_policy
    if deferred:
        policy["first_pass_deferred_views"] = deferred
    return policy


# ---------------------------------------------------------------------------
# Plan compilation
# ---------------------------------------------------------------------------

IV_STATEFUL_RUNTIME_MODE = "insufficient_validation_object_binding_v1"

IV_OBJECT_BINDING_QUESTION_NOTE = (
    "Insufficient-validation object-binding note: treat protocol-controlled "
    "business/accounting state, reward/share/debt/collateral/entitlement "
    "state, cached external-return-derived state, oracle/business state, and "
    "computed intermediate values as eligible consumed object candidates when "
    "they are later trusted by sensitive logic. Do not exclude such an object "
    "solely because it is stored inside the protocol or because a downstream "
    "consumer has a local guard; C1 should preserve the candidate and leave "
    "validation adequacy to the validation-gap condition."
)

IV_VALIDATION_GAP_QUESTION_NOTE = (
    "Insufficient-validation exact-scope note: audit the validation dimension "
    "required by each C1 object. Caller ownership, role, pause, and reentrancy "
    "guards do not validate reward/accounting proportionality, token/path source "
    "identity, callback payload content, oracle freshness, return-value consistency, "
    "or other business invariants. A contract-existence check does not establish "
    "that a supplied token, market, strategy, or target is configured or allowed. "
    "A candidate can be rejected only with positive exact-scope validation evidence; "
    "keep unresolved alternatives explicit without selecting them solely because they "
    "are unresolved. If a wrapper delegates consumption to an internal callee, inspect "
    "that named callee while follow-up budget remains. Access-control, reentrancy, "
    "market, token, and accounting replacement decisions belong to exclusions."
)


def _append_question_note_once(step: JudgeStep, note: str) -> None:
    current = str(step.question or "").strip()
    if not current or not note:
        return
    if note in current:
        return
    step.question = f"{current}\n\n{note}"


ACCESS_CONTROL_STATEFUL_RUNTIME_MODE = "access_control_authorization_binding_v1"
ACCESS_CONTROL_CANDIDATE_STATE_KEY = "access_control_candidate"
ACCESS_CONTROL_BINDING_MODES = {"prompt", "stateful"}
TOKEN_SEMANTIC_STATEFUL_RUNTIME_MODE = "token_semantic_candidate_binding_v1"
TOKEN_SEMANTIC_CANDIDATE_STATE_KEY = "token_semantic_candidate"
PROTOCOL_ACCOUNTING_STATEFUL_RUNTIME_MODE = "protocol_accounting_candidate_binding_v1"
PROTOCOL_ACCOUNTING_CANDIDATE_STATE_KEY = "protocol_accounting_candidate"


def normalize_access_control_binding_mode(value: str | None) -> str:
    mode = str(value or "stateful").strip().lower().replace("-", "_")
    aliases = {
        "prompt_only": "prompt",
        "prompt_level": "prompt",
        "strong": "stateful",
        "strong_binding": "stateful",
    }
    mode = aliases.get(mode, mode)
    if mode not in ACCESS_CONTROL_BINDING_MODES:
        raise ValueError(
            "access_control_binding_mode must be one of: prompt, stateful"
        )
    return mode


def apply_access_control_binding_mode(
    plan: EvidencePlan,
    *,
    rule: EvolvingRule,
    mode: str = "prompt",
) -> EvidencePlan:
    """Apply prompt-only or semantic-role stateful Access Control binding."""
    normalized_mode = normalize_access_control_binding_mode(mode)
    label = _normalized_attack_label(
        (rule.metadata or {}).get("attack_label", "")
    )
    if label != "access_control":
        return plan

    _clear_access_control_stateful_runtime(plan)
    policy = {
        "requested_mode": normalized_mode,
        "effective_mode": "prompt",
        "role_assignment_strategy": "semantic_text_scoring",
    }
    plan.metadata["access_control_binding_policy"] = policy
    if normalized_mode != "stateful":
        return plan

    positive_condition_ids = {
        str(condition.id or "").strip()
        for condition in list(rule.conditions or [])
        if str(condition.id or "").strip()
    }
    core_steps = [
        step
        for step in list(plan.judge_steps or [])
        if str(step.condition_id or step.id or "").strip()
        in positive_condition_ids
    ]
    condition_text_by_id = {
        str(condition.id or "").strip(): str(condition.description or "")
        for condition in list(rule.conditions or [])
        if str(condition.id or "").strip()
    }
    assignment, score_matrix = _infer_access_control_stateful_roles(
        core_steps,
        condition_text_by_id=condition_text_by_id,
    )
    if not assignment:
        assignment = _infer_access_control_canonical_roles(core_steps)
        if assignment:
            policy.update({
                "role_assignment_strategy": "semantic_text_scoring_then_canonical_core_ids",
                "canonical_fallback_used": True,
            })
        else:
            policy.update({
                "effective_mode": "prompt_fallback",
                "fallback_reason": "ambiguous_or_incomplete_positive_semantic_roles",
                "positive_step_count": len(core_steps),
                "role_score_matrix": score_matrix,
                "warning": (
                    "Access Control stateful binding requested but no "
                    "authorization anchor/gap/effect role assignment could be "
                    "made; downstream judges will not share candidate state."
                ),
            })
            return plan

    anchor = assignment["authorization_anchor"]
    gap = assignment["authorization_gap"]
    effect = assignment["protected_effect"]
    binding_view_route_audit = _ensure_access_control_binding_followup_views(
        anchor,
        gap,
        preserve_existing_routes=bool(
            (plan.metadata or {}).get("fixed_plan")
        ),
    )

    anchor.produces_state_key = ACCESS_CONTROL_CANDIDATE_STATE_KEY
    anchor.state_prompt_role = "ac_authorization_anchor"
    anchor.state_output_schema = {
        ACCESS_CONTROL_CANDIDATE_STATE_KEY: {
            "candidates": [{
                "candidate_id": "",
                "candidate_status": "candidate_anchor|unresolved_probe",
                "actor": "",
                "beneficiary": "",
                "sensitive_call_id": "",
                "sensitive_capability": "",
                "protected_target_or_resource": "",
                "effect_hint": "",
                "evidence_ids": [],
                "confidence": "low|medium|high",
            }],
            "selected_candidate_ids": [],
            "probe_candidate_ids": [],
            "binding_confidence": "low|medium|high",
            "unresolved_fields": [],
        }
    }

    gap.depends_on = _dedupe_step_ids([*gap.depends_on, anchor.id])
    gap.consumes_state_keys = _dedupe_step_ids([
        *gap.consumes_state_keys,
        ACCESS_CONTROL_CANDIDATE_STATE_KEY,
    ])
    gap.produces_state_key = "authorization_gap_summary"
    gap.state_prompt_role = "ac_authorization_gap"
    gap.state_output_schema = {
        "authorization_gap_summary": {
            "candidate_assessments": [{
                "candidate_id": "",
                "required_authority": "",
                "authorization_status": (
                    "missing|bypassed|self_granted|unauthorized|present|uncertain"
                ),
                "authorization_evidence_status": (
                    "source_confirmed_missing|missing_behavioral|present|"
                    "contradicted|source_required|uncertain"
                ),
                "authorization_gap": "",
                "same_chain_supported": False,
                "evidence_ids": [],
            }],
            "satisfying_candidate_ids": [],
            "unresolved_candidate_ids": [],
        }
    }

    effect.depends_on = _dedupe_step_ids([
        *effect.depends_on,
        anchor.id,
        gap.id,
    ])
    effect.consumes_state_keys = _dedupe_step_ids([
        *effect.consumes_state_keys,
        ACCESS_CONTROL_CANDIDATE_STATE_KEY,
        "authorization_gap_summary",
    ])
    effect.produces_state_key = "protected_effect_summary"
    effect.state_prompt_role = "ac_protected_effect"
    effect.state_output_schema = {
        "protected_effect_summary": {
            "candidate_assessments": [{
                "candidate_id": "",
                "actor_or_beneficiary": "",
                "protected_target_or_resource": "",
                "protected_effect": "",
                "effect_status": "supported|absent|uncertain",
                "beyond_entitlement_or_contribution": False,
                "causal_link_supported": False,
                "same_chain_supported": False,
                "evidence_ids": [],
            }],
            "attack_candidate_ids": [],
            "unresolved_candidate_ids": [],
        }
    }

    exclusion_condition_ids = {
        str(condition.id or "").strip()
        for condition in list(rule.exclusion_conditions or [])
        if str(condition.id or "").strip()
    }
    exclusion_steps = [
        step
        for step in list(plan.judge_steps or [])
        if str(step.condition_id or step.id or "").strip()
        in exclusion_condition_ids
    ]
    exclusion_state_keys: List[str] = []
    for exclusion in exclusion_steps:
        state_key = f"candidate_exclusion_summary__{exclusion.id}"
        exclusion.depends_on = _dedupe_step_ids([
            *exclusion.depends_on,
            anchor.id,
            gap.id,
            effect.id,
        ])
        exclusion.consumes_state_keys = _dedupe_step_ids([
            *exclusion.consumes_state_keys,
            ACCESS_CONTROL_CANDIDATE_STATE_KEY,
            "authorization_gap_summary",
            "protected_effect_summary",
        ])
        exclusion.produces_state_key = state_key
        exclusion.state_prompt_role = "ac_candidate_exclusion"
        exclusion.state_output_schema = {
            state_key: {
                "candidate_assessments": [{
                "candidate_id": "",
                "exclusion_status": "excluded|not_excluded|uncertain",
                "reason": "",
                    "same_chain_supported": False,
                    "evidence_ids": [],
                }],
                "excluded_candidate_ids": [],
                "unresolved_candidate_ids": [],
            }
        }
        exclusion_state_keys.append(state_key)

    original_step_order = [step.id for step in list(plan.judge_steps or [])]
    selected_ids = {anchor.id, gap.id, effect.id}
    remaining_steps = [
        step for step in list(plan.judge_steps or []) if step.id not in selected_ids
    ]
    plan.judge_steps = [anchor, gap, effect, *remaining_steps]
    dependency_edges = {
        gap.id: [anchor.id],
        effect.id: [anchor.id, gap.id],
        **{
            step.id: [anchor.id, gap.id, effect.id]
            for step in exclusion_steps
        },
    }
    role_steps = {
        "authorization_anchor": anchor.id,
        "authorization_gap": gap.id,
        "protected_effect": effect.id,
    }
    policy.update({
        "effective_mode": "stateful",
        "role_steps": role_steps,
        "role_score_matrix": score_matrix,
        "view_route_audit": binding_view_route_audit,
    })
    plan.metadata["stateful_runtime"] = {
        "enabled": True,
        "mode": ACCESS_CONTROL_STATEFUL_RUNTIME_MODE,
        "core_steps": [anchor.id, gap.id, effect.id],
        "role_steps": role_steps,
        "candidate_state_key": ACCESS_CONTROL_CANDIDATE_STATE_KEY,
        "legacy_candidate_state_key": "authorization_chain_summary",
        "dependency_edges": dependency_edges,
        "candidate_branching": True,
        "max_candidate_chains": 3,
        "candidate_exclusion_state_keys": exclusion_state_keys,
        "original_step_order": original_step_order,
        "description": (
            "The authorization anchor produces one primary and at most two concrete "
            "alternative candidate chains; every "
            "downstream condition assesses candidates by candidate_id; runtime emits "
            "when at least one same-chain candidate satisfies the core conditions and "
            "is not excluded."
        ),
    }
    return plan


def _clear_access_control_stateful_runtime(plan: EvidencePlan) -> None:
    metadata = dict(plan.metadata or {})
    stateful = dict(metadata.get("stateful_runtime") or {})
    if str(stateful.get("mode") or "") != ACCESS_CONTROL_STATEFUL_RUNTIME_MODE:
        return
    dependency_edges = dict(stateful.get("dependency_edges") or {})
    for step in list(plan.judge_steps or []):
        if str(step.state_prompt_role or "").startswith("ac_"):
            step.consumes_state_keys = []
            step.produces_state_key = ""
            step.state_prompt_role = ""
            step.state_output_schema = {}
        added = set(dependency_edges.get(step.id) or [])
        if added:
            step.depends_on = [
                dependency for dependency in step.depends_on
                if dependency not in added
            ]
    metadata.pop("stateful_runtime", None)
    plan.metadata = metadata
    original_order = list(stateful.get("original_step_order") or [])
    if original_order:
        order_index = {
            step_id: index for index, step_id in enumerate(original_order)
        }
        plan.judge_steps = sorted(
            list(plan.judge_steps or []),
            key=lambda step: order_index.get(step.id, len(order_index)),
        )


def _infer_access_control_stateful_roles(
    steps: List[JudgeStep],
    *,
    condition_text_by_id: Dict[str, str] | None = None,
) -> tuple[Dict[str, JudgeStep], Dict[str, Dict[str, int]]]:
    roles = ("authorization_anchor", "authorization_gap", "protected_effect")
    score_matrix: Dict[str, Dict[str, int]] = {}
    condition_text_by_id = dict(condition_text_by_id or {})
    for step in steps:
        condition_id = str(step.condition_id or step.id or "").strip()
        text = " ".join([
            str(step.question or ""),
            str(condition_text_by_id.get(condition_id) or ""),
        ]).lower()
        score_matrix[step.id] = {
            role: _access_control_role_score(text, role)
            for role in roles
        }
    if len(steps) < len(roles):
        return {}, score_matrix

    best: tuple[int, JudgeStep, JudgeStep, JudgeStep] | None = None
    for anchor in steps:
        for gap in steps:
            if gap.id == anchor.id:
                continue
            for effect in steps:
                if effect.id in {anchor.id, gap.id}:
                    continue
                score = (
                    score_matrix[anchor.id]["authorization_anchor"]
                    + score_matrix[gap.id]["authorization_gap"]
                    + score_matrix[effect.id]["protected_effect"]
                )
                candidate = (score, anchor, gap, effect)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        return {}, score_matrix
    selected_scores = (
        score_matrix[best[1].id]["authorization_anchor"],
        score_matrix[best[2].id]["authorization_gap"],
        score_matrix[best[3].id]["protected_effect"],
    )
    if any(score <= 0 for score in selected_scores):
        return {}, score_matrix
    return {
        "authorization_anchor": best[1],
        "authorization_gap": best[2],
        "protected_effect": best[3],
    }, score_matrix


def _infer_access_control_canonical_roles(
    steps: List[JudgeStep],
) -> Dict[str, JudgeStep]:
    """Last-resort Access Control binding for canonical three-core rules.

    The primary path is semantic text scoring. This fallback only handles the
    conventional C1/C2/C3 core layout used by cold-start Access Control rules,
    so a missed phrase does not silently disable state passing for the whole
    episode.
    """
    by_condition = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list(steps or [])
    }
    required = ("C1", "C2", "C3")
    if not all(condition_id in by_condition for condition_id in required):
        return {}
    if len({by_condition[condition_id].id for condition_id in required}) < 3:
        return {}
    return {
        "authorization_anchor": by_condition["C1"],
        "authorization_gap": by_condition["C2"],
        "protected_effect": by_condition["C3"],
    }


def _ensure_access_control_binding_followup_views(
    anchor: JudgeStep,
    gap: JudgeStep,
    *,
    preserve_existing_routes: bool = False,
) -> Dict[str, Any]:
    """Apply bounded stateful AC follow-up without rewriting fixed plans."""
    priorities = {
        "authorization_anchor": [
            "critical_call_argument_view",
            "state_change_view",
            "event_view",
        ],
        "authorization_gap": [
            "source_unavailable_auth_view",
            "critical_call_argument_view",
            "semantic_state_delta_view",
            "beneficiary_controller_view",
            "value_release_view",
            "trace_outline_view",
        ],
    }
    audit: Dict[str, Any] = {}
    for role, step in (
        ("authorization_anchor", anchor),
        ("authorization_gap", gap),
    ):
        before = _dedupe_known_views(list(step.allowed_followup_views or []))
        budget = condition_view_budget(step, attack_label="access_control")
        ranked = _dedupe_known_views(
            [*before, *priorities[role]]
            if preserve_existing_routes
            else [*priorities[role], *before]
        )
        limit = max(0, int(budget["max_followup_views"]))
        selected = ranked[:limit]
        step.allowed_followup_views = selected
        step.max_followups = max(2, int(step.max_followups or 0))
        audit[role] = {
            "strategy": (
                "fixed_plan_routes_then_stateful_defaults"
                if preserve_existing_routes
                else "stateful_role_priority"
            ),
            "max_followup_views": limit,
            "before": before,
            "selected": selected,
            "deprioritized": [view for view in ranked if view not in selected],
        }
    return audit


def _access_control_role_score(text: str, role: str) -> int:
    patterns: Dict[str, tuple[tuple[str, int], ...]] = {
        "authorization_anchor": (
            ("authorization-chain anchor", 12),
            ("sensitive or protected capability", 12),
            ("sensitive/protected capability", 12),
            ("protected capability", 9),
            ("sensitive capability", 7),
            ("sensitive invocation", 7),
            ("sensitive operation", 7),
            ("privileged-operation", 7),
            ("privileged operation", 7),
            ("privileged asset operation", 7),
            ("privileged mint", 7),
            ("privileged burn", 7),
            ("protected execution path", 6),
            ("callback-protected execution path", 6),
            ("protected target", 6),
            ("protected resource", 6),
            ("protocol-controlled state", 6),
            ("shared resources", 6),
            ("privileged access controls", 6),
            ("permission or role change", 5),
            ("configuration", 4),
            ("initializer", 4),
            ("proxy update", 4),
            ("invocation path", 5),
            ("reaches", 4),
            ("exposes", 4),
            ("locate", 3),
        ),
        "authorization_gap": (
            ("lacks legitimate", 12),
            ("lacks the required authorization", 12),
            ("lack the required authorization", 12),
            ("lacks required authorization", 11),
            ("lack required authorization", 11),
            ("required authority", 9),
            ("required authorization", 9),
            ("authorization gap", 9),
            ("missing owner", 7),
            ("missing role", 7),
            ("missing authorization", 7),
            ("bypassed authorization", 7),
            ("self-authorization", 7),
            ("self-granted", 7),
            ("caller validation", 6),
            ("legitimately invoke", 5),
            ("unauthorized", 5),
            ("authority", 4),
        ),
        "protected_effect": (
            ("same unauthorized invocation", 12),
            ("protected effect", 10),
            ("protected state mutation", 10),
            ("protected state", 7),
            ("release/appropriation", 8),
            ("privilege escalation", 8),
            ("unauthorized approval", 8),
            ("unauthorized invocation", 7),
            ("causes", 8),
            ("cause a protected", 8),
            ("directly cause", 7),
            ("causally", 8),
            ("beyond legitimate entitlement", 8),
            ("beyond the actor's entitlement", 8),
            ("beyond the caller's legitimate entitlement", 8),
            ("moves protocol-controlled", 7),
            ("privilege/approval/configuration", 7),
            ("state mutation", 6),
            ("asset release", 5),
            ("resulting", 3),
        ),
    }
    return sum(
        weight for phrase, weight in patterns.get(role, ()) if phrase in text
    )


def apply_token_semantic_stateful_runtime(
    plan: EvidencePlan,
    *,
    rule: EvolvingRule,
) -> EvidencePlan:
    """Annotate Token Semantic plans with same-candidate state flow."""
    label = _normalized_attack_label(
        (rule.metadata or {}).get("attack_label", "")
    )
    if label != "token_semantic_exploitation":
        return plan

    _clear_token_semantic_stateful_runtime(plan)
    policy = {
        "requested_mode": "stateful",
        "effective_mode": "prompt",
        "role_assignment_strategy": "semantic_text_scoring",
    }
    plan.metadata["token_semantic_binding_policy"] = policy

    positive_condition_ids = {
        str(condition.id or "").strip()
        for condition in list(rule.conditions or [])
        if str(condition.id or "").strip()
    }
    core_steps = [
        step
        for step in list(plan.judge_steps or [])
        if str(step.condition_id or step.id or "").strip()
        in positive_condition_ids
    ]
    condition_text_by_id = {
        str(condition.id or "").strip(): str(condition.description or "")
        for condition in list(rule.conditions or [])
        if str(condition.id or "").strip()
    }
    assignment, score_matrix = _infer_token_semantic_stateful_roles(
        core_steps,
        condition_text_by_id=condition_text_by_id,
    )
    if not assignment:
        policy.update({
            "effective_mode": "prompt_fallback",
            "fallback_reason": "ambiguous_or_incomplete_token_semantic_roles",
            "positive_step_count": len(core_steps),
            "role_score_matrix": score_matrix,
            "warning": (
                "Token Semantic stateful binding requested but no token "
                "semantic anchor/reliance/outcome role assignment could be "
                "made; downstream judges will not share token candidate state."
            ),
        })
        return plan

    anchor = assignment["token_semantic_anchor"]
    reliance = assignment["semantic_reliance"]
    outcome = assignment["semantic_outcome"]
    _ensure_token_semantic_binding_followup_views(anchor, reliance, outcome)

    anchor.produces_state_key = TOKEN_SEMANTIC_CANDIDATE_STATE_KEY
    anchor.state_prompt_role = "ts_token_semantic_anchor"
    anchor.state_output_schema = {
        TOKEN_SEMANTIC_CANDIDATE_STATE_KEY: {
            "candidates": [{
                "candidate_id": "",
                "token_contract": "",
                "token_symbol_or_label": "",
                "mechanism_type": (
                    "fee_on_transfer|reflection|rebase|deflationary_burn|"
                    "locked_balance|hook_or_callback|balance_event_mismatch|"
                    "supply_mutation|other|uncertain"
                ),
                "origin_status": (
                    "token_contract_semantics|protocol_internal_accounting|"
                    "mixed|uncertain"
                ),
                "nominal_transfer_or_supply_signal": "",
                "actual_balance_or_state_signal": "",
                "affected_addresses": [],
                "state_evidence_ids": [],
                "transfer_event_evidence_ids": [],
                "evidence_ids": [],
                "confidence": "low|medium|high",
            }],
            "selected_candidate_ids": [],
            "binding_confidence": "low|medium|high",
            "unresolved_fields": [],
        }
    }

    reliance.depends_on = _dedupe_step_ids([*reliance.depends_on, anchor.id])
    reliance.consumes_state_keys = _dedupe_step_ids([
        *reliance.consumes_state_keys,
        TOKEN_SEMANTIC_CANDIDATE_STATE_KEY,
    ])
    reliance.produces_state_key = "token_semantic_reliance_summary"
    reliance.state_prompt_role = "ts_semantic_reliance"
    reliance.state_output_schema = {
        "token_semantic_reliance_summary": {
            "candidate_assessments": [{
                "candidate_id": "",
                "protocol_component": "",
                "relied_operation": "",
                "reliance_status": "supported|absent|uncertain",
                "same_candidate_supported": False,
                "token_origin_supported": False,
                "reliance_evidence": "",
                "evidence_ids": [],
            }],
            "satisfying_candidate_ids": [],
            "unresolved_candidate_ids": [],
        }
    }

    outcome.depends_on = _dedupe_step_ids([
        *outcome.depends_on,
        anchor.id,
        reliance.id,
    ])
    outcome.consumes_state_keys = _dedupe_step_ids([
        *outcome.consumes_state_keys,
        TOKEN_SEMANTIC_CANDIDATE_STATE_KEY,
        "token_semantic_reliance_summary",
    ])
    outcome.produces_state_key = "token_semantic_outcome_summary"
    outcome.state_prompt_role = "ts_semantic_outcome"
    outcome.state_output_schema = {
        "token_semantic_outcome_summary": {
            "candidate_assessments": [{
                "candidate_id": "",
                "attacker_or_beneficiary": "",
                "outcome_type": (
                    "excess_assets|unbacked_credit|bad_debt|price_distortion|"
                    "protocol_loss|unauthorized_payout|uncertain"
                ),
                "outcome_status": "supported|absent|uncertain",
                "causal_link_supported": False,
                "same_candidate_supported": False,
                "evidence_ids": [],
            }],
            "attack_candidate_ids": [],
            "unresolved_candidate_ids": [],
        }
    }

    exclusion_condition_ids = {
        str(condition.id or "").strip()
        for condition in list(rule.exclusion_conditions or [])
        if str(condition.id or "").strip()
    }
    exclusion_steps = [
        step
        for step in list(plan.judge_steps or [])
        if str(step.condition_id or step.id or "").strip()
        in exclusion_condition_ids
    ]
    exclusion_state_keys: List[str] = []
    for exclusion in exclusion_steps:
        state_key = f"token_semantic_exclusion_summary__{exclusion.id}"
        exclusion.depends_on = _dedupe_step_ids([
            *exclusion.depends_on,
            anchor.id,
            reliance.id,
            outcome.id,
        ])
        exclusion.consumes_state_keys = _dedupe_step_ids([
            *exclusion.consumes_state_keys,
            TOKEN_SEMANTIC_CANDIDATE_STATE_KEY,
            "token_semantic_reliance_summary",
            "token_semantic_outcome_summary",
        ])
        exclusion.produces_state_key = state_key
        exclusion.state_prompt_role = "ts_candidate_exclusion"
        exclusion.state_output_schema = {
            state_key: {
                "candidate_assessments": [{
                    "candidate_id": "",
                    "exclusion_status": "excluded|not_excluded|uncertain",
                    "reason": "",
                    "same_candidate_supported": False,
                    "evidence_ids": [],
                }],
                "excluded_candidate_ids": [],
                "unresolved_candidate_ids": [],
            }
        }
        exclusion_state_keys.append(state_key)

    original_step_order = [step.id for step in list(plan.judge_steps or [])]
    selected_ids = {anchor.id, reliance.id, outcome.id}
    remaining_steps = [
        step for step in list(plan.judge_steps or []) if step.id not in selected_ids
    ]
    plan.judge_steps = [anchor, reliance, outcome, *remaining_steps]
    dependency_edges = {
        reliance.id: [anchor.id],
        outcome.id: [anchor.id, reliance.id],
        **{
            step.id: [anchor.id, reliance.id, outcome.id]
            for step in exclusion_steps
        },
    }
    role_steps = {
        "token_semantic_anchor": anchor.id,
        "semantic_reliance": reliance.id,
        "semantic_outcome": outcome.id,
    }
    policy.update({
        "effective_mode": "stateful",
        "role_steps": role_steps,
        "role_score_matrix": score_matrix,
    })
    plan.metadata["stateful_runtime"] = {
        "enabled": True,
        "mode": TOKEN_SEMANTIC_STATEFUL_RUNTIME_MODE,
        "core_steps": [anchor.id, reliance.id, outcome.id],
        "role_steps": role_steps,
        "candidate_state_key": TOKEN_SEMANTIC_CANDIDATE_STATE_KEY,
        "dependency_edges": dependency_edges,
        "candidate_branching": True,
        "max_candidate_chains": 3,
        "candidate_outcome_state_key": "token_semantic_outcome_summary",
        "candidate_exclusion_state_keys": exclusion_state_keys,
        "original_step_order": original_step_order,
        "description": (
            "The token semantic anchor produces up to three token-mechanism "
            "candidates; reliance/outcome/exclusion steps assess the same "
            "candidate_id so protocol accounting signals are not confused with "
            "token-contract semantic divergences."
        ),
    }
    return plan


def _clear_token_semantic_stateful_runtime(plan: EvidencePlan) -> None:
    metadata = dict(plan.metadata or {})
    stateful = dict(metadata.get("stateful_runtime") or {})
    if str(stateful.get("mode") or "") != TOKEN_SEMANTIC_STATEFUL_RUNTIME_MODE:
        return
    dependency_edges = dict(stateful.get("dependency_edges") or {})
    for step in list(plan.judge_steps or []):
        if str(step.state_prompt_role or "").startswith("ts_"):
            step.consumes_state_keys = []
            step.produces_state_key = ""
            step.state_prompt_role = ""
            step.state_output_schema = {}
        added = set(dependency_edges.get(step.id) or [])
        if added:
            step.depends_on = [
                dependency for dependency in step.depends_on
                if dependency not in added
            ]
    metadata.pop("stateful_runtime", None)
    plan.metadata = metadata
    original_order = list(stateful.get("original_step_order") or [])
    if original_order:
        order_index = {
            step_id: index for index, step_id in enumerate(original_order)
        }
        plan.judge_steps = sorted(
            list(plan.judge_steps or []),
            key=lambda step: order_index.get(step.id, len(order_index)),
        )


def _infer_token_semantic_stateful_roles(
    steps: List[JudgeStep],
    *,
    condition_text_by_id: Dict[str, str] | None = None,
) -> tuple[Dict[str, JudgeStep], Dict[str, Dict[str, int]]]:
    roles = ("token_semantic_anchor", "semantic_reliance", "semantic_outcome")
    score_matrix: Dict[str, Dict[str, int]] = {}
    condition_text_by_id = dict(condition_text_by_id or {})
    for step in steps:
        condition_id = str(step.condition_id or step.id or "").strip()
        text = " ".join([
            str(step.question or ""),
            str(condition_text_by_id.get(condition_id) or ""),
        ]).lower()
        score_matrix[step.id] = {
            role: _token_semantic_role_score(text, role)
            for role in roles
        }
    if len(steps) < len(roles):
        return {}, score_matrix

    best: tuple[int, JudgeStep, JudgeStep, JudgeStep] | None = None
    for anchor in steps:
        for reliance in steps:
            if reliance.id == anchor.id:
                continue
            for outcome in steps:
                if outcome.id in {anchor.id, reliance.id}:
                    continue
                score = (
                    score_matrix[anchor.id]["token_semantic_anchor"]
                    + score_matrix[reliance.id]["semantic_reliance"]
                    + score_matrix[outcome.id]["semantic_outcome"]
                )
                candidate = (score, anchor, reliance, outcome)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        return {}, score_matrix
    selected_scores = (
        score_matrix[best[1].id]["token_semantic_anchor"],
        score_matrix[best[2].id]["semantic_reliance"],
        score_matrix[best[3].id]["semantic_outcome"],
    )
    if any(score <= 0 for score in selected_scores):
        return {}, score_matrix
    return {
        "token_semantic_anchor": best[1],
        "semantic_reliance": best[2],
        "semantic_outcome": best[3],
    }, score_matrix


def _ensure_token_semantic_binding_followup_views(
    anchor: JudgeStep,
    reliance: JudgeStep,
    outcome: JudgeStep,
) -> None:
    anchor.default_evidence_refs = _dedupe_known_views([
        "operation_summary_view",
        "evidence_adequacy_view",
        "token_semantic_delta_summary_view",
        "token_accounting_origin_view",
        *list(anchor.default_evidence_refs or anchor.evidence_refs or []),
    ])
    anchor.evidence_refs = list(anchor.default_evidence_refs)
    anchor.allowed_followup_views = _dedupe_known_views([
        *list(anchor.allowed_followup_views or []),
        "semantic_state_delta_view",
        "transfer_event_view",
        "state_change_view",
        "event_view",
        "token_accounting_origin_view",
    ])
    anchor.allowed_tools = _dedupe_step_ids([
        *list(anchor.allowed_tools or []),
        "read_packet_view",
        "read_evidence_context",
        "read_evidence_by_id",
    ])
    anchor.max_followups = max(1, int(anchor.max_followups or 0))

    reliance.default_evidence_refs = _dedupe_known_views([
        "operation_summary_view",
        "evidence_adequacy_view",
        "token_accounting_origin_view",
        "critical_call_view",
        *list(reliance.default_evidence_refs or reliance.evidence_refs or []),
    ])
    reliance.evidence_refs = list(reliance.default_evidence_refs)
    reliance.allowed_followup_views = _dedupe_known_views([
        *list(reliance.allowed_followup_views or []),
        "token_semantic_delta_summary_view",
        "semantic_state_delta_view",
        "transfer_event_view",
        "critical_call_view",
        "amm_reserve_transition_view",
        "price_relevant_state_view",
    ])
    reliance.allowed_tools = _dedupe_step_ids([
        *list(reliance.allowed_tools or []),
        "read_packet_view",
        "read_evidence_context",
        "get_local_call_context",
        "read_evidence_by_id",
    ])
    reliance.max_followups = max(1, int(reliance.max_followups or 0))

    outcome.default_evidence_refs = _dedupe_known_views([
        "operation_summary_view",
        "evidence_adequacy_view",
        "token_semantic_delta_summary_view",
        "value_release_view",
        *list(outcome.default_evidence_refs or outcome.evidence_refs or []),
    ])
    outcome.evidence_refs = list(outcome.default_evidence_refs)
    outcome.allowed_followup_views = _dedupe_known_views([
        *list(outcome.allowed_followup_views or []),
        "token_accounting_origin_view",
        "participant_net_delta_view",
        "contribution_vs_payout_view",
        "value_release_view",
        "external_fundflow_view",
        "semantic_state_delta_view",
    ])
    outcome.allowed_tools = _dedupe_step_ids([
        *list(outcome.allowed_tools or []),
        "read_packet_view",
        "read_evidence_context",
        "get_local_call_context",
        "read_evidence_by_id",
    ])
    outcome.max_followups = max(1, int(outcome.max_followups or 0))


def _token_semantic_role_score(text: str, role: str) -> int:
    patterns: Dict[str, tuple[tuple[str, int], ...]] = {
        "token_semantic_anchor": (
            ("token semantic", 12),
            ("semantic exploit", 10),
            ("token behavior", 9),
            ("token mechanics", 9),
            ("fee-on-transfer", 10),
            ("fee on transfer", 10),
            ("deflationary", 9),
            ("reflection", 9),
            ("rebase", 9),
            ("elastic supply", 8),
            ("balance/event mismatch", 9),
            ("balance mismatch", 8),
            ("nominal transfer", 7),
            ("actual received", 7),
            ("actual balance", 7),
            ("transfer event", 5),
            ("balanceof", 5),
            ("mint/burn", 6),
            ("mint", 3),
            ("burn", 3),
            ("total supply", 5),
            ("hook", 5),
            ("locked balance", 6),
            ("token contract", 5),
            ("origin", 4),
        ),
        "semantic_reliance": (
            ("relies on", 12),
            ("rely on", 12),
            ("consumes", 9),
            ("uses the token", 8),
            ("protocol accounting", 8),
            ("internal accounting", 8),
            ("credits", 7),
            ("shares", 7),
            ("collateral", 7),
            ("debt", 7),
            ("reserve", 6),
            ("liquidity", 6),
            ("swap output", 7),
            ("pricing", 5),
            ("valuation", 5),
            ("balanceof", 5),
            ("reconciles", 8),
            ("does not reconcile", 10),
            ("assumes", 7),
            ("same token", 5),
        ),
        "semantic_outcome": (
            ("attacker-favorable", 12),
            ("attacker favorable", 12),
            ("profit", 6),
            ("protocol loss", 9),
            ("excess", 8),
            ("unbacked", 8),
            ("bad debt", 8),
            ("drain", 8),
            ("extract", 8),
            ("payout", 6),
            ("value release", 7),
            ("outcome", 5),
            ("causal", 8),
            ("causes", 8),
            ("resulting", 4),
            ("loss", 4),
            ("beneficiary", 4),
        ),
    }
    return sum(
        weight for phrase, weight in patterns.get(role, ()) if phrase in text
    )


def apply_insufficient_validation_stateful_runtime(
    plan: EvidencePlan,
    *,
    attack_label: str = "",
    preserve_view_policy: bool = False,
    preserve_question_policy: bool = False,
) -> EvidencePlan:
    """Annotate insufficient_validation plans with object-binding state flow."""
    if _normalized_attack_label(attack_label) != "insufficient_validation":
        return plan

    by_condition: Dict[str, JudgeStep] = {
        str(step.condition_id or step.id).strip().upper(): step
        for step in list(plan.judge_steps or [])
    }
    c1 = by_condition.get("C1")
    c2 = by_condition.get("C2")
    c3 = by_condition.get("C3")

    plan.metadata["stateful_runtime"] = {
        "enabled": True,
        "mode": IV_STATEFUL_RUNTIME_MODE,
        "core_steps": [
            step.id for step in (c1, c2, c3) if step is not None
        ],
        "description": (
            "C1 produces object candidates; C2 selects object and validation gap; "
            "C3 binds outcome to selected object/gap."
        ),
    }

    if c1 is not None:
        if not preserve_question_policy:
            _append_question_note_once(c1, IV_OBJECT_BINDING_QUESTION_NOTE)
        if not preserve_view_policy:
            c1.default_evidence_refs = _dedupe_known_views([
                *(c1.default_evidence_refs or c1.evidence_refs or []),
                "critical_call_argument_view",
                "semantic_state_delta_view",
            ])
            c1.evidence_refs = list(c1.default_evidence_refs)
            c1.allowed_followup_views = _dedupe_known_views([
                *(c1.allowed_followup_views or []),
                "participant_net_delta_view",
                "contribution_vs_payout_view",
                "semantic_state_delta_view",
            ])
        c1.produces_state_key = "object_binding_summary"
        c1.state_prompt_role = "iv_object_binding"
        c1.state_output_schema = {
            "object_binding_summary": {
                "object_candidates": [
                    {
                        "object_id": "",
                        "object_name": "",
                        "object_type": (
                            "user_parameter|external_return|callback_data|"
                            "oracle_or_external_state|business_accounting_state|"
                            "reward_share_debt_or_entitlement_state|"
                            "computed_intermediate|uncertain"
                        ),
                        "provenance": "",
                        "consumer_path": "",
                        "required_validation_dimensions": [],
                        "validation_observation": "",
                        "evidence_ids": [],
                    }
                ],
                "candidate_coverage": {
                    "candidate_count": 0,
                    "candidate_types_considered": [],
                    "sensitive_paths_considered": [],
                    "coverage_limitations": [],
                },
                "binding_confidence": "low|medium|high",
            }
        }
    if c2 is not None:
        if not preserve_question_policy:
            _append_question_note_once(c2, IV_VALIDATION_GAP_QUESTION_NOTE)
        if not preserve_view_policy:
            c2.default_evidence_refs = _dedupe_known_views([
                *(c2.default_evidence_refs or c2.evidence_refs or []),
                "critical_call_argument_view",
                "contribution_vs_payout_view",
            ])
            c2.evidence_refs = list(c2.default_evidence_refs)
            c2.allowed_followup_views = _dedupe_known_views([
                *(c2.allowed_followup_views or []),
                "value_release_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
            ])
        c2.depends_on = _dedupe_step_ids([
            *(c2.depends_on or []),
            *([c1.id] if c1 is not None else []),
        ])
        c2.consumes_state_keys = _dedupe_step_ids([
            *(c2.consumes_state_keys or []),
            "object_binding_summary",
        ])
        c2.produces_state_key = "validation_gap_summary"
        c2.state_prompt_role = "iv_validation_gap"
        c2.state_output_schema = {
            "validation_gap_summary": {
                "selected_object_id": "",
                "selected_object_name": "",
                "validation_status": (
                    "missing|inadequate|bypassed|stale|wrong_scope|present|"
                    "standard_design|guarded|uncertain"
                ),
                "validation_gap": "",
                "why_selected": "",
                "why_other_objects_not_selected": "",
                "candidate_assessments": [
                    {
                        "object_id": "",
                        "validation_status": (
                            "missing|inadequate|bypassed|stale|wrong_scope|present|"
                            "standard_design|guarded|uncertain"
                        ),
                        "required_validation_dimensions": [],
                        "observed_check_scope": [],
                        "scope_match": "exact|partial|wrong_scope|unknown",
                        "validation_gap": "",
                        "compensating_guards_considered": [],
                        "selection_rationale": "",
                        "evidence_ids": [],
                    }
                ],
                "evidence_ids": [],
            }
        }
    if c3 is not None:
        c3.depends_on = _dedupe_step_ids([
            *(c3.depends_on or []),
            *([c1.id] if c1 is not None else []),
            *([c2.id] if c2 is not None else []),
        ])
        c3.consumes_state_keys = _dedupe_step_ids([
            *(c3.consumes_state_keys or []),
            "object_binding_summary",
            "validation_gap_summary",
        ])
        c3.produces_state_key = "causal_outcome_summary"
        c3.state_prompt_role = "iv_causal_outcome"
        c3.state_output_schema = {
            "causal_outcome_summary": {
                "selected_object_id": "",
                "selected_validation_gap": "",
                "causal_link_supported": False,
                "outcome_type": (
                    "excess_claim|incorrect_mint|bad_debt|loss|"
                    "state_invariant_violation|unauthorized_payout|"
                    "accounting_break|uncertain"
                ),
                "evidence_ids": [],
            }
        }
    for step in list(plan.judge_steps or []):
        role = _infer_insufficient_validation_condition_role(
            str(step.condition_id or step.id),
            str(step.question or ""),
        )
        if role != "access_control_exclusion":
            continue
        if not preserve_view_policy:
            step.default_evidence_refs = _dedupe_known_views([
                *(step.default_evidence_refs or step.evidence_refs or []),
                "critical_call_argument_view",
                "source_unavailable_auth_view",
                "beneficiary_controller_view",
            ])
            step.evidence_refs = list(step.default_evidence_refs)
            step.allowed_followup_views = _dedupe_known_views([
                *(step.allowed_followup_views or []),
                "trace_outline_view",
                "semantic_state_delta_view",
                "state_change_view",
                "value_release_view",
            ])
        step.depends_on = _dedupe_step_ids([
            *(step.depends_on or []),
            *([c1.id] if c1 is not None else []),
            *([c2.id] if c2 is not None else []),
            *([c3.id] if c3 is not None else []),
        ])
        step.consumes_state_keys = _dedupe_step_ids([
            *(step.consumes_state_keys or []),
            "object_binding_summary",
            "validation_gap_summary",
            "causal_outcome_summary",
        ])
        step.state_prompt_role = "iv_access_control_exclusion_with_core_state"
        step.state_output_schema = {
            "access_control_exclusion_summary": {
                "selected_object_id": "",
                "challenged_selected_object": False,
                "primary_mechanism_candidate_id": "",
                "competing_candidate_assessments": [],
                "authorization_subject": "",
                "required_authority": "",
                "protected_capability": "",
                "guard_scope": "",
                "guard_status": (
                    "missing|bypassed|self_granted|attacker_controlled|"
                    "wrong_scope|effective|object_validation_only|uncertain"
                ),
                "primary_mechanism": "access_control|object_validation|uncertain",
                "evidence_ids": [],
            }
        }
    return plan


def apply_protocol_accounting_stateful_runtime(
    plan: EvidencePlan,
    *,
    rule: EvolvingRule,
) -> EvidencePlan:
    """Bind Protocol Accounting core steps to one bookkeeping candidate chain."""
    label = _normalized_attack_label((rule.metadata or {}).get("attack_label", ""))
    if label != "protocol_accounting_exploitation":
        return plan
    by_id = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list(plan.judge_steps or [])
    }
    c1 = by_id.get("C1")
    c2 = by_id.get("C2")
    c3 = by_id.get("C3")
    if c1 is None or c2 is None or c3 is None:
        return plan

    metadata = dict(plan.metadata or {})
    metadata["protocol_accounting_stateful_runtime"] = {
        "mode": PROTOCOL_ACCOUNTING_STATEFUL_RUNTIME_MODE,
        "description": (
            "C1 selects a protocol bookkeeping candidate; C2 verifies that a "
            "protocol operation consumes the same candidate; C3 verifies the "
            "same candidate-specific accounting outcome."
        ),
    }
    plan.metadata = metadata

    c1.produces_state_key = PROTOCOL_ACCOUNTING_CANDIDATE_STATE_KEY
    c1.state_prompt_role = "pa_accounting_anchor"
    c1.state_output_schema = {
        PROTOCOL_ACCOUNTING_CANDIDATE_STATE_KEY: {
            "candidate_id": "",
            "domain": (
                "reward|share|vault|debt|collateral|staking|claim|strategy|"
                "eligibility|exchange_rate|unknown"
            ),
            "protocol_component": "",
            "accounting_state": "",
            "operation_scope": "",
            "candidate_status": "supported|uncertain|absent",
            "evidence_ids": [],
        }
    }

    c2.depends_on = _dedupe_step_ids([*(c2.depends_on or []), c1.id])
    c2.consumes_state_keys = _dedupe_step_ids([
        *(c2.consumes_state_keys or []),
        PROTOCOL_ACCOUNTING_CANDIDATE_STATE_KEY,
    ])
    c2.produces_state_key = "protocol_accounting_consumption_summary"
    c2.state_prompt_role = "pa_accounting_consumption"
    c2.state_output_schema = {
        "protocol_accounting_consumption_summary": {
            "candidate_id": "",
            "consumption_status": "supported|uncertain|absent",
            "action_type": (
                "claim|withdraw|borrow|mint|redeem|stake|unstake|liquidate|"
                "strategy_action|unknown"
            ),
            "same_candidate_supported": False,
            "constraint_or_entitlement_bypass": "supported|uncertain|absent",
            "evidence_ids": [],
        }
    }

    c3.depends_on = _dedupe_step_ids([*(c3.depends_on or []), c1.id, c2.id])
    c3.consumes_state_keys = _dedupe_step_ids([
        *(c3.consumes_state_keys or []),
        PROTOCOL_ACCOUNTING_CANDIDATE_STATE_KEY,
        "protocol_accounting_consumption_summary",
    ])
    c3.produces_state_key = "protocol_accounting_outcome_summary"
    c3.state_prompt_role = "pa_accounting_outcome"
    c3.state_output_schema = {
        "protocol_accounting_outcome_summary": {
            "candidate_id": "",
            "outcome_status": "supported|uncertain|absent",
            "outcome_type": (
                "excess_reward|inflated_share|undercollateralized_borrow|"
                "incorrect_withdrawal|bad_debt|abnormal_vault_delta|"
                "protocol_loss|unknown"
            ),
            "same_candidate_supported": False,
            "disproportionate_to_contribution": "supported|uncertain|absent",
            "evidence_ids": [],
        }
    }

    for step in list(plan.judge_steps or []):
        cid = str(step.condition_id or step.id or "").strip().upper()
        if not cid.startswith("E"):
            continue
        step.depends_on = _dedupe_step_ids([
            *(step.depends_on or []),
            c1.id,
            c2.id,
            c3.id,
        ])
        step.consumes_state_keys = _dedupe_step_ids([
            *(step.consumes_state_keys or []),
            PROTOCOL_ACCOUNTING_CANDIDATE_STATE_KEY,
            "protocol_accounting_consumption_summary",
            "protocol_accounting_outcome_summary",
        ])
        state_key = f"{cid.lower()}_protocol_accounting_exclusion_summary"
        step.produces_state_key = state_key
        step.state_prompt_role = "pa_candidate_exclusion"
        step.state_output_schema = {
            state_key: {
                "candidate_id": "",
                "exclusion_status": "excluded|not_excluded|uncertain",
                "replaced_by": (
                    "market_or_oracle|token_semantics|access_control|"
                    "insufficient_validation|reentrancy|flash_capital|"
                    "normal_accounting|unknown"
                ),
                "same_candidate_supported": False,
                "evidence_ids": [],
            }
        }
    return plan


MARKET_MANIPULATION_STATEFUL_RUNTIME_MODE = "market_mechanism_profile_binding_v1"
MARKET_MANIPULATION_CANDIDATE_STATE_KEY = "market_mechanism_candidate"


def apply_market_manipulation_stateful_runtime(
    plan: EvidencePlan,
    *,
    rule: EvolvingRule,
) -> EvidencePlan:
    """Bind Market Manipulation core steps to one selected mechanism profile.

    This is prompt/state passing only; emit logic remains the rule's C1/C2/C3
    expression. The goal is to stop C1, C2, and C3 from silently switching
    between AMM/oracle, skim/donate, sandwich/MEV, and generic profit signals.
    """
    label = _normalized_attack_label((rule.metadata or {}).get("attack_label", ""))
    if label != "market_manipulation":
        return plan
    by_id = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list(plan.judge_steps or [])
    }
    c1 = by_id.get("C1")
    c2 = by_id.get("C2")
    c3 = by_id.get("C3")
    if c1 is None or c2 is None or c3 is None:
        return plan
    metadata = dict(plan.metadata or {})
    metadata["market_manipulation_stateful_runtime"] = {
        "mode": MARKET_MANIPULATION_STATEFUL_RUNTIME_MODE,
        "description": (
            "C1 selects a market-mechanism profile; C2 verifies consumption "
            "of the same profile; C3 verifies profile-specific outcome."
        ),
    }
    plan.metadata = metadata

    c1.produces_state_key = MARKET_MANIPULATION_CANDIDATE_STATE_KEY
    c1.state_prompt_role = "mm_market_mechanism_anchor"
    c1.state_output_schema = {
        MARKET_MANIPULATION_CANDIDATE_STATE_KEY: {
            "profile_id": "",
            "profile_type": (
                "price_state|pool_accounting_reserve_extraction|"
                "ordering_mev_arbitrage|uncertain"
            ),
            "market_source": "",
            "pool_or_pair": "",
            "token_side": "",
            "manipulation_window": "",
            "candidate_status": "supported|uncertain|absent",
            "evidence_ids": [],
        }
    }

    c2.depends_on = _dedupe_step_ids([*(c2.depends_on or []), c1.id])
    c2.consumes_state_keys = _dedupe_step_ids([
        *(c2.consumes_state_keys or []),
        MARKET_MANIPULATION_CANDIDATE_STATE_KEY,
    ])
    c2.produces_state_key = "market_consumption_summary"
    c2.state_prompt_role = "mm_market_consumption"
    c2.state_output_schema = {
        "market_consumption_summary": {
            "profile_id": "",
            "consumption_status": "supported|uncertain|absent",
            "consumer_operation": "",
            "consumer_scope": "same_pool_internal|external_downstream|ordering|uncertain",
            "same_profile_supported": False,
            "evidence_ids": [],
        }
    }

    c3.depends_on = _dedupe_step_ids([*(c3.depends_on or []), c1.id, c2.id])
    c3.consumes_state_keys = _dedupe_step_ids([
        *(c3.consumes_state_keys or []),
        MARKET_MANIPULATION_CANDIDATE_STATE_KEY,
        "market_consumption_summary",
    ])
    c3.produces_state_key = "market_outcome_summary"
    c3.state_prompt_role = "mm_market_outcome"
    c3.state_output_schema = {
        "market_outcome_summary": {
            "profile_id": "",
            "outcome_status": "supported|uncertain|absent",
            "outcome_type": (
                "forced_slippage|reserve_extraction|distorted_settlement|"
                "oracle_valuation_effect|ordering_extraction|accounting_effect|uncertain"
            ),
            "same_profile_supported": False,
            "evidence_ids": [],
        }
    }

    for step in list(plan.judge_steps or []):
        cid = str(step.condition_id or step.id or "").strip().upper()
        if not cid.startswith("E"):
            continue
        step.depends_on = _dedupe_step_ids([
            *(step.depends_on or []),
            c1.id,
            c2.id,
            c3.id,
        ])
        step.consumes_state_keys = _dedupe_step_ids([
            *(step.consumes_state_keys or []),
            MARKET_MANIPULATION_CANDIDATE_STATE_KEY,
            "market_consumption_summary",
            "market_outcome_summary",
        ])
        state_key = f"{cid.lower()}_market_exclusion_summary"
        step.produces_state_key = state_key
        step.state_prompt_role = "mm_candidate_exclusion"
        step.state_output_schema = {
            state_key: {
                "profile_id": "",
                "exclusion_status": "excluded|not_excluded|uncertain",
                "same_profile_supported": False,
                "evidence_ids": [],
            }
        }
    return plan


FLASHLOANS_STATEFUL_RUNTIME_MODE = "flash_capital_chain_binding_v1"
FLASHLOANS_CANDIDATE_STATE_KEY = "flash_capital_candidate"


def apply_flashloans_stateful_runtime(
    plan: EvidencePlan,
    *,
    rule: EvolvingRule,
) -> EvidencePlan:
    """Bind Flashloans core steps to one temporary-capital candidate chain."""
    label = _normalized_attack_label((rule.metadata or {}).get("attack_label", ""))
    if label != "flashloans":
        return plan
    by_id = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list(plan.judge_steps or [])
    }
    c1 = by_id.get("C1")
    c2 = by_id.get("C2")
    c3 = by_id.get("C3")
    if c1 is None or c2 is None or c3 is None:
        return plan
    metadata = dict(plan.metadata or {})
    metadata["flashloans_stateful_runtime"] = {
        "mode": FLASHLOANS_STATEFUL_RUNTIME_MODE,
        "description": (
            "C1 selects a temporary atomic-capital candidate; C2 verifies "
            "that the same candidate materially enables a flash-assisted exploit "
            "mechanism; C3 verifies "
            "that the outcome causally depends on that same chain."
        ),
    }
    plan.metadata = metadata

    c1.produces_state_key = FLASHLOANS_CANDIDATE_STATE_KEY
    c1.state_prompt_role = "fl_flash_capital_anchor"
    c1.state_output_schema = {
        FLASHLOANS_CANDIDATE_STATE_KEY: {
            "candidate_id": "",
            "capital_type": "flash_loan|flash_swap|flash_mint|borrow_repay|atomic_capital|uncertain",
            "provider_or_source": "",
            "borrower_or_callback": "",
            "asset": "",
            "amount_or_scale": "",
            "repayment_or_settlement": "supported|uncertain|absent",
            "callback_or_execution_scope": "",
            "call_burst_signal": "supported|uncertain|absent",
            "call_count_or_scale": "",
            "atomic_sequence_evidence": "",
            "candidate_status": "supported|uncertain|absent",
            "evidence_ids": [],
        }
    }

    c2.depends_on = _dedupe_step_ids([*(c2.depends_on or []), c1.id])
    c2.consumes_state_keys = _dedupe_step_ids([
        *(c2.consumes_state_keys or []),
        FLASHLOANS_CANDIDATE_STATE_KEY,
    ])
    c2.produces_state_key = "flash_exploit_mechanism_summary"
    c2.state_prompt_role = "fl_exploit_mechanism"
    c2.state_output_schema = {
        "flash_exploit_mechanism_summary": {
            "candidate_id": "",
            "mechanism_status": "supported|uncertain|absent",
            "mechanism_type": (
                "validation_bypass|accounting_amplification|oracle_or_price_distortion|"
                "solvency_or_collateral_break|callback_or_repayment_check_abuse|"
                "temporary_power|other_protocol_sensitive_effect|incidental_capital|uncertain"
            ),
            "same_candidate_supported": False,
            "capital_dependency_status": "material_enabler|incidental|uncertain|absent",
            "call_burst_support": "supported|uncertain|absent",
            "flash_assisted_chain_supported": False,
            "flash_as_weapon_supported": False,
            "incidental_capital_risk": "low|medium|high",
            "evidence_ids": [],
        }
    }

    c3.depends_on = _dedupe_step_ids([*(c3.depends_on or []), c1.id, c2.id])
    c3.consumes_state_keys = _dedupe_step_ids([
        *(c3.consumes_state_keys or []),
        FLASHLOANS_CANDIDATE_STATE_KEY,
        "flash_exploit_mechanism_summary",
    ])
    c3.produces_state_key = "flash_outcome_summary"
    c3.state_prompt_role = "fl_flash_outcome"
    c3.state_output_schema = {
        "flash_outcome_summary": {
            "candidate_id": "",
            "outcome_status": "supported|uncertain|absent",
            "outcome_type": (
                "attacker_value_gain|protocol_or_user_loss|bad_debt|excess_claim|"
                "distorted_settlement|attacker_favorable_state|uncertain"
            ),
            "same_candidate_supported": False,
            "same_mechanism_supported": False,
            "flash_assisted_outcome_supported": False,
            "causal_link_supported": False,
            "evidence_ids": [],
        }
    }

    for step in list(plan.judge_steps or []):
        cid = str(step.condition_id or step.id or "").strip().upper()
        if not cid.startswith("E"):
            continue
        step.depends_on = _dedupe_step_ids([
            *(step.depends_on or []),
            c1.id,
            c2.id,
            c3.id,
        ])
        step.consumes_state_keys = _dedupe_step_ids([
            *(step.consumes_state_keys or []),
            FLASHLOANS_CANDIDATE_STATE_KEY,
            "flash_exploit_mechanism_summary",
            "flash_outcome_summary",
        ])
        state_key = f"{cid.lower()}_flash_exclusion_summary"
        step.produces_state_key = state_key
        step.state_prompt_role = "fl_candidate_exclusion"
        step.state_output_schema = {
            state_key: {
                "candidate_id": "",
                "exclusion_status": "excluded|not_excluded|uncertain",
                "exclusion_type": (
                    "ordinary_arbitrage|ordinary_liquidation|incidental_capital|"
                    "other_primary_mechanism|capital_backed_execution|uncertain"
                ),
                "same_candidate_supported": False,
                "evidence_ids": [],
            }
        }
    return plan


REENTRANCY_STATEFUL_RUNTIME_MODE = "reentrancy_candidate_binding_v1"
REENTRANCY_CANDIDATE_STATE_KEY = "reentrancy_candidate"
REENTRANCY_BINDING_MODES = {"disabled", "soft", "stateful"}
REENTRANCY_STATE_SCHEMA_VERSION = "reentrancy_candidate_state_v2"


def normalize_reentrancy_binding_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": "stateful",
        "prompt": "soft",
        "prompt_only": "soft",
        "strong": "stateful",
        "off": "disabled",
        "none": "disabled",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in REENTRANCY_BINDING_MODES else "stateful"


def _clear_reentrancy_stateful_runtime(plan: EvidencePlan) -> None:
    metadata = dict(plan.metadata or {})
    prior = dict(metadata.get("stateful_runtime") or {})
    prior_is_reentrancy = (
        str(prior.get("mode") or "") == REENTRANCY_STATEFUL_RUNTIME_MODE
    )
    role_steps = dict(prior.get("role_steps") or {}) if prior_is_reentrancy else {}
    core_ids = {
        str(value or "").strip()
        for value in [
            *list(prior.get("core_steps") or []),
            *list(role_steps.values()),
        ]
        if str(value or "").strip()
    }
    reentrancy_role_steps = [
        step for step in list(plan.judge_steps or [])
        if str(step.state_prompt_role or "").strip().startswith("re_")
    ]
    core_ids.update(
        str(step.id or "").strip()
        for step in reentrancy_role_steps
        if str(step.id or "").strip()
    )
    state_keys = {
        REENTRANCY_CANDIDATE_STATE_KEY,
        "reentrancy_value_effect_summary",
        "reentrancy_causal_order_summary",
        *[
            str(value or "").strip()
            for value in list(prior.get("candidate_exclusion_state_keys") or [])
            if str(value or "").strip()
        ],
        *[
            str(step.produces_state_key or "").strip()
            for step in reentrancy_role_steps
            if str(step.produces_state_key or "").strip()
        ],
    }
    for step in list(plan.judge_steps or []):
        role = str(step.state_prompt_role or "").strip()
        is_reentrancy_step = role.startswith("re_") or (
            prior_is_reentrancy and step.id in core_ids
        )
        if not is_reentrancy_step:
            continue
        step.depends_on = [
            value for value in list(step.depends_on or [])
            if value not in core_ids
        ]
        step.consumes_state_keys = [
            value for value in list(step.consumes_state_keys or [])
            if value not in state_keys
        ]
        if str(step.produces_state_key or "") in state_keys or role.startswith("re_"):
            step.produces_state_key = ""
            step.state_output_schema = {}
        step.state_prompt_role = ""
    if prior_is_reentrancy:
        metadata.pop("stateful_runtime", None)
    metadata.pop("reentrancy_binding_policy", None)
    plan.metadata = metadata


def apply_reentrancy_stateful_runtime(
    plan: EvidencePlan,
    *,
    rule: EvolvingRule,
    mode: str | None = None,
) -> EvidencePlan:
    """Bind Reentrancy core steps to one nested-entry candidate chain."""
    label = _normalized_attack_label((rule.metadata or {}).get("attack_label", ""))
    if label != "reentrancy":
        return plan
    requested_mode = normalize_reentrancy_binding_mode(
        mode
        if mode is not None
        else (rule.metadata or {}).get("reentrancy_binding_mode")
    )
    _clear_reentrancy_stateful_runtime(plan)
    managed_condition_ids = {
        str(condition.id or "").strip().upper()
        for condition in [
            *list(rule.conditions or []),
            *list(rule.exclusion_conditions or []),
        ]
        if str(condition.id or "").strip()
    }
    normalized_generated_contracts: List[Dict[str, Any]] = []
    for step in list(plan.judge_steps or []):
        condition_id = str(step.condition_id or step.id or "").strip().upper()
        if condition_id not in managed_condition_ids:
            continue
        removed = {
            "depends_on": list(step.depends_on or []),
            "consumes_state_keys": list(step.consumes_state_keys or []),
            "produces_state_key": str(step.produces_state_key or ""),
            "state_prompt_role": str(step.state_prompt_role or ""),
            "state_output_schema": dict(step.state_output_schema or {}),
        }
        if any(bool(value) for value in removed.values()):
            normalized_generated_contracts.append({
                "condition_id": condition_id,
                "removed_generated_contract": removed,
            })
        # Reentrancy state contracts are runtime-owned. Rebuild them below so
        # an LLM-generated orphan key, future dependency, or incompatible
        # schema cannot survive beside the canonical candidate chain.
        step.depends_on = []
        step.consumes_state_keys = []
        step.produces_state_key = ""
        step.state_prompt_role = ""
        step.state_output_schema = {}
    metadata = dict(plan.metadata or {})
    metadata["reentrancy_binding_policy"] = {
        "requested_mode": requested_mode,
        "effective_mode": requested_mode,
        "state_schema_version": REENTRANCY_STATE_SCHEMA_VERSION,
        "role_steps": {},
    }
    if normalized_generated_contracts:
        metadata["reentrancy_generated_state_contract_normalization"] = {
            "applied": True,
            "reason": "runtime_owned_state_contract",
            "normalized_steps": normalized_generated_contracts,
        }
    plan.metadata = metadata
    if requested_mode == "disabled":
        return plan

    positive_condition_ids = {
        str(condition.id or "").strip().upper()
        for condition in list(rule.conditions or [])
        if str(condition.id or "").strip()
    }
    core_steps = [
        step
        for step in list(plan.judge_steps or [])
        if str(step.condition_id or step.id or "").strip().upper()
        in positive_condition_ids
    ]
    by_condition_id = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in core_steps
    }
    c1 = by_condition_id.get("C1")
    c2 = by_condition_id.get("C2")
    c3 = by_condition_id.get("C3")
    if c1 is None or c2 is None or c3 is None:
        if len(core_steps) < 3:
            return plan
        c1, c2, c3 = core_steps[:3]

    original_step_order = [step.id for step in list(plan.judge_steps or [])]
    dependency_edges: Dict[str, List[str]] = {}

    c1.produces_state_key = REENTRANCY_CANDIDATE_STATE_KEY
    c1.state_prompt_role = "re_reentry_anchor"
    c1.default_evidence_refs = _dedupe_step_ids([
        "operation_summary_view",
        "evidence_adequacy_view",
        "reentrancy_candidate_catalog_view",
        "critical_call_view",
        *[
            ref for ref in list(c1.default_evidence_refs or c1.evidence_refs or [])
            if ref not in {
                "reentrancy_state_order_summary_view",
                "trace_outline_view",
            }
        ],
    ])[:5]
    c1.evidence_refs = list(c1.default_evidence_refs)
    c1.allowed_followup_views = _dedupe_step_ids([
        "reentrancy_state_order_summary_view",
        "trace_outline_view",
        "critical_call_argument_view",
        "unknown_selector_view",
        "reentrancy_state_order_view",
        *(c1.allowed_followup_views or []),
    ])
    c1.state_output_schema = {
        REENTRANCY_CANDIDATE_STATE_KEY: {
            "candidates": [
                {
                    "candidate_id": "",
                    "outer_call_id": "",
                    "external_edge_id": "",
                    "reentry_call_id": "",
                    "callback_kind": "",
                    "logical_storage_context": "",
                    "outer_function": "",
                    "reentry_function": "",
                    "path_ids": [],
                    "evidence_ids": [],
                    "confidence": "low|medium|high",
                }
            ],
            "selected_candidate_ids": [],
            "unresolved_candidate_ids": [],
        }
    }

    c2.depends_on = _dedupe_step_ids([*(c2.depends_on or []), c1.id])
    dependency_edges[c2.id] = [c1.id]
    c2.consumes_state_keys = _dedupe_step_ids([
        *(c2.consumes_state_keys or []),
        REENTRANCY_CANDIDATE_STATE_KEY,
    ])
    c2.produces_state_key = "reentrancy_value_effect_summary"
    c2.state_prompt_role = "re_value_effect"
    c2.state_output_schema = {
        "reentrancy_value_effect_summary": {
            "candidate_results": [
                {
                    "candidate_id": "",
                    "effect_status": "supported|absent|uncertain",
                    "effect_type": "",
                    "same_path_supported": False,
                    "value_or_state_effect": "",
                    "evidence_ids": [],
                }
            ],
            "satisfying_candidate_ids": [],
            "unresolved_candidate_ids": [],
        }
    }

    c3.depends_on = _dedupe_step_ids([*(c3.depends_on or []), c1.id, c2.id])
    dependency_edges[c3.id] = [c1.id, c2.id]
    c3.consumes_state_keys = _dedupe_step_ids([
        *(c3.consumes_state_keys or []),
        REENTRANCY_CANDIDATE_STATE_KEY,
        "reentrancy_value_effect_summary",
    ])
    c3.produces_state_key = "reentrancy_causal_order_summary"
    c3.state_prompt_role = "re_state_order_causality"
    c3.state_output_schema = {
        "reentrancy_causal_order_summary": {
            "candidate_results": [
                {
                    "candidate_id": "",
                    "causality_status": "supported|absent|uncertain",
                    "mechanism_type": (
                        "delayed_protective_update|repeated_sensitive_effect|"
                        "cross_function_shared_accounting|read_only_stale_observation|"
                        "other|uncertain"
                    ),
                    "phase_order_pattern": "",
                    "state_not_finalized_before_external_edge": False,
                    "inner_consumption_or_effect_supported": False,
                    "outer_protective_write_after_callback": False,
                    "repeated_sensitive_effect_before_return": False,
                    "cross_function_shared_accounting_supported": False,
                    "read_only_stale_observation_supported": False,
                    "same_candidate_supported": False,
                    "evidence_ids": [],
                }
            ],
            "attack_candidate_ids": [],
            "safe_order_candidate_ids": [],
            "unresolved_candidate_ids": [],
        }
    }
    c3.max_followups = max(int(c3.max_followups or 0), 2)
    if "read_function_chunk" not in c3.allowed_tools:
        c3.allowed_tools.append("read_function_chunk")

    exclusion_state_keys: List[str] = []
    for step in list(plan.judge_steps or []):
        cid = str(step.condition_id or step.id or "").strip().upper()
        if not cid.startswith("E"):
            continue
        step.depends_on = _dedupe_step_ids([
            *(step.depends_on or []),
            c1.id,
            c2.id,
            c3.id,
        ])
        dependency_edges[step.id] = [c1.id, c2.id, c3.id]
        step.consumes_state_keys = _dedupe_step_ids([
            *(step.consumes_state_keys or []),
            REENTRANCY_CANDIDATE_STATE_KEY,
            "reentrancy_value_effect_summary",
            "reentrancy_causal_order_summary",
        ])
        state_key = f"{cid.lower()}_reentrancy_exclusion_summary"
        exclusion_state_keys.append(state_key)
        step.produces_state_key = state_key
        step.state_prompt_role = "re_candidate_exclusion"
        step.state_output_schema = {
            state_key: {
                "candidate_results": [
                    {
                        "candidate_id": "",
                        "exclusion_status": "excluded|not_excluded|uncertain",
                        "same_candidate_supported": False,
                        "reason": "",
                        "evidence_ids": [],
                    }
                ],
                "excluded_candidate_ids": [],
                "unresolved_candidate_ids": [],
            }
        }

    metadata = dict(plan.metadata or {})
    metadata["reentrancy_binding_policy"] = {
        "requested_mode": requested_mode,
        "effective_mode": requested_mode,
        "state_schema_version": REENTRANCY_STATE_SCHEMA_VERSION,
        "role_steps": {
            "reentry_anchor": c1.id,
            "value_effect": c2.id,
            "state_order_causality": c3.id,
        },
    }
    metadata["stateful_runtime"] = {
        "enabled": True,
        "mode": REENTRANCY_STATEFUL_RUNTIME_MODE,
        "binding_mode": requested_mode,
        "state_schema_version": REENTRANCY_STATE_SCHEMA_VERSION,
        "core_steps": [c1.id, c2.id, c3.id],
        "role_steps": {
            "reentry_anchor": c1.id,
            "value_effect": c2.id,
            "state_order_causality": c3.id,
        },
        "candidate_state_key": REENTRANCY_CANDIDATE_STATE_KEY,
        "candidate_outcome_state_key": "reentrancy_causal_order_summary",
        "candidate_exclusion_state_keys": exclusion_state_keys,
        "dependency_edges": dependency_edges,
        "candidate_branching": True,
        "max_candidate_chains": 5,
        "original_step_order": original_step_order,
        "description": (
            "C1 selects nested-entry candidates; C2, C3, and exclusions preserve "
            "candidate_id and assess value effect and candidate-local causality "
            "on the same execution chain. Soft mode preserves state without "
            "overriding Judge answers or final emit."
        ),
    }
    plan.metadata = metadata
    return plan


def _dedupe_step_ids(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in out:
            out.append(item)
    return out


def compile_rule_to_baseline_plan(rule: EvolvingRule) -> EvidencePlan:
    """Compile a rule into a packet-view-based evidence plan."""
    judge_steps: List[JudgeStep] = []
    positive_ids: List[str] = []
    exclusion_ids: List[str] = []
    attack_label = str((rule.metadata or {}).get("attack_label", "") or "")

    all_conditions = [
        *((condition, False) for condition in rule.conditions),
        *((condition, True) for condition in rule.exclusion_conditions),
    ]

    for condition, is_exclusion in all_conditions:
        judge_id = sanitize_step_id(condition.id, f"J{len(judge_steps) + 1}")
        policy = infer_followup_policy_for_rule(condition, attack_label=attack_label)
        default_refs = list(policy.get("default_evidence_refs", []) or infer_packet_views(condition))
        expected_answer = True if is_exclusion else condition.expected_answer

        judge_steps.append(
            JudgeStep(
                id=judge_id,
                question=make_local_judge_question(
                    condition.description,
                    attack_label=attack_label,
                ),
                evidence_refs=default_refs,
                default_evidence_refs=default_refs,
                allowed_followup_views=list(policy.get("allowed_followup_views", [])),
                allowed_tools=list(policy.get("allowed_tools", [])),
                max_followups=int(policy.get("max_followups", 0) or 0),
                expected_answer=expected_answer,
                condition_id=condition.id,
                view_budget=dict(
                    policy.get("view_budget")
                    or condition_view_budget(condition, attack_label=attack_label)
                ),
            )
        )

        if is_exclusion:
            exclusion_ids.append(judge_id)
        else:
            positive_ids.append(judge_id)

    positive_expr = " and ".join(positive_ids) if positive_ids else "False"
    exclusion_expr = " or ".join(exclusion_ids)
    emit_logic = positive_expr
    if exclusion_expr:
        emit_logic = f"({positive_expr}) and not ({exclusion_expr})"

    plan = EvidencePlan(
        plan_id=EvidencePlan.new_id(),
        rule_id=rule.rule_id,
        rule_version=rule.version,
        focus_steps=[],
        judge_steps=judge_steps,
        emit_logic=emit_logic,
        plan_note="Packet-view baseline plan compiled from rule conditions.",
        metadata={
            "generator": "baseline_packet",
            "attack_label": _normalized_attack_label(attack_label),
            **(
                {"view_profile": "reentrancy_fixed_v1"}
                if _normalized_attack_label(attack_label) == "reentrancy"
                else {}
            ),
        },
    )
    apply_insufficient_validation_stateful_runtime(
        plan,
        attack_label=attack_label,
    )
    apply_access_control_binding_mode(
        plan,
        rule=rule,
        mode="stateful",
    )
    apply_token_semantic_stateful_runtime(
        plan,
        rule=rule,
    )
    apply_protocol_accounting_stateful_runtime(
        plan,
        rule=rule,
    )
    apply_market_manipulation_stateful_runtime(
        plan,
        rule=rule,
    )
    apply_flashloans_stateful_runtime(
        plan,
        rule=rule,
    )
    apply_reentrancy_stateful_runtime(
        plan,
        rule=rule,
    )
    return sanitize_plan_judge_questions(plan, attack_label=attack_label)


def constrain_judge_step_first_pass(step: JudgeStep, attack_label: str = "") -> List[str]:
    """Keep round-0 packet views compact and defer extra views to follow-up."""
    defaults = list(step.default_evidence_refs or step.evidence_refs)
    budget = condition_view_budget(step, attack_label=attack_label)
    step.view_budget = dict(budget)
    selected, followups, deferred = constrain_first_pass_view_refs(
        defaults,
        list(step.allowed_followup_views or []),
        max_default_views=int(budget["max_default_views"]),
        max_large_views=int(budget["max_large_views"]),
        max_followup_views=int(budget["max_followup_views"]),
    )
    step.default_evidence_refs = selected
    step.evidence_refs = list(selected)
    step.allowed_followup_views = followups
    return deferred


def constrain_first_pass_view_refs(
    default_refs: List[str],
    allowed_followup_views: Optional[List[str]] = None,
    *,
    max_default_views: int = _FIRST_PASS_MAX_DEFAULT_VIEWS,
    max_large_views: int = _FIRST_PASS_MAX_LARGE_VIEWS,
    max_followup_views: Optional[int] = None,
) -> tuple[List[str], List[str], List[str]]:
    """Bound first-pass evidence breadth while preserving deferred access.

    Packet views are durable artifacts. A plan may expose many of them, but the
    first judge prompt should carry a small locating/evidence slice. Views
    pruned from round 0 are appended to follow-up views so interactive judging
    can fetch their full rows when the local question really needs them.
    """
    defaults = _dedupe_known_views(default_refs)
    followups = _dedupe_known_views(allowed_followup_views or [])
    if not defaults:
        defaults = _dedupe_known_views(_DEFAULT_PACKET_REFS)

    selected: List[str] = []
    deferred: List[str] = []
    large_count = 0

    def choose(view: str) -> bool:
        nonlocal large_count
        if view in selected:
            return True
        if len(selected) >= max_default_views:
            return False
        policy = packet_view_render_policy(view)
        is_large = str(policy.get("cost", "")).lower() == "large"
        if is_large and large_count >= max_large_views:
            return False
        selected.append(view)
        if is_large:
            large_count += 1
        return True

    for view in _FIRST_PASS_ALWAYS_KEEP:
        if view in defaults:
            choose(view)

    for view in defaults:
        if view in selected:
            continue
        # The outline is the first-pass trace locator; full trace rows belong
        # in follow-up when the plan already asked for the compact outline.
        if view == "trace_view" and "trace_outline_view" in defaults:
            deferred.append(view)
            continue
        if view == "reentrancy_state_order_view":
            deferred.append(view)
            continue
        if not choose(view):
            deferred.append(view)

    if not selected:
        selected = defaults[:max_default_views]

    existing_followups = list(followups)
    deferred_followups = _dedupe_known_views(deferred)
    followups = [
        *deferred_followups,
        *[
            view
            for view in existing_followups
            if view not in deferred_followups
        ],
    ]
    if max_followup_views is not None:
        followups = followups[: max(0, int(max_followup_views or 0))]
    return selected, followups, deferred


def judge_step_view_budget_audit(
    step: JudgeStep | Dict[str, Any],
    *,
    attack_label: str = "",
) -> Dict[str, Any]:
    """Describe which planned view routes are active under the step budget.

    This is intentionally audit-only. Runtime still applies the same bounded
    selection, while candidate generation can reject a patch whose newly added
    route would otherwise disappear behind an existing full follow-up list.
    """
    resolved = step if isinstance(step, JudgeStep) else JudgeStep.from_dict(step)
    budget = condition_view_budget(resolved, attack_label=attack_label)
    defaults = _dedupe_known_views(
        list(resolved.default_evidence_refs or resolved.evidence_refs or [])
    )
    requested_followups = _dedupe_known_views(
        list(resolved.allowed_followup_views or [])
    )
    active_defaults, active_followups, _ = constrain_first_pass_view_refs(
        defaults,
        requested_followups,
        max_default_views=int(budget["max_default_views"]),
        max_large_views=int(budget["max_large_views"]),
        max_followup_views=int(budget["max_followup_views"]),
    )
    all_routes = _dedupe_known_views([*defaults, *requested_followups])
    active_routes = _dedupe_known_views([*active_defaults, *active_followups])
    inactive_routes = [view for view in all_routes if view not in active_routes]
    return {
        "condition_id": str(resolved.condition_id or resolved.id or ""),
        "budget": dict(budget),
        "requested_default_views": defaults,
        "requested_followup_views": requested_followups,
        "active_default_views": active_defaults,
        "active_followup_views": active_followups,
        "inactive_views": inactive_routes,
        "over_budget": bool(inactive_routes),
    }


def _dedupe_known_views(view_names: List[str]) -> List[str]:
    views: List[str] = []
    for name in view_names:
        view = str(name or "")
        if view in PACKET_VIEW_CATALOG and view not in views:
            views.append(view)
    return views


def plan_from_dict(data: Dict) -> EvidencePlan:
    return EvidencePlan.from_dict(data)


# ---------------------------------------------------------------------------
# Judge question helpers
# ---------------------------------------------------------------------------

def sanitize_step_id(raw: str, fallback: str) -> str:
    text = re.sub(r"\W+", "_", str(raw or "").strip())
    if not text or not re.match(r"^[A-Za-z_]", text):
        text = fallback
    return text


def make_local_judge_question(
    condition_description: str,
    attack_label: Optional[str] = None,
) -> str:
    cleaned = neutralize_attack_label_mentions(
        condition_description.strip(),
        attack_label=attack_label,
    )
    return (
        "Based only on the provided evidence, is this local condition satisfied: "
        f"{cleaned}"
    )


def sanitize_plan_judge_questions(
    plan: EvidencePlan,
    attack_label: Optional[str] = None,
) -> EvidencePlan:
    sanitized_count = 0
    sanitized_steps: List[str] = []
    for step in plan.judge_steps:
        original = step.question.strip()
        if not original:
            continue
        updated = neutralize_attack_label_mentions(original, attack_label=attack_label)
        if updated != original:
            step.question = updated
            sanitized_count += 1
            sanitized_steps.append(step.id)
    if sanitized_count:
        plan.metadata["judge_question_sanitized_count"] = sanitized_count
        plan.metadata["judge_question_sanitized_steps"] = sanitized_steps
    return plan


def neutralize_attack_label_mentions(
    text: str,
    attack_label: Optional[str] = None,
) -> str:
    updated = text
    for variant in _attack_label_variants(attack_label):
        updated = re.sub(
            rf"(?i)\b{re.escape(variant)}\b",
            "the targeted mechanism",
            updated,
        )
    return updated


def _attack_label_variants(attack_label: Optional[str]) -> List[str]:
    raw = str(attack_label or "").strip()
    if not raw:
        return []
    variants = {
        raw,
        raw.replace("_", " "),
        raw.replace("_", "-"),
        raw.replace("-", " "),
        raw.replace("-", "_"),
    }
    return sorted(
        {item.strip() for item in variants if item.strip()},
        key=len,
        reverse=True,
    )
