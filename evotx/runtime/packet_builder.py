from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __name__ == "__main__" and not __package__:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from evotx.core.labels import normalize_attack_label as _normalize_attack_label
from evotx.runtime.packet_view_manifest import (
    VIEW_MANIFEST,
    VIEW_MANIFEST_VERSION,
    VIEW_ORDER,
    filter_views_for_prompt_policy,
    is_heavy_view,
    manifest_summary_rows,
    resolve_view_dependency_closure,
    summarize_packet_view_costs,
    view_cost,
    view_cost_score,
    view_tier,
)


# ============================================================
# Packet view manifest
# ============================================================

DEFAULT_PACKET_VIEWS = [
    "tx_card",
    "address_labels",
    "token_info",
    "evidence_adequacy_view",
    "operation_summary_view",
    "classification_digest_view",
    "trace_outline_view",
    "trace_view",
    "critical_call_view",
    "critical_call_argument_view",
    "source_unavailable_auth_view",
    "reentrancy_state_order_summary_view",
    "reentrancy_candidate_catalog_view",
    "reentrancy_state_order_view",
    "unknown_selector_view",
    "event_view",
    "state_change_view",
    "semantic_state_delta_view",
    "token_semantic_delta_summary_view",
    "token_accounting_origin_view",
    "price_relevant_state_view",
    "amm_reserve_transition_view",
    "market_mechanism_profile_view",
    "protocol_accounting_outcome_view",
    "transfer_event_view",
    "external_fundflow_view",
    "profit_loss_view",
    "participant_net_delta_view",
    "value_release_view",
    "contribution_vs_payout_view",
    "flash_or_atomic_capital_view",
    "beneficiary_controller_view",
]

JUDGE_VIEW_PROFILES = {
    "common": [
        "tx_card",
        "evidence_adequacy_view",
        "operation_summary_view",
        "classification_digest_view",
    ],
    "access_control": [
        "address_labels",
        "trace_outline_view",
        "critical_call_view",
        "critical_call_argument_view",
        "source_unavailable_auth_view",
        "unknown_selector_view",
        "value_release_view",
        "beneficiary_controller_view",
        "semantic_state_delta_view",
    ],
    "insufficient_validation": [
        "trace_outline_view",
        "critical_call_view",
        "critical_call_argument_view",
        "unknown_selector_view",
        "event_view",
        "semantic_state_delta_view",
        "value_release_view",
    ],
    "price_manipulation": [
        "price_relevant_state_view",
        "amm_reserve_transition_view",
        "event_view",
        "critical_call_view",
        "flash_or_atomic_capital_view",
        "value_release_view",
    ],
    "market_manipulation": [
        "market_mechanism_profile_view",
        "price_relevant_state_view",
        "amm_reserve_transition_view",
        "event_view",
        "critical_call_view",
        "transfer_event_view",
        "value_release_view",
    ],
    "reentrancy": [
        "trace_outline_view",
        "critical_call_view",
        "critical_call_argument_view",
        "reentrancy_state_order_summary_view",
        "reentrancy_candidate_catalog_view",
        "unknown_selector_view",
        "value_release_view",
        "semantic_state_delta_view",
    ],
    "flashloans": [
        "flash_or_atomic_capital_view",
        "trace_outline_view",
        "critical_call_view",
        "external_fundflow_view",
        "transfer_event_view",
        "value_release_view",
    ],
    "protocol_accounting_exploitation": [
        "trace_outline_view",
        "critical_call_view",
        "semantic_state_delta_view",
        "protocol_accounting_outcome_view",
        "contribution_vs_payout_view",
        "value_release_view",
        "participant_net_delta_view",
    ],
    "token_semantic_exploitation": [
        "transfer_event_view",
        "token_semantic_delta_summary_view",
        "token_accounting_origin_view",
        "semantic_state_delta_view",
        "critical_call_view",
        "unknown_selector_view",
        "amm_reserve_transition_view",
        "participant_net_delta_view",
        "value_release_view",
    ],
}

DEFAULT_PACKET_CONFIG = {
    "max_trace_nodes": 420,
    "max_address_labels": 160,
    "max_token_info": 160,
    "max_events": 180,
    "max_state_rows": 260,
    "max_transfers": 180,
    "max_fundflow_records": 240,
    "max_critical_calls": 200,
    "max_critical_call_arguments": 80,
    "max_source_unavailable_auth_rows": 80,
    "max_reentrancy_state_order_rows": 80,
    "max_reentrancy_state_nodes_per_row": 18,
    "max_unknown_selectors": 80,
    "max_semantic_state_rows": 240,
    "max_token_semantic_delta_rows": 80,
    "max_token_accounting_origin_rows": 80,
    "max_price_state_contracts": 80,
    "max_amm_pairs": 80,
    "max_market_mechanism_profiles": 24,
    "max_protocol_accounting_outcomes": 48,
    "max_participants": 120,
    "max_value_release_rows": 120,
    "max_value_release_records_per_row": 24,
    "max_value_release_state_rows": 20,
    "max_contribution_rows": 40,
    "max_atomic_capital_rows": 80,
    "max_beneficiary_rows": 20,
    "max_reentrancy_state_order_summary_rows": 16,
    "max_reentrancy_candidate_catalog_rows": 24,
    "include_reentrancy_state_order_details": False,
    "include_verbose_call_path_tail": False,
    "include_raw_evidence_store": False,
    "use_aliases": False,
}

LEAKAGE_LABEL_RE = re.compile(
    r"(exploiter|attacker|hacker|hack|victim|drainer|scam|exploit)",
    re.IGNORECASE,
)

ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
SELECTOR_RE = re.compile(r"^0x[a-fA-F0-9]{8}$")

CALL_TYPES = {
    "call",
    "staticcall",
    "delegatecall",
    "callcode",
    "create",
    "create2",
}
STATE_TYPES = {"sload", "sstore"}
UNKNOWN_SELECTOR_CALL_TYPES = {
    "call",
    "staticcall",
    "delegatecall",
    "callcode",
}

PACKET_FORMAT_VERSION = "evotx_packet_v3"
PACKET_DICTIONARY_FORMAT_VERSION = "evotx_packet_dictionary_v2"

CRITICAL_CALL_KEYWORDS = [
    "flash",
    "flashloan",
    "flashcallback",
    "uniswapv2call",
    "pancakecall",
    "pancakev3flashcallback",
    "swap",
    "exactinput",
    "exactoutput",
    "exchange",
    "mint",
    "burn",
    "redeem",
    "withdraw",
    "deposit",
    "borrow",
    "repay",
    "liquidate",
    "liquidation",
    "claim",
    "reward",
    "harvest",
    "getreserves",
    "get_virtual_price",
    "latestanswer",
    "latestrounddata",
    "consult",
    "peek",
    "read",
    "balanceof",
    "totalsupply",
    "sync",
    "skim",
]

VALUE_RELEASE_CALL_KEYWORDS = [
    "withdraw",
    "redeem",
    "borrow",
    "claim",
    "reward",
    "harvest",
    "transfer",
    "send",
    "payout",
    "pay",
    "sweep",
    "skim",
    "mint",
    "liquidate",
    "liquidation",
    "exit",
]

PRICE_STATE_KEYWORDS = [
    "reserve0",
    "reserve1",
    "reserves",
    "getreserves",
    "price",
    "pricecumulative",
    "klast",
    "totalsupply",
    "balance",
    "share",
    "exchangerate",
    "virtualprice",
    "collateral",
    "debt",
    "borrowindex",
    "liquidity",
    "tick",
    "sqrtprice",
    "feegrowth",
    "oracle",
]

_PROTOCOL_CONTRACT_LABEL_KEYWORDS = [
    "pair", "pool", "vault", "router", "lending", "amm",
    "swap", "exchange", "curve", "uniswap", "sushiswap",
    "balancer", "aave", "compound", "maker", "liquity",
    "chef", "gauge", "staking", "bridge", "aggregator",
    "oracle", "keeper", "controller", "proxy", "factory",
    "strategy", "yield", "harvest", "vesting",
    "manager", "registry", "feed",
]

_BALANCE_VARIABLE_PATTERNS = re.compile(
    r"(^|[\s_:.\-])(balances?|balanceof|_balance)([\s_:.\-]|$)",
    re.IGNORECASE,
)

_HIGH_RELEVANCE_VARIABLES = [
    "reserve0", "reserve1", "reserves", "getreserves",
    "price", "pricecumulative", "klast", "exchangerate",
    "virtualprice", "tick", "sqrtprice", "feegrowth",
]

_MEDIUM_RELEVANCE_VARIABLES = [
    "totalsupply", "share", "collateral", "debt",
    "borrowindex", "liquidity", "oracle",
]


# ============================================================
# IO
# ============================================================

def load_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def norm_hash(tx_hash: str) -> str:
    return str(tx_hash).lower().strip()


def file_exists(path: str | Path) -> bool:
    return Path(path).exists() and Path(path).is_file()


def find_existing_file(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if file_exists(p):
            return p
    return None


def normalize_attack_label(label: Any) -> str:
    return _normalize_attack_label(label, default="")


def resolve_packet_views(
    *,
    include_views: Optional[List[str]] = None,
    attack_label: Optional[str] = None,
    packet_profile: str = "full",
) -> List[str]:
    """Resolve packet views for full, judge, or minimal packet profiles."""
    if include_views is not None:
        return _unique([str(view) for view in include_views if str(view).strip()])

    profile = str(packet_profile or "full").strip().lower()
    if profile == "full":
        return list(DEFAULT_PACKET_VIEWS)
    if profile == "minimal":
        return list(JUDGE_VIEW_PROFILES["common"])
    if profile == "judge":
        label = normalize_attack_label(attack_label)
        views = list(JUDGE_VIEW_PROFILES["common"])
        views.extend(JUDGE_VIEW_PROFILES.get(label, []))
        return [view for view in _unique(views) if view in DEFAULT_PACKET_VIEWS]
    raise ValueError("--profile/packet_profile must be one of: full, judge, minimal")


# ============================================================
# Compact helpers
# ============================================================

def compact_value(
    v: Any,
    max_str_len: int = 220,
    max_list_len: int = 8,
    max_dict_items: int = 20,
) -> Any:
    """Compact long strings/lists/dicts without adding semantic judgment."""
    if isinstance(v, str):
        if len(v) > max_str_len:
            return v[:max_str_len] + f"...<truncated:{len(v)} chars>"
        return v

    if isinstance(v, list):
        if len(v) > max_list_len:
            return [
                compact_value(x, max_str_len, max_list_len, max_dict_items)
                for x in v[:max_list_len]
            ] + [f"...<truncated:{len(v) - max_list_len} items>"]
        return [compact_value(x, max_str_len, max_list_len, max_dict_items) for x in v]

    if isinstance(v, dict):
        out: Dict[str, Any] = {}
        for i, (k, val) in enumerate(v.items()):
            if i >= max_dict_items:
                out["..."] = f"<truncated:{len(v) - max_dict_items} keys>"
                break
            out[str(k)] = compact_value(val, max_str_len, max_list_len, max_dict_items)
        return out

    return v


def omit_empty(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def get_param_value(params: Any, candidate_names: List[str]) -> Any:
    if isinstance(params, dict):
        lowered = {str(k).lower(): k for k in params.keys()}
        for name in candidate_names:
            key = name if name in params else lowered.get(name.lower())
            if key is None:
                continue
            v = params[key]
            if isinstance(v, dict):
                return v.get("value")
            return v

    if isinstance(params, list):
        for item in params:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).lower() in {x.lower() for x in candidate_names}:
                return item.get("value")

    return None


def normalize_address(address: Any) -> str:
    text = str(address or "").strip().lower()
    return text if ADDRESS_RE.match(text) else text


def is_address(value: Any) -> bool:
    return bool(ADDRESS_RE.match(str(value or "").strip()))


def _is_data_not_available(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().replace(" ", "_") == "not_available"


def parse_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        if text.startswith(("0x", "0X")):
            return Decimal(int(text, 16))
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def format_decimal(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def decimal_delta(prev: Any, current: Any) -> Optional[Decimal]:
    p = parse_decimal(prev)
    c = parse_decimal(current)
    if p is None or c is None:
        return None
    return c - p


def safe_pct_change(first: Any, last: Any) -> Optional[str]:
    f = parse_decimal(first)
    l = parse_decimal(last)
    if f is None or l is None or f == 0:
        return None
    return format_decimal(((l - f) / abs(f)) * Decimal(100))


def normalize_token_amount(value: Any, decimals: Any) -> Optional[str]:
    amount = parse_decimal(value)
    if amount is None:
        return None
    try:
        dec = int(decimals)
    except (TypeError, ValueError):
        return None
    return format_decimal(amount / (Decimal(10) ** dec))


def address_label_map(syn: Dict[str, Any]) -> Dict[str, str]:
    raw = syn.get("label_address_map", {}) or {}
    if not isinstance(raw, dict):
        return {}
    return {normalize_address(k): str(v) for k, v in raw.items()}


def token_info_map(token_info: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(token_info, dict):
        out = {}
        for addr, info in token_info.items():
            if isinstance(info, dict):
                row = dict(info)
            else:
                row = {"value": info}
            row.setdefault("address", addr)
            out[normalize_address(addr)] = row
        return out
    if isinstance(token_info, list):
        out = {}
        for item in token_info:
            if not isinstance(item, dict):
                continue
            addr = normalize_address(item.get("address"))
            if addr:
                out[addr] = dict(item)
        return out
    return {}


def label_for_address(
    syn: Dict[str, Any],
    address: Any,
    *,
    sanitize: bool = True,
) -> Tuple[str, bool]:
    addr = normalize_address(address)
    raw = address_label_map(syn).get(addr, "")
    return sanitize_label_for_address(raw, addr, syn) if sanitize else (raw, False)


def label_for_node(
    syn: Dict[str, Any],
    address: Any,
    node_label: Any,
) -> Tuple[str, bool]:
    mapped, mapped_sanitized = label_for_address(syn, address)
    if mapped:
        return mapped, mapped_sanitized
    return sanitize_label_for_address(node_label, address, syn)


def sanitize_label_for_address(
    label: Any,
    address: Any = None,
    syn: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool]:
    raw = str(label or "").strip()
    if not raw:
        return "", False
    if not LEAKAGE_LABEL_RE.search(raw):
        return raw, False
    addr = normalize_address(address)
    sender = normalize_address((syn or {}).get("sender"))
    sanitized = "externally_labeled_eoa" if addr and addr == sender else "externally_labeled_address"
    return sanitized, True


def sanitize_label_value(label: Any, address: Any = None, syn: Optional[Dict[str, Any]] = None) -> Any:
    sanitized, _ = sanitize_label_for_address(label, address, syn)
    return sanitized


def build_packet_dictionary(
    syn: Dict[str, Any],
    trace_index: "TraceIndex",
    *,
    max_addresses: int = 1000,
    max_functions: int = 1000,
) -> Dict[str, Any]:
    addresses: Dict[str, Dict[str, Any]] = {}
    address_to_alias: Dict[str, str] = {}
    functions: Dict[str, str] = {}
    function_to_alias: Dict[str, str] = {}

    def add_address(value: Any, label: Any = None) -> None:
        addr = normalize_address(value)
        if not addr or addr in address_to_alias or len(address_to_alias) >= max_addresses:
            return
        raw_label = label
        if raw_label in (None, ""):
            raw_label = address_label_map(syn).get(addr)
        sanitized, label_sanitized = sanitize_label_for_address(raw_label, addr, syn)
        alias = f"A{len(address_to_alias)}"
        address_to_alias[addr] = alias
        addresses[alias] = omit_empty({
            "address": addr,
            "label": sanitized,
            "raw_label": raw_label,
            "label_sanitized": label_sanitized,
        })

    def add_function(value: Any) -> None:
        fn = str(value or "").strip()
        if not fn or fn in function_to_alias or len(function_to_alias) >= max_functions:
            return
        alias = f"F{len(function_to_alias)}"
        function_to_alias[fn] = alias
        functions[alias] = fn

    for addr, label in address_label_map(syn).items():
        add_address(addr, label)
    for _, node in iter_trace_nodes(trace_index):
        for key in ("address", "caller", "from", "to", "recipient", "token", "contract"):
            add_address(node.get(key), node.get(f"{key}_label"))
        add_function(get_function_name(node))
        parent_fn = get_function_name(node)
        if parent_fn:
            add_function(parent_fn)

    return {
        "format": PACKET_DICTIONARY_FORMAT_VERSION,
        "addresses": addresses,
        "functions": functions,
    }


def encode_address(addr: Any, dictionary: Dict[str, Any]) -> str:
    target = normalize_address(addr)
    for alias, item in dict(dictionary.get("addresses") or {}).items():
        if (
            isinstance(item, dict)
            and normalize_address(item.get("address")) == target
        ):
            return str(alias)
    return ""


def encode_function(fn: Any, dictionary: Dict[str, Any]) -> str:
    target = str(fn or "").strip()
    for alias, function in dict(dictionary.get("functions") or {}).items():
        if str(function or "").strip() == target:
            return str(alias)
    return ""


def maybe_encode_address_fields(
    row: Dict[str, Any],
    dictionary: Dict[str, Any],
    fields: List[str] = None,
) -> Dict[str, Any]:
    fields = fields or ["address", "caller", "callee", "from", "to", "recipient", "token", "contract"]
    out = dict(row)
    for field_name in fields:
        alias = encode_address(out.get(field_name), dictionary)
        if alias:
            out[f"{field_name}_alias"] = alias
    return out


def maybe_encode_function_fields(
    row: Dict[str, Any],
    dictionary: Dict[str, Any],
    fields: List[str] = None,
) -> Dict[str, Any]:
    fields = fields or ["function", "parent_function"]
    out = dict(row)
    for field_name in fields:
        alias = encode_function(out.get(field_name), dictionary)
        if alias:
            out[f"{field_name}_alias"] = alias
    return out


def build_evidence_store(
    syn: Dict[str, Any],
    trace_index: "TraceIndex",
    *,
    include_raw: bool = False,
    packet_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {**DEFAULT_PACKET_CONFIG, **(packet_config or {})}
    evidence: Dict[str, Any] = {}
    counts: Counter[str] = Counter()
    for node_key, node in iter_trace_nodes(trace_index):
        evidence_id = node_evidence_id(trace_index, node_key)
        node_type = get_node_type(node)
        counts[node_type] += 1
        display_id = trace_index.display_id_by_id.get(node_key, node.get("id", node_key))
        row = {
            "evidence_id": evidence_id,
            "id": display_id,
            "type": node_type,
            "depth": trace_index.depth_by_id.get(node_key),
            **parent_context(
                trace_index,
                node_key,
                include_call_path_tail=bool(cfg.get("include_verbose_call_path_tail")),
            ),
            "function": get_function_name(node),
            "selector": get_selector(node),
            "caller": node.get("caller"),
            "address": node.get("address"),
            "callee": node.get("address"),
            "value": node.get("value"),
        }
        if node_type == "event":
            row.update({
                "event_signature": node.get("event_signature") or node.get("signature"),
                "params": compact_value(node.get("params") or node.get("args"), max_dict_items=12, max_list_len=12),
            })
        if is_state_like(node):
            row.update({
                "slot_key": node.get("slot_key"),
                "prev": node.get("prev"),
                "current": node.get("current"),
                "state_change": compact_value(node.get("state_change"), max_dict_items=12, max_list_len=12),
            })
        if is_call_like(node):
            row.update({
                "params": compact_value(node.get("params") or node.get("args"), max_dict_items=12, max_list_len=12),
                "return_values": compact_value(node.get("return_values") or node.get("returns"), max_dict_items=8, max_list_len=8),
            })
            if include_raw:
                row["calldata"] = compact_value(node.get("calldata") or node.get("input"), max_str_len=8000)
                row["output"] = compact_value(node.get("output"), max_str_len=8000)
        else:
            if include_raw:
                row["raw"] = compact_value(node, max_dict_items=80, max_list_len=80, max_str_len=8000)
        evidence[evidence_id] = omit_empty(row)
    return {
        "format": "evotx_evidence_store_v1",
        "counts": dict(counts),
        "evidence": evidence,
    }


# ============================================================
# Trace indexing
# ============================================================

@dataclass
class TraceIndex:
    node_by_id: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    parent_by_id: Dict[Any, Any] = field(default_factory=dict)
    children_by_id: Dict[Any, List[Any]] = field(default_factory=lambda: defaultdict(list))
    depth_by_id: Dict[Any, int] = field(default_factory=dict)
    path_by_id: Dict[Any, str] = field(default_factory=dict)
    type_by_id: Dict[Any, str] = field(default_factory=dict)
    parent_call_by_id: Dict[Any, Any] = field(default_factory=dict)
    node_index_by_id: Dict[Any, int] = field(default_factory=dict)
    call_path_by_id: Dict[Any, List[str]] = field(default_factory=dict)
    call_path_ids_by_id: Dict[Any, List[Any]] = field(default_factory=dict)
    display_id_by_id: Dict[Any, Any] = field(default_factory=dict)
    sequence: List[Any] = field(default_factory=list)
    duplicate_node_ids: List[Any] = field(default_factory=list)


def build_trace_index(trace_root: Any) -> TraceIndex:
    index = TraceIndex()
    if not isinstance(trace_root, dict):
        return index

    seen_display_ids: set = set()

    def visit(
        node: Dict[str, Any],
        *,
        parent_key: Any,
        depth: int,
        path: str,
        parent_call_key: Any,
        call_path: List[str],
        call_id_path: List[Any],
    ) -> None:
        display_id = node.get("id", path)
        node_key = display_id
        if node_key in index.node_by_id:
            index.duplicate_node_ids.append(display_id)
            node_key = path
        seen_display_ids.add(display_id)

        node_type = get_node_type(node)
        index.node_by_id[node_key] = node
        index.parent_by_id[node_key] = parent_key
        index.depth_by_id[node_key] = depth
        index.path_by_id[node_key] = path
        index.type_by_id[node_key] = node_type
        index.parent_call_by_id[node_key] = parent_call_key
        index.node_index_by_id[node_key] = len(index.sequence)
        index.display_id_by_id[node_key] = display_id
        index.sequence.append(node_key)

        current_call_path = list(call_path)
        current_call_id_path = list(call_id_path)
        if is_call_like(node):
            current_call_path = current_call_path + [call_path_label(node, display_id)]
            current_call_id_path = current_call_id_path + [display_id]
            next_parent_call_key = node_key
        else:
            next_parent_call_key = parent_call_key
        index.call_path_by_id[node_key] = current_call_path
        index.call_path_ids_by_id[node_key] = current_call_id_path

        if parent_key is not None:
            index.children_by_id[parent_key].append(node_key)

        children = node.get("children") or []
        if isinstance(children, list):
            for i, child in enumerate(children):
                if isinstance(child, dict):
                    visit(
                        child,
                        parent_key=node_key,
                        depth=depth + 1,
                        path=f"{path}.children[{i}]",
                        parent_call_key=next_parent_call_key,
                        call_path=current_call_path,
                        call_id_path=current_call_id_path,
                    )

    visit(
        trace_root,
        parent_key=None,
        depth=0,
        path="trace",
        parent_call_key=None,
        call_path=[],
        call_id_path=[],
    )
    return index


def get_node_type(node: Dict[str, Any]) -> str:
    return str(node.get("type") or node.get("call_type") or "node").lower()


def is_call_like(node: Dict[str, Any]) -> bool:
    node_type = get_node_type(node)
    call_type = str(node.get("call_type") or "").lower()
    return node_type in CALL_TYPES or call_type in CALL_TYPES


def is_state_like(node: Dict[str, Any]) -> bool:
    return get_node_type(node) in STATE_TYPES or "state_change" in node


def call_path_label(node: Dict[str, Any], display_id: Any) -> str:
    node_type = get_node_type(node)
    fn = get_function_name(node) or get_selector(node) or "<unknown>"
    return f"{node_type}:{display_id} {compact_function_label(fn)}"


def compact_function_label(function: Any, max_len: int = 80) -> str:
    text = str(function or "").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def get_function_name(node: Dict[str, Any]) -> str:
    for key in ("function", "function_name", "decoded_function", "name"):
        value = node.get(key)
        if value:
            return str(value)
    return ""


def get_selector(node: Dict[str, Any]) -> str:
    for key in ("raw_selector", "function_selector", "selector"):
        value = str(node.get(key) or "").strip()
        if SELECTOR_RE.match(value):
            return value.lower()
    fn = get_function_name(node).strip()
    if SELECTOR_RE.match(fn):
        return fn.lower()
    calldata = str(node.get("calldata") or "").strip()
    if calldata.startswith("0x") and len(calldata) >= 10:
        return calldata[:10].lower()
    return ""


def node_evidence_id(index: TraceIndex, node_key: Any) -> str:
    node = index.node_by_id.get(node_key, {})
    node_type = get_node_type(node)
    display_id = index.display_id_by_id.get(node_key, node.get("id", node_key))
    return f"{node_type}:{display_id}"


def iter_trace_nodes(index: TraceIndex) -> Iterable[Tuple[Any, Dict[str, Any]]]:
    for node_key in index.sequence:
        node = index.node_by_id.get(node_key)
        if isinstance(node, dict):
            yield node_key, node


def parent_context(
    index: TraceIndex,
    node_key: Any,
    *,
    include_call_path_tail: bool = False,
) -> Dict[str, Any]:
    parent_id = index.parent_by_id.get(node_key)
    parent = index.node_by_id.get(parent_id, {}) if parent_id is not None else {}
    parent_display = index.display_id_by_id.get(parent_id, parent_id)
    context = {
        "parent_id": parent_display,
        "depth": index.depth_by_id.get(node_key),
        "parent_function": get_function_name(parent),
        "parent_address": parent.get("address"),
        "path_ids": index.call_path_ids_by_id.get(node_key, [])[-6:],
    }
    if include_call_path_tail:
        context["call_path_tail"] = index.call_path_by_id.get(node_key, [])[-3:]
    return omit_empty(context)


def collect_nearby_evidence_ids(
    index: TraceIndex,
    node_key: Any,
    *,
    target: str,
    max_items: int = 5,
) -> List[str]:
    if target == "event":
        predicate = lambda n: get_node_type(n) == "event"
    else:
        predicate = is_state_like

    found: List[Any] = []

    def collect_descendants(parent_key: Any) -> None:
        for child_key in index.children_by_id.get(parent_key, []):
            child = index.node_by_id.get(child_key, {})
            if predicate(child):
                found.append(child_key)
                if len(found) >= max_items:
                    return
            collect_descendants(child_key)
            if len(found) >= max_items:
                return

    collect_descendants(node_key)

    if len(found) < max_items:
        pos = index.node_index_by_id.get(node_key, 0)
        lo = max(0, pos - 10)
        hi = min(len(index.sequence), pos + 11)
        for nearby_key in index.sequence[lo:hi]:
            if nearby_key == node_key or nearby_key in found:
                continue
            nearby = index.node_by_id.get(nearby_key, {})
            if predicate(nearby):
                found.append(nearby_key)
                if len(found) >= max_items:
                    break

    return [node_evidence_id(index, key) for key in found[:max_items]]


def children_summary(index: TraceIndex, node_key: Any) -> Dict[str, int]:
    counter: Counter[str] = Counter()

    def visit(parent_key: Any) -> None:
        for child_key in index.children_by_id.get(parent_key, []):
            child = index.node_by_id.get(child_key, {})
            node_type = get_node_type(child)
            if node_type in CALL_TYPES:
                counter["call_nodes"] += 1
            elif node_type == "event":
                counter["event_nodes"] += 1
            elif node_type == "sload":
                counter["sload_nodes"] += 1
            elif node_type == "sstore":
                counter["sstore_nodes"] += 1
            visit(child_key)

    visit(node_key)
    return dict(counter)


def trace_counts(index: TraceIndex) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for _, node in iter_trace_nodes(index):
        node_type = get_node_type(node)
        if node_type in CALL_TYPES:
            counter["call_nodes"] += 1
            if is_decoded_call(node):
                counter["decoded_call_nodes"] += 1
            else:
                counter["unknown_call_nodes"] += 1
        elif node_type == "event":
            counter["event_nodes"] += 1
        elif node_type == "sload":
            counter["sload_nodes"] += 1
        elif node_type == "sstore":
            counter["sstore_nodes"] += 1
        counter["total_nodes"] += 1
    return dict(counter)


def is_decoded_call(node: Dict[str, Any]) -> bool:
    fn = get_function_name(node).strip()
    if not fn:
        return False
    if fn in {"transaction_root", "new <unknown>"}:
        return True
    if SELECTOR_RE.match(fn):
        return False
    selector_kind = str(node.get("selector_kind") or "").lower()
    if selector_kind == "unknown":
        return False
    return "(" in fn or any(ch.isalpha() for ch in fn)


def unknown_selectors(index: TraceIndex, max_items: int = 30) -> List[str]:
    selectors: List[str] = []
    seen: set = set()
    for _, node in iter_trace_nodes(index):
        if not is_call_like(node):
            continue
        selector = get_selector(node)
        if selector and not is_decoded_call(node) and selector not in seen:
            selectors.append(selector)
            seen.add(selector)
        if len(selectors) >= max_items:
            break
    return selectors


# ============================================================
# Render synthesized views
# ============================================================

def render_tx_card(syn: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "transaction_hash",
        "chain",
        "block_number",
        "timestamp",
        "status",
        "nonce",
        "position_in_block",
        "sender",
        "sender_label",
        "receiver",
        "receiver_label",
        "value",
        "gas_limit",
        "gas_used",
        "gas_price",
        "transaction_fee",
        "int_txn_count",
        "event_count",
        "revert_message",
    ]
    card = {k: syn.get(k) for k in keys if k in syn}
    for prefix in ("sender", "receiver"):
        label_key = f"{prefix}_label"
        if label_key in card:
            sanitized, flag = sanitize_label_for_address(
                card.get(label_key),
                card.get(prefix),
                syn,
            )
            card[label_key] = sanitized
            if flag:
                card[f"{label_key}_sanitized"] = True
    return omit_empty(card)


def render_address_label_view(
    syn: Dict[str, Any],
    max_items: int = 160,
    sanitize: bool = True,
    ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    labels = dict(address_label_map(syn))
    for key in ("sender", "receiver"):
        addr = normalize_address(syn.get(key))
        if addr and is_address(addr):
            labels.setdefault(addr, syn.get(f"{key}_label") or "")
    for i, (addr, label) in enumerate(labels.items()):
        if i >= max_items:
            break
        addr = normalize_address(addr)
        sanitized, flag = (
            sanitize_label_for_address(label, addr, syn)
            if sanitize
            else (label, False)
        )
        rows.append({
            "evidence_id": f"address:{addr}",
            "address": addr,
            "raw_label": label,
            "sanitized_label": sanitized,
            "label_sanitized": flag,
        })
    return rows


def render_token_info_view(syn: Dict[str, Any], max_items: int = 160) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, (addr, info) in enumerate(token_info_map(syn.get("token_info")).items()):
        if i >= max_items:
            break
        label, sanitized = label_for_address(syn, addr)
        rows.append(omit_empty({
            "evidence_id": f"token:{addr}",
            "address": addr,
            "label": label,
            "label_sanitized": sanitized,
            "symbol": info.get("symbol"),
            "name": info.get("name"),
            "decimals": info.get("decimals"),
        }))
    return rows


def render_trace_outline_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_items: int = 300,
) -> List[Dict[str, Any]]:
    """Compact call-tree outline without large params/return_values/state_change blobs."""
    rows: List[Dict[str, Any]] = []
    for node_key, node in iter_trace_nodes(trace_index):
        if len(rows) >= max_items:
            break

        node_type = get_node_type(node)
        if not is_call_like(node) and node_type != "event":
            continue
        display_id = trace_index.display_id_by_id.get(node_key, node.get("id", node_key))
        parent_key = trace_index.parent_by_id.get(node_key)
        parent_display = trace_index.display_id_by_id.get(parent_key) if parent_key is not None else None

        caller_label, caller_label_sanitized = label_for_node(
            syn, node.get("caller"), node.get("caller_label"),
        )
        addr_label, addr_label_sanitized = label_for_node(
            syn, node.get("address"), node.get("address_label"),
        )

        row: Dict[str, Any] = {
            "evidence_id": node_evidence_id(trace_index, node_key),
            "id": display_id,
            "type": node_type,
            "depth": trace_index.depth_by_id.get(node_key),
            "parent_id": parent_display,
            "function": compact_function_label(get_function_name(node)),
            "selector": get_selector(node) or None,
            "caller": node.get("caller"),
            "caller_label_sanitized": caller_label_sanitized or None,
            "address": node.get("address"),
            "address_label_sanitized": addr_label_sanitized or None,
        }
        parsed_value = parse_decimal(node.get("value"))
        if parsed_value not in (None, Decimal(0)):
            row["value"] = node.get("value")

        if node_type == "event":
            row["event_name"] = get_function_name(node) or None

        rows.append(omit_empty(row))
    return rows


def render_trace_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_nodes: int = 420,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node_key, node in iter_trace_nodes(trace_index):
        if len(rows) >= max_nodes:
            break

        node_type = get_node_type(node)
        display_id = trace_index.display_id_by_id.get(node_key, node.get("id", node_key))
        addr_label, addr_label_sanitized = label_for_node(
            syn, node.get("address"), node.get("address_label")
        )
        caller_label, caller_label_sanitized = label_for_node(
            syn, node.get("caller"), node.get("caller_label")
        )

        selector = get_selector(node)
        trace_parent = parent_context(trace_index, node_key)
        trace_parent.pop("parent_function", None)
        trace_parent.pop("parent_address", None)
        row = {
            "evidence_id": node_evidence_id(trace_index, node_key),
            "id": display_id,
            "type": node_type,
            "call_type": node.get("call_type"),
            **trace_parent,
            "address": node.get("address"),
            "address_label": addr_label,
            "address_label_sanitized": addr_label_sanitized or None,
            "caller": node.get("caller"),
            "caller_label": caller_label,
            "caller_label_sanitized": caller_label_sanitized or None,
            "function": get_function_name(node),
            "selector": selector or None,
            "gas": node.get("gas"),
        }
        selector_kind = str(node.get("selector_kind") or "").strip().lower()
        if selector_kind and selector_kind not in {"function_selector", "event_signature"}:
            row["selector_kind"] = selector_kind
        parsed_value = parse_decimal(node.get("value"))
        if parsed_value not in (None, Decimal(0)):
            row["value"] = node.get("value")
        if bool(node.get("revert")):
            row["revert"] = True
            row["revert_msg"] = node.get("revert_msg")

        args = node.get("args_in") or node.get("params")
        if args:
            row["args"] = compact_value(args)
        if node.get("return_values") or node.get("output"):
            row["return_values"] = compact_value(node.get("return_values") or node.get("output"))

        if node_type in STATE_TYPES:
            state_change = (
                node.get("state_change")
                if isinstance(node.get("state_change"), dict)
                else {}
            )
            row["slot_key"] = node.get("slot_key") or state_change.get("key")
            if node_type == "sload":
                row["value"] = node.get("value")
            else:
                row["prev"] = (
                    node.get("prev")
                    if node.get("prev") not in (None, "")
                    else state_change.get("prev")
                )
                row["current"] = (
                    node.get("current")
                    if node.get("current") not in (None, "")
                    else state_change.get("current")
                )

        if node_type == "event":
            event_args = node.get("args") or node.get("params")
            row["event"] = get_function_name(node) or None
            row.pop("function", None)
            row.pop("selector", None)
            row.pop("selector_kind", None)
            if event_args:
                row["args"] = compact_value(event_args)
            else:
                row["topics"] = compact_value(node.get("topics"), max_list_len=4)
                row["data"] = compact_value(node.get("data"))

        rows.append(omit_empty(row))
    return rows


def render_event_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_events: int = 180,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for node_key, node in iter_trace_nodes(trace_index):
        if len(events) >= max_events:
            break
        if get_node_type(node) != "event":
            continue
        event_id = trace_index.display_id_by_id.get(node_key, node.get("id", node_key))
        addr_label, addr_label_sanitized = label_for_node(
            syn, node.get("address"), node.get("address_label")
        )
        caller_label, caller_label_sanitized = label_for_node(
            syn, node.get("caller"), node.get("caller_label")
        )
        event_args = node.get("args") or node.get("params")
        event_row = {
            "evidence_id": node_evidence_id(trace_index, node_key),
            "id": event_id,
            **parent_context(trace_index, node_key),
            "address": node.get("address"),
            "address_label": addr_label,
            "address_label_sanitized": addr_label_sanitized or None,
            "event": get_function_name(node),
            "caller": node.get("caller"),
            "caller_label": caller_label,
            "caller_label_sanitized": caller_label_sanitized or None,
        }
        if event_args:
            event_row["args"] = compact_value(event_args)
        else:
            event_row["event_signature"] = node.get("event_signature")
            event_row["topics"] = compact_value(node.get("topics"), max_list_len=4)
            event_row["data"] = compact_value(node.get("data"))
        events.append(omit_empty(event_row))
    return events


def render_state_change_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_items: int = 260,
) -> List[Dict[str, Any]]:
    state_changes_raw = syn.get("state_changes") or []
    if _is_data_not_available(state_changes_raw):
        return {"available": False, "reason": "state_changes reported as not available by data source"}

    rows: List[Dict[str, Any]] = []

    for node_key, node in iter_trace_nodes(trace_index):
        if len(rows) >= max_items:
            break
        node_type = get_node_type(node)
        if node_type not in STATE_TYPES and "state_change" not in node:
            continue
        display_id = trace_index.display_id_by_id.get(node_key, node.get("id", node_key))
        addr_label, label_sanitized = label_for_node(
            syn, node.get("address"), node.get("address_label")
        )
        state_change = (
            node.get("state_change")
            if isinstance(node.get("state_change"), dict)
            else {}
        )
        state_row = {
            "evidence_id": node_evidence_id(trace_index, node_key),
            "id": display_id,
            "source": "trace_node",
            "op": node_type,
            **parent_context(trace_index, node_key),
            "address": node.get("address"),
            "address_label": addr_label,
            "address_label_sanitized": label_sanitized or None,
            "slot_key": node.get("slot_key") or state_change.get("key"),
            "value": node.get("value"),
            "prev": (
                node.get("prev")
                if node.get("prev") not in (None, "")
                else state_change.get("prev")
            ),
            "current": (
                node.get("current")
                if node.get("current") not in (None, "")
                else state_change.get("current")
            ),
        }
        if bool(node.get("revert")):
            state_row["revert"] = True
        rows.append(omit_empty(state_row))

    state_changes = syn.get("state_changes") or []
    if isinstance(state_changes, list):
        for group_index, group in enumerate(state_changes):
            if len(rows) >= max_items:
                break
            if not isinstance(group, dict):
                continue
            contract = normalize_address(group.get("address"))
            contract_label, label_sanitized = label_for_address(syn, contract)
            for slot_index, slot in enumerate(group.get("slots") or []):
                if len(rows) >= max_items:
                    break
                if not isinstance(slot, dict):
                    continue
                rows.append(omit_empty({
                    "evidence_id": f"state_change:{contract}:{slot_index}",
                    "source": "top_level_state_changes",
                    "contract": contract,
                    "contract_label": contract_label,
                    "contract_label_sanitized": label_sanitized or None,
                    "slot_key": slot.get("key") or slot.get("slot_key"),
                    "prev": slot.get("prev"),
                    "current": slot.get("current"),
                    "group_index": group_index,
                }))
    return rows


def render_transfer_event_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_items: int = 180,
) -> List[Dict[str, Any]]:
    transfers: List[Dict[str, Any]] = []
    for e in render_event_view(syn, trace_index, max_events=max_items * 4):
        event_name = str(e.get("event") or "")
        sig = str(e.get("event_signature") or "")
        if event_name.lower() != "transfer" and not sig.lower().startswith("transfer("):
            continue
        params = e.get("args") or e.get("params") or {}
        from_addr = normalize_address(
            get_param_value(params, ["from", "sender", "src", "_from"])
        )
        to_addr = normalize_address(
            get_param_value(params, ["to", "receiver", "dst", "_to"])
        )
        amount = get_param_value(params, ["value", "amount", "wad", "_value"])
        normalized_amount = format_decimal(parse_decimal(amount))
        token = normalize_address(e.get("address"))
        token_label, token_label_sanitized = label_for_address(syn, token)
        transfers.append(omit_empty({
            "evidence_id": f"transfer_event:{e.get('id')}",
            "event_id": e.get("id"),
            "depth": e.get("depth"),
            "parent_id": e.get("parent_id"),
            "parent_function": e.get("parent_function"),
            "token": token,
            "token_label": token_label or e.get("address_label"),
            "token_label_sanitized": token_label_sanitized or None,
            "from": from_addr,
            "to": to_addr,
            "amount": normalized_amount if normalized_amount is not None else amount,
            "raw_event_evidence_id": e.get("evidence_id"),
        }))
        if len(transfers) >= max_items:
            break
    return transfers


# ============================================================
# External fundflow and profit/loss views
# ============================================================

def render_external_fundflow_view(fundflow_obj: Any, max_items: int = 240) -> Dict[str, Any]:
    if fundflow_obj is None:
        return {"available": False, "reason": "fundflow file not found"}

    view: Dict[str, Any] = {
        "available": True,
        "source_type": type(fundflow_obj).__name__,
        "records": [],
        "raw_top_level_keys": list(fundflow_obj.keys()) if isinstance(fundflow_obj, dict) else [],
    }

    records = None
    selected_key = None
    if isinstance(fundflow_obj, dict):
        for key in (
            "transfers",
            "fund_flow",
            "fundFlows",
            "flows",
            "net_flows",
            "netFundFlows",
            "net_fund_flow",
            "netTransfers",
            "records",
            "items",
        ):
            if isinstance(fundflow_obj.get(key), list):
                selected_key = key
                records = fundflow_obj[key]
                break
        if records is None:
            view["summary"] = compact_value(fundflow_obj, max_dict_items=40)
            return view
    elif isinstance(fundflow_obj, list):
        records = fundflow_obj
    else:
        view["summary"] = compact_value(fundflow_obj)
        return view

    if selected_key:
        view["record_key"] = selected_key

    for i, record in enumerate(records[:max_items]):
        if isinstance(record, dict):
            rid = record.get("id", i)
            row = {"evidence_id": f"fundflow:{rid}", **compact_value(record)}
        else:
            row = {"evidence_id": f"fundflow:{i}", "value": compact_value(record)}
        view["records"].append(row)

    if len(records) > max_items:
        view["truncated"] = {
            "total": len(records),
            "shown": max_items,
            "omitted": len(records) - max_items,
        }
    return view


def render_profit_loss_view(profit_loss_obj: Any) -> Dict[str, Any]:
    note = (
        "This view is only a candidate hint. Judge must not infer value "
        "extraction solely from top profit/loss if the address also contributed "
        "comparable value; use participant_net_delta_view and "
        "contribution_vs_payout_view for accounting context."
    )
    if _is_data_not_available(profit_loss_obj):
        return {
            "available": False,
            "reason": "profit_loss data reported as not available by data source",
            "note": note,
        }
    if profit_loss_obj is None:
        return {
            "available": True,
            "candidate_present": False,
            "top_profit_address": None,
            "top_loss_address": None,
            "raw": None,
            "note": (
                "No top-profit-loss candidates were found or provided. Treat this "
                "as absence of this candidate hint, not as missing packet evidence "
                "or runtime failure. " + note
            ),
        }

    if isinstance(profit_loss_obj, dict):
        top_profit = (
            profit_loss_obj.get("topProfitAddress")
            or profit_loss_obj.get("top_profit_address")
            or profit_loss_obj.get("top_profit")
        )
        top_loss = (
            profit_loss_obj.get("topLossAddress")
            or profit_loss_obj.get("top_loss_address")
            or profit_loss_obj.get("top_loss")
        )
        return {
            "available": True,
            "candidate_present": bool(top_profit or top_loss),
            "top_profit_address": normalize_address(top_profit),
            "top_loss_address": normalize_address(top_loss),
            "raw": compact_value(profit_loss_obj),
            "note": note,
        }

    return {
        "available": True,
        "candidate_present": bool(profit_loss_obj),
        "raw": compact_value(profit_loss_obj),
        "note": note,
    }


# ============================================================
# New packet views
# ============================================================

def render_evidence_adequacy_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    packet_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {**DEFAULT_PACKET_CONFIG, **(packet_config or {})}
    meta = syn.get("metadata") if isinstance(syn.get("metadata"), dict) else {}
    counts = trace_counts(trace_index)
    total_nodes = int(meta.get("total_trace_nodes") or counts.get("total_nodes", 0))
    shown = min(total_nodes, int(cfg.get("max_trace_nodes", 420)))
    call_nodes = int(meta.get("call_nodes") or counts.get("call_nodes", 0))
    decoded = int(meta.get("xhr_decoded_call_nodes") or counts.get("decoded_call_nodes", 0))
    unknown = max(0, call_nodes - decoded)
    semantic_counts = _state_semantic_count_summary(syn)
    return {
        "trace": {
            "total_nodes": total_nodes,
            "shown_nodes_in_trace_view": shown,
            "truncated": total_nodes > shown,
            "omitted_nodes": max(0, total_nodes - shown),
        },
        "decode_coverage": {
            "call_nodes": call_nodes,
            "decoded_call_nodes": decoded,
            "unknown_call_nodes": unknown,
            "unknown_selectors": unknown_selectors(trace_index, max_items=30),
        },
        "state_coverage": {
            "state_change_nodes": int(meta.get("sstore_nodes") or counts.get("sstore_nodes", 0)),
            "semantic_state_rows": semantic_counts["semantic_state_rows"],
            "raw_slot_only_rows": semantic_counts["raw_slot_only_rows"],
            "state_changes_available": not semantic_counts.get("not_available", False),
        },
        "profit_loss_available": not _is_data_not_available(syn.get("profit_loss")),
        "view_notes": [
            "evidence_adequacy_view is metadata only; it is not attack evidence.",
            "A truncated trace means omitted nodes were not shown in trace_view.",
            "Unknown selectors indicate missing ABI decode, not proof of benign or malicious behavior.",
        ],
    }


def _state_semantic_count_summary(syn: Dict[str, Any]) -> Dict[str, Any]:
    semantic = 0
    raw = 0
    state_changes = syn.get("state_changes") or []
    if _is_data_not_available(state_changes):
        return {"semantic_state_rows": 0, "raw_slot_only_rows": 0, "not_available": True}
    if isinstance(state_changes, list):
        for group in state_changes:
            if not isinstance(group, dict):
                continue
            vars_ = group.get("variables") or []
            if vars_:
                for var in vars_:
                    semantic += max(1, len(_flatten_variable_rows(var)))
            else:
                raw += len(group.get("slots") or [])
    return {"semantic_state_rows": semantic, "raw_slot_only_rows": raw}


def render_critical_call_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_items: int = 200,
) -> Any:
    """Return critical call rows, or an absence-summary dict when no rows matched."""
    reentrant_rows = _detect_reentrant_calls(syn, trace_index, max_reentrant=40)
    keyword_rows: List[Dict[str, Any]] = []
    reentrant_node_keys = {r.get("evidence_id") for r in reentrant_rows}
    keyword_candidates: List[Tuple[int, int, Any, Dict[str, Any], str]] = []

    for sequence_index, (node_key, node) in enumerate(iter_trace_nodes(trace_index)):
        if not is_call_like(node):
            continue
        eid = node_evidence_id(trace_index, node_key)
        if eid in reentrant_node_keys:
            continue
        why = critical_call_reason(trace_index, node_key, node)
        if not why:
            continue
        keyword_candidates.append((
            _critical_call_candidate_score(node, why),
            sequence_index,
            node_key,
            node,
            why,
        ))

    keyword_candidates.sort(key=lambda item: (-item[0], item[1]))
    remaining_budget = max(0, int(max_items) - len(reentrant_rows))
    selected_candidates = _select_diverse_critical_call_candidates(
        keyword_candidates,
        max_items=remaining_budget,
    )
    for _, _, node_key, node, why in selected_candidates:
        eid = node_evidence_id(trace_index, node_key)
        addr_label, addr_label_sanitized = label_for_node(
            syn, node.get("address"), node.get("address_label")
        )
        caller_label, caller_label_sanitized = label_for_node(
            syn, node.get("caller"), node.get("caller_label")
        )
        row = {
            "evidence_id": eid,
            "id": trace_index.display_id_by_id.get(node_key, node.get("id", node_key)),
            "type": get_node_type(node),
            "call_type": node.get("call_type"),
            **parent_context(trace_index, node_key),
            "caller": node.get("caller"),
            "caller_label": caller_label,
            "caller_label_sanitized": caller_label_sanitized or None,
            "callee": node.get("address"),
            "address_label": addr_label,
            "address_label_sanitized": addr_label_sanitized or None,
            "function": get_function_name(node),
            "selector": get_selector(node),
            "value": node.get("value"),
            "nearby_event_ids": collect_nearby_evidence_ids(trace_index, node_key, target="event"),
            "nearby_state_ids": collect_nearby_evidence_ids(trace_index, node_key, target="state"),
            "why_included": why,
        }
        keyword_rows.append(omit_empty(row))

    rows = (reentrant_rows + keyword_rows)[:max_items]
    rows.sort(
        key=lambda row: (
            -_critical_rendered_row_score(row),
            str(row.get("id") or ""),
        )
    )

    if rows:
        return rows

    pkt_truncated = False
    meta = syn.get("metadata") if isinstance(syn.get("metadata"), dict) else {}
    total_nodes = meta.get("total_trace_nodes")
    if isinstance(total_nodes, (int, float)):
        pkt_truncated = int(total_nodes) > int(DEFAULT_PACKET_CONFIG.get("max_trace_nodes", 420))

    return {
        "available": True,
        "rows": [],
        "summary": {
            "matched_critical_call_count": 0,
            "searched_patterns": list(CRITICAL_CALL_KEYWORDS)[:20],
            "absence_interpretation": (
                "No critical price-manipulation-related calls matched in a complete trace packet."
                if not pkt_truncated
                else "Trace may be truncated; absence is not confirmed."
            ),
            "packet_trace_truncated": pkt_truncated,
        },
    }


def _critical_call_candidate_score(node: Dict[str, Any], reason: str) -> int:
    """Prioritize actionable calls before bounded critical-view selection."""
    function = get_function_name(node).lower()
    call_type = str(node.get("call_type") or get_node_type(node)).lower()
    reason_lower = str(reason or "").lower()
    score = 0
    if "unknown selector" in reason_lower:
        score += 100
    for keyword in (
        "withdraw",
        "redeem",
        "borrow",
        "repay",
        "liquidat",
        "flash",
        "callback",
        "mint",
        "burn",
        "claim",
        "harvest",
        "transferfrom",
        "approve",
        "exchange",
        "swap",
        "add_liquidity",
        "remove_liquidity",
        "upgrade",
        "admin",
        "set_rate",
    ):
        if keyword in function:
            score += 60
            break
    if call_type == "call":
        score += 15
    if parse_decimal(node.get("value")) not in (None, Decimal(0)):
        score += 25
    if any(
        keyword in function
        for keyword in (
            "price_oracle",
            "latestanswer",
            "latestrounddata",
            "getreserves",
            "consult",
            "peek",
        )
    ):
        score += 10
    if call_type == "staticcall":
        score -= 20
    if any(keyword in function for keyword in ("balanceof", "totalsupply")):
        score -= 45
    if reason_lower == "call_type: delegatecall":
        score -= 15
    return score


def _critical_call_candidate_category(
    node: Dict[str, Any],
    reason: str,
) -> str:
    function = get_function_name(node).lower()
    reason_lower = str(reason or "").lower()
    if any(token in function for token in ("callback", "hook", "fallback")):
        return "callback"
    if any(
        token in function
        for token in (
            "oracle",
            "price",
            "latestanswer",
            "latestrounddata",
            "getreserves",
            "consult",
            "peek",
        )
    ):
        return "oracle"
    if any(
        token in function
        for token in (
            "share",
            "reward",
            "debt",
            "collateral",
            "position",
            "account",
            "accrue",
            "index",
        )
    ):
        return "accounting"
    if any(
        token in function
        for token in (
            "withdraw",
            "redeem",
            "borrow",
            "repay",
            "liquidat",
            "mint",
            "burn",
            "claim",
            "harvest",
            "transferfrom",
            "approve",
        )
    ):
        return "value_or_privileged"
    if any(
        token in function
        for token in (
            "swap",
            "exchange",
            "liquidity",
            "flash",
        )
    ):
        return "market_or_atomic"
    if (
        any(token in function for token in ("owner", "admin", "role", "upgrade"))
        or function.split("(", 1)[0].startswith("set")
    ):
        return "authorization"
    if "unknown selector" in reason_lower:
        return "unknown_selector"
    if str(node.get("call_type") or "").lower() == "delegatecall":
        return "delegation"
    return "other"


def _select_diverse_critical_call_candidates(
    candidates: List[Tuple[int, int, Any, Dict[str, Any], str]],
    *,
    max_items: int,
) -> List[Tuple[int, int, Any, Dict[str, Any], str]]:
    """Keep category anchors before filling the remaining ranked budget."""
    if max_items <= 0:
        return []
    selected: List[Tuple[int, int, Any, Dict[str, Any], str]] = []
    selected_keys: set[Any] = set()
    seen_categories: set[str] = set()
    for candidate in candidates:
        category = _critical_call_candidate_category(candidate[3], candidate[4])
        if category in seen_categories:
            continue
        selected.append(candidate)
        selected_keys.add(candidate[2])
        seen_categories.add(category)
        if len(selected) >= max_items:
            return selected

    unknown_cap = max(1, max_items // 4)
    category_counts = Counter(
        _critical_call_candidate_category(candidate[3], candidate[4])
        for candidate in selected
    )
    for candidate in candidates:
        if candidate[2] in selected_keys:
            continue
        category = _critical_call_candidate_category(candidate[3], candidate[4])
        if category == "unknown_selector" and category_counts[category] >= unknown_cap:
            continue
        selected.append(candidate)
        selected_keys.add(candidate[2])
        category_counts[category] += 1
        if len(selected) >= max_items:
            break
    return selected


def _structural_reentry_shape_kind(node: Dict[str, Any]) -> str:
    function = get_function_name(node).lower()
    call_type = str(node.get("call_type") or get_node_type(node)).lower()
    if call_type == "staticcall" or any(
        keyword in function
        for keyword in (
            "balanceof",
            "totalsupply",
            "total_debt",
            "getusdprice",
            "price_oracle",
            "latestanswer",
            "latestrounddata",
        )
    ):
        return "read_only_nested_shape"
    if any(
        keyword in function
        for keyword in (
            "receiveflashloan",
            "executeoperation",
            "swapcallback",
            "uniswapv2call",
            "pancakecall",
        )
    ):
        return "standard_callback_shape"
    return "stateful_reentry_lead"


def _critical_rendered_row_score(row: Dict[str, Any]) -> int:
    reason = str(row.get("why_included") or "")
    score = _critical_call_candidate_score(row, reason)
    shape_kind = str(row.get("structural_reentry_kind") or "")
    if reason.startswith("reentrant_call"):
        score += 80
    if shape_kind == "read_only_nested_shape":
        score -= 120
    elif shape_kind == "standard_callback_shape":
        score -= 70
    elif shape_kind == "stateful_reentry_lead":
        score += 40
    return score


def render_critical_call_argument_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    critical_call_view: Optional[List[Dict[str, Any]]] = None,
    max_items: int = 80,
) -> List[Dict[str, Any]]:
    """Expose decoded arguments only for calls already selected as critical."""
    critical_rows = critical_call_view
    if critical_rows is None:
        rendered = render_critical_call_view(
            syn,
            trace_index,
            max_items=max_items,
        )
        critical_rows = rendered if isinstance(rendered, list) else []

    nodes_by_evidence_id = {
        node_evidence_id(trace_index, node_key): (node_key, node)
        for node_key, node in iter_trace_nodes(trace_index)
        if is_call_like(node)
    }
    rows: List[Dict[str, Any]] = []
    for critical in list(critical_rows or []):
        if len(rows) >= max_items:
            break
        evidence_id = str(critical.get("evidence_id") or "")
        node_entry = nodes_by_evidence_id.get(evidence_id)
        if node_entry is None:
            continue
        node_key, node = node_entry
        params = node.get("params") or node.get("args")
        args_in = node.get("args_in")
        return_values = node.get("return_values") or node.get("returns")
        price_consumption = _price_read_consumption_context(
            trace_index,
            node_key,
            node,
        )
        if not params and not args_in and not return_values and not price_consumption:
            continue
        rows.append(omit_empty({
            "evidence_id": evidence_id,
            "id": trace_index.display_id_by_id.get(
                node_key,
                node.get("id", node_key),
            ),
            **parent_context(trace_index, node_key),
            "caller": node.get("caller"),
            "callee": node.get("address"),
            "function": get_function_name(node),
            "selector": get_selector(node),
            "params": compact_value(
                params,
                max_str_len=500,
                max_list_len=20,
                max_dict_items=24,
            ),
            "args_in": compact_value(
                args_in if not params else None,
                max_str_len=500,
                max_list_len=20,
                max_dict_items=24,
            ),
            "return_values": compact_value(
                return_values,
                max_str_len=500,
                max_list_len=12,
                max_dict_items=16,
            ),
            "price_read_consumption_link": price_consumption,
        }))
    return rows


def _price_read_consumption_context(
    trace_index: TraceIndex,
    node_key: Any,
    node: Dict[str, Any],
) -> Dict[str, Any]:
    function = get_function_name(node).lower()
    if not any(
        keyword in function
        for keyword in (
            "price",
            "oracle",
            "consult",
            "peek",
            "latestanswer",
            "latestrounddata",
            "getreserves",
            "get_reserves",
            "get_p",
            "virtualprice",
            "virtual_price",
            "exchangerate",
            "exchange_rate",
        )
    ):
        return {}
    parent_key = trace_index.parent_by_id.get(node_key)
    parent = trace_index.node_by_id.get(parent_key, {}) if parent_key is not None else {}
    if not isinstance(parent, dict) or not parent:
        return {}
    return omit_empty({
        "read_evidence_id": node_evidence_id(trace_index, node_key),
        "read_function": get_function_name(node),
        "raw_output": compact_value(
            node.get("output") if not (node.get("return_values") or node.get("returns")) else None,
            max_str_len=300,
        ),
        "immediate_consumer_evidence_id": node_evidence_id(trace_index, parent_key),
        "immediate_consumer_function": get_function_name(parent),
        "immediate_consumer_address": parent.get("address"),
        "relation": "nested_read_return_available_to_immediate_parent",
        "limitation": (
            "structural parent/child linkage only; packet does not prove which "
            "parent expression consumed the returned value"
        ),
    })


def _detect_reentrant_calls(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_reentrant: int = 40,
) -> List[Dict[str, Any]]:
    """Structural reentrancy detection: find calls that re-enter a contract already active in the call stack.

    Tracks both caller and callee per frame so that A->B->A (where B calls back into A)
    is detected: frame 0 pushes (caller=A, callee=B), frame 1 sees callee=A which
    matches frame 0's caller A.
    """
    frame_stack: List[Tuple[int, str, str, Any]] = []  # (depth, caller, callee, display_id)
    rows: List[Dict[str, Any]] = []
    for node_key, node in iter_trace_nodes(trace_index):
        if len(rows) >= max_reentrant:
            break
        node_type = get_node_type(node)
        depth = trace_index.depth_by_id.get(node_key, 0)
        callee = normalize_address(node.get("address"))
        caller = normalize_address(node.get("caller"))
        display_id = trace_index.display_id_by_id.get(node_key, node.get("id", node_key))

        while frame_stack and frame_stack[-1][0] >= depth:
            frame_stack.pop()

        if is_call_like(node) and callee:
            active_addrs: set = set()
            for _, frame_caller, frame_callee, _ in frame_stack:
                active_addrs.add(frame_caller)
                active_addrs.add(frame_callee)
            if callee in active_addrs:
                addr_label, addr_label_sanitized = label_for_node(
                    syn, node.get("address"), node.get("address_label")
                )
                caller_label, caller_label_sanitized = label_for_node(
                    syn, node.get("caller"), node.get("caller_label")
                )
                first_idx = next(
                    (i for i, (d, fc, fce, _) in enumerate(frame_stack) if callee in (fc, fce)),
                    len(frame_stack),
                )
                first_depth = frame_stack[first_idx][0] if first_idx < len(frame_stack) else depth
                first_entry_id = frame_stack[first_idx][3] if first_idx < len(frame_stack) else None
                chain_ids = [fid for d, fc, fce, fid in frame_stack[first_idx:]]
                rows.append(omit_empty({
                    "evidence_id": node_evidence_id(trace_index, node_key),
                    "id": display_id,
                    "type": node_type,
                    "call_type": node.get("call_type"),
                    **parent_context(trace_index, node_key),
                    "caller": node.get("caller"),
                    "caller_label": caller_label,
                    "caller_label_sanitized": caller_label_sanitized or None,
                    "callee": node.get("address"),
                    "address_label": addr_label,
                    "address_label_sanitized": addr_label_sanitized or None,
                    "function": get_function_name(node),
                    "selector": get_selector(node),
                    "value": node.get("value"),
                    "nearby_event_ids": collect_nearby_evidence_ids(trace_index, node_key, target="event"),
                    "nearby_state_ids": collect_nearby_evidence_ids(trace_index, node_key, target="state"),
                    "why_included": "reentrant_call: callee is already active in call stack",
                    "structural_reentry_kind": _structural_reentry_shape_kind(node),
                    "reentrant_depth_gap": depth - first_depth,
                    "first_entry_id": first_entry_id,
                    "reentrant_chain_ids": chain_ids if chain_ids else None,
                }))

        if is_call_like(node):
            frame_stack.append((depth, caller or "", callee or "", display_id))

    return rows


def render_reentrancy_state_order_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    critical_call_view: List[Dict[str, Any]],
    max_items: int = 80,
    max_state_nodes_per_item: int = 18,
    include_details: bool = False,
) -> Dict[str, Any]:
    """Neutral state read/write ordering around structurally detected reentrant calls."""
    reentrant_rows = [
        row
        for row in list(critical_call_view or [])
        if isinstance(row, dict)
        and str(row.get("why_included", "")).startswith("reentrant_call")
    ]
    out: List[Dict[str, Any]] = []
    for row in reentrant_rows[:max_items]:
        call_id = row.get("id")
        node_key = find_node_key_by_display_id(trace_index, call_id)
        chain_ids = list(row.get("reentrant_chain_ids", []) or [])
        chain_keys = [
            key
            for key in (find_node_key_by_display_id(trace_index, item) for item in chain_ids)
            if key is not None
        ]
        state_keys = collect_state_order_keys_for_reentrant_candidate(
            trace_index,
            node_key=node_key,
            chain_keys=chain_keys,
            max_items=max_state_nodes_per_item,
        )
        state_rows = [
            render_reentrancy_state_access_row(syn, trace_index, state_key)
            for state_key in state_keys
        ]
        state_rows = [item for item in state_rows if item]

        state_order_evidence_ids = [
            item.get("evidence_id") for item in state_rows if item.get("evidence_id")
        ]
        phase_context = build_reentrancy_phase_context(
            syn,
            trace_index,
            outer_call_id=row.get("first_entry_id"),
            reentry_call_id=call_id,
        )
        candidate_validity = _reentrancy_structural_candidate_validity(
            phase_context
        )
        candidate = omit_empty({
            "evidence_id": f"reentrancy_state_order:call:{call_id}",
            "candidate_id": phase_context.get("candidate_id"),
            "source_call_evidence_id": row.get("evidence_id"),
            "call_id": call_id,
            "function": row.get("function"),
            "callee": row.get("callee"),
            "callee_alias": row.get("callee_alias"),
            "address_label": row.get("address_label"),
            "parent_id": row.get("parent_id"),
            "parent_function": row.get("parent_function"),
            "depth": row.get("depth"),
            "path_ids": row.get("path_ids"),
            "call_path_tail": row.get("call_path_tail") if row.get("call_path_tail") else None,
            "why_included": row.get("why_included"),
            "first_entry_id": row.get("first_entry_id"),
            "reentrant_depth_gap": row.get("reentrant_depth_gap"),
            "reentrant_chain_ids": chain_ids,
            "state_order": state_rows if include_details else None,
            "state_order_evidence_ids": state_order_evidence_ids,
            "state_access_count": len(state_rows),
            "slot_access_summary": summarize_state_slot_accesses(state_rows),
            "outer_call_id": phase_context.get("outer_call_id"),
            "outer_function": phase_context.get("outer_function"),
            "external_edge_id": phase_context.get("external_edge_id"),
            "external_edge_function": phase_context.get("external_edge_function"),
            "reentry_call_id": phase_context.get("reentry_call_id"),
            "reentry_function": phase_context.get("reentry_function"),
            "callback_kind": phase_context.get("callback_kind"),
            "outer_call_type": phase_context.get("outer_call_type"),
            "external_edge_call_type": phase_context.get(
                "external_edge_call_type"
            ),
            "reentry_call_type": phase_context.get("reentry_call_type"),
            **candidate_validity,
            "logical_storage_context": phase_context.get("logical_storage_context"),
            "phase_boundaries": phase_context.get("phase_boundaries"),
            "phase_state_access_summary": phase_context.get(
                "phase_state_access_summary"
            ),
            "phase_slot_witnesses": phase_context.get("phase_slot_witnesses"),
            "phase_witness_completeness": phase_context.get(
                "phase_witness_completeness"
            ),
            "nearby_event_ids": row.get("nearby_event_ids"),
            "nearby_state_ids": row.get("nearby_state_ids"),
            "notes": (
                [
                    "No nearby sload/sstore rows were found for this reentrant candidate in the packet; this is an evidence availability note, not a benign or attack judgment."
                ]
                if not state_rows else []
            ),
        })
        out.append(candidate)

    rows_with_state = sum(1 for item in out if item.get("state_access_count", 0))
    state_access_rows = sum(int(item.get("state_access_count", 0) or 0) for item in out)
    slot_summaries = sum(len(item.get("slot_access_summary", []) or []) for item in out)
    phase_patterns = Counter(
        str(witness.get("order_pattern") or "")
        for item in out
        for witness in list(item.get("phase_slot_witnesses", []) or [])
        if str(witness.get("order_pattern") or "")
    )
    return {
        "available": True,
        "summary": {
            "reentrant_candidate_count": len(reentrant_rows),
            "rows_returned": len(out),
            "rows_with_state_access": rows_with_state,
            "state_access_rows": state_access_rows,
            "slot_access_summary_count": slot_summaries,
            "phase_order_pattern_counts": dict(phase_patterns),
            "max_state_nodes_per_row": max_state_nodes_per_item,
            "interpretation_note": (
                "Neutral phase-aware sload/sstore summary around structurally "
                "detected reentrant calls. An SSTORE completed before the external "
                "edge is visible to nested EVM execution and is a safe-order signal, "
                "not stale-state proof. This view does not determine exploitability."
            ),
        },
        "rows": out,
    }


def render_reentrancy_state_order_summary_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    critical_call_view: List[Dict[str, Any]],
    max_items: int = 16,
) -> Dict[str, Any]:
    """Compact first-pass summary for reentrancy state-order evidence."""
    detailed_view = render_reentrancy_state_order_view(
        syn=syn,
        trace_index=trace_index,
        critical_call_view=critical_call_view,
        # Rank a wider structural pool before selecting the compact first-pass
        # rows. Trace-order selection otherwise lets read-only recursion crowd
        # out later stateful candidates.
        max_items=max(max_items, min(64, max_items * 4)),
        max_state_nodes_per_item=int(DEFAULT_PACKET_CONFIG["max_reentrancy_state_nodes_per_row"]),
        include_details=False,
    )
    compact_rows: List[Dict[str, Any]] = []
    for raw_row in list(detailed_view.get("rows", []) or []):
        if not isinstance(raw_row, dict):
            continue
        state_ids = list(raw_row.get("state_order_evidence_ids", []) or [])
        chain_ids = list(raw_row.get("reentrant_chain_ids", []) or [])
        slot_summaries: List[Dict[str, Any]] = []
        for raw_slot in list(raw_row.get("slot_access_summary", []) or [])[:4]:
            if not isinstance(raw_slot, dict):
                continue
            accesses = [
                item
                for item in list(raw_slot.get("accesses", []) or [])
                if isinstance(item, dict)
            ]
            order_indexes = [
                int(item["order_index"])
                for item in accesses
                if isinstance(item.get("order_index"), int)
            ]
            access_types = [
                str(item.get("type") or "")
                for item in accesses
                if str(item.get("type") or "")
            ]
            access_ids = list(raw_slot.get("access_evidence_ids", []) or [])
            slot_summaries.append(omit_empty({
                "slot_key_short": raw_slot.get("slot_key_short"),
                "access_count": int(raw_slot.get("access_count", 0) or 0),
                "read_count": int(raw_slot.get("read_count", 0) or 0),
                "write_count": int(raw_slot.get("write_count", 0) or 0),
                "first_order_index": min(order_indexes) if order_indexes else None,
                "last_order_index": max(order_indexes) if order_indexes else None,
                "access_pattern": access_types[:4],
                "representative_evidence_ids": access_ids[:2],
                "evidence_id_count": len(access_ids),
            }))

        compact_rows.append(omit_empty({
            "evidence_id": raw_row.get("evidence_id"),
            "candidate_id": raw_row.get("candidate_id"),
            "source_call_evidence_id": raw_row.get("source_call_evidence_id"),
            "call_id": raw_row.get("call_id"),
            "function": raw_row.get("function"),
            "callee": raw_row.get("callee"),
            "callee_alias": raw_row.get("callee_alias"),
            "parent_id": raw_row.get("parent_id"),
            "parent_function": raw_row.get("parent_function"),
            "depth": raw_row.get("depth"),
            "first_entry_id": raw_row.get("first_entry_id"),
            "reentrant_depth_gap": raw_row.get("reentrant_depth_gap"),
            "reentrant_chain_ids": chain_ids[:6],
            "reentrant_chain_id_count": len(chain_ids),
            "outer_call_id": raw_row.get("outer_call_id"),
            "outer_function": raw_row.get("outer_function"),
            "external_edge_id": raw_row.get("external_edge_id"),
            "external_edge_function": raw_row.get("external_edge_function"),
            "reentry_call_id": raw_row.get("reentry_call_id"),
            "reentry_function": raw_row.get("reentry_function"),
            "callback_kind": raw_row.get("callback_kind"),
            "outer_call_type": raw_row.get("outer_call_type"),
            "external_edge_call_type": raw_row.get(
                "external_edge_call_type"
            ),
            "reentry_call_type": raw_row.get("reentry_call_type"),
            "candidate_tier": raw_row.get("candidate_tier"),
            "formal_candidate": raw_row.get("formal_candidate"),
            "structural_only_reasons": list(
                raw_row.get("structural_only_reasons", []) or []
            )[:4],
            "logical_storage_context": raw_row.get("logical_storage_context"),
            "phase_boundaries": raw_row.get("phase_boundaries"),
            "phase_state_access_summary": raw_row.get(
                "phase_state_access_summary"
            ),
            "phase_slot_witnesses": [
                {
                    **witness,
                    "evidence_ids": list(witness.get("evidence_ids", []) or [])[:6],
                }
                for witness in list(raw_row.get("phase_slot_witnesses", []) or [])[:6]
                if isinstance(witness, dict)
            ],
            "phase_witness_completeness": raw_row.get(
                "phase_witness_completeness"
            ),
            "state_access_count": int(raw_row.get("state_access_count", 0) or 0),
            "state_order_evidence_ids": state_ids[:4],
            "state_order_evidence_id_count": len(state_ids),
            "slot_access_summary": slot_summaries,
            "slot_access_summary_count": len(
                list(raw_row.get("slot_access_summary", []) or [])
            ),
            "nearby_event_ids": list(raw_row.get("nearby_event_ids", []) or [])[:4],
            "nearby_state_ids": list(raw_row.get("nearby_state_ids", []) or [])[:4],
            "notes": list(raw_row.get("notes", []) or [])[:2],
        }))

    compact_rows.sort(
        key=lambda item: (
            0 if item.get("formal_candidate") else 1,
            0 if item.get("candidate_tier") == "state_order_candidate" else 1,
            -int(item.get("state_access_count", 0) or 0),
            str(item.get("candidate_id") or ""),
        )
    )
    compact_rows = compact_rows[:max_items]
    summary = dict(detailed_view.get("summary", {}) or {})
    summary.update({
        "interpretation_note": (
            "Compact first-pass phase-aware state-order summary. Per-access objects "
            "are omitted; request reentrancy_state_order_view or local call context "
            "for detail. A completed pre-edge SSTORE is not stale-state evidence."
        ),
        "view_kind": "summary",
        "rows_returned": len(compact_rows),
        "formal_candidate_count": sum(
            1 for row in compact_rows if row.get("formal_candidate")
        ),
        "structural_only_count": sum(
            1 for row in compact_rows if not row.get("formal_candidate")
        ),
        "detail_rows_inlined": False,
    })
    return {
        "available": detailed_view.get("available", True),
        "summary": summary,
        "rows": compact_rows,
    }


def render_reentrancy_candidate_catalog_view(
    reentrancy_state_order_summary_view: Any,
    value_release_view: Any,
    max_items: int = 24,
) -> Dict[str, Any]:
    """Rank stable structural candidates without inlining phase/state payloads."""
    summary_rows = (
        list(reentrancy_state_order_summary_view.get("rows", []) or [])
        if isinstance(reentrancy_state_order_summary_view, dict)
        else list(reentrancy_state_order_summary_view or [])
    )
    release_rows = (
        list(value_release_view or [])
        if isinstance(value_release_view, list)
        else list((value_release_view or {}).get("rows", []) or [])
        if isinstance(value_release_view, dict)
        else []
    )
    compact_rows: List[Dict[str, Any]] = []
    for raw in summary_rows:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        reentry_id = str(raw.get("reentry_call_id") or "").strip()
        related_releases = []
        for release in release_rows:
            if not isinstance(release, dict):
                continue
            release_path = {
                str(value)
                for value in [
                    *list(release.get("path_ids", []) or []),
                    release.get("call_id"),
                    release.get("parent_id"),
                ]
                if value not in (None, "")
            }
            if reentry_id and reentry_id in release_path:
                related_releases.append(release)

        callback_kind = str(raw.get("callback_kind") or "unknown_callback")
        score = 0
        rank_reasons: List[str] = []
        callback_scores = {
            "same_function": 10,
            "cross_function": 8,
            "token_hook": 8,
            "fallback_or_receive": 8,
            "read_only_callback": 6,
            "unknown_callback": 4,
            "flashloan_callback": 1,
            "dex_swap_callback": 1,
        }
        callback_score = callback_scores.get(callback_kind, 3)
        score += callback_score
        rank_reasons.append(f"callback_kind:{callback_kind}")
        release_count = sum(
            int(item.get("release_record_count", 0) or 0)
            for item in related_releases
        )
        repeated_release_count = sum(
            int(item.get("repeated_release_count", 0) or 0)
            for item in related_releases
        )
        if release_count:
            score += 8 + min(release_count, 5)
            rank_reasons.append("candidate_path_value_release")
        if repeated_release_count:
            score += 8 + min(repeated_release_count, 5)
            rank_reasons.append("candidate_path_repeated_release")

        phase_counts = dict(raw.get("phase_state_access_summary") or {})
        witness_rows = [
            item for item in list(raw.get("phase_slot_witnesses", []) or [])
            if isinstance(item, dict)
        ]
        patterns = _unique(
            item.get("order_pattern") for item in witness_rows
        )
        if any(
            pattern in {
                "outer_read_inner_access_outer_write_after_external",
                "inner_access_before_outer_update",
            }
            for pattern in patterns
        ):
            score += 8
            rank_reasons.append("delayed_or_inner_before_outer_update")
        if int(raw.get("reentrant_depth_gap", 0) or 0) > 1:
            score += 2
            rank_reasons.append("deep_nested_entry")

        structural = _reentrancy_structural_candidate_validity(raw)
        reentry_call_type = str(raw.get("reentry_call_type") or "").lower()
        read_only = (
            reentry_call_type == "staticcall"
            or callback_kind == "read_only_callback"
        )
        callback_only = callback_kind in {
            "flashloan_callback",
            "dex_swap_callback",
        }
        delayed_state = any(
            pattern in {
                "outer_read_inner_access_outer_write_after_external",
                "inner_access_before_outer_update",
            }
            for pattern in patterns
        )
        formal_candidate = bool(
            not read_only
            and (
                delayed_state
                or release_count > 0
                or repeated_release_count > 0
            )
            and not (callback_only and not release_count and not repeated_release_count)
        )
        if formal_candidate:
            score += 12
            rank_reasons.append("formal_candidate_gate_passed")
        else:
            score -= 12
            rank_reasons.append("structural_shape_only")

        evidence_ids = _unique([
            raw.get("evidence_id"),
            raw.get("source_call_evidence_id"),
            *list(raw.get("state_order_evidence_ids", []) or []),
            *[
                value
                for item in related_releases
                for value in [
                    item.get("evidence_id"),
                    *list(item.get("value_out_evidence_ids", []) or []),
                ]
            ],
        ])
        compact_rows.append(omit_empty({
            "evidence_id": f"reentrancy_candidate_catalog:{candidate_id}",
            "candidate_id": candidate_id,
            "rank_score": score,
            "rank_reasons": rank_reasons[:5],
            "outer_call_id": raw.get("outer_call_id"),
            "outer_function": raw.get("outer_function"),
            "external_edge_id": raw.get("external_edge_id"),
            "external_edge_function": raw.get("external_edge_function"),
            "reentry_call_id": raw.get("reentry_call_id"),
            "reentry_function": raw.get("reentry_function"),
            "callback_kind": callback_kind,
            "outer_call_type": raw.get("outer_call_type"),
            "external_edge_call_type": raw.get("external_edge_call_type"),
            "reentry_call_type": raw.get("reentry_call_type"),
            "candidate_tier": (
                "effect_linked_candidate"
                if formal_candidate and (release_count or repeated_release_count)
                else structural.get("candidate_tier")
            ),
            "formal_candidate": formal_candidate,
            "structural_only_reasons": (
                []
                if formal_candidate
                else structural.get("structural_only_reasons", [])
            ),
            "logical_storage_context": raw.get("logical_storage_context"),
            "path_ids": list(raw.get("reentrant_chain_ids", []) or [])[:8],
            "phase_access_counts": phase_counts,
            "phase_order_patterns": patterns[:4],
            "phase_witness_completeness": raw.get(
                "phase_witness_completeness"
            ),
            "candidate_path_value_release_count": release_count,
            "candidate_path_repeated_release_count": repeated_release_count,
            "anchor_evidence_ids": evidence_ids[:7],
            "anchor_evidence_id_count": len(evidence_ids),
        }))

    compact_rows.sort(
        key=lambda row: (
            0 if row.get("formal_candidate") else 1,
            -int(row.get("rank_score", 0) or 0),
            str(row.get("candidate_id") or ""),
        )
    )
    rows = compact_rows[: max(1, int(max_items or 1))]
    return {
        "available": True,
        "summary": {
            "candidate_count": len(compact_rows),
            "formal_candidate_count": sum(
                1 for row in compact_rows if row.get("formal_candidate")
            ),
            "structural_only_count": sum(
                1 for row in compact_rows if not row.get("formal_candidate")
            ),
            "rows_returned": len(rows),
            "ranking_policy": (
                "candidate-local value/repeated release, delayed state order, "
                "sensitive callback kind, then stable candidate_id"
            ),
            "stable_id_policy": (
                "Judge must select candidate_id values present in this catalog; "
                "invented candidate IDs are invalid."
            ),
        },
        "rows": rows,
    }


def find_node_key_by_display_id(trace_index: TraceIndex, display_id: Any) -> Any:
    normalized = _normalize_display_id(display_id)
    for node_key, value in trace_index.display_id_by_id.items():
        if _normalize_display_id(value) == normalized:
            return node_key
    return display_id if display_id in trace_index.node_by_id else None


def _normalize_display_id(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.lower().startswith(("call:", "sload:", "sstore:", "event:")):
            text = text.split(":", 1)[1]
        if text.isdigit():
            return int(text)
        return text
    return value


def collect_state_order_keys_for_reentrant_candidate(
    trace_index: TraceIndex,
    *,
    node_key: Any,
    chain_keys: List[Any],
    max_items: int,
) -> List[Any]:
    candidates: List[Any] = []
    if node_key is not None:
        candidates.extend(collect_descendant_state_keys(trace_index, node_key, limit=max_items))
        candidates.extend(collect_nearby_state_keys(trace_index, node_key, radius=10))
    for key in list(chain_keys or []):
        if key is None or key == node_key:
            continue
        if trace_index.depth_by_id.get(key, 0) <= 1:
            continue
        candidates.extend(collect_nearby_state_keys(trace_index, key, radius=6))
    unique_sorted = sorted(
        {key for key in candidates if key is not None},
        key=lambda key: trace_index.node_index_by_id.get(key, 10**9),
    )
    return unique_sorted[:max_items]


def collect_descendant_state_keys(
    trace_index: TraceIndex,
    parent_key: Any,
    *,
    limit: int,
) -> List[Any]:
    out: List[Any] = []

    def visit(key: Any) -> None:
        if len(out) >= limit:
            return
        for child_key in trace_index.children_by_id.get(key, []):
            child = trace_index.node_by_id.get(child_key, {})
            if is_state_like(child):
                out.append(child_key)
                if len(out) >= limit:
                    return
            visit(child_key)
            if len(out) >= limit:
                return

    if parent_key is not None:
        visit(parent_key)
    return out


def collect_nearby_state_keys(
    trace_index: TraceIndex,
    node_key: Any,
    *,
    radius: int,
) -> List[Any]:
    if node_key is None:
        return []
    pos = trace_index.node_index_by_id.get(node_key)
    if pos is None:
        return []
    lo = max(0, pos - radius)
    hi = min(len(trace_index.sequence), pos + radius + 1)
    out = []
    for key in trace_index.sequence[lo:hi]:
        if key == node_key:
            continue
        node = trace_index.node_by_id.get(key, {})
        if is_state_like(node):
            out.append(key)
    return out


_REENTRANCY_PHASE_NAMES = (
    "outer_before_external",
    "nested_before_reentry",
    "reentry_inner",
    "nested_after_reentry",
    "outer_after_external",
)


def build_reentrancy_phase_context(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    *,
    outer_call_id: Any,
    reentry_call_id: Any,
) -> Dict[str, Any]:
    """Summarize state access by execution phase for one re-entry candidate."""
    reentry_key = find_node_key_by_display_id(trace_index, reentry_call_id)
    if reentry_key is None:
        return {
            "reentry_call_id": reentry_call_id,
            "phase_witness_completeness": "unavailable",
        }

    call_path_ids = list(trace_index.call_path_ids_by_id.get(reentry_key, []) or [])
    outer_key = find_node_key_by_display_id(trace_index, outer_call_id)
    if outer_key is None or outer_key == reentry_key:
        outer_key = _infer_reentrancy_outer_key(
            trace_index,
            call_path_ids=call_path_ids,
            reentry_key=reentry_key,
        )
    outer_display_id = (
        trace_index.display_id_by_id.get(outer_key)
        if outer_key is not None
        else outer_call_id
    )
    external_key = _infer_reentrancy_external_edge_key(
        trace_index,
        call_path_ids=call_path_ids,
        outer_key=outer_key,
        reentry_key=reentry_key,
    )
    external_display_id = (
        trace_index.display_id_by_id.get(external_key)
        if external_key is not None
        else reentry_call_id
    )

    outer_bounds = _trace_subtree_bounds(trace_index, outer_key)
    external_bounds = _trace_subtree_bounds(trace_index, external_key)
    reentry_bounds = _trace_subtree_bounds(trace_index, reentry_key)
    if not outer_bounds or not external_bounds or not reentry_bounds:
        return {
            "outer_call_id": outer_display_id,
            "external_edge_id": external_display_id,
            "reentry_call_id": reentry_call_id,
            "phase_witness_completeness": "unavailable",
        }

    outer_start, outer_end = outer_bounds
    external_start, external_end = external_bounds
    reentry_start, reentry_end = reentry_bounds
    phase_rows: Dict[str, List[Dict[str, Any]]] = {
        phase: [] for phase in _REENTRANCY_PHASE_NAMES
    }
    for node_key in trace_index.sequence[outer_start + 1:outer_end]:
        node = trace_index.node_by_id.get(node_key, {})
        if not isinstance(node, dict) or not is_state_like(node):
            continue
        position = trace_index.node_index_by_id.get(node_key, -1)
        phase = _reentrancy_phase_for_position(
            position,
            external_start=external_start,
            external_end=external_end,
            reentry_start=reentry_start,
            reentry_end=reentry_end,
        )
        if not phase:
            continue
        rendered = render_reentrancy_state_access_row(syn, trace_index, node_key)
        if rendered:
            phase_rows[phase].append(rendered)

    phase_summary = {
        phase: {
            "access_count": len(rows),
            "read_count": sum(1 for row in rows if row.get("type") == "sload"),
            "write_count": sum(1 for row in rows if row.get("type") == "sstore"),
            "representative_evidence_ids": [
                row.get("evidence_id")
                for row in rows
                if row.get("evidence_id")
            ][:4],
        }
        for phase, rows in phase_rows.items()
        if rows
    }
    witnesses = summarize_reentrancy_phase_slot_witnesses(phase_rows)

    outer_node = trace_index.node_by_id.get(outer_key, {}) if outer_key is not None else {}
    external_node = (
        trace_index.node_by_id.get(external_key, {})
        if external_key is not None
        else {}
    )
    reentry_node = trace_index.node_by_id.get(reentry_key, {})
    outer_function = get_function_name(outer_node)
    reentry_function = get_function_name(reentry_node)
    logical_storage_context = (
        normalize_address(reentry_node.get("address"))
        or normalize_address(outer_node.get("address"))
        or str(reentry_node.get("address") or outer_node.get("address") or "")
    )
    candidate_id = (
        f"re:{outer_display_id}:{external_display_id}:{reentry_call_id}"
    )
    return omit_empty({
        "candidate_id": candidate_id,
        "outer_call_id": outer_display_id,
        "outer_function": outer_function,
        "external_edge_id": external_display_id,
        "external_edge_function": get_function_name(external_node),
        "reentry_call_id": reentry_call_id,
        "reentry_function": reentry_function,
        "callback_kind": classify_reentrancy_callback_kind(
            outer_node=outer_node,
            external_node=external_node,
            reentry_node=reentry_node,
        ),
        "outer_call_type": str(
            outer_node.get("call_type") or get_node_type(outer_node) or ""
        ).lower(),
        "external_edge_call_type": str(
            external_node.get("call_type")
            or get_node_type(external_node)
            or ""
        ).lower(),
        "reentry_call_type": str(
            reentry_node.get("call_type") or get_node_type(reentry_node) or ""
        ).lower(),
        "logical_storage_context": logical_storage_context,
        "phase_boundaries": {
            "outer_start": outer_start,
            "external_edge_start": external_start,
            "reentry_start": reentry_start,
            "reentry_end": reentry_end - 1,
            "external_edge_end": external_end - 1,
            "outer_end": outer_end - 1,
        },
        "phase_state_access_summary": phase_summary,
        "phase_slot_witnesses": witnesses,
        "phase_witness_completeness": (
            "complete"
            if witnesses
            else "partial_no_same_slot_phase_witness"
            if phase_summary
            else "partial_no_state_access"
        ),
    })


def _reentrancy_structural_candidate_validity(
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Separate nested call shape from an exploit-relevant state-order lead."""
    callback_kind = str(candidate.get("callback_kind") or "unknown_callback")
    reentry_call_type = str(candidate.get("reentry_call_type") or "").lower()
    witnesses = [
        row
        for row in list(candidate.get("phase_slot_witnesses", []) or [])
        if isinstance(row, dict)
    ]
    delayed_state = any(
        str(row.get("order_pattern") or "")
        in {
            "outer_read_inner_access_outer_write_after_external",
            "inner_access_before_outer_update",
        }
        for row in witnesses
    )
    reasons: List[str] = []
    if reentry_call_type == "staticcall" or callback_kind == "read_only_callback":
        reasons.append("read_only_reentry")
    if callback_kind in {"flashloan_callback", "dex_swap_callback"}:
        reasons.append("standard_callback_shape_without_effect_link")
    if not delayed_state:
        reasons.append("no_delayed_same_slot_state_witness")
    formal_candidate = bool(not reasons)
    return {
        "candidate_tier": (
            "state_order_candidate" if formal_candidate else "structural_shape_only"
        ),
        "formal_candidate": formal_candidate,
        "structural_only_reasons": reasons,
    }


def _infer_reentrancy_outer_key(
    trace_index: TraceIndex,
    *,
    call_path_ids: List[Any],
    reentry_key: Any,
) -> Any:
    if len(call_path_ids) >= 3:
        return find_node_key_by_display_id(trace_index, call_path_ids[-3])
    if len(call_path_ids) >= 2:
        return find_node_key_by_display_id(trace_index, call_path_ids[-2])
    return trace_index.parent_call_by_id.get(reentry_key)


def _infer_reentrancy_external_edge_key(
    trace_index: TraceIndex,
    *,
    call_path_ids: List[Any],
    outer_key: Any,
    reentry_key: Any,
) -> Any:
    outer_display = (
        trace_index.display_id_by_id.get(outer_key)
        if outer_key is not None
        else None
    )
    normalized_outer = _normalize_display_id(outer_display)
    for index, display_id in enumerate(call_path_ids[:-1]):
        if _normalize_display_id(display_id) == normalized_outer:
            return find_node_key_by_display_id(trace_index, call_path_ids[index + 1])
    if len(call_path_ids) >= 2:
        return find_node_key_by_display_id(trace_index, call_path_ids[-2])
    return reentry_key


def _trace_subtree_bounds(
    trace_index: TraceIndex,
    node_key: Any,
) -> Optional[Tuple[int, int]]:
    if node_key is None:
        return None
    start = trace_index.node_index_by_id.get(node_key)
    if start is None:
        return None
    depth = trace_index.depth_by_id.get(node_key, 0)
    end = len(trace_index.sequence)
    for position in range(start + 1, len(trace_index.sequence)):
        candidate_key = trace_index.sequence[position]
        if trace_index.depth_by_id.get(candidate_key, 0) <= depth:
            end = position
            break
    return start, end


def _reentrancy_phase_for_position(
    position: int,
    *,
    external_start: int,
    external_end: int,
    reentry_start: int,
    reentry_end: int,
) -> str:
    if position < external_start:
        return "outer_before_external"
    if position < reentry_start:
        return "nested_before_reentry"
    if position < reentry_end:
        return "reentry_inner"
    if position < external_end:
        return "nested_after_reentry"
    return "outer_after_external"


def summarize_reentrancy_phase_slot_witnesses(
    phase_rows: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for phase, rows in phase_rows.items():
        for row in rows:
            slot = str(row.get("slot_key") or "")
            if not slot:
                continue
            contract = normalize_address(row.get("contract")) or str(
                row.get("contract") or ""
            )
            grouped[(contract, slot)][phase].append(row)

    witnesses: List[Dict[str, Any]] = []
    for (contract, slot), by_phase in grouped.items():
        outer_before = list(by_phase.get("outer_before_external", []) or [])
        inner = list(by_phase.get("reentry_inner", []) or [])
        outer_after = list(by_phase.get("outer_after_external", []) or [])
        if not inner or not (outer_before or outer_after):
            continue
        pre_reads = [row for row in outer_before if row.get("type") == "sload"]
        pre_writes = [row for row in outer_before if row.get("type") == "sstore"]
        inner_reads = [row for row in inner if row.get("type") == "sload"]
        inner_writes = [row for row in inner if row.get("type") == "sstore"]
        post_writes = [row for row in outer_after if row.get("type") == "sstore"]

        if pre_writes:
            order_pattern = "state_updated_before_external_edge"
            delayed_update_signal = False
        elif pre_reads and (inner_reads or inner_writes) and post_writes:
            order_pattern = (
                "outer_read_inner_access_outer_write_after_external"
            )
            delayed_update_signal = True
        elif (inner_reads or inner_writes) and post_writes:
            order_pattern = "inner_access_before_outer_update"
            delayed_update_signal = True
        elif inner_reads and pre_reads:
            order_pattern = "outer_and_inner_read_without_observed_outer_update"
            delayed_update_signal = False
        else:
            order_pattern = "overlap_without_phase_causality"
            delayed_update_signal = False

        ordered_rows = sorted(
            [*outer_before, *inner, *outer_after],
            key=lambda row: int(row.get("order_index", 10**9) or 10**9),
        )
        witnesses.append(omit_empty({
            "contract": contract,
            "slot_key_short": _short_value(slot, 18),
            "order_pattern": order_pattern,
            "potential_delayed_update_witness": delayed_update_signal,
            "outer_before_external": {
                "read_count": len(pre_reads),
                "write_count": len(pre_writes),
            },
            "reentry_inner": {
                "read_count": len(inner_reads),
                "write_count": len(inner_writes),
            },
            "outer_after_external": {
                "write_count": len(post_writes),
            },
            "evidence_ids": [
                row.get("evidence_id")
                for row in ordered_rows
                if row.get("evidence_id")
            ][:8],
        }))

    priority = {
        "outer_read_inner_access_outer_write_after_external": 0,
        "inner_access_before_outer_update": 1,
        "state_updated_before_external_edge": 2,
        "outer_and_inner_read_without_observed_outer_update": 3,
        "overlap_without_phase_causality": 4,
    }
    return sorted(
        witnesses,
        key=lambda item: (
            priority.get(str(item.get("order_pattern") or ""), 9),
            str(item.get("contract") or ""),
            str(item.get("slot_key_short") or ""),
        ),
    )[:12]


def classify_reentrancy_callback_kind(
    *,
    outer_node: Dict[str, Any],
    external_node: Dict[str, Any],
    reentry_node: Dict[str, Any],
) -> str:
    outer_function = get_function_name(outer_node).lower()
    external_function = get_function_name(external_node).lower()
    reentry_function = get_function_name(reentry_node).lower()
    combined = " ".join((outer_function, external_function, reentry_function))
    if get_node_type(reentry_node) == "staticcall":
        return "read_only_callback"
    if any(term in combined for term in ("flashloan", "flash_loan", "executeoperation")):
        return "flashloan_callback"
    if any(term in combined for term in ("swapcallback", "uniswap", "pancakecall")):
        return "dex_swap_callback"
    if any(
        term in combined
        for term in (
            "tokensreceived",
            "tokenfallback",
            "onerc",
            "ontransfer",
            "onflashloan",
        )
    ):
        return "token_hook"
    if reentry_function in {"fallback", "fallback()", "receive", "receive()"}:
        return "fallback_or_receive"
    if outer_function and reentry_function and outer_function == reentry_function:
        return "same_function"
    if outer_function and reentry_function:
        return "cross_function"
    return "unknown_callback"


def render_reentrancy_state_access_row(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    node_key: Any,
) -> Dict[str, Any]:
    node = trace_index.node_by_id.get(node_key, {})
    if not isinstance(node, dict) or not is_state_like(node):
        return {}
    addr_label, addr_label_sanitized = label_for_node(
        syn,
        node.get("address"),
        node.get("address_label"),
    )
    node_type = get_node_type(node)
    row = {
        "evidence_id": node_evidence_id(trace_index, node_key),
        "id": trace_index.display_id_by_id.get(node_key, node.get("id", node_key)),
        "type": node_type,
        "order_index": trace_index.node_index_by_id.get(node_key),
        **parent_context(trace_index, node_key),
        "contract": node.get("address"),
        "contract_label": addr_label,
        "contract_label_sanitized": addr_label_sanitized or None,
        "slot_key": node.get("slot_key"),
        "value_read": node.get("value") if node_type == "sload" else node.get("value_read"),
        "prev": node.get("prev"),
        "current": node.get("current"),
        "value": node.get("value"),
        "state_change": compact_value(node.get("state_change"), max_dict_items=8, max_list_len=6),
    }
    return omit_empty(row)


def summarize_state_slot_accesses(state_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_slot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        slot = str(row.get("slot_key") or "")
        if not slot:
            continue
        by_slot[slot].append(row)

    summaries = []
    for slot, rows in by_slot.items():
        accesses = [
            {
                "evidence_id": row.get("evidence_id"),
                "type": row.get("type"),
                "order_index": row.get("order_index"),
                "parent_function": row.get("parent_function"),
            }
            for row in rows
        ]
        summaries.append({
            "slot_key_short": _short_value(slot, 18),
            "access_count": len(rows),
            "read_count": sum(1 for row in rows if row.get("type") == "sload"),
            "write_count": sum(1 for row in rows if row.get("type") == "sstore"),
            "access_evidence_ids": _unique(row.get("evidence_id") for row in rows),
            "accesses": accesses[:6],
        })
    return summaries[:12]


def critical_call_reason(trace_index: TraceIndex, node_key: Any, node: Dict[str, Any]) -> str:
    node_type = get_node_type(node)
    call_type = str(node.get("call_type") or "").lower()
    fn = get_function_name(node).lower()
    compact_fn = re.sub(r"[^a-z0-9_]+", "", fn)
    for kw in CRITICAL_CALL_KEYWORDS:
        if kw in compact_fn:
            return f"matched keyword: {kw}"
    if call_type == "delegatecall" or node_type == "delegatecall":
        return "call_type: delegatecall"
    if node_type in {"create", "create2"} or call_type in {"create", "create2"}:
        return f"call_type: {node_type}"
    if is_unknown_selector_node(node):
        summary = children_summary(trace_index, node_key)
        total_children = sum(summary.values())
        if total_children >= 5:
            return "unknown selector with child/events/state context"
    return ""


def render_operation_summary_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    critical_call_view: List[Dict[str, Any]],
    transfer_event_view: List[Dict[str, Any]],
    external_fundflow_view: Dict[str, Any],
    amm_reserve_transition_view: Any,
    flash_or_atomic_capital_view: Any,
    value_release_view: Any = None,
    fundflow_obj: Any = None,
) -> Dict[str, Any]:
    """Neutral operation/signal summary derived from available packet evidence."""
    primary_operation = "unknown"
    entry_function = ""
    entry_selector = ""
    entry_decode_status = "unknown"
    for node_key, node in iter_trace_nodes(trace_index):
        if not is_call_like(node):
            continue
        candidate_function = get_function_name(node).strip()
        if candidate_function == "transaction_root":
            continue
        entry_selector = get_selector(node)
        if is_decoded_call(node):
            entry_function = candidate_function
            entry_decode_status = "decoded"
        else:
            entry_decode_status = "unknown"
        break

    if entry_function:
        fn_compact = re.sub(r"[^a-z0-9_]+", "", entry_function.lower())
        if any(k in fn_compact for k in ("swap", "exactinput", "exactoutput", "exchange")):
            primary_operation = "swap"
        elif any(k in fn_compact for k in ("transfer",)):
            primary_operation = "erc20_transfer"
        elif any(k in fn_compact for k in ("deposit",)):
            primary_operation = "deposit"
        elif any(k in fn_compact for k in ("withdraw",)):
            primary_operation = "withdraw"
        elif any(k in fn_compact for k in ("borrow",)):
            primary_operation = "borrow"
        elif any(k in fn_compact for k in ("mint",)):
            primary_operation = "mint"
        elif any(k in fn_compact for k in ("repay",)):
            primary_operation = "repay"
        elif any(k in fn_compact for k in ("redeem",)):
            primary_operation = "withdraw"
        elif any(k in fn_compact for k in ("liquidat",)):
            primary_operation = "liquidation"

    token_addresses = set()
    for row in transfer_event_view:
        if isinstance(row, dict):
            token = normalize_address(row.get("token"))
            if token:
                token_addresses.add(token)
    single_token_flow = len(token_addresses) <= 1

    has_swap = has_sync = has_mint_or_burn = False
    has_borrow_or_repay = has_liquidation = has_oracle_read = False

    critical_evidence_ids = {
        str(call.get("evidence_id") or "")
        for call in critical_call_view
        if isinstance(call, dict) and call.get("evidence_id")
    }
    transfer_event_total = 0
    for node_key, call in iter_trace_nodes(trace_index):
        node_type = get_node_type(call)
        if node_type == "event":
            event_name = get_function_name(call).strip().lower()
            event_signature = str(call.get("event_signature") or "").lower()
            if event_name == "transfer" or event_signature.startswith("transfer("):
                transfer_event_total += 1
            continue
        if not is_call_like(call):
            continue
        fn_compact = re.sub(
            r"[^a-z0-9_]+",
            "",
            get_function_name(call).lower(),
        )
        why = critical_call_reason(trace_index, node_key, call).lower()
        if why:
            critical_evidence_ids.add(node_evidence_id(trace_index, node_key))

        if not has_swap and any(k in fn_compact for k in ("swap", "exactinput", "exactoutput", "exchange")):
            has_swap = True
        if not has_sync and "sync" in fn_compact:
            has_sync = True
        if not has_mint_or_burn and any(k in fn_compact for k in ("mint", "burn")):
            has_mint_or_burn = True
        if not has_borrow_or_repay and any(k in fn_compact for k in ("borrow", "repay")):
            has_borrow_or_repay = True
        if not has_liquidation and "liquidat" in fn_compact:
            has_liquidation = True
        if not has_oracle_read and any(
            k in fn_compact
            for k in ("latestanswer", "latestrounddata", "consult", "peek", "get_virtual_price", "getreserves")
        ):
            has_oracle_read = True
        if not has_oracle_read and any(k in why for k in ("oracle", "latestanswer", "latestrounddata")):
            has_oracle_read = True

    for row in transfer_event_view:
        if not isinstance(row, dict):
            continue
        ev = str(row.get("event") or "").lower()
        if not has_sync and "sync" in ev:
            has_sync = True
        if not has_mint_or_burn and any(k in ev for k in ("mint", "burn")):
            has_mint_or_burn = True

    has_amm = _non_empty_view(amm_reserve_transition_view)
    has_flash = _non_empty_view(flash_or_atomic_capital_view)
    has_value_release = _non_empty_view(value_release_view)

    fundflow_count = 0
    if isinstance(external_fundflow_view, dict):
        recs = external_fundflow_view.get("records")
        if isinstance(recs, list):
            truncated = external_fundflow_view.get("truncated")
            fundflow_count = (
                int(truncated.get("total", len(recs)) or len(recs))
                if isinstance(truncated, dict)
                else len(recs)
            )
    if not fundflow_count and isinstance(fundflow_obj, dict):
        for key in (
            "transfers",
            "fund_flow",
            "fundFlows",
            "flows",
            "records",
            "items",
        ):
            records = fundflow_obj.get(key)
            if isinstance(records, list):
                fundflow_count = len(records)
                break
    elif not fundflow_count and isinstance(fundflow_obj, list):
        fundflow_count = len(fundflow_obj)

    has_price_dep = any([has_swap, has_oracle_read, has_borrow_or_repay, has_liquidation, has_amm])

    pkt_truncated = False
    meta = syn.get("metadata") if isinstance(syn.get("metadata"), dict) else {}
    total_nodes = meta.get("total_trace_nodes")
    if isinstance(total_nodes, (int, float)):
        pkt_truncated = int(total_nodes) > int(DEFAULT_PACKET_CONFIG.get("max_trace_nodes", 420))

    return omit_empty({
        "evidence_id": "operation_summary:0",
        "primary_operation": primary_operation,
        "entry_function": entry_function or None,
        "entry_selector": entry_selector or None,
        "entry_decode_status": entry_decode_status,
        "single_token_flow": single_token_flow,
        "transfer_event_count": transfer_event_total,
        "transfer_event_count_shown": len(transfer_event_view),
        "transfer_event_view_truncated": transfer_event_total > len(transfer_event_view),
        "fundflow_record_count": fundflow_count,
        "has_swap": has_swap,
        "has_sync": has_sync,
        "has_mint_or_burn": has_mint_or_burn,
        "has_borrow_or_repay": has_borrow_or_repay,
        "has_liquidation": has_liquidation,
        "has_oracle_read": has_oracle_read,
        "has_amm_reserve_transition": has_amm,
        "has_flash_or_atomic_capital": has_flash,
        "has_value_release": has_value_release,
        "has_price_dependent_operation": has_price_dep,
        "critical_call_count": len(critical_evidence_ids),
        "critical_call_count_shown": len(critical_call_view),
        "critical_call_view_truncated": len(critical_evidence_ids) > len(critical_call_view),
        "total_trace_nodes": int(total_nodes) if isinstance(total_nodes, (int, float)) else None,
        "packet_trace_truncated": pkt_truncated,
        "note": "Neutral operation/signal summary. Absence flags are derived from available packet evidence and are not attack judgments.",
    })


def _non_empty_view(view_data: Any) -> bool:
    if isinstance(view_data, list):
        return len(view_data) > 0
    if isinstance(view_data, dict):
        for key in ("records", "items", "rows"):
            val = view_data.get(key)
            if isinstance(val, list):
                return len(val) > 0
        return bool(view_data)
    return False


def render_unknown_selector_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_items: int = 80,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node_key, node in iter_trace_nodes(trace_index):
        if len(rows) >= max_items:
            break
        if not is_call_like(node) or not is_unknown_selector_node(node):
            continue
        summary = children_summary(trace_index, node_key)
        selector = get_selector(node) or "<missing_selector>"
        callee = normalize_address(node.get("address"))
        caller = normalize_address(node.get("caller"))
        callee_label, callee_label_sanitized = label_for_node(
            syn, callee, node.get("address_label")
        )
        caller_label, caller_label_sanitized = label_for_node(
            syn, caller, node.get("caller_label")
        )
        hint_parts = ["unknown selector"]
        if summary.get("event_nodes"):
            hint_parts.append("events")
        if summary.get("sstore_nodes"):
            hint_parts.append("state writes")
        if summary.get("call_nodes"):
            hint_parts.append("child calls")
        rows.append(omit_empty({
            "evidence_id": f"unknown_selector:{selector}@{node_evidence_id(trace_index, node_key)}",
            "call_id": trace_index.display_id_by_id.get(node_key, node.get("id", node_key)),
            "selector": selector,
            "callee": callee,
            "callee_label": callee_label,
            "callee_label_sanitized": callee_label_sanitized or None,
            "caller": caller,
            "caller_label": caller_label,
            "caller_label_sanitized": caller_label_sanitized or None,
            **parent_context(trace_index, node_key),
            "children_summary": {
                "call_nodes": summary.get("call_nodes", 0),
                "event_nodes": summary.get("event_nodes", 0),
                "sload_nodes": summary.get("sload_nodes", 0),
                "sstore_nodes": summary.get("sstore_nodes", 0),
            },
            "nearby_event_ids": collect_nearby_evidence_ids(trace_index, node_key, target="event"),
            "nearby_state_ids": collect_nearby_evidence_ids(trace_index, node_key, target="state"),
            "semantic_hint": ", ".join(hint_parts),
            "semantic_confidence": "low",
        }))
    return rows


ACCESS_CONTROL_SENSITIVE_KEYWORDS = [
    "owner",
    "admin",
    "role",
    "whitelist",
    "allowlist",
    "permission",
    "auth",
    "upgrade",
    "implementation",
    "proxy",
    "initialize",
    "set",
    "grant",
    "revoke",
    "mint",
    "burn",
    "withdraw",
    "claim",
    "redeem",
    "borrow",
    "sweep",
    "skim",
    "approve",
    "transfer",
]

AUTH_SLOT_KEYWORDS = [
    "owner",
    "admin",
    "role",
    "whitelist",
    "allowlist",
    "permission",
    "auth",
    "operator",
    "guardian",
    "implementation",
    "proxy",
]

EIP1967_SLOT_PREFIXES = {
    "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc": "eip1967_implementation_slot",
    "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103": "eip1967_admin_slot",
    "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50": "eip1967_beacon_slot",
}

SOURCE_AUTH_IRRELEVANT_FUNCTIONS = {
    "transaction_root",
    "balanceof",
    "totalsupply",
    "allowance",
    "decimals",
    "symbol",
    "name",
    "wad_exp",
    "newton_d",
    "get_y",
    "get_p",
    "get_dy",
    "balances",
    "price_oracle",
    "transfer",
    "transferfrom",
}

SOURCE_AUTH_PERMISSION_KEYWORDS = {
    "owner",
    "admin",
    "role",
    "whitelist",
    "allowlist",
    "permission",
    "authorized",
    "authorised",
    "operator",
    "guardian",
    "stakingcontract",
}


def render_source_unavailable_auth_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    *,
    critical_call_view: Optional[List[Dict[str, Any]]] = None,
    unknown_selector_view: Optional[List[Dict[str, Any]]] = None,
    semantic_state_delta_view: Optional[List[Dict[str, Any]]] = None,
    max_items: int = 80,
) -> Dict[str, Any]:
    """Build candidate-indexable auth context without source-code assumptions.

    This view is constructed before judging. It does not know the C1 candidate;
    downstream steps should filter it by C1 evidence_ids/path_ids. The view only
    reports observable trace/storage context and never proves that a require
    check is absent.
    """
    unknown_ids = {
        str(row.get("evidence_id") or "").split("@")[-1]
        for row in list(unknown_selector_view or [])
        if isinstance(row, dict)
    }
    semantic_rows = (
        semantic_state_delta_view
        if isinstance(semantic_state_delta_view, list)
        else []
    )
    known_addresses = _known_entity_addresses(syn, trace_index)
    ranked_rows: List[Tuple[int, int, Dict[str, Any]]] = []
    for sequence_index, (node_key, node) in enumerate(iter_trace_nodes(trace_index)):
        node_type = get_node_type(node)
        call_type = str(node.get("call_type") or node_type).lower()
        if call_type not in UNKNOWN_SELECTOR_CALL_TYPES:
            continue
        call_eid = node_evidence_id(trace_index, node_key)
        selector_unknown = is_unknown_selector_node(node) or call_eid in unknown_ids
        if _source_auth_call_is_irrelevant(node, selector_unknown=selector_unknown):
            continue
        critical_reason = critical_call_reason(trace_index, node_key, node)
        sensitive_tags = _source_unavailable_sensitive_tags(
            node,
            critical_reason=critical_reason,
            selector_unknown=selector_unknown,
        )
        state_context = _auth_state_context_for_call(
            syn,
            trace_index,
            node_key,
            max_items=8,
            known_addresses=known_addresses,
        )

        callee = normalize_address(node.get("address"))
        caller = normalize_address(node.get("caller"))
        callee_label, callee_label_sanitized = label_for_node(
            syn, callee, node.get("address_label")
        )
        caller_label, caller_label_sanitized = label_for_node(
            syn, caller, node.get("caller_label")
        )
        semantic_hints = _semantic_auth_hints_for_contract(
            semantic_rows,
            callee,
            max_items=4,
        )
        auth_hints = _auth_slot_hints_from_state_context(state_context)
        if not sensitive_tags and not selector_unknown and not semantic_hints and not auth_hints:
            continue
        child_summary = children_summary(trace_index, node_key)
        permission_observation = _source_auth_permission_observation(
            node,
            call_eid=call_eid,
        )
        if permission_observation:
            observable_status = "present"
            status_reason = "an external permission query returned observable data"
        else:
            observable_status, status_reason = _observable_auth_status(
                auth_hints,
                semantic_hints,
                child_summary,
            )
        path_ids = list(trace_index.call_path_ids_by_id.get(node_key, []) or [])[-6:]
        path_evidence_ids = [
            f"call:{pid}" for pid in path_ids if pid not in (None, "")
        ]
        nearby_event_ids = collect_nearby_evidence_ids(
            trace_index,
            node_key,
            target="event",
        )
        nearby_state_ids = collect_nearby_evidence_ids(
            trace_index,
            node_key,
            target="state",
        )
        transaction_state_anchors = _source_auth_transaction_state_anchors(
            state_context,
            actor=caller,
            target=callee,
        )
        anchor_ids = _unique([
            call_eid,
            *path_evidence_ids,
            *nearby_event_ids,
            *nearby_state_ids,
            *[
                str(row.get("evidence_id") or "")
                for row in transaction_state_anchors
                if row.get("evidence_id")
            ],
            *[
                str(hint.get("evidence_id") or "")
                for hint in auth_hints
                if hint.get("evidence_id")
            ],
            *[
                str(row.get("evidence_id") or "")
                for row in semantic_hints
                if row.get("evidence_id")
            ],
        ])
        observations = _source_auth_observations(
            auth_hints=auth_hints,
            semantic_hints=semantic_hints,
            permission_observation=permission_observation,
        )
        row = omit_empty({
            "evidence_id": f"source_unavailable_auth:{call_eid}",
            "call_evidence_id": call_eid,
            "anchor_evidence_ids": anchor_ids,
            "path_ids": path_ids,
            "path_evidence_ids": path_evidence_ids,
            "call_id": trace_index.display_id_by_id.get(
                node_key,
                node.get("id", node_key),
            ),
            "type": get_node_type(node),
            "call_type": node.get("call_type"),
            **parent_context(trace_index, node_key),
            "caller": caller,
            "caller_label": caller_label,
            "caller_label_sanitized": caller_label_sanitized or None,
            "callee": callee,
            "callee_label": callee_label,
            "callee_label_sanitized": callee_label_sanitized or None,
            "function": get_function_name(node),
            "selector": get_selector(node),
            "selector_unknown_or_weak": selector_unknown,
            "critical_call_reason": critical_reason,
            "sensitive_action_tags": sensitive_tags,
            "children_summary": {
                "call_nodes": child_summary.get("call_nodes", 0),
                "event_nodes": child_summary.get("event_nodes", 0),
                "sload_nodes": child_summary.get("sload_nodes", 0),
                "sstore_nodes": child_summary.get("sstore_nodes", 0),
            },
            "observable_authorization_evidence": observable_status,
            "observable_authorization_reason": status_reason,
            "observations": observations,
            "transaction_state_anchors": transaction_state_anchors,
            "nearby_event_ids": nearby_event_ids,
            "nearby_state_ids": nearby_state_ids,
        })
        rank_score = _source_auth_candidate_score(
            node,
            selector_unknown=selector_unknown,
            sensitive_tags=sensitive_tags,
            permission_observation=permission_observation,
            auth_hints=auth_hints,
            semantic_hints=semantic_hints,
        )
        ranked_rows.append((rank_score, sequence_index, row))
    ranked_rows.sort(key=lambda item: (-item[0], item[1]))
    rows = [row for _, _, row in ranked_rows[:max_items]]
    status_counts = Counter(
        str(row.get("observable_authorization_evidence") or "unknown")
        for row in rows
    )
    informative_count = sum(
        count
        for status, count in status_counts.items()
        if status != "unknown"
    )
    return {
        "summary": {
            "row_count": len(rows),
            "status_counts": dict(status_counts),
            "informative_count": informative_count,
            "all_rows_unknown": bool(rows) and informative_count == 0,
            "status": (
                "insufficient_trace_granularity"
                if rows and informative_count == 0
                else "observable_auth_context_available"
                if informative_count
                else "no_candidate_rows"
            ),
            "projection_policy": (
                "Filter by C1 candidate evidence_ids/path_ids/callee/selector "
                "before using rows for C2 authorization-gap judging."
            ),
            "status_values": ["present", "weak", "absent", "unknown"],
            "not_source_proof": True,
            "limitation": (
                "Trace/storage observations do not prove that a source-level "
                "require or branch check is absent."
            ),
            "unknown_row_policy": (
                "Unfiltered first-pass rendering shows only representative "
                "unknown rows. Candidate-filtered follow-up retains matching "
                "rows so call identity and evidence anchors remain available."
            ),
        },
        "proxy_context": _source_unavailable_proxy_context(
            trace_index,
            known_addresses=known_addresses,
        ),
        "auth_contexts": rows,
    }


def _source_auth_call_is_irrelevant(
    node: Dict[str, Any],
    *,
    selector_unknown: bool,
) -> bool:
    if selector_unknown:
        return False
    function = get_function_name(node).strip().lower()
    if function.startswith("precompile:"):
        return True
    base = function.split("(", 1)[0].strip()
    if base in SOURCE_AUTH_IRRELEVANT_FUNCTIONS:
        return True
    call_type = str(node.get("call_type") or get_node_type(node)).lower()
    if call_type == "staticcall" and not any(
        keyword in base for keyword in SOURCE_AUTH_PERMISSION_KEYWORDS
    ):
        return True
    return False


def _source_auth_candidate_score(
    node: Dict[str, Any],
    *,
    selector_unknown: bool,
    sensitive_tags: List[str],
    permission_observation: Optional[Dict[str, Any]],
    auth_hints: List[Dict[str, Any]],
    semantic_hints: List[Dict[str, Any]],
) -> int:
    function = get_function_name(node).lower().split("(", 1)[0]
    score = 12 * len(sensitive_tags)
    if permission_observation:
        score += 60
    if semantic_hints:
        score += 45
    if auth_hints:
        score += 30
    if str(node.get("call_type") or "").lower() == "delegatecall":
        score += 35
    if selector_unknown:
        score += 20
    if any(
        token in function
        for token in (
            "owner",
            "admin",
            "role",
            "upgrade",
            "initialize",
            "withdraw",
            "claim",
            "redeem",
            "mint",
            "burn",
        )
    ):
        score += 35
    if function == "approve":
        # Approval is authorization-relevant when tied to a protected chain,
        # but ranks below direct privilege/state/value operations by default.
        score += 8
    if parse_decimal(node.get("value")) not in (None, Decimal(0)):
        score += 20
    return score


def _source_unavailable_sensitive_tags(
    node: Dict[str, Any],
    *,
    critical_reason: str,
    selector_unknown: bool,
) -> List[str]:
    text = " ".join(
        str(part or "")
        for part in (
            get_function_name(node),
            get_selector(node),
            node.get("call_type"),
            node.get("value"),
            critical_reason,
        )
    ).lower()
    tags: List[str] = []
    function = get_function_name(node).lower().split("(", 1)[0]
    for keyword in ACCESS_CONTROL_SENSITIVE_KEYWORDS:
        matched = function.startswith("set") if keyword == "set" else keyword in text
        if matched:
            tags.append(keyword)
    if any(keyword in function for keyword in SOURCE_AUTH_PERMISSION_KEYWORDS):
        tags.append("permission_query")
    if selector_unknown:
        tags.append("unknown_selector")
    if str(node.get("call_type") or "").lower() == "delegatecall":
        tags.append(str(node.get("call_type")).lower())
    value = parse_decimal(node.get("value"))
    if value is not None and value > 0:
        tags.append("native_value")
    return _unique(tags)[:8]


def _auth_state_context_for_call(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    node_key: Any,
    *,
    max_items: int,
    known_addresses: set[str],
) -> List[Dict[str, Any]]:
    state_keys: List[Any] = []

    def collect_descendants(parent_key: Any) -> None:
        for child_key in trace_index.children_by_id.get(parent_key, []):
            if len(state_keys) >= max_items:
                return
            child = trace_index.node_by_id.get(child_key, {})
            if is_state_like(child):
                state_keys.append(child_key)
            collect_descendants(child_key)
            if len(state_keys) >= max_items:
                return

    collect_descendants(node_key)
    if len(state_keys) < max_items:
        pos = trace_index.node_index_by_id.get(node_key, 0)
        lo = max(0, pos - 8)
        hi = min(len(trace_index.sequence), pos + 12)
        for nearby_key in trace_index.sequence[lo:hi]:
            if nearby_key == node_key or nearby_key in state_keys:
                continue
            nearby = trace_index.node_by_id.get(nearby_key, {})
            if is_state_like(nearby):
                state_keys.append(nearby_key)
                if len(state_keys) >= max_items:
                    break
    return [
        _auth_state_context_row(
            syn,
            trace_index,
            state_key,
            known_addresses=known_addresses,
        )
        for state_key in state_keys[:max_items]
    ]


def _auth_state_context_row(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    state_key: Any,
    *,
    known_addresses: set[str],
) -> Dict[str, Any]:
    node = trace_index.node_by_id.get(state_key, {})
    address = normalize_address(node.get("address"))
    label, label_sanitized = label_for_node(
        syn,
        address,
        node.get("address_label"),
    )
    state_change = node.get("state_change") if isinstance(node.get("state_change"), dict) else {}
    prev = node.get("prev") if node.get("prev") not in (None, "") else state_change.get("prev")
    current = node.get("current") if node.get("current") not in (None, "") else state_change.get("current")
    value = node.get("value") if node.get("value") not in (None, "") else state_change.get("value")
    slot_key = str(node.get("slot_key") or state_change.get("key") or "").strip()
    return omit_empty({
        "evidence_id": node_evidence_id(trace_index, state_key),
        "id": trace_index.display_id_by_id.get(state_key, node.get("id", state_key)),
        "type": get_node_type(node),
        "address": address,
        "address_label": label,
        "address_label_sanitized": label_sanitized or None,
        **parent_context(trace_index, state_key),
        "slot_key": slot_key,
        "prev": compact_value(prev, max_str_len=120),
        "current": compact_value(current, max_str_len=120),
        "value": compact_value(value, max_str_len=120),
        "slot_classification": _classify_auth_slot(
            slot_key=slot_key,
            prev=prev,
            current=current,
            value=value,
            state_change=state_change,
            known_addresses=known_addresses,
        ),
        "state_change_summary": compact_value(
            state_change,
            max_str_len=140,
            max_list_len=4,
            max_dict_items=8,
        ),
    })


def _classify_auth_slot(
    *,
    slot_key: str,
    prev: Any,
    current: Any,
    value: Any,
    state_change: Dict[str, Any],
    known_addresses: set[str],
) -> List[str]:
    tags: List[str] = []
    slot = str(slot_key or "").lower()
    if slot in EIP1967_SLOT_PREFIXES:
        tags.append(EIP1967_SLOT_PREFIXES[slot])
    slot_number = parse_decimal(slot)
    if (
        slot.startswith("0x")
        and len(slot) == 66
        and slot not in EIP1967_SLOT_PREFIXES
        and slot_number is not None
        and slot_number >= Decimal(2) ** 128
    ):
        tags.append("high_entropy_mapping_slot")
    joined = " ".join(
        str(part or "").lower()
        for part in (
            slot_key,
            state_change.get("variable"),
            state_change.get("name"),
            state_change.get("label"),
            state_change.get("type"),
        )
    )
    if any(keyword in joined for keyword in AUTH_SLOT_KEYWORDS):
        tags.append("owner_role_proxy_slot_hint")
    values = [prev, current, value]
    if any(
        _looks_known_address_word(
            item,
            known_addresses=known_addresses,
            standard_slot=slot in EIP1967_SLOT_PREFIXES,
        )
        for item in values
    ):
        tags.append("address_like_value")
    parsed_values = [
        parsed for parsed in (parse_decimal(item) for item in values)
        if parsed is not None
    ]
    if parsed_values and set(parsed_values).issubset({Decimal(0), Decimal(1)}):
        tags.append("boolean_like_value")
    nonzero = [parsed for parsed in parsed_values if parsed not in (None, Decimal(0))]
    if nonzero and not any(tag.endswith("value") for tag in tags):
        tags.append("amount_or_counter_like_value")
    return _unique(tags)[:6]


def _looks_known_address_word(
    value: Any,
    *,
    known_addresses: set[str],
    standard_slot: bool,
) -> bool:
    text = str(value or "").lower()
    candidate = ""
    if ADDRESS_RE.match(text):
        candidate = text
    if text.startswith("0x") and len(text) == 66:
        candidate = "0x" + text[-40:]
    if not candidate or not ADDRESS_RE.match(candidate) or int(candidate, 16) == 0:
        return False
    return standard_slot or candidate in known_addresses


def _known_entity_addresses(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
) -> set[str]:
    known = {
        normalize_address(value)
        for value in (
            list(address_label_map(syn).keys())
            + [syn.get("sender"), syn.get("receiver")]
        )
        if is_address(value)
    }
    for _, node in iter_trace_nodes(trace_index):
        for key in ("address", "caller", "from", "to", "recipient", "token", "contract"):
            value = normalize_address(node.get(key))
            if is_address(value):
                known.add(value)
    return known


def _source_auth_observations(
    *,
    auth_hints: List[Dict[str, Any]],
    semantic_hints: List[Dict[str, Any]],
    permission_observation: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    if permission_observation:
        observations.append(permission_observation)
    for hint in semantic_hints[:4]:
        observations.append(omit_empty({
            "kind": "semantic_authorization_state",
            "variable": hint.get("variable"),
            "variable_type": hint.get("variable_type"),
            "key_address": hint.get("key_address"),
            "prev": hint.get("prev"),
            "current": hint.get("current"),
            "refs": [hint.get("evidence_id")] if hint.get("evidence_id") else [],
        }))
    for hint in auth_hints[:4]:
        observations.append(omit_empty({
            "kind": "storage_authorization_hint",
            "slot_key": hint.get("slot_key"),
            "hint_types": hint.get("hint_types"),
            "confidence": hint.get("confidence"),
            "refs": [hint.get("evidence_id")] if hint.get("evidence_id") else [],
        }))
    return observations


def _source_auth_transaction_state_anchors(
    rows: List[Dict[str, Any]],
    *,
    actor: str,
    target: str,
) -> List[Dict[str, Any]]:
    """Preserve candidate-local trace facts without inferring authorization."""
    anchors: List[Dict[str, Any]] = []
    actor_norm = normalize_address(actor)
    target_norm = normalize_address(target)
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        node_type = str(row.get("type") or "").strip().lower()
        prev = row.get("prev")
        current = row.get("current")
        classifications = list(row.get("slot_classification") or [])
        changed = (
            prev not in (None, "")
            and current not in (None, "")
            and str(prev) != str(current)
        )
        is_write = "sstore" in node_type or "write" in node_type or changed
        if not is_write and not classifications:
            continue
        current_address = _storage_word_address(current or row.get("value"))
        anchors.append(omit_empty({
            "evidence_id": row.get("evidence_id"),
            "operation": "state_write" if is_write else "state_observation",
            "contract": row.get("address"),
            "slot_key": row.get("slot_key"),
            "prev": prev,
            "current": current,
            "value": row.get("value"),
            "slot_classification": classifications,
            "current_value_matches_actor": bool(
                current_address and actor_norm and current_address == actor_norm
            ),
            "current_value_matches_target": bool(
                current_address and target_norm and current_address == target_norm
            ),
        }))
        if len(anchors) >= 6:
            break
    return anchors


def _source_auth_permission_observation(
    node: Dict[str, Any],
    *,
    call_eid: str,
) -> Optional[Dict[str, Any]]:
    function = get_function_name(node).lower().split("(", 1)[0]
    if not any(keyword in function for keyword in SOURCE_AUTH_PERMISSION_KEYWORDS):
        return None
    result = node.get("return_values") or node.get("output")
    if result in (None, "", [], {}):
        return None
    return omit_empty({
        "kind": "external_permission_query",
        "function": get_function_name(node),
        "args": compact_value(node.get("args_in") or node.get("params")),
        "result": compact_value(result),
        "refs": [call_eid],
    })


def _storage_word_address(value: Any) -> str:
    text = str(value or "").lower()
    if ADDRESS_RE.match(text):
        return text
    if text.startswith("0x") and len(text) == 66:
        candidate = "0x" + text[-40:]
        if ADDRESS_RE.match(candidate) and int(candidate, 16) != 0:
            return candidate
    return ""


def _source_unavailable_proxy_context(
    trace_index: TraceIndex,
    *,
    known_addresses: set[str],
) -> List[Dict[str, Any]]:
    by_proxy: Dict[str, Dict[str, Any]] = {}
    for node_key, node in iter_trace_nodes(trace_index):
        if not is_state_like(node):
            continue
        state_change = (
            node.get("state_change")
            if isinstance(node.get("state_change"), dict)
            else {}
        )
        slot = str(node.get("slot_key") or state_change.get("key") or "").lower()
        slot_kind = EIP1967_SLOT_PREFIXES.get(slot)
        if not slot_kind:
            continue
        proxy = normalize_address(node.get("address"))
        if not is_address(proxy):
            continue
        raw_value = (
            node.get("current")
            or state_change.get("current")
            or node.get("value")
            or state_change.get("value")
        )
        target = _storage_word_address(raw_value)
        if target and target not in known_addresses:
            known_addresses.add(target)
        row = by_proxy.setdefault("{}".format(proxy), {"proxy": proxy})
        field_name = {
            "eip1967_implementation_slot": "implementation",
            "eip1967_admin_slot": "admin",
            "eip1967_beacon_slot": "beacon",
        }[slot_kind]
        ref_key = f"{field_name}_ref"
        value_key = field_name
        row[value_key] = target or compact_value(raw_value, max_str_len=100)
        row[ref_key] = node_evidence_id(trace_index, node_key)
    return [omit_empty(row) for row in by_proxy.values()]


def _semantic_auth_hints_for_contract(
    semantic_rows: List[Dict[str, Any]],
    contract: str,
    *,
    max_items: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    contract_norm = normalize_address(contract)
    for row in list(semantic_rows or []):
        if len(out) >= max_items:
            break
        if not isinstance(row, dict):
            continue
        if contract_norm and normalize_address(row.get("contract")) != contract_norm:
            continue
        text = " ".join(
            str(row.get(key) or "").lower()
            for key in (
                "variable",
                "variable_type",
                "key_path",
                "contract_label",
                "key_address_label",
            )
        )
        if not any(keyword in text for keyword in AUTH_SLOT_KEYWORDS):
            continue
        out.append(omit_empty({
            "evidence_id": row.get("evidence_id"),
            "contract": row.get("contract"),
            "variable": row.get("variable"),
            "variable_type": row.get("variable_type"),
            "key_address": row.get("key_address"),
            "prev": compact_value(row.get("prev"), max_str_len=80),
            "current": compact_value(row.get("current"), max_str_len=80),
            "delta": row.get("delta"),
            "hint": "semantic_owner_role_proxy_like_state",
        }))
    return out


def _auth_slot_hints_from_state_context(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    for row in rows:
        tags = list(row.get("slot_classification") or [])
        auth_tags = [
            tag for tag in tags
            if tag == "owner_role_proxy_slot_hint"
        ]
        if not auth_tags:
            continue
        hints.append({
            "evidence_id": row.get("evidence_id"),
            "slot_key": row.get("slot_key"),
            "hint_types": auth_tags,
            "confidence": "medium" if "owner_role_proxy_slot_hint" in auth_tags else "low",
        })
    return hints[:6]


def _observable_auth_status(
    auth_hints: List[Dict[str, Any]],
    semantic_hints: List[Dict[str, Any]],
    child_summary: Dict[str, int],
) -> Tuple[str, str]:
    if semantic_hints:
        return "present", "semantic owner/role/proxy-like state evidence is present"
    if any(
        hint.get("confidence") == "medium"
        for hint in auth_hints
    ):
        return "weak", "raw slot evidence has owner/role/proxy-like hints"
    if auth_hints:
        return "weak", "raw slot evidence has low-confidence auth-like hints"
    if int(child_summary.get("sload_nodes", 0) or 0) > 0:
        return "unknown", "state was read but no auth semantics could be inferred"
    if int(child_summary.get("sstore_nodes", 0) or 0) > 0:
        return "unknown", "state was written but no guard-read semantics could be inferred"
    return "unknown", "no authorization semantics could be established from available trace rows"


def is_unknown_selector_node(node: Dict[str, Any]) -> bool:
    node_type = get_node_type(node)
    call_type = str(node.get("call_type") or "").lower()
    if node_type not in UNKNOWN_SELECTOR_CALL_TYPES and call_type not in UNKNOWN_SELECTOR_CALL_TYPES:
        return False
    selector_kind = str(node.get("selector_kind") or "").lower()
    if selector_kind in {"precompile_input_prefix", "precompile"}:
        return False
    fn = get_function_name(node).strip()
    if fn.lower().startswith("precompile:"):
        return False
    selector = get_selector(node)
    calldata = str(node.get("calldata") or "").strip()
    # In synthesized traces calldata stores arguments after the selector for
    # many providers, so an unknown call can legitimately have calldata=""
    # while raw_selector/function_selector still carries the 4-byte selector.
    if not selector:
        return False

    has_explicit_selector = any(
        SELECTOR_RE.match(str(node.get(key) or "").strip())
        for key in ("raw_selector", "function_selector", "selector")
    )
    has_calldata_selector = calldata.startswith("0x") and len(calldata) >= 10
    if not has_explicit_selector and not has_calldata_selector:
        return False

    if selector_kind == "unknown":
        return True
    if not fn:
        return True
    if SELECTOR_RE.match(fn):
        return True
    if fn.lower() in {"<unknown>", "unknown"}:
        return True
    return False


def render_semantic_state_delta_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    token_info: Any = None,
    max_items: int = 240,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    token_map = token_info_map(token_info if token_info is not None else syn.get("token_info"))
    state_changes = syn.get("state_changes") or []
    if _is_data_not_available(state_changes):
        return {"available": False, "reason": "state_changes reported as not available by data source"}

    if isinstance(state_changes, list):
        for group_index, group in enumerate(state_changes):
            if len(rows) >= max_items:
                break
            if not isinstance(group, dict):
                continue
            contract = normalize_address(group.get("address"))
            contract_label, contract_label_sanitized = label_for_address(syn, contract)
            token_meta = token_map.get(contract, {})
            variables = group.get("variables") or []

            if variables:
                for var in variables:
                    for var_row in _flatten_variable_rows(var):
                        if len(rows) >= max_items:
                            break
                        seq = len(rows)
                        prev = var_row.get("prev")
                        current = var_row.get("current")
                        delta = decimal_delta(prev, current)
                        key_address = _last_address(var_row.get("keys", []))
                        key_label, key_label_sanitized = label_for_address(syn, key_address)
                        rows.append(omit_empty({
                            "evidence_id": f"state_semantic:{contract}:{seq}",
                            "source_evidence_id": f"state_change:{contract}:{group_index}",
                            "contract": contract,
                            "contract_label": contract_label,
                            "contract_label_sanitized": contract_label_sanitized or None,
                            "variable": var_row.get("variable"),
                            "variable_type": var_row.get("variable_type"),
                            "key_address": key_address,
                            "key_address_label": key_label,
                            "key_address_label_sanitized": key_label_sanitized or None,
                            "key_path": var_row.get("keys"),
                            "prev": prev,
                            "current": current,
                            "delta": _signed_decimal(delta),
                            "normalized_delta": _signed_decimal(
                                parse_decimal(normalize_token_amount(delta, token_meta.get("decimals")))
                            )
                            if delta is not None and token_meta.get("decimals") is not None
                            else None,
                            "token_symbol": token_meta.get("symbol"),
                            "token_decimals": token_meta.get("decimals"),
                            "semantic_confidence": "high",
                        }))
            else:
                for slot_index, slot in enumerate(group.get("slots") or []):
                    if len(rows) >= max_items:
                        break
                    if not isinstance(slot, dict):
                        continue
                    prev = slot.get("prev")
                    current = slot.get("current")
                    delta = decimal_delta(prev, current)
                    rows.append(omit_empty({
                        "evidence_id": f"state_semantic:{contract}:raw_slot:{slot_index}",
                        "source_evidence_id": f"state_change:{contract}:{slot_index}",
                        "contract": contract,
                        "contract_label": contract_label,
                        "contract_label_sanitized": contract_label_sanitized or None,
                        "slot_key": slot.get("key") or slot.get("slot_key"),
                        "variable_hint": "raw_slot_only",
                        "prev": prev,
                        "current": current,
                        "delta": _signed_decimal(delta),
                        "semantic_confidence": "low",
                        "reason": "No decoded variable metadata was present for this slot.",
                    }))

    if not rows:
        for node_key, node in iter_trace_nodes(trace_index):
            if len(rows) >= max_items:
                break
            if get_node_type(node) != "sstore":
                continue
            contract = normalize_address(node.get("address"))
            contract_label, contract_label_sanitized = label_for_address(syn, contract)
            prev = node.get("prev") or (node.get("state_change") or {}).get("prev")
            current = node.get("current") or (node.get("state_change") or {}).get("current")
            delta = decimal_delta(prev, current)
            rows.append(omit_empty({
                "evidence_id": f"state_semantic:{node_evidence_id(trace_index, node_key)}",
                "source_evidence_id": node_evidence_id(trace_index, node_key),
                "contract": contract,
                "contract_label": contract_label or node.get("address_label"),
                "contract_label_sanitized": contract_label_sanitized or None,
                "slot_key": node.get("slot_key") or (node.get("state_change") or {}).get("key"),
                "variable_hint": "raw_slot_only",
                "prev": prev,
                "current": current,
                "delta": _signed_decimal(delta),
                "semantic_confidence": "low",
                "reason": "Only raw trace storage slot metadata was available.",
                **parent_context(trace_index, node_key),
            }))
    return rows


def _flatten_variable_rows(var: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(var, dict):
        return []
    variable = var.get("name") or var.get("variable") or var.get("variable_name")
    variable_type = var.get("type") or var.get("variable_type")
    value = var.get("value")
    rows: List[Dict[str, Any]] = []

    def walk(value_obj: Any, keys: List[str]) -> None:
        if isinstance(value_obj, dict):
            if "prev" in value_obj or "current" in value_obj:
                rows.append({
                    "variable": variable,
                    "variable_type": variable_type,
                    "keys": list(keys),
                    "prev": value_obj.get("prev"),
                    "current": value_obj.get("current"),
                })
                return
            fields = value_obj.get("fields")
            if isinstance(fields, list):
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    field_name = str(field.get("name") or "")
                    field_type = field.get("type") or variable_type
                    before = len(rows)
                    walk(field.get("value"), keys + ([field_name] if field_name else []))
                    for row in rows[before:]:
                        if field_name and row.get("variable"):
                            row["variable"] = f"{row['variable']}.{field_name}"
                        row["variable_type"] = field_type
                return
            kvs = value_obj.get("kvs")
            if isinstance(kvs, list):
                for kv in kvs:
                    if not isinstance(kv, dict):
                        continue
                    key = str(kv.get("key") or kv.get("name") or "")
                    walk(kv.get("value"), keys + ([key] if key else []))
                return
        if value_obj not in (None, "", [], {}):
            rows.append({
                "variable": variable,
                "variable_type": variable_type,
                "keys": list(keys),
                "current": value_obj,
            })

    walk(value, [])
    return rows


def _last_address(values: List[Any]) -> Optional[str]:
    for value in reversed(values):
        if is_address(value):
            return normalize_address(value)
    return None


def _signed_decimal(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    formatted = format_decimal(abs(value))
    if value > 0:
        return f"+{formatted}"
    if value < 0:
        return f"-{formatted}"
    return "0"


_TOKEN_SEMANTIC_VARIABLE_KEYWORDS = (
    "balance",
    "balances",
    "gon",
    "gons",
    "rOwned",
    "tOwned",
    "reflection",
    "rebase",
    "supply",
    "totalSupply",
    "fee",
    "tax",
    "burn",
    "lock",
    "locked",
    "unlock",
    "hook",
)


def render_token_semantic_delta_summary_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    semantic_state_delta_view: Optional[List[Dict[str, Any]]] = None,
    transfer_event_view: Optional[List[Dict[str, Any]]] = None,
    token_info: Any = None,
    max_items: int = 80,
) -> List[Dict[str, Any]]:
    semantic_rows = semantic_state_delta_view or render_semantic_state_delta_view(
        syn,
        trace_index,
        token_info=token_info if token_info is not None else syn.get("token_info"),
    )
    if isinstance(semantic_rows, dict) and semantic_rows.get("available") is False:
        return [semantic_rows]
    transfer_rows = transfer_event_view or render_transfer_event_view(syn, trace_index)
    tokens = token_info_map(token_info if token_info is not None else syn.get("token_info"))
    transfer_by_token: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in list(transfer_rows or []):
        token = normalize_address(row.get("token"))
        if token:
            transfer_by_token[token].append(row)

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in list(semantic_rows or []):
        if not isinstance(row, dict):
            continue
        contract = normalize_address(row.get("contract"))
        if not contract:
            continue
        variable = str(row.get("variable") or row.get("variable_hint") or "").strip()
        haystack = " ".join([
            variable,
            str(row.get("variable_type") or ""),
            str(row.get("contract_label") or ""),
            str(row.get("slot_key") or ""),
        ]).lower()
        token_contract = contract in tokens or contract in transfer_by_token
        semantic_hint = _token_semantic_mechanism_hint(haystack)
        if not token_contract and semantic_hint == "unknown":
            continue

        if contract not in grouped:
            meta = tokens.get(contract, {})
            grouped[contract] = {
                "evidence_id": f"token_semantic_delta:{contract}",
                "token": contract,
                "token_symbol": meta.get("symbol"),
                "token_decimals": meta.get("decimals"),
                "token_label": row.get("contract_label"),
                "token_label_sanitized": row.get("contract_label_sanitized"),
                "token_contract_origin_supported": bool(token_contract),
                "mechanism_hints": [],
                "state_evidence_ids": [],
                "transfer_event_evidence_ids": [],
                "affected_addresses": [],
                "sample_state_deltas": [],
                "origin_boundary_note": (
                    "Candidate is anchored to the token contract's own state/events; "
                    "downstream reliance/outcome checks must preserve this token."
                    if token_contract
                    else "State row is not clearly on a known token contract; treat as boundary/uncertain."
                ),
            }
        item = grouped[contract]
        if semantic_hint != "unknown":
            item["mechanism_hints"].append(semantic_hint)
        if row.get("evidence_id"):
            item["state_evidence_ids"].append(row.get("evidence_id"))
        key_address = normalize_address(row.get("key_address"))
        if key_address:
            item["affected_addresses"].append(key_address)
        if len(item["sample_state_deltas"]) < 8:
            item["sample_state_deltas"].append(omit_empty({
                "evidence_id": row.get("evidence_id"),
                "variable": variable or row.get("variable_hint"),
                "key_address": key_address,
                "delta": row.get("normalized_delta") or row.get("delta"),
                "semantic_confidence": row.get("semantic_confidence"),
            }))

    for token, rows_for_token in transfer_by_token.items():
        if token not in grouped:
            meta = tokens.get(token, {})
            label, label_sanitized = label_for_address(syn, token)
            grouped[token] = {
                "evidence_id": f"token_semantic_delta:{token}",
                "token": token,
                "token_symbol": meta.get("symbol"),
                "token_decimals": meta.get("decimals"),
                "token_label": label,
                "token_label_sanitized": label_sanitized or None,
                "token_contract_origin_supported": bool(token in tokens),
                "mechanism_hints": [],
                "state_evidence_ids": [],
                "transfer_event_evidence_ids": [],
                "affected_addresses": [],
                "sample_state_deltas": [],
                "origin_boundary_note": (
                    "Transfer-event-only candidate; needs state/balance/supply "
                    "evidence before treating token semantics as satisfied."
                ),
            }
        item = grouped[token]
        item["transfer_event_evidence_ids"].extend(
            row.get("evidence_id") for row in rows_for_token[:12] if row.get("evidence_id")
        )
        if len(rows_for_token) >= 2:
            item["mechanism_hints"].append("multi_transfer_pattern")
        zero_amount = any(str(row.get("amount") or "").strip() in {"0", "0.0"} for row in rows_for_token)
        if zero_amount:
            item["mechanism_hints"].append("zero_amount_transfer_event")

    rows: List[Dict[str, Any]] = []
    for item in grouped.values():
        state_ids = _unique(item.get("state_evidence_ids") or [])
        event_ids = _unique(item.get("transfer_event_evidence_ids") or [])
        hints = _unique(item.get("mechanism_hints") or [])
        if not state_ids and not event_ids:
            continue
        confidence = "high" if item.get("token_contract_origin_supported") and state_ids else "medium"
        if not hints:
            confidence = "low"
        rows.append(omit_empty({
            **item,
            "mechanism_hints": hints,
            "state_evidence_ids": state_ids[:16],
            "transfer_event_evidence_ids": event_ids[:16],
            "affected_addresses": _unique(item.get("affected_addresses") or [])[:12],
            "sample_state_deltas": item.get("sample_state_deltas")[:8],
            "candidate_confidence": confidence,
        }))
        if len(rows) >= max_items:
            break
    return rows


def render_token_accounting_origin_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    token_semantic_delta_summary_view: Optional[List[Dict[str, Any]]] = None,
    semantic_state_delta_view: Optional[List[Dict[str, Any]]] = None,
    transfer_event_view: Optional[List[Dict[str, Any]]] = None,
    critical_call_view: Optional[List[Dict[str, Any]]] = None,
    max_items: int = 80,
) -> List[Dict[str, Any]]:
    token_rows = token_semantic_delta_summary_view or render_token_semantic_delta_summary_view(
        syn,
        trace_index,
        semantic_state_delta_view=semantic_state_delta_view,
        transfer_event_view=transfer_event_view,
        token_info=syn.get("token_info"),
        max_items=max_items,
    )
    if isinstance(token_rows, dict) and token_rows.get("available") is False:
        return [token_rows]
    semantic_rows = semantic_state_delta_view or []
    if isinstance(semantic_rows, dict):
        semantic_rows = []
    token_contracts = {
        normalize_address(row.get("token"))
        for row in list(token_rows or [])
        if isinstance(row, dict) and normalize_address(row.get("token"))
    }
    rows: List[Dict[str, Any]] = []
    for row in list(token_rows or []):
        if not isinstance(row, dict):
            continue
        token = normalize_address(row.get("token"))
        if not token:
            continue
        origin = "token_contract_semantics" if row.get("token_contract_origin_supported") else "uncertain_token_origin"
        rows.append(omit_empty({
            "evidence_id": f"token_accounting_origin:{token}",
            "token": token,
            "token_symbol": row.get("token_symbol"),
            "origin_classification": origin,
            "origin_confidence": row.get("candidate_confidence") or "medium",
            "mechanism_hints": row.get("mechanism_hints"),
            "token_semantic_candidate_evidence_id": row.get("evidence_id"),
            "state_evidence_ids": row.get("state_evidence_ids"),
            "transfer_event_evidence_ids": row.get("transfer_event_evidence_ids"),
            "boundary_note": (
                "This row supports token-semantic origin only for the listed token. "
                "Protocol-internal ledger/reentrancy/accounting effects on other "
                "contracts must not be substituted for this token candidate."
            ),
        }))
        if len(rows) >= max_items:
            return rows

    protocol_rows: Dict[str, Dict[str, Any]] = {}
    for row in list(semantic_rows or []):
        if not isinstance(row, dict):
            continue
        contract = normalize_address(row.get("contract"))
        if not contract or contract in token_contracts:
            continue
        variable = str(row.get("variable") or row.get("variable_hint") or "").lower()
        if not any(keyword in variable for keyword in ("balance", "ledger", "account", "share", "reserve", "debt", "collateral")):
            continue
        if contract not in protocol_rows:
            protocol_rows[contract] = {
                "evidence_id": f"token_accounting_origin:protocol_internal:{contract}",
                "contract": contract,
                "contract_label": row.get("contract_label"),
                "origin_classification": "protocol_internal_accounting",
                "origin_confidence": "medium",
                "state_evidence_ids": [],
                "boundary_note": (
                    "Accounting/state divergence appears on a non-token/protocol "
                    "contract. It may support another attack family, but does not "
                    "by itself satisfy token semantic origin."
                ),
            }
        protocol_rows[contract]["state_evidence_ids"].append(row.get("evidence_id"))
    for item in protocol_rows.values():
        item["state_evidence_ids"] = _unique(item.get("state_evidence_ids") or [])[:12]
        rows.append(omit_empty(item))
        if len(rows) >= max_items:
            break
    return rows


def _token_semantic_mechanism_hint(text: str) -> str:
    lowered = str(text or "").lower()
    if "gon" in lowered or "rowned" in lowered or "towned" in lowered or "reflection" in lowered:
        return "reflection_or_gon_balance"
    if "rebase" in lowered:
        return "rebase_or_elastic_supply"
    if "lock" in lowered or "vesting" in lowered:
        return "locked_or_vesting_balance"
    if "fee" in lowered or "tax" in lowered:
        return "fee_or_tax_parameter"
    if "burn" in lowered:
        return "deflationary_burn"
    if "totalsupply" in lowered or "supply" in lowered:
        return "supply_mutation"
    if "hook" in lowered:
        return "hook_driven_transfer"
    if "balance" in lowered:
        return "balance_accounting_delta"
    if "raw_slot" in lowered:
        return "raw_slot_token_state"
    return "unknown"


def _short_value(value: Any, max_chars: int = 24) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _is_protocol_contract_label(label: str) -> bool:
    lowered = str(label or "").lower()
    return any(kw in lowered for kw in _PROTOCOL_CONTRACT_LABEL_KEYWORDS)


def _is_protocol_address(
    syn: Dict[str, Any],
    address: Optional[str],
) -> bool:
    if not address:
        return False
    label_map = address_label_map(syn)
    label = label_map.get(normalize_address(address), "")
    return _is_protocol_contract_label(label)


def _classify_price_state_role(
    row: Dict[str, Any],
    syn: Dict[str, Any],
) -> Tuple[str, str, str]:
    """Return (state_role, price_relevance, included_reason) for a semantic state row."""
    variable = str(row.get("variable") or row.get("variable_hint") or "").lower()
    contract = normalize_address(row.get("contract"))
    contract_label = str(row.get("contract_label") or "").lower()
    key_address = normalize_address(row.get("key_address"))
    key_label = str(row.get("key_address_label") or "").lower()

    is_balance = bool(_BALANCE_VARIABLE_PATTERNS.search(variable))

    contract_is_protocol = _is_protocol_contract_label(contract_label) or _is_protocol_address(syn, contract)
    key_is_protocol = _is_protocol_contract_label(key_label) or _is_protocol_address(syn, key_address) if key_address else False

    # --- Classify state_role ---
    if any(kw in variable for kw in ("reserve0", "reserve1", "reserves", "getreserves")):
        state_role = "reserve"
    elif any(kw in variable for kw in ("oracle", "price", "consult", "peek")):
        state_role = "oracle"
    elif any(kw in variable for kw in ("collateral",)):
        state_role = "collateral"
    elif any(kw in variable for kw in ("debt", "borrowindex")):
        state_role = "debt"
    elif any(kw in variable for kw in ("exchangerate", "virtualprice", "share", "totalsupply")):
        state_role = "share_price_input"
    elif any(kw in variable for kw in ("tick", "sqrtprice", "feegrowth")):
        state_role = "reserve"
    elif is_balance:
        if key_is_protocol or contract_is_protocol:
            if "pair" in key_label or "pool" in key_label or "amm" in key_label:
                state_role = "pair_balance"
            elif "vault" in key_label:
                state_role = "vault_balance"
            elif contract_is_protocol and not key_is_protocol:
                state_role = "user_balance"
            else:
                state_role = "pair_balance" if key_is_protocol else "user_balance"
        else:
            state_role = "user_balance"
    elif "liquidity" in variable:
        state_role = "reserve"
    elif "klast" in variable:
        state_role = "reserve"
    elif "oracle" in contract_label:
        state_role = "oracle"
    elif contract_is_protocol and is_balance:
        state_role = "user_balance"
    else:
        state_role = "unknown"

    # --- Classify price_relevance ---
    if state_role in ("reserve", "oracle"):
        price_relevance = "high"
    elif state_role in ("pair_balance", "vault_balance", "share_price_input", "collateral", "debt"):
        price_relevance = "medium"
    elif state_role == "user_balance" and (key_is_protocol or contract_is_protocol):
        price_relevance = "medium"
    elif state_role == "user_balance":
        price_relevance = "low"
    else:
        price_relevance = "low"

    # --- included_reason ---
    if price_relevance == "high":
        included_reason = f"state_role={state_role}: directly price-determining state"
    elif price_relevance == "medium":
        included_reason = f"state_role={state_role}: price-influencing via protocol contract"
    else:
        included_reason = f"state_role={state_role}: ordinary user balance, low price relevance"

    return state_role, price_relevance, included_reason


def render_price_relevant_state_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    semantic_state_delta_view: Optional[List[Dict[str, Any]]] = None,
    max_contracts: int = 80,
) -> List[Dict[str, Any]]:
    semantic_rows = semantic_state_delta_view or render_semantic_state_delta_view(syn, trace_index)
    if isinstance(semantic_rows, dict) and semantic_rows.get("available") is False:
        return semantic_rows
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in semantic_rows:
        haystack = " ".join(
            str(row.get(k, ""))
            for k in (
                "variable",
                "variable_hint",
                "variable_type",
                "contract_label",
                "slot_key",
            )
        ).lower()
        if not any(keyword.lower() in haystack for keyword in PRICE_STATE_KEYWORDS):
            continue
        contract = normalize_address(row.get("contract"))
        if not contract:
            continue

        state_role, price_relevance, included_reason = _classify_price_state_role(row, syn)

        if price_relevance == "low" and state_role == "user_balance":
            continue

        if contract not in grouped:
            grouped[contract] = {
                "evidence_id": f"price_state_contract:{contract}",
                "contract": contract,
                "contract_label": row.get("contract_label"),
                "rows": [],
            }
        grouped[contract]["rows"].append(omit_empty({
            "evidence_id": f"price_state:{row.get('evidence_id')}",
            "source_evidence_id": row.get("source_evidence_id") or row.get("evidence_id"),
            "variable_hint": row.get("variable") or row.get("variable_hint") or "price_related_state",
            "prev": row.get("prev"),
            "current": row.get("current"),
            "delta": row.get("delta"),
            "depth": row.get("depth"),
            "parent_function": row.get("parent_function"),
            "semantic_confidence": row.get("semantic_confidence"),
            "state_role": state_role,
            "price_relevance": price_relevance,
            "included_reason": included_reason,
        }))
    _attach_price_state_write_order_context(grouped, trace_index)
    return list(grouped.values())[:max_contracts]


def _attach_price_state_write_order_context(
    grouped: Dict[str, Dict[str, Any]],
    trace_index: TraceIndex,
    *,
    max_writes_per_contract: int = 18,
) -> None:
    if not grouped:
        return
    writes_by_contract: Dict[str, List[Dict[str, Any]]] = {
        contract: [] for contract in grouped
    }
    for sequence_index, (node_key, node) in enumerate(iter_trace_nodes(trace_index)):
        if get_node_type(node) != "sstore":
            continue
        contract = normalize_address(node.get("address"))
        if contract not in writes_by_contract:
            continue
        writes = writes_by_contract[contract]
        if len(writes) >= max_writes_per_contract:
            continue
        state_change = (
            node.get("state_change")
            if isinstance(node.get("state_change"), dict)
            else {}
        )
        writes.append(omit_empty({
            "sequence_index": sequence_index,
            "evidence_id": node_evidence_id(trace_index, node_key),
            "slot_key": node.get("slot_key") or state_change.get("key"),
            "prev": node.get("prev")
            if node.get("prev") not in (None, "")
            else state_change.get("prev"),
            "current": node.get("current")
            if node.get("current") not in (None, "")
            else state_change.get("current"),
            **parent_context(trace_index, node_key),
        }))

    for contract, writes in writes_by_contract.items():
        if not writes:
            continue
        group = grouped[contract]
        group["ordered_write_context"] = writes
        group["ordered_write_context_scope"] = (
            "capped same-contract SSTORE sequence; rows are ordering context "
            "and are not all proven to be price-determining slots"
        )
        restorations: List[Dict[str, Any]] = []
        by_slot: Dict[str, List[Dict[str, Any]]] = {}
        for write in writes:
            slot = str(write.get("slot_key") or "")
            if slot:
                by_slot.setdefault(slot, []).append(write)
        for slot, slot_writes in by_slot.items():
            if len(slot_writes) < 2:
                continue
            first_prev = str(slot_writes[0].get("prev") or "").lower()
            last_current = str(slot_writes[-1].get("current") or "").lower()
            changed_between = any(
                str(item.get("current") or "").lower() != first_prev
                for item in slot_writes[:-1]
            )
            if first_prev and first_prev == last_current and changed_between:
                restorations.append({
                    "slot_key": slot,
                    "first_write_evidence_id": slot_writes[0].get("evidence_id"),
                    "last_write_evidence_id": slot_writes[-1].get("evidence_id"),
                    "write_count": len(slot_writes),
                    "pattern": "transient_change_restored_within_transaction",
                })
        if restorations:
            group["transient_restoration_candidates"] = restorations[:8]


def render_amm_reserve_transition_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    max_pairs: int = 80,
) -> List[Dict[str, Any]]:
    events = render_event_view(syn, trace_index, max_events=800)
    grouped: Dict[str, Dict[str, Any]] = {}
    for event in events:
        name = str(event.get("event") or "")
        sig = str(event.get("event_signature") or "")
        lowered = (name or sig).lower()
        if not any(x in lowered for x in ("sync", "swap", "mint", "burn")):
            continue
        pair = normalize_address(event.get("address"))
        if not pair:
            continue
        if pair not in grouped:
            label, _ = label_for_address(syn, pair)
            grouped[pair] = {
                "evidence_id": f"amm:{pair}",
                "pair": pair,
                "pair_label": label or event.get("address_label"),
                "events": [],
            }
        params = event.get("args") or event.get("params") or {}
        event_row = omit_empty({
            "evidence_id": event.get("evidence_id"),
            "event": name or sig,
            "reserve0": get_param_value(params, ["reserve0", "_reserve0"]),
            "reserve1": get_param_value(params, ["reserve1", "_reserve1"]),
            "amount0In": get_param_value(params, ["amount0In", "amount0_in"]),
            "amount1In": get_param_value(params, ["amount1In", "amount1_in"]),
            "amount0Out": get_param_value(params, ["amount0Out", "amount0_out"]),
            "amount1Out": get_param_value(params, ["amount1Out", "amount1_out"]),
            "sender": get_param_value(params, ["sender", "from"]),
            "to": get_param_value(params, ["to", "recipient"]),
        })
        grouped[pair]["events"].append(event_row)

    out: List[Dict[str, Any]] = []
    for group in list(grouped.values())[:max_pairs]:
        sync_events = [
            e for e in group["events"]
            if str(e.get("event", "")).lower().startswith("sync")
            and e.get("reserve0") is not None
            and e.get("reserve1") is not None
        ]
        if sync_events:
            first = sync_events[0]
            last = sync_events[-1]
            group["first_reserve"] = {
                "reserve0": first.get("reserve0"),
                "reserve1": first.get("reserve1"),
            }
            group["last_reserve"] = {
                "reserve0": last.get("reserve0"),
                "reserve1": last.get("reserve1"),
            }
            group["reserve_change_pct"] = omit_empty({
                "reserve0": safe_pct_change(first.get("reserve0"), last.get("reserve0")),
                "reserve1": safe_pct_change(first.get("reserve1"), last.get("reserve1")),
            })
            ratio_first = _ratio(first.get("reserve1"), first.get("reserve0"))
            ratio_last = _ratio(last.get("reserve1"), last.get("reserve0"))
            group["price_ratio_change_pct"] = safe_pct_change(ratio_first, ratio_last)
            group["transition_summary"] = "computed from Sync events only; no attack judgment"
        else:
            group["transition_summary"] = "AMM events were listed, but no safe Sync reserve transition could be computed."
        group["events"] = group["events"][:80]
        out.append(omit_empty(group))
    return out


def _ratio(numerator: Any, denominator: Any) -> Optional[Decimal]:
    n = parse_decimal(numerator)
    d = parse_decimal(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


def render_participant_net_delta_view(
    syn: Dict[str, Any],
    fundflow_obj: Any,
    transfer_event_view: Optional[List[Dict[str, Any]]] = None,
    profit_loss_obj: Any = None,
    address_labels: Optional[List[Dict[str, Any]]] = None,
    max_items: int = 120,
) -> List[Dict[str, Any]]:
    fundflow_view = render_external_fundflow_view(fundflow_obj, max_items=1000)
    records = fundflow_view.get("records", []) if isinstance(fundflow_view, dict) else []
    source = "external_fundflow_view"
    if not records and transfer_event_view:
        records = _records_from_transfer_events(transfer_event_view)
        source = "transfer_event_view"

    deltas: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tokens": defaultdict(Decimal), "eids": defaultdict(list)})
    token_labels: Dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        amount = parse_decimal(record.get("amount") or record.get("value"))
        if amount is None:
            continue
        token = normalize_address(record.get("token") or record.get("asset") or record.get("tokenAddress") or "native")
        token_label, _ = label_for_address(syn, token)
        token_labels[token] = token_label or str(record.get("token_symbol") or record.get("symbol") or token)
        from_addr = normalize_address(record.get("from") or record.get("src") or record.get("sender"))
        to_addr = normalize_address(record.get("to") or record.get("dst") or record.get("recipient"))
        eid = record.get("evidence_id")
        if from_addr:
            deltas[from_addr]["tokens"][token] -= amount
            if eid:
                deltas[from_addr]["eids"][token].append(eid)
        if to_addr:
            deltas[to_addr]["tokens"][token] += amount
            if eid:
                deltas[to_addr]["eids"][token].append(eid)

    sender = normalize_address(syn.get("sender"))
    rows: List[Dict[str, Any]] = []
    for address, info in sorted(
        deltas.items(),
        key=lambda item: (0 if item[0] == sender else 1, item[0]),
    ):
        if len(rows) >= max_items:
            break
        label, label_sanitized = label_for_address(syn, address)
        token_rows = []
        related: List[str] = []
        for token, delta in info["tokens"].items():
            if delta == 0:
                continue
            related.extend(info["eids"].get(token, []))
            token_rows.append({
                "token": token,
                "token_label": token_labels.get(token),
                "delta": _signed_decimal(delta),
                "direction": "in" if delta > 0 else "out",
            })
        if not token_rows:
            continue
        rows.append(omit_empty({
            "evidence_id": f"participant_delta:{address}",
            "address": address,
            "label": label,
            "label_sanitized": label_sanitized or None,
            "is_tx_sender": address == sender,
            "deltas": token_rows[:40],
            "related_evidence_ids": _unique(related)[:80],
            "source": source,
            "notes": "neutral net delta summary; not an attack judgment",
        }))
    return rows


def _records_from_transfer_events(transfer_event_view: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for row in transfer_event_view:
        if not isinstance(row, dict):
            continue
        records.append({
            "evidence_id": row.get("evidence_id"),
            "from": row.get("from"),
            "to": row.get("to"),
            "token": row.get("token"),
            "amount": row.get("amount"),
        })
    return records


def render_contribution_vs_payout_view(
    syn: Dict[str, Any],
    participant_net_delta_view: List[Dict[str, Any]],
    semantic_state_delta_view: List[Dict[str, Any]],
    critical_call_view: List[Dict[str, Any]],
    max_items: int = 40,
) -> List[Dict[str, Any]]:
    pattern = infer_contribution_pattern(critical_call_view)
    rows: List[Dict[str, Any]] = []
    for participant in participant_net_delta_view:
        if len(rows) >= max_items:
            break
        deltas = participant.get("deltas") or []
        inputs = []
        outputs = []
        for delta in deltas:
            entry = {
                "token": delta.get("token"),
                "token_label": delta.get("token_label"),
                "amount": str(delta.get("delta", "")).lstrip("+-"),
                "evidence_ids": participant.get("related_evidence_ids", []),
            }
            if delta.get("direction") == "out":
                inputs.append(entry)
            elif delta.get("direction") == "in":
                outputs.append(entry)
        if not inputs and not outputs:
            continue
        protocol_rows = []
        for state in semantic_state_delta_view[:80]:
            variable_text = str(state.get("variable") or state.get("variable_hint") or "").lower()
            if not any(k in variable_text for k in ("balance", "reserve", "supply", "share")):
                continue
            protocol_rows.append(omit_empty({
                "token": state.get("contract"),
                "token_label": state.get("contract_label"),
                "delta": state.get("delta"),
                "evidence_ids": [state.get("evidence_id")],
            }))
            if len(protocol_rows) >= 10:
                break
        rows.append(omit_empty({
            "evidence_id": f"contribution_payout:{participant.get('address')}",
            "pattern": pattern,
            "participant": participant.get("address"),
            "participant_label": participant.get("label"),
            "input_assets": inputs[:20],
            "output_assets": outputs[:20],
            "protocol_balance_delta": protocol_rows,
            "interpretation_note": (
                "neutral accounting summary; judge must decide whether payout is "
                "disproportionate from the selected evidence"
            ),
        }))
    return rows


_PROTOCOL_ACCOUNTING_KEYWORDS = (
    "reward",
    "rewards",
    "rewarddebt",
    "accrued",
    "accumulated",
    "accreward",
    "accper",
    "pending",
    "claim",
    "claimed",
    "share",
    "shares",
    "vault",
    "strategy",
    "asset",
    "assets",
    "debt",
    "borrow",
    "borrowindex",
    "collateral",
    "liability",
    "liabilities",
    "stake",
    "staked",
    "deposit",
    "withdraw",
    "redeem",
    "supply",
    "totalsupply",
    "balance",
    "exchange",
    "exchangerate",
    "supplyindex",
    "rewardindex",
    "principal",
    "entitlement",
    "allowance",
)

_PROTOCOL_ACCOUNTING_ACTION_KEYWORDS = (
    "claim",
    "harvest",
    "reward",
    "withdraw",
    "redeem",
    "borrow",
    "mint",
    "deposit",
    "stake",
    "unstake",
    "liquidate",
    "repay",
    "earn",
    "settle",
    "strategy",
)


def render_protocol_accounting_outcome_view(
    *,
    syn: Dict[str, Any],
    semantic_state_delta_view: Any,
    critical_call_view: Any,
    value_release_view: Any,
    contribution_vs_payout_view: Any,
    participant_net_delta_view: Any,
    max_items: int = 48,
) -> Dict[str, Any]:
    """Pre-judge protocol-accounting outcome candidates.

    This view is deliberately candidate-indexable rather than verdict-like.
    Judge state binding must decide whether a row belongs to the same C1/C2
    bookkeeping candidate and whether the outcome is exploitative.
    """
    semantic_rows = _view_rows(semantic_state_delta_view)
    critical_rows = _view_rows(critical_call_view)
    value_rows = _view_rows(value_release_view)
    contribution_rows = _view_rows(contribution_vs_payout_view)
    participant_rows = _view_rows(participant_net_delta_view)

    accounting_rows = [
        row for row in semantic_rows
        if _protocol_accounting_state_row(row)
    ][: max_items * 3]
    action_rows = [
        row for row in critical_rows
        if _protocol_accounting_action_row(row)
    ][:32]

    action_ids = _extract_ids_from_rows(action_rows)[:20]
    value_ids = _extract_ids_from_rows(value_rows)[:20]
    contribution_ids = _extract_ids_from_rows(contribution_rows)[:20]
    participant_ids = _extract_ids_from_rows(participant_rows)[:20]
    rows: List[Dict[str, Any]] = []

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for state in accounting_rows:
        contract = normalize_address(state.get("contract"))
        domain = _protocol_accounting_domain_hint(state)
        key = (contract or "unknown", domain)
        group = grouped.setdefault(key, {
            "contract": contract,
            "contract_label": state.get("contract_label"),
            "domain": domain,
            "state_rows": [],
            "state_evidence_ids": [],
            "delta_directions": Counter(),
            "semantic_confidence_counts": Counter(),
        })
        if len(group["state_rows"]) < 8:
            group["state_rows"].append(omit_empty({
                "evidence_id": state.get("evidence_id"),
                "source_evidence_id": state.get("source_evidence_id"),
                "variable": state.get("variable") or state.get("variable_hint"),
                "key_address": state.get("key_address"),
                "prev": state.get("prev"),
                "current": state.get("current"),
                "delta": state.get("delta"),
                "normalized_delta": state.get("normalized_delta"),
                "semantic_confidence": state.get("semantic_confidence"),
            }))
        if state.get("evidence_id"):
            group["state_evidence_ids"].append(state.get("evidence_id"))
        delta_direction = _protocol_delta_direction(state.get("delta"))
        group["delta_directions"][delta_direction] += 1
        group["semantic_confidence_counts"][
            str(state.get("semantic_confidence") or "unknown")
        ] += 1

    for index, ((contract, domain), group) in enumerate(grouped.items(), start=1):
        if len(rows) >= max_items:
            break
        evidence_ids = _unique(
            list(group.get("state_evidence_ids") or [])
            + action_ids
            + value_ids
            + contribution_ids
            + participant_ids
        )[:32]
        if not evidence_ids:
            continue
        rows.append(omit_empty({
            "evidence_id": f"protocol_accounting_outcome:{domain}:{index}",
            "candidate_outcome_id": f"pa_outcome_{index}",
            "domain": domain,
            "protocol_component": contract,
            "protocol_component_label": group.get("contract_label"),
            "accounting_state_evidence_ids": _unique(
                group.get("state_evidence_ids") or []
            )[:16],
            "state_delta_direction_counts": dict(group.get("delta_directions") or {}),
            "semantic_confidence_counts": dict(
                group.get("semantic_confidence_counts") or {}
            ),
            "sample_state_deltas": group.get("state_rows", [])[:8],
            "related_action_evidence_ids": action_ids[:12],
            "related_value_release_evidence_ids": value_ids[:12],
            "related_contribution_payout_evidence_ids": contribution_ids[:12],
            "related_participant_delta_evidence_ids": participant_ids[:12],
            "all_outcome_evidence_ids": evidence_ids,
            "outcome_signal": _protocol_accounting_outcome_signal(
                domain=domain,
                state_delta_counts=group.get("delta_directions") or Counter(),
                has_value=bool(value_ids),
                has_contribution=bool(contribution_ids),
                has_participant=bool(participant_ids),
            ),
            "limitation": (
                "Neutral accounting-outcome candidate. It can show state-level "
                "share/reward/debt/vault/liability impact even when direct "
                "token value_release is absent, but Judge must bind it to the "
                "same C1/C2 bookkeeping candidate before using it."
            ),
        }))

    if not rows and (value_rows or contribution_rows or participant_rows):
        rows.append({
            "evidence_id": "protocol_accounting_outcome:fallback:0",
            "candidate_outcome_id": "pa_outcome_fallback_0",
            "domain": "unknown",
            "related_value_release_evidence_ids": value_ids[:12],
            "related_contribution_payout_evidence_ids": contribution_ids[:12],
            "related_participant_delta_evidence_ids": participant_ids[:12],
            "all_outcome_evidence_ids": _unique(
                value_ids + contribution_ids + participant_ids
            )[:32],
            "outcome_signal": "external_or_participant_outcome_without_decoded_accounting_state",
            "limitation": (
                "Fallback row: direct accounting state was not decoded. This "
                "does not prove protocol-accounting exploitation by itself."
            ),
        })

    return {
        "evidence_id": "protocol_accounting_outcome:summary",
        "available": True,
        "outcome_candidate_count": len(rows),
        "rows": rows[:max_items],
        "notes": [
            "Use this view for PAE outcome conditions before treating empty value_release_view as no outcome.",
            "Rows are neutral candidates and must be tied to the selected bookkeeping candidate.",
            "Generic profit, swaps, or unrelated value movement remain insufficient.",
        ],
    }


def _protocol_accounting_state_row(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    haystack = " ".join(
        str(row.get(key) or "")
        for key in (
            "variable",
            "variable_hint",
            "variable_type",
            "contract_label",
            "slot_key",
        )
    ).lower().replace("_", "")
    return any(
        keyword.replace("_", "") in haystack
        for keyword in _PROTOCOL_ACCOUNTING_KEYWORDS
    )


def _protocol_accounting_action_row(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    text = " ".join(
        str(row.get(key) or "")
        for key in ("function", "function_name", "decoded_function", "why_included")
    ).lower()
    return any(keyword in text for keyword in _PROTOCOL_ACCOUNTING_ACTION_KEYWORDS)


def _protocol_accounting_domain_hint(row: Dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("variable", "variable_hint", "variable_type", "contract_label")
    ).lower().replace("_", "")
    if any(
        k in text
        for k in (
            "reward",
            "pending",
            "claim",
            "harvest",
            "accrued",
            "accumulated",
            "accreward",
            "accper",
        )
    ):
        return "reward"
    if any(k in text for k in ("share", "vault", "exchangerate", "strategy", "asset")):
        return "share_or_vault"
    if any(k in text for k in ("debt", "borrow", "collateral", "liability")):
        return "debt_or_collateral"
    if any(k in text for k in ("stake", "staked", "deposit", "withdraw", "principal")):
        return "staking_or_entitlement"
    if any(k in text for k in ("supply", "balance")):
        return "balance_or_supply"
    return "accounting_state"


def _protocol_delta_direction(value: Any) -> str:
    delta = parse_decimal(str(value or ""))
    if delta is None:
        return "unknown"
    if delta > 0:
        return "increase"
    if delta < 0:
        return "decrease"
    return "zero"


def _protocol_accounting_outcome_signal(
    *,
    domain: str,
    state_delta_counts: Counter,
    has_value: bool,
    has_contribution: bool,
    has_participant: bool,
) -> str:
    parts = [f"{domain}_state_delta"]
    if state_delta_counts.get("increase") and state_delta_counts.get("decrease"):
        parts.append("mixed_state_movement")
    elif state_delta_counts.get("increase"):
        parts.append("state_increase")
    elif state_delta_counts.get("decrease"):
        parts.append("state_decrease")
    if has_value:
        parts.append("value_release_present")
    if has_contribution:
        parts.append("contribution_payout_context")
    if has_participant:
        parts.append("participant_delta_context")
    return "|".join(parts)


def render_value_release_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    critical_call_view: List[Dict[str, Any]],
    transfer_event_view: List[Dict[str, Any]],
    participant_net_delta_view: List[Dict[str, Any]],
    semantic_state_delta_view: List[Dict[str, Any]],
    max_items: int = 120,
    max_release_records_per_row: int = 24,
    max_state_rows_per_row: int = 20,
) -> List[Dict[str, Any]]:
    """Neutral summary of value leaving sensitive calls or being minted/paid out."""
    semantic_rows = semantic_state_delta_view if isinstance(semantic_state_delta_view, list) else []
    participant_rows = (
        participant_net_delta_view if isinstance(participant_net_delta_view, list) else []
    )
    participant_by_address = {
        normalize_address(row.get("address")): row
        for row in participant_rows
        if isinstance(row, dict) and row.get("address")
    }

    rows: List[Dict[str, Any]] = []
    for call in critical_call_view or []:
        if len(rows) >= max_items:
            break
        if not isinstance(call, dict) or not _is_value_release_call(call):
            continue

        call_id = call.get("id")
        call_key = find_node_key_by_display_id(trace_index, call_id)
        if call_key is None:
            continue
        descendant_keys = collect_descendant_keys(trace_index, call_key, limit=3000)
        descendant_display_ids = {
            trace_index.display_id_by_id.get(key, key) for key in descendant_keys
        }

        target = normalize_address(call.get("callee") or call.get("address"))
        target_label, target_label_sanitized = label_for_address(syn, target)

        transfer_records = _value_release_records_from_transfers(
            syn=syn,
            target=target,
            transfer_event_view=transfer_event_view,
            trace_index=trace_index,
            descendant_keys=descendant_keys,
            descendant_display_ids=descendant_display_ids,
            max_items=max_release_records_per_row,
        )
        native_records = _native_value_release_records(
            syn=syn,
            trace_index=trace_index,
            descendant_keys=descendant_keys,
            target=target,
            max_items=max_release_records_per_row,
        )
        release_records = (transfer_records + native_records)[:max_release_records_per_row]
        if not release_records:
            continue

        recipients = _value_release_recipient_summary(syn, release_records)
        recipient_addresses = [row.get("recipient") for row in recipients if row.get("recipient")]
        target_balance_changes = _value_release_balance_changes(
            syn=syn,
            semantic_rows=semantic_rows,
            addresses=[target],
            max_items=max_state_rows_per_row,
        )
        recipient_balance_changes = _value_release_balance_changes(
            syn=syn,
            semantic_rows=semantic_rows,
            addresses=recipient_addresses,
            max_items=8,
        )

        related_participant_deltas = []
        for address in _unique([target, *recipient_addresses]):
            row = participant_by_address.get(normalize_address(address))
            if row:
                related_participant_deltas.append({
                    "address": row.get("address"),
                    "label": row.get("label"),
                    "deltas": row.get("deltas"),
                    "evidence_id": row.get("evidence_id"),
                })

        evidence_ids = _unique(
            [call.get("evidence_id")]
            + [record.get("evidence_id") for record in release_records]
            + [eid for record in release_records for eid in record.get("related_evidence_ids", []) or []]
            + [row.get("evidence_id") for row in target_balance_changes]
            + [row.get("evidence_id") for row in related_participant_deltas]
        )

        rows.append(omit_empty({
            "evidence_id": f"value_release:call:{call_id}",
            "sensitive_call_evidence_id": call.get("evidence_id"),
            "call_id": call_id,
            "function": call.get("function"),
            "target_contract": target,
            "target_contract_label": target_label or call.get("address_label"),
            "target_contract_label_sanitized": target_label_sanitized or None,
            "parent_id": call.get("parent_id"),
            "parent_function": call.get("parent_function"),
            "depth": call.get("depth"),
            "path_ids": call.get("path_ids"),
            "call_path_tail": call.get("call_path_tail") if call.get("call_path_tail") else None,
            "release_record_count": len(release_records),
            "repeated_release_count": sum(
                group.get("count", 0)
                for group in _repeated_release_groups(release_records)
            ),
            "repeated_release_groups": _repeated_release_groups(release_records),
            "value_out_evidence_ids": _unique(record.get("evidence_id") for record in release_records),
            "value_out_summary": _value_release_records_summary(release_records),
            "recipients": recipients,
            "target_balance_change_evidence_ids": _unique(row.get("evidence_id") for row in target_balance_changes),
            "target_balance_change_summary": _balance_change_summary(target_balance_changes),
            "recipient_balance_change_evidence_ids": _unique(row.get("evidence_id") for row in recipient_balance_changes),
            "recipient_balance_change_summary": _balance_change_summary(recipient_balance_changes),
            "related_participant_delta_evidence_ids": _unique(row.get("evidence_id") for row in related_participant_deltas),
            "related_evidence_ids": evidence_ids,
            "notes": (
                "Neutral value-release summary. It reports transfers/native value "
                "and balance-delta evidence under sensitive calls; it does not judge "
                "whether the release is authorized, repeated exploit payout, or normal."
            ),
        }))
    return rows


def _value_release_records_summary(release_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_asset: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    recipients = set()
    for record in release_records:
        by_asset[str(record.get("asset_type") or "unknown")] += 1
        by_kind[str(record.get("release_kind") or "unknown")] += 1
        recipient = normalize_address(record.get("recipient"))
        if recipient:
            recipients.add(recipient)
    return {
        "record_count": len(release_records),
        "asset_type_counts": dict(by_asset),
        "release_kind_counts": dict(by_kind),
        "recipient_count": len(recipients),
    }


def _balance_change_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_variable: Counter[str] = Counter()
    contracts = set()
    for row in rows or []:
        by_variable[str(row.get("variable") or row.get("variable_hint") or "unknown")] += 1
        contract = normalize_address(row.get("contract"))
        if contract:
            contracts.add(contract)
    return {
        "row_count": len(rows or []),
        "variable_counts": dict(by_variable),
        "contract_count": len(contracts),
    }


def _is_value_release_call(call: Dict[str, Any]) -> bool:
    fn_compact = re.sub(r"[^a-z0-9_]+", "", str(call.get("function") or "").lower())
    why = str(call.get("why_included") or "").lower()
    return any(keyword in fn_compact for keyword in VALUE_RELEASE_CALL_KEYWORDS) or (
        "reentrant_call" in why and any(k in fn_compact for k in ("fallback", "receive"))
    )


def collect_descendant_keys(
    trace_index: TraceIndex,
    parent_key: Any,
    *,
    limit: int,
) -> List[Any]:
    out: List[Any] = []

    def visit(key: Any) -> None:
        if len(out) >= limit:
            return
        out.append(key)
        for child_key in trace_index.children_by_id.get(key, []):
            visit(child_key)
            if len(out) >= limit:
                return

    if parent_key is not None:
        visit(parent_key)
    return out


def _value_release_records_from_transfers(
    *,
    syn: Dict[str, Any],
    target: str,
    transfer_event_view: List[Dict[str, Any]],
    trace_index: TraceIndex,
    descendant_keys: List[Any],
    descendant_display_ids: set,
    max_items: int,
) -> List[Dict[str, Any]]:
    descendant_key_set = set(descendant_keys)
    rows: List[Dict[str, Any]] = []
    for transfer in transfer_event_view or []:
        if len(rows) >= max_items:
            break
        if not isinstance(transfer, dict):
            continue
        event_key = find_node_key_by_display_id(trace_index, transfer.get("event_id"))
        parent_id = transfer.get("parent_id")
        if event_key not in descendant_key_set and parent_id not in descendant_display_ids:
            continue

        from_addr = normalize_address(transfer.get("from"))
        to_addr = normalize_address(transfer.get("to"))
        token = normalize_address(transfer.get("token"))
        token_label, token_label_sanitized = label_for_address(syn, token)
        recipient_label, recipient_label_sanitized = label_for_address(syn, to_addr)

        release_kind = "descendant_transfer"
        if _is_zero_address(from_addr):
            release_kind = "mint_to_recipient"
        elif target and from_addr == target:
            release_kind = "target_contract_out"

        rows.append(omit_empty({
            "evidence_id": transfer.get("evidence_id"),
            "release_kind": release_kind,
            "asset_type": "token",
            "token": token,
            "token_label": token_label or transfer.get("token_label"),
            "token_label_sanitized": token_label_sanitized or None,
            "from": from_addr,
            "to": to_addr,
            "recipient": to_addr,
            "recipient_label": recipient_label,
            "recipient_label_sanitized": recipient_label_sanitized or None,
            "amount": transfer.get("amount"),
            "amount_numeric": format_decimal(parse_decimal(transfer.get("amount"))),
            "parent_id": transfer.get("parent_id"),
            "parent_function": transfer.get("parent_function"),
            "related_evidence_ids": [transfer.get("raw_event_evidence_id")],
        }))
    return rows


def _native_value_release_records(
    *,
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    descendant_keys: List[Any],
    target: str,
    max_items: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in descendant_keys:
        if len(rows) >= max_items:
            break
        node = trace_index.node_by_id.get(key, {})
        if not isinstance(node, dict) or not is_call_like(node):
            continue
        value = parse_decimal(node.get("value"))
        if value is None or value <= 0:
            continue
        caller = normalize_address(node.get("caller"))
        recipient = normalize_address(node.get("address"))
        recipient_label, recipient_label_sanitized = label_for_node(
            syn, recipient, node.get("address_label")
        )
        release_kind = "native_value_call"
        if target and caller == target:
            release_kind = "target_contract_native_out"
        rows.append(omit_empty({
            "evidence_id": node_evidence_id(trace_index, key),
            "release_kind": release_kind,
            "asset_type": "native",
            "token": "native",
            "from": caller,
            "to": recipient,
            "recipient": recipient,
            "recipient_label": recipient_label,
            "recipient_label_sanitized": recipient_label_sanitized or None,
            "amount": format_decimal(value),
            "call_id": trace_index.display_id_by_id.get(key, node.get("id", key)),
            "function": get_function_name(node),
            **parent_context(trace_index, key),
        }))
    return rows


def _value_release_recipient_summary(
    syn: Dict[str, Any],
    release_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in release_records:
        recipient = normalize_address(record.get("recipient"))
        token = normalize_address(record.get("token") or "native")
        if not recipient:
            continue
        key = (recipient, token)
        item = grouped.setdefault(key, {
            "recipient": recipient,
            "token": token,
            "token_label": record.get("token_label"),
            "count": 0,
            "amount_sum": Decimal(0),
            "evidence_ids": [],
        })
        item["count"] += 1
        amount = parse_decimal(record.get("amount"))
        if amount is not None:
            item["amount_sum"] += amount
        if record.get("evidence_id"):
            item["evidence_ids"].append(record.get("evidence_id"))

    rows = []
    for item in grouped.values():
        label, label_sanitized = label_for_address(syn, item["recipient"])
        rows.append(omit_empty({
            "recipient": item["recipient"],
            "recipient_label": label,
            "recipient_label_sanitized": label_sanitized or None,
            "token": item["token"],
            "token_label": item.get("token_label"),
            "release_count": item["count"],
            "amount_sum": format_decimal(item["amount_sum"]),
            "evidence_ids": _unique(item["evidence_ids"]),
        }))
    return sorted(rows, key=lambda row: (-int(row.get("release_count", 0)), row.get("recipient", "")))[:20]


def _repeated_release_groups(release_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in release_records:
        recipient = normalize_address(record.get("recipient"))
        token = normalize_address(record.get("token") or "native")
        if not recipient:
            continue
        key = (recipient, token)
        item = counter.setdefault(key, {
            "recipient": recipient,
            "token": token,
            "count": 0,
            "evidence_ids": [],
        })
        item["count"] += 1
        if record.get("evidence_id"):
            item["evidence_ids"].append(record.get("evidence_id"))
    return [
        {**item, "evidence_ids": _unique(item["evidence_ids"])}
        for item in counter.values()
        if item["count"] > 1
    ][:20]


def _value_release_balance_changes(
    *,
    syn: Dict[str, Any],
    semantic_rows: List[Dict[str, Any]],
    addresses: List[Any],
    max_items: int,
) -> List[Dict[str, Any]]:
    address_set = {normalize_address(address) for address in addresses if normalize_address(address)}
    if not address_set:
        return []
    rows: List[Dict[str, Any]] = []
    for state in semantic_rows:
        if len(rows) >= max_items:
            break
        if not isinstance(state, dict):
            continue
        variable_text = str(
            state.get("variable")
            or state.get("variable_hint")
            or state.get("type")
            or ""
        ).lower()
        if not any(k in variable_text for k in ("balance", "reserve", "supply", "share", "debt", "collateral")):
            continue
        key_address = normalize_address(state.get("key_address") or state.get("owner") or state.get("account"))
        contract = normalize_address(state.get("contract"))
        key_path = [normalize_address(item) for item in state.get("key_path", []) or []]
        if (
            key_address not in address_set
            and contract not in address_set
            and not any(item in address_set for item in key_path)
        ):
            continue
        contract_label, contract_label_sanitized = label_for_address(syn, contract)
        rows.append(omit_empty({
            "evidence_id": state.get("evidence_id"),
            "source_evidence_id": state.get("source_evidence_id"),
            "contract": contract,
            "contract_label": contract_label or state.get("contract_label"),
            "contract_label_sanitized": contract_label_sanitized or None,
            "variable": state.get("variable") or state.get("variable_hint"),
            "key_address": key_address,
            "prev": state.get("prev"),
            "current": state.get("current"),
            "delta": state.get("delta"),
            "normalized_delta": state.get("normalized_delta"),
            "semantic_confidence": state.get("semantic_confidence"),
        }))
    return rows


def _is_zero_address(address: Any) -> bool:
    text = normalize_address(address)
    return text in {
        "0x0000000000000000000000000000000000000000",
        "0x0",
        "",
    }


def infer_contribution_pattern(critical_call_view: List[Dict[str, Any]]) -> str:
    text = " ".join(str(row.get("function", "")) for row in critical_call_view).lower()
    if any(k in text for k in ("deposit", "mint")):
        return "deposit_and_mint"
    if any(k in text for k in ("redeem", "withdraw")):
        return "redeem_and_withdraw"
    if "borrow" in text:
        return "borrow"
    if "repay" in text:
        return "repay"
    if any(k in text for k in ("claim", "reward", "harvest")):
        return "claim_reward"
    return "unknown"


def render_flash_or_atomic_capital_view(
    syn: Dict[str, Any],
    trace_index: TraceIndex,
    fundflow_obj: Any,
    critical_call_view: Optional[List[Dict[str, Any]]] = None,
    max_items: int = 80,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    critical_rows = critical_call_view if critical_call_view is not None else render_critical_call_view(syn, trace_index)
    if isinstance(critical_rows, dict):
        critical_rows = critical_rows.get("rows", [])

    for call in critical_rows:
        if len(rows) >= max_items:
            break
        fn = str(call.get("function") or "").lower()
        if "flash" not in fn and "uniswapv2call" not in fn and "pancakecall" not in fn:
            continue
        params = call.get("params") or {}
        provider = normalize_address(call.get("callee"))
        borrower = normalize_address(
            get_param_value(params, ["recipient", "receiver", "borrower", "to"])
            or call.get("caller")
        )
        token = normalize_address(get_param_value(params, ["token", "asset"]))
        amount = first_nonzero_param(
            params,
            ["amount", "amount0", "amount1", "assets", "value"],
        )
        provider_label, _ = label_for_address(syn, provider)
        rows.append(omit_empty({
            "evidence_id": f"atomic_capital:{len(rows) + 1}",
            "capital_type": "flash_loan",
            "provider": provider,
            "provider_label": provider_label or call.get("address_label"),
            "borrower": borrower,
            "token": token,
            "borrow_amount": amount,
            "borrow_evidence_ids": [call.get("evidence_id")],
            "repay_evidence_ids": [],
            "confidence": "high",
            "notes": "derived from flash-related call/function evidence; no attack judgment",
        }))

    for candidate in infer_borrow_repay_pairs(syn, fundflow_obj):
        if len(rows) >= max_items:
            break
        rows.append({
            "evidence_id": f"atomic_capital:{len(rows) + 1}",
            **candidate,
        })
    return rows


def infer_borrow_repay_pairs(syn: Dict[str, Any], fundflow_obj: Any) -> List[Dict[str, Any]]:
    view = render_external_fundflow_view(fundflow_obj, max_items=1000)
    records = view.get("records", []) if isinstance(view, dict) else []
    pairs: List[Dict[str, Any]] = []
    for i, first in enumerate(records):
        if not isinstance(first, dict):
            continue
        src = normalize_address(first.get("from"))
        dst = normalize_address(first.get("to"))
        token = normalize_address(first.get("token"))
        amount = parse_decimal(first.get("amount"))
        if not src or not dst or amount is None or amount <= 0:
            continue
        for second in records[i + 1:]:
            if not isinstance(second, dict):
                continue
            if normalize_address(second.get("from")) != dst:
                continue
            if normalize_address(second.get("to")) != src:
                continue
            if normalize_address(second.get("token")) != token:
                continue
            repay = parse_decimal(second.get("amount"))
            if repay is None or repay <= 0:
                continue
            if repay < amount * Decimal("0.90"):
                continue
            provider_label, _ = label_for_address(syn, src)
            pairs.append(omit_empty({
                "capital_type": "single_tx_borrow_repay",
                "provider": src,
                "provider_label": provider_label,
                "borrower": dst,
                "token": token,
                "borrow_amount": format_decimal(amount),
                "repay_amount": format_decimal(repay),
                "borrow_evidence_ids": [first.get("evidence_id")],
                "repay_evidence_ids": [second.get("evidence_id")],
                "fee": format_decimal(repay - amount) if repay >= amount else None,
                "confidence": "medium",
                "notes": "neutral same-transaction opposite-direction fund flow pair",
            }))
            break
        if len(pairs) >= 40:
            break
    return pairs


def first_nonzero_param(params: Any, names: List[str]) -> Any:
    fallback = None
    for name in names:
        value = get_param_value(params, [name])
        if value in (None, ""):
            continue
        if fallback is None:
            fallback = value
        numeric = parse_decimal(value)
        if numeric is None or numeric != 0:
            return value
    return fallback


def render_beneficiary_controller_view(
    syn: Dict[str, Any],
    tx_card: Dict[str, Any],
    address_labels: List[Dict[str, Any]],
    trace_view: List[Dict[str, Any]],
    critical_call_view: List[Dict[str, Any]],
    participant_net_delta_view: List[Dict[str, Any]],
    contribution_vs_payout_view: List[Dict[str, Any]],
    external_fundflow_view: Any,
    profit_loss_view: Any,
    transfer_event_view: List[Dict[str, Any]],
    max_items: int = 20,
) -> Dict[str, Any]:
    """Neutral same-transaction beneficiary/controller relationship summary."""
    label_map = _labels_from_address_view(address_labels)
    tx_sender = normalize_address(tx_card.get("sender") or syn.get("sender"))
    entry_contract = normalize_address(tx_card.get("receiver") or syn.get("receiver"))
    if not entry_contract:
        entry_contract = _first_call_address(trace_view)

    direct_callees = _direct_callees(tx_sender, entry_contract, trace_view, critical_call_view)
    positive = _top_delta_addresses(participant_net_delta_view, direction="in", limit=max_items)
    negative = _top_delta_addresses(participant_net_delta_view, direction="out", limit=max_items)
    _add_profit_loss_hints(positive, negative, profit_loss_view, label_map)

    paths = []
    path_evidence = []
    first_call_eid = _first_call_evidence_id(trace_view)
    if first_call_eid:
        path_evidence.append(first_call_eid)
    for row in positive[:max_items]:
        recipient = normalize_address(row.get("address"))
        evidence_ids = _unique(path_evidence + list(row.get("related_evidence_ids", []) or []))
        if recipient and tx_sender and recipient == tx_sender:
            relation_kind = "sender_is_profit_recipient"
            strength = "high"
        elif recipient and entry_contract and recipient == entry_contract:
            relation_kind = "entry_contract_received_profit"
            strength = "medium"
        elif tx_sender and entry_contract:
            relation_kind = "tx_sender_called_entry_contract"
            strength = "medium" if entry_contract in direct_callees else "low"
        else:
            relation_kind = "no_direct_link"
            strength = "low"
        paths.append(omit_empty({
            "from": "tx_sender",
            "to": entry_contract,
            "to_profit_recipient": recipient,
            "relation_kind": relation_kind,
            "relation_strength": strength,
            "evidence_ids": evidence_ids,
        }))

    hints = []
    if tx_sender:
        hints.append(_controller_hint(tx_sender, "tx_sender", label_map, "high"))
    if entry_contract:
        hints.append(_controller_hint(entry_contract, "entry_contract", label_map, "high"))
    for row in positive[:max_items]:
        hints.append(_controller_hint(row.get("address"), "profit_recipient", label_map, "medium"))
    for row in negative[:max_items]:
        hints.append(_controller_hint(row.get("address"), "loss_source", label_map, "medium"))

    return omit_empty({
        "evidence_id": "beneficiary_controller:summary",
        "tx_sender": tx_sender,
        "entry_contract": entry_contract,
        "direct_callees": direct_callees[:max_items],
        "top_positive_delta_addresses": positive[:max_items],
        "top_negative_delta_addresses": negative[:max_items],
        "beneficiary_paths": paths[:max_items],
        "controller_hints": _unique_dicts(hints)[: max_items * 2],
        "notes": [
            "Neutral beneficiary/controller summary; not an attribution judgment.",
            "No cross-transaction relationship is inferred unless explicitly available.",
            "Same-transaction call/fund-flow links do not prove control by themselves.",
        ],
    })


def render_classification_digest_view(
    syn: Dict[str, Any],
    *,
    operation_summary_view: Dict[str, Any],
    critical_call_view: Any,
    unknown_selector_view: Any,
    value_release_view: Any,
    participant_net_delta_view: Any,
    contribution_vs_payout_view: Any,
    protocol_accounting_outcome_view: Any,
    price_relevant_state_view: Any,
    amm_reserve_transition_view: Any,
    flash_or_atomic_capital_view: Any,
    semantic_state_delta_view: Any,
    transfer_event_view: Any,
    market_mechanism_profile_view: Any = None,
    reentrancy_state_order_summary_view: Any = None,
) -> Dict[str, Any]:
    """Neutral cross-view signal digest. It is not a final label decision."""
    top_signals: List[Dict[str, Any]] = []

    def add_signal(signal: str, evidence_ids: Iterable[Any], view_sources: List[str], why: str = "") -> None:
        ids = _unique(str(eid) for eid in evidence_ids if eid)
        if not ids:
            return
        top_signals.append(omit_empty({
            "signal": signal,
            "evidence_ids": ids[:7],
            "support_count": len(ids),
            "view_sources": view_sources,
            "why": why,
        }))

    flash_rows = _view_rows(flash_or_atomic_capital_view)
    add_signal("flash or atomic capital pattern", _extract_ids_from_rows(flash_rows), ["flash_or_atomic_capital_view"])

    value_rows = _view_rows(value_release_view)
    add_signal("value release under sensitive calls", _extract_ids_from_rows(value_rows), ["value_release_view"])

    price_rows = _view_rows(price_relevant_state_view)
    add_signal("price/accounting relevant state delta", _extract_ids_from_rows(price_rows), ["price_relevant_state_view", "semantic_state_delta_view"])

    amm_rows = _view_rows(amm_reserve_transition_view)
    add_signal("AMM reserve or market-state transition", _extract_ids_from_rows(amm_rows), ["amm_reserve_transition_view"])

    market_profile_rows = _view_rows(market_mechanism_profile_view)
    add_signal(
        "market mechanism profile candidate",
        _extract_ids_from_rows(market_profile_rows),
        ["market_mechanism_profile_view"],
        "compact market-root candidate profile; not a verdict",
    )

    re_rows = [
        row
        for row in _view_rows(reentrancy_state_order_summary_view)
        if row.get("formal_candidate") is True
    ]
    add_signal(
        "formal reentrancy state-order candidate",
        _extract_ids_from_rows(re_rows),
        ["reentrancy_state_order_summary_view"],
        "read-only recursion and standard callback shape alone are excluded",
    )

    unknown_rows = _view_rows(unknown_selector_view)
    unknown_with_context = [
        row for row in unknown_rows
        if row.get("nearby_event_ids") or row.get("nearby_state_ids") or row.get("children_summary")
    ]
    add_signal("unknown selector with nearby state/event context", _extract_ids_from_rows(unknown_with_context), ["unknown_selector_view"])

    cp_rows = _view_rows(contribution_vs_payout_view)
    add_signal("contribution versus payout accounting row", _extract_ids_from_rows(cp_rows), ["contribution_vs_payout_view"])

    pa_outcome_rows = _view_rows(protocol_accounting_outcome_view)
    add_signal(
        "protocol accounting outcome candidate",
        _extract_ids_from_rows(pa_outcome_rows),
        ["protocol_accounting_outcome_view"],
        "state-level reward/share/debt/vault/liability outcome candidate; not a verdict",
    )

    transfer_rows = _view_rows(transfer_event_view)
    semantic_rows = _view_rows(semantic_state_delta_view)
    if transfer_rows and semantic_rows:
        add_signal(
            "token transfer and state-delta evidence both present",
            _extract_ids_from_rows(transfer_rows[:12]) + _extract_ids_from_rows(semantic_rows[:12]),
            ["transfer_event_view", "semantic_state_delta_view"],
            "compare event/fundflow/state deltas for token-semantic or accounting mismatch",
        )

    participant_rows = _view_rows(participant_net_delta_view)
    add_signal("participant net delta summary present", _extract_ids_from_rows(participant_rows), ["participant_net_delta_view"])

    operation_flags = {}
    if isinstance(operation_summary_view, dict):
        operation_flags = {
            key: value
            for key, value in operation_summary_view.items()
            if key.startswith("has_")
            or key in {
                "primary_operation",
                "entry_function",
                "entry_selector",
                "entry_decode_status",
                "single_token_flow",
                "packet_trace_truncated",
            }
        }

    return {
        "evidence_id": "classification_digest:0",
        "available": True,
        "operation_flags": compact_value(operation_flags, max_dict_items=40, max_list_len=8),
        "top_signals": top_signals[:20],
        "anti_overfit_notes": [
            "Profit alone is not root-cause evidence.",
            "Flashloan presence alone is not sufficient.",
            "A callback shape alone is not enough to distinguish access control from insufficient validation.",
            "This digest aggregates evidence ids only and does not assign a final label.",
        ],
    }


def _view_rows(view: Any) -> List[Dict[str, Any]]:
    if isinstance(view, list):
        return [row for row in view if isinstance(row, dict)]
    if isinstance(view, dict):
        for key in ("rows", "records", "events", "deltas", "transfers", "pairs"):
            value = view.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if view.get("evidence_id"):
            return [view]
    return []


def _extract_ids_from_rows(rows: Iterable[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for row in rows or []:
        if row.get("evidence_id"):
            ids.append(str(row.get("evidence_id")))
        for key in (
            "related_evidence_ids",
            "state_order_evidence_ids",
            "value_out_evidence_ids",
            "target_balance_change_evidence_ids",
            "recipient_balance_change_evidence_ids",
            "evidence_ids",
        ):
            value = row.get(key)
            if isinstance(value, list):
                ids.extend(str(item) for item in value if item)
    return _unique(ids)


def _row_haystack(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key, ""))
        for key in (
            "event",
            "event_signature",
            "function",
            "function_name",
            "decoded_function",
            "parent_function",
            "why_included",
            "variable",
            "variable_hint",
            "contract_label",
            "address_label",
            "transition_summary",
            "mechanism_hint",
            "origin_hint",
        )
    ).lower()


def _row_address(row: Dict[str, Any]) -> str:
    for key in (
        "pair",
        "contract",
        "address",
        "callee",
        "target_contract",
        "token",
    ):
        addr = normalize_address(row.get(key))
        if addr:
            return addr
    return ""


def _market_related_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    keywords: Iterable[str],
    limit: int,
) -> List[Dict[str, Any]]:
    lowered = [str(item).lower() for item in keywords]
    selected: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        haystack = _row_haystack(row)
        if any(keyword in haystack for keyword in lowered):
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def render_market_mechanism_profile_view(
    *,
    price_relevant_state_view: Any,
    amm_reserve_transition_view: Any,
    critical_call_view: Any,
    event_view: Any,
    token_semantic_delta_summary_view: Any = None,
    token_accounting_origin_view: Any = None,
    value_release_view: Any = None,
    contribution_vs_payout_view: Any = None,
    participant_net_delta_view: Any = None,
    external_fundflow_view: Any = None,
    flash_or_atomic_capital_view: Any = None,
    max_profiles: int = 24,
) -> Dict[str, Any]:
    """Build compact market-root candidate profiles before judging.

    The rows are candidate anchors only. They help C1/C2/C3 keep one market
    object/profile in view, but they do not decide whether an attack occurred.
    """
    profiles: List[Dict[str, Any]] = []
    price_rows = _view_rows(price_relevant_state_view)
    amm_rows = _view_rows(amm_reserve_transition_view)
    critical_rows = _view_rows(critical_call_view)
    event_rows = _view_rows(event_view)
    token_rows = _view_rows(token_semantic_delta_summary_view)
    token_origin_rows = _view_rows(token_accounting_origin_view)
    value_rows = _view_rows(value_release_view)
    contribution_rows = _view_rows(contribution_vs_payout_view)
    participant_rows = _view_rows(participant_net_delta_view)
    fundflow_rows = _view_rows(external_fundflow_view)
    flash_rows = _view_rows(flash_or_atomic_capital_view)

    market_consumer_rows = _market_related_rows(
        critical_rows + event_rows,
        keywords=(
            "swap",
            "skim",
            "sync",
            "mint",
            "burn",
            "borrow",
            "redeem",
            "withdraw",
            "liquidat",
            "oracle",
            "price",
            "settle",
            "valuation",
            "exchange",
        ),
        limit=20,
    )
    outcome_rows = (value_rows + contribution_rows + participant_rows + fundflow_rows)[:24]
    competing_rows = (token_rows + token_origin_rows)[:16]

    def add_profile(
        *,
        profile_type: str,
        source_row: Dict[str, Any],
        source_kind: str,
        source_evidence_ids: Iterable[Any],
        reason: str,
    ) -> None:
        if len(profiles) >= max_profiles:
            return
        source_ids = _unique(str(item) for item in source_evidence_ids if item)[:16]
        consumer_ids = _extract_ids_from_rows(market_consumer_rows)[:16]
        outcome_ids = _extract_ids_from_rows(outcome_rows)[:16]
        competing_ids = _extract_ids_from_rows(competing_rows)[:12]
        evidence_ids = _unique(source_ids + consumer_ids + outcome_ids)[:24]
        if not evidence_ids:
            return
        profile_index = len(profiles) + 1
        strength_score = int(bool(source_ids)) + int(bool(consumer_ids)) + int(bool(outcome_ids))
        candidate_strength = "high" if strength_score >= 3 else "medium" if strength_score == 2 else "low"
        profiles.append(omit_empty({
            "evidence_id": f"market_profile:{profile_type}:{profile_index}",
            "profile_id": f"market_profile_{profile_index}",
            "profile_type": profile_type,
            "market_source": _row_address(source_row),
            "market_source_label": source_row.get("pair_label")
            or source_row.get("contract_label")
            or source_row.get("address_label"),
            "source_kind": source_kind,
            "source_evidence_ids": source_ids,
            "consumer_evidence_ids": consumer_ids,
            "outcome_evidence_ids": outcome_ids,
            "all_profile_evidence_ids": evidence_ids,
            "candidate_strength": candidate_strength,
            "profile_reason": reason,
            "competing_root_hints": _unique([
                "token_semantic_or_token_accounting_origin"
                for _ in competing_ids[:1]
            ] + [
                "flash_or_atomic_capital_present"
                for _ in _extract_ids_from_rows(flash_rows)[:1]
            ]),
            "competing_root_evidence_ids": competing_ids,
            "limitation": (
                "Profile rows are neutral anchors. Judge must still prove the "
                "same selected profile is distorted/misused, consumed, and tied "
                "to outcome; generic profit or swap volume is not sufficient."
            ),
        }))

    for row in amm_rows[:max_profiles]:
        source_ids = [row.get("evidence_id")]
        source_ids.extend(_extract_ids_from_rows(_view_rows(row)))
        add_profile(
            profile_type="pool_accounting_reserve_extraction",
            source_row=row,
            source_kind="amm_or_pair_visible_balance",
            source_evidence_ids=source_ids,
            reason=(
                "AMM/pool event or reserve-like transition may anchor a "
                "skim/donate/sync/swap/pair-balance market profile."
            ),
        )

    for row in price_rows[:max_profiles]:
        child_ids = _extract_ids_from_rows(_view_rows(row))
        add_profile(
            profile_type="price_state",
            source_row=row,
            source_kind="price_or_valuation_state",
            source_evidence_ids=[row.get("evidence_id"), *child_ids],
            reason=(
                "Price, oracle, exchange-rate, reserve, liquidity, collateral, "
                "or debt state may anchor a market-state profile."
            ),
        )

    if flash_rows and market_consumer_rows and len(profiles) < max_profiles:
        add_profile(
            profile_type="ordering_mev_arbitrage",
            source_row=market_consumer_rows[0],
            source_kind="atomic_ordering_or_mev_context",
            source_evidence_ids=(
                _extract_ids_from_rows(flash_rows[:4])
                + _extract_ids_from_rows(market_consumer_rows[:8])
            ),
            reason=(
                "Atomic capital plus market-facing consumer operations may "
                "anchor an ordering/MEV/arbitrage-like profile."
            ),
        )

    return {
        "evidence_id": "market_mechanism_profile:summary",
        "available": True,
        "profile_count": len(profiles),
        "profiles": profiles[:max_profiles],
        "global_competing_root_hints": _unique([
            "token_semantic_or_token_accounting_origin"
            for _ in _extract_ids_from_rows(competing_rows)[:1]
        ] + [
            "flash_or_atomic_capital_present"
            for _ in _extract_ids_from_rows(flash_rows)[:1]
        ]),
        "notes": [
            "Use profile_id/evidence_ids to keep C1/C2/C3 on one market object.",
            "This view is not an attack verdict and is built before Judge candidates.",
            "Token/accounting/flash hints are boundary warnings, not automatic exclusions.",
        ],
    }


def _mechanism_hint(mechanism: str, strength: str, why: str, evidence_ids: Iterable[Any]) -> Dict[str, Any]:
    return {
        "mechanism": mechanism,
        "hint_strength": strength,
        "why": why,
        "evidence_ids": _unique(str(eid) for eid in evidence_ids if eid)[:20],
    }


def _labels_from_address_view(address_labels: List[Dict[str, Any]]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in address_labels or []:
        if not isinstance(row, dict):
            continue
        addr = normalize_address(row.get("address"))
        if addr:
            labels[addr] = str(row.get("sanitized_label") or row.get("raw_label") or "")
    return labels


def _first_call_address(trace_view: List[Dict[str, Any]]) -> str:
    for row in trace_view or []:
        if isinstance(row, dict) and row.get("type") in CALL_TYPES:
            return normalize_address(row.get("address") or row.get("callee"))
    return ""


def _first_call_evidence_id(trace_view: List[Dict[str, Any]]) -> str:
    for row in trace_view or []:
        if isinstance(row, dict) and row.get("type") in CALL_TYPES:
            return str(row.get("evidence_id") or "")
    return ""


def _direct_callees(
    tx_sender: str,
    entry_contract: str,
    trace_view: List[Dict[str, Any]],
    critical_call_view: List[Dict[str, Any]],
) -> List[str]:
    callees = []
    for row in list(trace_view or []) + list(critical_call_view or []):
        if not isinstance(row, dict):
            continue
        caller = normalize_address(row.get("caller"))
        callee = normalize_address(row.get("address") or row.get("callee"))
        if not callee:
            continue
        if tx_sender and caller == tx_sender:
            callees.append(callee)
        elif entry_contract and not callees:
            callees.append(entry_contract)
    return _unique(callees)


def _top_delta_addresses(
    participant_rows: List[Dict[str, Any]],
    *,
    direction: str,
    limit: int,
) -> List[Dict[str, Any]]:
    scored = []
    for row in participant_rows or []:
        if not isinstance(row, dict):
            continue
        relevant = []
        score = Decimal(0)
        for delta in row.get("deltas", []) or []:
            if not isinstance(delta, dict) or delta.get("direction") != direction:
                continue
            relevant.append(delta)
            amount = parse_decimal(str(delta.get("delta", "")).lstrip("+-"))
            if amount is not None:
                score += abs(amount)
        if not relevant:
            continue
        scored.append((
            score,
            omit_empty({
                "address": normalize_address(row.get("address")),
                "label": row.get("label"),
                "deltas": relevant[:8],
                "related_evidence_ids": list(row.get("related_evidence_ids", []) or []),
            }),
        ))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:limit]]


def _add_profit_loss_hints(
    positive: List[Dict[str, Any]],
    negative: List[Dict[str, Any]],
    profit_loss_view: Any,
    label_map: Dict[str, str],
) -> None:
    if not isinstance(profit_loss_view, dict):
        return
    top_profit = normalize_address(profit_loss_view.get("top_profit_address"))
    top_loss = normalize_address(profit_loss_view.get("top_loss_address"))
    if top_profit and not any(row.get("address") == top_profit for row in positive):
        positive.append({
            "address": top_profit,
            "label": label_map.get(top_profit),
            "related_evidence_ids": ["profit_loss_view"],
        })
    if top_loss and not any(row.get("address") == top_loss for row in negative):
        negative.append({
            "address": top_loss,
            "label": label_map.get(top_loss),
            "related_evidence_ids": ["profit_loss_view"],
        })


def _controller_hint(address: Any, hint_type: str, label_map: Dict[str, str], confidence: str) -> Dict[str, Any]:
    addr = normalize_address(address)
    return omit_empty({
        "address": addr,
        "hint_type": hint_type,
        "label": label_map.get(addr),
        "confidence": confidence,
    })


def _unique_dicts(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = (row.get("address"), row.get("hint_type"))
        if not row.get("address") or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _unique(values: Iterable[Any]) -> List[Any]:
    out = []
    seen = set()
    for value in values:
        if value in (None, "") or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


# ============================================================
# Packet builder
# ============================================================

def build_compact_evidence_packet(
    syn: Dict[str, Any],
    fundflow_obj: Any = None,
    profit_loss_obj: Any = None,
    include_views: Optional[List[str]] = None,
    packet_config: Optional[Dict[str, Any]] = None,
    attack_label: Optional[str] = None,
    packet_profile: str = "full",
) -> Dict[str, Any]:
    include_views = resolve_packet_views(
        include_views=include_views,
        attack_label=attack_label,
        packet_profile=packet_profile,
    )
    cfg = {**DEFAULT_PACKET_CONFIG, **(packet_config or {})}
    trace_index = build_trace_index(syn.get("trace", {}))
    packet: Dict[str, Any] = {}

    if fundflow_obj is None and syn.get("fund_flow") not in (None, "", [], {}):
        fundflow_obj = syn.get("fund_flow")

    if "tx_card" in include_views:
        packet["tx_card"] = render_tx_card(syn)
    if "address_labels" in include_views:
        packet["address_labels"] = render_address_label_view(
            syn,
            max_items=int(cfg["max_address_labels"]),
            sanitize=True,
        )
    if "token_info" in include_views:
        packet["token_info"] = render_token_info_view(syn, max_items=int(cfg["max_token_info"]))
    if "evidence_adequacy_view" in include_views:
        packet["evidence_adequacy_view"] = render_evidence_adequacy_view(syn, trace_index, cfg)
    if "trace_outline_view" in include_views:
        packet["trace_outline_view"] = render_trace_outline_view(syn, trace_index, max_items=int(cfg.get("max_trace_outline_nodes", 300)))
    if "trace_view" in include_views:
        packet["trace_view"] = render_trace_view(syn, trace_index, max_nodes=int(cfg["max_trace_nodes"]))
    if "critical_call_view" in include_views:
        ccv_result = render_critical_call_view(
            syn,
            trace_index,
            max_items=int(cfg["max_critical_calls"]),
        )
        if isinstance(ccv_result, list):
            packet["critical_call_view"] = ccv_result
        else:
            packet["critical_call_view"] = ccv_result.get("rows", [])
            packet["critical_call_view_absence"] = ccv_result.get("summary")
    if "critical_call_argument_view" in include_views:
        packet["critical_call_argument_view"] = render_critical_call_argument_view(
            syn,
            trace_index,
            critical_call_view=packet.get("critical_call_view"),
            max_items=int(cfg.get("max_critical_call_arguments", 80)),
        )
    if "reentrancy_state_order_summary_view" in include_views:
        summary_row_limit = int(cfg["max_reentrancy_state_order_summary_rows"])
        if "reentrancy_candidate_catalog_view" in include_views:
            summary_row_limit = max(
                summary_row_limit,
                int(cfg["max_reentrancy_candidate_catalog_rows"]),
            )
        packet["reentrancy_state_order_summary_view"] = render_reentrancy_state_order_summary_view(
            syn,
            trace_index,
            critical_call_view=packet.get("critical_call_view", []),
            max_items=summary_row_limit,
        )
    if "reentrancy_state_order_view" in include_views:
        packet["reentrancy_state_order_view"] = render_reentrancy_state_order_view(
            syn,
            trace_index,
            critical_call_view=packet.get("critical_call_view", []),
            max_items=int(cfg["max_reentrancy_state_order_rows"]),
            max_state_nodes_per_item=int(cfg["max_reentrancy_state_nodes_per_row"]),
            include_details=bool(cfg.get("include_reentrancy_state_order_details")),
        )
    if "unknown_selector_view" in include_views:
        packet["unknown_selector_view"] = render_unknown_selector_view(
            syn,
            trace_index,
            max_items=int(cfg["max_unknown_selectors"]),
        )
    if "event_view" in include_views:
        packet["event_view"] = render_event_view(syn, trace_index, max_events=int(cfg["max_events"]))
    if "state_change_view" in include_views:
        packet["state_change_view"] = render_state_change_view(
            syn,
            trace_index,
            max_items=int(cfg["max_state_rows"]),
        )
    if "semantic_state_delta_view" in include_views:
        packet["semantic_state_delta_view"] = render_semantic_state_delta_view(
            syn,
            trace_index,
            token_info=syn.get("token_info"),
            max_items=int(cfg["max_semantic_state_rows"]),
        )
    _ssdv = packet.get("semantic_state_delta_view")
    _ssdv_for_downstream = [] if isinstance(_ssdv, dict) else _ssdv
    if "source_unavailable_auth_view" in include_views:
        packet["source_unavailable_auth_view"] = render_source_unavailable_auth_view(
            syn,
            trace_index,
            critical_call_view=packet.get("critical_call_view"),
            unknown_selector_view=packet.get("unknown_selector_view"),
            semantic_state_delta_view=_ssdv_for_downstream,
            max_items=int(cfg.get("max_source_unavailable_auth_rows", 80)),
        )
    if "transfer_event_view" in include_views:
        packet["transfer_event_view"] = render_transfer_event_view(
            syn,
            trace_index,
            max_items=int(cfg["max_transfers"]),
        )
    if "token_semantic_delta_summary_view" in include_views:
        packet["token_semantic_delta_summary_view"] = render_token_semantic_delta_summary_view(
            syn,
            trace_index,
            semantic_state_delta_view=_ssdv_for_downstream,
            transfer_event_view=packet.get("transfer_event_view"),
            token_info=syn.get("token_info"),
            max_items=int(cfg.get("max_token_semantic_delta_rows", 80)),
        )
    if "token_accounting_origin_view" in include_views:
        packet["token_accounting_origin_view"] = render_token_accounting_origin_view(
            syn,
            trace_index,
            token_semantic_delta_summary_view=packet.get("token_semantic_delta_summary_view"),
            semantic_state_delta_view=_ssdv_for_downstream,
            transfer_event_view=packet.get("transfer_event_view"),
            critical_call_view=packet.get("critical_call_view"),
            max_items=int(cfg.get("max_token_accounting_origin_rows", 80)),
        )
    if "price_relevant_state_view" in include_views:
        packet["price_relevant_state_view"] = render_price_relevant_state_view(
            syn,
            trace_index,
            semantic_state_delta_view=_ssdv_for_downstream,
            max_contracts=int(cfg["max_price_state_contracts"]),
        )
    if "amm_reserve_transition_view" in include_views:
        packet["amm_reserve_transition_view"] = render_amm_reserve_transition_view(
            syn,
            trace_index,
            max_pairs=int(cfg["max_amm_pairs"]),
        )
    if "external_fundflow_view" in include_views:
        packet["external_fundflow_view"] = render_external_fundflow_view(
            fundflow_obj,
            max_items=int(cfg["max_fundflow_records"]),
        )
    if "profit_loss_view" in include_views:
        packet["profit_loss_view"] = render_profit_loss_view(profit_loss_obj)
    if "participant_net_delta_view" in include_views:
        packet["participant_net_delta_view"] = render_participant_net_delta_view(
            syn,
            fundflow_obj,
            transfer_event_view=packet.get("transfer_event_view"),
            profit_loss_obj=profit_loss_obj,
            address_labels=packet.get("address_labels"),
            max_items=int(cfg["max_participants"]),
        )
    if "value_release_view" in include_views:
        packet["value_release_view"] = render_value_release_view(
            syn=syn,
            trace_index=trace_index,
            critical_call_view=packet.get("critical_call_view", []),
            transfer_event_view=packet.get("transfer_event_view", []),
            participant_net_delta_view=packet.get("participant_net_delta_view", []),
            semantic_state_delta_view=_ssdv_for_downstream,
            max_items=int(cfg["max_value_release_rows"]),
            max_release_records_per_row=int(cfg["max_value_release_records_per_row"]),
            max_state_rows_per_row=int(cfg["max_value_release_state_rows"]),
        )
    if "reentrancy_candidate_catalog_view" in include_views:
        packet[
            "reentrancy_candidate_catalog_view"
        ] = render_reentrancy_candidate_catalog_view(
            packet.get("reentrancy_state_order_summary_view", {}),
            packet.get("value_release_view", []),
            max_items=int(cfg["max_reentrancy_candidate_catalog_rows"]),
        )
    if "contribution_vs_payout_view" in include_views:
        packet["contribution_vs_payout_view"] = render_contribution_vs_payout_view(
            syn,
            participant_net_delta_view=packet.get("participant_net_delta_view", []),
            semantic_state_delta_view=_ssdv_for_downstream,
            critical_call_view=packet.get("critical_call_view", []),
            max_items=int(cfg["max_contribution_rows"]),
        )
    if "protocol_accounting_outcome_view" in include_views:
        packet["protocol_accounting_outcome_view"] = render_protocol_accounting_outcome_view(
            syn=syn,
            semantic_state_delta_view=_ssdv_for_downstream,
            critical_call_view=packet.get("critical_call_view", []),
            value_release_view=packet.get("value_release_view", []),
            contribution_vs_payout_view=packet.get("contribution_vs_payout_view", []),
            participant_net_delta_view=packet.get("participant_net_delta_view", []),
            max_items=int(cfg.get("max_protocol_accounting_outcomes", 48)),
        )
    if "flash_or_atomic_capital_view" in include_views:
        packet["flash_or_atomic_capital_view"] = render_flash_or_atomic_capital_view(
            syn,
            trace_index,
            fundflow_obj,
            critical_call_view=packet.get("critical_call_view"),
            max_items=int(cfg["max_atomic_capital_rows"]),
        )
    if "market_mechanism_profile_view" in include_views:
        packet["market_mechanism_profile_view"] = render_market_mechanism_profile_view(
            price_relevant_state_view=packet.get("price_relevant_state_view", []),
            amm_reserve_transition_view=packet.get("amm_reserve_transition_view", []),
            critical_call_view=packet.get("critical_call_view", []),
            event_view=packet.get("event_view", []),
            token_semantic_delta_summary_view=packet.get("token_semantic_delta_summary_view", []),
            token_accounting_origin_view=packet.get("token_accounting_origin_view", []),
            value_release_view=packet.get("value_release_view", []),
            contribution_vs_payout_view=packet.get("contribution_vs_payout_view", []),
            participant_net_delta_view=packet.get("participant_net_delta_view", []),
            external_fundflow_view=packet.get("external_fundflow_view", {}),
            flash_or_atomic_capital_view=packet.get("flash_or_atomic_capital_view", []),
            max_profiles=int(cfg.get("max_market_mechanism_profiles", 24)),
        )
    if "beneficiary_controller_view" in include_views:
        packet["beneficiary_controller_view"] = render_beneficiary_controller_view(
            syn=syn,
            tx_card=packet.get("tx_card", {}),
            address_labels=packet.get("address_labels", []),
            trace_view=packet.get("trace_view", []),
            critical_call_view=packet.get("critical_call_view", []),
            participant_net_delta_view=packet.get("participant_net_delta_view", []),
            contribution_vs_payout_view=packet.get("contribution_vs_payout_view", []),
            external_fundflow_view=packet.get("external_fundflow_view", {}),
            profit_loss_view=packet.get("profit_loss_view", {}),
            transfer_event_view=packet.get("transfer_event_view", []),
            max_items=int(cfg["max_beneficiary_rows"]),
        )

    if "operation_summary_view" in include_views:
        packet["operation_summary_view"] = render_operation_summary_view(
            syn=syn,
            trace_index=trace_index,
            critical_call_view=packet.get("critical_call_view", []),
            transfer_event_view=packet.get("transfer_event_view", []),
            external_fundflow_view=packet.get("external_fundflow_view", {}),
            amm_reserve_transition_view=packet.get("amm_reserve_transition_view"),
            flash_or_atomic_capital_view=packet.get("flash_or_atomic_capital_view"),
            value_release_view=packet.get("value_release_view"),
            fundflow_obj=fundflow_obj,
        )
    if "classification_digest_view" in include_views:
        packet["classification_digest_view"] = render_classification_digest_view(
            syn=syn,
            operation_summary_view=packet.get("operation_summary_view", {}),
            critical_call_view=packet.get("critical_call_view", []),
            unknown_selector_view=packet.get("unknown_selector_view", []),
            value_release_view=packet.get("value_release_view", []),
            participant_net_delta_view=packet.get("participant_net_delta_view", []),
            contribution_vs_payout_view=packet.get("contribution_vs_payout_view", []),
            protocol_accounting_outcome_view=packet.get("protocol_accounting_outcome_view", {}),
            price_relevant_state_view=packet.get("price_relevant_state_view", []),
            amm_reserve_transition_view=packet.get("amm_reserve_transition_view", []),
            market_mechanism_profile_view=packet.get("market_mechanism_profile_view", {}),
            flash_or_atomic_capital_view=packet.get("flash_or_atomic_capital_view", []),
            semantic_state_delta_view=_ssdv_for_downstream,
            transfer_event_view=packet.get("transfer_event_view", []),
            reentrancy_state_order_summary_view=packet.get("reentrancy_state_order_summary_view", {}),
        )

    return packet


def resolve_input_paths(tx_hash: str, base_dir: str | Path = "data/cache") -> Dict[str, Optional[Path]]:
    tx = norm_hash(tx_hash)
    base = Path(base_dir)

    synthesized_candidates = [
        base / "systhesized" / f"{tx}_synthesized.json",
        base / "synthesized" / f"{tx}_synthesized.json",
        base / "systhesized" / f"{tx}_synthessized.json",
        base / "synthesized" / f"{tx}_synthessized.json",
    ]
    fundflow_candidates = [
        base / "fund_flow" / f"{tx}_fundflow.json",
        base / "fund_flow" / f"{tx}_fund_flow.json",
        base / "fund_flows" / f"{tx}_fundflow.json",
        base / "fund_flows" / f"{tx}_fund_flow.json",
        base / "fundflow" / f"{tx}_fundflow.json",
        base / "fundflow" / f"{tx}_fund_flow.json",
    ]
    profit_loss_candidates = [
        base / "profit_loss" / f"{tx}_top-profit-loss.json",
        base / "profit_loss" / f"{tx}_top_profit_loss.json",
    ]
    packet_path = base / "packet" / f"{tx}_packet.json"
    evidence_store_path = base / "packet" / f"{tx}_evidence_store.json"

    return {
        "synthesized": find_existing_file(synthesized_candidates),
        "fundflow": find_existing_file(fundflow_candidates),
        "profit_loss": find_existing_file(profit_loss_candidates),
        "packet": packet_path,
        "evidence_store": evidence_store_path,
    }


def build_or_load_packet(
    tx_hash: str,
    base_dir: str | Path = "data/cache",
    force_rebuild: bool = False,
    include_views: Optional[List[str]] = None,
    required_views: Optional[List[str]] = None,
    packet_config: Optional[Dict[str, Any]] = None,
    attack_label: Optional[str] = None,
    packet_profile: str = "full",
    write_evidence_store: bool = True,
    include_view_dependencies: bool = True,
    include_heavy_optional_dependencies: bool = False,
) -> Dict[str, Any]:
    tx = norm_hash(tx_hash)
    paths = resolve_input_paths(tx, base_dir=base_dir)
    packet_path = paths["packet"]
    requested_include_views = resolve_packet_views(
        include_views=include_views,
        attack_label=attack_label,
        packet_profile=packet_profile,
    )
    requested_required_views = _normalize_required_views(required_views)
    resolved_include_views = _unique(
        list(requested_include_views) + list(requested_required_views)
    )
    if include_view_dependencies:
        resolved_include_views = resolve_view_dependency_closure(
            resolved_include_views,
            include_optional=True,
            include_heavy_optional=include_heavy_optional_dependencies,
        )
    dependency_expanded_views = [
        view
        for view in resolved_include_views
        if view not in set(requested_include_views) | set(requested_required_views)
    ]
    cache_required_views = (
        requested_required_views
        if requested_required_views
        else (_normalize_required_views(include_views) if include_views is not None else [])
    )
    cache_check_views = _unique(list(cache_required_views) + list(dependency_expanded_views))

    if packet_path and packet_path.exists() and not force_rebuild:
        packet = load_json(packet_path)
        cache_format_matches = (
            str(packet.get("packet_format") or "") == PACKET_FORMAT_VERSION
        )
        if not cache_format_matches:
            print(
                f"[PacketBuilder] cached packet for tx={tx} uses "
                f"{packet.get('packet_format') or 'unknown format'}; rebuilding "
                f"as {PACKET_FORMAT_VERSION}"
            )
        elif write_evidence_store and paths.get("evidence_store"):
            packet = _ensure_cached_packet_evidence_store(
                packet=packet,
                paths=paths,
                packet_path=packet_path,
                packet_config=packet_config,
            )
        missing_views = (
            _missing_required_packet_views(packet, cache_check_views)
            if cache_format_matches
            else ["packet_format_upgrade"]
        )
        if cache_format_matches and not missing_views:
            packet.setdefault("cache_policy", {})
            packet["cache_policy"]["loaded_from_cache"] = True
            packet["cache_policy"]["required_views_checked"] = list(cache_required_views)
            packet["cache_policy"]["required_views_missing"] = []
            packet["cache_policy"]["dependency_views_checked"] = list(dependency_expanded_views)
            packet["cache_policy"]["dependency_views_missing"] = []
            packet.setdefault(
                "view_cost_summary",
                summarize_packet_view_costs(packet.get("views", {}) or {}),
            )
            if not isinstance(packet.get("build_config"), dict):
                cached_views = list((packet.get("views", {}) or {}).keys())
                packet["build_config"] = {
                    "resolved_include_views": cached_views,
                    "requested_required_views": list(requested_required_views),
                    "required_views_hash": _views_hash(requested_required_views),
                    "view_count": len(cached_views),
                    "backfilled_from_cache": True,
                }
                dump_json(packet, packet_path)
            return packet
        if cache_format_matches:
            print(
                f"[PacketBuilder] cached packet for tx={tx} is missing required views "
                f"{missing_views}; rebuilding packet"
            )

    if not paths["synthesized"]:
        raise FileNotFoundError(
            f"Cannot find synthesized file for tx={tx}. Checked under "
            f"{Path(base_dir) / 'systhesized'} and {Path(base_dir) / 'synthesized'}."
        )

    syn = load_json(paths["synthesized"])
    fundflow_obj = load_json(paths["fundflow"]) if paths["fundflow"] else None
    profit_loss_obj = load_json(paths["profit_loss"]) if paths["profit_loss"] else None
    cfg = {**DEFAULT_PACKET_CONFIG, **(packet_config or {})}
    trace_index = build_trace_index(syn.get("trace", {}))
    dictionary = build_packet_dictionary(syn, trace_index)
    evidence_store_ref = None
    if write_evidence_store and paths.get("evidence_store"):
        evidence_store = build_evidence_store(
            syn,
            trace_index,
            include_raw=bool(cfg.get("include_raw_evidence_store")),
            packet_config=cfg,
        )
        dump_json(evidence_store, paths["evidence_store"])
        evidence_store_ref = {
            "path": str(paths["evidence_store"]),
            "format": evidence_store.get("format", "evotx_evidence_store_v1"),
            "count": len(evidence_store.get("evidence", {}) or {}),
        }

    views = build_compact_evidence_packet(
        syn=syn,
        fundflow_obj=fundflow_obj,
        profit_loss_obj=profit_loss_obj,
        include_views=resolved_include_views,
        packet_config=cfg,
        attack_label=attack_label,
        packet_profile=packet_profile,
    )

    build_config = {
        "requested_include_views": list(requested_include_views),
        "requested_required_views": list(requested_required_views),
        "resolved_include_views": list(resolved_include_views),
        "dependency_expanded_views": list(dependency_expanded_views),
        "dependency_policy": {
            "include_view_dependencies": bool(include_view_dependencies),
            "include_heavy_optional_dependencies": bool(include_heavy_optional_dependencies),
        },
        "view_manifest_version": VIEW_MANIFEST_VERSION,
        "required_views_hash": _views_hash(requested_required_views),
        "view_count": len(views),
    }
    view_cost_summary = summarize_packet_view_costs(views)
    packet = {
        "packet_format": PACKET_FORMAT_VERSION,
        "transaction_hash": tx,
        "built_at": int(time.time()),
        "cache_policy": {
            "force_rebuild": force_rebuild,
            "loaded_from_cache": False,
            "required_views_checked": list(cache_required_views),
            "required_views_missing": [],
        },
        "sources": {
            "synthesized": str(paths["synthesized"]) if paths["synthesized"] else None,
            "fundflow": str(paths["fundflow"]) if paths["fundflow"] else None,
            "profit_loss": str(paths["profit_loss"]) if paths["profit_loss"] else None,
        },
        "packet_profile": packet_profile,
        "build_config": build_config,
        "view_cost_summary": view_cost_summary,
        "dictionary": dictionary,
        "evidence_store": evidence_store_ref,
        "views": views,
    }
    dump_json(packet, packet_path)
    return packet


def _normalize_required_views(values: Optional[Iterable[Any]]) -> List[str]:
    if values is None:
        return []
    return [
        view
        for view in _unique(str(value).strip() for value in values if str(value).strip())
        if view in DEFAULT_PACKET_VIEWS
    ]


def _missing_required_packet_views(
    packet: Dict[str, Any],
    required_views: List[str],
) -> List[str]:
    if not required_views:
        return []
    views = packet.get("views", {}) if isinstance(packet, dict) else {}
    if not isinstance(views, dict):
        return list(required_views)
    return [view for view in required_views if view not in views]


def _views_hash(views: List[str]) -> str:
    if not views:
        return ""
    payload = json.dumps(list(views), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ensure_cached_packet_evidence_store(
    *,
    packet: Dict[str, Any],
    paths: Dict[str, Optional[Path]],
    packet_path: Path,
    packet_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Backfill evidence_store/dictionary for legacy packet caches."""
    if not isinstance(packet, dict):
        return packet
    store_path = paths.get("evidence_store")
    if not store_path:
        return packet

    ref = packet.get("evidence_store") if isinstance(packet.get("evidence_store"), dict) else {}
    ref_path = Path(ref.get("path") or store_path)
    has_store = bool(ref) and file_exists(ref_path)
    has_dictionary = isinstance(packet.get("dictionary"), dict) and bool(packet.get("dictionary"))
    if has_store and has_dictionary:
        return packet
    if not paths.get("synthesized"):
        return packet

    cfg = {**DEFAULT_PACKET_CONFIG, **(packet_config or {})}
    syn = load_json(paths["synthesized"])
    trace_index = build_trace_index(syn.get("trace", {}))
    if not has_store:
        evidence_store = build_evidence_store(
            syn,
            trace_index,
            include_raw=bool(cfg.get("include_raw_evidence_store")),
            packet_config=cfg,
        )
        dump_json(evidence_store, store_path)
        packet["evidence_store"] = {
            "path": str(store_path),
            "format": evidence_store.get("format", "evotx_evidence_store_v1"),
            "count": len(evidence_store.get("evidence", {}) or {}),
        }
    if not has_dictionary:
        packet["dictionary"] = build_packet_dictionary(syn, trace_index)
    dump_json(packet, packet_path)
    return packet


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", help="Transaction hash")
    parser.add_argument("--base-dir", default="data/cache", help="Base cache dir")
    parser.add_argument("--force", action="store_true", help="Force rebuild packet")
    parser.add_argument("--profile", choices=["full", "judge", "minimal"], default="full")
    parser.add_argument("--label", help="Attack label used for --profile judge")
    parser.add_argument("--include-view", action="append", dest="include_views")
    parser.add_argument("--no-evidence-store", action="store_true")
    parser.add_argument("--no-view-dependencies", action="store_true")
    parser.add_argument("--include-heavy-dependencies", action="store_true")
    parser.add_argument("--print-view-manifest", action="store_true")
    args = parser.parse_args()

    if args.print_view_manifest:
        print(json.dumps({
            "view_manifest_version": VIEW_MANIFEST_VERSION,
            "views": manifest_summary_rows(),
        }, indent=2, ensure_ascii=False))
        if not args.tx:
            raise SystemExit(0)
    if not args.tx:
        parser.error("--tx is required unless --print-view-manifest is used alone")

    packet = build_or_load_packet(
        tx_hash=args.tx,
        base_dir=args.base_dir,
        force_rebuild=args.force,
        include_views=args.include_views,
        attack_label=args.label,
        packet_profile=args.profile,
        write_evidence_store=not args.no_evidence_store,
        include_view_dependencies=not args.no_view_dependencies,
        include_heavy_optional_dependencies=bool(args.include_heavy_dependencies),
    )
    build_config = dict(packet.get("build_config", {}) or {})
    view_cost_summary = dict(packet.get("view_cost_summary") or build_config.get("view_cost_summary") or {})
    print(json.dumps({
        "transaction_hash": packet["transaction_hash"],
        "packet_path": str(resolve_input_paths(args.tx, args.base_dir)["packet"]),
        "packet_profile": packet.get("packet_profile", args.profile),
        "requested_views": build_config.get("requested_include_views", args.include_views or []),
        "requested_required_views": build_config.get("requested_required_views", []),
        "resolved_views": build_config.get("resolved_include_views", list(packet["views"].keys())),
        "dependency_expanded_views": build_config.get("dependency_expanded_views", []),
        "views": list(packet["views"].keys()),
        "view_tier_counts": view_cost_summary.get("tier_counts", {}),
        "view_cost_score_total": view_cost_summary.get("cost_score_total", 0),
        "heavy_views_present": view_cost_summary.get("heavy_views_present", []),
        "empty_views": view_cost_summary.get("empty_views", []),
        "sources": packet["sources"],
    }, indent=2, ensure_ascii=False))
