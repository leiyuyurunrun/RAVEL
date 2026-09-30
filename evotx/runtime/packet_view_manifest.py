from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


VIEW_MANIFEST_VERSION = "evotx.packet_view_manifest.v1"

VIEW_ORDER = [
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

_COST_SCORE = {
    "cheap": 1,
    "medium": 2,
    "high": 4,
    "very_high": 8,
}


def _entry(
    *,
    tier: str,
    cost: str,
    prompt_role: str,
    default_usage: str,
    is_attack_evidence: bool,
    allow_first_pass: bool,
    allow_followup: bool,
    empty_policy: str,
    dependencies: List[str] | None = None,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "tier": tier,
        "cost": cost,
        "cost_score": _COST_SCORE[cost],
        "prompt_role": prompt_role,
        "default_usage": default_usage,
        "is_attack_evidence": bool(is_attack_evidence),
        "allow_first_pass": bool(allow_first_pass),
        "allow_followup": bool(allow_followup),
        "empty_policy": empty_policy,
        "dependencies": list(dependencies or []),
        "notes": notes,
    }


VIEW_DEPENDENCIES: Dict[str, Dict[str, List[str]]] = {
    "operation_summary_view": {
        "required": [],
        "optional": [
            "critical_call_view",
            "transfer_event_view",
            "external_fundflow_view",
            "amm_reserve_transition_view",
            "flash_or_atomic_capital_view",
            "value_release_view",
        ],
    },
    "classification_digest_view": {
        "required": ["operation_summary_view"],
        "optional": [
            "critical_call_view",
            "unknown_selector_view",
            "value_release_view",
            "participant_net_delta_view",
            "contribution_vs_payout_view",
            "protocol_accounting_outcome_view",
            "price_relevant_state_view",
            "amm_reserve_transition_view",
            "flash_or_atomic_capital_view",
            "semantic_state_delta_view",
            "transfer_event_view",
            "reentrancy_state_order_summary_view",
        ],
    },
    "critical_call_argument_view": {
        "required": ["critical_call_view"],
        "optional": [],
    },
    "source_unavailable_auth_view": {
        "required": [],
        "optional": [
            "critical_call_view",
            "unknown_selector_view",
            "semantic_state_delta_view",
            "state_change_view",
        ],
    },
    "price_relevant_state_view": {
        "required": [],
        "optional": ["semantic_state_delta_view"],
    },
    "market_mechanism_profile_view": {
        "required": [],
        "optional": [
            "price_relevant_state_view",
            "amm_reserve_transition_view",
            "critical_call_view",
            "event_view",
            "token_semantic_delta_summary_view",
            "token_accounting_origin_view",
            "value_release_view",
            "contribution_vs_payout_view",
            "participant_net_delta_view",
            "external_fundflow_view",
            "flash_or_atomic_capital_view",
        ],
    },
    "protocol_accounting_outcome_view": {
        "required": [],
        "optional": [
            "semantic_state_delta_view",
            "critical_call_view",
            "value_release_view",
            "contribution_vs_payout_view",
            "participant_net_delta_view",
        ],
    },
    "token_semantic_delta_summary_view": {
        "required": [],
        "optional": ["semantic_state_delta_view", "transfer_event_view"],
    },
    "token_accounting_origin_view": {
        "required": [],
        "optional": [
            "token_semantic_delta_summary_view",
            "semantic_state_delta_view",
            "transfer_event_view",
            "critical_call_view",
        ],
    },
    "participant_net_delta_view": {
        "required": [],
        "optional": ["transfer_event_view"],
    },
    "contribution_vs_payout_view": {
        "required": [],
        "optional": [
            "participant_net_delta_view",
            "semantic_state_delta_view",
            "critical_call_view",
        ],
    },
    "value_release_view": {
        "required": [],
        "optional": [
            "critical_call_view",
            "transfer_event_view",
            "participant_net_delta_view",
            "semantic_state_delta_view",
        ],
    },
    "flash_or_atomic_capital_view": {
        "required": [],
        "optional": ["critical_call_view", "external_fundflow_view"],
    },
    "beneficiary_controller_view": {
        "required": ["tx_card", "address_labels"],
        "optional": [
            "trace_view",
            "critical_call_view",
            "participant_net_delta_view",
            "contribution_vs_payout_view",
            "external_fundflow_view",
            "profit_loss_view",
            "transfer_event_view",
        ],
    },
    "reentrancy_state_order_summary_view": {
        "required": [],
        "optional": ["critical_call_view"],
    },
    "reentrancy_candidate_catalog_view": {
        "required": [
            "reentrancy_state_order_summary_view",
            "value_release_view",
        ],
        "optional": [],
    },
    "reentrancy_state_order_view": {
        "required": [],
        "optional": ["critical_call_view"],
    },
}


VIEW_MANIFEST: Dict[str, Dict[str, Any]] = {
    "tx_card": _entry(
        tier="meta",
        cost="cheap",
        prompt_role="global_context",
        default_usage="first_pass",
        is_attack_evidence=False,
        allow_first_pass=True,
        allow_followup=False,
        empty_policy="keep",
        notes="Transaction metadata and basic execution stats.",
    ),
    "address_labels": _entry(
        tier="meta",
        cost="cheap",
        prompt_role="global_context",
        default_usage="first_pass",
        is_attack_evidence=False,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="keep_summary_only",
        notes="Actor identity labels and sanitized roles; not standalone attack evidence.",
    ),
    "token_info": _entry(
        tier="meta",
        cost="cheap",
        prompt_role="global_context",
        default_usage="first_pass",
        is_attack_evidence=False,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        notes="Token metadata for interpreting amounts and symbols.",
    ),
    "evidence_adequacy_view": _entry(
        tier="meta",
        cost="cheap",
        prompt_role="routing_signal",
        default_usage="router_only",
        is_attack_evidence=False,
        allow_first_pass=True,
        allow_followup=False,
        empty_policy="keep",
        notes="Completeness and coverage metadata; not attack evidence.",
    ),
    "operation_summary_view": _entry(
        tier="router",
        cost="cheap",
        prompt_role="routing_signal",
        default_usage="first_pass",
        is_attack_evidence=False,
        allow_first_pass=True,
        allow_followup=False,
        empty_policy="keep",
        dependencies=VIEW_DEPENDENCIES["operation_summary_view"]["optional"],
        notes="Signal presence summary for evidence routing.",
    ),
    "classification_digest_view": _entry(
        tier="router",
        cost="cheap",
        prompt_role="routing_signal",
        default_usage="first_pass",
        is_attack_evidence=False,
        allow_first_pass=True,
        allow_followup=False,
        empty_policy="keep",
        dependencies=(
            VIEW_DEPENDENCIES["classification_digest_view"]["required"]
            + VIEW_DEPENDENCIES["classification_digest_view"]["optional"]
        ),
        notes="Compact cross-view digest; does not assign final attack label.",
    ),
    "critical_call_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "critical_call_argument_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=False,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["critical_call_argument_view"]["required"],
        notes="Decoded arguments and return values for calls already selected as critical; neutral parameter evidence, not an attack judgment.",
    ),
    "source_unavailable_auth_view": _entry(
        tier="compact",
        cost="cheap",
        prompt_role="authorization_context",
        default_usage="first_pass",
        is_attack_evidence=False,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["source_unavailable_auth_view"]["optional"],
        notes=(
            "Candidate-indexable authorization context for source-unavailable "
            "or selector-uncertain access-control judging. Rows summarize "
            "call-tree, raw slot, owner/role/proxy hints, and limitations; "
            "they do not prove absence of require/branch checks."
        ),
    ),
    "semantic_state_delta_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "token_semantic_delta_summary_view": _entry(
        tier="compact",
        cost="cheap",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["token_semantic_delta_summary_view"]["optional"],
        notes="Compact token-level semantic anomaly candidates derived from transfer events and semantic state deltas.",
    ),
    "token_accounting_origin_view": _entry(
        tier="compact",
        cost="cheap",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["token_accounting_origin_view"]["optional"],
        notes="Distinguishes token-contract-origin semantic divergence from protocol-internal accounting divergence.",
    ),
    "price_relevant_state_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["price_relevant_state_view"]["optional"],
    ),
    "amm_reserve_transition_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "market_mechanism_profile_view": _entry(
        tier="compact",
        cost="cheap",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["market_mechanism_profile_view"]["optional"],
        notes=(
            "Compact market-mechanism candidate profiles that bind market "
            "source, consumption, outcome evidence ids, and competing-root "
            "warnings without assigning an attack verdict."
        ),
    ),
    "protocol_accounting_outcome_view": _entry(
        tier="compact",
        cost="cheap",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["protocol_accounting_outcome_view"]["optional"],
        notes=(
            "Compact protocol-accounting outcome candidates from state-level "
            "reward/share/debt/vault/liability deltas plus value and payout "
            "context. Neutral until bound to the selected bookkeeping candidate."
        ),
    ),
    "transfer_event_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "participant_net_delta_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["participant_net_delta_view"]["optional"],
    ),
    "value_release_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["value_release_view"]["optional"],
    ),
    "contribution_vs_payout_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["contribution_vs_payout_view"]["optional"],
    ),
    "flash_or_atomic_capital_view": _entry(
        tier="compact",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["flash_or_atomic_capital_view"]["optional"],
    ),
    "trace_outline_view": _entry(
        tier="followup",
        cost="medium",
        prompt_role="detailed_evidence",
        default_usage="first_pass",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "event_view": _entry(
        tier="followup",
        cost="high",
        prompt_role="detailed_evidence",
        default_usage="followup_only",
        is_attack_evidence=True,
        allow_first_pass=False,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "external_fundflow_view": _entry(
        tier="followup",
        cost="high",
        prompt_role="detailed_evidence",
        default_usage="followup_only",
        is_attack_evidence=True,
        allow_first_pass=False,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "unknown_selector_view": _entry(
        tier="followup",
        cost="medium",
        prompt_role="detailed_evidence",
        default_usage="followup_only",
        is_attack_evidence=True,
        allow_first_pass=False,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "beneficiary_controller_view": _entry(
        tier="followup",
        cost="medium",
        prompt_role="detailed_evidence",
        default_usage="followup_only",
        is_attack_evidence=True,
        allow_first_pass=False,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=(
            VIEW_DEPENDENCIES["beneficiary_controller_view"]["required"]
            + VIEW_DEPENDENCIES["beneficiary_controller_view"]["optional"]
        ),
    ),
    "trace_view": _entry(
        tier="raw_heavy",
        cost="very_high",
        prompt_role="raw_trace",
        default_usage="debug_only",
        is_attack_evidence=True,
        allow_first_pass=False,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "state_change_view": _entry(
        tier="raw_heavy",
        cost="very_high",
        prompt_role="raw_state",
        default_usage="followup_only",
        is_attack_evidence=True,
        allow_first_pass=False,
        allow_followup=True,
        empty_policy="drop_if_empty",
    ),
    "reentrancy_state_order_summary_view": _entry(
        tier="conditional",
        cost="medium",
        prompt_role="semantic_evidence",
        default_usage="conditional_only",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["reentrancy_state_order_summary_view"]["optional"],
        notes=(
            "First-pass only when nonempty and reentrancy/state-order related. "
            "Contains candidate-local phase boundaries and compact slot witnesses."
        ),
    ),
    "reentrancy_candidate_catalog_view": _entry(
        tier="compact",
        cost="cheap",
        prompt_role="candidate_catalog",
        default_usage="conditional_only",
        is_attack_evidence=True,
        allow_first_pass=True,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=[
            *VIEW_DEPENDENCIES["reentrancy_candidate_catalog_view"]["required"],
            *VIEW_DEPENDENCIES["reentrancy_candidate_catalog_view"]["optional"],
        ],
        notes=(
            "Ranked stable candidate IDs only; use detailed state-order views "
            "for candidate-local phase evidence."
        ),
    ),
    "reentrancy_state_order_view": _entry(
        tier="conditional",
        cost="high",
        prompt_role="detailed_evidence",
        default_usage="followup_only",
        is_attack_evidence=True,
        allow_first_pass=False,
        allow_followup=True,
        empty_policy="drop_if_empty",
        dependencies=VIEW_DEPENDENCIES["reentrancy_state_order_view"]["optional"],
    ),
    "profit_loss_view": _entry(
        tier="hint",
        cost="medium",
        prompt_role="candidate_hint",
        default_usage="hint_only",
        is_attack_evidence=False,
        allow_first_pass=False,
        allow_followup=False,
        empty_policy="drop_if_empty",
        notes="Candidate hint only; cannot alone prove value extraction.",
    ),
}


def view_tier(view_name: str) -> str:
    return str(VIEW_MANIFEST.get(str(view_name), {}).get("tier", "unknown"))


def view_cost(view_name: str) -> str:
    return str(VIEW_MANIFEST.get(str(view_name), {}).get("cost", "medium"))


def view_cost_score(view_name: str) -> int:
    try:
        return int(VIEW_MANIFEST.get(str(view_name), {}).get("cost_score", 2) or 2)
    except Exception:
        return 2


def is_heavy_view(view_name: str) -> bool:
    return view_tier(view_name) == "raw_heavy" or view_cost(view_name) in {"high", "very_high"}


def is_prompt_first_pass_allowed(view_name: str) -> bool:
    return bool(VIEW_MANIFEST.get(str(view_name), {}).get("allow_first_pass", False))


def is_hint_only_view(view_name: str) -> bool:
    meta = VIEW_MANIFEST.get(str(view_name), {})
    return view_tier(view_name) == "hint" or meta.get("default_usage") == "hint_only"


def is_attack_evidence_view(view_name: str) -> bool:
    return bool(VIEW_MANIFEST.get(str(view_name), {}).get("is_attack_evidence", False))


def resolve_view_dependency_closure(
    requested_views: Iterable[str],
    *,
    include_optional: bool = True,
    include_heavy_optional: bool = False,
) -> List[str]:
    requested = _known_ordered(requested_views)
    wanted = set(requested)
    visiting: set[str] = set()

    def visit(view: str) -> None:
        if view in visiting:
            return
        visiting.add(view)
        deps = VIEW_DEPENDENCIES.get(view, {})
        candidates = list(deps.get("required", []) or [])
        if include_optional:
            candidates.extend(list(deps.get("optional", []) or []))
        for dep in candidates:
            if dep not in VIEW_MANIFEST:
                continue
            if dep in (deps.get("optional", []) or []) and is_heavy_view(dep) and not include_heavy_optional:
                continue
            wanted.add(dep)
            visit(dep)
        visiting.discard(view)

    for view in requested:
        visit(view)
    return [view for view in VIEW_ORDER if view in wanted]


def filter_views_for_prompt_policy(
    view_names: Iterable[str],
    *,
    usage: str,
    attack_label: str = "",
    allow_hint: bool = False,
    allow_meta: bool = True,
    allow_heavy: bool = False,
) -> List[str]:
    usage = str(usage or "").strip().lower()
    label_text = str(attack_label or "").lower()
    out: List[str] = []
    for view in _known_ordered(view_names):
        meta = VIEW_MANIFEST.get(view, {})
        tier = str(meta.get("tier", ""))
        if usage == "debug":
            out.append(view)
            continue
        if tier == "meta" and not allow_meta:
            continue
        if is_hint_only_view(view) and not allow_hint:
            continue
        if tier == "raw_heavy" and not allow_heavy:
            continue
        if usage == "first_pass":
            if tier == "raw_heavy" or is_hint_only_view(view):
                continue
            if tier == "conditional" and "reentr" not in label_text and "state_order" not in label_text:
                continue
            if not bool(meta.get("allow_first_pass", False)):
                continue
        elif usage == "followup":
            if tier == "raw_heavy" and not allow_heavy:
                continue
            if not bool(meta.get("allow_followup", False)):
                continue
        out.append(view)
    return out


def summarize_packet_view_costs(
    views: Dict[str, Any],
    *,
    manifest: Dict[str, Any] = VIEW_MANIFEST,
) -> Dict[str, Any]:
    tier_counts: Dict[str, int] = {}
    cost_counts: Dict[str, int] = {}
    chars_by_view: Dict[str, int] = {}
    heavy_views: List[str] = []
    hint_views: List[str] = []
    empty_views: List[str] = []
    score_total = 0
    view_names = [name for name in VIEW_ORDER if isinstance(views, dict) and name in views]
    for name in view_names:
        entry = manifest.get(name, {})
        tier = str(entry.get("tier", "unknown"))
        cost = str(entry.get("cost", "medium"))
        tier_counts[tier] = int(tier_counts.get(tier, 0) or 0) + 1
        cost_counts[cost] = int(cost_counts.get(cost, 0) or 0) + 1
        score = int(entry.get("cost_score", _COST_SCORE.get(cost, 2)) or 0)
        score_total += score
        data = views.get(name)
        chars_by_view[name] = len(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str))
        if tier == "raw_heavy" or cost in {"high", "very_high"}:
            heavy_views.append(name)
        if tier == "hint":
            hint_views.append(name)
        if _view_is_empty(data):
            empty_views.append(name)
    return {
        "view_count": len(view_names),
        "tier_counts": tier_counts,
        "cost_counts": cost_counts,
        "cost_score_total": score_total,
        "estimated_prompt_chars_by_view": chars_by_view,
        "heavy_views_present": heavy_views,
        "hint_only_views_present": hint_views,
        "empty_views": empty_views,
    }


def manifest_summary_rows() -> List[Dict[str, Any]]:
    return [
        {
            "view": view,
            "tier": view_tier(view),
            "cost": view_cost(view),
            "cost_score": view_cost_score(view),
            "default_usage": VIEW_MANIFEST[view]["default_usage"],
            "prompt_role": VIEW_MANIFEST[view]["prompt_role"],
            "allow_first_pass": VIEW_MANIFEST[view]["allow_first_pass"],
            "allow_followup": VIEW_MANIFEST[view]["allow_followup"],
        }
        for view in VIEW_ORDER
    ]


def _known_ordered(values: Iterable[str]) -> List[str]:
    wanted = {str(value).strip() for value in values or [] if str(value).strip() in VIEW_MANIFEST}
    return [view for view in VIEW_ORDER if view in wanted]


def _view_is_empty(data: Any) -> bool:
    if data is None:
        return True
    if isinstance(data, list):
        return len(data) == 0
    if isinstance(data, dict):
        for key in (
            "records",
            "rows",
            "events",
            "items",
            "deltas",
            "transfers",
            "release_records",
            "releases",
            "auth_contexts",
            "profiles",
            "unknown_selectors",
            "beneficiary_paths",
            "controller_hints",
            "pairs",
            "entries",
            "operations",
        ):
            value = data.get(key)
            if isinstance(value, list):
                return len(value) == 0
        return not bool(data)
    return False
