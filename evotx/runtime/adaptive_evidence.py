from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from evotx.core.labels import label_text
from evotx.core.schemas import EvidencePlan, JudgeStep
from evotx.runtime.packet_builder import (
    build_trace_index,
    load_json,
    resolve_input_paths,
    trace_counts,
)
from evotx.runtime.packet_view_manifest import (
    is_heavy_view,
    is_hint_only_view,
    view_tier,
)


ADAPTIVE_EVIDENCE_NOTE = (
    "The selected evidence views were chosen by an adaptive evidence policy "
    "based on trace size, view cost, and the local judge question. Absence of "
    "an omitted view should not be treated as absence of evidence; request "
    "follow-up only if needed."
)

ADAPTIVE_BASE_VIEWS = [
    "tx_card",
    "evidence_adequacy_view",
    "operation_summary_view",
    "classification_digest_view",
    "trace_outline_view",
    "critical_call_view",
]

ADAPTIVE_COMPACT_VIEWS = [
    "critical_call_argument_view",
    "semantic_state_delta_view",
    "transfer_event_view",
    "participant_net_delta_view",
    "value_release_view",
]


@dataclass
class AdaptiveEvidenceConfig:
    max_direct_trace_nodes: int = 80
    max_direct_trace_chars: int = 45000
    max_medium_trace_nodes: int = 250
    max_medium_trace_outline_chars: int = 25000
    first_pass_target_ratio: float = 0.75
    first_pass_hard_cap_ratio: float = 0.90


@dataclass
class EvidenceProfile:
    tx_hash: str
    trace_total_nodes: int
    trace_shown_nodes: int
    trace_truncated: bool
    call_nodes: int
    event_nodes: int
    state_change_nodes: int
    semantic_state_rows: int
    view_chars: Dict[str, int]
    empty_views: List[str]
    heavy_views_present: List[str]
    max_context_chars: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveEvidenceDecision:
    enabled: bool
    mode: str
    reason: str
    initial_refs_before: List[str]
    initial_refs_after: List[str]
    allowed_followup_before: List[str]
    allowed_followup_after: List[str]
    added_views: List[str]
    removed_views: List[str]
    deferred_views: List[str]
    estimated_chars: int
    budget: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_evidence_profile(packet: Dict[str, Any], max_context_chars: int) -> EvidenceProfile:
    views = packet.get("views", {}) if isinstance(packet, dict) else {}
    adequacy = views.get("evidence_adequacy_view", {}) if isinstance(views, dict) else {}
    trace_info = adequacy.get("trace", {}) if isinstance(adequacy, dict) else {}
    decode = adequacy.get("decode_coverage", {}) if isinstance(adequacy, dict) else {}
    state = adequacy.get("state_coverage", {}) if isinstance(adequacy, dict) else {}
    cost_summary = packet.get("view_cost_summary", {}) if isinstance(packet, dict) else {}
    if not isinstance(cost_summary, dict):
        cost_summary = {}
    chars = cost_summary.get("estimated_prompt_chars_by_view", {})
    if not isinstance(chars, dict):
        chars = {}
    return EvidenceProfile(
        tx_hash=str(packet.get("transaction_hash") or "") if isinstance(packet, dict) else "",
        trace_total_nodes=_int(trace_info.get("total_nodes")),
        trace_shown_nodes=_int(trace_info.get("shown_nodes_in_trace_view")),
        trace_truncated=bool(trace_info.get("truncated", False)),
        call_nodes=_int(decode.get("call_nodes")),
        event_nodes=_int(state.get("event_nodes") or decode.get("event_nodes")),
        state_change_nodes=_int(state.get("state_change_nodes")),
        semantic_state_rows=_int(state.get("semantic_state_rows")),
        view_chars={str(k): _int(v) for k, v in chars.items()},
        empty_views=_string_list(cost_summary.get("empty_views", [])),
        heavy_views_present=_string_list(cost_summary.get("heavy_views_present", [])),
        max_context_chars=max(0, int(max_context_chars or 0)),
    )


def choose_evidence_mode(
    profile: EvidenceProfile,
    *,
    max_context_chars: int,
    config: AdaptiveEvidenceConfig | None = None,
) -> Tuple[str, str]:
    cfg = config or AdaptiveEvidenceConfig()
    trace_chars = int(profile.view_chars.get("trace_view", 0) or 0)
    outline_chars = int(profile.view_chars.get("trace_outline_view", 0) or 0)
    direct_char_cap = min(cfg.max_direct_trace_chars, int(0.70 * max_context_chars))
    medium_outline_cap = min(cfg.max_medium_trace_outline_chars, int(0.45 * max_context_chars))

    if (
        not profile.trace_truncated
        and profile.trace_total_nodes <= cfg.max_direct_trace_nodes
        and trace_chars > 0
        and trace_chars <= direct_char_cap
    ):
        return (
            "small_trace_direct",
            "trace is small and non-truncated; trace_view fits direct first-pass budget",
        )
    if (
        not profile.trace_truncated
        and profile.trace_total_nodes <= cfg.max_medium_trace_nodes
        and outline_chars > 0
        and outline_chars <= medium_outline_cap
    ):
        return (
            "medium_hybrid",
            "trace is medium-sized and non-truncated; trace_outline_view fits hybrid budget",
        )
    return (
        "large_planned_views",
        "trace is large, truncated, or direct trace/outline budget would be exceeded",
    )


def adapt_step_evidence_refs(
    *,
    packet: Dict[str, Any],
    step: JudgeStep,
    attack_label: str,
    initial_refs: List[str],
    allowed_followup_views: List[str],
    profile: EvidenceProfile,
    mode: str,
    max_context_chars: int,
    max_view_chars: int,
    config: AdaptiveEvidenceConfig | None = None,
) -> AdaptiveEvidenceDecision:
    cfg = config or AdaptiveEvidenceConfig()
    normalized_mode = normalize_adaptive_mode(mode)
    before_initial = _dedupe(initial_refs)
    before_followup = _dedupe(allowed_followup_views)
    question = str(step.question or "")
    label = label_text(attack_label, default="")

    if normalized_mode in {"off", "planned", "planned_views", "large_planned_views"}:
        after_initial = _safe_first_pass_refs(
            before_initial,
            profile=profile,
            mode="planned_views" if normalized_mode in {"off", "planned"} else "large_planned_views",
            explicit_refs=set(before_initial),
        )
        after_followup = _safe_followup_refs(before_followup)
        decision_mode = "planned_views" if normalized_mode in {"off", "planned"} else "large_planned_views"
        reason = (
            "adaptive policy preserved planned first-pass views with conservative "
            "empty/hint/raw-heavy filtering"
        )
    elif normalized_mode == "small_trace_direct":
        refs = [
            "tx_card",
            "operation_summary_view",
            "classification_digest_view",
            "trace_view",
        ]
        refs.extend(_label_question_views(label, question, mode="small_trace_direct", profile=profile))
        after_initial = _safe_first_pass_refs(
            refs,
            profile=profile,
            mode="small_trace_direct",
            keep_refs={"tx_card", "operation_summary_view", "classification_digest_view", "trace_view"},
        )
        after_followup = _safe_followup_refs(
            [
                view
                for view in before_followup
                if view not in {"profit_loss_view", "trace_view"}
            ]
            + [
                "read_evidence_by_id",
                "read_evidence_context",
                "state_change_view",
            ],
            include_tools=False,
        )
        reason = "small trace direct mode selected compact context plus full trace_view"
    elif normalized_mode == "medium_hybrid":
        refs = [
            "tx_card",
            "operation_summary_view",
            "classification_digest_view",
            "trace_outline_view",
            "critical_call_view",
        ]
        refs.extend(_label_question_views(label, question, mode="medium_hybrid", profile=profile))
        after_initial = _safe_first_pass_refs(
            refs,
            profile=profile,
            mode="medium_hybrid",
            keep_refs={
                "tx_card",
                "operation_summary_view",
                "classification_digest_view",
                "trace_outline_view",
                "critical_call_view",
            },
        )
        after_followup = _safe_followup_refs(
            before_followup
            + [
                "trace_view",
                "state_change_view",
                "event_view",
                "external_fundflow_view",
            ]
        )
        reason = "medium hybrid mode selected trace outline plus compact semantic evidence"
    else:
        after_initial = _safe_first_pass_refs(before_initial, profile=profile, mode="planned_views")
        after_followup = _safe_followup_refs(before_followup)
        reason = f"unknown adaptive mode {mode!r}; preserved planned views"

    after_initial, deferred = _fit_budget(
        after_initial,
        profile=profile,
        mode=normalized_mode,
        max_context_chars=max_context_chars,
        max_view_chars=max_view_chars,
        cfg=cfg,
    )
    after_followup = _dedupe(
        [
            view
            for view in after_followup + deferred
            if view not in after_initial and not is_hint_only_view(view)
        ]
    )
    added = [view for view in after_initial if view not in before_initial]
    removed = [view for view in before_initial if view not in after_initial]
    estimated = _estimated_chars(after_initial, profile)
    return AdaptiveEvidenceDecision(
        enabled=True,
        mode="small_trace_direct" if normalized_mode == "direct_trace" else normalized_mode,
        reason=reason,
        initial_refs_before=before_initial,
        initial_refs_after=after_initial,
        allowed_followup_before=before_followup,
        allowed_followup_after=after_followup,
        added_views=added,
        removed_views=removed,
        deferred_views=deferred,
        estimated_chars=estimated,
        budget={
            "target_chars": int(cfg.first_pass_target_ratio * max_context_chars),
            "hard_cap_chars": int(cfg.first_pass_hard_cap_ratio * max_context_chars),
            "max_view_chars": int(max_view_chars or 0),
            "max_context_chars": int(max_context_chars or 0),
        },
    )


def collect_adaptive_candidate_views(
    *,
    attack_label: str = "",
    adaptive_evidence: bool = False,
    adaptive_evidence_mode: str = "off",
    tx_hash: str = "",
    base_dir: str | Path = "data/cache",
    config: AdaptiveEvidenceConfig | None = None,
) -> List[str]:
    if not adaptive_evidence or normalize_adaptive_mode(adaptive_evidence_mode) in {"off", "planned"}:
        return []
    cfg = config or AdaptiveEvidenceConfig()
    label = label_text(attack_label, default="")
    mode = normalize_adaptive_mode(adaptive_evidence_mode)
    views = list(ADAPTIVE_BASE_VIEWS) + list(ADAPTIVE_COMPACT_VIEWS)
    if _label_has(label, "access control"):
        views.extend([
            "address_labels",
            "critical_call_argument_view",
            "beneficiary_controller_view",
        ])
    if _label_has(label, "price manipulation", "market manipulation"):
        views.extend(["price_relevant_state_view", "amm_reserve_transition_view"])
        if _label_has(label, "market manipulation"):
            views.append("market_mechanism_profile_view")
    if _label_has(label, "reentrancy"):
        views.extend([
            "reentrancy_candidate_catalog_view",
            "reentrancy_state_order_summary_view",
        ])
    if _label_has(label, "protocol accounting"):
        views.append("contribution_vs_payout_view")
    if _label_has(label, "token semantic"):
        views.extend(["transfer_event_view", "semantic_state_delta_view"])
    if _label_has(label, "flashloans", "flashloan"):
        views.append("flash_or_atomic_capital_view")
    if _label_has(label, "insufficient validation"):
        views.extend(["critical_call_argument_view", "event_view"])

    if mode in {"small_trace_direct", "direct_trace"}:
        views.append("trace_view")
    elif mode == "auto":
        trace_profile = inspect_trace_size_for_tx(tx_hash, base_dir=base_dir)
        if 0 < int(trace_profile.get("trace_total_nodes", 0) or 0) <= cfg.max_direct_trace_nodes:
            views.append("trace_view")
    return _dedupe(views)


def inspect_trace_size_for_tx(tx_hash: str, base_dir: str | Path = "data/cache") -> Dict[str, Any]:
    try:
        paths = resolve_input_paths(tx_hash, base_dir=base_dir)
        synthesized = paths.get("synthesized")
        if not synthesized:
            return {}
        syn = load_json(synthesized)
        index = build_trace_index(syn.get("trace", {}))
        counts = trace_counts(index)
        return {
            "trace_total_nodes": int(counts.get("total_nodes", 0) or 0),
            "call_nodes": int(counts.get("call_nodes", 0) or 0),
            "event_nodes": int(counts.get("event_nodes", 0) or 0),
            "state_change_nodes": int(counts.get("state_change_nodes", 0) or 0),
            "synthesized_path": str(synthesized),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def resolve_adaptive_mode(
    *,
    requested_mode: str,
    profile: EvidenceProfile,
    max_context_chars: int,
    config: AdaptiveEvidenceConfig | None = None,
) -> Tuple[str, str]:
    mode = normalize_adaptive_mode(requested_mode)
    if mode == "off":
        return "planned_views", "adaptive evidence disabled"
    if mode in {"planned", "planned_views"}:
        return "planned_views", "planned mode requested"
    if mode == "direct_trace":
        return "small_trace_direct", "direct_trace alias requested"
    if mode in {"small_trace_direct", "medium_hybrid", "large_planned_views"}:
        return mode, f"{mode} requested"
    return choose_evidence_mode(profile, max_context_chars=max_context_chars, config=config)


def normalize_adaptive_mode(mode: str) -> str:
    normalized = str(mode or "auto").strip().lower()
    if normalized in {"", "on"}:
        return "auto"
    if normalized == "direct_trace":
        return "small_trace_direct"
    if normalized == "planned":
        return "planned_views"
    return normalized


def _label_question_views(
    label: str,
    question: str,
    *,
    mode: str,
    profile: EvidenceProfile,
) -> List[str]:
    text = _normalize_text(f"{label} {question}")
    views: List[str] = []
    if _label_has(label, "access control"):
        views.append("address_labels")
        if _contains_any(text, ["caller", "role", "admin", "owner", "authorized", "privilege", "beneficiary"]):
            views.append("critical_call_argument_view")
            views.append("beneficiary_controller_view")
        if _contains_any(text, ["value", "release", "withdraw", "transfer", "protected asset"]):
            views.append("value_release_view")
        if _contains_any(text, ["state", "approval", "privilege", "config"]):
            views.append("semantic_state_delta_view")
    elif _label_has(label, "insufficient validation"):
        views.append("critical_call_view")
        views.append("critical_call_argument_view")
        if _contains_any(text, ["callback", "return", "event", "input"]):
            views.append("event_view")
        views.append("semantic_state_delta_view")
        if _contains_any(text, ["outcome", "value", "payout", "asset", "release", "loss"]):
            views.append("value_release_view")
    elif _label_has(label, "price manipulation", "market manipulation"):
        if _label_has(label, "market manipulation"):
            views.append("market_mechanism_profile_view")
        views.append("price_relevant_state_view")
        if "amm_reserve_transition_view" not in set(profile.empty_views):
            views.append("amm_reserve_transition_view")
        if _contains_any(text, ["outcome", "value", "payout", "profit", "transfer"]):
            views.extend(["transfer_event_view", "participant_net_delta_view"])
    elif _label_has(label, "reentrancy"):
        if "reentrancy_state_order_summary_view" not in set(profile.empty_views):
            views.append("reentrancy_state_order_summary_view")
        views.append("critical_call_view")
        if _contains_any(text, ["outcome", "value", "payout", "release"]):
            views.append("value_release_view")
    elif _label_has(label, "protocol accounting"):
        views.append("semantic_state_delta_view")
        views.append("contribution_vs_payout_view")
        if _contains_any(text, ["outcome", "value", "payout", "loss", "release"]):
            views.extend(["participant_net_delta_view", "value_release_view"])
    elif _label_has(label, "token semantic"):
        views.extend(["transfer_event_view", "semantic_state_delta_view", "participant_net_delta_view"])
        if _contains_any(text, ["hook", "mint", "burn", "balanceof", "balance of"]):
            views.append("critical_call_view")
    elif _label_has(label, "flashloans", "flashloan"):
        if "flash_or_atomic_capital_view" not in set(profile.empty_views):
            views.append("flash_or_atomic_capital_view")
        views.append("critical_call_view")
    else:
        views.append("critical_call_view")
    if mode == "medium_hybrid" and "trace_view" in views:
        views.remove("trace_view")
    return _dedupe(views)


def _safe_first_pass_refs(
    refs: Iterable[str],
    *,
    profile: EvidenceProfile,
    mode: str,
    keep_refs: set[str] | None = None,
    explicit_refs: set[str] | None = None,
) -> List[str]:
    keep_refs = set(keep_refs or set())
    explicit_refs = set(explicit_refs or set())
    out: List[str] = []
    for view in _dedupe(refs):
        if view in {"read_evidence_by_id", "read_evidence_context"}:
            continue
        if is_hint_only_view(view):
            continue
        if view in profile.empty_views and view not in {"evidence_adequacy_view"}:
            continue
        if view == "reentrancy_state_order_summary_view" and view in profile.empty_views:
            continue
        if view_tier(view) == "raw_heavy":
            if view == "trace_view" and mode == "small_trace_direct":
                pass
            elif view in explicit_refs and _within_single_view_budget(view, profile):
                pass
            elif view in keep_refs:
                pass
            else:
                continue
        if view not in out:
            out.append(view)
    return out


def _safe_followup_refs(refs: Iterable[str], *, include_tools: bool = False) -> List[str]:
    out = []
    for view in _dedupe(refs):
        if not include_tools and view in {"read_evidence_by_id", "read_evidence_context"}:
            continue
        if is_hint_only_view(view):
            continue
        if view not in out:
            out.append(view)
    return out


def _fit_budget(
    refs: List[str],
    *,
    profile: EvidenceProfile,
    mode: str,
    max_context_chars: int,
    max_view_chars: int,
    cfg: AdaptiveEvidenceConfig,
) -> Tuple[List[str], List[str]]:
    target = int(cfg.first_pass_target_ratio * max_context_chars)
    hard_cap = int(cfg.first_pass_hard_cap_ratio * max_context_chars)
    cap = max(0, min(hard_cap, max_context_chars))
    protected = {
        "tx_card",
        "operation_summary_view",
        "classification_digest_view",
    }
    if mode in {"small_trace_direct", "direct_trace"}:
        protected.add("trace_view")
    if mode == "medium_hybrid":
        protected.update({"trace_outline_view", "critical_call_view"})
    removal_order = [
        "profit_loss_view",
        *profile.empty_views,
        "state_change_view",
        "event_view",
        "external_fundflow_view",
        "beneficiary_controller_view",
        "value_release_view",
        "contribution_vs_payout_view",
        "participant_net_delta_view",
        "semantic_state_delta_view",
        "transfer_event_view",
        "critical_call_view",
        "trace_outline_view",
    ]
    current = _dedupe(refs)
    deferred: List[str] = []
    while current and _estimated_chars(current, profile) > cap:
        removed = False
        for candidate in removal_order:
            if candidate in protected or candidate not in current:
                continue
            current.remove(candidate)
            deferred.append(candidate)
            removed = True
            break
        if not removed:
            break
    if _estimated_chars(current, profile) > target:
        for candidate in list(removal_order):
            if candidate in protected or candidate not in current:
                continue
            current.remove(candidate)
            deferred.append(candidate)
            if _estimated_chars(current, profile) <= target:
                break
    return current, _dedupe(deferred)


def _estimated_chars(refs: Iterable[str], profile: EvidenceProfile) -> int:
    return sum(int(profile.view_chars.get(view, 0) or 0) for view in refs)


def _within_single_view_budget(view: str, profile: EvidenceProfile) -> bool:
    return int(profile.view_chars.get(view, 0) or 0) <= min(
        45000,
        int(0.70 * max(1, profile.max_context_chars)),
    )


def _dedupe(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return _dedupe(str(value) for value in values)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_text(value: Any) -> str:
    return str(value or "").lower().replace("_", " ").replace("-", " ")


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _label_has(label: str, *needles: str) -> bool:
    return any(needle in label for needle in needles)
