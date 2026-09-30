from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evotx.runtime.evidence_renderer import (
    PROMPT_DICTIONARY_EXTRA_VALUES_KEY,
    PROMPT_DICTIONARY_KEY,
    alias_prompt_value,
    format_evidence_context_with_metadata,
)
from evotx.utils.json_utils import stable_json_dumps


FOLLOWUP_CONTEXT_MODES = {"legacy", "unified"}
FOLLOWUP_CONTEXT_VERSION = "evotx.followup_context.v1"
DEFAULT_FIXED_PROMPT_RESERVE_CHARS = 24000

_FOLLOWUP_ROUTER_VIEWS = {
    "tx_card",
    "evidence_adequacy_view",
    "operation_summary_view",
    "classification_digest_view",
}
_FOLLOWUP_ROW_KEYS = (
    "rows",
    "records",
    "events",
    "transfers",
    "deltas",
    "release_records",
    "releases",
    "profiles",
    "unknown_selectors",
    "beneficiary_paths",
    "controller_hints",
    "pairs",
    "entries",
    "operations",
    "items",
)

_STATE_ORDER_ROW_FIELDS = (
    "evidence_id",
    "source_call_evidence_id",
    "call_id",
    "function",
    "callee",
    "address_label",
    "parent_id",
    "parent_function",
    "depth",
    "why_included",
    "first_entry_id",
    "reentrant_depth_gap",
    "reentrant_chain_ids",
    "state_access_count",
    "nearby_event_ids",
    "nearby_state_ids",
)
_STATE_ACCESS_FIELDS = (
    "evidence_id",
    "type",
    "order_index",
    "parent_id",
    "parent_function",
    "call_id",
)
_SEMANTIC_STATE_ROW_FIELDS = (
    "evidence_id",
    "source_evidence_id",
    "call_id",
    "parent_id",
    "parent_function",
    "contract",
    "contract_label",
    "address_label",
    "slot_key",
    "variable_hint",
    "variable",
    "prev",
    "current",
    "delta",
    "semantic_confidence",
    "reason",
)
_VALUE_RELEASE_ROW_FIELDS = (
    "evidence_id",
    "sensitive_call_evidence_id",
    "call_id",
    "function",
    "parent_id",
    "parent_function",
    "depth",
    "path_ids",
    "target_contract",
    "target_contract_label",
    "release_record_count",
    "repeated_release_count",
    "value_out_summary",
    "target_balance_change_summary",
    "recipient_balance_change_summary",
    "recipients",
    "related_evidence_ids",
)
_LOCAL_CONTEXT_ROW_FIELDS = (
    "evidence_id",
    "id",
    "type",
    "call_type",
    "parent_id",
    "depth",
    "parent_function",
    "parent_address",
    "path_ids",
    "caller",
    "caller_label",
    "callee",
    "address",
    "address_label",
    "function",
    "value",
    "slot_key",
    "value_read",
    "prev",
    "current",
    "nearby_event_ids",
    "nearby_state_ids",
    "why_included",
    "reentrant_depth_gap",
    "first_entry_id",
    "reentrant_chain_ids",
)


def normalize_followup_context_mode(value: Any) -> str:
    mode = str(value or "unified").strip().lower()
    return mode if mode in FOLLOWUP_CONTEXT_MODES else "unified"


@dataclass(frozen=True)
class FollowupPromptContext:
    evidence_context: str
    evidence_render_metadata: List[Dict[str, Any]]
    previous_judge_summary: Dict[str, Any]
    previous_judge_summary_text: str
    tool_observations_text: str
    state_input_context: Dict[str, Any]
    metadata: Dict[str, Any]


class FollowupContextBuilder:
    """Build one budgeted context shared by initial and follow-up judge calls."""

    def __init__(
        self,
        *,
        max_view_chars: int,
        max_context_chars: int,
        fixed_prompt_reserve_chars: int = DEFAULT_FIXED_PROMPT_RESERVE_CHARS,
    ):
        self.max_view_chars = max(1, int(max_view_chars or 1))
        self.max_context_chars = max(512, int(max_context_chars or 512))
        self.fixed_prompt_reserve_chars = max(
            0,
            int(fixed_prompt_reserve_chars or 0),
        )

    def build(
        self,
        *,
        evidence_packet: Dict[str, Any],
        packet_evidence_adequacy: Optional[Dict[str, Any]] = None,
        previous_judge_result: Optional[Dict[str, Any]] = None,
        earlier_judge_results: Optional[List[Dict[str, Any]]] = None,
        tool_observations: Optional[List[Dict[str, Any]]] = None,
        state_input_context: Optional[Dict[str, Any]] = None,
    ) -> FollowupPromptContext:
        prompt_budget = self.max_context_chars
        reserve = _fixed_prompt_reserve(
            prompt_budget,
            requested=self.fixed_prompt_reserve_chars,
        )
        total_budget = max(256, prompt_budget - reserve)
        observations = list(tool_observations or [])
        state_value = dict(state_input_context or {})
        previous_summary = _previous_judge_summary(
            previous_judge_result,
            earlier_results=list(earlier_judge_results or []),
        )
        replay_packet, replay_metadata = _followup_replay_packet(
            dict(evidence_packet or {}),
            previous_summary=previous_summary,
            has_observations=bool(observations),
        )

        raw_caps = {
            "state": _section_cap(
                total_budget,
                ratio=0.12,
                minimum=128,
                maximum=12000,
            ),
            "previous": _section_cap(
                total_budget,
                ratio=0.16,
                minimum=128,
                maximum=20000,
            ),
            "observations": _section_cap(
                total_budget,
                ratio=0.35,
                minimum=256,
                maximum=60000,
            ),
        }
        state_cap, previous_cap, observations_cap = _allocate_non_evidence_caps(
            total_budget=total_budget,
            raw_caps=raw_caps,
            has_state=bool(state_value),
            has_previous=bool(previous_summary),
            has_observations=bool(observations),
        )

        compact_state, state_text, state_truncated = _fit_json(
            state_value,
            state_cap if state_value else 0,
        )
        compact_previous, previous_text, previous_truncated = _fit_previous_summary(
            previous_summary,
            previous_cap if previous_summary else 0,
        )
        prompt_dictionary = (
            dict(evidence_packet.get(PROMPT_DICTIONARY_KEY, {}) or {})
            if isinstance(evidence_packet.get(PROMPT_DICTIONARY_KEY), dict)
            else {}
        )
        prompt_observations = (
            alias_prompt_value(observations, prompt_dictionary)
            if prompt_dictionary
            else observations
        )
        if prompt_dictionary and observations:
            replay_packet = dict(replay_packet)
            replay_packet[PROMPT_DICTIONARY_EXTRA_VALUES_KEY] = observations
        observations_text, observations_meta = _format_observations(
            prompt_observations,
            observations_cap if observations else 0,
        )

        non_evidence_chars = len(state_text) + len(previous_text) + len(observations_text)
        evidence_budget = max(1, total_budget - non_evidence_chars)
        evidence_context, render_metadata = format_evidence_context_with_metadata(
            replay_packet,
            max_view_chars=min(self.max_view_chars, evidence_budget),
            max_context_chars=evidence_budget,
            packet_evidence_adequacy=dict(packet_evidence_adequacy or {}),
        )
        evidence_truncated_after_render = False
        if len(evidence_context) > evidence_budget:
            evidence_context = _truncate_string(
                evidence_context,
                evidence_budget,
                marker="...<unified evidence budget truncated>",
            )
            evidence_truncated_after_render = True

        used = {
            "evidence": len(evidence_context),
            "previous_judge_summary": len(previous_text),
            "tool_observations": len(observations_text),
            "state_input_context": len(state_text),
        }
        used["content_total"] = sum(used.values())
        used["fixed_prompt_reserve"] = reserve
        used["total"] = used["content_total"] + reserve
        metadata = {
            "version": FOLLOWUP_CONTEXT_VERSION,
            "mode": "unified",
            "total_budget_chars": prompt_budget,
            "content_budget_chars": total_budget,
            "fixed_prompt_reserve_chars": reserve,
            "allocated_caps": {
                "state_input_context": state_cap if state_value else 0,
                "previous_judge_summary": previous_cap if previous_summary else 0,
                "tool_observations": observations_cap if observations else 0,
                "evidence": evidence_budget,
            },
            "used_chars": used,
            "remaining_chars": max(0, prompt_budget - used["total"]),
            "within_budget": used["total"] <= prompt_budget,
            "previous_judge_summary_present": bool(previous_summary),
            "earlier_judge_summary_count": len(
                list(previous_summary.get("earlier_round_summaries", []) or [])
            ),
            "raw_previous_completion_included": False,
            "previous_judge_summary_truncated": bool(previous_truncated),
            "state_input_present": bool(state_value),
            "state_input_truncated": bool(state_truncated),
            "observation_count": len(observations),
            "observations": observations_meta,
            "evidence_render_truncated_after_unified_budget": bool(
                evidence_truncated_after_render
            ),
            "evidence_replay": replay_metadata,
        }
        return FollowupPromptContext(
            evidence_context=evidence_context,
            evidence_render_metadata=list(render_metadata or []),
            previous_judge_summary=dict(compact_previous or {}),
            previous_judge_summary_text=previous_text,
            tool_observations_text=observations_text,
            state_input_context=dict(compact_state or {}),
            metadata=metadata,
        )


def _fixed_prompt_reserve(total_budget: int, *, requested: int) -> int:
    if total_budget <= 768:
        return max(0, total_budget - 512)
    if total_budget < 40000:
        return min(max(0, total_budget - 512), max(512, int(total_budget * 0.20)))
    return min(
        max(0, total_budget - 512),
        min(max(20000, requested), 25000),
    )


def _followup_replay_packet(
    evidence_packet: Dict[str, Any],
    *,
    previous_summary: Dict[str, Any],
    has_observations: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not previous_summary and not has_observations:
        return evidence_packet, {
            "mode": "initial_full_selection",
            "replayed_views": list(evidence_packet.keys()),
            "dropped_views": [],
            "referenced_evidence_ids": [],
        }

    evidence_ids = _summary_evidence_ids(previous_summary)
    replay: Dict[str, Any] = {}
    dropped: List[str] = []
    for view_name, view_data in evidence_packet.items():
        if view_name in {
            PROMPT_DICTIONARY_KEY,
            PROMPT_DICTIONARY_EXTRA_VALUES_KEY,
        }:
            replay[view_name] = view_data
            continue
        if view_name in _FOLLOWUP_ROUTER_VIEWS:
            replay[view_name] = view_data
            continue
        projected = _project_view_to_evidence_ids(view_data, evidence_ids)
        if projected is not None:
            replay[view_name] = projected
        else:
            dropped.append(view_name)
    return replay, {
        "mode": "router_and_referenced_evidence",
        "replayed_views": list(replay.keys()),
        "dropped_views": dropped,
        "referenced_evidence_ids": evidence_ids,
    }


def _summary_evidence_ids(summary: Dict[str, Any]) -> List[str]:
    values: List[Any] = []
    for key in ("supporting_evidence_ids", "contradicting_evidence_ids"):
        values.extend(list(summary.get(key, []) or []))
    for request in list(summary.get("tool_requests", []) or []):
        if not isinstance(request, dict):
            continue
        args = request.get("args") if isinstance(request.get("args"), dict) else request
        values.append(args.get("evidence_id"))
        values.extend(list(args.get("evidence_ids", []) or []))
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result[:32]


def _project_view_to_evidence_ids(view_data: Any, evidence_ids: List[str]) -> Any:
    if not evidence_ids:
        return None
    if isinstance(view_data, list):
        rows = [row for row in view_data if _contains_evidence_id(row, evidence_ids)]
        return rows or None
    if not isinstance(view_data, dict):
        return view_data if _contains_evidence_id(view_data, evidence_ids) else None
    for row_key in _FOLLOWUP_ROW_KEYS:
        raw_rows = view_data.get(row_key)
        if not isinstance(raw_rows, list):
            continue
        rows = [row for row in raw_rows if _contains_evidence_id(row, evidence_ids)]
        if not rows:
            return None
        projected = dict(view_data)
        projected[row_key] = rows
        projected["followup_replay_note"] = (
            "Only rows cited by the preceding Judge are replayed; request the "
            "view again for uncited detail."
        )
        return projected
    return view_data if _contains_evidence_id(view_data, evidence_ids) else None


def _contains_evidence_id(value: Any, evidence_ids: List[str]) -> bool:
    if isinstance(value, str):
        return value in evidence_ids
    if isinstance(value, dict):
        return any(_contains_evidence_id(item, evidence_ids) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_evidence_id(item, evidence_ids) for item in value)
    return False


def _section_cap(
    total: int,
    *,
    ratio: float,
    minimum: int,
    maximum: int,
) -> int:
    return min(maximum, max(minimum, int(total * ratio)))


def _allocate_non_evidence_caps(
    *,
    total_budget: int,
    raw_caps: Dict[str, int],
    has_state: bool,
    has_previous: bool,
    has_observations: bool,
) -> tuple[int, int, int]:
    evidence_floor = max(256, int(total_budget * 0.43))
    non_evidence_cap = max(0, total_budget - min(total_budget, evidence_floor))
    weights = {
        "state": 0.21 if has_state else 0.0,
        "previous": 0.18 if has_previous else 0.0,
        "observations": 0.61 if has_observations else 0.0,
    }
    weight_total = sum(weights.values())
    if weight_total <= 0:
        return 0, 0, 0
    caps = {
        name: min(
            int(raw_caps.get(name, 0) or 0),
            int(non_evidence_cap * (weight / weight_total)),
        )
        for name, weight in weights.items()
    }
    return caps["state"], caps["previous"], caps["observations"]


def _previous_judge_summary(
    value: Optional[Dict[str, Any]],
    *,
    earlier_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {}
    summary: Dict[str, Any] = {
        "answer": value.get("answer", "uncertain"),
        "confidence": value.get("confidence", ""),
        "reason": value.get("reason", ""),
        "supporting_evidence_ids": list(
            value.get("supporting_evidence_ids", value.get("evidence_refs", [])) or []
        )[:16],
        "contradicting_evidence_ids": list(
            value.get("contradicting_evidence_ids", []) or []
        )[:16],
        "missing_evidence": list(value.get("missing_evidence", []) or [])[:12],
        "suggested_followup_views": list(
            value.get("suggested_followup_views", []) or []
        )[:8],
        "tool_requests": list(value.get("tool_requests", []) or [])[:4],
    }
    feature_analysis = value.get("condition_feature_analysis")
    if isinstance(feature_analysis, dict) and feature_analysis:
        summary["condition_feature_analysis"] = feature_analysis
    earlier_summaries: List[Dict[str, Any]] = []
    for fallback_round, item in enumerate(list(earlier_results or [])[-6:]):
        if not isinstance(item, dict):
            continue
        earlier_summaries.append({
            "round": item.get("round", fallback_round),
            "answer": item.get("answer", "uncertain"),
            "confidence": item.get("confidence", ""),
            "reason": item.get("reason", ""),
            "missing_evidence": list(item.get("missing_evidence", []) or [])[:8],
            "suggested_followup_views": list(
                item.get("suggested_followup_views", []) or []
            )[:6],
            "condition_feature_analysis": item.get(
                "condition_feature_analysis",
                {},
            ),
        })
    if earlier_summaries:
        summary["earlier_round_summaries"] = earlier_summaries
    summary["round"] = value.get("round", len(earlier_summaries))
    return summary


def _fit_previous_summary(
    value: Dict[str, Any],
    budget: int,
) -> tuple[Dict[str, Any], str, bool]:
    if not value or budget <= 0:
        return {}, "", bool(value)
    full = stable_json_dumps(value)
    if len(full) <= budget:
        return value, full, False

    earlier = []
    for item in list(value.get("earlier_round_summaries", []) or [])[-4:]:
        if not isinstance(item, dict):
            continue
        earlier.append({
            "round": item.get("round"),
            "answer": item.get("answer", "uncertain"),
            "confidence": item.get("confidence", ""),
            "reason": _truncate_string(str(item.get("reason") or ""), 140),
        })
    compact: Dict[str, Any] = {
        "earlier_round_summaries": earlier,
        "round": value.get("round"),
        "answer": value.get("answer", "uncertain"),
        "confidence": value.get("confidence", ""),
        "reason": _truncate_string(str(value.get("reason") or ""), max(80, budget // 3)),
        "supporting_evidence_ids": list(
            value.get("supporting_evidence_ids", []) or []
        )[:8],
        "contradicting_evidence_ids": list(
            value.get("contradicting_evidence_ids", []) or []
        )[:8],
        "missing_evidence": list(value.get("missing_evidence", []) or [])[:6],
    }
    text = stable_json_dumps(compact)
    while len(text) > budget and compact["earlier_round_summaries"]:
        compact["earlier_round_summaries"].pop(0)
        text = stable_json_dumps(compact)
    for key in (
        "contradicting_evidence_ids",
        "supporting_evidence_ids",
        "missing_evidence",
    ):
        if len(text) <= budget:
            break
        compact.pop(key, None)
        text = stable_json_dumps(compact)
    if len(text) > budget:
        reason_budget = max(24, budget - len(stable_json_dumps({
            "answer": compact.get("answer"),
            "confidence": compact.get("confidence"),
            "reason": "",
        })) - 8)
        compact = {
            "answer": compact.get("answer"),
            "confidence": compact.get("confidence"),
            "reason": _truncate_string(str(value.get("reason") or ""), reason_budget),
        }
        text = stable_json_dumps(compact)
    return compact, text[:budget], True


def _format_observations(
    observations: List[Dict[str, Any]],
    budget: int,
) -> tuple[str, Dict[str, Any]]:
    if not observations or budget <= 0:
        return "", {
            "count": len(observations),
            "latest_preserved_full": False,
            "latest_truncated": False,
            "older_summary_count": 0,
            "older_omitted_count": max(0, len(observations) - 1),
        }

    latest_index = len(observations) - 1
    latest = observations[latest_index]
    source_index = next(
        (
            index
            for index in range(latest_index, -1, -1)
            if _is_successful_source_observation(observations[index])
        ),
        None,
    )
    source_is_latest = source_index == latest_index
    has_pinned_source = source_index is not None and not source_is_latest
    distinct_source_keys = {
        _observation_logical_key(observation)
        for observation in observations
        if _is_successful_source_observation(observation)
    }
    has_multiple_sources = len(distinct_source_keys) > 1
    if has_pinned_source:
        source_ratio = 0.36 if has_multiple_sources else 0.48
        latest_ratio = 0.26 if has_multiple_sources else 0.38
        source_budget = max(256, int(budget * source_ratio))
        latest_budget = max(192, int(budget * latest_ratio))
    else:
        source_budget = 0
        latest_budget = max(256, int(budget * 0.72))
    overhead_budget = 140 if has_pinned_source else 80
    older_budget = max(0, budget - latest_budget - source_budget - overhead_budget)
    if source_is_latest:
        latest_value, latest_truncated = _fit_source_observation(
            latest,
            latest_budget,
        )
    else:
        latest_value, _, latest_truncated = _fit_json(latest, latest_budget)
    pinned_source_value: Dict[str, Any] | None = None
    source_truncated = False
    if has_pinned_source and source_index is not None:
        pinned_source_value, source_truncated = _fit_source_observation(
            observations[source_index],
            source_budget,
        )

    candidate_older_indexes = [
        index
        for index in range(latest_index - 1, -1, -1)
        if index != source_index
    ]
    older_indexes: List[int] = []
    seen_observation_keys: set[str] = set()
    duplicate_observation_count = 0
    for index in candidate_older_indexes:
        observation_key = _observation_logical_key(observations[index])
        if observation_key in seen_observation_keys:
            duplicate_observation_count += 1
            continue
        seen_observation_keys.add(observation_key)
        older_indexes.append(index)
        if len(older_indexes) >= 8:
            break

    per_observation_budget = (
        min(6000, max(1200, older_budget // max(1, len(older_indexes))))
        if older_indexes and older_budget > 0
        else 0
    )
    older_summaries = []
    for index in older_indexes:
        summary = _compact_historical_observation(
            observations[index],
            per_observation_budget,
        )
        summary["sequence_index"] = index
        older_summaries.append(summary)
    compact_older, _, older_truncated = _fit_json(
        older_summaries,
        older_budget if older_summaries else 0,
    )
    if not isinstance(compact_older, list):
        compact_older = []

    payload: Dict[str, Any] = {
        "ordering": "newest_first",
        "chronology_hint": (
            "To replay oldest-to-newest, read older_summaries in reverse list "
            "order, then read latest."
        ),
        "latest_sequence_index": len(observations) - 1,
        "latest": latest_value,
        "older_summaries": compact_older,
    }
    if pinned_source_value is not None and source_index is not None:
        payload["pinned_source_sequence_index"] = source_index
        payload["pinned_source"] = pinned_source_value
    text = stable_json_dumps(payload)
    while len(text) > budget and payload["older_summaries"]:
        payload["older_summaries"].pop()
        text = stable_json_dumps(payload)
    if len(text) > budget:
        overhead = len(stable_json_dumps({
            "ordering": "newest_first",
            "chronology_hint": "",
            "latest_sequence_index": len(observations) - 1,
            "latest": {},
            "pinned_source_sequence_index": source_index,
            "pinned_source": pinned_source_value or {},
            "older_summaries": [],
        }))
        reduced_latest_budget = max(64, budget - overhead)
        if source_is_latest:
            latest_value, latest_truncated = _fit_source_observation(
                latest,
                reduced_latest_budget,
            )
        else:
            latest_value, _, latest_truncated = _fit_json(
                latest,
                reduced_latest_budget,
            )
        payload["latest"] = latest_value
        text = stable_json_dumps(payload)
    if len(text) > budget:
        payload = {
            "ordering": "newest_first",
            "chronology_hint": (
                "Replay older_summaries in reverse list order, then latest."
            ),
            "latest_sequence_index": len(observations) - 1,
            "latest_summary": _observation_summary(latest),
            "older_summaries": [],
        }
        if pinned_source_value is not None and source_index is not None:
            remaining = max(64, budget - len(stable_json_dumps(payload)) - 60)
            pinned_source_value, source_truncated = _fit_source_observation(
                observations[source_index],
                remaining,
            )
            payload["pinned_source_sequence_index"] = source_index
            payload["pinned_source"] = pinned_source_value
        _, text, _ = _fit_json(payload, budget)
        latest_truncated = True

    returned_older_items = [
        item
        for item in list(payload.get("older_summaries", []) or [])
        if not (isinstance(item, dict) and set(item) == {"omitted_items"})
    ]
    returned_older = len(returned_older_items)
    returned_older_keys = [
        str(item.get("logical_view_key") or "unknown")
        for item in returned_older_items
        if isinstance(item, dict)
    ]
    returned_compact_rows = sum(
        int(item.get("compact_row_count", 0) or 0)
        for item in returned_older_items
        if isinstance(item, dict)
    )
    return text, {
        "count": len(observations),
        "latest_preserved_full": not latest_truncated,
        "latest_truncated": bool(latest_truncated),
        "older_summary_count": returned_older,
        "older_omitted_count": max(
            0,
            len(candidate_older_indexes) - returned_older,
        ),
        "older_summaries_truncated": bool(older_truncated),
        "older_compaction_mode": "latest_per_view_structured",
        "older_compact_view_keys": returned_older_keys,
        "older_compact_row_count": returned_compact_rows,
        "older_duplicate_view_count": duplicate_observation_count,
        "source_observation_found": source_index is not None,
        "source_sequence_index": source_index,
        "source_preserved_separately": bool(has_pinned_source),
        "source_truncated": bool(source_truncated or (source_is_latest and latest_truncated)),
    }


def _is_successful_source_observation(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    request = value.get("request") if isinstance(value.get("request"), dict) else {}
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    tool = str(request.get("tool") or value.get("tool") or result.get("tool") or "")
    status = str(
        result.get("tool_status")
        or result.get("status")
        or value.get("tool_status")
        or value.get("status")
        or ""
    ).lower()
    if tool != "read_function_chunk" or status != "ok":
        return False
    return bool(_source_snippets(result))


def _source_snippets(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    raw_snippets = evidence.get("snippets") or result.get("snippets") or []
    snippets: List[Dict[str, Any]] = []
    for item in list(raw_snippets or []):
        if isinstance(item, dict):
            snippets.append(dict(item))
        elif item:
            snippets.append({"source_code": str(item)})
    return snippets


def _fit_source_observation(
    value: Dict[str, Any],
    budget: int,
) -> tuple[Dict[str, Any], bool]:
    request = value.get("request") if isinstance(value.get("request"), dict) else {}
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    snippets = _source_snippets(result)[:6]
    base = {
        "tool": "read_function_chunk",
        "tool_status": result.get("tool_status") or result.get("status") or "ok",
        "args": request.get("args", value.get("args", {})),
        "reason": _truncate_string(str(request.get("reason") or value.get("reason") or ""), 300),
        "summary": _compact_value(result.get("summary", {}), 500, 8),
        "returned_evidence_ids": list(result.get("returned_evidence_ids", []) or [])[:16],
        "snippets": [],
    }
    metadata_keys = (
        "evidence_id",
        "path",
        "start_line",
        "end_line",
        "matched_key",
        "match_score",
        "truncated",
    )
    shell_len = len(stable_json_dumps(base))
    available = max(0, budget - shell_len - 32)
    per_snippet = max(120, available // max(1, len(snippets)))
    source_was_truncated = False
    for snippet in snippets:
        code = str(snippet.get("source_code") or "")
        code_budget = max(80, per_snippet - 220)
        compact = {
            key: snippet.get(key)
            for key in metadata_keys
            if key in snippet
        }
        compact["source_code"] = _truncate_string(code, code_budget)
        source_was_truncated = source_was_truncated or len(code) > code_budget
        base["snippets"].append(compact)
    compact_value, text, generic_truncated = _fit_json(base, budget)
    if isinstance(compact_value, dict) and len(text) <= budget:
        return compact_value, bool(source_was_truncated or generic_truncated)
    return {"tool": "read_function_chunk", "truncated": True}, True


def _observation_summary(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {"summary": _truncate_string(str(value), 300)}
    request = value.get("request") if isinstance(value.get("request"), dict) else {}
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    tool = request.get("tool") or result.get("tool") or value.get("tool") or ""
    status = (
        result.get("tool_status")
        or result.get("status")
        or value.get("tool_status")
        or value.get("status")
        or ""
    )
    return {
        "tool": tool,
        "tool_status": status,
        "args": request.get("args", value.get("args", {})),
        "reason": _truncate_string(
            str(request.get("reason") or value.get("reason") or ""),
            300,
        ),
        "summary": _compact_value(result.get("summary", ""), 300, 5),
        "returned_evidence_ids": list(
            result.get("returned_evidence_ids", []) or []
        )[:12],
    }


def _observation_logical_key(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown"
    request = value.get("request") if isinstance(value.get("request"), dict) else {}
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    tool = str(request.get("tool") or result.get("tool") or value.get("tool") or "unknown")
    args = request.get("args") if isinstance(request.get("args"), dict) else {}
    if tool == "read_packet_view":
        view_name = str(
            args.get("view")
            or args.get("view_name")
            or ((result.get("summary") or {}).get("view") if isinstance(result.get("summary"), dict) else "")
            or "unknown"
        )
        return f"read_packet_view:{view_name}"
    if tool in {"read_evidence_context", "get_local_call_context", "read_evidence_by_id"}:
        target = str(
            args.get("evidence_id")
            or ",".join(str(item) for item in list(args.get("evidence_ids", []) or [])[:4])
            or "unknown"
        )
        return f"{tool}:{target}"
    if tool == "read_function_chunk":
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        evidence_id = str(
            args.get("evidence_id")
            or summary.get("evidence_id")
            or "unknown"
        )
        target_function = str(
            args.get("target_function")
            or args.get("function_name")
            or summary.get("requested_function")
            or summary.get("resolved_function")
            or "unknown"
        )
        return f"read_function_chunk:{evidence_id}:{target_function}"
    return tool


def _compact_historical_observation(
    value: Any,
    budget: int,
) -> Dict[str, Any]:
    base = _observation_summary(value)
    base["logical_view_key"] = _observation_logical_key(value)
    base["compaction"] = "structured_rows"
    if not isinstance(value, dict) or budget <= 0:
        base["compact_row_count"] = 0
        return base

    request = value.get("request") if isinstance(value.get("request"), dict) else {}
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    tool = str(request.get("tool") or result.get("tool") or value.get("tool") or "")
    args = request.get("args") if isinstance(request.get("args"), dict) else {}
    view_name = str(args.get("view") or args.get("view_name") or "")
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    rows = list(evidence.get("rows", []) or [])

    if tool == "read_function_chunk" and _is_successful_source_observation(value):
        compact_source, truncated = _fit_source_observation(value, budget)
        compact_source["logical_view_key"] = _observation_logical_key(value)
        compact_source["compaction"] = "source_snippets"
        compact_source["compact_row_count"] = len(
            list(compact_source.get("snippets", []) or [])
        )
        compact_source["structured_compact_truncated"] = bool(truncated)
        return compact_source

    compact_rows: List[Dict[str, Any]] = []
    if tool == "read_packet_view" and view_name == "reentrancy_state_order_view":
        compact_rows = [
            _compact_state_order_row(row)
            for row in rows[:6]
            if isinstance(row, dict)
        ]
    elif tool == "read_packet_view" and view_name == "semantic_state_delta_view":
        compact_rows = [
            _select_fields(row, _SEMANTIC_STATE_ROW_FIELDS)
            for row in rows[:8]
            if isinstance(row, dict)
        ]
    elif tool == "read_packet_view" and view_name == "value_release_view":
        compact_rows = [
            _compact_value_release_row(row)
            for row in rows[:6]
            if isinstance(row, dict)
        ]
    elif tool in {"read_evidence_context", "get_local_call_context", "read_evidence_by_id"}:
        compact_rows = [
            _select_fields(row, _LOCAL_CONTEXT_ROW_FIELDS)
            for row in rows[:8]
            if isinstance(row, dict)
        ]
    elif rows:
        compact_rows = [
            _compact_value(row, 240, 6)
            for row in rows[:6]
            if isinstance(row, dict)
        ]

    base["compact_row_count"] = len(compact_rows)
    if compact_rows:
        base["compact_rows"] = compact_rows
    compact, _, truncated = _fit_json(base, budget)
    if not isinstance(compact, dict):
        compact = _observation_summary(value)
        compact["compact_row_count"] = 0
    compact["structured_compact_truncated"] = bool(truncated)
    return compact


def _compact_state_order_row(row: Dict[str, Any]) -> Dict[str, Any]:
    compact = _select_fields(row, _STATE_ORDER_ROW_FIELDS)
    slot_summaries: List[Dict[str, Any]] = []
    for slot in list(row.get("slot_access_summary", []) or [])[:8]:
        if not isinstance(slot, dict):
            continue
        compact_slot = {
            key: slot.get(key)
            for key in (
                "slot_key_short",
                "access_count",
                "read_count",
                "write_count",
            )
            if key in slot
        }
        compact_slot["accesses"] = [
            _select_fields(access, _STATE_ACCESS_FIELDS)
            for access in list(slot.get("accesses", []) or [])[:8]
            if isinstance(access, dict)
        ]
        slot_summaries.append(compact_slot)
    compact["slot_access_summary"] = slot_summaries
    return compact


def _compact_value_release_row(row: Dict[str, Any]) -> Dict[str, Any]:
    compact = _select_fields(row, _VALUE_RELEASE_ROW_FIELDS)
    compact["recipients"] = [
        _select_fields(
            recipient,
            (
                "recipient",
                "recipient_label",
                "token",
                "token_label",
                "amount_sum",
                "release_count",
                "evidence_ids",
            ),
        )
        for recipient in list(row.get("recipients", []) or [])[:4]
        if isinstance(recipient, dict)
    ]
    return compact


def _select_fields(
    value: Dict[str, Any],
    fields: tuple[str, ...],
) -> Dict[str, Any]:
    return {
        key: _compact_value(value.get(key), 300, 12)
        for key in fields
        if key in value
    }


def _fit_json(value: Any, budget: int) -> tuple[Any, str, bool]:
    if budget <= 0:
        empty = {} if isinstance(value, dict) else [] if isinstance(value, list) else ""
        return empty, "", bool(value)
    full = stable_json_dumps(value)
    if len(full) <= budget:
        return value, full, False

    for string_limit, list_limit in (
        (1000, 12),
        (600, 8),
        (350, 6),
        (200, 4),
        (100, 3),
        (60, 2),
        (30, 1),
    ):
        compact = _compact_value(value, string_limit, list_limit)
        text = stable_json_dumps(compact)
        if len(text) <= budget:
            return compact, text, True

    fallback = {
        "truncated": True,
        "summary": _truncate_string(full, max(16, budget - 45)),
    }
    text = stable_json_dumps(fallback)
    while len(text) > budget and fallback["summary"]:
        fallback["summary"] = fallback["summary"][:-1]
        text = stable_json_dumps(fallback)
    if len(text) > budget:
        fallback = {"truncated": True}
        text = stable_json_dumps(fallback)
    return fallback, text[:budget], True


def _compact_value(value: Any, string_limit: int, list_limit: int) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, string_limit, list_limit)
            for key, item in value.items()
        }
    if isinstance(value, list):
        compacted = [
            _compact_value(item, string_limit, list_limit)
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            compacted.append({"omitted_items": len(value) - list_limit})
        return compacted
    if isinstance(value, str):
        return _truncate_string(value, string_limit)
    return value


def _truncate_string(
    value: str,
    max_chars: int,
    marker: str = "...<truncated>",
) -> str:
    text = str(value or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)] + marker
