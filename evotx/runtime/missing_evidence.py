from __future__ import annotations

import re
import hashlib
from typing import Any, Dict, Iterable, List, Optional


VIEW_NAMES = {
    "trace_view",
    "operation_summary_view",
    "classification_digest_view",
    "critical_call_view",
    "reentrancy_candidate_catalog_view",
    "reentrancy_state_order_summary_view",
    "reentrancy_state_order_view",
    "unknown_selector_view",
    "event_view",
    "state_change_view",
    "semantic_state_delta_view",
    "price_relevant_state_view",
    "amm_reserve_transition_view",
    "market_mechanism_profile_view",
    "transfer_event_view",
    "external_fundflow_view",
    "profit_loss_view",
    "participant_net_delta_view",
    "value_release_view",
    "contribution_vs_payout_view",
    "flash_or_atomic_capital_view",
    "beneficiary_controller_view",
}


def normalize_missing_evidence(
    text: Any,
    condition_id: str,
    round_id: int,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize free-form judge missing evidence into an internal lifecycle row."""
    context = context or {}
    raw = str(text or "").strip()
    category = _classify_missing(raw, context)
    status = "open"
    blocking = True

    if category in {"not_applicable"}:
        status = "non_actionable"
        blocking = False
    elif category in {"external_market_data", "historical_relationship"}:
        blocking = False
    elif category == "prompt_render_truncation" and _packet_trace_truncated_false(context):
        # Prompt render truncation is actionable by read_packet_view, but it is
        # not packet evidence loss. It remains open only until a packet read
        # returns the requested view.
        blocking = True

    return {
        "id": f"missing:{condition_id}:{round_id}:{_slug(category)}:{_stable_short_hash(raw)}",
        "condition_id": condition_id,
        "round": round_id,
        "text": raw,
        "category": category,
        "status": status,
        "blocking": blocking,
        "resolved_by": None,
    }


def dedupe_missing_evidence(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (
            item.get("condition_id"),
            item.get("text"),
            item.get("category"),
            item.get("status"),
            item.get("resolved_by"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def resolve_missing_evidence_after_tool_call(
    missing_items: Iterable[Dict[str, Any]],
    tool_call: Dict[str, Any],
    tool_observation: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    observation = tool_observation or tool_call.get("observation") or {}
    summary = observation.get("summary") or tool_call.get("summary") or {}
    tool = str(tool_call.get("tool") or observation.get("tool") or "")
    args = dict(tool_call.get("args") or observation.get("args") or {})
    view = str(args.get("view") or summary.get("view") or "").strip()
    total_rows = _to_int(summary.get("total_rows"))
    returned_rows = _to_int(summary.get("returned_rows"))
    status = str(observation.get("tool_status") or tool_call.get("tool_status") or "")

    returned_views = _extract_returned_views(observation)

    resolved: List[Dict[str, Any]] = []
    for raw_item in missing_items:
        item = dict(raw_item)
        if item.get("status") != "open":
            resolved.append(item)
            continue
        text = str(item.get("text", "")).lower()
        category = item.get("category")

        if tool == "read_packet_view":
            all_rows_returned = returned_rows > 0 and (total_rows == 0 or returned_rows >= total_rows)
            if (
                category == "prompt_render_truncation"
                and view == "trace_view"
                and all_rows_returned
            ):
                item.update({
                    "status": "resolved",
                    "blocking": False,
                    "resolved_by": f"read_packet_view({view})",
                })
            elif category == "available_packet_view" and view and _mentions_view(text, view) and returned_rows > 0:
                item.update({
                    "status": "resolved",
                    "blocking": False,
                    "resolved_by": f"read_packet_view({view})",
                })
            elif view and _mentions_view(text, view) and returned_rows > 0:
                item.update({
                    "status": "stale",
                    "blocking": False,
                    "resolved_by": f"read_packet_view({view})",
                })
            elif view and returned_rows == 0 and _mentions_view(text, view):
                item.update({
                    "status": "resolved",
                    "blocking": False,
                    "resolved_by": f"read_packet_view({view}):view_empty",
                })

        elif tool == "read_evidence_by_id":
            if returned_rows > 0:
                for rv in returned_views:
                    if _mentions_view(text, rv):
                        item.update({
                            "status": "resolved",
                            "blocking": False,
                            "resolved_by": f"read_evidence_by_id:{rv}",
                        })
                        break
                if category == "source_tool" and returned_rows > 0:
                    item.update({
                        "status": "resolved",
                        "blocking": False,
                        "resolved_by": "read_evidence_by_id:returned_rows",
                    })

        elif tool == "read_evidence_context":
            if returned_rows > 0:
                for rv in returned_views:
                    if _mentions_view(text, rv):
                        item.update({
                            "status": "resolved",
                            "blocking": False,
                            "resolved_by": f"read_evidence_context:{rv}",
                        })
                        break
                evidence_id = str(args.get("evidence_id") or "").strip()
                if evidence_id and category == "available_packet_view":
                    item.update({
                        "status": "resolved",
                        "blocking": False,
                        "resolved_by": f"read_evidence_context({evidence_id})",
                    })

        elif tool == "read_function_chunk" and status in {"ok", "source_unavailable", "function_not_found"}:
            if category == "source_tool":
                item.update({
                    "status": "resolved" if status == "ok" else "non_actionable",
                    "blocking": status == "ok",
                    "resolved_by": f"read_function_chunk:{status}",
                })

        resolved.append(item)
    return dedupe_missing_evidence(resolved)


def open_blocking_texts(items: Iterable[Dict[str, Any]]) -> List[str]:
    return [
        str(item.get("text", ""))
        for item in dedupe_missing_evidence(items)
        if item.get("status") == "open" and item.get("blocking") and item.get("text")
    ]


def group_missing_by_condition(items: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    grouped: Dict[str, Dict[str, List[str]]] = {}
    for item in dedupe_missing_evidence(items):
        condition_id = str(item.get("condition_id") or "unknown")
        status = str(item.get("status") or "open")
        grouped.setdefault(condition_id, {
            "open": [],
            "resolved": [],
            "stale": [],
            "non_actionable": [],
        })
        bucket = status if status in grouped[condition_id] else "open"
        text = str(item.get("text", ""))
        if text and text not in grouped[condition_id][bucket]:
            grouped[condition_id][bucket].append(text)
    return grouped


def actionability_summary(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "open_blocking": 0,
        "resolved": 0,
        "non_actionable": 0,
        "external_market_data": 0,
        "prompt_render_truncation": 0,
        "packet_trace_truncation": 0,
    }
    for item in dedupe_missing_evidence(items):
        category = str(item.get("category") or "unknown")
        status = str(item.get("status") or "open")
        if status == "open" and item.get("blocking"):
            summary["open_blocking"] += 1
        if status == "resolved":
            summary["resolved"] += 1
        if status == "non_actionable":
            summary["non_actionable"] += 1
        if category in summary:
            summary[category] += 1
    return summary


def _classify_missing(text: str, context: Dict[str, Any]) -> str:
    lower = text.lower()
    if _mentions_trace_truncation(lower):
        if _packet_trace_truncated_true(context):
            return "packet_trace_truncation"
        return "prompt_render_truncation"
    if "external market" in lower or "market price" in lower or "usd price" in lower:
        return "external_market_data"
    if "historical" in lower or "prior transaction" in lower or "cross-transaction" in lower or "same block" in lower:
        return "historical_relationship"
    if "source code" in lower or "function source" in lower or "read_function" in lower or "access control" in lower:
        return "source_tool"
    if "amm reserve" in lower and any(word in lower for word in ("empty", "not applicable", "non-amm", "vault deposit")):
        return "not_applicable"
    if any(_mentions_view(lower, view) for view in VIEW_NAMES):
        return "available_packet_view"
    return "unknown"


def _mentions_trace_truncation(text: str) -> bool:
    return (
        ("trace_view" in text or "full trace" in text or "trace" in text)
        and ("truncated" in text or "untruncated" in text or "not fully" in text)
    )


def _mentions_view(text: str, view: str) -> bool:
    normalized = view.lower()
    return normalized in text or normalized.replace("_", " ") in text


def _packet_trace_truncated_true(context: Dict[str, Any]) -> bool:
    metadata = context.get("view_render_metadata") or []
    for row in metadata:
        if row.get("view") == "trace_view" and row.get("packet_trace_truncated"):
            return True
    adequacy = context.get("evidence_adequacy_summary") or context.get("evidence_adequacy_view") or {}
    trace = adequacy.get("trace", {}) if isinstance(adequacy, dict) else {}
    return bool(trace.get("truncated"))


def _packet_trace_truncated_false(context: Dict[str, Any]) -> bool:
    return not _packet_trace_truncated_true(context)


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "unknown")).strip("_") or "unknown"


def _stable_short_hash(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:10]


def _extract_returned_views(observation: Dict[str, Any]) -> List[str]:
    """Extract distinct view names from returned evidence rows via _view field."""
    evidence = observation.get("evidence") or {}
    rows = evidence.get("rows") or []
    views: List[str] = []
    seen: set = set()
    for row in rows:
        if isinstance(row, dict):
            v = str(row.get("_view") or "")
            if v and v not in seen:
                seen.add(v)
                views.append(v)
    return views
