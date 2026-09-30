from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Optional

from evotx.core.plan import make_local_judge_question


DIAGNOSIS_ROUTING_SCHEMA_VERSION = "evotx.diagnosis_routing.v1"

OWNER_MUTABLE_SURFACES = {
    "rule": {
        "semantic_rewrite",
        "add_positive_condition",
        "add_exclusion",
        "narrow_exclusion",
        "remove_positive_condition",
        "remove_exclusion",
    },
    "plan": {
        "change_evidence_route",
        "change_dependency_route",
        "restore_question_from_rule",
    },
    "packet": {"engineering_packet"},
    "runtime": {"engineering_runtime"},
    "none": set(),
}

_FAILED_TOOL_STATUSES = {
    "error",
    "failed",
    "function_not_found",
    "invalid",
    "invalid_args",
    "timeout",
    "tool_error",
    "validation_error",
}

_STEP_ROUTE_BLOCK_PATTERNS = (
    "is not allowed for this judge step",
    "is not in allowed_followup_views for this judge step",
)

_RULE_CATEGORIES = {
    "rule_too_broad",
    "rule_too_narrow",
    "exclusion_too_broad",
    "missing_exclusion",
    "missing_condition",
    "weak_evidence_as_sufficient",
    "bad_semantic_condition",
    "binding_semantic_failure",
}

_PLAN_EVIDENCE_CATEGORIES = {
    "bad_packet_view_selection",
    "binding_route_failure",
}


def is_authoritative_engineering_route(route: Dict[str, Any]) -> bool:
    """Return whether runtime facts, rather than reviewer prose, prove engineering ownership."""
    owner = str(route.get("owner") or "").strip().lower()
    category = str(route.get("category") or "").strip().lower()
    basis = [str(item or "").strip().lower() for item in list(route.get("fact_basis") or [])]
    if owner == "packet":
        return category == "packet_build_or_load_failure" or any(
            item == "packet_trace_truncated" or item.startswith("packet_view_absent:")
            for item in basis
        )
    if owner == "runtime":
        return category in {"runtime_emit_logic_bug", "binding_runtime_failure"} or any(
            item.startswith("tool_attempt_failed:")
            or item.startswith("fallback_tool_attempt_failed:")
            or item == "runtime_ignored_requested_tool"
            or item == "runtime_ignored_requested_fallback"
            for item in basis
        )
    return False


def _condition_matches(condition_id: str, affected: Iterable[Any]) -> bool:
    target = str(condition_id or "").strip().upper()
    values = _condition_ids(affected)
    return not target or not values or target in values


def _matching_missing_origins(
    missing_origins: List[Dict[str, Any]],
    condition_id: str,
) -> List[Dict[str, Any]]:
    target = str(condition_id or "").strip().upper()
    return [
        item
        for item in missing_origins
        if not target or str(item.get("condition_id") or "").strip().upper() == target
    ]


def _packet_trace_truncated(case: Dict[str, Any]) -> bool:
    evidence_digest = dict(case.get("evidence_digest") or {})
    adequacy = dict(evidence_digest.get("evidence_adequacy_view") or {})
    trace = dict(adequacy.get("trace") or {})
    return bool(
        trace.get("truncated")
        or adequacy.get("packet_trace_truncated")
        or adequacy.get("trace_truncated")
    )


def _packet_trace_truncation_origins(
    case: Dict[str, Any],
    root_causes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Project recorded trace truncation onto Reviewer-scoped affected conditions."""
    if not _packet_trace_truncated(case):
        return []
    condition_ids = {
        condition_id
        for root in root_causes
        if str(root.get("category") or "").strip().lower()
        in {"missing_evidence", "missing_packet_view", "packet_build_or_load_failure"}
        and str(root.get("failure_origin") or "").strip().lower()
        in {"packet", "packet_trace_truncation"}
        for condition_id in _condition_ids(root.get("affected_conditions"))
    }
    return [
        {
            "missing_evidence_id": f"runtime:packet_trace_truncated:{condition_id}",
            "condition_id": condition_id,
            "category": "packet_trace_truncation",
            "status": "open",
            "origin": "packet",
            "owner": "packet",
            "requested_operation": "engineering_packet",
            "requested_views": ["trace_view"],
            "attempted_tools": [],
            "fact_basis": ["packet_trace_truncated"],
            "reviewer_scope": "semantic_impact_only",
        }
        for condition_id in sorted(condition_ids)
    ]


def _has_root_category(
    root_causes: List[Dict[str, Any]],
    condition_id: str,
    categories: set[str],
) -> bool:
    return any(
        str(root.get("category") or "").strip().lower() in categories
        and _condition_matches(condition_id, root.get("affected_conditions"))
        for root in root_causes
    )


def _has_independent_plan_evidence_root(
    root_causes: List[Dict[str, Any]],
    condition_id: str,
) -> bool:
    if _has_root_category(root_causes, condition_id, _PLAN_EVIDENCE_CATEGORIES):
        return True
    for root in root_causes:
        if (
            str(root.get("category") or "").strip().lower() != "bad_judge_question"
            or not _condition_matches(condition_id, root.get("affected_conditions"))
        ):
            continue
        origin = str(
            root.get("failure_origin")
            or root.get("question_failure_kind")
            or ""
        ).strip().lower()
        if origin in {"evidence_route", "plan_evidence_route", "plan"}:
            return True
    return False


def _normalize_evidence_signal_owner(
    signal: Dict[str, Any],
    *,
    root_causes: List[Dict[str, Any]],
    missing_origins: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Project evidence signals onto the closed, runtime-observed capability surface."""
    normalized = deepcopy(signal)
    source_owner = str(normalized.get("update_target") or "none").strip().lower()
    if source_owner not in {"plan", "packet"}:
        return normalized

    condition_id = str(normalized.get("condition_id") or "").strip().upper()
    matching = _matching_missing_origins(missing_origins, condition_id)
    observed_owners = {
        str(item.get("owner") or "none").strip().lower()
        for item in matching
    }
    independent_plan_root = _has_independent_plan_evidence_root(
        root_causes,
        condition_id,
    )
    packet_build_failure = _has_root_category(
        root_causes,
        condition_id,
        {"packet_build_or_load_failure"},
    )

    target_owner = source_owner
    if source_owner == "packet":
        if "runtime" in observed_owners:
            target_owner = "runtime"
        elif "packet" in observed_owners or packet_build_failure:
            target_owner = "packet"
        elif "plan" in observed_owners:
            target_owner = "plan"
        else:
            target_owner = "none"
    elif not independent_plan_root and matching:
        if "runtime" in observed_owners:
            target_owner = "runtime"
        elif "packet" in observed_owners:
            target_owner = "packet"
        elif observed_owners and observed_owners.issubset({"none"}):
            target_owner = "none"

    if target_owner == source_owner:
        return normalized

    normalized["source_update_target"] = source_owner
    normalized["update_target"] = target_owner
    if target_owner == "plan":
        normalized["direction"] = "change_evidence_strategy"
    elif target_owner == "none":
        normalized["generalization_status"] = "insufficient"
        normalized["signal_gate_reason"] = "evidence_objectively_unavailable"
        normalized["condition_evidence_dependency"] = {}
    return normalized


def derive_missing_evidence_origins(
    missing_items: Iterable[Dict[str, Any]],
    *,
    condition_table: Iterable[Dict[str, Any]],
    packet_view_summary: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Classify missing-evidence ownership from recorded runtime facts.

    Reviewer text is deliberately not used to decide whether the owner is the
    Plan, Packet, Runtime, or no mutable knowledge component.
    """
    rows = {
        str(row.get("condition_id") or row.get("id") or "").strip().upper(): row
        for row in condition_table
        if isinstance(row, dict)
    }
    origins: List[Dict[str, Any]] = []
    for raw in missing_items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        condition_id = str(item.get("condition_id") or "").strip().upper()
        row = rows.get(condition_id, {})
        text = str(item.get("text") or "")
        category = str(item.get("category") or "unknown").strip().lower()
        status = str(item.get("status") or "open").strip().lower()
        requested_views = _requested_views(text, row, packet_view_summary)
        attempted_tools = _relevant_attempted_tools(
            category,
            requested_views,
            _attempted_tools(row),
        )
        ignored_requests = list(row.get("ignored_tool_requests") or [])
        origin, owner, operation, basis = _classify_missing_origin(
            category=category,
            status=status,
            requested_views=requested_views,
            attempted_tools=attempted_tools,
            ignored_requests=ignored_requests,
            packet_view_summary=packet_view_summary,
            resolved_by=str(item.get("resolved_by") or ""),
        )
        origins.append({
            "missing_evidence_id": str(item.get("id") or ""),
            "condition_id": condition_id,
            "category": category,
            "status": status,
            "origin": origin,
            "owner": owner,
            "requested_operation": operation,
            "requested_views": requested_views,
            "attempted_tools": attempted_tools,
            "fact_basis": basis,
            "reviewer_scope": "semantic_impact_only",
        })
    return _dedupe_dicts(
        origins,
        keys=("missing_evidence_id", "condition_id", "origin", "requested_operation"),
    )


def apply_diagnosis_routing(
    review: Dict[str, Any],
    case: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach owner/reachability records without collapsing a review to one owner."""
    normalized = deepcopy(review or {})
    routes: List[Dict[str, Any]] = []
    root_causes = [
        dict(item)
        for item in list(normalized.get("root_causes") or [])
        if isinstance(item, dict)
    ]
    missing_origin_history = list(
        dict(case.get("evidence_digest") or {}).get(
            "missing_evidence_origins", []
        )
        or case.get("missing_evidence_origins", [])
        or []
    )
    missing_origins = [
        dict(item)
        for item in missing_origin_history
        if isinstance(item, dict)
        and str(item.get("status") or "open").strip().lower()
        not in {"resolved", "stale", "non_actionable"}
    ]
    missing_origins = _dedupe_dicts(
        [
            *missing_origins,
            *_packet_trace_truncation_origins(case, root_causes),
        ],
        keys=("missing_evidence_id", "condition_id", "origin", "requested_operation"),
    )
    normalized["update_signals"] = [
        _normalize_evidence_signal_owner(
            dict(signal),
            root_causes=root_causes,
            missing_origins=missing_origins,
        )
        for signal in list(normalized.get("update_signals") or [])
        if isinstance(signal, dict)
    ]

    for root_index, root in enumerate(root_causes):
        root_routes = _routes_for_root_cause(
            root,
            review=normalized,
            case=case,
            missing_origins=missing_origins,
        )
        route_ids: List[str] = []
        for route in root_routes:
            route["source"] = "root_cause"
            route["source_index"] = root_index
            route["route_id"] = f"route_{len(routes) + 1:03d}"
            route_ids.append(route["route_id"])
            routes.append(route)
        root["diagnosis_route_ids"] = route_ids

    for diagnosis_index, diagnosis in enumerate(
        list(normalized.get("condition_diagnosis") or [])
    ):
        if not isinstance(diagnosis, dict):
            continue
        condition_ids = [diagnosis.get("condition_id")]
        diagnosis_specs = (
            ("is_rule_problem", "rule", "semantic_rewrite"),
            ("is_plan_problem", "plan", "change_evidence_route"),
            ("is_packet_problem", "packet", "engineering_packet"),
            ("is_runtime_problem", "runtime", "engineering_runtime"),
        )
        for flag, owner, operation in diagnosis_specs:
            if not bool(diagnosis.get(flag)):
                continue
            if flag == "is_plan_problem" and _condition_has_bad_question_root(
                root_causes,
                diagnosis.get("condition_id"),
            ):
                # The bad_judge_question root must establish question drift vs
                # evidence-route insufficiency; a generic Plan flag cannot do it.
                continue
            packet_owner_unproven = (
                owner == "packet"
                and "packet" not in {
                    str(item.get("owner") or "none").strip().lower()
                    for item in _matching_missing_origins(
                        missing_origins,
                        str(diagnosis.get("condition_id") or ""),
                    )
                }
                and not _has_root_category(
                    root_causes,
                    str(diagnosis.get("condition_id") or ""),
                    {"packet_build_or_load_failure"},
                )
            )
            route = _make_route(
                category="condition_diagnosis",
                owner=owner,
                operation=operation,
                affected_conditions=condition_ids,
                fact_basis=[flag],
                unreachable_reason=(
                    "packet_owner_not_supported_by_runtime_facts"
                    if packet_owner_unproven
                    else ""
                ),
            )
            route["source"] = "condition_diagnosis"
            route["source_index"] = diagnosis_index
            route["route_id"] = f"route_{len(routes) + 1:03d}"
            routes.append(route)

    for signal_index, signal in enumerate(list(normalized.get("update_signals") or [])):
        if not isinstance(signal, dict):
            continue
        target = str(signal.get("update_target") or "").strip().lower()
        if target not in {"rule", "plan", "packet", "runtime"}:
            continue
        operation = _operation_for_signal(signal, normalized)
        route = _make_route(
            category="update_signal",
            owner=target,
            operation=operation,
            affected_conditions=[signal.get("condition_id")],
            fact_basis=["reviewer_owner_specific_signal"],
        )
        route["source"] = "update_signal"
        route["source_index"] = signal_index
        route["route_id"] = f"route_{len(routes) + 1:03d}"
        # The signal matrix owns status and owner after review normalization.
        # Routing records the requested mutable surface for audit; it must not
        # veto a canonical signal because another root has a different owner.
        routes.append(route)

    routes = _dedupe_routes(routes)
    for root_index, root in enumerate(root_causes):
        root["diagnosis_route_ids"] = [
            str(route.get("route_id") or "")
            for route in routes
            if route.get("source") == "root_cause"
            and int(route.get("source_index", -1)) == root_index
        ]
    reachable = [route for route in routes if route.get("route_status") == "reachable"]
    owner_targets = sorted({str(route.get("owner")) for route in reachable if route.get("owner") != "none"})
    normalized["root_causes"] = root_causes
    normalized["diagnosis_routes"] = routes
    normalized["owner_targets"] = owner_targets
    normalized["routing_reachability"] = {
        "schema_version": DIAGNOSIS_ROUTING_SCHEMA_VERSION,
        "reachable_count": len(reachable),
        "unreachable_count": sum(
            1 for route in routes if route.get("route_status") == "unreachable"
        ),
        "deferred_count": sum(
            1 for route in routes if route.get("route_status") == "deferred"
        ),
        "owner_targets": owner_targets,
    }
    if routes:
        normalized["should_update_rule"] = "rule" in owner_targets
        normalized["should_update_plan_strategy"] = "plan" in owner_targets
        normalized["should_update_packet_builder"] = "packet" in owner_targets
        normalized["should_fix_runtime"] = "runtime" in owner_targets
    return normalized


def project_review_for_owner(
    review: Dict[str, Any],
    owner: str,
    *,
    source_review_index: int,
) -> Optional[Dict[str, Any]]:
    """Return one owner-specific view of a review, preserving source identity."""
    target = str(owner or "").strip().lower()
    all_routes = [
        dict(route)
        for route in list((review or {}).get("diagnosis_routes") or [])
        if isinstance(route, dict)
    ]
    routes = [
        dict(route)
        for route in all_routes
        if str(route.get("owner") or "").strip().lower() == target
        and str(route.get("route_status") or "") == "reachable"
    ]
    reachable_signal_indexes = {
        int(route.get("source_index", -1))
        for route in routes
        if route.get("source") == "update_signal"
    }
    signals = []
    for signal_index, signal in enumerate(
        list((review or {}).get("update_signals") or [])
    ):
        if not isinstance(signal, dict):
            continue
        if str(signal.get("update_target") or "").strip().lower() != target:
            continue
        source_signal_index = int(signal.get("source_signal_index", signal_index))
        escalation_owner = str(
            dict(signal.get("owner_escalation") or {}).get("owner") or ""
        ).strip().lower()
        if (
            all_routes
            and source_signal_index not in reachable_signal_indexes
            and escalation_owner != target
        ):
            continue
        signals.append(dict(signal))
    if not routes and not signals:
        return None

    projected = deepcopy(review or {})
    route_ids = {str(route.get("route_id") or "") for route in routes}
    projected["source_review_index"] = int(source_review_index)
    projected["update_target"] = target
    projected["diagnosis_routes"] = routes
    projected["owner_targets"] = [target]
    projected["update_signals"] = signals
    projected["root_causes"] = [
        dict(root)
        for root in list((review or {}).get("root_causes") or [])
        if isinstance(root, dict)
        and route_ids.intersection(root.get("diagnosis_route_ids") or [])
    ]
    diagnosis_flag = {
        "rule": "is_rule_problem",
        "plan": "is_plan_problem",
        "packet": "is_packet_problem",
        "runtime": "is_runtime_problem",
    }[target]
    projected["condition_diagnosis"] = [
        dict(item)
        for item in list((review or {}).get("condition_diagnosis") or [])
        if isinstance(item, dict) and bool(item.get(diagnosis_flag))
    ]
    projected["owner_projection"] = {
        "schema_version": DIAGNOSIS_ROUTING_SCHEMA_VERSION,
        "owner": target,
        "source_review_index": int(source_review_index),
        "route_ids": sorted(route_ids),
    }
    projected["should_update_rule"] = target == "rule"
    projected["should_update_plan_strategy"] = target == "plan"
    projected["should_update_packet_builder"] = target == "packet"
    projected["should_fix_runtime"] = target == "runtime"
    for candidate in ("rule", "plan", "packet", "runtime"):
        if candidate != target:
            projected[f"{candidate}_patch_suggestion"] = {"action": "none"}
    return projected


def build_diagnosis_routing_audit(reviews: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    review_list = list(reviews or [])
    route_rows = [
        {**dict(route), "source_review_index": review_index}
        for review_index, review in enumerate(review_list)
        if isinstance(review, dict)
        for route in list(review.get("diagnosis_routes") or [])
        if isinstance(route, dict)
    ]
    return {
        "schema_version": DIAGNOSIS_ROUTING_SCHEMA_VERSION,
        "review_count": len(review_list),
        "route_count": len(route_rows),
        "reachable_count": sum(1 for row in route_rows if row.get("route_status") == "reachable"),
        "unreachable_count": sum(1 for row in route_rows if row.get("route_status") == "unreachable"),
        "deferred_count": sum(1 for row in route_rows if row.get("route_status") == "deferred"),
        "owner_counts": {
            owner: sum(
                1
                for row in route_rows
                if row.get("owner") == owner and row.get("route_status") == "reachable"
            )
            for owner in ("rule", "plan", "packet", "runtime", "none")
        },
        "routes": route_rows,
    }


def review_has_operation(review: Dict[str, Any], owner: str, operation: str) -> bool:
    return any(
        isinstance(route, dict)
        and route.get("owner") == owner
        and route.get("requested_operation") == operation
        and route.get("route_status") == "reachable"
        for route in list((review or {}).get("diagnosis_routes") or [])
    )


def review_operation_condition_ids(
    review: Dict[str, Any], owner: str, operation: str
) -> List[str]:
    out: List[str] = []
    for route in list((review or {}).get("diagnosis_routes") or []):
        if not isinstance(route, dict):
            continue
        if (
            route.get("owner") != owner
            or route.get("requested_operation") != operation
            or route.get("route_status") != "reachable"
        ):
            continue
        for condition_id in list(route.get("affected_conditions") or []):
            value = str(condition_id or "").strip().upper()
            if value and value not in out:
                out.append(value)
    return out


def _classify_missing_origin(
    *,
    category: str,
    status: str,
    requested_views: List[str],
    attempted_tools: List[Dict[str, Any]],
    ignored_requests: List[Any],
    packet_view_summary: Dict[str, Dict[str, Any]],
    resolved_by: str,
) -> tuple[str, str, str, List[str]]:
    if status in {"resolved", "stale"}:
        return "resolved", "none", "none", [f"missing_status:{status}"]
    if status == "non_actionable" or category in {
        "external_market_data",
        "historical_relationship",
        "not_applicable",
    }:
        return "unavailable", "none", "none", [
            f"missing_category:{category}",
            f"resolved_by:{resolved_by}" if resolved_by else "runtime_marked_non_actionable",
        ]

    statuses = {str(item.get("status") or "").strip().lower() for item in attempted_tools}
    step_route_blocks = [
        item
        for item in attempted_tools
        if str(item.get("status") or "").strip().lower() == "blocked"
        and any(
            marker in str(
                item.get("validation_error") or item.get("reason") or ""
            ).strip().lower()
            for marker in _STEP_ROUTE_BLOCK_PATTERNS
        )
    ]
    if step_route_blocks:
        return "plan", "plan", "change_evidence_route", [
            "step_route_not_allowed:"
            + str(item.get("tool") or "unknown")
            for item in step_route_blocks
        ]
    source_unavailable = any(
        item.get("tool") == "read_function_chunk"
        and str(item.get("status") or "").strip().lower() == "source_unavailable"
        for item in attempted_tools
    )
    if source_unavailable:
        available_fallback_views = [
            view
            for view in requested_views
            if bool((packet_view_summary.get(view) or {}).get("available"))
        ]
        absent_fallback_views = [
            view for view in requested_views if view not in available_fallback_views
        ]
        fallback_attempts = [
            item for item in attempted_tools if item.get("tool") == "read_packet_view"
        ]
        failed_fallback_statuses = sorted({
            str(item.get("status") or "").strip().lower()
            for item in fallback_attempts
            if str(item.get("status") or "").strip().lower() in _FAILED_TOOL_STATUSES
        })
        if failed_fallback_statuses or ignored_requests:
            basis = ["read_function_chunk:source_unavailable"]
            basis.extend(
                f"fallback_tool_attempt_failed:{item}"
                for item in failed_fallback_statuses
            )
            if ignored_requests:
                basis.append("runtime_ignored_requested_fallback")
            return "runtime", "runtime", "engineering_runtime", basis

        attempted_fallback_views = {
            str(dict(item.get("args") or {}).get("view") or dict(item.get("args") or {}).get("view_name") or "")
            for item in fallback_attempts
        }
        successfully_read_fallback_views = {
            str(dict(item.get("args") or {}).get("view") or dict(item.get("args") or {}).get("view_name") or "")
            for item in fallback_attempts
            if str(item.get("status") or "").strip().lower() == "ok"
        }
        untried_fallback_views = [
            view for view in available_fallback_views
            if view not in attempted_fallback_views
        ]
        if untried_fallback_views:
            return "plan", "plan", "change_evidence_route", [
                "read_function_chunk:source_unavailable",
                *(
                    f"available_fallback_not_triggered:{view}"
                    for view in untried_fallback_views
                ),
            ]
        if available_fallback_views:
            if set(available_fallback_views).issubset(
                successfully_read_fallback_views
            ):
                return "unavailable", "none", "none", [
                    "read_function_chunk:source_unavailable",
                    "available_fallback_succeeded_but_semantic_fact_remained_missing",
                ]
            return "plan", "plan", "change_evidence_route", [
                "read_function_chunk:source_unavailable",
                "available_fallback_not_successfully_consumed",
            ]
        if absent_fallback_views:
            return "packet", "packet", "engineering_packet", [
                "read_function_chunk:source_unavailable",
                *(f"packet_view_absent:{view}" for view in absent_fallback_views),
            ]
        return "unavailable", "none", "none", ["read_function_chunk:source_unavailable"]
    failed = sorted(statuses & _FAILED_TOOL_STATUSES)
    if failed or ignored_requests:
        basis = [f"tool_attempt_failed:{item}" for item in failed]
        if ignored_requests:
            basis.append("runtime_ignored_requested_tool")
        return "runtime", "runtime", "engineering_runtime", basis

    absent_views = [
        view
        for view in requested_views
        if not bool((packet_view_summary.get(view) or {}).get("available"))
    ]
    if category == "packet_trace_truncation" or absent_views:
        basis = [f"packet_view_absent:{view}" for view in absent_views]
        if category == "packet_trace_truncation":
            basis.append("packet_trace_truncated")
        return "packet", "packet", "engineering_packet", basis

    successful_tools = {
        str(item.get("tool") or "")
        for item in attempted_tools
        if str(item.get("status") or "").strip().lower() == "ok"
    }
    if category == "prompt_render_truncation":
        return "plan", "plan", "change_evidence_route", ["packet_evidence_exists_but_prompt_route_truncated"]
    if requested_views:
        if "read_packet_view" not in successful_tools:
            return "plan", "plan", "change_evidence_route", [
                "requested_packet_view_available_but_not_successfully_read"
            ]
        return "unavailable", "none", "none", [
            "requested_packet_view_read_but_missing_semantic_fact_remained"
        ]
    if category == "source_tool":
        if "read_function_chunk" not in successful_tools:
            return "plan", "plan", "change_evidence_route", [
                "source_evidence_needed_but_source_tool_not_successfully_triggered"
            ]
        return "unavailable", "none", "none", [
            "source_tool_succeeded_but_requested_semantic_fact_was_not_observed"
        ]
    return "unknown", "none", "none", ["runtime_facts_do_not_identify_owner"]


def _routes_for_root_cause(
    root: Dict[str, Any],
    *,
    review: Dict[str, Any],
    case: Dict[str, Any],
    missing_origins: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    category = str(root.get("category") or "unknown").strip().lower()
    affected = _condition_ids(root.get("affected_conditions"))
    if category == "bad_judge_question":
        return [_route_bad_judge_question(root, review, case, affected)]
    if category == "missing_evidence":
        matching = [
            item
            for item in missing_origins
            if not affected or str(item.get("condition_id") or "").upper() in affected
        ]
        if not matching:
            return [_make_route(
                category=category,
                owner="none",
                operation="none",
                affected_conditions=affected,
                fact_basis=["missing_evidence_origin_not_recorded"],
                unreachable_reason="runtime_origin_unknown",
            )]
        routes = [
            _make_route(
                category=category,
                owner=str(item.get("owner") or "none"),
                operation=str(item.get("requested_operation") or "none"),
                affected_conditions=[item.get("condition_id")],
                fact_basis=list(item.get("fact_basis") or []),
                failure_origin=str(item.get("origin") or "unknown"),
                unreachable_reason=(
                    "evidence_objectively_unavailable"
                    if item.get("owner") == "none"
                    else ""
                ),
            )
            for item in matching
        ]
        semantic_impact = str(root.get("semantic_impact") or "uncertain").strip().lower()
        for route in routes:
            route["semantic_impact"] = semantic_impact
            if route.get("owner") == "plan" and semantic_impact == "non_blocking":
                route["route_status"] = "unreachable"
                route["unreachable_reason"] = "missing_fact_not_semantically_blocking"
                route["mutable_surface"] = ""
        return routes
    if category in _RULE_CATEGORIES:
        return [_make_route(category, "rule", "semantic_rewrite", affected)]
    if category == "bad_packet_view_selection":
        return [_make_route(category, "plan", "change_evidence_route", affected)]
    if category == "missing_packet_view":
        return [_route_missing_packet_view(category, affected, missing_origins)]
    if category == "packet_build_or_load_failure":
        return [_make_route(category, "packet", "engineering_packet", affected)]
    if category == "packet_evidence_too_coarse":
        return [_route_missing_packet_view(category, affected, missing_origins)]
    if category in {"runtime_emit_logic_bug", "binding_runtime_failure"}:
        return [_make_route(category, "runtime", "engineering_runtime", affected)]
    if category == "binding_route_failure":
        return [_make_route(category, "plan", "change_dependency_route", affected)]
    if category == "deferred_logic_structure_failure":
        return [_make_route(
            category,
            "none",
            "none",
            affected,
            route_status="deferred",
            unreachable_reason="current_and_not_logic_cannot_express_alternate_path",
        )]
    return [_make_route(
        category,
        "none",
        "none",
        affected,
        unreachable_reason="diagnosis_has_no_registered_mutable_surface",
    )]


def _route_bad_judge_question(
    root: Dict[str, Any],
    review: Dict[str, Any],
    case: Dict[str, Any],
    affected: List[str],
) -> Dict[str, Any]:
    origin = str(
        root.get("failure_origin")
        or root.get("question_failure_kind")
        or ""
    ).strip().lower()
    operation = str(root.get("requested_operation") or "").strip().lower()
    semantic_status = str(root.get("rule_semantic_status") or "").strip().lower()
    alignment_status = str(root.get("question_alignment_status") or "").strip().lower()
    diagnoses = [
        item
        for item in list(review.get("condition_diagnosis") or [])
        if isinstance(item, dict)
        and (
            not affected
            or str(item.get("condition_id") or "").strip().upper() in affected
        )
    ]
    if origin in {"rule", "rule_semantic", "semantic"} or any(
        bool(item.get("is_rule_problem")) for item in diagnoses
    ):
        return _make_route(
            "bad_judge_question", "rule", "semantic_rewrite", affected,
            failure_origin="rule_semantic",
        )
    if (
        origin in {"plan_question_drift", "judge_question_drift", "question_drift"}
        or operation == "restore_question_from_rule"
        or (semantic_status == "correct" and alignment_status == "drifted")
    ):
        differs = _question_differs_from_rule(case, affected)
        return _make_route(
            "bad_judge_question",
            "plan",
            "restore_question_from_rule",
            affected,
            failure_origin="plan_question_drift",
            fact_basis=["reviewer_rule_semantic_correct", "plan_question_diff_checked"],
            unreachable_reason=("question_already_matches_rule" if not differs else ""),
        )
    if (
        origin in {"evidence_route", "plan_evidence_route", "plan"}
        or operation == "change_evidence_route"
        or _plan_patch_changes_evidence(review)
    ):
        return _make_route(
            "bad_judge_question",
            "plan",
            "change_evidence_route",
            affected,
            failure_origin="evidence_route",
        )
    return _make_route(
        "bad_judge_question",
        "none",
        "none",
        affected,
        failure_origin="ambiguous",
        unreachable_reason="bad_judge_question_subtype_not_established",
    )


def _route_missing_packet_view(
    category: str,
    affected: List[str],
    missing_origins: List[Dict[str, Any]],
) -> Dict[str, Any]:
    matches = [
        item
        for item in missing_origins
        if not affected or str(item.get("condition_id") or "").upper() in affected
    ]
    owners = {str(item.get("owner") or "none") for item in matches}
    if "packet" in owners:
        packet_matches = [item for item in matches if item.get("owner") == "packet"]
        return _make_route(
            category,
            "packet",
            "engineering_packet",
            affected,
            failure_origin="packet",
            fact_basis=[
                str(value)
                for item in packet_matches
                for value in list(item.get("fact_basis") or [])
                if str(value)
            ],
        )
    if "plan" in owners:
        plan_matches = [item for item in matches if item.get("owner") == "plan"]
        return _make_route(
            category,
            "plan",
            "change_evidence_route",
            affected,
            failure_origin="plan",
            fact_basis=[
                str(value)
                for item in plan_matches
                for value in list(item.get("fact_basis") or [])
                if str(value)
            ],
        )
    unavailable_basis = [
        str(value)
        for item in matches
        for value in list(item.get("fact_basis") or [])
        if str(value)
    ]
    return _make_route(
        category,
        "none",
        "none",
        affected,
        failure_origin="unavailable" if matches else "unknown",
        fact_basis=unavailable_basis,
        unreachable_reason="packet_or_plan_origin_not_established",
    )


def _make_route(
    category: str,
    owner: str,
    operation: str,
    affected_conditions: Iterable[Any],
    *,
    failure_origin: str = "",
    fact_basis: Optional[List[str]] = None,
    route_status: str = "",
    unreachable_reason: str = "",
) -> Dict[str, Any]:
    target = str(owner or "none").strip().lower()
    requested = str(operation or "none").strip().lower()
    reachable = requested in OWNER_MUTABLE_SURFACES.get(target, set())
    status = route_status or ("reachable" if reachable else "unreachable")
    if unreachable_reason and status != "deferred":
        status = "unreachable"
    return {
        "category": str(category or "unknown").strip().lower(),
        "failure_origin": failure_origin or target,
        "owner": target,
        "requested_operation": requested,
        "mutable_surface": requested if reachable else "",
        "affected_conditions": _condition_ids(affected_conditions),
        "route_status": status,
        "unreachable_reason": unreachable_reason,
        "fact_basis": [str(item) for item in list(fact_basis or []) if str(item)],
    }


def _operation_for_signal(signal: Dict[str, Any], review: Dict[str, Any]) -> str:
    target = str(signal.get("update_target") or "").strip().lower()
    direction = str(signal.get("direction") or "").strip().lower()
    if target == "rule":
        action = str(dict(review.get("rule_patch_suggestion") or {}).get("action") or "")
        return {
            "add_condition": "add_positive_condition",
            "add_exclusion": "add_exclusion",
            "narrow_exclusion": "narrow_exclusion",
            "remove_condition": "remove_positive_condition",
            "remove_exclusion": "remove_exclusion",
        }.get(action, "semantic_rewrite")
    if target == "plan":
        if "depend" in direction:
            return "change_dependency_route"
        if direction in {
            "change_evidence_strategy",
            "change_default_views",
            "change_followup_views",
        }:
            return "change_evidence_route"
        return "none"
    if target == "packet":
        return "engineering_packet"
    if target == "runtime":
        return "engineering_runtime"
    return "none"


def _requested_views(
    text: str,
    row: Dict[str, Any],
    packet_view_summary: Dict[str, Dict[str, Any]],
) -> List[str]:
    lower = str(text or "").lower()
    views = [
        name
        for name in packet_view_summary
        if name.lower() in lower or name.lower().replace("_", " ") in lower
    ]
    for request in list(row.get("tool_requests") or []):
        if not isinstance(request, dict):
            continue
        args = request.get("args") if isinstance(request.get("args"), dict) else request
        view = str((args or {}).get("view") or (args or {}).get("view_name") or "")
        if view in packet_view_summary and view not in views:
            views.append(view)
    return views


def _attempted_tools(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in list(row.get("tool_observations") or []):
        if not isinstance(item, dict):
            continue
        out.append({
            "tool": str(item.get("tool") or ""),
            "status": str(item.get("tool_status") or "").strip().lower(),
            "args": dict(item.get("args") or {}) if isinstance(item.get("args"), dict) else {},
            "validation_error": str(item.get("validation_error") or ""),
            "reason": str(item.get("reason") or ""),
        })
    return out


def _relevant_attempted_tools(
    category: str,
    requested_views: List[str],
    attempted_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if category == "source_tool":
        return [
            item
            for item in attempted_tools
            if item.get("tool") == "read_function_chunk"
            or (
                item.get("tool") == "read_packet_view"
                and str(
                    dict(item.get("args") or {}).get("view")
                    or dict(item.get("args") or {}).get("view_name")
                    or ""
                ) in requested_views
            )
        ]
    if requested_views:
        relevant = []
        for item in attempted_tools:
            if item.get("tool") != "read_packet_view":
                continue
            args = dict(item.get("args") or {})
            view = str(args.get("view") or args.get("view_name") or "")
            if not view or view in requested_views:
                relevant.append(item)
        return relevant
    return attempted_tools


def _plan_patch_changes_evidence(review: Dict[str, Any]) -> bool:
    action = str(dict(review.get("plan_patch_suggestion") or {}).get("action") or "").lower()
    return action in {
        "change_default_views",
        "change_followup_views",
        "change_evidence_route",
        "change_dependencies",
    }


def _condition_has_bad_question_root(
    root_causes: List[Dict[str, Any]],
    condition_id: Any,
) -> bool:
    target = str(condition_id or "").strip().upper()
    for root in root_causes:
        if str(root.get("category") or "").strip().lower() != "bad_judge_question":
            continue
        affected = _condition_ids(root.get("affected_conditions"))
        if not target or not affected or target in affected:
            return True
    return False


def _question_differs_from_rule(case: Dict[str, Any], affected: List[str]) -> bool:
    rule = dict(case.get("rule_digest") or {})
    plan = dict(case.get("plan_digest") or {})
    attack_label = str(dict(rule.get("metadata") or {}).get("attack_label") or "")
    descriptions = {
        str(item.get("id") or "").strip().upper(): str(item.get("description") or "")
        for item in [
            *list(rule.get("conditions") or []),
            *list(rule.get("exclusion_conditions") or []),
        ]
        if isinstance(item, dict)
    }
    questions = {
        str(item.get("condition_id") or item.get("id") or "").strip().upper(): str(item.get("question") or "")
        for item in list(plan.get("judge_steps") or [])
        if isinstance(item, dict)
    }
    target_ids = affected or sorted(set(descriptions) & set(questions))
    for condition_id in target_ids:
        description = descriptions.get(condition_id)
        if not description or condition_id not in questions:
            continue
        canonical = make_local_judge_question(description, attack_label=attack_label)
        if _normalize_text(questions[condition_id]) != _normalize_text(canonical):
            return True
    return False


def _condition_ids(values: Any) -> List[str]:
    if isinstance(values, str):
        values = re.findall(r"\b[CE]\d+\b", values.upper()) or [values]
    out: List[str] = []
    for value in list(values or []):
        item = str(value or "").strip().upper()
        if re.fullmatch(r"[CE]\d+", item) and item not in out:
            out.append(item)
    return out


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _dedupe_routes(routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for route in routes:
        key = (
            route.get("source"),
            route.get("source_index"),
            route.get("owner"),
            route.get("requested_operation"),
            tuple(route.get("affected_conditions") or []),
            route.get("failure_origin"),
        )
        if key in seen:
            continue
        seen.add(key)
        route = dict(route)
        route["route_id"] = f"route_{len(out) + 1:03d}"
        out.append(route)
    return out


def _dedupe_dicts(
    rows: List[Dict[str, Any]], *, keys: tuple[str, ...]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
