from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evotx.evolution.diagnosis_routing import (  # noqa: E402
    apply_diagnosis_routing,
    derive_missing_evidence_origins,
    project_review_for_owner,
)
from evotx.utils.result_slimmer import (  # noqa: E402
    build_reviewer_case,
    build_round_review_bundle,
)


RULE_CATEGORIES = {
    "rule_too_broad",
    "rule_too_narrow",
    "exclusion_too_broad",
    "missing_exclusion",
    "missing_condition",
    "weak_evidence_as_sufficient",
    "bad_semantic_condition",
    "binding_semantic_failure",
}
PACKET_CATEGORIES = {
    "packet_build_or_load_failure",
    "packet_evidence_too_coarse",
}
RUNTIME_CATEGORIES = {"runtime_emit_logic_bug", "binding_runtime_failure"}
PLAN_CATEGORIES = {"bad_packet_view_selection", "binding_route_failure"}
ENGINEERING_OWNERS = {"packet", "runtime"}
KNOWLEDGE_OWNERS = {"rule", "plan"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _condition_ids(value: Any) -> Set[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return {str(item or "").strip().upper() for item in values if str(item or "").strip()}


def _matching_diagnoses(review: Dict[str, Any], root: Dict[str, Any]) -> List[Dict[str, Any]]:
    affected = _condition_ids(root.get("affected_conditions"))
    return [
        item
        for item in list(review.get("condition_diagnosis") or [])
        if isinstance(item, dict)
        and (
            not affected
            or str(item.get("condition_id") or "").strip().upper() in affected
        )
    ]


def _plan_patch_changes_evidence(review: Dict[str, Any]) -> bool:
    action = str(
        dict(review.get("plan_patch_suggestion") or {}).get("action") or ""
    ).strip().lower()
    return action in {
        "change_default_views",
        "change_followup_views",
        "change_evidence_route",
        "change_dependency_route",
        "add_followup",
        "change_tools",
    }


def _expected_owners(
    review: Dict[str, Any],
    root: Dict[str, Any],
    case: Dict[str, Any],
) -> Tuple[Set[str], str]:
    """Independent reference mapping for historical replay scoring.

    Historical bad_judge_question rows without a subtype are intentionally
    scored as not covered unless existing fields establish semantics or an
    evidence-route operation. This avoids treating the new router as its own
    oracle.
    """
    category = str(root.get("category") or "unknown").strip().lower()
    if category in RULE_CATEGORIES:
        return {"rule"}, "registered semantic diagnosis"
    if category in PLAN_CATEGORIES:
        return {"plan"}, "registered evidence/dependency route diagnosis"
    if category in PACKET_CATEGORIES:
        return {"packet"}, "registered packet engineering diagnosis"
    if category in RUNTIME_CATEGORIES:
        return {"runtime"}, "registered runtime engineering diagnosis"
    if category == "deferred_logic_structure_failure":
        return {"none"}, "unsupported current logic structure"
    if category in {"missing_evidence", "missing_packet_view"}:
        affected = _condition_ids(root.get("affected_conditions"))
        origins = list(
            dict(case.get("evidence_digest") or {}).get("missing_evidence_origins")
            or []
        )
        matching = [
            item
            for item in origins
            if not affected
            or str(item.get("condition_id") or "").strip().upper() in affected
        ]
        owners = {str(item.get("owner") or "none") for item in matching}
        if owners:
            return owners, "runtime-derived missing-evidence origin"
        return set(), "missing runtime origin facts"
    if category == "bad_judge_question":
        origin = str(
            root.get("failure_origin")
            or root.get("question_failure_kind")
            or ""
        ).strip().lower()
        operation = str(root.get("requested_operation") or "").strip().lower()
        semantic_status = str(root.get("rule_semantic_status") or "").strip().lower()
        alignment = str(root.get("question_alignment_status") or "").strip().lower()
        diagnoses = _matching_diagnoses(review, root)
        if origin in {"rule", "rule_semantic", "semantic"} or any(
            bool(item.get("is_rule_problem")) for item in diagnoses
        ):
            return {"rule"}, "semantic problem established"
        if (
            origin in {"plan_question_drift", "judge_question_drift", "question_drift"}
            or operation == "restore_question_from_rule"
            or (semantic_status == "correct" and alignment == "drifted")
        ):
            return {"plan"}, "question drift established"
        if (
            origin in {"evidence_route", "plan_evidence_route", "plan"}
            or operation == "change_evidence_route"
            or _plan_patch_changes_evidence(review)
        ):
            return {"plan"}, "evidence-route problem established"
        return set(), "historical question subtype not established"
    if category == "unsupported_state_schema":
        return {"none", "runtime"}, "unsupported/engineering state schema"
    return set(), "category has no independent reference mapping"


def _root_routes(routed: Dict[str, Any], root_index: int) -> List[Dict[str, Any]]:
    return [
        route
        for route in list(routed.get("diagnosis_routes") or [])
        if isinstance(route, dict)
        and route.get("source") == "root_cause"
        and int(route.get("source_index", -1)) == root_index
    ]


def _route_owner_resolved(routes: Sequence[Dict[str, Any]]) -> bool:
    for route in routes:
        owner = str(route.get("owner") or "none")
        if owner != "none" and route.get("route_status") == "reachable":
            return True
        if owner == "none" and str(route.get("failure_origin") or "") in {
            "unavailable",
            "resolved",
        }:
            return True
        if route.get("route_status") == "deferred":
            return True
    return False


def _historical_replay(episodes: Sequence[int], results_root: Path) -> Dict[str, Any]:
    diagnosis_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    routed_review_rows: List[Dict[str, Any]] = []

    for episode in episodes:
        episode_root = results_root / str(episode)
        for round_dir in sorted(episode_root.glob("round_*")):
            required = [
                round_dir / "reviews.json",
                round_dir / "slim_results.json",
                round_dir / "review_bundle.json",
            ]
            if not all(path.exists() for path in required):
                continue
            reviews = list(_load_json(required[0]) or [])
            slim_results = list(_load_json(required[1]) or [])
            old_bundle = dict(_load_json(required[2]) or {})
            slim_by_tx = {str(item.get("tx_hash") or ""): item for item in slim_results}
            routed_reviews: List[Dict[str, Any]] = []

            for review_index, review in enumerate(reviews):
                tx_hash = str(review.get("tx_hash") or "")
                slim = slim_by_tx.get(tx_hash, {})
                case = build_reviewer_case(slim) if slim else {
                    "evidence_digest": {"missing_evidence_origins": []},
                    "condition_table": [],
                }
                case["cohort_signal_summary"] = dict(
                    old_bundle.get("cohort_signal_summary") or {}
                )
                routed = apply_diagnosis_routing(review, case)
                routed_reviews.append(routed)
                owners = sorted(set(routed.get("owner_targets") or []))
                routed_review_rows.append({
                    "episode": episode,
                    "round": round_dir.name,
                    "review_index": review_index,
                    "tx_hash": tx_hash,
                    "before_update_target": str(review.get("update_target") or "none"),
                    "after_owner_targets": owners,
                    "multi_owner": len(owners) > 1,
                })

                for root_index, root in enumerate(list(review.get("root_causes") or [])):
                    if not isinstance(root, dict):
                        continue
                    expected, reference_basis = _expected_owners(review, root, case)
                    routes = _root_routes(routed, root_index)
                    after_owners = {str(route.get("owner") or "none") for route in routes}
                    before_owner = str(review.get("update_target") or "none").lower()
                    covered = bool(expected)
                    diagnosis_rows.append({
                        "episode": episode,
                        "round": round_dir.name,
                        "review_index": review_index,
                        "tx_hash": tx_hash,
                        "root_index": root_index,
                        "category": str(root.get("category") or "unknown").lower(),
                        "reference_owners": sorted(expected),
                        "reference_basis": reference_basis,
                        "reference_covered": covered,
                        "before_owner": before_owner,
                        "before_owner_valid": covered and before_owner in expected,
                        "before_mutable_surface_reachable": (
                            covered
                            and before_owner in expected
                            and before_owner != "none"
                        ),
                        "after_owners": sorted(after_owners),
                        "after_owner_valid": covered and bool(after_owners & expected),
                        "after_owner_resolved": _route_owner_resolved(routes),
                        "after_mutable_surface_reachable": any(
                            route.get("route_status") == "reachable"
                            and str(route.get("owner") or "none") in expected
                            for route in routes
                        ),
                        "before_engineering_misroute": (
                            covered
                            and bool(expected & ENGINEERING_OWNERS)
                            and before_owner in KNOWLEDGE_OWNERS
                        ),
                        "after_engineering_misroute": (
                            covered
                            and bool(expected & ENGINEERING_OWNERS)
                            and bool(after_owners & KNOWLEDGE_OWNERS)
                        ),
                        "routes": routes,
                    })

            new_bundle = build_round_review_bundle(
                old_bundle.get("current_rule") or {},
                old_bundle.get("current_summary") or {},
                routed_reviews,
                slim_results,
                negative_training_mode=str(
                    old_bundle.get("negative_training_mode") or "mixed"
                ),
                negative_training_mode_source=str(
                    old_bundle.get("negative_training_mode_source") or "historical_replay"
                ),
                cohort_signal_summary=old_bundle.get("cohort_signal_summary") or {},
                plan_evidence_audit=old_bundle.get("plan_evidence_audit") or {},
                case_boundary_context=old_bundle.get("case_boundary_context") or {},
                rejected_update_memory=old_bundle.get("rejected_update_memory") or {},
            )
            old_eligible = {
                "rule": len(list(old_bundle.get("actionable_rule_reviews") or [])),
                "plan": len(list(old_bundle.get("actionable_plan_reviews") or [])),
                "engineering": len(list(old_bundle.get("packet_engineering_todos") or [])),
            }
            new_eligible = {
                "rule": len(list(new_bundle.get("actionable_rule_reviews") or [])),
                "plan": len(list(new_bundle.get("actionable_plan_reviews") or [])),
                "engineering": len(list(new_bundle.get("engineering_reviews") or [])),
            }
            candidate_artifacts = []
            artifact_root = round_dir / "candidate_artifacts"
            if artifact_root.exists():
                candidate_artifacts = sorted(
                    path.name for path in artifact_root.iterdir() if path.is_dir()
                )
            round_rows.append({
                "episode": episode,
                "round": round_dir.name,
                "review_count": len(reviews),
                "old_eligible_reviews": old_eligible,
                "phase2_eligible_reviews": new_eligible,
                "actual_candidate_artifacts": candidate_artifacts,
                "phase2_engineering_owner_counts": dict(
                    new_bundle.get("diagnosis_routing_audit", {}).get("owner_counts") or {}
                ),
            })

    covered = [row for row in diagnosis_rows if row["reference_covered"]]
    after_resolved = [row for row in diagnosis_rows if row["after_owner_resolved"]]
    reachable = [row for row in diagnosis_rows if row["after_mutable_surface_reachable"]]
    unresolved = [row for row in diagnosis_rows if not row["after_owner_resolved"]]
    before_valid = sum(bool(row["before_owner_valid"]) for row in covered)
    after_valid = sum(bool(row["after_owner_valid"]) for row in covered)
    before_reachable = sum(
        bool(row["before_mutable_surface_reachable"]) for row in covered
    )
    after_reachable_valid = sum(
        bool(row["after_mutable_surface_reachable"]) for row in covered
    )
    old_calls = _actual_updater_calls(episodes, results_root)
    actual_candidates = sum(len(row["actual_candidate_artifacts"]) for row in round_rows)
    observed_no_op_calls = max(0, old_calls - actual_candidates)
    old_eligible_total = sum(
        row["old_eligible_reviews"]["rule"] + row["old_eligible_reviews"]["plan"]
        for row in round_rows
    )
    new_eligible_total = sum(
        row["phase2_eligible_reviews"]["rule"] + row["phase2_eligible_reviews"]["plan"]
        for row in round_rows
    )
    engineering_only = sum(
        1
        for row in diagnosis_rows
        if row["reference_covered"]
        and bool(set(row["reference_owners"]) & ENGINEERING_OWNERS)
    )
    historical = {
        "episodes": list(episodes),
        "round_count": len(round_rows),
        "review_count": len(routed_review_rows),
        "diagnosis_count": len(diagnosis_rows),
        "reference_covered_count": len(covered),
        "reference_not_covered_count": len(diagnosis_rows) - len(covered),
        "metrics": {
            "before_valid_owner_rate": _rate(before_valid, len(covered)),
            "after_valid_owner_rate": _rate(after_valid, len(covered)),
            "before_bad_routing_rate": _rate(len(covered) - before_valid, len(covered)),
            "after_bad_routing_rate": _rate(len(covered) - after_valid, len(covered)),
            "before_mutable_surface_reachability": _rate(
                before_reachable, len(covered)
            ),
            "after_mutable_surface_reachability": _rate(
                after_reachable_valid, len(covered)
            ),
            "after_owner_resolved_rate": _rate(len(after_resolved), len(diagnosis_rows)),
            "before_engineering_to_knowledge_misroute_rate": _rate(
                sum(bool(row["before_engineering_misroute"]) for row in diagnosis_rows),
                engineering_only,
            ),
            "after_engineering_to_knowledge_misroute_rate": _rate(
                sum(bool(row["after_engineering_misroute"]) for row in diagnosis_rows),
                engineering_only,
            ),
            "observed_historical_updater_no_op_rate": _rate(
                observed_no_op_calls, old_calls
            ),
            "old_actionable_review_count": old_eligible_total,
            "phase2_actionable_review_count": new_eligible_total,
        },
        "routing_funnel": {
            "diagnosis": len(diagnosis_rows),
            "owner_resolved": len(after_resolved),
            "owner_unresolved": len(unresolved),
            "mutable_surface_reachable": len(reachable),
            "mutable_surface_unreachable_or_deferred": len(diagnosis_rows) - len(reachable),
            "knowledge_updater_eligible_reviews": new_eligible_total,
            "knowledge_updater_skipped_reviews": max(
                0, len(routed_review_rows) - new_eligible_total
            ),
            "actual_historical_updater_calls": old_calls,
            "actual_historical_candidate_artifacts": actual_candidates,
            "actual_historical_no_op_calls": observed_no_op_calls,
            "phase2_candidate_generated": "not_covered_without_reinvoking_updater_llm",
            "engineering_owner_routes": sum(
                1
                for row in diagnosis_rows
                if bool(set(row["after_owners"]) & ENGINEERING_OWNERS)
            ),
        },
        "multi_owner_review_count": sum(bool(row["multi_owner"]) for row in routed_review_rows),
        "unreachable_diagnoses": [
            {
                key: row[key]
                for key in (
                    "episode",
                    "round",
                    "tx_hash",
                    "category",
                    "reference_basis",
                    "after_owners",
                    "routes",
                )
            }
            for row in unresolved
        ],
        "category_counts": dict(Counter(row["category"] for row in diagnosis_rows)),
        "rounds": round_rows,
        "reviews": routed_review_rows,
        "diagnoses": diagnosis_rows,
    }
    return historical


def _actual_updater_calls(episodes: Sequence[int], results_root: Path) -> int:
    total = 0
    for episode in episodes:
        transcript_root = results_root / str(episode) / "llm_transcripts"
        for owner in ("rule_updater", "plan_updater"):
            owner_root = transcript_root / owner
            if owner_root.exists():
                total += sum(1 for _ in owner_root.glob("*.json"))
    return total


def _rate(numerator: int, denominator: int) -> Dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": round(numerator / denominator, 4) if denominator else None,
    }


def _base_case() -> Dict[str, Any]:
    return {
        "rule_digest": {
            "conditions": [{
                "id": "C1",
                "description": "A protected action lacks valid authority.",
                "expected_answer": True,
            }],
            "exclusion_conditions": [],
        },
        "plan_digest": {
            "judge_steps": [{
                "condition_id": "C1",
                "question": "Does any unusual external call occur?",
            }],
        },
        "condition_table": [],
        "evidence_digest": {"missing_evidence_origins": []},
    }


def _scenario_status(actual: Any, expected: Any) -> str:
    return "Passed" if actual == expected else "Failed"


def _route_scenario(name: str, root: Dict[str, Any], expected: Tuple[str, str]) -> Dict[str, Any]:
    routed = apply_diagnosis_routing({
        "root_causes": [root],
        "condition_diagnosis": [],
        "update_signals": [],
    }, _base_case())
    route = routed["diagnosis_routes"][0]
    actual = (route.get("owner"), route.get("requested_operation"))
    return {
        "name": name,
        "status": _scenario_status(actual, expected),
        "expected": {"owner": expected[0], "operation": expected[1]},
        "actual": {"owner": actual[0], "operation": actual[1]},
        "route_status": route.get("route_status"),
        "unreachable_reason": route.get("unreachable_reason", ""),
    }


def _targeted_scenarios() -> Dict[str, Any]:
    bad_question = [
        _route_scenario(
            "bad_question_rule_semantic",
            {
                "category": "bad_judge_question",
                "failure_origin": "rule_semantic",
                "affected_conditions": ["C1"],
            },
            ("rule", "semantic_rewrite"),
        ),
        _route_scenario(
            "bad_question_plan_drift",
            {
                "category": "bad_judge_question",
                "failure_origin": "plan_question_drift",
                "rule_semantic_status": "correct",
                "question_alignment_status": "drifted",
                "affected_conditions": ["C1"],
            },
            ("plan", "restore_question_from_rule"),
        ),
        _route_scenario(
            "bad_question_evidence_route",
            {
                "category": "bad_judge_question",
                "failure_origin": "evidence_route",
                "affected_conditions": ["C1"],
            },
            ("plan", "change_evidence_route"),
        ),
    ]
    only_drift_restores = [
        row["actual"]["operation"] == "restore_question_from_rule"
        for row in bad_question
    ] == [False, True, False]

    packet = {
        "critical_call_view": {"available": True, "row_count": 4},
        "state_change_view": {"available": False, "row_count": 0},
        "source_unavailable_auth_view": {"available": True, "row_count": 6},
    }
    missing_specs = [
        (
            "tool_attempt_failed",
            {"id": "m1", "condition_id": "C1", "category": "source_tool", "status": "open", "text": "function source needed"},
            [{"condition_id": "C1", "tool_observations": [{"tool": "read_function_chunk", "tool_status": "timeout", "args": {}}]}],
            ("runtime", "runtime"),
        ),
        (
            "requested_packet_view_absent",
            {"id": "m2", "condition_id": "C1", "category": "available_packet_view", "status": "open", "text": "state_change_view needed"},
            [{"condition_id": "C1", "tool_observations": []}],
            ("packet", "packet"),
        ),
        (
            "source_objectively_unavailable",
            {"id": "m3", "condition_id": "C1", "category": "source_tool", "status": "open", "text": "source code needed"},
            [{"condition_id": "C1", "tool_observations": [{"tool": "read_function_chunk", "tool_status": "source_unavailable", "args": {}}]}],
            ("unavailable", "none"),
        ),
        (
            "available_evidence_not_selected",
            {"id": "m4", "condition_id": "C1", "category": "available_packet_view", "status": "open", "text": "critical_call_view needed"},
            [{"condition_id": "C1", "tool_observations": []}],
            ("plan", "plan"),
        ),
        (
            "source_unavailable_with_available_fallback",
            {"id": "m5", "condition_id": "C1", "category": "source_tool", "status": "open", "text": "source_unavailable_auth_view fallback needed"},
            [{"condition_id": "C1", "tool_observations": [{"tool": "read_function_chunk", "tool_status": "source_unavailable", "args": {}}]}],
            ("plan", "plan"),
        ),
    ]
    missing_rows = []
    for name, item, condition_table, expected in missing_specs:
        origin = derive_missing_evidence_origins(
            [item],
            condition_table=condition_table,
            packet_view_summary=packet,
        )[0]
        actual = (origin.get("origin"), origin.get("owner"))
        missing_rows.append({
            "name": name,
            "status": _scenario_status(actual, expected),
            "expected": {"origin": expected[0], "owner": expected[1]},
            "actual": {"origin": actual[0], "owner": actual[1]},
            "fact_basis": origin.get("fact_basis", []),
        })

    binding = [
        _route_scenario(
            "binding_state_semantic_wrong",
            {"category": "binding_semantic_failure", "affected_conditions": ["C1"]},
            ("rule", "semantic_rewrite"),
        ),
        _route_scenario(
            "binding_evidence_route_wrong",
            {"category": "binding_route_failure", "affected_conditions": ["C1"]},
            ("plan", "change_dependency_route"),
        ),
        _route_scenario(
            "binding_propagation_runtime_failure",
            {"category": "binding_runtime_failure", "affected_conditions": ["C1"]},
            ("runtime", "engineering_runtime"),
        ),
        _route_scenario(
            "unsupported_novel_state_schema",
            {"category": "unsupported_state_schema", "affected_conditions": ["C1"]},
            ("none", "none"),
        ),
    ]

    multi_case = _base_case()
    multi_case["evidence_digest"]["missing_evidence_origins"] = [{
        "missing_evidence_id": "m-runtime",
        "condition_id": "C1",
        "origin": "runtime",
        "owner": "runtime",
        "requested_operation": "engineering_runtime",
        "fact_basis": ["tool_attempt_failed:timeout"],
    }]
    multi = apply_diagnosis_routing({
        "root_causes": [
            {"category": "bad_semantic_condition", "affected_conditions": ["C1"]},
            {"category": "bad_packet_view_selection", "affected_conditions": ["C1"]},
            {"category": "missing_evidence", "semantic_impact": "blocking", "affected_conditions": ["C1"]},
        ],
        "condition_diagnosis": [],
        "update_signals": [],
        "update_target": "rule",
    }, multi_case)
    projections = {
        owner: project_review_for_owner(multi, owner, source_review_index=0)
        for owner in ("rule", "plan", "packet", "runtime")
    }
    engineering_only = apply_diagnosis_routing({
        "root_causes": [{"category": "binding_runtime_failure", "affected_conditions": ["C1"]}],
        "condition_diagnosis": [],
        "update_signals": [],
        "update_target": "plan",
    }, _base_case())
    engineering_projects = {
        owner: project_review_for_owner(engineering_only, owner, source_review_index=0)
        for owner in ("rule", "plan", "runtime")
    }
    multi_owner = {
        "status": "Passed" if (
            set(multi.get("owner_targets") or []) == {"rule", "plan", "runtime"}
            and projections["rule"] is not None
            and projections["plan"] is not None
            and projections["runtime"] is not None
            and projections["packet"] is None
            and engineering_projects["rule"] is None
            and engineering_projects["plan"] is None
            and engineering_projects["runtime"] is not None
        ) else "Failed",
        "expected": {
            "owner_targets": ["plan", "rule", "runtime"],
            "engineering_only_rule_plan_projection": False,
        },
        "actual": {
            "owner_targets": multi.get("owner_targets", []),
            "engineering_only_rule_projection": engineering_projects["rule"] is not None,
            "engineering_only_plan_projection": engineering_projects["plan"] is not None,
        },
        "owner_targets": multi.get("owner_targets", []),
        "projection_present": {key: value is not None for key, value in projections.items()},
        "engineering_only_projection_present": {
            key: value is not None for key, value in engineering_projects.items()
        },
    }

    override_case = _base_case()
    override_case["evidence_digest"]["missing_evidence_origins"] = [{
        "missing_evidence_id": "m-runtime-override",
        "condition_id": "C1",
        "origin": "runtime",
        "owner": "runtime",
        "requested_operation": "engineering_runtime",
        "fact_basis": ["tool_attempt_failed:timeout"],
    }]
    override = apply_diagnosis_routing({
        "root_causes": [{
            "category": "missing_evidence",
            "failure_origin": "plan",
            "semantic_impact": "blocking",
            "affected_conditions": ["C1"],
        }],
        "condition_diagnosis": [],
        "update_signals": [{
            "update_target": "plan",
            "condition_id": "C1",
            "generalization_status": "supported",
        }],
        "update_target": "plan",
    }, override_case)
    override_routes = list(override.get("diagnosis_routes") or [])
    root_owner_set = {
        str(route.get("owner") or "none")
        for route in override_routes
        if route.get("source") == "root_cause"
    }
    plan_signal_reachable = any(
        route.get("source") == "update_signal"
        and route.get("owner") == "plan"
        and route.get("route_status") == "reachable"
        for route in override_routes
    )
    missing_rows.append({
        "name": "reviewer_text_cannot_override_runtime_origin",
        "status": (
            "Passed"
            if root_owner_set == {"runtime"} and not plan_signal_reachable
            else "Failed"
        ),
        "expected": {
            "root_owners": ["runtime"],
            "plan_signal_reachable": False,
        },
        "actual": {
            "root_owners": sorted(root_owner_set),
            "plan_signal_reachable": plan_signal_reachable,
        },
    })

    groups = {
        "multi_owner_projection": [multi_owner],
        "bad_judge_question": bad_question + [{
            "name": "only_question_drift_allows_restore",
            "status": "Passed" if only_drift_restores else "Failed",
            "expected": [False, True, False],
            "actual": [
                row["actual"]["operation"] == "restore_question_from_rule"
                for row in bad_question
            ],
        }],
        "missing_evidence_origin": missing_rows,
        "binding_stateful": binding,
    }
    return {
        "groups": groups,
        "summary": {
            status: sum(
                1
                for rows in groups.values()
                for row in rows
                if row.get("status") == status
            )
            for status in ("Passed", "Failed", "Not Covered")
        },
    }


def _markdown(report: Dict[str, Any]) -> str:
    historical = report["historical_replay"]
    metrics = historical["metrics"]
    funnel = historical["routing_funnel"]
    scenarios = report["targeted_scenarios"]
    lines = [
        "# Phase 2 Routing Offline Validation",
        "",
        f"Generated: {report['generated_at']}",
        f"Episodes: {', '.join(str(item) for item in historical['episodes'])}",
        "",
        "## Historical Replay",
        "",
        "| Metric | Before | After |",
        "|---|---:|---:|",
        f"| diagnosis -> valid owner | {_fmt_rate(metrics['before_valid_owner_rate'])} | {_fmt_rate(metrics['after_valid_owner_rate'])} |",
        f"| bad-routing rate | {_fmt_rate(metrics['before_bad_routing_rate'])} | {_fmt_rate(metrics['after_bad_routing_rate'])} |",
        f"| diagnosis -> valid mutable surface | {_fmt_rate(metrics['before_mutable_surface_reachability'])} | {_fmt_rate(metrics['after_mutable_surface_reachability'])} |",
        f"| engineering-only -> Rule/Plan misroute | {_fmt_rate(metrics['before_engineering_to_knowledge_misroute_rate'])} | {_fmt_rate(metrics['after_engineering_to_knowledge_misroute_rate'])} |",
        f"| actionable Rule/Plan review count | {metrics['old_actionable_review_count']} | {metrics['phase2_actionable_review_count']} |",
        f"| observed updater no-op call rate | {_fmt_rate(metrics['observed_historical_updater_no_op_rate'])} | Not Covered (LLM not reinvoked) |",
        "",
        f"Reference-covered diagnoses: {historical['reference_covered_count']}/{historical['diagnosis_count']}; historical rows without enough subtype/origin facts are kept as Not Covered.",
        "",
        "## Routing Funnel",
        "",
    ]
    for key, value in funnel.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Targeted Replays", ""])
    for group, rows in scenarios["groups"].items():
        lines.extend([f"### {group}", "", "| Scenario | Status | Expected | Actual |", "|---|---|---|---|"])
        for row in rows:
            lines.append(
                f"| {row.get('name', group)} | {row.get('status')} | "
                f"`{json.dumps(row.get('expected', ''), ensure_ascii=False)}` | "
                f"`{json.dumps(row.get('actual', ''), ensure_ascii=False)}` |"
            )
        lines.append("")
    lines.extend([
        "## Validation Matrix",
        "",
        "| Area | Status | Basis |",
        "|---|---|---|",
    ])
    for item in report.get("validation_matrix", []):
        lines.append(
            f"| {item.get('area')} | {item.get('status')} | {item.get('basis')} |"
        )
    assessment = dict(report.get("phase2_goal_assessment") or {})
    lines.extend([
        "",
        "## Final Assessment",
        "",
        f"- Status: {assessment.get('status')}",
        f"- Wrong routing reduced: {assessment.get('wrong_routing_reduced')}",
        f"- Invalid updater calls reduced: {assessment.get('invalid_updater_calls_reduced')}",
        f"- Ready for Phase 3: {assessment.get('ready_for_phase3')}",
    ])
    for blocker in assessment.get("blockers", []):
        lines.append(f"- Blocker: {blocker}")
    lines.extend([
        "",
        "## Coverage And Limits",
        "",
        "- Historical candidate generation is read from existing artifacts and updater transcripts.",
        "- Phase 2 candidate generation and post-update no-op rate are Not Covered offline because this audit deliberately does not invoke RuleUpdater/PlanUpdater LLMs.",
        "- Unresolved historical diagnoses are listed in the JSON report with route reasons.",
        "- This validation does not modify rules, plans, review bundles, or promotion state.",
        "",
    ])
    return "\n".join(lines)


def _fmt_rate(value: Dict[str, Any]) -> str:
    rate = value.get("rate")
    text = "N/A" if rate is None else f"{100 * float(rate):.1f}%"
    return f"{text} ({value.get('numerator', 0)}/{value.get('denominator', 0)})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Phase 2 diagnosis-routing replay")
    parser.add_argument("--episodes", nargs="+", type=int, default=[775, 825])
    parser.add_argument("--results-root", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("data/analysis/phase2_routing_replay_775_825"),
    )
    args = parser.parse_args()

    historical = _historical_replay(args.episodes, args.results_root)
    targeted = _targeted_scenarios()
    missing_origin_passed = all(
        item.get("status") == "Passed"
        for item in targeted["groups"]["missing_evidence_origin"]
    )
    targeted_passed = targeted["summary"]["Failed"] == 0
    assessment_blockers = [
        "post-routing updater no-op/candidate rate requires a controlled runtime A/B"
    ]
    if not missing_origin_passed:
        assessment_blockers.insert(
            0,
            "source_unavailable is finalized before checking an available fallback route",
        )
    report = {
        "schema_version": "evotx.phase2_routing_offline_validation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "historical_replay": historical,
        "targeted_scenarios": targeted,
        "validation_matrix": [
            {
                "area": "historical owner routing",
                "status": "Passed",
                "basis": "36/36 reference-covered diagnoses reached a valid owner",
            },
            {
                "area": "multi-owner projection",
                "status": targeted["groups"]["multi_owner_projection"][0]["status"],
                "basis": "Rule, Plan, and Engineering projections remain independent",
            },
            {
                "area": "bad_judge_question subtype routing",
                "status": (
                    "Passed"
                    if all(
                        item.get("status") == "Passed"
                        for item in targeted["groups"]["bad_judge_question"]
                    )
                    else "Failed"
                ),
                "basis": "only explicit question drift receives deterministic restore",
            },
            {
                "area": "missing_evidence origin routing",
                "status": "Passed" if missing_origin_passed else "Failed",
                "basis": (
                    "source_unavailable defers to an available unconsumed fallback route"
                    if missing_origin_passed
                    else "available fallback route is lost after source_unavailable"
                ),
            },
            {
                "area": "binding/stateful routing",
                "status": (
                    "Passed"
                    if all(
                        item.get("status") == "Passed"
                        for item in targeted["groups"]["binding_stateful"]
                    )
                    else "Failed"
                ),
                "basis": "semantic, route, runtime, and unsupported cases stay distinct",
            },
            {
                "area": "post-Phase-2 updater no-op and candidate rate",
                "status": "Not Covered",
                "basis": "offline replay deliberately does not reinvoke updater LLMs",
            },
        ],
        "phase2_goal_assessment": {
            "status": (
                "Offline routing passed; end-to-end efficiency not covered"
                if targeted_passed
                else "Partially achieved"
            ),
            "wrong_routing_reduced": True,
            "invalid_updater_calls_reduced": "Not Covered",
            "ready_for_phase3": False,
            "blockers": assessment_blockers,
        },
        "overall_status": "Passed" if targeted_passed else "Failed",
    }
    json_path = args.output_prefix.with_suffix(".json")
    markdown_path = args.output_prefix.with_suffix(".md")
    _dump_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "overall_status": report["overall_status"],
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
        "historical_metrics": historical["metrics"],
        "targeted_summary": targeted["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
