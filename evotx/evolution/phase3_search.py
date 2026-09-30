from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from evotx.core.schemas import EvidencePlan, EvolvingRule
from evotx.utils.fingerprint_utils import rule_fingerprint, semantic_plan_fingerprint


PHASE3_PORTFOLIO_SCHEMA_VERSION = "evotx.phase3_candidate_portfolio.v1"
PLAN_STRATEGY_SCHEMA_VERSION = "evotx.plan_strategy_delta.v1"

def _condition_id(step: Dict[str, Any]) -> str:
    return str(step.get("condition_id") or step.get("id") or "").strip().upper()


def _strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item)))


def _step_map(plan: EvidencePlan | None) -> Dict[str, Dict[str, Any]]:
    if plan is None:
        return {}
    return {
        _condition_id(step.to_dict()): step.to_dict()
        for step in list(plan.judge_steps or [])
        if _condition_id(step.to_dict())
    }


@dataclass(frozen=True)
class PlanStrategyDelta:
    condition_id: str
    operations: tuple[str, ...]
    added_routes: tuple[str, ...] = field(default_factory=tuple)
    removed_routes: tuple[str, ...] = field(default_factory=tuple)
    demoted_routes: tuple[str, ...] = field(default_factory=tuple)
    promoted_routes: tuple[str, ...] = field(default_factory=tuple)
    reordered_fields: tuple[str, ...] = field(default_factory=tuple)
    added_tools: tuple[str, ...] = field(default_factory=tuple)
    removed_tools: tuple[str, ...] = field(default_factory=tuple)
    max_followups_before: int = 0
    max_followups_after: int = 0
    depends_on_before: tuple[str, ...] = field(default_factory=tuple)
    depends_on_after: tuple[str, ...] = field(default_factory=tuple)
    protected_dependency_routes_removed: tuple[str, ...] = field(default_factory=tuple)
    safety_tier: str = "safe"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": PLAN_STRATEGY_SCHEMA_VERSION,
            **asdict(self),
            "operations": list(self.operations),
            "added_routes": list(self.added_routes),
            "removed_routes": list(self.removed_routes),
            "demoted_routes": list(self.demoted_routes),
            "promoted_routes": list(self.promoted_routes),
            "reordered_fields": list(self.reordered_fields),
            "added_tools": list(self.added_tools),
            "removed_tools": list(self.removed_tools),
            "depends_on_before": list(self.depends_on_before),
            "depends_on_after": list(self.depends_on_after),
            "protected_dependency_routes_removed": list(
                self.protected_dependency_routes_removed
            ),
        }


def build_plan_strategy_deltas(
    base_plan: EvidencePlan | None,
    candidate_plan: EvidencePlan | None,
) -> List[PlanStrategyDelta]:
    """Describe an actual Plan diff as typed evidence-strategy operations."""
    before_by_id = _step_map(base_plan)
    after_by_id = _step_map(candidate_plan)
    deltas: List[PlanStrategyDelta] = []
    for condition_id in sorted(set(before_by_id) & set(after_by_id)):
        before = before_by_id[condition_id]
        after = after_by_id[condition_id]
        before_defaults = _strings(
            before.get("default_evidence_refs") or before.get("evidence_refs")
        )
        after_defaults = _strings(
            after.get("default_evidence_refs") or after.get("evidence_refs")
        )
        before_followups = _strings(before.get("allowed_followup_views"))
        after_followups = _strings(after.get("allowed_followup_views"))
        before_route_set = set(before_defaults) | set(before_followups)
        after_route_set = set(after_defaults) | set(after_followups)
        added_routes = sorted(after_route_set - before_route_set)
        removed_routes = sorted(before_route_set - after_route_set)
        demoted = sorted(
            (set(before_defaults) & set(after_followups)) - set(after_defaults)
        )
        promoted = sorted(
            (set(before_followups) & set(after_defaults)) - set(before_defaults)
        )
        reordered = []
        if set(before_defaults) == set(after_defaults) and before_defaults != after_defaults:
            reordered.append("default_evidence_refs")
        if set(before_followups) == set(after_followups) and before_followups != after_followups:
            reordered.append("allowed_followup_views")

        before_tools = _strings(before.get("allowed_tools"))
        after_tools = _strings(after.get("allowed_tools"))
        added_tools = sorted(set(after_tools) - set(before_tools))
        removed_tools = sorted(set(before_tools) - set(after_tools))
        before_followup_count = int(before.get("max_followups") or 0)
        after_followup_count = int(after.get("max_followups") or 0)
        before_depends = tuple(_strings(before.get("depends_on")))
        after_depends = tuple(_strings(after.get("depends_on")))

        operations: List[str] = []
        # Promoting an existing follow-up while displacing an old route is a
        # bounded replacement even though the promoted route was already in
        # the route union and therefore is not reported as newly added.
        if removed_routes and (added_routes or promoted):
            operations.append("replace_evidence_route")
        elif added_routes:
            operations.append("expand_evidence_route")
        elif removed_routes:
            operations.append("prune_evidence_route")
        if demoted:
            operations.append("demote_default_to_followup")
        if promoted:
            operations.append("promote_followup_to_default")
        if reordered:
            operations.append("reorder_evidence_route")
        # Adding a registered tool expands an existing evidence route. The
        # Plan validator still rejects unknown tools. Removing or swapping
        # tools changes verification semantics and remains advanced.
        if added_tools and not added_routes:
            operations.append("expand_evidence_route")
        if removed_tools:
            operations.append("change_verification_granularity")
        if before_followup_count != after_followup_count:
            operations.append("change_followup_budget")
        if before_depends != after_depends:
            operations.append("change_dependency_route")
        if not operations:
            continue
        safety_tier = (
            "high_risk"
            if "change_dependency_route" in operations
            else "feature_flag"
            if "prune_evidence_route" in operations
            else "safe"
        )
        deltas.append(PlanStrategyDelta(
            condition_id=condition_id,
            operations=tuple(operations),
            added_routes=tuple(added_routes),
            removed_routes=tuple(removed_routes),
            demoted_routes=tuple(demoted),
            promoted_routes=tuple(promoted),
            reordered_fields=tuple(reordered),
            added_tools=tuple(added_tools),
            removed_tools=tuple(removed_tools),
            max_followups_before=before_followup_count,
            max_followups_after=after_followup_count,
            depends_on_before=before_depends,
            depends_on_after=after_depends,
            # State dependencies protect the producer/consumer contract, not
            # every evidence route that happened to be a producer default.
            # Plan/schema validation owns depends_on and state-key safety.
            protected_dependency_routes_removed=(),
            safety_tier=safety_tier,
        ))
    return deltas


def plan_route_activation_audit(
    base_plan: EvidencePlan | None,
    candidate_plan: EvidencePlan | None,
) -> Dict[str, Any]:
    """Classify whether a Plan diff changes evidence that will actually execute.

    Adding a view only to ``allowed_followup_views`` expands what a Judge may
    request, but it does not make that request happen. Such an optional-only
    edit must not count as blocker coverage during candidate preflight.
    """
    before_by_id = _step_map(base_plan)
    after_by_id = _step_map(candidate_plan)
    conditions: Dict[str, Dict[str, Any]] = {}
    executable_ids: List[str] = []
    optional_only_ids: List[str] = []
    for condition_id in sorted(set(before_by_id) | set(after_by_id)):
        before = before_by_id.get(condition_id)
        after = after_by_id.get(condition_id)
        if before == after:
            continue
        if before is None or after is None:
            conditions[condition_id] = {
                "status": "structural_step_change",
                "execution_capable": True,
            }
            executable_ids.append(condition_id)
            continue

        before_defaults = _strings(
            before.get("default_evidence_refs") or before.get("evidence_refs")
        )
        after_defaults = _strings(
            after.get("default_evidence_refs") or after.get("evidence_refs")
        )
        before_followups = _strings(before.get("allowed_followup_views"))
        after_followups = _strings(after.get("allowed_followup_views"))
        added_followups = sorted(set(after_followups) - set(before_followups))
        promoted_defaults = sorted(
            (set(after_defaults) - set(before_defaults)) & set(before_followups)
        )
        added_defaults = sorted(set(after_defaults) - set(before_defaults))
        default_route_changed = before_defaults != after_defaults
        question_changed = str(before.get("question") or "") != str(
            after.get("question") or ""
        )
        expected_answer_changed = bool(before.get("expected_answer", True)) != bool(
            after.get("expected_answer", True)
        )
        state_or_dependency_changed = any(
            before.get(field) != after.get(field)
            for field in (
                "depends_on",
                "consumes_state_keys",
                "produces_state_key",
                "state_prompt_role",
                "state_output_schema",
            )
        )
        followup_budget_changed = int(before.get("max_followups") or 0) != int(
            after.get("max_followups") or 0
        )
        tool_route_changed = _strings(before.get("allowed_tools")) != _strings(
            after.get("allowed_tools")
        )
        execution_capable = bool(
            default_route_changed
            or question_changed
            or expected_answer_changed
            or state_or_dependency_changed
            or followup_budget_changed
            or tool_route_changed
        )
        status = "execution_capable" if execution_capable else "optional_followup_only"
        conditions[condition_id] = {
            "status": status,
            "execution_capable": execution_capable,
            "default_route_changed": default_route_changed,
            "question_changed": question_changed,
            "expected_answer_changed": expected_answer_changed,
            "state_or_dependency_changed": state_or_dependency_changed,
            "followup_budget_changed": followup_budget_changed,
            "tool_route_changed": tool_route_changed,
            "added_default_views": added_defaults,
            "promoted_default_views": promoted_defaults,
            "added_optional_followup_views": added_followups,
        }
        if execution_capable:
            executable_ids.append(condition_id)
        else:
            optional_only_ids.append(condition_id)
    return {
        "schema_version": "evotx.plan_route_activation_audit.v1",
        "conditions": conditions,
        "execution_capable_condition_ids": executable_ids,
        "optional_followup_only_condition_ids": optional_only_ids,
    }


def plan_strategy_safety_violations(
    deltas: Sequence[PlanStrategyDelta],
    *,
    plan_evidence_audit: Dict[str, Any] | None,
    allow_route_pruning: bool = False,
    allow_dependency_restructure: bool = False,
    require_route_failure_history: bool = True,
) -> List[Dict[str, Any]]:
    """Reject high-risk strategy changes before runtime candidate evaluation."""
    conditions = dict((plan_evidence_audit or {}).get("conditions") or {})
    violations: List[Dict[str, Any]] = []
    for delta in list(deltas or []):
        condition_audit = dict(conditions.get(delta.condition_id) or {})
        if delta.protected_dependency_routes_removed:
            violations.append({
                "condition_id": delta.condition_id,
                "operation": "remove_dependency_required_route",
                "routes": list(delta.protected_dependency_routes_removed),
                "reason": "dependency_required_evidence_route_cannot_be_removed",
            })
        if (
            "change_dependency_route" in delta.operations
            and not allow_dependency_restructure
        ):
            violations.append({
                "condition_id": delta.condition_id,
                "operation": "change_dependency_route",
                "reason": "dependency_restructure_feature_flag_disabled",
            })
        if "prune_evidence_route" in delta.operations:
            if not allow_route_pruning:
                violations.append({
                    "condition_id": delta.condition_id,
                    "operation": "prune_evidence_route",
                    "routes": list(delta.removed_routes),
                    "reason": "route_pruning_feature_flag_disabled",
                })
            else:
                unsupported_pruning = [
                    route
                    for route in delta.removed_routes
                    if not _route_has_failed_or_noncontributing_history(
                        route,
                        condition_audit,
                    )
                ]
                if unsupported_pruning:
                    violations.append({
                        "condition_id": delta.condition_id,
                        "operation": "prune_evidence_route",
                        "routes": unsupported_pruning,
                        "reason": (
                            "pruning_requires_failed_or_repeatedly_"
                            "noncontributing_route_history"
                        ),
                    })
        if (
            require_route_failure_history
            and "replace_evidence_route" in delta.operations
        ):
            unsupported = [
                route
                for route in delta.removed_routes
                if not _route_has_failed_or_noncontributing_history(
                    route,
                    condition_audit,
                )
            ]
            if unsupported:
                violations.append({
                    "condition_id": delta.condition_id,
                    "operation": "replace_evidence_route",
                    "routes": unsupported,
                    "reason": (
                        "replacement_requires_failed_or_repeatedly_"
                        "noncontributing_route_history"
                    ),
                })
        protected_demotions = [
            route
            for route in delta.demoted_routes
            if _route_has_protected_positive_default_contribution(
                route,
                condition_audit,
            )
        ]
        if (
            require_route_failure_history
            and protected_demotions
            and not _has_verified_replacement_route(
                [*delta.added_routes, *delta.promoted_routes],
                condition_audit,
            )
        ):
            violations.append({
                "condition_id": delta.condition_id,
                "operation": "demote_protected_positive_default_route",
                "routes": protected_demotions,
                "reason": (
                    "protected_positive_default_route_requires_verified_"
                    "replacement"
                ),
            })
    return violations


def retain_safe_plan_strategy_operations(
    base_plan: EvidencePlan,
    candidate_plan: EvidencePlan,
    violations: Sequence[Dict[str, Any]],
) -> tuple[EvidencePlan, Dict[str, Any]]:
    """Undo only unsafe typed Plan operations and retain independent safe edits."""
    adjusted = copy.deepcopy(candidate_plan)
    before = {
        str(step.condition_id or step.id).strip().upper(): step
        for step in list(base_plan.judge_steps or [])
    }
    after = {
        str(step.condition_id or step.id).strip().upper(): step
        for step in list(adjusted.judge_steps or [])
    }
    rejected: List[Dict[str, Any]] = []
    fully_restored_conditions: set[str] = set()

    for raw in list(violations or []):
        violation = dict(raw or {})
        condition_id = str(violation.get("condition_id") or "").strip().upper()
        base_step = before.get(condition_id)
        candidate_step = after.get(condition_id)
        if base_step is None or candidate_step is None:
            continue
        operation = str(violation.get("operation") or "")
        routes = _strings(violation.get("routes"))
        rejected.append(violation)

        if operation in {
            "replace_evidence_route",
            "prune_evidence_route",
            "demote_protected_positive_default_route",
        } or violation.get("reason") == "dependency_required_evidence_route_cannot_be_removed":
            _restore_plan_routes(base_step, candidate_step, routes)
            continue
        if operation == "change_dependency_route":
            candidate_step.depends_on = list(base_step.depends_on or [])
            candidate_step.consumes_state_keys = list(base_step.consumes_state_keys or [])
            candidate_step.produces_state_key = str(base_step.produces_state_key or "")
            candidate_step.state_prompt_role = str(base_step.state_prompt_role or "")
            candidate_step.state_output_schema = copy.deepcopy(
                base_step.state_output_schema or {}
            )
            continue

        # Unknown safety failures are not partially interpreted. Restore that
        # condition exactly while allowing independent conditions to survive.
        replacement = copy.deepcopy(base_step)
        index = adjusted.judge_steps.index(candidate_step)
        adjusted.judge_steps[index] = replacement
        after[condition_id] = replacement
        fully_restored_conditions.add(condition_id)

    remaining_deltas = build_plan_strategy_deltas(base_plan, adjusted)
    return adjusted, {
        "schema_version": "evotx.plan_strategy_partial_acceptance.v1",
        "applied": bool(rejected),
        "rejected_operations": rejected,
        "fully_restored_condition_ids": sorted(fully_restored_conditions),
        "retained_condition_ids": sorted(
            delta.condition_id for delta in remaining_deltas
        ),
        "retained_strategy_deltas": [delta.to_dict() for delta in remaining_deltas],
    }


def _restore_plan_routes(base_step: Any, candidate_step: Any, routes: Sequence[str]) -> None:
    wanted = set(_strings(routes))
    if not wanted:
        return
    base_defaults = list(base_step.default_evidence_refs or base_step.evidence_refs or [])
    base_followups = list(base_step.allowed_followup_views or [])
    candidate_defaults = list(candidate_step.default_evidence_refs or [])
    candidate_followups = list(candidate_step.allowed_followup_views or [])
    for route in base_defaults:
        if route not in wanted:
            continue
        if route in candidate_followups:
            candidate_followups.remove(route)
        if route not in candidate_defaults:
            insert_at = min(base_defaults.index(route), len(candidate_defaults))
            candidate_defaults.insert(insert_at, route)
    for route in base_followups:
        if route not in wanted:
            continue
        if route in candidate_defaults and route not in base_defaults:
            candidate_defaults.remove(route)
        if route not in candidate_followups:
            insert_at = min(base_followups.index(route), len(candidate_followups))
            candidate_followups.insert(insert_at, route)
    candidate_step.default_evidence_refs = candidate_defaults
    candidate_step.evidence_refs = list(candidate_defaults)
    candidate_step.allowed_followup_views = candidate_followups


def _route_has_failed_or_noncontributing_history(
    route: str,
    condition_audit: Dict[str, Any],
) -> bool:
    for item in list(condition_audit.get("route_failure_memory") or []):
        if not isinstance(item, dict) or str(item.get("route") or "") != route:
            continue
        if str(item.get("failure_type") or "") not in {
            "unavailable",
            "truncated",
            "no_answer_change",
            "repeatedly_insufficient",
        }:
            continue
        if int(item.get("support_count", 0) or 0) > 0:
            return True
    rendered = 0
    answer_changing = 0
    successful = 0
    tool_ok = 0
    tool_failed = 0
    for group in dict(condition_audit.get("groups") or {}).values():
        if not isinstance(group, dict):
            continue
        rendered += _counter_support(group.get("views"), "view", route)
        answer_changing += _counter_support(
            group.get("answer_changing_followup_views"), "view", route
        )
        successful += _counter_support(
            group.get("successful_followup_views"), "view", route
        )
        for item in list(group.get("tool_statuses") or []):
            if not isinstance(item, dict):
                continue
            status = str(item.get("tool_status") or "")
            if not status.startswith(f"{route}:"):
                continue
            count = int(item.get("support_count", item.get("count", 0)) or 0)
            suffix = status.split(":", 1)[1].strip().lower()
            if suffix == "ok":
                tool_ok += count
            else:
                tool_failed += count
    if tool_failed > 0 and tool_ok == 0:
        return True
    observations = max(rendered, successful)
    return observations >= 2 and answer_changing == 0


def _route_has_protected_positive_default_contribution(
    route: str,
    condition_audit: Dict[str, Any],
) -> bool:
    """Return true when correct positives actually used this route as evidence.

    This is intentionally evidence-driven, not view-name driven. It protects a
    default route only when the audit says correct positives rendered it while
    preserving the target condition.
    """
    wanted = str(route or "").strip()
    if not wanted:
        return False
    positive = dict(
        dict(condition_audit.get("groups") or {}).get("correct_positive") or {}
    )
    if int(positive.get("case_count", 0) or 0) <= 0:
        return False
    final_true = int(
        dict(positive.get("final_answer_counts") or {}).get("true", 0) or 0
    )
    if final_true <= 0:
        return False
    return _counter_support(positive.get("views"), "view", wanted) > 0


def _has_verified_replacement_route(
    routes: Sequence[str],
    condition_audit: Dict[str, Any],
) -> bool:
    wanted = {str(route or "").strip() for route in routes if str(route or "").strip()}
    if not wanted:
        return False
    observed = {
        str(route or "").strip()
        for route in list(condition_audit.get("observed_successful_followup_views") or [])
        if str(route or "").strip()
    }
    for group in dict(condition_audit.get("groups") or {}).values():
        if not isinstance(group, dict):
            continue
        for key in ("successful_followup_views", "answer_changing_followup_views"):
            for item in list(group.get(key) or []):
                if not isinstance(item, dict):
                    continue
                if int(item.get("support_count", item.get("count", 0)) or 0) <= 0:
                    continue
                route = str(item.get("view") or "").strip()
                if route:
                    observed.add(route)
    return bool(wanted.intersection(observed))


def build_condition_evidence_dependencies(
    review_bundle: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Return only explicit, cross-owner condition/evidence capability pairs."""
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for raw in list(matrix.get("signals") or []):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("generalization_status") or "").strip().lower() != "supported":
            continue
        dependency = raw.get("condition_evidence_dependency")
        if not isinstance(dependency, dict):
            continue
        condition_id = str(raw.get("condition_id") or "").strip().upper()
        capability_id = str(dependency.get("capability_id") or "").strip().lower()
        role = str(dependency.get("role") or "").strip().lower()
        owner = str(raw.get("update_target") or "").strip().lower()
        if not condition_id or not capability_id:
            continue
        if (owner, role) not in {("rule", "requires"), ("plan", "provides")}:
            continue
        item = grouped.setdefault((condition_id, capability_id), {
            "condition_id": condition_id,
            "capability_id": capability_id,
            "capability_description": str(
                dependency.get("capability_description") or ""
            )[:280],
            "rule_signal_ids": [],
            "plan_signal_ids": [],
            "explicit_dependency": True,
        })
        key = "rule_signal_ids" if owner == "rule" else "plan_signal_ids"
        signal_id = str(raw.get("signal_id") or "")
        if signal_id and signal_id not in item[key]:
            item[key].append(signal_id)
    return [
        item
        for item in grouped.values()
        if item["rule_signal_ids"] and item["plan_signal_ids"]
    ]


def _annotate_evidence_dependencies(
    spec: Dict[str, Any],
    dependencies: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    annotated = dict(spec)
    rule_conditions = _rule_condition_ids(annotated)
    plan_conditions = _plan_condition_ids(annotated)
    activation = plan_route_activation_audit(
        annotated.get("base_plan"),
        annotated.get("plan"),
    )
    executable_plan_conditions = {
        str(value or "").strip().upper()
        for value in list(activation.get("execution_capable_condition_ids") or [])
        if str(value or "").strip()
    }
    matches = [
        dict(item)
        for item in dependencies
        if str(item.get("condition_id") or "").strip().upper()
        in (rule_conditions | plan_conditions)
    ]
    if not matches:
        return annotated
    kind = str(annotated.get("update_kind") or "")
    for item in matches:
        condition_id = str(item.get("condition_id") or "").strip().upper()
        rule_signal_ids = _rule_signal_ids_for_condition(
            annotated,
            condition_id,
        )
        plan_signal_ids = set(_plan_signal_ids_for_conditions(
            annotated,
            [condition_id],
        ))
        item["semantic_requirement_satisfied"] = (
            kind in {"rule", "rule_plan"}
            and condition_id in rule_conditions
            and bool(
                rule_signal_ids.intersection(item.get("rule_signal_ids") or [])
            )
        )
        item["evidence_accessibility_satisfied"] = (
            kind in {"plan", "rule_plan"}
            and condition_id in plan_conditions
            and condition_id in executable_plan_conditions
            and bool(
                plan_signal_ids.intersection(item.get("plan_signal_ids") or [])
            )
        )
    annotated["condition_evidence_dependencies"] = matches
    return annotated


def _counter_support(value: Any, key: str, wanted: str) -> int:
    return sum(
        int(item.get("support_count", item.get("count", 0)) or 0)
        for item in list(value or [])
        if isinstance(item, dict) and str(item.get(key) or "") == wanted
    )


def apply_plan_strategy_overlay(
    target_plan: EvidencePlan,
    strategy_plan: EvidencePlan,
    *,
    condition_ids: Iterable[str],
    allow_dependency_restructure: bool = False,
) -> EvidencePlan:
    """Apply only evidence-strategy fields to a Rule-compatible Plan."""
    target = copy.deepcopy(target_plan)
    strategy_by_id = _step_map(strategy_plan)
    wanted = {
        str(value or "").strip().upper()
        for value in condition_ids
        if str(value or "").strip()
    }
    patched: List[str] = []
    for step in list(target.judge_steps or []):
        condition_id = str(step.condition_id or step.id or "").strip().upper()
        if condition_id not in wanted or condition_id not in strategy_by_id:
            continue
        source = strategy_by_id[condition_id]
        step.evidence_refs = _strings(source.get("evidence_refs"))
        step.default_evidence_refs = _strings(source.get("default_evidence_refs"))
        step.allowed_followup_views = _strings(source.get("allowed_followup_views"))
        step.allowed_tools = _strings(source.get("allowed_tools"))
        step.max_followups = int(source.get("max_followups") or 0)
        step.view_budget = {
            str(key): int(value)
            for key, value in dict(source.get("view_budget") or {}).items()
            if str(value).strip().lstrip("-").isdigit()
        }
        if allow_dependency_restructure:
            step.depends_on = _strings(source.get("depends_on"))
        patched.append(condition_id)
    metadata = dict(target.metadata or {})
    metadata["phase3_plan_strategy_overlay"] = {
        "condition_ids": sorted(patched),
        "allow_dependency_restructure": bool(allow_dependency_restructure),
        "policy": "evidence_strategy_fields_only",
    }
    target.metadata = metadata
    return target


def _rule_condition_ids(spec: Dict[str, Any]) -> set[str]:
    ids = {
        str(value or "").strip().upper()
        for value in list(spec.get("applied_rule_patch_conditions") or [])
        if str(value or "").strip()
    }
    for delta in list(spec.get("selected_atomic_deltas") or []):
        if isinstance(delta, dict) and str(delta.get("condition_id") or "").strip():
            ids.add(str(delta.get("condition_id")).strip().upper())
    return ids


def _plan_condition_ids(spec: Dict[str, Any]) -> set[str]:
    delta_ids = {
        str(delta.get("condition_id") or "").strip().upper()
        for delta in list(spec.get("plan_strategy_deltas") or [])
        if isinstance(delta, dict)
        and str(delta.get("condition_id") or "").strip()
    }
    if delta_ids:
        return delta_ids
    return {
        str(value or "").strip().upper()
        for value in list(spec.get("applied_plan_patch_conditions") or [])
        if str(value or "").strip()
    }


def _rule_signal_ids_for_condition(
    spec: Dict[str, Any],
    condition_id: str,
) -> set[str]:
    wanted = str(condition_id or "").strip().upper()
    return {
        str(signal_id)
        for delta in list(spec.get("selected_atomic_deltas") or [])
        if isinstance(delta, dict)
        and str(delta.get("condition_id") or "").strip().upper() == wanted
        for signal_id in list(delta.get("applied_signal_ids") or [])
        if str(signal_id)
    }


def _plan_spec_signal_ids(spec: Dict[str, Any]) -> set[str]:
    return {
        str(signal_id)
        for signal_id in list(spec.get("applied_plan_signal_ids") or [])
        if str(signal_id)
    }


def _portfolio_fingerprint(spec: Dict[str, Any]) -> tuple[str, str, str]:
    rule = spec.get("rule")
    plan = spec.get("plan")
    return (
        rule_fingerprint(rule) if isinstance(rule, EvolvingRule) else "",
        semantic_plan_fingerprint(plan) if isinstance(plan, EvidencePlan) else "",
        str(spec.get("candidate_plan_overlay_fingerprint") or ""),
    )


def _plan_spec_condition_ids(spec: Dict[str, Any]) -> List[str]:
    deltas = [
        dict(delta)
        for delta in list(spec.get("plan_strategy_deltas") or [])
        if isinstance(delta, dict)
    ]
    return sorted({
        str(delta.get("condition_id") or "").strip().upper()
        for delta in deltas
        if str(delta.get("condition_id") or "").strip()
    })


def _atomic_plan_strategy_specs(
    spec: Dict[str, Any],
    *,
    current_plan: EvidencePlan | None,
) -> List[Dict[str, Any]]:
    """Split a multi-condition Plan edit without changing its lifecycle."""
    strategy_plan = spec.get("plan")
    condition_ids = _plan_spec_condition_ids(spec)
    if (
        not isinstance(current_plan, EvidencePlan)
        or not isinstance(strategy_plan, EvidencePlan)
        or len(condition_ids) <= 1
    ):
        return []
    atomic: List[Dict[str, Any]] = []
    for condition_id in condition_ids:
        plan = apply_plan_strategy_overlay(
            current_plan,
            strategy_plan,
            condition_ids=[condition_id],
        )
        child = dict(spec)
        child.update({
            "name": f"{spec.get('name', 'plan_candidate')}__{condition_id.lower()}",
            "plan": plan,
            "candidate_strategy": "bounded_plan_strategy_atomic",
            "applied_plan_patch_conditions": [condition_id],
            "plan_strategy_deltas": [
                delta.to_dict()
                for delta in build_plan_strategy_deltas(current_plan, plan)
            ],
            "applied_plan_signal_ids": _plan_signal_ids_for_conditions(
                spec,
                [condition_id],
            ),
        })
        atomic.append(child)
    return atomic


def build_plan_activation_probe_specs(
    specs: Sequence[Dict[str, Any]],
    *,
    current_plan: EvidencePlan | None,
    max_probes: int = 2,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Add probe-only specs for optional routes that otherwise cannot execute.

    The probe temporarily promotes exactly one newly added optional route to a
    first-pass route for the same condition. It is not a formal Plan update; a
    passing probe must be converted into a normal supported candidate later.
    """
    output = [dict(spec) for spec in list(specs or [])]
    audit: Dict[str, Any] = {
        "schema_version": "evotx.phase3.evidence_activation_probe.v1",
        "generated_probe_count": 0,
        "probes": [],
        "skipped": [],
        "policy": (
            "optional-only Plan edits are not viability-bypassed; a bounded "
            "probe temporarily executes one target route and cannot promote "
            "directly"
        ),
    }
    if not isinstance(current_plan, EvidencePlan):
        return output, audit
    remaining = max(0, int(max_probes or 0))
    if remaining <= 0:
        return output, audit
    for spec_index, spec in enumerate(list(specs or [])):
        if remaining <= 0:
            break
        if not isinstance(spec, dict):
            continue
        if str(spec.get("update_kind") or "") != "plan":
            continue
        source_probe_only = bool(spec.get("probe_only"))
        if (
            source_probe_only
            and str(spec.get("candidate_strategy") or "")
            == "evidence_activation_probe"
        ):
            continue
        candidate_plan = spec.get("plan")
        if not isinstance(candidate_plan, EvidencePlan):
            continue
        activation = plan_route_activation_audit(current_plan, candidate_plan)
        probe_condition_ids = list(dict.fromkeys([
            *[
                str(value or "").strip().upper()
                for value in list(
                    activation.get("optional_followup_only_condition_ids") or []
                )
                if str(value or "").strip()
            ],
            *_activation_probe_metadata_condition_ids(spec),
        ]))
        for condition_id in probe_condition_ids:
            if remaining <= 0:
                break
            condition_id = str(condition_id or "").strip().upper()
            condition_audit = dict(
                (activation.get("conditions") or {}).get(condition_id) or {}
            )
            target_routes = _activation_probe_target_routes(
                spec,
                condition_id,
                fallback_routes=_strings(
                    condition_audit.get("added_optional_followup_views")
                ),
            )
            if not target_routes:
                audit["skipped"].append({
                    "candidate_name": str(spec.get("name") or ""),
                    "condition_id": condition_id,
                    "reason": "optional_only_without_added_route",
                })
                continue
            target_route = target_routes[0]
            probe_plan = _activation_probe_plan(
                current_plan,
                candidate_plan,
                condition_id=condition_id,
                target_route=target_route,
            )
            if probe_plan is None:
                audit["skipped"].append({
                    "candidate_name": str(spec.get("name") or ""),
                    "condition_id": condition_id,
                    "target_route": target_route,
                    "reason": "target_condition_missing",
                })
                continue
            plan_signal_ids = _plan_signal_ids_for_conditions(
                spec,
                [condition_id],
            ) or sorted(_plan_spec_signal_ids(spec))
            probe = dict(spec)
            probe.update({
                "name": (
                    f"{spec.get('name', 'plan_candidate')}"
                    f"__activation_probe__{condition_id.lower()}"
                ),
                "plan": probe_plan,
                "candidate_strategy": (
                    str(spec.get("candidate_strategy") or "")
                    if source_probe_only
                    else "evidence_activation_probe"
                ),
                "candidate_status": (
                    str(spec.get("candidate_status") or "experimental_probe")
                    if source_probe_only
                    else "activation_probe"
                ),
                "probe_only": True,
                "hypothesis_source_signal_ids": list(
                    spec.get("hypothesis_source_signal_ids") or plan_signal_ids
                ),
                "experimental_signal_ids": (
                    list(spec.get("experimental_signal_ids") or [])
                    if source_probe_only
                    else []
                ),
                "applied_plan_patch_conditions": [condition_id],
                "applied_plan_signal_ids": list(plan_signal_ids),
                "plan_strategy_deltas": [
                    delta.to_dict()
                    for delta in build_plan_strategy_deltas(current_plan, probe_plan)
                ],
                "phase3_activation_probe": {
                    "source_candidate": str(spec.get("name") or ""),
                    "condition_id": condition_id,
                    "target_routes": [target_route],
                    "source_signal_ids": list(plan_signal_ids),
                    "lifecycle_state": "probe_candidate",
                    "direct_promotion_allowed": False,
                    "partial_acceptance_allowed": False,
                },
                "hypothesis_lifecycle": {
                    "state": "probe_candidate",
                    "probe_result": "pending",
                    "conversion_reason": "",
                    "direct_promotion_allowed": False,
                },
            })
            if source_probe_only:
                # Preserve the experimental lifecycle while replacing its
                # non-executable optional route with the activation-bound form.
                output[spec_index] = probe
            else:
                output.append(probe)
            audit["probes"].append({
                "name": probe["name"],
                "source_candidate": str(spec.get("name") or ""),
                "condition_id": condition_id,
                "target_route": target_route,
                "source_signal_ids": list(plan_signal_ids),
            })
            remaining -= 1
    audit["generated_probe_count"] = len(audit["probes"])
    return output, audit


def _activation_probe_target_routes(
    spec: Dict[str, Any],
    condition_id: str,
    *,
    fallback_routes: Sequence[str],
) -> List[str]:
    plan = spec.get("plan")
    metadata = dict(plan.metadata or {}) if isinstance(plan, EvidencePlan) else {}
    probe_meta = dict(metadata.get("phase3_unverified_default_activation_probe") or {})
    wanted = str(condition_id or "").strip().upper()
    preferred: List[str] = []
    for item in list(probe_meta.get("demoted_default_promotions") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("condition_id") or "").strip().upper() != wanted:
            continue
        preferred.extend(_strings(item.get("target_routes")))
    return list(dict.fromkeys([*preferred, *_strings(fallback_routes)]))


def _activation_probe_metadata_condition_ids(spec: Dict[str, Any]) -> List[str]:
    plan = spec.get("plan")
    metadata = dict(plan.metadata or {}) if isinstance(plan, EvidencePlan) else {}
    probe_meta = dict(metadata.get("phase3_unverified_default_activation_probe") or {})
    return list(dict.fromkeys(
        str(item.get("condition_id") or "").strip().upper()
        for item in list(probe_meta.get("demoted_default_promotions") or [])
        if isinstance(item, dict)
        and str(item.get("condition_id") or "").strip()
    ))


def _activation_probe_plan(
    current_plan: EvidencePlan,
    candidate_plan: EvidencePlan,
    *,
    condition_id: str,
    target_route: str,
) -> EvidencePlan | None:
    plan = copy.deepcopy(candidate_plan)
    wanted = str(condition_id or "").strip().upper()
    target = str(target_route or "").strip()
    if not wanted or not target:
        return None
    found = False
    for step in list(plan.judge_steps or []):
        step_id = str(step.condition_id or step.id or "").strip().upper()
        if step_id != wanted:
            continue
        defaults = _strings(step.default_evidence_refs or step.evidence_refs)
        followups = _strings(step.allowed_followup_views)
        step.default_evidence_refs = [target] + [
            route for route in defaults if route != target
        ]
        step.evidence_refs = list(step.default_evidence_refs)
        if target not in followups:
            followups.insert(0, target)
        step.allowed_followup_views = followups
        found = True
        break
    if not found:
        return None
    metadata = dict(plan.metadata or {})
    metadata["phase3_evidence_activation_probe"] = {
        "condition_id": wanted,
        "target_routes": [target],
        "source": "optional_followup_activation_probe",
        "probe_only": True,
        "direct_promotion_allowed": False,
    }
    plan.metadata = metadata
    return plan


def _plan_signal_ids_for_conditions(
    spec: Dict[str, Any],
    condition_ids: Iterable[str],
) -> List[str]:
    wanted = {
        str(value or "").strip().upper()
        for value in condition_ids
        if str(value or "").strip()
    }
    materialization = dict(spec.get("plan_patch_materialization") or {})
    attributed = list(dict.fromkeys(
        str(signal_id)
        for patch in list(materialization.get("patches") or [])
        if isinstance(patch, dict)
        and bool(patch.get("accepted"))
        and str(patch.get("condition_id") or "").strip().upper() in wanted
        for signal_id in list(patch.get("signal_ids") or [])
        if str(signal_id)
    ))
    if attributed:
        return attributed
    if "patches" in materialization:
        return []
    return sorted(_plan_spec_signal_ids(spec))


def build_bounded_candidate_portfolio(
    specs: Sequence[Dict[str, Any]],
    *,
    current_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
    max_candidates: int = 6,
    enable_joint_candidates: bool = True,
    allow_dependency_restructure: bool = False,
    condition_evidence_dependencies: Sequence[Dict[str, Any]] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep diverse Rule/Plan strategies and add only matching joint candidates."""
    limit = max(1, int(max_candidates or 1))
    dependencies = list(condition_evidence_dependencies or [])
    rule_specs = [
        dict(spec) for spec in specs if spec.get("update_kind") == "rule"
    ]
    plan_specs = [
        dict(spec) for spec in specs if spec.get("update_kind") == "plan"
    ]
    other_specs = [
        dict(spec)
        for spec in specs
        if spec.get("update_kind") not in {"rule", "plan"}
    ]
    for plan_spec in plan_specs:
        deltas = build_plan_strategy_deltas(current_plan, plan_spec.get("plan"))
        plan_spec["plan_strategy_deltas"] = [delta.to_dict() for delta in deltas]
        plan_spec.setdefault("base_plan", current_plan)
    rule_specs = [
        _annotate_evidence_dependencies(spec, dependencies)
        for spec in rule_specs
    ]
    plan_specs = [
        _annotate_evidence_dependencies(spec, dependencies)
        for spec in plan_specs
    ]
    atomic_plan_specs = [
        atomic
        for plan_spec in plan_specs
        for atomic in _atomic_plan_strategy_specs(
            plan_spec,
            current_plan=current_plan,
        )
    ]

    joint_specs: List[Dict[str, Any]] = []
    if enable_joint_candidates:
        for rule_index, rule_spec in enumerate(rule_specs, start=1):
            if bool(rule_spec.get("probe_only")):
                continue
            rule_conditions = _rule_condition_ids(rule_spec)
            if not rule_conditions:
                continue
            for plan_index, plan_spec in enumerate(plan_specs, start=1):
                if bool(plan_spec.get("probe_only")):
                    continue
                plan_conditions = _plan_condition_ids(plan_spec)
                activation = plan_route_activation_audit(
                    current_plan,
                    plan_spec.get("plan"),
                )
                execution_capable = {
                    str(value or "").strip().upper()
                    for value in list(
                        activation.get("execution_capable_condition_ids") or []
                    )
                    if str(value or "").strip()
                }
                matched_dependencies = [
                    dict(dependency)
                    for dependency in dependencies
                    if str(
                        dependency.get("condition_id") or ""
                    ).strip().upper() in rule_conditions
                    and str(
                        dependency.get("condition_id") or ""
                    ).strip().upper() in plan_conditions
                    and str(
                        dependency.get("condition_id") or ""
                    ).strip().upper() in execution_capable
                    and bool(_rule_signal_ids_for_condition(
                        rule_spec,
                        str(dependency.get("condition_id") or ""),
                    ).intersection(
                        dependency.get("rule_signal_ids") or []
                    ))
                    and bool(set(_plan_signal_ids_for_conditions(
                        plan_spec,
                        [str(dependency.get("condition_id") or "")],
                    )).intersection(
                        dependency.get("plan_signal_ids") or []
                    ))
                ]
                shared = sorted({
                    str(dependency.get("condition_id") or "").strip().upper()
                    for dependency in matched_dependencies
                    if str(dependency.get("condition_id") or "").strip()
                })
                if not shared:
                    continue
                matched_rule_signal_ids = {
                    str(signal_id)
                    for dependency in matched_dependencies
                    for signal_id in list(dependency.get("rule_signal_ids") or [])
                    if str(signal_id)
                }
                matched_plan_signal_ids = {
                    str(signal_id)
                    for dependency in matched_dependencies
                    for signal_id in list(dependency.get("plan_signal_ids") or [])
                    if str(signal_id)
                }
                actual_rule_signal_ids = {
                    signal_id
                    for condition_id in shared
                    for signal_id in _rule_signal_ids_for_condition(
                        rule_spec,
                        condition_id,
                    )
                }
                actual_plan_signal_ids = {
                    signal_id
                    for condition_id in shared
                    for signal_id in _plan_signal_ids_for_conditions(
                        plan_spec,
                        [condition_id],
                    )
                }
                if (
                    not actual_rule_signal_ids
                    or not actual_plan_signal_ids
                    or not actual_rule_signal_ids.issubset(matched_rule_signal_ids)
                    or not actual_plan_signal_ids.issubset(matched_plan_signal_ids)
                ):
                    # Same-condition candidates may aggregate unrelated FN/FP
                    # proposals. They are not the exact provider/consumer pair.
                    continue
                strategy_plan = plan_spec.get("plan")
                if not isinstance(strategy_plan, EvidencePlan):
                    continue
                shared_plan_deltas = [
                    dict(delta)
                    for delta in list(
                        plan_spec.get("plan_strategy_deltas") or []
                    )
                    if isinstance(delta, dict)
                    and str(delta.get("condition_id") or "").strip().upper()
                    in shared
                ]
                joint = dict(rule_spec)
                joint.update({
                    "name": f"joint_candidate_{rule_index:02d}_{plan_index:02d}",
                    "update_kind": "rule_plan",
                    "candidate_strategy": "bounded_rule_plan_joint",
                    "applied_plan_patch_conditions": shared,
                    "joint_condition_ids": shared,
                    "plan_strategy_deltas": shared_plan_deltas,
                    "applied_plan_signal_ids": sorted(actual_plan_signal_ids),
                    "joint_condition_evidence_dependencies": (
                        matched_dependencies
                    ),
                    "candidate_plan_overlay": strategy_plan,
                    "candidate_plan_overlay_fingerprint": semantic_plan_fingerprint(
                        strategy_plan
                    ),
                    "allow_dependency_restructure": bool(
                        allow_dependency_restructure
                    ),
                    "candidate_status": "supported",
                    "probe_only": False,
                    "hypothesis_source_signal_ids": [],
                    "hypothesis_lifecycle": {},
                })
                existing_plan = rule_spec.get("plan")
                if isinstance(existing_plan, EvidencePlan):
                    joint["plan"] = apply_plan_strategy_overlay(
                        existing_plan,
                        strategy_plan,
                        condition_ids=shared,
                        allow_dependency_restructure=allow_dependency_restructure,
                    )
                    joint["requires_plan_regeneration"] = False
                elif isinstance(current_plan, EvidencePlan):
                    # Viability needs an executable preview even when the Rule
                    # candidate will regenerate its final Plan. Evaluation
                    # still regenerates first, then applies the same overlay.
                    joint["plan"] = apply_plan_strategy_overlay(
                        current_plan,
                        strategy_plan,
                        condition_ids=shared,
                        allow_dependency_restructure=(
                            allow_dependency_restructure
                        ),
                    )
                joint_specs.append(joint)

    joint_specs = [
        _annotate_evidence_dependencies(spec, dependencies)
        for spec in joint_specs
    ]

    ordered: List[Dict[str, Any]] = []
    if rule_specs:
        ordered.append(rule_specs.pop(0))
    if plan_specs:
        ordered.append(plan_specs.pop(0))
    if plan_specs and bool(plan_specs[0].get("probe_only")):
        ordered.append(plan_specs.pop(0))
    if joint_specs:
        ordered.append(joint_specs.pop(0))
    ordered.extend(rule_specs)
    ordered.extend(plan_specs)
    ordered.extend(atomic_plan_specs)
    ordered.extend(joint_specs)
    ordered.extend(other_specs)

    unique_specs: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    duplicate_names: List[str] = []
    for spec in ordered:
        fingerprint = _portfolio_fingerprint(spec)
        if fingerprint in seen:
            duplicate_names.append(str(spec.get("name") or ""))
            continue
        seen.add(fingerprint)
        unique_specs.append(spec)
    selected = unique_specs[:limit]
    truncated_names = [
        str(spec.get("name") or "") for spec in unique_specs[limit:]
    ]
    audit = {
        "schema_version": PHASE3_PORTFOLIO_SCHEMA_VERSION,
        "enabled": True,
        "max_candidates": limit,
        "input_candidate_count": len(list(specs or [])),
        "rule_only_count": sum(1 for item in selected if item.get("update_kind") == "rule"),
        "plan_only_count": sum(1 for item in selected if item.get("update_kind") == "plan"),
        "joint_count": sum(1 for item in selected if item.get("update_kind") == "rule_plan"),
        "probe_only_count": sum(1 for item in selected if item.get("probe_only")),
        "selected_candidates": [str(item.get("name") or "") for item in selected],
        "candidate_lifecycles": [
            {
                "name": str(item.get("name") or ""),
                "hypothesis_source_signal_ids": list(
                    item.get("hypothesis_source_signal_ids")
                    or item.get("experimental_signal_ids")
                    or []
                ),
                "lifecycle_state": str(
                    (item.get("hypothesis_lifecycle") or {}).get("state")
                    or "supported"
                ),
                "probe_only": bool(item.get("probe_only")),
            }
            for item in selected
        ],
        "deduplicated_candidates": duplicate_names,
        "truncated_candidate_count": len(truncated_names),
        "truncated_candidates": truncated_names,
        "safety_policy": {
            "joint_requires_explicit_condition_evidence_dependency": True,
            "joint_requires_exact_source_signal_lineage": True,
            "joint_requires_execution_capable_plan_route": True,
            "cartesian_product_disabled": True,
            "dependency_restructure_enabled": bool(allow_dependency_restructure),
            "experimental_candidate_direct_promotion": False,
            "supported_experimental_mixing": False,
        },
    }
    return selected, audit
