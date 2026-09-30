from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence

from evotx.core.plan import compile_rule_to_baseline_plan
from evotx.core.schemas import EvidencePlan, EvolvingRule, JudgeStep, RuleCondition
from evotx.planner.plan_validator import (
    dependency_affected_condition_ids,
    dependency_consistency_report,
)


RULE_PARTIAL_ACCEPTANCE_STAGE = "rule_condition_v1"
RULE_ATOMIC_DELTA_STAGE = "rule_atomic_delta_v1"

SEMANTIC_DIRECTIONS = {"tighten", "broaden", "refine"}


def _normalized_id(value: Any) -> str:
    return str(value or "").strip().upper()


@dataclass(frozen=True)
class RuleConditionDelta:
    delta_id: str
    collection: str
    operation: str
    condition_id: str
    before: Dict[str, Any] | None
    after: Dict[str, Any] | None
    semantic_direction: str = "refine"
    applied_signal_ids: tuple[str, ...] = field(default_factory=tuple)
    source_error_types: tuple[str, ...] = field(default_factory=tuple)
    covered_blocker_ids: tuple[str, ...] = field(default_factory=tuple)
    joint_resolution_signal_ids: tuple[str, ...] = field(default_factory=tuple)
    independently_verdict_capable: bool = False
    local_viability_status: str = "eligible"
    local_viability_reason: str = ""
    rationale: str = ""
    dependency_impact: Dict[str, Any] = field(default_factory=dict)

    @property
    def target_key(self) -> tuple[str, str]:
        return self.collection, _normalized_id(self.condition_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "artifact": "rule",
            "collection": self.collection,
            "operation": self.operation,
            "condition_id": self.condition_id,
            "before": copy.deepcopy(self.before),
            "after": copy.deepcopy(self.after),
            "semantic_direction": self.semantic_direction,
            "applied_signal_ids": list(self.applied_signal_ids),
            "source_error_types": list(self.source_error_types),
            "covered_blocker_ids": list(self.covered_blocker_ids),
            "joint_resolution_signal_ids": list(
                self.joint_resolution_signal_ids
            ),
            "independently_verdict_capable": bool(
                self.independently_verdict_capable
            ),
            "local_viability_status": self.local_viability_status,
            "local_viability_reason": self.local_viability_reason,
            "rationale": self.rationale,
            "dependency_impact": copy.deepcopy(self.dependency_impact),
            "stage": RULE_ATOMIC_DELTA_STAGE,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RuleConditionDelta":
        """Restore an attributed delta without deriving new lineage."""
        if not isinstance(raw, dict):
            raise ValueError("atomic Rule delta must be an object")
        return cls(
            delta_id=str(raw.get("delta_id") or ""),
            collection=str(raw.get("collection") or ""),
            operation=str(raw.get("operation") or ""),
            condition_id=str(raw.get("condition_id") or ""),
            before=copy.deepcopy(raw.get("before")),
            after=copy.deepcopy(raw.get("after")),
            semantic_direction=str(
                raw.get("semantic_direction") or "refine"
            ).strip().lower(),
            applied_signal_ids=tuple(dict.fromkeys(
                str(value)
                for value in list(raw.get("applied_signal_ids") or [])
                if str(value)
            )),
            source_error_types=tuple(dict.fromkeys(
                str(value or "").strip().upper()
                for value in list(raw.get("source_error_types") or [])
                if str(value or "").strip()
            )),
            covered_blocker_ids=tuple(dict.fromkeys(
                _normalized_id(value)
                for value in list(raw.get("covered_blocker_ids") or [])
                if _normalized_id(value)
            )),
            joint_resolution_signal_ids=tuple(dict.fromkeys(
                str(value)
                for value in list(raw.get("joint_resolution_signal_ids") or [])
                if str(value)
            )),
            independently_verdict_capable=bool(
                raw.get("independently_verdict_capable", False)
            ),
            local_viability_status=str(
                raw.get("local_viability_status") or "eligible"
            ),
            local_viability_reason=str(
                raw.get("local_viability_reason") or ""
            ),
            rationale=str(raw.get("rationale") or "")[:700],
            dependency_impact=copy.deepcopy(raw.get("dependency_impact") or {}),
        )


def attributed_rule_delta_lineage_report(
    base_rule: EvolvingRule,
    candidate_rule: EvolvingRule,
    raw_deltas: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify saved atomic deltas exactly describe the parent Rule diff."""
    errors: List[Dict[str, Any]] = []
    deltas: List[RuleConditionDelta] = []
    for index, raw in enumerate(list(raw_deltas or [])):
        if not isinstance(raw, dict):
            errors.append({"index": index, "reason": "delta_not_object"})
            continue
        if str(raw.get("stage") or "") != RULE_ATOMIC_DELTA_STAGE:
            errors.append({"index": index, "reason": "atomic_stage_missing"})
            continue
        try:
            delta = RuleConditionDelta.from_dict(raw)
        except Exception as exc:
            errors.append({
                "index": index,
                "reason": "delta_restore_failed",
                "error": str(exc)[:300],
            })
            continue
        if not delta.delta_id or not delta.applied_signal_ids:
            errors.append({
                "index": index,
                "delta_id": delta.delta_id,
                "reason": "attributed_signal_lineage_missing",
            })
        if delta.semantic_direction not in SEMANTIC_DIRECTIONS:
            errors.append({
                "index": index,
                "delta_id": delta.delta_id,
                "reason": "invalid_semantic_direction",
            })
        if delta.local_viability_status != "eligible":
            errors.append({
                "index": index,
                "delta_id": delta.delta_id,
                "reason": "atomic_delta_not_locally_viable",
            })
        deltas.append(delta)

    actual = build_rule_condition_deltas(base_rule, candidate_rule)
    actual_by_id = {delta.delta_id: delta for delta in actual}
    saved_by_id = {delta.delta_id: delta for delta in deltas}
    if len(saved_by_id) != len(deltas):
        errors.append({"reason": "duplicate_saved_delta_id"})
    if set(actual_by_id) != set(saved_by_id):
        errors.append({
            "reason": "saved_delta_set_does_not_match_parent_rule_diff",
            "actual_delta_ids": sorted(actual_by_id),
            "saved_delta_ids": sorted(saved_by_id),
        })
    for delta_id in sorted(set(actual_by_id) & set(saved_by_id)):
        actual_delta = actual_by_id[delta_id]
        saved_delta = saved_by_id[delta_id]
        if any((
            actual_delta.collection != saved_delta.collection,
            actual_delta.operation != saved_delta.operation,
            _normalized_id(actual_delta.condition_id)
            != _normalized_id(saved_delta.condition_id),
            actual_delta.before != saved_delta.before,
            actual_delta.after != saved_delta.after,
        )):
            errors.append({
                "delta_id": delta_id,
                "reason": "saved_delta_payload_does_not_match_parent_rule_diff",
            })
    return {
        "schema_version": "evotx.rule_atomic_delta_lineage_report.v1",
        "valid": not errors,
        "errors": errors,
        "deltas": deltas,
        "delta_ids": [delta.delta_id for delta in deltas],
    }


def rule_delta_subset_preserves_joint_lineage(
    deltas: Sequence[RuleConditionDelta],
    accepted_delta_ids: Iterable[str],
) -> bool:
    """Do not split a joint-resolution signal contract during partial search."""
    accepted_ids = {
        str(value) for value in list(accepted_delta_ids or []) if str(value)
    }
    selected = [delta for delta in deltas if delta.delta_id in accepted_ids]
    selected_signals = {
        signal_id for delta in selected for signal_id in delta.applied_signal_ids
    }
    for delta in selected:
        required = set(delta.joint_resolution_signal_ids)
        if required and not required.issubset(selected_signals):
            return False
    return True


def build_rule_condition_deltas(
    base_rule: EvolvingRule,
    candidate_rule: EvolvingRule,
    *,
    operation_manifest: Sequence[Dict[str, Any]] | None = None,
    base_plan: EvidencePlan | None = None,
    candidate_plan: EvidencePlan | None = None,
) -> List[RuleConditionDelta]:
    deltas: List[RuleConditionDelta] = []
    deltas.extend(
        _diff_condition_collection(
            base_rule.conditions,
            candidate_rule.conditions,
            collection="conditions",
            label="condition",
        )
    )
    deltas.extend(
        _diff_condition_collection(
            base_rule.exclusion_conditions,
            candidate_rule.exclusion_conditions,
            collection="exclusion_conditions",
            label="exclusion",
        )
    )
    return _attribute_rule_condition_deltas(
        deltas,
        operation_manifest=operation_manifest or [],
        base_plan=base_plan,
        candidate_plan=candidate_plan,
    )


def _attribute_rule_condition_deltas(
    deltas: Sequence[RuleConditionDelta],
    *,
    operation_manifest: Sequence[Dict[str, Any]],
    base_plan: EvidencePlan | None,
    candidate_plan: EvidencePlan | None,
) -> List[RuleConditionDelta]:
    manifest_by_target: Dict[tuple[str, str], Dict[str, Any]] = {}
    for raw in list(operation_manifest or []):
        if not isinstance(raw, dict):
            continue
        collection = str(raw.get("collection") or "").strip()
        condition_id = _normalized_id(raw.get("condition_id"))
        if collection and condition_id:
            manifest_by_target[(collection, condition_id)] = raw

    attributed: List[RuleConditionDelta] = []
    for delta in deltas:
        raw = manifest_by_target.get(delta.target_key, {})
        direction = str(
            raw.get("semantic_direction")
            or _default_delta_direction(delta)
        ).strip().lower()
        if direction not in SEMANTIC_DIRECTIONS:
            direction = "refine"
        source_error_types = tuple(sorted({
            str(value or "").strip().upper()
            for value in list(raw.get("source_error_types") or [])
            if str(value or "").strip()
        }))
        joint_signal_ids = tuple(dict.fromkeys(
            str(value)
            for value in list(raw.get("joint_resolution_signal_ids") or [])
            if str(value)
        ))
        status, reason = atomic_delta_direction_viability(
            semantic_direction=direction,
            source_error_types=source_error_types,
            joint_resolution_signal_ids=joint_signal_ids,
        )
        attributed.append(RuleConditionDelta(
            delta_id=delta.delta_id,
            collection=delta.collection,
            operation=delta.operation,
            condition_id=delta.condition_id,
            before=delta.before,
            after=delta.after,
            semantic_direction=direction,
            applied_signal_ids=tuple(dict.fromkeys(
                str(value)
                for value in list(raw.get("applied_signal_ids") or [])
                if str(value)
            )),
            source_error_types=source_error_types,
            covered_blocker_ids=tuple(dict.fromkeys(
                _normalized_id(value)
                for value in list(raw.get("covered_blocker_ids") or [])
                if _normalized_id(value)
            )),
            joint_resolution_signal_ids=joint_signal_ids,
            independently_verdict_capable=bool(
                raw.get("independently_verdict_capable", False)
            ),
            local_viability_status=status,
            local_viability_reason=reason,
            rationale=str(raw.get("rationale") or "")[:700],
            dependency_impact=_delta_dependency_impact(
                delta,
                base_plan=base_plan,
                candidate_plan=candidate_plan,
            ),
        ))
    return attributed


def _plan_step_dependency_contract(
    plan: EvidencePlan | None,
    condition_id: str,
) -> Dict[str, Any] | None:
    if plan is None:
        return None
    wanted = _normalized_id(condition_id)
    for step in list(plan.judge_steps or []):
        if _normalized_id(step.condition_id or step.id) != wanted:
            continue
        return {
            "step_id": str(step.id or ""),
            "depends_on": list(step.depends_on or []),
            "consumes_state_keys": list(step.consumes_state_keys or []),
            "produces_state_key": str(step.produces_state_key or ""),
            "state_prompt_role": str(step.state_prompt_role or ""),
            "state_output_schema": copy.deepcopy(step.state_output_schema or {}),
        }
    return None


def _delta_dependency_impact(
    delta: RuleConditionDelta,
    *,
    base_plan: EvidencePlan | None,
    candidate_plan: EvidencePlan | None,
) -> Dict[str, Any]:
    condition_id = _normalized_id(delta.condition_id)
    before = _plan_step_dependency_contract(base_plan, condition_id)
    after = _plan_step_dependency_contract(candidate_plan, condition_id)
    affected_consumers = sorted(
        dependency_affected_condition_ids(base_plan, [condition_id])
    )
    affected_dependencies = sorted({
        _normalized_id(value)
        for value in list((before or {}).get("depends_on") or [])
        if _normalized_id(value)
    })
    producer_touched = bool(
        (before or {}).get("produces_state_key")
        or (after or {}).get("produces_state_key")
    )
    consumer_touched = bool(
        list((before or {}).get("depends_on") or [])
        or list((before or {}).get("consumes_state_keys") or [])
        or list((after or {}).get("depends_on") or [])
        or list((after or {}).get("consumes_state_keys") or [])
    )
    if candidate_plan is None:
        validation_result: Dict[str, Any] = {
            "dependency_validation_status": (
                "pending_plan_materialization"
                if delta.operation in {"add", "remove"}
                or producer_touched
                or consumer_touched
                else "not_required"
            ),
        }
        producer_changed: bool | None = None
        consumer_changed: bool | None = None
    else:
        validation_result = dependency_consistency_report(
            candidate_plan,
            previous_plan=base_plan,
            changed_condition_ids=[condition_id],
        )
        producer_changed = (
            (before or {}).get("produces_state_key"),
            (before or {}).get("state_prompt_role"),
            (before or {}).get("state_output_schema"),
        ) != (
            (after or {}).get("produces_state_key"),
            (after or {}).get("state_prompt_role"),
            (after or {}).get("state_output_schema"),
        )
        consumer_changed = (
            (before or {}).get("depends_on"),
            (before or {}).get("consumes_state_keys"),
        ) != (
            (after or {}).get("depends_on"),
            (after or {}).get("consumes_state_keys"),
        )
    return {
        "affected_dependency_ids": sorted(set(
            affected_dependencies + affected_consumers
        )),
        "affected_producer_ids": [condition_id] if producer_touched else [],
        "affected_consumer_ids": affected_consumers,
        "producer_role_changed": producer_changed,
        "consumer_bindings_changed": consumer_changed,
        "requires_dependency_revalidation": bool(
            delta.operation in {"add", "remove"}
            or producer_touched
            or consumer_touched
            or affected_consumers
        ),
        "dependency_contract_before": before,
        "dependency_contract_after": after,
        "dependency_validation_result": validation_result,
    }


def rule_delta_dependency_preflight(
    *,
    base_plan: EvidencePlan | None,
    deltas: Sequence[RuleConditionDelta],
    accepted_delta_ids: Iterable[str],
) -> Dict[str, Any]:
    """Reject delta bundles that remove a producer but leave its consumers."""
    accepted = {
        str(value) for value in list(accepted_delta_ids or []) if str(value)
    }
    selected = [delta for delta in list(deltas or []) if delta.delta_id in accepted]
    removed = {
        _normalized_id(delta.condition_id)
        for delta in selected
        if delta.operation == "remove"
    }
    errors: list[Dict[str, Any]] = []
    edges = (
        dependency_consistency_report(base_plan)["dependency_edges_after"]
        if base_plan is not None
        else []
    )
    for edge in edges:
        producer_id = _normalized_id(edge.get("producer_condition_id"))
        consumer_id = _normalized_id(edge.get("consumer_condition_id"))
        if producer_id in removed and consumer_id not in removed:
            errors.append({
                "code": "producer_remove_leaves_consumer",
                "producer_condition_id": producer_id,
                "consumer_condition_id": consumer_id,
                "state_keys": list(edge.get("state_keys") or []),
            })
    return {
        "schema_version": "evotx.rule_delta_dependency_preflight.v1",
        "accepted_delta_ids": [
            delta.delta_id for delta in selected
        ],
        "dependency_edges_before": edges,
        "affected_producers": sorted({
            _normalized_id(delta.condition_id)
            for delta in selected
            if list((delta.dependency_impact or {}).get("affected_producer_ids") or [])
        }),
        "affected_consumers": sorted({
            value
            for delta in selected
            for value in list(
                (delta.dependency_impact or {}).get("affected_consumer_ids") or []
            )
        }),
        "dangling_dependency_count": len(errors),
        "binding_mismatch_count": 0,
        "dependency_validation_status": "passed" if not errors else "rejected",
        "dependency_rejection_reason": (
            "producer_remove_leaves_consumer" if errors else ""
        ),
        "errors": errors,
    }


def _default_delta_direction(delta: RuleConditionDelta) -> str:
    if delta.operation == "add":
        return "tighten"
    if delta.operation == "remove":
        return "broaden"
    return "refine"


def atomic_delta_direction_viability(
    *,
    semantic_direction: str,
    source_error_types: Sequence[str],
    joint_resolution_signal_ids: Sequence[str] = (),
) -> tuple[str, str]:
    """Check local repair direction without requiring a final verdict change."""
    direction = str(semantic_direction or "").strip().lower()
    errors = {
        str(value or "").strip().upper()
        for value in list(source_error_types or [])
        if str(value or "").strip()
    }
    if direction not in SEMANTIC_DIRECTIONS:
        return "rejected", "invalid_semantic_direction"
    if not errors or direction == "refine":
        return "eligible", "direction_is_neutral_or_unattributed"
    mismatch = (
        (errors == {"FP"} and direction != "tighten")
        or (errors == {"FN"} and direction != "broaden")
        or (errors == {"FP", "FN"})
    )
    if not mismatch:
        return "eligible", "repair_direction_matches_error"
    if joint_resolution_signal_ids:
        return "eligible", "direction_mismatch_covered_by_joint_resolution"
    return "rejected", "semantic_direction_mismatches_error_without_joint_resolution"


def _diff_condition_collection(
    base_items: Sequence[RuleCondition],
    candidate_items: Sequence[RuleCondition],
    *,
    collection: str,
    label: str,
) -> List[RuleConditionDelta]:
    base_by_id, base_order = _index_conditions(base_items, collection=collection)
    candidate_by_id, candidate_order = _index_conditions(
        candidate_items,
        collection=collection,
    )
    ordered_ids = list(base_order)
    ordered_ids.extend(item for item in candidate_order if item not in base_by_id)

    deltas: List[RuleConditionDelta] = []
    for condition_id in ordered_ids:
        before = base_by_id.get(condition_id)
        after = candidate_by_id.get(condition_id)
        if before is None and after is not None:
            operation = "add"
        elif before is not None and after is None:
            operation = "remove"
        elif before is not None and after is not None and before.to_dict() != after.to_dict():
            operation = "replace"
        else:
            continue
        display_id = (after or before).id
        deltas.append(
            RuleConditionDelta(
                delta_id=f"rule:{label}:{_normalized_id(display_id)}:{operation}",
                collection=collection,
                operation=operation,
                condition_id=str(display_id),
                before=before.to_dict() if before is not None else None,
                after=after.to_dict() if after is not None else None,
            )
        )
    return deltas


def _index_conditions(
    items: Sequence[RuleCondition],
    *,
    collection: str,
) -> tuple[Dict[str, RuleCondition], List[str]]:
    indexed: Dict[str, RuleCondition] = {}
    order: List[str] = []
    for raw in list(items or []):
        item = RuleCondition.from_dict(raw)
        condition_id = _normalized_id(item.id)
        if not condition_id:
            raise ValueError(f"{collection} contains an empty condition id")
        if condition_id in indexed:
            raise ValueError(f"{collection} contains duplicate condition id {condition_id}")
        indexed[condition_id] = item
        order.append(condition_id)
    return indexed, order


def materialize_rule_condition_subset(
    *,
    base_rule: EvolvingRule,
    candidate_rule: EvolvingRule,
    deltas: Sequence[RuleConditionDelta],
    accepted_delta_ids: Iterable[str],
    parent_candidate: str,
) -> EvolvingRule:
    accepted = {str(item) for item in accepted_delta_ids}
    known = {delta.delta_id for delta in deltas}
    unknown = accepted - known
    if unknown:
        raise ValueError(f"unknown rule delta ids: {sorted(unknown)}")
    if not accepted:
        raise ValueError("partial candidate must contain at least one rule delta")

    conditions = _materialize_condition_collection(
        base_rule.conditions,
        candidate_rule.conditions,
        deltas=deltas,
        accepted_delta_ids=accepted,
        collection="conditions",
    )
    exclusions = _materialize_condition_collection(
        base_rule.exclusion_conditions,
        candidate_rule.exclusion_conditions,
        deltas=deltas,
        accepted_delta_ids=accepted,
        collection="exclusion_conditions",
    )
    if not conditions:
        raise ValueError("partial candidate cannot remove every positive condition")
    _validate_global_condition_ids(conditions, exclusions)

    base_universe = _condition_universe(
        base_rule.conditions,
        base_rule.exclusion_conditions,
    )
    final_universe = _condition_universe(conditions, exclusions)
    decision_policy = (
        base_rule.decision_policy
        if final_universe == base_universe
        else _strict_decision_policy(conditions, exclusions)
    )
    omitted = [delta.delta_id for delta in deltas if delta.delta_id not in accepted]
    lineage = {
        "stage": RULE_PARTIAL_ACCEPTANCE_STAGE,
        "parent_candidate": str(parent_candidate or "rule_candidate"),
        "base_rule_version": base_rule.version,
        "parent_rule_version": candidate_rule.version,
        "accepted_delta_ids": [
            delta.delta_id for delta in deltas if delta.delta_id in accepted
        ],
        "omitted_delta_ids": omitted,
        "unsupported_parent_fields_ignored": [
            field
            for field, before, after in (
                ("description", base_rule.description, candidate_rule.description),
                ("decision_policy", base_rule.decision_policy, candidate_rule.decision_policy),
            )
            if before != after
        ],
    }
    return base_rule.next_version(
        new_description=base_rule.description,
        new_conditions=conditions,
        new_exclusions=exclusions,
        decision_policy=decision_policy,
        update_note=(
            "Accepted a regression-tested subset of rule condition/exclusion "
            f"deltas from {parent_candidate}: "
            + ", ".join(lineage["accepted_delta_ids"])
        ),
        metadata={"partial_candidate_acceptance": lineage},
    )


def materialize_synthesized_rule_candidate(
    *,
    base_rule: EvolvingRule,
    candidate_rule: EvolvingRule,
    deltas: Sequence[RuleConditionDelta],
    accepted_delta_ids: Iterable[str],
    parent_candidate: str,
    synthesis_strategy: str,
) -> EvolvingRule:
    """Materialize an unvalidated pre-regression delta composition."""
    accepted = {str(item) for item in accepted_delta_ids}
    known = {delta.delta_id for delta in deltas}
    unknown = accepted - known
    if unknown:
        raise ValueError(f"unknown synthesized rule delta ids: {sorted(unknown)}")
    if not accepted:
        raise ValueError("synthesized candidate must contain at least one delta")
    conditions = _materialize_condition_collection(
        base_rule.conditions,
        candidate_rule.conditions,
        deltas=deltas,
        accepted_delta_ids=accepted,
        collection="conditions",
    )
    exclusions = _materialize_condition_collection(
        base_rule.exclusion_conditions,
        candidate_rule.exclusion_conditions,
        deltas=deltas,
        accepted_delta_ids=accepted,
        collection="exclusion_conditions",
    )
    if not conditions:
        raise ValueError("synthesized candidate cannot remove every positive condition")
    _validate_global_condition_ids(conditions, exclusions)
    base_universe = _condition_universe(
        base_rule.conditions,
        base_rule.exclusion_conditions,
    )
    final_universe = _condition_universe(conditions, exclusions)
    decision_policy = (
        base_rule.decision_policy
        if final_universe == base_universe
        else _strict_decision_policy(conditions, exclusions)
    )
    selected = [
        delta.delta_id for delta in deltas if delta.delta_id in accepted
    ]
    lineage = {
        "schema_version": "evotx.rule_delta_synthesis.v1",
        "stage": "pre_runtime_validation",
        "validation_status": "unvalidated",
        "parent_candidate": str(parent_candidate or "rule_candidate"),
        "synthesis_strategy": str(synthesis_strategy or "bounded"),
        "accepted_delta_ids": selected,
        "omitted_delta_ids": [
            delta.delta_id for delta in deltas if delta.delta_id not in accepted
        ],
    }
    return base_rule.next_version(
        new_description=base_rule.description,
        new_conditions=conditions,
        new_exclusions=exclusions,
        decision_policy=decision_policy,
        update_note=(
            "Constructed an unvalidated blocker-aware candidate from atomic "
            f"rule deltas: {', '.join(selected)}"
        ),
        metadata={"rule_delta_synthesis": lineage},
    )


def _materialize_condition_collection(
    base_items: Sequence[RuleCondition],
    candidate_items: Sequence[RuleCondition],
    *,
    deltas: Sequence[RuleConditionDelta],
    accepted_delta_ids: set[str],
    collection: str,
) -> List[RuleCondition]:
    base_by_id, base_order = _index_conditions(base_items, collection=collection)
    candidate_by_id, candidate_order = _index_conditions(
        candidate_items,
        collection=collection,
    )
    delta_by_id = {
        _normalized_id(delta.condition_id): delta
        for delta in deltas
        if delta.collection == collection
    }
    materialized: List[RuleCondition] = []
    for condition_id in base_order:
        delta = delta_by_id.get(condition_id)
        if delta is None or delta.delta_id not in accepted_delta_ids:
            materialized.append(RuleCondition.from_dict(base_by_id[condition_id]))
            continue
        if delta.operation == "remove":
            continue
        replacement = candidate_by_id.get(condition_id)
        if replacement is None:
            raise ValueError(f"accepted {delta.delta_id} has no candidate condition")
        materialized.append(RuleCondition.from_dict(replacement))

    for condition_id in candidate_order:
        if condition_id in base_by_id:
            continue
        delta = delta_by_id.get(condition_id)
        if delta is not None and delta.delta_id in accepted_delta_ids:
            materialized.append(RuleCondition.from_dict(candidate_by_id[condition_id]))
    return materialized


def _validate_global_condition_ids(
    conditions: Sequence[RuleCondition],
    exclusions: Sequence[RuleCondition],
) -> None:
    seen: set[str] = set()
    for item in [*conditions, *exclusions]:
        condition_id = _normalized_id(item.id)
        if condition_id in seen:
            raise ValueError(f"condition id {condition_id} is duplicated across rule collections")
        seen.add(condition_id)


def _condition_universe(
    conditions: Sequence[RuleCondition],
    exclusions: Sequence[RuleCondition],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(_normalized_id(item.id) for item in conditions),
        tuple(_normalized_id(item.id) for item in exclusions),
    )


def _strict_decision_policy(
    conditions: Sequence[RuleCondition],
    exclusions: Sequence[RuleCondition],
) -> str:
    positive_ids = ", ".join(str(item.id) for item in conditions)
    exclusion_ids = ", ".join(str(item.id) for item in exclusions)
    policy = f"Require every positive condition ({positive_ids}) to be satisfied."
    if exclusions:
        policy += f" Reject the target verdict when any exclusion ({exclusion_ids}) is satisfied."
    return policy


def build_rule_condition_subset_plan(
    *,
    base_rule: EvolvingRule,
    candidate_rule: EvolvingRule,
    partial_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
    candidate_plan: EvidencePlan | None,
    deltas: Sequence[RuleConditionDelta],
    accepted_delta_ids: Iterable[str],
    parent_candidate: str,
) -> EvidencePlan:
    accepted = {str(item) for item in accepted_delta_ids}
    baseline = compile_rule_to_baseline_plan(partial_rule)
    baseline_steps = _judge_steps_by_condition(baseline)
    current_steps = _judge_steps_by_condition(current_plan)
    candidate_steps = _judge_steps_by_condition(candidate_plan)
    base_universe = _condition_universe(
        base_rule.conditions,
        base_rule.exclusion_conditions,
    )
    final_universe = _condition_universe(
        partial_rule.conditions,
        partial_rule.exclusion_conditions,
    )
    preserve_current_structure = (
        current_plan is not None and final_universe == base_universe
    )
    changed_targets = {
        delta.target_key
        for delta in deltas
        if delta.delta_id in accepted
    }
    directly_changed_condition_ids = {
        condition_id for _, condition_id in changed_targets
    }
    dependency_affected_ids = (
        dependency_affected_condition_ids(
            current_plan,
            directly_changed_condition_ids,
        )
        | dependency_affected_condition_ids(
            candidate_plan,
            directly_changed_condition_ids,
        )
    ) - directly_changed_condition_ids
    exclusion_ids = {
        _normalized_id(item.id) for item in partial_rule.exclusion_conditions
    }

    judge_steps: List[JudgeStep] = []
    for baseline_step in baseline.judge_steps:
        condition_id = _normalized_id(
            baseline_step.condition_id or baseline_step.id
        )
        collection = (
            "exclusion_conditions"
            if condition_id in exclusion_ids
            else "conditions"
        )
        source = None
        if (
            (collection, condition_id) in changed_targets
            or condition_id in dependency_affected_ids
        ):
            source = candidate_steps.get(condition_id)
            if source is None and condition_id in dependency_affected_ids:
                source = baseline_steps.get(condition_id)
        if source is None:
            source = current_steps.get(condition_id)
        if source is None:
            source = baseline_step
        anchor = (
            current_steps.get(condition_id)
            if preserve_current_structure
            else baseline_step
        ) or baseline_step
        data = copy.deepcopy(source.to_dict())
        data["id"] = anchor.id
        data["condition_id"] = anchor.condition_id or baseline_step.condition_id
        data["expected_answer"] = baseline_step.expected_answer
        judge_steps.append(JudgeStep.from_dict(data))

    metadata = copy.deepcopy(
        (
            current_plan.metadata
            if preserve_current_structure and current_plan
            else baseline.metadata
        )
        or {}
    )
    lineage = copy.deepcopy(
        (partial_rule.metadata or {}).get("partial_candidate_acceptance") or {}
    )
    metadata.update({
        "source": "rule_condition_partial_acceptance",
        "candidate_strategy": RULE_PARTIAL_ACCEPTANCE_STAGE,
        "derived_from_rule_id": partial_rule.rule_id,
        "derived_from_rule_version": partial_rule.version,
        "partial_candidate_acceptance": lineage,
        "parent_candidate": str(parent_candidate or "rule_candidate"),
        "directly_changed_plan_step_condition_ids": sorted(
            directly_changed_condition_ids
        ),
        "dependency_affected_plan_step_condition_ids": sorted(
            dependency_affected_ids
        ),
        "regenerated_plan_step_condition_ids": sorted(
            directly_changed_condition_ids | dependency_affected_ids
        ),
        "preserved_unchanged_plan_steps": sorted(
            set(_judge_steps_by_condition(current_plan))
            - directly_changed_condition_ids
            - dependency_affected_ids
        ),
    })
    return EvidencePlan(
        plan_id=EvidencePlan.new_id(),
        rule_id=partial_rule.rule_id,
        rule_version=partial_rule.version,
        focus_steps=copy.deepcopy(
            current_plan.focus_steps
            if preserve_current_structure and current_plan is not None
            else baseline.focus_steps
        ),
        judge_steps=judge_steps,
        emit_logic=(
            current_plan.emit_logic
            if preserve_current_structure and current_plan is not None
            else baseline.emit_logic
        ),
        plan_note=(
            "Plan materialized from current and parent-candidate judge steps for "
            "rule condition partial acceptance."
        ),
        metadata=metadata,
    )


def _judge_steps_by_condition(
    plan: EvidencePlan | None,
) -> Dict[str, JudgeStep]:
    if plan is None:
        return {}
    return {
        _normalized_id(step.condition_id or step.id): step
        for step in list(plan.judge_steps or [])
        if _normalized_id(step.condition_id or step.id)
    }


def generate_blocker_aware_rule_delta_subsets(
    deltas: Sequence[RuleConditionDelta],
    current_slim_results: Sequence[Dict[str, Any]],
    *,
    joint_resolution_groups: Sequence[Dict[str, Any]] = (),
    max_candidates: int = 4,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build a bounded set of complete candidates from locally valid deltas.

    Local deltas are not required to flip a transaction verdict. This function
    groups compatible deltas around observed blocker profiles; the caller still
    runs the existing full viability, canary, regression, guard, and promotion
    gates on every materialized candidate.
    """
    limit = max(1, int(max_candidates or 1))
    eligible = [
        delta
        for delta in list(deltas or [])
        if delta.local_viability_status == "eligible"
    ]
    rejected = [
        delta.to_dict()
        for delta in list(deltas or [])
        if delta.local_viability_status != "eligible"
    ]
    by_condition: Dict[str, List[RuleConditionDelta]] = {}
    for delta in eligible:
        by_condition.setdefault(_normalized_id(delta.condition_id), []).append(delta)

    proposed: List[Dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add_subset(raw_deltas: Iterable[RuleConditionDelta], strategy: str, **audit) -> None:
        selected = _close_delta_set_over_joint_resolution_groups(
            list(raw_deltas),
            eligible_deltas=eligible,
            joint_resolution_groups=joint_resolution_groups,
        )
        if selected is None:
            return
        selected_ids = tuple(
            delta.delta_id
            for delta in eligible
            if delta.delta_id in {item.delta_id for item in selected}
        )
        if not selected_ids or selected_ids in seen or len(proposed) >= limit:
            return
        seen.add(selected_ids)
        proposed.append({
            "accepted_delta_ids": list(selected_ids),
            "omitted_delta_ids": [
                delta.delta_id for delta in eligible
                if delta.delta_id not in selected_ids
            ],
            "search_strategy": strategy,
            **audit,
        })

    blocker_profiles: set[tuple[str, ...]] = set()
    for result in list(current_slim_results or []):
        if str(result.get("ground_truth") or "").strip().lower() != "attack":
            continue
        if str(result.get("predicted_verdict") or "").strip().lower() == "attack":
            continue
        blockers = tuple(sorted({
            _normalized_id(item.get("condition_id") or item.get("id"))
            for item in list(result.get("condition_table") or [])
            if isinstance(item, dict)
            and (
                (
                    not bool(item.get("is_exclusion"))
                    and not _answer_is_true(item.get("answer"))
                )
                or (
                    bool(item.get("is_exclusion"))
                    and _answer_is_true(item.get("answer"))
                )
            )
            and _normalized_id(item.get("condition_id") or item.get("id"))
        }))
        if blockers:
            blocker_profiles.add(blockers)

    for blockers in sorted(blocker_profiles, key=lambda value: (len(value), value)):
        selected = [
            delta
            for blocker in blockers
            for delta in by_condition.get(blocker, [])
            if delta.semantic_direction in {"broaden", "refine"}
        ]
        covered = {
            _normalized_id(delta.condition_id) for delta in selected
        }
        if set(blockers).issubset(covered):
            add_subset(
                selected,
                "fn_blocker_profile",
                blocker_condition_ids=list(blockers),
            )

    precision_deltas = [
        delta
        for delta in eligible
        if delta.semantic_direction in {"tighten", "refine"}
        and (not delta.source_error_types or "FP" in delta.source_error_types)
    ]
    if precision_deltas:
        add_subset(precision_deltas, "fp_precision_bundle")

    add_subset(eligible, "all_locally_compatible")
    audit = {
        "schema_version": "evotx.rule_delta_synthesis.v1",
        "max_candidates": limit,
        "atomic_delta_count": len(list(deltas or [])),
        "eligible_atomic_delta_count": len(eligible),
        "rejected_atomic_deltas": rejected,
        "blocker_profiles": [list(value) for value in sorted(blocker_profiles)],
        "synthesized_candidate_count": len(proposed),
        "candidates": copy.deepcopy(proposed),
        "safety_boundary": (
            "Synthesis only groups local deltas; every candidate must pass the "
            "existing runtime viability, canary, regression, guard, final "
            "validation, and promotion gates."
        ),
    }
    return proposed, audit


def _close_delta_set_over_joint_resolution_groups(
    selected: Sequence[RuleConditionDelta],
    *,
    eligible_deltas: Sequence[RuleConditionDelta],
    joint_resolution_groups: Sequence[Dict[str, Any]],
) -> List[RuleConditionDelta] | None:
    selected_by_id = {delta.delta_id: delta for delta in selected}
    changed = True
    while changed:
        changed = False
        selected_signals = {
            signal_id
            for delta in selected_by_id.values()
            for signal_id in delta.applied_signal_ids
        }
        for group in list(joint_resolution_groups or []):
            if not isinstance(group, dict):
                continue
            group_ids = {
                str(value) for value in list(group.get("signal_ids") or [])
                if str(value)
            }
            if not selected_signals.intersection(group_ids):
                continue
            covering = [
                delta for delta in eligible_deltas
                if set(delta.applied_signal_ids).intersection(group_ids)
            ]
            covered_signals = {
                signal_id for delta in covering
                for signal_id in delta.applied_signal_ids
                if signal_id in group_ids
            }
            if not group_ids.issubset(covered_signals):
                return None
            for delta in covering:
                if delta.delta_id not in selected_by_id:
                    selected_by_id[delta.delta_id] = delta
                    changed = True
    return list(selected_by_id.values())


def _answer_is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "yes", "1"}


def generate_rule_delta_subsets(
    delta_ids: Sequence[str],
    *,
    max_evaluations: int,
) -> List[Dict[str, Any]]:
    ordered = list(dict.fromkeys(str(item) for item in delta_ids if str(item)))
    total = len(ordered)
    limit = max(0, int(max_evaluations or 0))
    if total < 2 or limit <= 0:
        return []

    generated: List[Dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add_subset(items: Iterable[str], strategy: str) -> None:
        subset_set = set(items)
        subset = tuple(item for item in ordered if item in subset_set)
        if not subset or len(subset) == total or subset in seen or len(generated) >= limit:
            return
        seen.add(subset)
        generated.append({
            "accepted_delta_ids": list(subset),
            "omitted_delta_ids": [item for item in ordered if item not in subset],
            "search_strategy": strategy,
        })

    for omitted in ordered:
        add_subset(
            (item for item in ordered if item != omitted),
            "backward_leave_one_out",
        )
    for item in ordered:
        add_subset([item], "forward_singleton")
    for size in range(2, total - 1):
        for subset in combinations(ordered, size):
            add_subset(subset, f"bounded_combination_{size}")
            if len(generated) >= limit:
                return generated
    return generated
