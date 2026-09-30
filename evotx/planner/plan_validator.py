from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable

from evotx.core.logic import (
    UnsafeLogicExpression,
    extract_logic_names,
    safe_eval_bool_expr,
)
from evotx.core.plan import (
    REENTRANCY_CANDIDATE_STATE_KEY,
    REENTRANCY_STATE_SCHEMA_VERSION,
    REENTRANCY_STATEFUL_RUNTIME_MODE,
    followup_round_policy,
)
from evotx.core.schemas import EvidencePlan, EvolvingRule
from evotx.runtime.view_catalog import PACKET_VIEW_CATALOG


class PlanValidationError(Exception):
    pass


def _normalized_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _string_list(value: Any) -> list[str]:
    return [
        str(item or "").strip()
        for item in list(value or [])
        if str(item or "").strip()
    ]


def _schema_contains(candidate: Any, required: Any) -> bool:
    """Return whether a new producer schema preserves an old shape contract."""
    if isinstance(required, dict):
        if not isinstance(candidate, dict):
            return False
        return all(
            key in candidate and _schema_contains(candidate[key], value)
            for key, value in required.items()
        )
    if isinstance(required, list):
        if not isinstance(candidate, list):
            return False
        if not required:
            return True
        return bool(candidate) and _schema_contains(candidate[0], required[0])
    return type(candidate) is type(required)


def _dependency_snapshot(plan: EvidencePlan) -> Dict[str, Any]:
    steps: list[Dict[str, Any]] = []
    by_step_id: Dict[str, Dict[str, Any]] = {}
    by_condition_id: Dict[str, Dict[str, Any]] = {}
    producer_by_key: Dict[str, Dict[str, Any]] = {}
    duplicate_producer_keys: set[str] = set()
    for index, step in enumerate(list(plan.judge_steps or [])):
        step_id = str(step.id or "").strip()
        condition_id = _normalized_id(step.condition_id or step.id)
        record = {
            "index": index,
            "step_id": step_id,
            "condition_id": condition_id,
            "depends_on": _string_list(step.depends_on),
            "consumes_state_keys": _string_list(step.consumes_state_keys),
            "produces_state_key": str(step.produces_state_key or "").strip(),
            "state_prompt_role": str(step.state_prompt_role or "").strip(),
            "state_output_schema": dict(step.state_output_schema or {}),
        }
        steps.append(record)
        by_step_id[step_id] = record
        by_condition_id[condition_id] = record
        state_key = record["produces_state_key"]
        if state_key:
            if state_key in producer_by_key:
                duplicate_producer_keys.add(state_key)
            else:
                producer_by_key[state_key] = record
    return {
        "steps": steps,
        "by_step_id": by_step_id,
        "by_condition_id": by_condition_id,
        "producer_by_key": producer_by_key,
        "duplicate_producer_keys": duplicate_producer_keys,
    }


def _dependency_edges(snapshot: Dict[str, Any]) -> list[Dict[str, Any]]:
    edge_map: Dict[tuple[str, str], Dict[str, Any]] = {}
    producer_by_key = dict(snapshot.get("producer_by_key") or {})
    for consumer in list(snapshot.get("steps") or []):
        for producer_id in list(consumer.get("depends_on") or []):
            producer = (snapshot.get("by_step_id") or {}).get(producer_id)
            producer_condition = (
                producer.get("condition_id") if producer else _normalized_id(producer_id)
            )
            key = (str(producer_condition), str(consumer.get("condition_id") or ""))
            edge = edge_map.setdefault(key, {
                "producer_condition_id": key[0],
                "consumer_condition_id": key[1],
                "producer_step_id": producer_id,
                "consumer_step_id": consumer.get("step_id"),
                "state_keys": [],
                "sources": [],
            })
            if "depends_on" not in edge["sources"]:
                edge["sources"].append("depends_on")
        for state_key in list(consumer.get("consumes_state_keys") or []):
            producer = producer_by_key.get(state_key)
            if producer is None:
                continue
            key = (
                str(producer.get("condition_id") or ""),
                str(consumer.get("condition_id") or ""),
            )
            edge = edge_map.setdefault(key, {
                "producer_condition_id": key[0],
                "consumer_condition_id": key[1],
                "producer_step_id": producer.get("step_id"),
                "consumer_step_id": consumer.get("step_id"),
                "state_keys": [],
                "sources": [],
            })
            if state_key not in edge["state_keys"]:
                edge["state_keys"].append(state_key)
            if "state_binding" not in edge["sources"]:
                edge["sources"].append("state_binding")
    return sorted(
        edge_map.values(),
        key=lambda item: (
            item["producer_condition_id"],
            item["consumer_condition_id"],
        ),
    )


def dependency_affected_condition_ids(
    plan: EvidencePlan | None,
    changed_condition_ids: Iterable[str],
) -> set[str]:
    """Return transitive consumers affected by changed producer conditions."""
    if plan is None:
        return set()
    changed = {
        _normalized_id(value)
        for value in list(changed_condition_ids or [])
        if _normalized_id(value)
    }
    if not changed:
        return set()
    edges = _dependency_edges(_dependency_snapshot(plan))
    closure = set(changed)
    progressed = True
    while progressed:
        progressed = False
        for edge in edges:
            producer = _normalized_id(edge.get("producer_condition_id"))
            consumer = _normalized_id(edge.get("consumer_condition_id"))
            if producer in closure and consumer and consumer not in closure:
                closure.add(consumer)
                progressed = True
    return closure - changed


def dependency_consistency_report(
    plan: EvidencePlan,
    *,
    previous_plan: EvidencePlan | None = None,
    changed_condition_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """Build a deterministic dependency/state contract audit for one plan."""
    snapshot = _dependency_snapshot(plan)
    previous_snapshot = (
        _dependency_snapshot(previous_plan) if previous_plan is not None else None
    )
    errors: list[Dict[str, Any]] = []
    dangling_count = 0
    mismatch_count = 0
    by_step_id = dict(snapshot.get("by_step_id") or {})
    producer_by_key = dict(snapshot.get("producer_by_key") or {})

    for state_key in sorted(snapshot.get("duplicate_producer_keys") or []):
        mismatch_count += 1
        errors.append({
            "code": "duplicate_state_producer",
            "state_key": state_key,
        })

    adjacency: Dict[str, set[str]] = {
        step_id: set() for step_id in by_step_id
    }
    for record in list(snapshot.get("steps") or []):
        step_id = str(record.get("step_id") or "")
        for dependency_id in list(record.get("depends_on") or []):
            if dependency_id not in by_step_id:
                dangling_count += 1
                errors.append({
                    "code": "dangling_depends_on",
                    "consumer_step_id": step_id,
                    "dependency_step_id": dependency_id,
                })
                continue
            adjacency.setdefault(dependency_id, set()).add(step_id)
            if by_step_id[dependency_id]["index"] >= record["index"]:
                mismatch_count += 1
                errors.append({
                    "code": "dependency_not_executable_in_plan_order",
                    "consumer_step_id": step_id,
                    "dependency_step_id": dependency_id,
                })

        produced_key = str(record.get("produces_state_key") or "")
        output_schema = dict(record.get("state_output_schema") or {})
        if produced_key and produced_key not in output_schema:
            mismatch_count += 1
            errors.append({
                "code": "producer_output_contract_missing",
                "producer_step_id": step_id,
                "state_key": produced_key,
            })
        # Some terminal judges use state_output_schema only as an LLM response
        # contract and intentionally do not publish it into the shared store.

        for state_key in list(record.get("consumes_state_keys") or []):
            producer = producer_by_key.get(state_key)
            if producer is None:
                dangling_count += 1
                errors.append({
                    "code": "missing_state_producer",
                    "consumer_step_id": step_id,
                    "state_key": state_key,
                })
                continue
            producer_id = str(producer.get("step_id") or "")
            if producer["index"] >= record["index"]:
                mismatch_count += 1
                errors.append({
                    "code": "state_producer_not_executable_in_plan_order",
                    "producer_step_id": producer_id,
                    "consumer_step_id": step_id,
                    "state_key": state_key,
                })
            if producer_id not in set(record.get("depends_on") or []):
                mismatch_count += 1
                errors.append({
                    "code": "state_binding_missing_depends_on",
                    "producer_step_id": producer_id,
                    "consumer_step_id": step_id,
                    "state_key": state_key,
                })

    visit_state: Dict[str, int] = {}

    def visit(step_id: str) -> bool:
        state = visit_state.get(step_id, 0)
        if state == 1:
            return True
        if state == 2:
            return False
        visit_state[step_id] = 1
        for consumer_id in adjacency.get(step_id, set()):
            if visit(consumer_id):
                return True
        visit_state[step_id] = 2
        return False

    if any(visit(step_id) for step_id in by_step_id if visit_state.get(step_id, 0) == 0):
        mismatch_count += 1
        errors.append({"code": "dependency_cycle"})

    metadata_edges = dict(
        ((plan.metadata or {}).get("stateful_runtime") or {}).get(
            "dependency_edges", {}
        )
        or {}
    )
    for consumer_id, dependencies in metadata_edges.items():
        consumer = by_step_id.get(str(consumer_id or "").strip())
        if consumer is None:
            dangling_count += 1
            errors.append({
                "code": "metadata_dependency_consumer_missing",
                "consumer_step_id": consumer_id,
            })
            continue
        declared = set(consumer.get("depends_on") or [])
        for producer_id in _string_list(dependencies):
            if producer_id not in by_step_id:
                dangling_count += 1
                errors.append({
                    "code": "metadata_dependency_producer_missing",
                    "consumer_step_id": consumer_id,
                    "producer_step_id": producer_id,
                })
            elif producer_id not in declared:
                mismatch_count += 1
                errors.append({
                    "code": "metadata_dependency_not_materialized",
                    "consumer_step_id": consumer_id,
                    "producer_step_id": producer_id,
                })

    changed = {
        _normalized_id(value)
        for value in list(changed_condition_ids or [])
        if _normalized_id(value)
    }
    producer_role_changed: list[str] = []
    consumer_bindings_changed: list[str] = []
    if previous_snapshot is not None:
        before_by_condition = dict(previous_snapshot.get("by_condition_id") or {})
        after_by_condition = dict(snapshot.get("by_condition_id") or {})
        for condition_id in sorted(set(before_by_condition) | set(after_by_condition)):
            before = before_by_condition.get(condition_id)
            after = after_by_condition.get(condition_id)
            before_producer = bool(before and before.get("produces_state_key"))
            after_producer = bool(after and after.get("produces_state_key"))
            if before_producer or after_producer:
                before_contract = (
                    before.get("produces_state_key"),
                    before.get("state_prompt_role"),
                    before.get("state_output_schema"),
                ) if before else None
                after_contract = (
                    after.get("produces_state_key"),
                    after.get("state_prompt_role"),
                    after.get("state_output_schema"),
                ) if after else None
                if before_contract != after_contract:
                    producer_role_changed.append(condition_id)
            before_binding = (
                tuple(before.get("depends_on") or []),
                tuple(before.get("consumes_state_keys") or []),
            ) if before else None
            after_binding = (
                tuple(after.get("depends_on") or []),
                tuple(after.get("consumes_state_keys") or []),
            ) if after else None
            if before_binding != after_binding:
                consumer_bindings_changed.append(condition_id)

        previous_producers = dict(previous_snapshot.get("producer_by_key") or {})
        for state_key, before_producer in previous_producers.items():
            before_consumers = [
                item for item in list(previous_snapshot.get("steps") or [])
                if state_key in set(item.get("consumes_state_keys") or [])
            ]
            after_consumers = [
                item for item in list(snapshot.get("steps") or [])
                if state_key in set(item.get("consumes_state_keys") or [])
            ]
            after_by_condition = dict(snapshot.get("by_condition_id") or {})
            for before_consumer in before_consumers:
                consumer_condition_id = str(
                    before_consumer.get("condition_id") or ""
                )
                after_consumer = after_by_condition.get(consumer_condition_id)
                if (
                    after_consumer is not None
                    and state_key
                    not in set(after_consumer.get("consumes_state_keys") or [])
                    and consumer_condition_id not in changed
                ):
                    mismatch_count += 1
                    errors.append({
                        "code": "consumer_binding_removed_without_joint_update",
                        "state_key": state_key,
                        "producer_condition_id": before_producer.get(
                            "condition_id"
                        ),
                        "consumer_condition_id": consumer_condition_id,
                    })
            if not after_consumers:
                continue
            after_producer = producer_by_key.get(state_key)
            if after_producer is None:
                continue  # already reported as missing_state_producer
            before_schema = dict(before_producer.get("state_output_schema") or {}).get(
                state_key
            )
            after_schema = dict(after_producer.get("state_output_schema") or {}).get(
                state_key
            )
            all_consumers_jointly_changed = all(
                str(item.get("condition_id") or "") in changed
                for item in after_consumers
            )
            if (
                before_schema is not None
                and not _schema_contains(after_schema, before_schema)
                and not all_consumers_jointly_changed
            ):
                mismatch_count += 1
                errors.append({
                    "code": "producer_consumer_schema_incompatible",
                    "state_key": state_key,
                    "producer_condition_id": after_producer.get("condition_id"),
                    "consumer_condition_ids": sorted({
                        str(item.get("condition_id") or "")
                        for item in after_consumers
                    }),
                })
            if (
                str(before_producer.get("state_prompt_role") or "")
                != str(after_producer.get("state_prompt_role") or "")
                and not all_consumers_jointly_changed
            ):
                mismatch_count += 1
                errors.append({
                    "code": "producer_role_incompatible_with_existing_consumers",
                    "state_key": state_key,
                    "producer_condition_id": after_producer.get("condition_id"),
                    "consumer_condition_ids": sorted({
                        str(item.get("condition_id") or "")
                        for item in after_consumers
                    }),
                })

    before_edges = (
        _dependency_edges(previous_snapshot) if previous_snapshot is not None else []
    )
    after_edges = _dependency_edges(snapshot)
    affected_producers = sorted({
        condition_id
        for condition_id in changed | set(producer_role_changed)
        if (
            condition_id in (snapshot.get("by_condition_id") or {})
            and (snapshot["by_condition_id"][condition_id].get("produces_state_key"))
        )
        or (
            previous_snapshot is not None
            and condition_id in (previous_snapshot.get("by_condition_id") or {})
            and previous_snapshot["by_condition_id"][condition_id].get(
                "produces_state_key"
            )
        )
    })
    closure_source_plan = plan if after_edges else previous_plan
    affected_consumers = sorted(
        dependency_affected_condition_ids(
            closure_source_plan,
            changed | set(affected_producers),
        )
    )
    rejection_reason = ";".join(
        sorted({str(item.get("code") or "") for item in errors if item.get("code")})
    )
    return {
        "schema_version": "evotx.plan_dependency_validation.v1",
        "dependency_edges_before": before_edges,
        "dependency_edges_after": after_edges,
        "affected_producers": affected_producers,
        "affected_consumers": affected_consumers,
        "producer_role_changed": sorted(set(producer_role_changed)),
        "consumer_bindings_changed": sorted(set(consumer_bindings_changed)),
        "dangling_dependency_count": dangling_count,
        "binding_mismatch_count": mismatch_count,
        "dependency_validation_status": "passed" if not errors else "rejected",
        "dependency_rejection_reason": rejection_reason,
        "errors": errors,
    }


class PlanValidator:
    """Validate that an EvidencePlan is executable by the packet-based runtime."""

    def __init__(self, registry=None):
        self.registry = registry
        self.allowed_packet_views = set(PACKET_VIEW_CATALOG)
        self.allowed_followup_tools = {
            "read_packet_view",
            "read_evidence_by_id",
            "read_evidence_context",
            "get_local_call_context",
            "read_function_chunk",
        }

    def validate(self, plan: EvidencePlan, *, attack_label: str = "") -> None:
        if not plan.judge_steps:
            raise PlanValidationError("Plan has no judge steps.")

        judge_ids = [step.id for step in plan.judge_steps]
        judge_id_set = set(judge_ids)
        if len(judge_ids) != len(judge_id_set):
            raise PlanValidationError("Plan has duplicate judge step IDs.")

        known_judges = set()
        for step in plan.judge_steps:
            if not step.question.strip():
                raise PlanValidationError(f"Judge step {step.id} has empty question.")
            condition_id = str(step.condition_id or step.id).upper()
            if (condition_id.startswith("E") or step.id.upper().startswith("E")) and step.expected_answer is not True:
                raise PlanValidationError(
                    f"Exclusion judge step {step.id} must use expected_answer=true."
                )
            for ref in step.evidence_refs:
                if ref not in self.allowed_packet_views:
                    raise PlanValidationError(
                        f"Judge step {step.id} references unknown packet view: {ref}"
                    )
            for ref in step.default_evidence_refs:
                if ref not in self.allowed_packet_views:
                    raise PlanValidationError(
                        f"Judge step {step.id} references unknown default packet view: {ref}"
                    )
            for ref in step.allowed_followup_views:
                if ref not in self.allowed_packet_views:
                    raise PlanValidationError(
                        f"Judge step {step.id} allows unknown follow-up packet view: {ref}"
                    )
            for tool in step.allowed_tools:
                if tool not in self.allowed_followup_tools:
                    raise PlanValidationError(
                        f"Judge step {step.id} allows unknown evidence tool: {tool}"
                    )
            round_policy = followup_round_policy(
                step,
                attack_label=(
                    attack_label
                    or str((plan.metadata or {}).get("attack_label") or "")
                ),
                condition_id=condition_id,
            )
            max_allowed_followups = int(round_policy["max_followups"])
            if step.max_followups < 0 or step.max_followups > max_allowed_followups:
                raise PlanValidationError(
                    f"Judge step {step.id} max_followups must be between 0 and "
                    f"{max_allowed_followups}; two rounds require a semantic "
                    "sequence where round 2 depends on round 1."
                )
            for dep in step.depends_on:
                if dep not in known_judges:
                    raise PlanValidationError(
                        f"Judge step {step.id} depends on unknown or future step: {dep}"
                    )
            known_judges.add(step.id)

        dependency_report = dependency_consistency_report(plan)
        if dependency_report["dependency_validation_status"] != "passed":
            raise PlanValidationError(
                "Plan dependency/state contract is invalid: "
                + dependency_report["dependency_rejection_reason"]
            )

        stateful = dict((plan.metadata or {}).get("stateful_runtime") or {})
        if str(stateful.get("mode") or "") == REENTRANCY_STATEFUL_RUNTIME_MODE:
            self._validate_reentrancy_stateful_binding(plan)

        try:
            names = extract_logic_names(plan.emit_logic)
        except UnsafeLogicExpression as exc:
            raise PlanValidationError(f"Invalid emit_logic: {exc}") from exc

        unknown = names - judge_id_set
        if unknown:
            raise PlanValidationError(
                f"emit_logic references unknown judge IDs: {sorted(unknown)}"
            )
        dummy_values = {jid: True for jid in judge_id_set}
        try:
            safe_eval_bool_expr(plan.emit_logic, dummy_values)
        except Exception as exc:
            raise PlanValidationError(f"Invalid emit_logic: {exc}") from exc

    def validate_against_rule(
        self,
        plan: EvidencePlan,
        rule: EvolvingRule,
        *,
        attack_label: str = "",
        previous_plan: EvidencePlan | None = None,
        changed_condition_ids: Iterable[str] = (),
    ) -> None:
        """Validate the exact Rule/Plan contract after structural evolution."""
        self.validate(plan, attack_label=attack_label)
        if plan.rule_id != rule.rule_id or plan.rule_version != rule.version:
            raise PlanValidationError(
                "Plan rule identity/version does not match candidate Rule: "
                f"plan={plan.rule_id}@{plan.rule_version}, "
                f"rule={rule.rule_id}@{rule.version}."
            )
        positive_ids = {
            str(condition.id).strip().upper()
            for condition in list(rule.conditions or [])
        }
        exclusion_ids = {
            str(condition.id).strip().upper()
            for condition in list(rule.exclusion_conditions or [])
        }
        expected_ids = positive_ids | exclusion_ids
        step_by_condition = {}
        for step in list(plan.judge_steps or []):
            condition_id = str(step.condition_id or step.id).strip().upper()
            if condition_id in step_by_condition:
                raise PlanValidationError(
                    f"Plan has duplicate steps for Rule condition {condition_id}."
                )
            step_by_condition[condition_id] = step
        actual_ids = set(step_by_condition)
        if actual_ids != expected_ids:
            raise PlanValidationError(
                "Plan condition set does not exactly match candidate Rule: "
                f"missing={sorted(expected_ids - actual_ids)}, "
                f"extra={sorted(actual_ids - expected_ids)}."
            )
        for condition in list(rule.conditions or []):
            step = step_by_condition[str(condition.id).strip().upper()]
            if step.expected_answer is not bool(condition.expected_answer):
                raise PlanValidationError(
                    f"Plan step {step.id} expected_answer differs from Rule "
                    f"condition {condition.id}."
                )
        for condition in list(rule.exclusion_conditions or []):
            step = step_by_condition[str(condition.id).strip().upper()]
            if step.expected_answer is not True:
                raise PlanValidationError(
                    f"Plan exclusion step {step.id} must expect true."
                )
        self._validate_strict_rule_emit_semantics(
            plan,
            positive_ids=positive_ids,
            exclusion_ids=exclusion_ids,
            step_by_condition=step_by_condition,
        )
        if previous_plan is not None:
            dependency_report = dependency_consistency_report(
                plan,
                previous_plan=previous_plan,
                changed_condition_ids=changed_condition_ids,
            )
            if dependency_report["dependency_validation_status"] != "passed":
                raise PlanValidationError(
                    "Candidate Plan dependency/state contract is incompatible "
                    "with its predecessor: "
                    + dependency_report["dependency_rejection_reason"]
                )

    @staticmethod
    def dependency_report(
        plan: EvidencePlan,
        *,
        previous_plan: EvidencePlan | None = None,
        changed_condition_ids: Iterable[str] = (),
    ) -> Dict[str, Any]:
        return dependency_consistency_report(
            plan,
            previous_plan=previous_plan,
            changed_condition_ids=changed_condition_ids,
        )

    @staticmethod
    def _validate_strict_rule_emit_semantics(
        plan: EvidencePlan,
        *,
        positive_ids: set[str],
        exclusion_ids: set[str],
        step_by_condition: dict,
    ) -> None:
        ordered_ids = sorted(positive_ids | exclusion_ids)
        step_ids = {
            condition_id: step_by_condition[condition_id].id
            for condition_id in ordered_ids
        }
        if len(ordered_ids) <= 12:
            assignments = product((False, True), repeat=len(ordered_ids))
        else:
            canonical = tuple(
                condition_id in positive_ids for condition_id in ordered_ids
            )
            assignments = [
                canonical,
                *[
                    tuple(
                        (not value if index == changed_index else value)
                        for index, value in enumerate(canonical)
                    )
                    for changed_index in range(len(canonical))
                ],
            ]
        for values in assignments:
            by_condition = dict(zip(ordered_ids, values))
            runtime_values = {
                step_ids[condition_id]: value
                for condition_id, value in by_condition.items()
            }
            expected = all(
                by_condition[condition_id] for condition_id in positive_ids
            ) and not any(
                by_condition[condition_id] for condition_id in exclusion_ids
            )
            actual = bool(safe_eval_bool_expr(plan.emit_logic, runtime_values))
            if actual != expected:
                raise PlanValidationError(
                    "Plan emit_logic is not equivalent to all positive "
                    "conditions AND NOT(any exclusion)."
                )

    @staticmethod
    def _validate_reentrancy_stateful_binding(plan: EvidencePlan) -> None:
        stateful = dict((plan.metadata or {}).get("stateful_runtime") or {})
        if (
            str(stateful.get("state_schema_version") or "")
            != REENTRANCY_STATE_SCHEMA_VERSION
        ):
            raise PlanValidationError(
                "Stateful Reentrancy plan must use state schema "
                f"{REENTRANCY_STATE_SCHEMA_VERSION}; canonicalize the plan "
                "with apply_reentrancy_stateful_runtime before validation."
            )
        by_condition = {
            str(step.condition_id or step.id or "").strip().upper(): step
            for step in plan.judge_steps
        }
        c1 = by_condition.get("C1")
        c2 = by_condition.get("C2")
        c3 = by_condition.get("C3")
        if c1 is None or c2 is None or c3 is None:
            raise PlanValidationError(
                "Stateful Reentrancy plan requires C1, C2, and C3."
            )
        if (
            c1.produces_state_key != REENTRANCY_CANDIDATE_STATE_KEY
            or c1.state_prompt_role != "re_reentry_anchor"
        ):
            raise PlanValidationError(
                "Stateful Reentrancy C1 must produce reentrancy_candidate."
            )
        PlanValidator._validate_reentrancy_schema(
            c1,
            state_key=REENTRANCY_CANDIDATE_STATE_KEY,
            list_key="candidates",
            required_summary_keys={
                "candidates",
                "selected_candidate_ids",
                "unresolved_candidate_ids",
            },
            required_item_keys={
                "candidate_id",
                "outer_call_id",
                "external_edge_id",
                "reentry_call_id",
                "callback_kind",
                "logical_storage_context",
                "outer_function",
                "reentry_function",
                "path_ids",
                "evidence_ids",
                "confidence",
            },
        )
        if (
            c1.id not in c2.depends_on
            or REENTRANCY_CANDIDATE_STATE_KEY not in c2.consumes_state_keys
            or c2.state_prompt_role != "re_value_effect"
        ):
            raise PlanValidationError(
                "Stateful Reentrancy C2 must consume the C1 candidate."
            )
        PlanValidator._validate_reentrancy_schema(
            c2,
            state_key="reentrancy_value_effect_summary",
            list_key="candidate_results",
            required_summary_keys={
                "candidate_results",
                "satisfying_candidate_ids",
                "unresolved_candidate_ids",
            },
            required_item_keys={
                "candidate_id",
                "effect_status",
                "effect_type",
                "same_path_supported",
                "value_or_state_effect",
                "evidence_ids",
            },
        )
        required_c3_dependencies = {c1.id, c2.id}
        required_c3_state = {
            REENTRANCY_CANDIDATE_STATE_KEY,
            "reentrancy_value_effect_summary",
        }
        if (
            not required_c3_dependencies.issubset(set(c3.depends_on))
            or not required_c3_state.issubset(set(c3.consumes_state_keys))
            or c3.state_prompt_role != "re_state_order_causality"
        ):
            raise PlanValidationError(
                "Stateful Reentrancy C3 must consume the same C1/C2 candidate chain."
            )
        PlanValidator._validate_reentrancy_schema(
            c3,
            state_key="reentrancy_causal_order_summary",
            list_key="candidate_results",
            required_summary_keys={
                "candidate_results",
                "attack_candidate_ids",
                "safe_order_candidate_ids",
                "unresolved_candidate_ids",
            },
            required_item_keys={
                "candidate_id",
                "causality_status",
                "mechanism_type",
                "phase_order_pattern",
                "state_not_finalized_before_external_edge",
                "inner_consumption_or_effect_supported",
                "outer_protective_write_after_callback",
                "repeated_sensitive_effect_before_return",
                "cross_function_shared_accounting_supported",
                "read_only_stale_observation_supported",
                "same_candidate_supported",
                "evidence_ids",
            },
        )
        for condition_id, step in by_condition.items():
            if not condition_id.startswith("E"):
                continue
            if (
                not {c1.id, c2.id, c3.id}.issubset(set(step.depends_on))
                or "reentrancy_causal_order_summary"
                not in set(step.consumes_state_keys)
                or step.state_prompt_role != "re_candidate_exclusion"
            ):
                raise PlanValidationError(
                    f"Stateful Reentrancy exclusion {step.id} must assess the "
                    "same completed candidate chain."
                )
            PlanValidator._validate_reentrancy_schema(
                step,
                state_key=str(step.produces_state_key or ""),
                list_key="candidate_results",
                required_summary_keys={
                    "candidate_results",
                    "excluded_candidate_ids",
                    "unresolved_candidate_ids",
                },
                required_item_keys={
                    "candidate_id",
                    "exclusion_status",
                    "same_candidate_supported",
                    "reason",
                    "evidence_ids",
                },
            )

    @staticmethod
    def _validate_reentrancy_schema(
        step,
        *,
        state_key: str,
        list_key: str,
        required_summary_keys: set[str],
        required_item_keys: set[str],
    ) -> None:
        schema = dict(step.state_output_schema or {})
        summary = schema.get(state_key)
        if not isinstance(summary, dict):
            raise PlanValidationError(
                f"Stateful Reentrancy step {step.id} must define "
                f"state_output_schema.{state_key}."
            )
        actual_summary_keys = set(summary)
        if actual_summary_keys != required_summary_keys:
            raise PlanValidationError(
                f"Stateful Reentrancy step {step.id} has invalid {state_key} "
                f"keys: expected={sorted(required_summary_keys)} "
                f"actual={sorted(actual_summary_keys)}"
            )
        rows = summary.get(list_key)
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise PlanValidationError(
                f"Stateful Reentrancy step {step.id} must define a non-empty "
                f"{state_key}.{list_key} item schema."
            )
        actual_item_keys = set(rows[0])
        if actual_item_keys != required_item_keys:
            raise PlanValidationError(
                f"Stateful Reentrancy step {step.id} has invalid "
                f"{state_key}.{list_key} fields: "
                f"expected={sorted(required_item_keys)} "
                f"actual={sorted(actual_item_keys)}"
            )
