from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence

from evotx.core.labels import normalize_attack_label
from evotx.core.plan import (
    apply_insufficient_validation_stateful_runtime,
    apply_market_manipulation_stateful_runtime,
    apply_reentrancy_stateful_runtime,
    apply_token_semantic_stateful_runtime,
    condition_view_budget,
    compile_rule_to_baseline_plan,
    constrain_first_pass_view_refs,
    constrain_judge_step_first_pass,
    followup_round_policy,
    judge_step_view_budget_audit,
    make_local_judge_question,
    neutralize_attack_label_mentions,
    source_dependency_level,
)
from evotx.evolution.diagnosis_routing import (
    review_has_operation,
    review_operation_condition_ids,
)
from evotx.evolution.phase3_search import (
    build_plan_strategy_deltas,
    plan_strategy_safety_violations,
    retain_safe_plan_strategy_operations,
)
from evotx.core.schemas import EvidencePlan, EvolvingRule, JudgeStep
from evotx.evolution.negative_training import (
    negative_case_boundary_guidance,
    negative_training_mode_guidance,
    normalize_negative_training_mode,
)
from evotx.evolution.updater import _compact_review_bundle_for_update
from evotx.planner.plan_validator import PlanValidator
from evotx.runtime.view_catalog import compact_packet_view_catalog
from evotx.utils.fingerprint_utils import (
    plan_fingerprint,
    rule_fingerprint,
    semantic_plan_fingerprint,
)
from evotx.runtime.llm_transcripts import (
    build_truncation_audit,
    estimate_prompt_tokens,
    llm_finish_reason,
    llm_model_name,
    llm_response_text,
    llm_usage,
    write_llm_audit_transcript,
)
from evotx.utils.json_utils import (
    JsonExtractionError,
    extract_json_object,
    extract_last_schema_valid_object,
    repair_likely_mojibake,
    stable_json_dumps,
)


def _completion_exhausted_without_visible_output(llm: Any, completion: Any) -> bool:
    finish_reason = str(llm_finish_reason(llm, completion) or "").strip().lower()
    return (
        not llm_response_text(completion).strip()
        and finish_reason in {"length", "max_tokens", "max_output_tokens"}
    )


def _plan_update_completion_retry_prompt(prompt: str) -> str:
    return (
        str(prompt or "")
        + "\n\nENGINEERING RETRY: The previous response exhausted its output "
        "budget before emitting visible JSON. Repeat exactly the same Plan "
        "update task and signal intent. Emit the required strict JSON object "
        "immediately, without analysis, prose, or markdown. Do not change "
        "signal ids, target scope, or strategy intent.\n"
    )


class PlanUpdater:
    """Update reusable execution-plan strategy from plan-actionable reviews.

    This class intentionally does not let the LLM modify semantic rules or judge
    questions. It may adjust evidence routes and dependencies; a question may
    only be deterministically restored from the current Rule after an explicit
    plan-question-drift diagnosis.
    """

    def __init__(
        self,
        llm=None,
        transcript_dir: str | Path | None = None,
        *,
        enable_route_pruning: bool = False,
        enable_dependency_restructure: bool = False,
        enable_advanced_strategy: bool = False,
    ):
        self.llm = llm
        self.transcript_dir = transcript_dir
        self.enable_route_pruning = bool(enable_route_pruning)
        self.enable_dependency_restructure = bool(
            enable_dependency_restructure
        )
        self.enable_advanced_strategy = bool(enable_advanced_strategy)
        self.last_update_audits: List[Dict[str, Any]] = []

    def update_plan_from_bundle(
        self,
        rule: EvolvingRule,
        review_bundle: Dict[str, Any],
        base_plan: EvidencePlan | None = None,
        *,
        allow_noop: bool = True,
    ) -> EvidencePlan:
        actionable = _plan_actionable_reviews(review_bundle)
        plan = deepcopy(base_plan) if base_plan is not None else compile_rule_to_baseline_plan(rule)
        if not actionable:
            return plan

        original_plan = deepcopy(plan)
        plan, restore_audit = _restore_drifted_questions_from_rule(
            rule,
            plan,
            actionable,
        )
        llm_actionable = [
            review
            for review in actionable
            if _review_requires_plan_strategy_update(review)
        ]
        if not llm_actionable:
            if restore_audit.get("changed_condition_ids"):
                PlanValidator().validate(
                    plan,
                    attack_label=str((rule.metadata or {}).get("attack_label", "")),
                )
                _stamp_plan_metadata(
                    plan,
                    rule,
                    previous=original_plan,
                    source="deterministic_question_restore",
                )
                plan.metadata["deterministic_question_restore"] = restore_audit
                plan.metadata["semantic_plan_changed"] = True
                plan.metadata["scope_guard"] = {
                    "scoped": True,
                    "target_steps": list(restore_audit["changed_condition_ids"]),
                    "allow_emit_logic_change": False,
                    "allow_add_step": False,
                    "allow_remove_step": False,
                }
            return plan
        actionable = llm_actionable
        if restore_audit.get("changed_condition_ids"):
            plan.metadata["deterministic_question_restore"] = restore_audit

        scope = _infer_plan_update_scope(actionable)
        allow_global_update = _allow_global_plan_update(review_bundle, actionable)
        if allow_global_update:
            scope["allow_global_plan_update"] = True
        if not scope.get("scoped") and not allow_global_update:
            plan.metadata = {
                **dict(plan.metadata or {}),
                "scope_guard": scope,
                "semantic_plan_changed": False,
                "plan_update_noop_reason": "unscoped_plan_review",
            }
            return plan

        plan_audit_gate = _plan_evidence_history_gate(
            review_bundle,
            scope,
            actionable,
        )

        if self.llm is None:
            updated = self._fallback_update(rule, plan, actionable, scope=scope)
            violations = _unverified_default_view_promotions(
                plan,
                updated.to_dict(),
                review_bundle,
                scope,
            )
            if violations and self.enable_advanced_strategy:
                demoted_payload = _demote_unverified_default_promotions(
                    updated.to_dict(),
                    violations,
                )
                updated = self._plan_from_payload(
                    rule,
                    plan,
                    demoted_payload,
                    scope=scope,
                )
            strategy_deltas = build_plan_strategy_deltas(plan, updated)
            strategy_violations = plan_strategy_safety_violations(
                strategy_deltas,
                plan_evidence_audit=dict(
                    review_bundle.get("plan_evidence_audit") or {}
                ),
                allow_route_pruning=self.enable_route_pruning,
                allow_dependency_restructure=(
                    self.enable_dependency_restructure
                ),
                require_route_failure_history=self.enable_advanced_strategy,
            )
            if strategy_violations:
                noop_plan = deepcopy(plan)
                noop_plan.metadata = {
                    **dict(noop_plan.metadata or {}),
                    "semantic_plan_changed": False,
                    "plan_update_noop_reason": (
                        "phase3_plan_strategy_safety_rejected"
                    ),
                    "phase3_plan_strategy_violations": strategy_violations,
                }
                return noop_plan
            updated.metadata = {
                **dict(updated.metadata or {}),
                "phase3_plan_strategy_deltas": [
                    delta.to_dict() for delta in strategy_deltas
                ],
            }
            return updated

        prompt_bundle = _compact_review_bundle_for_update(
            review_bundle,
            actionable,
            update_target="plan",
        )
        prompt_actionable = list(prompt_bundle.get("prompt_actionable_reviews") or [])
        prompt = self._build_prompt(
            rule,
            plan,
            prompt_bundle,
            prompt_actionable,
            scope=scope,
            enable_route_pruning=self.enable_route_pruning,
            enable_dependency_restructure=(
                self.enable_dependency_restructure
            ),
            enable_advanced_strategy=self.enable_advanced_strategy,
        )
        raw_completion = ""
        data: Dict[str, Any] | None = None
        completion_retry: Dict[str, Any] = {}
        try:
            raw_completion = self.llm.complete(prompt)
            if _completion_exhausted_without_visible_output(
                self.llm,
                raw_completion,
            ):
                completion_retry = {
                    "attempted": True,
                    "reason": "max_tokens_without_visible_output",
                    "primary_finish_reason": llm_finish_reason(
                        self.llm,
                        raw_completion,
                    ),
                    "primary_visible_chars": len(
                        llm_response_text(raw_completion).strip()
                    ),
                }
                prompt = _plan_update_completion_retry_prompt(prompt)
                raw_completion = self.llm.complete(prompt)
                completion_retry.update({
                    "retry_finish_reason": llm_finish_reason(
                        self.llm,
                        raw_completion,
                    ),
                    "retry_visible_chars": len(
                        llm_response_text(raw_completion).strip()
                    ),
                    "succeeded": not _completion_exhausted_without_visible_output(
                        self.llm,
                        raw_completion,
                    ),
                })
                if not completion_retry["succeeded"]:
                    completion_retry["failure_reason"] = (
                        "updater_completion_exhausted"
                    )
                    raise JsonExtractionError(
                        "PlanUpdater exhausted max_tokens without emitting "
                        "visible JSON after one same-intent retry"
                    )
            raw_data, json_extraction = _extract_plan_update_payload(
                llm_response_text(raw_completion)
            )
            raw_data = repair_likely_mojibake(raw_data)
            raw_data["metadata"] = {
                **dict(raw_data.get("metadata") or {}),
                "plan_update_json_extraction": json_extraction,
                **(
                    {"completion_exhaustion_retry": completion_retry}
                    if completion_retry
                    else {}
                ),
            }
            if _cohort_signal_gate_enabled(review_bundle) and (
                not _plan_payload_uses_actionable_signals(raw_data, review_bundle)
                or _plan_payload_contains_case_specific_terms(raw_data)
            ):
                raw_data["generalization_gate_rejected"] = True
                raw_data["generalization_gate_rejected_reason"] = (
                    "unsupported_signal_ids_or_case_specific_output"
                )
                self._write_audit(
                    rule=rule,
                    plan=plan,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=raw_data,
                    updated_plan=plan,
                    review_bundle=prompt_bundle,
                    actionable_reviews=actionable,
                    scope=scope,
                    semantic_changed=False,
                    allow_noop=allow_noop,
                )
                return plan
            experimental_violations = _experimental_plan_payload_violations(
                raw_data,
                review_bundle,
            )
            if experimental_violations:
                raw_data["experimental_plan_scope_rejected"] = True
                raw_data["experimental_plan_scope_violations"] = (
                    experimental_violations
                )
                self._write_audit(
                    rule=rule,
                    plan=plan,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=raw_data,
                    updated_plan=plan,
                    review_bundle=prompt_bundle,
                    actionable_reviews=actionable,
                    scope=scope,
                    semantic_changed=False,
                    allow_noop=allow_noop,
                )
                return plan
            attack_label = str((rule.metadata or {}).get("attack_label", ""))
            uses_patch_schema = (
                "plan_patches" in raw_data or "step_patches" in raw_data
            )
            data = _candidate_payload_from_plan_update_payload(
                plan,
                raw_data,
                scope,
                attack_label=attack_label,
                rule=rule,
                joint_resolution_groups=list(
                    ((review_bundle or {}).get("update_signal_matrix") or {}).get(
                        "joint_resolution_groups", []
                    )
                    or []
                ),
                enable_advanced_strategy=self.enable_advanced_strategy,
                review_bundle=review_bundle,
            )
            if data.get("plan_patch_materialization_rejected"):
                self._write_audit(
                    rule=rule,
                    plan=plan,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=data,
                    updated_plan=plan,
                    review_bundle=prompt_bundle,
                    actionable_reviews=actionable,
                    scope=scope,
                    semantic_changed=False,
                    allow_noop=allow_noop,
                )
                return plan
            legacy_budget_violations = (
                []
                if uses_patch_schema
                else _full_plan_view_budget_violations(
                    plan,
                    data,
                    scope,
                    attack_label=attack_label,
                )
            )
            if legacy_budget_violations:
                data["plan_view_budget_rejected"] = True
                data["plan_view_budget_rejected_reason"] = (
                    "changed_step_routes_would_be_silently_truncated"
                )
                data["plan_view_budget_violations"] = legacy_budget_violations
                self._write_audit(
                    rule=rule,
                    plan=plan,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=data,
                    updated_plan=plan,
                    review_bundle=prompt_bundle,
                    actionable_reviews=actionable,
                    scope=scope,
                    semantic_changed=False,
                    allow_noop=allow_noop,
                )
                return plan
            attempted_scope_violations = _plan_payload_scope_violations(
                plan,
                data,
                scope,
            )
            if attempted_scope_violations:
                data["raw_plan_scope_audit"] = {
                    "violations": attempted_scope_violations,
                    "enforced": False,
                    "policy": (
                        "restore_raw_out_of_scope_fields_then_validate_"
                        "materialized_plan_diff"
                    ),
                }
            raw_view_promotion_violations = _unverified_default_view_promotions(
                plan,
                data,
                review_bundle,
                scope,
            )
            if raw_view_promotion_violations and self.enable_advanced_strategy:
                data = _demote_unverified_default_promotions(
                    data,
                    raw_view_promotion_violations,
                )
            data = _apply_minimal_plan_scope(
                plan,
                data,
                scope,
                attack_label=attack_label,
            )
            view_promotion_violations = _unverified_default_view_promotions(
                plan,
                data,
                review_bundle,
                scope,
            )
            if view_promotion_violations and self.enable_advanced_strategy:
                data = _demote_unverified_default_promotions(
                    data,
                    view_promotion_violations,
                )
            updated = self._plan_from_payload(rule, plan, data, scope=scope)
            strategy_deltas = build_plan_strategy_deltas(plan, updated)
            strategy_violations = [
                *(
                    _minimal_plan_delta_violations(strategy_deltas)
                    if not self.enable_advanced_strategy
                    else []
                ),
                *plan_strategy_safety_violations(
                strategy_deltas,
                plan_evidence_audit=dict(
                    review_bundle.get("plan_evidence_audit") or {}
                ),
                allow_route_pruning=self.enable_route_pruning,
                allow_dependency_restructure=(
                    self.enable_dependency_restructure
                ),
                require_route_failure_history=self.enable_advanced_strategy,
                ),
            ]
            if strategy_violations:
                partially_accepted, partial_audit = retain_safe_plan_strategy_operations(
                    plan,
                    updated,
                    strategy_violations,
                )
                partially_accepted, joint_partial_audit = (
                    _enforce_plan_joint_groups_after_strategy_safety(
                        plan,
                        partially_accepted,
                        materialization_audit=dict(
                            (data.get("metadata") or {}).get(
                                "plan_patch_materialization"
                            )
                            or {}
                        ),
                        strategy_violations=strategy_violations,
                    )
                )
                partial_audit["joint_resolution_group_safety"] = (
                    joint_partial_audit
                )
                post_safety_budget_audits = _reconstrain_plan_strategy_steps(
                    partially_accepted,
                    base_plan=plan,
                    condition_ids=list(
                        partial_audit.get("retained_condition_ids") or []
                    ),
                    attack_label=attack_label,
                    plan_evidence_audit=dict(
                        review_bundle.get("plan_evidence_audit") or {}
                    ),
                )
                retained_deltas = build_plan_strategy_deltas(plan, partially_accepted)
                remaining_violations = [
                    *(
                        _minimal_plan_delta_violations(retained_deltas)
                        if not self.enable_advanced_strategy
                        else []
                    ),
                    *plan_strategy_safety_violations(
                        retained_deltas,
                        plan_evidence_audit=dict(
                            review_bundle.get("plan_evidence_audit") or {}
                        ),
                        allow_route_pruning=self.enable_route_pruning,
                        allow_dependency_restructure=(
                            self.enable_dependency_restructure
                        ),
                        require_route_failure_history=self.enable_advanced_strategy,
                    ),
                ]
                data["phase3_plan_strategy_operation_filter"] = partial_audit
                partial_audit["post_safety_budget_audits"] = (
                    post_safety_budget_audits
                )
                data["phase3_plan_strategy_violations"] = strategy_violations
                if remaining_violations or not retained_deltas:
                    data["phase3_plan_strategy_rejected"] = True
                    data["phase3_plan_strategy_remaining_violations"] = (
                        remaining_violations
                    )
                    self._write_audit(
                        rule=rule,
                        plan=plan,
                        prompt=prompt,
                        raw_completion=raw_completion,
                        parsed_response=data,
                        updated_plan=plan,
                        review_bundle=prompt_bundle,
                        actionable_reviews=actionable,
                        scope=scope,
                        semantic_changed=False,
                        allow_noop=allow_noop,
                    )
                    return plan
                updated = partially_accepted
                strategy_deltas = retained_deltas
                _refresh_partially_accepted_plan_expectations(data, updated)
                _refresh_strategy_filtered_plan_lineage(
                    data,
                    updated,
                    retained_deltas,
                )
                data["phase3_plan_strategy_rejected"] = False
            updated.metadata = {
                **dict(updated.metadata or {}),
                "phase3_plan_strategy_deltas": [
                    delta.to_dict() for delta in strategy_deltas
                ],
                "phase3_plan_strategy_policy": {
                    "route_replacement": "bounded_by_execution_history",
                    "route_pruning_enabled": self.enable_route_pruning,
                    "dependency_restructure_enabled": (
                        self.enable_dependency_restructure
                    ),
                },
            }
            realization_failures = _plan_patch_realization_failures(updated, data)
            if realization_failures:
                data["plan_patch_realization_failures"] = realization_failures
                realization_resolution = _plan_patch_realization_resolution(
                    data,
                    realization_failures,
                )
                data["plan_patch_realization_resolution"] = realization_resolution
                if not realization_resolution["retain_realized_fields"]:
                    data["plan_patch_realization_rejected"] = True
                    data["plan_patch_realization_rejected_reason"] = (
                        "target_step_patch_was_not_preserved_by_runtime_constraints"
                    )
                    self._write_audit(
                        rule=rule,
                        plan=plan,
                        prompt=prompt,
                        raw_completion=raw_completion,
                        parsed_response=data,
                        updated_plan=plan,
                        review_bundle=prompt_bundle,
                        actionable_reviews=actionable,
                        scope=scope,
                        semantic_changed=False,
                        allow_noop=allow_noop,
                    )
                    noop_plan = deepcopy(plan)
                    noop_plan.metadata = {
                        **dict(noop_plan.metadata or {}),
                        "scope_guard": scope,
                        "semantic_plan_changed": False,
                        "plan_update_noop_reason": "plan_patch_not_realized",
                        "plan_patch_realization_failures": realization_failures,
                    }
                    return noop_plan
                data["plan_patch_realization_rejected"] = False
                data["plan_patch_realization_partial"] = True
                data["plan_patch_realization_dropped_fields"] = list(
                    realization_resolution["dropped_fields"]
                )
                updated.metadata = {
                    **dict(updated.metadata or {}),
                    "plan_patch_realization": realization_resolution,
                }
            if isinstance(updated.metadata, dict):
                updated.metadata.pop("plan_patch_expectations", None)
            scope_violations = _plan_scope_violations(plan, updated, scope)
            if scope_violations:
                data["plan_scope_rejected"] = True
                data["plan_scope_rejected_reason"] = "actual_plan_diff_exceeded_scope"
                data["plan_scope_violations"] = scope_violations
                self._write_audit(
                    rule=rule,
                    plan=plan,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=data,
                    updated_plan=plan,
                    review_bundle=prompt_bundle,
                    actionable_reviews=actionable,
                    scope=scope,
                    semantic_changed=False,
                    allow_noop=allow_noop,
                )
                return plan
            # Partial patch materialization already validates the merged raw
            # Plan. Validate the fully constrained Plan again because runtime
            # normalization can change budgets and state contracts.
            PlanValidator().validate_against_rule(
                updated,
                rule,
                attack_label=attack_label,
            )
            if semantic_plan_fingerprint(updated) == semantic_plan_fingerprint(plan):
                self._write_audit(
                    rule=rule,
                    plan=plan,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=data,
                    updated_plan=plan,
                    review_bundle=prompt_bundle,
                    actionable_reviews=actionable,
                    scope=scope,
                    semantic_changed=False,
                    allow_noop=allow_noop,
                )
                if allow_noop:
                    noop_plan = deepcopy(plan)
                    noop_plan.metadata = {
                        **dict(noop_plan.metadata or {}),
                        "scope_guard": scope,
                        "semantic_plan_changed": False,
                        "plan_update_noop_reason": "no_semantic_plan_change",
                    }
                    return noop_plan
            _stamp_plan_metadata(updated, rule, previous=plan, source="plan_update")
            updated.metadata["scope_guard"] = scope
            updated.metadata["semantic_plan_changed"] = True
            self._write_audit(
                rule=rule,
                plan=plan,
                prompt=prompt,
                raw_completion=raw_completion,
                parsed_response=data,
                updated_plan=updated,
                review_bundle=prompt_bundle,
                actionable_reviews=actionable,
                scope=scope,
                semantic_changed=True,
                allow_noop=allow_noop,
            )
            return updated
        except Exception as exc:
            update_generation_failure_reason = (
                str(completion_retry.get("failure_reason") or "")
                or (
                    "updater_json_extraction_failed"
                    if isinstance(exc, JsonExtractionError)
                    else ""
                )
            )
            self._write_audit(
                rule=rule,
                plan=plan,
                prompt=prompt,
                raw_completion=raw_completion,
                parsed_response=data,
                updated_plan=None,
                review_bundle=prompt_bundle,
                actionable_reviews=actionable,
                scope=scope,
                semantic_changed=False,
                allow_noop=allow_noop,
                error=repr(exc),
                completion_retry=completion_retry,
                update_generation_failure_reason=(
                    update_generation_failure_reason
                ),
            )
            if allow_noop:
                return plan
            raise

    def _write_audit(
        self,
        *,
        rule: EvolvingRule,
        plan: EvidencePlan,
        prompt: str,
        raw_completion: Any,
        parsed_response: Any,
        updated_plan: EvidencePlan | None,
        review_bundle: Dict[str, Any],
        actionable_reviews: List[Dict[str, Any]],
        scope: Dict[str, Any],
        semantic_changed: bool,
        allow_noop: bool,
        error: str = "",
        completion_retry: Dict[str, Any] | None = None,
        update_generation_failure_reason: str = "",
    ) -> None:
        attack_label = str((rule.metadata or {}).get("attack_label") or "")
        retry_audit = dict(completion_retry or {})
        if not retry_audit and isinstance(parsed_response, dict):
            retry_audit = dict(
                (parsed_response.get("metadata") or {}).get(
                    "completion_exhaustion_retry"
                )
                or {}
            )
        bundle_chars = len(stable_json_dumps(review_bundle))
        truncation = build_truncation_audit(
            input_name="review_bundle",
            before=review_bundle,
            after=review_bundle,
        )
        plan_summary = _plan_update_audit_summary(plan, updated_plan)
        self.last_update_audits.append({
            "parsed_response": deepcopy(parsed_response)
            if isinstance(parsed_response, dict)
            else {},
            "updated_plan_summary": deepcopy(plan_summary),
            "scope_guard": deepcopy(scope or {}),
            "semantic_changed": bool(semantic_changed),
            "error": str(error or ""),
            "completion_retry": deepcopy(retry_audit),
            "completion_failure_reason": str(
                update_generation_failure_reason
                or retry_audit.get("failure_reason")
                or ""
            ),
        })
        name = f"{attack_label or rule.rule_id}__plan_update__{plan.plan_id}"
        path = write_llm_audit_transcript(
            output_dir=self.transcript_dir,
            stage="plan_updater",
            name=name,
            model=llm_model_name(self.llm),
            prompt=prompt,
            raw_completion=raw_completion,
            parsed_response={
                "parsed_response": parsed_response,
                "updated_plan_summary": plan_summary,
            },
            usage=llm_usage(self.llm, raw_completion),
            finish_reason=llm_finish_reason(self.llm, raw_completion),
            metadata={
                "attack_label": attack_label,
                "current_rule_id": rule.rule_id,
                "current_rule_version": rule.version,
                "base_plan_id": plan.plan_id,
                "review_count": len(list((review_bundle or {}).get("actionable_rule_reviews", []) or []))
                + len(list((review_bundle or {}).get("non_rule_reviews", []) or [])),
                "plan_actionable_count": len(list(actionable_reviews or [])),
                "review_bundle_chars": bundle_chars,
                "prompt_chars": len(str(prompt or "")),
                "scope_guard": dict(scope or {}),
                "target_steps": list((scope or {}).get("target_steps", []) or []),
                "semantic_changed": bool(semantic_changed),
                "allow_noop": bool(allow_noop),
                "error": error,
                "completion_retry": retry_audit,
                "update_generation_failure_reason": str(
                    update_generation_failure_reason or ""
                ),
                **plan_summary,
            },
            truncation=truncation,
        )
        if self.transcript_dir is not None:
            print(
                "[LLM-Audit] plan_updater "
                f"label={attack_label or 'unknown'} "
                f"prompt_chars={len(str(prompt or ''))} "
                f"est_tokens={estimate_prompt_tokens(prompt)} "
                f"bundle_chars={bundle_chars} "
                f"truncated={str(bool(truncation.get('applied'))).lower()} "
                f"path={path or ''}"
            )

    @staticmethod
    def _build_prompt(
        rule: EvolvingRule,
        plan: EvidencePlan,
        review_bundle: Dict[str, Any],
        actionable_reviews: List[Dict[str, Any]],
        *,
        scope: Dict[str, Any] | None = None,
        enable_route_pruning: bool = False,
        enable_dependency_restructure: bool = False,
        enable_advanced_strategy: bool = False,
    ) -> str:
        negative_training_mode = normalize_negative_training_mode(
            review_bundle.get("negative_training_mode", "mixed")
        )
        negative_training_guidance = negative_training_mode_guidance(
            negative_training_mode
        )
        negative_case_guidance = negative_case_boundary_guidance(
            negative_training_mode
        )
        case_boundary_context = dict(
            review_bundle.get("case_boundary_context") or {}
        )
        active_refinement_feedback = dict(
            review_bundle.get("rejected_update_memory") or {}
        ).get("active_refinement_feedback") or {}
        plan_context = _plan_context_for_prompt(plan, scope or {})
        strategy_operations = (
            "expand_evidence_route|replace_evidence_route|"
            "reorder_evidence_route|demote_default_to_followup|"
            "prune_evidence_route|change_verification_granularity|"
            "change_followup_budget|change_dependency_route"
            if enable_advanced_strategy
            else "expand_evidence_route|replace_evidence_route|change_followup_budget"
        )
        strategy_guidance = (
            """
- Advanced bounded strategy is enabled. Reorder, demotion, granularity, and
  route-history-backed replacement remain subject to scope, dependency, and
  full Plan validation. Dependency rewiring and pruning still require their
  separate feature flags.
"""
            if enable_advanced_strategy
            else """
- Use only a local add/replace of initial evidence, a local add/replace of
  follow-up evidence, or a necessary max_followups adjustment.
- Do not propose route reordering, default demotion, route pruning,
  verification-granularity exploration, or dependency changes.
- A supported Plan signal may place an existing registered Packet view in
  default evidence so the complete train/guard validation can test it directly.
- In the minimal pipeline, expand_evidence_route may add an optional follow-up,
  but replace_evidence_route must put one newly added replacement route in
  default evidence. The materializer deterministically moves the previous
  lowest-priority default into follow-up when the default budget is full.
"""
        )
        return f"""
You are updating only the EvoTx evidence execution plan.

You are not updating the semantic detection rule. Do not rewrite rule
conditions, exclusions, attack labels, or decision policy. Only modify the plan
strategy used to gather and present evidence to local judge steps.

Allowed plan changes:
- change default_evidence_refs;
- change allowed_followup_views;
- change allowed_tools among read_packet_view, read_evidence_by_id,
  read_evidence_context, get_local_call_context, read_function_chunk;
- change max_followups between 0 and 1 for ordinary evidence expansion;
- use max_followups=2 only when round 2 depends on round 1, such as precise-row
  then source/function lookup, a positive reentrancy state-order row followed
  by local context, or access_control/insufficient_validation core evidence
  followed by semantic verification. Wanting two independent views is not a
  sequential dependency because one round may request up to two tools;
- add a concise plan_note/update_note.

State-order planning guidance:
- for stale-state/reentrancy/accounting-order questions, prefer compact
  round-0 locating views such as operation_summary_view,
  evidence_adequacy_view, reentrancy_candidate_catalog_view,
  reentrancy_state_order_summary_view,
  trace_outline_view, critical_call_view, and value_release_view;
- keep round-0 evidence narrow for every condition: ordinary
  default_evidence_refs <=4 views; reentrancy/source-required/state-order
  exception <=5;
  access_control/insufficient_validation/protocol_accounting core exception <=6 compact views;
  no more than one large view in ordinary first-pass; Reentrancy may use two
  when critical_call_view and another value/state locating view are both
  required by the local condition;
- use default_evidence_refs for compact locating/core evidence only. Put broad
  trace/state/event/fundflow detail in allowed_followup_views so the judge can
  fetch full rows only when the local condition needs them;
- ordinary allowed_followup_views <=5; source-required <=7; state-order <=6;
  access_control/insufficient_validation core <=6; protocol_accounting core <=8;
- keep full reentrancy_state_order_view available as an allowed follow-up view
  when extra ordering rows may be needed after prompt rendering;
- keep critical_call_argument_view available as an early follow-up after a
  concrete critical call is located when decoded callback target, actor,
  beneficiary, amount, or call parameters can resolve the local question. Do
  not use arguments alone as proof of stale-state ordering;
- prefer get_local_call_context for one concrete reentrant call id over adding
  every broad trace/state/fundflow view to the first-pass prompt.
- for reentrancy, preserve one candidate chain across affected core questions:
  C1 locates outer call -> external edge -> re-entry path and logical
  storage/accounting domain; C2 verifies a value-relevant effect on that same
  candidate; C3 verifies phase-correct state-order causality on that same
  candidate. Do not route separate conditions to unrelated callbacks, profits,
  value flows, or storage overlap;
- an SSTORE completed before the external edge is immediately visible to
  nested EVM execution. Treat state_updated_before_external_edge as a safe-order
  signal, not stale-state proof. Prefer an outer-read -> external-edge ->
  inner-read/effect -> outer-protective-write-after-callback witness, or an
  equivalent read-only/accounting stale-observation pattern;

Source-dependent planning guidance:
- if a local question depends on authorization checks, validation checks,
  selector meaning, storage-slot semantics, return-value handling, or
  source-level accounting formulas, keep round-0 locating evidence compact and
  allow two constrained follow-ups;
- read_function_chunk is only for source-required questions, not merely
  source-helpful questions;
- include read_evidence_by_id so the judge can first fetch a precise call/event/
  state row by evidence_id before requesting read_function_chunk;
- do not use call:0 or transaction-root targets when a concrete critical,
  unknown-selector, value-release, or child call row is available.

Access-control binding guidance:
- preserve one shared authorization chain across the affected core questions:
  actor/caller or beneficiary -> required authority -> sensitive capability and
  protected target/resource -> protected effect;
- preserve both the externally reachable entry-side parent and its trace-linked
  downstream sensitive-operation child in the C1 candidate. A Plan update must
  not narrow the candidate to the child merely because source shows a modifier;
  keep trace/call-argument routes capable of testing how the parent reaches or
  configures that guarded child;
- make each local question name enough of its chain role that unrelated calls,
  actors, assets, positions, or protocol components cannot satisfy different
  conditions independently;
- prefer overlapping evidence ids or explicit trace, proxy/delegatecall,
  callback, beneficiary/controller, and value-release links between roles;
- source unavailability may change the evidence strategy, but it must neither
  force uncertainty when packet evidence connects the chain nor permit generic
  suspicious behavior to replace the chain.
- for source-unavailable or selector-uncertain authorization-gap questions,
  prefer source_unavailable_auth_view as compact first-pass or early follow-up
  evidence. It must be filtered/projected by C1 candidate evidence_ids/path_ids
  and interpreted only as observable call-tree/storage context, not proof that
  no require/branch check exists.
- for access_control C2 authorization-gap plans, preserve a candidate-specific
  path for modifier/provenance checks: critical_call_argument_view,
  source_unavailable_auth_view, semantic_state_delta_view, state_change_view,
  beneficiary_controller_view, and value_release_view. If source shows a guard
  such as allowed(from), approval, allowance, permit, whitelist, owner, or role,
  the plan should still allow follow-up evidence that distinguishes legitimate
  pre-existing authority from self-granted or attacker-controlled authority on
  the same candidate chain.
- C2 evidence routing must support both guard presence and authorization-source
  provenance: owner/admin, role manager, module, whitelist, signature or
  delegation, callback sender, and same-transaction permission/configuration
  mutation. Do not treat a downstream modifier as conclusive when the same
  actor can select the module, mutate the role/whitelist, self-authorize, or
  enter through a parent path that changes the effective principal.

Insufficient-validation binding guidance:
- preserve C1 routes for user-controlled, externally derived, computed, and
  protocol-internal business/accounting-state candidates. A plan patch must not
  narrow evidence acquisition to one object class because one false positive
  used another class;
- route C2 to semantic content/source/freshness/range/invariant checks on the
  same selected object. Generic caller/owner/role/whitelist/callback-sender
  authorization belongs to E1 unless supplied authority data or signer/domain
  content is itself the selected object;
- when source is unavailable, keep evidence routes that can positively show an
  invalid, stale, inconsistent, adversarial, or boundary-breaking object being
  accepted and consumed. Do not route around source failure by treating the
  absence of a visible trace check as proof that validation was missing;
- route C3 to a concrete same-object outcome and E2 to an independently
  sufficient mechanism. Generic profit/swaps/value release are outcome context,
  not substitutes for object binding or validation-gap evidence.

Price-manipulation evidence guidance:
- keep price_manipulation separate from market_manipulation. Skim/donate,
  sandwich/MEV ordering, pool-accounting anomalies, swaps, flash capital, and
  profit are not price evidence unless a distorted reserve/price is actually
  consumed;
- a C2 consumer may be a same-market swap, settlement, mint/burn, or extraction
  when the evidence shows that its amount/outcome depends on the distorted
  reserve/price. Do not require a distinct external protocol, but do not accept
  bare skim/sync/reserve writing or operation co-occurrence. The actor's
  ordinary sequential swaps merely repricing against normal reserve updates
  are also insufficient unless an independently abnormal distortion is tied
  to the later amount, limit, settlement, or extraction;
- do not promote the entire critical_call_view into an unfiltered C2 route.
  First use price_state profiles or a price-read link to identify concrete
  consumer evidence IDs, then inspect projected critical-call rows or targeted
  local context;
- price_manipulation C2 may use max_followups=2 only as a dependent sequence:
  round 1 locates a candidate consumer, and round 2 decodes that same call's
  arguments, state context, or source. C3 may similarly inspect outcome first
  and then attribute it to the selected distorted price. C1 and exclusions
  remain ordinary one-round routes unless another shared policy applies;
- price manipulation does not require a stateful producer/consumer binding.
  Do not invent one merely to repair evidence routing.

Market-manipulation binding guidance:
- preserve one shared market-mechanism profile across affected core questions:
  C1 locates a market source/profile, C2 verifies consumption of that same
  profile, and C3 verifies profile-specific outcome. Plan updates may improve
  evidence acquisition but must not make generic profit, swap volume, flash
  capital, or value release sufficient by themselves;
- for market_manipulation C1/C2 plans, prefer market_mechanism_profile_view as
  compact first-pass evidence with price_relevant_state_view,
  amm_reserve_transition_view, critical_call_view, and event/source follow-up
  as needed. Use it to keep the same market_source/profile_id across C1/C2/C3;
- for market_manipulation exclusions, keep token_semantic_delta_summary_view,
  token_accounting_origin_view, reentrancy_candidate_catalog_view,
  reentrancy_state_order_summary_view,
  source_unavailable_auth_view, and protocol/accounting/value views available
  only to decide whether another primary root cause replaces the same market
  profile, not merely co-occurs.

Protocol-accounting binding guidance:
- preserve one shared bookkeeping candidate across affected core questions:
  C1 locates protocol-internal accounting state/formula/order, C2 verifies a
  protocol operation consuming or relying on that same candidate, and C3
  verifies candidate-specific abnormal payout/share/debt/collateral/vault
  outcome;
- for protocol_accounting C1/C2 plans, keep critical_call_view,
  critical_call_argument_view, trace_outline_view, semantic_state_delta_view,
  event_view, state_change_view, and read_function_chunk available enough to
  locate the concrete function/accounting formula after packet evidence finds
  the candidate;
- for protocol_accounting E2 plans, keep token_semantic_delta_summary_view,
  token_accounting_origin_view, reentrancy_state_order_summary_view,
  price_relevant_state_view, and amm_reserve_transition_view available so the
  judge can decide whether another mechanism fully replaces the accounting
  candidate rather than merely co-occurs.

Forbidden changes:
- changing judge question text. Question semantics are owned by the Rule/Plan
  generation boundary; Plan updates may only change evidence acquisition;
- concrete tx hashes, addresses, call ids, transfer ids, event ids, or slots;
- attack/benign/suspicious conclusions in the plan;
- negative-family-specific routing or view selection. The round-level
  negative_training_mode is dataset coverage only and must not become a
  shortcut for judging individual transactions;
- packet view names inside the rule;
- changing emit_logic semantics except preserving consistency with existing
  judge step ids.
- Treat each canonical Plan signal as a hypothesis, not a direct edit instruction. Apply
  only signal_ids listed in update_signal_matrix.plan_actionable_signal_ids and
  copy those ids into top-level applied_signal_ids. A signal whose status is
  experimental_plan may change only evidence routing fields (default/follow-up
  views, allowed read-only tools, or max_followups) on its target step; it may
  not change question semantics, dependencies, emit logic, or Rule content.
- When an experimental signal includes deterministic experimental_probe_routes,
  use only those registered routes. For additive or granularity probes, preserve
  the existing default route and add or promote only the named probe route.
- Merge compatible canonical Plan signals by semantic evidence need and failure
  mode while preserving every contributing signal_id in applied_signal_ids.
- update_signal_matrix.joint_resolution_groups are atomic synthesis contracts.
  If a plan patch consumes one signal in a group, applied_signal_ids must contain
  every signal in that group and the patch must preserve both evidence needs.
- Correct target positives define evidence-strategy preservation constraints.
  Do not make source code, selector identity, or one packet view mandatory when
  correct positives satisfy the condition through other target-specific packet
  evidence.
- Correct negatives and guard cases protect existing target boundaries, but do
  not justify naming or routing around a particular negative family.
- Case boundary context is the compact preservation contract from correct cases.
  A plan patch may add or refine evidence routes for the error, but it must not
  remove the views/tools/state bindings that kept correct positives detectable
  or correct negatives/guard cases outside the target family.
- Do not let an error-centric plan repair override case_boundary_context. If a
  requested evidence route conflicts with protected correct-case boundaries,
  leave that step unchanged or add the route only as a bounded follow-up.
- active_refinement_feedback is not transaction evidence and not an eligibility
  gate. Use only its failure reason and protected attribution to refine the
  current canonical signal. Exact duplicates are handled downstream.
- Never add CTF/challenge detection, playful-selector logic, contract/protocol
  names, presumed contract purpose, or transaction-specific behavior to a judge
  question. Unknown selectors are evidence limitations, never evidence that an
  interaction is intended or benign.
- Use plan_evidence_audit as the execution-history record. It shows which
  views/tools were actually tried, whether prompts or packets were truncated,
  whether missing evidence was resolved, and whether follow-up changed the
  local answer. Do not infer a plan defect from cohort counts alone.
- Plan can only change how existing Packet views and registered read-only tools
  are selected or ordered. If the update signal says the required witness
  capability is absent from the current Packet/runtime capability surface, do
  not create a Plan-only replacement/follow-up loop; leave the step unchanged
  and preserve the Packet/Runtime owner signal for engineering actionability.
{strategy_guidance}
- Before adding any view, inspect the current plan_context and
  plan_evidence_audit. If that view is already present in the same step's
  default_evidence_refs, evidence_refs, or allowed_followup_views, do not emit
  an add-view patch for it. Prefer changing priority/guidance, max_followups,
  or the allowed evidence routing only when the error shows that the existing
  view was not actually used well.
- If the attack_label is token_semantic_exploitation, preserve same-candidate
  stateful fields unless the target condition itself is being changed:
  produces_state_key, consumes_state_keys, state_prompt_role, and
  state_output_schema. Token Semantic plans should use
  token_semantic_delta_summary_view and token_accounting_origin_view to
  distinguish token-contract semantics from protocol-internal accounting.
- If the attack_label is market_manipulation, preserve same-profile stateful
  fields unless the target condition itself is being changed:
  produces_state_key, consumes_state_keys, state_prompt_role, and
  state_output_schema. Market plans should use
  market_mechanism_profile_view to bind market source, consumption, and outcome
  before using generic economic outcome views.
- If the attack_label is flashloans, preserve same-capital-chain stateful fields
  unless the target condition itself is being changed: produces_state_key,
  consumes_state_keys, state_prompt_role, and state_output_schema. Flashloans
  plans should preserve the C1 atomic-capital anchor and keep existing C1
  structural defaults unless an observed successful follow-up proves a safer
  replacement.
- If the attack_label is reentrancy, preserve same-candidate stateful fields
  unless the target condition itself is being changed: produces_state_key,
  consumes_state_keys, state_prompt_role, and state_output_schema. C2/C3 and
  exclusions must continue to consume the C1 reentrancy_candidate.

Phase 3 bounded restructuring feature flags:
- route_pruning_enabled={str(bool(enable_route_pruning)).lower()}
- dependency_restructure_enabled={str(bool(enable_dependency_restructure)).lower()}
- A disabled operation must not appear in plan_patches. Enabling an operation
  does not bypass execution-history, dependency, scope, or validation gates.

Current semantic rule:
{stable_json_dumps(rule.to_dict())}

Current plan:
{stable_json_dumps(plan_context)}

Compact packet view catalog (all available views):
{stable_json_dumps(compact_packet_view_catalog())}

Preservation boundary context:
{stable_json_dumps(case_boundary_context)}

Active refinement feedback:
{stable_json_dumps(active_refinement_feedback)}

Plan execution evidence audit:
{stable_json_dumps(review_bundle.get("plan_evidence_audit", {}))}

Canonical update signals:
{stable_json_dumps(review_bundle.get("update_signal_matrix", {}))}

Negative training context:
- mode="{negative_training_mode}"
- {negative_training_guidance}
- {negative_case_guidance}

Return strict JSON using this PATCH schema. Do not return the full plan.
Return exactly one JSON object. Do not include comments, trailing commas,
ellipsis placeholders, markdown outside JSON, or extra keys. The runtime first
tries strict schema extraction; lightweight JSON repair is only a fallback and
the repaired payload must still pass schema/materialization validation.
{{
  "applied_signal_ids": ["signal_001"],
  "plan_id": "{plan.plan_id}",
  "rule_id": "{rule.rule_id}",
  "rule_version": {rule.version},
  "plan_patches": [
    {{
      "condition_id": "C1",
      "applied_signal_ids": ["signal_001"],
      "strategy_operation": "{strategy_operations}",
      "rationale": "why this one judge step needs an evidence-strategy change",
      "expected_effect": "which missing observation becomes reachable and how it should resolve the supported failure signal",
      "set_fields": {{
        "default_evidence_refs": ["operation_summary_view"],
        "allowed_followup_views": ["critical_call_view"],
        "allowed_tools": ["read_packet_view"],
        "max_followups": 1,
        "depends_on": []
      }},
      "add_to_fields": {{
        "allowed_followup_views": ["critical_call_view"],
        "allowed_tools": ["read_evidence_by_id"]
      }},
      "remove_from_fields": {{
        "allowed_tools": ["read_function_chunk"]
      }}
    }}
  ],
  "plan_note": "what changed and why",
  "metadata": {{
    "source": "plan_update",
    "update_note": "short summary"
  }}
}}

Patch rules:
- Only include condition_ids granted by the supported plan signals.
- Omit a field when it should remain unchanged.
- Do not include question, expected_answer, state_prompt_role,
  state_output_schema, consumes_state_keys, produces_state_key, or emit_logic.
- Every out-of-scope step is an exact preservation anchor. A targeted step is
  not: use set_fields to provide the complete final prioritized list, or use
  remove_from_fields to evict a lower-value route when the evidence budget is
  already full.
- A set_fields array is replacement for non-expand operations. For
  expand_evidence_route it is normalized as an additive update: existing
  routes, tools, and dependencies remain active and newly requested values are
  appended. Use replace_evidence_route when an existing route must be removed.
  Runtime will verify that the final target step realizes the supplied patch
  after label-specific constraints are reapplied.
- If adding a follow-up View would exceed the target step's active View budget,
  do not use add_to_fields and rely on list order. Either return a no-op or use
  set_fields.allowed_followup_views to explicitly rank the complete desired
  list. Runtime will reject a patch when a newly added View would remain
  inactive after budgeting; it will never silently accept a truncated route.
- If the candidate's effect depends on a new evidence route, make that route
  reachable as bounded default evidence for replace_evidence_route, or through
  allowed_followup_views with read_packet_view and a positive max_followups
  budget for expand_evidence_route. Runtime validation will verify whether a
  follow-up route was actually selected before attributing any improvement to
  the Plan. Replace an old route only when
  plan_evidence_audit supplies the required failure or non-contribution history;
  otherwise prefer a bounded follow-up.
- If no safe plan patch exists, return an empty plan_patches array with the
  supported applied_signal_ids and explain the no-op in plan_note.

For backward compatibility only, the runtime may still accept an older full-plan
payload, but your response must use plan_patches.

Allowed patched fields match this shape:
{{
  "condition_id": "C1",
  "default_evidence_refs": ["..."],
  "allowed_followup_views": ["..."],
  "allowed_tools": ["read_packet_view"],
  "max_followups": 1,
  "depends_on": []
}}

Current step ids and condition ids:
{stable_json_dumps([
    {
        "id": step.id,
        "condition_id": step.condition_id or step.id,
        "default_evidence_refs": list(step.default_evidence_refs or step.evidence_refs or []),
        "allowed_followup_views": list(step.allowed_followup_views or []),
        "allowed_tools": list(step.allowed_tools or []),
        "max_followups": int(step.max_followups or 0),
        "depends_on": list(step.depends_on or []),
    }
    for step in list(plan.judge_steps or [])
])}
"""

    @staticmethod
    def _plan_from_payload(
        rule: EvolvingRule,
        previous: EvidencePlan,
        data: Dict[str, Any],
        *,
        scope: Dict[str, Any] | None = None,
    ) -> EvidencePlan:
        payload = {
            "plan_id": str(data.get("plan_id") or previous.plan_id),
            "rule_id": str(data.get("rule_id") or rule.rule_id),
            "rule_version": int(data.get("rule_version") or rule.version),
            "focus_steps": data.get("focus_steps", previous.to_dict().get("focus_steps", [])),
            "judge_steps": data.get("judge_steps", previous.to_dict().get("judge_steps", [])),
            "emit_logic": str(data.get("emit_logic") or previous.emit_logic),
            "plan_note": str(data.get("plan_note") or "Plan strategy updated from plan-actionable reviews."),
            "metadata": {
                **dict(previous.metadata or {}),
                **dict(data.get("metadata", {}) or {}),
            },
        }
        updated = EvidencePlan.from_dict(payload)
        target_steps = {
            str(item or "").strip().upper()
            for item in list((scope or {}).get("target_steps", []) or [])
            if str(item or "").strip()
        }
        constrain_all = bool((scope or {}).get("allow_global_plan_update"))
        for step in updated.judge_steps:
            step_id = str(step.id or step.condition_id or "").strip().upper()
            if constrain_all or step_id in target_steps:
                constrain_judge_step_first_pass(
                    step,
                    attack_label=str((rule.metadata or {}).get("attack_label", "")),
                )
                step.question = neutralize_attack_label_mentions(
                    step.question,
                    attack_label=str((rule.metadata or {}).get("attack_label", "")),
                )
        updated = _restore_unrelated_steps_exact(previous, updated, scope)
        return _reapply_label_stateful_runtime(rule, updated)

    @staticmethod
    def _fallback_update(
        rule: EvolvingRule,
        plan: EvidencePlan,
        actionable_reviews: List[Dict[str, Any]],
        *,
        scope: Dict[str, Any] | None = None,
    ) -> EvidencePlan:
        updated = deepcopy(plan)
        scope = dict(scope or _infer_plan_update_scope(actionable_reviews))
        if not scope.get("scoped") and not scope.get("allow_global_plan_update"):
            updated.metadata = {
                **dict(updated.metadata or {}),
                "scope_guard": scope,
                "semantic_plan_changed": False,
                "plan_update_noop_reason": "unscoped_plan_review",
            }
            return updated
        suggestions: List[Dict[str, Any]] = []
        for review in actionable_reviews:
            suggestion = review.get("plan_patch_suggestion", {}) or {}
            if isinstance(suggestion, dict) and suggestion.get("action") not in (None, "", "none"):
                suggestions.append(suggestion)
                _apply_simple_suggestion(
                    updated,
                    suggestion,
                    scope=scope,
                    attack_label=str((rule.metadata or {}).get("attack_label", "")),
                )
        updated = EvidencePlan.from_dict(
            _apply_minimal_plan_scope(
                plan,
                updated.to_dict(),
                scope,
                attack_label=str((rule.metadata or {}).get("attack_label", "")),
            )
        )
        if semantic_plan_fingerprint(updated) == semantic_plan_fingerprint(plan):
            return plan
        updated.plan_note = (
            (updated.plan_note + "\n") if updated.plan_note else ""
        ) + "Fallback plan update recorded plan-actionable suggestions."
        updated.metadata["fallback_plan_update"] = True
        updated.metadata["plan_patch_suggestions"] = suggestions
        updated.metadata["scope_guard"] = scope
        updated.metadata["semantic_plan_changed"] = True
        target_steps = {
            str(item or "").strip().upper()
            for item in list(scope.get("target_steps", []) or [])
            if str(item or "").strip()
        }
        for step in updated.judge_steps:
            step_id = str(step.id or step.condition_id or "").strip().upper()
            if scope.get("allow_global_plan_update") or step_id in target_steps:
                constrain_judge_step_first_pass(
                    step,
                    attack_label=str((rule.metadata or {}).get("attack_label", "")),
                )
                step.question = neutralize_attack_label_mentions(
                    step.question,
                    attack_label=str((rule.metadata or {}).get("attack_label", "")),
                )
        _stamp_plan_metadata(updated, rule, previous=plan, source="fallback_plan_update")
        updated = _restore_unrelated_steps_exact(plan, updated, scope)
        updated = _reapply_label_stateful_runtime(rule, updated)
        if _plan_scope_violations(plan, updated, scope):
            return plan
        return updated


def _candidate_payload_from_plan_update_payload(
    previous_plan: EvidencePlan,
    payload: Dict[str, Any],
    scope: Dict[str, Any] | None,
    *,
    attack_label: str = "",
    rule: EvolvingRule | None = None,
    joint_resolution_groups: Sequence[Dict[str, Any]] = (),
    enable_advanced_strategy: bool = False,
    review_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Convert the PlanUpdater patch schema into a full EvidencePlan payload.

    Older full-plan payloads are still accepted so existing callers and tests do
    not fork the updater path. The prompt now asks for plan_patches because the
    LLM should not rewrite unrelated steps just to adjust one condition.
    """
    if not isinstance(payload, dict):
        raise ValueError("Plan update payload must be a JSON object.")
    matrix_rows = [
        item
        for item in list(
            ((review_bundle or {}).get("update_signal_matrix") or {}).get(
                "signals", []
            )
            or []
        )
        if isinstance(item, dict) and str(item.get("signal_id") or "")
    ]
    if "plan_patches" not in payload and "step_patches" not in payload:
        if matrix_rows:
            rejected = previous_plan.to_dict()
            audit = {
                "schema_version": "evotx.plan_patch_materialization.v1",
                "rejected": True,
                "rejection_reason": "canonical_signal_requires_plan_patches",
                "patches": [],
                "accepted_patch_indexes": [],
                "rejected_patch_indexes": [],
                "accepted_signal_ids": [],
            }
            rejected["plan_patch_materialization"] = audit
            rejected["plan_patch_materialization_rejected"] = True
            rejected["plan_patch_materialization_rejected_reason"] = (
                audit["rejection_reason"]
            )
            return rejected
        return dict(payload)

    patches = payload.get("plan_patches")
    if patches is None:
        patches = payload.get("step_patches")
    if not isinstance(patches, list):
        raise ValueError("plan_patches must be an array.")

    full = previous_plan.to_dict()
    full["plan_id"] = previous_plan.plan_id
    full["rule_id"] = previous_plan.rule_id
    full["rule_version"] = previous_plan.rule_version
    full["emit_logic"] = previous_plan.emit_logic
    full["plan_note"] = str(
        payload.get("plan_note")
        or full.get("plan_note")
        or "Plan strategy updated from step-level patches."
    )
    full["metadata"] = {
        **dict(full.get("metadata") or {}),
        **dict(payload.get("metadata") or {}),
        "source": "plan_update",
        "plan_update_payload_schema": "patch",
    }
    if "applied_signal_ids" in payload:
        full["applied_signal_ids"] = list(_string_list(payload.get("applied_signal_ids")))

    materialized = _materialize_plan_patches_partially(
        previous_plan,
        full,
        patches,
        scope=scope,
        attack_label=attack_label,
        rule=rule,
        joint_resolution_groups=joint_resolution_groups,
        enable_advanced_strategy=enable_advanced_strategy,
        review_bundle=review_bundle,
    )
    full = materialized["payload"]
    expectations = materialized["expectations"]
    metadata = dict(full.get("metadata") or {})
    metadata["plan_patch_materialization"] = materialized["audit"]
    full["metadata"] = metadata
    full["plan_patch_materialization"] = materialized["audit"]
    if materialized["audit"].get("rejected"):
        full["plan_patch_materialization_rejected"] = True
        full["plan_patch_materialization_rejected_reason"] = str(
            materialized["audit"].get("rejection_reason") or ""
        )
    if expectations:
        metadata = dict(full.get("metadata") or {})
        metadata["plan_patch_expectations"] = expectations
        full["metadata"] = metadata
    return full


def _extract_plan_update_payload(text: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract a PlanUpdater payload with strict schema-first semantics."""
    metadata: Dict[str, Any] = {
        "strategy": "strict_first",
        "strict_schema_valid": False,
        "last_schema_valid_used": False,
        "trailing_comma_repair_used": False,
        "strict_error": "",
        "schema_error": "",
        "repair_error": "",
    }
    raw_text = str(text or "")
    try:
        strict = extract_json_object(raw_text)
        schema_error = _plan_update_payload_schema_error(strict)
        if schema_error is None:
            metadata["strict_schema_valid"] = True
            metadata["strategy"] = "strict"
            return strict, metadata
        metadata["schema_error"] = schema_error
    except (JsonExtractionError, ValueError) as exc:
        metadata["strict_error"] = repr(exc)

    selected, error, selection_metadata = extract_last_schema_valid_object(
        raw_text,
        _plan_update_payload_schema_error,
        schema_name="PlanUpdater payload",
    )
    metadata["last_schema_valid_selection"] = selection_metadata
    if error is None:
        metadata["last_schema_valid_used"] = True
        metadata["strategy"] = "last_schema_valid"
        return selected, metadata
    metadata["last_schema_valid_error"] = str(error or "")

    repaired_text = _remove_json_trailing_commas(raw_text)
    if repaired_text != raw_text:
        try:
            repaired = extract_json_object(repaired_text)
            schema_error = _plan_update_payload_schema_error(repaired)
            if schema_error is None:
                metadata["trailing_comma_repair_used"] = True
                metadata["strategy"] = "strict_after_trailing_comma_repair"
                return repaired, metadata
            metadata["repair_error"] = schema_error
        except (JsonExtractionError, ValueError) as exc:
            metadata["repair_error"] = repr(exc)

    if metadata.get("strict_error"):
        raise JsonExtractionError(str(metadata["strict_error"]))
    raise JsonExtractionError(
        "PlanUpdater payload failed schema validation: "
        + str(metadata.get("schema_error") or metadata.get("last_schema_valid_error") or "")
    )


def _remove_json_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[\}\]])", r"\1", str(text or ""))


def _plan_update_payload_schema_error(payload: Dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return "top-level payload must be an object"
    patch_keys = [key for key in ("plan_patches", "step_patches") if key in payload]
    if patch_keys:
        patches = payload.get(patch_keys[0])
        if not isinstance(patches, list):
            return f"{patch_keys[0]} must be an array"
        for index, patch in enumerate(patches):
            if not isinstance(patch, dict):
                return f"{patch_keys[0]}[{index}] must be an object"
            condition_id = str(
                patch.get("condition_id")
                or patch.get("step_id")
                or patch.get("id")
                or ""
            ).strip()
            if not condition_id:
                return f"{patch_keys[0]}[{index}] missing condition_id/step_id"
            errors = _plan_patch_schema_errors(patch)
            if errors:
                return f"{patch_keys[0]}[{index}]: {'; '.join(errors)}"
        return None
    if "judge_steps" in payload:
        if not isinstance(payload.get("judge_steps"), list):
            return "judge_steps must be an array"
        return None
    return "payload must contain plan_patches, step_patches, or judge_steps"


_MINIMAL_PLAN_STRATEGY_OPERATIONS = {
    "",
    "expand_evidence_route",
    "add_evidence_route",
    "add_evidence_view",
    "replace_evidence_route",
    "change_followup_budget",
}


def _minimal_plan_strategy_violations(
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Reject advanced Plan search operations from the default pipeline."""
    patches = payload.get("plan_patches")
    if patches is None:
        patches = payload.get("step_patches")
    if not isinstance(patches, list):
        return [{"reason": "minimal_plan_requires_typed_step_patches"}]
    violations: List[Dict[str, Any]] = []
    for patch_index, raw_patch in enumerate(patches):
        if not isinstance(raw_patch, dict):
            continue
        operation = str(
            raw_patch.get("strategy_operation")
            or raw_patch.get("operation")
            or ""
        ).strip().lower()
        if operation not in _MINIMAL_PLAN_STRATEGY_OPERATIONS:
            violations.append({
                "patch_index": patch_index,
                "operation": operation,
                "reason": "advanced_plan_strategy_not_in_minimal_pipeline",
            })
        touched_fields = _plan_patch_touched_fields(raw_patch)
        if "depends_on" in touched_fields:
            violations.append({
                "patch_index": patch_index,
                "operation": operation,
                "reason": "dependency_restructure_not_in_minimal_pipeline",
            })
    return violations


def _minimal_plan_delta_violations(
    deltas: Sequence[Any],
) -> List[Dict[str, Any]]:
    allowed = {
        "expand_evidence_route",
        "replace_evidence_route",
        "promote_followup_to_default",
        # A bounded replacement may move the displaced default route to
        # follow-up. Explicit demotion proposals are still rejected by
        # _minimal_plan_strategy_violations before materialization.
        "demote_default_to_followup",
        "change_followup_budget",
    }
    return [
        {
            "condition_id": str(delta.condition_id or ""),
            "operations": sorted(set(delta.operations) - allowed),
            "reason": "materialized_advanced_plan_strategy_not_in_minimal_pipeline",
        }
        for delta in list(deltas or [])
        if set(delta.operations) - allowed
    ]


def _materialize_plan_patches_partially(
    previous_plan: EvidencePlan,
    base_payload: Dict[str, Any],
    patches: Sequence[Any],
    *,
    scope: Dict[str, Any] | None,
    attack_label: str,
    rule: EvolvingRule | None,
    joint_resolution_groups: Sequence[Dict[str, Any]],
    enable_advanced_strategy: bool,
    review_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Retain locally valid independent patches and atomic joint groups."""
    local_results: List[Dict[str, Any]] = []
    scoped_targets = {
        str(value or "").strip().upper()
        for value in list((scope or {}).get("target_steps") or [])
        if str(value or "").strip()
    }
    allow_global = bool((scope or {}).get("allow_global_plan_update"))
    matrix_signals = {
        str(signal.get("signal_id") or ""): dict(signal)
        for signal in list(
            ((review_bundle or {}).get("update_signal_matrix") or {}).get(
                "signals", []
            )
            or []
        )
        if isinstance(signal, dict) and str(signal.get("signal_id") or "")
    }
    signal_conditions = {
        signal_id: str(signal.get("condition_id") or "").strip().upper()
        for signal_id, signal in matrix_signals.items()
        if str(signal.get("update_target") or "").strip().lower() == "plan"
    }
    for index, raw_patch in enumerate(list(patches or [])):
        patch = dict(raw_patch) if isinstance(raw_patch, dict) else {}
        condition_id = str(
            patch.get("step_id")
            or patch.get("condition_id")
            or patch.get("id")
            or ""
        ).strip().upper()
        signal_ids = list(dict.fromkeys(_string_list(
            patch.get("applied_signal_ids")
        )))
        result = {
            "patch_index": index,
            "condition_id": condition_id,
            "signal_ids": signal_ids,
            "strategy_operation": str(patch.get("strategy_operation") or ""),
            "accepted": False,
            "stage": "local_patch_materialization",
            "reason": "",
            "budget_audit": {},
        }
        try:
            if not isinstance(raw_patch, dict):
                raise ValueError("Each plan patch must be an object.")
            if not condition_id:
                raise ValueError("Plan patch missing condition_id/step_id.")
            if matrix_signals:
                if not signal_ids:
                    raise ValueError(
                        "plan_patch_requires_canonical_signal_ids"
                    )
                if any(signal_id not in signal_conditions for signal_id in signal_ids):
                    raise ValueError(
                        "plan_patch_signal_owner_is_not_plan"
                    )
                canonical_conditions = {
                    signal_conditions.get(signal_id, "") for signal_id in signal_ids
                }
                if "" in canonical_conditions:
                    raise ValueError(
                        "plan_patch_signal_missing_condition"
                    )
                if canonical_conditions != {condition_id}:
                    raise ValueError(
                        "plan_patch_signal_condition_mismatch"
                    )
            if scoped_targets and not allow_global and condition_id not in scoped_targets:
                raise ValueError(
                    f"Plan patch target {condition_id} is outside the owner-projected scope."
                )
            schema_errors = _plan_patch_schema_errors(patch)
            if schema_errors:
                raise ValueError("; ".join(schema_errors))
            raw_strategy_violations = (
                []
                if enable_advanced_strategy
                else _minimal_plan_strategy_violations({"plan_patches": [patch]})
            )
            if raw_strategy_violations:
                result["strategy_violations"] = raw_strategy_violations
                result["reason"] = "local_strategy_operation_rejected"
                local_results.append(result)
                continue
            local_payload = deepcopy(base_payload)
            local_steps = [
                dict(step)
                for step in list(local_payload.get("judge_steps") or [])
                if isinstance(step, dict)
            ]
            local_payload["judge_steps"] = local_steps
            local_by_id = {
                str(
                    step.get("id") or step.get("condition_id") or ""
                ).strip().upper(): step
                for step in local_steps
            }
            if condition_id not in local_by_id:
                raise ValueError(
                    f"Plan patch references unknown step {condition_id}."
                )
            result["budget_audit"] = _apply_plan_patch_to_step(
                local_by_id[condition_id],
                patch,
                attack_label=attack_label,
                materialize_minimal_replacement=(
                    not enable_advanced_strategy
                ),
            )
            local_plan = EvidencePlan.from_dict(local_payload)
            if rule is None:
                PlanValidator().validate(
                    local_plan,
                    attack_label=attack_label,
                )
            else:
                PlanValidator().validate_against_rule(
                    local_plan,
                    rule,
                    attack_label=attack_label,
                )
            result["accepted"] = True
            result["reason"] = "locally_valid"
        except Exception as exc:
            result["reason"] = str(exc)[:500]
            result["error_type"] = type(exc).__name__
        local_results.append(result)

    plan_groups = []
    for index, raw_group in enumerate(list(joint_resolution_groups or [])):
        if not isinstance(raw_group, dict):
            continue
        owner = str(raw_group.get("update_target") or "plan").strip().lower()
        if owner != "plan":
            continue
        signal_ids = {
            str(value)
            for value in list(raw_group.get("signal_ids") or [])
            if str(value)
        }
        if signal_ids:
            plan_groups.append((
                str(raw_group.get("group_id") or f"plan_joint_{index:03d}"),
                signal_ids,
            ))

    rejected_group_indexes: set[int] = set()
    group_audits: List[Dict[str, Any]] = []
    for group_id, group_signal_ids in plan_groups:
        member_indexes = {
            int(item["patch_index"])
            for item in local_results
            if group_signal_ids.intersection(item.get("signal_ids") or [])
        }
        if not member_indexes:
            continue
        represented = {
            signal_id
            for index in member_indexes
            for signal_id in local_results[index].get("signal_ids") or []
            if signal_id in group_signal_ids
        }
        group_valid = bool(
            group_signal_ids.issubset(represented)
            and all(local_results[index]["accepted"] for index in member_indexes)
        )
        if not group_valid:
            rejected_group_indexes.update(member_indexes)
        group_audits.append({
            "group_id": group_id,
            "signal_ids": sorted(group_signal_ids),
            "patch_indexes": sorted(member_indexes),
            "represented_signal_ids": sorted(represented),
            "accepted": group_valid,
            "reason": (
                "joint_group_locally_valid"
                if group_valid
                else "joint_group_requires_all_signals_and_patches_to_be_valid"
            ),
        })

    accepted_indexes = [
        int(item["patch_index"])
        for item in local_results
        if item["accepted"] and int(item["patch_index"]) not in rejected_group_indexes
    ]
    for index in rejected_group_indexes:
        local_results[index]["accepted"] = False
        local_results[index]["stage"] = "joint_group_atomicity"
        local_results[index]["reason"] = "joint_resolution_group_rejected_atomically"

    merged = deepcopy(base_payload)
    merged_steps = [
        dict(step)
        for step in list(merged.get("judge_steps") or [])
        if isinstance(step, dict)
    ]
    merged["judge_steps"] = merged_steps
    merged_by_id = {
        str(step.get("id") or step.get("condition_id") or "").strip().upper(): step
        for step in merged_steps
    }
    expectations: List[Dict[str, Any]] = []
    budget_audits: List[Dict[str, Any]] = []
    merge_error = ""
    try:
        for index in accepted_indexes:
            patch = dict(patches[index])
            condition_id = str(local_results[index]["condition_id"])
            budget_audit = _apply_plan_patch_to_step(
                merged_by_id[condition_id],
                patch,
                attack_label=attack_label,
                materialize_minimal_replacement=(
                    not enable_advanced_strategy
                ),
            )
            if budget_audit:
                budget_audits.append(budget_audit)
            touched_fields = _plan_patch_touched_fields(patch)
            if budget_audit.get("minimal_default_replacement"):
                touched_fields.update({
                    "evidence_refs",
                    "default_evidence_refs",
                    "allowed_followup_views",
                })
            if touched_fields:
                expectations.append({
                    "condition_id": condition_id,
                    "fields": {
                        field: deepcopy(merged_by_id[condition_id].get(field))
                        for field in sorted(touched_fields)
                    },
                })
        merged_plan = EvidencePlan.from_dict(merged)
        if rule is None:
            PlanValidator().validate(merged_plan, attack_label=attack_label)
        else:
            PlanValidator().validate_against_rule(
                merged_plan,
                rule,
                attack_label=attack_label,
            )
    except Exception as exc:
        merge_error = str(exc)[:500]

    accepted_signal_ids = list(dict.fromkeys(
        signal_id
        for index in accepted_indexes
        for signal_id in local_results[index].get("signal_ids") or []
    ))
    rejected = not accepted_indexes or bool(merge_error)
    if merge_error:
        for index in accepted_indexes:
            local_results[index]["accepted"] = False
            local_results[index]["stage"] = "whole_plan_validation"
            local_results[index]["reason"] = "merged_plan_validation_failed"
    if rejected:
        merged = deepcopy(base_payload)
        expectations = []
        budget_audits = []
        accepted_signal_ids = []
    metadata = dict(merged.get("metadata") or {})
    if budget_audits:
        metadata["candidate_followup_budget_audits"] = budget_audits
    metadata["applied_plan_signal_ids"] = accepted_signal_ids
    merged["metadata"] = metadata
    merged["applied_signal_ids"] = accepted_signal_ids
    audit = {
        "schema_version": "evotx.plan_patch_materialization.v1",
        "partial_materialization": len(accepted_indexes) < len(local_results),
        "whole_plan_validation_ran": bool(accepted_indexes),
        "whole_plan_validation_passed": bool(accepted_indexes) and not merge_error,
        "accepted_patch_indexes": accepted_indexes if not merge_error else [],
        "accepted_signal_ids": accepted_signal_ids,
        "rejected_patch_indexes": (
            list(range(len(local_results)))
            if merge_error
            else sorted(set(range(len(local_results))) - set(accepted_indexes))
        ),
        "patches": local_results,
        "joint_resolution_groups": group_audits,
        "rejected": rejected,
        "rejection_reason": (
            "merged_plan_validation_failed"
            if merge_error
            else "no_locally_valid_plan_patch"
            if not accepted_indexes
            else ""
        ),
        "whole_plan_validation_error": merge_error,
    }
    return {
        "payload": merged,
        "expectations": expectations,
        "audit": audit,
    }


def _plan_patch_schema_errors(patch: Dict[str, Any]) -> List[str]:
    allowed_keys = {
        "condition_id",
        "step_id",
        "id",
        "applied_signal_ids",
        "strategy_operation",
        "operation",
        "rationale",
        "expected_effect",
        "set_fields",
        "add_to_fields",
        "remove_from_fields",
        *_PATCHABLE_STEP_FIELDS,
    }
    errors = [
        f"unsupported Plan patch key: {key}"
        for key in patch
        if key not in allowed_keys
    ]
    for section_name in ("set_fields", "add_to_fields", "remove_from_fields"):
        raw_section = patch.get(section_name)
        if raw_section is None:
            continue
        if not isinstance(raw_section, dict):
            errors.append(f"{section_name} must be an object")
            continue
        errors.extend(
            f"unsupported Plan step field in {section_name}: {field}"
            for field in raw_section
            if field not in _PATCHABLE_STEP_FIELDS
        )
    return errors


def _plan_patch_realization_failures(
    updated_plan: EvidencePlan,
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Verify that runtime constraints did not erase a requested step patch."""
    expectations = list(
        dict((payload or {}).get("metadata") or {}).get(
            "plan_patch_expectations", []
        )
        or []
    )
    if not expectations:
        return []
    by_id = {
        str(step.condition_id or step.id or "").strip().upper(): step.to_dict()
        for step in list(updated_plan.judge_steps or [])
    }
    failures: List[Dict[str, Any]] = []
    for expectation in expectations:
        if not isinstance(expectation, dict):
            continue
        condition_id = str(
            expectation.get("condition_id") or ""
        ).strip().upper()
        actual = by_id.get(condition_id)
        if actual is None:
            failures.append({
                "condition_id": condition_id,
                "reason": "target_step_missing",
            })
            continue
        mismatches = []
        for field, expected in dict(expectation.get("fields") or {}).items():
            observed = actual.get(field)
            if field in {
                "evidence_refs",
                "default_evidence_refs",
                "allowed_followup_views",
                "allowed_tools",
                "depends_on",
            }:
                matches = sorted(set(_string_list(observed))) == sorted(
                    set(_string_list(expected))
                )
            else:
                matches = stable_json_dumps(observed) == stable_json_dumps(expected)
            if matches:
                continue
            mismatches.append({
                "field": field,
                "expected": expected,
                "observed": observed,
            })
        if mismatches:
            failures.append({
                "condition_id": condition_id,
                "reason": "patched_fields_not_realized",
                "mismatches": mismatches,
            })
    return failures


def _plan_patch_realization_resolution(
    payload: Dict[str, Any],
    failures: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep realized non-joint fields when runtime policy restores one field."""
    expectations = list(
        dict((payload or {}).get("metadata") or {}).get(
            "plan_patch_expectations", []
        )
        or []
    )
    expected_field_count = sum(
        len(dict(item.get("fields") or {}))
        for item in expectations
        if isinstance(item, dict)
    )
    dropped_fields: List[Dict[str, Any]] = []
    target_step_missing = False
    for failure in list(failures or []):
        if not isinstance(failure, dict):
            continue
        if str(failure.get("reason") or "") == "target_step_missing":
            target_step_missing = True
        condition_id = str(failure.get("condition_id") or "").strip().upper()
        for mismatch in list(failure.get("mismatches") or []):
            if not isinstance(mismatch, dict):
                continue
            dropped_fields.append({
                "condition_id": condition_id,
                "field": str(mismatch.get("field") or ""),
                "requested": mismatch.get("expected"),
                "retained_by_runtime": mismatch.get("observed"),
                "reason": "dropped_by_runtime_constraint",
            })
    materialization = dict(
        (payload or {}).get("plan_patch_materialization")
        or dict((payload or {}).get("metadata") or {}).get(
            "plan_patch_materialization", {}
        )
        or {}
    )
    joint_groups = [
        item
        for item in list(materialization.get("joint_resolution_groups") or [])
        if isinstance(item, dict)
    ]
    realized_field_count = max(0, expected_field_count - len(dropped_fields))
    retain_realized_fields = bool(
        realized_field_count
        and not target_step_missing
        and not joint_groups
    )
    reason = (
        "joint_resolution_group_requires_atomic_realization"
        if joint_groups
        else "target_step_missing"
        if target_step_missing
        else "no_requested_field_survived_runtime_constraints"
        if not realized_field_count
        else "non_joint_realized_fields_retained"
    )
    return {
        "retain_realized_fields": retain_realized_fields,
        "reason": reason,
        "expected_field_count": expected_field_count,
        "realized_field_count": realized_field_count,
        "dropped_fields": dropped_fields,
        "whole_plan_validation_required": retain_realized_fields,
    }


def _reconstrain_plan_strategy_steps(
    plan: EvidencePlan,
    *,
    base_plan: EvidencePlan,
    condition_ids: Sequence[str],
    attack_label: str,
    plan_evidence_audit: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Reapply budgets while allowing justified replacement slots.

    A failed/non-contributing original route may release its active slot, but
    dependency/default and correct-case-proven routes remain protected.
    """
    wanted = {
        str(condition_id or "").strip().upper()
        for condition_id in list(condition_ids or [])
        if str(condition_id or "").strip()
    }
    base_steps = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list(base_plan.judge_steps or [])
    }
    audit_conditions = dict((plan_evidence_audit or {}).get("conditions") or {})
    audits: List[Dict[str, Any]] = []
    for step in list(plan.judge_steps or []):
        condition_id = str(step.condition_id or step.id or "").strip().upper()
        if wanted and condition_id not in wanted:
            continue
        base_step = base_steps.get(condition_id)
        before = judge_step_view_budget_audit(step, attack_label=attack_label)
        dropped_added_views: List[str] = []
        protected_original_routes: List[str] = []
        protected_original_default_routes: List[str] = []
        protected_original_followup_routes: List[str] = []
        released_replacement_slots: List[str] = []
        unprotected_original_optional_routes: List[str] = []
        if base_step is not None:
            base_audit = judge_step_view_budget_audit(
                base_step,
                attack_label=attack_label,
            )
            condition_audit = dict(audit_conditions.get(condition_id) or {})
            released_replacement_slots = _released_replacement_slot_routes(
                base_audit,
                condition_audit,
            )
            # A bounded replacement releases only routes backed by explicit
            # failure/non-contribution history. Every other active route keeps
            # its slot, including evidence used by protected correct cases.
            protected_original_default_routes = [
                route
                for route in list(base_audit.get("active_default_views") or [])
                if route not in set(released_replacement_slots)
            ]
            protected_original_followup_routes = [
                route
                for route in list(base_audit.get("active_followup_views") or [])
                if route not in set(released_replacement_slots)
                and route not in set(protected_original_default_routes)
            ]
            protected_original_routes = list(dict.fromkeys([
                *protected_original_default_routes,
                *protected_original_followup_routes,
            ]))
            unprotected_original_optional_routes = [
                route
                for route in _active_budget_routes(base_audit)
                if route not in set(protected_original_routes)
                and route not in set(released_replacement_slots)
            ]
            base_routes = set(_requested_budget_routes(base_audit))
            _restore_required_original_routes(
                base_step,
                step,
                routes=protected_original_routes,
            )
            while True:
                candidate_audit = judge_step_view_budget_audit(
                    step,
                    attack_label=attack_label,
                )
                active_defaults = set(
                    candidate_audit.get("active_default_views") or []
                )
                active_routes = set(_active_budget_routes(candidate_audit))
                displaced_defaults = [
                    route
                    for route in protected_original_default_routes
                    if route not in active_defaults
                ]
                displaced_followups = [
                    route
                    for route in protected_original_followup_routes
                    if route not in active_routes
                ]
                if not displaced_defaults and not displaced_followups:
                    break
                removed = _drop_lowest_priority_added_route(
                    step,
                    base_routes=base_routes,
                )
                if not removed:
                    break
                dropped_added_views.append(removed)
        deferred = constrain_judge_step_first_pass(
            step,
            attack_label=attack_label,
        )
        after = judge_step_view_budget_audit(step, attack_label=attack_label)
        audits.append({
            "condition_id": condition_id,
            "strategy": "post_safety_restore_budget_reapplication",
            "before": before,
            "after": after,
            "deferred_or_trimmed_views": list(deferred or []),
            "protected_original_active_routes": protected_original_routes,
            "protected_original_default_routes": (
                protected_original_default_routes
            ),
            "protected_original_followup_routes": (
                protected_original_followup_routes
            ),
            "released_replacement_slots": released_replacement_slots,
            "unprotected_original_optional_routes": (
                unprotected_original_optional_routes
            ),
            "dropped_added_views_to_protect_original_routes": (
                dropped_added_views
            ),
            "original_active_routes_preserved": not bool(
                set(protected_original_routes)
                - set(_active_budget_routes(after))
            ),
            "original_default_route_placement_preserved": not bool(
                set(protected_original_default_routes)
                - set(after.get("active_default_views") or [])
            ),
        })
    return audits


def _restore_required_original_routes(
    base_step: JudgeStep,
    candidate_step: JudgeStep,
    *,
    routes: Sequence[str],
) -> None:
    """Restore non-released original routes before reapplying view budgets."""
    required = {str(route or "") for route in routes if str(route or "")}
    if not required:
        return
    base_defaults = list(
        base_step.default_evidence_refs or base_step.evidence_refs or []
    )
    base_followups = list(base_step.allowed_followup_views or [])
    defaults = list(
        candidate_step.default_evidence_refs
        or candidate_step.evidence_refs
        or []
    )
    followups = list(candidate_step.allowed_followup_views or [])

    for route in base_defaults:
        if route not in required:
            continue
        if route in followups:
            followups.remove(route)
        if route not in defaults:
            defaults.insert(min(base_defaults.index(route), len(defaults)), route)
    for route in base_followups:
        if route not in required or route in defaults or route in followups:
            continue
        followups.insert(min(base_followups.index(route), len(followups)), route)

    candidate_step.default_evidence_refs = defaults
    candidate_step.evidence_refs = list(defaults)
    candidate_step.allowed_followup_views = followups


def _released_replacement_slot_routes(
    base_audit: Dict[str, Any],
    condition_audit: Dict[str, Any],
) -> List[str]:
    released: List[str] = []
    for route in _active_budget_routes(base_audit):
        if _route_has_correct_case_contribution(route, condition_audit):
            continue
        if _route_has_failed_or_noncontributing_history(route, condition_audit):
            released.append(route)
    return released


def _route_has_failed_or_noncontributing_history(
    route: str,
    condition_audit: Dict[str, Any],
) -> bool:
    wanted = str(route or "").strip()
    if not wanted:
        return False
    for item in list(condition_audit.get("route_failure_memory") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("route") or "").strip() != wanted:
            continue
        if str(item.get("failure_type") or "").strip().lower() not in {
            "unavailable",
            "truncated",
            "no_answer_change",
            "repeatedly_insufficient",
        }:
            continue
        if int(item.get("support_count", 0) or 0) > 0:
            return True
    return False


def _route_has_correct_case_contribution(
    route: str,
    condition_audit: Dict[str, Any],
) -> bool:
    wanted = str(route or "").strip()
    if not wanted:
        return False
    for group_name in ("correct_positive", "correct_negative", "correct_guard"):
        group = dict((condition_audit.get("groups") or {}).get(group_name) or {})
        for key in ("answer_changing_followup_views", "successful_followup_views"):
            for item in list(group.get(key) or []):
                if not isinstance(item, dict):
                    continue
                if str(item.get("view") or "") == wanted and int(
                    item.get("support_count", item.get("count", 0)) or 0
                ) > 0:
                    return True
    return False


def _enforce_plan_joint_groups_after_strategy_safety(
    base_plan: EvidencePlan,
    candidate_plan: EvidencePlan,
    *,
    materialization_audit: Dict[str, Any],
    strategy_violations: Sequence[Dict[str, Any]],
) -> tuple[EvidencePlan, Dict[str, Any]]:
    """Prevent the strategy-safety gate from splitting a joint patch group."""
    plan = deepcopy(candidate_plan)
    patch_by_index = {
        int(item.get("patch_index") or 0): dict(item)
        for item in list(materialization_audit.get("patches") or [])
        if isinstance(item, dict)
    }
    base_steps = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list(base_plan.judge_steps or [])
    }
    candidate_steps = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list(plan.judge_steps or [])
    }
    violation_conditions = {
        str(item.get("condition_id") or "").strip().upper()
        for item in list(strategy_violations or [])
        if isinstance(item, dict)
        and str(item.get("condition_id") or "").strip()
    }
    groups: List[Dict[str, Any]] = []
    restored_conditions: set[str] = set()
    for raw_group in list(
        materialization_audit.get("joint_resolution_groups") or []
    ):
        if not isinstance(raw_group, dict) or not bool(raw_group.get("accepted")):
            continue
        member_patches = [
            patch_by_index[int(index)]
            for index in list(raw_group.get("patch_indexes") or [])
            if int(index) in patch_by_index
        ]
        condition_ids = sorted({
            str(patch.get("condition_id") or "").strip().upper()
            for patch in member_patches
            if str(patch.get("condition_id") or "").strip()
        })
        rejected = bool(set(condition_ids).intersection(violation_conditions))
        group_audit = {
            "group_id": str(raw_group.get("group_id") or ""),
            "signal_ids": list(raw_group.get("signal_ids") or []),
            "condition_ids": condition_ids,
            "accepted_after_strategy_safety": not rejected,
            "reason": (
                "joint_group_reverted_after_member_strategy_violation"
                if rejected
                else "joint_group_remained_complete"
            ),
        }
        groups.append(group_audit)
        if not rejected:
            continue
        for condition_id in condition_ids:
            if condition_id not in base_steps or condition_id not in candidate_steps:
                continue
            target = candidate_steps[condition_id]
            replacement = deepcopy(base_steps[condition_id])
            for index, step in enumerate(plan.judge_steps):
                if step is target:
                    plan.judge_steps[index] = replacement
                    candidate_steps[condition_id] = replacement
                    restored_conditions.add(condition_id)
                    break
    return plan, {
        "schema_version": "evotx.plan_joint_strategy_safety.v1",
        "groups": groups,
        "restored_condition_ids": sorted(restored_conditions),
        "joint_group_split_allowed": False,
    }


def _requested_budget_routes(audit: Dict[str, Any]) -> List[str]:
    return list(dict.fromkeys([
        *list(audit.get("requested_default_views") or []),
        *list(audit.get("requested_followup_views") or []),
    ]))


def _active_budget_routes(audit: Dict[str, Any]) -> List[str]:
    return list(dict.fromkeys([
        *list(audit.get("active_default_views") or []),
        *list(audit.get("active_followup_views") or []),
    ]))


def _drop_lowest_priority_added_route(
    step: JudgeStep,
    *,
    base_routes: set[str],
) -> str:
    """Drop one candidate-added tail route, preferring default replacements."""
    defaults = list(step.default_evidence_refs or step.evidence_refs or [])
    for index in range(len(defaults) - 1, -1, -1):
        route = str(defaults[index] or "")
        if route and route not in base_routes:
            del defaults[index]
            step.default_evidence_refs = defaults
            step.evidence_refs = list(defaults)
            return route

    followups = list(step.allowed_followup_views or [])
    for index in range(len(followups) - 1, -1, -1):
        route = str(followups[index] or "")
        if route and route not in base_routes:
            del followups[index]
            step.allowed_followup_views = followups
            return route
    return ""


def _refresh_partially_accepted_plan_expectations(
    payload: Dict[str, Any],
    updated_plan: EvidencePlan,
) -> None:
    """Audit the requested patch while validating the bounded retained subset."""
    metadata = dict((payload or {}).get("metadata") or {})
    requested = list(metadata.get("plan_patch_expectations") or [])
    if not requested:
        return
    by_id = {
        str(step.condition_id or step.id or "").strip().upper(): step.to_dict()
        for step in list(updated_plan.judge_steps or [])
    }
    realized: List[Dict[str, Any]] = []
    for expectation in requested:
        if not isinstance(expectation, dict):
            continue
        condition_id = str(
            expectation.get("condition_id") or ""
        ).strip().upper()
        step = by_id.get(condition_id)
        if step is None:
            continue
        fields = {
            field: deepcopy(step.get(field))
            for field in dict(expectation.get("fields") or {})
        }
        if fields:
            realized.append({
                "condition_id": condition_id,
                "fields": fields,
            })
    metadata["plan_patch_requested_expectations"] = requested
    metadata["plan_patch_expectations"] = realized
    metadata["plan_patch_expectation_mode"] = "strategy_operation_filter"
    payload["metadata"] = metadata


def _refresh_strategy_filtered_plan_lineage(
    payload: Dict[str, Any],
    updated_plan: EvidencePlan,
    retained_deltas: Sequence[Any],
) -> None:
    """Keep candidate lineage aligned with operations retained by safety."""
    metadata = dict((payload or {}).get("metadata") or {})
    materialization = dict(metadata.get("plan_patch_materialization") or {})
    retained_conditions = {
        str(delta.condition_id or "").strip().upper()
        for delta in list(retained_deltas or [])
        if str(delta.condition_id or "").strip()
    }
    retained_signal_ids: List[str] = []
    retained_patch_indexes: List[int] = []
    patches: List[Dict[str, Any]] = []
    for raw_patch in list(materialization.get("patches") or []):
        if not isinstance(raw_patch, dict):
            continue
        patch = dict(raw_patch)
        condition_id = str(patch.get("condition_id") or "").strip().upper()
        retained = bool(patch.get("accepted")) and condition_id in retained_conditions
        patch["accepted_after_strategy_safety"] = retained
        if retained:
            retained_patch_indexes.append(int(patch.get("patch_index") or 0))
            for signal_id in list(patch.get("signal_ids") or []):
                value = str(signal_id or "")
                if value and value not in retained_signal_ids:
                    retained_signal_ids.append(value)
        patches.append(patch)
    materialization["patches"] = patches
    materialization["accepted_patch_indexes_after_strategy_safety"] = (
        retained_patch_indexes
    )
    materialization["accepted_signal_ids"] = retained_signal_ids
    materialization["strategy_safety_filter_applied"] = True
    metadata["plan_patch_materialization"] = materialization
    metadata["applied_plan_signal_ids"] = retained_signal_ids
    payload["metadata"] = metadata
    payload["plan_patch_materialization"] = materialization
    payload["applied_signal_ids"] = retained_signal_ids
    updated_plan.metadata = {
        **dict(updated_plan.metadata or {}),
        "plan_patch_materialization": materialization,
        "applied_plan_signal_ids": retained_signal_ids,
    }


_PATCHABLE_STEP_FIELDS = {
    "evidence_refs",
    "default_evidence_refs",
    "allowed_followup_views",
    "allowed_tools",
    "max_followups",
    "depends_on",
}

_PRESERVED_VIEW_ROUTE_FIELDS = {
    "evidence_refs",
    "default_evidence_refs",
    "allowed_followup_views",
}


def _materialize_minimal_default_replacement(
    step: Dict[str, Any],
    patch: Dict[str, Any],
    original_routes: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Materialize one bounded default/follow-up replacement in minimal mode."""
    operation = str(
        patch.get("strategy_operation") or patch.get("operation") or ""
    ).strip().lower()
    if operation != "replace_evidence_route":
        return {}

    original_defaults = list(_string_list(
        original_routes.get("default_evidence_refs")
        or original_routes.get("evidence_refs")
    ))
    original_followups = list(_string_list(
        original_routes.get("allowed_followup_views")
    ))
    original_view_set = {
        view
        for values in original_routes.values()
        for view in list(values or [])
    }
    requested_defaults = list(_string_list(
        step.get("default_evidence_refs") or step.get("evidence_refs")
    ))
    requested_followups = list(_string_list(step.get("allowed_followup_views")))
    replacement = next((
        view for view in requested_defaults if view not in original_defaults
    ), "")
    replacement_source = "promoted_existing_followup"
    if not replacement:
        replacement = next((
            view for view in requested_followups if view not in original_view_set
        ), "")
        replacement_source = "new_registered_route"
    if not replacement:
        return {}

    displaced = next((
        view for view in original_defaults if view not in requested_defaults
    ), original_defaults[-1] if original_defaults else "")
    defaults = list(original_defaults)
    if displaced and displaced in defaults:
        defaults[defaults.index(displaced)] = replacement
    elif replacement not in defaults:
        defaults.append(replacement)

    followups = list(original_followups)
    removed_followup = ""
    if replacement in followups:
        replacement_index = followups.index(replacement)
        if displaced:
            followups[replacement_index] = displaced
        else:
            followups.pop(replacement_index)
    else:
        removed_followup = next((
            view for view in original_followups if view not in requested_followups
        ), original_followups[-1] if original_followups else "")
        if removed_followup and removed_followup in followups:
            replacement_index = followups.index(removed_followup)
            if displaced:
                followups[replacement_index] = displaced
            else:
                followups.pop(replacement_index)
        elif displaced and displaced not in followups:
            followups.append(displaced)

    followups = [view for view in followups if view not in defaults]

    step["default_evidence_refs"] = defaults
    step["evidence_refs"] = list(defaults)
    step["allowed_followup_views"] = followups
    return {
        "policy": "bounded_default_followup_replacement",
        "replacement_route": replacement,
        "displaced_lowest_priority_default": displaced,
        "removed_followup_route": removed_followup,
        "replacement_source": replacement_source,
    }


def _apply_plan_patch_to_step(
    step: Dict[str, Any],
    patch: Dict[str, Any],
    *,
    attack_label: str = "",
    materialize_minimal_replacement: bool = False,
) -> Dict[str, Any]:
    original_routes = _step_view_routes(step)
    original_list_fields = {
        field: list(_string_list(step.get(field)))
        for field in _PATCHABLE_STEP_FIELDS
        if field != "max_followups"
    }
    original_max_followups = int(step.get("max_followups") or 0)
    operation = str(
        patch.get("strategy_operation") or patch.get("operation") or ""
    ).strip().lower()
    set_fields = dict(patch.get("set_fields") or {})
    for field in _PATCHABLE_STEP_FIELDS:
        if field in patch:
            set_fields[field] = patch[field]
    if materialize_minimal_replacement and operation == "expand_evidence_route":
        removed_list_fields = {
            field
            for field in dict(patch.get("remove_from_fields") or {})
            if field in original_list_fields
        }
        if removed_list_fields:
            raise ValueError(
                "expand_evidence_route cannot remove existing routes, tools, or "
                "dependencies; "
                "declare replace_evidence_route for bounded replacement."
            )
        for field, value in list(set_fields.items()):
            if field == "max_followups":
                set_fields[field] = max(original_max_followups, int(value or 0))
            elif field in original_list_fields:
                set_fields[field] = _merge_preserved_route(
                    original_list_fields[field],
                    value,
                )
    for field, value in set_fields.items():
        if field not in _PATCHABLE_STEP_FIELDS:
            continue
        if field == "max_followups":
            step[field] = int(value or 0)
        elif field in {"evidence_refs", "default_evidence_refs", "allowed_followup_views", "allowed_tools", "depends_on"}:
            step[field] = list(_string_list(value))

    for section_name, add_mode in (("add_to_fields", True), ("remove_from_fields", False)):
        section = dict(patch.get(section_name) or {})
        for field, values in section.items():
            if field not in _PATCHABLE_STEP_FIELDS or field == "max_followups":
                continue
            current = list(_string_list(step.get(field)))
            wanted = list(_string_list(values))
            if add_mode:
                existing_views = _step_existing_view_refs(step)
                for item in wanted:
                    if (
                        field in {
                            "evidence_refs",
                            "default_evidence_refs",
                            "allowed_followup_views",
                        }
                        and item in existing_views
                    ):
                        continue
                    if item not in current:
                        current.append(item)
            else:
                remove = set(wanted)
                current = [item for item in current if item not in remove]
            step[field] = current

    if "default_evidence_refs" in set_fields and "evidence_refs" not in set_fields:
        step["evidence_refs"] = list(step.get("default_evidence_refs") or [])
    if "evidence_refs" in set_fields and "default_evidence_refs" not in set_fields:
        step["default_evidence_refs"] = list(step.get("evidence_refs") or [])

    minimal_default_replacement = (
        _materialize_minimal_default_replacement(
            step,
            patch,
            original_routes,
        )
        if materialize_minimal_replacement
        else {}
    )

    granularity_audit = _normalize_reentrancy_state_order_granularity(
        step,
        attack_label=attack_label,
    )

    touched_route_fields = _plan_patch_touched_fields(patch).intersection(
        _PRESERVED_VIEW_ROUTE_FIELDS
    )
    if minimal_default_replacement:
        touched_route_fields.update({
            "evidence_refs",
            "default_evidence_refs",
            "allowed_followup_views",
        })
    if "default_evidence_refs" in touched_route_fields:
        touched_route_fields.add("evidence_refs")
    if "evidence_refs" in touched_route_fields:
        touched_route_fields.add("default_evidence_refs")

    route_fields = {
        "evidence_refs",
        "default_evidence_refs",
        "allowed_followup_views",
    }
    route_patch_requested = bool(route_fields.intersection(set_fields)) or any(
        route_fields.intersection(dict(patch.get(section) or {}))
        for section in ("add_to_fields", "remove_from_fields")
    )
    explicit_followup_order = "allowed_followup_views" in set_fields
    candidate_budget_audit: Dict[str, Any] = {}
    if route_patch_requested:
        budget_audit = judge_step_view_budget_audit(
            step,
            attack_label=attack_label,
        )
        original_view_set = {
            view
            for values in original_routes.values()
            for view in list(values or [])
        }
        candidate_view_set = {
            *list(budget_audit.get("requested_default_views") or []),
            *list(budget_audit.get("requested_followup_views") or []),
        }
        added_views = sorted(candidate_view_set - original_view_set)
        inactive_views = set(budget_audit.get("inactive_views") or [])
        inactive_added_views = sorted(inactive_views.intersection(added_views))
        if budget_audit.get("over_budget") and not explicit_followup_order:
            raise ValueError(
                "Plan patch exceeds the target step view budget. Do not append "
                "a route to an already full list; explicitly rank the complete "
                "allowed_followup_views list with set_fields instead."
            )
        active_routes = {
            *list(budget_audit.get("active_default_views") or []),
            *list(budget_audit.get("active_followup_views") or []),
        }
        active_added_views = sorted(active_routes.intersection(added_views))
        if inactive_added_views and operation == "expand_evidence_route":
            raise ValueError(
                "expand_evidence_route adds views that do not fit the target "
                f"step budget: {inactive_added_views}. Declare a bounded "
                "replace_evidence_route instead."
            )
        if inactive_added_views and (
            not explicit_followup_order or not active_added_views
        ):
            raise ValueError(
                "Plan patch adds follow-up views that remain inactive after the "
                f"target step budget is applied: {inactive_added_views}."
            )
        candidate_budget_audit = {
            **budget_audit,
            "strategy": (
                "explicit_priority_order"
                if budget_audit.get("over_budget")
                else "within_budget"
            ),
            "added_views": added_views,
            "active_added_views": active_added_views,
            "inactive_added_views_dropped": inactive_added_views,
            "partial_acceptance": bool(inactive_added_views),
            "partial_acceptance_reason": (
                "explicit_priority_order_trimmed_at_view_budget"
                if inactive_added_views
                else ""
            ),
            **(
                {"granularity_normalization": granularity_audit}
                if granularity_audit
                else {}
            ),
            **(
                {"minimal_default_replacement": minimal_default_replacement}
                if minimal_default_replacement
                else {}
            ),
        }

    constrained = _constrained_step_dict(step, attack_label=attack_label)
    _restore_preserved_step_routes(
        constrained,
        original_routes,
        fields=_PRESERVED_VIEW_ROUTE_FIELDS - touched_route_fields,
    )
    step.clear()
    step.update(constrained)
    return candidate_budget_audit


def _normalize_reentrancy_state_order_granularity(
    step: Dict[str, Any],
    *,
    attack_label: str,
) -> Dict[str, Any]:
    """Keep the compact locator in round 0 and full state rows in follow-up."""
    if normalize_attack_label(attack_label, default="") != "reentrancy":
        return {}
    defaults = list(
        _string_list(
            step.get("default_evidence_refs") or step.get("evidence_refs")
        )
    )
    if "reentrancy_state_order_view" not in defaults:
        return {}
    followups = list(_string_list(step.get("allowed_followup_views")))
    index = defaults.index("reentrancy_state_order_view")
    defaults = [
        view for view in defaults if view != "reentrancy_state_order_view"
    ]
    if "reentrancy_state_order_summary_view" not in defaults:
        defaults.insert(
            min(index, len(defaults)),
            "reentrancy_state_order_summary_view",
        )
    followups = [
        view
        for view in followups
        if view not in {
            "reentrancy_state_order_view",
            "reentrancy_state_order_summary_view",
        }
    ]
    followups.insert(0, "reentrancy_state_order_view")
    step["default_evidence_refs"] = defaults
    step["evidence_refs"] = list(defaults)
    step["allowed_followup_views"] = followups
    return {
        "strategy": "compact_locator_then_full_followup",
        "default_view": "reentrancy_state_order_summary_view",
        "followup_view": "reentrancy_state_order_view",
    }


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _plan_patch_touched_fields(patch: Dict[str, Any]) -> set[str]:
    touched = {
        field for field in _PATCHABLE_STEP_FIELDS if field in dict(patch or {})
    }
    for section_name in ("set_fields", "add_to_fields", "remove_from_fields"):
        section = dict((patch or {}).get(section_name) or {})
        touched.update(field for field in section if field in _PATCHABLE_STEP_FIELDS)
    return touched


def _step_existing_view_refs(step: Dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for field in ("evidence_refs", "default_evidence_refs", "allowed_followup_views"):
        refs.update(_string_list(step.get(field)))
    return refs


def _step_view_routes(step: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        field: list(_string_list((step or {}).get(field)))
        for field in _PRESERVED_VIEW_ROUTE_FIELDS
    }


def _plan_context_for_prompt(
    plan: EvidencePlan,
    scope: Dict[str, Any],
) -> Dict[str, Any]:
    target_steps = {
        str(item or "").strip().upper()
        for item in list((scope or {}).get("target_steps") or [])
        if str(item or "").strip()
    }
    selected_steps = []
    omitted = 0
    for step in list(plan.judge_steps or []):
        condition_id = str(step.condition_id or step.id or "").strip().upper()
        step_id = str(step.id or "").strip().upper()
        if target_steps and condition_id not in target_steps and step_id not in target_steps:
            omitted += 1
            continue
        selected_steps.append(step.to_dict())
    if not selected_steps and target_steps:
        selected_steps = [step.to_dict() for step in list(plan.judge_steps or [])[:2]]
    return {
        "plan_id": plan.plan_id,
        "rule_id": plan.rule_id,
        "rule_version": plan.rule_version,
        "attack_label": (plan.metadata or {}).get("attack_label", ""),
        "plan_note": plan.plan_note,
        "emit_logic": plan.emit_logic,
        "scope": dict(scope or {}),
        "judge_steps": selected_steps,
        "omitted_judge_step_count": omitted,
        "context_policy": (
            "Only scoped target steps are shown to reduce prompt size. "
            "Unshown steps must remain unchanged."
        ),
        "metadata": {
            key: value
            for key, value in dict(plan.metadata or {}).items()
            if key
            in {
                "attack_label",
                "stateful_runtime",
                "stateful_binding_mode",
                "semantic_plan_changed",
                "scope_guard",
            }
        },
    }


def _plan_update_already_satisfied(
    plan: EvidencePlan,
    actionable_reviews: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Dict[str, Any]:
    """Skip LLM plan updates that only re-add existing views/tools."""
    checked: List[Dict[str, Any]] = []
    requested_any = False
    target_steps = {
        str(item or "").strip().upper()
        for item in list((scope or {}).get("target_steps") or [])
        if str(item or "").strip()
    }
    for review in list(actionable_reviews or []):
        if not isinstance(review, dict):
            continue
        suggestion = dict(review.get("plan_patch_suggestion") or {})
        requested_operations = {
            str(value or "").strip().lower()
            for value in [
                suggestion.get("action"),
                *[
                    signal.get("strategy_operation")
                    for signal in list(review.get("update_signals") or [])
                    if isinstance(signal, dict)
                ],
            ]
            if str(value or "").strip()
            and str(value or "").strip().lower() not in {"none", "no_change"}
        }
        additive_operations = {
            "add_evidence_route",
            "add_evidence_view",
            "add_followup_view",
            "add_default_view",
            "add_allowed_tool",
            "expand_evidence_route",
            "change_followup_views",
        }
        if requested_operations and not requested_operations.issubset(
            additive_operations
        ):
            return {
                "satisfied": False,
                "reason": "non_additive_strategy_requires_materialization",
                "requested_operations": sorted(requested_operations),
                "checked": checked,
            }
        condition_ids = _plan_review_condition_ids(review, target_steps)
        text = " ".join(
            str(value or "")
            for value in [
                suggestion.get("action"),
                suggestion.get("condition_id"),
                suggestion.get("proposed_change"),
                suggestion.get("rationale"),
                suggestion.get("expected_effect"),
                stable_json_dumps(review.get("update_signals", [])),
            ]
        )
        views = _extract_known_views(text)
        tools = _extract_known_tools(text)
        has_experimental_probe = any(
            isinstance(signal, dict)
            and str(signal.get("generalization_status") or "").strip().lower()
            == "experimental_plan"
            for signal in list(review.get("update_signals") or [])
        )
        if not views and not tools:
            return {
                "satisfied": False,
                "reason": "review_has_no_concrete_route_request",
                "checked": checked,
            }
        requested_any = True
        if not condition_ids and len(target_steps) == 1:
            condition_ids = set(target_steps)
        if not condition_ids:
            return {
                "satisfied": False,
                "reason": "unscoped_route_request",
                "checked": checked,
            }
        for condition_id in sorted(condition_ids):
            step = _find_step(plan, condition_id)
            if step is None:
                return {
                    "satisfied": False,
                    "reason": "target_step_not_found",
                    "condition_id": condition_id,
                    "checked": checked,
                }
            step_dict = step.to_dict()
            existing_views = _step_existing_view_refs(step_dict)
            existing_tools = set(_string_list(step_dict.get("allowed_tools")))
            missing_views = [view for view in views if view not in existing_views]
            missing_tools = [tool for tool in tools if tool not in existing_tools]
            checked.append({
                "condition_id": condition_id,
                "requested_views": views,
                "requested_tools": tools,
                "missing_views": missing_views,
                "missing_tools": missing_tools,
            })
            if missing_views or missing_tools:
                return {
                    "satisfied": False,
                    "reason": "route_request_not_yet_present",
                    "checked": checked,
                }
            if has_experimental_probe:
                return {
                    "satisfied": False,
                    "reason": "route_present_but_unexercised_experimental_probe",
                    "checked": checked,
                }
    return {
        "satisfied": bool(requested_any),
        "reason": "all_requested_routes_already_present"
        if requested_any
        else "no_concrete_route_request",
        "checked": checked,
    }


def _plan_review_condition_ids(
    review: Dict[str, Any],
    fallback_targets: set[str],
) -> set[str]:
    ids: set[str] = set()
    suggestion = dict(review.get("plan_patch_suggestion") or {})
    condition_id = str(suggestion.get("condition_id") or "").strip().upper()
    if re.fullmatch(r"[CE]\d+", condition_id):
        ids.add(condition_id)
    for signal in list(review.get("update_signals") or []):
        if not isinstance(signal, dict):
            continue
        if str(signal.get("update_target") or "").strip().lower() != "plan":
            continue
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        if re.fullmatch(r"[CE]\d+", condition_id):
            ids.add(condition_id)
    for diagnosis in list(review.get("condition_diagnosis") or []):
        if not isinstance(diagnosis, dict):
            continue
        if not diagnosis.get("is_plan_problem"):
            continue
        condition_id = str(diagnosis.get("condition_id") or "").strip().upper()
        if re.fullmatch(r"[CE]\d+", condition_id):
            ids.add(condition_id)
    if not ids and len(fallback_targets) == 1:
        ids.update(fallback_targets)
    return ids


def _extract_known_tools(text: str) -> List[str]:
    known = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
        "read_function_chunk",
    ]
    return [tool for tool in known if tool in str(text or "")]


def _merge_preserved_route(
    preserved: Any,
    proposed: Any,
) -> List[str]:
    merged = list(_string_list(preserved))
    for item in _string_list(proposed):
        if item not in merged:
            merged.append(item)
    return merged


def _restore_preserved_step_routes(
    step: Dict[str, Any],
    preserved_routes: Dict[str, List[str]],
    *,
    fields: set[str] | None = None,
) -> None:
    """Keep untouched evidence/follow-up routes when applying scoped patches.

    Targeted route fields may be replaced or pruned explicitly. Only route fields
    omitted by the patch are restored after downstream budget constraints.
    """
    restore_fields = set(_PRESERVED_VIEW_ROUTE_FIELDS if fields is None else fields)
    for field in restore_fields:
        preserved = list(preserved_routes.get(field) or [])
        if preserved:
            if field == "allowed_followup_views":
                step[field] = _merge_preserved_route(step.get(field), preserved)
            else:
                step[field] = _merge_preserved_route(preserved, step.get(field))
    if (
        "evidence_refs" in restore_fields
        and "default_evidence_refs" in restore_fields
        and step.get("default_evidence_refs")
    ):
        step["evidence_refs"] = _merge_preserved_route(
            step.get("default_evidence_refs"),
            step.get("evidence_refs"),
        )


def _append_rejected_plan_patch_op(
    step: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    metadata = dict(step.get("metadata") or {})
    ops = list(metadata.get("rejected_plan_patch_ops") or [])
    ops.append(payload)
    metadata["rejected_plan_patch_ops"] = ops
    step["metadata"] = metadata


def _reapply_label_stateful_runtime(
    rule: EvolvingRule,
    plan: EvidencePlan,
) -> EvidencePlan:
    label = normalize_attack_label(
        (rule.metadata or {}).get("attack_label", ""),
        default="",
    )
    if label == "token_semantic_exploitation":
        return apply_token_semantic_stateful_runtime(plan, rule=rule)
    if label == "market_manipulation":
        return apply_market_manipulation_stateful_runtime(plan, rule=rule)
    if label == "insufficient_validation":
        return apply_insufficient_validation_stateful_runtime(
            plan,
            attack_label=label,
            preserve_view_policy=True,
            preserve_question_policy=True,
        )
    if label == "reentrancy":
        stateful = dict((plan.metadata or {}).get("stateful_runtime") or {})
        mode = (
            stateful.get("binding_mode")
            or (rule.metadata or {}).get("reentrancy_binding_mode")
            or "soft"
        )
        return apply_reentrancy_stateful_runtime(
            plan,
            rule=rule,
            mode=str(mode),
        )
    return plan


def _plan_evidence_history_gate(
    review_bundle: Dict[str, Any],
    scope: Dict[str, Any],
    actionable_reviews: List[Dict[str, Any]],
) -> Dict[str, Any]:
    audit = dict((review_bundle or {}).get("plan_evidence_audit") or {})
    enabled = str(audit.get("schema_version") or "").startswith(
        "evotx.plan_evidence_audit."
    )
    if not enabled:
        return {"enabled": False, "sufficient": True, "target_conditions": []}

    conditions = dict(audit.get("conditions") or {})
    history_required_operations = {
        "replace_evidence_route",
        "demote_default_to_followup",
        "prune_evidence_route",
        "remove_evidence_route",
    }
    target_conditions: set[str] = set()
    requested_operations: set[str] = set()
    fallback_targets = {
        str(item or "").strip().upper()
        for item in list((scope or {}).get("target_steps", []) or [])
        if str(item or "").strip()
    }
    for review in list(actionable_reviews or []):
        if not isinstance(review, dict):
            continue
        suggestion = dict(review.get("plan_patch_suggestion") or {})
        operations = {
            str(value or "").strip().lower()
            for value in [
                suggestion.get("action"),
                *[
                    signal.get("strategy_operation")
                    for signal in list(review.get("update_signals") or [])
                    if isinstance(signal, dict)
                ],
            ]
            if str(value or "").strip().lower() in history_required_operations
        }
        if not operations:
            continue
        requested_operations.update(operations)
        target_conditions.update(
            _plan_review_condition_ids(review, fallback_targets)
        )
    if not requested_operations:
        return {
            "enabled": False,
            "sufficient": True,
            "target_conditions": [],
            "reason": "operation_does_not_require_route_failure_history",
        }

    present_conditions: List[str] = []
    telemetry_events = 0
    for condition_id in sorted(target_conditions):
        condition = dict(conditions.get(condition_id) or {})
        groups = dict(condition.get("groups") or {})
        condition_has_history = False
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            events = (
                int(group.get("tool_attempt_count", 0) or 0)
                + sum(
                    int(item.get("support_count", item.get("count", 0)) or 0)
                    for item in list(group.get("views", []) or [])
                )
                + sum(int(value or 0) for value in dict(group.get("missing_statuses") or {}).values())
            )
            telemetry_events += events
            condition_has_history = condition_has_history or bool(
                group.get("evidence_history_present") or events
            )
        if condition_has_history:
            present_conditions.append(condition_id)

    sufficient = bool(target_conditions) and set(target_conditions).issubset(
        set(present_conditions)
    )
    return {
        "enabled": True,
        "sufficient": sufficient,
        "target_conditions": sorted(target_conditions),
        "history_required_operations": sorted(requested_operations),
        "conditions_with_history": present_conditions,
        "telemetry_event_count": telemetry_events,
        "reason": "ok" if sufficient else "target_condition_history_missing",
    }


def _unverified_default_view_promotions(
    previous_plan: EvidencePlan,
    candidate_payload: Dict[str, Any],
    review_bundle: Dict[str, Any],
    scope: Dict[str, Any],
) -> List[Dict[str, Any]]:
    audit = dict((review_bundle or {}).get("plan_evidence_audit") or {})
    if not str(audit.get("schema_version") or "").startswith(
        "evotx.plan_evidence_audit."
    ):
        return []
    conditions = dict(audit.get("conditions") or {})
    target_steps = {
        str(item or "").strip().upper()
        for item in list((scope or {}).get("target_steps", []) or [])
        if str(item or "").strip()
    }
    previous_by_id = {
        str(step.id or step.condition_id or "").strip().upper(): step.to_dict()
        for step in previous_plan.judge_steps
    }
    candidate_by_id = {
        str(item.get("id") or item.get("condition_id") or "").strip().upper(): item
        for item in list((candidate_payload or {}).get("judge_steps", []) or [])
        if isinstance(item, dict)
    }
    violations: List[Dict[str, Any]] = []
    experimental_views = _experimental_plan_views_by_condition(review_bundle)
    for step_id, old_step in previous_by_id.items():
        if target_steps and step_id not in target_steps:
            continue
        new_step = candidate_by_id.get(step_id)
        if not isinstance(new_step, dict):
            continue
        old_defaults = {
            str(item)
            for item in list(
                old_step.get("default_evidence_refs")
                or old_step.get("evidence_refs")
                or []
            )
            if str(item)
        }
        new_defaults = {
            str(item)
            for item in list(
                new_step.get("default_evidence_refs")
                or new_step.get("evidence_refs")
                or []
            )
            if str(item)
        }
        added = sorted(new_defaults - old_defaults)
        condition_id = str(
            old_step.get("condition_id") or step_id
        ).strip().upper()
        successful = {
            str(item)
            for item in list(
                (conditions.get(condition_id) or {}).get(
                    "observed_successful_followup_views",
                    [],
                )
                or []
            )
            if str(item)
        }
        unverified = sorted(
            set(added)
            - successful
            - set(experimental_views.get(condition_id, set()))
        )
        if unverified:
            violations.append({
                "condition_id": condition_id,
                "views": unverified,
                "allowed_action": "keep_as_allowed_followup_view_until_observed_successful",
            })
    return violations


def _demote_unverified_default_promotions(
    candidate_payload: Dict[str, Any],
    violations: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert unverified default-route promotion into a probeable follow-up.

    This preserves the proposal as evidence-strategy search space without
    letting an unverified route enter the ordinary default/promotion path.
    """
    payload = deepcopy(candidate_payload or {})
    by_condition = {
        str(item.get("condition_id") or "").strip().upper(): sorted({
            str(view or "").strip()
            for view in list(item.get("views") or [])
            if str(view or "").strip()
        })
        for item in list(violations or [])
        if isinstance(item, dict)
        and str(item.get("condition_id") or "").strip()
    }
    if not by_condition:
        return payload
    demoted: List[Dict[str, Any]] = []
    for step in list(payload.get("judge_steps") or []):
        if not isinstance(step, dict):
            continue
        condition_id = str(
            step.get("condition_id") or step.get("id") or ""
        ).strip().upper()
        views = by_condition.get(condition_id)
        if not views:
            continue
        defaults = list(_string_list(
            step.get("default_evidence_refs") or step.get("evidence_refs")
        ))
        followups = list(_string_list(step.get("allowed_followup_views")))
        moved: List[str] = []
        for view in views:
            if view in defaults:
                defaults = [item for item in defaults if item != view]
                moved.append(view)
            if view not in followups:
                followups.insert(0, view)
        if not moved:
            continue
        step["default_evidence_refs"] = defaults
        step["evidence_refs"] = list(defaults)
        step["allowed_followup_views"] = followups
        demoted.append({
            "condition_id": condition_id,
            "target_routes": moved,
            "reason": "unverified_default_view_promotion_demoted_to_probe_route",
        })
    _refresh_plan_patch_expectations_from_payload(
        payload,
        affected_condition_ids=by_condition.keys(),
    )
    metadata = dict(payload.get("metadata") or {})
    metadata["phase3_unverified_default_activation_probe"] = {
        "schema_version": "evotx.phase3.unverified_default_activation_probe.v1",
        "lifecycle_state": "probe_required",
        "direct_promotion_allowed": False,
        "partial_acceptance_allowed": False,
        "demoted_default_promotions": demoted,
        "source_violations": [dict(item) for item in list(violations or [])],
    }
    payload["metadata"] = metadata
    payload["unverified_default_view_promotions"] = [
        dict(item) for item in list(violations or [])
    ]
    payload["plan_evidence_gate_degraded_to_probe"] = bool(demoted)
    payload["plan_evidence_gate_rejected"] = False
    return payload


def _refresh_plan_patch_expectations_from_payload(
    payload: Dict[str, Any],
    *,
    affected_condition_ids: Iterable[str],
) -> None:
    wanted = {
        str(condition_id or "").strip().upper()
        for condition_id in affected_condition_ids
        if str(condition_id or "").strip()
    }
    if not wanted:
        return
    metadata = dict((payload or {}).get("metadata") or {})
    expectations = list(metadata.get("plan_patch_expectations") or [])
    if not expectations:
        return
    by_id = {
        str(step.get("condition_id") or step.get("id") or "").strip().upper(): step
        for step in list((payload or {}).get("judge_steps") or [])
        if isinstance(step, dict)
    }
    refreshed: List[Dict[str, Any]] = []
    for expectation in expectations:
        if not isinstance(expectation, dict):
            continue
        condition_id = str(
            expectation.get("condition_id") or ""
        ).strip().upper()
        fields = dict(expectation.get("fields") or {})
        step = by_id.get(condition_id)
        if condition_id in wanted and step is not None:
            fields = {
                field: deepcopy(step.get(field))
                for field in fields
            }
        refreshed.append({
            **dict(expectation),
            "fields": fields,
        })
    metadata["plan_patch_expectations"] = refreshed
    metadata["plan_patch_expectation_refresh_reason"] = (
        "unverified_default_promotion_demoted_to_probe_route"
    )
    payload["metadata"] = metadata


def _restore_drifted_questions_from_rule(
    rule: EvolvingRule,
    plan: EvidencePlan,
    actionable_reviews: List[Dict[str, Any]],
) -> tuple[EvidencePlan, Dict[str, Any]]:
    target_ids: List[str] = []
    route_ids: List[str] = []
    for review in actionable_reviews:
        if not review_has_operation(
            review,
            "plan",
            "restore_question_from_rule",
        ):
            continue
        for condition_id in review_operation_condition_ids(
            review,
            "plan",
            "restore_question_from_rule",
        ):
            if condition_id not in target_ids:
                target_ids.append(condition_id)
        route_ids.extend(
            str(route.get("route_id") or "")
            for route in list(review.get("diagnosis_routes") or [])
            if isinstance(route, dict)
            and route.get("owner") == "plan"
            and route.get("requested_operation") == "restore_question_from_rule"
            and route.get("route_status") == "reachable"
        )

    descriptions = {
        str(condition.id or "").strip().upper(): str(condition.description or "")
        for condition in [
            *list(rule.conditions or []),
            *list(rule.exclusion_conditions or []),
        ]
    }
    attack_label = str((rule.metadata or {}).get("attack_label", ""))
    changed: List[str] = []
    before_after: List[Dict[str, str]] = []
    for step in list(plan.judge_steps or []):
        condition_id = str(step.condition_id or step.id or "").strip().upper()
        if condition_id not in target_ids or condition_id not in descriptions:
            continue
        canonical = make_local_judge_question(
            descriptions[condition_id],
            attack_label=attack_label,
        )
        if " ".join(str(step.question or "").split()).lower() == " ".join(
            canonical.split()
        ).lower():
            continue
        before_after.append({
            "condition_id": condition_id,
            "previous_question": str(step.question or ""),
            "restored_question": canonical,
        })
        step.question = canonical
        changed.append(condition_id)
    return plan, {
        "requested_condition_ids": target_ids,
        "changed_condition_ids": changed,
        "route_ids": sorted({item for item in route_ids if item}),
        "changes": before_after,
        "policy": "restore_only_when_rule_semantic_correct_and_question_drifted",
    }


def _review_requires_plan_strategy_update(review: Dict[str, Any]) -> bool:
    routes = [
        route
        for route in list((review or {}).get("diagnosis_routes") or [])
        if isinstance(route, dict) and route.get("owner") == "plan"
    ]
    non_restore_route = any(
        route.get("route_status") == "reachable"
        and route.get("requested_operation") != "restore_question_from_rule"
        for route in routes
    )
    if routes:
        return non_restore_route
    if _review_has_actionable_plan_signal(review):
        return True
    suggestion = dict((review or {}).get("plan_patch_suggestion") or {})
    action = str(suggestion.get("action") or "").strip().lower()
    return action not in {"", "none", "change_judge_question", "restore_question_from_rule"}


def _plan_actionable_reviews(review_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    actionable: List[Dict[str, Any]] = []
    seen: set[str] = set()
    cohort_gate_enabled = _cohort_signal_gate_enabled(review_bundle)
    canonical_contract = str(
        ((review_bundle or {}).get("canonical_signal_contract") or {}).get(
            "schema_version"
        )
        or ""
    ).startswith("evotx.canonical_update_signal_contract.")
    explicit_plan_reviews = list(
        (review_bundle or {}).get("actionable_plan_reviews", []) or []
    )
    review_sources = explicit_plan_reviews or (
        list((review_bundle or {}).get("non_rule_reviews", []) or [])
        + list((review_bundle or {}).get("actionable_rule_reviews", []) or [])
    )
    for review in review_sources:
        if not isinstance(review, dict):
            continue
        review_key = stable_json_dumps({
            "tx_hash": review.get("tx_hash"),
            "error_type": review.get("error_type"),
            "update_signals": review.get("update_signals", []),
            "plan_patch_suggestion": review.get("plan_patch_suggestion", {}),
        })
        if review_key in seen:
            continue
        if not canonical_contract and not _review_allows_plan_training(review):
            if not review_has_operation(
                review,
                "plan",
                "restore_question_from_rule",
            ):
                continue
        has_actionable_signal = _review_has_actionable_plan_signal(review)
        deterministic_restore = review_has_operation(
            review,
            "plan",
            "restore_question_from_rule",
        )
        if cohort_gate_enabled and not has_actionable_signal and not deterministic_restore:
            continue
        if not review.get("should_update_plan_strategy", False):
            continue
        if (
            not _review_has_plan_patch(review)
            and not has_actionable_signal
            and not deterministic_restore
        ):
            continue
        actionable.append(review)
        seen.add(review_key)
    return actionable


def _review_has_plan_patch(review: Dict[str, Any]) -> bool:
    suggestion = review.get("plan_patch_suggestion", {}) or {}
    if isinstance(suggestion, dict):
        action = str(suggestion.get("action") or "").strip().lower()
        if action not in {"", "none"}:
            return True
    if review.get("should_update_plan_strategy") is True:
        return True
    for diagnosis in list(review.get("condition_diagnosis", []) or []):
        if isinstance(diagnosis, dict) and diagnosis.get("is_plan_problem"):
            return True
    return False


_LARGE_DEFAULT_DEFER_VIEWS = {
    "trace_view",
    "state_change_view",
    "event_view",
    "reentrancy_state_order_view",
    "external_fundflow_view",
    "transfer_event_view",
    "profit_loss_view",
}


def _infer_plan_update_scope(actionable_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_steps: set[str] = set()
    supported_signal_steps: set[str] = set()
    allow_emit_logic_change = False
    allow_add_step = False
    allow_remove_step = False
    scoped = True

    for review in actionable_reviews:
        for signal in list(review.get("update_signals", []) or []):
            if not isinstance(signal, dict):
                continue
            if str(signal.get("generalization_status") or "").strip().lower() not in {
                "supported",
                "experimental_plan",
            }:
                continue
            if str(signal.get("update_target") or "").strip().lower() != "plan":
                continue
            condition_id = str(signal.get("condition_id") or "").strip().upper()
            if re.fullmatch(r"[CE]\d+", condition_id):
                supported_signal_steps.add(condition_id)
        suggestion = review.get("plan_patch_suggestion", {}) or {}
        if isinstance(suggestion, dict):
            action = str(suggestion.get("action") or "").strip().lower()
            condition_id = str(suggestion.get("condition_id") or "").strip().upper()
            if re.fullmatch(r"[CE]\d+", condition_id):
                target_steps.add(condition_id)
            text = " ".join(
                str(suggestion.get(key) or "")
                for key in ("proposed_change", "rationale")
            )
            target_steps.update(_extract_condition_ids(text))
            if action == "change_emit_logic":
                allow_emit_logic_change = True
            if action == "add_step":
                allow_add_step = True
            if action == "remove_step":
                allow_remove_step = True

        for cause in list(review.get("root_causes", []) or []):
            if not isinstance(cause, dict):
                continue
            category = str(cause.get("category") or "").strip().lower()
            if category == "runtime_emit_logic_bug" and review.get("update_target") == "plan":
                allow_emit_logic_change = True
            for condition_id in list(cause.get("affected_conditions", []) or []):
                normalized = str(condition_id or "").strip().upper()
                if re.fullmatch(r"[CE]\d+", normalized):
                    target_steps.add(normalized)
            target_steps.update(
                _extract_condition_ids(
                    " ".join(
                        str(cause.get(key) or "")
                        for key in ("description", "suggested_fix")
                    )
                )
            )

        for diagnosis in list(review.get("condition_diagnosis", []) or []):
            if not isinstance(diagnosis, dict) or not diagnosis.get("is_plan_problem"):
                continue
            normalized = str(diagnosis.get("condition_id") or "").strip().upper()
            if re.fullmatch(r"[CE]\d+", normalized):
                target_steps.add(normalized)

    if supported_signal_steps:
        target_steps = supported_signal_steps
    if not target_steps:
        scoped = False
    return {
        "scoped": scoped,
        "target_steps": sorted(target_steps),
        "allow_emit_logic_change": allow_emit_logic_change,
        "allow_add_step": allow_add_step,
        "allow_remove_step": allow_remove_step,
    }


def _allow_global_plan_update(
    review_bundle: Dict[str, Any],
    actionable_reviews: List[Dict[str, Any]],
) -> bool:
    # Global rewrites are outside bounded owner-projected Plan evolution.
    return False


def _apply_minimal_plan_scope(
    previous_plan: EvidencePlan,
    data: Dict[str, Any],
    scope: Dict[str, Any] | None,
    *,
    attack_label: str = "",
) -> Dict[str, Any]:
    scope = dict(scope or {})
    if not scope.get("scoped"):
        if scope.get("allow_global_plan_update"):
            guarded = _apply_constrained_global_plan_update(
                previous_plan,
                data,
                scope,
                attack_label=attack_label,
            )
            return guarded
        guarded = previous_plan.to_dict()
        guarded.setdefault("metadata", {})
        guarded["metadata"] = {
            **dict(guarded.get("metadata") or {}),
            "scope_guard": scope,
            "semantic_plan_changed": False,
            "plan_update_noop_reason": "unscoped_plan_review",
        }
        return guarded

    target_steps = {str(step_id or "").strip().upper() for step_id in scope.get("target_steps", [])}
    proposed_by_id = {
        str(item.get("id") or item.get("condition_id") or "").strip().upper(): dict(item)
        for item in list((data or {}).get("judge_steps", []) or [])
        if isinstance(item, dict) and (item.get("id") or item.get("condition_id"))
    }
    previous_steps = [step.to_dict() for step in previous_plan.judge_steps]
    guarded_steps: List[Dict[str, Any]] = []
    previous_ids = {str(step.get("id") or "").strip().upper() for step in previous_steps}
    rejected_question_changes: List[str] = []

    for old_step in previous_steps:
        step_key = str(old_step.get("id") or old_step.get("condition_id") or "").strip().upper()
        if (
            scope.get("allow_remove_step")
            and step_key in target_steps
            and step_key not in proposed_by_id
        ):
            continue
        if step_key in target_steps and step_key in proposed_by_id:
            new_step = dict(old_step)
            proposed = proposed_by_id[step_key]
            if (
                "question" in proposed
                and str(proposed.get("question") or "")
                != str(old_step.get("question") or "")
            ):
                rejected_question_changes.append(step_key)
            touched_route_fields = {
                field
                for field in _PRESERVED_VIEW_ROUTE_FIELDS
                if stable_json_dumps(proposed.get(field))
                != stable_json_dumps(old_step.get(field))
            }
            for field in (
                "evidence_refs",
                "default_evidence_refs",
                "allowed_followup_views",
                "allowed_tools",
                "max_followups",
                "depends_on",
            ):
                if field in proposed:
                    new_step[field] = proposed[field]
            new_step["id"] = old_step.get("id")
            new_step["condition_id"] = old_step.get("condition_id")
            new_step["expected_answer"] = old_step.get("expected_answer", True)
            constrained = _constrained_step_dict(new_step, attack_label=attack_label)
            _restore_preserved_step_routes(
                constrained,
                _step_view_routes(old_step),
                fields=_PRESERVED_VIEW_ROUTE_FIELDS - touched_route_fields,
            )
            guarded_steps.append(constrained)
        else:
            guarded_steps.append(old_step)

    if scope.get("allow_add_step"):
        for step_key, proposed in proposed_by_id.items():
            if step_key not in previous_ids:
                guarded_steps.append(_constrained_step_dict(proposed, attack_label=attack_label))

    guarded = dict(data or {})
    guarded["plan_id"] = previous_plan.plan_id
    guarded["rule_id"] = previous_plan.rule_id
    guarded["rule_version"] = previous_plan.rule_version
    guarded["focus_steps"] = previous_plan.to_dict().get("focus_steps", [])
    guarded["judge_steps"] = guarded_steps
    if not scope.get("allow_emit_logic_change"):
        guarded["emit_logic"] = previous_plan.emit_logic
    guarded.setdefault("metadata", {})
    guarded["metadata"] = {
        **dict(guarded.get("metadata") or {}),
        "scope_guard": {
            **scope,
            "dropped_unknown_steps": sorted(set(proposed_by_id) - previous_ids),
            "preserved_unrelated_steps": sorted(previous_ids - target_steps),
            "semantic_scope_gate": "judge_questions_immutable",
            "rejected_question_changes": sorted(set(rejected_question_changes)),
        },
    }
    return guarded


def _apply_constrained_global_plan_update(
    previous_plan: EvidencePlan,
    data: Dict[str, Any],
    scope: Dict[str, Any],
    *,
    attack_label: str = "",
) -> Dict[str, Any]:
    proposed_by_id = {
        str(item.get("id") or item.get("condition_id") or "").strip().upper(): dict(item)
        for item in list((data or {}).get("judge_steps", []) or [])
        if isinstance(item, dict) and (item.get("id") or item.get("condition_id"))
    }
    guarded_steps: List[Dict[str, Any]] = []
    previous_steps = [step.to_dict() for step in previous_plan.judge_steps]
    rejected_question_changes: List[str] = []
    for old_step in previous_steps:
        step_key = str(old_step.get("id") or old_step.get("condition_id") or "").strip().upper()
        proposed = proposed_by_id.get(step_key, {})
        if proposed:
            new_step = dict(old_step)
            if (
                "question" in proposed
                and str(proposed.get("question") or "")
                != str(old_step.get("question") or "")
            ):
                rejected_question_changes.append(step_key)
            touched_route_fields = {
                field
                for field in _PRESERVED_VIEW_ROUTE_FIELDS
                if stable_json_dumps(proposed.get(field))
                != stable_json_dumps(old_step.get(field))
            }
            for field in (
                "evidence_refs",
                "default_evidence_refs",
                "allowed_followup_views",
                "allowed_tools",
                "max_followups",
                "depends_on",
            ):
                if field in proposed:
                    new_step[field] = proposed[field]
            new_step["id"] = old_step.get("id")
            new_step["condition_id"] = old_step.get("condition_id")
            new_step["expected_answer"] = old_step.get("expected_answer", True)
            constrained = _constrained_step_dict(new_step, attack_label=attack_label)
            _restore_preserved_step_routes(
                constrained,
                _step_view_routes(old_step),
                fields=_PRESERVED_VIEW_ROUTE_FIELDS - touched_route_fields,
            )
            guarded_steps.append(constrained)
        else:
            guarded_steps.append(old_step)

    guarded = dict(data or {})
    guarded["plan_id"] = previous_plan.plan_id
    guarded["rule_id"] = previous_plan.rule_id
    guarded["rule_version"] = previous_plan.rule_version
    guarded["focus_steps"] = previous_plan.to_dict().get("focus_steps", [])
    guarded["judge_steps"] = guarded_steps
    if not scope.get("allow_emit_logic_change"):
        guarded["emit_logic"] = previous_plan.emit_logic
    guarded.setdefault("metadata", {})
    guarded["metadata"] = {
        **dict(guarded.get("metadata") or {}),
        "scope_guard": {
            **scope,
            "global_update_applied": True,
            "semantic_scope_gate": "judge_questions_immutable",
            "rejected_question_changes": sorted(set(rejected_question_changes)),
            "dropped_unknown_steps": sorted(set(proposed_by_id) - {
                str(step.get("id") or "").strip().upper() for step in previous_steps
            }),
        },
    }
    return guarded


def _restore_unrelated_steps_exact(
    previous_plan: EvidencePlan,
    updated_plan: EvidencePlan,
    scope: Dict[str, Any] | None,
) -> EvidencePlan:
    """Restore every out-of-scope JudgeStep byte-for-byte at field level."""
    scope = dict(scope or {})
    if scope.get("allow_global_plan_update"):
        return updated_plan
    target_steps = {
        str(item or "").strip().upper()
        for item in list(scope.get("target_steps", []) or [])
        if str(item or "").strip()
    }
    updated_by_id = {
        str(step.id or step.condition_id or "").strip().upper(): step
        for step in list(updated_plan.judge_steps or [])
    }
    restored: List[JudgeStep] = []
    previous_ids: set[str] = set()
    for previous_step in list(previous_plan.judge_steps or []):
        step_id = str(
            previous_step.id or previous_step.condition_id or ""
        ).strip().upper()
        previous_ids.add(step_id)
        if step_id in target_steps and step_id in updated_by_id:
            restored.append(updated_by_id[step_id])
        else:
            restored.append(deepcopy(previous_step))
    if scope.get("allow_add_step"):
        restored.extend(
            step
            for step_id, step in updated_by_id.items()
            if step_id not in previous_ids
        )
    updated_plan.judge_steps = restored
    return updated_plan


def _plan_payload_scope_violations(
    previous_plan: EvidencePlan,
    payload: Dict[str, Any],
    scope: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Reject raw updater output that attempts an out-of-scope step edit."""
    scope = dict(scope or {})
    if scope.get("allow_global_plan_update"):
        return []
    target_steps = {
        str(item or "").strip().upper()
        for item in list(scope.get("target_steps", []) or [])
        if str(item or "").strip()
    }
    previous_by_id = {
        str(step.id or step.condition_id or "").strip().upper(): step.to_dict()
        for step in list(previous_plan.judge_steps or [])
    }
    violations: List[Dict[str, Any]] = []
    for proposed in list((payload or {}).get("judge_steps", []) or []):
        if not isinstance(proposed, dict):
            continue
        step_id = str(
            proposed.get("id") or proposed.get("condition_id") or ""
        ).strip().upper()
        if not step_id:
            continue
        if step_id not in previous_by_id:
            if not scope.get("allow_add_step"):
                violations.append({"type": "step_added", "step_id": step_id})
            continue
        if step_id in target_steps:
            continue
        previous = previous_by_id[step_id]
        changed_fields = sorted(
            field
            for field, value in proposed.items()
            if field not in {"metadata"}
            and stable_json_dumps(previous.get(field)) != stable_json_dumps(value)
        )
        if changed_fields:
            violations.append({
                "type": "raw_out_of_scope_step_changed",
                "step_id": step_id,
                "changed_fields": changed_fields,
            })
    if (
        "emit_logic" in (payload or {})
        and not scope.get("allow_emit_logic_change")
        and str(payload.get("emit_logic") or "") != str(previous_plan.emit_logic or "")
    ):
        violations.append({"type": "raw_emit_logic_changed"})
    return violations


def _full_plan_view_budget_violations(
    previous_plan: EvidencePlan,
    payload: Dict[str, Any],
    scope: Dict[str, Any] | None,
    *,
    attack_label: str = "",
) -> List[Dict[str, Any]]:
    """Reject legacy full-plan edits whose changed routes exceed the budget."""
    scope = dict(scope or {})
    targets = {
        str(value or "").strip().upper()
        for value in list(scope.get("target_steps") or [])
        if str(value or "").strip()
    }
    previous_by_id = {
        str(step.condition_id or step.id or "").strip().upper(): step.to_dict()
        for step in list(previous_plan.judge_steps or [])
    }
    violations: List[Dict[str, Any]] = []
    for proposed in list((payload or {}).get("judge_steps") or []):
        if not isinstance(proposed, dict):
            continue
        condition_id = str(
            proposed.get("condition_id") or proposed.get("id") or ""
        ).strip().upper()
        if targets and condition_id not in targets:
            continue
        previous = previous_by_id.get(condition_id)
        if previous is None:
            continue
        route_fields = (
            "evidence_refs",
            "default_evidence_refs",
            "allowed_followup_views",
        )
        if all(
            list(_string_list(proposed.get(field)))
            == list(_string_list(previous.get(field)))
            for field in route_fields
        ):
            continue
        audit = judge_step_view_budget_audit(
            proposed,
            attack_label=attack_label,
        )
        if audit.get("over_budget"):
            violations.append({
                "condition_id": condition_id,
                "budget": audit.get("budget", {}),
                "inactive_views": audit.get("inactive_views", []),
            })
    return violations


def _plan_scope_violations(
    previous_plan: EvidencePlan,
    updated_plan: EvidencePlan,
    scope: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Compare the actual plan JSON diff against the granted write scope."""
    scope = dict(scope or {})
    if scope.get("allow_global_plan_update"):
        return []
    target_steps = {
        str(item or "").strip().upper()
        for item in list(scope.get("target_steps", []) or [])
        if str(item or "").strip()
    }
    previous_by_id = {
        str(step.id or step.condition_id or "").strip().upper(): step.to_dict()
        for step in list(previous_plan.judge_steps or [])
    }
    updated_by_id = {
        str(step.id or step.condition_id or "").strip().upper(): step.to_dict()
        for step in list(updated_plan.judge_steps or [])
    }
    violations: List[Dict[str, Any]] = []
    for step_id in sorted(set(previous_by_id) | set(updated_by_id)):
        if step_id not in previous_by_id:
            if not scope.get("allow_add_step"):
                violations.append({"type": "step_added", "step_id": step_id})
            continue
        if step_id not in updated_by_id:
            if not (scope.get("allow_remove_step") and step_id in target_steps):
                violations.append({"type": "step_removed", "step_id": step_id})
            continue
        if (
            step_id not in target_steps
            and stable_json_dumps(previous_by_id[step_id])
            != stable_json_dumps(updated_by_id[step_id])
        ):
            changed_fields = sorted(
                field
                for field in set(previous_by_id[step_id]) | set(updated_by_id[step_id])
                if stable_json_dumps(previous_by_id[step_id].get(field))
                != stable_json_dumps(updated_by_id[step_id].get(field))
            )
            violations.append({
                "type": "out_of_scope_step_changed",
                "step_id": step_id,
                "changed_fields": changed_fields,
            })
    if (
        not scope.get("allow_emit_logic_change")
        and str(updated_plan.emit_logic or "") != str(previous_plan.emit_logic or "")
    ):
        violations.append({"type": "emit_logic_changed"})
    return violations


def _constrained_step_dict(step_data: Dict[str, Any], *, attack_label: str = "") -> Dict[str, Any]:
    step = JudgeStep.from_dict(step_data)
    constrain_judge_step_first_pass(step, attack_label=attack_label)
    round_policy = followup_round_policy(
        step,
        attack_label=attack_label,
        condition_id=str(step.condition_id or step.id or ""),
    )
    step.max_followups = max(
        int(round_policy.get("minimum_followups", 0) or 0),
        min(int(round_policy["max_followups"]), int(step.max_followups or 0)),
    )
    return step.to_dict()


def _extract_condition_ids(text: str) -> set[str]:
    return {
        match.upper()
        for match in re.findall(r"\b[CE]\d+\b", str(text or ""), flags=re.IGNORECASE)
    }


def _apply_simple_suggestion(
    plan: EvidencePlan,
    suggestion: Dict[str, Any],
    *,
    scope: Dict[str, Any] | None = None,
    attack_label: str = "",
) -> None:
    condition_id = str(suggestion.get("condition_id") or "").strip().upper()
    if not condition_id:
        return
    if scope and scope.get("scoped"):
        targets = {str(item or "").strip().upper() for item in scope.get("target_steps", [])}
        if condition_id not in targets:
            return
    step = _find_step(plan, condition_id)
    if step is None:
        return
    text = " ".join(
        str(suggestion.get(key) or "")
        for key in ("proposed_change", "rationale")
    )
    candidate_views = _extract_known_views(text)
    if candidate_views:
        action = str(suggestion.get("action") or "").strip().lower()
        compact_views = [view for view in candidate_views if view not in _LARGE_DEFAULT_DEFER_VIEWS]
        broad_views = [view for view in candidate_views if view in _LARGE_DEFAULT_DEFER_VIEWS]
        if "reentrancy_state_order_view" in broad_views and "reentrancy_state_order_summary_view" not in compact_views:
            compact_views.append("reentrancy_state_order_summary_view")
        if action == "change_default_views":
            step.default_evidence_refs = _bounded_defaults(
                step.default_evidence_refs,
                compact_views,
                step=step,
                attack_label=attack_label,
            )
            step.evidence_refs = list(step.default_evidence_refs)
            step.allowed_followup_views = _merge(step.allowed_followup_views, broad_views)
        elif action == "add_default_views":
            step.default_evidence_refs = _bounded_defaults(
                step.default_evidence_refs,
                compact_views,
                append=True,
                step=step,
                attack_label=attack_label,
            )
            step.evidence_refs = list(step.default_evidence_refs)
            step.allowed_followup_views = _merge(step.allowed_followup_views, broad_views)
        elif action == "change_followup_views":
            step.allowed_followup_views = _merge(step.allowed_followup_views, candidate_views)
        elif action == "add_followup_views":
            step.allowed_followup_views = _merge(step.allowed_followup_views, candidate_views)
        else:
            step.allowed_followup_views = _merge(step.allowed_followup_views, candidate_views)
    if "read_function_chunk" in text and "read_function_chunk" not in step.allowed_tools:
        step.allowed_tools.append("read_function_chunk")
    source_level = source_dependency_level(f"{step.question} {text}", attack_label)
    label = normalize_attack_label(attack_label, default="")
    label_core_step = label in {"access_control", "insufficient_validation"} and str(
        step.condition_id or step.id or ""
    ).strip().upper().startswith("C")
    if label_core_step and "read_function_chunk" not in step.allowed_tools:
        step.allowed_tools.append("read_function_chunk")
    if "read_function_chunk" in step.allowed_tools and (source_level == "required" or label_core_step):
        for tool in ("read_packet_view", "read_evidence_by_id", "read_evidence_context"):
            if tool not in step.allowed_tools:
                step.allowed_tools.append(tool)
        step.max_followups = max(step.max_followups, 2)
    elif "read_function_chunk" in step.allowed_tools:
        for tool in ("read_packet_view", "read_evidence_by_id", "read_evidence_context"):
            if tool not in step.allowed_tools:
                step.allowed_tools.append(tool)
        step.max_followups = min(max(step.max_followups, 1), 1)
    if "get_local_call_context" in text and "get_local_call_context" not in step.allowed_tools:
        step.allowed_tools.append("get_local_call_context")
        step.max_followups = max(step.max_followups, 1)
    if source_level in {"helpful", "required"}:
        for tool in ("read_packet_view", "read_evidence_by_id", "read_evidence_context"):
            if tool not in step.allowed_tools:
                step.allowed_tools.append(tool)
    if source_level == "required" or label_core_step:
        step.max_followups = max(step.max_followups, 2)
    elif source_level == "helpful":
        step.max_followups = max(step.max_followups, 1)
    constrain_judge_step_first_pass(step, attack_label=attack_label)
    round_policy = followup_round_policy(
        step,
        attack_label=attack_label,
        condition_id=str(step.condition_id or step.id or ""),
    )
    step.max_followups = max(
        int(round_policy.get("minimum_followups", 0) or 0),
        min(int(round_policy["max_followups"]), int(step.max_followups or 0)),
    )


def _bounded_defaults(
    existing: List[str],
    items: List[str],
    *,
    append: bool = False,
    step: JudgeStep | None = None,
    attack_label: str = "",
) -> List[str]:
    base = list(existing or []) if append else [
        view
        for view in (
            "operation_summary_view",
            "evidence_adequacy_view",
            "classification_digest_view",
        )
        if view in (existing or []) or view in items
    ]
    merged = _merge(base, items)
    compact = [view for view in merged if view not in _LARGE_DEFAULT_DEFER_VIEWS]
    budget = condition_view_budget(step or " ".join(compact), attack_label=attack_label)
    selected, _, _ = constrain_first_pass_view_refs(
        compact,
        [],
        max_default_views=int(budget.get("max_default_views", 4) or 4),
        max_large_views=int(budget.get("max_large_views", 1) or 1),
        max_followup_views=int(budget.get("max_followup_views", 5) or 5),
    )
    return selected


def _find_step(plan: EvidencePlan, condition_id: str) -> JudgeStep | None:
    for step in plan.judge_steps:
        if str(step.condition_id or step.id).strip().upper() == condition_id:
            return step
        if str(step.id).strip().upper() == condition_id:
            return step
    return None


def _extract_known_views(text: str) -> List[str]:
    from evotx.runtime.view_catalog import PACKET_VIEW_CATALOG

    lowered = str(text or "")
    return [view for view in PACKET_VIEW_CATALOG if view in lowered]


def _merge(existing: List[str], items: List[str]) -> List[str]:
    merged = list(existing or [])
    for item in items:
        if item not in merged:
            merged.append(item)
    return merged


def _plan_update_audit_summary(
    previous: EvidencePlan,
    updated: EvidencePlan | None,
) -> Dict[str, Any]:
    if updated is None:
        return {
            "updated_plan_present": False,
            "updated_plan_id": "",
            "judge_step_count": 0,
            "changed_step_ids": [],
        }
    previous_by_id = {
        step.id: stable_json_dumps(step.to_dict())
        for step in list(previous.judge_steps or [])
    }
    changed: List[str] = []
    for step in list(updated.judge_steps or []):
        if previous_by_id.get(step.id) != stable_json_dumps(step.to_dict()):
            changed.append(step.id)
    return {
        "updated_plan_present": True,
        "updated_plan_id": updated.plan_id,
        "judge_step_count": len(updated.judge_steps),
        "changed_step_ids": changed,
    }


def _stamp_plan_metadata(
    plan: EvidencePlan,
    rule: EvolvingRule,
    *,
    previous: EvidencePlan | None,
    source: str,
) -> None:
    previous_version = 0
    if previous is not None:
        try:
            previous_version = int((previous.metadata or {}).get("plan_version") or 1)
        except Exception:
            previous_version = 1
    plan.metadata["plan_version"] = previous_version + 1
    plan.metadata["source"] = source
    plan.metadata["compatible_rule_fingerprint"] = rule_fingerprint(rule)
    plan.metadata["previous_plan_fingerprint"] = (
        plan_fingerprint(previous) if previous is not None else ""
    )
    plan.metadata["plan_fingerprint"] = plan_fingerprint(plan)


def _cohort_signal_gate_enabled(review_bundle: Dict[str, Any]) -> bool:
    return bool(
        ((review_bundle or {}).get("cohort_signal_summary") or {}).get(
            "schema_version"
        )
    )


def _review_has_actionable_plan_signal(review: Dict[str, Any]) -> bool:
    return any(
        isinstance(signal, dict)
        and str(signal.get("generalization_status") or "").lower().strip()
        in {"supported", "experimental_plan"}
        and str(signal.get("update_target") or review.get("update_target") or "")
        == "plan"
        for signal in list((review or {}).get("update_signals", []) or [])
    )


def _review_allows_plan_training(review: Dict[str, Any]) -> bool:
    error_type = str((review or {}).get("error_type") or "").strip().upper()
    target_observed = str(
        (review or {}).get("target_mechanism_observed") or ""
    ).strip().lower()
    gt_supported = str(
        (review or {}).get("ground_truth_supported_by_packet") or ""
    ).strip().lower()
    training_gate = dict((review or {}).get("training_gate") or {})
    blocked = dict(training_gate.get("blocked_reasons") or {})
    if str(blocked.get("plan") or "").strip():
        return False
    # Phase 2/result_slimmer owns the training contract. A false-positive Plan
    # boundary repair may be valid even when the target mechanism is absent; do
    # not re-block it here with a coarser legacy gt_supported check.
    if gt_supported == "no" and error_type != "FP":
        return False
    if error_type == "FN" and target_observed == "no":
        return False
    return True


def _experimental_plan_signal_map(
    review_bundle: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(signal.get("signal_id") or ""): dict(signal)
        for signal in list(
            ((review_bundle or {}).get("update_signal_matrix") or {}).get(
                "signals", []
            )
            or []
        )
        if isinstance(signal, dict)
        and str(signal.get("signal_id") or "")
        and str(signal.get("generalization_status") or "").strip().lower()
        == "experimental_plan"
    }


def _experimental_plan_payload_violations(
    payload: Dict[str, Any],
    review_bundle: Dict[str, Any],
) -> List[Dict[str, Any]]:
    experimental = _experimental_plan_signal_map(review_bundle)
    if not experimental:
        return []
    applied = set(_string_list((payload or {}).get("applied_signal_ids")))
    patches = [
        dict(item)
        for key in ("plan_patches", "step_patches")
        for item in list((payload or {}).get(key) or [])
        if isinstance(item, dict)
    ]
    for patch in patches:
        applied.update(_string_list(patch.get("applied_signal_ids")))
    used_experimental = applied.intersection(experimental)
    if not used_experimental:
        return []
    violations: List[Dict[str, Any]] = []
    if not patches:
        violations.append({
            "reason": "experimental_plan_requires_patch_schema",
            "signal_ids": sorted(used_experimental),
        })
        return violations

    patch_experimental_ids = {
        signal_id
        for patch in patches
        for signal_id in _string_list(patch.get("applied_signal_ids"))
        if signal_id in experimental
    }
    missing_patch_attribution = sorted(
        used_experimental - patch_experimental_ids
    )
    if missing_patch_attribution:
        violations.append({
            "reason": "experimental_signal_must_be_attributed_to_target_patch",
            "signal_ids": missing_patch_attribution,
        })

    allowed_fields = {
        "evidence_refs",
        "default_evidence_refs",
        "allowed_followup_views",
        "allowed_tools",
        "max_followups",
    }
    for patch in patches:
        patch_ids = set(_string_list(patch.get("applied_signal_ids")))
        experimental_ids = patch_ids.intersection(experimental)
        if not experimental_ids:
            continue
        condition_id = str(
            patch.get("condition_id")
            or patch.get("step_id")
            or patch.get("id")
            or ""
        ).strip().upper()
        signal_conditions = {
            str(experimental[signal_id].get("condition_id") or "").strip().upper()
            for signal_id in experimental_ids
        }
        if not condition_id or condition_id not in signal_conditions:
            violations.append({
                "reason": "experimental_plan_target_condition_mismatch",
                "signal_ids": sorted(experimental_ids),
                "condition_id": condition_id,
                "signal_condition_ids": sorted(signal_conditions),
            })
        touched = _plan_patch_touched_fields(patch)
        disallowed = sorted(touched - allowed_fields)
        if disallowed:
            violations.append({
                "reason": "experimental_plan_may_only_change_evidence_routing",
                "signal_ids": sorted(experimental_ids),
                "condition_id": condition_id,
                "disallowed_fields": disallowed,
            })
    return violations


def _experimental_plan_views_by_condition(
    review_bundle: Dict[str, Any],
) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for signal in _experimental_plan_signal_map(review_bundle).values():
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        if not condition_id:
            continue
        text = " ".join(
            str(signal.get(key) or "")
            for key in ("abstract_feature", "rationale")
        )
        out.setdefault(condition_id, set()).update(_extract_known_views(text))
    return out


def _plan_payload_uses_actionable_signals(
    payload: Dict[str, Any],
    review_bundle: Dict[str, Any],
) -> bool:
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    raw_actionable_ids = matrix.get("plan_actionable_signal_ids")
    if raw_actionable_ids is None:
        raw_actionable_ids = matrix.get("supported_signal_ids", [])
    actionable = {
        str(value)
        for value in list(raw_actionable_ids or [])
        if str(value)
    }
    raw_applied = (payload or {}).get("applied_signal_ids", [])
    if isinstance(raw_applied, str):
        raw_applied = [raw_applied]
    applied = {str(value) for value in list(raw_applied or []) if str(value)}
    for patch in list((payload or {}).get("plan_patches") or []):
        if not isinstance(patch, dict):
            continue
        patch_ids = patch.get("applied_signal_ids", [])
        if isinstance(patch_ids, str):
            patch_ids = [patch_ids]
        applied.update(str(value) for value in list(patch_ids or []) if str(value))
    for patch in list((payload or {}).get("step_patches") or []):
        if not isinstance(patch, dict):
            continue
        patch_ids = patch.get("applied_signal_ids", [])
        if isinstance(patch_ids, str):
            patch_ids = [patch_ids]
        applied.update(str(value) for value in list(patch_ids or []) if str(value))
    if not applied or not applied.issubset(actionable):
        return False
    for group in list(matrix.get("joint_resolution_groups") or []):
        if not isinstance(group, dict):
            continue
        if str(group.get("update_target") or "").lower().strip() != "plan":
            continue
        group_ids = {
            str(value)
            for value in list(group.get("signal_ids") or [])
            if str(value)
        }
        if applied & group_ids and not group_ids.issubset(applied):
            return False
    return True


# Backward-compatible private alias retained for existing callers/tests. Plan
# actionability now includes tightly scoped experimental evidence-route signals.
def _plan_payload_uses_supported_signals(
    payload: Dict[str, Any],
    review_bundle: Dict[str, Any],
) -> bool:
    return _plan_payload_uses_actionable_signals(payload, review_bundle)


def _plan_payload_contains_case_specific_terms(payload: Dict[str, Any]) -> bool:
    text = stable_json_dumps(payload or {}).lower()
    return bool(re.search(
        r"0x[0-9a-f]{6,}|\b(?:call|event|transfer|sload|sstore):\d+\b|"
        r"\b(?:ctf|challenge[- ]?like|joke selector|codeislaw)\b",
        text,
    ))
