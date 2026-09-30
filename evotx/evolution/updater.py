from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from evotx.core.labels import normalize_attack_label
from evotx.core.rule import baseline_relative_rule_budget_decision, rule_complexity
from evotx.core.schemas import EvolvingRule, RuleCondition
from evotx.evolution.negative_training import (
    negative_case_boundary_guidance,
    negative_training_mode_guidance,
    normalize_negative_training_mode,
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
    extract_json_object,
    repair_likely_mojibake,
    stable_json_dumps,
)


UPDATER_CANONICAL_SIGNAL_GUIDANCE = """
Canonical signal contract:
- update_signal_matrix has already decided status, owner, condition, direction,
  and scope. Do not reclassify a signal or reconstruct Reviewer diagnosis.
- Apply only the canonical actionable signal_ids supplied for this owner. Do not
  infer a Rule update from absent, conflicted, insufficient, Plan, Packet, or
  Runtime signals.
- Do not concatenate opposite directions for the same semantic feature.
  joint_resolution_groups, when present, are the only atomic multi-signal
  synthesis contract.
- If the canonical signals cannot produce one boundary-compatible local edit,
  return no rule_operations. Do not invent another direction from prompt text.
- Preserve every unrelated condition/exclusion exactly. Structural operations
  are allowed only through the typed rule_operations contract, supported
  signal attribution, deterministic scope checks, and configured feature flags.
- Preserve the unrelated condition/exclusion interface exactly; only the
  explicitly scoped typed operation may alter its target semantic unit.
- Existing ids and expected_answer are frozen; add operations receive a new
  deterministic id and expected_answer=true.
- The executable logic remains all positive conditions AND NOT(any exclusion).
  Do not simulate an alternate OR branch by weakening an existing condition.
- Do not copy protocol names, function names, or protocol-specific argument
  names such as onBehalf/receiver/market into the rule. Abstract them as actor,
  signer, beneficiary, target, authorization scope, validity window, or action
  parameters.
- For protocol accounting exploitation, preserve one shared protocol-internal
  bookkeeping chain: bookkeeping candidate -> same-candidate consumption or
  misuse -> candidate-specific accounting outcome. Direct token transfer is not
  required when state-level reward/share/vault/debt/collateral/staking/claim/
  liability impact is evidence-backed, but generic profit, ordinary accounting
  mechanics, reentrancy/state-order abuse, insufficient validation, token
  semantics, market/oracle effects, flash capital, or access control must not
  substitute for that candidate chain.
- For PAE exclusions or boundary tightening, state which required bookkeeping
  candidate/consumption/outcome requirement is absent, normal/entitlement-backed,
  or completely replaced by another primary root cause. Do not encode the
  negative source-family label itself.
- In update_note, cite only applied canonical signal ids and the boundary or
  active-refinement constraint that shaped the edit.
- Before returning, counterfactually test the proposed rule against benign and
  unseen non-target attacks sharing generic profit, flash-loan, callback,
  unknown-selector, public-entry, or asymmetric-value symptoms.
"""

UPDATER_LEGACY_REVIEW_GUIDANCE = """
Legacy review conflict safety:
- Reconcile opposite requested directions before editing the same condition.
- If the supplied reviews cannot support one local boundary-compatible edit,
  return the current rule unchanged.
- Preserve unrelated conditions, exclusions, ids, and expected_answer values.
"""


def _rule_updater_label_guidance(rule: EvolvingRule) -> str:
    label = normalize_attack_label(
        (rule.metadata or {}).get("attack_label", ""),
        default="",
    )
    if label != "price_manipulation":
        return ""
    return """Price-manipulation update boundary:
- Do not import the broader market_manipulation definition. Skim/donate,
  sandwich/MEV ordering, generic pool-accounting anomalies, swaps, flash
  capital, and profit are not sufficient price-manipulation semantics.
- A valid price path may begin with token-accounting or operation-order behavior
  that distorts a reserve/price, then use that distortion in a same-market swap,
  settlement, or extraction. Do not require the consumer to be a separate
  external protocol when causal price-sensitive consumption is evidenced.
- Preserve condition ownership: C1 owns the abnormal, value-backed/unbacked
  distortion boundary; C2 owns causal price-sensitive consumption and its
  amount/limit/settlement effect. C2 may reference the selected C1 distortion
  but must not copy the complete C1 boundary into a second semantic gate.
- Preserve the difference between causal same-market consumption and ordinary
  sequential AMM repricing. The actor's later swap observing the normal reserve
  update from its earlier swap is not sufficient unless an independently
  abnormal distortion demonstrably changed amount, limit, settlement, or
  extraction.
- Preserve the negative boundary: bare skim/sync/reserve writing and ordinary
  swaps without use of the distortion must remain insufficient.
- Prefer one causal semantic refinement covering downstream and same-market
  consumers over declaring an alternate OR branch. Do not weaken consumption
  into mere operation co-occurrence."""


class RuleUpdater:
    """Update and optionally compress evolving rules from review results."""

    def __init__(
        self,
        llm=None,
        transcript_dir: str | Path | None = None,
        *,
        enable_structural_evolution: bool = True,
        enable_remove_positive_condition: bool = False,
        enable_remove_exclusion: bool = False,
    ):
        self.llm = llm
        self.transcript_dir = transcript_dir
        self.enable_structural_evolution = bool(enable_structural_evolution)
        self.enable_remove_positive_condition = bool(
            enable_remove_positive_condition
        )
        self.enable_remove_exclusion = bool(enable_remove_exclusion)
        self.last_atomic_delta_manifest: List[Dict[str, Any]] = []
        self.last_structural_update_audit: Dict[str, Any] = {}

    def update_rule(
        self,
        rule: EvolvingRule,
        reviews: List[Dict[str, Any]] | Dict[str, Any],
        allow_noop: bool = True,
    ) -> EvolvingRule:
        if isinstance(reviews, dict):
            return self.update_rule_from_bundle(rule, reviews, allow_noop=allow_noop)

        actionable = [
            r
            for r in reviews
            if r.get("should_update_rule") and r.get("update_target", "rule") == "rule"
            and _review_allows_rule_training(r)
        ]
        if not actionable and allow_noop:
            return rule
        if self.llm is None:
            return self._fallback_update(rule, actionable)

        prompt = self._build_update_prompt(rule, actionable)
        raw_completion = ""
        data: Dict[str, Any] | None = None
        try:
            raw_completion = self.llm.complete(prompt)
            data = repair_likely_mojibake(
                extract_json_object(llm_response_text(raw_completion))
            )
            update_scope = _infer_rule_update_scope(actionable)
            if update_scope.get("scoped"):
                data = _apply_minimal_update_scope(rule, data, update_scope)
            data = _freeze_rule_update_interface(rule, data)
            updated = self._rule_from_update_payload(rule, data)
            final_rule = self._enforce_rule_budget(
                rule,
                updated,
                update_payload=data,
                update_scope=update_scope,
                allow_noop=allow_noop,
            )
            self._write_audit(
                rule=rule,
                prompt=prompt,
                raw_completion=raw_completion,
                parsed_response=data,
                updated_rule=final_rule,
                reviews=actionable,
                review_bundle=None,
                allow_noop=allow_noop,
                mode="list_update",
            )
            return final_rule
        except Exception as exc:
            self._write_audit(
                rule=rule,
                prompt=prompt,
                raw_completion=raw_completion,
                parsed_response=data,
                updated_rule=None,
                reviews=actionable,
                review_bundle=None,
                allow_noop=allow_noop,
                mode="list_update",
                error=repr(exc),
            )
            if allow_noop:
                return rule
            raise

    def update_rule_from_bundle(
        self,
        rule: EvolvingRule,
        review_bundle: Dict[str, Any],
        allow_noop: bool = True,
    ) -> EvolvingRule:
        self.last_atomic_delta_manifest = []
        self.last_structural_update_audit = {}
        cohort_gate_enabled = _cohort_signal_gate_enabled(review_bundle)
        canonical_contract = _canonical_signal_contract_enabled(review_bundle)
        allowed_signal_ids = _actionable_rule_signal_ids(review_bundle)
        actionable = [
            r
            for r in list(review_bundle.get("actionable_rule_reviews", []) or [])
            if r.get("update_target") == "rule" and r.get("should_update_rule")
            and (
                (
                    canonical_contract
                    and _review_uses_signal_ids(r, allowed_signal_ids)
                )
                or (
                    not canonical_contract
                    and _review_allows_rule_training(r)
                    and (
                        not cohort_gate_enabled
                        or _review_has_supported_signal(r, update_target="rule")
                    )
                )
            )
        ]
        if not actionable and allow_noop:
            return rule
        if self.llm is None:
            return self._fallback_update(rule, actionable)

        prompt_bundle = _compact_review_bundle_for_update(
            review_bundle,
            actionable,
            update_target="rule",
        )
        prompt_bundle["rule_structural_feature_flags"] = {
            "enable_structural_evolution": self.enable_structural_evolution,
            "enable_remove_positive_condition": (
                self.enable_remove_positive_condition
            ),
            "enable_remove_exclusion": self.enable_remove_exclusion,
        }
        prompt_actionable = list(prompt_bundle.get("prompt_actionable_reviews") or [])
        prompt = self._build_bundle_update_prompt(rule, prompt_bundle, prompt_actionable)
        raw_completion = ""
        data: Dict[str, Any] | None = None
        try:
            raw_completion = self.llm.complete(prompt)
            data = repair_likely_mojibake(
                extract_json_object(llm_response_text(raw_completion))
            )
            if cohort_gate_enabled and (
                not _payload_uses_actionable_rule_signals(data, review_bundle)
                or _payload_contains_case_specific_update_terms(data)
            ):
                data["generalization_gate_rejected"] = True
                data["generalization_gate_rejected_reason"] = (
                    "unsupported_signal_ids_or_case_specific_output"
                )
                self._write_audit(
                    rule=rule,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=data,
                    updated_rule=rule,
                    reviews=actionable,
                    review_bundle=prompt_bundle,
                    allow_noop=allow_noop,
                    mode="bundle_update",
                )
                return rule
            update_scope = _infer_rule_update_scope(actionable)
            update_scope = _configured_rule_update_scope(
                update_scope,
                enable_structural_evolution=self.enable_structural_evolution,
                enable_remove_positive_condition=(
                    self.enable_remove_positive_condition
                ),
                enable_remove_exclusion=self.enable_remove_exclusion,
            )
            data, operation_audit = _materialize_typed_rule_operations(
                rule,
                data,
                update_scope=update_scope,
                review_bundle=review_bundle,
            )
            self.last_atomic_delta_manifest = list(
                operation_audit.get("accepted_operations") or []
            )
            self.last_structural_update_audit = operation_audit
            if update_scope.get("scoped"):
                data = _apply_minimal_update_scope(
                    rule,
                    data,
                    update_scope,
                )
            data = _guard_rule_update_interface(
                rule,
                data,
                update_scope=update_scope,
            )
            if not _rule_payload_has_semantic_change(rule, data):
                data["semantic_noop"] = True
                self._write_audit(
                    rule=rule,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=data,
                    updated_rule=rule,
                    reviews=actionable,
                    review_bundle=prompt_bundle,
                    allow_noop=allow_noop,
                    mode="bundle_update",
                )
                return rule
            updated = self._rule_from_update_payload(rule, data)
            final_rule = self._enforce_rule_budget(
                rule,
                updated,
                update_payload=data,
                update_scope=update_scope,
                allow_noop=allow_noop,
            )
            self._write_audit(
                rule=rule,
                prompt=prompt,
                raw_completion=raw_completion,
                parsed_response=data,
                updated_rule=final_rule,
                reviews=actionable,
                review_bundle=prompt_bundle,
                allow_noop=allow_noop,
                mode="bundle_update",
            )
            return final_rule
        except Exception as exc:
            self._write_audit(
                rule=rule,
                prompt=prompt,
                raw_completion=raw_completion,
                parsed_response=data,
                updated_rule=None,
                reviews=actionable,
                review_bundle=prompt_bundle,
                allow_noop=allow_noop,
                mode="bundle_update",
                error=repr(exc),
            )
            if allow_noop:
                return rule
            raise

    def _write_audit(
        self,
        *,
        rule: EvolvingRule,
        prompt: str,
        raw_completion: Any,
        parsed_response: Any,
        updated_rule: EvolvingRule | None,
        reviews: List[Dict[str, Any]],
        review_bundle: Dict[str, Any] | None,
        allow_noop: bool,
        mode: str,
        error: str = "",
    ) -> None:
        attack_label = str((rule.metadata or {}).get("attack_label") or "")
        bundle_chars = len(stable_json_dumps(review_bundle if review_bundle is not None else reviews))
        truncation = build_truncation_audit(
            input_name="review_bundle" if review_bundle is not None else "reviews",
            before=review_bundle if review_bundle is not None else reviews,
            after=review_bundle if review_bundle is not None else reviews,
        )
        updated_summary = _rule_update_audit_summary(rule, updated_rule)
        boundary_summary = dict((review_bundle or {}).get("candidate_boundary_summary") or {})
        name = f"{attack_label or rule.rule_id}__rule_update__v{rule.version}__{mode}"
        path = write_llm_audit_transcript(
            output_dir=self.transcript_dir,
            stage="rule_updater",
            name=name,
            model=llm_model_name(self.llm),
            prompt=prompt,
            raw_completion=raw_completion,
            parsed_response={
                "parsed_response": parsed_response,
                "updated_rule_summary": updated_summary,
            },
            usage=llm_usage(self.llm, raw_completion),
            finish_reason=llm_finish_reason(self.llm, raw_completion),
            metadata={
                "attack_label": attack_label,
                "current_rule_id": rule.rule_id,
                "current_rule_version": rule.version,
                "review_count": len(list(reviews or [])),
                "actionable_rule_review_count": len(list(reviews or [])),
                "review_bundle_chars": bundle_chars,
                "prompt_chars": len(str(prompt or "")),
                "current_summary": (review_bundle or {}).get("current_summary", {}),
                "mode": mode,
                "allow_noop": bool(allow_noop),
                "error": error,
                "repair_context": (
                    "new_fn_recall_repair" if boundary_summary else ""
                ),
                "candidate_boundary_summary_present": bool(boundary_summary),
                "fixed_fp_count": int(boundary_summary.get("fixed_fp_count", 0) or 0),
                "new_fn_count": int(boundary_summary.get("new_fn_count", 0) or 0),
                "guard_fixed_fp_count": int(
                    boundary_summary.get("guard_fixed_fp_count", 0) or 0
                ),
                **updated_summary,
            },
            truncation=truncation,
        )
        if self.transcript_dir is not None:
            print(
                "[LLM-Audit] rule_updater "
                f"label={attack_label or 'unknown'} "
                f"prompt_chars={len(str(prompt or ''))} "
                f"est_tokens={estimate_prompt_tokens(prompt)} "
                f"bundle_chars={bundle_chars} "
                f"truncated={str(bool(truncation.get('applied'))).lower()} "
                f"path={path or ''}"
            )

    def compress_rule(self, rule: EvolvingRule) -> EvolvingRule:
        if self.llm is None:
            return rule
        prompt = f"""
You are compressing an EvoTx evolving rule without changing its semantics.

Remove redundant wording while preserving the rule/plan interface exactly.

Hard constraints:
- Preserve every condition id exactly.
- Preserve every exclusion condition id exactly.
- Do not add, remove, merge, split, or reorder conditions or exclusions.
- Do not change expected_answer.
- Do not change decision_policy except wording-only simplification that
  preserves referenced ids and the same boolean meaning.
- Keep every condition judgeable as a local yes/no question.
Do not add evidence_hints, packet views, tool names, or concrete identifiers.
Return strict JSON in the same schema used by the updater.

Current rule:
{stable_json_dumps(rule.to_dict())}
"""
        data = repair_likely_mojibake(
            extract_json_object(self.llm.complete(prompt))
        )
        compressed = self._rule_from_update_payload(rule, data, note_prefix="compression")
        if not _compression_preserves_rule_interface(rule, compressed):
            rejected = dict(rule.metadata or {})
            rejected["last_compression_rejected"] = {
                "reason": "condition_or_exclusion_id_or_expected_answer_changed",
                "original_condition_ids": [condition.id for condition in rule.conditions],
                "compressed_condition_ids": [condition.id for condition in compressed.conditions],
                "original_exclusion_ids": [condition.id for condition in rule.exclusion_conditions],
                "compressed_exclusion_ids": [
                    condition.id for condition in compressed.exclusion_conditions
                ],
            }
            rule.metadata = rejected
            return rule
        compressed.metadata = {
            **dict(compressed.metadata or {}),
            "rule_complexity": rule_complexity(compressed),
        }
        return compressed

    def _enforce_rule_budget(
        self,
        previous: EvolvingRule,
        updated: EvolvingRule,
        *,
        update_payload: Dict[str, Any],
        update_scope: Dict[str, Any],
        allow_noop: bool = True,
    ) -> EvolvingRule:
        complexity = rule_complexity(updated)
        metadata = dict(updated.metadata or {})
        metadata["rule_complexity"] = complexity
        metadata["rule_budget"] = {
            "enabled": True,
            "allow_over_budget": bool(metadata.get("allow_over_budget", False)),
            "hard_over_budget": bool(complexity.get("hard_over_budget")),
            "soft_over_budget": bool(complexity.get("soft_over_budget")),
            "budget_status": complexity.get("budget_status", "ok"),
        }
        updated.metadata = metadata
        if metadata["rule_budget"]["allow_over_budget"]:
            return updated
        if not complexity.get("hard_over_budget"):
            if complexity.get("soft_over_budget"):
                metadata["rule_budget"].update({
                    "allowed_with_soft_warning": True,
                    "reason": "source_dependency_budget_warning_allowed",
                    "soft_over_budget_reasons": list(
                        complexity.get("soft_over_budget_reasons", [])
                    ),
                    "hard_over_budget_reasons": list(
                        complexity.get("hard_over_budget_reasons", [])
                    ),
                })
                updated.metadata = metadata
            return updated

        previous_complexity = rule_complexity(previous)
        baseline_decision = baseline_relative_rule_budget_decision(
            previous_complexity,
            complexity,
        )
        metadata["rule_budget"]["baseline_relative"] = baseline_decision
        if baseline_decision["baseline_relative_grandfathered"]:
            metadata["rule_budget"].update({
                "baseline_relative_grandfathered": True,
                "reason": "candidate_does_not_worsen_inherited_hard_overage",
            })
            updated.metadata = metadata
            return updated

        consolidated = self._consolidate_rule_to_budget(previous, updated)
        if consolidated is not None and not rule_complexity(consolidated).get("hard_over_budget"):
            consolidated_metadata = dict(consolidated.metadata or {})
            consolidated_metadata["rule_complexity"] = rule_complexity(consolidated)
            consolidated_metadata["rule_budget"] = {
                "enabled": True,
                "consolidated": True,
                "rejected_initial_complexity": complexity,
                "scope_guard": dict(update_scope or {}),
            }
            consolidated.metadata = consolidated_metadata
            return consolidated

        rejected_rule = deepcopy(previous)
        rejected_metadata = dict(rejected_rule.metadata or {})
        rejected_metadata["last_rule_budget_rejected"] = {
            "reason": "hard_rule_complexity_budget_exceeded",
            "legacy_reason": "candidate_rule_over_budget",
            "candidate_complexity": complexity,
            "previous_complexity": previous_complexity,
            "baseline_relative": baseline_decision,
            "scope_guard": dict(update_scope or {}),
            "allow_new_conditions": bool((update_scope or {}).get("allow_new_conditions")),
            "allow_new_exclusions": bool((update_scope or {}).get("allow_new_exclusions")),
            "update_note": str((update_payload or {}).get("update_note") or ""),
        }
        rejected_rule.metadata = rejected_metadata
        return rejected_rule

    def _consolidate_rule_to_budget(
        self,
        previous: EvolvingRule,
        over_budget_rule: EvolvingRule,
    ) -> EvolvingRule | None:
        if self.llm is None:
            return None
        prompt = self._build_consolidation_prompt(previous, over_budget_rule)
        try:
            data = repair_likely_mojibake(
                extract_json_object(self.llm.complete(prompt))
            )
            # Budget consolidation may shorten wording, but it cannot become a
            # second unscoped structural updater that bypasses operation flags
            # or signal attribution.
            data = _freeze_rule_update_interface(over_budget_rule, data)
            consolidated = self._rule_from_update_payload(
                previous,
                data,
                note_prefix="budget_consolidation",
            )
            metadata = dict(consolidated.metadata or {})
            metadata["requires_plan_regeneration"] = (
                [c.id for c in consolidated.conditions] != [c.id for c in over_budget_rule.conditions]
                or [c.id for c in consolidated.exclusion_conditions]
                != [c.id for c in over_budget_rule.exclusion_conditions]
            )
            consolidated.metadata = metadata
            return consolidated
        except Exception:
            return None

    @staticmethod
    def _build_consolidation_prompt(
        previous: EvolvingRule,
        over_budget_rule: EvolvingRule,
    ) -> str:
        return f"""
You are consolidating an EvoTx rule candidate that exceeded the few-shot
execution budget. Keep the target attack-family semantics, but merge
supporting symptoms into shorter wording inside the existing condition slots.

Budget:
- Shorten condition descriptions without changing the slot count.
- Hard default budget remains max 3 core, max 2 exclusions, max 5 total
  conditions; a structural candidate beyond that count must be rejected by the
  caller rather than silently rewritten here.
- Source-dependent overage may be a label-aware warning for source-heavy
  families such as access_control and insufficient_validation; do not add
  conditions to satisfy that warning.

Consolidation constraints:
- Prefer 3 core conditions shaped as root cause / consumption-or-trigger / outcome.
- Do not turn flash loan, profit, swap, callback, unknown selector, gas/call
  complexity, or other supporting evidence into standalone conditions unless it
  is the target-family root cause.
- Preserve existing C/E ids when possible.
- Preserve the candidate condition/exclusion ids and order exactly. This pass
  may shorten or merge wording within an existing slot, but must not add,
  remove, merge, split, or reorder slots.
- Keep each condition judgeable as a local yes/no question.
- Do not output packet view names, tool names, concrete tx hashes, addresses,
  call ids, event ids, transfer ids, or storage slots.
- Return strict JSON in the updater schema only.

Previous accepted rule:
{stable_json_dumps(previous.to_dict())}

Over-budget candidate:
{stable_json_dumps(over_budget_rule.to_dict())}

Return JSON:
{{
  "description": "consolidated rule description",
  "conditions": [
    {{"id": "C1", "description": "...", "expected_answer": true}}
  ],
  "exclusion_conditions": [
    {{"id": "E1", "description": "...", "expected_answer": true}}
  ],
  "decision_policy": "how to combine conditions and exclusions",
  "update_note": "merged support-only conditions to satisfy rule budget",
  "metadata": {{"requires_plan_regeneration": false}}
}}
"""
    @staticmethod
    def _build_update_prompt(rule: EvolvingRule, reviews: List[Dict[str, Any]]) -> str:
        feature_diagnosis_summary = _feature_diagnosis_summary(reviews)
        return f"""
You are an EvoTx evolving transaction detection rule updater.

{UPDATER_LEGACY_REVIEW_GUIDANCE}

{_rule_updater_label_guidance(rule)}

Update the rule based on the root-cause analyses.

Requirements:
- Update the rule, not model parameters.
- Do not hardcode a fixed tool sequence.
- Do not encode attack-type-to-tool mappings.
- Do not output evidence_hints, packet views, or tool names.
- Do not output concrete tx hashes, addresses, call IDs, transfer IDs, event IDs, or storage slots.
- Add, remove, or rewrite evidence conditions only when supported by reviews.
- Prefer minimal scoped edits. If reviews identify one specific condition_id,
  rewrite only that condition and preserve unrelated conditions/exclusions.
- When changing one condition, keep it consistent with unchanged conditions,
  exclusions, and decision_policy. Do not introduce a dependency that requires
  silently rewriting an unchanged condition.
- Exclusions must remain target-positive boundary clauses, not negative-class
  profiles. Add or strengthen an exclusion only by stating which required
  target-positive mechanism is absent, fully authorized, fully entitlement-
  backed, or completely replaced by another primary root cause.
- Do not add an exclusion merely because a non-target family is present. The
  non-target evidence must defeat or replace the target-family mechanism
  required by the positive conditions.
- Add exclusion or boundary conditions for evidence-backed non-target negatives
  only through that target-positive boundary framing.
- Non-target negatives may be truly benign transactions or other attack
  families. Do not describe other attack families as benign behavior.
- For source-dependent targets, do not make verified source availability a hard
  semantic requirement when packet evidence can provide target-specific
  structural signals. However, do not broaden the rule so that source_unavailable
  plus generic exploit evidence becomes sufficient.
- For access-control rules, behavioral fallback must require access-control-
  specific evidence such as owner/admin/role/permission/initializer/proxy/
  implementation/delegatecall/authorization-slot signals, protected value
  release from a protocol-controlled or victim target, caller-to-beneficiary
  linkage, or an unknown/obfuscated selector directly changing protected
  configuration/state or releasing protected value.
- For access-control rules, value extraction, flash loans, swaps, callbacks,
  reentrancy, unusual profit, public entry functions, or generic economic
  anomalies are not sufficient by themselves. Preserve target-vs-non-target
  boundaries with exclusions or condition wording.
- For access-control rules, preserve one shared authorization chain across the
  core conditions: actor/caller or beneficiary -> required authority ->
  sensitive capability and protected target/resource -> protected effect.
  Never repair one condition by allowing it to use an unrelated actor, call,
  capability, asset, position, or protocol component. Multiple calls are valid
  only when the rule requires a trace-linked invocation/delegation/callback
  chain crossing the same authorization boundary.
- For access-control C1 semantics, preserve both the externally reachable
  entry-side parent and the trace-linked downstream sensitive-operation child
  when they differ. Do not narrow the anchor to a guarded child and thereby
  erase an exposed dispatcher/module/proxy/callback parent boundary. C1 is a
  candidate locator only; whether authority is required, missing, or
  legitimately permissionless belongs to C2/E1.
- For access-control C2 semantics, require an authorization-source and
  provenance audit in addition to guard presence: owner/admin, role manager,
  module, whitelist, signature/delegation, callback sender, and whether the
  authority was pre-existing legitimate, self-granted, attacker-controlled, or
  created in the same causal chain. A modifier alone is not a complete safety
  boundary when its authority source can be selected or mutated by the actor.
- For insufficient-validation rules, preserve the full C1 candidate family:
  user-controlled inputs, callback data, token/path/address and amount data,
  signatures/domains, external returns, oracle values, computed intermediates,
  and protocol-internal business/accounting state. Do not repair one FP by
  narrowing C1 to protocol-internal state only or by deleting an externally
  derived target variant used by correct positives.
- For insufficient-validation C2 semantics, require a concrete semantic
  content/source/freshness/range/invariant validation gap on the same selected
  object. Generic caller/owner/role/whitelist/callback-sender authorization is
  the E1 access-control boundary unless authority data or signer/domain content
  is itself the consumed object. Source_unavailable and no visible trace check
  do not prove that a passing branch, modifier, or require check was absent.
  Keep C2 local to the validation dimension and gap; E1/E2 own decisions that
  Access Control, Reentrancy, market, token, or accounting semantics replace
  the insufficient-validation root cause.
- For insufficient-validation C3 semantics, require a concrete same-object
  causal outcome; downstream swaps, profit, and value release alone are not
  sufficient. An independent primary mechanism may satisfy E2 even when the
  selected object appears as a conduit, if its semantic invalidity is not
  necessary to explain the outcome.
- For reentrancy rules, keep C1 as the nested-path locator, C2 as the
  candidate-local value/state effect, and C3 as stale/intermediate-state,
  delayed-finalization, repeated-consumption, and phase-order causality. Do not
  duplicate the C3 state-order proof inside C2.
- For market-manipulation rules, preserve one shared market-mechanism profile
  across the core conditions: market source/profile -> same-profile consumer
  operation -> profile-specific abnormal outcome. Do not repair recall by
  allowing C1/C2/C3 to mix unrelated AMM price movement, token mechanics,
  flash capital, generic swaps, or generic profit.
- For market-manipulation rules, generic profit, swap volume, ordinary
  arbitrage, flashloan presence, value release, or final asset delta is not
  sufficient unless tied to the same selected market source/profile. If another
  mechanism such as token semantics, protocol accounting, access control,
  insufficient validation, or reentrancy fully replaces the root cause, frame
  the boundary as absence/replacement of the target market profile.
- For protocol-accounting rules, preserve one shared protocol-internal
  bookkeeping candidate across core conditions: candidate state -> same-candidate
  protocol consumption/misuse -> candidate-specific accounting outcome. Direct
  token/native transfer is helpful but not mandatory when state-level reward,
  share, vault, debt, collateral, staking, claim, or liability impact is
  evidence-backed. Generic profit, normal deposit/withdraw/harvest mechanics,
  reentrancy/state-order abuse, insufficient validation, token semantics,
  market/oracle effects, flash capital, or access-control failure cannot replace
  the PAE candidate chain.
- Keep each condition local enough for a yes/no judge.
- State how conditions and exclusions should be combined.
- The current task may target a specific attack family. Use rule.metadata.attack_label
  and any review.raw_ground_truth fields to keep updates aligned to that attack
  family, instead of broadening into a generic anomaly rule.
- review.training_supervision.label_rationale, when present, is training-only
  CSV rationale. It may identify the intended abstract mechanism, but it is not
  transaction evidence and cannot by itself justify a rule condition or patch.
- Use feature_diagnosis when present. It summarizes recurring semantic
  features extracted from local judge decisions and is more specific than
  generic broaden/tighten language.
- For FN, consider broadening only when feature_diagnosis.update_implication ==
  "broaden" and observed_partial_match_features are target-family-specific, not
  generic exploit symptoms. Convert recurring positive-target partial features
  into abstract rule language.
- For FP, tighten when update_implication == "tighten" and matched_features are
  weak/generic. Add or strengthen exclusions when update_implication ==
  "add_exclusion" and boundary_notes identify how the target-positive mechanism
  is absent, authorized, entitlement-backed, or replaced by another primary
  root cause. Convert observed_missing_required_features into necessary
  evidence requirements where appropriate.
- For clarify_boundary, add wording that states a feature is necessary,
  insufficient, or excluded.
- For no_change, preserve that condition unless another review explicitly
  requires changing it.
- If feature_diagnosis conflicts with root_causes, trust root_causes for
  routing but mention the conflict in update_note.
- Do not copy concrete selectors, addresses, call ids, evidence ids, token
  amounts, storage slots, tx hashes, or chain-specific identifiers into the
  rule. Convert feature strings into abstract semantic phrases.
- Do not overfit to a single sample.
- Return strict JSON only.

Current rule:
{stable_json_dumps(rule.to_dict())}

Root-cause analyses:
{stable_json_dumps(reviews)}

Feature diagnosis summary:
{stable_json_dumps(feature_diagnosis_summary)}

Return JSON:
{{
  "description": "updated overall rule description",
  "conditions": [
    {{
      "id": "C1",
      "description": "...",
      "expected_answer": true
    }}
  ],
  "exclusion_conditions": [
    {{
      "id": "E1",
      "description": "...",
      "expected_answer": true
    }}
  ],
  "decision_policy": "how to combine conditions and exclusions",
  "update_note": "what changed and why; summarize accepted/rejected conflicts using only supplied review counts"
}}
"""

    @staticmethod
    def _build_bundle_update_prompt(
        rule: EvolvingRule,
        review_bundle: Dict[str, Any],
        actionable_reviews: List[Dict[str, Any]],
    ) -> str:
        case_boundary_context = dict(
            review_bundle.get("case_boundary_context") or {}
        )
        active_refinement_feedback = dict(
            review_bundle.get("rejected_update_memory") or {}
        ).get("active_refinement_feedback") or {}
        negative_training_mode = normalize_negative_training_mode(
            review_bundle.get("negative_training_mode", "mixed")
        )
        negative_training_guidance = negative_training_mode_guidance(
            negative_training_mode
        )
        negative_case_guidance = negative_case_boundary_guidance(
            negative_training_mode
        )
        structural_flags = dict(
            review_bundle.get("rule_structural_feature_flags") or {}
        )
        structural_evolution_enabled = bool(
            structural_flags.get("enable_structural_evolution", True)
        )
        remove_positive_enabled = bool(
            structural_flags.get("enable_remove_positive_condition", False)
        )
        remove_exclusion_enabled = bool(
            structural_flags.get("enable_remove_exclusion", False)
        )
        experimental_probe = bool(
            review_bundle.get("phase3_experimental_rule_probe")
        )
        signal_registry = (
            "experimental_rule_signal_ids"
            if experimental_probe
            else "supported_signal_ids"
        )
        lifecycle_instruction = (
            "These signals remain experimental and this output is probe-only."
            if experimental_probe
            else "These signals are supported normal-evolution inputs."
        )
        return f"""
You are updating only the semantic EvoTx rule.

{UPDATER_CANONICAL_SIGNAL_GUIDANCE}

{_rule_updater_label_guidance(rule)}

You are not fixing runtime bugs, packet builder bugs, source availability
issues, or plan view selection. Materialize only Rule-owner signal_ids present
in the canonical update_signal_matrix supplied below.

Do not update the rule based on runtime_emit_logic_bug, packet truncation,
missing packet views, missing source evidence, or tool-call failures unless the
canonical Rule signal explicitly owns a semantic condition error.

Requirements:
- Update the rule, not model parameters.
- Do not hardcode a fixed tool sequence.
- Do not encode attack-type-to-tool mappings.
- Do not output evidence_hints, packet views, or tool names.
- Do not output concrete tx hashes, addresses, call IDs, transfer IDs, event IDs, or storage slots.
- Add, remove, or rewrite evidence conditions only when supported by canonical
  Rule signals.
- Prefer minimal scoped edits. If the canonical signals identify one specific
  condition_id, rewrite only that condition and preserve all unrelated
  conditions and exclusions.
- When changing one condition, keep it logically consistent with the unchanged
  conditions, exclusions, and decision_policy. Do not introduce a new dependency
  that contradicts or requires rewriting an unchanged condition.
- Exclusions must remain target-positive boundary clauses, not negative-class
  profiles. Add or strengthen an exclusion only by stating which required
  target-positive mechanism is absent, fully authorized, fully entitlement-
  backed, or completely replaced by another primary root cause.
- Do not add an exclusion merely because a non-target family is present. The
  non-target evidence must defeat or replace the target-family mechanism
  required by the positive conditions.
- Add exclusion or boundary conditions only for evidence-backed non-target
  negatives through that target-positive boundary framing.
- If a canonical signal comes from negative_other / other_attack, tighten the target
  attack-family boundary rather than calling that sample benign.
- The round-level negative_training_mode is dataset-coverage context, not
  evidence about any individual transaction. Apply its guidance without
  encoding dataset labels into the semantic rule.
- contrastive_supervision and contrastive_label_coverage are also training-only
  metadata. Use them to understand which target-vs-non-target boundary needs
  evidence, but never copy a source-family label into a condition, exclusion,
  decision policy, update signal, or rule description.
- negative_training_mode="{negative_training_mode}": {negative_training_guidance}
- {negative_case_guidance}
- For source-dependent targets, do not make verified source availability a hard
  semantic requirement when packet evidence can provide target-specific
  structural signals. But never make source_unavailable plus generic exploit
  evidence sufficient.
- For access-control rules, behavioral fallback must require access-control-
  specific evidence such as owner/admin/role/permission/initializer/proxy/
  implementation/delegatecall/authorization-slot signals, protected value
  release from a protocol-controlled or victim target, caller-to-beneficiary
  linkage, or an unknown/obfuscated selector directly changing protected
  configuration/state or releasing protected value.
- For access-control rules, value extraction, flash loans, swaps, callbacks,
  reentrancy, unusual profit, public entry functions, or generic economic
  anomalies are not sufficient by themselves. Preserve target-vs-non-target
  boundaries with exclusions or condition wording.
- For insufficient-validation rules, preserve user/external/computed and
  protocol-internal C1 object variants. Never tighten a single false positive
  by making protocol-internal accounting state the only eligible object.
- For insufficient-validation C2, separate semantic validation from generic
  identity authorization: caller/owner/role/whitelist/callback-sender failures
  belong to E1 unless supplied authority data or signer/domain content is the
  object being validated. Source_unavailable plus no visible check is not proof
  that validation was absent. Keep replacement-root-cause decisions in E1/E2
  instead of expanding C2 into another attack family's complete boundary.
- For insufficient-validation C3/E2, require a same-object causal outcome and
  allow a sufficient independent mechanism to exclude the label even when the
  object remains a non-causal conduit.
- For access-control rules, preserve one shared authorization chain across the
  core conditions: actor/caller or beneficiary -> required authority ->
  sensitive capability and protected target/resource -> protected effect.
  Never repair one condition by allowing it to use an unrelated actor, call,
  capability, asset, position, or protocol component. Multiple calls are valid
  only when the rule requires a trace-linked invocation/delegation/callback
  chain crossing the same authorization boundary.
- For access-control rules, C1 only locates that concrete chain candidate; C2
  decides missing authorization and E1 decides legitimate authority or
  entitlement. C1 need not decide actual authorization, but it must identify a
  plausibly authority-bearing capability/resource or authorization-relevant
  state/effect anchor; generic public swap/transfer routing, beneficiary gain,
  or value asymmetry alone is insufficient. Do not require C1 to pre-classify
  the candidate as authorized or permissionless.
- When a supported access-control Rule signal identifies a positively supported
  replacement root cause with an intact or absent authorization boundary,
  update the existing E2 boundary instead of folding the replacement mechanism
  into C2. A coherent local Rule update may include both C2 and E2 only when the
  canonical matrix contains supported signals for both conditions.
- For reentrancy rules, C1 locates the nested path, C2 establishes its local
  value/state effect, and C3 establishes stale/order/repeated-consumption
  causality. Do not make C2 repeat C3.
- For protocol-accounting rules, preserve one shared protocol-internal
  bookkeeping candidate across the core conditions: candidate state ->
  same-candidate protocol consumption/misuse -> candidate-specific accounting
  outcome. Direct token/native transfer is helpful but not mandatory when
  state-level reward/share/vault/debt/collateral/staking/claim/liability impact
  is evidence-backed. Do not repair precision by naming a non-target family
  alone; write the boundary as absence, normal/entitlement-backed behavior, or
  replacement of the PAE candidate chain by another primary root cause.
- Keep each condition local enough for a yes/no judge.
- Keep updates aligned to rule.metadata.attack_label and raw_ground_truth.
- Do not broaden the rule into generic transaction anomaly detection.
- Do not copy concrete selectors, addresses, call ids, evidence ids, token
  amounts, storage slots, tx hashes, or chain-specific identifiers into the
  rule. Convert features into abstract semantic phrases.
- Do not overfit to a single sample.
- When active_refinement_feedback is present, preserve its observed improvement
  and repair only its attributed component-level regression. It is refinement
  context, not a second source of update eligibility.
- Return strict JSON only.
- Treat each canonical signal as a bounded hypothesis. Apply only signal_ids listed in
  update_signal_matrix.{signal_registry} and declare them in
  applied_signal_ids. If none survive correct-positive and correct-negative/
  guard checks, return the current rule unchanged.
- {lifecycle_instruction}
- Semantically merge compatible supported signals before writing rule text, but
  retain every contributing signal_id in applied_signal_ids. Do not turn a
  semantic cluster into a rule unless protected positive and boundary signals
  remain compatible.
- update_signal_matrix.joint_resolution_groups are atomic synthesis contracts.
  If applied_signal_ids intersects one group, include every signal_id in that
  group across rule_operations and preserve all listed semantic axes. Applying
  only the broadening or only the tightening half is invalid.
- Return typed rule_operations. Each operation must cite its own supported
  applied_signal_ids and an explicit semantic_direction.
- Structural evolution is enabled={str(structural_evolution_enabled).lower()}.
- rewrite_condition and narrow_exclusion keep the existing interface and are
  always available. When structural evolution is enabled, add_positive_condition
  and add_exclusion are also available by default.
- remove_positive_condition is enabled={str(remove_positive_enabled).lower()}.
- remove_exclusion is enabled={str(remove_exclusion_enabled).lower()}.
- add_positive_condition always means tighten; remove_positive_condition means
  broaden; add_exclusion means tighten; remove_exclusion means broaden.
- rewrite_condition must explicitly use tighten, broaden, or refine.
  narrow_exclusion is classifier-level broaden because fewer target positives
  are rejected by that exclusion.
- FP repair should use tighten/refine operations. FN repair should use
  broaden/refine operations. A mismatched direction is invalid unless the
  operation cites a complete update_signal_matrix.joint_resolution_group.
- Keep existing ids and expected_answer values stable. New positive conditions
  receive a C id and new exclusions receive an E id deterministically after
  parsing; do not invent ids for add operations.
- Do not express alternate sufficient attack paths by weakening a condition.
  Current execution is still all positives AND NOT(any exclusion). When a fix
  needs an OR branch, emit no operation and mention
  deferred_logic_structure_failure in update_note.
- Preserve the current top-level description and decision_policy. The runtime
  will deterministically rebuild the strict AND + NOT(exclusion) policy after
  an accepted structural operation.
- Correct target positives are preservation constraints. A tightening must state
  the abstract distinction that keeps their supported mechanism variants valid.
- Case boundary context is the preservation contract from ordinary/correct
  cases. Apply a signal only if the resulting rule keeps correct target-positive
  mechanisms detectable and keeps correct negatives/guard cases outside the
  target family. If this cannot be done, leave the rule unchanged.
- Do not let an error-centric repair override case_boundary_context. The error
  explains what failed; the boundary context says what must not break.
- active_refinement_feedback is not transaction evidence and not an eligibility
  gate. Use only its failure reason and protected attribution to refine the
  current canonical signal. Exact duplicates are handled downstream.
- Never encode contract/protocol names, unusual function names, CTF/challenge
  assumptions, playful or unknown selectors, or presumed contract purpose.

Current rule:
{stable_json_dumps(rule.to_dict())}

Negative training context:
{stable_json_dumps({
    "mode": negative_training_mode,
    "source": review_bundle.get("negative_training_mode_source", ""),
})}

Preservation boundary context:
{stable_json_dumps(case_boundary_context)}

Active refinement feedback:
{stable_json_dumps(active_refinement_feedback)}

Canonical update signals:
{stable_json_dumps(review_bundle.get("update_signal_matrix", {}))}

Return JSON:
{{
  "applied_signal_ids": ["signal_001"],
  "rule_operations": [
    {{
      "operation": "rewrite_condition|add_positive_condition|add_exclusion|narrow_exclusion|remove_positive_condition|remove_exclusion",
      "condition_id": "C1 or E1; empty only for add operations",
      "description": "...",
      "semantic_direction": "tighten|broaden|refine",
      "applied_signal_ids": ["signal_001"],
      "covered_blocker_ids": ["C1"],
      "independently_verdict_capable": false,
      "joint_resolution_signal_ids": [],
      "rationale": "abstract reusable reason"
    }}
  ],
  "update_note": "what changed and why; cite only applied canonical signal ids and preservation/refinement constraints"
}}
"""

    @staticmethod
    def _rule_from_update_payload(
        rule: EvolvingRule,
        data: Dict[str, Any],
        note_prefix: str = "update",
    ) -> EvolvingRule:
        new_conditions = [
            RuleCondition.from_dict(item) for item in data.get("conditions", [])
        ]
        new_exclusions = [
            RuleCondition.from_dict(item)
            for item in data.get("exclusion_conditions", [])
        ]
        update_note = str(data.get("update_note", note_prefix))
        return rule.next_version(
            new_description=str(data.get("description", rule.description)),
            new_conditions=new_conditions or rule.conditions,
            new_exclusions=new_exclusions
            if "exclusion_conditions" in data
            else rule.exclusion_conditions,
            decision_policy=str(data.get("decision_policy", rule.decision_policy)),
            update_note=update_note,
        )

    @staticmethod
    def _fallback_update(
        rule: EvolvingRule, reviews: List[Dict[str, Any]]
    ) -> EvolvingRule:
        notes = []
        for review in reviews:
            for cause in review.get("root_causes", []):
                fix = cause.get("suggested_fix")
                if fix:
                    notes.append(fix)
        if not notes:
            return rule

        metadata = {
            "pending_manual_fixes": notes,
            "fallback_update": True,
            "rule_complexity": rule_complexity(rule),
        }
        return rule.next_version(
            new_description=rule.description,
            new_conditions=rule.conditions,
            new_exclusions=rule.exclusion_conditions,
            update_note="Stored pending manual fixes; no LLM updater configured.",
            metadata=metadata,
        )


def _feature_diagnosis_summary(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for review in list(reviews or []):
        feature_diagnosis = list(review.get("feature_diagnosis", []) or [])
        if not feature_diagnosis:
            continue
        summary.append({
            "tx_hash": review.get("tx_hash"),
            "error_type": review.get("error_type"),
            "target_label": review.get("target_label"),
            "raw_ground_truth": review.get("raw_ground_truth"),
            "feature_diagnosis": feature_diagnosis,
        })
    return summary


def _compact_review_bundle_for_update(
    review_bundle: Dict[str, Any],
    actionable_reviews: List[Dict[str, Any]],
    *,
    update_target: str,
) -> Dict[str, Any]:
    """Build the smaller bundle actually sent to the updater LLM.

    The original bundle remains the authority for gates. This prompt bundle keeps
    the supported update signals and preservation context while avoiding replay
    of every review/audit field.
    """
    bundle = dict(review_bundle or {})
    target = str(update_target or "").strip().lower()
    matrix = dict(bundle.get("update_signal_matrix") or {})
    actionable_ids = _actionable_update_signal_ids(bundle, target)
    supported_ids = {
        str(signal.get("signal_id") or "")
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict)
        and str(signal.get("update_target") or "").strip().lower() == target
        and str(signal.get("signal_id") or "") in actionable_ids
        and str(signal.get("signal_id") or "")
    }
    relevant_conditions = _bundle_relevant_condition_ids(
        matrix,
        update_target=target,
        actionable_signal_ids=supported_ids,
    )
    compact = {
        "prompt_compaction": {
            "schema_version": "evotx.update_prompt_bundle.v1",
            "source_review_bundle_chars": len(stable_json_dumps(bundle)),
            "update_target": target,
            "relevant_condition_ids": sorted(relevant_conditions),
            "actionable_signal_ids": sorted(supported_ids),
            "policy": (
                "Prompt receives only canonical actionable signals, the relevant "
                "preservation boundary, active refinement feedback, and any "
                "owner-required execution audit."
            ),
        },
        "negative_training_mode": bundle.get("negative_training_mode", "mixed"),
        "negative_training_mode_source": bundle.get(
            "negative_training_mode_source",
            "default",
        ),
        "update_signal_matrix": _compact_update_signal_matrix_for_prompt(
            matrix,
            relevant_conditions,
            update_target=target,
            actionable_signal_ids=supported_ids,
        ),
        "canonical_signal_contract": dict(
            bundle.get("canonical_signal_contract") or {}
        ),
    }
    if supported_ids:
        compact["case_boundary_context"] = _compact_case_boundary_for_prompt(
            bundle.get("case_boundary_context", {}),
            relevant_conditions,
        )
        if target == "plan":
            compact["plan_evidence_audit"] = _compact_plan_audit_for_prompt(
                bundle.get("plan_evidence_audit", {}),
                relevant_conditions,
            )
    refinement_memory = _compact_rejected_memory_for_prompt(
        bundle.get("rejected_update_memory", {})
    )
    if refinement_memory:
        compact["rejected_update_memory"] = refinement_memory
    experimental_rule_probe = dict(
        bundle.get("phase3_experimental_rule_probe") or {}
    )
    if experimental_rule_probe:
        compact["phase3_experimental_rule_probe"] = experimental_rule_probe
    compact["prompt_compaction"]["section_chars"] = {
        key: len(stable_json_dumps(value))
        for key, value in compact.items()
        if key != "prompt_compaction"
    }
    compact["prompt_compaction"]["prompt_bundle_chars"] = len(
        stable_json_dumps(compact)
    )
    return compact


def _bundle_relevant_condition_ids(
    matrix: Dict[str, Any],
    *,
    update_target: str,
    actionable_signal_ids: set[str],
) -> set[str]:
    relevant: set[str] = set()
    target = str(update_target or "").strip().lower()
    for signal in list((matrix or {}).get("signals") or []):
        if not isinstance(signal, dict):
            continue
        if str(signal.get("update_target") or "").strip().lower() != target:
            continue
        if str(signal.get("signal_id") or "") not in actionable_signal_ids:
            continue
        status = str(signal.get("generalization_status") or "").strip().lower()
        if status != "supported" and not (
            target == "plan" and status == "experimental_plan"
        ):
            continue
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        if condition_id:
            relevant.add(condition_id)
    return relevant


def _compact_review_for_prompt(
    review: Dict[str, Any],
    relevant_conditions: set[str],
    supported_ids: set[str],
    *,
    update_target: str,
) -> Dict[str, Any]:
    target = str(update_target or "").strip().lower()
    signals = []
    for signal in list((review or {}).get("update_signals") or []):
        if not isinstance(signal, dict):
            continue
        signal_id = str(signal.get("signal_id") or "")
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        signal_target = str(
            signal.get("update_target") or review.get("update_target") or ""
        ).strip().lower()
        if signal_target != target:
            continue
        if supported_ids and signal_id and signal_id not in supported_ids:
            continue
        if relevant_conditions and condition_id and condition_id not in relevant_conditions:
            continue
        signals.append(_compact_json_for_prompt(signal, depth=3, max_list=8))
    return {
        "tx_hash": review.get("tx_hash"),
        "error_type": review.get("error_type"),
        "ground_truth": review.get("ground_truth"),
        "predicted_verdict": review.get("predicted_verdict"),
        "raw_ground_truth": review.get("raw_ground_truth"),
        "negative_kind": review.get("negative_kind"),
        "target_label": review.get("target_label"),
        "target_mechanism_observed": review.get("target_mechanism_observed"),
        "ground_truth_supported_by_packet": review.get(
            "ground_truth_supported_by_packet"
        ),
        "generic_symptoms_only": bool(review.get("generic_symptoms_only", False)),
        "counterfactual_boundary_risk": review.get("counterfactual_boundary_risk"),
        "update_target": review.get("update_target"),
        "root_causes": _compact_json_for_prompt(
            review.get("root_causes", []),
            depth=3,
            max_list=5,
        ),
        "condition_diagnosis": [
            _compact_json_for_prompt(item, depth=3, max_list=6)
            for item in list(review.get("condition_diagnosis") or [])
            if isinstance(item, dict)
            and (
                not relevant_conditions
                or str(item.get("condition_id") or "").strip().upper()
                in relevant_conditions
            )
        ][:8],
        "feature_diagnosis": _compact_json_for_prompt(
            review.get("feature_diagnosis", []),
            depth=3,
            max_list=8,
        ),
        "update_signals": signals[:12],
        "rule_patch_suggestion": _compact_json_for_prompt(
            review.get("rule_patch_suggestion", {}),
            depth=3,
            max_list=6,
        ),
        "plan_patch_suggestion": _compact_json_for_prompt(
            review.get("plan_patch_suggestion", {}),
            depth=3,
            max_list=6,
        ),
        "training_gate": _compact_json_for_prompt(
            review.get("training_gate", {}),
            depth=3,
            max_list=6,
        ),
        "must_not_change": list(review.get("must_not_change") or [])[:8],
        "review_note": _truncate_prompt_text(review.get("review_note"), 700),
    }


def _compact_update_signal_matrix_for_prompt(
    matrix: Dict[str, Any],
    relevant_conditions: set[str],
    *,
    update_target: str,
    actionable_signal_ids: set[str],
) -> Dict[str, Any]:
    target = str(update_target or "").strip().lower()
    signals = []
    supported_ids: List[str] = []
    experimental_plan_ids: List[str] = []
    experimental_rule_ids: List[str] = []
    for signal in list((matrix or {}).get("signals") or []):
        if not isinstance(signal, dict):
            continue
        signal_target = str(signal.get("update_target") or "").strip().lower()
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        status = str(signal.get("generalization_status") or "").strip().lower()
        signal_id = str(signal.get("signal_id") or "")
        if signal_target != target:
            continue
        if signal_id not in actionable_signal_ids:
            continue
        if relevant_conditions and condition_id and condition_id not in relevant_conditions:
            continue
        compact = _compact_json_for_prompt(signal, depth=3, max_list=8)
        signals.append(compact)
        if status == "supported":
            supported_ids.append(signal_id)
        elif status == "experimental_plan":
            experimental_plan_ids.append(signal_id)
        elif status == "experimental_rule":
            experimental_rule_ids.append(signal_id)
    return {
        "schema_version": (matrix or {}).get("schema_version", ""),
        "signals": signals[:24],
        "supported_signal_ids": supported_ids[:24],
        "experimental_plan_signal_ids": experimental_plan_ids[:24],
        "experimental_rule_signal_ids": experimental_rule_ids[:24],
        "plan_actionable_signal_ids": [
            *supported_ids,
            *experimental_plan_ids,
        ][:24],
        "joint_resolution_signal_ids": [
            signal_id
            for signal_id in list((matrix or {}).get("joint_resolution_signal_ids") or [])
            if signal_id in supported_ids
        ][:24],
        "joint_resolution_groups": [
            _compact_json_for_prompt(group, depth=3, max_list=12)
            for group in list((matrix or {}).get("joint_resolution_groups") or [])
            if isinstance(group, dict)
            and str(group.get("update_target") or "").strip().lower() == target
            and (
                not relevant_conditions
                or str(group.get("condition_id") or "").strip().upper()
                in relevant_conditions
            )
        ][:12],
    }


def _compact_current_summary(summary: Any) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    keep = {
        "total",
        "correct",
        "incorrect",
        "tp",
        "tn",
        "fp",
        "fn",
        "uncertain",
        "attack_recall",
        "attack_precision",
        "precision",
        "benign_specificity",
        "negative_specificity",
        "errors",
    }
    out = {key: summary.get(key) for key in keep if key in summary}
    for key in ("by_label", "by_ground_truth", "verdict_counts"):
        if key in summary:
            out[key] = _compact_json_for_prompt(summary.get(key), depth=2, max_list=8)
    return out


def _compact_cohort_summary_for_prompt(
    summary: Any,
    relevant_conditions: set[str],
) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    conditions = {}
    for condition_id, value in dict(summary.get("conditions") or {}).items():
        cid = str(condition_id or "").strip().upper()
        if relevant_conditions and cid not in relevant_conditions:
            continue
        conditions[condition_id] = _compact_json_for_prompt(
            value,
            depth=5,
            max_list=5,
            max_string=420,
        )
    return {
        "schema_version": summary.get("schema_version", ""),
        "counts": dict(summary.get("counts") or {}),
        "conditions": conditions,
        "protected_positive_signals": [
            _compact_json_for_prompt(item, depth=3, max_list=5)
            for item in list(summary.get("protected_positive_signals") or [])
            if not relevant_conditions
            or str(item.get("condition_id") or "").strip().upper()
            in relevant_conditions
        ][:8],
        "protected_boundary_signals": [
            _compact_json_for_prompt(item, depth=3, max_list=5)
            for item in list(summary.get("protected_boundary_signals") or [])
            if not relevant_conditions
            or str(item.get("condition_id") or "").strip().upper()
            in relevant_conditions
        ][:8],
        "contrastive_label_coverage": _compact_json_for_prompt(
            summary.get("contrastive_label_coverage", {}),
            depth=2,
            max_list=8,
        ),
    }


def _compact_plan_audit_for_prompt(
    audit: Any,
    relevant_conditions: set[str],
) -> Dict[str, Any]:
    if not isinstance(audit, dict):
        return {}
    conditions = {}
    for condition_id, value in dict(audit.get("conditions") or {}).items():
        cid = str(condition_id or "").strip().upper()
        if relevant_conditions and cid not in relevant_conditions:
            continue
        groups = {}
        for group_name, group in dict((value or {}).get("groups") or {}).items():
            if not isinstance(group, dict):
                continue
            groups[group_name] = {
                "case_count": group.get("case_count", 0),
                "initial_answer_counts": group.get("initial_answer_counts", {}),
                "final_answer_counts": group.get("final_answer_counts", {}),
                "answer_transitions": group.get("answer_transitions", {}),
                "followup_changed_answer_count": group.get(
                    "followup_changed_answer_count",
                    0,
                ),
                "near_miss_escalated_count": group.get(
                    "near_miss_escalated_count",
                    0,
                ),
                "prompt_truncated_view_counts": _top_counter_like(
                    group.get("prompt_truncated_view_counts", {})
                ),
                "tool_status_counts": _top_counter_like(
                    group.get("tool_status_counts", {})
                ),
                "followup_view_attempt_counts": _top_counter_like(
                    group.get("followup_view_attempt_counts", {})
                ),
                "successful_followup_view_counts": _top_counter_like(
                    group.get("successful_followup_view_counts", {})
                ),
                "answer_changing_followup_view_counts": _top_counter_like(
                    group.get("answer_changing_followup_view_counts", {})
                ),
                "missing_status_counts": _top_counter_like(
                    group.get("missing_status_counts", {})
                ),
                "execution_paths": _compact_json_for_prompt(
                    group.get("execution_paths", []),
                    depth=4,
                    max_list=3,
                    max_string=240,
                ),
            }
        conditions[condition_id] = {
            "is_exclusion": bool((value or {}).get("is_exclusion", False)),
            "groups": groups,
            "route_failure_memory": _compact_json_for_prompt(
                (value or {}).get("route_failure_memory", []),
                depth=3,
                max_list=12,
                max_string=180,
            ),
        }
    return {
        "schema_version": audit.get("schema_version", ""),
        "case_groups": dict(audit.get("case_groups") or {}),
        "conditions": conditions,
    }


def _compact_case_boundary_for_prompt(
    context: Any,
    relevant_conditions: set[str],
) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    boundaries = {}
    for condition_id, value in dict(context.get("condition_boundaries") or {}).items():
        cid = str(condition_id or "").strip().upper()
        if relevant_conditions and cid not in relevant_conditions:
            continue
        boundaries[condition_id] = _compact_json_for_prompt(
            value,
            depth=5,
            max_list=5,
            max_string=360,
        )
    return {
        "schema_version": context.get("schema_version", ""),
        "source": context.get("source", ""),
        "counts": dict(context.get("counts") or {}),
        "contrastive_label_coverage": _compact_json_for_prompt(
            context.get("contrastive_label_coverage", {}),
            depth=2,
            max_list=8,
        ),
        "protected_positive_signals": [
            _compact_json_for_prompt(item, depth=3, max_list=5)
            for item in list(context.get("protected_positive_signals") or [])
            if not relevant_conditions
            or str(item.get("condition_id") or "").strip().upper()
            in relevant_conditions
        ][:8],
        "protected_boundary_signals": [
            _compact_json_for_prompt(item, depth=3, max_list=5)
            for item in list(context.get("protected_boundary_signals") or [])
            if not relevant_conditions
            or str(item.get("condition_id") or "").strip().upper()
            in relevant_conditions
        ][:8],
        "condition_boundaries": boundaries,
        "preservation_policy": list(context.get("preservation_policy") or [])[:5],
        "contains_transaction_identifiers": bool(
            context.get("contains_transaction_identifiers", False)
        ),
    }


def _compact_rejected_memory_for_prompt(
    memory: Any,
) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    active_refinement_feedback = memory.get("active_refinement_feedback")
    if not isinstance(active_refinement_feedback, dict) or not active_refinement_feedback:
        return {}
    return {
        "active_refinement_feedback": _compact_json_for_prompt(
            active_refinement_feedback,
            depth=5,
            max_list=8,
        )
    }


def _compact_skipped_reviews(reviews: Any, *, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for review in list(reviews or []):
        if not isinstance(review, dict):
            continue
        out.append({
            "tx_hash": review.get("tx_hash"),
            "error_type": review.get("error_type"),
            "update_target": review.get("update_target"),
            "root_causes": _compact_json_for_prompt(
                review.get("root_causes", []),
                depth=2,
                max_list=3,
            ),
            "training_gate": _compact_json_for_prompt(
                review.get("training_gate", {}),
                depth=2,
                max_list=4,
            ),
            "review_note": _truncate_prompt_text(review.get("review_note"), 360),
        })
        if len(out) >= limit:
            break
    return out


def _top_counter_like(value: Any, *, limit: int = 8) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    items = sorted(
        value.items(),
        key=lambda item: int(item[1] or 0) if str(item[1]).lstrip("-").isdigit() else 0,
        reverse=True,
    )
    return {str(key): count for key, count in items[:limit]}


def _compact_json_for_prompt(
    value: Any,
    *,
    depth: int = 4,
    max_list: int = 8,
    max_dict: int = 24,
    max_string: int = 500,
) -> Any:
    if depth <= 0:
        return _truncate_prompt_text(value, max_string)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_dict:
                out["_truncated_keys"] = max(0, len(value) - max_dict)
                break
            out[str(key)] = _compact_json_for_prompt(
                item,
                depth=depth - 1,
                max_list=max_list,
                max_dict=max_dict,
                max_string=max_string,
            )
        return out
    if isinstance(value, list):
        out = [
            _compact_json_for_prompt(
                item,
                depth=depth - 1,
                max_list=max_list,
                max_dict=max_dict,
                max_string=max_string,
            )
            for item in value[:max_list]
        ]
        if len(value) > max_list:
            out.append({"_truncated_items": len(value) - max_list})
        return out
    return _truncate_prompt_text(value, max_string)


def _truncate_prompt_text(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 32)] + "...[truncated for updater]"


def _rule_update_audit_summary(
    previous: EvolvingRule,
    updated: EvolvingRule | None,
) -> Dict[str, Any]:
    if updated is None:
        return {
            "updated_rule_present": False,
            "updated_rule_id": "",
            "updated_rule_version": None,
            "condition_count": 0,
            "exclusion_count": 0,
            "changed_condition_ids": [],
            "added_condition_ids": [],
            "removed_condition_ids": [],
        }
    previous_by_id = {
        condition.id: (condition.description, bool(condition.expected_answer))
        for condition in list(previous.conditions or [])
        + list(previous.exclusion_conditions or [])
    }
    updated_by_id = {
        condition.id: (condition.description, bool(condition.expected_answer))
        for condition in list(updated.conditions or [])
        + list(updated.exclusion_conditions or [])
    }
    changed = sorted(
        condition_id
        for condition_id in set(previous_by_id) | set(updated_by_id)
        if previous_by_id.get(condition_id) != updated_by_id.get(condition_id)
    )
    return {
        "updated_rule_present": True,
        "updated_rule_id": updated.rule_id,
        "updated_rule_version": updated.version,
        "condition_count": len(updated.conditions),
        "exclusion_count": len(updated.exclusion_conditions),
        "changed_condition_ids": changed,
        "added_condition_ids": sorted(set(updated_by_id) - set(previous_by_id)),
        "removed_condition_ids": sorted(set(previous_by_id) - set(updated_by_id)),
    }


def _infer_rule_update_scope(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_conditions: set[str] = set()
    target_exclusions: set[str] = set()
    allow_new_conditions = False
    allow_new_exclusions = False
    remove_condition_ids: set[str] = set()
    remove_exclusion_ids: set[str] = set()
    scoped = True

    for review in reviews:
        root_scope = _infer_scope_from_root_causes(review)
        target_conditions.update(root_scope["target_conditions"])
        target_exclusions.update(root_scope["target_exclusions"])
        allow_new_conditions = allow_new_conditions or root_scope["allow_new_conditions"]
        allow_new_exclusions = allow_new_exclusions or root_scope["allow_new_exclusions"]

        suggestion = review.get("rule_patch_suggestion", {}) or {}
        if not isinstance(suggestion, dict):
            scoped = False
            continue
        action = str(suggestion.get("action") or "").strip()
        condition_id = str(suggestion.get("condition_id") or "").strip().upper()
        if action in {"", "none"}:
            continue
        if action == "add_condition":
            allow_new_conditions = True
            continue
        if action == "add_exclusion":
            allow_new_exclusions = True
            if condition_id:
                target_exclusions.add(condition_id)
            continue
        if action in {"remove_condition", "remove_positive_condition"}:
            if condition_id:
                target_conditions.add(condition_id)
                remove_condition_ids.add(condition_id)
            else:
                scoped = False
            continue
        if action == "remove_exclusion":
            if condition_id:
                target_exclusions.add(condition_id)
                remove_exclusion_ids.add(condition_id)
            else:
                scoped = False
            continue
        if action == "narrow_exclusion":
            if condition_id:
                target_exclusions.add(condition_id)
            else:
                scoped = False
            continue
        if not condition_id:
            scoped = False
            continue
        if condition_id.upper().startswith("E"):
            target_exclusions.add(condition_id)
        else:
            target_conditions.add(condition_id)

    has_targets = bool(target_conditions or target_exclusions or allow_new_conditions or allow_new_exclusions)
    return {
        "scoped": scoped and has_targets,
        "target_conditions": sorted(target_conditions),
        "target_exclusions": sorted(target_exclusions),
        "allow_new_conditions": allow_new_conditions,
        "allow_new_exclusions": allow_new_exclusions,
        "remove_condition_ids": sorted(remove_condition_ids),
        "remove_exclusion_ids": sorted(remove_exclusion_ids),
    }


def _infer_scope_from_root_causes(review: Dict[str, Any]) -> Dict[str, Any]:
    target_conditions: set[str] = set()
    target_exclusions: set[str] = set()
    allow_new_conditions = False
    allow_new_exclusions = False

    for diagnosis in list(review.get("condition_diagnosis", []) or []):
        if not isinstance(diagnosis, dict) or not diagnosis.get("is_rule_problem"):
            continue
        _add_condition_id_to_scope(
            diagnosis.get("condition_id"),
            target_conditions=target_conditions,
            target_exclusions=target_exclusions,
        )

    for cause in list(review.get("root_causes", []) or []):
        if not isinstance(cause, dict):
            continue
        category = str(cause.get("category") or "").strip().lower()
        text = " ".join(
            str(cause.get(key) or "")
            for key in ("description", "suggested_fix")
        )
        affected = list(cause.get("affected_conditions", []) or [])

        is_rule_category = category in {
            "rule_too_broad",
            "rule_too_narrow",
            "exclusion_too_broad",
            "missing_exclusion",
            "missing_condition",
            "weak_evidence_as_sufficient",
            "bad_semantic_condition",
        }
        if not is_rule_category:
            continue

        if category == "exclusion_too_broad":
            suggestion = review.get("rule_patch_suggestion", {}) or {}
            ids = list(affected) + _extract_condition_ids(text)
            if isinstance(suggestion, dict):
                ids.append(suggestion.get("condition_id"))
            for condition_id in ids:
                normalized = str(condition_id or "").strip().upper()
                if normalized.startswith("E") and re.fullmatch(r"E\d+", normalized):
                    target_exclusions.add(normalized)
            continue

        for condition_id in affected:
            _add_condition_id_to_scope(
                condition_id,
                target_conditions=target_conditions,
                target_exclusions=target_exclusions,
            )

        for condition_id in _extract_condition_ids(text):
            _add_condition_id_to_scope(
                condition_id,
                target_conditions=target_conditions,
                target_exclusions=target_exclusions,
            )

        if category == "missing_exclusion" or "exclusion" in text.lower():
            if not any(str(cid).upper().startswith("E") for cid in affected) and not any(
                cid.upper().startswith("E") for cid in _extract_condition_ids(text)
            ):
                allow_new_exclusions = True
            for condition_id in _extract_condition_ids(text):
                if condition_id.upper().startswith("E"):
                    target_exclusions.add(condition_id)

        if category == "missing_condition":
            if not affected and not _extract_condition_ids(text):
                allow_new_conditions = True

    return {
        "target_conditions": target_conditions,
        "target_exclusions": target_exclusions,
        "allow_new_conditions": allow_new_conditions,
        "allow_new_exclusions": allow_new_exclusions,
    }


def _add_condition_id_to_scope(
    condition_id: Any,
    *,
    target_conditions: set[str],
    target_exclusions: set[str],
) -> None:
    text = str(condition_id or "").strip()
    if not text:
        return
    normalized = text.upper()
    if not re.fullmatch(r"[CE]\d+", normalized):
        return
    if normalized.startswith("E"):
        target_exclusions.add(normalized)
    else:
        target_conditions.add(normalized)


def _extract_condition_ids(text: str) -> List[str]:
    return [match.upper() for match in re.findall(r"\b[CE]\d+\b", str(text or ""), flags=re.IGNORECASE)]


def _configured_rule_update_scope(
    scope: Dict[str, Any],
    *,
    enable_structural_evolution: bool,
    enable_remove_positive_condition: bool,
    enable_remove_exclusion: bool,
) -> Dict[str, Any]:
    configured = dict(scope or {})
    configured["structural_evolution_enabled"] = bool(
        enable_structural_evolution
    )
    configured["remove_positive_condition_enabled"] = bool(
        enable_remove_positive_condition
    )
    configured["remove_exclusion_enabled"] = bool(enable_remove_exclusion)
    if not enable_structural_evolution:
        configured["allow_new_conditions"] = False
        configured["allow_new_exclusions"] = False
        configured["remove_condition_ids"] = []
        configured["remove_exclusion_ids"] = []
    else:
        if not enable_remove_positive_condition:
            configured["remove_condition_ids"] = []
        if not enable_remove_exclusion:
            configured["remove_exclusion_ids"] = []
    return configured


def _materialize_typed_rule_operations(
    rule: EvolvingRule,
    data: Dict[str, Any],
    *,
    update_scope: Dict[str, Any],
    review_bundle: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Deterministically apply signal-attributed Rule operations.

    Legacy full-Rule responses remain supported for replay compatibility. New
    updater prompts use rule_operations so structural edits cannot be smuggled
    through an unconstrained full-object rewrite.
    """
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
    if "rule_operations" not in (data or {}) and matrix_rows:
        materialized = rule.to_dict()
        audit = {
            "schema_version": "evotx.rule_structural_update.v1",
            "mode": "canonical_signal_requires_typed_operations",
            "accepted_operations": [],
            "rejected_operations": [{
                "reason": "canonical_signal_requires_rule_operations",
            }],
        }
        materialized["structural_operation_audit"] = audit
        return materialized, audit
    if "rule_operations" not in (data or {}):
        return dict(data or {}), {
            "schema_version": "evotx.rule_structural_update.v1",
            "mode": "legacy_full_rule_payload",
            "accepted_operations": [],
            "rejected_operations": [],
        }

    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    signals = {
        str(item.get("signal_id")): dict(item)
        for item in list(matrix.get("signals") or [])
        if isinstance(item, dict) and str(item.get("signal_id") or "")
    }
    signal_conditions = {
        signal_id: str(signal.get("condition_id") or "").strip().upper()
        for signal_id, signal in signals.items()
        if str(signal.get("update_target") or "").strip().lower() == "rule"
    }
    supported = _actionable_rule_signal_ids(review_bundle)
    joint_groups = [
        dict(group) for group in list(matrix.get("joint_resolution_groups") or [])
        if isinstance(group, dict)
        and str(group.get("update_target") or "rule").strip().lower() == "rule"
    ]
    raw_top_level_applied = (data or {}).get("applied_signal_ids") or []
    if isinstance(raw_top_level_applied, str):
        raw_top_level_applied = [raw_top_level_applied]
    top_level_applied = {
        str(value) for value in list(raw_top_level_applied)
        if str(value)
    }
    current_conditions = {
        str(item.id).strip().upper(): item.to_dict()
        for item in list(rule.conditions or [])
    }
    current_exclusions = {
        str(item.id).strip().upper(): item.to_dict()
        for item in list(rule.exclusion_conditions or [])
    }
    next_condition_index = _next_rule_condition_index(current_conditions, "C")
    next_exclusion_index = _next_rule_condition_index(current_exclusions, "E")
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    operation_aliases = {
        "rewrite": "rewrite_condition",
        "semantic_rewrite": "rewrite_condition",
        "tighten_condition": "rewrite_condition",
        "relax_condition": "rewrite_condition",
        "add_condition": "add_positive_condition",
        "remove_condition": "remove_positive_condition",
    }
    for operation_index, raw in enumerate(
        list((data or {}).get("rule_operations") or [])
    ):
        if not isinstance(raw, dict):
            rejected.append({
                "operation_index": operation_index,
                "reason": "operation_is_not_an_object",
            })
            continue
        operation = str(raw.get("operation") or "").strip().lower()
        operation = operation_aliases.get(operation, operation)
        condition_id = str(raw.get("condition_id") or "").strip().upper()
        description = " ".join(str(raw.get("description") or "").split())
        raw_signal_ids = raw.get("applied_signal_ids") or []
        if isinstance(raw_signal_ids, str):
            raw_signal_ids = [raw_signal_ids]
        signal_ids = tuple(dict.fromkeys(
            str(value) for value in list(raw_signal_ids) if str(value)
        ))
        reject_reason = ""
        if not signal_ids or not set(signal_ids).issubset(supported):
            reject_reason = "operation_requires_actionable_signal_ids"
        elif top_level_applied and not set(signal_ids).issubset(top_level_applied):
            reject_reason = "operation_signal_not_declared_at_top_level"
        elif any(signal_id not in signal_conditions for signal_id in signal_ids):
            reject_reason = "operation_signal_owner_is_not_rule"

        fixed_direction = {
            "add_positive_condition": "tighten",
            "remove_positive_condition": "broaden",
            "add_exclusion": "tighten",
            "remove_exclusion": "broaden",
            "narrow_exclusion": "broaden",
        }.get(operation)
        direction = str(raw.get("semantic_direction") or "").strip().lower()
        if fixed_direction:
            if direction and direction != fixed_direction:
                reject_reason = reject_reason or "operation_semantic_direction_mismatch"
            direction = fixed_direction
        elif operation == "rewrite_condition":
            if direction not in {"tighten", "broaden", "refine"}:
                reject_reason = reject_reason or "rewrite_requires_semantic_direction"
        else:
            reject_reason = reject_reason or "unsupported_rule_operation"

        source_error_types = sorted({
            str(signals.get(signal_id, {}).get("source_error_type") or "")
            .strip().upper()
            for signal_id in signal_ids
            if str(signals.get(signal_id, {}).get("source_error_type") or "").strip()
        })
        joint_basis = _complete_joint_resolution_basis(
            set(signal_ids),
            joint_groups,
        )
        if (
            not reject_reason
            and not _semantic_direction_matches_error_types(
                direction,
                source_error_types,
            )
            and not joint_basis
        ):
            reject_reason = (
                "semantic_direction_mismatches_error_without_joint_resolution"
            )

        collection = ""
        delta_operation = ""
        if operation == "rewrite_condition":
            if condition_id in current_conditions:
                collection = "conditions"
            elif condition_id in current_exclusions:
                collection = "exclusion_conditions"
            else:
                reject_reason = reject_reason or "rewrite_target_not_found"
            targets = (
                set(update_scope.get("target_conditions") or [])
                | set(update_scope.get("target_exclusions") or [])
            )
            if condition_id and condition_id not in targets:
                reject_reason = reject_reason or "rewrite_target_outside_review_scope"
            delta_operation = "replace"
        elif operation == "narrow_exclusion":
            collection = "exclusion_conditions"
            delta_operation = "replace"
            if condition_id not in current_exclusions:
                reject_reason = reject_reason or "narrow_exclusion_target_not_found"
            if condition_id not in set(update_scope.get("target_exclusions") or []):
                reject_reason = reject_reason or "narrow_exclusion_outside_review_scope"
        elif operation == "add_positive_condition":
            collection = "conditions"
            delta_operation = "add"
            if not bool(update_scope.get("allow_new_conditions")):
                reject_reason = reject_reason or "add_positive_condition_not_in_scope"
            condition_id = f"C{next_condition_index}"
        elif operation == "add_exclusion":
            collection = "exclusion_conditions"
            delta_operation = "add"
            if not bool(update_scope.get("allow_new_exclusions")):
                reject_reason = reject_reason or "add_exclusion_not_in_scope"
            condition_id = f"E{next_exclusion_index}"
        elif operation == "remove_positive_condition":
            collection = "conditions"
            delta_operation = "remove"
            if condition_id not in set(update_scope.get("remove_condition_ids") or []):
                reject_reason = reject_reason or "remove_positive_condition_disabled_or_out_of_scope"
        elif operation == "remove_exclusion":
            collection = "exclusion_conditions"
            delta_operation = "remove"
            if condition_id not in set(update_scope.get("remove_exclusion_ids") or []):
                reject_reason = reject_reason or "remove_exclusion_disabled_or_out_of_scope"

        # Canonical signal routing owns the target condition. Scope is only a
        # preservation boundary; it must not permit one signal to rewrite a
        # different in-scope condition. Add operations create a new condition
        # and therefore retain their source-condition lineage without claiming
        # that the new identifier already exists.
        if operation not in {"add_positive_condition", "add_exclusion"}:
            canonical_conditions = {
                signal_conditions.get(signal_id, "") for signal_id in signal_ids
            }
            if "" in canonical_conditions:
                reject_reason = reject_reason or "operation_signal_missing_condition"
            elif canonical_conditions != {condition_id}:
                reject_reason = reject_reason or "operation_signal_condition_mismatch"

        if delta_operation != "remove" and not description:
            reject_reason = reject_reason or "operation_description_is_empty"
        if reject_reason:
            rejected.append({
                "operation_index": operation_index,
                "operation": operation,
                "condition_id": condition_id,
                "applied_signal_ids": list(signal_ids),
                "semantic_direction": direction,
                "reason": reject_reason,
            })
            continue

        if operation == "add_positive_condition":
            current_conditions[condition_id] = {
                "id": condition_id,
                "description": description,
                "expected_answer": True,
            }
            next_condition_index += 1
        elif operation == "add_exclusion":
            current_exclusions[condition_id] = {
                "id": condition_id,
                "description": description,
                "expected_answer": True,
            }
            next_exclusion_index += 1
        elif operation == "remove_positive_condition":
            current_conditions.pop(condition_id, None)
        elif operation == "remove_exclusion":
            current_exclusions.pop(condition_id, None)
        elif collection == "conditions":
            current_conditions[condition_id]["description"] = description
        else:
            current_exclusions[condition_id]["description"] = description

        covered = [
            str(value or "").strip().upper()
            for value in list(raw.get("covered_blocker_ids") or [])
            if str(value or "").strip()
        ]
        if condition_id and condition_id not in covered:
            covered.append(condition_id)
        accepted.append({
            "operation_index": operation_index,
            "semantic_operation": operation,
            "collection": collection,
            "operation": delta_operation,
            "condition_id": condition_id,
            "description": description,
            "semantic_direction": direction,
            "applied_signal_ids": list(signal_ids),
            "source_error_types": source_error_types,
            "covered_blocker_ids": covered,
            "independently_verdict_capable": bool(
                raw.get("independently_verdict_capable", False)
            ),
            "joint_resolution_signal_ids": sorted(joint_basis),
            "rationale": str(raw.get("rationale") or "")[:700],
        })

    materialized = dict(data or {})
    materialized["description"] = rule.description
    materialized["conditions"] = list(current_conditions.values())
    materialized["exclusion_conditions"] = list(current_exclusions.values())
    materialized["decision_policy"] = _strict_rule_decision_policy(
        materialized["conditions"],
        materialized["exclusion_conditions"],
    )
    materialized["applied_signal_ids"] = list(dict.fromkeys(
        signal_id for operation in accepted
        for signal_id in operation.get("applied_signal_ids", [])
    ))
    audit = {
        "schema_version": "evotx.rule_structural_update.v1",
        "mode": "typed_rule_operations",
        "accepted_operations": accepted,
        "rejected_operations": rejected,
        "feature_flags": {
            "structural_evolution": bool(
                update_scope.get("structural_evolution_enabled")
            ),
            "remove_positive_condition": bool(
                update_scope.get("remove_positive_condition_enabled")
            ),
            "remove_exclusion": bool(
                update_scope.get("remove_exclusion_enabled")
            ),
        },
    }
    materialized["structural_operation_audit"] = audit
    return materialized, audit


def _next_rule_condition_index(
    items: Dict[str, Dict[str, Any]],
    prefix: str,
) -> int:
    indices = [
        int(match.group(1))
        for condition_id in items
        for match in [re.fullmatch(rf"{re.escape(prefix)}(\d+)", condition_id)]
        if match
    ]
    return max(indices, default=0) + 1


def _complete_joint_resolution_basis(
    applied_signal_ids: set[str],
    joint_groups: List[Dict[str, Any]],
) -> set[str]:
    basis: set[str] = set()
    for group in list(joint_groups or []):
        group_ids = {
            str(value) for value in list(group.get("signal_ids") or [])
            if str(value)
        }
        if group_ids and group_ids.issubset(applied_signal_ids):
            basis.update(group_ids)
    return basis


def _semantic_direction_matches_error_types(
    direction: str,
    source_error_types: List[str],
) -> bool:
    errors = set(source_error_types or [])
    if not errors or direction == "refine":
        return True
    if errors == {"FP"}:
        return direction == "tighten"
    if errors == {"FN"}:
        return direction == "broaden"
    return False


def _strict_rule_decision_policy(
    conditions: List[Dict[str, Any]],
    exclusions: List[Dict[str, Any]],
) -> str:
    positive_ids = ", ".join(str(item.get("id") or "") for item in conditions)
    exclusion_ids = ", ".join(str(item.get("id") or "") for item in exclusions)
    policy = f"Require every positive condition ({positive_ids}) to be satisfied."
    if exclusions:
        policy += (
            f" Reject the target verdict when any exclusion ({exclusion_ids}) "
            "is satisfied."
        )
    return policy


def _apply_minimal_update_scope(
    rule: EvolvingRule,
    data: Dict[str, Any],
    scope: Dict[str, Any],
) -> Dict[str, Any]:
    target_conditions = set(scope.get("target_conditions", []))
    target_exclusions = set(scope.get("target_exclusions", []))
    allow_new_conditions = bool(scope.get("allow_new_conditions"))
    allow_new_exclusions = bool(scope.get("allow_new_exclusions"))
    remove_condition_ids = set(scope.get("remove_condition_ids", []))
    remove_exclusion_ids = set(scope.get("remove_exclusion_ids", []))

    proposed_conditions = {
        str(item.get("id")).strip().upper(): _normalized_condition_payload(item)
        for item in list(data.get("conditions", []) or [])
        if isinstance(item, dict) and item.get("id")
    }
    proposed_exclusions = {
        str(item.get("id")).strip().upper(): _normalized_condition_payload(item)
        for item in list(data.get("exclusion_conditions", []) or [])
        if isinstance(item, dict) and item.get("id")
    }

    guarded_conditions: List[Dict[str, Any]] = []
    old_condition_ids = {str(condition.id).strip().upper() for condition in rule.conditions}
    for condition in rule.conditions:
        condition_key = str(condition.id).strip().upper()
        if condition_key in remove_condition_ids:
            continue
        if condition_key in target_conditions and condition_key in proposed_conditions:
            guarded_conditions.append(proposed_conditions[condition_key])
        else:
            guarded_conditions.append(condition.to_dict())
    if allow_new_conditions:
        for condition_id, item in proposed_conditions.items():
            if condition_id not in old_condition_ids:
                guarded_conditions.append(item)

    guarded_exclusions: List[Dict[str, Any]] = []
    old_exclusion_ids = {str(condition.id).strip().upper() for condition in rule.exclusion_conditions}
    for condition in rule.exclusion_conditions:
        condition_key = str(condition.id).strip().upper()
        if condition_key in remove_exclusion_ids:
            continue
        if condition_key in target_exclusions and condition_key in proposed_exclusions:
            guarded_exclusions.append(proposed_exclusions[condition_key])
        else:
            guarded_exclusions.append(condition.to_dict())
    if allow_new_exclusions:
        for condition_id, item in proposed_exclusions.items():
            if condition_id not in old_exclusion_ids:
                guarded_exclusions.append(item)

    scoped_actions = set(target_conditions) | set(target_exclusions)
    structural_change = bool(
        allow_new_conditions
        or allow_new_exclusions
        or remove_condition_ids
        or remove_exclusion_ids
    )
    guarded = dict(data)
    guarded["conditions"] = guarded_conditions
    guarded["exclusion_conditions"] = guarded_exclusions
    if not structural_change:
        guarded["description"] = rule.description
        guarded["decision_policy"] = rule.decision_policy
    note = str(guarded.get("update_note", ""))
    guard_note = (
        "Minimal update scope enforced: preserved unrelated conditions/exclusions; "
        f"target_conditions={sorted(target_conditions)}, "
        f"target_exclusions={sorted(target_exclusions)}."
    )
    guarded["update_note"] = f"{note}\n{guard_note}".strip()
    guarded["scope_guard"] = {
        "target_conditions": sorted(target_conditions),
        "target_exclusions": sorted(target_exclusions),
        "allow_new_conditions": allow_new_conditions,
        "allow_new_exclusions": allow_new_exclusions,
        "remove_condition_ids": sorted(remove_condition_ids),
        "remove_exclusion_ids": sorted(remove_exclusion_ids),
        "structural_change": structural_change,
        "scoped_actions": sorted(scoped_actions),
    }
    return guarded


def _freeze_rule_update_interface(
    rule: EvolvingRule,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Allow wording edits while preserving the Rule/Plan contract exactly."""
    proposed_conditions = {
        str(item.get("id") or "").strip().upper(): item
        for item in list((data or {}).get("conditions") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    proposed_exclusions = {
        str(item.get("id") or "").strip().upper(): item
        for item in list((data or {}).get("exclusion_conditions") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }

    def frozen_rows(
        current: List[RuleCondition],
        proposed: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for condition in current:
            condition_id = str(condition.id).strip().upper()
            candidate = proposed.get(condition_id) or {}
            description = str(
                candidate.get("description") or condition.description
            ).strip()
            rows.append({
                "id": condition.id,
                "description": description or condition.description,
                "expected_answer": bool(condition.expected_answer),
            })
        return rows

    frozen = dict(data or {})
    frozen["description"] = rule.description
    frozen["decision_policy"] = rule.decision_policy
    frozen["conditions"] = frozen_rows(rule.conditions, proposed_conditions)
    frozen["exclusion_conditions"] = frozen_rows(
        rule.exclusion_conditions,
        proposed_exclusions,
    )
    frozen["interface_guard"] = {
        "condition_ids": [condition.id for condition in rule.conditions],
        "exclusion_ids": [
            condition.id for condition in rule.exclusion_conditions
        ],
        "expected_answers_frozen": True,
        "top_level_semantics_frozen": True,
    }
    return frozen


def _guard_rule_update_interface(
    rule: EvolvingRule,
    data: Dict[str, Any],
    *,
    update_scope: Dict[str, Any],
) -> Dict[str, Any]:
    """Preserve unrelated slots while allowing explicitly scoped structure."""
    proposed_conditions = {
        str(item.get("id") or "").strip().upper(): item
        for item in list((data or {}).get("conditions") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    proposed_exclusions = {
        str(item.get("id") or "").strip().upper(): item
        for item in list((data or {}).get("exclusion_conditions") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    target_conditions = set(update_scope.get("target_conditions") or [])
    target_exclusions = set(update_scope.get("target_exclusions") or [])
    remove_conditions = set(update_scope.get("remove_condition_ids") or [])
    remove_exclusions = set(update_scope.get("remove_exclusion_ids") or [])
    old_condition_ids = {
        str(item.id).strip().upper() for item in list(rule.conditions or [])
    }
    old_exclusion_ids = {
        str(item.id).strip().upper()
        for item in list(rule.exclusion_conditions or [])
    }

    guarded_conditions: List[Dict[str, Any]] = []
    for condition in list(rule.conditions or []):
        condition_id = str(condition.id).strip().upper()
        if condition_id in remove_conditions:
            continue
        candidate = proposed_conditions.get(condition_id) or condition.to_dict()
        description = (
            str(candidate.get("description") or condition.description).strip()
            if condition_id in target_conditions
            else condition.description
        )
        guarded_conditions.append({
            "id": condition.id,
            "description": description or condition.description,
            "expected_answer": bool(condition.expected_answer),
        })
    if bool(update_scope.get("allow_new_conditions")):
        for condition_id, item in proposed_conditions.items():
            if condition_id in old_condition_ids:
                continue
            guarded_conditions.append({
                "id": condition_id,
                "description": str(item.get("description") or "").strip(),
                "expected_answer": True,
            })

    guarded_exclusions: List[Dict[str, Any]] = []
    for condition in list(rule.exclusion_conditions or []):
        condition_id = str(condition.id).strip().upper()
        if condition_id in remove_exclusions:
            continue
        candidate = proposed_exclusions.get(condition_id) or condition.to_dict()
        description = (
            str(candidate.get("description") or condition.description).strip()
            if condition_id in target_exclusions
            else condition.description
        )
        guarded_exclusions.append({
            "id": condition.id,
            "description": description or condition.description,
            "expected_answer": bool(condition.expected_answer),
        })
    if bool(update_scope.get("allow_new_exclusions")):
        for condition_id, item in proposed_exclusions.items():
            if condition_id in old_exclusion_ids:
                continue
            guarded_exclusions.append({
                "id": condition_id,
                "description": str(item.get("description") or "").strip(),
                "expected_answer": True,
            })

    if not guarded_conditions:
        guarded_conditions = [condition.to_dict() for condition in rule.conditions]
    all_ids = [
        str(item.get("id") or "").strip().upper()
        for item in [*guarded_conditions, *guarded_exclusions]
    ]
    if not all(all_ids) or len(all_ids) != len(set(all_ids)):
        raise ValueError("Rule update produced empty or duplicate condition ids")
    if any(not str(item.get("description") or "").strip() for item in [
        *guarded_conditions,
        *guarded_exclusions,
    ]):
        raise ValueError("Rule update produced an empty condition description")

    structural_change = (
        set(all_ids)
        != old_condition_ids | old_exclusion_ids
    )
    guarded = dict(data or {})
    guarded["description"] = rule.description
    guarded["conditions"] = guarded_conditions
    guarded["exclusion_conditions"] = guarded_exclusions
    guarded["decision_policy"] = (
        _strict_rule_decision_policy(guarded_conditions, guarded_exclusions)
        if structural_change
        else rule.decision_policy
    )
    guarded["interface_guard"] = {
        "preserved_unrelated_slots": True,
        "expected_answers_frozen": True,
        "top_level_description_frozen": True,
        "logic_model": "all_positive_and_not_any_exclusion",
        "structural_change": structural_change,
        "remove_positive_condition_enabled": bool(
            update_scope.get("remove_positive_condition_enabled")
        ),
        "remove_exclusion_enabled": bool(
            update_scope.get("remove_exclusion_enabled")
        ),
    }
    return guarded


def _rule_payload_has_semantic_change(
    rule: EvolvingRule,
    data: Dict[str, Any],
) -> bool:
    return (
        [item.to_dict() for item in rule.conditions]
        != list((data or {}).get("conditions") or [])
        or [item.to_dict() for item in rule.exclusion_conditions]
        != list((data or {}).get("exclusion_conditions") or [])
        or rule.description != str((data or {}).get("description", rule.description))
        or rule.decision_policy
        != str((data or {}).get("decision_policy", rule.decision_policy))
    )


def _normalized_condition_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(item)
    if payload.get("id"):
        payload["id"] = str(payload.get("id")).strip().upper()
    return payload


def _condition_interface(conditions: List[RuleCondition]) -> List[tuple[str, bool]]:
    return [(str(condition.id), bool(condition.expected_answer)) for condition in conditions]


def _compression_preserves_rule_interface(
    original: EvolvingRule,
    compressed: EvolvingRule,
) -> bool:
    return (
        _condition_interface(original.conditions)
        == _condition_interface(compressed.conditions)
        and _condition_interface(original.exclusion_conditions)
        == _condition_interface(compressed.exclusion_conditions)
    )


def _cohort_signal_gate_enabled(review_bundle: Dict[str, Any]) -> bool:
    return bool(
        ((review_bundle or {}).get("cohort_signal_summary") or {}).get(
            "schema_version"
        )
    )


def _canonical_signal_contract_enabled(review_bundle: Dict[str, Any]) -> bool:
    return str(
        ((review_bundle or {}).get("canonical_signal_contract") or {}).get(
            "schema_version"
        )
        or ""
    ).startswith("evotx.canonical_update_signal_contract.")


def _actionable_update_signal_ids(
    review_bundle: Dict[str, Any],
    update_target: str,
) -> set[str]:
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    target = str(update_target or "").strip().lower()
    if target == "rule" and bool(
        (review_bundle or {}).get("phase3_experimental_rule_probe")
    ):
        source_ids = matrix.get("experimental_rule_signal_ids") or []
    elif target == "plan":
        source_ids = matrix.get("plan_actionable_signal_ids") or []
    else:
        source_ids = matrix.get("supported_signal_ids") or []
    signal_by_id = {
        str(signal.get("signal_id") or ""): signal
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict) and str(signal.get("signal_id") or "")
    }
    if not signal_by_id:
        return {str(value) for value in list(source_ids or []) if str(value)}
    return {
        str(signal_id)
        for signal_id in list(source_ids or [])
        if str(signal_id)
        and str(
            signal_by_id.get(str(signal_id), {}).get("update_target") or ""
        ).strip().lower() == target
    }


def _actionable_rule_signal_ids(review_bundle: Dict[str, Any]) -> set[str]:
    return _actionable_update_signal_ids(review_bundle, "rule")


def _review_uses_signal_ids(
    review: Dict[str, Any],
    allowed_signal_ids: set[str],
) -> bool:
    return any(
        isinstance(signal, dict)
        and str(signal.get("signal_id") or "") in allowed_signal_ids
        for signal in list((review or {}).get("update_signals") or [])
    )


def _review_has_supported_signal(
    review: Dict[str, Any],
    *,
    update_target: str,
) -> bool:
    if update_target == "rule" and not _review_allows_rule_training(review):
        return False
    return any(
        isinstance(signal, dict)
        and str(signal.get("generalization_status") or "").lower().strip()
        == "supported"
        and str(signal.get("update_target") or review.get("update_target") or "")
        == update_target
        for signal in list((review or {}).get("update_signals", []) or [])
    )


def _review_is_supported_fp_boundary_tightening(review: Dict[str, Any]) -> bool:
    """Allow FP boundary-tighten signals even when the target mechanism is absent."""
    if str((review or {}).get("error_type") or "").strip().upper() != "FP":
        return False
    directions = {"tighten", "add_exclusion", "clarify_boundary"}
    for signal in list((review or {}).get("update_signals", []) or []):
        if not isinstance(signal, dict):
            continue
        target = str(
            signal.get("update_target") or (review or {}).get("update_target") or ""
        ).strip().lower()
        status = str(signal.get("generalization_status") or "").strip().lower()
        direction = str(signal.get("direction") or "").strip().lower()
        if target == "rule" and status == "supported" and direction in directions:
            return True
    return False


def _review_is_supported_fn_exclusion_boundary_clarification(
    review: Dict[str, Any],
) -> bool:
    """Allow supported FN fixes that only narrow an over-firing exclusion."""
    if str((review or {}).get("error_type") or "").strip().upper() != "FN":
        return False
    overfired_exclusions: set[str] = set()
    for diagnosis in list((review or {}).get("condition_diagnosis") or []):
        if not isinstance(diagnosis, dict):
            continue
        condition_id = str(diagnosis.get("condition_id") or "").strip().upper()
        if not condition_id.startswith("E"):
            continue
        if diagnosis.get("observed_answer") is True and diagnosis.get(
            "expected_for_correct_verdict"
        ) is False:
            overfired_exclusions.add(condition_id)
    if not overfired_exclusions:
        return False
    directions = {"tighten", "clarify_boundary", "narrow_exclusion"}
    for signal in list((review or {}).get("update_signals", []) or []):
        if not isinstance(signal, dict):
            continue
        target = str(
            signal.get("update_target") or (review or {}).get("update_target") or ""
        ).strip()
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        status = str(signal.get("generalization_status") or "").strip().lower()
        direction = str(signal.get("direction") or "").strip().lower()
        if (
            target == "rule"
            and condition_id in overfired_exclusions
            and status == "supported"
            and direction in directions
            and str(signal.get("abstract_feature") or "").strip()
        ):
            return True
    return False


def _review_allows_rule_training(review: Dict[str, Any]) -> bool:
    error_type = str((review or {}).get("error_type") or "").strip().upper()
    fn_exclusion_boundary = _review_is_supported_fn_exclusion_boundary_clarification(
        review
    )
    target_observed = str(
        (review or {}).get("target_mechanism_observed") or ""
    ).strip().lower()
    gt_supported = str(
        (review or {}).get("ground_truth_supported_by_packet") or ""
    ).strip().lower()
    if gt_supported == "no" and not _review_is_supported_fp_boundary_tightening(
        review
    ):
        return False
    if error_type == "FN":
        if target_observed and target_observed != "yes" and not fn_exclusion_boundary:
            return False
        if bool((review or {}).get("generic_symptoms_only")) and not fn_exclusion_boundary:
            return False
    training_gate = dict((review or {}).get("training_gate") or {})
    blocked = dict(training_gate.get("blocked_reasons") or {})
    blocked_rule = str(blocked.get("rule") or "").strip()
    if fn_exclusion_boundary and blocked_rule in {
        "fn_target_mechanism_not_observed",
        "fn_target_mechanism_not_confirmed_for_rule_update",
        "fn_generic_symptoms_only",
    }:
        return True
    return not blocked_rule


def _payload_uses_actionable_rule_signals(
    payload: Dict[str, Any],
    review_bundle: Dict[str, Any],
) -> bool:
    supported = _actionable_rule_signal_ids(review_bundle)
    raw_applied = (payload or {}).get("applied_signal_ids", [])
    if isinstance(raw_applied, str):
        raw_applied = [raw_applied]
    applied = {str(value) for value in list(raw_applied or []) if str(value)}
    if not applied or not applied.issubset(supported):
        return False
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    return _applied_signals_satisfy_joint_groups(
        applied,
        matrix,
        update_target="rule",
    )


def _payload_uses_supported_signals(
    payload: Dict[str, Any],
    review_bundle: Dict[str, Any],
) -> bool:
    """Backward-compatible name for canonical Rule signal lineage validation."""
    return _payload_uses_actionable_rule_signals(payload, review_bundle)


def _applied_signals_satisfy_joint_groups(
    applied: set[str],
    matrix: Dict[str, Any],
    *,
    update_target: str,
) -> bool:
    target = str(update_target or "").lower().strip()
    for group in list((matrix or {}).get("joint_resolution_groups") or []):
        if not isinstance(group, dict):
            continue
        if str(group.get("update_target") or "").lower().strip() != target:
            continue
        group_ids = {
            str(value)
            for value in list(group.get("signal_ids") or [])
            if str(value)
        }
        if applied & group_ids and not group_ids.issubset(applied):
            return False
    return True


def _payload_contains_case_specific_update_terms(payload: Dict[str, Any]) -> bool:
    text = stable_json_dumps(payload or {}).lower()
    return bool(re.search(
        r"0x[0-9a-f]{6,}|\b(?:call|event|transfer|sload|sstore):\d+\b|"
        r"\b(?:ctf|challenge[- ]?like|joke selector|codeislaw)\b",
        text,
    ))
