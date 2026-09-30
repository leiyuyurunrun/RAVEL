from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional

from evotx.core.labels import normalize_attack_label
from evotx.evolution.diagnosis_routing import apply_diagnosis_routing
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
from evotx.utils.result_utils import as_reviewer_case, get_finding, get_ground_truth
from evotx.utils.json_utils import (
    JsonExtractionError,
    extract_json_object,
    json_object_candidates,
    stable_json_dumps,
)
from evotx.utils.result_slimmer import normalize_rejected_update_memory


REVIEW_COHORT_FEATURE_BUDGET_CHARS = 5000
REVIEW_CASE_BOUNDARY_FEATURES_PER_TYPE = 2
REVIEW_CASE_BOUNDARY_VIEW_LIMIT = 3
REVIEW_CASE_BOUNDARY_TOOL_LIMIT = 2
REVIEW_PLAN_AUDIT_RELEVANT_PATH_LIMIT = 1
REVIEW_SOURCE_SNIPPET_CHARS = 900

REVIEWER_FORMAT_RETRY_INSTRUCTION = """

FORMAT RETRY:
Your previous response could not be parsed as the required JSON object. Repeat
the exact same review task and preserve its semantic intent. Return only one
complete JSON object matching the requested schema. Do not use markdown fences,
commentary, or placeholder text such as {JSON}.
"""


REVIEWER_EVIDENCE_FIRST_GUIDANCE = """
Evidence-first label-blind audit:
- training_supervision.label_rationale, when present, is an optional CSV Cause
  supplied only to Reviewer/Updater. Use it to locate the intended abstract
  mechanism and to detect a likely evidence-routing miss, but never cite it as
  transaction evidence or treat it as proof that a condition is true. Require
  packet/source evidence before recommending a semantic change.
- contrastive_supervision and contrastive_label_coverage are training-only
  labels. They identify the target family, the non-target source family, and
  training coverage; they are not transaction evidence and must never be cited
  as proof that a condition is true or false.
- Use a non-target source label only to ask which required target-positive
  mechanism is absent or replaced. Never recommend a condition or exclusion
  that names, profiles, or directly rejects that source attack family.
- Before using ground_truth or raw_ground_truth, determine whether the supplied
  condition/evidence summaries actually contain the target-family root-cause
  mechanism. Ground truth is supervision about the expected class, not evidence
  that a particular mechanism is visible in this packet.
- Set target_mechanism_observed=yes only when target-specific evidence is
  present; use no when the packet contradicts it, and uncertain when the needed
  view/source/parameter evidence is unavailable.
- A positive target label with no target-specific proof does not by itself make
  the rule too narrow. Route missing/coarse/unused evidence to plan or packet;
  use unknown/none when the compact case cannot distinguish label noise from an
  evidence limitation.
- Set generic_symptoms_only=true when the available support is limited to
  profit, value extraction, flash loans, swaps, callbacks/reentrancy, unknown
  selectors, public entry calls, unusual amounts, or asymmetric payout without
  the target-family mechanism.
- Before recommending a rule edit, perform a counterfactual boundary test:
  would the proposed wording also fire on benign transactions or unseen
  non-target attack families that share these generic symptoms? If yes, do not
  broaden; route the evidence problem or propose a target-specific boundary.
- A rule update requires target-specific observed/partial features linked to the
  affected condition. Do not reverse-engineer a rule patch merely to agree with
  the supplied label.
- A single error case may propose an abstract update hypothesis, but it does not
  authorize a rule or plan edit by itself. Compare the hypothesis against
  cohort_signal_summary from correctly classified target positives, correctly
  classified negatives, and guard cases.
- Cohort feature observations intentionally preserve different natural-language
  descriptions. Compare them semantically; do not require exact wording. Cite
  the feature_ref values that support or conflict with each rule/plan signal.
- Aggregator audit fields are first-class evidence for routing. Distinguish a
  local Judge evidence failure from an unjustified dynamic override.
- Refine bad_judge_question instead of routing it wholesale. If the Rule
  semantic itself is wrong, set failure_origin=rule_semantic and request a Rule
  semantic rewrite. If the Rule semantic is correct but the Plan/Judge question
  drifted, set failure_origin=plan_question_drift,
  rule_semantic_status=correct, question_alignment_status=drifted, and request
  restore_question_from_rule. If the question is correct but its evidence path
  is insufficient, set failure_origin=evidence_route and request
  change_evidence_route.
- missing_evidence_origins contains only active current-round limitations and is
  authoritative runtime attribution. Do not guess or overwrite its
  Plan/Packet/Runtime/unavailable origin. missing_evidence_origin_history and
  condition-level missing_evidence_history are audit history only; resolved,
  stale, or non-actionable rows must not become current blockers.
- Treat available_packet_views and the configured runtime tools as the closed
  evidence universe for this run. Do not propose hypothetical decompilers,
  bytecode-semantics views, cross-transaction history, or any unregistered
  capability as a Packet/Runtime update.
- Emit Packet only when runtime attribution proves that a registered view should
  exist but was not constructed, packet/trace data was actually truncated, or
  Packet build/load failed. Emit Runtime only for an observed tool/runtime
  failure. If an existing view was not routed or consumed, emit Plan. If all
  available routes were consumed but the desired semantic fact remains
  unobservable, emit no actionable evidence update and use insufficient.
- If the Rule unnecessarily requires evidence outside this closed universe,
  emit a normal Rule refinement only when existing transaction evidence and
  cohort boundaries support a safer observable semantic boundary. Never convert
  an unavailable fact to Rule automatically.
- If source code and observed runtime trace/state/fund-flow conflict, prefer
  observed runtime facts unless the source is reliably bound to the executed
  implementation/version. Unpinned source modifiers, nonReentrant guards, or
  apparent safe ordering may explain semantics but are not decisive negative
  evidence against contradictory trace behavior.
- One case may contain independent owner-specific failures. Emit separate
  update_signals for Rule semantics, Plan evidence routing, and Packet/Runtime
  engineering as needed; do not collapse them into one update_target.
- Populate condition_evidence_dependency only for an explicit cross-owner pair:
  a Rule signal with role=requires and a Plan signal with role=provides must use
  the same condition_id and abstract capability_id. Do not add this field merely
  because a Plan contains views or follow-ups; it is reserved for cases where a
  semantic repair cannot be judged without a specific evidence capability.
- For stateful binding failures, use binding_semantic_failure when the required
  producer/consumer meaning is wrong, binding_route_failure when depends_on or
  the selected producer path is wrong, and binding_runtime_failure when schema
  handling or state propagation failed despite a valid Rule and Plan.
- Mark generalization_status=supported only when the error exposes a concrete
  distinction and the proposed change preserves the supplied correct-positive
  mechanism signals and correct-negative/guard boundaries. Use conflicted when
  it would reject a protected positive signal, and insufficient when no cohort
  distinction is available.
- For a Plan signal, supported means the target condition and existing evidence
  capability are concrete and the required route was not successfully executed.
  It does not require certainty that the route will change the final verdict;
  candidate validation owns that effectiveness decision. Use insufficient only
  when the condition/capability/route relation itself is not concrete.
- Treat case_boundary_context as the compact memory of ordinary/correct cases.
  It is not evidence for the error case, but it is mandatory preservation
  context: a repair signal must explain how correct positives remain detectable
  and how correct negatives/guard cases remain outside the target family.
- If a proposed repair cannot be reconciled with case_boundary_context, set its
  generalization_status to conflicted or insufficient instead of asking the
  updater to try it.
- The executable Rule currently supports only all positive conditions AND
  NOT(any exclusion). If the error requires an alternate sufficient attack
  path or an OR branch, diagnose deferred_logic_structure_failure. Do not
  compensate by weakening an existing condition until it accidentally covers
  both paths.
- Every rule_patch_suggestion must declare classifier-level semantic_direction:
  add positive condition=tighten, remove positive condition=broaden, add
  exclusion=tighten, remove exclusion=broaden, and rewrite=tighten, broaden,
  or refine. Prefer tighten/refine for FP and broaden/refine for FN; a mismatch
  needs an explicit joint-resolution rationale.
- Contract names, protocol names, unusual function names, unknown or playful
  selectors, CTF/challenge appearance, and assumed contract purpose are never
  sufficient generalization signals. Do not infer that behavior is intended or
  benign from those attributes.
"""


def _reviewer_label_specific_guidance(target_label: str) -> str:
    if normalize_attack_label(target_label) != "price_manipulation":
        return ""
    return """Price-manipulation review boundary:
- Keep price_manipulation distinct from the broader market_manipulation label.
  Skim/donate, sync, sandwich/MEV ordering, pool-accounting anomalies, flash
  capital, swaps, and profit are not positive price evidence by themselves.
- A token-accounting or operation-order defect may establish the price-source
  perturbation when it actually distorts a reserve or price input. A
  same-market swap, settlement, or extraction may satisfy consumption when its
  amount or outcome demonstrably uses that distorted reserve/price; the
  consumer need not be a separate external protocol.
- Do not infer consumption merely because the actor's later swap observes the
  normal reserve update produced by its earlier swap. For same-market
  consumption, identify the independently abnormal distortion and explain how
  it causally changed the consumer's amount, limit, settlement, or extraction
  beyond ordinary sequential AMM pricing.
- A bare skim/sync/reserve write, or a swap merely co-occurring with reserve
  movement, is insufficient without distorted-price consumption and a causal
  adverse outcome.
- Do not diagnose deferred_logic_structure_failure solely because a positive
  uses same-market consumption while the current wording names downstream
  borrow/mint/redeem examples. First test whether one causal semantic rewrite
  can cover both while retaining the bare-skim and ordinary-swap boundary.
- If the semantic condition is sound but broad critical-call evidence was not
  tied to the price source, route the issue to the Plan evidence strategy."""


def _extract_reviewer_json(text: str) -> tuple[Dict[str, Any], str]:
    """Select the last object, closing only complete containers cut at EOF."""
    candidates = json_object_candidates(text)
    if candidates:
        return candidates[-1][2], ""
    try:
        return extract_json_object(text), ""
    except JsonExtractionError as original_error:
        repaired = _complete_truncated_reviewer_object(text)
        if repaired is None:
            raise original_error
        return repaired, "close_unterminated_eof_container"


def _complete_truncated_reviewer_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").rstrip()
    starts = [index for index, char in enumerate(raw) if char == "{"]
    for start in reversed(starts):
        candidate = raw[start:]
        stack: list[str] = []
        in_string = False
        escaped = False
        invalid = False
        for char in candidate:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                expected = "{" if char == "}" else "["
                if not stack or stack[-1] != expected:
                    invalid = True
                    break
                stack.pop()
        if invalid or in_string or not stack or len(stack) > 8:
            continue
        suffix = "".join("}" if char == "{" else "]" for char in reversed(stack))
        try:
            parsed = json.loads(candidate + suffix)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class RuleReviewer:
    """Diagnose FP/FN traces and propose rule-level fixes."""

    def __init__(
        self,
        llm=None,
        transcript_dir: str | Path | None = None,
        max_prompt_chars: int = 0,
        compact_mode: str = "error-focused",
        negative_training_mode: str = "mixed",
    ):
        self.llm = llm
        self.transcript_dir = transcript_dir
        self.max_prompt_chars = max(0, int(max_prompt_chars or 0))
        self.compact_mode = _normalize_review_compact_mode(compact_mode)
        self.negative_training_mode = normalize_negative_training_mode(
            negative_training_mode
        )

    def review_error(self, case: Dict[str, Any]) -> Dict[str, Any]:
        case = as_reviewer_case(case)
        case["negative_training_mode"] = self.negative_training_mode
        if self.llm is None:
            return self._fallback_review(case, "No LLM reviewer is configured.")

        prompt, prompt_case, truncation = self._build_prompt_with_audit(case)
        raw_completion = ""
        try:
            raw_completion = self.llm.complete(prompt)
            review, deterministic_repair = _extract_reviewer_json(
                llm_response_text(raw_completion)
            )
            normalized = self._normalize_review(review, case)
            if deterministic_repair:
                normalized["review_execution"] = {
                    "status": "recovered",
                    "failure_stage": "reviewer_response_parse",
                    "failure_class": "truncated_json_container",
                    "retry_attempted": False,
                    "retry_count": 0,
                    "deterministic_repair": deterministic_repair,
                }
            self._write_audit(
                case,
                prompt_case,
                prompt,
                raw_completion,
                review,
                normalized,
                truncation=truncation,
            )
            return normalized
        except JsonExtractionError as first_exc:
            retry_pending = self._fallback_review(
                case,
                f"Review response parse failed; retrying format: {first_exc}",
                review_execution={
                    "status": "retrying",
                    "failure_stage": "reviewer_response_parse",
                    "failure_class": type(first_exc).__name__,
                    "retry_attempted": True,
                    "retry_count": 0,
                },
            )
            self._write_audit(
                case,
                prompt_case,
                prompt,
                raw_completion,
                None,
                retry_pending,
                truncation=truncation,
                error=repr(first_exc),
            )

            retry_prompt = f"{prompt}{REVIEWER_FORMAT_RETRY_INSTRUCTION}"
            retry_completion = ""
            try:
                retry_completion = self.llm.complete(retry_prompt)
                review, deterministic_repair = _extract_reviewer_json(
                    llm_response_text(retry_completion)
                )
                normalized = self._normalize_review(review, case)
                normalized["review_execution"] = {
                    "status": "recovered",
                    "failure_stage": "reviewer_response_parse",
                    "failure_class": type(first_exc).__name__,
                    "retry_attempted": True,
                    "retry_count": 1,
                    **(
                        {"deterministic_repair": deterministic_repair}
                        if deterministic_repair
                        else {}
                    ),
                }
                self._write_audit(
                    case,
                    prompt_case,
                    retry_prompt,
                    retry_completion,
                    review,
                    normalized,
                    truncation=truncation,
                )
                return normalized
            except Exception as retry_exc:
                fallback = self._fallback_review(
                    case,
                    f"Review failed after format retry: {retry_exc}",
                    review_execution={
                        "status": "failed",
                        "failure_stage": "reviewer_response_parse",
                        "failure_class": type(retry_exc).__name__,
                        "initial_failure_class": type(first_exc).__name__,
                        "retry_attempted": True,
                        "retry_count": 1,
                    },
                )
                self._write_audit(
                    case,
                    prompt_case,
                    retry_prompt,
                    retry_completion,
                    None,
                    fallback,
                    truncation=truncation,
                    error=repr(retry_exc),
                )
                return fallback
        except Exception as exc:
            fallback = self._fallback_review(
                case,
                f"Review failed: {exc}",
                review_execution={
                    "status": "failed",
                    "failure_stage": "reviewer_request_or_normalization",
                    "failure_class": type(exc).__name__,
                    "retry_attempted": False,
                    "retry_count": 0,
                },
            )
            self._write_audit(
                case,
                prompt_case,
                prompt,
                raw_completion,
                None,
                fallback,
                truncation=truncation,
                error=repr(exc),
            )
            return fallback
    def _build_prompt_with_audit(
        self,
        case: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        if self.compact_mode != "legacy":
            return self._build_error_focused_prompt_with_audit(case)

        prompt = self._build_prompt(case)
        if self.max_prompt_chars <= 0 or len(prompt) <= self.max_prompt_chars:
            return prompt, case, build_truncation_audit(
                input_name="reviewer_case",
                before=case,
                after=case,
                max_chars=self.max_prompt_chars,
            )

        prompt_case, removed_fields, removed_item_counts = _compact_reviewer_case_to_fit(
            case,
            max_prompt_chars=self.max_prompt_chars,
            build_prompt=self._build_prompt,
        )
        prompt = self._build_prompt(prompt_case)
        if len(prompt) > self.max_prompt_chars:
            prompt_case, compact_fields, compact_counts = _compact_reviewer_case_to_fit(
                case,
                max_prompt_chars=self.max_prompt_chars,
                build_prompt=self._build_compact_prompt,
            )
            prompt = self._build_compact_prompt(prompt_case)
            removed_fields = sorted(
                set(removed_fields + compact_fields + ["reviewer_prompt.guidance_compacted"])
            )
            for key, value in compact_counts.items():
                removed_item_counts[key] = removed_item_counts.get(key, 0) + int(value or 0)
        truncation = build_truncation_audit(
            input_name="reviewer_case",
            before=case,
            after=prompt_case,
            max_chars=self.max_prompt_chars,
            reason="reviewer_prompt_max_chars",
            removed_fields=removed_fields,
            removed_item_counts=removed_item_counts,
        )
        truncation["prompt_chars_after"] = len(prompt)
        truncation["prompt_over_limit_after_compaction"] = (
            len(prompt) > self.max_prompt_chars
        )
        truncation["compact_strategy"] = "legacy"
        return prompt, prompt_case, truncation

    def _build_error_focused_prompt_with_audit(
        self,
        case: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        focus_case = _build_error_focused_reviewer_case(case)
        prompt = self._build_error_focused_prompt(focus_case)
        if self.max_prompt_chars <= 0 or len(prompt) <= self.max_prompt_chars:
            truncation = build_truncation_audit(
                input_name="reviewer_case",
                before=case,
                after=focus_case,
                max_chars=self.max_prompt_chars,
                reason="reviewer_error_focused_compact",
                removed_fields=["reviewer_case.replaced_by_error_focused_packet"],
            )
            truncation.update(_error_focused_truncation_metadata(prompt, focus_case))
            return prompt, focus_case, truncation

        prompt_case, removed_fields, removed_counts = _compact_error_focused_case_to_fit(
            focus_case,
            max_prompt_chars=self.max_prompt_chars,
            build_prompt=self._build_error_focused_prompt,
        )
        prompt = self._build_error_focused_prompt(prompt_case)
        truncation = build_truncation_audit(
            input_name="reviewer_case",
            before=case,
            after=prompt_case,
            max_chars=self.max_prompt_chars,
            reason="reviewer_error_focused_prompt_max_chars",
            removed_fields=[
                "reviewer_case.replaced_by_error_focused_packet",
                *removed_fields,
            ],
            removed_item_counts=removed_counts,
        )
        truncation.update(_error_focused_truncation_metadata(prompt, prompt_case))
        truncation["prompt_over_limit_after_compaction"] = (
            self.max_prompt_chars > 0 and len(prompt) > self.max_prompt_chars
        )
        return prompt, prompt_case, truncation

    def _write_audit(
        self,
        case: Dict[str, Any],
        prompt_case: Dict[str, Any],
        prompt: str,
        raw_completion: Any,
        parsed_review: Any,
        normalized_review: Dict[str, Any],
        *,
        truncation: Dict[str, Any],
        error: str = "",
    ) -> None:
        tx_hash = str(case.get("tx_hash") or (case.get("transaction", {}) or {}).get("tx_hash") or "")
        target_label = str(case.get("target_label") or case.get("attack_label") or "")
        error_type = str(normalized_review.get("error_type") or infer_error_type(case))
        name = "__".join(
            part
            for part in (
                target_label or "unknown_label",
                error_type or "unknown_error",
                tx_hash[:12] or "review",
            )
            if part
        )
        case_chars = len(stable_json_dumps(case))
        prompt_case_chars = len(stable_json_dumps(prompt_case))
        path = write_llm_audit_transcript(
            output_dir=self.transcript_dir,
            stage="reviewer",
            name=name,
            model=llm_model_name(self.llm),
            prompt=prompt,
            raw_completion=raw_completion,
            parsed_response={
                "parsed_review": parsed_review,
                "normalized_review": normalized_review,
            },
            usage=llm_usage(self.llm, raw_completion),
            finish_reason=llm_finish_reason(self.llm, raw_completion),
            metadata={
                "tx_hash": tx_hash,
                "target_label": target_label,
                "error_type": error_type,
                "sample_role": case.get("sample_role", ""),
                "negative_kind": case.get("negative_kind", ""),
                "negative_training_mode": case.get("negative_training_mode", ""),
                "ground_truth": case.get("ground_truth"),
                "predicted_verdict": case.get("predicted_verdict"),
                "reviewer_case_chars": case_chars,
                "reviewer_prompt_case_chars": prompt_case_chars,
                "max_prompt_chars": self.max_prompt_chars,
                "review_compact_mode": self.compact_mode,
                "error": error,
            },
            truncation=truncation,
        )
        if self.transcript_dir is not None:
            print(
                "[LLM-Audit] reviewer "
                f"tx={tx_hash[:12] or 'unknown'} "
                f"prompt_chars={len(str(prompt or ''))} "
                f"est_tokens={estimate_prompt_tokens(prompt)} "
                f"truncated={str(bool(truncation.get('applied'))).lower()} "
                f"path={path or ''}"
            )

    @staticmethod
    def _build_prompt(case: Dict[str, Any]) -> str:
        raw_label = case.get("raw_ground_truth", case.get("ground_truth"))
        rule_digest = case.get("rule_digest") if isinstance(case.get("rule_digest"), dict) else {}
        raw_target_label = (
            case.get("target_label")
            or case.get("attack_label")
            or rule_digest.get("attack_label")
            or (rule_digest.get("metadata", {}) or {}).get("attack_label")
            or raw_label
        )
        target_label = normalize_attack_label(raw_target_label)
        negative_training_mode = normalize_negative_training_mode(
            case.get("negative_training_mode", "mixed")
        )
        negative_training_guidance = negative_training_mode_guidance(
            negative_training_mode
        )
        negative_case_guidance = negative_case_boundary_guidance(
            negative_training_mode
        )
        return f"""
You are an EvoTx transaction detection rule reviewer.

{REVIEWER_EVIDENCE_FIRST_GUIDANCE}

{_reviewer_label_specific_guidance(target_label)}

Analyze why the current evolving rule produced a wrong prediction.
You receive a compact reviewer_case, not the full packet evidence.
Base your diagnosis on rule_digest, plan_digest, condition_table,
cross_condition_fact_consistency_audit, finding_summary, evidence_digest, missing evidence summaries, tool-call
summaries, missing_evidence_actionability_summary, and evidence adequacy
metadata.
condition_table rows may include condition_feature_analysis. Use those fields
when diagnosing why a local condition was wrong. Do not ignore
condition_feature_analysis when present; it summarizes what the judge actually
matched, nearly matched, missed, contradicted, or considered boundary-risky.
If cross_condition_fact_consistency_audit reports a conflict, do not choose one
Judge narrative as truth and do not learn a Rule rewrite from the contradiction.
Route a targeted Plan evidence-resolution signal that reconstructs the shared
anchor, temporal order, and consumer arguments/local context.

The case may include raw_ground_truth, sample_role, and negative_kind. For this
task, ground_truth="attack" means the transaction belongs to the current target
attack family, while ground_truth="benign" means non-target. A non-target
sample can be truly benign or another attack family. Do not describe
negative_other / other_attack samples as benign behavior; diagnose them as
target-vs-non-target boundary cases when appropriate. Keep your diagnosis
aligned to the target attack family rather than drifting into generic
transaction anomaly logic.
If the observed negative set is narrow or dominated by one non-target family,
do not recommend changes that only separate the target from that one family.
Preserve a mixed non-target boundary: root-cause distinctions should remain
valid against benign cases and other attack families not present in the current
few-shot set.

Negative training context:
- negative_training_mode="{negative_training_mode}".
- {negative_training_guidance}
- {negative_case_guidance}
This mode describes dataset coverage, not evidence about this transaction. Use
the case-level sample_role and negative_kind for the local diagnosis.

Classify root causes at the rule/plan/judge/packet-evidence level:
- missing_evidence
- missing_packet_view
- bad_packet_view_selection
- packet_evidence_too_coarse
- bad_judge_question
- rule_too_broad
- rule_too_narrow
- exclusion_too_broad
- missing_exclusion
- weak_evidence_as_sufficient
- runtime_emit_logic_bug
- packet_build_or_load_failure
- unknown

Important routing rules:
- If all core conditions are true, all exclusions are false, but verdict is
  benign/uncertain, classify update_target="runtime".
- If missing evidence is due to omitted packet views or trace truncation,
  classify update_target="packet" or "plan", not rule.
- Do not treat prompt_render_truncation as packet failure when
  packet_trace_truncated is false.
- Do not treat non_actionable missing evidence as a rule failure.
- external_market_data can be noted as a limitation, but does not force a rule
  update by itself.
- Only set update_target="rule" if the semantic rule is too broad, too narrow,
  missing an exclusion, or a condition definition is wrong.
- For FP on negative_other / other_attack samples, prefer categories such as
  rule_too_broad, missing_exclusion, or weak_evidence_as_sufficient when the
  target rule cannot distinguish the target attack family from another attack
  family. Do not call that other attack family a benign explanation.
- When recommending missing_exclusion or add_exclusion, frame the exclusion as
  a target-positive boundary: specify which required target mechanism is absent,
  fully authorized, entitlement-backed, or completely replaced by another
  primary root cause. Do not recommend an exclusion solely because a non-target
  family label is present.
- For FN on a positive target where condition_table shows an exclusion condition
  answered true, diagnose whether the exclusion is too broad. Use
  exclusion_too_broad when the exclusion fires from a secondary mechanism or
  attack-path feature even though the target-family root cause remains present.
- For source-dependent targets, do not classify source_unavailable by itself as
  a semantic rule failure. If packet evidence contains access-control-specific
  structural signals but the plan/judge failed to use them after source was
  unavailable, prefer update_target="plan". If the semantic rule itself makes
  verified source mandatory despite strong target-specific packet evidence,
  update_target may be "rule".
- For access-control FP cases, diagnose whether the rule accepted generic
  exploit evidence as sufficient. Value extraction, flash loans, swaps,
  reentrancy/callback ordering, unusual profit, or public entry functions alone
  should be treated as weak_evidence_as_sufficient or rule_too_broad, not as
  access-control proof. Prefer tightening conditions or adding exclusions that
  require authorization/permission/privileged-path evidence.
- For access-control FN cases with unavailable source, distinguish two fixes:
  plan/judge fix when packet views already contain owner/admin/role/initializer/
  proxy/delegatecall/authorization-slot/protected-value-release evidence but the
  judge did not use it; rule fix only when the rule wording itself forbids
  behavioral fallback from such access-control-specific evidence.
- For every access-control case, verify whether the core condition rows refer
  to one coherent authorization chain: the same actor/caller or beneficiary,
  required authority, sensitive capability and protected target/resource, and
  resulting protected effect. If the prediction stitches unrelated calls,
  actors, assets, positions, or protocol components together, diagnose a
  rule/plan binding failure rather than treating generic suspicious evidence as
  access-control proof. Missing source alone is not a binding failure when
  transaction evidence connects the complete chain.
- For insufficient_validation cases, do not diagnose by fixed condition IDs.
  Read each condition's text and classify its semantic role:
  input/state consumption, validation/acceptance semantics, incorrect outcome
  or invariant violation, access-control exclusion, market/oracle exclusion,
  reentrancy/callback exclusion, token-semantic exclusion, or normal-operation
  exclusion.
- For insufficient_validation FN cases, watch for two common failure modes:
  (1) a core condition requires visibly malformed inputs and misses
  syntactically well-formed but semantically invalid inputs, callback data,
  return values, external state, boundary/precision cases, entitlement gaps, or
  business-invariant violations; this is a rule_too_narrow issue;
  (2) an exclusion fires merely because flashloan, callback/reentrancy,
  price/oracle movement, market interaction, or token/accounting behavior is
  present, even though the exploit succeeds because the protocol failed to
  validate consumed data/state; this is usually an exclusion_too_broad or
  weak_evidence_as_sufficient rule issue.
- For insufficient_validation FP cases, ask whether the rule accepted generic
  value extraction or another attack family as sufficient. If the positive
  conditions do not require a causal validation gap, diagnose rule_too_broad. If
  the plan failed to show the alternative-family evidence needed by an
  exclusion, diagnose plan/bad_packet_view_selection rather than rewriting the
  target rule.
- For insufficient_validation wrong-scope diagnoses, a supported update signal
  must name the consumed object, the invariant that should have been checked,
  the observed check, why that check covers a different scope, how the
  transaction controls or adversarially varies the consumed object, and the
  same-object causal outcome. Caller identity or a generic guard alone does not
  prove that payload/content/business-state validation was missing. If any of
  those links is unavailable, mark the signal insufficient and route the gap to
  evidence acquisition rather than rewriting the rule.
- For insufficient_validation plan diagnoses, use semantic-role view routing:
  conditions about consumed inputs/callbacks/unknown selectors need locating
  views such as trace_outline_view, critical_call_view, unknown_selector_view,
  operation_summary_view, and event_view; conditions about validation logic may
  need read_function_chunk after a concrete callee/function/evidence_id is
  located; conditions about outcome/invariants need semantic_state_delta_view,
  value_release_view, contribution_vs_payout_view, participant_net_delta_view,
  and state_change_view. Do not suggest a plan fix by saying "C1 should use X";
  describe the condition role that needs X.
- For protocol_accounting_exploitation reviews, diagnose whether the failure is
  candidate-chain breakage: C1 should locate one protocol-internal bookkeeping
  candidate, C2 should consume that same candidate, and C3 should show a
  candidate-specific abnormal outcome. If packet views omitted critical calls,
  call arguments, trace outline, semantic/state deltas, or source/function
  lookup needed to locate the candidate, prefer update_target="plan". If the
  rule accepted generic price/oracle movement, token semantics, reentrancy, or
  ordinary profit as accounting evidence, diagnose the rule boundary instead of
  adding negative-family-specific shortcuts.
- For protocol_accounting_exploitation outcome failures, do not require direct
  token/native value release when protocol_accounting_outcome_view or semantic
  state deltas show candidate-bound reward/share/vault/debt/collateral/staking/
  claim/liability impact. Conversely, do not accept ordinary deposit/withdraw/
  harvest/accounting mechanics, generic user profit, reentrancy/state-order
  abuse, insufficient validation, token semantics, market/oracle effects, or
  flash capital as PAE unless the same protocol-internal bookkeeping candidate
  remains the root cause.
- For protocol_accounting_exploitation FP on negative_other/other_attack,
  tightening or missing-exclusion signals are useful contrastive boundary
  signals when they identify which PAE requirement is absent or replaced. Do
  not mark them do-not-train merely because the target mechanism is not
  observed; reserve do-not-train mainly for unsupported FN broadening,
  packet-only limitations, runtime failures, or case-specific labels.
- should_update_rule must be true only when update_target == "rule".
- Runtime bug reviews must not enter RuleUpdater.

Feature-diagnosis guidance:
- For FN: if a positive target sample has answer=false or answer="uncertain"
  for a core condition, and that row has strong partial_match_features, this
  may indicate rule_too_narrow. Explain which partial_match_features are
  target-family-relevant and whether they should be incorporated into the rule.
  missing_required_features should identify what the condition currently
  requires but failed to observe. If partial features are generic exploit
  symptoms rather than target-family root-cause features, do not broaden.
- For FP: if a negative/non-target sample has answer=true for a core condition,
  inspect matched_features and boundary_notes. If matched_features are generic
  symptoms such as value extraction, unknown selector, flash loan, swap,
  callback, reentrancy, profit, or public entry call, diagnose
  weak_evidence_as_sufficient or rule_too_broad. missing_required_features in a
  true answer may indicate the current condition accepted insufficient evidence;
  use it to tighten the rule.
- For exclusions: if an exclusion wrongly fires on a positive FN, inspect its
  matched_features and boundary_notes to identify the misleading benign or
  authorized feature. If an exclusion fails to fire on a negative, inspect
  missing_required_features and contradicting_features to determine whether the
  exclusion is too narrow or whether this is truly non-normal behavior.
- observed_* feature_diagnosis fields must come from condition_feature_analysis
  or be concise abstractions of it; do not invent unrelated features.
- Keep feature_diagnosis abstract. Do not include concrete selectors,
  addresses, call ids, evidence ids, storage slots, token amounts, or tx hashes.

Return strict JSON:
{{
  "target_mechanism_observed": "yes|no|uncertain",
  "ground_truth_supported_by_packet": "yes|no|uncertain",
  "generic_symptoms_only": false,
  "counterfactual_boundary_risk": "low|medium|high",
  "error_type": "FP|FN|unknown",
  "update_target": "rule|plan|packet|runtime|none",
  "root_causes": [
    {{
      "category": "missing_evidence|missing_packet_view|bad_packet_view_selection|packet_evidence_too_coarse|bad_judge_question|rule_too_broad|rule_too_narrow|exclusion_too_broad|missing_exclusion|missing_condition|weak_evidence_as_sufficient|bad_semantic_condition|binding_semantic_failure|binding_route_failure|binding_runtime_failure|deferred_logic_structure_failure|runtime_emit_logic_bug|packet_build_or_load_failure|unknown",
      "failure_origin": "rule_semantic|plan_question_drift|evidence_route|packet|runtime|unavailable|unknown",
      "requested_operation": "semantic_rewrite|restore_question_from_rule|change_evidence_route|engineering_packet|engineering_runtime|none",
      "rule_semantic_status": "correct|incorrect|uncertain",
      "question_alignment_status": "aligned|drifted|uncertain",
      "semantic_impact": "blocking|non_blocking|uncertain",
      "description": "...",
      "affected_conditions": ["C1"],
      "suggested_fix": "..."
    }}
  ],
  "condition_diagnosis": [
    {{
      "condition_id": "C3",
      "observed_answer": false,
      "expected_for_correct_verdict": true,
      "diagnosis": "...",
      "is_rule_problem": true,
      "is_plan_problem": false,
      "is_packet_problem": false,
      "is_runtime_problem": false
    }}
  ],
  "feature_diagnosis": [
    {{
      "condition_id": "C2",
      "observed_matched_features": [],
      "observed_partial_match_features": [],
      "observed_missing_required_features": [],
      "observed_contradicting_features": [],
      "observed_boundary_notes": [],
      "feature_interpretation": "...",
      "update_implication": "broaden|tighten|add_exclusion|clarify_boundary|no_change",
      "rule_update_hint": "..."
    }}
  ],
  "update_signals": [
    {{
      "update_target": "rule|plan|packet|runtime|none",
      "condition_id": "C1",
      "direction": "broaden|tighten|refine|add_exclusion|narrow_exclusion|clarify_boundary|change_evidence_strategy|no_change",
      "strategy_operation": "none|replace_evidence_route|reorder_evidence_route|demote_default_to_followup|prune_evidence_route|change_verification_granularity|change_followup_budget|change_dependency_route",
      "condition_evidence_dependency": {{"role": "requires|provides|none", "capability_id": "abstract_snake_case_id", "capability_description": "abstract evidence capability required by this condition"}},
      "abstract_feature": "target-family semantic distinction, never a case identifier",
      "feature_axis": "stable semantic axis such as callback_payload_scope or accounting_proportionality",
      "wrong_scope_audit": {{
        "consumed_object": "abstract object consumed by the sensitive path",
        "required_invariant": "semantic invariant that should be validated",
        "observed_check": "check actually evidenced",
        "scope_mismatch": "why the observed check does not validate that invariant",
        "adversarial_control": "how the object is externally controlled or variable",
        "causal_outcome": "same-object incorrect protocol outcome"
      }},
      "correct_positive_compatibility": "how protected positive signals remain detectable",
      "correct_negative_compatibility": "how correct negative/guard boundaries remain intact",
      "cohort_support_refs": ["C1.cp.m.01"],
      "cohort_conflict_refs": [],
      "generalization_status": "supported|conflicted|insufficient",
      "rationale": "short contrastive rationale"
    }}
  ],
  "rule_patch_suggestion": {{
    "action": "none|rewrite_condition|add_condition|add_exclusion|narrow_exclusion|remove_condition|remove_exclusion|relax_condition|tighten_condition",
    "condition_id": "C3",
    "semantic_direction": "tighten|broaden|refine",
    "old_problem": "...",
    "proposed_semantic_change": "...",
    "rationale": "..."
  }},
  "plan_patch_suggestion": {{
    "action": "none|change_default_views|change_followup_views|replace_evidence_route|reorder_evidence_route|demote_default_to_followup|prune_evidence_route|change_verification_granularity|change_followup_budget|change_dependency_route",
    "condition_id": "...",
    "proposed_change": "...",
    "rationale": "..."
  }},
  "packet_patch_suggestion": {{
    "action": "none|add_view|improve_view|fix_truncation|add_summary",
    "view_name": "...",
    "proposed_change": "...",
    "rationale": "..."
  }},
  "runtime_patch_suggestion": {{
    "action": "none|fix_emit_logic|fix_confidence_handling|fix_evidence_resolution",
    "proposed_change": "...",
    "rationale": "..."
  }},
  "should_update_rule": true,
  "should_update_plan_strategy": false,
  "should_update_packet_builder": false,
  "should_fix_runtime": false,
  "must_not_change": [],
  "review_confidence": "low|medium|high",
  "review_note": "short summary"
}}

Compact reviewer_case:
{stable_json_dumps(case)}

Attack-family label for this task:
{target_label}

Raw ground truth label for this case:
{raw_label}
"""

    @staticmethod
    def _build_compact_prompt(case: Dict[str, Any]) -> str:
        raw_label = case.get("raw_ground_truth", case.get("ground_truth"))
        rule_digest = case.get("rule_digest") if isinstance(case.get("rule_digest"), dict) else {}
        raw_target_label = (
            case.get("target_label")
            or case.get("attack_label")
            or rule_digest.get("attack_label")
            or (rule_digest.get("metadata", {}) or {}).get("attack_label")
            or raw_label
        )
        target_label = normalize_attack_label(raw_target_label)
        negative_training_mode = normalize_negative_training_mode(
            case.get("negative_training_mode", "mixed")
        )
        negative_training_guidance = negative_training_mode_guidance(
            negative_training_mode
        )
        negative_case_guidance = negative_case_boundary_guidance(
            negative_training_mode
        )
        return f"""
You are an EvoTx transaction detection rule reviewer.

{REVIEWER_EVIDENCE_FIRST_GUIDANCE}

{_reviewer_label_specific_guidance(target_label)}

Diagnose why the rule produced a wrong prediction for the target attack family.
Use the compact reviewer_case. Do not request full packet evidence.

Routing:
- update_target="rule" only for semantic rule too broad/narrow, bad condition,
  missing exclusion, or too-broad exclusion.
- update_target="plan" for bad view selection or judge question strategy.
- update_target="packet" for missing/coarse packet views.
- update_target="runtime" for emit/aggregation bugs.
- {negative_case_guidance}
- Do not broaden rules from generic symptoms such as profit, flash loans, swaps,
  public entry calls, callbacks, unknown selectors, or value movement alone.
- negative_training_mode="{negative_training_mode}": {negative_training_guidance}
  Treat this as dataset coverage only; use case-level metadata for this case.

Use condition_table and condition_feature_analysis when present:
- FN core false/uncertain with target-specific partial features may imply broaden.
- FP core true with weak/generic matched features may imply tighten or exclusion.
- Exclusion mistakes should inspect matched_features and boundary_notes.
- feature_diagnosis observed_* fields must come from condition_feature_analysis
  or concise abstractions of it.
- Keep all feature/rule hints abstract; do not include concrete selectors,
  addresses, tx hashes, call ids, storage slots, evidence ids, or exact amounts.

Return strict JSON with this schema:
{{
  "target_mechanism_observed": "yes|no|uncertain",
  "ground_truth_supported_by_packet": "yes|no|uncertain",
  "generic_symptoms_only": false,
  "counterfactual_boundary_risk": "low|medium|high",
  "error_type": "FP|FN|unknown",
  "update_target": "rule|plan|packet|runtime|none",
  "root_causes": [
    {{
      "category": "missing_evidence|missing_packet_view|bad_packet_view_selection|packet_evidence_too_coarse|bad_judge_question|rule_too_broad|rule_too_narrow|exclusion_too_broad|missing_exclusion|missing_condition|weak_evidence_as_sufficient|bad_semantic_condition|binding_semantic_failure|binding_route_failure|binding_runtime_failure|deferred_logic_structure_failure|runtime_emit_logic_bug|packet_build_or_load_failure|unknown",
      "failure_origin": "rule_semantic|plan_question_drift|evidence_route|packet|runtime|unavailable|unknown",
      "requested_operation": "semantic_rewrite|restore_question_from_rule|change_evidence_route|engineering_packet|engineering_runtime|none",
      "rule_semantic_status": "correct|incorrect|uncertain",
      "question_alignment_status": "aligned|drifted|uncertain",
      "semantic_impact": "blocking|non_blocking|uncertain",
      "description": "...",
      "affected_conditions": ["C1"],
      "suggested_fix": "..."
    }}
  ],
  "condition_diagnosis": [
    {{
      "condition_id": "C1",
      "observed_answer": false,
      "expected_for_correct_verdict": true,
      "diagnosis": "...",
      "is_rule_problem": true,
      "is_plan_problem": false,
      "is_packet_problem": false,
      "is_runtime_problem": false
    }}
  ],
  "feature_diagnosis": [
    {{
      "condition_id": "C1",
      "observed_matched_features": [],
      "observed_partial_match_features": [],
      "observed_missing_required_features": [],
      "observed_contradicting_features": [],
      "observed_boundary_notes": [],
      "feature_interpretation": "...",
      "update_implication": "broaden|tighten|add_exclusion|clarify_boundary|no_change",
      "rule_update_hint": "..."
    }}
  ],
  "update_signals": [
    {{
      "update_target": "rule|plan|packet|runtime|none",
      "condition_id": "C1",
      "direction": "broaden|tighten|refine|add_exclusion|narrow_exclusion|clarify_boundary|change_evidence_strategy|no_change",
      "strategy_operation": "none|replace_evidence_route|reorder_evidence_route|demote_default_to_followup|prune_evidence_route|change_verification_granularity|change_followup_budget|change_dependency_route",
      "condition_evidence_dependency": {{"role": "requires|provides|none", "capability_id": "abstract_snake_case_id", "capability_description": "abstract evidence capability required by this condition"}},
      "abstract_feature": "target-family semantic distinction, never a case identifier",
      "feature_axis": "stable semantic axis such as callback_payload_scope or accounting_proportionality",
      "wrong_scope_audit": {{
        "consumed_object": "abstract object consumed by the sensitive path",
        "required_invariant": "semantic invariant that should be validated",
        "observed_check": "check actually evidenced",
        "scope_mismatch": "why the observed check does not validate that invariant",
        "adversarial_control": "how the object is externally controlled or variable",
        "causal_outcome": "same-object incorrect protocol outcome"
      }},
      "correct_positive_compatibility": "how protected positive signals remain detectable",
      "correct_negative_compatibility": "how correct negative/guard boundaries remain intact",
      "cohort_support_refs": ["C1.cp.m.01"],
      "cohort_conflict_refs": [],
      "generalization_status": "supported|conflicted|insufficient",
      "rationale": "short contrastive rationale"
    }}
  ],
  "rule_patch_suggestion": {{
    "action": "none|rewrite_condition|add_condition|add_exclusion|narrow_exclusion|remove_condition|remove_exclusion|relax_condition|tighten_condition",
    "condition_id": "",
    "semantic_direction": "tighten|broaden|refine",
    "old_problem": "",
    "proposed_semantic_change": "",
    "rationale": ""
  }},
  "plan_patch_suggestion": {{
    "action": "none|change_default_views|change_followup_views|replace_evidence_route|reorder_evidence_route|demote_default_to_followup|prune_evidence_route|change_verification_granularity|change_followup_budget|change_dependency_route",
    "condition_id": "",
    "proposed_change": "",
    "rationale": ""
  }},
  "packet_patch_suggestion": {{
    "action": "none|add_view|improve_view|fix_truncation|add_summary",
    "view_name": "",
    "proposed_change": "",
    "rationale": ""
  }},
  "runtime_patch_suggestion": {{
    "action": "none|fix_emit_logic|fix_confidence_handling|fix_evidence_resolution",
    "proposed_change": "",
    "rationale": ""
  }},
  "should_update_rule": true,
  "should_update_plan_strategy": false,
  "should_update_packet_builder": false,
  "should_fix_runtime": false,
  "must_not_change": [],
  "review_confidence": "low|medium|high",
  "review_note": "short summary"
}}

Compact reviewer_case:
{stable_json_dumps(case)}

Attack-family label:
{target_label}

Raw ground truth label:
{raw_label}
"""

    @staticmethod
    def _build_error_focused_prompt(focus_case: Dict[str, Any]) -> str:
        raw_label = focus_case.get("raw_ground_truth", focus_case.get("ground_truth"))
        target_label = normalize_attack_label(
            focus_case.get("target_label")
            or focus_case.get("attack_label")
            or raw_label
        )
        negative_training_mode = normalize_negative_training_mode(
            focus_case.get("negative_training_mode", "mixed")
        )
        negative_training_guidance = negative_training_mode_guidance(
            negative_training_mode
        )
        negative_case_guidance = negative_case_boundary_guidance(
            negative_training_mode
        )
        return f"""
You are an EvoTx transaction detection rule reviewer.

{REVIEWER_EVIDENCE_FIRST_GUIDANCE}

{_reviewer_label_specific_guidance(target_label)}

Analyze why the current evolving rule produced a wrong prediction. You receive
an error-focused reviewer packet, not the full packet evidence. This packet is
already compacted to prioritize the wrong verdict, relevant condition rows,
condition_feature_analysis, IV-stateful state summaries, and rule/plan context.

Use priority sections in this order:
1. error_focus and condition_error_table.
2. condition_feature_analysis, view_history, tool_history,
   active_missing_evidence, and stateful_binding_state in each condition row.
   missing_evidence_history is audit context only and must not revive resolved,
   stale, or non-actionable blockers.
3. aggregation_audit and plan_evidence_audit for runtime/plan attribution.
4. case_boundary_context, cohort_signal_summary, and rejected_update_memory to
   test whether a repair would break correct target positives or negatives and
   to refine a prior failed strategy without repeating its exact candidate.
5. rule_context and plan_context for the affected condition semantics.
6. evidence_limitations only to route plan/packet/source limitations correctly.

Routing rules:
- update_target="rule" only when the semantic rule is too broad/narrow, a
  condition definition is wrong, an exclusion is missing, or an exclusion is too
  broad.
- update_target="plan" for bad view selection, missing useful follow-up views,
  or judge question strategy.
- update_target="packet" for missing/coarse packet views or real packet trace
  truncation.
- update_target="runtime" for emit/aggregation bugs.
- Do not classify source_unavailable alone as a rule failure.
- available_packet_views and configured tools are a closed evidence universe.
  A desired fact that none of them can expose is insufficient/no update, not a
  request to invent a Packet capability. Packet is actionable only for a
  registered view missing from construction, actual packet/trace truncation, or
  Packet build/load failure. Existing-but-unused capability is Plan; an observed
  tool/runtime failure is Runtime.
- {negative_case_guidance}
- Do not broaden rules from generic symptoms such as profit, flash loans, swaps,
  callbacks, public entry calls, unknown selectors, or value movement alone.
- negative_training_mode="{negative_training_mode}": {negative_training_guidance}
  Treat this as dataset coverage only; use case-level metadata for this case.
- A repair signal is not supported merely because it explains this one error.
  It must also pass the case_boundary_context check: preserve correct-positive
  mechanism variants and correct-negative/guard target boundaries. When the
  boundary context is missing or incompatible, use generalization_status
  "insufficient" or "conflicted".
- rejected_update_memory is refinement feedback, not transaction evidence and
  not an eligibility gate. Use its failure reason and protected attribution to
  make a supported signal more specific. Do not downgrade a signal solely
  because its condition or direction resembles a prior rejection; exact
  duplicate candidates are removed deterministically downstream.

For insufficient_validation:
- Diagnose by semantic role, not fixed condition IDs: consumed object/state,
  validation/acceptance gap, causal incorrect outcome, and exclusions.
- For FN, check whether core false/uncertain rows missed a target-specific
  partial feature or whether an exclusion fired from a secondary mechanism.
- For FP, check whether true core rows accepted generic value extraction or
  another attack family without a causal validation gap.
- If iv_stateful_state is present, verify whether C1 object candidates, C2
  selected validation gap, and C3 causal outcome refer to the same object chain.
- A wrong-scope signal is supported only when wrong_scope_audit supplies all of
  consumed_object, required_invariant, observed_check, scope_mismatch,
  adversarial_control, and causal_outcome. Otherwise make it insufficient and
  request the missing object-specific evidence.
- Keep the validation-gap signal local to C2. If Access Control, Reentrancy,
  market, token, or accounting behavior independently replaces the proposed
  validation root cause, diagnose that at E1/E2 rather than embedding the
  replacement policy into C2.
- For access_control, if stateful_binding_state is present, verify that the
  selected authorization candidate, authorization gap, and protected effect
  retain the same actor/capability/target chain.
- For access_control, require an entry-parent/downstream-child relation only
  when they are different calls and trace evidence connects both. A concrete
  single-call candidate is not incomplete merely because it has no parent call.
- Treat access-control C1 as the chain locator. An unresolved authorization or
  entitlement question should produce a C2/E1 diagnosis or evidence request;
  it must not erase an otherwise concrete C1 candidate.
- The C1 locator must still be plausibly authority-bearing. Generic public
  swap/transfer routing, beneficiary gain, or value-flow asymmetry without a
  concrete protected capability/resource or authorization-relevant state/effect
  anchor is not an access-control candidate.
- For access_control authorization-gap reviews, preserve concrete authority
  provenance when observed, but do not treat a missing exhaustive provenance
  record as a separate semantic failure when the same-candidate boundary is
  already established by source or behavioral evidence.
- For an access_control FP on negative_other/other_attack, when another primary
  mechanism is positively supported and the authorization boundary is intact
  or absent, target the existing E2 replacement-root boundary. Target C2 only
  when its authorization-gap definition itself accepted evidence that was too
  weak; do not force every non-target FP into C2 tightening.
- For reentrancy, diagnose nested-path location at C1, candidate-local
  value/state effect at C2, and stale/intermediate-state, delayed-finalization,
  repeated-consumption, or phase-order causality at C3. Do not move a missing
  C3 causality witness into C2.
- For protocol_accounting_exploitation, diagnose candidate-chain breakage by
  semantic role rather than fixed condition IDs: one protocol-internal
  bookkeeping candidate should be located, the same candidate should support
  the accounting gap/mismatch, and the outcome should be candidate-specific.
  protocol_accounting_outcome_view and semantic state deltas may show
  reward/share/vault/debt/collateral/staking/claim/liability impact even when
  direct token/native value release is absent. Do not accept ordinary
  deposit/withdraw/harvest mechanics, generic profit, reentrancy/state-order
  abuse, insufficient validation, token semantics, market/oracle effects, or
  flash capital as PAE unless the same protocol-internal bookkeeping candidate
  remains the root cause.
- For protocol_accounting_exploitation FP on negative_other/other_attack,
  tightening or missing-exclusion signals are valid training signals when they
  identify which PAE requirement is absent or replaced. Do not mark them
  do-not-train merely because the target mechanism is not observed.

Feature diagnosis:
- observed_* fields must come from condition_feature_analysis or concise
  abstractions of it.
- Use matched_features for true local conditions, partial_match_features and
  missing_required_features for false/uncertain local conditions, and
  boundary_notes for target-vs-non-target risk.
- Keep feature and rule hints abstract. Do not include concrete selectors,
  addresses, tx hashes, call ids, storage slots, evidence ids, or exact amounts.

Return strict JSON with this schema:
{{
  "target_mechanism_observed": "yes|no|uncertain",
  "ground_truth_supported_by_packet": "yes|no|uncertain",
  "generic_symptoms_only": false,
  "counterfactual_boundary_risk": "low|medium|high",
  "error_type": "FP|FN|unknown",
  "update_target": "rule|plan|packet|runtime|none",
  "root_causes": [
    {{
      "category": "missing_evidence|missing_packet_view|bad_packet_view_selection|packet_evidence_too_coarse|bad_judge_question|rule_too_broad|rule_too_narrow|exclusion_too_broad|missing_exclusion|missing_condition|weak_evidence_as_sufficient|bad_semantic_condition|binding_semantic_failure|binding_route_failure|binding_runtime_failure|deferred_logic_structure_failure|runtime_emit_logic_bug|packet_build_or_load_failure|unknown",
      "failure_origin": "rule_semantic|plan_question_drift|evidence_route|packet|runtime|unavailable|unknown",
      "requested_operation": "semantic_rewrite|restore_question_from_rule|change_evidence_route|engineering_packet|engineering_runtime|none",
      "rule_semantic_status": "correct|incorrect|uncertain",
      "question_alignment_status": "aligned|drifted|uncertain",
      "semantic_impact": "blocking|non_blocking|uncertain",
      "description": "...",
      "affected_conditions": ["C1"],
      "suggested_fix": "..."
    }}
  ],
  "condition_diagnosis": [
    {{
      "condition_id": "C1",
      "observed_answer": false,
      "expected_for_correct_verdict": true,
      "diagnosis": "...",
      "is_rule_problem": true,
      "is_plan_problem": false,
      "is_packet_problem": false,
      "is_runtime_problem": false
    }}
  ],
  "feature_diagnosis": [
    {{
      "condition_id": "C1",
      "observed_matched_features": [],
      "observed_partial_match_features": [],
      "observed_missing_required_features": [],
      "observed_contradicting_features": [],
      "observed_boundary_notes": [],
      "feature_interpretation": "...",
      "update_implication": "broaden|tighten|add_exclusion|clarify_boundary|no_change",
      "rule_update_hint": "..."
    }}
  ],
  "update_signals": [
    {{
      "update_target": "rule|plan|packet|runtime|none",
      "condition_id": "C1",
      "direction": "broaden|tighten|refine|add_exclusion|narrow_exclusion|clarify_boundary|change_evidence_strategy|no_change",
      "strategy_operation": "none|replace_evidence_route|reorder_evidence_route|demote_default_to_followup|prune_evidence_route|change_verification_granularity|change_followup_budget|change_dependency_route",
      "condition_evidence_dependency": {{"role": "requires|provides|none", "capability_id": "abstract_snake_case_id", "capability_description": "abstract evidence capability required by this condition"}},
      "abstract_feature": "target-family semantic distinction, never a case identifier",
      "feature_axis": "stable semantic axis such as callback_payload_scope or accounting_proportionality",
      "wrong_scope_audit": {{
        "consumed_object": "abstract object consumed by the sensitive path",
        "required_invariant": "semantic invariant that should be validated",
        "observed_check": "check actually evidenced",
        "scope_mismatch": "why the observed check does not validate that invariant",
        "adversarial_control": "how the object is externally controlled or variable",
        "causal_outcome": "same-object incorrect protocol outcome"
      }},
      "correct_positive_compatibility": "how protected positive signals remain detectable",
      "correct_negative_compatibility": "how correct negative/guard boundaries remain intact",
      "cohort_support_refs": ["C1.cp.m.01"],
      "cohort_conflict_refs": [],
      "generalization_status": "supported|conflicted|insufficient",
      "rationale": "short contrastive rationale"
    }}
  ],
  "rule_patch_suggestion": {{
    "action": "none|rewrite_condition|add_condition|add_exclusion|narrow_exclusion|remove_condition|remove_exclusion|relax_condition|tighten_condition",
    "condition_id": "",
    "semantic_direction": "tighten|broaden|refine",
    "old_problem": "",
    "proposed_semantic_change": "",
    "rationale": ""
  }},
  "plan_patch_suggestion": {{
    "action": "none|change_default_views|change_followup_views|replace_evidence_route|reorder_evidence_route|demote_default_to_followup|prune_evidence_route|change_verification_granularity|change_followup_budget|change_dependency_route",
    "condition_id": "",
    "proposed_change": "",
    "rationale": ""
  }},
  "packet_patch_suggestion": {{
    "action": "none|add_view|improve_view|fix_truncation|add_summary",
    "view_name": "",
    "proposed_change": "",
    "rationale": ""
  }},
  "runtime_patch_suggestion": {{
    "action": "none|fix_emit_logic|fix_confidence_handling|fix_evidence_resolution",
    "proposed_change": "",
    "rationale": ""
  }},
  "should_update_rule": true,
  "should_update_plan_strategy": false,
  "should_update_packet_builder": false,
  "should_fix_runtime": false,
  "must_not_change": [],
  "review_confidence": "low|medium|high",
  "review_note": "short summary"
}}

Error-focused reviewer packet:
{stable_json_dumps(focus_case)}

Attack-family label:
{target_label}

Raw ground truth label:
{raw_label}
"""

    @staticmethod
    def _normalize_review(review: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
        root_causes = review.get("root_causes", [])
        if not isinstance(root_causes, list):
            root_causes = []
        root_causes = [
            cause for cause in root_causes if isinstance(cause, dict)
        ]
        condition_diagnosis = review.get("condition_diagnosis", [])
        if not isinstance(condition_diagnosis, list):
            condition_diagnosis = []
        feature_diagnosis = _normalize_feature_diagnosis(
            review.get("feature_diagnosis", [])
        )
        update_target = str(review.get("update_target") or "").strip().lower()
        if update_target not in {"rule", "plan", "packet", "runtime", "none"}:
            update_target = infer_update_target(case, root_causes)
        if _looks_like_runtime_emit_bug(case):
            update_target = "runtime"
            if not any(c.get("category") == "runtime_emit_logic_bug" for c in root_causes):
                root_causes.append({
                    "category": "runtime_emit_logic_bug",
                    "description": (
                        "All core conditions appear true and all exclusions appear false, "
                        "but the emitted verdict is not attack."
                    ),
                    "affected_conditions": [
                        row.get("condition_id", row.get("id"))
                        for row in case.get("condition_table", [])
                    ],
                    "suggested_fix": "Fix runtime emit logic or verdict aggregation, not the semantic rule.",
                })

        routing = _normalize_update_target_with_consistency(
            review,
            case,
            root_causes,
            condition_diagnosis,
            update_target,
        )
        update_target = routing["normalized_update_target"]
        rule_patch = _patch_suggestion(review.get("rule_patch_suggestion"), "none")
        plan_patch = _patch_suggestion(review.get("plan_patch_suggestion"), "none")
        packet_patch = _patch_suggestion(review.get("packet_patch_suggestion"), "none")
        runtime_patch = _patch_suggestion(review.get("runtime_patch_suggestion"), "none")
        update_signals = _normalize_update_signals(
            review.get("update_signals", []),
            case=case,
            default_update_target=update_target,
        )
        cohort_gate_enabled = bool(
            (case.get("cohort_signal_summary") or {}).get("schema_version")
        )
        boundary_context_enabled = bool(
            (case.get("case_boundary_context") or {}).get("schema_version")
        )
        supported_targets = {
            str(signal.get("update_target") or "")
            for signal in update_signals
            if signal.get("generalization_status") == "supported"
        }
        has_rule_root = _has_category(root_causes, RULE_CATEGORIES)
        has_plan_root = _has_category(root_causes, PLAN_CATEGORIES)
        has_packet_root = _has_category(root_causes, PACKET_CATEGORIES)
        has_runtime_root = _has_category(root_causes, RUNTIME_CATEGORIES)
        has_packet_diagnosis = any(
            bool(item.get("is_packet_problem"))
            for item in condition_diagnosis
            if isinstance(item, dict)
        )
        should_update_rule = bool(
            (
                update_target == "rule"
                and rule_patch.get("action") not in {"", "none"}
            )
            or (
                has_rule_root
                and any(bool(d.get("is_rule_problem")) for d in condition_diagnosis)
            )
        )
        should_update_plan = bool(
            (
                update_target == "plan"
                and plan_patch.get("action") not in {"", "none"}
            )
            or has_plan_root
        )
        should_update_packet = bool(
            packet_patch.get("action") not in {"", "none"}
            and (
                update_target == "packet"
                or has_packet_root
                or has_packet_diagnosis
                or bool(review.get("should_update_packet_builder", False))
            )
        )
        should_fix_runtime = bool(
            (
                update_target == "runtime"
                and runtime_patch.get("action") not in {"", "none"}
            )
            or has_runtime_root
        )
        if cohort_gate_enabled:
            should_update_rule = should_update_rule and "rule" in supported_targets
            should_update_plan = should_update_plan and "plan" in supported_targets
        target_mechanism_observed = _normalize_choice(
            review.get("target_mechanism_observed"),
            {"yes", "no", "uncertain"},
            "uncertain",
        )
        ground_truth_supported_by_packet = _normalize_choice(
            review.get("ground_truth_supported_by_packet"),
            {"yes", "no", "uncertain"},
            "uncertain",
        )
        counterfactual_boundary_risk = _normalize_choice(
            review.get("counterfactual_boundary_risk"),
            {"low", "medium", "high"},
            "medium",
        )
        normalized_review = {
            "target_mechanism_observed": target_mechanism_observed,
            "ground_truth_supported_by_packet": ground_truth_supported_by_packet,
            "generic_symptoms_only": bool(review.get("generic_symptoms_only", False)),
            "counterfactual_boundary_risk": counterfactual_boundary_risk,
            "error_type": review.get("error_type") or infer_error_type(case),
            "update_target": update_target,
            "root_causes": root_causes,
            "condition_diagnosis": condition_diagnosis,
            "feature_diagnosis": feature_diagnosis,
            "update_signals": update_signals,
            "generalization_gate": {
                "enabled": cohort_gate_enabled,
                "case_boundary_context_enabled": boundary_context_enabled,
                "supported_targets": sorted(supported_targets),
                "supported_signal_count": sum(
                    1
                    for signal in update_signals
                    if signal.get("generalization_status") == "supported"
                ),
                "conflicted_signal_count": sum(
                    1
                    for signal in update_signals
                    if signal.get("generalization_status") == "conflicted"
                ),
                "insufficient_signal_count": sum(
                    1
                    for signal in update_signals
                    if signal.get("generalization_status") == "insufficient"
                ),
            },
            "rule_patch_suggestion": rule_patch,
            "plan_patch_suggestion": plan_patch,
            "packet_patch_suggestion": packet_patch,
            "runtime_patch_suggestion": runtime_patch,
            "should_update_rule": should_update_rule,
            "should_update_plan_strategy": should_update_plan,
            "should_update_packet_builder": should_update_packet,
            "should_fix_runtime": should_fix_runtime,
            "must_not_change": list(review.get("must_not_change", []) or []),
            "review_confidence": _normalize_confidence(review.get("review_confidence", "medium")),
            "review_note": str(review.get("review_note", "")),
            "routing_normalization": routing,
            "raw_ground_truth": case.get("raw_ground_truth", case.get("ground_truth")),
            "sample_role": case.get("sample_role", ""),
            "negative_kind": case.get("negative_kind", ""),
            "contrastive_supervision": dict(
                case.get("contrastive_supervision") or {}
            ),
            "negative_training_mode": case.get("negative_training_mode", "mixed"),
            "case_boundary_context_available": boundary_context_enabled,
            "target_label": case.get("target_label", ""),
            "tx_hash": case.get("tx_hash") or (case.get("transaction", {}) or {}).get("tx_hash"),
            "ground_truth": case.get("ground_truth"),
            "predicted_verdict": case.get("predicted_verdict")
            or _finding_summary(case).get("verdict"),
        }
        return apply_diagnosis_routing(normalized_review, case)

    @staticmethod
    def _fallback_review(
        case: Dict[str, Any],
        note: str,
        *,
        review_execution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        update_target = "runtime" if _looks_like_runtime_emit_bug(case) else "none"
        fallback = {
            "target_mechanism_observed": "uncertain",
            "ground_truth_supported_by_packet": "uncertain",
            "generic_symptoms_only": False,
            "counterfactual_boundary_risk": "medium",
            "error_type": infer_error_type(case),
            "update_target": update_target,
            "root_causes": [
                {
                    "category": "runtime_emit_logic_bug" if update_target == "runtime" else "unknown",
                    "description": note,
                    "affected_conditions": [],
                    "suggested_fix": (
                        "Inspect the execution trace manually and update the rule "
                        "only with evidence-backed changes."
                    ),
                }
            ],
            "condition_diagnosis": [],
            "feature_diagnosis": [],
            "update_signals": [],
            "generalization_gate": {
                "enabled": bool(
                    (case.get("cohort_signal_summary") or {}).get("schema_version")
                ),
                "supported_targets": [],
                "supported_signal_count": 0,
            },
            "rule_patch_suggestion": _patch_suggestion(None, "none"),
            "plan_patch_suggestion": _patch_suggestion(None, "none"),
            "packet_patch_suggestion": _patch_suggestion(None, "none"),
            "runtime_patch_suggestion": _patch_suggestion(
                {"action": "fix_emit_logic"} if update_target == "runtime" else None,
                "none",
            ),
            "should_update_rule": False,
            "should_update_plan_strategy": False,
            "should_update_packet_builder": False,
            "should_fix_runtime": update_target == "runtime",
            "must_not_change": [],
            "review_confidence": "low",
            "review_note": note,
            "raw_ground_truth": case.get("raw_ground_truth", case.get("ground_truth")),
            "sample_role": case.get("sample_role", ""),
            "negative_kind": case.get("negative_kind", ""),
            "contrastive_supervision": dict(
                case.get("contrastive_supervision") or {}
            ),
            "negative_training_mode": case.get("negative_training_mode", "mixed"),
            "case_boundary_context_available": bool(
                (case.get("case_boundary_context") or {}).get("schema_version")
            ),
            "target_label": case.get("target_label", ""),
            "tx_hash": case.get("tx_hash") or (case.get("transaction", {}) or {}).get("tx_hash"),
            "ground_truth": case.get("ground_truth"),
            "predicted_verdict": case.get("predicted_verdict")
            or _finding_summary(case).get("verdict"),
        }
        if review_execution:
            fallback["review_execution"] = dict(review_execution)
        return apply_diagnosis_routing(fallback, case)


def infer_error_type(case: Dict[str, Any], default: Optional[str] = "unknown") -> str:
    pred = case.get("predicted_verdict") or _finding_summary(case).get("verdict") or get_finding(case).get("verdict")
    gold = case.get("ground_truth") or get_ground_truth(case)
    if gold == "attack" and pred != "attack":
        return "FN"
    if gold == "benign" and pred == "attack":
        return "FP"
    return default or "unknown"


def infer_update_target(case: Dict[str, Any], root_causes: list[Dict[str, Any]]) -> str:
    if _looks_like_runtime_emit_bug(case):
        return "runtime"
    categories = {str(cause.get("category", "")).strip() for cause in root_causes}
    if "deferred_logic_structure_failure" in categories:
        return "none"
    if categories & {"runtime_emit_logic_bug"}:
        return "runtime"
    if categories & {"packet_build_or_load_failure", "packet_evidence_too_coarse"}:
        return "packet"
    bad_question_origins = {
        str(
            cause.get("failure_origin")
            or cause.get("question_failure_kind")
            or ""
        ).strip().lower()
        for cause in root_causes
        if str(cause.get("category") or "").strip().lower()
        == "bad_judge_question"
    }
    if bad_question_origins & {"rule", "rule_semantic", "semantic"}:
        return "rule"
    if bad_question_origins & {
        "plan_question_drift",
        "judge_question_drift",
        "question_drift",
        "evidence_route",
        "plan_evidence_route",
        "plan",
    }:
        return "plan"
    if categories & {"missing_packet_view", "bad_packet_view_selection"}:
        return "plan"
    if categories & RULE_CATEGORIES:
        return "rule"
    return "none"


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
PLAN_CATEGORIES = {
    "missing_packet_view",
    "bad_packet_view_selection",
    "binding_route_failure",
}
PACKET_CATEGORIES = {
    "packet_build_or_load_failure",
    "packet_evidence_too_coarse",
}
RUNTIME_CATEGORIES = {
    "runtime_emit_logic_bug",
    "binding_runtime_failure",
}


def _normalize_review_compact_mode(value: Any) -> str:
    text = str(value or "error-focused").strip().lower().replace("_", "-")
    if text in {"legacy", "none", "off"}:
        return "legacy"
    if text in {"auto", "error-focused", "errorfocused", "focused"}:
        return "error-focused"
    return "error-focused"


def _build_error_focused_reviewer_case(case: Dict[str, Any]) -> Dict[str, Any]:
    rule_digest = dict(case.get("rule_digest") or {})
    plan_digest = dict(case.get("plan_digest") or {})
    finding_summary = dict(case.get("finding_summary") or {})
    evidence_digest = dict(case.get("evidence_digest") or {})
    rows = [
        row for row in list(case.get("condition_table", []) or [])
        if isinstance(row, dict)
    ]
    error_type = str(case.get("error_type") or infer_error_type(case) or "unknown")
    condition_rows_all = [
        _error_focused_condition_row(row, error_type=error_type)
        for row in rows
    ]
    relevant_ids = _relevant_condition_ids(condition_rows_all)
    condition_rows = _select_error_focused_condition_rows(
        condition_rows_all,
        relevant_ids,
    )
    return {
        "schema_version": "evotx.error_focused_reviewer_case.v1",
        "compact_strategy": "error_focused_v1",
        "negative_training_mode": case.get("negative_training_mode", "mixed"),
        "case_header": {
            "case_id": case.get("case_id", ""),
            "tx_hash": case.get("tx_hash", ""),
            "chain": case.get("chain", ""),
            "attack_label": case.get("attack_label", ""),
            "target_label": case.get("target_label") or case.get("attack_label", ""),
            "ground_truth": case.get("ground_truth"),
            "raw_ground_truth": case.get("raw_ground_truth"),
            "predicted_verdict": case.get("predicted_verdict"),
            "error_type": error_type,
            "sample_role": case.get("sample_role", ""),
            "negative_kind": case.get("negative_kind", ""),
            "contrastive_supervision": dict(
                case.get("contrastive_supervision") or {}
            ),
            "negative_training_mode": case.get("negative_training_mode", "mixed"),
        },
        "error_focus": _error_focus_summary(
            rows=condition_rows_all,
            finding_summary=finding_summary,
            plan_digest=plan_digest,
            error_type=error_type,
        ),
        "aggregation_audit": _compact_aggregation_audit(finding_summary),
        "cross_condition_fact_consistency_audit": copy.deepcopy(
            case.get("cross_condition_fact_consistency_audit", {})
        ),
        "condition_error_table": condition_rows,
        "rule_context": _error_focused_rule_context(rule_digest, relevant_ids),
        "plan_context": _error_focused_plan_context(plan_digest, relevant_ids),
        "evidence_limitations": {
            "missing_evidence_actionability_summary": evidence_digest.get(
                "missing_evidence_actionability_summary",
                {},
            ),
            "missing_evidence_origins": [
                dict(item)
                for item in list(
                    evidence_digest.get("missing_evidence_origins", [])
                    or []
                )[:16]
                if isinstance(item, dict)
            ],
            "missing_evidence_origin_history": [
                dict(item)
                for item in list(
                    evidence_digest.get("missing_evidence_origin_history", [])
                    or []
                )[:16]
                if isinstance(item, dict)
            ],
            "unresolved_or_missing_evidence": _list_prefix(
                evidence_digest.get("unresolved_or_missing_evidence"),
                8,
            ),
            "top_supporting_evidence_ids": _list_prefix(
                evidence_digest.get("top_supporting_evidence_ids"),
                8,
            ),
            "evidence_adequacy_view": _truncate_text(
                stable_json_dumps(evidence_digest.get("evidence_adequacy_view", {})),
                900,
            ),
            "available_packet_views": _available_packet_views(evidence_digest),
        },
        "cohort_signal_summary": _compact_cohort_signal_summary(
            case.get("cohort_signal_summary", {}),
            relevant_ids,
        ),
        "case_boundary_context": _compact_case_boundary_context(
            case.get("case_boundary_context", {}),
            relevant_ids,
        ),
        "rejected_update_memory": _compact_rejected_update_memory(
            case.get("rejected_update_memory", {}),
            relevant_ids,
        ),
        "plan_evidence_audit": _compact_plan_evidence_audit(
            case.get("plan_evidence_audit", {}),
            relevant_ids,
            error_type=error_type,
        ),
        "review_instruction": (
            "Preserve error-relevant condition rows, latest observations, and "
            "source/code observations. Use plan_evidence_audit to separate an "
            "untried evidence strategy from one that was tried and failed."
        ),
    }


def _select_error_focused_condition_rows(
    rows: list[Dict[str, Any]],
    relevant_ids: set[str],
) -> list[Dict[str, Any]]:
    """Keep reviewer rows centered on the actual error boundary."""
    selected = [
        row
        for row in rows
        if row.get("relevance") != "context"
    ]
    state_context = [
        row
        for row in rows
        if row.get("relevance") == "context"
        and _condition_row_has_stateful_signal(row)
    ][:2]
    selected.extend(state_context)
    return selected or rows[:3]


def _condition_row_has_stateful_signal(row: Dict[str, Any]) -> bool:
    state = row.get("stateful_binding_state")
    if not isinstance(state, dict):
        return False
    return bool(state.get("state_output") or state.get("state_input"))


def _compact_cohort_signal_summary(
    summary: Dict[str, Any],
    relevant_ids: list[str],
) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    relevant = set(str(item) for item in relevant_ids if str(item))
    conditions = {
        condition_id: copy.deepcopy(value)
        for condition_id, value in dict(summary.get("conditions") or {}).items()
        if not relevant or condition_id in relevant
    }
    feature_budget = _budget_reviewer_cohort_features(
        conditions,
        max_chars=REVIEW_COHORT_FEATURE_BUDGET_CHARS,
    )
    return {
        "schema_version": summary.get("schema_version", ""),
        "counts": dict(summary.get("counts") or {}),
        "conditions": conditions,
        "protected_positive_signals": [
            item
            for item in list(summary.get("protected_positive_signals", []) or [])
            if not relevant or str(item.get("condition_id") or "") in relevant
        ],
        "protected_boundary_signals": [
            item
            for item in list(summary.get("protected_boundary_signals", []) or [])
            if not relevant or str(item.get("condition_id") or "") in relevant
        ],
        "feature_observation_budget": feature_budget,
        "semantic_clustering": summary.get(
            "semantic_clustering",
            "deferred_to_reviewer_and_updater",
        ),
    }


def _compact_case_boundary_context(
    context: Dict[str, Any],
    relevant_ids: list[str],
) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    relevant = set(str(item) for item in relevant_ids if str(item))
    boundaries = {}
    for condition_id, value in dict(context.get("condition_boundaries") or {}).items():
        if relevant and condition_id not in relevant:
            continue
        if not isinstance(value, dict):
            continue
        copied = copy.deepcopy(value)
        groups = dict(copied.get("preserved_groups") or {})
        compact_groups = {}
        for group_name, group in groups.items():
            if not isinstance(group, dict):
                continue
            compact_groups[group_name] = _compact_case_boundary_group(group)
        copied["preserved_groups"] = compact_groups
        copied["observed_successful_followup_views"] = _list_prefix(
            copied.get("observed_successful_followup_views"),
            5,
        )
        boundaries[condition_id] = copied
    return {
        "schema_version": context.get("schema_version", ""),
        "source": context.get("source", ""),
        "counts": dict(context.get("counts") or {}),
        "protected_positive_signals": [
            item
            for item in list(context.get("protected_positive_signals", []) or [])
            if not relevant or str(item.get("condition_id") or "") in relevant
        ][:12],
        "protected_boundary_signals": [
            item
            for item in list(context.get("protected_boundary_signals", []) or [])
            if not relevant or str(item.get("condition_id") or "") in relevant
        ][:12],
        "condition_boundaries": boundaries,
        "preservation_policy": list(context.get("preservation_policy") or [])[:5],
        "contains_transaction_identifiers": bool(
            context.get("contains_transaction_identifiers", False)
        ),
    }


def _compact_rejected_update_memory(
    memory: Dict[str, Any],
    relevant_ids: list[str],
) -> Dict[str, Any]:
    compact = normalize_rejected_update_memory(memory, max_entries=12)
    relevant = {str(item).upper().strip() for item in relevant_ids if str(item)}
    entries = []
    for entry in list(compact.get("entries") or []):
        condition_id = str(entry.get("condition_id") or "").upper().strip()
        if relevant and condition_id and condition_id not in relevant:
            continue
        entries.append(entry)
        if len(entries) >= 8:
            break
    compact["entries"] = entries
    compact["preflight_gaps"] = [
        dict(item)
        for item in list(compact.get("preflight_gaps") or [])[:6]
        if isinstance(item, dict)
        and (
            not relevant
            or bool(
                relevant
                & {
                    str(value or "").strip().upper()
                    for value in list(item.get("uncovered_blocker_ids") or [])
                }
            )
        )
    ]
    compact["max_entries"] = 8
    return compact


def _compact_case_boundary_group(group: Dict[str, Any]) -> Dict[str, Any]:
    features = {}
    for feature_type, observations in dict(group.get("features") or {}).items():
        features[feature_type] = _list_prefix(
            observations,
            REVIEW_CASE_BOUNDARY_FEATURES_PER_TYPE,
        )
    return {
        "role": group.get("role", ""),
        "case_count": group.get("case_count", 0),
        "answer_counts": dict(group.get("answer_counts") or {}),
        "confidence_counts": dict(group.get("confidence_counts") or {}),
        "features": {key: value for key, value in features.items() if value},
        "views": _list_prefix(
            group.get("views"),
            REVIEW_CASE_BOUNDARY_VIEW_LIMIT,
        ),
        "tool_statuses": _list_prefix(
            group.get("tool_statuses"),
            REVIEW_CASE_BOUNDARY_TOOL_LIMIT,
        ),
    }


def _budget_reviewer_cohort_features(
    conditions: Dict[str, Any],
    *,
    max_chars: int,
) -> Dict[str, Any]:
    buckets: list[tuple[list[Any], list[Any]]] = []
    available = 0
    for condition in conditions.values():
        for group in dict(condition.get("groups") or {}).values():
            group["views"] = list(group.get("views", []) or [])[
                :REVIEW_CASE_BOUNDARY_VIEW_LIMIT
            ]
            group["tool_statuses"] = list(group.get("tool_statuses", []) or [])[
                :REVIEW_CASE_BOUNDARY_TOOL_LIMIT
            ]
            for observations in dict(group.get("features") or {}).values():
                source = list(observations or [])
                available += len(source)
                observations.clear()
                buckets.append((source, observations))

    used_chars = 0
    included = 0
    item_index = 0
    while True:
        progressed = False
        for source, destination in buckets:
            if item_index >= len(source):
                continue
            progressed = True
            item = source[item_index]
            item_chars = len(stable_json_dumps(item))
            if used_chars + item_chars > max(0, int(max_chars)):
                continue
            destination.append(item)
            used_chars += item_chars
            included += 1
        if not progressed:
            break
        item_index += 1

    for condition in conditions.values():
        for group in dict(condition.get("groups") or {}).values():
            features = dict(group.get("features") or {})
            group["features"] = {
                key: observations
                for key, observations in features.items()
                if observations
            }

    return {
        "max_chars": int(max_chars),
        "max_estimated_tokens": (int(max_chars) + 3) // 4,
        "used_chars": used_chars,
        "used_estimated_tokens": (used_chars + 3) // 4,
        "available_count": available,
        "included_count": included,
        "omitted_count": max(0, available - included),
        "truncated": included < available,
        "priority": "feature_observations_before_views_and_tool_statuses",
    }


def _compact_aggregation_audit(finding_summary: Dict[str, Any]) -> Dict[str, Any]:
    aggregation = dict(finding_summary.get("verdict_aggregation") or {})
    dynamic = dict(aggregation.get("dynamic_aggregation") or {})
    decision = dict(dynamic.get("decision") or {})
    followup = dict(dynamic.get("followup_round") or {})
    if not aggregation and not dynamic:
        return {}
    return {
        "deterministic": {
            "policy": aggregation.get("policy", ""),
            "verdict": dynamic.get("previous_verdict")
            or finding_summary.get("verdict"),
            "should_emit": aggregation.get("should_emit"),
            "has_uncertain": aggregation.get("has_uncertain"),
            "core_blocking_uncertain": _list_prefix(
                aggregation.get("core_blocking_uncertain"),
                5,
            ),
            "decisive_core_false": _list_prefix(
                aggregation.get("decisive_core_false"),
                5,
            ),
            "decisive_exclusion_true": _list_prefix(
                aggregation.get("decisive_exclusion_true"),
                5,
            ),
        },
        "dynamic": {
            "enabled": bool(dynamic.get("enabled")),
            "applied": bool(dynamic.get("applied")),
            "aggregation_round": dynamic.get("aggregation_round"),
            "previous_verdict": dynamic.get("previous_verdict"),
            "final_verdict": dynamic.get("final_verdict"),
            "verdict": decision.get("verdict"),
            "confidence": decision.get("confidence"),
            "override": bool(decision.get("override")),
            "override_basis": decision.get("override_basis", ""),
            "overridden_conditions": _list_prefix(
                decision.get("overridden_conditions"),
                5,
            ),
            "failed_condition_assessment": _list_prefix(
                decision.get("failed_condition_assessment"),
                5,
            ),
            "cross_condition_support": _list_prefix(
                decision.get("cross_condition_support"),
                5,
            ),
            "reason": _truncate_text(decision.get("reason", ""), 900),
            "remaining_risk": _truncate_text(
                decision.get("remaining_risk", ""),
                600,
            ),
        },
        "followup": {
            "completed": bool(followup.get("completed")),
            "requested_count": int(followup.get("requested_count", 0) or 0),
            "accepted_count": int(followup.get("accepted_count", 0) or 0),
            "rerun_count": int(followup.get("rerun_count", 0) or 0),
            "should_emit_before": followup.get("should_emit_before"),
            "should_emit_after": followup.get("should_emit_after"),
            "hard_verdict_after": followup.get("hard_verdict_after"),
            "judge_reruns": _list_prefix(followup.get("judge_reruns"), 5),
            "errors": _list_prefix(followup.get("errors"), 5),
        },
    }


def _error_focus_summary(
    *,
    rows: list[Dict[str, Any]],
    finding_summary: Dict[str, Any],
    plan_digest: Dict[str, Any],
    error_type: str,
) -> Dict[str, Any]:
    return {
        "error_type": error_type,
        "verdict": finding_summary.get("verdict"),
        "confidence": finding_summary.get("confidence"),
        "verdict_reason": _truncate_text(finding_summary.get("verdict_reason", ""), 500),
        "emit_logic": plan_digest.get("emit_logic", ""),
        "blocking_core_condition_ids": [
            row.get("condition_id")
            for row in rows
            if row.get("relevance") in {"blocking_core_for_fn", "uncertain_core_for_fn"}
        ],
        "fired_exclusion_condition_ids": [
            row.get("condition_id")
            for row in rows
            if row.get("relevance") == "fired_exclusion_for_fn"
        ],
        "fp_supporting_core_condition_ids": [
            row.get("condition_id")
            for row in rows
            if row.get("relevance") == "supporting_core_for_fp"
        ],
        "missing_or_uncertain_exclusion_condition_ids": [
            row.get("condition_id")
            for row in rows
            if row.get("relevance") in {
                "missing_exclusion_for_fp",
                "uncertain_exclusion_for_fp",
            }
        ],
        "uncertain_condition_ids": [
            row.get("condition_id")
            for row in rows
            if isinstance(row.get("answer"), str)
        ],
        "condition_relevance_summary": [
            {
                "condition_id": row.get("condition_id"),
                "answer": row.get("answer"),
                "relevance": row.get("relevance"),
                "error_hint": row.get("error_hint", ""),
            }
            for row in rows
            if row.get("relevance") != "context"
        ],
    }


def _error_focused_condition_row(
    row: Dict[str, Any],
    *,
    error_type: str,
) -> Dict[str, Any]:
    relevance = _condition_relevance(row, error_type=error_type)
    stateful_binding_state = _review_stateful_state(row)
    state_prompt_role = str(
        stateful_binding_state.get("state_prompt_role") or ""
    )
    missing_evidence_history = _review_missing_evidence_history(
        row.get("all_missing_evidence_debug", [])
    )
    return {
        "id": row.get("id", ""),
        "condition_id": row.get("condition_id") or row.get("id", ""),
        "is_exclusion": bool(row.get("is_exclusion")),
        "relevance": relevance,
        "expected_answer": row.get("expected_answer", True),
        "answer": row.get("answer"),
        "first_round_answer": row.get("first_round_answer"),
        "final_answer": row.get("final_answer"),
        "followup_changed_answer": bool(row.get("followup_changed_answer")),
        "confidence": row.get("confidence", "medium"),
        "error_hint": row.get("error_hint", ""),
        "question": _truncate_text(row.get("question", ""), 420),
        "reason_short": _truncate_text(row.get("reason_short", ""), 450),
        "condition_feature_analysis": _review_feature_analysis(
            row.get("condition_feature_analysis", {}),
            relevance=relevance,
        ),
        "stateful_binding_state": stateful_binding_state,
        "iv_stateful_state": (
            stateful_binding_state if state_prompt_role.startswith("iv_") else {}
        ),
        "supporting_evidence_ids": _list_prefix(row.get("supporting_evidence_ids"), 4),
        "contradicting_evidence_ids": _list_prefix(row.get("contradicting_evidence_ids"), 4),
        "missing_evidence": _list_prefix(row.get("missing_evidence"), 4),
        "tool_request_count": row.get("tool_request_count", 0),
        "tool_call_count": row.get("tool_call_count", 0),
        "view_history": _review_view_history(row.get("view_render_metadata", [])),
        "tool_history": _review_tool_history(row.get("tool_observations", [])),
        "active_missing_evidence": [
            dict(item)
            for item in missing_evidence_history
            if _missing_evidence_is_active(item)
        ],
        "missing_evidence_history": missing_evidence_history,
        "near_miss_escalated": bool(row.get("near_miss_escalated")),
    }


def _review_view_history(value: Any) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    items = [item for item in list(value or []) if isinstance(item, dict)]
    selected_indexes: list[int] = []
    for index, item in enumerate(items):
        if (
            item.get("prompt_render_truncated")
            or item.get("render_truncated")
            or item.get("truncated")
            or item.get("packet_trace_truncated")
            or item.get("packet_truncated")
            or item.get("near_miss_escalation")
        ):
            selected_indexes.append(index)
    selected_indexes.extend(range(max(0, len(items) - 2), len(items)))
    selected_indexes = _dedupe_ints(selected_indexes)[:5]
    for index in selected_indexes:
        item = items[index]
        if not isinstance(item, dict):
            continue
        out.append({
            "round": item.get("round"),
            "view": item.get("view", ""),
            "render_role": item.get("render_role", item.get("role", "")),
            "render_mode": item.get("render_mode", item.get("mode", "")),
            "rows_total": item.get("rows_total"),
            "rows_rendered": item.get("rows_rendered"),
            "chars_rendered": item.get("chars_rendered", item.get("rendered_chars")),
            "prompt_render_truncated": bool(
                item.get("prompt_render_truncated")
                or item.get("render_truncated")
                or item.get("truncated")
            ),
            "packet_trace_truncated": bool(
                item.get("packet_trace_truncated")
                or item.get("packet_truncated")
            ),
            "truncation_reason": _truncate_text(item.get("truncation_reason", ""), 180),
            "near_miss_escalation": bool(item.get("near_miss_escalation")),
        })
    return out


def _review_tool_history(value: Any) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    safe_summary_keys = (
        "row_count",
        "rows_returned",
        "matched",
        "snippet_count",
        "source_status",
        "resolution_status",
        "cache_hit",
        "truncated",
    )
    items = list(value or [])
    source_indexes = [
        index
        for index, item in enumerate(items)
        if isinstance(item, dict)
        and isinstance(item.get("source_evidence"), dict)
        and item.get("source_evidence")
    ]
    problem_indexes = [
        index
        for index, item in enumerate(items)
        if isinstance(item, dict)
        and str(item.get("tool_status") or "").lower()
        not in {"", "ok", "success"}
    ]
    latest_indexes = list(range(max(0, len(items) - 2), len(items)))
    selected_indexes = _dedupe_ints(
        [*source_indexes[-2:], *problem_indexes[-3:], *latest_indexes]
    )[:6]
    for index in selected_indexes:
        item = items[index]
        if not isinstance(item, dict):
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        safe_summary = {
            key: summary.get(key)
            for key in safe_summary_keys
            if key in summary
        }
        history_item = {
            "round": item.get("round"),
            "tool": item.get("tool", ""),
            "tool_status": item.get("tool_status", ""),
            "requested_view": args.get("view", ""),
            "returned_evidence_count": len(
                list(item.get("returned_evidence_ids", []) or [])
            ),
            "validation_error": _truncate_text(item.get("validation_error", ""), 220),
            "summary_signals": safe_summary,
        }
        source_evidence = item.get("source_evidence")
        if isinstance(source_evidence, dict) and source_evidence:
            history_item["source_evidence"] = _review_source_evidence(source_evidence)
        out.append(history_item)
    return out


def _review_source_evidence(value: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve source/code observations, but keep them reviewer-sized."""
    snippets = []
    for raw in list(value.get("snippets") or [])[:2]:
        if not isinstance(raw, dict):
            continue
        snippets.append({
            "evidence_id": raw.get("evidence_id", ""),
            "path": raw.get("path", ""),
            "start_line": raw.get("start_line"),
            "end_line": raw.get("end_line"),
            "matched_key": raw.get("matched_key", ""),
            "source_code": _truncate_text(
                raw.get("source_code", ""),
                REVIEW_SOURCE_SNIPPET_CHARS,
            ),
            "truncated": bool(raw.get("truncated")),
        })
    compact = {"snippets": snippets}
    removed_counts: dict[str, int] = {}
    return _limit_value(
        compact,
        max_string=REVIEW_SOURCE_SNIPPET_CHARS,
        max_list=2,
        path="reviewer_case.source_evidence",
        removed_counts=removed_counts,
    )


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _review_missing_evidence_history(value: Any) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for item in list(value or [])[:8]:
        if not isinstance(item, dict):
            continue
        out.append({
            "round": item.get("round"),
            "text": _truncate_text(item.get("text", ""), 300),
            "category": item.get("category", ""),
            "status": item.get("status", ""),
            "blocking": bool(item.get("blocking")),
            "resolved_by": _truncate_text(item.get("resolved_by", ""), 160),
        })
    return out


def _missing_evidence_is_active(item: Dict[str, Any]) -> bool:
    return str(item.get("status") or "open").strip().lower() not in {
        "resolved",
        "stale",
        "non_actionable",
    }


def _compact_plan_evidence_audit(
    audit: Dict[str, Any],
    relevant_ids: list[str],
    *,
    error_type: str = "unknown",
) -> Dict[str, Any]:
    if not isinstance(audit, dict):
        return {}
    relevant = {str(item) for item in relevant_ids if str(item)}
    return {
        "schema_version": audit.get("schema_version", ""),
        "case_groups": dict(audit.get("case_groups") or {}),
        "conditions": {
            condition_id: _compact_plan_audit_condition(
                condition,
                error_type=error_type,
            )
            for condition_id, condition in dict(audit.get("conditions") or {}).items()
            if not relevant or condition_id in relevant
        },
        "contains_transaction_identifiers": bool(
            audit.get("contains_transaction_identifiers", False)
        ),
        "interpretation": audit.get("interpretation", ""),
    }


def _compact_plan_audit_condition(
    condition: Any,
    *,
    error_type: str,
) -> Dict[str, Any]:
    if not isinstance(condition, dict):
        return {}
    groups: Dict[str, Any] = {}
    for group_name, group in dict(condition.get("groups") or {}).items():
        if not isinstance(group, dict):
            continue
        keep_path_details = _plan_audit_group_relevant_to_error(
            group_name,
            error_type=error_type,
        )
        groups[group_name] = {
            "case_count": group.get("case_count", 0),
            "initial_answer_counts": dict(group.get("initial_answer_counts") or {}),
            "final_answer_counts": dict(group.get("final_answer_counts") or {}),
            "answer_transitions": dict(group.get("answer_transitions") or {}),
            "followup_changed_answer_count": group.get(
                "followup_changed_answer_count",
                0,
            ),
            "near_miss_escalated_count": group.get(
                "near_miss_escalated_count",
                0,
            ),
            "views": _list_prefix(group.get("views"), 4 if keep_path_details else 2),
            "prompt_truncated_views": _list_prefix(
                group.get("prompt_truncated_views"),
                3,
            ),
            "packet_truncated_views": _list_prefix(
                group.get("packet_truncated_views"),
                3,
            ),
            "tool_statuses": _list_prefix(
                group.get("tool_statuses"),
                4 if keep_path_details else 2,
            ),
            "followup_view_attempts": _list_prefix(
                group.get("followup_view_attempts"),
                3 if keep_path_details else 1,
            ),
            "successful_followup_views": _list_prefix(
                group.get("successful_followup_views"),
                3 if keep_path_details else 1,
            ),
            "answer_changing_followup_views": _list_prefix(
                group.get("answer_changing_followup_views"),
                3 if keep_path_details else 1,
            ),
            "missing_statuses": dict(group.get("missing_statuses") or {}),
            "missing_categories": dict(group.get("missing_categories") or {}),
            "tool_attempt_count": group.get("tool_attempt_count", 0),
            "execution_paths": [
                _compact_plan_audit_execution_path(path)
                for path in list(group.get("execution_paths") or [])[
                    :REVIEW_PLAN_AUDIT_RELEVANT_PATH_LIMIT
                ]
                if isinstance(path, dict)
            ] if keep_path_details else [],
            "evidence_history_present": bool(group.get("evidence_history_present")),
        }
    return {
        "is_exclusion": bool(condition.get("is_exclusion")),
        "groups": groups,
        "route_failure_memory": _list_prefix(
            condition.get("route_failure_memory"),
            10,
        ),
    }


def _plan_audit_group_relevant_to_error(
    group_name: str,
    *,
    error_type: str,
) -> bool:
    group = str(group_name or "").strip().lower()
    error = str(error_type or "").strip().upper()
    if error == "FN":
        return group == "positive_error"
    if error == "FP":
        return group in {"negative_error", "guard_error"}
    return group.endswith("_error")


def _compact_plan_audit_execution_path(path: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": path.get("case_id", ""),
        "tx_hash": str(path.get("tx_hash") or "")[:14],
        "answer": path.get("answer"),
        "confidence": path.get("confidence"),
        "views": _list_prefix(path.get("views"), 4),
        "tools": _list_prefix(path.get("tools"), 4),
        "missing_evidence": _list_prefix(path.get("missing_evidence"), 4),
        "supporting_evidence_ids": _list_prefix(
            path.get("supporting_evidence_ids"),
            4,
        ),
    }


def _condition_relevance(row: Dict[str, Any], *, error_type: str) -> str:
    answer = row.get("answer")
    is_exclusion = bool(row.get("is_exclusion"))
    if error_type == "FN":
        if is_exclusion and answer is True:
            return "fired_exclusion_for_fn"
        if not is_exclusion and answer is not True:
            return "uncertain_core_for_fn" if isinstance(answer, str) else "blocking_core_for_fn"
    if error_type == "FP":
        if not is_exclusion and answer is True:
            return "supporting_core_for_fp"
        if is_exclusion and answer is not True:
            return "uncertain_exclusion_for_fp" if isinstance(answer, str) else "missing_exclusion_for_fp"
    if row.get("error_hint"):
        return "error_hint_context"
    return "context"


def _review_feature_analysis(value: Any, *, relevance: str) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    max_items = 4 if relevance != "context" else 2
    max_chars = 220 if relevance != "context" else 140
    return {
        key: [
            _truncate_text(item, max_chars)
            for item in _list_prefix(raw.get(key), max_items)
        ]
        for key in (
            "matched_features",
            "partial_match_features",
            "missing_required_features",
            "contradicting_features",
            "boundary_notes",
        )
    }


def _review_stateful_state(row: Dict[str, Any]) -> Dict[str, Any]:
    stateful = row.get("stateful_runtime") if isinstance(row.get("stateful_runtime"), dict) else {}
    state_input = row.get("state_input") if isinstance(row.get("state_input"), dict) else {}
    state_output = row.get("state_output") if isinstance(row.get("state_output"), dict) else {}
    if not stateful and not state_input and not state_output:
        return {}
    return {
        "state_prompt_role": stateful.get("state_prompt_role", ""),
        "consumes_state_keys": _list_prefix(stateful.get("consumes_state_keys"), 4),
        "produces_state_key": stateful.get("produces_state_key", ""),
        "state_input": _compact_state_dict_for_review(state_input),
        "state_output": _compact_state_dict_for_review(state_output),
    }


def _compact_state_dict_for_review(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    value = copy.deepcopy(value)
    if (
        isinstance(value.get("access_control_candidate"), dict)
        and isinstance(value.get("authorization_chain_summary"), dict)
    ):
        value.pop("authorization_chain_summary", None)
    removed_counts: dict[str, int] = {}
    return _limit_value(
        value,
        max_string=260,
        max_list=4,
        path="stateful_binding_state",
        removed_counts=removed_counts,
    )


def _relevant_condition_ids(rows: list[Dict[str, Any]]) -> set[str]:
    ids = {
        str(row.get("condition_id") or row.get("id") or "")
        for row in rows
        if row.get("relevance") != "context"
    }
    if ids:
        return {item for item in ids if item}
    return {
        str(row.get("condition_id") or row.get("id") or "")
        for row in rows
        if row.get("condition_id") or row.get("id")
    }


def _error_focused_rule_context(
    rule_digest: Dict[str, Any],
    relevant_ids: set[str],
) -> Dict[str, Any]:
    return {
        "rule_id": rule_digest.get("rule_id", ""),
        "version": rule_digest.get("version", ""),
        "attack_label": rule_digest.get("attack_label")
        or (rule_digest.get("metadata", {}) or {}).get("attack_label", ""),
        "conditions": _focused_rule_conditions(
            rule_digest.get("conditions"),
            relevant_ids,
        ),
        "exclusion_conditions": _focused_rule_conditions(
            rule_digest.get("exclusion_conditions"),
            relevant_ids,
        ),
        "decision_policy": _truncate_text(rule_digest.get("decision_policy", ""), 900),
    }


def _focused_rule_conditions(value: Any, relevant_ids: set[str]) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id", ""))
        out.append({
            "id": cid,
            "relevant_to_error": cid in relevant_ids,
            "description": _truncate_text(
                item.get("description", ""),
                700 if cid in relevant_ids else 260,
            ),
            "expected_answer": item.get("expected_answer", True),
        })
    return out


def _error_focused_plan_context(
    plan_digest: Dict[str, Any],
    relevant_ids: set[str],
) -> Dict[str, Any]:
    steps = []
    for step in list(plan_digest.get("judge_steps", []) or []):
        if not isinstance(step, dict):
            continue
        cid = str(step.get("condition_id") or step.get("id") or "")
        steps.append({
            "id": step.get("id", ""),
            "condition_id": cid,
            "relevant_to_error": cid in relevant_ids,
            "question": _truncate_text(
                step.get("question", ""),
                500 if cid in relevant_ids else 180,
            ),
            "expected_answer": step.get("expected_answer", True),
            "default_evidence_refs": _list_prefix(step.get("default_evidence_refs"), 8),
            "allowed_followup_views": _list_prefix(step.get("allowed_followup_views"), 8),
            "allowed_tools": _list_prefix(step.get("allowed_tools"), 8),
            "max_followups": step.get("max_followups", 0),
            "depends_on": _list_prefix(step.get("depends_on"), 4),
            "consumes_state_keys": _list_prefix(step.get("consumes_state_keys"), 4),
            "produces_state_key": step.get("produces_state_key", ""),
            "state_prompt_role": step.get("state_prompt_role", ""),
        })
    return {
        "plan_id": plan_digest.get("plan_id", ""),
        "emit_logic": plan_digest.get("emit_logic", ""),
        "judge_steps": steps,
    }


def _available_packet_views(evidence_digest: Dict[str, Any]) -> list[str]:
    summary = evidence_digest.get("packet_view_summary")
    if not isinstance(summary, dict):
        return []
    return [
        str(name)
        for name, info in summary.items()
        if isinstance(info, dict) and bool(info.get("available", True))
    ][:40]


def _compact_error_focused_case_to_fit(
    focus_case: Dict[str, Any],
    *,
    max_prompt_chars: int,
    build_prompt,
) -> tuple[Dict[str, Any], list[str], dict[str, int]]:
    levels = [
        {"max_string": 520, "max_list": 8, "name": "focused_medium"},
        {"max_string": 300, "max_list": 5, "name": "focused_tight"},
        {"max_string": 160, "max_list": 3, "name": "focused_minimal"},
        {"max_string": 80, "max_list": 2, "name": "focused_ultra"},
    ]
    best_case = copy.deepcopy(focus_case)
    best_counts: dict[str, int] = {}
    best_fields: list[str] = []
    for level in levels:
        counts: dict[str, int] = {}
        compact = _limit_value(
            focus_case,
            max_string=int(level["max_string"]),
            max_list=int(level["max_list"]),
            path="error_focused_reviewer_case",
            removed_counts=counts,
        )
        best_case = compact
        best_counts = counts
        best_fields = [f"error_focused_reviewer_case.{level['name']}"]
        if len(build_prompt(compact)) <= max_prompt_chars:
            break
    return best_case, best_fields, best_counts


def _error_focused_truncation_metadata(
    prompt: str,
    prompt_case: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "compact_strategy": "error_focused_v1",
        "prompt_chars_after": len(str(prompt or "")),
        "prompt_est_tokens_after": estimate_prompt_tokens(prompt),
        "priority_sections_preserved": [
            "case_header",
            "error_focus",
            "aggregation_audit",
            "condition_error_table",
            "condition_feature_analysis",
            "condition_error_table.latest_or_truncated_view_history",
            "condition_error_table.source_or_latest_tool_history",
            "condition_error_table.missing_evidence_history",
            "plan_evidence_audit.compact_group_summaries",
            "cohort_signal_summary.budgeted_feature_observations",
            "stateful_binding_state",
            "rule_context",
            "plan_context",
            "evidence_limitations",
        ],
        "condition_error_table_count": len(
            list(prompt_case.get("condition_error_table", []) or [])
        )
        if isinstance(prompt_case, dict)
        else 0,
    }


def _has_category(root_causes: list[Dict[str, Any]], categories: set[str]) -> bool:
    return any(str(cause.get("category") or "").strip().lower() in categories for cause in root_causes)


def _normalize_update_target_with_consistency(
    review: Dict[str, Any],
    case: Dict[str, Any],
    root_causes: list[Dict[str, Any]],
    condition_diagnosis: list[Dict[str, Any]],
    update_target: str,
) -> Dict[str, Any]:
    original = str(review.get("update_target") or update_target or "").strip().lower()
    if _looks_like_runtime_emit_bug(case):
        return {
            "original_update_target": original,
            "normalized_update_target": "runtime",
            "reason": "runtime_emit_logic_bug_detected_from_condition_table",
        }
    if _has_category(root_causes, {"deferred_logic_structure_failure"}):
        return {
            "original_update_target": original,
            "normalized_update_target": "none",
            "reason": "deferred_logic_structure_failure",
        }

    rule_patch = _patch_suggestion(review.get("rule_patch_suggestion"), "none")
    plan_patch = _patch_suggestion(review.get("plan_patch_suggestion"), "none")
    packet_patch = _patch_suggestion(review.get("packet_patch_suggestion"), "none")
    runtime_patch = _patch_suggestion(review.get("runtime_patch_suggestion"), "none")
    has_rule_root = _has_category(root_causes, RULE_CATEGORIES) or any(
        bool(item.get("is_rule_problem")) for item in condition_diagnosis
    )
    has_plan_root = _has_category(root_causes, PLAN_CATEGORIES) or any(
        bool(item.get("is_plan_problem")) for item in condition_diagnosis
    )
    has_packet_root = _has_category(root_causes, PACKET_CATEGORIES) or any(
        bool(item.get("is_packet_problem")) for item in condition_diagnosis
    )
    has_runtime_root = _has_category(root_causes, RUNTIME_CATEGORIES) or any(
        bool(item.get("is_runtime_problem")) for item in condition_diagnosis
    )

    if update_target == "rule" and not has_rule_root and rule_patch.get("action") in {"", "none"}:
        inferred = infer_update_target(case, root_causes)
        return {
            "original_update_target": original,
            "normalized_update_target": inferred,
            "reason": "rule_target_without_rule_root_cause_or_action",
        }
    if update_target == "none":
        inferred = infer_update_target(case, root_causes)
        if inferred != "none":
            return {
                "original_update_target": original,
                "normalized_update_target": inferred,
                "reason": "none_target_with_actionable_root_cause",
            }
    if update_target == "plan" and plan_patch.get("action") in {"", "none"} and not has_plan_root:
        inferred = infer_update_target(case, root_causes)
        return {
            "original_update_target": original,
            "normalized_update_target": inferred,
            "reason": "plan_target_without_plan_root_cause_or_action",
        }
    if update_target == "packet" and packet_patch.get("action") in {"", "none"} and not has_packet_root:
        inferred = infer_update_target(case, root_causes)
        return {
            "original_update_target": original,
            "normalized_update_target": inferred,
            "reason": "packet_target_without_packet_action",
        }
    if update_target == "runtime" and runtime_patch.get("action") in {"", "none"} and not has_runtime_root:
        inferred = infer_update_target(case, root_causes)
        return {
            "original_update_target": original,
            "normalized_update_target": inferred,
            "reason": "runtime_target_without_runtime_action",
        }
    return {
        "original_update_target": original,
        "normalized_update_target": update_target,
        "reason": "consistent",
    }


def _looks_like_runtime_emit_bug(case: Dict[str, Any]) -> bool:
    pred = case.get("predicted_verdict") or _finding_summary(case).get("verdict")
    gold = case.get("ground_truth")
    if gold != "attack" or pred == "attack":
        return False
    rows = list(case.get("condition_table", []) or [])
    if not rows:
        return False
    core_rows = [row for row in rows if not row.get("is_exclusion")]
    exclusion_rows = [row for row in rows if row.get("is_exclusion")]
    if not core_rows:
        return False
    all_core_true = all(row.get("answer") is True for row in core_rows)
    all_exclusions_false = all(row.get("answer") is False for row in exclusion_rows)
    return all_core_true and all_exclusions_false


def _patch_suggestion(value: Any, default_action: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    return {
        "action": str(value.get("action", default_action) or default_action),
        "condition_id": str(value.get("condition_id", "")),
        "semantic_direction": str(value.get("semantic_direction", "")),
        "view_name": str(value.get("view_name", "")),
        "old_problem": str(value.get("old_problem", "")),
        "proposed_semantic_change": str(value.get("proposed_semantic_change", "")),
        "proposed_change": str(value.get("proposed_change", "")),
        "rationale": str(value.get("rationale", "")),
    }


_CASE_SPECIFIC_UPDATE_RE = re.compile(
    r"0x[0-9a-fA-F]{6,}|\b(?:call|event|transfer|sload|sstore):\d+\b|"
    r"\b(?:ctf|challenge[- ]?like|joke selector|codeislaw)\b",
    re.IGNORECASE,
)


def _normalize_update_signals(
    value: Any,
    *,
    case: Dict[str, Any],
    default_update_target: str,
) -> list[Dict[str, Any]]:
    raw_signals = value if isinstance(value, list) else []
    cohort = dict(case.get("cohort_signal_summary") or {})
    counts = dict(cohort.get("counts") or {})
    correct_positive_count = int(counts.get("correct_positive", 0) or 0)
    correct_boundary_count = int(counts.get("correct_negative", 0) or 0) + int(
        counts.get("correct_guard", 0) or 0
    )
    correct_reference_count = correct_positive_count + correct_boundary_count
    allowed_targets = {"rule", "plan", "packet", "runtime", "none"}
    allowed_directions = {
        "broaden",
        "tighten",
        "refine",
        "add_exclusion",
        "narrow_exclusion",
        "clarify_boundary",
        "change_evidence_strategy",
        "no_change",
    }
    allowed_strategy_operations = {
        "none",
        "replace_evidence_route",
        "reorder_evidence_route",
        "demote_default_to_followup",
        "prune_evidence_route",
        "change_verification_granularity",
        "change_followup_budget",
        "change_dependency_route",
    }
    direction_aliases = {
        "change_default_views": "change_evidence_strategy",
        "change_followup_views": "change_evidence_strategy",
        "change_judge_question": "change_evidence_strategy",
    }
    available_feature_refs = _cohort_feature_refs(cohort)
    normalized: list[Dict[str, Any]] = []
    for raw in raw_signals:
        if not isinstance(raw, dict):
            continue
        update_target = str(
            raw.get("update_target") or default_update_target or "none"
        ).lower().strip()
        if update_target not in allowed_targets:
            update_target = "none"
        direction = str(raw.get("direction") or "no_change").lower().strip()
        direction = direction_aliases.get(direction, direction)
        if direction not in allowed_directions:
            direction = "no_change"
        strategy_operation = str(
            raw.get("strategy_operation") or "none"
        ).lower().strip()
        if strategy_operation not in allowed_strategy_operations:
            strategy_operation = "none"
        abstract_feature = " ".join(
            str(raw.get("abstract_feature") or "").split()
        )[:500]
        feature_axis = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(raw.get("feature_axis") or "").lower(),
        ).strip("_")[:160]
        positive_compatibility = " ".join(
            str(raw.get("correct_positive_compatibility") or "").split()
        )[:700]
        negative_compatibility = " ".join(
            str(raw.get("correct_negative_compatibility") or "").split()
        )[:700]
        rationale = " ".join(str(raw.get("rationale") or "").split())[:700]
        evidence_dependency = _normalize_condition_evidence_dependency(
            raw.get("condition_evidence_dependency"),
            update_target=update_target,
        )
        status = _normalize_choice(
            raw.get("generalization_status"),
            {"supported", "conflicted", "insufficient"},
            "insufficient",
        )
        support_refs = _normalize_feature_refs(
            raw.get("cohort_support_refs"),
            available_feature_refs,
        )
        conflict_refs = _normalize_feature_refs(
            raw.get("cohort_conflict_refs"),
            available_feature_refs,
        )
        combined_text = " ".join(
            (abstract_feature, positive_compatibility, negative_compatibility, rationale)
        )
        wrong_scope_audit = _normalize_wrong_scope_audit(
            raw.get("wrong_scope_audit")
        )
        target_label = normalize_attack_label(
            case.get("target_label") or case.get("attack_label") or "",
            default="",
        )
        wrong_scope_signal = bool(
            target_label == "insufficient_validation"
            and (
                wrong_scope_audit
                or re.search(r"\bwrong[-_ ]scope\b", combined_text, re.IGNORECASE)
            )
        )
        condition_id = str(raw.get("condition_id") or "").strip()
        structural_issues = []
        if not condition_id:
            structural_issues.append("missing_condition_id")
        if not abstract_feature:
            structural_issues.append("missing_abstract_feature")
        if update_target == "none":
            structural_issues.append("missing_update_owner")
        if direction == "no_change":
            structural_issues.append("missing_update_direction")
        if structural_issues:
            status = "insufficient"
        eligibility_diagnostics = []
        if correct_reference_count <= 0:
            eligibility_diagnostics.append("no_correct_reference_available")
        if _CASE_SPECIFIC_UPDATE_RE.search(combined_text):
            eligibility_diagnostics.append("case_specific_language_detected")
        if correct_positive_count > 0 and not positive_compatibility:
            eligibility_diagnostics.append("positive_compatibility_not_explained")
        if correct_boundary_count > 0 and not negative_compatibility:
            eligibility_diagnostics.append("negative_compatibility_not_explained")
        if wrong_scope_signal and not _wrong_scope_audit_complete(wrong_scope_audit):
            eligibility_diagnostics.append("wrong_scope_audit_incomplete")
        signal = {
            "update_target": update_target,
            "condition_id": condition_id,
            "direction": direction,
            "strategy_operation": strategy_operation,
            "condition_evidence_dependency": evidence_dependency,
            "abstract_feature": abstract_feature,
            "feature_axis": feature_axis,
            "wrong_scope_audit": wrong_scope_audit,
            "correct_positive_compatibility": positive_compatibility,
            "correct_negative_compatibility": negative_compatibility,
            "cohort_support_refs": support_refs,
            "cohort_conflict_refs": conflict_refs,
            "generalization_status": status,
            "rationale": rationale,
            "correct_reference_count": correct_reference_count,
            "reviewer_structural_issues": structural_issues,
            "reviewer_eligibility_diagnostics": eligibility_diagnostics,
        }
        if wrong_scope_signal and not _wrong_scope_audit_complete(wrong_scope_audit):
            signal["signal_gate_reason"] = "wrong_scope_audit_incomplete"
        normalized.append(signal)
    return normalized


def _normalize_condition_evidence_dependency(
    value: Any,
    *,
    update_target: str,
) -> Dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    role = str(raw.get("role") or "none").strip().lower()
    expected_role = {"rule": "requires", "plan": "provides"}.get(
        str(update_target or "").strip().lower()
    )
    if role != expected_role:
        return {}
    capability_id = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(raw.get("capability_id") or "").strip().lower(),
    ).strip("_")[:96]
    description = " ".join(
        str(raw.get("capability_description") or "").split()
    )[:280]
    if (
        not capability_id
        or not description
        or _CASE_SPECIFIC_UPDATE_RE.search(description)
    ):
        return {}
    return {
        "role": role,
        "capability_id": capability_id,
        "capability_description": description,
    }


_WRONG_SCOPE_AUDIT_FIELDS = (
    "consumed_object",
    "required_invariant",
    "observed_check",
    "scope_mismatch",
    "adversarial_control",
    "causal_outcome",
)


def _normalize_wrong_scope_audit(value: Any) -> Dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    return {
        field: " ".join(str(raw.get(field) or "").split())[:500]
        for field in _WRONG_SCOPE_AUDIT_FIELDS
        if str(raw.get(field) or "").strip()
    }


def _wrong_scope_audit_complete(value: Dict[str, str]) -> bool:
    return all(
        str((value or {}).get(field) or "").strip()
        for field in _WRONG_SCOPE_AUDIT_FIELDS
    )


def _cohort_feature_refs(cohort: Dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for condition in dict((cohort or {}).get("conditions") or {}).values():
        for group in dict((condition or {}).get("groups") or {}).values():
            for observations in dict((group or {}).get("features") or {}).values():
                for observation in list(observations or []):
                    if not isinstance(observation, dict):
                        continue
                    feature_ref = str(observation.get("feature_ref") or "").strip()
                    if feature_ref:
                        refs.add(feature_ref)
    return refs


def _normalize_feature_refs(value: Any, available_refs: set[str]) -> list[str]:
    raw_refs = [value] if isinstance(value, str) else list(value or [])
    refs: list[str] = []
    for raw_ref in raw_refs:
        feature_ref = str(raw_ref or "").strip()
        if not feature_ref or feature_ref in refs:
            continue
        if available_refs and feature_ref not in available_refs:
            continue
        refs.append(feature_ref)
    return refs[:12]


FEATURE_DIAGNOSIS_IMPLICATIONS = {
    "broaden",
    "tighten",
    "add_exclusion",
    "clarify_boundary",
    "no_change",
}


def _normalize_feature_diagnosis(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        implication = str(item.get("update_implication") or "no_change").strip().lower()
        if implication not in FEATURE_DIAGNOSIS_IMPLICATIONS:
            implication = "no_change"
        out.append({
            "condition_id": str(item.get("condition_id") or "").strip(),
            "observed_matched_features": _string_list(item.get("observed_matched_features")),
            "observed_partial_match_features": _string_list(
                item.get("observed_partial_match_features")
            ),
            "observed_missing_required_features": _string_list(
                item.get("observed_missing_required_features")
            ),
            "observed_contradicting_features": _string_list(
                item.get("observed_contradicting_features")
            ),
            "observed_boundary_notes": _string_list(item.get("observed_boundary_notes")),
            "feature_interpretation": str(item.get("feature_interpretation") or "").strip(),
            "update_implication": implication,
            "rule_update_hint": str(item.get("rule_update_hint") or "").strip(),
        })
    return out


def _compact_reviewer_case_to_fit(
    case: Dict[str, Any],
    *,
    max_prompt_chars: int,
    build_prompt,
) -> tuple[Dict[str, Any], list[str], dict[str, int]]:
    levels = [
        {
            "name": "drop_resolution",
            "max_string": 1800,
            "max_list": 40,
            "summarize_rounds": True,
            "drop_resolution": True,
        },
        {
            "name": "medium",
            "max_string": 900,
            "max_list": 20,
            "summarize_rounds": True,
            "drop_resolution": True,
        },
        {
            "name": "tight",
            "max_string": 420,
            "max_list": 10,
            "summarize_rounds": True,
            "drop_resolution": True,
        },
        {
            "name": "very_tight",
            "max_string": 220,
            "max_list": 6,
            "summarize_rounds": True,
            "drop_resolution": True,
        },
        {
            "name": "minimal",
            "max_string": 80,
            "max_list": 3,
            "summarize_rounds": True,
            "drop_resolution": True,
            "minimal": True,
        },
        {
            "name": "ultra_minimal",
            "max_string": 30,
            "max_list": 2,
            "summarize_rounds": True,
            "drop_resolution": True,
            "minimal": True,
            "ultra_minimal": True,
        },
    ]
    best_case = copy.deepcopy(case)
    best_removed_fields: list[str] = []
    best_removed_counts: dict[str, int] = {}
    for level in levels:
        removed_fields: list[str] = []
        removed_counts: dict[str, int] = {}
        compact = _compact_reviewer_case(case, level, removed_fields, removed_counts)
        best_case = compact
        best_removed_fields = removed_fields
        best_removed_counts = removed_counts
        if len(build_prompt(compact)) <= max_prompt_chars:
            break
    return best_case, sorted(set(best_removed_fields)), best_removed_counts


def _compact_reviewer_case(
    case: Dict[str, Any],
    level: Dict[str, Any],
    removed_fields: list[str],
    removed_counts: dict[str, int],
) -> Dict[str, Any]:
    compact = copy.deepcopy(case)
    if level.get("minimal"):
        compact = _minimal_reviewer_case(compact, level, removed_fields, removed_counts)
    evidence_digest = compact.get("evidence_digest")
    if isinstance(evidence_digest, dict) and level.get("drop_resolution"):
        if "evidence_id_resolution" in evidence_digest:
            evidence_digest.pop("evidence_id_resolution", None)
            removed_fields.append("evidence_digest.evidence_id_resolution")
    if isinstance(evidence_digest, dict):
        for key in ("top_supporting_evidence_ids", "unresolved_or_missing_evidence"):
            if isinstance(evidence_digest.get(key), list):
                evidence_digest[key] = _limit_list(
                    evidence_digest[key],
                    int(level.get("max_list", 20)),
                    f"evidence_digest.{key}",
                    removed_counts,
                )

    rows = compact.get("condition_table")
    if isinstance(rows, list):
        compact["condition_table"] = [
            _compact_condition_row(row, level, removed_fields, removed_counts)
            if isinstance(row, dict)
            else row
            for row in rows
        ]

    return _limit_value(
        compact,
        max_string=int(level.get("max_string", 900)),
        max_list=int(level.get("max_list", 20)),
        path="reviewer_case",
        removed_counts=removed_counts,
    )


def _minimal_reviewer_case(
    case: Dict[str, Any],
    level: Dict[str, Any],
    removed_fields: list[str],
    removed_counts: dict[str, int],
) -> Dict[str, Any]:
    out = copy.deepcopy(case)
    ultra = bool(level.get("ultra_minimal"))
    if isinstance(out.get("rule_digest"), dict):
        out["rule_digest"] = _minimal_rule_digest(out["rule_digest"], ultra=ultra)
        removed_fields.append("rule_digest.details")
    if isinstance(out.get("plan_digest"), dict):
        out["plan_digest"] = _minimal_plan_digest(out["plan_digest"], ultra=ultra)
        removed_fields.append("plan_digest.details")
    if isinstance(out.get("finding_summary"), dict):
        out["finding_summary"] = _minimal_finding_summary(out["finding_summary"])
        removed_fields.append("finding_summary.details")
    if isinstance(out.get("evidence_digest"), dict):
        out["evidence_digest"] = _minimal_evidence_digest(
            out["evidence_digest"],
            removed_counts,
            ultra=ultra,
        )
        removed_fields.append("evidence_digest.details")
    return out


def _minimal_rule_digest(value: Dict[str, Any], *, ultra: bool = False) -> Dict[str, Any]:
    return {
        "rule_id": value.get("rule_id", ""),
        "version": value.get("version", ""),
        "attack_label": value.get("attack_label")
        or (value.get("metadata", {}) or {}).get("attack_label", ""),
        "conditions": _minimal_conditions(value.get("conditions"), max_chars=60 if ultra else 140),
        "exclusion_conditions": _minimal_conditions(
            value.get("exclusion_conditions"),
            max_chars=60 if ultra else 140,
        ),
        "decision_policy": value.get("decision_policy", ""),
    }


def _minimal_conditions(value: Any, *, max_chars: int) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        out.append({
            "id": item.get("id", ""),
            "description": _truncate_text(item.get("description", ""), max_chars),
            "expected_answer": item.get("expected_answer", True),
        })
    return out


def _minimal_plan_digest(value: Dict[str, Any], *, ultra: bool = False) -> Dict[str, Any]:
    steps = value.get("judge_steps", [])
    if not isinstance(steps, list):
        steps = []
    return {
        "plan_id": value.get("plan_id", ""),
        "emit_logic": value.get("emit_logic", ""),
        "judge_steps": [
            {
                "id": step.get("id", ""),
                "condition_id": step.get("condition_id", ""),
                "question": _truncate_text(step.get("question", ""), 40 if ultra else 100),
                "expected_answer": step.get("expected_answer", True),
                "default_evidence_refs": _list_prefix(
                    step.get("default_evidence_refs"),
                    2 if ultra else 3,
                ),
                "allowed_followup_views": _list_prefix(
                    step.get("allowed_followup_views"),
                    2 if ultra else 3,
                ),
                "max_followups": step.get("max_followups", 0),
            }
            for step in steps
            if isinstance(step, dict)
        ],
    }


def _minimal_finding_summary(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verdict": value.get("verdict"),
        "confidence": value.get("confidence"),
        "verdict_reason": _truncate_text(value.get("verdict_reason", ""), 180),
        "missing_evidence": _list_prefix(value.get("missing_evidence"), 5),
        "all_missing_evidence_debug": _list_prefix(
            value.get("all_missing_evidence_debug"),
            5,
        ),
    }


def _minimal_evidence_digest(
    value: Dict[str, Any],
    removed_counts: dict[str, int],
    *,
    ultra: bool = False,
) -> Dict[str, Any]:
    packet_summary = value.get("packet_view_summary")
    packet_views: list[str] = []
    if isinstance(packet_summary, dict):
        packet_views = [
            str(name)
            for name, info in packet_summary.items()
            if isinstance(info, dict) and bool(info.get("available", True))
        ]
        max_views = 10 if ultra else 20
        if len(packet_views) > max_views:
            removed_counts["evidence_digest.packet_view_summary"] = len(packet_views) - max_views
            packet_views = packet_views[:max_views]
    return {
        "available_packet_views": packet_views,
        "top_supporting_evidence_ids": _list_prefix(
            value.get("top_supporting_evidence_ids"),
            4 if ultra else 8,
        ),
        "unresolved_or_missing_evidence": _list_prefix(
            value.get("unresolved_or_missing_evidence"),
            4 if ultra else 8,
        ),
        "missing_evidence_actionability_summary": value.get(
            "missing_evidence_actionability_summary",
            {},
        ),
        "evidence_adequacy_view": _truncate_text(
            stable_json_dumps(value.get("evidence_adequacy_view", {})),
            80 if ultra else 420,
        ),
    }


def _compact_condition_row(
    row: Dict[str, Any],
    level: Dict[str, Any],
    removed_fields: list[str],
    removed_counts: dict[str, int],
) -> Dict[str, Any]:
    if level.get("minimal"):
        keep_keys = {
            "id",
            "condition_id",
            "is_exclusion",
            "question",
            "expected_answer",
            "answer",
            "confidence",
            "reason_short",
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "missing_evidence",
            "condition_feature_analysis",
            "error_hint",
        }
    else:
        keep_keys = {
        "id",
        "condition_id",
        "is_exclusion",
        "question",
        "expected_answer",
        "answer",
        "first_round_answer",
        "final_answer",
        "followup_changed_answer",
        "confidence",
        "reason_short",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "missing_evidence",
        "tool_request_count",
        "tool_call_count",
        "near_miss_escalated",
        "condition_feature_analysis",
        "final_round",
        "error_hint",
        }
    out = {key: copy.deepcopy(value) for key, value in row.items() if key in keep_keys}
    dropped = sorted(set(row) - set(out))
    if dropped:
        removed_fields.extend(f"condition_table.{key}" for key in dropped)
    max_list = int(level.get("max_list", 20))
    evidence_id_limit = (
        1 if level.get("ultra_minimal") else 2 if level.get("minimal") else max(5, min(max_list, 12))
    )
    for key in (
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "missing_evidence",
        "suggested_followup_views",
    ):
        if isinstance(out.get(key), list):
            out[key] = _limit_list(
                out[key],
                evidence_id_limit,
                f"condition_table.{key}",
                removed_counts,
            )
    if level.get("minimal") and isinstance(out.get("condition_feature_analysis"), dict):
        out["condition_feature_analysis"] = _minimal_feature_analysis(
            out["condition_feature_analysis"],
            ultra=bool(level.get("ultra_minimal")),
        )
    if level.get("summarize_rounds"):
        for key in ("first_round", "final_round"):
            if isinstance(out.get(key), dict):
                out[key] = _round_summary(out[key])
                removed_fields.append(f"condition_table.{key}.details")
    return out


def _minimal_feature_analysis(value: Dict[str, Any], *, ultra: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "matched_features",
        "partial_match_features",
        "missing_required_features",
        "contradicting_features",
        "boundary_notes",
    ):
        items = value.get(key)
        if not isinstance(items, list):
            items = []
        out[key] = [_truncate_text(item, 25 if ultra else 70) for item in items[:1]]
    return out


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


def _round_summary(round_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer": round_data.get("answer"),
        "confidence": round_data.get("confidence"),
        "satisfied": round_data.get("satisfied"),
        "reason": round_data.get("reason_short") or round_data.get("reason", ""),
        "supporting_evidence_ids": _list_prefix(
            round_data.get("supporting_evidence_ids"),
            8,
        ),
        "contradicting_evidence_ids": _list_prefix(
            round_data.get("contradicting_evidence_ids"),
            8,
        ),
        "missing_evidence": _list_prefix(round_data.get("missing_evidence"), 8),
        "condition_feature_analysis": round_data.get("condition_feature_analysis", {}),
    }


def _limit_value(
    value: Any,
    *,
    max_string: int,
    max_list: int,
    path: str,
    removed_counts: dict[str, int],
) -> Any:
    if isinstance(value, str):
        if ".cohort_signal_summary" in path and path.endswith(".feature"):
            max_string = max(max_string, 180)
        if len(value) <= max_string:
            return value
        removed_counts[f"{path}.__string_chars_removed"] = (
            removed_counts.get(f"{path}.__string_chars_removed", 0)
            + len(value)
            - max_string
        )
        return value[:max_string].rstrip() + " ...[truncated]"
    if isinstance(value, list):
        preserve_feature_observations = (
            ".cohort_signal_summary" in path and ".features." in path
        )
        list_limit = max(max_list, 4) if preserve_feature_observations else max_list
        limited = value if path == "reviewer_case.condition_table" else _limit_list(
            value,
            list_limit,
            path,
            removed_counts,
        )
        return [
            _limit_value(
                item,
                max_string=max_string,
                max_list=max_list,
                path=f"{path}[]",
                removed_counts=removed_counts,
            )
            for item in limited
        ]
    if isinstance(value, dict):
        return {
            key: _limit_value(
                item,
                max_string=max_string,
                max_list=max_list,
                path=f"{path}.{key}",
                removed_counts=removed_counts,
            )
            for key, item in value.items()
        }
    return value


def _limit_list(
    value: Any,
    max_items: int,
    path: str,
    removed_counts: dict[str, int],
) -> list[Any]:
    if not isinstance(value, list):
        return []
    if len(value) <= max_items:
        return value
    removed_counts[path] = removed_counts.get(path, 0) + len(value) - max_items
    return list(value[:max_items])


def _list_prefix(value: Any, max_items: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return list(value[:max_items])


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        if value is None or value == "":
            return []
        value = [value]
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _normalize_confidence(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    if text in {"low", "medium", "high"}:
        return text
    return "medium"


def _normalize_choice(
    value: Any,
    allowed: set[str],
    default: str,
) -> str:
    text = str(value or default).strip().lower()
    return text if text in allowed else default


def _finding_summary(case: Dict[str, Any]) -> Dict[str, Any]:
    value = case.get("finding_summary")
    return value if isinstance(value, dict) else {}
