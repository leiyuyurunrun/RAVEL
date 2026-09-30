from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import normalize_condition_feature_analysis
from evotx.evolution.negative_training import (
    negative_training_mode_guidance,
    normalize_negative_training_mode,
)
from evotx.evolution.diagnosis_routing import (
    build_diagnosis_routing_audit,
    derive_missing_evidence_origins,
    is_authoritative_engineering_route,
    project_review_for_owner,
    review_has_operation,
)
from evotx.runtime.missing_evidence import (
    actionability_summary,
    group_missing_by_condition,
)
from evotx.runtime.view_catalog import PACKET_VIEW_CATALOG, packet_view_row_count


SLIM_RESULT_SCHEMA_VERSION = "evotx.slim_result.v1"
REVIEWER_CASE_SCHEMA_VERSION = "evotx.reviewer_case.v1"
COHORT_SIGNAL_SCHEMA_VERSION = "evotx.cohort_signal_summary.v2"
UPDATE_SIGNAL_MATRIX_SCHEMA_VERSION = "evotx.update_signal_matrix.v4"
PLAN_EVIDENCE_AUDIT_SCHEMA_VERSION = "evotx.plan_evidence_audit.v1"
CASE_BOUNDARY_CONTEXT_SCHEMA_VERSION = "evotx.case_boundary_context.v1"
REJECTED_UPDATE_MEMORY_SCHEMA_VERSION = "evotx.rejected_update_memory.v1"

COHORT_FEATURE_BUDGET_CHARS = 60000
COHORT_FEATURE_MAX_PER_BUCKET = 64
CASE_BOUNDARY_FEATURE_LIMIT_PER_TYPE = 6
CASE_BOUNDARY_CONDITION_LIMIT = 16

_CONCRETE_FEATURE_RE = re.compile(
    r"0x[0-9a-fA-F]{6,}|\b(?:call|event|transfer|sload|sstore):\d+\b|"
    r"\b(?:ctf|challenge[- ]?like|joke selector|codeislaw)\b",
    re.IGNORECASE,
)

_MANDATORY_SIGNAL_RE = re.compile(
    r"\b(?:require|requires|required|must|mandatory|necessary|only when|"
    r"only if|identifiable|identified|identification|confirm|confirmed|"
    r"confirmation|demonstrated)\b",
    re.IGNORECASE,
)
_POSITIVE_UNAVAILABLE_RE = re.compile(
    r"\b(?:unavailable|missing|absent|omitted|truncated|unresolved|inferred|"
    r"not available|without source|would strengthen|structural evidence is sufficient|"
    r"confirmation .* strengthen)\b",
    re.IGNORECASE,
)
_POSITIVE_UNCONFIRMED_RE = re.compile(
    r"\b(?:inferred|inference only|not confirmed|unconfirmed|not established|"
    r"not demonstrated|cannot confirm|could not confirm|uncertain|ambiguous|"
    r"would strengthen|requires? (?:semantic |source[- ]level )?confirmation)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_POSITIVE_PATH_RE = re.compile(
    r"\b(?:either\b.+\bor|alternative evidence path|fallback evidence path|"
    r"without requiring|structural evidence (?:remains|is) sufficient|"
    r"preserve(?:s|d)? the structural path)\b",
    re.IGNORECASE,
)
_RESTRICTIVE_SIGNAL_DIRECTIONS = {
    "tighten",
    "add_exclusion",
    "clarify_boundary",
    "change_evidence_strategy",
}
_MANDATORY_CONCEPT_ALIASES = {
    "semantics": "semantic",
    "semantically": "semantic",
    "identified": "identify",
    "identifiable": "identify",
    "identification": "identify",
    "confirmed": "confirm",
    "confirmation": "confirm",
    "storage": "state",
    "slot": "state",
    "slots": "state",
    "accounting": "accounting",
    "entitlement": "entitlement",
    "entitlements": "entitlement",
    "source": "source",
    "source-level": "source",
    "ordering": "order",
    "ordered": "order",
}
_MANDATORY_CONCEPT_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
_CROSS_REVIEW_AXIS_STOPWORDS = {
    "absent",
    "condition",
    "evidence",
    "gap",
    "generic",
    "missing",
    "mechanism",
    "positive",
    "protocol",
    "require",
    "required",
    "semantic",
    "source",
    "target",
    "transaction",
    "validate",
    "validation",
}


def build_slim_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact result record for evolution-time review and storage."""
    if _is_slim_result(result):
        return dict(result)

    transaction = _transaction(result)
    detector = _detector_context(result)
    inference = _inference(result)
    evaluation = _evaluation(result)
    rule = _rule(result)
    plan = _plan(result)
    trace = _trace(result)
    evidence = _evidence(result)
    finding = _finding(result)
    case_metadata = evaluation.get("case_metadata", {}) or {}

    condition_table = _build_condition_table(plan, trace, finding, evaluation)
    supporting_by_condition = _supporting_by_condition(finding, condition_table)
    tool_calls = _tool_call_summary(trace, condition_table)
    packet_summary = _packet_view_summary(evidence)
    adequacy = _compact(evidence.get("evidence_adequacy_view", {}), max_chars=16000)
    evidence_resolution = _evidence_id_resolution(finding, condition_table, evidence, tool_calls)
    all_missing_debug = _all_missing_debug(finding, condition_table)
    missing_status_by_condition = group_missing_by_condition(all_missing_debug)
    missing_actionability = actionability_summary(all_missing_debug)
    missing_origins = derive_missing_evidence_origins(
        all_missing_debug,
        condition_table=condition_table,
        packet_view_summary=packet_summary,
    )
    raw_attack_label = detector.get("attack_label") or (rule.get("metadata", {}) or {}).get("attack_label", "attack")
    cross_condition_audit = build_cross_condition_fact_consistency_audit(
        condition_table,
        attack_label=raw_attack_label,
    )

    return {
        "schema_version": SLIM_RESULT_SCHEMA_VERSION,
        "tx_hash": transaction.get("tx_hash", "unknown"),
        "chain": transaction.get("chain", "eth"),
        "attack_label": normalize_attack_label(raw_attack_label),
        "raw_attack_label": raw_attack_label,
        "rule_id": detector.get("rule_id") or rule.get("rule_id", ""),
        "rule_version": int(detector.get("rule_version") or rule.get("version", 1) or 1),
        "rule_source": detector.get("rule_source"),
        "plan_id": plan.get("plan_id", trace.get("plan_id", "")),
        "ground_truth": evaluation.get("ground_truth"),
        "raw_ground_truth": evaluation.get("raw_ground_truth"),
        "sample_role": case_metadata.get("sample_role", ""),
        "negative_kind": case_metadata.get("negative_kind", ""),
        "target_label": normalize_attack_label(case_metadata.get("target_label", ""), default=""),
        "raw_target_label": case_metadata.get("target_label", ""),
        "predicted_verdict": finding.get("verdict"),
        "confidence": finding.get("confidence", "medium"),
        "verdict_reason": finding.get("verdict_reason", ""),
        "emit_logic": plan.get("emit_logic", ""),
        "rule_digest": _rule_digest(rule),
        "plan_digest": _plan_digest(plan),
        "condition_table": condition_table,
        "cross_condition_fact_consistency_audit": cross_condition_audit,
        "finding_summary": _finding_summary(
            finding,
            missing_status_by_condition=missing_status_by_condition,
            all_missing_debug=all_missing_debug,
        ),
        "tool_call_summary": tool_calls,
        "missing_evidence_by_condition": missing_status_by_condition,
        "missing_evidence_actionability_summary": missing_actionability,
        "missing_evidence_origins": missing_origins,
        "supporting_evidence_by_condition": supporting_by_condition,
        "evidence_adequacy_summary": adequacy,
        "packet_view_summary": packet_summary,
        "evidence_id_resolution": evidence_resolution,
    }


def build_reviewer_case(
    result: Dict[str, Any],
    max_evidence_items: int = 30,
    *,
    include_label_rationale: bool = False,
) -> Dict[str, Any]:
    """Build the compact case passed to RuleReviewer LLM."""
    if _is_reviewer_case(result):
        return dict(result)

    source_label_rationale = str(
        (_evaluation(result).get("label_rationale") or result.get("label_rationale") or "")
    ).strip()
    slim = build_slim_result(result)
    error_type = _infer_error_type_from_values(
        slim.get("ground_truth"),
        slim.get("predicted_verdict"),
    )
    top_supporting = _unique(
        _flatten_values(slim.get("supporting_evidence_by_condition", {}))
    )[:max_evidence_items]
    missing = _unique(
        _flatten_open_missing(slim.get("missing_evidence_by_condition", {}))
        + list(slim.get("finding_summary", {}).get("missing_evidence", []))
    )[:max_evidence_items]

    contrastive_supervision = _contrastive_supervision(slim)
    evidence = _evidence(result)
    market_digest = _market_mechanism_reviewer_digest(evidence)
    missing_origin_history = list(slim.get("missing_evidence_origins") or [])
    if not missing_origin_history:
        missing_origin_history = derive_missing_evidence_origins(
            list(
                dict(slim.get("finding_summary") or {}).get(
                    "all_missing_evidence_debug", []
                )
                or []
            ),
            condition_table=list(slim.get("condition_table") or []),
            packet_view_summary=dict(slim.get("packet_view_summary") or {}),
        )
    active_missing_origins = [
        dict(item)
        for item in missing_origin_history
        if isinstance(item, dict) and _missing_evidence_is_active(item)
    ]
    reviewer_case = {
        "schema_version": REVIEWER_CASE_SCHEMA_VERSION,
        "case_id": f"{slim.get('tx_hash', 'unknown')}::{slim.get('rule_id', 'rule')}::v{slim.get('rule_version', 1)}",
        "tx_hash": slim.get("tx_hash", "unknown"),
        "chain": slim.get("chain", "eth"),
        "attack_label": slim.get("attack_label", "attack"),
        "ground_truth": slim.get("ground_truth"),
        "raw_ground_truth": slim.get("raw_ground_truth"),
        "sample_role": slim.get("sample_role", ""),
        "negative_kind": slim.get("negative_kind", ""),
        "target_label": slim.get("target_label", ""),
        "contrastive_supervision": contrastive_supervision,
        "predicted_verdict": slim.get("predicted_verdict"),
        "error_type": error_type,
        "rule_digest": slim.get("rule_digest", {}),
        "plan_digest": slim.get("plan_digest", {}),
        "condition_table": slim.get("condition_table", []),
        "cross_condition_fact_consistency_audit": (
            slim.get("cross_condition_fact_consistency_audit")
            or build_cross_condition_fact_consistency_audit(
                list(slim.get("condition_table") or []),
                attack_label=str(slim.get("attack_label") or ""),
            )
        ),
        "finding_summary": slim.get("finding_summary", {}),
        "evidence_digest": {
            "packet_view_summary": slim.get("packet_view_summary", {}),
            "evidence_adequacy_view": slim.get("evidence_adequacy_summary", {}),
            "top_supporting_evidence_ids": top_supporting,
            "unresolved_or_missing_evidence": missing,
            "evidence_id_resolution": slim.get("evidence_id_resolution", {}),
            "missing_evidence_actionability_summary": slim.get(
                "missing_evidence_actionability_summary", {}
            ),
            "missing_evidence_origins": active_missing_origins,
            "missing_evidence_origin_history": missing_origin_history,
        },
    }
    if slim.get("attack_label") == "market_manipulation" and market_digest:
        reviewer_case["evidence_digest"]["market_mechanism_digest"] = market_digest
    if include_label_rationale:
        label_rationale = source_label_rationale
        if label_rationale:
            reviewer_case["training_supervision"] = {
                "label_rationale": label_rationale,
                "usage_policy": (
                    "Training-only CSV rationale for Reviewer/Updater diagnosis. "
                    "It is not transaction evidence, must not be cited as proof, "
                    "and is never sent to Judge."
                ),
            }
    return reviewer_case


def _missing_evidence_is_active(item: Dict[str, Any]) -> bool:
    return str(item.get("status") or "open").strip().lower() not in {
        "resolved",
        "stale",
        "non_actionable",
    }


def build_cross_condition_fact_consistency_audit(
    condition_table: Iterable[Dict[str, Any]],
    *,
    attack_label: str,
) -> Dict[str, Any]:
    """Flag local Judge claims that require evidence resolution before learning.

    This audit does not choose which Judge is correct. It only recognizes a
    narrow, high-value contradiction in the price-manipulation boundary: C2
    denies distorted-value consumption while E1's negative decision explicitly
    relies on that consumption being present. The Reviewer must route the
    conflict to targeted Plan evidence rather than learning either claim as a
    Rule fact.
    """
    label = normalize_attack_label(attack_label, default="")
    rows = {
        str(row.get("condition_id") or row.get("id") or "").strip().upper(): row
        for row in list(condition_table or [])
        if isinstance(row, dict)
    }
    conflicts: List[Dict[str, Any]] = []
    if label == "price_manipulation":
        consumer = dict(rows.get("C2") or {})
        absence_exclusion = dict(rows.get("E1") or {})
        consumer_answer = _normalized_local_answer(consumer.get("answer"))
        exclusion_answer = _normalized_local_answer(absence_exclusion.get("answer"))
        consumer_text = _condition_fact_text(consumer)
        exclusion_text = _condition_fact_text(absence_exclusion)
        exclusion_asserts_consumption = bool(re.search(
            r"(?:causal(?:ly)?|priced off|valuation[- ]dependent|"
            r"determined by|read(?:s|ing)? .{0,80}(?:reserve|price)|"
            r"perturb(?:ed|ation).{0,80}(?:mint|redeem|swap|settlement))",
            exclusion_text,
            flags=re.IGNORECASE,
        ))
        opposing_temporal_claims = bool(
            re.search(
                r"(?:before|preced(?:e|es|ed|ing)|prior to).{0,120}"
                r"(?:distort|perturb|reserve)",
                consumer_text,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"(?:after|just[- ]perturbed|immediately after|"
                r"swap then|getreserves then mint|perturb.{0,80}before)",
                exclusion_text,
                flags=re.IGNORECASE,
            )
        )
        if (
            consumer_answer in {"false", "uncertain"}
            and exclusion_answer == "false"
            and exclusion_asserts_consumption
        ):
            conflicts.append({
                "conflict_id": "C2__E1__distorted_value_consumption",
                "condition_ids": ["C2", "E1"],
                "fact_axis": "distorted_value_consumption_and_temporal_order",
                "conflict_type": (
                    "opposing_temporal_claims"
                    if opposing_temporal_claims
                    else "opposed_local_semantic_claims"
                ),
                "answers": {
                    "C2": consumer_answer,
                    "E1": exclusion_answer,
                },
                "anchor_evidence_ids": list(dict.fromkeys([
                    *list(consumer.get("supporting_evidence_ids") or []),
                    *list(absence_exclusion.get("supporting_evidence_ids") or []),
                ]))[:16],
                "resolution_owner": "plan",
                "required_resolution": (
                    "Reconstruct one candidate-local sequence from distortion "
                    "anchor to consumer call and inspect that call's arguments "
                    "or local context before treating either claim as training fact."
                ),
                "blocks_rule_learning_from_conflict": True,
            })
    return {
        "schema_version": "evotx.cross_condition_fact_consistency_audit.v1",
        "attack_label": label,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "policy": (
            "Audit-only: do not choose a winning Judge claim. Resolve conflicts "
            "through targeted evidence acquisition before semantic Rule updates."
        ),
    }


def _normalized_local_answer(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    text = str(value or "").strip().lower()
    return text if text in {"true", "false", "uncertain"} else "uncertain"


def _condition_fact_text(row: Dict[str, Any]) -> str:
    analysis = dict(row.get("condition_feature_analysis") or {})
    values: List[str] = [str(row.get("reason_short") or "")]
    for key in (
        "matched_features",
        "partial_match_features",
        "missing_required_features",
        "contradicting_features",
        "boundary_notes",
    ):
        values.extend(str(value or "") for value in list(analysis.get(key) or []))
    return " ".join(values)


def _market_mechanism_reviewer_digest(evidence: Dict[str, Any]) -> Dict[str, Any]:
    view = evidence.get("market_mechanism_profile_view")
    if not isinstance(view, dict):
        return {}
    profiles = []
    for row in list(view.get("profiles") or [])[:8]:
        if not isinstance(row, dict):
            continue
        profiles.append({
            "profile_id": row.get("profile_id"),
            "profile_type": row.get("profile_type"),
            "market_source": row.get("market_source"),
            "source_kind": row.get("source_kind"),
            "candidate_strength": row.get("candidate_strength"),
            "source_evidence_ids": list(row.get("source_evidence_ids") or [])[:8],
            "consumer_evidence_ids": list(row.get("consumer_evidence_ids") or [])[:8],
            "outcome_evidence_ids": list(row.get("outcome_evidence_ids") or [])[:8],
            "competing_root_hints": list(row.get("competing_root_hints") or [])[:4],
            "competing_root_evidence_ids": list(row.get("competing_root_evidence_ids") or [])[:6],
        })
    if not profiles:
        return {}
    return {
        "view": "market_mechanism_profile_view",
        "profile_count": int(view.get("profile_count", len(profiles)) or 0),
        "profiles": profiles,
        "notes": list(view.get("notes") or [])[:3],
    }


def _choice_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _review_training_gate_reason(
    review: Dict[str, Any],
    update_target: str,
) -> str:
    """Return a reason when a review must not train this update target."""
    explicit = review.get("training_gate")
    if isinstance(explicit, dict):
        blocked = dict(explicit.get("blocked_reasons") or {})
        reason = str(blocked.get(update_target) or "").strip()
        if reason:
            return reason

    error_type = str(review.get("error_type") or "").strip().upper()
    target_observed = _choice_text(review.get("target_mechanism_observed"))
    gt_supported = _choice_text(review.get("ground_truth_supported_by_packet"))
    generic_only = bool(review.get("generic_symptoms_only"))
    fp_boundary_signal = _review_has_fp_boundary_training_signal(
        review,
        update_target=update_target,
    )
    fn_exclusion_boundary_signal = _review_has_fn_exclusion_boundary_training_signal(
        review,
        update_target=update_target,
    )

    if (
        gt_supported == "no"
        and update_target != "plan"
        and not fp_boundary_signal
    ):
        return "ground_truth_not_supported_by_packet"
    if (
        update_target == "rule"
        and error_type == "FN"
        and target_observed == "no"
        and not fn_exclusion_boundary_signal
    ):
        return "fn_target_mechanism_not_observed"
    if update_target == "rule" and error_type == "FN":
        if target_observed and target_observed != "yes" and not fn_exclusion_boundary_signal:
            return "fn_target_mechanism_not_confirmed_for_rule_update"
        if generic_only and not fn_exclusion_boundary_signal:
            return "fn_generic_symptoms_only"
    return ""


def _review_has_fp_boundary_training_signal(
    review: Dict[str, Any],
    *,
    update_target: str,
) -> bool:
    """Allow FP contrastive tightening even when target mechanism is absent.

    For false positives, the packet often proves that the target mechanism is
    absent or replaced by a non-target root cause. That should block recall
    broadening, but it is exactly the evidence needed for tightening,
    clarification, exclusion, or evidence-strategy boundary updates.
    """
    if str(review.get("error_type") or "").strip().upper() != "FP":
        return False
    target = str(update_target or "").strip().lower()
    if target not in {"rule", "plan"}:
        return False
    allowed_directions = {
        "tighten",
        "refine",
        "add_exclusion",
        "clarify_boundary",
        "change_evidence_strategy",
    }
    for raw_signal in list(review.get("update_signals") or []):
        if not isinstance(raw_signal, dict):
            continue
        signal_target = str(
            raw_signal.get("update_target") or review.get("update_target") or ""
        ).strip().lower()
        if signal_target != target:
            continue
        direction = str(raw_signal.get("direction") or "").strip().lower()
        if direction in {
            "change_default_views",
            "change_followup_views",
            "change_judge_question",
        }:
            direction = "change_evidence_strategy"
        abstract_feature = _cohort_feature_text(raw_signal.get("abstract_feature"))
        if direction in allowed_directions and abstract_feature:
            return True
    return False


def _review_has_fn_exclusion_boundary_training_signal(
    review: Dict[str, Any],
    *,
    update_target: str,
) -> bool:
    """Allow supported FN fixes that only narrow an over-firing exclusion.

    These reviews are often marked generic_symptoms_only because the core target
    mechanism is not packet-confirmed; nevertheless the actionable mistake can
    be that an exclusion condition fired too broadly. Only E-step, supported,
    boundary-clarification signals qualify.
    """
    if str(review.get("error_type") or "").strip().upper() != "FN":
        return False
    if str(update_target or "").strip().lower() != "rule":
        return False
    overfired_exclusions: set[str] = set()
    for diagnosis in list(review.get("condition_diagnosis") or []):
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
    allowed_directions = {"tighten", "clarify_boundary", "narrow_exclusion"}
    for raw_signal in list(review.get("update_signals") or []):
        if not isinstance(raw_signal, dict):
            continue
        signal_target = str(
            raw_signal.get("update_target") or review.get("update_target") or ""
        ).strip().lower()
        condition_id = str(raw_signal.get("condition_id") or "").strip().upper()
        status = str(raw_signal.get("generalization_status") or "").strip().lower()
        direction = str(raw_signal.get("direction") or "").strip().lower()
        abstract_feature = _cohort_feature_text(raw_signal.get("abstract_feature"))
        if (
            signal_target == "rule"
            and condition_id in overfired_exclusions
            and status == "supported"
            and direction in allowed_directions
            and abstract_feature
        ):
            return True
    return False


def _review_allows_training_update(
    review: Dict[str, Any],
    update_target: str,
) -> bool:
    return not _review_training_gate_reason(review, update_target)


def _review_requested_training_targets(review: Dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    update_target = str(review.get("update_target") or "").strip().lower()
    if update_target in {"rule", "plan"}:
        targets.add(update_target)
    if bool(review.get("should_update_rule")):
        targets.add("rule")
    if bool(review.get("should_update_plan_strategy")):
        targets.add("plan")
    for signal in list(review.get("update_signals") or []):
        if not isinstance(signal, dict):
            continue
        signal_target = str(signal.get("update_target") or "").strip().lower()
        if signal_target in {"rule", "plan"}:
            targets.add(signal_target)
    return targets


def _review_blocks_requested_training(review: Dict[str, Any]) -> bool:
    requested_targets = _review_requested_training_targets(review)
    if not requested_targets:
        return False
    blocked = dict((review.get("training_gate") or {}).get("blocked_reasons") or {})
    return all(target in blocked for target in requested_targets)


def _annotate_review_training_gate(review: Dict[str, Any]) -> Dict[str, Any]:
    review_execution = dict(review.get("review_execution") or {})
    if str(review_execution.get("status") or "").strip().lower() == "failed":
        review["training_gate"] = {
            **dict(review.get("training_gate") or {}),
            "rule_allowed": False,
            "plan_allowed": False,
            "blocked_reasons": {
                "rule": "review_execution_failed",
                "plan": "review_execution_failed",
            },
            "requested_owner_targets": [],
            "blocked_owner_targets": [],
            "trainable_owner_targets": [],
            "training_disposition": "review_execution_failed",
        }
        review["do_not_train"] = True
        review["do_not_train_reasons"] = {
            "rule": "review_execution_failed",
            "plan": "review_execution_failed",
        }
        return review

    blocked_reasons: Dict[str, str] = {}
    for target in ("rule", "plan"):
        reason = _review_training_gate_reason(review, target)
        if reason:
            blocked_reasons[target] = reason
    review["training_gate"] = {
        **dict(review.get("training_gate") or {}),
        "rule_allowed": "rule" not in blocked_reasons,
        "plan_allowed": "plan" not in blocked_reasons,
        "blocked_reasons": blocked_reasons,
    }
    requested = _review_requested_training_targets(review)
    blocked_requested = sorted(requested & set(blocked_reasons))
    trainable_requested = sorted(requested - set(blocked_reasons))
    if requested and not trainable_requested:
        disposition = "non_targetable"
    elif "rule" in blocked_requested and "plan" in trainable_requested:
        disposition = "evidence_route_only"
    elif blocked_requested:
        disposition = "owner_limited"
    elif requested:
        disposition = "trainable"
    else:
        disposition = "no_update_requested"
    review["training_gate"].update({
        "requested_owner_targets": sorted(requested),
        "blocked_owner_targets": blocked_requested,
        "trainable_owner_targets": trainable_requested,
        "training_disposition": disposition,
    })
    review["do_not_train"] = _review_blocks_requested_training(review)
    if review["do_not_train"]:
        review["do_not_train_reasons"] = {
            target: reason
            for target, reason in blocked_reasons.items()
            if target in requested
        }
    return review


def _review_owner_projection(
    review: Dict[str, Any],
    owner: str,
    *,
    source_review_index: int,
) -> Optional[Dict[str, Any]]:
    canonical_signals = [
        dict(signal)
        for signal in list(review.get("update_signals") or [])
        if isinstance(signal, dict) and str(signal.get("signal_id") or "")
    ]
    if canonical_signals:
        owner_signals = [
            signal
            for signal in canonical_signals
            if str(signal.get("update_target") or "").strip().lower() == owner
        ]
        if not owner_signals:
            return None
        projected = deepcopy(review)
        projected["source_review_index"] = source_review_index
        projected["update_target"] = owner
        projected["update_signals"] = owner_signals
        projected["diagnosis_routes"] = [
            dict(route)
            for route in list(review.get("diagnosis_routes") or [])
            if isinstance(route, dict)
            and str(route.get("owner") or "").strip().lower() == owner
        ]
        projected["owner_targets"] = [owner]
        projected["owner_projection"] = {
            "schema_version": "evotx.canonical_owner_projection.v1",
            "authority": "update_signal_matrix",
            "owner": owner,
            "source_review_index": source_review_index,
            "signal_ids": [
                str(signal.get("signal_id") or "") for signal in owner_signals
            ],
        }
        projected["should_update_rule"] = owner == "rule"
        projected["should_update_plan_strategy"] = owner == "plan"
        projected["should_update_packet_builder"] = owner == "packet"
        projected["should_fix_runtime"] = owner == "runtime"
        for candidate in ("rule", "plan", "packet", "runtime"):
            if candidate != owner:
                projected[f"{candidate}_patch_suggestion"] = {"action": "none"}
        return projected

    projection = project_review_for_owner(
        review,
        owner,
        source_review_index=source_review_index,
    )
    if projection is not None:
        return projection

    # Backward compatibility for saved pre-Phase-2 review bundles and tests.
    # Once diagnosis_routes exist, absence of an owner route is authoritative.
    if list(review.get("diagnosis_routes") or []):
        return None
    requested = str(review.get("update_target") or "").strip().lower() == owner
    requested = requested or any(
        isinstance(signal, dict)
        and str(signal.get("update_target") or "").strip().lower() == owner
        for signal in list(review.get("update_signals") or [])
    )
    requested = requested or bool(
        review.get({
            "rule": "should_update_rule",
            "plan": "should_update_plan_strategy",
            "packet": "should_update_packet_builder",
            "runtime": "should_fix_runtime",
        }[owner])
    )
    if not requested:
        return None
    legacy = deepcopy(review)
    legacy["source_review_index"] = source_review_index
    legacy["update_target"] = owner
    legacy["owner_projection"] = {
        "schema_version": "legacy_pre_phase2",
        "owner": owner,
        "source_review_index": source_review_index,
        "route_ids": [],
    }
    return legacy


def build_round_review_bundle(
    rule: Any,
    current_summary: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    slim_results: List[Dict[str, Any]],
    *,
    negative_training_mode: str = "mixed",
    negative_training_mode_source: str = "default",
    cohort_signal_summary: Optional[Dict[str, Any]] = None,
    guard_slim_results: Optional[List[Dict[str, Any]]] = None,
    plan_evidence_audit: Optional[Dict[str, Any]] = None,
    case_boundary_context: Optional[Dict[str, Any]] = None,
    rejected_update_memory: Optional[Dict[str, Any]] = None,
    enable_advanced_candidate_lifecycles: bool = True,
) -> Dict[str, Any]:
    """Build the compact, gated input for RuleUpdater."""
    resolved_negative_training_mode = normalize_negative_training_mode(
        negative_training_mode
    )
    rule_dict = rule.to_dict() if hasattr(rule, "to_dict") else dict(rule or {})
    normalized_reviews = [
        _annotate_review_training_gate(dict(r or {})) for r in reviews
    ]
    review_execution_failures = [
        {
            "source_review_index": review_index,
            "tx_hash": str(review.get("tx_hash") or ""),
            "error_type": str(review.get("error_type") or "unknown"),
            **dict(review.get("review_execution") or {}),
        }
        for review_index, review in enumerate(normalized_reviews)
        if str(
            (review.get("review_execution") or {}).get("status") or ""
        ).strip().lower()
        == "failed"
    ]
    cohort_summary = dict(
        cohort_signal_summary
        or build_cohort_signal_summary(
            slim_results,
            guard_slim_results=guard_slim_results,
        )
    )
    plan_audit = dict(
        plan_evidence_audit
        or build_plan_evidence_audit(
            slim_results,
            guard_slim_results=guard_slim_results,
        )
    )
    update_signal_matrix = build_update_signal_matrix(
        normalized_reviews,
        cohort_summary,
        plan_evidence_audit=plan_audit,
        enable_advanced_candidate_lifecycles=(
            enable_advanced_candidate_lifecycles
        ),
    )
    normalized_reviews = _attach_matrix_signals_to_reviews(
        normalized_reviews,
        update_signal_matrix,
    )
    boundary_context = dict(
        case_boundary_context
        or build_case_boundary_context(
            slim_results,
            guard_slim_results=guard_slim_results,
            cohort_signal_summary=cohort_summary,
            plan_evidence_audit=plan_audit,
        )
    )
    rejected_memory = normalize_rejected_update_memory(rejected_update_memory)
    use_signal_gate = bool(cohort_summary.get("schema_version"))
    owner_projections = {
        owner: [
            projection
            for review_index, review in enumerate(normalized_reviews)
            for projection in [
                _review_owner_projection(
                    review,
                    owner,
                    source_review_index=review_index,
                )
            ]
            if projection is not None
        ]
        for owner in ("rule", "plan", "packet", "runtime")
    }
    actionable_rule_reviews = [
        projection
        for projection in owner_projections["rule"]
        if bool(projection.get("should_update_rule"))
        and (
            not use_signal_gate
            or _matrix_has_supported_update_signal(
                update_signal_matrix,
                source_review_index=int(
                    projection.get("source_review_index", -1)
                ),
                update_target="rule",
            )
        )
    ]
    actionable_plan_reviews = [
        projection
        for projection in owner_projections["plan"]
        if bool(projection.get("should_update_plan_strategy"))
        and (
            review_has_operation(
                projection,
                "plan",
                "restore_question_from_rule",
            )
            or (
                not use_signal_gate
                or _matrix_has_actionable_update_signal(
                    update_signal_matrix,
                    source_review_index=int(
                        projection.get("source_review_index", -1)
                    ),
                    update_target="plan",
                    allow_experimental_plan=True,
                )
            )
        )
    ]
    engineering_reviews = [
        *owner_projections["packet"],
        *owner_projections["runtime"],
    ]
    # Preserve Phase 2 owner projections instead of comparing projected
    # objects with structurally different raw reviews. The old comparison made
    # nearly every raw review look non-Rule and leaked mixed ownership back into
    # downstream Plan handling.
    non_rule_reviews = [
        *owner_projections["plan"],
        *owner_projections["packet"],
        *owner_projections["runtime"],
    ]
    do_not_train_reviews = [
        r
        for r in normalized_reviews
        if bool(r.get("do_not_train")) or _review_blocks_requested_training(r)
    ]
    deferred_logic_failures = [
        {
            "review_index": review_index,
            "tx_hash": review.get("tx_hash", ""),
            "error_type": review.get("error_type", "unknown"),
            "reason": "alternate_or_branch_not_expressible_by_current_and_not_logic",
            "root_causes": [
                cause for cause in list(review.get("root_causes") or [])
                if isinstance(cause, dict)
                and str(cause.get("category") or "").strip().lower()
                == "deferred_logic_structure_failure"
            ],
        }
        for review_index, review in enumerate(normalized_reviews)
        if _review_has_deferred_logic_structure_failure(review)
    ]

    error_summary = {
        "total_reviews": len(normalized_reviews),
        "fp": sum(1 for r in normalized_reviews if r.get("error_type") == "FP"),
        "fn": sum(1 for r in normalized_reviews if r.get("error_type") == "FN"),
        "fp_negative_benign": sum(
            1
            for r in normalized_reviews
            if r.get("error_type") == "FP" and r.get("negative_kind") == "benign"
        ),
        "fp_negative_other_attack": sum(
            1
            for r in normalized_reviews
            if r.get("error_type") == "FP" and r.get("negative_kind") == "other_attack"
        ),
        "rule_actionable": len(actionable_rule_reviews),
        "plan_actionable": len(actionable_plan_reviews),
        "packet_actionable": len(owner_projections["packet"]),
        "runtime_bugs": len(owner_projections["runtime"]),
        "do_not_train": len(do_not_train_reviews),
        "update_signal_count": len(list(update_signal_matrix.get("signals") or [])),
        "supported_update_signal_count": len(
            list(update_signal_matrix.get("supported_signal_ids") or [])
        ),
        "experimental_plan_signal_count": len(
            list(update_signal_matrix.get("experimental_plan_signal_ids") or [])
        ),
        "experimental_rule_signal_count": len(
            list(update_signal_matrix.get("experimental_rule_signal_ids") or [])
        ),
        "conflicted_update_signal_count": len(
            list(update_signal_matrix.get("conflicted_signal_ids") or [])
        ),
        "insufficient_update_signal_count": len(
            list(update_signal_matrix.get("insufficient_signal_ids") or [])
        ),
        "joint_resolution_group_count": len(
            list(update_signal_matrix.get("joint_resolution_groups") or [])
        ),
        "deferred_logic_structure_failure_count": len(
            deferred_logic_failures
        ),
        "review_execution_failure_count": len(review_execution_failures),
        "case_boundary_context_available": bool(
            boundary_context.get("schema_version")
        ),
    }
    canonical_signal_contract = {
        "schema_version": "evotx.canonical_update_signal_contract.v1",
        "authority": "update_signal_matrix",
        "owner_field": "update_target",
        "lifecycle_field": "generalization_status",
        "owner_projection_policy": "mechanical_filter_by_signal_id",
        "immutable_after_bundle_construction": True,
        "signal_ids": [
            str(signal.get("signal_id") or "")
            for signal in list(update_signal_matrix.get("signals") or [])
            if isinstance(signal, dict) and str(signal.get("signal_id") or "")
        ],
    }

    return {
        "current_rule": rule_dict,
        "current_summary": dict(current_summary or {}),
        "error_summary": error_summary,
        "negative_training_mode": resolved_negative_training_mode,
        "negative_training_mode_source": str(
            negative_training_mode_source or "default"
        ),
        "cohort_signal_summary": cohort_summary,
        "plan_evidence_audit": plan_audit,
        "case_boundary_context": boundary_context,
        "rejected_update_memory": rejected_memory,
        "update_signal_matrix": update_signal_matrix,
        "canonical_signal_contract": canonical_signal_contract,
        "diagnosis_routing_audit": build_diagnosis_routing_audit(
            normalized_reviews
        ),
        "owner_projections": owner_projections,
        "deferred_logic_structure_failures": deferred_logic_failures,
        "actionable_rule_reviews": actionable_rule_reviews,
        "actionable_plan_reviews": actionable_plan_reviews,
        "engineering_reviews": engineering_reviews,
        "do_not_train_reviews": do_not_train_reviews,
        "review_execution_failures": review_execution_failures,
        "non_rule_reviews": non_rule_reviews,
        "slim_result_count": len(slim_results or []),
        "update_constraints": [
            "Do not update the rule for runtime emit-logic bugs.",
            "Do not update the rule for packet-only missing view problems unless the semantic condition is also wrong.",
            "Do not broaden the rule into generic transaction anomaly detection.",
            "Do not train semantic rule updates from FN reviews where the target mechanism is not observed, the ground truth is not packet-supported, or only generic symptoms were observed.",
            "For negative_other samples, treat FP as target-vs-non-target boundary errors; do not describe other attack families as benign behavior.",
            negative_training_mode_guidance(resolved_negative_training_mode),
            "Every rule or plan update must preserve case_boundary_context: correct target positives remain detectable, and correct negatives/guard cases keep their target-vs-non-target boundary.",
            "Use rejected_update_memory as refinement feedback, not an eligibility gate; exact duplicate candidates are removed deterministically downstream.",
            "Do not add packet view names, tool names, evidence_hints, concrete tx hashes, addresses, call ids, transfer ids, event ids, or storage slots to the rule.",
            "Current Rule logic is AND + NOT(exclusion). Record alternate sufficient paths as deferred_logic_structure_failure instead of weakening condition wording to imitate OR.",
        ],
    }


def normalize_rejected_update_memory(
    memory: Optional[Dict[str, Any]],
    *,
    max_entries: int = 20,
) -> Dict[str, Any]:
    """Return a bounded, prompt-safe memory of rejected abstract update directions."""
    if not isinstance(memory, dict):
        entries: List[Dict[str, Any]] = []
    else:
        entries = []
        for raw in list(memory.get("entries") or [])[-max_entries:]:
            if not isinstance(raw, dict):
                continue
            abstract_feature = _cohort_feature_text(raw.get("abstract_feature"))
            if not abstract_feature:
                continue
            dependency = raw.get("condition_evidence_dependency")
            dependency_capability = ""
            if isinstance(dependency, dict):
                dependency_capability = str(
                    dependency.get("capability_id") or ""
                ).strip().lower()
            entries.append({
                "memory_id": str(raw.get("memory_id") or "")[:48],
                "attack_label": normalize_attack_label(
                    str(raw.get("attack_label") or ""),
                    default="",
                ),
                "source_round": int(raw.get("source_round") or 0),
                "source_candidate": str(raw.get("source_candidate") or "")[:80],
                "source_signal_id": str(raw.get("source_signal_id") or "")[:64],
                "source_error_type": str(raw.get("source_error_type") or "")[:16],
                "update_kind": str(raw.get("update_kind") or "")[:24],
                "condition_id": str(raw.get("condition_id") or "")[:32],
                "direction": str(raw.get("direction") or "")[:32],
                "strategy_operation": str(
                    raw.get("strategy_operation") or raw.get("operation") or ""
                )[:64],
                "capability_id": str(
                    raw.get("capability_id") or dependency_capability or ""
                )[:96],
                "abstract_feature": abstract_feature,
                "rejection_type": str(raw.get("rejection_type") or "")[:64],
                "reject_reason": " ".join(
                    str(raw.get("reject_reason") or "").split()
                )[:240],
                "boundary_hint": " ".join(
                    str(raw.get("boundary_hint") or "").split()
                )[:240],
                "rejection_causal_audit": _compact_rejection_causal_audit(
                    raw.get("rejection_causal_audit")
                ),
            })
    preflight_gaps: List[Dict[str, Any]] = []
    construction_rejections: List[Dict[str, Any]] = []
    active_refinement_feedback: Dict[str, Any] = {}
    if isinstance(memory, dict):
        for raw in list(memory.get("preflight_gaps") or [])[-12:]:
            if not isinstance(raw, dict):
                continue
            uncovered = sorted({
                str(value or "").strip().upper()[:32]
                for value in list(raw.get("uncovered_blocker_ids") or [])
                if str(value or "").strip()
            })
            uncovered_evidence = []
            for item in list(raw.get("uncovered_evidence_capabilities") or [])[:12]:
                if not isinstance(item, dict):
                    continue
                condition_id = str(item.get("condition_id") or "").strip().upper()
                capability_id = str(item.get("capability_id") or "").strip().lower()
                if not condition_id or not capability_id:
                    continue
                uncovered_evidence.append({
                    "condition_id": condition_id[:32],
                    "capability_id": capability_id[:96],
                    "uncovered_components": sorted({
                        str(value or "").strip().lower()[:48]
                        for value in list(item.get("uncovered_components") or [])
                        if str(value or "").strip()
                    })[:6],
                })
            if not uncovered and not uncovered_evidence:
                continue
            preflight_gaps.append({
                "source_round": int(raw.get("source_round") or 0),
                "source_candidate": str(raw.get("source_candidate") or "")[:80],
                "update_kind": str(raw.get("update_kind") or "")[:24],
                "changed_condition_ids": sorted({
                    str(value or "").strip().upper()[:32]
                    for value in list(raw.get("changed_condition_ids") or [])
                    if str(value or "").strip()
                })[:12],
                "uncovered_blocker_ids": uncovered[:12],
                **(
                    {"uncovered_evidence_capabilities": uncovered_evidence}
                    if uncovered_evidence
                    else {}
                ),
                "reason": " ".join(str(raw.get("reason") or "").split())[:240],
                "required_next_direction": " ".join(
                    str(raw.get("required_next_direction") or "").split()
                )[:280],
            })
        for raw in list(memory.get("construction_rejections") or [])[-12:]:
            if not isinstance(raw, dict):
                continue
            construction_rejections.append({
                "source_round": int(raw.get("source_round") or 0),
                "owner": str(raw.get("owner") or "plan")[:24],
                "rejection_stage": str(
                    raw.get("rejection_stage") or "candidate_construction"
                )[:48],
                "reason": " ".join(str(raw.get("reason") or "").split())[:240],
                "scope_target_steps": sorted({
                    str(value or "").strip().upper()[:32]
                    for value in list(raw.get("scope_target_steps") or [])
                    if str(value or "").strip()
                })[:12],
                "strategy_violations": list(
                    raw.get("strategy_violations") or []
                )[:8],
                "remaining_violations": list(
                    raw.get("remaining_violations") or []
                )[:8],
                "retained_strategy_deltas": list(
                    raw.get("retained_strategy_deltas") or []
                )[:8],
                "post_safety_budget_audits": list(
                    raw.get("post_safety_budget_audits") or []
                )[:8],
                "candidate_survived_construction": bool(
                    raw.get("candidate_survived_construction")
                ),
                "accepted_patch_signal_ids": [
                    str(value)[:96]
                    for value in list(
                        raw.get("accepted_patch_signal_ids") or []
                    )[:16]
                ],
                "rejected_patch_components": [
                    {
                        "patch_index": int(item.get("patch_index") or 0),
                        "condition_id": str(
                            item.get("condition_id") or ""
                        )[:32],
                        "signal_ids": [
                            str(value)[:96]
                            for value in list(item.get("signal_ids") or [])[:8]
                        ],
                        "strategy_operation": str(
                            item.get("strategy_operation") or ""
                        )[:64],
                        "stage": str(item.get("stage") or "")[:64],
                        "reason": " ".join(
                            str(item.get("reason") or "").split()
                        )[:240],
                    }
                    for item in list(
                        raw.get("rejected_patch_components") or []
                    )[:8]
                    if isinstance(item, dict)
                ],
                "error": " ".join(str(raw.get("error") or "").split())[:240],
            })
        raw_feedback = memory.get("active_refinement_feedback")
        if isinstance(raw_feedback, dict):
            raw_plan_activation = raw_feedback.get("plan_effect_activation")
            if not isinstance(raw_plan_activation, dict):
                raw_plan_activation = {}
            active_refinement_feedback = {
                "schema_version": str(
                    raw_feedback.get("schema_version")
                    or "evotx.refinement_feedback.v1"
                )[:64],
                "source_round": int(raw_feedback.get("source_round") or 0),
                "source_candidate": str(
                    raw_feedback.get("source_candidate") or ""
                )[:80],
                "source_candidate_status": str(
                    raw_feedback.get("source_candidate_status") or ""
                )[:48],
                "source_update_kind": str(
                    raw_feedback.get("source_update_kind") or ""
                )[:24],
                "source_signal_ids": [
                    str(value)[:80]
                    for value in list(raw_feedback.get("source_signal_ids") or [])[:8]
                    if str(value)
                ],
                "signal_directions": [
                    str(value).strip().lower()[:80]
                    for value in list(raw_feedback.get("signal_directions") or [])[:8]
                    if str(value).strip()
                ],
                "rejection_reason": " ".join(
                    str(raw_feedback.get("rejection_reason") or "").split()
                )[:240],
                "refinable": bool(raw_feedback.get("refinable")),
                "refinability_reason": " ".join(
                    str(raw_feedback.get("refinability_reason") or "").split()
                )[:240],
                "improved_targets": [
                    str(value).strip().lower()[:96]
                    for value in list(raw_feedback.get("improved_targets") or [])[:8]
                    if str(value).strip()
                ],
                "unattributed_improved_targets": [
                    str(value).strip().lower()[:96]
                    for value in list(
                        raw_feedback.get("unattributed_improved_targets") or []
                    )[:8]
                    if str(value).strip()
                ],
                "preserved_behaviors": [
                    str(value).strip().lower()[:96]
                    for value in list(raw_feedback.get("preserved_behaviors") or [])[:8]
                    if str(value).strip()
                ],
                "regressions": [
                    {
                        "tx_hash": str(item.get("tx_hash") or "").strip().lower()[:96],
                        "role": str(item.get("role") or "")[:40],
                        "ground_truth": str(item.get("ground_truth") or "")[:24],
                        "baseline_verdict": str(
                            item.get("baseline_verdict") or ""
                        )[:24],
                        "candidate_verdict": str(
                            item.get("candidate_verdict") or ""
                        )[:24],
                        "condition_flips": [
                            {
                                "condition_id": str(
                                    flip.get("condition_id") or ""
                                )[:32],
                                "baseline_answer": flip.get("baseline_answer"),
                                "candidate_answer": flip.get("candidate_answer"),
                            }
                            for flip in list(item.get("condition_flips") or [])[:8]
                            if isinstance(flip, dict)
                        ],
                    }
                    for item in list(raw_feedback.get("regressions") or [])[:8]
                    if isinstance(item, dict)
                ],
                "affected_components": [
                    {
                        "kind": str(item.get("kind") or "")[:24],
                        "condition_id": str(item.get("condition_id") or "")[:32],
                        "operation": str(item.get("operation") or "")[:64],
                        "capability_id": str(item.get("capability_id") or "")[:96],
                    }
                    for item in list(raw_feedback.get("affected_components") or [])[:12]
                    if isinstance(item, dict)
                ],
                "desired_refinement": [
                    " ".join(str(value).split())[:260]
                    for value in list(raw_feedback.get("desired_refinement") or [])[:8]
                    if str(value).strip()
                ],
                "next_allowed_refinement_kind": str(
                    raw_feedback.get("next_allowed_refinement_kind") or ""
                )[:80],
                "must_preserve_candidate_effect": bool(
                    raw_feedback.get("must_preserve_candidate_effect")
                ),
                "plan_effect_activation": {
                    "changed_condition_ids": [
                        str(value).strip().upper()[:32]
                        for value in list(
                            raw_plan_activation.get("changed_condition_ids") or []
                        )[:8]
                        if str(value).strip()
                    ],
                    "required_routes_by_condition": {
                        str(condition_id).strip().upper()[:32]: [
                            str(route).strip()[:96]
                            for route in list(routes or [])[:8]
                            if str(route).strip()
                        ]
                        for condition_id, routes in dict(
                            raw_plan_activation.get("required_routes_by_condition")
                            or {}
                        ).items()
                        if str(condition_id).strip()
                    },
                    "attributed_improved_txs": [
                        str(value).strip().lower()[:96]
                        for value in list(
                            raw_plan_activation.get("attributed_improved_txs") or []
                        )[:8]
                        if str(value).strip()
                    ],
                    "unattributed_improved_txs": [
                        str(value).strip().lower()[:96]
                        for value in list(
                            raw_plan_activation.get("unattributed_improved_txs") or []
                        )[:8]
                        if str(value).strip()
                    ],
                    "all_improvements_attributed": bool(
                        raw_plan_activation.get("all_improvements_attributed")
                    ),
                },
            }
    return {
        "schema_version": REJECTED_UPDATE_MEMORY_SCHEMA_VERSION,
        "scope": "episode_rounds_compact",
        "max_entries": int(max_entries),
        "entries": entries[-max_entries:],
        "preflight_gaps": preflight_gaps,
        "construction_rejections": construction_rejections,
        **(
            {"active_refinement_feedback": active_refinement_feedback}
            if active_refinement_feedback
            else {}
        ),
        "policy": [
            "This is rejected-direction memory, not transaction evidence.",
            "Do not repeat the same direction with the same strategy/capability unless a new supported signal adds the missing boundary.",
            "A refined boundary, new evidence capability, or new mechanism may be tried even when it affects the same condition.",
            "Use it as a taboo list for abstract repairs that previously harmed recall/precision/guard boundaries.",
            "preflight_gaps are coverage requirements, not taboo directions; a later portfolio should resolve the listed blockers rather than avoid them.",
            "construction_rejections describe updater-stage failures; preserve safe retained deltas while repairing their recorded budget or safety conflict.",
            "active_refinement_feedback guides the next round while the accepted incumbent remains authoritative unless an explicitly enabled portfolio policy selects another search base; it is not promoted knowledge.",
        ],
        "contains_transaction_identifiers": False,
    }


def _compact_rejection_causal_audit(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    def compact_regressions(key: str) -> List[Dict[str, Any]]:
        compacted: List[Dict[str, Any]] = []
        for raw in list(value.get(key) or [])[:8]:
            if not isinstance(raw, dict):
                continue
            compacted.append({
                "role": str(raw.get("role") or "")[:40],
                "ground_truth": str(raw.get("ground_truth") or "")[:24],
                "baseline_verdict": str(raw.get("baseline_verdict") or "")[:24],
                "candidate_verdict": str(raw.get("candidate_verdict") or "")[:24],
                "condition_flips": [
                    {
                        "condition_id": str(flip.get("condition_id") or "")[:32],
                        "baseline_answer": flip.get("baseline_answer"),
                        "candidate_answer": flip.get("candidate_answer"),
                    }
                    for flip in list(raw.get("condition_flips") or [])[:8]
                    if isinstance(flip, dict)
                ],
            })
        return compacted

    regressions = compact_regressions("protected_regressions")
    unattributed_runtime_regressions = compact_regressions(
        "unattributed_runtime_regressions"
    )
    return {
        "candidate_terminal_reason": str(
            value.get("candidate_terminal_reason") or ""
        )[:64],
        "condition_scope": str(value.get("condition_scope") or "")[:32],
        "condition_effect": str(value.get("condition_effect") or "")[:32],
        "candidate_update_kind": str(
            value.get("candidate_update_kind") or ""
        )[:24],
        "rule_condition_ids": [
            str(item)[:32]
            for item in list(value.get("rule_condition_ids") or [])[:12]
        ],
        "plan_condition_ids": [
            str(item)[:32]
            for item in list(value.get("plan_condition_ids") or [])[:12]
        ],
        "rule_components": [
            {
                "delta_id": str(item.get("delta_id") or "")[:96],
                "condition_id": str(item.get("condition_id") or "")[:32],
                "operation": str(item.get("operation") or "")[:48],
                "semantic_direction": str(
                    item.get("semantic_direction") or ""
                )[:24],
                "source_signal_ids": [
                    str(signal_id)[:96]
                    for signal_id in list(
                        item.get("source_signal_ids") or []
                    )[:8]
                ],
            }
            for item in list(value.get("rule_components") or [])[:12]
            if isinstance(item, dict)
        ],
        "plan_source_signal_ids": [
            str(item)[:96]
            for item in list(value.get("plan_source_signal_ids") or [])[:12]
        ],
        "rejection_gate": str(value.get("rejection_gate") or "")[:64],
        "regression_delta": {
            str(key)[:32]: raw_value
            for key, raw_value in dict(
                value.get("regression_delta") or {}
            ).items()
            if isinstance(raw_value, (bool, int, float, str))
        },
        "protected_regression_count": len(regressions),
        "protected_regressions": regressions,
        "unattributed_runtime_regressions": unattributed_runtime_regressions,
    }


def build_case_boundary_context(
    slim_results: Iterable[Dict[str, Any]],
    *,
    guard_slim_results: Optional[Iterable[Dict[str, Any]]] = None,
    cohort_signal_summary: Optional[Dict[str, Any]] = None,
    plan_evidence_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a compact preservation contract from correctly handled cases.

    The context is intentionally aggregate-only. It gives Reviewer/Updater the
    shape of ordinary/correct cases that must not be broken, without adding
    concrete transaction identifiers or full packets back into the prompt.
    """
    training_slims = [
        build_slim_result(raw) for raw in list(slim_results or [])
    ]
    guard_slims = [
        build_slim_result(raw) for raw in list(guard_slim_results or [])
    ]
    cohort = dict(
        cohort_signal_summary
        or build_cohort_signal_summary(
            training_slims,
            guard_slim_results=guard_slims,
        )
    )
    plan_audit = dict(
        plan_evidence_audit
        or build_plan_evidence_audit(
            training_slims,
            guard_slim_results=guard_slims,
        )
    )
    plan_conditions = dict((plan_audit.get("conditions") or {}))

    condition_boundaries: Dict[str, Any] = {}
    for condition_id, condition in sorted(
        dict(cohort.get("conditions") or {}).items()
    )[:CASE_BOUNDARY_CONDITION_LIMIT]:
        if not isinstance(condition, dict):
            continue
        groups: Dict[str, Any] = {}
        for group_name, group in sorted(
            dict(condition.get("groups") or {}).items()
        ):
            if group_name not in {
                "correct_positive",
                "correct_negative",
                "correct_guard",
            } or not isinstance(group, dict):
                continue
            groups[group_name] = _case_boundary_group_summary(
                group,
                group_name=group_name,
            )
        if not groups:
            continue
        plan_condition = dict(plan_conditions.get(condition_id) or {})
        condition_boundaries[condition_id] = {
            "is_exclusion": bool(condition.get("is_exclusion")),
            "preserved_groups": groups,
            "observed_successful_followup_views": list(
                plan_condition.get("observed_successful_followup_views") or []
            )[:8],
        }

    counts = dict(cohort.get("counts") or {})
    return {
        "schema_version": CASE_BOUNDARY_CONTEXT_SCHEMA_VERSION,
        "source": "correct_training_and_guard_cases_only",
        "counts": counts,
        "contrastive_label_coverage": dict(
            cohort.get("contrastive_label_coverage") or {}
        ),
        "protected_positive_signals": list(
            cohort.get("protected_positive_signals") or []
        )[:24],
        "protected_boundary_signals": list(
            cohort.get("protected_boundary_signals") or []
        )[:24],
        "condition_boundaries": condition_boundaries,
        "preservation_policy": [
            "FN repair may broaden only through target-specific features compatible with correct_positive groups.",
            "FP repair may tighten only through target-boundary features compatible with correct_negative and correct_guard groups.",
            "Plan changes may add evidence routes, but must not make a previously correct condition lose all observed evidence paths.",
        ],
        "contains_transaction_identifiers": False,
    }


def _case_boundary_group_summary(
    group: Dict[str, Any],
    *,
    group_name: str,
) -> Dict[str, Any]:
    feature_summary: Dict[str, Any] = {}
    for feature_type, observations in dict(group.get("features") or {}).items():
        items = [
            {
                "feature_ref": item.get("feature_ref", ""),
                "feature": item.get("feature", ""),
                "support_count": int(item.get("support_count", 0) or 0),
            }
            for item in list(observations or [])
            if isinstance(item, dict) and str(item.get("feature") or "").strip()
        ][:CASE_BOUNDARY_FEATURE_LIMIT_PER_TYPE]
        if items:
            feature_summary[feature_type] = items
    return {
        "role": _case_boundary_group_role(group_name),
        "case_count": int(group.get("case_count", 0) or 0),
        "answer_counts": dict(group.get("answer_counts") or {}),
        "confidence_counts": dict(group.get("confidence_counts") or {}),
        "features": feature_summary,
        "views": list(group.get("views") or [])[:8],
        "tool_statuses": list(group.get("tool_statuses") or [])[:6],
    }


def _case_boundary_group_role(group_name: str) -> str:
    if group_name == "correct_positive":
        return "preserve_target_positive_mechanisms"
    if group_name == "correct_guard":
        return "preserve_hard_negative_guard_boundaries"
    return "preserve_correct_negative_boundaries"


def build_cohort_signal_summary(
    slim_results: Iterable[Dict[str, Any]],
    *,
    guard_slim_results: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    training_slims = [
        build_slim_result(raw) for raw in list(slim_results or [])
    ]
    contrastive_label_coverage = _contrastive_label_coverage(training_slims)
    groups: Dict[str, List[Dict[str, Any]]] = {
        "correct_positive": [],
        "correct_negative": [],
        "correct_guard": [],
    }
    for slim in training_slims:
        ground_truth = str(slim.get("ground_truth") or "").lower().strip()
        verdict = str(slim.get("predicted_verdict") or "").lower().strip()
        if ground_truth == "attack" and verdict == "attack":
            groups["correct_positive"].append(slim)
        elif ground_truth == "benign" and verdict == "benign":
            groups["correct_negative"].append(slim)
    for raw in list(guard_slim_results or []):
        slim = build_slim_result(raw)
        if (
            str(slim.get("ground_truth") or "").lower().strip() == "benign"
            and str(slim.get("predicted_verdict") or "").lower().strip() == "benign"
        ):
            groups["correct_guard"].append(slim)

    conditions: Dict[str, Dict[str, Any]] = {}
    for group_name, cases in groups.items():
        for slim in cases:
            tools_by_condition: Dict[str, List[Dict[str, Any]]] = {}
            for tool in list(slim.get("tool_call_summary", []) or []):
                condition_id = str(
                    tool.get("condition_id") or tool.get("judge_id") or ""
                ).strip()
                if condition_id:
                    tools_by_condition.setdefault(condition_id, []).append(tool)
            for row in list(slim.get("condition_table", []) or []):
                condition_id = str(
                    row.get("condition_id") or row.get("id") or ""
                ).strip()
                if not condition_id:
                    continue
                condition = conditions.setdefault(condition_id, {
                    "is_exclusion": bool(row.get("is_exclusion")),
                    "groups": {},
                })
                stats = condition["groups"].setdefault(group_name, {
                    "case_count": 0,
                    "answer_counts": Counter(),
                    "confidence_counts": Counter(),
                    "features": {
                        "matched_features": Counter(),
                        "partial_match_features": Counter(),
                        "missing_required_features": Counter(),
                        "contradicting_features": Counter(),
                        "boundary_notes": Counter(),
                    },
                    "view_counts": Counter(),
                    "tool_status_counts": Counter(),
                })
                stats["case_count"] += 1
                stats["answer_counts"][_cohort_answer_key(row.get("answer"))] += 1
                stats["confidence_counts"][
                    str(row.get("confidence") or "unknown").lower().strip()
                ] += 1
                feature_analysis = normalize_condition_feature_analysis(
                    row.get("condition_feature_analysis", {})
                )
                for feature_key, values in feature_analysis.items():
                    for feature in {
                        value
                        for value in (
                            _cohort_feature_text(item) for item in list(values or [])
                        )
                        if value
                    }:
                        stats["features"][feature_key][feature] += 1
                for metadata in list(row.get("view_render_metadata", []) or []):
                    view = str(metadata.get("view") or "").strip()
                    if view:
                        stats["view_counts"][view] += 1
                for tool in tools_by_condition.get(condition_id, []):
                    tool_name = str(tool.get("tool") or "unknown").strip()
                    status = str(tool.get("tool_status") or "unknown").strip()
                    stats["tool_status_counts"][f"{tool_name}:{status}"] += 1

    finalized_conditions: Dict[str, Any] = {}
    for condition_id, condition in sorted(conditions.items()):
        finalized_groups: Dict[str, Any] = {}
        for group_name, stats in sorted(condition["groups"].items()):
            finalized_groups[group_name] = {
                "case_count": int(stats["case_count"]),
                "answer_counts": dict(sorted(stats["answer_counts"].items())),
                "confidence_counts": dict(sorted(stats["confidence_counts"].items())),
                "features": {
                    key: _feature_observation_items(
                        counter,
                        condition_id=condition_id,
                        group_name=group_name,
                        feature_type=key,
                    )
                    for key, counter in stats["features"].items()
                },
                "views": _counter_items(stats["view_counts"], limit=10, key_name="view"),
                "tool_statuses": _counter_items(
                    stats["tool_status_counts"],
                    limit=8,
                    key_name="tool_status",
                ),
            }
        finalized_conditions[condition_id] = {
            "is_exclusion": bool(condition.get("is_exclusion")),
            "groups": finalized_groups,
        }

    feature_budget = _apply_cohort_feature_budget(
        finalized_conditions,
        max_chars=COHORT_FEATURE_BUDGET_CHARS,
        max_per_bucket=COHORT_FEATURE_MAX_PER_BUCKET,
    )

    return {
        "schema_version": COHORT_SIGNAL_SCHEMA_VERSION,
        "counts": {name: len(cases) for name, cases in groups.items()},
        "contrastive_label_coverage": contrastive_label_coverage,
        "conditions": finalized_conditions,
        "protected_positive_signals": _protected_positive_signals(
            finalized_conditions,
            len(groups["correct_positive"]),
        ),
        "protected_boundary_signals": _protected_boundary_signals(
            finalized_conditions,
        ),
        "feature_observation_budget": feature_budget,
        "semantic_clustering": "deferred_to_reviewer_and_updater",
        "contains_transaction_identifiers": False,
    }


def _contrastive_supervision(slim: Dict[str, Any]) -> Dict[str, Any]:
    raw_case_label = str(
        slim.get("raw_ground_truth") or slim.get("ground_truth") or ""
    ).strip()
    raw_target_label = str(
        slim.get("raw_target_label")
        or slim.get("target_label")
        or slim.get("raw_attack_label")
        or slim.get("attack_label")
        or ""
    ).strip()
    return {
        "training_only": True,
        "target_label": normalize_attack_label(raw_target_label, default=""),
        "case_label": raw_case_label,
        "normalized_case_label": normalize_attack_label(
            raw_case_label,
            default="",
        ),
        "sample_role": str(slim.get("sample_role") or ""),
        "negative_kind": str(slim.get("negative_kind") or ""),
    }


def _contrastive_label_coverage(
    training_slims: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    target_labels: Counter[str] = Counter()
    negative_labels: Counter[str] = Counter()
    negative_kinds: Counter[str] = Counter()
    unknown_label_count = 0

    for slim in training_slims:
        ground_truth = str(slim.get("ground_truth") or "").lower().strip()
        supervision = _contrastive_supervision(slim)
        if ground_truth == "attack":
            label = str(supervision.get("target_label") or "").strip()
            if label:
                target_labels[label] += 1
            continue

        negative_kind = str(slim.get("negative_kind") or "unknown_other").strip()
        negative_kinds[negative_kind or "unknown_other"] += 1
        case_label = str(supervision.get("normalized_case_label") or "").strip()
        if negative_kind == "other_attack":
            if (
                case_label
                and case_label not in {"non_target", "other", "unknown", "negative"}
            ):
                negative_labels[case_label] += 1
            else:
                unknown_label_count += 1
        elif negative_kind == "unknown_other":
            unknown_label_count += 1

    return {
        "source": "fewshot_training_only",
        "guard_included": False,
        "target_label_counts": dict(sorted(target_labels.items())),
        "negative_label_counts": dict(sorted(negative_labels.items())),
        "negative_kind_counts": dict(sorted(negative_kinds.items())),
        "unknown_label_count": unknown_label_count,
    }


def build_plan_evidence_audit(
    slim_results: Iterable[Dict[str, Any]],
    *,
    guard_slim_results: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Summarize plan execution history without case-specific identifiers."""
    condition_stats: Dict[str, Dict[str, Any]] = {}
    case_groups = Counter()
    cases: List[tuple[Dict[str, Any], bool]] = [
        (build_slim_result(item), False) for item in list(slim_results or [])
    ]
    cases.extend(
        (build_slim_result(item), True)
        for item in list(guard_slim_results or [])
    )

    for slim, is_guard in cases:
        group_name = _plan_audit_case_group(slim, is_guard=is_guard)
        case_groups[group_name] += 1
        tools_by_condition: Dict[str, List[Dict[str, Any]]] = {}
        for tool in list(slim.get("tool_call_summary", []) or []):
            condition_id = str(
                tool.get("condition_id") or tool.get("judge_id") or ""
            ).strip()
            if condition_id:
                tools_by_condition.setdefault(condition_id, []).append(tool)

        for row in list(slim.get("condition_table", []) or []):
            condition_id = str(
                row.get("condition_id") or row.get("id") or ""
            ).strip()
            if not condition_id:
                continue
            condition = condition_stats.setdefault(condition_id, {
                "is_exclusion": bool(row.get("is_exclusion")),
                "groups": {},
            })
            stats = condition["groups"].setdefault(group_name, {
                "case_count": 0,
                "initial_answer_counts": Counter(),
                "final_answer_counts": Counter(),
                "answer_transitions": Counter(),
                "followup_changed_answer_count": 0,
                "near_miss_escalated_count": 0,
                "view_counts": Counter(),
                "prompt_truncated_view_counts": Counter(),
                "packet_truncated_view_counts": Counter(),
                "tool_status_counts": Counter(),
                "followup_view_attempt_counts": Counter(),
                "successful_followup_view_counts": Counter(),
                "answer_changing_followup_view_counts": Counter(),
                "missing_status_counts": Counter(),
                "missing_category_counts": Counter(),
                "route_failure_counts": Counter(),
                "tool_attempt_count": 0,
                "execution_paths": [],
            })
            stats["case_count"] += 1
            initial_answer = _cohort_answer_key(
                row.get("first_round_answer", row.get("answer"))
            )
            final_answer = _cohort_answer_key(
                row.get("final_answer", row.get("answer"))
            )
            stats["initial_answer_counts"][initial_answer] += 1
            stats["final_answer_counts"][final_answer] += 1
            stats["answer_transitions"][f"{initial_answer}->{final_answer}"] += 1
            answer_changed = bool(row.get("followup_changed_answer"))
            stats["followup_changed_answer_count"] += int(answer_changed)
            stats["near_miss_escalated_count"] += int(
                bool(row.get("near_miss_escalated"))
            )

            path_views: List[Dict[str, Any]] = []
            for metadata in list(row.get("view_render_metadata", []) or []):
                if not isinstance(metadata, dict):
                    continue
                view = str(metadata.get("view") or "").strip()
                if not view:
                    continue
                stats["view_counts"][view] += 1
                if bool(
                    metadata.get("prompt_render_truncated")
                    or metadata.get("render_truncated")
                    or metadata.get("truncated")
                ):
                    stats["prompt_truncated_view_counts"][view] += 1
                    stats["route_failure_counts"][(view, "truncated")] += 1
                if bool(
                    metadata.get("packet_trace_truncated")
                    or metadata.get("packet_truncated")
                ):
                    stats["packet_truncated_view_counts"][view] += 1
                    stats["route_failure_counts"][(view, "truncated")] += 1
                if len(path_views) < 12:
                    path_views.append({
                        "round": metadata.get("round"),
                        "view": view,
                        "prompt_truncated": bool(
                            metadata.get("prompt_render_truncated")
                            or metadata.get("render_truncated")
                            or metadata.get("truncated")
                        ),
                        "packet_truncated": bool(
                            metadata.get("packet_trace_truncated")
                            or metadata.get("packet_truncated")
                        ),
                    })

            path_tools: List[Dict[str, Any]] = []
            for tool in tools_by_condition.get(condition_id, []):
                tool_name = str(tool.get("tool") or "unknown").strip()
                status = str(tool.get("tool_status") or "unknown").strip()
                stats["tool_attempt_count"] += 1
                stats["tool_status_counts"][f"{tool_name}:{status}"] += 1
                args = tool.get("args") if isinstance(tool.get("args"), dict) else {}
                requested_view = str(args.get("view") or "").strip()
                returned_count = len(list(tool.get("returned_evidence_ids", []) or []))
                if len(path_tools) < 10:
                    path_tools.append({
                        "round": tool.get("round"),
                        "tool": tool_name,
                        "status": status,
                        "requested_view": requested_view,
                        "returned_evidence_count": returned_count,
                        "validation_failed": bool(tool.get("validation_error")),
                    })
                if not requested_view:
                    continue
                stats["followup_view_attempt_counts"][requested_view] += 1
                if status == "ok" and returned_count > 0:
                    stats["successful_followup_view_counts"][requested_view] += 1
                    if answer_changed:
                        stats["answer_changing_followup_view_counts"][requested_view] += 1
                else:
                    stats["route_failure_counts"][(requested_view, "unavailable")] += 1

            path_missing: List[Dict[str, Any]] = []
            for missing in list(row.get("all_missing_evidence_debug", []) or []):
                if not isinstance(missing, dict):
                    continue
                status = str(missing.get("status") or "unknown").strip().lower()
                category = str(missing.get("category") or "unknown").strip().lower()
                stats["missing_status_counts"][status] += 1
                stats["missing_category_counts"][category] += 1
                if len(path_missing) < 10:
                    path_missing.append({
                        "round": missing.get("round"),
                        "category": category,
                        "status": status,
                        "blocking": bool(missing.get("blocking")),
                        "resolved": status == "resolved",
                    })

            if (
                group_name in {
                    "positive_error",
                    "negative_error",
                    "negative_uncertain",
                    "guard_error",
                }
                and len(stats["execution_paths"]) < 8
                and (path_tools or path_missing or answer_changed or any(
                    item.get("prompt_truncated") or item.get("packet_truncated")
                    for item in path_views
                ))
            ):
                stats["execution_paths"].append({
                    "sequence": (
                        "initial_judge -> followup_tools -> final_judge"
                        if path_tools
                        else "initial_judge -> final_judge"
                    ),
                    "initial_answer": initial_answer,
                    "final_answer": final_answer,
                    "answer_changed": answer_changed,
                    "views": path_views,
                    "tools": path_tools,
                    "missing_evidence": path_missing,
                })

    finalized: Dict[str, Any] = {}
    for condition_id, condition in sorted(condition_stats.items()):
        groups: Dict[str, Any] = {}
        successful_views = Counter()
        route_failures = Counter()
        route_failure_groups: Dict[tuple[str, str], set[str]] = {}
        for group_name, stats in sorted(condition["groups"].items()):
            successful_views.update(stats["successful_followup_view_counts"])
            observed_routes = set(stats["view_counts"]) | set(
                stats["followup_view_attempt_counts"]
            )
            for route in observed_routes:
                observations = max(
                    int(stats["view_counts"].get(route, 0)),
                    int(stats["followup_view_attempt_counts"].get(route, 0)),
                )
                answer_changes = int(
                    stats["answer_changing_followup_view_counts"].get(route, 0)
                )
                successful = int(
                    stats["successful_followup_view_counts"].get(route, 0)
                )
                if observations >= 2 and answer_changes == 0:
                    stats["route_failure_counts"][(route, "no_answer_change")] += observations
                if successful >= 2 and answer_changes == 0:
                    stats["route_failure_counts"][(route, "repeatedly_insufficient")] += successful
            route_failures.update(stats["route_failure_counts"])
            for key in stats["route_failure_counts"]:
                route_failure_groups.setdefault(key, set()).add(group_name)
            groups[group_name] = {
                "case_count": int(stats["case_count"]),
                "initial_answer_counts": dict(sorted(stats["initial_answer_counts"].items())),
                "final_answer_counts": dict(sorted(stats["final_answer_counts"].items())),
                "answer_transitions": dict(sorted(stats["answer_transitions"].items())),
                "followup_changed_answer_count": int(stats["followup_changed_answer_count"]),
                "near_miss_escalated_count": int(stats["near_miss_escalated_count"]),
                "views": _counter_items(stats["view_counts"], limit=16, key_name="view"),
                "prompt_truncated_views": _counter_items(
                    stats["prompt_truncated_view_counts"], limit=12, key_name="view"
                ),
                "packet_truncated_views": _counter_items(
                    stats["packet_truncated_view_counts"], limit=12, key_name="view"
                ),
                "tool_statuses": _counter_items(
                    stats["tool_status_counts"], limit=16, key_name="tool_status"
                ),
                "followup_view_attempts": _counter_items(
                    stats["followup_view_attempt_counts"], limit=12, key_name="view"
                ),
                "successful_followup_views": _counter_items(
                    stats["successful_followup_view_counts"], limit=12, key_name="view"
                ),
                "answer_changing_followup_views": _counter_items(
                    stats["answer_changing_followup_view_counts"], limit=12, key_name="view"
                ),
                "missing_statuses": dict(sorted(stats["missing_status_counts"].items())),
                "missing_categories": dict(sorted(stats["missing_category_counts"].items())),
                "tool_attempt_count": int(stats["tool_attempt_count"]),
                "execution_paths": list(stats["execution_paths"]),
                "evidence_history_present": bool(
                    stats["view_counts"]
                    or stats["tool_attempt_count"]
                    or stats["missing_status_counts"]
                ),
            }
        finalized[condition_id] = {
            "is_exclusion": bool(condition.get("is_exclusion")),
            "groups": groups,
            "route_failure_memory": [
                {
                    "route": route,
                    "failure_type": failure_type,
                    "support_count": int(count),
                    "historical_outcome": "failed_or_noncontributing",
                    "case_groups": sorted(
                        route_failure_groups.get((route, failure_type), set())
                    ),
                }
                for (route, failure_type), count in sorted(route_failures.items())
            ],
            "observed_successful_followup_views": [
                item["view"]
                for item in _counter_items(successful_views, limit=16, key_name="view")
            ],
        }

    return {
        "schema_version": PLAN_EVIDENCE_AUDIT_SCHEMA_VERSION,
        "case_groups": dict(sorted(case_groups.items())),
        "conditions": finalized,
        "contains_transaction_identifiers": False,
        "interpretation": (
            "Execution telemetry only. route_failure_memory is derived from "
            "observed unavailable/truncated/noncontributing outcomes and may "
            "authorize bounded route replacement; it does not prove semantic "
            "rule changes."
        ),
    }


def _plan_audit_case_group(slim: Dict[str, Any], *, is_guard: bool) -> str:
    ground_truth = str(slim.get("ground_truth") or "").lower().strip()
    verdict = str(slim.get("predicted_verdict") or "").lower().strip()
    if is_guard:
        return "correct_guard" if ground_truth == "benign" and verdict == "benign" else "guard_error"
    if ground_truth == "attack" and verdict == "attack":
        return "correct_positive"
    if ground_truth == "benign" and verdict == "benign":
        return "correct_negative"
    if ground_truth == "attack":
        return "positive_error"
    if ground_truth == "benign":
        return "negative_uncertain" if verdict == "uncertain" else "negative_error"
    return "unknown"


def _review_has_deferred_logic_structure_failure(
    review: Dict[str, Any],
) -> bool:
    return any(
        isinstance(cause, dict)
        and str(cause.get("category") or "").strip().lower()
        == "deferred_logic_structure_failure"
        for cause in list((review or {}).get("root_causes") or [])
    )


def build_update_signal_matrix(
    reviews: Iterable[Dict[str, Any]],
    cohort_signal_summary: Dict[str, Any],
    *,
    plan_evidence_audit: Optional[Dict[str, Any]] = None,
    enable_advanced_candidate_lifecycles: bool = True,
) -> Dict[str, Any]:
    signals: List[Dict[str, Any]] = []
    cohort_counts = dict((cohort_signal_summary or {}).get("counts") or {})
    correct_reference_count = sum(int(value or 0) for value in cohort_counts.values())
    signal_index = 0
    review_list = list(reviews or [])
    for review_index, review in enumerate(review_list):
        deferred_logic_failure = _review_has_deferred_logic_structure_failure(
            review
        )
        for source_signal_index, raw_signal in enumerate(
            list(review.get("update_signals", []) or [])
        ):
            if not isinstance(raw_signal, dict):
                continue
            signal_index += 1
            signal_id = f"signal_{signal_index:03d}"
            status = str(
                raw_signal.get("generalization_status") or "insufficient"
            ).lower().strip()
            if status not in {"supported", "conflicted", "insufficient"}:
                status = "insufficient"
            abstract_feature = _cohort_feature_text(
                raw_signal.get("abstract_feature")
            )
            if not abstract_feature:
                status = "insufficient"
            canonical_update_target = str(
                raw_signal.get("update_target")
                or review.get("update_target")
                or "none"
            )
            source_update_target = str(
                raw_signal.get("source_update_target")
                or canonical_update_target
            )
            training_gate_reason = _review_training_gate_reason(
                review,
                canonical_update_target.lower().strip(),
            )
            signal_gate_reason = str(
                raw_signal.get("signal_gate_reason") or ""
            ).strip()
            if signal_gate_reason:
                training_gate_reason = signal_gate_reason
            if deferred_logic_failure:
                training_gate_reason = "deferred_logic_structure_failure"
            if training_gate_reason:
                status = "insufficient"
            packet_capability_escalation = _plan_signal_packet_capability_escalation(
                raw_signal,
                review,
                update_target=canonical_update_target,
            )
            if packet_capability_escalation:
                status = "insufficient"
                training_gate_reason = "packet_or_runtime_capability_absent"
            update_target = str(
                packet_capability_escalation.get("owner")
                or canonical_update_target
            ).strip().lower()
            signal = {
                "signal_id": signal_id,
                "source_review_index": review_index,
                "source_signal_index": source_signal_index,
                "source_error_type": str(review.get("error_type") or "unknown"),
                "update_target": update_target,
                "source_update_target": source_update_target,
                "condition_id": str(raw_signal.get("condition_id") or ""),
                "direction": str(raw_signal.get("direction") or "no_change"),
                "strategy_operation": str(
                    raw_signal.get("strategy_operation") or "none"
                ),
                "condition_evidence_dependency": (
                    _normalize_matrix_evidence_dependency(
                        raw_signal.get("condition_evidence_dependency"),
                        update_target=update_target,
                    )
                ),
                "abstract_feature": abstract_feature,
                "feature_axis": _cohort_feature_text(
                    raw_signal.get("feature_axis")
                )[:160],
                "generalization_status": status,
                "training_gate_reason": training_gate_reason,
                "owner_escalation": packet_capability_escalation,
                "deferred_logic_structure_failure": deferred_logic_failure,
                "error_support_count": 1,
                "correct_reference_count": correct_reference_count,
                "correct_positive_compatibility": str(
                    raw_signal.get("correct_positive_compatibility") or ""
                )[:500],
                "correct_negative_compatibility": str(
                    raw_signal.get("correct_negative_compatibility") or ""
                )[:500],
                "cohort_support_refs": list(
                    raw_signal.get("cohort_support_refs") or []
                )[:12],
                "cohort_conflict_refs": list(
                    raw_signal.get("cohort_conflict_refs") or []
                )[:12],
                "wrong_scope_audit": (
                    dict(raw_signal.get("wrong_scope_audit") or {})
                    if isinstance(raw_signal.get("wrong_scope_audit"), dict)
                    else {}
                ),
                "rationale": str(raw_signal.get("rationale") or "")[:700],
            }
            signal = apply_positive_mandatory_conflict_gate(
                signal,
                cohort_signal_summary,
            )
            signals.append(signal)
    _apply_cross_review_signal_conflicts(signals)
    if enable_advanced_candidate_lifecycles:
        for signal in signals:
            if _qualifies_as_experimental_plan_signal(
                signal,
                reviews=review_list,
                plan_evidence_audit=plan_evidence_audit or {},
            ):
                signal["generalization_status"] = "experimental_plan"
                signal["experimental_plan_probe"] = True
                signal["experimental_plan_reason"] = (
                    "concrete_untried_evidence_route_without_rule_semantic_change"
                )
            elif _qualifies_as_experimental_rule_signal(
                signal,
                reviews=review_list,
            ):
                signal["generalization_status"] = "experimental_rule"
                signal["experimental_rule_probe"] = True
                signal["experimental_rule_reason"] = (
                    "abstract_boundary_hypothesis_requires_runtime_probe"
                )
    joint_resolution_groups = (
        _cross_review_joint_resolution_groups(signals)
        if enable_advanced_candidate_lifecycles
        else []
    )
    supported_ids = [
        signal["signal_id"]
        for signal in signals
        if signal.get("generalization_status") == "supported"
    ]
    conflicted_ids = [
        signal["signal_id"]
        for signal in signals
        if signal.get("generalization_status") == "conflicted"
    ]
    insufficient_ids = [
        signal["signal_id"]
        for signal in signals
        if signal.get("generalization_status") == "insufficient"
    ]
    experimental_plan_ids = [
        signal["signal_id"]
        for signal in signals
        if signal.get("generalization_status") == "experimental_plan"
    ]
    experimental_rule_ids = [
        signal["signal_id"]
        for signal in signals
        if signal.get("generalization_status") == "experimental_rule"
    ]
    return {
        "schema_version": UPDATE_SIGNAL_MATRIX_SCHEMA_VERSION,
        "signals": signals,
        "supported_signal_ids": supported_ids,
        "conflicted_signal_ids": conflicted_ids,
        "insufficient_signal_ids": insufficient_ids,
        "experimental_plan_signal_ids": experimental_plan_ids,
        "experimental_rule_signal_ids": experimental_rule_ids,
        "plan_actionable_signal_ids": [
            signal["signal_id"]
            for signal in signals
            if str(signal.get("update_target") or "").strip().lower() == "plan"
            and signal.get("generalization_status")
            in {"supported", "experimental_plan"}
        ],
        "joint_resolution_signal_ids": sorted({
            signal_id
            for group in joint_resolution_groups
            for signal_id in list(group.get("signal_ids") or [])
        }),
        "joint_resolution_groups": joint_resolution_groups,
        "correct_reference_count": correct_reference_count,
        "cross_review_conflict_count": sum(
            len(list(signal.get("cross_review_conflicts", []) or []))
            for signal in signals
        ),
        "cross_review_conflict_policy": (
            "opposite FP/FN directions conflict only when they address the same "
            "semantic feature axis; distinct axes remain supported and must be "
            "resolved jointly by the updater"
        ),
    }


def _plan_signal_packet_capability_escalation(
    raw_signal: Dict[str, Any],
    review: Dict[str, Any],
    *,
    update_target: str,
) -> Dict[str, Any]:
    """Escalate a Plan intent only when runtime facts prove engineering ownership."""
    if str(update_target or "").strip().lower() != "plan":
        return {}
    explicit_status = str(
        raw_signal.get("packet_capability_status")
        or raw_signal.get("capability_surface_status")
        or raw_signal.get("missing_evidence_origin")
        or ""
    ).strip().lower()
    condition_id = str(raw_signal.get("condition_id") or "").strip().upper()
    corroborated_review_owner = ""
    for route in list(review.get("diagnosis_routes") or []):
        if not isinstance(route, dict) or not is_authoritative_engineering_route(route):
            continue
        affected = {
            str(value or "").strip().upper()
            for value in list(route.get("affected_conditions") or [])
            if str(value or "").strip()
        }
        if condition_id and affected and condition_id not in affected:
            continue
        corroborated_review_owner = str(route.get("owner") or "").strip().lower()
        break
    absent_status = explicit_status in {
        "packet_absent",
        "packet_capability_absent",
        "runtime_capability_absent",
        "absent_from_packet",
        "absent_from_runtime",
    }
    # Reviewer-declared status/owner describes intent only. Engineering
    # ownership requires a deterministic diagnosis route backed by runtime
    # Packet/tool facts.
    corroborated_owner = corroborated_review_owner
    if absent_status and corroborated_owner in {"packet", "runtime"}:
        return {
            "owner": corroborated_owner,
            "reason": "required_evidence_capability_absent_from_plan_surface",
            "source": "update_signal_and_review_capability_diagnosis",
        }

    return {}


def _qualifies_as_experimental_rule_signal(
    signal: Dict[str, Any],
    *,
    reviews: List[Dict[str, Any]],
) -> bool:
    """Admit one abstract Rule hypothesis to probe without making it supported."""
    if str(signal.get("generalization_status") or "").strip().lower() != "insufficient":
        return False
    if str(signal.get("update_target") or "").strip().lower() != "rule":
        return False
    if str(signal.get("direction") or "").strip().lower() not in {
        "broaden",
        "tighten",
        "refine",
        "add_exclusion",
        "narrow_exclusion",
        "clarify_boundary",
    }:
        return False
    if str(signal.get("training_gate_reason") or "").strip():
        return False
    if signal.get("deferred_logic_structure_failure"):
        return False
    if list(signal.get("cohort_conflict_refs") or []) or list(
        signal.get("cross_review_conflicts") or []
    ):
        return False
    if int(signal.get("correct_reference_count") or 0) <= 0:
        return False
    abstract_feature = str(signal.get("abstract_feature") or "").strip()
    feature_axis = str(signal.get("feature_axis") or "").strip()
    if not abstract_feature or not feature_axis:
        return False
    if _CONCRETE_FEATURE_RE.search(
        " ".join((abstract_feature, str(signal.get("rationale") or "")))
    ):
        return False
    raw_review_index = signal.get("source_review_index", -1)
    review_index = int(-1 if raw_review_index in (None, "") else raw_review_index)
    if review_index < 0 or review_index >= len(reviews):
        return False
    review = dict(reviews[review_index] or {})
    if not _review_allows_training_update(review, "rule"):
        return False
    suggestion = dict(review.get("rule_patch_suggestion") or {})
    return str(suggestion.get("action") or "").strip().lower() not in {
        "",
        "none",
        "remove_condition",
        "remove_exclusion",
    }


def _qualifies_as_experimental_plan_signal(
    signal: Dict[str, Any],
    *,
    reviews: List[Dict[str, Any]],
    plan_evidence_audit: Dict[str, Any],
) -> bool:
    """Allow a bounded evidence-route experiment without relaxing Rule gates."""
    status = str(signal.get("generalization_status") or "").strip().lower()
    if status not in {"insufficient", "conflicted"}:
        return False
    if str(signal.get("update_target") or "").strip().lower() != "plan":
        return False
    if str(signal.get("direction") or "").strip().lower() != "change_evidence_strategy":
        return False
    training_gate_reason = str(
        signal.get("training_gate_reason") or ""
    ).strip().lower()
    if training_gate_reason not in {
        "",
        "ground_truth_not_supported_by_packet",
        "fn_target_mechanism_not_confirmed_for_plan_update",
    }:
        return False
    if list(signal.get("cross_review_conflicts") or []):
        return False
    if signal.get("deferred_logic_structure_failure"):
        return False
    strategy_operation = str(
        signal.get("strategy_operation") or ""
    ).strip().lower()
    if status == "conflicted" and strategy_operation not in {
        "add_evidence_route",
        "change_verification_granularity",
        "change_followup_strategy",
        "replace_evidence_route",
    }:
        return False

    raw_review_index = signal.get("source_review_index", -1)
    review_index = int(
        -1 if raw_review_index in (None, "") else raw_review_index
    )
    if review_index < 0 or review_index >= len(reviews):
        return False
    review = dict(reviews[review_index] or {})
    review_gate_reason = _review_training_gate_reason(review, "plan")
    if review_gate_reason not in {
        "",
        "ground_truth_not_supported_by_packet",
        "fn_target_mechanism_not_confirmed_for_plan_update",
    }:
        return False
    has_plan_intent = bool(review.get("should_update_plan_strategy")) or any(
        isinstance(item, dict) and bool(item.get("is_plan_problem"))
        for item in list(review.get("condition_diagnosis") or [])
    )
    suggestion = dict(review.get("plan_patch_suggestion") or {})
    has_plan_intent = has_plan_intent or str(
        suggestion.get("action") or ""
    ).strip().lower() not in {"", "none"}
    if not has_plan_intent:
        return False

    proposal_text = " ".join(
        str(value or "")
        for value in (
            signal.get("abstract_feature"),
            signal.get("rationale"),
            suggestion.get("proposed_change"),
            suggestion.get("rationale"),
        )
    ).lower()
    if not re.search(
        r"\b(?:missing|unresolved|truncat|omitted|source|context|followup|"
        r"view|evidence|state|event|call|selector|wrapper)\b",
        proposal_text,
    ):
        return False
    proposed_routes = set(
        re.findall(r"\b[a-z][a-z0-9_]*(?:_view|_context)\b", proposal_text)
    )
    for tool in (
        "get_local_call_context",
        "read_function_chunk",
        "read_evidence_context",
        "read_evidence_by_id",
    ):
        if tool in proposal_text:
            proposed_routes.add(tool)
    proposed_routes.update(_bounded_probe_routes_from_semantics(proposal_text))
    condition_id = str(signal.get("condition_id") or "").strip().upper()
    condition_audit = dict(
        ((plan_evidence_audit or {}).get("conditions") or {}).get(
            condition_id,
            {},
        )
        or {}
    )
    failed_routes: set[str] = set()
    for item in list(condition_audit.get("route_failure_memory") or []):
        if not isinstance(item, dict):
            continue
        failure_type = str(item.get("failure_type") or "").strip().lower()
        historical_outcome = str(
            item.get("historical_outcome") or ""
        ).strip().lower()
        route = str(item.get("route") or "").strip()
        if route and (
            failure_type in {
                "unavailable",
                "truncated",
                "no_answer_change",
                "repeatedly_insufficient",
            }
            or historical_outcome == "failed_or_noncontributing"
        ):
            failed_routes.add(route)
    if strategy_operation == "replace_evidence_route":
        mentioned_failed_routes = {
            route
            for route in failed_routes
            if route in proposed_routes
            or route.lower().replace("_", " ") in proposal_text
        }
        if not mentioned_failed_routes:
            return False
    # Reviewer prose may spell a registered route as words (for example,
    # "semantic state delta view"). Recover only route names already observed
    # in the deterministic audit; do not invent arbitrary view names.
    known_routes: set[str] = set()
    for group in dict(condition_audit.get("groups") or {}).values():
        if not isinstance(group, dict):
            continue
        for key in (
            "views",
            "followup_view_attempts",
            "successful_followup_views",
            "answer_changing_followup_views",
        ):
            for item in list(group.get(key) or []):
                route = (
                    str(item.get("view") or "")
                    if isinstance(item, dict)
                    else str(item or "")
                ).strip()
                if route:
                    known_routes.add(route)
    for route in known_routes:
        if route.lower().replace("_", " ") in proposal_text:
            proposed_routes.add(route)
    if not proposed_routes:
        return False

    error_group_name = (
        "positive_error"
        if str(signal.get("source_error_type") or "").strip().upper() == "FN"
        else "negative_error"
    )
    error_group = dict(
        (condition_audit.get("groups") or {}).get(error_group_name) or {}
    )
    successful_views = set(
        str(value)
        for value in [
            (
                item.get("view")
                if isinstance(item, dict)
                else item
            )
            for item in list(error_group.get("successful_followup_views") or [])
        ]
        if str(value or "")
    )
    attempted_tools: set[str] = set()
    for item in list(error_group.get("tool_statuses") or []):
        if not isinstance(item, dict):
            continue
        tool_status = str(item.get("tool_status") or "")
        if ":" in tool_status:
            attempted_tools.add(tool_status.split(":", 1)[0])

    probe_routes = (
        proposed_routes - failed_routes
        if strategy_operation == "replace_evidence_route"
        else proposed_routes
    )
    untried_route_exists = any(
        route not in successful_views and route not in attempted_tools
        for route in probe_routes
    )
    if untried_route_exists:
        signal["experimental_probe_routes"] = sorted(probe_routes)[:8]
        if strategy_operation == "replace_evidence_route":
            signal["experimental_failed_routes"] = sorted(failed_routes)[:8]
        signal["experimental_probe_conflict_refs"] = list(
            signal.get("cohort_conflict_refs") or []
        )[:12]
    return bool(untried_route_exists)


def _bounded_probe_routes_from_semantics(proposal_text: str) -> set[str]:
    """Map a narrow evidence need to existing routes without inventing tools."""
    text = str(proposal_text or "").lower()
    routes: set[str] = set()
    if re.search(
        r"\b(?:argument[- ]level|argument lineage|rate lineage|same[- ]source|"
        r"call ordering|temporal order|trace[- ]internal|consumer call)\b",
        text,
    ):
        routes.update({
            "critical_call_argument_view",
            "get_local_call_context",
            "read_evidence_context",
        })
    if re.search(r"\b(?:semantic state delta|state lineage|state transition)\b", text):
        routes.add("semantic_state_delta_view")
    allowed_tools = {
        "get_local_call_context",
        "read_evidence_context",
        "read_evidence_by_id",
    }
    return {
        route
        for route in routes
        if route in PACKET_VIEW_CATALOG or route in allowed_tools
    }


def _attach_matrix_signals_to_reviews(
    reviews: List[Dict[str, Any]],
    matrix: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Expose deterministic matrix status/ids to the appropriate Updater."""
    by_review: Dict[int, List[Dict[str, Any]]] = {}
    for signal in list((matrix or {}).get("signals") or []):
        if not isinstance(signal, dict):
            continue
        raw_review_index = signal.get("source_review_index", -1)
        review_index = int(
            -1 if raw_review_index in (None, "") else raw_review_index
        )
        if review_index >= 0:
            by_review.setdefault(review_index, []).append(dict(signal))

    attached: List[Dict[str, Any]] = []
    for review_index, raw_review in enumerate(list(reviews or [])):
        review = dict(raw_review or {})
        matrix_signals = by_review.get(review_index)
        if matrix_signals is not None:
            review["update_signals"] = matrix_signals
        has_experimental_plan = any(
            str(signal.get("generalization_status") or "").strip().lower()
            == "experimental_plan"
            for signal in list(review.get("update_signals") or [])
            if isinstance(signal, dict)
        )
        if has_experimental_plan:
            review["should_update_plan_strategy"] = True
            gate = dict(review.get("generalization_gate") or {})
            gate["experimental_plan_probe"] = True
            review["generalization_gate"] = gate
        has_experimental_rule = any(
            str(signal.get("generalization_status") or "").strip().lower()
            == "experimental_rule"
            for signal in list(review.get("update_signals") or [])
            if isinstance(signal, dict)
        )
        if has_experimental_rule:
            review["should_update_rule"] = True
            gate = dict(review.get("generalization_gate") or {})
            gate["experimental_rule_probe"] = True
            review["generalization_gate"] = gate
        attached.append(review)
    return attached


def apply_positive_mandatory_conflict_gate(
    signal: Dict[str, Any],
    cohort_signal_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit inferred positive-boundary conflicts without changing lifecycle."""
    gated = dict(signal or {})
    if str(gated.get("generalization_status") or "").lower().strip() != "supported":
        return gated
    direction = str(gated.get("direction") or "").lower().strip()
    if direction not in _RESTRICTIVE_SIGNAL_DIRECTIONS:
        return gated
    abstract_feature = str(gated.get("abstract_feature") or "")
    if _ALTERNATIVE_POSITIVE_PATH_RE.search(abstract_feature):
        return gated
    if not _MANDATORY_SIGNAL_RE.search(abstract_feature):
        return _apply_positive_inference_conflict_gate(
            gated,
            cohort_signal_summary,
        )

    condition_id = str(gated.get("condition_id") or "").upper().strip()
    condition = dict(
        ((cohort_signal_summary or {}).get("conditions") or {}).get(
            condition_id,
            {},
        )
        or {}
    )
    positive_group = dict(
        ((condition.get("groups") or {}).get("correct_positive") or {})
    )
    feature_groups = dict(positive_group.get("features") or {})
    candidates: List[Dict[str, Any]] = []
    for item in list(feature_groups.get("missing_required_features", []) or []):
        if isinstance(item, dict):
            candidates.append(dict(item))
    for item in list(feature_groups.get("boundary_notes", []) or []):
        if not isinstance(item, dict):
            continue
        if _POSITIVE_UNAVAILABLE_RE.search(str(item.get("feature") or "")):
            candidates.append(dict(item))
    if not candidates:
        return _apply_positive_inference_conflict_gate(
            gated,
            cohort_signal_summary,
        )

    explicit_conflicts = {
        str(value)
        for value in list(gated.get("cohort_conflict_refs", []) or [])
        if str(value)
    }
    proposal_concepts = _mandatory_feature_concepts(abstract_feature)
    conflicts: List[Dict[str, Any]] = []
    for candidate in candidates:
        feature_ref = str(candidate.get("feature_ref") or "")
        feature = str(candidate.get("feature") or "")
        concepts = _mandatory_feature_concepts(feature)
        shared_concepts = sorted(proposal_concepts & concepts)
        explicitly_referenced = feature_ref in explicit_conflicts
        if not explicitly_referenced and len(shared_concepts) < 2:
            continue
        conflicts.append({
            "feature_ref": feature_ref,
            "feature": feature[:500],
            "shared_concepts": shared_concepts[:12],
            "explicit_conflict_ref": explicitly_referenced,
        })
    if not conflicts:
        return _apply_positive_inference_conflict_gate(
            gated,
            cohort_signal_summary,
        )

    gated["inferred_conflict_warning"] = True
    gated["inferred_conflict_reason"] = (
        "mandatory_feature_missing_or_unavailable_in_correct_positive"
    )
    gated["mandatory_positive_conflicts"] = conflicts[:12]
    gated["cohort_conflict_refs"] = _unique(
        list(gated.get("cohort_conflict_refs", []) or [])
        + [item.get("feature_ref") for item in conflicts]
    )[:12]
    return gated


def _apply_positive_inference_conflict_gate(
    signal: Dict[str, Any],
    cohort_signal_summary: Dict[str, Any],
) -> Dict[str, Any]:
    gated = dict(signal or {})
    inference_conflicts = _restrictive_positive_inference_conflicts(
        gated,
        cohort_signal_summary,
    )
    if not inference_conflicts:
        return gated
    gated["inferred_conflict_warning"] = True
    gated["inferred_conflict_reason"] = (
        "restrictive_boundary_not_confirmed_in_correct_positive"
    )
    gated["positive_inference_conflicts"] = inference_conflicts[:12]
    gated["cohort_conflict_refs"] = _unique(
        list(gated.get("cohort_conflict_refs", []) or [])
        + [item.get("feature_ref") for item in inference_conflicts]
    )[:12]
    return gated


def _restrictive_positive_inference_conflicts(
    signal: Dict[str, Any],
    cohort_signal_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find positive boundaries that only infer the proposed requirement."""
    condition_id = str(signal.get("condition_id") or "").upper().strip()
    condition = dict(
        ((cohort_signal_summary or {}).get("conditions") or {}).get(
            condition_id,
            {},
        )
        or {}
    )
    positive_group = dict(
        ((condition.get("groups") or {}).get("correct_positive") or {})
    )
    feature_groups = dict(positive_group.get("features") or {})
    proposal_text = " ".join(
        str(signal.get(key) or "")
        for key in (
            "abstract_feature",
            "correct_positive_compatibility",
            "rationale",
        )
    )
    proposal_concepts = _mandatory_feature_concepts(proposal_text)
    explicit_conflicts = {
        str(value)
        for value in list(signal.get("cohort_conflict_refs", []) or [])
        if str(value)
    }
    conflicts: List[Dict[str, Any]] = []
    for observations in feature_groups.values():
        for raw in list(observations or []):
            if not isinstance(raw, dict):
                continue
            feature = str(raw.get("feature") or "")
            if not _POSITIVE_UNCONFIRMED_RE.search(feature):
                continue
            feature_ref = str(raw.get("feature_ref") or "")
            shared_concepts = sorted(
                proposal_concepts & _mandatory_feature_concepts(feature)
            )
            explicitly_referenced = feature_ref in explicit_conflicts
            compatibility_admits_uncertainty = bool(
                _POSITIVE_UNCONFIRMED_RE.search(
                    str(signal.get("correct_positive_compatibility") or "")
                )
            )
            if (
                not explicitly_referenced
                and not compatibility_admits_uncertainty
                and not shared_concepts
            ):
                continue
            conflicts.append({
                "feature_ref": feature_ref,
                "feature": feature[:500],
                "shared_concepts": shared_concepts[:12],
                "explicit_conflict_ref": explicitly_referenced,
                "compatibility_admits_uncertainty": compatibility_admits_uncertainty,
            })
    return conflicts


def _mandatory_feature_concepts(value: Any) -> set[str]:
    concepts: set[str] = set()
    for token in re.findall(r"[a-z][a-z-]{2,}", str(value or "").lower()):
        normalized = _MANDATORY_CONCEPT_ALIASES.get(token, token)
        if normalized in _MANDATORY_CONCEPT_STOPWORDS:
            continue
        concepts.add(normalized)
    return concepts


def _apply_cross_review_signal_conflicts(signals: List[Dict[str, Any]]) -> None:
    conflicts_by_id: Dict[str, List[Dict[str, Any]]] = {}
    joint_peers_by_id: Dict[str, set[str]] = {}
    for left_index, left in enumerate(signals):
        for right in signals[left_index + 1:]:
            if not _signals_share_cross_review_scope(left, right):
                continue
            left_direction = str(
                left.get("direction") or "no_change"
            ).lower().strip()
            right_direction = str(
                right.get("direction") or "no_change"
            ).lower().strip()
            same_semantic_axis = _signals_share_semantic_axis(left, right)

            mutation_pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
            if left_direction == "no_change" and right_direction != "no_change":
                mutation_pairs.append((right, left))
            elif right_direction == "no_change" and left_direction != "no_change":
                mutation_pairs.append((left, right))
            elif _directions_conflict(left_direction, right_direction):
                if not same_semantic_axis:
                    if (
                        left.get("generalization_status") == "supported"
                        and right.get("generalization_status") == "supported"
                    ):
                        left_id = str(left.get("signal_id") or "")
                        right_id = str(right.get("signal_id") or "")
                        joint_peers_by_id.setdefault(left_id, set()).add(right_id)
                        joint_peers_by_id.setdefault(right_id, set()).add(left_id)
                    continue
                mutation_pairs.extend(((left, right), (right, left)))

            if not same_semantic_axis:
                continue
            for mutation, blocker in mutation_pairs:
                if mutation.get("generalization_status") != "supported":
                    continue
                blocker_direction = str(
                    blocker.get("direction") or "no_change"
                ).lower().strip()
                if (
                    blocker_direction != "no_change"
                    and blocker.get("generalization_status") != "supported"
                ):
                    continue
                mutation_id = str(mutation.get("signal_id") or "")
                conflicts_by_id.setdefault(mutation_id, []).append({
                    "blocking_signal_id": blocker.get("signal_id", ""),
                    "blocking_review_index": blocker.get("source_review_index"),
                    "blocking_error_type": blocker.get(
                        "source_error_type", "unknown"
                    ),
                    "blocking_direction": blocker_direction,
                    "semantic_axis": _signal_semantic_axis(mutation),
                    "reason": (
                        "opposite-error review preserves the same semantic feature"
                        if blocker_direction == "no_change"
                        else "opposite-error reviews request incompatible directions "
                        "for the same semantic feature"
                    ),
                })

    for signal in signals:
        signal_id = str(signal.get("signal_id") or "")
        conflicts = conflicts_by_id.get(signal_id, [])
        if conflicts:
            signal["cross_review_conflicts"] = conflicts
            signal["inferred_conflict_warning"] = True
            signal["inferred_conflict_reason"] = (
                "cross_review_semantic_axis_conflict_requires_validation"
            )
            continue
        peers = sorted(joint_peers_by_id.get(signal_id, set()))
        if peers and signal.get("generalization_status") == "supported":
            signal["joint_resolution_required"] = True
            signal["joint_resolution_peer_ids"] = peers
            signal["joint_resolution_reason"] = (
                "Opposite error directions address distinct semantic features in "
                "the same condition and must be synthesized in one candidate."
            )


def _signal_semantic_axis(signal: Dict[str, Any]) -> str:
    explicit = " ".join(str(signal.get("feature_axis") or "").lower().split())
    if explicit:
        return explicit
    concepts = sorted(_mandatory_feature_concepts(signal.get("abstract_feature")))
    return ":".join(concepts[:8])


def _signals_share_semantic_axis(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> bool:
    left_axis = " ".join(str(left.get("feature_axis") or "").lower().split())
    right_axis = " ".join(str(right.get("feature_axis") or "").lower().split())
    if left_axis and right_axis:
        return left_axis == right_axis
    left_text = " ".join(str(left.get("abstract_feature") or "").lower().split())
    right_text = " ".join(str(right.get("abstract_feature") or "").lower().split())
    if left_text and left_text == right_text:
        return True
    left_concepts = (
        _mandatory_feature_concepts(left_text) - _CROSS_REVIEW_AXIS_STOPWORDS
    )
    right_concepts = (
        _mandatory_feature_concepts(right_text) - _CROSS_REVIEW_AXIS_STOPWORDS
    )
    return len(left_concepts & right_concepts) >= 2


def _cross_review_joint_resolution_groups(
    signals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    supported_ids = {
        str(signal.get("signal_id") or "")
        for signal in signals
        if signal.get("generalization_status") == "supported"
    }
    grouped: Dict[tuple[str, str], set[str]] = {}
    for signal in signals:
        if signal.get("generalization_status") != "supported":
            continue
        peers = {
            str(value)
            for value in list(signal.get("joint_resolution_peer_ids") or [])
            if str(value) in supported_ids
        }
        if not peers:
            signal.pop("joint_resolution_required", None)
            signal.pop("joint_resolution_peer_ids", None)
            signal.pop("joint_resolution_reason", None)
            continue
        key = (
            str(signal.get("update_target") or "").lower().strip(),
            str(signal.get("condition_id") or "").upper().strip(),
        )
        grouped.setdefault(key, set()).update(
            {str(signal.get("signal_id") or ""), *peers}
        )
    return [
        {
            "group_id": f"joint_{index:03d}",
            "update_target": key[0],
            "condition_id": key[1],
            "signal_ids": sorted(signal_ids),
            "policy": "apply all listed signals together or apply none",
        }
        for index, (key, signal_ids) in enumerate(
            sorted(grouped.items()),
            start=1,
        )
        if len(signal_ids) >= 2
    ]


def _signals_share_cross_review_scope(
    signal: Dict[str, Any],
    blocker: Dict[str, Any],
) -> bool:
    if signal.get("source_review_index") == blocker.get("source_review_index"):
        return False
    signal_error = str(signal.get("source_error_type") or "").upper().strip()
    blocker_error = str(blocker.get("source_error_type") or "").upper().strip()
    if {signal_error, blocker_error} != {"FP", "FN"}:
        return False
    signal_target = str(signal.get("update_target") or "").lower().strip()
    blocker_target = str(blocker.get("update_target") or "").lower().strip()
    signal_condition = str(signal.get("condition_id") or "").upper().strip()
    blocker_condition = str(blocker.get("condition_id") or "").upper().strip()
    return bool(
        signal_target
        and signal_target == blocker_target
        and signal_condition
        and signal_condition == blocker_condition
    )


def _directions_conflict(left: str, right: str) -> bool:
    broadening = {"broaden", "narrow_exclusion"}
    restricting = {"tighten", "add_exclusion"}
    return bool(
        (left in broadening and right in restricting)
        or (right in broadening and left in restricting)
    )


def _matrix_has_actionable_update_signal(
    matrix: Dict[str, Any],
    *,
    source_review_index: int,
    update_target: str,
    allow_experimental_plan: bool = False,
) -> bool:
    allowed_statuses = {"supported"}
    if allow_experimental_plan and update_target == "plan":
        allowed_statuses.add("experimental_plan")
    if update_target == "rule":
        allowed_statuses.add("experimental_rule")
    return any(
        isinstance(signal, dict)
        and str(signal.get("generalization_status") or "").lower().strip()
        in allowed_statuses
        and int(signal.get("source_review_index", -1)) == source_review_index
        and str(signal.get("update_target") or "") == update_target
        for signal in list((matrix or {}).get("signals", []) or [])
    )


def _matrix_has_supported_update_signal(
    matrix: Dict[str, Any],
    *,
    source_review_index: int,
    update_target: str,
) -> bool:
    return _matrix_has_actionable_update_signal(
        matrix,
        source_review_index=source_review_index,
        update_target=update_target,
    )


def _cohort_answer_key(answer: Any) -> str:
    if answer is True:
        return "true"
    if answer is False:
        return "false"
    return str(answer or "uncertain").lower().strip() or "uncertain"


def _cohort_feature_text(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text or _CONCRETE_FEATURE_RE.search(text):
        return ""
    return text[:280]


def _normalize_matrix_evidence_dependency(
    value: Any,
    *,
    update_target: str,
) -> Dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    role = str(raw.get("role") or "").strip().lower()
    expected = {"rule": "requires", "plan": "provides"}.get(
        str(update_target or "").strip().lower()
    )
    capability_id = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(raw.get("capability_id") or "").strip().lower(),
    ).strip("_")[:96]
    description = _cohort_feature_text(raw.get("capability_description"))
    if role != expected or not capability_id or not description:
        return {}
    return {
        "role": role,
        "capability_id": capability_id,
        "capability_description": description,
    }


_FEATURE_GROUP_CODES = {
    "correct_positive": "cp",
    "correct_negative": "cn",
    "correct_guard": "cg",
}

_FEATURE_TYPE_CODES = {
    "matched_features": "m",
    "partial_match_features": "p",
    "missing_required_features": "r",
    "contradicting_features": "c",
    "boundary_notes": "b",
}


def _feature_observation_items(
    counter: Counter,
    *,
    condition_id: str,
    group_name: str,
    feature_type: str,
) -> List[Dict[str, Any]]:
    group_code = _FEATURE_GROUP_CODES.get(group_name, "cohort")
    type_code = _FEATURE_TYPE_CODES.get(feature_type, "f")
    observations: List[Dict[str, Any]] = []
    for index, (feature, count) in enumerate(counter.most_common(), start=1):
        observations.append({
            "feature_ref": f"{condition_id}.{group_code}.{type_code}.{index:02d}",
            "feature": feature,
            "support_count": int(count),
        })
    return observations


def _apply_cohort_feature_budget(
    conditions: Dict[str, Any],
    *,
    max_chars: int,
    max_per_bucket: int,
) -> Dict[str, Any]:
    """Budget feature observations without favoring the first condition/group."""
    buckets: List[List[Dict[str, Any]]] = []
    destinations: List[List[Dict[str, Any]]] = []
    total_available = 0
    for condition in conditions.values():
        for group in dict(condition.get("groups") or {}).values():
            for observations in dict(group.get("features") or {}).values():
                source = list(observations or [])
                total_available += len(source)
                buckets.append(source)
                observations.clear()
                destinations.append(observations)

    used_chars = 0
    included = 0
    exhausted = False
    for item_index in range(max(0, int(max_per_bucket))):
        for source, destination in zip(buckets, destinations):
            if item_index >= len(source):
                continue
            item = source[item_index]
            item_chars = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            if used_chars + item_chars > max(0, int(max_chars)):
                exhausted = True
                continue
            destination.append(item)
            used_chars += item_chars
            included += 1

    omitted = max(0, total_available - included)
    return {
        "max_chars": int(max_chars),
        "max_estimated_tokens": (int(max_chars) + 3) // 4,
        "used_chars": used_chars,
        "used_estimated_tokens": (used_chars + 3) // 4,
        "max_per_bucket": int(max_per_bucket),
        "available_count": total_available,
        "included_count": included,
        "omitted_count": omitted,
        "truncated": bool(exhausted or omitted),
        "allocation": "round_robin_across_condition_group_feature_type",
    }


def _counter_items(
    counter: Counter,
    *,
    limit: int,
    key_name: str = "feature",
    min_support: int = 1,
) -> List[Dict[str, Any]]:
    return [
        {key_name: key, "support_count": int(count)}
        for key, count in counter.most_common()
        if int(count) >= int(min_support)
    ][:limit]


def _protected_positive_signals(
    conditions: Dict[str, Any],
    positive_count: int,
) -> List[Dict[str, Any]]:
    if positive_count <= 0:
        return []
    protected: List[Dict[str, Any]] = []
    for condition_id, condition in conditions.items():
        stats = dict((condition.get("groups") or {}).get("correct_positive") or {})
        answers = dict(stats.get("answer_counts") or {})
        if not stats or int(stats.get("case_count", 0) or 0) != positive_count:
            continue
        expected = "false" if condition.get("is_exclusion") else "true"
        if int(answers.get(expected, 0) or 0) == positive_count:
            protected.append({
                "condition_id": condition_id,
                "expected_answer": expected,
                "support_count": positive_count,
            })
    return protected


def _protected_boundary_signals(conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
    protected: List[Dict[str, Any]] = []
    for condition_id, condition in conditions.items():
        for group_name in ("correct_negative", "correct_guard"):
            stats = dict((condition.get("groups") or {}).get(group_name) or {})
            case_count = int(stats.get("case_count", 0) or 0)
            if case_count <= 0:
                continue
            answers = dict(stats.get("answer_counts") or {})
            boundary_answer = "true" if condition.get("is_exclusion") else "false"
            support_count = int(answers.get(boundary_answer, 0) or 0)
            if support_count:
                protected.append({
                    "condition_id": condition_id,
                    "group": group_name,
                    "boundary_answer": boundary_answer,
                    "support_count": support_count,
                    "case_count": case_count,
                })
    return protected


def reviewer_case_contains_full_evidence_views(case: Dict[str, Any]) -> bool:
    """Debug helper: reviewer input should not contain full packet evidence."""
    text_keys = {"trace_view", "critical_call_view", "event_view", "state_change_view"}
    if not isinstance(case, dict):
        return False
    stack = [case]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            for key in text_keys & set(obj.keys()):
                if not _looks_like_view_summary_entry(obj.get(key)):
                    return True
            for key, value in obj.items():
                if key == "packet_view_summary":
                    continue
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(obj, list):
            stack.extend(v for v in obj if isinstance(v, (dict, list)))
    return False


def _looks_like_view_summary_entry(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "row_count" in value
        and "available" in value
        and "description" in value
    )


def _is_slim_result(result: Dict[str, Any]) -> bool:
    return isinstance(result, dict) and result.get("schema_version") == SLIM_RESULT_SCHEMA_VERSION


def _is_reviewer_case(result: Dict[str, Any]) -> bool:
    return isinstance(result, dict) and result.get("schema_version") == REVIEWER_CASE_SCHEMA_VERSION


def _transaction(result: Dict[str, Any]) -> Dict[str, Any]:
    if "transaction" in result:
        return dict(result.get("transaction", {}) or {})
    return {
        "tx_hash": result.get("tx_hash", "unknown"),
        "chain": result.get("chain", "eth"),
    }


def _detector_context(result: Dict[str, Any]) -> Dict[str, Any]:
    if "detector_context" in result:
        return dict(result.get("detector_context", {}) or {})
    return {
        "attack_label": result.get("attack_label", "attack"),
        "rule_id": result.get("rule_id", ""),
        "rule_version": result.get("rule_version", 1),
        "rule_source": result.get("rule_source"),
    }


def _inference(result: Dict[str, Any]) -> Dict[str, Any]:
    return dict(result.get("inference", {}) or {})


def _evaluation(result: Dict[str, Any]) -> Dict[str, Any]:
    if "evaluation" in result:
        return dict(result.get("evaluation", {}) or {})
    return {
        "ground_truth": result.get("ground_truth"),
        "raw_ground_truth": result.get("raw_ground_truth"),
        "label_rationale": result.get("label_rationale", ""),
        "report": result.get("report", ""),
        "case_metadata": result.get("case_metadata", {}),
    }


def _rule(result: Dict[str, Any]) -> Dict[str, Any]:
    inference = _inference(result)
    if "rule" in inference:
        return dict(inference.get("rule", {}) or {})
    if "rule" in result:
        return dict(result.get("rule", {}) or {})
    if "rule_digest" in result:
        return dict(result.get("rule_digest", {}) or {})
    return {}


def _plan(result: Dict[str, Any]) -> Dict[str, Any]:
    inference = _inference(result)
    if "plan" in inference:
        return dict(inference.get("plan", {}) or {})
    if "plan" in result:
        return dict(result.get("plan", {}) or {})
    if "plan_digest" in result:
        return dict(result.get("plan_digest", {}) or {})
    return {}


def _trace(result: Dict[str, Any]) -> Dict[str, Any]:
    inference = _inference(result)
    if "trace" in inference:
        return dict(inference.get("trace", {}) or {})
    return dict(result.get("trace", {}) or {})


def _evidence(result: Dict[str, Any]) -> Dict[str, Any]:
    inference = _inference(result)
    if "evidence" in inference:
        return dict(inference.get("evidence", {}) or {})
    return dict(result.get("evidence", {}) or {})


def _finding(result: Dict[str, Any]) -> Dict[str, Any]:
    inference = _inference(result)
    if "finding" in inference:
        return dict(inference.get("finding", {}) or {})
    if "finding" in result:
        return dict(result.get("finding", {}) or {})
    if "finding_summary" in result:
        return dict(result.get("finding_summary", {}) or {})
    return {
        "verdict": result.get("predicted_verdict"),
        "confidence": result.get("confidence", "medium"),
    }


def _rule_digest(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rule_id": rule.get("rule_id", ""),
        "version": int(rule.get("version", 1) or 1),
        "name": rule.get("name", ""),
        "description": _shorten(rule.get("description", ""), 2200),
        "conditions": [
            {
                "id": item.get("id"),
                "description": _shorten(item.get("description", ""), 1000),
                "expected_answer": item.get("expected_answer", True),
            }
            for item in list(rule.get("conditions", []) or [])
            if isinstance(item, dict)
        ],
        "exclusion_conditions": [
            {
                "id": item.get("id"),
                "description": _shorten(item.get("description", ""), 1000),
                "expected_answer": item.get("expected_answer", True),
            }
            for item in list(rule.get("exclusion_conditions", []) or [])
            if isinstance(item, dict)
        ],
        "decision_policy": _shorten(rule.get("decision_policy", ""), 1600),
        "metadata": {
            "attack_label": (rule.get("metadata", {}) or {}).get("attack_label"),
            "source": (rule.get("metadata", {}) or {}).get("source"),
        },
    }


def _plan_digest(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_id": plan.get("plan_id", ""),
        "emit_logic": plan.get("emit_logic", ""),
        "judge_steps": [
            {
                "id": item.get("id"),
                "condition_id": item.get("condition_id", item.get("id")),
                "question": _shorten(item.get("question", ""), 700),
                "expected_answer": item.get("expected_answer", True),
                "default_evidence_refs": list(item.get("default_evidence_refs", item.get("evidence_refs", [])) or []),
                "allowed_followup_views": list(item.get("allowed_followup_views", []) or []),
                "allowed_tools": list(item.get("allowed_tools", []) or []),
                "max_followups": item.get("max_followups", 0),
                "depends_on": list(item.get("depends_on", []) or []),
                "consumes_state_keys": list(item.get("consumes_state_keys", []) or []),
                "produces_state_key": item.get("produces_state_key", ""),
                "state_prompt_role": item.get("state_prompt_role", ""),
            }
            for item in list(plan.get("judge_steps", []) or [])
            if isinstance(item, dict)
        ],
    }


def _condition_feature_analysis_from_sources(
    *sources: Dict[str, Any],
) -> Dict[str, Any]:
    fallback = normalize_condition_feature_analysis({})
    direct_candidates: List[Any] = []
    parsed_candidates: List[Any] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        if "condition_feature_analysis" in source:
            direct_candidates.append(source.get("condition_feature_analysis"))
        parsed_response = source.get("parsed_response")
        if isinstance(parsed_response, dict) and "condition_feature_analysis" in parsed_response:
            parsed_candidates.append(parsed_response.get("condition_feature_analysis"))
    for candidate in direct_candidates + parsed_candidates:
        normalized = normalize_condition_feature_analysis(candidate)
        if _condition_feature_analysis_non_empty(normalized):
            return normalized
        fallback = normalized
    return fallback


def _condition_feature_analysis_non_empty(value: Dict[str, Any]) -> bool:
    normalized = normalize_condition_feature_analysis(value)
    return any(bool(items) for items in normalized.values())


def _first_non_empty_dict(*sources: Dict[str, Any], key: str) -> Dict[str, Any]:
    fallback: Dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        value = source.get(key)
        if isinstance(value, dict):
            if value:
                return dict(value)
            fallback = dict(value)
    return fallback


def _build_condition_table(
    plan: Dict[str, Any],
    trace: Dict[str, Any],
    finding: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    plan_steps = {
        str(step.get("id")): step
        for step in list(plan.get("judge_steps", []) or [])
        if isinstance(step, dict)
    }
    execution = trace.get("execution", {}) if isinstance(trace, dict) else {}
    trace_judges = {
        str(j.get("judge_id") or j.get("id")): j
        for j in list(execution.get("judge_calls", []) or [])
        if isinstance(j, dict)
    }
    step_traces = {
        str(j.get("judge_id") or j.get("id")): j
        for j in list(execution.get("judge_step_traces", []) or [])
        if isinstance(j, dict)
    }
    ids = list(plan_steps)
    for jid in trace_judges:
        if jid not in ids:
            ids.append(jid)

    rows: List[Dict[str, Any]] = []
    for jid in ids:
        step = plan_steps.get(jid, {})
        judge = trace_judges.get(jid, {})
        step_trace = step_traces.get(jid, {})
        rounds = list(step_trace.get("judge_calls", []) or [])
        final_round = rounds[-1] if rounds else {}
        tool_calls = _summarize_step_tool_calls(
            step_trace.get("tool_calls", []),
            default_judge_id=jid,
            default_condition_id=step_trace.get("condition_id") or step.get("condition_id") or jid,
        )
        answer = judge.get("answer", final_round.get("answer", "uncertain"))
        first_round = rounds[0] if rounds else {}
        first_round_answer = first_round.get("answer", answer)
        final_answer = final_round.get("answer", answer)
        final_reason = judge.get("reason", final_round.get("reason", ""))
        condition_feature_analysis = _condition_feature_analysis_from_sources(
            judge,
            final_round,
            first_round,
        )
        state_input = _first_non_empty_dict(
            judge,
            final_round,
            first_round,
            step_trace,
            key="state_input",
        )
        state_output = _first_non_empty_dict(
            judge,
            final_round,
            first_round,
            step_trace,
            key="state_output",
        )
        stateful_runtime = _first_non_empty_dict(
            judge,
            final_round,
            first_round,
            step_trace,
            key="stateful_runtime",
        )
        condition_id = str(
            judge.get("condition_id")
            or step.get("condition_id")
            or step_trace.get("condition_id")
            or jid
        )
        is_exclusion = condition_id.upper().startswith("E") or str(jid).upper().startswith("E")
        missing = list(judge.get("missing_evidence", final_round.get("missing_evidence", [])) or [])
        row = {
            "id": jid,
            "condition_id": condition_id,
            "is_exclusion": is_exclusion,
            "question": _shorten(judge.get("question") or step.get("question", ""), 900),
            "expected_answer": judge.get("expected_answer", step.get("expected_answer", True)),
            "answer": answer,
            "first_round_answer": first_round_answer,
            "final_answer": final_answer,
            "followup_changed_answer": bool(rounds) and first_round_answer != final_answer,
            "confidence": judge.get("confidence", final_round.get("confidence", "medium")),
            "reason_short": _shorten(final_reason, 450),
            "supporting_evidence_ids": list(
                judge.get("supporting_evidence_ids", final_round.get("supporting_evidence_ids", [])) or []
            ),
            "contradicting_evidence_ids": list(
                judge.get("contradicting_evidence_ids", final_round.get("contradicting_evidence_ids", [])) or []
            ),
            "missing_evidence": missing,
            "tool_requests": _collect_tool_requests(rounds, judge),
            "condition_feature_analysis": condition_feature_analysis,
            "state_input": state_input,
            "state_output": state_output,
            "stateful_runtime": stateful_runtime,
            "tool_observations": tool_calls,
            "tool_request_count": len(_collect_tool_requests(rounds, judge)),
            "tool_call_count": len(tool_calls),
            "reused_judge_result": bool(
                judge.get("reused_judge_result")
                or step_trace.get("reused_judge_result")
            ),
            "reuse_fingerprint": judge.get("reuse_fingerprint")
            or step_trace.get("reuse_fingerprint", ""),
            "reuse_source": dict(
                judge.get("reuse_source")
                or step_trace.get("reuse_source")
                or {}
            ),
            "ignored_tool_requests": list(
                judge.get("ignored_tool_requests")
                or step_trace.get("ignored_tool_requests")
                or []
            ),
            "near_miss_escalated": bool(
                judge.get("near_miss_escalated")
                or step_trace.get("near_miss_escalation")
            ),
            "near_miss_pre_escalation": dict(
                judge.get("near_miss_pre_escalation")
                or step_trace.get("near_miss_escalation")
                or {}
            ),
            "judge_error": dict(
                judge.get("judge_error")
                or step_trace.get("judge_error")
                or {}
            ),
            "source_decisiveness_guards": list(
                judge.get("source_decisiveness_guards")
                or step_trace.get("source_decisiveness_guards")
                or []
            ),
            "do_not_train": bool(
                judge.get("do_not_train")
                or step_trace.get("do_not_train")
                or (
                    isinstance(
                        judge.get("judge_error")
                        or step_trace.get("judge_error"),
                        dict,
                    )
                    and bool(
                        (
                            judge.get("judge_error")
                            or step_trace.get("judge_error")
                            or {}
                        ).get("do_not_train")
                    )
                )
            ),
            "all_missing_evidence_debug": list(
                judge.get("all_missing_evidence_debug")
                or step_trace.get("all_missing_evidence_debug")
                or _flatten_round_missing_debug(rounds)
            ),
            "view_render_metadata": list(
                judge.get("view_render_metadata")
                or step_trace.get("view_render_metadata")
                or []
            ),
            "final_round": final_round.get("round", len(rounds) - 1 if rounds else 0),
            "error_hint": "",
        }
        row["error_hint"] = _condition_error_hint(row, finding, evaluation)
        rows.append(row)
    return rows


def _summarize_step_tool_calls(
    tool_calls: Any,
    default_judge_id: str = "",
    default_condition_id: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for call in list(tool_calls or []):
        if not isinstance(call, dict):
            continue
        obs = call.get("observation", {}) if isinstance(call.get("observation"), dict) else {}
        item = {
            "judge_id": call.get("judge_id") or default_judge_id,
            "condition_id": call.get("condition_id") or default_condition_id,
            "round": call.get("round"),
            "tool": call.get("tool"),
            "tool_status": call.get("tool_status") or obs.get("tool_status"),
            "args": _compact(call.get("args", {}), max_chars=1200),
            "summary": _compact(call.get("summary") or obs.get("summary", {}), max_chars=1600),
            "returned_evidence_ids": list(call.get("returned_evidence_ids") or obs.get("returned_evidence_ids") or []),
            "validation_error": call.get("validation_error"),
        }
        source_evidence = _source_evidence_digest(obs)
        if source_evidence:
            item["source_evidence"] = source_evidence
        out.append(item)
    return out


def _source_evidence_digest(observation: Any, max_chars: int = 8000) -> Dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    status = str(observation.get("tool_status") or observation.get("status") or "").lower()
    evidence = observation.get("evidence") if isinstance(observation.get("evidence"), dict) else {}
    raw_snippets = evidence.get("snippets") or observation.get("snippets") or []
    if status != "ok" or not raw_snippets:
        return {}
    snippets = []
    for raw in list(raw_snippets or [])[:4]:
        item = raw if isinstance(raw, dict) else {"source_code": str(raw)}
        snippets.append({
            "evidence_id": item.get("evidence_id", ""),
            "path": item.get("path", ""),
            "start_line": item.get("start_line"),
            "end_line": item.get("end_line"),
            "matched_key": item.get("matched_key", ""),
            "source_code": _shorten(item.get("source_code", ""), max(400, max_chars // 4)),
            "truncated": bool(item.get("truncated")),
        })
    return _compact({"snippets": snippets}, max_chars=max_chars)


def _flatten_round_missing_debug(rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for round_item in rounds:
        for item in list(round_item.get("missing_evidence_debug", []) or []):
            if isinstance(item, dict):
                out.append(dict(item))
    return out


def _collect_tool_requests(rounds: List[Dict[str, Any]], judge: Dict[str, Any]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for round_item in rounds:
        for request in list(round_item.get("tool_requests", []) or []):
            if isinstance(request, dict):
                requests.append(_compact(request, max_chars=1800))
    for request in list(judge.get("tool_requests", []) or []):
        if isinstance(request, dict):
            requests.append(_compact(request, max_chars=1800))
    return requests[:4]


def _condition_error_hint(
    row: Dict[str, Any],
    finding: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> str:
    gold = evaluation.get("ground_truth")
    pred = finding.get("verdict")
    answer = row.get("answer")
    judge_error = row.get("judge_error") if isinstance(row.get("judge_error"), dict) else {}
    if judge_error or row.get("do_not_train"):
        return "judge_transport_error"
    if answer == "uncertain":
        return "condition_uncertain"
    if gold == "attack" and pred != "attack" and not row.get("is_exclusion") and answer is not True:
        return "failed_core_condition_for_positive_case"
    if gold == "benign" and pred == "attack" and row.get("is_exclusion") and answer is not True:
        return "benign_exclusion_not_supported_for_negative_case"
    if row.get("tool_observations"):
        blocked = [t for t in row["tool_observations"] if t.get("tool_status") == "blocked"]
        if blocked:
            return "followup_tool_blocked"
    return ""


def _finding_summary(
    finding: Dict[str, Any],
    *,
    missing_status_by_condition: Optional[Dict[str, Dict[str, List[str]]]] = None,
    all_missing_debug: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    open_missing = list(finding.get("missing_evidence", []) or [])[:80]
    return {
        "verdict": finding.get("verdict"),
        "confidence": finding.get("confidence", "medium"),
        "verdict_reason": finding.get("verdict_reason", ""),
        "verdict_aggregation": dict(finding.get("verdict_aggregation", {}) or {}),
        "supporting_evidence": list(finding.get("supporting_evidence", []) or [])[:80],
        "supporting_evidence_count": len(list(finding.get("supporting_evidence", []) or [])),
        "judge_results": list(finding.get("judge_results", []) or []),
        "missing_evidence": open_missing,
        "attack_supporting_evidence_by_condition": dict(
            finding.get("attack_supporting_evidence_by_condition", {}) or {}
        ),
        "benign_exclusion_evidence_by_condition": dict(
            finding.get("benign_exclusion_evidence_by_condition", {}) or {}
        ),
        "failed_core_conditions": dict(finding.get("failed_core_conditions", {}) or {}),
        "missing_evidence_by_condition": missing_status_by_condition
        or _legacy_missing_by_condition(finding.get("missing_evidence_by_condition", {}) or {}),
        "all_missing_evidence_debug": list(
            all_missing_debug
            if all_missing_debug is not None
            else finding.get("all_missing_evidence_debug", [])
            or []
        ),
        "ignored_tool_requests": list(finding.get("ignored_tool_requests", []) or []),
    }


def _legacy_missing_by_condition(raw: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    out = {}
    for condition_id, value in dict(raw or {}).items():
        if isinstance(value, dict) and {"open", "resolved", "stale", "non_actionable"} & set(value.keys()):
            out[str(condition_id)] = {
                "open": list(value.get("open", []) or []),
                "resolved": list(value.get("resolved", []) or []),
                "stale": list(value.get("stale", []) or []),
                "non_actionable": list(value.get("non_actionable", []) or []),
            }
        else:
            out[str(condition_id)] = {
                "open": list(value if isinstance(value, list) else [value]),
                "resolved": [],
                "stale": [],
                "non_actionable": [],
            }
    return out


def _supporting_by_condition(
    finding: Dict[str, Any],
    condition_table: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_condition = {
        row["condition_id"]: list(row.get("supporting_evidence_ids", []) or [])
        for row in condition_table
    }
    return {
        "all_by_condition": by_condition,
        "attack_supporting_evidence_by_condition": dict(
            finding.get("attack_supporting_evidence_by_condition", {}) or {}
        ),
        "benign_exclusion_evidence_by_condition": dict(
            finding.get("benign_exclusion_evidence_by_condition", {}) or {}
        ),
    }


def _missing_by_condition(
    finding: Dict[str, Any],
    condition_table: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    out = {
        row["condition_id"]: list(row.get("missing_evidence", []) or [])
        for row in condition_table
        if row.get("missing_evidence")
    }
    out.update(dict(finding.get("missing_evidence_by_condition", {}) or {}))
    return out


def _all_missing_debug(
    finding: Dict[str, Any],
    condition_table: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in list(finding.get("all_missing_evidence_debug", []) or []):
        if isinstance(item, dict):
            items.append(dict(item))
    for row in condition_table:
        for item in list(row.get("all_missing_evidence_debug", []) or []):
            if isinstance(item, dict):
                items.append(dict(item))
    if items:
        return _dedupe_missing_debug(items)
    for row in condition_table:
        for text in list(row.get("missing_evidence", []) or []):
            items.append({
                "condition_id": row.get("condition_id"),
                "round": row.get("final_round", 0),
                "text": text,
                "category": "unknown",
                "status": "open",
                "blocking": True,
                "resolved_by": None,
            })
    return _dedupe_missing_debug(items)


def _dedupe_missing_debug(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        key = (
            item.get("condition_id"),
            item.get("round"),
            item.get("text"),
            item.get("category"),
            item.get("status"),
            item.get("resolved_by"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _tool_call_summary(
    trace: Dict[str, Any],
    condition_table: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    calls = []
    execution = trace.get("execution", {}) if isinstance(trace, dict) else {}
    raw_calls = list(execution.get("tool_calls", []) or [])
    if not raw_calls or any(isinstance(call, dict) and not call.get("judge_id") for call in raw_calls):
        for row in condition_table:
            for call in row.get("tool_observations", []):
                calls.append({
                    **call,
                    "judge_id": call.get("judge_id") or row.get("id"),
                    "condition_id": call.get("condition_id") or row.get("condition_id"),
                })
        return calls
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        calls.append({
            "judge_id": call.get("judge_id"),
            "condition_id": call.get("condition_id"),
            "round": call.get("round"),
            "tool": call.get("tool"),
            "tool_status": call.get("tool_status"),
            "args": _compact(call.get("args", {}), max_chars=1200),
            "summary": _compact(call.get("summary", {}), max_chars=1600),
            "returned_evidence_ids": list(call.get("returned_evidence_ids", []) or []),
            "validation_error": call.get("validation_error"),
        })
    return calls


def _packet_view_summary(evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in PACKET_VIEW_CATALOG:
        available = name in evidence
        out[name] = {
            "available": available,
            "row_count": packet_view_row_count(evidence.get(name)) if available else 0,
            "description": PACKET_VIEW_CATALOG[name].get("description", ""),
        }
    return out


def _evidence_id_resolution(
    finding: Dict[str, Any],
    condition_table: List[Dict[str, Any]],
    evidence: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    supporting = _unique(
        list(finding.get("supporting_evidence", []) or [])
        + _flatten_values(finding.get("attack_supporting_evidence_by_condition", {}))
        + _flatten_values(finding.get("benign_exclusion_evidence_by_condition", {}))
        + [
            eid
            for row in condition_table
            for eid in list(row.get("supporting_evidence_ids", []) or [])
        ]
    )
    supporting = _unique(_normalize_evidence_id(eid) for eid in supporting)
    known = {_normalize_evidence_id(eid) for eid in _collect_evidence_ids(evidence)}
    for call in tool_calls:
        known.update(_normalize_evidence_id(eid) for eid in call.get("returned_evidence_ids", []) if eid)
    unresolved = [eid for eid in supporting if eid not in known]
    return {
        "supporting_total": len(supporting),
        "resolved": len(supporting) - len(unresolved),
        "unresolved": len(unresolved),
        "unresolved_ids": unresolved[:80],
    }


def _normalize_evidence_id(value: Any) -> str:
    text = str(value or "")
    if text.lower().startswith("address:"):
        return "address:" + text.split(":", 1)[1].lower()
    return text


def _collect_evidence_ids(obj: Any) -> List[str]:
    ids: List[str] = []
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if current.get("evidence_id"):
                ids.append(str(current["evidence_id"]))
            stack.extend(v for v in current.values() if isinstance(v, (dict, list)))
        elif isinstance(current, list):
            stack.extend(v for v in current if isinstance(v, (dict, list)))
    return _unique(ids)


def _infer_error_type_from_values(gold: Any, pred: Any, default: str = "unknown") -> str:
    if gold == "attack" and pred != "attack":
        return "FN"
    if gold == "benign" and pred == "attack":
        return "FP"
    return default


def _compact(value: Any, max_chars: int = 4000) -> Any:
    text = repr(value)
    if len(text) <= max_chars:
        return value
    return {
        "truncated": True,
        "approx_chars": len(text),
        "preview": text[:max_chars],
    }


def _shorten(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...<truncated {len(text) - max_chars} chars>"


def _flatten_values(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            out.extend(_flatten_values(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_flatten_values(value))
    elif obj is not None:
        out.append(str(obj))
    return out


def _flatten_open_missing(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        if "open" in obj and isinstance(obj.get("open"), list):
            return [str(x) for x in obj.get("open", []) if x]
        for value in obj.values():
            out.extend(_flatten_open_missing(value))
    elif isinstance(obj, list):
        out.extend(str(x) for x in obj if x)
    elif obj is not None:
        out.append(str(obj))
    return out


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
