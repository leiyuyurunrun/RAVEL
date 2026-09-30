from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from evotx.utils.result_utils import get_finding, get_ground_truth


@dataclass
class RegressionSummary:
    total: int
    correct: int
    errors: int
    fp: int
    fn: int
    uncertain: int
    uncertain_attack: int = 0
    uncertain_benign: int = 0
    negative_benign: int = 0
    negative_other_attack: int = 0
    negative_unknown_other: int = 0
    fp_negative_benign: int = 0
    fp_negative_other_attack: int = 0
    fp_negative_unknown_other: int = 0

    @property
    def negative_total(self) -> int:
        return (
            self.negative_benign
            + self.negative_other_attack
            + self.negative_unknown_other
        )

    @property
    def target_total(self) -> int:
        return max(0, self.total - self.negative_total)

    @property
    def attack_tp(self) -> int:
        # summarize() already counts uncertain attack predictions in fn.
        return max(0, self.target_total - self.fn)

    @property
    def predicted_attack_total(self) -> int:
        return max(0, self.attack_tp + self.fp)

    @property
    def attack_recall(self) -> float | None:
        if self.target_total <= 0:
            return None
        return self.attack_tp / self.target_total

    @property
    def attack_precision(self) -> float | None:
        predicted = self.predicted_attack_total
        if predicted <= 0:
            return None
        return self.attack_tp / predicted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "errors": self.errors,
            "fp": self.fp,
            "fn": self.fn,
            "uncertain": self.uncertain,
            "uncertain_by_gold": {
                "attack": self.uncertain_attack,
                "benign": self.uncertain_benign,
            },
            "negative_breakdown": {
                "benign": self.negative_benign,
                "other_attack": self.negative_other_attack,
                "unknown_other": self.negative_unknown_other,
            },
            "fp_by_negative_kind": {
                "benign": self.fp_negative_benign,
                "other_attack": self.fp_negative_other_attack,
                "unknown_other": self.fp_negative_unknown_other,
            },
            "attack_tp": self.attack_tp,
            "attack_target_total": self.target_total,
            "attack_predicted_total": self.predicted_attack_total,
            "attack_recall": (
                round(self.attack_recall, 6)
                if self.attack_recall is not None
                else None
            ),
            "attack_precision": (
                round(self.attack_precision, 6)
                if self.attack_precision is not None
                else None
            ),
            "precision": (
                round(self.attack_precision, 6)
                if self.attack_precision is not None
                else None
            ),
        }


@dataclass
class CandidateRepairPolicy:
    """Policy for rejected-candidate boundary repair.

    This policy does not relax final regression acceptance. It only decides
    whether a rejected rule candidate is worth one targeted review/update pass
    over newly introduced false positives.
    """

    enabled: bool = True
    final_max_fp_increase: int = 1
    temporary_fp_increase: int = 2
    repair_min_fn_decrease: int = 1
    max_repair_rounds: int = 1

    @classmethod
    def from_args(cls, args: Any) -> "CandidateRepairPolicy":
        return cls(
            enabled=bool(getattr(args, "enable_candidate_repair", True)),
            final_max_fp_increase=int(getattr(args, "max_fp_increase", 1) or 0),
            temporary_fp_increase=int(getattr(args, "temporary_fp_increase", 2) or 0),
            repair_min_fn_decrease=int(getattr(args, "repair_min_fn_decrease", 1) or 0),
            max_repair_rounds=int(getattr(args, "max_repair_rounds", 1) or 0),
        ).normalized()

    def normalized(self) -> "CandidateRepairPolicy":
        final_max_fp_increase = max(0, int(self.final_max_fp_increase or 0))
        temporary_fp_increase = max(0, int(self.temporary_fp_increase or 0))
        repair_min_fn_decrease = max(0, int(self.repair_min_fn_decrease or 0))
        max_repair_rounds = max(0, int(self.max_repair_rounds or 0))
        enabled = bool(self.enabled)
        if max_repair_rounds <= 0:
            enabled = False
        if temporary_fp_increase <= 0:
            enabled = False
        return CandidateRepairPolicy(
            enabled=enabled,
            final_max_fp_increase=final_max_fp_increase,
            temporary_fp_increase=temporary_fp_increase,
            repair_min_fn_decrease=repair_min_fn_decrease,
            max_repair_rounds=max_repair_rounds,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "final_max_fp_increase": self.final_max_fp_increase,
            "temporary_fp_increase": self.temporary_fp_increase,
            "repair_min_fn_decrease": self.repair_min_fn_decrease,
            "max_repair_rounds": self.max_repair_rounds,
            "note": (
                "Temporary FP increase only permits a targeted repair attempt; "
                "final repaired candidate acceptance still uses the attack "
                "recall/precision gate. New FP cases remain useful as targeted "
                "boundary-repair examples when precision would regress."
            ),
        }


class RegressionEvaluator:
    """Compare old/new rule results on verified positives and hard negatives."""

    def should_accept(
        self,
        old_results: List[Dict[str, Any]],
        new_results: List[Dict[str, Any]],
        max_fp_increase: int = 1,
        max_fn_increase: int = 1,
        max_uncertain_increase: int = 0,
        max_uncertain_benign_increase: int = 0,
        acceptance_mode: str = "rule",
        enforce_uncertain_gate: bool = True,
    ) -> bool:
        old = self.summarize(old_results)
        new = self.summarize(new_results)

        decision = self._acceptance_decision(
            old,
            new,
            max_fp_increase=max_fp_increase,
            max_fn_increase=max_fn_increase,
            max_uncertain_increase=max_uncertain_increase,
            max_uncertain_benign_increase=max_uncertain_benign_increase,
            acceptance_mode=acceptance_mode,
            enforce_uncertain_gate=enforce_uncertain_gate,
        )
        return bool(decision["accept"])

    def compare(
        self,
        old_results: List[Dict[str, Any]],
        new_results: List[Dict[str, Any]],
        max_fp_increase: int = 1,
        max_fn_increase: int = 1,
        max_uncertain_increase: int = 0,
        max_uncertain_benign_increase: int = 0,
        acceptance_mode: str = "rule",
        enforce_uncertain_gate: bool = True,
    ) -> Dict[str, Any]:
        old = self.summarize(old_results)
        new = self.summarize(new_results)
        decision = self._acceptance_decision(
            old,
            new,
            max_fp_increase=max_fp_increase,
            max_fn_increase=max_fn_increase,
            max_uncertain_increase=max_uncertain_increase,
            max_uncertain_benign_increase=max_uncertain_benign_increase,
            acceptance_mode=acceptance_mode,
            enforce_uncertain_gate=enforce_uncertain_gate,
        )
        per_case_changes = self._per_case_changes(old_results, new_results)
        change_counts = _count_change_types(per_case_changes)
        uncertain_gate = dict(decision.get("uncertain_gate") or {})
        new_uncertain_cases = [
            item for item in per_case_changes if str(item.get("change_type")) in {
                "new_uncertain_benign",
                "fp_to_uncertain",
            }
        ]
        fixed_uncertain_cases = [
            item for item in per_case_changes if str(item.get("change_type")) in {
                "fixed_uncertain_benign",
                "uncertain_to_correct",
                "uncertain_to_fp",
            }
        ]
        return {
            "old": old.to_dict(),
            "new": new.to_dict(),
            "delta": {
                "errors": new.errors - old.errors,
                "fp": new.fp - old.fp,
                "fn": new.fn - old.fn,
                "uncertain": new.uncertain - old.uncertain,
                "uncertain_attack": new.uncertain_attack - old.uncertain_attack,
                "uncertain_benign": new.uncertain_benign - old.uncertain_benign,
            },
            "per_case_changes": per_case_changes,
            "fixed_cases": [
                item for item in per_case_changes if item["change_type"] in {
                    "fixed_fn",
                    "fixed_fp",
                    "fp_to_uncertain",
                    "fixed_uncertain_benign",
                    "uncertain_to_correct",
                }
            ],
            "regressed_cases": [
                item for item in per_case_changes if item["change_type"] in {
                    "new_fn",
                    "new_fp",
                    "new_uncertain_benign",
                    "uncertain_to_fp",
                }
            ],
            "unchanged_errors": [
                item for item in per_case_changes if item["change_type"] in {
                    "unchanged_error",
                    "unchanged_uncertain",
                }
            ],
            "new_uncertain_cases": new_uncertain_cases,
            "fixed_uncertain_cases": fixed_uncertain_cases,
            "accept": decision["accept"],
            "accept_reason": decision["accept_reason"],
            "reject_reason": decision["reject_reason"],
            "uncertain_gate": uncertain_gate,
            "acceptance_policy": {
                "mode": acceptance_mode,
                "max_fp_increase": max_fp_increase,
                "max_fn_increase": max_fn_increase,
                "max_error_increase": 0,
                "max_uncertain_increase": max_uncertain_increase,
                "max_uncertain_benign_increase": max_uncertain_benign_increase,
                "primary_metric": "bounded errors + observable improvement",
                "primary_error": "fp + fn with configured directional budgets",
                "uncertain_policy": (
                    "configured uncertainty budgets are enforced"
                    if enforce_uncertain_gate
                    else "uncertainty is diagnostic only"
                ),
                "priority": [
                    "reject if fp/fn/total-error budgets are exceeded",
                    (
                        "reject if uncertainty budgets are exceeded"
                        if enforce_uncertain_gate
                        else "record uncertainty changes without vetoing acceptance"
                    ),
                    "accept only an observable correct-count or error-count improvement",
                    "record recall and precision as diagnostics",
                    "reject otherwise",
                ],
            },
            "acceptance_tradeoff": {
                **change_counts,
                "fp_regression": new.fp > old.fp,
                "fn_regression": new.fn > old.fn,
                "recall_regression": (
                    (new.attack_recall or 0.0) < (old.attack_recall or 0.0)
                ),
                "precision_regression": (
                    (new.attack_precision or 0.0) < (old.attack_precision or 0.0)
                ),
                "attack_recall_delta": (
                    (new.attack_recall or 0.0) - (old.attack_recall or 0.0)
                ),
                "attack_precision_delta": (
                    (new.attack_precision or 0.0) - (old.attack_precision or 0.0)
                ),
                "uncertain_regression": new.uncertain > old.uncertain,
                "uncertain_benign_regression": new.uncertain_benign > old.uncertain_benign,
                "fp_increase": max(0, new.fp - old.fp),
                "fp_reduction": max(0, old.fp - new.fp),
                "fn_increase": max(0, new.fn - old.fn),
                "fn_reduction": max(0, old.fn - new.fn),
                "uncertain_increase": max(0, new.uncertain - old.uncertain),
                "uncertain_reduction": max(0, old.uncertain - new.uncertain),
                "uncertain_benign_increase": max(
                    0, new.uncertain_benign - old.uncertain_benign
                ),
                "uncertain_benign_reduction": max(
                    0, old.uncertain_benign - new.uncertain_benign
                ),
            },
        }

    def compare_with_guard(
        self,
        old_train_results: List[Dict[str, Any]],
        new_train_results: List[Dict[str, Any]],
        *,
        old_guard_results: List[Dict[str, Any]] | None = None,
        new_guard_results: List[Dict[str, Any]] | None = None,
        max_fp_increase: int = 1,
        max_fn_increase: int = 1,
        max_uncertain_increase: int = 0,
        max_uncertain_benign_increase: int = 0,
        max_guard_fp_increase: int = 0,
        max_guard_error_increase: int = 0,
        max_guard_uncertain_increase: int = 0,
        acceptance_mode: str = "rule",
        enforce_uncertain_gate: bool = True,
    ) -> Dict[str, Any]:
        """Compare train results and apply a fixed hard-negative guard gate.

        The guard set is a dev/acceptance gate only. It does not change the
        normal train-set regression comparison; it can only veto an otherwise
        accepted candidate when hard negatives regress beyond configured
        thresholds.
        """
        train_comparison = self.compare(
            old_train_results,
            new_train_results,
            max_fp_increase=max_fp_increase,
            max_fn_increase=max_fn_increase,
            max_uncertain_increase=max_uncertain_increase,
            max_uncertain_benign_increase=max_uncertain_benign_increase,
            acceptance_mode=acceptance_mode,
            enforce_uncertain_gate=enforce_uncertain_gate,
        )
        old_guard_results = list(old_guard_results or [])
        new_guard_results = list(new_guard_results or [])
        guard_enabled = bool(old_guard_results or new_guard_results)

        result = dict(train_comparison)
        result["train_comparison"] = train_comparison
        result["guard_comparison"] = {}
        result["guard_gate"] = {
            "enabled": guard_enabled,
            "accept": True,
            "reject_reason": "",
            "thresholds": {
                "max_guard_fp_increase": max_guard_fp_increase,
                "max_guard_error_increase": max_guard_error_increase,
                "max_guard_uncertain_increase": max_guard_uncertain_increase,
            },
        }
        result["accept"] = bool(train_comparison.get("accept"))
        result["accept_reason"] = train_comparison.get("accept_reason", "")
        result["reject_reason"] = train_comparison.get("reject_reason", "")

        if not guard_enabled:
            return result

        guard_comparison = self.compare(
            old_guard_results,
            new_guard_results,
            max_fp_increase=max_guard_fp_increase,
            max_fn_increase=max_fn_increase,
            max_uncertain_increase=max_guard_uncertain_increase,
            max_uncertain_benign_increase=max_guard_uncertain_increase,
            acceptance_mode="guard",
        )
        guard_delta = dict(guard_comparison.get("delta") or {})
        fp_delta = int(guard_delta.get("fp", 0) or 0)
        error_delta = int(guard_delta.get("errors", 0) or 0)
        uncertain_delta = int(guard_delta.get("uncertain", 0) or 0)
        violations = []
        if fp_delta > int(max_guard_fp_increase or 0):
            violations.append(
                f"guard FP increased by {fp_delta}, allowed {max_guard_fp_increase}"
            )
        if error_delta > int(max_guard_error_increase or 0):
            violations.append(
                f"guard errors increased by {error_delta}, allowed {max_guard_error_increase}"
            )
        if uncertain_delta > int(max_guard_uncertain_increase or 0):
            violations.append(
                "guard uncertain increased by "
                f"{uncertain_delta}, allowed {max_guard_uncertain_increase}"
            )

        guard_accept = not violations
        guard_gate = {
            "enabled": True,
            "accept": guard_accept,
            "reject_reason": "; ".join(violations),
            "thresholds": {
                "max_guard_fp_increase": max_guard_fp_increase,
                "max_guard_error_increase": max_guard_error_increase,
                "max_guard_uncertain_increase": max_guard_uncertain_increase,
            },
            "delta": {
                "errors": error_delta,
                "fp": fp_delta,
                "fn": int(guard_delta.get("fn", 0) or 0),
                "uncertain": uncertain_delta,
                "uncertain_benign": int(guard_delta.get("uncertain_benign", 0) or 0),
                "uncertain_attack": int(guard_delta.get("uncertain_attack", 0) or 0),
            },
            "old_guard": dict(guard_comparison.get("old") or {}),
            "new_guard": dict(guard_comparison.get("new") or {}),
            "fixed_cases": list(guard_comparison.get("fixed_cases") or []),
            "regressed_cases": list(guard_comparison.get("regressed_cases") or []),
            "guard_new_fp_txs": sorted(new_fp_txs_from_comparison(guard_comparison)),
            "guard_new_uncertain_txs": sorted(new_uncertain_txs_from_comparison(guard_comparison)),
        }

        result["guard_comparison"] = guard_comparison
        result["guard_gate"] = guard_gate
        result["accept"] = bool(train_comparison.get("accept")) and guard_accept
        if result["accept"]:
            result["accept_reason"] = (
                f"{train_comparison.get('accept_reason', '')} "
                "Hard-negative guard did not regress."
            ).strip()
            result["reject_reason"] = ""
        elif not train_comparison.get("accept"):
            result["accept_reason"] = ""
            result["reject_reason"] = train_comparison.get("reject_reason", "")
        else:
            result["accept_reason"] = ""
            result["reject_reason"] = (
                "Rejected by hard-negative guard: "
                f"{guard_gate['reject_reason']}"
            )
        result["acceptance_policy"] = {
            **dict(result.get("acceptance_policy") or {}),
            "hard_negative_guard": {
                "enabled": True,
                "used_for_review": False,
                "max_guard_fp_increase": max_guard_fp_increase,
                "max_guard_error_increase": max_guard_error_increase,
                "max_guard_uncertain_increase": max_guard_uncertain_increase,
                "max_uncertain_increase": max_uncertain_increase,
                "max_uncertain_benign_increase": max_uncertain_benign_increase,
                "note": (
                    "Train comparison must accept first; guard comparison can "
                    "only veto candidates that regress fixed hard negatives."
                ),
            },
        }
        return result

    @staticmethod
    def summarize(results: List[Dict[str, Any]]) -> RegressionSummary:
        total = len(results)
        fp = fn = uncertain = correct = 0
        uncertain_attack = uncertain_benign = 0
        negative_benign = negative_other_attack = negative_unknown_other = 0
        fp_negative_benign = fp_negative_other_attack = fp_negative_unknown_other = 0
        for result in results:
            pred = get_finding(result).get("verdict")
            gold = get_ground_truth(result)
            negative_kind = _negative_kind(result)

            if pred == "uncertain":
                uncertain += 1
                if gold == "attack":
                    uncertain_attack += 1
                elif gold == "benign":
                    uncertain_benign += 1

            if gold == "attack":
                if pred == "attack":
                    correct += 1
                else:
                    fn += 1
            elif gold == "benign":
                if negative_kind == "benign":
                    negative_benign += 1
                elif negative_kind == "other_attack":
                    negative_other_attack += 1
                else:
                    negative_unknown_other += 1
                if pred == "benign":
                    correct += 1
                elif pred == "attack":
                    fp += 1
                    if negative_kind == "benign":
                        fp_negative_benign += 1
                    elif negative_kind == "other_attack":
                        fp_negative_other_attack += 1
                    else:
                        fp_negative_unknown_other += 1

        return RegressionSummary(
            total=total,
            correct=correct,
            errors=fp + fn,
            fp=fp,
            fn=fn,
            uncertain=uncertain,
            uncertain_attack=uncertain_attack,
            uncertain_benign=uncertain_benign,
            negative_benign=negative_benign,
            negative_other_attack=negative_other_attack,
            negative_unknown_other=negative_unknown_other,
            fp_negative_benign=fp_negative_benign,
            fp_negative_other_attack=fp_negative_other_attack,
            fp_negative_unknown_other=fp_negative_unknown_other,
        )

    @staticmethod
    def _acceptance_decision(
        old: RegressionSummary,
        new: RegressionSummary,
        max_fp_increase: int = 1,
        max_fn_increase: int = 1,
        max_uncertain_increase: int = 0,
        max_uncertain_benign_increase: int = 0,
        acceptance_mode: str = "rule",
        enforce_uncertain_gate: bool = True,
    ) -> Dict[str, Any]:
        error_delta = new.errors - old.errors
        fp_delta = new.fp - old.fp
        fn_delta = new.fn - old.fn
        uncertain_delta = new.uncertain - old.uncertain
        uncertain_benign_delta = new.uncertain_benign - old.uncertain_benign
        uncertain_gate = {
            "max_uncertain_increase": max_uncertain_increase,
            "max_uncertain_benign_increase": max_uncertain_benign_increase,
            "uncertain_delta": uncertain_delta,
            "uncertain_attack_delta": new.uncertain_attack - old.uncertain_attack,
            "uncertain_benign_delta": uncertain_benign_delta,
            "accept": True,
            "reject_reason": "",
        }

        def _metric_value(value: float | None) -> float:
            return 0.0 if value is None else float(value)

        old_recall = _metric_value(old.attack_recall)
        new_recall = _metric_value(new.attack_recall)
        old_precision = _metric_value(old.attack_precision)
        new_precision = _metric_value(new.attack_precision)
        recall_delta = new_recall - old_recall
        precision_delta = new_precision - old_precision
        metric_gate = {
            "mode": "bounded_error_budget",
            "old": {
                "attack_tp": old.attack_tp,
                "attack_target_total": old.target_total,
                "attack_predicted_total": old.predicted_attack_total,
                "attack_recall": old.attack_recall,
                "attack_precision": old.attack_precision,
            },
            "new": {
                "attack_tp": new.attack_tp,
                "attack_target_total": new.target_total,
                "attack_predicted_total": new.predicted_attack_total,
                "attack_recall": new.attack_recall,
                "attack_precision": new.attack_precision,
            },
            "delta": {
                "attack_recall": recall_delta,
                "attack_precision": precision_delta,
            },
            "requirements": [
                f"fp increase must not exceed {max_fp_increase}",
                f"fn increase must not exceed {max_fn_increase}",
                "total errors must not increase",
                (
                    "configured uncertainty budgets must not be exceeded"
                    if enforce_uncertain_gate
                    else "uncertainty changes are diagnostic only"
                ),
                (
                    "correct count must improve, or errors must decrease within configured uncertainty budgets"
                    if enforce_uncertain_gate
                    else "correct count must improve, or errors must decrease"
                ),
            ],
        }

        def _attach_gates(decision: Dict[str, Any]) -> Dict[str, Any]:
            violations = []
            if enforce_uncertain_gate and uncertain_delta > int(max_uncertain_increase or 0):
                violations.append(
                    f"uncertain increased by {uncertain_delta}, allowed {max_uncertain_increase}"
                )
            if enforce_uncertain_gate and uncertain_benign_delta > int(max_uncertain_benign_increase or 0):
                violations.append(
                    "benign-side uncertain increased by "
                    f"{uncertain_benign_delta}, allowed {max_uncertain_benign_increase}"
                )
            if violations:
                uncertain_gate["accept"] = False
                uncertain_gate["reject_reason"] = "; ".join(violations)
            uncertain_gate["enforced"] = bool(enforce_uncertain_gate)
            uncertain_gate["policy"] = (
                "configured_budget"
                if enforce_uncertain_gate
                else "disabled_for_final_strict_validation"
            )
            decision["uncertain_gate"] = uncertain_gate
            decision["metric_gate"] = metric_gate
            return decision

        budget_violations = []
        if fp_delta > int(max_fp_increase or 0):
            budget_violations.append(
                f"fp increased by {fp_delta}, allowed {max_fp_increase}"
            )
        if fn_delta > int(max_fn_increase or 0):
            budget_violations.append(
                f"fn increased by {fn_delta}, allowed {max_fn_increase}"
            )
        if error_delta > 0:
            budget_violations.append(f"total errors increased by {error_delta}")
        if enforce_uncertain_gate and uncertain_delta > int(max_uncertain_increase or 0):
            budget_violations.append(
                f"uncertain increased by {uncertain_delta}, allowed {max_uncertain_increase}"
            )
        if enforce_uncertain_gate and uncertain_benign_delta > int(max_uncertain_benign_increase or 0):
            budget_violations.append(
                "benign-side uncertain increased by "
                f"{uncertain_benign_delta}, allowed {max_uncertain_benign_increase}"
            )
        if budget_violations:
            metric_gate["accept"] = False
            metric_gate["reject_reason"] = "; ".join(budget_violations)
            return _attach_gates({
                "accept": False,
                "accept_reason": "",
                "reject_reason": metric_gate["reject_reason"],
            })

        correct_improved = new.correct > old.correct
        errors_improved_within_budget = error_delta < 0
        if not correct_improved and not errors_improved_within_budget:
            metric_gate["accept"] = False
            metric_gate["reject_reason"] = (
                "candidate has no observable bounded improvement "
                f"(correct {old.correct}->{new.correct}, errors "
                f"{old.errors}->{new.errors}, uncertain "
                f"{old.uncertain}->{new.uncertain})."
            )
            return _attach_gates({
                "accept": False,
                "accept_reason": "",
                "reject_reason": metric_gate["reject_reason"],
            })

        metric_gate["accept"] = True
        metric_gate["accept_reason"] = (
            "Accepted by bounded regression gate: "
            f"correct {old.correct}->{new.correct}, errors "
            f"{old.errors}->{new.errors}, fp {old.fp}->{new.fp}, "
            f"fn {old.fn}->{new.fn}, uncertain "
            f"{old.uncertain}->{new.uncertain}."
        )
        return _attach_gates({
            "accept": True,
            "accept_reason": metric_gate["accept_reason"],
            "reject_reason": "",
        })

    @staticmethod
    def _per_case_changes(
        old_results: List[Dict[str, Any]],
        new_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        old_by_tx = {str(_tx(result)): result for result in old_results}
        new_by_tx = {str(_tx(result)): result for result in new_results}
        changes: List[Dict[str, Any]] = []
        for tx_hash in sorted(set(old_by_tx) | set(new_by_tx)):
            old = old_by_tx.get(tx_hash, {})
            new = new_by_tx.get(tx_hash, {})
            gold = get_ground_truth(new) or get_ground_truth(old)
            old_pred = get_finding(old).get("verdict")
            new_pred = get_finding(new).get("verdict")
            old_conf = get_finding(old).get("confidence")
            new_conf = get_finding(new).get("confidence")
            old_error = _case_error_type(gold, old_pred)
            new_error = _case_error_type(gold, new_pred)
            change_type = _change_type(old_error, new_error, old_pred, new_pred, old_conf, new_conf)
            changes.append({
                "tx_hash": tx_hash,
                "ground_truth": gold,
                "old_pred": old_pred,
                "new_pred": new_pred,
                "old_confidence": old_conf,
                "new_confidence": new_conf,
                "old_error_type": old_error,
                "new_error_type": new_error,
                "change_type": change_type,
            })
        return changes


def _tx(result: Dict[str, Any]) -> str:
    from evotx.utils.result_utils import get_tx_hash

    return get_tx_hash(result)


def _negative_kind(result: Dict[str, Any]) -> str:
    if "negative_kind" in result:
        return str(result.get("negative_kind") or "unknown_other")
    evaluation = result.get("evaluation", {}) if isinstance(result.get("evaluation"), dict) else {}
    metadata = evaluation.get("case_metadata", {}) if isinstance(evaluation.get("case_metadata"), dict) else {}
    kind = str(metadata.get("negative_kind") or "").strip()
    if kind:
        return kind
    raw = str(evaluation.get("raw_ground_truth") or result.get("raw_ground_truth") or "").strip().lower()
    if raw in {"benign", "normal", "safe", "legitimate"}:
        return "benign"
    return "unknown_other"


def _case_error_type(gold: Any, pred: Any) -> str | None:
    if gold == "attack":
        return None if pred == "attack" else "fn"
    if gold == "benign":
        if pred == "attack":
            return "fp"
        if pred == "uncertain":
            return "uncertain_benign"
        return None
    return None


def _change_type(
    old_error: str | None,
    new_error: str | None,
    old_pred: Any,
    new_pred: Any,
    old_conf: Any,
    new_conf: Any,
) -> str:
    if old_error == "fn" and new_error is None:
        return "fixed_fn"
    if old_error == "fp" and new_error is None:
        return "fixed_fp"
    if old_error == "uncertain_benign" and new_error is None:
        return "fixed_uncertain_benign"
    if old_error is None and new_error == "fn":
        return "new_fn"
    if old_error is None and new_error == "fp":
        return "new_fp"
    if old_error is None and new_error == "uncertain_benign":
        return "new_uncertain_benign"
    if old_error == "uncertain_benign" and new_error == "fp":
        return "uncertain_to_fp"
    if old_error == "fp" and new_error == "uncertain_benign":
        return "fp_to_uncertain"
    if old_error == "uncertain_benign" and new_error == "uncertain_benign":
        return "unchanged_uncertain"
    if old_error is not None and new_error is not None:
        return "unchanged_error"
    if old_pred == "uncertain" and new_pred != "uncertain":
        return "uncertain_to_correct"
    if old_pred == new_pred and old_conf != new_conf:
        return "confidence_change"
    return "unchanged_correct"


def _count_change_types(per_case_changes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "fixed_fp": 0,
        "fixed_fn": 0,
        "new_fp": 0,
        "new_fn": 0,
        "new_uncertain_benign": 0,
        "fixed_uncertain_benign": 0,
        "unchanged_uncertain": 0,
        "uncertain_to_fp": 0,
        "fp_to_uncertain": 0,
        "uncertain_to_correct": 0,
        "unchanged_error": 0,
        "unchanged_correct": 0,
        "confidence_change": 0,
    }
    for item in per_case_changes:
        change_type = str(item.get("change_type") or "")
        if change_type in counts:
            counts[change_type] += 1
    return counts


def new_fp_txs_from_comparison(comparison: Dict[str, Any]) -> set[str]:
    return {
        str(item.get("tx_hash") or "").lower()
        for item in list((comparison or {}).get("per_case_changes", []) or [])
        if item.get("change_type") == "new_fp" and item.get("tx_hash")
    }


def new_uncertain_txs_from_comparison(comparison: Dict[str, Any]) -> set[str]:
    return {
        str(item.get("tx_hash") or "").lower()
        for item in list((comparison or {}).get("per_case_changes", []) or [])
        if item.get("change_type") in {"new_uncertain_benign", "fp_to_uncertain"}
        and item.get("tx_hash")
    }


def candidate_repair_diagnostic(
    candidate_evaluation: Dict[str, Any],
    policy: CandidateRepairPolicy,
    allow_guard_repair: bool = False,
) -> Dict[str, Any]:
    policy = policy.normalized()
    comparison = dict(candidate_evaluation.get("comparison") or {})
    guard_gate = dict(comparison.get("guard_gate") or {})
    delta = dict(comparison.get("delta") or {})
    old = dict(comparison.get("old") or {})
    new = dict(comparison.get("new") or {})
    fp_delta = int(delta.get("fp", int(new.get("fp", 0)) - int(old.get("fp", 0))) or 0)
    fn_delta = int(delta.get("fn", int(new.get("fn", 0)) - int(old.get("fn", 0))) or 0)
    error_delta = int(
        delta.get("errors", int(new.get("errors", 0)) - int(old.get("errors", 0))) or 0
    )
    new_fp_txs = sorted(new_fp_txs_from_comparison(comparison))

    diagnostic = {
        "name": candidate_evaluation.get("name", ""),
        "update_kind": candidate_evaluation.get("update_kind", ""),
        "accepted_by_regression": bool(comparison.get("accept")),
        "eligible": False,
        "reason": "",
        "delta": {
            "errors": error_delta,
            "fp": fp_delta,
            "fn": fn_delta,
            "uncertain": int(delta.get("uncertain", 0) or 0),
        },
        "new_fp_txs": new_fp_txs,
        "policy": policy.to_dict(),
    }

    if guard_gate.get("enabled") and not bool(guard_gate.get("accept", True)):
        diagnostic["guard_gate"] = guard_gate
        diagnostic["guard_repair_enabled"] = bool(allow_guard_repair)
        if not allow_guard_repair:
            diagnostic["reason"] = "candidate_rejected_by_hard_negative_guard"
            diagnostic["guard_gate_reject_reason"] = guard_gate.get("reject_reason", "")
            diagnostic["guard_new_fp_txs"] = sorted(guard_gate.get("guard_new_fp_txs") or [])
            return diagnostic

    if not policy.enabled:
        diagnostic["reason"] = "candidate_repair_disabled"
        return diagnostic
    if comparison.get("accept"):
        diagnostic["reason"] = "already_accepted_no_repair_needed"
        return diagnostic
    if fp_delta <= 0:
        diagnostic["reason"] = "no_new_fp_cases_for_precision_boundary_repair"
        return diagnostic
    if fp_delta > policy.temporary_fp_increase:
        diagnostic["reason"] = "fp_delta_exceeds_temporary_repair_threshold"
        return diagnostic
    if -fn_delta < policy.repair_min_fn_decrease and error_delta >= 0:
        diagnostic["reason"] = "insufficient_fn_or_error_improvement_for_repair"
        return diagnostic
    if not new_fp_txs:
        diagnostic["reason"] = "no_new_fp_cases_for_repair"
        return diagnostic
    if len(new_fp_txs) > policy.temporary_fp_increase:
        diagnostic["reason"] = "new_fp_count_exceeds_temporary_repair_threshold"
        return diagnostic
    diagnostic["eligible"] = True
    diagnostic["reason"] = "eligible_new_fp_boundary_repair"
    return diagnostic


def candidate_repair_eligible(
    candidate_evaluation: Dict[str, Any],
    policy: CandidateRepairPolicy,
) -> bool:
    return bool(candidate_repair_diagnostic(candidate_evaluation, policy)["eligible"])
