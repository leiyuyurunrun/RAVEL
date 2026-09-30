from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.evolution.regression import RegressionEvaluator
from experiments.train_fewshot import (
    load_hard_negative_guard_cases,
    select_candidate_evaluation,
)


def _result(tx_hash: str, gold: str, pred: str, *, negative_kind: str = "") -> dict:
    return {
        "transaction": {"tx_hash": tx_hash, "chain": "eth"},
        "inference": {
            "finding": {
                "verdict": pred,
                "confidence": "medium",
            }
        },
        "evaluation": {
            "ground_truth": gold,
            "case_metadata": {"negative_kind": negative_kind} if negative_kind else {},
        },
    }


def main() -> None:
    evaluator = RegressionEvaluator()

    old_train = [_result("0xpos", "attack", "benign")]
    new_train = [_result("0xpos", "attack", "attack")]
    old_guard = [_result("0xguard", "benign", "benign", negative_kind="other_attack")]
    new_guard_regressed = [
        _result("0xguard", "benign", "attack", negative_kind="other_attack")
    ]
    comparison = evaluator.compare_with_guard(
        old_train,
        new_train,
        old_guard_results=old_guard,
        new_guard_results=new_guard_regressed,
        max_guard_fp_increase=0,
        max_guard_error_increase=0,
        max_guard_uncertain_increase=0,
    )
    assert comparison["train_comparison"]["accept"] is True
    assert comparison["guard_gate"]["accept"] is False
    assert comparison["accept"] is False

    new_guard_ok = [_result("0xguard", "benign", "benign", negative_kind="other_attack")]
    comparison_ok = evaluator.compare_with_guard(
        old_train,
        new_train,
        old_guard_results=old_guard,
        new_guard_results=new_guard_ok,
    )
    assert comparison_ok["accept"] is True

    selected = select_candidate_evaluation(
        [
            {
                "name": "guard_worse",
                "update_kind": "rule",
                "summary": {"errors": 1, "fn": 0, "fp": 1, "uncertain": 0},
                "guard_summary": {"errors": 1, "fp": 1, "uncertain": 0},
                "comparison": {"accept": True},
            },
            {
                "name": "guard_better",
                "update_kind": "rule",
                "summary": {"errors": 1, "fn": 0, "fp": 1, "uncertain": 0},
                "guard_summary": {"errors": 0, "fp": 0, "uncertain": 0},
                "comparison": {"accept": True},
            },
        ],
        accepted_only=True,
    )
    assert selected and selected["name"] == "guard_better"

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "guard.csv"
        csv_path.write_text(
            "HackId,txHash,Chain,Type,Cause\n"
            "1,0xabc,eth,Reentrancy,non-target hard negative\n"
            "2,0xabc,eth,Reentrancy,duplicate should be skipped\n",
            encoding="utf-8",
        )
        cases = load_hard_negative_guard_cases(
            SimpleNamespace(hard_neg_csv=str(csv_path)),
            "access_control",
        )
        assert len(cases) == 1
        assert cases[0]["ground_truth"] == "benign"
        assert cases[0]["metadata"]["negative_kind"] == "other_attack"
        assert cases[0]["metadata"]["guard_role"] == "hard_negative_guard"
        assert cases[0]["metadata"]["used_for_review"] is False

    print(json.dumps({
        "guard_veto_delta": comparison["guard_gate"]["delta"],
        "guard_veto_accept": comparison["accept"],
        "guard_ok_accept": comparison_ok["accept"],
        "tie_break_selected": selected["name"],
        "loader_case_count": len(cases),
    }, indent=2))


if __name__ == "__main__":
    main()
