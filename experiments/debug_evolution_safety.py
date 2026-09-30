from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evotx.core.schemas import EvidencePlan, EvolvingRule, JudgeStep, RuleCondition
from evotx.evolution.plan_updater import PlanUpdater
from evotx.evolution.regression import (
    CandidateRepairPolicy,
    RegressionEvaluator,
    candidate_repair_diagnostic,
)
from evotx.evolution.reviewer import RuleReviewer
from evotx.evolution.updater import (
    RuleUpdater,
    _apply_minimal_update_scope,
    _infer_rule_update_scope,
)
from evotx.utils.fingerprint_utils import semantic_plan_fingerprint


def _result(tx_hash: str, gold: str, pred: str) -> dict:
    return {
        "transaction": {"tx_hash": tx_hash, "chain": "eth"},
        "evaluation": {"ground_truth": gold},
        "inference": {"finding": {"verdict": pred, "confidence": "medium"}},
    }


def _rule() -> EvolvingRule:
    return EvolvingRule(
        rule_id="safety_rule",
        version=1,
        name="safety",
        description="Safety debug rule",
        conditions=[
            RuleCondition(id="C1", description="Does sensitive logic execute?"),
            RuleCondition(id="C2", description="Does the causal mechanism hold?"),
        ],
        exclusion_conditions=[
            RuleCondition(id="E1", description="Does authorized behavior explain it?")
        ],
        decision_policy="C1 and C2 and not E1",
        metadata={"attack_label": "access_control"},
    )


def _plan(rule: EvolvingRule) -> EvidencePlan:
    return EvidencePlan(
        plan_id="debug_plan",
        rule_id=rule.rule_id,
        rule_version=rule.version,
        focus_steps=[],
        judge_steps=[
            JudgeStep(
                id="C1",
                condition_id="C1",
                question="Does sensitive logic execute?",
                default_evidence_refs=["operation_summary_view", "evidence_adequacy_view"],
                evidence_refs=["operation_summary_view", "evidence_adequacy_view"],
                allowed_followup_views=["trace_view"],
                allowed_tools=["read_packet_view"],
                max_followups=1,
            ),
            JudgeStep(
                id="C2",
                condition_id="C2",
                question="Does the causal mechanism hold?",
                default_evidence_refs=["operation_summary_view", "critical_call_view"],
                evidence_refs=["operation_summary_view", "critical_call_view"],
                allowed_followup_views=["trace_view"],
                allowed_tools=["read_packet_view"],
                max_followups=1,
            ),
        ],
        emit_logic="C1 and C2",
        metadata={"plan_version": 1},
    )


class FakePlanLLM:
    def complete(self, _prompt: str) -> str:
        rule = _rule()
        plan = _plan(rule).to_dict()
        plan["judge_steps"][0]["question"] = "BAD unrelated C1 rewrite"
        plan["judge_steps"][1]["question"] = "Does C2 use the corrected evidence strategy?"
        plan["judge_steps"][1]["expected_answer"] = False
        plan["emit_logic"] = "C2"
        return json.dumps(plan)


class FakeCompressionLLM:
    def complete(self, _prompt: str) -> str:
        rule = _rule().to_dict()
        rule["conditions"][0]["id"] = "X1"
        return json.dumps(rule)


def main() -> None:
    evaluator = RegressionEvaluator()

    uncertain_change = evaluator.compare(
        [_result("0xb", "benign", "benign")],
        [_result("0xb", "benign", "uncertain")],
    )
    assert uncertain_change["per_case_changes"][0]["change_type"] == "new_uncertain_benign"

    gated = evaluator.compare(
        [_result("0xa", "attack", "benign"), _result("0xb", "benign", "benign")],
        [_result("0xa", "attack", "attack"), _result("0xb", "benign", "uncertain")],
    )
    assert gated["delta"]["errors"] == -1
    assert gated["accept"] is True
    assert gated["uncertain_gate"]["accept"] is False
    assert gated["uncertain_gate"]["enforced"] is False

    allowed = evaluator.compare(
        [_result("0xa", "attack", "benign"), _result("0xb", "benign", "benign")],
        [_result("0xa", "attack", "attack"), _result("0xb", "benign", "uncertain")],
        max_uncertain_benign_increase=1,
        max_uncertain_increase=1,
    )
    assert allowed["accept"] is True

    guard_reject = evaluator.compare_with_guard(
        [_result("0xa", "attack", "benign")],
        [_result("0xa", "attack", "attack")],
        old_guard_results=[_result("0xg", "benign", "benign")],
        new_guard_results=[_result("0xg", "benign", "attack")],
    )
    assert guard_reject["accept"] is False
    diagnostic = candidate_repair_diagnostic(
        {"name": "guard_candidate", "comparison": guard_reject},
        CandidateRepairPolicy(),
    )
    assert diagnostic["reason"] == "candidate_rejected_by_hard_negative_guard"
    assert diagnostic["eligible"] is False

    bad_review = RuleReviewer._normalize_review(
        {
            "update_target": "rule",
            "root_causes": [{"category": "missing_evidence"}],
            "rule_patch_suggestion": {"action": "none"},
        },
        {"ground_truth": "attack", "predicted_verdict": "benign", "condition_table": []},
    )
    assert bad_review["should_update_rule"] is False

    runtime_review = RuleReviewer._normalize_review(
        {"update_target": "rule", "root_causes": [], "rule_patch_suggestion": {"action": "none"}},
        {
            "ground_truth": "attack",
            "predicted_verdict": "benign",
            "condition_table": [
                {"condition_id": "C1", "answer": True, "is_exclusion": False},
                {"condition_id": "E1", "answer": False, "is_exclusion": True},
            ],
        },
    )
    assert runtime_review["update_target"] == "runtime"

    exclusion_review = {
        "root_causes": [{"category": "exclusion_too_broad", "affected_conditions": ["E1"]}],
        "rule_patch_suggestion": {"action": "rewrite_condition", "condition_id": "e1"},
        "condition_diagnosis": [],
    }
    scope = _infer_rule_update_scope([exclusion_review])
    assert scope["target_exclusions"] == ["E1"]
    guarded = _apply_minimal_update_scope(
        _rule(),
        {
            "conditions": [{"id": "c1", "description": "BAD C1", "expected_answer": True}],
            "exclusion_conditions": [
                {"id": "e1", "description": "Rewritten E1", "expected_answer": True}
            ],
        },
        scope,
    )
    assert guarded["conditions"][0]["description"] != "BAD C1"
    assert guarded["exclusion_conditions"][0]["description"] == "Rewritten E1"

    rule = _rule()
    plan = _plan(rule)
    metadata_only = EvidencePlan.from_dict({**plan.to_dict(), "metadata": {"plan_version": 99}})
    assert semantic_plan_fingerprint(plan) == semantic_plan_fingerprint(metadata_only)
    changed = EvidencePlan.from_dict(plan.to_dict())
    changed.judge_steps[0].default_evidence_refs.append("critical_call_view")
    assert semantic_plan_fingerprint(plan) != semantic_plan_fingerprint(changed)

    updater = PlanUpdater(llm=FakePlanLLM())
    review_bundle = {
        "non_rule_reviews": [
            {
                "update_target": "plan",
                "should_update_plan_strategy": True,
                "plan_patch_suggestion": {
                    "action": "change_judge_question",
                    "condition_id": "C2",
                },
                "root_causes": [
                    {"category": "bad_judge_question", "affected_conditions": ["C2"]}
                ],
            }
        ]
    }
    updated_plan = updater.update_plan_from_bundle(rule, review_bundle, base_plan=plan)
    assert updated_plan.judge_steps[0].question == plan.judge_steps[0].question
    assert updated_plan.judge_steps[1].question != plan.judge_steps[1].question
    assert updated_plan.judge_steps[1].expected_answer is True
    assert updated_plan.emit_logic == plan.emit_logic

    compressed = RuleUpdater(llm=FakeCompressionLLM()).compress_rule(_rule())
    assert [condition.id for condition in compressed.conditions] == ["C1", "C2"]

    print(json.dumps({
        "uncertain_change": uncertain_change["per_case_changes"][0]["change_type"],
        "uncertain_gate_accept": gated["uncertain_gate"]["accept"],
        "guard_repair_reason": diagnostic["reason"],
        "reviewer_bad_rule_target": bad_review["should_update_rule"],
        "runtime_target": runtime_review["update_target"],
        "rule_scope": scope,
        "plan_scope_guard": updated_plan.metadata.get("scope_guard", {}),
        "compression_preserved_ids": [condition.id for condition in compressed.conditions],
    }, indent=2))


if __name__ == "__main__":
    main()
