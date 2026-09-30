from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.core.plan import compile_rule_to_baseline_plan
from evotx.core.rule import make_rule, rule_complexity
from evotx.core.schemas import RuleCondition
from evotx.evolution.plan_updater import PlanUpdater
from evotx.evolution.updater import RuleUpdater
from evotx.runtime.judge_reuse import judge_step_fingerprint
from evotx.runtime.packet_runtime import _early_stop_decision
from evotx.utils.fingerprint_utils import semantic_plan_fingerprint


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, _prompt: str) -> str:
        if not self.responses:
            raise AssertionError("FakeLLM exhausted")
        return self.responses.pop(0)


def _base_rule():
    return make_rule(
        name="Debug rule",
        description="debug",
        conditions=[
            RuleCondition(id="C1", description="Root cause evidence is present."),
            RuleCondition(id="C2", description="Trigger or consumed state is present."),
            RuleCondition(id="C3", description="Outcome harm is present."),
        ],
        exclusion_conditions=[
            RuleCondition(id="E1", description="Normal authorized behavior explains it."),
            RuleCondition(id="E2", description="Independent non-target mechanism explains it."),
        ],
        decision_policy="C1 and C2 and C3 and not (E1 or E2)",
        metadata={"attack_label": "debug"},
    )


def _rule_review(condition_id="C2"):
    return {
        "actionable_rule_reviews": [
            {
                "update_target": "rule",
                "should_update_rule": True,
                "rule_patch_suggestion": {
                    "action": "rewrite_condition",
                    "condition_id": condition_id,
                    "proposed_change": "tighten condition",
                },
                "root_causes": [
                    {
                        "category": "bad_semantic_condition",
                        "affected_conditions": [condition_id],
                    }
                ],
            }
        ],
        "non_rule_reviews": [],
    }


def test_rule_budget_rejects_growth():
    rule = _base_rule()
    over_budget_payload = """
{
  "description": "too long",
  "conditions": [
    {"id": "C1", "description": "c1", "expected_answer": true},
    {"id": "C2", "description": "c2", "expected_answer": true},
    {"id": "C3", "description": "c3", "expected_answer": true},
    {"id": "C4", "description": "support symptom", "expected_answer": true}
  ],
  "exclusion_conditions": [
    {"id": "E1", "description": "e1", "expected_answer": true},
    {"id": "E2", "description": "e2", "expected_answer": true},
    {"id": "E3", "description": "extra exclusion", "expected_answer": true}
  ],
  "decision_policy": "C1 and C2 and C3 and C4 and not (E1 or E2 or E3)",
  "update_note": "over budget"
}
"""
    updater = RuleUpdater(llm=FakeLLM([over_budget_payload, over_budget_payload]))
    growth_review = {
        "actionable_rule_reviews": [
            {
                "update_target": "rule",
                "should_update_rule": True,
                "rule_patch_suggestion": {
                    "action": "add_condition",
                    "condition_id": "C4",
                    "proposed_change": "add support condition",
                },
                "root_causes": [{"category": "missing_condition"}],
            },
            {
                "update_target": "rule",
                "should_update_rule": True,
                "rule_patch_suggestion": {
                    "action": "add_exclusion",
                    "condition_id": "E3",
                    "proposed_change": "add boundary exclusion",
                },
                "root_causes": [{"category": "missing_exclusion"}],
            },
        ],
        "non_rule_reviews": [],
    }
    updated = updater.update_rule_from_bundle(rule, growth_review, allow_noop=True)
    assert updated.version == rule.version
    assert updated.metadata.get("last_rule_budget_rejected")


def test_rule_budget_allows_scoped_rewrite():
    rule = _base_rule()
    rewrite_payload = """
{
  "description": "debug",
  "conditions": [
    {"id": "C2", "description": "Trigger evidence is more target specific.", "expected_answer": true}
  ],
  "exclusion_conditions": [],
  "decision_policy": "C1 and C2 and C3 and not (E1 or E2)",
  "update_note": "rewrite C2"
}
"""
    updater = RuleUpdater(llm=FakeLLM([rewrite_payload]))
    updated = updater.update_rule_from_bundle(rule, _rule_review("C2"), allow_noop=True)
    assert updated.version == rule.version + 1
    assert rule_complexity(updated)["total_conditions"] == 5
    assert updated.conditions[1].description.startswith("Trigger evidence")
    assert updated.conditions[0].description == rule.conditions[0].description


def test_early_stop_policy():
    plan = compile_rule_to_baseline_plan(_base_rule())
    assert _early_stop_decision(
        plan=plan,
        judge_results=[
            {"id": "C1", "condition_id": "C1", "answer": False, "confidence": "medium"},
            {"id": "C2", "condition_id": "C2", "answer": False, "confidence": "high"},
        ],
        judge_values={"C1": False, "C2": False},
        enabled=True,
        policy="conservative_negative",
    )["triggered"]
    assert not _early_stop_decision(
        plan=plan,
        judge_results=[
            {"id": "C1", "condition_id": "C1", "answer": False, "confidence": "medium"}
        ],
        judge_values={"C1": False},
        enabled=True,
        policy="conservative_negative",
    ).get("triggered")
    assert not _early_stop_decision(
        plan=plan,
        judge_results=[
            {"id": "C1", "condition_id": "C1", "answer": "uncertain", "confidence": "low"},
            {"id": "C2", "condition_id": "C2", "answer": False, "confidence": "medium"},
        ],
        judge_values={"C1": False, "C2": False},
        enabled=True,
        policy="conservative_negative",
    ).get("triggered")
    assert not _early_stop_decision(
        plan=plan,
        judge_results=[
            {"id": "C1", "condition_id": "C1", "answer": False, "confidence": "medium"},
            {"id": "C2", "condition_id": "C2", "answer": False, "confidence": "high"},
        ],
        judge_values={"C1": False, "C2": False},
        enabled=False,
        policy="conservative_negative",
    ).get("triggered")


def test_partial_reuse_fingerprint():
    plan = compile_rule_to_baseline_plan(_base_rule())
    selected_hash = "views-hash"
    packet_identity = "packet-identity"
    base = [
        judge_step_fingerprint(
            step,
            tx_hash="0xabc",
            packet_identity_fingerprint=packet_identity,
            selected_view_names=list(step.default_evidence_refs),
            selected_views_hash=selected_hash,
        )
        for step in plan.judge_steps[:3]
    ]
    changed_c3 = plan.judge_steps[2].to_dict()
    changed_c3["question"] = changed_c3["question"] + " tighter?"
    candidate = [
        judge_step_fingerprint(
            plan.judge_steps[0],
            tx_hash="0xabc",
            packet_identity_fingerprint=packet_identity,
            selected_view_names=list(plan.judge_steps[0].default_evidence_refs),
            selected_views_hash=selected_hash,
        ),
        judge_step_fingerprint(
            plan.judge_steps[1],
            tx_hash="0xabc",
            packet_identity_fingerprint=packet_identity,
            selected_view_names=list(plan.judge_steps[1].default_evidence_refs),
            selected_views_hash=selected_hash,
        ),
        judge_step_fingerprint(
            changed_c3,
            tx_hash="0xabc",
            packet_identity_fingerprint=packet_identity,
            selected_view_names=list(plan.judge_steps[2].default_evidence_refs),
            selected_views_hash=selected_hash,
        ),
    ]
    assert base[0] == candidate[0]
    assert base[1] == candidate[1]
    assert base[2] != candidate[2]
    assert base[0] != judge_step_fingerprint(
        plan.judge_steps[0],
        tx_hash="0xdef",
        packet_identity_fingerprint=packet_identity,
        selected_view_names=list(plan.judge_steps[0].default_evidence_refs),
        selected_views_hash=selected_hash,
    )


def test_plan_updater_unscoped_noop_and_global_allow():
    rule = _base_rule()
    plan = compile_rule_to_baseline_plan(rule)
    payload = """
{
  "judge_steps": [
    {"id": "C1", "condition_id": "C1", "question": "Changed globally?", "expected_answer": true}
  ],
  "emit_logic": "C1",
  "metadata": {"source": "plan_update"}
}
"""
    unscoped_bundle = {
        "non_rule_reviews": [
            {
                "update_target": "plan",
                "should_update_plan_strategy": True,
                "plan_patch_suggestion": {"action": "change_default_views"},
            }
        ]
    }
    updater = PlanUpdater(llm=FakeLLM([payload]))
    unchanged = updater.update_plan_from_bundle(rule, unscoped_bundle, base_plan=plan)
    assert semantic_plan_fingerprint(unchanged) == semantic_plan_fingerprint(plan)
    assert unchanged.metadata.get("plan_update_noop_reason") == "unscoped_plan_review"

    global_bundle = {
        "update_constraints": ["allow_global_plan_update=True"],
        "non_rule_reviews": [
            {
                "update_target": "plan",
                "should_update_plan_strategy": True,
                "plan_patch_suggestion": {"action": "global_followup_policy"},
            }
        ],
    }
    updated = PlanUpdater(llm=FakeLLM([payload])).update_plan_from_bundle(
        rule,
        global_bundle,
        base_plan=plan,
    )
    assert semantic_plan_fingerprint(updated) != semantic_plan_fingerprint(plan)
    assert updated.judge_steps[0].question == "Changed globally?"
    assert updated.emit_logic == plan.emit_logic


def main() -> None:
    test_rule_budget_rejects_growth()
    test_rule_budget_allows_scoped_rewrite()
    test_early_stop_policy()
    test_partial_reuse_fingerprint()
    test_plan_updater_unscoped_noop_and_global_allow()
    print({"debug_evolution_speed": "ok"})


if __name__ == "__main__":
    main()
