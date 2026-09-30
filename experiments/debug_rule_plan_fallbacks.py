from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evotx.core.plan import (
    compile_rule_to_baseline_plan,
    condition_view_budget,
    source_dependency_level,
)
from evotx.core.rule import make_cold_start_rule, make_rule, rule_complexity
from evotx.planner.plan_generator import PlanGenerator
from evotx.utils.json_utils import stable_json_dumps


LABELS = [
    "access_control",
    "price_manipulation",
    "market_manipulation",
    "reentrancy",
    "insufficient_validation",
    "protocol_accounting_exploitation",
    "token_semantic_exploitation",
    "flashloans",
]

ALIASES = {
    "Access Control": "access_control",
    "Price Manipulation": "price_manipulation",
    "Market manipulation": "market_manipulation",
    "Protocol accounting exploitation": "protocol_accounting_exploitation",
    "Token semantic exploitation": "token_semantic_exploitation",
    "flashloan": "flashloans",
    "flash loans": "flashloans",
}

FORBIDDEN_PHRASES = [
    "better explained as",
    "more consistent with",
    "could be interpreted as",
    "is likely",
    "seems to be",
]


def _rule_text(rule) -> str:
    parts = [rule.description, rule.decision_policy]
    parts.extend(condition.description for condition in rule.conditions)
    parts.extend(condition.description for condition in rule.exclusion_conditions)
    return "\n".join(parts).lower()


def _referenced_condition_ids(text: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b[CE]\d+\b", text or "")}


def _assert_plan_budgets(plan, attack_label: str) -> None:
    for step in plan.judge_steps:
        budget = condition_view_budget(step, attack_label=attack_label)
        assert len(step.default_evidence_refs) <= budget["max_default_views"], step.to_dict()
        assert len(step.allowed_followup_views) <= budget["max_followup_views"], step.to_dict()


class _FakePlannerLLM:
    def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "plan_id": "fake_plan",
                "judge_steps": [
                    {
                        "id": "c1",
                        "condition_id": "c1",
                        "question": "Does condition one hold?",
                        "default_evidence_refs": [
                            "operation_summary_view",
                            "evidence_adequacy_view",
                            "trace_outline_view",
                            "critical_call_view",
                        ],
                        "evidence_refs": [
                            "operation_summary_view",
                            "evidence_adequacy_view",
                            "trace_outline_view",
                            "critical_call_view",
                        ],
                        "allowed_followup_views": ["trace_view", "state_change_view"],
                        "allowed_tools": ["read_packet_view"],
                        "max_followups": 1,
                        "expected_answer": True,
                    },
                    {
                        "id": "C2",
                        "condition_id": "C2",
                        "question": "Does condition two hold?",
                        "default_evidence_refs": [
                            "operation_summary_view",
                            "evidence_adequacy_view",
                        ],
                        "evidence_refs": [
                            "operation_summary_view",
                            "evidence_adequacy_view",
                        ],
                        "allowed_followup_views": [],
                        "allowed_tools": ["read_packet_view"],
                        "max_followups": 1,
                        "expected_answer": True,
                    },
                    {
                        "id": "e1",
                        "condition_id": "e1",
                        "question": "Does the exclusion hold?",
                        "default_evidence_refs": [
                            "operation_summary_view",
                            "evidence_adequacy_view",
                        ],
                        "evidence_refs": [
                            "operation_summary_view",
                            "evidence_adequacy_view",
                        ],
                        "allowed_followup_views": [],
                        "allowed_tools": ["read_packet_view"],
                        "max_followups": 1,
                        "expected_answer": True,
                    },
                    {
                        "id": "C99",
                        "condition_id": "C99",
                        "question": "Extra split condition that should be dropped.",
                        "default_evidence_refs": ["trace_view"],
                        "evidence_refs": ["trace_view"],
                        "allowed_followup_views": [],
                        "allowed_tools": ["read_packet_view"],
                        "max_followups": 1,
                        "expected_answer": True,
                    },
                ],
                "emit_logic": "C1 or C2",
                "metadata": {"generator": "fake"},
            }
        )


def run_checks() -> dict:
    report: dict = {"labels": {}, "checks": []}
    for label in LABELS:
        rule = make_cold_start_rule(attack_label=label)
        plan = compile_rule_to_baseline_plan(rule)
        text = _rule_text(rule)
        forbidden = [phrase for phrase in FORBIDDEN_PHRASES if phrase in text]
        positive_text = " ".join(c.description.lower() for c in rule.conditions)
        complexity = rule_complexity(rule)
        condition_ids = {condition.id for condition in rule.conditions}
        exclusion_ids = {condition.id for condition in rule.exclusion_conditions}
        referenced = _referenced_condition_ids(rule.decision_policy)
        label_report = {
            "rule_id": rule.rule_id,
            "attack_label": (rule.metadata or {}).get("attack_label"),
            "source": (rule.metadata or {}).get("source"),
            "positive_conditions": len(rule.conditions),
            "exclusion_conditions": len(rule.exclusion_conditions),
            "total_conditions": complexity["total_conditions"],
            "forbidden_phrases": forbidden,
            "judge_steps": len(plan.judge_steps),
            "complexity": complexity,
        }
        report["labels"][label] = label_report

        assert label_report["source"] == "cold_start_fallback", label_report
        assert label_report["attack_label"] == label, label_report
        assert len(rule.conditions) == 3, label_report
        assert 1 <= len(rule.exclusion_conditions) <= 2, label_report
        assert complexity["total_conditions"] <= 5, label_report
        assert len(plan.judge_steps) == complexity["total_conditions"], label_report
        assert referenced <= (condition_ids | exclusion_ids), (label, referenced, condition_ids, exclusion_ids)
        assert not forbidden, label_report
        _assert_plan_budgets(plan, label)
        if label == "price_manipulation":
            assert "flashloan" not in positive_text, positive_text
            assert "flash loan" not in positive_text, positive_text
            assert "atomic capital" not in positive_text, positive_text

    for raw_label, expected in ALIASES.items():
        rule = make_cold_start_rule(attack_label=raw_label)
        actual = (rule.metadata or {}).get("attack_label")
        report.setdefault("aliases", {})[raw_label] = actual
        assert actual == expected, (raw_label, actual, expected)

    assert source_dependency_level("msg.sender onlyOwner role check", "access_control") == "required"
    assert source_dependency_level("validation logic and callback sender check", "insufficient_validation") == "required"
    assert source_dependency_level(
        "protocol consumes user supplied data/state",
        "insufficient_validation",
    ) != "required"
    assert source_dependency_level("value extraction and profit outcome", "") == "none"

    required_rule = make_rule(
        name="Required Source Rule",
        description="source required",
        conditions=[
            {
                "id": "C1",
                "description": "The function lacks msg.sender onlyOwner role check on a protected path.",
                "expected_answer": True,
            }
        ],
        metadata={"attack_label": "access_control"},
    )
    required_step = compile_rule_to_baseline_plan(required_rule).judge_steps[0]
    assert required_step.max_followups == 2, required_step.to_dict()
    assert "read_function_chunk" in required_step.allowed_tools, required_step.to_dict()
    assert len(required_step.default_evidence_refs) <= 5, required_step.to_dict()
    assert len(required_step.allowed_followup_views) <= 7, required_step.to_dict()

    helpful_rule = make_rule(
        name="Helpful Source Rule",
        description="source helpful",
        conditions=[
            {
                "id": "C1",
                "description": "A protocol path consumes user supplied data or state before value-sensitive logic.",
                "expected_answer": True,
            }
        ],
        metadata={"attack_label": "insufficient_validation"},
    )
    helpful_step = compile_rule_to_baseline_plan(helpful_rule).judge_steps[0]
    assert helpful_step.max_followups <= 1, helpful_step.to_dict()
    assert "read_function_chunk" not in helpful_step.allowed_tools, helpful_step.to_dict()

    ac_rule = make_cold_start_rule(attack_label="access_control")
    ac_plan = compile_rule_to_baseline_plan(ac_rule)
    identity_step = next(step for step in ac_plan.judge_steps if step.condition_id == "C2")
    assert (
        "address_labels" in identity_step.default_evidence_refs
        or "beneficiary_controller_view" in identity_step.default_evidence_refs
    ), identity_step.to_dict()
    assert "unknown_selector_view" not in identity_step.default_evidence_refs, identity_step.to_dict()

    rt_rule = make_cold_start_rule(attack_label="reentrancy")
    rt_plan = compile_rule_to_baseline_plan(rt_rule)
    state_step = next(
        step for step in rt_plan.judge_steps
        if "reentrancy_state_order_summary_view" in step.default_evidence_refs
    )
    assert "reentrancy_state_order_view" in state_step.allowed_followup_views, state_step.to_dict()
    assert len(state_step.default_evidence_refs) <= 5, state_step.to_dict()
    assert len(state_step.allowed_followup_views) <= 6, state_step.to_dict()

    emit_rule = make_rule(
        name="Emit Logic Rule",
        description="emit logic",
        conditions=[
            {"id": "C1", "description": "First condition.", "expected_answer": True},
            {"id": "C2", "description": "Second condition.", "expected_answer": True},
        ],
        exclusion_conditions=[
            {"id": "E1", "description": "Benign exclusion.", "expected_answer": True},
        ],
        metadata={"attack_label": "generic"},
    )
    generated = PlanGenerator(llm=_FakePlannerLLM()).generate(emit_rule)
    assert len(generated.judge_steps) == 3, [step.to_dict() for step in generated.judge_steps]
    assert [step.id for step in generated.judge_steps] == ["C1", "C2", "E1"], generated.to_dict()
    assert generated.emit_logic == "(C1 and C2) and not (E1)", generated.emit_logic

    report["source_dependency_samples"] = {
        "required_access_control": required_step.to_dict(),
        "helpful_insufficient_validation": helpful_step.to_dict(),
        "access_control_identity": identity_step.to_dict(),
        "reentrancy_state_order": state_step.to_dict(),
        "planner_emit_logic": generated.emit_logic,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate cold-start fallback rules and packet plan budgets."
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args()

    report = run_checks()
    if args.json:
        print(stable_json_dumps(report))
        return
    print("All fallback/plan budget checks passed.")
    print(stable_json_dumps(report["source_dependency_samples"]))


if __name__ == "__main__":
    main()
