from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evotx.core.plan import compile_rule_to_baseline_plan
from evotx.core.schemas import EvolvingRule, RuleCondition
from evotx.runtime.judge_reuse import build_judge_reuse_cache
from experiments.train_fewshot import (
    build_candidate_judge_reuse_index,
    build_candidate_specs,
    regenerate_candidate_plan,
)
from types import SimpleNamespace


class FakeRuleUpdater:
    def update_rule_from_bundle(self, rule, review_bundle, allow_noop=True):
        return rule.next_version(
            new_description=rule.description,
            new_conditions=rule.conditions,
            new_exclusions=rule.exclusion_conditions,
            update_note="fake rule update",
        )


class FakePlanUpdater:
    def update_plan_from_bundle(self, rule, review_bundle, base_plan=None, allow_noop=True):
        plan = compile_rule_to_baseline_plan(rule) if base_plan is None else base_plan
        data = plan.to_dict()
        if data.get("judge_steps"):
            data["judge_steps"][0]["question"] = (
                data["judge_steps"][0].get("question", "") + " Use compact locating evidence."
            ).strip()
        data["plan_note"] = f"{data.get('plan_note', '')}\nfake semantic plan update".strip()
        metadata = dict(data.get("metadata") or {})
        metadata["fake_plan_update"] = True
        metadata["plan_version"] = int(metadata.get("plan_version") or 1) + 1
        data["metadata"] = metadata
        from evotx.core.schemas import EvidencePlan

        return EvidencePlan.from_dict(data)


class MetadataOnlyPlanUpdater:
    def update_plan_from_bundle(self, rule, review_bundle, base_plan=None, allow_noop=True):
        plan = compile_rule_to_baseline_plan(rule) if base_plan is None else base_plan
        data = plan.to_dict()
        metadata = dict(data.get("metadata") or {})
        metadata["metadata_only_update"] = True
        metadata["plan_version"] = int(metadata.get("plan_version") or 1) + 1
        data["metadata"] = metadata
        from evotx.core.schemas import EvidencePlan

        return EvidencePlan.from_dict(data)


def _rule() -> EvolvingRule:
    return EvolvingRule(
        rule_id="debug_rule",
        version=1,
        name="debug",
        description="debug rule",
        conditions=[
            RuleCondition(id="C1", description="Does a sensitive operation occur?"),
            RuleCondition(id="C2", description="Does value move?"),
            RuleCondition(id="C3", description="Is the effect causal?"),
        ],
        exclusion_conditions=[
            RuleCondition(id="E1", description="Does authorized behavior explain it?")
        ],
        decision_policy="C1 and C2 and C3 and not E1",
        metadata={"attack_label": "access_control"},
    )


def _bundle(*, rule=False, plan=False) -> dict:
    out = {
        "actionable_rule_reviews": [],
        "non_rule_reviews": [],
    }
    if rule:
        out["actionable_rule_reviews"].append({
            "update_target": "rule",
            "should_update_rule": True,
            "rule_patch_suggestion": {"action": "modify_condition"},
        })
    if plan:
        out["non_rule_reviews"].append({
            "update_target": "plan",
            "should_update_plan_strategy": True,
            "plan_patch_suggestion": {"action": "change_default_views"},
        })
    return out


def _names(specs):
    return [spec["name"] for spec in specs]


def main() -> None:
    rule = _rule()
    plan = compile_rule_to_baseline_plan(rule)
    updater = FakeRuleUpdater()
    plan_updater = FakePlanUpdater()

    plan_only = build_candidate_specs(
        current_rule=rule,
        current_plan=plan,
        review_bundle=_bundle(plan=True),
        update_target="auto",
        updater=updater,
        plan_updater=plan_updater,
    )
    assert _names(plan_only) == ["plan_candidate"], _names(plan_only)

    metadata_only = build_candidate_specs(
        current_rule=rule,
        current_plan=plan,
        review_bundle=_bundle(plan=True),
        update_target="auto",
        updater=updater,
        plan_updater=MetadataOnlyPlanUpdater(),
    )
    assert _names(metadata_only) == [], _names(metadata_only)

    rule_only = build_candidate_specs(
        current_rule=rule,
        current_plan=plan,
        review_bundle=_bundle(rule=True),
        update_target="auto",
        updater=updater,
        plan_updater=plan_updater,
    )
    assert _names(rule_only) == ["rule_candidate"], _names(rule_only)
    assert rule_only[0]["requires_plan_regeneration"] is True
    assert rule_only[0]["plan"] is None

    both = build_candidate_specs(
        current_rule=rule,
        current_plan=plan,
        review_bundle=_bundle(rule=True, plan=True),
        update_target="auto",
        updater=updater,
        plan_updater=plan_updater,
    )
    assert set(_names(both)) == {"plan_candidate", "rule_candidate"}, _names(both)
    assert len(both) == 2, _names(both)
    assert "combined_candidate" not in _names(both)
    assert all(spec.get("update_kind") in {"plan", "rule"} for spec in both)

    regenerated_plan, fallback = regenerate_candidate_plan(
        rule_only[0]["rule"],
        args=SimpleNamespace(planner_model=None, llm_model=None, llm_provider=None),
        planner_guidance={"source": "debug"},
    )
    assert fallback is True
    assert regenerated_plan.metadata["plan_generation_fallback"] is True
    assert regenerated_plan.metadata["candidate_strategy"] == "updated_rule_regenerated_plan"

    reuse_index, reuse_scope = build_candidate_judge_reuse_index(
        [
            {
                "transaction": {"tx_hash": "0x1"},
                "detector_context": {
                    "rule_id": rule.rule_id,
                    "rule_version": rule.version,
                },
                "inference": {"plan": plan.to_dict()},
            }
        ],
        candidate_rule=rule_only[0]["rule"],
        candidate_plan=regenerated_plan,
    )
    assert "0x1" in reuse_index
    assert reuse_scope["pre_filter"] == "tx_hash_only"
    assert reuse_scope["step_level_guard"] == "PacketRuntime.judge_step_fingerprint"

    previous_result = {
        "transaction": {"tx_hash": "0x1"},
        "detector_context": {
            "rule_id": rule.rule_id,
            "rule_version": rule.version,
        },
        "inference": {
            "plan": plan.to_dict(),
            "trace": {
                "judge_calls": [
                    {
                        "judge_id": plan.judge_steps[0].id,
                        "id": plan.judge_steps[0].id,
                        "answer": True,
                        "confidence": "medium",
                    }
                ],
                "judge_step_traces": [
                    {
                        "judge_id": plan.judge_steps[0].id,
                        "initial_views": list(plan.judge_steps[0].default_evidence_refs),
                        "selected_view_names": list(plan.judge_steps[0].default_evidence_refs),
                        "selected_views_hash": "selected-view-debug",
                        "final_answer": True,
                    }
                ],
                "metadata": {"packet_fingerprint": "packet-debug"},
            },
        },
    }
    reuse_cache = build_judge_reuse_cache(
        previous_result,
        packet_fingerprint="packet-debug",
    )
    assert reuse_cache, "direct inference.trace judge_calls should build reuse cache"

    weak_previous_result = {
        "transaction": {"tx_hash": "0x1"},
        "detector_context": {
            "rule_id": rule.rule_id,
            "rule_version": rule.version,
        },
        "inference": {
            "plan": plan.to_dict(),
            "trace": {
                "judge_calls": [
                    {
                        "judge_id": plan.judge_steps[0].id,
                        "id": plan.judge_steps[0].id,
                        "answer": True,
                        "confidence": "medium",
                    }
                ],
                "judge_step_traces": [
                    {
                        "judge_id": plan.judge_steps[0].id,
                        "final_answer": True,
                    }
                ],
                "metadata": {"packet_fingerprint": "packet-debug"},
            },
        },
    }
    weak_reuse_cache = build_judge_reuse_cache(
        weak_previous_result,
        packet_fingerprint="packet-debug",
    )
    assert not weak_reuse_cache, "weak entries without selected_views_hash should not be reused"

    print({
        "plan_only": _names(plan_only),
        "metadata_only_plan": _names(metadata_only),
        "rule_only": _names(rule_only),
        "both": _names(both),
        "combined_candidate_enabled": False,
        "rule_candidate_plan_generation_fallback": fallback,
        "regenerated_plan_source": regenerated_plan.metadata.get("source"),
        "candidate_reuse_scope_reusable": reuse_scope["reusable_by_tx"],
        "candidate_reuse_scope_skipped": reuse_scope["skipped"],
        "candidate_reuse_step_guard": reuse_scope["step_level_guard"],
        "direct_trace_reuse_cache_entries": len(reuse_cache),
        "weak_trace_reuse_cache_entries": len(weak_reuse_cache),
    })


if __name__ == "__main__":
    main()
