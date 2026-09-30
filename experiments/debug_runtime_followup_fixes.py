from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evotx.core.schemas import EvidencePlan, JudgeResult, JudgeStep
from evotx.runtime.evidence_tool_registry import EvidenceToolRegistry
from evotx.runtime.packet_runtime import (
    PacketRuntime,
    _aggregate_verdict,
    _analyze_dynamic_aggregation_context,
    _dynamic_attack_override_allowed,
    _find_near_miss_escalations,
    _make_near_miss_escalation_step,
    _prepare_dynamic_aggregation_followup_requests,
    _sanitize_dynamic_aggregation_decision,
    _should_follow_up,
    collect_required_packet_views_from_plan,
    validate_judge_evidence_ids,
)
from evotx.runtime.packet_view_registry import PacketViewRegistry
from evotx.runtime.source_tools import SourceToolRegistry


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_plan() -> EvidencePlan:
    return EvidencePlan.from_dict({
        "plan_id": "debug_plan",
        "rule_id": "debug_rule",
        "rule_version": 1,
        "focus_steps": [],
        "emit_logic": "C1 and C2 and not E1",
        "judge_steps": [
            {
                "id": "C1",
                "question": "Does the local packet show the core condition?",
                "default_evidence_refs": ["critical_call_view"],
                "allowed_followup_views": ["trace_view", "state_change_view"],
                "allowed_tools": ["read_packet_view", "read_evidence_by_id"],
                "max_followups": 1,
            },
            {
                "id": "C2",
                "question": "Does the outcome follow from the core condition?",
                "default_evidence_refs": ["value_release_view"],
                "allowed_followup_views": ["participant_net_delta_view"],
            },
            {
                "id": "E1",
                "question": "Does an exclusion apply?",
                "default_evidence_refs": ["operation_summary_view"],
                "expected_answer": True,
            },
        ],
    })


def test_required_views() -> None:
    required = collect_required_packet_views_from_plan(_sample_plan())
    for view in (
        "tx_card",
        "evidence_adequacy_view",
        "operation_summary_view",
        "classification_digest_view",
        "critical_call_view",
        "trace_view",
        "state_change_view",
    ):
        assert view in required, view
    print("[ok] required view collection")


def test_followup_policy() -> None:
    core_step = JudgeStep.from_dict({
        "id": "C1",
        "question": "source-dependent core",
        "allowed_tools": ["read_function_chunk"],
        "max_followups": 2,
    })
    exclusion_step = JudgeStep.from_dict({
        "id": "E1",
        "question": "exclusion",
        "allowed_tools": ["read_packet_view"],
        "max_followups": 1,
    })
    false_low = JudgeResult(
        id="C1",
        question="q",
        answer=False,
        confidence="low",
        reason="need source",
        tool_requests=[{"tool": "read_function_chunk", "args": {"evidence_id": "call:1"}}],
    )
    false_medium = JudgeResult(
        id="C1",
        question="q",
        answer=False,
        confidence="medium",
        reason="final",
        tool_requests=[{"tool": "read_packet_view", "args": {"view": "trace_view"}}],
    )
    true_low = JudgeResult(
        id="C1",
        question="q",
        answer=True,
        confidence="low",
        reason="final",
        tool_requests=[{"tool": "read_packet_view", "args": {"view": "trace_view"}}],
    )
    assert _should_follow_up(false_low, 1, step=core_step)
    assert not _should_follow_up(false_low, 1, step=exclusion_step)
    assert not _should_follow_up(false_medium, 1, step=core_step)
    assert not _should_follow_up(true_low, 1, step=core_step)
    print("[ok] follow-up policy")


def test_evidence_store_context_and_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "packet" / "tx_evidence_store.json"
        store = {
            "format": "evotx_evidence_store_v1",
            "evidence": {
                "call:0": {
                    "evidence_id": "call:0",
                    "id": 0,
                    "type": "call",
                    "function": "transaction_root",
                    "callee": "0x0000000000000000000000000000000000000001",
                },
                "call:1": {
                    "evidence_id": "call:1",
                    "id": 1,
                    "parent_id": 0,
                    "type": "call",
                    "function": "withdraw",
                    "callee": "0x0000000000000000000000000000000000000002",
                },
            },
        }
        _write_json(store_path, store)
        packet = {
            "transaction_hash": "0xdebug",
            "views": {"tx_card": {"chain": "eth"}},
            "evidence_store": {"path": str(store_path), "count": 2},
        }
        registry = PacketViewRegistry(packet)
        context = registry.get_local_call_context({"call_id": 0})
        assert context["tool_status"] == "ok"
        assert context["summary"]["child_count"] == 1
        evidence_context = registry.read_evidence_context({"evidence_id": "call:1"})
        assert evidence_context["tool_status"] == "ok"
        assert "call:0" in evidence_context["returned_evidence_ids"]

        tool_registry = EvidenceToolRegistry(packet)
        result = JudgeResult(
            id="C1",
            question="q",
            answer=True,
            confidence="high",
            reason="r",
            supporting_evidence_ids=["call:1", "fake:999"],
        )
        validation = validate_judge_evidence_ids(result, tool_registry)
        assert result.supporting_evidence_ids == ["call:1"]
        assert validation["invalid_evidence_ids"] == ["fake:999"]
    print("[ok] evidence_store context and evidence id validation")


def test_tool_request_validation() -> None:
    validated = PacketRuntime._validate_tool_requests(
        [{"tool": "read_evidence_by_id", "args": {"evidence_id": "call:12"}}],
        allowed_tools=["read_evidence_by_id"],
        allowed_followup_views=[],
    )
    assert validated[0]["valid"]
    assert validated[0]["args"]["evidence_ids"] == ["call:12"]
    blocked = PacketRuntime._validate_tool_requests(
        [{"tool": "read_evidence_by_id", "args": {"evidence_ids": ["bad id"]}}],
        allowed_tools=["read_evidence_by_id"],
        allowed_followup_views=[],
    )
    assert not blocked[0]["valid"]
    print("[ok] read_evidence_by_id validation")


def test_aggregation_policy() -> None:
    plan = _sample_plan()
    attack, aggregation = _aggregate_verdict(
        plan=plan,
        judge_results=[
            {"id": "C1", "answer": True, "confidence": "high"},
            {"id": "C2", "answer": True, "confidence": "high"},
            {"id": "E1", "answer": "uncertain", "confidence": "low"},
        ],
        judge_values={"C1": True, "C2": True, "E1": False},
        should_emit=True,
        has_uncertain=True,
        emit_logic_error=None,
    )
    assert attack == "attack"
    assert aggregation["exclusion_uncertain_policy"] == "warning_only"
    uncertain, _ = _aggregate_verdict(
        plan=plan,
        judge_results=[
            {"id": "C1", "answer": "uncertain", "confidence": "low"},
            {"id": "C2", "answer": True, "confidence": "high"},
            {"id": "E1", "answer": False, "confidence": "high"},
        ],
        judge_values={"C1": False, "C2": True, "E1": False},
        should_emit=False,
        has_uncertain=True,
        emit_logic_error=None,
    )
    assert uncertain == "uncertain"
    print("[ok] aggregation exclusion uncertain policy")


def test_access_control_cluster_near_miss() -> None:
    plan = EvidencePlan.from_dict({
        "plan_id": "ac_plan",
        "rule_id": "ac_rule",
        "rule_version": 1,
        "focus_steps": [],
        "emit_logic": "(C1 and C2 and C3) and not (E1 or E2)",
        "judge_steps": [
            {"id": "C1", "question": "sensitive op"},
            {"id": "C2", "question": "auth boundary"},
            {"id": "C3", "question": "effect"},
            {"id": "E1", "question": "authorized"},
            {"id": "E2", "question": "other mechanism"},
        ],
    })
    near = _find_near_miss_escalations(
        plan=plan,
        judge_results=[
            {
                "id": "C1",
                "condition_id": "C1",
                "answer": True,
                "confidence": "medium",
                "supporting_evidence_ids": ["call:1"],
            },
            {
                "id": "C2",
                "condition_id": "C2",
                "answer": False,
                "confidence": "medium",
                "reason": "No explicit owner or role evidence is shown; source semantics are unresolved.",
                "condition_feature_analysis": {
                    "missing_required_features": ["authorization boundary semantics unresolved"]
                },
            },
            {
                "id": "C3",
                "condition_id": "C3",
                "answer": True,
                "confidence": "medium",
                "supporting_evidence_ids": ["value_release:call:1"],
            },
            {"id": "E1", "condition_id": "E1", "answer": False, "confidence": "high"},
            {
                "id": "E2",
                "condition_id": "E2",
                "answer": True,
                "confidence": "medium",
                "reason": "Better explained by another mechanism because access-control evidence is not shown.",
                "supporting_evidence_ids": ["classification_digest:0"],
            },
        ],
        should_emit=False,
        emit_logic_error=None,
        attack_label="access_control",
    )
    assert [item["condition_id"] for item in near] == ["C2", "E2"]
    assert all(item["cluster_id"] == "access_control_authorization_boundary" for item in near)
    assert all(item["force_source_followup"] for item in near)
    print("[ok] access-control clustered near-miss")


def test_access_control_single_near_miss_forces_source() -> None:
    plan = EvidencePlan.from_dict({
        "plan_id": "ac_plan",
        "rule_id": "ac_rule",
        "rule_version": 1,
        "focus_steps": [],
        "emit_logic": "(C1 and C2 and C3) and not (E1 or E2)",
        "judge_steps": [
            {"id": "C1", "question": "sensitive op"},
            {"id": "C2", "question": "auth boundary"},
            {"id": "C3", "question": "effect"},
            {"id": "E1", "question": "authorized"},
            {"id": "E2", "question": "other mechanism"},
        ],
    })
    near = _find_near_miss_escalations(
        plan=plan,
        judge_results=[
            {
                "id": "C1",
                "condition_id": "C1",
                "answer": True,
                "confidence": "high",
                "supporting_evidence_ids": ["call:7"],
            },
            {
                "id": "C2",
                "condition_id": "C2",
                "answer": False,
                "confidence": "medium",
                "reason": "No explicit owner or role evidence is shown; source semantics are unresolved.",
            },
            {
                "id": "C3",
                "condition_id": "C3",
                "answer": True,
                "confidence": "high",
                "supporting_evidence_ids": ["value_release:call:7"],
            },
            {"id": "E1", "condition_id": "E1", "answer": False, "confidence": "high"},
            {"id": "E2", "condition_id": "E2", "answer": False, "confidence": "high"},
        ],
        should_emit=False,
        emit_logic_error=None,
        attack_label="access_control",
    )
    assert len(near) == 1
    assert near[0]["condition_id"] == "C2"
    assert near[0]["force_source_followup"]
    assert near[0]["source_probe_evidence_ids"] == ["call:7", "value_release:call:7"]
    print("[ok] access-control single near-miss source follow-up")


def test_dynamic_aggregation_hard_blocker_is_agent_declared() -> None:
    plan = _sample_plan()
    analysis = _analyze_dynamic_aggregation_context(
        plan=plan,
        judge_results=[
            {"id": "C1", "condition_id": "C1", "answer": True, "confidence": "high"},
            {"id": "C2", "condition_id": "C2", "answer": True, "confidence": "high"},
            {"id": "E1", "condition_id": "E1", "answer": True, "confidence": "high"},
        ],
        attack_label="access_control",
        near_miss_escalations=[],
    )
    assert not analysis["hard_blockers"]
    assert _dynamic_attack_override_allowed(analysis)
    decision = _sanitize_dynamic_aggregation_decision(
        {
            "verdict": "attack",
            "confidence": "high",
            "reason": "agent found a hard blocker but returned attack",
            "hard_blockers": [{"condition_id": "E1", "reason": "agent-declared"}],
        },
        previous_verdict="benign",
        previous_reason="emit_logic_false",
        analysis=analysis,
    )
    assert decision["verdict"] == "benign"
    assert not decision["override"]
    assert decision["warnings"] == [
        "attack_override_rejected_by_agent_declared_hard_blockers"
    ]
    print("[ok] dynamic aggregation agent-declared hard blocker")


def test_dynamic_aggregation_followup_request_policy() -> None:
    plan = _sample_plan()
    judge_results = [
        {
            "id": "C1",
            "condition_id": "C1",
            "answer": False,
            "confidence": "medium",
            "supporting_evidence_ids": ["call:1"],
        },
        {
            "id": "C2",
            "condition_id": "C2",
            "answer": True,
            "confidence": "high",
            "supporting_evidence_ids": ["value_release:call:2"],
        },
        {
            "id": "E1",
            "condition_id": "E1",
            "answer": False,
            "confidence": "high",
        },
    ]
    first_pass = _sanitize_dynamic_aggregation_decision(
        {
            "verdict": "uncertain",
            "confidence": "medium",
            "reason": "source is needed",
            "followup_requests": [{
                "condition_id": "C1",
                "reason": "inspect the protected call",
                "evidence_ids": ["call:1", "invented:9"],
            }],
        },
        previous_verdict="benign",
        previous_reason="emit_logic_false",
        analysis={},
        allow_followup_requests=True,
    )
    accepted, rejected = _prepare_dynamic_aggregation_followup_requests(
        first_pass["followup_requests"] + [
            {
                "condition_id": "C1",
                "reason": "duplicate",
                "evidence_ids": ["call:1"],
            },
            {
                "condition_id": "C9",
                "reason": "unknown condition",
                "evidence_ids": ["call:1"],
            },
        ],
        plan=plan,
        judge_results=judge_results,
        near_miss_escalations=[{
            "judge_id": "C1",
            "condition_id": "C1",
            "source_probe_evidence_ids": [],
        }],
    )
    assert len(accepted) == 1
    assert accepted[0]["judge_id"] == "C1"
    assert accepted[0]["evidence_ids"] == ["call:1"]
    assert {item["rejection_reason"] for item in rejected} == {
        "duplicate_condition_request",
        "unknown_condition_id",
    }

    final_pass = _sanitize_dynamic_aggregation_decision(
        {
            "verdict": "uncertain",
            "followup_requests": [{
                "condition_id": "C1",
                "evidence_ids": ["call:1"],
            }],
        },
        previous_verdict="uncertain",
        previous_reason="still unresolved",
        analysis={},
        allow_followup_requests=False,
    )
    assert final_pass["followup_requests"] == []
    assert "followup_requests_ignored_after_max_rounds" in final_pass["warnings"]

    escalation_step = _make_near_miss_escalation_step(
        plan.judge_steps[0],
        {
            "near_miss_policy": "dynamic_aggregator_source_followup",
            "single_rejudge_only": True,
            "aggregator_followup_reason": "source is needed",
        },
    )
    assert escalation_step.max_followups == 0
    assert "global aggregator requested" in escalation_step.question
    print("[ok] dynamic aggregation one-round follow-up policy")


def test_dynamic_aggregation_routes_followup_to_owning_judge() -> None:
    class StubRuntime(PacketRuntime):
        def _run_near_miss_escalation(self, **kwargs):
            step = kwargs["step"]
            near_miss = kwargs["near_miss"]
            assert step.id == "C1"
            assert near_miss["single_rejudge_only"]
            assert near_miss["source_probe_evidence_ids"] == ["call:1"]
            return (
                JudgeResult(
                    id="C1",
                    question=step.question,
                    answer=True,
                    reason="source confirms the protected operation",
                    confidence="high",
                    supporting_evidence_ids=["call:1", "source_chunk:0"],
                ),
                {
                    "judge_calls": [{"round": 0, "answer": True}],
                    "tool_calls": [{
                        "round": -1,
                        "tool": "read_function_chunk",
                        "tool_status": "ok",
                    }],
                    "ignored_tool_requests": [],
                    "view_render_metadata": [],
                    "judge_parse_metadata": [],
                    "all_missing_evidence_debug": [],
                    "evidence_id_validations": [],
                    "invalid_evidence_ids": [],
                    "selected_view_names": ["critical_call_view"],
                    "selected_views_hash": "debug",
                    "final_answer": True,
                },
            )

    plan = _sample_plan()
    judge_results = [
        {
            "id": "C1",
            "condition_id": "C1",
            "answer": False,
            "confidence": "medium",
            "supporting_evidence_ids": ["call:1"],
        },
        {
            "id": "C2",
            "condition_id": "C2",
            "answer": True,
            "confidence": "high",
            "supporting_evidence_ids": ["value_release:call:2"],
        },
        {
            "id": "E1",
            "condition_id": "E1",
            "answer": False,
            "confidence": "high",
        },
    ]
    traces = [{"tool_calls": [], "judge_calls": []} for _ in judge_results]
    judge_values = {"C1": False, "C2": True, "E1": False}
    runtime = StubRuntime(judge_model=object())
    followup_trace = runtime._run_dynamic_aggregation_followup_round(
        packet={},
        plan=plan,
        judge_results=judge_results,
        judge_step_traces=traces,
        judge_values=judge_values,
        followup_requests=[{
            "condition_id": "C1",
            "reason": "inspect source",
            "evidence_ids": ["call:1"],
        }],
        evidence_tool_registry=EvidenceToolRegistry({}),
        tx_hash="0xdebug",
        chain="eth",
        near_miss_escalations=[{
            "judge_id": "C1",
            "condition_id": "C1",
            "source_probe_evidence_ids": [],
        }],
    )
    assert followup_trace["rerun_count"] == 1
    assert judge_results[0]["answer"] is True
    assert judge_results[0]["dynamic_aggregation_followup"] is True
    assert judge_results[1]["answer"] is True
    assert judge_values["C1"] is True
    assert len(traces[0]["judge_calls"]) == 1
    print("[ok] dynamic aggregation routes follow-up to owning judge")


def test_source_proxy_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        proxy = "0x0000000000000000000000000000000000000001"
        impl = "0x0000000000000000000000000000000000000002"
        _write_json(cache_dir / "eth" / f"{proxy}__bundle.json", {
            "meta": {"implementation": impl, "contract_name": "Proxy"},
            "functions": {},
            "from_cache": True,
            "cache_path": str(cache_dir / "eth" / f"{proxy}__bundle.json"),
        })
        _write_json(cache_dir / "eth" / f"{impl}__bundle.json", {
            "meta": {"implementation": "", "contract_name": "Impl"},
            "functions": {
                "withdraw": [{
                    "path": "Impl.sol",
                    "start_line": 1,
                    "end_line": 3,
                    "content": "function withdraw() external { }",
                }]
            },
            "from_cache": True,
            "cache_path": str(cache_dir / "eth" / f"{impl}__bundle.json"),
        })
        source = SourceToolRegistry(cache_dir=cache_dir)
        result = source.read_function_chunk(proxy, "withdraw", chain="eth")
        assert result["tool_status"] == "ok"
        assert result["summary"]["implementation_fallback_used"]
    print("[ok] source implementation fallback")


if __name__ == "__main__":
    test_required_views()
    test_followup_policy()
    test_evidence_store_context_and_validation()
    test_tool_request_validation()
    test_aggregation_policy()
    test_access_control_cluster_near_miss()
    test_access_control_single_near_miss_forces_source()
    test_dynamic_aggregation_hard_blocker_is_agent_declared()
    test_dynamic_aggregation_followup_request_policy()
    test_dynamic_aggregation_routes_followup_to_owning_judge()
    test_source_proxy_fallback()
    print("[ok] runtime follow-up fixes debug checks passed")
