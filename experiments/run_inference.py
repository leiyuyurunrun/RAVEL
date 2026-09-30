"""Run single-transaction EvoTx inference using packet-based runtime."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.adapters.llm_adapter import (
    OpenAICompatibleLLM,
    resolve_adaptive_llm_route,
)
from evotx.core.labels import normalize_attack_label
from evotx.core.plan import (
    apply_access_control_binding_mode,
    apply_insufficient_validation_stateful_runtime,
    apply_reentrancy_stateful_runtime,
    compile_rule_to_baseline_plan,
)
from evotx.core.schemas import EvidencePlan, EvolvingRule
from evotx.planner.plan_generator import PlanGenerator
from evotx.planner.plan_validator import PlanValidationError, PlanValidator
from evotx.runtime.judge import JudgeModel
from evotx.runtime.judge_reuse import build_judge_reuse_cache
from evotx.runtime.llm_transcripts import LLMTranscriptRecorder
from evotx.runtime.packet_runtime import (
    PacketRuntime,
    apply_judge_followup_policy,
    disable_stateful_bindings_in_plan,
    normalize_judge_followup_mode,
)
from evotx.runtime.source_tools import SourceToolRegistry
from evotx.storage.rule_store import RuleStore
from evotx.storage.plan_store import PlanStore
from evotx.storage.result_store import ResultStore
from evotx.storage.trace_store import TraceStore
from evotx.utils.json_utils import read_json, stable_json_dumps, write_json
from evotx.utils.fingerprint_utils import (
    object_fingerprint,
    semantic_plan_fingerprint,
)
from evotx.utils.logging_utils import configure_session_logging
from evotx.utils.result_utils import (
    build_result_record,
    get_finding,
    get_plan,
    get_trace,
)
from evotx.utils.trace_utils import build_trace_record_from_result


# ---------------------------------------------------------------------------
# Result adapter: PacketRuntime output -> old runtime_result format
# ---------------------------------------------------------------------------

def _packet_result_to_runtime_result(
    packet_result: Dict[str, Any],
    plan: Dict[str, Any],
    packet: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert PacketRuntime output to the runtime_result format expected by
    build_result_record, so all downstream consumers (reviewer, regression,
    result_utils) continue to work unchanged."""
    rule_dict = packet_result.get("rule", {})
    judge_results = packet_result.get("judge_results", [])
    rt = packet_result.get("runtime_trace", {})
    verdict = packet_result["verdict"]
    supporting_ids = packet_result.get("supporting_evidence_ids", [])
    evidence_groups = packet_result.get("evidence_groups", {}) or {}

    # -- finding --
    satisfied_results = [
        jr for jr in judge_results if jr.get("answer") is True
    ]
    confidences = [
        jr.get("confidence", "medium")
        for jr in satisfied_results
    ]
    if any(c == "low" for c in confidences):
        confidence = "low"
    elif confidences and all(c == "high" for c in confidences):
        confidence = "high"
    else:
        confidence = "medium"

    missing = sorted({
        m
        for jr in judge_results
        for m in jr.get("missing_evidence", [])
    })
    all_missing_debug = [
        item
        for jr in judge_results
        for item in jr.get("all_missing_evidence_debug", [])
    ]
    ignored_tool_requests = [
        item
        for jr in judge_results
        for item in jr.get("ignored_tool_requests", [])
    ]

    finding = {
        "rule_id": rule_dict.get("rule_id", ""),
        "rule_version": rule_dict.get("version", 1),
        "rule_name": rule_dict.get("name", ""),
        "verdict": verdict,
        "description": f"Packet-based runtime: {rule_dict.get('name', '')}",
        "supporting_evidence": supporting_ids,
        "judge_results": [
            jr.get("id", "")
            for jr in satisfied_results
        ],
        "confidence": confidence,
        "verdict_reason": packet_result.get("verdict_reason", ""),
        "verdict_aggregation": packet_result.get("verdict_aggregation", {}),
        "missing_evidence": missing,
        "all_missing_evidence_debug": all_missing_debug,
        "ignored_tool_requests": ignored_tool_requests,
        "attack_supporting_evidence_by_condition": evidence_groups.get(
            "attack_supporting_evidence_by_condition", {}
        ),
        "benign_exclusion_evidence_by_condition": evidence_groups.get(
            "benign_exclusion_evidence_by_condition", {}
        ),
        "failed_core_conditions": evidence_groups.get("failed_core_conditions", {}),
        "missing_evidence_by_condition": evidence_groups.get(
            "missing_evidence_by_condition", {}
        ),
    }

    # -- trace --
    judge_calls = []
    for jr in judge_results:
        judge_calls.append({
            "judge_id": jr.get("id"),
            "condition_id": jr.get("condition_id"),
            "question": jr.get("question"),
            "answer": jr.get("answer"),
            "expected_answer": jr.get("expected_answer", True),
            "satisfied": bool(jr.get("satisfied", False)),
            "reason": jr.get("reason"),
            "confidence": jr.get("confidence"),
            "evidence_refs": list(jr.get("evidence_refs", [])),
            "supporting_evidence_ids": jr.get("supporting_evidence_ids", []),
            "missing_evidence": jr.get("missing_evidence", []),
            "all_missing_evidence_debug": jr.get("all_missing_evidence_debug", []),
            "tool_calls": jr.get("tool_calls", []),
            "ignored_tool_requests": jr.get("ignored_tool_requests", []),
            "view_render_metadata": jr.get("view_render_metadata", []),
            "selected_view_names": jr.get("selected_view_names", []),
            "selected_views_hash": jr.get("selected_views_hash", ""),
            "reused_judge_result": jr.get("reused_judge_result", False),
            "reuse_fingerprint": jr.get("reuse_fingerprint", ""),
            "reuse_source": jr.get("reuse_source", {}),
            "state_input": jr.get("state_input", {}),
            "state_output": jr.get("state_output", {}),
            "stateful_runtime": jr.get("stateful_runtime", {}),
            "judge_followup_policy": jr.get("judge_followup_policy", {}),
        })

    tool_calls = [
        call
        for jr in judge_results
        for call in jr.get("tool_calls", [])
    ]
    runtime_tool_calls = rt.get("tool_calls", [])
    if runtime_tool_calls:
        tool_calls = runtime_tool_calls

    trace = {
        "tx_hash": packet_result["tx_hash"],
        "rule_id": rule_dict.get("rule_id", ""),
        "rule_version": rule_dict.get("version", 1),
        "plan_id": plan.get("plan_id", ""),
        "tool_calls": tool_calls,
        "judge_calls": judge_calls,
        "judge_step_traces": rt.get("judge_step_traces", []),
        "emissions": [{"verdict": verdict}],
        "errors": rt.get("errors", []),
        "started_at": None,
        "ended_at": None,
        "metadata": {
            "packet_source": packet_result.get("packet_source", ""),
            "packet_loaded_from_cache": rt.get("packet_loaded_from_cache", False),
            "packet_frozen": rt.get("packet_frozen", False),
            "packet_replayed_from_baseline": rt.get(
                "packet_replayed_from_baseline", False
            ),
            "packet_identity_fingerprint": rt.get("packet_identity_fingerprint", ""),
            "packet_fingerprint": rt.get("packet_fingerprint", ""),
            "plan_identity": rt.get("plan_identity", {}),
            "runtime_policy_identity": rt.get("runtime_policy_identity", ""),
            "runtime_policy": rt.get("runtime_policy", {}),
            "packet_build_config": rt.get("packet_build_config", {}),
            "packet_view_cost_summary": rt.get("packet_view_cost_summary", {}),
            "view_manifest_version": rt.get("view_manifest_version", ""),
            "evidence_profile": rt.get("evidence_profile", {}),
            "adaptive_evidence": rt.get("adaptive_evidence", {}),
            "packet_required_views": rt.get("packet_required_views", []),
            "packet_required_views_missing": rt.get("packet_required_views_missing", []),
            "judge_call_count": rt.get("judge_call_count", 0),
            "elapsed_seconds": rt.get("elapsed_seconds", 0),
            "has_uncertain": rt.get("has_uncertain", False),
            "uncertain_judge_ids": rt.get("uncertain_judge_ids", []),
            "verdict_reason": packet_result.get("verdict_reason", ""),
            "verdict_aggregation": packet_result.get("verdict_aggregation", {}),
            "judge_reuse": rt.get("judge_reuse", {}),
            "early_stop": rt.get("early_stop", {}),
            "parallel_judge": rt.get("parallel_judge", {}),
            "stateful_runtime": rt.get("stateful_runtime", {}),
            "judge_followup_policy": rt.get("judge_followup_policy", {}),
            "skipped_judge_step_count": rt.get("skipped_judge_step_count", 0),
            "judge_step_traces": rt.get("judge_step_traces", []),
        },
    }

    # -- evidence --
    evidence = packet.get("views", {}) if packet else {}

    return {
        "tx_hash": packet_result["tx_hash"],
        "rule": rule_dict,
        "plan": plan,
        "evidence": evidence,
        "packet_snapshot": _packet_snapshot_for_result(packet, packet_result),
        "finding": finding,
        "trace": trace,
    }


def _packet_snapshot_for_result(
    packet: Dict[str, Any],
    packet_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Store a replayable packet without duplicating ``inference.evidence``."""
    if not isinstance(packet, dict) or not packet:
        return {}
    runtime_trace = dict(packet_result.get("runtime_trace", {}) or {})
    if not runtime_trace.get("packet_frozen"):
        return {}
    inline_store = packet.get("evidence_store_inline", {})
    snapshot = {
        key: copy.deepcopy(value)
        for key, value in packet.items()
        if key not in {"views", "evidence_store_inline"}
        and not str(key).startswith("_frozen_packet_")
    }
    store_ref = snapshot.get("evidence_store", {})
    store_path = str(store_ref.get("path") or "") if isinstance(store_ref, dict) else ""
    if isinstance(inline_store, dict) and inline_store:
        snapshot["evidence_store_inline"] = copy.deepcopy(inline_store)
    elif store_path:
        try:
            snapshot["evidence_store_inline"] = read_json(store_path)
        except Exception:
            snapshot["evidence_store_inline"] = {}
    snapshot["snapshot_schema_version"] = "evotx.packet_snapshot.v1"
    snapshot["packet_identity_fingerprint"] = str(
        runtime_trace.get("packet_identity_fingerprint") or ""
    )
    snapshot["packet_source"] = str(packet_result.get("packet_source") or "")
    snapshot["views_source"] = "inference.evidence"
    return snapshot


def _frozen_packet_from_result(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None
    inference = result.get("inference", {})
    if not isinstance(inference, dict):
        return None
    snapshot = inference.get("packet_snapshot", {})
    evidence = inference.get("evidence", {})
    if not isinstance(snapshot, dict) or not snapshot:
        return None
    if snapshot.get("snapshot_schema_version") != "evotx.packet_snapshot.v1":
        return None
    if not isinstance(evidence, dict) or not evidence:
        return None
    packet = copy.deepcopy(snapshot)
    packet["views"] = copy.deepcopy(evidence)
    packet["_frozen_packet_identity_fingerprint"] = str(
        packet.pop("packet_identity_fingerprint", "") or ""
    )
    packet["_frozen_packet_source"] = str(
        packet.pop("packet_source", "") or "frozen:baseline_result"
    )
    packet.pop("snapshot_schema_version", None)
    packet.pop("views_source", None)
    return packet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-transaction EvoTx inference using packet-based runtime."
    )
    parser.add_argument("--tx-hash", required=True)
    parser.add_argument("--chain", default="eth")
    parser.add_argument("--rule-file")
    parser.add_argument(
        "--label",
        help="Attack-family label for selecting the latest rule.",
    )
    parser.add_argument("--rules-dir", default="data/rules")
    parser.add_argument("--tool-manifest", default="configs/tool_manifest.json")
    parser.add_argument("--tx-context", help="Optional transaction summary JSON.")
    parser.add_argument(
        "--llm-model",
        help="Default model reused for planner/judge unless overridden.",
    )
    parser.add_argument(
        "--llm-provider",
        help="Optional provider override for all inference LLMs: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument("--planner-model")
    parser.add_argument(
        "--plan-file",
        help="Optional fixed EvidencePlan JSON. When set, planner generation is skipped.",
    )
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=8192,
        help=(
            "Maximum output tokens for each Judge request; "
            "MiniMax uses max_completion_tokens and is capped at 32768."
        ),
    )
    parser.add_argument(
        "--aggregator-max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens for each dynamic Aggregator request.",
    )
    parser.add_argument(
        "--judge-thinking",
        choices=["default", "disabled", "adaptive"],
        default="disabled",
        help="MiniMax thinking mode for Judge calls.",
    )
    parser.add_argument(
        "--aggregator-thinking",
        choices=["default", "disabled", "adaptive"],
        default="disabled",
        help="MiniMax thinking mode for dynamic Aggregator calls.",
    )
    parser.add_argument("--plans-dir", default="data/plans")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--traces-dir", default="data/traces")
    parser.add_argument("--logs-dir", default="data/log")
    parser.add_argument("--base-cache-dir", default="data/cache")
    parser.add_argument(
        "--enable-source-tools",
        action="store_true",
        help=(
            "Enable read_function_chunk for judge follow-ups and the optional "
            "aggregator-requested follow-up round."
        ),
    )
    parser.add_argument("--source-cache-dir", default="data/cache/contracts")
    parser.add_argument("--force-refresh-source", action="store_true", help="Refresh explorer source cache for read_function_chunk.")
    parser.add_argument(
        "--max-view-chars",
        type=int,
        default=20000,
        help="Maximum rendered characters per packet view passed to the judge.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=60000,
        help="Maximum total rendered evidence context characters passed to the judge.",
    )
    parser.add_argument(
        "--force-rebuild-packet",
        action="store_true",
        help="Rebuild compact packet from source cache instead of using cached packet.",
    )
    parser.add_argument(
        "--adaptive-evidence",
        action="store_true",
        default=False,
        help="Enable conservative adaptive first-pass evidence routing.",
    )
    parser.add_argument(
        "--adaptive-evidence-mode",
        choices=["off", "auto", "planned", "small_trace_direct", "medium_hybrid", "direct_trace"],
        default="off",
        help="Adaptive evidence routing mode. Default off preserves the plan exactly.",
    )
    parser.add_argument(
        "--adaptive-evidence-max-direct-trace-nodes",
        type=int,
        default=80,
        help="Max trace nodes for direct-trace first-pass routing.",
    )
    parser.add_argument(
        "--adaptive-evidence-max-direct-trace-chars",
        type=int,
        default=45000,
        help="Max trace_view chars for direct-trace first-pass routing.",
    )
    parser.add_argument(
        "--adaptive-evidence-max-medium-trace-nodes",
        type=int,
        default=250,
        help="Max trace nodes for medium hybrid routing.",
    )
    parser.add_argument(
        "--adaptive-evidence-debug",
        action="store_true",
        default=False,
        help="Print adaptive evidence routing details in runtime traces/debug artifacts.",
    )
    parser.add_argument(
        "--enforce-label-plan-template",
        dest="enforce_label_plan_template",
        action="store_true",
        default=True,
        help="Apply label-specific deterministic plan templates during plan generation.",
    )
    parser.add_argument(
        "--no-enforce-label-plan-template",
        dest="enforce_label_plan_template",
        action="store_false",
        help="Do not overwrite generated plan defaults with label templates.",
    )
    parser.add_argument(
        "--parallel-judge",
        action="store_true",
        default=False,
        help="Run independent judge steps concurrently within one transaction.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent judge workers when --parallel-judge is enabled.",
    )
    parser.add_argument(
        "--structured-judge-logs",
        dest="structured_judge_logs",
        action="store_true",
        default=True,
        help="Print per-judge structured JSONL logs during parallel execution.",
    )
    parser.add_argument(
        "--no-structured-judge-logs",
        dest="structured_judge_logs",
        action="store_false",
        help="Disable per-judge structured JSONL logs.",
    )
    parser.add_argument(
        "--enable-iv-stateful-runtime",
        dest="iv_stateful_runtime",
        action="store_true",
        default=True,
        help="Enable insufficient_validation object-binding stateful runtime.",
    )
    parser.add_argument(
        "--disable-iv-stateful-runtime",
        dest="iv_stateful_runtime",
        action="store_false",
        help="Disable insufficient_validation object-binding stateful runtime.",
    )
    parser.add_argument(
        "--access-control-binding-mode",
        choices=["prompt", "stateful"],
        default="stateful",
        help=(
            "Use prompt-only Access Control chain consistency, or enable "
            "stateful semantic-role binding between core Judge steps."
        ),
    )
    parser.add_argument(
        "--reentrancy-binding-mode",
        choices=["disabled", "soft", "stateful"],
        default="soft",
        help=(
            "Disable Reentrancy candidate binding, preserve candidate state "
            "without answer overrides (soft), or enable strong candidate-level "
            "normalization and emit (stateful)."
        ),
    )
    parser.add_argument(
        "--disable-stateful-bindings",
        action="store_true",
        default=False,
        help=(
            "Ablation mode: ignore depends_on, consumes_state_keys, "
            "produces_state_key, state_prompt_role, state_output_schema, and "
            "judge state_output at runtime."
        ),
    )
    parser.add_argument(
        "--enable-dynamic-aggregation",
        dest="dynamic_aggregation",
        action="store_true",
        default=True,
        help=(
            "Enable dynamic aggregation with at most one aggregator-requested "
            "source follow-up round owned by the local judges."
        ),
    )
    parser.add_argument(
        "--disable-dynamic-aggregation",
        dest="dynamic_aggregation",
        action="store_false",
        help="Disable soft-rule dynamic aggregation and use only emit_logic aggregation.",
    )
    parser.add_argument(
        "--followup-context-mode",
        choices=["legacy", "unified"],
        default="unified",
        help=(
            "Judge follow-up context policy. unified applies one shared budget, "
            "keeps the previous judge summary, and prioritizes the newest tool observation."
        ),
    )
    parser.add_argument(
        "--judge-followup-mode",
        choices=["plan", "disabled", "expanded"],
        default="plan",
        help=(
            "Judge evidence-delivery ablation: plan keeps iterative follow-up; "
            "disabled uses only initial plan views; expanded preloads selected "
            "follow-up views, then performs one Judge call."
        ),
    )
    parser.add_argument(
        "--judge-expanded-followup-views",
        type=int,
        default=2,
        help="Maximum high-priority follow-up views promoted in expanded mode.",
    )
    parser.add_argument(
        "--rate-limit-serial-fallback",
        dest="rate_limit_serial_fallback",
        action="store_true",
        default=True,
        help="On judge HTTP 429, retry affected steps serially and disable later parallel judge calls.",
    )
    parser.add_argument(
        "--disable-rate-limit-serial-fallback",
        dest="rate_limit_serial_fallback",
        action="store_false",
        help="Disable automatic judge concurrency downgrade after HTTP 429.",
    )
    parser.add_argument(
        "--rate-limit-retry-attempts",
        type=int,
        default=3,
        help="Maximum serial retries for a rate-limited judge step.",
    )
    parser.add_argument(
        "--rate-limit-retry-delay-seconds",
        type=float,
        default=2.0,
        help="Initial 429 retry delay; subsequent retries use exponential backoff.",
    )
    parser.add_argument(
        "--save-llm-transcripts",
        action="store_true",
        help="Save judge prompts, raw completions, usage, render metadata, and tool observations.",
    )
    parser.add_argument(
        "--llm-transcripts-dir",
        default="data/transcripts",
        help="Directory used when --save-llm-transcripts is enabled.",
    )
    parser.add_argument("--ground-truth", choices=["attack", "benign"])
    parser.add_argument(
        "--ground-truth-label",
        help="Optional raw task label, e.g. reentrancy.",
    )
    args = parser.parse_args()
    log_path = configure_session_logging(
        "run_inference",
        logs_dir=args.logs_dir,
        suffix=f"{args.chain}__{args.tx_hash[:12]}__{args.label or 'rule'}",
    )
    print(
        f"[RunInference] tx={args.tx_hash} chain={args.chain} "
        f"label={args.label or '(from rule file)'}"
    )

    rule, rule_source = load_rule_input(
        rule_file=args.rule_file,
        label=args.label,
        rules_dir=args.rules_dir,
    )
    resolved_planner, resolved_judge, _ = resolve_inference_models(
        llm_model=args.llm_model,
        planner_model=args.planner_model,
        judge_model=args.judge_model,
        env_model=None,
    )
    print(
        "[RunInference] Models: "
        f"planner={resolved_planner or 'disabled'} "
        f"judge={resolved_judge or 'disabled'}"
    )
    tool_manifest = read_json(args.tool_manifest, default={}) or {}
    tx_context = read_json(args.tx_context, default={}) if args.tx_context else {}
    fixed_plan = load_plan_input(args.plan_file) if args.plan_file else None

    result = run_single_inference(
        tx_hash=args.tx_hash,
        chain=args.chain,
        rule=rule,
        rule_source=rule_source,
        tool_manifest=tool_manifest,
        tx_context=tx_context or {},
        llm_provider=args.llm_provider,
        planner_model=resolved_planner,
        judge_model=resolved_judge,
        judge_max_tokens=args.judge_max_tokens,
        aggregator_max_tokens=args.aggregator_max_tokens,
        judge_thinking=args.judge_thinking,
        aggregator_thinking=args.aggregator_thinking,
        fixed_plan=fixed_plan,
        plan_source=args.plan_file,
        base_cache_dir=args.base_cache_dir,
        force_rebuild_packet=args.force_rebuild_packet,
        enable_source_tools=args.enable_source_tools,
        source_cache_dir=args.source_cache_dir,
        force_refresh_source=args.force_refresh_source,
        max_view_chars=args.max_view_chars,
        max_context_chars=args.max_context_chars,
        adaptive_evidence=args.adaptive_evidence,
        adaptive_evidence_mode=args.adaptive_evidence_mode,
        adaptive_evidence_max_direct_trace_nodes=args.adaptive_evidence_max_direct_trace_nodes,
        adaptive_evidence_max_direct_trace_chars=args.adaptive_evidence_max_direct_trace_chars,
        adaptive_evidence_max_medium_trace_nodes=args.adaptive_evidence_max_medium_trace_nodes,
        adaptive_evidence_debug=args.adaptive_evidence_debug,
        enforce_label_plan_template=args.enforce_label_plan_template,
        parallel_judge=args.parallel_judge,
        judge_concurrency=args.judge_concurrency,
        structured_judge_logs=args.structured_judge_logs,
        iv_stateful_runtime=args.iv_stateful_runtime,
        access_control_binding_mode=args.access_control_binding_mode,
        reentrancy_binding_mode=args.reentrancy_binding_mode,
        disable_stateful_bindings=args.disable_stateful_bindings,
        dynamic_aggregation=args.dynamic_aggregation,
        followup_context_mode=args.followup_context_mode,
        judge_followup_mode=args.judge_followup_mode,
        judge_expanded_followup_views=args.judge_expanded_followup_views,
        rate_limit_serial_fallback=args.rate_limit_serial_fallback,
        rate_limit_retry_attempts=args.rate_limit_retry_attempts,
        rate_limit_retry_delay_seconds=args.rate_limit_retry_delay_seconds,
        ground_truth=args.ground_truth,
        raw_ground_truth=args.ground_truth_label,
        llm_transcripts_dir=(
            args.llm_transcripts_dir if args.save_llm_transcripts else None
        ),
        transcript_phase="run_inference",
    )

    plans_dir = Path(args.plans_dir)
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan = get_plan(result)
    plan_path = (
        plans_dir
        / f"{args.tx_hash.lower()}__{rule.rule_id}__{plan['plan_id']}.json"
    )
    write_json(plan_path, plan)

    result_path = ResultStore(args.results_dir).save(result)
    trace_path = TraceStore(args.traces_dir).save(
        build_trace_record_from_result(result)
    )

    print(f"Plan: {plan_path}")
    print(f"Result: {result_path}")
    print(f"Trace: {trace_path}")
    print(f"Rule source: {rule_source}")
    print(f"Attack label: {rule.metadata.get('attack_label', 'attack')}")
    print(f"Verdict: {get_finding(result).get('verdict')}")
    print(f"Session log: {log_path}")


# ---------------------------------------------------------------------------
# Shared inference function (used by train_fewshot, evaluate, etc.)
# ---------------------------------------------------------------------------

def build_runtime_policy_identity(config: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical identity for settings that can change runtime judgments."""
    raw = dict(config or {})
    followup_mode = normalize_judge_followup_mode(
        str(raw.get("judge_followup_mode") or "plan")
    )
    judge_model, judge_provider = resolve_adaptive_llm_route(
        model=raw.get("judge_model"),
        provider=raw.get("llm_provider"),
        thinking=str(raw.get("judge_thinking") or "disabled"),
        adaptive_model=raw.get("adaptive_model"),
        adaptive_provider=raw.get("adaptive_provider"),
    )
    aggregator_model, aggregator_provider = resolve_adaptive_llm_route(
        model=raw.get("judge_model"),
        provider=raw.get("llm_provider"),
        thinking=str(raw.get("aggregator_thinking") or "disabled"),
        adaptive_model=raw.get("adaptive_model"),
        adaptive_provider=raw.get("adaptive_provider"),
    )
    payload = {
        "schema_version": "evotx.runtime_policy_identity.v1",
        "judge": {
            "model": str(judge_model or ""),
            "provider": str(judge_provider or ""),
            "max_tokens": max(1, int(raw.get("judge_max_tokens") or 8192)),
            "thinking": str(raw.get("judge_thinking") or "disabled"),
        },
        "aggregator": {
            "model": str(aggregator_model or ""),
            "provider": str(aggregator_provider or ""),
            "max_tokens": max(
                1, int(raw.get("aggregator_max_tokens") or 8192)
            ),
            "thinking": str(raw.get("aggregator_thinking") or "disabled"),
            "enabled": bool(raw.get("dynamic_aggregation"))
            and followup_mode == "plan",
        },
        "evidence": {
            "source_followup_enabled": bool(raw.get("enable_source_tools")),
            "max_view_chars": int(raw.get("max_view_chars") or 0),
            "max_context_chars": int(raw.get("max_context_chars") or 0),
            "adaptive_enabled": bool(raw.get("adaptive_evidence"))
            and followup_mode == "plan",
            "adaptive_mode": str(raw.get("adaptive_evidence_mode") or "off"),
            "adaptive_max_direct_trace_nodes": int(
                raw.get("adaptive_evidence_max_direct_trace_nodes") or 0
            ),
            "adaptive_max_direct_trace_chars": int(
                raw.get("adaptive_evidence_max_direct_trace_chars") or 0
            ),
            "adaptive_max_medium_trace_nodes": int(
                raw.get("adaptive_evidence_max_medium_trace_nodes") or 0
            ),
        },
        "execution": {
            "early_stop": bool(raw.get("runtime_early_stop")),
            "early_stop_policy": str(
                raw.get("runtime_early_stop_policy") or "conservative_negative"
            ),
            "parallel_judge": bool(raw.get("parallel_judge")),
            "judge_concurrency": max(1, int(raw.get("judge_concurrency") or 1)),
            "iv_stateful_runtime": bool(raw.get("iv_stateful_runtime", True)),
            "access_control_binding_mode": str(
                raw.get("access_control_binding_mode") or "stateful"
            ),
            "reentrancy_binding_mode": str(
                raw.get("reentrancy_binding_mode") or "soft"
            ),
            "disable_stateful_bindings": bool(
                raw.get("disable_stateful_bindings")
            ),
            "dynamic_aggregation": bool(raw.get("dynamic_aggregation"))
            and followup_mode == "plan",
            "followup_context_mode": str(
                raw.get("followup_context_mode") or "unified"
            ),
            "judge_followup_mode": followup_mode,
            "judge_expanded_followup_views": max(
                0, int(raw.get("judge_expanded_followup_views") or 0)
            ),
        },
    }
    return {
        "payload": payload,
        "fingerprint": object_fingerprint(payload, prefix="runtime:"),
    }


def prepare_effective_runtime_plan(
    rule: EvolvingRule,
    plan: EvidencePlan | Dict[str, Any],
    *,
    iv_stateful_runtime: bool = True,
    access_control_binding_mode: str = "stateful",
    reentrancy_binding_mode: str = "soft",
    disable_stateful_bindings: bool = False,
    judge_followup_mode: str = "plan",
    judge_expanded_followup_views: int = 2,
) -> EvidencePlan:
    """Preview the deterministic Plan actually handed to Judge execution."""
    effective = _clone_plan(plan)
    attack_label = str((rule.metadata or {}).get("attack_label") or "")
    if iv_stateful_runtime:
        apply_insufficient_validation_stateful_runtime(
            effective,
            attack_label=attack_label,
        )
    apply_access_control_binding_mode(
        effective,
        rule=rule,
        mode=access_control_binding_mode,
    )
    apply_reentrancy_stateful_runtime(
        effective,
        rule=rule,
        mode=reentrancy_binding_mode,
    )
    if disable_stateful_bindings:
        effective = disable_stateful_bindings_in_plan(effective)
    effective, _ = apply_judge_followup_policy(
        effective,
        mode=judge_followup_mode,
        expanded_view_count=judge_expanded_followup_views,
    )
    return effective

def run_single_inference(
    tx_hash: str,
    chain: str,
    rule: EvolvingRule,
    rule_source: Optional[str],
    tool_manifest: Dict[str, Any],
    tx_context: Optional[Dict[str, Any]] = None,
    static_evidence: Optional[Dict[str, Any]] = None,
    llm_provider: Optional[str] = None,
    planner_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_max_tokens: int = 8192,
    aggregator_max_tokens: int = 8192,
    planner_thinking: str = "adaptive",
    judge_thinking: str = "disabled",
    aggregator_thinking: str = "disabled",
    adaptive_model: Optional[str] = None,
    adaptive_provider: Optional[str] = None,
    fixed_plan: Optional[EvidencePlan | Dict[str, Any]] = None,
    plan_source: Optional[str] = None,
    env_model: Optional[str] = None,
    use_environment: bool = False,
    base_cache_dir: str = "data/cache",
    force_rebuild_packet: bool = False,
    freeze_packet: bool = False,
    enable_source_tools: bool = False,
    source_cache_dir: str = "data/cache/contracts",
    force_refresh_source: bool = False,
    max_view_chars: int = 20000,
    max_context_chars: int = 60000,
    ground_truth: Optional[str] = None,
    raw_ground_truth: Optional[str] = None,
    label_rationale: str = "",
    report: str = "",
    case_metadata: Optional[Dict[str, Any]] = None,
    reuse_result: Optional[Dict[str, Any]] = None,
    runtime_early_stop: bool = False,
    runtime_early_stop_policy: str = "conservative_negative",
    adaptive_evidence: bool = False,
    adaptive_evidence_mode: str = "off",
    adaptive_evidence_max_direct_trace_nodes: int = 80,
    adaptive_evidence_max_direct_trace_chars: int = 45000,
    adaptive_evidence_max_medium_trace_nodes: int = 250,
    adaptive_evidence_debug: bool = False,
    enforce_label_plan_template: bool = True,
    parallel_judge: bool = False,
    judge_concurrency: int = 1,
    structured_judge_logs: bool = True,
    iv_stateful_runtime: bool = True,
    access_control_binding_mode: str = "stateful",
    reentrancy_binding_mode: str = "soft",
    disable_stateful_bindings: bool = False,
    dynamic_aggregation: bool = True,
    followup_context_mode: str = "unified",
    judge_followup_mode: str = "plan",
    judge_expanded_followup_views: int = 2,
    rate_limit_serial_fallback: bool = True,
    rate_limit_retry_attempts: int = 3,
    rate_limit_retry_delay_seconds: float = 2.0,
    llm_transcripts_dir: Optional[str] = None,
    transcript_phase: str = "",
) -> Dict[str, Any]:
    normalized_judge_followup_mode = normalize_judge_followup_mode(
        judge_followup_mode
    )
    effective_dynamic_aggregation = bool(dynamic_aggregation) and (
        normalized_judge_followup_mode == "plan"
    )
    effective_adaptive_evidence = bool(adaptive_evidence) and (
        normalized_judge_followup_mode == "plan"
    )
    tx_context = {
        **(tx_context or {}),
        "tx_hash": tx_hash,
        "chain": chain,
    }
    runtime_policy_identity = build_runtime_policy_identity({
        "llm_provider": llm_provider,
        "judge_model": judge_model,
        "judge_max_tokens": judge_max_tokens,
        "aggregator_max_tokens": aggregator_max_tokens,
        "judge_thinking": judge_thinking,
        "aggregator_thinking": aggregator_thinking,
        "adaptive_model": adaptive_model,
        "adaptive_provider": adaptive_provider,
        "enable_source_tools": enable_source_tools,
        "max_view_chars": max_view_chars,
        "max_context_chars": max_context_chars,
        "runtime_early_stop": runtime_early_stop,
        "runtime_early_stop_policy": runtime_early_stop_policy,
        "adaptive_evidence": adaptive_evidence,
        "adaptive_evidence_mode": adaptive_evidence_mode,
        "adaptive_evidence_max_direct_trace_nodes": (
            adaptive_evidence_max_direct_trace_nodes
        ),
        "adaptive_evidence_max_direct_trace_chars": (
            adaptive_evidence_max_direct_trace_chars
        ),
        "adaptive_evidence_max_medium_trace_nodes": (
            adaptive_evidence_max_medium_trace_nodes
        ),
        "parallel_judge": parallel_judge,
        "judge_concurrency": judge_concurrency,
        "iv_stateful_runtime": iv_stateful_runtime,
        "access_control_binding_mode": access_control_binding_mode,
        "reentrancy_binding_mode": reentrancy_binding_mode,
        "disable_stateful_bindings": disable_stateful_bindings,
        "dynamic_aggregation": dynamic_aggregation,
        "followup_context_mode": followup_context_mode,
        "judge_followup_mode": normalized_judge_followup_mode,
        "judge_expanded_followup_views": judge_expanded_followup_views,
    })

    print(
        f"[RunInference] Packet runtime for tx={tx_hash}, "
        f"rule={rule.rule_id} v{rule.version}"
    )

    # 1. Generate or load plan
    resolved_planner_model, resolved_planner_provider = resolve_adaptive_llm_route(
        model=planner_model,
        provider=llm_provider,
        thinking=planner_thinking,
        adaptive_model=adaptive_model,
        adaptive_provider=adaptive_provider,
    )
    resolved_judge_model, resolved_judge_provider = resolve_adaptive_llm_route(
        model=judge_model,
        provider=llm_provider,
        thinking=judge_thinking,
        adaptive_model=adaptive_model,
        adaptive_provider=adaptive_provider,
    )
    resolved_aggregator_model, resolved_aggregator_provider = (
        resolve_adaptive_llm_route(
            model=judge_model,
            provider=llm_provider,
            thinking=aggregator_thinking,
            adaptive_model=adaptive_model,
            adaptive_provider=adaptive_provider,
        )
    )
    planner_llm = (
        OpenAICompatibleLLM(
            model=resolved_planner_model,
            provider=resolved_planner_provider,
            minimax_thinking=planner_thinking,
        )
        if resolved_planner_model
        else None
    )
    judge_llm = (
        OpenAICompatibleLLM(
            model=resolved_judge_model,
            provider=resolved_judge_provider,
            max_tokens=max(1, int(judge_max_tokens or 8192)),
            minimax_thinking=judge_thinking,
        )
        if resolved_judge_model
        else None
    )
    aggregator_llm = (
        OpenAICompatibleLLM(
            model=resolved_aggregator_model,
            provider=resolved_aggregator_provider,
            max_tokens=max(1, int(aggregator_max_tokens or 8192)),
            minimax_thinking=aggregator_thinking,
        )
        if resolved_aggregator_model and effective_dynamic_aggregation
        else None
    )
    effective_judge_max_tokens = int(
        getattr(judge_llm, "max_tokens", 0)
        or max(1, int(judge_max_tokens or 8192))
    )
    effective_aggregator_max_tokens = int(
        getattr(aggregator_llm, "max_tokens", 0)
        or max(1, int(aggregator_max_tokens or 8192))
    )
    print(
        f"[RunInference] LLM: planner={'yes' if planner_llm else 'no'} "
        f"judge={'yes' if judge_llm else 'no'} "
        f"source_tools={'yes' if enable_source_tools else 'no'} "
        f"max_view_chars={max_view_chars} max_context_chars={max_context_chars} "
        f"adaptive_evidence={'yes' if adaptive_evidence else 'no'} "
        f"adaptive_mode={adaptive_evidence_mode} "
        f"enforce_label_plan_template={'yes' if enforce_label_plan_template else 'no'} "
        f"parallel_judge={'yes' if parallel_judge else 'no'} "
        f"judge_concurrency={judge_concurrency} "
        f"judge_max_tokens={effective_judge_max_tokens} "
        f"aggregator_max_tokens={effective_aggregator_max_tokens} "
        f"judge_thinking={getattr(judge_llm, 'minimax_thinking', '') or 'provider_default'} "
        f"aggregator_thinking={getattr(aggregator_llm, 'minimax_thinking', '') or 'disabled'} "
        f"iv_stateful_runtime={'yes' if iv_stateful_runtime else 'no'} "
        f"access_control_binding_mode={access_control_binding_mode} "
        f"reentrancy_binding_mode={reentrancy_binding_mode} "
        f"disable_stateful_bindings={'yes' if disable_stateful_bindings else 'no'} "
        f"dynamic_aggregation_requested={'yes' if dynamic_aggregation else 'no'} "
        f"dynamic_aggregation_effective={'yes' if effective_dynamic_aggregation else 'no'} "
        f"followup_context_mode={followup_context_mode} "
        f"judge_followup_mode={normalized_judge_followup_mode} "
        f"judge_expanded_followup_views={judge_expanded_followup_views} "
        f"rate_limit_serial_fallback={'yes' if rate_limit_serial_fallback else 'no'} "
        f"rate_limit_retries={rate_limit_retry_attempts} "
        f"default_provider={llm_provider or '(auto)'} "
        f"adaptive_provider={adaptive_provider or '(inherit)'}"
    )
    resolved_llm_configs = {}
    if planner_llm is not None:
        resolved_llm_configs["planner"] = planner_llm.config_summary()
    if judge_llm is not None:
        resolved_llm_configs["judge"] = judge_llm.config_summary()
    if aggregator_llm is not None:
        resolved_llm_configs["aggregator"] = aggregator_llm.config_summary()
    if resolved_llm_configs:
        print(
            "[RunInference] Resolved LLM config: "
            f"{stable_json_dumps(resolved_llm_configs)}"
        )

    artifact_plan_fingerprint = ""
    if fixed_plan is not None:
        plan = _clone_plan(fixed_plan)
        artifact_plan_fingerprint = semantic_plan_fingerprint(plan)
        plan.metadata.setdefault("fixed_plan", True)
        if plan_source:
            plan.metadata.setdefault("plan_source", plan_source)
        apply_reentrancy_stateful_runtime(
            plan,
            rule=rule,
            mode=reentrancy_binding_mode,
        )
        PlanValidator().validate(
            plan,
            attack_label=str((rule.metadata or {}).get("attack_label") or ""),
        )
        print(
            f"[RunInference] Using fixed plan {plan.plan_id}: "
            f"{len(plan.judge_steps)} judge steps source={plan_source or '(in-memory)'}"
        )
    else:
        planner = PlanGenerator(
            llm=planner_llm,
            enforce_label_plan_template=enforce_label_plan_template,
        )
        plan = planner.generate(rule, tx_context)
        try:
            PlanValidator().validate(
                plan,
                attack_label=str((rule.metadata or {}).get("attack_label") or ""),
            )
        except PlanValidationError as exc:
            print(
                f"[RunInference] Generated plan is invalid ({exc}); "
                "falling back to baseline packet plan."
            )
            plan = compile_rule_to_baseline_plan(rule)
            PlanValidator().validate(
                plan,
                attack_label=str((rule.metadata or {}).get("attack_label") or ""),
            )
        print(
            f"[RunInference] Generated plan {plan.plan_id}: "
            f"{len(plan.judge_steps)} judge steps"
        )
        artifact_plan_fingerprint = semantic_plan_fingerprint(plan)
    plan.metadata.setdefault("plan_template_policy", {
        "enforce_label_plan_template": bool(enforce_label_plan_template),
        "stage": "run_inference",
        "label": normalize_attack_label((rule.metadata or {}).get("attack_label", ""), default=""),
    })
    if iv_stateful_runtime:
        apply_insufficient_validation_stateful_runtime(
            plan,
            attack_label=str((rule.metadata or {}).get("attack_label") or ""),
        )
    apply_access_control_binding_mode(
        plan,
        rule=rule,
        mode=access_control_binding_mode,
    )
    apply_reentrancy_stateful_runtime(
        plan,
        rule=rule,
        mode=reentrancy_binding_mode,
    )
    if disable_stateful_bindings:
        plan = disable_stateful_bindings_in_plan(plan)
        print(
            "[RunInference] Stateful binding ablation enabled: "
            "depends_on/state keys/state_output schema cleared for runtime."
        )
    PlanValidator().validate(
        plan,
        attack_label=str((rule.metadata or {}).get("attack_label") or ""),
    )
    expected_effective_plan = prepare_effective_runtime_plan(
        rule,
        plan,
        iv_stateful_runtime=iv_stateful_runtime,
        access_control_binding_mode=access_control_binding_mode,
        reentrancy_binding_mode=reentrancy_binding_mode,
        disable_stateful_bindings=disable_stateful_bindings,
        judge_followup_mode=normalized_judge_followup_mode,
        judge_expanded_followup_views=judge_expanded_followup_views,
    )
    expected_effective_plan_fingerprint = semantic_plan_fingerprint(
        expected_effective_plan
    )
    binding_policy = dict(
        (plan.metadata or {}).get("access_control_binding_policy") or {}
    )
    if binding_policy:
        print(
            "[RunInference] Access Control binding: "
            f"requested={binding_policy.get('requested_mode')} "
            f"effective={binding_policy.get('effective_mode')} "
            f"roles={binding_policy.get('role_steps', {})}"
        )
        if str(binding_policy.get("effective_mode") or "") == "prompt_fallback":
            print(
                "[RunInference][WARNING] Access Control stateful binding "
                "fell back to prompt-only mode; downstream judges will not "
                "share access_control_candidate state. "
                f"reason={binding_policy.get('fallback_reason', '')} "
                f"scores={stable_json_dumps(binding_policy.get('role_score_matrix', {}))}"
            )

    # 2. Execute via PacketRuntime
    source_registry = (
        SourceToolRegistry(
            cache_dir=source_cache_dir,
            force_refresh=force_refresh_source,
        )
        if enable_source_tools
        else None
    )
    judge = JudgeModel(
        llm=judge_llm,
        max_view_chars=max_view_chars,
        max_context_chars=max_context_chars,
        source_tool_registry=source_registry,
        enable_source_followup=enable_source_tools,
        transcript_recorder=(
            LLMTranscriptRecorder(llm_transcripts_dir)
            if llm_transcripts_dir
            else None
        ),
        followup_context_mode=followup_context_mode,
    )
    runtime = PacketRuntime(
        judge_model=judge,
        aggregator_llm=aggregator_llm,
        base_dir=base_cache_dir,
        transcript_phase=transcript_phase,
    )
    judge_reuse_cache = build_judge_reuse_cache(
        reuse_result,
        judge_model=str(getattr(judge_llm, "model", "") or ""),
        judge_provider=str(getattr(judge_llm, "provider", "") or ""),
        max_view_chars=max_view_chars,
        max_context_chars=max_context_chars,
        source_followup_enabled=enable_source_tools,
        judge_max_tokens=effective_judge_max_tokens,
        judge_thinking=str(getattr(judge_llm, "minimax_thinking", "") or ""),
        followup_context_mode=followup_context_mode,
        judge_followup_mode=normalized_judge_followup_mode,
        adaptive_evidence_enabled=effective_adaptive_evidence,
        adaptive_evidence_mode=(
            adaptive_evidence_mode if effective_adaptive_evidence else "off"
        ),
    )
    if judge_reuse_cache:
        print(
            f"[RunInference] Judge reuse cache prepared: "
            f"{len(judge_reuse_cache)} reusable step(s)"
        )
    frozen_packet = _frozen_packet_from_result(reuse_result)
    if frozen_packet is not None:
        print("[RunInference] Replaying frozen packet from baseline result")
    elif reuse_result is not None and freeze_packet:
        print(
            "[RunInference] Baseline result has no packet snapshot; "
            "building a new full packet for backward compatibility"
        )
    packet_result = runtime.execute(
        tx_hash=tx_hash,
        rule=rule,
        plan=plan,
        force_rebuild_packet=force_rebuild_packet,
        packet_override=frozen_packet,
        freeze_packet=freeze_packet,
        judge_reuse_cache=judge_reuse_cache,
        early_stop=runtime_early_stop,
        early_stop_policy=runtime_early_stop_policy,
        adaptive_evidence=adaptive_evidence,
        adaptive_evidence_mode=adaptive_evidence_mode,
        adaptive_evidence_max_direct_trace_nodes=adaptive_evidence_max_direct_trace_nodes,
        adaptive_evidence_max_direct_trace_chars=adaptive_evidence_max_direct_trace_chars,
        adaptive_evidence_max_medium_trace_nodes=adaptive_evidence_max_medium_trace_nodes,
        adaptive_evidence_debug=adaptive_evidence_debug,
        parallel_judge=parallel_judge,
        judge_concurrency=judge_concurrency,
        structured_judge_logs=structured_judge_logs,
        iv_stateful_runtime=iv_stateful_runtime,
        access_control_binding_mode=access_control_binding_mode,
        reentrancy_binding_mode=reentrancy_binding_mode,
        disable_stateful_bindings=disable_stateful_bindings,
        dynamic_aggregation=dynamic_aggregation,
        judge_followup_mode=normalized_judge_followup_mode,
        judge_expanded_followup_views=judge_expanded_followup_views,
        rate_limit_serial_fallback=rate_limit_serial_fallback,
        rate_limit_retry_attempts=rate_limit_retry_attempts,
        rate_limit_retry_delay_seconds=rate_limit_retry_delay_seconds,
    )

    # 3. Use the exact packet seen by Judge. Rebuilding here can observe
    # mutable fundflow/profit artifacts and silently change candidate inputs.
    packet = packet_result.pop("_packet_snapshot", {})
    effective_plan = packet_result.pop("_effective_plan", plan.to_dict())
    effective_runtime_plan_fingerprint = semantic_plan_fingerprint(effective_plan)
    plan_identity = {
        "schema_version": "evotx.runtime_plan_identity.v1",
        "artifact_plan_fingerprint": artifact_plan_fingerprint,
        "expected_effective_runtime_plan_fingerprint": (
            expected_effective_plan_fingerprint
        ),
        "effective_runtime_plan_fingerprint": effective_runtime_plan_fingerprint,
        "identity_match": (
            artifact_plan_fingerprint == effective_runtime_plan_fingerprint
        ),
        "runtime_preparation_match": (
            expected_effective_plan_fingerprint
            == effective_runtime_plan_fingerprint
        ),
    }
    runtime_trace = packet_result.setdefault("runtime_trace", {})
    runtime_trace["plan_identity"] = plan_identity
    runtime_trace["runtime_policy_identity"] = str(
        runtime_policy_identity.get("fingerprint") or ""
    )
    runtime_trace["runtime_policy"] = dict(
        runtime_policy_identity.get("payload") or {}
    )

    # 4. Convert to standard result format (compatible with all downstream)
    runtime_result = _packet_result_to_runtime_result(
        packet_result, effective_plan, packet
    )

    return build_result_record(
        tx_hash=tx_hash,
        chain=chain,
        rule=rule,
        rule_source=rule_source,
        plan_source=plan_source or plan.metadata.get("plan_source"),
        runtime_result=runtime_result,
        ground_truth=ground_truth,
        raw_ground_truth=raw_ground_truth,
        label_rationale=label_rationale,
        report=report,
        case_metadata=case_metadata,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_rule_input(
    rule_file: Optional[str],
    label: Optional[str],
    rules_dir: str,
) -> tuple[EvolvingRule, str]:
    if rule_file:
        rule = EvolvingRule.from_dict(read_json(rule_file))
        if rule.metadata and rule.metadata.get("attack_label"):
            raw_label = rule.metadata.get("raw_attack_label") or rule.metadata.get("attack_label")
            rule.metadata.setdefault("raw_attack_label", raw_label)
            rule.metadata.setdefault("display_label", raw_label)
            rule.metadata["attack_label"] = normalize_attack_label(raw_label)
        return rule, rule_file

    if label and label.strip():
        canonical_label = normalize_attack_label(label)
        store = RuleStore(rules_dir)
        path = store.find_latest_by_attack_label(canonical_label)
        if path is None:
            raise FileNotFoundError(
                f"No latest rule found for label '{label}' under {rules_dir}. "
                "Provide --rule-file or create a cold-start/few-shot rule first."
            )
        rule = store.load_by_attack_label(canonical_label)
        if rule.metadata:
            raw_label = rule.metadata.get("raw_attack_label") or rule.metadata.get("attack_label")
            rule.metadata.setdefault("raw_attack_label", raw_label)
            rule.metadata.setdefault("display_label", raw_label)
            rule.metadata["attack_label"] = normalize_attack_label(raw_label)
        return rule, str(path)

    raise ValueError("Provide either --rule-file or --label.")


def load_plan_input(plan_file: str | None) -> EvidencePlan | None:
    if not plan_file:
        return None
    return PlanStore(Path(plan_file).parent).load(plan_file)


def _clone_plan(plan: EvidencePlan | Dict[str, Any]) -> EvidencePlan:
    if isinstance(plan, EvidencePlan):
        return EvidencePlan.from_dict(copy.deepcopy(plan.to_dict()))
    return EvidencePlan.from_dict(copy.deepcopy(plan))


def resolve_inference_models(
    llm_model: Optional[str],
    planner_model: Optional[str],
    judge_model: Optional[str],
    env_model: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    default_model = llm_model or planner_model or judge_model or env_model
    resolved_planner = planner_model or default_model
    resolved_judge = judge_model or resolved_planner or default_model
    resolved_env = env_model or resolved_planner or resolved_judge or default_model
    return resolved_planner, resolved_judge, resolved_env


if __name__ == "__main__":
    main()
