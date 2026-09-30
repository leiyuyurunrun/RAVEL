from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.adapters.llm_adapter import describe_llm_config
from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import EvolvingRule
from evotx.storage.plan_store import PlanStore
from evotx.evolution.regression import RegressionEvaluator
from evotx.utils.case_utils import load_cases_from_split_csvs, load_labeled_cases
from evotx.utils.csv_archive_utils import archive_split_csvs
from evotx.utils.json_utils import read_json, stable_json_dumps, write_json
from evotx.utils.logging_utils import configure_session_logging
from evotx.utils.result_slimmer import build_slim_result
from experiments.run_inference import run_single_inference


EVAL_RESULTS_FILENAME = "eval_results.json"
EVAL_SLIM_RESULTS_FILENAME = "eval_slim_results.json"
EVAL_SUMMARY_FILENAME = "eval_summary.json"
EVAL_RUN_MANIFEST_FILENAME = "eval_run_manifest.json"
FAILED_CSV_FILENAME = "failed.csv"
FAILED_CSV_COLUMNS = ["HackId", "txHash", "Chain", "Type", "Cause"]


def main() -> None:
    run_started_at = _now_iso()
    run_started_perf = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Stage 3: evaluate a fixed EvoTx rule on held-out transactions."
    )
    parser.add_argument(
        "--results",
        help="Existing result JSON file, JSON list, or directory of result JSON files.",
    )
    parser.add_argument(
        "--cases-file",
        help="Held-out labeled cases in JSON or CSV. Used only when running fresh evaluation.",
    )
    parser.add_argument("--pos-csv", "--pos_csv", dest="pos_csv")
    parser.add_argument("--neg-csv", dest="neg_csv")
    parser.add_argument(
        "--only-head",
        "--only_head",
        dest="only_head",
        type=int,
        help=(
            "Evaluate only the first K rows from each split CSV input "
            "(first K positive rows and first K negative rows). With --cases-file, "
            "uses the first K total cases."
        ),
    )
    parser.add_argument(
        "--benign-csv",
        help="Legacy strict-benign negative CSV. Prefer --neg-csv for mixed non-target negatives.",
    )
    parser.add_argument(
        "--malicious-csv",
        help="Legacy positive CSV. Prefer --pos-csv.",
    )
    parser.add_argument(
        "--label",
        help="Positive target label name used for --pos-csv rows, e.g. price_manipulation.",
    )
    parser.add_argument(
        "--positive-label",
        help="Optional positive label name for the current attack family, e.g. price_manipulation.",
    )
    parser.add_argument("--rule-file")
    parser.add_argument(
        "--plan-file",
        help="Optional fixed EvidencePlan JSON. When set, planner generation is skipped.",
    )
    parser.add_argument("--tool-manifest", default="configs/tool_manifest.json")
    parser.add_argument("--llm-model")
    parser.add_argument(
        "--llm-provider",
        help="Optional provider override for evaluation LLMs: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument("--planner-model")
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
    parser.add_argument("--env-model")
    parser.add_argument("--use-environment", action="store_true")
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
    parser.add_argument(
        "--force-refresh-source",
        action="store_true",
        help="Refresh explorer source cache for read_function_chunk.",
    )
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
        help="Rebuild compact packets from source cache during evaluation.",
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
        help="Keep adaptive evidence routing details in runtime traces/debug artifacts.",
    )
    parser.add_argument(
        "--parallel-judge",
        action="store_true",
        default=False,
        help="Run independent judge steps concurrently within each transaction.",
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
            "Choose prompt-only Access Control consistency or strong stateful "
            "authorization-chain binding."
        ),
    )
    parser.add_argument(
        "--reentrancy-binding-mode",
        choices=["disabled", "soft", "stateful"],
        default="soft",
        help=(
            "Disable Reentrancy candidate binding, preserve candidate state "
            "without answer/emit overrides (soft), or enable strong stateful "
            "candidate normalization."
        ),
    )
    parser.add_argument(
        "--disable-stateful-bindings",
        action="store_true",
        default=False,
        help=(
            "Ablation mode: ignore depends_on, state keys, state prompt roles, "
            "state schemas, and judge state_output during runtime."
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
            "Judge follow-up context policy. unified shares one context budget "
            "and carries a compact previous-judge summary."
        ),
    )
    parser.add_argument(
        "--judge-followup-mode",
        choices=["plan", "disabled", "expanded"],
        default="plan",
        help=(
            "Judge follow-up ablation. disabled and expanded force one Judge "
            "call and disable near-miss, adaptive evidence, and dynamic aggregation."
        ),
    )
    parser.add_argument(
        "--judge-expanded-followup-views",
        type=int,
        default=2,
        help="Maximum follow-up views promoted per Judge in expanded mode.",
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
        help="Initial 429 retry delay; later retries use exponential backoff.",
    )
    parser.add_argument(
        "--save-llm-transcripts",
        action="store_true",
        help="Save judge prompts, raw completions, usage, render metadata, and tool observations.",
    )
    parser.add_argument(
        "--llm-transcripts-dir",
        help=(
            "Directory used when --save-llm-transcripts is enabled. If omitted, "
            "writes under the episode result directory when --episode is set, "
            "otherwise under data/results/llm_transcripts."
        ),
    )
    parser.add_argument("--logs-dir", default="data/log")
    parser.add_argument(
        "--results-out",
        default=f"data/results/{EVAL_RESULTS_FILENAME}",
        help=(
            "Evaluation result output file, or a directory root. If a directory "
            f"is given, writes {EVAL_RESULTS_FILENAME} under it."
        ),
    )
    parser.add_argument(
        "--out",
        default="data/results/evaluation_summary.json",
        help=(
            "Evaluation summary output file, or a directory root. If a directory "
            f"is given, writes {EVAL_SUMMARY_FILENAME} under it."
        ),
    )
    parser.add_argument(
        "--slim-results-out",
        help=(
            "Optional slim evaluation result output file, or a directory root. "
            f"If omitted, writes a sibling {EVAL_SLIM_RESULTS_FILENAME}."
        ),
    )
    parser.add_argument(
        "--failed-csv-out",
        help=(
            "Optional failed-case CSV output file, or a directory root. "
            f"If omitted, writes a sibling {FAILED_CSV_FILENAME}."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        default="data/csv",
        help="Root directory for episode-scoped CSV input archives.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        help=(
            "Optional experiment episode id. When set, evaluation logs and "
            "result files are nested under the episode directory."
        ),
    )
    parser.add_argument(
        "--run-variant",
        default="",
        help=(
            "Optional filesystem-safe evaluation output namespace nested below "
            "the episode. The canonical evolution variant writes directly to "
            "the episode directory; other variants such as wo_evolution_v1 use "
            "their own subdirectory. This does not select or relocate Rule/Plan "
            "artifacts."
        ),
    )
    args = parser.parse_args()
    if args.only_head is not None and args.only_head < 0:
        raise ValueError("--only-head must be non-negative.")
    apply_episode_paths(args)
    log_path = configure_session_logging(
        "evaluate",
        logs_dir=args.logs_dir,
        suffix=args.label or args.positive_label or "heldout",
    )
    print(f"[Evaluate] Started at: {run_started_at}")
    print(
        "[Evaluate] Run scope: "
        f"episode={args.episode if args.episode is not None else 'none'} "
        f"variant={args.run_variant or 'default'}"
    )

    if args.results:
        results = load_results(args.results)
        resolved_positive_label = args.positive_label or "attack"
        print(f"[Evaluate] Loading precomputed results from {args.results}")
    else:
        has_case_input = bool(args.cases_file) or bool(args.pos_csv or args.malicious_csv) or bool(args.neg_csv or args.benign_csv)
        if not has_case_input or not args.rule_file:
            raise ValueError(
                "Provide either --results or a rule file plus case input "
                "(--cases-file or --pos-csv/--neg-csv/--label)."
            )
        results, resolved_positive_label = run_evaluation(args)
        write_json(args.results_out, results)

    resolved_positive_label = normalize_attack_label(resolved_positive_label)
    slim_results = [build_slim_result(result) for result in results]
    write_json(args.slim_results_out, slim_results)

    summary = RegressionEvaluator.summarize(results).to_dict()
    summary = enrich_evaluation_summary(
        summary,
        slim_results,
        target_label=args.label or resolved_positive_label,
    )
    failed_csv_count = write_failed_csv(args.failed_csv_out, results, slim_results)
    summary["failed_csv"] = args.failed_csv_out
    summary["failed_csv_count"] = failed_csv_count
    if args.episode is not None:
        summary["episode"] = args.episode
    run_finished_at = _now_iso()
    run_elapsed_seconds = round(time.perf_counter() - run_started_perf, 3)
    summary["run_timing"] = {
        "started_at": run_started_at,
        "finished_at": run_finished_at,
        "elapsed_seconds": run_elapsed_seconds,
        "elapsed_hms": _format_elapsed_hms(run_elapsed_seconds),
    }
    write_json(args.out, summary)

    print("Stage 3 complete.")
    print(f"Episode: {args.episode if args.episode is not None else 'none'}")
    print(f"Positive label: {resolved_positive_label}")
    print(f"Evaluated results: {summary['total']}")
    print(f"Summary: {stable_json_dumps(summary)}")
    print(f"Started at: {summary['run_timing']['started_at']}")
    print(f"Finished at: {summary['run_timing']['finished_at']}")
    print(
        "Total elapsed: "
        f"{summary['run_timing']['elapsed_hms']} "
        f"({summary['run_timing']['elapsed_seconds']}s)"
    )
    if not args.results:
        print(f"Saved evaluation results: {args.results_out}")
    print(f"Saved slim evaluation results: {args.slim_results_out}")
    print(f"Saved failed cases CSV: {args.failed_csv_out} ({failed_csv_count} rows)")
    print(f"Saved summary: {args.out}")
    if args.save_llm_transcripts:
        print(f"LLM transcripts: {args.llm_transcripts_dir}")
    print(f"Session log: {log_path}")


def run_evaluation(args) -> tuple[List[Dict[str, Any]], str]:
    rule = EvolvingRule.from_dict(read_json(args.rule_file))
    fixed_plan = PlanStore(Path(args.plan_file).parent).load(args.plan_file) if args.plan_file else None
    tool_manifest = read_json(args.tool_manifest, default={}) or {}
    cases, resolved_positive_label = load_cases_input(args)
    csv_archive = archive_split_csvs(
        stage="eval",
        benign_csv=args.benign_csv,
        malicious_csv=args.malicious_csv,
        pos_csv=args.pos_csv,
        neg_csv=args.neg_csv,
        label=resolved_positive_label,
        episode=args.episode,
        csv_root=args.csv_dir,
    )
    run_manifest_path = _derive_run_manifest_path(args.results_out)
    write_json(
        run_manifest_path,
        build_eval_run_manifest(
            args,
            resolved_positive_label=resolved_positive_label,
            csv_archive=csv_archive,
        ),
    )
    print(
        f"[Evaluate] Running fresh evaluation: label={resolved_positive_label} "
        f"cases={len(cases)} runtime=packet_runtime "
        f"force_rebuild_packet={args.force_rebuild_packet} "
        f"source_tools={args.enable_source_tools} "
        f"fixed_plan={'yes' if fixed_plan else 'no'} "
        f"only_head={args.only_head if args.only_head is not None else 'none'} "
        f"max_view_chars={args.max_view_chars} max_context_chars={args.max_context_chars} "
        f"adaptive_evidence={args.adaptive_evidence} "
        f"adaptive_mode={args.adaptive_evidence_mode} "
        f"parallel_judge={args.parallel_judge} "
        f"judge_concurrency={args.judge_concurrency} "
        f"judge_max_tokens={args.judge_max_tokens} "
        f"aggregator_max_tokens={args.aggregator_max_tokens} "
        f"judge_thinking={args.judge_thinking} "
        f"aggregator_thinking={args.aggregator_thinking} "
        f"iv_stateful_runtime={args.iv_stateful_runtime} "
        f"access_control_binding_mode={args.access_control_binding_mode} "
        f"reentrancy_binding_mode={args.reentrancy_binding_mode} "
        f"dynamic_aggregation={args.dynamic_aggregation} "
        f"followup_context_mode={args.followup_context_mode} "
        f"judge_followup_mode={args.judge_followup_mode} "
        f"judge_expanded_followup_views={args.judge_expanded_followup_views} "
        f"rate_limit_serial_fallback={args.rate_limit_serial_fallback} "
        f"rate_limit_retries={args.rate_limit_retry_attempts}"
    )
    if csv_archive:
        print(f"[Evaluate] CSV archive: {stable_json_dumps(csv_archive)}")
    print(f"[Evaluate] Run manifest: {run_manifest_path}")
    print(
        "[Evaluate] Models: "
        f"planner={args.planner_model or args.llm_model or 'disabled'} "
        f"judge={args.judge_model or args.llm_model or args.planner_model or 'disabled'} "
        f"env={args.env_model or args.llm_model or args.planner_model or args.judge_model or 'disabled'} "
        f"provider={args.llm_provider or '(auto)'}"
    )
    resolved_llm_configs = {
        "planner": describe_llm_config(
            model=args.planner_model or args.llm_model,
            provider=args.llm_provider,
        ),
        "judge": describe_llm_config(
            model=args.judge_model or args.llm_model or args.planner_model,
            provider=args.llm_provider,
            max_tokens=args.judge_max_tokens,
            minimax_thinking=args.judge_thinking,
        ),
        "aggregator": describe_llm_config(
            model=args.judge_model or args.llm_model or args.planner_model,
            provider=args.llm_provider,
            max_tokens=args.aggregator_max_tokens,
            minimax_thinking=args.aggregator_thinking,
        ),
        "env": describe_llm_config(
            model=args.env_model or args.llm_model or args.planner_model or args.judge_model,
            provider=args.llm_provider,
        ),
    }
    print(
        "[Evaluate] Resolved LLM configs: "
        f"{stable_json_dumps(resolved_llm_configs)}"
    )

    results: List[Dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        print(
            f"[Evaluate] Case {index}/{total}: tx={case['tx_hash']} chain={case['chain']} "
            f"ground_truth={case.get('ground_truth')} raw_label={case.get('raw_ground_truth')}"
        )
        result = run_single_inference(
            tx_hash=case["tx_hash"],
            chain=case["chain"],
            rule=rule,
            rule_source=args.rule_file,
            tool_manifest=tool_manifest,
            tx_context=case.get("tx_context", {}) or {},
            static_evidence=case.get("static_evidence", {}) or {},
            llm_provider=args.llm_provider,
            planner_model=args.planner_model or args.llm_model,
            judge_model=args.judge_model or args.llm_model,
            judge_max_tokens=args.judge_max_tokens,
            aggregator_max_tokens=args.aggregator_max_tokens,
            judge_thinking=args.judge_thinking,
            aggregator_thinking=args.aggregator_thinking,
            fixed_plan=fixed_plan,
            plan_source=args.plan_file,
            env_model=args.env_model,
            use_environment=args.use_environment,
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
            ground_truth=case.get("ground_truth"),
            raw_ground_truth=case.get("raw_ground_truth"),
            label_rationale=case.get("label_rationale", ""),
            report=case.get("report", ""),
            case_metadata=case.get("metadata", {}),
            llm_transcripts_dir=args.llm_transcripts_dir,
            transcript_phase="evaluate",
        )
        results.append(result)
    return results, resolved_positive_label


def apply_episode_paths(args) -> None:
    episode = None
    if args.episode is not None:
        if args.episode < 0:
            raise ValueError("--episode must be a non-negative integer.")
        episode = str(args.episode)
        args.logs_dir = str(_append_episode_dir(args.logs_dir, episode))
    run_variant = _normalize_run_variant(
        getattr(args, "run_variant", "")
    )
    args.run_variant = run_variant
    output_variant = "" if run_variant.lower() == "evolution" else run_variant
    if output_variant:
        args.logs_dir = str(_append_episode_dir(args.logs_dir, output_variant))
    args.results_out = str(
        _resolve_output_path(args.results_out, episode, EVAL_RESULTS_FILENAME)
    )
    args.out = str(
        _resolve_output_path(args.out, episode, EVAL_SUMMARY_FILENAME)
    )
    if args.slim_results_out:
        args.slim_results_out = str(
            _resolve_output_path(
                args.slim_results_out,
                episode,
                EVAL_SLIM_RESULTS_FILENAME,
            )
        )
    else:
        args.slim_results_out = str(_derive_slim_results_path(args.results_out))
    if args.failed_csv_out:
        args.failed_csv_out = str(
            _resolve_output_path(args.failed_csv_out, episode, FAILED_CSV_FILENAME)
        )
    else:
        args.failed_csv_out = str(Path(args.out).with_name(FAILED_CSV_FILENAME))
    if output_variant:
        args.results_out = str(
            _insert_variant_before_file(args.results_out, output_variant)
        )
        args.out = str(_insert_variant_before_file(args.out, output_variant))
        args.slim_results_out = str(
            _insert_variant_before_file(args.slim_results_out, output_variant)
        )
        args.failed_csv_out = str(
            _insert_variant_before_file(args.failed_csv_out, output_variant)
        )
    if getattr(args, "save_llm_transcripts", False):
        if args.llm_transcripts_dir:
            transcript_dir = (
                _append_episode_dir(args.llm_transcripts_dir, episode)
                if episode
                else Path(args.llm_transcripts_dir)
            )
            if output_variant:
                transcript_dir = _append_episode_dir(
                    str(transcript_dir), output_variant
                )
            args.llm_transcripts_dir = str(transcript_dir)
        else:
            result_dir = Path(args.out).parent if args.out else Path("data/results")
            args.llm_transcripts_dir = str(result_dir / "llm_transcripts")
        Path(args.llm_transcripts_dir).mkdir(parents=True, exist_ok=True)
    else:
        args.llm_transcripts_dir = None


def _append_episode_dir(path_like: str, episode: str) -> Path:
    path = Path(path_like)
    if path.name == episode:
        return path
    return path / episode


def _insert_episode_before_file(path_like: str, episode: str) -> Path:
    path = Path(path_like)
    if path.parent.name == episode:
        return path
    return path.parent / episode / path.name


def _insert_variant_before_file(path_like: str, variant: str) -> Path:
    path = Path(path_like)
    if path.parent.name == variant:
        return path
    return path.parent / variant / path.name


def _normalize_run_variant(value: Any) -> str:
    variant = str(value or "").strip()
    if not variant:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", variant):
        raise ValueError(
            "--run-variant must be a single filesystem-safe name using only "
            "letters, digits, '.', '_' or '-'."
        )
    return variant


def _resolve_output_path(
    path_like: str,
    episode: str | None,
    default_filename: str,
) -> Path:
    path = Path(path_like)
    if path.suffix:
        if episode is None:
            return path
        return _insert_episode_before_file(str(path), episode)

    if episode is None:
        return path / default_filename
    if path.name == episode:
        return path / default_filename
    return path / episode / default_filename


def _derive_slim_results_path(results_out: str) -> Path:
    path = Path(results_out)
    if path.name == EVAL_RESULTS_FILENAME:
        return path.with_name(EVAL_SLIM_RESULTS_FILENAME)
    if path.suffix:
        stem = path.stem
        if stem.endswith("_results"):
            stem = stem[: -len("_results")]
        return path.with_name(f"{stem}_slim_results{path.suffix}")
    return path / EVAL_SLIM_RESULTS_FILENAME


def _derive_run_manifest_path(results_out: str) -> Path:
    path = Path(results_out)
    if path.suffix.lower() == ".json":
        if path.name == EVAL_RESULTS_FILENAME:
            return path.with_name(EVAL_RUN_MANIFEST_FILENAME)
        stem = path.stem
        if stem.endswith("_results"):
            stem = stem[: -len("_results")]
        return path.with_name(f"{stem}_run_manifest{path.suffix}")
    return path / EVAL_RUN_MANIFEST_FILENAME


def build_eval_run_manifest(
    args: Any,
    *,
    resolved_positive_label: str,
    csv_archive: Dict[str, str],
) -> Dict[str, Any]:
    resolved_judge_config = describe_llm_config(
        model=args.judge_model or args.llm_model or args.planner_model,
        provider=args.llm_provider,
        max_tokens=args.judge_max_tokens,
        minimax_thinking=args.judge_thinking,
    )
    positive_csv = (
        csv_archive.get("pos_csv")
        or args.pos_csv
        or args.malicious_csv
        or ""
    )
    negative_csv = (
        csv_archive.get("neg_csv")
        or args.neg_csv
        or args.benign_csv
        or ""
    )
    return {
        "schema_version": "evotx.eval_run_manifest.v1",
        "created_at": _now_iso(),
        "episode": args.episode,
        "run_variant": str(getattr(args, "run_variant", "") or ""),
        "label": normalize_attack_label(
            args.label or resolved_positive_label
        ),
        "positive_label": args.positive_label or resolved_positive_label,
        "inputs": {
            "pos_csv": str(positive_csv),
            "neg_csv": str(negative_csv),
            "negative_source_kind": csv_archive.get(
                "negative_source_kind",
                "mixed_non_target" if args.neg_csv else "benign",
            ),
        },
        "artifacts": {
            "rule_file": str(args.rule_file or ""),
            "plan_file": str(args.plan_file or ""),
            "tool_manifest": str(args.tool_manifest or ""),
        },
        "models": {
            "llm_model": str(args.llm_model or ""),
            "llm_provider": str(args.llm_provider or ""),
            "planner_model": str(args.planner_model or ""),
            "judge_model": str(args.judge_model or ""),
            "env_model": str(args.env_model or ""),
        },
        "runtime": {
            "use_environment": bool(args.use_environment),
            "base_cache_dir": str(args.base_cache_dir),
            "enable_source_tools": bool(args.enable_source_tools),
            "source_cache_dir": str(args.source_cache_dir),
            "force_refresh_source": bool(args.force_refresh_source),
            "force_rebuild_packet": bool(args.force_rebuild_packet),
            "max_view_chars": int(args.max_view_chars),
            "max_context_chars": int(args.max_context_chars),
            "adaptive_evidence": bool(args.adaptive_evidence),
            "adaptive_evidence_mode": str(args.adaptive_evidence_mode),
            "adaptive_evidence_max_direct_trace_nodes": int(
                args.adaptive_evidence_max_direct_trace_nodes
            ),
            "adaptive_evidence_max_direct_trace_chars": int(
                args.adaptive_evidence_max_direct_trace_chars
            ),
            "adaptive_evidence_max_medium_trace_nodes": int(
                args.adaptive_evidence_max_medium_trace_nodes
            ),
            "adaptive_evidence_debug": bool(args.adaptive_evidence_debug),
            "parallel_judge": bool(args.parallel_judge),
            "judge_concurrency": int(args.judge_concurrency),
            "judge_max_tokens": int(args.judge_max_tokens),
            "aggregator_max_tokens": int(args.aggregator_max_tokens),
            "judge_thinking": str(args.judge_thinking),
            "aggregator_thinking": str(args.aggregator_thinking),
            "structured_judge_logs": bool(args.structured_judge_logs),
            "iv_stateful_runtime": bool(args.iv_stateful_runtime),
            "access_control_binding_mode": str(args.access_control_binding_mode),
            "reentrancy_binding_mode": str(args.reentrancy_binding_mode),
            "disable_stateful_bindings": bool(args.disable_stateful_bindings),
            "dynamic_aggregation": bool(args.dynamic_aggregation),
            "followup_context_mode": str(args.followup_context_mode),
            "judge_followup_mode": str(args.judge_followup_mode),
            "judge_expanded_followup_views": int(
                args.judge_expanded_followup_views
            ),
            "rate_limit_serial_fallback": bool(
                args.rate_limit_serial_fallback
            ),
            "rate_limit_retry_attempts": int(args.rate_limit_retry_attempts),
            "rate_limit_retry_delay_seconds": float(
                args.rate_limit_retry_delay_seconds
            ),
            "save_llm_transcripts": bool(args.save_llm_transcripts),
            "transport": str(resolved_judge_config.get("transport") or ""),
            "base_url": str(resolved_judge_config.get("base_url") or ""),
        },
        "outputs": {
            "results": str(args.results_out),
            "slim_results": str(args.slim_results_out),
            "summary": str(args.out),
            "failed_csv": str(args.failed_csv_out),
        },
    }


def enrich_evaluation_summary(
    summary: Dict[str, Any],
    slim_results: List[Dict[str, Any]],
    target_label: str | None = None,
) -> Dict[str, Any]:
    enriched = dict(summary or {})
    pred_counts = Counter(str(item.get("predicted_verdict")) for item in slim_results)
    reason_counts = Counter(
        str(
            item.get("verdict_reason")
            or (item.get("finding_summary", {}) or {}).get("verdict_reason")
            or "unknown"
        )
        for item in slim_results
    )
    raw_counts = Counter(str(item.get("raw_ground_truth")) for item in slim_results)
    gold_counts = Counter(str(item.get("ground_truth")) for item in slim_results)
    target_key = _normalize_summary_label(target_label or "attack")

    attack_total = gold_counts.get("attack", 0)
    benign_total = gold_counts.get("benign", 0)
    attack_tp = sum(
        1
        for item in slim_results
        if item.get("ground_truth") == "attack"
        and item.get("predicted_verdict") == "attack"
    )
    attack_pred_total = pred_counts.get("attack", 0)
    benign_tn = sum(
        1
        for item in slim_results
        if item.get("ground_truth") == "benign"
        and item.get("predicted_verdict") == "benign"
    )
    negative_items = [item for item in slim_results if item.get("ground_truth") == "benign"]
    benign_negative_items = [
        item for item in negative_items if _negative_kind(item) == "benign"
    ]
    other_attack_negative_items = [
        item for item in negative_items if _negative_kind(item) == "other_attack"
    ]
    unknown_other_negative_items = [
        item
        for item in negative_items
        if _negative_kind(item) not in {"benign", "other_attack"}
    ]

    enriched["predicted_verdict_counts"] = dict(pred_counts)
    enriched["verdict_reason_counts"] = dict(reason_counts)
    enriched["raw_ground_truth_detail_counts"] = dict(raw_counts)
    enriched["binary_ground_truth_counts"] = dict(gold_counts)
    enriched["ground_truth_counts"] = {
        target_key: attack_total,
        "non_target": benign_total,
    }
    enriched["attack_recall"] = (
        round(attack_tp / attack_total, 6) if attack_total else None
    )
    precision = round(attack_tp / attack_pred_total, 6) if attack_pred_total else None
    enriched["attack_precision"] = precision
    enriched["precision"] = precision
    enriched["benign_specificity"] = (
        round(benign_tn / benign_total, 6) if benign_total else None
    )
    enriched["negative_specificity"] = enriched["benign_specificity"]
    enriched["negative_breakdown"] = {
        "benign": len(benign_negative_items),
        "other_attack": len(other_attack_negative_items),
        "unknown_other": len(unknown_other_negative_items),
    }
    enriched["fp_by_negative_kind"] = {
        "benign": sum(
            1 for item in benign_negative_items if item.get("predicted_verdict") == "attack"
        ),
        "other_attack": sum(
            1 for item in other_attack_negative_items if item.get("predicted_verdict") == "attack"
        ),
        "unknown_other": sum(
            1 for item in unknown_other_negative_items if item.get("predicted_verdict") == "attack"
        ),
    }
    enriched["error_cases"] = [
        _summary_error_case(item)
        for item in slim_results
        if _is_error_case(item)
    ]
    return enriched


def _normalize_summary_label(label: str) -> str:
    return normalize_attack_label(label)


def write_failed_csv(
    path_like: str,
    results: List[Dict[str, Any]],
    slim_results: List[Dict[str, Any]],
) -> int:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _failed_csv_row(result, slim)
        for result, slim in zip(results, slim_results)
        if _is_error_case(slim)
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILED_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _failed_csv_row(result: Dict[str, Any], slim: Dict[str, Any]) -> Dict[str, str]:
    transaction = result.get("transaction", {}) if isinstance(result, dict) else {}
    evaluation = result.get("evaluation", {}) if isinstance(result, dict) else {}
    metadata = evaluation.get("case_metadata", {}) if isinstance(evaluation, dict) else {}
    if not isinstance(transaction, dict):
        transaction = {}
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "HackId": _first_non_empty(
            metadata.get("HackId"),
            metadata.get("hack_id"),
            metadata.get("Id"),
            metadata.get("id"),
        ),
        "txHash": _first_non_empty(
            slim.get("tx_hash"),
            transaction.get("tx_hash"),
            result.get("tx_hash"),
        ),
        "Chain": _first_non_empty(
            slim.get("chain"),
            transaction.get("chain"),
            result.get("chain"),
        ),
        "Type": _first_non_empty(
            metadata.get("Type"),
            metadata.get("type"),
            evaluation.get("raw_ground_truth"),
            slim.get("raw_ground_truth"),
        ),
        "Cause": _first_non_empty(
            metadata.get("Cause"),
            metadata.get("cause"),
            metadata.get("Description"),
            metadata.get("description"),
            evaluation.get("report"),
            evaluation.get("label_rationale"),
        ),
    }


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _is_error_case(item: Dict[str, Any]) -> bool:
    gold = item.get("ground_truth")
    pred = item.get("predicted_verdict")
    return (gold == "attack" and pred != "attack") or (
        gold == "benign" and pred == "attack"
    )


def _summary_error_case(item: Dict[str, Any]) -> Dict[str, Any]:
    finding = item.get("finding_summary", {}) or {}
    failed_conditions = []
    uncertain_conditions = []
    supported_exclusions_for_positive = []
    supported_exclusion_details_for_positive = []
    is_positive_miss = item.get("ground_truth") == "attack" and item.get("predicted_verdict") != "attack"
    for row in item.get("condition_table", []) or []:
        condition_id = row.get("condition_id") or row.get("id")
        answer = row.get("answer")
        if answer == "uncertain":
            uncertain_conditions.append(condition_id)
        if row.get("error_hint"):
            failed_conditions.append({
                "condition_id": condition_id,
                "answer": answer,
                "confidence": row.get("confidence"),
                "error_hint": row.get("error_hint"),
            })
        is_exclusion = bool(row.get("is_exclusion")) or str(condition_id or "").startswith("E")
        if is_positive_miss and is_exclusion and _answer_is_true(answer):
            supported_exclusions_for_positive.append(condition_id)
            supported_exclusion_details_for_positive.append({
                "condition_id": condition_id,
                "answer": answer,
                "confidence": row.get("confidence"),
                "supporting_evidence_ids": row.get("supporting_evidence_ids", []),
                "reason": row.get("reason", ""),
            })

    return {
        "tx_hash": item.get("tx_hash"),
        "chain": item.get("chain"),
        "ground_truth": item.get("ground_truth"),
        "raw_ground_truth": item.get("raw_ground_truth"),
        "sample_role": item.get("sample_role"),
        "negative_kind": _negative_kind(item),
        "predicted_verdict": item.get("predicted_verdict"),
        "confidence": item.get("confidence"),
        "verdict_reason": item.get("verdict_reason")
        or finding.get("verdict_reason", ""),
        "failed_conditions": failed_conditions,
        "uncertain_conditions": uncertain_conditions,
        "supported_exclusions_for_positive": supported_exclusions_for_positive,
        "supported_exclusion_details_for_positive": supported_exclusion_details_for_positive,
        "open_missing_count": (
            item.get("missing_evidence_actionability_summary", {}) or {}
        ).get("open_blocking", 0),
    }


def _answer_is_true(answer: Any) -> bool:
    if answer is True:
        return True
    if isinstance(answer, str):
        return answer.strip().lower() == "true"
    return False


def _negative_kind(item: Dict[str, Any]) -> str:
    kind = str(item.get("negative_kind") or "").strip()
    if kind:
        return kind
    raw = str(item.get("raw_ground_truth") or "").strip().lower()
    if raw in {"benign", "normal", "safe", "legitimate"}:
        return "benign"
    return "unknown_other"

def load_cases_input(args) -> tuple[List[Dict[str, Any]], str]:
    pos_csv = args.pos_csv or args.malicious_csv
    has_negative_csv = bool(args.neg_csv or args.benign_csv)
    if pos_csv or has_negative_csv:
        if not pos_csv or not has_negative_csv or not args.label:
            raise ValueError(
                "When using split CSV input, provide --pos-csv, --neg-csv, and --label together "
                "(legacy --malicious-csv/--benign-csv are still accepted)."
            )
        return load_cases_from_split_csvs(
            pos_csv=args.pos_csv,
            neg_csv=args.neg_csv,
            benign_csv=args.benign_csv,
            malicious_csv=args.malicious_csv,
            label=args.label,
            head_per_csv=args.only_head,
        )

    if not args.cases_file:
        raise ValueError(
            "Provide either --cases-file or --pos-csv --neg-csv --label."
        )
    cases, label = load_labeled_cases(
        args.cases_file,
        positive_label=args.positive_label,
    )
    if args.only_head is not None:
        cases = cases[: args.only_head]
    return cases, label


def load_results(path_like: str) -> List[Dict[str, Any]]:
    path = Path(path_like)
    if path.is_dir():
        return [read_json(item) for item in sorted(path.glob("*.json"))]

    data = read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported result payload: {path}")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_elapsed_hms(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    whole_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


if __name__ == "__main__":
    main()
