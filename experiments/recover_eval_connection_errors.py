from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.evolution.regression import RegressionEvaluator
from evotx.utils.json_utils import read_json, write_json
from evotx.utils.result_slimmer import build_slim_result
from evotx.utils.result_utils import get_finding, get_tx_hash
from experiments.evaluate import enrich_evaluation_summary, write_failed_csv


TX_PATTERN = re.compile(r"0x[0-9a-fA-F]{64}")
CASE_PATTERN = re.compile(
    r"^\[Evaluate\] Case (?P<case>\d+)/\d+:\s+tx=(?P<tx>0x[0-9a-fA-F]{64})"
)
RUNTIME_TX_PATTERN = re.compile(
    r"^\[RunInference\] Packet runtime for tx=(?P<tx>0x[0-9a-fA-F]{64})"
)
ERROR_NAME_PATTERN = re.compile(
    r"\b("
    r"APIConnectionError|APITimeoutError|ConnectionError|ConnectError|"
    r"RemoteProtocolError|ReadTimeout|ConnectTimeout"
    r")\b",
    re.IGNORECASE,
)
TERMINAL_RUNTIME_ERROR_PATTERN = re.compile(
    r"^\[(?:PacketRuntime|RunInference)\].*\b(?:raised|failed)\b.*"
    r"(?:APIConnectionError|APITimeoutError|ConnectionError|ConnectError|"
    r"RemoteProtocolError|ReadTimeout|ConnectTimeout)",
    re.IGNORECASE,
)
ADAPTER_ERROR_PATTERN = re.compile(
    r"^\[LLM Adapter Error\].*"
    r"(?:APIConnectionError|APITimeoutError|ConnectionError|ConnectError|"
    r"RemoteProtocolError|ReadTimeout|ConnectTimeout)",
    re.IGNORECASE,
)
CONNECTION_MARKERS = (
    "apiconnectionerror",
    "apitimeouterror",
    "judge_transport_error",
    "connectionerror",
    "connecterror",
    "remoteprotocolerror",
    "readtimeout",
    "connecttimeout",
)
TX_COLUMN_NAMES = {
    "txhash",
    "tx_hash",
    "hash",
    "transaction_hash",
    "transactionhash",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an episode evaluation log for terminal API connection errors, "
            "optionally rerun only those transactions, and replace successful "
            "reruns in the original evaluation outputs."
        )
    )
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--results-file")
    parser.add_argument("--slim-results-file")
    parser.add_argument("--summary-file")
    parser.add_argument("--failed-csv-file")
    parser.add_argument("--run-manifest-file")
    parser.add_argument("--rule-file")
    parser.add_argument("--plan-file")
    parser.add_argument("--pos-csv")
    parser.add_argument("--neg-csv")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-provider")
    parser.add_argument("--planner-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--env-model")
    parser.add_argument(
        "--glm-en-anthropic",
        choices=["auto", "enabled", "disabled"],
        default="auto",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only export connection-error cases; do not call an LLM or replace results.",
    )
    parser.add_argument(
        "--rerun-and-apply",
        action="store_true",
        help="Rerun unresolved cases and atomically replace successful results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build all recovery artifacts and print the rerun command without executing it.",
    )
    args = parser.parse_args()

    if args.episode < 0:
        raise ValueError("--episode must be a non-negative integer.")
    if args.scan_only and args.rerun_and_apply:
        raise ValueError("Choose either --scan-only or --rerun-and-apply.")
    if not args.scan_only and not args.rerun_and_apply:
        args.scan_only = True

    paths = resolve_episode_artifact_paths(
        episode=args.episode,
        output_prefix=args.output_prefix,
        results_file=args.results_file,
        slim_results_file=args.slim_results_file,
        summary_file=args.summary_file,
        failed_csv_file=args.failed_csv_file,
        run_manifest_file=args.run_manifest_file,
    )
    log_path = resolve_eval_log(args.episode, args.log_file)
    original_results = load_result_list(paths["results"])
    original_by_tx = {
        normalize_tx_hash(get_tx_hash(item)): item for item in original_results
    }

    run_config = resolve_run_config(
        log_path=log_path,
        results=original_results,
        manifest_path=paths["run_manifest"],
        overrides={
            "rule_file": args.rule_file,
            "plan_file": args.plan_file,
            "pos_csv": args.pos_csv,
            "neg_csv": args.neg_csv,
            "llm_model": args.llm_model,
            "llm_provider": args.llm_provider,
            "planner_model": args.planner_model,
            "judge_model": args.judge_model,
            "env_model": args.env_model,
        },
    )
    error_records = scan_connection_error_log(
        log_path,
        results_by_tx=original_by_tx,
    )
    case_rows = resolve_connection_error_cases(
        error_records,
        pos_csv=run_config["inputs"]["pos_csv"],
        neg_csv=run_config["inputs"]["neg_csv"],
    )
    case_artifacts = write_connection_error_case_artifacts(
        episode=args.episode,
        output_prefix=args.output_prefix,
        log_path=log_path,
        run_config=run_config,
        cases=case_rows,
    )

    unresolved_cases = [
        item for item in case_rows if item["recovery_status"] == "needs_rerun"
    ]
    print(
        "[EvalRecovery] "
        f"log={display_path(log_path)} terminal_connection_error_txs={len(case_rows)} "
        f"unresolved={len(unresolved_cases)}"
    )
    print(
        "[EvalRecovery] Cases: "
        f"json={display_path(case_artifacts['json'])} "
        f"csv={display_path(case_artifacts['csv'])}"
    )

    if args.scan_only:
        return 0
    if not unresolved_cases:
        print("[EvalRecovery] No unresolved connection-error cases require a rerun.")
        return 0

    validate_replay_config(run_config)
    run_dir = create_recovery_run_dir(
        paths["results"].parent,
        output_prefix=args.output_prefix,
    )
    subset_paths = write_recovery_split_csvs(
        run_dir,
        unresolved_cases,
        run_config=run_config,
    )
    command = build_evaluate_command(
        run_config=run_config,
        subset_paths=subset_paths,
        run_dir=run_dir,
    )
    recovery_manifest_path = run_dir / "recovery_manifest.json"
    recovery_manifest = {
        "schema_version": "evotx.eval_connection_recovery.v1",
        "episode": args.episode,
        "started_at": now_iso(),
        "source_log": display_path(log_path),
        "source_results": display_path(paths["results"]),
        "source_rule": run_config["artifacts"]["rule_file"],
        "source_plan": run_config["artifacts"]["plan_file"],
        "detected_tx_count": len(case_rows),
        "requested_rerun_count": len(unresolved_cases),
        "requested_tx_hashes": [item["tx_hash"] for item in unresolved_cases],
        "command": subprocess.list2cmdline(command),
        "status": "dry_run" if args.dry_run else "running",
    }
    write_json(recovery_manifest_path, recovery_manifest)
    print(f"[EvalRecovery] Rerun command: {subprocess.list2cmdline(command)}")

    if args.dry_run:
        print(
            f"[EvalRecovery] Dry run manifest: {display_path(recovery_manifest_path)}"
        )
        return 0

    child_env = build_child_environment(
        run_config,
        glm_en_anthropic=args.glm_en_anthropic,
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=child_env,
        check=False,
    )
    if completed.returncode != 0:
        recovery_manifest.update(
            {
                "status": "rerun_failed",
                "finished_at": now_iso(),
                "return_code": completed.returncode,
            }
        )
        write_json(recovery_manifest_path, recovery_manifest)
        raise RuntimeError(
            f"Targeted evaluation rerun failed with exit code {completed.returncode}."
        )

    rerun_results_path = run_dir / "rerun_eval_results.json"
    rerun_results = load_result_list(rerun_results_path)
    merged_results, merge_audit = merge_recovered_results(
        original_results,
        rerun_results,
        requested_tx_hashes=[item["tx_hash"] for item in unresolved_cases],
    )
    if merge_audit["replaced_count"]:
        backup_dir = backup_evaluation_outputs(paths, run_dir / "backup")
        try:
            rewrite_evaluation_outputs(
                paths=paths,
                results=merged_results,
                episode=args.episode,
                target_label=run_config["label"],
                recovery_audit={
                    **merge_audit,
                    "source_log": display_path(log_path),
                    "recovery_run_dir": display_path(run_dir),
                    "backup_dir": display_path(backup_dir),
                },
            )
        except Exception:
            restore_evaluation_outputs(paths, backup_dir)
            recovery_manifest.update(
                {
                    "status": "apply_failed_rolled_back",
                    "finished_at": now_iso(),
                    "backup_dir": display_path(backup_dir),
                    **merge_audit,
                }
            )
            write_json(recovery_manifest_path, recovery_manifest)
            raise
    else:
        backup_dir = None

    recovery_manifest.update(
        {
            "status": (
                "applied"
                if merge_audit["replaced_count"]
                else "no_successful_replacements"
            ),
            "finished_at": now_iso(),
            "return_code": completed.returncode,
            "rerun_results": display_path(rerun_results_path),
            "backup_dir": display_path(backup_dir) if backup_dir else "",
            **merge_audit,
        }
    )
    write_json(recovery_manifest_path, recovery_manifest)
    latest_manifest = run_dir.parent / "latest_recovery.json"
    write_json(latest_manifest, recovery_manifest)
    print(
        "[EvalRecovery] Applied replacements: "
        f"{merge_audit['replaced_count']}; unresolved after rerun: "
        f"{merge_audit['unresolved_count']}"
    )
    print(
        f"[EvalRecovery] Recovery manifest: {display_path(recovery_manifest_path)}"
    )
    return 0


def resolve_episode_artifact_paths(
    *,
    episode: int,
    output_prefix: str = "",
    results_file: str | None = None,
    slim_results_file: str | None = None,
    summary_file: str | None = None,
    failed_csv_file: str | None = None,
    run_manifest_file: str | None = None,
) -> Dict[str, Path]:
    result_dir = PROJECT_ROOT / "data" / "results" / str(episode)
    prefix = normalize_prefix(output_prefix)
    names = {
        "results": f"{prefix}results.json",
        "slim_results": f"{prefix}slim_results.json",
        "summary": f"{prefix}summary.json",
        "failed_csv": f"{prefix}failed.csv",
        "run_manifest": f"{prefix}run_manifest.json",
    }
    overrides = {
        "results": results_file,
        "slim_results": slim_results_file,
        "summary": summary_file,
        "failed_csv": failed_csv_file,
        "run_manifest": run_manifest_file,
    }
    paths = {
        key: resolve_repo_path(overrides[key]) if overrides[key] else result_dir / name
        for key, name in names.items()
    }
    if not paths["results"].exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {display_path(paths['results'])}"
        )
    return paths


def resolve_eval_log(episode: int, explicit: str | None = None) -> Path:
    if explicit:
        path = resolve_repo_path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation log not found: {path}")
        return path
    log_dir = PROJECT_ROOT / "data" / "log" / str(episode)
    candidates = sorted(
        log_dir.glob("evaluate__*.log"),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No evaluate__*.log found under {display_path(log_dir)}"
        )
    return candidates[0]


def scan_connection_error_log(
    log_path: str | Path,
    *,
    results_by_tx: Mapping[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    path = Path(log_path)
    active_tx = ""
    active_case = 0
    records: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            case_match = CASE_PATTERN.search(line)
            runtime_match = RUNTIME_TX_PATTERN.search(line)
            if case_match:
                active_case = int(case_match.group("case"))
                active_tx = normalize_tx_hash(case_match.group("tx"))
            elif runtime_match:
                active_tx = normalize_tx_hash(runtime_match.group("tx"))

            is_terminal = bool(TERMINAL_RUNTIME_ERROR_PATTERN.search(line))
            is_adapter_error = bool(ADAPTER_ERROR_PATTERN.search(line))
            if not active_tx or not (is_terminal or is_adapter_error):
                continue
            if active_tx not in records:
                records[active_tx] = {
                    "tx_hash": active_tx,
                    "case_index": active_case,
                    "terminal_error_lines": [],
                    "adapter_error_lines": [],
                    "error_types": [],
                    "messages": [],
                }
                order.append(active_tx)
            record = records[active_tx]
            error_match = ERROR_NAME_PATTERN.search(line)
            if error_match:
                error_type = error_match.group(1)
                if error_type not in record["error_types"]:
                    record["error_types"].append(error_type)
            if is_terminal:
                record["terminal_error_lines"].append(line_number)
                record["messages"].append(
                    {"line": line_number, "kind": "runtime_terminal", "text": line}
                )
            elif is_adapter_error:
                record["adapter_error_lines"].append(line_number)

    output: List[Dict[str, Any]] = []
    for tx_hash in order:
        record = records[tx_hash]
        current_result = (results_by_tx or {}).get(tx_hash)
        current_has_error = (
            result_has_connection_error(current_result)
            if current_result is not None
            else True
        )
        if not record["terminal_error_lines"]:
            continue
        record["terminal_error_count"] = len(record["terminal_error_lines"])
        record["adapter_error_count"] = len(record["adapter_error_lines"])
        record["first_error_line"] = min(record["terminal_error_lines"])
        record["last_error_line"] = max(record["terminal_error_lines"])
        record["recovery_status"] = (
            "needs_rerun" if current_has_error else "already_repaired"
        )
        output.append(record)
    return output


def result_has_connection_error(result: Dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    finding = get_finding(result)
    for value in iter_nested_strings(finding):
        normalized = value.lower().replace("_", "")
        if any(marker.replace("_", "") in normalized for marker in CONNECTION_MARKERS):
            return True
    return False


def iter_nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from iter_nested_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from iter_nested_strings(nested)


def resolve_connection_error_cases(
    error_records: Sequence[Dict[str, Any]],
    *,
    pos_csv: str,
    neg_csv: str,
) -> List[Dict[str, Any]]:
    sources = [
        ("positive", resolve_repo_path(pos_csv)),
        ("negative", resolve_repo_path(neg_csv)),
    ]
    indexed: Dict[str, Dict[str, Any]] = {}
    for split, path in sources:
        headers, rows = read_csv_rows(path)
        tx_column = find_tx_column(headers)
        for row_number, row in enumerate(rows, start=2):
            tx_hash = normalize_tx_hash(row.get(tx_column, ""))
            if not tx_hash:
                continue
            if tx_hash in indexed:
                raise ValueError(
                    f"Duplicate tx hash {tx_hash} across archived evaluation CSVs."
                )
            indexed[tx_hash] = {
                "source_split": split,
                "source_csv": display_path(path),
                "source_row_number": row_number,
                "csv_headers": headers,
                "csv_row": row,
            }

    cases: List[Dict[str, Any]] = []
    for record in error_records:
        tx_hash = normalize_tx_hash(record["tx_hash"])
        source = indexed.get(tx_hash)
        case = dict(record)
        if source:
            case.update(source)
        else:
            case.update(
                {
                    "source_split": "missing",
                    "source_csv": "",
                    "source_row_number": None,
                    "csv_headers": [],
                    "csv_row": {},
                }
            )
            case["recovery_status"] = "missing_csv_row"
        cases.append(case)
    return cases


def write_connection_error_case_artifacts(
    *,
    episode: int,
    output_prefix: str,
    log_path: Path,
    run_config: Dict[str, Any],
    cases: Sequence[Dict[str, Any]],
) -> Dict[str, Path]:
    result_dir = PROJECT_ROOT / "data" / "results" / str(episode)
    prefix = normalize_prefix(output_prefix)
    json_path = result_dir / f"{prefix}connection_error_cases.json"
    csv_path = result_dir / f"{prefix}connection_error_cases.csv"
    payload = {
        "schema_version": "evotx.eval_connection_error_cases.v1",
        "generated_at": now_iso(),
        "episode": episode,
        "source_log": display_path(log_path),
        "source_csvs": dict(run_config["inputs"]),
        "terminal_connection_error_tx_count": len(cases),
        "needs_rerun_count": sum(
            1 for item in cases if item["recovery_status"] == "needs_rerun"
        ),
        "cases": list(cases),
    }
    write_json(json_path, payload)
    write_case_csv(csv_path, cases)
    return {"json": json_path, "csv": csv_path}


def write_case_csv(path: Path, cases: Sequence[Dict[str, Any]]) -> None:
    metadata_fields = [
        "recovery_status",
        "case_index",
        "tx_hash",
        "source_split",
        "source_csv",
        "source_row_number",
        "error_types",
        "terminal_error_count",
        "adapter_error_count",
        "first_error_line",
        "last_error_line",
    ]
    row_fields: List[str] = []
    for case in cases:
        for field in case.get("csv_headers", []):
            if field not in row_fields and field not in metadata_fields:
                row_fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields + row_fields)
        writer.writeheader()
        for case in cases:
            row = {
                "recovery_status": case.get("recovery_status", ""),
                "case_index": case.get("case_index", ""),
                "tx_hash": case.get("tx_hash", ""),
                "source_split": case.get("source_split", ""),
                "source_csv": case.get("source_csv", ""),
                "source_row_number": case.get("source_row_number", ""),
                "error_types": "|".join(case.get("error_types", [])),
                "terminal_error_count": case.get("terminal_error_count", 0),
                "adapter_error_count": case.get("adapter_error_count", 0),
                "first_error_line": case.get("first_error_line", ""),
                "last_error_line": case.get("last_error_line", ""),
            }
            row.update(case.get("csv_row", {}))
            writer.writerow(row)


def resolve_run_config(
    *,
    log_path: Path,
    results: Sequence[Dict[str, Any]],
    manifest_path: Path,
    overrides: Mapping[str, str | None],
) -> Dict[str, Any]:
    if manifest_path.exists():
        config = read_json(manifest_path)
    else:
        config = legacy_run_config_from_log(log_path, results)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid eval run manifest: {manifest_path}")

    config = copy.deepcopy(config)
    config.setdefault("inputs", {})
    config.setdefault("artifacts", {})
    config.setdefault("models", {})
    config.setdefault("runtime", {})
    for key in ("pos_csv", "neg_csv"):
        if overrides.get(key):
            config["inputs"][key] = str(overrides[key])
    for key in ("rule_file", "plan_file"):
        if overrides.get(key):
            config["artifacts"][key] = str(overrides[key])
    for key in (
        "llm_model",
        "llm_provider",
        "planner_model",
        "judge_model",
        "env_model",
    ):
        if overrides.get(key):
            config["models"][key] = str(overrides[key])
    return config


def legacy_run_config_from_log(
    log_path: Path,
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    runtime_line = first_matching_line(
        log_text, "[Evaluate] Running fresh evaluation:"
    )
    model_line = first_matching_line(log_text, "[Evaluate] Models:")
    runtime_pairs = parse_key_value_line(runtime_line)
    model_pairs = parse_key_value_line(model_line)
    csv_paths = parse_archived_csv_paths(log_text)
    retry_delay_match = re.search(
        r"Judge transport error detected;.*?\bdelay=([0-9.]+)s",
        log_text,
    )
    detector_context = {}
    if results:
        detector_context = dict(results[0].get("detector_context", {}) or {})
    label = (
        runtime_pairs.get("label")
        or detector_context.get("attack_label")
        or "attack"
    )
    runtime = {
        "use_environment": False,
        "base_cache_dir": "data/cache",
        "enable_source_tools": parse_bool(runtime_pairs.get("source_tools"), True),
        "source_cache_dir": "data/cache/contracts",
        "force_refresh_source": False,
        "force_rebuild_packet": parse_bool(
            runtime_pairs.get("force_rebuild_packet"), True
        ),
        "max_view_chars": parse_int(runtime_pairs.get("max_view_chars"), 20000),
        "max_context_chars": parse_int(
            runtime_pairs.get("max_context_chars"), 60000
        ),
        "adaptive_evidence": parse_bool(
            runtime_pairs.get("adaptive_evidence"), False
        ),
        "adaptive_evidence_mode": runtime_pairs.get("adaptive_mode", "off"),
        "adaptive_evidence_max_direct_trace_nodes": 80,
        "adaptive_evidence_max_direct_trace_chars": 45000,
        "adaptive_evidence_max_medium_trace_nodes": 250,
        "adaptive_evidence_debug": False,
        "parallel_judge": parse_bool(runtime_pairs.get("parallel_judge"), False),
        "judge_concurrency": parse_int(
            runtime_pairs.get("judge_concurrency"), 1
        ),
        "judge_max_tokens": parse_int(
            runtime_pairs.get("judge_max_tokens"), 8192
        ),
        "aggregator_max_tokens": parse_int(
            runtime_pairs.get("aggregator_max_tokens"), 8192
        ),
        "judge_thinking": runtime_pairs.get("judge_thinking", "disabled"),
        "aggregator_thinking": runtime_pairs.get(
            "aggregator_thinking", "disabled"
        ),
        "structured_judge_logs": True,
        "iv_stateful_runtime": parse_bool(
            runtime_pairs.get("iv_stateful_runtime"), True
        ),
        "access_control_binding_mode": runtime_pairs.get(
            "access_control_binding_mode", "stateful"
        ),
        "reentrancy_binding_mode": runtime_pairs.get(
            "reentrancy_binding_mode", "soft"
        ),
        "dynamic_aggregation": parse_bool(
            runtime_pairs.get("dynamic_aggregation"), True
        ),
        "followup_context_mode": runtime_pairs.get(
            "followup_context_mode", "unified"
        ),
        "rate_limit_serial_fallback": parse_bool(
            runtime_pairs.get("rate_limit_serial_fallback"), True
        ),
        "rate_limit_retry_attempts": parse_int(
            runtime_pairs.get("rate_limit_retries"), 3
        ),
        "rate_limit_retry_delay_seconds": (
            float(retry_delay_match.group(1)) if retry_delay_match else 2.0
        ),
        "save_llm_transcripts": True,
        "transport": (
            "anthropic"
            if '"transport": "anthropic"' in log_text.split(
                "[Evaluate] Case 1/", 1
            )[0]
            else "openai"
        ),
    }
    planner = empty_if_disabled(model_pairs.get("planner", ""))
    judge = empty_if_disabled(model_pairs.get("judge", ""))
    env_model = empty_if_disabled(model_pairs.get("env", ""))
    common_model = judge or planner or env_model
    return {
        "schema_version": "evotx.eval_run_manifest.legacy_log_v1",
        "label": label,
        "positive_label": label,
        "inputs": csv_paths,
        "artifacts": {
            "rule_file": str(detector_context.get("rule_source") or ""),
            "plan_file": str(detector_context.get("plan_source") or ""),
            "tool_manifest": "configs/tool_manifest.json",
        },
        "models": {
            "llm_model": common_model,
            "llm_provider": model_pairs.get("provider", ""),
            "planner_model": planner if planner != common_model else "",
            "judge_model": judge if judge != common_model else "",
            "env_model": env_model if env_model != common_model else "",
        },
        "runtime": runtime,
    }


def validate_replay_config(config: Dict[str, Any]) -> None:
    required = {
        "positive CSV": config["inputs"].get("pos_csv"),
        "negative CSV": config["inputs"].get("neg_csv"),
        "rule": config["artifacts"].get("rule_file"),
        "plan": config["artifacts"].get("plan_file"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Cannot reconstruct targeted evaluation; missing "
            + ", ".join(missing)
            + ". Pass explicit overrides."
        )
    for name, value in required.items():
        path = resolve_repo_path(str(value))
        if not path.exists():
            raise FileNotFoundError(f"Recovery {name} not found: {path}")
    if not config["models"].get("llm_model") and not config["models"].get(
        "judge_model"
    ):
        raise ValueError(
            "Cannot infer the evaluation LLM model. Pass --llm-model."
        )


def create_recovery_run_dir(result_dir: Path, *, output_prefix: str) -> Path:
    prefix = normalize_prefix(output_prefix)
    root_name = f"{prefix}connection_error_recovery".rstrip("_")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    path = result_dir / root_name / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_recovery_split_csvs(
    run_dir: Path,
    cases: Sequence[Dict[str, Any]],
    *,
    run_config: Dict[str, Any],
) -> Dict[str, Path]:
    paths = {
        "positive": run_dir / "connection_error_pos.csv",
        "negative": run_dir / "connection_error_neg.csv",
    }
    source_paths = {
        "positive": resolve_repo_path(run_config["inputs"]["pos_csv"]),
        "negative": resolve_repo_path(run_config["inputs"]["neg_csv"]),
    }
    for split in ("positive", "negative"):
        source_headers, _ = read_csv_rows(source_paths[split])
        selected_rows = [
            dict(item["csv_row"])
            for item in cases
            if item["source_split"] == split
        ]
        write_csv_rows(paths[split], source_headers, selected_rows)
    return paths


def build_evaluate_command(
    *,
    run_config: Dict[str, Any],
    subset_paths: Mapping[str, Path],
    run_dir: Path,
) -> List[str]:
    models = run_config["models"]
    runtime = run_config["runtime"]
    artifacts = run_config["artifacts"]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "evaluate.py"),
        "--pos-csv",
        str(subset_paths["positive"]),
        "--neg-csv",
        str(subset_paths["negative"]),
        "--label",
        str(run_config["label"]),
        "--positive-label",
        str(run_config.get("positive_label") or run_config["label"]),
        "--rule-file",
        str(resolve_repo_path(artifacts["rule_file"])),
        "--plan-file",
        str(resolve_repo_path(artifacts["plan_file"])),
        "--tool-manifest",
        str(resolve_repo_path(artifacts.get("tool_manifest") or "configs/tool_manifest.json")),
        "--base-cache-dir",
        str(resolve_repo_path(runtime.get("base_cache_dir") or "data/cache")),
        "--source-cache-dir",
        str(
            resolve_repo_path(
                runtime.get("source_cache_dir") or "data/cache/contracts"
            )
        ),
        "--logs-dir",
        str(run_dir / "logs"),
        "--results-out",
        str(run_dir / "rerun_eval_results.json"),
        "--slim-results-out",
        str(run_dir / "rerun_eval_slim_results.json"),
        "--out",
        str(run_dir / "rerun_eval_summary.json"),
        "--failed-csv-out",
        str(run_dir / "rerun_failed.csv"),
        "--max-view-chars",
        str(runtime.get("max_view_chars", 20000)),
        "--max-context-chars",
        str(runtime.get("max_context_chars", 60000)),
        "--judge-max-tokens",
        str(runtime.get("judge_max_tokens", 8192)),
        "--aggregator-max-tokens",
        str(runtime.get("aggregator_max_tokens", 8192)),
        "--judge-thinking",
        str(runtime.get("judge_thinking", "disabled")),
        "--aggregator-thinking",
        str(runtime.get("aggregator_thinking", "disabled")),
        "--judge-concurrency",
        str(runtime.get("judge_concurrency", 1)),
        "--adaptive-evidence-mode",
        str(runtime.get("adaptive_evidence_mode", "off")),
        "--adaptive-evidence-max-direct-trace-nodes",
        str(runtime.get("adaptive_evidence_max_direct_trace_nodes", 80)),
        "--adaptive-evidence-max-direct-trace-chars",
        str(runtime.get("adaptive_evidence_max_direct_trace_chars", 45000)),
        "--adaptive-evidence-max-medium-trace-nodes",
        str(runtime.get("adaptive_evidence_max_medium_trace_nodes", 250)),
        "--access-control-binding-mode",
        str(runtime.get("access_control_binding_mode", "stateful")),
        "--reentrancy-binding-mode",
        str(runtime.get("reentrancy_binding_mode", "soft")),
        "--followup-context-mode",
        str(runtime.get("followup_context_mode", "unified")),
        "--rate-limit-retry-attempts",
        str(runtime.get("rate_limit_retry_attempts", 3)),
        "--rate-limit-retry-delay-seconds",
        str(runtime.get("rate_limit_retry_delay_seconds", 2.0)),
    ]
    for flag, value in (
        ("--llm-model", models.get("llm_model")),
        ("--llm-provider", models.get("llm_provider")),
        ("--planner-model", models.get("planner_model")),
        ("--judge-model", models.get("judge_model")),
        ("--env-model", models.get("env_model")),
    ):
        if value:
            command.extend([flag, str(value)])

    boolean_flags = [
        ("use_environment", "--use-environment", None),
        ("enable_source_tools", "--enable-source-tools", None),
        ("force_refresh_source", "--force-refresh-source", None),
        ("force_rebuild_packet", "--force-rebuild-packet", None),
        ("adaptive_evidence", "--adaptive-evidence", None),
        ("adaptive_evidence_debug", "--adaptive-evidence-debug", None),
        ("parallel_judge", "--parallel-judge", None),
        (
            "structured_judge_logs",
            "--structured-judge-logs",
            "--no-structured-judge-logs",
        ),
        (
            "iv_stateful_runtime",
            "--enable-iv-stateful-runtime",
            "--disable-iv-stateful-runtime",
        ),
        (
            "dynamic_aggregation",
            "--enable-dynamic-aggregation",
            "--disable-dynamic-aggregation",
        ),
        (
            "rate_limit_serial_fallback",
            "--rate-limit-serial-fallback",
            "--disable-rate-limit-serial-fallback",
        ),
    ]
    for key, true_flag, false_flag in boolean_flags:
        enabled = bool(runtime.get(key, False))
        if enabled:
            command.append(true_flag)
        elif false_flag:
            command.append(false_flag)
    if runtime.get("save_llm_transcripts", True):
        command.extend(
            [
                "--save-llm-transcripts",
                "--llm-transcripts-dir",
                str(run_dir / "llm_transcripts"),
            ]
        )
    return command


def build_child_environment(
    run_config: Dict[str, Any],
    *,
    glm_en_anthropic: str,
) -> Dict[str, str]:
    child_env = dict(os.environ)
    provider = str(run_config["models"].get("llm_provider") or "").lower()
    inferred_anthropic = (
        str(run_config["runtime"].get("transport") or "").lower() == "anthropic"
    )
    if provider == "glm-en":
        if glm_en_anthropic == "enabled":
            child_env["GLM_EN_ENABLE_ANTHROPIC"] = "1"
        elif glm_en_anthropic == "disabled":
            child_env["GLM_EN_ENABLE_ANTHROPIC"] = "0"
        elif inferred_anthropic:
            child_env["GLM_EN_ENABLE_ANTHROPIC"] = "1"
    return child_env


def merge_recovered_results(
    original_results: Sequence[Dict[str, Any]],
    rerun_results: Sequence[Dict[str, Any]],
    *,
    requested_tx_hashes: Sequence[str],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    requested = {normalize_tx_hash(item) for item in requested_tx_hashes}
    rerun_by_tx: Dict[str, Dict[str, Any]] = {}
    duplicate_rerun_txs: List[str] = []
    for item in rerun_results:
        tx_hash = normalize_tx_hash(get_tx_hash(item))
        if not tx_hash:
            continue
        if tx_hash in rerun_by_tx:
            duplicate_rerun_txs.append(tx_hash)
        rerun_by_tx[tx_hash] = item
    if duplicate_rerun_txs:
        raise ValueError(
            "Duplicate tx hashes in rerun results: "
            + ", ".join(sorted(set(duplicate_rerun_txs)))
        )

    merged: List[Dict[str, Any]] = []
    replaced: List[str] = []
    unresolved: List[str] = []
    changes: List[Dict[str, Any]] = []
    seen_original: set[str] = set()
    for original in original_results:
        tx_hash = normalize_tx_hash(get_tx_hash(original))
        if tx_hash in seen_original:
            raise ValueError(f"Duplicate tx hash in original results: {tx_hash}")
        seen_original.add(tx_hash)
        rerun = rerun_by_tx.get(tx_hash)
        if tx_hash not in requested or rerun is None:
            merged.append(original)
            continue
        if result_has_connection_error(rerun):
            unresolved.append(tx_hash)
            merged.append(original)
            continue
        recovered = copy.deepcopy(rerun)
        if "evaluation" in original:
            recovered["evaluation"] = copy.deepcopy(original["evaluation"])
        old_slim = build_slim_result(original)
        new_slim = build_slim_result(recovered)
        changes.append(
            {
                "tx_hash": tx_hash,
                "old_verdict": old_slim.get("predicted_verdict"),
                "new_verdict": new_slim.get("predicted_verdict"),
                "ground_truth": new_slim.get("ground_truth"),
            }
        )
        merged.append(recovered)
        replaced.append(tx_hash)

    missing_original = sorted(requested - seen_original)
    missing_rerun = sorted(
        tx_hash
        for tx_hash in requested
        if tx_hash in seen_original and tx_hash not in rerun_by_tx
    )
    unresolved.extend(missing_rerun)
    return merged, {
        "replaced_count": len(replaced),
        "replaced_tx_hashes": replaced,
        "unresolved_count": len(sorted(set(unresolved))),
        "unresolved_tx_hashes": sorted(set(unresolved)),
        "missing_original_tx_hashes": missing_original,
        "changes": changes,
    }


def backup_evaluation_outputs(
    paths: Mapping[str, Path],
    backup_dir: Path,
) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=False)
    for key in ("results", "slim_results", "summary", "failed_csv"):
        source = paths[key]
        if source.exists():
            shutil.copy2(source, backup_dir / source.name)
    return backup_dir


def restore_evaluation_outputs(
    paths: Mapping[str, Path],
    backup_dir: Path,
) -> None:
    for key in ("results", "slim_results", "summary", "failed_csv"):
        target = paths[key]
        backup = backup_dir / target.name
        if backup.exists():
            shutil.copy2(backup, target)


def rewrite_evaluation_outputs(
    *,
    paths: Mapping[str, Path],
    results: List[Dict[str, Any]],
    episode: int,
    target_label: str,
    recovery_audit: Dict[str, Any],
) -> None:
    slim_results = [build_slim_result(item) for item in results]
    summary = RegressionEvaluator.summarize(results).to_dict()
    summary = enrich_evaluation_summary(
        summary,
        slim_results,
        target_label=target_label,
    )
    previous_summary = read_json(paths["summary"], default={}) or {}
    summary["episode"] = episode
    summary["failed_csv"] = display_path(paths["failed_csv"])
    summary["run_timing"] = previous_summary.get("run_timing", {})
    summary["connection_error_recovery"] = {
        "applied_at": now_iso(),
        **recovery_audit,
    }

    atomic_write_json(paths["results"], results)
    atomic_write_json(paths["slim_results"], slim_results)
    failed_tmp = paths["failed_csv"].with_name(
        paths["failed_csv"].name + ".connection_recovery.tmp"
    )
    failed_count = write_failed_csv(
        str(failed_tmp),
        results,
        slim_results,
    )
    os.replace(failed_tmp, paths["failed_csv"])
    summary["failed_csv_count"] = failed_count
    atomic_write_json(paths["summary"], summary)


def atomic_write_json(path: Path, payload: Any) -> None:
    temp_path = path.with_name(path.name + ".connection_recovery.tmp")
    write_json(temp_path, payload)
    os.replace(temp_path, path)


def load_result_list(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON result list: {path}")
    return payload


def read_csv_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        return headers, [dict(row) for row in reader]


def write_csv_rows(
    path: Path,
    headers: Sequence[str],
    rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        writer.writerows(rows)


def find_tx_column(headers: Sequence[str]) -> str:
    for header in headers:
        if str(header).strip().lower() in TX_COLUMN_NAMES:
            return header
    raise ValueError(
        "CSV has no transaction hash column. Expected one of: "
        + ", ".join(sorted(TX_COLUMN_NAMES))
    )


def parse_archived_csv_paths(log_text: str) -> Dict[str, str]:
    match = re.search(
        r"^\[CSVArchive\] Archived split CSVs:\s+"
        r"pos=(?P<pos>.+?)\s+neg=(?P<neg>.+?)\s+negative_source_kind="
        r"(?P<kind>\S+)",
        log_text,
        flags=re.MULTILINE,
    )
    if not match:
        return {"pos_csv": "", "neg_csv": "", "negative_source_kind": ""}
    return {
        "pos_csv": match.group("pos").strip(),
        "neg_csv": match.group("neg").strip(),
        "negative_source_kind": match.group("kind").strip(),
    }


def first_matching_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    return ""


def parse_key_value_line(line: str) -> Dict[str, str]:
    return {
        key: value
        for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", line or "")
    }


def parse_bool(value: Any, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "enabled"}


def parse_int(value: Any, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def empty_if_disabled(value: str) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"disabled", "none"} else text


def normalize_tx_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if TX_PATTERN.fullmatch(text) else ""


def normalize_prefix(value: str) -> str:
    text = str(value or "").strip().strip("_")
    return f"{text}_" if text else "eval_"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT.resolve()))
    except (OSError, ValueError):
        return str(resolved)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
