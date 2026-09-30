from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.evolution.regression import RegressionEvaluator
from evotx.utils.json_utils import read_json, stable_json_dumps, write_json


STRICT_ARGS = SimpleNamespace(
    max_fp_increase=0,
    max_fn_increase=1,
    max_uncertain_increase=0,
    max_uncertain_benign_increase=0,
    max_guard_fp_increase=1,
    max_guard_error_increase=1,
    max_guard_uncertain_increase=1,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrospectively validate few-shot round candidates using cached "
            "current/candidate results, without calling any LLM judge."
        )
    )
    parser.add_argument(
        "--episode",
        required=True,
        help=(
            "Episode id under data/results. Also accepts ranges/lists such as "
            "200..600, 200,225,300, or 200..205,225."
        ),
    )
    parser.add_argument("--attack-label", default="", help="Artifact prefix, e.g. access_control.")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--rules-dir", default="data/rules")
    parser.add_argument("--plans-dir", default="data/plans")
    parser.add_argument(
        "--candidate",
        default="",
        help="Optional candidate name/path fragment filter, e.g. rule_candidate.",
    )
    parser.add_argument(
        "--round",
        dest="round_filter",
        default="",
        help="Optional round filter such as 1, 01, or round_01.",
    )
    parser.add_argument(
        "--write-artifacts",
        dest="write_artifacts",
        action="store_true",
        default=True,
        help="Write passed retrospective rule/plan copies. Enabled by default.",
    )
    parser.add_argument(
        "--no-write-artifacts",
        dest="write_artifacts",
        action="store_false",
        help="Only write retrospective reports, not rule/plan copies.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="In batch mode, fail instead of skipping missing episode directories.",
    )
    args = parser.parse_args()

    episodes = parse_episode_specs(str(args.episode))
    if len(episodes) > 1:
        run_batch(args, episodes)
        return
    process_episode(args, episodes[0], fail_on_missing=True)


def parse_episode_specs(spec: str) -> List[str]:
    episodes: List[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;\s]+", str(spec or "").strip()):
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\.\.(\d+)", part)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            step = 1 if end >= start else -1
            values = range(start, end + step, step)
            for value in values:
                text = str(value)
                if text not in seen:
                    seen.add(text)
                    episodes.append(text)
            continue
        text = part.strip()
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            episodes.append(text)
    if not episodes:
        raise ValueError("No episode ids parsed from --episode.")
    return episodes


def run_batch(args: argparse.Namespace, episodes: List[str]) -> None:
    batch_reports: List[Dict[str, Any]] = []
    for episode in episodes:
        try:
            summary = process_episode(
                args,
                episode,
                fail_on_missing=bool(args.fail_on_missing),
            )
            if summary is None:
                batch_reports.append({
                    "episode": episode,
                    "status": "skipped_missing",
                    "reason": f"{Path(args.results_dir) / episode} not found",
                })
            else:
                batch_reports.append({
                    "episode": episode,
                    "status": "processed",
                    "candidate_count": summary.get("candidate_count", 0),
                    "passed_count": summary.get("passed_count", 0),
                    "failed_count": summary.get("failed_count", 0),
                    "summary_path": summary.get("summary_path", ""),
                })
        except Exception as exc:
            if args.fail_on_missing:
                raise
            batch_reports.append({
                "episode": episode,
                "status": "error",
                "reason": str(exc),
            })

    batch_summary = {
        "episode_spec": str(args.episode),
        "episode_count": len(episodes),
        "processed_count": sum(1 for item in batch_reports if item.get("status") == "processed"),
        "skipped_missing_count": sum(1 for item in batch_reports if item.get("status") == "skipped_missing"),
        "error_count": sum(1 for item in batch_reports if item.get("status") == "error"),
        "artifact_policy": (
            "passed_candidates_written_as_named_copies_and_best_alias"
            if args.write_artifacts
            else "report_only"
        ),
        "reports": batch_reports,
    }
    batch_path = (
        Path(args.results_dir)
        / f"retrospective_batch_{sanitize_name(str(args.episode))}.json"
    )
    write_json(batch_path, batch_summary)
    print(f"[Retrospective] Batch wrote {batch_path}")
    print(
        "[Retrospective] Batch summary: "
        f"{stable_json_dumps({k: batch_summary[k] for k in ['episode_count', 'processed_count', 'skipped_missing_count', 'error_count']})}"
    )


def process_episode(
    args: argparse.Namespace,
    episode: str,
    *,
    fail_on_missing: bool = False,
) -> Dict[str, Any] | None:
    results_root = Path(args.results_dir) / episode
    if not results_root.exists():
        if fail_on_missing:
            raise FileNotFoundError(f"Episode results not found: {results_root}")
        print(f"[Retrospective] Skip missing episode {episode}: {results_root}")
        return None

    attack_label = infer_attack_label(results_root, args.attack_label)
    retrospective_dir = results_root / "retrospective"
    retrospective_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(
        iter_candidate_records(
            results_root,
            candidate_filter=str(args.candidate or ""),
            round_filter=str(args.round_filter or ""),
        )
    )
    evaluator = RegressionEvaluator()
    reports: List[Dict[str, Any]] = []

    for record in candidates:
        report = validate_candidate(record, evaluator=evaluator)
        if report.get("passed") and args.write_artifacts:
            report["retrospective_artifacts"] = write_retrospective_artifacts(
                record=record,
                report=report,
                attack_label=attack_label,
                rules_dir=Path(args.rules_dir) / episode,
                plans_dir=Path(args.plans_dir) / episode,
            )
        report_path = (
            retrospective_dir
            / f"{record['round_name']}__{record['candidate_name']}__retrospective.json"
        )
        write_json(report_path, report)
        report["report_path"] = str(report_path)
        reports.append(report)

    selected_retrospective_artifacts: Dict[str, Any] = {}
    if args.write_artifacts:
        selected = select_best_passed_report(reports)
        if selected:
            selected_retrospective_artifacts = write_retrospective_alias_artifacts(
                report=selected,
                attack_label=attack_label,
                rules_dir=Path(args.rules_dir) / episode,
                plans_dir=Path(args.plans_dir) / episode,
            )

    summary = {
        "episode": episode,
        "attack_label": attack_label,
        "result_source": "cached_round_candidate_results",
        "candidate_count": len(reports),
        "passed_count": sum(1 for item in reports if item.get("passed")),
        "failed_count": sum(1 for item in reports if not item.get("passed")),
        "selected_retrospective_artifacts": selected_retrospective_artifacts,
        "artifact_policy": (
            "passed_candidates_written_as_named_copies_and_best_alias"
            if args.write_artifacts
            else "report_only"
        ),
        "note": (
            "Retrospective artifacts are not latest aliases. Each passed pair "
            "is named by round and candidate, and the best passed pair is also "
            "materialized as <label>__retrospective.json and "
            "<label>__plan_retrospective.json for eval fallback."
        ),
        "reports": reports,
    }
    summary_path = retrospective_dir / "retrospective_validation.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    print(f"[Retrospective] Wrote {summary_path}")
    print(
        "[Retrospective] Summary: "
        f"{stable_json_dumps({k: summary[k] for k in ['candidate_count', 'passed_count', 'failed_count']})}"
    )
    return summary


def infer_attack_label(results_root: Path, provided: str) -> str:
    if provided.strip():
        return sanitize_name(provided.strip())
    final_report = read_json(results_root / "final_report.json", default={}) or {}
    for key in ("attack_label", "label", "positive_label"):
        value = str(final_report.get(key) or "").strip()
        if value:
            return sanitize_name(value)
    detector_context = final_report.get("detector_context")
    if isinstance(detector_context, dict):
        value = str(detector_context.get("attack_label") or "").strip()
        if value:
            return sanitize_name(value)
        for nested_key in ("initial_rule", "final_rule"):
            nested = detector_context.get(nested_key)
            if isinstance(nested, dict):
                value = str(
                    nested.get("attack_label")
                    or nested.get("raw_attack_label")
                    or ""
                ).strip()
                if value:
                    return sanitize_name(value)
    return "retrospective"


def iter_candidate_records(
    results_root: Path,
    *,
    candidate_filter: str = "",
    round_filter: str = "",
) -> Iterable[Dict[str, Any]]:
    wanted_round = normalize_round_filter(round_filter)
    wanted_candidate = candidate_filter.lower().strip()
    seen: set[Path] = set()
    for round_dir in sorted(results_root.glob("round_*")):
        if not round_dir.is_dir():
            continue
        if wanted_round and round_dir.name != wanted_round:
            continue
        candidate_dirs = [round_dir]
        candidate_dirs.extend(
            path.parent
            for path in sorted(round_dir.glob("*/*/candidate_artifact.json"))
            if path.is_file()
        )
        for candidate_dir in candidate_dirs:
            artifact_path = candidate_dir / "candidate_artifact.json"
            if not artifact_path.exists():
                continue
            resolved = artifact_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidate_name = (
                str((read_json(artifact_path, default={}) or {}).get("name") or "")
                or candidate_dir.name
            )
            if candidate_dir == round_dir:
                candidate_name = "selected_candidate"
            if wanted_candidate and wanted_candidate not in str(candidate_dir).lower() and wanted_candidate not in candidate_name.lower():
                continue
            yield {
                "round_dir": round_dir,
                "round_name": round_dir.name,
                "candidate_dir": candidate_dir,
                "candidate_name": sanitize_name(candidate_name),
                "artifact_path": artifact_path,
            }


def normalize_round_filter(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("round_"):
        return text
    try:
        return f"round_{int(text):02d}"
    except ValueError:
        return text


def validate_candidate(record: Dict[str, Any], *, evaluator: RegressionEvaluator) -> Dict[str, Any]:
    round_dir = Path(record["round_dir"])
    candidate_dir = Path(record["candidate_dir"])
    artifact = read_json(record["artifact_path"], default={}) or {}

    current_full = read_json(round_dir / "full_results.json", default=[])
    candidate_full = read_json(candidate_dir / "candidate_full_results.json", default=[])
    current_guard_full = read_json(round_dir / "current_guard_full_results.json", default=[])
    candidate_guard_full = read_json(
        candidate_dir / "candidate_guard_full_results.json",
        default=[],
    )

    current_summary = read_json(round_dir / "current_summary.json", default={}) or {}
    candidate_summary = read_json(candidate_dir / "candidate_summary.json", default={}) or {}
    current_guard_summary = read_json(
        round_dir / "current_guard_summary.json",
        default={},
    ) or {}
    candidate_guard_summary = read_json(
        candidate_dir / "candidate_guard_summary.json",
        default={},
    ) or {}

    if isinstance(current_full, list) and isinstance(candidate_full, list) and current_full and candidate_full:
        if current_guard_full or candidate_guard_full:
            comparison = evaluator.compare_with_guard(
                current_full,
                candidate_full,
                old_guard_results=current_guard_full or [],
                new_guard_results=candidate_guard_full or [],
                max_fp_increase=STRICT_ARGS.max_fp_increase,
                max_fn_increase=STRICT_ARGS.max_fn_increase,
                max_uncertain_increase=STRICT_ARGS.max_uncertain_increase,
                max_uncertain_benign_increase=STRICT_ARGS.max_uncertain_benign_increase,
                max_guard_fp_increase=STRICT_ARGS.max_guard_fp_increase,
                max_guard_error_increase=STRICT_ARGS.max_guard_error_increase,
                max_guard_uncertain_increase=STRICT_ARGS.max_guard_uncertain_increase,
                acceptance_mode=str(artifact.get("update_kind") or "rule"),
            )
        else:
            comparison = evaluator.compare(
                current_full,
                candidate_full,
                max_fp_increase=STRICT_ARGS.max_fp_increase,
                max_fn_increase=STRICT_ARGS.max_fn_increase,
                max_uncertain_increase=STRICT_ARGS.max_uncertain_increase,
                max_uncertain_benign_increase=STRICT_ARGS.max_uncertain_benign_increase,
                acceptance_mode=str(artifact.get("update_kind") or "rule"),
            )
        comparison_source = "full_results"
    else:
        comparison = compare_summaries_with_optional_guard(
            current_summary=current_summary,
            candidate_summary=candidate_summary,
            current_guard_summary=current_guard_summary,
            candidate_guard_summary=candidate_guard_summary,
        )
        comparison_source = "summary_only"

    return {
        "round": record["round_name"],
        "candidate_name": record["candidate_name"],
        "candidate_dir": str(candidate_dir),
        "update_kind": artifact.get("update_kind", ""),
        "candidate_strategy": artifact.get("candidate_strategy", ""),
        "candidate_artifact": artifact,
        "comparison_source": comparison_source,
        "strict_policy": {
            "max_fp_increase": 0,
            "max_uncertain_increase": 0,
            "max_uncertain_benign_increase": 0,
            "max_guard_fp_increase": 1,
            "max_guard_error_increase": 1,
            "max_guard_uncertain_increase": 1,
        },
        "current_summary": current_summary,
        "candidate_summary": candidate_summary,
        "current_guard_summary": current_guard_summary,
        "candidate_guard_summary": candidate_guard_summary,
        "comparison": comparison,
        "passed": bool(comparison.get("accept")),
        "pass_reason": comparison.get("accept_reason", ""),
        "reject_reason": comparison.get("reject_reason", ""),
    }


def compare_summaries_with_optional_guard(
    *,
    current_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
    current_guard_summary: Dict[str, Any],
    candidate_guard_summary: Dict[str, Any],
) -> Dict[str, Any]:
    train = compare_summary_pair(current_summary, candidate_summary)
    result = dict(train)
    result["train_comparison"] = train
    result["guard_comparison"] = {}
    result["guard_gate"] = {"enabled": False, "accept": True, "reject_reason": ""}
    if current_guard_summary or candidate_guard_summary:
        guard = compare_summary_pair(current_guard_summary, candidate_guard_summary)
        guard_delta = dict(guard.get("delta") or {})
        violations = []
        for key in ("fp", "errors", "uncertain"):
            delta = int(guard_delta.get(key, 0) or 0)
            if delta > 0:
                violations.append(f"guard {key} increased by {delta}, allowed 0")
        guard_accept = not violations
        result["guard_comparison"] = guard
        result["guard_gate"] = {
            "enabled": True,
            "accept": guard_accept,
            "reject_reason": "; ".join(violations),
            "delta": guard_delta,
        }
        result["accept"] = bool(train.get("accept")) and guard_accept
        if not result["accept"] and train.get("accept"):
            result["accept_reason"] = ""
            result["reject_reason"] = (
                "Rejected by hard-negative guard: "
                f"{result['guard_gate']['reject_reason']}"
            )
    return result


def compare_summary_pair(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    old_metrics = normalize_summary(old)
    new_metrics = normalize_summary(new)
    delta = {
        key: int(new_metrics.get(key, 0) or 0) - int(old_metrics.get(key, 0) or 0)
        for key in ("errors", "fp", "fn", "uncertain", "uncertain_benign")
    }
    old_recall = float(old_metrics.get("attack_recall_for_gate", 0.0) or 0.0)
    new_recall = float(new_metrics.get("attack_recall_for_gate", 0.0) or 0.0)
    old_precision = float(old_metrics.get("attack_precision_for_gate", 0.0) or 0.0)
    new_precision = float(new_metrics.get("attack_precision_for_gate", 0.0) or 0.0)
    recall_delta = new_recall - old_recall
    precision_delta = new_precision - old_precision
    accept = False
    accept_reason = ""
    reject_reason = ""
    eps = 1e-12
    if recall_delta < -eps:
        reject_reason = (
            f"attack_recall decreased from {old_recall:.6f} "
            f"to {new_recall:.6f}."
        )
    elif precision_delta < -eps:
        reject_reason = (
            f"attack_precision decreased from {old_precision:.6f} "
            f"to {new_precision:.6f}."
        )
    elif recall_delta > eps or precision_delta > eps:
        accept = True
        accept_reason = (
            "Accepted by attack metric gate: "
            f"recall {old_recall:.6f}->{new_recall:.6f}, "
            f"precision {old_precision:.6f}->{new_precision:.6f}."
        )
    else:
        reject_reason = (
            "attack_recall and attack_precision did not improve "
            f"(recall {old_recall:.6f}->{new_recall:.6f}, "
            f"precision {old_precision:.6f}->{new_precision:.6f})."
        )
    return {
        "old": old_metrics,
        "new": new_metrics,
        "delta": delta,
        "metric_delta": {
            "attack_recall": recall_delta,
            "attack_precision": precision_delta,
        },
        "accept": accept,
        "accept_reason": accept_reason,
        "reject_reason": reject_reason,
        "acceptance_policy": {
            "mode": "summary_only_attack_metric",
            "primary_metric": "attack_recall + attack_precision",
            "note": (
                "Per-case changes are unavailable; decision uses cached summaries "
                "and accepts only if attack_recall/attack_precision do not decline "
                "and at least one improves. FP/FN/uncertain counts are diagnostic."
            ),
        },
    }


def normalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    uncertain_by_gold = summary.get("uncertain_by_gold")
    if not isinstance(uncertain_by_gold, dict):
        uncertain_by_gold = {}
    negative_breakdown = summary.get("negative_breakdown")
    if not isinstance(negative_breakdown, dict):
        negative_breakdown = {}
    negative_total = (
        int(negative_breakdown.get("benign", 0) or 0)
        + int(negative_breakdown.get("other_attack", 0) or 0)
        + int(negative_breakdown.get("unknown_other", 0) or 0)
    )
    total = int(summary.get("total", 0) or 0)
    fn = int(summary.get("fn", 0) or 0)
    fp = int(summary.get("fp", 0) or 0)
    uncertain_attack = int(uncertain_by_gold.get("attack", 0) or 0)
    target_total = int(summary.get("attack_target_total", 0) or 0)
    if target_total <= 0:
        target_total = max(0, total - negative_total)
    attack_tp = int(summary.get("attack_tp", 0) or 0)
    if attack_tp <= 0:
        attack_tp = max(0, target_total - fn - uncertain_attack)
    predicted_attack = int(summary.get("attack_predicted_total", 0) or 0)
    if predicted_attack <= 0:
        predicted_attack = max(0, attack_tp + fp)
    attack_recall = (
        float(summary.get("attack_recall"))
        if summary.get("attack_recall") is not None
        else (attack_tp / target_total if target_total else None)
    )
    attack_precision = (
        float(summary.get("attack_precision"))
        if summary.get("attack_precision") is not None
        else (attack_tp / predicted_attack if predicted_attack else None)
    )
    return {
        "total": total,
        "correct": int(summary.get("correct", 0) or 0),
        "errors": int(summary.get("errors", 0) or 0),
        "fp": fp,
        "fn": fn,
        "uncertain": int(summary.get("uncertain", 0) or 0),
        "uncertain_attack": uncertain_attack,
        "uncertain_benign": int(uncertain_by_gold.get("benign", 0) or 0),
        "negative_total": negative_total,
        "attack_tp": attack_tp,
        "attack_target_total": target_total,
        "attack_predicted_total": predicted_attack,
        "attack_recall": attack_recall,
        "attack_precision": attack_precision,
        "attack_recall_for_gate": 0.0 if attack_recall is None else attack_recall,
        "attack_precision_for_gate": (
            0.0 if attack_precision is None else attack_precision
        ),
    }


def write_retrospective_artifacts(
    *,
    record: Dict[str, Any],
    report: Dict[str, Any],
    attack_label: str,
    rules_dir: Path,
    plans_dir: Path,
) -> Dict[str, Any]:
    candidate_dir = Path(record["candidate_dir"])
    artifact = report.get("candidate_artifact") or {}
    suffix = f"{record['round_name']}__{record['candidate_name']}"
    suffix = sanitize_name(suffix)
    rules_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    rule_source = (
        (((artifact.get("rule") or {}) if isinstance(artifact.get("rule"), dict) else {}).get("rule_source"))
        or str(candidate_dir / "candidate_rule.json")
    )
    plan_source = (
        (((artifact.get("plan") or {}) if isinstance(artifact.get("plan"), dict) else {}).get("plan_source"))
        or str(candidate_dir / "candidate_plan.json")
    )
    out: Dict[str, Any] = {
        "policy": "retrospective_copy_no_latest_promotion",
        "source_candidate_dir": str(candidate_dir),
        "update_kind": report.get("update_kind", ""),
    }
    if rule_source and Path(rule_source).exists():
        rule_data = read_json(rule_source, default={}) or {}
        rule_path = rules_dir / f"{attack_label}__retrospective_{suffix}.json"
        write_json(rule_path, rule_data)
        out["rule_path"] = str(rule_path)
    else:
        out["rule_missing"] = str(rule_source or "")
    if plan_source and Path(plan_source).exists():
        plan_data = read_json(plan_source, default={}) or {}
        plan_path = plans_dir / f"{attack_label}__plan_retrospective_{suffix}.json"
        write_json(plan_path, plan_data)
        out["plan_path"] = str(plan_path)
    else:
        out["plan_missing"] = str(plan_source or "")
    return out


def select_best_passed_report(reports: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    passed = [
        report
        for report in reports
        if report.get("passed")
        and isinstance(report.get("retrospective_artifacts"), dict)
    ]
    if not passed:
        return None
    return sorted(passed, key=retrospective_selection_key)[0]


def retrospective_selection_key(report: Dict[str, Any]) -> tuple:
    metrics = retrospective_candidate_metrics(report)
    round_number = parse_round_number(str(report.get("round") or ""))
    candidate_name = str(report.get("candidate_name") or "")
    return (
        metrics.get("errors", 10**9),
        metrics.get("fp", 10**9),
        metrics.get("fn", 10**9),
        metrics.get("uncertain", 10**9),
        -round_number,
        candidate_name,
    )


def retrospective_candidate_metrics(report: Dict[str, Any]) -> Dict[str, int]:
    comparison = report.get("comparison")
    if isinstance(comparison, dict):
        new_metrics = comparison.get("new")
        if isinstance(new_metrics, dict) and new_metrics:
            return normalize_summary(new_metrics)
    candidate_summary = report.get("candidate_summary")
    if isinstance(candidate_summary, dict):
        return normalize_summary(candidate_summary)
    return normalize_summary({})


def parse_round_number(value: str) -> int:
    match = re.search(r"(\d+)", str(value or ""))
    return int(match.group(1)) if match else 0


def write_retrospective_alias_artifacts(
    *,
    report: Dict[str, Any],
    attack_label: str,
    rules_dir: Path,
    plans_dir: Path,
) -> Dict[str, Any]:
    artifacts = report.get("retrospective_artifacts")
    if not isinstance(artifacts, dict):
        return {}
    rules_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {
        "policy": "best_retrospective_alias_no_latest_promotion",
        "source_round": report.get("round", ""),
        "source_candidate": report.get("candidate_name", ""),
        "source_candidate_dir": report.get("candidate_dir", ""),
        "update_kind": report.get("update_kind", ""),
        "selection_metrics": retrospective_candidate_metrics(report),
        "source_named_artifacts": dict(artifacts),
    }

    rule_source = str(artifacts.get("rule_path") or "")
    if rule_source and Path(rule_source).exists():
        rule_alias = rules_dir / f"{attack_label}__retrospective.json"
        write_json(rule_alias, read_json(rule_source, default={}) or {})
        out["rule_path"] = str(rule_alias)
    else:
        out["rule_missing"] = rule_source

    plan_source = str(artifacts.get("plan_path") or "")
    if plan_source and Path(plan_source).exists():
        plan_alias = plans_dir / f"{attack_label}__plan_retrospective.json"
        write_json(plan_alias, read_json(plan_source, default={}) or {})
        out["plan_path"] = str(plan_alias)
    else:
        out["plan_missing"] = plan_source
    return out


def sanitize_name(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "candidate"


if __name__ == "__main__":
    main()
