param(
    [Parameter(Mandatory = $true)]
    [string[]]$Episodes,

    [ValidateSet("fewshot", "eval")]
    [string]$Phase = "fewshot",

    [string]$ResultsRoot = "data\results",
    [string]$Round = "latest",
    [string]$OutDir = "data\analysis",
    [string]$OutputPrefix = "",
    [switch]$IncludeUncertain,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "."
Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Python not found: .\.venv\Scripts\python.exe"
}

if ([string]::IsNullOrWhiteSpace($OutputPrefix)) {
    $OutputPrefix = "failure_mismatch_${Phase}"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$PythonScript = @'
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def expand_episodes(raw_values: Iterable[str]) -> List[int]:
    values: List[int] = []
    for raw in raw_values:
        for token in str(raw).replace(",", " ").split():
            token = token.strip()
            if not token:
                continue
            if ".." in token:
                left, right = token.split("..", 1)
                start = int(left)
                end = int(right)
                step = 1 if end >= start else -1
                values.extend(range(start, end + step, step))
            else:
                values.append(int(token))
    out: List[int] = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def result_files_for_episode(
    *,
    results_root: Path,
    episode: int,
    phase: str,
    round_selector: str,
) -> List[Path]:
    episode_dir = results_root / str(episode)
    if phase == "eval":
        candidates = [
            episode_dir / "eval_slim_results.json",
            episode_dir / "eval_results.json",
        ]
        return [path for path in candidates if path.exists()][:1]

    if not episode_dir.exists():
        return []
    round_dirs = sorted(
        item for item in episode_dir.glob("round_*") if item.is_dir()
    )
    if not round_dirs:
        candidates = [
            episode_dir / "slim_results.json",
            episode_dir / "current_results.json",
        ]
        return [path for path in candidates if path.exists()][:1]
    if round_selector == "latest":
        selected_dirs = [round_dirs[-1]]
    elif round_selector == "all":
        selected_dirs = round_dirs
    else:
        selected_dirs = [episode_dir / round_selector]
    files: List[Path] = []
    for round_dir in selected_dirs:
        candidates = [
            round_dir / "slim_results.json",
            round_dir / "current_results.json",
        ]
        files.extend([path for path in candidates if path.exists()][:1])
    return files


def as_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return [item for item in payload["results"] if isinstance(item, dict)]
        return [payload]
    return []


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def predicted_verdict(result: Dict[str, Any]) -> str:
    finding = result.get("finding_summary")
    if not isinstance(finding, dict):
        finding = {}
    return first_nonempty(
        result.get("predicted_verdict"),
        finding.get("verdict"),
        ((result.get("inference") or {}).get("finding") or {}).get("verdict")
        if isinstance(result.get("inference"), dict)
        else "",
    ).lower()


def ground_truth(result: Dict[str, Any]) -> str:
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    return first_nonempty(result.get("ground_truth"), evaluation.get("ground_truth")).lower()


def category(result: Dict[str, Any]) -> str:
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    return first_nonempty(
        result.get("attack_label"),
        result.get("target_label"),
        evaluation.get("target_label"),
        result.get("raw_ground_truth"),
        evaluation.get("raw_ground_truth"),
        "unknown",
    )


def condition_table(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    table = result.get("condition_table")
    if isinstance(table, list):
        return [item for item in table if isinstance(item, dict)]
    finding = result.get("finding_summary")
    if isinstance(finding, dict):
        aggregation = finding.get("verdict_aggregation")
        if isinstance(aggregation, dict):
            judge_values = aggregation.get("judge_values")
            if isinstance(judge_values, dict):
                return [
                    {
                        "id": key,
                        "condition_id": key,
                        "is_exclusion": str(key).upper().startswith("E"),
                        "answer": value,
                        "expected_answer": True,
                    }
                    for key, value in judge_values.items()
                ]
    return []


def normalize_answer(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def step_answer(step: Dict[str, Any]) -> bool | None:
    for key in ("final_answer", "answer", "first_round_answer"):
        if key in step:
            value = normalize_answer(step.get(key))
            if value is not None:
                return value
    return None


def step_id(step: Dict[str, Any]) -> str:
    return first_nonempty(step.get("condition_id"), step.get("id"), "unknown").upper()


def is_exclusion(step: Dict[str, Any], condition_id: str) -> bool:
    if "is_exclusion" in step:
        return bool(step.get("is_exclusion"))
    return condition_id.upper().startswith("E")


def error_type(result: Dict[str, Any]) -> str:
    gold = ground_truth(result)
    pred = predicted_verdict(result)
    if gold == "attack" and pred != "attack":
        return "fn"
    if gold == "benign" and pred == "attack":
        return "fp"
    return ""


def mismatches_for_failed_case(
    result: Dict[str, Any],
    *,
    include_uncertain: bool,
) -> Tuple[int, List[Tuple[str, str]]]:
    err = error_type(result)
    table = condition_table(result)
    mismatches: List[Tuple[str, str]] = []
    for step in table:
        cid = step_id(step)
        value = step_answer(step)
        exclusion = is_exclusion(step, cid)

        if err == "fn":
            if exclusion:
                if value is True:
                    mismatches.append((cid, "exclusion_true_on_attack"))
                elif value is None and include_uncertain:
                    mismatches.append((cid, "exclusion_uncertain_on_attack"))
            else:
                if value is False:
                    mismatches.append((cid, "core_false_on_attack"))
                elif value is None and include_uncertain:
                    mismatches.append((cid, "core_uncertain_on_attack"))
        elif err == "fp":
            if exclusion:
                if value is False:
                    mismatches.append((cid, "exclusion_false_on_benign"))
                elif value is None and include_uncertain:
                    mismatches.append((cid, "exclusion_uncertain_on_benign"))
            else:
                if value is True:
                    mismatches.append((cid, "core_true_on_benign"))
                elif value is None and include_uncertain:
                    mismatches.append((cid, "core_uncertain_on_benign"))
    return len(table), mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--phase", choices=["fewshot", "eval"], required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--round", default="latest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    episodes = expand_episodes(args.episodes)
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed_counter: Counter[Tuple[int, str, str, str, str]] = Counter()
    summary_counter: Counter[Tuple[int, str, int, int, str]] = Counter()
    inputs: Dict[int, List[str]] = {}
    skipped: Dict[int, str] = {}

    for episode in episodes:
        files = result_files_for_episode(
            results_root=results_root,
            episode=episode,
            phase=args.phase,
            round_selector=args.round,
        )
        inputs[episode] = [str(path) for path in files]
        if not files:
            skipped[episode] = "result_file_not_found"
            continue
        if args.dry_run:
            continue
        for path in files:
            for result in as_list(load_json(path)):
                err = error_type(result)
                if err not in {"fp", "fn"}:
                    continue
                cat = category(result)
                rule_count, mismatches = mismatches_for_failed_case(
                    result,
                    include_uncertain=bool(args.include_uncertain),
                )
                if not mismatches:
                    mismatches = [("(none)", "unattributed")]
                for rule_id, mismatch in mismatches:
                    failed_counter[(episode, cat, rule_id, mismatch, err)] += 1
                summary_counter[(episode, cat, rule_count, len(mismatches), err)] += 1

    failed_path = out_dir / f"{args.output_prefix}_failed.csv"
    summary_path = out_dir / f"{args.output_prefix}_failed_summary.csv"
    manifest_path = out_dir / f"{args.output_prefix}_manifest.json"

    if not args.dry_run:
        with failed_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["episode", "category", "rule", "mismatch", "count", "error"],
            )
            writer.writeheader()
            for (episode, cat, rule_id, mismatch, err), count in sorted(failed_counter.items()):
                writer.writerow({
                    "episode": episode,
                    "category": cat,
                    "rule": rule_id,
                    "mismatch": mismatch,
                    "count": count,
                    "error": err,
                })

        with summary_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["episode", "category", "rule_count", "mismatchs", "count", "error"],
            )
            writer.writeheader()
            for (episode, cat, rule_count, mismatch_count, err), count in sorted(summary_counter.items()):
                writer.writerow({
                    "episode": episode,
                    "category": cat,
                    "rule_count": rule_count,
                    "mismatchs": mismatch_count,
                    "count": count,
                    "error": err,
                })

    manifest = {
        "phase": args.phase,
        "episodes": episodes,
        "round": args.round,
        "include_uncertain": bool(args.include_uncertain),
        "inputs": inputs,
        "skipped": skipped,
        "outputs": {
            "failed": str(failed_path),
            "failed_summary": str(summary_path),
            "manifest": str(manifest_path),
        },
        "semantics": {
            "fn": "attack ground truth but predicted non-attack; mismatches are false/uncertain core conditions or true exclusions",
            "fp": "benign ground truth but predicted attack; mismatches are true core conditions or false/uncertain exclusions",
            "rule": "condition id such as C1/C2/C3/E1/E2",
            "mismatch": "condition-level error role",
            "mismatchs": "number of condition-level mismatches in one failed case",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'@

$TempScript = Join-Path $env:TEMP ("evotx_export_failure_mismatch_stats_{0}.py" -f ([guid]::NewGuid().ToString("N")))
Set-Content -Path $TempScript -Value $PythonScript -Encoding UTF8

try {
    $ArgsList = @(
        $TempScript,
        "--episodes"
    )
    $ArgsList += $Episodes
    $ArgsList += @(
        "--phase", $Phase,
        "--results-root", $ResultsRoot,
        "--round", $Round,
        "--out-dir", $OutDir,
        "--output-prefix", $OutputPrefix
    )
    if ($IncludeUncertain) {
        $ArgsList += "--include-uncertain"
    }
    if ($DryRun) {
        $ArgsList += "--dry-run"
    }

    Write-Host "[ExportFailureMismatch] phase=$Phase episodes=$($Episodes -join ',') round=$Round"
    & .\.venv\Scripts\python.exe @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Export script failed with code $LASTEXITCODE"
    }
}
finally {
    if (Test-Path $TempScript) {
        Remove-Item -LiteralPath $TempScript -Force
    }
}
