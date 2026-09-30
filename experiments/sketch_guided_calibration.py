from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.core.schemas import EvolvingRule
from evotx.utils.json_utils import read_json, stable_json_dumps, write_json
from evotx.utils.result_utils import get_finding
from experiments.run_inference import run_single_inference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional stage: run sketch-guided calibration on an unlabeled disjoint transaction pool."
    )
    parser.add_argument("--pool-file", required=True)
    parser.add_argument("--rule-file", required=True)
    parser.add_argument("--tool-manifest", default="configs/tool_manifest.json")
    parser.add_argument(
        "--llm-provider",
        help="Optional provider override for calibration LLMs: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument("--planner-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--env-model")
    parser.add_argument("--use-environment", action="store_true")
    parser.add_argument("--base-cache-dir", default="data/cache")
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
    parser.add_argument("--results-out", default="data/results/calibration_results.json")
    parser.add_argument("--out", default="data/results/calibration_summary.json")
    args = parser.parse_args()

    rule = EvolvingRule.from_dict(read_json(args.rule_file))
    tool_manifest = read_json(args.tool_manifest, default={}) or {}
    pool = load_pool(args.pool_file)

    results: List[Dict[str, Any]] = []
    for item in pool:
        result = run_single_inference(
            tx_hash=item["tx_hash"],
            chain=item["chain"],
            rule=rule,
            rule_source=args.rule_file,
            tool_manifest=tool_manifest,
            tx_context=item.get("tx_context", {}) or {},
            static_evidence=item.get("static_evidence", {}) or {},
            llm_provider=args.llm_provider,
            planner_model=args.planner_model,
            judge_model=args.judge_model,
            env_model=args.env_model,
            use_environment=args.use_environment,
            base_cache_dir=args.base_cache_dir,
            max_view_chars=args.max_view_chars,
            max_context_chars=args.max_context_chars,
            ground_truth=None,
            raw_ground_truth=None,
            case_metadata=item.get("metadata", {}),
        )
        results.append(result)

    summary = summarize_verdicts(results)
    write_json(args.results_out, results)
    write_json(args.out, summary)

    print("Optional calibration stage complete.")
    print(f"Pool size: {summary['total']}")
    print(f"Summary: {stable_json_dumps(summary)}")
    print(f"Saved calibration results: {args.results_out}")
    print(f"Saved calibration summary: {args.out}")


def load_pool(path_like: str) -> List[Dict[str, Any]]:
    data = read_json(path_like, default=[])
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError("--pool-file must contain a JSON list or an object with a 'cases' list.")

    pool: List[Dict[str, Any]] = []
    for index, raw in enumerate(data, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Pool item #{index} is not a JSON object.")
        tx_hash = str(raw.get("tx_hash", "")).strip()
        if not tx_hash:
            raise ValueError(f"Pool item #{index} is missing tx_hash.")
        pool.append(
            {
                "tx_hash": tx_hash,
                "chain": str(raw.get("chain", "eth")),
                "tx_context": dict(raw.get("tx_context", {})),
                "static_evidence": dict(raw.get("static_evidence", {})),
                "metadata": dict(raw.get("metadata", {})),
            }
        )
    return pool


def summarize_verdicts(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"attack": 0, "benign": 0, "uncertain": 0}
    for result in results:
        verdict = get_finding(result).get("verdict", "uncertain")
        counts[verdict] = counts.get(verdict, 0) + 1
    return {
        "total": len(results),
        **counts,
    }


if __name__ == "__main__":
    main()
