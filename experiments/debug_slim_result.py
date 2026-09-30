"""Inspect EvoTx full-result slimming without calling any LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.utils.json_utils import read_json
from evotx.utils.result_slimmer import (
    build_reviewer_case,
    build_slim_result,
    reviewer_case_contains_full_evidence_views,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print size and structure diagnostics for a slim review input."
    )
    parser.add_argument("--result", required=True, help="Path to a full result JSON file")
    args = parser.parse_args()

    result_path = Path(args.result)
    result = read_json(result_path)
    if result is None:
        raise FileNotFoundError(result_path)

    slim = build_slim_result(result)
    reviewer_case = build_reviewer_case(result)

    print(f"result_path: {result_path}")
    print(f"full_result_size: {_json_size(result)} bytes")
    print(f"slim_result_size: {_json_size(slim)} bytes")
    print(f"reviewer_case_size: {_json_size(reviewer_case)} bytes")
    print(
        "review_input_contains_full_evidence_views: "
        f"{reviewer_case_contains_full_evidence_views(reviewer_case)}"
    )

    print("\ncondition_table:")
    for row in slim.get("condition_table", []):
        print(
            f"  - {row.get('id')} condition={row.get('condition_id')} "
            f"answer={row.get('answer')} confidence={row.get('confidence')} "
            f"missing={len(row.get('missing_evidence', []))} "
            f"tools={len(row.get('tool_observations', []))} "
            f"hint={row.get('error_hint', '')}"
        )

    print("\nmissing_evidence_by_condition:")
    print(json.dumps(slim.get("missing_evidence_by_condition", {}), ensure_ascii=False, indent=2))

    print("\ntool_call_summary:")
    print(json.dumps(slim.get("tool_call_summary", []), ensure_ascii=False, indent=2))

    print("\nevidence_id_resolution:")
    print(json.dumps(slim.get("evidence_id_resolution", {}), ensure_ascii=False, indent=2))

    print("\nmissing_evidence_actionability_summary:")
    print(json.dumps(slim.get("missing_evidence_actionability_summary", {}), ensure_ascii=False, indent=2))


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
