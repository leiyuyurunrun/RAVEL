from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.adapters.llm_adapter import OpenAICompatibleLLM
from evotx.core.schemas import EvolvingRule
from evotx.evolution.regression import RegressionEvaluator
from evotx.evolution.negative_training import resolve_negative_training_mode
from evotx.evolution.reviewer import RuleReviewer, infer_error_type
from evotx.utils.json_utils import read_json, stable_json_dumps, write_json
from evotx.utils.result_slimmer import (
    build_case_boundary_context,
    build_cohort_signal_summary,
    build_plan_evidence_audit,
    build_reviewer_case,
    build_round_review_bundle,
    build_slim_result,
)
from evotx.utils.result_utils import (
    get_finding,
    get_ground_truth,
    get_raw_ground_truth,
    get_tx_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run only the EvoTx reviewer over existing full_results/slim_results. "
            "This does not run packet inference, updater, candidate evaluation, or acceptance."
        )
    )
    parser.add_argument(
        "--full-results",
        help="Existing full_results.json from train_fewshot/evaluate.",
    )
    parser.add_argument(
        "--slim-results",
        help=(
            "Optional existing slim_results.json. Used for review if --full-results "
            "is omitted, or for review_bundle slim count/context when supplied."
        ),
    )
    parser.add_argument(
        "--rule-file",
        required=True,
        help="Rule JSON used to populate review_bundle.current_rule.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for reviews.json and review_bundle.json.",
    )
    parser.add_argument("--reviews-out", help="Override reviews output path.")
    parser.add_argument("--review-bundle-out", help="Override review bundle output path.")
    parser.add_argument("--llm-model", default="glm-5.1")
    parser.add_argument(
        "--llm-provider",
        help="Optional provider override: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument("--review-model", help="Optional reviewer model override.")
    parser.add_argument(
        "--negative-training-mode",
        choices=["auto", "benign_only", "attack_contrastive", "mixed"],
        default="auto",
        help=(
            "Negative-set semantics exposed to the reviewer and stored in the "
            "review bundle. auto infers from negative_kind metadata."
        ),
    )
    parser.add_argument(
        "--save-llm-transcripts",
        action="store_true",
        help="Save reviewer prompt/raw completion/parsed response audit JSON.",
    )
    parser.add_argument(
        "--llm-transcripts-dir",
        help="Transcript directory. Defaults to <out-dir>/llm_transcripts when enabled.",
    )
    parser.add_argument(
        "--only-tx",
        action="append",
        default=[],
        help="Limit review to a tx hash. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=0,
        help="Optional cap on reviewed error cases; 0 means no cap.",
    )
    parser.add_argument(
        "--max-review-prompt-chars",
        type=int,
        default=0,
        help=(
            "Optional reviewer prompt character cap. Default 0 keeps existing "
            "behavior with no reviewer prompt compaction."
        ),
    )
    parser.add_argument(
        "--review-compact-mode",
        choices=["auto", "legacy", "error-focused"],
        default="error-focused",
        help=(
            "Reviewer prompt compact strategy. error-focused preserves wrong "
            "condition rows, condition_feature_analysis, and IV stateful state."
        ),
    )
    args = parser.parse_args()

    if not args.full_results and not args.slim_results:
        raise SystemExit("Provide --full-results or --slim-results.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reviews_out = Path(args.reviews_out) if args.reviews_out else out_dir / "reviews.json"
    bundle_out = (
        Path(args.review_bundle_out)
        if args.review_bundle_out
        else out_dir / "review_bundle.json"
    )
    transcript_dir = None
    if args.save_llm_transcripts:
        transcript_dir = Path(args.llm_transcripts_dir) if args.llm_transcripts_dir else out_dir / "llm_transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)

    full_results = _read_list(args.full_results) if args.full_results else []
    slim_results = _read_list(args.slim_results) if args.slim_results else []
    review_inputs = full_results or slim_results
    if not slim_results:
        slim_results = [build_slim_result(item) for item in review_inputs]
    requested_negative_training_mode = args.negative_training_mode
    negative_training_mode = resolve_negative_training_mode(
        requested_negative_training_mode,
        slim_results,
    )
    negative_training_mode_source = (
        "auto_inferred"
        if requested_negative_training_mode == "auto"
        else "explicit_cli"
    )

    rule = EvolvingRule.from_dict(read_json(args.rule_file))
    current_summary = RegressionEvaluator.summarize(review_inputs).to_dict()
    reviewer = RuleReviewer(
        llm=OpenAICompatibleLLM(
            model=args.review_model or args.llm_model,
            provider=args.llm_provider,
            minimax_thinking="adaptive",
        ),
        transcript_dir=transcript_dir,
        max_prompt_chars=args.max_review_prompt_chars,
        compact_mode=args.review_compact_mode,
        negative_training_mode=negative_training_mode,
    )

    cohort_signal_summary = build_cohort_signal_summary(slim_results)
    plan_evidence_audit = build_plan_evidence_audit(slim_results)
    case_boundary_context = build_case_boundary_context(
        slim_results,
        cohort_signal_summary=cohort_signal_summary,
        plan_evidence_audit=plan_evidence_audit,
    )
    reviews = _build_reviews(
        review_inputs,
        reviewer,
        cohort_signal_summary=cohort_signal_summary,
        plan_evidence_audit=plan_evidence_audit,
        case_boundary_context=case_boundary_context,
        only_tx_hashes={str(tx).lower() for tx in args.only_tx if str(tx).strip()},
        max_reviews=int(args.max_reviews or 0),
    )
    review_bundle = build_round_review_bundle(
        rule,
        current_summary,
        reviews,
        slim_results,
        negative_training_mode=negative_training_mode,
        negative_training_mode_source=negative_training_mode_source,
        cohort_signal_summary=cohort_signal_summary,
        plan_evidence_audit=plan_evidence_audit,
        case_boundary_context=case_boundary_context,
    )
    review_bundle["source"] = {
        "kind": "existing_results_review_only",
        "full_results": str(args.full_results or ""),
        "slim_results": str(args.slim_results or ""),
        "rule_file": str(args.rule_file),
    }

    write_json(reviews_out, reviews)
    write_json(bundle_out, review_bundle)
    write_json(out_dir / "case_boundary_context.json", case_boundary_context)

    print("[ReviewExisting] Completed reviewer-only pass.")
    print(f"[ReviewExisting] Results reviewed: {len(review_inputs)}")
    print(f"[ReviewExisting] Error reviews written: {len(reviews)}")
    print(f"[ReviewExisting] Current summary: {stable_json_dumps(current_summary)}")
    print(
        "[ReviewExisting] Negative training mode: "
        f"{negative_training_mode} ({negative_training_mode_source})"
    )
    print(f"[ReviewExisting] Reviews: {reviews_out}")
    print(f"[ReviewExisting] Review bundle: {bundle_out}")
    if transcript_dir:
        print(f"[ReviewExisting] Reviewer transcripts: {transcript_dir / 'reviewer'}")


def _read_list(path: str | None) -> List[Dict[str, Any]]:
    data = read_json(path, default=[])
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list at {path}")
    return [dict(item or {}) for item in data if isinstance(item, dict)]


def _build_reviews(
    results: List[Dict[str, Any]],
    reviewer: RuleReviewer,
    *,
    cohort_signal_summary: Dict[str, Any],
    plan_evidence_audit: Dict[str, Any],
    case_boundary_context: Dict[str, Any],
    only_tx_hashes: set[str],
    max_reviews: int,
) -> List[Dict[str, Any]]:
    reviews: List[Dict[str, Any]] = []
    for result in results:
        tx_hash = get_tx_hash(result)
        if only_tx_hashes and str(tx_hash).lower() not in only_tx_hashes:
            continue
        if infer_error_type(result) == "unknown":
            continue
        reviewer_case = build_reviewer_case(result)
        reviewer_case["cohort_signal_summary"] = dict(cohort_signal_summary or {})
        reviewer_case["plan_evidence_audit"] = dict(plan_evidence_audit or {})
        reviewer_case["case_boundary_context"] = dict(case_boundary_context or {})
        review = reviewer.review_error(reviewer_case)
        review["tx_hash"] = tx_hash
        review["ground_truth"] = get_ground_truth(result)
        review["raw_ground_truth"] = get_raw_ground_truth(result)
        review["sample_role"] = reviewer_case.get("sample_role", "")
        review["negative_kind"] = reviewer_case.get("negative_kind", "")
        review["target_label"] = reviewer_case.get("target_label", "")
        review["predicted_verdict"] = get_finding(result).get("verdict")
        reviews.append(review)
        if max_reviews > 0 and len(reviews) >= max_reviews:
            break
    return reviews


if __name__ == "__main__":
    main()
