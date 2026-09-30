from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import hashlib
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.adapters.llm_adapter import (
    OpenAICompatibleLLM,
    describe_llm_config,
    resolve_adaptive_llm_route,
)
from evotx.core.labels import normalize_attack_label
from evotx.core.rule import (
    baseline_relative_rule_budget_decision,
    generate_cold_start_rule,
    rule_complexity,
)
from evotx.core.plan import (
    apply_reentrancy_stateful_runtime,
    compile_rule_to_baseline_plan,
    followup_round_policy,
    judge_step_view_budget_audit,
)
from evotx.core.schemas import EvidencePlan, EvolvingRule
from evotx.evolution.plan_updater import PlanUpdater
from evotx.evolution.phase3_search import (
    apply_plan_strategy_overlay,
    build_bounded_candidate_portfolio,
    build_condition_evidence_dependencies,
    build_plan_activation_probe_specs,
    build_plan_strategy_deltas,
    plan_route_activation_audit,
)
from evotx.evolution.candidate_delta import (
    RULE_PARTIAL_ACCEPTANCE_STAGE,
    attributed_rule_delta_lineage_report,
    build_rule_condition_deltas,
    build_rule_condition_subset_plan,
    generate_blocker_aware_rule_delta_subsets,
    generate_rule_delta_subsets,
    materialize_rule_condition_subset,
    materialize_synthesized_rule_candidate,
    rule_delta_dependency_preflight,
    rule_delta_subset_preserves_joint_lineage,
)
from evotx.evolution.negative_training import (
    resolve_negative_training_mode,
    validate_negative_training_coverage,
)
from evotx.evolution.regression import (
    CandidateRepairPolicy,
    RegressionEvaluator,
    candidate_repair_diagnostic as regression_candidate_repair_diagnostic,
    candidate_repair_eligible as regression_candidate_repair_eligible,
    new_fp_txs_from_comparison,
)
from evotx.evolution.reviewer import RuleReviewer, infer_error_type
from evotx.evolution.diagnosis_routing import is_authoritative_engineering_route
from evotx.evolution.updater import RuleUpdater
from evotx.planner.plan_generator import PlanGenerator
from evotx.planner.plan_validator import (
    PlanValidationError,
    PlanValidator,
    dependency_affected_condition_ids,
    dependency_consistency_report,
)
from evotx.storage.rule_store import RuleStore
from evotx.storage.plan_store import PlanStore
from evotx.utils.case_utils import load_cases_from_split_csvs, load_labeled_cases
from evotx.utils.csv_archive_utils import archive_split_csvs
from evotx.utils.json_utils import read_json, stable_json_dumps, write_json
from evotx.utils.logging_utils import configure_session_logging
from evotx.utils.report_utils import build_fewshot_final_report
from evotx.utils.fingerprint_utils import (
    rule_fingerprint,
    semantic_plan_payload,
    semantic_plan_fingerprint,
)
from evotx.utils.result_utils import (
    get_finding,
    get_ground_truth,
    get_plan,
    get_raw_ground_truth,
    get_rule,
    get_trace,
    get_tx_hash,
)
from evotx.utils.result_slimmer import (
    build_case_boundary_context,
    build_cohort_signal_summary,
    build_plan_evidence_audit,
    build_reviewer_case,
    build_round_review_bundle,
    build_slim_result,
    normalize_rejected_update_memory,
)
from evotx.runtime.packet_view_manifest import VIEW_MANIFEST_VERSION
from evotx.runtime.packet_runtime import disable_stateful_bindings_in_plan
from experiments.run_inference import (
    build_runtime_policy_identity,
    prepare_effective_runtime_plan,
    run_single_inference,
)


def main() -> None:
    run_started_at = _now_iso()
    run_started_perf = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Stage 2: few-shot EvoTx rule evolution with runtime traces and threshold-based stopping."
    )
    parser.add_argument(
        "--cases-file",
        help="Labeled cases in JSON or CSV. Required columns/fields: tx_hash, chain, ground_truth.",
    )
    parser.add_argument("--pos-csv", "--pos_csv", dest="pos_csv")
    parser.add_argument("--neg-csv", dest="neg_csv")
    parser.add_argument(
        "--num-shots",
        "--num_shots",
        dest="num_shots",
        type=int,
        help=(
            "Maximum rows loaded from each positive and negative split CSV. "
            "When omitted, all rows are used. Guard rows are not truncated."
        ),
    )
    parser.add_argument(
        "--negative-training-mode",
        choices=["auto", "benign_only", "attack_contrastive", "mixed"],
        default="auto",
        help=(
            "Round-level negative-set semantics exposed only to reviewer/updaters. "
            "auto infers the mode from case negative_kind metadata."
        ),
    )
    parser.add_argument(
        "--hard-neg-csv",
        "--hard-negative-csv",
        "--guard-neg-csv",
        dest="hard_neg_csv",
        help=(
            "Fixed hard-negative guard/dev CSV. Rows are evaluated as benign "
            "for the current target and used only as candidate acceptance gate."
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
        help="Optional fixed EvidencePlan JSON. When set, planner generation is skipped unless a plan update is accepted.",
    )
    parser.add_argument(
        "--generate-initial-plan-with-agent",
        action="store_true",
        default=False,
        help=(
            "When --plan-file is absent, call the configured Planner LLM to "
            "generate the reusable initial plan instead of compiling a "
            "deterministic baseline plan."
        ),
    )
    parser.add_argument(
        "--review-bundle",
        help="Optional existing review_bundle.json. Round 1 will use it instead of calling the reviewer.",
    )
    parser.add_argument(
        "--evolution-mode",
        choices=["explore", "strict"],
        default="explore",
        help="Policy template for candidate acceptance and repair during evolution rounds.",
    )
    parser.add_argument(
        "--continue-after-rejected-candidate",
        dest="continue_after_rejected_candidate",
        action="store_true",
        default=True,
        help=(
            "When a viable candidate is rejected and cannot be repaired, keep "
            "the current artifacts, record compact rejected-candidate memory, "
            "and let the next round try a different update direction."
        ),
    )
    parser.add_argument(
        "--stop-after-rejected-candidate",
        dest="continue_after_rejected_candidate",
        action="store_false",
        help="Stop immediately when a candidate is rejected and no repair is eligible.",
    )
    parser.add_argument(
        "--rejected-candidate-retry-rounds",
        type=int,
        default=2,
        help=(
            "Maximum consecutive rejected-candidate rounds allowed before "
            "stopping. Only applies when --continue-after-rejected-candidate "
            "is enabled."
        ),
    )
    parser.add_argument(
        "--final-validation-mode",
        choices=["none", "same", "strict"],
        default="strict",
        help=(
            "Optional final validation pass after evolution. none=skip, "
            "same=validate with evolution thresholds, strict=validate with strict thresholds."
        ),
    )
    parser.add_argument(
        "--force-final-validation-rerun",
        action="store_true",
        default=False,
        help=(
            "Force re-running final rule/plan on train and guard cases during "
            "final validation instead of reusing the latest equivalent results."
        ),
    )
    parser.add_argument(
        "--final-validation-result-source",
        choices=["auto", "reuse", "rerun"],
        default="auto",
        help=(
            "How final validation obtains judge results. auto reuses cached "
            "current/candidate results when case sets are comparable and reruns "
            "otherwise; reuse never calls the judge and fails closed if cached "
            "results are missing or incomparable; rerun always calls the judge. "
            "--force-final-validation-rerun remains as a compatibility alias "
            "for rerun."
        ),
    )
    parser.add_argument(
        "--update-target",
        choices=["auto", "rule", "plan"],
        default="auto",
        help="Which artifact type to update from reviews. auto may update rule and/or plan.",
    )
    parser.add_argument("--attack-description")
    parser.add_argument("--human-example", default="")
    parser.add_argument("--human-example-file")
    parser.add_argument("--tool-manifest", default="configs/tool_manifest.json")
    parser.add_argument("--llm-model")
    parser.add_argument(
        "--llm-provider",
        help="Optional provider override for all few-shot LLMs: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument(
        "--adaptive-model",
        help=(
            "Optional second-tier model used by roles whose thinking mode is "
            "adaptive. Falls back to the role/global model when omitted."
        ),
    )
    parser.add_argument(
        "--adaptive-provider",
        help=(
            "Optional second-tier provider used by roles whose thinking mode "
            "is adaptive. Falls back to --llm-provider when omitted."
        ),
    )
    parser.add_argument("--cold-start-model")
    parser.add_argument("--planner-model")
    parser.add_argument(
        "--cold-start-max-tokens",
        type=int,
        default=32768,
        help="Maximum output tokens for cold-start Rule generation.",
    )
    parser.add_argument(
        "--planner-max-tokens",
        type=int,
        default=32768,
        help="Maximum output tokens for initial and candidate Plan generation.",
    )
    parser.add_argument(
        "--cold-start-thinking",
        choices=["default", "disabled", "adaptive"],
        default="adaptive",
        help=(
            "Thinking mode for cold-start Rule generation. 'default' omits "
            "the provider thinking field."
        ),
    )
    parser.add_argument(
        "--planner-thinking",
        choices=["default", "disabled", "adaptive"],
        default="adaptive",
        help=(
            "Thinking mode for initial and candidate Plan generation. "
            "'default' omits the provider thinking field."
        ),
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
        help="Judge thinking mode; 'default' omits the provider thinking field.",
    )
    parser.add_argument(
        "--aggregator-thinking",
        choices=["default", "disabled", "adaptive"],
        default="disabled",
        help=(
            "Dynamic Aggregator thinking mode; 'default' omits the provider "
            "thinking field."
        ),
    )
    parser.add_argument("--env-model")
    parser.add_argument("--review-model")
    parser.add_argument("--update-model")
    parser.add_argument(
        "--review-max-tokens",
        type=int,
        default=32768,
        help="Maximum output tokens for Reviewer calls.",
    )
    parser.add_argument(
        "--update-max-tokens",
        type=int,
        default=16384,
        help="Maximum output tokens shared by RuleUpdater and PlanUpdater calls.",
    )
    parser.add_argument(
        "--review-thinking",
        choices=["default", "disabled", "adaptive"],
        default="adaptive",
        help="Reviewer thinking mode; 'default' omits the provider thinking field.",
    )
    parser.add_argument(
        "--update-thinking",
        choices=["default", "disabled", "adaptive"],
        default="adaptive",
        help=(
            "RuleUpdater/PlanUpdater thinking mode; 'default' omits the "
            "provider thinking field."
        ),
    )
    parser.add_argument(
        "--enable-review-label-rationale",
        dest="review_label_rationale",
        action="store_true",
        default=False,
        help=(
            "Pass CSV Cause/label_rationale only to Reviewer and Updater as "
            "training supervision. It is never sent to Judge or cold start."
        ),
    )
    parser.add_argument(
        "--disable-review-label-rationale",
        dest="review_label_rationale",
        action="store_false",
        help="Do not pass CSV Cause/label_rationale to Reviewer or Updater.",
    )
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
        help="Rebuild compact packets from source cache during each inference run.",
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
        "--save-llm-transcripts",
        action="store_true",
        help=(
            "Save every judge prompt, raw completion, token usage, render "
            "metadata, and follow-up tool observations to separate transcript files."
        ),
    )
    parser.add_argument(
        "--llm-transcripts-dir",
        help=(
            "Optional transcript output directory. Defaults to "
            "{artifacts_dir}/llm_transcripts after episode path resolution."
        ),
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
    parser.add_argument("--rules-dir", default="data/rules")
    parser.add_argument("--plans-dir", default="data/plans")
    parser.add_argument("--artifacts-dir", default="data/results")
    parser.add_argument("--logs-dir", default="data/log")
    parser.add_argument(
        "--csv-dir",
        default="data/csv",
        help="Root directory for episode-scoped CSV input archives.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        help=(
            "Optional experiment episode id. When set, outputs are nested under "
            "{artifacts_dir}/{episode}, {rules_dir}/{episode}, and {logs_dir}/{episode}."
        ),
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--stop-errors", type=int, default=0)
    parser.add_argument(
        "--max-fp-increase",
        type=int,
        default=1,
        help=(
            "Compatibility/repair threshold for candidate diagnostics. Main "
            "regression acceptance uses errors first, then equal-error FN reduction."
        ),
    )
    parser.add_argument(
        "--max-fn-increase",
        type=int,
        default=1,
        help=(
            "Compatibility field for reports and repair diagnostics. Main "
            "regression acceptance rejects equal-error FN increases."
        ),
    )
    parser.add_argument(
        "--max-uncertain-increase",
        type=int,
        default=0,
        help=(
            "Maximum allowed increase in uncertain predictions for candidate "
            "acceptance after the primary error/FN rule accepts."
        ),
    )
    parser.add_argument(
        "--max-uncertain-benign-increase",
        type=int,
        default=0,
        help=(
            "Maximum allowed increase in benign-side uncertain predictions. "
            "Default 0 treats benign->uncertain as a safety regression."
        ),
    )
    parser.add_argument(
        "--max-guard-fp-increase",
        type=int,
        default=1,
        help="Maximum allowed FP increase on fixed hard-negative guard cases.",
    )
    parser.add_argument(
        "--max-guard-error-increase",
        type=int,
        default=1,
        help="Maximum allowed error increase on fixed hard-negative guard cases.",
    )
    parser.add_argument(
        "--max-guard-uncertain-increase",
        type=int,
        default=1,
        help="Maximum allowed uncertain increase on fixed hard-negative guard cases.",
    )
    parser.add_argument(
        "--allow-guard-repair",
        action="store_true",
        default=False,
        help=(
            "Allow hard-negative guard failures to enter candidate repair. "
            "Default is false so guard cases remain gate/dev-only."
        ),
    )
    parser.add_argument(
        "--enable-candidate-judge-reuse",
        dest="enable_candidate_judge_reuse",
        action="store_true",
        default=True,
        help=(
            "Allow candidate evaluation to reuse judge steps from current "
            "results when fingerprints match. This is enabled by default; "
            "PacketRuntime still performs step-level fingerprint checks before "
            "reusing any judge result."
        ),
    )
    parser.add_argument(
        "--disable-candidate-judge-reuse",
        dest="enable_candidate_judge_reuse",
        action="store_false",
        help=(
            "Disable candidate judge-step reuse and force full candidate "
            "evaluation."
        ),
    )
    parser.add_argument(
        "--enable-candidate-canary",
        dest="enable_candidate_canary",
        action="store_true",
        default=False,
        help=(
            "Opt into a bounded candidate canary before full train/guard "
            "validation. The default minimal pipeline validates the complete "
            "few-shot and guard sets directly."
        ),
    )
    parser.add_argument(
        "--disable-candidate-canary",
        dest="enable_candidate_canary",
        action="store_false",
        help="Skip staged candidate canary validation.",
    )
    parser.add_argument(
        "--candidate-canary-error-cases",
        type=int,
        default=2,
        help="Maximum currently fixable error cases included in a candidate canary.",
    )
    parser.add_argument(
        "--candidate-canary-positive-cases",
        type=int,
        default=3,
        help="Maximum protected correct-positive cases included in a candidate canary.",
    )
    parser.add_argument(
        "--candidate-canary-boundary-cases",
        type=int,
        default=2,
        help="Maximum protected correct-negative cases included in a candidate canary.",
    )
    parser.add_argument(
        "--candidate-canary-guard-cases",
        type=int,
        default=2,
        help="Maximum hard-negative guard cases included in a candidate canary.",
    )
    parser.add_argument(
        "--enable-rule-partial-acceptance",
        dest="enable_rule_partial_acceptance",
        action="store_true",
        default=False,
        help=(
            "After a multi-delta rule candidate is rejected, evaluate bounded "
            "condition/exclusion subsets before invoking LLM candidate repair."
        ),
    )
    parser.add_argument(
        "--enable-rule-structural-evolution",
        dest="enable_rule_structural_evolution",
        action="store_true",
        default=True,
        help=(
            "Enable supported-signal semantic rewrites, positive-condition "
            "addition, exclusion addition, and exclusion narrowing."
        ),
    )
    parser.add_argument(
        "--disable-rule-structural-evolution",
        dest="enable_rule_structural_evolution",
        action="store_false",
        help=(
            "Keep the Rule condition/exclusion interface fixed for "
            "ablation/replay; scoped rewrites and exclusion narrowing remain."
        ),
    )
    parser.add_argument(
        "--enable-remove-positive-condition",
        action="store_true",
        default=False,
        help=(
            "Allow supported removal of a positive Rule condition. Disabled "
            "by default because it directly broadens the target boundary."
        ),
    )
    parser.add_argument(
        "--enable-remove-exclusion",
        action="store_true",
        default=False,
        help=(
            "Allow supported removal of a Rule exclusion. Disabled by default "
            "because it directly broadens the target boundary."
        ),
    )
    parser.add_argument(
        "--max-synthesized-rule-candidates",
        type=int,
        default=1,
        help=(
            "Maximum blocker-aware Rule candidates formed from compatible "
            "atomic deltas when the advanced Phase 3 portfolio is enabled."
        ),
    )
    parser.add_argument(
        "--enable-phase3-candidate-portfolio",
        dest="enable_phase3_candidate_portfolio",
        action="store_true",
        default=False,
        help=(
            "Opt into the advanced multi-direction Rule/Plan/joint/probe "
            "portfolio. The default minimal pipeline keeps at most one Rule "
            "candidate and one Plan candidate."
        ),
    )
    parser.add_argument(
        "--disable-phase3-candidate-portfolio",
        dest="enable_phase3_candidate_portfolio",
        action="store_false",
        help=(
            "Disable bounded Rule-only, Plan-only, and compatible joint "
            "candidate exploration."
        ),
    )
    parser.add_argument(
        "--max-phase3-candidates",
        type=int,
        default=6,
        help="Maximum candidates retained in the Phase 3 strategy portfolio.",
    )
    parser.add_argument(
        "--enable-experimental-rule-hypotheses",
        dest="enable_experimental_rule_hypotheses",
        action="store_true",
        default=False,
        help="Opt into probe-only experimental Rule hypotheses.",
    )
    parser.add_argument(
        "--disable-experimental-rule-hypotheses",
        dest="enable_experimental_rule_hypotheses",
        action="store_false",
        help=(
            "Disable probe-only Rule hypotheses derived from abstract, "
            "non-conflicting insufficient signals."
        ),
    )
    parser.add_argument(
        "--enable-plan-route-pruning",
        action="store_true",
        default=False,
        help=(
            "Allow outright removal of a Plan evidence route. Disabled by "
            "default; demotion to follow-up remains available."
        ),
    )
    parser.add_argument(
        "--enable-plan-dependency-restructure",
        action="store_true",
        default=False,
        help=(
            "Allow PlanUpdater to change depends_on. Disabled by default "
            "because producer/consumer rewiring is high risk."
        ),
    )
    parser.add_argument(
        "--disable-rule-partial-acceptance",
        dest="enable_rule_partial_acceptance",
        action="store_false",
        help="Disable rejected rule-candidate condition subset evaluation.",
    )
    parser.add_argument(
        "--rule-partial-max-deltas",
        type=int,
        default=6,
        help=(
            "Maximum condition/exclusion delta count eligible for partial "
            "acceptance. Larger candidates fall through to normal repair."
        ),
    )
    parser.add_argument(
        "--rule-partial-max-evaluations",
        type=int,
        default=10,
        help="Maximum evaluated rule-delta subsets per rejected parent candidate.",
    )
    parser.add_argument(
        "--enable-runtime-early-stop",
        action="store_true",
        default=False,
        help=(
            "Enable conservative negative early stop during few-shot runtime "
            "evaluation. Default false keeps full judge traces."
        ),
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
            "Choose prompt-only Access Control object consistency or strong "
            "stateful authorization-chain binding."
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
            "Ablation mode: ignore plan depends_on, consumes_state_keys, "
            "produces_state_key, state_prompt_role, state_output_schema, and "
            "judge state_output during runtime."
        ),
    )
    parser.add_argument(
        "--enable-dynamic-aggregation",
        dest="dynamic_aggregation",
        action="store_true",
        default=False,
        help=(
            "Opt into dynamic aggregation with at most one aggregator-requested "
            "source follow-up round owned by the local judges. The default "
            "minimal pipeline uses Plan emit_logic directly."
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
            "across evidence, prior judge summary, tool observations, and state."
        ),
    )
    parser.add_argument(
        "--judge-followup-mode",
        choices=["plan", "disabled", "expanded"],
        default="plan",
        help=(
            "Judge follow-up ablation. plan preserves iterative evidence; "
            "disabled performs one call with plan defaults; expanded promotes "
            "high-priority follow-up views before the one call."
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
        "--enable-candidate-repair",
        dest="enable_candidate_repair",
        action="store_true",
        default=False,
        help=(
            "When a rule candidate reduces FN/errors but introduces a small "
            "temporary FP increase, review the new FP and attempt one boundary "
            "repair before rejecting it."
        ),
    )
    parser.add_argument(
        "--disable-candidate-repair",
        dest="enable_candidate_repair",
        action="store_false",
        help="Disable rejected-candidate new-FP repair.",
    )
    parser.add_argument(
        "--temporary-fp-increase",
        type=int,
        default=2,
        help=(
            "Maximum temporary FP increase allowed for entering candidate "
            "repair. Final acceptance still uses the simple regression rule."
        ),
    )
    parser.add_argument(
        "--repair-min-fn-decrease",
        type=int,
        default=1,
        help="Minimum candidate FN decrease required to enter new-FP repair.",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=1,
        help="Maximum targeted repair attempts after a rejected candidate.",
    )
    parser.add_argument(
        "--enable-recall-repair",
        dest="enable_recall_repair",
        action="store_true",
        default=False,
        help=(
            "Enable rejected rule-candidate recall repair when a candidate fixes "
            "FP but introduces new FN."
        ),
    )
    parser.add_argument(
        "--disable-recall-repair",
        dest="enable_recall_repair",
        action="store_false",
        help="Disable rejected rule-candidate new-FN recall repair.",
    )
    parser.add_argument(
        "--recall-repair-min-fixed-fp",
        type=int,
        default=1,
        help="Minimum fixed FP count required to enter recall repair.",
    )
    parser.add_argument(
        "--recall-repair-max-new-fn",
        type=int,
        default=10,
        help="Maximum new FN cases to review during recall repair.",
    )
    parser.add_argument(
        "--label-aware-source-budget",
        dest="label_aware_source_budget",
        action="store_true",
        default=True,
        help=(
            "Use label-aware source-dependent rule budget warnings for "
            "source-heavy families such as access_control and insufficient_validation."
        ),
    )
    parser.add_argument(
        "--strict-source-budget",
        dest="strict_source_budget",
        action="store_true",
        default=False,
        help=(
            "Ablation mode: restore the old hard source-dependent condition "
            "budget for every label."
        ),
    )
    parser.add_argument(
        "--allow-source-budget-warning",
        dest="allow_source_budget_warning",
        action="store_true",
        default=True,
        help="Allow label-aware source-dependent budget overage as a warning.",
    )
    parser.add_argument(
        "--disallow-source-budget-warning",
        dest="allow_source_budget_warning",
        action="store_false",
        help="Treat source-dependent budget overage as a hard rejection.",
    )
    parser.add_argument("--compress-rule", action="store_true")
    raw_argv = list(sys.argv[1:])
    args = parser.parse_args()
    if args.num_shots is not None and args.num_shots <= 0:
        raise ValueError("--num-shots must be a positive integer.")
    if bool(args.force_final_validation_rerun):
        args.final_validation_result_source = "rerun"
    mode_policy = apply_evolution_mode_defaults(args, raw_argv)
    normalize_candidate_repair_args(args)
    apply_episode_paths(args)
    log_path = configure_session_logging(
        "few_shot",
        logs_dir=args.logs_dir,
        suffix=args.label or args.positive_label or "evolving",
    )
    print(f"[FewShot] Started at: {run_started_at}")
    print(f"[FewShot] Worktree root: {Path(__file__).resolve().parents[1]}")
    print(f"[FewShot] Evolution mode: {args.evolution_mode}")
    print(f"[FewShot] Final validation mode: {args.final_validation_mode}")
    print(
        "[FewShot] Final validation result source: "
        f"{args.final_validation_result_source}"
    )
    print(
        "[FewShot] Mode defaults applied: "
        f"{stable_json_dumps(mode_policy.get('mode_defaults_applied', {}))}"
    )
    print(
        "[FewShot] Explicit policy overrides: "
        f"{stable_json_dumps(mode_policy.get('explicit_policy_overrides', []))}"
    )

    tool_manifest = read_json(args.tool_manifest, default={}) or {}
    raw_positive_label = args.label or args.positive_label or ""
    cases, resolved_positive_label = load_cases_input(args)
    resolved_positive_label = normalize_attack_label(resolved_positive_label)
    requested_negative_training_mode = args.negative_training_mode
    args.negative_training_mode = resolve_negative_training_mode(
        requested_negative_training_mode,
        cases,
    )
    args.negative_training_mode_source = (
        "auto_inferred"
        if requested_negative_training_mode == "auto"
        else "explicit_cli"
    )
    negative_training_coverage = validate_negative_training_coverage(
        args.negative_training_mode,
        cases,
    )
    print(
        "[FewShot] Negative training coverage: "
        f"{stable_json_dumps(negative_training_coverage)}"
    )
    hard_guard_cases = load_hard_negative_guard_cases(args, resolved_positive_label)
    csv_archive = archive_split_csvs(
        stage="train",
        benign_csv=args.benign_csv,
        malicious_csv=args.malicious_csv,
        pos_csv=args.pos_csv,
        neg_csv=args.neg_csv,
        label=resolved_positive_label,
        episode=args.episode,
        csv_root=args.csv_dir,
    )
    counts = count_labels(cases)
    guard_counts = count_labels(hard_guard_cases)
    print(
        f"[FewShot] label={resolved_positive_label} cases={counts['total']} "
        f"(attack={counts['attack']}, negative={counts['negative']}, "
        f"negative_benign={counts['negative_benign']}, "
        f"negative_other={counts['negative_other']}) "
        f"negative_training_mode={args.negative_training_mode} "
        f"negative_training_mode_source={args.negative_training_mode_source} "
        f"runtime=packet_runtime force_rebuild_packet={args.force_rebuild_packet} "
        f"source_tools={args.enable_source_tools} "
        f"max_view_chars={args.max_view_chars} max_context_chars={args.max_context_chars} "
        f"update_target={args.update_target} "
        f"candidate_repair={args.enable_candidate_repair} "
        f"continue_after_rejected_candidate={args.continue_after_rejected_candidate} "
        f"rejected_candidate_retry_rounds={args.rejected_candidate_retry_rounds} "
        f"candidate_judge_reuse={args.enable_candidate_judge_reuse} "
        f"candidate_canary={args.enable_candidate_canary} "
        f"runtime_early_stop={args.enable_runtime_early_stop} "
        f"parallel_judge={args.parallel_judge} "
        f"judge_concurrency={args.judge_concurrency} "
        f"judge_max_tokens={args.judge_max_tokens} "
        f"aggregator_max_tokens={args.aggregator_max_tokens} "
        f"judge_thinking={args.judge_thinking} "
        f"aggregator_thinking={args.aggregator_thinking} "
        f"iv_stateful_runtime={args.iv_stateful_runtime} "
        f"access_control_binding_mode={args.access_control_binding_mode} "
        f"reentrancy_binding_mode={args.reentrancy_binding_mode} "
        f"disable_stateful_bindings={args.disable_stateful_bindings} "
        f"dynamic_aggregation={args.dynamic_aggregation} "
        f"followup_context_mode={args.followup_context_mode} "
        f"judge_followup_mode={args.judge_followup_mode} "
        f"judge_expanded_followup_views={args.judge_expanded_followup_views} "
        f"rate_limit_serial_fallback={args.rate_limit_serial_fallback} "
        f"rate_limit_retries={args.rate_limit_retry_attempts} "
        f"adaptive_evidence={args.adaptive_evidence} "
        f"adaptive_mode={args.adaptive_evidence_mode} "
        f"review_compact_mode={args.review_compact_mode} "
        f"cold_start_thinking={args.cold_start_thinking} "
        f"planner_thinking={args.planner_thinking} "
        f"review_thinking={args.review_thinking} "
        f"update_thinking={args.update_thinking} "
        f"generate_initial_plan_with_agent={args.generate_initial_plan_with_agent} "
        f"phase3_candidate_portfolio={args.enable_phase3_candidate_portfolio} "
        f"max_phase3_candidates={args.max_phase3_candidates} "
        f"experimental_rule_hypotheses={args.enable_experimental_rule_hypotheses} "
        f"plan_route_pruning={args.enable_plan_route_pruning} "
        f"plan_dependency_restructure={args.enable_plan_dependency_restructure} "
        f"temporary_fp_increase={args.temporary_fp_increase} "
        f"num_shots={args.num_shots if args.num_shots is not None else 'all'} "
        f"episode={args.episode if args.episode is not None else 'none'}"
    )
    print(f"[FewShot] hard_negative_guard_cases={len(hard_guard_cases)}")
    print(
        f"[FewShot] Output dirs: rules={args.rules_dir} "
        f"plans={args.plans_dir} artifacts={args.artifacts_dir} logs={args.logs_dir}"
    )
    if csv_archive:
        print(f"[FewShot] CSV archive: {stable_json_dumps(csv_archive)}")
    print(
        "[FewShot] Models: "
        f"llm={args.llm_model or 'unset'} cold_start={args.cold_start_model or args.llm_model or 'unset'} "
        f"planner={args.planner_model or args.llm_model or 'unset'} "
        f"judge={args.judge_model or args.llm_model or 'unset'} "
        f"env={args.env_model or args.llm_model or 'unset'} "
        f"review={args.review_model or args.llm_model or 'unset'} "
        f"update={args.update_model or args.llm_model or 'unset'} "
        f"provider={args.llm_provider or '(auto)'} "
        f"adaptive_model={args.adaptive_model or '(inherit)'} "
        f"adaptive_provider={args.adaptive_provider or '(inherit)'}"
    )
    resolved_llm_configs = build_llm_config_summary(args)
    print(
        "[FewShot] Resolved LLM configs: "
        f"{stable_json_dumps(resolved_llm_configs)}"
    )
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_episode_alignment = build_episode_artifact_alignment(
        args.episode,
        artifacts_dir,
    )
    write_json(
        artifacts_dir / "run_identity.json",
        artifact_episode_alignment,
    )
    if not artifact_episode_alignment["aligned"]:
        raise ValueError(
            "Episode/artifact directory mismatch: "
            f"requested={artifact_episode_alignment['requested_episode']} "
            f"artifact_dir={artifact_episode_alignment['artifact_directory']}"
        )
    initial_artifact_provenance = build_initial_artifact_provenance(args)
    write_json(
        artifacts_dir / "initial_artifact_provenance.json",
        initial_artifact_provenance,
    )
    print(
        "[FewShot] Input RuleFile: "
        f"{_quoted_cli_artifact_path(args.rule_file)}"
    )
    print(
        "[FewShot] Input PlanFile: "
        f"{_quoted_cli_artifact_path(args.plan_file)}"
    )
    print(
        "[FewShot] Initialization provenance: "
        f"rule_mode={initial_artifact_provenance['rule']['initialization_mode']} "
        f"plan_mode={initial_artifact_provenance['plan']['initialization_mode']}"
    )
    if args.save_llm_transcripts:
        if not args.llm_transcripts_dir:
            args.llm_transcripts_dir = str(artifacts_dir / "llm_transcripts")
        Path(args.llm_transcripts_dir).mkdir(parents=True, exist_ok=True)
        print(f"[FewShot] LLM transcripts enabled: {args.llm_transcripts_dir}")
    else:
        args.llm_transcripts_dir = None

    initial_rule = apply_rule_budget_config(load_or_initialize_rule(args, tool_manifest), args)
    initial_rule.metadata.setdefault("raw_attack_label", raw_positive_label or resolved_positive_label)
    initial_rule.metadata.setdefault("display_label", raw_positive_label or resolved_positive_label)
    initial_rule.metadata["attack_label"] = normalize_attack_label(
        initial_rule.metadata.get("attack_label") or resolved_positive_label
    )
    rule_store = RuleStore(args.rules_dir)
    plan_store = PlanStore(args.plans_dir)
    initial_rule_path = rule_store.save(
        initial_rule,
        attack_label=resolved_positive_label,
        latest=False,
    )
    initial_artifact_provenance["rule"]["archive_path"] = str(
        initial_rule_path
    )
    write_json(
        artifacts_dir / "initial_artifact_provenance.json",
        initial_artifact_provenance,
    )
    print(f"[FewShot] Initial rule archive: {initial_rule_path}")
    final_rule_path = initial_rule_path
    initial_plan = load_initial_plan(args.plan_file) if args.plan_file else None
    initial_plan_path = ""
    if initial_plan is None:
        if args.generate_initial_plan_with_agent:
            planner_model, planner_provider = resolve_role_llm_route(
                args,
                model=args.planner_model or args.llm_model,
                thinking=args.planner_thinking,
            )
            planner_llm = make_llm(
                planner_model,
                planner_provider,
                max_tokens=args.planner_max_tokens,
                minimax_thinking=args.planner_thinking,
            )
            initial_plan = build_initial_agent_plan(initial_rule, planner_llm)
            print(
                "[FewShot] No --plan-file supplied; generated and saved the "
                "reusable initial plan with the Planner Agent."
            )
        else:
            initial_plan = build_initial_baseline_plan(initial_rule)
            print(
                "[FewShot] No --plan-file supplied; compiled and saved a reusable "
                "baseline plan from the initial rule."
            )
    elif args.generate_initial_plan_with_agent:
        print(
            "[FewShot] --plan-file takes precedence; "
            "--generate-initial-plan-with-agent was not used."
        )
    initial_attack_label = normalize_attack_label(
        (initial_rule.metadata or {}).get("attack_label", ""),
        default="",
    )
    if initial_attack_label == "reentrancy":
        apply_reentrancy_stateful_runtime(
            initial_plan,
            rule=initial_rule,
            mode=args.reentrancy_binding_mode,
        )
    initial_plan = normalize_plan_for_runtime_policy(
        initial_plan,
        args=args,
        reason="initial_plan",
    )
    if initial_attack_label == "reentrancy":
        PlanValidator().validate(
            initial_plan,
            attack_label=initial_attack_label,
        )
    initial_plan_path = str(
        plan_store.save(
            initial_plan,
            attack_label=resolved_positive_label,
            latest=False,
        )
    )
    initial_artifact_provenance["plan"].update({
        "archive_path": initial_plan_path,
        "artifact_metadata_source": str(
            (initial_plan.metadata or {}).get("source") or ""
        ),
    })
    write_json(
        artifacts_dir / "initial_artifact_provenance.json",
        initial_artifact_provenance,
    )
    print(f"[FewShot] Initial plan archive: {initial_plan_path}")
    final_plan = initial_plan
    final_plan_path = initial_plan_path

    review_model, review_provider = resolve_role_llm_route(
        args,
        model=args.review_model or args.llm_model,
        thinking=args.review_thinking,
    )
    update_model, update_provider = resolve_role_llm_route(
        args,
        model=args.update_model or args.llm_model,
        thinking=args.update_thinking,
    )
    reviewer_llm = make_llm(
        review_model,
        review_provider,
        max_tokens=args.review_max_tokens,
        minimax_thinking=args.review_thinking,
    )
    updater_llm = make_llm(
        update_model,
        update_provider,
        max_tokens=args.update_max_tokens,
        minimax_thinking=args.update_thinking,
    )
    reviewer = RuleReviewer(
        llm=reviewer_llm,
        transcript_dir=args.llm_transcripts_dir,
        max_prompt_chars=args.max_review_prompt_chars,
        compact_mode=args.review_compact_mode,
        negative_training_mode=args.negative_training_mode,
    )
    updater = RuleUpdater(
        llm=updater_llm,
        transcript_dir=args.llm_transcripts_dir,
        enable_structural_evolution=bool(
            args.enable_rule_structural_evolution
        ),
        enable_remove_positive_condition=bool(
            args.enable_remove_positive_condition
        ),
        enable_remove_exclusion=bool(args.enable_remove_exclusion),
    )
    plan_updater = PlanUpdater(
        llm=updater_llm,
        transcript_dir=args.llm_transcripts_dir,
        enable_route_pruning=bool(args.enable_plan_route_pruning),
        enable_dependency_restructure=bool(
            args.enable_plan_dependency_restructure
        ),
        enable_advanced_strategy=bool(
            args.enable_phase3_candidate_portfolio
        ),
    )
    if any(
        str(getattr(llm, "provider", "") or "").lower() == "minimax"
        for llm in (reviewer_llm, updater_llm)
        if llm is not None
    ):
        print(
            "[FewShot] MiniMax thinking policy: "
            f"review={getattr(reviewer_llm, 'minimax_thinking', '') or 'provider_default'} "
            f"update={getattr(updater_llm, 'minimax_thinking', '') or 'provider_default'} "
            f"judge={args.judge_thinking} aggregator={args.aggregator_thinking} "
            f"planner={args.planner_thinking} cold_start={args.cold_start_thinking}"
        )
    evaluator = RegressionEvaluator()

    current_rule = initial_rule
    current_rule_source = str(initial_rule_path)
    current_plan = initial_plan
    current_plan_source = initial_plan_path
    external_review_bundle = read_json(args.review_bundle, default=None) if args.review_bundle else None
    accepted_rounds = 0
    stop_reason = "max_rounds_reached"
    final_summary: Dict[str, Any] = {}
    baseline_full_results_for_validation: List[Dict[str, Any]] | None = None
    baseline_summary_for_validation: Dict[str, Any] | None = None
    baseline_guard_full_results_for_validation: List[Dict[str, Any]] | None = None
    baseline_guard_summary_for_validation: Dict[str, Any] | None = None
    latest_final_full_results: List[Dict[str, Any]] | None = None
    latest_final_summary: Dict[str, Any] | None = None
    latest_final_guard_full_results: List[Dict[str, Any]] | None = None
    latest_final_guard_summary: Dict[str, Any] | None = None
    latest_final_rule_source = current_rule_source
    latest_final_plan_source = current_plan_source
    reusable_current_full_results: List[Dict[str, Any]] | None = None
    reusable_current_summary: Dict[str, Any] | None = None
    reusable_current_guard_full_results: List[Dict[str, Any]] | None = None
    reusable_current_guard_summary: Dict[str, Any] | None = None
    reusable_current_from_round: int | None = None
    reusable_current_source_file = ""
    reusable_current_guard_source_file = ""
    reusable_current_reuse_type = ""
    incumbent_full_results: List[Dict[str, Any]] | None = None
    incumbent_summary: Dict[str, Any] | None = None
    incumbent_guard_full_results: List[Dict[str, Any]] | None = None
    incumbent_guard_summary: Dict[str, Any] | None = None
    incumbent_from_round: int | None = None
    incumbent_source_file = ""
    incumbent_guard_source_file = ""
    incumbent_rule_hash = ""
    incumbent_plan_hash = ""
    last_candidate_evaluations: List[Dict[str, Any]] = []
    rejected_update_memory: Dict[str, Any] = empty_rejected_update_memory()
    search_rule = current_rule
    search_rule_source = current_rule_source
    search_plan = current_plan
    search_plan_source = current_plan_source
    search_full_results: List[Dict[str, Any]] | None = None
    search_summary: Dict[str, Any] | None = None
    search_guard_full_results: List[Dict[str, Any]] | None = None
    search_guard_summary: Dict[str, Any] | None = None
    search_base_source = "incumbent"
    search_base_source_round: int | None = None
    active_refinement_feedback: Dict[str, Any] = {}
    consecutive_rejected_candidate_rounds = 0
    cached_review_input_fingerprint = ""
    cached_reviews: List[Dict[str, Any]] = []
    cached_review_source_round: int | None = None
    seen_minimal_candidate_fingerprints: set[str] = set()

    for round_index in range(1, args.max_rounds + 1):
        round_dir = artifacts_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[FewShot] Round {round_index}/{args.max_rounds} with rule "
            f"{current_rule.rule_id} v{current_rule.version}"
        )

        if (
            reusable_current_full_results is None
            and incumbent_artifact_matches_current(
                current_rule=current_rule,
                current_plan=current_plan,
                incumbent_rule_hash=incumbent_rule_hash,
                incumbent_plan_hash=incumbent_plan_hash,
                incumbent_full_results=incumbent_full_results,
            )
        ):
            reusable_current_full_results = copy.deepcopy(incumbent_full_results)
            reusable_current_summary = dict(incumbent_summary or {})
            reusable_current_guard_full_results = (
                copy.deepcopy(incumbent_guard_full_results)
                if hard_guard_cases and incumbent_guard_full_results is not None
                else None
            )
            reusable_current_guard_summary = (
                dict(incumbent_guard_summary or {}) if hard_guard_cases else None
            )
            reusable_current_from_round = incumbent_from_round
            reusable_current_source_file = incumbent_source_file
            reusable_current_guard_source_file = incumbent_guard_source_file
            reusable_current_reuse_type = "unchanged_incumbent_across_rounds"

        current_reuse_metadata: Dict[str, Any] = {}
        current_reuse_source_round = reusable_current_from_round
        if reusable_current_full_results is not None:
            current_reuse_metadata = {
                "reused": True,
                "source_round": current_reuse_source_round,
                "source_file": reusable_current_source_file
                or f"round_{current_reuse_source_round:02d}/candidate_full_results.json"
                if current_reuse_source_round is not None
                else "",
                "rule_id": current_rule.rule_id,
                "rule_version": current_rule.version,
                "rule_source": current_rule_source,
                "plan_source": current_plan_source,
                "reuse_type": reusable_current_reuse_type
                or "accepted_candidate_as_next_current",
            }
            print(
                "[FewShot] Reusing cached current results from "
                f"round {current_reuse_source_round} as current results for "
                f"round {round_index}; skipping current-rule inference."
            )
            current_full_results = retarget_reused_results(
                reusable_current_full_results,
                rule_source=current_rule_source,
                plan_source=current_plan_source,
                reuse_metadata=current_reuse_metadata,
            )
            current_summary = dict(
                reusable_current_summary
                or evaluator.summarize(current_full_results).to_dict()
            )
            reusable_current_full_results = None
            reusable_current_summary = None
            if not hard_guard_cases or reusable_current_guard_full_results is None:
                reusable_current_from_round = None
                reusable_current_source_file = ""
                reusable_current_guard_source_file = ""
                reusable_current_reuse_type = ""
        else:
            current_full_results = run_cases(
                cases=cases,
                rule=current_rule,
                rule_source=current_rule_source,
                tool_manifest=tool_manifest,
                llm_provider=args.llm_provider,
                planner_model=args.planner_model or args.llm_model,
                judge_model=args.judge_model or args.llm_model,
                env_model=args.env_model or args.llm_model,
                use_environment=args.use_environment,
                base_cache_dir=args.base_cache_dir,
                force_rebuild_packet=args.force_rebuild_packet,
                enable_source_tools=args.enable_source_tools,
                source_cache_dir=args.source_cache_dir,
                force_refresh_source=args.force_refresh_source,
                max_view_chars=args.max_view_chars,
                max_context_chars=args.max_context_chars,
                phase=f"round_{round_index:02d}/current",
                llm_transcripts_dir=args.llm_transcripts_dir,
                fixed_plan=current_plan,
                fixed_plan_source=current_plan_source,
                runtime_early_stop=bool(args.enable_runtime_early_stop),
                **runtime_adaptive_kwargs(args),
            )
            current_summary = evaluator.summarize(current_full_results).to_dict()

        current_guard_full_results: List[Dict[str, Any]] = []
        current_guard_summary: Dict[str, Any] | None = None
        if hard_guard_cases:
            if reusable_current_guard_full_results is not None:
                print(
                    "[FewShot] Reusing accepted candidate guard results from "
                    f"round {current_reuse_source_round} as current guard results."
                )
                current_guard_full_results = retarget_reused_results(
                    reusable_current_guard_full_results,
                    rule_source=current_rule_source,
                    plan_source=current_plan_source,
                    reuse_metadata={
                        "reused": True,
                        "source_round": current_reuse_source_round,
                        "source_file": reusable_current_guard_source_file
                        or f"round_{current_reuse_source_round:02d}/candidate_guard_full_results.json"
                        if current_reuse_source_round is not None
                        else "",
                        "rule_id": current_rule.rule_id,
                        "rule_version": current_rule.version,
                        "rule_source": current_rule_source,
                        "plan_source": current_plan_source,
                        "guard_role": "hard_negative_guard",
                        "reuse_type": reusable_current_reuse_type
                        or "accepted_candidate_as_next_current",
                    },
                )
                current_guard_summary = dict(
                    reusable_current_guard_summary
                    or evaluator.summarize(current_guard_full_results).to_dict()
                )
                reusable_current_guard_full_results = None
                reusable_current_guard_summary = None
                reusable_current_from_round = None
                reusable_current_source_file = ""
                reusable_current_guard_source_file = ""
                reusable_current_reuse_type = ""
            else:
                current_guard_full_results = run_cases(
                    cases=hard_guard_cases,
                    rule=current_rule,
                    rule_source=current_rule_source,
                    tool_manifest=tool_manifest,
                    llm_provider=args.llm_provider,
                    planner_model=args.planner_model or args.llm_model,
                    judge_model=args.judge_model or args.llm_model,
                    env_model=args.env_model or args.llm_model,
                    use_environment=args.use_environment,
                    base_cache_dir=args.base_cache_dir,
                    force_rebuild_packet=args.force_rebuild_packet,
                    enable_source_tools=args.enable_source_tools,
                    source_cache_dir=args.source_cache_dir,
                    force_refresh_source=args.force_refresh_source,
                    max_view_chars=args.max_view_chars,
                    max_context_chars=args.max_context_chars,
                    phase=f"round_{round_index:02d}/current_guard",
                    llm_transcripts_dir=args.llm_transcripts_dir,
                    fixed_plan=current_plan,
                    fixed_plan_source=current_plan_source,
                    runtime_early_stop=bool(args.enable_runtime_early_stop),
                    **runtime_adaptive_kwargs(args),
                )
                current_guard_summary = evaluator.summarize(current_guard_full_results).to_dict()
            current_guard_slim_results = [
                build_slim_result(result) for result in current_guard_full_results
            ]
            write_json(round_dir / "current_guard_full_results.json", current_guard_full_results)
            write_json(round_dir / "current_guard_slim_results.json", current_guard_slim_results)
            write_json(round_dir / "current_guard_summary.json", current_guard_summary)
            print(
                f"[FewShot] Guard summary current: "
                f"{stable_json_dumps(current_guard_summary)}"
            )
        current_slim_results = [build_slim_result(result) for result in current_full_results]
        write_json(round_dir / "full_results.json", current_full_results)
        write_json(round_dir / "slim_results.json", current_slim_results)
        write_json(round_dir / "current_results.json", current_slim_results)
        write_json(round_dir / "current_summary.json", current_summary)
        write_json(round_dir / "current_case_plans.json", extract_case_plans(current_full_results))
        if current_plan is not None:
            write_json(round_dir / "current_plan.json", current_plan.to_dict())
            write_json(
                round_dir / "current_artifact.json",
                {
                    "rule": {
                        "rule_id": current_rule.rule_id,
                        "rule_version": current_rule.version,
                        "rule_source": current_rule_source,
                    },
                    "plan": {
                        "plan_id": current_plan.plan_id,
                        "plan_source": current_plan_source,
                        "plan_version": (current_plan.metadata or {}).get("plan_version"),
                    },
                },
            )
        if current_reuse_metadata:
            write_json(round_dir / "current_reuse_metadata.json", current_reuse_metadata)

        final_summary = current_summary
        if round_index == 1 and baseline_full_results_for_validation is None:
            baseline_full_results_for_validation = copy.deepcopy(current_full_results)
            baseline_summary_for_validation = dict(current_summary)
            baseline_guard_full_results_for_validation = copy.deepcopy(
                current_guard_full_results
            )
            baseline_guard_summary_for_validation = (
                dict(current_guard_summary or {}) if current_guard_summary else {}
            )
        if incumbent_full_results is None:
            incumbent_full_results = copy.deepcopy(current_full_results)
            incumbent_summary = dict(current_summary)
            incumbent_guard_full_results = copy.deepcopy(current_guard_full_results)
            incumbent_guard_summary = dict(current_guard_summary or {})
            incumbent_from_round = round_index
            incumbent_source_file = f"round_{round_index:02d}/full_results.json"
            incumbent_guard_source_file = (
                f"round_{round_index:02d}/current_guard_full_results.json"
                if hard_guard_cases
                else ""
            )
            incumbent_rule_hash = artifact_hash(current_rule)
            incumbent_plan_hash = artifact_hash(current_plan)
        latest_final_full_results = copy.deepcopy(current_full_results)
        latest_final_summary = dict(current_summary)
        latest_final_guard_full_results = copy.deepcopy(current_guard_full_results)
        latest_final_guard_summary = dict(current_guard_summary or {})
        latest_final_rule_source = current_rule_source
        latest_final_plan_source = current_plan_source
        print(f"[FewShot] Current summary: {stable_json_dumps(current_summary)}")

        if search_base_source == "incumbent" or search_full_results is None:
            round_search_rule = current_rule
            round_search_rule_source = current_rule_source
            round_search_plan = current_plan
            round_search_plan_source = current_plan_source
            round_search_full_results = current_full_results
            round_search_summary = current_summary
            round_search_guard_full_results = current_guard_full_results
            round_search_guard_summary = current_guard_summary or {}
        else:
            round_search_rule = search_rule
            round_search_rule_source = search_rule_source
            round_search_plan = search_plan
            round_search_plan_source = search_plan_source
            round_search_full_results = retarget_reused_results(
                search_full_results,
                rule_source=round_search_rule_source,
                plan_source=round_search_plan_source,
                reuse_metadata={
                    "reused": True,
                    "source_round": search_base_source_round,
                    "source_file": search_base_source,
                    "rule_id": round_search_rule.rule_id,
                    "rule_version": round_search_rule.version,
                    "rule_source": round_search_rule_source,
                    "plan_source": round_search_plan_source,
                    "reuse_type": "rejected_candidate_refinement_search_base",
                },
            )
            round_search_summary = dict(
                search_summary
                or evaluator.summarize(round_search_full_results).to_dict()
            )
            round_search_guard_full_results = copy.deepcopy(
                search_guard_full_results or []
            )
            round_search_guard_summary = dict(search_guard_summary or {})
        round_search_slim_results = [
            build_slim_result(result) for result in round_search_full_results
        ]
        round_search_guard_slim_results = [
            build_slim_result(result) for result in round_search_guard_full_results
        ]
        write_json(round_dir / "search_results.json", round_search_slim_results)
        write_json(round_dir / "search_summary.json", round_search_summary)
        write_json(
            round_dir / "search_artifact.json",
            {
                "rule": {
                    "rule_id": round_search_rule.rule_id,
                    "rule_version": round_search_rule.version,
                    "rule_source": round_search_rule_source,
                    "rule_hash": artifact_hash(round_search_rule),
                },
                "plan": {
                    "plan_id": round_search_plan.plan_id
                    if round_search_plan is not None
                    else "",
                    "plan_source": round_search_plan_source,
                    "plan_version": (round_search_plan.metadata or {}).get("plan_version")
                    if round_search_plan is not None
                    else None,
                    "plan_hash": artifact_hash(round_search_plan),
                },
                "search_base_source": search_base_source,
                "search_base_source_round": search_base_source_round,
                "active_refinement_feedback": dict(active_refinement_feedback or {}),
            },
        )
        if round_search_plan is not None:
            write_json(round_dir / "search_plan.json", round_search_plan.to_dict())
        if active_refinement_feedback:
            write_json(
                round_dir / "active_refinement_feedback.json",
                active_refinement_feedback,
            )
        round_transition_audit: Dict[str, Any] = {
            "schema_version": "evotx.round_transition_audit.v1",
            "round": round_index,
            "incumbent_rule_hash": artifact_hash(current_rule),
            "incumbent_plan_hash": artifact_hash(current_plan),
            "incumbent_rule_source": current_rule_source,
            "incumbent_plan_source": current_plan_source,
            "search_base_rule_hash": artifact_hash(round_search_rule),
            "search_base_plan_hash": artifact_hash(round_search_plan),
            "search_base_source": search_base_source,
            "search_base_source_round": search_base_source_round,
            "active_refinement_feedback_present": bool(active_refinement_feedback),
            "accepted": False,
            "rejection_reason": "",
            "refinable": False,
            "reuse_type": (
                "rejected_candidate_refinement"
                if search_base_source != "incumbent"
                else "incumbent_anchor"
            ),
        }
        write_json(round_dir / "round_transition_audit.json", round_transition_audit)

        if current_summary["errors"] <= args.stop_errors:
            stop_reason = f"error_threshold_reached({current_summary['errors']}<={args.stop_errors})"
            round_transition_audit["stop_reason"] = stop_reason
            write_json(round_dir / "round_transition_audit.json", round_transition_audit)
            break

        if external_review_bundle is not None and round_index == 1:
            review_bundle = dict(external_review_bundle)
            review_bundle["negative_training_mode"] = args.negative_training_mode
            review_bundle["negative_training_mode_source"] = (
                args.negative_training_mode_source
            )
            reviews = _reviews_from_bundle(review_bundle)
            review_bundle.setdefault("current_rule", round_search_rule.to_dict())
            review_bundle.setdefault("current_summary", round_search_summary)
            review_bundle.setdefault(
                "cohort_signal_summary",
                build_cohort_signal_summary(
                    round_search_slim_results,
                    guard_slim_results=round_search_guard_slim_results,
                ),
            )
            review_bundle.setdefault(
                "plan_evidence_audit",
                build_plan_evidence_audit(
                    round_search_slim_results,
                    guard_slim_results=round_search_guard_slim_results,
                ),
            )
            review_bundle.setdefault(
                "case_boundary_context",
                build_case_boundary_context(
                    round_search_slim_results,
                    guard_slim_results=round_search_guard_slim_results,
                    cohort_signal_summary=review_bundle.get("cohort_signal_summary"),
                    plan_evidence_audit=review_bundle.get("plan_evidence_audit"),
                ),
            )
            review_bundle.setdefault(
                "rejected_update_memory",
                memory_with_active_refinement_feedback(
                    rejected_update_memory,
                    active_refinement_feedback,
                ),
            )
            review_bundle.setdefault(
                "update_constraints",
                [
                    "External review bundle supplied by --review-bundle.",
                    "Apply only reviews matching --update-target.",
                ],
            )
            print(
                f"[FewShot] Using external review bundle: {args.review_bundle} "
                f"reviews={len(reviews)}"
            )
        else:
            cohort_signal_summary = build_cohort_signal_summary(
                round_search_slim_results,
                guard_slim_results=round_search_guard_slim_results,
            )
            plan_evidence_audit = build_plan_evidence_audit(
                round_search_slim_results,
                guard_slim_results=round_search_guard_slim_results,
            )
            case_boundary_context = build_case_boundary_context(
                round_search_slim_results,
                guard_slim_results=round_search_guard_slim_results,
                cohort_signal_summary=cohort_signal_summary,
                plan_evidence_audit=plan_evidence_audit,
            )
            write_json(
                round_dir / "cohort_signal_summary.json",
                cohort_signal_summary,
            )
            write_json(
                round_dir / "plan_evidence_audit.json",
                plan_evidence_audit,
            )
            write_json(
                round_dir / "case_boundary_context.json",
                case_boundary_context,
            )
            review_input_fingerprint = build_review_input_fingerprint(
                current_rule=round_search_rule,
                current_plan=round_search_plan,
                current_slim_results=round_search_slim_results,
                current_guard_slim_results=round_search_guard_slim_results,
                cohort_signal_summary=cohort_signal_summary,
                plan_evidence_audit=plan_evidence_audit,
                case_boundary_context=case_boundary_context,
                rejected_update_memory=memory_with_active_refinement_feedback(
                    rejected_update_memory,
                    active_refinement_feedback,
                ),
            )
            if (
                cached_reviews
                and review_input_fingerprint == cached_review_input_fingerprint
            ):
                reviews = copy.deepcopy(cached_reviews)
                review_reuse_metadata = {
                    "reused": True,
                    "source_round": cached_review_source_round,
                    "current_round": round_index,
                    "input_fingerprint": review_input_fingerprint,
                    "reason": (
                        "unchanged_rule_plan_results_evidence_and_rejected_memory"
                    ),
                    "rejected_memory_in_fingerprint": True,
                }
                write_json(
                    round_dir / "review_reuse_metadata.json",
                    review_reuse_metadata,
                )
                print(
                    "[FewShot] Reusing Reviewer outputs from round "
                    f"{cached_review_source_round}; rule, plan, evidence, and "
                    "rejected memory are unchanged."
                )
            else:
                reviews = build_reviews(
                    round_search_full_results,
                    reviewer,
                    include_label_rationale=bool(args.review_label_rationale),
                    cohort_signal_summary=cohort_signal_summary,
                    plan_evidence_audit=plan_evidence_audit,
                    case_boundary_context=case_boundary_context,
                    rejected_update_memory=memory_with_active_refinement_feedback(
                        rejected_update_memory,
                        active_refinement_feedback,
                    ),
                )
                if any(
                    str(
                        (review.get("review_execution") or {}).get("status")
                        or ""
                    ).strip().lower()
                    == "failed"
                    for review in reviews
                    if isinstance(review, dict)
                ):
                    cached_review_input_fingerprint = ""
                    cached_reviews = []
                    cached_review_source_round = None
                else:
                    cached_review_input_fingerprint = review_input_fingerprint
                    cached_reviews = copy.deepcopy(reviews)
                    cached_review_source_round = round_index
            review_bundle = build_round_review_bundle(
                round_search_rule,
                round_search_summary,
                reviews,
                round_search_slim_results,
                negative_training_mode=args.negative_training_mode,
                negative_training_mode_source=args.negative_training_mode_source,
                cohort_signal_summary=cohort_signal_summary,
                guard_slim_results=round_search_guard_slim_results,
                plan_evidence_audit=plan_evidence_audit,
                case_boundary_context=case_boundary_context,
                rejected_update_memory=memory_with_active_refinement_feedback(
                    rejected_update_memory,
                    active_refinement_feedback,
                ),
                enable_advanced_candidate_lifecycles=bool(
                    args.enable_phase3_candidate_portfolio
                ),
            )
        review_bundle["search_state"] = {
            "source": search_base_source,
            "source_round": search_base_source_round,
            "rule_hash": artifact_hash(round_search_rule),
            "plan_hash": artifact_hash(round_search_plan),
            "incumbent_rule_hash": artifact_hash(current_rule),
            "incumbent_plan_hash": artifact_hash(current_plan),
        }
        review_bundle["budget_trace"] = rule_budget_trace(round_search_rule, args=args)
        review_bundle["rejected_update_memory"] = normalize_rejected_update_memory(
            review_bundle.get("rejected_update_memory")
            or memory_with_active_refinement_feedback(
                rejected_update_memory,
                active_refinement_feedback,
            )
        )
        review_bundle, preupdate_memory_gate = apply_rejected_memory_preupdate_gate(
            review_bundle
        )
        review_bundle["preupdate_rejected_memory_gate"] = preupdate_memory_gate
        write_json(
            round_dir / "preupdate_rejected_memory_gate.json",
            preupdate_memory_gate,
        )
        write_json(
            round_dir / "reviews_after_preupdate_memory_gate.json",
            _reviews_from_bundle(review_bundle),
        )
        packet_engineering_todos = build_packet_engineering_todos(review_bundle)
        review_bundle["packet_engineering_todos"] = packet_engineering_todos
        packet_blocker_audit = build_packet_blocker_audit(
            packet_engineering_todos,
            review_bundle,
        )
        review_bundle["packet_blocker_audit"] = packet_blocker_audit
        write_json(round_dir / "rejected_update_memory_input.json", review_bundle["rejected_update_memory"])
        write_json(round_dir / "reviews.json", reviews)
        write_json(round_dir / "review_bundle.json", review_bundle)
        write_json(
            round_dir / "diagnosis_routing_audit.json",
            dict(review_bundle.get("diagnosis_routing_audit") or {}),
        )
        if review_bundle.get("deferred_logic_structure_failures"):
            write_json(
                round_dir / "deferred_logic_structure_failures.json",
                {
                    "schema_version": "evotx.deferred_logic_structure.v1",
                    "logic_model": "all_positive_and_not_any_exclusion",
                    "failures": review_bundle[
                        "deferred_logic_structure_failures"
                    ],
                    "policy": (
                        "Deferred from Phase 1; do not weaken condition wording "
                        "to approximate an alternate OR branch."
                    ),
                },
            )
        if packet_engineering_todos:
            write_json(round_dir / "packet_engineering_todos.json", packet_engineering_todos)
            print(f"[FewShot] Packet/runtime engineering TODOs: {len(packet_engineering_todos)}")
        if packet_blocker_audit.get("hard_blocking_count"):
            write_json(round_dir / "packet_blocker_audit.json", packet_blocker_audit)
        target_counts = count_review_targets(review_bundle)
        print(f"[FewShot] Review targets: {stable_json_dumps(target_counts)}")
        if not reviews and not target_counts.get("rule") and not target_counts.get("plan"):
            stop_reason = "no_actionable_fp_fn_cases"
            break

        plan_updater.last_update_audits = []
        candidate_specs = build_candidate_specs(
            current_rule=round_search_rule,
            current_plan=round_search_plan,
            review_bundle=review_bundle,
            update_target=args.update_target,
            updater=updater,
            plan_updater=plan_updater,
            disable_stateful_bindings=bool(args.disable_stateful_bindings),
            enable_experimental_rule_hypotheses=bool(
                args.enable_experimental_rule_hypotheses
            ),
            enable_phase3_candidate_portfolio=bool(
                args.enable_phase3_candidate_portfolio
            ),
        )
        if review_bundle.get("minimal_updater_projection"):
            write_json(
                round_dir / "minimal_updater_projection.json",
                review_bundle["minimal_updater_projection"],
            )
        minimal_candidate_selection: Dict[str, Any] = {
            "schema_version": "evotx.minimal_candidate_selection.v1",
            "enabled": not bool(args.enable_phase3_candidate_portfolio),
        }
        if bool(args.enable_phase3_candidate_portfolio):
            candidate_specs, rule_delta_synthesis = synthesize_rule_candidate_specs(
                candidate_specs,
                current_rule=round_search_rule,
                current_plan=round_search_plan,
                current_slim_results=round_search_slim_results,
                review_bundle=review_bundle,
                max_candidates=args.max_synthesized_rule_candidates,
                disable_stateful_bindings=bool(args.disable_stateful_bindings),
            )
        else:
            candidate_specs, minimal_candidate_selection = (
                build_minimal_candidate_specs(candidate_specs)
            )
            minimal_candidate_selection["enabled"] = True
            rule_delta_synthesis = {
                "schema_version": "evotx.rule_delta_synthesis_round.v1",
                "enabled": False,
                "reason": "minimal_pipeline_uses_coherent_updater_rule_candidate",
                "output_candidate_count": len([
                    item for item in candidate_specs
                    if str(item.get("update_kind") or "") == "rule"
                ]),
                "selected_atomic_delta_ids": [
                    str(delta.get("delta_id") or "")
                    for item in candidate_specs
                    if str(item.get("update_kind") or "") == "rule"
                    for delta in list(item.get("selected_atomic_deltas") or [])
                    if isinstance(delta, dict)
                ],
            }
            write_json(
                round_dir / "minimal_candidate_selection.json",
                minimal_candidate_selection,
            )
        write_json(
            round_dir / "rule_delta_synthesis.json",
            rule_delta_synthesis,
        )
        phase3_activation_probe_audit = {
            "schema_version": "evotx.phase3.evidence_activation_probe.v1",
            "generated_probe_count": 0,
            "probes": [],
            "skipped": [],
            "enabled": bool(args.enable_phase3_candidate_portfolio),
        }
        if bool(args.enable_phase3_candidate_portfolio):
            candidate_specs, phase3_activation_probe_audit = (
                build_plan_activation_probe_specs(
                    candidate_specs,
                    current_plan=round_search_plan,
                    max_probes=max(1, min(2, int(args.max_phase3_candidates or 1))),
                )
            )
            phase3_activation_probe_audit["enabled"] = True
            write_json(
                round_dir / "phase3_activation_probe_audit.json",
                phase3_activation_probe_audit,
            )
            candidate_specs, phase3_portfolio = build_bounded_candidate_portfolio(
                candidate_specs,
                current_rule=round_search_rule,
                current_plan=round_search_plan,
                max_candidates=args.max_phase3_candidates,
                enable_joint_candidates=True,
                allow_dependency_restructure=bool(
                    args.enable_plan_dependency_restructure
                ),
                condition_evidence_dependencies=(
                    build_condition_evidence_dependencies(review_bundle)
                ),
            )
        else:
            phase3_portfolio = {
                "schema_version": "evotx.phase3_candidate_portfolio.v1",
                "enabled": False,
                "mode": "minimal_one_rule_one_plan",
                "input_candidate_count": len(candidate_specs),
                "selected_candidates": [
                    str(spec.get("name") or "") for spec in candidate_specs
                ],
                "minimal_candidate_selection": minimal_candidate_selection,
            }
            write_json(
                round_dir / "phase3_activation_probe_audit.json",
                phase3_activation_probe_audit,
            )
        generated_candidate_specs = list(candidate_specs)
        minimal_candidate_dedupe = {
            "enabled": False,
            "executed_candidate_names": [
                str(spec.get("name") or "") for spec in candidate_specs
            ],
            "exact_duplicates": [],
        }
        if not bool(args.enable_phase3_candidate_portfolio):
            candidate_specs, minimal_candidate_dedupe = (
                dedupe_minimal_candidate_specs(
                    candidate_specs,
                    fallback_plan=round_search_plan,
                    seen_fingerprints=seen_minimal_candidate_fingerprints,
                )
            )
            phase3_portfolio["minimal_exact_candidate_dedupe"] = (
                minimal_candidate_dedupe
            )
        phase3_portfolio["plan_signal_coverage"] = (
            build_plan_signal_coverage_audit(
                review_bundle,
                candidate_specs,
            )
        )
        proposed_candidate_specs = generated_candidate_specs
        write_json(round_dir / "phase3_candidate_portfolio.json", phase3_portfolio)
        candidate_specs, candidate_viability = filter_candidate_specs_by_viability(
            candidate_specs,
            current_plan=round_search_plan,
            current_rule=round_search_rule,
            current_slim_results=round_search_slim_results,
            review_bundle=review_bundle,
        )
        candidate_viability["preupdate_rejected_memory_gate"] = dict(
            preupdate_memory_gate
        )
        candidate_viability["phase3_candidate_portfolio"] = phase3_portfolio
        candidate_viability["phase3_activation_probe_audit"] = (
            phase3_activation_probe_audit
        )
        candidate_viability["minimal_exact_candidate_dedupe"] = (
            minimal_candidate_dedupe
        )
        plan_construction_rejections = build_plan_candidate_construction_rejections(
            plan_updater.last_update_audits
        )
        candidate_viability["pre_candidate_rejections"] = [
            *list(
                review_bundle.get(
                    "rule_candidate_construction_rejections"
                ) or []
            ),
            *plan_construction_rejections,
        ]
        write_json(round_dir / "candidate_viability.json", candidate_viability)
        if not candidate_specs:
            signal_terminal_audit = build_reachable_signal_terminal_audit(
                review_bundle,
                proposed_candidate_specs,
                candidate_viability=candidate_viability,
            )
            phase3_portfolio["signal_terminal_audit"] = signal_terminal_audit
            write_json(round_dir / "signal_terminal_audit.json", signal_terminal_audit)
            write_json(round_dir / "evolution_lineage.json", signal_terminal_audit)
            write_json(round_dir / "phase3_candidate_portfolio.json", phase3_portfolio)
            round_transition_audit.update({
                "candidate_count": 0,
                "validation_rule_hash": "",
                "validation_plan_hash": "",
                "rejection_reason": "no_viable_candidate_specs",
                "next_incumbent_rule_hash": artifact_hash(current_rule),
                "next_incumbent_plan_hash": artifact_hash(current_plan),
                "next_search_base_rule_hash": artifact_hash(round_search_rule),
                "next_search_base_plan_hash": artifact_hash(round_search_plan),
            })
            write_json(round_dir / "round_transition_audit.json", round_transition_audit)
            write_json(
                round_dir / "phase1_evolution_validation.json",
                {
                    "schema_version": "evotx.phase1_evolution_validation.v1",
                    "offline_replay": {
                        "purpose": (
                            "Reviewer-to-delta-to-synthesis construction only"
                        ),
                        "rule_delta_synthesis": rule_delta_synthesis,
                        "candidate_viability": candidate_viability,
                        "rejection_funnel": build_phase1_rejection_funnel(
                            rule_delta_synthesis,
                            candidate_viability,
                        ),
                    },
                    "runtime_ab": {
                        "required": True,
                        "status": "not_run_no_full_candidate_passed_viability",
                        "runtime_improvement_observed": False,
                        "promotion_claim_allowed": False,
                        "acceptance_policy_changed": False,
                        "promotion_safety_changed": False,
                    },
                },
            )
            rejected_update_memory = update_rejected_update_memory(
                rejected_update_memory,
                review_bundle=review_bundle,
                candidate_evaluations=[],
                candidate_viability=candidate_viability,
                round_index=round_index,
                attack_label=resolved_positive_label,
            )
            write_json(round_dir / "rejected_update_memory.json", rejected_update_memory)
            stop_reason = classify_zero_candidate_stop_reason(
                preupdate_memory_gate=preupdate_memory_gate,
                candidate_viability=candidate_viability,
                plan_construction_rejections=plan_construction_rejections,
                packet_engineering_todos=packet_engineering_todos,
                review_bundle=review_bundle,
            )
            construction_refinement_feedback = (
                build_construction_refinement_feedback(
                    candidate_viability,
                    round_index=round_index,
                )
            )
            deferred_refinement_feedback = (
                build_deferred_signal_refinement_feedback(
                    review_bundle,
                    signal_terminal_audit,
                    round_index=round_index,
                )
            )
            if deferred_refinement_feedback:
                construction_refinement_feedback = deferred_refinement_feedback
            consecutive_rejected_candidate_rounds += 1
            can_continue_after_rejection = bool(
                construction_refinement_feedback.get("refinable")
                and _should_continue_after_rejected_candidate(
                    args=args,
                    round_index=round_index,
                    consecutive_rejected_candidate_rounds=(
                        consecutive_rejected_candidate_rounds
                    ),
                )
            )
            if can_continue_after_rejection:
                active_refinement_feedback = dict(
                    construction_refinement_feedback
                )
                rejected_update_memory = memory_with_active_refinement_feedback(
                    rejected_update_memory,
                    active_refinement_feedback,
                )
                write_json(
                    round_dir / "rejected_update_memory.json",
                    rejected_update_memory,
                )
            round_transition_audit.update({
                "stop_reason": "" if can_continue_after_rejection else stop_reason,
                "round_terminal_reason": (
                    "deferred_next_round"
                    if can_continue_after_rejection
                    and deferred_refinement_feedback
                    else stop_reason
                ),
                "rejection_reason": (
                    "deferred_next_round"
                    if can_continue_after_rejection
                    and deferred_refinement_feedback
                    else stop_reason
                ),
                "packet_blocker_audit": packet_blocker_audit,
                "review_execution_failures": list(
                    review_bundle.get("review_execution_failures") or []
                ),
                "signal_terminal_audit": signal_terminal_audit,
                "next_incumbent_rule_hash": artifact_hash(current_rule),
                "next_incumbent_plan_hash": artifact_hash(current_plan),
                "next_search_base_rule_hash": artifact_hash(current_rule),
                "next_search_base_plan_hash": artifact_hash(current_plan),
                "next_search_base_source": (
                    "incumbent" if can_continue_after_rejection else search_base_source
                ),
                "next_search_base_source_round": incumbent_from_round,
                "refinement_feedback": construction_refinement_feedback,
                "continue_after_rejected_candidate": {
                    "enabled": bool(args.continue_after_rejected_candidate),
                    "will_continue": can_continue_after_rejection,
                    "consecutive_rejected_candidate_rounds": int(
                        consecutive_rejected_candidate_rounds
                    ),
                    "max_consecutive_rejected_candidate_rounds": int(
                        args.rejected_candidate_retry_rounds or 0
                    ),
                },
                "reuse_type": (
                    "deferred_signal_advance"
                    if can_continue_after_rejection
                    and deferred_refinement_feedback
                    else "construction_rejection_refinement"
                    if can_continue_after_rejection
                    else "candidate_count_zero_with_explicit_terminal_audit"
                ),
            })
            write_json(round_dir / "round_transition_audit.json", round_transition_audit)
            if can_continue_after_rejection:
                search_rule = current_rule
                search_rule_source = current_rule_source
                search_plan = current_plan
                search_plan_source = current_plan_source
                search_full_results = None
                search_summary = None
                search_guard_full_results = None
                search_guard_summary = None
                search_base_source = "incumbent"
                search_base_source_round = incumbent_from_round
                write_json(
                    round_dir / "rejected_candidate_continue.json",
                    {
                        "continued": True,
                        "reason": "candidate_construction_rejection_recorded",
                        "next_round": round_index + 1,
                        "consecutive_rejected_candidate_rounds": int(
                            consecutive_rejected_candidate_rounds
                        ),
                        "refinement_feedback": construction_refinement_feedback,
                    },
                )
                print(
                    "[FewShot] No candidate reached validation; continuing "
                    "from the accepted incumbent with the next attributed "
                    "signal."
                )
                continue
            break
        candidate_names = {str(spec.get("name") or "") for spec in candidate_specs}
        print(
            "[FewShot] Generated candidate specs: "
            f"plan_candidate={'plan_candidate' in candidate_names} "
            f"rule_candidate={'rule_candidate' in candidate_names} "
            f"combined_candidate={any(str(spec.get('update_kind') or '') == 'rule_plan' for spec in candidate_specs)} "
            f"portfolio={stable_json_dumps(phase3_portfolio)}"
        )

        planner_guidance = build_planner_guidance_from_review_bundle(review_bundle)
        candidate_evaluations: List[Dict[str, Any]] = []
        for spec in candidate_specs:
            if args.compress_rule and spec["rule"].version != round_search_rule.version:
                spec = dict(spec)
                spec["rule"] = updater.compress_rule(spec["rule"])
            candidate_evaluations.append(
                evaluate_candidate_spec(
                    spec=spec,
                    args=args,
                    cases=cases,
                    current_rule=round_search_rule,
                    current_full_results=round_search_full_results,
                    acceptance_full_results=incumbent_full_results or current_full_results,
                    acceptance_guard_full_results=(
                        incumbent_guard_full_results
                        if incumbent_guard_full_results is not None
                        else current_guard_full_results
                    ),
                    round_dir=round_dir,
                    round_index=round_index,
                    evaluator=evaluator,
                    tool_manifest=tool_manifest,
                    planner_guidance=planner_guidance,
                    hard_guard_cases=hard_guard_cases,
                    current_guard_full_results=round_search_guard_full_results,
                )
            )
        probe_confirmations: List[Dict[str, Any]] = []
        for probe in list(candidate_evaluations):
            comparison = dict(probe.get("comparison") or {})
            if not bool(probe.get("probe_only")) or not bool(
                comparison.get("probe_passed")
            ):
                continue
            confirmation_spec = build_confirmed_hypothesis_spec(
                probe,
                current_rule=round_search_rule,
                current_plan=round_search_plan,
            )
            if confirmation_spec is None:
                continue
            confirmation = evaluate_candidate_spec(
                spec=confirmation_spec,
                args=args,
                cases=cases,
                current_rule=round_search_rule,
                current_full_results=round_search_full_results,
                acceptance_full_results=incumbent_full_results or current_full_results,
                acceptance_guard_full_results=(
                    incumbent_guard_full_results
                    if incumbent_guard_full_results is not None
                    else current_guard_full_results
                ),
                round_dir=round_dir,
                round_index=round_index,
                evaluator=evaluator,
                tool_manifest=tool_manifest,
                planner_guidance=planner_guidance,
                hard_guard_cases=hard_guard_cases,
                current_guard_full_results=round_search_guard_full_results,
            )
            candidate_evaluations.append(confirmation)
            probe_confirmations.append({
                "source_probe": str(probe.get("name") or ""),
                "confirmation_candidate": str(confirmation.get("name") or ""),
                "confirmation_accept": bool(
                    (confirmation.get("comparison") or {}).get("accept")
                ),
                "hypothesis_source_signal_ids": list(
                    probe.get("hypothesis_source_signal_ids")
                    or probe.get("experimental_signal_ids")
                    or []
                ),
                "lifecycle_state": "confirmed_supported",
                "probe_result": "passed",
                "conversion_reason": (
                    "probe_passed_then_normal_validation_candidate_created"
                ),
            })
        attach_candidate_outcomes(
            candidate_evaluations,
            round_index=round_index,
            baseline_full_results=incumbent_full_results or current_full_results,
            review_bundle=review_bundle,
        )
        write_json(
            round_dir / "phase3_probe_confirmations.json",
            {
                "schema_version": "evotx.phase3_probe_confirmation.v1",
                "confirmations": probe_confirmations,
                "policy": (
                    "Experimental candidates are canary-only and cannot be "
                    "promoted. A passing probe is materialized as a separate "
                    "confirmation candidate that must pass full validation."
                ),
            },
        )
        primary_candidate_evaluations = list(candidate_evaluations)
        last_candidate_evaluations = list(candidate_evaluations)

        write_json(
            round_dir / "candidate_evaluations.json",
            [
                candidate_evaluation_summary(item, args=args)
                for item in candidate_evaluations
            ],
        )
        write_json(
            round_dir / "phase1_evolution_validation.json",
            {
                "schema_version": "evotx.phase1_evolution_validation.v1",
                "offline_replay": {
                    "purpose": (
                        "Validate Reviewer -> atomic delta -> synthesis "
                        "construction only; not sufficient for promotion."
                    ),
                    "rule_delta_synthesis": rule_delta_synthesis,
                    "candidate_viability": candidate_viability,
                    "rejection_funnel": build_phase1_rejection_funnel(
                        rule_delta_synthesis,
                        candidate_viability,
                        candidate_evaluations,
                    ),
                },
                "runtime_ab": {
                    "required": True,
                    "status": "completed",
                    "runtime_improvement_observed": any(
                        candidate_outcome_decision(item) == "accept"
                        for item in candidate_evaluations
                    ),
                    "promotion_claim_allowed": False,
                    "same_fewshot_case_count": len(cases),
                    "baseline_summary": current_summary,
                    "candidates": [
                        {
                            "name": item.get("name", ""),
                            "candidate_strategy": item.get(
                                "candidate_strategy", ""
                            ),
                            "summary": item.get("summary", {}),
                            "comparison": item.get("comparison", {}),
                            "canary_gate": item.get("canary_gate", {}),
                            "dependency_validation": item.get(
                                "dependency_validation", {}
                            ),
                            "targeted_repair_txs": candidate_repair_target_txs(
                                item.get("candidate_viability") or {}
                            ),
                        }
                        for item in candidate_evaluations
                    ],
                    "acceptance_policy_changed": False,
                    "promotion_safety_changed": False,
                },
            },
        )
        selected_candidate = select_candidate_evaluation(
            candidate_evaluations,
            accepted_only=True,
        )
        signal_terminal_audit = build_reachable_signal_terminal_audit(
            review_bundle,
            proposed_candidate_specs,
            candidate_viability=candidate_viability,
            candidate_evaluations=candidate_evaluations,
        )
        phase3_portfolio["signal_terminal_audit"] = signal_terminal_audit
        round_transition_audit["candidate_terminals"] = copy.deepcopy(
            signal_terminal_audit.get("candidates") or []
        )
        write_json(round_dir / "signal_terminal_audit.json", signal_terminal_audit)
        write_json(round_dir / "evolution_lineage.json", signal_terminal_audit)
        write_json(round_dir / "phase3_candidate_portfolio.json", phase3_portfolio)
        legacy_candidate = selected_candidate or select_candidate_evaluation(
            candidate_evaluations,
            accepted_only=False,
        )
        if legacy_candidate is not None:
            write_legacy_candidate_artifacts(round_dir, legacy_candidate)

        if selected_candidate is None and bool(args.enable_rule_partial_acceptance):
            print(
                "[FewShot] No full candidate passed; starting bounded rule "
                "condition/exclusion partial acceptance search."
            )
            partial_outcome = attempt_rule_partial_candidate_acceptance(
                args=args,
                candidate_evaluations=primary_candidate_evaluations,
                cases=cases,
                hard_guard_cases=hard_guard_cases,
                current_rule=round_search_rule,
                current_plan=round_search_plan,
                current_full_results=round_search_full_results,
                current_guard_full_results=round_search_guard_full_results,
                acceptance_full_results=incumbent_full_results or current_full_results,
                acceptance_guard_full_results=(
                    incumbent_guard_full_results
                    if incumbent_guard_full_results is not None
                    else current_guard_full_results
                ),
                round_dir=round_dir,
                round_index=round_index,
                evaluator=evaluator,
                tool_manifest=tool_manifest,
            )
            candidate_evaluations.extend(partial_outcome.get("evaluations", []))
            last_candidate_evaluations = list(candidate_evaluations)
            write_json(
                round_dir / "candidate_evaluations.json",
                [
                    candidate_evaluation_summary(item, args=args)
                    for item in candidate_evaluations
                ],
            )
            selected_candidate = partial_outcome.get("selected")
            if selected_candidate is not None:
                write_legacy_candidate_artifacts(round_dir, selected_candidate)
                print(
                    "[FewShot] Rule partial candidate accepted: "
                    f"deltas={stable_json_dumps((selected_candidate.get('partial_acceptance') or {}).get('accepted_delta_ids', []))}"
                )

        legacy_candidate = selected_candidate or select_candidate_evaluation(
            candidate_evaluations,
            accepted_only=False,
        )
        advanced_refinement_base, next_refinement_feedback = (
            select_refinement_base_candidate(
                candidate_evaluations,
                round_index=round_index,
                baseline_full_results=incumbent_full_results or current_full_results,
            )
        )
        # The default minimal loop always starts the next round from the last
        # accepted incumbent. Rejected candidates contribute feedback only.
        refinement_base_candidate = (
            advanced_refinement_base
            if bool(args.enable_phase3_candidate_portfolio)
            else None
        )
        if selected_candidate is None:
            feedback_candidate_name = str(
                next_refinement_feedback.get("source_candidate") or ""
            )
            feedback_candidate = next(
                (
                    item
                    for item in candidate_evaluations
                    if str(item.get("name") or "") == feedback_candidate_name
                ),
                None,
            )
            if feedback_candidate is not None:
                legacy_candidate = feedback_candidate
                write_legacy_candidate_artifacts(round_dir, legacy_candidate)
        legacy_outcome = dict(
            (legacy_candidate or {}).get("candidate_outcome") or {}
        )
        round_transition_audit["round_terminal_reason"] = (
            str(legacy_outcome.get("terminal_reason") or "accepted")
            if selected_candidate is not None
            else str(
                legacy_outcome.get("terminal_reason")
                or "no_candidate_accepted"
            )
        )
        round_transition_audit["refinement_feedback"] = (
            {
                "source_candidate": str(
                    next_refinement_feedback.get("source_candidate") or ""
                ),
                "owner": str(
                    next_refinement_feedback.get("source_update_kind") or ""
                ),
                "rejection_reason": str(
                    next_refinement_feedback.get("rejection_reason") or ""
                ),
                "refinable": bool(next_refinement_feedback.get("refinable")),
                "refinability_reason": str(
                    next_refinement_feedback.get("refinability_reason") or ""
                ),
                "next_allowed_refinement_kind": str(
                    next_refinement_feedback.get("next_allowed_refinement_kind")
                    or ""
                ),
                "next_round_action": str(
                    next_refinement_feedback.get("next_round_action") or "stop"
                ),
            }
            if selected_candidate is None
            else {}
        )
        if legacy_candidate is not None:
            transition_refinable = (
                False
                if selected_candidate is not None
                else bool(next_refinement_feedback.get("refinable"))
            )
            transition_refinability_reason = (
                ""
                if selected_candidate is not None
                else str(
                    next_refinement_feedback.get("refinability_reason") or ""
                )
            )
            round_transition_audit.update({
                "candidate_rule_hash": artifact_hash(legacy_candidate.get("rule")),
                "candidate_plan_hash": artifact_hash(legacy_candidate.get("plan")),
                "validation_rule_hash": artifact_hash(legacy_candidate.get("rule")),
                "validation_plan_hash": artifact_hash(legacy_candidate.get("plan")),
                "candidate_name": str(legacy_candidate.get("name") or ""),
                "candidate_status": str(
                    legacy_candidate.get("candidate_status") or "supported"
                ),
                "candidate_probe_only": bool(legacy_candidate.get("probe_only")),
                "candidate_count": len(candidate_evaluations),
                "rejection_reason": str(
                    legacy_outcome.get("detail_code")
                    or (legacy_candidate.get("comparison") or {}).get("reject_reason")
                    or ""
                ),
                "refinable": transition_refinable,
                "refinability_reason": transition_refinability_reason,
            })
            write_json(round_dir / "round_transition_audit.json", round_transition_audit)

        rejected_update_memory = update_rejected_update_memory(
            rejected_update_memory,
            review_bundle=review_bundle,
            candidate_evaluations=candidate_evaluations,
            candidate_viability=candidate_viability,
            round_index=round_index,
            attack_label=resolved_positive_label,
        )
        write_json(round_dir / "rejected_update_memory.json", rejected_update_memory)
        print(
            "[FewShot] Rejected update memory entries: "
            f"{len(list(rejected_update_memory.get('entries') or []))}"
        )

        if selected_candidate is None:
            rejected_rule_candidate = select_repair_candidate(
                primary_candidate_evaluations,
                args=args,
            )
            if rejected_rule_candidate is None:
                guard_rejected_candidates = [
                    candidate_evaluation_summary(item, args=args)
                    for item in candidate_evaluations
                    if _candidate_rejected_by_guard(item)
                ]
                consecutive_rejected_candidate_rounds += 1
                next_round_action = str(
                    (next_refinement_feedback or {}).get("next_round_action")
                    or "stop"
                )
                can_continue_after_rejection = bool(
                    next_round_action in {"refine", "advance_signal"}
                    and _should_continue_after_rejected_candidate(
                        args=args,
                        round_index=round_index,
                        consecutive_rejected_candidate_rounds=(
                            consecutive_rejected_candidate_rounds
                        ),
                    )
                )
                write_json(
                    round_dir / "repair_status.json",
                    {
                        "attempted": False,
                        "reason": (
                            "candidate_rejected_by_hard_negative_guard"
                            if guard_rejected_candidates
                            and not bool(args.allow_guard_repair)
                            else "no_repair_eligible_candidate"
                        ),
                        "allow_guard_repair": bool(args.allow_guard_repair),
                        "guard_repair_enabled": bool(args.allow_guard_repair),
                        "continue_after_rejected_candidate": {
                            "enabled": bool(args.continue_after_rejected_candidate),
                            "will_continue": bool(can_continue_after_rejection),
                            "consecutive_rejected_candidate_rounds": int(
                                consecutive_rejected_candidate_rounds
                            ),
                            "max_consecutive_rejected_candidate_rounds": int(
                                args.rejected_candidate_retry_rounds or 0
                            ),
                            "remaining_rounds": max(0, int(args.max_rounds) - round_index),
                        },
                        "guard_rejected_candidates": guard_rejected_candidates,
                        "policy": candidate_repair_policy(args).to_dict(),
                        "candidate_evaluations": [
                            candidate_repair_diagnostic(item, args=args)
                            for item in primary_candidate_evaluations
                            if str(item.get("update_kind") or "") in {"plan", "rule"}
                        ],
                    },
                )
                if can_continue_after_rejection:
                    if incumbent_full_results is not None:
                        reusable_current_full_results = copy.deepcopy(incumbent_full_results)
                        reusable_current_summary = dict(incumbent_summary or {})
                        reusable_current_guard_full_results = (
                            copy.deepcopy(incumbent_guard_full_results)
                            if incumbent_guard_full_results is not None
                            else None
                        )
                        reusable_current_guard_summary = dict(
                            incumbent_guard_summary or {}
                        )
                        reusable_current_from_round = incumbent_from_round
                        reusable_current_source_file = incumbent_source_file
                        reusable_current_guard_source_file = incumbent_guard_source_file
                        reusable_current_reuse_type = "incumbent_anchor_unchanged_current"
                    if refinement_base_candidate is not None:
                        search_rule = refinement_base_candidate["rule"]
                        search_rule_source = str(
                            Path(str(refinement_base_candidate.get("candidate_dir") or ""))
                            / "candidate_rule.json"
                        )
                        search_plan = refinement_base_candidate.get("plan")
                        search_plan_source = (
                            str(
                                Path(str(refinement_base_candidate.get("candidate_dir") or ""))
                                / "candidate_plan.json"
                            )
                            if search_plan is not None
                            else ""
                        )
                        search_full_results = copy.deepcopy(
                            refinement_base_candidate.get("full_results") or []
                        )
                        search_summary = dict(
                            refinement_base_candidate.get("summary") or {}
                        )
                        search_guard_full_results = copy.deepcopy(
                            refinement_base_candidate.get("guard_full_results") or []
                        )
                        search_guard_summary = dict(
                            refinement_base_candidate.get("guard_summary") or {}
                        )
                        search_base_source = (
                            f"round_{round_index:02d}/candidate_artifacts/"
                            f"{refinement_base_candidate.get('name', '')}"
                        )
                        search_base_source_round = round_index
                        active_refinement_feedback = dict(next_refinement_feedback)
                        rejected_update_memory = memory_with_active_refinement_feedback(
                            rejected_update_memory,
                            active_refinement_feedback,
                        )
                    else:
                        search_rule = current_rule
                        search_rule_source = current_rule_source
                        search_plan = current_plan
                        search_plan_source = current_plan_source
                        search_full_results = None
                        search_summary = None
                        search_guard_full_results = None
                        search_guard_summary = None
                        search_base_source = "incumbent"
                        search_base_source_round = incumbent_from_round
                        active_refinement_feedback = dict(next_refinement_feedback or {})
                        rejected_update_memory = memory_with_active_refinement_feedback(
                            rejected_update_memory,
                            active_refinement_feedback,
                        )
                    write_json(round_dir / "rejected_update_memory.json", rejected_update_memory)
                    round_transition_audit.update({
                        "accepted": False,
                        "rejection_reason": str(
                            round_transition_audit.get("rejection_reason")
                            or (next_refinement_feedback or {}).get("rejection_reason")
                            or "candidate_rejected"
                        ),
                        "refinable": bool(
                            (next_refinement_feedback or {}).get("refinable")
                        ),
                        "refinability_reason": str(
                            (next_refinement_feedback or {}).get("refinability_reason")
                            or ""
                        ),
                        "next_incumbent_rule_hash": artifact_hash(current_rule),
                        "next_incumbent_plan_hash": artifact_hash(current_plan),
                        "next_search_base_rule_hash": artifact_hash(search_rule),
                        "next_search_base_plan_hash": artifact_hash(search_plan),
                        "next_search_base_source": search_base_source,
                        "next_search_base_source_round": search_base_source_round,
                        "reuse_type": (
                            "rejected_candidate_refinement"
                            if refinement_base_candidate is not None
                            else "incumbent_anchor_unchanged_current"
                        ),
                    })
                    write_json(round_dir / "round_transition_audit.json", round_transition_audit)
                    write_json(
                        round_dir / "rejected_candidate_continue.json",
                        {
                            "continued": True,
                            "reason": "rejected_candidate_memory_recorded",
                            "next_round": round_index + 1,
                            "consecutive_rejected_candidate_rounds": int(
                                consecutive_rejected_candidate_rounds
                            ),
                            "rejected_update_memory_entry_count": len(
                                list(rejected_update_memory.get("entries") or [])
                            ),
                            "refinable": bool(
                                (next_refinement_feedback or {}).get("refinable")
                            ),
                            "search_base_source": search_base_source,
                        },
                    )
                    print(
                        "[FewShot] Candidate rejected without repair; continuing "
                        "with rejected_update_memory for the next round."
                    )
                    continue
                stop_reason = "candidate_artifact_rejected_no_repair_eligible"
                round_transition_audit.update({
                    "stop_reason": stop_reason,
                    "accepted": False,
                    "next_incumbent_rule_hash": artifact_hash(current_rule),
                    "next_incumbent_plan_hash": artifact_hash(current_plan),
                    "next_search_base_rule_hash": artifact_hash(current_rule),
                    "next_search_base_plan_hash": artifact_hash(current_plan),
                    "next_search_base_source": "incumbent",
                    "reuse_type": "terminal_rejected_candidate_not_refined",
                })
                write_json(round_dir / "round_transition_audit.json", round_transition_audit)
                break
            repair = attempt_repair_rejected_candidate(
                args=args,
                cases=cases,
                hard_guard_cases=hard_guard_cases,
                current_full_results=current_full_results,
                current_guard_full_results=current_guard_full_results,
                current_rule=current_rule,
                current_rule_source=current_rule_source,
                candidate_rule=rejected_rule_candidate["rule"],
                candidate_plan=rejected_rule_candidate.get("plan"),
                update_kind=rejected_rule_candidate.get("update_kind", ""),
                candidate_full_results=rejected_rule_candidate["full_results"],
                candidate_guard_full_results=rejected_rule_candidate.get("guard_full_results", []),
                candidate_slim_results=rejected_rule_candidate["slim_results"],
                candidate_summary=rejected_rule_candidate["summary"],
                comparison=rejected_rule_candidate["comparison"],
                round_dir=round_dir,
                round_index=round_index,
                reviewer=reviewer,
                updater=updater,
                plan_updater=plan_updater,
                evaluator=evaluator,
                tool_manifest=tool_manifest,
                acceptance_full_results=incumbent_full_results or current_full_results,
                acceptance_guard_full_results=(
                    incumbent_guard_full_results
                    if incumbent_guard_full_results is not None
                    else current_guard_full_results
                ),
            )
            if repair.get("accepted"):
                current_rule = repair["rule"]
                repair_kind = str(repair.get("repair_kind") or "rule")
                accepted_rounds += 1
                consecutive_rejected_candidate_rounds = 0
                final_rule_path = rule_store.save(
                    current_rule,
                    attack_label=resolved_positive_label,
                    latest=False,
                )
                current_rule_source = str(final_rule_path)
                current_plan = repair.get("plan") or build_initial_baseline_plan(current_rule)
                final_plan = current_plan
                final_plan_path = str(
                    plan_store.save(
                        current_plan,
                        attack_label=resolved_positive_label,
                        latest=False,
                    )
                )
                current_plan_source = final_plan_path
                final_summary = dict(repair.get("summary", {}))
                reusable_current_full_results = retarget_reused_results(
                    repair.get("full_results", []),
                    rule_source=current_rule_source,
                    plan_source=current_plan_source,
                    reuse_metadata={
                        "reused": True,
                        "source_round": round_index,
                        "source_file": f"round_{round_index:02d}/repair_full_results.json",
                        "rule_id": current_rule.rule_id,
                        "rule_version": current_rule.version,
                        "rule_source": current_rule_source,
                        "plan_source": current_plan_source,
                        "reuse_type": "accepted_candidate_as_next_current",
                        "repair_kind": repair_kind,
                    },
                )
                reusable_current_summary = dict(final_summary)
                reusable_current_from_round = round_index
                reusable_current_source_file = f"round_{round_index:02d}/repair_full_results.json"
                reusable_current_guard_source_file = f"round_{round_index:02d}/repair_guard_full_results.json"
                reusable_current_reuse_type = "accepted_repair_as_next_current"
                reusable_current_guard_full_results = None
                reusable_current_guard_summary = None
                latest_final_full_results = copy.deepcopy(repair.get("full_results", []))
                latest_final_summary = dict(final_summary)
                latest_final_guard_full_results = copy.deepcopy(
                    repair.get("guard_full_results", [])
                )
                latest_final_guard_summary = dict(repair.get("guard_summary", {}) or {})
                latest_final_rule_source = current_rule_source
                latest_final_plan_source = current_plan_source
                incumbent_full_results = copy.deepcopy(reusable_current_full_results)
                incumbent_summary = dict(final_summary)
                incumbent_guard_full_results = copy.deepcopy(
                    latest_final_guard_full_results or []
                )
                incumbent_guard_summary = dict(latest_final_guard_summary or {})
                incumbent_from_round = round_index
                incumbent_source_file = reusable_current_source_file
                incumbent_guard_source_file = reusable_current_guard_source_file
                incumbent_rule_hash = artifact_hash(current_rule)
                incumbent_plan_hash = artifact_hash(current_plan)
                search_rule = current_rule
                search_rule_source = current_rule_source
                search_plan = current_plan
                search_plan_source = current_plan_source
                search_full_results = copy.deepcopy(reusable_current_full_results)
                search_summary = dict(final_summary)
                search_guard_full_results = copy.deepcopy(
                    latest_final_guard_full_results or []
                )
                search_guard_summary = dict(latest_final_guard_summary or {})
                search_base_source = "accepted_repair_as_next_current"
                search_base_source_round = round_index
                active_refinement_feedback = {}
                rejected_update_memory = memory_with_active_refinement_feedback(
                    rejected_update_memory,
                    None,
                )
                round_transition_audit.update({
                    "accepted": True,
                    "accept_reason": str(repair.get("accept_reason") or ""),
                    "repair_kind": repair_kind,
                    "next_incumbent_rule_hash": artifact_hash(current_rule),
                    "next_incumbent_plan_hash": artifact_hash(current_plan),
                    "next_search_base_rule_hash": artifact_hash(search_rule),
                    "next_search_base_plan_hash": artifact_hash(search_plan),
                    "next_search_base_source": search_base_source,
                    "next_search_base_source_round": search_base_source_round,
                    "reuse_type": "accepted_repair_as_next_current",
                })
                write_json(round_dir / "round_transition_audit.json", round_transition_audit)
                print(
                    f"[FewShot] Repair {repair_kind} accepted after candidate rejection: "
                    f"{current_rule.rule_id} v{current_rule.version}"
                )
                continue

            rejected_kind = str(rejected_rule_candidate.get("update_kind") or "")
            consecutive_rejected_candidate_rounds += 1
            next_round_action = str(
                (next_refinement_feedback or {}).get("next_round_action")
                or "stop"
            )
            can_continue_after_rejection = bool(
                next_round_action in {"refine", "advance_signal"}
                and _should_continue_after_rejected_candidate(
                    args=args,
                    round_index=round_index,
                    consecutive_rejected_candidate_rounds=(
                        consecutive_rejected_candidate_rounds
                    ),
                )
            )
            if can_continue_after_rejection:
                if incumbent_full_results is not None:
                    reusable_current_full_results = copy.deepcopy(incumbent_full_results)
                    reusable_current_summary = dict(incumbent_summary or {})
                    reusable_current_guard_full_results = (
                        copy.deepcopy(incumbent_guard_full_results)
                        if incumbent_guard_full_results is not None
                        else None
                    )
                    reusable_current_guard_summary = dict(
                        incumbent_guard_summary or {}
                    )
                    reusable_current_from_round = incumbent_from_round
                    reusable_current_source_file = incumbent_source_file
                    reusable_current_guard_source_file = incumbent_guard_source_file
                    reusable_current_reuse_type = "incumbent_anchor_unchanged_current"
                if refinement_base_candidate is not None:
                    search_rule = refinement_base_candidate["rule"]
                    search_rule_source = str(
                        Path(str(refinement_base_candidate.get("candidate_dir") or ""))
                        / "candidate_rule.json"
                    )
                    search_plan = refinement_base_candidate.get("plan")
                    search_plan_source = (
                        str(
                            Path(str(refinement_base_candidate.get("candidate_dir") or ""))
                            / "candidate_plan.json"
                        )
                        if search_plan is not None
                        else ""
                    )
                    search_full_results = copy.deepcopy(
                        refinement_base_candidate.get("full_results") or []
                    )
                    search_summary = dict(refinement_base_candidate.get("summary") or {})
                    search_guard_full_results = copy.deepcopy(
                        refinement_base_candidate.get("guard_full_results") or []
                    )
                    search_guard_summary = dict(
                        refinement_base_candidate.get("guard_summary") or {}
                    )
                    search_base_source = (
                        f"round_{round_index:02d}/candidate_artifacts/"
                        f"{refinement_base_candidate.get('name', '')}"
                    )
                    search_base_source_round = round_index
                    active_refinement_feedback = dict(next_refinement_feedback)
                    rejected_update_memory = memory_with_active_refinement_feedback(
                        rejected_update_memory,
                        active_refinement_feedback,
                    )
                else:
                    search_rule = current_rule
                    search_rule_source = current_rule_source
                    search_plan = current_plan
                    search_plan_source = current_plan_source
                    search_full_results = None
                    search_summary = None
                    search_guard_full_results = None
                    search_guard_summary = None
                    search_base_source = "incumbent"
                    search_base_source_round = incumbent_from_round
                    active_refinement_feedback = dict(next_refinement_feedback or {})
                    rejected_update_memory = memory_with_active_refinement_feedback(
                        rejected_update_memory,
                        active_refinement_feedback,
                    )
                write_json(round_dir / "rejected_update_memory.json", rejected_update_memory)
                round_transition_audit.update({
                    "accepted": False,
                    "rejection_reason": str(
                        round_transition_audit.get("rejection_reason")
                        or (next_refinement_feedback or {}).get("rejection_reason")
                        or "repair_rejected"
                    ),
                    "refinable": bool(
                        (next_refinement_feedback or {}).get("refinable")
                    ),
                    "refinability_reason": str(
                        (next_refinement_feedback or {}).get("refinability_reason")
                        or ""
                    ),
                    "next_incumbent_rule_hash": artifact_hash(current_rule),
                    "next_incumbent_plan_hash": artifact_hash(current_plan),
                    "next_search_base_rule_hash": artifact_hash(search_rule),
                    "next_search_base_plan_hash": artifact_hash(search_plan),
                    "next_search_base_source": search_base_source,
                    "next_search_base_source_round": search_base_source_round,
                    "reuse_type": (
                        "rejected_candidate_refinement"
                        if refinement_base_candidate is not None
                        else "incumbent_anchor_unchanged_current"
                    ),
                })
                write_json(round_dir / "round_transition_audit.json", round_transition_audit)
                write_json(
                    round_dir / "rejected_candidate_continue.json",
                    {
                        "continued": True,
                        "reason": "repair_rejected_memory_recorded",
                        "next_round": round_index + 1,
                        "rejected_kind": rejected_kind,
                        "consecutive_rejected_candidate_rounds": int(
                            consecutive_rejected_candidate_rounds
                        ),
                        "rejected_update_memory_entry_count": len(
                            list(rejected_update_memory.get("entries") or [])
                        ),
                        "refinable": bool(
                            (next_refinement_feedback or {}).get("refinable")
                        ),
                        "search_base_source": search_base_source,
                    },
                )
                print(
                    "[FewShot] Candidate repair did not pass; continuing with "
                    "rejected_update_memory for the next round."
                )
                continue
            stop_reason = (
                "candidate_plan_rejected"
                if rejected_kind == "plan"
                else "candidate_rule_rejected"
            )
            round_transition_audit.update({
                "stop_reason": stop_reason,
                "accepted": False,
                "next_incumbent_rule_hash": artifact_hash(current_rule),
                "next_incumbent_plan_hash": artifact_hash(current_plan),
                "next_search_base_rule_hash": artifact_hash(current_rule),
                "next_search_base_plan_hash": artifact_hash(current_plan),
                "next_search_base_source": "incumbent",
                "reuse_type": "terminal_repair_rejected",
            })
            write_json(round_dir / "round_transition_audit.json", round_transition_audit)
            break

        candidate_rule = selected_candidate["rule"]
        candidate_plan = selected_candidate["plan"]
        candidate_summary = selected_candidate["summary"]
        candidate_full_results = selected_candidate["full_results"]
        update_kind = selected_candidate["update_kind"]
        selected_dependency_validation = dict(
            selected_candidate.get("dependency_validation") or {}
        )
        if (
            selected_dependency_validation.get("dependency_validation_status")
            not in {None, "", "passed"}
        ):
            raise PlanValidationError(
                "Accepted candidate reached promotion with an invalid dependency "
                "contract: "
                + str(
                    selected_dependency_validation.get(
                        "dependency_rejection_reason", ""
                    )
                )
            )
        print(
            f"[FewShot] Accepted {update_kind} candidate: "
            f"{selected_candidate['name']} "
            f"reason={selected_candidate['comparison'].get('accept_reason', '')}"
        )
        consecutive_rejected_candidate_rounds = 0
        rule_changed = candidate_rule.version != current_rule.version
        current_rule = candidate_rule
        accepted_rounds += 1
        final_rule_path = rule_store.save(
            current_rule,
            attack_label=resolved_positive_label,
            latest=False,
        )
        current_rule_source = str(final_rule_path)
        if candidate_plan is not None:
            current_plan = candidate_plan
            final_plan = candidate_plan
            final_plan_path = str(
                plan_store.save(
                    current_plan,
                    attack_label=resolved_positive_label,
                    latest=False,
                )
            )
            current_plan_source = final_plan_path
        elif rule_changed:
            current_plan = build_initial_baseline_plan(current_rule)
            final_plan = current_plan
            final_plan_path = str(
                plan_store.save(
                    current_plan,
                    attack_label=resolved_positive_label,
                    latest=False,
                )
            )
            current_plan_source = final_plan_path
        final_summary = candidate_summary
        reusable_current_full_results = retarget_reused_results(
            candidate_full_results,
            rule_source=current_rule_source,
            plan_source=current_plan_source,
            reuse_metadata={
                "reused": True,
                "source_round": round_index,
                "source_file": f"round_{round_index:02d}/candidate_full_results.json",
                "rule_id": current_rule.rule_id,
                "rule_version": current_rule.version,
                "rule_source": current_rule_source,
                "plan_source": current_plan_source,
                "reuse_type": "accepted_candidate_as_next_current",
            },
        )
        reusable_current_summary = dict(candidate_summary)
        reusable_current_from_round = round_index
        reusable_current_source_file = f"round_{round_index:02d}/candidate_full_results.json"
        reusable_current_guard_source_file = f"round_{round_index:02d}/candidate_guard_full_results.json"
        reusable_current_reuse_type = "accepted_candidate_as_next_current"
        latest_final_full_results = copy.deepcopy(candidate_full_results)
        latest_final_summary = dict(candidate_summary)
        latest_final_guard_full_results = copy.deepcopy(
            selected_candidate.get("guard_full_results", [])
        )
        latest_final_guard_summary = dict(selected_candidate.get("guard_summary", {}) or {})
        latest_final_rule_source = current_rule_source
        latest_final_plan_source = current_plan_source
        if hard_guard_cases and selected_candidate.get("guard_full_results") is not None:
            reusable_current_guard_full_results = retarget_reused_results(
                selected_candidate.get("guard_full_results", []),
                rule_source=current_rule_source,
                plan_source=current_plan_source,
                reuse_metadata={
                    "reused": True,
                    "source_round": round_index,
                    "source_file": f"round_{round_index:02d}/candidate_guard_full_results.json",
                    "rule_id": current_rule.rule_id,
                    "rule_version": current_rule.version,
                    "rule_source": current_rule_source,
                    "plan_source": current_plan_source,
                    "guard_role": "hard_negative_guard",
                    "reuse_type": "accepted_candidate_as_next_current",
                },
            )
            reusable_current_guard_summary = dict(
                selected_candidate.get("guard_summary") or {}
            )
        else:
            reusable_current_guard_full_results = None
            reusable_current_guard_summary = None
        incumbent_full_results = copy.deepcopy(reusable_current_full_results)
        incumbent_summary = dict(candidate_summary)
        incumbent_guard_full_results = (
            copy.deepcopy(reusable_current_guard_full_results)
            if reusable_current_guard_full_results is not None
            else []
        )
        incumbent_guard_summary = dict(reusable_current_guard_summary or {})
        incumbent_from_round = round_index
        incumbent_source_file = reusable_current_source_file
        incumbent_guard_source_file = reusable_current_guard_source_file
        incumbent_rule_hash = artifact_hash(current_rule)
        incumbent_plan_hash = artifact_hash(current_plan)
        search_rule = current_rule
        search_rule_source = current_rule_source
        search_plan = current_plan
        search_plan_source = current_plan_source
        search_full_results = copy.deepcopy(reusable_current_full_results)
        search_summary = dict(candidate_summary)
        search_guard_full_results = (
            copy.deepcopy(reusable_current_guard_full_results)
            if reusable_current_guard_full_results is not None
            else []
        )
        search_guard_summary = dict(reusable_current_guard_summary or {})
        search_base_source = "accepted_candidate_as_next_current"
        search_base_source_round = round_index
        active_refinement_feedback = {}
        rejected_update_memory = memory_with_active_refinement_feedback(
            rejected_update_memory,
            None,
        )
        round_transition_audit.update({
            "accepted": True,
            "accept_reason": str(
                selected_candidate["comparison"].get("accept_reason", "")
            ),
            "rejection_reason": "",
            "refinable": False,
            "next_incumbent_rule_hash": artifact_hash(current_rule),
            "next_incumbent_plan_hash": artifact_hash(current_plan),
            "next_search_base_rule_hash": artifact_hash(search_rule),
            "next_search_base_plan_hash": artifact_hash(search_plan),
            "next_search_base_source": search_base_source,
            "next_search_base_source_round": search_base_source_round,
            "reuse_type": "accepted_candidate_as_next_current",
        })
        write_json(round_dir / "round_transition_audit.json", round_transition_audit)

    run_finished_at = _now_iso()
    run_elapsed_seconds = round(time.perf_counter() - run_started_perf, 3)
    run_timing = {
        "started_at": run_started_at,
        "finished_at": run_finished_at,
        "elapsed_seconds": run_elapsed_seconds,
        "elapsed_hms": _format_elapsed_hms(run_elapsed_seconds),
    }

    final_validation = run_final_validation(
        args=args,
        cases=cases,
        hard_guard_cases=hard_guard_cases,
        current_rule=current_rule,
        current_rule_source=current_rule_source,
        current_plan=current_plan,
        current_plan_source=current_plan_source,
        baseline_full_results=baseline_full_results_for_validation,
        baseline_summary=baseline_summary_for_validation,
        baseline_guard_full_results=baseline_guard_full_results_for_validation,
        baseline_guard_summary=baseline_guard_summary_for_validation,
        latest_final_full_results=latest_final_full_results,
        latest_final_guard_full_results=latest_final_guard_full_results,
        artifacts_dir=artifacts_dir,
        evaluator=evaluator,
        tool_manifest=tool_manifest,
    )
    strict_validation_failed = (
        str(args.final_validation_mode or "strict") == "strict"
        and not bool(final_validation.get("passed"))
    )
    final_dependency_validation = (
        dependency_consistency_report(current_plan)
        if current_plan is not None
        else {
            "schema_version": "evotx.plan_dependency_validation.v1",
            "dependency_validation_status": "rejected",
            "dependency_rejection_reason": "missing_final_plan",
            "dangling_dependency_count": 0,
            "binding_mismatch_count": 1,
            "errors": [{"code": "missing_final_plan"}],
        }
    )
    if current_plan is not None:
        try:
            PlanValidator().validate_against_rule(
                current_plan,
                current_rule,
                attack_label=str(
                    (current_rule.metadata or {}).get("attack_label") or ""
                ),
            )
        except PlanValidationError as exc:
            final_dependency_validation["dependency_validation_status"] = (
                "rejected"
            )
            final_dependency_validation["dependency_rejection_reason"] = str(exc)
            final_dependency_validation.setdefault("errors", []).append({
                "code": "final_rule_plan_dependency_validation_failed",
                "error": str(exc),
            })
    write_json(
        artifacts_dir / "final_dependency_validation.json",
        final_dependency_validation,
    )
    latest_promoted = (
        not strict_validation_failed
        and final_dependency_validation.get("dependency_validation_status")
        == "passed"
    )
    latest_alias_rule_path = None
    latest_alias_plan_path = None
    latest_alias_written = False
    if latest_promoted:
        latest_alias_rule_path = rule_store.save(
            current_rule,
            attack_label=resolved_positive_label,
            latest=True,
        )
        if current_plan is not None:
            latest_alias_plan_path = plan_store.save(
                current_plan,
                attack_label=resolved_positive_label,
                latest=True,
            )
        latest_alias_written = True
        print("[FewShot] Latest artifacts promoted after final validation.")
    else:
        print(
            "[FewShot] Strict final validation failed; versioned artifacts were "
            "kept, but latest aliases were not written. Eval-latest scripts "
            "should fall back to v1 when no validated latest exists."
        )

    report = build_fewshot_final_report(
        attack_label=resolved_positive_label,
        attack_description=resolve_attack_description(
            attack_description=args.attack_description,
            label=resolved_positive_label,
        ),
        cases_file=args.cases_file,
        pos_csv=args.pos_csv,
        neg_csv=args.neg_csv,
        benign_csv=args.benign_csv,
        malicious_csv=args.malicious_csv,
        episode=args.episode,
        tool_manifest_path=args.tool_manifest,
        use_environment=args.use_environment,
        base_cache_dir=args.base_cache_dir,
        force_rebuild_packet=args.force_rebuild_packet,
        enable_source_tools=args.enable_source_tools,
        source_cache_dir=args.source_cache_dir,
        force_refresh_source=args.force_refresh_source,
        max_view_chars=args.max_view_chars,
        max_context_chars=args.max_context_chars,
        rules_dir=args.rules_dir,
        artifacts_dir=artifacts_dir,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        adaptive_model=args.adaptive_model,
        adaptive_provider=args.adaptive_provider,
        cold_start_model=args.cold_start_model,
        planner_model=args.planner_model,
        judge_model=args.judge_model,
        env_model=args.env_model,
        review_model=args.review_model,
        update_model=args.update_model,
        max_rounds=args.max_rounds,
        stop_errors=args.stop_errors,
        max_fp_increase=args.max_fp_increase,
        max_fn_increase=args.max_fn_increase,
        max_uncertain_increase=args.max_uncertain_increase,
        max_uncertain_benign_increase=args.max_uncertain_benign_increase,
        enable_candidate_repair=args.enable_candidate_repair,
        temporary_fp_increase=args.temporary_fp_increase,
        repair_min_fn_decrease=args.repair_min_fn_decrease,
        max_repair_rounds=args.max_repair_rounds,
        compress_rule=args.compress_rule,
        label_counts=counts,
        negative_training_mode=args.negative_training_mode,
        negative_training_mode_source=args.negative_training_mode_source,
        initial_rule=initial_rule,
        initial_rule_path=initial_rule_path,
        final_rule=current_rule,
        final_rule_path=final_rule_path,
        accepted_rounds=accepted_rounds,
        stop_reason=stop_reason,
        final_summary=final_summary,
        plans_dir=args.plans_dir,
        initial_plan=initial_plan,
        initial_plan_path=initial_plan_path,
        final_plan=final_plan,
        final_plan_path=final_plan_path,
        run_timing=run_timing,
        llm_runtime_config=resolved_llm_configs,
    )
    report.setdefault("inputs", {})["initial_artifact_provenance"] = (
        copy.deepcopy(initial_artifact_provenance)
    )
    report.setdefault("inputs", {}).setdefault("case_sources", {})[
        "num_shots"
    ] = args.num_shots
    report.setdefault("dataset", {})["num_shots_per_class"] = args.num_shots
    initial_rule_ref = report.setdefault("detector_context", {}).setdefault(
        "initial_rule", {}
    )
    initial_rule_ref.update({
        "input_path": initial_artifact_provenance["rule"]["cli_path"],
        "input_resolved_path": initial_artifact_provenance["rule"][
            "resolved_path"
        ],
        "archive_path": initial_artifact_provenance["rule"]["archive_path"],
        "initialization_mode": initial_artifact_provenance["rule"][
            "initialization_mode"
        ],
    })
    initial_plan_ref = report.setdefault("detector_context", {}).setdefault(
        "initial_plan", {}
    )
    initial_plan_ref.update({
        "input_path": initial_artifact_provenance["plan"]["cli_path"],
        "input_resolved_path": initial_artifact_provenance["plan"][
            "resolved_path"
        ],
        "archive_path": initial_artifact_provenance["plan"]["archive_path"],
        "initialization_mode": initial_artifact_provenance["plan"][
            "initialization_mode"
        ],
    })
    report["evolution_policy"] = {
        "evolution_mode": args.evolution_mode,
        "update_target": args.update_target,
        "final_validation_mode": args.final_validation_mode,
        "final_validation_result_source": args.final_validation_result_source,
        "force_final_validation_rerun": bool(args.force_final_validation_rerun),
        "continue_after_rejected_candidate": bool(
            args.continue_after_rejected_candidate
        ),
        "rejected_candidate_retry_rounds": int(
            args.rejected_candidate_retry_rounds or 0
        ),
        "consecutive_rejected_candidate_rounds_at_finish": int(
            consecutive_rejected_candidate_rounds
        ),
        "mode_defaults_applied": dict(getattr(args, "mode_defaults_applied", {}) or {}),
        "explicit_policy_overrides": list(
            getattr(args, "explicit_policy_overrides", []) or []
        ),
    }
    report["run_identity"] = artifact_episode_alignment
    report["final_validation"] = final_validation
    report["rejected_update_memory"] = normalize_rejected_update_memory(
        rejected_update_memory
    )
    report["latest_artifact_promotion"] = {
        "promoted": latest_promoted,
        "blocked_by_strict_validation": strict_validation_failed,
        "latest_alias_written": latest_alias_written,
        "latest_alias_policy": (
            "strict_validation_promoted"
            if latest_promoted
            else "strict_validation_failed_no_latest_alias"
        ),
        "rule_version_path": str(final_rule_path),
        "plan_version_path": str(final_plan_path or ""),
        "rule_alias_version_path": str(latest_alias_rule_path or ""),
        "plan_alias_version_path": str(latest_alias_plan_path or ""),
    }
    report["dependency_validation"] = final_dependency_validation
    report["llm_transcripts"] = {
        "enabled": bool(args.save_llm_transcripts),
        "dir": args.llm_transcripts_dir or "",
        "stages": {
            "judge": bool(args.save_llm_transcripts),
            "reviewer": bool(args.save_llm_transcripts),
            "rule_updater": bool(args.save_llm_transcripts),
            "plan_updater": bool(args.save_llm_transcripts),
        },
    }
    report["reviewer_policy"] = {
        "review_compact_mode": args.review_compact_mode,
        "max_review_prompt_chars": int(args.max_review_prompt_chars or 0),
        "review_thinking": args.review_thinking,
        "update_thinking": args.update_thinking,
        "label_rationale_training_supervision": bool(args.review_label_rationale),
        "label_rationale_visible_to_judge": False,
        "label_rationale_visible_to_cold_start": False,
        "error_focused_compact_preserves": [
            "wrong_condition_rows",
            "condition_feature_analysis",
            "stateful_binding_state",
            "iv_stateful_state",
            "rule_context",
            "plan_context",
            "evidence_limitations",
        ],
        "rejected_update_memory": {
            "enabled": True,
            "scope": "episode_rounds_compact",
            "max_entries": 20,
            "contains_transaction_identifiers": False,
            "purpose": (
                "Provide prior failure reason and protected attribution as "
                "refinement feedback; exact duplicate candidates are removed "
                "deterministically downstream."
            ),
        },
        "unchanged_input_reuse": {
            "enabled": True,
            "fingerprint_excludes_rejected_memory": False,
            "rejected_memory_policy": "feedback_only",
        },
    }
    report["stateful_runtime_policy"] = {
        "iv_stateful_runtime_enabled_by_cli": bool(args.iv_stateful_runtime),
        "access_control_binding_mode": args.access_control_binding_mode,
        "reentrancy_binding_mode": args.reentrancy_binding_mode,
        "disable_stateful_bindings": bool(args.disable_stateful_bindings),
        "label_gated": False,
        "supported_labels": [
            "insufficient_validation",
            "access_control",
            "token_semantic_exploitation",
            "market_manipulation",
            "protocol_accounting_exploitation",
            "flashloans",
            "reentrancy",
        ],
        "modes": {
            "insufficient_validation": "insufficient_validation_object_binding_v1",
            "access_control": (
                "access_control_authorization_binding_v1"
                if args.access_control_binding_mode == "stateful"
                else "prompt_only_authorization_chain"
            ),
            "token_semantic_exploitation": "token_semantic_candidate_binding_v1",
            "market_manipulation": "market_mechanism_profile_binding_v1",
            "protocol_accounting_exploitation": "protocol_accounting_candidate_binding_v1",
            "flashloans": "flash_capital_chain_binding_v1",
            "reentrancy": (
                f"reentrancy_candidate_binding_v1:{args.reentrancy_binding_mode}"
            ),
        },
        "does_not_affect_other_labels": False,
        "plan_declared_stateful_runtime_supported": True,
    }
    report.setdefault("inputs", {}).setdefault("runtime", {})["early_stop"] = {
        "enabled": bool(args.enable_runtime_early_stop),
        "policy": "conservative_negative",
        "default_enabled": False,
    }
    report.setdefault("inputs", {}).setdefault("runtime", {})[
        "followup_context"
    ] = {
        "mode": args.followup_context_mode,
        "budget_chars": int(args.max_context_chars or 0),
        "judge_max_tokens": int(args.judge_max_tokens or 8192),
        "aggregator_max_tokens": int(args.aggregator_max_tokens or 8192),
        "judge_thinking": args.judge_thinking,
        "aggregator_thinking": args.aggregator_thinking,
    }
    report.setdefault("inputs", {}).setdefault("runtime", {})[
        "judge_followup_policy"
    ] = {
        "mode": str(getattr(args, "judge_followup_mode", "plan") or "plan"),
        "expanded_view_count": int(
            getattr(args, "judge_expanded_followup_views", 2) or 0
        ),
        "single_call": str(
            getattr(args, "judge_followup_mode", "plan") or "plan"
        ) != "plan",
        "disables_adaptive_evidence": str(
            getattr(args, "judge_followup_mode", "plan") or "plan"
        ) != "plan",
        "disables_near_miss": str(
            getattr(args, "judge_followup_mode", "plan") or "plan"
        ) != "plan",
        "disables_dynamic_aggregation": str(
            getattr(args, "judge_followup_mode", "plan") or "plan"
        ) != "plan",
    }
    report.setdefault("inputs", {}).setdefault("runtime", {})[
        "rate_limit_fallback"
    ] = {
        "enabled": bool(args.rate_limit_serial_fallback),
        "fallback_concurrency": 1,
        "retry_attempts": int(args.rate_limit_retry_attempts or 0),
        "initial_delay_seconds": float(
            args.rate_limit_retry_delay_seconds or 0.0
        ),
        "backoff": "exponential",
    }
    report["hard_negative_guard"] = {
        "enabled": bool(hard_guard_cases),
        "csv": args.hard_neg_csv or "",
        "case_counts": guard_counts,
        "used_for_review": False,
        "allow_guard_repair": bool(args.allow_guard_repair),
        "acceptance_thresholds": {
            "max_guard_fp_increase": args.max_guard_fp_increase,
            "max_guard_error_increase": args.max_guard_error_increase,
            "max_guard_uncertain_increase": args.max_guard_uncertain_increase,
        },
        "policy": (
            "Guard cases are fixed hard negatives for candidate acceptance only. "
            "They are not included in reviewer/updater inputs by default."
        ),
    }
    report["candidate_policy"] = {
        "candidate_types": [
            "rule_candidate",
            "plan_candidate",
            "rule_plan_candidate",
            "experimental_probe",
        ],
        "combined_candidate_enabled": bool(
            args.enable_phase3_candidate_portfolio
        ),
        "phase3_candidate_portfolio": {
            "enabled": bool(args.enable_phase3_candidate_portfolio),
            "max_candidates": int(args.max_phase3_candidates or 0),
            "joint_requires_shared_condition": True,
            "cartesian_product_disabled": True,
        },
        "experimental_rule_hypotheses": {
            "enabled": bool(args.enable_experimental_rule_hypotheses),
            "evaluation_scope": "canary_probe_then_full_confirmation",
            "direct_promotion_allowed": False,
        },
        "bounded_plan_restructuring": {
            "route_pruning_enabled": bool(args.enable_plan_route_pruning),
            "dependency_restructure_enabled": bool(
                args.enable_plan_dependency_restructure
            ),
            "route_replacement_requires_history": True,
        },
        "rule_candidate_plan_strategy": "regenerate_plan_from_updated_rule",
        "candidate_judge_reuse_default": True,
        "candidate_judge_reuse_enabled": bool(args.enable_candidate_judge_reuse),
        "candidate_canary": {
            "enabled": bool(args.enable_candidate_canary),
            "error_case_limit": int(args.candidate_canary_error_cases or 0),
            "protected_positive_limit": int(
                args.candidate_canary_positive_cases or 0
            ),
            "protected_boundary_limit": int(
                args.candidate_canary_boundary_cases or 0
            ),
            "protected_guard_limit": int(args.candidate_canary_guard_cases or 0),
            "policy": "repair targets plus protected train/guard boundaries",
        },
        "plan_candidate_repair_enabled": True,
        "evolution_mode": args.evolution_mode,
        "final_validation_mode": args.final_validation_mode,
        "final_validation_does_not_rollback": False,
        "strict_validation_failure_blocks_latest": True,
        "strict_validation_failure_blocks_validated_promotion": True,
        "eval_entrypoint_latest_alias_always_written": False,
    }
    report["rule_budget_policy"] = {
        "budget_mode": "strict" if args.strict_source_budget else "label_aware",
        "label_aware_source_budget": bool(args.label_aware_source_budget)
        and not bool(args.strict_source_budget),
        "strict_source_budget": bool(args.strict_source_budget),
        "allow_source_budget_warning": bool(args.allow_source_budget_warning)
        and not bool(args.strict_source_budget),
        "initial_budget_trace": rule_budget_trace(initial_rule, args=args),
        "final_budget_trace": rule_budget_trace(current_rule, args=args),
        "candidate_budget_stats": budget_candidate_stats(last_candidate_evaluations),
    }
    report["packet_view_policy"] = {
        "view_manifest_version": VIEW_MANIFEST_VERSION,
        "view_tier_and_cost_explicit": True,
        "dependency_closure_recorded_in_build_config": True,
        "adaptive_evidence_routing_enabled": bool(args.adaptive_evidence),
        "adaptive_evidence_mode": args.adaptive_evidence_mode,
        "adaptive_evidence_thresholds": {
            "max_direct_trace_nodes": args.adaptive_evidence_max_direct_trace_nodes,
            "max_direct_trace_chars": args.adaptive_evidence_max_direct_trace_chars,
            "max_medium_trace_nodes": args.adaptive_evidence_max_medium_trace_nodes,
        },
    }
    report["plan_template_policy"] = {
        "initial_plan_mode": (
            "plan_file"
            if args.plan_file
            else "agent"
            if args.generate_initial_plan_with_agent
            else "baseline"
        ),
        "cold_start_or_baseline_enforce_label_plan_template": True,
        "evolution_regenerated_plan_enforce_label_plan_template": False,
        "label": resolved_positive_label,
        "note": (
            "Cold-start/baseline initialization may use label templates for stable "
            "defaults. Evolution plan regeneration disables forced templates so "
            "plan agent/updater changes are preserved."
        ),
    }
    last_candidate_runtime = _aggregate_candidate_runtime(last_candidate_evaluations)
    report["evolution_speed"] = {
        "candidate_judge_reuse": {
            "enabled": bool(args.enable_candidate_judge_reuse),
            **last_candidate_runtime.get("candidate_judge_reuse", {}),
        },
        "runtime_early_stop": {
            "enabled": bool(args.enable_runtime_early_stop),
            **last_candidate_runtime.get("runtime_early_stop", {}),
        },
        "adaptive_evidence": {
            "enabled": bool(args.adaptive_evidence),
            "mode": args.adaptive_evidence_mode,
            **last_candidate_runtime.get("adaptive_evidence", {}),
        },
        "parallel_judge": {
            "enabled": bool(args.parallel_judge),
            "judge_concurrency": int(args.judge_concurrency or 1),
            **last_candidate_runtime.get("parallel_judge", {}),
        },
        "complexity": {
            "initial_rule_complexity": rule_complexity(initial_rule),
            "final_rule_complexity": rule_complexity(current_rule),
            "initial_budget_trace": rule_budget_trace(initial_rule, args=args),
            "final_budget_trace": rule_budget_trace(current_rule, args=args),
            "candidate_budget_stats": budget_candidate_stats(last_candidate_evaluations),
            "final_plan_complexity": plan_complexity(final_plan),
            "last_candidate_rule_complexity": [
                rule_complexity(item["rule"]) for item in last_candidate_evaluations
            ],
            "last_candidate_plan_complexity": [
                plan_complexity(item.get("plan")) for item in last_candidate_evaluations
            ],
        },
        "rule_repair": {
            "plan_regeneration_default": "llm_regenerated_for_rule_repair_then_baseline_fallback",
        },
    }
    report["evolution_safety_policy"] = {
        "evolution_mode": args.evolution_mode,
        "final_validation_mode": args.final_validation_mode,
        "final_validation_does_not_rollback": False,
        "strict_validation_failure_blocks_latest": True,
        "strict_validation_failure_blocks_validated_promotion": True,
        "eval_entrypoint_latest_alias_always_written": False,
        "uncertain_regression_guard": True,
        "semantic_plan_change_only": True,
        "plan_minimal_scope_guard": True,
        "reviewer_updater_taxonomy_aligned": True,
        "compress_rule_preserve_condition_ids": True,
        "guard_repair_default": False,
    }
    report_path = artifacts_dir / "final_report.json"
    write_json(report_path, report)

    print("Stage 2 complete.")
    print(f"Positive label: {resolved_positive_label}")
    print(
        "Cases: "
        f"attack={counts['attack']}, negative={counts['negative']}, "
        f"negative_benign={counts['negative_benign']}, "
        f"negative_other={counts['negative_other']}, "
        f"other_attack={counts['negative_other_attack']}, total={counts['total']}"
    )
    print(f"Final rule: {current_rule.rule_id} v{current_rule.version}")
    if final_plan is not None:
        print(
            f"Final plan: {final_plan.plan_id} "
            f"plan_v={(final_plan.metadata or {}).get('plan_version')} "
            f"path={final_plan_path}"
        )
    print(f"Accepted rounds: {accepted_rounds}")
    print(f"Stop reason: {stop_reason}")
    print(f"Final summary: {stable_json_dumps(final_summary)}")
    print(f"Started at: {run_timing['started_at']}")
    print(f"Finished at: {run_timing['finished_at']}")
    print(f"Total elapsed: {run_timing['elapsed_hms']} ({run_timing['elapsed_seconds']}s)")
    print(f"Artifacts: {artifacts_dir}")
    if args.save_llm_transcripts:
        print(f"LLM transcripts: {args.llm_transcripts_dir}")
    print(f"Final versioned rule: {final_rule_path}")
    print(f"Session log: {log_path}")


def apply_episode_paths(args) -> None:
    if args.episode is not None:
        if args.episode < 0:
            raise ValueError("--episode must be a non-negative integer.")
        episode = str(args.episode)
        args.rules_dir = str(_append_episode_path(args.rules_dir, episode))
        args.plans_dir = str(_append_episode_path(args.plans_dir, episode))
        args.artifacts_dir = str(_append_episode_path(args.artifacts_dir, episode))
        args.logs_dir = str(_append_episode_path(args.logs_dir, episode))


def build_episode_artifact_alignment(
    requested_episode: int | None,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """Make report provenance explicit before any expensive runtime work."""
    directory_name = artifacts_dir.name.strip()
    directory_episode = int(directory_name) if directory_name.isdigit() else None
    aligned = (
        requested_episode is None
        or directory_episode == int(requested_episode)
    )
    return {
        "schema_version": "evotx.run_identity.v1",
        "requested_episode": requested_episode,
        "artifact_directory": str(artifacts_dir),
        "artifact_directory_episode": directory_episode,
        "aligned": aligned,
    }


def apply_evolution_mode_defaults(args, raw_argv: List[str]) -> Dict[str, Any]:
    mode = str(getattr(args, "evolution_mode", "explore") or "explore")
    applied: Dict[str, Any] = {}
    explicit: List[str] = []

    def set_default(attr: str, value: Any, *flags: str) -> None:
        if _flag_provided(raw_argv, *flags):
            explicit.append(attr)
            return
        if not hasattr(args, attr):
            return
        setattr(args, attr, value)
        applied[attr] = value

    if mode == "strict":
        set_default(
            "continue_after_rejected_candidate",
            False,
            "--continue-after-rejected-candidate",
            "--stop-after-rejected-candidate",
        )
        set_default("rejected_candidate_retry_rounds", 0, "--rejected-candidate-retry-rounds")
        set_default("max_uncertain_increase", 0, "--max-uncertain-increase")
        set_default("max_uncertain_benign_increase", 0, "--max-uncertain-benign-increase")
        set_default("max_guard_fp_increase", 1, "--max-guard-fp-increase")
        set_default("max_guard_error_increase", 1, "--max-guard-error-increase")
        set_default("max_guard_uncertain_increase", 1, "--max-guard-uncertain-increase")
        set_default("temporary_fp_increase", 0, "--temporary-fp-increase")
        set_default("max_repair_rounds", 0, "--max-repair-rounds")
        set_default(
            "enable_candidate_repair",
            False,
            "--enable-candidate-repair",
            "--disable-candidate-repair",
        )
        set_default(
            "enable_recall_repair",
            False,
            "--enable-recall-repair",
            "--disable-recall-repair",
        )
    else:
        set_default(
            "continue_after_rejected_candidate",
            True,
            "--continue-after-rejected-candidate",
            "--stop-after-rejected-candidate",
        )
        set_default("rejected_candidate_retry_rounds", 2, "--rejected-candidate-retry-rounds")
        set_default("max_uncertain_increase", 2, "--max-uncertain-increase")
        set_default("max_uncertain_benign_increase", 1, "--max-uncertain-benign-increase")
        # Intermediate acceptance and final promotion share one guard boundary.
        set_default("max_guard_fp_increase", 1, "--max-guard-fp-increase")
        set_default("max_guard_error_increase", 1, "--max-guard-error-increase")
        set_default("max_guard_uncertain_increase", 1, "--max-guard-uncertain-increase")
        set_default("temporary_fp_increase", 3, "--temporary-fp-increase")
        set_default("repair_min_fn_decrease", 1, "--repair-min-fn-decrease")
        set_default("max_repair_rounds", 0, "--max-repair-rounds")
        set_default(
            "enable_candidate_repair",
            False,
            "--enable-candidate-repair",
            "--disable-candidate-repair",
        )
        set_default(
            "enable_recall_repair",
            False,
            "--enable-recall-repair",
            "--disable-recall-repair",
        )
        set_default("recall_repair_min_fixed_fp", 1, "--recall-repair-min-fixed-fp")
        set_default("recall_repair_max_new_fn", 10, "--recall-repair-max-new-fn")

    args.mode_defaults_applied = dict(applied)
    args.explicit_policy_overrides = sorted(set(explicit))
    return {
        "evolution_mode": mode,
        "mode_defaults_applied": dict(applied),
        "explicit_policy_overrides": sorted(set(explicit)),
    }


def _flag_provided(raw_argv: List[str], *flags: str) -> bool:
    flag_set = {str(flag) for flag in flags if str(flag)}
    for arg in raw_argv or []:
        text = str(arg)
        for flag in flag_set:
            if text == flag or text.startswith(flag + "="):
                return True
    return False


def _should_continue_after_rejected_candidate(
    *,
    args,
    round_index: int,
    consecutive_rejected_candidate_rounds: int,
) -> bool:
    if not bool(getattr(args, "continue_after_rejected_candidate", False)):
        return False
    if int(round_index) >= int(getattr(args, "max_rounds", 0) or 0):
        return False
    max_retries = int(getattr(args, "rejected_candidate_retry_rounds", 0) or 0)
    if max_retries <= 0:
        return False
    return int(consecutive_rejected_candidate_rounds) <= max_retries


def strict_policy_args(args):
    copied = copy.copy(args)
    copied.max_fp_increase = 0
    # Final strict promotion treats uncertainty changes as diagnostic. FP/FN,
    # total-error, regression, and hard-negative guard checks remain enforced.
    copied.max_uncertain_increase = 1
    copied.max_uncertain_benign_increase = 1
    copied.enforce_uncertain_gate = False
    copied.max_guard_fp_increase = 1
    copied.max_guard_error_increase = 1
    copied.max_guard_uncertain_increase = 1
    copied.temporary_fp_increase = 0
    copied.max_repair_rounds = 0
    copied.enable_candidate_repair = False
    if hasattr(copied, "enable_recall_repair"):
        copied.enable_recall_repair = False
    return copied


def normalize_candidate_repair_args(args) -> None:
    policy = CandidateRepairPolicy.from_args(args)
    args.enable_candidate_repair = policy.enabled
    args.max_fp_increase = policy.final_max_fp_increase
    args.max_fn_increase = max(0, int(args.max_fn_increase or 0))
    args.max_uncertain_increase = max(0, int(getattr(args, "max_uncertain_increase", 0) or 0))
    args.max_uncertain_benign_increase = max(
        0, int(getattr(args, "max_uncertain_benign_increase", 0) or 0)
    )
    args.temporary_fp_increase = policy.temporary_fp_increase
    args.repair_min_fn_decrease = policy.repair_min_fn_decrease
    args.max_repair_rounds = policy.max_repair_rounds
    args.recall_repair_min_fixed_fp = max(
        0,
        int(getattr(args, "recall_repair_min_fixed_fp", 1) or 0),
    )
    args.recall_repair_max_new_fn = max(
        0,
        int(getattr(args, "recall_repair_max_new_fn", 10) or 0),
    )


def _append_episode_path(path_like: str, episode: str) -> Path:
    path = Path(path_like)
    if path.name == episode:
        return path
    return path / episode


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_elapsed_hms(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000))
    whole_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def load_or_initialize_rule(args, tool_manifest: Dict[str, Any]) -> EvolvingRule:
    if args.rule_file:
        rule = EvolvingRule.from_dict(read_json(args.rule_file))
        raw_label = args.label or (rule.metadata or {}).get("attack_label", "")
        if raw_label:
            rule.metadata.setdefault("raw_attack_label", raw_label)
            rule.metadata.setdefault("display_label", raw_label)
            rule.metadata["attack_label"] = normalize_attack_label(raw_label)
        return rule

    attack_description = resolve_attack_description(
        attack_description=args.attack_description,
        label=args.label or args.positive_label,
    )

    human_example = load_human_example(args.human_example, args.human_example_file)
    cold_start_model, cold_start_provider = resolve_role_llm_route(
        args,
        model=args.cold_start_model or args.llm_model,
        thinking=args.cold_start_thinking,
    )
    cold_start_llm = make_llm(
        cold_start_model,
        cold_start_provider,
        max_tokens=args.cold_start_max_tokens,
        minimax_thinking=args.cold_start_thinking,
    )
    return generate_cold_start_rule(
        attack_description=attack_description,
        llm=cold_start_llm,
        human_example=human_example,
        attack_label=normalize_attack_label(args.label or args.positive_label),
    )


def build_initial_artifact_provenance(args) -> Dict[str, Any]:
    """Record CLI inputs separately from episode-local artifact archives."""
    rule_file = str(getattr(args, "rule_file", "") or "").strip()
    plan_file = str(getattr(args, "plan_file", "") or "").strip()
    plan_mode = (
        "plan_file"
        if plan_file
        else "agent_generated"
        if bool(getattr(args, "generate_initial_plan_with_agent", False))
        else "baseline_compiled_from_initial_rule"
    )
    return {
        "schema_version": "evotx.initial_artifact_provenance.v1",
        "episode": getattr(args, "episode", None),
        "rule": {
            "cli_path": rule_file,
            "resolved_path": _resolved_cli_artifact_path(rule_file),
            "initialization_mode": (
                "rule_file" if rule_file else "cold_start_agent"
            ),
            "archive_path": "",
        },
        "plan": {
            "cli_path": plan_file,
            "resolved_path": _resolved_cli_artifact_path(plan_file),
            "initialization_mode": plan_mode,
            "archive_path": "",
            "artifact_metadata_source": "",
        },
    }


def _resolved_cli_artifact_path(path_like: str) -> str:
    text = str(path_like or "").strip()
    return str(Path(text).resolve()) if text else ""


def _quoted_cli_artifact_path(path_like: str) -> str:
    text = str(path_like or "").strip()
    return f'"{text}"' if text else '""'


def apply_rule_budget_config(rule: EvolvingRule, args) -> EvolvingRule:
    metadata = dict(rule.metadata or {})
    if metadata.get("attack_label"):
        raw_label = metadata.get("raw_attack_label") or metadata.get("attack_label")
        metadata.setdefault("raw_attack_label", raw_label)
        metadata.setdefault("display_label", raw_label)
        metadata["attack_label"] = normalize_attack_label(raw_label)
    strict = bool(getattr(args, "strict_source_budget", False))
    label_aware = bool(getattr(args, "label_aware_source_budget", True)) and not strict
    allow_warning = bool(getattr(args, "allow_source_budget_warning", True)) and not strict
    metadata["strict_source_budget"] = strict
    metadata["label_aware_source_budget"] = label_aware
    metadata["allow_source_budget_warning"] = allow_warning
    metadata["rule_budget_mode"] = "strict" if strict else "label_aware" if label_aware else "default"
    metadata["rule_complexity"] = rule_complexity(rule)
    rule.metadata = metadata
    rule.metadata["rule_complexity"] = rule_complexity(rule)
    return rule


def rule_budget_trace(rule: EvolvingRule, args=None) -> Dict[str, Any]:
    complexity = rule_complexity(rule)
    metadata = dict(rule.metadata or {})
    budget_mode = str(metadata.get("rule_budget_mode") or complexity.get("budget_mode") or "")
    if args is not None and getattr(args, "strict_source_budget", False):
        budget_mode = "strict"
    return {
        "budget_mode": budget_mode or "label_aware",
        "label": str(metadata.get("attack_label") or ""),
        "hard_over_budget": bool(complexity.get("hard_over_budget")),
        "soft_over_budget": bool(complexity.get("soft_over_budget")),
        "hard_over_budget_reasons": list(complexity.get("hard_over_budget_reasons", [])),
        "soft_over_budget_reasons": list(complexity.get("soft_over_budget_reasons", [])),
        "budget_status": complexity.get("budget_status", "ok"),
        "label_budget_profile": complexity.get("label_budget_profile", "default"),
        "source_dependent_overage_policy": complexity.get(
            "source_dependent_overage_policy", "hard"
        ),
        "candidate_allowed_despite_soft_warning": (
            bool(complexity.get("soft_over_budget"))
            and not bool(complexity.get("hard_over_budget"))
        ),
    }


def load_initial_plan(plan_file: str) -> EvidencePlan:
    return PlanStore(Path(plan_file).parent).load(plan_file)


def build_initial_baseline_plan(rule: EvolvingRule) -> EvidencePlan:
    plan = compile_rule_to_baseline_plan(rule)
    metadata = dict(plan.metadata or {})
    metadata.setdefault("source", "baseline_compiled_from_initial_rule")
    metadata.setdefault("initial_plan_mode", "baseline")
    metadata.setdefault("plan_version", 1)
    metadata.setdefault("reusable_artifact", True)
    metadata.setdefault(
        "note",
        "Baseline plan saved so rule-only few-shot runs always have a reusable plan artifact.",
    )
    plan.metadata = metadata
    return plan


def build_initial_agent_plan(
    rule: EvolvingRule,
    planner_llm,
) -> EvidencePlan:
    if planner_llm is None:
        raise ValueError(
            "--generate-initial-plan-with-agent requires --planner-model or "
            "--llm-model."
        )
    attack_label = str((rule.metadata or {}).get("attack_label") or "")
    plan = PlanGenerator(
        llm=planner_llm,
        allow_fallback=False,
        enforce_label_plan_template=True,
    ).generate(
        rule,
        {
            "stage": "cold_start_initial_plan",
            "reusable_across_transactions": True,
            "transaction_evidence_available": False,
            "instruction": (
                "Generate a reusable attack-family plan from rule semantics; "
                "do not assume facts from any individual transaction."
            ),
        },
    )
    PlanValidator().validate_against_rule(
        plan,
        rule,
        attack_label=attack_label,
    )
    metadata = dict(plan.metadata or {})
    metadata.update({
        "source": "initial_plan_agent",
        "initial_plan_mode": "agent",
        "plan_version": 1,
        "reusable_artifact": True,
    })
    plan.metadata = metadata
    return plan


def normalize_plan_for_runtime_policy(
    plan: EvidencePlan,
    *,
    args=None,
    disable_stateful_bindings: bool | None = None,
    reason: str,
) -> EvidencePlan:
    """Apply artifact-level ablations before diffing, saving, or execution."""
    disabled = (
        bool(disable_stateful_bindings)
        if disable_stateful_bindings is not None
        else bool(getattr(args, "disable_stateful_bindings", False))
    )
    if not disabled:
        return plan
    normalized = disable_stateful_bindings_in_plan(plan)
    metadata = dict(normalized.metadata or {})
    ablation = dict(metadata.get("stateful_binding_ablation") or {})
    ablation["normalization_stage"] = str(reason or "candidate")
    ablation["artifact_policy"] = (
        "persist the same binding-disabled plan used by PacketRuntime"
    )
    metadata["stateful_binding_ablation"] = ablation
    normalized.metadata = metadata
    return normalized


def regenerate_candidate_plan(
    rule: EvolvingRule,
    *,
    args,
    planner_guidance: Dict[str, Any] | None,
    current_rule: EvolvingRule | None = None,
    current_plan: EvidencePlan | None = None,
    plan_patch_condition_ids: List[str] | None = None,
) -> tuple[EvidencePlan, bool]:
    """Generate one stable plan artifact for a rule candidate."""
    if rule.metadata:
        raw_label = rule.metadata.get("raw_attack_label") or rule.metadata.get("attack_label", "")
        rule.metadata.setdefault("raw_attack_label", raw_label)
        rule.metadata.setdefault("display_label", raw_label)
        rule.metadata["attack_label"] = normalize_attack_label(raw_label)
    current_signatures = (
        _rule_condition_signature_by_id(current_rule)
        if current_rule is not None
        else {}
    )
    candidate_signatures = _rule_condition_signature_by_id(rule)
    changed_condition_ids = sorted(
        condition_id
        for condition_id, signature in candidate_signatures.items()
        if current_signatures.get(condition_id) != signature
    )
    explicit_plan_patch_ids = sorted({
        str(item or "").strip().upper()
        for item in list(plan_patch_condition_ids or [])
        if str(item or "").strip()
    })
    guidance_context = {
        "planner_guidance": dict(planner_guidance or {}),
        "candidate_context": {
            "candidate_type": "rule_candidate",
            "reason": "rule was updated; regenerate evidence plan from new rule",
            "do_not_copy_review_text_into_rule": True,
            "changed_rule_condition_ids": changed_condition_ids,
            "explicit_plan_patch_condition_ids": explicit_plan_patch_ids,
            "locality_instruction": (
                "Redesign only changed_rule_condition_ids and explicit_plan_patch_condition_ids. "
                "All other judge steps are frozen and will be restored from the current plan."
            ),
        },
    }
    try:
        planner_model, planner_provider = resolve_role_llm_route(
            args,
            model=args.planner_model or args.llm_model,
            thinking=args.planner_thinking,
        )
        planner_llm = make_llm(
            planner_model,
            planner_provider,
            max_tokens=args.planner_max_tokens,
            minimax_thinking=args.planner_thinking,
        )
        if planner_llm is None:
            raise RuntimeError("planner LLM is not configured")
        plan = PlanGenerator(
            llm=planner_llm,
            enforce_label_plan_template=False,
        ).generate(rule, guidance_context)
        plan = preserve_unchanged_candidate_plan_steps(
            generated_plan=plan,
            current_plan=current_plan,
            current_rule=current_rule,
            candidate_rule=rule,
            protected_condition_ids=plan_patch_condition_ids,
        )
        view_budget_violations = candidate_plan_view_budget_violations(
            candidate_plan=plan,
            current_plan=current_plan,
            target_condition_ids=(
                explicit_plan_patch_ids or changed_condition_ids
            ),
            attack_label=str((rule.metadata or {}).get("attack_label") or ""),
        )
        if view_budget_violations:
            raise PlanValidationError(
                "Regenerated candidate plan has changed steps whose View routes "
                "would be silently truncated: "
                f"{view_budget_violations}"
            )
        _canonicalize_candidate_plan_followups(
            plan,
            attack_label=str((rule.metadata or {}).get("attack_label") or ""),
            reason="llm_regenerated_for_rule_candidate",
        )
        plan = normalize_plan_for_runtime_policy(
            plan,
            args=args,
            reason="rule_candidate_regeneration",
        )
        PlanValidator().validate_against_rule(
            plan,
            rule,
            attack_label=str((rule.metadata or {}).get("attack_label") or ""),
        )
        metadata = dict(plan.metadata or {})
        metadata.update({
            "source": "llm_regenerated_for_rule_candidate",
            "candidate_strategy": "updated_rule_regenerated_plan",
            "derived_from_rule_id": rule.rule_id,
            "derived_from_rule_version": rule.version,
            "plan_generation_fallback": False,
            "plan_template_policy": {
                "enforce_label_plan_template": False,
                "stage": "evolution",
                "label": str((rule.metadata or {}).get("attack_label", "")),
            },
        })
        plan.metadata = metadata
        return plan, False
    except Exception as exc:
        plan = compile_rule_to_baseline_plan(rule)
        plan = preserve_unchanged_candidate_plan_steps(
            generated_plan=plan,
            current_plan=current_plan,
            current_rule=current_rule,
            candidate_rule=rule,
            protected_condition_ids=plan_patch_condition_ids,
        )
        plan = normalize_plan_for_runtime_policy(
            plan,
            args=args,
            reason="rule_candidate_fallback",
        )
        metadata = dict(plan.metadata or {})
        metadata.update({
            "source": "baseline_after_rule_candidate_plan_generation_failure",
            "candidate_strategy": "updated_rule_regenerated_plan",
            "derived_from_rule_id": rule.rule_id,
            "derived_from_rule_version": rule.version,
            "plan_generation_fallback": True,
            "fallback_reason": repr(exc),
            "plan_template_policy": {
                "enforce_label_plan_template": True,
                "stage": "baseline_fallback",
                "label": str((rule.metadata or {}).get("attack_label", "")),
            },
        })
        plan.metadata = metadata
        PlanValidator().validate_against_rule(
            plan,
            rule,
            attack_label=str((rule.metadata or {}).get("attack_label") or ""),
        )
        return plan, True


def candidate_plan_view_budget_violations(
    *,
    candidate_plan: EvidencePlan,
    current_plan: EvidencePlan | None,
    target_condition_ids: List[str] | None,
    attack_label: str,
) -> List[Dict[str, Any]]:
    """Reject changed candidate routes that exceed their executable budget.

    Unchanged legacy steps are grandfathered because this check protects the
    update boundary, not artifact loading. A changed step must make its active
    View choice explicit instead of relying on Runtime's positional slicing.
    """
    targets = {
        str(value or "").strip().upper()
        for value in list(target_condition_ids or [])
        if str(value or "").strip()
    }
    current_by_id = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list((current_plan.judge_steps if current_plan else []) or [])
    }
    violations: List[Dict[str, Any]] = []
    for step in list(candidate_plan.judge_steps or []):
        condition_id = str(step.condition_id or step.id or "").strip().upper()
        if targets and condition_id not in targets:
            continue
        previous = current_by_id.get(condition_id)
        candidate_routes = {
            "default_evidence_refs": list(
                step.default_evidence_refs or step.evidence_refs or []
            ),
            "allowed_followup_views": list(step.allowed_followup_views or []),
        }
        previous_routes = (
            {
                "default_evidence_refs": list(
                    previous.default_evidence_refs or previous.evidence_refs or []
                ),
                "allowed_followup_views": list(
                    previous.allowed_followup_views or []
                ),
            }
            if previous is not None
            else {}
        )
        if previous is not None and candidate_routes == previous_routes:
            continue
        audit = judge_step_view_budget_audit(
            step,
            attack_label=attack_label,
        )
        if audit.get("over_budget"):
            violations.append({
                "condition_id": condition_id,
                "budget": audit.get("budget", {}),
                "inactive_views": audit.get("inactive_views", []),
                "requested_default_views": audit.get(
                    "requested_default_views", []
                ),
                "requested_followup_views": audit.get(
                    "requested_followup_views", []
                ),
            })
    return violations


def preserve_unchanged_candidate_plan_steps(
    *,
    generated_plan: EvidencePlan,
    current_plan: EvidencePlan | None,
    current_rule: EvolvingRule | None,
    candidate_rule: EvolvingRule | None,
    protected_condition_ids: List[str] | None = None,
) -> EvidencePlan:
    """Keep old judge steps for rule conditions whose text did not change.

    Rule candidates often regenerate a full plan, but judge-step reuse only
    works when unchanged local questions and selected views remain stable.
    This merge is deliberately conservative: if a condition was rewritten or a
    plan patch targets it, the regenerated step wins.
    """
    if current_plan is None or current_rule is None or candidate_rule is None:
        return generated_plan

    current_text = _rule_condition_signature_by_id(current_rule)
    candidate_text = _rule_condition_signature_by_id(candidate_rule)
    protected = {str(item or "").strip().upper() for item in list(protected_condition_ids or [])}
    directly_changed_ids = {
        condition_id
        for condition_id, signature in candidate_text.items()
        if current_text.get(condition_id) != signature
    } | (set(current_text) - set(candidate_text)) | protected
    dependency_affected_ids = (
        _stateful_downstream_condition_ids(
            current_plan,
            directly_changed_ids,
        )
        | _stateful_downstream_condition_ids(
            generated_plan,
            directly_changed_ids,
        )
    ) - directly_changed_ids
    preserve_ids = {
        condition_id
        for condition_id, text in current_text.items()
        if condition_id in candidate_text
        and candidate_text[condition_id] == text
        and condition_id not in directly_changed_ids
        and condition_id not in dependency_affected_ids
    }
    current_steps = {
        str(step.condition_id or step.id or "").strip().upper(): step
        for step in list(current_plan.judge_steps or [])
        if str(step.condition_id or step.id or "").strip()
    }
    if not current_steps:
        return generated_plan

    data = generated_plan.to_dict()
    generated_steps = list(data.get("judge_steps", []) or [])
    generated_by_condition = {
        str(step.get("condition_id") or step.get("id") or "").strip().upper(): dict(step)
        for step in generated_steps
        if isinstance(step, dict)
        and str(step.get("condition_id") or step.get("id") or "").strip()
    }
    replaced: List[str] = []
    merged_steps: List[Dict[str, Any]] = []
    consumed: set[str] = set()
    for current_step in list(current_plan.judge_steps or []):
        condition_id = str(
            current_step.condition_id or current_step.id or ""
        ).strip().upper()
        generated_step = generated_by_condition.get(condition_id)
        if condition_id in preserve_ids:
            merged_steps.append(copy.deepcopy(current_step.to_dict()))
            replaced.append(condition_id)
            consumed.add(condition_id)
        elif generated_step is not None:
            regenerated_step = copy.deepcopy(generated_step)
            regenerated_step["id"] = current_step.id
            regenerated_step["condition_id"] = (
                current_step.condition_id or condition_id
            )
            merged_steps.append(regenerated_step)
            consumed.add(condition_id)
    for raw_step in generated_steps:
        step = dict(raw_step or {})
        condition_id = str(
            step.get("condition_id") or step.get("id") or ""
        ).strip().upper()
        if condition_id and condition_id not in consumed:
            merged_steps.append(step)
            consumed.add(condition_id)

    data["judge_steps"] = merged_steps
    metadata = dict(data.get("metadata") or {})
    metadata["preserved_unchanged_plan_steps"] = sorted(set(replaced))
    metadata["preserved_unchanged_plan_step_count"] = len(set(replaced))
    metadata["changed_rule_condition_ids"] = sorted(
        directly_changed_ids - protected
    )
    metadata["removed_rule_condition_ids"] = sorted(set(current_text) - set(candidate_text))
    metadata["explicit_plan_patch_condition_ids"] = sorted(protected)
    metadata["directly_changed_plan_step_condition_ids"] = sorted(
        directly_changed_ids
    )
    metadata["dependency_affected_plan_step_condition_ids"] = sorted(
        dependency_affected_ids
    )
    metadata["regenerated_plan_step_condition_ids"] = sorted(
        condition_id
        for condition_id in consumed
        if condition_id not in preserve_ids
    )
    data["metadata"] = metadata
    return EvidencePlan.from_dict(data)


def _rule_condition_signature_by_id(rule: EvolvingRule) -> Dict[str, tuple[str, bool]]:
    out: Dict[str, tuple[str, bool]] = {}
    for item in list(rule.conditions or []) + list(rule.exclusion_conditions or []):
        condition_id = str(item.id or "").strip().upper()
        if condition_id:
            out[condition_id] = (
                " ".join(str(item.description or "").split()),
                bool(item.expected_answer),
            )
    return out


def _rule_condition_layout(
    rule: EvolvingRule,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def ids(items: List[RuleCondition]) -> tuple[str, ...]:
        return tuple(
            str(item.id or "").strip().upper()
            for item in list(items or [])
            if str(item.id or "").strip()
        )

    return ids(rule.conditions), ids(rule.exclusion_conditions)


def _is_insufficient_validation_rule(rule: EvolvingRule | None) -> bool:
    if rule is None:
        return False
    metadata = dict(rule.metadata or {})
    raw_label = metadata.get("raw_attack_label") or metadata.get("attack_label") or ""
    return normalize_attack_label(str(raw_label)) == "insufficient_validation"


def build_local_rule_candidate_plan(
    *,
    current_rule: EvolvingRule,
    candidate_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
) -> EvidencePlan:
    """Deterministically rebind a Rule candidate to the incumbent Plan."""
    if current_plan is None:
        plan = compile_rule_to_baseline_plan(candidate_rule)
        metadata = dict(plan.metadata or {})
        metadata.update({
            "source": "baseline_for_local_rule_candidate",
            "candidate_strategy": "updated_rule_local_plan_rebind",
            "derived_from_rule_id": candidate_rule.rule_id,
            "derived_from_rule_version": candidate_rule.version,
            "plan_generation_fallback": False,
            "planner_llm_used": False,
            "rule_plan_rebind_mode": "deterministic_baseline",
        })
        plan.metadata = metadata
        PlanValidator().validate_against_rule(
            plan,
            candidate_rule,
            attack_label=str(
                (candidate_rule.metadata or {}).get("attack_label") or ""
            ),
        )
        return plan

    current_signatures = _rule_condition_signature_by_id(current_rule)
    candidate_signatures = _rule_condition_signature_by_id(candidate_rule)
    current_layout = _rule_condition_layout(current_rule)
    candidate_layout = _rule_condition_layout(candidate_rule)
    structural_change = current_layout != candidate_layout
    changed_condition_ids = sorted(
        condition_id
        for condition_id, signature in candidate_signatures.items()
        if current_signatures.get(condition_id) != signature
    )
    added_condition_ids = sorted(
        set(candidate_signatures) - set(current_signatures)
    )
    removed_condition_ids = sorted(
        set(current_signatures) - set(candidate_signatures)
    )
    role_changed_condition_ids = sorted({
        condition_id
        for condition_id in set(current_signatures) & set(candidate_signatures)
        if (
            condition_id in set(current_layout[0])
        ) != (
            condition_id in set(candidate_layout[0])
        )
    })
    dependency_source_ids = sorted({
        *changed_condition_ids,
        *removed_condition_ids,
        *role_changed_condition_ids,
    })

    if structural_change:
        compiled_plan = compile_rule_to_baseline_plan(candidate_rule)
        plan = preserve_unchanged_candidate_plan_steps(
            generated_plan=compiled_plan,
            current_plan=current_plan,
            current_rule=current_rule,
            candidate_rule=candidate_rule,
            protected_condition_ids=role_changed_condition_ids,
        )
        rebound_by_condition = {
            str(step.condition_id or step.id or "").strip().upper(): (
                step.to_dict()
            )
            for step in list(plan.judge_steps or [])
            if str(step.condition_id or step.id or "").strip()
        }
        data = plan.to_dict()
        data["judge_steps"] = [
            copy.deepcopy(rebound_by_condition[condition_id])
            for condition_id in [
                *candidate_layout[0],
                *candidate_layout[1],
            ]
            if condition_id in rebound_by_condition
        ]
        data["focus_steps"] = [
            copy.deepcopy(step.to_dict())
            for step in list(current_plan.focus_steps or [])
        ]
        data["emit_logic"] = compiled_plan.emit_logic
        data["plan_note"] = (
            "Deterministic local Plan rebind for a structural Rule candidate."
        )
    else:
        dependency_affected_ids = sorted(
            dependency_affected_condition_ids(
                current_plan,
                dependency_source_ids,
            )
        )
        candidate_conditions = {
            str(item.id or "").strip().upper(): item
            for item in (
                list(candidate_rule.conditions or [])
                + list(candidate_rule.exclusion_conditions or [])
            )
            if str(item.id or "").strip()
        }
        data = copy.deepcopy(current_plan.to_dict())
        # Rule decision_policy is explanatory prose in many cold-start rules.
        # A text-only rebind retains the validated executable expression.
        data["emit_logic"] = current_plan.emit_logic
        for raw_step in list(data.get("judge_steps") or []):
            if not isinstance(raw_step, dict):
                continue
            condition_id = str(
                raw_step.get("condition_id") or raw_step.get("id") or ""
            ).strip().upper()
            condition = candidate_conditions.get(condition_id)
            if condition_id not in changed_condition_ids or condition is None:
                continue
            raw_step["question"] = (
                "Based only on cited transaction evidence, determine whether "
                "this rule condition is satisfied: "
                f"{condition.description}"
            )
            raw_step["expected_answer"] = bool(condition.expected_answer)

    data["plan_id"] = (
        f"{current_plan.plan_id}__rule_v{candidate_rule.version}"
    )
    data["rule_id"] = candidate_rule.rule_id
    data["rule_version"] = candidate_rule.version
    data["created_at"] = current_plan.created_at

    metadata = dict(data.get("metadata") or {})
    dependency_affected_ids = sorted({
        *list(
            metadata.get("dependency_affected_plan_step_condition_ids") or []
        ),
        *list(
            dependency_affected_condition_ids(
                current_plan,
                dependency_source_ids,
            )
        ),
    })
    metadata.update({
        "source": "local_rule_candidate_rebind",
        "candidate_strategy": "updated_rule_local_plan_rebind",
        "derived_from_rule_id": candidate_rule.rule_id,
        "derived_from_rule_version": candidate_rule.version,
        "changed_rule_condition_ids": dependency_source_ids,
        "added_rule_condition_ids": added_condition_ids,
        "removed_rule_condition_ids": removed_condition_ids,
        "role_changed_rule_condition_ids": role_changed_condition_ids,
        "structural_rule_change": structural_change,
        "rule_condition_layout_before": {
            "positive": list(current_layout[0]),
            "exclusion": list(current_layout[1]),
        },
        "rule_condition_layout_after": {
            "positive": list(candidate_layout[0]),
            "exclusion": list(candidate_layout[1]),
        },
        "preserved_unchanged_plan_steps": sorted(
            set(metadata.get("preserved_unchanged_plan_steps") or [])
            if structural_change
            else (
                set(candidate_signatures)
                - set(changed_condition_ids)
                - set(dependency_affected_ids)
            )
        ),
        "regenerated_plan_step_condition_ids": sorted({
            *list(
                metadata.get("regenerated_plan_step_condition_ids") or []
            ),
            *changed_condition_ids,
            *role_changed_condition_ids,
        }),
        "directly_changed_plan_step_condition_ids": sorted({
            *changed_condition_ids,
            *role_changed_condition_ids,
        } - set(removed_condition_ids)),
        "dependency_affected_plan_step_condition_ids": dependency_affected_ids,
        "dependency_affected_step_policy": (
            "reuse_only_when_state_contract_validation_passes"
        ),
        "plan_generation_fallback": False,
        "planner_llm_used": False,
        "rule_plan_rebind_mode": (
            "deterministic_structural"
            if structural_change
            else "deterministic_text_only"
        ),
    })
    if structural_change:
        metadata["explicit_plan_patch_condition_ids"] = []
    data["metadata"] = metadata
    plan = EvidencePlan.from_dict(data)
    _canonicalize_candidate_plan_followups(
        plan,
        attack_label=str((candidate_rule.metadata or {}).get("attack_label") or ""),
        reason="local_rule_candidate_rebind",
    )
    PlanValidator().validate_against_rule(
        plan,
        candidate_rule,
        attack_label=str(
            (candidate_rule.metadata or {}).get("attack_label") or ""
        ),
    )
    return plan


def _canonicalize_candidate_plan_followups(
    plan: EvidencePlan,
    *,
    attack_label: str,
    reason: str,
) -> EvidencePlan:
    """Clamp inherited candidate-plan follow-up counts to current policy.

    Local rule candidates preserve most of the incumbent plan. Older plans may
    carry max_followups=2 on steps whose current question no longer has a
    semantic two-hop sequence. Keep PlanValidator strict and normalize here.
    """
    normalized: List[Dict[str, Any]] = []
    label = normalize_attack_label(
        attack_label
        or str((plan.metadata or {}).get("attack_label") or ""),
        default="",
    )
    for step in list(plan.judge_steps or []):
        condition_id = str(step.condition_id or step.id or "").strip().upper()
        policy = followup_round_policy(
            step,
            attack_label=label,
            condition_id=condition_id,
        )
        max_allowed = int(policy.get("max_followups", 1) or 0)
        minimum_required = int(policy.get("minimum_followups", 0) or 0)
        current = int(step.max_followups or 0)
        if current > max_allowed:
            step.max_followups = max_allowed
            normalized.append({
                "step_id": step.id,
                "condition_id": condition_id,
                "previous_max_followups": current,
                "current_max_followups": max_allowed,
                "followup_round_policy": policy,
            })
        elif current < minimum_required:
            step.max_followups = minimum_required
            normalized.append({
                "step_id": step.id,
                "condition_id": condition_id,
                "previous_max_followups": current,
                "current_max_followups": minimum_required,
                "followup_round_policy": policy,
            })
        elif current < 0:
            step.max_followups = 0
            normalized.append({
                "step_id": step.id,
                "condition_id": condition_id,
                "previous_max_followups": current,
                "current_max_followups": 0,
                "followup_round_policy": policy,
            })
    if normalized:
        metadata = dict(plan.metadata or {})
        metadata["candidate_plan_followup_canonicalization"] = {
            "reason": reason,
            "normalized_steps": normalized,
            "policy": (
                "Candidate generation may inherit old plan follow-up counts; "
                "normalization clamps only runtime strategy, not rule semantics."
            ),
        }
        plan.metadata = metadata
    return plan


def build_candidate_specs(
    *,
    current_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
    review_bundle: Dict[str, Any],
    update_target: str,
    updater: RuleUpdater,
    plan_updater: PlanUpdater,
    disable_stateful_bindings: bool = False,
    enable_experimental_rule_hypotheses: bool = True,
    enable_phase3_candidate_portfolio: bool = True,
) -> List[Dict[str, Any]]:
    targets = count_review_targets(review_bundle)
    allow_rule = (
        update_target in {"auto", "rule"}
        and targets.get("rule", 0) > 0
    )
    allow_plan = (
        update_target in {"auto", "plan"}
        and targets.get("plan", 0) > 0
    )
    generation_priority = _candidate_generation_priority(review_bundle, targets)
    canonical_lineage_required = str(
        (review_bundle.get("canonical_signal_contract") or {}).get(
            "schema_version"
        )
        or ""
    ).startswith("evotx.canonical_update_signal_contract.")
    specs: List[Dict[str, Any]] = []
    rule_update_bundle = review_bundle
    plan_update_bundle = review_bundle
    if not enable_phase3_candidate_portfolio:
        rule_update_bundle, rule_projection = (
            build_minimal_updater_signal_projection(
                review_bundle,
                owner="rule",
            )
        )
        plan_update_bundle, plan_projection = (
            build_minimal_updater_signal_projection(
                review_bundle,
                owner="plan",
            )
        )
        review_bundle["minimal_updater_projection"] = {
            "schema_version": "evotx.minimal_updater_projection.v1",
            "rule": rule_projection,
            "plan": plan_projection,
        }

    if allow_rule:
        updated_rule = updater.update_rule_from_bundle(
            current_rule,
            rule_update_bundle,
            allow_noop=True,
        )
        if updated_rule.version != current_rule.version:
            atomic_rule_deltas = build_rule_condition_deltas(
                current_rule,
                updated_rule,
                operation_manifest=list(
                    getattr(updater, "last_atomic_delta_manifest", []) or []
                ),
                base_plan=current_plan,
            )
            delta_payloads = [delta.to_dict() for delta in atomic_rule_deltas]
            applied_rule_signal_ids = sorted({
                str(signal_id)
                for delta in delta_payloads
                for signal_id in list(delta.get("applied_signal_ids") or [])
                if str(signal_id)
            })
            suggested_plan_patch_conditions = _review_patch_condition_ids(
                rule_update_bundle,
                patch_key="plan_patch_suggestion",
                include_plan_diagnosis=True,
            )
            same_rule_structure = (
                _rule_condition_layout(current_rule)
                == _rule_condition_layout(updated_rule)
            )
            use_local_rule_plan = (
                same_rule_structure
                or not enable_phase3_candidate_portfolio
            )
            local_plan = None
            local_plan_construction_failed = bool(
                canonical_lineage_required and not applied_rule_signal_ids
            )
            if local_plan_construction_failed:
                rejection = {
                    "owner": "rule",
                    "stage": "candidate_construction",
                    "reason": "actionable_signal_not_materialized",
                    "failure_class": "update_generation_failed",
                    "scope_target_steps": sorted({
                        str(delta.get("condition_id") or "").strip().upper()
                        for delta in delta_payloads
                        if str(delta.get("condition_id") or "").strip()
                    }),
                    "accepted_patch_signal_ids": [],
                }
                review_bundle.setdefault(
                    "rule_candidate_construction_rejections", []
                ).append(rejection)
                print(
                    "[FewShot] Rule candidate rejected because no canonical "
                    "signal was materialized: "
                    f"{stable_json_dumps(rejection)}"
                )
            elif use_local_rule_plan:
                try:
                    local_plan = build_local_rule_candidate_plan(
                        current_rule=current_rule,
                        candidate_rule=updated_rule,
                        current_plan=current_plan,
                    )
                except PlanValidationError as exc:
                    if enable_phase3_candidate_portfolio:
                        raise
                    local_plan_construction_failed = True
                    rejection = {
                        "owner": "rule",
                        "stage": "candidate_construction",
                        "reason": "deterministic_plan_rebind_invalid",
                        "rejection_reason": str(exc),
                        "scope_target_steps": sorted({
                            str(delta.get("condition_id") or "")
                            .strip().upper()
                            for delta in delta_payloads
                            if str(delta.get("condition_id") or "").strip()
                        }),
                        "accepted_patch_signal_ids": sorted({
                            str(signal_id)
                            for delta in delta_payloads
                            for signal_id in list(
                                delta.get("applied_signal_ids") or []
                            )
                            if str(signal_id)
                        }),
                        "planner_fallback_used": False,
                    }
                    review_bundle.setdefault(
                        "rule_candidate_construction_rejections", []
                    ).append(rejection)
                    print(
                        "[FewShot] Rule candidate deterministic Plan rebind "
                        f"rejected: {stable_json_dumps(rejection)}"
                    )
            if local_plan is not None:
                local_plan = normalize_plan_for_runtime_policy(
                    local_plan,
                    disable_stateful_bindings=disable_stateful_bindings,
                    reason="local_rule_candidate",
                )
            if not local_plan_construction_failed:
                specs.append({
                    "name": "rule_candidate",
                    "update_kind": "rule",
                    "rule": updated_rule,
                    "plan": local_plan,
                    "base_rule": current_rule,
                    "base_plan": current_plan,
                    "requires_plan_regeneration": not use_local_rule_plan,
                    "candidate_strategy": (
                        "updated_rule_local_plan_rebind"
                        if use_local_rule_plan
                        else "updated_rule_regenerated_plan"
                    ),
                    "applied_rule_patch_conditions": _review_patch_condition_ids(
                        rule_update_bundle,
                        patch_key="rule_patch_suggestion",
                        include_rule_diagnosis=True,
                    ),
                    # Keep Plan-owner proposals out of Rule-only candidates.
                    "applied_plan_patch_conditions": [],
                    "deferred_plan_patch_conditions": (
                        suggested_plan_patch_conditions
                    ),
                    "regenerated_plan_from_rule": not use_local_rule_plan,
                    "candidate_generation_priority": dict(generation_priority),
                    "atomic_rule_deltas": [
                        delta.to_dict() for delta in atomic_rule_deltas
                    ],
                    "atomic_rule_delta_objects": atomic_rule_deltas,
                    "structural_update_audit": dict(
                        getattr(
                            updater,
                            "last_structural_update_audit",
                            {},
                        ) or {}
                    ),
                })
        elif (updated_rule.metadata or {}).get("last_rule_budget_rejected"):
            print(
                "[FewShot] Rule candidate rejected by complexity budget: "
                f"{stable_json_dumps((updated_rule.metadata or {}).get('last_rule_budget_rejected'))}"
            )

        probe_bundle = (
            build_experimental_rule_probe_bundle(rule_update_bundle)
            if (
                enable_phase3_candidate_portfolio
                and enable_experimental_rule_hypotheses
            )
            else None
        )
        if probe_bundle is not None:
            probe_rule = updater.update_rule_from_bundle(
                current_rule,
                probe_bundle,
                allow_noop=True,
            )
            if probe_rule.version != current_rule.version:
                probe_deltas = build_rule_condition_deltas(
                    current_rule,
                    probe_rule,
                    operation_manifest=list(
                        getattr(updater, "last_atomic_delta_manifest", []) or []
                    ),
                    base_plan=current_plan,
                )
                probe_signal_ids = list(
                    (probe_bundle.get("phase3_experimental_rule_probe") or {}).get(
                        "signal_ids", []
                    )
                )
                specs.append({
                    "name": "rule_hypothesis_probe",
                    "update_kind": "rule",
                    "rule": probe_rule,
                    "plan": None,
                    "base_rule": current_rule,
                    "base_plan": current_plan,
                    "requires_plan_regeneration": True,
                    "candidate_strategy": "experimental_rule_hypothesis",
                    "candidate_status": "experimental_probe",
                    "probe_only": True,
                    "experimental_signal_ids": probe_signal_ids,
                    "hypothesis_source_signal_ids": probe_signal_ids,
                    "hypothesis_lifecycle": {
                        "state": "probe_candidate",
                        "probe_result": "pending",
                        "conversion_reason": "",
                        "direct_promotion_allowed": False,
                    },
                    "applied_rule_patch_conditions": sorted({
                        str(delta.condition_id) for delta in probe_deltas
                    }),
                    "applied_plan_patch_conditions": [],
                    "regenerated_plan_from_rule": True,
                    "candidate_generation_priority": dict(generation_priority),
                    "atomic_rule_deltas": [
                        delta.to_dict() for delta in probe_deltas
                    ],
                    "atomic_rule_delta_objects": probe_deltas,
                    "structural_update_audit": dict(
                        getattr(updater, "last_structural_update_audit", {}) or {}
                    ),
                })

    if allow_plan:
        base_plan = current_plan or compile_rule_to_baseline_plan(current_rule)
        plan_projections = [(
            "plan_candidate",
            "supported",
            build_plan_lifecycle_projection(
                plan_update_bundle,
                lifecycle="supported",
            ),
        )]
        if enable_phase3_candidate_portfolio:
            plan_projections.append((
                "plan_hypothesis_probe",
                "experimental_probe",
                build_plan_lifecycle_projection(
                    plan_update_bundle,
                    lifecycle="experimental_probe",
                ),
            ))
        supported_plan_conditions = _supported_plan_signal_conditions(
            plan_update_bundle
        )
        if enable_phase3_candidate_portfolio and len(supported_plan_conditions) > 1:
            for condition_id in supported_plan_conditions[:3]:
                plan_projections.append(
                    (
                        f"plan_candidate__{condition_id.lower()}",
                        "supported",
                        build_plan_lifecycle_projection(
                            plan_update_bundle,
                            lifecycle="supported",
                            condition_id=condition_id,
                        ),
                    )
                )
        for candidate_name, lifecycle, projected_bundle in plan_projections:
            if projected_bundle is None:
                continue
            updated_plan = plan_updater.update_plan_from_bundle(
                current_rule,
                projected_bundle,
                base_plan=base_plan,
                allow_noop=True,
            )
            updated_plan = normalize_plan_for_runtime_policy(
                updated_plan,
                disable_stateful_bindings=disable_stateful_bindings,
                reason=candidate_name,
            )
            semantic_changed = _plan_changed(updated_plan, base_plan)
            scope_guard = dict((updated_plan.metadata or {}).get("scope_guard") or {})
            print(
                f"[FewShot] {candidate_name} semantic changed: "
                f"{semantic_changed}; scope targets={scope_guard.get('target_steps', [])}"
            )
            if not semantic_changed:
                continue
            projected_matrix = dict(
                projected_bundle.get("update_signal_matrix") or {}
            )
            experimental_ids = [
                str(value)
                for value in list(
                    projected_matrix.get("experimental_plan_signal_ids") or []
                )
                if str(value)
            ]
            strategy_deltas = build_plan_strategy_deltas(base_plan, updated_plan)
            strategy_condition_ids = sorted({
                str(delta.condition_id or "").strip().upper()
                for delta in strategy_deltas
                if str(delta.condition_id or "").strip()
            })
            probe_only = lifecycle == "experimental_probe"
            patch_materialization = dict(
                (updated_plan.metadata or {}).get("plan_patch_materialization")
                or {}
            )
            applied_plan_signal_ids = list(dict.fromkeys(
                str(value)
                for value in list(
                    patch_materialization.get("accepted_signal_ids") or []
                )
                if str(value)
            ))
            if (
                canonical_lineage_required
                and not probe_only
                and not applied_plan_signal_ids
            ):
                update_audits = getattr(plan_updater, "last_update_audits", None)
                if update_audits:
                    update_audits[-1][
                        "candidate_lineage_rejected"
                    ] = True
                    update_audits[-1][
                        "candidate_lineage_rejection_reason"
                    ] = "actionable_signal_not_materialized"
                print(
                    f"[FewShot] {candidate_name} rejected because no canonical "
                    "signal was materialized."
                )
                continue
            spec = {
                "name": candidate_name,
                "update_kind": "plan",
                "rule": current_rule,
                "plan": updated_plan,
                "base_rule": current_rule,
                "base_plan": base_plan,
                "requires_plan_regeneration": False,
                "candidate_strategy": (
                    "experimental_plan_hypothesis"
                    if probe_only
                    else "same_rule_updated_plan"
                ),
                "applied_rule_patch_conditions": [],
                # The materialized Plan diff is the candidate-local scope.
                # Raw suggestions can mention unrelated projected conditions.
                "applied_plan_patch_conditions": strategy_condition_ids,
                "regenerated_plan_from_rule": False,
                "candidate_generation_priority": dict(generation_priority),
                "candidate_status": lifecycle,
                "probe_only": probe_only,
                "experimental_signal_ids": experimental_ids,
                "applied_plan_signal_ids": applied_plan_signal_ids,
                "plan_patch_materialization": patch_materialization,
                "plan_strategy_deltas": [
                    delta.to_dict() for delta in strategy_deltas
                ],
            }
            if probe_only:
                spec["hypothesis_source_signal_ids"] = experimental_ids
                spec["hypothesis_lifecycle"] = {
                    "state": "probe_candidate",
                    "probe_result": "pending",
                    "conversion_reason": "",
                    "direct_promotion_allowed": False,
                }
            specs.append(spec)

    return specs


def build_minimal_updater_signal_projection(
    review_bundle: Dict[str, Any],
    *,
    owner: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Project one source-error signal group for an owner into one Updater call.

    Reviewer routing remains authoritative. The Updater is responsible for
    composing one coherent local candidate from one case-local projection.
    """
    normalized_owner = str(owner or "").strip().lower()
    bundle = copy.deepcopy(review_bundle or {})
    matrix = dict(bundle.get("update_signal_matrix") or {})
    supported = [
        dict(signal)
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict)
        and str(signal.get("update_target") or "").strip().lower()
        == normalized_owner
        and str(signal.get("generalization_status") or "").strip().lower()
        == "supported"
        and str(signal.get("condition_id") or "").strip()
    ]
    audit: Dict[str, Any] = {
        "owner": normalized_owner,
        "selected_condition_id": "",
        "selected_condition_ids": [],
        "selected_signal_ids": [],
        "deferred_signals": [],
        "selection_basis": (
            "active_refinement_then_fn_then_stable_source_identity_"
            "with_same_condition_direction_support"
        ),
    }
    if not supported:
        audit["reason"] = "no_supported_owner_signal"
        return bundle, audit

    source_reviews: Dict[int, Dict[str, Any]] = {}
    owner_projection_map = dict(bundle.get("owner_projections") or {})
    for projections in owner_projection_map.values():
        for raw in list(projections or []):
            if not isinstance(raw, dict):
                continue
            source_reviews.setdefault(
                int(raw.get("source_review_index", -1)),
                raw,
            )
    for raw in list(bundle.get("reviews") or []):
        if isinstance(raw, dict):
            source_reviews.setdefault(
                int(raw.get("source_review_index", -1)),
                raw,
            )

    memory = dict(bundle.get("rejected_update_memory") or {})
    refinement = dict(
        bundle.get("active_refinement_feedback")
        or memory.get("active_refinement_feedback")
        or {}
    )
    refinement_owner = str(
        refinement.get("source_update_kind") or ""
    ).strip().lower()
    refinement_conditions = {
        str(item.get("condition_id") or "").strip().upper()
        for item in list(refinement.get("affected_components") or [])
        if isinstance(item, dict)
        and str(item.get("kind") or "").strip().lower() == normalized_owner
        and str(item.get("condition_id") or "").strip()
    }
    refinement_conditions.update(
        str(value or "").strip().upper()
        for key in ("improved_condition_ids", "regressed_condition_ids")
        for value in list(refinement.get(key) or [])
        if str(value or "").strip()
    )
    refinement_signal_ids = {
        str(value or "")
        for value in list(refinement.get("source_signal_ids") or [])
        if str(value or "")
    }
    refinement_directions = {
        str(value or "").strip().lower()
        for value in list(refinement.get("signal_directions") or [])
        if str(value or "").strip()
    }
    if (
        refinement.get("refinable")
        and refinement_owner == normalized_owner
        and refinement_conditions
        and refinement_directions
    ):
        focused = [
            signal for signal in supported
            if str(signal.get("condition_id") or "").strip().upper()
            in refinement_conditions
            and str(signal.get("direction") or "").strip().lower()
            in refinement_directions
        ]
        if focused:
            supported = focused

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for signal in supported:
        grouped.setdefault(int(signal.get("source_review_index", -1)), []).append(
            signal
        )

    def group_priority(item: tuple[int, List[Dict[str, Any]]]) -> tuple[Any, ...]:
        source_index, signals = item
        condition_ids = sorted({
            str(signal.get("condition_id") or "").strip().upper()
            for signal in signals
            if str(signal.get("condition_id") or "").strip()
        })
        refinement_compatible = bool(
            refinement.get("refinable")
            and refinement_owner == normalized_owner
            and (
                not refinement_conditions
                or bool(set(condition_ids) & refinement_conditions)
            )
        )
        if refinement_compatible and refinement_directions:
            group_signal_ids = {
                str(signal.get("signal_id") or "") for signal in signals
            }
            group_directions = {
                str(signal.get("direction") or "").strip().lower()
                for signal in signals
                if str(signal.get("direction") or "").strip()
            }
            refinement_compatible = bool(
                group_directions.intersection(refinement_directions)
            )
        elif refinement_compatible and refinement_signal_ids:
            # Signal numbering is only a fallback for legacy feedback that has
            # no stable direction. Fresh cross-round feedback uses direction.
            group_signal_ids = {
                str(signal.get("signal_id") or "") for signal in signals
            }
            refinement_compatible = bool(
                group_signal_ids.intersection(refinement_signal_ids)
            )
        error_types = {
            str(signal.get("source_error_type") or "").strip().upper()
            for signal in signals
        }
        error_priority = 0 if "FN" in error_types else 1 if "FP" in error_types else 2
        source = source_reviews.get(source_index, {})
        tx_hash = str(source.get("tx_hash") or "").strip().lower()
        return (
            0 if refinement_compatible else 1,
            error_priority,
            tx_hash,
            tuple(condition_ids),
            source_index,
        )

    ordered_groups = sorted(grouped.items(), key=group_priority)
    selected_source_review_index, selected_signals = ordered_groups[0]
    selected_priority = group_priority(ordered_groups[0])

    # Keep a source-local diagnosis coherent, then include support from other
    # cases for the exact same owner/condition/direction. Transaction identity
    # must not cause an otherwise identical canonical signal to wait forever.
    selected_keys = {
        (
            str(signal.get("condition_id") or "").strip().upper(),
            str(signal.get("direction") or "").strip().lower(),
        )
        for signal in selected_signals
    }
    selected_ids_before_expansion = {
        str(signal.get("signal_id") or "") for signal in selected_signals
    }
    selected_signals = [
        *selected_signals,
        *[
            signal for signal in supported
            if str(signal.get("signal_id") or "")
            not in selected_ids_before_expansion
            and (
                str(signal.get("condition_id") or "").strip().upper(),
                str(signal.get("direction") or "").strip().lower(),
            ) in selected_keys
        ],
    ]

    def serialized_priority(priority: tuple[Any, ...]) -> List[Any]:
        return [
            priority[0],
            priority[1],
            priority[2],
            list(priority[3]),
            priority[4],
        ]
    deferred_signals = [
        {
            "signal_id": str(signal.get("signal_id") or ""),
            "source_review_index": signal.get("source_review_index", -1),
            "source_error_type": str(signal.get("source_error_type") or ""),
            "condition_id": str(signal.get("condition_id") or ""),
            "direction": str(signal.get("direction") or ""),
            "reason": "lower_deterministic_priority",
            "selection_priority": serialized_priority(group_priority((
                int(signal.get("source_review_index", -1)),
                grouped[int(signal.get("source_review_index", -1))],
            ))),
        }
        for signal in supported
        if str(signal.get("signal_id") or "") not in {
            str(selected.get("signal_id") or "")
            for selected in selected_signals
        }
    ]
    selected_conditions = sorted({
        str(signal.get("condition_id") or "").strip().upper()
        for signal in selected_signals
        if str(signal.get("condition_id") or "").strip()
    })
    selected_ids = {
        str(signal.get("signal_id") or "")
        for signal in selected_signals
        if str(signal.get("signal_id") or "")
    }

    selected_by_review: Dict[int, List[Dict[str, Any]]] = {}
    for signal in selected_signals:
        source_index = int(signal.get("source_review_index", -1))
        selected_by_review.setdefault(source_index, []).append(dict(signal))

    def project_reviews(raw_reviews: Any) -> List[Dict[str, Any]]:
        projected: List[Dict[str, Any]] = []
        for raw in list(raw_reviews or []):
            if not isinstance(raw, dict):
                continue
            matching = [
                dict(signal)
                for signal in list(raw.get("update_signals") or [])
                if isinstance(signal, dict)
                and str(signal.get("signal_id") or "") in selected_ids
            ]
            if not matching:
                continue
            review = copy.deepcopy(raw)
            review["update_signals"] = matching
            review["update_target"] = normalized_owner
            review["should_update_rule"] = normalized_owner == "rule"
            review["should_update_plan_strategy"] = normalized_owner == "plan"
            projected.append(review)
        return projected

    owner_review_sources = list(owner_projection_map.get(normalized_owner) or [])
    if not owner_review_sources:
        owner_review_sources = list(bundle.get(
            "actionable_rule_reviews"
            if normalized_owner == "rule"
            else "actionable_plan_reviews"
        ) or [])
    owner_reviews = project_reviews(owner_review_sources)

    # The matrix is the canonical owner/status contract. If the earlier
    # diagnosis projection omitted an owner that the matrix selected, create a
    # minimal owner-local review instead of silently dropping the signal.
    projected_indexes = {
        int(review.get("source_review_index", -1))
        for review in owner_reviews
        if isinstance(review, dict)
    }
    for source_index, signals in selected_by_review.items():
        if source_index in projected_indexes:
            continue
        source = dict(source_reviews.get(source_index) or {})
        projected = {
            "source_review_index": source_index,
            "error_type": str(
                source.get("error_type")
                or signals[0].get("source_error_type")
                or ""
            ),
            "update_target": normalized_owner,
            "should_update_rule": normalized_owner == "rule",
            "should_update_plan_strategy": normalized_owner == "plan",
            "update_signals": [dict(signal) for signal in signals],
            "must_not_change": list(source.get("must_not_change") or []),
            "review_note": str(source.get("review_note") or ""),
            "canonical_signal_projection": True,
        }
        patch_key = (
            "rule_patch_suggestion"
            if normalized_owner == "rule"
            else "plan_patch_suggestion"
        )
        if isinstance(source.get(patch_key), dict):
            projected[patch_key] = copy.deepcopy(source[patch_key])
        owner_reviews.append(projected)

    matrix["signals"] = selected_signals
    matrix["supported_signal_ids"] = sorted(selected_ids)
    matrix["experimental_rule_signal_ids"] = []
    matrix["experimental_plan_signal_ids"] = []
    matrix["conflicted_signal_ids"] = []
    matrix["insufficient_signal_ids"] = []
    matrix["joint_resolution_groups"] = []
    matrix["plan_actionable_signal_ids"] = (
        sorted(selected_ids) if normalized_owner == "plan" else []
    )
    bundle["update_signal_matrix"] = matrix
    bundle["actionable_rule_reviews"] = (
        owner_reviews if normalized_owner == "rule" else []
    )
    bundle["actionable_plan_reviews"] = (
        owner_reviews if normalized_owner == "plan" else []
    )
    bundle["non_rule_reviews"] = (
        owner_reviews if normalized_owner == "plan" else []
    )
    bundle["owner_projections"] = {normalized_owner: owner_reviews}

    audit.update({
        "selected_source_review_index": selected_source_review_index,
        "selected_source_review_indexes": sorted({
            int(signal.get("source_review_index", -1))
            for signal in selected_signals
        }),
        "selected_condition_id": (
            selected_conditions[0] if len(selected_conditions) == 1 else ""
        ),
        "selected_condition_ids": selected_conditions,
        "selected_signal_ids": sorted(selected_ids),
        "selected_priority": serialized_priority(selected_priority),
        "active_refinement_considered": bool(refinement),
        "active_refinement_condition_ids": sorted(refinement_conditions),
        "active_refinement_signal_ids": sorted(refinement_signal_ids),
        "active_refinement_directions": sorted(refinement_directions),
        "deferred_signals": deferred_signals,
    })
    bundle["minimal_updater_projection"] = dict(audit)
    return bundle, audit


def build_minimal_candidate_specs(
    specs: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep one coherent supported Rule candidate and one Plan candidate."""
    selected: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for update_kind, preferred_name in (
        ("rule", "rule_candidate"),
        ("plan", "plan_candidate"),
    ):
        eligible = [
            dict(spec)
            for spec in list(specs or [])
            if str(spec.get("update_kind") or "") == update_kind
            and not bool(spec.get("probe_only"))
            and str(spec.get("candidate_status") or "supported") == "supported"
        ]
        if not eligible:
            continue
        candidate = next(
            (
                item for item in eligible
                if str(item.get("name") or "") == preferred_name
            ),
            eligible[0],
        )
        if update_kind == "rule":
            atomic_deltas = [
                dict(item)
                for item in list(candidate.get("atomic_rule_deltas") or [])
                if isinstance(item, dict)
            ]
            ineligible = [
                item
                for item in atomic_deltas
                if str(item.get("local_viability_status") or "eligible")
                != "eligible"
            ]
            if not atomic_deltas:
                dropped.append({
                    "name": str(candidate.get("name") or preferred_name),
                    "reason": "attributed_atomic_delta_lineage_unavailable",
                    "ineligible_delta_ids": [],
                })
                continue
            candidate["direction_viability_warnings"] = [
                {
                    "delta_id": str(item.get("delta_id") or ""),
                    "condition_id": str(item.get("condition_id") or ""),
                    "reason": str(item.get("local_viability_reason") or ""),
                }
                for item in ineligible
            ]
            candidate["selected_atomic_deltas"] = atomic_deltas
            candidate["selected_atomic_delta_ids"] = [
                str(item.get("delta_id") or "") for item in atomic_deltas
            ]
            candidate["applied_rule_patch_conditions"] = sorted({
                str(item.get("condition_id") or "").strip().upper()
                for item in atomic_deltas
                if str(item.get("condition_id") or "").strip()
            })
        candidate["candidate_strategy"] = (
            "minimal_coherent_rule_update"
            if update_kind == "rule"
            else "minimal_local_plan_update"
        )
        candidate["minimal_pipeline"] = True
        selected.append(candidate)
        for item in eligible:
            if str(item.get("name") or "") != str(candidate.get("name") or ""):
                dropped.append({
                    "name": str(item.get("name") or ""),
                    "reason": "advanced_alternative_not_in_minimal_pipeline",
                })
    return selected, {
        "schema_version": "evotx.minimal_candidate_selection.v1",
        "mode": "one_rule_one_plan",
        "input_candidate_count": len(list(specs or [])),
        "selected_candidates": [
            str(item.get("name") or "") for item in selected
        ],
        "dropped_candidates": dropped,
        "rule_subset_search_enabled": False,
        "experimental_candidates_enabled": False,
        "joint_candidates_enabled": False,
    }


def build_experimental_rule_probe_bundle(
    review_bundle: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Project one experimental Rule hypothesis without changing its lifecycle."""
    bundle = copy.deepcopy(review_bundle or {})
    matrix = dict(bundle.get("update_signal_matrix") or {})
    experimental_ids = [
        str(value)
        for value in list(matrix.get("experimental_rule_signal_ids") or [])
        if str(value)
    ][:1]
    if not experimental_ids:
        return None
    selected_ids = set(experimental_ids)
    selected_signals: List[Dict[str, Any]] = []
    selected_review_indexes: set[int] = set()
    for raw in list(matrix.get("signals") or []):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("signal_id") or "") not in selected_ids:
            continue
        signal = dict(raw)
        selected_signals.append(signal)
        selected_review_indexes.add(int(signal.get("source_review_index", -1)))
    if not selected_signals:
        return None
    matrix["signals"] = selected_signals
    matrix["supported_signal_ids"] = []
    matrix["experimental_rule_signal_ids"] = experimental_ids
    matrix["conflicted_signal_ids"] = []
    matrix["insufficient_signal_ids"] = []
    matrix["joint_resolution_groups"] = []
    matrix["plan_actionable_signal_ids"] = []
    bundle["update_signal_matrix"] = matrix
    probe_reviews: List[Dict[str, Any]] = []
    rule_projections = list(
        dict(bundle.get("owner_projections") or {}).get("rule") or []
    )
    if not rule_projections:
        rule_projections = list(bundle.get("actionable_rule_reviews") or [])
    for review in rule_projections:
        if not isinstance(review, dict):
            continue
        if int(review.get("source_review_index", -1)) not in selected_review_indexes:
            continue
        projected = dict(review)
        projected["update_signals"] = [
            dict(signal) for signal in selected_signals
            if int(signal.get("source_review_index", -1))
            == int(projected.get("source_review_index", -1))
        ]
        projected["phase3_experimental_rule_probe"] = True
        projected["should_update_rule"] = True
        projected["update_target"] = "rule"
        probe_reviews.append(projected)
    if not probe_reviews:
        return None
    bundle["actionable_rule_reviews"] = probe_reviews
    bundle["actionable_plan_reviews"] = []
    bundle["non_rule_reviews"] = []
    bundle["phase3_experimental_rule_probe"] = {
        "signal_ids": experimental_ids,
        "probe_only": True,
        "direct_promotion_allowed": False,
    }
    return bundle


def build_plan_lifecycle_projection(
    review_bundle: Dict[str, Any],
    *,
    lifecycle: str,
    condition_id: str | None = None,
) -> Dict[str, Any] | None:
    """Isolate supported Plan updates from experimental probe hypotheses."""
    if lifecycle not in {"supported", "experimental_probe"}:
        raise ValueError(f"unsupported Plan lifecycle: {lifecycle}")
    bundle = copy.deepcopy(review_bundle or {})
    matrix = dict(bundle.get("update_signal_matrix") or {})
    canonical_matrix_authoritative = bool(
        bundle.get("canonical_signal_contract")
        or str(matrix.get("schema_version") or "").startswith(
            "evotx.canonical_update_signal"
        )
    )
    legacy_without_signal_matrix = (
        not canonical_matrix_authoritative
        and not bool(matrix.get("signals"))
    )
    has_phase2_owner_projection = bool(
        isinstance(bundle.get("owner_projections"), dict)
        or "actionable_plan_reviews" in bundle
    )
    wanted_status = "supported" if lifecycle == "supported" else "experimental_plan"
    wanted_condition = str(condition_id or "").strip().upper()
    selected_signals = [
        dict(signal)
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict)
        and str(signal.get("update_target") or "").strip().lower() == "plan"
        and str(signal.get("generalization_status") or "").strip().lower()
        == wanted_status
        and (
            not wanted_condition
            or str(signal.get("condition_id") or "").strip().upper()
            == wanted_condition
        )
    ]
    if lifecycle == "experimental_probe":
        selected_signals = selected_signals[:1]
    selected_ids = {
        str(signal.get("signal_id") or "")
        for signal in selected_signals
        if str(signal.get("signal_id") or "")
    }
    selected_by_id = {
        str(signal.get("signal_id") or ""): dict(signal)
        for signal in selected_signals
        if str(signal.get("signal_id") or "")
    }

    projected_reviews: List[Dict[str, Any]] = []
    # Phase 2 projections are the ownership contract. Re-reading raw reviews
    # here would reintroduce Rule/Packet/Runtime roots into Plan scope. Keep the
    # raw fallback only for historical pre-routing bundles.
    review_sources = list(bundle.get("actionable_plan_reviews") or [])
    if not review_sources and has_phase2_owner_projection:
        review_sources = [
            review
            for review in list(bundle.get("non_rule_reviews") or [])
            if isinstance(review, dict)
            and (
                str(review.get("update_target") or "").strip().lower() == "plan"
                or any(
                    isinstance(signal, dict)
                    and str(signal.get("signal_id") or "") in selected_ids
                    for signal in list(review.get("update_signals") or [])
                )
            )
        ]
    if legacy_without_signal_matrix and not has_phase2_owner_projection:
        review_sources.extend(_reviews_from_bundle(bundle))
    seen_reviews: set[str] = set()
    for review in review_sources:
        if not isinstance(review, dict):
            continue
        review_key = stable_json_dumps({
            "source_review_index": review.get("source_review_index"),
            "error_type": review.get("error_type"),
            "plan_patch_suggestion": review.get("plan_patch_suggestion"),
            "update_signals": review.get("update_signals"),
        })
        if review_key in seen_reviews:
            continue
        seen_reviews.add(review_key)
        matching = []
        for signal in list(review.get("update_signals") or []):
            if not isinstance(signal, dict):
                continue
            signal_id = str(signal.get("signal_id") or "")
            if signal_id not in selected_ids:
                continue
            matching.append(dict(selected_by_id.get(signal_id) or signal))
        deterministic_restore = (
            lifecycle == "supported"
            and _review_requests_deterministic_plan_restore(review)
        )
        legacy_plan_update = bool(
            lifecycle == "supported"
            and legacy_without_signal_matrix
            and review.get("should_update_plan_strategy")
        )
        if not matching and not deterministic_restore and not legacy_plan_update:
            continue
        projected = dict(review)
        projected["update_signals"] = matching
        projected["update_target"] = "plan"
        projected["should_update_rule"] = False
        projected["should_update_plan_strategy"] = True
        projected_reviews.append(projected)
    if not projected_reviews:
        return None

    selected_id_list = sorted(selected_ids)
    matrix["signals"] = selected_signals
    matrix["supported_signal_ids"] = (
        selected_id_list if lifecycle == "supported" else []
    )
    matrix["experimental_plan_signal_ids"] = (
        selected_id_list if lifecycle == "experimental_probe" else []
    )
    matrix["experimental_rule_signal_ids"] = []
    matrix["conflicted_signal_ids"] = []
    matrix["insufficient_signal_ids"] = []
    matrix["plan_actionable_signal_ids"] = selected_id_list
    matrix["joint_resolution_groups"] = []
    bundle["update_signal_matrix"] = matrix
    bundle["actionable_rule_reviews"] = []
    bundle["actionable_plan_reviews"] = projected_reviews
    bundle["non_rule_reviews"] = projected_reviews
    bundle["phase3_plan_lifecycle_projection"] = {
        "lifecycle": lifecycle,
        "hypothesis_source_signal_ids": selected_id_list,
        "direct_promotion_allowed": lifecycle == "supported",
        "condition_id": wanted_condition,
    }
    return bundle


def _supported_plan_signal_conditions(review_bundle: Dict[str, Any]) -> List[str]:
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    conditions = sorted({
        str(signal.get("condition_id") or "").strip().upper()
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict)
        and str(signal.get("update_target") or "").strip().lower() == "plan"
        and str(signal.get("generalization_status") or "").strip().lower()
        == "supported"
        and re.fullmatch(r"[CE]\d+", str(signal.get("condition_id") or "").strip().upper())
    })
    return conditions


def build_plan_signal_coverage_audit(
    review_bundle: Dict[str, Any],
    candidate_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    plan_signals = [
        dict(signal)
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict)
        and str(signal.get("update_target") or "").strip().lower() == "plan"
        and str(signal.get("generalization_status") or "").strip().lower()
        in {"supported", "experimental_plan"}
        and str(signal.get("signal_id") or "")
    ]
    supported_ids = {
        str(signal.get("signal_id") or "")
        for signal in plan_signals
        if str(signal.get("generalization_status") or "").strip().lower()
        == "supported"
    }
    deferred_ids = _minimal_deferred_signal_ids(
        review_bundle,
        owner="plan",
    )
    applied_ids: set[str] = set()
    selected_by_candidate: List[Dict[str, Any]] = []
    for spec in list(candidate_specs or []):
        raw_ids = [
            *list(spec.get("applied_plan_signal_ids") or []),
            *[
                signal_id
                for delta in list(spec.get("plan_strategy_deltas") or [])
                if isinstance(delta, dict)
                for signal_id in list(delta.get("source_signal_ids") or [])
            ],
        ]
        ids = sorted({
            str(value)
            for value in raw_ids
            if str(value)
        })
        applied_ids.update(ids)
        selected_by_candidate.append({
            "candidate_name": str(spec.get("name") or ""),
            "update_kind": str(spec.get("update_kind") or ""),
            "candidate_status": str(spec.get("candidate_status") or ""),
            "probe_only": bool(spec.get("probe_only")),
            "applied_plan_signal_ids": ids,
            "applied_plan_patch_conditions": sorted({
                str(value or "").strip().upper()
                for value in list(spec.get("applied_plan_patch_conditions") or [])
                if str(value or "").strip()
            }),
        })
    by_condition: Dict[str, Dict[str, Any]] = {}
    for signal in plan_signals:
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        if not condition_id:
            condition_id = "UNKNOWN"
        bucket = by_condition.setdefault(
            condition_id,
            {
                "supported_signal_ids": [],
                "experimental_signal_ids": [],
                "covered_signal_ids": [],
                "deferred_signal_ids": [],
                "uncovered_signal_ids": [],
            },
        )
        signal_id = str(signal.get("signal_id") or "")
        if str(signal.get("generalization_status") or "").strip().lower() == "supported":
            bucket["supported_signal_ids"].append(signal_id)
        else:
            bucket["experimental_signal_ids"].append(signal_id)
        if signal_id in applied_ids:
            bucket["covered_signal_ids"].append(signal_id)
        elif signal_id in deferred_ids:
            bucket["deferred_signal_ids"].append(signal_id)
        else:
            bucket["uncovered_signal_ids"].append(signal_id)
    return {
        "schema_version": "evotx.plan_signal_coverage_audit.v1",
        "supported_plan_signal_ids": sorted(supported_ids),
        "candidate_applied_plan_signal_ids": sorted(applied_ids),
        "deferred_supported_plan_signal_ids": sorted(
            supported_ids.intersection(deferred_ids)
        ),
        "uncovered_supported_plan_signal_ids": sorted(
            supported_ids - applied_ids - deferred_ids
        ),
        "selected_by_candidate": selected_by_candidate,
        "by_condition": by_condition,
    }


def _minimal_deferred_signal_ids(
    review_bundle: Dict[str, Any] | None,
    *,
    owner: str | None = None,
) -> set[str]:
    projection = dict(
        (review_bundle or {}).get("minimal_updater_projection") or {}
    )
    owners = [str(owner).strip().lower()] if owner else ["rule", "plan"]
    return {
        str(item.get("signal_id") or "")
        for owner_name in owners
        for item in list(
            dict(projection.get(owner_name) or {}).get("deferred_signals")
            or []
        )
        if isinstance(item, dict) and str(item.get("signal_id") or "")
    }


def build_deferred_signal_refinement_feedback(
    review_bundle: Dict[str, Any] | None,
    signal_terminal_audit: Dict[str, Any] | None,
    *,
    round_index: int,
) -> Dict[str, Any]:
    """Advance one already-supported signal that the minimal projection deferred."""
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    signals = {
        str(signal.get("signal_id") or ""): dict(signal)
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict) and str(signal.get("signal_id") or "")
    }
    deferred = next((
        dict(terminal)
        for terminal in list((signal_terminal_audit or {}).get("terminals") or [])
        if isinstance(terminal, dict)
        and str(terminal.get("terminal_reason") or "") == "deferred_next_round"
        and str(terminal.get("signal_id") or "") in signals
    ), None)
    if deferred is None:
        return {}
    signal_id = str(deferred.get("signal_id") or "")
    signal = signals[signal_id]
    owner = str(signal.get("update_target") or "").strip().lower()
    condition_id = str(signal.get("condition_id") or "").strip().upper()
    direction = str(signal.get("direction") or "").strip().lower()
    if owner not in {"rule", "plan"} or not condition_id:
        return {}
    return {
        "schema_version": "evotx.refinement_feedback.v1",
        "source_round": int(round_index),
        "source_candidate": f"deferred_signal:{signal_id}",
        "source_candidate_status": "deferred_supported_signal",
        "source_update_kind": owner,
        "source_signal_ids": [signal_id],
        "signal_directions": [direction] if direction else [],
        "rejection_reason": "deferred_next_round",
        "refinable": True,
        "refinability_reason": "supported_signal_not_yet_materialized",
        "improved_targets": [],
        "preserved_behaviors": [],
        "regressions": [],
        "affected_components": [{
            "kind": owner,
            "condition_id": condition_id,
            "operation": direction or "local_update",
            "capability_id": "",
        }],
        "next_allowed_refinement_kind": "advance_supported_signal",
        "next_round_action": "advance_signal",
        "must_preserve_candidate_effect": False,
        "desired_refinement": [
            "Materialize this deferred canonical signal without changing its owner or condition."
        ],
    }


def _review_requests_deterministic_plan_restore(review: Dict[str, Any]) -> bool:
    return any(
        isinstance(item, dict)
        and str(item.get("requested_operation") or "").strip().lower()
        == "restore_question_from_rule"
        for item in list(review.get("root_causes") or [])
    )


def build_confirmed_hypothesis_spec(
    probe: Dict[str, Any],
    *,
    current_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
) -> Dict[str, Any] | None:
    """Convert only a passed probe into a normal, still-unvalidated candidate."""
    comparison = dict(probe.get("comparison") or {})
    if not bool(probe.get("probe_only")) or not bool(comparison.get("probe_passed")):
        return None
    source_signal_ids = list(
        probe.get("hypothesis_source_signal_ids")
        or probe.get("experimental_signal_ids")
        or []
    )
    source_signal_set = {
        str(value) for value in source_signal_ids if str(value)
    }
    component_signal_ids = {
        str(value)
        for delta in list(probe.get("selected_atomic_deltas") or [])
        if isinstance(delta, dict)
        for value in list(delta.get("applied_signal_ids") or [])
        if str(value)
    } | {
        str(value)
        for value in list(probe.get("applied_plan_signal_ids") or [])
        if str(value)
    }
    if (
        not source_signal_set
        or not component_signal_ids
        or not component_signal_ids.issubset(source_signal_set)
        or str(probe.get("candidate_strategy") or "")
        == "bounded_supported_plus_plan_probe"
    ):
        return None
    is_activation_probe = (
        str(probe.get("candidate_strategy") or "") == "evidence_activation_probe"
    )
    return {
        "name": f"{probe.get('name', 'probe')}__confirmed",
        "update_kind": str(probe.get("update_kind") or "rule"),
        "rule": probe["rule"],
        "plan": probe.get("plan"),
        "base_rule": current_rule,
        "base_plan": current_plan,
        "requires_plan_regeneration": False,
        "candidate_strategy": (
            f"{probe.get('candidate_strategy', 'experimental_probe')}_confirmed"
        ),
        "candidate_status": "supported_after_probe",
        "probe_only": False,
        "skip_candidate_canary": not is_activation_probe,
        "experimental_signal_ids": list(probe.get("experimental_signal_ids") or []),
        "hypothesis_source_signal_ids": source_signal_ids,
        "phase3_activation_probe": dict(probe.get("phase3_activation_probe") or {}),
        "plan_strategy_deltas": list(probe.get("plan_strategy_deltas") or []),
        "applied_plan_signal_ids": list(
            probe.get("applied_plan_signal_ids") or []
        ),
        "plan_patch_materialization": dict(
            probe.get("plan_patch_materialization") or {}
        ),
        "condition_evidence_dependencies": list(
            probe.get("condition_evidence_dependencies") or []
        ),
        "candidate_viability": dict(probe.get("candidate_viability") or {}),
        "selected_atomic_deltas": list(probe.get("selected_atomic_deltas") or []),
        "rule_delta_synthesis": dict(probe.get("rule_delta_synthesis") or {}),
        "applied_rule_patch_conditions": list(
            probe.get("applied_rule_patch_conditions") or []
        ),
        "applied_plan_patch_conditions": list(
            probe.get("applied_plan_patch_conditions") or []
        ),
        "regenerated_plan_from_rule": bool(probe.get("regenerated_plan_from_rule")),
        "phase3_probe_confirmation": {
            "source_probe": str(probe.get("name") or ""),
            "probe_canary": dict(comparison.get("candidate_canary") or {}),
            "direct_probe_promotion": False,
        },
        "hypothesis_lifecycle": {
            "state": "confirmed_supported",
            "probe_result": "passed",
            "conversion_reason": (
                "probe_passed; materialized as a separate candidate that still "
                "requires normal full validation"
            ),
            "source_probe": str(probe.get("name") or ""),
            "direct_promotion_allowed": False,
        },
    }


def _plan_candidate_signal_status(review_bundle: Dict[str, Any]) -> str:
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    plan_signals = {
        str(signal.get("signal_id") or ""): str(
            signal.get("generalization_status") or ""
        ).strip().lower()
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict)
        and str(signal.get("update_target") or "").strip().lower() == "plan"
        and str(signal.get("signal_id") or "")
    }
    actionable = {
        str(value)
        for value in list(matrix.get("plan_actionable_signal_ids") or [])
        if str(value) in plan_signals
    }
    experimental = {
        str(value)
        for value in list(matrix.get("experimental_plan_signal_ids") or [])
        if str(value)
    }
    return (
        "experimental_probe"
        if actionable
        and actionable.issubset(experimental)
        and all(plan_signals.get(value) == "experimental_plan" for value in actionable)
        else "supported"
    )


def synthesize_rule_candidate_specs(
    specs: List[Dict[str, Any]],
    *,
    current_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
    current_slim_results: List[Dict[str, Any]],
    review_bundle: Dict[str, Any] | None,
    max_candidates: int,
    disable_stateful_bindings: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compose local Rule deltas before full-candidate viability/runtime A/B."""
    output: List[Dict[str, Any]] = []
    parent_audits: List[Dict[str, Any]] = []
    joint_groups = list(
        ((review_bundle or {}).get("update_signal_matrix") or {}).get(
            "joint_resolution_groups", []
        )
        or []
    )
    for spec in list(specs or []):
        if str(spec.get("update_kind") or "") != "rule":
            output.append(spec)
            continue
        deltas = list(spec.get("atomic_rule_delta_objects") or [])
        if not deltas:
            deltas = build_rule_condition_deltas(
                current_rule,
                spec["rule"],
                base_plan=current_plan,
                candidate_plan=(
                    spec.get("plan")
                    if isinstance(spec.get("plan"), EvidencePlan)
                    else None
                ),
            )
        atomic_dependency_preflight = [
            {
                "delta_id": delta.delta_id,
                **rule_delta_dependency_preflight(
                    base_plan=current_plan,
                    deltas=deltas,
                    accepted_delta_ids=[delta.delta_id],
                ),
            }
            for delta in deltas
        ]
        dependency_eligible_delta_ids = {
            item["delta_id"]
            for item in atomic_dependency_preflight
            if item["dependency_validation_status"] == "passed"
        }
        dependency_eligible_deltas = [
            delta for delta in deltas
            if delta.delta_id in dependency_eligible_delta_ids
        ]
        subsets, audit = generate_blocker_aware_rule_delta_subsets(
            dependency_eligible_deltas,
            current_slim_results,
            joint_resolution_groups=joint_groups,
            max_candidates=max_candidates,
        )
        audit["atomic_dependency_preflight"] = atomic_dependency_preflight
        audit["atomic_delta_count"] = len(deltas)
        audit["dependency_eligible_atomic_delta_count"] = len(
            dependency_eligible_deltas
        )
        audit["atomic_dependency_rejected_delta_ids"] = [
            item["delta_id"]
            for item in atomic_dependency_preflight
            if item["dependency_validation_status"] != "passed"
        ]
        audit["parent_candidate"] = str(spec.get("name") or "rule_candidate")
        parent_audits.append(audit)
        if not subsets:
            continue
        all_delta_ids = {delta.delta_id for delta in deltas}
        dependency_rejections: List[Dict[str, Any]] = []
        for subset_index, subset in enumerate(subsets, start=1):
            selected_ids = set(subset.get("accepted_delta_ids") or [])
            dependency_preflight = rule_delta_dependency_preflight(
                base_plan=current_plan,
                deltas=deltas,
                accepted_delta_ids=selected_ids,
            )
            if dependency_preflight["dependency_validation_status"] != "passed":
                dependency_rejections.append({
                    "subset_index": subset_index,
                    "search_strategy": subset.get("search_strategy", ""),
                    **dependency_preflight,
                })
                continue
            candidate_spec = dict(spec)
            if selected_ids == all_delta_ids:
                candidate_rule = spec["rule"]
                candidate_plan = spec.get("plan")
                requires_plan_regeneration = bool(
                    spec.get("requires_plan_regeneration")
                )
            else:
                candidate_rule = materialize_synthesized_rule_candidate(
                    base_rule=current_rule,
                    candidate_rule=spec["rule"],
                    deltas=deltas,
                    accepted_delta_ids=selected_ids,
                    parent_candidate=str(spec.get("name") or "rule_candidate"),
                    synthesis_strategy=str(subset.get("search_strategy") or ""),
                )
                structural = any(
                    delta.operation in {"add", "remove"}
                    for delta in deltas
                    if delta.delta_id in selected_ids
                )
                candidate_plan = (
                    build_local_rule_candidate_plan(
                        current_rule=current_rule,
                        candidate_rule=candidate_rule,
                        current_plan=current_plan,
                    )
                    if _is_insufficient_validation_rule(current_rule)
                    and current_plan is not None
                    and not structural
                    else None
                )
                if candidate_plan is not None:
                    candidate_plan = normalize_plan_for_runtime_policy(
                        candidate_plan,
                        disable_stateful_bindings=disable_stateful_bindings,
                        reason="blocker_aware_rule_synthesis",
                    )
                requires_plan_regeneration = candidate_plan is None
            selected_deltas = [
                delta for delta in deltas if delta.delta_id in selected_ids
            ]
            direct_dependency_ids = sorted({
                str(delta.condition_id or "").strip().upper()
                for delta in selected_deltas
                if str(delta.condition_id or "").strip()
            })
            dependency_base_plan = current_plan
            if dependency_base_plan is not None and disable_stateful_bindings:
                dependency_base_plan = normalize_plan_for_runtime_policy(
                    dependency_base_plan,
                    disable_stateful_bindings=True,
                    reason="pre_viability_dependency_baseline",
                )
            dependency_preview_plan = candidate_plan
            if dependency_preview_plan is None:
                dependency_preview_plan = preserve_unchanged_candidate_plan_steps(
                    generated_plan=compile_rule_to_baseline_plan(candidate_rule),
                    current_plan=current_plan,
                    current_rule=current_rule,
                    candidate_rule=candidate_rule,
                    protected_condition_ids=list(
                        spec.get("applied_plan_patch_conditions") or []
                    ),
                )
                dependency_preview_plan = normalize_plan_for_runtime_policy(
                    dependency_preview_plan,
                    disable_stateful_bindings=disable_stateful_bindings,
                    reason="pre_viability_dependency_preview",
                )
            pre_viability_dependency_validation = dependency_consistency_report(
                dependency_preview_plan,
                previous_plan=dependency_base_plan,
                changed_condition_ids=direct_dependency_ids,
            )
            try:
                PlanValidator().validate_against_rule(
                    dependency_preview_plan,
                    candidate_rule,
                    attack_label=str(
                        (candidate_rule.metadata or {}).get("attack_label") or ""
                    ),
                    previous_plan=dependency_base_plan,
                    changed_condition_ids=direct_dependency_ids,
                )
            except PlanValidationError as exc:
                pre_viability_dependency_validation[
                    "dependency_validation_status"
                ] = "rejected"
                pre_viability_dependency_validation[
                    "dependency_rejection_reason"
                ] = str(exc)
                pre_viability_dependency_validation.setdefault(
                    "errors", []
                ).append({
                    "code": "pre_viability_rule_plan_dependency_invalid",
                    "error": str(exc),
                })
            if (
                pre_viability_dependency_validation[
                    "dependency_validation_status"
                ]
                != "passed"
            ):
                dependency_rejections.append({
                    "subset_index": subset_index,
                    "search_strategy": subset.get("search_strategy", ""),
                    "stage": "materialized_candidate_before_viability",
                    **pre_viability_dependency_validation,
                })
                continue
            candidate_spec.update({
                "name": f"rule_candidate_synth_{subset_index:02d}",
                "rule": candidate_rule,
                "plan": candidate_plan,
                "requires_plan_regeneration": requires_plan_regeneration,
                "candidate_strategy": "blocker_aware_rule_synthesis",
                "parent_candidate": str(spec.get("name") or "rule_candidate"),
                "atomic_rule_deltas": [delta.to_dict() for delta in deltas],
                "atomic_rule_delta_objects": deltas,
                "selected_atomic_delta_ids": [
                    delta.delta_id for delta in selected_deltas
                ],
                "selected_atomic_deltas": [
                    delta.to_dict() for delta in selected_deltas
                ],
                "rule_delta_synthesis": dict(subset),
                "dependency_preflight": dependency_preflight,
                "pre_viability_dependency_validation": (
                    pre_viability_dependency_validation
                ),
                "applied_rule_patch_conditions": sorted({
                    str(delta.condition_id) for delta in selected_deltas
                }),
                "regenerated_plan_from_rule": requires_plan_regeneration,
            })
            output.append(candidate_spec)
        audit["dependency_rejected_candidates"] = dependency_rejections
        audit["dependency_rejected_candidate_count"] = len(
            dependency_rejections
        )
        audit["dependency_valid_candidate_count"] = max(
            0,
            len(subsets) - len(dependency_rejections),
        )
    return output, {
        "schema_version": "evotx.rule_delta_synthesis_round.v1",
        "parent_candidate_count": len(parent_audits),
        "output_candidate_count": len(output),
        "parents": parent_audits,
        "validation_layers": {
            "offline_replay": (
                "Reviewer-to-delta-to-synthesis construction only"
            ),
            "runtime_ab": (
                "Required below: same few-shot cases run before/after through "
                "existing canary and regression evaluation"
            ),
        },
    }


def build_phase1_rejection_funnel(
    rule_delta_synthesis: Dict[str, Any],
    candidate_viability: Dict[str, Any],
    candidate_evaluations: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    parents = list((rule_delta_synthesis or {}).get("parents") or [])
    evaluations = list(candidate_evaluations or [])
    probe_evaluations = [
        item for item in evaluations if bool(item.get("probe_only"))
    ]
    promotion_evaluations = [
        item for item in evaluations if not bool(item.get("probe_only"))
    ]
    canary_failed = sum(
        1 for item in promotion_evaluations
        if str((item.get("comparison") or {}).get("candidate_evaluation_scope") or "")
        == "canary_only"
    )
    accepted = sum(
        1 for item in promotion_evaluations
        if bool((item.get("comparison") or {}).get("accept"))
    )
    pre_candidate_rejections = list(
        (candidate_viability or {}).get("pre_candidate_rejections") or []
    )
    portfolio = dict(
        (candidate_viability or {}).get("phase3_candidate_portfolio") or {}
    )
    hard_guard_failed = sum(
        1
        for item in promotion_evaluations
        if bool(((item.get("comparison") or {}).get("guard_gate") or {}).get("enabled"))
        and not bool(
            ((item.get("comparison") or {}).get("guard_gate") or {}).get(
                "accept", True
            )
        )
    )
    return {
        "construction_failed": sum(
            1
            for item in pre_candidate_rejections
            if not bool(item.get("candidate_survived_construction"))
        ),
        "construction_component_rejected": sum(
            len(list(item.get("rejected_patch_components") or []))
            for item in pre_candidate_rejections
        ),
        "portfolio_deduplicated": len(
            list(portfolio.get("deduplicated_candidates") or [])
        ),
        "portfolio_truncated": int(
            portfolio.get("truncated_candidate_count") or 0
        ),
        "atomic_delta_generated": sum(
            int(parent.get("atomic_delta_count") or 0) for parent in parents
        ),
        "atomic_delta_rejected": sum(
            len(list(parent.get("rejected_atomic_deltas") or []))
            for parent in parents
        ),
        "synthesis_failed": sum(
            1 for parent in parents
            if int(parent.get("synthesized_candidate_count") or 0) == 0
        ),
        "candidate_synthesized": sum(
            int(parent.get("synthesized_candidate_count") or 0)
            for parent in parents
        ),
        "dependency_validation_failed": sum(
            int(parent.get("dependency_rejected_candidate_count") or 0)
            for parent in parents
        ),
        "viability_failed": int(
            (candidate_viability or {}).get("rejected_candidate_count") or 0
        ),
        "runtime_evaluated": len(evaluations),
        "experimental_probe_evaluated": len(probe_evaluations),
        "experimental_probe_passed": sum(
            1
            for item in probe_evaluations
            if bool((item.get("comparison") or {}).get("probe_passed"))
        ),
        "canary_failed": canary_failed,
        "hard_guard_failed": hard_guard_failed,
        "regression_failed": max(
            0,
            len(promotion_evaluations)
            - canary_failed
            - hard_guard_failed
            - accepted,
        ),
        "runtime_ab_accepted": accepted,
    }


def candidate_repair_target_txs(
    viability: Dict[str, Any] | None,
) -> List[str]:
    """Derive candidate repair targets from the canonical per-case audit."""
    return list(dict.fromkeys(
        str(case.get("tx_hash") or "").strip().lower()
        for case in list((viability or {}).get("case_audits") or [])
        if isinstance(case, dict)
        and bool(case.get("potentially_fixable"))
        and str(case.get("tx_hash") or "").strip()
    ))


def _source_review_tx_by_index(
    review_bundle: Dict[str, Any] | None,
) -> Dict[int, str]:
    """Resolve case lineage without exposing transaction hashes to Updaters."""
    bundle = review_bundle or {}
    resolved: Dict[int, str] = {}

    def add_review(review: Any, *, fallback_index: int | None = None) -> None:
        if not isinstance(review, dict):
            return
        raw_index = review.get("source_review_index", fallback_index)
        try:
            source_index = int(raw_index)
        except (TypeError, ValueError):
            return
        tx_hash = str(review.get("tx_hash") or "").strip().lower()
        if source_index >= 0 and tx_hash:
            resolved.setdefault(source_index, tx_hash)

    for review_index, review in enumerate(list(bundle.get("reviews") or [])):
        add_review(review, fallback_index=review_index)
    owner_projections = bundle.get("owner_projections")
    if isinstance(owner_projections, dict):
        for projections in owner_projections.values():
            for review in list(projections or []):
                add_review(review)
    for key in (
        "actionable_rule_reviews",
        "actionable_plan_reviews",
        "engineering_reviews",
        "non_rule_reviews",
    ):
        for review in list(bundle.get(key) or []):
            add_review(review)
    return resolved


def filter_candidate_specs_by_viability(
    specs: List[Dict[str, Any]],
    *,
    current_plan: EvidencePlan | None,
    current_rule: EvolvingRule,
    current_slim_results: List[Dict[str, Any]],
    review_bundle: Dict[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep structurally safe, attributable, executable candidates.

    Runtime effectiveness belongs to canary evaluation. This preflight does not
    predict whether a candidate will clear every decisive blocker.
    """
    accepted: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    signal_by_id = {
        str(signal.get("signal_id") or ""): dict(signal)
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict) and str(signal.get("signal_id") or "")
    }
    source_review_tx_by_index = _source_review_tx_by_index(review_bundle)
    canonical_contract = bool(
        (review_bundle or {}).get("canonical_signal_contract")
    )
    for spec in list(specs or []):
        update_kind = str(spec.get("update_kind") or "").strip().lower()
        selected_atomic_deltas = [
            dict(item) for item in list(spec.get("selected_atomic_deltas") or [])
            if isinstance(item, dict)
        ]
        changed_ids = _candidate_changed_condition_ids(
            spec,
            current_plan=current_plan,
            current_rule=current_rule,
        )
        rule_changed_ids = _candidate_rule_changed_condition_ids(
            spec,
            current_rule=current_rule,
        )
        candidate_plan = (
            spec.get("plan")
            if isinstance(spec.get("plan"), EvidencePlan)
            else current_plan
        )
        plan_activation = plan_route_activation_audit(
            current_plan,
            candidate_plan,
        )
        plan_execution_ids = {
            str(item).strip().upper()
            for item in list(
                plan_activation.get("execution_capable_condition_ids") or []
            )
            if str(item).strip()
        }
        optional_plan_ids = {
            str(item).strip().upper()
            for item in list(
                plan_activation.get("optional_followup_only_condition_ids") or []
            )
            if str(item).strip()
        }
        candidate_steps_by_id = {
            str(step.condition_id or step.id or "").strip().upper(): step
            for step in list(
                candidate_plan.judge_steps
                if isinstance(candidate_plan, EvidencePlan)
                else []
            )
        }
        # An added optional route is structurally runnable when the target step
        # can issue a packet-view follow-up. Whether the Judge actually selects
        # that route is a runtime attribution question, not a viability gate.
        runtime_activation_required_ids = {
            condition_id
            for condition_id in optional_plan_ids
            for step in [candidate_steps_by_id.get(condition_id)]
            if step is not None
            and int(step.max_followups or 0) > 0
            and "read_packet_view" in set(step.allowed_tools or [])
        }
        runnable_plan_ids = plan_execution_ids | runtime_activation_required_ids
        executable_changed_ids = set(rule_changed_ids) | runnable_plan_ids
        declared_changed_ids = _candidate_explicit_changed_condition_ids(spec)
        dependency_plan = (
            spec.get("plan")
            if isinstance(spec.get("plan"), EvidencePlan)
            else current_plan
        )
        dependent_ids = _stateful_downstream_condition_ids(
            dependency_plan,
            executable_changed_ids,
        )
        effective_changed_ids = set(executable_changed_ids) | set(dependent_ids)
        explicit_evidence_dependencies = [
            dict(item)
            for item in list(spec.get("condition_evidence_dependencies") or [])
            if isinstance(item, dict)
            and bool(item.get("explicit_dependency"))
            and str(item.get("condition_id") or "").strip()
            and str(item.get("capability_id") or "").strip()
        ]
        applied_signal_ids = {
            str(value)
            for value in [
                *list(spec.get("applied_plan_signal_ids") or []),
                *list(spec.get("experimental_signal_ids") or []),
                *list(spec.get("hypothesis_source_signal_ids") or []),
                *[
                    signal_id
                    for delta in selected_atomic_deltas
                    for signal_id in list(delta.get("applied_signal_ids") or [])
                ],
            ]
            if str(value)
        }
        source_signal_review_indexes = {
            int(signal_by_id[signal_id]["source_review_index"])
            for signal_id in applied_signal_ids
            if signal_id in signal_by_id
            and str(
                signal_by_id[signal_id].get("source_review_index", "")
            ).lstrip("-").isdigit()
        }
        source_signal_txs = {
            source_review_tx_by_index[source_review_index]
            for source_review_index in source_signal_review_indexes
            if source_review_index in source_review_tx_by_index
        }
        source_signal_conditions_by_tx: Dict[str, set[str]] = {}
        for signal_id in applied_signal_ids:
            signal = signal_by_id.get(signal_id, {})
            try:
                source_review_index = int(signal.get("source_review_index", -1))
            except (TypeError, ValueError):
                continue
            source_tx = source_review_tx_by_index.get(source_review_index, "")
            condition_id = str(signal.get("condition_id") or "").strip().upper()
            if source_tx and condition_id:
                source_signal_conditions_by_tx.setdefault(source_tx, set()).add(
                    condition_id
                )
        rejection_details: List[Dict[str, Any]] = []
        if update_kind not in {"rule", "plan", "rule_plan"}:
            rejection_details.append({
                "code": "unsupported_candidate_update_kind",
                "update_kind": update_kind,
            })
        if not changed_ids:
            rejection_details.append({"code": "semantic_candidate_noop"})

        plan_target_ids = {
            str(value or "").strip().upper()
            for value in list(spec.get("applied_plan_patch_conditions") or [])
            if re.fullmatch(r"[CE]\d+", str(value or "").strip().upper())
        }
        if update_kind in {"plan", "rule_plan"}:
            if not isinstance(candidate_plan, EvidencePlan):
                rejection_details.append({"code": "missing_candidate_plan"})
            elif plan_target_ids and not plan_target_ids.issubset(runnable_plan_ids):
                rejection_details.append({
                    "code": "plan_route_not_execution_capable",
                    "target_condition_ids": sorted(plan_target_ids),
                    "execution_capable_condition_ids": sorted(plan_execution_ids),
                    "runtime_activation_required_condition_ids": sorted(
                        runtime_activation_required_ids
                    ),
                })
            actual_plan_changed_ids = set(changed_ids) - set(rule_changed_ids)
            if (
                actual_plan_changed_ids
                and not actual_plan_changed_ids.issubset(runnable_plan_ids)
            ):
                rejection_details.append({
                    "code": "plan_route_not_execution_capable",
                    "target_condition_ids": sorted(actual_plan_changed_ids),
                    "execution_capable_condition_ids": sorted(plan_execution_ids),
                    "runtime_activation_required_condition_ids": sorted(
                        runtime_activation_required_ids
                    ),
                })

        incomplete_dependencies = []
        for dependency in explicit_evidence_dependencies:
            semantic_satisfied = bool(
                dependency.get("semantic_requirement_satisfied")
            )
            evidence_satisfied = bool(
                dependency.get("evidence_accessibility_satisfied")
            )
            if (
                update_kind == "rule_plan"
                and not (semantic_satisfied and evidence_satisfied)
            ) or (
                update_kind == "rule"
                and semantic_satisfied
                and not evidence_satisfied
            ):
                incomplete_dependencies.append({
                    **dependency,
                    "semantic_requirement_satisfied": semantic_satisfied,
                    "evidence_accessibility_satisfied": evidence_satisfied,
                })
        if incomplete_dependencies:
            rejection_details.append({
                "code": "explicit_condition_evidence_dependency_incomplete",
                "dependencies": incomplete_dependencies,
            })

        dependency_preflight = dict(
            spec.get("pre_viability_dependency_validation") or {}
        )
        if (
            dependency_preflight
            and str(dependency_preflight.get("dependency_validation_status") or "")
            != "passed"
        ):
            rejection_details.append({
                "code": "dependency_or_binding_invalid",
                "reason": str(
                    dependency_preflight.get("dependency_rejection_reason") or ""
                ),
            })

        if canonical_contract:
            missing_signal_ids = sorted(applied_signal_ids - set(signal_by_id))
            if missing_signal_ids:
                rejection_details.append({
                    "code": "candidate_signal_lineage_missing",
                    "signal_ids": missing_signal_ids,
                })
            deterministic_restore = bool(
                isinstance(candidate_plan, EvidencePlan)
                and dict(candidate_plan.metadata or {}).get(
                    "deterministic_question_restore"
                )
            )
            if not applied_signal_ids and not deterministic_restore:
                rejection_details.append({
                    "code": "candidate_has_no_attributed_signal_lineage"
                })
            if applied_signal_ids:
                allowed_statuses = (
                    {"experimental_rule", "experimental_plan"}
                    if bool(spec.get("probe_only"))
                    and str(spec.get("candidate_strategy") or "")
                    != "evidence_activation_probe"
                    else {"supported"}
                )
                wrong_lifecycle_ids = sorted(
                    signal_id
                    for signal_id in applied_signal_ids
                    if signal_id in signal_by_id
                    and str(
                        signal_by_id[signal_id].get("generalization_status") or ""
                    ).strip().lower() not in allowed_statuses
                )
                if wrong_lifecycle_ids:
                    rejection_details.append({
                        "code": "candidate_signal_lifecycle_mismatch",
                        "signal_ids": wrong_lifecycle_ids,
                        "allowed_statuses": sorted(allowed_statuses),
                    })

        case_audits: List[Dict[str, Any]] = []
        for result in list(current_slim_results or []):
            ground_truth = str(result.get("ground_truth") or "").strip().lower()
            predicted = str(result.get("predicted_verdict") or "").strip().lower()
            normalized_tx_hash = str(result.get("tx_hash") or "").strip().lower()
            is_error = (
                ground_truth == "attack" and predicted != "attack"
            ) or (
                ground_truth != "attack" and predicted == "attack"
            )
            reviewed_uncertain = bool(
                predicted == "uncertain"
                and normalized_tx_hash in source_signal_conditions_by_tx
            )
            if not is_error and not reviewed_uncertain:
                continue
            tx_hash = str(result.get("tx_hash") or "")
            conditions = list(result.get("condition_table", []) or [])
            condition_ids = {
                str(item.get("condition_id") or item.get("id") or "").strip().upper()
                for item in conditions
                if isinstance(item, dict)
            }
            blockers = (
                set(source_signal_conditions_by_tx.get(normalized_tx_hash, set()))
                if reviewed_uncertain
                else
                {
                    str(item.get("condition_id") or item.get("id") or "").strip().upper()
                    for item in conditions
                    if isinstance(item, dict)
                    and (
                        (
                            not bool(item.get("is_exclusion"))
                            and not _condition_answer_is_true(item.get("answer"))
                        )
                        or (
                            bool(item.get("is_exclusion"))
                            and _condition_answer_is_true(item.get("answer"))
                        )
                    )
                }
                if ground_truth == "attack"
                else condition_ids
            )
            covered = blockers & effective_changed_ids
            source_lineage_match = (
                normalized_tx_hash in source_signal_txs
                if source_signal_txs
                else True
            )
            case_audits.append({
                "tx_hash": tx_hash,
                "ground_truth": ground_truth,
                "predicted_verdict": predicted,
                "case_repair_direction": (
                    "uncertain_resolution"
                    if reviewed_uncertain
                    else "recall"
                    if ground_truth == "attack"
                    else "precision"
                ),
                "reviewed_uncertain_source": reviewed_uncertain,
                "blocking_condition_ids": sorted(blockers),
                "covered_blocking_condition_ids": sorted(blockers & changed_ids),
                "dependency_covered_blocking_condition_ids": sorted(
                    blockers & dependent_ids
                ),
                "effective_covered_blocking_condition_ids": sorted(
                    blockers & effective_changed_ids
                ),
                "uncovered_blocking_condition_ids": sorted(
                    blockers - effective_changed_ids
                ),
                "coverage_status": (
                    "all_decisive_blockers_covered"
                    if blockers and blockers.issubset(effective_changed_ids)
                    else "partial_blocker_overlap"
                    if covered
                    else "no_current_blocker_overlap"
                ),
                "source_signal_match": source_lineage_match,
                "source_signal_tx_hashes": sorted(source_signal_txs),
                "potentially_fixable": bool(covered and source_lineage_match),
            })
        repair_target_txs = candidate_repair_target_txs({
            "case_audits": case_audits,
        })
        if bool(spec.get("probe_only")) and not repair_target_txs:
            rejection_details.append({
                "code": "activation_probe_missing_repair_target",
                "reason": (
                    "probe candidate has no potentially fixable current error "
                    "derived from the canonical case audits"
                ),
            })
        viable = not rejection_details
        error_directions = {
            str(signal_by_id.get(signal_id, {}).get("source_error_type") or "")
            .strip().upper()
            for signal_id in applied_signal_ids
        }
        if not error_directions:
            for result in list(current_slim_results or []):
                ground_truth = str(result.get("ground_truth") or "").strip().lower()
                predicted = str(result.get("predicted_verdict") or "").strip().lower()
                if ground_truth == "attack" and predicted != "attack":
                    error_directions.add("FN")
                elif ground_truth != "attack" and predicted == "attack":
                    error_directions.add("FP")
        candidate_repair_direction = (
            "mixed"
            if {"FP", "FN"}.issubset(error_directions)
            else "recall_repair"
            if "FN" in error_directions
            else "precision_tightening"
            if "FP" in error_directions
            else "none"
        )
        audit = {
            "candidate_name": str(spec.get("name") or ""),
            "update_kind": update_kind,
            "changed_condition_ids": sorted(changed_ids),
            "rule_changed_condition_ids": sorted(rule_changed_ids),
            "plan_execution_condition_ids": sorted(plan_execution_ids),
            "runtime_activation_required_condition_ids": sorted(
                runtime_activation_required_ids
            ),
            "executable_changed_condition_ids": sorted(executable_changed_ids),
            "plan_route_activation_audit": plan_activation,
            "declared_changed_condition_ids": sorted(declared_changed_ids),
            "declared_only_condition_ids": sorted(
                declared_changed_ids - changed_ids
            ),
            "actual_only_condition_ids": sorted(
                changed_ids - declared_changed_ids
            ),
            "stateful_dependent_condition_ids": sorted(dependent_ids),
            "effective_changed_condition_ids": sorted(effective_changed_ids),
            "candidate_repair_direction": candidate_repair_direction,
            "applied_signal_ids": sorted(applied_signal_ids),
            "selected_atomic_deltas": selected_atomic_deltas,
            "condition_evidence_dependencies": explicit_evidence_dependencies,
            "case_audits": case_audits,
            "repair_target_txs": repair_target_txs,
            "pre_judge_rejections": rejection_details,
            "viable": viable,
            "reason": (
                "candidate_is_attributable_and_executable"
                if viable
                else "candidate_not_executable"
            ),
        }
        audits.append(audit)
        if viable:
            spec = dict(spec)
            spec["candidate_viability"] = audit
            accepted.append(spec)
        else:
            print(
                "[FewShot] Skipping non-viable candidate before evaluation: "
                f"{stable_json_dumps(audit)}"
            )
    return accepted, {
        "schema_version": "evotx.candidate_viability.v1",
        "candidate_count": len(list(specs or [])),
        "viable_candidate_count": len(accepted),
        "rejected_candidate_count": len(list(specs or [])) - len(accepted),
        "candidates": audits,
    }


def _actionable_review_signal_condition_ids(
    review_bundle: Dict[str, Any] | None,
    *,
    update_kind: str,
) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {"recall": set(), "precision": set()}
    wanted_targets = (
        {"rule", "plan"}
        if str(update_kind or "").strip().lower() == "rule_plan"
        else {str(update_kind or "").strip().lower()}
    )
    for review in _reviews_from_bundle(review_bundle or {}):
        error_type = str(review.get("error_type") or "").strip().upper()
        direction = "recall" if error_type == "FN" else "precision" if error_type == "FP" else ""
        if not direction:
            continue
        for signal in list(review.get("update_signals") or []):
            if not isinstance(signal, dict):
                continue
            status = str(
                signal.get("generalization_status") or ""
            ).strip().lower()
            allowed_statuses = {"supported"}
            if "plan" in wanted_targets:
                allowed_statuses.add("experimental_plan")
            if "rule" in wanted_targets:
                allowed_statuses.add("experimental_rule")
            if status not in allowed_statuses:
                continue
            signal_target = str(
                signal.get("update_target") or review.get("update_target") or ""
            ).strip().lower()
            if signal_target not in wanted_targets:
                continue
            condition_id = str(signal.get("condition_id") or "").strip().upper()
            if re.fullmatch(r"[CE]\d+", condition_id):
                out[direction].add(condition_id)
    return out


def _do_not_train_review_txs(review_bundle: Dict[str, Any] | None) -> set[str]:
    out: set[str] = set()
    for review in _reviews_from_bundle(review_bundle or {}):
        if not bool(review.get("do_not_train")):
            continue
        tx_hash = str(review.get("tx_hash") or "").strip().lower()
        if tx_hash:
            out.add(tx_hash)
    return out


def _current_error_profile(
    current_slim_results: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    profile = {
        "recall_error_count": 0,
        "precision_error_count": 0,
        "uncertain_attack_count": 0,
        "uncertain_benign_count": 0,
        "error_count": 0,
        "precision_only": False,
        "txs": {
            "recall": [],
            "precision": [],
            "uncertain_attack": [],
            "uncertain_benign": [],
        },
    }
    txs = profile["txs"]
    for result in list(current_slim_results or []):
        ground_truth = str(result.get("ground_truth") or "").strip().lower()
        predicted = str(result.get("predicted_verdict") or "").strip().lower()
        tx_hash = str(result.get("tx_hash") or "")
        if ground_truth == "attack":
            if predicted == "attack":
                continue
            if predicted == "uncertain":
                profile["uncertain_attack_count"] += 1
                txs["uncertain_attack"].append(tx_hash)
            else:
                profile["recall_error_count"] += 1
                txs["recall"].append(tx_hash)
        else:
            if predicted == "attack":
                profile["precision_error_count"] += 1
                txs["precision"].append(tx_hash)
            elif predicted == "uncertain":
                profile["uncertain_benign_count"] += 1
                txs["uncertain_benign"].append(tx_hash)
    profile["error_count"] = (
        int(profile["recall_error_count"])
        + int(profile["precision_error_count"])
        + int(profile["uncertain_attack_count"])
        + int(profile["uncertain_benign_count"])
    )
    profile["precision_only"] = (
        int(profile["precision_error_count"]) > 0
        and int(profile["recall_error_count"]) == 0
        and int(profile["uncertain_attack_count"]) == 0
    )
    return profile


def _actionable_update_signals(
    review_bundle: Dict[str, Any] | None,
    *,
    update_kind: str,
) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    wanted_targets = (
        {"rule", "plan"}
        if str(update_kind or "").strip().lower() == "rule_plan"
        else {str(update_kind or "").strip().lower()}
    )
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    for raw in list(matrix.get("signals") or []):
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("generalization_status") or "").strip().lower()
        allowed_statuses = {"supported"}
        if "plan" in wanted_targets:
            allowed_statuses.add("experimental_plan")
        if "rule" in wanted_targets:
            allowed_statuses.add("experimental_rule")
        if status not in allowed_statuses:
            continue
        if str(raw.get("update_target") or "").strip().lower() not in wanted_targets:
            continue
        signals.append(dict(raw))
    return signals


def _signal_strategy_operation(signal: Dict[str, Any]) -> str:
    return str(
        signal.get("strategy_operation")
        or signal.get("operation")
        or ""
    ).strip().lower()


def _signal_capability_id(signal: Dict[str, Any]) -> str:
    dependency = signal.get("condition_evidence_dependency")
    if not isinstance(dependency, dict):
        return ""
    capability_id = str(dependency.get("capability_id") or "").strip().lower()
    role = str(dependency.get("role") or "").strip().lower()
    if role == "none":
        return ""
    return capability_id


def _condition_answer_is_true(value: Any) -> bool:
    return value is True or str(value or "").strip().lower() == "true"


def _candidate_changed_condition_ids(
    spec: Dict[str, Any],
    *,
    current_plan: EvidencePlan | None,
    current_rule: EvolvingRule,
) -> set[str]:
    """Return condition ids whose executable Rule/Plan payload actually changed.

    Updater metadata is useful for attribution auditing, but it is not evidence
    of a semantic diff. In particular, regenerated or statefully invalidated
    downstream steps must not be reported as direct patches unless their Rule
    condition or Judge-step policy differs from the accepted artifact.
    """
    changed: set[str] = set()
    explicit_ids = _candidate_explicit_changed_condition_ids(spec)
    comparable_artifact_found = False
    candidate_rule = spec.get("rule")
    if isinstance(candidate_rule, EvolvingRule):
        comparable_artifact_found = True
        old_signatures = _rule_condition_signature_by_id(current_rule)
        new_signatures = _rule_condition_signature_by_id(candidate_rule)
        changed.update(
            condition_id
            for condition_id in set(old_signatures) | set(new_signatures)
            if old_signatures.get(condition_id) != new_signatures.get(condition_id)
        )
    candidate_plan = spec.get("plan")
    if (
        current_plan is not None
        and isinstance(candidate_plan, EvidencePlan)
    ):
        comparable_artifact_found = True
        old_steps = {
            _judge_step_condition_key(step.to_dict()): stable_json_dumps(
                _judge_step_policy_payload(step.to_dict())
            )
            for step in list(current_plan.judge_steps or [])
            if _judge_step_condition_key(step.to_dict())
        }
        new_steps = {
            _judge_step_condition_key(step.to_dict()): stable_json_dumps(
                _judge_step_policy_payload(step.to_dict())
            )
            for step in list(candidate_plan.judge_steps or [])
            if _judge_step_condition_key(step.to_dict())
        }
        changed.update(
            condition_id
            for condition_id in set(old_steps) | set(new_steps)
            if old_steps.get(condition_id) != new_steps.get(condition_id)
        )
    if not comparable_artifact_found:
        changed.update(explicit_ids)
    return changed


def _candidate_rule_changed_condition_ids(
    spec: Dict[str, Any],
    *,
    current_rule: EvolvingRule,
) -> set[str]:
    candidate_rule = spec.get("rule")
    if not isinstance(candidate_rule, EvolvingRule):
        return set()
    old_signatures = _rule_condition_signature_by_id(current_rule)
    new_signatures = _rule_condition_signature_by_id(candidate_rule)
    return {
        condition_id
        for condition_id in set(old_signatures) | set(new_signatures)
        if old_signatures.get(condition_id) != new_signatures.get(condition_id)
    }


def _candidate_explicit_changed_condition_ids(spec: Dict[str, Any]) -> set[str]:
    explicit: set[str] = set()
    for key in (
        "applied_rule_patch_conditions",
        "applied_plan_patch_conditions",
    ):
        explicit.update(
            str(item).strip().upper()
            for item in list((spec or {}).get(key) or [])
            if str(item).strip()
        )
    if not explicit:
        plan = (spec or {}).get("plan")
        metadata = dict((plan.metadata if isinstance(plan, EvidencePlan) else {}) or {})
        for key in (
            "changed_rule_condition_ids",
            "explicit_plan_patch_condition_ids",
            "regenerated_plan_step_condition_ids",
        ):
            explicit.update(
                str(item).strip().upper()
                for item in list(metadata.get(key) or [])
                if str(item).strip()
            )
    return {item for item in explicit if re.fullmatch(r"[CE]\d+", item)}


def _stateful_downstream_condition_ids(
    plan: EvidencePlan | Dict[str, Any] | None,
    changed_ids: set[str],
) -> set[str]:
    if not changed_ids:
        return set()
    edges = _stateful_dependency_edges(plan)
    if not edges:
        return set()
    changed = {str(item).strip().upper() for item in changed_ids if str(item)}
    closure = set(changed)
    progressed = True
    while progressed:
        progressed = False
        for target, dependencies in edges.items():
            if target in closure:
                continue
            if dependencies & closure:
                closure.add(target)
                progressed = True
    return closure - changed


def _stateful_dependency_edges(
    plan: EvidencePlan | Dict[str, Any] | None,
) -> Dict[str, set[str]]:
    """Return target_condition -> dependency_condition ids.

    Plans may carry dependencies in metadata, explicit depends_on fields, or
    state key producer/consumer wiring. Candidate reuse must respect all three.
    """
    if plan is None:
        return {}
    data = plan.to_dict() if isinstance(plan, EvidencePlan) else dict(plan or {})
    steps = [
        step for step in list(data.get("judge_steps", []) or [])
        if isinstance(step, dict)
    ]
    edges: Dict[str, set[str]] = {}

    metadata = dict(data.get("metadata") or {})
    stateful = dict(metadata.get("stateful_runtime") or {})
    raw_edges = dict(stateful.get("dependency_edges") or {})
    for target, dependencies in raw_edges.items():
        target_id = str(target or "").strip().upper()
        if not target_id:
            continue
        edges.setdefault(target_id, set()).update(
            str(dep or "").strip().upper()
            for dep in list(dependencies or [])
            if str(dep or "").strip()
        )

    producer_by_key: Dict[str, str] = {}
    for step in steps:
        condition_id = str(
            step.get("condition_id") or step.get("id") or ""
        ).strip().upper()
        produced = str(step.get("produces_state_key") or "").strip()
        if condition_id and produced:
            producer_by_key[produced] = condition_id

    for step in steps:
        condition_id = str(
            step.get("condition_id") or step.get("id") or ""
        ).strip().upper()
        if not condition_id:
            continue
        for dep in list(step.get("depends_on") or []):
            dep_id = str(dep or "").strip().upper()
            if dep_id:
                edges.setdefault(condition_id, set()).add(dep_id)
        for state_key in list(step.get("consumes_state_keys") or []):
            dep_id = producer_by_key.get(str(state_key or "").strip())
            if dep_id and dep_id != condition_id:
                edges.setdefault(condition_id, set()).add(dep_id)

    return {
        target: {dep for dep in dependencies if dep and dep != target}
        for target, dependencies in edges.items()
        if target and dependencies
    }


def _candidate_generation_priority(
    review_bundle: Dict[str, Any],
    targets: Dict[str, int],
) -> Dict[str, Any]:
    error_summary = dict((review_bundle or {}).get("error_summary") or {})
    current_summary = dict((review_bundle or {}).get("current_summary") or {})
    plan_only_targets = max(0, int(targets.get("plan", 0) or 0) - int(targets.get("mixed_rule_plan", 0) or 0))
    rule_actionable = int(
        error_summary.get("rule_actionable", targets.get("rule", 0)) or 0
    )
    plan_actionable = int(
        error_summary.get("plan_actionable", plan_only_targets) or 0
    )
    errors = int(
        current_summary.get("errors", error_summary.get("total_reviews", 0)) or 0
    )
    fn = int(current_summary.get("fn", error_summary.get("fn", 0)) or 0)
    fp = int(current_summary.get("fp", error_summary.get("fp", 0)) or 0)
    primary = "balanced"
    reason = "balanced_or_missing_priority_signal"
    if errors >= 2 and rule_actionable > plan_actionable:
        primary = "rule"
        reason = "rule_actionable_reviews_dominate"
    elif errors >= 2 and fn > 0 and rule_actionable >= plan_actionable:
        primary = "rule"
        reason = "fn_present_and_rule_not_less_actionable"
    elif plan_actionable > rule_actionable:
        primary = "plan"
        reason = "plan_actionable_reviews_dominate"
    return {
        "primary": primary,
        "reason": reason,
        "errors": errors,
        "fn": fn,
        "fp": fp,
        "rule_actionable": rule_actionable,
        "plan_actionable": plan_actionable,
        "mixed_rule_plan": int(targets.get("mixed_rule_plan", 0) or 0),
        "plan_only_targets": plan_only_targets,
    }


def _review_patch_condition_ids(
    review_bundle: Dict[str, Any],
    *,
    patch_key: str,
    include_rule_diagnosis: bool = False,
    include_plan_diagnosis: bool = False,
) -> List[str]:
    ids: set[str] = set()
    for review in _reviews_from_bundle(review_bundle):
        suggestion = review.get(patch_key, {}) or {}
        if isinstance(suggestion, dict):
            condition_id = str(suggestion.get("condition_id") or "").strip().upper()
            if condition_id:
                ids.add(condition_id)
        for diagnosis in list(review.get("condition_diagnosis", []) or []):
            if not isinstance(diagnosis, dict):
                continue
            if include_rule_diagnosis and diagnosis.get("is_rule_problem"):
                condition_id = str(diagnosis.get("condition_id") or "").strip().upper()
                if condition_id:
                    ids.add(condition_id)
            if include_plan_diagnosis and diagnosis.get("is_plan_problem"):
                condition_id = str(diagnosis.get("condition_id") or "").strip().upper()
                if condition_id:
                    ids.add(condition_id)
    return sorted(item for item in ids if re.fullmatch(r"[CE]\d+", item))


def _plan_changed(updated: EvidencePlan, base: EvidencePlan) -> bool:
    return semantic_plan_fingerprint(updated) != semantic_plan_fingerprint(base)


def build_review_input_fingerprint(
    *,
    current_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
    current_slim_results: List[Dict[str, Any]],
    current_guard_slim_results: List[Dict[str, Any]],
    cohort_signal_summary: Dict[str, Any],
    plan_evidence_audit: Dict[str, Any],
    case_boundary_context: Dict[str, Any],
    rejected_update_memory: Dict[str, Any] | None = None,
) -> str:
    """Fingerprint every input that may change the Reviewer's next proposal."""
    payload = {
        "rule": current_rule.to_dict(),
        "plan_fingerprint": (
            semantic_plan_fingerprint(current_plan) if current_plan is not None else ""
        ),
        "current_results": current_slim_results,
        "guard_results": current_guard_slim_results,
        "cohort_signal_summary": cohort_signal_summary,
        "plan_evidence_audit": plan_evidence_audit,
        "case_boundary_context": case_boundary_context,
        "rejected_update_memory": normalize_rejected_update_memory(
            rejected_update_memory or {}
        ),
    }
    return hashlib.sha256(
        stable_json_dumps(payload).encode("utf-8")
    ).hexdigest()


def artifact_hash(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return hashlib.sha256(
        stable_json_dumps(value).encode("utf-8")
    ).hexdigest()[:16]


def _semantic_rule_payload(rule: EvolvingRule | Dict[str, Any] | None) -> Dict[str, Any]:
    if isinstance(rule, EvolvingRule):
        payload = rule.to_dict()
    else:
        payload = dict(rule or {})
    return {
        "description": " ".join(str(payload.get("description") or "").split()),
        "conditions": [
            {
                "id": str(item.get("id") or "").strip().upper(),
                "description": " ".join(
                    str(item.get("description") or "").split()
                ),
                "expected_answer": bool(item.get("expected_answer", True)),
            }
            for item in list(payload.get("conditions") or [])
            if isinstance(item, dict)
        ],
        "exclusion_conditions": [
            {
                "id": str(item.get("id") or "").strip().upper(),
                "description": " ".join(
                    str(item.get("description") or "").split()
                ),
                "expected_answer": bool(item.get("expected_answer", True)),
            }
            for item in list(payload.get("exclusion_conditions") or [])
            if isinstance(item, dict)
        ],
        "decision_policy": " ".join(
            str(payload.get("decision_policy") or "").split()
        ),
    }


def minimal_candidate_semantic_fingerprint(
    spec: Dict[str, Any],
    *,
    fallback_plan: EvidencePlan | None,
) -> str:
    plan = spec.get("plan")
    if not isinstance(plan, EvidencePlan):
        plan = spec.get("base_plan")
    if not isinstance(plan, EvidencePlan):
        plan = fallback_plan
    plan_payload = semantic_plan_payload(plan) if isinstance(plan, EvidencePlan) else {}
    # Rule linkage/version fields are archival identity, not execution semantics.
    plan_payload.pop("rule_id", None)
    plan_payload.pop("rule_version", None)
    if isinstance(plan, EvidencePlan):
        metadata = dict(plan.metadata or {})
        runtime_metadata_keys = (
            "stateful_runtime",
            "access_control_binding_policy",
            "token_semantic_stateful_runtime",
            "protocol_accounting_stateful_runtime",
            "market_manipulation_stateful_runtime",
            "flashloans_stateful_runtime",
            "reentrancy_stateful_runtime",
        )
        plan_payload["runtime_metadata"] = {
            key: metadata[key]
            for key in runtime_metadata_keys
            if key in metadata
        }
    payload = {
        "rule": _semantic_rule_payload(spec.get("rule")),
        "plan": plan_payload,
    }
    return hashlib.sha256(
        stable_json_dumps(payload).encode("utf-8")
    ).hexdigest()


def _candidate_spec_signal_ids(spec: Dict[str, Any]) -> List[str]:
    return sorted({
        str(signal_id)
        for signal_id in [
            *list(spec.get("applied_plan_signal_ids") or []),
            *[
                value
                for delta in list(spec.get("selected_atomic_deltas") or [])
                if isinstance(delta, dict)
                for value in list(delta.get("applied_signal_ids") or [])
            ],
        ]
        if str(signal_id)
    })


def dedupe_minimal_candidate_specs(
    specs: List[Dict[str, Any]],
    *,
    fallback_plan: EvidencePlan | None,
    seen_fingerprints: set[str],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    for spec in list(specs or []):
        fingerprint = minimal_candidate_semantic_fingerprint(
            spec,
            fallback_plan=fallback_plan,
        )
        spec["semantic_candidate_fingerprint"] = fingerprint
        if fingerprint in seen_fingerprints:
            spec["exact_candidate_duplicate"] = True
            duplicates.append({
                "candidate_name": str(spec.get("name") or ""),
                "semantic_candidate_fingerprint": fingerprint,
                "signal_ids": _candidate_spec_signal_ids(spec),
                "reason": "exact_rule_plan_semantic_duplicate",
            })
            continue
        seen_fingerprints.add(fingerprint)
        kept.append(spec)
    return kept, {
        "schema_version": "evotx.minimal_candidate_dedupe.v1",
        "enabled": True,
        "policy": "skip_only_exact_rule_plan_execution_semantic_duplicates",
        "input_candidate_count": len(list(specs or [])),
        "executed_candidate_names": [str(spec.get("name") or "") for spec in kept],
        "exact_duplicates": duplicates,
    }


def incumbent_artifact_matches_current(
    *,
    current_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
    incumbent_rule_hash: str,
    incumbent_plan_hash: str,
    incumbent_full_results: List[Dict[str, Any]] | None,
) -> bool:
    """Return whether cached incumbent results describe the current artifact."""
    if incumbent_full_results is None or not incumbent_rule_hash:
        return False
    return (
        artifact_hash(current_rule) == incumbent_rule_hash
        and artifact_hash(current_plan) == incumbent_plan_hash
    )


def _result_is_correct(result: Dict[str, Any] | None) -> bool:
    slim = build_slim_result(result or {})
    ground_truth = str(slim.get("ground_truth") or "").strip().lower()
    predicted = str(slim.get("predicted_verdict") or "").strip().lower()
    if not ground_truth or not predicted:
        return False
    if ground_truth == "attack":
        return predicted == "attack"
    return predicted not in {"attack", "uncertain", ""}


def _result_is_repair_improvement(
    baseline: Dict[str, Any] | None,
    candidate: Dict[str, Any] | None,
) -> bool:
    """Return true for a full fix or the existing bounded FP-to-uncertain step."""
    if _result_is_correct(candidate):
        return True
    baseline_slim = build_slim_result(baseline or {})
    candidate_slim = build_slim_result(candidate or {})
    ground_truth = str(
        candidate_slim.get("ground_truth")
        or baseline_slim.get("ground_truth")
        or ""
    ).strip().lower()
    old_verdict = str(
        baseline_slim.get("predicted_verdict") or ""
    ).strip().lower()
    new_verdict = str(
        candidate_slim.get("predicted_verdict") or ""
    ).strip().lower()
    return (
        ground_truth == "benign"
        and old_verdict == "attack"
        and new_verdict == "uncertain"
    )


def _result_verdict(result: Dict[str, Any] | None) -> str:
    return str(
        build_slim_result(result or {}).get("predicted_verdict") or ""
    ).strip().lower()


def _condition_answer_map(result: Dict[str, Any] | None) -> Dict[str, str]:
    slim = build_slim_result(result or {})
    out: Dict[str, str] = {}
    for row in list(slim.get("condition_table") or []):
        if not isinstance(row, dict):
            continue
        condition_id = str(
            row.get("condition_id") or row.get("id") or row.get("condition") or ""
        ).strip().upper()
        if not condition_id:
            continue
        raw_answer: Any = ""
        for key in ("answer", "result", "verdict", "status"):
            value = row.get(key)
            if value is not None and value != "":
                raw_answer = value
                break
        answer = str(raw_answer).strip().lower()
        if answer:
            out[condition_id] = answer
    return out


def _condition_flips(
    baseline_result: Dict[str, Any] | None,
    candidate_result: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    baseline = _condition_answer_map(baseline_result)
    candidate = _condition_answer_map(candidate_result)
    flips: List[Dict[str, Any]] = []
    for condition_id in sorted(set(baseline) | set(candidate)):
        before = baseline.get(condition_id, "")
        after = candidate.get(condition_id, "")
        if before == after:
            continue
        flips.append({
            "condition_id": condition_id,
            "baseline_answer": before,
            "candidate_answer": after,
        })
    return flips


def _guard_regression_rows(comparison: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    guard_comparison = dict(
        (comparison or {}).get("guard_comparison") or {}
    )
    rows: List[Dict[str, Any]] = []
    for raw in list(guard_comparison.get("regressed_cases") or []):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row.setdefault("role", "hard_negative_guard")
        rows.append(row)
    return rows


def _candidate_refinement_feedback(
    item: Dict[str, Any],
    *,
    round_index: int,
    baseline_full_results: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    comparison = dict(item.get("comparison") or {})
    update_kind = str(item.get("update_kind") or "").strip().lower()
    reject_reason = (
        str(comparison.get("reject_reason") or "")
        or str((comparison.get("guard_gate") or {}).get("reject_reason") or "")
        or "candidate_rejected"
    )
    plan_effect_activation = dict(
        comparison.get("plan_effect_activation") or {}
    )
    canary = dict(comparison.get("candidate_canary") or {})
    has_recorded_semantic_regression = bool(
        list(comparison.get("regressed_cases") or [])
        or list(canary.get("protected_regressions") or [])
        or _guard_regression_rows(comparison)
    )
    plan_route_not_activated = (
        update_kind == "plan"
        and (
            reject_reason == "plan_effect_not_attributable"
            or (
                bool(
                    plan_effect_activation.get(
                        "unactivated_repair_target_txs"
                    )
                )
                and not bool(
                    plan_effect_activation.get("attributed_improved_txs")
                )
                and not has_recorded_semantic_regression
            )
        )
    )
    runtime_failures = [
        dict(raw)
        for raw in list(
            comparison.get("candidate_runtime_failures")
            or canary.get("runtime_failures")
            or []
        )
        if isinstance(raw, dict)
    ]
    runtime_failure_txs = {
        str(raw.get("tx_hash") or "").strip().lower()
        for raw in runtime_failures
        if str(raw.get("tx_hash") or "").strip()
    }
    baseline_by_tx = _index_results_by_tx(baseline_full_results or [])
    candidate_by_tx = _index_results_by_tx(item.get("full_results") or [])
    improved_targets: List[str] = []
    preserved_behaviors: List[str] = []
    regressions: List[Dict[str, Any]] = []
    for tx_hash, candidate_result in candidate_by_tx.items():
        if tx_hash in runtime_failure_txs:
            continue
        baseline_result = baseline_by_tx.get(tx_hash)
        if baseline_result is None:
            continue
        baseline_correct = _result_is_correct(baseline_result)
        candidate_correct = _result_is_correct(candidate_result)
        if not baseline_correct and _result_is_repair_improvement(
            baseline_result,
            candidate_result,
        ):
            improved_targets.append(tx_hash)
            flips = _condition_flips(baseline_result, candidate_result)
            if flips:
                candidate_result.setdefault(
                    "_refinement_condition_flips",
                    flips,
                )
        elif baseline_correct and candidate_correct:
            preserved_behaviors.append(tx_hash)
        elif baseline_correct and not candidate_correct:
            regressions.append({
                "tx_hash": tx_hash,
                "role": "baseline_correct",
                "ground_truth": str(
                    build_slim_result(baseline_result).get("ground_truth") or ""
                ),
                "baseline_verdict": _result_verdict(baseline_result),
                "candidate_verdict": _result_verdict(candidate_result),
                "condition_flips": _condition_flips(
                    baseline_result,
                    candidate_result,
                ),
            })
    for raw in list(canary.get("protected_regressions") or []):
        if isinstance(raw, dict):
            regressions.append(dict(raw))
    regressions.extend(_guard_regression_rows(comparison))
    deduped_regressions: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for regression in regressions:
        key = (
            str(regression.get("tx_hash") or "").strip().lower(),
            str(regression.get("role") or ""),
            str(regression.get("baseline_verdict") or regression.get("old_pred") or ""),
            str(regression.get("candidate_verdict") or regression.get("new_pred") or ""),
        )
        deduped_regressions[key] = regression
    regressions = list(deduped_regressions.values())
    improved_targets = list(dict.fromkeys(
        [
            *[str(tx).strip().lower() for tx in canary.get("fixed_repair_target_txs", [])],
            *improved_targets,
        ]
    ))[:12]
    unattributed_improved_targets: List[str] = []
    unattributed_runtime_regressions: List[Dict[str, Any]] = []
    if update_kind == "plan":
        attributed_txs = {
            str(tx_hash).strip().lower()
            for tx_hash in list(
                plan_effect_activation.get("attributed_improved_txs") or []
            )
            if str(tx_hash).strip()
        }
        unattributed_improved_targets = [
            tx_hash for tx_hash in improved_targets
            if tx_hash not in attributed_txs
        ]
        # Only repair-target improvements caused by an executed changed route
        # may become the next Plan refinement boundary.
        improved_targets = [
            tx_hash for tx_hash in improved_targets
            if tx_hash in attributed_txs
        ]
    if plan_route_not_activated:
        # A Plan route that never ran cannot establish a semantic regression
        # boundary. Preserve the raw rerun flips for audit only.
        unattributed_runtime_regressions = [dict(item) for item in regressions]
        regressions = []
    preserved_behaviors = list(dict.fromkeys(preserved_behaviors))[:12]
    regression_count = len(regressions)
    dependency_status = str(
        (item.get("dependency_validation") or comparison.get("dependency_validation") or {}).get(
            "dependency_validation_status"
        )
        or "passed"
    )
    malformed = (
        item.get("rule") is None
        or dependency_status not in {"", "passed"}
        or bool(comparison.get("rule_budget_rejected"))
    )
    broad_regression = regression_count > max(3, len(improved_targets) + 2)
    refinable = (
        not bool(comparison.get("accept"))
        and not malformed
        and not bool(runtime_failures)
        and (
            plan_route_not_activated
            or bool(improved_targets)
        )
        and not broad_regression
    )
    if runtime_failures:
        refinability_reason = "validation_incomplete"
    elif malformed:
        refinability_reason = "malformed_or_dependency_invalid_candidate"
    elif plan_route_not_activated:
        refinability_reason = "plan_route_not_activated"
    elif not improved_targets:
        refinability_reason = "no_target_improvement"
    elif broad_regression:
        refinability_reason = "broad_protected_regression"
    else:
        refinability_reason = "partial_target_repair_with_component_level_regression"

    affected_components: List[Dict[str, Any]] = []
    for delta in list(item.get("selected_atomic_deltas") or []):
        if not isinstance(delta, dict):
            continue
        affected_components.append({
            "kind": "rule",
            "condition_id": str(delta.get("condition_id") or "").strip().upper(),
            "operation": str(delta.get("operation") or ""),
            "capability_id": "",
        })
    for delta in list(item.get("plan_strategy_deltas") or []):
        if not isinstance(delta, dict):
            continue
        affected_components.append({
            "kind": "plan",
            "condition_id": str(delta.get("condition_id") or "").strip().upper(),
            "operation": str(
                delta.get("operation")
                or delta.get("strategy_operation")
                or ""
            ),
            "capability_id": str(delta.get("capability_id") or ""),
        })
    if not affected_components:
        for condition_id in [
            *list(item.get("applied_rule_patch_conditions") or []),
            *list(item.get("applied_plan_patch_conditions") or []),
        ]:
            affected_components.append({
                "kind": str(item.get("update_kind") or ""),
                "condition_id": str(condition_id or "").strip().upper(),
                "operation": "candidate_patch",
                "capability_id": "",
            })
    desired_refinement = (
        [
            "Preserve semantic improvements observed on unaffected cases, but do not derive a Rule or Plan boundary from the runtime failure itself.",
            "Retry the failed condition before using this candidate as a refinement base.",
        ]
        if runtime_failures
        else [
            "Keep this refinement owned by Plan; do not convert an unactivated evidence route into a Rule semantic boundary.",
            "Refine the existing Plan proposal so its changed evidence route is actually selected and rendered for the affected condition during validation.",
            "Do not preserve unattributed verdict flips as candidate improvements.",
        ]
        if plan_route_not_activated
        else [
            "Preserve candidate behavior on improved target errors while repairing only the component-level regression.",
            "Do not promote this candidate unless the refined candidate passes canary, regression, hard guard, and promotion validation against the incumbent baseline.",
        ]
    )
    if regressions:
        flipped = sorted({
            str(flip.get("condition_id") or "").strip().upper()
            for regression in regressions
            if isinstance(regression, dict)
            for flip in list(regression.get("condition_flips") or [])
            if isinstance(flip, dict) and str(flip.get("condition_id") or "").strip()
        })
        if flipped:
            desired_refinement.append(
                "Restore protected behavior for affected conditions: "
                + ", ".join(flipped[:8])
                + "."
            )
    improved_condition_ids = sorted({
        str(flip.get("condition_id") or "").strip().upper()
        for tx_hash in improved_targets
        for result in [candidate_by_tx.get(str(tx_hash).strip().lower())]
        if isinstance(result, dict)
        for flip in list(result.get("_refinement_condition_flips") or [])
        if isinstance(flip, dict) and str(flip.get("condition_id") or "").strip()
    })
    regressed_condition_ids = sorted({
        str(flip.get("condition_id") or "").strip().upper()
        for regression in regressions
        if isinstance(regression, dict)
        for flip in list(regression.get("condition_flips") or [])
        if isinstance(flip, dict) and str(flip.get("condition_id") or "").strip()
    })
    regression_boundary = [
        {
            "tx_hash": str(regression.get("tx_hash") or ""),
            "role": str(regression.get("role") or ""),
            "ground_truth": str(regression.get("ground_truth") or ""),
            "baseline_verdict": str(regression.get("baseline_verdict") or ""),
            "candidate_verdict": str(regression.get("candidate_verdict") or ""),
            "condition_flips": list(regression.get("condition_flips") or []),
        }
        for regression in regressions
        if isinstance(regression, dict)
    ][:12]
    next_allowed_refinement_kind = (
        "retry_validation_without_semantic_change"
        if runtime_failures
        else "activate_existing_plan_route"
        if plan_route_not_activated
        else "preserve_improved_targets_repair_component_regression"
        if regressions
        else "continue_refinement_without_broadening"
        if improved_targets
        else "new_search_direction_required"
    )
    return {
        "schema_version": "evotx.refinement_feedback.v1",
        "source_round": int(round_index),
        "source_candidate": str(item.get("name") or ""),
        "source_candidate_status": str(item.get("candidate_status") or "supported"),
        "source_update_kind": str(item.get("update_kind") or ""),
        "rejection_reason": reject_reason,
        "refinable": refinable,
        "refinability_reason": refinability_reason,
        "improved_targets": improved_targets,
        "unattributed_improved_targets": unattributed_improved_targets,
        "improved_condition_ids": improved_condition_ids,
        "preserved_behaviors": preserved_behaviors,
        "regressions": regressions[:12],
        "unattributed_runtime_regressions": (
            unattributed_runtime_regressions[:12]
        ),
        "regressed_condition_ids": regressed_condition_ids,
        "regression_boundary": regression_boundary,
        "runtime_failures": runtime_failures[:12],
        "validation_incomplete": bool(runtime_failures),
        "affected_components": affected_components[:16],
        "next_allowed_refinement_kind": next_allowed_refinement_kind,
        "must_preserve_candidate_effect": bool(improved_targets),
        "plan_effect_activation": plan_effect_activation,
        "desired_refinement": desired_refinement,
    }


def build_candidate_outcome(
    item: Dict[str, Any],
    *,
    round_index: int,
    baseline_full_results: List[Dict[str, Any]] | None,
    has_deferred_signal: bool,
) -> Dict[str, Any]:
    """Classify validation once for terminal audit, memory, and round control."""
    comparison = dict(item.get("comparison") or {})
    feedback = _candidate_refinement_feedback(
        item,
        round_index=round_index,
        baseline_full_results=baseline_full_results,
    )
    canary = dict(comparison.get("candidate_canary") or {})
    guard_gate = dict(comparison.get("guard_gate") or {})
    runtime_failures = list(
        comparison.get("candidate_runtime_failures")
        or canary.get("runtime_failures")
        or []
    )
    reject_reason = str(
        comparison.get("reject_reason")
        or guard_gate.get("reject_reason")
        or "candidate_rejected"
    )
    dependency_status = str(
        (
            item.get("dependency_validation")
            or comparison.get("dependency_validation")
            or {}
        ).get("dependency_validation_status")
        or "passed"
    )

    if runtime_failures:
        terminal_reason = "candidate_not_executable"
        terminal_stage = "runtime"
        detail_code = "candidate_validation_incomplete"
    elif dependency_status not in {"", "passed"} or bool(
        comparison.get("rule_budget_rejected")
    ):
        terminal_reason = "candidate_not_executable"
        terminal_stage = "structural_validation"
        detail_code = reject_reason
    elif (
        reject_reason == "plan_effect_not_attributable"
        or feedback.get("refinability_reason") == "plan_route_not_activated"
    ):
        terminal_reason = "candidate_not_executable"
        terminal_stage = "runtime"
        detail_code = str(
            (comparison.get("plan_effect_failure") or {}).get("reason")
            or "plan_route_not_activated"
        )
    elif guard_gate.get("enabled") and not bool(guard_gate.get("accept", True)):
        terminal_reason = "protected_regression"
        terminal_stage = "hard_guard"
        detail_code = reject_reason
    elif bool(comparison.get("accept")):
        terminal_reason = "accepted"
        terminal_stage = "validation"
        detail_code = str(comparison.get("accept_reason") or "accepted")
    elif list(feedback.get("regressions") or []):
        terminal_reason = "protected_regression"
        terminal_stage = (
            "canary"
            if bool(canary.get("enabled"))
            and comparison.get("candidate_evaluation_scope") == "canary_only"
            else "regression"
        )
        detail_code = reject_reason
    else:
        terminal_reason = "candidate_ineffective"
        terminal_stage = (
            "canary"
            if bool(canary.get("enabled"))
            and comparison.get("candidate_evaluation_scope") == "canary_only"
            else "validation"
        )
        detail_code = reject_reason

    refinable = bool(feedback.get("refinable"))
    if terminal_reason == "accepted":
        refinable = False
        feedback["refinability_reason"] = ""
        next_round_action = "promote"
    elif runtime_failures:
        # Engineering retry has already been attempted in this validation.
        # An unresolved execution failure must not become semantic memory.
        refinable = False
        next_round_action = "stop"
    elif refinable:
        next_round_action = "refine"
    elif terminal_reason in {"candidate_ineffective", "protected_regression"}:
        # A semantically resolved rejection may make this candidate unsuitable
        # as the next search base without exhausting the episode's search. Keep
        # the accepted incumbent and let the next round use attributed memory.
        next_round_action = "advance_signal"
    elif has_deferred_signal:
        next_round_action = "advance_signal"
    else:
        next_round_action = "stop"

    feedback.update({
        "rejection_reason": detail_code if terminal_reason != "accepted" else "",
        "refinable": refinable,
        "next_round_action": next_round_action,
    })
    return {
        "schema_version": "evotx.candidate_outcome.v1",
        "decision": "accept" if terminal_reason == "accepted" else "reject",
        "terminal_reason": terminal_reason,
        "terminal_stage": terminal_stage,
        "detail_code": detail_code,
        "refinable": refinable,
        "refinability_reason": str(feedback.get("refinability_reason") or ""),
        "next_round_action": next_round_action,
        "runtime_failure": bool(runtime_failures),
        "refinement_feedback": feedback,
    }


def attach_candidate_outcomes(
    evaluations: List[Dict[str, Any]],
    *,
    round_index: int,
    baseline_full_results: List[Dict[str, Any]] | None,
    review_bundle: Dict[str, Any],
) -> None:
    has_deferred_signal = bool(_minimal_deferred_signal_ids(review_bundle))
    for item in list(evaluations or []):
        outcome = build_candidate_outcome(
            item,
            round_index=round_index,
            baseline_full_results=baseline_full_results,
            has_deferred_signal=has_deferred_signal,
        )
        item["candidate_outcome"] = outcome
        comparison = item.get("comparison")
        if isinstance(comparison, dict):
            comparison["candidate_outcome"] = outcome
            candidate_dir = str(item.get("candidate_dir") or "")
            if candidate_dir:
                write_json(Path(candidate_dir) / "candidate_outcome.json", outcome)
                write_json(Path(candidate_dir) / "comparison.json", comparison)


def candidate_outcome_decision(item: Dict[str, Any]) -> str:
    """Return the classified decision, with legacy artifact compatibility."""
    decision = str(
        ((item or {}).get("candidate_outcome") or {}).get("decision") or ""
    ).strip().lower()
    if decision in {"accept", "reject"}:
        return decision
    return (
        "accept"
        if bool(((item or {}).get("comparison") or {}).get("accept"))
        else "reject"
    )


def select_refinement_base_candidate(
    evaluations: List[Dict[str, Any]],
    *,
    round_index: int,
    baseline_full_results: List[Dict[str, Any]] | None,
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    scored: List[tuple[tuple[int, int, int, int, str], Dict[str, Any], Dict[str, Any]]] = []
    best_feedback: Dict[str, Any] = {}
    continuation_feedback: List[tuple[str, Dict[str, Any]]] = []
    plan_activation_feedback: List[tuple[str, Dict[str, Any]]] = []
    for item in list(evaluations or []):
        if candidate_outcome_decision(item) == "accept":
            continue
        if str(item.get("update_kind") or "") not in {"rule", "plan", "rule_plan"}:
            continue
        outcome = dict(item.get("candidate_outcome") or {})
        feedback = dict(outcome.get("refinement_feedback") or {})
        if not feedback:
            feedback = _candidate_refinement_feedback(
                item,
                round_index=round_index,
                baseline_full_results=baseline_full_results,
            )
        if feedback.get("refinability_reason") == "plan_route_not_activated":
            plan_activation_feedback.append((
                str(item.get("name") or ""),
                feedback,
            ))
        if str(feedback.get("next_round_action") or "") in {
            "refine",
            "advance_signal",
        }:
            continuation_feedback.append((
                str(item.get("name") or ""),
                feedback,
            ))
        if not best_feedback or (
            len(feedback.get("improved_targets") or []),
            -len(feedback.get("regressions") or []),
        ) > (
            len(best_feedback.get("improved_targets") or []),
            -len(best_feedback.get("regressions") or []),
        ):
            best_feedback = feedback
        if feedback.get("runtime_failures"):
            continue
        if not bool(feedback.get("refinable")):
            continue
        summary = dict(item.get("summary") or {})
        scored.append((
            (
                -len(feedback.get("improved_targets") or []),
                len(feedback.get("regressions") or []),
                int(summary.get("errors", 10**6) or 10**6),
                1 if bool(item.get("probe_only")) else 0,
                str(item.get("name") or ""),
            ),
            item,
            feedback,
        ))
    scored_with_observed_improvement = [
        row
        for row in scored
        if list(row[2].get("improved_targets") or [])
    ]
    if scored_with_observed_improvement:
        # Keep failures candidate-local. An unactivated Plan route must remain
        # Plan feedback, but it must not suppress a separate Rule candidate
        # that actually repaired a target and only needs boundary refinement.
        _, selected, feedback = sorted(
            scored_with_observed_improvement,
            key=lambda row: row[0],
        )[0]
        return selected, feedback
    if plan_activation_feedback:
        # With no independently observed semantic improvement, retain the Plan
        # execution-contract failure and refine route activation from the
        # accepted incumbent rather than inheriting the rejected candidate.
        _, feedback = sorted(plan_activation_feedback, key=lambda row: row[0])[0]
        return None, feedback
    if not scored:
        if continuation_feedback:
            _, feedback = sorted(continuation_feedback, key=lambda row: row[0])[0]
            return None, feedback
        return None, best_feedback
    _, selected, feedback = sorted(scored, key=lambda row: row[0])[0]
    return selected, feedback


def memory_with_active_refinement_feedback(
    memory: Dict[str, Any],
    feedback: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload = normalize_rejected_update_memory(memory or {})
    payload.pop("active_refinement_feedback", None)
    if feedback:
        payload["active_refinement_feedback"] = dict(feedback)
    return normalize_rejected_update_memory(payload)


def build_construction_refinement_feedback(
    candidate_viability: Dict[str, Any] | None,
    *,
    round_index: int,
) -> Dict[str, Any]:
    """Convert an owner-local construction rejection into next-round feedback."""
    rejections = [
        dict(item)
        for item in list(
            (candidate_viability or {}).get("pre_candidate_rejections") or []
        )
        if isinstance(item, dict)
    ]
    mutable = [
        item
        for item in rejections
        if str(item.get("owner") or "").strip().lower() in {"rule", "plan"}
    ]
    if not mutable:
        return {}
    rejection = next(
        (
            item
            for item in mutable
            if str(item.get("failure_class") or "").strip().lower()
            != "update_generation_failed"
        ),
        mutable[0],
    )
    owner = str(rejection.get("owner") or "").strip().lower()
    failure_class = str(rejection.get("failure_class") or "").strip().lower()
    reason = str(
        rejection.get("reason")
        or rejection.get("rejection_reason")
        or "candidate_construction_rejected"
    )
    refinable = failure_class != "update_generation_failed"
    condition_ids = sorted({
        str(value or "").strip().upper()
        for value in list(rejection.get("scope_target_steps") or [])
        if str(value or "").strip()
    })
    return {
        "schema_version": "evotx.refinement_feedback.v1",
        "source_round": int(round_index),
        "source_candidate": f"{owner}_candidate_construction",
        "source_candidate_status": "construction_rejected",
        "source_update_kind": owner,
        "rejection_reason": reason,
        "refinable": refinable,
        "refinability_reason": (
            "owner_local_candidate_construction_rejected"
            if refinable
            else "updater_completion_exhausted"
        ),
        "improved_targets": [],
        "preserved_behaviors": [],
        "regressions": [],
        "affected_components": [
            {
                "kind": owner,
                "condition_id": condition_id,
                "operation": "candidate_construction",
                "capability_id": "",
            }
            for condition_id in condition_ids
        ],
        "next_allowed_refinement_kind": (
            "refine_owner_local_construction"
            if refinable
            else "retry_same_updater_task"
        ),
        "must_preserve_candidate_effect": False,
        "desired_refinement": [
            "Keep the canonical signal owner and scope unchanged.",
            "Generate a different bounded local operation that satisfies the "
            "recorded construction safety constraint.",
        ] if refinable else [],
    }


def _explicit_packet_capability_todos(
    packet_engineering_todos: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in list(packet_engineering_todos or [])
        if isinstance(item, dict)
        and str(item.get("update_target") or "").strip().lower() == "packet"
        and (
            bool(item.get("hard_blocking"))
            or str(item.get("patch_action") or "").strip().lower()
            not in {"", "none"}
        )
    ]


def classify_zero_candidate_stop_reason(
    *,
    preupdate_memory_gate: Dict[str, Any],
    candidate_viability: Dict[str, Any],
    plan_construction_rejections: List[Dict[str, Any]],
    packet_engineering_todos: List[Dict[str, Any]],
    review_bundle: Dict[str, Any],
) -> str:
    """Attribute a zero-candidate round to its first concrete failure."""
    dedupe = dict(
        (candidate_viability or {}).get("minimal_exact_candidate_dedupe") or {}
    )
    if (
        bool(dedupe.get("enabled"))
        and list(dedupe.get("exact_duplicates") or [])
        and not list(dedupe.get("executed_candidate_names") or [])
    ):
        return "exact_candidate_duplicate"
    if preupdate_memory_gate.get("blocked_targets"):
        return "candidate_rejected_by_memory_before_update"
    if candidate_viability.get("rejected_candidate_count"):
        return "candidate_artifact_not_viable"
    if plan_construction_rejections:
        return "candidate_construction_rejected"
    if list((review_bundle or {}).get("review_execution_failures") or []):
        return "review_execution_failed"

    hard_engineering_todos = [
        item
        for item in list(packet_engineering_todos or [])
        if isinstance(item, dict) and bool(item.get("hard_blocking"))
    ]
    engineering_targets = {
        str(item.get("update_target") or "").strip().lower()
        for item in hard_engineering_todos
        if isinstance(item, dict)
    }
    if engineering_targets == {"runtime"}:
        return "runtime_repair_required"
    if _explicit_packet_capability_todos(packet_engineering_todos):
        return "packet_evidence_update_required"
    if hard_engineering_todos:
        return "packet_evidence_update_required"
    bundle = review_bundle or {}
    matrix = dict(bundle.get("update_signal_matrix") or {})
    canonical_matrix_authoritative = bool(
        bundle.get("canonical_signal_contract")
        or str(matrix.get("schema_version") or "").startswith(
            "evotx.canonical_update_signal"
        )
    )
    if canonical_matrix_authoritative:
        actionable = any(
            isinstance(signal, dict)
            and str(signal.get("update_target") or "").strip().lower()
            in {"rule", "plan"}
            and str(signal.get("generalization_status") or "").strip().lower()
            == "supported"
            for signal in list(matrix.get("signals") or [])
        )
        return "update_generation_failed" if actionable else "no_valid_signal"
    return "artifact_update_noop"


def build_reachable_signal_terminal_audit(
    review_bundle: Dict[str, Any],
    candidate_specs: List[Dict[str, Any]],
    *,
    candidate_viability: Dict[str, Any] | None = None,
    candidate_evaluations: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build one end-to-end signal/candidate lineage with terminal reasons."""
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    specs = [dict(spec) for spec in list(candidate_specs or []) if isinstance(spec, dict)]

    def signal_ids(item: Dict[str, Any]) -> set[str]:
        return {
            str(value)
            for value in [
            *list(item.get("applied_plan_signal_ids") or []),
            *list(item.get("experimental_signal_ids") or []),
            *list(item.get("hypothesis_source_signal_ids") or []),
            *[
                signal_id
                for delta in list(
                    item.get("selected_atomic_deltas")
                    or item.get("atomic_rule_deltas")
                    or []
                )
                if isinstance(delta, dict)
                for signal_id in list(delta.get("applied_signal_ids") or [])
            ],
            ]
            if str(value)
        }

    viability_by_name = {
        str(item.get("candidate_name") or ""): dict(item)
        for item in list((candidate_viability or {}).get("candidates") or [])
        if isinstance(item, dict) and str(item.get("candidate_name") or "")
    }
    evaluation_by_name = {
        str(item.get("name") or ""): dict(item)
        for item in list(candidate_evaluations or [])
        if isinstance(item, dict) and str(item.get("name") or "")
    }
    construction_rejections = [
        dict(item)
        for item in list(
            (candidate_viability or {}).get("pre_candidate_rejections") or []
        )
        if isinstance(item, dict)
    ]
    deferred_signal_ids = _minimal_deferred_signal_ids(review_bundle)
    explicit_packet_todos = _explicit_packet_capability_todos(
        list((review_bundle or {}).get("packet_engineering_todos") or [])
    )

    def explicit_packet_todo_for_signal(
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        if str(signal.get("update_target") or "").strip().lower() != "packet":
            return {}
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        for todo in explicit_packet_todos:
            todo_conditions = {
                str(value or "").strip().upper()
                for value in list(todo.get("condition_ids") or [])
                if str(value or "").strip()
            }
            if not todo_conditions or condition_id in todo_conditions:
                return todo
        return {}

    def construction_rejection_signal_ids(item: Dict[str, Any]) -> set[str]:
        materialization = dict(item.get("plan_patch_materialization") or {})
        return {
            str(value)
            for value in [
                *list(item.get("accepted_patch_signal_ids") or []),
                *list(materialization.get("accepted_signal_ids") or []),
            ]
            if str(value)
        }

    def construction_rejection_for_signal(
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        signal_id = str(signal.get("signal_id") or "")
        target = str(signal.get("update_target") or "").strip().lower()
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        for rejection in construction_rejections:
            explicit_ids = construction_rejection_signal_ids(rejection)
            if explicit_ids and signal_id in explicit_ids:
                return rejection
            if explicit_ids:
                continue
            owner = str(rejection.get("owner") or "").strip().lower()
            scope = {
                str(value or "").strip().upper()
                for value in list(rejection.get("scope_target_steps") or [])
                if str(value or "").strip()
            }
            if owner == target and (not scope or condition_id in scope):
                return rejection
        return {}

    candidate_terminals: List[Dict[str, Any]] = []
    for spec in specs:
        name = str(spec.get("name") or "")
        viability = dict(viability_by_name.get(name) or {})
        evaluation = dict(evaluation_by_name.get(name) or {})
        comparison = dict(evaluation.get("comparison") or {})
        outcome = dict(evaluation.get("candidate_outcome") or {})
        reject_reason = str(comparison.get("reject_reason") or "").strip()
        reject_reason_code = reject_reason.lower()
        canary = dict(comparison.get("candidate_canary") or {})
        canary_active = bool(canary.get("enabled")) or (
            "enabled" not in canary and "passed" in canary
        )
        guard_gate = dict(comparison.get("guard_gate") or {})
        rejection_type = _candidate_rejection_type(comparison)
        runtime_failures = list(
            comparison.get("candidate_runtime_failures")
            or canary.get("runtime_failures")
            or []
        )
        terminal_stage = ""
        detail_code = ""
        if bool(spec.get("exact_candidate_duplicate")):
            terminal_reason = "exact_candidate_duplicate"
            terminal_stage = "candidate_dedupe"
            detail_code = "exact_rule_plan_semantic_duplicate"
        elif outcome:
            terminal_reason = str(
                outcome.get("terminal_reason") or "candidate_ineffective"
            )
            terminal_stage = str(outcome.get("terminal_stage") or "validation")
            detail_code = str(outcome.get("detail_code") or "candidate_rejected")
        elif comparison.get("accept") is True:
            terminal_reason = "accepted"
            terminal_stage = "validation"
            detail_code = str(comparison.get("accept_reason") or "accepted")
        elif viability and not bool(viability.get("viable")):
            terminal_reason = "candidate_not_executable"
            terminal_stage = "viability"
            details = list(viability.get("pre_judge_rejections") or [])
            detail_code = str((details[0] if details else {}).get("code") or viability.get("reason") or "")
        elif evaluation:
            if runtime_failures:
                terminal_reason = "candidate_not_executable"
                terminal_stage = "runtime"
                detail_code = "candidate_validation_incomplete"
            elif rejection_type == "plan_route_not_activated":
                terminal_reason = "candidate_not_executable"
                terminal_stage = "runtime"
                detail_code = "plan_route_not_activated"
            elif reject_reason_code in {
                "activation_probe_target_route_not_observed",
                "route_not_activated",
                "candidate_not_executable",
                "candidate_validation_incomplete",
                "rejected by hard rule complexity budget.",
            }:
                terminal_reason = "candidate_not_executable"
                terminal_stage = "runtime"
                detail_code = reject_reason_code
            elif guard_gate.get("enabled") and not bool(
                guard_gate.get("accept", True)
            ):
                terminal_reason = "protected_regression"
                terminal_stage = "hard_guard"
                detail_code = str(
                    guard_gate.get("reject_reason")
                    or comparison.get("reject_reason")
                    or "hard_negative_guard_rejection"
                )
            elif rejection_type == "uncertain_gate_rejection":
                terminal_reason = "uncertain_gate_rejection"
                terminal_stage = "regression"
                detail_code = str(
                    comparison.get("reject_reason")
                    or "uncertain_gate_rejection"
                )
            elif list(canary.get("protected_regressions") or []):
                terminal_reason = "protected_regression"
                terminal_stage = (
                    "canary"
                    if comparison.get("candidate_evaluation_scope") == "canary_only"
                    or canary_active
                    else "regression"
                )
                detail_code = reject_reason or rejection_type
            else:
                terminal_reason = "candidate_ineffective"
                terminal_stage = (
                    "canary"
                    if comparison.get("candidate_evaluation_scope") == "canary_only"
                    or canary_active
                    else "regression"
                )
                detail_code = reject_reason or "candidate_rejected"
            if not terminal_stage:
                terminal_stage = (
                    "canary"
                    if comparison.get("candidate_evaluation_scope") == "canary_only"
                    or canary_active
                    else "regression"
                )
                detail_code = str(
                    reject_reason or "candidate_rejected"
                )
        else:
            terminal_reason = "candidate_not_executable"
            terminal_stage = "viability"
            detail_code = "candidate_did_not_reach_runtime"
        candidate_terminals.append({
            "candidate_name": name,
            "candidate_type": str(spec.get("update_kind") or ""),
            "candidate_strategy": str(spec.get("candidate_strategy") or ""),
            "signal_ids": sorted(signal_ids(spec)),
            "terminal_stage": terminal_stage,
            "terminal_reason": terminal_reason,
            "detail_code": detail_code,
        })

    terminal_rank = {
        "accepted": 6,
        "exact_candidate_duplicate": 5,
        "protected_regression": 5,
        "uncertain_gate_rejection": 5,
        "candidate_ineffective": 4,
        "candidate_not_executable": 3,
        "update_generation_failed": 2,
        "missing_packet_capability": 1,
        "no_valid_signal": 0,
    }
    terminals: List[Dict[str, Any]] = []
    for signal in list(matrix.get("signals") or []):
        if not isinstance(signal, dict):
            continue
        signal_id = str(signal.get("signal_id") or "")
        status = str(signal.get("generalization_status") or "").strip().lower()
        linked = [
            item for item in candidate_terminals
            if signal_id in set(item.get("signal_ids") or [])
        ]
        escalation = dict(signal.get("owner_escalation") or {})
        update_target = str(signal.get("update_target") or "").strip().lower()
        construction_rejection = construction_rejection_for_signal(signal)
        explicit_packet_todo = explicit_packet_todo_for_signal(signal)
        if linked:
            best = max(
                linked,
                key=lambda item: terminal_rank.get(
                    str(item.get("terminal_reason") or ""), -1
                ),
            )
            terminal = str(best.get("terminal_reason") or "")
            reason = str(best.get("detail_code") or "")
            terminal_stage = str(best.get("terminal_stage") or "")
        elif signal_id in deferred_signal_ids:
            terminal = "deferred_next_round"
            reason = "condition_local_updater_projection"
            terminal_stage = "updater_projection"
        elif (
            str(signal.get("training_gate_reason") or "").strip().lower()
            == "packet_or_runtime_capability_absent"
            and (
                update_target in {"packet", "runtime"}
                or escalation.get("owner") in {"packet", "runtime"}
            )
        ):
            terminal = "missing_packet_capability"
            reason = str(
                escalation.get("reason")
                or "packet_or_runtime_capability_absent"
            )
            terminal_stage = "routing"
        elif explicit_packet_todo:
            terminal = "missing_packet_capability"
            reason = str(
                explicit_packet_todo.get("patch_action")
                or "packet_owner_requires_engineering"
            )
            terminal_stage = "routing"
        elif status not in {"supported", "experimental_plan", "experimental_rule"}:
            terminal = "no_valid_signal"
            reason = str(signal.get("training_gate_reason") or f"status_{status or 'unknown'}")
            terminal_stage = "signal_contract"
        elif update_target in {"packet", "runtime"} or escalation.get("owner") in {
            "packet",
            "runtime",
        }:
            terminal = "missing_packet_capability"
            reason = str(
                escalation.get("reason")
                or f"{update_target or escalation.get('owner')}_owner_requires_engineering"
            )
            terminal_stage = "routing"
        elif construction_rejection:
            failure_class = str(
                construction_rejection.get("failure_class") or ""
            ).strip().lower()
            terminal = (
                "update_generation_failed"
                if failure_class == "update_generation_failed"
                else "candidate_not_executable"
            )
            reason = str(
                construction_rejection.get("reason")
                or "candidate_construction_rejected"
            )
            terminal_stage = "candidate_construction"
        else:
            terminal = "update_generation_failed"
            reason = "actionable_signal_not_materialized"
            terminal_stage = "candidate_construction"
        terminals.append({
            "signal_id": signal_id,
            "update_target": str(signal.get("update_target") or ""),
            "condition_id": str(signal.get("condition_id") or ""),
            "generalization_status": status,
            "terminal_stage": terminal_stage,
            "terminal_reason": terminal,
            "detail_code": reason,
            "candidate_names": [
                str(item.get("candidate_name") or "") for item in linked
            ],
        })
    terminal_counts: Dict[str, int] = {}
    for item in terminals:
        reason = str(item.get("terminal_reason") or "")
        terminal_counts[reason] = terminal_counts.get(reason, 0) + 1
    return {
        "schema_version": "evotx.evolution_lineage.v1",
        "signal_count": len(terminals),
        "terminal_reason_counts": terminal_counts,
        "terminals": terminals,
        "candidates": candidate_terminals,
        "candidate_count": len(specs),
        "policy": (
            "Each signal and candidate receives one terminal stage and reason; "
            "the audit never changes acceptance behavior."
        ),
    }


def apply_rejected_memory_preupdate_gate(
    review_bundle: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compatibility audit: rejected memory is feedback, never a hard gate."""
    bundle = copy.deepcopy(review_bundle or {})
    rejected_memory = normalize_rejected_update_memory(
        bundle.get("rejected_update_memory") or {}
    )
    entries = list(rejected_memory.get("entries") or [])
    return bundle, {
        "schema_version": "evotx.preupdate_rejected_memory_gate.v1",
        "enabled": False,
        "policy": "feedback_only_exact_candidate_dedupe_in_portfolio",
        "hard_gate_applied": False,
        "memory_entry_count": len(entries),
        "blocked_signal_count": 0,
        "blocked_signals": [],
        "blocked_targets": [],
        "supported_target_counts_before": {},
        "supported_target_counts_after": {},
    }


def count_review_targets(review_bundle: Dict[str, Any]) -> Dict[str, int]:
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    matrix_signals = [
        dict(signal)
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict)
    ]
    canonical_matrix_authoritative = bool(
        (review_bundle or {}).get("canonical_signal_contract")
        or str(matrix.get("schema_version") or "").startswith(
            "evotx.canonical_update_signal"
        )
    )
    if matrix_signals or canonical_matrix_authoritative:
        owner_indexes: Dict[str, set[int]] = {
            "rule": set(),
            "plan": set(),
            "packet": set(),
            "runtime": set(),
        }
        for signal in matrix_signals:
            owner = str(signal.get("update_target") or "").strip().lower()
            status = str(
                signal.get("generalization_status") or ""
            ).strip().lower()
            allowed_statuses = {"supported"}
            if owner == "rule":
                allowed_statuses.add("experimental_rule")
            elif owner == "plan":
                allowed_statuses.add("experimental_plan")
            if owner not in owner_indexes or status not in allowed_statuses:
                continue
            owner_indexes[owner].add(
                int(signal.get("source_review_index", -1))
            )
        return {
            "rule": len(owner_indexes["rule"]),
            "plan": len(owner_indexes["plan"]),
            "mixed_rule_plan": len(
                owner_indexes["rule"] & owner_indexes["plan"]
            ),
            "packet": len(owner_indexes["packet"]),
            "runtime": len(owner_indexes["runtime"]),
            "total_reviews": len(set().union(*owner_indexes.values())),
        }

    actionable_rule = [
        item
        for item in list((review_bundle or {}).get("actionable_rule_reviews", []) or [])
        if item.get("update_target") == "rule" and item.get("should_update_rule")
    ]
    non_rule = list((review_bundle or {}).get("non_rule_reviews", []) or [])
    engineering = list(
        (review_bundle or {}).get("engineering_reviews", []) or []
    )
    explicit_plan = list(
        (review_bundle or {}).get("actionable_plan_reviews", []) or []
    )
    if explicit_plan:
        plan = explicit_plan
    else:
        all_reviews = actionable_rule + non_rule
        plan = [
            item
            for item in all_reviews
            if item.get("should_update_plan_strategy", False)
            and _review_has_supported_target_signal(item, "plan")
        ]
    mixed_plan = [item for item in plan if item in actionable_rule]
    engineering_source = engineering or non_rule
    packet = [
        item for item in engineering_source if item.get("update_target") == "packet"
    ]
    runtime = [
        item for item in engineering_source if item.get("update_target") == "runtime"
    ]
    return {
        "rule": len(actionable_rule),
        "plan": len(plan),
        "mixed_rule_plan": len(mixed_plan),
        "packet": len(packet),
        "runtime": len(runtime),
        "total_reviews": len(actionable_rule) + len(non_rule),
    }


def _review_has_plan_patch(review: Dict[str, Any]) -> bool:
    if not isinstance(review, dict):
        return False
    suggestion = review.get("plan_patch_suggestion", {}) or {}
    if isinstance(suggestion, dict):
        action = str(suggestion.get("action") or "").strip().lower()
        if action not in {"", "none"}:
            return True
    if review.get("should_update_plan_strategy") is True:
        return True
    for diagnosis in list(review.get("condition_diagnosis", []) or []):
        if isinstance(diagnosis, dict) and diagnosis.get("is_plan_problem"):
            return True
    return False


def _review_has_supported_target_signal(
    review: Dict[str, Any],
    target: str,
) -> bool:
    allowed_statuses = {"supported"}
    if target == "plan":
        allowed_statuses.add("experimental_plan")
    if target == "rule":
        allowed_statuses.add("experimental_rule")
    for signal in list(review.get("update_signals", []) or []):
        if not isinstance(signal, dict):
            continue
        signal_target = str(
            signal.get("update_target") or review.get("update_target") or ""
        )
        if signal_target != target:
            continue
        status = str(signal.get("generalization_status") or "").lower().strip()
        if status in allowed_statuses:
            return True
    return False


def _compact_review_text(value: Any, limit: int = 700) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _unique_compact_values(values: List[Any], *, limit: int, max_items: int) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        text = _compact_review_text(value, limit)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def build_packet_engineering_todos(
    review_bundle: Dict[str, Any],
    *,
    max_items: int = 12,
) -> List[Dict[str, Any]]:
    """Preserve packet/runtime evidence gaps without turning them into rule edits."""
    todos: List[Dict[str, Any]] = []
    review_source = list(
        (review_bundle or {}).get("engineering_reviews", []) or []
    ) or list((review_bundle or {}).get("non_rule_reviews", []) or [])
    for review in review_source:
        if not isinstance(review, dict):
            continue
        target = str(review.get("update_target") or "").strip().lower()
        if target not in {"packet", "runtime"}:
            continue

        signals = [
            dict(signal)
            for signal in list(review.get("update_signals", []) or [])
            if isinstance(signal, dict)
            and str(signal.get("update_target") or review.get("update_target") or "")
            .strip()
            .lower()
            == target
        ]
        patch_key = f"{target}_patch_suggestion"
        suggestion = dict(review.get(patch_key) or {})
        patch_action = str(suggestion.get("action") or "").strip().lower()
        routes = [
            dict(route)
            for route in list(review.get("diagnosis_routes") or [])
            if isinstance(route, dict)
            and str(route.get("owner") or "").strip().lower() == target
            and str(route.get("route_status") or "") == "reachable"
        ]
        if target == "packet":
            routes = [
                route for route in routes
                if is_authoritative_engineering_route(route)
            ]
            if not routes:
                continue
        if not signals and patch_action in {"", "none"} and not routes:
            continue

        condition_values: List[Any] = [
            signal.get("condition_id") for signal in signals if signal.get("condition_id")
        ]
        if suggestion.get("condition_id"):
            condition_values.append(suggestion.get("condition_id"))
        for route in routes:
            condition_values.extend(list(route.get("affected_conditions") or []))
        for diagnosis in list(review.get("condition_diagnosis", []) or []):
            if not isinstance(diagnosis, dict):
                continue
            if diagnosis.get(f"is_{target}_problem") and diagnosis.get("condition_id"):
                condition_values.append(diagnosis.get("condition_id"))

        todos.append(
            {
                "update_target": target,
                "source_error_type": str(review.get("error_type") or ""),
                "condition_ids": [
                    item.upper()
                    for item in _unique_compact_values(
                        condition_values,
                        limit=40,
                        max_items=8,
                    )
                ],
                "signal_statuses": _unique_compact_values(
                    [signal.get("generalization_status") for signal in signals],
                    limit=40,
                    max_items=6,
                ),
                "hard_blocking": any(
                    str(signal.get("generalization_status") or "")
                    .strip()
                    .lower()
                    == "supported"
                    for signal in signals
                ),
                "hard_blocking_signal_ids": _unique_compact_values(
                    [
                        signal.get("signal_id")
                        for signal in signals
                        if str(signal.get("generalization_status") or "")
                        .strip()
                        .lower()
                        == "supported"
                    ],
                    limit=64,
                    max_items=8,
                ),
                "directions": _unique_compact_values(
                    [signal.get("direction") for signal in signals],
                    limit=60,
                    max_items=6,
                ),
                "abstract_features": _unique_compact_values(
                    [signal.get("abstract_feature") for signal in signals],
                    limit=420,
                    max_items=5,
                ),
                "patch_action": patch_action,
                "diagnosis_routes": routes,
                "view_name": _compact_review_text(
                    suggestion.get("view_name")
                    or suggestion.get("tool_name")
                    or suggestion.get("source_name"),
                    120,
                ),
                "proposed_change": _compact_review_text(
                    suggestion.get("proposed_change")
                    or suggestion.get("change")
                    or suggestion.get("rationale"),
                    700,
                ) or _compact_review_text(
                    "; ".join(
                        str(item)
                        for route in routes
                        for item in list(route.get("fact_basis") or [])
                    ),
                    700,
                ),
                "review_rationale": _compact_review_text(
                    review.get("rationale")
                    or review.get("summary")
                    or review.get("misclassification_reason"),
                    700,
                ),
            }
        )
        if len(todos) >= max_items:
            break
    return todos


def build_packet_blocker_audit(
    packet_engineering_todos: List[Dict[str, Any]],
    review_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Explain hard Packet/Runtime blockers without re-routing them to Plan."""
    hard_items = [
        dict(item)
        for item in list(packet_engineering_todos or [])
        if isinstance(item, dict) and bool(item.get("hard_blocking"))
    ]
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    signals_by_id = {
        str(signal.get("signal_id") or ""): dict(signal)
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict) and str(signal.get("signal_id") or "")
    }
    blockers: List[Dict[str, Any]] = []
    blocked_signal_ids: set[str] = set()
    for item in hard_items:
        signal_ids = [
            str(value)
            for value in list(item.get("hard_blocking_signal_ids") or [])
            if str(value)
        ]
        related_signals = [
            signals_by_id[signal_id]
            for signal_id in signal_ids
            if signal_id in signals_by_id
        ]
        blocked_signal_ids.update(signal_ids)
        blockers.append({
            "update_target": str(item.get("update_target") or ""),
            "condition_ids": list(item.get("condition_ids") or []),
            "signal_ids": signal_ids,
            "view_name": str(item.get("view_name") or ""),
            "proposed_change": str(item.get("proposed_change") or ""),
            "directions": list(item.get("directions") or []),
            "abstract_features": list(item.get("abstract_features") or []),
            "missing_capability_basis": [
                str(signal.get("abstract_feature") or "")
                for signal in related_signals
                if str(signal.get("abstract_feature") or "")
            ],
            "why_rule_plan_cannot_fix": (
                "The diagnosis requires a Packet/Runtime evidence capability. "
                "Rule semantics and Plan routing can only consume available "
                "capabilities; they must not fabricate missing packet views, "
                "runtime facts, or source/trace material."
            ),
            "blocked_mutable_owners": ["rule", "plan"],
        })
    independent_mutable_signal_ids = sorted({
        str(signal.get("signal_id") or "")
        for signal in signals_by_id.values()
        if str(signal.get("signal_id") or "")
        and str(signal.get("signal_id") or "") not in blocked_signal_ids
        and str(signal.get("update_target") or "").strip().lower()
        in {"rule", "plan"}
        and str(signal.get("generalization_status") or "").strip().lower()
        in {"supported", "experimental_plan", "experimental_rule"}
    })
    return {
        "schema_version": "evotx.packet_blocker_audit.v1",
        "hard_blocking_count": len(blockers),
        "blockers": blockers,
        "blocked_signal_ids": sorted(blocked_signal_ids),
        "independent_mutable_signal_ids": independent_mutable_signal_ids,
        "rule_plan_candidate_generation_blocked_by_packet_capability": bool(
            blockers and not independent_mutable_signal_ids
        ),
        "policy": (
            "Hard Packet/Runtime blockers stop repeated Rule/Plan evolution "
            "for missing evidence capabilities and should be addressed in the "
            "packet/runtime layer."
        ),
    }


def build_plan_candidate_construction_rejections(
    update_audits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep bounded PlanUpdater failures that occur before candidate viability."""
    rejections: List[Dict[str, Any]] = []
    for raw_audit in list(update_audits or []):
        if not isinstance(raw_audit, dict):
            continue
        payload = dict(raw_audit.get("parsed_response") or {})
        materialization = dict(
            payload.get("plan_patch_materialization")
            or (payload.get("metadata") or {}).get(
                "plan_patch_materialization"
            )
            or {}
        )
        rejected_patch_components = [
            dict(item)
            for item in list(materialization.get("patches") or [])
            if isinstance(item, dict) and not bool(item.get("accepted"))
        ]
        lineage_rejected = bool(raw_audit.get("candidate_lineage_rejected"))
        rejected = bool(
            payload.get("phase3_plan_strategy_rejected")
            or payload.get("plan_evidence_gate_rejected")
            or payload.get("plan_patch_realization_rejected")
            or payload.get("plan_patch_materialization_rejected")
            or raw_audit.get("error")
            or rejected_patch_components
            or lineage_rejected
        )
        if not rejected:
            continue
        partial = dict(
            payload.get("phase3_plan_strategy_operation_filter")
            or payload.get("phase3_plan_strategy_partial_acceptance")
            or {}
        )
        realization_resolution = dict(
            payload.get("plan_patch_realization_resolution") or {}
        )
        realization_rejected_fields = [
            dict(item)
            for item in list(realization_resolution.get("dropped_fields") or [])
            if isinstance(item, dict)
        ]
        rejections.append({
            "owner": "plan",
            "rejection_stage": "candidate_construction",
            "reason": str(
                raw_audit.get("completion_failure_reason")
                or raw_audit.get("candidate_lineage_rejection_reason")
                or payload.get("plan_evidence_gate_rejected_reason")
                or payload.get("plan_patch_realization_rejected_reason")
                or payload.get("plan_patch_materialization_rejected_reason")
                or (
                    "plan_patch_partially_materialized"
                    if rejected_patch_components
                    and bool(raw_audit.get("semantic_changed"))
                    else ""
                )
                or raw_audit.get("error")
                or "phase3_plan_strategy_safety_rejected"
            )[:160],
            "failure_class": (
                "update_generation_failed"
                if raw_audit.get("completion_failure_reason") or lineage_rejected
                else "candidate_not_executable"
            ),
            "candidate_survived_construction": bool(
                raw_audit.get("semantic_changed") and not lineage_rejected
            ),
            "scope_target_steps": list(
                (raw_audit.get("scope_guard") or {}).get("target_steps") or []
            )[:12],
            "strategy_violations": list(
                payload.get("phase3_plan_strategy_violations") or []
            )[:12],
            "remaining_violations": list(
                payload.get("phase3_plan_strategy_remaining_violations") or []
            )[:12],
            "retained_strategy_deltas": list(
                partial.get("retained_strategy_deltas") or []
            )[:12],
            "rejected_operations": [
                *list(partial.get("rejected_operations") or []),
                *realization_rejected_fields,
            ][:12],
            "post_safety_budget_audits": list(
                partial.get("post_safety_budget_audits") or []
            )[:12],
            "plan_patch_materialization": materialization,
            "plan_patch_realization_resolution": realization_resolution,
            "rejected_patch_components": rejected_patch_components[:12],
            "accepted_patch_signal_ids": list(
                materialization.get("accepted_signal_ids") or []
            )[:24],
            "error": str(raw_audit.get("error") or "")[:240],
        })
    return rejections[-12:]


def _reviews_from_bundle(review_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    reviews: List[Dict[str, Any]] = []
    for key in ("actionable_rule_reviews", "non_rule_reviews"):
        for item in list((review_bundle or {}).get(key, []) or []):
            if isinstance(item, dict):
                reviews.append(dict(item))
    if reviews:
        return reviews
    raw = (review_bundle or {}).get("reviews", [])
    return [dict(item) for item in raw if isinstance(item, dict)]


def load_human_example(inline_text: str, file_path: str | None) -> str:
    if inline_text.strip():
        return inline_text.strip()
    if not file_path:
        return ""
    return Path(file_path).read_text(encoding="utf-8").strip()


def make_llm(
    model: str | None,
    provider: str | None = None,
    *,
    max_tokens: int | None = None,
    minimax_thinking: str = "adaptive",
):
    if not model and not provider:
        return None
    return OpenAICompatibleLLM(
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        minimax_thinking=minimax_thinking,
    )


def resolve_role_llm_route(
    args,
    *,
    model: str | None,
    thinking: str | None,
) -> tuple[str | None, str | None]:
    return resolve_adaptive_llm_route(
        model=model,
        provider=getattr(args, "llm_provider", None),
        thinking=thinking,
        adaptive_model=getattr(args, "adaptive_model", None),
        adaptive_provider=getattr(args, "adaptive_provider", None),
    )


def describe_role_llm_config(
    args,
    *,
    model: str | None,
    thinking: str | None,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    resolved_model, resolved_provider = resolve_role_llm_route(
        args,
        model=model,
        thinking=thinking,
    )
    config = describe_llm_config(
        model=resolved_model,
        provider=resolved_provider,
        max_tokens=max_tokens,
        minimax_thinking=thinking,
    )
    adaptive_override = (
        str(thinking or "").strip().lower() == "adaptive"
        and bool(
            getattr(args, "adaptive_model", None)
            or getattr(args, "adaptive_provider", None)
        )
    )
    config["routing_tier"] = "adaptive" if adaptive_override else "default"
    return config


def build_llm_config_summary(args) -> Dict[str, Any]:
    return {
        "cold_start": describe_role_llm_config(
            args,
            model=args.cold_start_model or args.llm_model,
            max_tokens=args.cold_start_max_tokens,
            thinking=args.cold_start_thinking,
        ),
        "planner": describe_role_llm_config(
            args,
            model=args.planner_model or args.llm_model,
            max_tokens=args.planner_max_tokens,
            thinking=args.planner_thinking,
        ),
        "judge": describe_role_llm_config(
            args,
            model=args.judge_model or args.llm_model,
            max_tokens=args.judge_max_tokens,
            thinking=args.judge_thinking,
        ),
        "aggregator": describe_role_llm_config(
            args,
            model=args.judge_model or args.llm_model,
            max_tokens=args.aggregator_max_tokens,
            thinking=args.aggregator_thinking,
        ),
        "env": describe_llm_config(
            model=args.env_model or args.llm_model,
            provider=args.llm_provider,
        ),
        "review": describe_role_llm_config(
            args,
            model=args.review_model or args.llm_model,
            max_tokens=args.review_max_tokens,
            thinking=args.review_thinking,
        ),
        "update": describe_role_llm_config(
            args,
            model=args.update_model or args.llm_model,
            max_tokens=args.update_max_tokens,
            thinking=args.update_thinking,
        ),
    }


def resolve_attack_description(
    attack_description: str | None,
    label: str | None,
) -> str:
    if attack_description and attack_description.strip():
        return attack_description.strip()
    if label and label.strip():
        return label.strip().replace("_", " ").replace("-", " ")
    raise ValueError(
        "Provide either --rule-file or an attack-family seed via --attack-description/--label."
    )


def load_cases_input(args) -> tuple[List[Dict[str, Any]], str]:
    pos_csv = args.pos_csv or args.malicious_csv
    has_negative_csv = bool(args.neg_csv or args.benign_csv)
    if pos_csv or has_negative_csv:
        if not pos_csv or not has_negative_csv or not args.label:
            raise ValueError(
                "When using split CSV input, provide --pos-csv, --neg-csv, and --label together "
                "(legacy --malicious-csv/--benign-csv are still accepted)."
            )
        cases, resolved_label = load_cases_from_split_csvs(
            pos_csv=args.pos_csv,
            neg_csv=args.neg_csv,
            benign_csv=args.benign_csv,
            malicious_csv=args.malicious_csv,
            label=args.label,
            head_per_csv=getattr(args, "num_shots", None),
        )
        num_shots = getattr(args, "num_shots", None)
        if num_shots is not None:
            positive_count = sum(
                1 for case in cases if case.get("ground_truth") == "attack"
            )
            negative_count = sum(
                1 for case in cases if case.get("ground_truth") == "benign"
            )
            if positive_count != num_shots or negative_count != num_shots:
                raise ValueError(
                    "--num-shots requested "
                    f"{num_shots} rows per class, but loaded "
                    f"positive={positive_count}, negative={negative_count}."
                )
        return cases, resolved_label

    if not args.cases_file:
        raise ValueError(
            "Provide either --cases-file or --pos-csv --neg-csv --label."
        )
    if getattr(args, "num_shots", None) is not None:
        raise ValueError("--num-shots is only supported with split CSV input.")
    return load_labeled_cases(
        args.cases_file,
        positive_label=args.positive_label,
    )


def load_hard_negative_guard_cases(
    args,
    resolved_positive_label: str,
) -> List[Dict[str, Any]]:
    """Load fixed hard negatives used only as candidate acceptance guard."""
    if not args.hard_neg_csv:
        return []
    path = Path(args.hard_neg_csv)
    if not path.exists():
        raise FileNotFoundError(f"--hard-neg-csv not found: {path}")

    cases: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            row = dict(row or {})
            tx_hash = _first_csv_field(
                row,
                ["tx_hash", "txHash", "hash", "transaction_hash", "TransactionHash"],
            )
            if not tx_hash:
                continue
            dedupe_key = tx_hash.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            chain = _first_csv_field(row, ["chain", "Chain", "network"], default="eth")
            raw_type = _first_csv_field(
                row,
                ["Type", "type", "label", "raw_ground_truth"],
                default="hard_negative_guard",
            )
            cause = _first_csv_field(
                row,
                ["Cause", "cause", "label_rationale", "rationale", "description"],
            )
            metadata = {
                "negative_kind": "other_attack",
                "guard_role": "hard_negative_guard",
                "used_for_review": False,
                "target_label": resolved_positive_label,
                "source_csv": str(path),
                "csv_row_index": row_index,
            }
            for key in ("HackId", "hackId", "id", "Type", "Cause", "URL", "url"):
                value = row.get(key)
                if value not in (None, ""):
                    metadata[key] = value

            cases.append({
                "tx_hash": tx_hash,
                "chain": chain or "eth",
                "ground_truth": "benign",
                "raw_ground_truth": raw_type,
                "tx_context": {},
                "static_evidence": {},
                "label_rationale": cause,
                "report": cause,
                "metadata": metadata,
            })
    return cases


def _first_csv_field(
    row: Dict[str, Any],
    names: List[str],
    default: str = "",
) -> str:
    lower_map = {
        str(key).strip().lower(): value
        for key, value in (row or {}).items()
        if key is not None
    }
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name]).strip()
        value = lower_map.get(str(name).strip().lower())
        if value not in (None, ""):
            return str(value).strip()
    return default


def count_labels(cases: List[Dict[str, Any]]) -> Dict[str, int]:
    attack = sum(1 for case in cases if case.get("ground_truth") == "attack")
    negative = sum(1 for case in cases if case.get("ground_truth") == "benign")
    negative_benign = sum(
        1
        for case in cases
        if (case.get("metadata", {}) or {}).get("negative_kind") == "benign"
    )
    negative_other_attack = sum(
        1
        for case in cases
        if (case.get("metadata", {}) or {}).get("negative_kind") == "other_attack"
    )
    negative_unknown_other = sum(
        1
        for case in cases
        if (case.get("metadata", {}) or {}).get("negative_kind") == "unknown_other"
    )
    return {
        "attack": attack,
        "benign": negative,
        "negative": negative,
        "negative_benign": negative_benign,
        "negative_other": negative - negative_benign,
        "negative_other_attack": negative_other_attack,
        "negative_unknown_other": negative_unknown_other,
        "total": len(cases),
    }


def retarget_reused_results(
    results: List[Dict[str, Any]],
    *,
    rule_source: str,
    plan_source: str | None = None,
    reuse_metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    copied = copy.deepcopy(results)
    for result in copied:
        result["runtime_reuse"] = dict(reuse_metadata)
        detector_context = result.get("detector_context")
        if isinstance(detector_context, dict):
            detector_context["rule_source"] = rule_source
            if plan_source is not None:
                detector_context["plan_source"] = plan_source
        evaluation = result.get("evaluation")
        if isinstance(evaluation, dict):
            case_metadata = evaluation.setdefault("case_metadata", {})
            if isinstance(case_metadata, dict):
                case_metadata["result_reuse"] = dict(reuse_metadata)
    return copied


def build_planner_guidance_from_review_bundle(
    review_bundle: Dict[str, Any],
    max_items: int = 6,
) -> Dict[str, Any]:
    plan_reviews = []
    review_sources = [
        ("non_rule_reviews", review)
        for review in list((review_bundle or {}).get("non_rule_reviews", []) or [])
    ] + [
        ("actionable_rule_reviews", review)
        for review in list((review_bundle or {}).get("actionable_rule_reviews", []) or [])
    ]
    for source_key, review in review_sources:
        if not isinstance(review, dict):
            continue
        suggestion = review.get("plan_patch_suggestion", {}) or {}
        if not isinstance(suggestion, dict) or suggestion.get("action") in (None, "", "none"):
            continue
        if source_key == "non_rule_reviews" and review.get("update_target") != "plan":
            continue
        plan_reviews.append({
            "tx_hash": review.get("tx_hash", ""),
            "error_type": review.get("error_type", ""),
            "condition_id": suggestion.get("condition_id", ""),
            "action": suggestion.get("action", ""),
            "proposed_change": suggestion.get("proposed_change", ""),
            "rationale": suggestion.get("rationale", ""),
            "review_confidence": review.get("review_confidence", ""),
            "review_source": source_key,
        })
        if len(plan_reviews) >= max_items:
            break
    if not plan_reviews:
        return {}
    return {
        "source": "previous_round_plan_patch_reviews",
        "scope": "temporary_plan_generation_only",
        "instructions": [
            "Use these suggestions only to improve temporary evidence plans and judge questions.",
            "Do not copy these suggestions into the semantic rule.",
            "Prefer changing default_evidence_refs, allowed_followup_views, allowed_tools, max_followups, or question clarity.",
        ],
        "plan_reviews": plan_reviews,
    }


def run_cases(
    cases: List[Dict[str, Any]],
    rule: EvolvingRule,
    rule_source: str,
    tool_manifest: Dict[str, Any],
    llm_provider: str | None,
    planner_model: str | None,
    judge_model: str | None,
    env_model: str | None,
    use_environment: bool,
    base_cache_dir: str,
    force_rebuild_packet: bool,
    enable_source_tools: bool,
    source_cache_dir: str,
    force_refresh_source: bool,
    max_view_chars: int,
    max_context_chars: int,
    judge_max_tokens: int = 8192,
    aggregator_max_tokens: int = 8192,
    planner_thinking: str = "adaptive",
    judge_thinking: str = "disabled",
    aggregator_thinking: str = "disabled",
    adaptive_model: str | None = None,
    adaptive_provider: str | None = None,
    phase: str = "run",
    llm_transcripts_dir: str | None = None,
    planner_guidance: Dict[str, Any] | None = None,
    fixed_plan: EvidencePlan | Dict[str, Any] | None = None,
    fixed_plan_source: str | None = None,
    reuse_results_by_tx: Dict[str, Dict[str, Any]] | None = None,
    candidate_judge_reuse_enabled: bool = False,
    runtime_early_stop: bool = False,
    adaptive_evidence: bool = False,
    adaptive_evidence_mode: str = "off",
    adaptive_evidence_max_direct_trace_nodes: int = 80,
    adaptive_evidence_max_direct_trace_chars: int = 45000,
    adaptive_evidence_max_medium_trace_nodes: int = 250,
    adaptive_evidence_debug: bool = False,
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
    retry_failed_conditions: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(cases)
    reuse_results_by_tx = dict(reuse_results_by_tx or {})
    for index, case in enumerate(cases, start=1):
        reuse_result = reuse_results_by_tx.get(str(case["tx_hash"]).lower())
        reuse_lookup_status = (
            "yes" if reuse_result else "no"
        ) if candidate_judge_reuse_enabled else "not_applicable_baseline"
        effective_force_rebuild_packet = _effective_force_rebuild_packet(
            requested=force_rebuild_packet,
            candidate_judge_reuse_enabled=candidate_judge_reuse_enabled,
            reuse_result_found=reuse_result is not None,
        )
        print(
            f"[FewShot] Case {index}/{total} [{phase}]: tx={case['tx_hash']} chain={case['chain']} "
            f"ground_truth={case.get('ground_truth')} raw_label={case.get('raw_ground_truth')} "
            f"candidate_judge_reuse_enabled={candidate_judge_reuse_enabled} "
            f"reuse_result_found_by_tx={reuse_lookup_status} "
            f"force_rebuild_packet_requested={bool(force_rebuild_packet)} "
            f"force_rebuild_packet_effective={effective_force_rebuild_packet}"
        )
        result = run_single_inference(
            tx_hash=case["tx_hash"],
            chain=case["chain"],
            rule=rule,
            rule_source=rule_source,
            tool_manifest=tool_manifest,
            tx_context=_merge_planner_guidance(
                case.get("tx_context", {}) or {},
                planner_guidance,
            ),
            static_evidence=case.get("static_evidence", {}) or {},
            llm_provider=llm_provider,
            planner_model=planner_model,
            judge_model=judge_model,
            judge_max_tokens=judge_max_tokens,
            aggregator_max_tokens=aggregator_max_tokens,
            planner_thinking=planner_thinking,
            judge_thinking=judge_thinking,
            aggregator_thinking=aggregator_thinking,
            adaptive_model=adaptive_model,
            adaptive_provider=adaptive_provider,
            fixed_plan=fixed_plan,
            plan_source=fixed_plan_source,
            env_model=env_model,
            use_environment=use_environment,
            base_cache_dir=base_cache_dir,
            force_rebuild_packet=effective_force_rebuild_packet,
            freeze_packet=True,
            enable_source_tools=enable_source_tools,
            source_cache_dir=source_cache_dir,
            force_refresh_source=force_refresh_source,
            max_view_chars=max_view_chars,
            max_context_chars=max_context_chars,
            ground_truth=case.get("ground_truth"),
            raw_ground_truth=case.get("raw_ground_truth"),
            label_rationale=case.get("label_rationale", ""),
            report=case.get("report", ""),
            case_metadata=case.get("metadata", {}),
            reuse_result=reuse_result,
            runtime_early_stop=runtime_early_stop,
            runtime_early_stop_policy="conservative_negative",
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
            followup_context_mode=followup_context_mode,
            judge_followup_mode=judge_followup_mode,
            judge_expanded_followup_views=judge_expanded_followup_views,
            rate_limit_serial_fallback=rate_limit_serial_fallback,
            rate_limit_retry_attempts=rate_limit_retry_attempts,
            rate_limit_retry_delay_seconds=rate_limit_retry_delay_seconds,
            llm_transcripts_dir=llm_transcripts_dir,
            transcript_phase=phase,
        )
        retry_condition_ids = sorted({
            condition_id
            for condition_id, call in _judge_calls_by_condition(result).items()
            if _judge_runtime_failure_kind(call)[0]
        })
        retry_audit: Dict[str, Any] = {
            "enabled": bool(retry_failed_conditions),
            "attempted": False,
            "condition_ids": retry_condition_ids,
            "original_failed_condition_ids": retry_condition_ids,
            "dependency_expanded_condition_ids": [],
            "recovered_condition_ids": [],
            "unresolved_condition_ids": retry_condition_ids,
            "isolation_preserved": True,
            "changed_unaffected_condition_ids": [],
        }
        if retry_failed_conditions and retry_condition_ids:
            retry_audit["attempted"] = True
            downstream_retry_ids = sorted(
                _stateful_downstream_condition_ids(
                    fixed_plan,
                    set(retry_condition_ids),
                )
            )
            expanded_retry_condition_ids = sorted(
                set(retry_condition_ids) | set(downstream_retry_ids)
            )
            retry_audit["dependency_expanded_condition_ids"] = downstream_retry_ids
            retry_audit["condition_ids"] = expanded_retry_condition_ids
            pre_retry_result = result
            retry_reuse_result = _filter_reuse_result_by_invalidated_steps(
                result,
                set(expanded_retry_condition_ids),
            )
            print(
                "[FewShot] Candidate condition retry: "
                f"tx={case['tx_hash']} conditions={expanded_retry_condition_ids}"
            )
            retry_result = run_single_inference(
                tx_hash=case["tx_hash"],
                chain=case["chain"],
                rule=rule,
                rule_source=rule_source,
                tool_manifest=tool_manifest,
                tx_context=_merge_planner_guidance(
                    case.get("tx_context", {}) or {},
                    planner_guidance,
                ),
                static_evidence=case.get("static_evidence", {}) or {},
                llm_provider=llm_provider,
                planner_model=planner_model,
                judge_model=judge_model,
                judge_max_tokens=judge_max_tokens,
                aggregator_max_tokens=aggregator_max_tokens,
                planner_thinking=planner_thinking,
                judge_thinking=judge_thinking,
                aggregator_thinking=aggregator_thinking,
                adaptive_model=adaptive_model,
                adaptive_provider=adaptive_provider,
                fixed_plan=fixed_plan,
                plan_source=fixed_plan_source,
                env_model=env_model,
                use_environment=use_environment,
                base_cache_dir=base_cache_dir,
                force_rebuild_packet=False,
                freeze_packet=True,
                enable_source_tools=enable_source_tools,
                source_cache_dir=source_cache_dir,
                force_refresh_source=force_refresh_source,
                max_view_chars=max_view_chars,
                max_context_chars=max_context_chars,
                ground_truth=case.get("ground_truth"),
                raw_ground_truth=case.get("raw_ground_truth"),
                label_rationale=case.get("label_rationale", ""),
                report=case.get("report", ""),
                case_metadata=case.get("metadata", {}),
                reuse_result=retry_reuse_result,
                runtime_early_stop=runtime_early_stop,
                runtime_early_stop_policy="conservative_negative",
                adaptive_evidence=adaptive_evidence,
                adaptive_evidence_mode=adaptive_evidence_mode,
                adaptive_evidence_max_direct_trace_nodes=adaptive_evidence_max_direct_trace_nodes,
                adaptive_evidence_max_direct_trace_chars=adaptive_evidence_max_direct_trace_chars,
                adaptive_evidence_max_medium_trace_nodes=adaptive_evidence_max_medium_trace_nodes,
                adaptive_evidence_debug=adaptive_evidence_debug,
                parallel_judge=False,
                judge_concurrency=1,
                structured_judge_logs=structured_judge_logs,
                iv_stateful_runtime=iv_stateful_runtime,
                access_control_binding_mode=access_control_binding_mode,
                reentrancy_binding_mode=reentrancy_binding_mode,
                disable_stateful_bindings=disable_stateful_bindings,
                dynamic_aggregation=dynamic_aggregation,
                followup_context_mode=followup_context_mode,
                judge_followup_mode=judge_followup_mode,
                judge_expanded_followup_views=judge_expanded_followup_views,
                rate_limit_serial_fallback=rate_limit_serial_fallback,
                rate_limit_retry_attempts=rate_limit_retry_attempts,
                rate_limit_retry_delay_seconds=rate_limit_retry_delay_seconds,
                llm_transcripts_dir=llm_transcripts_dir,
                transcript_phase=f"{phase}/condition_retry",
            )
            isolation = _candidate_condition_retry_isolation_audit(
                pre_retry_result,
                retry_result,
                retried_condition_ids=set(expanded_retry_condition_ids),
            )
            retry_audit.update(isolation)
            if isolation["isolation_preserved"]:
                _preserve_parent_runtime_identity_after_engineering_retry(
                    pre_retry_result,
                    retry_result,
                    retry_audit=retry_audit,
                )
            result = retry_result if isolation["isolation_preserved"] else pre_retry_result
            unresolved = sorted({
                condition_id
                for condition_id, call in _judge_calls_by_condition(result).items()
                if _judge_runtime_failure_kind(call)[0]
            })
            retry_audit["unresolved_condition_ids"] = unresolved
            retry_audit["recovered_condition_ids"] = sorted(
                set(retry_condition_ids) - set(unresolved)
            )
        evaluation = result.get("evaluation", {}) if isinstance(result, dict) else {}
        case_metadata = evaluation.setdefault("case_metadata", {}) if isinstance(evaluation, dict) else {}
        if isinstance(case_metadata, dict):
            case_metadata["candidate_judge_reuse"] = {
                "enabled": bool(candidate_judge_reuse_enabled),
                "reuse_result_found_by_tx": bool(reuse_result),
                "lookup_status": reuse_lookup_status,
                "force_rebuild_packet_requested": bool(force_rebuild_packet),
                "force_rebuild_packet_effective": effective_force_rebuild_packet,
                "reuse_type": "candidate_vs_current_judge_step"
                if candidate_judge_reuse_enabled
                else "",
            }
            case_metadata["candidate_condition_retry"] = retry_audit
        results.append(result)
    return results


def _effective_force_rebuild_packet(
    *,
    requested: bool,
    candidate_judge_reuse_enabled: bool,
    reuse_result_found: bool,
) -> bool:
    if candidate_judge_reuse_enabled and reuse_result_found:
        return False
    return bool(requested)


def _preserve_parent_runtime_identity_after_engineering_retry(
    parent_result: Dict[str, Any],
    retry_result: Dict[str, Any],
    *,
    retry_audit: Dict[str, Any],
) -> None:
    """Keep a local recovery attempt under its parent validation identity."""
    parent_metadata = _runtime_trace_metadata(parent_result)
    retry_metadata = _runtime_trace_metadata(retry_result)
    parent_identity = str(parent_metadata.get("runtime_policy_identity") or "")
    retry_identity = str(retry_metadata.get("runtime_policy_identity") or "")
    retry_audit["parent_runtime_policy_identity"] = parent_identity
    retry_audit["engineering_retry_runtime_policy_identity"] = retry_identity
    if not parent_identity:
        return

    inference = retry_result.get("inference")
    if not isinstance(inference, dict):
        return
    trace = inference.get("trace")
    if not isinstance(trace, dict):
        return
    metadata = trace.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    metadata["engineering_retry_runtime_policy"] = {
        "runtime_policy_identity": retry_identity,
        "runtime_policy": dict(retry_metadata.get("runtime_policy") or {}),
    }
    metadata["runtime_policy_identity"] = parent_identity
    metadata["runtime_policy"] = dict(parent_metadata.get("runtime_policy") or {})


def build_candidate_canary_selection(
    *,
    spec: Dict[str, Any],
    args,
    cases: List[Dict[str, Any]],
    hard_guard_cases: List[Dict[str, Any]] | None,
    current_full_results: List[Dict[str, Any]],
    current_guard_full_results: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    """Select repair targets plus a bounded set of protected boundaries."""
    viability = dict(spec.get("candidate_viability") or {})
    target_txs = candidate_repair_target_txs(viability)[
        : max(0, int(getattr(args, "candidate_canary_error_cases", 2) or 0))
    ]
    changed_ids = {
        str(item).strip().upper()
        for item in list(viability.get("effective_changed_condition_ids") or [])
        if str(item).strip()
    }
    protected_positive: List[str] = []
    protected_boundary: List[str] = []
    for result in list(current_full_results or []):
        slim = build_slim_result(result)
        tx_hash = str(slim.get("tx_hash") or "").strip().lower()
        ground_truth = str(slim.get("ground_truth") or "").strip().lower()
        predicted = str(slim.get("predicted_verdict") or "").strip().lower()
        if not tx_hash or tx_hash in target_txs:
            continue
        if ground_truth == "attack" and predicted == "attack":
            condition_ids = {
                str(item.get("condition_id") or item.get("id") or "").strip().upper()
                for item in list(slim.get("condition_table") or [])
                if isinstance(item, dict)
            }
            if not changed_ids or changed_ids & condition_ids:
                protected_positive.append(tx_hash)
        elif ground_truth != "attack" and predicted != "attack":
            protected_boundary.append(tx_hash)

    protected_positive = protected_positive[
        : max(0, int(getattr(args, "candidate_canary_positive_cases", 3) or 0))
    ]
    protected_boundary = protected_boundary[
        : max(0, int(getattr(args, "candidate_canary_boundary_cases", 2) or 0))
    ]
    guard_txs: List[str] = []
    for result in list(current_guard_full_results or []):
        slim = build_slim_result(result)
        tx_hash = str(slim.get("tx_hash") or "").strip().lower()
        if (
            tx_hash
            and str(slim.get("ground_truth") or "").strip().lower() != "attack"
            and str(slim.get("predicted_verdict") or "").strip().lower() != "attack"
        ):
            guard_txs.append(tx_hash)
    guard_txs = guard_txs[
        : max(0, int(getattr(args, "candidate_canary_guard_cases", 2) or 0))
    ]

    train_roles = {
        **{tx: "repair_target" for tx in target_txs},
        **{tx: "protected_positive" for tx in protected_positive},
        **{tx: "protected_boundary" for tx in protected_boundary},
    }
    selected_train_cases = [
        case
        for case in list(cases or [])
        if str(case.get("tx_hash") or case.get("hash") or "").strip().lower()
        in train_roles
    ]
    selected_guard_cases = [
        case
        for case in list(hard_guard_cases or [])
        if str(case.get("tx_hash") or case.get("hash") or "").strip().lower()
        in set(guard_txs)
    ]
    return {
        "enabled": True,
        "changed_condition_ids": sorted(changed_ids),
        "train_roles": train_roles,
        "guard_roles": {tx: "protected_guard" for tx in guard_txs},
        "repair_target_txs": target_txs,
        "protected_positive_txs": protected_positive,
        "protected_boundary_txs": protected_boundary,
        "protected_guard_txs": guard_txs,
        "train_cases": selected_train_cases,
        "guard_cases": selected_guard_cases,
    }


def _judge_calls_by_condition(result: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    execution = dict(
        (((result or {}).get("inference") or {}).get("trace") or {}).get(
            "execution"
        )
        or {}
    )
    calls: Dict[str, Dict[str, Any]] = {}
    for raw in list(execution.get("judge_calls") or []):
        if not isinstance(raw, dict):
            continue
        condition_id = str(
            raw.get("condition_id") or raw.get("judge_id") or ""
        ).strip().upper()
        if condition_id:
            calls[condition_id] = dict(raw)
    return calls


def _candidate_condition_retry_isolation_audit(
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    retried_condition_ids: set[str],
) -> Dict[str, Any]:
    """Require a local retry to preserve every unaffected Judge outcome."""
    retried = {
        str(value or "").strip().upper()
        for value in retried_condition_ids
        if str(value or "").strip()
    }
    before_calls = _judge_calls_by_condition(before)
    after_calls = _judge_calls_by_condition(after)
    changed: List[str] = []
    for condition_id in sorted(set(before_calls) - retried):
        before_call = before_calls[condition_id]
        after_call = after_calls.get(condition_id)
        if after_call is None or _judge_retry_result_fingerprint(
            before_call
        ) != _judge_retry_result_fingerprint(after_call):
            changed.append(condition_id)
    return {
        "isolation_preserved": not bool(changed),
        "changed_unaffected_condition_ids": changed,
        "isolation_failure_reason": (
            "unaffected_condition_changed_during_local_retry" if changed else ""
        ),
    }


def _judge_retry_result_fingerprint(call: Dict[str, Any]) -> str:
    return stable_json_dumps({
        "answer": call.get("answer"),
        "final_answer": call.get("final_answer"),
        "reason": str(call.get("reason") or ""),
        "supporting_evidence_ids": list(
            call.get("supporting_evidence_ids") or []
        ),
        "missing_evidence": list(call.get("missing_evidence") or []),
        "tool_calls": [
            {
                "tool": str(item.get("tool") or item.get("name") or ""),
                "status": str(
                    item.get("tool_status") or item.get("status") or ""
                ),
            }
            for item in list(call.get("tool_calls") or [])
            if isinstance(item, dict)
        ],
    })


def _judge_input_fingerprint(call: Dict[str, Any]) -> str:
    policy = dict(call.get("judge_followup_policy") or {})
    return stable_json_dumps({
        "question": str(call.get("question") or ""),
        "expected_answer": call.get("expected_answer"),
        "evidence_refs": list(call.get("evidence_refs") or []),
        "selected_views_hash": str(call.get("selected_views_hash") or ""),
        "state_input": dict(call.get("state_input") or {}),
        "effective_default_views": list(
            policy.get("effective_default_views") or []
        ),
        "effective_followup_views": list(
            policy.get("effective_followup_views") or []
        ),
        "effective_allowed_tools": list(
            policy.get("effective_allowed_tools") or []
        ),
        "effective_max_followups": policy.get("effective_max_followups"),
    })


def _judge_effective_evidence_fingerprint(call: Dict[str, Any]) -> str:
    """Fingerprint evidence actually selected for a Judge invocation.

    Optional follow-up/tool availability is deliberately excluded. Merely
    allowing a route does not mean the route affected this invocation.
    """
    policy = dict(call.get("judge_followup_policy") or {})
    return stable_json_dumps({
        "question": str(call.get("question") or ""),
        "expected_answer": call.get("expected_answer"),
        "evidence_refs": list(call.get("evidence_refs") or []),
        "selected_views_hash": str(call.get("selected_views_hash") or ""),
        "state_input": dict(call.get("state_input") or {}),
        "effective_default_views": list(
            policy.get("effective_default_views") or []
        ),
    })


def _judge_runtime_failure_kind(call: Dict[str, Any]) -> tuple[str, str]:
    answer = str(
        call.get("final_answer") or call.get("answer") or ""
    ).strip().lower()
    if answer != "uncertain":
        return "", ""
    missing_text = " ".join(
        str(value)
        for value in list(call.get("missing_evidence") or [])
    )
    detail = " ".join((str(call.get("reason") or ""), missing_text)).lower()
    if any(
        token in detail
        for token in (
            "invalid json",
            "json repair failed",
            "valid judge output",
            "json extraction",
        )
    ):
        return "json_parse_failure", "judge output could not be parsed"
    if any(
        token in detail
        for token in (
            "api connection",
            "apiconnectionerror",
            "connection error",
            "judge execution failed",
            "runtime error",
            "timed out",
            "timeout",
            "transport error",
        )
    ):
        return "judge_runtime_failure", detail[:240]

    failed_statuses: List[str] = []
    for tool_call in list(call.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            continue
        status = str(
            tool_call.get("tool_status") or tool_call.get("status") or ""
        ).strip().lower()
        if any(
            token in status
            for token in (
                "timeout",
                "connection",
                "transport",
                "runtime_error",
                "tool_unavailable",
                "failed",
            )
        ):
            failed_statuses.append(status)
    if failed_statuses:
        return "tool_runtime_failure", ",".join(sorted(set(failed_statuses)))
    return "", ""


def _candidate_runtime_failures(
    baseline_result: Dict[str, Any] | None,
    candidate_result: Dict[str, Any] | None,
    *,
    changed_condition_ids: set[str],
) -> List[Dict[str, Any]]:
    baseline_calls = _judge_calls_by_condition(baseline_result)
    candidate_calls = _judge_calls_by_condition(candidate_result)
    failures: List[Dict[str, Any]] = []
    for condition_id in sorted(set(baseline_calls) - set(candidate_calls)):
        if condition_id in changed_condition_ids:
            continue
        failures.append({
            "condition_id": condition_id,
            "failure_kind": "condition_result_missing",
            "detail": "unchanged candidate condition produced no Judge result",
            "plan_condition_unchanged": True,
            "judge_input_unchanged": False,
        })
    for condition_id, candidate_call in candidate_calls.items():
        baseline_call = baseline_calls.get(condition_id)
        failure_kind, detail = _judge_runtime_failure_kind(candidate_call)
        if not failure_kind:
            continue
        input_unchanged = bool(
            baseline_call is not None
            and _judge_input_fingerprint(baseline_call)
            == _judge_input_fingerprint(candidate_call)
        )
        failures.append({
            "condition_id": condition_id,
            "failure_kind": failure_kind,
            "detail": detail,
            "plan_condition_unchanged": condition_id not in changed_condition_ids,
            "judge_input_unchanged": input_unchanged,
        })
    return failures


def collect_candidate_runtime_failures(
    baseline_results: List[Dict[str, Any]],
    candidate_results: List[Dict[str, Any]],
    *,
    changed_condition_ids: set[str],
    role: str,
) -> List[Dict[str, Any]]:
    """Collect execution failures from a complete candidate validation run."""
    baseline_by_tx = _index_results_by_tx(baseline_results)
    candidate_by_tx = _index_results_by_tx(candidate_results)
    failures: List[Dict[str, Any]] = []
    for tx_hash in sorted(set(baseline_by_tx) | set(candidate_by_tx)):
        candidate_result = candidate_by_tx.get(tx_hash)
        if candidate_result is None:
            failures.append({
                "condition_id": "",
                "failure_kind": "candidate_result_missing",
                "detail": "candidate validation produced no result for baseline case",
                "plan_condition_unchanged": False,
                "judge_input_unchanged": False,
                "tx_hash": tx_hash,
                "role": role,
            })
            continue
        candidate_calls = _judge_calls_by_condition(candidate_result)
        runtime_errors = [
            dict(item)
            for item in list((get_trace(candidate_result) or {}).get("errors") or [])
            if isinstance(item, dict)
        ]
        if not candidate_calls and runtime_errors:
            failures.append({
                "condition_id": "",
                "failure_kind": "packet_or_runtime_failure",
                "detail": str(runtime_errors[0].get("error") or "runtime failed")[:240],
                "plan_condition_unchanged": False,
                "judge_input_unchanged": False,
                "tx_hash": tx_hash,
                "role": role,
            })
            continue
        for failure in _candidate_runtime_failures(
            baseline_by_tx.get(tx_hash),
            candidate_result,
            changed_condition_ids=changed_condition_ids,
        ):
            failures.append({
                **failure,
                "tx_hash": tx_hash,
                "role": role,
            })
    return failures


def evaluate_candidate_canary_gate(
    *,
    selection: Dict[str, Any],
    baseline_results: List[Dict[str, Any]],
    candidate_results: List[Dict[str, Any]],
    baseline_guard_results: List[Dict[str, Any]],
    candidate_guard_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reject before full validation when a canary fixes nothing or regresses."""
    baseline_by_tx = _index_results_by_tx(baseline_results)
    candidate_by_tx = _index_results_by_tx(candidate_results)
    baseline_guard_by_tx = _index_results_by_tx(baseline_guard_results)
    candidate_guard_by_tx = _index_results_by_tx(candidate_guard_results)
    fixed_targets: List[str] = []
    unresolved_targets: List[str] = []
    regressions: List[Dict[str, Any]] = []
    runtime_failures: List[Dict[str, Any]] = []
    changed_condition_ids = {
        str(value or "").strip().upper()
        for value in list(selection.get("changed_condition_ids") or [])
        if str(value or "").strip()
    }

    def verdict(result: Dict[str, Any] | None) -> tuple[str, str]:
        slim = build_slim_result(result or {})
        return (
            str(slim.get("ground_truth") or "").strip().lower(),
            str(slim.get("predicted_verdict") or "").strip().lower(),
        )

    def condition_answers(result: Dict[str, Any] | None) -> Dict[str, Any]:
        slim = build_slim_result(result or {})
        answers: Dict[str, Any] = {}
        for row in list(slim.get("condition_table") or []):
            if not isinstance(row, dict):
                continue
            condition_id = str(
                row.get("condition_id") or row.get("id") or ""
            ).strip().upper()
            if condition_id:
                answers[condition_id] = row.get(
                    "final_answer",
                    row.get("answer"),
                )
        return answers

    def condition_flips(
        baseline: Dict[str, Any] | None,
        candidate: Dict[str, Any] | None,
    ) -> List[Dict[str, Any]]:
        before = condition_answers(baseline)
        after = condition_answers(candidate)
        return [
            {
                "condition_id": condition_id,
                "baseline_answer": before.get(condition_id),
                "candidate_answer": after.get(condition_id),
            }
            for condition_id in sorted(set(before) | set(after))
            if before.get(condition_id) != after.get(condition_id)
        ]

    for tx_hash, role in dict(selection.get("train_roles") or {}).items():
        case_runtime_failures = _candidate_runtime_failures(
            baseline_by_tx.get(tx_hash),
            candidate_by_tx.get(tx_hash),
            changed_condition_ids=changed_condition_ids,
        )
        for failure in case_runtime_failures:
            runtime_failures.append({
                **failure,
                "tx_hash": tx_hash,
                "role": role,
            })
        runtime_failure_condition_ids = {
            str(item.get("condition_id") or "").strip().upper()
            for item in case_runtime_failures
        }
        baseline_truth, baseline_predicted = verdict(baseline_by_tx.get(tx_hash))
        candidate_truth, candidate_predicted = verdict(candidate_by_tx.get(tx_hash))
        ground_truth = candidate_truth or baseline_truth
        candidate_correct = (
            candidate_predicted == "attack"
            if ground_truth == "attack"
            else candidate_predicted not in {"attack", "uncertain", ""}
        )
        baseline_correct = (
            baseline_predicted == "attack"
            if ground_truth == "attack"
            else baseline_predicted not in {"attack", "uncertain", ""}
        )
        if role == "repair_target":
            if candidate_correct:
                fixed_targets.append(tx_hash)
            else:
                unresolved_targets.append(tx_hash)
        elif baseline_correct and not candidate_correct:
            semantic_flips = [
                flip
                for flip in condition_flips(
                    baseline_by_tx.get(tx_hash),
                    candidate_by_tx.get(tx_hash),
                )
                if str(flip.get("condition_id") or "").strip().upper()
                not in runtime_failure_condition_ids
            ]
            if semantic_flips or not case_runtime_failures:
                regressions.append({
                    "tx_hash": tx_hash,
                    "role": role,
                    "ground_truth": ground_truth,
                    "baseline_verdict": baseline_predicted,
                    "candidate_verdict": candidate_predicted,
                    "condition_flips": semantic_flips,
                })

    for tx_hash in dict(selection.get("guard_roles") or {}):
        case_runtime_failures = _candidate_runtime_failures(
            baseline_guard_by_tx.get(tx_hash),
            candidate_guard_by_tx.get(tx_hash),
            changed_condition_ids=changed_condition_ids,
        )
        for failure in case_runtime_failures:
            runtime_failures.append({
                **failure,
                "tx_hash": tx_hash,
                "role": "protected_guard",
            })
        runtime_failure_condition_ids = {
            str(item.get("condition_id") or "").strip().upper()
            for item in case_runtime_failures
        }
        baseline_truth, baseline_predicted = verdict(
            baseline_guard_by_tx.get(tx_hash)
        )
        candidate_truth, candidate_predicted = verdict(
            candidate_guard_by_tx.get(tx_hash)
        )
        ground_truth = candidate_truth or baseline_truth
        baseline_correct = baseline_predicted not in {"attack", "uncertain", ""}
        candidate_correct = candidate_predicted not in {"attack", "uncertain", ""}
        if baseline_correct and not candidate_correct:
            semantic_flips = [
                flip
                for flip in condition_flips(
                    baseline_guard_by_tx.get(tx_hash),
                    candidate_guard_by_tx.get(tx_hash),
                )
                if str(flip.get("condition_id") or "").strip().upper()
                not in runtime_failure_condition_ids
            ]
            if semantic_flips or not case_runtime_failures:
                regressions.append({
                    "tx_hash": tx_hash,
                    "role": "protected_guard",
                    "ground_truth": ground_truth,
                    "baseline_verdict": baseline_predicted,
                    "candidate_verdict": candidate_predicted,
                    "condition_flips": semantic_flips,
                })

    target_txs = list(selection.get("repair_target_txs") or [])
    passed = (
        not runtime_failures
        and not regressions
        and (not target_txs or bool(fixed_targets))
    )
    return {
        "schema_version": "evotx.candidate_canary.v1",
        "enabled": True,
        "passed": passed,
        "repair_target_count": len(target_txs),
        "fixed_repair_target_txs": fixed_targets,
        "unresolved_repair_target_txs": unresolved_targets,
        "protected_regressions": regressions,
        "runtime_failure_count": len(runtime_failures),
        "runtime_failures": runtime_failures,
        "reject_reason": (
            "candidate_validation_incomplete"
            if runtime_failures
            else "candidate_canary_protected_boundary_regression"
            if regressions
            else "candidate_canary_fixed_no_target_error"
            if target_txs and not fixed_targets
            else ""
        ),
    }


def activation_probe_execution_audit(
    spec: Dict[str, Any],
    slim_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    probe = dict(spec.get("phase3_activation_probe") or {})
    condition_id = str(probe.get("condition_id") or "").strip().upper()
    target_routes = [
        str(route or "").strip()
        for route in list(probe.get("target_routes") or [])
        if str(route or "").strip()
    ]
    audit = {
        "schema_version": "evotx.phase3.activation_probe_execution.v1",
        "enabled": bool(condition_id and target_routes),
        "condition_id": condition_id,
        "target_routes": target_routes,
        "observed_routes": [],
        "missing_routes": list(target_routes),
        "tx_observations": [],
        "all_target_routes_observed": False,
    }
    if not audit["enabled"]:
        return audit
    observed: set[str] = set()
    for slim in list(slim_results or []):
        tx_hash = str(slim.get("tx_hash") or get_tx_hash(slim) or "")
        tx_seen: set[str] = set()
        for row in list(slim.get("condition_table") or []):
            if not isinstance(row, dict):
                continue
            row_condition = str(
                row.get("condition_id") or row.get("id") or ""
            ).strip().upper()
            if row_condition != condition_id:
                continue
            for meta in list(row.get("view_render_metadata") or []):
                if not isinstance(meta, dict):
                    continue
                view = str(meta.get("view") or "").strip()
                if view not in target_routes:
                    continue
                rendered = int(meta.get("rows_rendered", 0) or 0) > 0
                if rendered:
                    tx_seen.add(view)
        if tx_seen:
            observed.update(tx_seen)
        audit["tx_observations"].append({
            "tx_hash": tx_hash,
            "observed_routes": sorted(tx_seen),
        })
    audit["observed_routes"] = sorted(observed)
    audit["missing_routes"] = [
        route for route in target_routes if route not in observed
    ]
    audit["all_target_routes_observed"] = not bool(audit["missing_routes"])
    return audit


def plan_candidate_effect_activation_audit(
    *,
    spec: Dict[str, Any],
    base_plan: EvidencePlan | None,
    candidate_plan: EvidencePlan | None,
    baseline_results: List[Dict[str, Any]],
    candidate_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Attribute Plan fixes to an effective evidence/input change."""
    deltas = [
        delta.to_dict()
        for delta in build_plan_strategy_deltas(base_plan, candidate_plan)
    ]
    required_by_condition: Dict[str, set[str]] = {}
    changed_condition_ids: set[str] = set()
    for delta in deltas:
        condition_id = str(delta.get("condition_id") or "").strip().upper()
        if condition_id:
            changed_condition_ids.add(condition_id)
        routes = {
            str(value).strip()
            for value in [
                *list(delta.get("added_routes") or []),
                *list(delta.get("promoted_routes") or []),
                *list(delta.get("added_tools") or []),
            ]
            if str(value).strip()
        }
        if condition_id and routes:
            required_by_condition.setdefault(condition_id, set()).update(routes)

    viability = dict(spec.get("candidate_viability") or {})
    repair_target_txs = candidate_repair_target_txs(viability)
    case_audit_by_tx = {
        str(item.get("tx_hash") or "").strip().lower(): dict(item)
        for item in list(viability.get("case_audits") or [])
        if isinstance(item, dict) and str(item.get("tx_hash") or "").strip()
    }
    audit: Dict[str, Any] = {
        "schema_version": "evotx.plan_effect_activation.v1",
        "enabled": str(spec.get("update_kind") or "").strip().lower() == "plan",
        "route_change_requires_activation": bool(required_by_condition),
        "effect_change_requires_attribution": bool(changed_condition_ids),
        "changed_condition_ids": sorted(changed_condition_ids),
        "required_routes_by_condition": {
            key: sorted(value) for key, value in required_by_condition.items()
        },
        "improved_cases": [],
        "attributed_improved_txs": [],
        "unattributed_improved_txs": [],
        "incidental_improved_txs": [],
        "all_improvements_attributed": True,
        "repair_target_txs": repair_target_txs,
        "target_route_observations": [],
        "unactivated_repair_target_txs": [],
        "all_repair_target_routes_activated": True,
    }
    if not audit["enabled"] or not changed_condition_ids:
        return audit

    downstream_by_condition = {
        condition_id: _stateful_downstream_condition_ids(
            candidate_plan,
            {condition_id},
        )
        for condition_id in changed_condition_ids
    }
    baseline_by_tx = _index_results_by_tx(baseline_results)
    candidate_by_tx = _index_results_by_tx(candidate_results)

    for tx_hash in repair_target_txs:
        candidate_result = candidate_by_tx.get(tx_hash)
        case_audit = case_audit_by_tx.get(tx_hash, {})
        blocker_ids = {
            str(value or "").strip().upper()
            for value in [
                *list(case_audit.get("blocking_condition_ids") or []),
                *list(
                    case_audit.get(
                        "effective_covered_blocking_condition_ids"
                    ) or []
                ),
            ]
            if str(value or "").strip()
        }
        relevant_condition_ids = [
            condition_id
            for condition_id in sorted(required_by_condition)
            if not blocker_ids
            or bool(
                {
                    condition_id,
                    *downstream_by_condition.get(condition_id, set()),
                }
                & blocker_ids
            )
        ]
        observed_by_condition: Dict[str, set[str]] = {}
        if candidate_result is not None:
            candidate_slim = build_slim_result(candidate_result)
            for row in list(candidate_slim.get("condition_table") or []):
                if not isinstance(row, dict):
                    continue
                condition_id = str(
                    row.get("condition_id") or row.get("id") or ""
                ).strip().upper()
                if condition_id:
                    observed_by_condition[condition_id] = (
                        _observed_condition_routes(row)
                    )
        activated_by_condition: Dict[str, List[str]] = {}
        missing_by_condition: Dict[str, List[str]] = {}
        for condition_id in relevant_condition_ids:
            required = required_by_condition[condition_id]
            observed = observed_by_condition.get(condition_id, set())
            activated = sorted(required & observed)
            missing = sorted(required - observed)
            if activated:
                activated_by_condition[condition_id] = activated
            if missing:
                missing_by_condition[condition_id] = missing
        all_required_activated = not bool(missing_by_condition)
        audit["target_route_observations"].append({
            "tx_hash": tx_hash,
            "candidate_result_available": candidate_result is not None,
            "relevant_condition_ids": relevant_condition_ids,
            "required_routes_by_condition": {
                condition_id: sorted(required_by_condition[condition_id])
                for condition_id in relevant_condition_ids
            },
            "observed_routes_by_condition": {
                condition_id: sorted(observed_by_condition.get(condition_id, set()))
                for condition_id in relevant_condition_ids
            },
            "activated_routes_by_condition": activated_by_condition,
            "missing_routes_by_condition": missing_by_condition,
            "all_required_routes_activated": all_required_activated,
        })
        if relevant_condition_ids and not all_required_activated:
            audit["unactivated_repair_target_txs"].append(tx_hash)

    audit["all_repair_target_routes_activated"] = not bool(
        audit["unactivated_repair_target_txs"]
    )
    repair_target_set = set(repair_target_txs)
    for tx_hash, candidate_result in candidate_by_tx.items():
        baseline_result = baseline_by_tx.get(tx_hash)
        if baseline_result is None or _result_is_correct(baseline_result):
            continue
        if not _result_is_repair_improvement(
            baseline_result,
            candidate_result,
        ):
            continue
        flips = _condition_flips(baseline_result, candidate_result)
        flip_ids = {
            str(item.get("condition_id") or "").strip().upper()
            for item in flips
            if str(item.get("condition_id") or "").strip()
        }
        is_repair_target = tx_hash in repair_target_set
        case_audit = case_audit_by_tx.get(tx_hash, {})
        blocker_ids = {
            str(value or "").strip().upper()
            for value in [
                *list(case_audit.get("blocking_condition_ids") or []),
                *list(
                    case_audit.get(
                        "effective_covered_blocking_condition_ids"
                    ) or []
                ),
            ]
            if str(value or "").strip()
        }
        candidate_slim = build_slim_result(candidate_result)
        observed_by_condition: Dict[str, set[str]] = {}
        for row in list(candidate_slim.get("condition_table") or []):
            if not isinstance(row, dict):
                continue
            condition_id = str(
                row.get("condition_id") or row.get("id") or ""
            ).strip().upper()
            if condition_id:
                observed_by_condition[condition_id] = _observed_condition_routes(row)
        activated: Dict[str, List[str]] = {}
        for condition_id in sorted(required_by_condition):
            matched = required_by_condition[condition_id] & observed_by_condition.get(
                condition_id,
                set(),
            )
            if matched:
                activated[condition_id] = sorted(matched)
        baseline_calls = _judge_calls_by_condition(baseline_result)
        candidate_calls = _judge_calls_by_condition(candidate_result)
        effective_input_changed: List[str] = []
        for condition_id in sorted(changed_condition_ids):
            before_call = baseline_calls.get(condition_id)
            after_call = candidate_calls.get(condition_id)
            if before_call is None or after_call is None:
                continue
            if _judge_effective_evidence_fingerprint(
                before_call
            ) != _judge_effective_evidence_fingerprint(after_call):
                effective_input_changed.append(condition_id)

        # Route edits require an observed added/promoted route. Question or
        # input-only edits are attributable through their effective Judge input
        # fingerprint and do not need a fictional route activation.
        attributable_sources = set(activated) | {
            condition_id
            for condition_id in effective_input_changed
            if condition_id not in required_by_condition
        }
        attributable_flips: List[str] = []
        for source_condition in sorted(attributable_sources):
            affected = {
                source_condition,
                *downstream_by_condition.get(source_condition, set()),
            }
            target_flips = flip_ids & blocker_ids if blocker_ids else flip_ids
            attributable_flips.extend(sorted(target_flips & affected))
        attributed = bool(
            is_repair_target
            and attributable_sources
            and set(attributable_flips)
        )
        audit["improved_cases"].append({
            "tx_hash": tx_hash,
            "repair_target": is_repair_target,
            "condition_flips": flips,
            "observed_routes_by_condition": {
                key: sorted(value) for key, value in observed_by_condition.items()
                if key in required_by_condition
            },
            "activated_changed_routes": activated,
            "effective_input_changed_condition_ids": effective_input_changed,
            "attributable_condition_flips": sorted(set(attributable_flips)),
            "attributed_to_changed_route": attributed,
        })
        if not is_repair_target:
            audit["incidental_improved_txs"].append(tx_hash)
        else:
            audit[
                "attributed_improved_txs"
                if attributed
                else "unattributed_improved_txs"
            ].append(tx_hash)
    audit["all_improvements_attributed"] = bool(
        audit["attributed_improved_txs"]
    ) and not bool(audit["unattributed_improved_txs"])
    return audit


def _observed_condition_routes(row: Dict[str, Any]) -> set[str]:
    observed: set[str] = set()
    for meta in list(row.get("view_render_metadata") or []):
        if not isinstance(meta, dict):
            continue
        view = str(meta.get("view") or "").strip()
        if view and int(meta.get("rows_rendered", 0) or 0) > 0:
            observed.add(view)
    for value in list(row.get("selected_view_names") or []):
        if str(value).strip():
            observed.add(str(value).strip())
    for raw in [
        *list(row.get("tool_observations") or []),
        *list(row.get("tool_calls") or []),
    ]:
        if not isinstance(raw, dict):
            continue
        request = raw.get("request") if isinstance(raw.get("request"), dict) else {}
        if (
            raw.get("plan_effect_attribution_eligible") is False
            or str(raw.get("route_owner") or "").strip().lower() == "runtime"
            or request.get("plan_effect_attribution_eligible") is False
            or str(request.get("route_owner") or "").strip().lower() == "runtime"
        ):
            continue
        result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        tool_status = str(
            raw.get("tool_status")
            or raw.get("status")
            or result.get("tool_status")
            or result.get("status")
            or ""
        ).strip().lower()
        if tool_status not in {"ok", "success", "completed"}:
            continue
        tool = str(
            raw.get("tool") or raw.get("tool_name") or raw.get("name") or ""
        ).strip()
        if tool:
            observed.add(tool)
        args = raw.get("args") or raw.get("arguments") or {}
        if isinstance(args, dict):
            view = str(args.get("view") or args.get("view_name") or "").strip()
            if view:
                observed.add(view)
    return observed


def evaluate_candidate_spec(
    *,
    spec: Dict[str, Any],
    args,
    cases: List[Dict[str, Any]],
    hard_guard_cases: List[Dict[str, Any]] | None,
    current_rule: EvolvingRule,
    current_full_results: List[Dict[str, Any]],
    current_guard_full_results: List[Dict[str, Any]] | None,
    round_dir: Path,
    round_index: int,
    evaluator: RegressionEvaluator,
    tool_manifest: Dict[str, Any],
    planner_guidance: Dict[str, Any],
    acceptance_full_results: List[Dict[str, Any]] | None = None,
    acceptance_guard_full_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    name = str(spec.get("name") or "candidate")
    update_kind = str(spec.get("update_kind") or "rule")
    candidate_rule: EvolvingRule = spec["rule"]
    candidate_plan: EvidencePlan | None = spec.get("plan")
    candidate_strategy = str(spec.get("candidate_strategy") or "")
    plan_regenerated = False
    plan_generation_fallback = False
    artifact_group = str(spec.get("artifact_group") or "candidate_artifacts")
    candidate_dir = round_dir / artifact_group / name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_rule_path = candidate_dir / "candidate_rule.json"
    write_json(candidate_rule_path, candidate_rule.to_dict())
    candidate_plan_path = ""
    if bool(spec.get("requires_plan_regeneration")) or (
        candidate_plan is None and update_kind == "rule"
    ) or (
        candidate_plan is None and update_kind == "rule_plan"
    ):
        candidate_plan, plan_generation_fallback = regenerate_candidate_plan(
            candidate_rule,
            args=args,
            planner_guidance=planner_guidance,
            current_rule=spec.get("base_rule") or current_rule,
            current_plan=spec.get("base_plan"),
            plan_patch_condition_ids=spec.get("applied_plan_patch_conditions") or [],
        )
        plan_regenerated = True
        plan_metadata = dict(candidate_plan.metadata or {})
        plan_metadata.setdefault("candidate_strategy", "updated_rule_regenerated_plan")
        plan_metadata.setdefault("derived_from_rule_id", candidate_rule.rule_id)
        plan_metadata.setdefault("derived_from_rule_version", candidate_rule.version)
        if plan_generation_fallback:
            plan_metadata["plan_generation_fallback"] = True
        else:
            plan_metadata["source"] = "llm_regenerated_for_rule_candidate"
        candidate_plan.metadata = plan_metadata

    strategy_overlay = spec.get("candidate_plan_overlay")
    if (
        candidate_plan is not None
        and isinstance(strategy_overlay, EvidencePlan)
    ):
        candidate_plan = apply_plan_strategy_overlay(
            candidate_plan,
            strategy_overlay,
            condition_ids=list(spec.get("applied_plan_patch_conditions") or []),
            allow_dependency_restructure=bool(
                spec.get("allow_dependency_restructure", False)
            ),
        )

    if candidate_plan is not None:
        candidate_plan = normalize_plan_for_runtime_policy(
            candidate_plan,
            args=args,
            reason="candidate_evaluation_artifact",
        )
        dependency_base_plan = (
            spec.get("base_plan")
            if isinstance(spec.get("base_plan"), EvidencePlan)
            else None
        )
        if dependency_base_plan is not None and bool(
            getattr(args, "disable_stateful_bindings", False)
        ):
            dependency_base_plan = normalize_plan_for_runtime_policy(
                dependency_base_plan,
                args=args,
                reason="dependency_validation_baseline",
            )
        current_rule_signatures = _rule_condition_signature_by_id(current_rule)
        candidate_rule_signatures = _rule_condition_signature_by_id(
            candidate_rule
        )
        dependency_changed_ids = sorted(
            {
                condition_id
                for condition_id in (
                    set(current_rule_signatures) | set(candidate_rule_signatures)
                )
                if current_rule_signatures.get(condition_id)
                != candidate_rule_signatures.get(condition_id)
            }
            | {
                str(value or "").strip().upper()
                for value in list(
                    spec.get("applied_plan_patch_conditions") or []
                )
                if str(value or "").strip()
            }
        )
        dependency_validation = dependency_consistency_report(
            candidate_plan,
            previous_plan=dependency_base_plan,
            changed_condition_ids=dependency_changed_ids,
        )
        dependency_validation["preflight"] = dict(
            spec.get("dependency_preflight") or {}
        )
        dependency_validation["pre_viability"] = dict(
            spec.get("pre_viability_dependency_validation") or {}
        )
        dependency_validation["atomic_delta_impacts"] = [
            {
                "delta_id": item.get("delta_id", ""),
                "condition_id": item.get("condition_id", ""),
                "operation": item.get("operation", ""),
                "dependency_impact": dict(
                    item.get("dependency_impact") or {}
                ),
            }
            for item in list(spec.get("selected_atomic_deltas") or [])
            if isinstance(item, dict)
        ]
        write_json(
            candidate_dir / "dependency_validation.json",
            dependency_validation,
        )
        if (
            dependency_validation["dependency_validation_status"]
            != "passed"
        ):
            raise PlanValidationError(
                "Candidate rejected by dependency consistency validation: "
                + dependency_validation["dependency_rejection_reason"]
            )
        PlanValidator().validate_against_rule(
            candidate_plan,
            candidate_rule,
            attack_label=str(
                (candidate_rule.metadata or {}).get("attack_label") or ""
            ),
            previous_plan=dependency_base_plan,
            changed_condition_ids=dependency_changed_ids,
        )
        plan_path = candidate_dir / "candidate_plan.json"
        write_json(plan_path, candidate_plan.to_dict())
        candidate_plan_path = str(plan_path)
        spec["dependency_validation"] = dependency_validation
        plan_locality = dict(candidate_plan.metadata or {})
        print(
            "[FewShot] Candidate plan locality: "
            f"changed_rule={plan_locality.get('changed_rule_condition_ids', [])} "
            f"plan_patch={plan_locality.get('explicit_plan_patch_condition_ids', [])} "
            f"regenerated={plan_locality.get('regenerated_plan_step_condition_ids', [])} "
            f"preserved={plan_locality.get('preserved_unchanged_plan_steps', [])}"
        )

    print(
        f"[FewShot] Evaluating {update_kind} candidate {name}: "
        f"rule={candidate_rule.rule_id} v{candidate_rule.version} "
        f"plan={candidate_plan.plan_id if candidate_plan else '(dynamic/baseline after accept)'}"
    )
    if plan_regenerated:
        print(
            "[FewShot] rule_candidate plan regenerated: "
            f"path={candidate_plan_path} fallback={plan_generation_fallback}"
        )
    print(
        f"[FewShot] candidate judge reuse: "
        f"{'enabled' if args.enable_candidate_judge_reuse else 'disabled'}"
    )
    reuse_full_results = spec.get("reuse_full_results")
    if reuse_full_results is None:
        reuse_full_results = current_full_results
    baseline_plan_for_reuse = (
        spec.get("base_plan")
        if isinstance(spec.get("base_plan"), EvidencePlan)
        else _first_evidence_plan_from_results(reuse_full_results)
    )
    candidate_changed_condition_ids = _candidate_changed_condition_ids(
        spec,
        current_plan=baseline_plan_for_reuse,
        current_rule=current_rule,
    )
    candidate_reuse_results, candidate_reuse_scope = (
        build_candidate_judge_reuse_index(
            reuse_full_results,
            candidate_rule=candidate_rule,
            candidate_plan=candidate_plan,
            baseline_plan=baseline_plan_for_reuse,
            changed_condition_ids=candidate_changed_condition_ids,
            disable_stateful_bindings=bool(args.disable_stateful_bindings),
        )
        if bool(getattr(args, "enable_candidate_judge_reuse", False))
        else ({}, {"enabled": False})
    )
    if bool(getattr(args, "enable_candidate_judge_reuse", False)):
        print(
            "[FewShot] candidate judge reuse scope: "
            f"{stable_json_dumps(candidate_reuse_scope)}"
        )
    reuse_guard_full_results = (
        spec.get("reuse_guard_full_results")
        if spec.get("reuse_guard_full_results") is not None
        else current_guard_full_results or []
    )
    candidate_guard_reuse_results = (
        build_candidate_judge_reuse_index(
            reuse_guard_full_results,
            candidate_rule=candidate_rule,
            candidate_plan=candidate_plan,
            baseline_plan=baseline_plan_for_reuse,
            changed_condition_ids=candidate_changed_condition_ids,
            disable_stateful_bindings=bool(args.disable_stateful_bindings),
        )[0]
        if bool(getattr(args, "enable_candidate_judge_reuse", False))
        else {}
    )

    canary_selection: Dict[str, Any] = {"enabled": False}
    canary_gate: Dict[str, Any] = {"enabled": False, "passed": True}
    canary_full_results: List[Dict[str, Any]] = []
    canary_guard_full_results: List[Dict[str, Any]] = []
    probe_only = bool(spec.get("probe_only", False))
    skip_candidate_canary = bool(spec.get("skip_candidate_canary", False))
    if bool(getattr(args, "enable_candidate_canary", True)) and not skip_candidate_canary:
        canary_selection = build_candidate_canary_selection(
            spec=spec,
            args=args,
            cases=cases,
            hard_guard_cases=hard_guard_cases,
            current_full_results=current_full_results,
            current_guard_full_results=current_guard_full_results,
        )
        if canary_selection.get("train_cases"):
            canary_full_results = run_cases(
                cases=canary_selection["train_cases"],
                rule=candidate_rule,
                rule_source=str(candidate_rule_path),
                tool_manifest=tool_manifest,
                llm_provider=args.llm_provider,
                planner_model=args.planner_model or args.llm_model,
                judge_model=args.judge_model or args.llm_model,
                env_model=args.env_model or args.llm_model,
                use_environment=args.use_environment,
                base_cache_dir=args.base_cache_dir,
                force_rebuild_packet=args.force_rebuild_packet,
                enable_source_tools=args.enable_source_tools,
                source_cache_dir=args.source_cache_dir,
                force_refresh_source=args.force_refresh_source,
                max_view_chars=args.max_view_chars,
                max_context_chars=args.max_context_chars,
                phase=f"round_{round_index:02d}/{name}_canary",
                llm_transcripts_dir=args.llm_transcripts_dir,
                planner_guidance=None,
                fixed_plan=candidate_plan,
                fixed_plan_source=candidate_plan_path,
                reuse_results_by_tx=candidate_reuse_results or None,
                candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
                runtime_early_stop=bool(args.enable_runtime_early_stop),
                retry_failed_conditions=True,
                **runtime_adaptive_kwargs(args),
            )
        if canary_selection.get("guard_cases"):
            canary_guard_full_results = run_cases(
                cases=canary_selection["guard_cases"],
                rule=candidate_rule,
                rule_source=str(candidate_rule_path),
                tool_manifest=tool_manifest,
                llm_provider=args.llm_provider,
                planner_model=args.planner_model or args.llm_model,
                judge_model=args.judge_model or args.llm_model,
                env_model=args.env_model or args.llm_model,
                use_environment=args.use_environment,
                base_cache_dir=args.base_cache_dir,
                force_rebuild_packet=args.force_rebuild_packet,
                enable_source_tools=args.enable_source_tools,
                source_cache_dir=args.source_cache_dir,
                force_refresh_source=args.force_refresh_source,
                max_view_chars=args.max_view_chars,
                max_context_chars=args.max_context_chars,
                phase=f"round_{round_index:02d}/{name}_canary_guard",
                llm_transcripts_dir=args.llm_transcripts_dir,
                planner_guidance=None,
                fixed_plan=candidate_plan,
                fixed_plan_source=candidate_plan_path,
                reuse_results_by_tx=candidate_guard_reuse_results or None,
                candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
                runtime_early_stop=bool(args.enable_runtime_early_stop),
                retry_failed_conditions=True,
                **runtime_adaptive_kwargs(args),
            )
        train_txs = set((canary_selection.get("train_roles") or {}).keys())
        guard_txs = set((canary_selection.get("guard_roles") or {}).keys())
        canary_gate = evaluate_candidate_canary_gate(
            selection=canary_selection,
            baseline_results=[
                item for item in current_full_results
                if str(get_tx_hash(item) or "").strip().lower() in train_txs
            ],
            candidate_results=canary_full_results,
            baseline_guard_results=[
                item for item in list(current_guard_full_results or [])
                if str(get_tx_hash(item) or "").strip().lower() in guard_txs
            ],
            candidate_guard_results=canary_guard_full_results,
        )
        write_json(candidate_dir / "canary_selection.json", {
            key: value for key, value in canary_selection.items()
            if key not in {"train_cases", "guard_cases"}
        })
        write_json(candidate_dir / "canary_gate.json", canary_gate)
        write_json(candidate_dir / "canary_full_results.json", canary_full_results)
        if canary_guard_full_results:
            write_json(
                candidate_dir / "canary_guard_full_results.json",
                canary_guard_full_results,
            )
        print(
            "[FewShot] Candidate canary: "
            f"{stable_json_dumps(canary_gate)}"
        )

    canary_rejected = bool(canary_gate.get("enabled")) and not bool(
        canary_gate.get("passed", True)
    )
    if probe_only:
        candidate_full_results = canary_full_results
        candidate_guard_full_results = canary_guard_full_results
    elif canary_rejected:
        candidate_full_results = canary_full_results
        candidate_guard_full_results = canary_guard_full_results
    else:
        if bool(getattr(args, "enable_candidate_judge_reuse", False)):
            candidate_reuse_results.update(_index_results_by_tx(canary_full_results))
            candidate_guard_reuse_results.update(
                _index_results_by_tx(canary_guard_full_results)
            )
        candidate_full_results = run_cases(
            cases=cases,
            rule=candidate_rule,
            rule_source=str(candidate_rule_path),
            tool_manifest=tool_manifest,
            llm_provider=args.llm_provider,
            planner_model=args.planner_model or args.llm_model,
            judge_model=args.judge_model or args.llm_model,
            env_model=args.env_model or args.llm_model,
            use_environment=args.use_environment,
            base_cache_dir=args.base_cache_dir,
            force_rebuild_packet=args.force_rebuild_packet,
            enable_source_tools=args.enable_source_tools,
            source_cache_dir=args.source_cache_dir,
            force_refresh_source=args.force_refresh_source,
            max_view_chars=args.max_view_chars,
            max_context_chars=args.max_context_chars,
            phase=f"round_{round_index:02d}/{name}",
            llm_transcripts_dir=args.llm_transcripts_dir,
            planner_guidance=None,
            fixed_plan=candidate_plan,
            fixed_plan_source=candidate_plan_path,
            reuse_results_by_tx=candidate_reuse_results or None,
            candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
            runtime_early_stop=bool(args.enable_runtime_early_stop),
            retry_failed_conditions=True,
            **runtime_adaptive_kwargs(args),
        )
        candidate_guard_full_results = []
        if hard_guard_cases:
            candidate_guard_full_results = run_cases(
                cases=hard_guard_cases,
                rule=candidate_rule,
                rule_source=str(candidate_rule_path),
                tool_manifest=tool_manifest,
                llm_provider=args.llm_provider,
                planner_model=args.planner_model or args.llm_model,
                judge_model=args.judge_model or args.llm_model,
                env_model=args.env_model or args.llm_model,
                use_environment=args.use_environment,
                base_cache_dir=args.base_cache_dir,
                force_rebuild_packet=args.force_rebuild_packet,
                enable_source_tools=args.enable_source_tools,
                source_cache_dir=args.source_cache_dir,
                force_refresh_source=args.force_refresh_source,
                max_view_chars=args.max_view_chars,
                max_context_chars=args.max_context_chars,
                phase=f"round_{round_index:02d}/{name}_guard",
                llm_transcripts_dir=args.llm_transcripts_dir,
                planner_guidance=None,
                fixed_plan=candidate_plan,
                fixed_plan_source=candidate_plan_path,
                reuse_results_by_tx=candidate_guard_reuse_results or None,
                candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
                runtime_early_stop=bool(args.enable_runtime_early_stop),
                retry_failed_conditions=True,
                **runtime_adaptive_kwargs(args),
            )

    candidate_slim_results = [build_slim_result(result) for result in candidate_full_results]
    candidate_summary = evaluator.summarize(candidate_full_results).to_dict()
    candidate_guard_slim_results = [
        build_slim_result(result) for result in candidate_guard_full_results
    ]
    candidate_guard_summary = (
        evaluator.summarize(candidate_guard_full_results).to_dict()
        if candidate_guard_full_results
        else {}
    )
    activation_probe_audit = activation_probe_execution_audit(
        spec,
        candidate_slim_results,
    )
    full_runtime_failures: List[Dict[str, Any]] = []
    if not probe_only and not canary_rejected:
        full_runtime_failures.extend(
            collect_candidate_runtime_failures(
                current_full_results,
                candidate_full_results,
                changed_condition_ids=set(candidate_changed_condition_ids),
                role="train",
            )
        )
        full_runtime_failures.extend(
            collect_candidate_runtime_failures(
                list(current_guard_full_results or []),
                candidate_guard_full_results,
                changed_condition_ids=set(candidate_changed_condition_ids),
                role="protected_guard",
            )
        )

    if probe_only:
        probe_passed = (
            bool(canary_gate.get("enabled"))
            and bool(canary_gate.get("passed", False))
            and (
                not bool(activation_probe_audit.get("enabled"))
                or bool(activation_probe_audit.get("all_target_routes_observed"))
            )
        )
        if probe_passed:
            probe_reject_reason = "experimental_probe_only"
        elif bool(activation_probe_audit.get("enabled")) and not bool(
            activation_probe_audit.get("all_target_routes_observed")
        ):
            probe_reject_reason = "activation_probe_target_route_not_observed"
        else:
            probe_reject_reason = str(
                canary_gate.get("reject_reason")
                or "experimental_probe_requires_passing_canary"
            )
        comparison = {
            "accept": False,
            "accept_reason": "",
            "reject_reason": probe_reject_reason,
            "candidate_canary": canary_gate,
            "activation_probe_execution": activation_probe_audit,
            "candidate_evaluation_scope": "experimental_probe_canary",
            "candidate_status": "experimental_probe",
            "probe_passed": probe_passed,
            "promotion_eligible": False,
        }
    elif canary_rejected:
        canary_runtime_failures = list(canary_gate.get("runtime_failures") or [])
        comparison = {
            "accept": False,
            "accept_reason": "",
            "reject_reason": str(canary_gate.get("reject_reason") or "candidate_canary_rejected"),
            "candidate_canary": canary_gate,
            "candidate_evaluation_scope": "canary_only",
            "validation_incomplete": bool(canary_runtime_failures),
            "promotion_eligible": not bool(canary_runtime_failures),
        }
    else:
        # Round search uses the configured evolution policy. Strict policy is
        # reserved for final validation/promotion below.
        candidate_acceptance_args = args
        comparison = compare_train_and_guard(
            evaluator=evaluator,
            args=candidate_acceptance_args,
            current_full_results=acceptance_full_results or current_full_results,
            new_full_results=candidate_full_results,
            current_guard_full_results=(
                acceptance_guard_full_results
                if acceptance_guard_full_results is not None
                else current_guard_full_results
            ),
            new_guard_full_results=candidate_guard_full_results,
            hard_guard_cases=hard_guard_cases,
            acceptance_mode="plan" if update_kind == "plan" else "rule",
        )
        comparison["candidate_canary"] = canary_gate
        comparison["candidate_evaluation_scope"] = "full_after_canary"
    plan_effect_activation = plan_candidate_effect_activation_audit(
        spec=spec,
        base_plan=(
            spec.get("base_plan")
            if isinstance(spec.get("base_plan"), EvidencePlan)
            else baseline_plan_for_reuse
        ),
        candidate_plan=candidate_plan,
        baseline_results=acceptance_full_results or current_full_results,
        candidate_results=candidate_full_results,
    )
    comparison["plan_effect_activation"] = plan_effect_activation
    activation_runtime_failures = [
        *list(canary_gate.get("runtime_failures") or []),
        *full_runtime_failures,
    ]
    unactivated_repair_targets = list(
        plan_effect_activation.get("unactivated_repair_target_txs") or []
    )
    plan_route_not_activated = bool(unactivated_repair_targets)
    has_recorded_semantic_regression = bool(
        list(comparison.get("regressed_cases") or [])
        or list(canary_gate.get("protected_regressions") or [])
        or _guard_regression_rows(comparison)
    )
    unattributed_plan_effect = bool(
        plan_effect_activation.get("effect_change_requires_attribution")
        and not plan_effect_activation.get("attributed_improved_txs")
    )
    plan_activation_contract_failure = bool(
        not activation_runtime_failures
        and unattributed_plan_effect
        and (
            comparison.get("accept")
            or (
                plan_route_not_activated
                and not has_recorded_semantic_regression
            )
        )
    )
    if (
        plan_activation_contract_failure
    ):
        effect_failure_reason = (
            "plan_route_not_activated"
            if plan_route_not_activated
            else "target_blocker_unchanged"
        )
        comparison["accept"] = False
        comparison["accept_reason"] = ""
        comparison["reject_reason"] = (
            "plan_effect_not_attributable"
            if plan_route_not_activated
            else "candidate_ineffective"
        )
        comparison["plan_effect_activation_warning"] = (
            effect_failure_reason
        )
        comparison["plan_effect_failure"] = {
            "owner": "plan",
            "reason": effect_failure_reason,
            "changed_condition_ids": list(
                plan_effect_activation.get("changed_condition_ids") or []
            ),
            "required_routes_by_condition": dict(
                plan_effect_activation.get("required_routes_by_condition") or {}
            ),
            "unattributed_improved_txs": list(
                plan_effect_activation.get("unattributed_improved_txs") or []
            ),
            "unactivated_repair_target_txs": unactivated_repair_targets,
            "semantic_refinement_allowed": False,
        }
        comparison["plan_effect_activation"]["acceptance_gate_applied"] = True
        comparison["plan_effect_activation"]["policy"] = (
            "full_validation_improvement_requires_executed_plan_effect"
        )
    comparison["candidate_runtime_failures"] = [
        *list(canary_gate.get("runtime_failures") or []),
        *full_runtime_failures,
    ]
    if full_runtime_failures:
        comparison["accept"] = False
        comparison["accept_reason"] = ""
        comparison["reject_reason"] = "candidate_validation_incomplete"
        comparison["validation_incomplete"] = True
        comparison["promotion_eligible"] = False
    comparison["acceptance_baseline"] = {
        "source": (
            "incumbent_anchor"
            if acceptance_full_results is not None
            and acceptance_full_results is not current_full_results
            else "current_round"
        ),
        "train_summary": evaluator.summarize(
            acceptance_full_results or current_full_results
        ).to_dict(),
        "guard_summary": (
            evaluator.summarize(
                acceptance_guard_full_results
                if acceptance_guard_full_results is not None
                else current_guard_full_results or []
            ).to_dict()
            if hard_guard_cases
            else {}
        ),
    }
    comparison["changed_artifact"] = update_kind
    comparison["rule_changed"] = candidate_rule.version != current_rule.version
    if update_kind in {"plan", "rule_plan"} and candidate_plan is not None:
        comparison["semantic_plan_changed"] = True
        comparison["metadata_only_plan_change"] = False
        comparison["plan_scope_guard"] = dict((candidate_plan.metadata or {}).get("scope_guard") or {})
    else:
        comparison["semantic_plan_changed"] = bool(
            plan_regenerated or spec.get("semantic_plan_changed", False)
        )
        comparison["metadata_only_plan_change"] = False
        comparison["plan_scope_guard"] = {}
    comparison["plan_changed"] = bool(comparison["semantic_plan_changed"] or plan_regenerated)
    comparison["initial_rule_complexity"] = rule_complexity(current_rule)
    comparison["candidate_rule_complexity"] = rule_complexity(candidate_rule)
    comparison["rule_budget_baseline_relative"] = (
        baseline_relative_rule_budget_decision(
            comparison["initial_rule_complexity"],
            comparison["candidate_rule_complexity"],
        )
    )
    comparison["plan_complexity"] = plan_complexity(candidate_plan)
    comparison["candidate_runtime_stats"] = aggregate_runtime_stats(candidate_full_results)
    if candidate_guard_full_results:
        comparison["candidate_guard_runtime_stats"] = aggregate_runtime_stats(
            candidate_guard_full_results
        )
    comparison["rule_budget_overrun"] = bool(
        comparison["candidate_rule_complexity"].get("over_budget")
    )
    comparison["rule_budget_hard_overrun"] = bool(
        comparison["candidate_rule_complexity"].get("hard_over_budget")
    )
    comparison["rule_budget_soft_warning"] = bool(
        comparison["candidate_rule_complexity"].get("soft_over_budget")
    )
    comparison["candidate_allowed_despite_soft_warning"] = (
        bool(comparison["candidate_rule_complexity"].get("soft_over_budget"))
        and not bool(comparison["candidate_rule_complexity"].get("hard_over_budget"))
    )
    comparison["budget_trace"] = rule_budget_trace(candidate_rule, args=args)
    comparison["rule_budget_rejected"] = bool(
        (candidate_rule.metadata or {}).get("last_rule_budget_rejected")
    )
    plan_only_candidate = bool(update_kind == "plan" and not comparison["rule_changed"])
    comparison["rule_budget_ignored_for_plan_only_candidate"] = bool(
        plan_only_candidate and comparison["rule_budget_hard_overrun"]
    )
    hard_budget_blocks_candidate = bool(
        comparison["rule_budget_hard_overrun"]
        and not plan_only_candidate
        and not comparison["rule_budget_baseline_relative"].get(
            "baseline_relative_grandfathered"
        )
    )
    comparison["rule_budget_hard_rejection_effective"] = hard_budget_blocks_candidate
    if hard_budget_blocks_candidate:
        comparison["accept"] = False
        comparison["accept_reason"] = ""
        comparison["reject_reason"] = "Rejected by hard rule complexity budget."
    comparison["applied_rule_patch_conditions"] = list(
        spec.get("applied_rule_patch_conditions") or []
    )
    comparison["applied_plan_patch_conditions"] = list(
        spec.get("applied_plan_patch_conditions") or []
    )
    comparison["regenerated_plan_from_rule"] = bool(spec.get("regenerated_plan_from_rule"))
    comparison["candidate_generation_priority"] = dict(
        spec.get("candidate_generation_priority") or {}
    )
    comparison["candidate_status"] = str(
        spec.get("candidate_status") or comparison.get("candidate_status") or "supported"
    )
    comparison["probe_only"] = probe_only
    comparison.setdefault("promotion_eligible", not probe_only)
    hypothesis_lifecycle = dict(spec.get("hypothesis_lifecycle") or {})
    if probe_only:
        probe_passed = bool(comparison.get("probe_passed"))
        hypothesis_lifecycle.update({
            "state": "probe_passed" if probe_passed else "probe_rejected",
            "probe_result": "passed" if probe_passed else "rejected",
            "conversion_reason": (
                "eligible_for_separate_supported_confirmation"
                if probe_passed
                else "probe_did_not_pass_canary"
            ),
            "direct_promotion_allowed": False,
        })
    comparison["hypothesis_lifecycle"] = hypothesis_lifecycle
    comparison["partial_acceptance"] = dict(spec.get("partial_acceptance") or {})
    comparison["dependency_validation"] = dict(
        spec.get("dependency_validation") or {}
    )
    if activation_probe_audit.get("enabled"):
        comparison["activation_probe_execution"] = activation_probe_audit

    write_json(
        candidate_dir / "plan_effect_activation.json",
        plan_effect_activation,
    )

    write_json(candidate_dir / "candidate_full_results.json", candidate_full_results)
    write_json(candidate_dir / "candidate_slim_results.json", candidate_slim_results)
    write_json(candidate_dir / "candidate_results.json", candidate_slim_results)
    write_json(candidate_dir / "candidate_summary.json", candidate_summary)
    write_json(candidate_dir / "candidate_case_plans.json", extract_case_plans(candidate_full_results))
    if hard_guard_cases:
        write_json(candidate_dir / "candidate_guard_full_results.json", candidate_guard_full_results)
        write_json(candidate_dir / "candidate_guard_slim_results.json", candidate_guard_slim_results)
        write_json(candidate_dir / "candidate_guard_summary.json", candidate_guard_summary)
        write_json(
            candidate_dir / "guard_comparison.json",
            {
                "guard_comparison": comparison.get("guard_comparison", {}),
                "guard_gate": comparison.get("guard_gate", {}),
            },
        )
    write_json(candidate_dir / "comparison.json", comparison)
    write_json(
        candidate_dir / "candidate_artifact.json",
        {
            "name": name,
            "update_kind": update_kind,
            "candidate_strategy": candidate_strategy,
            "rule": {
                "rule_id": candidate_rule.rule_id,
                "rule_version": candidate_rule.version,
                "rule_source": str(candidate_rule_path),
                "changed": candidate_rule.version != current_rule.version,
            },
            "plan": {
                "plan_id": candidate_plan.plan_id if candidate_plan else "",
                "plan_source": candidate_plan_path,
                "plan_version": (candidate_plan.metadata or {}).get("plan_version")
                if candidate_plan else None,
                "changed": update_kind in {"plan", "rule_plan"} or plan_regenerated,
                "regenerated": plan_regenerated,
                "source": (candidate_plan.metadata or {}).get("source", "")
                if candidate_plan else "",
                "plan_generation_fallback": plan_generation_fallback,
            },
            "complexity": {
                "initial_rule_complexity": rule_complexity(current_rule),
                "candidate_rule_complexity": rule_complexity(candidate_rule),
                "plan_complexity": plan_complexity(candidate_plan),
            },
            "runtime_stats": comparison.get("candidate_runtime_stats", {}),
            "budget_trace": comparison.get("budget_trace", {}),
            "candidate_generation_priority": comparison.get(
                "candidate_generation_priority", {}
            ),
            "applied_rule_patch_conditions": comparison.get(
                "applied_rule_patch_conditions", []
            ),
            "applied_plan_patch_conditions": comparison.get(
                "applied_plan_patch_conditions", []
            ),
            "partial_acceptance": comparison.get("partial_acceptance", {}),
            "candidate_canary": dict(comparison.get("candidate_canary") or {}),
            "candidate_evaluation_scope": str(
                comparison.get("candidate_evaluation_scope") or ""
            ),
            "dependency_validation": comparison.get(
                "dependency_validation", {}
            ),
            "hypothesis_source_signal_ids": list(
                spec.get("hypothesis_source_signal_ids")
                or spec.get("experimental_signal_ids")
                or []
            ),
            "hypothesis_lifecycle": hypothesis_lifecycle,
            "condition_evidence_dependencies": list(
                spec.get("condition_evidence_dependencies") or []
            ),
        },
    )
    print(f"[FewShot] {name} summary: {stable_json_dumps(candidate_summary)}")
    if hard_guard_cases:
        print(
            f"[FewShot] {name} guard summary: "
            f"{stable_json_dumps(candidate_guard_summary)}"
        )
        print(
            f"[FewShot] {name} guard gate: "
            f"{stable_json_dumps(comparison.get('guard_gate', {}))}"
        )
    uncertain_gate = dict(comparison.get("uncertain_gate") or {})
    if uncertain_gate and not bool(uncertain_gate.get("accept", True)):
        if _uncertain_gate_blocks_acceptance(uncertain_gate):
            print(
                f"[FewShot] {name} rejected by uncertain gate: "
                f"{uncertain_gate.get('reject_reason', '')}"
            )
        else:
            print(
                f"[FewShot] {name} uncertain warning: "
                f"{uncertain_gate.get('reject_reason', '')}"
            )
    guard_gate = dict(comparison.get("guard_gate") or {})
    if guard_gate.get("enabled") and not bool(guard_gate.get("accept", True)):
        print(
            f"[FewShot] {name} rejected by hard-negative guard: "
            f"{guard_gate.get('reject_reason', '')}"
        )
    print(f"[FewShot] {name} comparison: {stable_json_dumps(comparison)}")
    return {
        "name": name,
        "update_kind": update_kind,
        "candidate_strategy": candidate_strategy,
        "candidate_status": str(spec.get("candidate_status") or "supported"),
        "probe_only": probe_only,
        "experimental_signal_ids": list(spec.get("experimental_signal_ids") or []),
        "hypothesis_source_signal_ids": list(
            spec.get("hypothesis_source_signal_ids")
            or spec.get("experimental_signal_ids")
            or []
        ),
        "hypothesis_lifecycle": hypothesis_lifecycle,
        "phase3_activation_probe": dict(spec.get("phase3_activation_probe") or {}),
        "condition_evidence_dependencies": list(
            spec.get("condition_evidence_dependencies") or []
        ),
        "plan_strategy_deltas": list(spec.get("plan_strategy_deltas") or []),
        "applied_plan_signal_ids": list(
            spec.get("applied_plan_signal_ids") or []
        ),
        "plan_patch_materialization": dict(
            spec.get("plan_patch_materialization") or {}
        ),
        "candidate_viability": dict(spec.get("candidate_viability") or {}),
        "selected_atomic_deltas": list(
            spec.get("selected_atomic_deltas") or []
        ),
        "rule_delta_synthesis": dict(
            spec.get("rule_delta_synthesis") or {}
        ),
        "dependency_validation": dict(
            spec.get("dependency_validation") or {}
        ),
        "rule": candidate_rule,
        "rule_path": str(candidate_rule_path),
        "plan": candidate_plan,
        "plan_path": candidate_plan_path,
        "full_results": candidate_full_results,
        "slim_results": candidate_slim_results,
        "summary": candidate_summary,
        "plan_regenerated": plan_regenerated,
        "plan_generation_fallback": plan_generation_fallback,
        "applied_rule_patch_conditions": list(spec.get("applied_rule_patch_conditions") or []),
        "applied_plan_patch_conditions": list(spec.get("applied_plan_patch_conditions") or []),
        "regenerated_plan_from_rule": bool(spec.get("regenerated_plan_from_rule")),
        "partial_acceptance": dict(spec.get("partial_acceptance") or {}),
        "guard_enabled": bool(hard_guard_cases),
        "guard_full_results": candidate_guard_full_results,
        "guard_slim_results": candidate_guard_slim_results,
        "guard_summary": candidate_guard_summary,
        "comparison": comparison,
        "candidate_dir": str(candidate_dir),
    }


def candidate_evaluation_summary(item: Dict[str, Any], *, args=None) -> Dict[str, Any]:
    comparison = dict(item.get("comparison") or {})
    outcome = dict(item.get("candidate_outcome") or {})
    outcome_decision = candidate_outcome_decision(item)
    repair_diagnostic = (
        candidate_repair_diagnostic(item, args=args) if args is not None else {}
    )
    return {
        "name": item.get("name", ""),
        "update_kind": item.get("update_kind", ""),
        "candidate_strategy": item.get("candidate_strategy", ""),
        "candidate_status": item.get("candidate_status", "supported"),
        "probe_only": bool(item.get("probe_only")),
        "probe_passed": bool(comparison.get("probe_passed")),
        "promotion_eligible": bool(comparison.get("promotion_eligible", True)),
        "candidate_outcome": dict(item.get("candidate_outcome") or {}),
        "experimental_signal_ids": list(item.get("experimental_signal_ids") or []),
        "hypothesis_source_signal_ids": list(
            item.get("hypothesis_source_signal_ids")
            or item.get("experimental_signal_ids")
            or []
        ),
        "hypothesis_lifecycle": dict(
            item.get("hypothesis_lifecycle")
            or comparison.get("hypothesis_lifecycle")
            or {}
        ),
        "phase3_activation_probe": dict(
            item.get("phase3_activation_probe") or {}
        ),
        "activation_probe_execution": dict(
            comparison.get("activation_probe_execution") or {}
        ),
        "condition_evidence_dependencies": list(
            item.get("condition_evidence_dependencies") or []
        ),
        "plan_strategy_deltas": list(item.get("plan_strategy_deltas") or []),
        "candidate_dir": item.get("candidate_dir", ""),
        "rule_id": item["rule"].rule_id,
        "rule_version": item["rule"].version,
        "plan_id": item["plan"].plan_id if item.get("plan") is not None else "",
        "plan_version": (item["plan"].metadata or {}).get("plan_version")
        if item.get("plan") is not None else None,
        "rule_changed": bool(comparison.get("rule_changed")),
        "plan_changed": bool(comparison.get("plan_changed")),
        "plan_regenerated": bool(item.get("plan_regenerated")),
        "plan_generation_fallback": bool(item.get("plan_generation_fallback")),
        "summary": dict(item.get("summary") or {}),
        "guard_enabled": bool(item.get("guard_enabled")),
        "guard_summary": dict(item.get("guard_summary") or {}),
        "guard_gate": dict(comparison.get("guard_gate") or {}),
        "guard_gate_enabled": bool((comparison.get("guard_gate") or {}).get("enabled")),
        "guard_gate_accept": bool((comparison.get("guard_gate") or {}).get("accept", True)),
        "guard_gate_reject_reason": (comparison.get("guard_gate") or {}).get("reject_reason", ""),
        "uncertain_gate": dict(comparison.get("uncertain_gate") or {}),
        "semantic_plan_changed": bool(comparison.get("semantic_plan_changed")),
        "metadata_only_plan_change": bool(comparison.get("metadata_only_plan_change")),
        "plan_scope_guard": dict(comparison.get("plan_scope_guard") or {}),
        "rule_complexity": rule_complexity(item["rule"]),
        "plan_complexity": plan_complexity(item.get("plan")),
        "initial_rule_complexity": dict(comparison.get("initial_rule_complexity") or {}),
        "candidate_rule_complexity": dict(
            comparison.get("candidate_rule_complexity") or rule_complexity(item["rule"])
        ),
        "rule_budget_overrun": bool(comparison.get("rule_budget_overrun")),
        "rule_budget_hard_overrun": bool(comparison.get("rule_budget_hard_overrun")),
        "rule_budget_soft_warning": bool(comparison.get("rule_budget_soft_warning")),
        "candidate_allowed_despite_soft_warning": bool(
            comparison.get("candidate_allowed_despite_soft_warning")
        ),
        "budget_trace": dict(comparison.get("budget_trace") or {}),
        "rule_budget_rejected": bool(comparison.get("rule_budget_rejected")),
        "applied_rule_patch_conditions": list(
            item.get("applied_rule_patch_conditions")
            or comparison.get("applied_rule_patch_conditions")
            or []
        ),
        "applied_plan_patch_conditions": list(
            item.get("applied_plan_patch_conditions")
            or comparison.get("applied_plan_patch_conditions")
            or []
        ),
        "partial_acceptance": dict(
            item.get("partial_acceptance")
            or comparison.get("partial_acceptance")
            or {}
        ),
        "regenerated_plan_from_rule": bool(
            item.get("regenerated_plan_from_rule")
            or comparison.get("regenerated_plan_from_rule")
        ),
        "candidate_judge_reuse": dict(
            (comparison.get("candidate_runtime_stats") or {}).get("candidate_judge_reuse")
            or {}
        ),
        "candidate_canary": dict(comparison.get("candidate_canary") or {}),
        "candidate_evaluation_scope": str(
            comparison.get("candidate_evaluation_scope") or ""
        ),
        "dependency_validation": dict(
            item.get("dependency_validation")
            or comparison.get("dependency_validation")
            or {}
        ),
        "runtime_early_stop": dict(
            (comparison.get("candidate_runtime_stats") or {}).get("runtime_early_stop")
            or {}
        ),
        "adaptive_evidence": dict(
            (comparison.get("candidate_runtime_stats") or {}).get("adaptive_evidence")
            or {}
        ),
        "parallel_judge": dict(
            (comparison.get("candidate_runtime_stats") or {}).get("parallel_judge")
            or {}
        ),
        "accept": outcome_decision == "accept",
        "accept_reason": (
            str(outcome.get("detail_code") or comparison.get("accept_reason") or "")
            if outcome_decision == "accept"
            else ""
        ),
        "reject_reason": (
            str(outcome.get("detail_code") or comparison.get("reject_reason") or "")
            if outcome_decision == "reject"
            else ""
        ),
        "repair_eligible_diagnostic": repair_diagnostic,
        "repair_diagnostic_reason": repair_diagnostic.get("reason", ""),
        "comparison_path": str(Path(item.get("candidate_dir", "")) / "comparison.json"),
    }


def empty_rejected_update_memory() -> Dict[str, Any]:
    return normalize_rejected_update_memory({"entries": []})


def update_rejected_update_memory(
    memory: Dict[str, Any],
    *,
    review_bundle: Dict[str, Any],
    candidate_evaluations: List[Dict[str, Any]],
    candidate_viability: Dict[str, Any] | None = None,
    round_index: int,
    attack_label: str,
    max_entries: int = 20,
) -> Dict[str, Any]:
    """Accumulate compact taboo memory from rejected abstract update directions."""
    normalized = normalize_rejected_update_memory(memory, max_entries=max_entries)
    existing_entries = list(normalized.get("entries") or [])
    existing_preflight_gaps = list(normalized.get("preflight_gaps") or [])
    existing_construction_rejections = list(
        normalized.get("construction_rejections") or []
    )
    matrix = dict((review_bundle or {}).get("update_signal_matrix") or {})
    signals = [
        dict(signal)
        for signal in list(matrix.get("signals") or [])
        if isinstance(signal, dict)
    ]
    signal_by_id = {
        str(signal.get("signal_id") or ""): signal
        for signal in signals
        if str(signal.get("signal_id") or "")
    }
    new_entries: List[Dict[str, Any]] = []
    for item in list(candidate_evaluations or []):
        comparison = dict(item.get("comparison") or {})
        if candidate_outcome_decision(item) == "accept":
            continue
        new_entries.extend(
            _rejected_candidate_memory_entries(
                item,
                signals=signals,
                round_index=round_index,
                attack_label=attack_label,
            )
        )

    new_preflight_gaps: List[Dict[str, Any]] = []
    for audit in list((candidate_viability or {}).get("candidates") or []):
        if not isinstance(audit, dict) or bool(audit.get("viable")):
            continue
        update_kind = str(audit.get("update_kind") or "").strip().lower()
        applied_signal_ids = {
            str(value)
            for value in list(audit.get("applied_signal_ids") or [])
            if str(value)
        }
        canonical_target_ids = {
            str(signal_by_id[signal_id].get("condition_id") or "").strip().upper()
            for signal_id in applied_signal_ids
            if signal_id in signal_by_id
            and str(
                signal_by_id[signal_id].get("update_target") or ""
            ).strip().lower() == update_kind
            and str(signal_by_id[signal_id].get("condition_id") or "").strip()
        }
        declared_changed_ids = {
            str(condition_id or "").strip().upper()
            for condition_id in list(audit.get("changed_condition_ids") or [])
            if str(condition_id or "").strip()
        }
        effective_changed_ids = {
            str(condition_id or "").strip().upper()
            for condition_id in list(
                audit.get("effective_changed_condition_ids") or []
            )
            if str(condition_id or "").strip()
        }
        target_ids = (
            canonical_target_ids or declared_changed_ids or effective_changed_ids
        )
        uncovered = sorted({
            str(condition_id or "").strip().upper()
            for case in list(audit.get("case_audits") or [])
            if isinstance(case, dict)
            for condition_id in list(case.get("uncovered_blocking_condition_ids") or [])
            if str(condition_id or "").strip()
            and (
                not canonical_target_ids
                or str(condition_id or "").strip().upper()
                in canonical_target_ids
            )
        })
        evidence_capabilities: Dict[tuple[str, str], Dict[str, Any]] = {}
        for case in list(audit.get("case_audits") or []):
            if not isinstance(case, dict):
                continue
            for dependency in list(
                case.get("condition_evidence_dependency_audit") or []
            ):
                if not isinstance(dependency, dict) or bool(dependency.get("complete")):
                    continue
                condition_id = str(
                    dependency.get("condition_id") or ""
                ).strip().upper()
                capability_id = str(
                    dependency.get("capability_id") or ""
                ).strip().lower()
                if not condition_id or not capability_id:
                    continue
                if canonical_target_ids and condition_id not in canonical_target_ids:
                    continue
                evidence_capabilities[(condition_id, capability_id)] = {
                    "condition_id": condition_id[:32],
                    "capability_id": capability_id[:96],
                    "uncovered_components": sorted({
                        str(value or "").strip().lower()[:48]
                        for value in list(
                            dependency.get("uncovered_components") or []
                        )
                        if str(value or "").strip()
                    })[:6],
                }
        uncovered_evidence = list(evidence_capabilities.values())[:12]
        if not uncovered and not uncovered_evidence:
            continue
        changed = sorted(target_ids)
        new_preflight_gaps.append({
            "source_round": int(round_index),
            "source_candidate": str(audit.get("candidate_name") or "")[:80],
            "update_kind": update_kind[:24],
            "changed_condition_ids": changed[:12],
            "uncovered_blocker_ids": uncovered[:12],
            "uncovered_evidence_capabilities": uncovered_evidence,
            "reason": str(audit.get("reason") or "")[:240],
            "required_next_direction": (
                (
                    "Generate an executable owner-correct update only for canonical "
                    "conditions "
                    + ", ".join(changed)
                    + "."
                )
                if canonical_target_ids
                else (
                    "Retain only compatible prior edits and add an executable, "
                    "owner-correct strategy for "
                    + (
                        f"blockers {', '.join(uncovered)}"
                        if uncovered
                        else "the missing evidence capabilities"
                    )
                    + "."
                )
            )[:280],
        })

    new_construction_rejections = [
        {
            **dict(item),
            "source_round": int(round_index),
        }
        for item in list(
            (candidate_viability or {}).get("pre_candidate_rejections") or []
        )
        if isinstance(item, dict)
    ]

    merged: Dict[tuple[str, str, str, str, str, str, str], Dict[str, Any]] = {}
    for entry in existing_entries + new_entries:
        feature = " ".join(str(entry.get("abstract_feature") or "").split())
        if not feature:
            continue
        entry = dict(entry)
        entry["abstract_feature"] = feature[:280]
        key = _rejected_memory_key(entry)
        merged[key] = entry

    entries = list(merged.values())[-max_entries:]
    for index, entry in enumerate(entries, start=1):
        entry["memory_id"] = f"mem_{index:03d}"
    preflight_by_key: Dict[tuple[str, str, tuple[str, ...], tuple[str, ...]], Dict[str, Any]] = {}
    for gap in [*existing_preflight_gaps, *new_preflight_gaps]:
        key = (
            str(gap.get("source_candidate") or ""),
            str(gap.get("update_kind") or ""),
            tuple(sorted(str(value) for value in gap.get("uncovered_blocker_ids", []))),
            tuple(sorted(
                str(value.get("capability_id") or "")
                for value in list(gap.get("uncovered_evidence_capabilities") or [])
                if isinstance(value, dict)
            )),
        )
        preflight_by_key[key] = dict(gap)
    construction_by_key: Dict[tuple[str, str, tuple[str, ...]], Dict[str, Any]] = {}
    for rejection in [
        *existing_construction_rejections,
        *new_construction_rejections,
    ]:
        key = (
            str(rejection.get("owner") or "plan"),
            str(rejection.get("reason") or ""),
            tuple(sorted(
                str(value or "").strip().upper()
                for value in list(rejection.get("scope_target_steps") or [])
                if str(value or "").strip()
            )),
        )
        construction_by_key[key] = dict(rejection)
    return normalize_rejected_update_memory(
        {
            "entries": entries,
            "preflight_gaps": list(preflight_by_key.values())[-12:],
            "construction_rejections": list(
                construction_by_key.values()
            )[-12:],
        },
        max_entries=max_entries,
    )


def _rejected_candidate_memory_entries(
    item: Dict[str, Any],
    *,
    signals: List[Dict[str, Any]],
    round_index: int,
    attack_label: str,
) -> List[Dict[str, Any]]:
    update_kind = str(item.get("update_kind") or "")
    comparison = dict(item.get("comparison") or {})
    outcome = dict(item.get("candidate_outcome") or {})
    if bool(outcome.get("runtime_failure")) or list(
        comparison.get("candidate_runtime_failures")
        or (comparison.get("candidate_canary") or {}).get("runtime_failures")
        or []
    ):
        return []
    if update_kind == "rule_plan":
        raw_changed_conditions = [
            *list(item.get("applied_rule_patch_conditions") or []),
            *list(item.get("applied_plan_patch_conditions") or []),
        ]
    else:
        raw_changed_conditions = (
            item.get("applied_rule_patch_conditions")
            if update_kind == "rule"
            else item.get("applied_plan_patch_conditions")
        )
    changed_conditions = {
        str(condition_id).upper().strip()
        for condition_id in list(raw_changed_conditions or [])
        if str(condition_id).strip()
    }
    exact_signal_ids = {
        str(signal_id)
        for delta in list(item.get("selected_atomic_deltas") or [])
        if isinstance(delta, dict)
        for signal_id in list(delta.get("applied_signal_ids") or [])
        if str(signal_id)
    } | {
        str(signal_id)
        for signal_id in list(item.get("applied_plan_signal_ids") or [])
        if str(signal_id)
    }
    candidate_signals = [
        signal
        for signal in signals
        if (
            str(signal.get("signal_id") or "") in exact_signal_ids
            if exact_signal_ids
            else str(signal.get("generalization_status") or "") == "supported"
        )
        and (
            str(signal.get("update_target") or "") == update_kind
            or (
                update_kind == "rule_plan"
                and str(signal.get("update_target") or "") in {"rule", "plan"}
            )
        )
        and (
            not changed_conditions
            or str(signal.get("condition_id") or "").upper().strip()
            in changed_conditions
        )
    ]
    reject_reason = str(
        outcome.get("detail_code")
        or comparison.get("reject_reason")
        or (comparison.get("guard_gate") or {}).get("reject_reason")
        or "candidate_rejected"
    )
    rejection_type = str(
        outcome.get("terminal_reason") or _candidate_rejection_type(comparison)
    )
    boundary_hint = _candidate_rejection_boundary_hint(
        comparison,
        update_kind,
        outcome=outcome,
    )
    if not candidate_signals:
        causal_audit = _candidate_rejection_causal_audit(item, comparison)
        return _fallback_rejected_candidate_memory_entries(
            item,
            changed_conditions=changed_conditions,
            update_kind=update_kind,
            comparison=comparison,
            round_index=round_index,
            attack_label=attack_label,
            rejection_type=rejection_type,
            reject_reason=reject_reason,
            boundary_hint=boundary_hint,
            causal_audit=causal_audit,
        )
    refinement = dict(outcome.get("refinement_feedback") or {})
    regressed_condition_ids = {
        str(value or "").strip().upper()
        for value in list(refinement.get("regressed_condition_ids") or [])
        if str(value or "").strip()
    }
    entries: List[Dict[str, Any]] = []
    for signal in candidate_signals:
        condition_id = str(signal.get("condition_id") or "").strip().upper()
        signal_rejection_type = rejection_type
        if (
            rejection_type == "protected_regression"
            and regressed_condition_ids
            and condition_id not in regressed_condition_ids
        ):
            signal_rejection_type = "candidate_ineffective"
        signal_causal_audit = _candidate_rejection_causal_audit(
            item,
            comparison,
            condition_id=condition_id,
            signal_id=str(signal.get("signal_id") or ""),
        )
        entries.append(_memory_entry_from_signal(
            signal,
            round_index=round_index,
            attack_label=attack_label,
            source_candidate=str(item.get("name") or ""),
            rejection_type=signal_rejection_type,
            reject_reason=reject_reason,
            boundary_hint=boundary_hint,
            causal_audit=signal_causal_audit,
        ))
    return entries


def _fallback_rejected_candidate_memory_entries(
    item: Dict[str, Any],
    *,
    changed_conditions: set[str],
    update_kind: str,
    comparison: Dict[str, Any],
    round_index: int,
    attack_label: str,
    rejection_type: str,
    reject_reason: str,
    boundary_hint: str,
    causal_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Record a candidate-level taboo even when no supported signal survived."""
    if update_kind not in {"rule", "plan"}:
        return []
    condition_ids = sorted(changed_conditions) or [""]
    delta = dict((comparison or {}).get("delta") or {})
    tradeoff = dict((comparison or {}).get("acceptance_tradeoff") or {})
    feature = (
        f"Rejected {update_kind} candidate changed condition "
        "{condition_id}; rejection_type={rejection_type}; "
        f"delta_errors={delta.get('errors', '')}; delta_fp={delta.get('fp', '')}; "
        f"delta_fn={delta.get('fn', '')}; fixed_fp={tradeoff.get('fixed_fp', '')}; "
        f"new_fn={tradeoff.get('new_fn', '')}."
    )
    entries: List[Dict[str, Any]] = []
    for condition_id in condition_ids[:8]:
        entries.append(
            {
                "attack_label": normalize_attack_label(attack_label, default=""),
                "source_round": int(round_index),
                "source_candidate": str(item.get("name") or ""),
                "source_signal_id": "candidate_level_rejection",
                "source_error_type": "",
                "update_kind": update_kind,
                "condition_id": condition_id,
                "direction": "avoid_repeating_rejected_candidate",
                "strategy_operation": "",
                "capability_id": "",
                "abstract_feature": feature.format(
                    condition_id=condition_id or "unknown",
                    rejection_type=rejection_type,
                ),
                "rejection_type": rejection_type,
                "reject_reason": reject_reason,
                "boundary_hint": boundary_hint,
                "rejection_causal_audit": causal_audit,
            }
        )
    return entries


def _memory_entry_from_signal(
    signal: Dict[str, Any],
    *,
    round_index: int,
    attack_label: str,
    source_candidate: str,
    rejection_type: str,
    reject_reason: str,
    boundary_hint: str,
    causal_audit: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "attack_label": normalize_attack_label(attack_label, default=""),
        "source_round": int(round_index),
        "source_candidate": source_candidate,
        "source_signal_id": str(signal.get("signal_id") or ""),
        "source_error_type": str(signal.get("source_error_type") or ""),
        "update_kind": str(signal.get("update_target") or ""),
        "condition_id": str(signal.get("condition_id") or ""),
        "direction": str(signal.get("direction") or ""),
        "strategy_operation": _signal_strategy_operation(signal),
        "capability_id": _signal_capability_id(signal),
        "abstract_feature": str(signal.get("abstract_feature") or ""),
        "rejection_type": rejection_type,
        "reject_reason": reject_reason,
        "boundary_hint": boundary_hint,
        "rejection_causal_audit": causal_audit,
    }


def _candidate_rejection_causal_audit(
    item: Dict[str, Any],
    comparison: Dict[str, Any],
    *,
    condition_id: str = "",
    signal_id: str = "",
) -> Dict[str, Any]:
    """Preserve component-level failure causes without transaction identifiers."""
    canary = dict((comparison or {}).get("candidate_canary") or {})
    outcome = dict(item.get("candidate_outcome") or {})
    refinement = dict(outcome.get("refinement_feedback") or {})
    rejection_gate = str(
        outcome.get("terminal_reason") or _candidate_rejection_type(comparison)
    )
    regressions: List[Dict[str, Any]] = []
    raw_regressions = [
        *list(refinement.get("regressions") or []),
        *list(canary.get("protected_regressions") or []),
        *_guard_regression_rows(comparison),
    ]
    seen_regressions: set[tuple[str, str, str, str]] = set()
    for raw in raw_regressions:
        if not isinstance(raw, dict):
            continue
        raw_flips = [
            dict(flip)
            for flip in list(raw.get("condition_flips") or [])
            if isinstance(flip, dict)
        ]
        if condition_id and raw_flips and condition_id not in {
            str(flip.get("condition_id") or "").strip().upper()
            for flip in raw_flips
        }:
            continue
        key = (
            str(raw.get("tx_hash") or "").strip().lower(),
            str(raw.get("role") or ""),
            str(raw.get("baseline_verdict") or raw.get("old_pred") or ""),
            str(raw.get("candidate_verdict") or raw.get("new_pred") or ""),
        )
        if key in seen_regressions:
            continue
        seen_regressions.add(key)
        regressions.append({
            "role": str(raw.get("role") or "")[:40],
            "ground_truth": str(raw.get("ground_truth") or "")[:24],
            "baseline_verdict": str(
                raw.get("baseline_verdict") or raw.get("old_pred") or ""
            )[:24],
            "candidate_verdict": str(
                raw.get("candidate_verdict") or raw.get("new_pred") or ""
            )[:24],
            "condition_flips": [
                {
                    "condition_id": str(flip.get("condition_id") or "")[:32],
                    "baseline_answer": flip.get("baseline_answer"),
                    "candidate_answer": flip.get("candidate_answer"),
                }
                for flip in raw_flips[:8]
                if isinstance(flip, dict)
                and (
                    not condition_id
                    or str(flip.get("condition_id") or "").strip().upper()
                    == condition_id
                )
            ],
        })
        if len(regressions) >= 8:
            break
    plan_route_not_activated = bool(
        outcome.get("detail_code") == "plan_route_not_activated"
        or rejection_gate == "plan_route_not_activated"
    )
    semantic_regressions = [] if plan_route_not_activated else regressions
    regressed_condition_ids = {
        str(value or "").strip().upper()
        for value in list(refinement.get("regressed_condition_ids") or [])
        if str(value or "").strip()
    }
    improved_condition_ids = {
        str(value or "").strip().upper()
        for value in list(refinement.get("improved_condition_ids") or [])
        if str(value or "").strip()
    }
    condition_effect = (
        "regressed"
        if condition_id and condition_id in regressed_condition_ids
        else "improved"
        if condition_id and condition_id in improved_condition_ids
        else "unchanged_or_unattributed"
        if condition_id
        else "candidate_level"
    )
    return {
        "candidate_terminal_reason": rejection_gate,
        "condition_scope": condition_id,
        "condition_effect": condition_effect,
        "candidate_update_kind": str(item.get("update_kind") or "")[:24],
        "rule_condition_ids": sorted({
            str(value or "").strip().upper()
            for value in list(item.get("applied_rule_patch_conditions") or [])
            if str(value or "").strip()
        })[:12],
        "plan_condition_ids": sorted({
            str(value or "").strip().upper()
            for value in list(item.get("applied_plan_patch_conditions") or [])
            if str(value or "").strip()
        })[:12],
        "rule_components": [
            {
                "delta_id": str(delta.get("delta_id") or "")[:96],
                "condition_id": str(delta.get("condition_id") or "")[:32],
                "operation": str(delta.get("operation") or "")[:48],
                "semantic_direction": str(
                    delta.get("semantic_direction") or ""
                )[:24],
                "source_signal_ids": [
                    str(value)[:96]
                    for value in list(delta.get("applied_signal_ids") or [])[:8]
                ],
            }
            for delta in list(item.get("selected_atomic_deltas") or [])[:12]
            if isinstance(delta, dict)
            and (
                not condition_id
                or str(delta.get("condition_id") or "").strip().upper()
                == condition_id
            )
            and (
                not signal_id
                or signal_id in {
                    str(value)
                    for value in list(delta.get("applied_signal_ids") or [])
                }
            )
        ],
        "plan_source_signal_ids": [
            str(value)[:96]
            for value in list(item.get("applied_plan_signal_ids") or [])[:12]
            if not signal_id or str(value) == signal_id
        ],
        "rejection_gate": rejection_gate,
        "plan_effect_failure": dict(comparison.get("plan_effect_failure") or {}),
        "regression_delta": (
            {} if plan_route_not_activated else dict(comparison.get("delta") or {})
        ),
        "guard_attribution": (
            {}
            if plan_route_not_activated
            else dict(comparison.get("guard_gate") or {})
        ),
        "protected_regression_count": len(semantic_regressions),
        "protected_regressions": semantic_regressions,
        "unattributed_runtime_regressions": (
            regressions if plan_route_not_activated else []
        ),
        "unattributed_runtime_regression_delta": (
            dict(comparison.get("delta") or {})
            if plan_route_not_activated
            else {}
        ),
        "unattributed_runtime_guard_attribution": (
            dict(comparison.get("guard_gate") or {})
            if plan_route_not_activated
            else {}
        ),
    }


def _candidate_rejection_type(comparison: Dict[str, Any]) -> str:
    canary = dict(comparison.get("candidate_canary") or {})
    if list(
        comparison.get("candidate_runtime_failures")
        or canary.get("runtime_failures")
        or []
    ):
        return "candidate_validation_incomplete"
    if str(comparison.get("reject_reason") or "") == (
        "plan_effect_not_attributable"
    ):
        return "plan_route_not_activated"
    if bool(canary.get("enabled")) and not bool(canary.get("passed", True)):
        return "candidate_canary_rejection"
    guard_gate = dict(comparison.get("guard_gate") or {})
    uncertain_gate = dict(comparison.get("uncertain_gate") or {})
    if guard_gate.get("enabled") and not bool(guard_gate.get("accept", True)):
        return "hard_negative_guard_rejection"
    if (
        uncertain_gate
        and not bool(uncertain_gate.get("accept", True))
        and _uncertain_gate_blocks_acceptance(uncertain_gate)
    ):
        return "uncertain_gate_rejection"
    delta = dict(comparison.get("delta") or {})
    if int(delta.get("fp", 0) or 0) > 0:
        return "new_fp_regression"
    if int(delta.get("errors", 0) or 0) > 0:
        return "error_regression"
    return "candidate_rejected"


def _candidate_rejection_boundary_hint(
    comparison: Dict[str, Any],
    update_kind: str,
    *,
    outcome: Dict[str, Any] | None = None,
) -> str:
    canary = dict(comparison.get("candidate_canary") or {})
    classified = dict(outcome or {})
    if (
        classified.get("detail_code") == "plan_route_not_activated"
        or (
            not classified
            and _candidate_rejection_type(comparison) == "plan_route_not_activated"
        )
    ):
        return (
            "Keep this direction owned by Plan and retry only with an "
            "execution-capable refinement of the existing evidence route; "
            "do not learn a semantic boundary from unattributed verdict flips."
        )
    if bool(canary.get("enabled")) and not bool(canary.get("passed", True)):
        return (
            "Retry only with a new supported boundary that fixes a selected "
            "error without regressing protected positive, negative, or guard canaries."
        )
    delta = dict(comparison.get("delta") or {})
    guard_gate = dict(comparison.get("guard_gate") or {})
    if guard_gate.get("enabled") and not bool(guard_gate.get("accept", True)):
        return (
            "Retry only with a boundary that preserves fixed hard-negative guard "
            f"cases for this {update_kind} update."
        )
    if int(delta.get("fp", 0) or 0) > 0:
        return (
            "Retry only with an explicit target-vs-non-target boundary or "
            "exclusion that prevents the new false positives."
        )
    if int(delta.get("fn", 0) or 0) > 0:
        return (
            "Retry only if the narrowed condition keeps protected target "
            "positives detectable."
        )
    return "Retry only if a new supported signal explains why this rejection no longer applies."


def _rejected_memory_key(
    entry: Dict[str, Any],
) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(entry.get("attack_label") or ""),
        str(entry.get("update_kind") or ""),
        str(entry.get("condition_id") or "").upper().strip(),
        str(entry.get("direction") or "").lower().strip(),
        str(entry.get("strategy_operation") or "").lower().strip(),
        str(entry.get("capability_id") or "").lower().strip(),
        " ".join(str(entry.get("abstract_feature") or "").lower().split()),
    )


def select_candidate_evaluation(
    evaluations: List[Dict[str, Any]],
    *,
    accepted_only: bool,
) -> Dict[str, Any] | None:
    candidates = [
        item
        for item in evaluations
        if not bool(item.get("probe_only"))
        and bool((item.get("comparison") or {}).get("promotion_eligible", True))
        and (
            not accepted_only
            or candidate_outcome_decision(item) == "accept"
        )
    ]
    if not candidates:
        return None

    kind_preference = {"plan": 0, "rule": 1}

    def score(item: Dict[str, Any]) -> tuple:
        summary = item.get("summary") or {}
        guard_summary = item.get("guard_summary") or {}
        complexity = plan_complexity(item.get("plan"))
        rule_cost = rule_complexity(item["rule"]) if item.get("rule") is not None else {}
        soft_budget_penalty = len(rule_cost.get("soft_over_budget_reasons", []) or [])
        hard_budget_penalty = 1 if rule_cost.get("hard_over_budget") else 0
        return (
            -int(summary.get("correct", 0) or 0),
            int(summary.get("errors", 10**9)),
            int(summary.get("fn", 10**9)),
            int(summary.get("fp", 10**9)),
            int(summary.get("uncertain", 10**9)),
            int(guard_summary.get("fp", 0)),
            int(guard_summary.get("errors", 0)),
            int(guard_summary.get("uncertain", 0)),
            hard_budget_penalty,
            soft_budget_penalty,
            kind_preference.get(str(item.get("update_kind") or ""), 9),
            int(complexity.get("judge_step_count", 10**9)),
            int(complexity.get("total_max_followups", 10**9)),
            int(complexity.get("source_followup_step_count", 10**9)),
            int(complexity.get("allowed_followup_view_count_total", 10**9)),
            str(item.get("name") or ""),
        )

    return sorted(candidates, key=score)[0]


def attempt_rule_partial_candidate_acceptance(
    *,
    args,
    candidate_evaluations: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
    hard_guard_cases: List[Dict[str, Any]] | None,
    current_rule: EvolvingRule,
    current_plan: EvidencePlan | None,
    current_full_results: List[Dict[str, Any]],
    current_guard_full_results: List[Dict[str, Any]] | None,
    round_dir: Path,
    round_index: int,
    evaluator: RegressionEvaluator,
    tool_manifest: Dict[str, Any],
    acceptance_full_results: List[Dict[str, Any]] | None = None,
    acceptance_guard_full_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    max_deltas = max(0, int(getattr(args, "rule_partial_max_deltas", 6) or 0))
    max_evaluations = max(
        0,
        int(getattr(args, "rule_partial_max_evaluations", 10) or 0),
    )
    delta_reports: List[Dict[str, Any]] = []
    parent_reports: List[Dict[str, Any]] = []
    partial_evaluations: List[Dict[str, Any]] = []

    rejected_rule_candidates = [
        item
        for item in candidate_evaluations
        if str(item.get("update_kind") or "") == "rule"
        and not bool(item.get("probe_only"))
        and not bool((item.get("comparison") or {}).get("accept"))
    ]
    for parent in rejected_rule_candidates:
        parent_name = str(parent.get("name") or "rule_candidate")
        report: Dict[str, Any] = {
            "parent_candidate": parent_name,
            "parent_candidate_dir": str(parent.get("candidate_dir") or ""),
            "stage": RULE_PARTIAL_ACCEPTANCE_STAGE,
            "eligible": False,
            "reason": "",
            "evaluated_subsets": [],
            "materialization_failures": [],
        }
        lineage_report = attributed_rule_delta_lineage_report(
            current_rule,
            parent["rule"],
            list(parent.get("selected_atomic_deltas") or []),
        )
        report["atomic_delta_lineage"] = {
            key: value
            for key, value in lineage_report.items()
            if key != "deltas"
        }
        if not lineage_report["valid"]:
            report["reason"] = "attributed_atomic_delta_lineage_unavailable"
            parent_reports.append(report)
            continue
        deltas = list(lineage_report["deltas"])

        delta_report = {
            "parent_candidate": parent_name,
            "base_rule_version": current_rule.version,
            "parent_rule_version": parent["rule"].version,
            "delta_count": len(deltas),
            "deltas": [delta.to_dict() for delta in deltas],
            "ignored_non_condition_fields": [
                field
                for field, before, after in (
                    ("description", current_rule.description, parent["rule"].description),
                    (
                        "decision_policy",
                        current_rule.decision_policy,
                        parent["rule"].decision_policy,
                    ),
                )
                if before != after
            ],
        }
        delta_reports.append(delta_report)
        report["delta_count"] = len(deltas)
        report["delta_ids"] = [delta.delta_id for delta in deltas]
        if len(deltas) < 2:
            report["reason"] = "fewer_than_two_condition_deltas"
            parent_reports.append(report)
            continue
        if max_deltas <= 0 or len(deltas) > max_deltas:
            report["reason"] = "condition_delta_count_exceeds_limit"
            report["max_deltas"] = max_deltas
            parent_reports.append(report)
            continue

        subsets = generate_rule_delta_subsets(
            [delta.delta_id for delta in deltas],
            max_evaluations=max_evaluations,
        )
        if not subsets:
            report["reason"] = "no_subset_within_evaluation_budget"
            parent_reports.append(report)
            continue
        report["eligible"] = True
        report["reason"] = "bounded_subset_search"
        report["planned_subset_count"] = len(subsets)

        parent_applied_plan_conditions = {
            str(item or "").strip().upper()
            for item in list(parent.get("applied_plan_patch_conditions") or [])
        }
        for subset_index, subset in enumerate(subsets, start=1):
            accepted_delta_ids = list(subset["accepted_delta_ids"])
            accepted_delta_set = set(accepted_delta_ids)
            if not rule_delta_subset_preserves_joint_lineage(
                deltas,
                accepted_delta_ids,
            ):
                report["materialization_failures"].append({
                    "stage": RULE_PARTIAL_ACCEPTANCE_STAGE,
                    "parent_candidate": parent_name,
                    "search_strategy": subset["search_strategy"],
                    "accepted_delta_ids": accepted_delta_ids,
                    "omitted_delta_ids": list(subset["omitted_delta_ids"]),
                    "subset_index": subset_index,
                    "subset_count": len(subsets),
                    "error": "joint_resolution_group_cannot_be_split",
                })
                continue
            dependency_preflight = rule_delta_dependency_preflight(
                base_plan=current_plan,
                deltas=deltas,
                accepted_delta_ids=accepted_delta_ids,
            )
            accepted_condition_ids = [
                str(delta.condition_id)
                for delta in deltas
                if delta.delta_id in accepted_delta_set
            ]
            partial_context = {
                "stage": RULE_PARTIAL_ACCEPTANCE_STAGE,
                "parent_candidate": parent_name,
                "search_strategy": subset["search_strategy"],
                "accepted_delta_ids": accepted_delta_ids,
                "omitted_delta_ids": list(subset["omitted_delta_ids"]),
                "subset_index": subset_index,
                "subset_count": len(subsets),
                "dependency_preflight": dependency_preflight,
            }
            if dependency_preflight["dependency_validation_status"] != "passed":
                report["materialization_failures"].append({
                    **partial_context,
                    "error": "dependency_preflight_rejected",
                    "dependency_rejection_reason": dependency_preflight.get(
                        "dependency_rejection_reason", ""
                    ),
                })
                continue
            try:
                partial_rule = materialize_rule_condition_subset(
                    base_rule=current_rule,
                    candidate_rule=parent["rule"],
                    deltas=deltas,
                    accepted_delta_ids=accepted_delta_ids,
                    parent_candidate=parent_name,
                )
                selected_delta_payloads = [
                    delta.to_dict()
                    for delta in deltas
                    if delta.delta_id in accepted_delta_set
                ]
                materialized_lineage = attributed_rule_delta_lineage_report(
                    current_rule,
                    partial_rule,
                    selected_delta_payloads,
                )
                if not materialized_lineage["valid"]:
                    raise ValueError(
                        "partial Rule materialization does not exactly match "
                        "the selected attributed atomic deltas"
                    )
                partial_plan = build_rule_condition_subset_plan(
                    base_rule=current_rule,
                    candidate_rule=parent["rule"],
                    partial_rule=partial_rule,
                    current_plan=current_plan,
                    candidate_plan=parent.get("plan"),
                    deltas=deltas,
                    accepted_delta_ids=accepted_delta_ids,
                    parent_candidate=parent_name,
                )
                PlanValidator().validate_against_rule(
                    partial_plan,
                    partial_rule,
                    attack_label=str(
                        (partial_rule.metadata or {}).get("attack_label") or ""
                    ),
                )
            except Exception as exc:
                report["materialization_failures"].append({
                    **partial_context,
                    "error": repr(exc),
                })
                continue

            reuse_parent = len(accepted_delta_ids) * 2 >= len(deltas)
            subset_name = f"subset_{subset_index:02d}"
            spec = {
                "name": subset_name,
                "artifact_group": f"partial_candidates/{parent_name}",
                "update_kind": "rule",
                "rule": partial_rule,
                "plan": partial_plan,
                "base_rule": current_rule,
                "base_plan": current_plan,
                "requires_plan_regeneration": False,
                "candidate_strategy": RULE_PARTIAL_ACCEPTANCE_STAGE,
                "applied_rule_patch_conditions": accepted_condition_ids,
                "applied_plan_patch_conditions": [
                    condition_id
                    for condition_id in accepted_condition_ids
                    if condition_id.strip().upper() in parent_applied_plan_conditions
                ],
                "regenerated_plan_from_rule": False,
                "semantic_plan_changed": (
                    current_plan is None
                    or semantic_plan_fingerprint(partial_plan)
                    != semantic_plan_fingerprint(current_plan)
                ),
                "candidate_generation_priority": dict(
                    (parent.get("comparison") or {}).get(
                        "candidate_generation_priority", {}
                    )
                ),
                "selected_atomic_delta_ids": accepted_delta_ids,
                "selected_atomic_deltas": [
                    dict(item) for item in selected_delta_payloads
                ],
                "partial_acceptance": partial_context,
                "reuse_full_results": (
                    parent.get("full_results")
                    if reuse_parent
                    else current_full_results
                ),
                "reuse_guard_full_results": (
                    parent.get("guard_full_results", [])
                    if reuse_parent
                    else current_guard_full_results or []
                ),
            }
            evaluation = evaluate_candidate_spec(
                spec=spec,
                args=args,
                cases=cases,
                current_rule=current_rule,
                current_full_results=current_full_results,
                round_dir=round_dir,
                round_index=round_index,
                evaluator=evaluator,
                tool_manifest=tool_manifest,
                planner_guidance={},
                hard_guard_cases=hard_guard_cases,
                current_guard_full_results=current_guard_full_results,
                acceptance_full_results=acceptance_full_results,
                acceptance_guard_full_results=acceptance_guard_full_results,
            )
            evaluation["partial_acceptance"] = partial_context
            partial_evaluations.append(evaluation)
            report["evaluated_subsets"].append({
                **partial_context,
                "candidate_dir": evaluation.get("candidate_dir", ""),
                "summary": dict(evaluation.get("summary") or {}),
                "guard_summary": dict(evaluation.get("guard_summary") or {}),
                "accept": bool((evaluation.get("comparison") or {}).get("accept")),
                "accept_reason": (evaluation.get("comparison") or {}).get(
                    "accept_reason", ""
                ),
                "reject_reason": (evaluation.get("comparison") or {}).get(
                    "reject_reason", ""
                ),
                "reuse_source": "parent_candidate" if reuse_parent else "current",
            })
        parent_reports.append(report)

    selected = select_candidate_evaluation(
        partial_evaluations,
        accepted_only=True,
    )
    write_json(round_dir / "candidate_deltas.json", {
        "stage": RULE_PARTIAL_ACCEPTANCE_STAGE,
        "parents": delta_reports,
    })
    manifest = {
        "stage": RULE_PARTIAL_ACCEPTANCE_STAGE,
        "enabled": True,
        "max_deltas": max_deltas,
        "max_evaluations_per_parent": max_evaluations,
        "parent_candidate_count": len(rejected_rule_candidates),
        "evaluated_subset_count": len(partial_evaluations),
        "accepted": selected is not None,
        "selected_candidate": str(selected.get("name") or "") if selected else "",
        "selected_candidate_dir": str(selected.get("candidate_dir") or "") if selected else "",
        "selected_delta_ids": list(
            (selected.get("partial_acceptance") or {}).get("accepted_delta_ids", [])
        ) if selected else [],
        "parents": parent_reports,
    }
    write_json(round_dir / "partial_search_manifest.json", manifest)
    write_json(round_dir / "partial_acceptance.json", {
        "stage": RULE_PARTIAL_ACCEPTANCE_STAGE,
        "accepted": selected is not None,
        "selected": (
            candidate_evaluation_summary(selected, args=args) if selected else {}
        ),
        "selected_delta_ids": manifest["selected_delta_ids"],
        "evaluated_subset_count": len(partial_evaluations),
    })
    return {
        "selected": selected,
        "evaluations": partial_evaluations,
        "manifest": manifest,
    }


def select_repair_candidate(
    evaluations: List[Dict[str, Any]],
    *,
    args,
) -> Dict[str, Any] | None:
    if not getattr(args, "enable_candidate_repair", True):
        return None
    if int(getattr(args, "max_repair_rounds", 1) or 0) <= 0:
        return None

    new_fp_repair_candidates = [
        item
        for item in evaluations
        if not bool(item.get("probe_only"))
        and str(item.get("update_kind") or "") in {"plan", "rule", "rule_plan"}
        and _candidate_new_fp_repair_eligible(item, args=args)
    ]
    if new_fp_repair_candidates:
        return select_candidate_evaluation(new_fp_repair_candidates, accepted_only=False)

    recall_repair_candidates = [
        item
        for item in evaluations
        if not bool(item.get("probe_only"))
        and not _candidate_rejected_by_canary(item)
        and candidate_recall_repair_eligible(item, args=args)
    ]
    if not recall_repair_candidates:
        return None
    return sorted(recall_repair_candidates, key=_recall_repair_candidate_score)[0]


def _candidate_repair_eligible(item: Dict[str, Any], *, args) -> bool:
    return _candidate_new_fp_repair_eligible(
        item,
        args=args,
    ) or candidate_recall_repair_eligible(item, args=args)


def _candidate_new_fp_repair_eligible(item: Dict[str, Any], *, args) -> bool:
    if _candidate_rejected_by_canary(item):
        return False
    if _candidate_rejected_by_guard(item):
        if not bool(getattr(args, "allow_guard_repair", False)):
            return False
        return bool(_guard_new_fp_txs(item))
    return regression_candidate_repair_eligible(item, candidate_repair_policy(args))


def candidate_repair_diagnostic(item: Dict[str, Any], *, args) -> Dict[str, Any]:
    diagnostic = regression_candidate_repair_diagnostic(
        item,
        candidate_repair_policy(args),
        allow_guard_repair=bool(getattr(args, "allow_guard_repair", False)),
    )
    if _candidate_rejected_by_canary(item):
        diagnostic["eligible"] = False
        diagnostic["reason"] = "candidate_rejected_by_canary"
        diagnostic["repair_type"] = "none"
        diagnostic["candidate_canary"] = dict(
            ((item.get("comparison") or {}).get("candidate_canary") or {})
        )
        return diagnostic
    if _candidate_rejected_by_guard(item):
        diagnostic["guard_gate"] = dict(
            ((item.get("comparison") or {}).get("guard_gate") or {})
        )
        diagnostic["guard_new_fp_txs"] = sorted(_guard_new_fp_txs(item))
        if not bool(getattr(args, "allow_guard_repair", False)):
            diagnostic["eligible"] = False
            diagnostic["reason"] = "candidate_rejected_by_hard_negative_guard"
        elif _guard_new_fp_txs(item):
            diagnostic["eligible"] = True
            diagnostic["reason"] = "eligible_hard_negative_guard_new_fp_repair"
        else:
            diagnostic["eligible"] = False
            diagnostic["reason"] = "guard_rejected_without_new_fp_cases"
    recall_diagnostic = candidate_recall_repair_diagnostic(item, args=args)
    diagnostic.setdefault("repair_type", "new_fp_boundary_repair" if diagnostic.get("eligible") else "none")
    if diagnostic.get("eligible"):
        diagnostic["repair_type"] = (
            "new_fp_boundary_repair"
            if diagnostic.get("reason") != "eligible_hard_negative_guard_new_fp_repair"
            else "new_fp_boundary_repair"
        )
    elif recall_diagnostic.get("eligible"):
        diagnostic.update(recall_diagnostic)
    else:
        diagnostic.update({
            key: value
            for key, value in recall_diagnostic.items()
            if key
            in {
                "repair_type",
                "fixed_fp_count",
                "new_fn_count",
                "fixed_fp_txs",
                "new_fn_txs",
                "guard_fixed_fp_txs",
                "guard_accept",
                "recall_repair_reason",
            }
        })
    return diagnostic


def _candidate_rejected_by_guard(item: Dict[str, Any]) -> bool:
    guard_gate = ((item.get("comparison") or {}).get("guard_gate") or {})
    return bool(guard_gate.get("enabled")) and not bool(guard_gate.get("accept", True))


def _candidate_rejected_by_canary(item: Dict[str, Any]) -> bool:
    canary = dict(
        ((item.get("comparison") or {}).get("candidate_canary") or {})
    )
    return bool(canary.get("enabled")) and not bool(canary.get("passed", True))


def _guard_new_fp_txs(item: Dict[str, Any]) -> set[str]:
    guard_comparison = ((item.get("comparison") or {}).get("guard_comparison") or {})
    return new_fp_txs_from_comparison(guard_comparison)


def new_fn_txs_from_comparison(comparison: Dict[str, Any]) -> set[str]:
    txs: set[str] = set()
    changes = list((comparison or {}).get("per_case_changes", []) or [])
    if not changes:
        changes = list((comparison or {}).get("regressed_cases", []) or [])
    for item in changes:
        if not isinstance(item, dict) or not item.get("tx_hash"):
            continue
        change_type = str(item.get("change_type") or "")
        old_pred = item.get("old_pred")
        new_pred = item.get("new_pred")
        ground_truth = item.get("ground_truth")
        new_error_type = item.get("new_error_type")
        if (
            change_type == "new_fn"
            or new_error_type == "fn"
            or (
                ground_truth == "attack"
                and old_pred == "attack"
                and new_pred in {"benign", "uncertain", None}
            )
        ):
            txs.add(str(item.get("tx_hash")).lower())
    return txs


def fixed_fp_txs_from_comparison(comparison: Dict[str, Any]) -> set[str]:
    txs: set[str] = set()
    changes = list((comparison or {}).get("per_case_changes", []) or [])
    if not changes:
        changes = list((comparison or {}).get("fixed_cases", []) or [])
    for item in changes:
        if not isinstance(item, dict) or not item.get("tx_hash"):
            continue
        if (
            item.get("change_type") == "fixed_fp"
            or (
                item.get("old_error_type") == "fp"
                and item.get("new_error_type") in {None, ""}
            )
        ):
            txs.add(str(item.get("tx_hash")).lower())
    return txs


def guard_fixed_fp_txs_from_comparison(comparison: Dict[str, Any]) -> set[str]:
    guard_comparison = dict((comparison or {}).get("guard_comparison") or {})
    return fixed_fp_txs_from_comparison(guard_comparison)


def candidate_recall_repair_eligible(item: Dict[str, Any], *, args) -> bool:
    diagnostic = candidate_recall_repair_diagnostic(item, args=args)
    return bool(diagnostic.get("eligible"))


def candidate_recall_repair_diagnostic(item: Dict[str, Any], *, args) -> Dict[str, Any]:
    comparison = dict((item or {}).get("comparison") or {})
    tradeoff = dict(comparison.get("acceptance_tradeoff") or {})
    guard_gate = dict(comparison.get("guard_gate") or {})
    fixed_fp_txs = fixed_fp_txs_from_comparison(comparison)
    new_fn_txs = new_fn_txs_from_comparison(comparison)
    guard_fixed = guard_fixed_fp_txs_from_comparison(comparison)
    fixed_fp_count = max(int(tradeoff.get("fixed_fp", 0) or 0), len(fixed_fp_txs))
    new_fn_count = max(int(tradeoff.get("new_fn", 0) or 0), len(new_fn_txs))
    guard_accept = not guard_gate.get("enabled") or bool(guard_gate.get("accept", True))
    delta = dict(comparison.get("delta") or {})
    error_delta = int(delta.get("errors", 0) or 0)
    reason = ""
    eligible = True
    if str((item or {}).get("update_kind") or "") != "rule":
        eligible = False
        reason = "recall_repair_only_for_rule_candidates"
    elif bool(comparison.get("accept")):
        eligible = False
        reason = "candidate_already_accepted"
    elif not bool(getattr(args, "enable_candidate_repair", True)):
        eligible = False
        reason = "candidate_repair_disabled"
    elif not bool(getattr(args, "enable_recall_repair", True)):
        eligible = False
        reason = "recall_repair_disabled"
    elif int(getattr(args, "max_repair_rounds", 1) or 0) <= 0:
        eligible = False
        reason = "max_repair_rounds_zero"
    elif not guard_accept:
        eligible = False
        reason = "guard_rejected_no_recall_repair"
    elif error_delta > 0:
        eligible = False
        reason = "candidate_errors_increased"
    elif fixed_fp_count < int(getattr(args, "recall_repair_min_fixed_fp", 1) or 0):
        eligible = False
        reason = "insufficient_fixed_fp_for_recall_repair"
    elif new_fn_count < 1:
        eligible = False
        reason = "no_new_fn_cases_for_recall_repair"
    else:
        reason = "eligible_rejected_candidate_new_fn_recall_repair"
    return {
        "eligible": eligible,
        "repair_type": "new_fn_recall_repair" if eligible else "none",
        "reason": reason,
        "recall_repair_reason": reason,
        "delta": delta,
        "fixed_fp_count": fixed_fp_count,
        "new_fn_count": new_fn_count,
        "fixed_fp_txs": sorted(fixed_fp_txs),
        "new_fn_txs": sorted(new_fn_txs),
        "guard_fixed_fp_txs": sorted(guard_fixed),
        "guard_accept": guard_accept,
    }


def _recall_repair_candidate_score(item: Dict[str, Any]) -> tuple:
    comparison = dict((item or {}).get("comparison") or {})
    diagnostic = candidate_recall_repair_diagnostic(item, args=_RecallRepairDefaultArgs())
    delta = dict(comparison.get("delta") or {})
    return (
        0 if diagnostic.get("guard_accept") else 1,
        -int(diagnostic.get("fixed_fp_count", 0) or 0),
        int(diagnostic.get("new_fn_count", 10**9) or 10**9),
        int(delta.get("errors", 0) or 0),
        str(item.get("name") or ""),
    )


class _RecallRepairDefaultArgs:
    enable_candidate_repair = True
    enable_recall_repair = True
    max_repair_rounds = 1
    recall_repair_min_fixed_fp = 1


def candidate_repair_policy(args) -> CandidateRepairPolicy:
    return CandidateRepairPolicy.from_args(args)


def write_legacy_candidate_artifacts(round_dir: Path, item: Dict[str, Any]) -> None:
    """Mirror the selected candidate into the historical round_xx filenames."""
    write_json(round_dir / "candidate_rule.json", item["rule"].to_dict())
    if item.get("plan") is not None:
        write_json(round_dir / "candidate_plan.json", item["plan"].to_dict())
    write_json(round_dir / "candidate_full_results.json", item["full_results"])
    write_json(round_dir / "candidate_slim_results.json", item["slim_results"])
    write_json(round_dir / "candidate_results.json", item["slim_results"])
    write_json(round_dir / "candidate_summary.json", item["summary"])
    write_json(round_dir / "candidate_case_plans.json", extract_case_plans(item["full_results"]))
    if item.get("guard_enabled"):
        write_json(round_dir / "candidate_guard_full_results.json", item.get("guard_full_results", []))
        write_json(round_dir / "candidate_guard_slim_results.json", item.get("guard_slim_results", []))
        write_json(round_dir / "candidate_guard_summary.json", item.get("guard_summary", {}))
        write_json(
            round_dir / "guard_comparison.json",
            {
                "guard_comparison": (item.get("comparison") or {}).get("guard_comparison", {}),
                "guard_gate": (item.get("comparison") or {}).get("guard_gate", {}),
            },
        )
    write_json(round_dir / "comparison.json", item["comparison"])
    write_json(
        round_dir / "candidate_artifact.json",
        {
            "selected_from": item.get("candidate_dir", ""),
            "update_kind": item.get("update_kind", ""),
            "candidate_strategy": item.get("candidate_strategy", ""),
            "rule": {
                "rule_id": item["rule"].rule_id,
                "rule_version": item["rule"].version,
                "rule_source": item.get("rule_path", ""),
                "changed": bool((item.get("comparison") or {}).get("rule_changed")),
            },
            "plan": {
                "plan_id": item["plan"].plan_id if item.get("plan") is not None else "",
                "plan_source": item.get("plan_path", ""),
                "plan_version": (item["plan"].metadata or {}).get("plan_version")
                if item.get("plan") is not None else None,
                "changed": bool((item.get("comparison") or {}).get("plan_changed")),
                "regenerated": bool(item.get("plan_regenerated")),
                "source": (item["plan"].metadata or {}).get("source", "")
                if item.get("plan") is not None else "",
                "plan_generation_fallback": bool(item.get("plan_generation_fallback")),
            },
            "complexity": {
                "candidate_rule_complexity": rule_complexity(item["rule"]),
                "plan_complexity": plan_complexity(item.get("plan")),
            },
            "runtime_stats": (item.get("comparison") or {}).get("candidate_runtime_stats", {}),
        },
    )


def _index_results_by_tx(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for result in results or []:
        tx_hash = ""
        if isinstance(result, dict):
            tx_hash = str(
                (result.get("transaction", {}) or {}).get("tx_hash")
                or result.get("tx_hash")
                or ""
            ).lower()
        if tx_hash:
            out[tx_hash] = result
    return out


def _first_evidence_plan_from_results(
    results: List[Dict[str, Any]] | None,
) -> EvidencePlan | None:
    for result in list(results or []):
        plan_data = get_plan(result)
        if not isinstance(plan_data, dict) or not plan_data.get("judge_steps"):
            continue
        try:
            return EvidencePlan.from_dict(plan_data)
        except Exception:
            continue
    return None


def build_candidate_judge_reuse_index(
    results: List[Dict[str, Any]],
    *,
    candidate_rule: EvolvingRule,
    candidate_plan: EvidencePlan | None,
    baseline_plan: EvidencePlan | Dict[str, Any] | None = None,
    changed_condition_ids: set[str] | None = None,
    disable_stateful_bindings: bool = False,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Index previous tx results before optional candidate judge-step reuse.

    Do not pre-filter by rule/plan version here. A rule or plan update may only
    change one local judge step, and unchanged steps should remain reusable.
    PacketRuntime performs the precise step-level fingerprint check over tx,
    question, evidence refs, follow-up views/tools, expected answer, model, and
    packet context before reusing any individual judge result.
    """
    normalized_candidate_plan = (
        normalize_plan_for_runtime_policy(
            candidate_plan,
            disable_stateful_bindings=True,
            reason="candidate_reuse_audit",
        )
        if candidate_plan is not None and disable_stateful_bindings
        else candidate_plan
    )
    indexed: Dict[str, Dict[str, Any]] = {}
    skipped = {
        "missing_tx": 0,
    }
    invalidated_by_tx: Dict[str, List[str]] = {}
    for result in results or []:
        tx_hash = get_tx_hash(result).lower()
        if not tx_hash:
            skipped["missing_tx"] += 1
            continue
        invalidated = _candidate_reuse_invalidated_step_ids(
            result,
            candidate_plan=normalized_candidate_plan,
            baseline_plan=baseline_plan,
            explicit_changed_condition_ids=changed_condition_ids,
            disable_stateful_bindings=disable_stateful_bindings,
        )
        if invalidated:
            invalidated_by_tx[tx_hash] = sorted(invalidated)
            indexed[tx_hash] = _filter_reuse_result_by_invalidated_steps(
                result,
                invalidated,
            )
        else:
            indexed[tx_hash] = result
    step_policy_audit = _candidate_step_policy_audit(
        results,
        candidate_plan=normalized_candidate_plan,
        baseline_plan=baseline_plan,
        explicit_changed_condition_ids=changed_condition_ids,
        disable_stateful_bindings=disable_stateful_bindings,
    )
    return indexed, {
        "enabled": True,
        "provided_results": len(results or []),
        "reusable_by_tx": len(indexed),
        "skipped": skipped,
        "pre_filter": "tx_hash_only",
        "step_level_guard": "PacketRuntime.judge_step_fingerprint",
        "candidate_rule_id": candidate_rule.rule_id,
        "candidate_rule_version": candidate_rule.version,
        "candidate_plan_id": (
            normalized_candidate_plan.plan_id
            if normalized_candidate_plan is not None
            else ""
        ),
        "candidate_plan_version": (
            (normalized_candidate_plan.metadata or {}).get("plan_version")
            if normalized_candidate_plan is not None
            else None
        ),
        "binding_policy_normalized": bool(disable_stateful_bindings),
        "stateful_reuse_invalidation": {
            "enabled": bool(invalidated_by_tx),
            "invalidated_tx_count": len(invalidated_by_tx),
            "invalidated_condition_ids": sorted({
                condition_id
                for values in invalidated_by_tx.values()
                for condition_id in values
            }),
            "invalidated_by_tx": invalidated_by_tx,
        },
        "step_policy_audit": step_policy_audit,
    }


def _candidate_step_policy_audit(
    results: List[Dict[str, Any]],
    *,
    candidate_plan: EvidencePlan | None,
    baseline_plan: EvidencePlan | Dict[str, Any] | None = None,
    explicit_changed_condition_ids: set[str] | None = None,
    disable_stateful_bindings: bool = False,
) -> Dict[str, Any]:
    if candidate_plan is None:
        return {
            "available": False,
            "reason": "candidate_plan_missing",
        }
    baseline_plan_data = _plan_dict_for_reuse_diff(baseline_plan)
    if not baseline_plan_data:
        for result in list(results or []):
            plan = get_plan(result)
            if isinstance(plan, dict) and plan.get("judge_steps"):
                baseline_plan_data = plan
                break
    if not baseline_plan_data:
        return {
            "available": False,
            "reason": "baseline_plan_missing",
        }
    if disable_stateful_bindings:
        baseline_plan_data = disable_stateful_bindings_in_plan(
            EvidencePlan.from_dict(baseline_plan_data)
        ).to_dict()
        candidate_plan = disable_stateful_bindings_in_plan(candidate_plan)

    baseline_steps = {
        _judge_step_condition_key(item): _judge_step_policy_payload(item)
        for item in list(baseline_plan_data.get("judge_steps", []) or [])
        if isinstance(item, dict) and _judge_step_condition_key(item)
    }
    candidate_steps = {
        _judge_step_condition_key(item): _judge_step_policy_payload(item)
        for item in list(candidate_plan.to_dict().get("judge_steps", []) or [])
        if isinstance(item, dict) and _judge_step_condition_key(item)
    }
    shared_ids = sorted(set(baseline_steps) & set(candidate_steps))
    unchanged_ids = [
        step_id
        for step_id in shared_ids
        if baseline_steps[step_id] == candidate_steps[step_id]
    ]
    changed_ids = [
        step_id
        for step_id in shared_ids
        if baseline_steps[step_id] != candidate_steps[step_id]
    ]
    explicit_ids = {
        str(item).strip().upper()
        for item in (explicit_changed_condition_ids or set())
        if str(item).strip()
    }
    dependency_basis = set(changed_ids) | explicit_ids
    stateful_dependent_ids = _stateful_downstream_condition_ids(
        candidate_plan,
        dependency_basis,
    )
    stateful_invalidated_ids = dependency_basis | stateful_dependent_ids
    reuse_eligible_ids = [
        step_id
        for step_id in unchanged_ids
        if step_id not in stateful_invalidated_ids
    ]
    return {
        "available": True,
        "baseline_plan_id": str(baseline_plan_data.get("plan_id") or ""),
        "candidate_plan_id": str(candidate_plan.plan_id or ""),
        "unchanged_step_ids": unchanged_ids,
        "changed_step_ids": changed_ids,
        "explicit_changed_condition_ids": sorted(explicit_ids),
        "stateful_dependent_step_ids": sorted(stateful_dependent_ids),
        "stateful_reuse_invalidated_step_ids": sorted(stateful_invalidated_ids),
        "added_step_ids": sorted(set(candidate_steps) - set(baseline_steps)),
        "removed_step_ids": sorted(set(baseline_steps) - set(candidate_steps)),
        "step_policy_eligible_for_reuse": reuse_eligible_ids,
        "binding_policy_normalized": bool(disable_stateful_bindings),
        "note": (
            "Eligibility covers local step policy and stateful dependencies; Runtime "
            "also verifies model, packet identity, selected-view content, and "
            "execution settings."
        ),
    }


def _candidate_reuse_invalidated_step_ids(
    result: Dict[str, Any],
    *,
    candidate_plan: EvidencePlan | None,
    baseline_plan: EvidencePlan | Dict[str, Any] | None = None,
    explicit_changed_condition_ids: set[str] | None = None,
    disable_stateful_bindings: bool = False,
) -> set[str]:
    if candidate_plan is None:
        return set()
    baseline_plan_data = _plan_dict_for_reuse_diff(baseline_plan)
    if not baseline_plan_data:
        baseline_plan_data = get_plan(result)
    if (
        not isinstance(baseline_plan_data, dict)
        or not baseline_plan_data.get("judge_steps")
    ):
        return set()
    if disable_stateful_bindings:
        baseline_plan_data = disable_stateful_bindings_in_plan(
            EvidencePlan.from_dict(baseline_plan_data)
        ).to_dict()
        candidate_plan = disable_stateful_bindings_in_plan(candidate_plan)
    baseline_steps = {
        _judge_step_condition_key(item): _judge_step_policy_payload(item)
        for item in list(baseline_plan_data.get("judge_steps", []) or [])
        if isinstance(item, dict) and _judge_step_condition_key(item)
    }
    candidate_steps = {
        _judge_step_condition_key(item): _judge_step_policy_payload(item)
        for item in list(candidate_plan.to_dict().get("judge_steps", []) or [])
        if isinstance(item, dict) and _judge_step_condition_key(item)
    }
    shared_ids = set(baseline_steps) & set(candidate_steps)
    changed = {
        step_id
        for step_id in shared_ids
        if baseline_steps.get(step_id) != candidate_steps.get(step_id)
    }
    changed.update(
        str(item).strip().upper()
        for item in (explicit_changed_condition_ids or set())
        if str(item).strip()
    )
    downstream = _stateful_downstream_condition_ids(candidate_plan, changed)
    return changed | downstream


def _plan_dict_for_reuse_diff(
    plan: EvidencePlan | Dict[str, Any] | None,
) -> Dict[str, Any]:
    if isinstance(plan, EvidencePlan):
        return plan.to_dict()
    if isinstance(plan, dict):
        return copy.deepcopy(plan)
    return {}


def _filter_reuse_result_by_invalidated_steps(
    result: Dict[str, Any],
    invalidated_ids: set[str],
) -> Dict[str, Any]:
    invalidated = {
        str(item).strip().upper()
        for item in invalidated_ids
        if str(item).strip()
    }
    if not invalidated:
        return result
    copied = copy.deepcopy(result)

    def keep_item(item: Any) -> bool:
        if not isinstance(item, dict):
            return True
        item_id = str(
            item.get("judge_id")
            or item.get("condition_id")
            or item.get("id")
            or ""
        ).strip().upper()
        return not item_id or item_id not in invalidated

    def filter_trace(trace: Any) -> None:
        if not isinstance(trace, dict):
            return
        for key in ("judge_calls", "judge_step_traces"):
            value = trace.get(key)
            if isinstance(value, list):
                trace[key] = [item for item in value if keep_item(item)]
        execution = trace.get("execution")
        if isinstance(execution, dict):
            filter_trace(execution)
        metadata = trace.get("metadata")
        if isinstance(metadata, dict):
            filter_trace(metadata)

    def filter_plan(plan_data: Any) -> None:
        if not isinstance(plan_data, dict):
            return
        steps = plan_data.get("judge_steps")
        if isinstance(steps, list):
            plan_data["judge_steps"] = [item for item in steps if keep_item(item)]

    inference = copied.get("inference")
    if isinstance(inference, dict):
        filter_plan(inference.get("plan"))
        filter_trace(inference.get("trace"))
    filter_plan(copied.get("plan"))
    filter_trace(copied.get("trace"))
    return copied


def _judge_step_policy_payload(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "condition_id": str(step.get("condition_id") or step.get("id") or ""),
        "question": str(step.get("question") or ""),
        "expected_answer": bool(step.get("expected_answer", True)),
        "evidence_refs": _normalized_policy_set(step.get("evidence_refs")),
        "default_evidence_refs": _normalized_policy_set(
            step.get("default_evidence_refs")
        ),
        "allowed_followup_views": _normalized_policy_set(
            step.get("allowed_followup_views")
        ),
        "allowed_tools": _normalized_policy_set(step.get("allowed_tools")),
        "max_followups": int(step.get("max_followups", 0) or 0),
        "depends_on": _normalized_policy_set(step.get("depends_on")),
        "consumes_state_keys": _normalized_policy_set(
            step.get("consumes_state_keys")
        ),
        "produces_state_key": str(step.get("produces_state_key", "") or ""),
        "state_prompt_role": str(step.get("state_prompt_role", "") or ""),
        "state_output_schema": (
            dict(step.get("state_output_schema") or {})
            if isinstance(step.get("state_output_schema"), dict)
            else {}
        ),
    }


def _normalized_policy_set(value: Any) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return sorted({str(item) for item in values if str(item)})


def _judge_step_condition_key(step: Dict[str, Any]) -> str:
    return str(step.get("condition_id") or step.get("id") or "").strip().upper()


def extract_case_plans(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []
    for result in results or []:
        plan = get_plan(result)
        if not plan:
            continue
        plans.append({
            "tx_hash": get_tx_hash(result),
            "plan_id": plan.get("plan_id", ""),
            "rule_id": plan.get("rule_id", ""),
            "rule_version": plan.get("rule_version", ""),
            "emit_logic": plan.get("emit_logic", ""),
            "judge_step_count": len(plan.get("judge_steps", []) or []),
            "metadata": plan.get("metadata", {}),
            "plan": plan,
        })
    return plans


def plan_complexity(plan: EvidencePlan | Dict[str, Any] | None) -> Dict[str, Any]:
    if plan is None:
        return {
            "judge_step_count": 0,
            "total_max_followups": 0,
            "source_followup_step_count": 0,
            "default_view_count_total": 0,
            "allowed_followup_view_count_total": 0,
        }
    plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
    steps = list(plan_dict.get("judge_steps", []) or [])
    total_max_followups = 0
    source_followup_steps = 0
    default_total = 0
    followup_total = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        max_followups = int(step.get("max_followups", 0) or 0)
        tools = set(step.get("allowed_tools", []) or [])
        total_max_followups += max_followups
        default_total += len(step.get("default_evidence_refs", []) or step.get("evidence_refs", []) or [])
        followup_total += len(step.get("allowed_followup_views", []) or [])
        if "read_function_chunk" in tools or max_followups >= 2:
            source_followup_steps += 1
    return {
        "judge_step_count": len(steps),
        "total_max_followups": total_max_followups,
        "source_followup_step_count": source_followup_steps,
        "default_view_count_total": default_total,
        "allowed_followup_view_count_total": followup_total,
    }


def aggregate_runtime_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    reuse_enabled = False
    reuse_attempted = 0
    reuse_reused = 0
    skipped_by_reason: Dict[str, int] = {}
    early_stop_enabled = False
    early_stop_triggered = 0
    skipped_judge_steps = 0
    adaptive_enabled = False
    adaptive_mode_counts: Dict[str, int] = {}
    adaptive_added = 0
    adaptive_removed = 0
    adaptive_estimated_chars_total = 0
    adaptive_estimated_chars_count = 0
    parallel_requested = False
    parallel_effective = False
    parallel_disabled_by_reason: Dict[str, int] = {}
    parallel_submitted = 0
    parallel_completed = 0
    parallel_failed = 0
    parallel_elapsed = 0.0
    parallel_structured_logs = 0
    rate_limit_fallback_triggered = 0
    rate_limit_retry_count = 0
    rate_limit_recovered_judge_count = 0
    total_judge_elapsed_seconds = 0.0
    tx_elapsed_seconds_total = 0.0
    tx_elapsed_count = 0
    judge_call_count = 0
    executed_judge_call_count = 0
    for result in list(results or []):
        metadata = _runtime_trace_metadata(result)
        reuse = dict(metadata.get("judge_reuse") or {})
        if reuse.get("enabled"):
            reuse_enabled = True
        reuse_attempted += int(reuse.get("attempted", 0) or 0)
        reuse_reused += int(reuse.get("reused", 0) or 0)
        for reason, count in dict(reuse.get("skipped_by_reason") or {}).items():
            skipped_by_reason[str(reason)] = (
                int(skipped_by_reason.get(str(reason), 0) or 0)
                + int(count or 0)
            )
        early = dict(metadata.get("early_stop") or {})
        if early.get("enabled"):
            early_stop_enabled = True
        if early.get("triggered"):
            early_stop_triggered += 1
        skipped_judge_steps += int(
            metadata.get("skipped_judge_step_count", 0)
            or len(early.get("skipped_judge_ids", []) or [])
        )
        adaptive = dict(metadata.get("adaptive_evidence") or {})
        if adaptive.get("enabled"):
            adaptive_enabled = True
        for mode, count in dict(adaptive.get("mode_counts") or {}).items():
            adaptive_mode_counts[str(mode)] = (
                int(adaptive_mode_counts.get(str(mode), 0) or 0)
                + int(count or 0)
            )
        adaptive_added += int(adaptive.get("total_added_views", 0) or 0)
        adaptive_removed += int(adaptive.get("total_removed_views", 0) or 0)
        avg_chars = float(adaptive.get("avg_estimated_chars", 0) or 0)
        mode_total = sum(int(v or 0) for v in dict(adaptive.get("mode_counts") or {}).values())
        if mode_total:
            adaptive_estimated_chars_total += int(round(avg_chars * mode_total))
            adaptive_estimated_chars_count += mode_total
        parallel = dict(metadata.get("parallel_judge") or {})
        if parallel.get("requested"):
            parallel_requested = True
        if parallel.get("effective"):
            parallel_effective = True
        disabled_reason = str(parallel.get("disabled_reason") or "")
        if disabled_reason:
            parallel_disabled_by_reason[disabled_reason] = (
                int(parallel_disabled_by_reason.get(disabled_reason, 0) or 0) + 1
            )
        parallel_submitted += int(parallel.get("submitted_step_count", 0) or 0)
        parallel_completed += int(parallel.get("completed_step_count", 0) or 0)
        parallel_failed += int(parallel.get("failed_step_count", 0) or 0)
        parallel_elapsed += float(parallel.get("elapsed_parallel_seconds", 0) or 0)
        parallel_structured_logs += int(parallel.get("structured_log_count", 0) or 0)
        if parallel.get("rate_limit_serial_fallback_triggered"):
            rate_limit_fallback_triggered += 1
        rate_limit_retry_count += int(
            parallel.get("rate_limit_retry_count", 0) or 0
        )
        rate_limit_recovered_judge_count += len(
            list(parallel.get("rate_limit_recovered_judge_ids", []) or [])
        )
        for timing in dict(parallel.get("step_timings") or {}).values():
            if isinstance(timing, dict):
                total_judge_elapsed_seconds += float(timing.get("duration_seconds", 0) or 0)
        elapsed_seconds = float(metadata.get("elapsed_seconds", 0) or 0)
        if elapsed_seconds:
            tx_elapsed_seconds_total += elapsed_seconds
            tx_elapsed_count += 1
        metadata_judge_calls = int(metadata.get("judge_call_count", 0) or 0)
        metadata_reused = int(reuse.get("reused", 0) or 0)
        judge_call_count += metadata_judge_calls
        executed_judge_call_count += max(0, metadata_judge_calls - metadata_reused)
    return {
        "candidate_judge_reuse": {
            "enabled": reuse_enabled,
            "attempted_judge_call_count": reuse_attempted,
            "reused_judge_call_count": reuse_reused,
            "skipped_by_reason": skipped_by_reason,
            "reuse_hit_rate": (
                round(reuse_reused / reuse_attempted, 4) if reuse_attempted else 0.0
            ),
        },
        "runtime_early_stop": {
            "enabled": early_stop_enabled,
            "triggered_count": early_stop_triggered,
            "skipped_judge_step_count": skipped_judge_steps,
        },
        "adaptive_evidence": {
            "enabled": adaptive_enabled,
            "mode_counts": adaptive_mode_counts,
            "total_added_views": adaptive_added,
            "total_removed_views": adaptive_removed,
            "avg_estimated_chars": (
                round(adaptive_estimated_chars_total / adaptive_estimated_chars_count, 2)
                if adaptive_estimated_chars_count
                else 0.0
            ),
        },
        "parallel_judge": {
            "requested": parallel_requested,
            "effective": parallel_effective,
            "disabled_by_reason": parallel_disabled_by_reason,
            "submitted_step_count": parallel_submitted,
            "completed_step_count": parallel_completed,
            "failed_step_count": parallel_failed,
            "elapsed_parallel_seconds": round(parallel_elapsed, 4),
            "structured_log_count": parallel_structured_logs,
            "total_judge_elapsed_seconds": round(total_judge_elapsed_seconds, 4),
            "average_tx_elapsed_seconds": (
                round(tx_elapsed_seconds_total / tx_elapsed_count, 4)
                if tx_elapsed_count
                else 0.0
            ),
            "judge_call_count": judge_call_count,
            "executed_judge_call_count": executed_judge_call_count,
            "rate_limit_fallback_triggered_tx_count": rate_limit_fallback_triggered,
            "rate_limit_retry_count": rate_limit_retry_count,
            "rate_limit_recovered_judge_count": rate_limit_recovered_judge_count,
        },
    }


def budget_candidate_stats(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    hard_rejected = 0
    soft_warning = 0
    accepted_with_soft = 0
    hard_reasons: Dict[str, int] = {}
    soft_by_label: Dict[str, int] = {}
    rejected_reasons: Dict[str, int] = {}
    for item in list(evaluations or []):
        comparison = dict(item.get("comparison") or {})
        complexity = dict(
            comparison.get("candidate_rule_complexity") or rule_complexity(item["rule"])
        )
        if complexity.get("hard_over_budget"):
            hard_rejected += 1
            for reason in list(complexity.get("hard_over_budget_reasons") or []):
                hard_reasons[str(reason)] = int(hard_reasons.get(str(reason), 0) or 0) + 1
        if complexity.get("soft_over_budget"):
            soft_warning += 1
            label = str(
                (item["rule"].metadata or {}).get("attack_label")
                or complexity.get("label_budget_profile")
                or ""
            )
            soft_by_label[label] = int(soft_by_label.get(label, 0) or 0) + 1
            if comparison.get("accept"):
                accepted_with_soft += 1
        reject_reason = str(comparison.get("reject_reason") or "")
        if reject_reason:
            rejected_reasons[reject_reason] = (
                int(rejected_reasons.get(reject_reason, 0) or 0) + 1
            )
    return {
        "rejected_by_hard_budget_count": hard_rejected,
        "allowed_with_soft_budget_warning_count": soft_warning,
        "accepted_with_soft_budget_warning": accepted_with_soft,
        "soft_budget_warning_by_label": soft_by_label,
        "hard_budget_reasons": hard_reasons,
        "rejected_reasons": rejected_reasons,
    }


def _aggregate_candidate_runtime(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    reuse = {
        "attempted_judge_call_count": 0,
        "reused_judge_call_count": 0,
        "skipped_by_reason": {},
    }
    early = {
        "triggered_count": 0,
        "skipped_judge_step_count": 0,
    }
    adaptive = {
        "mode_counts": {},
        "total_added_views": 0,
        "total_removed_views": 0,
        "avg_estimated_chars": 0.0,
    }
    parallel = {
        "requested": False,
        "effective": False,
        "disabled_by_reason": {},
        "submitted_step_count": 0,
        "completed_step_count": 0,
        "failed_step_count": 0,
        "elapsed_parallel_seconds": 0.0,
        "structured_log_count": 0,
        "total_judge_elapsed_seconds": 0.0,
        "average_tx_elapsed_seconds": 0.0,
        "judge_call_count": 0,
        "executed_judge_call_count": 0,
        "rate_limit_fallback_triggered_tx_count": 0,
        "rate_limit_retry_count": 0,
        "rate_limit_recovered_judge_count": 0,
    }
    parallel_tx_elapsed_total = 0.0
    parallel_tx_elapsed_count = 0
    adaptive_estimated_chars_total = 0
    adaptive_estimated_chars_count = 0
    for item in list(evaluations or []):
        stats = (item.get("comparison") or {}).get("candidate_runtime_stats") or {}
        candidate_reuse = dict(stats.get("candidate_judge_reuse") or {})
        reuse["attempted_judge_call_count"] += int(
            candidate_reuse.get("attempted_judge_call_count", 0) or 0
        )
        reuse["reused_judge_call_count"] += int(
            candidate_reuse.get("reused_judge_call_count", 0) or 0
        )
        for reason, count in dict(candidate_reuse.get("skipped_by_reason") or {}).items():
            skipped = reuse["skipped_by_reason"]
            skipped[str(reason)] = int(skipped.get(str(reason), 0) or 0) + int(count or 0)
        candidate_early = dict(stats.get("runtime_early_stop") or {})
        early["triggered_count"] += int(candidate_early.get("triggered_count", 0) or 0)
        early["skipped_judge_step_count"] += int(
            candidate_early.get("skipped_judge_step_count", 0) or 0
        )
        candidate_adaptive = dict(stats.get("adaptive_evidence") or {})
        for mode, count in dict(candidate_adaptive.get("mode_counts") or {}).items():
            counts = adaptive["mode_counts"]
            counts[str(mode)] = int(counts.get(str(mode), 0) or 0) + int(count or 0)
        adaptive["total_added_views"] += int(
            candidate_adaptive.get("total_added_views", 0) or 0
        )
        adaptive["total_removed_views"] += int(
            candidate_adaptive.get("total_removed_views", 0) or 0
        )
        mode_total = sum(
            int(v or 0) for v in dict(candidate_adaptive.get("mode_counts") or {}).values()
        )
        if mode_total:
            adaptive_estimated_chars_total += int(
                round(float(candidate_adaptive.get("avg_estimated_chars", 0) or 0) * mode_total)
            )
            adaptive_estimated_chars_count += mode_total
        candidate_parallel = dict(stats.get("parallel_judge") or {})
        parallel["requested"] = bool(parallel["requested"]) or bool(
            candidate_parallel.get("requested")
        )
        parallel["effective"] = bool(parallel["effective"]) or bool(
            candidate_parallel.get("effective")
        )
        for reason, count in dict(candidate_parallel.get("disabled_by_reason") or {}).items():
            disabled = parallel["disabled_by_reason"]
            disabled[str(reason)] = int(disabled.get(str(reason), 0) or 0) + int(count or 0)
        for key in (
            "submitted_step_count",
            "completed_step_count",
            "failed_step_count",
            "structured_log_count",
            "judge_call_count",
            "executed_judge_call_count",
            "rate_limit_fallback_triggered_tx_count",
            "rate_limit_retry_count",
            "rate_limit_recovered_judge_count",
        ):
            parallel[key] = int(parallel.get(key, 0) or 0) + int(
                candidate_parallel.get(key, 0) or 0
            )
        parallel["elapsed_parallel_seconds"] = float(
            parallel.get("elapsed_parallel_seconds", 0) or 0
        ) + float(candidate_parallel.get("elapsed_parallel_seconds", 0) or 0)
        parallel["total_judge_elapsed_seconds"] = float(
            parallel.get("total_judge_elapsed_seconds", 0) or 0
        ) + float(candidate_parallel.get("total_judge_elapsed_seconds", 0) or 0)
        avg_tx = float(candidate_parallel.get("average_tx_elapsed_seconds", 0) or 0)
        submitted = int(candidate_parallel.get("submitted_step_count", 0) or 0)
        if avg_tx and submitted:
            parallel_tx_elapsed_total += avg_tx
            parallel_tx_elapsed_count += 1
    attempted = int(reuse["attempted_judge_call_count"] or 0)
    reused = int(reuse["reused_judge_call_count"] or 0)
    reuse["reuse_hit_rate"] = round(reused / attempted, 4) if attempted else 0.0
    adaptive["avg_estimated_chars"] = (
        round(adaptive_estimated_chars_total / adaptive_estimated_chars_count, 2)
        if adaptive_estimated_chars_count
        else 0.0
    )
    parallel["elapsed_parallel_seconds"] = round(
        float(parallel.get("elapsed_parallel_seconds", 0) or 0),
        4,
    )
    parallel["total_judge_elapsed_seconds"] = round(
        float(parallel.get("total_judge_elapsed_seconds", 0) or 0),
        4,
    )
    parallel["average_tx_elapsed_seconds"] = (
        round(parallel_tx_elapsed_total / parallel_tx_elapsed_count, 4)
        if parallel_tx_elapsed_count
        else 0.0
    )
    return {
        "candidate_judge_reuse": reuse,
        "runtime_early_stop": early,
        "adaptive_evidence": adaptive,
        "parallel_judge": parallel,
    }


def _runtime_trace_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    trace = get_trace(result)
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else {}
    return dict(metadata or {}) if isinstance(metadata, dict) else {}


def _merge_planner_guidance(
    tx_context: Dict[str, Any],
    planner_guidance: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not planner_guidance:
        return dict(tx_context or {})
    merged = dict(tx_context or {})
    merged["planner_guidance"] = dict(planner_guidance)
    return merged


def runtime_adaptive_kwargs(args) -> Dict[str, Any]:
    return {
        "adaptive_evidence": bool(getattr(args, "adaptive_evidence", False)),
        "adaptive_evidence_mode": str(
            getattr(args, "adaptive_evidence_mode", "off") or "off"
        ),
        "adaptive_evidence_max_direct_trace_nodes": int(
            getattr(args, "adaptive_evidence_max_direct_trace_nodes", 80) or 0
        ),
        "adaptive_evidence_max_direct_trace_chars": int(
            getattr(args, "adaptive_evidence_max_direct_trace_chars", 45000) or 0
        ),
        "adaptive_evidence_max_medium_trace_nodes": int(
            getattr(args, "adaptive_evidence_max_medium_trace_nodes", 250) or 0
        ),
        "adaptive_evidence_debug": bool(
            getattr(args, "adaptive_evidence_debug", False)
        ),
        "parallel_judge": bool(getattr(args, "parallel_judge", False)),
        "judge_concurrency": int(getattr(args, "judge_concurrency", 1) or 1),
        "judge_max_tokens": int(
            getattr(args, "judge_max_tokens", 8192) or 8192
        ),
        "aggregator_max_tokens": int(
            getattr(args, "aggregator_max_tokens", 8192) or 8192
        ),
        "planner_thinking": str(
            getattr(args, "planner_thinking", "adaptive") or "adaptive"
        ),
        "judge_thinking": str(
            getattr(args, "judge_thinking", "disabled") or "disabled"
        ),
        "aggregator_thinking": str(
            getattr(args, "aggregator_thinking", "disabled") or "disabled"
        ),
        "adaptive_model": getattr(args, "adaptive_model", None),
        "adaptive_provider": getattr(args, "adaptive_provider", None),
        "structured_judge_logs": bool(
            getattr(args, "structured_judge_logs", True)
        ),
        "iv_stateful_runtime": bool(
            getattr(args, "iv_stateful_runtime", True)
        ),
        "access_control_binding_mode": str(
            getattr(args, "access_control_binding_mode", "stateful") or "stateful"
        ),
        "reentrancy_binding_mode": str(
            getattr(args, "reentrancy_binding_mode", "soft") or "soft"
        ),
        "disable_stateful_bindings": bool(
            getattr(args, "disable_stateful_bindings", False)
        ),
        "dynamic_aggregation": bool(
            getattr(args, "dynamic_aggregation", False)
        ),
        "followup_context_mode": str(
            getattr(args, "followup_context_mode", "unified") or "unified"
        ),
        "judge_followup_mode": str(
            getattr(args, "judge_followup_mode", "plan") or "plan"
        ),
        "judge_expanded_followup_views": max(
            0,
            int(getattr(args, "judge_expanded_followup_views", 2) or 0),
        ),
        "rate_limit_serial_fallback": bool(
            getattr(args, "rate_limit_serial_fallback", True)
        ),
        "rate_limit_retry_attempts": int(
            getattr(args, "rate_limit_retry_attempts", 3) or 0
        ),
        "rate_limit_retry_delay_seconds": float(
            getattr(args, "rate_limit_retry_delay_seconds", 2.0) or 0.0
        ),
    }


def runtime_policy_identity_for_args(args) -> Dict[str, Any]:
    config = {
        **runtime_adaptive_kwargs(args),
        "llm_provider": getattr(args, "llm_provider", None),
        "judge_model": (
            getattr(args, "judge_model", None)
            or getattr(args, "llm_model", None)
        ),
        "enable_source_tools": bool(
            getattr(args, "enable_source_tools", False)
        ),
        "max_view_chars": int(getattr(args, "max_view_chars", 0) or 0),
        "max_context_chars": int(
            getattr(args, "max_context_chars", 0) or 0
        ),
        "runtime_early_stop": bool(
            getattr(args, "enable_runtime_early_stop", False)
        ),
        "runtime_early_stop_policy": "conservative_negative",
    }
    return build_runtime_policy_identity(config)


def build_reviews(
    results: List[Dict[str, Any]],
    reviewer: RuleReviewer,
    only_tx_hashes: set[str] | None = None,
    cohort_signal_summary: Dict[str, Any] | None = None,
    plan_evidence_audit: Dict[str, Any] | None = None,
    case_boundary_context: Dict[str, Any] | None = None,
    rejected_update_memory: Dict[str, Any] | None = None,
    include_label_rationale: bool = False,
) -> List[Dict[str, Any]]:
    reviews: List[Dict[str, Any]] = []
    if cohort_signal_summary is None:
        cohort_signal_summary = build_cohort_signal_summary(
            [build_slim_result(result) for result in results]
        )
    if plan_evidence_audit is None:
        plan_evidence_audit = build_plan_evidence_audit(
            [build_slim_result(result) for result in results]
        )
    if case_boundary_context is None:
        case_boundary_context = build_case_boundary_context(
            [build_slim_result(result) for result in results],
            cohort_signal_summary=cohort_signal_summary,
            plan_evidence_audit=plan_evidence_audit,
        )
    only_tx_hashes = {str(tx).lower() for tx in (only_tx_hashes or set()) if str(tx)}
    rejected_update_memory = normalize_rejected_update_memory(
        rejected_update_memory or {}
    )
    for result in results:
        tx_hash = get_tx_hash(result)
        if only_tx_hashes and str(tx_hash).lower() not in only_tx_hashes:
            continue
        error_type = infer_error_type(result)
        predicted_verdict = str(
            result.get("predicted_verdict")
            or get_finding(result).get("verdict")
            or ""
        ).strip().lower()
        if error_type == "unknown" and predicted_verdict != "uncertain":
            continue
        reviewer_case = build_reviewer_case(
            result,
            include_label_rationale=include_label_rationale,
        )
        transport_review = _transport_error_review_from_case(reviewer_case)
        if transport_review:
            transport_review["tx_hash"] = tx_hash
            transport_review["ground_truth"] = get_ground_truth(result)
            transport_review["raw_ground_truth"] = get_raw_ground_truth(result)
            transport_review["sample_role"] = reviewer_case.get("sample_role", "")
            transport_review["negative_kind"] = reviewer_case.get("negative_kind", "")
            transport_review["target_label"] = reviewer_case.get("target_label", "")
            transport_review["predicted_verdict"] = get_finding(result).get("verdict")
            reviews.append(transport_review)
            continue
        reviewer_case["cohort_signal_summary"] = dict(cohort_signal_summary or {})
        reviewer_case["plan_evidence_audit"] = dict(plan_evidence_audit or {})
        reviewer_case["case_boundary_context"] = dict(case_boundary_context or {})
        reviewer_case["rejected_update_memory"] = dict(rejected_update_memory or {})
        review = reviewer.review_error(reviewer_case)
        review["tx_hash"] = tx_hash
        review["ground_truth"] = get_ground_truth(result)
        review["raw_ground_truth"] = get_raw_ground_truth(result)
        review["sample_role"] = reviewer_case.get("sample_role", "")
        review["negative_kind"] = reviewer_case.get("negative_kind", "")
        review["target_label"] = reviewer_case.get("target_label", "")
        review["predicted_verdict"] = get_finding(result).get("verdict")
        if include_label_rationale:
            training_supervision = dict(
                reviewer_case.get("training_supervision") or {}
            )
            if training_supervision:
                review["training_supervision"] = training_supervision
        reviews.append(review)
    return reviews


def _transport_error_review_from_case(
    reviewer_case: Dict[str, Any],
) -> Dict[str, Any]:
    rows = [
        row for row in list((reviewer_case or {}).get("condition_table") or [])
        if isinstance(row, dict)
    ]
    error_rows = []
    for row in rows:
        judge_error = row.get("judge_error") if isinstance(row.get("judge_error"), dict) else {}
        missing = {
            str(item or "").strip().lower()
            for item in list(row.get("missing_evidence") or [])
        }
        if judge_error or "judge_transport_error" in missing:
            error_rows.append(row)
    if not error_rows:
        return {}
    affected = [
        str(row.get("condition_id") or row.get("id") or "").strip()
        for row in error_rows
        if str(row.get("condition_id") or row.get("id") or "").strip()
    ]
    classes = sorted({
        str((row.get("judge_error") or {}).get("class") or "").strip()
        for row in error_rows
        if isinstance(row.get("judge_error"), dict)
        and str((row.get("judge_error") or {}).get("class") or "").strip()
    })
    error_type = str((reviewer_case or {}).get("error_type") or "unknown").upper()
    reason = (
        "One or more decisive Judge calls failed at the LLM transport/provider "
        "layer before producing a valid evidence judgment. This is not a "
        "trainable rule or plan signal."
    )
    return {
        "target_mechanism_observed": "uncertain",
        "ground_truth_supported_by_packet": "uncertain",
        "generic_symptoms_only": False,
        "counterfactual_boundary_risk": "high",
        "error_type": error_type,
        "update_target": "runtime",
        "root_causes": [{
            "category": "judge_transport_error",
            "description": reason,
            "affected_conditions": affected,
            "suggested_fix": (
                "Retry with provider-safe prompt/transport handling, switch "
                "provider, or exclude this run from rule/plan training."
            ),
        }],
        "condition_diagnosis": [
            {
                "condition_id": str(row.get("condition_id") or row.get("id") or ""),
                "observed_answer": row.get("answer"),
                "expected_for_correct_verdict": row.get("expected_answer"),
                "diagnosis": (
                    "Judge result is unavailable because the LLM call failed "
                    f"with class={((row.get('judge_error') or {}).get('class') or 'unknown')}."
                ),
                "is_rule_problem": False,
                "is_plan_problem": False,
                "is_packet_problem": False,
                "is_runtime_problem": True,
            }
            for row in error_rows
        ],
        "feature_diagnosis": [],
        "update_signals": [],
        "generalization_gate": {
            "enabled": True,
            "case_boundary_context_enabled": False,
            "supported_targets": [],
            "supported_signal_count": 0,
            "conflicted_signal_count": 0,
            "insufficient_signal_count": 0,
        },
        "rule_patch_suggestion": {
            "action": "none",
            "condition_id": "",
            "proposed_change": "",
            "rationale": "Transport/provider errors are not rule evidence.",
        },
        "plan_patch_suggestion": {
            "action": "none",
            "condition_id": "",
            "proposed_change": "",
            "rationale": "Transport/provider errors are not evidence-routing failures.",
        },
        "packet_patch_suggestion": {
            "action": "none",
            "condition_id": "",
            "proposed_change": "",
            "rationale": "Transport/provider errors occurred before packet evidence could be judged.",
        },
        "runtime_patch_suggestion": {
            "action": "fix_judge_transport_error_handling",
            "condition_id": ",".join(affected),
            "proposed_change": (
                "Classify judge transport failures separately from evidence "
                "uncertainty and prevent them from training rule/plan updates."
            ),
            "rationale": reason,
        },
        "should_update_rule": False,
        "should_update_plan_strategy": False,
        "should_update_packet_builder": False,
        "should_fix_runtime": True,
        "must_not_change": [
            "Do not change rule semantics from provider/content-filter failures.",
            "Do not change plan evidence routes from provider/content-filter failures.",
        ],
        "review_confidence": "high",
        "review_note": reason,
        "transport_error_classes": classes,
        "training_gate": {
            "rule_allowed": False,
            "plan_allowed": False,
            "blocked_reasons": {
                "rule": "judge_transport_error",
                "plan": "judge_transport_error",
            },
        },
        "do_not_train": True,
        "do_not_train_reasons": {
            "rule": "judge_transport_error",
            "plan": "judge_transport_error",
        },
        "raw_ground_truth": reviewer_case.get("raw_ground_truth"),
        "sample_role": reviewer_case.get("sample_role", ""),
        "negative_kind": reviewer_case.get("negative_kind", ""),
        "contrastive_supervision": dict(
            reviewer_case.get("contrastive_supervision") or {}
        ),
        "negative_training_mode": reviewer_case.get("negative_training_mode", "mixed"),
        "case_boundary_context_available": False,
        "target_label": reviewer_case.get("target_label", ""),
    }


def compare_train_and_guard(
    *,
    evaluator: RegressionEvaluator,
    args,
    current_full_results: List[Dict[str, Any]],
    new_full_results: List[Dict[str, Any]],
    current_guard_full_results: List[Dict[str, Any]] | None = None,
    new_guard_full_results: List[Dict[str, Any]] | None = None,
    hard_guard_cases: List[Dict[str, Any]] | None = None,
    acceptance_mode: str = "rule",
) -> Dict[str, Any]:
    if hard_guard_cases:
        return _apply_effective_candidate_gates(evaluator.compare_with_guard(
            current_full_results,
            new_full_results,
            old_guard_results=current_guard_full_results or [],
            new_guard_results=new_guard_full_results or [],
            max_fp_increase=args.max_fp_increase,
            max_fn_increase=args.max_fn_increase,
            max_uncertain_increase=args.max_uncertain_increase,
            max_uncertain_benign_increase=args.max_uncertain_benign_increase,
            enforce_uncertain_gate=bool(
                getattr(args, "enforce_uncertain_gate", True)
            ),
            max_guard_fp_increase=args.max_guard_fp_increase,
            max_guard_error_increase=args.max_guard_error_increase,
            max_guard_uncertain_increase=args.max_guard_uncertain_increase,
            acceptance_mode=acceptance_mode,
        ))
    return _apply_effective_candidate_gates(evaluator.compare(
        current_full_results,
        new_full_results,
        max_fp_increase=args.max_fp_increase,
        max_fn_increase=args.max_fn_increase,
        max_uncertain_increase=args.max_uncertain_increase,
        max_uncertain_benign_increase=args.max_uncertain_benign_increase,
        enforce_uncertain_gate=bool(
            getattr(args, "enforce_uncertain_gate", True)
        ),
        acceptance_mode=acceptance_mode,
    ))


def _uncertain_gate_blocks_acceptance(uncertain_gate: Dict[str, Any]) -> bool:
    if not isinstance(uncertain_gate, dict):
        return False
    if bool(uncertain_gate.get("enforced")):
        return True
    policy = str(uncertain_gate.get("policy") or "").strip().lower()
    if not policy:
        return False
    return policy not in {
        "diagnostic_only_under_attack_metric_acceptance",
        "disabled_for_final_strict_validation",
        "diagnostic",
        "warning",
    }


def _apply_effective_candidate_gates(comparison: Dict[str, Any]) -> Dict[str, Any]:
    comparison = dict(comparison or {})
    uncertain_gate = dict(comparison.get("uncertain_gate") or {})
    if (
        bool(comparison.get("accept"))
        and uncertain_gate
        and not bool(uncertain_gate.get("accept", True))
        and _uncertain_gate_blocks_acceptance(uncertain_gate)
    ):
        reason = str(uncertain_gate.get("reject_reason") or "uncertain gate rejected")
        comparison["accept"] = False
        comparison["accept_reason"] = ""
        comparison["reject_reason"] = reason
        comparison["effective_gate_rejection"] = {
            "gate": "uncertain_gate",
            "reason": reason,
        }
    return comparison


def _case_tx_set(cases: List[Dict[str, Any]] | None) -> set[str]:
    txs: set[str] = set()
    for case in cases or []:
        tx_hash = str(case.get("tx_hash") or case.get("hash") or "").strip().lower()
        if tx_hash:
            txs.add(tx_hash)
    return txs


def _result_tx_set(results: List[Dict[str, Any]] | None) -> set[str]:
    txs: set[str] = set()
    for result in results or []:
        tx_hash = str(get_tx_hash(result) or "").strip().lower()
        if tx_hash and tx_hash != "unknown":
            txs.add(tx_hash)
    return txs


def final_validation_reuse_check(
    *,
    cases: List[Dict[str, Any]],
    hard_guard_cases: List[Dict[str, Any]] | None,
    latest_final_full_results: List[Dict[str, Any]] | None,
    latest_final_guard_full_results: List[Dict[str, Any]] | None,
    current_rule: EvolvingRule | Dict[str, Any] | None = None,
    effective_plan: EvidencePlan | Dict[str, Any] | None = None,
    runtime_policy_identity: str = "",
) -> Dict[str, Any]:
    """Check whether cached final candidate/current results can replace rerun.

    Reuse is valid only for the exact artifact and runtime identities that
    produced the cached per-case judgments.
    """
    case_txs = _case_tx_set(cases)
    result_txs = _result_tx_set(latest_final_full_results)
    guard_case_txs = _case_tx_set(hard_guard_cases)
    guard_result_txs = _result_tx_set(latest_final_guard_full_results)
    reasons: List[str] = []
    expected_rule_fingerprint = (
        rule_fingerprint(current_rule) if current_rule is not None else ""
    )
    expected_plan_fingerprint = (
        semantic_plan_fingerprint(effective_plan)
        if effective_plan is not None
        else ""
    )

    if not latest_final_full_results:
        reasons.append("missing_cached_train_results")
    elif len(latest_final_full_results) != len(cases):
        reasons.append(
            "train_result_count_mismatch"
            f"(expected={len(cases)}, actual={len(latest_final_full_results)})"
        )
    if case_txs and result_txs and case_txs != result_txs:
        missing = sorted(case_txs - result_txs)
        extra = sorted(result_txs - case_txs)
        reasons.append(
            "train_tx_set_mismatch"
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    elif case_txs and not result_txs:
        reasons.append("cached_train_results_have_no_tx_hashes")

    if hard_guard_cases:
        if latest_final_guard_full_results is None:
            reasons.append("missing_cached_guard_results")
        elif len(latest_final_guard_full_results) != len(hard_guard_cases):
            reasons.append(
                "guard_result_count_mismatch"
                f"(expected={len(hard_guard_cases)}, "
                f"actual={len(latest_final_guard_full_results)})"
            )
        elif guard_case_txs and guard_result_txs and guard_case_txs != guard_result_txs:
            missing = sorted(guard_case_txs - guard_result_txs)
            extra = sorted(guard_result_txs - guard_case_txs)
            reasons.append(
                "guard_tx_set_mismatch"
                f"(missing={len(missing)}, extra={len(extra)})"
            )
        elif guard_case_txs and not guard_result_txs:
            reasons.append("cached_guard_results_have_no_tx_hashes")

    def validate_result_identity(
        result: Dict[str, Any],
        *,
        role: str,
    ) -> None:
        tx_hash = str(get_tx_hash(result) or "unknown").strip().lower()
        prefix = f"{role}:{tx_hash}"
        if expected_rule_fingerprint:
            actual_rule_fingerprint = rule_fingerprint(get_rule(result))
            if actual_rule_fingerprint != expected_rule_fingerprint:
                reasons.append(f"{prefix}:rule_fingerprint_mismatch")
        if expected_plan_fingerprint:
            actual_plan_fingerprint = semantic_plan_fingerprint(get_plan(result))
            if actual_plan_fingerprint != expected_plan_fingerprint:
                reasons.append(f"{prefix}:effective_plan_fingerprint_mismatch")

        trace_metadata = _runtime_trace_metadata(result)
        actual_runtime_identity = str(
            trace_metadata.get("runtime_policy_identity") or ""
        )
        if runtime_policy_identity:
            if not actual_runtime_identity:
                reasons.append(f"{prefix}:missing_runtime_policy_identity")
            elif actual_runtime_identity != runtime_policy_identity:
                reasons.append(f"{prefix}:runtime_policy_identity_mismatch")

        packet_identity = str(
            trace_metadata.get("packet_identity_fingerprint")
            or trace_metadata.get("packet_fingerprint")
            or ""
        )
        inference = dict(result.get("inference") or {})
        snapshot = dict(inference.get("packet_snapshot") or {})
        snapshot_identity = str(snapshot.get("packet_identity_fingerprint") or "")
        if not packet_identity:
            reasons.append(f"{prefix}:missing_packet_identity")
        if not snapshot_identity:
            reasons.append(f"{prefix}:missing_packet_snapshot_identity")
        if (
            packet_identity
            and snapshot_identity
            and packet_identity != snapshot_identity
        ):
            reasons.append(f"{prefix}:packet_identity_mismatch")

        plan_identity = dict(trace_metadata.get("plan_identity") or {})
        recorded_effective = str(
            plan_identity.get("effective_runtime_plan_fingerprint") or ""
        )
        if expected_plan_fingerprint and recorded_effective:
            if recorded_effective != expected_plan_fingerprint:
                reasons.append(f"{prefix}:recorded_effective_plan_mismatch")
        if plan_identity and not bool(plan_identity.get("runtime_preparation_match")):
            reasons.append(f"{prefix}:runtime_plan_preparation_mismatch")

    for result in list(latest_final_full_results or []):
        if isinstance(result, dict):
            validate_result_identity(result, role="train")
    for result in list(latest_final_guard_full_results or []):
        if isinstance(result, dict):
            validate_result_identity(result, role="guard")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "train_case_count": len(case_txs),
        "cached_train_result_count": len(result_txs),
        "guard_case_count": len(guard_case_txs),
        "cached_guard_result_count": len(guard_result_txs),
        "expected_rule_fingerprint": expected_rule_fingerprint,
        "expected_effective_plan_fingerprint": expected_plan_fingerprint,
        "expected_runtime_policy_identity": runtime_policy_identity,
        "note": (
            "Cached validation reuse compares the already-produced per-case "
            "judge results under the selected validation policy; it avoids a "
            "second stochastic judge pass."
        ),
    }


def write_unavailable_final_validation(
    *,
    artifacts_dir: Path,
    mode: str,
    evolution_mode: str,
    result_source_requested: str,
    reuse_check: Dict[str, Any],
) -> Dict[str, Any]:
    final_validation = {
        "enabled": True,
        "mode": mode,
        "evolution_mode": evolution_mode,
        "rerun": False,
        "result_source_requested": result_source_requested,
        "result_source_effective": "unavailable",
        "strict_policy": mode == "strict",
        "baseline_summary": {},
        "final_summary": {},
        "baseline_guard_summary": {},
        "final_guard_summary": {},
        "comparison": {},
        "passed": False,
        "pass_reason": "",
        "reject_reason": (
            "Cached final-validation results were requested but are missing "
            "or not comparable."
        ),
        "reuse_check": reuse_check,
        "note": (
            "No judge rerun was performed because result_source=reuse fails "
            "closed when cached results are unavailable or incomparable."
        ),
    }
    write_json(artifacts_dir / "final_validation_full_results.json", [])
    write_json(artifacts_dir / "final_validation_slim_results.json", [])
    write_json(artifacts_dir / "final_validation_summary.json", {})
    write_json(artifacts_dir / "final_validation_guard_full_results.json", [])
    write_json(artifacts_dir / "final_validation_guard_slim_results.json", [])
    write_json(artifacts_dir / "final_validation_guard_summary.json", {})
    write_json(artifacts_dir / "final_validation_comparison.json", {})
    write_json(artifacts_dir / "final_validation.json", final_validation)
    print(
        "[FewShot] Final validation reuse unavailable: "
        f"{stable_json_dumps(reuse_check)}"
    )
    return final_validation


def run_final_validation(
    *,
    args,
    cases: List[Dict[str, Any]],
    hard_guard_cases: List[Dict[str, Any]] | None,
    current_rule: EvolvingRule,
    current_rule_source: str,
    current_plan: EvidencePlan | None,
    current_plan_source: str | None,
    baseline_full_results: List[Dict[str, Any]] | None,
    baseline_summary: Dict[str, Any] | None,
    baseline_guard_full_results: List[Dict[str, Any]] | None,
    baseline_guard_summary: Dict[str, Any] | None,
    latest_final_full_results: List[Dict[str, Any]] | None,
    latest_final_guard_full_results: List[Dict[str, Any]] | None,
    artifacts_dir: Path,
    evaluator: RegressionEvaluator,
    tool_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    mode = str(getattr(args, "final_validation_mode", "strict") or "strict")
    if mode == "none":
        return {
            "enabled": False,
            "mode": "none",
            "evolution_mode": getattr(args, "evolution_mode", "explore"),
            "note": "Final validation skipped by --final-validation-mode none.",
        }

    validation_args = strict_policy_args(args) if mode == "strict" else args
    strict_policy = mode == "strict"
    result_source_requested = str(
        getattr(args, "final_validation_result_source", "auto") or "auto"
    )
    if bool(getattr(args, "force_final_validation_rerun", False)):
        result_source_requested = "rerun"
    effective_validation_plan = (
        prepare_effective_runtime_plan(
            current_rule,
            current_plan,
            iv_stateful_runtime=bool(args.iv_stateful_runtime),
            access_control_binding_mode=str(args.access_control_binding_mode),
            reentrancy_binding_mode=str(args.reentrancy_binding_mode),
            disable_stateful_bindings=bool(args.disable_stateful_bindings),
            judge_followup_mode=str(args.judge_followup_mode),
            judge_expanded_followup_views=int(
                args.judge_expanded_followup_views or 0
            ),
        )
        if current_plan is not None
        else None
    )
    expected_runtime_identity = runtime_policy_identity_for_args(args)
    reuse_check = final_validation_reuse_check(
        cases=cases,
        hard_guard_cases=hard_guard_cases,
        latest_final_full_results=latest_final_full_results,
        latest_final_guard_full_results=latest_final_guard_full_results,
        current_rule=current_rule,
        effective_plan=effective_validation_plan,
        runtime_policy_identity=str(
            expected_runtime_identity.get("fingerprint") or ""
        ),
    )
    if result_source_requested == "rerun":
        rerun = True
        result_source_effective = "rerun"
    elif result_source_requested == "reuse":
        if not bool(reuse_check.get("passed")):
            return write_unavailable_final_validation(
                artifacts_dir=artifacts_dir,
                mode=mode,
                evolution_mode=str(getattr(args, "evolution_mode", "explore")),
                result_source_requested=result_source_requested,
                reuse_check=reuse_check,
            )
        rerun = False
        result_source_effective = "reuse"
    else:
        rerun = not bool(reuse_check.get("passed"))
        result_source_effective = "rerun" if rerun else "reuse"

    print(
        "[FewShot] Final validation "
        f"mode={mode} result_source={result_source_requested} "
        f"effective={result_source_effective} rerun={str(rerun).lower()}"
    )
    if not bool(reuse_check.get("passed")):
        print(
            "[FewShot] Final validation cached reuse check: "
            f"{stable_json_dumps(reuse_check)}"
        )

    if rerun:
        final_full_results = run_cases(
            cases=cases,
            rule=current_rule,
            rule_source=current_rule_source,
            tool_manifest=tool_manifest,
            llm_provider=args.llm_provider,
            planner_model=args.planner_model or args.llm_model,
            judge_model=args.judge_model or args.llm_model,
            env_model=args.env_model or args.llm_model,
            use_environment=args.use_environment,
            base_cache_dir=args.base_cache_dir,
            force_rebuild_packet=args.force_rebuild_packet,
            enable_source_tools=args.enable_source_tools,
            source_cache_dir=args.source_cache_dir,
            force_refresh_source=args.force_refresh_source,
            max_view_chars=args.max_view_chars,
            max_context_chars=args.max_context_chars,
            phase="final_validation",
            llm_transcripts_dir=args.llm_transcripts_dir,
            fixed_plan=current_plan,
            fixed_plan_source=current_plan_source,
            runtime_early_stop=bool(args.enable_runtime_early_stop),
            **runtime_adaptive_kwargs(args),
        )
        final_guard_full_results: List[Dict[str, Any]] = []
        if hard_guard_cases:
            final_guard_full_results = run_cases(
                cases=hard_guard_cases,
                rule=current_rule,
                rule_source=current_rule_source,
                tool_manifest=tool_manifest,
                llm_provider=args.llm_provider,
                planner_model=args.planner_model or args.llm_model,
                judge_model=args.judge_model or args.llm_model,
                env_model=args.env_model or args.llm_model,
                use_environment=args.use_environment,
                base_cache_dir=args.base_cache_dir,
                force_rebuild_packet=args.force_rebuild_packet,
                enable_source_tools=args.enable_source_tools,
                source_cache_dir=args.source_cache_dir,
                force_refresh_source=args.force_refresh_source,
                max_view_chars=args.max_view_chars,
                max_context_chars=args.max_context_chars,
                phase="final_validation_guard",
                llm_transcripts_dir=args.llm_transcripts_dir,
                fixed_plan=current_plan,
                fixed_plan_source=current_plan_source,
                runtime_early_stop=bool(args.enable_runtime_early_stop),
                **runtime_adaptive_kwargs(args),
            )
    else:
        final_full_results = retarget_reused_results(
            latest_final_full_results or [],
            rule_source=current_rule_source,
            plan_source=current_plan_source,
            reuse_metadata={
                "reused": True,
                "reuse_type": "final_validation_latest_final_results",
                "rule_id": current_rule.rule_id,
                "rule_version": current_rule.version,
                "rule_source": current_rule_source,
                "plan_source": current_plan_source or "",
            },
        )
        final_guard_full_results = retarget_reused_results(
            latest_final_guard_full_results or [],
            rule_source=current_rule_source,
            plan_source=current_plan_source,
            reuse_metadata={
                "reused": True,
                "reuse_type": "final_validation_latest_final_guard_results",
                "rule_id": current_rule.rule_id,
                "rule_version": current_rule.version,
                "rule_source": current_rule_source,
                "plan_source": current_plan_source or "",
                "guard_role": "hard_negative_guard",
            },
        )

    final_slim_results = [build_slim_result(result) for result in final_full_results]
    final_summary = evaluator.summarize(final_full_results).to_dict()
    final_guard_slim_results = [
        build_slim_result(result) for result in final_guard_full_results
    ]
    final_guard_summary = (
        evaluator.summarize(final_guard_full_results).to_dict()
        if final_guard_full_results
        else {}
    )

    comparison: Dict[str, Any] = {}
    if baseline_full_results is not None:
        comparison = compare_train_and_guard(
            evaluator=evaluator,
            args=validation_args,
            current_full_results=baseline_full_results,
            new_full_results=final_full_results,
            current_guard_full_results=baseline_guard_full_results or [],
            new_guard_full_results=final_guard_full_results,
            hard_guard_cases=hard_guard_cases,
            acceptance_mode="rule",
        )
    passed = bool(comparison.get("accept")) if comparison else (
        int(final_summary.get("errors", 10**9) or 0) <= int(args.stop_errors or 0)
    )
    final_validation = {
        "enabled": True,
        "mode": mode,
        "evolution_mode": getattr(args, "evolution_mode", "explore"),
        "rerun": rerun,
        "result_source_requested": result_source_requested,
        "result_source_effective": result_source_effective,
        "reuse_check": reuse_check,
        "strict_policy": strict_policy,
        "baseline_summary": dict(baseline_summary or {}),
        "final_summary": final_summary,
        "baseline_guard_summary": dict(baseline_guard_summary or {}),
        "final_guard_summary": final_guard_summary,
        "comparison": comparison,
        "passed": passed,
        "pass_reason": comparison.get("accept_reason", "") if comparison else "",
        "reject_reason": comparison.get("reject_reason", "") if comparison else "",
        "policy_thresholds": {
            "max_uncertain_increase": validation_args.max_uncertain_increase,
            "max_uncertain_benign_increase": validation_args.max_uncertain_benign_increase,
            "max_guard_fp_increase": validation_args.max_guard_fp_increase,
            "max_guard_error_increase": validation_args.max_guard_error_increase,
            "max_guard_uncertain_increase": validation_args.max_guard_uncertain_increase,
            "enable_candidate_repair": bool(validation_args.enable_candidate_repair),
            "uncertain_gate": {
                "enforced": bool(
                    getattr(validation_args, "enforce_uncertain_gate", True)
                ),
                "policy": (
                    "configured_budget"
                    if bool(getattr(validation_args, "enforce_uncertain_gate", True))
                    else "disabled_for_final_strict_validation"
                ),
            },
        },
        "note": (
            "Final validation controls promotion of versioned artifacts to latest. "
            "Strict failure preserves version files but blocks latest aliases."
        ),
    }

    write_json(artifacts_dir / "final_validation_full_results.json", final_full_results)
    write_json(artifacts_dir / "final_validation_slim_results.json", final_slim_results)
    write_json(artifacts_dir / "final_validation_summary.json", final_summary)
    write_json(
        artifacts_dir / "final_validation_guard_full_results.json",
        final_guard_full_results,
    )
    write_json(
        artifacts_dir / "final_validation_guard_slim_results.json",
        final_guard_slim_results,
    )
    write_json(
        artifacts_dir / "final_validation_guard_summary.json",
        final_guard_summary,
    )
    write_json(artifacts_dir / "final_validation_comparison.json", comparison)
    write_json(artifacts_dir / "final_validation.json", final_validation)

    print(f"[FewShot] Final validation summary: {stable_json_dumps(final_summary)}")
    print(f"[FewShot] Final validation comparison: {stable_json_dumps(comparison)}")
    print(
        "[FewShot] Final validation "
        f"passed={str(passed).lower()} "
        f"reason={comparison.get('accept_reason') or comparison.get('reject_reason') or ''}"
    )
    return final_validation


def attempt_repair_rejected_candidate(
    *,
    args,
    cases: List[Dict[str, Any]],
    hard_guard_cases: List[Dict[str, Any]] | None,
    current_full_results: List[Dict[str, Any]],
    current_guard_full_results: List[Dict[str, Any]] | None,
    current_rule: EvolvingRule,
    current_rule_source: str,
    candidate_rule: EvolvingRule,
    candidate_plan: EvidencePlan | None,
    update_kind: str,
    candidate_full_results: List[Dict[str, Any]],
    candidate_guard_full_results: List[Dict[str, Any]] | None,
    candidate_slim_results: List[Dict[str, Any]],
    candidate_summary: Dict[str, Any],
    comparison: Dict[str, Any],
    round_dir: Path,
    round_index: int,
    reviewer: RuleReviewer,
    updater: RuleUpdater,
    plan_updater: PlanUpdater,
    evaluator: RegressionEvaluator,
    tool_manifest: Dict[str, Any],
    rejected_update_memory: Dict[str, Any] | None = None,
    acceptance_full_results: List[Dict[str, Any]] | None = None,
    acceptance_guard_full_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Try one targeted repair when a candidate is rejected due to new FPs.

    This keeps the normal regression gate intact: the repaired rule is accepted
    only if it beats the original current rule under the same comparison policy.
    """
    update_kind = str(update_kind or "")
    if update_kind == "plan":
        return attempt_repair_rejected_plan_candidate(
            args=args,
            cases=cases,
            hard_guard_cases=hard_guard_cases,
            current_rule=current_rule,
            current_rule_source=current_rule_source,
            current_full_results=current_full_results,
            current_guard_full_results=current_guard_full_results,
            candidate_plan=candidate_plan,
            candidate_full_results=candidate_full_results,
            candidate_guard_full_results=candidate_guard_full_results,
            candidate_slim_results=candidate_slim_results,
            candidate_summary=candidate_summary,
            comparison=comparison,
            round_dir=round_dir,
            round_index=round_index,
            reviewer=reviewer,
            plan_updater=plan_updater,
            evaluator=evaluator,
            tool_manifest=tool_manifest,
            rejected_update_memory=rejected_update_memory,
            acceptance_full_results=acceptance_full_results,
            acceptance_guard_full_results=acceptance_guard_full_results,
        )

    guard_repair = (
        bool(((comparison or {}).get("guard_gate") or {}).get("enabled"))
        and not bool(((comparison or {}).get("guard_gate") or {}).get("accept", True))
        and bool(getattr(args, "allow_guard_repair", False))
    )
    repair_comparison_source = (
        (comparison.get("guard_comparison") or {}) if guard_repair else comparison
    )
    repair_review_results = (
        list(candidate_guard_full_results or []) if guard_repair else candidate_full_results
    )
    new_fp_txs = new_fp_txs_from_comparison(repair_comparison_source)
    if not new_fp_txs:
        candidate_item = {
            "update_kind": update_kind,
            "rule": candidate_rule,
            "plan": candidate_plan,
            "full_results": candidate_full_results,
            "guard_full_results": candidate_guard_full_results or [],
            "slim_results": candidate_slim_results,
            "summary": candidate_summary,
            "comparison": comparison,
        }
        if not guard_repair and candidate_recall_repair_eligible(candidate_item, args=args):
            return attempt_new_fn_recall_repair(
                args=args,
                cases=cases,
                hard_guard_cases=hard_guard_cases,
                current_full_results=current_full_results,
                current_guard_full_results=current_guard_full_results,
                current_rule=current_rule,
                current_rule_source=current_rule_source,
                candidate_rule=candidate_rule,
                candidate_plan=candidate_plan,
                candidate_full_results=candidate_full_results,
                candidate_guard_full_results=candidate_guard_full_results,
                candidate_slim_results=candidate_slim_results,
                candidate_summary=candidate_summary,
                comparison=comparison,
                round_dir=round_dir,
                round_index=round_index,
                reviewer=reviewer,
                updater=updater,
                evaluator=evaluator,
                tool_manifest=tool_manifest,
                rejected_update_memory=rejected_update_memory,
                acceptance_full_results=acceptance_full_results,
                acceptance_guard_full_results=acceptance_guard_full_results,
            )
        recall_diagnostic = candidate_recall_repair_diagnostic(candidate_item, args=args)
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": False,
                "reason": (
                    "guard_rejected_without_new_fp_cases"
                    if guard_repair
                    else "candidate_rejected_without_new_fp_cases"
                ),
                "repair_source": "hard_negative_guard" if guard_repair else "train",
                "repair_type": "none",
                "fixed_fp_count": recall_diagnostic.get("fixed_fp_count", 0),
                "new_fn_count": recall_diagnostic.get("new_fn_count", 0),
                "fixed_fp_txs": recall_diagnostic.get("fixed_fp_txs", []),
                "new_fn_txs": recall_diagnostic.get("new_fn_txs", []),
                "recall_repair_reason": recall_diagnostic.get("reason", ""),
            },
        )
        return {"accepted": False, "reason": "no_new_fp_cases"}
    if len(new_fp_txs) > int(getattr(args, "temporary_fp_increase", 0) or 0):
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": False,
                "reason": "new_fp_count_exceeds_temporary_threshold",
                "repair_source": "hard_negative_guard" if guard_repair else "train",
                "new_fp_count": len(new_fp_txs),
                "temporary_fp_increase": int(getattr(args, "temporary_fp_increase", 0) or 0),
                "new_fp_txs": sorted(new_fp_txs),
            },
        )
        return {"accepted": False, "reason": "new_fp_threshold_exceeded"}

    print(
        "[FewShot] Candidate rejected with new FP cases; attempting one "
        f"targeted repair on {len(new_fp_txs)} new FP case(s) from "
        f"{'hard-negative guard' if guard_repair else 'train'}."
    )
    repair_guard_slim_results = [
        build_slim_result(result) for result in list(candidate_guard_full_results or [])
    ]
    repair_cohort_signal_summary = build_cohort_signal_summary(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
    )
    repair_plan_evidence_audit = build_plan_evidence_audit(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
    )
    repair_case_boundary_context = build_case_boundary_context(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
        cohort_signal_summary=repair_cohort_signal_summary,
        plan_evidence_audit=repair_plan_evidence_audit,
    )
    write_json(round_dir / "repair_case_boundary_context.json", repair_case_boundary_context)
    repair_reviews = build_reviews(
        repair_review_results,
        reviewer,
        only_tx_hashes=new_fp_txs,
        cohort_signal_summary=repair_cohort_signal_summary,
        plan_evidence_audit=repair_plan_evidence_audit,
        case_boundary_context=repair_case_boundary_context,
        rejected_update_memory=rejected_update_memory,
        include_label_rationale=bool(args.review_label_rationale),
    )
    write_json(round_dir / "repair_reviews.json", repair_reviews)
    if not repair_reviews:
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": True,
                "accepted": False,
                "reason": "no_actionable_reviews_for_new_fp_cases",
                "repair_source": "hard_negative_guard" if guard_repair else "train",
                "new_fp_txs": sorted(new_fp_txs),
            },
        )
        return {"accepted": False, "reason": "no_repair_reviews"}

    repair_bundle = build_round_review_bundle(
        candidate_rule,
        candidate_summary,
        repair_reviews,
        candidate_slim_results,
        negative_training_mode=args.negative_training_mode,
        negative_training_mode_source=args.negative_training_mode_source,
        cohort_signal_summary=repair_cohort_signal_summary,
        guard_slim_results=repair_guard_slim_results,
        plan_evidence_audit=repair_plan_evidence_audit,
        case_boundary_context=repair_case_boundary_context,
        rejected_update_memory=rejected_update_memory,
        enable_advanced_candidate_lifecycles=bool(
            getattr(args, "enable_phase3_candidate_portfolio", False)
        ),
    )
    repair_constraints = list(repair_bundle.get("update_constraints") or [])
    repair_constraints.extend([
        "This is a rejected-candidate repair pass driven only by new false positives.",
        "Prefer tightening or adding target-boundary exclusions over broadening core conditions.",
        "Do not leak or name the concrete non-target family used as negatives.",
    ])
    if guard_repair:
        repair_constraints.append(
            "The new false positives came from the fixed hard-negative guard set; "
            "use them only to tighten target boundaries, not to redefine the target."
        )
    repair_bundle["update_constraints"] = repair_constraints
    write_json(round_dir / "repair_review_bundle.json", repair_bundle)

    repair_rule = updater.update_rule_from_bundle(
        candidate_rule,
        repair_bundle,
        allow_noop=True,
    )
    if repair_rule.version == candidate_rule.version:
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": True,
                "accepted": False,
                "reason": "repair_rule_update_noop",
                "repair_source": "hard_negative_guard" if guard_repair else "train",
                "new_fp_txs": sorted(new_fp_txs),
            },
        )
        return {"accepted": False, "reason": "repair_noop"}

    if args.compress_rule:
        repair_rule = updater.compress_rule(repair_rule)

    repair_rule_path = round_dir / "repair_rule.json"
    write_json(repair_rule_path, repair_rule.to_dict())
    repair_plan, repair_plan_fallback = regenerate_candidate_plan(
        repair_rule,
        args=args,
        planner_guidance=build_planner_guidance_from_review_bundle(repair_bundle),
        current_rule=candidate_rule,
        current_plan=candidate_plan,
    )
    repair_plan_metadata = dict(repair_plan.metadata or {})
    repair_plan_metadata.update({
        "repair_kind": "rule",
        "candidate_strategy": "updated_rule_repair_regenerated_plan",
        "plan_generation_fallback": bool(repair_plan_fallback),
        "source": (
            "baseline_after_rule_repair_plan_generation_failure"
            if repair_plan_fallback
            else "llm_regenerated_for_rule_repair"
        ),
    })
    repair_plan.metadata = repair_plan_metadata
    repair_plan_path = round_dir / "repair_plan.json"
    write_json(repair_plan_path, repair_plan.to_dict())
    print(
        f"[FewShot] Repair rule prepared: {repair_rule.rule_id} v{repair_rule.version} "
        f"plan_regenerated=True fallback={repair_plan_fallback}"
    )

    repair_full_results = run_cases(
        cases=cases,
        rule=repair_rule,
        rule_source=str(repair_rule_path),
        tool_manifest=tool_manifest,
        llm_provider=args.llm_provider,
        planner_model=args.planner_model or args.llm_model,
        judge_model=args.judge_model or args.llm_model,
        env_model=args.env_model or args.llm_model,
        use_environment=args.use_environment,
        base_cache_dir=args.base_cache_dir,
        force_rebuild_packet=args.force_rebuild_packet,
        enable_source_tools=args.enable_source_tools,
        source_cache_dir=args.source_cache_dir,
        force_refresh_source=args.force_refresh_source,
        max_view_chars=args.max_view_chars,
        max_context_chars=args.max_context_chars,
        phase=f"round_{round_index:02d}/repair",
        llm_transcripts_dir=args.llm_transcripts_dir,
        fixed_plan=repair_plan,
        fixed_plan_source=str(repair_plan_path),
        reuse_results_by_tx=(
            build_candidate_judge_reuse_index(
                candidate_full_results,
                candidate_rule=repair_rule,
                candidate_plan=repair_plan,
                baseline_plan=candidate_plan,
                disable_stateful_bindings=bool(args.disable_stateful_bindings),
            )[0]
            if bool(getattr(args, "enable_candidate_judge_reuse", False))
            else None
        ),
        candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
        runtime_early_stop=bool(args.enable_runtime_early_stop),
        **runtime_adaptive_kwargs(args),
    )
    repair_slim_results = [build_slim_result(result) for result in repair_full_results]
    repair_summary = evaluator.summarize(repair_full_results).to_dict()
    repair_guard_full_results: List[Dict[str, Any]] = []
    repair_guard_slim_results: List[Dict[str, Any]] = []
    repair_guard_summary: Dict[str, Any] = {}
    if hard_guard_cases:
        repair_guard_full_results = run_cases(
            cases=hard_guard_cases,
            rule=repair_rule,
            rule_source=str(repair_rule_path),
            tool_manifest=tool_manifest,
            llm_provider=args.llm_provider,
            planner_model=args.planner_model or args.llm_model,
            judge_model=args.judge_model or args.llm_model,
            env_model=args.env_model or args.llm_model,
            use_environment=args.use_environment,
            base_cache_dir=args.base_cache_dir,
            force_rebuild_packet=args.force_rebuild_packet,
            enable_source_tools=args.enable_source_tools,
            source_cache_dir=args.source_cache_dir,
            force_refresh_source=args.force_refresh_source,
            max_view_chars=args.max_view_chars,
            max_context_chars=args.max_context_chars,
            phase=f"round_{round_index:02d}/repair_guard",
            llm_transcripts_dir=args.llm_transcripts_dir,
            fixed_plan=repair_plan,
            fixed_plan_source=str(repair_plan_path),
            reuse_results_by_tx=(
                build_candidate_judge_reuse_index(
                    candidate_guard_full_results or [],
                    candidate_rule=repair_rule,
                    candidate_plan=repair_plan,
                    baseline_plan=candidate_plan,
                    disable_stateful_bindings=bool(args.disable_stateful_bindings),
                )[0]
                if bool(getattr(args, "enable_candidate_judge_reuse", False))
                else None
            ),
            candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
            runtime_early_stop=bool(args.enable_runtime_early_stop),
            **runtime_adaptive_kwargs(args),
        )
        repair_guard_slim_results = [
            build_slim_result(result) for result in repair_guard_full_results
        ]
        repair_guard_summary = evaluator.summarize(repair_guard_full_results).to_dict()
        write_json(round_dir / "repair_guard_full_results.json", repair_guard_full_results)
        write_json(round_dir / "repair_guard_slim_results.json", repair_guard_slim_results)
        write_json(round_dir / "repair_guard_summary.json", repair_guard_summary)
    repair_comparison = compare_train_and_guard(
        evaluator=evaluator,
        args=args,
        current_full_results=current_full_results,
        new_full_results=repair_full_results,
        current_guard_full_results=current_guard_full_results,
        new_guard_full_results=repair_guard_full_results,
        hard_guard_cases=hard_guard_cases,
        acceptance_mode="rule",
    )

    write_json(round_dir / "repair_full_results.json", repair_full_results)
    write_json(round_dir / "repair_slim_results.json", repair_slim_results)
    write_json(round_dir / "repair_results.json", repair_slim_results)
    write_json(round_dir / "repair_summary.json", repair_summary)
    write_json(round_dir / "repair_comparison.json", repair_comparison)
    write_json(
        round_dir / "repair_status.json",
        {
            "attempted": True,
            "repair_kind": "rule",
            "accepted": bool(repair_comparison.get("accept")),
            "new_fp_txs": sorted(new_fp_txs),
            "repair_source": "hard_negative_guard" if guard_repair else "train",
            "temporary_fp_increase": int(getattr(args, "temporary_fp_increase", 0) or 0),
            "final_max_fp_increase": int(getattr(args, "max_fp_increase", 0) or 0),
            "repair_rule_id": repair_rule.rule_id,
            "repair_rule_version": repair_rule.version,
            "repair_plan_source": (repair_plan.metadata or {}).get("source", ""),
            "plan_regenerated": True,
            "plan_generation_fallback": bool(repair_plan_fallback),
            "fallback_reason": (repair_plan.metadata or {}).get("fallback_reason", ""),
            "guard_summary": repair_guard_summary,
            "accept_reason": repair_comparison.get("accept_reason", ""),
            "reject_reason": repair_comparison.get("reject_reason", ""),
        },
    )
    print(f"[FewShot] Repair summary: {stable_json_dumps(repair_summary)}")
    print(f"[FewShot] Repair comparison: {stable_json_dumps(repair_comparison)}")
    if not repair_comparison.get("accept"):
        return {"accepted": False, "reason": "repair_rejected"}
    return {
        "accepted": True,
        "repair_kind": "rule",
        "rule": repair_rule,
        "plan": repair_plan,
        "plan_regenerated": True,
        "plan_generation_fallback": bool(repair_plan_fallback),
        "summary": repair_summary,
        "full_results": repair_full_results,
        "comparison": repair_comparison,
    }


def attempt_new_fn_recall_repair(
    *,
    args,
    cases: List[Dict[str, Any]],
    hard_guard_cases: List[Dict[str, Any]] | None,
    current_full_results: List[Dict[str, Any]],
    current_guard_full_results: List[Dict[str, Any]] | None,
    current_rule: EvolvingRule,
    current_rule_source: str,
    candidate_rule: EvolvingRule,
    candidate_plan: EvidencePlan | None,
    candidate_full_results: List[Dict[str, Any]],
    candidate_guard_full_results: List[Dict[str, Any]] | None,
    candidate_slim_results: List[Dict[str, Any]],
    candidate_summary: Dict[str, Any],
    comparison: Dict[str, Any],
    round_dir: Path,
    round_index: int,
    reviewer: RuleReviewer,
    updater: RuleUpdater,
    evaluator: RegressionEvaluator,
    tool_manifest: Dict[str, Any],
    rejected_update_memory: Dict[str, Any] | None = None,
    acceptance_full_results: List[Dict[str, Any]] | None = None,
    acceptance_guard_full_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    diagnostic = candidate_recall_repair_diagnostic(
        {
            "update_kind": "rule",
            "rule": candidate_rule,
            "plan": candidate_plan,
            "full_results": candidate_full_results,
            "guard_full_results": candidate_guard_full_results or [],
            "slim_results": candidate_slim_results,
            "summary": candidate_summary,
            "comparison": comparison,
        },
        args=args,
    )
    if not diagnostic.get("eligible"):
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": False,
                "repair_kind": "rule_recall",
                "repair_type": "none",
                "reason": diagnostic.get("reason", "recall_repair_not_eligible"),
                **diagnostic,
            },
        )
        return {"accepted": False, "repair_kind": "rule_recall", "reason": diagnostic.get("reason", "")}

    new_fn_txs_all = _ordered_new_fn_txs(comparison)
    max_new_fn = max(0, int(getattr(args, "recall_repair_max_new_fn", 10) or 0))
    if max_new_fn > 0:
        review_new_fn_txs = set(new_fn_txs_all[:max_new_fn])
    else:
        review_new_fn_txs = set(new_fn_txs_all)
    omitted_new_fn_count = max(0, len(new_fn_txs_all) - len(review_new_fn_txs))
    fixed_fp_txs = fixed_fp_txs_from_comparison(comparison)
    guard_fixed_fp_txs = guard_fixed_fp_txs_from_comparison(comparison)
    guard_gate = dict((comparison or {}).get("guard_gate") or {})
    guard_accept = not guard_gate.get("enabled") or bool(guard_gate.get("accept", True))
    print(
        "[FewShot] Candidate rejected after fixing FP but introducing new FN; "
        "attempting recall repair: "
        f"fixed_fp={len(fixed_fp_txs)} new_fn={len(new_fn_txs_all)} "
        f"guard_accept={str(guard_accept).lower()}"
    )

    repair_guard_slim_results = [
        build_slim_result(result) for result in list(candidate_guard_full_results or [])
    ]
    repair_cohort_signal_summary = build_cohort_signal_summary(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
    )
    repair_plan_evidence_audit = build_plan_evidence_audit(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
    )
    repair_case_boundary_context = build_case_boundary_context(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
        cohort_signal_summary=repair_cohort_signal_summary,
        plan_evidence_audit=repair_plan_evidence_audit,
    )
    write_json(round_dir / "repair_case_boundary_context.json", repair_case_boundary_context)
    repair_reviews = build_reviews(
        candidate_full_results,
        reviewer,
        only_tx_hashes=review_new_fn_txs,
        cohort_signal_summary=repair_cohort_signal_summary,
        plan_evidence_audit=repair_plan_evidence_audit,
        case_boundary_context=repair_case_boundary_context,
        rejected_update_memory=rejected_update_memory,
        include_label_rationale=bool(args.review_label_rationale),
    )
    write_json(round_dir / "repair_reviews.json", repair_reviews)
    if not repair_reviews:
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": True,
                "accepted": False,
                "repair_kind": "rule_recall",
                "repair_type": "new_fn_recall_repair",
                "reason": "no_actionable_reviews_for_new_fn_cases",
                "base_rule": "candidate_rule",
                "new_fn_txs": sorted(review_new_fn_txs),
                "fixed_fp_txs": sorted(fixed_fp_txs),
                "guard_fixed_fp_txs": sorted(guard_fixed_fp_txs),
                "candidate_boundary_summary_present": False,
            },
        )
        return {"accepted": False, "repair_kind": "rule_recall", "reason": "no_repair_reviews"}

    repair_bundle = build_round_review_bundle(
        candidate_rule,
        candidate_summary,
        repair_reviews,
        candidate_slim_results,
        negative_training_mode=args.negative_training_mode,
        negative_training_mode_source=args.negative_training_mode_source,
        cohort_signal_summary=repair_cohort_signal_summary,
        guard_slim_results=repair_guard_slim_results,
        plan_evidence_audit=repair_plan_evidence_audit,
        case_boundary_context=repair_case_boundary_context,
        rejected_update_memory=rejected_update_memory,
        enable_advanced_candidate_lifecycles=bool(
            getattr(args, "enable_phase3_candidate_portfolio", False)
        ),
    )
    boundary_summary = _candidate_boundary_summary(
        evaluator=evaluator,
        current_full_results=current_full_results,
        candidate_summary=candidate_summary,
        comparison=comparison,
        fixed_fp_txs=fixed_fp_txs,
        new_fn_txs=set(new_fn_txs_all),
        guard_fixed_fp_txs=guard_fixed_fp_txs,
    )
    repair_bundle["candidate_boundary_summary"] = boundary_summary
    repair_bundle["repair_context"] = "new_fn_recall_repair"
    repair_constraints = list(repair_bundle.get("update_constraints") or [])
    repair_constraints.extend([
        "This is a rejected-candidate recall repair pass driven only by new false negatives introduced by the candidate.",
        "The base rule is the rejected candidate rule, not the original current rule.",
        "Preserve the candidate's fixed false-positive boundaries and hard-negative guard improvements.",
        "Do not revert to the original broad rule.",
        "Only broaden target-positive coverage where new-FN reviews show recurring target-family-specific features.",
        "Do not broaden based on generic exploit symptoms such as profit, flash loan, swap, value extraction, token callback, or reentrancy alone.",
        "If a new-FN review shows that a tightened condition is now too strict, relax only the minimal part needed to recover target-family positives.",
        "Keep exclusions and tightened non-target boundaries unless the new-FN reviews prove they wrongly exclude target positives.",
    ])
    repair_bundle["update_constraints"] = repair_constraints
    write_json(round_dir / "repair_review_bundle.json", repair_bundle)
    print(
        "[FewShot] Recall repair bundle: "
        f"reviews={len(repair_reviews)} fixed_fp_boundary={len(fixed_fp_txs)} "
        f"guard_fixed_fp={len(guard_fixed_fp_txs)}"
    )

    write_json(
        round_dir / "repair_status.json",
        {
            "attempted": True,
            "accepted": False,
            "repair_kind": "rule_recall",
            "repair_type": "new_fn_recall_repair",
            "reason": "recall_repair_started",
            "base_rule": "candidate_rule",
            "candidate_rule_version": candidate_rule.version,
            "new_fn_txs": sorted(review_new_fn_txs),
            "all_new_fn_txs": new_fn_txs_all,
            "truncated_new_fn_review": omitted_new_fn_count > 0,
            "omitted_new_fn_count": omitted_new_fn_count,
            "fixed_fp_txs": sorted(fixed_fp_txs),
            "guard_fixed_fp_txs": sorted(guard_fixed_fp_txs),
            "candidate_boundary_summary_present": True,
        },
    )

    repair_rule = updater.update_rule_from_bundle(
        candidate_rule,
        repair_bundle,
        allow_noop=True,
    )
    if repair_rule.version == candidate_rule.version:
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": True,
                "accepted": False,
                "repair_kind": "rule_recall",
                "repair_type": "new_fn_recall_repair",
                "reason": "recall_repair_rule_update_noop",
                "base_rule": "candidate_rule",
                "candidate_rule_version": candidate_rule.version,
                "repair_rule_version": repair_rule.version,
                "new_fn_txs": sorted(review_new_fn_txs),
                "fixed_fp_txs": sorted(fixed_fp_txs),
                "guard_fixed_fp_txs": sorted(guard_fixed_fp_txs),
                "candidate_boundary_summary_present": True,
            },
        )
        return {"accepted": False, "repair_kind": "rule_recall", "reason": "recall_repair_noop"}

    if args.compress_rule:
        repair_rule = updater.compress_rule(repair_rule)

    repair_rule_path = round_dir / "repair_rule.json"
    write_json(repair_rule_path, repair_rule.to_dict())
    repair_plan, repair_plan_fallback = regenerate_candidate_plan(
        repair_rule,
        args=args,
        planner_guidance=build_planner_guidance_from_review_bundle(repair_bundle),
        current_rule=candidate_rule,
        current_plan=candidate_plan,
    )
    repair_plan_metadata = dict(repair_plan.metadata or {})
    repair_plan_metadata.update({
        "repair_kind": "rule_recall",
        "repair_type": "new_fn_recall_repair",
        "candidate_strategy": "recall_repair_regenerated_plan",
        "plan_generation_fallback": bool(repair_plan_fallback),
        "source": (
            "baseline_after_rule_recall_repair_plan_generation_failure"
            if repair_plan_fallback
            else "llm_regenerated_for_rule_recall_repair"
        ),
    })
    repair_plan.metadata = repair_plan_metadata
    repair_plan_path = round_dir / "repair_plan.json"
    write_json(repair_plan_path, repair_plan.to_dict())

    repair_reuse_results, _ = (
        build_candidate_judge_reuse_index(
            candidate_full_results,
            candidate_rule=repair_rule,
            candidate_plan=repair_plan,
            baseline_plan=candidate_plan,
            disable_stateful_bindings=bool(args.disable_stateful_bindings),
        )
        if bool(getattr(args, "enable_candidate_judge_reuse", False))
        else ({}, {})
    )
    repair_full_results = run_cases(
        cases=cases,
        rule=repair_rule,
        rule_source=str(repair_rule_path),
        tool_manifest=tool_manifest,
        llm_provider=args.llm_provider,
        planner_model=args.planner_model or args.llm_model,
        judge_model=args.judge_model or args.llm_model,
        env_model=args.env_model or args.llm_model,
        use_environment=args.use_environment,
        base_cache_dir=args.base_cache_dir,
        force_rebuild_packet=args.force_rebuild_packet,
        enable_source_tools=args.enable_source_tools,
        source_cache_dir=args.source_cache_dir,
        force_refresh_source=args.force_refresh_source,
        max_view_chars=args.max_view_chars,
        max_context_chars=args.max_context_chars,
        phase=f"round_{round_index:02d}/recall_repair",
        llm_transcripts_dir=args.llm_transcripts_dir,
        fixed_plan=repair_plan,
        fixed_plan_source=str(repair_plan_path),
        reuse_results_by_tx=repair_reuse_results or None,
        candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
        runtime_early_stop=bool(args.enable_runtime_early_stop),
        **runtime_adaptive_kwargs(args),
    )
    repair_slim_results = [build_slim_result(result) for result in repair_full_results]
    repair_summary = evaluator.summarize(repair_full_results).to_dict()
    repair_guard_full_results: List[Dict[str, Any]] = []
    repair_guard_slim_results: List[Dict[str, Any]] = []
    repair_guard_summary: Dict[str, Any] = {}
    if hard_guard_cases:
        repair_guard_reuse_results, _ = (
            build_candidate_judge_reuse_index(
                candidate_guard_full_results or [],
                candidate_rule=repair_rule,
                candidate_plan=repair_plan,
                baseline_plan=candidate_plan,
                disable_stateful_bindings=bool(args.disable_stateful_bindings),
            )
            if bool(getattr(args, "enable_candidate_judge_reuse", False))
            else ({}, {})
        )
        repair_guard_full_results = run_cases(
            cases=hard_guard_cases,
            rule=repair_rule,
            rule_source=str(repair_rule_path),
            tool_manifest=tool_manifest,
            llm_provider=args.llm_provider,
            planner_model=args.planner_model or args.llm_model,
            judge_model=args.judge_model or args.llm_model,
            env_model=args.env_model or args.llm_model,
            use_environment=args.use_environment,
            base_cache_dir=args.base_cache_dir,
            force_rebuild_packet=args.force_rebuild_packet,
            enable_source_tools=args.enable_source_tools,
            source_cache_dir=args.source_cache_dir,
            force_refresh_source=args.force_refresh_source,
            max_view_chars=args.max_view_chars,
            max_context_chars=args.max_context_chars,
            phase=f"round_{round_index:02d}/recall_repair_guard",
            llm_transcripts_dir=args.llm_transcripts_dir,
            fixed_plan=repair_plan,
            fixed_plan_source=str(repair_plan_path),
            reuse_results_by_tx=repair_guard_reuse_results or None,
            candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
            runtime_early_stop=bool(args.enable_runtime_early_stop),
            **runtime_adaptive_kwargs(args),
        )
        repair_guard_slim_results = [
            build_slim_result(result) for result in repair_guard_full_results
        ]
        repair_guard_summary = evaluator.summarize(repair_guard_full_results).to_dict()
        write_json(round_dir / "repair_guard_full_results.json", repair_guard_full_results)
        write_json(round_dir / "repair_guard_slim_results.json", repair_guard_slim_results)
        write_json(round_dir / "repair_guard_summary.json", repair_guard_summary)

    repair_comparison = compare_train_and_guard(
        evaluator=evaluator,
        args=args,
        current_full_results=acceptance_full_results or current_full_results,
        new_full_results=repair_full_results,
        current_guard_full_results=(
            acceptance_guard_full_results
            if acceptance_guard_full_results is not None
            else current_guard_full_results
        ),
        new_guard_full_results=repair_guard_full_results,
        hard_guard_cases=hard_guard_cases,
        acceptance_mode="rule",
    )

    write_json(round_dir / "repair_full_results.json", repair_full_results)
    write_json(round_dir / "repair_slim_results.json", repair_slim_results)
    write_json(round_dir / "repair_results.json", repair_slim_results)
    write_json(round_dir / "repair_summary.json", repair_summary)
    write_json(round_dir / "repair_comparison.json", repair_comparison)
    write_json(
        round_dir / "repair_status.json",
        {
            "attempted": True,
            "accepted": bool(repair_comparison.get("accept")),
            "repair_kind": "rule_recall",
            "repair_type": "new_fn_recall_repair",
            "base_rule": "candidate_rule",
            "candidate_rule_version": candidate_rule.version,
            "repair_rule_version": repair_rule.version,
            "new_fn_txs": sorted(review_new_fn_txs),
            "all_new_fn_txs": new_fn_txs_all,
            "truncated_new_fn_review": omitted_new_fn_count > 0,
            "omitted_new_fn_count": omitted_new_fn_count,
            "fixed_fp_txs": sorted(fixed_fp_txs),
            "guard_fixed_fp_txs": sorted(guard_fixed_fp_txs),
            "candidate_boundary_summary_present": True,
            "repair_plan_source": (repair_plan.metadata or {}).get("source", ""),
            "plan_regenerated": True,
            "plan_generation_fallback": bool(repair_plan_fallback),
            "fallback_reason": (repair_plan.metadata or {}).get("fallback_reason", ""),
            "accept_reason": repair_comparison.get("accept_reason", ""),
            "reject_reason": repair_comparison.get("reject_reason", ""),
            "repair_summary": repair_summary,
            "guard_summary": repair_guard_summary,
        },
    )
    print(f"[FewShot] Recall repair summary: {stable_json_dumps(repair_summary)}")
    print(f"[FewShot] Recall repair comparison: {stable_json_dumps(repair_comparison)}")
    if not repair_comparison.get("accept"):
        return {"accepted": False, "repair_kind": "rule_recall", "reason": "recall_repair_rejected"}
    return {
        "accepted": True,
        "repair_kind": "rule_recall",
        "rule": repair_rule,
        "plan": repair_plan,
        "plan_regenerated": True,
        "plan_generation_fallback": bool(repair_plan_fallback),
        "summary": repair_summary,
        "full_results": repair_full_results,
        "guard_full_results": repair_guard_full_results,
        "guard_summary": repair_guard_summary,
        "comparison": repair_comparison,
    }


def _ordered_new_fn_txs(comparison: Dict[str, Any]) -> List[str]:
    ordered: List[str] = []
    for item in list((comparison or {}).get("regressed_cases", []) or []) + list(
        (comparison or {}).get("per_case_changes", []) or []
    ):
        if not isinstance(item, dict) or not item.get("tx_hash"):
            continue
        tx_hash = str(item.get("tx_hash")).lower()
        if tx_hash in ordered:
            continue
        if tx_hash in new_fn_txs_from_comparison({"per_case_changes": [item]}):
            ordered.append(tx_hash)
    return ordered


def _candidate_boundary_summary(
    *,
    evaluator: RegressionEvaluator,
    current_full_results: List[Dict[str, Any]],
    candidate_summary: Dict[str, Any],
    comparison: Dict[str, Any],
    fixed_fp_txs: set[str],
    new_fn_txs: set[str],
    guard_fixed_fp_txs: set[str],
) -> Dict[str, Any]:
    guard_gate = dict((comparison or {}).get("guard_gate") or {})
    return {
        "source": "rejected_rule_candidate",
        "purpose": "preserve_fixed_fp_boundary_during_recall_repair",
        "current_summary": evaluator.summarize(current_full_results).to_dict(),
        "candidate_summary": dict(candidate_summary or {}),
        "train_delta": dict((comparison or {}).get("delta") or {}),
        "fixed_fp_count": len(fixed_fp_txs),
        "fixed_fp_txs": sorted(fixed_fp_txs),
        "new_fn_count": len(new_fn_txs),
        "new_fn_txs": sorted(new_fn_txs),
        "guard_accept": not guard_gate.get("enabled") or bool(guard_gate.get("accept", True)),
        "guard_delta": dict(guard_gate.get("delta") or {}),
        "guard_fixed_fp_count": len(guard_fixed_fp_txs),
        "guard_fixed_fp_txs": sorted(guard_fixed_fp_txs),
        "boundary_preservation_instruction": (
            "The candidate fixed these false positives and guard false positives. "
            "Recall repair should preserve this non-target boundary and only restore "
            "target-positive coverage supported by new-FN reviews."
        ),
    }


def attempt_repair_rejected_plan_candidate(
    *,
    args,
    cases: List[Dict[str, Any]],
    hard_guard_cases: List[Dict[str, Any]] | None,
    current_rule: EvolvingRule,
    current_rule_source: str,
    current_full_results: List[Dict[str, Any]],
    current_guard_full_results: List[Dict[str, Any]] | None,
    candidate_plan: EvidencePlan | None,
    candidate_full_results: List[Dict[str, Any]],
    candidate_guard_full_results: List[Dict[str, Any]] | None,
    candidate_slim_results: List[Dict[str, Any]],
    candidate_summary: Dict[str, Any],
    comparison: Dict[str, Any],
    round_dir: Path,
    round_index: int,
    reviewer: RuleReviewer,
    plan_updater: PlanUpdater,
    evaluator: RegressionEvaluator,
    tool_manifest: Dict[str, Any],
    rejected_update_memory: Dict[str, Any] | None = None,
    acceptance_full_results: List[Dict[str, Any]] | None = None,
    acceptance_guard_full_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    guard_repair = (
        bool(((comparison or {}).get("guard_gate") or {}).get("enabled"))
        and not bool(((comparison or {}).get("guard_gate") or {}).get("accept", True))
        and bool(getattr(args, "allow_guard_repair", False))
    )
    repair_comparison_source = (
        (comparison.get("guard_comparison") or {}) if guard_repair else comparison
    )
    repair_review_results = (
        list(candidate_guard_full_results or []) if guard_repair else candidate_full_results
    )
    new_fp_txs = new_fp_txs_from_comparison(repair_comparison_source)
    if not new_fp_txs:
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": False,
                "repair_kind": "plan",
                "reason": (
                    "guard_rejected_without_new_fp_cases"
                    if guard_repair
                    else "candidate_rejected_without_new_fp_cases"
                ),
                "repair_source": "hard_negative_guard" if guard_repair else "train",
            },
        )
        return {"accepted": False, "repair_kind": "plan", "reason": "no_new_fp_cases"}
    if len(new_fp_txs) > int(getattr(args, "temporary_fp_increase", 0) or 0):
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": False,
                "repair_kind": "plan",
                "reason": "new_fp_count_exceeds_temporary_threshold",
                "repair_source": "hard_negative_guard" if guard_repair else "train",
                "new_fp_count": len(new_fp_txs),
                "temporary_fp_increase": int(getattr(args, "temporary_fp_increase", 0) or 0),
                "new_fp_txs": sorted(new_fp_txs),
            },
        )
        return {"accepted": False, "repair_kind": "plan", "reason": "new_fp_threshold_exceeded"}

    print(
        "[FewShot] Repair selected: update_kind=plan "
        f"reason=new_fp_txs count={len(new_fp_txs)} "
        f"source={'hard-negative guard' if guard_repair else 'train'}"
    )
    repair_guard_slim_results = [
        build_slim_result(result) for result in list(candidate_guard_full_results or [])
    ]
    repair_cohort_signal_summary = build_cohort_signal_summary(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
    )
    repair_plan_evidence_audit = build_plan_evidence_audit(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
    )
    repair_case_boundary_context = build_case_boundary_context(
        candidate_slim_results,
        guard_slim_results=repair_guard_slim_results,
        cohort_signal_summary=repair_cohort_signal_summary,
        plan_evidence_audit=repair_plan_evidence_audit,
    )
    write_json(round_dir / "repair_case_boundary_context.json", repair_case_boundary_context)
    repair_reviews = build_reviews(
        repair_review_results,
        reviewer,
        only_tx_hashes=new_fp_txs,
        cohort_signal_summary=repair_cohort_signal_summary,
        plan_evidence_audit=repair_plan_evidence_audit,
        case_boundary_context=repair_case_boundary_context,
        rejected_update_memory=rejected_update_memory,
        include_label_rationale=bool(args.review_label_rationale),
    )
    write_json(round_dir / "repair_reviews.json", repair_reviews)
    if not repair_reviews:
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": True,
                "repair_kind": "plan",
                "accepted": False,
                "reason": "no_actionable_reviews_for_new_fp_cases",
                "repair_source": "hard_negative_guard" if guard_repair else "train",
                "new_fp_txs": sorted(new_fp_txs),
            },
        )
        return {"accepted": False, "repair_kind": "plan", "reason": "no_repair_reviews"}

    base_plan = candidate_plan or compile_rule_to_baseline_plan(current_rule)
    repair_bundle = build_round_review_bundle(
        current_rule,
        candidate_summary,
        repair_reviews,
        candidate_slim_results,
        negative_training_mode=args.negative_training_mode,
        negative_training_mode_source=args.negative_training_mode_source,
        cohort_signal_summary=repair_cohort_signal_summary,
        guard_slim_results=repair_guard_slim_results,
        plan_evidence_audit=repair_plan_evidence_audit,
        case_boundary_context=repair_case_boundary_context,
        rejected_update_memory=rejected_update_memory,
        enable_advanced_candidate_lifecycles=bool(
            getattr(args, "enable_phase3_candidate_portfolio", False)
        ),
    )
    repair_constraints = list(repair_bundle.get("update_constraints") or [])
    repair_constraints.extend([
        "This is a rejected plan-candidate repair pass driven only by new false positives.",
        "Do not change the semantic rule.",
        "Modify only EvidencePlan: judge question clarity, default_evidence_refs, allowed_followup_views, allowed_tools, max_followups, or emit_logic if needed.",
        "Prefer tightening target-boundary exclusions or evidence requirements rather than broadening root-cause conditions.",
    ])
    if guard_repair:
        repair_constraints.append(
            "The new false positives came from the fixed hard-negative guard set; "
            "use them only to tighten plan evidence strategy and target boundaries."
        )
    repair_bundle["update_constraints"] = repair_constraints
    write_json(round_dir / "repair_review_bundle.json", repair_bundle)

    repaired_plan = plan_updater.update_plan_from_bundle(
        current_rule,
        repair_bundle,
        base_plan=base_plan,
        allow_noop=True,
    )
    repaired_plan = normalize_plan_for_runtime_policy(
        repaired_plan,
        args=args,
        reason="plan_candidate_repair",
    )
    repair_scope_guard = dict((repaired_plan.metadata or {}).get("scope_guard") or {})
    print(
        "[FewShot] Repair plan update semantic changed: "
        f"{_plan_changed(repaired_plan, base_plan)}; "
        f"scope targets={repair_scope_guard.get('target_steps', [])}"
    )
    if not _plan_changed(repaired_plan, base_plan):
        write_json(
            round_dir / "repair_status.json",
            {
                "attempted": True,
                "repair_kind": "plan",
                "accepted": False,
                "reason": "repair_plan_update_noop",
                "repair_source": "hard_negative_guard" if guard_repair else "train",
                "new_fp_txs": sorted(new_fp_txs),
            },
        )
        return {"accepted": False, "repair_kind": "plan", "reason": "repair_noop"}

    repaired_plan_path = round_dir / "repair_plan.json"
    write_json(repaired_plan_path, repaired_plan.to_dict())
    print(
        f"[FewShot] Repair plan prepared: {repaired_plan.plan_id} "
        f"plan_v={(repaired_plan.metadata or {}).get('plan_version')}"
    )

    repair_full_results = run_cases(
        cases=cases,
        rule=current_rule,
        rule_source=current_rule_source,
        tool_manifest=tool_manifest,
        llm_provider=args.llm_provider,
        planner_model=args.planner_model or args.llm_model,
        judge_model=args.judge_model or args.llm_model,
        env_model=args.env_model or args.llm_model,
        use_environment=args.use_environment,
        base_cache_dir=args.base_cache_dir,
        force_rebuild_packet=args.force_rebuild_packet,
        enable_source_tools=args.enable_source_tools,
        source_cache_dir=args.source_cache_dir,
        force_refresh_source=args.force_refresh_source,
        max_view_chars=args.max_view_chars,
        max_context_chars=args.max_context_chars,
        phase=f"round_{round_index:02d}/repair_plan",
        llm_transcripts_dir=args.llm_transcripts_dir,
        fixed_plan=repaired_plan,
        fixed_plan_source=str(repaired_plan_path),
        reuse_results_by_tx=(
            build_candidate_judge_reuse_index(
                candidate_full_results,
                candidate_rule=current_rule,
                candidate_plan=repaired_plan,
                baseline_plan=candidate_plan,
                disable_stateful_bindings=bool(args.disable_stateful_bindings),
            )[0]
            if bool(getattr(args, "enable_candidate_judge_reuse", False))
            else None
        ),
        candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
        runtime_early_stop=bool(args.enable_runtime_early_stop),
        **runtime_adaptive_kwargs(args),
    )
    repair_slim_results = [build_slim_result(result) for result in repair_full_results]
    repair_summary = evaluator.summarize(repair_full_results).to_dict()
    repair_guard_full_results: List[Dict[str, Any]] = []
    repair_guard_slim_results: List[Dict[str, Any]] = []
    repair_guard_summary: Dict[str, Any] = {}
    if hard_guard_cases:
        repair_guard_full_results = run_cases(
            cases=hard_guard_cases,
            rule=current_rule,
            rule_source=current_rule_source,
            tool_manifest=tool_manifest,
            llm_provider=args.llm_provider,
            planner_model=args.planner_model or args.llm_model,
            judge_model=args.judge_model or args.llm_model,
            env_model=args.env_model or args.llm_model,
            use_environment=args.use_environment,
            base_cache_dir=args.base_cache_dir,
            force_rebuild_packet=args.force_rebuild_packet,
            enable_source_tools=args.enable_source_tools,
            source_cache_dir=args.source_cache_dir,
            force_refresh_source=args.force_refresh_source,
            max_view_chars=args.max_view_chars,
            max_context_chars=args.max_context_chars,
            phase=f"round_{round_index:02d}/repair_plan_guard",
            llm_transcripts_dir=args.llm_transcripts_dir,
            fixed_plan=repaired_plan,
            fixed_plan_source=str(repaired_plan_path),
            reuse_results_by_tx=(
                build_candidate_judge_reuse_index(
                    candidate_guard_full_results or [],
                    candidate_rule=current_rule,
                    candidate_plan=repaired_plan,
                    baseline_plan=candidate_plan,
                    disable_stateful_bindings=bool(args.disable_stateful_bindings),
                )[0]
                if bool(getattr(args, "enable_candidate_judge_reuse", False))
                else None
            ),
            candidate_judge_reuse_enabled=bool(args.enable_candidate_judge_reuse),
            runtime_early_stop=bool(args.enable_runtime_early_stop),
            **runtime_adaptive_kwargs(args),
        )
        repair_guard_slim_results = [
            build_slim_result(result) for result in repair_guard_full_results
        ]
        repair_guard_summary = evaluator.summarize(repair_guard_full_results).to_dict()
        write_json(round_dir / "repair_guard_full_results.json", repair_guard_full_results)
        write_json(round_dir / "repair_guard_slim_results.json", repair_guard_slim_results)
        write_json(round_dir / "repair_guard_summary.json", repair_guard_summary)

    repair_comparison = compare_train_and_guard(
        evaluator=evaluator,
        args=args,
        current_full_results=acceptance_full_results or current_full_results,
        new_full_results=repair_full_results,
        current_guard_full_results=(
            acceptance_guard_full_results
            if acceptance_guard_full_results is not None
            else current_guard_full_results
        ),
        new_guard_full_results=repair_guard_full_results,
        hard_guard_cases=hard_guard_cases,
        acceptance_mode="plan",
    )

    write_json(round_dir / "repair_full_results.json", repair_full_results)
    write_json(round_dir / "repair_slim_results.json", repair_slim_results)
    write_json(round_dir / "repair_results.json", repair_slim_results)
    write_json(round_dir / "repair_summary.json", repair_summary)
    write_json(round_dir / "repair_comparison.json", repair_comparison)
    write_json(
        round_dir / "repair_status.json",
        {
            "attempted": True,
            "repair_kind": "plan",
            "accepted": bool(repair_comparison.get("accept")),
            "new_fp_txs": sorted(new_fp_txs),
            "repair_source": "hard_negative_guard" if guard_repair else "train",
            "temporary_fp_increase": int(getattr(args, "temporary_fp_increase", 0) or 0),
            "final_max_fp_increase": int(getattr(args, "max_fp_increase", 0) or 0),
            "plan_id": repaired_plan.plan_id,
            "plan_version": (repaired_plan.metadata or {}).get("plan_version"),
            "guard_summary": repair_guard_summary,
            "accept_reason": repair_comparison.get("accept_reason", ""),
            "reject_reason": repair_comparison.get("reject_reason", ""),
        },
    )
    print(f"[FewShot] Repair plan summary: {stable_json_dumps(repair_summary)}")
    print(f"[FewShot] Repair plan comparison: {stable_json_dumps(repair_comparison)}")
    if not repair_comparison.get("accept"):
        return {"accepted": False, "repair_kind": "plan", "reason": "repair_rejected"}
    return {
        "accepted": True,
        "repair_kind": "plan",
        "rule": current_rule,
        "plan": repaired_plan,
        "summary": repair_summary,
        "full_results": repair_full_results,
        "guard_full_results": repair_guard_full_results,
        "guard_summary": repair_guard_summary,
        "comparison": repair_comparison,
    }


if __name__ == "__main__":
    main()
