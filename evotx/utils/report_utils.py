from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import EvolvingRule
from evotx.core.schemas import EvidencePlan


FEWSHOT_REPORT_SCHEMA_VERSION = "evotx.report.fewshot.v1"


def build_fewshot_final_report(
    *,
    attack_label: str,
    attack_description: str,
    cases_file: Optional[str],
    benign_csv: Optional[str],
    malicious_csv: Optional[str],
    pos_csv: Optional[str],
    neg_csv: Optional[str],
    episode: Optional[int],
    tool_manifest_path: str,
    use_environment: bool,
    base_cache_dir: str,
    force_rebuild_packet: bool,
    enable_source_tools: bool = False,
    source_cache_dir: str = "data/cache/contracts",
    force_refresh_source: bool = False,
    max_view_chars: int = 20000,
    max_context_chars: int = 60000,
    rules_dir: str,
    artifacts_dir: str | Path,
    llm_model: Optional[str],
    llm_provider: Optional[str],
    adaptive_model: Optional[str],
    adaptive_provider: Optional[str],
    cold_start_model: Optional[str],
    planner_model: Optional[str],
    judge_model: Optional[str],
    env_model: Optional[str],
    review_model: Optional[str],
    update_model: Optional[str],
    max_rounds: int,
    stop_errors: int,
    max_fp_increase: int,
    max_fn_increase: int,
    max_uncertain_increase: int,
    max_uncertain_benign_increase: int,
    compress_rule: bool,
    enable_candidate_repair: bool,
    temporary_fp_increase: int,
    repair_min_fn_decrease: int,
    max_repair_rounds: int,
    label_counts: Dict[str, int],
    negative_training_mode: str,
    negative_training_mode_source: str,
    initial_rule: EvolvingRule,
    initial_rule_path: str | Path,
    final_rule: EvolvingRule,
    final_rule_path: str | Path,
    accepted_rounds: int,
    stop_reason: str,
    final_summary: Dict[str, Any],
    plans_dir: Optional[str] = None,
    initial_plan: Optional[EvidencePlan] = None,
    initial_plan_path: Optional[str | Path] = None,
    final_plan: Optional[EvidencePlan] = None,
    final_plan_path: Optional[str | Path] = None,
    run_timing: Optional[Dict[str, Any]] = None,
    llm_runtime_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    artifacts_root = Path(artifacts_dir)
    round_dirs = [
        str(path)
        for path in sorted(artifacts_root.glob("round_*"))
        if path.is_dir()
    ]

    return {
        "schema_version": FEWSHOT_REPORT_SCHEMA_VERSION,
        "stage": "few_shot_evolving",
        "episode": episode,
        "run_timing": dict(run_timing or {}),
        "task": {
            "attack_label": normalize_attack_label(attack_label),
            "raw_attack_label": attack_label,
            "attack_description": attack_description,
        },
        "inputs": {
            "case_sources": {
                "cases_file": cases_file,
                "pos_csv": pos_csv,
                "neg_csv": neg_csv,
                "benign_csv": benign_csv,
                "malicious_csv": malicious_csv,
                "negative_csv_semantics": (
                    "mixed_non_target" if neg_csv else "strict_benign" if benign_csv else ""
                ),
                "negative_training_mode": negative_training_mode,
                "negative_training_mode_source": negative_training_mode_source,
            },
            "tool_manifest": tool_manifest_path,
            "runtime": {
                "type": "packet_runtime",
                "use_environment": use_environment,
                "base_cache_dir": base_cache_dir,
                "force_rebuild_packet": force_rebuild_packet,
                "max_view_chars": max_view_chars,
                "max_context_chars": max_context_chars,
                "source_followup": {
                    "enabled": enable_source_tools,
                    "tool": "read_function_chunk",
                    "max_hops": 1,
                    "source_cache_dir": source_cache_dir,
                    "force_refresh_source": force_refresh_source,
                },
            },
            "models": {
                "llm_model": llm_model,
                "llm_provider": llm_provider,
                "adaptive_model": adaptive_model,
                "adaptive_provider": adaptive_provider,
                "cold_start_model": cold_start_model,
                "planner_model": planner_model,
                "judge_model": judge_model,
                "env_model": env_model,
                "review_model": review_model,
                "update_model": update_model,
                "resolved_runtime_config": dict(llm_runtime_config or {}),
            },
            "stop_policy": {
                "max_rounds": max_rounds,
                "stop_errors": stop_errors,
                "max_fp_increase": max_fp_increase,
                "max_fn_increase": max_fn_increase,
                "max_uncertain_increase": max_uncertain_increase,
                "max_uncertain_benign_increase": max_uncertain_benign_increase,
                "uncertain_regression_guard": {
                    "enabled": True,
                    "note": (
                        "Primary errors remain fp + fn. Uncertain increases are "
                        "a separate safety gate; benign->uncertain is treated as "
                        "a regression unless explicitly allowed."
                    ),
                },
                "candidate_repair": {
                    "enabled": enable_candidate_repair,
                    "temporary_fp_increase": temporary_fp_increase,
                    "repair_min_fn_decrease": repair_min_fn_decrease,
                    "max_repair_rounds": max_repair_rounds,
                    "final_acceptance_policy": (
                        "accept if attack_recall and attack_precision do not "
                        "decrease and at least one improves; FP/FN/uncertain "
                        "counts are diagnostic, while hard-negative guard "
                        "thresholds can still veto boundary regressions"
                    ),
                },
                "compress_rule": compress_rule,
            },
        },
        "dataset": {
            "label_counts": dict(label_counts),
        },
        "detector_context": {
            "initial_rule": _rule_ref(initial_rule, initial_rule_path),
            "final_rule": _rule_ref(final_rule, final_rule_path),
            "initial_plan": _plan_ref(initial_plan, initial_plan_path),
            "final_plan": _plan_ref(final_plan, final_plan_path),
            "rules_dir": rules_dir,
            "plans_dir": plans_dir or "",
        },
        "evolution": {
            "accepted_rounds": accepted_rounds,
            "stop_reason": stop_reason,
            "final_summary": dict(final_summary),
        },
        "artifacts": {
            "root_dir": str(artifacts_root),
            "round_dirs": round_dirs,
        },
    }


def _rule_ref(rule: EvolvingRule, path_like: str | Path) -> Dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "rule_version": rule.version,
        "rule_name": rule.name,
        "rule_path": str(path_like),
        "attack_label": normalize_attack_label((rule.metadata or {}).get("attack_label", "attack")),
        "raw_attack_label": str((rule.metadata or {}).get("raw_attack_label", (rule.metadata or {}).get("attack_label", "attack"))),
    }


def _plan_ref(
    plan: Optional[EvidencePlan],
    path_like: Optional[str | Path],
) -> Dict[str, Any]:
    if plan is None:
        return {}
    return {
        "plan_id": plan.plan_id,
        "rule_id": plan.rule_id,
        "rule_version": plan.rule_version,
        "plan_version": (plan.metadata or {}).get("plan_version"),
        "plan_path": str(path_like or ""),
        "source": (plan.metadata or {}).get("source", ""),
    }
