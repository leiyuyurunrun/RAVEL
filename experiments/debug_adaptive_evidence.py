from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.core.plan import (
    compile_rule_to_baseline_plan,
    constrain_first_pass_view_refs,
)
from evotx.core.rule import make_cold_start_rule
from evotx.core.schemas import EvidencePlan, JudgeStep
from evotx.runtime.adaptive_evidence import (
    AdaptiveEvidenceConfig,
    adapt_step_evidence_refs,
    collect_adaptive_candidate_views,
    extract_evidence_profile,
    resolve_adaptive_mode,
)
from evotx.runtime.packet_builder import build_or_load_packet
from evotx.utils.json_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain conservative adaptive evidence routing for one packet/plan."
    )
    parser.add_argument("--tx", required=True, help="Transaction hash.")
    parser.add_argument("--base-dir", default="data/cache", help="Packet cache base dir.")
    parser.add_argument("--label", default="", help="Attack-family label.")
    parser.add_argument(
        "--mode",
        choices=["off", "auto", "planned", "small_trace_direct", "medium_hybrid", "direct_trace"],
        default="auto",
        help="Adaptive evidence mode to explain.",
    )
    parser.add_argument("--plan", help="Optional EvidencePlan JSON path.")
    parser.add_argument("--packet", help="Optional packet JSON path.")
    parser.add_argument("--max-view-chars", type=int, default=80000)
    parser.add_argument("--max-context-chars", type=int, default=200000)
    parser.add_argument("--max-direct-trace-nodes", type=int, default=80)
    parser.add_argument("--max-direct-trace-chars", type=int, default=45000)
    parser.add_argument("--max-medium-trace-nodes", type=int, default=250)
    args = parser.parse_args()

    packet = _load_packet(args)
    plan = _load_plan(args)
    config = AdaptiveEvidenceConfig(
        max_direct_trace_nodes=max(0, int(args.max_direct_trace_nodes or 0)),
        max_direct_trace_chars=max(0, int(args.max_direct_trace_chars or 0)),
        max_medium_trace_nodes=max(0, int(args.max_medium_trace_nodes or 0)),
    )
    profile = extract_evidence_profile(packet, max_context_chars=args.max_context_chars)
    chosen_mode, reason = resolve_adaptive_mode(
        requested_mode=args.mode,
        profile=profile,
        max_context_chars=args.max_context_chars,
        config=config,
    )
    attack_label = args.label or str((plan.metadata or {}).get("attack_label") or "")
    decisions = [
        _decision_for_step(
            packet=packet,
            step=step,
            attack_label=attack_label,
            profile=profile,
            mode=chosen_mode,
            max_context_chars=args.max_context_chars,
            max_view_chars=args.max_view_chars,
            config=config,
        )
        for step in list(plan.judge_steps or [])
    ]
    required_views = collect_adaptive_candidate_views(
        adaptive_evidence=args.mode not in {"off", "planned"},
        adaptive_evidence_mode=args.mode,
        attack_label=attack_label,
        tx_hash=args.tx,
        base_dir=args.base_dir,
        config=config,
    )
    summary = {
        "tx_hash": args.tx,
        "attack_label": attack_label,
        "mode_requested": args.mode,
        "mode_chosen": chosen_mode,
        "mode_reason": reason,
        "profile": profile.to_dict(),
        "required_views_for_adaptive_build": required_views,
        "decisions": decisions,
        "warnings": _warnings(profile, decisions),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _load_packet(args) -> Dict[str, Any]:
    if args.packet:
        return read_json(args.packet, default={}) or {}
    return build_or_load_packet(
        args.tx,
        base_dir=args.base_dir,
        force_rebuild=False,
        packet_profile="full",
    )


def _load_plan(args) -> EvidencePlan:
    if args.plan:
        return EvidencePlan.from_dict(read_json(args.plan, default={}) or {})
    label = args.label or "generic_attack"
    return compile_rule_to_baseline_plan(make_cold_start_rule(label))


def _decision_for_step(
    *,
    packet: Dict[str, Any],
    step: JudgeStep,
    attack_label: str,
    profile,
    mode: str,
    max_context_chars: int,
    max_view_chars: int,
    config: AdaptiveEvidenceConfig,
) -> Dict[str, Any]:
    initial_refs, followups, deferred = constrain_first_pass_view_refs(
        step.default_evidence_refs or step.evidence_refs,
        step.allowed_followup_views,
    )
    decision = adapt_step_evidence_refs(
        packet=packet,
        step=step,
        attack_label=attack_label,
        initial_refs=initial_refs,
        allowed_followup_views=followups,
        profile=profile,
        mode=mode,
        max_context_chars=max_context_chars,
        max_view_chars=max_view_chars,
        config=config,
    )
    payload = decision.to_dict()
    payload.update({
        "judge_id": step.id,
        "condition_id": step.condition_id or step.id,
        "question": step.question,
        "first_pass_deferred_by_plan": deferred,
    })
    return payload


def _warnings(profile, decisions: List[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    view_chars = dict(profile.view_chars or {})
    if view_chars.get("trace_view", 0) > 0 and view_chars.get("trace_view", 0) > 45000:
        warnings.append("trace_view too large for small_trace_direct default cap")
    selected_first_pass = {
        view
        for decision in decisions
        for view in list(decision.get("initial_refs_after") or [])
    }
    if "state_change_view" in view_chars and "state_change_view" not in selected_first_pass:
        warnings.append("state_change_view excluded from first-pass because it is raw_heavy")
    if any("profit_loss_view" in list(d.get("removed_views") or []) for d in decisions):
        warnings.append("profit_loss_view removed because it is hint-only")
    removed_empty = sorted({
        view
        for decision in decisions
        for view in list(decision.get("removed_views") or [])
        if view in set(profile.empty_views or [])
    })
    if removed_empty:
        warnings.append(f"empty views removed from first-pass: {removed_empty}")
    return warnings


if __name__ == "__main__":
    main()
