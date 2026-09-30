"""Minimal CLI for packet-based EvoTx runtime."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.adapters.llm_adapter import OpenAICompatibleLLM
from evotx.core.schemas import (
    EvolvingRule,
    EvidencePlan,
    JudgeStep,
    RuleCondition,
)
from evotx.runtime.judge import JudgeModel
from evotx.runtime.packet_runtime import PacketRuntime
from evotx.runtime.source_tools import SourceToolRegistry
from evotx.utils.json_utils import write_json


def build_sample_rule() -> EvolvingRule:
    return EvolvingRule(
        rule_id="general_transaction_anomaly",
        version=1,
        name="General Transaction Anomaly Evidence Rule",
        description=(
            "Detect abnormal transaction behavior by requiring concrete evidence "
            "of unauthorized or repeated value-releasing operations, and excluding "
            "benign patterns such as normal withdrawals or governance operations."
        ),
        conditions=[
            RuleCondition(
                id="C1",
                description=(
                    "The transaction shows repeated or unauthorized collection, "
                    "withdrawal, mint, reward, or asset transfer behavior."
                ),
            ),
            RuleCondition(
                id="C2",
                description=(
                    "The same sender or controlled contract repeatedly triggers a "
                    "value-releasing operation in the same transaction."
                ),
            ),
        ],
        exclusion_conditions=[
            RuleCondition(
                id="E1",
                description=(
                    "The behavior is better explained by a benign pattern such as "
                    "normal user withdrawal, governance operation, or expected "
                    "protocol callback."
                ),
            )
        ],
        decision_policy=(
            "Emit attack when C1 and C2 are satisfied and no benign exclusion "
            "(E1) explains the behavior. If evidence is insufficient, keep "
            "uncertain."
        ),
        metadata={
            "source": "sample",
            "attack_label": "general_transaction_anomaly",
        },
    )


def build_sample_plan(rule: EvolvingRule) -> EvidencePlan:
    return EvidencePlan(
        plan_id=EvidencePlan.new_id(),
        rule_id=rule.rule_id,
        rule_version=rule.version,
        focus_steps=[],
        judge_steps=[
            JudgeStep(
                id="j1",
                question=(
                    "Does the provided transaction evidence show repeated or "
                    "unauthorized collection, withdrawal, mint, reward, or asset "
                    "transfer behavior?"
                ),
                evidence_refs=[
                    "tx_card",
                    "trace_view",
                    "event_view",
                    "state_change_view",
                    "transfer_event_view",
                    "external_fundflow_view",
                    "profit_loss_view",
                ],
                expected_answer=True,
                condition_id="C1",
            ),
            JudgeStep(
                id="j2",
                question=(
                    "Does the provided transaction evidence show that the same "
                    "sender or controlled contract repeatedly triggers a "
                    "value-releasing operation in the same transaction?"
                ),
                evidence_refs=[
                    "trace_view",
                    "event_view",
                    "transfer_event_view",
                    "external_fundflow_view",
                ],
                expected_answer=True,
                condition_id="C2",
            ),
            JudgeStep(
                id="j3",
                question=(
                    "Does the provided evidence suggest a benign explanation such "
                    "as normal user withdrawal, governance operation, or expected "
                    "protocol callback?"
                ),
                evidence_refs=[
                    "tx_card",
                    "address_labels",
                    "trace_view",
                    "event_view",
                ],
                expected_answer=True,
                condition_id="E1",
            ),
        ],
        emit_logic="j1 and j2 and not j3",
        plan_note="Sample plan for packet-based runtime testing.",
        metadata={"generator": "sample"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run packet-based EvoTx runtime on a single transaction."
    )
    parser.add_argument("--tx", required=True, help="Transaction hash")
    parser.add_argument(
        "--base-dir",
        default="data/cache",
        help="Cache base directory (default: data/cache)",
    )
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument(
        "--llm-provider",
        help="Optional provider override: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument(
        "--results-dir",
        default="data/results",
        help="Output directory (default: data/results)",
    )
    parser.add_argument(
        "--force-rebuild-packet",
        action="store_true",
        help="Force rebuild evidence packet",
    )
    parser.add_argument(
        "--enable-source-tools",
        action="store_true",
        help="Allow each judge step to make one read_function_chunk source-code follow-up.",
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
    )
    parser.add_argument(
        "--aggregator-thinking",
        choices=["default", "disabled", "adaptive"],
        default="disabled",
    )
    parser.add_argument(
        "--followup-context-mode",
        choices=["legacy", "unified"],
        default="unified",
        help="Judge follow-up context policy.",
    )
    parser.add_argument(
        "--enable-dynamic-aggregation",
        dest="dynamic_aggregation",
        action="store_true",
        default=True,
        help="Enable dynamic aggregation after local Judge execution.",
    )
    parser.add_argument(
        "--disable-dynamic-aggregation",
        dest="dynamic_aggregation",
        action="store_false",
        help="Disable dynamic aggregation and use only emit_logic.",
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
        "--disable-rate-limit-serial-fallback",
        dest="rate_limit_serial_fallback",
        action="store_false",
        default=True,
        help="Disable automatic serial retry and concurrency downgrade after judge HTTP 429.",
    )
    parser.add_argument("--rate-limit-retry-attempts", type=int, default=3)
    parser.add_argument("--rate-limit-retry-delay-seconds", type=float, default=2.0)
    args = parser.parse_args()

    llm = (
        OpenAICompatibleLLM(
            model=args.model,
            provider=args.llm_provider,
            max_tokens=max(1, int(args.judge_max_tokens or 8192)),
            minimax_thinking=args.judge_thinking,
        )
        if args.model
        else None
    )
    source_registry = (
        SourceToolRegistry(
            cache_dir=args.source_cache_dir,
            force_refresh=args.force_refresh_source,
        )
        if args.enable_source_tools
        else None
    )
    judge_model = JudgeModel(
        llm=llm,
        max_view_chars=args.max_view_chars,
        max_context_chars=args.max_context_chars,
        source_tool_registry=source_registry,
        enable_source_followup=args.enable_source_tools,
        followup_context_mode=args.followup_context_mode,
    )
    aggregator_llm = (
        OpenAICompatibleLLM(
            model=args.model,
            provider=args.llm_provider,
            max_tokens=max(1, int(args.aggregator_max_tokens or 8192)),
            minimax_thinking=args.aggregator_thinking,
        )
        if args.model and args.dynamic_aggregation
        else None
    )
    runtime = PacketRuntime(
        judge_model=judge_model,
        aggregator_llm=aggregator_llm,
        base_dir=args.base_dir,
    )

    rule = build_sample_rule()
    plan = build_sample_plan(rule)

    print(
        f"[RunPacketRuntime] tx={args.tx} base_dir={args.base_dir} "
        f"model={args.model or 'disabled'} provider={args.llm_provider or '(auto)'} "
        f"source_tools={args.enable_source_tools} "
        f"max_view_chars={args.max_view_chars} max_context_chars={args.max_context_chars} "
        f"judge_max_tokens={int(getattr(llm, 'max_tokens', 0) or args.judge_max_tokens)} "
        f"aggregator_max_tokens={int(getattr(aggregator_llm, 'max_tokens', 0) or args.aggregator_max_tokens)} "
        f"judge_thinking={args.judge_thinking} "
        f"aggregator_thinking={args.aggregator_thinking} "
        f"followup_context_mode={args.followup_context_mode} "
        f"dynamic_aggregation={args.dynamic_aggregation} "
        f"disable_stateful_bindings={args.disable_stateful_bindings} "
        f"rate_limit_serial_fallback={args.rate_limit_serial_fallback}"
    )

    result = runtime.execute(
        tx_hash=args.tx,
        rule=rule,
        plan=plan,
        force_rebuild_packet=args.force_rebuild_packet,
        dynamic_aggregation=args.dynamic_aggregation,
        disable_stateful_bindings=args.disable_stateful_bindings,
        rate_limit_serial_fallback=args.rate_limit_serial_fallback,
        rate_limit_retry_attempts=args.rate_limit_retry_attempts,
        rate_limit_retry_delay_seconds=args.rate_limit_retry_delay_seconds,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = (
        results_dir
        / f"{args.tx.lower()}__{rule.rule_id}__v{rule.version}_packet.json"
    )
    write_json(result_path, result)

    print(f"\nVerdict: {result['verdict']}")
    print(f"Judge calls: {result['runtime_trace']['judge_call_count']}")
    print(f"Supporting evidence IDs: {len(result['supporting_evidence_ids'])}")
    print(f"Result saved: {result_path}")


if __name__ == "__main__":
    main()
