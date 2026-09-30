from __future__ import annotations

from typing import Any, Dict, Optional

from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import EvolvingRule


TRACE_SCHEMA_VERSION = "evotx.trace.v1"
EMBEDDED_TRACE_SCHEMA_VERSION = "evotx.trace.embedded.v1"


def build_embedded_trace_view(raw_trace: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(raw_trace, dict):
        raw_trace = {}

    if raw_trace.get("schema_version") == EMBEDDED_TRACE_SCHEMA_VERSION:
        return dict(raw_trace)

    if raw_trace.get("schema_version") == TRACE_SCHEMA_VERSION:
        trace_body = raw_trace.get("trace", {}) or {}
        return dict(trace_body) if isinstance(trace_body, dict) else {}

    return {
        "schema_version": EMBEDDED_TRACE_SCHEMA_VERSION,
        "plan_id": str(raw_trace.get("plan_id", "")),
        "execution": {
            "tool_calls": list(raw_trace.get("tool_calls", [])),
            "judge_calls": list(raw_trace.get("judge_calls", [])),
            "judge_step_traces": list(raw_trace.get("judge_step_traces", [])),
            "emissions": list(raw_trace.get("emissions", [])),
            "errors": list(raw_trace.get("errors", [])),
        },
        "timing": {
            "started_at": raw_trace.get("started_at"),
            "ended_at": raw_trace.get("ended_at"),
        },
        "metadata": dict(raw_trace.get("metadata", {})),
    }


def build_trace_record(
    tx_hash: str,
    chain: str,
    rule: EvolvingRule | Dict[str, Any],
    rule_source: Optional[str],
    plan: Dict[str, Any],
    raw_trace: Dict[str, Any],
    finding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rule_dict = _rule_to_dict(rule)
    raw_attack_label = (
        (rule_dict.get("metadata", {}) or {}).get("attack_label")
        or "attack"
    )
    attack_label = normalize_attack_label(raw_attack_label)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "transaction": {
            "tx_hash": tx_hash,
            "chain": chain,
        },
        "detector_context": {
            "attack_label": attack_label,
            "raw_attack_label": raw_attack_label,
            "rule_id": str(rule_dict.get("rule_id", "")),
            "rule_version": int(rule_dict.get("version", 1)),
            "rule_name": str(rule_dict.get("name", "")),
            "rule_source": rule_source
            or f"in_memory:{rule_dict.get('rule_id', 'rule')}__v{rule_dict.get('version', 1)}",
        },
        "inference": {
            "plan": dict(plan or {}),
            "finding": dict(finding or {}),
        },
        "trace": build_embedded_trace_view(raw_trace),
    }


def build_trace_record_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("schema_version") == TRACE_SCHEMA_VERSION:
        return dict(result)

    transaction = result.get("transaction", {}) or {}
    detector_context = result.get("detector_context", {}) or {}
    inference = result.get("inference", {}) or {}
    trace_view = build_embedded_trace_view(inference.get("trace", result.get("trace", {})))
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "transaction": {
            "tx_hash": str(transaction.get("tx_hash", result.get("tx_hash", "unknown"))),
            "chain": str(transaction.get("chain", result.get("chain", "eth"))),
        },
        "detector_context": {
            "attack_label": str(detector_context.get("attack_label", result.get("attack_label", "attack"))),
            "rule_id": str(detector_context.get("rule_id", (result.get("rule", {}) or {}).get("rule_id", ""))),
            "rule_version": int(detector_context.get("rule_version", (result.get("rule", {}) or {}).get("version", 1))),
            "rule_name": str(detector_context.get("rule_name", (result.get("rule", {}) or {}).get("name", ""))),
            "rule_source": detector_context.get("rule_source", result.get("rule_source")),
        },
        "inference": {
            "plan": dict(inference.get("plan", result.get("plan", {})) or {}),
            "finding": dict(inference.get("finding", result.get("finding", {})) or {}),
        },
        "trace": trace_view,
    }


def get_trace_tx_hash(trace: Dict[str, Any]) -> str:
    if "transaction" in trace:
        return str((trace.get("transaction", {}) or {}).get("tx_hash", "unknown"))
    return str(trace.get("tx_hash", "unknown"))


def get_trace_rule_id(trace: Dict[str, Any]) -> str:
    if "detector_context" in trace:
        return str((trace.get("detector_context", {}) or {}).get("rule_id", "rule"))
    return str(trace.get("rule_id", "rule"))


def get_trace_plan_id(trace: Dict[str, Any]) -> str:
    if trace.get("schema_version") == TRACE_SCHEMA_VERSION:
        return str(((trace.get("trace", {}) or {}).get("plan_id", "plan")))
    if trace.get("schema_version") == EMBEDDED_TRACE_SCHEMA_VERSION:
        return str(trace.get("plan_id", "plan"))
    return str(trace.get("plan_id", "plan"))


def _rule_to_dict(rule: EvolvingRule | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(rule, EvolvingRule):
        return rule.to_dict()
    return dict(rule or {})
