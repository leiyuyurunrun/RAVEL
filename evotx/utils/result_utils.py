from __future__ import annotations

from typing import Any, Dict, Optional

from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import EvolvingRule
from evotx.utils.trace_utils import build_embedded_trace_view


RESULT_SCHEMA_VERSION = "evotx.result.v1"


def build_result_record(
    tx_hash: str,
    chain: str,
    rule: EvolvingRule | Dict[str, Any],
    rule_source: Optional[str],
    runtime_result: Dict[str, Any],
    plan_source: Optional[str] = None,
    ground_truth: Optional[str] = None,
    raw_ground_truth: Optional[str] = None,
    label_rationale: str = "",
    report: str = "",
    case_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rule_dict = _rule_to_dict(rule)
    raw_attack_label = (
        (rule_dict.get("metadata", {}) or {}).get("attack_label")
        or "attack"
    )
    attack_label = normalize_attack_label(raw_attack_label)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "transaction": {
            "tx_hash": tx_hash,
            "chain": chain,
        },
        "detector_context": {
            "attack_label": attack_label,
            "rule_id": str(rule_dict.get("rule_id", "")),
            "rule_version": int(rule_dict.get("version", 1)),
            "rule_name": str(rule_dict.get("name", "")),
            "rule_source": rule_source
            or f"in_memory:{rule_dict.get('rule_id', 'rule')}__v{rule_dict.get('version', 1)}",
            "plan_id": str((runtime_result.get("plan", {}) or {}).get("plan_id", "")),
            "plan_source": plan_source
            or str((runtime_result.get("plan", {}) or {}).get("metadata", {}).get("plan_source", "")),
        },
        "inference": {
            "rule": rule_dict,
            "plan": runtime_result.get("plan", {}),
            "evidence": runtime_result.get("evidence", {}),
            **(
                {"packet_snapshot": runtime_result.get("packet_snapshot")}
                if runtime_result.get("packet_snapshot")
                else {}
            ),
            "trace": build_embedded_trace_view(runtime_result.get("trace", {})),
            "finding": runtime_result.get("finding", {}),
        },
        "evaluation": {
            "ground_truth": ground_truth,
            "raw_ground_truth": raw_ground_truth,
            "label_rationale": label_rationale,
            "report": report,
            "case_metadata": case_metadata or {},
        },
    }


def get_tx_hash(result: Dict[str, Any]) -> str:
    if "tx_hash" in result:
        return str(result.get("tx_hash", "unknown"))
    if "transaction" in result:
        return str((result.get("transaction", {}) or {}).get("tx_hash", "unknown"))
    return str(result.get("tx_hash", "unknown"))


def get_chain(result: Dict[str, Any]) -> str:
    if "chain" in result:
        return str(result.get("chain", "eth"))
    if "transaction" in result:
        return str((result.get("transaction", {}) or {}).get("chain", "eth"))
    return str(result.get("chain", "eth"))


def get_rule(result: Dict[str, Any]) -> Dict[str, Any]:
    if "inference" in result and "rule" in (result.get("inference", {}) or {}):
        return dict((result.get("inference", {}) or {}).get("rule", {}) or {})
    if "rule_digest" in result:
        return dict(result.get("rule_digest", {}) or {})
    if "artifacts" in result:
        return dict((result.get("artifacts", {}) or {}).get("rule", {}) or {})
    return dict(result.get("rule", {}) or {})


def get_plan(result: Dict[str, Any]) -> Dict[str, Any]:
    if "inference" in result and "plan" in (result.get("inference", {}) or {}):
        return dict((result.get("inference", {}) or {}).get("plan", {}) or {})
    if "plan_digest" in result:
        return dict(result.get("plan_digest", {}) or {})
    if "artifacts" in result:
        return dict((result.get("artifacts", {}) or {}).get("plan", {}) or {})
    return dict(result.get("plan", {}) or {})


def get_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    if "inference" in result and "trace" in (result.get("inference", {}) or {}):
        return dict((result.get("inference", {}) or {}).get("trace", {}) or {})
    if "artifacts" in result:
        return dict((result.get("artifacts", {}) or {}).get("trace", {}) or {})
    return dict(result.get("trace", {}) or {})


def get_finding(result: Dict[str, Any]) -> Dict[str, Any]:
    if "inference" in result and "finding" in (result.get("inference", {}) or {}):
        return dict((result.get("inference", {}) or {}).get("finding", {}) or {})
    if "finding_summary" in result:
        finding = dict(result.get("finding_summary", {}) or {})
        finding.setdefault("verdict", result.get("predicted_verdict"))
        finding.setdefault("confidence", result.get("confidence", "medium"))
        return finding
    return dict(result.get("finding", {}) or {})


def get_evidence(result: Dict[str, Any]) -> Dict[str, Any]:
    if "inference" in result and "evidence" in (result.get("inference", {}) or {}):
        return dict((result.get("inference", {}) or {}).get("evidence", {}) or {})
    if "artifacts" in result:
        return dict((result.get("artifacts", {}) or {}).get("evidence", {}) or {})
    return dict(result.get("evidence", {}) or {})


def get_verdict(result: Dict[str, Any]) -> Optional[str]:
    return get_finding(result).get("verdict")


def get_ground_truth(result: Dict[str, Any]) -> Optional[str]:
    if "ground_truth" in result:
        return result.get("ground_truth")
    if "evaluation" in result:
        return (result.get("evaluation", {}) or {}).get("ground_truth")
    return result.get("ground_truth")


def get_raw_ground_truth(result: Dict[str, Any]) -> Optional[str]:
    if "raw_ground_truth" in result:
        return result.get("raw_ground_truth")
    if "evaluation" in result:
        return (result.get("evaluation", {}) or {}).get("raw_ground_truth")
    return result.get("raw_ground_truth")


def get_label_rationale(result: Dict[str, Any]) -> str:
    if "evaluation" in result:
        return str((result.get("evaluation", {}) or {}).get("label_rationale", ""))
    return str(result.get("label_rationale", ""))


def get_report(result: Dict[str, Any]) -> str:
    if "evaluation" in result:
        return str((result.get("evaluation", {}) or {}).get("report", ""))
    return str(result.get("report", ""))


def get_case_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    if "evaluation" in result:
        return dict((result.get("evaluation", {}) or {}).get("case_metadata", {}) or {})
    return dict(result.get("case_metadata", {}) or {})


def get_attack_label(result: Dict[str, Any]) -> str:
    if "attack_label" in result:
        return normalize_attack_label(result.get("attack_label", "attack"))
    if "detector_context" in result:
        return normalize_attack_label((result.get("detector_context", {}) or {}).get("attack_label", "attack"))
    rule = get_rule(result)
    return normalize_attack_label((rule.get("metadata", {}) or {}).get("attack_label", result.get("attack_label", "attack")))


def get_rule_source(result: Dict[str, Any]) -> Optional[str]:
    if "rule_source" in result:
        return result.get("rule_source")
    if "detector_context" in result:
        return (result.get("detector_context", {}) or {}).get("rule_source")
    return result.get("rule_source")


def as_reviewer_case(result: Dict[str, Any]) -> Dict[str, Any]:
    from evotx.utils.result_slimmer import build_reviewer_case

    return build_reviewer_case(result)


def _rule_to_dict(rule: EvolvingRule | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(rule, EvolvingRule):
        return rule.to_dict()
    return dict(rule or {})
