from __future__ import annotations

import hashlib
from typing import Any

from evotx.core.schemas import EvidencePlan, EvolvingRule
from evotx.utils.json_utils import stable_json_dumps


def object_fingerprint(payload: Any, *, prefix: str = "") -> str:
    """Return a short stable fingerprint for JSON-serializable payloads."""
    text = stable_json_dumps(payload)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def rule_fingerprint(rule: EvolvingRule | dict[str, Any]) -> str:
    if isinstance(rule, EvolvingRule):
        payload = rule.to_dict()
    else:
        payload = dict(rule or {})
    payload.pop("history", None)
    payload.pop("created_at", None)
    payload.pop("updated_at", None)
    return object_fingerprint(payload, prefix="rule:")


def plan_fingerprint(plan: EvidencePlan | dict[str, Any]) -> str:
    if isinstance(plan, EvidencePlan):
        payload = plan.to_dict()
    else:
        payload = dict(plan or {})
    payload.pop("created_at", None)
    return object_fingerprint(payload, prefix="plan:")


def semantic_plan_payload(plan: EvidencePlan | dict[str, Any]) -> dict[str, Any]:
    """Return only execution-semantic plan fields.

    Metadata, notes, timestamps, and version stamps are intentionally excluded
    so metadata-only edits do not create a meaningful plan candidate.
    """
    if isinstance(plan, EvidencePlan):
        payload = plan.to_dict()
    else:
        payload = dict(plan or {})
    semantic_steps = []
    for item in list(payload.get("judge_steps", []) or []):
        if not isinstance(item, dict):
            continue
        semantic_steps.append({
            "id": str(item.get("id") or ""),
            "condition_id": str(item.get("condition_id") or ""),
            "question": str(item.get("question") or ""),
            "evidence_refs": _semantic_string_sequence(item.get("evidence_refs")),
            "default_evidence_refs": _semantic_string_sequence(
                item.get("default_evidence_refs")
            ),
            "allowed_followup_views": _semantic_string_sequence(
                item.get("allowed_followup_views")
            ),
            "allowed_tools": _semantic_string_set(item.get("allowed_tools")),
            "max_followups": int(item.get("max_followups") or 0),
            "expected_answer": bool(item.get("expected_answer", True)),
            "depends_on": _semantic_string_set(item.get("depends_on")),
            "consumes_state_keys": _semantic_string_set(
                item.get("consumes_state_keys")
            ),
            "produces_state_key": str(item.get("produces_state_key") or ""),
            "state_prompt_role": str(item.get("state_prompt_role") or ""),
            "state_output_schema": dict(item.get("state_output_schema") or {}),
            "view_budget": dict(item.get("view_budget") or {}),
        })
    return {
        "rule_id": str(payload.get("rule_id") or ""),
        "rule_version": int(payload.get("rule_version") or 0),
        "focus_steps": list(payload.get("focus_steps") or []),
        "judge_steps": semantic_steps,
        "emit_logic": str(payload.get("emit_logic") or ""),
    }


def semantic_plan_fingerprint(plan: EvidencePlan | dict[str, Any]) -> str:
    return object_fingerprint(semantic_plan_payload(plan), prefix="plansem:")


def _semantic_string_set(value: Any) -> list[str]:
    """Normalize genuinely set-like plan fields."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return sorted({str(item) for item in values if str(item)})


def _semantic_string_sequence(value: Any) -> list[str]:
    """Normalize an execution-ordered route while preserving first occurrence."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return list(dict.fromkeys(str(item) for item in values if str(item)))
