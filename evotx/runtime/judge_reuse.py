from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Optional

from evotx.core.schemas import JudgeStep
from evotx.utils.json_utils import stable_json_dumps


JUDGE_STEP_FINGERPRINT_VERSION = "evotx.judge_step_fingerprint.v9"


def judge_step_fingerprint(
    step: JudgeStep | Dict[str, Any],
    *,
    judge_model: str = "",
    judge_provider: str = "",
    max_view_chars: int = 20000,
    max_context_chars: int = 60000,
    judge_max_tokens: int = 8192,
    judge_thinking: str = "disabled",
    source_followup_enabled: bool = False,
    followup_context_mode: str = "unified",
    judge_followup_mode: str = "plan",
    tx_hash: str = "",
    packet_identity_fingerprint: str = "",
    packet_fingerprint: str = "",
    selected_view_names: Optional[List[str]] = None,
    selected_views_hash: str = "",
    adaptive_evidence_enabled: bool = False,
    adaptive_evidence_mode: str = "off",
) -> str:
    data = _step_to_dict(step)
    payload = {
        "version": JUDGE_STEP_FINGERPRINT_VERSION,
        "judge_id": str(data.get("id") or ""),
        "condition_id": str(data.get("condition_id") or data.get("id") or ""),
        "question": str(data.get("question") or ""),
        "expected_answer": bool(data.get("expected_answer", True)),
        "evidence_refs": list(data.get("evidence_refs", []) or []),
        "default_evidence_refs": list(data.get("default_evidence_refs", []) or []),
        "allowed_followup_views": list(data.get("allowed_followup_views", []) or []),
        "allowed_tools": list(data.get("allowed_tools", []) or []),
        "max_followups": int(data.get("max_followups", 0) or 0),
        "depends_on": list(data.get("depends_on", []) or []),
        "consumes_state_keys": list(data.get("consumes_state_keys", []) or []),
        "produces_state_key": str(data.get("produces_state_key", "") or ""),
        "state_prompt_role": str(data.get("state_prompt_role", "") or ""),
        "state_output_schema": (
            dict(data.get("state_output_schema") or {})
            if isinstance(data.get("state_output_schema"), dict)
            else {}
        ),
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "max_view_chars": int(max_view_chars or 0),
        "max_context_chars": int(max_context_chars or 0),
        "judge_max_tokens": int(judge_max_tokens or 0),
        "judge_thinking": str(judge_thinking or ""),
        "source_followup_enabled": bool(source_followup_enabled),
        "followup_context_mode": str(followup_context_mode or "unified"),
        "judge_followup_mode": str(judge_followup_mode or "plan"),
        "tx_hash": str(tx_hash or "").lower().strip(),
        "packet_identity_fingerprint": str(
            packet_identity_fingerprint or packet_fingerprint or ""
        ),
        "selected_view_names": list(selected_view_names or []),
        "selected_views_hash": str(selected_views_hash or ""),
        "adaptive_evidence_enabled": bool(adaptive_evidence_enabled),
        "adaptive_evidence_mode": str(adaptive_evidence_mode or "off"),
    }
    digest = hashlib.sha256(
        stable_json_dumps(payload).encode("utf-8")
    ).hexdigest()
    return digest


def build_judge_reuse_cache(
    previous_result: Optional[Dict[str, Any]],
    *,
    judge_model: str = "",
    judge_provider: str = "",
    max_view_chars: int = 20000,
    max_context_chars: int = 60000,
    judge_max_tokens: int = 8192,
    judge_thinking: str = "disabled",
    source_followup_enabled: bool = False,
    followup_context_mode: str = "unified",
    judge_followup_mode: str = "plan",
    packet_identity_fingerprint: str = "",
    packet_fingerprint: str = "",
    adaptive_evidence_enabled: bool = False,
    adaptive_evidence_mode: str = "off",
) -> Dict[str, Dict[str, Any]]:
    if not previous_result:
        return {}

    requested_followup_mode = str(judge_followup_mode or "plan").strip().lower()
    if _recorded_judge_followup_mode(previous_result) != requested_followup_mode:
        return {}

    inference = previous_result.get("inference", {}) or {}
    previous_plan = inference.get("plan", {}) or previous_result.get("plan", {}) or {}
    trace = inference.get("trace", {}) or previous_result.get("trace", {}) or {}
    execution = trace.get("execution", {}) if isinstance(trace, dict) else {}
    trace_metadata = trace.get("metadata", {}) if isinstance(trace, dict) else {}
    packet_context = _packet_context(
        previous_result,
        trace_metadata,
        packet_identity_fingerprint or packet_fingerprint,
    )

    plan_steps = {
        str(step.get("id")): step
        for step in list(previous_plan.get("judge_steps", []) or [])
        if isinstance(step, dict) and step.get("id")
    }
    raw_judge_calls = []
    raw_step_traces = []
    if isinstance(trace, dict):
        raw_judge_calls.extend(list(trace.get("judge_calls", []) or []))
        raw_step_traces.extend(list(trace.get("judge_step_traces", []) or []))
    if isinstance(execution, dict):
        raw_judge_calls.extend(list(execution.get("judge_calls", []) or []))
        raw_step_traces.extend(list(execution.get("judge_step_traces", []) or []))
    if isinstance(trace_metadata, dict):
        raw_step_traces.extend(list(trace_metadata.get("judge_step_traces", []) or []))

    judge_calls = {
        str(item.get("judge_id") or item.get("id")): item
        for item in raw_judge_calls
        if isinstance(item, dict) and (item.get("judge_id") or item.get("id"))
    }
    step_traces = {
        str(item.get("judge_id") or item.get("id")): item
        for item in raw_step_traces
        if isinstance(item, dict) and (item.get("judge_id") or item.get("id"))
    }

    cache: Dict[str, Dict[str, Any]] = {}
    for judge_id, step in plan_steps.items():
        judge_call = judge_calls.get(judge_id)
        if not judge_call:
            continue
        step_trace = step_traces.get(judge_id, {})
        selected_view_names = _selected_view_names(step_trace, judge_call)
        selected_views_hash = str(
            (step_trace.get("selected_views_hash") if isinstance(step_trace, dict) else "")
            or judge_call.get("selected_views_hash")
            or ""
        )
        if not packet_context["packet_identity_fingerprint"] or not selected_views_hash:
            continue
        fingerprint = judge_step_fingerprint(
            step,
            judge_model=judge_model,
            judge_provider=judge_provider,
            max_view_chars=max_view_chars,
            max_context_chars=max_context_chars,
            judge_max_tokens=judge_max_tokens,
            judge_thinking=judge_thinking,
            source_followup_enabled=source_followup_enabled,
            followup_context_mode=followup_context_mode,
            judge_followup_mode=requested_followup_mode,
            tx_hash=_tx_hash(previous_result),
            packet_identity_fingerprint=packet_context["packet_identity_fingerprint"],
            selected_view_names=selected_view_names,
            selected_views_hash=selected_views_hash,
            adaptive_evidence_enabled=adaptive_evidence_enabled,
            adaptive_evidence_mode=adaptive_evidence_mode if adaptive_evidence_enabled else "off",
        )
        cache[fingerprint] = {
            "fingerprint": fingerprint,
            "judge_id": judge_id,
            "judge_result": copy.deepcopy(judge_call),
            "step_trace": copy.deepcopy(step_trace),
            "source": {
                "tx_hash": _tx_hash(previous_result),
                "packet_identity_fingerprint": packet_context["packet_identity_fingerprint"],
                "packet_fingerprint": packet_context["packet_identity_fingerprint"],
                "packet_source": packet_context["packet_source"],
                "selected_view_names": list(selected_view_names),
                "selected_views_hash": selected_views_hash,
                "adaptive_evidence_enabled": bool(adaptive_evidence_enabled),
                "adaptive_evidence_mode": adaptive_evidence_mode
                if adaptive_evidence_enabled
                else "off",
                "followup_context_mode": str(
                    followup_context_mode or "unified"
                ),
                "judge_followup_mode": requested_followup_mode,
                "judge_max_tokens": int(judge_max_tokens or 0),
                "judge_thinking": str(judge_thinking or ""),
                "rule_id": (previous_result.get("detector_context", {}) or {}).get("rule_id"),
                "rule_version": (previous_result.get("detector_context", {}) or {}).get("rule_version"),
                "plan_id": previous_plan.get("plan_id"),
                "plan_version": (previous_plan.get("metadata", {}) or {}).get("plan_version"),
            },
        }
    return cache


def _recorded_judge_followup_mode(result: Dict[str, Any]) -> str:
    inference = result.get("inference", {}) if isinstance(result, dict) else {}
    trace = inference.get("trace", {}) if isinstance(inference, dict) else {}
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else {}
    policy = metadata.get("judge_followup_policy", {}) if isinstance(metadata, dict) else {}
    if not policy and isinstance(result.get("runtime_trace"), dict):
        policy = result["runtime_trace"].get("judge_followup_policy", {}) or {}
    return str((policy or {}).get("mode") or "plan").strip().lower()


def _packet_context(
    result: Dict[str, Any],
    trace_metadata: Dict[str, Any],
    explicit_packet_fingerprint: str = "",
) -> Dict[str, str]:
    runtime_trace = result.get("runtime_trace", {}) if isinstance(result, dict) else {}
    if not isinstance(runtime_trace, dict):
        runtime_trace = {}
    packet_identity_fingerprint = str(
        explicit_packet_fingerprint
        or runtime_trace.get("packet_identity_fingerprint")
        or trace_metadata.get("packet_identity_fingerprint")
        or runtime_trace.get("packet_fingerprint")
        or trace_metadata.get("packet_fingerprint")
        or ""
    )
    packet_source = str(
        runtime_trace.get("packet_source")
        or trace_metadata.get("packet_source")
        or ""
    )
    return {
        "packet_identity_fingerprint": packet_identity_fingerprint,
        "packet_fingerprint": packet_identity_fingerprint,
        "packet_source": packet_source,
    }


def _step_to_dict(step: JudgeStep | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(step, JudgeStep):
        return step.to_dict()
    return dict(step or {})


def _selected_view_names(
    step_trace: Dict[str, Any],
    judge_call: Dict[str, Any],
) -> List[str]:
    values = []
    if isinstance(step_trace, dict):
        values = (
            step_trace.get("selected_view_names")
            or step_trace.get("initial_views")
            or []
        )
    if not values:
        values = judge_call.get("selected_view_names") or judge_call.get("evidence_refs") or []
    out: List[str] = []
    seen = set()
    for value in list(values or []):
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _tx_hash(result: Dict[str, Any]) -> str:
    if "transaction" in result:
        return str((result.get("transaction", {}) or {}).get("tx_hash", ""))
    return str(result.get("tx_hash", ""))
