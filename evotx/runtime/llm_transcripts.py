from __future__ import annotations

import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from evotx.utils.json_utils import write_json


class LLMTranscriptRecorder:
    """Write opt-in LLM judge transcripts outside normal result payloads."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()

    def save_judge_transcript(
        self,
        payload: Dict[str, Any],
        *,
        phase: str = "",
        tx_hash: str = "",
        judge_id: str = "",
        round_id: int | str = 0,
    ) -> Path:
        with self._lock:
            self._counter += 1
            tx_part = _safe_segment(tx_hash or "unknown_tx")
            judge_part = _safe_segment(judge_id or "judge")
            round_part = _safe_segment(f"round{round_id}")
            target_dir = self.root_dir
            for part in _safe_phase_parts(phase):
                target_dir = target_dir / part
            target_dir = target_dir / tx_part
            path = target_dir / f"{self._counter:05d}__{judge_part}__{round_part}.json"
            while path.exists():
                self._counter += 1
                path = target_dir / f"{self._counter:05d}__{judge_part}__{round_part}.json"
            write_json(path, payload)
            return path


def _safe_phase_parts(phase: str) -> Iterable[str]:
    parts = str(phase or "run").replace("\\", "/").split("/")
    return [_safe_segment(part) for part in parts if part.strip()]


def _safe_segment(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text[:160] or "unknown"


_AUDIT_WRITE_LOCK = threading.Lock()


def write_llm_audit_transcript(
    *,
    output_dir: str | Path | None,
    stage: str,
    name: str,
    model: str | None,
    prompt: str,
    raw_completion: Any = "",
    parsed_response: Any = None,
    usage: Any = None,
    finish_reason: str = "",
    metadata: Dict[str, Any] | None = None,
    truncation: Dict[str, Any] | None = None,
) -> str | None:
    """Write an opt-in LLM audit transcript without affecting the caller."""
    if output_dir is None:
        return None
    try:
        stage_part = _safe_segment(stage or "llm")
        name_part = _safe_segment(name or "call")
        root = Path(output_dir) / stage_part
        created_at = datetime.now(timezone.utc)
        raw_text = llm_response_text(raw_completion)
        prompt_text = str(prompt or "")
        payload = {
            "schema_version": "evotx.llm_audit.v1",
            "stage": stage_part,
            "name": str(name or ""),
            "created_at": created_at.isoformat(),
            "model": str(model or ""),
            "prompt_chars": len(prompt_text),
            "prompt_est_tokens": estimate_prompt_tokens(prompt_text),
            "raw_completion_chars": len(raw_text),
            "parsed_response_present": parsed_response is not None,
            "truncation": truncation or build_truncation_audit(
                input_name="prompt",
                before=prompt_text,
                after=prompt_text,
            ),
            "metadata": dict(metadata or {}),
            "prompt": prompt_text,
            "raw_completion": raw_text,
            "parsed_response": parsed_response,
            "usage": usage if usage is not None else {},
            "finish_reason": str(finish_reason or ""),
        }
        with _AUDIT_WRITE_LOCK:
            root.mkdir(parents=True, exist_ok=True)
            stamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
            path = root / f"{stamp}__{name_part}.json"
            counter = 1
            while path.exists():
                counter += 1
                path = root / f"{stamp}__{counter:02d}__{name_part}.json"
            write_json(path, payload)
        return str(path)
    except Exception as exc:
        print(f"[LLM-Audit] warning: failed to write {stage} transcript: {exc!r}")
        return None


def build_truncation_audit(
    *,
    input_name: str,
    before: Any,
    after: Any,
    max_chars: int | None = None,
    reason: str = "",
    removed_fields: list[str] | None = None,
    removed_item_counts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    before_chars = _serialized_chars(before)
    after_chars = _serialized_chars(after)
    removed_fields = list(removed_fields or [])
    removed_item_counts = dict(removed_item_counts or {})
    return {
        "applied": bool(
            after_chars < before_chars
            or removed_fields
            or any(int(v or 0) for v in removed_item_counts.values())
        ),
        "input_name": str(input_name or ""),
        "reason": str(reason or ""),
        "max_chars": int(max_chars or 0),
        "input_chars_before": before_chars,
        "input_chars_after": after_chars,
        "removed_fields": removed_fields,
        "removed_item_counts": removed_item_counts,
    }


def estimate_prompt_tokens(text: str) -> int:
    chars = len(str(text or ""))
    return 0 if chars <= 0 else max(1, int(math.ceil(chars / 4)))


def llm_model_name(llm: Any) -> str:
    for attr in ("model", "model_name", "deployment", "engine"):
        value = getattr(llm, attr, "")
        if value:
            return str(value)
    return ""


def llm_usage(llm: Any, response: Any = None) -> Any:
    for source in (response, llm):
        if source is None:
            continue
        for attr in ("usage", "last_usage"):
            value = getattr(source, attr, None)
            if value is not None:
                return value
        if isinstance(source, dict) and "usage" in source:
            return source.get("usage")
    return {}


def llm_finish_reason(llm: Any, response: Any = None) -> str:
    for source in (response, llm):
        if source is None:
            continue
        for attr in ("finish_reason", "last_finish_reason"):
            value = getattr(source, attr, None)
            if value:
                return str(value)
        if isinstance(source, dict) and source.get("finish_reason"):
            return str(source.get("finish_reason"))
    return ""


def llm_response_text(response: Any) -> str:
    if response is None:
        return ""
    for attr in ("text", "content"):
        value = getattr(response, attr, None)
        if value is not None:
            return str(value)
    if isinstance(response, dict):
        for key in ("text", "content", "message"):
            if response.get(key) is not None:
                return str(response.get(key))
    return str(response)


def _serialized_chars(value: Any) -> int:
    try:
        import json

        return len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return len(str(value or ""))
