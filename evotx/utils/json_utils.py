from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional


class JsonExtractionError(ValueError):
    pass


_THINK_BLOCK_RE = re.compile(
    r"<think(?:\s[^>]*)?>.*?</think\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_OPEN_RE = re.compile(r"<think(?:\s[^>]*)?>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_GBK_UTF8_MOJIBAKE_RE = re.compile(r"\u9225[\u3400-\u9fff][^\s]{0,96}")


def strip_tagged_thinking(text: str) -> tuple[str, Dict[str, Any]]:
    """Remove visible reasoning blocks before structured-output parsing.

    The raw response remains available to transcript recording. An unclosed
    opening tag discards the remaining tail so a temporary JSON object inside
    truncated reasoning cannot be mistaken for the final answer.
    """
    original = str(text or "")
    cleaned, block_count = _THINK_BLOCK_RE.subn("", original)
    unclosed = False
    open_match = _THINK_OPEN_RE.search(cleaned)
    if open_match is not None:
        cleaned = cleaned[: open_match.start()]
        unclosed = True
    cleaned, orphan_close_count = _THINK_CLOSE_RE.subn("", cleaned)
    cleaned = cleaned.strip()
    return cleaned, {
        "thinking_filter_applied": True,
        "thinking_block_count": int(block_count),
        "thinking_orphan_close_count": int(orphan_close_count),
        "thinking_unclosed": bool(unclosed),
        "thinking_chars_removed": max(0, len(original) - len(cleaned)),
    }


def structured_output_text(
    text: str,
    *,
    provider: str = "",
) -> tuple[str, Dict[str, Any]]:
    """Return provider-scoped text suitable for strict JSON extraction."""
    if str(provider or "").strip().lower() == "minimax":
        return strip_tagged_thinking(text)
    return str(text or ""), {
        "thinking_filter_applied": False,
        "thinking_block_count": 0,
        "thinking_orphan_close_count": 0,
        "thinking_unclosed": False,
        "thinking_chars_removed": 0,
    }


def repair_likely_mojibake(value: Any) -> Any:
    """Repair high-confidence UTF-8-as-GBK corruption in generated artifacts.

    This intentionally operates only where callers opt in after parsing an LLM
    Rule/Plan/Updater payload. Packet evidence, labels, and raw transcripts are
    never normalized by this helper.
    """
    if isinstance(value, dict):
        return {
            key: repair_likely_mojibake(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [repair_likely_mojibake(item) for item in value]
    if not isinstance(value, str) or "\u9225" not in value:
        return value

    def repair_match(match: re.Match[str]) -> str:
        text = match.group(0)
        try:
            repaired = text.encode("gbk").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
        return repaired if "\u9225" not in repaired else text

    try:
        repaired = value.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        repaired = _GBK_UTF8_MOJIBAKE_RE.sub(repair_match, value)
    return repaired if repaired.count("\u9225") < value.count("\u9225") else value


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    if text is None:
        raise JsonExtractionError("Cannot parse JSON from None.")

    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
        raise JsonExtractionError("Top-level JSON value is not an object.")
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        raise JsonExtractionError("No JSON object start found.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        ch = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise JsonExtractionError(
                        f"Extracted JSON candidate is invalid: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise JsonExtractionError(
                        "Extracted JSON value is not an object."
                    )
                return value

    raise JsonExtractionError("No complete JSON object found.")


def extract_last_schema_valid_object(
    text: str,
    schema_validator: Callable[[Dict[str, Any]], Optional[str]],
    *,
    schema_name: str = "JSON",
) -> tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    """Select the last complete JSON object accepted by a schema validator."""
    candidates = json_object_candidates(text)
    valid_candidates: List[tuple[int, int, Dict[str, Any]]] = []
    schema_errors: List[str] = []
    for index, (start, _, candidate) in enumerate(candidates):
        error = schema_validator(candidate)
        if error is None:
            valid_candidates.append((index, start, candidate))
        else:
            schema_errors.append(f"candidate[{index}]: {error}")

    metadata: Dict[str, Any] = {
        "json_selection_strategy": "last_schema_valid",
        "json_candidate_count": len(candidates),
        "schema_valid_candidate_count": len(valid_candidates),
        "selected_candidate_index": None,
        "selected_candidate_start": None,
    }
    if valid_candidates:
        selected_index, selected_start, selected = valid_candidates[-1]
        metadata["selected_candidate_index"] = selected_index
        metadata["selected_candidate_start"] = selected_start
        return selected, None, metadata

    if candidates:
        error = f"No JSON object matched the {schema_name} schema"
        if schema_errors:
            error += ": " + "; ".join(schema_errors[-3:])
        return {}, error, metadata

    try:
        extract_json_object(text)
    except (JsonExtractionError, ValueError) as exc:
        return {}, repr(exc), metadata
    return {}, f"No JSON object matched the {schema_name} schema", metadata


def json_object_candidates(text: str) -> List[tuple[int, int, Dict[str, Any]]]:
    """Return complete, top-level object candidates embedded in arbitrary text."""
    cleaned = str(text or "")
    decoder = json.JSONDecoder()
    candidates: List[tuple[int, int, Dict[str, Any]]] = []
    seen_spans: set[tuple[int, int]] = set()
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        end = start + consumed
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        candidates.append((start, end, value))
    return [
        candidate
        for index, candidate in enumerate(candidates)
        if not any(
            outer_index != index
            and outer_start < candidate[0]
            and candidate[1] <= outer_end
            for outer_index, (outer_start, outer_end, _) in enumerate(candidates)
        )
    ]


def read_json(path: str | Path | None, default: Optional[Any] = None) -> Any:
    if path is None:
        return default
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
