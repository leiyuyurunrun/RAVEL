from __future__ import annotations

import re
from typing import Any, Dict, Optional

from evotx.runtime.packet_view_registry import PACKET_VIEW_TOOLS, PacketViewRegistry


ALLOWED_EVIDENCE_TOOLS = {
    *PACKET_VIEW_TOOLS,
    "read_function_chunk",
}


class EvidenceToolRegistry:
    """Strict read-only evidence tool registry for constrained judge follow-up."""

    def __init__(
        self,
        packet: Dict[str, Any],
        source_registry: Optional[Any] = None,
        default_chain: str = "eth",
        max_chars: int = 24000,
        debug_allow_list_views: bool = False,
    ):
        self.packet_registry = PacketViewRegistry(
            packet,
            max_chars=max_chars,
            debug_allow_list_views=debug_allow_list_views,
        )
        self.source_registry = source_registry
        self.default_chain = default_chain

    def call(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tool_name = str(tool_name or "").strip()
        args = dict(args or {})

        if tool_name in PACKET_VIEW_TOOLS:
            return self.packet_registry.call(tool_name, args)

        if tool_name == "read_function_chunk":
            if self.source_registry is None:
                return _tool_error(
                    tool=tool_name,
                    args=args,
                    tool_status="tool_unavailable",
                    note="read_function_chunk is disabled because no SourceToolRegistry is configured.",
                )
            args, resolution = self._resolve_source_args(args)
            if resolution:
                args["source_resolution"] = resolution
            args["max_snippets"] = _bounded_int(args.get("max_snippets"), default=3, minimum=1, maximum=5)
            args["max_chars_per_snippet"] = _bounded_int(
                args.get("max_chars_per_snippet"),
                default=3500,
                minimum=500,
                maximum=6000,
            )
            request = {"tool": "read_function_chunk", **args}
            result = self.source_registry.call(request, default_chain=self.default_chain)
            result.setdefault("tool", "read_function_chunk")
            result.setdefault("tool_status", "ok")
            result.setdefault(
                "note",
                "Read-only source evidence result; no attack judgment is made by this tool.",
            )
            result["args"] = args
            if resolution:
                result.setdefault("summary", {})["source_resolution"] = resolution
            result["returned_evidence_ids"] = _source_returned_evidence_ids(result)
            return result

        return _tool_error(
            tool=tool_name,
            args=args,
            tool_status="unsupported_tool",
            note=f"Unsupported evidence tool: {tool_name}",
        )

    def _resolve_source_args(self, args: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        resolved = dict(args or {})
        preserve_evidence_target = bool(
            resolved.pop("preserve_evidence_target", False)
        )
        evidence_id = str(resolved.get("evidence_id") or "").strip()
        target_function = str(
            resolved.get("target_function")
            or resolved.get("requested_function")
            or _target_function_from_focus(resolved.get("focus"))
            or ""
        ).strip()
        if target_function and not _is_meaningful_source_function(target_function):
            target_function = ""
        address = str(
            resolved.get("address")
            or resolved.get("contract")
            or resolved.get("callee")
            or ""
        ).strip()
        function_name = str(
            resolved.get("function_name")
            or resolved.get("function")
            or resolved.get("selector")
            or resolved.get("raw_selector")
            or resolved.get("function_selector")
            or ""
        ).strip()
        if function_name and not _is_meaningful_source_function(function_name):
            function_name = ""
            for key in ("function_name", "function", "selector", "raw_selector", "function_selector"):
                if key in resolved and not _is_meaningful_source_function(resolved.get(key)):
                    resolved.pop(key, None)
        resolution: Dict[str, Any] = {}
        if evidence_id and (not _is_address_like(address) or not function_name):
            requested_evidence_id = evidence_id
            rows = self.packet_registry.find_evidence_rows(evidence_id, limit=8)
            row = _choose_source_target_row(rows)
            fallback_reason = ""
            if (
                row
                and not target_function
                and not preserve_evidence_target
                and _needs_more_specific_source_row(row, evidence_id=evidence_id)
            ):
                child_rows = self.packet_registry.find_descendant_call_rows(
                    row.get("id"),
                    limit=40,
                    max_depth=8,
                )
                if not child_rows:
                    child_rows = self.packet_registry.find_child_call_rows(row.get("id"), limit=12)
                child_row = _choose_source_target_row(child_rows)
                if child_row and _source_row_score(child_row) > _source_row_score(row):
                    fallback_reason = (
                        "requested evidence resolved to a root-like or weak source target; "
                        "selected a more concrete descendant call instead"
                    )
                    row = child_row
            if row:
                resolved_evidence_id = str(row.get("evidence_id") or "").strip()
                if fallback_reason and resolved_evidence_id:
                    evidence_id = resolved_evidence_id
                    resolved["evidence_id"] = resolved_evidence_id
                row_address = _row_source_address(row)
                row_function = _row_source_function(row)
                if not _is_meaningful_source_function(row_function):
                    row_function = ""
                if not _is_address_like(address) and row_address:
                    address = row_address
                    resolved["address"] = row_address
                if not function_name and row_function:
                    function_name = row_function
                    resolved["function_name"] = row_function
                resolution = {
                    "evidence_id": evidence_id,
                    "requested_evidence_id": requested_evidence_id,
                    "resolved_evidence_id": resolved_evidence_id or evidence_id,
                    "matched_view": row.get("_view", ""),
                    "matched_path": row.get("_path", ""),
                    "resolved_address": address,
                    "resolved_function_name": function_name,
                    "requested_target_function": target_function,
                    "source": "packet_evidence_id",
                    "preserve_evidence_target": preserve_evidence_target,
                }
                if _is_selector_like_function(function_name):
                    resolution["selector_based_lookup"] = True
                    resolution["note"] = (
                        "source lookup is selector-based and may fail without ABI/signature mapping"
                    )
                if fallback_reason:
                    resolution["fallback_reason"] = fallback_reason
                    resolution["fallback_target_policy"] = (
                        "retargeted_source_chunk_is_supporting_only; prefer a "
                        "trace-linked target_function or concrete child call for "
                        "decisive source reasoning"
                    )
            else:
                resolution = {
                    "evidence_id": evidence_id,
                    "matched": False,
                    "source": "packet_evidence_id",
                }
        if not resolved.get("chain"):
            resolved["chain"] = self.default_chain
        if address:
            resolved["address"] = address
        if function_name:
            resolved["function_name"] = function_name
        if target_function:
            resolved["target_function"] = target_function
            resolved["function_name"] = target_function
        return resolved, resolution


def _target_function_from_focus(value: Any) -> str:
    """Extract an explicitly named internal function from a follow-up focus."""
    text = str(value or "")
    for pattern in (
        r"\btarget_function\s*[=:]\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)",
        r"\b(_[A-Za-z][A-Za-z0-9_]*)\b",
    ):
        match = re.search(pattern, text)
        if match:
            return str(match.group(1) or "").strip()
    return ""


def _tool_error(tool: str, args: Dict[str, Any], tool_status: str, note: str) -> Dict[str, Any]:
    return {
        "tool": tool,
        "tool_status": tool_status,
        "args": args,
        "summary": {"matched": False},
        "evidence": {},
        "returned_evidence_ids": [],
        "note": note,
    }


def _source_returned_evidence_ids(result: Dict[str, Any]) -> list[str]:
    evidence = result.get("evidence") or {}
    snippets = list(evidence.get("snippets") or [])
    snippets.extend(list(evidence.get("modifier_snippets") or []))
    ids = []
    for idx, snippet in enumerate(snippets):
        if isinstance(snippet, dict) and snippet.get("evidence_id"):
            ids.append(str(snippet["evidence_id"]))
        else:
            ids.append(f"source_chunk:{idx}")
    return ids


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _choose_source_target_row(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    scored = sorted(rows, key=_source_row_score, reverse=True)
    return scored[0]


def _source_row_score(row: Dict[str, Any]) -> int:
    score = 0
    if _is_address_like(_row_source_address(row)):
        score += 5
    function_name = _row_source_function(row)
    if _is_meaningful_source_function(function_name):
        score += 8
    elif function_name:
        score -= 8
    view = str(row.get("_view") or "")
    if view == "critical_call_view":
        score += 8
    elif view == "value_release_view":
        score += 7
    elif view == "unknown_selector_view":
        score += 5
    elif view == "trace_view":
        score += 3
    elif view == "trace_outline_view":
        score += 1
    depth = row.get("depth")
    try:
        score += min(max(int(depth), 0), 8)
    except (TypeError, ValueError):
        pass
    if row.get("nearby_state_ids"):
        score += 2
    if row.get("nearby_event_ids"):
        score += 1
    why = str(row.get("why_included") or "").lower()
    for keyword in (
        "value",
        "release",
        "withdraw",
        "mint",
        "burn",
        "delegatecall",
        "initializer",
        "unknown",
        "reentrant",
        "privileged",
        "admin",
        "owner",
    ):
        if keyword in why:
            score += 2
            break
    if _is_root_like_source_row(row):
        score -= 20
    return score


def _row_source_address(row: Dict[str, Any]) -> str:
    for key in (
        "callee",
        "callee_address",
        "address",
        "contract",
        "target_contract",
        "parent_address",
        "to",
    ):
        value = str(row.get(key) or "").strip().lower()
        if _is_address_like(value):
            return value
    return ""


def _row_source_function(row: Dict[str, Any]) -> str:
    for key in (
        "function_name",
        "function",
        "function_signature",
        "decoded_signature",
        "decoded_function",
        "method",
        "name",
        "parent_function",
        "selector",
        "raw_selector",
        "function_selector",
    ):
        value = str(row.get(key) or "").strip()
        if value and value not in {"<unknown>", "None", "null"}:
            return value
    return ""


def _is_meaningful_source_function(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {
        "transaction_root",
        "new <unknown>",
        "<unknown>",
        "unknown",
        "none",
        "null",
    }:
        return False
    if lowered.startswith("transaction_root"):
        return False
    if lowered.startswith("new <unknown>"):
        return False
    return True


def _needs_more_specific_source_row(row: Dict[str, Any], *, evidence_id: str = "") -> bool:
    if not row:
        return False
    if _is_root_like_source_row(row):
        return True
    if not _is_meaningful_source_function(_row_source_function(row)):
        return True
    eid = str(evidence_id or row.get("evidence_id") or "").strip().lower()
    if eid in {"call:0", "trace:0", "trace:call:0"}:
        return True
    return False


def _is_root_like_source_row(row: Dict[str, Any]) -> bool:
    evidence_id = str(row.get("evidence_id") or "").strip().lower()
    if evidence_id in {"call:0", "trace:0", "trace:call:0"}:
        return True
    function_name = str(_row_source_function(row) or "").strip().lower()
    if not _is_meaningful_source_function(function_name):
        return True
    try:
        depth = int(row.get("depth"))
    except (TypeError, ValueError):
        depth = None
    if depth == 0 and (
        function_name.startswith("transaction_root")
        or _is_selector_like_function(function_name)
    ):
        return True
    return False


def _is_selector_like_function(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text.startswith("0x"):
        return False
    hex_part = text[2:]
    return len(hex_part) in {8, 64} and all(ch in "0123456789abcdef" for ch in hex_part)


def _is_address_like(value: Any) -> bool:
    text = str(value or "").strip()
    return len(text) == 42 and text.startswith("0x")
