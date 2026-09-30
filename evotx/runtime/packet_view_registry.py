from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evotx.runtime.packet_view_manifest import VIEW_MANIFEST
from evotx.runtime.view_catalog import packet_view_row_count


PACKET_VIEW_TOOLS = {
    "list_packet_views",
    "read_packet_view",
    "read_evidence_by_id",
    "read_evidence_context",
    "get_local_call_context",
}


class PacketViewRegistry:
    """Read-only registry for compact packet evidence views."""

    def __init__(
        self,
        packet: Dict[str, Any],
        max_chars: int = 24000,
        debug_allow_list_views: bool = False,
    ):
        self.packet = packet or {}
        self.views = self.packet.get("views", {}) if isinstance(self.packet, dict) else {}
        self.max_chars = max_chars
        self.debug_allow_list_views = debug_allow_list_views
        self._evidence_store_cache: Optional[Dict[str, Any]] = None
        self._store_by_evidence_id: Optional[Dict[str, Dict[str, Any]]] = None
        self._store_by_display_id: Optional[Dict[str, Dict[str, Any]]] = None
        self._store_children_by_parent_id: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def call(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = dict(args or {})
        if tool_name == "list_packet_views":
            if not self.debug_allow_list_views:
                return _tool_response(
                    tool="list_packet_views",
                    args=args,
                    tool_status="rejected",
                    summary={"matched": False},
                    evidence={"views": []},
                    note=(
                        "list_packet_views is debug-only; use "
                        "allowed_followup_view_summary and request "
                        "read_packet_view directly."
                    ),
                )
            return self.list_packet_views(args)
        if tool_name == "read_packet_view":
            return self.read_packet_view(args)
        if tool_name == "read_evidence_by_id":
            return self.read_evidence_by_id(args)
        if tool_name == "read_evidence_context":
            return self.read_evidence_context(args)
        if tool_name == "get_local_call_context":
            return self.get_local_call_context(args)
        return _tool_response(
            tool=tool_name,
            args=args,
            tool_status="unsupported_tool",
            summary={"matched": False},
            evidence={},
            note=f"Unsupported packet evidence tool: {tool_name}",
        )

    def list_packet_views(self, args: Dict[str, Any]) -> Dict[str, Any]:
        rows = []
        for name, view in self.views.items():
            manifest_entry = dict(VIEW_MANIFEST.get(str(name), {}))
            rows.append({
                "view": name,
                "exists": True,
                "type": type(view).__name__,
                "row_count": packet_view_row_count(view),
                "tier": manifest_entry.get("tier", ""),
                "cost": manifest_entry.get("cost", ""),
                "cost_score": manifest_entry.get("cost_score", 0),
                "default_usage": manifest_entry.get("default_usage", ""),
                "prompt_role": manifest_entry.get("prompt_role", ""),
                "allow_first_pass": manifest_entry.get("allow_first_pass", False),
                "allow_followup": manifest_entry.get("allow_followup", False),
            })
        return _tool_response(
            tool="list_packet_views",
            args=args,
            tool_status="ok",
            summary={"view_count": len(rows)},
            evidence={"views": rows},
        )

    def read_packet_view(self, args: Dict[str, Any]) -> Dict[str, Any]:
        view_name = str(
            args.get("view") or args.get("view_name") or ""
        ).strip()
        if view_name and not args.get("view"):
            args = {**dict(args or {}), "view": view_name}
            args.pop("view_name", None)
        limit = _bounded_int(args.get("limit"), default=50, minimum=1, maximum=1000)
        keywords = [str(k).lower() for k in args.get("keywords", []) if str(k).strip()]
        evidence_ids = {str(eid) for eid in args.get("evidence_ids", []) if str(eid)}

        if view_name not in self.views:
            store_ref = self.packet.get("evidence_store", {}) if isinstance(self.packet, dict) else {}
            store_path = Path(str(store_ref.get("path") or "")) if isinstance(store_ref, dict) and store_ref.get("path") else None
            return _tool_response(
                tool="read_packet_view",
                args=args,
                tool_status="view_not_found",
                summary={
                    "matched": False,
                    "view": view_name,
                    "available_views": sorted(self.views.keys()),
                    "build_config": dict(self.packet.get("build_config", {}) or {}),
                    "evidence_store_exists": bool(store_path and store_path.exists()),
                },
                evidence={"rows": []},
                note=(
                    f"Packet view not found: {view_name}. View not materialized in current packet; "
                    "runtime should rebuild packet with required views. This is a packet construction "
                    "issue, not proof that the underlying evidence is unavailable."
                ),
            )

        rows = _flatten_evidence_rows(self.views[view_name], view_name=view_name)
        if not evidence_ids:
            direct_rows = _direct_evidence_rows(self.views[view_name], view_name=view_name)
            if direct_rows:
                rows = direct_rows
        filtered = []
        wanted_alternates = set()
        for eid in evidence_ids:
            wanted_alternates.update(_evidence_id_lookup_alternates(eid))
        for row in rows:
            if evidence_ids and not _row_matches_evidence_ids(row, wanted_alternates):
                continue
            if keywords:
                haystack = _json_dumps(row).lower()
                if not any(keyword in haystack for keyword in keywords):
                    continue
            filtered.append(row)
            if len(filtered) >= limit:
                break

        delivered_rows = _limit_payload(filtered, self.max_chars)
        evidence = {
            "view": view_name,
            "rows": delivered_rows,
        }
        return _tool_response(
            tool="read_packet_view",
            args=args,
            tool_status="ok",
            summary={
                "view": view_name,
                "total_rows": len(rows),
                "returned_rows": _delivered_row_count(delivered_rows),
                "filtered_by_keywords": keywords,
                "filtered_by_evidence_ids": sorted(evidence_ids),
            },
            evidence=evidence,
            returned_evidence_ids=_extract_payload_evidence_ids(delivered_rows),
        )

    def read_evidence_by_id(self, args: Dict[str, Any]) -> Dict[str, Any]:
        raw_ids = args.get("evidence_ids", [])
        if args.get("evidence_id") and not raw_ids:
            raw_ids = [args.get("evidence_id")]
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        evidence_ids = _unique(str(eid).strip() for eid in raw_ids if str(eid).strip())
        limit = _bounded_int(args.get("limit"), default=80, minimum=1, maximum=200)
        if not evidence_ids:
            return _tool_response(
                tool="read_evidence_by_id",
                args=args,
                tool_status="invalid_args",
                summary={"matched": False},
                evidence={"rows": []},
                note="evidence_ids is required.",
            )

        wanted = set(evidence_ids)
        rows = []
        found_ids = set()
        for view_name, view in self.views.items():
            for row in _flatten_evidence_rows(view, view_name=view_name):
                if str(row.get("evidence_id", "")) in wanted:
                    rows.append(row)
                    found_ids.add(str(row.get("evidence_id", "")))
                    if len(rows) >= limit:
                        break
            if len(rows) >= limit:
                break
        if len(rows) < limit:
            for eid in evidence_ids:
                if eid in found_ids:
                    continue
                store_row = self._read_evidence_store_row(eid)
                if store_row:
                    row = dict(store_row)
                    row["_view"] = "evidence_store"
                    row["_path"] = f"evidence.{eid}"
                    rows.append(row)
                    found_ids.add(eid)
                    if len(rows) >= limit:
                        break

        delivered_rows = _limit_payload(rows, self.max_chars)
        return _tool_response(
            tool="read_evidence_by_id",
            args=args,
            tool_status="ok",
            summary={
                "requested": evidence_ids,
                "returned_rows": _delivered_row_count(delivered_rows),
                "missing": [eid for eid in evidence_ids if eid not in found_ids],
            },
            evidence={"rows": delivered_rows},
            returned_evidence_ids=_extract_payload_evidence_ids(delivered_rows),
        )

    def find_evidence_rows(self, evidence_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        wanted = str(evidence_id or "").strip()
        if not wanted:
            return []
        wanted_alternates = _evidence_id_lookup_alternates(wanted)
        rows: List[Dict[str, Any]] = []
        for view_name, view in self.views.items():
            for row in _flatten_evidence_rows(view, view_name=view_name):
                row_eid = str(row.get("evidence_id", ""))
                if _normalize_evidence_id_for_lookup(row_eid) in wanted_alternates:
                    rows.append(row)
                    if len(rows) >= limit:
                        return rows
        store_row = self._read_evidence_store_row(wanted)
        if store_row:
            row = dict(store_row)
            row["_view"] = "evidence_store"
            row["_path"] = f"evidence.{wanted}"
            rows.append(row)
        return rows

    def _read_evidence_store_row(self, evidence_id: str) -> Dict[str, Any]:
        self._ensure_evidence_store_indexes()
        assert self._store_by_evidence_id is not None
        for candidate in _evidence_id_lookup_alternates(evidence_id):
            row = self._store_by_evidence_id.get(candidate)
            if row:
                return dict(row)
        return {}

    def _load_evidence_store(self) -> Dict[str, Any]:
        if self._evidence_store_cache is not None:
            return self._evidence_store_cache
        inline_store = (
            self.packet.get("evidence_store_inline", {})
            if isinstance(self.packet, dict)
            else {}
        )
        if isinstance(inline_store, dict) and inline_store:
            self._evidence_store_cache = inline_store
            return self._evidence_store_cache
        ref = self.packet.get("evidence_store", {}) if isinstance(self.packet, dict) else {}
        path_value = ref.get("path") if isinstance(ref, dict) else ""
        if not path_value:
            self._evidence_store_cache = {}
            return self._evidence_store_cache
        path = Path(path_value)
        if not path.exists():
            self._evidence_store_cache = {}
            return self._evidence_store_cache
        try:
            with path.open("r", encoding="utf-8") as f:
                self._evidence_store_cache = json.load(f)
        except Exception:
            self._evidence_store_cache = {}
        return self._evidence_store_cache

    def _ensure_evidence_store_indexes(self) -> None:
        if self._store_by_evidence_id is not None:
            return
        self._store_by_evidence_id = {}
        self._store_by_display_id = {}
        self._store_children_by_parent_id = {}
        store = self._load_evidence_store()
        evidence = store.get("evidence", {}) if isinstance(store, dict) else {}
        if not isinstance(evidence, dict):
            return
        for evidence_id, row_obj in evidence.items():
            if not isinstance(row_obj, dict):
                continue
            row = dict(row_obj)
            row.setdefault("evidence_id", str(evidence_id))
            row["_view"] = "evidence_store"
            row["_path"] = f"evidence.{evidence_id}"
            eid = str(row.get("evidence_id") or evidence_id)
            self._store_by_evidence_id[eid] = row
            self._store_by_evidence_id[_normalize_evidence_id_for_lookup(eid)] = row
            for key in _display_id_lookup_keys(row.get("id")):
                self._store_by_display_id[key] = row
            parent_id = row.get("parent_id")
            for key in _display_id_lookup_keys(parent_id):
                self._store_children_by_parent_id.setdefault(key, []).append(row)
            parent_eid = row.get("_parent_evidence_id") or row.get("parent_evidence_id")
            if parent_eid:
                self._store_children_by_parent_id.setdefault(str(parent_eid), []).append(row)

    def _store_row_by_display_id(self, display_id: Any) -> Optional[Dict[str, Any]]:
        self._ensure_evidence_store_indexes()
        assert self._store_by_display_id is not None
        for key in _display_id_lookup_keys(display_id):
            row = self._store_by_display_id.get(key)
            if row:
                return dict(row)
        return None

    def _store_children(self, parent_id: Any, limit: int = 12) -> List[Dict[str, Any]]:
        self._ensure_evidence_store_indexes()
        assert self._store_children_by_parent_id is not None
        rows: List[Dict[str, Any]] = []
        seen = set()
        for key in _display_id_lookup_keys(parent_id):
            for row in self._store_children_by_parent_id.get(key, []):
                marker = str(row.get("evidence_id") or row.get("_path") or id(row))
                if marker in seen:
                    continue
                seen.add(marker)
                rows.append(dict(row))
                if len(rows) >= limit:
                    return rows
        return rows

    def find_child_call_rows(self, display_id: Any, limit: int = 12) -> List[Dict[str, Any]]:
        if display_id is None:
            return []
        rows = self._find_children(display_id, limit=limit)
        out = []
        for row in rows:
            evidence_id = str(row.get("evidence_id") or "")
            row_type = str(row.get("type") or "").lower()
            view = str(row.get("_view") or "")
            if not (
                evidence_id.startswith("call:")
                or row_type in {"call", "delegatecall", "staticcall"}
                or view in {"trace_view", "critical_call_view", "unknown_selector_view", "value_release_view"}
            ):
                continue
            if row.get("callee") or row.get("address") or row.get("function") or row.get("function_name"):
                out.append(row)
                if len(out) >= limit:
                    break
        return out

    def find_descendant_call_rows(
        self,
        display_id: Any,
        limit: int = 40,
        max_depth: int = 6,
    ) -> List[Dict[str, Any]]:
        """Return lightweight descendant call-like rows for source targeting.

        This is intentionally bounded. It is used when a judge asks for source
        around a broad/root call id; the runtime can then choose a concrete
        child call instead of sending transaction_root/call:0 to source lookup.
        """
        if display_id is None:
            return []
        out: List[Dict[str, Any]] = []
        queue: List[tuple[Any, int]] = [(display_id, 0)]
        seen_ids = {str(display_id)}
        while queue and len(out) < limit:
            parent_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            children = self._find_children(parent_id, limit=200)
            for row in children:
                child_id = row.get("id")
                child_key = str(child_id)
                if child_key and child_key not in seen_ids:
                    seen_ids.add(child_key)
                    queue.append((child_id, depth + 1))
                if not _is_call_like_source_row(row):
                    continue
                out.append(row)
                if len(out) >= limit:
                    break
        return out

    def read_evidence_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        evidence_id = str(args.get("evidence_id") or "").strip()
        if not evidence_id and isinstance(args.get("evidence_ids"), list):
            evidence_id = next(
                (str(eid).strip() for eid in args.get("evidence_ids", []) if str(eid).strip()),
                "",
            )
        radius = _bounded_int(args.get("radius"), default=2, minimum=0, maximum=8)
        include_parent = bool(args.get("include_parent", True))
        include_children = bool(args.get("include_children", True))
        include_nearby_events = bool(args.get("include_nearby_events", True))
        include_nearby_state = bool(args.get("include_nearby_state", True))
        if not evidence_id:
            return _tool_response(
                tool="read_evidence_context",
                args=args,
                tool_status="invalid_args",
                summary={"matched": False},
                evidence={},
                note="evidence_id is required.",
            )

        all_rows_by_view = {
            name: _flatten_evidence_rows(view, view_name=name)
            for name, view in self.views.items()
        }
        wanted_alternates = _evidence_id_lookup_alternates(evidence_id)
        target_rows = [
            row
            for rows in all_rows_by_view.values()
            for row in rows
            if _normalize_evidence_id_for_lookup(row.get("evidence_id", "")) in wanted_alternates
        ]
        if not target_rows:
            store_row = self._read_evidence_store_row(evidence_id)
            if store_row:
                row = dict(store_row)
                row["_view"] = "evidence_store"
                row["_path"] = f"evidence.{evidence_id}"
                target_rows.append(row)
        if not target_rows:
            return _tool_response(
                tool="read_evidence_context",
                args=args,
                tool_status="not_found",
                summary={"matched": False, "evidence_id": evidence_id},
                evidence={},
                note=f"Evidence id not found: {evidence_id}",
            )

        target = _choose_best_context_row(target_rows)
        context_rows = [target]
        related_ids: List[str] = []

        for view_name in ("trace_view", "critical_call_view", "event_view", "state_change_view"):
            rows = all_rows_by_view.get(view_name, [])
            for idx, row in enumerate(rows):
                if str(row.get("evidence_id", "")) != evidence_id:
                    continue
                lo = max(0, idx - radius)
                hi = min(len(rows), idx + radius + 1)
                context_rows.extend(rows[lo:hi])
                break

        if include_parent and target.get("parent_id") is not None:
            parent_row = self._find_row_by_display_id(target.get("parent_id"))
            if parent_row:
                context_rows.append(parent_row)

        if include_children:
            target_display_id = target.get("id")
            if target_display_id is not None:
                context_rows.extend(self._find_children(target_display_id, limit=12))

        if include_nearby_events:
            related_ids.extend(str(x) for x in target.get("nearby_event_ids", []) if x)
        if include_nearby_state:
            related_ids.extend(str(x) for x in target.get("nearby_state_ids", []) if x)

        if related_ids:
            related = self.read_evidence_by_id({"evidence_ids": _unique(related_ids), "limit": 40})
            context_rows.extend(related.get("evidence", {}).get("rows", []))

        context_rows = _unique_rows(context_rows)
        delivered_rows = _limit_payload(context_rows, self.max_chars)
        return _tool_response(
            tool="read_evidence_context",
            args=args,
            tool_status="ok",
            summary={
                "evidence_id": evidence_id,
                "returned_rows": _delivered_row_count(delivered_rows),
                "radius": radius,
            },
            evidence={"rows": delivered_rows},
            returned_evidence_ids=_extract_payload_evidence_ids(delivered_rows),
        )

    def _find_row_by_display_id(
        self,
        display_id: Any,
        view_names: Optional[Iterable[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        for view_name in view_names or ("trace_view", "critical_call_view", "event_view", "state_change_view"):
            for row in _flatten_evidence_rows(self.views.get(view_name, []), view_name=view_name):
                if row.get("id") == display_id:
                    return row
        return self._store_row_by_display_id(display_id)

    def _find_best_call_row_by_display_id(self, display_id: Any) -> Optional[Dict[str, Any]]:
        trace_row = self._find_row_by_display_id(display_id, ("trace_view",))
        critical_row = self._find_row_by_display_id(display_id, ("critical_call_view",))
        if trace_row and critical_row:
            return _merge_context_rows(trace_row, critical_row)
        return critical_row or trace_row or self._find_row_by_display_id(display_id)

    def _find_children(self, parent_id: Any, limit: int = 12) -> List[Dict[str, Any]]:
        rows = []
        for view_name in ("trace_view", "critical_call_view", "event_view", "state_change_view"):
            for row in _flatten_evidence_rows(self.views.get(view_name, []), view_name=view_name):
                if row.get("parent_id") == parent_id:
                    rows.append(row)
                    if len(rows) >= limit:
                        return rows
        if len(rows) < limit:
            for row in self._store_children(parent_id, limit=limit - len(rows)):
                rows.append(row)
                if len(rows) >= limit:
                    break
        return rows

    def _rebuild_call_path(self, row: Dict[str, Any]) -> List[str]:
        path_parts: List[str] = []
        current = row
        visited = set()
        for _ in range(20):
            pid = current.get("parent_id")
            if pid is None or pid in visited:
                break
            visited.add(pid)
            parent = self._find_row_by_display_id(pid)
            if parent is None:
                break
            fn = parent.get("function") or parent.get("selector") or "<unknown>"
            path_parts.append(f"{parent.get('id')}:{fn}")
            current = parent
        path_parts.reverse()
        fn = row.get("function") or row.get("selector") or "<unknown>"
        path_parts.append(f"{row.get('id')}:{fn}")
        return path_parts

    def get_local_call_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        call_id = _normalize_call_id(
            args.get("call_id", args.get("id", args.get("evidence_id")))
        )
        child_limit = _bounded_int(args.get("child_limit"), default=8, minimum=1, maximum=30)
        include_events = bool(args.get("include_events", True))
        include_state = bool(args.get("include_state", True))

        if call_id is None or (isinstance(call_id, str) and not call_id.strip()):
            return _tool_response(
                tool="get_local_call_context",
                args=args,
                tool_status="invalid_args",
                summary={"matched": False},
                evidence={},
                note="call_id is required.",
            )

        target = self._find_best_call_row_by_display_id(call_id)
        if target is None:
            return _tool_response(
                tool="get_local_call_context",
                args=args,
                tool_status="not_found",
                summary={"matched": False, "call_id": call_id},
                evidence={},
                note=f"Call id not found in views: {call_id}",
            )

        evidence: Dict[str, Any] = {"call": target}
        notes: List[str] = []

        if str(target.get("why_included", "")).startswith("reentrant_call"):
            chain_rows = [
                row
                for row in (
                    self._find_best_call_row_by_display_id(chain_id)
                    for chain_id in target.get("reentrant_chain_ids", [])
                )
                if row
            ]
            evidence["reentrancy_context"] = {
                "why_included": target.get("why_included"),
                "first_entry_id": target.get("first_entry_id"),
                "reentrant_depth_gap": target.get("reentrant_depth_gap"),
                "reentrant_chain_ids": list(target.get("reentrant_chain_ids", [])),
                "chain_length": len(chain_rows),
            }
            if chain_rows:
                evidence["reentrant_chain_calls"] = chain_rows

        parent_id = target.get("parent_id")
        if parent_id is not None:
            parent = self._find_best_call_row_by_display_id(parent_id)
            if parent:
                evidence["parent_call"] = parent

        children = self._find_children(call_id, limit=child_limit + 1)
        if len(children) > child_limit:
            notes.append(f"Children truncated: {len(children)} found, showing {child_limit}.")
            children = children[:child_limit]
        evidence["children_calls"] = children

        related_ids: List[str] = []
        local_call_rows = [target, *children]
        for local_row in local_call_rows:
            if include_events:
                related_ids.extend(
                    str(value)
                    for value in list(local_row.get("nearby_event_ids") or [])
                    if value
                )
            if include_state:
                related_ids.extend(
                    str(value)
                    for value in list(local_row.get("nearby_state_ids") or [])
                    if value
                )

        if related_ids:
            related = self.read_evidence_by_id({
                "evidence_ids": _unique(related_ids)[:40],
                "limit": 40,
            })
            related_rows = _unique_rows_by_evidence_id(
                related.get("evidence", {}).get("rows", [])
            )
            event_rows = [r for r in related_rows if _is_event_row(r)]
            state_rows = [r for r in related_rows if _is_state_row(r)]
            if event_rows:
                evidence["subtree_events"] = event_rows
            if state_rows:
                evidence["subtree_state_changes"] = state_rows

        call_path_tail = target.get("call_path_tail")
        if call_path_tail:
            evidence["call_path"] = call_path_tail
        else:
            evidence["call_path"] = self._rebuild_call_path(target)

        if notes:
            evidence["notes"] = notes

        delivered_evidence = _limit_payload_dict(evidence, self.max_chars)
        return _tool_response(
            tool="get_local_call_context",
            args=args,
            tool_status="ok",
            summary={
                "call_id": call_id,
                "has_parent": bool(delivered_evidence.get("parent_call")),
                "child_count": len(delivered_evidence.get("children_calls", [])),
                "event_count": len(delivered_evidence.get("subtree_events", [])),
                "state_count": len(delivered_evidence.get("subtree_state_changes", [])),
                "is_reentrant_candidate": bool(
                    str(target.get("why_included", "")).startswith("reentrant_call")
                ),
                "reentrant_chain_count": len(
                    delivered_evidence.get("reentrant_chain_calls", [])
                ),
                "first_entry_id": target.get("first_entry_id"),
            },
            evidence=delivered_evidence,
            returned_evidence_ids=_extract_payload_evidence_ids(delivered_evidence),
        )


def _flatten_evidence_rows(view: Any, view_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def visit(obj: Any, path: str, parent: Optional[Dict[str, Any]] = None) -> None:
        if isinstance(obj, dict):
            if "evidence_id" in obj:
                row = dict(obj)
                row["_view"] = view_name
                row["_path"] = path
                if parent and parent.get("evidence_id") and parent.get("evidence_id") != row.get("evidence_id"):
                    row["_parent_evidence_id"] = parent.get("evidence_id")
                rows.append(row)
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    visit(value, f"{path}.{key}", obj if "evidence_id" in obj else parent)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                visit(item, f"{path}[{idx}]", parent)

    visit(view, view_name)
    return rows


def _direct_evidence_rows(view: Any, view_name: str) -> List[Dict[str, Any]]:
    if isinstance(view, list):
        candidates = view
        base_path = view_name
    elif isinstance(view, dict):
        candidates = None
        base_path = view_name
        for key in (
            "rows",
            "records",
            "events",
            "transfers",
            "release_records",
            "releases",
            "deltas",
            "auth_contexts",
            "profiles",
            "unknown_selectors",
            "beneficiary_paths",
            "controller_hints",
            "pairs",
            "entries",
            "operations",
        ):
            value = view.get(key)
            if isinstance(value, list):
                candidates = value
                base_path = f"{view_name}.{key}"
                break
        if candidates is None:
            candidates = [view] if view.get("evidence_id") else []
    else:
        candidates = []

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        if not isinstance(item, dict) or "evidence_id" not in item:
            continue
        row = dict(item)
        row["_view"] = view_name
        row["_path"] = f"{base_path}[{idx}]"
        rows.append(row)
    return rows


def _view_count(view: Any) -> int:
    if isinstance(view, list):
        return len(view)
    if isinstance(view, dict):
        for key in ("records", "rows", "views", "events"):
            if isinstance(view.get(key), list):
                return len(view[key])
        return 1
    return 0


def _tool_response(
    *,
    tool: str,
    args: Dict[str, Any],
    tool_status: str,
    summary: Dict[str, Any],
    evidence: Dict[str, Any],
    note: str = "Read-only packet evidence result; no attack judgment is made by this tool.",
    returned_evidence_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "tool": tool,
        "tool_status": tool_status,
        "args": args,
        "summary": summary,
        "evidence": evidence,
        "returned_evidence_ids": returned_evidence_ids or [],
        "note": note,
    }


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _normalize_call_id(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower().startswith("call:"):
            stripped = stripped.split(":", 1)[1]
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return value


def _display_id_lookup_keys(value: Any) -> List[str]:
    keys: List[str] = []
    if value is None or value == "":
        return keys
    raw = str(value).strip()
    candidates = [raw]
    normalized = _normalize_call_id(raw)
    candidates.append(str(normalized))
    if raw.isdigit():
        candidates.append(f"call:{raw}")
    if raw.lower().startswith("call:"):
        suffix = raw.split(":", 1)[1]
        candidates.append(suffix)
    for item in candidates:
        text = str(item or "").strip()
        if text and text not in keys:
            keys.append(text)
    return keys


def _normalize_evidence_id_for_lookup(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("address:"):
        return "address:" + text.split(":", 1)[1].lower()
    return text


def _evidence_id_lookup_alternates(value: Any) -> set[str]:
    text = str(value or "").strip()
    alternates = {_normalize_evidence_id_for_lookup(text)}
    if ":" in text:
        suffix = text.split(":", 1)[1]
        if ":" in suffix:
            alternates.add(_normalize_evidence_id_for_lookup(suffix))
    return alternates


def _choose_best_context_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    best = rows[0]
    for row in rows[1:]:
        if _row_context_score(row) > _row_context_score(best):
            best = row
    return best


def _row_context_score(row: Dict[str, Any]) -> int:
    score = 0
    view = str(row.get("_view"))
    if view == "critical_call_view":
        score += 5
    elif view in {"semantic_state_delta_view", "price_relevant_state_view"}:
        score += 4
    elif view in {"state_change_view", "event_view"}:
        score += 3
    elif view == "trace_view":
        score += 1
    for key in (
        "why_included",
        "first_entry_id",
        "reentrant_depth_gap",
        "reentrant_chain_ids",
        "nearby_event_ids",
        "nearby_state_ids",
    ):
        if row.get(key):
            score += 2
    return score


def _unique_rows_by_evidence_id(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    no_id: List[Dict[str, Any]] = []
    for row in rows:
        evidence_id = str(row.get("evidence_id", ""))
        if evidence_id:
            grouped.setdefault(evidence_id, []).append(row)
        else:
            no_id.append(row)
    out = [_choose_best_context_row(group) for group in grouped.values()]
    out.extend(_unique_rows(no_id))
    return out


def _merge_context_rows(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(overlay or {}).items():
        if key in {"nearby_event_ids", "nearby_state_ids", "reentrant_chain_ids"}:
            merged[key] = _unique(
                [str(x) for x in merged.get(key, []) or []]
                + [str(x) for x in value or []]
            )
            if key == "reentrant_chain_ids":
                merged[key] = [_normalize_call_id(x) for x in merged[key]]
        elif key in {"why_included", "first_entry_id", "reentrant_depth_gap"}:
            merged[key] = value
        elif key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value

    views = _unique(
        str(v)
        for v in (base.get("_view"), overlay.get("_view"))
        if v
    )
    if len(views) > 1:
        merged["_merged_views"] = views
    return merged


def _is_event_row(row: Dict[str, Any]) -> bool:
    evidence_id = str(row.get("evidence_id", ""))
    return (
        row.get("_view") == "event_view"
        or str(row.get("type", "")).lower() == "event"
        or evidence_id.startswith("event:")
    )


def _is_state_row(row: Dict[str, Any]) -> bool:
    evidence_id = str(row.get("evidence_id", ""))
    row_type = str(row.get("type", "")).lower()
    return (
        row.get("_view") in ("state_change_view", "semantic_state_delta_view", "price_relevant_state_view")
        or row_type in ("sload", "sstore")
        or evidence_id.startswith("sload:")
        or evidence_id.startswith("sstore:")
        or evidence_id.startswith("state_semantic:")
        or evidence_id.startswith("price_state:")
    )


def _is_call_like_source_row(row: Dict[str, Any]) -> bool:
    evidence_id = str(row.get("evidence_id") or "")
    row_type = str(row.get("type") or "").lower()
    view = str(row.get("_view") or "")
    if not (
        evidence_id.startswith("call:")
        or row_type in {"call", "delegatecall", "staticcall", "callcode"}
        or view in {"trace_view", "critical_call_view", "unknown_selector_view", "value_release_view"}
    ):
        return False
    return bool(
        row.get("callee")
        or row.get("callee_address")
        or row.get("address")
        or row.get("contract")
        or row.get("target_contract")
        or row.get("to")
        or row.get("function")
        or row.get("function_name")
        or row.get("function_signature")
        or row.get("decoded_signature")
        or row.get("decoded_function")
        or row.get("method")
        or row.get("name")
        or row.get("selector")
    )


def _row_matches_evidence_ids(
    row: Dict[str, Any],
    wanted_alternates: set[str],
) -> bool:
    if not wanted_alternates:
        return True
    return bool(_row_candidate_evidence_ids(row) & wanted_alternates)


def _row_candidate_evidence_ids(row: Dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    scalar_keys = (
        "evidence_id",
        "source_evidence_id",
        "call_evidence_id",
        "_parent_evidence_id",
        "parent_evidence_id",
    )
    list_keys = (
        "anchor_evidence_ids",
        "related_evidence_ids",
        "path_evidence_ids",
        "nearby_state_ids",
        "nearby_event_ids",
        "supporting_evidence_ids",
        "state_order_evidence_ids",
    )
    for key in scalar_keys:
        value = row.get(key)
        if value not in (None, ""):
            candidates.update(_evidence_id_lookup_alternates(value))
    for key in list_keys:
        for value in list(row.get(key) or []):
            if value not in (None, ""):
                candidates.update(_evidence_id_lookup_alternates(value))
    for value in list(row.get("path_ids") or []):
        if value in (None, ""):
            continue
        candidates.update(_evidence_id_lookup_alternates(value))
        candidates.update(_evidence_id_lookup_alternates(f"call:{value}"))
    call_id = row.get("call_id") or row.get("id")
    if call_id not in (None, ""):
        candidates.update(_evidence_id_lookup_alternates(call_id))
        candidates.update(_evidence_id_lookup_alternates(f"call:{call_id}"))
    return candidates


def _extract_payload_evidence_ids(payload: Any) -> List[str]:
    evidence_ids: List[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            evidence_id = str(item.get("evidence_id") or "").strip()
            if evidence_id and evidence_id != "tool_payload_truncated":
                evidence_ids.append(evidence_id)
            for child in item.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(payload)
    return _unique(evidence_ids)


def _delivered_row_count(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if str((row or {}).get("evidence_id") or "")
        != "tool_payload_truncated"
    )


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _unique_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = (row.get("_view"), row.get("_path"), row.get("evidence_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _limit_payload(rows: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    total = 0
    for row in rows:
        size = len(_json_dumps(row))
        if kept and total + size > max_chars:
            kept.append({
                "evidence_id": "tool_payload_truncated",
                "note": f"Tool payload truncated after {len(kept)} rows to stay within max_chars.",
            })
            break
        kept.append(row)
        total += size
    return kept


def _limit_payload_dict(d: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    total = len(_json_dumps(d))
    if total <= max_chars:
        return d
    result: Dict[str, Any] = {}
    total = 0
    for key, value in d.items():
        size = len(_json_dumps({key: value}))
        if total + size > max_chars:
            result["_truncated"] = True
            break
        result[key] = value
        total += size
    return result
