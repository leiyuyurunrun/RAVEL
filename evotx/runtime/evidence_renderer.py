from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evotx.runtime.view_catalog import (
    packet_view_render_policy,
    packet_view_row_count,
)
from evotx.utils.json_utils import stable_json_dumps


PROMPT_DICTIONARY_KEY = "__prompt_dictionary__"
PROMPT_DICTIONARY_EXTRA_VALUES_KEY = "__prompt_dictionary_extra_values__"

_FUNCTION_ALIAS_FIELDS = {
    "function",
    "function_name",
    "decoded_function",
    "parent_function",
    "target_function",
    "fn",
}

_FORMATTED_NUMBER_RE = re.compile(
    r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$"
)

_SUMMARY_ANCHOR_VIEWS = {
    "classification_digest_view",
    "market_mechanism_profile_view",
    "protocol_accounting_outcome_view",
    "beneficiary_controller_view",
}

_ROW_KEYS = (
    "rows",
    "records",
    "events",
    "transfers",
    "deltas",
    "release_records",
    "releases",
    "unknown_selectors",
    "auth_contexts",
    "profiles",
    "beneficiary_paths",
    "controller_hints",
    "pairs",
    "entries",
    "operations",
    "items",
)

_SUMMARY_SKIP_KEYS = set(_ROW_KEYS) | {
    "children",
    "calls",
    "trace",
    "full_trace",
    "raw",
    "raw_trace",
}

_LARGE_PROMPT_ROW_FIELDS: Dict[str, tuple[str, ...]] = {
    "critical_call_view": (
        "evidence_id",
        "id",
        "type",
        "call_type",
        "parent_id",
        "depth",
        "parent_function",
        "caller",
        "caller_label",
        "callee",
        "address_label",
        "function",
        "selector",
        "value",
        "why_included",
        "structural_reentry_kind",
        "nearby_event_ids",
        "nearby_state_ids",
        "first_entry_id",
        "reentrant_depth_gap",
        "reentrant_chain_ids",
    ),
    "value_release_view": (
        "evidence_id",
        "sensitive_call_evidence_id",
        "call_id",
        "function",
        "target_contract",
        "target_contract_label",
        "parent_id",
        "parent_function",
        "depth",
        "release_record_count",
        "repeated_release_count",
        "value_out_records",
        "recipients",
        "target_balance_changes",
        "recipient_balance_changes",
        "related_participant_deltas",
        "related_evidence_ids",
    ),
    "participant_net_delta_view": (
        "evidence_id",
        "address",
        "label",
        "label_sanitized",
        "is_tx_sender",
        "deltas",
        "related_evidence_ids",
        "source",
    ),
    "external_fundflow_view": (
        "evidence_id",
        "id",
        "from",
        "to",
        "token",
        "tokenType",
        "amount",
        "isReverted",
    ),
    "transfer_event_view": (
        "evidence_id",
        "event_id",
        "depth",
        "parent_id",
        "parent_function",
        "token",
        "token_label",
        "from",
        "to",
        "amount",
        "raw_event_evidence_id",
    ),
    "event_view": (
        "evidence_id",
        "id",
        "parent_id",
        "depth",
        "parent_function",
        "address",
        "address_label",
        "event",
        "args",
        "caller",
        "caller_label",
    ),
    "semantic_state_delta_view": (
        "evidence_id",
        "source_evidence_id",
        "contract",
        "contract_label",
        "variable",
        "variable_type",
        "key_address",
        "prev",
        "current",
        "delta",
        "normalized_delta",
        "semantic_confidence",
    ),
    "state_change_view": (
        "evidence_id",
        "id",
        "source",
        "op",
        "parent_id",
        "depth",
        "parent_function",
        "address",
        "address_label",
        "slot_key",
        "prev",
        "current",
        "value",
    ),
    "source_unavailable_auth_view": (
        "evidence_id",
        "call_evidence_id",
        "anchor_evidence_ids",
        "path_evidence_ids",
        "call_id",
        "type",
        "call_type",
        "parent_id",
        "depth",
        "parent_function",
        "caller",
        "caller_label_sanitized",
        "callee",
        "callee_label_sanitized",
        "function",
        "selector",
        "selector_unknown_or_weak",
        "critical_call_reason",
        "sensitive_action_tags",
        "observable_authorization_evidence",
        "observable_authorization_reason",
        "observations",
        "nearby_event_ids",
        "nearby_state_ids",
    ),
    "reentrancy_state_order_summary_view": (
        "evidence_id",
        "candidate_id",
        "formal_candidate",
        "candidate_tier",
        "structural_only_reasons",
        "outer_call_id",
        "outer_function",
        "outer_call_type",
        "external_edge_id",
        "external_edge_function",
        "external_edge_call_type",
        "reentry_call_id",
        "reentry_function",
        "reentry_call_type",
        "callback_kind",
        "logical_storage_context",
        "reentrant_depth_gap",
        "phase_witness_completeness",
        "phase_slot_witnesses",
        "state_access_count",
        "state_order_evidence_ids",
        "nearby_event_ids",
        "nearby_state_ids",
    ),
    "trace_view": (
        "evidence_id",
        "id",
        "type",
        "call_type",
        "parent_id",
        "depth",
        "caller",
        "caller_label",
        "callee",
        "address",
        "address_label",
        "function",
        "selector",
        "value",
        "args",
        "slot_key",
        "prev",
        "current",
        "event",
        "nearby_event_ids",
        "nearby_state_ids",
    ),
    "contribution_vs_payout_view": (
        "evidence_id",
        "participant",
        "participant_label",
        "input_assets",
        "output_assets",
        "contribution_records",
        "payout_records",
        "protocol_balance_deltas",
        "net_effect",
        "related_evidence_ids",
    ),
    "market_mechanism_profile_view": (
        "evidence_id",
        "profile_id",
        "profile_type",
        "market_source",
        "market_source_label",
        "source_kind",
        "source_evidence_ids",
        "consumer_evidence_ids",
        "outcome_evidence_ids",
        "all_profile_evidence_ids",
        "candidate_strength",
        "profile_reason",
        "competing_root_hints",
        "competing_root_evidence_ids",
        "limitation",
    ),
}


def format_evidence_context(
    evidence_packet: Dict[str, Any],
    max_view_chars: int = 20000,
    max_context_chars: int = 60000,
    packet_evidence_adequacy: Optional[Dict[str, Any]] = None,
) -> str:
    text, _ = format_evidence_context_with_metadata(
        evidence_packet,
        max_view_chars=max_view_chars,
        max_context_chars=max_context_chars,
        packet_evidence_adequacy=packet_evidence_adequacy,
    )
    return text


def format_evidence_context_with_metadata(
    evidence_packet: Dict[str, Any],
    max_view_chars: int = 20000,
    max_context_chars: int = 60000,
    packet_evidence_adequacy: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Format selected packet views with stable order and row-level budgets.

    Planner decides which views are relevant. This renderer decides how those
    views fit into the prompt, so a large low-priority view cannot crowd out
    structural/state evidence selected later in a dynamic plan.
    """
    if not evidence_packet:
        return "(No evidence views provided.)", []

    prepared_packet, prompt_dictionary = _prepare_prompt_packet(evidence_packet)
    chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []
    total_chars = 0
    ordered_items = _ordered_view_items(prepared_packet)
    original_order = list(prepared_packet.keys())
    rendered_order = [name for name, _ in ordered_items]
    packet_trace_truncated = _packet_trace_truncated(
        packet_evidence_adequacy or {})

    if prompt_dictionary:
        dictionary_body = stable_json_dumps(prompt_dictionary)
        dictionary_chunk = f"=== packet_dictionary ===\n{dictionary_body}"
        dictionary_chunk, removed = _truncate_text_with_removed(
            dictionary_chunk,
            min(max_view_chars, max_context_chars),
            "packet dictionary",
        )
        chunks.append(dictionary_chunk)
        total_chars = len(dictionary_chunk) + 2
        metadata.append({
            "view": "packet_dictionary",
            "rows_total": (
                len(prompt_dictionary.get("entities", {}))
                + len(prompt_dictionary.get("functions", {}))
            ),
            "rows_rendered": (
                len(prompt_dictionary.get("entities", {}))
                + len(prompt_dictionary.get("functions", {}))
            ),
            "chars_before": len(dictionary_body),
            "chars_after": len(dictionary_chunk),
            "chars_removed": removed,
            "prompt_render_truncated": bool(removed),
            "render_truncated": bool(removed),
            "packet_truncated": False,
            "truncation_reason": "prompt_max_chars" if removed else "",
            "render_role": "alias_legend",
            "row_budget_exhausted": bool(removed),
            "render_order_overridden": False,
        })

    for view_name, view_data in ordered_items:
        policy = packet_view_render_policy(view_name)
        rows_total = packet_view_row_count(view_data)
        packet_truncated = (
            packet_trace_truncated
            if view_name == "trace_view"
            else _view_packet_truncated(view_data)
        )

        chunk, row_meta = _render_single_view(
            view_name=view_name,
            view_data=view_data,
            policy=policy,
            max_view_chars=max_view_chars,
        )

        remaining = max_context_chars - total_chars
        if remaining <= 0:
            chunks.append(
                "...<context truncated: global evidence budget exhausted>")
            metadata.append(_render_metadata_row(
                view_name=view_name,
                rows_total=rows_total,
                rows_rendered=0,
                chars_before=row_meta["chars_before"],
                chars_after=0,
                chars_removed=row_meta["chars_before"],
                render_truncated=True,
                packet_truncated=packet_truncated,
                reason="global_context_budget",
                policy=policy,
                original_order=original_order,
                rendered_order=rendered_order,
                row_budget_exhausted=True,
            ))
            break

        if len(chunk) > remaining:
            truncated_chunk, global_removed = _truncate_text_with_removed(
                chunk,
                remaining,
                "global evidence context",
            )
            chunks.append(truncated_chunk)
            metadata.append(_render_metadata_row(
                view_name=view_name,
                rows_total=rows_total,
                rows_rendered=row_meta["rows_rendered"],
                chars_before=row_meta["chars_before"],
                chars_after=max(0, row_meta["chars_after"] - global_removed),
                chars_removed=row_meta["chars_removed"] + global_removed,
                render_truncated=True,
                packet_truncated=packet_truncated,
                reason="global_context_budget",
                policy=policy,
                original_order=original_order,
                rendered_order=rendered_order,
                row_budget_exhausted=row_meta["row_budget_exhausted"],
            ))
            break

        chunks.append(chunk)
        metadata.append(_render_metadata_row(
            view_name=view_name,
            rows_total=rows_total,
            rows_rendered=row_meta["rows_rendered"],
            chars_before=row_meta["chars_before"],
            chars_after=row_meta["chars_after"],
            chars_removed=row_meta["chars_removed"],
            render_truncated=row_meta["render_truncated"],
            packet_truncated=packet_truncated,
            reason=row_meta["truncation_reason"],
            policy=policy,
            original_order=original_order,
            rendered_order=rendered_order,
            row_budget_exhausted=row_meta["row_budget_exhausted"],
        ))
        total_chars += len(chunk) + 2

    return "\n\n".join(chunks), metadata


def alias_prompt_value(
    value: Any,
    dictionary: Dict[str, Any],
) -> Any:
    """Replace exact address/function values with stable packet aliases."""
    address_to_alias, function_to_alias = _dictionary_reverse_maps(dictionary)

    def visit(item: Any, field_name: str = "") -> Any:
        if isinstance(item, dict):
            return {
                key: visit(child, str(key))
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [visit(child, field_name) for child in item]
        if not isinstance(item, str):
            return item
        if _FORMATTED_NUMBER_RE.fullmatch(item.strip()):
            return item.replace(",", "")
        address_alias = address_to_alias.get(item.strip().lower())
        if address_alias:
            return address_alias
        if field_name in _FUNCTION_ALIAS_FIELDS:
            function_alias = function_to_alias.get(item.strip())
            if function_alias:
                return function_alias
        return item

    return visit(value)


def _prepare_prompt_packet(
    evidence_packet: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    packet = dict(evidence_packet or {})
    raw_dictionary = packet.pop(PROMPT_DICTIONARY_KEY, {})
    packet = _compact_overlapping_prompt_views(packet)
    dictionary = (
        copy.deepcopy(raw_dictionary)
        if isinstance(raw_dictionary, dict)
        else {}
    )
    extra_values = packet.pop(PROMPT_DICTIONARY_EXTRA_VALUES_KEY, None)
    if not dictionary:
        return {
            name: alias_prompt_value(
                _project_summary_anchor_refs(name, value),
                {},
            )
            for name, value in packet.items()
        }, {}

    _merge_address_view_labels(dictionary, packet.get("address_labels"))
    usage_source = {
        "views": packet,
        "extra": extra_values,
    }
    used_addresses, used_functions = _collect_dictionary_usage(
        usage_source,
        dictionary,
    )
    legend = _prompt_dictionary_legend(
        dictionary,
        used_addresses=used_addresses,
        used_functions=used_functions,
    )
    if legend.get("entities"):
        packet.pop("address_labels", None)
    prepared = {
        name: _drop_prompt_label_duplicates(
            alias_prompt_value(
                _project_summary_anchor_refs(name, value),
                dictionary,
            )
        )
        for name, value in packet.items()
    }
    return prepared, legend


def _compact_overlapping_prompt_views(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Remove prompt-only overlap while keeping the persisted packet intact."""
    compact = copy.deepcopy(packet)

    if "transfer_event_view" in compact and isinstance(compact.get("event_view"), list):
        compact["event_view"] = [
            row
            for row in compact["event_view"]
            if not (
                isinstance(row, dict)
                and str(row.get("event") or "").strip().lower() == "transfer"
            )
        ]

    if "critical_call_view" in compact and isinstance(
        compact.get("critical_call_argument_view"), list
    ):
        compact["critical_call_argument_view"] = [
            {
                key: row.get(key)
                for key in (
                    "evidence_id",
                    "id",
                    "caller",
                    "callee",
                    "depth",
                    "parent_id",
                    "parent_function",
                    "parent_address",
                    "path_ids",
                    "function",
                    "selector",
                    "params",
                    "args_in",
                    "return_values",
                    "price_read_consumption_link",
                )
                if row.get(key) not in (None, "", [], {})
            }
            for row in compact["critical_call_argument_view"]
            if isinstance(row, dict)
        ]

    auth_view = compact.get("source_unavailable_auth_view")
    if isinstance(auth_view, dict):
        summary = dict(auth_view.get("summary") or {})
        rows = [
            row
            for row in list(auth_view.get("auth_contexts") or [])
            if isinstance(row, dict)
        ]
        if rows and not summary.get("candidate_filtered"):
            informative = [
                row
                for row in rows
                if str(
                    row.get("observable_authorization_evidence") or "unknown"
                ).lower()
                != "unknown"
            ]
            unknown = [row for row in rows if row not in informative]
            selected = informative[:8]
            selected.extend(unknown[: max(0, 4 - len(selected))])
            projected = dict(auth_view)
            projected["auth_contexts"] = selected
            summary.update({
                "prompt_projection": "informative_then_representative_unknown",
                "prompt_rows_before_projection": len(rows),
                "prompt_rows_after_projection": len(selected),
            })
            projected["summary"] = summary
            compact["source_unavailable_auth_view"] = projected

    return compact


def _dictionary_reverse_maps(
    dictionary: Dict[str, Any],
) -> tuple[Dict[str, str], Dict[str, str]]:
    address_to_alias = {
        str(address).lower(): str(alias)
        for address, alias in dict(dictionary.get("address_to_alias") or {}).items()
    }
    if not address_to_alias:
        for alias, item in dict(dictionary.get("addresses") or {}).items():
            if not isinstance(item, dict):
                continue
            address = str(item.get("address") or "").strip().lower()
            if address:
                address_to_alias[address] = str(alias)
    function_to_alias = {
        str(function): str(alias)
        for function, alias in dict(dictionary.get("function_to_alias") or {}).items()
    }
    if not function_to_alias:
        function_to_alias = {
            str(function): str(alias)
            for alias, function in dict(dictionary.get("functions") or {}).items()
        }
    return address_to_alias, function_to_alias


def _collect_dictionary_usage(
    value: Any,
    dictionary: Dict[str, Any],
) -> tuple[set[str], set[str]]:
    address_to_alias, function_to_alias = _dictionary_reverse_maps(dictionary)
    used_addresses: set[str] = set()
    used_functions: set[str] = set()

    def visit(item: Any, field_name: str = "") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, str(key))
            return
        if isinstance(item, list):
            for child in item:
                visit(child, field_name)
            return
        if not isinstance(item, str):
            return
        address_alias = address_to_alias.get(item.strip().lower())
        if address_alias:
            used_addresses.add(address_alias)
        if field_name in _FUNCTION_ALIAS_FIELDS:
            function_alias = function_to_alias.get(item.strip())
            if function_alias:
                used_functions.add(function_alias)

    visit(value)
    return used_addresses, used_functions


def _prompt_dictionary_legend(
    dictionary: Dict[str, Any],
    *,
    used_addresses: set[str],
    used_functions: set[str],
) -> Dict[str, Any]:
    addresses = dict(dictionary.get("addresses") or {})
    functions = dict(dictionary.get("functions") or {})
    entities: Dict[str, Dict[str, Any]] = {}
    for alias in sorted(used_addresses, key=_alias_sort_key):
        source = addresses.get(alias)
        if not isinstance(source, dict):
            continue
        entity = {
            "address": source.get("address"),
            "raw_label": source.get("raw_label") or source.get("label"),
        }
        sanitized_label = source.get("label")
        if (
            sanitized_label not in (None, "")
            and sanitized_label != entity.get("raw_label")
        ):
            entity["sanitized_label"] = sanitized_label
        if source.get("label_sanitized"):
            entity["label_sanitized"] = True
        entities[alias] = {
            key: value
            for key, value in entity.items()
            if value not in (None, "")
        }
    selected_functions = {
        alias: functions.get(alias)
        for alias in sorted(used_functions, key=_alias_sort_key)
        if functions.get(alias)
    }
    if not entities and not selected_functions:
        return {}
    return {
        "format": "prompt_alias_legend_v1",
        "usage": (
            "Views use A* for addresses and F* for function signatures. "
            "Evidence IDs remain canonical."
        ),
        "entities": entities,
        "functions": selected_functions,
    }


def _alias_sort_key(alias: str) -> tuple[str, int, str]:
    text = str(alias)
    suffix = text[1:]
    return text[:1], int(suffix) if suffix.isdigit() else 10**9, text


def _merge_address_view_labels(
    dictionary: Dict[str, Any],
    address_view: Any,
) -> None:
    if not isinstance(address_view, list):
        return
    address_to_alias, _ = _dictionary_reverse_maps(dictionary)
    addresses = dictionary.setdefault("addresses", {})
    for row in address_view:
        if not isinstance(row, dict):
            continue
        alias = address_to_alias.get(str(row.get("address") or "").lower())
        target = addresses.get(alias) if alias else None
        if not isinstance(target, dict):
            continue
        if row.get("raw_label") not in (None, ""):
            target["raw_label"] = row.get("raw_label")
        if row.get("sanitized_label") not in (None, ""):
            target["label"] = row.get("sanitized_label")
        if row.get("label_sanitized") is not None:
            target["label_sanitized"] = bool(row.get("label_sanitized"))


def _drop_prompt_label_duplicates(value: Any) -> Any:
    if isinstance(value, list):
        return [_drop_prompt_label_duplicates(item) for item in value]
    if not isinstance(value, dict):
        return value
    compact = {
        key: _drop_prompt_label_duplicates(item)
        for key, item in value.items()
    }
    address_fields = {
        "address",
        "caller",
        "callee",
        "sender",
        "receiver",
        "from",
        "to",
        "recipient",
        "participant",
        "beneficiary",
        "contract",
        "target_contract",
        "token",
        "pair",
        "proxy",
        "implementation",
        "admin",
        "beacon",
    }
    for field_name in address_fields:
        field_value = compact.get(field_name)
        if not (
            isinstance(field_value, str)
            and field_value.startswith("A")
            and field_value[1:].isdigit()
        ):
            continue
        for label_key in (
            f"{field_name}_label",
            f"{field_name}_label_sanitized",
        ):
            compact.pop(label_key, None)
        if field_name == "address":
            compact.pop("label", None)
            compact.pop("label_sanitized", None)
            compact.pop("address_label", None)
            compact.pop("address_label_sanitized", None)
        if field_name == "callee":
            compact.pop("address_label", None)
            compact.pop("address_label_sanitized", None)
    return compact


def _project_summary_anchor_refs(view_name: str, value: Any) -> Any:
    if view_name not in _SUMMARY_ANCHOR_VIEWS:
        return value

    def visit(item: Any) -> Any:
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return item
        projected: Dict[str, Any] = {}
        for key, child in item.items():
            if (
                isinstance(child, list)
                and (key == "evidence_ids" or key.endswith("_evidence_ids"))
                and len(child) > 7
            ):
                anchors = _diverse_anchor_refs(child, max_items=7)
                projected[key] = anchors
                projected[f"{key}_support_count"] = len(child)
                expansion = {
                    "tool": "read_packet_view",
                    "view": view_name,
                }
                if item.get("evidence_id"):
                    expansion["anchor"] = item.get("evidence_id")
                projected[f"{key}_expansion"] = expansion
            else:
                projected[key] = visit(child)
        return projected

    return visit(value)


def _diverse_anchor_refs(values: List[Any], *, max_items: int) -> List[Any]:
    unique: List[Any] = []
    seen_values = set()
    for value in values:
        marker = stable_json_dumps(value)
        if marker in seen_values:
            continue
        seen_values.add(marker)
        unique.append(value)
    selected: List[Any] = []
    seen_kinds = set()
    for value in unique:
        kind = str(value).split(":", 1)[0]
        if kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        selected.append(value)
        if len(selected) >= max_items:
            return selected
    for value in unique:
        if value in selected:
            continue
        selected.append(value)
        if len(selected) >= max_items:
            break
    return selected


def truncate_text(text: str, max_chars: int, label: str) -> str:
    truncated, _ = _truncate_text_with_removed(text, max_chars, label)
    return truncated


def truncate_text_with_removed(text: str, max_chars: int, label: str) -> tuple[str, int]:
    return _truncate_text_with_removed(text, max_chars, label)


def _ordered_view_items(evidence_packet: Dict[str, Any]) -> List[Tuple[str, Any]]:
    indexed = list(enumerate(evidence_packet.items()))
    indexed.sort(
        key=lambda item: (
            int(packet_view_render_policy(item[1][0]).get("priority", 50)),
            item[0],
        )
    )
    return [item for _, item in indexed]


def _render_single_view(
    *,
    view_name: str,
    view_data: Any,
    policy: Dict[str, Any],
    max_view_chars: int,
) -> tuple[str, Dict[str, Any]]:
    policy_max_chars = max(0, int(policy.get("max_prompt_chars", 0) or 0))
    if policy_max_chars:
        max_view_chars = min(max_view_chars, policy_max_chars)
    if isinstance(view_data, dict) and view_data.get("available") is False:
        chunk = (
            f"=== {view_name} ===\n"
            f"(Not available: {view_data.get('reason', 'unknown')})"
        )
        return chunk, _row_meta(
            chars_before=len(chunk),
            chars_after=len(chunk),
            chars_removed=0,
            render_truncated=False,
            truncation_reason="",
            rows_rendered=0,
            row_budget_exhausted=False,
        )

    if view_data == [] or view_data == {}:
        chunk = (
            f"=== {view_name} ===\n"
            "(View is available but empty: no records were found.)"
        )
        return chunk, _row_meta(
            chars_before=len(chunk),
            chars_after=len(chunk),
            chars_removed=0,
            render_truncated=False,
            truncation_reason="",
            rows_rendered=0,
            row_budget_exhausted=False,
        )

    body_before = stable_json_dumps(view_data)
    chars_before = len(body_before)
    render_mode = str(policy.get("default_render_mode") or "top_rows")

    if render_mode == "full":
        body, removed = _truncate_text_with_removed(
            body_before,
            max_view_chars,
            f"view {view_name}",
        )
        chunk = f"=== {view_name} ===\n{body}"
        return chunk, _row_meta(
            chars_before=chars_before,
            chars_after=max(0, chars_before - removed),
            chars_removed=removed,
            render_truncated=removed > 0,
            truncation_reason="prompt_max_chars" if removed > 0 else "",
            rows_rendered=packet_view_row_count(view_data),
            row_budget_exhausted=removed > 0,
        )

    summary, rows = _split_summary_and_rows(view_data)
    if not rows:
        body, removed = _truncate_text_with_removed(
            body_before,
            max_view_chars,
            f"view {view_name}",
        )
        chunk = f"=== {view_name} ===\n{body}"
        return chunk, _row_meta(
            chars_before=chars_before,
            chars_after=max(0, chars_before - removed),
            chars_removed=removed,
            render_truncated=removed > 0,
            truncation_reason="prompt_max_chars" if removed > 0 else "",
            rows_rendered=packet_view_row_count(view_data),
            row_budget_exhausted=removed > 0,
        )

    prompt_rows = _prepare_prompt_rows(view_name, rows)
    max_prompt_rows = max(0, int(policy.get("max_prompt_rows", 0) or 0))
    row_limit_exhausted = bool(
        max_prompt_rows and len(prompt_rows) > max_prompt_rows
    )
    if row_limit_exhausted:
        prompt_rows = prompt_rows[:max_prompt_rows]
    rendered_rows, row_budget_exhausted = _render_rows_with_budget(
        prompt_rows,
        view_name=view_name,
        summary=summary,
        max_chars=max_view_chars,
    )
    row_budget_exhausted = row_budget_exhausted or row_limit_exhausted
    payload = {
        "render_note": (
            "Budget-aware row rendering. If more rows/details are needed, "
            "request read_packet_view for this view."
        ),
        "rows_total": len(rows),
        "rows_rendered": len(rendered_rows),
        "summary": summary,
        "rows": rendered_rows,
    }
    if prompt_rows != rows:
        payload["row_render_note"] = (
            "Nested row detail was compacted for the first prompt. Request "
            "read_packet_view or local context for full evidence rows."
        )
    if row_budget_exhausted:
        payload["render_truncated"] = True
        payload["truncation_reason"] = (
            "prompt_row_limit" if row_limit_exhausted else "prompt_max_chars"
        )
    body = stable_json_dumps(payload)
    body, removed = _truncate_text_with_removed(
        body,
        max_view_chars,
        f"view {view_name}",
    )
    row_budget_exhausted = row_budget_exhausted or removed > 0
    chunk = f"=== {view_name} ===\n{body}"
    chars_after = len(body)
    chars_removed = max(0, chars_before - chars_after)
    return chunk, _row_meta(
        chars_before=chars_before,
        chars_after=chars_after,
        chars_removed=chars_removed,
        render_truncated=row_budget_exhausted or chars_removed > 0,
        truncation_reason=(
            "prompt_row_limit"
            if row_limit_exhausted
            else "prompt_max_chars"
            if row_budget_exhausted or chars_removed > 0
            else ""
        ),
        rows_rendered=len(rendered_rows),
        row_budget_exhausted=row_budget_exhausted,
    )


def _split_summary_and_rows(view_data: Any) -> tuple[Any, List[Any]]:
    if isinstance(view_data, list):
        return {"list_view": True, "row_count": len(view_data)}, list(view_data)

    if not isinstance(view_data, dict):
        return {}, []

    for key in _ROW_KEYS:
        value = view_data.get(key)
        if isinstance(value, list):
            summary = {
                k: v
                for k, v in view_data.items()
                if k != key and k not in _SUMMARY_SKIP_KEYS
            }
            summary["row_key"] = key
            summary["row_count"] = len(value)
            return _compact_summary(summary), list(value)
    return {}, []


def _prepare_prompt_rows(view_name: str, rows: List[Any]) -> List[Any]:
    """Keep heavy nested evidence rows useful in the first judge prompt."""
    if view_name == "critical_call_view":
        indexed = list(enumerate(rows))
        indexed.sort(
            key=lambda item: (
                -_critical_prompt_row_score(item[1]),
                item[0],
            )
        )
        rows = [row for _, row in indexed]

    if view_name == "semantic_state_delta_view":
        semantic_rows = [
            row
            for row in rows
            if not (
                isinstance(row, dict)
                and str(row.get("semantic_confidence") or "").lower() == "low"
                and str(row.get("variable_hint") or "").lower() == "raw_slot_only"
            )
        ]
        raw_slot_rows = [row for row in rows if row not in semantic_rows]
        # Full raw-slot rows remain available through read_packet_view. The
        # first prompt keeps decoded semantics plus a small fallback preview.
        rows = (
            semantic_rows + raw_slot_rows[:8]
            if semantic_rows
            else raw_slot_rows[:16]
        )

    if view_name in _LARGE_PROMPT_ROW_FIELDS:
        return [_project_large_prompt_row(view_name, row) for row in rows]
    if view_name != "reentrancy_state_order_view":
        return rows

    compact_rows: List[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            compact_rows.append(row)
            continue
        compact = dict(row)
        state_order = list(compact.get("state_order", []) or [])
        if state_order:
            compact["state_order_total"] = len(state_order)
            compact["state_order_preview"] = state_order[:4]
            compact.pop("state_order", None)
            compact["state_order_render_note"] = (
                "Preview only in the first prompt; fetch this row/view for "
                "full state access ordering."
            )
        slot_summary = list(compact.get("slot_access_summary", []) or [])
        if slot_summary:
            compact["slot_access_summary_total"] = len(slot_summary)
            compact["slot_access_summary"] = slot_summary[:4]
        state_ids = list(compact.get("state_order_evidence_ids", []) or [])
        if state_ids:
            compact["state_order_evidence_id_count"] = len(state_ids)
            compact["state_order_evidence_ids"] = state_ids[:8]
        compact_rows.append(compact)
    return compact_rows


def _critical_prompt_row_score(row: Any) -> int:
    if not isinstance(row, dict):
        return -100
    function = str(row.get("function") or "").lower()
    reason = str(row.get("why_included") or "").lower()
    call_type = str(row.get("call_type") or row.get("type") or "").lower()
    score = 0
    if "reentrant_call" in reason:
        score += 100
    shape_kind = str(row.get("structural_reentry_kind") or "")
    if shape_kind == "read_only_nested_shape":
        score -= 150
    elif shape_kind == "standard_callback_shape":
        score -= 90
    elif shape_kind == "stateful_reentry_lead":
        score += 40
    for keyword in (
        "withdraw",
        "redeem",
        "borrow",
        "repay",
        "liquidat",
        "flash",
        "callback",
        "mint",
        "burn",
        "claim",
        "harvest",
        "transferfrom",
        "approve",
        "exchange",
        "swap",
        "add_liquidity",
        "remove_liquidity",
        "upgrade",
        "admin",
        "set_rate",
    ):
        if keyword in function:
            score += 35
            break
    if "unknown selector" in reason:
        score += 30
    if call_type == "call":
        score += 10
    if call_type == "staticcall" or any(
        keyword in function
        for keyword in (
            "balanceof",
            "totalsupply",
            "latestanswer",
            "latestrounddata",
            "price_oracle",
        )
    ):
        score -= 25
    if reason == "call_type: delegatecall":
        score -= 10
    return score


def _project_large_prompt_row(view_name: str, row: Any) -> Any:
    """Render a locating projection for broad/high-cost first-pass rows."""
    if not isinstance(row, dict):
        return row
    fields = _LARGE_PROMPT_ROW_FIELDS.get(view_name)
    if not fields:
        return row
    projected: Dict[str, Any] = {}
    for key in fields:
        if key not in row:
            continue
        projected[key] = _preview_prompt_value(row.get(key))
    if len(projected) < len(row):
        projected["first_prompt_projection"] = (
            "Compact row projection; use read_packet_view or local context "
            "for full row detail."
        )
    return projected


def _preview_prompt_value(value: Any, *, max_items: int = 6, max_chars: int = 700) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + f"...<truncated {len(value) - max_chars} chars>"
    if isinstance(value, list):
        preview = [_preview_prompt_value(
            item, max_items=4, max_chars=300) for item in value[:max_items]]
        if len(value) > max_items:
            return {
                "preview": preview,
                "items_total": len(value),
                "items_omitted": len(value) - max_items,
            }
        return preview
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, item in list(value.items())[:max_items]:
            compact[str(key)] = _preview_prompt_value(
                item,
                max_items=4,
                max_chars=300,
            )
        if len(value) > max_items:
            compact["fields_omitted"] = len(value) - max_items
        return compact
    return value


def _compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, dict):
            dumped = stable_json_dumps(value)
            compact[key] = value if len(dumped) <= 4000 else {
                "summary_truncated": True}
        elif isinstance(value, list):
            compact[key] = value[:8]
            if len(value) > 8:
                compact[f"{key}_omitted"] = len(value) - 8
    return compact


def _render_rows_with_budget(
    rows: List[Any],
    *,
    view_name: str,
    summary: Any,
    max_chars: int,
) -> tuple[List[Any], bool]:
    rendered: List[Any] = []
    base_payload = {
        "render_note": "",
        "rows_total": len(rows),
        "rows_rendered": 0,
        "summary": summary,
        "rows": [],
    }
    overhead = len(stable_json_dumps(base_payload)) + 1000
    row_budget = max(1000, max_chars - overhead)
    used = 0
    for row in rows:
        compact_row = _compact_row(row)
        size = len(stable_json_dumps(compact_row))
        if rendered and used + size > row_budget:
            return rendered, True
        rendered.append(compact_row)
        used += size
    return rendered, len(rendered) < len(rows)


def _compact_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    compact: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > 2000:
            compact[key] = value[:2000] + \
                f"...<truncated {len(value) - 2000} chars>"
        elif isinstance(value, list) and len(value) > 20:
            compact[key] = value[:20]
            compact[f"{key}_omitted"] = len(value) - 20
        elif isinstance(value, dict):
            dumped = stable_json_dumps(value)
            if len(dumped) > 3000:
                compact[key] = {"summary_truncated": True,
                                "chars_before": len(dumped)}
            else:
                compact[key] = value
        else:
            compact[key] = value
    return compact


def _row_meta(
    *,
    chars_before: int,
    chars_after: int,
    chars_removed: int,
    render_truncated: bool,
    truncation_reason: str,
    rows_rendered: int,
    row_budget_exhausted: bool,
) -> Dict[str, Any]:
    return {
        "chars_before": chars_before,
        "chars_after": chars_after,
        "chars_removed": chars_removed,
        "render_truncated": render_truncated,
        "truncation_reason": truncation_reason,
        "rows_rendered": rows_rendered,
        "row_budget_exhausted": row_budget_exhausted,
    }


def _render_metadata_row(
    *,
    view_name: str,
    rows_total: int,
    rows_rendered: int,
    chars_before: int,
    chars_after: int,
    chars_removed: int,
    render_truncated: bool,
    packet_truncated: bool,
    reason: str,
    policy: Dict[str, Any],
    original_order: Iterable[str],
    rendered_order: Iterable[str],
    row_budget_exhausted: bool,
) -> Dict[str, Any]:
    original_order = list(original_order)
    rendered_order = list(rendered_order)
    return {
        "view": view_name,
        "rows_total": rows_total,
        "rows_rendered": max(0, rows_rendered),
        "chars_before": max(0, chars_before),
        "chars_after": max(0, chars_after),
        "chars_removed": max(0, chars_removed),
        "prompt_render_truncated": bool(render_truncated),
        "render_truncated": bool(render_truncated),
        "packet_trace_truncated": bool(packet_truncated) if view_name == "trace_view" else False,
        "packet_truncated": bool(packet_truncated),
        "truncation_reason": reason,
        "render_priority": policy.get("priority"),
        "render_role": policy.get("role"),
        "render_cost": policy.get("cost"),
        "render_mode": policy.get("default_render_mode"),
        "row_budget_exhausted": bool(row_budget_exhausted),
        "render_order_overridden": original_order != rendered_order,
    }


def _packet_trace_truncated(adequacy: Dict[str, Any]) -> bool:
    trace = adequacy.get("trace", {}) if isinstance(adequacy, dict) else {}
    return bool(trace.get("truncated"))


def _view_packet_truncated(view_data: Any) -> bool:
    if isinstance(view_data, dict):
        truncated = view_data.get("truncated")
        if isinstance(truncated, dict):
            return bool(
                truncated.get("omitted")
                or truncated.get("total", 0) > truncated.get("shown", 0)
            )
        return bool(truncated)
    return False


def _truncate_text_with_removed(text: str, max_chars: int, label: str) -> tuple[str, int]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, 0
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n...<truncated {omitted} chars from {label}>", omitted
