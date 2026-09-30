"""Compare full/judge/minimal packet profiles without running any LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.runtime.packet_builder import (
    PACKET_FORMAT_VERSION,
    build_compact_evidence_packet,
    build_evidence_store,
    build_packet_dictionary,
    build_trace_index,
    dump_json,
    load_json,
    resolve_input_paths,
)
from evotx.runtime.packet_view_registry import PacketViewRegistry
from evotx.runtime.view_catalog import packet_view_row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare EvoTx packet profile sizes and compact-view checks."
    )
    parser.add_argument("--tx", required=True, help="Transaction hash")
    parser.add_argument("--base-dir", default="data/cache")
    parser.add_argument("--label", default="", help="Attack label for judge profile")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["full", "judge", "minimal"],
        choices=["full", "judge", "minimal"],
    )
    parser.add_argument(
        "--write-evidence-store",
        action="store_true",
        help="Write evidence_store JSON so PacketViewRegistry lookup can be tested.",
    )
    args = parser.parse_args()

    paths = resolve_input_paths(args.tx, args.base_dir)
    if not paths.get("synthesized"):
        raise FileNotFoundError(f"No synthesized JSON found for tx={args.tx}")

    syn = load_json(paths["synthesized"])
    fundflow_obj = load_json(paths["fundflow"]) if paths.get("fundflow") else None
    profit_loss_obj = load_json(paths["profit_loss"]) if paths.get("profit_loss") else None
    trace_index = build_trace_index(syn.get("trace", {}))
    dictionary = build_packet_dictionary(syn, trace_index)
    evidence_store = build_evidence_store(syn, trace_index)

    packets: Dict[str, Dict[str, Any]] = {}
    for profile in args.profiles:
        views = build_compact_evidence_packet(
            syn=syn,
            fundflow_obj=fundflow_obj,
            profit_loss_obj=profit_loss_obj,
            attack_label=args.label,
            packet_profile=profile,
        )
        packets[profile] = {
            "packet_format": PACKET_FORMAT_VERSION,
            "transaction_hash": str(args.tx).lower(),
            "packet_profile": profile,
            "dictionary": dictionary,
            "evidence_store": {
                "path": str(paths.get("evidence_store") or ""),
                "format": evidence_store.get("format"),
                "count": len(evidence_store.get("evidence", {}) or {}),
            },
            "views": views,
        }

    if args.write_evidence_store and paths.get("evidence_store"):
        dump_json(evidence_store, paths["evidence_store"])

    profile_rows = []
    for profile, packet in packets.items():
        views = packet.get("views", {})
        profile_rows.append({
            "profile": profile,
            "packet_bytes": _json_size(packet),
            "view_count": len(views),
            "views": {
                name: packet_view_row_count(view)
                for name, view in sorted(views.items())
            },
            "checks": _profile_checks(packet),
        })

    lookup_result: Dict[str, Any] = {}
    if args.write_evidence_store:
        sample_id = _first_evidence_id(evidence_store.get("evidence", {}).keys())
        if sample_id and packets:
            packet = packets.get("judge") or next(iter(packets.values()))
            registry = PacketViewRegistry(packet)
            lookup_result = registry.call(
                "read_evidence_by_id",
                {"evidence_ids": [sample_id], "limit": 1},
            ).get("summary", {})

    print(json.dumps({
        "tx": str(args.tx).lower(),
        "label": args.label,
        "source_paths": {
            key: str(value) if value else None
            for key, value in paths.items()
        },
        "evidence_store_count": len(evidence_store.get("evidence", {}) or {}),
        "dictionary": {
            "address_count": len(dictionary.get("addresses", {}) or {}),
            "function_count": len(dictionary.get("functions", {}) or {}),
        },
        "profiles": profile_rows,
        "evidence_store_lookup_summary": lookup_result,
    }, ensure_ascii=False, indent=2))


def _json_size(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _profile_checks(packet: Dict[str, Any]) -> Dict[str, Any]:
    views = packet.get("views", {})
    summary_view = views.get("reentrancy_state_order_summary_view", {})
    summary_rows = _rows(summary_view)
    digest = views.get("classification_digest_view", {})
    return {
        "has_classification_digest_view": bool(digest),
        "classification_digest_signal_count": len(digest.get("top_signals", []) or [])
        if isinstance(digest, dict) else 0,
        "has_reentrancy_state_order_summary_view": bool(summary_view),
        "summary_rows": len(summary_rows),
        "summary_rows_inline_state_order": sum(
            1 for row in summary_rows if isinstance(row, dict) and row.get("state_order")
        ),
        "summary_state_order_evidence_ids": sum(
            len(row.get("state_order_evidence_ids", []) or [])
            for row in summary_rows if isinstance(row, dict)
        ),
    }


def _rows(view: Any) -> list[Dict[str, Any]]:
    if isinstance(view, dict) and isinstance(view.get("rows"), list):
        return [row for row in view["rows"] if isinstance(row, dict)]
    if isinstance(view, list):
        return [row for row in view if isinstance(row, dict)]
    return []


def _first_evidence_id(values: Iterable[Any]) -> str:
    for value in values:
        text = str(value or "")
        if text:
            return text
    return ""


if __name__ == "__main__":
    main()
