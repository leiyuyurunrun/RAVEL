"""Build and inspect an EvoTx packet without running any LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.runtime.packet_builder import build_or_load_packet, resolve_input_paths
from evotx.runtime.packet_view_registry import PacketViewRegistry
from evotx.runtime.packet_view_manifest import VIEW_MANIFEST_VERSION, manifest_summary_rows
from evotx.runtime.view_catalog import PACKET_VIEW_CATALOG, packet_view_row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a packet and print compact view diagnostics."
    )
    parser.add_argument("--tx", help="Transaction hash")
    parser.add_argument("--base-dir", default="data/cache", help="Base cache directory")
    parser.add_argument("--force", action="store_true", help="Force packet rebuild")
    parser.add_argument("--profile", choices=["full", "judge", "minimal"], default="full")
    parser.add_argument("--label", help="Attack label used with --profile judge")
    parser.add_argument("--include-view", action="append", dest="include_views")
    parser.add_argument("--no-evidence-store", action="store_true")
    parser.add_argument("--no-view-dependencies", action="store_true")
    parser.add_argument("--include-heavy-dependencies", action="store_true")
    parser.add_argument("--print-view-manifest", action="store_true")
    args = parser.parse_args()

    if args.print_view_manifest:
        print_json({
            "view_manifest_version": VIEW_MANIFEST_VERSION,
            "views": manifest_summary_rows(),
        })
        if not args.tx:
            return

    if not args.tx:
        parser.error("--tx is required unless --print-view-manifest is used alone")

    packet = build_or_load_packet(
        tx_hash=args.tx,
        base_dir=args.base_dir,
        force_rebuild=args.force,
        include_views=args.include_views,
        attack_label=args.label,
        packet_profile=args.profile,
        write_evidence_store=not args.no_evidence_store,
        include_view_dependencies=not args.no_view_dependencies,
        include_heavy_optional_dependencies=bool(args.include_heavy_dependencies),
    )
    packet_path = resolve_input_paths(args.tx, args.base_dir)["packet"]
    views = packet.get("views", {})
    registry = PacketViewRegistry(packet, debug_allow_list_views=True)
    list_result = registry.call("list_packet_views", {})

    print(f"packet_path: {packet_path}")
    print(f"packet_format: {packet.get('packet_format')}")
    print(f"packet_profile: {packet.get('packet_profile')}")
    print("build_config:")
    print_json(packet.get("build_config", {}))
    print("view_cost_summary:")
    print_json(packet.get("view_cost_summary", {}))
    print("evidence_store:")
    print_json(packet.get("evidence_store", {}))
    print("dictionary summary:")
    dictionary = packet.get("dictionary", {}) if isinstance(packet, dict) else {}
    print_json({
        "format": dictionary.get("format"),
        "address_count": len(dictionary.get("addresses", {}) or {}),
        "function_count": len(dictionary.get("functions", {}) or {}),
    })
    print("views:")
    for row in list_result.get("evidence", {}).get("views", []):
        name = row.get("view")
        catalog_entry = PACKET_VIEW_CATALOG.get(str(name), {})
        print(
            f"  - {name}: {row.get('row_count')} rows "
            f"| tier={row.get('tier')} cost={row.get('cost')} "
            f"| {catalog_entry.get('description', '')}"
        )

    print("\nevidence_adequacy_view:")
    print_json(views.get("evidence_adequacy_view", {}))

    for name in (
        "critical_call_view",
        "classification_digest_view",
        "reentrancy_state_order_summary_view",
        "reentrancy_state_order_view",
        "unknown_selector_view",
        "participant_net_delta_view",
        "value_release_view",
        "flash_or_atomic_capital_view",
        "beneficiary_controller_view",
    ):
        print(f"\n{name} first 5:")
        sample = views.get(name, [])
        if isinstance(sample, list):
            print_json(sample[:5])
        elif isinstance(sample, dict) and isinstance(sample.get("rows"), list):
            print_json({**sample, "rows": sample["rows"][:5]})
        else:
            print_json(sample)


def view_count(view: Any) -> str:
    count = packet_view_row_count(view)
    return f"{count} rows"


def print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
