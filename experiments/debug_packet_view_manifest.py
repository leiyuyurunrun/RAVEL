"""Validate packet view manifest, dependency expansion, and build metadata."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.runtime.packet_builder import (
    DEFAULT_PACKET_VIEWS,
    JUDGE_VIEW_PROFILES,
    build_or_load_packet,
)
from evotx.runtime.packet_view_manifest import (
    VIEW_DEPENDENCIES,
    VIEW_MANIFEST,
    VIEW_MANIFEST_VERSION,
    VIEW_ORDER,
    filter_views_for_prompt_policy,
    resolve_view_dependency_closure,
    summarize_packet_view_costs,
)


REQUIRED_MANIFEST_FIELDS = {
    "tier",
    "cost",
    "cost_score",
    "default_usage",
    "prompt_role",
    "allow_first_pass",
    "allow_followup",
    "empty_policy",
}


def main() -> None:
    _assert_manifest_coverage()
    _assert_dependency_closure()
    _assert_cost_summary()
    _assert_prompt_policy()
    _assert_build_config()
    print(json.dumps({
        "ok": True,
        "view_manifest_version": VIEW_MANIFEST_VERSION,
        "manifest_view_count": len(VIEW_MANIFEST),
    }, indent=2, ensure_ascii=False))


def _assert_manifest_coverage() -> None:
    assert DEFAULT_PACKET_VIEWS == VIEW_ORDER, "DEFAULT_PACKET_VIEWS and VIEW_ORDER diverged"

    for view in DEFAULT_PACKET_VIEWS:
        assert view in VIEW_MANIFEST, f"missing manifest entry for default view: {view}"
        missing = REQUIRED_MANIFEST_FIELDS - set(VIEW_MANIFEST[view])
        assert not missing, f"manifest entry {view} missing fields: {sorted(missing)}"

    for profile, views in JUDGE_VIEW_PROFILES.items():
        for view in views:
            assert view in VIEW_MANIFEST, f"profile {profile} references unknown view: {view}"

    for view, deps in VIEW_DEPENDENCIES.items():
        assert view in VIEW_MANIFEST, f"dependency key references unknown view: {view}"
        for dep in list(deps.get("required", []) or []) + list(deps.get("optional", []) or []):
            assert dep in VIEW_MANIFEST, f"{view} dependency references unknown view: {dep}"


def _assert_dependency_closure() -> None:
    closure = resolve_view_dependency_closure(["classification_digest_view"])
    for expected in (
        "operation_summary_view",
        "critical_call_view",
        "value_release_view",
        "participant_net_delta_view",
        "semantic_state_delta_view",
        "classification_digest_view",
    ):
        assert expected in closure, f"classification digest closure missing {expected}"
    assert "trace_view" not in closure, "heavy optional trace_view should not be auto-expanded"
    assert "state_change_view" not in closure, "heavy optional state_change_view should not be auto-expanded"

    expected_order = [view for view in DEFAULT_PACKET_VIEWS if view in set(closure)]
    assert closure == expected_order, "dependency closure order must follow DEFAULT_PACKET_VIEWS"


def _assert_cost_summary() -> None:
    summary = summarize_packet_view_costs({
        "tx_card": {"transaction_hash": "0xabc"},
        "trace_view": [{"id": 1}],
        "state_change_view": [{"id": 2}],
        "reentrancy_state_order_summary_view": {"rows": []},
        "profit_loss_view": {"summary": {}},
    })
    assert summary["cost_score_total"] > 0
    assert "trace_view" in summary["heavy_views_present"]
    assert "state_change_view" in summary["heavy_views_present"]
    assert "profit_loss_view" in summary["hint_only_views_present"]
    assert "reentrancy_state_order_summary_view" in summary["empty_views"]


def _assert_prompt_policy() -> None:
    views = [
        "tx_card",
        "classification_digest_view",
        "trace_view",
        "state_change_view",
        "profit_loss_view",
        "critical_call_view",
    ]
    first_pass = filter_views_for_prompt_policy(views, usage="first_pass")
    assert "trace_view" not in first_pass
    assert "state_change_view" not in first_pass
    assert "profit_loss_view" not in first_pass
    assert "critical_call_view" in first_pass

    followup_without_heavy = filter_views_for_prompt_policy(views, usage="followup")
    followup_with_heavy = filter_views_for_prompt_policy(
        views,
        usage="followup",
        allow_heavy=True,
    )
    assert "trace_view" not in followup_without_heavy
    assert "trace_view" in followup_with_heavy
    assert set(filter_views_for_prompt_policy(views, usage="debug")) == set(views)


def _assert_build_config() -> None:
    tx = "0x" + "1" * 64
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        syn_dir = base / "synthesized"
        syn_dir.mkdir(parents=True)
        (syn_dir / f"{tx}_synthesized.json").write_text(
            json.dumps(_minimal_synthesized(tx), ensure_ascii=False),
            encoding="utf-8",
        )
        packet = build_or_load_packet(
            tx_hash=tx,
            base_dir=base,
            force_rebuild=True,
            packet_profile="minimal",
            required_views=["classification_digest_view"],
            write_evidence_store=False,
        )
    build_config = dict(packet.get("build_config") or {})
    assert build_config.get("requested_required_views") == ["classification_digest_view"]
    assert build_config.get("view_manifest_version") == VIEW_MANIFEST_VERSION
    assert "classification_digest_view" in build_config.get("resolved_include_views", [])
    assert "critical_call_view" in build_config.get("dependency_expanded_views", [])
    assert "view_cost_summary" not in build_config
    assert isinstance(packet.get("view_cost_summary"), dict)


def _minimal_synthesized(tx: str) -> Dict[str, Any]:
    return {
        "transaction_hash": tx,
        "hash": tx,
        "chain": "debug",
        "status": 1,
        "from": "0x0000000000000000000000000000000000000001",
        "to": "0x0000000000000000000000000000000000000002",
        "value": "0",
        "gas_used": "21000",
        "trace": {},
        "token_info": {},
        "label_address_map": {},
    }


if __name__ == "__main__":
    main()
