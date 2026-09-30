from __future__ import annotations

from typing import Any, Dict, Iterable

from evotx.runtime.packet_view_manifest import VIEW_MANIFEST


PACKET_VIEW_CATALOG: Dict[str, Dict[str, str]] = {
    "tx_card": {
        "description": "Basic transaction metadata: sender, receiver, status, chain, gas, block, transaction counts.",
        "default_use": "Identify transaction-level context and sender/receiver roles.",
    },
    "address_labels": {
        "description": "Sanitized address labels and roles. Leakage labels such as exploiter, attacker, hacker, victim should be removed or sanitized.",
        "default_use": "Check whether actors are known protocol components, routers, governance, multisig, keeper, pool, or ordinary EOAs.",
    },
    "token_info": {
        "description": "Token metadata such as address, symbol, name, and decimals.",
        "default_use": "Interpret token amounts, normalized deltas, and asset identities when available.",
    },
    "evidence_adequacy_view": {
        "description": "Metadata about packet completeness, trace truncation, decode coverage, unknown selectors, and state semantic coverage.",
        "default_use": "Determine whether missing evidence is due to absent behavior or incomplete packet coverage.",
    },
    "operation_summary_view": {
        "description": "Neutral operation/signal presence summary derived from the full trace and fundflow input: real entry function/selector and decode status, boolean flags for swap/sync/mint/burn/borrow/repay/liquidation/oracle/AMM reserve transition/flash capital/value release/price-dependent operations, total-versus-shown counts, and packet completeness. This is the canonical signal-presence view; there is no separate signal_presence_view. Absence flags are negative signals, not attack judgments.",
        "default_use": "Primary view for determining whether a class of operation or signal exists in the transaction. Use absence flags to answer false instead of uncertain when packet is complete.",
    },
    "classification_digest_view": {
        "description": "Neutral cross-view signal digest with top evidence signals, candidate mechanism hints, evidence ids, and anti-overfit notes. It aggregates packet evidence but does not assign a final attack label.",
        "default_use": "Use as a compact first-pass orientation view before reading specialized views or follow-up evidence.",
    },
    "trace_outline_view": {
        "description": "Compact call-tree outline with id, type, depth, parent_id, function, selector, caller/callee labels, and value only. No large params/args/return_values/state_change blobs.",
        "default_use": "Understand overall execution structure without the bulk of full trace_view. Prefer over trace_view for initial judge context.",
    },
    "trace_view": {
        "description": "Compact execution trace rows including calls, depth, caller/callee, function names, selectors, args, return values, and limited context.",
        "default_use": "Inspect broader execution flow when critical_call_view is insufficient.",
    },
    "critical_call_view": {
        "description": "Task-oriented subset of important calls, including flash loans, swaps, mint, burn, borrow, repay, redeem, withdraw, liquidate, oracle reads, balanceOf, totalSupply, delegatecall, create, callbacks, and structurally detected reentrant calls (calls that re-enter a contract already active in the call stack). Reentrant calls are always included regardless of trace size or truncation.",
        "default_use": "Primary view for identifying value-sensitive operations, atomic capital patterns, and reentrancy/callback patterns. Rows with why_included='reentrant_call' indicate structural reentrancy detected from the full trace.",
    },
    "critical_call_argument_view": {
        "description": "Decoded params, args_in, and return values for calls already selected by critical_call_view, with evidence id, caller, callee, function, selector, and parent context. It exposes actor, beneficiary, target, amount, signature-scope, and callback arguments when decoded without assigning attack semantics.",
        "default_use": "Check whether caller, signer, on-behalf actor, receiver, beneficiary, target, operator, spender, nonce/deadline, or amount parameters align with the claimed authorization or entitlement path.",
    },
    "source_unavailable_auth_view": {
        "description": "Candidate-indexable authorization context built before judging for source-unavailable or selector-uncertain access-control cases. Rows bind sensitive/unknown call paths to nearby state summaries, owner/role/proxy hints, and observable authorization status. An unfiltered all-unknown view is summarized; candidate-filtered follow-up retains matching call anchors.",
        "default_use": "Use with C1 candidate evidence_ids/path_ids to project only relevant rows for C2 authorization-gap judging. Interpret statuses as observable context only; absent/weak rows do not prove that no require or branch check exists.",
    },
    "reentrancy_state_order_view": {
        "description": "Detailed neutral phase-aware ordering evidence around structurally detected reentrant calls. It binds outer call, external edge, re-entry call, callback kind, logical storage context, phase boundaries, same-contract slot witnesses, and optional per-access rows.",
        "default_use": "Follow-up view when a candidate-specific state-order question needs raw access detail after the compact summary locates the path.",
    },
    "reentrancy_state_order_summary_view": {
        "description": "Compact first-pass phase-aware summary for nested re-entry shapes. It exposes candidate id, outer/external-edge/re-entry boundaries and call types, callback kind, logical storage context, phase state counts, same-contract slot order patterns, and whether the row passed the formal-candidate gate. Read-only recursion and standard callbacks without an effect link remain structural_shape_only.",
        "default_use": "Primary first-pass view for candidate-local reentrancy ordering. Prefer formal_candidate rows; do not treat structural_shape_only, read-only recursion, or a standard flash/swap callback alone as an exploit. Treat state_updated_before_external_edge as safe-order evidence; request detail only when phases remain unresolved.",
    },
    "reentrancy_candidate_catalog_view": {
        "description": "Small ranked catalog of deterministic reentrancy candidate IDs. Rows bind outer/external-edge/re-entry calls and summarize formal-candidate gating, candidate-local value release, repeated release, phase patterns, and stable anchor evidence IDs without raw state payloads.",
        "default_use": "Use first in Reentrancy C1. Select only formal_candidate rows and candidate_id values present in this catalog; structural_shape_only rows need new effect-specific evidence before selection. Keep up to five plausible candidates, then request candidate-filtered detail downstream.",
    },
    "unknown_selector_view": {
        "description": "Undecoded or weakly decoded calls with selector, caller/callee, children summary, nearby events, and nearby state changes.",
        "default_use": "Inspect potentially important calls whose function semantics are not decoded.",
    },
    "event_view": {
        "description": "Decoded events/logs such as Transfer, Swap, Sync, Mint, Burn, Deposit, Withdraw, Approval, Liquidate, Reward.",
        "default_use": "Verify protocol operations and token/event-level effects.",
    },
    "state_change_view": {
        "description": "Raw storage and state-change rows, including sload/sstore, slot keys, previous/current values, and embedded state_change metadata.",
        "default_use": "Inspect raw state transitions when semantic views are insufficient.",
    },
    "semantic_state_delta_view": {
        "description": "Decoded or semi-decoded state changes such as balances, totalSupply, reserves, mappings, accounting variables, key addresses, deltas, and normalized amounts when available.",
        "default_use": "Understand storage changes without relying only on raw slot IDs.",
    },
    "token_semantic_delta_summary_view": {
        "description": "Compact token-level semantic anomaly candidates derived from transfer events and semantic state deltas. Rows bind a token contract, mechanism hint, event ids, state ids, affected addresses, and whether the evidence appears token-contract-origin.",
        "default_use": "Primary compact first-pass view for token semantic exploitation C1-style anomaly anchoring. Use before broad semantic_state_delta_view when judging fee-on-transfer, reflection, rebase, locked-balance, hook, supply, or event/balance mismatch evidence.",
    },
    "token_accounting_origin_view": {
        "description": "Compact origin classifier for accounting divergence. It separates token-contract-origin semantic behavior from protocol-internal ledger/reentrancy/accounting divergence and provides evidence ids for the distinction.",
        "default_use": "Use to prevent confusing protocol accounting or reentrancy ledger artifacts with token semantic exploitation. Downstream reliance/outcome conditions should stay anchored to token-level origin candidates.",
    },
    "price_relevant_state_view": {
        "description": "Subset of state changes related to reserves, balances, price, oracle, exchange rate, virtual price, share price, collateral, debt, liquidity, tick, sqrtPrice, fee growth, or kLast.",
        "default_use": "Primary view for price-source perturbation.",
    },
    "amm_reserve_transition_view": {
        "description": "AMM reserve transitions reconstructed from Sync/Swap/Mint/Burn events and reserve-like state changes, including first/last reserves and estimated reserve/price-ratio changes when safely computable.",
        "default_use": "Judge whether AMM price-relevant state was significantly perturbed or restored.",
    },
    "market_mechanism_profile_view": {
        "description": "Compact neutral market-mechanism profiles built before judging. Each row binds candidate market source evidence, consumer evidence, outcome evidence, and competing-root hints such as token semantics or flash capital.",
        "default_use": "Keep market_manipulation C1/C2/C3 on the same market object/profile without using generic profit, swap volume, or flash capital as standalone proof.",
    },
    "protocol_accounting_outcome_view": {
        "description": "Compact neutral protocol-accounting outcome candidates. Rows summarize reward/share/vault/debt/collateral/staking/claim/liability state deltas together with related action, value-release, contribution-payout, and participant-delta evidence IDs.",
        "default_use": "Use for protocol_accounting_exploitation outcome conditions before treating empty value_release_view as no material outcome. Must be bound to the same selected bookkeeping candidate; generic profit or unrelated value movement remains insufficient.",
    },
    "transfer_event_view": {
        "description": "Transfer events extracted from synthesized trace.",
        "default_use": "Inspect token movement tied to calls/events.",
    },
    "external_fundflow_view": {
        "description": "Net fund flow and transfer records from external fundflow cache.",
        "default_use": "Inspect transaction-level asset movement.",
    },
    "profit_loss_view": {
        "description": "Top profit/loss candidate hints. This is not sufficient by itself to prove value extraction; it must be checked against participant net delta and contribution/payout evidence.",
        "default_use": "Use only as a hint for possible beneficiaries and loss sources.",
    },
    "participant_net_delta_view": {
        "description": "Per-address token/native asset deltas, showing what each participant sent and received without merging assets into USD value.",
        "default_use": "Check whether an address actually gained value or merely exchanged one asset for another.",
    },
    "value_release_view": {
        "description": "Neutral summary of sensitive calls that emit token transfers or native-value calls, including target contract, recipients, release records, repeated recipient/token releases, target balance changes, and related evidence ids.",
        "default_use": "Primary view for judging whether withdraw/redeem/borrow/claim/mint/transfer-like calls caused value out, repeated payout, or target balance movement. Use with participant_net_delta_view and contribution_vs_payout_view for outcome conditions.",
    },
    "contribution_vs_payout_view": {
        "description": "Neutral accounting summary for deposit/mint/redeem/withdraw/claim-like patterns, pairing participant input assets with output assets and protocol balance deltas.",
        "default_use": "Avoid mistaking normal deposits or redemptions for extraction.",
    },
    "flash_or_atomic_capital_view": {
        "description": "Candidate flash loan, flash swap, flash mint, or same-transaction borrow-repay patterns, including provider, borrower, borrow/repay amount, token, fee, and evidence IDs when available.",
        "default_use": "Primary view for atomic capital / flash-loan evidence.",
    },
    "beneficiary_controller_view": {
        "description": "Neutral summary of tx sender, entry contract, profit/loss recipients, and same-transaction call/fund-flow relationships.",
        "default_use": "Help judge distinguish direct profit, indirect beneficiary, protocol fee distribution, and no-link cases without asserting control.",
    },
}


def compact_packet_view_catalog() -> Dict[str, Dict[str, str]]:
    """Return the planning-facing view catalog without runtime manifest noise."""
    return {
        name: {
            "description": str(entry.get("description") or ""),
            "default_use": str(entry.get("default_use") or ""),
        }
        for name, entry in PACKET_VIEW_CATALOG.items()
    }


PACKET_VIEW_RENDER_POLICY: Dict[str, Dict[str, Any]] = {
    "tx_card": {
        "priority": 0,
        "role": "metadata",
        "cost": "small",
        "default_render_mode": "full",
    },
    "evidence_adequacy_view": {
        "priority": 1,
        "role": "metadata",
        "cost": "small",
        "default_render_mode": "full",
    },
    "operation_summary_view": {
        "priority": 2,
        "role": "signal_presence",
        "cost": "small",
        "default_render_mode": "full",
    },
    "classification_digest_view": {
        "priority": 3,
        "role": "signal_digest",
        "cost": "small",
        "default_render_mode": "full",
    },
    "address_labels": {
        "priority": 4,
        "role": "actor",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "trace_outline_view": {
        "priority": 5,
        "role": "structural",
        "cost": "medium",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 80,
        "max_prompt_chars": 20000,
    },
    "critical_call_view": {
        "priority": 6,
        "role": "structural",
        "cost": "large",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 48,
        "max_prompt_chars": 30000,
    },
    "critical_call_argument_view": {
        "priority": 7,
        "role": "call_arguments",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "source_unavailable_auth_view": {
        "priority": 7,
        "role": "authorization_context",
        "cost": "small",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 24,
        "max_prompt_chars": 18000,
    },
    "unknown_selector_view": {
        "priority": 7,
        "role": "structural",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "value_release_view": {
        "priority": 8,
        "role": "value_release",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
    "reentrancy_state_order_summary_view": {
        "priority": 4,
        "role": "state_order",
        "cost": "small",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 16,
        "max_prompt_chars": 24000,
    },
    "reentrancy_candidate_catalog_view": {
        "priority": 3,
        "role": "candidate_catalog",
        "cost": "small",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 24,
        "max_prompt_chars": 18000,
    },
    "reentrancy_state_order_view": {
        "priority": 30,
        "role": "state_order",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
    "semantic_state_delta_view": {
        "priority": 10,
        "role": "state",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
    "token_semantic_delta_summary_view": {
        "priority": 4,
        "role": "token_semantic",
        "cost": "small",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 24,
        "max_prompt_chars": 18000,
    },
    "token_accounting_origin_view": {
        "priority": 4,
        "role": "token_semantic",
        "cost": "small",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 24,
        "max_prompt_chars": 16000,
    },
    "price_relevant_state_view": {
        "priority": 10,
        "role": "state",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "amm_reserve_transition_view": {
        "priority": 11,
        "role": "state",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "market_mechanism_profile_view": {
        "priority": 4,
        "role": "market_profile",
        "cost": "small",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 12,
        "max_prompt_chars": 18000,
    },
    "protocol_accounting_outcome_view": {
        "priority": 4,
        "role": "accounting_outcome",
        "cost": "small",
        "default_render_mode": "top_rows",
        "max_prompt_rows": 16,
        "max_prompt_chars": 20000,
    },
    "state_change_view": {
        "priority": 12,
        "role": "state",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
    "event_view": {
        "priority": 13,
        "role": "event",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
    "transfer_event_view": {
        "priority": 14,
        "role": "fundflow",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
    "flash_or_atomic_capital_view": {
        "priority": 15,
        "role": "atomic_capital",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "beneficiary_controller_view": {
        "priority": 16,
        "role": "actor",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "participant_net_delta_view": {
        "priority": 17,
        "role": "fundflow",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
    "contribution_vs_payout_view": {
        "priority": 18,
        "role": "fundflow",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "external_fundflow_view": {
        "priority": 19,
        "role": "fundflow",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
    "profit_loss_view": {
        "priority": 20,
        "role": "economic_hint",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "token_info": {
        "priority": 21,
        "role": "metadata",
        "cost": "medium",
        "default_render_mode": "top_rows",
    },
    "trace_view": {
        "priority": 80,
        "role": "broad_trace",
        "cost": "large",
        "default_render_mode": "top_rows",
    },
}


DEFAULT_PACKET_VIEW_RENDER_POLICY: Dict[str, Any] = {
    "priority": 50,
    "role": "generic",
    "cost": "medium",
    "default_render_mode": "top_rows",
}


def packet_view_row_count(view_data: Any) -> int:
    """Estimate the useful row count for a packet view without expanding it."""
    if view_data is None:
        return 0
    if isinstance(view_data, list):
        return len(view_data)
    if isinstance(view_data, dict):
        for key in (
            "records",
            "rows",
            "events",
            "items",
            "deltas",
            "transfers",
            "release_records",
            "releases",
            "auth_contexts",
            "profiles",
            "unknown_selectors",
            "beneficiary_paths",
            "controller_hints",
            "pairs",
            "entries",
            "operations",
        ):
            value = view_data.get(key)
            if isinstance(value, list):
                return len(value)
        for key in ("trace", "decode_coverage", "state_coverage"):
            value = view_data.get(key)
            if isinstance(value, dict):
                return 1
        return 1 if view_data else 0
    return 0


def packet_view_render_policy(view_name: str) -> Dict[str, Any]:
    policy = dict(DEFAULT_PACKET_VIEW_RENDER_POLICY)
    policy.update(PACKET_VIEW_RENDER_POLICY.get(str(view_name), {}))
    manifest_entry = VIEW_MANIFEST.get(str(view_name), {})
    if manifest_entry:
        policy.update({
            "tier": manifest_entry.get("tier"),
            "manifest_cost": manifest_entry.get("cost"),
            "cost_score": manifest_entry.get("cost_score"),
            "prompt_role": manifest_entry.get("prompt_role"),
            "default_usage": manifest_entry.get("default_usage"),
            "allow_first_pass": manifest_entry.get("allow_first_pass"),
            "allow_followup": manifest_entry.get("allow_followup"),
            "is_attack_evidence": manifest_entry.get("is_attack_evidence"),
            "empty_policy": manifest_entry.get("empty_policy"),
        })
    return policy


def build_allowed_view_summary(
    packet: Dict[str, Any],
    allowed_followup_views: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Summarize only the views a judge step is allowed to request later."""
    views = packet.get("views", {}) if isinstance(packet, dict) else {}
    summary: Dict[str, Dict[str, Any]] = {}
    for name in allowed_followup_views:
        catalog_entry = PACKET_VIEW_CATALOG.get(str(name), {})
        available = isinstance(views, dict) and name in views
        view_data = views.get(name) if available else None
        manifest_entry = dict(VIEW_MANIFEST.get(str(name), {}))
        summary[str(name)] = {
            "available": bool(available),
            "row_count": packet_view_row_count(view_data) if available else 0,
            "cost": manifest_entry.get("cost", "unknown"),
            "use": str(catalog_entry.get("default_use", ""))[:240],
        }
    return summary
