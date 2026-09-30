from __future__ import annotations

from typing import Any, Dict, Optional

from evotx.core.plan import (
    _DEFAULT_PACKET_REFS,
    ORDINARY_MAX_DEFAULT_VIEWS,
    ORDINARY_MAX_FOLLOWUP_VIEWS,
    SOURCE_MAX_FOLLOWUP_VIEWS,
    SPECIAL_MAX_DEFAULT_VIEWS,
    STATE_ORDER_MAX_FOLLOWUP_VIEWS,
    apply_access_control_binding_mode,
    apply_flashloans_stateful_runtime,
    apply_insufficient_validation_stateful_runtime,
    apply_market_manipulation_stateful_runtime,
    apply_protocol_accounting_stateful_runtime,
    apply_reentrancy_stateful_runtime,
    apply_token_semantic_stateful_runtime,
    compile_rule_to_baseline_plan,
    condition_view_budget,
    constrain_judge_step_first_pass,
    followup_round_policy,
    is_reentrant_state_order_text,
    source_dependency_level,
    sanitize_plan_judge_questions,
)
from evotx.core.labels import normalize_attack_label
from evotx.core.logic import UnsafeLogicExpression, extract_logic_names
from evotx.core.schemas import EvolvingRule, EvidencePlan, JudgeStep
from evotx.runtime.view_catalog import PACKET_VIEW_CATALOG
from evotx.utils.json_utils import (
    extract_json_object,
    repair_likely_mojibake,
    stable_json_dumps,
)


_AVAILABLE_PACKET_VIEWS = list(PACKET_VIEW_CATALOG)

_ALLOWED_FOLLOWUP_TOOLS = {
    "read_packet_view",
    "read_evidence_by_id",
    "read_evidence_context",
    "get_local_call_context",
    "read_function_chunk",
}

_VIEW_DESCRIPTIONS = {
    name: entry["description"] for name, entry in PACKET_VIEW_CATALOG.items()
}


_LABEL_GUIDANCE: Dict[str, str] = {
    "access_control": """
Label-specific planning guidance for access_control:
- Preserve condition ownership: C1 locates one concrete actor/capability/target
  candidate, C2 decides required and missing authorization for that candidate,
  C3 decides its protected effect, and E1/E2 test legitimate authority,
  entitlement, or a positively supported replacement mechanism.
- Preserve candidate_id and overlapping evidence ids across those roles. A
  single sensitive call is enough; retain a separate entry parent and downstream
  call only when trace evidence shows they differ and are causally connected.
- C1 should use critical_call_view, trace_outline_view, operation_summary_view,
  value_release_view, and state/configuration evidence to locate the candidate.
  Unknown selectors remain candidates only when tied to protected state or value.
- C2 first-pass/default evidence should retain critical_call_view together with
  critical_call_argument_view, address/caller context,
  source_unavailable_auth_view, and beneficiary_controller_view. Use targeted
  source and same-transaction permission changes to decide the concrete
  authorization boundary. Summary views may supplement but should not replace
  the concrete call/argument pair. Source is helpful but not mandatory when
  behavioral evidence is specific; absence of source or a modifier alone
  proves neither side.
- C3 should use value release, state/configuration deltas, beneficiary,
  contribution/entitlement, and fund-flow evidence for the same candidate.
- Generic public calls, swaps, callbacks, flash loans, profit, or value movement
  do not prove access control. User-supplied content validation belongs to
  insufficient_validation unless it exposes an unauthorized protected
  capability. Exclusions require positive same-candidate evidence, not labels.
""",
    "price_manipulation": """
Label-specific planning guidance for price_manipulation:
- Treat this as a transaction-visible price-source perturbation and consumption problem. Do not prove price manipulation from profit, swap volume, flashloan use, or beneficiary gain alone.
- Route each local condition by its semantic role:
  - For price-source perturbation conditions, first-pass views should prioritize operation_summary_view, evidence_adequacy_view, price_relevant_state_view, amm_reserve_transition_view, event_view, critical_call_view, and trace_outline_view. These should expose reserve changes, oracle/valuation reads, exchange-rate changes, sync/skim/swap/mint/burn effects, price cumulative/tick/sqrtPrice updates, or pool-balance changes.
  - For conditions about an operation consuming the distorted price, do not require a distinct external protocol. A same-market swap, settlement, mint/burn, or extraction can qualify when its amount or outcome demonstrably depends on an abnormal, independently established distortion. The actor's ordinary sequential swaps merely repricing against normal AMM reserve updates are not sufficient consumption. Bare skim, sync, reserve writing, token transfer, or swap co-occurrence is also insufficient. Use the price_state profiles in market_mechanism_profile_view to locate candidate consumer evidence, then inspect only the projected critical_call_view / critical_call_argument_view rows and targeted local context.
  - For restoration, unwind, temporary capital, or one-transaction round-trip conditions, prefer flash_or_atomic_capital_view, amm_reserve_transition_view, transfer_event_view, external_fundflow_view, critical_call_view, and trace_outline_view.
  - For final extraction or protocol/user loss conditions, then use value_release_view, participant_net_delta_view, contribution_vs_payout_view, beneficiary_controller_view, external_fundflow_view, transfer_event_view, and profit_loss_view. Treat profit_loss_view only as a candidate hint.
- Do not overfit to AMM pairs only. Price manipulation can involve oracle reads, exchange rates, virtual prices, collateral valuations, settlement prices, slippage guards, pool balance visibility, or reserve-like state. If the packet lacks clean reserve transitions, use price_relevant_state_view, semantic_state_delta_view, event_view, and critical_call_view to look for valuation consumption instead of declaring the evidence absent.
- Exclusions should distinguish ordinary arbitrage, legitimate liquidation, normal large swaps, routine AMM sync/swap behavior, capital-backed deposit/redeem activity, token semantic mismatch, protocol accounting exploitation, access control, reentrancy, insufficient validation, and flashloan-only capital amplification when those mechanisms are primary.
- Include read_function_chunk only when a local question needs source-level price/oracle/exchange-rate/slippage/valuation formula semantics after packet views locate a concrete function.
- Consumption conditions must not merely ask whether a borrow/mint/redeem/liquidate/swap exists. They should ask whether the evidence shows a concrete perturbed price-relevant source and a causally linked value-sensitive operation consuming or depending on that same source; that operation may be in the same market or in a downstream protocol.
- For price-consumption steps, max_followups=2 is allowed only as a dependent sequence: first locate the candidate price-read/consumer call, then inspect local context, arguments, state rows, or source for that selected call. Do not use the second round for an unrelated broad view.
- Keep price_manipulation separate from market_manipulation. Do not import skim/donate, sandwich/MEV, ordering, or generic pool-accounting profiles as positive price evidence unless the transaction also proves that a distorted price/reserve was actually consumed.
- Keep C1 and C2 local: C1 establishes the abnormal price-relevant distortion;
  C2 establishes transaction-visible consumption and amount/settlement
  causality. C2 may reference C1's selected distortion but should not repeat the
  complete value-backed/unbacked definition.
- Outcome conditions must not rely on value release or profit alone. They should ask whether the same perturbed-and-consumed price source plausibly caused abnormal mint, borrow, redeem, reward, settlement, accounting shift, protocol loss, or attacker-favorable asset delta.
- If the condition text generated by the rule is vague, rewrite the judge question to include "the same concrete price-relevant source" and "not merely generic profit/swap volume/flashloan".
""",
    "market_manipulation": """
Label-specific planning guidance for market_manipulation:
- Treat this as broader than price_manipulation but still market-rooted: AMM reserves, pair/pool visible balances, oracle/valuation inputs, slippage/settlement checks, skim/sync behavior, donation/pre-seeding effects, migration/predictable-pair setup, sandwich/MEV positioning, and arbitrage-like extraction may all matter.
- Preserve one selected market-mechanism profile across C1/C2/C3. The main profiles are: (1) price-state manipulation such as AMM/oracle/slippage/sandwich, (2) pool-accounting or reserve extraction such as skim/donate/sync/pair-visible balance misuse, and (3) ordering/MEV/arbitrage extraction. Do not let C1 use one profile while C2/C3 answer from unrelated profit, swaps, flash capital, or generic value flow.
- Route each local condition by its semantic role:
  - For market-state distortion or market-state misuse conditions, first-pass views should prioritize operation_summary_view, evidence_adequacy_view, market_mechanism_profile_view, price_relevant_state_view, amm_reserve_transition_view, event_view, critical_call_view, and trace_outline_view. Do not use profit_loss_view, participant_net_delta_view, value_release_view, or generic beneficiary gain as primary proof before market-facing state distortion/misuse is checked.
  - For pool-balance, donation, pre-seeded pair, migration-before-pair-creation, predictable pair address, skim/sync, pair-visible balance, or settlement-balance conditions, include transfer_event_view, event_view, critical_call_view, trace_outline_view, external_fundflow_view, beneficiary_controller_view, price_relevant_state_view, and amm_reserve_transition_view as first-pass or early follow-up. Market manipulation may exist even when reserve transition rows are incomplete or no clean AMM reserve pair can be reconstructed.
  - For downstream consumption conditions, prioritize market_mechanism_profile_view, critical_call_view, trace_outline_view, event_view, price_relevant_state_view, semantic_state_delta_view, and amm_reserve_transition_view so the judge can see how the same selected market state was consumed by swap, mint, redeem, borrow, liquidation, settlement, valuation, skim/sync/donation settlement, sandwich victim execution, MEV ordering, or accounting logic. AMM-internal skim/donate/sync can be the consumer only for the pool-accounting/reserve-extraction profile.
  - For value extraction or protocol-loss outcome conditions, use value_release_view, contribution_vs_payout_view, participant_net_delta_view, beneficiary_controller_view, external_fundflow_view, transfer_event_view, and profit_loss_view. Treat profit_loss_view as a hint, never as root-cause proof. C3 often needs two follow-up rounds when value_release/profit_loss is empty or one-sided but AMM reserve transitions are present.
- Token-semantic exclusions should be explicit. If the local question concerns locked balances, fee-on-transfer, deflationary mechanics, reflection/rebase, special balanceOf behavior, transfer hooks, mint/burn/tax/fee logic, or arbitrary token accounting side effects, include transfer_event_view, semantic_state_delta_view, state_change_view, critical_call_view, event_view, and read_function_chunk when needed. Do not classify pure token semantic exploitation as market manipulation merely because it changes pool price or pair-visible balances.
- Slippage, oracle, valuation, settlement-price, minOut, freshness, or liquidity-protection failures should not be automatically excluded as insufficient validation. If the consumed market state, oracle value, AMM price, pool balance, or settlement rate is the transaction-visible root of value extraction, the plan should let the judge evaluate it as market manipulation or market-facing misuse. Include read_function_chunk only when the local question needs source-level slippage/oracle/valuation/settlement guard semantics.
- Exclusions should separate normal large swaps, legitimate liquidation, routine keeper/oracle maintenance, access control, insufficient validation, protocol accounting exploitation, token semantic exploitation, reentrancy, and pure flashloan capital amplification when those mechanisms are primary. Do not exclude sandwich, MEV, skim/donate, or arbitrage-like behavior merely by name when it is the selected target mechanism.
""",
    "reentrancy": """
Label-specific planning guidance for reentrancy:
- Treat reentrancy as a structural execution-order vulnerability, not merely the presence of a callback, flash loan, nested call, or profit. A useful plan should help the judge connect nested execution to stale state, repeated entitlement, delayed state finalization, read-only price/accounting observation, or value release that depends on the nested timing.
- Preserve one nested-entry candidate across the core conditions: C1 locates outer call -> external edge -> re-entry path and logical storage/accounting domain; C2 verifies a value-relevant effect on that same candidate; C3 verifies phase-correct stale/intermediate-state causality on that same candidate. Do not allow unrelated callbacks, profit, value flow, or slot overlap to satisfy separate conditions.
- In EVM execution an SSTORE completed before the external edge is immediately visible to nested execution. A C3 plan must distinguish state_updated_before_external_edge from outer-read -> external-edge -> inner-read/effect -> outer-protective-write-after-callback, or an equivalent read-only/accounting stale-observation pattern.
- Route each local condition by its semantic role:
  - For structural nested-entry conditions, first-pass evidence should prioritize operation_summary_view, evidence_adequacy_view, reentrancy_candidate_catalog_view, and critical_call_view. The catalog ranks deterministic packet candidate IDs; the Judge must not invent candidate IDs and may retain up to five plausible candidates. Use reentrancy_state_order_summary_view and trace_outline_view as follow-up when the catalog anchor needs phase or path detail. Broad trace_view and full reentrancy_state_order_view should usually be later follow-up.
  - For state-order, stale-state, repeated-claim, delayed-burn/reset, allowance/balance/entitlement, share/debt/collateral, or read-only reentrancy conditions, prioritize reentrancy_state_order_summary_view, semantic_state_delta_view, trace_outline_view, critical_call_view, event_view, and price_relevant_state_view when oracle/virtual-price/read-only semantics may matter. Keep full reentrancy_state_order_view and state_change_view for follow-up detail.
  - For value-effect conditions, use value_release_view, participant_net_delta_view, contribution_vs_payout_view, transfer_event_view, external_fundflow_view, and profit_loss_view, but do not let value flow alone prove reentrancy without structural/order evidence.
  - For exclusions, include views that distinguish benign callback scaffolding, access-control bypass, insufficient validation, market/oracle manipulation, token semantic mismatch, and pure flashloan capital amplification from true nested state/order exploitation.
- Keep critical_call_argument_view available as an early follow-up after critical_call_view identifies a concrete call. Use it to inspect decoded callback targets, actors, beneficiaries, amounts, and call parameters. Arguments can disambiguate the nested path, but they do not by themselves prove stale-state ordering or repeated consumption.
- A structural C1 should use the small ranked reentrancy_candidate_catalog_view instead of relying on the chronological prefix of the larger state-order summary. A state-order causality condition may still use operation_summary_view, evidence_adequacy_view, reentrancy_state_order_summary_view, semantic_state_delta_view, and critical_call_view when those views answer different local questions.
- Do not over-constrain every positive condition to exact same-function re-entry. Same-function re-entry, cross-function re-entry within the same protocol/accounting domain, token hook re-entry, callback-driven re-entry into a logically related state-mutating function, and read-only reentrancy that consumes stale price/accounting state can all be relevant. The key is whether the nested path observes or exploits pre-finalized state or repeated entitlement.
- Standard callbacks such as flash-loan callbacks, DEX swap callbacks, ERC token receiver hooks, and token transfer hooks are not automatically benign. They support an exclusion only when they do not re-enter a vulnerable protocol/accounting domain, do not consume stale state, and do not enable repeated or excess value release.
- Conversely, callbacks into an attacker contract, complex multi-protocol orchestration, flashloan-funded swaps, and generic value extraction should not by themselves satisfy reentrancy. If the primary mechanism is unauthorized caller access, invalid input/callback data, market-state manipulation, token semantic behavior, or pure accounting formula abuse without nested stale-state/order exploitation, the plan should make room for an exclusion.
- Use get_local_call_context only after packet views locate a concrete reentrant/critical/unknown-selector call id; prefer child critical calls, value-release rows, or reentrancy_state_order rows rather than call:0 or a transaction-root target. The follow-up should compare outer and inner call relation, parent/child context, nearby events, and nearby state/order rows.
- Include read_function_chunk when a local question needs source-level semantics such as nonReentrant/guard behavior, checks-effects-interactions order, balance reset/burn/claim-flag timing, callback caller validation, token hook behavior, view/oracle/virtual-price read semantics, or whether a related function belongs to the same protocol/accounting domain. Source is helpful but not mandatory when packet evidence already establishes the nested order and value effect.
- For read-only reentrancy, do not require direct sstore writes in the inner path. The decisive evidence may be an oracle/virtual-price/balance/rate read during a nested call, followed by a downstream borrow/mint/redeem/swap/withdraw that consumes the stale or temporarily inconsistent value.
""",
    "insufficient_validation": """
Label-specific planning guidance for insufficient_validation:
- Treat this as source- and parameter-semantics-sensitive. The planner should help the judge determine whether a callable protocol path accepted malformed, fake, inconsistent, boundary, precision/rounding, callback-data, token/path/market/amount, return-value, oracle/external-state, or business-state assumptions as valid.
- For insufficient_validation plan initialization, prefer this stable semantic template:
  - C1/path-consumption condition: locate the value-sensitive path, rank candidate objects, and select at most two or three diverse mechanism families for first-pass binding without deciding whether validation is adequate. Keep lower-ranked candidates as follow-up leads. A standard same-token ERC-20 balance/allowance/supply query used in its ordinary token operation, or generic swap parameters alone, is not a sufficient target-object candidate. Prefer operation_summary_view, evidence_adequacy_view, critical_call_view, critical_call_argument_view, and semantic_state_delta_view. Keep trace_outline_view as early follow-up when the compact call views do not expose enough path context.
  - C2/validation-gap condition: audit the selected C1 candidates before selecting the strongest same-object gap. Ask about semantic content/source/freshness/range/invariant validation, not generic caller/owner/role/whitelist/callback-sender authorization; route the latter to E1 unless authority data or signer/domain content is itself the consumed object. Require an exact scope match: caller ownership, pause, reentrancy, balance/allowance, and contract-existence checks do not respectively prove accounting proportionality, token/path/market allowlisting, callback content, oracle freshness, or business-invariant validation. Prefer operation_summary_view, evidence_adequacy_view, trace_outline_view, critical_call_view, critical_call_argument_view, semantic_state_delta_view, and contribution_vs_payout_view. If selected candidates are guarded, request the next ranked callback-data, token/path/address, amount/allowance, signature/domain, external-return, computed-value, or business-state candidate as follow-up rather than expanding every candidate in one prompt. When a wrapper only exposes outer guards and delegates candidate consumption, use a second read_function_chunk with target_function for the named internal callee. Keep value_release_view, participant_net_delta_view, unknown_selector_view, event_view, and state_change_view available as follow-up; allow read_function_chunk mainly here. If source is unavailable, use positive behavioral and contribution/value/accounting evidence; no visible check in a complete trace does not prove a passing branch or modifier was absent.
  - Keep C2 local to the selected object's validation dimension and observed
    gap. Do not repeat the full Access Control, Reentrancy, market, token, or
    accounting replacement policy in C2; the matching exclusion owns that
    owner decision.
  - C3/outcome condition: inspect a concrete incorrect outcome/state/value effect caused by the same selected object. Prefer operation_summary_view, evidence_adequacy_view, semantic_state_delta_view, and value_release_view; put participant_net_delta_view, contribution_vs_payout_view, critical_call_view, and trace_outline_view in follow-up. Generic downstream swaps, profit, or value release without the same-object link are insufficient.
  - E2/independent-mechanism condition: allow another mechanism to exclude insufficient_validation when it fully explains the outcome and semantic invalidity of the selected object is unnecessary, even if that object remains a conduit in the route.
  - Do not put reentrancy_state_order_summary_view in IV C2 default unless the local question explicitly mentions reentrancy, nested execution, stale state-order, or callback order; even then prefer it as follow-up.
  - Keep condition questions compact. Examples belong in guidance, not inside the condition text.
- Do not assign packet views by condition id. Infer the semantic role of each local condition from its text, then choose compact first-pass views and follow-up views. The rule may name these roles in any order.
- Do not turn insufficient_validation into generic value-flow detection. Value release is outcome evidence; it is not by itself proof that validation was missing.
- If a local condition asks about a value-sensitive operation consuming user-supplied or externally sourced data/state, prioritize locating views: operation_summary_view, evidence_adequacy_view, trace_outline_view, critical_call_view, critical_call_argument_view, unknown_selector_view, and event_view. Use value_release_view only when the same condition also asks whether the operation releases or moves value.
- If a local condition asks whether inputs, callback data, return values, external state, token/path/market addresses, amounts, signatures, price/oracle values, boundary/range values, precision/rounding cases, or business assumptions were invalid or adversarial, prioritize operation_summary_view, evidence_adequacy_view, trace_outline_view, critical_call_view, unknown_selector_view, semantic_state_delta_view, and event_view. Put trace_view and state_change_view in follow-up unless the condition explicitly needs broad call or raw storage detail.
- If a local condition asks whether the protocol accepted or failed to validate such data/state, include packet views that identify the concrete callee/function/evidence_id first, then allow read_function_chunk when the remaining uncertainty is source-level validation logic, parameter checks, callback initiator/data checks, return-value checks, token/path whitelisting, min/max/range checks, rounding/precision guards, oracle freshness/deviation checks, or business invariant enforcement.
- If a local condition asks about an incorrect outcome, invariant violation, allowance/share/debt/collateral drift, accounting mismatch, disproportionate effect, or value release caused by the accepted data/state, prioritize semantic_state_delta_view, value_release_view, contribution_vs_payout_view, participant_net_delta_view, beneficiary_controller_view, and state_change_view as follow-up. Treat profit_loss_view only as a hint.
- If a local exclusion asks whether the behavior is normal dust, rounding, routine settlement, or expected protocol behavior, include semantic_state_delta_view before treating tiny external transfers as normal. A dust-sized transfer can still support insufficient_validation if it triggers material state/accounting delta, allowance consumption, share/reward drift, or value release.
- If a local exclusion asks whether access control is primary, provide evidence for caller/owner/role/whitelist/callback-sender authorization boundaries using tx_card, address_labels, beneficiary_controller_view, trace_outline_view, critical_call_view, unknown_selector_view, value_release_view, and read_function_chunk when needed. Do not let an arbitrary call or callback shape alone become access control if the callable path accepted adversarial data/targets/amounts.
- If a local exclusion asks whether market/oracle/price manipulation is primary, include price_relevant_state_view, amm_reserve_transition_view, flash_or_atomic_capital_view, event_view, critical_call_view, and semantic_state_delta_view. The judge should distinguish: (1) the exploited protocol lacked validation of a consumed price/oracle/external state, which supports insufficient_validation; (2) the primary root cause is market/oracle distortion independent of a validation gap, which supports exclusion.
- If a local exclusion asks whether reentrancy/callback/control-flow is primary, include reentrancy_state_order_summary_view, trace_outline_view, critical_call_view, semantic_state_delta_view, get_local_call_context, and read_function_chunk when needed. Keep full reentrancy_state_order_view as follow-up detail. The judge should distinguish stale-state/nested-execution exploitation from callback data, callback initiator, target market, recipient, amount, path, return value, or state assumption being accepted without validation.
- If a local exclusion asks whether token-semantic or protocol-accounting exploitation is primary, include transfer_event_view, semantic_state_delta_view, state_change_view, operation_summary_view, contribution_vs_payout_view, and value_release_view as appropriate. Exclude only when token behavior or bookkeeping formula/order is the primary root cause and invalid input/state acceptance is not causal.
- Exclusions should filter access-control, market-state manipulation, token semantic mismatch, protocol accounting exploitation, reentrancy, and pure flashloan capital amplification only when those mechanisms are primary and the invalid input/state acceptance is not the root cause.
""",
    "protocol_accounting_exploitation": """
Label-specific planning guidance for protocol_accounting_exploitation:
- Treat this as protocol bookkeeping misuse, not merely profitable transfer flow. The plan should help the judge inspect how a protocol records, updates, consumes, or reuses internal accounting state.
- Preserve one shared bookkeeping candidate across the core semantic roles: C1 identifies the protocol-internal accounting candidate, C2 verifies consumption or reliance on that same candidate, and C3 verifies candidate-specific abnormal outcome. Do not let C1 use a reward/share state, C2 use an unrelated value transfer, and C3 use generic profit.
- Route each local condition by its semantic role:
  - For root-cause accounting-state, bookkeeping, share/reward/debt/collateral, vault/strategy, eligibility, stale variable, or accounting-order conditions, first-pass views should prioritize operation_summary_view, evidence_adequacy_view, semantic_state_delta_view, trace_outline_view, critical_call_view, and reentrancy_state_order_summary_view when stale or incomplete update order may matter. Keep state_change_view and full reentrancy_state_order_view available for detailed slot/order rows.
  - For formula-consumption, claim eligibility, reward distribution, share mint/redeem, collateral/debt, exchange-rate, or vault accounting questions, include semantic_state_delta_view, critical_call_view, event_view, contribution_vs_payout_view, and read_function_chunk when source-level formula or invariant semantics are needed.
  - For contribution-vs-payout, repeated claim, excessive reward, inflated share, incorrect withdrawal, or undercollateralized borrow outcomes, use contribution_vs_payout_view, value_release_view, participant_net_delta_view, beneficiary_controller_view, semantic_state_delta_view, and profit_loss_view as a hint.
  - For detailed proof or contradiction, put state_change_view, trace_view, transfer_event_view, external_fundflow_view, event_view, and broad critical_call_view detail in allowed_followup_views rather than overloading the first prompt.
- Do not let generic profit, flashloan use, swaps, or many transfers satisfy an accounting root-cause condition before checking accounting variables, formulas, state-update order, eligibility, or contribution/payout consistency.
- Exclusions should separate AMM/oracle/market manipulation, token semantic mismatch, access control, insufficient validation, classic reentrancy, and pure flashloan capital amplification when those mechanisms are primary and the protocol's own bookkeeping logic is not the root cause.
""",
    "token_semantic_exploitation": """
Label-specific planning guidance for token_semantic_exploitation:
- Treat this as a mismatch between nominal token operations and actual balance/accounting effects. The root evidence should show token behavior semantics, not only market price movement or final profit.
- Prefer same-candidate stateful flow when the rule has multiple positive
  conditions: the first root-cause condition should produce
  token_semantic_candidate; downstream protocol-reliance and outcome
  conditions should consume token_semantic_candidate and preserve candidate_id.
  Exclusion steps should also consume the same candidate state when present.
- Route each local condition by its semantic role:
  - For transfer/event/accounting mismatch conditions, first-pass views should prioritize operation_summary_view, evidence_adequacy_view, token_semantic_delta_summary_view, and token_accounting_origin_view. Keep transfer_event_view, semantic_state_delta_view, state_change_view, and event_view as follow-up detail unless the rule explicitly needs raw rows.
  - For self-transfer, special-address transfer, locked/excluded/reflected balance exposure, fee/tax/deflationary mechanics, rebase/reflection, arbitrary mint/burn, balanceOf anomaly, or transfer-side-effect conditions, use token_semantic_delta_summary_view and token_accounting_origin_view first, then allow critical_call_view, unknown_selector_view, semantic_state_delta_view, state_change_view, transfer_event_view, event_view, and read_function_chunk when source-level token semantics are needed.
  - For protocol reliance on the token behavior, include token_accounting_origin_view, critical_call_view, amm_reserve_transition_view, price_relevant_state_view, and semantic_state_delta_view. The plan should help the judge distinguish token-contract-origin semantic divergence from protocol-internal ledger/reentrancy/accounting divergence.
  - For AMM pair interaction, pair-visible balance, skim/sync/swap extraction, or reserve/accounting mismatch conditions, include token_accounting_origin_view, amm_reserve_transition_view, price_relevant_state_view, critical_call_view, trace_outline_view, event_view, transfer_event_view, external_fundflow_view, and beneficiary_controller_view as first-pass or early follow-up.
  - For downstream extraction conditions, use value_release_view, participant_net_delta_view, contribution_vs_payout_view, external_fundflow_view, beneficiary_controller_view, and profit_loss_view as outcome evidence, but require token-behavior mismatch evidence for the root cause.
- Do not add a generic condition saying market/accounting/profit is sufficient
  for token semantic exploitation. Plan changes may add views/tools, but must
  preserve the rule's distinction between token-contract semantics and protocol
  accounting effects.
- Do not classify pure market manipulation, pure protocol accounting exploitation, access control, insufficient validation, reentrancy, or flashloan-only capital amplification as token semantic exploitation unless non-standard token transfer/balance/accounting behavior is the primary mechanism.
- Exclusion steps should check whether the receiving protocol correctly accounts for actual received amounts, whether fee-on-transfer/deflation is explicitly handled, and whether the observed effect is an ordinary token mechanic with no mismatch between nominal operation and actual accounting effect.
""",
    "flashloans": """
Label-specific planning guidance for flashloans:
- Flashloan presence alone is not sufficient. The plan should help the judge establish an atomic sequence: temporary borrow or flash swap/mint -> state/accounting/price perturbation or protocol-sensitive operation -> value extraction or protocol loss -> repayment/unwind.
- Preserve one selected temporary atomic-capital candidate across C1/C2/C3. C1 should identify provider/source, borrower/callback context, asset, amount/scale, repayment or settlement, call-burst/call-count support when visible, and evidence ids. C2 must explain why that same candidate materially enables the exploit chain, not merely financing. C3 must prove the same candidate/mechanism caused the outcome.
- Route each local condition by its semantic role:
  - For temporary-capital and repayment/unwind conditions, first-pass views should prioritize operation_summary_view, evidence_adequacy_view, flash_or_atomic_capital_view, trace_outline_view, critical_call_view, external_fundflow_view, and transfer_event_view. The judge should see provider, borrower, token, borrow amount, repay amount, fee, and related evidence ids when available.
  - For exploit-enabler conditions, use operation_summary_view, trace_outline_view, critical_call_view, semantic_state_delta_view, price_relevant_state_view, amm_reserve_transition_view, state_change_view, event_view, and source/function tools when needed. Call-count/call-burst evidence is a strong support signal when it links the selected capital candidate to scale, ordering, solvency, callback reach, price/accounting impact, or a protocol-sensitive operation; it is not sufficient by itself.
  - For market-state amplification conditions, use price_relevant_state_view, amm_reserve_transition_view, event_view, critical_call_view, trace_outline_view, external_fundflow_view, and transfer_event_view, but require evidence that the temporary capital materially enables the distortion.
  - For protocol-accounting amplification conditions, use semantic_state_delta_view, contribution_vs_payout_view, value_release_view, critical_call_view, event_view, and state_change_view as follow-up for detailed accounting rows.
  - For validation/solvency/callback/repayment-check bypass conditions, include critical_call_view, unknown_selector_view, trace_outline_view, semantic_state_delta_view, and read_function_chunk when source-level callback, solvency, or repayment-check semantics are needed.
  - For extraction/outcome conditions, use value_release_view, participant_net_delta_view, contribution_vs_payout_view, beneficiary_controller_view, external_fundflow_view, transfer_event_view, and profit_loss_view as hints.
- Exclusions should distinguish ordinary arbitrage or liquidation with temporary capital, flashloan-only capital provision, access control, insufficient validation, reentrancy, token semantics, market manipulation, protocol accounting exploitation, and direct asset drain when flashloan capital is incidental rather than a material enabler of the exploit. Do not exclude solely because the payload resembles another attack family.
- Do not let final profit, large swaps, or a borrowed amount alone satisfy the root cause. The local condition should connect temporary capital to a state/accounting/price/solvency/callback effect that would not be possible or material without the atomic capital.
""",
}


def _label_specific_guidance(label: str) -> str:
    key = normalize_attack_label(label)
    guidance = _LABEL_GUIDANCE.get(key, "").strip()
    if guidance:
        return guidance
    return (
        "No label-specific guidance matched. Use only the semantic routing "
        "guidance above and the rule's local condition semantics."
    )


class PlanGenerator:
    """Compile an evolving rule into a packet-view-based evidence plan."""

    def __init__(
        self,
        llm=None,
        tool_manifest: Optional[Dict[str, Any]] = None,
        allow_fallback: bool = True,
        enforce_label_plan_template: bool = False,
    ):
        self.llm = llm
        self.allow_fallback = allow_fallback
        self.enforce_label_plan_template = bool(enforce_label_plan_template)

    def generate(
        self,
        rule: EvolvingRule,
        tx_context_summary: Optional[Dict[str, Any]] = None,
    ) -> EvidencePlan:
        """Generate a plan. Falls back to a deterministic keyword-based compiler."""
        tx_context_summary = tx_context_summary or {}

        if self.llm is None:
            return compile_rule_to_baseline_plan(rule)

        prompt = self._build_prompt(rule, tx_context_summary)
        try:
            text = self.llm.complete(prompt)
            data = repair_likely_mojibake(extract_json_object(text))
            return self._to_plan(
                data,
                rule,
                tx_context_summary,
                enforce_label_plan_template=self.enforce_label_plan_template,
            )
        except Exception:
            if not self.allow_fallback:
                raise
            plan = compile_rule_to_baseline_plan(rule)
            plan.plan_note = (
                "LLM plan generation failed; using baseline keyword-based plan."
            )
            plan.metadata["generator"] = "baseline_after_llm_failure"
            return plan

    def _build_prompt(
        self, rule: EvolvingRule, tx_context_summary: Dict[str, Any]
    ) -> str:
        label = normalize_attack_label(
            (rule.metadata or {}).get("attack_label", ""))
        label_guidance = _label_specific_guidance(label)

        view_docs = "\n".join(
            f"{i+1}. {name}\n   {_VIEW_DESCRIPTIONS[name]}"
            for i, name in enumerate(_AVAILABLE_PACKET_VIEWS)
        )

        return f"""
You are an EvoTx packet-plan generator.

Task label:
{label}

You are given a semantic evolving rule.
The rule describes what local conditions should be judged.
Your task is to choose which packet views are needed for each condition and produce an EvidencePlan.

Runtime setting:
- A compact evidence packet has already been built from a real transaction.
- Judge LLM receives one local question plus selected packet views.
- Judge LLM returns answer, confidence, reason, and supporting_evidence_ids.
- Do not reference the legacy environment. Optional follow-up tools are read-only evidence tools only.

Available packet views:
{view_docs}

Rules:
- Every judge_step should include operation_summary_view and evidence_adequacy_view in default_evidence_refs. They provide boolean signal flags and packet completeness metadata that help the judge distinguish absent signals from missing evidence.
- operation_summary_view is the canonical signal-presence view. Do not invent signal_presence_view.
- Include trace_outline_view in default_evidence_refs when the condition involves execution structure, call patterns, or when trace_view would otherwise be needed.
- Choose views by the semantic purpose of each local question, not by attack label or condition id. Do not use the same default_evidence_refs for every condition. Root-cause conditions and value-extraction conditions usually require different evidence views.
- Do not solve every source-helpful condition by using the same locating default views. Source locating views are required only when the condition truly needs function/source resolution.
- For local root-cause conditions, select views that expose the mechanism named by the condition before selecting value/profit views. If a label-specific guidance block is present below, use it to choose specialized views for the current label.
- For outcome/effect conditions, default_evidence_refs must include at least one outcome view such as semantic_state_delta_view, value_release_view, participant_net_delta_view, or contribution_vs_payout_view when relevant.
- For authorization conditions, default_evidence_refs should include address_labels or beneficiary_controller_view when relevant.
- For market/price conditions, default_evidence_refs should include price_relevant_state_view or amm_reserve_transition_view when relevant.
- For token semantic conditions, default_evidence_refs should prefer token_semantic_delta_summary_view and token_accounting_origin_view when relevant; keep transfer_event_view or semantic_state_delta_view for follow-up detail unless raw rows are directly needed.
- If the question is about value extraction, beneficiary gain, protocol/user loss, disproportionate payout, repeated release, withdraw/redeem/borrow/claim outcome, or final profit/loss, prefer value_release_view, participant_net_delta_view, contribution_vs_payout_view, beneficiary_controller_view, external_fundflow_view, and profit_loss_view. Treat profit_loss_view as a hint, not sufficient proof by itself.
Label-specific guidance for this task:
{label_guidance}

- Budget policy: ordinary default_evidence_refs <= {ORDINARY_MAX_DEFAULT_VIEWS}; source-required or state-order exceptions <= {SPECIAL_MAX_DEFAULT_VIEWS}; access_control core and insufficient_validation initialization roles may use <=6 compact default views when the semantic role needs it. Ordinary allowed_followup_views <= {ORDINARY_MAX_FOLLOWUP_VIEWS}; source-required <= {SOURCE_MAX_FOLLOWUP_VIEWS}; state-order <= {STATE_ORDER_MAX_FOLLOWUP_VIEWS}. Keep views compact and directly relevant to the local question.
- Put large or diagnostic views into allowed_followup_views instead of default_evidence_refs. trace_view, state_change_view, external_fundflow_view, transfer_event_view, full reentrancy_state_order_view, and profit_loss_view should not be placed into every condition. Treat profit_loss_view only as a hint and rarely as default for root-cause conditions.
- Avoid the bad pattern of always giving every condition operation_summary_view, participant_net_delta_view, value_release_view, contribution_vs_payout_view, and profit_loss_view. This over-emphasizes value movement and can hide the actual root-cause evidence.
- Source-required local questions require a compact two-hop evidence shape: packet views first locate the concrete callee/function/evidence_id, then Judge may request a precise evidence row/context and read_function_chunk for that function if source semantics are needed.
- Do not rely on call:0, transaction_root, or a top-level entry selector as the source target when a more specific critical_call_view, value_release_view, unknown_selector_view, or child trace call row can identify the relevant function.
- Do not hard-code this source policy to a particular condition id. Apply it to any judge step whose local question depends on function-level authorization, validation, visibility, selector meaning, storage-slot semantics, or source-level formulas.
- For source-required steps, keep first-pass locating evidence compact: operation_summary_view, evidence_adequacy_view, classification_digest_view, trace_outline_view, critical_call_view, and unknown_selector_view when selector semantics are relevant. Put value/state/detail views in allowed_followup_views unless the local question explicitly needs caller or authorization identity as first-pass evidence. Include read_function_chunk in allowed_tools with max_followups=2.
- For source-helpful but not source-required steps, do not force read_function_chunk and keep max_followups <= 1.
- Source evidence is a follow-up enhancer, not the only basis for judgment. The plan should give the judge enough packet evidence to make a fallback judgment if verified source is unavailable.
- For value-release, repeated payout, recipient, withdraw/redeem/borrow/claim/mint outcome, or protocol-balance-loss conditions, include value_release_view with participant_net_delta_view and contribution_vs_payout_view.
- If Transaction context summary contains planner_guidance from prior reviews, use it only as temporary planning guidance. It may change default_evidence_refs, allowed_followup_views, allowed_tools, or judge question clarity, but it must not be copied into the semantic rule or treated as ground truth.
- Do not create focus_steps.
- Do not reference legacy environment tool names; only allowed_tools may contain read-only evidence tool names.
- Do not reference environment.
- Do not ask "is this a {label} attack?"
- Do not include concrete tx hash, address, call id, transfer id, event id, or storage slot.
- Each judge_step should ask exactly one local yes/no question derived from one rule condition.
- Each judge_step should include only packet views needed for that condition.
- Round-0 evidence has a hard breadth budget. Keep default_evidence_refs compact: ordinary <= {ORDINARY_MAX_DEFAULT_VIEWS}, source-required/state-order <= {SPECIAL_MAX_DEFAULT_VIEWS}, and at most one large view. Put extra broad trace/state/event/fundflow detail in allowed_followup_views.
- Large packet views in the first prompt are compact projections. Use them to locate concrete evidence rows, then use follow-up reads when the local condition needs full row detail.
- Use one judge step per existing rule condition/exclusion.
- Do not split one rule condition into multiple judge steps.
- Do not add extra judge_steps beyond the rule conditions.
- State contracts are runtime-owned for labels with deterministic bindings.
  Leave depends_on, consumes_state_keys, produces_state_key,
  state_prompt_role, and state_output_schema empty in generated steps; the
  compiler will inject the canonical producer/consumer chain after parsing.
- If the rule has too many conditions, do not solve that here; preserve rule ids and compile the existing rule.
- Include exclusion judge steps when rule has exclusion_conditions.
- evidence_refs is kept for compatibility and should match default_evidence_refs.
- default_evidence_refs are the packet views sent to Judge on round 0.
- allowed_followup_views are the only packet views Judge may request later.
- View descriptions are provided by the packet view catalog; Judge does not need to call list_packet_views.
- Runtime uses a budget-aware renderer. You do not need to solve prompt budget by ordering views manually, but include compact structural/signal views such as operation_summary_view, evidence_adequacy_view, classification_digest_view, trace_outline_view, critical_call_view, reentrancy_state_order_summary_view, unknown_selector_view, and value_release_view when they are semantically relevant. Broad or bulky views such as trace_view, full reentrancy_state_order_view, state_change_view, external_fundflow_view, transfer_event_view, and profit_loss_view can be placed in allowed_followup_views when useful but not essential for the first-pass local judgment.
- allowed_tools should normally include read_packet_view, read_evidence_by_id, read_evidence_context, and get_local_call_context.
- Include read_function_chunk only for source-required conditions that need source semantics, selector meaning, storage-slot semantics, validation logic, or source-level formulas; do not add it merely because source might be helpful.
- When read_function_chunk is allowed, the Judge may call it either with explicit chain/address/function_name or with a concrete evidence_id from a call-like packet row; Runtime will resolve the row to callee/function when possible.
- get_local_call_context retrieves parent/children calls and subtree events/state for a specific call by its display id. Include it for conditions involving call structure or execution flow.
- For an access-control authorization-anchor condition, use
  get_local_call_context when a concrete unknown-selector or state-writing call
  remains semantically unresolved after compact views. Treat that call as an
  unresolved evidence probe, not as a satisfying authority-bearing candidate.
- For an access-control authorization-gap condition, use candidate-local
  context or source evidence to establish authority provenance and same-chain
  control. A modifier or permission query alone is not enough unless the
  pre-existing/delegated controller and the guarded entry-to-sensitive chain are
  both evidenced.
- Do not include list_packet_views in allowed_tools.
- allowed_tools must be chosen from: read_packet_view, read_evidence_by_id, read_evidence_context, get_local_call_context, read_function_chunk.
- Use max_followups=0 for one-shot judging and max_followups=1 for ordinary evidence expansion. Use max_followups=2 only when the second round semantically depends on the first observation, such as precise-row then source/function lookup, positive reentrancy state-order row then local context, or access_control/insufficient_validation core row then semantic verification. Wanting two independent views is not a sequential dependency; one round may request up to two tools.
- emit_logic should require all positive condition step ids and negate exclusion step ids.
- By default emit_logic will be compiled as all positive condition ids AND NOT any exclusion ids. Do not weaken this to OR logic.
- emit_logic must be a plain boolean expression over judge step ids only.
- Exclusion steps should ask whether a benign explanation exists, should use expected_answer=true, and should be negated in emit_logic.
- Exclusion questions MUST be concrete and evidence-based: ask whether specific observable facts exist (e.g., "Does the trace show ONLY standard protocol callbacks such as DEX swap callbacks, flash-loan callbacks, or token receiver hooks, with NO re-entry into any sensitive value-releasing function?").
- Exclusion questions MUST NOT use subjective comparative phrasing like "better explained as", "more consistent with", "could be interpreted as", or "is likely". These phrasings allow the judge to ignore concrete evidence and produce logically contradictory answers.
- Each exclusion question must be independently answerable from its own evidence views without depending on other judge steps' answers.
- Do not use function calls, list syntax, comparisons, arithmetic, or Python helpers such as all(...), any(...), len(...), sum(...).
- Valid examples: "C1 and C2 and C3", "(C1 and C2) and not E1", "(C1 and C2 and C3) and not (E1 or E2)".
- Runtime treats uncertain as false in emit_logic.

Input rule:
{stable_json_dumps(rule.to_dict())}

Transaction context summary:
{stable_json_dumps(tx_context_summary)}

Return strict JSON only:
{{
  "plan_id": "...",
  "rule_id": "{rule.rule_id}",
  "rule_version": {rule.version},
  "focus_steps": [],
  "judge_steps": [
    {{
      "id": "C1",
      "question": "Based only on the provided evidence, is this local condition satisfied: ...",
      "evidence_refs": ["operation_summary_view", "evidence_adequacy_view", "trace_outline_view", "critical_call_view"],
      "default_evidence_refs": ["operation_summary_view", "evidence_adequacy_view", "trace_outline_view", "critical_call_view"],
      "allowed_followup_views": ["trace_view", "state_change_view"],
      "allowed_tools": ["read_packet_view", "read_evidence_by_id", "read_evidence_context", "get_local_call_context"],
      "max_followups": 1,
      "expected_answer": true,
      "condition_id": "C1",
      "depends_on": [],
      "consumes_state_keys": [],
      "produces_state_key": "",
      "state_prompt_role": "",
      "state_output_schema": {{}}
    }}
  ],
  "emit_logic": "C1 and C2 and C3 and not (E1 or E2)",
  "plan_note": "Packet-view plan generated from semantic evolving rule.",
  "metadata": {{
    "generator": "llm_packet_plan",
    "attack_label": "{label}"
  }}
}}
"""

    @staticmethod
    def _to_plan(
        data: Dict[str, Any],
        rule: EvolvingRule,
        tx_context_summary: Optional[Dict[str, Any]] = None,
        *,
        enforce_label_plan_template: bool = True,
    ) -> EvidencePlan:
        raw_label = str((rule.metadata or {}).get("attack_label", ""))
        label = normalize_attack_label(raw_label)
        allow_emit_logic_change = _allow_emit_logic_change(
            tx_context_summary or {})
        raw_steps = [
            JudgeStep.from_dict(item) for item in data.get("judge_steps", [])
        ]
        raw_by_condition: Dict[str, JudgeStep] = {}
        for step in raw_steps:
            condition_id = str(step.condition_id or step.id).upper()
            if condition_id and condition_id not in raw_by_condition:
                raw_by_condition[condition_id] = step
        baseline = compile_rule_to_baseline_plan(rule)
        baseline_by_condition = {
            str(step.condition_id or step.id).upper(): step
            for step in baseline.judge_steps
        }
        ordered_condition_ids = [
            str(condition.id).upper()
            for condition in [*list(rule.conditions or []), *list(rule.exclusion_conditions or [])]
        ]
        judge_steps = []
        for condition_id in ordered_condition_ids:
            step = raw_by_condition.get(
                condition_id) or baseline_by_condition.get(condition_id)
            if step is None:
                continue
            judge_steps.append(JudgeStep.from_dict(step.to_dict()))
        for step in judge_steps:
            valid = [v for v in step.evidence_refs if v in _AVAILABLE_PACKET_VIEWS]
            step.evidence_refs = valid or list(_DEFAULT_PACKET_REFS)
            default_refs = [
                v for v in step.default_evidence_refs if v in _AVAILABLE_PACKET_VIEWS
            ]
            step.default_evidence_refs = default_refs or list(
                step.evidence_refs)
            step.allowed_followup_views = [
                v for v in step.allowed_followup_views if v in _AVAILABLE_PACKET_VIEWS
            ]
            step.allowed_tools = [
                t for t in step.allowed_tools if t in _ALLOWED_FOLLOWUP_TOOLS
            ]
            condition_id = str(step.condition_id or step.id).upper()
            iv_role = (
                _infer_insufficient_validation_condition_role(step)
                if label == "insufficient_validation"
                else ""
            )
            source_level = source_dependency_level(
                step.question, attack_label=label)
            source_required = source_level == "required"
            state_order_dependent = is_reentrant_state_order_text(
                step.question,
                attack_label=label,
            )

            if label == "access_control" and enforce_label_plan_template:
                _ensure_access_control_core_plan_policy(step)
            if label == "insufficient_validation" and enforce_label_plan_template:
                iv_role = _ensure_insufficient_validation_core_plan_policy(
                    step)
            if label == "price_manipulation" and enforce_label_plan_template:
                pm_role = _ensure_price_manipulation_core_plan_policy(step)
            else:
                pm_role = ""
            if label == "insufficient_validation":
                if iv_role == "validation_gap":
                    _ensure_iv_validation_gap_source_policy(step)
                elif source_required:
                    _ensure_source_followup_policy(
                        step,
                        attack_label=label,
                        source_required=True,
                        preserve_defaults=True,
                    )
                elif source_level == "helpful":
                    _ensure_source_helpful_followup_only(
                        step,
                        attack_label=label,
                    )
                else:
                    _ensure_basic_followup_tools(step)
                    step.max_followups = min(
                        max(1, int(step.max_followups or 0)), 1)
            else:
                if source_required:
                    _ensure_source_followup_policy(
                        step,
                        attack_label=label,
                        source_required=True,
                        preserve_defaults=True,
                    )
                elif source_level == "helpful":
                    _ensure_source_helpful_followup_only(
                        step,
                        attack_label=label,
                    )

            if state_order_dependent:
                # Keep source follow-up capability when it is semantically useful,
                # but let the state-order policy own the first prompt budget.
                if label == "insufficient_validation":
                    _ensure_iv_state_order_followup(step)
                else:
                    _ensure_state_order_followup_policy(step)

            if label == "access_control" and condition_id.startswith("C"):
                _ensure_label_core_source_policy(step)
            _ensure_role_minimum_views(
                step,
                label=label,
                condition_id=condition_id,
                iv_role=iv_role,
            )
            step.default_evidence_refs = _reorder_defaults_by_role(
                step.default_evidence_refs,
                label=label,
                condition_id=condition_id,
                iv_role=iv_role,
            )
            step.evidence_refs = list(step.default_evidence_refs)
            constrain_judge_step_first_pass(step, attack_label=label)
            round_policy = followup_round_policy(
                step,
                attack_label=label,
                condition_id=condition_id,
            )
            requested_followups = max(0, int(step.max_followups or 0))
            allowed_followups = int(round_policy["max_followups"])
            minimum_followups = int(round_policy.get("minimum_followups", 0) or 0)
            step.max_followups = max(
                minimum_followups,
                min(allowed_followups, requested_followups),
            )
            if not enforce_label_plan_template and condition_id in raw_by_condition:
                raw_step = raw_by_condition[condition_id]
                original_defaults = [
                    view
                    for view in (
                        raw_step.default_evidence_refs
                        or raw_step.evidence_refs
                        or []
                    )
                    if view in _AVAILABLE_PACKET_VIEWS
                ]
                original_followups = [
                    view
                    for view in list(raw_step.allowed_followup_views or [])
                    if view in _AVAILABLE_PACKET_VIEWS
                ]
                if original_defaults:
                    step.default_evidence_refs = list(original_defaults)
                    step.evidence_refs = list(original_defaults)
                step.allowed_followup_views = list(original_followups)
                step.max_followups = max(0, int(raw_step.max_followups or 0))
                step.view_budget = dict(raw_step.view_budget or {})
            step.id = condition_id
            step.condition_id = condition_id
            if condition_id.startswith("E") or step.id.upper().startswith("E"):
                step.expected_answer = True

        _ensure_label_dependencies(judge_steps, label=label)
        plan_quality_warnings = _collect_plan_quality_warnings(
            judge_steps, label=label)

        plan = EvidencePlan(
            plan_id=data.get("plan_id") or EvidencePlan.new_id(),
            rule_id=rule.rule_id,
            rule_version=rule.version,
            focus_steps=[],
            judge_steps=judge_steps,
            emit_logic=_safe_emit_logic(
                str(data.get("emit_logic", "")),
                judge_steps,
                allow_emit_logic_change=allow_emit_logic_change,
            ),
            plan_note=str(data.get("plan_note", "")),
            metadata={
                **dict(data.get("metadata", {})),
                "generator": data.get("generator", "llm"),
                "raw_attack_label": raw_label,
                "attack_label": label,
                "plan_quality_warnings": plan_quality_warnings,
                "plan_quality_warning_count": len(plan_quality_warnings),
                "plan_template_policy": {
                    "enforce_label_plan_template": bool(enforce_label_plan_template),
                    "stage": "plan_generator",
                    "label": label,
                },
            },
        )
        apply_insufficient_validation_stateful_runtime(
            plan,
            attack_label=label,
            preserve_view_policy=not enforce_label_plan_template,
        )
        apply_access_control_binding_mode(
            plan,
            rule=rule,
            mode="stateful",
        )
        apply_token_semantic_stateful_runtime(
            plan,
            rule=rule,
        )
        apply_protocol_accounting_stateful_runtime(
            plan,
            rule=rule,
        )
        apply_market_manipulation_stateful_runtime(
            plan,
            rule=rule,
        )
        apply_flashloans_stateful_runtime(
            plan,
            rule=rule,
        )
        apply_reentrancy_stateful_runtime(
            plan,
            rule=rule,
        )
        return sanitize_plan_judge_questions(
            plan,
            attack_label=(rule.metadata or {}).get("attack_label"),
        )


def _ensure_source_followup_policy(
    step: JudgeStep,
    *,
    attack_label: str = "",
    source_required: bool = True,
    preserve_defaults: bool = True,
) -> None:
    selector_semantics = _question_needs_selector_default(step)
    if _question_needs_identity_default(step, attack_label):
        locating_default_views = [
            "operation_summary_view",
            "evidence_adequacy_view",
            "address_labels",
            "beneficiary_controller_view",
            "critical_call_view",
            "source_unavailable_auth_view",
            "trace_outline_view",
        ]
        if selector_semantics:
            locating_default_views.append("unknown_selector_view")
    else:
        locating_default_views = [
            "operation_summary_view",
            "evidence_adequacy_view",
            "classification_digest_view",
            "trace_outline_view",
            "critical_call_view",
        ]
        if selector_semantics:
            locating_default_views.append("unknown_selector_view")
    locating_followup_views = [
        "trace_view",
        "critical_call_view",
        "unknown_selector_view",
        "source_unavailable_auth_view",
        "semantic_state_delta_view",
        "state_change_view",
        "value_release_view",
        "beneficiary_controller_view",
        "address_labels",
        "event_view",
        "transfer_event_view",
        "participant_net_delta_view",
        "contribution_vs_payout_view",
    ]
    default_limit = int(condition_view_budget(
        step, attack_label)["max_default_views"])
    if preserve_defaults:
        original_defaults = list(
            step.default_evidence_refs or step.evidence_refs or [])
        compact_defaults = _merge_known_views(
            original_defaults,
            locating_default_views,
            limit=default_limit,
        )
    else:
        compact_defaults = _merge_known_views(
            locating_default_views, limit=default_limit)
    step.default_evidence_refs = compact_defaults
    step.evidence_refs = list(compact_defaults)
    for view in locating_followup_views:
        if view in _AVAILABLE_PACKET_VIEWS and view not in step.allowed_followup_views:
            step.allowed_followup_views.append(view)
    for tool in (
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ):
        if tool not in step.allowed_tools:
            step.allowed_tools.append(tool)
    if source_required and "read_function_chunk" not in step.allowed_tools:
        step.allowed_tools.append("read_function_chunk")
    followup_limit = SOURCE_MAX_FOLLOWUP_VIEWS if source_required else ORDINARY_MAX_FOLLOWUP_VIEWS
    step.allowed_followup_views = _trim_known_views(
        step.allowed_followup_views, followup_limit)
    step.max_followups = max(2 if source_required else 1,
                             int(step.max_followups or 0))


def _ensure_source_helpful_followup_only(
    step: JudgeStep,
    *,
    attack_label: str = "",
) -> None:
    _ensure_basic_followup_tools(step)
    helpful_views = [
        "critical_call_view",
        "unknown_selector_view",
        "semantic_state_delta_view",
        "state_change_view",
        "event_view",
    ]
    step.allowed_followup_views = _merge_known_views(
        step.allowed_followup_views,
        helpful_views,
        limit=ORDINARY_MAX_FOLLOWUP_VIEWS,
    )
    step.max_followups = min(max(1, int(step.max_followups or 0)), 1)


def _ensure_basic_followup_tools(step: JudgeStep) -> None:
    for tool in (
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ):
        if tool not in step.allowed_tools:
            step.allowed_tools.append(tool)


def _append_tools(step: JudgeStep, tools: list[str]) -> None:
    for tool in tools:
        if tool not in step.allowed_tools:
            step.allowed_tools.append(tool)


def _ensure_role_minimum_views(
    step: JudgeStep,
    *,
    label: str,
    condition_id: str,
    iv_role: str = "",
) -> None:
    default_minimums: list[str] = []
    followup_minimums: list[str] = []
    if label == "insufficient_validation":
        if condition_id == "C1" or iv_role == "path_consumption":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "critical_call_view",
            ]
        elif condition_id == "C2" or iv_role == "validation_gap":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "critical_call_view",
                "semantic_state_delta_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "C3" or iv_role == "incorrect_outcome":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "value_release_view",
            ]
            followup_minimums = [
                "participant_net_delta_view",
                "contribution_vs_payout_view",
                "state_change_view",
                "beneficiary_controller_view",
            ]
        elif condition_id == "E1" or iv_role == "access_control_exclusion":
            followup_minimums = [
                "tx_card",
                "address_labels",
                "beneficiary_controller_view",
                "unknown_selector_view",
                "value_release_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "E2" or iv_role == "other_mechanism_exclusion":
            followup_minimums = [
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "reentrancy_state_order_summary_view",
                "transfer_event_view",
                "semantic_state_delta_view",
                "state_change_view",
            ]
    elif label == "access_control":
        if condition_id == "C1":
            default_minimums = ["tx_card", "operation_summary_view",
                                "critical_call_view", "trace_outline_view"]
        elif condition_id == "C2":
            default_minimums = [
                "tx_card",
                "address_labels",
                "critical_call_view",
                "critical_call_argument_view",
                "source_unavailable_auth_view",
                "unknown_selector_view",
                "beneficiary_controller_view",
            ]
            followup_minimums = [
                "source_unavailable_auth_view",
                "critical_call_argument_view",
                "unknown_selector_view",
                "beneficiary_controller_view",
                "semantic_state_delta_view",
                "state_change_view",
                "value_release_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "C3":
            default_minimums = [
                "tx_card",
                "value_release_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
            ]
        elif condition_id == "E1":
            default_minimums = ["tx_card", "address_labels",
                                "beneficiary_controller_view", "critical_call_view"]
        elif condition_id == "E2":
            default_minimums = [
                "tx_card",
                "operation_summary_view",
                "critical_call_view",
                "contribution_vs_payout_view",
                "value_release_view",
            ]
    elif label == "market_manipulation":
        if condition_id == "C1":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "critical_call_view",
            ]
            followup_minimums = [
                "market_mechanism_profile_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
                "transfer_event_view",
                "external_fundflow_view",
            ]
        elif condition_id == "C2":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "critical_call_view",
                "price_relevant_state_view",
            ]
            followup_minimums = [
                "market_mechanism_profile_view",
                "amm_reserve_transition_view",
                "event_view",
                "semantic_state_delta_view",
                "external_fundflow_view",
                "contribution_vs_payout_view",
            ]
        elif condition_id == "C3":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "value_release_view",
                "contribution_vs_payout_view",
            ]
            followup_minimums = [
                "market_mechanism_profile_view",
                "external_fundflow_view",
                "beneficiary_controller_view",
                "profit_loss_view",
                "transfer_event_view",
                "critical_call_view",
            ]
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "E1":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
            ]
            followup_minimums = [
                "market_mechanism_profile_view",
                "critical_call_view",
                "event_view",
                "external_fundflow_view",
                "contribution_vs_payout_view",
            ]
        elif condition_id == "E2":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
            ]
            followup_minimums = [
                "market_mechanism_profile_view",
                "transfer_event_view",
                "semantic_state_delta_view",
                "state_change_view",
                "critical_call_view",
                "external_fundflow_view",
            ]
    elif label == "protocol_accounting_exploitation":
        if condition_id == "C1":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "trace_outline_view",
                "critical_call_view",
                "reentrancy_state_order_summary_view",
            ]
            followup_minimums = [
                "state_change_view",
                "critical_call_argument_view",
                "event_view",
                "unknown_selector_view",
                "value_release_view",
                "contribution_vs_payout_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "C2":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
            ]
            followup_minimums = [
                "state_change_view",
                "value_release_view",
                "participant_net_delta_view",
                "event_view",
                "trace_outline_view",
                "unknown_selector_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "C3":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "semantic_state_delta_view",
            ]
            followup_minimums = [
                "state_change_view",
                "critical_call_view",
                "external_fundflow_view",
                "beneficiary_controller_view",
                "profit_loss_view",
                "trace_outline_view",
            ]
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "E1":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "contribution_vs_payout_view",
                "semantic_state_delta_view",
                "critical_call_view",
            ]
            followup_minimums = [
                "value_release_view",
                "participant_net_delta_view",
                "state_change_view",
                "critical_call_argument_view",
                "beneficiary_controller_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "E2":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
                "reentrancy_state_order_summary_view",
                "critical_call_view",
            ]
            followup_minimums = [
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "flash_or_atomic_capital_view",
                "semantic_state_delta_view",
                "state_change_view",
                "value_release_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
    elif label == "flashloans":
        if condition_id == "C1":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "trace_outline_view",
                "critical_call_view",
            ]
            followup_minimums = [
                "external_fundflow_view",
                "transfer_event_view",
                "event_view",
                "value_release_view",
                "participant_net_delta_view",
            ]
        elif condition_id == "C2":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "trace_outline_view",
                "critical_call_view",
                "semantic_state_delta_view",
            ]
            followup_minimums = [
                "critical_call_argument_view",
                "state_change_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "unknown_selector_view",
                "external_fundflow_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "C3":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
            ]
            followup_minimums = [
                "external_fundflow_view",
                "beneficiary_controller_view",
                "profit_loss_view",
                "transfer_event_view",
                "semantic_state_delta_view",
            ]
            step.max_followups = max(2, int(step.max_followups or 0))
        elif condition_id == "E1":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "external_fundflow_view",
                "contribution_vs_payout_view",
            ]
            followup_minimums = [
                "trace_outline_view",
                "critical_call_view",
                "value_release_view",
                "participant_net_delta_view",
                "profit_loss_view",
            ]
        elif condition_id == "E2":
            default_minimums = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "semantic_state_delta_view",
                "critical_call_view",
            ]
            followup_minimums = [
                "reentrancy_state_order_summary_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "state_change_view",
                "unknown_selector_view",
            ]
            _append_tools(step, ["read_function_chunk"])
            step.max_followups = max(2, int(step.max_followups or 0))
    default_limit = int(condition_view_budget(
        step, label)["max_default_views"])
    step.default_evidence_refs = _merge_known_views(
        step.default_evidence_refs or step.evidence_refs,
        default_minimums,
        limit=default_limit,
    )
    step.evidence_refs = list(step.default_evidence_refs)
    step.allowed_followup_views = _merge_known_views(
        step.allowed_followup_views,
        followup_minimums,
        limit=SOURCE_MAX_FOLLOWUP_VIEWS,
    )


def _reorder_defaults_by_role(
    views: list[str],
    *,
    label: str,
    condition_id: str,
    iv_role: str = "",
) -> list[str]:
    priority: list[str] = ["operation_summary_view", "evidence_adequacy_view"]
    if label == "insufficient_validation":
        if condition_id == "C1" or iv_role == "path_consumption":
            priority = ["operation_summary_view", "evidence_adequacy_view",
                        "trace_outline_view", "critical_call_view"]
        elif condition_id == "C2" or iv_role == "validation_gap":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "trace_outline_view",
                "critical_call_view",
                "unknown_selector_view",
            ]
        elif condition_id == "C3" or iv_role == "incorrect_outcome":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "value_release_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
            ]
        elif condition_id == "E1" or iv_role == "access_control_exclusion":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "tx_card",
                "address_labels",
                "beneficiary_controller_view",
                "critical_call_view",
            ]
        elif condition_id == "E2" or iv_role == "other_mechanism_exclusion":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "reentrancy_state_order_summary_view",
                "transfer_event_view",
                "semantic_state_delta_view",
            ]
    elif label == "access_control":
        if condition_id == "C1":
            priority = ["tx_card", "operation_summary_view", "evidence_adequacy_view",
                        "critical_call_view", "trace_outline_view", "value_release_view"]
        elif condition_id == "C2":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "source_unavailable_auth_view",
                "address_labels",
                "beneficiary_controller_view",
                "tx_card",
                "unknown_selector_view",
            ]
        elif condition_id == "C3":
            priority = ["tx_card", "operation_summary_view", "evidence_adequacy_view", "value_release_view",
                        "semantic_state_delta_view", "contribution_vs_payout_view", "participant_net_delta_view"]
        elif condition_id == "E1":
            priority = ["tx_card", "address_labels", "beneficiary_controller_view",
                        "critical_call_view", "operation_summary_view", "evidence_adequacy_view"]
        elif condition_id == "E2":
            priority = ["tx_card", "operation_summary_view", "evidence_adequacy_view",
                        "critical_call_view", "contribution_vs_payout_view", "value_release_view"]
    elif label == "market_manipulation":
        if condition_id == "C1":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "critical_call_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
            ]
        elif condition_id == "C2":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "critical_call_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "external_fundflow_view",
            ]
        elif condition_id == "C3":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "external_fundflow_view",
                "beneficiary_controller_view",
            ]
        elif condition_id == "E1":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "critical_call_view",
                "external_fundflow_view",
            ]
        elif condition_id == "E2":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "market_mechanism_profile_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
                "amm_reserve_transition_view",
                "transfer_event_view",
                "semantic_state_delta_view",
            ]
    elif label == "protocol_accounting_exploitation":
        if condition_id == "C1":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "trace_outline_view",
                "critical_call_view",
                "reentrancy_state_order_summary_view",
            ]
        elif condition_id == "C2":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
            ]
        elif condition_id == "C3":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "semantic_state_delta_view",
            ]
        elif condition_id == "E1":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "contribution_vs_payout_view",
                "semantic_state_delta_view",
                "critical_call_view",
            ]
        elif condition_id == "E2":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "token_semantic_delta_summary_view",
                "token_accounting_origin_view",
                "reentrancy_state_order_summary_view",
                "critical_call_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
            ]
    elif label == "flashloans":
        if condition_id == "C1":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "trace_outline_view",
                "critical_call_view",
                "external_fundflow_view",
                "transfer_event_view",
            ]
        elif condition_id == "C2":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "trace_outline_view",
                "critical_call_view",
                "semantic_state_delta_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
            ]
        elif condition_id == "C3":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "external_fundflow_view",
            ]
        elif condition_id == "E1":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "external_fundflow_view",
                "contribution_vs_payout_view",
                "value_release_view",
            ]
        elif condition_id == "E2":
            priority = [
                "operation_summary_view",
                "evidence_adequacy_view",
                "flash_or_atomic_capital_view",
                "semantic_state_delta_view",
                "critical_call_view",
                "reentrancy_state_order_summary_view",
                "token_semantic_delta_summary_view",
                "price_relevant_state_view",
            ]
    return _merge_known_views(priority, views)


def _ensure_label_dependencies(judge_steps: list[JudgeStep], *, label: str) -> None:
    if label != "insufficient_validation":
        return
    by_id = {str(step.condition_id or step.id).upper()             : step for step in judge_steps}
    if "C2" in by_id and "C1" in by_id:
        if "C1" not in by_id["C2"].depends_on:
            by_id["C2"].depends_on.append("C1")
    if "C3" in by_id:
        for dep in ("C1", "C2"):
            if dep in by_id and dep not in by_id["C3"].depends_on:
                by_id["C3"].depends_on.append(dep)


def _collect_plan_quality_warnings(
    judge_steps: list[JudgeStep],
    *,
    label: str,
) -> list[dict]:
    warnings: list[dict] = []
    groups: Dict[tuple[str, ...], list[str]] = {}
    for step in judge_steps:
        key = tuple(step.default_evidence_refs or [])
        groups.setdefault(key, []).append(str(step.condition_id or step.id))
    for refs, ids in groups.items():
        if len(ids) >= 2:
            warnings.append({
                "type": "duplicate_default_views",
                "steps": ids,
                "default_evidence_refs": list(refs),
            })
        if len(ids) >= 3:
            warnings.append({
                "type": "excessive_duplicate_defaults",
                "steps": ids,
                "default_evidence_refs": list(refs),
            })
    by_id = {str(step.condition_id or step.id).upper()             : step for step in judge_steps}
    if label == "price_manipulation":
        c1 = by_id.get("C1")
        c2 = by_id.get("C2")
        c3 = by_id.get("C3")

        price_views = {"price_relevant_state_view",
                       "amm_reserve_transition_view"}
        structure_views = {"critical_call_view", "trace_outline_view"}
        outcome_views = {
            "value_release_view",
            "participant_net_delta_view",
            "contribution_vs_payout_view",
            "external_fundflow_view",
        }

        if c1 and not (price_views & set(c1.default_evidence_refs)):
            warnings.append({
                "type": "price_manipulation_missing_price_source_defaults",
                "step": "C1",
            })

        if c2 and not (
            (price_views & set(c2.default_evidence_refs))
            and (structure_views & set(c2.default_evidence_refs))
            and "critical_call_argument_view" in set(c2.default_evidence_refs)
        ):
            warnings.append({
                "type": "price_manipulation_missing_consumption_link_defaults",
                "step": "C2",
            })

        if c3 and not (outcome_views & set(c3.default_evidence_refs)):
            warnings.append({
                "type": "price_manipulation_missing_outcome_defaults",
                "step": "C3",
            })
    if label == "insufficient_validation":
        c2 = by_id.get("C2")
        c3 = by_id.get("C3")
        if c2 and "semantic_state_delta_view" not in c2.default_evidence_refs:
            warnings.append(
                {"type": "insufficient_validation_missing_validation_gap_defaults", "step": "C2"})
        if c3 and not ({"semantic_state_delta_view", "value_release_view"} & set(c3.default_evidence_refs)):
            warnings.append(
                {"type": "insufficient_validation_missing_outcome_defaults", "step": "C3"})
    if label == "access_control":
        c2 = by_id.get("C2")
        c3 = by_id.get("C3")
        if c2 and not ({"address_labels", "beneficiary_controller_view"} & set(c2.default_evidence_refs)):
            warnings.append(
                {"type": "access_control_missing_identity_defaults", "step": "C2"})
        if c2 and "source_unavailable_auth_view" not in set(c2.default_evidence_refs):
            warnings.append({
                "type": "access_control_missing_source_unavailable_auth_context",
                "step": "C2",
            })
        if c3 and not ({"value_release_view", "semantic_state_delta_view"} & set(c3.default_evidence_refs)):
            warnings.append(
                {"type": "access_control_missing_effect_defaults", "step": "C3"})
    return warnings


def _ensure_iv_validation_gap_source_policy(step: JudgeStep) -> None:
    for tool in (
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
        "read_function_chunk",
    ):
        if tool not in step.allowed_tools:
            step.allowed_tools.append(tool)
    for view in (
        "unknown_selector_view",
        "semantic_state_delta_view",
        "event_view",
        "state_change_view",
        "value_release_view",
        "participant_net_delta_view",
    ):
        if view in _AVAILABLE_PACKET_VIEWS and view not in step.allowed_followup_views:
            step.allowed_followup_views.append(view)
    if _iv_question_mentions_state_order(step):
        view = "reentrancy_state_order_summary_view"
        if view in _AVAILABLE_PACKET_VIEWS and view not in step.allowed_followup_views:
            step.allowed_followup_views.append(view)
    step.max_followups = max(2, int(step.max_followups or 0))


def _ensure_iv_state_order_followup(step: JudgeStep) -> None:
    for view in (
        "reentrancy_state_order_summary_view",
        "critical_call_view",
        "trace_outline_view",
        "semantic_state_delta_view",
        "value_release_view",
    ):
        if view in _AVAILABLE_PACKET_VIEWS and view not in step.allowed_followup_views:
            step.allowed_followup_views.append(view)
    _ensure_basic_followup_tools(step)
    step.max_followups = max(1, int(step.max_followups or 0))


def _ensure_label_core_source_policy(step: JudgeStep) -> None:
    for tool in (
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
        "read_function_chunk",
    ):
        if tool not in step.allowed_tools:
            step.allowed_tools.append(tool)
    step.allowed_followup_views = _trim_known_views(
        step.allowed_followup_views,
        6,
    )
    step.max_followups = max(2, int(step.max_followups or 0))


def _infer_price_manipulation_condition_role(step: JudgeStep) -> str:
    cid = str(step.condition_id or step.id or "").upper()
    text = f"{step.question} {step.condition_id or ''} {step.id or ''}".lower()

    if cid == "C1":
        return "price_source_perturbation"
    if cid == "C2":
        return "price_consumption"
    if cid == "C3":
        return "price_outcome"
    if cid == "E1":
        return "ordinary_market_exclusion"
    if cid == "E2":
        return "legitimate_activity_exclusion"

    if any(k in text for k in (
        "perturb", "distort", "manipulat", "reserve", "oracle",
        "price source", "exchange-rate", "exchange rate",
        "share price", "lp price", "vault balance", "pool balance",
        "tick", "sqrtprice", "sync"
    )):
        return "price_source_perturbation"

    if any(k in text for k in (
        "consume", "depends on", "depend", "borrow", "mint",
        "redeem", "withdraw", "liquidate", "collateral",
        "valuation", "share issuance", "reward calculation",
        "exchange-rate update"
    )):
        return "price_consumption"

    if any(k in text for k in (
        "profit", "loss", "outcome", "payout", "release",
        "undercollateralized", "abnormal reward", "inflated",
        "adverse accounting", "protocol-adverse"
    )):
        return "price_outcome"

    if any(k in text for k in (
        "arbitrage", "ordinary large swap", "normal liquidation",
        "only closed-loop", "no protocol-sensitive"
    )):
        return "ordinary_market_exclusion"

    if any(k in text for k in (
        "authorized", "governance", "admin", "keeper",
        "oracle maintenance", "legitimate", "capital-backed",
        "repayment", "deposit", "collateral"
    )):
        return "legitimate_activity_exclusion"

    return "unknown"


def _infer_insufficient_validation_condition_role(step: JudgeStep) -> str:
    condition_id = str(step.condition_id or step.id or "").strip().upper()
    text = f"{step.question} {step.condition_id or ''} {step.id or ''}".lower()
    if condition_id == "C1":
        return "path_consumption"
    if condition_id == "C2":
        return "validation_gap"
    if condition_id == "C3":
        return "incorrect_outcome"
    if condition_id.startswith("E") and any(
        keyword in text
        for keyword in (
            "access",
            "caller",
            "owner",
            "role",
            "whitelist",
            "allowlist",
            "authorization",
            "callback-sender",
            "callback sender",
        )
    ):
        return "access_control_exclusion"
    if condition_id.startswith("E") and any(
        keyword in text
        for keyword in (
            "market",
            "oracle",
            "reentrancy",
            "token",
            "accounting",
            "price",
        )
    ):
        return "other_mechanism_exclusion"
    if any(
        keyword in text
        for keyword in (
            "consume",
            "input",
            "callback data",
            "return value",
            "oracle",
            "external state",
            "business assumption",
        )
    ):
        return "path_consumption"
    if any(
        keyword in text
        for keyword in (
            "invalid",
            "stale",
            "inconsistent",
            "boundary",
            "precision",
            "malformed",
            "fake",
            "validation",
            "check",
            "verify",
            "guard",
        )
    ):
        return "validation_gap"
    if any(
        keyword in text
        for keyword in (
            "outcome",
            "payout",
            "release",
            "loss",
            "corruption",
            "invariant",
            "protocol loss",
            "user loss",
        )
    ):
        return "incorrect_outcome"
    return "unknown"


def _iv_question_mentions_state_order(step: JudgeStep) -> bool:
    text = str(step.question or "").lower().replace("_", " ").replace("-", " ")
    return any(
        keyword in text
        for keyword in (
            "reentrancy",
            "reentrant",
            "nested execution",
            "stale nested",
            "state order",
            "callback order",
        )
    )


def _ensure_insufficient_validation_core_plan_policy(step: JudgeStep) -> str:
    role = _infer_insufficient_validation_condition_role(step)
    base_tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ]
    templates = {
        "path_consumption": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "semantic_state_delta_view",
            ],
            [
                "trace_outline_view",
                "unknown_selector_view",
                "event_view",
                "state_change_view",
                "value_release_view",
            ],
            base_tools,
            1,
        ),
        "validation_gap": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
            ],
            [
                "trace_outline_view",
                "unknown_selector_view",
                "semantic_state_delta_view",
                "event_view",
                "state_change_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
            ],
            [*base_tools, "read_function_chunk"],
            2,
        ),
        "incorrect_outcome": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "semantic_state_delta_view",
                "value_release_view",
            ],
            [
                "critical_call_view",
                "trace_outline_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
                "unknown_selector_view",
                "state_change_view",
                "beneficiary_controller_view",
            ],
            base_tools,
            1,
        ),
        "access_control_exclusion": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "critical_call_view",
                "critical_call_argument_view",
                "source_unavailable_auth_view",
                "beneficiary_controller_view",
            ],
            [
                "trace_outline_view",
                "tx_card",
                "address_labels",
                "unknown_selector_view",
                "value_release_view",
                "semantic_state_delta_view",
                "trace_view",
            ],
            [*base_tools, "read_function_chunk"],
            2,
        ),
        "other_mechanism_exclusion": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "critical_call_view",
            ],
            [
                "unknown_selector_view",
                "semantic_state_delta_view",
                "event_view",
                "state_change_view",
                "value_release_view",
                "beneficiary_controller_view",
                "trace_view",
            ],
            base_tools,
            1,
        ),
    }
    if role not in templates:
        return role
    defaults, followups, tools, max_followups = templates[role]
    if role == "validation_gap" and _iv_question_mentions_state_order(step):
        followups = [*followups, "reentrancy_state_order_summary_view"]
    step.default_evidence_refs = [
        view for view in defaults if view in _AVAILABLE_PACKET_VIEWS
    ]
    step.evidence_refs = list(step.default_evidence_refs)
    step.allowed_followup_views = [
        view for view in followups if view in _AVAILABLE_PACKET_VIEWS
    ]
    step.allowed_tools = [
        tool for tool in tools if tool in _ALLOWED_FOLLOWUP_TOOLS]
    step.max_followups = max(max_followups, int(step.max_followups or 0))
    return role


def _ensure_price_manipulation_core_plan_policy(step: JudgeStep) -> str:
    role = _infer_price_manipulation_condition_role(step)

    base_tools = [
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ]

    templates = {
        "price_source_perturbation": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "event_view",
                "critical_call_view",
            ],
            [
                "trace_outline_view",
                "semantic_state_delta_view",
                "transfer_event_view",
                "state_change_view",
                "trace_view",
            ],
            base_tools,
            1,
        ),
        "price_consumption": (
            [
                "market_mechanism_profile_view",
                "price_relevant_state_view",
                "critical_call_view",
                "critical_call_argument_view",
            ],
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "amm_reserve_transition_view",
                "semantic_state_delta_view",
                "event_view",
                "transfer_event_view",
                "state_change_view",
                "trace_view",
                "value_release_view",
            ],
            base_tools,
            2,
        ),
        "price_outcome": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "price_relevant_state_view",
                "amm_reserve_transition_view",
                "value_release_view",
                "participant_net_delta_view",
                "contribution_vs_payout_view",
            ],
            [
                "external_fundflow_view",
                "transfer_event_view",
                "profit_loss_view",
                "beneficiary_controller_view",
                "semantic_state_delta_view",
            ],
            base_tools,
            2,
        ),
        "ordinary_market_exclusion": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "trace_outline_view",
                "critical_call_view",
                "amm_reserve_transition_view",
                "price_relevant_state_view",
            ],
            [
                "event_view",
                "transfer_event_view",
                "semantic_state_delta_view",
                "participant_net_delta_view",
            ],
            base_tools,
            1,
        ),
        "legitimate_activity_exclusion": (
            [
                "operation_summary_view",
                "evidence_adequacy_view",
                "address_labels",
                "beneficiary_controller_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
            ],
            [
                "trace_outline_view",
                "critical_call_view",
                "value_release_view",
                "external_fundflow_view",
            ],
            base_tools,
            1,
        ),
    }

    if role not in templates:
        return role

    defaults, followups, tools, max_followups = templates[role]
    step.default_evidence_refs = [
        v for v in defaults if v in _AVAILABLE_PACKET_VIEWS
    ]
    step.evidence_refs = list(step.default_evidence_refs)
    step.allowed_followup_views = [
        v for v in followups if v in _AVAILABLE_PACKET_VIEWS
    ]
    step.allowed_tools = [
        t for t in tools if t in _ALLOWED_FOLLOWUP_TOOLS
    ]
    step.max_followups = max(max_followups, int(step.max_followups or 0))
    return role


def _ensure_access_control_core_plan_policy(step: JudgeStep) -> None:
    condition_id = str(step.condition_id or step.id or "").strip().upper()
    templates = {
        "C1": (
            [
                "tx_card",
                "operation_summary_view",
                "classification_digest_view",
                "critical_call_view",
                "trace_outline_view",
                "value_release_view",
            ],
            [
                "unknown_selector_view",
                "semantic_state_delta_view",
                "value_release_view",
                "beneficiary_controller_view",
                "state_change_view",
                "contribution_vs_payout_view",
                "reentrancy_state_order_summary_view",
            ],
        ),
        "C2": (
            [
                "tx_card",
                "address_labels",
                "critical_call_view",
                "critical_call_argument_view",
                "source_unavailable_auth_view",
                "unknown_selector_view",
                "beneficiary_controller_view",
                "value_release_view",
            ],
            [
                "trace_outline_view",
                "source_unavailable_auth_view",
                "critical_call_argument_view",
                "unknown_selector_view",
                "beneficiary_controller_view",
                "value_release_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "semantic_state_delta_view",
                "state_change_view",
                "reentrancy_state_order_summary_view",
            ],
        ),
        "C3": (
            [
                "tx_card",
                "value_release_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "critical_call_view",
            ],
            [
                "trace_outline_view",
                "critical_call_argument_view",
                "state_change_view",
                "value_release_view",
                "semantic_state_delta_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "external_fundflow_view",
                "transfer_event_view",
                "beneficiary_controller_view",
            ],
        ),
        "E1": (
            [
                "tx_card",
                "address_labels",
                "beneficiary_controller_view",
                "critical_call_view",
            ],
            [
                "trace_outline_view",
                "unknown_selector_view",
                "semantic_state_delta_view",
                "value_release_view",
            ],
        ),
        "E2": (
            [
                "tx_card",
                "operation_summary_view",
                "critical_call_view",
                "contribution_vs_payout_view",
                "participant_net_delta_view",
                "value_release_view",
            ],
            [
                "trace_outline_view",
                "semantic_state_delta_view",
                "unknown_selector_view",
                "state_change_view",
                "beneficiary_controller_view",
            ],
        ),
    }
    if condition_id not in templates:
        return
    defaults, followups = templates[condition_id]
    step.default_evidence_refs = [
        view for view in defaults if view in _AVAILABLE_PACKET_VIEWS
    ]
    step.evidence_refs = list(step.default_evidence_refs)
    for view in followups:
        if view in _AVAILABLE_PACKET_VIEWS and view not in step.allowed_followup_views:
            step.allowed_followup_views.append(view)
    if condition_id == "C1":
        step.allowed_tools = [
            tool
            for tool in step.allowed_tools
            if tool not in {"get_local_call_context", "read_function_chunk"}
        ]
    elif condition_id == "C2":
        for tool in ("get_local_call_context", "read_function_chunk"):
            if tool in _ALLOWED_FOLLOWUP_TOOLS and tool not in step.allowed_tools:
                step.allowed_tools.append(tool)
    elif condition_id == "C3":
        step.allowed_tools = [
            tool for tool in step.allowed_tools if tool != "read_function_chunk"
        ]


def _trim_known_views(views: list[str], limit: int) -> list[str]:
    result: list[str] = []
    for view in views:
        if view in _AVAILABLE_PACKET_VIEWS and view not in result:
            result.append(view)
        if len(result) >= max(0, int(limit or 0)):
            break
    return result


def _merge_known_views(*groups: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    for group in groups:
        for view in list(group or []):
            if view in _AVAILABLE_PACKET_VIEWS and view not in result:
                result.append(view)
                if limit is not None and len(result) >= max(0, int(limit or 0)):
                    return result
    return result


def _question_needs_selector_default(step: JudgeStep) -> bool:
    text = str(step.question or "").lower()
    return any(
        keyword in text
        for keyword in (
            "selector",
            "undecoded",
            "unknown function",
            "function semantics",
        )
    )


def _ensure_state_order_followup_policy(step: JudgeStep) -> None:
    compact_defaults = [
        "operation_summary_view",
        "evidence_adequacy_view",
        "trace_outline_view",
        "critical_call_view",
        "reentrancy_state_order_summary_view",
    ]
    if _question_needs_value_release_default(step):
        compact_defaults.append("value_release_view")
    broad_followups = [
        "reentrancy_state_order_view",
        "critical_call_view",
        "semantic_state_delta_view",
        "state_change_view",
        "unknown_selector_view",
        "trace_view",
        "event_view",
        "participant_net_delta_view",
        "contribution_vs_payout_view",
        "external_fundflow_view",
        "transfer_event_view",
        "value_release_view",
    ]
    step.default_evidence_refs = [
        view for view in compact_defaults if view in _AVAILABLE_PACKET_VIEWS
    ][:SPECIAL_MAX_DEFAULT_VIEWS]
    step.evidence_refs = list(step.default_evidence_refs)
    for view in broad_followups:
        if view in _AVAILABLE_PACKET_VIEWS and view not in step.allowed_followup_views:
            step.allowed_followup_views.append(view)
    for tool in (
        "read_packet_view",
        "read_evidence_by_id",
        "read_evidence_context",
        "get_local_call_context",
    ):
        if tool not in step.allowed_tools:
            step.allowed_tools.append(tool)
    step.allowed_followup_views = _trim_known_views(
        step.allowed_followup_views,
        STATE_ORDER_MAX_FOLLOWUP_VIEWS,
    )
    step.max_followups = max(1, int(step.max_followups or 0))


def _question_needs_identity_default(step: JudgeStep, attack_label: str = "") -> bool:
    label = normalize_attack_label(attack_label)
    if label != "access_control":
        return False
    text = f"{step.question} {step.condition_id} {step.id}".lower()
    keywords = (
        "caller",
        "owner",
        "role",
        "admin",
        "whitelist",
        "allowlist",
        "authorized",
        "authorization",
        "permission",
        "callback sender",
        "callback caller",
        "beneficiary",
        "keeper",
    )
    return any(keyword in text for keyword in keywords)


def _question_needs_value_release_default(step: JudgeStep) -> bool:
    text = str(step.question or "").lower()
    keywords = (
        "payout",
        "release",
        "withdraw",
        "claim",
        "borrow",
        "mint",
        "redeem",
        "asset",
        "value",
        "profit",
        "loss",
    )
    return any(keyword in text for keyword in keywords)


def _allow_emit_logic_change(tx_context_summary: Dict[str, Any]) -> bool:
    if bool(tx_context_summary.get("allow_emit_logic_change")):
        return True
    guidance = tx_context_summary.get("planner_guidance")
    if isinstance(guidance, dict):
        return bool(guidance.get("allow_emit_logic_change"))
    if isinstance(guidance, list):
        return any(
            isinstance(item, dict) and bool(
                item.get("allow_emit_logic_change"))
            for item in guidance
        )
    return False


def _safe_emit_logic(
    raw_emit_logic: str,
    judge_steps: list[JudgeStep],
    *,
    allow_emit_logic_change: bool = False,
) -> str:
    """Keep LLM emit_logic executable; fall back to C... and not E... shape."""
    baseline = _baseline_emit_logic_from_steps(judge_steps)
    if not allow_emit_logic_change:
        return baseline
    judge_ids = {step.id for step in judge_steps}
    try:
        names = extract_logic_names(raw_emit_logic)
        if raw_emit_logic.strip() and not (names - judge_ids) and _mentions_all_step_ids(
            raw_emit_logic,
            judge_steps,
        ):
            return raw_emit_logic
    except UnsafeLogicExpression:
        pass
    return baseline


def _mentions_all_step_ids(raw_emit_logic: str, judge_steps: list[JudgeStep]) -> bool:
    try:
        names = extract_logic_names(raw_emit_logic)
    except UnsafeLogicExpression:
        return False
    required = {step.id for step in judge_steps}
    if not required.issubset(names):
        return False
    text = f" {raw_emit_logic.lower()} "
    exclusions = [
        step.id
        for step in judge_steps
        if str(step.condition_id or step.id).upper().startswith("E")
        or step.id.upper().startswith("E")
    ]
    if exclusions and " not " not in text:
        return False
    return True


def _baseline_emit_logic_from_steps(judge_steps: list[JudgeStep]) -> str:
    positive_ids = []
    exclusion_ids = []
    for step in judge_steps:
        condition_id = str(step.condition_id or step.id)
        if condition_id.upper().startswith("E") or step.id.upper().startswith("E"):
            exclusion_ids.append(step.id)
        else:
            positive_ids.append(step.id)

    positive_expr = " and ".join(positive_ids) if positive_ids else "False"
    exclusion_expr = " or ".join(exclusion_ids)
    if exclusion_expr:
        return f"({positive_expr}) and not ({exclusion_expr})"
    return positive_expr
