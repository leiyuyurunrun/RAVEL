from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import JudgeResult, normalize_condition_feature_analysis
from evotx.runtime.evidence_renderer import (
    format_evidence_context_with_metadata as renderer_format_evidence_context_with_metadata,
    truncate_text as renderer_truncate_text,
    truncate_text_with_removed as renderer_truncate_text_with_removed,
)
from evotx.runtime.followup_context import (
    FollowupPromptContext,
    normalize_followup_context_mode,
)
from evotx.runtime.llm_transcripts import LLMTranscriptRecorder
from evotx.runtime.source_tools import SourceToolRegistry
from evotx.runtime.view_catalog import packet_view_row_count
from evotx.utils.json_utils import (
    extract_last_schema_valid_object,
    stable_json_dumps,
    structured_output_text,
)


CONDITION_FEATURE_ANALYSIS_PROMPT = """
In addition to the final local judgment, always return condition_feature_analysis.

condition_feature_analysis is diagnostic metadata for rule evolution only.
It must not change the answer. A partial match is not sufficient to answer true.

Definitions:
- matched_features: concise abstract features from the evidence that support satisfying this local condition.
- partial_match_features: features that resemble or are close to the condition but do not fully satisfy it.
- missing_required_features: required features that are absent or unresolved; these explain why answer is false or uncertain.
- contradicting_features: evidence-backed features that argue against satisfying this condition.
- boundary_notes: concise notes about target-vs-non-target boundary risks, weak evidence, generic symptoms, or ambiguity.

Rules:
- condition_feature_analysis is required in every final answer.
- Do not leave all five lists empty unless there is truly no relevant evidence.
- For answer=true, matched_features should usually be non-empty.
- For answer=false, at least one of partial_match_features, missing_required_features, or contradicting_features should usually be non-empty.
- For answer="uncertain", missing_required_features should usually be non-empty, and partial_match_features should capture any observed near-miss signals.
- boundary_notes should capture generic symptoms, weak evidence, or target-vs-non-target ambiguity.
- condition_feature_analysis must use abstract semantic feature strings.
- Do NOT include concrete tx hashes, addresses, selectors, call ids, event ids, transfer ids, evidence ids, storage slots, sload/sstore ids, exact token amounts, or chain-specific identifiers in feature strings.
- Concrete identifiers must only appear in supporting_evidence_ids, contradicting_evidence_ids, tool_requests, or reason, not in condition_feature_analysis.
- Rewrite concrete observations into abstract phrases:
  - selector values such as 0x... -> "unknown/obfuscated selector"
  - call ids such as call:0 -> "entry call" or "root-like entry call"
  - sload/sstore ids -> "early storage reads with unresolved authorization semantics" or "state writes with unresolved semantics"
  - concrete addresses -> "unlabeled caller", "target contract", "protocol-controlled contract", or "beneficiary"
  - exact token amounts -> "large multi-token value release", "zero visible contribution", or "disproportionate payout"
  - nonce values -> "fresh caller account"
- Cite concrete evidence using supporting_evidence_ids / contradicting_evidence_ids, not inside feature strings.
- Keep each list short, preferably 0-5 items.
- If answer is false or uncertain, try to fill partial_match_features and missing_required_features.
- If answer is true, fill matched_features and boundary_notes when there are weak or generic signals.
- Do not set answer=true merely because partial_match_features is non-empty.
- partial_match_features means "close but insufficient", not "satisfied".
"""


MINIMAX_JSON_REPAIR_MAX_TOKENS = 2048
GLM_EXHAUSTED_RETRY_DEFAULT_MAX_TOKENS = 32768


CONDITION_FEATURE_ANALYSIS_SCHEMA_FIELD = '''  "condition_feature_analysis": {
    "matched_features": [],
    "partial_match_features": [],
    "missing_required_features": [],
    "contradicting_features": [],
    "boundary_notes": []
  }'''


CONDITION_FEATURE_ANALYSIS_TOOL_REQUEST_EXAMPLE_FIELD = '''  "condition_feature_analysis": {
    "matched_features": [],
    "partial_match_features": [
      "observed value-sensitive operation but beneficiary or entitlement relation is unresolved"
    ],
    "missing_required_features": [
      "need value-release recipient and contribution-vs-payout evidence"
    ],
    "contradicting_features": [],
    "boundary_notes": [
      "near-match evidence should not be treated as satisfied before follow-up"
    ]
  }'''


def _normalize_state_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _state_output_schema_field(state_prompt_role: str) -> str:
    if not str(state_prompt_role or "").strip():
        return ""
    return ',\n  "state_output": {}'


def _stateful_runtime_metadata(
    *,
    state_prompt_role: str = "",
    consumes_state_keys: Optional[List[str]] = None,
    produces_state_key: str = "",
) -> Dict[str, Any]:
    role = str(state_prompt_role or "")
    return {
        "enabled": bool(role or consumes_state_keys or produces_state_key),
        "state_prompt_role": role,
        "consumes_state_keys": list(consumes_state_keys or []),
        "produces_state_key": str(produces_state_key or ""),
    }


def _stateful_runtime_prompt_section(
    *,
    state_input_context: Optional[Dict[str, Any]] = None,
    state_prompt_role: str = "",
    state_output_schema: Optional[Dict[str, Any]] = None,
) -> str:
    role = str(state_prompt_role or "").strip()
    if not role:
        return ""
    input_context = _normalize_state_dict(state_input_context)
    output_schema = _normalize_state_dict(state_output_schema)
    role_instructions = {
        "iv_object_binding": """
- Return state_output.object_binding_summary.
- Enumerate up to 3 value-sensitive consumed object candidates.
- Candidate objects may include user-supplied parameters, callback data, return values, token/path/market addresses, amount/reward/share/debt/collateral values, oracle/external state, business/accounting state, boundary/precision/rounding values, and entitlement or balance-sufficiency objects.
- Protocol-controlled business/accounting state is still an eligible candidate
  when it carries upstream external, user, oracle, reward, share, debt,
  collateral, entitlement, or cached-return provenance and later drives
  sensitive logic. Do not discard it merely because the value is stored inside
  protocol storage or the final consumer has a local guard.
- For each candidate, include object_id, object_name, object_type, provenance,
  consumer_path, required_validation_dimensions, validation_observation, and
  evidence_ids when available. Dimensions may include source identity/allowlist,
  range, proportionality, freshness, format, signer/domain, return consistency,
  callback content, or a named business invariant.
- Do not decide final validation adequacy here.
- If an object appears validated, you may still list it as a candidate and
  describe that as validation_observation; C2 decides whether validation is
  present, inadequate, bypassed, stale, or missing.
""",
        "iv_validation_gap": """
- Read state_input_context.object_binding_summary.
- Assess the bounded candidates before selecting one. Return one
  candidate_assessments row per object_id with validation_status, validation_gap,
  compensating_guards_considered, evidence_ids, and selection_rationale.
- Select one candidate object or mark the selection uncertain only after the
  per-candidate audit.
- Judge missing, inadequate, bypassed, stale, wrong_scope, present,
  standard_design, guarded, or uncertain validation on that same object.
- For each candidate_assessments row, state required_validation_dimensions,
  observed_check_scope, and scope_match. A guard counts only for the exact
  property it checks.
- Caller/owner/role checks establish entitlement, nonReentrant establishes
  execution locking, whenNotPaused establishes operating state, balance and
  allowance establish funds/approval, and isContract establishes code presence.
  None of those automatically validates reward/accounting proportionality,
  token/path/market/strategy allowlisting, callback payload content, oracle
  freshness, external-return consistency, or another business invariant.
- Do not inherit a wrapper guard as validation for every internal state object.
  If the wrapper delegates an unresolved candidate to a named internal callee,
  request that target_function before answering false while budget remains.
- A false assessment for a candidate requires positive exact-scope validation
  evidence for that candidate. Keep unresolved alternatives explicit and request
  candidate-specific evidence; do not turn an unrelated low-ranked alternative
  into the selected gap merely because it is unresolved.
- If the first or most visible candidate is present, guarded, or standard_design,
  pivot to the remaining C1 candidates before answering false. In particular,
  inspect decoded callback data, token/path/market addresses, amount/allowance
  parameters, signature/domain bindings, external returns, and business state
  that feed the same sensitive path.
- Do not switch to an object absent from C1. When packet evidence exposes a
  plausible candidate but cannot establish its source-level check, keep it
  uncertain and request candidate-anchored evidence instead of treating a
  different guarded object as dispositive.
- Decide validation adequacy only. Do not decide whether Access Control,
  Reentrancy, market, token, or accounting behavior is the primary replacement
  mechanism; the corresponding exclusion owns that decision.
- Return state_output.validation_gap_summary.
""",
        "iv_causal_outcome": """
- Read state_input_context.object_binding_summary and state_input_context.validation_gap_summary.
- Judge whether the selected object and selected validation gap causally drive the incorrect outcome.
- Do not answer true merely because there is profit, value release, a flash loan, mint, swap, or callback.
- Return state_output.causal_outcome_summary.
""",
        "iv_access_control_exclusion_with_core_state": """
- Read the same object_binding_summary, validation_gap_summary, and
  causal_outcome_summary produced by C1-C3.
- Treat the upstream selected object as a hypothesis, not an established root
  cause. Independently compare every C1 candidate and the transaction-visible
  authorization path. Explicitly challenge an oracle, callback, amount, or
  accounting narrative when caller/principal identity, self-granted authority,
  signature scope, delegation, or protected-capability evidence better explains
  the outcome.
- Distinguish authorization of a subject to exercise a protected capability
  from validation of a consumed data/state/source/address/payload object.
- Source/token/path/market allowlists, contract-address registration,
  signature/message/domain binding, format, bounds, and consistency checks are
  object validation unless they directly authorize the caller or principal to
  exercise the protected capability.
- A caller/owner storage read is not proof of effective authorization unless it
  guards the actual sensitive operation, uses a pre-existing legitimate
  authority, and was not established or controlled by the attacker in the same
  causal chain.
- Return state_output.access_control_exclusion_summary using the supplied
  schema. Preserve the upstream selected_object_id for audit, but also report
  challenged_selected_object, competing_candidate_assessments, and the
  primary_mechanism_candidate_id used for the exclusion decision.
""",
        "ac_authorization_anchor": """
- Return state_output.access_control_candidate.
- Select one primary transaction-visible candidate. Keep up to two alternatives
  only when concrete evidence supports genuinely distinct candidate chains.
- Each candidate must bind candidate_id, actor or beneficiary, sensitive call
  or capability, protected target/resource, effect hint, and evidence_ids.
- A single sensitive call is a valid anchor. When entry and sensitive operation
  are different calls, also preserve their call ids/path relation; do not
  require a parent/child pair when the same call carries the capability.
- If source or ABI is unavailable, do not reject a candidate solely because a
  selector is unknown. A transaction-visible unknown/undecoded selector may be
  a candidate only when it is trace-linked to privileged state writes,
  approval/role/configuration changes, mint/burn/withdraw/redeem effects, or
  release of protocol-controlled/shared/victim assets.
- Use candidate_status=candidate_anchor when the concrete chain is located, or
  unresolved_probe when its exact semantics still need bounded follow-up. Both
  statuses remain available to C2; status does not decide whether authorization
  is required, missing, or legitimately permissionless.
- Ordinary ERC-20 transfer/transferFrom, AMM liquidity mint/burn, user
  withdrawal/redemption, and generic value movement are not candidates by name
  alone. Preserve them only when the concrete path also exposes a plausible
  protected target/resource, protocol-controlled value, configuration/state
  effect, or authorization-relevant entry relation. C2/E1 decide entitlement
  and permissionlessness.
- Preserve a transaction-visible externally controlled call chain when caller
  supplied payload or recipient arguments are consumed by a contract holding
  the affected asset and the same chain routes that asset to a different
  beneficiary. Do not discard that chain merely because a downstream call is
  named swap, transfer, withdraw, redeem, or mint.
- Assign a stable, concise candidate_id and keep candidates distinct by their
  actor/capability/target chain.
- Do not decide whether authorization is required, missing, or legitimate in
  this step.
""",
        "ac_authorization_gap": """
- Read state_input_context.access_control_candidate.
- Evaluate every listed candidate separately and preserve its candidate_id.
- For each candidate, judge the required authority and whether it is missing,
  bypassed, self-granted, unauthorized, present, or uncertain for that exact
  actor/capability/target chain.
- For each candidate also set authorization_evidence_status to one of:
  source_confirmed_missing, missing_behavioral, present, contradicted,
  source_required, or uncertain.
- Record authority source/provenance when the evidence establishes it, but do
  not make a complete provenance taxonomy mandatory before deciding a concrete
  missing or present authorization boundary.
- Treat a missing-check subpattern as candidate-local: public/external access,
  missing owner/role/caller/callback validation, self-granted authority, or
  attacker-controlled beneficiary can support an authorization gap only when it
  guards the same sensitive capability identified by the candidate.
- source_required means packet evidence alone did not show a target-specific
  authorization gap. It must not be placed in satisfying_candidate_ids.
- A source-level modifier or check is relevant only when it guards the same
  candidate capability. Observed same-transaction self-authorization or bypass
  may contradict an otherwise familiar modifier.
- A downstream module, router, initialization-state, accounting, or other
  business guard proves authorization is present only when the evidence binds
  that guard to the candidate actor/principal's authority for the exact
  protected capability. A guard that only validates module readiness, object
  state, parameters, or downstream business invariants does not establish that
  the upstream actor was authorized to reach the capability.
- Conversely, the absence of a modifier or explicit check does not prove that
  an operation is intentionally permissionless. For a concrete operation over
  protocol-controlled assets, protected configuration, roles, approvals, or a
  privileged external-call path, absence of the required check may be
  source_confirmed_missing. Treat an operation as legitimately permissionless
  only when its user entitlement or public business semantics are positively
  established for the same capability and target.
- When a separate parent/dispatcher/module/proxy/callback is present, include it
  only if trace evidence connects it to authority selection or bypass.
- Missing validation or allowlisting of a user-supplied token, path, router,
  calldata, market, or payload is an object-validation issue, not an actor
  authorization gap, unless that exact check authorizes the actor or principal
  to exercise the candidate's protected capability.
- missing_behavioral may support the gap without source only when packet anchors
  are access-control-specific: owner/admin/role/permission/initializer/proxy/
  implementation/delegatecall/authorization-slot evidence, self-granted
  authority, caller-to-beneficiary mismatch on the same capability, or an
  unknown selector directly changing protected configuration/state or releasing
  protected value.
- source_unavailable_auth_view.transaction_state_anchors are factual trace
  anchors, not proof that a source guard is absent. Use them for
  missing_behavioral only when the same candidate chain visibly performs a
  protected configuration write, self-grant, authority bypass, or protected
  value release.
- Generic value movement, swaps, flash loans, callbacks, reentrancy, public
  entry, source_unavailable, or profit are not missing_behavioral by themselves.
- Do not switch to an unrelated suspicious call, actor, asset, or protocol component.
- When requesting follow-up evidence, prefer candidate evidence_ids,
  sensitive_call_id, path_ids, function, or target_contract from
  access_control_candidate over broad top-row scans.
- For source unavailable or source-present-but-authorization-provenance unclear
  cases, prefer candidate-filtered source_unavailable_auth_view,
  critical_call_argument_view, semantic_state_delta_view, state_change_view,
  beneficiary_controller_view, and value_release_view before broad scans. Use
  read_function_chunk on the candidate sensitive_call_id first when source can
  answer the guard/provenance question. Source for an entry parent is supplemental
  unless that parent directly selects or enforces the candidate authority.
- Set satisfying_candidate_ids only to candidates whose authorization_status is
  missing, bypassed, self_granted, or unauthorized, whose same_chain_supported
  is true, and whose authorization_evidence_status is source_confirmed_missing
  or missing_behavioral.
- Set unresolved_candidate_ids for candidates that remain uncertain.
- The top-level answer is true when satisfying_candidate_ids is non-empty,
  uncertain when no candidate satisfies but at least one remains unresolved,
  and false otherwise.
- Return state_output.authorization_gap_summary.
""",
        "ac_protected_effect": """
- Read state_input_context.access_control_candidate and
  state_input_context.authorization_gap_summary.
- Evaluate each satisfying_candidate_id and unresolved_candidate_id separately.
  Judge whether that exact chain causally produces a protected
  state/configuration/approval/external-call/value-release effect on the same
  target/resource and whether it exceeds entitlement or contribution.
- Effect evidence for an unresolved authorization-gap candidate is diagnostic
  evidence only. Preserve its assessment and keep it unresolved; only a
  satisfying_candidate_id may enter attack_candidate_ids.
- If authorization_gap_summary has neither satisfying nor unresolved candidate
  ids, do not create a new effect chain.
- Do not answer true from generic profit, value movement, callbacks, swaps, or
  effects belonging to a different chain.
- For every supported candidate, cite evidence that ties the same candidate_id
  or sensitive_call_id/path_ids to both the protected effect and the
  beyond-entitlement/contribution finding. A protected effect without this
  entitlement/contribution link remains unresolved, not true.
- When requesting follow-up evidence, anchor packet/source reads to the
  candidate evidence_ids, sensitive_call_id, and path_ids of the same
  satisfying_candidate_id. Prefer value_release_view, critical_call_argument_view,
  semantic_state_delta_view, state_change_view, beneficiary_controller_view,
  contribution_vs_payout_view, and participant_net_delta_view before broad scans.
- Set attack_candidate_ids only when authorization gap, causal link, protected
  effect, same-chain binding, and beyond_entitlement_or_contribution all hold
  for the same candidate_id.
- Set effect_status to supported, absent, or uncertain for every assessed candidate.
- The top-level answer is true when attack_candidate_ids is non-empty, uncertain
  when none qualifies but at least one candidate remains unresolved, and false
  otherwise.
- Return state_output.protected_effect_summary.
""",
        "ac_candidate_exclusion": """
- Read the candidate chains, per-candidate authorization gaps, and
  protected_effect_summary.attack_candidate_ids.
- Evaluate confirmed attack_candidate_ids when present. Before a full attack
  candidate exists, evaluate the bounded satisfying/unresolved authorization
  candidates, falling back to selected C1 candidates. Do not use evidence from
  one candidate to exclude another candidate.
- Set excluded_candidate_ids only when this exclusion is supported on the same
  actor/capability/target chain. Put unresolved cases in unresolved_candidate_ids.
- Set exclusion_status to excluded, not_excluded, or uncertain for every
  attack_candidate_id.
- If no legitimate authority or entitlement is positively established for a
  candidate, record exclusion_status=not_excluded and answer false; never encode
  that absence as answer true.
- The top-level answer is true when excluded_candidate_ids is non-empty,
  uncertain when none is excluded but at least one remains unresolved, and false
  otherwise.
- Return the state_output key named in state_output_schema.
""",
        "ts_token_semantic_anchor": """
- Return state_output.token_semantic_candidate.
- Enumerate 1-3 token-semantic candidates, each binding one token contract,
  one token-side mechanism, and concrete evidence ids. Do not invent candidates.
- Candidate mechanisms include fee-on-transfer/reflection/rebase/deflationary
  burn, token hooks/callback-driven transfer behavior, balance-vs-event
  mismatch, supply mutation, locked/vesting balance, or other non-standard
  token-contract semantics.
- Distinguish token_contract_semantics from protocol_internal_accounting:
  storage deltas in a lending/pool/router contract are not token semantics
  unless they are tied to the token contract's own balance/supply/transfer
  behavior.
- For each candidate include candidate_id, token_contract,
  token_symbol_or_label, mechanism_type, origin_status,
  nominal_transfer_or_supply_signal, actual_balance_or_state_signal,
  affected_addresses, state_evidence_ids, transfer_event_evidence_ids,
  evidence_ids, and confidence.
- Do not decide whether the protocol exploited or relied on the behavior here.
""",
        "ts_semantic_reliance": """
- Read state_input_context.token_semantic_candidate.
- Evaluate every listed candidate separately and preserve candidate_id.
- Judge whether the protocol/AMM/market/accounting operation consumes or relies
  on that same token semantic candidate without reconciling the real token-side
  behavior.
- Do not switch to a different token, path, or generic profit signal. Generic
  price movement, reentrancy, flash-loan flow, or protocol-internal ledger
  updates are not enough unless tied to the same token candidate.
- Set satisfying_candidate_ids only when reliance_status is supported,
  same_candidate_supported is true, and token_origin_supported is true.
- Set unresolved_candidate_ids when same-candidate reliance may exist but the
  available packet/source evidence is incomplete.
- The top-level answer is true when satisfying_candidate_ids is non-empty,
  uncertain when no candidate satisfies but at least one remains unresolved,
  and false otherwise.
- Return state_output.token_semantic_reliance_summary.
""",
        "ts_semantic_outcome": """
- Read state_input_context.token_semantic_candidate and
  state_input_context.token_semantic_reliance_summary.
- Evaluate only satisfying_candidate_ids. Judge whether that same token
  semantic candidate causally produces attacker-favorable or protocol-adverse
  outcome: excess assets, unbacked credit, bad debt, distorted pricing,
  protocol loss, or unauthorized payout.
- Do not answer true from empty profit_loss alone, generic value movement, or
  an outcome on a different token/component.
- Set attack_candidate_ids only when outcome_status is supported,
  causal_link_supported is true, and same_candidate_supported is true for the
  same candidate_id.
- The top-level answer is true when attack_candidate_ids is non-empty,
  uncertain when none qualifies but at least one candidate remains unresolved,
  and false otherwise.
- Return state_output.token_semantic_outcome_summary.
""",
        "ts_candidate_exclusion": """
- Read token_semantic_candidate, token_semantic_reliance_summary, and
  token_semantic_outcome_summary.attack_candidate_ids.
- Evaluate the local exclusion separately for every attack_candidate_id.
- Ordinary AMM activity, reentrancy, or protocol accounting excludes the token
  semantic candidate only when it fully explains the same candidate chain
  without token-contract semantic divergence.
- Set excluded_candidate_ids only when exclusion_status is excluded and
  same_candidate_supported is true. Put unresolved same-candidate cases in
  unresolved_candidate_ids.
- The top-level answer is true when excluded_candidate_ids is non-empty,
  uncertain when none is excluded but at least one remains unresolved, and false
  otherwise.
- Return the state_output key named in state_output_schema.
""",
        "mm_market_mechanism_anchor": """
- Return state_output.market_mechanism_candidate.
- Select one concrete market-mechanism profile, or mark it uncertain/absent.
- Valid profile_type values are price_state, pool_accounting_reserve_extraction,
  ordering_mev_arbitrage, or uncertain.
- Bind the profile to concrete evidence: market source, pool/pair/oracle/balance,
  token side, manipulation window, and supporting evidence ids.
- Do not select a profile from generic profit, flash capital, large transfer
  volume, or a normal swap alone.
""",
        "mm_market_consumption": """
- Read state_input_context.market_mechanism_candidate.
- Judge whether a value-sensitive operation consumes or acts on the same
  selected market profile.
- Do not switch from a price-state profile to token mechanics, reentrancy,
  access control, or generic profit evidence.
- AMM-internal skim/donate/sync can satisfy consumption only for the
  pool_accounting_reserve_extraction profile; otherwise prefer an external or
  downstream consumer operation.
- Return state_output.market_consumption_summary.
""",
        "mm_market_outcome": """
- Read state_input_context.market_mechanism_candidate and
  state_input_context.market_consumption_summary.
- Judge whether the same selected profile causally produces profile-specific
  attacker-favorable or protocol/victim-adverse outcome.
- Use contribution-vs-payout, participant net delta, external fundflow, value
  release, and reserve/settlement evidence before treating empty profit_loss as
  negative.
- Do not answer true from unrelated profit, balanced swaps, pure token mechanics,
  or value movement not tied to the selected profile.
- Return state_output.market_outcome_summary.
""",
        "mm_candidate_exclusion": """
- Read market_mechanism_candidate, market_consumption_summary, and
  market_outcome_summary.
- Evaluate the exclusion only against the same selected profile. Do not exclude
  sandwich, MEV, skim/donate, or arbitrage-like behavior merely by name when it
  is the selected target mechanism.
- Exclude only when the evidence fully explains the same outcome by ordinary
  market operation, token semantics, access control, insufficient validation,
  reentrancy, or accounting formula abuse without causal market manipulation.
- Return the state_output key named in state_output_schema.
""",
        "pa_accounting_anchor": """
- Return state_output.protocol_accounting_candidate.
- Select one concrete protocol-internal bookkeeping candidate, or mark it
  uncertain/absent. Valid candidates include reward/share/vault/debt/
  collateral/staking/claim/strategy/eligibility/exchange-rate accounting.
- Bind the candidate to a protocol component, accounting state, operation
  scope, and concrete evidence ids.
- Do not select a candidate from generic profit, swaps, flash capital,
  token-transfer volume, oracle/AMM price movement, or value release alone.
- If packet evidence cannot distinguish internal bookkeeping from external
  market/token/reentrancy behavior, answer uncertain rather than true.
""",
        "pa_accounting_consumption": """
- Read state_input_context.protocol_accounting_candidate.
- Judge whether a protocol operation consumes, reuses, skips, or relies on the
  same selected bookkeeping candidate.
- Valid consumption actions include claim, withdraw, borrow, mint, redeem,
  stake, unstake, liquidate, strategy action, or reward distribution.
- Do not switch to an unrelated call, token, market, state slot, or beneficiary.
- Value movement alone is insufficient unless it is tied to the selected
  accounting candidate and an entitlement/constraint/reconciliation issue.
- Return state_output.protocol_accounting_consumption_summary.
""",
        "pa_accounting_outcome": """
- Read protocol_accounting_candidate and
  protocol_accounting_consumption_summary.
- Judge whether the same candidate and consumption path causally produce a
  candidate-specific abnormal outcome: excess reward, inflated share,
  undercollateralized borrow, incorrect withdrawal, bad debt, abnormal vault
  delta, protocol loss, or comparable accounting impact.
- Use protocol_accounting_outcome_view, contribution-vs-payout, value release,
  participant net delta, state/semantic deltas, and event/call evidence before
  treating profit as proof. Empty direct value release is not enough to reject
  C3 when state-level share/reward/debt/vault/liability outcome evidence exists.
  Profit or value release without same-candidate causality is not true.
- Return state_output.protocol_accounting_outcome_summary.
""",
        "pa_candidate_exclusion": """
- Read protocol_accounting_candidate, protocol_accounting_consumption_summary,
  and protocol_accounting_outcome_summary.
- Evaluate this exclusion only against the same selected bookkeeping candidate.
- Exclude only when the same observed outcome is fully explained by normal
  accounting, market/oracle manipulation, token semantics, access control,
  insufficient validation, reentrancy/state-order abuse, or pure flash capital
  without the selected bookkeeping candidate being the root cause.
- Do not exclude merely because one of those mechanisms is present in the
  transaction; it must replace the accounting candidate as the causal chain.
- If a non-target mechanism fully explains the same outcome, state which
  target-positive protocol-accounting candidate requirement is absent or
  replaced. Do not rely on the source-family label itself.
- Return the state_output key named in state_output_schema.
""",
        "fl_flash_capital_anchor": """
- Return state_output.flash_capital_candidate.
- Select one concrete temporary atomic-capital candidate, or mark it
  uncertain/absent.
- Bind the candidate to provider/source, borrower or callback context, asset,
  amount/scale, repayment/settlement/unwind evidence, execution scope, and
  supporting evidence ids.
- Treat an unusually dense atomic call burst, high repeated call count, or
  tightly clustered callback sequence as a supporting locator for the candidate
  when it is coupled with borrow/repay/flash-swap/flash-mint semantics. Call
  density alone is not enough, but it should be captured in the state output.
- A large swap, callback, profit, value release, or generic multi-call sequence
  alone is insufficient.
""",
        "fl_exploit_mechanism": """
- Read state_input_context.flash_capital_candidate.
- Judge whether the same temporary capital candidate is a material enabler of
  a flash-assisted protocol-sensitive exploit mechanism: validation bypass, accounting
  amplification, oracle/price-state distortion, solvency/collateral break,
  callback or repayment-check abuse, temporary voting/balance/liquidity power,
  or comparable effect.
- A payload may resemble market manipulation, reentrancy, insufficient
  validation, token semantics, or accounting exploitation; do not reject it for
  that reason alone. Reject only when the temporary capital is merely incidental
  financing/routing and the observed exploit mechanism would remain equally
  supported without the selected candidate.
- Dense atomic call count or repeated callback activity is a supporting signal
  when it helps connect the candidate to scale, ordering, solvency, collateral,
  price/accounting impact, or callback reach.
- If only flash capital plus normal swaps/value movement are visible, answer
  uncertain or false according to evidence strength, not true.
- Return state_output.flash_exploit_mechanism_summary.
""",
        "fl_flash_outcome": """
- Read state_input_context.flash_capital_candidate and
  state_input_context.flash_exploit_mechanism_summary.
- Judge whether the same candidate and mechanism causally produce material
  attacker-favorable value movement, protocol/user loss, bad debt, excess
  claim, distorted settlement, or attacker-favorable state after
  repayment/settlement/unwind.
- Do not answer true from generic profit, value release, or balance delta unless
  it depends on the selected flash-capital exploit chain.
- Return state_output.flash_outcome_summary.
""",
        "fl_candidate_exclusion": """
- Read flash_capital_candidate, flash_exploit_mechanism_summary, and
  flash_outcome_summary.
- Evaluate exclusions only against the same temporary-capital candidate chain.
- Exclude when the same facts are fully explained by ordinary arbitrage,
  liquidation, capital-backed execution, incidental flash financing, or another
  primary mechanism independent of flash capital as the necessary enabler.
- Do not exclude merely because another attack-family signal is present; exclude
  only when that signal replaces the flash-capital candidate as the material
  enabler of the outcome.
- Return the state_output key named in state_output_schema.
""",
        "re_reentry_anchor": """
- Return state_output.reentrancy_candidate.
- Start from reentrancy_candidate_catalog_view and enumerate at most five
  concrete nested-entry candidates. candidate_id must exactly match a stable
  candidate_id present in that catalog; never synthesize an ID from visible
  call numbers. Each candidate must bind outer_call_id, external_edge_id,
  reentry_call_id, callback_kind, logical
  storage/accounting context, path_ids, and evidence_ids.
- Include same-function, cross-function, token-hook, fallback, callback, and
  read-only candidates when structurally supported. Do not invent candidates.
- A callback or repeated call is only a candidate locator. Do not decide stale
  state, repeated consumption, or final exploit status here.
""",
        "re_value_effect": """
- Read state_input_context.reentrancy_candidate and evaluate every candidate by
  candidate_id.
- Judge whether that exact nested path causes or enables a value-relevant state
  or asset effect. Preserve candidate_id and same_path_supported.
- Do not require stale-state, delayed-finalization, repeated-consumption, or
  phase-order proof here; re_state_order_causality owns that decision.
- Generic transaction profit, flash capital, swaps, unrelated transfers, or an
  effect on another call path do not satisfy this condition.
- Return state_output.reentrancy_value_effect_summary.
""",
        "re_state_order_causality": """
- Read reentrancy_candidate and reentrancy_value_effect_summary. Evaluate only
  the same candidate_ids; do not create a new suspicious path.
- A supported candidate-local mechanism may be: (1) a delayed protective
  update where critical outer state remains unfinalized at the external edge
  and nested execution consumes the intermediate state; (2) a repeated
  sensitive value/state effect before the prior frame returns; (3) cross-
  function re-entry through shared entitlement/accounting state; or (4) a
  read-only stale observation consumed by a downstream value-sensitive action.
- Set mechanism_type and the corresponding explicit support flag. Precise
  outer-pre/external-edge/inner/outer-post phase evidence is strongest, but it
  is not mandatory for mechanisms (2)-(4) when candidate-local trace, effect,
  and shared-state/downstream-consumption evidence establish causality.
- In EVM execution an SSTORE completed before the external edge is immediately
  visible to nested execution. A later nested SLOAD does not make that completed
  write stale. Mark state_updated_before_external_edge as safe-order evidence.
- Generic slot overlap, callbacks, repeated calls, nested output, or value flow
  without candidate-local phase order is not enough. Request target-specific
  follow-up evidence or answer uncertain when phases cannot be resolved.
- Return state_output.reentrancy_causal_order_summary.
""",
        "re_candidate_exclusion": """
- Read reentrancy_candidate, reentrancy_value_effect_summary, and
  reentrancy_causal_order_summary.
- Evaluate the exclusion separately for each attack_candidate_id and preserve
  candidate_id. Evidence from one callback must not exclude another candidate.
- Exclude only when the same candidate has finalized protective state before
  the external edge, or is fully explained by expected callback/proxy/multicall
  or independently entitlement-backed settled iterations without stale-state
  or repeated-consumption causality.
- Return the state_output key named in state_output_schema.
""",
    }.get(role, "- Return state_output with concise structured state for downstream dependent steps.")
    return f"""
Stateful runtime context:
- state_prompt_role: {role}
- state_input_context:
{stable_json_dumps(input_context)}
- state_output_schema:
{stable_json_dumps(output_schema)}

Stateful binding instructions:
{role_instructions}
"""


def _primary_output_metadata(
    text: str,
    *,
    llm: Any,
    usage: Dict[str, Any],
) -> Dict[str, Any]:
    output_empty = not str(text or "").strip()
    finish_reason = str(
        getattr(llm, "last_finish_reason", "")
        or getattr(llm, "finish_reason", "")
        or ""
    ).strip().lower()
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
    max_tokens = int(getattr(llm, "max_tokens", 0) or 0)
    reached_token_limit = bool(
        finish_reason in {"length", "max_tokens"}
        or (max_tokens > 0 and completion_tokens >= max_tokens)
    )
    return {
        "primary_output_empty": output_empty,
        "primary_finish_reason": finish_reason,
        "primary_completion_tokens": completion_tokens,
        "primary_reasoning_tokens": reasoning_tokens,
        "primary_max_tokens": max_tokens,
        "primary_output_exhausted": bool(
            output_empty and reached_token_limit
        ),
    }


def _glm_exhausted_retry_max_tokens(current_max_tokens: int) -> int:
    try:
        cap = int(
            os.getenv(
                "GLM_EXHAUSTED_RETRY_MAX_TOKENS",
                str(GLM_EXHAUSTED_RETRY_DEFAULT_MAX_TOKENS),
            )
            or GLM_EXHAUSTED_RETRY_DEFAULT_MAX_TOKENS
        )
    except (TypeError, ValueError):
        cap = GLM_EXHAUSTED_RETRY_DEFAULT_MAX_TOKENS
    current = max(0, int(current_max_tokens or 0))
    if current <= 0:
        return 0
    grown = max(current + 4096, int(current * 1.5))
    return max(current, min(max(1, cap), grown))


class JudgeModel:
    """Packet-based evidence judge that may request constrained follow-up evidence."""

    def __init__(
        self,
        llm=None,
        max_view_chars: int = 20000,
        max_context_chars: int = 60000,
        source_tool_registry: Optional[SourceToolRegistry] = None,
        enable_source_followup: bool = True,
        transcript_recorder: Optional[LLMTranscriptRecorder] = None,
        followup_context_mode: str = "unified",
    ):
        self.llm = llm
        self.max_view_chars = max_view_chars
        self.max_context_chars = max_context_chars
        self.source_tool_registry = source_tool_registry
        self.enable_source_followup = enable_source_followup
        self.transcript_recorder = transcript_recorder
        self.followup_context_mode = normalize_followup_context_mode(
            followup_context_mode
        )
        self.last_transcript_path = ""
        self.last_repair_transcript: Dict[str, Any] = {}
        self.last_followup_context_metadata: Dict[str, Any] = {}
        self.last_prompt_budget_metadata: Dict[str, Any] = {}
        self.last_visible_evidence_ids: List[str] = []

    def clone_for_worker(self) -> "JudgeModel":
        """Create an isolated judge wrapper for one parallel runtime worker.

        The clone keeps read-only configuration and shared registries, but owns
        separate last_* metadata fields so concurrent judge calls cannot race
        when PacketRuntime reads parse/render/transcript metadata.
        """
        llm = self.llm
        if hasattr(llm, "clone_for_worker"):
            try:
                llm = llm.clone_for_worker()
            except Exception:
                llm = self.llm
        return JudgeModel(
            llm=llm,
            max_view_chars=self.max_view_chars,
            max_context_chars=self.max_context_chars,
            source_tool_registry=self.source_tool_registry,
            enable_source_followup=self.enable_source_followup,
            transcript_recorder=self.transcript_recorder,
            followup_context_mode=self.followup_context_mode,
        )

    def judge(
        self,
        judge_id: str,
        question: str,
        evidence_packet: Dict[str, Any],
        condition_id: Optional[str] = None,
        evidence_refs: Optional[List[str]] = None,
        expected_answer: bool = True,
        tx_hash: Optional[str] = None,
        chain: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        allowed_followup_views: Optional[List[str]] = None,
        allowed_followup_view_summary: Optional[Dict[str, Any]] = None,
        packet_evidence_adequacy: Optional[Dict[str, Any]] = None,
        tool_observations: Optional[List[Dict[str, Any]]] = None,
        allow_tool_requests: bool = True,
        transcript_context: Optional[Dict[str, Any]] = None,
        state_input_context: Optional[Dict[str, Any]] = None,
        state_prompt_role: str = "",
        state_output_schema: Optional[Dict[str, Any]] = None,
        stateful_runtime: Optional[Dict[str, Any]] = None,
        followup_context: Optional[FollowupPromptContext] = None,
        attack_label: str = "",
    ) -> JudgeResult:
        self.last_visible_evidence_ids = []
        if followup_context is not None:
            state_input_context = followup_context.state_input_context
        self.last_followup_context_metadata = dict(
            followup_context.metadata if followup_context is not None else {
                "mode": self.followup_context_mode,
            }
        )
        state_input_context = _normalize_state_dict(state_input_context)
        state_output_schema = _normalize_state_dict(state_output_schema)
        stateful_runtime_meta = dict(
            stateful_runtime
            or _stateful_runtime_metadata(
                state_prompt_role=state_prompt_role,
            )
        )
        if self.llm is None:
            return JudgeResult(
                id=judge_id,
                question=question,
                answer="uncertain",
                reason="No LLM judge configured.",
                confidence="low",
                evidence_refs=list(evidence_refs or []),
                missing_evidence=["judge_model"],
                expected_answer=expected_answer,
                tool_requests=[],
                tool_calls=[],
                condition_feature_analysis=normalize_condition_feature_analysis({}),
                state_input=state_input_context,
                state_output={},
                stateful_runtime=stateful_runtime_meta,
            )

        prompt, render_metadata = self._build_prompt_with_metadata(
            question,
            evidence_packet,
            condition_id,
            max_view_chars=self.max_view_chars,
            max_context_chars=self.max_context_chars,
            tx_hash=tx_hash,
            chain=chain,
            allowed_tools=list(allowed_tools or []),
            allowed_followup_views=list(allowed_followup_views or []),
            allowed_followup_view_summary=dict(allowed_followup_view_summary or {}),
            packet_evidence_adequacy=dict(packet_evidence_adequacy or {}),
            tool_observations=list(tool_observations or []),
            allow_tool_requests=allow_tool_requests,
            state_input_context=state_input_context,
            state_prompt_role=state_prompt_role,
            state_output_schema=state_output_schema,
            followup_context=followup_context,
            attack_label=attack_label,
        )
        self.last_prompt_budget_metadata = {
            "max_context_chars": int(self.max_context_chars or 0),
            "prompt_chars": len(prompt),
            "within_budget": len(prompt) <= int(self.max_context_chars or 0),
            "attack_label": _judge_attack_label(attack_label, question),
        }
        self.last_visible_evidence_ids = _visible_evidence_ids_in_prompt(
            prompt,
            evidence_packet=evidence_packet,
            tool_observations=list(tool_observations or []),
        )
        self.last_render_metadata = render_metadata
        self.last_parse_metadata = {}
        self.last_transcript_path = ""
        self.last_repair_transcript = {}
        text = self.llm.complete(prompt)
        primary_usage = dict(getattr(self.llm, "last_usage", {}) or {})
        primary_output_metadata = _primary_output_metadata(
            text,
            llm=self.llm,
            usage=primary_usage,
        )
        exhausted_retry_metadata: Dict[str, Any] = {
            "exhausted_retry_attempted": False,
        }
        provider = str(getattr(self.llm, "provider", "") or "").strip().lower()
        if (
            provider == "glm"
            and primary_output_metadata.get("primary_output_exhausted")
            and hasattr(self.llm, "clone_with_max_tokens")
        ):
            initial_max_tokens = int(getattr(self.llm, "max_tokens", 0) or 0)
            retry_max_tokens = _glm_exhausted_retry_max_tokens(initial_max_tokens)
            if retry_max_tokens > initial_max_tokens:
                retry_llm = self.llm.clone_with_max_tokens(retry_max_tokens)
                retry_text = retry_llm.complete(prompt)
                retry_usage = dict(getattr(retry_llm, "last_usage", {}) or {})
                retry_output_metadata = _primary_output_metadata(
                    retry_text,
                    llm=retry_llm,
                    usage=retry_usage,
                )
                exhausted_retry_metadata = {
                    "exhausted_retry_attempted": True,
                    "exhausted_retry_provider": provider,
                    "exhausted_retry_initial_max_tokens": initial_max_tokens,
                    "exhausted_retry_max_tokens": retry_max_tokens,
                    "exhausted_retry_initial_usage": primary_usage,
                    "exhausted_retry_initial_finish_reason": (
                        primary_output_metadata.get("primary_finish_reason", "")
                    ),
                    "exhausted_retry_succeeded": not bool(
                        retry_output_metadata.get("primary_output_exhausted")
                    ),
                }
                text = retry_text
                primary_usage = retry_usage
                primary_output_metadata = retry_output_metadata
        data, parse_metadata = self._extract_with_repair(
            text,
            state_output_schema=state_output_schema,
        )
        if parse_metadata.get("parse_status") == "failed" and self.llm is not None:
            retry_prompt = _build_schema_only_judge_retry_prompt(prompt)
            retry_text = self.llm.complete(retry_prompt)
            retry_usage = dict(getattr(self.llm, "last_usage", {}) or {})
            retry_parse_text, retry_thinking_metadata = structured_output_text(
                retry_text,
                provider=provider,
            )
            retry_data, retry_error, retry_selection = (
                _try_extract_judge_json_object(
                    retry_parse_text,
                    state_output_schema=state_output_schema,
                )
            )
            schema_retry = {
                "schema_retry_attempted": True,
                "schema_retry_succeeded": retry_error is None,
                "schema_retry_error": retry_error or "",
                "schema_retry_usage": retry_usage,
                "schema_retry_thinking_filter": retry_thinking_metadata,
                "schema_retry_selection": retry_selection,
            }
            self.last_repair_transcript = {
                **dict(self.last_repair_transcript or {}),
                "schema_retry": {
                    "prompt": retry_prompt,
                    "raw_completion": retry_text,
                    "usage": retry_usage,
                    "error": retry_error or "",
                },
            }
            if retry_error is None:
                data = retry_data
                text = retry_text
                primary_usage = retry_usage
                parse_metadata = {
                    **dict(parse_metadata or {}),
                    "parse_status": "repaired",
                    "repair_mode": "schema_only_rejudge",
                    **schema_retry,
                    **retry_selection,
                }
            else:
                parse_metadata = {
                    **dict(parse_metadata or {}),
                    **schema_retry,
                }
        deterministic_schema_repairs = list(
            data.pop("_deterministic_schema_repairs", [])
            if isinstance(data, dict)
            else []
        )
        if deterministic_schema_repairs:
            parse_metadata = {
                **dict(parse_metadata or {}),
                "deterministic_schema_repairs": deterministic_schema_repairs,
            }
        feature_parse_metadata = _condition_feature_analysis_parse_metadata(data)
        data = _with_normalized_condition_feature_analysis(data)
        parse_metadata = {
            **dict(parse_metadata or {}),
            **primary_output_metadata,
            **exhausted_retry_metadata,
            **feature_parse_metadata,
        }
        self.last_parse_metadata = parse_metadata
        self._record_transcript(
            prompt=prompt,
            raw_completion=text,
            usage=primary_usage,
            render_metadata=render_metadata,
            parse_metadata=parse_metadata,
            parsed_response=data,
            tx_hash=tx_hash,
            chain=chain,
            judge_id=judge_id,
            condition_id=condition_id,
            evidence_refs=evidence_refs,
            allowed_tools=allowed_tools,
            allowed_followup_views=allowed_followup_views,
            tool_observations=tool_observations,
            allow_tool_requests=allow_tool_requests,
            transcript_context=transcript_context,
            followup_context_metadata=self.last_followup_context_metadata,
        )

        return self._parse_result(
            judge_id,
            question,
            data,
            evidence_refs=evidence_refs,
            expected_answer=expected_answer,
            tool_calls=[],
            state_input_context=state_input_context,
            state_prompt_role=state_prompt_role,
            stateful_runtime=stateful_runtime_meta,
        )
    def _extract_with_repair(
        self,
        text: str,
        *,
        state_output_schema: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        provider = str(getattr(self.llm, "provider", "") or "").strip().lower()
        parse_text, thinking_metadata = structured_output_text(
            text,
            provider=provider,
        )
        data, error, selection_metadata = _try_extract_judge_json_object(
            parse_text,
            state_output_schema=state_output_schema,
        )
        if error is None:
            return data, {
                "parse_status": "ok",
                "repair_attempted": False,
                **thinking_metadata,
                **selection_metadata,
            }

        if self.llm is not None:
            try:
                repair_prompt = _build_json_repair_prompt(parse_text, error)
                repair_llm = self.llm
                if provider == "minimax" and hasattr(
                    self.llm,
                    "clone_with_max_tokens",
                ):
                    repair_llm = self.llm.clone_with_max_tokens(
                        MINIMAX_JSON_REPAIR_MAX_TOKENS
                    )
                repair_text = repair_llm.complete(repair_prompt)
                self.last_repair_transcript = {
                    "prompt": repair_prompt,
                    "raw_completion": repair_text,
                    "usage": dict(getattr(repair_llm, "last_usage", {}) or {}),
                    "max_tokens": int(
                        getattr(repair_llm, "max_tokens", 0) or 0
                    ),
                }
                repair_parse_text, repair_thinking_metadata = structured_output_text(
                    repair_text,
                    provider=provider,
                )
                repaired, repair_error, repair_selection_metadata = (
                    _try_extract_judge_json_object(
                        repair_parse_text,
                        state_output_schema=state_output_schema,
                    )
                )
                if repair_error is None:
                    return repaired, {
                        "parse_status": "repaired",
                        "repair_attempted": True,
                        "initial_error": error,
                        **thinking_metadata,
                        "repair_thinking_filter": repair_thinking_metadata,
                        **repair_selection_metadata,
                    }
                return _invalid_json_uncertain(error, repair_error), {
                    "parse_status": "failed",
                    "repair_attempted": True,
                    "initial_error": error,
                    "repair_error": repair_error,
                    **thinking_metadata,
                    "repair_thinking_filter": repair_thinking_metadata,
                    "initial_selection": selection_metadata,
                    "repair_selection": repair_selection_metadata,
                }
            except Exception as exc:
                self.last_repair_transcript = {
                    "error": repr(exc),
                }
                return _invalid_json_uncertain(error, repr(exc)), {
                    "parse_status": "failed",
                    "repair_attempted": True,
                    "initial_error": error,
                    "repair_error": repr(exc),
                    **thinking_metadata,
                }

        return _invalid_json_uncertain(error), {
            "parse_status": "failed",
            "repair_attempted": False,
            "initial_error": error,
            **thinking_metadata,
        }

    def _record_transcript(
        self,
        *,
        prompt: str,
        raw_completion: str,
        usage: Dict[str, Any],
        render_metadata: List[Dict[str, Any]],
        parse_metadata: Dict[str, Any],
        parsed_response: Dict[str, Any],
        tx_hash: Optional[str],
        chain: Optional[str],
        judge_id: str,
        condition_id: Optional[str],
        evidence_refs: Optional[List[str]],
        allowed_tools: Optional[List[str]],
        allowed_followup_views: Optional[List[str]],
        tool_observations: Optional[List[Dict[str, Any]]],
        allow_tool_requests: bool,
        transcript_context: Optional[Dict[str, Any]],
        followup_context_metadata: Optional[Dict[str, Any]],
    ) -> None:
        if self.transcript_recorder is None:
            return
        context = dict(transcript_context or {})
        round_id = context.get("round", context.get("round_id", 0))
        payload = {
            "kind": "judge",
            "phase": context.get("phase", ""),
            "round": round_id,
            "tx_hash": tx_hash or context.get("tx_hash", ""),
            "chain": chain or context.get("chain", ""),
            "judge_id": judge_id,
            "condition_id": condition_id or context.get("condition_id", judge_id),
            "evidence_refs": list(evidence_refs or []),
            "allowed_tools": list(allowed_tools or []),
            "allowed_followup_views": list(allowed_followup_views or []),
            "allow_tool_requests": bool(allow_tool_requests),
            "prompt": prompt,
            "raw_completion": raw_completion,
            "usage": dict(usage or {}),
            "render_metadata": list(render_metadata or []),
            "tool_observations": list(tool_observations or []),
            "followup_context": dict(followup_context_metadata or {}),
            "prompt_budget": dict(self.last_prompt_budget_metadata or {}),
            "parse_metadata": dict(parse_metadata or {}),
            "parsed_response": dict(parsed_response or {}),
            "repair": dict(self.last_repair_transcript or {}),
            "context": context,
        }
        path = self.transcript_recorder.save_judge_transcript(
            payload,
            phase=str(context.get("phase", "")),
            tx_hash=str(payload["tx_hash"]),
            judge_id=str(judge_id),
            round_id=round_id,
        )
        self.last_transcript_path = str(path)

    @staticmethod
    def _build_prompt(
        question: str,
        evidence_packet: Dict[str, Any],
        condition_id: Optional[str] = None,
        max_view_chars: int = 20000,
        max_context_chars: int = 60000,
        tx_hash: Optional[str] = None,
        chain: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        allowed_followup_views: Optional[List[str]] = None,
        allowed_followup_view_summary: Optional[Dict[str, Any]] = None,
        packet_evidence_adequacy: Optional[Dict[str, Any]] = None,
        tool_observations: Optional[List[Dict[str, Any]]] = None,
        allow_tool_requests: bool = True,
        state_input_context: Optional[Dict[str, Any]] = None,
        state_prompt_role: str = "",
        state_output_schema: Optional[Dict[str, Any]] = None,
        followup_context: Optional[FollowupPromptContext] = None,
        attack_label: str = "",
    ) -> str:
        prompt, _ = JudgeModel._build_prompt_with_metadata(
            question=question,
            evidence_packet=evidence_packet,
            condition_id=condition_id,
            max_view_chars=max_view_chars,
            max_context_chars=max_context_chars,
            tx_hash=tx_hash,
            chain=chain,
            allowed_tools=allowed_tools,
            allowed_followup_views=allowed_followup_views,
            allowed_followup_view_summary=allowed_followup_view_summary,
            packet_evidence_adequacy=packet_evidence_adequacy,
            tool_observations=tool_observations,
            allow_tool_requests=allow_tool_requests,
            state_input_context=state_input_context,
            state_prompt_role=state_prompt_role,
            state_output_schema=state_output_schema,
            followup_context=followup_context,
            attack_label=attack_label,
        )
        return prompt

    @staticmethod
    def _build_prompt_with_metadata(
        question: str,
        evidence_packet: Dict[str, Any],
        condition_id: Optional[str] = None,
        max_view_chars: int = 20000,
        max_context_chars: int = 60000,
        tx_hash: Optional[str] = None,
        chain: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        allowed_followup_views: Optional[List[str]] = None,
        allowed_followup_view_summary: Optional[Dict[str, Any]] = None,
        packet_evidence_adequacy: Optional[Dict[str, Any]] = None,
        tool_observations: Optional[List[Dict[str, Any]]] = None,
        allow_tool_requests: bool = True,
        state_input_context: Optional[Dict[str, Any]] = None,
        state_prompt_role: str = "",
        state_output_schema: Optional[Dict[str, Any]] = None,
        followup_context: Optional[FollowupPromptContext] = None,
        attack_label: str = "",
    ) -> tuple[str, List[Dict[str, Any]]]:
        if followup_context is not None:
            evidence_context = followup_context.evidence_context
            render_metadata = list(
                followup_context.evidence_render_metadata or []
            )
            state_input_context = followup_context.state_input_context
        else:
            evidence_context, render_metadata = _format_evidence_context_with_metadata(
                evidence_packet,
                max_view_chars=max_view_chars,
                max_context_chars=max_context_chars,
                packet_evidence_adequacy=packet_evidence_adequacy,
            )
        condition_note = ""
        if condition_id:
            condition_note = (
                f"\nThis question corresponds to rule condition: {condition_id}"
            )
        tx_note = ""
        if tx_hash or chain:
            tx_note = f"\nTransaction: tx_hash={tx_hash or 'unknown'} chain={chain or 'unknown'}"

        allowed_tools = list(allowed_tools or [])
        allowed_followup_views = list(allowed_followup_views or [])
        allowed_followup_view_summary = dict(allowed_followup_view_summary or {})
        packet_evidence_adequacy = dict(packet_evidence_adequacy or {})
        tool_observations = list(tool_observations or [])
        stateful_runtime_section = _stateful_runtime_prompt_section(
            state_input_context=state_input_context,
            state_prompt_role=state_prompt_role,
            state_output_schema=state_output_schema,
        )
        state_output_schema_field = _state_output_schema_field(state_prompt_role)
        normalized_attack_label = _judge_attack_label(attack_label, question)
        family_instructions = _judge_family_instructions(normalized_attack_label)
        family_followup_guidance = _judge_family_followup_guidance(
            normalized_attack_label
        )

        followup_instructions = ""
        if allowed_tools and allow_tool_requests:
            followup_instructions = FOLLOWUP_TOOL_INSTRUCTIONS.format(
                allowed_tools=stable_json_dumps(allowed_tools),
                allowed_followup_view_summary=stable_json_dumps(
                    allowed_followup_view_summary
                ),
                allowed_followup_views=stable_json_dumps(allowed_followup_views),
                family_followup_guidance=family_followup_guidance,
            )
        elif tool_observations:
            followup_instructions = (
                "\nFollow-up evidence observations are already provided below. "
                "Do not request more tools; produce a final local judgment from "
                "the original evidence plus these observations.\n"
            )
        previous_judge_summary_text = ""
        if followup_context is not None and followup_context.previous_judge_summary_text:
            previous_judge_summary_text = (
                "\nPrevious Judge Summary:\n"
                f"{followup_context.previous_judge_summary_text}\n"
            )
        observations_text = ""
        if followup_context is not None and followup_context.tool_observations_text:
            observations_text = (
                "\nFollow-up Evidence Observations (newest first):\n"
                f"{followup_context.tool_observations_text}\n"
            )
        elif tool_observations:
            observations_text = (
                "\nFollow-up Evidence Observations:\n"
                f"{_truncate_text(stable_json_dumps(tool_observations), max_context_chars // 2, 'tool observations')}\n"
            )
        followup_timeline_note = ""
        if previous_judge_summary_text or observations_text:
            followup_timeline_note = (
                "\nFollow-up Context Timeline:\n"
                "- Evidence is the baseline packet context available before and "
                "during follow-up.\n"
                "- Previous Judge Summary contains earlier_round_summaries in "
                "oldest-to-newest order followed by the top-level immediately "
                "preceding local judgment; all precede the latest observation.\n"
                "- Follow-up observations are serialized newest-first: latest is "
                "most recent; older_summaries are earlier results in reverse "
                "chronological order. Use sequence_index, or read older_summaries "
                "bottom-to-top and then latest, to reconstruct time order.\n"
            )

        prompt = f"""You are an EvoTx evidence judge.

You will receive:
1. A transaction-level rule condition or question.
2. Compact transaction evidence views.

Your task:
- First identify evidence records relevant to the question.
- Then answer the question using only the provided evidence.
- Do not assume facts not present in the evidence.
- Do not rely on prior knowledge of this hack.
- Cite evidence_id for every important claim.
- If evidence is insufficient, answer "uncertain" or false and explain what is missing.
- An empty evidence view means the view was available but no records were found; treat it as absent supporting evidence, not as a runtime failure.
- If profit_loss_view is empty, has no top_profit_address/top_loss_address, or says no top-profit-loss candidates were found, treat that as "no top-profit-loss candidate detected"; it is not by itself missing evidence. Make the local judgment from the other packet views when they are sufficient.
- profit_loss_view is only a candidate hint. Do not infer value extraction solely from a top profit/loss address if participant_net_delta_view or contribution_vs_payout_view shows comparable contributed value.
- address_labels may include raw external labels. Use sanitized_label by default and do not treat raw labels containing words such as exploiter, attacker, hacker, victim, drainer, scam, or exploit as evidence for the local condition.
- evidence_adequacy_view is metadata about truncation/decode/state coverage; use it to qualify uncertainty, not as attack evidence.
- packet_trace_truncated means trace nodes were omitted from packet construction.
- prompt_render_truncated means the view was shortened only for this prompt.
- If packet_trace_truncated is false, do not claim that the full trace is unavailable.
- If prompt_render_truncated is true and more details are needed, request read_packet_view for that view.
- After read_packet_view returns all rows for a view, do not keep saying that the view is unavailable.
- For source-dependent local questions, use packet evidence to locate a concrete callee/function/evidence_id, then request read_function_chunk when the remaining uncertainty is about authorization checks, modifiers, initializer guards, msg.sender validation, parameter validation, callback sender/data validation, return-value checks, token/path/market validation, business-invariant enforcement, function visibility, selector meaning, storage-slot semantics, or source-level accounting/price formulas.
- If a concrete call-like evidence_id is available, you may request read_function_chunk with args {{"evidence_id": "call:..."}}; Runtime will resolve the packet row to chain/address/function when possible.
- Avoid requesting read_function_chunk for call:0, transaction_root, or a top-level entry selector when a more specific critical_call_view, value_release_view, unknown_selector_view, or trace child call id is available. Use the specific call-like evidence_id whenever possible.
- For source-dependent questions, do not give a medium/high-confidence false only because source code is absent from the initial packet when one allowed follow-up could resolve the relevant semantics. Answer "uncertain" and request that evidence.
- Source code is strong supporting evidence, but it is not the only possible evidence. The source-unavailable rule below applies only when read_function_chunk returns source_unavailable, invalid_address, function_not_found, or no snippets. In that case, re-evaluate the original packet evidence and answer from it only when it positively establishes the requested semantic fact.
- When source is unavailable, a complete execution trace establishes observed calls, state accesses, and effects but does not prove that a passing require/assert, branch predicate, modifier, or semantic validation check was absent. No visible check is not positive evidence. Require transaction-visible evidence that a specific invalid, stale, inconsistent, adversarial, or boundary-breaking object was accepted and causally consumed; otherwise remain uncertain (or false when complete evidence positively shows an adequate check).
- When read_function_chunk returns status=ok with a snippet for the exact resolved function, evaluate that source directly. Source with source_provenance_status=explorer_verified_unpinned or decisive_negative_eligible=false may positively support function semantics or a missing local check when the observed trace follows the same path and does not contradict the snippet. Qualify confidence for provenance. It cannot by itself prove that a safety check was present, justify a decisive safe/false conclusion, or override contradictory observed trace behavior. A modifier, nonReentrant guard, or source-level safe ordering from such source is supporting context only; if observed trace/state/fund-flow shows the contested behavior occurred, answer from the runtime facts.
- Prefer precise follow-ups: use read_evidence_context or read_function_chunk(evidence_id=...) when a concrete evidence_id is available; use read_packet_view with evidence_ids or keywords when possible; avoid broad read_packet_view(trace_view) unless no narrower call/context id is available.
- Return exactly one strict JSON object and nothing else.
- Do not output <think> tags, chain-of-thought, analysis, commentary, JSON examples, or markdown fences.
- Begin the response with "{{" and end it with "}}".
- If the question asks about absence of a pattern (e.g., "no re-entry", "no sensitive function"), answer based strictly on whether the evidence shows that pattern. Do NOT use subjective comparative reasoning like "better explained as" or "more consistent with". If the evidence shows the pattern exists, the answer is false; if it does not, the answer is true.
{CONDITION_FEATURE_ANALYSIS_PROMPT}
{stateful_runtime_section}

Absence-aware evidence policy:
When evidence_adequacy_view indicates packet_trace_truncated=false, the packet is complete. In that case:
- empty critical_call_view means no critical operation matched the configured signal set and no reentrant calls were detected.
- empty amm_reserve_transition_view means no AMM reserve transition was detected.
- empty flash_or_atomic_capital_view means no flash or atomic capital pattern was detected.
- empty unknown_selector_view means no unresolved critical selector was detected.
- empty price_relevant_state_view means no price-relevant state change was detected.
- small same-token fundflow means limited observed asset movement, not missing fundflow.
When a view shows "(Not available: ...)" instead of evidence rows, the data source could not provide that information. This is different from an empty view: it means evidence may exist but is inaccessible. In that case, answer uncertain if the missing data is critical for the question, or rely on other available views.
- state_changes_available=false in evidence_adequacy_view means state_change_view, semantic_state_delta_view, and price_relevant_state_view are incomplete because the data source could not provide state changes. Do not treat their emptiness as negative evidence.
- profit_loss_available=false means profit/loss candidate data was not provided by the data source.
- operation_summary_view is the authoritative signal presence summary. Use its absence flags as negative evidence.
- operation_summary_view is the canonical signal-presence view; do not request or refer to a separate signal_presence_view.
- value_release_view summarizes sensitive calls that caused token/native value out, recipients, repeated releases, and target balance changes. Use it for local value-release/outcome questions, but do not treat the view name itself as proof of an attack.
- protocol_accounting_outcome_view summarizes protocol-internal bookkeeping outcome candidates such as reward/share/vault/debt/collateral/staking/claim or liability state deltas with related value and payout evidence ids. For protocol-accounting outcome questions, consult it before treating an empty value_release_view as no material outcome. It is neutral and must be bound to the same selected bookkeeping candidate.
{family_instructions}

Do not use uncertain when the evidence is complete but simply does not show the required signal. Answer false instead.
{condition_note}{tx_note}
{followup_instructions}
Question:
{question}

Evidence Render Metadata:
{stable_json_dumps(render_metadata)}

{followup_timeline_note}
Evidence (baseline packet context):
{evidence_context}
{previous_judge_summary_text}
{observations_text}

Return strict JSON:
{{
  "answer": true | false | "uncertain",
  "confidence": "low" | "medium" | "high",
  "reason": "...",
  "supporting_evidence_ids": [],
  "contradicting_evidence_ids": [],
  "missing_evidence": [],
  "suggested_followup_views": [],
  "tool_requests": [],
{CONDITION_FEATURE_ANALYSIS_SCHEMA_FIELD}{state_output_schema_field}
}}
"""
        prompt = _enforce_judge_prompt_budget(
            prompt,
            max_context_chars=max_context_chars,
            evidence_context=evidence_context,
            previous_judge_summary_text=previous_judge_summary_text,
            tool_observations_text=observations_text,
        )
        return prompt, render_metadata

    @staticmethod
    def _build_followup_prompt(
        question: str,
        evidence_packet: Dict[str, Any],
        condition_id: Optional[str],
        first_response: Dict[str, Any],
        tool_call: Dict[str, Any],
        max_view_chars: int,
        max_context_chars: int,
        tx_hash: Optional[str] = None,
        chain: Optional[str] = None,
    ) -> str:
        evidence_context = _format_evidence_context(
            evidence_packet,
            max_view_chars=max_view_chars,
            max_context_chars=max_context_chars,
        )
        condition_note = ""
        if condition_id:
            condition_note = (
                f"\nThis question corresponds to rule condition: {condition_id}"
            )
        tx_note = ""
        if tx_hash or chain:
            tx_note = f"\nTransaction: tx_hash={tx_hash or 'unknown'} chain={chain or 'unknown'}"

        return f"""You are an EvoTx evidence judge.

You previously requested one source-code follow-up. That one permitted tool call has now been executed. You must now produce the final judgment. Do not request another tool call.
- If the source tool returned source_unavailable, invalid_address, function_not_found, or no snippets, treat that as unavailable verified source for this one-hop lookup. Do not make another tool request. Use the original packet evidence and the tool status to make the best final local judgment.
- Do not treat source_unavailable alone as evidence that the transaction is benign or malicious. If source code is essential and original packet evidence cannot support the condition, answer "uncertain"; otherwise answer from the already available trace/state/event/fund-flow evidence.
{CONDITION_FEATURE_ANALYSIS_PROMPT}
{condition_note}{tx_note}

Question:
{question}

Original Evidence:
{evidence_context}

Your First Response:
{stable_json_dumps(first_response)}

Additional Source Evidence:
{stable_json_dumps(tool_call)}

Return strict final JSON only:
{{
  "answer": true | false | "uncertain",
  "confidence": "low" | "medium" | "high",
  "reason": "...",
  "supporting_evidence_ids": [],
  "contradicting_evidence_ids": [],
  "missing_evidence": [],
  "suggested_followup_views": [],
{CONDITION_FEATURE_ANALYSIS_SCHEMA_FIELD}
}}
"""

    @staticmethod
    def _parse_result(
        judge_id: str,
        question: str,
        data: Dict[str, Any],
        evidence_refs: Optional[List[str]] = None,
        expected_answer: bool = True,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        state_input_context: Optional[Dict[str, Any]] = None,
        state_prompt_role: str = "",
        stateful_runtime: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        answer_raw = data.get("answer", "uncertain")
        if isinstance(answer_raw, bool):
            answer: Union[bool, str] = answer_raw
        elif isinstance(answer_raw, str):
            normalized = answer_raw.lower().strip()
            if normalized in ("true", "yes"):
                answer = True
            elif normalized in ("false", "no"):
                answer = False
            else:
                answer = normalized
        else:
            answer = "uncertain"

        confidence = str(data.get("confidence", "medium"))
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"

        return JudgeResult(
            id=judge_id,
            question=question,
            answer=answer,
            reason=str(data.get("reason", "")),
            confidence=confidence,
            evidence_refs=list(evidence_refs or []),
            supporting_evidence_ids=list(
                data.get("supporting_evidence_ids", [])
            ),
            contradicting_evidence_ids=list(
                data.get("contradicting_evidence_ids", [])
            ),
            missing_evidence=list(data.get("missing_evidence", [])),
            suggested_followup_views=list(
                data.get("suggested_followup_views", [])
            ),
            tool_requests=_extract_tool_requests(data),
            expected_answer=expected_answer,
            tool_calls=list(tool_calls or []),
            condition_feature_analysis=normalize_condition_feature_analysis(
                data.get("condition_feature_analysis", {})
            ),
            state_input=_normalize_state_dict(state_input_context),
            state_output=_normalize_state_dict(data.get("state_output", {})),
            stateful_runtime=dict(
                stateful_runtime
                or _stateful_runtime_metadata(state_prompt_role=state_prompt_role)
            ),
        )

    @staticmethod
    def _extract_tool_request(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        request = data.get("tool_request") or data.get("tool_call")
        if isinstance(request, dict):
            tool = str(
                request.get("tool")
                or request.get("name")
                or request.get("tool_name")
                or ""
            ).strip()
            if tool == "read_function_chunk":
                return request
        if str(
            data.get("tool") or data.get("name") or data.get("tool_name") or ""
        ).strip() == "read_function_chunk":
            return data
        return None

    @staticmethod
    def _normalize_tool_request(
        request: Dict[str, Any],
        evidence_packet: Dict[str, Any],
        chain: Optional[str] = None,
    ) -> Dict[str, Any]:
        default_chain = chain or _infer_chain_from_evidence(evidence_packet) or "eth"
        return {
            "tool": "read_function_chunk",
            "address": str(request.get("address") or request.get("contract") or ""),
            "function_name": str(
                request.get("target_function")
                or request.get("function_name")
                or request.get("function")
                or request.get("selector")
                or ""
            ),
            "target_function": str(request.get("target_function") or ""),
            "chain": str(request.get("chain") or default_chain),
            "max_snippets": int(request.get("max_snippets", 3) or 3),
            "max_chars_per_snippet": int(
                request.get("max_chars_per_snippet", 3500) or 3500
            ),
        }


FOLLOWUP_TOOL_INSTRUCTIONS = """
Follow-up view manifest (only views allowed for this local condition):
{allowed_followup_view_summary}

Allowed evidence tools:
{allowed_tools}

Request tools only when one listed tool can resolve a concrete missing fact.
Return at most 2 tool_requests. Normally a true or false answer returns no tool
requests. Exception: an access-control authorization-anchor answer may be false
or uncertain while preserving an unresolved_probe; in that case request local
evidence for the concrete probe call before closing the condition.

Tool routing:
- read_packet_view: use args {{"view": "listed_view_name", "evidence_ids":
  ["optional:evidence_id"], "keywords": ["optional keyword"], "limit": 20}}.
  The key is exactly "view", not "view_name". Prefer evidence_ids/keywords/limits.
- read_evidence_by_id/read_evidence_context: expand a cited evidence id.
- get_local_call_context: inspect one specific call and its local subtree. Use
  it before another broad view when a concrete call id is known but the current
  packet view is truncated, selector semantics are unresolved, or the relation
  between an entry call and nearby child/state/event effects is missing.
- read_function_chunk: only for unresolved function/selector/storage/validation/
  authorization/formula semantics after a concrete call-like evidence_id or
  address/function has been located. Avoid call:0 and transaction_root.
  To follow a wrapper into a named internal callee, pass
  args {{"evidence_id": "call:...", "target_function": "_claim"}}. Request
  separate target_function values when more than one internal callee matters.
- Never request list_packet_views, broad trace_view without need, or a tool to
  decide the attack label. Tools return evidence only.
{family_followup_guidance}
"""


def _judge_attack_label(attack_label: str, question: str) -> str:
    normalized = normalize_attack_label(attack_label)
    if normalized:
        return normalized
    text = str(question or "").lower().replace("_", " ").replace("-", " ")
    inferred = (
        ("reentrancy", ("reentr", "re-enter", "callback ordering")),
        ("price_manipulation", ("price manipul", "oracle", "reserve distortion")),
        ("access_control", ("access control", "authoriz", "privileg", "onlyowner")),
        ("insufficient_validation", ("insufficient validation", "validation gap")),
        ("flashloans", ("flash loan", "flashloan", "atomic capital")),
    )
    for label, keywords in inferred:
        if any(keyword in text for keyword in keywords):
            return label
    return ""


def _judge_family_instructions(attack_label: str) -> str:
    if attack_label == "reentrancy":
        return """Reentrancy-specific policy:
- A critical_call_view row whose why_included starts with reentrant_call is structural nested re-entry, not by itself proof of exploitable stale state.
- Use reentrancy_candidate_catalog_view first for C1 and select only its stable candidate IDs. Use reentrancy_state_order_summary_view for candidate-local phase/order questions.
- Request reentrancy_state_order_view or get_local_call_context only when compact candidate evidence cannot resolve the local condition.
- Distinguish re-entry existence, candidate-local value/state effect, and causal mechanism; do not let one substitute for another. Causality may be delayed protective update, repeated sensitive effect before return, cross-function shared accounting, or a downstream-consumed read-only stale observation."""
    if attack_label == "access_control":
        return """Access-control-specific policy:
- Generic profit, swaps, flash loans, callbacks, reentrancy, or a public entry call do not prove an authorization failure.
- Bind evidence to one candidate chain: actor/caller or beneficiary -> required
  authority -> sensitive capability and protected target/resource -> effect.
- A single sensitive call may carry the full candidate. Preserve an entry parent
  and downstream child only when they differ and trace evidence links them.
- Concrete owner/admin/role/permission/initializer/proxy/delegatecall,
  authorization-state, source, caller/beneficiary, or protected-effect evidence
  may establish the boundary. Record provenance when observed; do not require a
  complete provenance taxonomy when same-candidate evidence already resolves it.
- An unknown selector tied to protected state/value should trigger local
  evidence acquisition, not an unsupported true or premature false.
- A modifier or permission query applies only when it guards the same candidate
  capability; observed self-authorization or bypass on that path can contradict it.
- Keep actor authorization distinct from validation of a user-supplied token,
  path, router, calldata, market, or payload. The latter belongs to insufficient
  validation unless it directly grants the actor access to the protected capability.
- Do not answer a local condition true by combining an actor from one call, authority semantics from another unrelated function, and an effect on a different asset, position, or protocol component. Multiple calls count only when evidence ids or trace relations connect them to the same authorization boundary.
- When source is unavailable, transaction-visible caller/beneficiary links, proxy/delegatecall or callback paths, authorization-state changes, protected-target balance changes, and contribution-versus-payout evidence may establish the chain. If only generic suspicious effects remain, answer false or uncertain according to evidence completeness.
- For exclusions, distinguish a normally public sensitive target from a public entry that merely reaches a protected target."""
    if attack_label == "price_manipulation":
        return """Price-manipulation-specific policy:
- If the complete operation summary shows no swap, reserve transition, oracle read, borrow/repay, mint/burn, liquidation, or price-dependent operation, answer false for a core price-manipulation condition.
- Victim loss or one-sided transfer alone is insufficient; connect price-source perturbation to a price-dependent action and outcome.
- For a consumption condition, the actor's ordinary sequential swaps merely
  repricing against the AMM's normal reserve updates are not sufficient. A
  same-market consumer may still qualify, but evidence must show that an
  abnormal, independently established distortion causally determined its
  amount, limit, settlement, or extraction beyond ordinary swap sequencing."""
    if attack_label == "insufficient_validation":
        return """Insufficient-validation-specific policy:
- Bind the consumed value-sensitive object, its missing/inadequate check, and the causal outcome to the same object.
- Rank candidate objects and actively inspect no more than two or three diverse
  mechanism families in one pass. Put lower-ranked alternatives into a named
  follow-up request instead of diluting the decision across every candidate.
- Profit, flash capital, swaps, callbacks, minting, or value release alone do not establish a validation gap.
- Match guard scope to object semantics. Authorization/pause/reentrancy guards do not prove accounting proportionality or external-object validity; isContract does not prove allowlisting; balance/allowance does not prove token/path/source authenticity.
- Mark validation present/guarded only with positive evidence for the candidate's exact required dimension. Otherwise use wrong_scope or uncertain and inspect a named internal consumer when available."""
    if attack_label == "flashloans":
        return """Flash-loan-specific policy:
- Distinguish atomic borrowing/repayment from ordinary borrowing and from the downstream exploit mechanism.
- Require evidence of same-transaction capital provision and repayment/settlement for flash-capital conditions."""
    return ""


def _judge_family_followup_guidance(attack_label: str) -> str:
    if attack_label == "reentrancy":
        return (
            "- For unresolved stale-state ordering, fetch reentrancy_state_order_view; "
            "for one call subtree, use get_local_call_context."
        )
    if attack_label == "access_control":
        return (
            "- Use source only to resolve a located protected function's authority, "
            "visibility, modifier, or role semantics. Keep follow-up evidence on "
            "the same actor/capability/target chain already located by packet evidence."
        )
    if attack_label == "price_manipulation":
        return (
            "- Fetch reserve/oracle/formula evidence tied to the visible "
            "price-dependent operation, not generic fundflow."
        )
    if attack_label == "insufficient_validation":
        return (
            "- Follow the already selected object; do not switch objects merely "
            "because another tool view looks suspicious."
        )
    return ""


def _enforce_judge_prompt_budget(
    prompt: str,
    *,
    max_context_chars: int,
    evidence_context: str,
    previous_judge_summary_text: str,
    tool_observations_text: str,
) -> str:
    limit = max(512, int(max_context_chars or 512))
    value = str(prompt or "")
    if len(value) <= limit:
        return value

    sections = (
        (evidence_context, 512, "baseline evidence"),
        (previous_judge_summary_text, 128, "previous judge summary"),
        (tool_observations_text, 512, "latest tool observations"),
    )
    for section, minimum, label in sections:
        if not section or section not in value or len(value) <= limit:
            continue
        overflow = len(value) - limit
        target = max(minimum, len(section) - overflow - 96)
        if target >= len(section):
            continue
        replacement = renderer_truncate_text(section, target, label)
        value = value.replace(section, replacement, 1)

    if len(value) <= limit:
        return value

    marker = "\n...<prompt hard-truncated to max_context_chars>...\n"
    available = max(1, limit - len(marker))
    tail_chars = min(6000, max(256, available // 3))
    head_chars = max(0, available - tail_chars)
    return value[:head_chars] + marker + value[-tail_chars:]


def _format_evidence_context(
    evidence_packet: Dict[str, Any],
    max_view_chars: int = 20000,
    max_context_chars: int = 60000,
) -> str:
    text, _ = renderer_format_evidence_context_with_metadata(
        evidence_packet,
        max_view_chars=max_view_chars,
        max_context_chars=max_context_chars,
    )
    return text


def _format_evidence_context_with_metadata(
    evidence_packet: Dict[str, Any],
    max_view_chars: int = 20000,
    max_context_chars: int = 60000,
    packet_evidence_adequacy: Optional[Dict[str, Any]] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """Format selected evidence views into a bounded judge context."""
    return renderer_format_evidence_context_with_metadata(
        evidence_packet,
        max_view_chars=max_view_chars,
        max_context_chars=max_context_chars,
        packet_evidence_adequacy=packet_evidence_adequacy,
    )


def _visible_evidence_ids_in_prompt(
    prompt: str,
    *,
    evidence_packet: Dict[str, Any],
    tool_observations: List[Dict[str, Any]],
) -> List[str]:
    """Return evidence IDs that survived projection and final prompt truncation."""
    candidates = _collect_explicit_evidence_ids({
        "evidence_packet": evidence_packet,
        "tool_observations": tool_observations,
    })
    citable_text = _citable_prompt_sections(prompt)
    return [
        evidence_id
        for evidence_id in candidates
        if stable_json_dumps(evidence_id) in citable_text
    ]


def _citable_prompt_sections(prompt: str) -> str:
    value = str(prompt or "")
    baseline_marker = "Evidence (baseline packet context):\n"
    previous_marker = "\nPrevious Judge Summary:\n"
    observation_markers = (
        "\nFollow-up Evidence Observations (newest first):\n",
        "\nFollow-up Evidence Observations:\n",
    )
    terminal_marker = "\nReturn strict JSON:\n"
    sections: List[str] = []

    baseline_start = value.find(baseline_marker)
    if baseline_start >= 0:
        baseline_start += len(baseline_marker)
        end_candidates = [
            position
            for position in (
                value.find(previous_marker, baseline_start),
                *(value.find(marker, baseline_start) for marker in observation_markers),
                value.find(terminal_marker, baseline_start),
            )
            if position >= 0
        ]
        baseline_end = min(end_candidates) if end_candidates else len(value)
        sections.append(value[baseline_start:baseline_end])

    for marker in observation_markers:
        observation_start = value.find(marker)
        if observation_start < 0:
            continue
        observation_start += len(marker)
        observation_end = value.find(terminal_marker, observation_start)
        if observation_end < 0:
            observation_end = len(value)
        sections.append(value[observation_start:observation_end])
        break

    return "\n".join(sections)


def _collect_explicit_evidence_ids(value: Any) -> List[str]:
    collected: List[str] = []
    seen: set[str] = set()

    def add(candidate: Any) -> None:
        evidence_id = str(candidate or "").strip()
        if not evidence_id or evidence_id in seen:
            return
        seen.add(evidence_id)
        collected.append(evidence_id)

    def visit(item: Any, field_name: str = "") -> None:
        normalized_field = str(field_name or "").strip().lower()
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, str(key))
            return
        if isinstance(item, list):
            if normalized_field == "evidence_ids" or normalized_field.endswith(
                "_evidence_ids"
            ):
                for child in item:
                    if not isinstance(child, (dict, list)):
                        add(child)
                return
            for child in item:
                visit(child, normalized_field)
            return
        if normalized_field == "evidence_id" or normalized_field.endswith(
            "_evidence_id"
        ):
            add(item)
            return
        if normalized_field in {
            "call_id",
            "entry_call_id",
            "sensitive_call_id",
        } and str(item or "").strip().lower().startswith("call:"):
            add(item)

    visit(value)
    return collected


def _truncate_text(text: str, max_chars: int, label: str) -> str:
    return renderer_truncate_text(text, max_chars, label)


def _truncate_text_with_removed(text: str, max_chars: int, label: str) -> tuple[str, int]:
    return renderer_truncate_text_with_removed(text, max_chars, label)


def _render_metadata_row(
    view_name: str,
    rows_total: int,
    chars_before: int,
    chars_after: int,
    chars_removed: int,
    render_truncated: bool,
    packet_truncated: bool,
    reason: str,
) -> Dict[str, Any]:
    return {
        "view": view_name,
        "rows_total": rows_total,
        "rows_rendered": rows_total,
        "chars_before": chars_before,
        "chars_after": max(0, chars_after),
        "chars_removed": max(0, chars_removed),
        "prompt_render_truncated": bool(render_truncated),
        "render_truncated": bool(render_truncated),
        "packet_trace_truncated": bool(packet_truncated) if view_name == "trace_view" else False,
        "packet_truncated": bool(packet_truncated),
        "truncation_reason": reason,
    }


def _packet_trace_truncated(adequacy: Dict[str, Any]) -> bool:
    trace = adequacy.get("trace", {}) if isinstance(adequacy, dict) else {}
    return bool(trace.get("truncated"))


def _view_packet_truncated(view_data: Any) -> bool:
    if isinstance(view_data, dict):
        truncated = view_data.get("truncated")
        if isinstance(truncated, dict):
            return bool(truncated.get("omitted") or truncated.get("total", 0) > truncated.get("shown", 0))
        return bool(truncated)
    return False


def _try_extract_judge_json_object(
    text: str,
    *,
    state_output_schema: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    schema_validator = lambda data: _judge_response_schema_error(
        data,
        state_output_schema=state_output_schema,
    )
    data, error, metadata = extract_last_schema_valid_object(
        text,
        schema_validator,
        schema_name="Judge",
    )
    if error is None:
        return data, error, metadata

    completed, closure_count = _complete_truncated_judge_object(
        text,
        state_output_schema=state_output_schema,
    )
    if not completed:
        return data, error, metadata
    repaired, repaired_error, repaired_metadata = extract_last_schema_valid_object(
        completed,
        schema_validator,
        schema_name="Judge",
    )
    if repaired_error is not None:
        return data, error, metadata
    return repaired, None, {
        **repaired_metadata,
        "deterministic_syntax_repair": "close_unterminated_eof_container",
        "deterministic_syntax_closure_count": closure_count,
    }


def _complete_truncated_judge_object(
    text: str,
    *,
    state_output_schema: Optional[Dict[str, Any]] = None,
) -> tuple[str, int]:
    """Close only structurally complete JSON containers truncated at EOF."""
    raw = str(text or "").rstrip()
    starts = [index for index, char in enumerate(raw) if char == "{"]
    for start in reversed(starts):
        candidate = raw[start:]
        stack: List[str] = []
        in_string = False
        escaped = False
        invalid = False
        for char in candidate:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                expected = "{" if char == "}" else "["
                if not stack or stack[-1] != expected:
                    invalid = True
                    break
                stack.pop()
        if invalid or in_string or not stack or len(stack) > 8:
            continue
        suffix = "".join("}" if char == "{" else "]" for char in reversed(stack))
        completed = candidate + suffix
        try:
            parsed = json.loads(completed)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and _judge_response_schema_error(
            parsed,
            state_output_schema=state_output_schema,
        ) is None:
            return completed, len(stack)
    return "", 0


def _judge_response_schema_error(
    data: Dict[str, Any],
    *,
    state_output_schema: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Accept either a final judgment or a tool-only intermediate envelope."""
    if _is_tool_only_judge_envelope(data):
        return None
    return _judge_schema_error(data, state_output_schema=state_output_schema)


def _is_tool_only_judge_envelope(data: Any) -> bool:
    if not isinstance(data, dict) or any(
        key in data for key in ("answer", "confidence", "reason", "state_output")
    ):
        return False
    requests = data.get("tool_requests")
    if not isinstance(requests, list) or not requests:
        return False
    for request in requests:
        if not isinstance(request, dict):
            return False
        tool = str(
            request.get("tool")
            or request.get("name")
            or request.get("tool_name")
            or ""
        ).strip()
        if not tool:
            return False
        if "args" in request and not isinstance(request.get("args"), dict):
            return False
    return True


def _judge_schema_error(
    data: Dict[str, Any],
    *,
    state_output_schema: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not isinstance(data, dict):
        return "top-level value is not an object"
    _normalize_judge_output_structure_in_place(data)
    missing = [key for key in ("answer", "confidence", "reason") if key not in data]
    if missing:
        return f"missing required fields: {', '.join(missing)}"

    answer = data.get("answer")
    if not isinstance(answer, bool):
        normalized = str(answer or "").strip().lower()
        if normalized not in {"true", "false", "yes", "no", "uncertain"}:
            return "answer must be true, false, or uncertain"
    if str(data.get("confidence") or "").strip().lower() not in {
        "low",
        "medium",
        "high",
    }:
        return "confidence must be low, medium, or high"
    if not isinstance(data.get("reason"), str):
        return "reason must be a string"

    for key in (
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "missing_evidence",
        "suggested_followup_views",
        "tool_requests",
    ):
        if key in data and not isinstance(data.get(key), list):
            return f"{key} must be an array"
    if "condition_feature_analysis" in data and not isinstance(
        data.get("condition_feature_analysis"), dict
    ):
        return "condition_feature_analysis must be an object"
    expected_state = dict(state_output_schema or {})
    if expected_state:
        if not isinstance(data.get("state_output"), dict):
            return "state_output must be an object when state_output_schema is declared"
        state_error = _state_output_shape_error(
            data.get("state_output"),
            expected_state,
            path="state_output",
        )
        if state_error:
            return state_error
        exclusion_error = _access_control_exclusion_contract_error(
            data,
            expected_state,
        )
        if exclusion_error:
            return exclusion_error
    return None


def _access_control_exclusion_contract_error(
    data: Dict[str, Any],
    expected_state: Dict[str, Any],
) -> Optional[str]:
    """Validate AC exclusion Boolean/state agreement as an output contract."""
    state_output = data.get("state_output")
    if not isinstance(state_output, dict):
        return None
    for state_key, exemplar in expected_state.items():
        if not str(state_key).startswith("candidate_exclusion_summary__"):
            continue
        if not isinstance(exemplar, dict):
            continue
        exemplar_assessments = exemplar.get("candidate_assessments")
        if not (
            isinstance(exemplar_assessments, list)
            and exemplar_assessments
            and isinstance(exemplar_assessments[0], dict)
            and "same_chain_supported" in exemplar_assessments[0]
        ):
            continue
        summary = state_output.get(state_key)
        if not isinstance(summary, dict):
            continue
        assessments = list(summary.get("candidate_assessments") or [])
        confirmed_excluded = {
            str(item.get("candidate_id") or "").strip()
            for item in assessments
            if isinstance(item, dict)
            and str(item.get("candidate_id") or "").strip()
            and str(item.get("exclusion_status") or "").strip().lower()
            == "excluded"
            and item.get("same_chain_supported") is True
        }
        declared_excluded = {
            str(value or "").strip()
            for value in list(summary.get("excluded_candidate_ids") or [])
            if str(value or "").strip()
        }
        if declared_excluded != confirmed_excluded:
            return (
                f"state_output.{state_key}.excluded_candidate_ids must exactly "
                "match assessments with exclusion_status=excluded and "
                "same_chain_supported=true"
            )
        unresolved = {
            str(value or "").strip()
            for value in list(summary.get("unresolved_candidate_ids") or [])
            if str(value or "").strip()
        }
        answer = data.get("answer")
        if isinstance(answer, str):
            normalized = answer.strip().lower()
            if normalized in {"true", "yes"}:
                answer = True
            elif normalized in {"false", "no"}:
                answer = False
            else:
                answer = "uncertain"
        expected_answer: Union[bool, str] = (
            True if declared_excluded else "uncertain" if unresolved else False
        )
        if answer != expected_answer:
            return (
                f"answer must be {str(expected_answer).lower()} for "
                f"state_output.{state_key}: true iff excluded_candidate_ids is "
                "non-empty, uncertain iff only unresolved candidates remain"
            )
    return None


def _state_output_shape_error(
    value: Any,
    exemplar: Any,
    *,
    path: str,
) -> Optional[str]:
    """Validate the required shape expressed by a Plan state exemplar."""
    if isinstance(exemplar, dict):
        if not isinstance(value, dict):
            return f"{path} must be an object"
        for key, child_exemplar in exemplar.items():
            if key not in value:
                return f"{path}.{key} is required by state_output_schema"
            error = _state_output_shape_error(
                value.get(key),
                child_exemplar,
                path=f"{path}.{key}",
            )
            if error:
                return error
        return None
    if isinstance(exemplar, list):
        if not isinstance(value, list):
            return f"{path} must be an array"
        if exemplar:
            item_exemplar = exemplar[0]
            for index, item in enumerate(value):
                error = _state_output_shape_error(
                    item,
                    item_exemplar,
                    path=f"{path}[{index}]",
                )
                if error:
                    return error
        return None
    if isinstance(exemplar, bool):
        return None if isinstance(value, bool) else f"{path} must be a boolean"
    if isinstance(exemplar, int) and not isinstance(exemplar, bool):
        return (
            None
            if isinstance(value, int) and not isinstance(value, bool)
            else f"{path} must be an integer"
        )
    if isinstance(exemplar, float):
        return (
            None
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else f"{path} must be a number"
        )
    if isinstance(exemplar, str):
        return None if isinstance(value, str) else f"{path} must be a string"
    return None


def _normalize_judge_output_structure_in_place(data: Dict[str, Any]) -> None:
    """Repair known output-shape mistakes without changing Judge semantics."""
    repairs = list(data.get("_deterministic_schema_repairs") or [])
    aliases = {
        "contrading_evidence_ids": "contradicting_evidence_ids",
    }
    for wrong_key, canonical_key in aliases.items():
        if wrong_key not in data:
            continue
        if canonical_key not in data:
            data[canonical_key] = data.get(wrong_key)
        data.pop(wrong_key, None)
        repairs.append(f"field_alias:{wrong_key}->{canonical_key}")

    feature_analysis = data.get("condition_feature_analysis")
    if isinstance(feature_analysis, dict) and isinstance(
        feature_analysis.get("state_output"),
        dict,
    ):
        if not isinstance(data.get("state_output"), dict):
            data["state_output"] = feature_analysis.get("state_output")
            repairs.append("hoist_state_output_from_condition_feature_analysis")
        feature_analysis.pop("state_output", None)

    for key in (
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "missing_evidence",
        "suggested_followup_views",
        "tool_requests",
    ):
        data.setdefault(key, [])
    if repairs:
        data["_deterministic_schema_repairs"] = list(dict.fromkeys(repairs))


def _with_normalized_condition_feature_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data or {})
    normalized["condition_feature_analysis"] = normalize_condition_feature_analysis(
        normalized.get("condition_feature_analysis")
    )
    return normalized


def _condition_feature_analysis_parse_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = data.get("condition_feature_analysis") if isinstance(data, dict) else None
    if raw is None:
        status = "missing"
    elif not isinstance(raw, dict):
        status = "invalid_type"
    else:
        status = "non_empty" if _condition_feature_analysis_non_empty(raw) else "empty"
    normalized = normalize_condition_feature_analysis(raw)
    return {
        "feature_analysis_parse_status": status,
        "feature_analysis_present": isinstance(raw, dict),
        "feature_analysis_nonempty": _condition_feature_analysis_non_empty(normalized),
        "feature_analysis_keys_present": sorted(raw.keys()) if isinstance(raw, dict) else [],
    }


def _condition_feature_analysis_non_empty(value: Any) -> bool:
    normalized = normalize_condition_feature_analysis(value)
    return any(bool(items) for items in normalized.values())


def _invalid_json_uncertain(
    initial_error: str,
    repair_error: Optional[str] = None,
) -> Dict[str, Any]:
    reason = "Judge returned invalid JSON"
    if repair_error:
        reason += " and JSON repair failed"
    return {
        "answer": "uncertain",
        "confidence": "low",
        "reason": f"{reason}.",
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "missing_evidence": ["valid judge output"],
        "suggested_followup_views": [],
        "tool_requests": [],
        "condition_feature_analysis": normalize_condition_feature_analysis({}),
        "state_output": {},
        "parse_error": {
            "initial_error": initial_error,
            "repair_error": repair_error or "",
        },
    }


def _build_json_repair_prompt(raw_text: str, error: str) -> str:
    return f"""The previous answer was intended to be JSON but could not be parsed.

Parse error:
{error}

Repair the answer into one strict JSON object only. Do not add <think> tags,
chain-of-thought, commentary, markdown fences, JSON examples, or extra text.
Begin the response with "{{" and end it with "}}". Preserve the intended fields when possible.
If a field is missing, use a safe default.
condition_feature_analysis is required. Preserve any non-empty feature analysis
from the raw answer. If the raw answer contains enough diagnostic text, fill
concise abstract feature strings; otherwise use the empty fixed structure.

Required schema:
{{
  "answer": true | false | "uncertain",
  "confidence": "low" | "medium" | "high",
  "reason": "...",
  "supporting_evidence_ids": [],
  "contradicting_evidence_ids": [],
  "missing_evidence": [],
  "suggested_followup_views": [],
  "tool_requests": [],
{CONDITION_FEATURE_ANALYSIS_SCHEMA_FIELD},
  "state_output": {{}}
}}

Raw previous answer:
{_truncate_text(str(raw_text or ""), 12000, "raw invalid judge output")}
"""


def _build_schema_only_judge_retry_prompt(prompt: str) -> str:
    return f"""{str(prompt or '').rstrip()}

Your previous response could not be parsed after one JSON repair attempt.
Re-evaluate the same local condition from the evidence above. Return exactly one
JSON object matching the required Judge schema. Do not emit markdown, prose,
draft objects, or <think> tags. Begin with {{ and end with }}.
""".strip()


def _extract_tool_requests(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    raw_requests = data.get("tool_requests")
    if isinstance(raw_requests, list):
        candidates = raw_requests
    elif isinstance(raw_requests, dict):
        candidates = [raw_requests]
    else:
        legacy = data.get("tool_request") or data.get("tool_call")
        candidates = [legacy] if isinstance(legacy, dict) else []
        if str(data.get("tool") or "").strip():
            candidates.append(data)

    for item in candidates:
        if not isinstance(item, dict):
            continue
        tool = str(
            item.get("tool") or item.get("name") or item.get("tool_name") or ""
        ).strip()
        if not tool:
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        if not args:
            args = {
                key: value
                for key, value in item.items()
                if key not in {
                    "tool",
                    "name",
                    "tool_name",
                    "reason",
                    "tool_request",
                    "tool_call",
                    "tool_requests",
                    "condition_feature_analysis",
                    "state_output",
                }
            }
        requests.append({
            "tool": tool,
            "args": dict(args),
            "reason": str(item.get("reason", "")),
        })
    return requests


def _infer_chain_from_evidence(evidence_packet: Dict[str, Any]) -> Optional[str]:
    tx_card = evidence_packet.get("tx_card")
    if isinstance(tx_card, dict):
        chain = tx_card.get("chain")
        if chain:
            return str(chain)
    return None
