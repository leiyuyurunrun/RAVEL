from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import uuid

from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import EvolvingRule, RuleCondition
from evotx.utils.json_utils import extract_json_object, repair_likely_mojibake


DEFAULT_RULE_COMPLEXITY_BUDGET: Dict[str, int] = {
    "max_core_conditions": 3,
    "max_exclusion_conditions": 2,
    "max_total_conditions": 5,
    "max_source_dependent_conditions": 2,
}


LABEL_RULE_COMPLEXITY_BUDGET: Dict[str, Dict[str, Any]] = {
    "access_control": {
        "max_source_dependent_conditions": 5,
        "source_dependent_overage_policy": "warning",
    },
    "insufficient_validation": {
        "max_source_dependent_conditions": 5,
        "source_dependent_overage_policy": "warning",
    },
    "protocol_accounting_exploitation": {
        "max_source_dependent_conditions": 4,
        "source_dependent_overage_policy": "warning",
    },
    "token_semantic_exploitation": {
        "max_source_dependent_conditions": 4,
        "source_dependent_overage_policy": "warning",
    },
    "reentrancy": {
        "max_source_dependent_conditions": 3,
        "source_dependent_overage_policy": "warning",
    },
    "market_manipulation": {
        "max_source_dependent_conditions": 5,
        "source_dependent_overage_policy": "warning",
    },
    "price_manipulation": {
        "max_source_dependent_conditions": 5,
        "source_dependent_overage_policy": "warning",
    },
}


_RULE_BUDGET_REASON_FIELDS: Dict[str, str] = {
    "core_conditions": "core_conditions",
    "exclusion_conditions": "exclusion_conditions",
    "total_conditions": "total_conditions",
    "source_dependent_conditions": "source_dependent_conditions",
}


def baseline_relative_rule_budget_decision(
    previous_complexity: Dict[str, Any],
    candidate_complexity: Dict[str, Any],
) -> Dict[str, Any]:
    """Allow inherited hard overages only when the candidate does not worsen them.

    Absolute budgets continue to reject every new or enlarged overage. This only
    prevents a semantic rewrite from being frozen because its accepted parent
    already exceeded a newer budget.
    """
    candidate_reasons = list(candidate_complexity.get("hard_over_budget_reasons") or [])
    worsened: List[str] = []
    inherited: List[str] = []
    for reason in candidate_reasons:
        field = _RULE_BUDGET_REASON_FIELDS.get(str(reason))
        if not field:
            worsened.append(str(reason))
            continue
        before = int(previous_complexity.get(field, 0) or 0)
        after = int(candidate_complexity.get(field, 0) or 0)
        if after > before:
            worsened.append(str(reason))
        else:
            inherited.append(str(reason))
    hard_over_budget = bool(candidate_complexity.get("hard_over_budget"))
    return {
        "hard_over_budget": hard_over_budget,
        "candidate_allowed": not hard_over_budget or not worsened,
        "baseline_relative_grandfathered": hard_over_budget and not worsened,
        "inherited_hard_over_budget_reasons": inherited,
        "new_or_worsened_hard_over_budget_reasons": worsened,
    }


def condition_from_dict(data: Dict[str, Any] | RuleCondition) -> RuleCondition:
    return RuleCondition.from_dict(data)


def rule_from_dict(data: Dict[str, Any] | EvolvingRule) -> EvolvingRule:
    return EvolvingRule.from_dict(data)


def make_rule(
    name: str,
    description: str,
    conditions: Iterable[Dict[str, Any] | RuleCondition],
    exclusion_conditions: Optional[Iterable[Dict[str, Any] | RuleCondition]] = None,
    decision_policy: str = "",
    rule_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvolvingRule:
    """Create a version-1 evolving rule from local evidence conditions."""
    metadata = dict(metadata or {})
    if metadata.get("attack_label"):
        raw_label = metadata.get("raw_attack_label") or metadata.get("attack_label")
        metadata.setdefault("raw_attack_label", raw_label)
        metadata.setdefault("display_label", raw_label)
        metadata["attack_label"] = normalize_attack_label(metadata.get("attack_label"))
    rule = EvolvingRule(
        rule_id=rule_id or "rule_" + uuid.uuid4().hex[:10],
        version=1,
        name=name,
        description=description,
        conditions=[RuleCondition.from_dict(x) for x in conditions],
        exclusion_conditions=[
            RuleCondition.from_dict(x) for x in (exclusion_conditions or [])
        ],
        decision_policy=decision_policy,
        metadata=metadata,
    )
    _attach_rule_complexity(rule)
    return rule


def rule_complexity(rule: EvolvingRule) -> Dict[str, Any]:
    """Return a compact rule-cost estimate used by evolution/reporting."""
    conditions = list(rule.conditions or [])
    exclusions = list(rule.exclusion_conditions or [])
    metadata = dict(rule.metadata or {})
    attack_label = _make_attack_label(str(metadata.get("attack_label", "") or ""))
    strict_source_budget = bool(metadata.get("strict_source_budget", False)) or (
        str(metadata.get("rule_budget_mode") or "").strip().lower() == "strict"
    )
    label_aware_source_budget = bool(metadata.get("label_aware_source_budget", True))
    allow_source_budget_warning = bool(metadata.get("allow_source_budget_warning", True))
    source_levels = {"none": 0, "helpful": 0, "required": 0}
    condition_levels: Dict[str, str] = {}
    try:
        from evotx.core.plan import source_dependency_level

        for condition in [*conditions, *exclusions]:
            level = source_dependency_level(condition.description, attack_label)
            if level not in source_levels:
                level = "none"
            condition_levels[str(condition.id)] = level
            source_levels[level] += 1
    except Exception:
        condition_levels = {}
        source_levels = {"none": 0, "helpful": 0, "required": 0}

    source_dependent = source_levels["helpful"] + source_levels["required"]
    budget = dict(DEFAULT_RULE_COMPLEXITY_BUDGET)
    label_profile = "default"
    source_dependent_overage_policy = "hard"
    if (
        not strict_source_budget
        and label_aware_source_budget
        and attack_label in LABEL_RULE_COMPLEXITY_BUDGET
    ):
        label_budget = dict(LABEL_RULE_COMPLEXITY_BUDGET[attack_label])
        budget.update({
            key: int(value)
            for key, value in label_budget.items()
            if key.startswith("max_")
        })
        source_dependent_overage_policy = str(
            label_budget.get("source_dependent_overage_policy") or "hard"
        ).strip().lower()
        label_profile = attack_label
    if strict_source_budget or not allow_source_budget_warning:
        source_dependent_overage_policy = "hard"
        label_profile = "strict" if strict_source_budget else label_profile

    max_core = int(budget["max_core_conditions"])
    max_exclusions = int(budget["max_exclusion_conditions"])
    max_total = int(budget["max_total_conditions"])
    max_source_dependent = int(budget["max_source_dependent_conditions"])
    total = len(conditions) + len(exclusions)
    hard_over_budget_reasons: List[str] = []
    soft_over_budget_reasons: List[str] = []
    if len(conditions) > max_core:
        hard_over_budget_reasons.append("core_conditions")
    if len(exclusions) > max_exclusions:
        hard_over_budget_reasons.append("exclusion_conditions")
    if total > max_total:
        hard_over_budget_reasons.append("total_conditions")
    default_max_source_dependent = int(
        DEFAULT_RULE_COMPLEXITY_BUDGET["max_source_dependent_conditions"]
    )
    source_warning_threshold = (
        default_max_source_dependent
        if source_dependent_overage_policy == "warning"
        else max_source_dependent
    )
    if source_dependent > source_warning_threshold:
        if source_dependent_overage_policy == "warning":
            soft_over_budget_reasons.append("source_dependent_conditions")
        else:
            hard_over_budget_reasons.append("source_dependent_conditions")
    hard_over_budget = bool(hard_over_budget_reasons)
    soft_over_budget = bool(soft_over_budget_reasons)
    budget_status = (
        "hard_reject"
        if hard_over_budget
        else "soft_warning"
        if soft_over_budget
        else "ok"
    )
    over_budget_reasons = [*hard_over_budget_reasons, *soft_over_budget_reasons]
    return {
        "core_conditions": len(conditions),
        "exclusion_conditions": len(exclusions),
        "total_conditions": total,
        "estimated_judge_steps": total,
        "source_dependent_conditions": source_dependent,
        "source_dependent_helpful_conditions": source_levels["helpful"],
        "source_dependent_required_conditions": source_levels["required"],
        "source_dependency_breakdown": {
            "none": source_levels["none"],
            "helpful": source_levels["helpful"],
            "required": source_levels["required"],
            "condition_levels": condition_levels,
        },
        "budget": {
            **budget,
            "default_max_source_dependent_conditions": default_max_source_dependent,
            "source_warning_threshold": source_warning_threshold,
        },
        "hard_over_budget": hard_over_budget,
        "soft_over_budget": soft_over_budget,
        "hard_over_budget_reasons": hard_over_budget_reasons,
        "soft_over_budget_reasons": soft_over_budget_reasons,
        "budget_status": budget_status,
        "label_budget_profile": label_profile,
        "source_dependent_overage_policy": source_dependent_overage_policy,
        "budget_mode": "strict" if strict_source_budget else "label_aware",
        "candidate_allowed": not hard_over_budget,
        "over_budget": bool(over_budget_reasons),
        "over_budget_reasons": over_budget_reasons,
    }


def _attach_rule_complexity(rule: EvolvingRule) -> None:
    metadata = dict(rule.metadata or {})
    metadata["rule_complexity"] = rule_complexity(rule)
    rule.metadata = metadata


# ---------------------------------------------------------------------------
# Label-specific fallback rules
# ---------------------------------------------------------------------------

def _generic_fallback(
    attack_description: str,
    attack_label: str,
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    base_description = (
        attack_description.strip()
        if attack_description and attack_description.strip()
        else "General transaction-level attack detection"
    )
    name = rule_name or _make_rule_name(base_description)
    return make_rule(
        rule_id=rule_id or _make_rule_id(name),
        name=name,
        description=(
            f"Detect {base_description} by requiring concrete evidence of "
            "abnormal asset movement, a suspicious execution context, and a "
            "state or event-level effect that is not fully accounted for by a "
            "benign pattern."
        ),
        conditions=[
            RuleCondition(
                id="C1",
                description=(
                    "The transaction contains concrete abnormal fund-flow evidence, "
                    "such as attacker-side profit, victim-side loss, or a short-lived "
                    "asset disturbance that is material to the transaction."
                ),
            ),
            RuleCondition(
                id="C2",
                description=(
                    "The call trace or local call context supports a suspicious "
                    "attacker-controlled execution step."
                ),
            ),
            RuleCondition(
                id="C3",
                description=(
                    "State changes, event logs, or accounting signals support that "
                    "the transaction produced a protocol-level effect rather than "
                    "only routine routing."
                ),
            ),
        ],
        exclusion_conditions=[
            RuleCondition(
                id="E1",
                description=(
                    "The observed behavior is fully accounted for by a benign pattern "
                    "such as normal arbitrage, liquidation, governance execution, "
                    "multisig operation, oracle update, router callback, or migration."
                ),
            )
        ],
        decision_policy=(
            "Emit attack only when the required evidence conditions are satisfied "
            "and no stronger benign exclusion explains the transaction. If evidence "
            "is missing or contradictory, keep the result uncertain."
        ),
        metadata={
            "source": "cold_start_fallback",
            "attack_description": base_description,
            "attack_label": attack_label or _make_attack_label(base_description),
        },
    )


def _fallback_rule(
    *,
    label: str,
    rule_id: str | None,
    rule_name: str | None,
    default_rule_id: str,
    default_name: str,
    description: str,
    conditions: List[str],
    exclusions: List[str],
    decision_policy: str,
) -> EvolvingRule:
    return make_rule(
        rule_id=rule_id or default_rule_id,
        name=rule_name or default_name,
        description=description,
        conditions=[
            RuleCondition(id=f"C{i + 1}", description=text, expected_answer=True)
            for i, text in enumerate(conditions)
        ],
        exclusion_conditions=[
            RuleCondition(id=f"E{i + 1}", description=text, expected_answer=True)
            for i, text in enumerate(exclusions)
        ],
        decision_policy=decision_policy,
        metadata={"source": "cold_start_fallback", "attack_label": label},
    )


def _price_manipulation_fallback(
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    return _fallback_rule(
        label="price_manipulation",
        rule_id=rule_id,
        rule_name=rule_name,
        default_rule_id="price_manipulation_evidence_rule",
        default_name="Price Manipulation Transaction Evidence Rule",
        description=(
            "Detect transaction-level evidence that a price-relevant source was "
            "temporarily perturbed and then consumed by a value-sensitive protocol "
            "operation, producing attacker-favorable or protocol-adverse effects."
        ),
        conditions=[
            "The transaction shows temporary perturbation or distortion of a price-relevant source, such as an AMM reserve, pool balance, token balance, vault balance, oracle-read value, LP/share-price input, exchange-rate input, or protocol accounting value.",
            "A causally linked value-sensitive operation consumes or depends on the perturbed value. The consumer may be a downstream protocol operation or a same-market swap, settlement, or extraction whose amount is determined by an abnormal distorted reserve or price; the actor's ordinary sequential swaps merely repricing against normal reserve updates, a bare skim, sync, reserve write, transfer, or swap co-occurrence are insufficient.",
            "The transaction produces a material attacker-favorable or protocol-adverse outcome, such as asset extraction, inflated minted or redeemed assets, undercollateralized borrow, abnormal reward, victim/protocol asset loss, or adverse accounting shift.",
        ],
        exclusions=[
            "The trace shows only a bare skim/sync, closed-loop DEX arbitrage, ordinary large swap, normal liquidation, or routine AMM behavior, and no value-sensitive same-market or downstream operation actually consumes a distorted price-related value.",
            "The trace shows authorized governance, admin, keeper, oracle maintenance, or legitimate capital-backed contribution, repayment, deposit, redeem, or collateral activity that fully supports the payout.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported and neither E1 nor "
            "E2 is supported. Temporal or causal consistency between perturbation, "
            "consumption, and outcome supports the rule, but flashloan or atomic "
            "capital is not required."
        ),
    )


def _reentrancy_fallback(
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    return _fallback_rule(
        label="reentrancy",
        rule_id=rule_id,
        rule_name=rule_name,
        default_rule_id="reentrancy_evidence_rule",
        default_name="Reentrancy Transaction Evidence Rule",
        description=(
            "Detect transaction-level evidence of nested callback or repeated entry "
            "into a sensitive function before outer execution completes, causing "
            "asset release or accounting inconsistency."
        ),
        conditions=[
            "The transaction contains a nested callback or repeated entry into the same or logically related sensitive function before the outer execution completes.",
            "Asset release, mint, withdraw, redeem, claim, share accounting, debt/collateral update, or other value-relevant effect occurs during or because of the nested execution.",
            "State or accounting order, stale state, or repeated state transition supports that nested execution changes value.",
        ],
        exclusions=[
            "The trace shows only a normal protocol callback, router multicall, flashloan callback, DEX callback, or token hook without repeated sensitive value release or stale-state exploitation.",
            "The trace shows a normal single claim, withdraw, redeem, borrow, mint, governance, admin, or migration path with expected entitlement and no repeated sensitive path.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported and neither E1 nor E2 is "
            "supported. If any core condition is uncertain or contradictory, keep the "
            "verdict uncertain."
        ),
    )


def _access_control_fallback(
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    return _fallback_rule(
        label="access_control",
        rule_id=rule_id,
        rule_name=rule_name,
        default_rule_id="access_control_evidence_rule",
        default_name="Access Control Transaction Evidence Rule",
        description=(
            "Detect transaction-level evidence that a protected authorization "
            "boundary is missing or bypassed, allowing an actor to execute a "
            "sensitive operation or release protocol-controlled value."
        ),
        conditions=[
            "The transaction exposes one concrete actor/caller or beneficiary reaching a plausibly sensitive capability or protected target/resource. C1 locates the candidate and does not decide whether authorization is missing or the operation is legitimately permissionless.",
            "For that same candidate, observable source or behavioral evidence shows the required authority is missing, bypassed, self-granted, or otherwise not held by the actor for the sensitive capability and target/resource.",
            "The same unauthorized candidate causes a protected state, configuration, approval, external-call, or value-release effect on that target/resource beyond legitimate entitlement or contribution.",
        ],
        exclusions=[
            "Observable evidence shows a recognized authorized role, governance, timelock, multisig, keeper, admin, valid signature, delegation, or user-owned position explains the sensitive action.",
            "Observable evidence shows the protected authorization-boundary mechanism required by the positive conditions is absent: the action is fully entitlement-backed or user-owned, or another primary mechanism, such as validation of user-supplied target/data/amount/path/signature/return-value semantics, completely explains the root cause and there is no caller/authorization boundary for a privileged capability missing or bypassed. Public function names, ordinary callback shape, valid-looking signatures, or profit_loss_view hints alone are not enough.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported and neither E1 nor E2 "
            "is supported."
        ),
    )


def _insufficient_validation_fallback(
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    return _fallback_rule(
        label="insufficient_validation",
        rule_id=rule_id,
        rule_name=rule_name,
        default_rule_id="insufficient_validation_evidence_rule",
        default_name="Insufficient Validation Transaction Evidence Rule",
        description=(
            "Detect transaction-level evidence that a protocol accepts invalid, "
            "fake, inconsistent, adversarial, stale, or semantically unsafe data "
            "or state before executing value-sensitive logic."
        ),
        conditions=[
            "A value-sensitive protocol path consumes a specific user-supplied or externally sourced data/state object whose semantic validity should be checked before it controls sensitive logic.",
            "A concrete validation check on that object is absent or inadequate, such as bounds/range, freshness, format, source, entitlement, balance, proportionality, or invariant consistency, so invalid/adversarial/stale/inconsistent data is accepted.",
            "The same unvalidated object accepted in C2 causally drives an incorrect outcome, excessive payout, unauthorized asset release, invariant violation, or state/accounting corruption.",
        ],
        exclusions=[
            "Observable evidence shows the primary root cause is caller, owner, role, whitelist, or callback-sender authorization failure on a protected path, even if secondary data/state/value effects appear, rather than validation of data content or state consistency.",
            "The primary root cause is independent market/oracle/price distortion, pure reentrancy/control-flow, token-semantic mismatch, accounting formula/order bug, direct storage corruption, or overflow/underflow, not the C2 unvalidated object; routine storage reads or token transfers do not satisfy this exclusion when consumed state violates business invariants.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported and neither E1 nor E2 "
            "is supported."
        ),
    )


def _market_manipulation_fallback(
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    return _fallback_rule(
        label="market_manipulation",
        rule_id=rule_id,
        rule_name=rule_name,
        default_rule_id="market_manipulation_evidence_rule",
        default_name="Market Manipulation Transaction Evidence Rule",
        description=(
            "Detect transaction-visible market-facing exploitation across AMM/DEX, "
            "oracle, slippage, skim/donate, sandwich, MEV, and arbitrage-like "
            "mechanisms. The evidence should bind one concrete market mechanism "
            "profile to the consumed market state and the resulting value outcome."
        ),
        conditions=[
            "The transaction establishes a concrete market-mechanism profile: price-state manipulation (AMM/oracle/slippage/sandwich), pool-accounting or reserve extraction (skim/donate/sync/pair-visible balance), or ordering/MEV/arbitrage extraction. C1 must identify the affected market source, pool/pair/oracle/balance, token side, and manipulation window; generic swaps, flash capital, profit, or large transfers alone are insufficient.",
            "A value-sensitive operation consumes or acts on the same market-mechanism profile selected by C1, such as swap, mint, burn, redeem, borrow, liquidation, settlement, valuation, skim, sync, donation settlement, sandwich victim execution, MEV ordering, or arbitrage realization. AMM-internal skim/donate/sync can qualify only for the pool-accounting/reserve-extraction profile; otherwise C2 should require an external or downstream consumer of the distorted market state.",
            "The same selected market-mechanism profile causally produces attacker-favorable or protocol/victim-adverse value movement, forced slippage, reserve extraction, distorted settlement, oracle/valuation effect, or accounting effect. C3 must distinguish disproportionate or profile-specific extraction from ordinary balanced swaps, routine arbitrage without exploitation, token-mechanics-only effects, or unrelated profit.",
        ],
        exclusions=[
            "Observable evidence shows only a normal large swap, routine liquidation, maintenance sync, keeper/oracle upkeep, or capital-backed pool operation with no selected market-mechanism profile and no profile-specific extraction. Do not exclude sandwich, MEV, skim/donate, or arbitrage-like behavior merely by name when it is the selected target mechanism.",
            "Observable evidence shows the primary root cause and value outcome are completely explained by token semantic mismatch, access-control failure, pure validation failure, reentrancy, or internal accounting formula abuse, without the selected market-mechanism profile being causal.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported for the same selected "
            "market-mechanism profile and neither E1 nor E2 is supported."
        ),
    )


def _protocol_accounting_exploitation_fallback(
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    return _fallback_rule(
        label="protocol_accounting_exploitation",
        rule_id=rule_id,
        rule_name=rule_name,
        default_rule_id="protocol_accounting_exploitation_evidence_rule",
        default_name="Protocol Accounting Exploitation Transaction Evidence Rule",
        description=(
            "Detect transaction-level evidence that a protocol's own bookkeeping "
            "state, update order, or formula is abused to obtain value beyond "
            "legitimate contribution."
        ),
        conditions=[
            "The transaction identifies one concrete protocol-internal bookkeeping candidate, such as shares, rewards, debt, collateral, vault balance, eligibility, claim counter, exchange rate, strategy accounting, or accounting index, whose update order, formula, eligibility, or state transition is skipped, stale, reused, inconsistent, or otherwise flawed. C1 must bind the candidate to a protocol component and evidence ids; generic profit, swaps, token transfer volume, or external price movement alone are insufficient.",
            "A protocol operation consumes or relies on the same bookkeeping candidate selected by C1, such as claim, withdraw, borrow, mint, redeem, stake, unstake, liquidation, strategy action, or reward distribution, and the evidence shows the consumed accounting state or entitlement is not reconciled with legitimate contribution, balance, debt, collateral, or eligibility.",
            "The same bookkeeping candidate and consumption path causally produce a protocol-adverse or attacker-favorable outcome, such as disproportionate payout, repeated claim, inflated shares, undercollateralized borrow, incorrect withdrawal, excess reward/yield, bad debt, abnormal vault delta, or protocol loss. C3 must distinguish candidate-specific accounting impact from ordinary value movement or final profit.",
        ],
        exclusions=[
            "Observable evidence shows the same payout, credit, share, debt, collateral, reward, or withdrawal is fully explained by legitimate contribution, collateral, repayment, balance, eligibility, or normal protocol accounting for the selected candidate.",
            "Observable evidence shows the primary mechanism and outcome are fully explained by market/oracle manipulation, token semantic mismatch, access control, insufficient validation, classic reentrancy/state-order abuse, or pure flashloan capital use, without the selected protocol bookkeeping candidate being the root cause.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported and neither E1 nor E2 is "
            "supported."
        ),
    )


def _token_semantic_exploitation_fallback(
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    return _fallback_rule(
        label="token_semantic_exploitation",
        rule_id=rule_id,
        rule_name=rule_name,
        default_rule_id="token_semantic_exploitation_evidence_rule",
        default_name="Token Semantic Exploitation Transaction Evidence Rule",
        description=(
            "Detect transaction-level evidence that non-standard token behavior "
            "creates a mismatch between nominal token operations and actual balance, "
            "supply, reserve, or protocol accounting effects."
        ),
        conditions=[
            "Token behavior differs from nominal ERC20-style assumptions, such as fee-on-transfer, reflection or rebase, self-transfer effect, locked or excluded balance exposure, special balanceOf behavior, transfer hook, mint/burn/tax side effect, or non-standard accounting.",
            "Another protocol trusts nominal transfer, event, balance, supply, reserve, or share behavior and records or consumes an incorrect amount, balance, supply, reserve, or share.",
            "The mismatch causes attacker-favorable extraction, pool/protocol imbalance, inflated shares, incorrect payout, or accounting drift.",
        ],
        exclusions=[
            "Observable evidence shows the receiving protocol accounts for actual received amounts correctly and no token/accounting mismatch exists.",
            "Observable evidence shows the primary mechanism is pure market manipulation, access control, insufficient validation, protocol accounting formula abuse, or reentrancy.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported and neither E1 nor E2 is "
            "supported."
        ),
    )


def _flashloans_fallback(
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    return _fallback_rule(
        label="flashloans",
        rule_id=rule_id,
        rule_name=rule_name,
        default_rule_id="flashloans_evidence_rule",
        default_name="Flashloans Transaction Evidence Rule",
        description=(
            "Detect transaction-level evidence that temporary atomic capital "
            "is a necessary causal enabler for an exploit mechanism, not merely "
            "incidental capital for ordinary arbitrage, liquidation, routing, or "
            "another primary attack family."
        ),
        conditions=[
            "The transaction contains a concrete temporary atomic-capital candidate such as flash loan, flash swap, flash mint, or same-transaction borrow-repay/settlement pattern. C1 must identify the provider or source, borrower or callback context, asset, amount or scale when available, repayment/settlement/unwind evidence, and supporting evidence ids; a large swap, callback, or profit alone is insufficient.",
            "The same temporary atomic-capital candidate selected by C1 is the primary enabler of an exploit mechanism: validation bypass, accounting amplification, oracle/price-state distortion, solvency or collateral assumption break, callback/repayment-check abuse, temporary voting/balance/liquidity power, or another protocol-sensitive effect. C2 must distinguish flash capital as the weapon from flash capital as incidental financing for ordinary arbitrage, liquidation, routing, or a different primary attack.",
            "The same temporary atomic-capital candidate and exploit mechanism causally produce material attacker-favorable value movement, protocol/user loss, bad debt, excess claim, distorted settlement, or attacker-favorable state after repayment/settlement/unwind. C3 must not answer true from generic profit or value release unless the outcome depends on the C1/C2 flash-capital chain.",
        ],
        exclusions=[
            "Observable evidence shows the temporary capital is only incidental financing for ordinary arbitrage, liquidation, routing, refinancing, or capital-backed execution, and no protocol-sensitive exploit mechanism depends on it.",
            "Observable evidence shows the primary root cause and value outcome are fully explained by access control, insufficient validation, reentrancy, token semantic mismatch, market manipulation, or protocol accounting exploitation independent of temporary atomic capital as the necessary causal enabler.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported for the same temporary "
            "atomic-capital candidate and neither E1 nor E2 is supported."
        ),
    )


# ---------------------------------------------------------------------------
# Cold-start rule generation
# ---------------------------------------------------------------------------

def make_cold_start_rule(
    attack_description: str | None = None,
    attack_label: str | None = None,
    rule_name: str | None = None,
    rule_id: str | None = None,
) -> EvolvingRule:
    """A conservative fallback rule for zero-shot EvoTx cold start."""
    label = _make_attack_label(attack_label or attack_description or "")
    if label == "price_manipulation":
        return _price_manipulation_fallback(rule_id=rule_id, rule_name=rule_name)
    if label == "reentrancy":
        return _reentrancy_fallback(rule_id=rule_id, rule_name=rule_name)
    if label == "access_control":
        return _access_control_fallback(rule_id=rule_id, rule_name=rule_name)
    if label == "insufficient_validation":
        return _insufficient_validation_fallback(rule_id=rule_id, rule_name=rule_name)
    if label == "market_manipulation":
        return _market_manipulation_fallback(rule_id=rule_id, rule_name=rule_name)
    if label == "protocol_accounting_exploitation":
        return _protocol_accounting_exploitation_fallback(
            rule_id=rule_id,
            rule_name=rule_name,
        )
    if label == "token_semantic_exploitation":
        return _token_semantic_exploitation_fallback(
            rule_id=rule_id,
            rule_name=rule_name,
        )
    if label == "flashloans":
        return _flashloans_fallback(rule_id=rule_id, rule_name=rule_name)
    desc = attack_description or label or "General transaction-level attack detection"
    return _generic_fallback(
        attack_description=desc,
        attack_label=label or "attack",
        rule_id=rule_id,
        rule_name=rule_name,
    )


def generate_cold_start_rule(
    attack_description: str,
    llm=None,
    human_example: str = "",
    attack_label: str | None = None,
    rule_id: str | None = None,
    rule_name: str | None = None,
) -> EvolvingRule:
    """
    Build the initial evolving rule from model knowledge.

    The human example is treated as a sketch or orientation aid, not as the rule
    itself and not as a transaction trace.
    """
    if llm is None:
        return make_cold_start_rule(
            attack_description=attack_description,
            attack_label=attack_label,
            rule_name=rule_name,
            rule_id=rule_id,
        )

    raw_label = attack_label or _make_attack_label(attack_description)
    label = normalize_attack_label(raw_label)
    prompt = _build_cold_start_prompt(
        attack_description=attack_description,
        label=label,
        human_example=human_example,
        rule_id=rule_id or _make_rule_id(rule_name or attack_description),
        rule_name=rule_name or _make_rule_name(attack_description),
    )

    data = repair_likely_mojibake(extract_json_object(llm.complete(prompt)))
    _strip_evidence_hints(data)
    return make_rule(
        name=str(data.get("name") or rule_name or _make_rule_name(attack_description)),
        description=str(data.get("description", attack_description)),
        conditions=data.get("conditions", []),
        exclusion_conditions=data.get("exclusion_conditions", []),
        decision_policy=str(data.get("decision_policy", "")),
        rule_id=str(data.get("rule_id") or rule_id or _make_rule_id(attack_description)),
        metadata={
            **dict(data.get("metadata", {})),
            "attack_description": attack_description,
            "raw_attack_label": data.get("metadata", {}).get("attack_label", raw_label),
            "attack_label": normalize_attack_label(
                data.get("metadata", {}).get("attack_label", label)
            ),
            "display_label": data.get("metadata", {}).get("display_label", raw_label),
            "human_example_provided": bool(human_example.strip()),
            "source": data.get("metadata", {}).get("source", "cold_start_llm"),
        },
    )


def _build_cold_start_prompt(
    attack_description: str,
    label: str,
    human_example: str,
    rule_id: str,
    rule_name: str,
) -> str:
    common_guidance = _common_rule_shape_guidance()
    label_guidance = _label_specific_guidance(label)
    source_guidance = _source_dependency_guidance(label)
    return f"""
You are initializing an EvoTx cold-start transaction-level evidence rule.

Task label:
{label}

Attack description:
{attack_description}

Your task:
Create a general evolving rule for this transaction-level attack mechanism.

Important:
- No real transaction trace is available during cold start.
- You are not selecting evidence views here.
- You are not generating a plan.
- You are not calling tools.
- Do not mention packet views, evidence groups, tool names, or runtime files.
- Do not output evidence_hints.
- Do not output concrete tx hashes, addresses, call IDs, transfer IDs, event IDs, or storage slots.
- Do not ask "is this a {label} attack?" as a condition.
- Each condition must be a local yes/no semantic condition.
- Each condition must have expected_answer.
- Conditions should not be duplicates.
- The rule should be broad enough to cover mechanism variants, but narrow enough to exclude common benign behaviors.
- Exclusions must be target-positive boundary clauses, not negative-class
  profiles. Write each exclusion from the target label's perspective: identify
  what required positive mechanism is absent, fully authorized, fully
  entitlement-backed, or completely replaced by another primary root cause.
- Do not create an exclusion merely because a non-target family or benign
  pattern is present. The exclusion should fire only when that pattern defeats
  or replaces the target-positive mechanism required by the core conditions.

{common_guidance}

{label_guidance}

{source_guidance}

Return strict JSON only:
{{
  "rule_id": "{rule_id}",
  "name": "{rule_name}",
  "description": "overall rule description",
  "conditions": [
    {{
      "id": "C1",
      "description": "...",
      "expected_answer": true
    }}
  ],
  "exclusion_conditions": [
    {{
      "id": "E1",
      "description": "...",
      "expected_answer": true
    }}
  ],
  "decision_policy": "how to combine conditions and exclusions",
  "metadata": {{
    "attack_label": "{label}",
    "attack_description": "{attack_description}",
    "source": "cold_start_llm"
  }}
}}

Example for label = "price_manipulation":

{{
  "rule_id": "price_manipulation_evidence_rule",
  "name": "Price Manipulation Transaction Evidence Rule",
  "description": "Detect transaction-level evidence that a price-relevant source was temporarily perturbed and then consumed by a value-sensitive protocol operation, producing attacker-favorable or protocol-adverse effects.",
  "conditions": [
    {{
      "id": "C1",
      "description": "The transaction shows temporary perturbation of a price-relevant source, such as an AMM reserve, pool balance, token balance, vault balance, oracle-read value, LP/share-price input, exchange-rate input, or protocol accounting value.",
      "expected_answer": true
    }},
    {{
      "id": "C2",
      "description": "A causally linked value-sensitive operation consumes or depends on the perturbed value. The consumer may be downstream or a same-market swap, settlement, or extraction whose amount is determined by an abnormal distorted reserve or price; ordinary sequential swaps merely repricing against normal reserve updates, bare skim/sync, or operation co-occurrence are insufficient.",
      "expected_answer": true
    }},
    {{
      "id": "C3",
      "description": "The transaction produces a material attacker-favorable or protocol-adverse outcome, such as attacker-side profit, protocol/victim asset loss, inflated minted or redeemed assets, undercollateralized borrow, abnormal reward, or adverse accounting state shift.",
      "expected_answer": true
    }}
  ],
  "exclusion_conditions": [
    {{
      "id": "E1",
      "description": "The trace shows only bare skim/sync, closed-loop DEX arbitrage, an ordinary large swap, or normal liquidation, and no same-market or downstream value-sensitive operation consumes a distorted price-related value.",
      "expected_answer": true
    }},
    {{
      "id": "E2",
      "description": "The trace shows authorized governance, admin, keeper, oracle maintenance, or legitimate capital-backed contribution, repayment, deposit, redeem, or collateral activity that fully supports the payout.",
      "expected_answer": true
    }}
  ],
  "decision_policy": "Emit only if C1, C2, and C3 are supported and neither E1 nor E2 is supported. If any core condition is uncertain or contradictory, keep the verdict uncertain.",
  "metadata": {{
    "attack_label": "price_manipulation",
    "source": "cold_start_llm"
  }}
}}

Human sketch example:
{human_example or "(none)"}
"""


def _common_rule_shape_guidance() -> str:
    return """Common rule-shape guidance:
- Prefer exactly 3 positive conditions and 1-2 exclusion conditions.
- Hard budget: no more than 3 positive conditions and 2 exclusions unless the target mechanism truly cannot be represented by local transaction evidence.
- Each condition should describe one local semantic fact and should be concise.
- Do not enumerate long example lists inside condition descriptions; use examples only to guide abstraction.
- Do not turn supporting route symptoms into standalone conditions. Flash loan, profit, swap, large transfer, callback, reentrancy-shaped trace, price movement, unknown selector, gas use, failed intermediate operation, suspicious timing, or call complexity should not become standalone conditions unless that symptom is the target-family root cause.
- Do not infer the root cause merely from profit, value extraction, bad outcome, complex call route, suspicious execution shape, stale-looking state, zero-balance read, state mutation, or failed intermediate operation.
- Merge supporting symptoms into the relevant root-cause, consumption, authorization, validation, accounting, or outcome condition.
- Describe semantic facts, not evidence channels. Do not require source code, packet views, evidence groups, tool names, runtime files, tx hashes, addresses, call IDs, transfer IDs, event IDs, or storage slots in condition text.
- If a semantic fact cannot be established from available transaction-level evidence, the condition should remain uncertain rather than being guessed."""


def _source_dependency_guidance(label: str) -> str:
    label = _make_attack_label(label)
    if label in {"access_control", "insufficient_validation"}:
        return """Source-dependency note:
- This label is often source/ABI-dependent, especially for the core root-cause condition.
- The judge may use source-level evidence to confirm the missing or bypassed check, but the generated rule must still describe the semantic fact, not the evidence channel.
- When source is available, use it to confirm authorization checks, validation checks, and business-invariant guards.
- When source is unavailable, require concrete transaction-level behavioral evidence for the same semantic fact; otherwise keep the condition uncertain.
- Do not write conditions such as "source code shows...", "packet view shows...", "the tool reports...", or "read_function_chunk indicates..."."""
    return ""


def _label_specific_guidance(label: str) -> str:
    label = _make_attack_label(label)
    if label == "price_manipulation":
        return """For label "price_manipulation":
The rule should usually require:
C1. A price-relevant source is temporarily perturbed.
    Examples: AMM reserve, pool balance, token balance, vault balance, oracle-read value, LP/share price input, exchange-rate input, protocol accounting value.
C2. A value-sensitive operation consumes or depends on the perturbed value.
    Examples: borrow, mint, redeem, withdraw, liquidate, reward calculation, collateral valuation, share issuance, LP pricing, exchange-rate update, or a same-market swap/settlement/extraction whose amount is determined by the distorted reserve or price.
C3. The transaction produces a material attacker-favorable or protocol-adverse outcome.
    Examples: attacker-side profit, victim/protocol asset loss, inflated minted/redeemed assets, undercollateralized borrow, abnormal reward, adverse accounting state shift.
Temporal or causal consistency should support C1-C3, not become a separate condition by default.

Important price-manipulation boundary:
- price_manipulation is not market_manipulation. A skim, donate, sync, sandwich/MEV position, ordering pattern, pool-accounting anomaly, flash loan, swap, or profit is not price-manipulation proof by name or co-occurrence.
- A token-accounting or operation-order defect may still establish C1 when it actually distorts a price-relevant reserve/value. C2 can then be satisfied by a same-market operation if transaction evidence shows that operation used the distorted reserve/price to determine execution or settlement; it need not be a distinct external protocol.
- The actor's ordinary sequential swaps merely observing normal AMM reserve updates do not establish C2. For a same-market path, require an independently abnormal distortion and evidence that it causally changed the later amount, limit, settlement, or extraction.
- Bare skim/sync/reserve writing without price-sensitive consumption, and ordinary swaps performed without exploiting the distortion, remain outside the positive boundary.
- Do not classify such a same-market price-consumption path as an unsupported alternate OR branch merely because an existing rule example emphasized borrow/mint/redeem. First consider a single causal semantic generalization that still requires distorted-price consumption and adverse outcome.
- Keep the condition roles local: C1 owns whether the price-relevant source is abnormally distorted; C2 owns whether a value-sensitive operation actually consumes that distortion. C2 may reference the selected distortion but should not restate the full C1 value-backed/unbacked definition.

The rule should exclude:
E1. Bare skim/sync, closed-loop DEX arbitrage, ordinary large swap, or normal liquidation where no same-market or downstream value-sensitive operation actually consumes a distorted value.
E2. Authorized governance/admin/keeper/oracle maintenance or legitimate capital-backed contribution, repayment, deposit, redeem, or collateral activity that fully supports the payout."""

    if label == "reentrancy":
        return """For label "reentrancy":
The rule should usually require:
C1. Select a concrete nested-entry candidate: outer call, external edge, re-entry call, logical storage/accounting domain, and trace path.
C2. For that same candidate, asset release, mint, withdraw, redeem, claim, share accounting, debt/collateral update, or another value-relevant effect occurs during or because of nested execution. C2 establishes the candidate-local effect only; stale-state, delayed-finalization, repeated-consumption, and phase-order causality belong to C3.
C3. For that same C1/C2 candidate, a phase-correct state/accounting-order witness shows critical state was not finalized before the external edge and the nested path consumed stale/intermediate state before the outer protective update, enabling repeated or excess effect.
Material attacker-favorable or protocol-adverse outcome should support C2/C3, not become a separate condition by default.

All positive conditions must preserve one candidate_id or equivalent nested path.
Do not stitch C1 from one callback, C2 from unrelated transaction profit, and C3
from generic storage overlap elsewhere. In EVM execution an SSTORE completed
before an external call is immediately visible to nested execution; a later
SLOAD does not make that completed write stale. Generic sload/sstore overlap,
repeated calls, callbacks, or output alone are insufficient without the
outer-read/external-edge/inner-consumption/outer-write ordering or an equivalent
read-only/accounting witness.

The rule should exclude:
E1. For the same selected candidate, protective state is finalized before the external edge and nested execution observes the finalized state.
E2. The same selected candidate is a normal protocol callback, router multicall, flashloan callback, DEX callback, token hook, or independently entitlement-backed iteration without repeated sensitive value release or stale-state exploitation."""

    if label == "access_control":
        return """For label "access_control":
The rule should capture a missing or bypassed authorization boundary, not generic value extraction.

Shared authorization-chain invariant:
- All positive conditions must describe one coherent chain: actor/caller or beneficiary -> required authority -> sensitive capability and protected target/resource -> protected effect.
- Do not satisfy different positive conditions with unrelated actors, calls, capabilities, assets, positions, or protocol components. Multiple calls may participate only when transaction evidence connects them as one invocation/delegation/callback chain crossing the same authorization boundary.
- The conditions remain local yes/no facts, but each one must name enough of the shared chain role to prevent cross-object evidence stitching.

The rule should usually require:
C1. Locate one concrete, transaction-visible, plausibly authority-bearing candidate: actor/caller or beneficiary, sensitive call or capability, protected target/resource, effect hint, and evidence ids. A single call may be the complete candidate. Preserve a separate entry parent and downstream operation only when they differ and trace evidence connects them; C1 is a locator and does not decide whether authority is missing or the operation is legitimately permissionless.
C1 may retain an unresolved selector or call when trace-linked configuration,
approval, role, initializer, protocol-controlled value, or protected-state
effects make it a plausible candidate. Ordinary ERC-20 transfer/transferFrom,
AMM liquidity mint/burn, entitlement-based withdraw/redeem, and generic value
movement are not candidates by name alone, but whether a concrete operation is
legitimately permissionless or entitlement-backed is resolved by C2/E1 rather
than used as a C1 terminal classification.
Generic swap/transfer routing, beneficiary gain, or value-flow asymmetry alone
does not make a call plausibly authority-bearing; there must be a concrete
protected capability/resource or authorization-relevant state/effect anchor.
C2. Establish from source or concrete behavioral evidence that the same actor/caller, callback sender, or beneficiary lacks or bypasses the required authority for that exact capability and target/resource. Record authority source/provenance when available, but do not require an exhaustive provenance taxonomy. A familiar modifier is decisive only for the same candidate capability and may be contradicted by observed self-authorization or bypass on the same trace path.
C3. Establish that the same unauthorized invocation causally produces a protected state/effect on the same target/resource: privilege/approval/configuration change, protected external-call execution, or movement of protocol-controlled/shared/victim assets beyond legitimate entitlement or contribution.

Important boundary:
- Public function names, callback shape, swap, flash loan, profit, valid-looking signature, or generic exploit shape do not prove or exclude access control by themselves.
- Unverified user-supplied target, calldata, token/path/amount, signature, return value, or business-state semantics are usually insufficient_validation unless the path exposes a privileged capability or protocol-controlled assets to an unauthorized caller.
- Source code may confirm the boundary, but source unavailability does not break an otherwise transaction-visible authorization chain. Conversely, generic behavioral evidence without a coherent actor/capability/target/effect chain is insufficient.

The rule should exclude:
E1. A recognized role, governance, timelock, multisig, keeper, admin, owner, valid delegation/signature, or user-owned position fully explains the same sensitive operation and effect.
E2. Another primary mechanism fully replaces the missing-authorization root cause, and there is no unauthorized protected capability, privileged configuration, abnormal callback authorization, or release of protocol/victim/shared assets beyond entitlement."""

    if label == "insufficient_validation":
        return """For label "insufficient_validation":
The rule should capture a causal validation gap, not generic value extraction.

The rule should usually require:
C1. A value-sensitive protocol path consumes a specific consumed data/state object that should be semantically validated before it controls sensitive logic. Preserve all eligible mechanism classes: user parameters, callback data, token/path/address, amount/allowance, signature/domain, external return, oracle input, computed intermediate value, and protocol-internal business/accounting state. Enumerate and rank candidates, then bind at most two or three diverse mechanism families for the first pass. Keep lower-ranked alternatives as explicit follow-up candidates instead of expanding every candidate at once. C1 binds candidates but does not decide validation adequacy. A standard same-token ERC-20 balance/allowance/supply query used by its ordinary transfer/mint/burn path, or generic swap parameters alone, is not sufficient to establish the target object.
C2. Audit the selected C1 candidates for a concrete missing or inadequate validation check on semantic content, source, freshness, range, or invariant, including a check that is absent, bypassed, stale, or wrong-scope. The returned C2 condition must preserve source/content identity or allowlisting, proportionality/business-invariant consistency, freshness/range, and wrong-scope validation as distinct possibilities even when phrased compactly. Examples include bounds/range, freshness, format, source/content binding, entitlement, balance sufficiency, proportionality, precision/rounding, return value, token/path/amount consistency, or business-invariant consistency. Generic caller/owner/role/whitelist/callback-sender identity authorization belongs to E1 unless the questioned defect is validation of supplied content, signer/domain semantics, or a consumed authority value. Caller ownership, nonReentrant, pause, balance, allowance, or contract-existence checks do not automatically validate those other dimensions. If selected candidates are guarded or standard design, request the next ranked candidate family rather than auditing every alternative in one prompt.
C2 should state the selected object's validation dimension and gap compactly. Do not embed the complete Access Control, Reentrancy, market, or accounting exclusion policy in C2; E1/E2 own those replacement-root-cause decisions.
C3. The same unvalidated object causally drives a concrete incorrect protocol outcome, disproportionate payout, asset release, invariant violation, state/accounting corruption, or protocol/user loss. Downstream swaps, profit, or value movement without a same-object causal link are insufficient.

Important boundary:
- Do not infer C2 merely from a bad outcome; validation-gap evidence and outcome evidence remain separate stages.
- Routine storage reads, token transfers, callbacks, and value flow are investigative context, not proof of a missing validation check.
- A complete trace with no visible validation call does not prove that a passing branch, modifier, or require check was absent. When source is unavailable, require positive behavioral evidence that a specific invalid, stale, inconsistent, adversarial, or boundary-breaking object was accepted and consumed.
- The bad object does not need to look syntactically malformed. Ordinary-looking calldata, valid token addresses, nonzero amounts, expected callbacks, or normal public functions can still be insufficient validation if semantic validation is missing.
- Missing caller identity, owner, role, whitelist, or callback-sender authorization is access control unless the root cause is validation of callback data, return values, token/path/amount content, signer/precompile semantics, or consumed business state.
- Market/oracle movement, reentrancy-shaped traces, token semantics, accounting-order bugs, or flash loans may appear in the route, but they should exclude insufficient_validation only when they fully replace the missing-validation root cause.

The rule should exclude:
E1. Caller/owner/role/whitelist/callback-sender authorization failure is the primary root cause on a protected path. E1 should independently compare the authorization path with all C1 candidates and may challenge an incorrect upstream object selection.
E2. An independent market/oracle/price manipulation, reentrancy/control-flow, token-semantic, accounting formula/order, direct storage corruption, or overflow/underflow mechanism fully explains the outcome without needing acceptance of the selected unvalidated object as a causal validation gap. The object may appear as a conduit; exclude when the independent mechanism is sufficient and the selected object's semantic invalidity is not necessary to the outcome."""

    return ""


def _strip_evidence_hints(data: Dict[str, Any]) -> None:
    """Remove evidence_hints from LLM output if accidentally included."""
    for cond in data.get("conditions", []):
        cond.pop("evidence_hints", None)
    for cond in data.get("exclusion_conditions", []):
        cond.pop("evidence_hints", None)


def _make_rule_name(attack_description: str) -> str:
    text = " ".join(str(attack_description).strip().split())
    if not text:
        return "General Transaction Attack Evidence Rule"
    return f"{text[:80]} Evidence Rule"


def _make_rule_id(text: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    clean = "_".join(part for part in clean.split("_") if part)
    clean = clean[:50] or "general_transaction_anomaly"
    return clean


def _make_attack_label(text: str) -> str:
    return normalize_attack_label(text)
