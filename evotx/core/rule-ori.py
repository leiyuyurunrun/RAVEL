from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import uuid

from evotx.core.labels import normalize_attack_label
from evotx.core.schemas import EvolvingRule, RuleCondition
from evotx.utils.json_utils import extract_json_object


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
            "A downstream value-sensitive protocol operation consumes or depends on the perturbed value, such as borrow, mint, redeem, withdraw, liquidate, reward calculation, collateral valuation, share issuance, LP pricing, settlement, or exchange-rate update.",
            "The transaction produces a material attacker-favorable or protocol-adverse outcome, such as asset extraction, inflated minted or redeemed assets, undercollateralized borrow, abnormal reward, victim/protocol asset loss, or adverse accounting shift.",
        ],
        exclusions=[
            "The trace shows only closed-loop DEX arbitrage, ordinary large swap, normal liquidation, or routine AMM behavior, and no protocol-sensitive operation consumes a distorted price-related value.",
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
            "The transaction reaches a sensitive-by-effect operation, such as authorization or permission setting, callback handling that should verify sender, initialization/configuration, protocol-owned external-call capability, or movement of protocol-controlled/shared/victim assets.",
            "The caller, actor, callback sender, beneficiary, or execution path lacks legitimate authorization for that sensitive capability, such as missing owner/role/signature/caller checks, self-granted authorization, unprotected initializer, unauthorized callback, or public access to protocol-owned asset movement.",
            "The unauthorized invocation grants or changes privilege/approval/configuration, executes a protected external call, or moves protocol-controlled, shared, victim, treasury, deployer, pool, or privileged assets beyond caller entitlement or contribution.",
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
            "Detect transaction-visible misuse or distortion of market-facing "
            "state or execution context, followed by protocol-sensitive consumption "
            "and value movement."
        ),
        conditions=[
            "Transaction-visible market state is distorted, pre-seeded, donated, synchronized, skimmed, migrated, sandwiched, or otherwise misused through AMM, pool, oracle, settlement-facing, or market-facing behavior.",
            "A downstream operation consumes the distorted market-facing state for swap, mint, burn, redeem, borrow, liquidation, settlement, valuation, or accounting.",
            "The consumed market state leads to attacker-favorable or protocol-adverse value movement or accounting effect.",
        ],
        exclusions=[
            "Observable evidence shows only ordinary arbitrage, normal large swap, normal liquidation, or routine pool operation with no protocol-sensitive consumption of distorted market state.",
            "Observable evidence shows the primary root cause is token semantic mismatch, access-control failure, pure validation failure, reentrancy, or internal accounting formula abuse.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported and neither E1 nor E2 is "
            "supported."
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
            "Protocol bookkeeping state such as shares, rewards, debt, collateral, vault balance, eligibility, claim counter, exchange rate, or accounting index is updated, consumed, reused, or ordered incorrectly.",
            "The transaction obtains disproportionate payout, repeated claim, inflated shares, undercollateralized borrow, incorrect withdrawal, or reward/yield beyond contribution.",
            "The effect is caused by the protocol's own accounting logic, state, order, or formula rather than only external price movement or final profit.",
        ],
        exclusions=[
            "Observable evidence shows payout is fully explained by legitimate contribution, collateral, repayment, or normal accounting.",
            "Observable evidence shows the primary mechanism is market or oracle manipulation, token semantic mismatch, access control, insufficient validation, or classic reentrancy.",
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
            "materially enables a state, accounting, price, solvency, callback, "
            "or protocol-sensitive effect followed by value extraction or loss."
        ),
        conditions=[
            "The transaction contains temporary atomic capital such as flash loan, flash swap, flash mint, or same-transaction borrow-repay pattern.",
            "The temporary capital materially enables a state, accounting, price, solvency, callback, or protocol-sensitive effect that would be impossible or much less material without atomic capital.",
            "The transaction extracts value, causes protocol/user loss, or leaves attacker-favorable state after repayment or unwind.",
        ],
        exclusions=[
            "Observable evidence shows the flashloan is incidental capital for ordinary arbitrage or liquidation and not the enabling mechanism.",
            "Observable evidence shows the primary root cause is access control, insufficient validation, reentrancy, token semantic mismatch, market manipulation, or protocol accounting exploitation independent of temporary capital.",
        ],
        decision_policy=(
            "Emit only if C1, C2, and C3 are supported and neither E1 nor E2 is "
            "supported."
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

    data = extract_json_object(llm.complete(prompt))
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
    label_guidance = _label_specific_guidance(label)
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
- Prefer exactly 3 positive conditions and 1-2 exclusion conditions.
- Hard budget: no more than 3 positive conditions and 2 exclusions unless the target mechanism truly cannot be represented by local transaction evidence.
- Do not turn supporting evidence into separate conditions. Flash loan, profit, swap, large transfer, callback, unknown selector, gas use, or call complexity should not become standalone conditions unless that symptom is the target-family root cause.
- Merge supporting symptoms into the relevant root-cause, consumption, authorization, validation, or outcome condition.
- Conditions should not be duplicates.
- The rule should be broad enough to cover mechanism variants, but narrow enough to exclude common benign behaviors.
- Exclusions must be target-positive boundary clauses, not negative-class
  profiles. Write each exclusion from the target label's perspective: identify
  what required positive mechanism is absent, fully authorized, fully
  entitlement-backed, or completely replaced by another primary root cause.
- Do not create an exclusion merely because a non-target family or benign
  pattern is present. The exclusion should fire only when that pattern defeats
  or replaces the target-positive mechanism required by the core conditions.

{label_guidance}

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
      "description": "A value-sensitive protocol operation consumes or depends on the perturbed value, such as borrow, mint, redeem, withdraw, liquidate, reward calculation, collateral valuation, share issuance, LP pricing, or exchange-rate update.",
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
      "description": "The trace shows only closed-loop DEX arbitrage, ordinary large swap, or normal liquidation, and no protocol-sensitive operation consumes a distorted price-related value.",
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


def _label_specific_guidance(label: str) -> str:
    label = _make_attack_label(label)
    if label == "price_manipulation":
        return """For label "price_manipulation":
The rule should usually require:
C1. A price-relevant source is temporarily perturbed.
    Examples: AMM reserve, pool balance, token balance, vault balance, oracle-read value, LP/share price input, exchange-rate input, protocol accounting value.
C2. A value-sensitive operation consumes or depends on the perturbed value.
    Examples: borrow, mint, redeem, withdraw, liquidate, reward calculation, collateral valuation, share issuance, LP pricing, exchange-rate update.
C3. The transaction produces a material attacker-favorable or protocol-adverse outcome.
    Examples: attacker-side profit, victim/protocol asset loss, inflated minted/redeemed assets, undercollateralized borrow, abnormal reward, adverse accounting state shift.
Temporal or causal consistency should support C1-C3, not become a separate condition by default.

The rule should exclude:
E1. Closed-loop DEX arbitrage, ordinary large swap, or normal liquidation where no protocol-sensitive operation consumes a distorted value.
E2. Authorized governance/admin/keeper/oracle maintenance or legitimate capital-backed contribution, repayment, deposit, redeem, or collateral activity that fully supports the payout."""

    if label == "reentrancy":
        return """For label "reentrancy":
The rule should usually require:
C1. Nested callback or repeated entry into the same or logically related sensitive function before the outer execution completes.
C2. Asset release, mint, withdraw, redeem, claim, share accounting, debt/collateral update, or other value-relevant effect occurs during or because of nested execution.
C3. State/accounting order, stale state, or repeated state transition supports that nested execution changes value.
Material attacker-favorable or protocol-adverse outcome should support C2/C3, not become a separate condition by default.

The rule should exclude:
E1. Normal protocol callback, router multicall, flashloan callback, DEX callback, or token hook without repeated sensitive value release or stale-state exploitation.
E2. Normal single claim/withdraw/redeem/borrow/mint/governance/admin/migration path with expected entitlement and no repeated sensitive path."""

    if label == "access_control":
        return """For label "access_control":
The rule must stay compact: exactly C1/C2/C3 when possible, plus one or two exclusions. Do not add C4/C5 to cover subtypes. Merge subtypes into the three semantic roles below.

The rule should usually require:
C1. A sensitive-by-effect operation is reached. This includes admin/role or permission-modifying flows, authorization-setting flows, callback handlers expected to validate caller/sender, unprotected initialization/configuration of protocol-critical state, and public functions that move protocol-controlled, shared, victim, or privileged assets beyond caller-owned entitlement.
C2. The actor, caller, callback sender, beneficiary, or execution path lacks legitimate entitlement or exploits a missing/bypassed authorization boundary. This includes missing owner/role/signature/caller validation, self-authorization or impersonation, unprotected initializer paths, callback handlers invoked by non-authorized or non-standard paths, and behavioral packet evidence when source is unavailable.
C3. The invocation causes protected state mutation, privilege/approval/authorization effect, critical configuration change, or release/movement of protocol-controlled, shared, victim, or privileged assets beyond legitimate contribution, role, or entitlement.

The rule should exclude:
E1. Recognized authorization or entitlement only when it fully explains the same sensitive operation and protected effect required by C1-C3: governance, timelock, multisig, keeper, admin, protocol owner, valid delegation/signature, or user-owned position is current, standard, and sufficient for the action.
E2. Normal public-function interaction or another primary mechanism only when it replaces the access-control root cause required by C1-C3: the caller acts on its own position/assets, value movement is proportional to legitimate contribution/collateral/entitlement, and there is no unrecognized critical initialization/configuration, self-granted authorization, abnormal callback path, privileged capability exposure, or release of protocol/victim/shared assets beyond entitlement.

Exclusion design for access_control:
- Start from the positive access-control mechanism. Ask what would make the
  apparent unauthorized protected operation actually authorized, entitlement-
  backed, or not access-control-rooted.
- Do not start from a negative family name such as insufficient_validation,
  reentrancy, price manipulation, token semantics, or normal public call. Those
  labels support an exclusion only when they fully explain the root cause and
  the positive protected authorization-boundary mechanism is absent.
- Public function, callback, swap, flash loan, profit, valid-looking signature,
  or generic exploit shape is not an exclusion by itself.

Unverified user-supplied target, calldata, token/path/amount, signature, return
value, or business-state semantics are usually insufficient_validation, not
access_control, unless the path exposes a privileged capability or
protocol-controlled assets to an unauthorized caller.

Do not mark behavior benign merely because the function is public, named deposit/withdraw/swap/borrow/callback, or has a valid-looking signature. Profit/loss evidence is only an outcome hint and cannot alone prove or exclude access control."""

    if label == "insufficient_validation":
        return """For label "insufficient_validation":
Cold-start shape:
- Use exactly three compact positive conditions when possible.
- C1 goal: locate the specific consumed data/state object that should be
  validated before it controls sensitive logic. Do not treat arbitrary calldata
  or ordinary state reads as sufficient by themselves.
- C2 goal: identify a concrete missing or inadequate validation check on that
  same object. The check may be bounds/range, freshness, format, source,
  entitlement, balance sufficiency, proportionality, precision/rounding, return
  value, token/path/amount consistency, or business-invariant consistency.
- C3 goal: show the same unvalidated object from C2 causally drives an
  incorrect outcome, value release, loss, invariant failure, or state/accounting
  corruption.
- Keep each condition concise, ideally one sentence. Do not enumerate every
  example inside condition descriptions; keep examples in this guidance.
- Do not create separate conditions for flashloan, profit, callback,
  reentrancy, price movement, unknown selector, or gas/call complexity unless
  that item is the target-family root cause.
- Source code can be helpful, especially for C2, but the rule must support
  packet-based behavioral fallback when verified source is unavailable.

The rule should define the target family as acceptance of invalid, adversarial,
or semantically unsafe transaction inputs, callback data, external return
values, token/path/market addresses, amounts, signatures, oracle/external
state, or business-state assumptions by a value-sensitive protocol path.

Important semantic scope:
- Do not require the bad input to look syntactically malformed. Many true
  cases use ordinary-looking calldata, valid token addresses, nonzero amounts,
  or expected callbacks, but bypass semantic validation of entitlement,
  balance sufficiency, range/boundary constraints, precision/rounding guards,
  callback initiator/data, return values, price/oracle freshness, or business
  invariants.
- Do not turn the rule into generic value extraction. The positive rule should
  still require a causal validation gap: a protocol path trusted data/state that
  a correct implementation would reject, clamp, verify, or handle differently.
- Do not infer C2 merely from a bad outcome, value extraction, zero-balance
  read, stale-looking state, state mutation, complex call route, or failed
  intermediate operation. C2 needs a concrete validation subject and a concrete
  missing or inadequate check.
- Distinguish callback/caller boundaries carefully: missing caller identity,
  owner, role, whitelist, or callback-sender authorization is access control;
  missing validation of callback userData, return values, token/path/amount
  content, signer/precompile semantics, or consumed business state can be
  insufficient_validation.
- Flash loans, callbacks, reentrancy-shaped traces, price/oracle movement, or
  market interactions may be part of the route. Exclude them only when the
  primary root cause is independent of a missing validation gap. Do not exclude
  them when the exploit succeeds because the protocol failed to validate
  callback data, consumed external state, price/oracle value, state consistency,
  signer/precompile behavior, token/path/amount consistency, or business
  invariants.
- Routine storage reads, token transfers, reward accounting, or expected public
  functions do not exclude insufficient_validation when the consumed state or
  input violates obvious business invariants such as impossible reward
  proportionality, unsupported duration/enum values, fake signer semantics, or
  unapproved target/token/path data.

The rule should usually cover these semantic roles, without depending on the
specific condition IDs:
1. A value-sensitive protocol path consumes a specific user-supplied or
   externally sourced data/state object that should be semantically validated.
2. A concrete validation check on that same object is absent, bypassed, or
   inadequate, allowing invalid/adversarial/stale/inconsistent data/state to
   reach sensitive logic.
3. The same unvalidated object causes the incorrect protocol outcome, invariant
   violation, excessive payout, asset release, or protocol/user loss.

The rule should exclude:
E1. Access-control or authorization-boundary failures where the primary root
    cause is caller/owner/role/whitelist/callback-sender permission on a
    protected path, even if secondary data/state/value effects appear.
E2. Independent market/oracle/price manipulation, reentrancy/control-flow,
    token-semantic, accounting formula/order, direct storage corruption, or
    overflow/underflow mechanisms when acceptance of the specific unvalidated
    object from C2 is not the causal root."""

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
