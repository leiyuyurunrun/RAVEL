from __future__ import annotations

from typing import Any, Dict, Iterable


NEGATIVE_TRAINING_MODES = {
    "auto",
    "benign_only",
    "attack_contrastive",
    "mixed",
}


def normalize_negative_training_mode(
    value: Any,
    *,
    allow_auto: bool = False,
) -> str:
    mode = str(value or ("auto" if allow_auto else "mixed")).strip().lower()
    mode = mode.replace("-", "_")
    allowed = NEGATIVE_TRAINING_MODES if allow_auto else NEGATIVE_TRAINING_MODES - {"auto"}
    if mode not in allowed:
        raise ValueError(
            f"Unsupported negative_training_mode={value!r}; "
            f"expected one of {sorted(allowed)}."
        )
    return mode


def summarize_negative_training_kinds(cases: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"benign": 0, "other_attack": 0, "unknown_other": 0}
    for case in cases or []:
        if str((case or {}).get("ground_truth") or "").strip().lower() == "attack":
            continue
        kind = _negative_kind(case or {})
        counts[kind] = counts.get(kind, 0) + 1
    counts["total_negative"] = sum(counts.values())
    return counts


def infer_negative_training_mode(cases: Iterable[Dict[str, Any]]) -> str:
    counts = summarize_negative_training_kinds(cases)
    benign = counts.get("benign", 0)
    other_attack = counts.get("other_attack", 0)
    unknown = counts.get("unknown_other", 0)

    if unknown or not (benign or other_attack):
        return "mixed"
    if benign and other_attack:
        return "mixed"
    if other_attack:
        return "attack_contrastive"
    return "benign_only"


def resolve_negative_training_mode(
    requested: Any,
    cases: Iterable[Dict[str, Any]],
) -> str:
    mode = normalize_negative_training_mode(requested, allow_auto=True)
    if mode == "auto":
        return infer_negative_training_mode(cases)
    return mode


def validate_negative_training_coverage(
    mode: Any,
    cases: Iterable[Dict[str, Any]],
) -> Dict[str, int]:
    normalized = normalize_negative_training_mode(mode)
    counts = summarize_negative_training_kinds(cases)
    if normalized == "attack_contrastive":
        invalid = counts.get("benign", 0) + counts.get("unknown_other", 0)
        if invalid:
            raise ValueError(
                "negative_training_mode='attack_contrastive' requires every "
                "negative case to have a known non-target attack Type/label; "
                f"found benign={counts.get('benign', 0)} and "
                f"unknown_other={counts.get('unknown_other', 0)}."
            )
    elif normalized == "benign_only":
        invalid = counts.get("other_attack", 0) + counts.get("unknown_other", 0)
        if invalid:
            raise ValueError(
                "negative_training_mode='benign_only' requires every negative "
                "case to be known benign; "
                f"found other_attack={counts.get('other_attack', 0)} and "
                f"unknown_other={counts.get('unknown_other', 0)}."
            )
    return counts


def negative_training_mode_guidance(mode: Any) -> str:
    normalized = normalize_negative_training_mode(mode)
    if normalized == "benign_only":
        return (
            "The training negatives are known benign or normal operations. Use them to "
            "identify legitimate authorization, entitlement, invariant-preserving, and "
            "normal-operation boundaries. Do not assume the observed negatives cover "
            "other attack families, and do not weaken the target-positive mechanism into "
            "generic anomaly detection."
        )
    if normalized == "attack_contrastive":
        return (
            "The training negatives are non-target attack transactions. Use them to "
            "identify which required target-positive mechanism is absent, authorized, "
            "entitlement-backed, or replaced by another primary cause. Do not encode, "
            "name, or profile the observed negative attack families; preserve boundaries "
            "against benign and unseen non-target cases."
        )
    return (
        "The training negatives contain a mixed or incompletely identified non-target "
        "boundary. Keep distinctions valid for both benign operations and other attack "
        "families, and do not overfit conditions or exclusions to one observed subgroup."
    )


def negative_case_boundary_guidance(mode: Any) -> str:
    normalized = normalize_negative_training_mode(mode)
    if normalized == "benign_only":
        return (
            "For benign negative cases, diagnose which legitimate authorization, "
            "entitlement, invariant-preserving transition, or normal protocol operation "
            "the detector incorrectly treated as the target mechanism. A missing CSV "
            "Cause is expected and is not missing transaction evidence. Do not invent an "
            "alternate attack cause or treat benign identity, protocol, function, or "
            "selector names as the boundary."
        )
    if normalized == "attack_contrastive":
        return (
            "For non-target attack cases, diagnose the target-vs-non-target mechanism "
            "boundary rather than describing the transaction as benign. State which "
            "required target-positive mechanism is absent, authorized, entitlement-backed, "
            "or replaced, without encoding the source attack-family label."
        )
    return (
        "Use each case's negative_kind: apply legitimate-operation boundary analysis to "
        "benign negatives and target-vs-non-target mechanism analysis to other_attack "
        "negatives. Unknown negatives must not be assigned an invented benign or attack "
        "cause."
    )


def _negative_kind(case: Dict[str, Any]) -> str:
    containers = [
        case,
        case.get("metadata") if isinstance(case.get("metadata"), dict) else {},
        case.get("case_metadata") if isinstance(case.get("case_metadata"), dict) else {},
    ]
    for container in containers:
        kind = str((container or {}).get("negative_kind") or "").strip().lower()
        if kind in {"benign", "other_attack", "unknown_other"}:
            return kind

    role = str(case.get("sample_role") or "").strip().lower()
    if not role:
        metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
        role = str(metadata.get("sample_role") or "").strip().lower()
    if role == "negative_benign":
        return "benign"
    return "unknown_other"
