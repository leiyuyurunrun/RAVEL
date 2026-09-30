from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evotx.core.labels import normalize_attack_label
from evotx.utils.json_utils import read_json


_TX_HASH_FIELDS = [
    "tx_hash",
    "txHash",
    "TxHash",
    "hash",
    "Hash",
    "transaction_hash",
    "TransactionHash",
]
_CHAIN_FIELDS = ["chain", "Chain", "network", "Network"]
_REPORT_FIELDS = ["report", "Report", "cause", "Cause", "description", "Description"]
_RATIONALE_FIELDS = [
    "label_rationale",
    "LabelRationale",
    "rationale",
    "Rationale",
    "type",
    "Type",
    "cause",
    "Cause",
    "description",
    "Description",
]
_GROUND_TRUTH_FIELDS = ["ground_truth", "GroundTruth", "label", "Label"]
_TYPE_LABEL_FIELDS = ["type", "Type"]


def load_labeled_cases(
    path_like: str,
    positive_label: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    raw_cases = _load_raw_cases(path_like)
    resolved_positive_label = _resolve_positive_label(raw_cases, positive_label)

    cases: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_cases, start=1):
        tx_hash = _extract_tx_hash(raw)
        if not tx_hash:
            raise ValueError(f"Case #{index} is missing tx_hash.")

        raw_ground_truth = _extract_case_label(raw, positive_label=resolved_positive_label)
        ground_truth = normalize_ground_truth(
            raw_ground_truth,
            positive_label=resolved_positive_label,
        )
        if (
            ground_truth not in {"attack", "benign"}
            and positive_label
            and raw_ground_truth
        ):
            ground_truth = "benign"
        if ground_truth not in {"attack", "benign"}:
            raise ValueError(
                f"Case #{index} has unsupported ground_truth '{raw_ground_truth}'. "
                f"Resolved positive label is '{resolved_positive_label}'."
            )
        negative_kind = (
            None
            if ground_truth == "attack"
            else _infer_negative_kind(
                raw_ground_truth,
                target_label=resolved_positive_label,
                sample_role="negative_other",
            )
        )
        sample_role = (
            "positive_target"
            if ground_truth == "attack"
            else "negative_benign" if negative_kind == "benign" else "negative_other"
        )

        cases.append(
            {
                "tx_hash": tx_hash,
                "chain": _extract_chain(raw),
                "ground_truth": ground_truth,
                "raw_ground_truth": raw_ground_truth,
                "tx_context": _coerce_object(_get_first(raw, ["tx_context", "txContext"])),
                "static_evidence": _coerce_object(_get_first(raw, ["static_evidence", "staticEvidence"])),
                "label_rationale": _get_first(raw, _RATIONALE_FIELDS),
                "report": _get_first(raw, _REPORT_FIELDS),
                "metadata": _case_metadata(
                    raw,
                    sample_role=sample_role,
                    negative_kind=negative_kind,
                    target_label=resolved_positive_label,
                ),
            }
        )
    return cases, resolved_positive_label


def load_cases_from_split_csvs(
    label: str,
    benign_csv: Optional[str] = None,
    malicious_csv: Optional[str] = None,
    pos_csv: Optional[str] = None,
    neg_csv: Optional[str] = None,
    head_per_csv: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    positive_csv = pos_csv or malicious_csv
    if not positive_csv:
        raise ValueError("Provide --pos-csv or legacy --malicious-csv.")

    raw_positive_label = label.strip()
    resolved_positive_label = normalize_attack_label(raw_positive_label)
    if not raw_positive_label:
        raise ValueError("--label must be non-empty when using split CSV input.")

    positive_rows = _head_rows(_load_csv_rows(positive_csv), head_per_csv)

    cases: List[Dict[str, Any]] = []
    for index, raw in enumerate(positive_rows, start=1):
        tx_hash = _extract_tx_hash(raw)
        if not tx_hash:
            raise ValueError(f"Positive case #{index} is missing hash/tx_hash.")
        cases.append(
            {
                "tx_hash": tx_hash,
                "chain": _extract_chain(raw),
                "ground_truth": "attack",
                "raw_ground_truth": resolved_positive_label,
                "tx_context": {},
                "static_evidence": {},
                "label_rationale": _get_first(raw, _RATIONALE_FIELDS),
                "report": _get_first(raw, _REPORT_FIELDS),
                "metadata": _case_metadata(
                    raw,
                    source_csv=positive_csv,
                    sample_role="positive_target",
                    target_label=resolved_positive_label,
                ),
            }
        )

    if benign_csv:
        cases.extend(
            _load_negative_csv_cases(
                csv_path=benign_csv,
                target_label=resolved_positive_label,
                sample_role="negative_benign",
                default_raw_ground_truth="benign",
                head=head_per_csv,
            )
        )
    if neg_csv:
        cases.extend(
            _load_negative_csv_cases(
                csv_path=neg_csv,
                target_label=resolved_positive_label,
                sample_role="negative_other",
                default_raw_ground_truth="non_target",
                head=head_per_csv,
            )
        )
    return cases, resolved_positive_label


def _load_negative_csv_cases(
    *,
    csv_path: str,
    target_label: str,
    sample_role: str,
    default_raw_ground_truth: str,
    head: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = _head_rows(_load_csv_rows(csv_path), head)
    cases: List[Dict[str, Any]] = []
    for index, raw in enumerate(rows, start=1):
        tx_hash = _extract_tx_hash(raw)
        if not tx_hash:
            raise ValueError(f"Negative case #{index} in {csv_path} is missing hash/tx_hash.")
        raw_ground_truth = _resolve_negative_raw_ground_truth(
            raw,
            sample_role=sample_role,
            default=default_raw_ground_truth,
        )
        negative_kind = _infer_negative_kind(
            raw_ground_truth,
            target_label=target_label,
            sample_role=sample_role,
        )
        if negative_kind == "target_label_in_negative_csv":
            raise ValueError(
                f"Negative case #{index} in {csv_path} has Type/label "
                f"{raw_ground_truth!r}, which contains target label "
                f"{target_label!r}."
            )
        cases.append(
            {
                "tx_hash": tx_hash,
                "chain": _extract_chain(raw),
                "ground_truth": "benign",
                "raw_ground_truth": raw_ground_truth,
                "tx_context": {},
                "static_evidence": {},
                "label_rationale": _get_first(raw, _RATIONALE_FIELDS),
                "report": _get_first(raw, _REPORT_FIELDS),
                "metadata": _case_metadata(
                    raw,
                    source_csv=csv_path,
                    sample_role=sample_role,
                    negative_kind=negative_kind,
                    target_label=target_label,
                ),
            }
        )
    return cases


def _resolve_negative_raw_ground_truth(
    raw: Dict[str, Any],
    *,
    sample_role: str,
    default: str,
) -> str:
    if sample_role == "negative_other":
        type_label = _get_first(raw, _TYPE_LABEL_FIELDS)
        if type_label:
            return type_label
    return _get_first(raw, _GROUND_TRUTH_FIELDS, default=default)


def _head_rows(rows: List[Dict[str, Any]], head: Optional[int]) -> List[Dict[str, Any]]:
    if head is None:
        return rows
    if head < 0:
        raise ValueError("head must be non-negative.")
    return rows[:head]


def normalize_ground_truth(
    raw_label: str,
    positive_label: Optional[str] = None,
) -> str:
    normalized = _normalize_label(raw_label)
    if normalized == "benign":
        return "benign"
    if normalized == "attack":
        return "attack"

    chosen_positive = _normalize_label(positive_label or "")
    if chosen_positive and _label_contains(raw_label, positive_label or ""):
        return "attack"
    return normalized


def _load_raw_cases(path_like: str) -> List[Dict[str, Any]]:
    path = Path(path_like)
    if path.suffix.lower() == ".csv":
        return _load_csv_rows(path)

    data = read_json(path, default=[])
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(
            "--cases-file must contain a JSON list, an object with a 'cases' list, or a CSV file."
        )
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every case entry must be a JSON object.")
    return [dict(item) for item in data]


def _resolve_positive_label(
    raw_cases: List[Dict[str, Any]],
    preferred_label: Optional[str],
) -> str:
    if preferred_label and preferred_label.strip():
        return normalize_attack_label(preferred_label)

    custom_labels = sorted(
        {
            _normalize_label(_extract_case_label(case, positive_label=preferred_label))
            for case in raw_cases
            if _normalize_label(_extract_case_label(case, positive_label=preferred_label))
            not in {"", "attack", "benign"}
        }
    )
    if not custom_labels:
        return "attack"
    if len(custom_labels) == 1:
        return custom_labels[0]
    raise ValueError(
        "Multiple positive labels were found in the cases file. "
        "Please specify one with --positive-label."
    )


def _coerce_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_label(label: str) -> str:
    return normalize_attack_label(label, default="")


def _extract_case_label(
    raw: Dict[str, Any],
    positive_label: Optional[str] = None,
) -> str:
    explicit_label = _get_first(raw, _GROUND_TRUTH_FIELDS)
    if explicit_label:
        return explicit_label

    type_label = _get_first(raw, _TYPE_LABEL_FIELDS)
    return type_label


def _label_contains(raw_label: str, positive_label: str) -> bool:
    normalized_raw = _normalize_label(raw_label)
    normalized_positive = _normalize_label(positive_label)
    if not normalized_positive:
        return False
    if normalized_raw == normalized_positive:
        return True
    if normalized_positive in normalized_raw:
        return True

    parts = [
        _normalize_label(part)
        for part in str(raw_label)
        .replace("|", ",")
        .replace(";", ",")
        .split(",")
    ]
    return normalized_positive in {part for part in parts if part}


def _infer_negative_kind(
    raw_ground_truth: str,
    *,
    target_label: str,
    sample_role: str,
) -> str:
    if sample_role == "negative_benign":
        return "benign"

    normalized = _normalize_label(raw_ground_truth)
    target = _normalize_label(target_label)
    if normalized in {"", "other", "non_target", "unknown", "negative"}:
        return "unknown_other"
    if normalized in {"benign", "normal", "safe", "legitimate"}:
        return "benign"
    if target and _label_contains(raw_ground_truth, target_label):
        return "target_label_in_negative_csv"
    return "other_attack"


def _extract_tx_hash(raw: Dict[str, Any]) -> str:
    return _get_first(raw, _TX_HASH_FIELDS)


def _extract_chain(raw: Dict[str, Any]) -> str:
    return (_get_first(raw, _CHAIN_FIELDS, default="eth") or "eth").lower()


def _get_first(raw: Dict[str, Any], names: List[str], default: str = "") -> str:
    for name in names:
        value = raw.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()

    lowered = {str(key).lower(): value for key, value in raw.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _case_metadata(
    raw: Dict[str, Any],
    source_csv: Optional[str] = None,
    sample_role: Optional[str] = None,
    negative_kind: Optional[str] = None,
    target_label: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = _coerce_object(_get_first(raw, ["metadata", "Metadata"]))
    for key in ["HackId", "hack_id", "id", "Id", "Type", "Cause", "Description"]:
        value = raw.get(key)
        if value is not None and str(value).strip():
            metadata[key] = str(value).strip()
    if source_csv:
        metadata["source_csv"] = str(source_csv)
    if sample_role:
        metadata["sample_role"] = sample_role
    if negative_kind:
        metadata["negative_kind"] = negative_kind
    if target_label:
        metadata["target_label"] = target_label
    return metadata


def _load_csv_rows(path_like: str | Path) -> List[Dict[str, Any]]:
    path = Path(path_like)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]
