"""Build the eight-class v3 CSV dataset layout from curated v2 sources.

For each target class, copy its few-shot positive/negative and evaluation
positive files unchanged. Build its evaluation negative file from:

1. the target class's ``evaluation_neg_redundant.csv``;
2. every other class's ``evaluation_pos.csv``.

Rows are aligned by header name, deduplicated by ``txHash``, and sanitized
against the target class's copied few-shot/evaluation files to prevent train or
positive-set leakage into evaluation negatives.

Run from the repository root:

    python csv_utils/build_v3_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


DEFAULT_SOURCE_ROOT = Path("data/csv/v2")
DEFAULT_OUTPUT_ROOT = Path("data/csv/v3")

FEWSHOT_POS_SUFFIX = "Fewshot_pos.csv"
FEWSHOT_NEG_SUFFIX = "Fewshot_neg.csv"
EVAL_POS_SUFFIX = "evaluation_pos.csv"
EVAL_NEG_REDUNDANT_SUFFIX = "evaluation_neg_redundant.csv"
EVAL_NEG_SUFFIX = "evaluation_neg.csv"

COPY_SUFFIXES = (
    FEWSHOT_POS_SUFFIX,
    FEWSHOT_NEG_SUFFIX,
    EVAL_POS_SUFFIX,
)
SANITIZE_SUFFIXES = (
    FEWSHOT_POS_SUFFIX,
    FEWSHOT_NEG_SUFFIX,
    EVAL_POS_SUFFIX,
)


@dataclass(frozen=True)
class LabelSpec:
    output_folder: str
    source_folder: str
    prefix: str

    def source_path(self, root: Path, suffix: str) -> Path:
        return root / self.source_folder / f"{self.prefix}-{suffix}"

    def output_path(self, root: Path, suffix: str) -> Path:
        return root / self.output_folder / f"{self.prefix}-{suffix}"


LABELS: Tuple[LabelSpec, ...] = (
    LabelSpec("Access control", "Access control-ori", "access_control"),
    LabelSpec("Flashloans", "Flashloans-ori", "flashloans"),
    LabelSpec(
        "Insufficient validation",
        "Insufficient validation",
        "insufficient_validation",
    ),
    LabelSpec("Market manipulation", "Market manipulation", "market_manipulation"),
    LabelSpec("Price manipulation", "Price manipulation", "price_manipulation"),
    LabelSpec(
        "Protocol accounting exploitation",
        "Protocol accounting exploitation",
        "protocol_accounting_exploitation",
    ),
    LabelSpec("Reentrancy", "Reentrancy", "reentrancy"),
    LabelSpec(
        "Token semantic exploitation",
        "Token semantic exploitation",
        "token_semantic_exploitation",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build eight v3 class datasets from the curated v2 CSV files."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Source root containing the eight v2 class folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination root for the generated v3 class folders.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        header = [str(name or "").strip() for name in reader.fieldnames]
        if "txHash" not in header:
            raise ValueError(f"CSV is missing txHash: {path}")
        rows = []
        for raw_row in reader:
            row = {
                str(key or "").strip(): str(value or "")
                for key, value in (raw_row or {}).items()
                if key is not None
            }
            if any(value.strip() for value in row.values()):
                rows.append(row)
        return header, rows


def collect_headers(headers: Iterable[Sequence[str]]) -> List[str]:
    combined: List[str] = []
    seen: Set[str] = set()
    for header in headers:
        for name in header:
            if name and name not in seen:
                seen.add(name)
                combined.append(name)
    return combined


def tx_hash(row: Dict[str, str]) -> str:
    return str(row.get("txHash") or "").strip().lower()


def collect_tx_hashes(paths: Iterable[Path]) -> Set[str]:
    hashes: Set[str] = set()
    for path in paths:
        _, rows = read_csv(path)
        hashes.update(key for row in rows if (key := tx_hash(row)))
    return hashes


def write_csv_atomic(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(header),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})
    temp_path.replace(path)


def validate_sources(source_root: Path) -> None:
    missing: List[Path] = []
    required_suffixes = (*COPY_SUFFIXES, EVAL_NEG_REDUNDANT_SUFFIX)
    for label in LABELS:
        for suffix in required_suffixes:
            path = label.source_path(source_root, suffix)
            if not path.is_file():
                missing.append(path)
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required v2 CSV files:\n{details}")


def prepare_output_folder(output_root: Path, label: LabelSpec) -> Path:
    folder = output_root / label.output_folder
    folder.mkdir(parents=True, exist_ok=True)
    expected = {
        label.output_path(output_root, suffix).name
        for suffix in (*COPY_SUFFIXES, EVAL_NEG_SUFFIX)
    }
    for existing in folder.glob("*.csv"):
        if existing.name not in expected:
            existing.unlink()
    return folder


def build_evaluation_negatives(
    target: LabelSpec,
    source_root: Path,
    output_root: Path,
) -> Dict[str, int]:
    source_paths = [target.source_path(source_root, EVAL_NEG_REDUNDANT_SUFFIX)]
    source_paths.extend(
        label.source_path(source_root, EVAL_POS_SUFFIX)
        for label in LABELS
        if label != target
    )

    headers: List[List[str]] = []
    source_rows: List[Tuple[str, List[Dict[str, str]]]] = []
    own_input = 0
    cross_input = 0
    for index, path in enumerate(source_paths):
        header, rows = read_csv(path)
        headers.append(header)
        source_rows.append((str(path), rows))
        if index == 0:
            own_input = len(rows)
        else:
            cross_input += len(rows)

    forbidden = collect_tx_hashes(
        target.source_path(source_root, suffix) for suffix in SANITIZE_SUFFIXES
    )
    seen: Set[str] = set()
    combined: List[Dict[str, str]] = []
    duplicates = 0
    leaked = 0
    blank_hashes = 0
    for _, rows in source_rows:
        for row in rows:
            key = tx_hash(row)
            if not key:
                blank_hashes += 1
                continue
            if key in forbidden:
                leaked += 1
                continue
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            combined.append(row)

    output_path = target.output_path(output_root, EVAL_NEG_SUFFIX)
    write_csv_atomic(output_path, collect_headers(headers), combined)
    return {
        "own_input": own_input,
        "cross_input": cross_input,
        "duplicates_removed": duplicates,
        "leakage_removed": leaked,
        "blank_hashes_removed": blank_hashes,
        "output": len(combined),
    }


def validate_output(output_root: Path) -> None:
    errors: List[str] = []
    for label in LABELS:
        folder = output_root / label.output_folder
        expected = {
            label.output_path(output_root, suffix).name
            for suffix in (*COPY_SUFFIXES, EVAL_NEG_SUFFIX)
        }
        actual = {path.name for path in folder.glob("*.csv")}
        if actual != expected:
            errors.append(
                f"{label.output_folder}: expected={sorted(expected)} actual={sorted(actual)}"
            )
            continue

        negative_path = label.output_path(output_root, EVAL_NEG_SUFFIX)
        _, negative_rows = read_csv(negative_path)
        negative_hashes = {tx_hash(row) for row in negative_rows if tx_hash(row)}
        forbidden_hashes = collect_tx_hashes(
            label.output_path(output_root, suffix) for suffix in SANITIZE_SUFFIXES
        )
        overlap = negative_hashes & forbidden_hashes
        if overlap:
            errors.append(
                f"{label.output_folder}: evaluation_neg overlaps copied sets "
                f"for {len(overlap)} txHash(es)"
            )
    if errors:
        raise ValueError("Invalid v3 output:\n" + "\n".join(f"  - {e}" for e in errors))


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError("source-root and output-root must be different")

    validate_sources(source_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for label in LABELS:
        prepare_output_folder(output_root, label)
        for suffix in COPY_SUFFIXES:
            shutil.copy2(
                label.source_path(source_root, suffix),
                label.output_path(output_root, suffix),
            )
        stats = build_evaluation_negatives(label, source_root, output_root)
        print(
            f"[ok] {label.output_folder}: "
            f"own={stats['own_input']} cross={stats['cross_input']} "
            f"dedup={stats['duplicates_removed']} "
            f"leakage={stats['leakage_removed']} "
            f"blank_hash={stats['blank_hashes_removed']} "
            f"evaluation_neg={stats['output']}"
        )

    validate_output(output_root)
    print(f"[done] generated {len(LABELS)} classes under {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
