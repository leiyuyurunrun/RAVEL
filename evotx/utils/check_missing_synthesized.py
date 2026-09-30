#!/usr/bin/env python3
"""Check whether CSV transaction hashes have required cached packet inputs.

The checker walks a CSV file or a directory recursively, reads transaction
hashes from common hash columns, and verifies that each hash has:

- data/cache/fund_flow/{hash}_fundflow.json
- data/cache/profit_loss/{hash}_top-profit-loss.json
- data/cache/synthesized/{hash}_synthesized.json
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


HASH_COLUMNS = {
    "hash",
    "txhash",
    "tx_hash",
    "transactionhash",
    "transaction_hash",
}

CACHE_SPECS: Dict[str, Tuple[str, str]] = {
    "fund_flow": ("fund_flow", "_fundflow.json"),
    "profit_loss": ("profit_loss", "_top-profit-loss.json"),
    "synthesized": ("synthesized", "_synthesized.json"),
}

HEX_HASH_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(value or "").strip().lower())


def _normalize_hash(value: str) -> str:
    return str(value or "").strip().lower()


def find_csv_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    return []


def _find_hash_column(fieldnames: Iterable[str] | None) -> str | None:
    for field in fieldnames or []:
        if _normalize_header(field) in HASH_COLUMNS:
            return field
    return None


def extract_hashes_from_csv(csv_path: Path) -> List[str]:
    """Extract unique tx hashes from one CSV while preserving file order."""
    hashes: List[str] = []
    seen = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        hash_column = _find_hash_column(reader.fieldnames)
        if not hash_column:
            raise ValueError(
                f"No tx hash column found. Expected one of: "
                f"{', '.join(sorted(HASH_COLUMNS))}"
            )

        for row in reader:
            value = _normalize_hash(row.get(hash_column, ""))
            if not value:
                continue
            if not HEX_HASH_RE.match(value):
                # Keep the check strict enough to avoid reading IDs or labels as
                # hashes, but allow short fake hashes such as 0x12232323 in tests.
                continue
            if value not in seen:
                hashes.append(value)
                seen.add(value)

    return hashes


def missing_cache_files(tx_hash: str, cache_root: Path) -> List[str]:
    missing = []
    for name, (subdir, suffix) in CACHE_SPECS.items():
        expected = cache_root / subdir / f"{tx_hash}{suffix}"
        if not expected.exists():
            missing.append(name)
    return missing


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively check CSV txHash rows against fund_flow, profit_loss, "
            "and synthesized cache files."
        )
    )
    parser.add_argument("path", help="CSV file or directory containing CSV files.")
    parser.add_argument(
        "--cache-root",
        default=str(_repo_root() / "data" / "cache"),
        help="Cache root containing fund_flow, profit_loss, and synthesized dirs.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.path)
    cache_root = Path(args.cache_root)
    csv_files = find_csv_files(input_path)

    if not csv_files:
        print(f"No CSV files found under: {input_path}")
        return 1

    missing_dirs = [
        str(cache_root / subdir)
        for subdir, _suffix in CACHE_SPECS.values()
        if not (cache_root / subdir).exists()
    ]
    if missing_dirs:
        print("Missing cache directories:")
        for item in missing_dirs:
            print(f"  {item}")
        return 1

    base_dir = input_path.resolve() if input_path.is_dir() else input_path.resolve().parent
    file_stats = {}
    all_missing: List[Tuple[str, str, List[str]]] = []

    print(f"Checking {len(csv_files)} CSV file(s)")
    print(f"Cache root: {cache_root.resolve()}")
    print()

    for csv_file in csv_files:
        try:
            hashes = extract_hashes_from_csv(csv_file)
        except Exception as exc:
            rel = _relative_for_display(csv_file, base_dir)
            file_stats[rel] = {"total": 0, "missing": [], "error": str(exc)}
            continue

        missing_rows = []
        for tx_hash in hashes:
            missing = missing_cache_files(tx_hash, cache_root)
            if missing:
                missing_rows.append((tx_hash, missing))
                all_missing.append((_relative_for_display(csv_file, base_dir), tx_hash, missing))

        file_stats[_relative_for_display(csv_file, base_dir)] = {
            "total": len(hashes),
            "missing": missing_rows,
            "error": "",
        }

    print("=" * 80)
    print("Per-file result")
    print("=" * 80)
    for rel, stats in file_stats.items():
        if stats["error"]:
            print(f"ERROR {rel}: {stats['error']}")
            continue
        missing_count = len(stats["missing"])
        status = "OK" if missing_count == 0 else "MISSING"
        print(f"{status:7} {rel}: {stats['total']} hash(es), {missing_count} missing")

    if all_missing:
        print()
        print("=" * 80)
        print(f"Missing cache files: {len(all_missing)} tx hash occurrence(s)")
        print("=" * 80)
        for rel, tx_hash, missing in all_missing:
            print(f"{rel} | {tx_hash} | missing: {', '.join(missing)}")
        return 1

    print()
    print("All tx hashes have fund_flow, profit_loss, and synthesized cache files.")
    return 0


def _relative_for_display(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
