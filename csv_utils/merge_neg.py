"""Build per-label ``evaluation_shared4neg.csv`` from two sources.

For every label directory under ``data/csv/v2`` we merge:

  - ``data/csv/v2/{label}/evaluation_neg_redundant.csv``         (own)
  - ``data/csv/v2/{other}/Fewshot_pos.csv`` for every other label (cross)

and write the result to ``data/csv/v2/{label}/evaluation_shared4neg.csv``,
overwriting any existing file. Output is unquoted to match the format of
the original ``evaluation_neg_redundant.csv`` files.

Run:

    python csv_utils/merge_neg.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


# === Configuration =========================================================
CSV_ROOT = Path(r"data\csv\v2")
NEG_REDUNDANT_NAME = "evaluation_neg_redundant.csv"
FEWSHOT_POS_NAME = "Fewshot_pos.csv"
FEWSHOT_NEG_NAME = "Fewshot_neg.csv"
EVAL_POS_NAME = "evaluation_pos.csv"
OUTPUT_NAME = "evaluation_shared4neg.csv"
# Files that supply txHashes to be excluded from the merged neg output
# (sanitization runs after the merge for every label).
SANITIZE_AGAINST = [EVAL_POS_NAME, FEWSHOT_POS_NAME, FEWSHOT_NEG_NAME]
# ===========================================================================


def _is_label_dir(path: Path) -> bool:
    """A label directory contains at least one CSV."""
    return path.is_dir() and any(child.suffix.lower() == ".csv" for child in path.iterdir())


def _discover_labels(root: Path) -> List[Path]:
    """Return the subdirectories of ``root`` that look like label folders."""
    return sorted(p for p in root.iterdir() if _is_label_dir(p))


def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV with the standard library, tolerating a UTF-8 BOM."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = [row for row in reader]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _build_hash_index(
    header: List[str],
    rows: List[List[str]],
) -> Tuple[int, Dict[str, List[List[str]]]]:
    """Index data rows by txHash (lowercased). Returns (hash_col, index)."""
    hash_idx = header.index("txHash")
    index: Dict[str, List[List[str]]] = {}
    for row in rows:
        if len(row) <= hash_idx:
            continue
        if not any(cell.strip() for cell in row):
            continue
        key = (row[hash_idx] or "").strip().lower()
        if not key:
            continue
        index.setdefault(key, []).append(row)
    return hash_idx, index


def _unify_row_widths(
    header: List[str],
    rows: List[List[str]],
) -> List[List[str]]:
    target_len = len(header)
    out: List[List[str]] = []
    for row in rows:
        if len(row) < target_len:
            row = row + [""] * (target_len - len(row))
        elif len(row) > target_len:
            row = row[:target_len]
        out.append(row)
    return out


def _merge_for_label(
    label_dir: Path,
    other_labels: List[Path],
) -> Dict[str, int]:
    """Build the shared4neg CSV for a single label directory."""
    neg_path = label_dir / NEG_REDUNDANT_NAME
    if not neg_path.exists():
        return {"own_in": 0, "cross_in": 0, "cross_added": 0, "out": 0}

    own_header, own_data = _read_csv(neg_path)
    own_header_width = len(own_header)
    own_data = _unify_row_widths(own_header, own_data)
    _, own_index = _build_hash_index(own_header, own_data)

    combined: List[List[str]] = list(own_data)
    seen: Dict[str, int] = {h: 0 for h in own_index}
    cross_added = 0
    cross_input = 0

    # Each other label contributes its Fewshot_pos rows.
    for other in other_labels:
        pos_path = other / FEWSHOT_POS_NAME
        if not pos_path.exists():
            continue
        pos_header, pos_data = _read_csv(pos_path)
        pos_data = _unify_row_widths(own_header, pos_data)
        _, pos_index = _build_hash_index(own_header, pos_data)
        cross_input += sum(len(v) for v in pos_index.values())
        for key, rows in pos_index.items():
            if key in seen:
                continue
            seen[key] = 0
            combined.extend(rows)
            cross_added += len(rows)

    out_path = label_dir / OUTPUT_NAME
    _write_csv(out_path, own_header, combined)
    return {
        "own_in": len(own_data),
        "cross_in": cross_input,
        "cross_added": cross_added,
        "out": len(combined),
    }


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    """Write the merged CSV with no quoting (matches the source format)."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _collect_txhashes(path: Path) -> Set[str]:
    """Return the lowercased txHash set from a CSV, or empty if missing."""
    if not path.exists():
        return set()
    header, data = _read_csv(path)
    if "txHash" not in header:
        return set()
    hash_idx = header.index("txHash")
    out: Set[str] = set()
    for row in data:
        if len(row) <= hash_idx:
            continue
        if not any(cell.strip() for cell in row):
            continue
        key = (row[hash_idx] or "").strip().lower()
        if key:
            out.add(key)
    return out


def sanitize_output(
    label_dir: Path,
    source_names: List[str],
) -> Dict[str, int]:
    """Drop rows from the freshly merged ``evaluation_shared4neg.csv``
    whose ``txHash`` appears in any of ``source_names`` in the same label.

    Returns stats: ``before``, ``removed``, ``after``, ``sources_used``,
    and ``source_sizes``.
    """
    out_path = label_dir / OUTPUT_NAME
    if not out_path.exists():
        return {"before": 0, "removed": 0, "after": 0, "sources_used": 0}

    header, data = _read_csv(out_path)
    if not header or "txHash" not in header:
        return {"before": len(data), "removed": 0, "after": len(data), "sources_used": 0}
    hash_idx = header.index("txHash")
    data = _unify_row_widths(header, data)

    # Build the forbidden-hash set across every requested source file.
    forbidden: Set[str] = set()
    used = 0
    for name in source_names:
        hashes = _collect_txhashes(label_dir / name)
        if hashes:
            used += 1
        forbidden |= hashes

    if not forbidden:
        return {
            "before": len(data),
            "removed": 0,
            "after": len(data),
            "sources_used": used,
        }

    kept: List[List[str]] = []
    removed = 0
    for row in data:
        key = (row[hash_idx] or "").strip().lower()
        if key in forbidden:
            removed += 1
            continue
        kept.append(row)

    _write_csv(out_path, header, kept)
    return {
        "before": len(data),
        "removed": removed,
        "after": len(kept),
        "sources_used": used,
    }


def main() -> int:
    if not CSV_ROOT.exists():
        print(f"[fail] {CSV_ROOT}: csv root not found", file=sys.stderr)
        return 1

    labels = _discover_labels(CSV_ROOT)
    if not labels:
        print(f"[fail] {CSV_ROOT}: no label subdirectories found", file=sys.stderr)
        return 1

    print(f"[info] discovered {len(labels)} label folder(s):")
    for label in labels:
        print(f"  - {label.name}")

    for label in labels:
        others = [l for l in labels if l != label]
        try:
            stats = _merge_for_label(label, others)
        except Exception as exc:
            print(f"[fail] {label.name}: {exc}", file=sys.stderr)
            continue
        out_path = label / OUTPUT_NAME
        print(
            f"[ok]   {out_path}: own={stats['own_in']} "
            f"cross_in={stats['cross_in']} cross_added={stats['cross_added']} "
            f"out={stats['out']}"
        )

        # Post-merge sanitization: drop rows whose txHash collides with
        # the same label's pos / fewshot sources. These are positive
        # examples that must never appear in the neg set.
        try:
            san = sanitize_output(label, SANITIZE_AGAINST)
        except Exception as exc:
            print(f"[fail] {label.name} (sanitize): {exc}", file=sys.stderr)
            continue
        print(
            f"[clean] {out_path}: before={san['before']} "
            f"removed={san['removed']} after={san['after']} "
            f"sources_used={san['sources_used']}/{len(SANITIZE_AGAINST)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
