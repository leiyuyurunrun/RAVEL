"""Normalize evaluation CSV files: deduplicate by txHash and unify quoting.

The script reads a CSV file whose rows are either fully quoted (standard
form) or unquoted (legacy form), normalizes every data row to the quoted
form, and writes the result back. Rows that share the same ``txHash`` are
collapsed, with the first occurrence preserved.

Run as a script:

    python -m evotx.utils.format_consistency PATH [PATH ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


HASH_COLUMN = "txHash"

# python -m csv_utils.format_consistency "data\csv\v2\Insufficient validation\evaluation_negNew.csv"


def _parse_mixed_csv(path: Path) -> Tuple[List[str], List[List[str]], int]:
    """Read a CSV with mixed quoted/unquoted rows.

    The header is expected to be a single line, but data rows may either
    use proper CSV quoting or omit quotes entirely. ``csv.reader`` handles
    both shapes when the dialect is permissive enough, so we fall back to a
    manual split when the standard reader fails (e.g. lines with embedded
    newlines inside unquoted cells).
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return [], [], 0

    # Try the standard reader first.
    lines = raw.splitlines(keepends=False)
    try:
        reader = csv.reader(lines)
        rows = [row for row in reader]
        header = rows[0]
        data = rows[1:]
        return header, data, 0
    except csv.Error:
        pass

    # Fallback: re-parse using a permissive state machine that tolerates
    # mixed quoting and embedded newlines inside unquoted fields.
    return _parse_permissive(raw)


def _parse_permissive(raw: str) -> Tuple[List[str], List[List[str]], int]:
    """Tokenize a CSV that mixes quoted and unquoted rows.

    Rules:
    - Quoted fields start with ``"`` and end at the next unescaped ``"``.
    - ``""`` inside a quoted field is an escaped quote.
    - Unquoted fields extend to the next comma or newline.
    - Embedded newlines inside a quoted field are preserved.
    """
    rows: List[List[str]] = []
    fields: List[str] = []
    buf: List[str] = []
    in_quotes = False
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]
        if in_quotes:
            if ch == '"':
                # Lookahead for an escaped quote.
                if i + 1 < n and raw[i + 1] == '"':
                    buf.append('"')
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue

        if ch == '"':
            in_quotes = True
            i += 1
            continue
        if ch == ',':
            fields.append("".join(buf))
            buf = []
            i += 1
            continue
        if ch == '\n':
            fields.append("".join(buf))
            rows.append(fields)
            fields = []
            buf = []
            i += 1
            continue
        if ch == '\r':
            i += 1
            continue
        buf.append(ch)
        i += 1

    # Flush trailing field/row if the file did not end with a newline.
    if buf or fields:
        fields.append("".join(buf))
        rows.append(fields)

    if not rows:
        return [], [], 0

    header = rows[0]
    data = rows[1:]
    return header, data, 0


def _hash_column_index(header: List[str]) -> int:
    """Locate the ``txHash`` column, falling back to a case-insensitive search."""
    if HASH_COLUMN in header:
        return header.index(HASH_COLUMN)
    for idx, name in enumerate(header):
        if name.strip().lower() == HASH_COLUMN.lower():
            return idx
    raise ValueError(f"Column '{HASH_COLUMN}' not found in header: {header}")


def _normalize_row(row: List[str], target_len: int) -> List[str]:
    """Pad or trim a row so that it matches the header length."""
    if len(row) < target_len:
        row = row + [""] * (target_len - len(row))
    elif len(row) > target_len:
        row = row[:target_len]
    return row


def normalize_csv(path: Path) -> Dict[str, int]:
    """Deduplicate and re-quote a single CSV file in-place.

    Returns a stats dict with the row counts before/after and the number
    of duplicates that were removed.
    """
    header, data, _ = _parse_mixed_csv(path)
    if not header:
        return {"rows_in": 0, "rows_out": 0, "duplicates": 0}

    hash_idx = _hash_column_index(header)
    target_len = len(header)

    seen: Dict[str, int] = {}
    unique_rows: List[List[str]] = []
    for raw_row in data:
        row = _normalize_row(list(raw_row), target_len)
        if not any(cell.strip() for cell in row):
            # Skip blank lines.
            continue
        tx_hash = (row[hash_idx] or "").strip().lower()
        if not tx_hash:
            # Keep rows that lack a hash so we don't silently drop them.
            unique_rows.append(row)
            continue
        if tx_hash in seen:
            seen[tx_hash] += 1
            continue
        seen[tx_hash] = 0
        unique_rows.append(row)

    duplicates_removed = sum(seen.values())

    _write_csv(path, header, unique_rows)

    return {
        "rows_in": len(data),
        "rows_out": len(unique_rows),
        "duplicates": duplicates_removed,
    }


def _write_csv(path: Path, header: List[str], rows: Iterable[List[str]]) -> None:
    """Write the header and rows with ``QUOTE_NONNUMERIC`` quoting."""
    # Use ``lineterminator='\n'`` to keep the file POSIX-friendly on
    # Windows hosts; the project runs on Windows but the CSV is consumed
    # by tools that prefer LF.
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(
            fh,
            quoting=csv.QUOTE_NONNUMERIC,
            lineterminator="\n",
        )
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _process_paths(paths: List[Path]) -> None:
    for path in paths:
        if not path.exists():
            print(f"[skip] {path}: file not found", file=sys.stderr)
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*.csv")):
                _process_paths([child])
            continue
        try:
            stats = normalize_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[fail] {path}: {exc}", file=sys.stderr)
            continue
        print(
            f"[ok]   {path}: in={stats['rows_in']} out={stats['rows_out']} "
            f"dup={stats['duplicates']}"
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deduplicate CSVs by txHash and unify quoting.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="CSV file(s) or directories to process.",
    )
    args = parser.parse_args(argv)
    _process_paths([p for p in args.paths])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
