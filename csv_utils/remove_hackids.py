"""Remove specific HackId rows from a CSV and write the result to a new file.

The output CSV is a copy of the input with rows whose ``HackId`` matches
``EVAL_NEG_REMOVED`` removed. The script prints the number of dropped
rows and any HackIds that were not found in the source.

Edit ``CSV_PATH_IN``/``CSV_PATH_OUT`` and ``EVAL_NEG_REMOVED`` below, then
run:

    python csv_utils/remove_hackids.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


# === Configuration =========================================================
# Source CSV to read from.
from pathlib import Path

# CSV_PATH_IN = Path(
#     r"data\csv\v2\Insufficient validation\evaluation_shared4neg.csv"
# )

# CSV_PATH_OUT = Path(
#     r"data\csv\v2\Insufficient validation\evaluation_shared4neg-filtered.csv"
# )

# EVAL_NEG_REMOVED = [
#     "204",
#     "179", "210", "263", "326", "381",
#     "188", "131", "254", "209",
#     "29", "41", "245", "360",
#     "15", "63", "355", "336"
# ]

from pathlib import Path

# CSV_PATH_IN = Path(
#     r"data\csv\v2\Access control\evaluation_shared4neg.csv"
# )

# CSV_PATH_OUT = Path(
#     r"data\csv\v2\Access control\evaluation_shared4neg-filtered.csv"
# )

# EVAL_NEG_REMOVED = [
#     "35",   # arbitrary logic / malicious payload execution，太接近 AC 的 privileged-path abuse
#     "40",   # allowance + unvalidated paraswapData，和 approval/authorization 邊界過近
#     "48",   # arbitrary call by crafted inputData，太接近 AC/arbitrary execution
#     "125",  # allowance 不足仍 transferFrom，本質接近 approval/authorization bypass
#     "157",  # call injection + approved funds withdrawal，和 arbitrary execution / approval abuse 太近
#     "162",  # 描述中明確包含 bypass access controls / unauthorized transfers
#     "182",  # malicious calldata transferFrom victim funds，和 approval/authorization 邊界過近
#     "235",  # invalid market address 操縱 _msgSender 並 drain victims，身份/授權邊界不乾淨
#     "333"   # arbitrary external calls theft from approved users，太接近 AC arbitrary-call 類
# ]

from pathlib import Path

# CSV_PATH_IN = Path(
#     r"data\csv\v2\Reentrancy\evaluation_shared4neg.csv"
# )

# CSV_PATH_OUT = Path(
#     r"data\csv\v2\Reentrancy\evaluation_shared4neg-filtered.csv"
# )

# EVAL_NEG_REMOVED = []

# from pathlib import Path

# CSV_PATH_IN = Path(
#     r"data\csv\v2\Price manipulation\evaluation_shared4neg.csv"
# )

# CSV_PATH_OUT = Path(
#     r"data\csv\v2\Price manipulation\evaluation_shared4neg-filtered.csv"
# )


# 第二次提问的时候保留了11、179
# EVAL_NEG_REMOVED = [
#     "11",
#     "179",
#     "263",

#     "188",
#     "131",
#     "254",
#     "209",
#     "107",

#     "340",
#     "362",
#     "367"
# ]

from pathlib import Path

# CSV_PATH_IN = Path(
#     r"data\csv\v2\Flashloans\evaluation_shared4neg.csv"
# )

# CSV_PATH_OUT = Path(
#     r"data\csv\v2\Flashloans\evaluation_shared4neg-filtered.csv"
# )

# EVAL_NEG_REMOVED = [
#     "29",   # cause 明确使用 flash loan 操纵 WBNB-USDT 价格并完成获利
#     "41",   # cause 明确使用 flash loan 操纵 lending/oracle/collateral 过程
#     "146",  # flashloan callback validation issue，和 Flashloans 边界太近
#     "360",  # 多标签含 Flashloans，staking reward 基于 flashloan/spot price manipulation
#     "389"   # 多标签含 Flashloans，ERC777 reentrancy + flashloan
# ]
# from pathlib import Path

# CSV_PATH_IN = Path(
#     r"data\csv\v2\Market manipulation\evaluation_shared4neg.csv"
# )

# CSV_PATH_OUT = Path(
#     r"data\csv\v2\Market manipulation\evaluation_shared4neg-filtered.csv"
# )

# EVAL_NEG_REMOVED = [
#     "11",
#     "179",
#     "263",
#     "326",

#     "29",
#     "41",
#     "86",
#     "245",
#     "360",

#     "340",
#     "362",
#     "367",

#     "63",
#     "355"
# ]

# from pathlib import Path

# CSV_PATH_IN = Path(
#     r"data\csv\v2\Protocol accounting exploitation\evaluation_shared4neg.csv"
# )

# CSV_PATH_OUT = Path(
#     r"data\csv\v2\Protocol accounting exploitation\evaluation_shared4neg-filtered.csv"
# )

# EVAL_NEG_REMOVED = [
#     "309",
#     "352",
#     "292",

#     "179",
#     "210",
#     "326",
#     "381",

#     "209",
#     "360",
#     "389"
# ]

from pathlib import Path

CSV_PATH_IN = Path(
    r"data\csv\v2\Token semantic exploitation\evaluation_shared4neg.csv"
)

CSV_PATH_OUT = Path(
    r"data\csv\v2\Token semantic exploitation\evaluation_shared4neg-filtered.csv"
)

EVAL_NEG_REMOVED = [
    "194",
    "202",
    "289",
    "292",
    "309",
    "389"
]
# ===========================================================================


HACKID_COLUMN = "HackId"


def _parse_mixed_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV with mixed quoted/unquoted rows.

    Falls back to a permissive state-machine parser when the standard
    ``csv.reader`` chokes on embedded newlines inside unquoted fields.
    A leading UTF-8 BOM on the file is stripped so the first column name
    does not get a hidden ``\\ufeff`` prefix.
    """
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    if not raw.strip():
        return [], []

    lines = raw.splitlines(keepends=False)
    try:
        rows = [row for row in csv.reader(lines)]
        return rows[0], rows[1:]
    except csv.Error:
        pass

    return _parse_permissive(raw)


def _parse_permissive(raw: str) -> Tuple[List[str], List[List[str]]]:
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

    if buf or fields:
        fields.append("".join(buf))
        rows.append(fields)

    if not rows:
        return [], []
    return rows[0], rows[1:]


def _normalize_row(row: List[str], target_len: int) -> List[str]:
    if len(row) < target_len:
        return row + [""] * (target_len - len(row))
    if len(row) > target_len:
        return row[:target_len]
    return row


def _clean_header_cell(name: str) -> str:
    """Normalize a header cell for comparison.

    Tolerates:
    - A leading UTF-8 BOM (``\\ufeff``) inserted by some editors.
    - Surrounding double quotes, e.g. ``"HackId"`` or ``"HackId""``.
    - Surrounding whitespace from manual edits.
    """
    return name.lstrip("﻿").strip().strip('"').strip()


def _hackid_column_index(header: List[str]) -> int:
    """Locate the ``HackId`` column.

    Tolerates cosmetic variations seen in the wild (BOM, surrounding
    quotes, whitespace) by normalizing each header cell before comparing.
    """
    cleaned = [_clean_header_cell(h) for h in header]
    if HACKID_COLUMN in cleaned:
        return cleaned.index(HACKID_COLUMN)
    for idx, name in enumerate(cleaned):
        if name.lower() == HACKID_COLUMN.lower():
            return idx
    raise ValueError(f"Column '{HACKID_COLUMN}' not found in header: {header}")


def filter_hackids(
    src: Path,
    dst: Path,
    ids_to_remove: Iterable[str],
) -> Dict[str, int]:
    """Read ``src``, drop rows whose ``HackId`` is in ``ids_to_remove``,
    and write the remaining rows to ``dst`` (same header, same columns).

    Returns stats: ``rows_in``, ``rows_out``, ``removed``, ``missing``.
    """
    header, data = _parse_mixed_csv(src)
    if not header:
        # Still write an empty file with an empty header so the
        # destination is never missing or ambiguous.
        _write_csv(dst, [], [])
        return {"rows_in": 0, "rows_out": 0, "removed": 0, "missing": 0}

    hackid_idx = _hackid_column_index(header)
    target_len = len(header)
    targets: Set[str] = {str(i).strip()
                         for i in ids_to_remove if str(i).strip()}

    kept: List[List[str]] = []
    seen_matches: Set[str] = set()
    for raw_row in data:
        row = _normalize_row(list(raw_row), target_len)
        if not any(cell.strip() for cell in row):
            continue
        current = (row[hackid_idx] or "").strip().strip('"')
        if current in targets:
            seen_matches.add(current)
            continue
        kept.append(row)

    _write_csv(dst, header, kept)

    missing = len(targets - seen_matches)
    return {
        "rows_in": len(data),
        "rows_out": len(kept),
        "removed": len(seen_matches),
        "missing": missing,
    }


def _write_csv(path: Path, header: List[str], rows: Iterable[List[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(
            fh,
            quoting=csv.QUOTE_NONNUMERIC,
            lineterminator="\n",
        )
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def main() -> int:
    if not CSV_PATH_IN.exists():
        print(f"[fail] {CSV_PATH_IN}: input file not found", file=sys.stderr)
        return 1

    # Make sure the destination directory exists so writes don't fail.
    CSV_PATH_OUT.parent.mkdir(parents=True, exist_ok=True)

    try:
        stats = filter_hackids(CSV_PATH_IN, CSV_PATH_OUT, EVAL_NEG_REMOVED)
    except Exception as exc:
        print(f"[fail] {CSV_PATH_IN} -> {CSV_PATH_OUT}: {exc}",
              file=sys.stderr)
        return 1

    print(
        f"[ok]   {CSV_PATH_IN} -> {CSV_PATH_OUT}: "
        f"in={stats['rows_in']} out={stats['rows_out']} "
        f"removed={stats['removed']} missing={stats['missing']}"
    )
    if stats["missing"]:
        print(
            f"[warn] {stats['missing']} HackId(s) not found in the file",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
