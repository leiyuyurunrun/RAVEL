from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, Optional


def archive_split_csvs(
    *,
    stage: str,
    benign_csv: Optional[str] = None,
    malicious_csv: Optional[str] = None,
    pos_csv: Optional[str] = None,
    neg_csv: Optional[str] = None,
    label: Optional[str] = None,
    episode: Optional[int],
    csv_root: str | Path = "data/csv",
) -> Dict[str, str]:
    """Copy split CSV inputs into an episode-scoped archive directory.

    The archive is only created for split CSV workflows with an explicit
    episode. This keeps ad-hoc runs quiet while making episode-based
    experiments self-contained.
    """
    positive_src = pos_csv or malicious_csv
    negative_src = neg_csv or benign_csv
    if episode is None or not positive_src or not negative_src:
        return {}

    safe_stage = _normalize_stage(stage)
    safe_label = _safe_label(label)
    dst_dir = Path(csv_root) / str(episode)
    dst_dir.mkdir(parents=True, exist_ok=True)

    positive_dst = dst_dir / f"{safe_stage}_{safe_label}.csv"
    negative_prefix = "neg" if neg_csv else "benign"
    negative_dst = dst_dir / f"{safe_stage}_{negative_prefix}4{safe_label}.csv"

    _copy_csv(positive_src, positive_dst)
    _copy_csv(negative_src, negative_dst)

    archived = {
        "pos_csv": str(positive_dst),
        "neg_csv": str(negative_dst),
        "positive_csv": str(positive_dst),
        "negative_csv": str(negative_dst),
        "negative_source_kind": "mixed_non_target" if neg_csv else "benign",
    }
    print(
        f"[CSVArchive] Archived split CSVs: "
        f"pos={archived['pos_csv']} neg={archived['neg_csv']} "
        f"negative_source_kind={archived['negative_source_kind']}"
    )
    return archived


def _copy_csv(src_like: str | Path, dst: Path) -> None:
    src = Path(src_like)
    if not src.exists():
        raise FileNotFoundError(f"CSV input not found: {src}")
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _normalize_stage(stage: str) -> str:
    text = str(stage or "").strip().lower()
    if text not in {"train", "eval"}:
        raise ValueError(f"Unsupported CSV archive stage: {stage}")
    return text


def _safe_label(label: Optional[str]) -> str:
    text = str(label or "attack").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "attack"
