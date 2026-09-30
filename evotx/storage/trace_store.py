from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from evotx.utils.trace_utils import (
    get_trace_plan_id,
    get_trace_rule_id,
    get_trace_tx_hash,
)
from evotx.utils.json_utils import read_json, write_json


class TraceStore:
    """Filesystem store for runtime traces."""

    def __init__(self, root: str | Path = "data/traces"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, trace: Dict[str, Any]) -> Path:
        tx_hash = get_trace_tx_hash(trace).lower()
        rule_id = get_trace_rule_id(trace)
        plan_id = get_trace_plan_id(trace)
        path = self.root / f"{tx_hash}__{rule_id}__{plan_id}.json"
        write_json(path, trace)
        return path

    def load(self, path: str | Path) -> Dict[str, Any]:
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(path)
        return data

    def list(self) -> List[Path]:
        return sorted(self.root.glob("*.json"))
