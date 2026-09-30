from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from evotx.utils.result_utils import get_rule, get_tx_hash
from evotx.utils.json_utils import read_json, write_json


class ResultStore:
    """Filesystem store for transaction-level inference/evaluation results."""

    def __init__(self, root: str | Path = "data/results"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, result: Dict[str, Any]) -> Path:
        tx_hash = get_tx_hash(result).lower()
        rule = get_rule(result)
        rule_id = str(rule.get("rule_id", "rule"))
        rule_version = str(rule.get("version", "v"))
        path = self.root / f"{tx_hash}__{rule_id}__v{rule_version}.json"
        write_json(path, result)
        return path

    def load(self, path: str | Path) -> Dict[str, Any]:
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(path)
        return data

    def list(self) -> List[Path]:
        return sorted(self.root.glob("*.json"))
