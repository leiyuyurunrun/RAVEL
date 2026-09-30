from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from evotx.core.labels import label_slug
from evotx.core.schemas import EvidencePlan
from evotx.utils.json_utils import read_json, write_json


class PlanStore:
    """Filesystem store for reusable EvoTx evidence plans."""

    def __init__(self, root: str | Path = "data/plans"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        plan: EvidencePlan,
        *,
        attack_label: str | None = None,
        latest: bool = True,
    ) -> Path:
        slug = _slug(attack_label or plan.rule_id or "plan")
        version = _plan_version(plan)
        path = self.root / f"{slug}__plan_v{version}.json"
        write_json(path, plan.to_dict())
        if latest:
            write_json(self.root / f"{slug}__plan_latest.json", plan.to_dict())
        return path

    def save_named(self, plan: EvidencePlan, name: str) -> Path:
        path = self.root / name
        write_json(path, plan.to_dict())
        return path

    def load(self, path: str | Path) -> EvidencePlan:
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(path)
        if isinstance(data, dict) and isinstance(data.get("plan"), dict):
            data = data["plan"]
        return EvidencePlan.from_dict(data)

    def find_latest_by_attack_label(self, attack_label: str) -> Path | None:
        slug = _slug(attack_label)
        path = self.root / f"{slug}__plan_latest.json"
        return path if path.exists() else None

    def list(self) -> List[Path]:
        return sorted(self.root.glob("*.json"))


def _plan_version(plan: EvidencePlan) -> int:
    value = (plan.metadata or {}).get("plan_version")
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(plan.rule_version or 1))


def _slug(text: str) -> str:
    return label_slug(text, default="plan")
