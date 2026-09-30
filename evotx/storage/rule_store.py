from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from evotx.core.labels import label_slug
from evotx.core.schemas import EvolvingRule
from evotx.utils.json_utils import read_json, write_json


class RuleStore:
    """Filesystem-backed versioned rule store."""

    def __init__(self, root: str | Path = "data/rules"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        rule: EvolvingRule,
        *,
        attack_label: str | None = None,
        latest: bool = True,
    ) -> Path:
        """Save a rule and expose a stable attack-label alias when available.

        Rule ids may come from an LLM cold start or later evolution and are not
        guaranteed to match the dataset label. Batch scripts, however, should be
        able to resolve `{label}__latest.json` just like plan artifacts do.
        """
        payload = rule.to_dict()
        canonical_path = self.path_for(rule.rule_id, rule.version)
        write_json(canonical_path, payload)
        if latest:
            write_json(self.root / f"{rule.rule_id}__latest.json", payload)

        label_slug = self._normalize_label(
            attack_label or (rule.metadata or {}).get("attack_label", "")
        )
        if label_slug and label_slug != rule.rule_id:
            label_path = self.path_for(label_slug, rule.version)
            write_json(label_path, payload)
            if latest:
                write_json(self.root / f"{label_slug}__latest.json", payload)
            return label_path

        return canonical_path

    def load(self, rule_id: str, version: Optional[int] = None) -> EvolvingRule:
        path = (
            self.root / f"{rule_id}__latest.json"
            if version is None
            else self.path_for(rule_id, version)
        )
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(path)
        return EvolvingRule.from_dict(data)

    def list_versions(self, rule_id: str) -> List[int]:
        versions = []
        for path in self.root.glob(f"{rule_id}__v*.json"):
            stem = path.stem
            if "__v" not in stem:
                continue
            try:
                versions.append(int(stem.rsplit("__v", 1)[1]))
            except ValueError:
                pass
        return sorted(versions)

    def path_for(self, rule_id: str, version: int) -> Path:
        return self.root / f"{rule_id}__v{version}.json"

    def find_latest_by_attack_label(self, attack_label: str) -> Optional[Path]:
        normalized = self._normalize_label(attack_label)
        if not normalized:
            return None

        direct_path = self.root / f"{normalized}__latest.json"
        if direct_path.exists():
            return direct_path

        for path in sorted(self.root.glob("*__latest.json")):
            data = read_json(path, default={}) or {}
            metadata = data.get("metadata", {}) or {}
            candidate = self._normalize_label(metadata.get("attack_label", ""))
            if candidate == normalized:
                return path
        return None

    def load_by_attack_label(self, attack_label: str) -> EvolvingRule:
        path = self.find_latest_by_attack_label(attack_label)
        if path is None:
            raise FileNotFoundError(
                f"No latest rule found for attack label: {attack_label}"
            )
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(path)
        return EvolvingRule.from_dict(data)

    @staticmethod
    def _normalize_label(value: str) -> str:
        return label_slug(value, default="")
