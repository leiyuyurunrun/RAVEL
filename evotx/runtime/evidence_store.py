from __future__ import annotations

from typing import Dict, Iterable, List

from evotx.core.schemas import ToolEvidence
from evotx.utils.json_utils import stable_json_dumps


class EvidenceStore:
    """In-memory evidence cache for one runtime execution."""

    def __init__(self):
        self._items: Dict[str, ToolEvidence] = {}

    def add(self, item: ToolEvidence) -> None:
        self._items[item.id] = item

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def get(self, evidence_id: str) -> ToolEvidence:
        if evidence_id not in self._items:
            raise KeyError(f"Evidence not found: {evidence_id}")
        return self._items[evidence_id]

    def get_many(self, ids: Iterable[str]) -> List[ToolEvidence]:
        return [self.get(evidence_id) for evidence_id in ids]

    def all(self) -> List[ToolEvidence]:
        return list(self._items.values())

    def to_dict(self) -> Dict[str, Dict]:
        return {item_id: item.to_dict() for item_id, item in self._items.items()}

    def to_compact_context(self, ids: Iterable[str]) -> str:
        """Build a judge-facing compact evidence context without dumping raw data."""
        chunks = []
        for item in self.get_many(ids):
            chunks.append(
                "\n".join(
                    [
                        f"[Evidence ID: {item.id}]",
                        f"View: {item.view}",
                        f"Tool: {item.tool}",
                        f"Query intent: {item.query_intent}",
                        f"Status: {item.tool_status}",
                        f"Summary: {stable_json_dumps(item.summary)}",
                        f"Evidence: {stable_json_dumps(item.evidence)}",
                        f"Note: {item.note}",
                    ]
                )
            )
        return "\n\n---\n\n".join(chunks)
