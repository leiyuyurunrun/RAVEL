from __future__ import annotations

from typing import Any, Dict, Optional

from evotx.core.schemas import RuntimeTrace


class TraceRecorder:
    """Small helper around RuntimeTrace for consistent trace records."""

    def __init__(
        self,
        tx_hash: str,
        rule_id: str,
        rule_version: int,
        plan_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.trace = RuntimeTrace(
            tx_hash=tx_hash,
            rule_id=rule_id,
            rule_version=rule_version,
            plan_id=plan_id,
            metadata=metadata or {},
        )

    def record_tool_call(self, **payload: Any) -> None:
        self.trace.tool_calls.append(payload)

    def record_judge_call(self, **payload: Any) -> None:
        self.trace.judge_calls.append(payload)

    def record_emission(self, **payload: Any) -> None:
        self.trace.emissions.append(payload)

    def record_error(self, stage: str, **payload: Any) -> None:
        self.trace.errors.append({"stage": stage, **payload})

    def finish(self) -> RuntimeTrace:
        self.trace.finish()
        return self.trace

    def to_dict(self) -> Dict[str, Any]:
        return self.trace.to_dict()
