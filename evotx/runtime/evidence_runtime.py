from __future__ import annotations

from typing import Any, Optional

from evotx.runtime.judge import JudgeModel
from evotx.runtime.packet_runtime import PacketRuntime


class EvidenceRuntime(PacketRuntime):
    """Backward-compatible entry point for the packet-based runtime.

    The old focus/query runtime depended on ToolRegistry and environment tools.
    The current EvoTx flow executes directly over compact evidence packets, so
    this class intentionally delegates to PacketRuntime while accepting the old
    constructor shape.
    """

    def __init__(
        self,
        tool_registry: Any = None,
        judge_model: Optional[JudgeModel] = None,
        uncertain_on_error: bool = True,
        base_dir: str = "data/cache",
    ):
        if isinstance(tool_registry, JudgeModel) and judge_model is None:
            judge_model = tool_registry
            tool_registry = None

        super().__init__(
            judge_model=judge_model or JudgeModel(),
            base_dir=base_dir,
        )
        self.tool_registry = tool_registry
        self.uncertain_on_error = uncertain_on_error