from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union
import time
import uuid


EvidenceView = Literal[
    "fund",
    "trace",
    "state",
    "event",
    "address",
]

Verdict = Literal["attack", "benign", "uncertain"]
Confidence = Literal["low", "medium", "high"]


def _now() -> float:
    return time.time()


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass
class RuleCondition:
    """A local, judgeable condition in an evolving transaction rule."""

    id: str
    description: str
    expected_answer: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "RuleCondition") -> "RuleCondition":
        if isinstance(data, cls):
            return data
        d = dict(data)
        d.pop("evidence_hints", None)
        return cls(
            id=str(d["id"]),
            description=str(d["description"]),
            expected_answer=bool(d.get("expected_answer", True)),
        )


@dataclass
class EvolvingRule:
    """Long-lived detection knowledge evolved from real traces."""

    rule_id: str
    version: int
    name: str
    description: str
    conditions: List[RuleCondition]
    exclusion_conditions: List[RuleCondition] = field(default_factory=list)
    decision_policy: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "EvolvingRule") -> "EvolvingRule":
        if isinstance(data, cls):
            return data
        return cls(
            rule_id=str(data["rule_id"]),
            version=int(data.get("version", 1)),
            name=str(data.get("name", data.get("rule_id", "unnamed_rule"))),
            description=str(data.get("description", "")),
            conditions=[
                RuleCondition.from_dict(x) for x in data.get("conditions", [])
            ],
            exclusion_conditions=[
                RuleCondition.from_dict(x)
                for x in data.get("exclusion_conditions", [])
            ],
            decision_policy=str(data.get("decision_policy", "")),
            history=list(data.get("history", [])),
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", _now())),
            updated_at=float(data.get("updated_at", _now())),
        )

    def next_version(
        self,
        new_description: str,
        new_conditions: List[RuleCondition],
        new_exclusions: Optional[List[RuleCondition]] = None,
        update_note: str = "",
        decision_policy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvolvingRule":
        timestamp = _now()
        return EvolvingRule(
            rule_id=self.rule_id,
            version=self.version + 1,
            name=self.name,
            description=new_description,
            conditions=[RuleCondition.from_dict(x) for x in new_conditions],
            exclusion_conditions=[
                RuleCondition.from_dict(x)
                for x in (new_exclusions if new_exclusions is not None else self.exclusion_conditions)
            ],
            decision_policy=decision_policy
            if decision_policy is not None
            else self.decision_policy,
            history=self.history
            + [
                {
                    "from_version": self.version,
                    "to_version": self.version + 1,
                    "note": update_note,
                    "timestamp": timestamp,
                }
            ],
            metadata={**self.metadata, **(metadata or {})},
            created_at=self.created_at,
            updated_at=timestamp,
        )


@dataclass
class FocusStep:
    """A plan step that narrows the transaction evidence space."""

    id: str
    view: EvidenceView
    query_intent: str
    tool: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "FocusStep") -> "FocusStep":
        if isinstance(data, cls):
            return data
        return cls(
            id=str(data["id"]),
            view=data["view"],
            query_intent=str(data.get("query_intent", "")),
            tool=data.get("tool"),
            params=dict(data.get("params", {})),
            depends_on=list(data.get("depends_on", [])),
            budget=dict(data.get("budget", {})),
        )


@dataclass
class JudgeStep:
    """A local yes/no semantic judgment over focused evidence."""

    id: str
    question: str
    evidence_refs: List[str] = field(default_factory=list)
    default_evidence_refs: List[str] = field(default_factory=list)
    allowed_followup_views: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    max_followups: int = 0
    expected_answer: bool = True
    depends_on: List[str] = field(default_factory=list)
    consumes_state_keys: List[str] = field(default_factory=list)
    produces_state_key: str = ""
    state_prompt_role: str = ""
    state_output_schema: Dict[str, Any] = field(default_factory=dict)
    condition_id: Optional[str] = None
    view_budget: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "JudgeStep") -> "JudgeStep":
        if isinstance(data, cls):
            return data
        evidence_refs = list(
            data.get("evidence_refs")
            or data.get("default_evidence_refs")
            or []
        )
        default_refs = list(data.get("default_evidence_refs") or evidence_refs)
        return cls(
            id=str(data["id"]),
            question=str(data.get("question", "")),
            evidence_refs=evidence_refs,
            default_evidence_refs=default_refs,
            allowed_followup_views=list(data.get("allowed_followup_views", [])),
            allowed_tools=list(data.get("allowed_tools", [])),
            max_followups=int(data.get("max_followups", 0) or 0),
            expected_answer=bool(data.get("expected_answer", True)),
            depends_on=list(data.get("depends_on", [])),
            consumes_state_keys=list(data.get("consumes_state_keys", [])),
            produces_state_key=str(data.get("produces_state_key", "") or ""),
            state_prompt_role=str(data.get("state_prompt_role", "") or ""),
            state_output_schema=_dict_or_empty(data.get("state_output_schema", {})),
            condition_id=data.get("condition_id"),
            view_budget={
                str(key): int(value)
                for key, value in _dict_or_empty(data.get("view_budget", {})).items()
                if str(value).strip().lstrip("-").isdigit()
            },
        )


@dataclass
class ToolRequest:
    """A constrained read-only evidence follow-up requested by Judge."""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "ToolRequest") -> "ToolRequest":
        if isinstance(data, cls):
            return data
        return cls(
            tool=str(data.get("tool") or data.get("name") or ""),
            args=dict(data.get("args", {})),
            reason=str(data.get("reason", "")),
        )


@dataclass
class EvidencePlan:
    """A transaction-local execution plan compiled from a rule."""

    plan_id: str
    rule_id: str
    rule_version: int
    focus_steps: List[FocusStep]
    judge_steps: List[JudgeStep]
    emit_logic: str
    plan_note: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "EvidencePlan") -> "EvidencePlan":
        if isinstance(data, cls):
            return data
        return cls(
            plan_id=str(data.get("plan_id") or cls.new_id()),
            rule_id=str(data["rule_id"]),
            rule_version=int(data.get("rule_version", 1)),
            focus_steps=[
                FocusStep.from_dict(x) for x in data.get("focus_steps", [])
            ],
            judge_steps=[
                JudgeStep.from_dict(x) for x in data.get("judge_steps", [])
            ],
            emit_logic=str(data.get("emit_logic", "")),
            plan_note=str(data.get("plan_note", "")),
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", _now())),
        )

    @staticmethod
    def new_id() -> str:
        return "plan_" + uuid.uuid4().hex[:12]


@dataclass
class ToolEvidence:
    """A compact evidence slice returned by a registered tool."""

    id: str
    view: EvidenceView
    query_intent: str
    tool: Optional[str]
    params: Dict[str, Any]
    summary: Dict[str, Any]
    evidence: Dict[str, Any]
    raw: Optional[Dict[str, Any]] = None
    tool_status: str = "ok"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "ToolEvidence") -> "ToolEvidence":
        if isinstance(data, cls):
            return data
        return cls(
            id=str(data["id"]),
            view=data["view"],
            tool=data.get("tool"),
            query_intent=str(data.get("query_intent", "")),
            params=dict(data.get("params", {})),
            summary=dict(data.get("summary", {})),
            evidence=dict(data.get("evidence", {})),
            raw=data.get("raw"),
            tool_status=str(data.get("tool_status", "ok")),
            note=str(data.get("note", "")),
        )


CONDITION_FEATURE_ANALYSIS_KEYS = (
    "matched_features",
    "partial_match_features",
    "missing_required_features",
    "contradicting_features",
    "boundary_notes",
)
CONDITION_FEATURE_ANALYSIS_MAX_ITEMS = 5
CONDITION_FEATURE_ANALYSIS_MAX_CHARS = 240


def normalize_condition_feature_analysis(value: Any) -> Dict[str, Any]:
    """Normalize judge diagnostic condition features for old/new results."""
    raw = value if isinstance(value, dict) else {}
    normalized: Dict[str, Any] = {}
    for key in CONDITION_FEATURE_ANALYSIS_KEYS:
        item = raw.get(key, [])
        if isinstance(item, list):
            normalized[key] = _normalize_condition_feature_list(item)
        elif item is None or item == "":
            normalized[key] = []
        else:
            text = str(item).strip()
            normalized[key] = _normalize_condition_feature_list([text])
    return normalized


def _normalize_condition_feature_analysis(value: Any) -> Dict[str, Any]:
    return normalize_condition_feature_analysis(value)


def _normalize_condition_feature_list(values: List[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if len(text) > CONDITION_FEATURE_ANALYSIS_MAX_CHARS:
            text = text[: CONDITION_FEATURE_ANALYSIS_MAX_CHARS - 3].rstrip() + "..."
        out.append(text)
        if len(out) >= CONDITION_FEATURE_ANALYSIS_MAX_ITEMS:
            break
    return out


@dataclass
class JudgeResult:
    """The answer to one local judge question."""

    id: str
    question: str
    answer: Union[bool, str]
    reason: str
    confidence: str = "medium"
    evidence_refs: List[str] = field(default_factory=list)
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    suggested_followup_views: List[str] = field(default_factory=list)
    tool_requests: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    condition_feature_analysis: Dict[str, Any] = field(
        default_factory=lambda: normalize_condition_feature_analysis({})
    )
    state_input: Dict[str, Any] = field(default_factory=dict)
    state_output: Dict[str, Any] = field(default_factory=dict)
    stateful_runtime: Dict[str, Any] = field(default_factory=dict)
    expected_answer: bool = True
    satisfied: Optional[bool] = None

    def __post_init__(self) -> None:
        self.condition_feature_analysis = normalize_condition_feature_analysis(
            self.condition_feature_analysis
        )
        if not isinstance(self.state_input, dict):
            self.state_input = {}
        if not isinstance(self.state_output, dict):
            self.state_output = {}
        if not isinstance(self.stateful_runtime, dict):
            self.stateful_runtime = {}
        if self.satisfied is not None:
            return
        if isinstance(self.answer, str):
            self.satisfied = False
        else:
            self.satisfied = self.answer == self.expected_answer

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "JudgeResult") -> "JudgeResult":
        if isinstance(data, cls):
            return data
        answer_raw = data.get("answer", "uncertain")
        if isinstance(answer_raw, bool):
            answer: Union[bool, str] = answer_raw
        elif isinstance(answer_raw, str):
            normalized = answer_raw.lower().strip()
            if normalized in ("true", "yes"):
                answer = True
            elif normalized in ("false", "no"):
                answer = False
            else:
                answer = normalized
        else:
            answer = bool(answer_raw)
        return cls(
            id=str(data["id"]),
            question=str(data.get("question", "")),
            answer=answer,
            reason=str(data.get("reason", "")),
            confidence=data.get("confidence", "medium"),
            evidence_refs=list(data.get("evidence_refs", [])),
            supporting_evidence_ids=list(data.get("supporting_evidence_ids", [])),
            contradicting_evidence_ids=list(data.get("contradicting_evidence_ids", [])),
            missing_evidence=list(data.get("missing_evidence", [])),
            suggested_followup_views=list(data.get("suggested_followup_views", [])),
            tool_requests=list(data.get("tool_requests", [])),
            tool_calls=list(data.get("tool_calls", [])),
            condition_feature_analysis=normalize_condition_feature_analysis(
                data.get("condition_feature_analysis", {})
            ),
            state_input=_dict_or_empty(data.get("state_input", {})),
            state_output=_dict_or_empty(data.get("state_output", {})),
            stateful_runtime=_dict_or_empty(data.get("stateful_runtime", {})),
            expected_answer=bool(data.get("expected_answer", True)),
            satisfied=data.get("satisfied"),
        )


@dataclass
class Finding:
    """Final transaction-level result emitted from local judge results."""

    rule_id: str
    rule_version: int
    rule_name: str
    verdict: Verdict
    description: str
    supporting_evidence: List[str]
    judge_results: List[str]
    confidence: Confidence = "medium"
    missing_evidence: List[str] = field(default_factory=list)
    attack_supporting_evidence_by_condition: Dict[str, List[str]] = field(default_factory=dict)
    benign_exclusion_evidence_by_condition: Dict[str, List[str]] = field(default_factory=dict)
    failed_core_conditions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    missing_evidence_by_condition: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "Finding") -> "Finding":
        if isinstance(data, cls):
            return data
        return cls(
            rule_id=str(data["rule_id"]),
            rule_version=int(data.get("rule_version", 1)),
            rule_name=str(data.get("rule_name", "")),
            verdict=data.get("verdict", "uncertain"),
            description=str(data.get("description", "")),
            supporting_evidence=list(data.get("supporting_evidence", [])),
            judge_results=list(data.get("judge_results", [])),
            confidence=data.get("confidence", "medium"),
            missing_evidence=list(data.get("missing_evidence", [])),
            attack_supporting_evidence_by_condition=dict(
                data.get("attack_supporting_evidence_by_condition", {})
            ),
            benign_exclusion_evidence_by_condition=dict(
                data.get("benign_exclusion_evidence_by_condition", {})
            ),
            failed_core_conditions=dict(data.get("failed_core_conditions", {})),
            missing_evidence_by_condition=dict(
                data.get("missing_evidence_by_condition", {})
            ),
        )


@dataclass
class RuntimeTrace:
    """Execution trace for one transaction-rule-plan run."""

    tx_hash: str
    rule_id: str
    rule_version: int
    plan_id: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    judge_calls: List[Dict[str, Any]] = field(default_factory=list)
    emissions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=_now)
    ended_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        self.ended_at = _now()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "RuntimeTrace") -> "RuntimeTrace":
        if isinstance(data, cls):
            return data
        return cls(
            tx_hash=str(data["tx_hash"]),
            rule_id=str(data["rule_id"]),
            rule_version=int(data.get("rule_version", 1)),
            plan_id=str(data["plan_id"]),
            tool_calls=list(data.get("tool_calls", [])),
            judge_calls=list(data.get("judge_calls", [])),
            emissions=list(data.get("emissions", [])),
            errors=list(data.get("errors", [])),
            started_at=float(data.get("started_at", _now())),
            ended_at=data.get("ended_at"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class LabeledCase:
    """A real transaction case used for inference, review, or evaluation."""

    tx_hash: str
    chain: str
    ground_truth: Optional[Verdict] = None
    label_rationale: str = ""
    report: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "LabeledCase") -> "LabeledCase":
        if isinstance(data, cls):
            return data
        return cls(
            tx_hash=str(data["tx_hash"]),
            chain=str(data.get("chain", "eth")),
            ground_truth=data.get("ground_truth"),
            label_rationale=str(data.get("label_rationale", "")),
            report=str(data.get("report", "")),
            metadata=dict(data.get("metadata", {})),
        )
