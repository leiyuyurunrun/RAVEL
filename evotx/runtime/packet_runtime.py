from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from evotx.core.labels import normalize_attack_label
from evotx.core.logic import extract_logic_names, safe_eval_bool_expr
from evotx.core.plan import (
    ACCESS_CONTROL_STATEFUL_RUNTIME_MODE,
    ACCESS_CONTROL_CANDIDATE_STATE_KEY,
    FLASHLOANS_STATEFUL_RUNTIME_MODE,
    MARKET_MANIPULATION_STATEFUL_RUNTIME_MODE,
    PROTOCOL_ACCOUNTING_STATEFUL_RUNTIME_MODE,
    REENTRANCY_STATE_SCHEMA_VERSION,
    REENTRANCY_STATEFUL_RUNTIME_MODE,
    TOKEN_SEMANTIC_CANDIDATE_STATE_KEY,
    TOKEN_SEMANTIC_STATEFUL_RUNTIME_MODE,
    constrain_first_pass_view_refs,
    is_reentrant_state_order_text,
    normalize_access_control_binding_mode,
    normalize_reentrancy_binding_mode,
    resolve_judge_step_view_budget,
)
from evotx.planner.plan_validator import dependency_affected_condition_ids
from evotx.core.schemas import (
    EvolvingRule,
    EvidencePlan,
    JudgeResult,
    JudgeStep,
    normalize_condition_feature_analysis,
)
from evotx.runtime.adaptive_evidence import (
    ADAPTIVE_EVIDENCE_NOTE,
    AdaptiveEvidenceConfig,
    AdaptiveEvidenceDecision,
    EvidenceProfile,
    adapt_step_evidence_refs,
    collect_adaptive_candidate_views,
    extract_evidence_profile,
    normalize_adaptive_mode,
    resolve_adaptive_mode,
)
from evotx.runtime.evidence_tool_registry import (
    ALLOWED_EVIDENCE_TOOLS,
    EvidenceToolRegistry,
)
from evotx.runtime.evidence_renderer import PROMPT_DICTIONARY_KEY
from evotx.runtime.followup_context import FollowupContextBuilder
from evotx.runtime.judge import JudgeModel
from evotx.runtime.judge_reuse import judge_step_fingerprint
from evotx.runtime.missing_evidence import (
    dedupe_missing_evidence,
    open_blocking_texts,
    normalize_missing_evidence,
    resolve_missing_evidence_after_tool_call,
)
from evotx.runtime.packet_builder import (
    DEFAULT_PACKET_VIEWS,
    build_or_load_packet,
    resolve_input_paths,
)
from evotx.runtime.view_catalog import (
    PACKET_VIEW_CATALOG,
    build_allowed_view_summary,
    packet_view_render_policy,
)
from evotx.utils.json_utils import (
    JsonExtractionError,
    extract_last_schema_valid_object,
    stable_json_dumps,
    structured_output_text,
)


_STRUCTURED_LOG_PRINT_LOCK = threading.Lock()
_RATE_LIMIT_FALLBACK_LOCK = threading.Lock()
_RATE_LIMIT_SERIAL_PROVIDERS: Dict[str, Dict[str, Any]] = {}
_ACCESS_CONTROL_SOURCE_FAILURE_STATUSES = {
    "function_not_found",
    "invalid_address",
    "invalid_function_name",
    "source_unavailable",
    "tool_unavailable",
}
_SOURCE_TERMINAL_FAILURE_STATUSES = set(_ACCESS_CONTROL_SOURCE_FAILURE_STATUSES)
_ACCESS_CONTROL_SOURCE_FALLBACK_VIEWS = (
    "source_unavailable_auth_view",
)

JUDGE_FOLLOWUP_MODES = {"plan", "disabled", "expanded"}


def normalize_judge_followup_mode(mode: str) -> str:
    normalized = str(mode or "plan").strip().lower()
    if normalized not in JUDGE_FOLLOWUP_MODES:
        raise ValueError(
            "judge_followup_mode must be one of: "
            + ", ".join(sorted(JUDGE_FOLLOWUP_MODES))
        )
    return normalized


def apply_judge_followup_policy(
    plan: EvidencePlan | Dict[str, Any],
    *,
    mode: str = "plan",
    expanded_view_count: int = 2,
) -> tuple[EvidencePlan, Dict[str, Any]]:
    """Build an execution-only plan for the Judge follow-up ablation."""
    normalized_mode = normalize_judge_followup_mode(mode)
    effective_plan = EvidencePlan.from_dict(copy.deepcopy(
        plan.to_dict() if isinstance(plan, EvidencePlan) else plan
    ))
    promoted_limit = max(0, int(expanded_view_count or 0))
    step_audits: List[Dict[str, Any]] = []

    for step in effective_plan.judge_steps:
        original_defaults = _unique_strings(
            list(step.default_evidence_refs or step.evidence_refs)
        )
        original_followups = _unique_strings(step.allowed_followup_views or [])
        original_tools = _unique_strings(step.allowed_tools or [])
        original_max_followups = max(0, int(step.max_followups or 0))
        promoted: List[str] = []

        if normalized_mode == "expanded" and promoted_limit > 0:
            candidates = [
                view
                for view in original_followups
                if view in PACKET_VIEW_CATALOG and view not in original_defaults
            ]
            original_order = {view: index for index, view in enumerate(candidates)}
            candidates.sort(key=lambda view: (
                int(packet_view_render_policy(view).get("priority", 50) or 50),
                original_order[view],
            ))
            promoted = candidates[:promoted_limit]

        if normalized_mode != "plan":
            effective_defaults = _unique_strings(original_defaults + promoted)
            step.default_evidence_refs = effective_defaults
            step.evidence_refs = list(effective_defaults)
            step.allowed_followup_views = []
            step.allowed_tools = []
            step.max_followups = 0

            large_count = sum(
                1
                for view in effective_defaults
                if str(packet_view_render_policy(view).get("cost", "")).lower()
                == "large"
            )
            step.view_budget = {
                **dict(step.view_budget or {}),
                "max_default_views": max(1, len(effective_defaults)),
                "max_followup_views": 0,
                "max_large_views": max(0, large_count),
            }

        step_audits.append({
            "judge_id": step.id,
            "condition_id": step.condition_id or step.id,
            "original_default_views": original_defaults,
            "original_followup_views": original_followups,
            "original_allowed_tools": original_tools,
            "original_max_followups": original_max_followups,
            "promoted_views": promoted,
            "effective_default_views": list(
                step.default_evidence_refs or step.evidence_refs
            ),
            "effective_followup_views": list(step.allowed_followup_views or []),
            "effective_allowed_tools": list(step.allowed_tools or []),
            "effective_max_followups": max(0, int(step.max_followups or 0)),
        })

    audit = {
        "schema_version": "evotx.judge_followup_policy.v1",
        "mode": normalized_mode,
        "expanded_view_count": promoted_limit,
        "single_call": normalized_mode != "plan",
        "adaptive_evidence_enabled": normalized_mode == "plan",
        "near_miss_enabled": normalized_mode == "plan",
        "dynamic_aggregation_enabled": normalized_mode == "plan",
        "steps": step_audits,
    }
    effective_plan.metadata = {
        **dict(effective_plan.metadata or {}),
        "runtime_judge_followup_policy": copy.deepcopy(audit),
    }
    return effective_plan, audit


def _judge_followup_step_audit(
    policy: Dict[str, Any],
    judge_id: str,
) -> Dict[str, Any]:
    for item in list(policy.get("steps", []) or []):
        if str(item.get("judge_id") or "") == str(judge_id or ""):
            return {
                "mode": str(policy.get("mode") or "plan"),
                "single_call": bool(policy.get("single_call")),
                **copy.deepcopy(item),
            }
    return {
        "mode": str(policy.get("mode") or "plan"),
        "single_call": bool(policy.get("single_call")),
        "judge_id": str(judge_id or ""),
    }


def _llm_provider_key(llm: Any) -> str:
    provider = str(getattr(llm, "provider", "") or "").strip().lower()
    if provider:
        return provider
    return str(type(llm).__name__ if llm is not None else "unknown").lower()


def _judge_provider_key(judge_model: JudgeModel) -> str:
    return _llm_provider_key(getattr(judge_model, "llm", None))


def _rate_limit_serial_fallback_active(provider: str) -> bool:
    key = str(provider or "unknown").strip().lower()
    with _RATE_LIMIT_FALLBACK_LOCK:
        return key in _RATE_LIMIT_SERIAL_PROVIDERS


def _activate_rate_limit_serial_fallback(
    provider: str,
    *,
    error: BaseException,
) -> Dict[str, Any]:
    key = str(provider or "unknown").strip().lower()
    with _RATE_LIMIT_FALLBACK_LOCK:
        state = _RATE_LIMIT_SERIAL_PROVIDERS.setdefault(
            key,
            {
                "provider": key,
                "activated_at": datetime.now(timezone.utc).isoformat(),
                "trigger_count": 0,
                "last_error": "",
            },
        )
        state["trigger_count"] = int(state.get("trigger_count", 0) or 0) + 1
        state["last_error"] = repr(error)
        return dict(state)


def _reset_rate_limit_serial_fallback(provider: str = "") -> None:
    """Reset process-local fallback state; intended for tests and fresh runs."""
    key = str(provider or "").strip().lower()
    with _RATE_LIMIT_FALLBACK_LOCK:
        if key:
            _RATE_LIMIT_SERIAL_PROVIDERS.pop(key, None)
        else:
            _RATE_LIMIT_SERIAL_PROVIDERS.clear()


def _is_rate_limit_error(error: BaseException) -> bool:
    pending: List[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        name = type(current).__name__.lower()
        text = str(current).lower()
        status_code = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        response_status = getattr(response, "status_code", None)
        code = getattr(current, "code", None)
        if (
            "ratelimit" in name
            or "rate_limit" in name
            or status_code == 429
            or response_status == 429
            or str(code or "") in {"429", "1302", "1305"}
            or "error code: 429" in text
            or "status code: 429" in text
            or "'code': '1302'" in text
            or "'code': '1305'" in text
            or "速率限制" in text
            or "访问量过大" in text
        ):
            return True
        for nested in (getattr(current, "__cause__", None), getattr(current, "__context__", None)):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _is_retryable_transport_error(error: BaseException) -> bool:
    pending: List[BaseException] = [error]
    seen: set[int] = set()
    retryable_names = {
        "apiconnectionerror",
        "apitimeouterror",
        "connecterror",
        "connectionerror",
        "connecttimeout",
        "readtimeout",
        "timeouterror",
    }
    retryable_text = (
        "connection error",
        "connection reset",
        "connection aborted",
        "server disconnected",
        "remote protocol error",
        "timed out",
        "timeout",
    )
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        name = type(current).__name__.lower()
        text = str(current).lower()
        if name in retryable_names or any(token in name for token in retryable_names):
            return True
        if any(token in text for token in retryable_text):
            return True
        for nested in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _call_with_rate_limit_retry(
    call: Callable[[], Any],
    *,
    enabled: bool,
    provider: str,
    max_retries: int,
    initial_delay_seconds: float,
) -> tuple[Any, Dict[str, Any]]:
    retry_limit = max(0, int(max_retries or 0))
    base_delay = max(0.0, float(initial_delay_seconds or 0.0))
    metadata: Dict[str, Any] = {
        "triggered": False,
        "provider": str(provider or "unknown"),
        "retry_count": 0,
        "max_retries": retry_limit,
        "delays_seconds": [],
        "errors": [],
        "recovered": False,
        "rate_limit_triggered": False,
        "transport_retry_triggered": False,
        "retry_kinds": [],
    }
    while True:
        try:
            value = call()
            metadata["recovered"] = bool(metadata["triggered"])
            return value, metadata
        except Exception as exc:
            rate_limited = _is_rate_limit_error(exc)
            transport_retryable = (
                not rate_limited and _is_retryable_transport_error(exc)
            )
            if not enabled or not (rate_limited or transport_retryable):
                raise
            metadata["triggered"] = True
            metadata["errors"].append(repr(exc))
            retry_kind = "rate_limit" if rate_limited else "transport"
            metadata["retry_kinds"].append(retry_kind)
            metadata["rate_limit_triggered"] = bool(
                metadata["rate_limit_triggered"] or rate_limited
            )
            metadata["transport_retry_triggered"] = bool(
                metadata["transport_retry_triggered"] or transport_retryable
            )
            if rate_limited:
                _activate_rate_limit_serial_fallback(provider, error=exc)
            effective_retry_limit = retry_limit if rate_limited else min(retry_limit, 1)
            if int(metadata["retry_count"]) >= effective_retry_limit:
                raise
            retry_number = int(metadata["retry_count"]) + 1
            delay = base_delay * (2 ** (retry_number - 1))
            metadata["retry_count"] = retry_number
            metadata["delays_seconds"].append(delay)
            print(
                f"[PacketRuntime] Judge {retry_kind} error detected; "
                f"provider={provider} retry={retry_number}/{retry_limit} "
                f"delay={delay}s concurrency=1"
            )
            if delay > 0:
                time.sleep(delay)


def collect_required_packet_views_from_plan(
    plan: EvidencePlan,
    *,
    adaptive_evidence: bool = False,
    adaptive_evidence_mode: str = "off",
    attack_label: str = "",
    tx_hash: str = "",
    base_dir: str = "data/cache",
    adaptive_config: AdaptiveEvidenceConfig | None = None,
) -> List[str]:
    """Collect all packet views the plan may need at judge or follow-up time."""
    catalog = set(PACKET_VIEW_CATALOG)
    required: List[str] = []

    def add(view_name: Any) -> None:
        view = str(view_name or "").strip()
        if not view or view not in catalog or view in required:
            return
        required.append(view)

    for base_view in (
        "tx_card",
        "evidence_adequacy_view",
        "operation_summary_view",
        "classification_digest_view",
    ):
        add(base_view)
    if normalize_attack_label(attack_label, default="") == "access_control":
        # Source failure fallback is runtime-owned and must not disappear when
        # a generated or updated plan omits the packet view.
        add("source_unavailable_auth_view")
    state_order_views = {
        "reentrancy_candidate_catalog_view",
        "reentrancy_state_order_summary_view",
        "reentrancy_state_order_view",
    }
    state_order_needed = any(
        bool(
            state_order_views.intersection(
                list(step.default_evidence_refs or [])
                + list(step.evidence_refs or [])
                + list(step.allowed_followup_views or [])
            )
        )
        or (
            normalize_attack_label(attack_label, default="") == "reentrancy"
            and is_reentrant_state_order_text(
                step.question,
                attack_label=attack_label,
            )
        )
        for step in list(plan.judge_steps or [])
    )
    if state_order_needed:
        add("reentrancy_state_order_summary_view")
    for step in list(plan.judge_steps or []):
        for ref in list(step.default_evidence_refs or []):
            add(ref)
        for ref in list(step.evidence_refs or []):
            add(ref)
        for ref in list(step.allowed_followup_views or []):
            add(ref)
    for ref in collect_adaptive_candidate_views(
        attack_label=attack_label,
        adaptive_evidence=adaptive_evidence,
        adaptive_evidence_mode=adaptive_evidence_mode,
        tx_hash=tx_hash,
        base_dir=base_dir,
        config=adaptive_config,
    ):
        add(ref)
    return required


class PacketRuntime:
    """Packet-based EvoTx runtime: load cached evidence -> judge -> verdict."""

    def __init__(
        self,
        judge_model: JudgeModel,
        base_dir: str = "data/cache",
        debug_allow_list_views: bool = False,
        transcript_phase: str = "",
        aggregator_llm: Any | None = None,
    ):
        self.judge_model = judge_model
        self.aggregator_llm = aggregator_llm or getattr(judge_model, "llm", None)
        self.base_dir = base_dir
        self.debug_allow_list_views = debug_allow_list_views
        self.transcript_phase = transcript_phase

    def execute(
        self,
        tx_hash: str,
        rule: EvolvingRule,
        plan: EvidencePlan,
        force_rebuild_packet: bool = False,
        packet_override: Dict[str, Any] | None = None,
        freeze_packet: bool = False,
        judge_reuse_cache: Dict[str, Dict[str, Any]] | None = None,
        early_stop: bool = False,
        early_stop_policy: str = "conservative_negative",
        adaptive_evidence: bool = False,
        adaptive_evidence_mode: str = "auto",
        adaptive_evidence_max_direct_trace_nodes: int = 80,
        adaptive_evidence_max_direct_trace_chars: int = 45000,
        adaptive_evidence_max_medium_trace_nodes: int = 250,
        adaptive_evidence_debug: bool = False,
        parallel_judge: bool = False,
        judge_concurrency: int = 1,
        structured_judge_logs: bool = True,
        iv_stateful_runtime: bool = True,
        access_control_binding_mode: str = "stateful",
        reentrancy_binding_mode: str = "soft",
        dynamic_aggregation: bool = True,
        judge_followup_mode: str = "plan",
        judge_expanded_followup_views: int = 2,
        rate_limit_serial_fallback: bool = True,
        rate_limit_retry_attempts: int = 3,
        rate_limit_retry_delay_seconds: float = 2.0,
        disable_stateful_bindings: bool = False,
    ) -> Dict[str, Any]:
        if disable_stateful_bindings:
            plan = disable_stateful_bindings_in_plan(plan)
        plan, judge_followup_policy = apply_judge_followup_policy(
            plan,
            mode=judge_followup_mode,
            expanded_view_count=judge_expanded_followup_views,
        )
        normalized_judge_followup_mode = str(judge_followup_policy["mode"])
        adaptive_evidence_requested = bool(adaptive_evidence)
        dynamic_aggregation_requested = bool(dynamic_aggregation)
        if normalized_judge_followup_mode != "plan":
            adaptive_evidence = False
            adaptive_evidence_mode = "off"
            dynamic_aggregation = False
        judge_followup_policy.update({
            "adaptive_evidence_requested": adaptive_evidence_requested,
            "adaptive_evidence_effective": bool(adaptive_evidence),
            "dynamic_aggregation_requested": dynamic_aggregation_requested,
            "dynamic_aggregation_effective": bool(dynamic_aggregation),
        })
        print(
            f"[PacketRuntime] Start tx={tx_hash} "
            f"rule={rule.rule_id} v{rule.version} plan={plan.plan_id} "
            f"judge_followup_mode={normalized_judge_followup_mode}"
        )

        errors: List[Dict[str, Any]] = []
        started_at = time.time()
        attack_label = str((rule.metadata or {}).get("attack_label") or "")
        adaptive_config = AdaptiveEvidenceConfig(
            max_direct_trace_nodes=max(0, int(adaptive_evidence_max_direct_trace_nodes or 0)),
            max_direct_trace_chars=max(0, int(adaptive_evidence_max_direct_trace_chars or 0)),
            max_medium_trace_nodes=max(0, int(adaptive_evidence_max_medium_trace_nodes or 0)),
        )
        normalized_adaptive_mode = normalize_adaptive_mode(adaptive_evidence_mode)
        adaptive_enabled = bool(adaptive_evidence) and normalized_adaptive_mode not in {
            "off",
            "planned",
            "planned_views",
        }
        stateful_enabled = _stateful_runtime_enabled(
            rule,
            plan,
            iv_stateful_runtime=bool(iv_stateful_runtime),
            access_control_binding_mode=access_control_binding_mode,
            reentrancy_binding_mode=reentrancy_binding_mode,
            disable_stateful_bindings=bool(disable_stateful_bindings),
        )
        stateful_metadata = _plan_stateful_runtime_metadata(rule, plan)
        judge_provider_key = _judge_provider_key(self.judge_model)
        rate_limit_fallback_was_active = bool(rate_limit_serial_fallback) and (
            _rate_limit_serial_fallback_active(judge_provider_key)
        )
        parallel_config = _parallel_judge_config(
            requested=bool(parallel_judge),
            judge_concurrency=judge_concurrency,
            early_stop=bool(early_stop),
            rate_limit_serial_fallback_active=rate_limit_fallback_was_active,
        )
        if parallel_config.get("disabled_reason") == "early_stop_enabled":
            print("[PacketRuntime] Parallel judge disabled because early_stop is enabled.")
        if stateful_enabled and parallel_config.get("effective"):
            parallel_config["enabled"] = False
            parallel_config["effective"] = False
            parallel_config["disabled_reason"] = "stateful_runtime_enabled"
            print(
                "[PacketRuntime] Parallel judge disabled because stateful runtime is enabled."
            )
        if parallel_config.get("disabled_reason") == "rate_limit_serial_fallback_active":
            print(
                "[PacketRuntime] Parallel judge disabled because a prior 429 "
                f"activated serial fallback for provider={judge_provider_key}."
            )

        # 1. Load or build evidence packet
        try:
            required_views = collect_required_packet_views_from_plan(
                plan,
                adaptive_evidence=adaptive_enabled,
                adaptive_evidence_mode=normalized_adaptive_mode,
                attack_label=attack_label,
                tx_hash=tx_hash,
                base_dir=self.base_dir,
                adaptive_config=adaptive_config,
            )
            paths = resolve_input_paths(tx_hash, base_dir=self.base_dir)
            packet_path = paths.get("packet")
            if packet_override is not None:
                packet = copy.deepcopy(packet_override)
                packet_tx_hash = str(packet.get("transaction_hash") or "").lower()
                if packet_tx_hash and packet_tx_hash != str(tx_hash).lower():
                    raise ValueError(
                        "Frozen packet transaction mismatch: "
                        f"expected={tx_hash} actual={packet_tx_hash}"
                    )
                packet_loaded_from_cache = False
                packet_source = str(
                    packet.pop("_frozen_packet_source", "frozen:baseline_result")
                    or "frozen:baseline_result"
                )
            else:
                pre_existing = (
                    packet_path is not None
                    and packet_path.exists()
                    and not force_rebuild_packet
                )
                include_views = (
                    list(DEFAULT_PACKET_VIEWS)
                    if freeze_packet
                    else required_views
                )
                packet = build_or_load_packet(
                    tx_hash,
                    base_dir=self.base_dir,
                    force_rebuild=force_rebuild_packet,
                    include_views=include_views,
                    # A frozen baseline must be complete enough for a later
                    # candidate plan without consulting mutable cache inputs.
                    required_views=(include_views if freeze_packet else required_views),
                )
                packet_loaded_from_cache = bool(
                    (packet.get("cache_policy", {}) or {}).get("loaded_from_cache", pre_existing)
                )
                packet_source = str(packet_path) if packet_path else ""
            packet_required_views_missing = [
                view
                for view in required_views
                if view not in ((packet.get("views", {}) or {}) if isinstance(packet, dict) else {})
            ]
            if packet_override is not None and packet_required_views_missing:
                raise ValueError(
                    "Frozen packet is missing plan-required views: "
                    + ", ".join(packet_required_views_missing)
                )
            packet_identity_fingerprint = str(
                packet.pop("_frozen_packet_identity_fingerprint", "")
                or _packet_identity_fingerprint(packet)
            )
            packet_fingerprint = packet_identity_fingerprint
            chain = _infer_packet_chain(packet)
            evidence_profile = extract_evidence_profile(
                packet,
                max_context_chars=int(self.judge_model.max_context_chars or 0),
            )
            adaptive_selected_mode, adaptive_mode_reason = (
                resolve_adaptive_mode(
                    requested_mode=normalized_adaptive_mode,
                    profile=evidence_profile,
                    max_context_chars=int(self.judge_model.max_context_chars or 0),
                    config=adaptive_config,
                )
                if adaptive_enabled
                else ("planned_views", "adaptive evidence disabled")
            )
        except Exception as exc:
            print(f"[PacketRuntime] Packet loading failed: {exc!r}")
            return self._error_result(
                tx_hash, rule, plan, f"Packet loading failed: {exc}", started_at
            )

        # 2. Run judge steps
        judge_results: List[Dict[str, Any]] = []
        judge_step_traces: List[Dict[str, Any]] = []
        judge_values: Dict[str, bool] = {}
        has_uncertain = False
        reused_judge_ids: List[str] = []
        judge_reuse_cache = dict(judge_reuse_cache or {})
        judge_reuse_stats: Dict[str, Any] = {
            "enabled": bool(judge_reuse_cache),
            "attempted": 0,
            "reused": 0,
            "skipped": 0,
            "skipped_by_reason": {},
            "reused_judge_ids": [],
            "skipped_judge_ids": [],
            "fingerprint_fields": [
                "tx_hash",
                "judge_id",
                "condition_id",
                "question",
                "expected_answer",
                "evidence_refs",
                "default_evidence_refs",
                "allowed_followup_views",
                "allowed_tools",
                "max_followups",
                "depends_on",
                "consumes_state_keys",
                "produces_state_key",
                "state_prompt_role",
                "state_output_schema",
                "judge_model",
                "judge_provider",
                "max_view_chars",
                "max_context_chars",
                "judge_max_tokens",
                "judge_thinking",
                "source_followup_enabled",
                "followup_context_mode",
                "judge_followup_mode",
                "packet_identity_fingerprint",
                "selected_view_names",
                "selected_views_hash",
                "adaptive_evidence_enabled",
                "adaptive_evidence_mode",
            ],
        }
        early_stop_trace: Dict[str, Any] = {
            "enabled": bool(early_stop),
            "policy": early_stop_policy,
            "triggered": False,
            "reason": "",
            "after_judge_id": "",
            "skipped_judge_ids": [],
        }
        evidence_tool_registry = EvidenceToolRegistry(
            packet,
            source_registry=(
                self.judge_model.source_tool_registry
                if self.judge_model.enable_source_followup
                else None
            ),
            default_chain=chain,
            debug_allow_list_views=self.debug_allow_list_views,
        )
        state_store: Dict[str, Any] = {}
        normalized_attack_label = normalize_attack_label(attack_label, default="")
        plan_declares_stateful = _plan_declares_stateful_runtime(rule, plan)
        stateful_requested = (
            bool(iv_stateful_runtime)
            if normalized_attack_label == "insufficient_validation"
            else normalize_access_control_binding_mode(
                access_control_binding_mode
            ) == "stateful"
            if normalized_attack_label == "access_control"
            else normalize_reentrancy_binding_mode(
                reentrancy_binding_mode
            ) != "disabled"
            if normalized_attack_label == "reentrancy"
            else plan_declares_stateful
        )
        stateful_trace: Dict[str, Any] = {
            "enabled": bool(stateful_enabled),
            "requested": bool(stateful_requested),
            "ablation_disabled": bool(disable_stateful_bindings),
            "requested_mode": (
                access_control_binding_mode
                if normalized_attack_label == "access_control"
                else normalize_reentrancy_binding_mode(
                    reentrancy_binding_mode
                )
                if normalized_attack_label == "reentrancy"
                else str(stateful_metadata.get("mode") or "plan_declared")
                if plan_declares_stateful
                else "enabled" if bool(iv_stateful_runtime) else "disabled"
            ),
            "label_gated": normalized_attack_label in {
                "insufficient_validation",
                "access_control",
            },
            "plan_declares_stateful": bool(plan_declares_stateful),
            "mode": str(stateful_metadata.get("mode") or ""),
            "binding_mode": (
                normalize_reentrancy_binding_mode(reentrancy_binding_mode)
                if normalized_attack_label == "reentrancy"
                else ""
            ),
            "core_steps": list(stateful_metadata.get("core_steps") or []),
            "state_keys_produced": [],
            "state_binding_audit": [],
            "warnings": [],
        }

        parallel_trace = {
            **dict(parallel_config),
            "rate_limit_serial_fallback_enabled": bool(
                rate_limit_serial_fallback
            ),
            "rate_limit_serial_fallback_triggered": False,
            "rate_limit_fallback_provider": judge_provider_key,
            "rate_limit_retry_attempts": max(
                0,
                int(rate_limit_retry_attempts or 0),
            ),
            "rate_limit_retry_delay_seconds": max(
                0.0,
                float(rate_limit_retry_delay_seconds or 0.0),
            ),
            "rate_limit_retry_count": 0,
            "rate_limit_retried_judge_ids": [],
            "rate_limit_recovered_judge_ids": [],
        }
        steps_for_serial = list(plan.judge_steps)
        if parallel_config.get("effective"):
            parallel_out = self._run_judge_steps_parallel(
                packet=packet,
                plan=plan,
                tx_hash=tx_hash,
                chain=chain,
                packet_identity_fingerprint=packet_identity_fingerprint,
                judge_reuse_cache=judge_reuse_cache,
                judge_concurrency=int(parallel_config.get("judge_concurrency", 1) or 1),
                structured_judge_logs=bool(structured_judge_logs),
                adaptive_enabled=adaptive_enabled,
                adaptive_mode=adaptive_selected_mode,
                attack_label=attack_label,
                evidence_profile=evidence_profile,
                adaptive_config=adaptive_config,
                rate_limit_serial_fallback=bool(rate_limit_serial_fallback),
                rate_limit_retry_attempts=rate_limit_retry_attempts,
                rate_limit_retry_delay_seconds=rate_limit_retry_delay_seconds,
                judge_provider_key=judge_provider_key,
                disable_stateful_bindings=bool(disable_stateful_bindings),
                judge_followup_mode=normalized_judge_followup_mode,
            )
            judge_results = list(parallel_out.get("judge_results", []) or [])
            judge_step_traces = list(parallel_out.get("judge_step_traces", []) or [])
            judge_values = dict(parallel_out.get("judge_values", {}) or {})
            has_uncertain = bool(parallel_out.get("has_uncertain"))
            errors.extend(list(parallel_out.get("errors", []) or []))
            reused_judge_ids = list(parallel_out.get("reused_judge_ids", []) or [])
            judge_reuse_stats = dict(parallel_out.get("judge_reuse_stats") or judge_reuse_stats)
            parallel_trace = dict(parallel_out.get("parallel_trace") or parallel_trace)
            for result_dict, step_trace in zip(judge_results, judge_step_traces):
                step_policy = _judge_followup_step_audit(
                    judge_followup_policy,
                    str(result_dict.get("id") or step_trace.get("judge_id") or ""),
                )
                step_trace["judge_followup_policy"] = step_policy
                result_dict["judge_followup_policy"] = copy.deepcopy(step_policy)
            steps_for_serial = []
            for result_dict in judge_results:
                print(
                    f"[PacketRuntime] Judge {result_dict.get('id')}: "
                    f"answer={result_dict.get('answer')} expected={result_dict.get('expected_answer')} "
                    f"satisfied={result_dict.get('satisfied')} confidence={result_dict.get('confidence')}"
                )

        for step_index, step in enumerate(steps_for_serial):
            try:
                state_input = (
                    _state_input_for_step(step, state_store)
                    if stateful_enabled
                    else {}
                )
                stateful_step_meta = _step_stateful_runtime_metadata(
                    step,
                    stateful_enabled=stateful_enabled,
                    bindings_disabled=bool(disable_stateful_bindings),
                    binding_mode=(
                        reentrancy_binding_mode
                        if normalized_attack_label == "reentrancy"
                        else ""
                    ),
                )
                if (
                    str(step.state_prompt_role or "") == "re_reentry_anchor"
                    and stateful_step_meta.get("enabled")
                ):
                    catalog = packet.get("reentrancy_candidate_catalog_view")
                    catalog_rows = (
                        list(catalog.get("rows", []) or [])
                        if isinstance(catalog, dict)
                        else list(catalog or [])
                        if isinstance(catalog, list)
                        else []
                    )
                    stateful_step_meta["allowed_candidate_ids"] = _unique_strings(
                        row.get("candidate_id")
                        for row in catalog_rows
                        if isinstance(row, dict)
                    )
                binding_audit = _state_binding_audit_for_step(
                    step,
                    state_store,
                    state_input=state_input,
                    stateful_enabled=stateful_enabled,
                )
                if binding_audit:
                    stateful_trace.setdefault("state_binding_audit", []).append(binding_audit)
                    missing_keys = list(binding_audit.get("missing_state_keys") or [])
                    if missing_keys:
                        warning = (
                            f"step {step.id} consumes missing state key(s): "
                            + ", ".join(missing_keys)
                        )
                        stateful_trace.setdefault("warnings", []).append(warning)
                        print(f"[PacketRuntime][Stateful WARNING] {warning}")
                reused = self._reuse_judge_step(
                    step,
                    judge_reuse_cache,
                    tx_hash=tx_hash,
                    packet=packet,
                    packet_identity_fingerprint=packet_identity_fingerprint,
                    reuse_stats=judge_reuse_stats,
                    adaptive_enabled=adaptive_enabled,
                    adaptive_mode=adaptive_selected_mode,
                    attack_label=attack_label,
                    evidence_profile=evidence_profile,
                    adaptive_config=adaptive_config,
                    judge_followup_mode=normalized_judge_followup_mode,
                    state_input_context=state_input,
                    evidence_tool_registry=evidence_tool_registry,
                )
                if reused:
                    result_dict, step_trace = reused
                    reused_judge_ids.append(step.id)
                    judge_reuse_stats["reused"] += 1
                    judge_reuse_stats["reused_judge_ids"].append(step.id)
                    print(
                        f"[PacketRuntime] Judge {step.id}: reused cached result "
                        f"fingerprint={step_trace.get('reuse_fingerprint', '')[:12]}"
                    )
                else:
                    def run_serial_judge_step():
                        return self._run_judge_step(
                            packet=packet,
                            step=step,
                            evidence_tool_registry=evidence_tool_registry,
                            tx_hash=tx_hash,
                            chain=chain,
                            adaptive_enabled=adaptive_enabled,
                            adaptive_mode=adaptive_selected_mode,
                            attack_label=attack_label,
                            evidence_profile=evidence_profile,
                            adaptive_config=adaptive_config,
                            state_input_context=state_input,
                            stateful_runtime=stateful_step_meta,
                            judge_followup_mode=normalized_judge_followup_mode,
                        )

                    (
                        (result, step_trace),
                        rate_limit_retry,
                    ) = _call_with_rate_limit_retry(
                        run_serial_judge_step,
                        enabled=bool(rate_limit_serial_fallback),
                        provider=judge_provider_key,
                        max_retries=rate_limit_retry_attempts,
                        initial_delay_seconds=rate_limit_retry_delay_seconds,
                    )
                    if rate_limit_retry.get("triggered"):
                        step_trace["judge_retry"] = rate_limit_retry
                        if rate_limit_retry.get("rate_limit_triggered"):
                            step_trace["rate_limit_retry"] = rate_limit_retry
                            parallel_trace["rate_limit_serial_fallback_triggered"] = True
                            parallel_trace["rate_limit_fallback_provider"] = judge_provider_key
                            parallel_trace["effective_concurrency_after_rate_limit"] = 1
                            parallel_trace["rate_limit_retry_count"] = int(
                                parallel_trace.get("rate_limit_retry_count", 0) or 0
                            ) + int(rate_limit_retry.get("retry_count", 0) or 0)
                            parallel_trace.setdefault(
                                "rate_limit_retried_judge_ids", []
                            ).append(step.id)
                            if rate_limit_retry.get("recovered"):
                                parallel_trace.setdefault(
                                    "rate_limit_recovered_judge_ids",
                                    [],
                                ).append(step.id)
                        if rate_limit_retry.get("transport_retry_triggered"):
                            step_trace["transport_retry"] = rate_limit_retry
                            parallel_trace["transport_retry_count"] = int(
                                parallel_trace.get("transport_retry_count", 0) or 0
                            ) + int(rate_limit_retry.get("retry_count", 0) or 0)
                            parallel_trace.setdefault(
                                "transport_retried_judge_ids", []
                            ).append(step.id)
                            if rate_limit_retry.get("recovered"):
                                parallel_trace.setdefault(
                                    "transport_recovered_judge_ids",
                                    [],
                                ).append(step.id)
                    result_dict = result.to_dict()
                    result_dict["condition_id"] = step.condition_id or step.id
                    result_dict["tool_calls"] = list(step_trace.get("tool_calls", []))
                    result_dict["judge_rounds"] = list(step_trace.get("judge_calls", []))
                    result_dict["judge_output_exhaustions"] = list(
                        step_trace.get("judge_output_exhaustions", [])
                    )
                    result_dict["source_decisiveness_guards"] = list(
                        step_trace.get("source_decisiveness_guards", [])
                    )
                    result_dict["view_render_metadata"] = list(step_trace.get("view_render_metadata", []))
                    result_dict["selected_view_names"] = list(
                        step_trace.get("selected_view_names", [])
                    )
                    result_dict["selected_views_hash"] = str(
                        step_trace.get("selected_views_hash", "")
                    )
                    result_dict["all_missing_evidence_debug"] = list(
                        step_trace.get("all_missing_evidence_debug", [])
                    )
                    result_dict["ignored_tool_requests"] = list(
                        step_trace.get("ignored_tool_requests", [])
                    )
                    result_dict["invalid_evidence_ids"] = list(
                        step_trace.get("invalid_evidence_ids", [])
                    )
                    result_dict["evidence_id_validations"] = list(
                        step_trace.get("evidence_id_validations", [])
                    )
                _attach_stateful_result_fields(
                    result_dict,
                    step_trace,
                    step,
                    state_input=state_input,
                    stateful_runtime=stateful_step_meta,
                )
                step_trace["judge_followup_policy"] = _judge_followup_step_audit(
                    judge_followup_policy,
                    step.id,
                )
                result_dict["judge_followup_policy"] = copy.deepcopy(
                    step_trace["judge_followup_policy"]
                )
                _update_state_store_from_result(
                    state_store,
                    step,
                    result_dict,
                )
                stateful_trace["state_keys_produced"] = sorted(state_store.keys())
                judge_results.append(result_dict)
                judge_step_traces.append(step_trace)

                answer = result_dict.get("answer")
                judge_values[step.id] = answer is True
                if isinstance(answer, str):
                    has_uncertain = True

                print(
                    f"[PacketRuntime] Judge {step.id}: "
                    f"answer={result_dict.get('answer')} expected={result_dict.get('expected_answer')} "
                    f"satisfied={result_dict.get('satisfied')} confidence={result_dict.get('confidence')}"
                )
                stop_decision = _early_stop_decision(
                    plan=plan,
                    judge_results=judge_results,
                    judge_values=judge_values,
                    enabled=bool(early_stop),
                    policy=early_stop_policy,
                )
                if stop_decision.get("triggered"):
                    remaining_steps = list(plan.judge_steps[step_index + 1 :])
                    skipped_ids = [remaining.id for remaining in remaining_steps]
                    early_stop_trace.update({
                        **stop_decision,
                        "after_judge_id": step.id,
                        "skipped_judge_ids": skipped_ids,
                    })
                    for remaining in remaining_steps:
                        judge_values[remaining.id] = False
                        skipped_result = _skipped_judge_result(remaining, stop_decision)
                        skipped_trace = _skipped_judge_trace(remaining, stop_decision)
                        judge_results.append(skipped_result)
                        judge_step_traces.append(skipped_trace)
                    print(
                        "[PacketRuntime] Early stop triggered: "
                        f"reason={stop_decision.get('reason')} skipped={skipped_ids}"
                    )
                    break
            except Exception as exc:
                judge_values[step.id] = False
                has_uncertain = True
                error_result = _serial_error_judge_result(
                    step,
                    exc,
                    state_input=(
                        _state_input_for_step(step, state_store)
                        if stateful_enabled
                        else {}
                    ),
                    stateful_runtime=_step_stateful_runtime_metadata(
                        step,
                        stateful_enabled=stateful_enabled,
                        bindings_disabled=bool(disable_stateful_bindings),
                        binding_mode=(
                            reentrancy_binding_mode
                            if normalized_attack_label == "reentrancy"
                            else ""
                        ),
                    ),
                )
                error_trace = _serial_error_step_trace(
                    step,
                    exc,
                    state_input=dict(error_result.get("state_input") or {}),
                    stateful_runtime=dict(error_result.get("stateful_runtime") or {}),
                )
                judge_results.append(error_result)
                judge_step_traces.append(error_trace)
                errors.append(
                    {
                        "stage": "judge",
                        "step_id": step.id,
                        "error": repr(exc),
                        "error_class": _judge_exception_class(exc),
                        "do_not_train": True,
                    }
                )
                print(f"[PacketRuntime] Judge {step.id} raised: {exc!r}")

        # 3. Evaluate emit_logic
        emit_logic_error = None
        try:
            should_emit = safe_eval_bool_expr(plan.emit_logic, judge_values)
            print(
                f"[PacketRuntime] emit_logic '{plan.emit_logic}' => "
                f"{should_emit} values={judge_values}"
            )
        except Exception as exc:
            should_emit = False
            emit_logic_error = repr(exc)
            has_uncertain = True
            errors.append(
                {
                    "stage": "emit_logic",
                    "error": repr(exc),
                    "values": judge_values,
                }
            )
            print(f"[PacketRuntime] emit_logic failed: {exc!r}")

        # 4. Retry the one verdict-critical condition that lacks evidence.
        near_miss_escalations: List[Dict[str, Any]] = []
        near_misses: List[Dict[str, Any]] = []
        if (
            normalized_judge_followup_mode == "plan"
            and not early_stop_trace.get("triggered")
        ):
            near_misses = _find_near_miss_escalations(
                plan=plan,
                judge_results=judge_results,
                should_emit=should_emit,
                emit_logic_error=emit_logic_error,
                attack_label=attack_label,
            )
        if near_misses:
            step_by_id = {step.id: step for step in plan.judge_steps}
        attempted_near_miss_blockers: set[str] = set()
        for near_miss in near_misses:
            target_step = step_by_id.get(str(near_miss.get("judge_id") or ""))
            target_index = int(near_miss.get("judge_index", -1))
            blocker_key = str(
                near_miss.get("condition_id")
                or near_miss.get("judge_id")
                or target_index
            )
            if blocker_key in attempted_near_miss_blockers:
                near_miss_escalations.append({
                    "attempted": False,
                    "judge_id": near_miss.get("judge_id"),
                    "condition_id": near_miss.get("condition_id"),
                    "reason": "near_miss_blocker_already_attempted",
                })
                continue
            attempted_near_miss_blockers.add(blocker_key)
            if target_step is not None and 0 <= target_index < len(judge_results):
                print(
                    "[PacketRuntime] Near-miss escalation: "
                    f"single blocker {target_step.id} "
                    f"type={near_miss.get('blocker_type')} "
                    f"policy={near_miss.get('near_miss_policy', '')}"
                )
                try:
                    old_trace = (
                        judge_step_traces[target_index]
                        if target_index < len(judge_step_traces)
                        else {}
                    )
                    escalated_result, escalated_trace = self._run_near_miss_escalation(
                        packet=packet,
                        step=target_step,
                        near_miss=near_miss,
                        evidence_tool_registry=evidence_tool_registry,
                        tx_hash=tx_hash,
                        chain=chain,
                        attack_label=attack_label,
                        prior_step_trace=old_trace,
                        state_input_context=dict(old_trace.get("state_input", {}) or {}),
                        stateful_runtime=dict(old_trace.get("stateful_runtime", {}) or {}),
                    )
                    if escalated_result is None:
                        skip_meta = dict(
                            escalated_trace.get("near_miss_escalation", {}) or {}
                        )
                        near_miss_escalations.append({
                            "attempted": False,
                            "judge_id": target_step.id,
                            "condition_id": target_step.condition_id or target_step.id,
                            "blocker_type": near_miss.get("blocker_type"),
                            "near_miss_policy": near_miss.get("near_miss_policy", ""),
                            "reason": skip_meta.get(
                                "skip_reason", "near_miss_rerun_skipped"
                            ),
                            "new_observation_count": int(
                                skip_meta.get("new_observation_count", 0) or 0
                            ),
                        })
                        print(
                            f"[PacketRuntime] Near-miss escalation {target_step.id} "
                            f"skipped: {skip_meta.get('skip_reason', 'no_new_observation')}"
                        )
                        continue
                    merged_trace = _merge_escalated_step_trace(
                        old_trace,
                        escalated_trace,
                        near_miss=near_miss,
                    )
                    escalated_dict = _materialize_escalated_judge_result(
                        escalated_result,
                        target_step,
                        merged_trace,
                        state_input=dict(old_trace.get("state_input", {}) or {}),
                        stateful_runtime=dict(old_trace.get("stateful_runtime", {}) or {}),
                    )
                    escalated_dict = _enforce_near_miss_upgrade_guard(
                        original_result=dict(judge_results[target_index] or {}),
                        escalated_result=escalated_dict,
                        escalated_trace=escalated_trace,
                    )
                    merged_trace["final_answer"] = escalated_dict.get("answer")
                    merged_trace["near_miss_upgrade_guard"] = dict(
                        escalated_dict.get("near_miss_upgrade_guard", {}) or {}
                    )
                    escalated_dict["near_miss_escalated"] = True
                    escalated_dict["near_miss_pre_escalation"] = dict(near_miss)

                    previous_state_output = dict(
                        (judge_results[target_index] or {}).get("state_output") or {}
                    )
                    judge_results[target_index] = escalated_dict
                    judge_step_traces[target_index] = merged_trace
                    judge_values[target_step.id] = escalated_dict.get("answer") is True
                    produced_state_key = str(
                        getattr(target_step, "produces_state_key", "") or ""
                    )
                    producer_state_changed = bool(
                        produced_state_key
                        and previous_state_output.get(produced_state_key)
                        != dict(escalated_dict.get("state_output") or {}).get(
                            produced_state_key
                        )
                    )
                    dependency_retry = {
                        "triggered": False,
                        "producer_condition_id": (
                            target_step.condition_id or target_step.id
                        ),
                        "producer_state_key": produced_state_key,
                        "producer_state_changed": producer_state_changed,
                        "rerun_condition_ids": [],
                    }
                    if producer_state_changed:
                        dependency_retry = self._rerun_stateful_dependency_closure(
                            packet=packet,
                            plan=plan,
                            source_step=target_step,
                            judge_results=judge_results,
                            judge_step_traces=judge_step_traces,
                            judge_values=judge_values,
                            evidence_tool_registry=evidence_tool_registry,
                            tx_hash=tx_hash,
                            chain=chain,
                            attack_label=attack_label,
                            adaptive_enabled=adaptive_enabled,
                            adaptive_mode=adaptive_selected_mode,
                            evidence_profile=evidence_profile,
                            adaptive_config=adaptive_config,
                            stateful_enabled=stateful_enabled,
                            disable_stateful_bindings=bool(
                                disable_stateful_bindings
                            ),
                            reentrancy_binding_mode=reentrancy_binding_mode,
                            judge_followup_mode=normalized_judge_followup_mode,
                            judge_followup_policy=judge_followup_policy,
                            rate_limit_serial_fallback=bool(
                                rate_limit_serial_fallback
                            ),
                            rate_limit_retry_attempts=rate_limit_retry_attempts,
                            rate_limit_retry_delay_seconds=(
                                rate_limit_retry_delay_seconds
                            ),
                            errors=errors,
                            stateful_trace=stateful_trace,
                        )
                    has_uncertain = any(
                        isinstance(result.get("answer"), str)
                        for result in judge_results
                    )
                    near_miss_escalations.append({
                        "judge_id": target_step.id,
                        "condition_id": target_step.condition_id or target_step.id,
                        "blocker_type": near_miss.get("blocker_type"),
                        "near_miss_policy": near_miss.get("near_miss_policy", ""),
                        "cluster_id": near_miss.get("cluster_id", ""),
                        "cluster_members": list(near_miss.get("cluster_members", []) or []),
                        "before_answer": near_miss.get("answer"),
                        "before_confidence": near_miss.get("confidence"),
                        "after_answer": escalated_dict.get("answer"),
                        "after_confidence": escalated_dict.get("confidence"),
                        "changed_answer": near_miss.get("answer") != escalated_dict.get("answer"),
                        "tool_call_count": len(merged_trace.get("tool_calls", []))
                        - len(old_trace.get("tool_calls", []) or []),
                        "forced_source_followup": bool(
                            near_miss.get("force_source_followup")
                        ),
                        "source_probe_evidence_ids": list(
                            near_miss.get("source_probe_evidence_ids", []) or []
                        ),
                        "new_observation_count": int(
                            (
                                escalated_trace.get("near_miss_escalation", {})
                                or {}
                            ).get("new_observation_count", 0)
                            or 0
                        ),
                        "target_condition_evidence_count": int(
                            (
                                escalated_trace.get("near_miss_escalation", {})
                                or {}
                            ).get("target_condition_evidence_count", 0)
                            or 0
                        ),
                        "upgrade_guard": dict(
                            escalated_dict.get("near_miss_upgrade_guard", {}) or {}
                        ),
                        "dependency_retry": dependency_retry,
                    })

                    try:
                        should_emit = safe_eval_bool_expr(plan.emit_logic, judge_values)
                        print(
                            f"[PacketRuntime] emit_logic after near-miss escalation "
                            f"'{plan.emit_logic}' => {should_emit} values={judge_values}"
                        )
                    except Exception as exc:
                        should_emit = False
                        emit_logic_error = repr(exc)
                        has_uncertain = True
                        errors.append(
                            {
                                "stage": "near_miss_emit_logic",
                                "error": repr(exc),
                                "values": judge_values,
                            }
                        )
                        print(
                            f"[PacketRuntime] emit_logic after near-miss escalation failed: {exc!r}"
                        )
                except Exception as exc:
                    errors.append({
                        "stage": "near_miss_escalation",
                        "step_id": target_step.id,
                        "error": repr(exc),
                    })
                    near_miss_escalations.append({
                        "judge_id": target_step.id,
                        "condition_id": target_step.condition_id or target_step.id,
                        "blocker_type": near_miss.get("blocker_type"),
                        "error": repr(exc),
                    })
                    print(
                        f"[PacketRuntime] Near-miss escalation {target_step.id} raised: {exc!r}"
                    )
            else:
                near_miss_escalations.append({
                    "attempted": False,
                    "reason": "target_step_not_found",
                    "near_miss": near_miss,
                })

        # 5. Determine verdict
        verdict, verdict_aggregation = _aggregate_verdict(
            plan=plan,
            judge_results=judge_results,
            judge_values=judge_values,
            should_emit=should_emit,
            has_uncertain=has_uncertain,
            emit_logic_error=emit_logic_error,
        )
        verdict_aggregation["near_miss_escalations"] = near_miss_escalations
        verdict_aggregation["early_stop"] = dict(early_stop_trace)
        stateful_trace["candidate_emit"] = copy.deepcopy(
            verdict_aggregation.get("candidate_emit") or {}
        )
        if early_stop_trace.get("triggered") and verdict == "benign":
            verdict_aggregation["previous_decision_source"] = (
                verdict_aggregation.get("decision_source", "")
            )
            verdict_aggregation["reason"] = "early_stopped_strong_negative"
            verdict_aggregation["decision_source"] = "early_stop_policy"
            verdict_aggregation["early_stop_reason"] = early_stop_trace.get("reason", "")
        elif dynamic_aggregation:
            initial_dynamic_verdict = verdict
            verdict, verdict_aggregation = self._apply_dynamic_aggregation(
                rule=rule,
                plan=plan,
                judge_results=judge_results,
                judge_values=judge_values,
                previous_verdict=verdict,
                verdict_aggregation=verdict_aggregation,
                attack_label=attack_label,
                tx_hash=tx_hash,
                chain=chain,
                near_miss_escalations=near_miss_escalations,
                aggregation_round=1,
                allow_followup_requests=True,
            )
            first_dynamic_record = copy.deepcopy(
                verdict_aggregation.get("dynamic_aggregation", {}) or {}
            )
            followup_requests = list(
                ((first_dynamic_record.get("decision") or {}).get("followup_requests") or [])
            )
            if followup_requests:
                followup_round = self._run_dynamic_aggregation_followup_round(
                    packet=packet,
                    plan=plan,
                    judge_results=judge_results,
                    judge_step_traces=judge_step_traces,
                    judge_values=judge_values,
                    followup_requests=followup_requests,
                    evidence_tool_registry=evidence_tool_registry,
                    tx_hash=tx_hash,
                    chain=chain,
                    attack_label=attack_label,
                    near_miss_escalations=near_miss_escalations,
                )
                followup_round["should_emit_before"] = bool(should_emit)
                followup_round["provisional_aggregator_verdict"] = verdict
                if int(followup_round.get("accepted_count", 0) or 0) <= 0:
                    followup_round["skipped_final_aggregation"] = True
                    followup_round["skip_reason"] = (
                        "all_aggregator_followup_requests_rejected"
                    )
                    decision = dict(first_dynamic_record.get("decision", {}) or {})
                    decision["followup_requests"] = []
                    decision["verdict"] = initial_dynamic_verdict
                    decision["override"] = False
                    decision["reason"] = (
                        "Aggregator follow-up requests were rejected by Runtime "
                        "policy; preserving the pre-aggregator hard verdict."
                    )
                    decision["warnings"] = _unique_strings(
                        list(decision.get("warnings", []) or [])
                        + ["aggregator_followup_rejected_preserved_hard_verdict"]
                    )
                    first_dynamic_record["decision"] = decision
                    first_dynamic_record["followup_round"] = followup_round
                    first_dynamic_record["previous_verdict"] = initial_dynamic_verdict
                    first_dynamic_record["final_verdict"] = initial_dynamic_verdict
                    first_dynamic_record["applied"] = False
                    verdict = initial_dynamic_verdict
                    verdict_aggregation["verdict"] = verdict
                    verdict_aggregation["reason"] = (
                        "dynamic_followup_rejected_preserved_hard_verdict"
                    )
                    verdict_aggregation["aggregator_override"] = {
                        "attempted": True,
                        "applied": False,
                        "reason": "all_followup_requests_rejected",
                    }
                    verdict_aggregation["dynamic_aggregation"] = first_dynamic_record
                else:
                    errors.extend(list(followup_round.get("errors", []) or []))
                    has_uncertain = any(
                        isinstance(result.get("answer"), str)
                        for result in judge_results
                    )
                    followup_emit_error = None
                    try:
                        should_emit = safe_eval_bool_expr(plan.emit_logic, judge_values)
                        print(
                            "[PacketRuntime] emit_logic after dynamic aggregation follow-up "
                            f"'{plan.emit_logic}' => {should_emit} values={judge_values}"
                        )
                    except Exception as exc:
                        should_emit = False
                        emit_logic_error = repr(exc)
                        followup_emit_error = emit_logic_error
                        has_uncertain = True
                        errors.append({
                            "stage": "dynamic_aggregation_followup_emit_logic",
                            "error": emit_logic_error,
                            "values": dict(judge_values),
                        })
                    followup_round["emit_logic"] = plan.emit_logic
                    followup_round["should_emit_after"] = bool(should_emit)
                    followup_round["emit_logic_error"] = followup_emit_error

                    hard_verdict, hard_aggregation = _aggregate_verdict(
                        plan=plan,
                        judge_results=judge_results,
                        judge_values=judge_values,
                        should_emit=should_emit,
                        has_uncertain=has_uncertain,
                        emit_logic_error=emit_logic_error,
                    )
                    hard_aggregation["near_miss_escalations"] = near_miss_escalations
                    hard_aggregation["early_stop"] = dict(early_stop_trace)
                    followup_round["hard_verdict_after"] = hard_verdict
                    verdict, verdict_aggregation = self._apply_dynamic_aggregation(
                        rule=rule,
                        plan=plan,
                        judge_results=judge_results,
                        judge_values=judge_values,
                        previous_verdict=hard_verdict,
                        verdict_aggregation=hard_aggregation,
                        attack_label=attack_label,
                        tx_hash=tx_hash,
                        chain=chain,
                        near_miss_escalations=near_miss_escalations,
                        aggregation_round=2,
                        allow_followup_requests=False,
                        force_agent_call=True,
                        followup_context=followup_round,
                    )
                    final_dynamic_record = copy.deepcopy(
                        verdict_aggregation.get("dynamic_aggregation", {}) or {}
                    )
                    final_dynamic_record["passes"] = [
                        first_dynamic_record,
                        copy.deepcopy(final_dynamic_record),
                    ]
                    final_dynamic_record["followup_round"] = followup_round
                    final_dynamic_record["previous_verdict"] = initial_dynamic_verdict
                    final_dynamic_record["final_verdict"] = verdict
                    final_dynamic_record["applied"] = verdict != initial_dynamic_verdict
                    verdict_aggregation["dynamic_aggregation"] = final_dynamic_record
        print(
            f"[PacketRuntime] verdict aggregation: verdict={verdict} "
            f"reason={verdict_aggregation.get('reason')}"
        )

        # 6. Collect grouped and flat evidence IDs
        evidence_groups = _group_judge_evidence(judge_results)
        supporting_evidence_ids = evidence_groups["supporting_evidence_union"]

        elapsed = round(time.time() - started_at, 2)

        result = {
            "tx_hash": tx_hash,
            "rule": rule.to_dict(),
            "plan": plan.to_dict(),
            "packet_source": packet_source,
            "judge_results": judge_results,
            "emit_logic": plan.emit_logic,
            "verdict": verdict,
            "verdict_reason": verdict_aggregation.get("reason", ""),
            "verdict_aggregation": verdict_aggregation,
            "supporting_evidence_ids": supporting_evidence_ids,
            "evidence_groups": evidence_groups,
            # Private hand-off to run_inference. The public result stores a
            # replayable snapshot without duplicating views.
            "_packet_snapshot": packet,
            "_effective_plan": plan.to_dict(),
            "runtime_trace": {
                "packet_loaded_from_cache": packet_loaded_from_cache,
                "packet_frozen": bool(freeze_packet or packet_override is not None),
                "packet_replayed_from_baseline": bool(packet_override is not None),
                "packet_required_views": required_views,
                "packet_required_views_missing": packet_required_views_missing,
                "packet_build_config": dict(packet.get("build_config", {}) or {}),
                "packet_view_cost_summary": dict(packet.get("view_cost_summary", {}) or {}),
                "view_manifest_version": (
                    (packet.get("build_config", {}) or {}).get("view_manifest_version", "")
                ),
                "evidence_profile": evidence_profile.to_dict(),
                "adaptive_evidence": _adaptive_evidence_summary(
                    judge_step_traces,
                    enabled=adaptive_enabled,
                    requested_mode=normalized_adaptive_mode,
                    selected_mode=adaptive_selected_mode,
                    reason=adaptive_mode_reason,
                    debug=bool(adaptive_evidence_debug),
                ),
                "packet_identity_fingerprint": packet_identity_fingerprint,
                "packet_fingerprint": packet_fingerprint,
                "judge_call_count": len(judge_results),
                "executed_judge_call_count": len([
                    result for result in judge_results
                    if not result.get("skipped_by_early_stop")
                ]),
                "skipped_judge_step_count": len(early_stop_trace.get("skipped_judge_ids", []) or []),
                "judge_step_traces": judge_step_traces,
                "tool_calls": [
                    call
                    for trace in judge_step_traces
                    for call in trace.get("tool_calls", [])
                ],
                "errors": errors,
                "elapsed_seconds": elapsed,
                "has_uncertain": has_uncertain,
                "verdict_aggregation": verdict_aggregation,
                "uncertain_judge_ids": [
                    jr.get("id")
                    for jr in judge_results
                    if isinstance(jr.get("answer"), str)
                ],
                "source_tool_call_count": sum(
                    1
                    for trace in judge_step_traces
                    for call in trace.get("tool_calls", [])
                    if call.get("tool") == "read_function_chunk"
                ),
                "near_miss_escalation_count": len(near_miss_escalations),
                "near_miss_escalations": near_miss_escalations,
                "dynamic_aggregation": verdict_aggregation.get(
                    "dynamic_aggregation",
                    {"enabled": bool(dynamic_aggregation), "applied": False},
                ),
                "judge_followup_policy": judge_followup_policy,
                "early_stop": early_stop_trace,
                "parallel_judge": parallel_trace,
                "stateful_runtime": stateful_trace,
                "followup_context": {
                    "mode": str(
                        getattr(
                            self.judge_model,
                            "followup_context_mode",
                            "unified",
                        )
                        or "unified"
                    ),
                    "budget_chars": int(
                        self.judge_model.max_context_chars or 0
                    ),
                    "judge_max_tokens": int(
                        getattr(self.judge_model.llm, "max_tokens", 0) or 0
                    ),
                },
                "reused_judge_call_count": len(reused_judge_ids),
                "reused_judge_ids": reused_judge_ids,
                "judge_reuse": {
                    **judge_reuse_stats,
                    "skipped": int(judge_reuse_stats.get("attempted", 0))
                    - int(judge_reuse_stats.get("reused", 0)),
                },
            },
        }

        print(
            f"[PacketRuntime] Finished tx={tx_hash}: verdict={verdict} "
            f"evidence_ids={len(supporting_evidence_ids)} elapsed={elapsed}s"
        )

        return result

    def _run_judge_steps_parallel(
        self,
        *,
        packet: Dict[str, Any],
        plan: EvidencePlan,
        tx_hash: str,
        chain: str,
        packet_identity_fingerprint: str,
        judge_reuse_cache: Dict[str, Dict[str, Any]],
        judge_concurrency: int,
        structured_judge_logs: bool,
        adaptive_enabled: bool,
        adaptive_mode: str,
        attack_label: str,
        evidence_profile: EvidenceProfile,
        adaptive_config: AdaptiveEvidenceConfig,
        rate_limit_serial_fallback: bool = True,
        rate_limit_retry_attempts: int = 3,
        rate_limit_retry_delay_seconds: float = 2.0,
        judge_provider_key: str = "unknown",
        disable_stateful_bindings: bool = False,
        judge_followup_mode: str = "plan",
    ) -> Dict[str, Any]:
        submitted_at = time.time()
        steps = list(plan.judge_steps or [])
        source_tool_lock = threading.Lock()
        submitted_count = len(steps)
        parallel_trace: Dict[str, Any] = {
            "enabled": True,
            "requested": True,
            "effective": True,
            "judge_concurrency": int(judge_concurrency or 1),
            "disabled_reason": "",
            "submitted_step_count": submitted_count,
            "completed_step_count": 0,
            "failed_step_count": 0,
            "elapsed_parallel_seconds": 0.0,
            "log_format": "jsonl" if structured_judge_logs else "disabled",
            "structured_log_count": 0,
            "max_inflight_observed": min(submitted_count, int(judge_concurrency or 1)),
            "completed_order": [],
            "result_order": [step.id for step in steps],
            "step_timings": {},
            "rate_limit_serial_fallback_enabled": bool(
                rate_limit_serial_fallback
            ),
            "rate_limit_serial_fallback_triggered": False,
            "rate_limit_fallback_provider": str(judge_provider_key or "unknown"),
            "rate_limit_retry_attempts": max(
                0,
                int(rate_limit_retry_attempts or 0),
            ),
            "rate_limit_retry_delay_seconds": max(
                0.0,
                float(rate_limit_retry_delay_seconds or 0.0),
            ),
            "rate_limit_retry_count": 0,
            "rate_limit_retried_judge_ids": [],
            "rate_limit_recovered_judge_ids": [],
            "transport_retry_count": 0,
            "transport_retried_judge_ids": [],
            "transport_recovered_judge_ids": [],
            "effective_concurrency_after_rate_limit": int(
                judge_concurrency or 1
            ),
        }
        judge_reuse_stats: Dict[str, Any] = {
            "enabled": bool(judge_reuse_cache),
            "attempted": 0,
            "reused": 0,
            "skipped": 0,
            "skipped_by_reason": {},
            "reused_judge_ids": [],
            "skipped_judge_ids": [],
            "fingerprint_fields": [
                "tx_hash",
                "judge_id",
                "condition_id",
                "question",
                "expected_answer",
                "evidence_refs",
                "default_evidence_refs",
                "allowed_followup_views",
                "allowed_tools",
                "max_followups",
                "depends_on",
                "consumes_state_keys",
                "produces_state_key",
                "state_prompt_role",
                "state_output_schema",
                "judge_model",
                "judge_provider",
                "max_view_chars",
                "max_context_chars",
                "judge_max_tokens",
                "judge_thinking",
                "source_followup_enabled",
                "followup_context_mode",
                "judge_followup_mode",
                "packet_identity_fingerprint",
                "selected_view_names",
                "selected_views_hash",
                "adaptive_evidence_enabled",
                "adaptive_evidence_mode",
            ],
        }

        def run_one(
            step_index: int,
            step: JudgeStep,
            *,
            execution_mode: str = "parallel",
            retry_attempt: int = 0,
        ) -> Dict[str, Any]:
            logs: List[Dict[str, Any]] = []
            step_started = time.time()

            def log(phase: str, message: str = "", level: str = "INFO", **fields) -> None:
                event = emit_judge_log(
                    tx_hash=tx_hash,
                    judge_id=step.id,
                    condition_id=step.condition_id or step.id,
                    phase=phase,
                    message=message,
                    level=level,
                    **fields,
                )
                logs.append(event)
                if structured_judge_logs:
                    _print_structured_judge_log(event)

            local_reuse_stats: Dict[str, Any] = {
                "enabled": bool(judge_reuse_cache),
                "attempted": 0,
                "reused": 0,
                "skipped": 0,
                "skipped_by_reason": {},
                "reused_judge_ids": [],
                "skipped_judge_ids": [],
            }
            log(
                "judge_submit",
                step_index=step_index,
                execution_mode=execution_mode,
                retry_attempt=retry_attempt,
            )
            log(
                "judge_start",
                step_index=step_index,
                execution_mode=execution_mode,
                retry_attempt=retry_attempt,
            )
            try:
                reused = None
                if judge_reuse_cache:
                    log("reuse_attempt", cache_entries=len(judge_reuse_cache))
                    reused = self._reuse_judge_step(
                        step,
                        judge_reuse_cache,
                        tx_hash=tx_hash,
                        packet=packet,
                        packet_identity_fingerprint=packet_identity_fingerprint,
                        reuse_stats=local_reuse_stats,
                        adaptive_enabled=adaptive_enabled,
                        adaptive_mode=adaptive_mode,
                        attack_label=attack_label,
                        evidence_profile=evidence_profile,
                        adaptive_config=adaptive_config,
                        judge_followup_mode=judge_followup_mode,
                    )
                if reused:
                    result_dict, step_trace = reused
                    local_reuse_stats["reused"] = int(local_reuse_stats.get("reused", 0) or 0) + 1
                    local_reuse_stats.setdefault("reused_judge_ids", []).append(step.id)
                    log("reuse_hit", fingerprint=str(step_trace.get("reuse_fingerprint", ""))[:12])
                else:
                    if judge_reuse_cache:
                        log(
                            "reuse_miss",
                            skipped_by_reason=dict(local_reuse_stats.get("skipped_by_reason") or {}),
                        )
                    worker_judge = self.judge_model.clone_for_worker()
                    source_registry = (
                        worker_judge.source_tool_registry
                        if worker_judge.enable_source_followup
                        else None
                    )
                    if source_registry is not None:
                        source_registry = _LockedSourceRegistry(source_registry, source_tool_lock)
                    evidence_tool_registry = EvidenceToolRegistry(
                        packet,
                        source_registry=source_registry,
                        default_chain=chain,
                        debug_allow_list_views=self.debug_allow_list_views,
                    )
                    log("judge_llm_start")
                    result, step_trace = self._run_judge_step(
                        packet=packet,
                        step=step,
                        evidence_tool_registry=evidence_tool_registry,
                        tx_hash=tx_hash,
                        chain=chain,
                        adaptive_enabled=adaptive_enabled,
                        adaptive_mode=adaptive_mode,
                        attack_label=attack_label,
                        evidence_profile=evidence_profile,
                        adaptive_config=adaptive_config,
                        judge_model=worker_judge,
                        judge_followup_mode=judge_followup_mode,
                    )
                    tool_calls = list(step_trace.get("tool_calls", []) or [])
                    if tool_calls:
                        log(
                            "followup_start",
                            tool_call_count=len(tool_calls),
                            judge_round_count=len(
                                list(step_trace.get("judge_calls", []) or [])
                            ),
                        )
                    for call in tool_calls:
                        log(
                            "tool_call_done",
                            tool=call.get("tool", ""),
                            round=call.get("round", 0),
                            tool_status=call.get("tool_status", ""),
                            returned_evidence_ids=list(
                                call.get("returned_evidence_ids", []) or []
                            ),
                        )
                    if tool_calls:
                        log(
                            "followup_done",
                            tool_call_count=len(tool_calls),
                            judge_round_count=len(
                                list(step_trace.get("judge_calls", []) or [])
                            ),
                        )
                    log("judge_llm_done")
                    result_dict = _judge_result_dict_from_trace(result, step, step_trace)

                if disable_stateful_bindings:
                    _clear_stateful_binding_result_fields(
                        result_dict,
                        step_trace,
                        stateful_runtime={
                            "enabled": False,
                            "bindings_disabled": True,
                            "ablation": "disable_stateful_bindings",
                        },
                    )

                duration = round(time.time() - step_started, 4)
                log(
                    "judge_done",
                    answer=result_dict.get("answer"),
                    confidence=result_dict.get("confidence"),
                    satisfied=result_dict.get("satisfied"),
                    duration_seconds=duration,
                )
                step_trace.setdefault("structured_logs", []).extend(logs)
                step_trace["parallel_judge"] = {
                    "enabled": True,
                    "step_index": step_index,
                    "execution_mode": execution_mode,
                    "retry_attempt": retry_attempt,
                    "degraded_to_serial": execution_mode != "parallel",
                    "started_at": step_started,
                    "ended_at": time.time(),
                    "duration_seconds": duration,
                }
                return {
                    "step_index": step_index,
                    "step_id": step.id,
                    "result_dict": result_dict,
                    "step_trace": step_trace,
                    "judge_value": result_dict.get("answer") is True,
                    "has_uncertain": isinstance(result_dict.get("answer"), str),
                    "errors": [],
                    "reuse_stats": local_reuse_stats,
                    "structured_log_count": len(logs),
                    "duration_seconds": duration,
                    "failed": False,
                    "rate_limited": False,
                    "transport_retryable": False,
                }
            except Exception as exc:
                duration = round(time.time() - step_started, 4)
                rate_limited = _is_rate_limit_error(exc)
                transport_retryable = (
                    not rate_limited and _is_retryable_transport_error(exc)
                )
                log(
                    "judge_error",
                    level="ERROR",
                    error=repr(exc),
                    duration_seconds=duration,
                    rate_limited=rate_limited,
                    transport_retryable=transport_retryable,
                    execution_mode=execution_mode,
                    retry_attempt=retry_attempt,
                )
                result_dict = _parallel_error_judge_result(step, exc)
                step_trace = _parallel_error_step_trace(step, exc, logs, duration)
                return {
                    "step_index": step_index,
                    "step_id": step.id,
                    "result_dict": result_dict,
                    "step_trace": step_trace,
                    "judge_value": False,
                    "has_uncertain": True,
                    "errors": [
                        {
                            "stage": "parallel_judge",
                            "step_id": step.id,
                            "error": repr(exc),
                            "error_class": _judge_exception_class(exc),
                            "do_not_train": True,
                        }
                    ],
                    "reuse_stats": local_reuse_stats,
                    "structured_log_count": len(logs),
                    "duration_seconds": duration,
                    "failed": True,
                    "rate_limited": rate_limited,
                    "transport_retryable": transport_retryable,
                    "exception_repr": repr(exc),
                    "exception": exc,
                }

        results_by_index: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, int(judge_concurrency or 1))) as executor:
            future_map = {
                executor.submit(run_one, step_index, step): step_index
                for step_index, step in enumerate(steps)
            }
            for future in as_completed(future_map):
                item = future.result()
                step_index = int(item.get("step_index", future_map[future]))
                results_by_index[step_index] = item
                parallel_trace["completed_step_count"] += 1
                parallel_trace["completed_order"].append(item.get("step_id", ""))
                parallel_trace["structured_log_count"] += int(
                    item.get("structured_log_count", 0) or 0
                )
                if item.get("failed"):
                    parallel_trace["failed_step_count"] += 1
                parallel_trace["step_timings"][str(item.get("step_id", step_index))] = {
                    "duration_seconds": item.get("duration_seconds", 0),
                    "failed": bool(item.get("failed")),
                }
                _merge_judge_reuse_stats(judge_reuse_stats, item.get("reuse_stats") or {})

        rate_limited_indices = sorted(
            step_index
            for step_index, item in results_by_index.items()
            if item.get("failed") and item.get("rate_limited")
        )
        if rate_limited_indices and rate_limit_serial_fallback:
            parallel_trace["rate_limit_serial_fallback_triggered"] = True
            parallel_trace["effective_concurrency_after_rate_limit"] = 1
            parallel_trace["initial_rate_limited_judge_ids"] = [
                steps[index].id for index in rate_limited_indices
            ]
            fallback_state: Dict[str, Any] = {}
            for index in rate_limited_indices:
                initial_item = results_by_index[index]
                initial_error = initial_item.get("exception")
                if not isinstance(initial_error, BaseException):
                    initial_error = RuntimeError(
                        str(initial_item.get("exception_repr") or "rate limit")
                    )
                fallback_state = _activate_rate_limit_serial_fallback(
                    judge_provider_key,
                    error=initial_error,
                )
            parallel_trace["rate_limit_fallback_state"] = fallback_state
            print(
                "[PacketRuntime] Parallel judge rate limit detected; "
                f"provider={judge_provider_key} affected_steps="
                f"{parallel_trace['initial_rate_limited_judge_ids']} "
                "switching retries and subsequent transactions to concurrency=1."
            )

            retry_limit = max(0, int(rate_limit_retry_attempts or 0))
            base_delay = max(0.0, float(rate_limit_retry_delay_seconds or 0.0))
            for step_index in rate_limited_indices:
                step = steps[step_index]
                initial_item = results_by_index[step_index]
                final_item = initial_item
                retry_history: List[Dict[str, Any]] = []
                for retry_attempt in range(1, retry_limit + 1):
                    delay = base_delay * (2 ** (retry_attempt - 1))
                    print(
                        "[PacketRuntime] Retrying rate-limited judge serially: "
                        f"judge={step.id} retry={retry_attempt}/{retry_limit} "
                        f"delay={delay}s"
                    )
                    if delay > 0:
                        time.sleep(delay)
                    retry_item = run_one(
                        step_index,
                        step,
                        execution_mode="serial_rate_limit_retry",
                        retry_attempt=retry_attempt,
                    )
                    parallel_trace["rate_limit_retry_count"] += 1
                    parallel_trace["structured_log_count"] += int(
                        retry_item.get("structured_log_count", 0) or 0
                    )
                    _merge_judge_reuse_stats(
                        judge_reuse_stats,
                        retry_item.get("reuse_stats") or {},
                    )
                    retry_history.append({
                        "attempt": retry_attempt,
                        "delay_seconds": delay,
                        "failed": bool(retry_item.get("failed")),
                        "rate_limited": bool(retry_item.get("rate_limited")),
                        "error": str(retry_item.get("exception_repr") or ""),
                    })
                    final_item = retry_item
                    if not retry_item.get("failed"):
                        parallel_trace["rate_limit_recovered_judge_ids"].append(
                            step.id
                        )
                        break
                    if not retry_item.get("rate_limited"):
                        break

                parallel_trace["rate_limit_retried_judge_ids"].append(step.id)
                retry_trace = {
                    "triggered": True,
                    "provider": judge_provider_key,
                    "original_concurrency": int(judge_concurrency or 1),
                    "fallback_concurrency": 1,
                    "initial_error": str(
                        initial_item.get("exception_repr") or ""
                    ),
                    "attempts": retry_history,
                    "recovered": not bool(final_item.get("failed")),
                }
                final_step_trace = dict(final_item.get("step_trace") or {})
                final_step_trace["rate_limit_retry"] = retry_trace
                final_item["step_trace"] = final_step_trace
                results_by_index[step_index] = final_item

            parallel_trace["failed_step_count"] = sum(
                1 for item in results_by_index.values() if item.get("failed")
            )

        transport_indices = sorted(
            step_index
            for step_index, item in results_by_index.items()
            if item.get("failed") and item.get("transport_retryable")
        )
        if (
            transport_indices
            and rate_limit_serial_fallback
            and int(rate_limit_retry_attempts or 0) > 0
        ):
            base_delay = max(0.0, float(rate_limit_retry_delay_seconds or 0.0))
            for step_index in transport_indices:
                step = steps[step_index]
                initial_item = results_by_index[step_index]
                print(
                    "[PacketRuntime] Retrying transport-failed judge serially: "
                    f"judge={step.id} retry=1/1 delay={base_delay}s"
                )
                if base_delay > 0:
                    time.sleep(base_delay)
                retry_item = run_one(
                    step_index,
                    step,
                    execution_mode="serial_transport_retry",
                    retry_attempt=1,
                )
                parallel_trace["transport_retry_count"] += 1
                parallel_trace["structured_log_count"] += int(
                    retry_item.get("structured_log_count", 0) or 0
                )
                _merge_judge_reuse_stats(
                    judge_reuse_stats,
                    retry_item.get("reuse_stats") or {},
                )
                parallel_trace["transport_retried_judge_ids"].append(step.id)
                recovered = not bool(retry_item.get("failed"))
                if recovered:
                    parallel_trace["transport_recovered_judge_ids"].append(step.id)
                retry_trace = {
                    "triggered": True,
                    "provider": judge_provider_key,
                    "original_concurrency": int(judge_concurrency or 1),
                    "retry_concurrency": 1,
                    "initial_error": str(
                        initial_item.get("exception_repr") or ""
                    ),
                    "attempts": [{
                        "attempt": 1,
                        "delay_seconds": base_delay,
                        "failed": bool(retry_item.get("failed")),
                        "transport_retryable": bool(
                            retry_item.get("transport_retryable")
                        ),
                        "error": str(retry_item.get("exception_repr") or ""),
                    }],
                    "recovered": recovered,
                }
                final_step_trace = dict(retry_item.get("step_trace") or {})
                final_step_trace["transport_retry"] = retry_trace
                retry_item["step_trace"] = final_step_trace
                results_by_index[step_index] = retry_item
            parallel_trace["failed_step_count"] = sum(
                1 for item in results_by_index.values() if item.get("failed")
            )

        judge_results: List[Dict[str, Any]] = []
        judge_step_traces: List[Dict[str, Any]] = []
        judge_values: Dict[str, bool] = {}
        errors: List[Dict[str, Any]] = []
        has_uncertain = False
        reused_judge_ids: List[str] = []
        for step_index, step in enumerate(steps):
            item = results_by_index.get(step_index)
            if item is None:
                exc = RuntimeError("parallel worker did not return a result")
                item = {
                    "result_dict": _parallel_error_judge_result(step, exc),
                    "step_trace": _parallel_error_step_trace(step, exc, [], 0.0),
                    "judge_value": False,
                    "has_uncertain": True,
                    "errors": [{
                        "stage": "parallel_judge",
                        "step_id": step.id,
                        "error": repr(exc),
                        "error_class": _judge_exception_class(exc),
                        "do_not_train": True,
                    }],
                }
            result_dict = dict(item.get("result_dict") or {})
            step_trace = dict(item.get("step_trace") or {})
            judge_results.append(result_dict)
            judge_step_traces.append(step_trace)
            judge_values[step.id] = bool(item.get("judge_value"))
            has_uncertain = has_uncertain or bool(item.get("has_uncertain"))
            errors.extend(list(item.get("errors") or []))
            if result_dict.get("reused_judge_result"):
                reused_judge_ids.append(step.id)

        parallel_trace["elapsed_parallel_seconds"] = round(time.time() - submitted_at, 4)
        judge_reuse_stats["skipped"] = (
            int(judge_reuse_stats.get("attempted", 0) or 0)
            - int(judge_reuse_stats.get("reused", 0) or 0)
        )
        return {
            "judge_results": judge_results,
            "judge_step_traces": judge_step_traces,
            "judge_values": judge_values,
            "has_uncertain": has_uncertain,
            "errors": errors,
            "reused_judge_ids": reused_judge_ids,
            "judge_reuse_stats": judge_reuse_stats,
            "parallel_trace": parallel_trace,
        }

    def _prepare_projected_step_selected_views(
        self,
        *,
        packet: Dict[str, Any],
        initial_refs: List[str],
        step: JudgeStep,
        attack_label: str,
        state_input_context: Dict[str, Any] | None,
        evidence_tool_registry: EvidenceToolRegistry | None,
    ) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Prepare exactly the selected evidence that the Judge will receive."""
        selected = self._select_views(packet, initial_refs)
        projection_registry = evidence_tool_registry or EvidenceToolRegistry(
            packet,
            debug_allow_list_views=self.debug_allow_list_views,
        )
        selected, selected_view_projection = (
            _project_access_control_stateful_selected_views(
                selected,
                step=step,
                attack_label=attack_label,
                state_input_context=state_input_context,
                evidence_tool_registry=projection_registry,
            )
        )
        selected, price_projection = _project_price_manipulation_selected_views(
            selected,
            step=step,
            attack_label=attack_label,
        )
        if price_projection:
            selected_view_projection = price_projection
        selected, reentrancy_projection = (
            _project_reentrancy_stateful_selected_views(
                selected,
                step=step,
                attack_label=attack_label,
                state_input_context=state_input_context,
            )
        )
        if reentrancy_projection:
            selected_view_projection = reentrancy_projection
        return (
            selected,
            _selected_views_hash_from_selected(selected, list(initial_refs)),
            selected_view_projection,
        )

    def _reuse_judge_step(
        self,
        step: JudgeStep,
        judge_reuse_cache: Dict[str, Dict[str, Any]],
        *,
        tx_hash: str,
        packet: Dict[str, Any],
        packet_identity_fingerprint: str = "",
        reuse_stats: Optional[Dict[str, Any]] = None,
        adaptive_enabled: bool = False,
        adaptive_mode: str = "planned_views",
        attack_label: str = "",
        evidence_profile: EvidenceProfile | None = None,
        adaptive_config: AdaptiveEvidenceConfig | None = None,
        judge_followup_mode: str = "plan",
        state_input_context: Dict[str, Any] | None = None,
        evidence_tool_registry: EvidenceToolRegistry | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]] | None:
        if reuse_stats is not None and judge_reuse_cache:
            reuse_stats["attempted"] = int(reuse_stats.get("attempted", 0) or 0) + 1

        def skip(reason: str) -> None:
            if reuse_stats is None or not judge_reuse_cache:
                return
            skipped_by_reason = reuse_stats.setdefault("skipped_by_reason", {})
            skipped_by_reason[reason] = int(skipped_by_reason.get(reason, 0) or 0) + 1
            reuse_stats.setdefault("skipped_judge_ids", []).append({
                "judge_id": step.id,
                "reason": reason,
            })

        initial_refs, _, _, adaptive_decision = self._prepare_step_evidence_refs(
            packet=packet,
            step=step,
            attack_label=attack_label,
            adaptive_enabled=adaptive_enabled,
            adaptive_mode=adaptive_mode,
            evidence_profile=evidence_profile,
            adaptive_config=adaptive_config,
            allow_runtime_default_injection=(
                normalize_judge_followup_mode(judge_followup_mode) == "plan"
            ),
        )
        selected_view_names = list(initial_refs)
        _, selected_views_hash, _ = self._prepare_projected_step_selected_views(
            packet=packet,
            initial_refs=selected_view_names,
            step=step,
            attack_label=attack_label,
            state_input_context=state_input_context,
            evidence_tool_registry=evidence_tool_registry,
        )
        fingerprint = judge_step_fingerprint(
            step,
            judge_model=str(getattr(self.judge_model.llm, "model", "") or ""),
            judge_provider=str(getattr(self.judge_model.llm, "provider", "") or ""),
            max_view_chars=int(self.judge_model.max_view_chars or 0),
            max_context_chars=int(self.judge_model.max_context_chars or 0),
            judge_max_tokens=int(
                getattr(self.judge_model.llm, "max_tokens", 0) or 0
            ),
            judge_thinking=str(
                getattr(self.judge_model.llm, "minimax_thinking", "") or ""
            ),
            source_followup_enabled=bool(self.judge_model.enable_source_followup),
            followup_context_mode=str(
                getattr(self.judge_model, "followup_context_mode", "unified")
                or "unified"
            ),
            judge_followup_mode=judge_followup_mode,
            tx_hash=tx_hash,
            packet_identity_fingerprint=packet_identity_fingerprint,
            selected_view_names=selected_view_names,
            selected_views_hash=selected_views_hash,
            adaptive_evidence_enabled=adaptive_enabled,
            adaptive_evidence_mode=adaptive_mode if adaptive_enabled else "off",
        )
        entry = judge_reuse_cache.get(fingerprint)
        if not entry:
            skip(
                _classify_reuse_miss(
                    step,
                    judge_reuse_cache,
                    tx_hash=tx_hash,
                    packet_identity_fingerprint=packet_identity_fingerprint,
                    selected_views_hash=selected_views_hash,
                )
            )
            return None
        source_tx = str((entry.get("source", {}) or {}).get("tx_hash") or "").lower().strip()
        current_tx = str(tx_hash or "").lower().strip()
        if not source_tx:
            skip("missing_source_tx")
            return None
        if source_tx != current_tx:
            skip("tx_mismatch")
            return None
        if entry.get("fingerprint") != fingerprint:
            skip("step_policy_mismatch")
            return None
        source = entry.get("source", {}) or {}
        source_packet_identity = str(
            source.get("packet_identity_fingerprint")
            or source.get("packet_fingerprint")
            or ""
        )
        if source_packet_identity != str(packet_identity_fingerprint or ""):
            skip("packet_identity_mismatch")
            return None
        if str(source.get("selected_views_hash") or "") != str(selected_views_hash or ""):
            skip("selected_views_hash_mismatch")
            return None

        result_dict = copy.deepcopy(entry.get("judge_result", {}) or {})
        if not result_dict:
            skip("empty_judge_result")
            return None
        step_trace = copy.deepcopy(entry.get("step_trace", {}) or {})
        if list(getattr(step, "consumes_state_keys", []) or []):
            if state_input_context is None:
                skip("stateful_input_dependency")
                return None
            cached_state_input = result_dict.get("state_input")
            if not isinstance(cached_state_input, dict):
                cached_state_input = step_trace.get("state_input")
            if not isinstance(cached_state_input, dict):
                skip("stateful_input_missing")
                return None
            if stable_json_dumps(cached_state_input) != stable_json_dumps(
                dict(state_input_context or {})
            ):
                skip("stateful_input_mismatch")
                return None
        produces_state_key = str(getattr(step, "produces_state_key", "") or "")
        if produces_state_key:
            cached_state_output = result_dict.get("state_output")
            if not isinstance(cached_state_output, dict) or cached_state_output.get(produces_state_key) is None:
                skip("stateful_output_missing")
                return None
        result_dict["id"] = step.id
        result_dict["judge_id"] = step.id
        result_dict["condition_id"] = step.condition_id or step.id
        result_dict["expected_answer"] = step.expected_answer
        result_dict["selected_view_names"] = list(selected_view_names)
        result_dict["selected_views_hash"] = selected_views_hash
        result_dict["satisfied"] = _judge_satisfied(
            result_dict.get("answer"),
            step.expected_answer,
        )
        result_dict["reused_judge_result"] = True
        result_dict["reuse_fingerprint"] = fingerprint
        result_dict["reuse_source"] = copy.deepcopy(entry.get("source", {}))
        result_dict["reuse_context"] = {
            "tx_hash": tx_hash,
            "packet_identity_fingerprint": packet_identity_fingerprint,
            "selected_view_names": selected_view_names,
            "selected_views_hash": selected_views_hash,
            "adaptive_evidence_enabled": adaptive_enabled,
            "adaptive_evidence_mode": adaptive_mode if adaptive_enabled else "off",
            "state_input_matched": bool(
                list(getattr(step, "consumes_state_keys", []) or [])
            ),
        }

        if not step_trace:
            step_trace = {
                "judge_id": step.id,
                "condition_id": step.condition_id or step.id,
                "judge_calls": [copy.deepcopy(result_dict)],
                "tool_calls": list(result_dict.get("tool_calls", []) or []),
                "ignored_tool_requests": list(result_dict.get("ignored_tool_requests", []) or []),
                "view_render_metadata": list(result_dict.get("view_render_metadata", []) or []),
                "all_missing_evidence_debug": list(result_dict.get("all_missing_evidence_debug", []) or []),
                "final_answer": result_dict.get("answer"),
            }
        step_trace["selected_view_names"] = selected_view_names
        step_trace["selected_views_hash"] = selected_views_hash
        step_trace["judge_id"] = step.id
        step_trace["condition_id"] = step.condition_id or step.id
        step_trace["reused_judge_result"] = True
        step_trace["reuse_fingerprint"] = fingerprint
        step_trace["reuse_source"] = copy.deepcopy(entry.get("source", {}))
        step_trace["reuse_context"] = {
            "tx_hash": tx_hash,
            "packet_identity_fingerprint": packet_identity_fingerprint,
            "selected_view_names": selected_view_names,
            "selected_views_hash": selected_views_hash,
            "adaptive_evidence_enabled": adaptive_enabled,
            "adaptive_evidence_mode": adaptive_mode if adaptive_enabled else "off",
            "state_input_matched": bool(
                list(getattr(step, "consumes_state_keys", []) or [])
            ),
        }
        step_trace["adaptive_evidence"] = adaptive_decision.to_dict()
        step_trace.setdefault("final_answer", result_dict.get("answer"))

        result_dict["tool_calls"] = list(step_trace.get("tool_calls", []))
        result_dict["judge_rounds"] = list(step_trace.get("judge_calls", []))
        result_dict["judge_output_exhaustions"] = list(
            step_trace.get("judge_output_exhaustions", [])
        )
        result_dict["source_decisiveness_guards"] = list(
            step_trace.get("source_decisiveness_guards", [])
        )
        result_dict["view_render_metadata"] = list(step_trace.get("view_render_metadata", []))
        result_dict["all_missing_evidence_debug"] = list(
            step_trace.get("all_missing_evidence_debug", [])
        )
        result_dict["ignored_tool_requests"] = list(
            step_trace.get("ignored_tool_requests", [])
        )
        return result_dict, step_trace

    def _run_judge_step(
        self,
        *,
        packet: Dict[str, Any],
        step: JudgeStep,
        evidence_tool_registry: EvidenceToolRegistry,
        tx_hash: str,
        chain: str,
        adaptive_enabled: bool = False,
        adaptive_mode: str = "planned_views",
        attack_label: str = "",
        evidence_profile: EvidenceProfile | None = None,
        adaptive_config: AdaptiveEvidenceConfig | None = None,
        judge_model: JudgeModel | None = None,
        state_input_context: Dict[str, Any] | None = None,
        stateful_runtime: Dict[str, Any] | None = None,
        initial_tool_observations: List[Dict[str, Any]] | None = None,
        initial_tool_calls: List[Dict[str, Any]] | None = None,
        initial_previous_judge_result: Dict[str, Any] | None = None,
        initial_earlier_judge_results: List[Dict[str, Any]] | None = None,
        judge_followup_mode: str = "plan",
    ) -> tuple[JudgeResult, Dict[str, Any]]:
        active_judge_model = judge_model or self.judge_model
        state_input_context = dict(state_input_context or {})
        stateful_runtime = dict(stateful_runtime or {})
        preloaded_observations = list(initial_tool_observations or [])
        preloaded_tool_calls = list(initial_tool_calls or [])
        inherited_previous_result = dict(initial_previous_judge_result or {})
        inherited_earlier_results = list(initial_earlier_judge_results or [])
        inherited_judge_results = list(inherited_earlier_results)
        if inherited_previous_result:
            inherited_judge_results.append(inherited_previous_result)
        allowed_tools = list(step.allowed_tools or [])
        (
            initial_refs,
            allowed_followup_views,
            first_pass_deferred_views,
            adaptive_decision,
        ) = self._prepare_step_evidence_refs(
            packet=packet,
            step=step,
            attack_label=attack_label,
            adaptive_enabled=adaptive_enabled,
            adaptive_mode=adaptive_mode,
            evidence_profile=evidence_profile,
            adaptive_config=adaptive_config,
            allow_runtime_default_injection=(
                normalize_judge_followup_mode(judge_followup_mode) == "plan"
            ),
        )
        selected_view_names = list(initial_refs)
        selected, selected_views_hash, selected_view_projection = (
            self._prepare_projected_step_selected_views(
                packet=packet,
                initial_refs=selected_view_names,
                step=step,
                attack_label=attack_label,
                state_input_context=state_input_context,
                evidence_tool_registry=evidence_tool_registry,
            )
        )
        if isinstance(packet.get("dictionary"), dict) and packet.get("dictionary"):
            selected = dict(selected)
            selected[PROMPT_DICTIONARY_KEY] = packet["dictionary"]
        packet_evidence_adequacy = dict(
            ((packet.get("views", {}) or {}).get("evidence_adequacy_view", {}) or {})
        )
        if adaptive_enabled:
            packet_evidence_adequacy["adaptive_evidence_note"] = ADAPTIVE_EVIDENCE_NOTE
        allowed_followup_view_summary = build_allowed_view_summary(
            packet,
            allowed_followup_views,
        )
        max_followups = max(0, int(step.max_followups or 0))
        followup_context_mode = str(
            getattr(active_judge_model, "followup_context_mode", "unified")
            or "unified"
        )
        context_builder = (
            FollowupContextBuilder(
                max_view_chars=int(active_judge_model.max_view_chars or 1),
                max_context_chars=int(active_judge_model.max_context_chars or 512),
            )
            if followup_context_mode == "unified"
            else None
        )

        def build_followup_context(
            previous_result: JudgeResult | Dict[str, Any] | None,
            observations: List[Dict[str, Any]],
            earlier_judge_results: List[Dict[str, Any]] | None = None,
        ):
            if context_builder is None:
                return None
            if isinstance(previous_result, JudgeResult):
                previous_result_dict = previous_result.to_dict()
            elif isinstance(previous_result, dict):
                previous_result_dict = dict(previous_result)
            else:
                previous_result_dict = None
            return context_builder.build(
                evidence_packet=selected,
                packet_evidence_adequacy=packet_evidence_adequacy,
                previous_judge_result=previous_result_dict,
                earlier_judge_results=list(earlier_judge_results or []),
                tool_observations=observations,
                state_input_context=state_input_context,
            )

        step_trace: Dict[str, Any] = {
            "judge_id": step.id,
            "condition_id": step.condition_id or step.id,
            "initial_views": initial_refs,
            "selected_view_names": selected_view_names,
            "selected_views_hash": selected_views_hash,
            "selected_view_projection": selected_view_projection,
            "first_pass_deferred_views": first_pass_deferred_views,
            "allowed_tools": allowed_tools,
            "allowed_followup_views": allowed_followup_views,
            "allowed_followup_view_summary": allowed_followup_view_summary,
            "adaptive_evidence": adaptive_decision.to_dict(),
            "max_followups": max_followups,
            "judge_calls": [],
            "tool_calls": list(preloaded_tool_calls),
            "ignored_tool_requests": [],
            "view_render_metadata": [],
            "judge_parse_metadata": [],
            "judge_output_exhaustions": [],
            "source_decisiveness_guards": [],
            "all_missing_evidence_debug": [],
            "state_input": dict(state_input_context),
            "state_output": {},
            "stateful_runtime": dict(stateful_runtime),
            "followup_context_mode": followup_context_mode,
            "followup_context_builds": [],
        }

        initial_followup_context = build_followup_context(
            inherited_previous_result or None,
            preloaded_observations,
            inherited_earlier_results,
        )
        result = active_judge_model.judge(
            judge_id=step.id,
            question=step.question,
            evidence_packet=selected,
            condition_id=step.condition_id,
            evidence_refs=initial_refs,
            expected_answer=step.expected_answer,
            tx_hash=tx_hash,
            chain=chain,
            allowed_tools=allowed_tools if max_followups > 0 else [],
            allowed_followup_views=allowed_followup_views if max_followups > 0 else [],
            allowed_followup_view_summary=(
                allowed_followup_view_summary if max_followups > 0 else {}
            ),
            packet_evidence_adequacy=packet_evidence_adequacy,
            tool_observations=preloaded_observations,
            allow_tool_requests=max_followups > 0,
            state_input_context=state_input_context,
            state_prompt_role=str(step.state_prompt_role or ""),
            state_output_schema=dict(step.state_output_schema or {}),
            stateful_runtime=stateful_runtime,
            followup_context=initial_followup_context,
            attack_label=attack_label,
            transcript_context={
                "phase": self.transcript_phase,
                "round": 0,
                "tx_hash": tx_hash,
                "chain": chain,
                "judge_id": step.id,
                "condition_id": step.condition_id or step.id,
                "attack_label": attack_label,
                "initial_views": initial_refs,
                "selected_view_names": selected_view_names,
                "selected_views_hash": selected_views_hash,
                "selected_view_projection": selected_view_projection,
                "adaptive_evidence": adaptive_decision.to_dict(),
                "adaptive_evidence_note": ADAPTIVE_EVIDENCE_NOTE
                if adaptive_enabled
                else "",
                "preloaded_tool_observation_count": len(preloaded_observations),
                "inherited_judge_round_count": len(inherited_judge_results),
                "state_input": state_input_context,
                "stateful_runtime": stateful_runtime,
            },
        )
        parse_metadata = dict(getattr(active_judge_model, "last_parse_metadata", {}) or {})
        parse_metadata["round"] = 0
        if getattr(active_judge_model, "last_transcript_path", ""):
            parse_metadata["transcript_path"] = active_judge_model.last_transcript_path
        step_trace["judge_parse_metadata"].append(parse_metadata)
        if parse_metadata.get("parse_status") in {"repaired", "failed"}:
            print(
                f"[PacketRuntime] Judge {step.id}: JSON parse "
                f"status={parse_metadata.get('parse_status')} "
                f"repair_attempted={parse_metadata.get('repair_attempted')}"
            )
        render_metadata = list(getattr(active_judge_model, "last_render_metadata", []) or [])
        step_trace["view_render_metadata"].extend(render_metadata)
        initial_context_metadata = dict(
            getattr(active_judge_model, "last_followup_context_metadata", {}) or {}
        )
        initial_context_metadata["round"] = 0
        step_trace["followup_context_builds"].append(initial_context_metadata)
        evidence_validation = validate_judge_evidence_ids(
            result,
            evidence_tool_registry,
            visible_evidence_ids=getattr(
                active_judge_model,
                "last_visible_evidence_ids",
                None,
            ),
        )
        step_trace.setdefault("evidence_id_validations", []).append(evidence_validation)
        step_trace["invalid_evidence_ids"] = _unique_strings(
            list(step_trace.get("invalid_evidence_ids", []) or [])
            + list(evidence_validation.get("invalid_evidence_ids", []) or [])
        )
        automatic_probe_request = _inject_access_control_anchor_probe_request(
            result,
            step=step,
            attack_label=attack_label,
            allowed_tools=allowed_tools,
            existing_observations=preloaded_observations,
        )
        if automatic_probe_request:
            step_trace.setdefault("automatic_probe_requests", []).append(
                automatic_probe_request
            )
        followup_allowed = _should_follow_up(result, max_followups, step=step)
        ignored = _ignore_final_answer_tool_requests(
            result,
            step,
            round_id=0,
            followup_allowed=followup_allowed,
        )
        step_trace["ignored_tool_requests"].extend(ignored)
        missing_items = _normalize_missing_for_round(
            result,
            step,
            round_id=0,
            packet=packet,
            render_metadata=render_metadata,
        )
        initial_round_summary = _summarize_judge_round(
            result,
            round_id=0,
            missing_items=missing_items,
            evidence_validation=evidence_validation,
        )
        initial_round_summary["parse_metadata"] = _compact_judge_parse_metadata(
            parse_metadata
        )
        step_trace["judge_calls"].append(initial_round_summary)
        if parse_metadata.get("primary_output_exhausted"):
            step_trace["judge_output_exhaustions"].append({
                "round": 0,
                **_compact_judge_parse_metadata(parse_metadata),
            })

        current_result = result
        all_observations: List[Dict[str, Any]] = list(preloaded_observations)

        for followup_index in range(max_followups):
            if not _should_follow_up(
                current_result,
                max_followups - followup_index,
                step=step,
            ):
                break

            request_round = followup_index
            judge_round = followup_index + 1
            validated_requests = self._validate_tool_requests(
                current_result.tool_requests,
                allowed_tools=allowed_tools,
                allowed_followup_views=allowed_followup_views,
                debug_allow_list_views=self.debug_allow_list_views,
            )
            observations: List[Dict[str, Any]] = []
            for item in validated_requests[:2]:
                if not item.get("valid"):
                    call = {
                        "judge_id": step.id,
                        "condition_id": step.condition_id or step.id,
                        "round": request_round,
                        "tool": item.get("tool"),
                        "args": item.get("args", {}),
                        "tool_status": "blocked",
                        "validation_error": item.get("error"),
                        "returned_evidence_ids": [],
                    }
                    step_trace["tool_calls"].append(call)
                    observations.append({
                        "tool": item.get("tool"),
                        "tool_status": "blocked",
                        "reason": item.get("error"),
                        "request": item,
                    })
                    continue

                tool_name = item["tool"]
                args, access_control_refinement = _refine_access_control_candidate_request(
                    tool_name=tool_name,
                    args=item.get("args", {}),
                    step=step,
                    attack_label=attack_label,
                    state_input_context=state_input_context,
                    evidence_tool_registry=evidence_tool_registry,
                )
                args, token_semantic_refinement = _refine_token_semantic_candidate_request(
                    tool_name=tool_name,
                    args=args,
                    step=step,
                    attack_label=attack_label,
                    state_input_context=state_input_context,
                    evidence_tool_registry=evidence_tool_registry,
                )
                args, request_refinement = _refine_reentrancy_state_order_request(
                    tool_name=tool_name,
                    args=args,
                    step=step,
                    attack_label=attack_label,
                    current_result=current_result,
                    evidence_tool_registry=evidence_tool_registry,
                )
                item["args"] = args
                stateful_refinements = [
                    refinement for refinement in (
                        access_control_refinement,
                        token_semantic_refinement,
                    )
                    if refinement
                ]
                if stateful_refinements and request_refinement:
                    request_refinement = {
                        "kind": "compound_request_refinement",
                        "refinements": [
                            *stateful_refinements,
                            request_refinement,
                        ],
                    }
                elif stateful_refinements:
                    request_refinement = (
                        stateful_refinements[0]
                        if len(stateful_refinements) == 1
                        else {
                            "kind": "compound_request_refinement",
                            "refinements": stateful_refinements,
                        }
                    )
                print(
                    f"[PacketRuntime] Judge {step.id}: follow-up tool "
                    f"{tool_name} args={args}"
                )
                reused_source_failure = None
                if tool_name == "read_function_chunk":
                    reused_source_failure = _terminal_source_failure_for_request(
                        all_observations + observations,
                        args,
                    )
                if reused_source_failure:
                    print(
                        f"[PacketRuntime] Judge {step.id}: reusing terminal "
                        "source failure instead of repeating read_function_chunk"
                    )
                    tool_result = _reused_terminal_source_failure_result(
                        args,
                        reused_source_failure,
                    )
                else:
                    tool_result = evidence_tool_registry.call(tool_name, args)
                call = {
                    "judge_id": step.id,
                    "condition_id": step.condition_id or step.id,
                    "round": request_round,
                    "tool": tool_name,
                    "args": args,
                    "reason": item.get("reason", ""),
                    "tool_status": tool_result.get("tool_status", "unknown"),
                    "returned_evidence_ids": tool_result.get("returned_evidence_ids", []),
                    "summary": tool_result.get("summary", {}),
                    "observation": tool_result,
                }
                if reused_source_failure:
                    call["source_failure_reused"] = True
                if request_refinement:
                    call["request_refinement"] = request_refinement
                step_trace["tool_calls"].append(call)
                missing_items = resolve_missing_evidence_after_tool_call(
                    missing_items,
                    call,
                    tool_result,
                )
                observations.append({
                    "request": item,
                    "result": tool_result,
                })
                fallback_observations, fallback_calls = (
                    _access_control_source_failure_fallback_observations(
                        source_tool_name=tool_name,
                        source_tool_result=tool_result,
                        step=step,
                        attack_label=attack_label,
                        state_input_context=state_input_context,
                        evidence_tool_registry=evidence_tool_registry,
                        allowed_tools=allowed_tools,
                        allowed_followup_views=allowed_followup_views,
                        existing_observations=all_observations + observations,
                        request_round=request_round,
                    )
                )
                for fallback_call in fallback_calls:
                    step_trace["tool_calls"].append(fallback_call)
                    missing_items = resolve_missing_evidence_after_tool_call(
                        missing_items,
                        fallback_call,
                        fallback_call.get("observation", {}),
                    )
                observations.extend(fallback_observations)

            if not observations:
                break

            all_observations.extend(observations)
            source_terminal_failure_seen = _observations_include_terminal_source_failure(
                observations
            )
            allow_more_tools = judge_round < max_followups and not source_terminal_failure_seen
            prepared_followup_context = build_followup_context(
                current_result,
                all_observations,
                inherited_judge_results
                + list(step_trace.get("judge_calls", []) or [])[:-1],
            )
            followup_result = active_judge_model.judge(
                judge_id=step.id,
                question=step.question,
                evidence_packet=selected,
                condition_id=step.condition_id,
                evidence_refs=initial_refs,
                expected_answer=step.expected_answer,
                tx_hash=tx_hash,
                chain=chain,
                allowed_tools=allowed_tools if allow_more_tools else [],
                allowed_followup_views=allowed_followup_views if allow_more_tools else [],
                allowed_followup_view_summary=(
                    allowed_followup_view_summary if allow_more_tools else {}
                ),
                packet_evidence_adequacy=packet_evidence_adequacy,
                tool_observations=all_observations,
                allow_tool_requests=allow_more_tools,
                state_input_context=state_input_context,
                state_prompt_role=str(step.state_prompt_role or ""),
                state_output_schema=dict(step.state_output_schema or {}),
                stateful_runtime=stateful_runtime,
                followup_context=prepared_followup_context,
                attack_label=attack_label,
                transcript_context={
                    "phase": self.transcript_phase,
                    "round": judge_round,
                    "tx_hash": tx_hash,
                    "chain": chain,
                    "judge_id": step.id,
                    "condition_id": step.condition_id or step.id,
                    "attack_label": attack_label,
                    "initial_views": initial_refs,
                    "selected_view_names": selected_view_names,
                    "selected_views_hash": selected_views_hash,
                    "selected_view_projection": selected_view_projection,
                    "tool_observation_count": len(all_observations),
                    "adaptive_evidence": adaptive_decision.to_dict(),
                    "adaptive_evidence_note": ADAPTIVE_EVIDENCE_NOTE
                    if adaptive_enabled
                    else "",
                    "state_input": state_input_context,
                    "stateful_runtime": stateful_runtime,
                },
            )
            parse_metadata_n = dict(getattr(active_judge_model, "last_parse_metadata", {}) or {})
            parse_metadata_n["round"] = judge_round
            if getattr(active_judge_model, "last_transcript_path", ""):
                parse_metadata_n["transcript_path"] = active_judge_model.last_transcript_path
            step_trace["judge_parse_metadata"].append(parse_metadata_n)
            if parse_metadata_n.get("parse_status") in {"repaired", "failed"}:
                print(
                    f"[PacketRuntime] Judge {step.id}: follow-up JSON parse "
                    f"status={parse_metadata_n.get('parse_status')} "
                    f"repair_attempted={parse_metadata_n.get('repair_attempted')}"
                )
            render_metadata_n = list(getattr(active_judge_model, "last_render_metadata", []) or [])
            step_trace["view_render_metadata"].extend(render_metadata_n)
            context_metadata_n = dict(
                getattr(active_judge_model, "last_followup_context_metadata", {}) or {}
            )
            context_metadata_n["round"] = judge_round
            step_trace["followup_context_builds"].append(context_metadata_n)
            evidence_validation_n = validate_judge_evidence_ids(
                followup_result,
                evidence_tool_registry,
                visible_evidence_ids=getattr(
                    active_judge_model,
                    "last_visible_evidence_ids",
                    None,
                ),
            )
            step_trace.setdefault("evidence_id_validations", []).append(evidence_validation_n)
            step_trace["invalid_evidence_ids"] = _unique_strings(
                list(step_trace.get("invalid_evidence_ids", []) or [])
                + list(evidence_validation_n.get("invalid_evidence_ids", []) or [])
            )
            followup_allowed_n = _should_follow_up(
                followup_result,
                max_followups - judge_round,
                step=step,
            )
            step_trace["ignored_tool_requests"].extend(
                _ignore_final_answer_tool_requests(
                    followup_result,
                    step,
                    round_id=judge_round,
                    followup_allowed=followup_allowed_n,
                )
            )
            followup_missing = _normalize_missing_for_round(
                followup_result,
                step,
                round_id=judge_round,
                packet=packet,
                render_metadata=render_metadata_n,
            )
            for call in step_trace["tool_calls"]:
                followup_missing = resolve_missing_evidence_after_tool_call(
                    followup_missing,
                    call,
                    call.get("observation", {}),
                )
            output_exhausted = _should_preserve_previous_judge_result(
                parse_metadata_n
            )
            if not output_exhausted:
                missing_items = dedupe_missing_evidence(
                    list(missing_items) + list(followup_missing)
                )
            followup_result.tool_calls = list(step_trace["tool_calls"])
            followup_round_summary = _summarize_judge_round(
                followup_result,
                round_id=judge_round,
                missing_items=followup_missing,
                evidence_validation=evidence_validation_n,
            )
            followup_round_summary["parse_metadata"] = _compact_judge_parse_metadata(
                parse_metadata_n
            )
            step_trace["judge_calls"].append(followup_round_summary)
            if output_exhausted:
                step_trace["judge_output_exhaustions"].append({
                    "round": judge_round,
                    "preserved_previous_answer": current_result.answer,
                    "preserved_previous_confidence": current_result.confidence,
                    **_compact_judge_parse_metadata(parse_metadata_n),
                })
                print(
                    f"[PacketRuntime] Judge {step.id}: output exhausted at "
                    f"round={judge_round}; preserving prior local judgment "
                    f"answer={current_result.answer}."
                )
                break
            source_guard = _unpinned_source_negative_guard(
                previous_result=current_result,
                candidate_result=followup_result,
                observations=observations,
            )
            if source_guard["applied"]:
                step_trace["source_decisiveness_guards"].append({
                    "round": judge_round,
                    **source_guard,
                })
                print(
                    f"[PacketRuntime] Judge {step.id}: unpinned source cannot "
                    "turn an unresolved runtime judgment into false; preserving "
                    f"answer={current_result.answer}."
                )
                break
            current_result = followup_result

        missing_items = _finalize_missing_for_result(current_result, missing_items)
        current_result.missing_evidence = open_blocking_texts(missing_items)
        current_result.tool_calls = list(step_trace["tool_calls"])
        step_trace["state_output"] = dict(current_result.state_output or {})
        step_trace["state_input"] = dict(state_input_context)
        step_trace["stateful_runtime"] = dict(stateful_runtime)
        step_trace["all_missing_evidence_debug"] = dedupe_missing_evidence(missing_items)
        step_trace["final_answer"] = current_result.answer
        step_trace.setdefault("invalid_evidence_ids", [])
        return current_result, step_trace

    def _prepare_step_evidence_refs(
        self,
        *,
        packet: Dict[str, Any],
        step: JudgeStep,
        attack_label: str,
        adaptive_enabled: bool,
        adaptive_mode: str,
        evidence_profile: EvidenceProfile | None,
        adaptive_config: AdaptiveEvidenceConfig | None,
        allow_runtime_default_injection: bool = True,
    ) -> tuple[List[str], List[str], List[str], AdaptiveEvidenceDecision]:
        planned_defaults = list(step.default_evidence_refs or step.evidence_refs)
        planned_followups = list(step.allowed_followup_views or [])
        label_deferred: List[str] = []
        view_budget = resolve_judge_step_view_budget(step)
        planned_initial, planned_followup, planned_deferred = constrain_first_pass_view_refs(
            planned_defaults,
            planned_followups,
            max_default_views=int(view_budget["max_default_views"]),
            max_large_views=int(view_budget["max_large_views"]),
            max_followup_views=int(view_budget["max_followup_views"]),
        )
        planned_deferred = _unique_strings(label_deferred + planned_deferred)
        if not adaptive_enabled:
            decision = AdaptiveEvidenceDecision(
                enabled=False,
                mode="planned_views",
                reason="adaptive evidence disabled",
                initial_refs_before=list(planned_initial),
                initial_refs_after=list(planned_initial),
                allowed_followup_before=list(planned_followup),
                allowed_followup_after=list(planned_followup),
                added_views=[],
                removed_views=[],
                deferred_views=list(planned_deferred),
                estimated_chars=0,
                budget={
                    "max_view_chars": int(self.judge_model.max_view_chars or 0),
                    "max_context_chars": int(self.judge_model.max_context_chars or 0),
                },
            )
            return planned_initial, planned_followup, planned_deferred, decision
        profile = evidence_profile or extract_evidence_profile(
            packet,
            max_context_chars=int(self.judge_model.max_context_chars or 0),
        )
        decision = adapt_step_evidence_refs(
            packet=packet,
            step=step,
            attack_label=attack_label,
            initial_refs=list(planned_initial),
            allowed_followup_views=list(planned_followup),
            profile=profile,
            mode=adaptive_mode,
            max_context_chars=int(self.judge_model.max_context_chars or 0),
            max_view_chars=int(self.judge_model.max_view_chars or 0),
            config=adaptive_config,
        )
        return (
            list(decision.initial_refs_after),
            list(decision.allowed_followup_after),
            list(planned_deferred) + list(decision.deferred_views),
            decision,
        )

    def _run_near_miss_escalation(
        self,
        *,
        packet: Dict[str, Any],
        step: JudgeStep,
        near_miss: Dict[str, Any],
        evidence_tool_registry: EvidenceToolRegistry,
        tx_hash: str,
        chain: str,
        attack_label: str = "",
        prior_step_trace: Dict[str, Any] | None = None,
        state_input_context: Dict[str, Any] | None = None,
        stateful_runtime: Dict[str, Any] | None = None,
    ) -> tuple[JudgeResult | None, Dict[str, Any]]:
        constrained_near_miss = dict(near_miss or {})
        constrained_near_miss["single_rejudge_only"] = True
        escalation_step = _make_near_miss_escalation_step(
            step,
            constrained_near_miss,
        )
        prior_judge_rounds = list(
            dict(prior_step_trace or {}).get("judge_calls", []) or []
        )
        prior_observations = _tool_observations_from_step_trace(prior_step_trace)
        requested_observations, requested_tool_calls = (
            self._near_miss_requested_observations(
                near_miss=constrained_near_miss,
                escalation_step=escalation_step,
                evidence_tool_registry=evidence_tool_registry,
                prior_observations=prior_observations,
                limit=1,
            )
        )
        remaining_observation_slots = max(0, 2 - len(requested_observations))
        forced_observations, forced_tool_calls = self._forced_source_followup(
            near_miss=constrained_near_miss,
            escalation_step=escalation_step,
            evidence_tool_registry=evidence_tool_registry,
            prior_observations=prior_observations + requested_observations,
            limit=min(1, remaining_observation_slots),
        )
        new_observations = requested_observations + forced_observations
        new_tool_calls = requested_tool_calls + forced_tool_calls
        target_condition_evidence_count = sum(
            1
            for observation in new_observations
            if _near_miss_observation_has_target_evidence(observation)
        )
        if not new_observations:
            return None, {
                "judge_calls": [],
                "tool_calls": [],
                "judge_output_exhaustions": [],
                "near_miss_escalation": {
                    "enabled": True,
                    "attempted": False,
                    "skip_reason": "no_new_observation",
                    "source_judge_id": step.id,
                    "blocker_type": constrained_near_miss.get("blocker_type"),
                    "new_observation_count": 0,
                    "target_condition_evidence_count": 0,
                },
            }
        result, step_trace = self._run_judge_step(
            packet=packet,
            step=escalation_step,
            evidence_tool_registry=evidence_tool_registry,
            tx_hash=tx_hash,
            chain=chain,
            attack_label=attack_label,
            state_input_context=state_input_context,
            stateful_runtime=stateful_runtime,
            initial_tool_observations=prior_observations + new_observations,
            initial_tool_calls=new_tool_calls,
            initial_previous_judge_result=(
                prior_judge_rounds[-1] if prior_judge_rounds else None
            ),
            initial_earlier_judge_results=prior_judge_rounds[:-1],
        )
        step_trace["near_miss_escalation"] = {
            "enabled": True,
            "source_judge_id": step.id,
            "blocker_type": constrained_near_miss.get("blocker_type"),
            "near_miss_policy": constrained_near_miss.get("near_miss_policy", ""),
            "cluster_id": constrained_near_miss.get("cluster_id", ""),
            "cluster_members": list(
                constrained_near_miss.get("cluster_members", []) or []
            ),
            "pre_escalation_answer": constrained_near_miss.get("answer"),
            "pre_escalation_confidence": constrained_near_miss.get("confidence"),
            "forced_source_followup": bool(forced_tool_calls),
            "inherited_judge_round_count": len(prior_judge_rounds),
            "inherited_tool_observation_count": len(prior_observations),
            "new_observation_count": len(new_observations),
            "target_condition_evidence_count": target_condition_evidence_count,
            "single_rejudge_only": True,
            "source_probe_evidence_ids": list(
                constrained_near_miss.get("source_probe_evidence_ids", []) or []
            ),
        }
        return result, step_trace

    def _rerun_stateful_dependency_closure(
        self,
        *,
        packet: Dict[str, Any],
        plan: EvidencePlan,
        source_step: JudgeStep,
        judge_results: List[Dict[str, Any]],
        judge_step_traces: List[Dict[str, Any]],
        judge_values: Dict[str, bool],
        evidence_tool_registry: EvidenceToolRegistry,
        tx_hash: str,
        chain: str,
        attack_label: str,
        adaptive_enabled: bool,
        adaptive_mode: str,
        evidence_profile: EvidenceProfile | None,
        adaptive_config: AdaptiveEvidenceConfig | None,
        stateful_enabled: bool,
        disable_stateful_bindings: bool,
        reentrancy_binding_mode: str,
        judge_followup_mode: str,
        judge_followup_policy: Dict[str, Any],
        rate_limit_serial_fallback: bool,
        rate_limit_retry_attempts: int,
        rate_limit_retry_delay_seconds: float,
        errors: List[Dict[str, Any]],
        stateful_trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rerun only consumers whose state input changed after a producer retry."""
        source_condition_id = str(
            source_step.condition_id or source_step.id or ""
        ).strip().upper()
        affected = dependency_affected_condition_ids(
            plan,
            {source_condition_id},
        )
        if not affected:
            return {
                "triggered": False,
                "producer_condition_id": source_condition_id,
                "producer_state_key": str(source_step.produces_state_key or ""),
                "producer_state_changed": True,
                "rerun_condition_ids": [],
            }

        affected_ids = {
            str(value or "").strip().upper() for value in affected if str(value or "").strip()
        }
        state_store: Dict[str, Any] = {}
        rerun_ids: List[str] = []
        failures: List[Dict[str, Any]] = []
        for index, step in enumerate(plan.judge_steps):
            if index >= len(judge_results):
                break
            condition_id = str(step.condition_id or step.id or "").strip().upper()
            if condition_id not in affected_ids:
                _update_state_store_from_result(
                    state_store,
                    step,
                    judge_results[index],
                )
                continue

            state_input = _state_input_for_step(step, state_store)
            stateful_step_meta = _step_stateful_runtime_metadata(
                step,
                stateful_enabled=stateful_enabled,
                bindings_disabled=disable_stateful_bindings,
                binding_mode=reentrancy_binding_mode,
            )
            try:
                def run_dependency_judge_step():
                    return self._run_judge_step(
                        packet=packet,
                        step=step,
                        evidence_tool_registry=evidence_tool_registry,
                        tx_hash=tx_hash,
                        chain=chain,
                        adaptive_enabled=adaptive_enabled,
                        adaptive_mode=adaptive_mode,
                        attack_label=attack_label,
                        evidence_profile=evidence_profile,
                        adaptive_config=adaptive_config,
                        state_input_context=state_input,
                        stateful_runtime=stateful_step_meta,
                        judge_followup_mode=judge_followup_mode,
                    )

                (result, step_trace), retry_audit = _call_with_rate_limit_retry(
                    run_dependency_judge_step,
                    enabled=rate_limit_serial_fallback,
                    provider=_judge_provider_key(self.judge_model),
                    max_retries=rate_limit_retry_attempts,
                    initial_delay_seconds=rate_limit_retry_delay_seconds,
                )
                if retry_audit.get("triggered"):
                    step_trace["judge_retry"] = retry_audit
                step_trace["judge_followup_policy"] = _judge_followup_step_audit(
                    judge_followup_policy,
                    step.id,
                )
                result_dict = _materialize_escalated_judge_result(
                    result,
                    step,
                    step_trace,
                    state_input=state_input,
                    stateful_runtime=stateful_step_meta,
                )
                result_dict["judge_followup_policy"] = copy.deepcopy(
                    step_trace["judge_followup_policy"]
                )
            except Exception as exc:
                result_dict = _serial_error_judge_result(
                    step,
                    exc,
                    state_input=state_input,
                    stateful_runtime=stateful_step_meta,
                )
                step_trace = _serial_error_step_trace(
                    step,
                    exc,
                    state_input=state_input,
                    stateful_runtime=stateful_step_meta,
                )
                failure = {
                    "stage": "near_miss_dependency_retry",
                    "step_id": step.id,
                    "error": repr(exc),
                    "error_class": _judge_exception_class(exc),
                    "do_not_train": True,
                }
                failures.append(failure)
                errors.append(failure)

            judge_results[index] = result_dict
            judge_step_traces[index] = step_trace
            judge_values[step.id] = result_dict.get("answer") is True
            _update_state_store_from_result(state_store, step, result_dict)
            rerun_ids.append(condition_id)

        stateful_trace["state_keys_produced"] = sorted(state_store.keys())
        return {
            "triggered": bool(rerun_ids),
            "producer_condition_id": source_condition_id,
            "producer_state_key": str(source_step.produces_state_key or ""),
            "producer_state_changed": True,
            "rerun_condition_ids": rerun_ids,
            "runtime_failures": failures,
        }

    def _forced_source_followup(
        self,
        *,
        near_miss: Dict[str, Any],
        escalation_step: JudgeStep,
        evidence_tool_registry: EvidenceToolRegistry,
        prior_observations: List[Dict[str, Any]] | None = None,
        limit: int = 1,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if limit <= 0:
            return [], []
        if not bool(near_miss.get("force_source_followup")):
            return [], []
        if "read_function_chunk" not in set(escalation_step.allowed_tools or []):
            return [], []

        observations: List[Dict[str, Any]] = []
        calls: List[Dict[str, Any]] = []
        seen_requests = _near_miss_observation_request_signatures(
            prior_observations or []
        )
        for evidence_id in _source_probe_evidence_ids(near_miss)[:limit]:
            args = {"evidence_id": evidence_id}
            request_signature = _near_miss_request_signature(
                "read_function_chunk",
                args,
            )
            if request_signature in seen_requests:
                continue
            seen_requests.add(request_signature)
            if str(near_miss.get("near_miss_policy") or "").startswith(
                "dynamic_aggregator"
            ):
                reason = (
                    "Dynamic aggregator requested this source probe; the owning "
                    "local judge must consume it before the final aggregation pass."
                )
            else:
                reason = (
                    "Forced near-miss source probe for an important failure cluster "
                    "before re-judging this condition."
                )
            tool_result = evidence_tool_registry.call("read_function_chunk", args)
            call = {
                "judge_id": escalation_step.id,
                "condition_id": escalation_step.condition_id or escalation_step.id,
                "round": -1,
                "tool": "read_function_chunk",
                "args": args,
                "reason": reason,
                "tool_status": tool_result.get("tool_status", "unknown"),
                "returned_evidence_ids": tool_result.get("returned_evidence_ids", []),
                "summary": tool_result.get("summary", {}),
                "observation": tool_result,
                "forced_source_followup": True,
                "near_miss_policy": near_miss.get("near_miss_policy", ""),
                "cluster_id": near_miss.get("cluster_id", ""),
            }
            calls.append(call)
            observations.append({
                "request": {
                    "tool": "read_function_chunk",
                    "args": args,
                    "reason": reason,
                    "forced_source_followup": True,
                },
                "result": tool_result,
                "target_condition_specific": True,
            })
        return observations, calls

    def _near_miss_requested_observations(
        self,
        *,
        near_miss: Dict[str, Any],
        escalation_step: JudgeStep,
        evidence_tool_registry: EvidenceToolRegistry,
        prior_observations: List[Dict[str, Any]],
        limit: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if limit <= 0:
            return [], []
        requests: List[Dict[str, Any]] = []
        requested_views: set[str] = set()
        for view_name in list(near_miss.get("suggested_followup_views", []) or []):
            view_name = str(view_name or "").strip()
            if not view_name or view_name in requested_views:
                continue
            requests.append({
                "tool": "read_packet_view",
                "args": {"view": view_name},
                "reason": (
                    "Fetch the target Judge's suggested evidence before the "
                    "single near-miss rejudgment."
                ),
            })
            requested_views.add(view_name)
        for raw_request in list(near_miss.get("tool_requests", []) or []):
            if not isinstance(raw_request, dict):
                continue
            item = copy.deepcopy(raw_request)
            view_name = str((item.get("args") or {}).get("view") or "").strip()
            if (
                str(item.get("tool") or "") == "read_packet_view"
                and view_name in requested_views
            ):
                continue
            requests.append(item)
            if view_name:
                requested_views.add(view_name)

        validated = self._validate_tool_requests(
            requests,
            allowed_tools=list(escalation_step.allowed_tools or []),
            allowed_followup_views=list(
                escalation_step.allowed_followup_views or []
            ),
            debug_allow_list_views=self.debug_allow_list_views,
        )
        seen_requests = _near_miss_observation_request_signatures(
            prior_observations
        )
        observations: List[Dict[str, Any]] = []
        calls: List[Dict[str, Any]] = []
        for item in validated:
            if len(observations) >= limit or not item.get("valid"):
                continue
            tool_name = str(item.get("tool") or "")
            args = dict(item.get("args") or {})
            signature = _near_miss_request_signature(tool_name, args)
            if signature in seen_requests:
                continue
            seen_requests.add(signature)
            tool_result = evidence_tool_registry.call(tool_name, args)
            call = {
                "judge_id": escalation_step.id,
                "condition_id": escalation_step.condition_id or escalation_step.id,
                "round": -1,
                "tool": tool_name,
                "args": args,
                "reason": str(item.get("reason") or ""),
                "tool_status": tool_result.get("tool_status", "unknown"),
                "returned_evidence_ids": tool_result.get(
                    "returned_evidence_ids", []
                ),
                "summary": tool_result.get("summary", {}),
                "observation": tool_result,
                "near_miss_prefetch": True,
            }
            calls.append(call)
            observations.append({
                "request": {
                    "tool": tool_name,
                    "args": args,
                    "reason": str(item.get("reason") or ""),
                },
                "result": tool_result,
                "target_condition_specific": True,
            })
        return observations, calls

    def _run_dynamic_aggregation_followup_round(
        self,
        *,
        packet: Dict[str, Any],
        plan: EvidencePlan,
        judge_results: List[Dict[str, Any]],
        judge_step_traces: List[Dict[str, Any]],
        judge_values: Dict[str, bool],
        followup_requests: List[Dict[str, Any]],
        evidence_tool_registry: EvidenceToolRegistry,
        tx_hash: str,
        chain: str,
        attack_label: str = "",
        near_miss_escalations: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        accepted, rejected = _prepare_dynamic_aggregation_followup_requests(
            followup_requests,
            plan=plan,
            judge_results=judge_results,
            near_miss_escalations=list(near_miss_escalations or []),
        )
        trace: Dict[str, Any] = {
            "enabled": True,
            "max_rounds": 1,
            "round": 1,
            "requested_count": len(followup_requests),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_requests": copy.deepcopy(accepted),
            "rejected_requests": copy.deepcopy(rejected),
            "judge_reruns": [],
            "errors": [],
            "completed": False,
        }
        step_by_id = {str(step.id).upper(): step for step in plan.judge_steps}
        for request in accepted:
            judge_id = str(request.get("judge_id") or "")
            target_step = step_by_id.get(judge_id.upper())
            target_index = int(request.get("judge_index", -1))
            if target_step is None or not (0 <= target_index < len(judge_results)):
                trace["rejected_requests"].append({
                    **request,
                    "rejection_reason": "target_judge_not_found",
                })
                trace["rejected_count"] = len(trace["rejected_requests"])
                continue

            old_result = dict(judge_results[target_index] or {})
            old_trace = (
                judge_step_traces[target_index]
                if target_index < len(judge_step_traces)
                else {}
            )
            policy = "dynamic_aggregator_source_followup"
            blocker_type = (
                "exclusion_aggregator_followup"
                if _is_exclusion_step(target_step)
                else "core_aggregator_followup"
            )
            escalation = {
                "judge_id": target_step.id,
                "judge_index": target_index,
                "condition_id": target_step.condition_id or target_step.id,
                "answer": old_result.get("answer"),
                "confidence": old_result.get("confidence", "medium"),
                "blocker_type": blocker_type,
                "near_miss_policy": policy,
                "cluster_id": str(request.get("cluster_id") or ""),
                "cluster_members": list(request.get("cluster_members", []) or []),
                "force_source_followup": True,
                "single_rejudge_only": True,
                "source_probe_evidence_ids": list(
                    request.get("evidence_ids", []) or []
                ),
                "aggregator_followup_reason": str(request.get("reason") or ""),
            }
            print(
                "[PacketRuntime] Dynamic aggregator follow-up: "
                f"judge={target_step.id} evidence_ids="
                f"{escalation['source_probe_evidence_ids']}"
            )
            try:
                escalated_result, escalated_trace = self._run_near_miss_escalation(
                    packet=packet,
                    step=target_step,
                    near_miss=escalation,
                    evidence_tool_registry=evidence_tool_registry,
                    tx_hash=tx_hash,
                    chain=chain,
                    attack_label=attack_label,
                    prior_step_trace=old_trace,
                    state_input_context=dict(old_trace.get("state_input", {}) or {}),
                    stateful_runtime=dict(old_trace.get("stateful_runtime", {}) or {}),
                )
                if escalated_result is None:
                    trace["judge_reruns"].append({
                        "judge_id": target_step.id,
                        "condition_id": target_step.condition_id or target_step.id,
                        "attempted": False,
                        "reason": (
                            escalated_trace.get("near_miss_escalation", {}) or {}
                        ).get("skip_reason", "no_new_observation"),
                    })
                    continue
                merged_trace = _merge_escalated_step_trace(
                    old_trace,
                    escalated_trace,
                    near_miss=escalation,
                )
                merged_trace["dynamic_aggregation_followup"] = {
                    "enabled": True,
                    "round": 1,
                    "request": copy.deepcopy(request),
                }
                escalated_dict = _materialize_escalated_judge_result(
                    escalated_result,
                    target_step,
                    merged_trace,
                    state_input=dict(old_trace.get("state_input", {}) or {}),
                    stateful_runtime=dict(old_trace.get("stateful_runtime", {}) or {}),
                )
                escalated_dict = _enforce_near_miss_upgrade_guard(
                    original_result=old_result,
                    escalated_result=escalated_dict,
                    escalated_trace=escalated_trace,
                )
                merged_trace["final_answer"] = escalated_dict.get("answer")
                merged_trace["near_miss_upgrade_guard"] = dict(
                    escalated_dict.get("near_miss_upgrade_guard", {}) or {}
                )
                escalated_dict["dynamic_aggregation_followup"] = True
                escalated_dict["dynamic_aggregation_followup_request"] = copy.deepcopy(
                    request
                )
                judge_results[target_index] = escalated_dict
                judge_step_traces[target_index] = merged_trace
                judge_values[target_step.id] = escalated_dict.get("answer") is True
                new_tool_calls = list(merged_trace.get("tool_calls", []) or [])[len(
                    list(old_trace.get("tool_calls", []) or [])
                ):]
                trace["judge_reruns"].append({
                    "judge_id": target_step.id,
                    "condition_id": target_step.condition_id or target_step.id,
                    "before_answer": old_result.get("answer"),
                    "before_confidence": old_result.get("confidence"),
                    "after_answer": escalated_dict.get("answer"),
                    "after_confidence": escalated_dict.get("confidence"),
                    "changed_answer": old_result.get("answer")
                    != escalated_dict.get("answer"),
                    "source_probe_evidence_ids": list(
                        request.get("evidence_ids", []) or []
                    ),
                    "tool_call_count": len(new_tool_calls),
                    "source_tool_statuses": [
                        str(call.get("tool_status") or "unknown")
                        for call in new_tool_calls
                        if call.get("tool") == "read_function_chunk"
                    ],
                })
            except Exception as exc:
                error = {
                    "stage": "dynamic_aggregation_followup",
                    "judge_id": target_step.id,
                    "condition_id": target_step.condition_id or target_step.id,
                    "error": repr(exc),
                }
                trace["errors"].append(error)
                trace["judge_reruns"].append({
                    **request,
                    "error": repr(exc),
                })
                print(
                    f"[PacketRuntime] Dynamic aggregator follow-up "
                    f"{target_step.id} raised: {exc!r}"
                )
        trace["completed"] = True
        trace["rerun_count"] = len([
            item for item in trace["judge_reruns"] if not item.get("error")
        ])
        return trace

    def _apply_dynamic_aggregation(
        self,
        *,
        rule: EvolvingRule,
        plan: EvidencePlan,
        judge_results: List[Dict[str, Any]],
        judge_values: Dict[str, bool],
        previous_verdict: str,
        verdict_aggregation: Dict[str, Any],
        attack_label: str,
        tx_hash: str,
        chain: str,
        near_miss_escalations: List[Dict[str, Any]],
        aggregation_round: int = 1,
        allow_followup_requests: bool = True,
        force_agent_call: bool = False,
        followup_context: Dict[str, Any] | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        analysis = _analyze_dynamic_aggregation_context(
            plan=plan,
            judge_results=judge_results,
            attack_label=attack_label,
            near_miss_escalations=near_miss_escalations,
        )
        decision = _deterministic_dynamic_aggregation_decision(
            previous_verdict=previous_verdict,
            analysis=analysis,
            previous_reason=str(verdict_aggregation.get("reason") or ""),
        )
        agent_response: Dict[str, Any] = {}
        agent_error = ""
        agent_parse_metadata: Dict[str, Any] = {}
        prompt = ""
        raw_completion = ""
        source_followup_available = bool(
            getattr(self.judge_model, "enable_source_followup", False)
        )
        followup_requests_available = bool(
            allow_followup_requests and source_followup_available
        )
        if (
            force_agent_call
            or _should_call_dynamic_aggregator_agent(
                previous_verdict=previous_verdict,
                analysis=analysis,
            )
        ) and self.aggregator_llm is not None:
            prompt = _build_dynamic_aggregator_prompt(
                rule=rule,
                plan=plan,
                judge_results=judge_results,
                judge_values=judge_values,
                previous_verdict=previous_verdict,
                verdict_aggregation=verdict_aggregation,
                analysis=analysis,
                aggregation_round=aggregation_round,
                allow_followup_requests=followup_requests_available,
                followup_context=followup_context,
            )
            try:
                raw_completion = self.aggregator_llm.complete(prompt)
                parse_completion, thinking_metadata = structured_output_text(
                    raw_completion,
                    provider=_llm_provider_key(self.aggregator_llm),
                )
                agent_response, parse_error, agent_parse_metadata = (
                    _try_extract_dynamic_aggregator_json_object(parse_completion)
                )
                agent_parse_metadata = {
                    **thinking_metadata,
                    **dict(agent_parse_metadata or {}),
                    "parse_status": "ok" if parse_error is None else "failed",
                    "fallback_reason": "" if parse_error is None else (
                        "output_truncated_without_schema_valid_json"
                        if str(
                            getattr(
                                self.aggregator_llm,
                                "last_finish_reason",
                                "",
                            )
                            or ""
                        ).lower()
                        in {"length", "max_tokens"}
                        else "no_schema_valid_aggregator_json"
                    ),
                }
                if parse_error is not None:
                    raise JsonExtractionError(parse_error)
                decision = _sanitize_dynamic_aggregation_decision(
                    agent_response,
                    previous_verdict=previous_verdict,
                    previous_reason=str(verdict_aggregation.get("reason") or ""),
                    analysis=analysis,
                    allow_followup_requests=followup_requests_available,
                )
            except (JsonExtractionError, Exception) as exc:
                agent_error = repr(exc)
                agent_parse_metadata.setdefault("parse_status", "failed")
                agent_parse_metadata.setdefault(
                    "fallback_reason",
                    "aggregator_request_or_parse_failed",
                )
                decision.setdefault("warnings", []).append(
                    f"dynamic_aggregator_agent_failed: {agent_error}"
                )
            finally:
                _record_dynamic_aggregation_transcript(
                    judge_model=self.judge_model,
                    llm=self.aggregator_llm,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    parsed_response=agent_response,
                    tx_hash=tx_hash,
                    chain=chain,
                    analysis=analysis,
                    error=agent_error,
                    parse_metadata=agent_parse_metadata,
                    phase=self.transcript_phase,
                    aggregation_round=aggregation_round,
                    followup_context=followup_context,
                )

        decision = _enforce_post_followup_upgrade_guard(
            decision,
            previous_verdict=previous_verdict,
            aggregation_round=aggregation_round,
            followup_context=followup_context,
        )
        final_verdict = str(decision.get("verdict") or previous_verdict)
        previous_decision_source = str(
            verdict_aggregation.get("decision_source") or "unknown"
        )
        dynamic_record = {
            "enabled": True,
            "applied": final_verdict != previous_verdict,
            "previous_verdict": previous_verdict,
            "final_verdict": final_verdict,
            "decision": decision,
            "analysis": analysis,
            "agent_response": agent_response,
            "agent_error": agent_error,
            "parse_metadata": agent_parse_metadata,
            "aggregation_round": aggregation_round,
            "allow_followup_requests": followup_requests_available,
            "source_followup_available": source_followup_available,
        }
        updated = dict(verdict_aggregation)
        updated["dynamic_aggregation"] = dynamic_record
        updated["aggregator_override"] = {
            "attempted": bool(agent_response) or bool(force_agent_call),
            "applied": final_verdict != previous_verdict,
            "mode": str(decision.get("mode") or "deterministic"),
            "previous_decision_source": previous_decision_source,
        }
        if final_verdict != previous_verdict:
            updated["previous_decision_source"] = previous_decision_source
            updated["decision_source"] = (
                "dynamic_aggregator_agent"
                if str(decision.get("mode") or "") == "agent"
                else "dynamic_aggregation_policy"
            )
            updated["reason"] = str(
                decision.get("reason")
                or f"dynamic_aggregation_override_from_{previous_verdict}"
            )
            updated["previous_reason"] = verdict_aggregation.get("reason", "")
        elif aggregation_round > 1 and decision.get("mode") == "agent":
            updated["reason"] = str(
                decision.get("reason")
                or verdict_aggregation.get("reason", "")
            )
        else:
            updated.setdefault("reason", verdict_aggregation.get("reason", ""))
        return final_verdict, updated

    @staticmethod
    def _validate_tool_requests(
        tool_requests: List[Dict[str, Any]],
        *,
        allowed_tools: List[str],
        allowed_followup_views: List[str],
        debug_allow_list_views: bool = False,
    ) -> List[Dict[str, Any]]:
        allowed_tool_set = set(allowed_tools) & ALLOWED_EVIDENCE_TOOLS
        allowed_view_set = set(allowed_followup_views)
        out: List[Dict[str, Any]] = []
        for request in list(tool_requests or [])[:2]:
            tool = str(request.get("tool") or "").strip()
            args = dict(request.get("args") or {})
            item = {
                "tool": tool,
                "args": args,
                "reason": str(request.get("reason", "")),
                "valid": True,
            }
            if tool == "list_packet_views" and not debug_allow_list_views:
                item["valid"] = False
                item["error"] = (
                    "list_packet_views is debug-only; use "
                    "allowed_followup_view_summary and request read_packet_view directly."
                )
            elif tool not in allowed_tool_set:
                item["valid"] = False
                item["error"] = f"Tool '{tool}' is not allowed for this judge step."
            elif tool == "read_packet_view":
                view = str(
                    args.get("view") or args.get("view_name") or ""
                ).strip()
                if view:
                    args["view"] = view
                    args.pop("view_name", None)
                    item["args"] = args
                if view not in allowed_view_set:
                    item["valid"] = False
                    item["error"] = (
                        f"View '{view}' is not in allowed_followup_views for this judge step."
                    )
            elif tool == "read_evidence_context":
                if not args.get("evidence_id") and isinstance(args.get("evidence_ids"), list):
                    first_id = next((str(eid).strip() for eid in args.get("evidence_ids", []) if str(eid).strip()), "")
                    if first_id:
                        args["evidence_id"] = first_id
                        item["args"] = args
                evidence_id = str(args.get("evidence_id") or "").strip()
                if not _looks_like_evidence_id(evidence_id):
                    item["valid"] = False
                    item["error"] = (
                        "read_evidence_context requires a concrete evidence_id "
                        "such as call:14, event:80, transfer_event:33, or "
                        "state_semantic:sstore:25."
                    )
            elif tool == "read_evidence_by_id":
                if args.get("evidence_id") and not args.get("evidence_ids"):
                    args["evidence_ids"] = [args.get("evidence_id")]
                raw_ids = args.get("evidence_ids") or []
                if isinstance(raw_ids, str):
                    raw_ids = [raw_ids]
                normalized_ids = _unique_strings(
                    str(eid).strip()
                    for eid in raw_ids
                    if str(eid).strip()
                )
                if not normalized_ids:
                    item["valid"] = False
                    item["error"] = "read_evidence_by_id requires non-empty evidence_ids."
                elif any(not _looks_like_evidence_id(eid) for eid in normalized_ids):
                    item["valid"] = False
                    item["error"] = (
                        "read_evidence_by_id evidence_ids must be concrete packet evidence ids "
                        "such as call:14, event:80, transfer_event:33, or state_semantic:sstore:25."
                    )
                else:
                    args["evidence_ids"] = normalized_ids[:20]
                    item["args"] = args
            elif tool == "read_function_chunk":
                evidence_id = str(args.get("evidence_id") or "").strip()
                address = str(
                    args.get("address")
                    or args.get("contract")
                    or args.get("callee")
                    or ""
                ).strip()
                function_name = str(
                    args.get("target_function")
                    or args.get("function_name")
                    or args.get("function")
                    or args.get("selector")
                    or ""
                ).strip()
                if evidence_id:
                    if not _looks_like_evidence_id(evidence_id):
                        item["valid"] = False
                        item["error"] = (
                            "read_function_chunk evidence_id must be a concrete packet evidence id "
                            "such as call:14 or critical_call:call:14."
                        )
                elif not (address and function_name):
                    item["valid"] = False
                    item["error"] = (
                        "read_function_chunk requires either evidence_id or both "
                        "address/contract and function_name/function/selector."
                    )
            out.append(item)
        return out

    def _select_views(
        self, packet: Dict[str, Any], view_names: List[str]
    ) -> Dict[str, Any]:
        views = packet.get("views", {})
        selected: Dict[str, Any] = {}
        for name in view_names:
            if name in views:
                selected[name] = views[name]
            else:
                selected[name] = {
                    "available": False,
                    "reason": f"View '{name}' not found in packet",
                }
        return selected

    @staticmethod
    def _error_result(
        tx_hash: str,
        rule: EvolvingRule,
        plan: EvidencePlan,
        error_msg: str,
        started_at: float,
    ) -> Dict[str, Any]:
        return {
            "tx_hash": tx_hash,
            "rule": rule.to_dict(),
            "plan": plan.to_dict(),
            "packet_source": "",
            "judge_results": [],
            "emit_logic": plan.emit_logic,
            "verdict": "uncertain",
            "supporting_evidence_ids": [],
            "runtime_trace": {
                "packet_loaded_from_cache": False,
                "judge_call_count": 0,
                "errors": [{"stage": "packet_load", "error": error_msg}],
                "elapsed_seconds": round(time.time() - started_at, 2),
            },
        }


def _packet_identity_fingerprint(packet: Dict[str, Any]) -> str:
    payload = {
        "transaction_hash": packet.get("transaction_hash") if isinstance(packet, dict) else "",
        "packet_format": packet.get("packet_format") if isinstance(packet, dict) else "",
        "packet_profile": packet.get("packet_profile") if isinstance(packet, dict) else "",
        "packet_source": packet.get("sources") if isinstance(packet, dict) else {},
        "build_config": _packet_identity_build_config(
            packet.get("build_config") if isinstance(packet, dict) else {}
        ),
        "evidence_store": packet.get("evidence_store") if isinstance(packet, dict) else {},
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _packet_identity_build_config(build_config: Any) -> Dict[str, Any]:
    if not isinstance(build_config, dict):
        return {}
    excluded = {
        "include_views",
        "required_views",
        "requested_include_views",
        "requested_required_views",
        "resolved_include_views",
        "dependency_expanded_views",
        "required_views_hash",
        "dependency_policy",
        "selected_view_names",
        "views",
        "view_names",
        "view_count",
        "view_tier_counts",
        "view_cost_score_total",
        "view_cost_breakdown",
        "view_cost_summary",
        "empty_views",
        "heavy_views_present",
        "built_at",
        "created_at",
        "timestamp",
        "updated_at",
    }
    return {
        str(key): value
        for key, value in build_config.items()
        if str(key) not in excluded
    }


def _selected_views_hash(
    packet: Dict[str, Any],
    selected_view_names: List[str],
) -> str:
    views = packet.get("views", {}) if isinstance(packet, dict) else {}
    selected_views = {}
    for view in list(selected_view_names or []):
        if isinstance(views, dict) and view in views:
            selected_views[view] = views.get(view)
        else:
            selected_views[view] = {
                "available": False,
                "reason": f"View '{view}' not found in packet",
            }
    return _selected_views_hash_from_selected(selected_views, selected_view_names)


def _selected_views_hash_from_selected(
    selected_views: Dict[str, Any],
    selected_view_names: List[str],
) -> str:
    payload = {
        "selected_view_names": list(selected_view_names or []),
        "selected_views": dict(selected_views or {}),
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _project_access_control_stateful_selected_views(
    selected: Dict[str, Any],
    *,
    step: JudgeStep,
    attack_label: str,
    state_input_context: Dict[str, Any] | None,
    evidence_tool_registry: EvidenceToolRegistry,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if normalize_attack_label(attack_label, default="") != "access_control":
        return selected, {}
    role = str(getattr(step, "state_prompt_role", "") or "")
    if role != "ac_authorization_gap":
        return selected, {}
    if "source_unavailable_auth_view" not in dict(selected or {}):
        return selected, {}

    state_input = dict(state_input_context or {})
    candidate_ids = _access_control_relevant_candidate_ids(state_input, role=role)
    evidence_ids = _access_control_candidate_evidence_ids(
        state_input,
        candidate_ids=candidate_ids,
    )
    if not evidence_ids:
        return selected, {
            "applied": False,
            "view": "source_unavailable_auth_view",
            "reason": "no_selected_access_control_candidate_evidence",
            "candidate_ids": candidate_ids,
        }
    matched = _access_control_candidate_view_evidence_ids(
        "source_unavailable_auth_view",
        evidence_ids,
        evidence_tool_registry,
    )
    if not matched:
        return selected, {
            "applied": False,
            "view": "source_unavailable_auth_view",
            "reason": "no_source_unavailable_auth_rows_for_candidate",
            "candidate_ids": candidate_ids,
            "candidate_evidence_ids": evidence_ids[:12],
        }

    response = evidence_tool_registry.call(
        "read_packet_view",
        {
            "view": "source_unavailable_auth_view",
            "evidence_ids": matched[:12],
            "limit": max(4, min(24, len(matched))),
        },
    )
    rows = list(((response.get("evidence") or {}).get("rows") or []))
    if not rows:
        return selected, {
            "applied": False,
            "view": "source_unavailable_auth_view",
            "reason": "candidate_filter_returned_no_rows",
            "candidate_ids": candidate_ids,
            "filtered_evidence_ids": matched[:12],
        }

    original_view = dict(selected.get("source_unavailable_auth_view") or {})
    original_rows = list(original_view.get("auth_contexts") or original_view.get("rows") or [])
    projected_view = {
        key: value
        for key, value in original_view.items()
        if key not in {"auth_contexts", "rows"}
    }
    summary = dict(projected_view.get("summary") or {})
    summary.update({
        "candidate_filtered": True,
        "filtered_by_candidate_ids": candidate_ids,
        "filtered_by_evidence_ids": matched[:12],
        "rows_before_filter": len(original_rows),
        "rows_after_filter": len(rows),
    })
    projected_view["summary"] = summary
    projected_view["auth_contexts"] = rows
    updated = dict(selected)
    updated["source_unavailable_auth_view"] = projected_view
    return updated, {
        "applied": True,
        "view": "source_unavailable_auth_view",
        "candidate_ids": candidate_ids,
        "candidate_evidence_ids": evidence_ids[:12],
        "filtered_evidence_ids": matched[:12],
        "rows_before_filter": len(original_rows),
        "rows_after_filter": len(rows),
    }


def _project_price_manipulation_selected_views(
    selected: Dict[str, Any],
    *,
    step: JudgeStep,
    attack_label: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep C2 critical-call evidence tied to price-state consumer anchors."""
    if normalize_attack_label(attack_label, default="") != "price_manipulation":
        return selected, {}
    condition_id = str(step.condition_id or step.id or "").strip().upper()
    if condition_id != "C2":
        return selected, {}

    profile_view = selected.get("market_mechanism_profile_view")
    price_profiles = []
    if isinstance(profile_view, dict):
        price_profiles = [
            dict(profile)
            for profile in list(profile_view.get("profiles") or [])
            if isinstance(profile, dict)
            and str(profile.get("profile_type") or "").strip().lower()
            == "price_state"
        ]

    anchor_ids: List[str] = []
    for profile in price_profiles:
        anchor_ids.extend(profile.get("consumer_evidence_ids") or [])

    argument_rows = selected.get("critical_call_argument_view")
    if isinstance(argument_rows, list):
        for row in argument_rows:
            if not isinstance(row, dict):
                continue
            link = row.get("price_read_consumption_link")
            if not isinstance(link, dict):
                continue
            anchor_ids.extend([
                row.get("evidence_id"),
                link.get("read_evidence_id"),
                link.get("immediate_consumer_evidence_id"),
            ])
    anchor_ids = _unique_strings(anchor_ids)[:16]
    if not price_profiles and not anchor_ids:
        return selected, {
            "applied": False,
            "kind": "price_consumer_view_projection",
            "reason": "no_price_state_profile_or_price_read_link",
        }

    updated = dict(selected)
    projection: Dict[str, Any] = {
        "applied": False,
        "kind": "price_consumer_view_projection",
        "condition_id": condition_id,
        "price_profile_ids": [
            str(profile.get("profile_id") or profile.get("evidence_id") or "")
            for profile in price_profiles
        ],
        "consumer_evidence_ids": anchor_ids,
        "views": {},
    }
    if isinstance(profile_view, dict) and price_profiles:
        projected_profile_view = dict(profile_view)
        projected_profile_view["profiles"] = price_profiles
        projected_profile_view["profile_count"] = len(price_profiles)
        projected_profile_view["projection"] = {
            "profile_type": "price_state",
            "excluded_broader_market_profiles": True,
        }
        updated["market_mechanism_profile_view"] = projected_profile_view
        projection["views"]["market_mechanism_profile_view"] = {
            "rows_before_filter": len(list(profile_view.get("profiles") or [])),
            "rows_after_filter": len(price_profiles),
        }
        projection["applied"] = True

    anchor_set = set(anchor_ids)
    for view_name in ("critical_call_view", "critical_call_argument_view"):
        raw_rows = selected.get(view_name)
        if not isinstance(raw_rows, list) or not anchor_set:
            continue
        filtered_rows = [
            row
            for row in raw_rows
            if isinstance(row, dict)
            and str(row.get("evidence_id") or "") in anchor_set
        ]
        if not filtered_rows:
            continue
        updated[view_name] = filtered_rows
        projection["views"][view_name] = {
            "rows_before_filter": len(raw_rows),
            "rows_after_filter": len(filtered_rows),
        }
        projection["applied"] = True
    return updated, projection


def _project_reentrancy_stateful_selected_views(
    selected: Dict[str, Any],
    *,
    step: JudgeStep,
    attack_label: str,
    state_input_context: Dict[str, Any] | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if normalize_attack_label(attack_label, default="") != "reentrancy":
        return selected, {}
    role = str(getattr(step, "state_prompt_role", "") or "")
    if role not in {
        "re_value_effect",
        "re_state_order_causality",
        "re_candidate_exclusion",
    }:
        return selected, {}
    state_input = dict(state_input_context or {})
    anchor = dict(state_input.get("reentrancy_candidate") or {})
    candidate_ids = _unique_strings(
        [
            *list(anchor.get("selected_candidate_ids") or []),
            *[
                item.get("candidate_id")
                for item in list(anchor.get("candidates") or [])
                if isinstance(item, dict)
            ],
        ]
    )
    if role in {"re_state_order_causality", "re_candidate_exclusion"}:
        value_effect = dict(
            state_input.get("reentrancy_value_effect_summary") or {}
        )
        satisfying = _unique_strings(
            value_effect.get("satisfying_candidate_ids") or []
        )
        if satisfying:
            candidate_ids = satisfying
    if role == "re_candidate_exclusion":
        causal = dict(
            state_input.get("reentrancy_causal_order_summary") or {}
        )
        attack_ids = _unique_strings(causal.get("attack_candidate_ids") or [])
        if attack_ids:
            candidate_ids = attack_ids
    if not candidate_ids:
        return selected, {
            "applied": False,
            "reason": "no_reentrancy_candidate_ids_in_state_input",
            "state_prompt_role": role,
        }

    candidate_id_set = set(candidate_ids)
    updated = dict(selected)
    projected_views: Dict[str, Dict[str, int]] = {}
    for view_name in (
        "reentrancy_candidate_catalog_view",
        "reentrancy_state_order_summary_view",
        "reentrancy_state_order_view",
    ):
        raw_view = selected.get(view_name)
        if not isinstance(raw_view, dict):
            continue
        original_rows = [
            row for row in list(raw_view.get("rows", []) or [])
            if isinstance(row, dict)
        ]
        filtered_rows = [
            row for row in original_rows
            if str(row.get("candidate_id") or "").strip() in candidate_id_set
        ]
        if not filtered_rows:
            continue
        projected = dict(raw_view)
        projected["rows"] = filtered_rows
        summary = dict(projected.get("summary") or {})
        summary.update({
            "candidate_filtered": True,
            "filtered_by_candidate_ids": candidate_ids,
            "rows_before_filter": len(original_rows),
            "rows_after_filter": len(filtered_rows),
        })
        projected["summary"] = summary
        updated[view_name] = projected
        projected_views[view_name] = {
            "rows_before_filter": len(original_rows),
            "rows_after_filter": len(filtered_rows),
        }
    return updated, {
        "applied": bool(projected_views),
        "kind": "reentrancy_candidate_view_projection",
        "state_prompt_role": role,
        "candidate_ids": candidate_ids,
        "views": projected_views,
    }


def _stateful_runtime_enabled(
    rule: EvolvingRule,
    plan: EvidencePlan,
    *,
    iv_stateful_runtime: bool = True,
    access_control_binding_mode: str = "stateful",
    reentrancy_binding_mode: str = "soft",
    disable_stateful_bindings: bool = False,
) -> bool:
    if disable_stateful_bindings:
        return False
    label = normalize_attack_label(
        (rule.metadata or {}).get("attack_label", ""),
        default="",
    )
    stateful = _plan_stateful_runtime_metadata(rule, plan)
    if label == "insufficient_validation":
        return bool(iv_stateful_runtime) and bool(stateful.get("enabled"))
    if label == "access_control":
        return (
            normalize_access_control_binding_mode(
                access_control_binding_mode
            ) == "stateful"
            and bool(stateful.get("enabled"))
        )
    if label == "token_semantic_exploitation":
        return (
            str(stateful.get("mode") or "") == TOKEN_SEMANTIC_STATEFUL_RUNTIME_MODE
            and bool(stateful.get("enabled"))
        )
    if label == "protocol_accounting_exploitation":
        return (
            str(stateful.get("mode") or "") == PROTOCOL_ACCOUNTING_STATEFUL_RUNTIME_MODE
            and _plan_has_stateful_steps(plan)
        )
    if label == "market_manipulation":
        return (
            str(stateful.get("mode") or "") == MARKET_MANIPULATION_STATEFUL_RUNTIME_MODE
            and _plan_has_stateful_steps(plan)
        )
    if label == "flashloans":
        return (
            str(stateful.get("mode") or "") == FLASHLOANS_STATEFUL_RUNTIME_MODE
            and _plan_has_stateful_steps(plan)
        )
    if label == "reentrancy":
        return (
            normalize_reentrancy_binding_mode(reentrancy_binding_mode)
            != "disabled"
            and str(stateful.get("mode") or "")
            == REENTRANCY_STATEFUL_RUNTIME_MODE
            and _plan_has_stateful_steps(plan)
        )
    return False


def disable_stateful_bindings_in_plan(plan: EvidencePlan) -> EvidencePlan:
    """Return a runtime ablation copy with all condition state links removed."""
    ablated = EvidencePlan.from_dict(plan.to_dict())
    for focus_step in list(getattr(ablated, "focus_steps", []) or []):
        focus_step.depends_on = []
    for step in list(getattr(ablated, "judge_steps", []) or []):
        step.depends_on = []
        step.consumes_state_keys = []
        step.produces_state_key = ""
        step.state_prompt_role = ""
        step.state_output_schema = {}
    metadata = dict(ablated.metadata or {})
    original_stateful_keys = [
        key
        for key in (
            "stateful_runtime",
            "access_control_binding_policy",
            "token_semantic_stateful_runtime",
            "protocol_accounting_stateful_runtime",
            "market_manipulation_stateful_runtime",
            "flashloans_stateful_runtime",
            "reentrancy_stateful_runtime",
        )
        if key in metadata
    ]
    for key in original_stateful_keys:
        metadata.pop(key, None)
    metadata["stateful_bindings_disabled"] = True
    metadata["stateful_binding_ablation"] = {
        "enabled": True,
        "removed_fields": [
            "focus_steps.depends_on",
            "judge_steps.depends_on",
            "judge_steps.consumes_state_keys",
            "judge_steps.produces_state_key",
            "judge_steps.state_prompt_role",
            "judge_steps.state_output_schema",
            "judge_result.state_input",
            "judge_result.state_output",
        ],
        "removed_metadata_keys": original_stateful_keys,
    }
    ablated.metadata = metadata
    return ablated


def _plan_stateful_runtime_metadata(
    rule: EvolvingRule,
    plan: EvidencePlan,
) -> Dict[str, Any]:
    label = normalize_attack_label(
        (rule.metadata or {}).get("attack_label", ""),
        default="",
    )
    metadata = dict(plan.metadata or {})
    generic = dict(metadata.get("stateful_runtime") or {})
    if generic:
        return generic
    label_key = {
        "protocol_accounting_exploitation": "protocol_accounting_stateful_runtime",
        "market_manipulation": "market_manipulation_stateful_runtime",
        "flashloans": "flashloans_stateful_runtime",
    }.get(label, "")
    if label_key:
        return dict(metadata.get(label_key) or {})
    return {}


def _plan_has_stateful_steps(plan: EvidencePlan) -> bool:
    for step in list(getattr(plan, "judge_steps", []) or []):
        if (
            str(getattr(step, "state_prompt_role", "") or "").strip()
            or list(getattr(step, "consumes_state_keys", []) or [])
            or str(getattr(step, "produces_state_key", "") or "").strip()
        ):
            return True
    return False


def _plan_declares_stateful_runtime(
    rule: EvolvingRule,
    plan: EvidencePlan,
) -> bool:
    return bool(_plan_stateful_runtime_metadata(rule, plan)) or _plan_has_stateful_steps(plan)


def _state_input_for_step(
    step: JudgeStep,
    state_store: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        key: state_store.get(key)
        for key in list(getattr(step, "consumes_state_keys", []) or [])
        if key in state_store
    }


def _state_binding_audit_for_step(
    step: JudgeStep,
    state_store: Dict[str, Any],
    *,
    state_input: Dict[str, Any],
    stateful_enabled: bool,
) -> Dict[str, Any]:
    consumes = [
        str(key or "").strip()
        for key in list(getattr(step, "consumes_state_keys", []) or [])
        if str(key or "").strip()
    ]
    produces = str(getattr(step, "produces_state_key", "") or "").strip()
    role = str(getattr(step, "state_prompt_role", "") or "").strip()
    if not consumes and not produces and not role:
        return {}
    missing = [key for key in consumes if key not in state_store]
    return {
        "judge_id": step.id,
        "condition_id": step.condition_id or step.id,
        "enabled": bool(stateful_enabled),
        "state_prompt_role": role,
        "consumes_state_keys": consumes,
        "available_state_keys": sorted(state_store.keys()),
        "missing_state_keys": missing,
        "state_input_present": bool(state_input),
        "produces_state_key": produces,
    }


def _step_stateful_runtime_metadata(
    step: JudgeStep,
    *,
    stateful_enabled: bool,
    bindings_disabled: bool = False,
    binding_mode: str = "",
) -> Dict[str, Any]:
    consumes_state_keys = list(getattr(step, "consumes_state_keys", []) or [])
    produces_state_key = str(getattr(step, "produces_state_key", "") or "")
    state_prompt_role = str(getattr(step, "state_prompt_role", "") or "")
    enabled = bool(
        stateful_enabled
        and (state_prompt_role or consumes_state_keys or produces_state_key)
    )
    return {
        "enabled": enabled,
        "bindings_disabled": bool(bindings_disabled),
        "state_prompt_role": state_prompt_role if enabled else "",
        "depends_on": list(getattr(step, "depends_on", []) or []),
        "consumes_state_keys": consumes_state_keys if enabled else [],
        "produces_state_key": produces_state_key if enabled else "",
        "binding_mode": (
            normalize_reentrancy_binding_mode(binding_mode)
            if enabled and state_prompt_role.startswith("re_")
            else ""
        ),
        "state_schema_version": (
            REENTRANCY_STATE_SCHEMA_VERSION
            if enabled and state_prompt_role.startswith("re_")
            else ""
        ),
    }


def _attach_stateful_result_fields(
    result_dict: Dict[str, Any],
    step_trace: Dict[str, Any],
    step: JudgeStep,
    *,
    state_input: Dict[str, Any],
    stateful_runtime: Dict[str, Any],
) -> None:
    if bool((stateful_runtime or {}).get("bindings_disabled")):
        _clear_stateful_binding_result_fields(
            result_dict,
            step_trace,
            stateful_runtime=stateful_runtime,
        )
        return
    if not isinstance(result_dict.get("state_output"), dict):
        result_dict["state_output"] = {}
    result_dict["state_input"] = dict(state_input or {})
    result_dict["stateful_runtime"] = dict(stateful_runtime or {})
    _normalize_access_control_candidate_stateful_result(
        result_dict,
        step=step,
        state_input=state_input,
    )
    _normalize_token_semantic_candidate_stateful_result(
        result_dict,
        step=step,
        state_input=state_input,
    )
    _normalize_reentrancy_candidate_stateful_result(
        result_dict,
        step=step,
        state_input=state_input,
        binding_mode=str((stateful_runtime or {}).get("binding_mode") or ""),
        allowed_candidate_ids=list(
            (stateful_runtime or {}).get("allowed_candidate_ids") or []
        ),
    )
    step_trace["state_input"] = dict(state_input or {})
    step_trace["state_output"] = dict(result_dict.get("state_output") or {})
    step_trace["stateful_runtime"] = dict(stateful_runtime or {})
    if isinstance(result_dict.get("reentrancy_normalization"), dict):
        step_trace["reentrancy_normalization"] = dict(
            result_dict.get("reentrancy_normalization") or {}
        )
    step_trace.setdefault("depends_on", list(getattr(step, "depends_on", []) or []))


def _clear_stateful_binding_result_fields(
    result_dict: Dict[str, Any],
    step_trace: Dict[str, Any],
    *,
    stateful_runtime: Optional[Dict[str, Any]] = None,
) -> None:
    runtime = dict(stateful_runtime or {})
    runtime.setdefault("enabled", False)
    runtime.setdefault("bindings_disabled", True)
    result_dict["state_input"] = {}
    result_dict["state_output"] = {}
    result_dict["stateful_runtime"] = runtime
    step_trace["state_input"] = {}
    step_trace["state_output"] = {}
    step_trace["stateful_runtime"] = runtime
    step_trace["depends_on"] = []


def _normalize_access_control_candidate_stateful_result(
    result_dict: Dict[str, Any],
    *,
    step: JudgeStep,
    state_input: Dict[str, Any],
) -> None:
    role = str(getattr(step, "state_prompt_role", "") or "")
    state_output = result_dict.get("state_output")
    if not isinstance(state_output, dict) or not role.startswith("ac_"):
        return

    if role == "ac_authorization_anchor":
        summary = _access_control_candidate_summary_from_state(state_output)
        if not isinstance(summary, dict):
            return
        parsed_answer = result_dict.get("answer")
        candidates = _unique_candidate_records(
            _access_control_candidate_records(summary),
            limit=3,
        )
        if not candidates and parsed_answer is not True:
            unresolved_candidate = _access_control_unresolved_anchor_candidate(
                result_dict,
                summary,
            )
            if unresolved_candidate:
                candidates = [unresolved_candidate]
                unresolved_ids = _unique_strings(
                    summary.get("unresolved_candidate_ids") or []
                )
                unresolved_ids.append(unresolved_candidate["candidate_id"])
                summary["unresolved_candidate_ids"] = _unique_strings(
                    unresolved_ids
                )
        for candidate in candidates:
            candidate_status = str(
                candidate.get("candidate_status") or ""
            ).strip().lower()
            if candidate_status not in {"candidate_anchor", "unresolved_probe"}:
                # Legacy authority/permission classifications are audit hints,
                # not C1 eligibility decisions. C2/E1 own those semantics.
                candidate["candidate_status"] = (
                    "candidate_anchor" if parsed_answer is True else "unresolved_probe"
                )
        summary["candidates"] = candidates
        summary["chain_candidates"] = candidates
        candidate_ids = _unique_strings(
            item.get("candidate_id") for item in candidates
        )
        candidate_id_set = set(candidate_ids)
        raw_selected = _unique_strings(summary.get("selected_candidate_ids") or [])
        selected = [
            candidate_id for candidate_id in raw_selected
            if not candidate_id_set or candidate_id in candidate_id_set
        ]
        if not selected:
            selected = list(candidate_ids)
        summary["selected_candidate_ids"] = selected[:3]
        probe_ids = _unique_strings(
            item.get("candidate_id")
            for item in candidates
            if str(item.get("candidate_status") or "").lower().strip()
            == "unresolved_probe"
        )[:3]
        summary["probe_candidate_ids"] = probe_ids
        summary["unresolved_candidate_ids"] = _unique_strings([
            *list(summary.get("unresolved_candidate_ids") or []),
            *probe_ids,
        ])[:3]
        summary = _compact_access_control_state_summary(
            summary,
            summary_kind="candidate_anchor",
        )
        state_output[ACCESS_CONTROL_CANDIDATE_STATE_KEY] = summary
        state_output["authorization_chain_summary"] = summary
        return

    anchor = _access_control_candidate_summary_from_state(state_input or {})
    candidate_ids = {
        str(item.get("candidate_id") or "").strip()
        for item in _access_control_candidate_records(anchor)
        if isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
    }
    has_selected_candidate_ids = "selected_candidate_ids" in anchor
    selected_candidate_ids = {
        str(value).strip()
        for value in list(anchor.get("selected_candidate_ids") or [])
        if str(value).strip() in candidate_ids
    }

    if role == "ac_authorization_gap":
        summary = state_output.get("authorization_gap_summary")
        if not isinstance(summary, dict):
            return
        assessments = _candidate_assessments_with_legacy_fallback(summary)
        assessments = [
            item for item in assessments
            if str(item.get("candidate_id") or "").strip() in candidate_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        satisfying = []
        unresolved = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            status = str(item.get("authorization_status") or "").strip().lower()
            if status in {"self-granted", "self granted"}:
                status = "self_granted"
                item["authorization_status"] = status
            evidence_status = str(
                item.get("authorization_evidence_status")
                or item.get("evidence_status")
                or ""
            ).strip().lower()
            if evidence_status in {"source-confirmed-missing", "source confirmed missing"}:
                evidence_status = "source_confirmed_missing"
                item["authorization_evidence_status"] = evidence_status
            if evidence_status in {"missing-behavioral", "behavioral_missing"}:
                evidence_status = "missing_behavioral"
                item["authorization_evidence_status"] = evidence_status
            same_chain = bool(item.get("same_chain_supported"))
            if evidence_status in {"source_required", "source-required", "source_only", "source-only"}:
                unresolved.append(candidate_id)
                continue
            if evidence_status in {"uncertain", "insufficient"}:
                unresolved.append(candidate_id)
                continue
            if evidence_status in {
                "present",
                "contradicted",
                "contradicted_by_legitimate_auth",
            } or status == "present":
                # The Judge owns whether observed authority is legitimate for
                # this candidate. Runtime only normalizes the typed outcome.
                continue
            if (
                same_chain
                and status in {
                    "missing", "bypassed", "self_granted", "unauthorized"
                }
                and (
                    not evidence_status
                    or evidence_status in {
                        "source_confirmed_missing",
                        "missing_behavioral",
                    }
                )
            ):
                satisfying.append(candidate_id)
            elif status == "uncertain" or not status:
                unresolved.append(candidate_id)
        unresolved.extend(sorted(candidate_ids - assessed_ids))
        summary["candidate_assessments"] = assessments
        summary["satisfying_candidate_ids"] = _unique_strings(satisfying)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        summary = _compact_access_control_state_summary(
            summary,
            summary_kind="authorization_gap",
        )
        state_output["authorization_gap_summary"] = summary
        return

    gap = dict((state_input or {}).get("authorization_gap_summary") or {})
    satisfying_gap_ids = {
        str(value).strip()
        for value in list(gap.get("satisfying_candidate_ids") or [])
        if str(value).strip()
    }
    unresolved_gap_ids = {
        str(value).strip()
        for value in list(gap.get("unresolved_candidate_ids") or [])
        if str(value).strip()
    }
    if role == "ac_protected_effect":
        summary = state_output.get("protected_effect_summary")
        if not isinstance(summary, dict):
            return
        effect_target_ids = satisfying_gap_ids | unresolved_gap_ids
        assessments = _candidate_assessments_with_legacy_fallback(summary)
        assessments = [
            item for item in assessments
            if str(item.get("candidate_id") or "").strip() in effect_target_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        attack_ids = []
        unresolved = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            causal = _state_bool(item.get("causal_link_supported"))
            same_chain = _state_bool(item.get("same_chain_supported"))
            beyond_entitlement = _state_bool(item.get("beyond_entitlement_or_contribution"))
            protected_effect = str(item.get("protected_effect") or "").strip()
            effect_status = str(item.get("effect_status") or "").strip().lower()
            if (
                candidate_id in satisfying_gap_ids
                and
                effect_status == "supported"
                and causal
                and same_chain
                and beyond_entitlement
                and protected_effect
            ):
                attack_ids.append(candidate_id)
            elif effect_status in {"uncertain", ""}:
                unresolved.append(candidate_id)
            elif effect_status == "supported" and causal and same_chain and protected_effect:
                unresolved.append(candidate_id)
        unresolved.extend(sorted(effect_target_ids - assessed_ids))
        summary["candidate_assessments"] = assessments
        summary["attack_candidate_ids"] = _unique_strings(attack_ids)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        summary = _compact_access_control_state_summary(
            summary,
            summary_kind="protected_effect",
        )
        state_output["protected_effect_summary"] = summary
        return

    if role == "ac_candidate_exclusion":
        state_key = str(getattr(step, "produces_state_key", "") or "")
        summary = state_output.get(state_key)
        if not state_key or not isinstance(summary, dict):
            return
        effect = dict((state_input or {}).get("protected_effect_summary") or {})
        attack_ids = {
            str(value).strip()
            for value in list(effect.get("attack_candidate_ids") or [])
            if str(value).strip()
        }
        target_ids = set(attack_ids)
        if not target_ids:
            target_ids.update(
                str(value).strip()
                for value in list(gap.get("satisfying_candidate_ids") or [])
                + list(gap.get("unresolved_candidate_ids") or [])
                if str(value).strip()
            )
        if not target_ids:
            target_ids.update(
                selected_candidate_ids
                if has_selected_candidate_ids
                else candidate_ids
            )
        assessments = _candidate_assessments_with_legacy_fallback(summary)
        assessments = [
            item for item in assessments
            if str(item.get("candidate_id") or "").strip() in target_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        excluded = []
        unresolved = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            status = str(item.get("exclusion_status") or "").strip().lower()
            if status == "excluded" and bool(item.get("same_chain_supported")):
                excluded.append(candidate_id)
            elif status in {"uncertain", ""}:
                unresolved.append(candidate_id)
        unresolved.extend(sorted(target_ids - assessed_ids))
        summary["candidate_assessments"] = assessments
        summary["excluded_candidate_ids"] = _unique_strings(excluded)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        summary = _compact_access_control_state_summary(
            summary,
            summary_kind="candidate_exclusion",
        )
        state_output[state_key] = summary


def _normalize_token_semantic_candidate_stateful_result(
    result_dict: Dict[str, Any],
    *,
    step: JudgeStep,
    state_input: Dict[str, Any],
) -> None:
    role = str(getattr(step, "state_prompt_role", "") or "")
    state_output = result_dict.get("state_output")
    if not isinstance(state_output, dict) or not role.startswith("ts_"):
        return

    if role == "ts_token_semantic_anchor":
        summary = _token_semantic_candidate_summary_from_state(state_output)
        if not isinstance(summary, dict):
            return
        candidates = _unique_candidate_records(
            _token_semantic_candidate_records(summary),
            limit=3,
        )
        summary["candidates"] = candidates
        selected = _unique_strings(
            list(summary.get("selected_candidate_ids") or [])
            or [item.get("candidate_id") for item in candidates]
        )
        summary["selected_candidate_ids"] = selected[:3]
        state_output[TOKEN_SEMANTIC_CANDIDATE_STATE_KEY] = summary
        if candidates:
            _set_stateful_answer(result_dict, step, True)
        return

    anchor = _token_semantic_candidate_summary_from_state(state_input or {})
    candidate_ids = {
        str(item.get("candidate_id") or "").strip()
        for item in _token_semantic_candidate_records(anchor)
        if isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
    }

    if role == "ts_semantic_reliance":
        summary = state_output.get("token_semantic_reliance_summary")
        if not isinstance(summary, dict):
            return
        assessments = _candidate_assessments_with_legacy_fallback(summary)
        assessments = [
            item for item in assessments
            if str(item.get("candidate_id") or "").strip() in candidate_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        satisfying = []
        unresolved = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            status = str(item.get("reliance_status") or "").strip().lower()
            if status in {"relied", "consumed"}:
                status = "supported"
                item["reliance_status"] = status
            same_candidate = bool(item.get("same_candidate_supported"))
            token_origin = bool(item.get("token_origin_supported"))
            if status == "supported" and same_candidate and token_origin:
                satisfying.append(candidate_id)
            elif status in {"uncertain", ""} or (
                status == "supported" and not (same_candidate and token_origin)
            ):
                unresolved.append(candidate_id)
        unresolved.extend(sorted(candidate_ids - assessed_ids))
        summary["candidate_assessments"] = assessments
        summary["satisfying_candidate_ids"] = _unique_strings(satisfying)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        state_output["token_semantic_reliance_summary"] = summary
        _set_candidate_summary_answer(result_dict, step, satisfying, unresolved)
        return

    reliance = dict((state_input or {}).get("token_semantic_reliance_summary") or {})
    satisfying_reliance_ids = {
        str(value).strip()
        for value in list(reliance.get("satisfying_candidate_ids") or [])
        if str(value).strip()
    }
    if role == "ts_semantic_outcome":
        summary = state_output.get("token_semantic_outcome_summary")
        if not isinstance(summary, dict):
            return
        assessments = _candidate_assessments_with_legacy_fallback(summary)
        assessments = [
            item for item in assessments
            if str(item.get("candidate_id") or "").strip() in satisfying_reliance_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        attack_ids = []
        unresolved = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            status = str(item.get("outcome_status") or "").strip().lower()
            causal = bool(item.get("causal_link_supported"))
            same_candidate = bool(item.get("same_candidate_supported"))
            outcome_type = str(item.get("outcome_type") or "").strip().lower()
            if (
                status == "supported"
                and causal
                and same_candidate
                and outcome_type
                and outcome_type != "uncertain"
            ):
                attack_ids.append(candidate_id)
            elif status in {"uncertain", ""}:
                unresolved.append(candidate_id)
        unresolved.extend(sorted(satisfying_reliance_ids - assessed_ids))
        summary["candidate_assessments"] = assessments
        summary["attack_candidate_ids"] = _unique_strings(attack_ids)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        state_output["token_semantic_outcome_summary"] = summary
        _set_candidate_summary_answer(result_dict, step, attack_ids, unresolved)
        return

    if role == "ts_candidate_exclusion":
        state_key = str(getattr(step, "produces_state_key", "") or "")
        summary = state_output.get(state_key)
        if not state_key or not isinstance(summary, dict):
            return
        outcome = dict((state_input or {}).get("token_semantic_outcome_summary") or {})
        attack_ids = {
            str(value).strip()
            for value in list(outcome.get("attack_candidate_ids") or [])
            if str(value).strip()
        }
        assessments = _candidate_assessments_with_legacy_fallback(summary)
        assessments = [
            item for item in assessments
            if str(item.get("candidate_id") or "").strip() in attack_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        excluded = []
        unresolved = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            status = str(item.get("exclusion_status") or "").strip().lower()
            if status == "excluded" and bool(item.get("same_candidate_supported")):
                excluded.append(candidate_id)
            elif status in {"uncertain", ""}:
                unresolved.append(candidate_id)
        unresolved.extend(sorted(attack_ids - assessed_ids))
        summary["candidate_assessments"] = assessments
        summary["excluded_candidate_ids"] = _unique_strings(excluded)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        state_output[state_key] = summary
        _set_candidate_summary_answer(result_dict, step, excluded, unresolved)


def _candidate_assessments_with_legacy_fallback(
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    assessments = [
        dict(item) for item in list(summary.get("candidate_assessments") or [])
        if isinstance(item, dict)
    ]
    if assessments:
        return assessments
    selected = str(summary.get("selected_candidate_id") or "").strip()
    return [dict(summary)] if selected else []


def _normalize_reentrancy_candidate_stateful_result(
    result_dict: Dict[str, Any],
    *,
    step: JudgeStep,
    state_input: Dict[str, Any],
    binding_mode: str = "stateful",
    allowed_candidate_ids: List[str] | None = None,
) -> None:
    role = str(getattr(step, "state_prompt_role", "") or "")
    state_output = result_dict.get("state_output")
    if not isinstance(state_output, dict) or not role.startswith("re_"):
        return
    normalized_mode = normalize_reentrancy_binding_mode(binding_mode)
    enforce_answer = normalized_mode == "stateful"
    raw_answer = result_dict.get("answer")
    diagnostics: Dict[str, Any] = {
        "role": role,
        "binding_mode": normalized_mode,
        "state_schema_version": REENTRANCY_STATE_SCHEMA_VERSION,
        "raw_answer": raw_answer,
        "effective_answer": raw_answer,
        "answer_overridden": False,
        "override_reason": "",
        "legacy_fields_migrated": [],
        "dropped_candidate_ids": [],
    }

    if role == "re_reentry_anchor":
        summary = state_output.get("reentrancy_candidate")
        if not isinstance(summary, dict):
            recovered_ids = _reentrancy_candidate_ids_from_result_evidence(
                result_dict,
                allowed_candidate_ids=allowed_candidate_ids,
            )
            if result_dict.get("answer") is True and recovered_ids:
                summary = {
                    "candidates": [
                        {
                            "candidate_id": candidate_id,
                            "candidate_status": "confirmed_reentry",
                            "evidence_ids": [
                                f"reentrancy_candidate_catalog:{candidate_id}"
                            ],
                        }
                        for candidate_id in recovered_ids
                    ],
                    "selected_candidate_ids": list(recovered_ids),
                    "unresolved_candidate_ids": [],
                    "runtime_recovered": True,
                    "recovery_source": "judge_supporting_catalog_evidence",
                }
                state_output["reentrancy_candidate"] = summary
                diagnostics["state_output_recovered"] = True
                diagnostics["recovered_candidate_ids"] = list(recovered_ids)
            else:
                diagnostics["override_reason"] = "missing_reentrancy_candidate_state"
                result_dict["reentrancy_normalization"] = diagnostics
                return
        candidates = _unique_candidate_records(
            [
                dict(item)
                for item in list(summary.get("candidates") or [])
                if isinstance(item, dict)
            ],
            limit=5,
        )
        allowed_ids = set(_unique_strings(allowed_candidate_ids or []))
        if allowed_ids:
            raw_candidate_ids = _unique_strings(
                item.get("candidate_id") for item in candidates
            )
            candidates = [
                item for item in candidates
                if str(item.get("candidate_id") or "").strip() in allowed_ids
            ]
            diagnostics["dropped_candidate_ids"] = [
                candidate_id for candidate_id in raw_candidate_ids
                if candidate_id not in allowed_ids
            ]
            if diagnostics["dropped_candidate_ids"]:
                diagnostics["override_reason"] = (
                    "candidate_ids_not_present_in_packet_catalog"
                )
        candidate_ids = _unique_strings(
            item.get("candidate_id") for item in candidates
        )
        selected = [
            candidate_id
            for candidate_id in _unique_strings(
                list(summary.get("selected_candidate_ids") or []) or candidate_ids
            )
            if candidate_id in set(candidate_ids)
        ]
        unresolved = [
            candidate_id
            for candidate_id in _unique_strings(
                summary.get("unresolved_candidate_ids") or []
            )
            if not candidate_ids or candidate_id in set(candidate_ids)
        ]
        summary["candidates"] = candidates
        summary["selected_candidate_ids"] = selected[:5]
        summary["unresolved_candidate_ids"] = unresolved[:5]
        state_output["reentrancy_candidate"] = summary
        if enforce_answer:
            _set_candidate_summary_answer(result_dict, step, selected, unresolved)
            diagnostics["override_reason"] = "stateful_candidate_anchor_normalization"
        _finish_reentrancy_normalization(result_dict, diagnostics)
        return

    anchor = dict((state_input or {}).get("reentrancy_candidate") or {})
    candidate_ids = {
        str(item.get("candidate_id") or "").strip()
        for item in list(anchor.get("candidates") or [])
        if isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
    }

    if role == "re_value_effect":
        summary = state_output.get("reentrancy_value_effect_summary")
        if not isinstance(summary, dict):
            diagnostics["override_reason"] = "missing_value_effect_state"
            result_dict["reentrancy_normalization"] = diagnostics
            return
        assessments, migrated = _reentrancy_candidate_results_with_diagnostics(
            summary
        )
        diagnostics["legacy_fields_migrated"].extend(migrated)
        raw_assessment_ids = _unique_strings(
            item.get("candidate_id") for item in assessments
        )
        assessments = [
            item
            for item in assessments
            if str(item.get("candidate_id") or "").strip() in candidate_ids
        ]
        diagnostics["dropped_candidate_ids"] = [
            candidate_id for candidate_id in raw_assessment_ids
            if candidate_id not in candidate_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        satisfying: List[str] = []
        unresolved: List[str] = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            status = str(item.get("effect_status") or "").strip().lower()
            same_path = _state_bool(item.get("same_path_supported"))
            if status == "supported" and same_path:
                satisfying.append(candidate_id)
            elif status in {"uncertain", ""} or (
                status == "supported" and not same_path
            ):
                unresolved.append(candidate_id)
        unresolved.extend(sorted(candidate_ids - assessed_ids))
        summary["candidate_results"] = assessments
        summary["satisfying_candidate_ids"] = _unique_strings(satisfying)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        state_output["reentrancy_value_effect_summary"] = summary
        if enforce_answer:
            _set_candidate_summary_answer(result_dict, step, satisfying, unresolved)
            diagnostics["override_reason"] = "stateful_value_effect_normalization"
        _finish_reentrancy_normalization(result_dict, diagnostics)
        return

    value_effect = dict(
        (state_input or {}).get("reentrancy_value_effect_summary") or {}
    )
    satisfying_effect_ids = {
        str(value).strip()
        for value in list(value_effect.get("satisfying_candidate_ids") or [])
        if str(value).strip()
    }
    if role == "re_state_order_causality":
        summary = state_output.get("reentrancy_causal_order_summary")
        if not isinstance(summary, dict):
            diagnostics["override_reason"] = "missing_causal_order_state"
            result_dict["reentrancy_normalization"] = diagnostics
            return
        raw_assessments, migrated = _reentrancy_candidate_results_with_diagnostics(
            summary
        )
        diagnostics["legacy_fields_migrated"].extend(migrated)
        assessments = [
            _canonicalize_reentrancy_causality_result(item, diagnostics)
            for item in raw_assessments
            if str(item.get("candidate_id") or "").strip()
            in satisfying_effect_ids
        ]
        diagnostics["dropped_candidate_ids"] = [
            candidate_id
            for candidate_id in _unique_strings(
                item.get("candidate_id") for item in raw_assessments
            )
            if candidate_id not in satisfying_effect_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        attack_ids: List[str] = []
        safe_order_ids: List[str] = []
        unresolved: List[str] = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            status = str(item.get("causality_status") or "").strip().lower()
            pattern = str(item.get("phase_order_pattern") or "").strip().lower()
            same_candidate = _state_bool(item.get("same_candidate_supported"))
            unfinalized = _state_bool(
                item.get("state_not_finalized_before_external_edge")
            )
            inner_effect = _state_bool(
                item.get("inner_consumption_or_effect_supported")
            )
            repeated_effect = _state_bool(
                item.get("repeated_sensitive_effect_before_return")
            )
            shared_accounting = _state_bool(
                item.get("cross_function_shared_accounting_supported")
            )
            stale_observation = _state_bool(
                item.get("read_only_stale_observation_supported")
            )
            if pattern == "state_updated_before_external_edge":
                safe_order_ids.append(candidate_id)
                continue
            mechanism_supported = (
                (unfinalized and inner_effect)
                or repeated_effect
                or shared_accounting
                or stale_observation
            )
            if status == "supported" and same_candidate and mechanism_supported:
                attack_ids.append(candidate_id)
            elif status in {"uncertain", ""} or (
                status == "supported"
                and not (same_candidate and mechanism_supported)
            ):
                unresolved.append(candidate_id)
        unresolved.extend(sorted(satisfying_effect_ids - assessed_ids))
        summary["candidate_results"] = assessments
        summary["attack_candidate_ids"] = _unique_strings(attack_ids)
        summary["safe_order_candidate_ids"] = _unique_strings(safe_order_ids)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        state_output["reentrancy_causal_order_summary"] = summary
        if enforce_answer:
            _set_candidate_summary_answer(result_dict, step, attack_ids, unresolved)
            diagnostics["override_reason"] = "stateful_causality_normalization"
        _finish_reentrancy_normalization(result_dict, diagnostics)
        return

    if role == "re_candidate_exclusion":
        state_key = str(getattr(step, "produces_state_key", "") or "")
        summary = state_output.get(state_key)
        if not state_key or not isinstance(summary, dict):
            return
        causal = dict(
            (state_input or {}).get("reentrancy_causal_order_summary") or {}
        )
        attack_ids = {
            str(value).strip()
            for value in list(causal.get("attack_candidate_ids") or [])
            if str(value).strip()
        }
        raw_assessments, migrated = _reentrancy_candidate_results_with_diagnostics(
            summary
        )
        diagnostics["legacy_fields_migrated"].extend(migrated)
        assessments = [
            item
            for item in raw_assessments
            if str(item.get("candidate_id") or "").strip() in attack_ids
        ]
        assessed_ids = {
            str(item.get("candidate_id") or "").strip() for item in assessments
        }
        excluded: List[str] = []
        unresolved: List[str] = []
        for item in assessments:
            candidate_id = str(item.get("candidate_id") or "").strip()
            status = str(item.get("exclusion_status") or "").strip().lower()
            if status == "excluded" and _state_bool(
                item.get("same_candidate_supported")
            ):
                excluded.append(candidate_id)
            elif status in {"uncertain", ""}:
                unresolved.append(candidate_id)
        unresolved.extend(sorted(attack_ids - assessed_ids))
        summary["candidate_results"] = assessments
        summary["excluded_candidate_ids"] = _unique_strings(excluded)
        summary["unresolved_candidate_ids"] = _unique_strings(unresolved)
        state_output[state_key] = summary
        if enforce_answer:
            _set_candidate_summary_answer(result_dict, step, excluded, unresolved)
            diagnostics["override_reason"] = "stateful_exclusion_normalization"
        _finish_reentrancy_normalization(result_dict, diagnostics)


def _reentrancy_candidate_ids_from_result_evidence(
    result_dict: Dict[str, Any],
    *,
    allowed_candidate_ids: List[str] | None = None,
) -> List[str]:
    prefix = "reentrancy_candidate_catalog:"
    recovered = []
    for evidence_id in list(result_dict.get("supporting_evidence_ids") or []):
        text = str(evidence_id or "").strip()
        if text.startswith(prefix):
            recovered.append(text[len(prefix):])
    allowed = set(_unique_strings(allowed_candidate_ids or []))
    recovered = _unique_strings(recovered)
    if allowed:
        recovered = [candidate_id for candidate_id in recovered if candidate_id in allowed]
    return recovered[:5]


def _reentrancy_candidate_results(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    results, _ = _reentrancy_candidate_results_with_diagnostics(summary)
    return results


def _reentrancy_candidate_results_with_diagnostics(
    summary: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[str]]:
    migrated: List[str] = []
    raw = summary.get("candidate_results")
    if not isinstance(raw, list):
        for legacy_key in (
            "candidate_assessments",
            "candidate_assessessions",
            "candidate_assesssessions",
        ):
            legacy = summary.get(legacy_key)
            if isinstance(legacy, list):
                raw = legacy
                migrated.append(f"{legacy_key}->candidate_results")
                break
    return (
        [dict(item) for item in list(raw or []) if isinstance(item, dict)],
        migrated,
    )


def _canonicalize_reentrancy_causality_result(
    item: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(item)
    migrated = diagnostics.setdefault("legacy_fields_migrated", [])
    if not str(normalized.get("causality_status") or "").strip():
        legacy_status = str(normalized.get("order_status") or "").strip().lower()
        if legacy_status:
            normalized["causality_status"] = (
                "absent" if legacy_status == "safe_order" else legacy_status
            )
            migrated.append("order_status->causality_status")
    if not str(normalized.get("phase_order_pattern") or "").strip():
        legacy_pattern = str(normalized.get("order_pattern") or "").strip()
        if legacy_pattern:
            normalized["phase_order_pattern"] = legacy_pattern
            migrated.append("order_pattern->phase_order_pattern")
    causal_link = _state_bool(normalized.get("causal_link_supported"))
    if causal_link:
        if "state_not_finalized_before_external_edge" not in normalized:
            normalized["state_not_finalized_before_external_edge"] = True
        if "inner_consumption_or_effect_supported" not in normalized:
            normalized["inner_consumption_or_effect_supported"] = True
        migrated.append("causal_link_supported->canonical_causality_flags")
    pattern = str(normalized.get("phase_order_pattern") or "").strip().lower()
    if "outer" in pattern and "after" in pattern and "callback" in pattern:
        normalized.setdefault("outer_protective_write_after_callback", True)
    normalized.setdefault("mechanism_type", "uncertain")
    normalized.setdefault("phase_order_pattern", "")
    normalized.setdefault("state_not_finalized_before_external_edge", False)
    normalized.setdefault("inner_consumption_or_effect_supported", False)
    normalized.setdefault("outer_protective_write_after_callback", False)
    normalized.setdefault("repeated_sensitive_effect_before_return", False)
    normalized.setdefault("cross_function_shared_accounting_supported", False)
    normalized.setdefault("read_only_stale_observation_supported", False)
    normalized.setdefault("same_candidate_supported", False)
    normalized.setdefault("evidence_ids", [])
    return normalized


def _finish_reentrancy_normalization(
    result_dict: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> None:
    diagnostics["legacy_fields_migrated"] = _unique_strings(
        diagnostics.get("legacy_fields_migrated") or []
    )
    diagnostics["effective_answer"] = result_dict.get("answer")
    diagnostics["answer_overridden"] = (
        diagnostics.get("raw_answer") != diagnostics.get("effective_answer")
    )
    if not diagnostics["answer_overridden"]:
        diagnostics["override_reason"] = ""
    result_dict["reentrancy_normalization"] = diagnostics


def _state_bool(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "yes", "1", "supported"}


def _access_control_candidate_summary_from_state(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    summary = state.get(ACCESS_CONTROL_CANDIDATE_STATE_KEY)
    if isinstance(summary, dict):
        return dict(summary)
    legacy = state.get("authorization_chain_summary")
    if isinstance(legacy, dict):
        return dict(legacy)
    return {}


def _access_control_candidate_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    raw = summary.get("candidates")
    if not isinstance(raw, list):
        raw = summary.get("chain_candidates")
    return [dict(item) for item in list(raw or []) if isinstance(item, dict)]


def _token_semantic_candidate_summary_from_state(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    summary = state.get(TOKEN_SEMANTIC_CANDIDATE_STATE_KEY)
    return dict(summary) if isinstance(summary, dict) else {}


def _token_semantic_candidate_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    raw = summary.get("candidates")
    return [dict(item) for item in list(raw or []) if isinstance(item, dict)]


def _unique_candidate_records(value: Any, *, limit: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(value or []):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        records.append(dict(item))
        if len(records) >= max(1, int(limit or 1)):
            break
    return records


def _compact_access_control_state_summary(
    summary: Dict[str, Any],
    *,
    summary_kind: str,
) -> Dict[str, Any]:
    """Keep AC depends-on state small while preserving chain identity."""
    if not isinstance(summary, dict):
        return {}
    out: Dict[str, Any] = {
        "schema_version": "evotx.access_control_state.compact.v1",
        "summary_kind": summary_kind,
    }
    list_keys = {
        "selected_candidate_ids",
        "probe_candidate_ids",
        "satisfying_candidate_ids",
        "unresolved_candidate_ids",
        "attack_candidate_ids",
        "excluded_candidate_ids",
    }
    for key in list_keys:
        if key in summary:
            out[key] = _unique_strings(summary.get(key) or [])[:8]

    binding_confidence = str(summary.get("binding_confidence") or "").strip()
    if binding_confidence:
        out["binding_confidence"] = binding_confidence
    unresolved_fields = _unique_strings(summary.get("unresolved_fields") or [])[:8]
    if unresolved_fields:
        out["unresolved_fields"] = [
            _truncate_text(value, 220) for value in unresolved_fields
        ]

    candidates = _unique_candidate_records(
        _access_control_candidate_records(summary),
        limit=3,
    )
    if candidates:
        compact_candidates = [
            _compact_access_control_candidate_record(item)
            for item in candidates
        ]
        out["candidates"] = compact_candidates
        out["chain_candidates"] = compact_candidates

    assessments = [
        _compact_access_control_candidate_assessment(item)
        for item in list(summary.get("candidate_assessments") or [])
        if isinstance(item, dict)
    ][:8]
    if assessments:
        out["candidate_assessments"] = assessments

    for key in (
        "reason",
        "rationale",
        "summary",
        "limitation",
        "missing_evidence_summary",
    ):
        value = summary.get(key)
        if value:
            out[key] = _truncate_text(str(value), 260)
    return out


def _access_control_unresolved_anchor_candidate(
    result_dict: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep a low-confidence entry anchor without converting a C1 false to true."""
    unresolved_fields = _unique_strings(summary.get("unresolved_fields") or [])
    missing_evidence = _unique_strings(result_dict.get("missing_evidence") or [])
    feature_analysis = dict(result_dict.get("condition_feature_analysis") or {})
    partial_features = _unique_strings(
        feature_analysis.get("partial_match_features") or []
    )
    boundary_features = _unique_strings(
        feature_analysis.get("boundary_features") or []
    )
    unresolved_text = " ".join(
        [
            *unresolved_fields,
            *missing_evidence,
            *partial_features,
            *boundary_features,
            str(result_dict.get("reason") or ""),
        ]
    ).lower()
    unresolved_semantic_cues = (
        "unknown selector",
        "selector is unknown",
        "undecoded selector",
        "selector is undecoded",
        "unresolved selector",
        "unresolved call",
        "unresolved semantic",
        "state write",
        "storage write",
        "truncated",
        "omitted",
        "source unavailable",
        "source_unavailable",
    )
    if not any(cue in unresolved_text for cue in unresolved_semantic_cues):
        return {}

    evidence_ids = _unique_strings(
        list(result_dict.get("supporting_evidence_ids") or [])
        + list(result_dict.get("contradicting_evidence_ids") or [])
    )
    call_evidence_ids = [
        value for value in evidence_ids if _call_id_suffix(value)
    ]
    if not call_evidence_ids:
        return {}

    entry_evidence_id = call_evidence_ids[0]
    entry_suffix = _call_id_suffix(entry_evidence_id) or "unknown"
    candidate: Dict[str, Any] = {
        "candidate_id": f"ac_unresolved_call_{entry_suffix}",
        "candidate_status": "unresolved_probe",
        "probe_reason": _truncate_text(unresolved_text, 220),
        "entry_call_id": entry_evidence_id,
        "entry_call_evidence_id": entry_evidence_id,
        "capability_type": "unresolved_sensitive_capability",
        "auth_surface": "unresolved",
        "confidence": "low",
        "evidence_ids": call_evidence_ids[:12],
        "supporting_evidence_ids": call_evidence_ids[:12],
    }
    if partial_features:
        candidate["effect_hint"] = _truncate_text("; ".join(partial_features), 220)
    return candidate


def _compact_access_control_candidate_record(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "candidate_id",
        "candidate_status",
        "probe_reason",
        "entry_call_id",
        "entry_call_evidence_id",
        "entry_function",
        "entry_actor",
        "sensitive_call_id",
        "sensitive_call_evidence_id",
        "function",
        "target_contract",
        "capability_type",
        "sensitive_capability",
        "protected_target_or_resource",
        "actor",
        "beneficiary",
        "expected_authority_type",
        "auth_surface",
        "effect_hint",
        "entry_to_sensitive_relation",
        "confidence",
    ):
        value = item.get(key)
        if value not in (None, "", []):
            out[key] = _truncate_text(str(value), 180)
    for key in (
        "path_ids",
        "evidence_ids",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
    ):
        values = _unique_strings(item.get(key) or [])[:12]
        if values:
            out[key] = values
    return out


def _compact_access_control_candidate_assessment(
    item: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "candidate_id",
        "required_authority",
        "sensitive_capability",
        "protected_target_or_resource",
        "authorization_status",
        "authorization_evidence_status",
        "evidence_status",
        "entry_authorization_status",
        "sensitive_operation_guard_status",
        "authority_source_type",
        "authority_controller",
        "authority_provenance",
        "self_authorization_status",
        "normalization_reason",
        "protected_effect",
        "effect_status",
        "exclusion_status",
        "confidence",
    ):
        value = item.get(key)
        if value not in (None, "", []):
            out[key] = _truncate_text(str(value), 180)
    for key in (
        "same_chain_supported",
        "causal_link_supported",
        "beyond_entitlement_or_contribution",
    ):
        if key in item:
            out[key] = bool(item.get(key))
    for key in (
        "path_ids",
        "evidence_ids",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "missing_evidence",
    ):
        values = _unique_strings(item.get(key) or [])[:12]
        if values:
            out[key] = values
    for key in ("reason", "rationale", "limitation"):
        value = item.get(key)
        if value:
            out[key] = _truncate_text(str(value), 220)
    return out


def _set_candidate_summary_answer(
    result_dict: Dict[str, Any],
    step: JudgeStep,
    satisfying_ids: List[str],
    unresolved_ids: List[str],
) -> None:
    if satisfying_ids:
        answer: Any = True
    elif unresolved_ids:
        answer = "uncertain"
    else:
        answer = False
    _set_stateful_answer(result_dict, step, answer)


def _set_stateful_answer(
    result_dict: Dict[str, Any],
    step: JudgeStep,
    answer: Any,
) -> None:
    result_dict["answer"] = answer
    result_dict["satisfied"] = (
        False if isinstance(answer, str) else answer == bool(step.expected_answer)
    )


def _access_control_candidate_emit_decision(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stateful = dict((plan.metadata or {}).get("stateful_runtime") or {})
    if (
        str(stateful.get("mode") or "") != ACCESS_CONTROL_STATEFUL_RUNTIME_MODE
        or not stateful.get("candidate_branching")
    ):
        return {"enabled": False}

    latest_state: Dict[str, Any] = {}
    results_by_id: Dict[str, Dict[str, Any]] = {}
    unresolved: List[str] = []
    for result in list(judge_results or []):
        result_id = str(
            result.get("condition_id") or result.get("id") or ""
        ).strip()
        if result_id:
            results_by_id[result_id] = result
        output = result.get("state_output")
        if isinstance(output, dict):
            latest_state.update(output)
    effect = dict(latest_state.get("protected_effect_summary") or {})
    attack_ids = _unique_strings(list(effect.get("attack_candidate_ids") or []))
    if not attack_ids and not isinstance(effect.get("candidate_assessments"), list):
        return {
            "enabled": True,
            "enforce_emit": True,
            "state_status": "unresolved",
            "reason": "candidate_state_unavailable",
            "attack_candidate_ids": [],
            "excluded_candidate_ids": [],
            "remaining_attack_candidate_ids": [],
            "unresolved_candidate_ids": ["candidate_state_unavailable"],
            "should_emit": False,
        }
    unresolved.extend(list(effect.get("unresolved_candidate_ids") or []))

    role_steps = dict(stateful.get("role_steps") or {})
    core_results = [
        results_by_id[step_id]
        for step_id in [
            str(role_steps.get("authorization_anchor") or ""),
            str(role_steps.get("authorization_gap") or ""),
            str(role_steps.get("protected_effect") or ""),
        ]
        if step_id and step_id in results_by_id
    ]
    if any(result.get("answer") is False for result in core_results):
        attack_ids = []
        unresolved = []
    elif any(_answer_is_uncertain(result.get("answer")) for result in core_results):
        unresolved.extend(attack_ids)
        attack_ids = []

    excluded: List[str] = []
    exclusion_keys = list(stateful.get("candidate_exclusion_state_keys") or [])
    unresolved_by_exclusion: List[str] = []
    for key in exclusion_keys:
        raw_summary = latest_state.get(str(key))
        if not isinstance(raw_summary, dict):
            unresolved_by_exclusion.extend(attack_ids)
            continue
        summary = dict(raw_summary)
        exclusion_result = next(
            (
                result
                for result in list(judge_results or [])
                if isinstance(result.get("state_output"), dict)
                and key in result.get("state_output", {})
            ),
            {},
        )
        exclusion_answer = exclusion_result.get("answer")
        if exclusion_answer is True:
            excluded.extend(list(summary.get("excluded_candidate_ids") or []))
            unresolved_by_exclusion.extend(
                list(summary.get("unresolved_candidate_ids") or [])
            )
        elif _answer_is_uncertain(exclusion_answer):
            unresolved_by_exclusion.extend(attack_ids)
    excluded_ids = _unique_strings(excluded)
    unresolved.extend(unresolved_by_exclusion)
    unresolved_ids = _unique_strings(unresolved)
    remaining = [
        candidate_id for candidate_id in attack_ids
        if candidate_id not in excluded_ids and candidate_id not in unresolved_ids
    ]
    return {
        "enabled": True,
        # Judge Boolean answers own local semantics. Runtime only binds those
        # answers to candidate IDs before transaction-level aggregation.
        "enforce_emit": True,
        "semantic_authority": "judge_boolean_emit_logic",
        "state_status": "resolved",
        "policy": "same_candidate_core_then_candidate_local_exclusions_or",
        "attack_candidate_ids": attack_ids,
        "excluded_candidate_ids": excluded_ids,
        "remaining_attack_candidate_ids": remaining,
        "unresolved_candidate_ids": unresolved_ids,
        "should_emit": bool(remaining),
    }


def _token_semantic_candidate_emit_decision(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stateful = dict((plan.metadata or {}).get("stateful_runtime") or {})
    if (
        str(stateful.get("mode") or "") != TOKEN_SEMANTIC_STATEFUL_RUNTIME_MODE
        or not stateful.get("candidate_branching")
    ):
        return {"enabled": False}

    latest_state: Dict[str, Any] = {}
    unresolved: List[str] = []
    for result in list(judge_results or []):
        output = result.get("state_output")
        if isinstance(output, dict):
            latest_state.update(output)
    outcome = dict(latest_state.get("token_semantic_outcome_summary") or {})
    attack_ids = _unique_strings(list(outcome.get("attack_candidate_ids") or []))
    if not attack_ids and not isinstance(outcome.get("candidate_assessments"), list):
        return {
            "enabled": True,
            "state_status": "unresolved",
            "reason": "candidate_state_unavailable",
            "attack_candidate_ids": [],
            "excluded_candidate_ids": [],
            "remaining_attack_candidate_ids": [],
            "unresolved_candidate_ids": ["candidate_state_unavailable"],
            "should_emit": False,
        }
    unresolved.extend(list(outcome.get("unresolved_candidate_ids") or []))

    excluded: List[str] = []
    unresolved_by_exclusion: List[str] = []
    for key in list(stateful.get("candidate_exclusion_state_keys") or []):
        raw_summary = latest_state.get(str(key))
        if not isinstance(raw_summary, dict):
            unresolved_by_exclusion.extend(attack_ids)
            continue
        summary = dict(raw_summary)
        excluded.extend(list(summary.get("excluded_candidate_ids") or []))
        unresolved_by_exclusion.extend(
            list(summary.get("unresolved_candidate_ids") or [])
        )
    excluded_ids = _unique_strings(excluded)
    unresolved.extend(unresolved_by_exclusion)
    unresolved_ids = _unique_strings(unresolved)
    remaining = [
        candidate_id for candidate_id in attack_ids
        if candidate_id not in excluded_ids and candidate_id not in unresolved_ids
    ]
    return {
        "enabled": True,
        "policy": "same_token_semantic_candidate_core_then_candidate_local_exclusions_or",
        "state_status": "resolved",
        "attack_candidate_ids": attack_ids,
        "excluded_candidate_ids": excluded_ids,
        "remaining_attack_candidate_ids": remaining,
        "unresolved_candidate_ids": unresolved_ids,
        "should_emit": bool(remaining),
    }


def _reentrancy_candidate_emit_decision(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stateful = dict((plan.metadata or {}).get("stateful_runtime") or {})
    if (
        str(stateful.get("mode") or "") != REENTRANCY_STATEFUL_RUNTIME_MODE
        or normalize_reentrancy_binding_mode(
            str(stateful.get("binding_mode") or "")
        ) != "stateful"
        or not stateful.get("candidate_branching")
    ):
        return {
            "enabled": False,
            "reason": "reentrancy_candidate_emit_requires_stateful_binding",
            "binding_mode": normalize_reentrancy_binding_mode(
                str(stateful.get("binding_mode") or "")
            ),
        }

    latest_state: Dict[str, Any] = {}
    for result in list(judge_results or []):
        output = result.get("state_output")
        if isinstance(output, dict):
            latest_state.update(output)
    causal = dict(latest_state.get("reentrancy_causal_order_summary") or {})
    candidate_results = _reentrancy_candidate_results(causal)
    assessed_ids = {
        str(item.get("candidate_id") or "").strip()
        for item in candidate_results
        if str(item.get("candidate_id") or "").strip()
    }
    attack_ids = [
        candidate_id
        for candidate_id in _unique_strings(
            list(causal.get("attack_candidate_ids") or [])
        )
        if not assessed_ids or candidate_id in assessed_ids
    ]
    if not attack_ids and not candidate_results:
        return {
            "enabled": True,
            "state_status": "unresolved",
            "reason": "candidate_state_unavailable",
            "attack_candidate_ids": [],
            "excluded_candidate_ids": [],
            "remaining_attack_candidate_ids": [],
            "unresolved_candidate_ids": ["candidate_state_unavailable"],
            "should_emit": False,
        }

    excluded: List[str] = []
    unresolved: List[str] = list(causal.get("unresolved_candidate_ids") or [])
    for key in list(stateful.get("candidate_exclusion_state_keys") or []):
        raw_summary = latest_state.get(str(key))
        if not isinstance(raw_summary, dict):
            unresolved.extend(attack_ids)
            continue
        excluded.extend(list(raw_summary.get("excluded_candidate_ids") or []))
        unresolved.extend(list(raw_summary.get("unresolved_candidate_ids") or []))
    excluded_ids = _unique_strings(excluded)
    unresolved_ids = _unique_strings(unresolved)
    remaining = [
        candidate_id
        for candidate_id in attack_ids
        if candidate_id not in excluded_ids and candidate_id not in unresolved_ids
    ]
    return {
        "enabled": True,
        "policy": "same_reentrancy_candidate_core_then_candidate_local_exclusions_or",
        "state_status": "resolved",
        "attack_candidate_ids": attack_ids,
        "excluded_candidate_ids": excluded_ids,
        "remaining_attack_candidate_ids": remaining,
        "unresolved_candidate_ids": unresolved_ids,
        "should_emit": bool(remaining),
    }


def _stateful_candidate_emit_decision(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reentrancy_emit = _reentrancy_candidate_emit_decision(
        plan=plan,
        judge_results=judge_results,
    )
    if reentrancy_emit.get("enabled"):
        return reentrancy_emit
    token_semantic_emit = _token_semantic_candidate_emit_decision(
        plan=plan,
        judge_results=judge_results,
    )
    if token_semantic_emit.get("enabled"):
        return token_semantic_emit
    return _access_control_candidate_emit_decision(
        plan=plan,
        judge_results=judge_results,
    )


def _update_state_store_from_result(
    state_store: Dict[str, Any],
    step: JudgeStep,
    result_dict: Dict[str, Any],
) -> None:
    key = str(getattr(step, "produces_state_key", "") or "")
    if not key:
        return
    state_output = result_dict.get("state_output")
    if not isinstance(state_output, dict):
        return
    produced = state_output.get(key)
    if produced is not None:
        state_store[key] = produced


def _parallel_judge_config(
    *,
    requested: bool,
    judge_concurrency: int,
    early_stop: bool,
    rate_limit_serial_fallback_active: bool = False,
) -> Dict[str, Any]:
    concurrency = max(1, int(judge_concurrency or 1))
    effective = (
        bool(requested)
        and concurrency > 1
        and not bool(early_stop)
        and not bool(rate_limit_serial_fallback_active)
    )
    disabled_reason = ""
    if requested and concurrency <= 1:
        disabled_reason = "judge_concurrency_le_1"
    if requested and early_stop:
        disabled_reason = "early_stop_enabled"
    if requested and rate_limit_serial_fallback_active:
        disabled_reason = "rate_limit_serial_fallback_active"
    return {
        "enabled": effective,
        "requested": bool(requested),
        "effective": effective,
        "judge_concurrency": concurrency,
        "effective_concurrency_after_rate_limit": (
            1 if rate_limit_serial_fallback_active else concurrency
        ),
        "rate_limit_serial_fallback_active_at_start": bool(
            rate_limit_serial_fallback_active
        ),
        "disabled_reason": disabled_reason,
        "submitted_step_count": 0,
        "completed_step_count": 0,
        "failed_step_count": 0,
        "elapsed_parallel_seconds": 0.0,
        "log_format": "jsonl",
        "structured_log_count": 0,
        "max_inflight_observed": 0,
        "completed_order": [],
        "result_order": [],
    }


def emit_judge_log(
    *,
    tx_hash: str,
    judge_id: str,
    condition_id: str,
    phase: str,
    message: str = "",
    level: str = "INFO",
    **fields,
) -> Dict[str, Any]:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": str(level or "INFO"),
        "component": "PacketRuntime",
        "tx_hash": str(tx_hash or ""),
        "judge_id": str(judge_id or ""),
        "condition_id": str(condition_id or judge_id or ""),
        "phase": str(phase or ""),
        "message": str(message or ""),
    }
    event.update(fields)
    return event


def _print_structured_judge_log(event: Dict[str, Any]) -> None:
    with _STRUCTURED_LOG_PRINT_LOCK:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True))


class _LockedSourceRegistry:
    def __init__(self, source_registry: Any, lock: threading.Lock):
        self._source_registry = source_registry
        self._lock = lock

    def call(self, request: Dict[str, Any], default_chain: str = "eth") -> Dict[str, Any]:
        with self._lock:
            return self._source_registry.call(request, default_chain=default_chain)


def _merge_judge_reuse_stats(target: Dict[str, Any], local: Dict[str, Any]) -> None:
    target["enabled"] = bool(target.get("enabled")) or bool(local.get("enabled"))
    for key in ("attempted", "reused", "skipped"):
        target[key] = int(target.get(key, 0) or 0) + int(local.get(key, 0) or 0)
    target_skipped = target.setdefault("skipped_by_reason", {})
    for reason, count in dict(local.get("skipped_by_reason") or {}).items():
        target_skipped[str(reason)] = int(target_skipped.get(str(reason), 0) or 0) + int(count or 0)
    target.setdefault("reused_judge_ids", []).extend(
        list(local.get("reused_judge_ids", []) or [])
    )
    target.setdefault("skipped_judge_ids", []).extend(
        list(local.get("skipped_judge_ids", []) or [])
    )


def _judge_result_dict_from_trace(
    result: JudgeResult,
    step: JudgeStep,
    step_trace: Dict[str, Any],
) -> Dict[str, Any]:
    result_dict = result.to_dict()
    result_dict["condition_id"] = step.condition_id or step.id
    result_dict["tool_calls"] = list(step_trace.get("tool_calls", []))
    result_dict["judge_rounds"] = list(step_trace.get("judge_calls", []))
    result_dict["judge_output_exhaustions"] = list(
        step_trace.get("judge_output_exhaustions", [])
    )
    result_dict["source_decisiveness_guards"] = list(
        step_trace.get("source_decisiveness_guards", [])
    )
    result_dict["view_render_metadata"] = list(step_trace.get("view_render_metadata", []))
    result_dict["selected_view_names"] = list(step_trace.get("selected_view_names", []))
    result_dict["selected_views_hash"] = str(step_trace.get("selected_views_hash", ""))
    result_dict["all_missing_evidence_debug"] = list(
        step_trace.get("all_missing_evidence_debug", [])
    )
    result_dict["ignored_tool_requests"] = list(step_trace.get("ignored_tool_requests", []))
    result_dict["invalid_evidence_ids"] = list(step_trace.get("invalid_evidence_ids", []))
    result_dict["evidence_id_validations"] = list(
        step_trace.get("evidence_id_validations", [])
    )
    result_dict["state_input"] = dict(step_trace.get("state_input", {}) or {})
    result_dict["state_output"] = dict(
        result_dict.get("state_output")
        or step_trace.get("state_output")
        or {}
    )
    result_dict["stateful_runtime"] = dict(
        result_dict.get("stateful_runtime")
        or step_trace.get("stateful_runtime")
        or {}
    )
    return result_dict


def _parallel_error_judge_result(step: JudgeStep, exc: Exception) -> Dict[str, Any]:
    error_class = _judge_exception_class(exc)
    return {
        "id": step.id,
        "judge_id": step.id,
        "condition_id": step.condition_id or step.id,
        "question": step.question,
        "answer": "uncertain",
        "expected_answer": step.expected_answer,
        "satisfied": False,
        "reason": f"Parallel judge worker failed: {exc!r}",
        "confidence": "low",
        "evidence_refs": list(step.default_evidence_refs or step.evidence_refs or []),
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "missing_evidence": ["judge_transport_error", error_class],
        "tool_requests": [],
        "tool_calls": [],
        "condition_feature_analysis": normalize_condition_feature_analysis({}),
        "judge_rounds": [],
        "view_render_metadata": [],
        "selected_view_names": [],
        "selected_views_hash": "",
        "state_input": {},
        "state_output": {},
        "stateful_runtime": {},
        "judge_error": {
            "class": error_class,
            "message": repr(exc),
            "do_not_train": True,
            "parallel": True,
        },
        "do_not_train": True,
    }


def _judge_exception_class(exc: Exception) -> str:
    text = repr(exc)
    lowered = text.lower()
    if "contentfilter" in lowered or "content filter" in lowered or "1301" in lowered:
        return "content_filter"
    if "ratelimit" in lowered or "rate limit" in lowered or "429" in lowered:
        return "rate_limit"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "json" in lowered or "parse" in lowered:
        return "parse_error"
    return type(exc).__name__ or "judge_exception"


def _serial_error_judge_result(
    step: JudgeStep,
    exc: Exception,
    *,
    state_input: Dict[str, Any],
    stateful_runtime: Dict[str, Any],
) -> Dict[str, Any]:
    error_class = _judge_exception_class(exc)
    return {
        "id": step.id,
        "judge_id": step.id,
        "condition_id": step.condition_id or step.id,
        "question": step.question,
        "answer": "uncertain",
        "expected_answer": step.expected_answer,
        "satisfied": False,
        "reason": f"Judge call failed before a valid evidence judgment: {error_class}.",
        "confidence": "low",
        "evidence_refs": list(step.default_evidence_refs or step.evidence_refs or []),
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "missing_evidence": ["judge_transport_error", error_class],
        "tool_requests": [],
        "tool_calls": [],
        "condition_feature_analysis": normalize_condition_feature_analysis({}),
        "judge_rounds": [],
        "view_render_metadata": [],
        "selected_view_names": [],
        "selected_views_hash": "",
        "state_input": dict(state_input or {}),
        "state_output": {},
        "stateful_runtime": dict(stateful_runtime or {}),
        "judge_error": {
            "class": error_class,
            "message": repr(exc),
            "do_not_train": True,
        },
        "do_not_train": True,
    }


def _serial_error_step_trace(
    step: JudgeStep,
    exc: Exception,
    *,
    state_input: Dict[str, Any],
    stateful_runtime: Dict[str, Any],
) -> Dict[str, Any]:
    error_class = _judge_exception_class(exc)
    return {
        "judge_id": step.id,
        "condition_id": step.condition_id or step.id,
        "judge_calls": [],
        "tool_calls": [],
        "final_answer": "uncertain",
        "state_input": dict(state_input or {}),
        "state_output": {},
        "stateful_runtime": dict(stateful_runtime or {}),
        "errors": [{
            "stage": "judge",
            "error": repr(exc),
            "error_class": error_class,
            "do_not_train": True,
        }],
        "judge_error": {
            "class": error_class,
            "message": repr(exc),
            "do_not_train": True,
        },
    }


def _parallel_error_step_trace(
    step: JudgeStep,
    exc: Exception,
    structured_logs: List[Dict[str, Any]],
    duration_seconds: float,
) -> Dict[str, Any]:
    error_class = _judge_exception_class(exc)
    return {
        "judge_id": step.id,
        "condition_id": step.condition_id or step.id,
        "judge_calls": [],
        "tool_calls": [],
        "structured_logs": list(structured_logs or []),
        "final_answer": "uncertain",
        "state_input": {},
        "state_output": {},
        "stateful_runtime": {},
        "parallel_judge": {
            "enabled": True,
            "duration_seconds": duration_seconds,
            "failed": True,
            "error": repr(exc),
        },
        "errors": [{
            "stage": "parallel_judge",
            "error": repr(exc),
            "error_class": error_class,
            "do_not_train": True,
        }],
        "judge_error": {
            "class": error_class,
            "message": repr(exc),
            "do_not_train": True,
            "parallel": True,
        },
    }


def _adaptive_evidence_summary(
    judge_step_traces: List[Dict[str, Any]],
    *,
    enabled: bool,
    requested_mode: str,
    selected_mode: str,
    reason: str,
    debug: bool = False,
) -> Dict[str, Any]:
    mode_counts: Dict[str, int] = {}
    total_added = 0
    total_removed = 0
    estimated_values: List[int] = []
    for trace in judge_step_traces:
        decision = trace.get("adaptive_evidence", {}) if isinstance(trace, dict) else {}
        if not isinstance(decision, dict):
            continue
        mode = str(decision.get("mode") or "planned_views")
        mode_counts[mode] = int(mode_counts.get(mode, 0) or 0) + 1
        total_added += len(list(decision.get("added_views", []) or []))
        total_removed += len(list(decision.get("removed_views", []) or []))
        if decision.get("estimated_chars") is not None:
            try:
                estimated_values.append(int(decision.get("estimated_chars") or 0))
            except (TypeError, ValueError):
                pass
    avg_estimated = (
        round(sum(estimated_values) / len(estimated_values), 2)
        if estimated_values
        else 0
    )
    return {
        "enabled": bool(enabled),
        "debug": bool(debug),
        "requested_mode": requested_mode,
        "selected_mode": selected_mode,
        "reason": reason,
        "mode_counts": mode_counts,
        "total_added_views": total_added,
        "total_removed_views": total_removed,
        "avg_estimated_chars": avg_estimated,
        "small_trace_direct_count": int(mode_counts.get("small_trace_direct", 0) or 0),
        "medium_hybrid_count": int(mode_counts.get("medium_hybrid", 0) or 0),
        "planned_views_count": int(mode_counts.get("planned_views", 0) or 0),
    }


def _unique_strings(values) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _infer_packet_chain(packet: Dict[str, Any]) -> str:
    views = packet.get("views", {}) if isinstance(packet, dict) else {}
    tx_card = views.get("tx_card", {}) if isinstance(views, dict) else {}
    if isinstance(tx_card, dict) and tx_card.get("chain"):
        return str(tx_card.get("chain"))
    return "eth"


def _judge_satisfied(answer: Any, expected_answer: bool) -> bool:
    if isinstance(answer, str):
        return False
    return bool(answer) == bool(expected_answer)


def _classify_reuse_miss(
    step: JudgeStep,
    judge_reuse_cache: Dict[str, Dict[str, Any]],
    *,
    tx_hash: str,
    packet_identity_fingerprint: str,
    selected_views_hash: str,
) -> str:
    judge_id = str(step.id or "")
    entries = [
        entry
        for entry in judge_reuse_cache.values()
        if str(entry.get("judge_id") or "") == judge_id
    ]
    if not entries:
        return "step_policy_mismatch"
    current_tx = str(tx_hash or "").lower().strip()
    tx_entries = [
        entry
        for entry in entries
        if str((entry.get("source", {}) or {}).get("tx_hash") or "").lower().strip()
        == current_tx
    ]
    if not tx_entries:
        return "tx_mismatch"
    packet_entries = []
    for entry in tx_entries:
        source = entry.get("source", {}) or {}
        source_packet_identity = str(
            source.get("packet_identity_fingerprint")
            or source.get("packet_fingerprint")
            or ""
        )
        if source_packet_identity == str(packet_identity_fingerprint or ""):
            packet_entries.append(entry)
    if not packet_entries:
        return "packet_identity_mismatch"
    expected_answer = bool(step.expected_answer)
    question = str(step.question or "")
    policy_entries = []
    saw_policy_fields = False
    for entry in packet_entries:
        judge_result = entry.get("judge_result", {}) or {}
        previous_question = str(judge_result.get("question") or "")
        previous_expected = bool(judge_result.get("expected_answer", True))
        if previous_question or "expected_answer" in judge_result:
            saw_policy_fields = True
        if previous_question == question and previous_expected == expected_answer:
            policy_entries.append(entry)
    if saw_policy_fields and not policy_entries:
        return "step_policy_mismatch"
    comparable_entries = policy_entries or packet_entries
    if not any(
        str((entry.get("source", {}) or {}).get("selected_views_hash") or "")
        == str(selected_views_hash or "")
        for entry in comparable_entries
    ):
        return "selected_views_hash_mismatch"
    return "step_policy_mismatch"


def _inject_access_control_anchor_probe_request(
    result: JudgeResult,
    *,
    step: JudgeStep,
    attack_label: str,
    allowed_tools: List[str],
    existing_observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Route a concrete unresolved C1 call through bounded local context."""
    if normalize_attack_label(attack_label, default="") != "access_control":
        return {}
    if str(step.state_prompt_role or "") != "ac_authorization_anchor":
        return {}
    available_tools = set(allowed_tools or [])
    if result.answer is True or not available_tools.intersection({
        "get_local_call_context",
        "read_evidence_context",
    }):
        return {}

    result_dict = result.to_dict()
    state_output = dict(result_dict.get("state_output") or {})
    summary = _access_control_candidate_summary_from_state(state_output) or {}
    probe_candidates = [
        item
        for item in _access_control_candidate_records(summary)
        if str(item.get("candidate_status") or "").strip().lower()
        == "unresolved_probe"
        or str(item.get("candidate_id") or "").strip()
        in set(_unique_strings(summary.get("unresolved_candidate_ids") or []))
    ]
    if not probe_candidates:
        synthesized = _access_control_unresolved_anchor_candidate(
            result_dict,
            summary,
        )
        if synthesized:
            probe_candidates = [synthesized]
    if not probe_candidates:
        return {}

    probe = probe_candidates[0]
    call_id = str(
        probe.get("entry_call_id")
        or probe.get("entry_call_evidence_id")
        or probe.get("sensitive_call_id")
        or probe.get("sensitive_call_evidence_id")
        or ""
    ).strip()
    if not _call_id_suffix(call_id):
        return {}
    if "get_local_call_context" in available_tools:
        tool_name = "get_local_call_context"
        args = {
            "evidence_id": call_id,
            "child_limit": 12,
            "include_events": True,
            "include_state": True,
        }
    else:
        tool_name = "read_evidence_context"
        args = {
            "evidence_id": call_id,
            "radius": 2,
            "include_parent": True,
            "include_children": True,
            "include_nearby_events": True,
            "include_nearby_state": True,
        }
    signature = _near_miss_request_signature(tool_name, args)
    seen = _near_miss_observation_request_signatures(existing_observations)
    existing_requests = list(result.tool_requests or [])
    if signature in seen or any(
        _near_miss_request_signature(
            str(item.get("tool") or ""),
            dict(item.get("args") or {}),
        )
        == signature
        for item in existing_requests
        if isinstance(item, dict)
    ):
        return {}

    request = {
        "tool": tool_name,
        "args": args,
        "reason": (
            "Automatic access-control C1 probe: inspect the concrete unresolved "
            "entry call and nearby child/state/event evidence before closing the "
            "authorization-anchor condition."
        ),
        "automatic_probe": True,
        "candidate_id": str(probe.get("candidate_id") or ""),
    }
    result.tool_requests = [request, *existing_requests][:2]
    return dict(request)


def _access_control_anchor_has_probe_request(
    result: JudgeResult,
    step: Optional[JudgeStep],
) -> bool:
    if step is None or str(step.state_prompt_role or "") != "ac_authorization_anchor":
        return False
    return any(
        isinstance(item, dict)
        and (
            bool(item.get("automatic_probe"))
            or str(item.get("tool") or "") == "get_local_call_context"
        )
        for item in list(result.tool_requests or [])
    )


def _should_follow_up(
    result: JudgeResult,
    max_followups: int,
    *,
    step: Optional[JudgeStep] = None,
) -> bool:
    if max_followups <= 0:
        return False
    if not result.tool_requests:
        return False
    answer = result.answer
    if isinstance(answer, str) and answer.lower().strip() == "uncertain":
        return True
    if answer is True:
        return False
    if answer is False and _access_control_anchor_has_probe_request(result, step):
        return True
    confidence = str(result.confidence or "").lower().strip()
    if answer is False and confidence == "low":
        if step is None:
            return True
        if _is_exclusion_step(step):
            return False
        if "read_function_chunk" in set(step.allowed_tools or []):
            return True
        return True
    return False


def _looks_like_evidence_id(value: str) -> bool:
    if not value or len(value) > 240:
        return False
    if ":" not in value:
        return False
    return not any(ch.isspace() for ch in value)


def _refine_access_control_candidate_request(
    *,
    tool_name: str,
    args: Dict[str, Any] | None,
    step: JudgeStep,
    attack_label: str,
    state_input_context: Dict[str, Any] | None,
    evidence_tool_registry: EvidenceToolRegistry,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    refined = dict(args or {})
    if normalize_attack_label(attack_label, default="") != "access_control":
        return refined, {}
    role = str(step.state_prompt_role or "")
    if not role.startswith("ac_") or role == "ac_authorization_anchor":
        return refined, {}
    state_input = dict(state_input_context or {})
    candidate_ids = _access_control_relevant_candidate_ids(state_input, role=role)
    evidence_ids = _access_control_candidate_evidence_ids(
        state_input,
        candidate_ids=candidate_ids,
    )
    if not evidence_ids:
        return refined, {}

    original_args = dict(refined)
    tool = str(tool_name or "")
    if tool == "read_packet_view":
        if (
            refined.get("evidence_id")
            or list(refined.get("evidence_ids", []) or [])
            or list(refined.get("keywords", []) or [])
        ):
            return refined, {}
        view = str(refined.get("view") or "")
        anchor_scope = "candidate_chain"
        matched: List[str] = []
        if view == "source_unavailable_auth_view":
            entry_evidence_ids = _access_control_candidate_entry_evidence_ids(
                state_input,
                candidate_ids=candidate_ids,
            )
            matched = _access_control_candidate_view_evidence_ids(
                view,
                entry_evidence_ids,
                evidence_tool_registry,
            )
            if matched:
                anchor_scope = "entry_call"
        if not matched:
            matched = _access_control_candidate_view_evidence_ids(
                view,
                evidence_ids,
                evidence_tool_registry,
            )
        if not matched:
            return refined, {}
        refined["evidence_ids"] = matched[:12]
        current_limit = int(refined.get("limit", 0) or 0)
        refined["limit"] = min(24, max(4, current_limit, len(matched)))
    elif tool == "read_evidence_by_id":
        if refined.get("evidence_id") or list(refined.get("evidence_ids", []) or []):
            return refined, {}
        refined["evidence_ids"] = evidence_ids[:12]
    elif tool == "read_evidence_context":
        if refined.get("evidence_id"):
            return refined, {}
        refined["evidence_id"] = evidence_ids[0]
    elif tool == "read_function_chunk":
        sensitive_evidence_ids = _access_control_candidate_sensitive_evidence_ids(
            state_input,
            candidate_ids=candidate_ids,
        )
        entry_evidence_ids = _access_control_candidate_entry_evidence_ids(
            state_input,
            candidate_ids=candidate_ids,
        )
        if (
            refined.get("evidence_id")
            or refined.get("call_id")
            or refined.get("address")
            or refined.get("function")
            or refined.get("target_function")
        ):
            explicit_evidence_id = _call_evidence_id(
                refined.get("evidence_id") or refined.get("call_id")
            )
            if (
                explicit_evidence_id
                and len(sensitive_evidence_ids) == 1
                and explicit_evidence_id not in set(sensitive_evidence_ids)
            ):
                refined.pop("call_id", None)
                refined["evidence_id"] = sensitive_evidence_ids[0]
                refined["preserve_evidence_target"] = True
                return refined, {
                    "kind": "access_control_candidate_anchor_refinement",
                    "reason": (
                        "Authorization source follow-up was retargeted from an "
                        "entry or path call to the candidate's sensitive call."
                    ),
                    "role": role,
                    "anchor_scope": "sensitive_call",
                    "candidate_ids": candidate_ids,
                    "original_args": original_args,
                    "refined_evidence_ids": [sensitive_evidence_ids[0]],
                }
            if explicit_evidence_id in set(sensitive_evidence_ids):
                refined["preserve_evidence_target"] = True
                return refined, {
                    "kind": "access_control_candidate_anchor_refinement",
                    "reason": (
                        "The requested source target is the candidate's sensitive "
                        "capability call."
                    ),
                    "role": role,
                    "anchor_scope": "sensitive_call",
                    "candidate_ids": candidate_ids,
                    "original_args": original_args,
                    "refined_evidence_ids": [explicit_evidence_id],
                }
            if explicit_evidence_id in set(entry_evidence_ids):
                refined["preserve_evidence_target"] = True
                return refined, {
                    "kind": "access_control_candidate_anchor_refinement",
                    "reason": (
                        "The requested source target is the C1 entry anchor; "
                        "preserve it instead of retargeting to a descendant call."
                    ),
                    "role": role,
                    "anchor_scope": "entry_call",
                    "candidate_ids": candidate_ids,
                    "original_args": original_args,
                    "refined_evidence_ids": [explicit_evidence_id],
                }
            return refined, {}
        call_evidence_id = (
            sensitive_evidence_ids[0]
            if sensitive_evidence_ids
            else entry_evidence_ids[0]
            if entry_evidence_ids
            else _first_call_like_evidence_id(evidence_ids)
        )
        if not call_evidence_id:
            return refined, {}
        refined["evidence_id"] = call_evidence_id
        if call_evidence_id in set(entry_evidence_ids):
            refined["preserve_evidence_target"] = True
    elif tool == "get_local_call_context":
        if refined.get("evidence_id") or refined.get("call_id") or refined.get("id"):
            return refined, {}
        sensitive_evidence_ids = _access_control_candidate_sensitive_evidence_ids(
            state_input,
            candidate_ids=candidate_ids,
        )
        entry_evidence_ids = _access_control_candidate_entry_evidence_ids(
            state_input,
            candidate_ids=candidate_ids,
        )
        call_evidence_id = _first_call_like_evidence_id(
            [*sensitive_evidence_ids, *entry_evidence_ids, *evidence_ids]
        )
        if not call_evidence_id:
            return refined, {}
        refined["evidence_id"] = call_evidence_id
    else:
        return refined, {}

    return refined, {
        "kind": "access_control_candidate_anchor_refinement",
        "reason": (
            "Broad follow-up request was narrowed to evidence from the C1 "
            "access_control_candidate so downstream judges validate the same object."
        ),
        "role": role,
        "anchor_scope": anchor_scope if tool == "read_packet_view" else "candidate_chain",
        "candidate_ids": candidate_ids,
        "original_args": original_args,
        "refined_evidence_ids": list(refined.get("evidence_ids") or [refined.get("evidence_id")]),
    }


def _access_control_source_failure_fallback_observations(
    *,
    source_tool_name: str,
    source_tool_result: Dict[str, Any],
    step: JudgeStep,
    attack_label: str,
    state_input_context: Dict[str, Any] | None,
    evidence_tool_registry: EvidenceToolRegistry,
    allowed_tools: List[str],
    allowed_followup_views: List[str],
    existing_observations: List[Dict[str, Any]],
    request_round: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if str(source_tool_name or "") != "read_function_chunk":
        return [], []
    if normalize_attack_label(attack_label, default="") != "access_control":
        return [], []
    role = str(step.state_prompt_role or "")
    if role != "ac_authorization_gap" and not _is_access_control_authorization_gap({
        "id": step.id,
        "condition_id": step.condition_id or step.id,
        "question": step.question,
        "is_exclusion": False,
    }):
        return [], []
    status = str(source_tool_result.get("tool_status") or "").lower().strip()
    if status not in _ACCESS_CONTROL_SOURCE_FAILURE_STATUSES:
        return [], []

    allowed_tool_set = set(allowed_tools or [])
    if "read_packet_view" not in allowed_tool_set:
        return [], []
    allowed_views = set(allowed_followup_views or [])
    seen_requests = _near_miss_observation_request_signatures(existing_observations)
    observations: List[Dict[str, Any]] = []
    calls: List[Dict[str, Any]] = []
    view_limits = {
        "source_unavailable_auth_view": 8,
        "unknown_selector_view": 6,
        "critical_call_argument_view": 8,
        "state_change_view": 8,
        "semantic_state_delta_view": 8,
        "beneficiary_controller_view": 6,
        "critical_call_view": 8,
        "value_release_view": 8,
    }
    for view in _ACCESS_CONTROL_SOURCE_FALLBACK_VIEWS:
        if view not in allowed_views:
            continue
        args = {"view": view, "limit": view_limits.get(view, 6)}
        refined_args, refinement = _refine_access_control_candidate_request(
            tool_name="read_packet_view",
            args=args,
            step=step,
            attack_label=attack_label,
            state_input_context=state_input_context,
            evidence_tool_registry=evidence_tool_registry,
        )
        anchored = bool(refinement)
        anchor_scope = (
            str(refinement.get("anchor_scope") or "") if anchored else ""
        )
        signature = _near_miss_request_signature(
            "read_packet_view",
            refined_args,
        )
        if signature in seen_requests:
            continue
        seen_requests.add(signature)
        if anchored:
            tool_result = evidence_tool_registry.call(
                "read_packet_view",
                refined_args,
            )
        else:
            tool_result = {
                "tool": "read_packet_view",
                "tool_status": "candidate_anchor_unavailable",
                "args": dict(args),
                "summary": {
                    "returned_rows": 0,
                    "candidate_filtered": False,
                    "reason": (
                        "C1 did not provide a usable entry/candidate evidence "
                        "anchor; runtime refused an unfiltered authorization view."
                    ),
                },
                "evidence": {"rows": []},
                "returned_evidence_ids": [],
            }
            refined_args = args
        if anchor_scope == "entry_call":
            fallback_anchor_status = "entry_filtered"
            reason = (
                "read_function_chunk could not provide verified source for the "
                "candidate authorization-gap check, so runtime attached the "
                "source_unavailable_auth_view filtered to the same C1 entry."
            )
        elif anchored:
            fallback_anchor_status = "candidate_chain_filtered"
            reason = (
                "read_function_chunk could not provide verified source for the "
                "candidate authorization-gap check, so runtime attached the "
                "source_unavailable_auth_view filtered to the same C1 candidate "
                "chain."
            )
        else:
            fallback_anchor_status = "candidate_anchor_unavailable"
            reason = (
                "read_function_chunk could not provide verified source for the "
                "authorization-gap check, but no usable C1 candidate anchor was "
                "available; runtime did not expose an unrelated unfiltered view."
            )
        call = {
            "judge_id": step.id,
            "condition_id": step.condition_id or step.id,
            "round": request_round,
            "tool": "read_packet_view",
            "args": refined_args,
            "reason": reason,
            "tool_status": tool_result.get("tool_status", "unknown"),
            "returned_evidence_ids": tool_result.get("returned_evidence_ids", []),
            "summary": tool_result.get("summary", {}),
            "observation": tool_result,
            "automatic_source_failure_fallback": True,
            "route_owner": "plan",
            "plan_effect_attribution_eligible": True,
            "source_failure_status": status,
            "fallback_anchor_status": fallback_anchor_status,
            "plan_declared_read_packet_view": (
                "read_packet_view" in set(allowed_tools or [])
            ),
            "plan_declared_fallback_view": view in allowed_views,
        }
        if refinement:
            call["request_refinement"] = refinement
        calls.append(call)
        observations.append({
            "request": {
                "tool": "read_packet_view",
                "args": refined_args,
                "reason": reason,
                "automatic_source_failure_fallback": True,
                "fallback_anchor_status": fallback_anchor_status,
                "route_owner": "plan",
                "plan_effect_attribution_eligible": True,
            },
            "result": tool_result,
            "target_condition_specific": bool(anchored),
        })

    if "get_local_call_context" in set(allowed_tools or []):
        state_input = dict(state_input_context or {})
        candidate_ids = _access_control_relevant_candidate_ids(
            state_input,
            role=role,
        )
        local_ids = [
            *_access_control_candidate_sensitive_evidence_ids(
                state_input,
                candidate_ids=candidate_ids,
            ),
            *_access_control_candidate_entry_evidence_ids(
                state_input,
                candidate_ids=candidate_ids,
            ),
        ]
        local_evidence_id = _first_call_like_evidence_id(local_ids)
        if local_evidence_id:
            local_args = {
                "evidence_id": local_evidence_id,
                "child_limit": 12,
                "include_events": True,
                "include_state": True,
            }
            local_signature = _near_miss_request_signature(
                "get_local_call_context",
                local_args,
            )
            if local_signature not in seen_requests:
                local_result = evidence_tool_registry.call(
                    "get_local_call_context",
                    local_args,
                )
                local_reason = (
                    "Source lookup failed, so runtime attached bounded local "
                    "call/child/state/event context for the same C1 candidate "
                    "before asking C2 to resolve authorization provenance."
                )
                calls.append({
                    "judge_id": step.id,
                    "condition_id": step.condition_id or step.id,
                    "round": request_round,
                    "tool": "get_local_call_context",
                    "args": local_args,
                    "reason": local_reason,
                    "tool_status": local_result.get("tool_status", "unknown"),
                    "returned_evidence_ids": local_result.get(
                        "returned_evidence_ids", []
                    ),
                    "summary": local_result.get("summary", {}),
                    "observation": local_result,
                    "automatic_source_failure_fallback": True,
                    "route_owner": "plan",
                    "plan_effect_attribution_eligible": True,
                    "source_failure_status": status,
                    "fallback_anchor_status": "candidate_local_context",
                })
                observations.append({
                    "request": {
                        "tool": "get_local_call_context",
                        "args": local_args,
                        "reason": local_reason,
                        "automatic_source_failure_fallback": True,
                        "route_owner": "plan",
                        "plan_effect_attribution_eligible": True,
                    },
                    "result": local_result,
                    "target_condition_specific": True,
                })
    return observations, calls


def _observations_include_terminal_source_failure(
    observations: List[Dict[str, Any]],
) -> bool:
    for observation in list(observations or []):
        result = _tool_observation_result(observation)
        if str(result.get("tool") or "").strip() != "read_function_chunk":
            continue
        status = str(result.get("tool_status") or "").lower().strip()
        if status in _SOURCE_TERMINAL_FAILURE_STATUSES:
            return True
    return False


def _terminal_source_failure_for_request(
    observations: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    requested = _source_lookup_signatures(args)
    if not requested:
        return {}
    for observation in list(observations or []):
        result = _tool_observation_result(observation)
        if str(result.get("tool") or "").strip() != "read_function_chunk":
            continue
        status = str(result.get("tool_status") or "").lower().strip()
        if status not in _SOURCE_TERMINAL_FAILURE_STATUSES:
            continue
        observed = set()
        observed.update(_source_lookup_signatures(result.get("args") or {}))
        observed.update(_source_lookup_signatures(result.get("query") or {}))
        request = dict(observation.get("request") or {})
        observed.update(_source_lookup_signatures(request.get("args") or {}))
        if requested & observed:
            return result
    return {}


def _reused_terminal_source_failure_result(
    args: Dict[str, Any],
    previous: Dict[str, Any],
) -> Dict[str, Any]:
    status = str(previous.get("tool_status") or "source_unavailable").lower().strip()
    return {
        "tool": "read_function_chunk",
        "tool_status": status,
        "args": dict(args or {}),
        "query": dict((previous or {}).get("query") or {}),
        "summary": {
            **dict((previous or {}).get("summary") or {}),
            "matched": False,
            "reused_terminal_failure": True,
        },
        "evidence": {"snippets": [], "matched_keys": []},
        "returned_evidence_ids": [],
        "note": (
            "A previous read_function_chunk lookup for this same target already "
            f"returned terminal status={status}; Runtime reused that failure "
            "and should fall back to packet-level evidence."
        ),
    }


def _tool_observation_result(observation: Any) -> Dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    result = observation.get("result")
    if isinstance(result, dict):
        return result
    if str(observation.get("tool") or "") == "read_function_chunk":
        return observation
    return {}


def _source_lookup_signatures(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    signatures: set[str] = set()
    evidence_id = str(value.get("evidence_id") or "").strip()
    if evidence_id:
        signatures.add(f"evidence:{evidence_id.lower()}")
    resolution = dict(value.get("source_resolution") or {})
    if resolution.get("evidence_id"):
        signatures.add(f"evidence:{str(resolution.get('evidence_id')).lower()}")
    chain = str(value.get("chain") or "").strip().lower()
    address = str(
        value.get("address")
        or value.get("contract")
        or value.get("callee")
        or resolution.get("resolved_address")
        or ""
    ).strip().lower()
    function_name = str(
        value.get("function_name")
        or value.get("function")
        or value.get("target_function")
        or value.get("selector")
        or value.get("raw_selector")
        or value.get("function_selector")
        or resolution.get("resolved_function_name")
        or ""
    ).strip().lower()
    if address and function_name:
        signatures.add(f"target:{chain}:{address}:{function_name}")
    elif address:
        signatures.add(f"address:{chain}:{address}")
    return signatures


def _tool_result_returned_row_count(result: Dict[str, Any]) -> int:
    summary = dict(result.get("summary") or {}) if isinstance(result, dict) else {}
    for key in ("returned_rows", "snippet_count"):
        if key not in summary:
            continue
        try:
            return int(summary.get(key) or 0)
        except (TypeError, ValueError):
            pass
    returned_ids = list(result.get("returned_evidence_ids", []) or [])
    if returned_ids:
        return len(returned_ids)
    evidence = result.get("evidence") if isinstance(result, dict) else {}
    if isinstance(evidence, dict):
        rows = evidence.get("rows")
        if isinstance(rows, list):
            return len(rows)
        snippets = evidence.get("snippets")
        if isinstance(snippets, list):
            return len(snippets)
    return 0


def _access_control_relevant_candidate_ids(
    state_input: Dict[str, Any],
    *,
    role: str,
) -> List[str]:
    candidate_summary = _access_control_candidate_summary_from_state(state_input)
    has_selected_field = "selected_candidate_ids" in candidate_summary
    selected = _unique_strings(candidate_summary.get("selected_candidate_ids") or [])
    unresolved_anchor_ids = _unique_strings(
        [
            *list(candidate_summary.get("unresolved_candidate_ids") or []),
            *list(candidate_summary.get("probe_candidate_ids") or []),
        ]
    )
    candidates = _access_control_candidate_records(candidate_summary)
    all_ids = _unique_strings(
        str(item.get("candidate_id") or "").strip()
        for item in candidates
        if item.get("candidate_id")
    )
    fallback_ids = selected if has_selected_field else all_ids
    if role == "ac_authorization_gap" and not fallback_ids:
        fallback_ids = unresolved_anchor_ids
    gap_summary = dict(state_input.get("authorization_gap_summary") or {})
    effect_summary = dict(state_input.get("protected_effect_summary") or {})
    if role == "ac_protected_effect":
        ids = _unique_strings([
            *list(gap_summary.get("satisfying_candidate_ids") or []),
            *list(gap_summary.get("unresolved_candidate_ids") or []),
        ])
        return ids or fallback_ids
    if role == "ac_candidate_exclusion":
        ids = _unique_strings(effect_summary.get("attack_candidate_ids") or [])
        return ids or _unique_strings(gap_summary.get("satisfying_candidate_ids") or []) or fallback_ids
    return fallback_ids


def _access_control_candidate_evidence_ids(
    state_input: Dict[str, Any],
    *,
    candidate_ids: List[str],
) -> List[str]:
    if not candidate_ids:
        return []
    summary = _access_control_candidate_summary_from_state(state_input)
    records = _access_control_candidate_records(summary)
    wanted = {str(value or "").strip() for value in candidate_ids if str(value or "").strip()}
    evidence_ids: List[str] = []
    entry_scalar_fields = (
        "entry_call_id",
        "entry_call_evidence_id",
        "entry_evidence_id",
    )
    list_fields = (
        "evidence_ids",
        "source_probe_evidence_ids",
        "call_evidence_ids",
        "invocation_evidence_ids",
        "path_evidence_ids",
        "related_evidence_ids",
        "path_ids",
    )
    scalar_fields = (
        "sensitive_call_id",
        "sensitive_call_evidence_id",
        "source_call_evidence_id",
        "call_id",
        "evidence_id",
    )
    for record in records:
        candidate_id = str(record.get("candidate_id") or "").strip()
        if wanted and candidate_id not in wanted:
            continue
        for field in entry_scalar_fields:
            evidence_id = _call_evidence_id(record.get(field))
            if evidence_id:
                evidence_ids.append(evidence_id)
        for field in list_fields:
            for value in _access_control_iter_values(record.get(field)):
                evidence_id = _call_evidence_id(value)
                if evidence_id:
                    evidence_ids.append(evidence_id)
        for field in scalar_fields:
            evidence_id = _call_evidence_id(record.get(field))
            if evidence_id:
                evidence_ids.append(evidence_id)
    return _unique_strings(evidence_ids)


def _access_control_candidate_entry_evidence_ids(
    state_input: Dict[str, Any],
    *,
    candidate_ids: List[str],
) -> List[str]:
    """Return only explicit entry-call anchors for selected C1 candidates."""
    if not candidate_ids:
        return []
    summary = _access_control_candidate_summary_from_state(state_input)
    records = _access_control_candidate_records(summary)
    wanted = {
        str(value or "").strip()
        for value in candidate_ids
        if str(value or "").strip()
    }
    evidence_ids: List[str] = []
    for record in records:
        candidate_id = str(record.get("candidate_id") or "").strip()
        if wanted and candidate_id not in wanted:
            continue
        for field in (
            "entry_call_id",
            "entry_call_evidence_id",
            "entry_evidence_id",
        ):
            evidence_id = _call_evidence_id(record.get(field))
            if evidence_id:
                evidence_ids.append(evidence_id)
    return _unique_strings(evidence_ids)


def _access_control_candidate_sensitive_evidence_ids(
    state_input: Dict[str, Any],
    *,
    candidate_ids: List[str],
) -> List[str]:
    """Return downstream sensitive-call anchors without broad candidate rows."""
    if not candidate_ids:
        return []
    summary = _access_control_candidate_summary_from_state(state_input)
    records = _access_control_candidate_records(summary)
    wanted = {
        str(value or "").strip()
        for value in candidate_ids
        if str(value or "").strip()
    }
    evidence_ids: List[str] = []
    for record in records:
        candidate_id = str(record.get("candidate_id") or "").strip()
        if wanted and candidate_id not in wanted:
            continue
        for field in (
            "sensitive_call_id",
            "sensitive_call_evidence_id",
            "source_call_evidence_id",
        ):
            evidence_id = _call_evidence_id(record.get(field))
            if evidence_id:
                evidence_ids.append(evidence_id)
    return _unique_strings(evidence_ids)


def _refine_token_semantic_candidate_request(
    *,
    tool_name: str,
    args: Dict[str, Any] | None,
    step: JudgeStep,
    attack_label: str,
    state_input_context: Dict[str, Any] | None,
    evidence_tool_registry: EvidenceToolRegistry,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    refined = dict(args or {})
    if normalize_attack_label(attack_label, default="") != "token_semantic_exploitation":
        return refined, {}
    role = str(step.state_prompt_role or "")
    if not role.startswith("ts_") or role == "ts_token_semantic_anchor":
        return refined, {}
    state_input = dict(state_input_context or {})
    candidate_ids = _token_semantic_relevant_candidate_ids(state_input, role=role)
    evidence_ids = _token_semantic_candidate_evidence_ids(
        state_input,
        candidate_ids=candidate_ids,
    )
    if not evidence_ids:
        return refined, {}

    original_args = dict(refined)
    tool = str(tool_name or "")
    if tool == "read_packet_view":
        if (
            refined.get("evidence_id")
            or list(refined.get("evidence_ids", []) or [])
            or list(refined.get("keywords", []) or [])
        ):
            return refined, {}
        view = str(refined.get("view") or "")
        matched = _access_control_candidate_view_evidence_ids(
            view,
            evidence_ids,
            evidence_tool_registry,
        )
        if not matched:
            return refined, {}
        refined["evidence_ids"] = matched[:12]
        current_limit = int(refined.get("limit", 0) or 0)
        refined["limit"] = min(24, max(4, current_limit, len(matched)))
    elif tool == "read_evidence_by_id":
        if refined.get("evidence_id") or list(refined.get("evidence_ids", []) or []):
            return refined, {}
        refined["evidence_ids"] = evidence_ids[:12]
    elif tool == "read_evidence_context":
        if refined.get("evidence_id"):
            return refined, {}
        refined["evidence_id"] = evidence_ids[0]
    elif tool in {"read_function_chunk", "get_local_call_context"}:
        if (
            refined.get("evidence_id")
            or refined.get("call_id")
            or refined.get("address")
            or refined.get("function")
            or refined.get("target_function")
            or refined.get("id")
        ):
            return refined, {}
        call_evidence_id = _first_call_like_evidence_id(evidence_ids)
        if not call_evidence_id:
            return refined, {}
        refined["evidence_id"] = call_evidence_id
    else:
        return refined, {}

    return refined, {
        "kind": "token_semantic_candidate_anchor_refinement",
        "reason": (
            "Broad follow-up request was narrowed to evidence from the token "
            "semantic candidate so downstream judges validate the same token "
            "mechanism instead of unrelated protocol accounting rows."
        ),
        "role": role,
        "candidate_ids": candidate_ids,
        "original_args": original_args,
        "refined_evidence_ids": list(
            refined.get("evidence_ids") or [refined.get("evidence_id")]
        ),
    }


def _token_semantic_relevant_candidate_ids(
    state_input: Dict[str, Any],
    *,
    role: str,
) -> List[str]:
    candidate_summary = _token_semantic_candidate_summary_from_state(state_input)
    selected = _unique_strings(candidate_summary.get("selected_candidate_ids") or [])
    candidates = _token_semantic_candidate_records(candidate_summary)
    all_ids = _unique_strings(
        str(item.get("candidate_id") or "").strip()
        for item in candidates
        if item.get("candidate_id")
    )
    reliance_summary = dict(state_input.get("token_semantic_reliance_summary") or {})
    outcome_summary = dict(state_input.get("token_semantic_outcome_summary") or {})
    if role == "ts_semantic_outcome":
        ids = _unique_strings(reliance_summary.get("satisfying_candidate_ids") or [])
        return ids or selected or all_ids
    if role == "ts_candidate_exclusion":
        ids = _unique_strings(outcome_summary.get("attack_candidate_ids") or [])
        return ids or _unique_strings(
            reliance_summary.get("satisfying_candidate_ids") or []
        ) or selected or all_ids
    return selected or all_ids


def _token_semantic_candidate_evidence_ids(
    state_input: Dict[str, Any],
    *,
    candidate_ids: List[str],
) -> List[str]:
    summary = _token_semantic_candidate_summary_from_state(state_input)
    records = _token_semantic_candidate_records(summary)
    wanted = {str(value or "").strip() for value in candidate_ids if str(value or "").strip()}
    evidence_ids: List[str] = []
    list_fields = (
        "evidence_ids",
        "state_evidence_ids",
        "transfer_event_evidence_ids",
        "related_evidence_ids",
        "call_evidence_ids",
    )
    scalar_fields = (
        "evidence_id",
        "call_id",
        "source_call_evidence_id",
    )
    for record in records:
        candidate_id = str(record.get("candidate_id") or "").strip()
        if wanted and candidate_id not in wanted:
            continue
        for field in list_fields:
            for value in _access_control_iter_values(record.get(field)):
                evidence_id = _call_evidence_id(value)
                if evidence_id:
                    evidence_ids.append(evidence_id)
        for field in scalar_fields:
            evidence_id = _call_evidence_id(record.get(field))
            if evidence_id:
                evidence_ids.append(evidence_id)
    return _unique_strings(evidence_ids)


def _access_control_iter_values(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _access_control_candidate_view_evidence_ids(
    view: str,
    evidence_ids: List[str],
    evidence_tool_registry: EvidenceToolRegistry,
) -> List[str]:
    if not view:
        return []
    matched: List[str] = []
    packet_registry = evidence_tool_registry.packet_registry
    for evidence_id in evidence_ids:
        for alternate in _candidate_evidence_id_alternates(evidence_id):
            try:
                response = packet_registry.read_packet_view({
                    "view": view,
                    "evidence_ids": [alternate],
                    "limit": 1,
                })
            except Exception:
                response = {}
            summary = dict(response.get("summary") or {}) if isinstance(response, dict) else {}
            evidence = dict(response.get("evidence") or {}) if isinstance(response, dict) else {}
            rows = list(evidence.get("rows") or [])
            if int(summary.get("returned_rows", len(rows)) or 0) > 0:
                matched.append(alternate)
                break
        if len(matched) >= 12:
            break
    return _unique_strings(matched)


def _candidate_evidence_id_alternates(evidence_id: Any) -> List[str]:
    value = _call_evidence_id(evidence_id)
    if not value:
        return []
    suffix = _call_id_suffix(value)
    alternates = [value]
    if suffix:
        alternates.extend([
            f"call:{suffix}",
            f"value_release:call:{suffix}",
            f"critical_call:call:{suffix}",
            f"source_unavailable_auth:call:{suffix}",
            f"unknown_selector:call:{suffix}",
            f"reentrancy_state_order:call:{suffix}",
            f"state_semantic:call:{suffix}",
        ])
    return _unique_strings(alternates)


def _first_call_like_evidence_id(evidence_ids: List[str]) -> str:
    for evidence_id in evidence_ids:
        value = _call_evidence_id(evidence_id)
        if value and "call:" in value:
            return value
    return ""


def _call_evidence_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isdigit():
        return f"call:{text}"
    if text.startswith("call:") or ":call:" in text:
        return text
    if _looks_like_evidence_id(text):
        return text
    return ""


def _call_id_suffix(evidence_id: Any) -> str:
    value = str(evidence_id or "")
    marker_index = value.rfind("call:")
    if marker_index < 0:
        return ""
    suffix = value[marker_index + len("call:") :].split(":", 1)[0]
    return suffix if suffix.isdigit() else ""


def _refine_reentrancy_state_order_request(
    *,
    tool_name: str,
    args: Dict[str, Any] | None,
    step: JudgeStep,
    attack_label: str,
    current_result: JudgeResult,
    evidence_tool_registry: EvidenceToolRegistry,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    refined = dict(args or {})
    if (
        str(tool_name or "") != "read_packet_view"
        or str(refined.get("view") or "") != "reentrancy_state_order_view"
        or refined.get("evidence_id")
        or list(refined.get("evidence_ids", []) or [])
        or list(refined.get("keywords", []) or [])
        or normalize_attack_label(attack_label, default="") != "reentrancy"
    ):
        return refined, {}
    step_views = {
        str(view or "")
        for view in (
            list(step.default_evidence_refs or step.evidence_refs)
            + list(step.allowed_followup_views or [])
        )
    }
    if not (
        is_reentrant_state_order_text(
            step.question,
            attack_label=attack_label,
        )
        or "reentrancy_state_order_view" in step_views
        or "reentrancy_state_order_summary_view" in step_views
    ):
        return refined, {}

    anchors = _unique_strings(
        list(current_result.supporting_evidence_ids or [])
        + list(current_result.contradicting_evidence_ids or [])
    )
    candidates: List[str] = []
    for evidence_id in anchors:
        value = str(evidence_id or "").strip()
        if value.startswith("reentrancy_state_order:call:"):
            candidates.append(value)
            continue
        marker_index = value.rfind("call:")
        if marker_index < 0:
            continue
        call_suffix = value[marker_index + len("call:") :].split(":", 1)[0]
        if call_suffix.isdigit():
            candidates.append(f"reentrancy_state_order:call:{call_suffix}")

    matched: List[str] = []
    packet_registry = evidence_tool_registry.packet_registry
    for evidence_id in _unique_strings(candidates):
        if packet_registry.find_evidence_rows(evidence_id, limit=1):
            matched.append(evidence_id)
        if len(matched) >= 4:
            break
    if not matched:
        return refined, {}

    original_args = dict(refined)
    refined["evidence_ids"] = matched
    refined["limit"] = min(8, max(len(matched), int(refined.get("limit", 0) or 0)))
    return refined, {
        "kind": "reentrancy_state_order_anchor_refinement",
        "reason": (
            "Broad state-order view request was narrowed to existing reentrant "
            "call evidence so tool payload budget is spent on the relevant chain."
        ),
        "original_args": original_args,
        "refined_evidence_ids": matched,
    }


def _is_exclusion_step(step: JudgeStep) -> bool:
    judge_id = str(step.id or "").upper()
    condition_id = str(step.condition_id or "").upper()
    return judge_id.startswith("E") or condition_id.startswith("E")


def _summarize_judge_round(
    result: JudgeResult,
    round_id: int,
    missing_items: List[Dict[str, Any]] | None = None,
    evidence_validation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "round": round_id,
        "answer": result.answer,
        "confidence": result.confidence,
        "reason": result.reason,
        "supporting_evidence_ids": list(result.supporting_evidence_ids),
        "contradicting_evidence_ids": list(result.contradicting_evidence_ids),
        "missing_evidence": list(result.missing_evidence),
        "missing_evidence_debug": list(missing_items or []),
        "tool_requests": list(result.tool_requests),
        "condition_feature_analysis": normalize_condition_feature_analysis(
            result.condition_feature_analysis
        ),
        "state_input": dict(result.state_input or {}),
        "state_output": dict(result.state_output or {}),
        "stateful_runtime": dict(result.stateful_runtime or {}),
        "evidence_id_validation": dict(evidence_validation or {}),
    }


def _compact_judge_parse_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    source = dict(metadata or {})
    return {
        key: source.get(key)
        for key in (
            "parse_status",
            "repair_attempted",
            "primary_output_empty",
            "primary_output_exhausted",
            "primary_finish_reason",
            "primary_completion_tokens",
            "primary_reasoning_tokens",
            "primary_max_tokens",
        )
        if key in source
    }


def _should_preserve_previous_judge_result(
    parse_metadata: Dict[str, Any] | None,
) -> bool:
    return bool(dict(parse_metadata or {}).get("primary_output_exhausted"))


def _ignore_final_answer_tool_requests(
    result: JudgeResult,
    step: JudgeStep,
    round_id: int,
    *,
    followup_allowed: bool = False,
) -> List[Dict[str, Any]]:
    if not result.tool_requests:
        return []
    if followup_allowed:
        return []
    if isinstance(result.answer, str) and result.answer.lower().strip() == "uncertain":
        return []
    ignored = []
    for request in result.tool_requests:
        ignored.append({
            "judge_id": step.id,
            "condition_id": step.condition_id or step.id,
            "round": round_id,
            "tool": request.get("tool"),
            "args": dict(request.get("args") or {}),
            "reason": (
                f"answer was final {result.answer} with {result.confidence} confidence; "
                "tool requests are allowed only for uncertain answers"
            ),
            "request_reason": request.get("reason", ""),
        })
    result.tool_requests = []
    return ignored


def validate_judge_evidence_ids(
    result: JudgeResult,
    evidence_tool_registry: EvidenceToolRegistry,
    *,
    visible_evidence_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Filter citations to evidence that was actually visible to this Judge."""
    original_supporting = _unique_strings(result.supporting_evidence_ids)
    original_contradicting = _unique_strings(result.contradicting_evidence_ids)
    valid_supporting: List[str] = []
    valid_contradicting: List[str] = []
    invalid: List[str] = []

    visible = (
        set(_unique_strings(visible_evidence_ids))
        if visible_evidence_ids is not None
        else None
    )

    def exists(evidence_id: str) -> bool:
        if visible is not None:
            return evidence_id in visible
        try:
            return bool(evidence_tool_registry.packet_registry.find_evidence_rows(evidence_id, limit=1))
        except Exception:
            return False

    for evidence_id in original_supporting:
        if exists(evidence_id):
            valid_supporting.append(evidence_id)
        else:
            invalid.append(evidence_id)
    for evidence_id in original_contradicting:
        if exists(evidence_id):
            valid_contradicting.append(evidence_id)
        else:
            invalid.append(evidence_id)

    result.supporting_evidence_ids = valid_supporting
    result.contradicting_evidence_ids = valid_contradicting

    confidence_before = result.confidence
    if result.answer is True:
        if original_supporting and not valid_supporting:
            result.confidence = "low"
        elif not valid_supporting and str(result.confidence).lower().strip() == "high":
            result.confidence = "medium"
    return {
        "supporting_original": original_supporting,
        "supporting_valid": valid_supporting,
        "contradicting_original": original_contradicting,
        "contradicting_valid": valid_contradicting,
        "invalid_evidence_ids": _unique_strings(invalid),
        "validation_scope": (
            "visible_judge_context"
            if visible is not None
            else "packet_registry_compatibility_fallback"
        ),
        "visible_evidence_ids": sorted(visible) if visible is not None else [],
        "confidence_before": confidence_before,
        "confidence_after": result.confidence,
    }


def _normalize_missing_for_round(
    result: JudgeResult,
    step: JudgeStep,
    round_id: int,
    packet: Dict[str, Any],
    render_metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    context = {
        "view_render_metadata": render_metadata,
        "evidence_adequacy_view": (packet.get("views", {}) or {}).get("evidence_adequacy_view", {}),
    }
    return dedupe_missing_evidence(
        normalize_missing_evidence(
            text,
            condition_id=step.condition_id or step.id,
            round_id=round_id,
            context=context,
        )
        for text in list(result.missing_evidence or [])
    )


def _finalize_missing_for_result(
    result: JudgeResult,
    missing_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    finalized = []
    final_answer = not isinstance(result.answer, str)
    final_confident = result.confidence in {"medium", "high"}
    for item in dedupe_missing_evidence(missing_items):
        updated = dict(item)
        if final_answer and final_confident and updated.get("status") == "open":
            updated["status"] = "stale"
            updated["blocking"] = False
            updated["resolved_by"] = "final_answer_with_medium_or_high_confidence"
        finalized.append(updated)
    return dedupe_missing_evidence(finalized)


def _is_exclusion_judge(judge_result: Dict[str, Any]) -> bool:
    if "is_exclusion" in judge_result:
        return bool(judge_result.get("is_exclusion"))
    judge_id = str(judge_result.get("id") or "").upper()
    condition_id = str(judge_result.get("condition_id") or "").upper()
    return judge_id.startswith("E") or condition_id.startswith("E")


def _access_control_condition_text(judge_result: Dict[str, Any]) -> str:
    return str(
        judge_result.get("question")
        or judge_result.get("condition_description")
        or ""
    ).lower().replace("_", " ").replace("-", " ")


def _is_access_control_authorization_gap(
    judge_result: Dict[str, Any],
) -> bool:
    if _is_exclusion_judge(judge_result):
        return False
    text = _access_control_condition_text(judge_result)
    authorization_terms = (
        "authorization",
        "authorisation",
        "permission",
        "privilege",
        "entitlement",
        "owner",
        "role",
        "signature",
        "caller",
        "sender validation",
    )
    gap_terms = (
        "missing",
        "absent",
        "bypass",
        "lacks",
        "unauthorized",
        "unprotected",
        "non authorized",
        "self authorization",
        "impersonation",
    )
    return any(term in text for term in authorization_terms) and any(
        term in text for term in gap_terms
    )


def _is_access_control_authorization_support(
    judge_result: Dict[str, Any],
) -> bool:
    if not _is_exclusion_judge(judge_result):
        return False
    text = _access_control_condition_text(judge_result)
    return any(term in text for term in (
        "recognized authorization",
        "recognised authorization",
        "fully backed",
        "valid delegation",
        "valid signature",
        "legitimate governance",
        "legitimate admin",
        "sufficient authorization",
    ))


def _is_access_control_alternative_explanation(
    judge_result: Dict[str, Any],
) -> bool:
    if not _is_exclusion_judge(judge_result):
        return False
    text = _access_control_condition_text(judge_result)
    return any(term in text for term in (
        "fully explained by",
        "primary mechanism",
        "non access control",
        "own assets",
        "proportional to legitimate",
        "alternative mechanism",
    ))


def _early_stop_decision(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
    judge_values: Dict[str, bool],
    enabled: bool,
    policy: str,
) -> Dict[str, Any]:
    if not enabled or policy != "conservative_negative":
        return {"triggered": False}

    decisive_core_false: List[Dict[str, Any]] = []
    decisive_exclusion_true: List[Dict[str, Any]] = []
    try:
        relevant_ids = set(extract_logic_names(plan.emit_logic))
    except Exception:
        relevant_ids = {str(step.id) for step in plan.judge_steps}
    if not relevant_ids:
        relevant_ids = {str(step.id) for step in plan.judge_steps}

    for result in list(judge_results or []):
        if result.get("skipped_by_early_stop"):
            continue
        judge_id = str(result.get("id") or "")
        if judge_id not in relevant_ids:
            continue
        answer = result.get("answer")
        confidence = str(result.get("confidence") or "medium").lower().strip()
        if _answer_is_uncertain(answer) or not _is_decisive_confidence(confidence):
            continue
        summary = _judge_summary_for_aggregation(result)
        if _is_exclusion_judge(result):
            if answer is True:
                decisive_exclusion_true.append(summary)
        elif answer is False:
            decisive_core_false.append(summary)

    reason = ""
    if len(decisive_core_false) >= 2:
        reason = "two_decisive_core_false"
    elif len(decisive_exclusion_true) >= 2:
        reason = "two_decisive_exclusion_true"
    elif decisive_core_false and decisive_exclusion_true:
        reason = "core_false_plus_exclusion_true"
    if not reason:
        return {"triggered": False}
    if _optimistic_emit_possible(plan, judge_values):
        return {"triggered": False, "reason": "optimistic_emit_still_possible"}
    return {
        "triggered": True,
        "reason": reason,
        "decisive_core_false": decisive_core_false,
        "decisive_exclusion_true": decisive_exclusion_true,
    }


def _optimistic_emit_possible(plan: EvidencePlan, judge_values: Dict[str, bool]) -> bool:
    optimistic_values = {
        str(step.id): bool(judge_values.get(step.id, True))
        for step in plan.judge_steps
    }
    try:
        return bool(safe_eval_bool_expr(plan.emit_logic, optimistic_values))
    except Exception:
        return True


def _skipped_judge_result(
    step: JudgeStep,
    stop_decision: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": step.id,
        "judge_id": step.id,
        "condition_id": step.condition_id or step.id,
        "question": step.question,
        "answer": False,
        "expected_answer": step.expected_answer,
        "satisfied": _judge_satisfied(False, step.expected_answer),
        "reason": "Skipped by conservative runtime early stop after decisive negative evidence.",
        "confidence": "medium",
        "evidence_refs": list(step.default_evidence_refs or step.evidence_refs or []),
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "missing_evidence": [],
        "tool_calls": [],
        "judge_rounds": [],
        "view_render_metadata": [],
        "selected_view_names": [],
        "selected_views_hash": "",
        "skipped_by_early_stop": True,
        "early_stop_reason": stop_decision.get("reason", ""),
    }


def _skipped_judge_trace(
    step: JudgeStep,
    stop_decision: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "judge_id": step.id,
        "condition_id": step.condition_id or step.id,
        "skipped_by_early_stop": True,
        "early_stop_reason": stop_decision.get("reason", ""),
        "judge_calls": [],
        "tool_calls": [],
        "ignored_tool_requests": [],
        "view_render_metadata": [],
        "all_missing_evidence_debug": [],
        "final_answer": False,
    }


def _find_single_blocker_near_miss(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
    should_emit: bool,
    emit_logic_error: str | None,
) -> Dict[str, Any] | None:
    if emit_logic_error or should_emit:
        return None

    try:
        relevant_ids = set(extract_logic_names(plan.emit_logic))
    except Exception:
        relevant_ids = {str(result.get("id") or "") for result in judge_results}
    if not relevant_ids:
        relevant_ids = {str(result.get("id") or "") for result in judge_results}

    blockers: List[Dict[str, Any]] = []
    relevant_seen = 0
    for index, result in enumerate(judge_results):
        judge_id = str(result.get("id") or "")
        if judge_id not in relevant_ids:
            continue
        relevant_seen += 1
        answer = result.get("answer")
        is_exclusion = _is_exclusion_judge(result)
        blocker_type = ""
        if is_exclusion:
            if answer is True:
                blocker_type = "exclusion_true"
        else:
            if answer is False:
                blocker_type = "core_false"
            elif _answer_is_uncertain(answer):
                blocker_type = "core_uncertain"
        if not blocker_type:
            continue
        summary = _judge_summary_for_aggregation(result)
        summary.update({
            "judge_index": index,
            "judge_id": judge_id,
            "blocker_type": blocker_type,
        })
        blockers.append(summary)

    if relevant_seen == 0 or not blockers:
        return None

    judge_values = {
        str(result.get("id") or ""): result.get("answer") is True
        for result in judge_results
        if str(result.get("id") or "")
    }
    decisive_blockers: List[Dict[str, Any]] = []
    for blocker in blockers:
        counterfactual = dict(judge_values)
        blocker_id = str(blocker.get("judge_id") or "")
        counterfactual[blocker_id] = bool(
            blocker.get("blocker_type") != "exclusion_true"
        )
        try:
            becomes_emit = safe_eval_bool_expr(plan.emit_logic, counterfactual)
        except Exception:
            return None
        if becomes_emit:
            decisive_blockers.append(blocker)
    if len(decisive_blockers) != 1:
        return None

    blocker = decisive_blockers[0]
    blocker["near_miss_policy"] = (
        "single verdict-critical blocker; rerun only this local judge with "
        "expanded evidence and constrained follow-up"
    )
    blocker["pre_escalation_should_emit"] = bool(should_emit)
    return blocker


def _find_near_miss_escalations(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
    should_emit: bool,
    emit_logic_error: str | None,
    attack_label: str = "",
) -> List[Dict[str, Any]]:
    single = _find_single_blocker_near_miss(
        plan=plan,
        judge_results=judge_results,
        should_emit=should_emit,
        emit_logic_error=emit_logic_error,
    )
    if single:
        return [
            _augment_single_near_miss_source_followup(
                single,
                judge_results=judge_results,
                attack_label=attack_label,
            )
        ]
    return []


def _augment_single_near_miss_source_followup(
    near_miss: Dict[str, Any],
    *,
    judge_results: List[Dict[str, Any]],
    attack_label: str = "",
) -> Dict[str, Any]:
    out = dict(near_miss)
    label = normalize_attack_label(attack_label, default="")
    if label == "reentrancy":
        condition_id = str(
            out.get("condition_id") or out.get("judge_id") or ""
        ).strip().upper()
        if condition_id == "C3":
            suggested = _unique_strings([
                *list(out.get("suggested_followup_views") or []),
                "reentrancy_state_order_view",
            ])
            out["suggested_followup_views"] = suggested
            diagnostic_text = " ".join([
                str(out.get("reason") or ""),
                *[str(value or "") for value in list(out.get("missing_evidence") or [])],
            ]).lower()
            if re.search(
                r"(?:no|missing|absent|unavailable|insufficient).{0,60}"
                r"(?:state|slot|storage|ordering|telemetry|source)|"
                r"(?:structural[- ]only|source semantics unresolved)",
                diagnostic_text,
            ):
                by_id = _judge_results_by_condition(judge_results)
                probe_ids = _reentrancy_source_probe_ids(by_id)
                if probe_ids:
                    out["force_source_followup"] = True
                    out["source_probe_evidence_ids"] = probe_ids
                    out["source_followup_policy"] = (
                        "reentrancy C3 lacks state/order telemetry; inspect the "
                        "candidate-local function semantics before re-judging"
                    )
        return out

    if label != "access_control":
        return out

    by_id = _judge_results_by_condition(judge_results)
    condition_id = str(out.get("condition_id") or out.get("judge_id") or "").upper()
    target = by_id.get(condition_id)
    if not target or not (
        _is_access_control_authorization_gap(target)
        or _is_access_control_alternative_explanation(target)
    ):
        return out

    positive_anchor_ids = [
        key
        for key, row in by_id.items()
        if not _is_exclusion_judge(row)
        and row.get("answer") is True
        and _is_decisive_confidence(str(row.get("confidence") or "medium"))
    ]
    boundary_ids = [
        key
        for key, row in by_id.items()
        if _is_access_control_authorization_gap(row)
        or _is_access_control_alternative_explanation(row)
    ]
    preferred = tuple(_unique_strings([
        condition_id,
        *positive_anchor_ids,
        *boundary_ids,
    ]))
    probe_ids = _cluster_source_probe_ids(
        by_id=by_id,
        preferred_condition_ids=preferred,
    )
    if not probe_ids:
        return out

    out["force_source_followup"] = True
    out["source_probe_evidence_ids"] = probe_ids
    out.setdefault("cluster_id", "access_control_authorization_boundary")
    out.setdefault("cluster_members", [condition_id])
    out["source_followup_policy"] = (
        "single access-control authorization-boundary blocker; force a precise "
        "read_function_chunk probe before re-judging"
    )
    return out


def _find_access_control_authorization_cluster_near_misses(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
    should_emit: bool,
    emit_logic_error: str | None,
    attack_label: str = "",
) -> List[Dict[str, Any]]:
    if should_emit or emit_logic_error:
        return []
    if normalize_attack_label(attack_label, default="") != "access_control":
        return []

    by_id = _judge_results_by_condition(judge_results)
    try:
        relevant_ids = {
            str(value or "").upper()
            for value in extract_logic_names(plan.emit_logic)
        }
    except Exception:
        relevant_ids = set(by_id)
    relevant_rows = {
        key: row
        for key, row in by_id.items()
        if not relevant_ids or key in relevant_ids
    }
    authorization_gaps = [
        row
        for row in relevant_rows.values()
        if _is_access_control_authorization_gap(row)
        and row.get("answer") in {False, "uncertain"}
    ]
    alternative_explanations = [
        row
        for row in relevant_rows.values()
        if _is_access_control_alternative_explanation(row)
        and row.get("answer") is True
        and _is_decisive_confidence(str(row.get("confidence") or "medium"))
    ]
    if not authorization_gaps or not alternative_explanations:
        return []
    if any(
        _is_access_control_authorization_support(row)
        and row.get("answer") is True
        and _is_decisive_confidence(str(row.get("confidence") or "medium"))
        for row in relevant_rows.values()
    ):
        return []
    authorization_gap = authorization_gaps[0]
    alternative_explanation = alternative_explanations[0]
    if _classify_condition_failure(
        authorization_gap,
        is_exclusion=False,
    ) == "true_contradiction":
        return []

    other_core = [
        row
        for row in relevant_rows.values()
        if not _is_exclusion_judge(row) and row is not authorization_gap
    ]
    positive_anchors = [
        row
        for row in other_core
        if row.get("answer") is True
        and _is_decisive_confidence(str(row.get("confidence") or "medium"))
    ]
    if not other_core or len(positive_anchors) != len(other_core):
        return []

    gap_id = str(
        authorization_gap.get("condition_id")
        or authorization_gap.get("id")
        or ""
    ).upper()
    alternative_id = str(
        alternative_explanation.get("condition_id")
        or alternative_explanation.get("id")
        or ""
    ).upper()
    positive_anchor_ids = [
        str(row.get("condition_id") or row.get("id") or "").upper()
        for row in positive_anchors
    ]
    probe_ids = _cluster_source_probe_ids(
        by_id=relevant_rows,
        preferred_condition_ids=tuple(_unique_strings([
            *positive_anchor_ids,
            alternative_id,
            gap_id,
        ])),
    )
    cluster_members = [gap_id, alternative_id]
    out: List[Dict[str, Any]] = []
    for result in (authorization_gap, alternative_explanation):
        summary = _judge_summary_for_aggregation(result)
        summary.update({
            "judge_index": int(result.get("_judge_index", -1)),
            "judge_id": str(result.get("id") or ""),
            "blocker_type": (
                "exclusion_true" if _is_exclusion_judge(result) else "core_false"
            ),
            "near_miss_policy": "failure_cluster_escalation",
            "cluster_id": "access_control_authorization_boundary",
            "cluster_members": list(cluster_members),
            "cluster_reason": (
                "An authorization-gap core condition and an alternative-"
                "explanation exclusion conflict while every other emit-logic "
                "core condition is supported. Treat them as one semantic "
                "authorization-boundary evidence cluster."
            ),
            "force_source_followup": True,
            "source_probe_evidence_ids": list(probe_ids),
            "pre_escalation_should_emit": bool(should_emit),
        })
        out.append(summary)
    return out


def _make_near_miss_escalation_step(
    step: JudgeStep,
    near_miss: Dict[str, Any],
) -> JudgeStep:
    data = step.to_dict()
    expanded_refs = _near_miss_default_refs(step, near_miss)
    data["question"] = _near_miss_question(step.question, near_miss)
    data["evidence_refs"] = expanded_refs
    data["default_evidence_refs"] = expanded_refs
    data["allowed_followup_views"] = _near_miss_followup_views(step, near_miss)
    data["allowed_tools"] = _near_miss_allowed_tools(step)
    data["max_followups"] = (
        0
        if near_miss.get("single_rejudge_only")
        else max(1, int(step.max_followups or 0))
    )
    return JudgeStep.from_dict(data)


def _near_miss_question(question: str, near_miss: Dict[str, Any]) -> str:
    blocker_type = str(near_miss.get("blocker_type") or "")
    policy = str(near_miss.get("near_miss_policy") or "")
    common = """

Near-miss escalation:
This local judge is verdict-critical. Re-evaluate this condition more carefully
using the expanded evidence below. Do not change the meaning of the condition.
If constrained follow-up evidence is necessary, answer "uncertain" and request
it.
"""
    if policy == "failure_cluster_escalation":
        common += """
This condition is part of a failure cluster. Multiple local blockers may share
one unresolved evidence gap, so do not treat the sibling blocker as independent
proof. Focus on whether this condition is truly contradicted, merely missing
source/local semantics, or already semantically covered by stronger positive
anchors.
"""
    elif policy == "dynamic_aggregator_source_followup":
        request_reason = _truncate_text(
            str(near_miss.get("aggregator_followup_reason") or ""),
            700,
        )
        common += f"""
The global aggregator requested one source-evidence follow-up for this condition.
Its request is a routing hint, not a judgment and not evidence. Independently
re-evaluate only this condition using the preloaded source observation, and make
the final local answer without asking for another round.
Aggregator request reason: {request_reason or "important evidence gap"}
"""
    if blocker_type.startswith("exclusion"):
        specific = """
This is an exclusion check. Answer true only when concrete evidence directly
shows this exclusion applies to the sensitive target function or protected
effect. Do not answer true merely because the top-level entry call is public,
because callbacks/reentrancy are present, or because another exploit family is
plausible. If the public/reentrant/callback path is only a vehicle to reach a
protected target or release protected value, answer false for the exclusion.
"""
    else:
        specific = """
This is a core-condition check. Answer false only when the packet evidence is
complete enough and the required signal is genuinely absent. If the packet
contains a relevant critical call, value release, unknown selector, protected
state change, authorization slot, or beneficiary/controller link but the local
context/source semantics are unresolved, request follow-up evidence instead of
returning a medium/high-confidence false.
"""
    return f"{question.rstrip()}{common}{specific}".strip()


def _near_miss_default_refs(
    step: JudgeStep,
    near_miss: Dict[str, Any],
    max_refs: int = 10,
) -> List[str]:
    refs = list(step.default_evidence_refs or step.evidence_refs or [])
    allowed = list(step.allowed_followup_views or [])
    blocker_type = str(near_miss.get("blocker_type") or "")
    if blocker_type.startswith("exclusion"):
        priority = [
            "operation_summary_view",
            "evidence_adequacy_view",
            "critical_call_view",
            "critical_call_argument_view",
            "value_release_view",
            "beneficiary_controller_view",
            "reentrancy_state_order_summary_view",
            "trace_outline_view",
            "unknown_selector_view",
            "semantic_state_delta_view",
            "state_change_view",
            "event_view",
            "participant_net_delta_view",
            "contribution_vs_payout_view",
            "external_fundflow_view",
            "transfer_event_view",
        ]
    else:
        priority = [
            "operation_summary_view",
            "evidence_adequacy_view",
            "critical_call_view",
            "critical_call_argument_view",
            "value_release_view",
            "beneficiary_controller_view",
            "unknown_selector_view",
            "trace_outline_view",
            "reentrancy_state_order_summary_view",
            "semantic_state_delta_view",
            "state_change_view",
            "event_view",
            "participant_net_delta_view",
            "contribution_vs_payout_view",
            "external_fundflow_view",
            "transfer_event_view",
        ]
    for view in priority:
        if view in allowed and view not in refs:
            refs.append(view)
        if len(refs) >= max_refs:
            break
    return refs[:max_refs]


def _near_miss_followup_views(
    step: JudgeStep,
    near_miss: Dict[str, Any],
) -> List[str]:
    refs = list(step.allowed_followup_views or [])
    for view in _near_miss_default_refs(step, near_miss, max_refs=12):
        if view not in refs:
            refs.append(view)
    return refs


def _near_miss_allowed_tools(step: JudgeStep) -> List[str]:
    return list(step.allowed_tools or [])


def _merge_escalated_step_trace(
    original_trace: Dict[str, Any],
    escalated_trace: Dict[str, Any],
    *,
    near_miss: Dict[str, Any],
) -> Dict[str, Any]:
    merged = copy.deepcopy(original_trace or {})
    merged.setdefault("judge_calls", [])
    merged.setdefault("tool_calls", [])
    merged.setdefault("ignored_tool_requests", [])
    merged.setdefault("view_render_metadata", [])
    merged.setdefault("judge_parse_metadata", [])
    merged.setdefault("judge_output_exhaustions", [])
    merged.setdefault("source_decisiveness_guards", [])
    merged.setdefault("all_missing_evidence_debug", [])
    merged.setdefault("evidence_id_validations", [])
    merged.setdefault("invalid_evidence_ids", [])

    round_offset = len(merged.get("judge_calls", []) or [])
    for item in list(escalated_trace.get("judge_calls", []) or []):
        updated = copy.deepcopy(item)
        updated["round"] = int(updated.get("round", 0) or 0) + round_offset
        updated["near_miss_escalation"] = True
        merged["judge_calls"].append(updated)
    for item in list(escalated_trace.get("tool_calls", []) or []):
        updated = copy.deepcopy(item)
        updated["round"] = int(updated.get("round", 0) or 0) + round_offset
        updated["near_miss_escalation"] = True
        merged["tool_calls"].append(updated)
    for key in (
        "ignored_tool_requests",
        "view_render_metadata",
        "judge_parse_metadata",
        "judge_output_exhaustions",
        "source_decisiveness_guards",
        "evidence_id_validations",
    ):
        for item in list(escalated_trace.get(key, []) or []):
            updated = copy.deepcopy(item)
            if isinstance(updated, dict) and "round" in updated:
                updated["round"] = int(updated.get("round", 0) or 0) + round_offset
            if isinstance(updated, dict):
                updated["near_miss_escalation"] = True
            merged[key].append(updated)
    merged["invalid_evidence_ids"] = _unique_strings(
        list(merged.get("invalid_evidence_ids", []) or [])
        + list(escalated_trace.get("invalid_evidence_ids", []) or [])
    )

    merged["all_missing_evidence_debug"] = dedupe_missing_evidence(
        list(merged.get("all_missing_evidence_debug", []) or [])
        + list(escalated_trace.get("all_missing_evidence_debug", []) or [])
    )
    merged["final_answer"] = escalated_trace.get("final_answer")
    merged["near_miss_escalation"] = {
        "enabled": True,
        "blocker_type": near_miss.get("blocker_type"),
        "pre_escalation_answer": near_miss.get("answer"),
        "pre_escalation_confidence": near_miss.get("confidence"),
        "round_offset": round_offset,
    }
    return merged


def _tool_observations_from_step_trace(
    step_trace: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for call in list(dict(step_trace or {}).get("tool_calls", []) or []):
        if not isinstance(call, dict):
            continue
        result = copy.deepcopy(call.get("observation"))
        if not isinstance(result, dict):
            result = {
                "tool_status": call.get("tool_status", "unknown"),
                "returned_evidence_ids": list(
                    call.get("returned_evidence_ids", []) or []
                ),
                "summary": copy.deepcopy(call.get("summary", {})),
            }
            if call.get("validation_error"):
                result["validation_error"] = call.get("validation_error")
        observations.append({
            "request": {
                "tool": call.get("tool"),
                "args": copy.deepcopy(call.get("args", {})),
                "reason": call.get("reason", ""),
            },
            "result": result,
            "history_source": "pre_escalation_step_trace",
            "source_round": call.get("round"),
        })
    return observations


def _materialize_escalated_judge_result(
    result: JudgeResult,
    step: JudgeStep,
    merged_trace: Dict[str, Any],
    *,
    state_input: Dict[str, Any] | None = None,
    stateful_runtime: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result_dict = result.to_dict()
    result_dict["condition_id"] = step.condition_id or step.id
    result_dict["tool_calls"] = list(merged_trace.get("tool_calls", []) or [])
    result_dict["judge_rounds"] = list(merged_trace.get("judge_calls", []) or [])
    result_dict["judge_output_exhaustions"] = list(
        merged_trace.get("judge_output_exhaustions", []) or []
    )
    result_dict["source_decisiveness_guards"] = list(
        merged_trace.get("source_decisiveness_guards", []) or []
    )
    result_dict["view_render_metadata"] = list(
        merged_trace.get("view_render_metadata", []) or []
    )
    result_dict["selected_view_names"] = list(
        merged_trace.get("selected_view_names", []) or []
    )
    result_dict["selected_views_hash"] = str(
        merged_trace.get("selected_views_hash", "") or ""
    )
    result_dict["all_missing_evidence_debug"] = list(
        merged_trace.get("all_missing_evidence_debug", []) or []
    )
    result_dict["ignored_tool_requests"] = list(
        merged_trace.get("ignored_tool_requests", []) or []
    )
    result_dict["invalid_evidence_ids"] = list(
        merged_trace.get("invalid_evidence_ids", []) or []
    )
    result_dict["evidence_id_validations"] = list(
        merged_trace.get("evidence_id_validations", []) or []
    )
    _attach_stateful_result_fields(
        result_dict,
        merged_trace,
        step,
        state_input=dict(state_input or {}),
        stateful_runtime=dict(stateful_runtime or {}),
    )
    return result_dict


def _judge_results_by_condition(
    judge_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, result in enumerate(list(judge_results or [])):
        key = str(result.get("condition_id") or result.get("id") or "").upper()
        if not key:
            continue
        item = dict(result)
        item["_judge_index"] = index
        by_id[key] = item
    return by_id


def _cluster_source_probe_ids(
    *,
    by_id: Dict[str, Dict[str, Any]],
    preferred_condition_ids: tuple[str, ...],
    limit: int = 6,
) -> List[str]:
    ids: List[str] = []
    for condition_id in preferred_condition_ids:
        row = by_id.get(str(condition_id).upper())
        if not row:
            continue
        ids.extend(list(row.get("supporting_evidence_ids", []) or []))
    return _unique_strings(ids)[:limit]


def _reentrancy_source_probe_ids(
    by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Prioritize source probes on one selected reentrancy candidate path."""
    c1 = dict(by_id.get("C1") or {})
    c2 = dict(by_id.get("C2") or {})
    candidate_state = dict((c1.get("state_output") or {}).get(
        "reentrancy_candidate"
    ) or {})
    selected_ids = set(_unique_strings(
        candidate_state.get("selected_candidate_ids") or []
    ))
    candidates = [
        dict(item)
        for item in list(candidate_state.get("candidates") or [])
        if isinstance(item, dict)
        and (
            not selected_ids
            or str(item.get("candidate_id") or "") in selected_ids
        )
    ]
    probe_ids: List[str] = []
    for candidate in candidates:
        probe_ids.append(_source_probe_call_id(candidate.get("outer_call_id")))

    effect_state = dict((c2.get("state_output") or {}).get(
        "reentrancy_value_effect_summary"
    ) or {})
    for result in list(effect_state.get("candidate_results") or []):
        if not isinstance(result, dict):
            continue
        if selected_ids and str(result.get("candidate_id") or "") not in selected_ids:
            continue
        for evidence_id in list(result.get("evidence_ids") or []):
            probe_ids.append(_source_probe_call_id(evidence_id))

    for candidate in candidates:
        probe_ids.append(_source_probe_call_id(candidate.get("reentry_call_id")))
    for candidate in candidates:
        probe_ids.append(_source_probe_call_id(candidate.get("external_edge_id")))

    if not any(probe_ids):
        for condition_id in ("C1", "C2", "C3"):
            for evidence_id in list(
                (by_id.get(condition_id) or {}).get("supporting_evidence_ids") or []
            ):
                probe_ids.append(_source_probe_call_id(evidence_id))
    return _unique_strings(value for value in probe_ids if value)[:6]


def _source_probe_call_id(value: Any) -> str:
    evidence_id = _call_evidence_id(value)
    if ":call:" in evidence_id:
        return f"call:{evidence_id.rsplit(':call:', 1)[1]}"
    return evidence_id if evidence_id.startswith("call:") else ""


def _source_probe_evidence_ids(near_miss: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    ids.extend(list(near_miss.get("source_probe_evidence_ids", []) or []))
    ids.extend(list(near_miss.get("supporting_evidence_ids", []) or []))
    return _unique_strings(ids)


def _active_dynamic_blocker_condition_ids(
    judge_results: List[Dict[str, Any]],
) -> set[str]:
    blockers: set[str] = set()
    for result in list(judge_results or []):
        answer = result.get("answer")
        is_exclusion = _is_exclusion_judge(result)
        is_blocker = (
            (is_exclusion and answer is True)
            or (not is_exclusion and (answer is False or _answer_is_uncertain(answer)))
        )
        if not is_blocker:
            continue
        for value in (result.get("condition_id"), result.get("id")):
            key = str(value or "").upper().strip()
            if key:
                blockers.add(key)
    return blockers


def _near_miss_escalated_condition_ids(
    near_miss_escalations: List[Dict[str, Any]],
) -> set[str]:
    escalated: set[str] = set()
    for item in list(near_miss_escalations or []):
        if not isinstance(item, dict) or item.get("attempted") is False:
            continue
        for value in (item.get("condition_id"), item.get("judge_id")):
            key = str(value or "").upper().strip()
            if key:
                escalated.add(key)
    return escalated


def _attempted_source_evidence_ids_by_condition(
    judge_results: List[Dict[str, Any]],
    *,
    near_miss_escalations: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    attempted: Dict[str, List[str]] = {}

    def add(keys: List[Any], evidence_ids: List[Any]) -> None:
        normalized_ids = _unique_strings(evidence_ids)
        for raw_key in keys:
            key = str(raw_key or "").upper().strip()
            if not key:
                continue
            attempted[key] = _unique_strings(
                list(attempted.get(key, []) or []) + normalized_ids
            )

    for result in list(judge_results or []):
        ids: List[Any] = []
        for call in list(result.get("tool_calls", []) or []):
            if str(call.get("tool") or "") != "read_function_chunk":
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            ids.append(args.get("evidence_id"))
            ids.extend(list(args.get("evidence_ids", []) or []))
        add([result.get("condition_id"), result.get("id")], ids)

    for item in list(near_miss_escalations or []):
        if not isinstance(item, dict) or item.get("attempted") is False:
            continue
        add(
            [item.get("condition_id"), item.get("judge_id")],
            list(item.get("source_probe_evidence_ids", []) or []),
        )
    return attempted


def _near_miss_cluster_members(
    near_miss_escalations: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    members_by_cluster: Dict[str, List[str]] = {}
    for item in list(near_miss_escalations or []):
        if not isinstance(item, dict):
            continue
        cluster_id = str(item.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        members_by_cluster[cluster_id] = _unique_strings(
            list(members_by_cluster.get(cluster_id, []) or [])
            + list(item.get("cluster_members", []) or [])
            + [item.get("condition_id"), item.get("judge_id")]
        )
    return members_by_cluster


def _prepare_dynamic_aggregation_followup_requests(
    requests: List[Dict[str, Any]],
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
    near_miss_escalations: List[Dict[str, Any]] | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_condition = _judge_results_by_condition(judge_results)
    step_by_key: Dict[str, JudgeStep] = {}
    for step in list(plan.judge_steps or []):
        for key in (step.id, step.condition_id or ""):
            normalized = str(key or "").upper().strip()
            if normalized:
                step_by_key[normalized] = step

    known_evidence_ids: List[str] = []
    for result in list(judge_results or []):
        known_evidence_ids.extend(list(result.get("supporting_evidence_ids", []) or []))
        known_evidence_ids.extend(list(result.get("contradicting_evidence_ids", []) or []))
    known_evidence = {
        value
        for value in _unique_strings(known_evidence_ids)
        if _looks_like_evidence_id(value)
        and not value.lower().startswith("source_chunk:")
    }
    active_blockers = _active_dynamic_blocker_condition_ids(judge_results)
    escalated_conditions = _near_miss_escalated_condition_ids(
        near_miss_escalations or []
    )
    attempted_source_ids = _attempted_source_evidence_ids_by_condition(
        judge_results,
        near_miss_escalations=near_miss_escalations or [],
    )
    cluster_members_by_id = _near_miss_cluster_members(
        near_miss_escalations or []
    )

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_judges: set[str] = set()
    for raw in list(requests or []):
        if not isinstance(raw, dict):
            rejected.append({
                "request": raw,
                "rejection_reason": "request_must_be_an_object",
            })
            continue
        condition_key = str(
            raw.get("condition_id") or raw.get("judge_id") or ""
        ).upper().strip()
        step = step_by_key.get(condition_key)
        if step is None:
            rejected.append({
                "request": copy.deepcopy(raw),
                "rejection_reason": "unknown_condition_id",
            })
            continue
        judge_key = str(step.id).upper()
        if judge_key in seen_judges:
            rejected.append({
                "request": copy.deepcopy(raw),
                "rejection_reason": "duplicate_condition_request",
            })
            continue
        result = by_condition.get(
            str(step.condition_id or step.id).upper()
        ) or by_condition.get(judge_key)
        if result is None:
            rejected.append({
                "request": copy.deepcopy(raw),
                "rejection_reason": "condition_has_no_judge_result",
            })
            continue

        target_condition = str(step.condition_id or step.id).upper().strip()
        if target_condition not in active_blockers and judge_key not in active_blockers:
            rejected.append({
                "request": copy.deepcopy(raw),
                "rejection_reason": "condition_is_not_an_active_blocker",
            })
            continue

        requested_ids = list(
            raw.get("evidence_ids")
            or raw.get("source_probe_evidence_ids")
            or []
        )
        if not requested_ids:
            rejected.append({
                "request": copy.deepcopy(raw),
                "rejection_reason": "explicit_evidence_ids_required",
            })
            continue
        normalized_ids = [
            evidence_id
            for evidence_id in _unique_strings(requested_ids)
            if evidence_id in known_evidence
        ][:2]
        if not normalized_ids:
            rejected.append({
                "request": copy.deepcopy(raw),
                "rejection_reason": "no_known_source_probe_evidence_ids",
            })
            continue

        cluster_id = str(raw.get("cluster_id") or "").strip()
        cluster_members = _unique_strings(
            list(raw.get("cluster_members", []) or [])
            + list(cluster_members_by_id.get(cluster_id, []) or [])
        )
        normalized_cluster_members = {
            str(member or "").upper().strip()
            for member in cluster_members
            if str(member or "").strip()
        }
        shared_blocker_members = sorted(
            normalized_cluster_members.intersection(active_blockers)
        )
        shared_blocker_gap = bool(
            cluster_id
            and len(shared_blocker_members) >= 2
            and (
                target_condition in shared_blocker_members
                or judge_key in shared_blocker_members
            )
        )
        target_was_escalated = bool(
            target_condition in escalated_conditions
            or judge_key in escalated_conditions
        )
        attempted_ids = set(
            attempted_source_ids.get(target_condition, [])
            + attempted_source_ids.get(judge_key, [])
        )
        new_evidence_ids = [
            evidence_id
            for evidence_id in normalized_ids
            if evidence_id not in attempted_ids
        ]
        if target_was_escalated and not new_evidence_ids:
            rejected.append({
                "request": copy.deepcopy(raw),
                "rejection_reason": "duplicate_after_near_miss_escalation",
                "attempted_evidence_ids": sorted(attempted_ids),
            })
            continue
        if not target_was_escalated and not shared_blocker_gap:
            rejected.append({
                "request": copy.deepcopy(raw),
                "rejection_reason": "aggregator_followup_requires_shared_blocker_gap",
                "active_blocker_cluster_members": shared_blocker_members,
            })
            continue

        seen_judges.add(judge_key)
        accepted_ids = new_evidence_ids if target_was_escalated else normalized_ids
        accepted.append({
            "condition_id": step.condition_id or step.id,
            "judge_id": step.id,
            "judge_index": int(result.get("_judge_index", -1)),
            "reason": _truncate_text(str(raw.get("reason") or ""), 600),
            "evidence_ids": accepted_ids,
            "cluster_id": cluster_id,
            "cluster_members": cluster_members,
            "followup_policy_basis": (
                "new_evidence_after_near_miss"
                if target_was_escalated
                else "shared_multi_blocker_gap"
            ),
            "near_miss_already_escalated": target_was_escalated,
            "previously_attempted_evidence_ids": sorted(attempted_ids),
        })
    return accepted, rejected


def _analyze_dynamic_aggregation_context(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
    attack_label: str,
    near_miss_escalations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    logic_names = []
    try:
        logic_names = sorted(extract_logic_names(plan.emit_logic))
    except Exception:
        logic_names = [str(jr.get("id") or "") for jr in judge_results]
    relevant = set(logic_names or [str(jr.get("id") or "") for jr in judge_results])

    assessments: List[Dict[str, Any]] = []
    true_core_ids: List[str] = []
    true_exclusion_ids: List[str] = []
    source_followup_by_condition = _source_followup_by_condition(judge_results)
    cluster_hints = _near_miss_cluster_hints(near_miss_escalations)
    for jr in list(judge_results or []):
        judge_id = str(jr.get("id") or "")
        condition_id = str(jr.get("condition_id") or judge_id)
        if relevant and judge_id not in relevant:
            continue
        is_exclusion = _is_exclusion_judge(jr)
        answer = jr.get("answer")
        if answer is True and not is_exclusion:
            true_core_ids.append(condition_id)
        if answer is True and is_exclusion:
            true_exclusion_ids.append(condition_id)

        is_blocker = (
            (is_exclusion and answer is True)
            or ((not is_exclusion) and (answer is False or _answer_is_uncertain(answer)))
        )
        if not is_blocker:
            continue

        failure_type = _classify_condition_failure(jr, is_exclusion=is_exclusion)
        cluster_hint = cluster_hints.get(condition_id) or cluster_hints.get(judge_id) or {}
        cluster_id = str(cluster_hint.get("cluster_id") or f"condition:{condition_id}")
        assessment = {
            "condition_id": condition_id,
            "judge_id": judge_id,
            "answer": answer,
            "confidence": jr.get("confidence", "medium"),
            "is_exclusion": is_exclusion,
            "failure_type_hint": failure_type,
            "cluster_id": cluster_id,
            "cluster_hint_source": str(cluster_hint.get("source") or ""),
            "source_followup_forced": bool(cluster_hint.get("source_followup_forced")),
            "source_followup_performed": bool(
                source_followup_by_condition.get(condition_id)
                or source_followup_by_condition.get(judge_id)
            ),
            "supporting_evidence_count": len(list(jr.get("supporting_evidence_ids", []) or [])),
            "contradicting_evidence_count": len(list(jr.get("contradicting_evidence_ids", []) or [])),
            "reason": _truncate_text(str(jr.get("reason") or ""), 360),
            "near_miss_escalated": bool(jr.get("near_miss_escalated")),
            "judge_output_exhaustion_count": len(
                list(jr.get("judge_output_exhaustions", []) or [])
            ),
        }
        assessments.append(assessment)

    clusters = _cluster_assessments(assessments, near_miss_escalations)
    typed_anchors = _detect_typed_positive_anchors(
        judge_results,
        attack_label=attack_label,
    )
    pending_forced_source_followup = [
        cluster
        for cluster in clusters
        if cluster.get("source_followup_forced")
        and not cluster.get("source_followup_performed")
    ]
    hard_blockers, soft_blockers = _partition_dynamic_blockers(assessments)
    return {
        "schema_version": "evotx.dynamic_aggregation_context.v1",
        "attack_label": normalize_attack_label(attack_label, default=""),
        "emit_logic": plan.emit_logic,
        "logic_names": logic_names,
        "true_core_ids": true_core_ids,
        "true_exclusion_ids": true_exclusion_ids,
        "positive_core_true_count": len(true_core_ids),
        "condition_assessments": assessments,
        "blocker_candidates": assessments,
        "hard_blockers": hard_blockers,
        "soft_blockers": soft_blockers,
        "clusters": clusters,
        "typed_positive_anchors": typed_anchors,
        "typed_positive_anchor_count": len(typed_anchors),
        "pending_forced_source_followup": pending_forced_source_followup,
        "near_miss_escalations": list(near_miss_escalations or []),
    }


def _source_followup_by_condition(
    judge_results: List[Dict[str, Any]],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for jr in list(judge_results or []):
        condition_id = str(jr.get("condition_id") or jr.get("id") or "")
        count = 0
        for call in list(jr.get("tool_calls", []) or []):
            if str(call.get("tool") or "") == "read_function_chunk":
                count += 1
        if condition_id:
            out[condition_id] = int(out.get(condition_id, 0) or 0) + count
        judge_id = str(jr.get("id") or "")
        if judge_id:
            out[judge_id] = int(out.get(judge_id, 0) or 0) + count
    return out


def _classify_condition_failure(
    judge_result: Dict[str, Any],
    *,
    is_exclusion: bool,
) -> str:
    answer = judge_result.get("answer")
    if _answer_is_uncertain(answer):
        return "missing_evidence"
    cfa = normalize_condition_feature_analysis(
        judge_result.get("condition_feature_analysis", {})
    )
    contradicting = list(judge_result.get("contradicting_evidence_ids", []) or [])
    contradicting_features = list(cfa.get("contradicting_features", []) or [])
    missing = (
        list(judge_result.get("missing_evidence", []) or [])
        + list(cfa.get("missing_required_features", []) or [])
    )
    partial = list(cfa.get("partial_match_features", []) or [])
    reason = str(judge_result.get("reason") or "").lower()
    if contradicting or (
        contradicting_features
        and not missing
        and any(
            token in " ".join(contradicting_features).lower()
            for token in ("authorized", "valid", "standard", "own position", "checked")
        )
    ):
        return "true_contradiction"
    if _reason_has_authorization_counterevidence(reason):
        return "true_contradiction"
    if missing:
        return "missing_evidence"
    if partial or any(
        token in reason
        for token in (
            "no evidence",
            "not shown",
            "not visible",
            "packet alone cannot",
            "unknown selector",
            "unresolved",
            "no signal",
        )
    ):
        return "over_narrow_or_unresolved"
    if is_exclusion and any(
        token in reason
        for token in ("better explained", "non-access", "reentrancy", "price", "oracle")
    ):
        return "alternative_mechanism_claim"
    return "unclassified"


def _reason_has_authorization_counterevidence(reason: str) -> bool:
    text = str(reason or "").lower()
    if not text:
        return False
    negative_prefixes = (
        "no evidence",
        "none of",
        "not shown",
        "no signal",
        "cannot determine",
        "packet alone cannot",
        "no authorization",
        "no access-control",
        "without evidence",
    )
    if any(prefix in text for prefix in negative_prefixes):
        return False
    return any(
        token in text
        for token in (
            "go through",
            "goes through",
            "visible check",
            "visible checks",
            "authorization check",
            "authorization checks",
            "caller check",
            "caller checks",
            "role check",
            "permission check",
            "signature check",
            "valid signature",
            "borrowallowed",
            "redeemallowed",
            "onlyowner",
            "authorized governance",
            "recognized authorized",
        )
    )


def _near_miss_cluster_hints(
    near_miss_escalations: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    hints: Dict[str, Dict[str, Any]] = {}
    for item in list(near_miss_escalations or []):
        cluster_id = str(item.get("cluster_id") or "")
        members = list(item.get("cluster_members", []) or [])
        condition_id = str(item.get("condition_id") or item.get("judge_id") or "")
        if condition_id and condition_id not in members:
            members.append(condition_id)
        if not cluster_id or not members:
            continue
        hint = {
            "cluster_id": cluster_id,
            "source": "near_miss_escalation",
            "source_followup_forced": bool(item.get("forced_source_followup")),
        }
        for member in members:
            if member:
                hints[str(member)] = dict(hint)
    return hints


def _cluster_assessments(
    assessments: List[Dict[str, Any]],
    near_miss_escalations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for item in assessments:
        by_cluster.setdefault(str(item.get("cluster_id") or ""), []).append(item)
    escalated_clusters = {
        str(item.get("cluster_id") or "")
        for item in list(near_miss_escalations or [])
        if item.get("cluster_id")
    }
    forced_source_clusters = {
        str(item.get("cluster_id") or "")
        for item in list(near_miss_escalations or [])
        if item.get("cluster_id") and item.get("forced_source_followup")
    }
    clusters: List[Dict[str, Any]] = []
    for cluster_id, items in sorted(by_cluster.items()):
        if not cluster_id:
            continue
        source_performed = any(bool(item.get("source_followup_performed")) for item in items)
        clusters.append({
            "cluster_id": cluster_id,
            "conditions": [item.get("condition_id") for item in items],
            "size": len(items),
            "failure_type_hints": sorted({
                str(item.get("failure_type_hint") or "") for item in items
            }),
            "source_followup_forced": cluster_id in forced_source_clusters,
            "source_followup_performed": source_performed,
            "near_miss_escalated": cluster_id in escalated_clusters,
        })
    return clusters


def _detect_typed_positive_anchors(
    judge_results: List[Dict[str, Any]],
    *,
    attack_label: str,
) -> List[Dict[str, Any]]:
    if normalize_attack_label(attack_label, default="") != "access_control":
        return []
    anchor_defs = {
        "explicit_authorization_boundary": (
            "owner", "role", "signature", "authorization", "permission", "whitelist",
            "caller validation", "onlyowner",
        ),
        "initializer_proxy_admin": (
            "initializer", "initialize", "proxy", "admin", "upgrade",
        ),
        "protected_asset_public_path": (
            "protocol-controlled", "shared asset", "victim asset", "protected asset",
            "public function", "treasury", "liquidity", "beyond entitlement",
        ),
        "approval_allowance_transfer": (
            "approval", "allowance", "transferfrom", "approval-setting",
        ),
        "callback_sender_validation": (
            "callback", "callback sender", "sender validation",
        ),
        "burn_mint_withdraw_sweep": (
            "burn", "mint", "withdraw", "redeem", "sweep", "release",
        ),
    }
    sources: Dict[str, List[str]] = {name: [] for name in anchor_defs}
    for jr in list(judge_results or []):
        if _is_exclusion_judge(jr):
            continue
        cid = str(jr.get("condition_id") or jr.get("id") or "")
        cfa = normalize_condition_feature_analysis(jr.get("condition_feature_analysis", {}))
        text_parts: List[str] = []
        if jr.get("answer") is True:
            text_parts.append(str(jr.get("reason") or ""))
            text_parts.extend(list(cfa.get("matched_features", []) or []))
            text_parts.extend(list(cfa.get("boundary_notes", []) or []))
        elif _is_access_control_authorization_gap(jr):
            text_parts.extend(list(cfa.get("partial_match_features", []) or []))
        text = " ".join(text_parts).lower()
        if not text:
            continue
        for name, keywords in anchor_defs.items():
            if any(keyword in text for keyword in keywords):
                sources[name].append(cid)
    anchors = []
    for name, condition_ids in sources.items():
        if condition_ids:
            anchors.append({
                "anchor": name,
                "condition_ids": sorted(set(condition_ids)),
            })
    return anchors


def _deterministic_dynamic_aggregation_decision(
    *,
    previous_verdict: str,
    analysis: Dict[str, Any],
    previous_reason: str,
) -> Dict[str, Any]:
    if previous_verdict == "attack":
        return {
            "verdict": "attack",
            "confidence": "medium",
            "override": False,
            "reason": previous_reason or "emit_logic_attack",
            "mode": "deterministic",
            "followup_requests": [],
        }
    if analysis.get("pending_forced_source_followup"):
        return {
            "verdict": "uncertain",
            "confidence": "medium",
            "override": previous_verdict != "uncertain",
            "reason": "dynamic_aggregation_pending_forced_source_followup",
            "mode": "deterministic",
            "followup_requests": [],
        }
    return {
        "verdict": previous_verdict,
        "confidence": "medium",
        "override": False,
        "reason": previous_reason or "dynamic_aggregation_no_override",
        "mode": "deterministic",
        "followup_requests": [],
    }


def _should_call_dynamic_aggregator_agent(
    *,
    previous_verdict: str,
    analysis: Dict[str, Any],
) -> bool:
    if previous_verdict == "attack":
        return False
    return bool(
        analysis.get("blocker_candidates")
        or analysis.get("clusters")
        or analysis.get("typed_positive_anchors")
    )


def _build_dynamic_aggregator_prompt(
    *,
    rule: EvolvingRule,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
    judge_values: Dict[str, bool],
    previous_verdict: str,
    verdict_aggregation: Dict[str, Any],
    analysis: Dict[str, Any],
    aggregation_round: int = 1,
    allow_followup_requests: bool = True,
    followup_context: Dict[str, Any] | None = None,
) -> str:
    attack_label = normalize_attack_label(
        str((rule.metadata or {}).get("attack_label", ""))
    )
    family_binding_policy = ""
    if attack_label == "access_control":
        family_binding_policy = """
Access-control binding policy:
- Before combining local results, identify one coherent authorization chain:
  actor/caller or beneficiary -> required authority -> sensitive capability and
  protected target/resource -> protected effect.
- Cross-condition support is valid only when evidence ids or explicit trace,
  proxy/delegatecall, callback, beneficiary/controller, or value-release links
  show that both conditions concern that same chain.
- Do not override a blocker by stitching together unrelated actors, calls,
  capabilities, assets, positions, or protocol components. Missing source may
  be a soft evidence gap; a missing coherent authorization chain is not.
"""
    rule_digest = {
        "rule_id": rule.rule_id,
        "version": rule.version,
        "name": rule.name,
        "description": _truncate_text(rule.description, 900),
        "conditions": [
            {
                "id": condition.id,
                "description": _truncate_text(condition.description, 500),
            }
            for condition in list(rule.conditions or [])
        ],
        "exclusion_conditions": [
            {
                "id": condition.id,
                "description": _truncate_text(condition.description, 500),
            }
            for condition in list(rule.exclusion_conditions or [])
        ],
        "decision_policy": _truncate_text(rule.decision_policy, 900),
        "metadata": {
            "attack_label": (rule.metadata or {}).get("attack_label", ""),
        },
    }
    judge_digest = [
        _compact_judge_for_aggregator(jr)
        for jr in list(judge_results or [])
    ]
    semantic_answers = {
        str(item.get("id") or item.get("condition_id") or ""): item.get("answer")
        for item in judge_digest
        if str(item.get("id") or item.get("condition_id") or "")
    }
    payload = {
        "rule": rule_digest,
        "plan": {
            "plan_id": plan.plan_id,
            "emit_logic": plan.emit_logic,
            "plan_note": _truncate_text(plan.plan_note, 700),
        },
        "semantic_answers": semantic_answers,
        "emit_logic_values": dict(judge_values or {}),
        "previous_verdict": previous_verdict,
        "previous_reason": verdict_aggregation.get("reason", ""),
        "aggregation_round": int(aggregation_round),
        "followup_requests_allowed": bool(allow_followup_requests),
        "completed_followup_round": dict(followup_context or {}),
        "dynamic_analysis": analysis,
        "judge_results": judge_digest,
    }
    if allow_followup_requests:
        followup_policy = """
- Near-miss escalation already owns single-blocker local evidence completion.
  Do not request another follow-up for a condition listed in
  dynamic_analysis.near_miss_escalations unless you provide an explicit packet
  evidence id that has not already been source-probed for that condition.
- For a condition not already near-miss escalated, request source follow-up only
  when at least two active blockers share one unresolved evidence gap. Provide a
  non-empty cluster_id and list all affected blocker ids in cluster_members.
- Use only condition ids and evidence ids present in the input. The runtime will
  reject duplicate probes, single-blocker repeats, invented ids, and non-blockers.
  It routes accepted requests to the owning local judge; you do not interpret the
  new source evidence yourself.
- Request follow-up only when it can materially change the final verdict. When
  requesting it, use a provisional uncertain verdict rather than assuming what
  the missing source will show.
"""
        followup_schema = """[
    {
      "condition_id": "existing condition id",
      "reason": "why source evidence can resolve this condition",
      "evidence_ids": ["existing packet evidence id not previously probed"],
      "cluster_id": "shared gap id, or empty only for a new id after near-miss",
      "cluster_members": ["active blocker condition ids sharing the gap"]
    }
  ]"""
    elif aggregation_round > 1:
        followup_policy = """
- This is the final aggregation pass after the one allowed follow-up round.
  You must decide from the updated judge results. Do not request more evidence;
  followup_requests must be empty.
"""
        followup_schema = "[]"
    else:
        followup_policy = """
- Source follow-up is unavailable for this run. Decide from the current judge
  results and keep followup_requests empty; do not assume missing source facts.
"""
        followup_schema = "[]"
    return f"""
You are the EvoTx dynamic aggregation agent. Local judges evaluated a structured
rule, but rules are imperfect and local judge failures may be soft, redundant,
or caused by missing evidence. Decide the final transaction-level verdict.

Use the rule as the default frame, not as an inflexible theorem.
You are responsible for deciding which failed local conditions are hard blockers
and which are soft blockers. Runtime-provided failure types, clusters, and typed
anchors are hints only; they are not verdict guards.

Decision policy:
- semantic_answers are the authoritative local Judge states. emit_logic_values
  are lossy booleans used only to evaluate the expression; an uncertain answer
  appears false there and must never be interpreted as semantic contradiction.
- Preserve the previous verdict unless the global evidence justifies changing it.
- Decide condition importance from the rule text, decision policy, label, and
  judge evidence. Do not infer importance from condition ids such as C1/E1 alone.
- Classify a failed core condition by why it failed, not only by answer=false.
  A medium/high-confidence true_contradiction backed by direct contradicting
  evidence is a hard blocker. A failure classified as missing_evidence or
  over_narrow_or_unresolved, without decisive counterevidence, is a soft blocker.
- A true exclusion is a hard blocker only when it has direct supporting evidence
  and medium/high confidence. A low-confidence or unsupported exclusion claim is
  a soft blocker that may be overridden with an explicit weak_exclusion basis.
- A false exclusion means only that the exclusion is unsupported. It is not
  positive evidence for any core condition and cannot resolve a false or
  uncertain core condition.
- Evidence assigned to one condition may supplement another only when the cited
  evidence ids directly address the second condition's wording. Record every such
  transfer in cross_condition_support; generic transaction shape, typed anchors,
  or the absence of an exclusion are not substitutes for local evidence.
- Follow-up is preferred when it can materially resolve a soft blocker. If you
  request follow-up, return a provisional uncertain verdict, never attack.
- On the final pass, if the requested Judge remains uncertain, its answer did
  not change, or source tools were unavailable, preserve uncertain. Failed
  follow-up is not new positive evidence and cannot justify an attack upgrade.
- After considering follow-up, you may emit attack with at most one independent
  soft-blocker cluster. Multiple soft conditions may count as one only when they
  share the same runtime cluster_id and the same unresolved evidence gap.
- When overriding a soft blocker, all other core conditions must be positively
  supported. List every ignored condition in overridden_conditions, select a
  concrete override_basis, explain the remaining risk, and cap confidence at
  medium.
- Do not emit "attack" if you identify a hard blocker. A hard blocker is any
  failed condition or exclusion that, under this specific rule and evidence,
  materially defeats the target label rather than merely exposing an imprecise
  local judge.
- If source/local follow-up is still needed to resolve an important blocker,
  output "uncertain" and explain which condition needs it.
- When multiple failed conditions share a runtime cluster hint, decide whether
  they are one unresolved evidence gap or independent semantic blockers.
{family_binding_policy.rstrip()}
{followup_policy.rstrip()}

Return only a JSON object:
{{
  "verdict": "attack|benign|uncertain",
  "confidence": "low|medium|high",
  "override": true,
  "reason": "short reason",
  "hard_blockers": [],
  "soft_blockers": [],
  "failed_condition_assessment": [
    {{
      "condition_id": "condition id",
      "importance": "essential|important|supporting",
      "failure_type": "true_contradiction|missing_evidence|over_narrow_or_unresolved|alternative_mechanism_claim|unclassified",
      "can_override": true,
      "reason": "short reason"
    }}
  ],
  "typed_positive_anchors_used": [],
  "overridden_conditions": ["condition id"],
  "override_basis": "none|missing_evidence|over_narrow_or_unresolved|same_cluster_evidence_gap|cross_condition_support|redundant_condition|weak_exclusion",
  "cross_condition_support": [
    {{
      "source_condition_id": "condition providing evidence",
      "target_condition_id": "soft condition being supplemented",
      "evidence_ids": ["existing evidence id"],
      "reason": "why the evidence directly addresses the target condition"
    }}
  ],
  "remaining_risk": "short residual-risk statement",
  "followup_requests": {followup_schema}
}}

Input:
{stable_json_dumps(payload)}
""".strip()


def _compact_judge_for_aggregator(jr: Dict[str, Any]) -> Dict[str, Any]:
    cfa = normalize_condition_feature_analysis(jr.get("condition_feature_analysis", {}))
    recent_rounds = []
    for raw_round in list(jr.get("judge_rounds", []) or [])[-3:]:
        if not isinstance(raw_round, dict):
            continue
        recent_rounds.append({
            "round": raw_round.get("round"),
            "answer": raw_round.get("answer"),
            "confidence": raw_round.get("confidence"),
            "reason": _truncate_text(str(raw_round.get("reason") or ""), 360),
            "supporting_evidence_ids": list(
                raw_round.get("supporting_evidence_ids", []) or []
            )[:8],
            "contradicting_evidence_ids": list(
                raw_round.get("contradicting_evidence_ids", []) or []
            )[:8],
            "parse_metadata": _compact_judge_parse_metadata(
                raw_round.get("parse_metadata", {})
            ),
        })
    attempted_source_ids: List[str] = []
    for call in list(jr.get("tool_calls", []) or []):
        if str(call.get("tool") or "") != "read_function_chunk":
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        attempted_source_ids.append(args.get("evidence_id"))
        attempted_source_ids.extend(list(args.get("evidence_ids", []) or []))
    return {
        "id": jr.get("id"),
        "condition_id": jr.get("condition_id") or jr.get("id"),
        "is_exclusion": _is_exclusion_judge(jr),
        "answer": jr.get("answer"),
        "confidence": jr.get("confidence", "medium"),
        "reason": _truncate_text(str(jr.get("reason") or ""), 700),
        "supporting_evidence_ids": list(jr.get("supporting_evidence_ids", []) or [])[:12],
        "contradicting_evidence_ids": list(jr.get("contradicting_evidence_ids", []) or [])[:12],
        "missing_evidence": list(jr.get("missing_evidence", []) or [])[:8],
        "condition_feature_analysis": cfa,
        "source_tool_call_count": sum(
            1
            for call in list(jr.get("tool_calls", []) or [])
            if str(call.get("tool") or "") == "read_function_chunk"
        ),
        "source_probe_attempted_evidence_ids": _unique_strings(
            attempted_source_ids
        )[:12],
        "near_miss_escalated": bool(jr.get("near_miss_escalated")),
        "recent_judge_rounds": recent_rounds,
        "judge_output_exhaustions": list(
            jr.get("judge_output_exhaustions", []) or []
        )[-3:],
    }


def _enforce_post_followup_upgrade_guard(
    decision: Dict[str, Any],
    *,
    previous_verdict: str,
    aggregation_round: int,
    followup_context: Dict[str, Any] | None,
) -> Dict[str, Any]:
    guarded = dict(decision or {})
    if (
        int(aggregation_round or 0) <= 1
        or previous_verdict == "attack"
        or str(guarded.get("verdict") or previous_verdict) != "attack"
    ):
        return guarded

    reruns = [
        item
        for item in list((followup_context or {}).get("judge_reruns", []) or [])
        if isinstance(item, dict) and not item.get("error")
    ]
    materially_resolved = any(
        bool(item.get("changed_answer"))
        and item.get("after_answer") in {True, False}
        for item in reruns
    )
    source_statuses = [
        str(status or "unknown")
        for item in reruns
        for status in list(item.get("source_tool_statuses", []) or [])
    ]
    source_failed = bool(source_statuses) and not any(
        status == "ok" for status in source_statuses
    )
    if materially_resolved and not source_failed:
        return guarded

    warnings = list(guarded.get("warnings", []) or [])
    warnings.append("attack_upgrade_rejected_followup_unresolved")
    if source_failed:
        warnings.append("attack_upgrade_rejected_source_followup_unavailable")
    guarded.update({
        "verdict": previous_verdict,
        "override": False,
        "reason": (
            "Follow-up did not resolve the verdict-critical local condition; "
            "preserving the deterministic verdict."
        ),
        "warnings": warnings,
        "post_followup_upgrade_guard": {
            "applied": True,
            "materially_resolved": materially_resolved,
            "source_failed": source_failed,
            "rerun_count": len(reruns),
        },
    })
    return guarded


def _sanitize_dynamic_aggregation_decision(
    raw: Dict[str, Any],
    *,
    previous_verdict: str,
    previous_reason: str,
    analysis: Dict[str, Any],
    allow_followup_requests: bool = True,
) -> Dict[str, Any]:
    verdict = str(raw.get("verdict") or previous_verdict).lower().strip()
    if verdict not in {"attack", "benign", "uncertain"}:
        verdict = previous_verdict
    confidence = str(raw.get("confidence") or "medium").lower().strip()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    warnings: List[str] = []
    followup_requests = _normalize_dynamic_aggregation_followup_request_shapes(
        raw.get("followup_requests", [])
    )
    if not allow_followup_requests and followup_requests:
        warnings.append("followup_requests_ignored_after_max_rounds")
        followup_requests = []
    if verdict == "attack" and followup_requests:
        warnings.append("attack_deferred_until_requested_followup_completes")
        verdict = "uncertain"
    agent_hard_blockers = list(raw.get("hard_blockers", []) or [])
    overridden_conditions = _normalize_dynamic_condition_ids(
        raw.get("overridden_conditions", [])
    )
    override_basis = str(raw.get("override_basis") or "none").lower().strip()
    allowed_override_bases = {
        "none",
        "missing_evidence",
        "over_narrow_or_unresolved",
        "same_cluster_evidence_gap",
        "cross_condition_support",
        "redundant_condition",
        "weak_exclusion",
    }
    if override_basis not in allowed_override_bases:
        override_basis = "none"
    cross_condition_support = _normalize_cross_condition_support(
        raw.get("cross_condition_support", [])
    )
    if verdict == "attack" and not _dynamic_attack_override_allowed(
        analysis,
        agent_decision={
            **raw,
            "overridden_conditions": overridden_conditions,
            "override_basis": override_basis,
            "cross_condition_support": cross_condition_support,
        },
    ):
        warnings.append("attack_override_rejected_by_runtime_hard_blockers")
        warnings.append("attack_override_rejected_by_runtime_policy")
        verdict = previous_verdict if previous_verdict != "attack" else "attack"
    _, soft_blockers = _partition_dynamic_blockers(
        list(analysis.get("condition_assessments", []) or [])
    )
    if verdict == "attack" and soft_blockers and confidence == "high":
        confidence = "medium"
        warnings.append("attack_confidence_capped_for_soft_blocker_override")
    return {
        "verdict": verdict,
        "confidence": confidence,
        "override": verdict != previous_verdict,
        "reason": str(raw.get("reason") or previous_reason or "dynamic_aggregation"),
        "mode": "agent",
        "hard_blockers": agent_hard_blockers,
        "soft_blockers": list(raw.get("soft_blockers", []) or []),
        "failed_condition_assessment": list(
            raw.get("failed_condition_assessment", []) or []
        ),
        "typed_positive_anchors_used": list(
            raw.get("typed_positive_anchors_used", []) or []
        ),
        "overridden_conditions": overridden_conditions,
        "override_basis": override_basis,
        "cross_condition_support": cross_condition_support,
        "remaining_risk": _truncate_text(str(raw.get("remaining_risk") or ""), 700),
        "followup_requests": followup_requests,
        "warnings": warnings,
    }


def _normalize_dynamic_aggregation_followup_request_shapes(
    raw_requests: Any,
) -> List[Dict[str, Any]]:
    if isinstance(raw_requests, dict):
        items = [raw_requests]
    elif isinstance(raw_requests, list):
        items = raw_requests
    else:
        return []
    normalized: List[Dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        condition_id = str(
            raw.get("condition_id") or raw.get("judge_id") or ""
        ).strip()
        raw_evidence_ids = (
            raw.get("evidence_ids")
            or raw.get("source_probe_evidence_ids")
            or []
        )
        if not isinstance(raw_evidence_ids, (list, tuple, set)):
            raw_evidence_ids = [raw_evidence_ids]
        raw_cluster_members = raw.get("cluster_members", []) or []
        if not isinstance(raw_cluster_members, (list, tuple, set)):
            raw_cluster_members = [raw_cluster_members]
        evidence_ids = [
            value
            for value in _unique_strings(
                list(raw_evidence_ids)
            )
            if _looks_like_evidence_id(value)
        ][:4]
        normalized.append({
            "condition_id": condition_id,
            "reason": _truncate_text(str(raw.get("reason") or ""), 600),
            "evidence_ids": evidence_ids,
            "cluster_id": str(raw.get("cluster_id") or ""),
            "cluster_members": _unique_strings(
                list(raw_cluster_members)
            ),
        })
    return normalized


def _normalize_cross_condition_support(raw_support: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_support, dict):
        items = [raw_support]
    elif isinstance(raw_support, list):
        items = raw_support
    else:
        return []
    normalized: List[Dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        source_condition_id = str(raw.get("source_condition_id") or "").strip()
        target_condition_id = str(raw.get("target_condition_id") or "").strip()
        raw_evidence_ids = raw.get("evidence_ids", []) or []
        if not isinstance(raw_evidence_ids, (list, tuple, set)):
            raw_evidence_ids = [raw_evidence_ids]
        evidence_ids = [
            value
            for value in _unique_strings(list(raw_evidence_ids))
            if _looks_like_evidence_id(value)
        ][:8]
        if not target_condition_id or not evidence_ids:
            continue
        normalized.append({
            "source_condition_id": source_condition_id,
            "target_condition_id": target_condition_id,
            "evidence_ids": evidence_ids,
            "reason": _truncate_text(str(raw.get("reason") or ""), 500),
        })
    return normalized


def _normalize_dynamic_condition_ids(raw_ids: Any) -> List[str]:
    if isinstance(raw_ids, str):
        values = [raw_ids]
    elif isinstance(raw_ids, (list, tuple, set)):
        values = list(raw_ids)
    else:
        values = []
    return _unique_strings(values)


def _dynamic_assessment_is_blocker(assessment: Dict[str, Any]) -> bool:
    answer = assessment.get("answer")
    is_exclusion = bool(assessment.get("is_exclusion"))
    return (is_exclusion and answer is True) or (
        not is_exclusion and (answer is False or _answer_is_uncertain(answer))
    )


def _dynamic_assessment_is_hard_blocker(assessment: Dict[str, Any]) -> bool:
    if not _dynamic_assessment_is_blocker(assessment):
        return False
    answer = assessment.get("answer")
    is_exclusion = bool(assessment.get("is_exclusion"))
    confidence = str(assessment.get("confidence") or "").lower().strip()
    failure_type = str(assessment.get("failure_type_hint") or "").lower().strip()
    supporting_count_raw = assessment.get("supporting_evidence_count")
    contradicting_count = int(assessment.get("contradicting_evidence_count", 0) or 0)

    if is_exclusion and answer is True:
        if confidence == "low" and int(supporting_count_raw or 0) == 0:
            return False
        if supporting_count_raw is None and not confidence and not failure_type:
            return True
        return confidence in {"medium", "high"} and int(supporting_count_raw or 0) > 0

    if _answer_is_uncertain(answer):
        return False
    if failure_type in {"missing_evidence", "over_narrow_or_unresolved"}:
        return False
    if failure_type == "true_contradiction":
        return confidence != "low" or contradicting_count > 0
    if failure_type == "alternative_mechanism_claim":
        return False
    # Unknown false answers stay hard unless Runtime has an explicit soft-failure signal.
    return True


def _partition_dynamic_blockers(
    assessments: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    hard: List[Dict[str, Any]] = []
    soft: List[Dict[str, Any]] = []
    for assessment in list(assessments or []):
        if not _dynamic_assessment_is_blocker(assessment):
            continue
        target = hard if _dynamic_assessment_is_hard_blocker(assessment) else soft
        target.append(dict(assessment))
    return hard, soft


def _dynamic_soft_blocker_cluster_ids(
    soft_blockers: List[Dict[str, Any]],
) -> List[str]:
    cluster_ids: List[str] = []
    for assessment in soft_blockers:
        condition_id = str(
            assessment.get("condition_id") or assessment.get("judge_id") or ""
        ).strip()
        cluster_id = str(assessment.get("cluster_id") or "").strip()
        cluster_ids.append(cluster_id or f"condition:{condition_id}")
    return sorted(set(cluster_ids))


def _dynamic_attack_override_allowed(
    analysis: Dict[str, Any],
    *,
    agent_decision: Dict[str, Any] | None = None,
) -> bool:
    if agent_decision and list(agent_decision.get("hard_blockers", []) or []):
        return False
    hard_blockers, soft_blockers = _partition_dynamic_blockers(
        list(analysis.get("condition_assessments", []) or [])
    )
    if hard_blockers:
        return False
    if not soft_blockers:
        return True
    if len(_dynamic_soft_blocker_cluster_ids(soft_blockers)) > 1:
        return False
    positive_core_true_count = int(
        analysis.get("positive_core_true_count", 0)
        or len(list(analysis.get("true_core_ids", []) or []))
    )
    if positive_core_true_count < 1:
        return False
    if not agent_decision:
        return False
    soft_condition_ids = {
        str(item.get("condition_id") or item.get("judge_id") or "").strip()
        for item in soft_blockers
    }
    soft_condition_ids.discard("")
    overridden_conditions = set(_normalize_dynamic_condition_ids(
        agent_decision.get("overridden_conditions", [])
    ))
    if not soft_condition_ids.issubset(overridden_conditions):
        return False
    override_basis = str(agent_decision.get("override_basis") or "none").lower().strip()
    if override_basis in {"", "none"}:
        return False
    soft_failure_types = {
        str(item.get("failure_type_hint") or "").lower().strip()
        for item in soft_blockers
    }
    if override_basis == "missing_evidence" and not soft_failure_types.issubset({
        "missing_evidence",
    }):
        return False
    if (
        override_basis == "over_narrow_or_unresolved"
        and not soft_failure_types.issubset({"over_narrow_or_unresolved"})
    ):
        return False
    if override_basis == "weak_exclusion" and not all(
        bool(item.get("is_exclusion")) for item in soft_blockers
    ):
        return False
    if override_basis == "same_cluster_evidence_gap" and len(
        _dynamic_soft_blocker_cluster_ids(soft_blockers)
    ) != 1:
        return False
    if override_basis == "redundant_condition":
        importance_by_condition = {
            str(item.get("condition_id") or "").strip(): str(
                item.get("importance") or ""
            ).lower().strip()
            for item in list(agent_decision.get("failed_condition_assessment", []) or [])
            if isinstance(item, dict)
        }
        if any(
            importance_by_condition.get(condition_id) != "supporting"
            for condition_id in soft_condition_ids
        ):
            return False
    if override_basis == "cross_condition_support":
        supported_targets = {
            str(item.get("target_condition_id") or "").strip()
            for item in list(agent_decision.get("cross_condition_support", []) or [])
            if isinstance(item, dict) and list(item.get("evidence_ids", []) or [])
        }
        if not soft_condition_ids.issubset(supported_targets):
            return False
    return True


def _record_dynamic_aggregation_transcript(
    *,
    judge_model: JudgeModel,
    llm: Any | None = None,
    prompt: str,
    raw_completion: str,
    parsed_response: Dict[str, Any],
    tx_hash: str,
    chain: str,
    analysis: Dict[str, Any],
    error: str,
    parse_metadata: Dict[str, Any] | None = None,
    phase: str,
    aggregation_round: int = 1,
    followup_context: Dict[str, Any] | None = None,
) -> None:
    recorder = getattr(judge_model, "transcript_recorder", None)
    if recorder is None or not prompt:
        return
    try:
        active_llm = llm or getattr(judge_model, "llm", None)
        recorder.save_judge_transcript(
            {
                "kind": "dynamic_aggregator",
                "phase": phase,
                "round": f"aggregation_{aggregation_round}",
                "tx_hash": tx_hash,
                "chain": chain,
                "judge_id": "AGGREGATOR",
                "condition_id": "AGGREGATOR",
                "prompt": prompt,
                "raw_completion": raw_completion,
                "parsed_response": parsed_response,
                "analysis": analysis,
                "parse_metadata": dict(parse_metadata or {}),
                "followup_context": dict(followup_context or {}),
                "error": error,
                "usage": dict(getattr(active_llm, "last_usage", {}) or {}),
                "model": str(getattr(active_llm, "model", "") or ""),
                "provider": str(getattr(active_llm, "provider", "") or ""),
                "thinking": str(
                    getattr(active_llm, "minimax_thinking", "") or ""
                ),
            },
            phase=phase,
            tx_hash=tx_hash,
            judge_id="AGGREGATOR",
            round_id=f"aggregation_{aggregation_round}",
        )
    except Exception as exc:
        print(f"[PacketRuntime] dynamic aggregation transcript failed: {exc!r}")


def _try_extract_dynamic_aggregator_json_object(
    text: str,
) -> tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    return extract_last_schema_valid_object(
        text,
        _dynamic_aggregator_schema_error,
        schema_name="Aggregator",
    )


def _dynamic_aggregator_schema_error(data: Dict[str, Any]) -> Optional[str]:
    if not isinstance(data, dict):
        return "top-level value is not an object"
    missing = [key for key in ("verdict", "confidence", "reason") if key not in data]
    if missing:
        return f"missing required fields: {', '.join(missing)}"
    if str(data.get("verdict") or "").lower().strip() not in {
        "attack",
        "benign",
        "uncertain",
    }:
        return "verdict must be attack, benign, or uncertain"
    if str(data.get("confidence") or "").lower().strip() not in {
        "low",
        "medium",
        "high",
    }:
        return "confidence must be low, medium, or high"
    if not isinstance(data.get("reason"), str):
        return "reason must be a string"
    for key in (
        "hard_blockers",
        "soft_blockers",
        "failed_condition_assessment",
        "typed_positive_anchors_used",
        "overridden_conditions",
        "cross_condition_support",
        "followup_requests",
    ):
        if key in data and not isinstance(data.get(key), list):
            return f"{key} must be an array"
    return None


def _truncate_text(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _aggregate_verdict(
    *,
    plan: EvidencePlan,
    judge_results: List[Dict[str, Any]],
    judge_values: Dict[str, bool],
    should_emit: bool,
    has_uncertain: bool,
    emit_logic_error: str | None,
) -> tuple[str, Dict[str, Any]]:
    """Combine local judge answers into the transaction verdict.

    The important policy point is strong-negative-dominates-uncertain: a single
    failed core condition may be noisy, but multiple negative signals should
    reject the attack chain even if another condition remains uncertain.
    """
    try:
        logic_names = sorted(extract_logic_names(plan.emit_logic))
    except Exception:
        logic_names = sorted(judge_values)
    relevant_ids = set(logic_names or judge_values.keys())
    candidate_emit = _stateful_candidate_emit_decision(
        plan=plan,
        judge_results=judge_results,
    )
    candidate_emit_enforced = bool(
        candidate_emit.get("enabled")
        and candidate_emit.get("enforce_emit", True)
    )
    candidate_local_exclusions = candidate_emit_enforced

    decisive_core_false: List[Dict[str, Any]] = []
    decisive_exclusion_true: List[Dict[str, Any]] = []
    core_blocking_uncertain: List[Dict[str, Any]] = []
    exclusion_uncertain: List[Dict[str, Any]] = []
    low_confidence_false: List[Dict[str, Any]] = []

    for jr in judge_results:
        judge_id = str(jr.get("id") or "")
        if relevant_ids and judge_id not in relevant_ids:
            continue
        answer = jr.get("answer")
        confidence = str(jr.get("confidence") or "medium").lower().strip()
        is_exclusion = _is_exclusion_judge(jr)
        summary = _judge_summary_for_aggregation(jr)

        if _answer_is_uncertain(answer):
            if is_exclusion:
                exclusion_uncertain.append(summary)
            else:
                core_blocking_uncertain.append(summary)
            continue
        if not is_exclusion and answer is False:
            if _is_decisive_confidence(confidence):
                decisive_core_false.append(summary)
            else:
                low_confidence_false.append(summary)
        elif is_exclusion and answer is True:
            if candidate_local_exclusions:
                continue
            if _is_decisive_confidence(confidence):
                decisive_exclusion_true.append(summary)

    aggregation = {
        "policy": "strong_negative_dominates_uncertain",
        "emit_logic": plan.emit_logic,
        "emit_logic_variables": logic_names,
        "judge_values": dict(judge_values),
        "should_emit": bool(should_emit),
        "has_uncertain": bool(has_uncertain),
        "emit_logic_error": emit_logic_error,
        "decisive_core_false": decisive_core_false,
        "decisive_exclusion_true": decisive_exclusion_true,
        "strong_negative_policy": {
            "min_core_false": 2,
            "min_exclusion_true": 2,
            "core_false_plus_exclusion_true": True,
        },
        "core_blocking_uncertain": core_blocking_uncertain,
        "exclusion_uncertain": exclusion_uncertain,
        "exclusion_uncertain_policy": "warning_only",
        "blocking_uncertain": core_blocking_uncertain,
        "low_confidence_false": low_confidence_false,
        "candidate_emit": candidate_emit,
    }

    if emit_logic_error:
        aggregation["reason"] = "emit_logic_error"
        aggregation["decision_source"] = "emit_logic_error"
        return "uncertain", aggregation

    if candidate_emit_enforced:
        remaining = list(
            candidate_emit.get("remaining_attack_candidate_ids") or []
        )
        unresolved = list(candidate_emit.get("unresolved_candidate_ids") or [])
        if remaining:
            aggregation["reason"] = "same_candidate_chain_satisfied"
            aggregation["decision_source"] = "stateful_candidate_emit"
            return "attack", aggregation
        if unresolved:
            aggregation["reason"] = "candidate_chain_unresolved"
            aggregation["decision_source"] = "stateful_candidate_emit"
            return "uncertain", aggregation
        aggregation["reason"] = "no_unexcluded_candidate_chain_satisfied"
        aggregation["decision_source"] = "stateful_candidate_emit"
        return "benign", aggregation

    has_strong_negative = (
        len(decisive_core_false) >= 2
        or len(decisive_exclusion_true) >= 2
        or (len(decisive_core_false) >= 1 and len(decisive_exclusion_true) >= 1)
    )
    if has_strong_negative:
        aggregation["reason"] = "strong_negative_evidence"
        aggregation["decision_source"] = "strong_negative_policy"
        return "benign", aggregation

    if should_emit and not core_blocking_uncertain and not decisive_exclusion_true:
        aggregation["reason"] = "emit_logic_true_without_core_blocking_uncertain"
        aggregation["decision_source"] = "plan_emit_logic"
        return "attack", aggregation

    if core_blocking_uncertain:
        aggregation["reason"] = "core_blocking_uncertain_without_decisive_negative"
        aggregation["decision_source"] = "uncertainty_policy"
        return "uncertain", aggregation

    aggregation["reason"] = "emit_logic_false_without_core_blocking_uncertain"
    aggregation["decision_source"] = "plan_emit_logic"
    return "benign", aggregation


def _judge_summary_for_aggregation(judge_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": judge_result.get("id"),
        "condition_id": judge_result.get("condition_id") or judge_result.get("id"),
        "answer": judge_result.get("answer"),
        "confidence": judge_result.get("confidence", "medium"),
        "is_exclusion": _is_exclusion_judge(judge_result),
        "reason": judge_result.get("reason", ""),
        "supporting_evidence_ids": list(judge_result.get("supporting_evidence_ids", [])),
        "missing_evidence": list(judge_result.get("missing_evidence", [])),
        "suggested_followup_views": list(
            judge_result.get("suggested_followup_views", []) or []
        ),
        "tool_requests": [
            copy.deepcopy(item)
            for item in list(judge_result.get("tool_requests", []) or [])
            if isinstance(item, dict)
        ],
    }


def _near_miss_request_signature(tool: str, args: Dict[str, Any]) -> str:
    return f"{str(tool or '').strip()}:{stable_json_dumps(dict(args or {}))}"


def _near_miss_observation_request_signatures(
    observations: List[Dict[str, Any]],
) -> set[str]:
    signatures: set[str] = set()
    for observation in list(observations or []):
        request = dict((observation or {}).get("request") or {})
        tool = str(request.get("tool") or request.get("name") or "").strip()
        if not tool:
            continue
        signatures.add(
            _near_miss_request_signature(tool, dict(request.get("args") or {}))
        )
    return signatures


def _near_miss_observation_has_target_evidence(
    observation: Dict[str, Any],
) -> bool:
    if not bool(observation.get("target_condition_specific")):
        return False
    result = dict(observation.get("result") or {})
    if str(result.get("tool_status") or "").lower().strip() != "ok":
        return False
    if list(result.get("returned_evidence_ids", []) or []):
        return True
    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        for key in ("rows", "snippets", "contexts", "calls", "events", "states"):
            if bool(evidence.get(key)):
                return True
    return False


def _enforce_near_miss_upgrade_guard(
    *,
    original_result: Dict[str, Any],
    escalated_result: Dict[str, Any],
    escalated_trace: Dict[str, Any],
) -> Dict[str, Any]:
    guarded = copy.deepcopy(escalated_result or {})
    before_answer = original_result.get("answer")
    after_answer = guarded.get("answer")
    escalation_meta = dict(
        escalated_trace.get("near_miss_escalation", {}) or {}
    )
    prior_output_exhausted = bool(
        list(original_result.get("judge_output_exhaustions", []) or [])
    )
    escalation_output_exhausted = bool(
        list(escalated_trace.get("judge_output_exhaustions", []) or [])
    )
    output_exhausted = prior_output_exhausted or escalation_output_exhausted
    target_evidence_count = int(
        escalation_meta.get("target_condition_evidence_count", 0) or 0
    )
    rejection_reasons: List[str] = []
    if after_answer is True and before_answer is not True and output_exhausted:
        rejection_reasons.append("output_exhausted_cannot_upgrade_true")
    if (
        after_answer is True
        and _answer_is_uncertain(before_answer)
        and target_evidence_count <= 0
    ):
        rejection_reasons.append(
            "uncertain_upgrade_requires_target_condition_evidence"
        )

    applied = bool(rejection_reasons)
    if applied:
        judgment_fields = (
            "answer",
            "confidence",
            "reason",
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "missing_evidence",
            "suggested_followup_views",
            "tool_requests",
            "condition_feature_analysis",
            "state_output",
            "satisfied",
        )
        for field in judgment_fields:
            if field in original_result:
                guarded[field] = copy.deepcopy(original_result.get(field))
    guarded["near_miss_upgrade_guard"] = {
        "applied": applied,
        "before_answer": before_answer,
        "candidate_answer": after_answer,
        "final_answer": guarded.get("answer"),
        "output_exhausted": output_exhausted,
        "prior_output_exhausted": prior_output_exhausted,
        "escalation_output_exhausted": escalation_output_exhausted,
        "target_condition_evidence_count": target_evidence_count,
        "rejection_reasons": rejection_reasons,
    }
    return guarded


def _answer_is_uncertain(answer: Any) -> bool:
    return isinstance(answer, str) and answer.lower().strip() == "uncertain"


def _unpinned_source_negative_guard(
    *,
    previous_result: JudgeResult,
    candidate_result: JudgeResult,
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Enforce the existing source/trace contract without another LLM pass."""
    unpinned_source_observed = False
    runtime_evidence_observed = _judge_result_has_transaction_evidence(
        candidate_result
    )
    new_runtime_observation_observed = False
    for raw_observation in list(observations or []):
        if not isinstance(raw_observation, dict):
            continue
        request = dict(raw_observation.get("request") or {})
        tool = str(
            request.get("tool")
            or request.get("name")
            or raw_observation.get("tool")
            or ""
        ).strip()
        result = dict(raw_observation.get("result") or {})
        summary = dict(result.get("summary") or {})
        if tool == "read_function_chunk":
            if (
                summary.get("decisive_negative_eligible") is False
                or str(summary.get("source_provenance_status") or "").strip()
                == "explorer_verified_unpinned"
            ):
                unpinned_source_observed = True
            continue
        if _tool_result_has_material_evidence(result):
            runtime_evidence_observed = True
            new_runtime_observation_observed = True

    previous_transaction_evidence = _judge_result_transaction_evidence_ids(
        previous_result
    )
    candidate_transaction_evidence = _judge_result_transaction_evidence_ids(
        candidate_result
    )
    new_transaction_evidence = sorted(
        candidate_transaction_evidence - previous_transaction_evidence
    )
    contradictory_transaction_evidence = sorted(
        _transaction_evidence_ids(candidate_result.contradicting_evidence_ids)
    )
    applied = bool(
        _answer_is_uncertain(previous_result.answer)
        and candidate_result.answer is False
        and unpinned_source_observed
        and (
            bool(contradictory_transaction_evidence)
            or not (
                new_transaction_evidence
                or new_runtime_observation_observed
            )
        )
    )
    return {
        "applied": applied,
        "before_answer": previous_result.answer,
        "candidate_answer": candidate_result.answer,
        "final_answer": previous_result.answer if applied else candidate_result.answer,
        "reason": (
            "unpinned_source_not_decisive_negative"
            if applied
            else "source_decisiveness_contract_not_triggered"
        ),
        "unpinned_source_observed": unpinned_source_observed,
        "runtime_evidence_observed": runtime_evidence_observed,
        "new_runtime_observation_observed": new_runtime_observation_observed,
        "new_transaction_evidence_ids": new_transaction_evidence,
        "contradictory_transaction_evidence_ids": (
            contradictory_transaction_evidence
        ),
    }


def _judge_result_has_transaction_evidence(result: JudgeResult) -> bool:
    return bool(_judge_result_transaction_evidence_ids(result))


def _judge_result_transaction_evidence_ids(result: JudgeResult) -> set[str]:
    return _transaction_evidence_ids([
        *list(result.supporting_evidence_ids or []),
        *list(result.contradicting_evidence_ids or []),
    ])


def _transaction_evidence_ids(evidence_ids: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for value in evidence_ids:
        evidence_id = str(value or "").strip().lower()
        if not evidence_id:
            continue
        if evidence_id.startswith(("contract_source:", "source:")):
            continue
        out.add(evidence_id)
    return out


def _tool_result_has_material_evidence(result: Dict[str, Any]) -> bool:
    if str((result or {}).get("tool_status") or "").strip().lower() != "ok":
        return False
    if list((result or {}).get("returned_evidence_ids") or []):
        return True
    evidence = (result or {}).get("evidence")
    if not isinstance(evidence, dict):
        return False
    return any(
        bool(evidence.get(key))
        for key in ("rows", "contexts", "calls", "events", "states")
    )


def _is_decisive_confidence(confidence: str) -> bool:
    return confidence.lower().strip() in {"medium", "high"}


def _group_judge_evidence(judge_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    attack_support: Dict[str, List[str]] = {}
    benign_support: Dict[str, List[str]] = {}
    failed_core: Dict[str, Dict[str, Any]] = {}
    missing_by_condition: Dict[str, List[str]] = {}
    union: List[str] = []
    seen: set = set()

    for jr in judge_results:
        judge_id = str(jr.get("id") or "")
        group_id = str(jr.get("condition_id") or judge_id)
        answer_is_true = jr.get("answer") is True
        evidence_ids = list(jr.get("supporting_evidence_ids", []))
        missing = list(jr.get("missing_evidence", []))
        if missing:
            missing_by_condition[group_id] = missing

        if answer_is_true:
            if _is_exclusion_judge(jr):
                benign_support[group_id] = evidence_ids
            else:
                attack_support[group_id] = evidence_ids
            for eid in evidence_ids:
                if eid not in seen:
                    seen.add(eid)
                    union.append(eid)
        elif not _is_exclusion_judge(jr):
            failed_core[group_id] = {
                "answer": jr.get("answer"),
                "reason": jr.get("reason", ""),
                "evidence_ids": evidence_ids,
            }

    return {
        "attack_supporting_evidence_by_condition": attack_support,
        "benign_exclusion_evidence_by_condition": benign_support,
        "failed_core_conditions": failed_core,
        "missing_evidence_by_condition": missing_by_condition,
        "supporting_evidence_union": union,
    }
