#!/usr/bin/env python3
"""One-shot raw-trace LLM baseline for AC vs insufficient validation.

This utility intentionally bypasses packet views and EvoTx runtime logic. It
reads one trace JSON file, appends it to a fixed question, calls the existing
OpenAICompatibleLLM adapter once, and prints the raw model answer.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evotx.adapters.llm_adapter import OpenAICompatibleLLM
from evotx.utils.json_utils import write_json


QUESTION = (
    "根据trace 判断这笔交易属于access control还是insufficient validation, "
    "简要说明理由"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send one trace.json directly to an LLM and ask whether it is "
            "access control or insufficient validation."
        )
    )
    parser.add_argument("trace_json", help="Path to the trace.json file.")
    parser.add_argument("--llm-model", default="glm-5.1")
    parser.add_argument(
        "--llm-provider",
        help="Optional provider override: glm, minimax, volcano/ark, xunfei/astron, or openai.",
    )
    parser.add_argument("--api-key", help="Optional API key override.")
    parser.add_argument(
        "-ZhipuApiKey",
        "--zhipu-api-key",
        dest="zhipu_api_key",
        help="Optional Zhipu/GLM API key override for this one call.",
    )
    parser.add_argument("--base-url", help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-trace-chars",
        type=int,
        default=0,
        help=(
            "Optional trace text cap. Default 0 sends the whole file. "
            "If set, the trace text is truncated from the end."
        ),
    )
    parser.add_argument(
        "--out",
        help="Optional JSON path to save prompt, raw response, usage, and timing.",
    )
    parser.add_argument(
        "--skip-api-probe",
        action="store_true",
        help="Skip the small API reachability probe before sending the trace.",
    )
    args = parser.parse_args()

    print("[TraceLLM] Starting raw trace classification.", flush=True)
    print(f"[TraceLLM] Project root: {PROJECT_ROOT}", flush=True)
    print(f"[TraceLLM] Trace file: {args.trace_json}", flush=True)
    print(
        "[TraceLLM] Requested model/provider: "
        f"model={args.llm_model} provider={args.llm_provider or '(auto)'}",
        flush=True,
    )
    print(f"[TraceLLM] API key source: {_api_key_source(args)}", flush=True)
    print(
        f"[TraceLLM] max_trace_chars={args.max_trace_chars or 'unlimited'} "
        f"out={args.out or '(not saving)'}",
        flush=True,
    )

    trace_path = Path(args.trace_json)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    if not trace_path.is_file():
        raise ValueError(f"Trace path is not a file: {trace_path}")

    print("[TraceLLM] Reading trace file...", flush=True)
    trace_text = trace_path.read_text(encoding="utf-8-sig")
    original_chars = len(trace_text)
    truncated = False
    if args.max_trace_chars and args.max_trace_chars > 0:
        trace_text, truncated = _truncate_from_end(trace_text, args.max_trace_chars)

    prompt = build_prompt(trace_text, trace_path=trace_path, truncated=truncated)
    print(
        "[TraceLLM] Trace loaded: "
        f"original_chars={original_chars} sent_trace_chars={len(trace_text)} "
        f"prompt_chars={len(prompt)} truncated={truncated}",
        flush=True,
    )
    print("[TraceLLM] Initializing LLM adapter...", flush=True)
    llm = OpenAICompatibleLLM(
        model=args.llm_model,
        provider=args.llm_provider,
        api_key=args.zhipu_api_key or args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        minimax_thinking="disabled",
    )
    print(
        f"[TraceLLM] LLM adapter ready: provider={llm.provider} "
        f"model={llm.model}",
        flush=True,
    )

    probe_payload: Dict[str, Any] | None = None
    if not args.skip_api_probe:
        probe_payload = _probe_api(llm)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started_perf = time.perf_counter()
    print(
        "[TraceLLM] Sending main trace request. This may take a while for large traces...",
        flush=True,
    )
    response = llm.complete(prompt)
    elapsed = round(time.perf_counter() - started_perf, 3)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print("[TraceLLM] Main trace request completed.", flush=True)

    print(response)
    print()
    print(
        "[TraceLLM] "
        f"model={args.llm_model} provider={llm.provider} "
        f"trace_chars={original_chars} sent_chars={len(trace_text)} "
        f"truncated={truncated} elapsed={elapsed}s"
    )

    if args.out:
        payload: Dict[str, Any] = {
            "trace_path": str(trace_path),
            "question": QUESTION,
            "model": args.llm_model,
            "provider": llm.provider,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed,
            "trace_chars_original": original_chars,
            "trace_chars_sent": len(trace_text),
            "trace_truncated": truncated,
            "api_probe": probe_payload,
            "usage": dict(getattr(llm, "last_usage", {}) or {}),
            "prompt": prompt,
            "raw_completion": response,
        }
        write_json(args.out, payload)
        print(f"[TraceLLM] Saved conversation: {args.out}")


def build_prompt(trace_text: str, *, trace_path: Path, truncated: bool) -> str:
    truncation_note = ""
    if truncated:
        truncation_note = (
            "\n注意：下面的 trace 内容因为 --max-trace-chars 被截断，只包含前部内容。"
        )
    return f"""{QUESTION}

只在以下两个类别中二选一：
1. access control
2. insufficient validation

请输出：
- category: access control 或 insufficient validation
- reason: 简要说明理由

trace_file: {trace_path}
{truncation_note}

trace_json:
{trace_text}
"""


def _truncate_from_end(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    marker = (
        "\n\n[TRUNCATED: trace content exceeded --max-trace-chars; "
        f"{len(text) - max_chars} chars removed from the end]\n"
    )
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker, True


def _api_key_source(args: argparse.Namespace) -> str:
    if args.zhipu_api_key:
        return "-ZhipuApiKey/--zhipu-api-key"
    if args.api_key:
        return "--api-key"
    return ".env/provider environment"


def _probe_api(llm: OpenAICompatibleLLM) -> Dict[str, Any]:
    print("[TraceLLM] Probing API reachability with a tiny request...", flush=True)
    started = time.perf_counter()
    try:
        response = llm.complete(
            "Return exactly this JSON object and nothing else: {\"status\":\"ok\"}"
        )
        elapsed = round(time.perf_counter() - started, 3)
        usage = dict(getattr(llm, "last_usage", {}) or {})
        print(
            "[TraceLLM] API probe succeeded: "
            f"elapsed={elapsed}s usage={usage} "
            f"response_preview={response[:120]!r}",
            flush=True,
        )
        return {
            "status": "ok",
            "elapsed_seconds": elapsed,
            "usage": usage,
            "response_preview": response[:500],
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 3)
        print(
            "[TraceLLM] API probe failed: "
            f"{type(exc).__name__}: {exc} elapsed={elapsed}s",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
