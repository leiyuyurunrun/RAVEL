"""Judge one attack type directly from raw trace JSON with an LLM.

This script intentionally bypasses the EvoTx packet/runtime pipeline. It reads
raw trace JSON, appends one attack description, asks one target-attack question
per run, and writes structured per-transaction judgments.
"""

from __future__ import annotations

# fmt: off
from pathlib import Path; import sys; PROJECT_ROOT = Path(__file__).resolve().parents[1]  # noqa: E402
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402
# fmt: on

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from evotx.adapters.llm_adapter import OpenAICompatibleLLM
from evotx.utils.json_utils import (
    extract_json_object,
    extract_last_schema_valid_object,
    stable_json_dumps,
    structured_output_text,
    write_json,
)


ANSWER_TARGET = "target_attack"
ANSWER_NON_TARGET = "non_target"
ANSWER_UNCERTAIN = "uncertain"
VALID_ANSWERS = {ANSWER_TARGET, ANSWER_NON_TARGET, ANSWER_UNCERTAIN}
VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_STAGE_STATUS = {"pass", "fail", "uncertain"}
TX_HASH_COLUMNS = (
    "txHash",
    "TxHash",
    "tx_hash",
    "hash",
    "Hash",
    "transaction_hash",
    "TransactionHash",
)


def build_judge_prompt(
    *,
    trace_text: str,
    trace_path: Path,
    label: str,
    attack_description: str,
    truncated: bool,
    trace_chars_original: int,
    trace_chars_sent: int,
) -> str:
    truncation_note = "no"
    if truncated:
        truncation_note = (
            "yes; the trace text was truncated by --max-trace-chars. "
            "If decisive evidence is missing, answer uncertain."
        )
    schema = {
        "answer": f"{ANSWER_TARGET}|{ANSWER_NON_TARGET}|{ANSWER_UNCERTAIN}",
        "attack_label": label,
        "confidence": "low|medium|high",
        "reason": "short final rationale, no hidden reasoning",
        "judgment_stages": [
            {
                "name": "definition_alignment|trace_mechanism|alternative_exclusions|final_decision",
                "status": "pass|fail|uncertain",
                "finding": "stage-specific observation",
                "evidence_ids": ["call:12", "event:3", "state:slot:..."],
            }
        ],
        "evidence": [
            {
                "id": "trace-local id, call id, event id, or json path",
                "supports": f"{ANSWER_TARGET}|{ANSWER_NON_TARGET}|{ANSWER_UNCERTAIN}",
                "detail": "concise evidence description",
            }
        ],
    }
    return f"""You are a blockchain transaction security judge.

Task:
Decide whether this single transaction is the target attack type:
{label}

Attack description:
{attack_description}

Decision labels:
- {ANSWER_TARGET}: the trace contains transaction-visible evidence matching this target attack type.
- {ANSWER_NON_TARGET}: the trace is benign or better explained by a different attack type.
- {ANSWER_UNCERTAIN}: the provided trace is insufficient, truncated, or ambiguous.

Required judgment stages:
1. definition_alignment: compare the attack definition with what must be proven.
2. trace_mechanism: identify the concrete trace mechanism, calls, events, state changes, value movement, or accounting effects.
3. alternative_exclusions: check whether another mechanism better explains the trace.
4. final_decision: combine the stages into one answer.

Output rules:
- Return strict JSON only. No markdown, no prose outside JSON.
- Do not output <think> or hidden reasoning. Put only concise findings in JSON.
- Evidence must reference concrete trace ids, call ids, event ids, state slots, function names/selectors, addresses, or JSON paths when available.
- If the trace is too large or incomplete and key evidence is missing, answer uncertain rather than guessing.

JSON schema:
{stable_json_dumps(schema)}

Trace file: {trace_path}
Trace truncated: {truncation_note}
Trace chars original: {trace_chars_original}
Trace chars sent: {trace_chars_sent}

trace_json:
{trace_text}
"""


def parse_judgment_response(
    *,
    raw_response: str,
    provider: str,
    label: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    parse_text, thinking_metadata = structured_output_text(
        raw_response,
        provider=provider,
    )
    data, error, selection_metadata = extract_last_schema_valid_object(
        parse_text,
        _judgment_schema_error,
        schema_name="trace_judgment",
    )
    metadata: Dict[str, Any] = {
        **thinking_metadata,
        **selection_metadata,
        "parse_error": error,
        "parse_status": "ok" if not error else "fallback",
    }
    if error:
        data = _legacy_parse_response(parse_text, label=label)
        metadata["fallback_parser"] = "legacy_answer_reason"
    normalized = _normalize_judgment(data, label=label)
    return normalized, metadata


def _judgment_schema_error(data: Dict[str, Any]) -> Optional[str]:
    answer = _normalize_answer_value(data.get("answer") or data.get("verdict") or data.get("predict"))
    if answer not in VALID_ANSWERS:
        return "answer/verdict must be target_attack, non_target, or uncertain"
    confidence = str(data.get("confidence") or "").strip().lower()
    if confidence and confidence not in VALID_CONFIDENCE:
        return "confidence must be low, medium, or high"
    stages = data.get("judgment_stages", data.get("stages", []))
    if stages is not None and not isinstance(stages, list):
        return "judgment_stages/stages must be a list"
    evidence = data.get("evidence", [])
    if evidence is not None and not isinstance(evidence, list):
        return "evidence must be a list"
    return None


def _legacy_parse_response(text: str, *, label: str) -> Dict[str, Any]:
    try:
        data = extract_json_object(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    answer = ANSWER_UNCERTAIN
    reason = "Could not parse a structured LLM answer."
    answer_match = re.search(r"^answer:\s*(.+?)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if answer_match:
        answer = _normalize_answer_value(answer_match.group(1), label=label)
    reason_match = re.search(r"^reason:\s*(.+?)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if reason_match:
        reason = reason_match.group(1).strip()
    return {
        "answer": answer,
        "attack_label": label,
        "confidence": "low",
        "reason": reason,
        "judgment_stages": [],
        "evidence": [],
    }


def _normalize_judgment(data: Dict[str, Any], *, label: str) -> Dict[str, Any]:
    answer = _normalize_answer_value(
        data.get("answer") or data.get("verdict") or data.get("predict"),
        label=label,
    )
    confidence = str(data.get("confidence") or "low").strip().lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"
    stages = data.get("judgment_stages", data.get("stages", []))
    if not isinstance(stages, list):
        stages = []
    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    return {
        "answer": answer,
        "attack_label": str(data.get("attack_label") or label),
        "confidence": confidence,
        "reason": str(data.get("reason") or data.get("rationale") or "").strip(),
        "judgment_stages": [_normalize_stage(item) for item in stages],
        "evidence": [_normalize_evidence(item) for item in evidence],
    }


def _normalize_stage(value: Any) -> Dict[str, Any]:
    item = dict(value or {}) if isinstance(value, dict) else {"finding": str(value)}
    status = str(item.get("status") or item.get("verdict") or "uncertain").strip().lower()
    if status not in VALID_STAGE_STATUS:
        status = "uncertain"
    evidence_ids = item.get("evidence_ids", [])
    if isinstance(evidence_ids, str):
        evidence_ids = [evidence_ids]
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    return {
        "name": str(item.get("name") or item.get("stage") or "").strip(),
        "status": status,
        "finding": str(item.get("finding") or item.get("reason") or "").strip(),
        "evidence_ids": [str(x) for x in evidence_ids if str(x).strip()],
    }


def _normalize_evidence(value: Any) -> Dict[str, Any]:
    item = dict(value or {}) if isinstance(value, dict) else {"detail": str(value)}
    return {
        "id": str(item.get("id") or item.get("evidence_id") or item.get("path") or "").strip(),
        "supports": _normalize_answer_value(item.get("supports") or item.get("support")),
        "detail": str(item.get("detail") or item.get("description") or item.get("text") or "").strip(),
    }


def _normalize_answer_value(value: Any, *, label: str = "") -> str:
    text = str(value or "").strip().lower()
    label_text = str(label or "").strip().lower()
    aliases = {
        "target": ANSWER_TARGET,
        "attack": ANSWER_TARGET,
        "target_attack": ANSWER_TARGET,
        "yes": ANSWER_TARGET,
        "true": ANSWER_TARGET,
        "positive": ANSWER_TARGET,
        "non-target": ANSWER_NON_TARGET,
        "non_target": ANSWER_NON_TARGET,
        "not_target": ANSWER_NON_TARGET,
        "benign": ANSWER_NON_TARGET,
        "normal": ANSWER_NON_TARGET,
        "negative": ANSWER_NON_TARGET,
        "no": ANSWER_NON_TARGET,
        "false": ANSWER_NON_TARGET,
        "unknown": ANSWER_UNCERTAIN,
        "uncertain": ANSWER_UNCERTAIN,
        "ambiguous": ANSWER_UNCERTAIN,
    }
    if label_text and text == label_text:
        return ANSWER_TARGET
    return aliases.get(text, ANSWER_UNCERTAIN)


def classify_one(
    *,
    row: Dict[str, Any],
    label: str,
    attack_description: str,
    trace_folder: Optional[Path],
    llm_config: Dict[str, Any],
    max_trace_chars: int,
    trace_truncate_mode: str,
    save_prompt: bool,
) -> Dict[str, Any]:
    tx_hash = get_tx_hash(row)
    trace_path = find_trace_path(row=row, trace_folder=trace_folder)
    base_result = {
        "input_index": int(row.get("_input_index", 0) or 0),
        "HackId": str(row.get("HackId", "")),
        "TxHash": tx_hash,
        "Type": str(row.get("Type", "")),
        "source": str(row.get("_source", "")),
    }
    if trace_path is None:
        return {
            **base_result,
            "predict": ANSWER_UNCERTAIN,
            "confidence": "low",
            "reason": "trace file not found",
            "elapsed": 0.0,
            "judgment_stages": [],
            "evidence": [],
            "parse_metadata": {},
        }

    try:
        trace_text = trace_path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        return {
            **base_result,
            "predict": ANSWER_UNCERTAIN,
            "confidence": "low",
            "reason": f"failed to read trace: {exc}",
            "elapsed": 0.0,
            "judgment_stages": [],
            "evidence": [],
            "parse_metadata": {},
        }

    trace_chars_original = len(trace_text)
    trace_text, truncated = truncate_trace(
        trace_text,
        max_chars=max_trace_chars,
        mode=trace_truncate_mode,
    )
    prompt = build_judge_prompt(
        trace_text=trace_text,
        trace_path=trace_path,
        label=label,
        attack_description=attack_description,
        truncated=truncated,
        trace_chars_original=trace_chars_original,
        trace_chars_sent=len(trace_text),
    )

    started = time.perf_counter()
    try:
        llm = OpenAICompatibleLLM(**llm_config)
        raw_response = llm.complete(prompt)
        elapsed = round(time.perf_counter() - started, 3)
        judgment, parse_metadata = parse_judgment_response(
            raw_response=raw_response,
            provider=llm.provider,
            label=label,
        )
        usage = dict(getattr(llm, "last_usage", {}) or {})
        finish_reason = str(getattr(llm, "last_finish_reason", "") or "")
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 3)
        return {
            **base_result,
            "predict": ANSWER_UNCERTAIN,
            "confidence": "low",
            "reason": f"LLM call failed: {type(exc).__name__}: {exc}",
            "elapsed": elapsed,
            "judgment_stages": [],
            "evidence": [],
            "parse_metadata": {},
        }

    result = {
        **base_result,
        "trace_path": str(trace_path),
        "predict": judgment["answer"],
        "confidence": judgment["confidence"],
        "reason": judgment["reason"],
        "elapsed": elapsed,
        "judgment_stages": judgment["judgment_stages"],
        "evidence": judgment["evidence"],
        "parse_metadata": parse_metadata,
        "usage": usage,
        "finish_reason": finish_reason,
        "trace_chars_original": trace_chars_original,
        "trace_chars_sent": len(trace_text),
        "trace_truncated": truncated,
        "raw_response": raw_response,
    }
    if save_prompt:
        result["prompt"] = prompt
    return result


def read_csv_rows(csv_path: str | Path, *, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = dict(row)
            if get_tx_hash(row):
                row["_source"] = source
                rows.append(row)
    return rows


def get_tx_hash(row: Dict[str, Any]) -> str:
    for key in TX_HASH_COLUMNS:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def find_trace_path(*, row: Dict[str, Any], trace_folder: Optional[Path]) -> Optional[Path]:
    explicit = str(row.get("trace_path") or row.get("TracePath") or "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    tx_hash = get_tx_hash(row)
    if not tx_hash or trace_folder is None:
        return None
    candidates = [
        trace_folder / f"{tx_hash}_trace.json",
        trace_folder / f"{tx_hash}.json",
        trace_folder / tx_hash / "trace.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(trace_folder.rglob(f"{tx_hash}*trace*.json"))
    return matches[0] if matches else None


def truncate_trace(text: str, *, max_chars: int, mode: str) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = (
        "\n\n[TRUNCATED: raw trace exceeded --max-trace-chars; "
        f"{len(text) - max_chars} characters omitted; mode={mode}]\n\n"
    )
    keep = max(0, max_chars - len(marker))
    if mode == "tail":
        return marker + text[-keep:], True
    if mode == "middle":
        head = keep // 2
        tail = keep - head
        return text[:head] + marker + text[-tail:], True
    return text[:keep] + marker, True


def write_results_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_index",
        "HackId",
        "TxHash",
        "Type",
        "source",
        "predict",
        "confidence",
        "reason",
        "elapsed",
        "trace_path",
        "trace_truncated",
        "finish_reason",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[JudgeByTrace] Results CSV: {out_path}")


def write_llm_scripts(
    *,
    rows: List[Dict[str, Any]],
    out_dir: Path,
    label: str,
    model: str,
    provider: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row.get("input_index", 0) or 0))
    manifest: List[Dict[str, Any]] = []
    for fallback_index, row in enumerate(ordered, 1):
        index = int(row.get("input_index", 0) or fallback_index)
        tx_hash = str(row.get("TxHash") or f"row_{index}")
        source = str(row.get("source") or "unknown")
        filename = f"{index:05d}__{_safe_filename(source)}__{_safe_filename(tx_hash)[:72]}.json"
        script_path = out_dir / filename
        payload = {
            "schema_version": 1,
            "stage": "judge_by_trace",
            "label": label,
            "model": model,
            "provider": provider,
            "input_index": index,
            "tx_hash": tx_hash,
            "source": source,
            "type": row.get("Type", ""),
            "trace_path": row.get("trace_path", ""),
            "trace_chars_original": row.get("trace_chars_original"),
            "trace_chars_sent": row.get("trace_chars_sent"),
            "trace_truncated": row.get("trace_truncated"),
            "prompt": row.get("prompt", ""),
            "raw_response": row.get("raw_response", ""),
            "parsed": {
                "answer": row.get("predict"),
                "confidence": row.get("confidence"),
                "reason": row.get("reason"),
                "judgment_stages": row.get("judgment_stages", []),
                "evidence": row.get("evidence", []),
            },
            "parse_metadata": row.get("parse_metadata", {}),
            "usage": row.get("usage", {}),
            "finish_reason": row.get("finish_reason", ""),
            "elapsed": row.get("elapsed"),
        }
        write_json(script_path, payload)
        manifest.append(
            {
                "input_index": index,
                "tx_hash": tx_hash,
                "source": source,
                "predict": row.get("predict"),
                "confidence": row.get("confidence"),
                "script_path": str(script_path),
            }
        )
    write_json(out_dir / "manifest.json", {"items": manifest})
    print(f"[JudgeByTrace] LLM scripts: {out_dir}")


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", str(value or "").strip())
    return text.strip("._") or "unknown"


def compute_metrics(
    *,
    pos_rows: List[Dict[str, Any]],
    neg_rows: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    result_map = {row["TxHash"]: row for row in results}
    target = len(pos_rows)
    nontarget = len(neg_rows)
    tp = tn = fp = fn = uncertain_pos = uncertain_neg = 0

    for row in pos_rows:
        pred = result_map.get(get_tx_hash(row), {}).get("predict", ANSWER_UNCERTAIN)
        if pred == ANSWER_TARGET:
            tp += 1
        elif pred == ANSWER_NON_TARGET:
            fn += 1
        else:
            uncertain_pos += 1

    for row in neg_rows:
        pred = result_map.get(get_tx_hash(row), {}).get("predict", ANSWER_UNCERTAIN)
        if pred == ANSWER_NON_TARGET:
            tn += 1
        elif pred == ANSWER_TARGET:
            fp += 1
        else:
            uncertain_neg += 1

    predict_target = sum(1 for row in results if row.get("predict") == ANSWER_TARGET)
    predict_nontarget = sum(1 for row in results if row.get("predict") == ANSWER_NON_TARGET)
    uncertain = uncertain_pos + uncertain_neg
    total = target + nontarget
    correct = tp + tn
    errors = fp + fn
    return {
        "target": target,
        "nontarget": nontarget,
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "correct": correct,
        "errors": errors,
        "uncertain": uncertain,
        "uncertain_pos": uncertain_pos,
        "uncertain_neg": uncertain_neg,
        "predict_target": predict_target,
        "predict_nontarget": predict_nontarget,
        "attack_recall": round(tp / target, 4) if target else 0.0,
        "attack_precision": round(tp / predict_target, 4) if predict_target else 0.0,
        "negative_precision": round(tn / predict_nontarget, 4) if predict_nontarget else 0.0,
    }


def load_attack_description(args: argparse.Namespace) -> str:
    if args.attack_description:
        return str(args.attack_description)
    if args.attack_description_file:
        return Path(args.attack_description_file).read_text(encoding="utf-8-sig").strip()
    if args.attack_catalog:
        payload = json.loads(Path(args.attack_catalog).read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            value = payload.get(args.label)
            if isinstance(value, dict):
                value = value.get("description") or value.get("attack_description")
            if value:
                return str(value)
    raise ValueError(
        "Provide --attack-description, --attack-description-file, or "
        "--attack-catalog containing the selected --label."
    )


def load_input_rows(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.trace_json:
        trace_path = Path(args.trace_json)
        row = {
            "TxHash": args.tx_hash or trace_path.stem.replace("_trace", ""),
            "trace_path": str(trace_path),
            "Type": args.label,
            "_source": "single",
        }
        return [row], []
    if not args.eval_pos and not args.eval_neg:
        raise ValueError("Provide --trace-json or at least one of --eval-pos/--eval-neg.")
    pos_rows = read_csv_rows(args.eval_pos, source="pos") if args.eval_pos else []
    neg_rows = read_csv_rows(args.eval_neg, source="neg") if args.eval_neg else []
    return pos_rows, neg_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge raw traces for one target attack type with GLM or MiniMax."
    )
    parser.add_argument("--label", required=True, help="Target attack label.")
    parser.add_argument("--attack-description", help="Inline attack description.")
    parser.add_argument("--attack-description-file", help="Text file with attack description.")
    parser.add_argument(
        "--attack-catalog",
        help="JSON mapping labels to descriptions; used when --attack-description is omitted.",
    )
    parser.add_argument("--trace-json", help="Single trace JSON file to judge.")
    parser.add_argument("--tx-hash", help="Optional tx hash for --trace-json mode.")
    parser.add_argument("--eval-pos", help="CSV with positive cases.")
    parser.add_argument("--eval-neg", help="CSV with negative/non-target cases.")
    parser.add_argument("--trace-folder", help="Folder containing trace JSON files for CSV mode.")
    parser.add_argument("--llm-model", default="glm-5.1")
    parser.add_argument("--llm-provider", default="glm", help="glm, minimax, openai, etc.")
    parser.add_argument("--api-key", help="Optional provider API key override.")
    parser.add_argument(
        "-ZhipuApiKey",
        "--zhipu-api-key",
        dest="zhipu_api_key",
        help="Optional Zhipu/GLM API key override.",
    )
    parser.add_argument("--base-url", help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--minimax-thinking",
        choices=["disabled", "adaptive", "provider-default"],
        default="disabled",
        help="MiniMax thinking mode. Parser still strips visible <think> blocks.",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-trace-chars", type=int, default=0)
    parser.add_argument(
        "--trace-truncate-mode",
        choices=["head", "tail", "middle"],
        default="middle",
    )
    parser.add_argument("--out-dir", default="data/results/only_trace")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument(
        "--save-llm-scripts",
        action="store_true",
        help="Write one transcript JSON per LLM call, including prompt and raw response.",
    )
    parser.add_argument(
        "--llm-scripts-dir",
        help=(
            "Optional transcript output directory. If omitted with "
            "--save-llm-scripts, defaults to <out-dir>/llm_scripts/<run_tag>."
        ),
    )
    args = parser.parse_args()

    attack_description = load_attack_description(args)
    pos_rows, neg_rows = load_input_rows(args)
    all_rows = [*pos_rows, *neg_rows]
    for input_index, row in enumerate(all_rows, 1):
        row["_input_index"] = input_index
    trace_folder = Path(args.trace_folder) if args.trace_folder else None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^\w.-]+", "_", args.label).strip("_")
    provider_tag = re.sub(r"[^\w.-]+", "_", str(args.llm_provider or "auto"))
    out_dir = Path(args.out_dir)
    out_tag = f"{safe_label}_{provider_tag}_{ts}"
    llm_scripts_dir: Optional[Path] = None
    if args.llm_scripts_dir:
        llm_scripts_dir = Path(args.llm_scripts_dir)
    elif args.save_llm_scripts:
        llm_scripts_dir = out_dir / "llm_scripts" / out_tag
    save_prompt = bool(args.save_prompts or llm_scripts_dir)

    print(
        "[JudgeByTrace] "
        f"label={args.label} model={args.llm_model} provider={args.llm_provider} "
        f"pos={len(pos_rows)} neg={len(neg_rows)} parallel={args.parallel}"
    )
    print(
        "[JudgeByTrace] "
        f"trace_folder={trace_folder or '(single/explicit)'} "
        f"max_trace_chars={args.max_trace_chars or 'unlimited'}"
    )

    minimax_thinking: Optional[str]
    minimax_thinking = None if args.minimax_thinking == "provider-default" else args.minimax_thinking
    llm_config = {
        "model": args.llm_model,
        "provider": args.llm_provider,
        "api_key": args.zhipu_api_key or args.api_key,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "minimax_thinking": minimax_thinking,
    }

    results: List[Dict[str, Any]] = []
    if args.parallel <= 1:
        for index, row in enumerate(all_rows, 1):
            results.append(
                classify_one(
                    row=row,
                    label=args.label,
                    attack_description=attack_description,
                    trace_folder=trace_folder,
                    llm_config=llm_config,
                    max_trace_chars=args.max_trace_chars,
                    trace_truncate_mode=args.trace_truncate_mode,
                    save_prompt=save_prompt,
                )
            )
            print(f"[JudgeByTrace] {index}/{len(all_rows)} done", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
            futures = [
                executor.submit(
                    classify_one,
                    row=row,
                    label=args.label,
                    attack_description=attack_description,
                    trace_folder=trace_folder,
                    llm_config=llm_config,
                    max_trace_chars=args.max_trace_chars,
                    trace_truncate_mode=args.trace_truncate_mode,
                    save_prompt=save_prompt,
                )
                for row in all_rows
            ]
            for index, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                print(f"[JudgeByTrace] {index}/{len(all_rows)} done", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / f"{out_tag}_results.csv"
    details_json = out_dir / f"{out_tag}_details.json"
    summary_json = out_dir / f"{out_tag}_summary.json"
    write_results_csv(results, results_csv)
    write_json(details_json, {"results": results})
    if llm_scripts_dir is not None:
        write_llm_scripts(
            rows=results,
            out_dir=llm_scripts_dir,
            label=args.label,
            model=args.llm_model,
            provider=args.llm_provider,
        )
    metrics = compute_metrics(pos_rows=pos_rows, neg_rows=neg_rows, results=results)
    summary = {
        "label": args.label,
        "attack_description": attack_description,
        "timestamp": ts,
        "model": args.llm_model,
        "provider": args.llm_provider,
        "minimax_thinking": args.minimax_thinking,
        "max_tokens": args.max_tokens,
        "max_trace_chars": args.max_trace_chars,
        "trace_truncate_mode": args.trace_truncate_mode,
        "llm_scripts_dir": str(llm_scripts_dir) if llm_scripts_dir else "",
        "results_csv": str(results_csv),
        "details_json": str(details_json),
        "metrics": metrics,
    }
    write_json(summary_json, summary)

    print(f"[JudgeByTrace] Details JSON: {details_json}")
    print(f"[JudgeByTrace] Summary JSON: {summary_json}")
    print("\n=== Summary ===")
    for key in (
        "total",
        "correct",
        "errors",
        "tp",
        "tn",
        "fp",
        "fn",
        "uncertain",
        "attack_recall",
        "attack_precision",
        "negative_precision",
    ):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
