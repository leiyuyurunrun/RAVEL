from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional runtime helper
    load_dotenv = None

try:
    import requests
except ImportError:  # pragma: no cover - optional runtime helper
    requests = None

try:
    from evotx.utils.solidityParser.loc_parser import get_loc_info
except Exception:  # pragma: no cover - parser is best-effort
    get_loc_info = None


CHAIN_IDS = {
    "eth": "1",
    "ethereum": "1",
    "bsc": "56",
    "bnb": "56",
    "poly": "137",
    "polygon": "137",
    "arbi": "42161",
    "arbitrum": "42161",
    "opt": "10",
    "optimism": "10",
    "base": "8453",
    "avax": "43114",
    "avalanche": "43114",
    "mantle": "5000",
    "gnosis": "100",
    "linea": "59144",
}

DEFAULT_ETHERSCAN_PROXY_URL = "http://127.0.0.1:7897"


class SourceToolRegistry:
    """Small one-hop source-code tool registry for judge-time follow-up."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/contracts",
        api_key: Optional[str] = None,
        force_refresh: bool = False,
        proxy_url: Optional[str] = None,
    ):
        if load_dotenv:
            load_dotenv()
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key or _resolve_api_key()
        self.force_refresh = force_refresh
        self.proxy_url = (
            proxy_url
            if proxy_url is not None
            else (
                os.getenv("ETHERSCAN_PROXY_URL")
                or os.getenv("SOURCE_CODE_PROXY_URL")
                or DEFAULT_ETHERSCAN_PROXY_URL
            )
        ).strip()
        self.proxy_route_id = _proxy_route_id(self.proxy_url)
        self.read_result_cache: Dict[str, Dict[str, Any]] = {}
        self.failure_cache_ttl_seconds = max(
            0,
            int(os.getenv("SOURCE_FAILURE_CACHE_TTL_SECONDS", "86400") or 86400),
        )

    def call(self, request: Dict[str, Any], default_chain: str = "eth") -> Dict[str, Any]:
        tool = str(request.get("tool") or request.get("name") or "").strip()
        if tool != "read_function_chunk":
            return _tool_response(
                tool="unknown",
                query={"request": request},
                summary={"matched": False},
                evidence={"snippets": []},
                note=f"Unsupported source tool: {tool}",
                tool_status="unsupported_tool",
            )
        return self.read_function_chunk(
            address=str(request.get("address", "")),
            function_name=str(
                request.get("target_function")
                or request.get("function_name")
                or request.get("function")
                or ""
            ),
            chain=str(request.get("chain") or default_chain or "eth"),
            max_snippets=int(request.get("max_snippets", 3) or 3),
            max_chars_per_snippet=int(request.get("max_chars_per_snippet", 3500) or 3500),
        )

    def read_function_chunk(
        self,
        address: str,
        function_name: str,
        chain: str = "eth",
        max_snippets: int = 3,
        max_chars_per_snippet: int = 3500,
    ) -> Dict[str, Any]:
        chain = _normalize_chain(chain)
        address = _normalize_address(address)
        function_name = str(function_name or "").strip()
        query = {
            "tool": "read_function_chunk",
            "chain": chain,
            "address": address,
            "function_name": function_name,
            "max_snippets": max_snippets,
            "max_chars_per_snippet": max_chars_per_snippet,
        }
        cache_key = f"{chain}:{address}:{function_name.lower()}:{max_snippets}:{max_chars_per_snippet}"
        if cache_key in self.read_result_cache:
            print(
                f"[SourceToolRegistry] read_function_chunk call-cache hit "
                f"chain={chain} address={address} function={function_name}"
            )
            cached = json.loads(json.dumps(self.read_result_cache[cache_key], ensure_ascii=False))
            cached.setdefault("summary", {})["from_call_cache"] = True
            return cached

        if not _is_valid_address(address):
            print(
                f"[SourceToolRegistry] read_function_chunk invalid_address "
                f"chain={chain} address={address} function={function_name}"
            )
            return _tool_response(
                tool="read_function_chunk",
                query=query,
                summary={"matched": False, "snippet_count": 0},
                evidence={"snippets": [], "matched_keys": []},
                note="address is empty or malformed; cannot read contract source.",
                tool_status="invalid_address",
            )
        if not function_name:
            print(
                f"[SourceToolRegistry] read_function_chunk invalid_function_name "
                f"chain={chain} address={address}"
            )
            return _tool_response(
                tool="read_function_chunk",
                query=query,
                summary={"matched": False, "snippet_count": 0},
                evidence={"snippets": [], "matched_keys": []},
                note="function_name is empty; cannot select source chunk.",
                tool_status="invalid_function_name",
            )

        try:
            print(
                f"[SourceToolRegistry] read_function_chunk start "
                f"chain={chain} address={address} function={function_name}"
            )
            bundle = self.get_or_download_bundle(address=address, chain=chain)
        except Exception as exc:
            print(
                f"[SourceToolRegistry] read_function_chunk source_unavailable "
                f"chain={chain} address={address} function={function_name} error={exc}"
            )
            result = _tool_response(
                tool="read_function_chunk",
                query=query,
                summary={"matched": False, "snippet_count": 0},
                evidence={"snippets": [], "matched_keys": []},
                note=f"Contract source download or parse failed: {exc}",
                tool_status="source_unavailable",
            )
            self.read_result_cache[cache_key] = result
            return result

        functions = bundle.get("functions", {}) or {}
        snippets, matched_keys = _match_function_snippets(
            functions=functions,
            query=function_name,
            max_snippets=max_snippets,
            max_chars_per_snippet=max_chars_per_snippet,
        )
        implementation_summary: Dict[str, Any] = {}
        matched_bundle = bundle
        if not snippets:
            implementation = _implementation_address(bundle)
            if implementation and implementation != address:
                implementation_summary = {
                    "implementation_fallback_attempted": True,
                    "implementation_address": implementation,
                    "original_address": address,
                }
                try:
                    implementation_bundle = self.get_or_download_bundle(
                        address=implementation,
                        chain=chain,
                    )
                    implementation_functions = implementation_bundle.get("functions", {}) or {}
                    impl_snippets, impl_matched_keys = _match_function_snippets(
                        functions=implementation_functions,
                        query=function_name,
                        max_snippets=max_snippets,
                        max_chars_per_snippet=max_chars_per_snippet,
                    )
                    if impl_snippets:
                        snippets = impl_snippets
                        matched_keys = impl_matched_keys
                        matched_bundle = implementation_bundle
                        implementation_summary.update({
                            "implementation_fallback_used": True,
                            "matched_on": "implementation",
                        })
                    else:
                        implementation_summary["implementation_fallback_used"] = False
                except Exception as exc:
                    implementation_summary.update({
                        "implementation_fallback_used": False,
                        "implementation_fallback_error": repr(exc),
                    })

        summary = {
            "matched": bool(snippets),
            "snippet_count": len(snippets),
            "matched_key_count": len(matched_keys),
            "top_matched_key": matched_keys[0]["key"] if matched_keys else None,
            "from_source_cache": bool(matched_bundle.get("from_cache")),
            "contract_name": (matched_bundle.get("meta", {}) or {}).get("contract_name"),
            "requested_function": function_name,
            "resolved_function": (
                matched_keys[0]["key"] if matched_keys else ""
            ),
            "source_provenance_status": "explorer_verified_unpinned",
            "transaction_block_pinned": False,
            "runtime_bytecode_match_verified": False,
            "decisive_negative_eligible": False,
            "trace_conflict_policy": "observed_runtime_trace_overrides_unpinned_source_negative",
            "source_negative_evidence_scope": "supporting_only_unless_runtime_binding_verified",
        }
        summary.update(implementation_summary)
        if implementation_summary.get("implementation_fallback_used"):
            summary["original_address"] = address
            summary["implementation_address"] = implementation_summary.get("implementation_address")
        modifier_snippets, modifier_summary = _modifier_snippets_for_function_snippets(
            snippets=snippets,
            bundle=matched_bundle,
            max_chars_per_snippet=max_chars_per_snippet,
        )
        if modifier_summary:
            summary.update(modifier_summary)
        result = _tool_response(
            tool="read_function_chunk",
            query=query,
            summary=summary,
            evidence={
                "matched_keys": matched_keys,
                "snippets": snippets,
                "modifier_snippets": modifier_snippets,
                "source_cache_path": matched_bundle.get("cache_path"),
            },
            note=(
                "Returned explorer-verified contract function source chunks, "
                "but the source is not pinned or bytecode-matched to the "
                "transaction block. It may support interpretation but is not "
                "eligible as the sole decisive negative when trace behavior "
                "contradicts it. "
                "Observed trace/state/fund-flow evidence takes precedence over "
                "unpinned source-level guard/order claims. "
                "Modifier snippets are attached when the selected function uses "
                "source-level guards such as onlyOwner/allowed/onlyRole. "
                "Use these only as local supporting or contradicting evidence for the current judge question."
                if snippets
                else (
                    "Verified source was available, but no matching function chunk was found. "
                    "Current and implementation source were checked."
                    if implementation_summary.get("implementation_fallback_attempted")
                    else "Verified source was available, but no matching function chunk was found."
                )
            ),
            tool_status="ok" if snippets else "function_not_found",
        )
        print(
            f"[SourceToolRegistry] read_function_chunk done "
            f"chain={chain} address={address} function={function_name} "
            f"status={result['tool_status']} snippets={len(snippets)} "
            f"from_source_cache={bool(bundle.get('from_cache'))}"
        )
        self.read_result_cache[cache_key] = result
        return result

    def get_or_download_bundle(self, address: str, chain: str) -> Dict[str, Any]:
        chain = _normalize_chain(chain)
        address = _normalize_address(address)
        cache_path = self._bundle_cache_path(chain, address)
        failure_path = self._failure_cache_path(chain, address)
        if cache_path.exists() and not self.force_refresh:
            print(
                f"[SourceToolRegistry] source bundle cache hit "
                f"chain={chain} address={address} path={cache_path}"
            )
            bundle = _read_json(cache_path)
            bundle["from_cache"] = True
            bundle["cache_path"] = str(cache_path)
            return bundle

        if failure_path.exists() and not self.force_refresh:
            failure = _read_json(failure_path)
            if str(failure.get("retrieval_route_id") or "") != self.proxy_route_id:
                # A failure from direct access (or a different proxy route)
                # must not suppress a retry after network routing changes.
                failure_path.unlink(missing_ok=True)
                failure = {}
            failed_at = int(failure.get("failed_at", 0) or 0)
            age_seconds = max(0, int(time.time()) - failed_at)
            if failure and (
                self.failure_cache_ttl_seconds <= 0
                or age_seconds <= self.failure_cache_ttl_seconds
            ):
                print(
                    f"[SourceToolRegistry] source failure cache hit "
                    f"chain={chain} address={address} path={failure_path}"
                )
                raise RuntimeError(
                    "Cached source retrieval failure: "
                    f"{failure.get('error') or 'source unavailable'}"
                )
            failure_path.unlink(missing_ok=True)

        print(
            f"[SourceToolRegistry] downloading verified source "
            f"chain={chain} address={address}"
        )
        try:
            raw = self._download_verified_source(address=address, chain=chain)
            functions = parse_source_files(raw.get("files", {}))
        except Exception as exc:
            _write_json(
                failure_path,
                {
                    "cache_format": "evotx.contract_source.failure.v1",
                    "address": address,
                    "chain": chain,
                    "failed_at": int(time.time()),
                    "error": repr(exc),
                    "retrieval_route_id": self.proxy_route_id,
                },
            )
            raise
        bundle = {
            "cache_format": "evotx.contract_source.bundle.v1",
            "address": address,
            "chain": chain,
            "fetched_at": int(time.time()),
            "meta": {
                "contract_name": raw.get("contract_name"),
                "compiler_version": raw.get("compiler_version"),
                "proxy": raw.get("proxy"),
                "implementation": raw.get("implementation"),
            },
            "files": raw.get("files", {}),
            "functions": functions,
            "from_cache": False,
            "cache_path": str(cache_path),
        }
        _write_json(cache_path, bundle)
        failure_path.unlink(missing_ok=True)
        return bundle

    def _bundle_cache_path(self, chain: str, address: str) -> Path:
        return self.cache_dir / chain / f"{address}__bundle.json"

    def _failure_cache_path(self, chain: str, address: str) -> Path:
        return self.cache_dir / chain / f"{address}__source_failure.json"

    def _download_verified_source(self, address: str, chain: str) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests is not installed")
        if not self.api_key:
            raise RuntimeError(
                "No explorer API key found. Set ETHERSCAN_API_KEY, EXPLORER_API_KEY, or chain-specific scan API key."
            )
        chain_id = CHAIN_IDS.get(chain)
        if not chain_id:
            raise RuntimeError(f"Unsupported chain for explorer source API: {chain}")

        request_kwargs: Dict[str, Any] = {
            "params": {
                "chainid": chain_id,
                "module": "contract",
                "action": "getsourcecode",
                "address": address,
                "apikey": self.api_key,
            },
            "timeout": 30,
        }
        if self.proxy_url:
            request_kwargs["proxies"] = {
                "http": self.proxy_url,
                "https": self.proxy_url,
            }

        response = requests.get(
            "https://api.etherscan.io/v2/api",
            **request_kwargs,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "1" or not data.get("result"):
            raise RuntimeError(f"Explorer API failed: {data}")
        item = data["result"][0]
        source_code = item.get("SourceCode", "")
        if not source_code:
            raise RuntimeError(f"No verified source returned for {address}")
        return {
            "files": _decode_source_code(source_code, item.get("ContractName") or "Contract"),
            "contract_name": item.get("ContractName"),
            "compiler_version": item.get("CompilerVersion"),
            "proxy": item.get("Proxy"),
            "implementation": item.get("Implementation"),
        }


def parse_source_files(files: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    functions: Dict[str, List[Dict[str, Any]]] = {}
    for path, content in (files or {}).items():
        if not isinstance(content, str) or not content.strip():
            continue
        parsed = _parse_with_solidity_listener(content)
        if not parsed:
            parsed = _parse_with_regex(content)
        lines = content.splitlines()
        for name, loc in parsed:
            start = max(int(loc.get("start_line", 1)), 1)
            end = max(int(loc.get("end_line", start)), start)
            body = "\n".join(lines[start - 1 : end])
            item = {
                "path": path,
                "start_line": start,
                "end_line": end,
                "content": body,
                "source_kind": "function",
            }
            functions.setdefault(name, []).append(item)
        for name, loc in _parse_modifier_definitions_with_regex(content):
            start = max(int(loc.get("start_line", 1)), 1)
            end = max(int(loc.get("end_line", start)), start)
            body = "\n".join(lines[start - 1 : end])
            item = {
                "path": path,
                "start_line": start,
                "end_line": end,
                "content": body,
                "source_kind": "modifier",
            }
            functions.setdefault(name, []).append(item)
    return functions


def _parse_with_solidity_listener(content: str) -> List[tuple[str, Dict[str, int]]]:
    if get_loc_info is None:
        return []
    try:
        info = get_loc_info(content)
    except Exception:
        return []
    out: List[tuple[str, Dict[str, int]]] = []
    for contract in info.values():
        if not isinstance(contract, dict):
            continue
        for function_name, loc in (contract.get("functions", {}) or {}).items():
            out.append((function_name or "fallback", loc))
    return out


def _parse_with_regex(content: str) -> List[tuple[str, Dict[str, int]]]:
    lines = content.splitlines()
    starts: List[tuple[str, int]] = []
    pattern = re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(|\b(fallback|receive)\s*\(")
    for index, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if not match:
            continue
        starts.append(((match.group(1) or match.group(2) or "fallback"), index))
    out: List[tuple[str, Dict[str, int]]] = []
    for idx, (name, start_line) in enumerate(starts):
        next_start = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines) + 1
        end_line = min(next_start - 1, start_line + 120)
        out.append((name, {"start_line": start_line, "end_line": end_line}))
    return out


def _parse_modifier_definitions_with_regex(content: str) -> List[tuple[str, Dict[str, int]]]:
    lines = content.splitlines()
    starts: List[tuple[str, int]] = []
    pattern = re.compile(r"\bmodifier\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|\{)")
    for index, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if match:
            starts.append((str(match.group(1) or ""), index))
    out: List[tuple[str, Dict[str, int]]] = []
    for idx, (name, start_line) in enumerate(starts):
        next_start = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines) + 1
        end_line = _solidity_block_end_line(lines, start_line, fallback_end=min(next_start - 1, start_line + 120))
        out.append((name, {"start_line": start_line, "end_line": end_line}))
    return out


def _solidity_block_end_line(lines: List[str], start_line: int, *, fallback_end: int) -> int:
    balance = 0
    opened = False
    for index in range(max(1, start_line), len(lines) + 1):
        line = lines[index - 1]
        balance += line.count("{")
        if "{" in line:
            opened = True
        balance -= line.count("}")
        if opened and balance <= 0:
            return index
    return fallback_end


def _match_function_snippets(
    functions: Dict[str, List[Dict[str, Any]]],
    query: str,
    max_snippets: int,
    max_chars_per_snippet: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized = _normalize_function_query(query)
    candidates: List[tuple[str, int]] = []
    for key in functions.keys():
        key_norm = _normalize_function_query(key)
        score = 0
        if key_norm == normalized:
            score = 100
        elif normalized and normalized in key_norm:
            score = 70
        elif key_norm and key_norm in normalized:
            score = 60
        if score:
            candidates.append((key, score))
    candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)

    snippets: List[Dict[str, Any]] = []
    matched_keys: List[Dict[str, Any]] = []
    seen = set()
    for key, score in candidates:
        local_count = 0
        for src in functions.get(key, []):
            content = str(src.get("content") or "")
            if not content or content in seen:
                continue
            seen.add(content)
            source_code, truncated = _truncate_source(content, max_chars_per_snippet)
            snippets.append({
                "evidence_id": f"contract_source:{src.get('path', 'unknown')}:{src.get('start_line', '')}-{src.get('end_line', '')}",
                "path": src.get("path", "unknown"),
                "start_line": src.get("start_line"),
                "end_line": src.get("end_line"),
                "matched_key": key,
                "match_score": score,
                "source_kind": src.get("source_kind", "function"),
                "source_code": source_code,
                "truncated": truncated,
            })
            local_count += 1
            if len(snippets) >= max_snippets:
                break
        if local_count:
            matched_keys.append({"key": key, "score": score, "snippet_count": local_count})
        if len(snippets) >= max_snippets:
            break
    return snippets, matched_keys


_NON_MODIFIER_TOKENS = {
    "after",
    "anonymous",
    "calldata",
    "constant",
    "external",
    "internal",
    "memory",
    "override",
    "payable",
    "private",
    "public",
    "pure",
    "returns",
    "storage",
    "view",
    "virtual",
}


def _modifier_snippets_for_function_snippets(
    *,
    snippets: List[Dict[str, Any]],
    bundle: Dict[str, Any],
    max_chars_per_snippet: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not snippets:
        return [], {}
    files = bundle.get("files", {}) if isinstance(bundle, dict) else {}
    if not isinstance(files, dict) or not files:
        return [], {}
    modifier_defs = _modifier_definitions_from_files(files)
    if not modifier_defs:
        return [], {}

    requested_names: List[str] = []
    for snippet in snippets:
        for name in _modifier_invocations_from_function_source(
            str(snippet.get("source_code") or ""),
            available_modifier_names=set(modifier_defs.keys()),
        ):
            if name not in requested_names:
                requested_names.append(name)

    attached: List[Dict[str, Any]] = []
    missing: List[str] = []
    seen: set[str] = set()
    for name in requested_names[:4]:
        item = modifier_defs.get(name)
        if not item:
            missing.append(name)
            continue
        signature = f"{item.get('path')}:{item.get('start_line')}:{item.get('end_line')}"
        if signature in seen:
            continue
        seen.add(signature)
        source_code, truncated = _truncate_source(
            str(item.get("content") or ""),
            max(500, min(max_chars_per_snippet, 3500)),
        )
        attached.append({
            "evidence_id": (
                f"contract_modifier:{item.get('path', 'unknown')}:"
                f"{item.get('start_line', '')}-{item.get('end_line', '')}"
            ),
            "path": item.get("path", "unknown"),
            "start_line": item.get("start_line"),
            "end_line": item.get("end_line"),
            "matched_key": name,
            "source_kind": "modifier",
            "source_code": source_code,
            "truncated": truncated,
            "attached_reason": "modifier_invoked_by_returned_function",
        })

    summary = {
        "attached_modifier_count": len(attached),
        "attached_modifier_names": [str(item.get("matched_key") or "") for item in attached],
    }
    if missing:
        summary["missing_modifier_names"] = missing
    return attached, summary if requested_names else {}


def _modifier_definitions_from_files(files: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path, content in (files or {}).items():
        if not isinstance(content, str) or not content.strip():
            continue
        lines = content.splitlines()
        for name, loc in _parse_modifier_definitions_with_regex(content):
            if not name or name in out:
                continue
            start = max(int(loc.get("start_line", 1)), 1)
            end = max(int(loc.get("end_line", start)), start)
            out[name] = {
                "path": path,
                "start_line": start,
                "end_line": end,
                "content": "\n".join(lines[start - 1 : end]),
            }
    return out


def _modifier_invocations_from_function_source(
    source_code: str,
    *,
    available_modifier_names: set[str],
) -> List[str]:
    text = str(source_code or "")
    if not text.strip() or not available_modifier_names:
        return []
    header = text.split("{", 1)[0]
    match = re.search(r"\bfunction\b[^{;]*?\)", header)
    if match:
        tail = header[match.end():]
    else:
        tail = header
    if "returns" in tail:
        tail = tail.split("returns", 1)[0]
    names: List[str] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(?:\([^;{}]*\))?", tail):
        name = str(match.group(1) or "")
        if (
            not name
            or name in _NON_MODIFIER_TOKENS
            or name not in available_modifier_names
            or name in names
        ):
            continue
        names.append(name)
    return names


def _decode_source_code(source_code: str, contract_name: str) -> Dict[str, str]:
    text = str(source_code or "").strip()
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1].strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("sources"), dict):
                files = {}
                for file_name, meta in obj["sources"].items():
                    if isinstance(meta, dict):
                        files[file_name] = str(meta.get("content", ""))
                if files:
                    return files
        except Exception:
            pass
    return {f"{contract_name or 'Contract'}.sol": str(source_code or "")}


def _tool_response(
    tool: str,
    query: Dict[str, Any],
    summary: Dict[str, Any],
    evidence: Dict[str, Any],
    note: str,
    tool_status: str,
) -> Dict[str, Any]:
    return {
        "tool": tool,
        "tool_status": tool_status,
        "query": query,
        "summary": summary,
        "evidence": evidence,
        "note": note,
    }


def _resolve_api_key() -> Optional[str]:
    return (
        os.getenv("ETHERSCAN_API_KEY")
        or os.getenv("EXPLORER_API_KEY")
        or os.getenv("BSCSAN_API_KEY")
        or os.getenv("BSCSCAN_API_KEY")
        or os.getenv("POLYGONSCAN_API_KEY")
        or os.getenv("ARBISCAN_API_KEY")
        or os.getenv("BASESCAN_API_KEY")
    )


def _normalize_chain(chain: str) -> str:
    text = str(chain or "eth").strip().lower()
    aliases = {"ethereum": "eth", "bnb": "bsc", "polygon": "poly", "arbitrum": "arbi", "optimism": "opt", "mantle": "mantle", "gnosis": "gnosis", "arb": "arbi", "op": "opt", "linea": "linea", "chain_59144": "linea", "chain_5000": "mantle", "chain_100": "gnosis"}
    return aliases.get(text, text or "eth")


def _proxy_route_id(proxy_url: str) -> str:
    value = str(proxy_url or "").strip()
    if not value:
        return "direct"
    parsed = urlsplit(value)
    host = str(parsed.hostname or "unknown").lower()
    port = f":{parsed.port}" if parsed.port is not None else ""
    scheme = str(parsed.scheme or "proxy").lower()
    return f"proxy:{scheme}://{host}{port}"


def _normalize_address(address: str) -> str:
    return str(address or "").strip().lower()


def _normalize_function_query(query: str) -> str:
    text = str(query or "").strip().lower()
    if "(" in text:
        text = text.split("(", 1)[0]
    return text.replace("0x", "")


def _implementation_address(bundle: Dict[str, Any]) -> str:
    meta = bundle.get("meta", {}) if isinstance(bundle, dict) else {}
    implementation = _normalize_address(str((meta or {}).get("implementation") or ""))
    if not _is_valid_address(implementation):
        return ""
    if implementation == "0x0000000000000000000000000000000000000000":
        return ""
    return implementation


def _is_valid_address(address: str) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", str(address or "").strip()))


def _truncate_source(content: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(content) <= max_chars:
        return content, False
    return content[:max_chars] + "\n...[truncated by read_function_chunk]...", True


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
