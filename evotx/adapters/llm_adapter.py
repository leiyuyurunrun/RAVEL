from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import httpx

XUNFEI_DEFAULT_MODEL = "xopglm5"
XUNFEI_DEFAULT_BASE_URL = (
    "https://maas-coding-api.cn-huabei-1.xf-yun.com/anthropic"
)
XUNFEI_DEFAULT_TIMEOUT_SECONDS = 600.0
XUNFEI_DEFAULT_MAX_TOKENS = 32768
GLM_EN_DEFAULT_TIMEOUT_SECONDS = 600.0
GLM_EN_DEFAULT_MAX_TOKENS = 32768
GLM_EN_DEFAULT_CONNECTION_RETRIES = 4
GLM_EN_DEFAULT_CONNECTION_RETRY_DELAY_SECONDS = 5.0
GLM_EN_DEFAULT_CONNECTION_RETRY_JITTER_RATIO = 0.25
GLM_EN_DEFAULT_CONNECTION_RETRY_MAX_DELAY_SECONDS = 60.0
GLM_EN_DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS = 8192
MINIMAX_MAX_COMPLETION_TOKENS = 32768
ANTHROPIC_DEFAULT_MAX_TOKENS = 32768
ZHIPU_GENERAL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_CODING_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
ZHIPU_ANTHROPIC_BASE_URL = "https://open.bigmodel.cn/api/anthropic"
MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimaxi.com/anthropic"
ZAI_ANTHROPIC_BASE_URL = "https://api.z.ai/api/anthropic"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
OPENAI_RESPONSES_BASE_URL = "https://8-219-220-32.sslip.io/v1"
_PROXY_ENVIRONMENT_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def _configured_proxy_environment_variables() -> List[str]:
    return [name for name in _PROXY_ENVIRONMENT_VARIABLES if os.getenv(name)]


def _configured_system_proxy_schemes() -> List[str]:
    try:
        proxies = urllib.request.getproxies()
    except Exception:
        return []
    return sorted(
        str(scheme).lower()
        for scheme, value in proxies.items()
        if str(scheme).lower() in {"http", "https", "all"} and value
    )


def resolve_adaptive_llm_route(
    *,
    model: Optional[str],
    provider: Optional[str],
    thinking: Optional[str],
    adaptive_model: Optional[str] = None,
    adaptive_provider: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the optional second LLM tier for adaptive-thinking roles."""
    use_adaptive_tier = str(thinking or "").strip().lower() == "adaptive"
    resolved_model = (
        adaptive_model if use_adaptive_tier and adaptive_model else model
    )
    resolved_provider = (
        adaptive_provider if use_adaptive_tier and adaptive_provider else provider
    )
    return resolved_model, resolved_provider


class OpenAICompatibleLLM:
    """Minimal provider-neutral adapter exposing EvoTx's LLM surface.

    Most configured providers use OpenAI-compatible chat completions. Xunfei
    always uses Anthropic Messages; GLM and MiniMax can opt into their
    Anthropic-compatible endpoints. The transport distinction stays inside this
    adapter so all EvoTx stages continue to use the same methods.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        minimax_thinking: Optional[str] = None,
    ):
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleLLM requires python-dotenv. Install the "
                "EvoTx runtime dependencies before using LLM-powered stages."
            ) from exc

        load_dotenv()
        self.temperature = temperature
        self.provider = _infer_provider(model=model, provider=provider)
        self.model = _resolve_model(
            model, self.provider, provider_arg=provider)
        self.transport = _provider_transport(self.provider)
        self.openai_api_route = _openai_api_route(self.provider)
        self.last_usage: Dict[str, Any] = {}
        # Observable thinking metadata from the most recent provider response.
        # ``thinking_tokens=None`` means the response exposed thinking content
        # but did not provide a separate token count; it must not be treated as 0.
        self.last_thinking_metadata: Dict[str, Any] = {
            "thinking_present": None,
            "thinking_chars": None,
            "thinking_tokens_estimated": None,
            "thinking_tokens": None,
            "thinking_tokens_source": None,
        }
        self.last_finish_reason = ""
        self.api_key = api_key or _resolve_api_key(self.provider)
        if not self.api_key:
            raise ValueError(
                f"No API key found for provider '{self.provider}'. "
                "Set it in .env or pass api_key explicitly."
            )

        self.base_url = base_url or _resolve_base_url(self.provider)
        self.timeout = _resolve_timeout(self.provider, timeout)
        self.max_tokens = _resolve_max_tokens(self.provider, max_tokens)
        self.sdk_max_retries = _resolve_sdk_connection_retries(
            self.provider,
            self.transport,
        )
        self.minimax_thinking = _resolve_minimax_thinking_type(
            provider=self.provider,
            model=self.model,
            value=minimax_thinking,
        )
        self.glm_thinking_override = _resolve_glm_stage_thinking_override(
            value=minimax_thinking,
        )
        self.stage_thinking = self.glm_thinking_override or self.minimax_thinking
        self.last_request_config: Dict[str, Any] = {}
        proxy_environment_variables = _configured_proxy_environment_variables()
        system_proxy_schemes = _configured_system_proxy_schemes()
        use_system_proxy = self.provider == "openai"
        self.network_route = (
            "proxy"
            if use_system_proxy and system_proxy_schemes
            else "system_default"
            if use_system_proxy
            else "direct"
        )
        self.network_route_scope = "application_http_client"
        self.application_proxy = "enabled" if use_system_proxy else "disabled"
        self.http_client_trust_env = use_system_proxy
        self.proxy_environment_variables = proxy_environment_variables
        self.system_proxy_schemes = system_proxy_schemes

        if self.transport == "anthropic":
            try:
                from anthropic import Anthropic, DefaultHttpxClient
            except ImportError as exc:
                raise ImportError(
                    f"The {self.provider} provider requires the anthropic package. "
                    "Install EvoTx dependencies or run 'pip install anthropic'."
                ) from exc
            # http_client = DefaultHttpxClient(trust_env=False)
            direct_transport = httpx.HTTPTransport(
                trust_env=False,
            )

            http_client = DefaultHttpxClient(
                trust_env=False,
                transport=direct_transport,
            )
            self.http_client_trust_env = bool(http_client.trust_env)
            client_kwargs: Dict[str, Any] = {
                "base_url": self.base_url,
                # Agent traffic must stay direct even when the process has a
                # proxy configured for explorer/source-code downloads.
                "http_client": http_client,
            }
            if self.timeout is not None:
                client_kwargs["timeout"] = self.timeout
            if self.sdk_max_retries is not None:
                client_kwargs["max_retries"] = self.sdk_max_retries
            if self.provider in {"glm", "minimax"}:
                # Both official SDK examples authenticate with x-api-key.
                client_kwargs["api_key"] = self.api_key
            else:
                # Preserve Xunfei/glm-en's existing Bearer-token behavior.
                client_kwargs["auth_token"] = self.api_key
            self.client = Anthropic(**client_kwargs)
            print(
                "[LLM RUNTIME NETWORK DEBUG]",
                "adapter_file=", __file__,
                "provider=", self.provider,
                "transport=", self.transport,
                "base_url=", self.base_url,
                "http_client_type=", type(http_client),
                "trust_env=", getattr(http_client, "trust_env", "UNKNOWN"),
                flush=True,
            )
        else:
            try:
                from openai import DefaultHttpxClient, OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAI-compatible providers require the openai package. "
                    "Install EvoTx runtime dependencies first."
                ) from exc
            http_client = DefaultHttpxClient(trust_env=use_system_proxy)
            self.http_client_trust_env = bool(http_client.trust_env)
            client_kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                # OpenAI uses the host proxy; all other providers stay direct.
                "http_client": http_client,
            }
            if self.timeout is not None:
                client_kwargs["timeout"] = self.timeout
            if self.sdk_max_retries is not None:
                client_kwargs["max_retries"] = self.sdk_max_retries
            self.client = OpenAI(**client_kwargs)
            print(
                "[LLM RUNTIME NETWORK DEBUG]",
                "adapter_file=", __file__,
                "provider=", self.provider,
                "transport=", self.transport,
                "base_url=", self.base_url,
                "http_client_type=", type(http_client),
                "trust_env=", getattr(http_client, "trust_env", "UNKNOWN"),
                flush=True,
            )

    def clone_for_worker(self) -> "OpenAICompatibleLLM":
        """Return an independent adapter instance for concurrent judge workers."""
        return self.clone_with_max_tokens(self.max_tokens)

    def clone_with_max_tokens(
        self,
        max_tokens: Optional[int],
    ) -> "OpenAICompatibleLLM":
        """Return an equivalent adapter with a stage-specific output budget."""
        return OpenAICompatibleLLM(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            provider=self.provider,
            timeout=self.timeout,
            max_tokens=max_tokens,
            minimax_thinking=self.stage_thinking,
        )

    def complete(self, prompt: str) -> str:
        try:
            self._log_request_start(messages_count=1, tools=False)
            if self.transport == "anthropic":
                response = self._anthropic_request(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                self._record_response_metadata(response)
                return _anthropic_text(response)

            if self._uses_openai_responses():
                kwargs = self._openai_responses_kwargs(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                response = self._openai_responses_create(**kwargs)
                self._record_response_metadata(response)
                return _openai_responses_text(response)
            kwargs = self._openai_chat_kwargs(
                messages=[{"role": "user", "content": prompt}],
                response_format=(
                    {"type": "json_object"}
                    if self.provider == "minimax"
                    else None
                ),
            )
            response = self._openai_chat_completion_create(**kwargs)
            self._record_response_metadata(response)
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(
                "[LLM Adapter Error] complete failed: "
                f"{_exception_chain_summary(exc)}"
            )
            raise

    def generate(self, messages, tools=None):
        """Call the selected provider and normalize optional tool calls."""
        msg_count = len(messages)
        self._log_request_start(messages_count=msg_count, tools=bool(tools))

        try:
            if self.transport == "anthropic":
                response = self._anthropic_request(
                    messages=messages,
                    tools=tools,
                    temperature=0.1,
                )
                self._record_response_metadata(response)
                msg_dict = {
                    "role": "assistant",
                    "content": _anthropic_text(response),
                }
                tool_calls = _anthropic_tool_calls(response)
                if tool_calls:
                    print(
                        "[LLM] Tool calls: "
                        f"{[item['function']['name'] for item in tool_calls]}"
                    )
                    msg_dict["tool_calls"] = tool_calls
                return msg_dict

            if self._uses_openai_responses():
                kwargs = self._openai_responses_kwargs(
                    messages=messages,
                    temperature=0.1,
                    tools=tools,
                )
                response = self._openai_responses_create(**kwargs)
                self._record_response_metadata(response)
                msg_dict = {
                    "role": "assistant",
                    "content": _openai_responses_text(response),
                }
                tool_calls = _openai_responses_tool_calls(response)
                if tool_calls:
                    print(
                        "[LLM] Tool calls: "
                        f"{[item['function']['name'] for item in tool_calls]}"
                    )
                    msg_dict["tool_calls"] = tool_calls
                return msg_dict

            kwargs = self._openai_chat_kwargs(
                messages=messages,
                temperature=0.1,
                tools=tools,
            )

            response = self._openai_chat_completion_create(**kwargs)
            message = response.choices[0].message
            self._record_response_metadata(response)
            msg_dict = {
                "role": "assistant",
                "content": message.content or "",
            }
            if getattr(message, "tool_calls", None):
                print(
                    "[LLM] Tool calls: "
                    f"{[item.function.name for item in message.tool_calls]}"
                )
                msg_dict["tool_calls"] = [
                    {
                        "id": tool.id,
                        "type": tool.type,
                        "function": {
                            "name": tool.function.name,
                            "arguments": tool.function.arguments,
                        },
                    }
                    for tool in message.tool_calls
                ]
            return msg_dict
        except Exception as exc:
            print(
                "[LLM Adapter Error] Request failed: "
                f"{_exception_chain_summary(exc)}"
            )
            return {"role": "assistant", "content": f"LLM request failed: {exc}"}

    def simple_query(self, prompt, system_message="You are a helpful assistant."):
        """Small helper for direct JSON-ish LLM calls."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        try:
            self._log_request_start(messages_count=len(messages), tools=False)
            if self.transport == "anthropic":
                response = self._anthropic_request(
                    messages=messages,
                    temperature=0.1,
                )
                self._record_response_metadata(response)
                content = _anthropic_text(response)
            else:
                if self._uses_openai_responses():
                    kwargs = self._openai_responses_kwargs(
                        messages=messages,
                        temperature=0.1,
                    )
                    response = self._openai_responses_create(**kwargs)
                    self._record_response_metadata(response)
                    content = _openai_responses_text(response)
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return content
                response_format = None
                if self.provider in {"glm", "glm-en", "minimax"}:
                    response_format = {"type": "json_object"}
                    print(
                        "[LLM] Requesting response_format=json_object "
                        f"(provider={self.provider}, model={self.model})"
                    )
                kwargs = self._openai_chat_kwargs(
                    messages=messages,
                    temperature=0.1,
                    response_format=response_format,
                )
                response = self._openai_chat_completion_create(**kwargs)
                self._record_response_metadata(response)
                content = response.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        except Exception as exc:
            print(
                "[LLM Adapter Error] simple_query failed: "
                f"{_exception_chain_summary(exc)}"
            )
            return {"error": str(exc)}

    def _anthropic_request(
        self,
        *,
        messages,
        tools=None,
        temperature: Optional[float] = None,
    ):
        system, normalized_messages = _to_anthropic_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": normalized_messages,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None and not (
            self.provider == "minimax" and temperature <= 0
        ):
            kwargs["temperature"] = temperature
        thinking_override = (
            self.minimax_thinking
            if self.provider == "minimax"
            else self.glm_thinking_override
        )
        anthropic_thinking = _resolve_anthropic_thinking(
            provider=self.provider,
            max_tokens=self.max_tokens,
            override=thinking_override,
        )
        if anthropic_thinking:
            kwargs["thinking"] = anthropic_thinking
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
            kwargs["tool_choice"] = {"type": "auto"}
        self.last_request_config = {
            **self.config_summary(),
            "temperature": kwargs.get("temperature"),
            "tools": bool(tools),
        }
        return self._anthropic_messages_create(**kwargs)

    def _uses_openai_responses(self) -> bool:
        route = getattr(self, "openai_api_route", None)
        return (route or _openai_api_route(self.provider)) == "responses"

    def _anthropic_messages_create(self, **kwargs):
        max_retries = _resolve_anthropic_connection_retries(self.provider)
        retry_delay = _resolve_anthropic_connection_retry_delay(self.provider)
        jitter_ratio = _resolve_anthropic_connection_retry_jitter_ratio(
            self.provider
        )
        max_delay = _resolve_anthropic_connection_retry_max_delay(
            self.provider
        )

        # GLM / GLM-EN:
        # adaptive 最终映射为 thinking.type=enabled。
        # 对这类长 thinking 请求优先使用 streaming，避免服务端在完整
        # response 返回之前主动断开非流式连接。
        thinking = kwargs.get("thinking") or {}
        thinking_type = (
            str(thinking.get("type") or "").strip().lower()
            if isinstance(thinking, dict)
            else ""
        )

        prefer_stream = (
            self.provider in {"glm", "glm-en"}
            and thinking_type == "enabled"
        )

        for attempt in range(max_retries + 1):
            try:
                if prefer_stream:
                    print(
                        "[LLM] Using Anthropic streaming for thinking request: "
                        f"provider={self.provider} "
                        f"thinking={thinking_type} "
                        f"max_tokens={kwargs.get('max_tokens')}"
                    )
                    with self.client.messages.stream(**kwargs) as stream:
                        return stream.get_final_message()

                # disabled / non-thinking request still takes the fast non-streaming path.
                try:
                    return self.client.messages.create(**kwargs)
                except ValueError as exc:
                    if "Streaming is required for operations" not in str(exc):
                        raise

                    print(
                        "[LLM] Anthropic-compatible request exceeds the "
                        "non-streaming duration limit; switching to streaming "
                        f"provider={self.provider} "
                        f"max_tokens={kwargs.get('max_tokens')}"
                    )
                    with self.client.messages.stream(**kwargs) as stream:
                        return stream.get_final_message()

            except Exception as exc:
                retryable = _is_retryable_anthropic_connection_error(exc)

                if attempt >= max_retries or not retryable:
                    print(
                        "[LLM] Anthropic-compatible request failed: "
                        f"request_attempt={attempt + 1}/{max_retries + 1} "
                        f"adapter_retries={max_retries} "
                        f"sdk_retries={_display_sdk_retries(self.sdk_max_retries)} "
                        f"provider={self.provider} retryable={retryable} "
                        f"retry_exhausted={retryable and attempt >= max_retries} "
                        f"error_chain={_exception_chain_summary(exc)}"
                    )
                    raise

                wait_seconds = _connection_retry_wait_seconds(
                    initial_delay_seconds=retry_delay,
                    retry_index=attempt,
                    jitter_ratio=jitter_ratio,
                    max_delay_seconds=max_delay,
                )

                print(
                    "[LLM] Retrying Anthropic-compatible request after "
                    f"{type(exc).__name__}: adapter_retry={attempt + 1}/"
                    f"{max_retries} request_attempt={attempt + 1}/"
                    f"{max_retries + 1} provider={self.provider} "
                    f"base_url={self.base_url} sdk_retries="
                    f"{_display_sdk_retries(self.sdk_max_retries)} "
                    f"wait_seconds={wait_seconds:.2f} "
                    f"jitter_ratio={jitter_ratio:.2f} "
                    f"error_chain={_exception_chain_summary(exc)}"
                )

                time.sleep(wait_seconds)

    def _openai_chat_completion_create(self, **kwargs):
        max_retries = _resolve_openai_connection_retries(self.provider)
        retry_delay = _resolve_openai_connection_retry_delay(self.provider)
        jitter_ratio = _resolve_openai_connection_retry_jitter_ratio(self.provider)
        max_delay = _resolve_openai_connection_retry_max_delay(self.provider)
        for attempt in range(max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                retryable = _is_retryable_openai_connection_error(exc)
                if attempt >= max_retries or not retryable:
                    print(
                        "[LLM] OpenAI-compatible request failed: "
                        f"request_attempt={attempt + 1}/{max_retries + 1} "
                        f"adapter_retries={max_retries} "
                        f"sdk_retries={_display_sdk_retries(self.sdk_max_retries)} "
                        f"provider={self.provider} retryable={retryable} "
                        f"retry_exhausted={retryable and attempt >= max_retries} "
                        f"error_chain={_exception_chain_summary(exc)}"
                    )
                    raise
                wait_seconds = _connection_retry_wait_seconds(
                    initial_delay_seconds=retry_delay,
                    retry_index=attempt,
                    jitter_ratio=jitter_ratio,
                    max_delay_seconds=max_delay,
                )
                print(
                    "[LLM] Retrying OpenAI-compatible request after "
                    f"{type(exc).__name__}: adapter_retry={attempt + 1}/"
                    f"{max_retries} request_attempt={attempt + 1}/"
                    f"{max_retries + 1} provider={self.provider} "
                    f"sdk_retries={_display_sdk_retries(self.sdk_max_retries)} "
                    f"wait_seconds={wait_seconds:.2f} "
                    f"jitter_ratio={jitter_ratio:.2f} "
                    f"error_chain={_exception_chain_summary(exc)}"
                )
                time.sleep(wait_seconds)

    def _openai_responses_create(self, **kwargs):
        max_retries = _resolve_openai_connection_retries(self.provider)
        retry_delay = _resolve_openai_connection_retry_delay(self.provider)
        jitter_ratio = _resolve_openai_connection_retry_jitter_ratio(self.provider)
        max_delay = _resolve_openai_connection_retry_max_delay(self.provider)
        for attempt in range(max_retries + 1):
            try:
                return self.client.responses.create(**kwargs)
            except Exception as exc:
                retryable = _is_retryable_openai_connection_error(exc)
                if attempt >= max_retries or not retryable:
                    print(
                        "[LLM] OpenAI Responses request failed: "
                        f"request_attempt={attempt + 1}/{max_retries + 1} "
                        f"adapter_retries={max_retries} "
                        f"sdk_retries={_display_sdk_retries(self.sdk_max_retries)} "
                        f"provider={self.provider} retryable={retryable} "
                        f"retry_exhausted={retryable and attempt >= max_retries} "
                        f"error_chain={_exception_chain_summary(exc)}"
                    )
                    raise
                wait_seconds = _connection_retry_wait_seconds(
                    initial_delay_seconds=retry_delay,
                    retry_index=attempt,
                    jitter_ratio=jitter_ratio,
                    max_delay_seconds=max_delay,
                )
                print(
                    "[LLM] Retrying OpenAI Responses request after "
                    f"{type(exc).__name__}: adapter_retry={attempt + 1}/"
                    f"{max_retries} request_attempt={attempt + 1}/"
                    f"{max_retries + 1} provider={self.provider} "
                    f"sdk_retries={_display_sdk_retries(self.sdk_max_retries)} "
                    f"wait_seconds={wait_seconds:.2f} "
                    f"jitter_ratio={jitter_ratio:.2f} "
                    f"error_chain={_exception_chain_summary(exc)}"
                )
                time.sleep(wait_seconds)

    def _openai_responses_kwargs(
        self,
        *,
        messages,
        temperature: Optional[float] = None,
        tools=None,
    ) -> Dict[str, Any]:
        instructions, response_input = _to_openai_responses_input(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": response_input,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = _to_openai_responses_tools(tools)
            kwargs["tool_choice"] = "auto"
        self.last_request_config = {
            **self.config_summary(),
            "temperature": kwargs.get("temperature"),
            "tools": bool(tools),
        }
        return kwargs

    def _openai_chat_kwargs(
        self,
        *,
        messages,
        temperature: Optional[float] = None,
        tools=None,
        response_format: Optional[dict] = None,
        extra_body: Optional[dict] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.max_tokens is not None and self.provider == "minimax":
            kwargs["max_completion_tokens"] = self.max_tokens
        elif self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format is not None:
            kwargs["response_format"] = response_format

        merged_extra_body = dict(extra_body or {})
        if self.provider == "minimax":
            merged_extra_body.setdefault("reasoning_split", True)
            minimax_thinking = str(
                getattr(self, "minimax_thinking", "") or ""
            ).strip().lower()
            if minimax_thinking and minimax_thinking != "default":
                merged_extra_body["thinking"] = {"type": minimax_thinking}
        if self.provider in {"glm", "glm-en"}:
            thinking_type = _resolve_glm_thinking_type(
                self.provider,
                override=getattr(self, "glm_thinking_override", ""),
            )
            if thinking_type:
                merged_extra_body["thinking"] = {"type": thinking_type}
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body
        self.last_request_config = {
            **self.config_summary(),
            "temperature": kwargs.get("temperature"),
            "tools": bool(tools),
            "response_format": response_format,
        }
        return kwargs

    def _record_response_metadata(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.last_usage = _usage_to_dict(usage)
        self.last_thinking_metadata = _response_thinking_metadata(
            response,
            normalized_usage=self.last_usage,
        )

        # Keep normalized usage self-contained for existing callers that persist
        # last_usage. Unknown thinking-token counts remain None, not zero.
        self.last_usage["thinking_present"] = self.last_thinking_metadata[
            "thinking_present"
        ]
        self.last_usage["thinking_chars"] = self.last_thinking_metadata[
            "thinking_chars"
        ]
        self.last_usage["thinking_tokens_estimated"] = (
            self.last_thinking_metadata["thinking_tokens_estimated"]
        )
        self.last_usage["thinking_tokens"] = self.last_thinking_metadata[
            "thinking_tokens"
        ]
        self.last_usage["thinking_tokens_source"] = self.last_thinking_metadata[
            "thinking_tokens_source"
        ]
        if "reasoning_tokens" not in self.last_usage:
            self.last_usage["reasoning_tokens"] = self.last_thinking_metadata[
                "thinking_tokens"
            ]

        self.last_finish_reason = str(
            getattr(response, "stop_reason", "")
            or _openai_finish_reason(response)
            or _openai_responses_finish_reason(response)
            or ""
        )
        if self.last_usage:
            extras = []
            for key in (
                "reasoning_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                value = self.last_usage.get(key)
                if value is not None:
                    extras.append(f"{key}={value}")
            extras.append(
                "thinking_present="
                f"{self.last_thinking_metadata.get('thinking_present')}"
            )
            thinking_chars = self.last_thinking_metadata.get("thinking_chars")
            if thinking_chars is not None:
                extras.append(f"thinking_chars={thinking_chars}")
            estimated_tokens = self.last_thinking_metadata.get(
                "thinking_tokens_estimated"
            )
            if estimated_tokens is not None:
                extras.append(
                    f"thinking_tokens_estimated={estimated_tokens}"
                )
            thinking_source = self.last_thinking_metadata.get(
                "thinking_tokens_source"
            )
            if thinking_source:
                extras.append(f"thinking_tokens_source={thinking_source}")
            extra_text = f", {', '.join(extras)}" if extras else ""
            print(
                "[LLM] Response usage: "
                f"prompt_tokens={self.last_usage.get('prompt_tokens')}, "
                f"completion_tokens={self.last_usage.get('completion_tokens')}, "
                f"total={self.last_usage.get('total_tokens')}"
                f"{extra_text}"
            )

    def config_summary(self) -> Dict[str, Any]:
        summary = describe_llm_config(
            model=getattr(self, "model", None),
            provider=getattr(self, "provider", None),
            base_url=getattr(self, "base_url", None),
            timeout=getattr(self, "timeout", None),
            max_tokens=getattr(self, "max_tokens", None),
            minimax_thinking=(
                getattr(self, "stage_thinking", None)
                or getattr(self, "minimax_thinking", None)
            ),
        )
        summary["sdk_connection_retries"] = _display_sdk_retries(
            getattr(self, "sdk_max_retries", None)
        )
        summary.update({
            "network_route": getattr(self, "network_route", "direct"),
            "network_route_scope": getattr(
                self,
                "network_route_scope",
                "application_http_client",
            ),
            "application_proxy": getattr(
                self,
                "application_proxy",
                "disabled",
            ),
            "http_client_trust_env": bool(
                getattr(self, "http_client_trust_env", False)
            ),
            "proxy_environment_variables": list(
                getattr(self, "proxy_environment_variables", []) or []
            ),
            "system_proxy_present": bool(
                getattr(self, "system_proxy_schemes", [])
            ),
            "system_proxy_schemes": list(
                getattr(self, "system_proxy_schemes", []) or []
            ),
            "proxy_environment_ignored": bool(
                getattr(self, "proxy_environment_variables", [])
            ) and not bool(getattr(self, "http_client_trust_env", False)),
            "system_proxy_ignored": bool(
                getattr(self, "system_proxy_schemes", [])
            ) and not bool(getattr(self, "http_client_trust_env", False)),
            "external_network_route_observable": False,
        })
        return summary

    def _log_request_start(self, *, messages_count: int, tools: bool) -> None:
        config = self.config_summary()
        thinking_config = config.get("thinking")
        thinking_display: Any = thinking_config
        if isinstance(thinking_config, dict) and not (
            set(thinking_config) - {"transport"}
        ):
            thinking_display = "provider_default"
        elif not thinking_display:
            thinking_display = "provider_default"
        proxy_env_names = list(config.get("proxy_environment_variables") or [])
        proxy_env_state = (
            ("ignored:" if config.get("proxy_environment_ignored") else "enabled:")
            + ",".join(proxy_env_names)
            if proxy_env_names
            else "none"
        )
        system_proxy_schemes = list(config.get("system_proxy_schemes") or [])
        system_proxy_state = (
            ("ignored:" if config.get("system_proxy_ignored") else "enabled:")
            + ",".join(system_proxy_schemes)
            if system_proxy_schemes
            else "none"
        )
        adapter_retries = config.get(
            f"{self.transport}_connection_retries",
            0,
        )
        print(
            "[LLM] Sending request: "
            f"messages={messages_count} tools={'yes' if tools else 'no'} "
            f"provider={config.get('provider')} "
            f"transport={config.get('transport')} "
            f"api_route={config.get('api_route')} "
            f"model={config.get('model')} "
            f"base_url={config.get('base_url') or '(default)'} "
            f"max_tokens={config.get('max_tokens')} "
            f"thinking={thinking_display} "
            f"timeout={config.get('timeout_seconds')} "
            f"network_route={config.get('network_route')} "
            f"route_scope={config.get('network_route_scope')} "
            f"application_proxy={config.get('application_proxy')} "
            f"trust_env={str(config.get('http_client_trust_env')).lower()} "
            f"proxy_env={proxy_env_state} "
            f"system_proxy={system_proxy_state} "
            "external_route=unobservable "
            f"adapter_retries={adapter_retries} "
            f"sdk_retries={config.get('sdk_connection_retries')}"
        )


def _infer_provider(model: str, provider: Optional[str] = None) -> str:
    configured = (
        provider
        or os.getenv("EVOTX_LLM_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or ""
    ).strip().lower()
    if configured:
        aliases = {
            "zhipu": "glm",
            "bigmodel": "glm",
            "zai": "glm-en",
            "z.ai": "glm-en",
            "glm_en": "glm-en",
            "glmen": "glm-en",
            "minimaxi": "minimax",
            "ark": "volcano",
            "volc": "volcano",
            "volcengine": "volcano",
            "volcano": "volcano",
            "volcanic": "volcano",
            "doubao": "volcano",
            "bytedance": "volcano",
            "xfyun": "xunfei",
            "iflytek": "xunfei",
            "astron": "xunfei",
            "astroncodingplan": "xunfei",
            "xingchen": "xunfei",
            "xunfei": "xunfei",
        }
        return aliases.get(configured, configured)

    model_lower = str(model or "").lower()
    if "astron-code" in model_lower:
        return "xunfei"
    if "deepseek-v4-pro" in model_lower:
        return "volcano"
    if "minimax" in model_lower or model_lower.startswith("abab"):
        return "minimax"
    if "glm" in model_lower or "zhipu" in model_lower:
        return "glm"
    return "openai"


def describe_llm_config(
    *,
    model: Optional[str],
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    minimax_thinking: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_provider = _infer_provider(model=model or "", provider=provider)
    resolved_model = _resolve_model(
        model,
        resolved_provider,
        provider_arg=provider,
    )
    resolved_transport = _provider_transport(resolved_provider)
    resolved_api_route = _openai_api_route(resolved_provider)
    resolved_base_url = base_url or _resolve_base_url(resolved_provider)
    resolved_timeout = _resolve_timeout(resolved_provider, timeout)
    resolved_max_tokens = _resolve_max_tokens(resolved_provider, max_tokens)
    resolved_minimax_thinking = _resolve_minimax_thinking_type(
        provider=resolved_provider,
        model=resolved_model,
        value=minimax_thinking,
    )
    glm_thinking_override = _resolve_glm_stage_thinking_override(
        value=minimax_thinking,
    )
    proxy_environment_variables = _configured_proxy_environment_variables()
    system_proxy_schemes = _configured_system_proxy_schemes()
    use_system_proxy = resolved_provider == "openai"
    thinking: Dict[str, Any] = {}
    if resolved_transport == "anthropic":
        thinking_override = (
            resolved_minimax_thinking
            if resolved_provider == "minimax"
            else glm_thinking_override
        )
        anthropic_thinking = _resolve_anthropic_thinking(
            provider=resolved_provider,
            max_tokens=resolved_max_tokens,
            override=thinking_override,
        )
        if anthropic_thinking:
            thinking = {"transport": "anthropic", **anthropic_thinking}
        else:
            thinking = {"transport": "anthropic"}
    elif resolved_provider in {"glm", "glm-en"}:
        thinking_type = _resolve_glm_thinking_type(
            resolved_provider,
            override=glm_thinking_override,
        )
        if thinking_type:
            thinking = {"transport": "openai", "type": thinking_type}
    elif (
        resolved_provider == "minimax"
        and resolved_minimax_thinking
        and resolved_minimax_thinking != "default"
    ):
        thinking = {
            "transport": "openai",
            "type": resolved_minimax_thinking,
        }

    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "transport": resolved_transport,
        "api_route": resolved_api_route,
        "base_url": resolved_base_url,
        "timeout_seconds": resolved_timeout,
        "max_tokens": resolved_max_tokens,
        "network_route": (
            "proxy"
            if use_system_proxy and system_proxy_schemes
            else "system_default"
            if use_system_proxy
            else "direct"
        ),
        "network_route_scope": "application_http_client",
        "application_proxy": "enabled" if use_system_proxy else "disabled",
        "http_client_trust_env": use_system_proxy,
        "proxy_environment_present": bool(proxy_environment_variables),
        "proxy_environment_variables": proxy_environment_variables,
        "proxy_environment_ignored": (
            bool(proxy_environment_variables) and not use_system_proxy
        ),
        "system_proxy_present": bool(system_proxy_schemes),
        "system_proxy_schemes": system_proxy_schemes,
        "system_proxy_ignored": bool(system_proxy_schemes) and not use_system_proxy,
        "external_network_route_observable": False,
        "thinking": thinking,
        "minimax_thinking": resolved_minimax_thinking,
        "glm_thinking_override": glm_thinking_override,
        "openai_connection_retries": _resolve_openai_connection_retries(
            resolved_provider
        ),
        "openai_connection_retry_delay_seconds": (
            _resolve_openai_connection_retry_delay(resolved_provider)
        ),
        "anthropic_connection_retries": _resolve_anthropic_connection_retries(
            resolved_provider
        ),
        "anthropic_connection_retry_delay_seconds": (
            _resolve_anthropic_connection_retry_delay(resolved_provider)
        ),
        "openai_connection_retry_jitter_ratio": (
            _resolve_openai_connection_retry_jitter_ratio(resolved_provider)
        ),
        "openai_connection_retry_max_delay_seconds": (
            _resolve_openai_connection_retry_max_delay(resolved_provider)
        ),
        "anthropic_connection_retry_jitter_ratio": (
            _resolve_anthropic_connection_retry_jitter_ratio(resolved_provider)
        ),
        "anthropic_connection_retry_max_delay_seconds": (
            _resolve_anthropic_connection_retry_max_delay(resolved_provider)
        ),
        "sdk_connection_retries": _display_sdk_retries(
            _resolve_sdk_connection_retries(
                resolved_provider,
                resolved_transport,
            )
        ),
    }


def _resolve_model(
    model: Optional[str],
    provider: str,
    provider_arg: Optional[str] = None,
) -> str:
    model_text = str(model or "").strip()
    provider_explicit = bool(str(provider_arg or "").strip())
    if provider == "volcano" and (
        not model_text or (provider_explicit and model_text == "glm-5.1")
    ):
        return os.getenv("VOLCANO_MODEL") or os.getenv("ARK_MODEL") or "deepseek-v4-pro"
    if provider == "xunfei" and (
        not model_text or (provider_explicit and model_text == "glm-5.1")
    ):
        anthropic_model = (
            os.getenv("ANTHROPIC_MODEL")
            if _xunfei_anthropic_env_enabled()
            else None
        )
        return (
            os.getenv("XUNFEI_MODEL")
            or os.getenv("XFYUN_MODEL")
            or os.getenv("ASTRON_MODEL")
            or anthropic_model
            or XUNFEI_DEFAULT_MODEL
        )
    if model_text:
        return model_text
    if provider == "glm-en":
        return os.getenv("GLM_EN_MODEL") or os.getenv("ZAI_MODEL") or "glm-5.1"
    if provider == "glm":
        return os.getenv("ZHIPU_MODEL") or os.getenv("GLM_MODEL") or "glm-5.1"
    if provider == "minimax":
        return os.getenv("MINIMAX_MODEL") or "abab6.5s-chat"
    return os.getenv("OPENAI_MODEL") or "gpt-4o-mini"


def _resolve_api_key(provider: str) -> Optional[str]:
    if provider == "xunfei":
        anthropic_key = None
        if _xunfei_anthropic_env_enabled():
            anthropic_key = (
                os.getenv("ANTHROPIC_AUTH_TOKEN")
                or os.getenv("ANTHROPIC_API_KEY")
            )
        return (
            os.getenv("XUNFEI_API_KEY")
            or os.getenv("XFYUN_API_KEY")
            or os.getenv("ASTRON_API_KEY")
            or anthropic_key
        )
    if provider == "minimax":
        return (
            os.getenv("MINIMAX_API_KEY")
            or os.getenv("MINIMAX_API_TOKEN")
            or (
                os.getenv("ANTHROPIC_API_KEY")
                if _anthropic_transport_enabled(provider)
                else None
            )
            or (
                os.getenv("ANTHROPIC_AUTH_TOKEN")
                if _anthropic_transport_enabled(provider)
                else None
            )
        )
    if provider == "volcano":
        return (
            os.getenv("VOLCANO_API_KEY")
            or os.getenv("VOCANO_API_KEY")
            or os.getenv("VOLCENGINE_API_KEY")
            or os.getenv("ARK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    if provider == "glm":
        return (
            os.getenv("ZHIPU_API_KEY")
            or os.getenv("GLM_API_KEY")
            or (
                os.getenv("ANTHROPIC_API_KEY")
                if _anthropic_transport_enabled(provider)
                else None
            )
            or (
                os.getenv("ANTHROPIC_AUTH_TOKEN")
                if _anthropic_transport_enabled(provider)
                else None
            )
            or os.getenv("OPENAI_API_KEY")
        )
    if provider == "glm-en":
        return (
            os.getenv("GLM_EN_API_KEY")
            or os.getenv("ZAI_API_KEY")
            or os.getenv("ZAI_API_TOKEN")
            or os.getenv("ZHIPU_API_KEY")
            or os.getenv("GLM_API_KEY")
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or os.getenv("ANTHROPIC_API_KEY")
        )
    return os.getenv("OPENAI_API_KEY")


def _resolve_base_url(provider: str) -> Optional[str]:
    if provider == "xunfei":
        anthropic_base_url = (
            os.getenv("ANTHROPIC_BASE_URL")
            if _xunfei_anthropic_env_enabled()
            else None
        )
        return (
            os.getenv("XUNFEI_BASE_URL")
            or os.getenv("XFYUN_BASE_URL")
            or os.getenv("ASTRON_BASE_URL")
            or anthropic_base_url
            or XUNFEI_DEFAULT_BASE_URL
        )
    if provider == "minimax":
        if _anthropic_transport_enabled(provider):
            return (
                os.getenv("MINIMAX_ANTHROPIC_BASE_URL")
                or MINIMAX_ANTHROPIC_BASE_URL
            )
        return os.getenv("MINIMAX_BASE_URL") or "https://api.minimaxi.com/v1"
    if provider == "volcano":
        return (
            os.getenv("VOLCANO_BASE_URL")
            or os.getenv("VOCANO_BASE_URL")
            or os.getenv("VOLCENGINE_BASE_URL")
            or os.getenv("ARK_BASE_URL")
            or "https://ark.cn-beijing.volces.com/api/coding/v3"
        )
    if provider == "glm":
        if _anthropic_transport_enabled(provider):
            return (
                os.getenv("ZHIPU_ANTHROPIC_BASE_URL")
                or os.getenv("GLM_ANTHROPIC_BASE_URL")
                or ZHIPU_ANTHROPIC_BASE_URL
            )
        return (
            os.getenv("ZHIPU_BASE_URL")
            or os.getenv("GLM_BASE_URL")
            or ZHIPU_CODING_BASE_URL
        )
    if provider == "glm-en":
        if _glm_en_anthropic_enabled():
            return (
                os.getenv("GLM_EN_ANTHROPIC_BASE_URL")
                or os.getenv("ZAI_ANTHROPIC_BASE_URL")
                or ZAI_ANTHROPIC_BASE_URL
            )
        return (
            os.getenv("GLM_EN_BASE_URL")
            or os.getenv("ZAI_BASE_URL")
            or os.getenv("ZAI_CODING_BASE_URL")
            or ZAI_CODING_BASE_URL
        )
    return os.getenv("OPENAI_BASE_URL") or OPENAI_RESPONSES_BASE_URL
# 在这里，如果要改记得是这

def _resolve_timeout(provider: str, value: Optional[float]) -> Optional[float]:
    if value is not None:
        return float(value)
    if provider == "glm-en":
        for name in ("GLM_EN_TIMEOUT_SECONDS", "ZAI_TIMEOUT_SECONDS"):
            raw = str(os.getenv(name) or "").strip()
            if raw:
                return max(1.0, float(raw))
        return GLM_EN_DEFAULT_TIMEOUT_SECONDS
    if provider != "xunfei":
        return None
    for name in ("XUNFEI_TIMEOUT_SECONDS", "XFYUN_TIMEOUT_SECONDS", "ASTRON_TIMEOUT_SECONDS"):
        raw = str(os.getenv(name) or "").strip()
        if raw:
            return max(1.0, float(raw))
    timeout_ms = (
        str(os.getenv("API_TIMEOUT_MS") or "").strip()
        if _xunfei_anthropic_env_enabled()
        else ""
    )
    if timeout_ms:
        return max(1.0, float(timeout_ms) / 1000.0)
    return XUNFEI_DEFAULT_TIMEOUT_SECONDS


def _resolve_max_tokens(provider: str, value: Optional[int]) -> Optional[int]:
    if value is not None:
        resolved = max(1, int(value))
        if provider == "minimax":
            return min(resolved, MINIMAX_MAX_COMPLETION_TOKENS)
        return resolved
    if provider == "glm-en":
        for name in ("GLM_EN_MAX_TOKENS", "ZAI_MAX_TOKENS"):
            raw = str(os.getenv(name) or "").strip()
            if raw:
                return max(1, int(raw))
        return GLM_EN_DEFAULT_MAX_TOKENS
    if provider in {"glm", "minimax"} and _anthropic_transport_enabled(provider):
        names = (
            ("ZHIPU_ANTHROPIC_MAX_TOKENS", "GLM_ANTHROPIC_MAX_TOKENS")
            if provider == "glm"
            else ("MINIMAX_ANTHROPIC_MAX_TOKENS", "MINIMAX_MAX_TOKENS")
        )
        for name in names:
            raw = str(os.getenv(name) or "").strip()
            if raw:
                resolved = max(1, int(raw))
                return (
                    min(resolved, MINIMAX_MAX_COMPLETION_TOKENS)
                    if provider == "minimax"
                    else resolved
                )
        return ANTHROPIC_DEFAULT_MAX_TOKENS
    if provider != "xunfei":
        return None
    for name in ("XUNFEI_MAX_TOKENS", "XFYUN_MAX_TOKENS", "ASTRON_MAX_TOKENS"):
        raw = str(os.getenv(name) or "").strip()
        if raw:
            return max(1, int(raw))
    return XUNFEI_DEFAULT_MAX_TOKENS


def _resolve_openai_connection_retries(provider: str) -> int:
    names = (
        ("GLM_EN_CONNECTION_RETRIES", "ZAI_CONNECTION_RETRIES")
        if provider == "glm-en"
        else (f"{provider.upper()}_CONNECTION_RETRIES",)
    )
    for name in names:
        raw = str(os.getenv(name) or "").strip()
        if raw:
            return max(0, int(raw))
    return GLM_EN_DEFAULT_CONNECTION_RETRIES if provider == "glm-en" else 0


def _resolve_openai_connection_retry_delay(provider: str) -> float:
    names = (
        (
            "GLM_EN_CONNECTION_RETRY_DELAY_SECONDS",
            "ZAI_CONNECTION_RETRY_DELAY_SECONDS",
        )
        if provider == "glm-en"
        else (f"{provider.upper()}_CONNECTION_RETRY_DELAY_SECONDS",)
    )
    for name in names:
        raw = str(os.getenv(name) or "").strip()
        if raw:
            return max(0.1, float(raw))
    if provider == "glm-en":
        return GLM_EN_DEFAULT_CONNECTION_RETRY_DELAY_SECONDS
    return 1.0


def _resolve_openai_connection_retry_jitter_ratio(provider: str) -> float:
    names = (
        (
            "GLM_EN_CONNECTION_RETRY_JITTER_RATIO",
            "ZAI_CONNECTION_RETRY_JITTER_RATIO",
        )
        if provider == "glm-en"
        else (f"{provider.upper()}_CONNECTION_RETRY_JITTER_RATIO",)
    )
    return _resolve_retry_float(
        names,
        default=(
            GLM_EN_DEFAULT_CONNECTION_RETRY_JITTER_RATIO
            if provider == "glm-en"
            else 0.0
        ),
        minimum=0.0,
        maximum=1.0,
    )


def _resolve_openai_connection_retry_max_delay(provider: str) -> float:
    names = (
        (
            "GLM_EN_CONNECTION_RETRY_MAX_DELAY_SECONDS",
            "ZAI_CONNECTION_RETRY_MAX_DELAY_SECONDS",
        )
        if provider == "glm-en"
        else (f"{provider.upper()}_CONNECTION_RETRY_MAX_DELAY_SECONDS",)
    )
    return _resolve_retry_float(
        names,
        default=(
            GLM_EN_DEFAULT_CONNECTION_RETRY_MAX_DELAY_SECONDS
            if provider == "glm-en"
            else 60.0
        ),
        minimum=0.1,
    )


def _resolve_anthropic_connection_retries(provider: str) -> int:
    if provider == "glm-en":
        names = (
            "GLM_EN_ANTHROPIC_CONNECTION_RETRIES",
            "ZAI_ANTHROPIC_CONNECTION_RETRIES",
            "ANTHROPIC_CONNECTION_RETRIES",
        )
        default = GLM_EN_DEFAULT_CONNECTION_RETRIES
    else:
        names = (
            f"{provider.upper().replace('-', '_')}_ANTHROPIC_CONNECTION_RETRIES",
            "ANTHROPIC_CONNECTION_RETRIES",
        )
        default = 0
    for name in names:
        raw = str(os.getenv(name) or "").strip()
        if raw:
            return max(0, int(raw))
    return default


def _resolve_anthropic_connection_retry_delay(provider: str) -> float:
    if provider == "glm-en":
        names = (
            "GLM_EN_ANTHROPIC_CONNECTION_RETRY_DELAY_SECONDS",
            "ZAI_ANTHROPIC_CONNECTION_RETRY_DELAY_SECONDS",
            "ANTHROPIC_CONNECTION_RETRY_DELAY_SECONDS",
        )
        default = GLM_EN_DEFAULT_CONNECTION_RETRY_DELAY_SECONDS
    else:
        names = (
            f"{provider.upper().replace('-', '_')}_ANTHROPIC_CONNECTION_RETRY_DELAY_SECONDS",
            "ANTHROPIC_CONNECTION_RETRY_DELAY_SECONDS",
        )
        default = 1.0
    for name in names:
        raw = str(os.getenv(name) or "").strip()
        if raw:
            return max(0.1, float(raw))
    return default


def _resolve_anthropic_connection_retry_jitter_ratio(provider: str) -> float:
    if provider == "glm-en":
        names = (
            "GLM_EN_ANTHROPIC_CONNECTION_RETRY_JITTER_RATIO",
            "ZAI_ANTHROPIC_CONNECTION_RETRY_JITTER_RATIO",
            "GLM_EN_CONNECTION_RETRY_JITTER_RATIO",
            "ZAI_CONNECTION_RETRY_JITTER_RATIO",
            "ANTHROPIC_CONNECTION_RETRY_JITTER_RATIO",
        )
        default = GLM_EN_DEFAULT_CONNECTION_RETRY_JITTER_RATIO
    else:
        names = (
            f"{provider.upper().replace('-', '_')}_ANTHROPIC_CONNECTION_RETRY_JITTER_RATIO",
            "ANTHROPIC_CONNECTION_RETRY_JITTER_RATIO",
        )
        default = 0.0
    return _resolve_retry_float(
        names,
        default=default,
        minimum=0.0,
        maximum=1.0,
    )


def _resolve_anthropic_connection_retry_max_delay(provider: str) -> float:
    if provider == "glm-en":
        names = (
            "GLM_EN_ANTHROPIC_CONNECTION_RETRY_MAX_DELAY_SECONDS",
            "ZAI_ANTHROPIC_CONNECTION_RETRY_MAX_DELAY_SECONDS",
            "GLM_EN_CONNECTION_RETRY_MAX_DELAY_SECONDS",
            "ZAI_CONNECTION_RETRY_MAX_DELAY_SECONDS",
            "ANTHROPIC_CONNECTION_RETRY_MAX_DELAY_SECONDS",
        )
        default = GLM_EN_DEFAULT_CONNECTION_RETRY_MAX_DELAY_SECONDS
    else:
        names = (
            f"{provider.upper().replace('-', '_')}_ANTHROPIC_CONNECTION_RETRY_MAX_DELAY_SECONDS",
            "ANTHROPIC_CONNECTION_RETRY_MAX_DELAY_SECONDS",
        )
        default = 60.0
    return _resolve_retry_float(
        names,
        default=default,
        minimum=0.1,
    )


def _resolve_sdk_connection_retries(
    provider: str,
    transport: str,
) -> Optional[int]:
    if provider != "glm-en":
        return None
    transport_key = transport.upper().replace("-", "_")
    names = (
        f"GLM_EN_{transport_key}_SDK_MAX_RETRIES",
        f"ZAI_{transport_key}_SDK_MAX_RETRIES",
        "GLM_EN_SDK_MAX_RETRIES",
        "ZAI_SDK_MAX_RETRIES",
    )
    for name in names:
        raw = str(os.getenv(name) or "").strip()
        if raw:
            return max(0, int(raw))
    return 0


def _resolve_retry_float(
    names: Tuple[str, ...],
    *,
    default: float,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    resolved = float(default)
    for name in names:
        raw = str(os.getenv(name) or "").strip()
        if raw:
            resolved = float(raw)
            break
    resolved = max(minimum, resolved)
    if maximum is not None:
        resolved = min(maximum, resolved)
    return resolved


def _connection_retry_wait_seconds(
    *,
    initial_delay_seconds: float,
    retry_index: int,
    jitter_ratio: float,
    max_delay_seconds: float,
) -> float:
    capped_delay = min(
        max(0.1, float(max_delay_seconds)),
        max(0.1, float(initial_delay_seconds)) * (2 ** max(0, retry_index)),
    )
    ratio = min(1.0, max(0.0, float(jitter_ratio)))
    if ratio <= 0:
        return capped_delay
    jitter = random.uniform(-capped_delay * ratio, capped_delay * ratio)
    return max(0.1, min(float(max_delay_seconds), capped_delay + jitter))


def _display_sdk_retries(value: Optional[int]) -> Any:
    return "provider_default" if value is None else value


def _exception_chain_summary(exc: BaseException, max_depth: int = 6) -> str:
    parts: List[str] = []
    pending: Optional[BaseException] = exc
    seen: set[int] = set()
    while pending is not None and len(parts) < max(1, max_depth):
        if id(pending) in seen:
            parts.append("cycle_detected")
            break
        seen.add(id(pending))
        text = " ".join(str(pending).split())
        if len(text) > 600:
            text = f"{text[:597]}..."
        parts.append(
            f"{type(pending).__name__}: {text}"
            if text
            else type(pending).__name__
        )
        nested = getattr(pending, "__cause__", None)
        if not isinstance(nested, BaseException):
            nested = getattr(pending, "__context__", None)
        pending = nested if isinstance(nested, BaseException) else None
    return " <- ".join(parts)


def _is_retryable_openai_connection_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(
        marker in name or marker in text
        for marker in (
            "apiconnectionerror",
            "api connection error",
            "connection error",
            "apitimeouterror",
            "timeout",
            "readtimeout",
            "connecttimeout",
            "remoteprotocolerror",
            "server disconnected",
        )
    )


def _is_retryable_anthropic_connection_error(exc: Exception) -> bool:
    return _is_retryable_openai_connection_error(exc)


def _provider_transport(provider: str) -> str:
    if provider == "xunfei":
        return "anthropic"
    if provider in {"glm", "glm-en", "minimax"} and _anthropic_transport_enabled(
        provider
    ):
        return "anthropic"
    return "openai"


def _openai_api_route(provider: str) -> str:
    return "responses" if provider == "openai" else "chat_completions"


def _resolve_glm_stage_thinking_override(value: Optional[str]) -> str:
    configured = str(value or "").strip().lower()
    aliases = {
        "enabled": "enabled",
        "on": "enabled",
        "true": "enabled",
        "adaptive": "",
        "default": "default",
        "provider-default": "default",
        "provider_default": "default",
        "disabled": "disabled",
        "off": "disabled",
        "false": "disabled",
    }
    return aliases.get(configured, "")


def _resolve_glm_thinking_type(
    provider: str = "glm",
    *,
    override: str = "",
) -> str:
    override = str(override or "").strip().lower()
    if override == "default":
        return ""
    if override in {"enabled", "disabled"}:
        return override
    names = (
        ("GLM_EN_THINKING", "ZAI_THINKING", "GLM_THINKING")
        if provider == "glm-en"
        else ("GLM_THINKING",)
    )
    for name in names:
        value = str(os.getenv(name) or "").strip().lower()
        if value in {"enabled", "disabled"}:
            return value
    return ""


def _glm_en_anthropic_enabled() -> bool:
    return _anthropic_transport_enabled("glm-en")


def _anthropic_transport_enabled(provider: str) -> bool:
    provider_names = {
        "glm": (
            "GLM_ENABLE_ANTHROPIC",
            "ZHIPU_ENABLE_ANTHROPIC",
        ),
        "glm-en": (
            "GLM_EN_ENABLE_ANTHROPIC",
            "ZAI_ENABLE_ANTHROPIC",
            "ENABLE_GLM_EN_ANTHROPIC",
        ),
        "minimax": ("MINIMAX_ENABLE_ANTHROPIC",),
    }
    names = (
        *provider_names.get(provider, ()),
        "EVOTX_ENABLE_ANTHROPIC",
        "ENABLE_ANTHROPIC",
    )
    # Existing PowerShell wrappers expose EnableAnthropic through this legacy
    # variable. Keep it as the final fallback for GLM and MiniMax as well.
    if provider in {"glm", "minimax"}:
        names = (*names, "GLM_EN_ENABLE_ANTHROPIC")
    for name in names:
        value = str(os.getenv(name) or "").strip().lower()
        if value in {"1", "true", "yes", "on", "enabled"}:
            return True
        if value in {"0", "false", "no", "off", "disabled"}:
            return False
    return False


def _resolve_anthropic_thinking(
    *,
    provider: str,
    max_tokens: Optional[int],
    override: str = "",
) -> Optional[Dict[str, Any]]:
    if str(override or "").strip().lower() == "default":
        return None
    if provider == "minimax":
        if not _anthropic_transport_enabled(provider):
            return None
        thinking_type = str(override or "").strip().lower()
        if not thinking_type:
            thinking_type = str(
                os.getenv("MINIMAX_ANTHROPIC_THINKING")
                or os.getenv("MINIMAX_THINKING")
                or ""
            ).strip().lower()
        thinking_type = {
            "enabled": "adaptive",
            "on": "adaptive",
            "true": "adaptive",
            "off": "disabled",
            "false": "disabled",
        }.get(thinking_type, thinking_type)
        if thinking_type in {"adaptive", "disabled"}:
            return {"type": thinking_type}
        return None

    if provider not in {"glm", "glm-en"} or not _anthropic_transport_enabled(
        provider
    ):
        return None
    thinking_type = str(override or "").strip().lower()
    if thinking_type not in {"enabled", "disabled"}:
        thinking_type = ""
        names = (
            (
                "GLM_EN_ANTHROPIC_THINKING",
                "ZAI_ANTHROPIC_THINKING",
                "GLM_EN_THINKING",
                "ZAI_THINKING",
                "GLM_THINKING",
            )
            if provider == "glm-en"
            else (
                "ZHIPU_ANTHROPIC_THINKING",
                "GLM_ANTHROPIC_THINKING",
                "GLM_THINKING",
            )
        )
        for name in names:
            value = str(os.getenv(name) or "").strip().lower()
            if value in {"enabled", "disabled"}:
                thinking_type = value
                break
    if not thinking_type:
        thinking_type = "enabled"
    if thinking_type != "enabled":
        return {"type": "disabled"}

    budget = _resolve_anthropic_thinking_budget(max_tokens=max_tokens)
    if budget <= 0:
        return {"type": "enabled"}
    return {"type": "enabled", "budget_tokens": budget}


def _resolve_anthropic_thinking_budget(
    *,
    max_tokens: Optional[int],
) -> int:
    for name in (
        "GLM_EN_ANTHROPIC_THINKING_BUDGET_TOKENS",
        "ZAI_ANTHROPIC_THINKING_BUDGET_TOKENS",
        "ANTHROPIC_THINKING_BUDGET_TOKENS",
    ):
        raw = str(os.getenv(name) or "").strip()
        if raw:
            return max(1, int(raw))
    if max_tokens is None:
        return GLM_EN_DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS
    if max_tokens <= 1024:
        return max(1, max_tokens // 2)
    return min(
        GLM_EN_DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS,
        max(1, int(max_tokens) - 1),
    )


def _resolve_minimax_thinking_type(
    *,
    provider: str,
    model: str,
    value: Optional[str],
) -> str:
    if provider != "minimax":
        return ""
    requested = str(value or "").strip().lower()
    if requested in {"default", "provider-default", "provider_default"}:
        return "default"
    override = str(os.getenv("MINIMAX_THINKING_OVERRIDE") or "").strip()
    configured = str(
        override
        or (value if value is not None else os.getenv("MINIMAX_THINKING") or "")
    ).strip().lower()
    aliases = {
        "enabled": "adaptive",
        "on": "adaptive",
        "true": "adaptive",
        "off": "disabled",
        "false": "disabled",
    }
    configured = aliases.get(configured, configured)
    if configured not in {"adaptive", "disabled", "default"}:
        return ""
    return configured


def _xunfei_anthropic_env_enabled() -> bool:
    base_url = str(os.getenv("ANTHROPIC_BASE_URL") or "").strip().lower()
    return "xf-yun.com" in base_url


def _to_openai_responses_input(
    messages,
) -> Tuple[str, List[Dict[str, Any]]]:
    instructions: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for item in messages or []:
        message = dict(item or {})
        role = str(message.get("role") or "user").strip().lower()
        content = _content_to_text(message.get("content", ""))
        if role in {"system", "developer"}:
            if content:
                instructions.append(content)
            continue
        if role == "tool":
            normalized.append({
                "type": "function_call_output",
                "call_id": str(message.get("tool_call_id") or ""),
                "output": content,
            })
            continue
        if role == "assistant" and message.get("tool_calls"):
            if content:
                normalized.append({"role": "assistant", "content": content})
            for raw_call in list(message.get("tool_calls") or []):
                call = dict(raw_call or {})
                function = dict(call.get("function") or {})
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                normalized.append({
                    "type": "function_call",
                    "call_id": str(call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": arguments,
                })
            continue
        normalized.append({
            "role": "assistant" if role == "assistant" else "user",
            "content": content,
        })
    return "\n\n".join(instructions), normalized


def _to_openai_responses_tools(tools) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in tools or []:
        tool = dict(item or {})
        function = dict(tool.get("function") or {})
        if function:
            converted: Dict[str, Any] = {
                "type": "function",
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": dict(
                    function.get("parameters") or {"type": "object"}
                ),
            }
            if "strict" in function:
                converted["strict"] = bool(function.get("strict"))
            normalized.append(converted)
            continue
        normalized.append(tool)
    return normalized


def _openai_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    parts: List[str] = []
    for item in list(getattr(response, "output", []) or []):
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, dict):
            item_type = item.get("type")
        if item_type != "message":
            continue
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        for block in list(content or []):
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type not in {"output_text", "refusal"}:
                continue
            value = getattr(block, "text", None)
            if value is None:
                value = getattr(block, "refusal", None)
            if value is None and isinstance(block, dict):
                value = block.get("text") or block.get("refusal")
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def _openai_responses_tool_calls(response: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in list(getattr(response, "output", []) or []):
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, dict):
            item_type = item.get("type")
        if item_type != "function_call":
            continue
        call_id = getattr(item, "call_id", None)
        name = getattr(item, "name", None)
        arguments = getattr(item, "arguments", None)
        if isinstance(item, dict):
            call_id = call_id or item.get("call_id") or item.get("id")
            name = name or item.get("name")
            arguments = arguments if arguments is not None else item.get("arguments")
        calls.append({
            "id": str(call_id or ""),
            "type": "function",
            "function": {
                "name": str(name or ""),
                "arguments": (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments or {}, ensure_ascii=False)
                ),
            },
        })
    return calls


def _to_anthropic_messages(messages) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for item in messages or []:
        message = dict(item or {})
        role = str(message.get("role") or "user").strip().lower()
        content = message.get("content", "")
        if role == "system":
            text = _content_to_text(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            normalized.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(message.get("tool_call_id") or ""),
                            "content": _content_to_text(content),
                        }
                    ],
                }
            )
            continue

        anth_content: Any = content if content not in (None, "") else ""
        tool_calls = list(message.get("tool_calls") or [])
        if role == "assistant" and tool_calls:
            blocks: List[Dict[str, Any]] = []
            text = _content_to_text(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tool_call in tool_calls:
                function = dict((tool_call or {}).get("function") or {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_arguments": arguments}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str((tool_call or {}).get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "input": arguments if isinstance(arguments, dict) else {},
                    }
                )
            anth_content = blocks
        normalized.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": anth_content,
            }
        )
    return "\n\n".join(system_parts), normalized


def _to_anthropic_tools(tools) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in tools or []:
        tool = dict(item or {})
        function = dict(tool.get("function") or {})
        if function:
            normalized.append(
                {
                    "name": str(function.get("name") or ""),
                    "description": str(function.get("description") or ""),
                    "input_schema": dict(function.get("parameters") or {"type": "object"}),
                }
            )
            continue
        normalized.append(
            {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "input_schema": dict(
                    tool.get("input_schema")
                    or tool.get("parameters")
                    or {"type": "object"}
                ),
            }
        )
    return normalized


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def _anthropic_text(response: Any) -> str:
    parts: List[str] = []
    for block in list(getattr(response, "content", []) or []):
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
    return "".join(parts)


def _anthropic_tool_calls(response: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for block in list(getattr(response, "content", []) or []):
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "tool_use":
            continue
        tool_id = getattr(block, "id", None)
        name = getattr(block, "name", None)
        arguments = getattr(block, "input", None)
        if isinstance(block, dict):
            tool_id = tool_id or block.get("id")
            name = name or block.get("name")
            arguments = arguments if arguments is not None else block.get(
                "input")
        calls.append(
            {
                "id": str(tool_id or ""),
                "type": "function",
                "function": {
                    "name": str(name or ""),
                    "arguments": json.dumps(arguments or {}, ensure_ascii=False),
                },
            }
        )
    return calls


def _openai_finish_reason(response: Any) -> str:
    choices = list(getattr(response, "choices", []) or [])
    if not choices:
        return ""
    return str(getattr(choices[0], "finish_reason", "") or "")


def _openai_responses_finish_reason(response: Any) -> str:
    status = str(getattr(response, "status", "") or "")
    if status != "incomplete":
        return status
    details = getattr(response, "incomplete_details", None)
    reason = _usage_detail_value(details, "reason")
    return str(reason or status)


def _usage_detail_value(details: Any, name: str) -> Any:
    if details is None:
        return None
    value = getattr(details, name, None)
    if value is None and isinstance(details, dict):
        value = details.get(name)
    return value


def _usage_to_dict(usage) -> dict:
    if not usage:
        return {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)
    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    # OpenAI-compatible providers expose reasoning tokens here.
    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = _usage_detail_value(
        completion_details,
        "reasoning_tokens",
    )
    if reasoning_tokens is not None:
        result["reasoning_tokens"] = reasoning_tokens
        result["reasoning_tokens_source"] = (
            "usage.completion_tokens_details.reasoning_tokens"
        )

    # MiniMax Anthropic-compatible responses expose the same concept as
    # output_tokens_details.thinking_tokens. Normalize it to reasoning_tokens
    # while retaining the provider-native field/source.
    output_details = getattr(usage, "output_tokens_details", None)
    output_reasoning_tokens = _usage_detail_value(
        output_details,
        "reasoning_tokens",
    )
    if output_reasoning_tokens is not None:
        result["reasoning_tokens"] = output_reasoning_tokens
        result["reasoning_tokens_source"] = (
            "usage.output_tokens_details.reasoning_tokens"
        )
    thinking_tokens = _usage_detail_value(output_details, "thinking_tokens")
    if thinking_tokens is not None:
        result["thinking_tokens"] = thinking_tokens
        result["thinking_tokens_source"] = (
            "usage.output_tokens_details.thinking_tokens"
        )
        if "reasoning_tokens" not in result:
            result["reasoning_tokens"] = thinking_tokens
            result["reasoning_tokens_source"] = (
                "usage.output_tokens_details.thinking_tokens"
            )
    for name in (
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, name, None)
        if value is not None:
            result[name] = value
    return result


def _response_thinking_metadata(
    response: Any,
    *,
    normalized_usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return observable thinking metadata without exposing thinking text.

    ``thinking_tokens=None`` means the response exposed thinking but did not
    expose a separate token count. Unknown must not be conflated with zero.
    """
    usage = dict(normalized_usage or {})
    thinking_tokens = usage.get("reasoning_tokens")
    thinking_tokens_source = usage.get("reasoning_tokens_source")

    # None means the response shape did not reveal whether thinking occurred.
    # False is reserved for an exposed reasoning channel with no thinking.
    thinking_present: Optional[bool] = None
    thinking_chars: Optional[int] = None

    # Anthropic-compatible shape: typed thinking/text content blocks.
    content_blocks = list(getattr(response, "content", []) or [])
    if content_blocks:
        thinking_present = False
        thinking_chars = 0
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type != "thinking":
                continue
            thinking = getattr(block, "thinking", None)
            if thinking is None and isinstance(block, dict):
                thinking = block.get("thinking")
            thinking_text = str(thinking or "")
            if thinking_text.strip():
                thinking_present = True
                thinking_chars += len(thinking_text)

    # OpenAI-compatible shape: reasoning fields live on the assistant message.
    choices = list(getattr(response, "choices", []) or [])
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            sentinel = object()
            if isinstance(message, dict):
                reasoning_content = message.get("reasoning_content", sentinel)
                reasoning_details = message.get("reasoning_details", sentinel)
            else:
                reasoning_content = getattr(message, "reasoning_content", sentinel)
                reasoning_details = getattr(message, "reasoning_details", sentinel)

            openai_reasoning_exposed = (
                reasoning_content is not sentinel
                or reasoning_details is not sentinel
                or thinking_tokens is not None
            )
            if openai_reasoning_exposed and thinking_present is None:
                thinking_present = False
                thinking_chars = 0
            if reasoning_content is not sentinel and str(
                reasoning_content or ""
            ).strip():
                thinking_present = True
                thinking_chars = len(str(reasoning_content or ""))
            if reasoning_details is not sentinel and reasoning_details:
                thinking_present = True
                if thinking_chars is None:
                    thinking_chars = 0
                thinking_chars += len(
                    json.dumps(reasoning_details, ensure_ascii=False, default=str)
                )

    # Provider-reported token counts are definitive when present.
    if thinking_tokens is not None:
        if int(thinking_tokens) > 0:
            thinking_present = True
        elif thinking_present is None:
            thinking_present = False

    return {
        "thinking_present": thinking_present,
        "thinking_chars": thinking_chars,
        "thinking_tokens_estimated": (
            (thinking_chars + 3) // 4
            if thinking_chars is not None and thinking_chars > 0
            else 0
            if thinking_chars == 0
            else None
        ),
        "thinking_tokens": thinking_tokens,
        "thinking_tokens_source": thinking_tokens_source,
    }
