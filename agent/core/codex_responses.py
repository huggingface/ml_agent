"""ChatGPT Codex Responses adapter.

The normal OpenAI/GPT path goes through LiteLLM and the public API. Codex
OAuth does not: Pi sends OAuth bearer tokens to ChatGPT's
``/backend-api/codex/responses`` endpoint. This module keeps that provider
path explicit so the rest of the agent loop can keep its LiteLLM-shaped
message/tool handling.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from agent.core import telemetry
from agent.core.codex_oauth import (
    codex_request_headers,
    get_codex_credentials_for_user,
)
from agent.core.llm_params import UnsupportedEffortError
from agent.core.session import Event

CODEX_MODEL_PREFIX = "openai-codex/"
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
_CODEX_PROVIDER_MARKER = "_ml_intern_provider"
_CODEX_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
_MAX_RETRIES = 3
_BASE_RETRY_DELAY_SECONDS = 1


class CodexAuthRequiredError(RuntimeError):
    """Raised when a session selects openai-codex without OAuth credentials."""


class CodexAPIError(RuntimeError):
    """Raised for ChatGPT Codex response errors."""


@dataclass
class CodexCompletionResult:
    content: str | None
    tool_calls_acc: dict[int, dict]
    token_count: int
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)


def is_openai_codex_model(model_name: str | None) -> bool:
    return bool(model_name and model_name.startswith(CODEX_MODEL_PREFIX))


def codex_model_id(model_name: str) -> str:
    if not is_openai_codex_model(model_name):
        raise ValueError(f"Not an OpenAI Codex OAuth model: {model_name}")
    model_id = model_name.removeprefix(CODEX_MODEL_PREFIX).strip()
    if not model_id:
        raise ValueError("OpenAI Codex model id is empty")
    return model_id


def is_codex_llm_params(params: dict[str, Any]) -> bool:
    return params.get(_CODEX_PROVIDER_MARKER) == "openai-codex"


def _normalize_reasoning_effort(
    reasoning_effort: str | None,
    *,
    strict: bool,
) -> str | None:
    if not reasoning_effort:
        return None
    if reasoning_effort == "minimal":
        return "low"
    if reasoning_effort == "max":
        if strict:
            raise UnsupportedEffortError("OpenAI Codex doesn't accept effort='max'")
        return "xhigh"
    if reasoning_effort in _CODEX_EFFORTS:
        return None if reasoning_effort == "none" else reasoning_effort
    if strict:
        raise UnsupportedEffortError(
            f"OpenAI Codex doesn't accept effort={reasoning_effort!r}"
        )
    return None


async def resolve_codex_llm_params(
    model_name: str,
    *,
    user_id: str | None,
    reasoning_effort: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    credentials = await get_codex_credentials_for_user(user_id)
    if credentials is None:
        raise CodexAuthRequiredError(
            "Connect a ChatGPT Plus/Pro account before using the Codex subscription model."
        )
    return {
        _CODEX_PROVIDER_MARKER: "openai-codex",
        "model": model_name,
        "codex_model": codex_model_id(model_name),
        "credentials": credentials,
        "reasoning_effort": _normalize_reasoning_effort(
            reasoning_effort,
            strict=strict,
        ),
    }


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        try:
            return message.model_dump(mode="json")
        except TypeError:
            return message.model_dump()
    data: dict[str, Any] = {}
    for key in ("role", "content", "tool_calls", "tool_call_id", "name"):
        value = getattr(message, key, None)
        if value is not None:
            data[key] = value
    return data


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _content_parts_for_user(content: Any) -> list[dict[str, str]]:
    text = _flatten_content(content)
    if not text:
        return []
    return [{"type": "input_text", "text": text}]


def _assistant_text_item(text: str, index: int) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        "status": "completed",
        "id": f"msg_{index}",
    }


def _tool_call_dict(tool_call: Any) -> dict[str, Any] | None:
    if hasattr(tool_call, "model_dump"):
        try:
            data = tool_call.model_dump(mode="json")
        except TypeError:
            data = tool_call.model_dump()
    elif isinstance(tool_call, dict):
        data = tool_call
    else:
        data = {
            "id": getattr(tool_call, "id", None),
            "function": getattr(tool_call, "function", None),
        }
    function = data.get("function") or {}
    if not isinstance(function, dict):
        function = {
            "name": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", None),
        }
    call_id = str(data.get("id") or "").strip()
    name = str(function.get("name") or "").strip()
    arguments = function.get("arguments")
    if not call_id or not name:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {})
    normalized_id = call_id.split("|", 1)
    item_id = normalized_id[1] if len(normalized_id) == 2 else None
    item: dict[str, Any] = {
        "type": "function_call",
        "call_id": normalized_id[0],
        "name": name,
        "arguments": arguments,
    }
    if item_id:
        item["id"] = item_id
    return item


def convert_messages_for_codex(messages: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    msg_index = 0

    for raw_message in messages:
        message = _message_to_dict(raw_message)
        role = message.get("role")
        if role in {"system", "developer"}:
            text = _flatten_content(message.get("content")).strip()
            if text:
                instructions.append(text)
            continue

        if role == "user":
            content = _content_parts_for_user(message.get("content"))
            if content:
                input_items.append({"role": "user", "content": content})
        elif role == "assistant":
            text = _flatten_content(message.get("content")).strip()
            if text:
                input_items.append(_assistant_text_item(text, msg_index))
            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                item = _tool_call_dict(tool_call)
                if item:
                    input_items.append(item)
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "").split("|", 1)[0]
            if call_id:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _flatten_content(message.get("content")),
                    }
                )
        msg_index += 1

    return "\n\n".join(instructions) or "You are a helpful assistant.", input_items


def convert_tools_for_codex(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            function = tool if isinstance(tool, dict) else {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object"},
                "strict": None,
            }
        )
    return converted


def _codex_url(base_url: str = CODEX_BASE_URL) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _request_body(
    *,
    session_id: str | None,
    messages: list[Any],
    tools: list[dict[str, Any]] | None,
    llm_params: dict[str, Any],
) -> dict[str, Any]:
    instructions, input_items = convert_messages_for_codex(messages)
    body: dict[str, Any] = {
        "model": llm_params["codex_model"],
        "store": False,
        "stream": True,
        "instructions": instructions,
        "input": input_items,
        "text": {"verbosity": "low"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": session_id,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    converted_tools = convert_tools_for_codex(tools)
    if converted_tools:
        body["tools"] = converted_tools
    if llm_params.get("reasoning_effort"):
        body["reasoning"] = {
            "effort": llm_params["reasoning_effort"],
            "summary": "auto",
        }
    return body


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    data_lines: list[str] = []
    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if line == "":
            if data_lines:
                data = "\n".join(data_lines).strip()
                data_lines = []
                if data and data != "[DONE]":
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise CodexAPIError(f"Invalid Codex SSE JSON: {exc}") from exc
                    if isinstance(parsed, dict):
                        yield parsed
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        data = "\n".join(data_lines).strip()
        if data and data != "[DONE]":
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                yield parsed


def _usage_from_codex(response: dict[str, Any] | None) -> dict[str, Any]:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    details = usage.get("input_tokens_details")
    cached_tokens = (
        int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    )
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


def _finish_reason(
    response: dict[str, Any] | None, tool_calls_acc: dict[int, dict]
) -> str:
    status = response.get("status") if isinstance(response, dict) else None
    if status == "incomplete":
        return "length"
    if status in {"failed", "cancelled"}:
        return "error"
    if tool_calls_acc:
        return "tool_calls"
    return "stop"


def _response_error_message(status: int, text: str) -> str:
    try:
        parsed = json.loads(text)
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, dict):
            code = str(err.get("code") or err.get("type") or "")
            message = str(err.get("message") or "")
            if status == 429 or "usage_limit" in code or "rate_limit" in code:
                return message or "You have hit your ChatGPT usage limit."
            if message:
                return message
    except Exception:
        pass
    return text[:500] or f"Codex request failed with HTTP {status}"


def _is_retryable(status: int, text: str) -> bool:
    if status in {429, 500, 502, 503, 504}:
        return True
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "rate limit",
            "overloaded",
            "service unavailable",
            "connection refused",
        )
    )


async def _post_with_retries(
    client: httpx.AsyncClient,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            request = client.build_request(
                "POST",
                _codex_url(),
                headers=headers,
                json=body,
            )
            response = await client.send(request, stream=True)
            if response.is_success:
                return response
            text = await response.aread()
            error_text = text.decode("utf-8", errors="replace")
            await response.aclose()
            if attempt < _MAX_RETRIES and _is_retryable(
                response.status_code, error_text
            ):
                await _sleep_retry(attempt)
                continue
            raise CodexAPIError(
                _response_error_message(response.status_code, error_text)
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                await _sleep_retry(attempt)
                continue
            raise
    if last_error:
        raise last_error
    raise CodexAPIError("Codex request failed after retries")


async def _sleep_retry(attempt: int) -> None:
    import asyncio

    await asyncio.sleep(_BASE_RETRY_DELAY_SECONDS * (2**attempt))


def _ensure_tool_slot(
    tool_calls_acc: dict[int, dict],
    index: int,
) -> dict[str, Any]:
    if index not in tool_calls_acc:
        tool_calls_acc[index] = {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
    return tool_calls_acc[index]


async def call_codex_responses(
    session: Any,
    messages: list[Any],
    tools: list[dict[str, Any]] | None,
    llm_params: dict[str, Any],
    *,
    emit_events: bool,
) -> CodexCompletionResult:
    credentials = llm_params["credentials"]
    body = _request_body(
        session_id=getattr(session, "session_id", None),
        messages=messages,
        tools=tools,
        llm_params=llm_params,
    )
    headers = codex_request_headers(
        credentials,
        session_id=getattr(session, "session_id", None),
    )

    content_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    final_response: dict[str, Any] | None = None
    t_start = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        response = await _post_with_retries(client, headers=headers, body=body)
        try:
            async for event in _iter_sse_events(response):
                event_type = event.get("type")
                if event_type == "error":
                    message = event.get("message") or event.get("code") or "Codex error"
                    raise CodexAPIError(str(message))
                if event_type == "response.failed":
                    failed_response = event.get("response")
                    error = (
                        failed_response.get("error")
                        if isinstance(failed_response, dict)
                        else None
                    )
                    if isinstance(error, dict):
                        raise CodexAPIError(str(error.get("message") or error))
                    raise CodexAPIError("Codex response failed")
                if event_type in {
                    "response.completed",
                    "response.done",
                    "response.incomplete",
                }:
                    response_payload = event.get("response")
                    final_response = (
                        response_payload if isinstance(response_payload, dict) else {}
                    )
                    if event_type == "response.incomplete":
                        final_response["status"] = "incomplete"
                    break
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        content_parts.append(delta)
                        if emit_events:
                            await session.send_event(
                                Event(
                                    event_type="assistant_chunk",
                                    data={"content": delta},
                                )
                            )
                elif event_type == "response.output_item.added":
                    item = event.get("item")
                    if (
                        not isinstance(item, dict)
                        or item.get("type") != "function_call"
                    ):
                        continue
                    index = int(event.get("output_index") or len(tool_calls_acc))
                    slot = _ensure_tool_slot(tool_calls_acc, index)
                    call_id = str(item.get("call_id") or "")
                    item_id = str(item.get("id") or "")
                    slot["id"] = f"{call_id}|{item_id}" if item_id else call_id
                    slot["function"]["name"] = str(item.get("name") or "")
                    slot["function"]["arguments"] = str(item.get("arguments") or "")
                elif event_type == "response.function_call_arguments.delta":
                    index = int(event.get("output_index") or len(tool_calls_acc) - 1)
                    if index < 0:
                        continue
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        slot = _ensure_tool_slot(tool_calls_acc, index)
                        slot["function"]["arguments"] += delta
                elif event_type == "response.function_call_arguments.done":
                    index = int(event.get("output_index") or len(tool_calls_acc) - 1)
                    if index < 0:
                        continue
                    arguments = event.get("arguments")
                    if isinstance(arguments, str):
                        slot = _ensure_tool_slot(tool_calls_acc, index)
                        slot["function"]["arguments"] = arguments
                elif event_type == "response.output_item.done":
                    item = event.get("item")
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "function_call":
                        index = int(event.get("output_index") or len(tool_calls_acc))
                        slot = _ensure_tool_slot(tool_calls_acc, index)
                        call_id = str(item.get("call_id") or "")
                        item_id = str(item.get("id") or "")
                        slot["id"] = f"{call_id}|{item_id}" if item_id else call_id
                        slot["function"]["name"] = str(item.get("name") or "")
                        if isinstance(item.get("arguments"), str):
                            slot["function"]["arguments"] = str(item["arguments"])
                    elif item.get("type") == "message" and not content_parts:
                        for part in item.get("content") or []:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "output_text"
                            ):
                                text = part.get("text")
                                if isinstance(text, str):
                                    content_parts.append(text)
        finally:
            await response.aclose()

    usage = _usage_from_codex(final_response)
    finish_reason = _finish_reason(final_response, tool_calls_acc)
    await telemetry.record_llm_call(
        session,
        model=str(llm_params.get("model") or llm_params.get("codex_model")),
        response={"usage": usage},
        latency_ms=int((time.monotonic() - t_start) * 1000),
        finish_reason=finish_reason,
    )

    return CodexCompletionResult(
        content="".join(content_parts) or None,
        tool_calls_acc=tool_calls_acc,
        token_count=int(usage.get("total_tokens") or 0),
        finish_reason=finish_reason,
        usage=usage,
    )
