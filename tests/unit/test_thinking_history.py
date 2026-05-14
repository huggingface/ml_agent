from types import SimpleNamespace

import pytest
from litellm import ChatCompletionMessageToolCall, Message

from agent.core import agent_loop
from agent.core.agent_loop import (
    LLMResult,
    _call_llm_streaming,
    _assistant_message_from_result,
    _extract_thinking_state,
)


def test_extract_thinking_state_from_litellm_message():
    message = Message(
        role="assistant",
        content="working",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    thinking_blocks, reasoning_content = _extract_thinking_state(message)

    assert thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert reasoning_content == "reasoned"


def test_extract_thinking_state_from_provider_fields():
    message = SimpleNamespace(
        provider_specific_fields={
            "thinking_blocks": [{"type": "thinking", "thinking": "reasoned"}],
            "reasoning_content": "reasoned",
        },
    )

    thinking_blocks, reasoning_content = _extract_thinking_state(message)

    assert thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert reasoning_content == "reasoned"


def test_assistant_message_from_result_preserves_thinking_with_tool_calls():
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "bash", "arguments": '{"command": "date"}'},
    )
    result = LLMResult(
        content=None,
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    message = _assistant_message_from_result(
        result,
        model_name="anthropic/claude-opus-4-6",
        tool_calls=[tool_call],
    )

    assert message.tool_calls == [tool_call]
    assert message.thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert message.reasoning_content == "reasoned"


def test_assistant_message_from_result_strips_non_anthropic_reasoning_content():
    result = LLMResult(
        content=None,
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
        reasoning_content="reasoned",
    )

    message = _assistant_message_from_result(
        result,
        model_name="openai/Qwen/Qwen3-Next-80B-A3B-Instruct",
    )

    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_assistant_message_from_result_omits_absent_thinking_fields():
    result = LLMResult(
        content="done",
        tool_calls_acc={},
        token_count=12,
        finish_reason="stop",
    )

    message = _assistant_message_from_result(
        result,
        model_name="anthropic/claude-opus-4-6",
    )

    assert message.content == "done"
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


@pytest.mark.asyncio
async def test_streaming_call_rebuilds_anthropic_thinking_state(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    def fake_chunk_builder(chunks, **_kwargs):
        assert len(chunks) == 2
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=Message(
                        role="assistant",
                        content="done",
                        thinking_blocks=[{"type": "thinking", "thinking": "reasoned"}],
                        reasoning_content="reasoned",
                    )
                )
            ]
        )

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4-6"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "stream_chunk_builder", fake_chunk_builder)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "anthropic/claude-opus-4-6"},
    )

    assert result.content == "done"
    assert result.thinking_blocks == [{"type": "thinking", "thinking": "reasoned"}]
    assert result.reasoning_content == "reasoned"


@pytest.mark.asyncio
async def test_streaming_call_rebuilds_anthropic_delta_thinking_state(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "reasoned",
                                "signature": "",
                            }
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "signed",
                            }
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    def fake_chunk_builder(chunks, **_kwargs):
        assert len(chunks) == 4
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=Message(
                        role="assistant",
                        content="done",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "reasoned",
                                "signature": "signed",
                            }
                        ],
                        reasoning_content="reasoned",
                    )
                )
            ]
        )

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4-7"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "stream_chunk_builder", fake_chunk_builder)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "anthropic/claude-opus-4-7"},
    )

    assert result.content == "done"
    assert result.thinking_blocks == [
        {"type": "thinking", "thinking": "reasoned", "signature": "signed"}
    ]
    assert result.reasoning_content == "reasoned"


@pytest.mark.asyncio
async def test_streaming_call_skips_chunk_rebuild_for_non_anthropic(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    def fail_chunk_builder(*_args, **_kwargs):
        raise AssertionError("stream_chunk_builder should not run")

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="openai/Qwen/Qwen3"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "stream_chunk_builder", fail_chunk_builder)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/Qwen/Qwen3"},
    )

    assert result.content == "done"
    assert result.thinking_blocks is None
    assert result.reasoning_content is None


@pytest.mark.asyncio
async def test_streaming_retry_resets_after_emitted_assistant_chunk(monkeypatch):
    calls = 0

    async def failing_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="stale", tool_calls=None),
                    finish_reason=None,
                )
            ],
        )
        raise Exception("litellm.InternalServerError: AnthropicError - Overloaded")

    async def success_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="fresh", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return failing_stream()
        return success_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4-6"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "_retry_delay_with_jitter", lambda _delay: 0)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "anthropic/claude-opus-4-6"},
    )

    assert result.content == "fresh"
    assert calls == 2
    event_types = [event.event_type for event in events]
    assert event_types.count("assistant_stream_reset") == 1
    assert event_types.index("assistant_stream_reset") < event_types.index("tool_log")
    chunk_contents = [
        event.data["content"]
        for event in events
        if event.event_type == "assistant_chunk"
    ]
    assert chunk_contents == ["stale", "fresh"]

    reset = next(
        event for event in events if event.event_type == "assistant_stream_reset"
    )
    assert reset.data == {
        "attempt": 1,
        "next_attempt": 2,
        "max_attempts": agent_loop._MAX_LLM_RETRIES,
        "reason": "transient_error_retry",
        "delay_s": 0,
    }


@pytest.mark.asyncio
async def test_streaming_retry_does_not_reset_before_assistant_chunk(monkeypatch):
    calls = 0

    async def success_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="fresh", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )

    async def fake_acompletion(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Exception("litellm.InternalServerError: AnthropicError - Overloaded")
        return success_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4-6"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "_retry_delay_with_jitter", lambda _delay: 0)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "anthropic/claude-opus-4-6"},
    )

    assert result.content == "fresh"
    assert calls == 2
    assert "assistant_stream_reset" not in [event.event_type for event in events]
