from litellm import ChatCompletionMessageToolCall, Message

from agent.core.llm_messages import to_provider_messages


def test_to_provider_messages_strips_timestamp_without_mutating_history():
    message = Message(role="user", content="hello")
    message.timestamp = "2026-05-01T22:45:04.139632"

    provider_messages = to_provider_messages([message])

    assert provider_messages == [{"role": "user", "content": "hello"}]
    assert message.timestamp == "2026-05-01T22:45:04.139632"


def test_to_provider_messages_preserves_tool_call_payload():
    tool_call = ChatCompletionMessageToolCall(
        id="call_123",
        type="function",
        function={"name": "research", "arguments": "{}"},
    )
    message = Message(role="assistant", content=None, tool_calls=[tool_call])
    message.timestamp = "2026-05-01T22:45:04.139632"

    provider_message = to_provider_messages([message])[0]

    assert provider_message["role"] == "assistant"
    assert provider_message["tool_calls"] == [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "research", "arguments": "{}"},
        }
    ]
    assert "timestamp" not in provider_message
