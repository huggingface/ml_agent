"""Helpers for preparing chat history for LLM provider APIs."""

from typing import Any


_INTERNAL_MESSAGE_FIELDS = {
    "timestamp",
}


def to_provider_messages(messages: list[Any]) -> list[Any]:
    """Return message payloads with local-only metadata removed.

    Conversation history may carry fields that are useful for persistence and
    trace rendering, but OpenAI-compatible providers reject unknown top-level
    chat message keys. Keep the stored history untouched and strip those fields
    only for outbound LLM calls.
    """
    return [_to_provider_message(message) for message in messages]


def _to_provider_message(message: Any) -> Any:
    if isinstance(message, dict):
        payload = dict(message)
    elif hasattr(message, "model_dump"):
        payload = message.model_dump(mode="json", exclude_none=True)
    else:
        return message

    for field in _INTERNAL_MESSAGE_FIELDS:
        payload.pop(field, None)

    return payload
