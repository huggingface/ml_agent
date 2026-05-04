"""Anthropic prompt caching breakpoints for outgoing LLM requests.

Caching is GA on Anthropic's API and natively supported by litellm >=1.83
via ``cache_control`` blocks. We apply two breakpoints (out of 4 allowed):

  1. The tool block — caches all tool definitions as a single prefix.
  2. The system message — caches the rendered system prompt.

Together these cover the ~4-5K static tokens that were being re-billed on
every turn. Subsequent turns within the 5-minute TTL hit cache_read pricing
(~10% of input cost) instead of full input.

Non-Anthropic models (HF router, OpenAI) are passed through unchanged.
"""

from typing import Any


def with_prompt_caching(
    messages: list[Any],
    tools: list[dict] | None,
    model_name: str | None,
) -> tuple[list[Any], list[dict] | None]:
    """Return (messages, tools) with cache_control breakpoints for Anthropic.

    No-op for non-Anthropic models. Original objects are not mutated; a fresh
    list with replaced first message and last tool is returned, so callers
    that share the underlying ``ContextManager.items`` list don't see their
    persisted history rewritten.
    """
    if not model_name or "anthropic" not in model_name:
        return messages, tools

    if tools:
        new_tools = list(tools)
        last = dict(new_tools[-1])
        last["cache_control"] = {"type": "ephemeral"}
        new_tools[-1] = last
        tools = new_tools

    if messages:
        first = messages[0]
        role = (
            first.get("role")
            if isinstance(first, dict)
            else getattr(first, "role", None)
        )
        if role == "system":
            content = (
                first.get("content")
                if isinstance(first, dict)
                else getattr(first, "content", None)
            )
            if isinstance(content, str) and content:
                cached_block = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                new_first = {"role": "system", "content": cached_block}
                messages = [new_first] + list(messages[1:])

    return messages, tools
