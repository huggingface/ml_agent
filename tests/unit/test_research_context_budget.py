"""Tests for dynamic research sub-agent context budget.

Regression for the hard-coded 170k/190k budget that assumed a 200k context
window regardless of the actual research model. With claude-sonnet-4-6 having
a 1M context window, the sub-agent was being terminated at ~19% capacity.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.research_tool import _get_research_model, research_handler


# ── _get_research_model ────────────────────────────────────────────────


def test_anthropic_main_model_uses_sonnet_for_research():
    assert (
        _get_research_model("anthropic/claude-opus-4-7")
        == "anthropic/claude-sonnet-4-6"
    )
    assert (
        _get_research_model("anthropic/claude-opus-4-6")
        == "anthropic/claude-sonnet-4-6"
    )


def test_bedrock_anthropic_model_uses_bedrock_sonnet():
    result = _get_research_model("bedrock/us.anthropic.claude-opus-4-6-v1")
    assert result == "bedrock/us.anthropic.claude-sonnet-4-6"


def test_non_anthropic_model_falls_back_to_same_model():
    assert _get_research_model("openai/gpt-5.5") == "openai/gpt-5.5"
    assert _get_research_model("moonshotai/Kimi-K2.6") == "moonshotai/Kimi-K2.6"


# ── research_handler calls _get_max_tokens_safe ────────────────────────


@pytest.mark.asyncio
async def test_research_handler_calls_get_max_tokens_safe_with_research_model():
    """`_get_max_tokens_safe` must be called with the *research* model id,
    not the main model id, so the budget reflects the sub-agent's model."""
    fake_session = MagicMock()
    fake_session.config.model_name = "anthropic/claude-opus-4-7"
    fake_session.config.reasoning_effort = None
    fake_session.hf_token = None
    fake_session.tool_router.get_tool_specs_for_llm.return_value = []
    fake_session.send_event = AsyncMock()

    with (
        patch(
            "agent.tools.research_tool._get_max_tokens_safe", return_value=1_000_000
        ) as mock_ctx,
        patch(
            "agent.tools.research_tool.acompletion",
            side_effect=RuntimeError("abort"),
        ),
    ):
        await research_handler({"task": "test task"}, session=fake_session)

    # Must be called with the research model (sonnet), not the main model (opus)
    mock_ctx.assert_called_once_with("anthropic/claude-sonnet-4-6")
