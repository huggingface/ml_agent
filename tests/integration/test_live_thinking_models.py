"""Opt-in live provider checks for thinking metadata replay.

These tests intentionally call paid model APIs and are skipped unless
``ML_INTERN_LIVE_LLM_TESTS=1`` plus the relevant provider key are set.
They cover the concrete model families involved in #87 without making
default CI depend on external credentials or provider availability.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from litellm import Message

from agent.core.agent_loop import (
    _assistant_message_from_result,
    _call_llm_streaming,
)
from agent.core.llm_params import _resolve_llm_params


if env_file := os.environ.get("ML_INTERN_LIVE_ENV_FILE"):
    load_dotenv(Path(env_file))

LIVE_TESTS_ENABLED = os.environ.get("ML_INTERN_LIVE_LLM_TESTS") == "1"
OPUS_47_MODEL = "anthropic/claude-opus-4-7"
LATEST_GPT_MODEL = "openai/gpt-5.2"
REPORT_RESULT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "report_result",
            "description": "Report the final test result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The exact marker requested by the test.",
                    }
                },
                "required": ["answer"],
            },
        },
    }
]


def _skip_without_live_flag() -> None:
    if not LIVE_TESTS_ENABLED:
        pytest.skip("set ML_INTERN_LIVE_LLM_TESTS=1 to run paid live LLM tests")


def _skip_without_env(name: str) -> None:
    if not os.environ.get(name):
        pytest.skip(f"set {name} to run this live provider test")


def _session(model_name: str):
    events = []

    async def send_event(event):
        events.append(event)

    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name),
        is_cancelled=False,
        send_event=send_event,
        events=events,
    )


@pytest.mark.asyncio
async def test_live_opus_47_preserves_thinking_metadata_for_replay():
    _skip_without_live_flag()
    _skip_without_env("ANTHROPIC_API_KEY")

    session = _session(OPUS_47_MODEL)
    llm_params = _resolve_llm_params(
        OPUS_47_MODEL,
        reasoning_effort="high",
    )

    result = await _call_llm_streaming(
        session,
        messages=[
            Message(
                role="user",
                content=(
                    "Use careful reasoning for this small check. "
                    "If 17 * 19 = 323, call report_result with answer OPUS_OK."
                ),
            )
        ],
        tools=REPORT_RESULT_TOOL,
        llm_params=llm_params,
    )

    replay = _assistant_message_from_result(
        result,
        model_name=OPUS_47_MODEL,
    )

    assert result.content or result.tool_calls_acc
    assert result.thinking_blocks, (
        "Opus returned no thinking_blocks with reasoning_effort='high' - "
        "check that adaptive thinking params are being forwarded correctly"
    )
    assert getattr(replay, "thinking_blocks", None) == result.thinking_blocks
    assert getattr(replay, "reasoning_content", None) == result.reasoning_content


@pytest.mark.asyncio
async def test_live_latest_gpt_does_not_replay_reasoning_metadata():
    _skip_without_live_flag()
    _skip_without_env("OPENAI_API_KEY")

    session = _session(LATEST_GPT_MODEL)
    llm_params = _resolve_llm_params(
        LATEST_GPT_MODEL,
        reasoning_effort="low",
    )

    result = await _call_llm_streaming(
        session,
        messages=[
            Message(
                role="user",
                content="Call report_result with answer GPT_OK.",
            )
        ],
        tools=REPORT_RESULT_TOOL,
        llm_params=llm_params,
    )

    # Even if a GPT-family response carries provider reasoning internally,
    # OpenAI-compatible history must not echo it back on the next tool turn.
    # Force the non-None strip path when the live model omits reasoning details.
    result.reasoning_content = result.reasoning_content or "synthetic-reasoning"
    replay = _assistant_message_from_result(
        result,
        model_name=LATEST_GPT_MODEL,
    )

    assert result.content or result.tool_calls_acc
    assert getattr(replay, "thinking_blocks", None) is None
    assert getattr(replay, "reasoning_content", None) is None
