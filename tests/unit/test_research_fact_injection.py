"""Tests for N-turn goal-anchor injection in the research sub-agent.

Every ``_RESEARCH_FACT_INTERVAL`` iterations the loop appends a
``[SYSTEM: GOAL ANCHOR]`` user message restating the original task and any
thinking text the model produced alongside tool calls.  This keeps the task
visible near the end of the message list rather than buried under tool rounds.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from litellm.types.utils import ChatCompletionMessageToolCall, Function as LLMFunction

from agent.tools.research_tool import (
    _RESEARCH_FACT_INTERVAL,
    _RESEARCH_FACT_SUMMARY_MAX,
    _build_fact_anchor,
    _should_inject_fact,
    research_handler,
)


# ── helpers ───────────────────────────────────────────────────────────


def _tool_resp(content=None):
    tc = ChatCompletionMessageToolCall(
        id="tc_0",
        type="function",
        function=LLMFunction(name="bash", arguments='{"cmd": "ls"}'),
    )
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "tool_calls"
    usage = MagicMock()
    usage.total_tokens = 100
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _text_resp(content="Final summary."):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.total_tokens = 200
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class FakeConfig:
    model_name = "openai/test"
    reasoning_effort = None


class FakeToolRouter:
    def get_tool_specs_for_llm(self):
        return [
            {
                "type": "function",
                "function": {"name": "bash", "description": "run", "parameters": {}},
            }
        ]

    async def call_tool(self, name, args, session=None, tool_call_id=None):
        return "output", True


class FakeSession:
    def __init__(self):
        self.config = FakeConfig()
        self.hf_token = None
        self.tool_router = FakeToolRouter()

    async def send_event(self, _):
        pass


def _patch(monkeypatch, fake_acompletion, fake_doom=None):
    monkeypatch.setattr("agent.tools.research_tool.acompletion", fake_acompletion)
    monkeypatch.setattr(
        "agent.tools.research_tool.with_prompt_caching",
        lambda msgs, tools, model: (msgs, tools),
    )
    monkeypatch.setattr(
        "agent.tools.research_tool.check_for_doom_loop",
        fake_doom if fake_doom is not None else lambda _: None,
    )
    monkeypatch.setattr(
        "agent.tools.research_tool._resolve_llm_params",
        lambda *_, **__: {"model": "openai/test"},
    )
    monkeypatch.setattr(
        "agent.tools.research_tool.telemetry",
        MagicMock(record_llm_call=AsyncMock()),
    )


# ── _should_inject_fact ───────────────────────────────────────────────


def test_no_injection_at_zero():
    assert _should_inject_fact(0) is False


def test_no_injection_before_interval():
    for i in range(1, _RESEARCH_FACT_INTERVAL):
        assert _should_inject_fact(i) is False


def test_injection_at_interval():
    assert _should_inject_fact(_RESEARCH_FACT_INTERVAL) is True


def test_injection_repeats_at_multiples():
    assert _should_inject_fact(_RESEARCH_FACT_INTERVAL * 2) is True
    assert _should_inject_fact(_RESEARCH_FACT_INTERVAL * 3) is True


# ── _build_fact_anchor ────────────────────────────────────────────────


def test_anchor_marker_present():
    assert "GOAL ANCHOR" in _build_fact_anchor("find LoRA recipe", "")


def test_anchor_contains_task_verbatim():
    task = "find optimal lr schedule for 7B fine-tuning"
    assert task in _build_fact_anchor(task, "")


def test_anchor_includes_progress():
    anchor = _build_fact_anchor("task", "DPO outperforms RLHF on alignment.")
    assert "Progress so far:" in anchor
    assert "DPO outperforms RLHF" in anchor


def test_anchor_omits_progress_when_empty():
    assert "Progress so far:" not in _build_fact_anchor("task", "")


def test_anchor_truncates_at_max():
    long = "x" * (_RESEARCH_FACT_SUMMARY_MAX + 50)
    anchor = _build_fact_anchor("task", long)
    assert "…" in anchor
    assert "x" * (_RESEARCH_FACT_SUMMARY_MAX + 1) not in anchor


def test_anchor_preserves_short_progress():
    short = "y" * (_RESEARCH_FACT_SUMMARY_MAX - 1)
    anchor = _build_fact_anchor("task", short)
    assert "…" not in anchor
    assert short in anchor


# ── integration: anchor in messages passed to the LLM ────────────────


@pytest.mark.asyncio
async def test_anchor_injected_at_iteration_n(monkeypatch):
    task = "find best LoRA training recipe for LLaMA-3"
    n = _RESEARCH_FACT_INTERVAL
    call_no = 0
    captured = None

    async def fake_llm(messages, **kw):
        nonlocal call_no, captured
        if call_no == n:
            captured = list(messages)
            return _text_resp()
        call_no += 1
        return _tool_resp()

    _patch(monkeypatch, fake_llm)
    result, ok = await research_handler({"task": task}, session=FakeSession())

    assert ok
    anchors = [m for m in captured if "GOAL ANCHOR" in str(getattr(m, "content", ""))]
    assert len(anchors) == 1
    assert task in anchors[0].content
    assert "Progress so far:" not in anchors[0].content  # no thinking text was emitted


@pytest.mark.asyncio
async def test_no_anchor_before_interval(monkeypatch):
    task = "compare RLHF vs DPO"
    call_no = 0
    captured = None

    async def fake_llm(messages, **kw):
        nonlocal call_no, captured
        if call_no == _RESEARCH_FACT_INTERVAL - 1:
            captured = list(messages)
            return _text_resp()
        call_no += 1
        return _tool_resp()

    _patch(monkeypatch, fake_llm)
    await research_handler({"task": task}, session=FakeSession())

    assert captured is not None
    assert not any("GOAL ANCHOR" in str(getattr(m, "content", "")) for m in captured)


@pytest.mark.asyncio
async def test_second_cycle_injects_again(monkeypatch):
    """Anchor fires at 2N as well as N — the mechanism is truly periodic."""
    task = "evaluate dataset mixing strategies"
    n = _RESEARCH_FACT_INTERVAL
    call_no = 0
    captured_2n = None

    async def fake_llm(messages, **kw):
        nonlocal call_no, captured_2n
        if call_no == n * 2:
            captured_2n = list(messages)
            return _text_resp()
        call_no += 1
        return _tool_resp()

    _patch(monkeypatch, fake_llm)
    await research_handler({"task": task}, session=FakeSession())

    anchors = [
        m for m in captured_2n if "GOAL ANCHOR" in str(getattr(m, "content", ""))
    ]
    assert len(anchors) == 2, "Expected anchors at both N and 2N"


@pytest.mark.asyncio
async def test_thinking_text_appears_in_anchor(monkeypatch):
    """Thinking text the model emits alongside tool calls surfaces in the anchor."""
    task = "evaluate RLHF vs DPO for alignment"
    thinking = "DPO avoids the reward model and is cheaper to train."
    call_no = 0
    captured = None

    async def fake_llm(messages, **kw):
        nonlocal call_no, captured
        if call_no == _RESEARCH_FACT_INTERVAL:
            captured = list(messages)
            return _text_resp()
        content = thinking if call_no == 0 else None
        call_no += 1
        return _tool_resp(content=content)

    _patch(monkeypatch, fake_llm)
    await research_handler({"task": task}, session=FakeSession())

    anchors = [m for m in captured if "GOAL ANCHOR" in str(getattr(m, "content", ""))]
    assert len(anchors) == 1
    assert "Progress so far:" in anchors[0].content
    assert thinking[:40] in anchors[0].content


@pytest.mark.asyncio
async def test_doom_loop_and_anchor_coexist(monkeypatch):
    """Doom-loop guard and goal anchor can both fire in the same iteration."""
    task = "find training recipe"
    call_no = 0
    captured = None
    doom_call_no = 0

    def fake_doom(messages):
        nonlocal doom_call_no
        doom_call_no += 1
        # Fire at the Nth iteration (doom is called once per iteration)
        if doom_call_no == _RESEARCH_FACT_INTERVAL + 1:
            return "[SYSTEM: REPETITION GUARD] You are stuck."
        return None

    async def fake_llm(messages, **kw):
        nonlocal call_no, captured
        if call_no == _RESEARCH_FACT_INTERVAL:
            captured = list(messages)
            return _text_resp()
        call_no += 1
        return _tool_resp()

    _patch(monkeypatch, fake_llm, fake_doom=fake_doom)
    await research_handler({"task": task}, session=FakeSession())

    has_doom = any(
        "REPETITION GUARD" in str(getattr(m, "content", "")) for m in captured
    )
    has_anchor = any("GOAL ANCHOR" in str(getattr(m, "content", "")) for m in captured)
    assert has_doom, "Doom-loop guard message missing"
    assert has_anchor, "Goal anchor missing"


# ── capability: anchor re-states task after context has drifted ───────


@pytest.mark.asyncio
async def test_anchor_freshens_buried_task(monkeypatch):
    """Core capability claim: after N tool calls the original task is re-stated
    near the end of the message list so the model sees it with full weight.

    Without injection the task only appears at message[1] — the very start of
    history.  With injection a verbatim copy also appears near the tail,
    immediately before the LLM call that must use the findings.
    """
    task = "find optimal batch size and lr schedule for 7B fine-tuning"
    n = _RESEARCH_FACT_INTERVAL
    call_no = 0
    snapshot = None

    async def fake_llm(messages, **kw):
        nonlocal call_no, snapshot
        if call_no == n:
            snapshot = list(messages)
            return _text_resp()
        call_no += 1
        return _tool_resp()

    _patch(monkeypatch, fake_llm)
    _, ok = await research_handler({"task": task}, session=FakeSession())
    assert ok
    assert snapshot is not None

    user_msgs = [
        (i, m) for i, m in enumerate(snapshot) if getattr(m, "role", None) == "user"
    ]
    task_mentions = [
        (i, m) for i, m in user_msgs if task in str(getattr(m, "content", ""))
    ]

    # Task appears at least twice: initial message + anchor
    assert len(task_mentions) >= 2, (
        "Task should appear in both the initial user message and the GOAL ANCHOR"
    )

    first_pos = task_mentions[0][0]
    anchor_pos = task_mentions[-1][0]

    # The anchor is not the initial message
    assert anchor_pos > first_pos

    # Between them there are N tool-result messages — the context has genuinely drifted
    between = snapshot[first_pos + 1 : anchor_pos]
    tool_results = [m for m in between if getattr(m, "role", None) == "tool"]
    assert len(tool_results) == n, (
        f"Expected {n} tool results between initial task and anchor, got {len(tool_results)}"
    )

    # Anchor is the last user message before the LLM call — maximum recency
    last_user_pos = max(i for i, m in user_msgs)
    assert anchor_pos == last_user_pos, "Anchor should be the most recent user message"
