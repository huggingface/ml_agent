"""Regression test for the malformed-JSON loop in observatory session
7750e82f (2026-04-25): GLM-5.1 produced six consecutive ``write`` calls
whose ``arguments`` strings JSON-parse-failed (truncated mid-stream by
the provider). The soft retry hint didn't move the model. The detector
in ``_detect_repeated_malformed`` looks for the streak so the agent loop
can inject a hard system-prompt forcing a different strategy.
"""

from litellm import Message

from agent.core.agent_loop import _detect_repeated_malformed


def _malformed_tool_msg(name: str, call_id: str) -> Message:
    return Message(
        role="tool",
        content=(
            f"ERROR: Tool call to '{name}' had malformed JSON arguments and "
            f"was NOT executed. Retry with smaller content — for 'write', "
            f"split into multiple smaller writes using 'edit'."
        ),
        tool_call_id=call_id,
        name=name,
    )


def test_two_consecutive_malformed_same_tool_triggers():
    items = [
        Message(role="user", content="write a big plan"),
        Message(role="assistant", content=None),
        _malformed_tool_msg("write", "1"),
        Message(role="assistant", content=None),
        _malformed_tool_msg("write", "2"),
    ]
    assert _detect_repeated_malformed(items, threshold=2) == "write"


def test_one_malformed_does_not_trigger():
    items = [
        Message(role="user", content="write a plan"),
        Message(role="assistant", content=None),
        _malformed_tool_msg("write", "1"),
    ]
    assert _detect_repeated_malformed(items, threshold=2) is None


def test_two_malformed_different_tools_does_not_trigger():
    items = [
        Message(role="assistant", content=None),
        _malformed_tool_msg("write", "1"),
        Message(role="assistant", content=None),
        _malformed_tool_msg("bash", "2"),
    ]
    assert _detect_repeated_malformed(items, threshold=2) is None


def test_streak_broken_by_successful_tool_call_does_not_trigger():
    items = [
        Message(role="assistant", content=None),
        _malformed_tool_msg("write", "1"),
        Message(role="assistant", content=None),
        Message(role="tool", content="ok", tool_call_id="2", name="write"),
        Message(role="assistant", content=None),
        _malformed_tool_msg("write", "3"),
    ]
    assert _detect_repeated_malformed(items, threshold=2) is None
