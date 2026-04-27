"""Regression tests for `_patch_dangling_tool_calls`.

Reproduces the failure mode behind observatory sessions 8dd2ce30 and
59c9e678 (2026-04-25): a tool call cancelled mid-execution leaves an
orphan ``tool_use`` in history; the user types a follow-up; Bedrock
rejects the next request with HTTP 400 ``messages.N: tool_use ids were
found without tool_result blocks immediately after``.
"""

from litellm import ChatCompletionMessageToolCall, Message

from agent.context_manager.manager import ContextManager


def _tool_call(call_id: str, name: str = "research") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function={"name": name, "arguments": "{}"},
    )


def _make_cm() -> ContextManager:
    cm = ContextManager.__new__(ContextManager)
    cm.system_prompt = "system"
    cm.model_max_tokens = 100_000
    cm.compact_size = 1_000
    cm.running_context_usage = 0
    cm.untouched_messages = 5
    cm.items = [Message(role="system", content="system")]
    return cm


def test_orphan_tool_use_followed_by_user_message_is_patched():
    cm = _make_cm()
    cm.items.extend([
        Message(role="user", content="Research X"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[_tool_call("call_abc", "research")],
        ),
        Message(role="user", content="??"),
    ])
    msgs = cm.get_messages()
    tool_msgs = [m for m in msgs if getattr(m, "role", None) == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_abc"
    assert "interrupted" in (tool_msgs[0].content or "").lower() or "not executed" in (tool_msgs[0].content or "").lower()


def test_no_orphan_means_no_stub():
    cm = _make_cm()
    cm.items.extend([
        Message(role="user", content="Research X"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[_tool_call("call_abc", "research")],
        ),
        Message(role="tool", content="ok", tool_call_id="call_abc", name="research"),
    ])
    cm.get_messages()
    tool_msgs = [m for m in cm.items if getattr(m, "role", None) == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "ok"


def test_multiple_dangling_tool_calls_in_one_assistant_message_are_all_patched():
    cm = _make_cm()
    cm.items.extend([
        Message(role="user", content="do two things"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                _tool_call("call_1", "research"),
                _tool_call("call_2", "bash"),
            ],
        ),
        Message(role="user", content="follow up"),
    ])
    cm.get_messages()
    tool_ids = {
        getattr(m, "tool_call_id", None)
        for m in cm.items
        if getattr(m, "role", None) == "tool"
    }
    assert tool_ids == {"call_1", "call_2"}


def test_orphan_in_earlier_turn_still_gets_patched():
    """Two-turn history where the FIRST turn was interrupted.

    Old patcher stopped at the first user msg encountered while scanning
    backwards, so this case never got fixed and Bedrock rejected.
    """
    cm = _make_cm()
    cm.items.extend([
        Message(role="user", content="turn 1"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[_tool_call("call_old", "research")],
        ),
        Message(role="user", content="turn 2 — please retry"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[_tool_call("call_new", "bash")],
        ),
        Message(role="tool", content="ok", tool_call_id="call_new", name="bash"),
    ])
    cm.get_messages()
    tool_ids = {
        getattr(m, "tool_call_id", None)
        for m in cm.items
        if getattr(m, "role", None) == "tool"
    }
    assert "call_old" in tool_ids
    assert "call_new" in tool_ids
