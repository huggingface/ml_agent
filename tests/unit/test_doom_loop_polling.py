"""Regression test for doom-loop false-positive on legitimate polling.

Reproduces the failure mode in observatory sessions 40fcb414 ($32.59),
8e90352e ($62.63), and 403178bf ($5.71) on 2026-04-25: the agent polled a
long-running job with `bash sleep 300 && wc -l output` four times in a
row. The arguments were byte-identical, but the results moved (27210 →
36454 → 45770 → 55138 — actual progress). The detector hashed args only
and false-fired DOOM LOOP, which made the agent abandon perfectly valid
polling.

After the fix the signature includes the tool result hash, so identical
args + different results no longer trips the detector.
"""

from litellm import ChatCompletionMessageToolCall, Message

from agent.core.doom_loop import check_for_doom_loop


def _assistant(call_id: str, name: str, args: str) -> Message:
    return Message(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=call_id,
                type="function",
                function={"name": name, "arguments": args},
            )
        ],
    )


def _tool(call_id: str, name: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id, name=name)


_POLL_ARGS = '{"command": "sleep 300 && ls /app/images/ | wc -l"}'


def test_polling_with_progressing_results_does_not_fire():
    msgs = [
        Message(role="user", content="run the job"),
        _assistant("c1", "bash", _POLL_ARGS),
        _tool("c1", "bash", "27210"),
        _assistant("c2", "bash", _POLL_ARGS),
        _tool("c2", "bash", "36454"),
        _assistant("c3", "bash", _POLL_ARGS),
        _tool("c3", "bash", "45770"),
        _assistant("c4", "bash", _POLL_ARGS),
        _tool("c4", "bash", "55138"),
    ]
    assert check_for_doom_loop(msgs) is None


def test_truly_stuck_polling_with_identical_results_still_fires():
    """If the same poll returns the same number, the job is genuinely
    stuck and the detector SHOULD fire."""
    msgs = [
        _assistant("c1", "bash", _POLL_ARGS),
        _tool("c1", "bash", "55138"),
        _assistant("c2", "bash", _POLL_ARGS),
        _tool("c2", "bash", "55138"),
        _assistant("c3", "bash", _POLL_ARGS),
        _tool("c3", "bash", "55138"),
    ]
    prompt = check_for_doom_loop(msgs)
    assert prompt is not None
    assert "DOOM LOOP" in prompt
    assert "bash" in prompt


def test_identical_calls_with_no_results_yet_still_fires():
    """If three identical calls have no tool results (e.g. all cancelled
    or errored before a result was recorded), treat as a real loop."""
    msgs = [
        _assistant("c1", "write", '{"path": "/tmp/x", "content": "..."}'),
        _assistant("c2", "write", '{"path": "/tmp/x", "content": "..."}'),
        _assistant("c3", "write", '{"path": "/tmp/x", "content": "..."}'),
    ]
    prompt = check_for_doom_loop(msgs)
    assert prompt is not None
    assert "DOOM LOOP" in prompt
    assert "write" in prompt


def test_different_args_does_not_fire():
    msgs = [
        _assistant("c1", "bash", '{"command": "ls /a"}'),
        _tool("c1", "bash", "ok"),
        _assistant("c2", "bash", '{"command": "ls /b"}'),
        _tool("c2", "bash", "ok"),
        _assistant("c3", "bash", '{"command": "ls /c"}'),
        _tool("c3", "bash", "ok"),
    ]
    assert check_for_doom_loop(msgs) is None
