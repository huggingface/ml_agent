"""Regression test for sandbox_create not surfacing the hardware lockout.

In observatory session d6f8454c (2026-04-25) the agent called
sandbox_create 18 times across 11 distinct hardware tiers (a10g-large,
a100-large, t4-small, cpu-upgrade, cpu-basic, zero-a10g, l4x1, t4-medium,
a10g-small, l40sx1, …). Every call returned 'Sandbox already active' for
the same sandbox, but the message did not say that hardware can't be
changed by re-calling, so the agent thought "still pending, retry with a
different flavor" and burned 17 useless turns.

The fix makes the response explicit when the requested hardware differs
from what's already active.
"""

import asyncio
from types import SimpleNamespace

from agent.tools.sandbox_tool import sandbox_create_handler


def _session_with_sandbox():
    sb = SimpleNamespace(
        space_id="user/sandbox-abc123",
        url="https://huggingface.co/spaces/user/sandbox-abc123",
    )
    return SimpleNamespace(sandbox=sb)


def test_already_active_with_different_hw_warns_about_lockout():
    session = _session_with_sandbox()
    out, ok = asyncio.run(
        sandbox_create_handler({"hardware": "a100-large"}, session=session)
    )
    assert ok is True
    # The message should mention the lockout AND the requested flavor.
    assert "cannot be changed" in out.lower()
    assert "a100-large" in out
    assert "delete" in out.lower()


def test_already_active_no_hw_request_just_returns_handle():
    session = _session_with_sandbox()
    out, ok = asyncio.run(sandbox_create_handler({}, session=session))
    assert ok is True
    assert "user/sandbox-abc123" in out
    # No spurious lockout note when the agent didn't request a flavor.
    assert "cannot be changed" not in out.lower()
