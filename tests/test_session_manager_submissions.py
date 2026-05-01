"""Tests for the Mongo-backed pending-submissions consumer in SessionManager.

Covers US-004 (Plan Step 3):
* All ``submit*`` methods enqueue via the persistence store rather than an
  in-memory queue.
* ``interrupt`` branches between holder fast-path (direct ``session.cancel``)
  and non-holder enqueue (``op_type="interrupt"``).
* ``_drain_and_process`` claims, dispatches, and marks submissions done in
  FIFO order — and handles ``interrupt`` / ``shutdown`` ops inline without
  going through ``process_submission``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.session import OpType  # noqa: E402
from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from session_manager import (  # noqa: E402
    AgentSession,
    Operation,
    SessionManager,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────


class FakeRuntimeSession:
    def __init__(self, *, hf_token: str | None = None, model: str = "test-model"):
        self.hf_token = hf_token
        self.context_manager = SimpleNamespace(items=[])
        self.pending_approval = None
        self.turn_count = 0
        self.config = SimpleNamespace(model_name=model)
        self.notification_destinations = []
        self.is_running = True
        self.cancel_called = False

    def cancel(self) -> None:
        self.cancel_called = True


def _bare_manager() -> SessionManager:
    """Skip ``__init__``'s expensive config load; install the bits we need."""
    manager = object.__new__(SessionManager)
    manager.config = SimpleNamespace(model_name="test-model")
    manager.sessions = {}
    manager._lock = asyncio.Lock()
    manager.persistence_store = None
    manager.mode = "main"
    manager._holder_id = "main:test-host:deadbeef"
    manager._heartbeat_task = None
    return manager


def _make_enabled_mock_store() -> AsyncMock:
    store = AsyncMock()
    store.enabled = True
    store.enqueue_pending_submission = AsyncMock(return_value="abc123")
    store.claim_pending_submission = AsyncMock(return_value=None)
    store.mark_submission_done = AsyncMock(return_value=None)
    store.load_session = AsyncMock(return_value={"_id": "s1"})
    return store


def _runtime_agent_session(session_id: str = "s1") -> AgentSession:
    return AgentSession(
        session_id=session_id,
        session=FakeRuntimeSession(),  # type: ignore[arg-type]
        tool_router=object(),  # type: ignore[arg-type]
        user_id="dev",
    )


# ── A. submit* enqueues the right op_type + payload ───────────────────────


@pytest.mark.asyncio
async def test_submit_user_input_enqueues_to_mongo():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    manager.sessions["s1"] = _runtime_agent_session("s1")

    ok = await manager.submit_user_input("s1", "hi there")

    assert ok is True
    store.enqueue_pending_submission.assert_awaited_once_with(
        "s1", op_type="user_input", payload={"text": "hi there"}
    )


@pytest.mark.asyncio
async def test_submit_approval_enqueues_correct_payload():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    manager.sessions["s1"] = _runtime_agent_session("s1")

    approvals = [{"tool_call_id": "tc1", "approved": True}]
    ok = await manager.submit_approval("s1", approvals)

    assert ok is True
    store.enqueue_pending_submission.assert_awaited_once_with(
        "s1", op_type="exec_approval", payload={"approvals": approvals}
    )


@pytest.mark.asyncio
async def test_submit_undo_enqueues_undo_op():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    manager.sessions["s1"] = _runtime_agent_session("s1")

    assert await manager.undo("s1") is True
    store.enqueue_pending_submission.assert_awaited_once_with(
        "s1", op_type="undo", payload={}
    )


@pytest.mark.asyncio
async def test_submit_compact_enqueues_compact_op():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    manager.sessions["s1"] = _runtime_agent_session("s1")

    assert await manager.compact("s1") is True
    store.enqueue_pending_submission.assert_awaited_once_with(
        "s1", op_type="compact", payload={}
    )


@pytest.mark.asyncio
async def test_submit_operation_translates_op_type_value():
    """The legacy ``submit(Operation)`` path must enqueue ``op_type.value``."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    manager.sessions["s1"] = _runtime_agent_session("s1")

    op = Operation(op_type=OpType.USER_INPUT, data={"text": "hello"})
    ok = await manager.submit("s1", op)

    assert ok is True
    store.enqueue_pending_submission.assert_awaited_once_with(
        "s1", op_type="user_input", payload={"text": "hello"}
    )


@pytest.mark.asyncio
async def test_submit_falls_back_to_load_when_not_in_memory():
    """Cross-process case: session not in self.sessions but exists in Mongo."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.load_session = AsyncMock(return_value={"_id": "s1"})
    manager.persistence_store = store
    # NOT in manager.sessions

    ok = await manager.submit_user_input("s1", "hi")

    assert ok is True
    store.load_session.assert_awaited_once_with("s1")
    store.enqueue_pending_submission.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_returns_false_when_session_unknown():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.load_session = AsyncMock(return_value=None)
    manager.persistence_store = store

    ok = await manager.submit_user_input("ghost", "hi")

    assert ok is False
    store.enqueue_pending_submission.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_returns_false_when_store_disabled():
    """No Mongo + no in-memory session = nowhere to deliver. Return False."""
    manager = _bare_manager()
    manager.persistence_store = NoopSessionStore()

    ok = await manager.submit_user_input("nobody", "hi")
    assert ok is False


# ── B. interrupt branches on holder ───────────────────────────────────────


@pytest.mark.asyncio
async def test_interrupt_holder_fast_path():
    """When the session is held by us, interrupt cancels directly — no enqueue."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    agent_session = _runtime_agent_session("s1")
    manager.sessions["s1"] = agent_session

    ok = await manager.interrupt("s1")

    assert ok is True
    assert agent_session.session.cancel_called is True  # type: ignore[attr-defined]
    store.enqueue_pending_submission.assert_not_awaited()


@pytest.mark.asyncio
async def test_interrupt_non_holder_enqueues():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.load_session = AsyncMock(return_value={"_id": "s1"})
    manager.persistence_store = store
    # NOT in self.sessions

    ok = await manager.interrupt("s1")

    assert ok is True
    store.enqueue_pending_submission.assert_awaited_once_with(
        "s1", op_type="interrupt", payload={}
    )


@pytest.mark.asyncio
async def test_interrupt_unknown_session_returns_false():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.load_session = AsyncMock(return_value=None)
    manager.persistence_store = store

    ok = await manager.interrupt("ghost")

    assert ok is False
    store.enqueue_pending_submission.assert_not_awaited()


@pytest.mark.asyncio
async def test_interrupt_no_store_returns_false_when_not_local():
    manager = _bare_manager()
    manager.persistence_store = NoopSessionStore()

    ok = await manager.interrupt("ghost")
    assert ok is False


# ── C. _drain_and_process inline ops ──────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_and_process_handles_interrupt_op():
    """An ``interrupt`` op goes inline: cancel + mark_done; no agent-loop dispatch."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    agent_session = _runtime_agent_session("s1")
    manager.sessions["s1"] = agent_session

    # Feed exactly one interrupt doc, then None to terminate.
    store.claim_pending_submission = AsyncMock(
        side_effect=[
            {"_id": "sub-1", "op_type": "interrupt", "payload": {}},
            None,
        ]
    )

    await manager._drain_and_process(agent_session)

    assert agent_session.session.cancel_called is True  # type: ignore[attr-defined]
    store.mark_submission_done.assert_awaited_once_with("sub-1")


@pytest.mark.asyncio
async def test_drain_and_process_handles_shutdown_op():
    """A ``shutdown`` op flips ``is_running = False`` and stops draining."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    agent_session = _runtime_agent_session("s1")
    manager.sessions["s1"] = agent_session

    store.claim_pending_submission = AsyncMock(
        side_effect=[
            {"_id": "sub-1", "op_type": "shutdown", "payload": {}},
            None,
        ]
    )

    await manager._drain_and_process(agent_session)

    assert agent_session.session.is_running is False
    store.mark_submission_done.assert_awaited_once_with("sub-1")


@pytest.mark.asyncio
async def test_drain_and_process_fifo_dispatches_via_process_submission(
    monkeypatch,
):
    """Three queued user_input submissions are dispatched in arrival order."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    manager.persist_session_snapshot = AsyncMock(return_value=None)  # type: ignore[method-assign]

    agent_session = _runtime_agent_session("s1")
    manager.sessions["s1"] = agent_session

    docs = [
        {"_id": f"sub-{i}", "op_type": "user_input", "payload": {"text": f"msg-{i}"}}
        for i in range(3)
    ]
    store.claim_pending_submission = AsyncMock(side_effect=[*docs, None])
    store.mark_submission_done = AsyncMock(return_value=None)

    seen: list[str] = []

    async def fake_process_submission(session: Any, submission: Any) -> bool:
        # ``submission.operation.data`` is what the agent loop reads.
        seen.append(submission.operation.data["text"])
        return True

    import session_manager as sm  # type: ignore

    monkeypatch.setattr(sm, "process_submission", fake_process_submission)

    await manager._drain_and_process(agent_session)

    assert seen == ["msg-0", "msg-1", "msg-2"]
    assert store.mark_submission_done.await_count == 3
    # FIFO assertion: the marks happen in arrival order.
    marked_ids = [c.args[0] for c in store.mark_submission_done.await_args_list]
    assert marked_ids == ["sub-0", "sub-1", "sub-2"]


@pytest.mark.asyncio
async def test_drain_and_process_marks_done_even_on_handler_exception(
    monkeypatch,
):
    """A handler that raises must NOT redeliver the poison submission."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    manager.persist_session_snapshot = AsyncMock(return_value=None)  # type: ignore[method-assign]

    agent_session = _runtime_agent_session("s1")
    manager.sessions["s1"] = agent_session

    store.claim_pending_submission = AsyncMock(
        side_effect=[
            {"_id": "poison", "op_type": "user_input", "payload": {"text": "boom"}},
            None,
        ]
    )

    async def exploding_process(session: Any, submission: Any) -> bool:
        raise RuntimeError("boom")

    import session_manager as sm  # type: ignore

    monkeypatch.setattr(sm, "process_submission", exploding_process)

    await manager._drain_and_process(agent_session)

    store.mark_submission_done.assert_awaited_once_with("poison")


@pytest.mark.asyncio
async def test_drain_and_process_skips_when_store_disabled():
    """No Mongo → drain is a no-op, no claims attempted."""
    manager = _bare_manager()
    manager.persistence_store = NoopSessionStore()

    agent_session = _runtime_agent_session("s1")
    manager.sessions["s1"] = agent_session

    # Should return immediately without raising.
    await manager._drain_and_process(agent_session)


# ── D. _build_operation reconstructs Operation correctly ──────────────────


def test_build_operation_from_string():
    manager = _bare_manager()
    op = manager._build_operation("user_input", {"text": "hello"})
    assert op.op_type == OpType.USER_INPUT
    assert op.data == {"text": "hello"}


def test_build_operation_from_enum():
    manager = _bare_manager()
    op = manager._build_operation(OpType.COMPACT, {})
    assert op.op_type == OpType.COMPACT
    assert op.data is None  # empty payload normalises to None


def test_build_operation_unknown_op_type_raises():
    manager = _bare_manager()
    with pytest.raises(ValueError):
        manager._build_operation("totally_made_up", {})
