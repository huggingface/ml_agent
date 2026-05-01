"""Tests for SessionManager lease holder identity, heartbeat, and lease-loss
requeue wiring.

These tests target the boundaries added in US-002 + US-003:
* Holder identity at construction (mode + holder_id)
* ``claim_lease`` on session creation
* The lease-renewal heartbeat loop
* The lease-loss handler that requeues claimed submissions and drops the session
* ``release_lease`` in the ``_run_session`` finally path

The persistence store is replaced with an ``AsyncMock`` because Mongo isn't
available here. We never spin up a real agent runtime — only the manager
boundary is exercised.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from session_manager import AgentSession, SessionManager  # noqa: E402


# ── Fixtures / helpers ────────────────────────────────────────────────────


class FakeRuntimeSession:
    def __init__(self, *, hf_token: str | None = None, model: str = "test-model"):
        self.hf_token = hf_token
        self.context_manager = SimpleNamespace(items=[])
        self.pending_approval = None
        self.turn_count = 0
        self.config = SimpleNamespace(model_name=model)
        self.notification_destinations = []


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
    """An AsyncMock that satisfies SessionManager's lease surface."""
    store = AsyncMock()
    store.enabled = True
    store.claim_lease = AsyncMock(return_value={"lease": {"holder_id": "x"}})
    store.renew_lease = AsyncMock(return_value={"lease": {"holder_id": "x"}})
    store.release_lease = AsyncMock(return_value=None)
    store.requeue_claimed_for = AsyncMock(return_value=0)
    return store


def _runtime_agent_session(session_id: str = "s1") -> AgentSession:
    return AgentSession(
        session_id=session_id,
        session=FakeRuntimeSession(),  # type: ignore[arg-type]
        tool_router=object(),  # type: ignore[arg-type]
        user_id="dev",
    )


# ── A. Holder identity at construction ────────────────────────────────────


def test_session_manager_init_assigns_holder_id(monkeypatch):
    monkeypatch.delenv("MODE", raising=False)
    manager = SessionManager()
    assert manager.mode == "main"
    assert re.match(r"^(main|worker):.+:[0-9a-f]{8}$", manager._holder_id), (
        manager._holder_id
    )


def test_session_manager_invalid_mode_warns_and_defaults(monkeypatch, caplog):
    monkeypatch.setenv("MODE", "bogus")
    with caplog.at_level(logging.WARNING):
        manager = SessionManager()
    assert manager.mode == "main"
    assert manager._holder_id.startswith("main:")
    assert any("Unknown MODE" in record.message for record in caplog.records), (
        [r.message for r in caplog.records]
    )


def test_session_manager_worker_mode(monkeypatch):
    monkeypatch.setenv("MODE", "worker")
    manager = SessionManager()
    assert manager.mode == "worker"
    assert manager._holder_id.startswith("worker:")


# ── B. create_session claims the lease ────────────────────────────────────


async def test_create_session_claims_lease(monkeypatch):
    """create_session must call store.claim_lease with the holder_id and TTL."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    # Background task collaborators that we don't need to drive.
    manager.persist_session_snapshot = AsyncMock(return_value=None)  # type: ignore[method-assign]
    manager._track_pro_status = AsyncMock(return_value=None)  # type: ignore[method-assign]

    # Stub out runtime construction & task launch.
    fake_session = FakeRuntimeSession()

    def fake_create_session_sync(**kwargs: Any) -> tuple[Any, Any]:
        return object(), fake_session

    manager._create_session_sync = fake_create_session_sync  # type: ignore[method-assign]

    started_event = asyncio.Event()

    async def fake_start_agent_session(**kwargs: Any) -> AgentSession:
        agent_session = kwargs["agent_session"]
        manager.sessions[agent_session.session_id] = agent_session
        # Don't actually run the loop — just return.
        started_event.set()
        return agent_session

    manager._start_agent_session = fake_start_agent_session  # type: ignore[method-assign]

    sid = await manager.create_session(user_id="dev", model="test-model")
    assert isinstance(sid, str) and len(sid) > 0
    store.claim_lease.assert_awaited_once()
    args, kwargs = store.claim_lease.call_args
    # claim_lease(session_id, holder_id, ttl_s=30)
    assert args[0] == sid
    assert args[1] == manager._holder_id
    assert kwargs.get("ttl_s") == 30
    assert started_event.is_set()


async def test_create_session_raises_when_claim_lease_fails():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.claim_lease = AsyncMock(return_value=None)  # simulate a holder collision
    manager.persistence_store = store
    manager.persist_session_snapshot = AsyncMock(return_value=None)  # type: ignore[method-assign]

    def fake_create_session_sync(**kwargs: Any) -> tuple[Any, Any]:
        return object(), FakeRuntimeSession()

    manager._create_session_sync = fake_create_session_sync  # type: ignore[method-assign]

    async def fake_start_agent_session(**kwargs: Any) -> AgentSession:
        return kwargs["agent_session"]

    manager._start_agent_session = fake_start_agent_session  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Failed to claim lease"):
        await manager.create_session(user_id="dev", model="test-model")


async def test_create_session_skips_lease_check_for_noop_store():
    """When the store is disabled (Noop), a None return is acceptable."""
    manager = _bare_manager()
    manager.persistence_store = NoopSessionStore()
    manager.persist_session_snapshot = AsyncMock(return_value=None)  # type: ignore[method-assign]

    def fake_create_session_sync(**kwargs: Any) -> tuple[Any, Any]:
        return object(), FakeRuntimeSession()

    manager._create_session_sync = fake_create_session_sync  # type: ignore[method-assign]

    async def fake_start_agent_session(**kwargs: Any) -> AgentSession:
        manager.sessions[kwargs["agent_session"].session_id] = kwargs["agent_session"]
        return kwargs["agent_session"]

    manager._start_agent_session = fake_start_agent_session  # type: ignore[method-assign]

    sid = await manager.create_session(user_id="dev", model="test-model")
    assert sid in manager.sessions


# ── C. Heartbeat loop renews held sessions ────────────────────────────────


async def _drive_heartbeat_one_tick(manager: SessionManager) -> None:
    """Run the heartbeat loop just long enough for one tick to land.

    We rebind ``asyncio.sleep`` only for the loop's awaits by patching the
    module-level reference inside session_manager. The first sleep returns
    immediately; the second cancels the task so we exit cleanly.
    """
    import session_manager as sm  # type: ignore

    original_sleep = sm.asyncio.sleep
    call_count = {"n": 0}

    async def fake_sleep(delay: float) -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            # After the post-iteration sleep on the next loop-around, stop.
            raise asyncio.CancelledError
        # First call (the HEARTBEAT_INTERVAL_S sleep) — yield once and return.
        await original_sleep(0)

    sm.asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        task = asyncio.create_task(manager._lease_heartbeat_loop())
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sm.asyncio.sleep = original_sleep  # type: ignore[assignment]


async def test_heartbeat_loop_renews_held_sessions():
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store
    manager.sessions["sess-A"] = _runtime_agent_session("sess-A")
    manager.sessions["sess-B"] = _runtime_agent_session("sess-B")

    await _drive_heartbeat_one_tick(manager)

    # Both sessions should have been renewed at least once.
    renewed_session_ids = {
        call.args[0] for call in store.renew_lease.await_args_list
    }
    assert {"sess-A", "sess-B"}.issubset(renewed_session_ids)
    # Each renewal call uses our holder_id and ttl_s=30.
    for call in store.renew_lease.await_args_list:
        assert call.args[1] == manager._holder_id
        assert call.kwargs.get("ttl_s") == 30


async def test_heartbeat_skips_when_store_disabled():
    manager = _bare_manager()
    manager.persistence_store = NoopSessionStore()
    manager.sessions["sess-A"] = _runtime_agent_session("sess-A")

    await _drive_heartbeat_one_tick(manager)
    # No renew calls performed; session is untouched.
    assert "sess-A" in manager.sessions


# ── D. Lease loss triggers requeue + drop ─────────────────────────────────


async def test_heartbeat_loss_triggers_requeue_and_drop(caplog):
    manager = _bare_manager()
    store = _make_enabled_mock_store()

    async def fake_renew(session_id: str, holder_id: str, ttl_s: int = 30):
        if session_id == "sess-A":
            return None  # Lost it.
        return {"lease": {"holder_id": holder_id}}

    store.renew_lease = AsyncMock(side_effect=fake_renew)
    store.requeue_claimed_for = AsyncMock(return_value=2)
    manager.persistence_store = store
    manager.sessions["sess-A"] = _runtime_agent_session("sess-A")
    manager.sessions["sess-B"] = _runtime_agent_session("sess-B")

    with caplog.at_level(logging.WARNING):
        await _drive_heartbeat_one_tick(manager)

    store.requeue_claimed_for.assert_awaited()
    for call in store.requeue_claimed_for.await_args_list:
        assert call.args[0] == manager._holder_id
    assert "sess-A" not in manager.sessions
    assert "sess-B" in manager.sessions
    assert any("Lease lost" in r.message for r in caplog.records), (
        [r.message for r in caplog.records]
    )


async def test_on_lease_lost_handles_requeue_exception(caplog):
    """If requeue_claimed_for raises, the session is still dropped."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.requeue_claimed_for = AsyncMock(side_effect=RuntimeError("boom"))
    manager.persistence_store = store

    fake = _runtime_agent_session("sess-X")
    manager.sessions["sess-X"] = fake

    with caplog.at_level(logging.ERROR):
        await manager._on_lease_lost("sess-X")

    assert "sess-X" not in manager.sessions
    assert any("requeue_claimed_for failed" in r.message for r in caplog.records)


# ── E. release_lease in finally path ──────────────────────────────────────


async def test_run_session_releases_lease_in_finally():
    """``_run_session`` must call ``release_lease`` in its finally block."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    # Build an AgentSession that triggers the loop's exit immediately.
    runtime_session = SimpleNamespace(
        is_running=False,
        send_event=AsyncMock(return_value=None),
        config=SimpleNamespace(save_sessions=False, model_name="test-model"),
        cancel=lambda: None,
        notification_destinations=[],
    )
    # _run_session calls _cleanup_sandbox(session) so it must have getattr targets.
    runtime_session.sandbox = None  # type: ignore[attr-defined]

    event_queue: asyncio.Queue = asyncio.Queue()

    class FakeRouter:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    agent_session = AgentSession(
        session_id="sess-finally",
        session=runtime_session,  # type: ignore[arg-type]
        tool_router=FakeRouter(),  # type: ignore[arg-type]
        user_id="dev",
    )
    manager.sessions["sess-finally"] = agent_session
    # Avoid persisting snapshot in finally (touches Mongo path indirectly).
    manager.persist_session_snapshot = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await manager._run_session(
        "sess-finally",
        event_queue,
        agent_session.tool_router,  # type: ignore[arg-type]
    )

    store.release_lease.assert_awaited()
    args, kwargs = store.release_lease.call_args
    assert args[0] == "sess-finally"
    assert args[1] == manager._holder_id


# ── F. close() cancels heartbeat task ─────────────────────────────────────


async def test_close_cancels_heartbeat_task():
    manager = _bare_manager()
    manager.persistence_store = NoopSessionStore()
    manager.messaging_gateway = SimpleNamespace(close=AsyncMock(return_value=None))

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_loop():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    manager._heartbeat_task = asyncio.create_task(fake_loop())
    await started.wait()

    await manager.close()
    assert cancelled.is_set()
    assert manager._heartbeat_task is None
