"""Tests for US-006: lifespan shutdown sweep + grace-period auto-release
sweeper + manual ``/api/session/{id}/background`` route + ``migrating``
event emission.

These tests target ``release_session_to_background`` on SessionManager and
the grace-period sweep loop predicate. The persistence store is replaced
with an ``AsyncMock`` because Mongo isn't available here. We never spin up
a real agent runtime — only the manager / route boundary is exercised.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.session import Event  # noqa: E402
from session_manager import AgentSession, SessionManager  # noqa: E402


# ── Fixtures / helpers ────────────────────────────────────────────────────


class FakeRuntimeSession:
    def __init__(self, *, model: str = "test-model"):
        self.context_manager = SimpleNamespace(items=[])
        self.pending_approval = None
        self.turn_count = 0
        self.config = SimpleNamespace(model_name=model)
        self.notification_destinations = []
        self.is_running = True
        self.is_in_tool_call = False
        self.send_event = AsyncMock(return_value=None)


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
    manager._grace_sweep_task = None
    manager._idle_eviction_task = None
    manager._subscriber_counts = {}
    manager._no_subscriber_since = {}
    return manager


def _make_enabled_mock_store() -> AsyncMock:
    store = AsyncMock()
    store.enabled = True
    store.release_lease = AsyncMock(return_value=None)
    store.requeue_claimed_for = AsyncMock(return_value=0)
    store.poll_pending_submissions_after = AsyncMock(return_value=[])
    return store


def _agent_session_with_task(
    session_id: str = "s1",
    *,
    holder_id: str | None = None,
    with_task: bool = True,
) -> AgentSession:
    runtime = FakeRuntimeSession()
    task: asyncio.Task | None = None
    if with_task:
        # Use an unresolved Future under a Task wrapper so the task stays
        # pending without ever calling asyncio.sleep — tests that patch
        # asyncio.sleep to drive a sweep loop must not see this task as a
        # sleeping interleaver.
        loop = asyncio.get_event_loop()
        pending_future: asyncio.Future = loop.create_future()

        async def _wait_forever() -> None:
            try:
                await pending_future
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(_wait_forever())
    return AgentSession(
        session_id=session_id,
        session=runtime,  # type: ignore[arg-type]
        tool_router=object(),  # type: ignore[arg-type]
        user_id="dev",
        task=task,
        holder_id=holder_id,
    )


# ── A. release_session_to_background ──────────────────────────────────────


async def test_release_session_to_background_emits_migrating_event():
    """Emits a ``migrating`` event with the supplied reason, then requeues
    and releases."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    agent_session = _agent_session_with_task("s1", holder_id=manager._holder_id)
    manager.sessions["s1"] = agent_session

    try:
        result = await manager.release_session_to_background(
            "s1", reason="manual"
        )
    finally:
        if not agent_session.task.done():  # type: ignore[union-attr]
            agent_session.task.cancel()  # type: ignore[union-attr]

    assert result is True
    agent_session.session.send_event.assert_awaited_once()
    sent_event: Event = agent_session.session.send_event.await_args.args[0]
    assert sent_event.event_type == "migrating"
    assert sent_event.data == {"reason": "manual"}

    store.requeue_claimed_for.assert_awaited_once_with(manager._holder_id)
    store.release_lease.assert_awaited_once_with("s1", manager._holder_id)


async def test_release_session_to_background_drops_from_sessions():
    """The session is removed from the in-memory map and its task is cancelled."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    agent_session = _agent_session_with_task("s2", holder_id=manager._holder_id)
    manager.sessions["s2"] = agent_session
    task = agent_session.task

    result = await manager.release_session_to_background("s2", reason="manual")

    assert result is True
    assert manager.sessions.get("s2") is None
    # The release call invokes ``task.cancel()`` but doesn't await — so the
    # task may still be in the cancelling state. Drain to confirm it lands.
    assert task is not None
    try:
        await task  # type: ignore[misc]
    except asyncio.CancelledError:
        pass
    assert task.cancelled() or task.done()  # type: ignore[union-attr]


async def test_release_session_to_background_idempotent_on_unknown_id():
    """Returns False for an unknown session_id without raising."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    result = await manager.release_session_to_background(
        "does-not-exist", reason="manual"
    )

    assert result is False
    store.requeue_claimed_for.assert_not_awaited()
    store.release_lease.assert_not_awaited()


# ── B. Grace-period sweep predicate ──────────────────────────────────────


async def _drive_grace_sweep_one_tick(manager: SessionManager) -> None:
    """Run the grace sweep loop just long enough for one iteration to land.

    First sleep returns immediately; second cancels the task so we exit
    cleanly. Mirrors ``_drive_heartbeat_one_tick`` in the lease test file.
    """
    import session_manager as sm  # type: ignore

    original_sleep = sm.asyncio.sleep
    call_count = {"n": 0}

    async def fake_sleep(delay: float) -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise asyncio.CancelledError
        await original_sleep(0)

    sm.asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        task = asyncio.create_task(manager._grace_period_sweep_loop())
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sm.asyncio.sleep = original_sleep  # type: ignore[assignment]


async def test_grace_sweep_skips_when_subscribers_present():
    """If ``_no_subscriber_since[sid]`` is None (subscriber attached), the
    sweep does NOT release."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    agent_session = _agent_session_with_task("s-sub", holder_id=manager._holder_id)
    agent_session.is_processing = True
    manager.sessions["s-sub"] = agent_session
    # Subscriber currently attached → no zero-point recorded.
    manager._subscriber_counts["s-sub"] = 1

    manager.release_session_to_background = AsyncMock(return_value=True)  # type: ignore[method-assign]

    try:
        await _drive_grace_sweep_one_tick(manager)
    finally:
        if not agent_session.task.done():  # type: ignore[union-attr]
            agent_session.task.cancel()  # type: ignore[union-attr]
            try:
                await agent_session.task  # type: ignore[misc]
            except asyncio.CancelledError:
                pass

    manager.release_session_to_background.assert_not_awaited()


async def test_grace_sweep_releases_after_grace_period(monkeypatch):
    """Subscribers absent for > grace period AND session has work → release."""
    monkeypatch.setenv("GRACE_PERIOD_SECONDS", "180")

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    agent_session = _agent_session_with_task("s-old", holder_id=manager._holder_id)
    agent_session.is_processing = True
    manager.sessions["s-old"] = agent_session
    # Last subscriber detached 200s ago — past the 180s default.
    manager._no_subscriber_since["s-old"] = time.time() - 200

    manager.release_session_to_background = AsyncMock(return_value=True)  # type: ignore[method-assign]

    try:
        await _drive_grace_sweep_one_tick(manager)
    finally:
        if not agent_session.task.done():  # type: ignore[union-attr]
            agent_session.task.cancel()  # type: ignore[union-attr]
            try:
                await agent_session.task  # type: ignore[misc]
            except asyncio.CancelledError:
                pass

    manager.release_session_to_background.assert_awaited_once()
    args, kwargs = manager.release_session_to_background.await_args
    assert args[0] == "s-old"
    assert kwargs.get("reason") == "grace_period_elapsed"


async def test_grace_sweep_doesnt_release_idle_no_work_session(monkeypatch):
    """Grace expired but session has no in-flight work and no pending
    submissions → do NOT release."""
    monkeypatch.setenv("GRACE_PERIOD_SECONDS", "180")

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.poll_pending_submissions_after = AsyncMock(return_value=[])
    manager.persistence_store = store

    agent_session = _agent_session_with_task("s-idle", holder_id=manager._holder_id)
    agent_session.is_processing = False
    agent_session.session.is_in_tool_call = False  # type: ignore[attr-defined]
    manager.sessions["s-idle"] = agent_session
    manager._no_subscriber_since["s-idle"] = time.time() - 200

    manager.release_session_to_background = AsyncMock(return_value=True)  # type: ignore[method-assign]

    try:
        await _drive_grace_sweep_one_tick(manager)
    finally:
        if not agent_session.task.done():  # type: ignore[union-attr]
            agent_session.task.cancel()  # type: ignore[union-attr]
            try:
                await agent_session.task  # type: ignore[misc]
            except asyncio.CancelledError:
                pass

    manager.release_session_to_background.assert_not_awaited()


# ── C. /api/session/{id}/background route ────────────────────────────────


async def test_background_route_returns_404_unknown_session(monkeypatch):
    """When release_session_to_background returns False, the route raises 404."""
    from fastapi import HTTPException

    from routes import agent as agent_routes

    async def fake_check(session_id: str, user: dict, request: Any = None):
        return SimpleNamespace(session_id=session_id, is_active=True, user_id="dev")

    monkeypatch.setattr(agent_routes, "_check_session_access", fake_check)
    monkeypatch.setattr(
        agent_routes.session_manager,
        "release_session_to_background",
        AsyncMock(return_value=False),
    )

    with pytest.raises(HTTPException) as exc_info:
        await agent_routes.background_session(
            "missing", user={"user_id": "dev"}
        )
    assert exc_info.value.status_code == 404


async def test_background_route_returns_200_on_success(monkeypatch):
    """When release_session_to_background returns True, the route returns
    a ``{"status": "released", ...}`` dict."""
    from routes import agent as agent_routes

    async def fake_check(session_id: str, user: dict, request: Any = None):
        return SimpleNamespace(session_id=session_id, is_active=True, user_id="dev")

    release_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(agent_routes, "_check_session_access", fake_check)
    monkeypatch.setattr(
        agent_routes.session_manager,
        "release_session_to_background",
        release_mock,
    )

    result = await agent_routes.background_session(
        "s-ok", user={"user_id": "dev"}
    )
    assert result == {"status": "released", "session_id": "s-ok"}
    release_mock.assert_awaited_once_with("s-ok", reason="user_requested")
