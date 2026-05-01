"""Tests for US-007: worker entrypoint, ``claim_dormant_session``, the
worker claim tick, and idle eviction.

The persistence store is replaced with an ``AsyncMock`` because Mongo
isn't available here. We never spin up a real agent runtime — only the
manager / module boundary is exercised.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

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
    store.poll_pending_submissions_after = AsyncMock(return_value=[])
    store.claim_lease = AsyncMock(return_value={"lease": {"holder_id": "x"}})
    store.load_session = AsyncMock(return_value=None)
    return store


def _agent_session_with_task(
    session_id: str = "s1",
    *,
    holder_id: str | None = None,
    last_submission_at: float = 0.0,
    is_processing: bool = False,
    is_in_tool_call: bool = False,
    with_task: bool = True,
) -> AgentSession:
    runtime = FakeRuntimeSession()
    runtime.is_in_tool_call = is_in_tool_call
    task: asyncio.Task | None = None
    if with_task:
        loop = asyncio.get_event_loop()
        pending_future: asyncio.Future = loop.create_future()

        async def _wait_forever() -> None:
            try:
                await pending_future
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(_wait_forever())
    ag = AgentSession(
        session_id=session_id,
        session=runtime,  # type: ignore[arg-type]
        tool_router=object(),  # type: ignore[arg-type]
        user_id="dev",
        task=task,
        holder_id=holder_id,
        is_processing=is_processing,
        last_submission_at=last_submission_at,
    )
    return ag


async def _drive_idle_eviction_one_tick(manager: SessionManager) -> None:
    """Run the idle-eviction loop just long enough for one iteration to land.

    First sleep returns immediately; second cancels the task so we exit
    cleanly. Mirrors the helpers in test_lifespan_grace_sweep.
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
        task = asyncio.create_task(manager._idle_eviction_loop())
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sm.asyncio.sleep = original_sleep  # type: ignore[assignment]


async def _cleanup_session_task(ag: AgentSession) -> None:
    if ag.task and not ag.task.done():
        ag.task.cancel()
        try:
            await ag.task
        except asyncio.CancelledError:
            pass


# ── Idle eviction tests ──────────────────────────────────────────────────


async def test_idle_eviction_releases_idle_session(monkeypatch):
    """An idle session past TTL with no pending submissions is evicted."""
    monkeypatch.setenv("IDLE_EVICTION_SECONDS", "1800")

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    ag = _agent_session_with_task(
        "s-idle",
        holder_id=manager._holder_id,
        last_submission_at=time.time() - 2000,  # 2000s > 1800s
        is_processing=False,
        is_in_tool_call=False,
    )
    manager.sessions["s-idle"] = ag

    try:
        await _drive_idle_eviction_one_tick(manager)
    finally:
        await _cleanup_session_task(ag)

    store.release_lease.assert_awaited_once_with("s-idle", manager._holder_id)
    assert "s-idle" not in manager.sessions


async def test_idle_eviction_skips_active_session(monkeypatch):
    """Recently-active session is not evicted."""
    monkeypatch.setenv("IDLE_EVICTION_SECONDS", "1800")

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    ag = _agent_session_with_task(
        "s-active",
        holder_id=manager._holder_id,
        last_submission_at=time.time(),  # just now
    )
    manager.sessions["s-active"] = ag

    try:
        await _drive_idle_eviction_one_tick(manager)
    finally:
        await _cleanup_session_task(ag)

    store.release_lease.assert_not_awaited()
    assert "s-active" in manager.sessions


async def test_idle_eviction_skips_in_tool_call(monkeypatch):
    """Session past idle TTL but mid tool-call is NOT evicted."""
    monkeypatch.setenv("IDLE_EVICTION_SECONDS", "1800")

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    ag = _agent_session_with_task(
        "s-tool",
        holder_id=manager._holder_id,
        last_submission_at=time.time() - 2000,
        is_in_tool_call=True,
    )
    manager.sessions["s-tool"] = ag

    try:
        await _drive_idle_eviction_one_tick(manager)
    finally:
        await _cleanup_session_task(ag)

    store.release_lease.assert_not_awaited()
    assert "s-tool" in manager.sessions


async def test_idle_eviction_skips_with_pending_submissions(monkeypatch):
    """Session past idle TTL with pending submissions is NOT evicted."""
    monkeypatch.setenv("IDLE_EVICTION_SECONDS", "1800")

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.poll_pending_submissions_after = AsyncMock(
        return_value=[{"_id": "x", "session_id": "s-pending"}]
    )
    manager.persistence_store = store

    ag = _agent_session_with_task(
        "s-pending",
        holder_id=manager._holder_id,
        last_submission_at=time.time() - 2000,
    )
    manager.sessions["s-pending"] = ag

    try:
        await _drive_idle_eviction_one_tick(manager)
    finally:
        await _cleanup_session_task(ag)

    store.release_lease.assert_not_awaited()
    assert "s-pending" in manager.sessions


async def test_idle_eviction_skips_processing_session(monkeypatch):
    """Session past TTL but ``is_processing=True`` is NOT evicted."""
    monkeypatch.setenv("IDLE_EVICTION_SECONDS", "1800")

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    ag = _agent_session_with_task(
        "s-proc",
        holder_id=manager._holder_id,
        last_submission_at=time.time() - 2000,
        is_processing=True,
    )
    manager.sessions["s-proc"] = ag

    try:
        await _drive_idle_eviction_one_tick(manager)
    finally:
        await _cleanup_session_task(ag)

    store.release_lease.assert_not_awaited()
    assert "s-proc" in manager.sessions


async def test_idle_eviction_skips_non_holder_sessions(monkeypatch):
    """Sessions held by another process are NOT touched."""
    monkeypatch.setenv("IDLE_EVICTION_SECONDS", "1800")

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    ag = _agent_session_with_task(
        "s-other",
        holder_id="some-other-holder",
        last_submission_at=time.time() - 2000,
    )
    manager.sessions["s-other"] = ag

    try:
        await _drive_idle_eviction_one_tick(manager)
    finally:
        await _cleanup_session_task(ag)

    store.release_lease.assert_not_awaited()
    assert "s-other" in manager.sessions


# ── claim_dormant_session tests ──────────────────────────────────────────


async def test_claim_dormant_session_bypasses_user_check(monkeypatch):
    """Worker can claim a session whose owner is some real user_id, not
    'dev' and not 'worker' — the user-ownership check in
    ``ensure_session_loaded`` would block this; ``claim_dormant_session``
    must not.
    """
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.load_session = AsyncMock(
        return_value={
            "metadata": {
                "user_id": "alice",
                "model": "test-model",
                "title": "alice's chat",
                "created_at": datetime.utcnow(),
                "turn_count": 0,
                "pending_approval": [],
                "claude_counted": False,
                "notification_destinations": [],
            },
            "messages": [],
        }
    )
    store.claim_lease = AsyncMock(return_value={"lease": {"holder_id": manager._holder_id}})
    manager.persistence_store = store

    fake_session = FakeRuntimeSession()
    fake_router = object()

    def fake_create_sync(**kwargs):
        return fake_router, fake_session

    started_holder = {"called": False}

    async def fake_start(agent_session, event_queue, tool_router):
        started_holder["called"] = True
        manager.sessions[agent_session.session_id] = agent_session
        return agent_session

    monkeypatch.setattr(manager, "_create_session_sync", fake_create_sync)
    monkeypatch.setattr(manager, "_start_agent_session", fake_start)

    result = await manager.claim_dormant_session("s-alice")

    assert result is not None
    assert result.session_id == "s-alice"
    assert result.user_id == "alice"  # ownership preserved, not coerced
    assert result.holder_id == manager._holder_id
    store.claim_lease.assert_awaited_once_with(
        "s-alice", manager._holder_id, ttl_s=30
    )
    assert started_holder["called"] is True

    # Cleanup any task created during the rebuild path.
    if result.task and not result.task.done():
        result.task.cancel()


async def test_claim_dormant_session_returns_none_on_lease_taken():
    """If ``claim_lease`` returns None (someone else holds it), we bail out
    without building a session."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.load_session = AsyncMock(
        return_value={"metadata": {"user_id": "alice"}, "messages": []}
    )
    store.claim_lease = AsyncMock(return_value=None)
    manager.persistence_store = store

    result = await manager.claim_dormant_session("s-locked")

    assert result is None
    assert "s-locked" not in manager.sessions
    store.claim_lease.assert_awaited_once()


async def test_claim_dormant_session_returns_none_when_session_missing():
    """Unknown session_id → load_session returns None → bail out without
    even attempting to claim the lease."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.load_session = AsyncMock(return_value=None)
    manager.persistence_store = store

    result = await manager.claim_dormant_session("s-missing")

    assert result is None
    store.claim_lease.assert_not_awaited()


async def test_claim_dormant_session_returns_existing_in_memory():
    """If we already hold the session in-memory, return it — don't re-claim."""
    manager = _bare_manager()
    store = _make_enabled_mock_store()
    manager.persistence_store = store

    ag = _agent_session_with_task("s-held", holder_id=manager._holder_id)
    manager.sessions["s-held"] = ag

    try:
        result = await manager.claim_dormant_session("s-held")
    finally:
        await _cleanup_session_task(ag)

    assert result is ag
    store.load_session.assert_not_awaited()
    store.claim_lease.assert_not_awaited()


# ── Worker claim tick tests ──────────────────────────────────────────────


async def _async_iter(items):
    """Wrap a list into an async iterator (mimics motor cursor behavior)."""
    for x in items:
        yield x


def _make_db_with_pending(pending_docs):
    """Build a fake ``store.db`` whose ``pending_submissions.find()`` yields
    the given docs."""
    cursor = MagicMock()
    cursor.limit = MagicMock(return_value=cursor)

    def _aiter(self_):
        return _async_iter(pending_docs).__aiter__()

    cursor.__aiter__ = _aiter

    pending_collection = MagicMock()
    pending_collection.find = MagicMock(return_value=cursor)

    db = MagicMock()
    db.pending_submissions = pending_collection
    return db


async def test_worker_claim_tick_skips_already_held_sessions(monkeypatch):
    """A session already in ``manager.sessions`` must NOT be re-claimed by
    the worker tick."""
    import main as main_module

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.db = _make_db_with_pending(
        [
            {"session_id": "s-held"},
            {"session_id": "s-held"},  # duplicate is fine
        ]
    )
    manager.persistence_store = store

    ag = _agent_session_with_task("s-held", holder_id=manager._holder_id)
    manager.sessions["s-held"] = ag

    claim_calls: list[str] = []

    async def fake_claim(sid: str):
        claim_calls.append(sid)
        return None

    monkeypatch.setattr(manager, "claim_dormant_session", fake_claim)

    with patch.object(main_module, "session_manager", manager):
        try:
            await main_module._worker_claim_tick()
        finally:
            await _cleanup_session_task(ag)

    assert claim_calls == []  # held → never tried
    store.load_session.assert_not_awaited()


async def test_worker_claim_tick_claims_new_session(monkeypatch):
    """A pending session NOT held by us is passed to ``claim_dormant_session``."""
    import main as main_module

    manager = _bare_manager()
    store = _make_enabled_mock_store()
    store.db = _make_db_with_pending([{"session_id": "s-new"}])
    manager.persistence_store = store

    claim_calls: list[str] = []

    async def fake_claim(sid: str):
        claim_calls.append(sid)
        return None

    monkeypatch.setattr(manager, "claim_dormant_session", fake_claim)

    with patch.object(main_module, "session_manager", manager):
        await main_module._worker_claim_tick()

    assert claim_calls == ["s-new"]


async def test_worker_claim_tick_noop_when_store_disabled(monkeypatch):
    """When persistence is disabled, the tick is a clean no-op."""
    import main as main_module

    manager = _bare_manager()
    store = AsyncMock()
    store.enabled = False
    manager.persistence_store = store

    claim_calls: list[str] = []

    async def fake_claim(sid: str):
        claim_calls.append(sid)

    monkeypatch.setattr(manager, "claim_dormant_session", fake_claim)

    with patch.object(main_module, "session_manager", manager):
        await main_module._worker_claim_tick()

    assert claim_calls == []
