"""Regression tests for server-side session persistence restore/access."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, UTC
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from session_manager import AgentSession, SessionManager  # noqa: E402


class FakeRuntimeSession:
    def __init__(self, *, hf_token: str | None = None, model: str = "test-model"):
        self.hf_token = hf_token
        self.context_manager = SimpleNamespace(items=[])
        self.pending_approval = None
        self.turn_count = 0
        self.config = SimpleNamespace(model_name=model)
        self.notification_destinations = []


class RestoreStore(NoopSessionStore):
    enabled = True

    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        delay: float = 0,
    ) -> None:
        self.metadata = metadata or {
            "session_id": "persisted-session",
            "user_id": "owner",
            "model": "test-model",
            "created_at": datetime.now(UTC),
        }
        self.messages = messages or []
        self.delay = delay
        self.load_calls = 0

    async def load_session(self, session_id: str, **_: Any) -> dict[str, Any] | None:
        self.load_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        metadata = dict(self.metadata)
        metadata.setdefault("session_id", session_id)
        metadata.setdefault("_id", session_id)
        return {"metadata": metadata, "messages": self.messages}


def _manager_with_store(store: NoopSessionStore) -> SessionManager:
    manager = object.__new__(SessionManager)
    manager.config = SimpleNamespace(model_name="test-model")
    manager.sessions = {}
    manager._lock = asyncio.Lock()
    manager.persistence_store = store
    return manager


def _runtime_agent_session(
    session_id: str,
    *,
    user_id: str = "owner",
    hf_token: str | None = "owner-token",
) -> AgentSession:
    runtime_session = FakeRuntimeSession(hf_token=hf_token)
    return AgentSession(
        session_id=session_id,
        session=runtime_session,  # type: ignore[arg-type]
        tool_router=object(),  # type: ignore[arg-type]
        submission_queue=asyncio.Queue(),
        user_id=user_id,
        hf_token=hf_token,
    )


def _install_fake_runtime(manager: SessionManager) -> asyncio.Event:
    stop = asyncio.Event()
    manager.run_calls = 0  # type: ignore[attr-defined]

    def fake_create_session_sync(**kwargs: Any):
        return object(), FakeRuntimeSession(
            hf_token=kwargs.get("hf_token"),
            model=kwargs.get("model") or "test-model",
        )

    async def fake_run_session(*_: Any) -> None:
        manager.run_calls += 1  # type: ignore[attr-defined]
        await stop.wait()

    manager._create_session_sync = fake_create_session_sync  # type: ignore[method-assign]
    manager._run_session = fake_run_session  # type: ignore[method-assign]
    return stop


async def _cancel_runtime_tasks(manager: SessionManager) -> None:
    tasks = [
        agent_session.task
        for agent_session in manager.sessions.values()
        if agent_session.task and not agent_session.task.done()
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_existing_session_rejects_cross_user_token_overwrite():
    manager = _manager_with_store(NoopSessionStore())
    existing = _runtime_agent_session("s1", user_id="victim", hf_token="victim-token")
    manager.sessions["s1"] = existing

    result = await manager.ensure_session_loaded(
        "s1", user_id="attacker", hf_token="attacker-token"
    )

    assert result is None
    assert existing.hf_token == "victim-token"
    assert existing.session.hf_token == "victim-token"


@pytest.mark.asyncio
async def test_existing_session_updates_token_after_access_check():
    manager = _manager_with_store(NoopSessionStore())
    existing = _runtime_agent_session("s1", user_id="owner", hf_token="old-token")
    manager.sessions["s1"] = existing

    result = await manager.ensure_session_loaded(
        "s1", user_id="owner", hf_token="new-token"
    )

    assert result is existing
    assert existing.hf_token == "new-token"
    assert existing.session.hf_token == "new-token"


@pytest.mark.asyncio
async def test_concurrent_lazy_restore_starts_only_one_agent_task():
    store = RestoreStore(delay=0.01)
    manager = _manager_with_store(store)
    stop = _install_fake_runtime(manager)

    try:
        first, second = await asyncio.gather(
            manager.ensure_session_loaded("persisted-session", user_id="owner"),
            manager.ensure_session_loaded("persisted-session", user_id="owner"),
        )
        await asyncio.sleep(0)

        assert first is second
        assert list(manager.sessions) == ["persisted-session"]
        assert manager.run_calls == 1  # type: ignore[attr-defined]
        assert not stop.is_set()
    finally:
        stop.set()
        await _cancel_runtime_tasks(manager)


@pytest.mark.asyncio
async def test_lazy_restore_preserves_pending_approval_tool_calls():
    store = RestoreStore(
        metadata={
            "session_id": "approval-session",
            "user_id": "owner",
            "model": "test-model",
            "pending_approval": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "create_file",
                        "arguments": '{"path":"app.py"}',
                    },
                }
            ],
        }
    )
    manager = _manager_with_store(store)
    stop = _install_fake_runtime(manager)

    try:
        restored = await manager.ensure_session_loaded("approval-session", user_id="owner")

        assert restored is not None
        tool_calls = restored.session.pending_approval["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].function.name == "create_file"
        assert tool_calls[0].function.arguments == '{"path":"app.py"}'
    finally:
        stop.set()
        await _cancel_runtime_tasks(manager)


@pytest.mark.asyncio
async def test_list_sessions_dev_uses_store_dev_visibility():
    class ListStore(NoopSessionStore):
        enabled = True

        def __init__(self) -> None:
            self.seen_user_id: str | None = None

        async def list_sessions(self, user_id: str, **_: Any) -> list[dict[str, Any]]:
            self.seen_user_id = user_id
            if user_id == "dev":
                return [
                    {
                        "session_id": "s1",
                        "user_id": "alice",
                        "model": "m",
                        "created_at": datetime.now(UTC),
                    },
                    {
                        "session_id": "s2",
                        "user_id": "bob",
                        "model": "m",
                        "created_at": datetime.now(UTC),
                    },
                ]
            return []

    store = ListStore()
    manager = _manager_with_store(store)

    sessions = await manager.list_sessions(user_id="dev")

    assert store.seen_user_id == "dev"
    assert {session["session_id"] for session in sessions} == {"s1", "s2"}
