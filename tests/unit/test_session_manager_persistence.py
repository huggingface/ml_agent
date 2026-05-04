"""Regression tests for server-side session persistence restore/access."""

from __future__ import annotations

import asyncio
import sys
import threading
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
        self.auto_approval_enabled = False
        self.auto_approval_cost_cap_usd = None
        self.auto_approval_estimated_spend_usd = 0.0
        self.sandbox = None
        self.sandbox_hardware = None
        self.sandbox_preload_task = None
        self.sandbox_preload_cancel_event = None

    def auto_approval_policy_summary(self):
        cap = self.auto_approval_cost_cap_usd
        remaining = None if cap is None else max(0, cap - self.auto_approval_estimated_spend_usd)
        return {
            "enabled": self.auto_approval_enabled,
            "cost_cap_usd": cap,
            "estimated_spend_usd": self.auto_approval_estimated_spend_usd,
            "remaining_usd": remaining,
        }

    def set_auto_approval_policy(self, *, enabled, cost_cap_usd):
        self.auto_approval_enabled = enabled
        self.auto_approval_cost_cap_usd = cost_cap_usd


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
        self.updated_fields: list[tuple[str, dict[str, Any]]] = []

    async def load_session(self, session_id: str, **_: Any) -> dict[str, Any] | None:
        self.load_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        metadata = dict(self.metadata)
        metadata.setdefault("session_id", session_id)
        metadata.setdefault("_id", session_id)
        return {"metadata": metadata, "messages": self.messages}

    async def update_session_fields(self, session_id: str, **fields: Any) -> None:
        self.updated_fields.append((session_id, fields))
        self.metadata.update(fields)


class CloseableResource:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _manager_with_store(store: NoopSessionStore) -> SessionManager:
    manager = object.__new__(SessionManager)
    manager.config = SimpleNamespace(model_name="test-model")
    manager.sessions = {}
    manager._lock = asyncio.Lock()
    manager.persistence_store = store
    manager.messaging_gateway = CloseableResource()
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


@pytest.mark.asyncio
async def test_update_session_auto_approval_defaults_to_five_dollars():
    manager = _manager_with_store(NoopSessionStore())
    existing = _runtime_agent_session("s1", user_id="owner")
    manager.sessions["s1"] = existing

    summary = await manager.update_session_auto_approval(
        "s1",
        enabled=True,
        cost_cap_usd=None,
        cap_provided=False,
    )

    assert summary["enabled"] is True
    assert summary["cost_cap_usd"] == 5.0
    assert summary["remaining_usd"] == 5.0


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
async def test_close_cancels_preload_and_deletes_owned_sandbox(monkeypatch):
    deleted: list[str] = []

    async def fake_record_sandbox_destroy(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "agent.core.telemetry.record_sandbox_destroy",
        fake_record_sandbox_destroy,
    )

    store = NoopSessionStore()
    manager = _manager_with_store(store)
    gateway = CloseableResource()
    persistence = CloseableResource()
    manager.messaging_gateway = gateway  # type: ignore[assignment]
    manager.persistence_store = persistence  # type: ignore[assignment]

    cancel_event = asyncio.Event()
    preload_cancel_event = threading.Event()

    async def preload():
        while not preload_cancel_event.is_set():
            await asyncio.sleep(0)
        cancel_event.set()

    session = FakeRuntimeSession(hf_token="token")
    session.session_id = "s1"
    session.persistence_store = NoopSessionStore()
    session.sandbox = SimpleNamespace(
        space_id="owner/sandbox-12345678",
        _owns_space=True,
        delete=lambda: deleted.append("owner/sandbox-12345678"),
    )
    session.sandbox_hardware = "cpu-basic"
    session.sandbox_preload_cancel_event = preload_cancel_event
    session.sandbox_preload_task = asyncio.create_task(preload())
    manager.sessions["s1"] = AgentSession(
        session_id="s1",
        session=session,  # type: ignore[arg-type]
        tool_router=object(),  # type: ignore[arg-type]
        submission_queue=asyncio.Queue(),
        user_id="owner",
        hf_token="token",
    )

    await manager.close()

    assert preload_cancel_event.is_set()
    assert cancel_event.is_set()
    assert deleted == ["owner/sandbox-12345678"]
    assert gateway.closed is True
    assert persistence.closed is True


@pytest.mark.asyncio
async def test_close_closes_resources_when_sandbox_cleanup_fails():
    manager = _manager_with_store(NoopSessionStore())
    gateway = CloseableResource()
    persistence = CloseableResource()
    manager.messaging_gateway = gateway  # type: ignore[assignment]
    manager.persistence_store = persistence  # type: ignore[assignment]
    manager.sessions["s1"] = _runtime_agent_session("s1")
    manager.sessions["s2"] = _runtime_agent_session("s2")
    cleaned: list[str] = []

    async def fake_cleanup(session):
        cleaned.append(session.hf_token)
        if session.hf_token == "owner-token":
            raise RuntimeError("boom")

    manager._cleanup_sandbox = fake_cleanup  # type: ignore[method-assign]

    await manager.close()

    assert cleaned == ["owner-token", "owner-token"]
    assert gateway.closed is True
    assert persistence.closed is True


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
async def test_existing_session_retries_preload_after_token_recovered():
    manager = _manager_with_store(NoopSessionStore())
    existing = _runtime_agent_session("s1", user_id="owner", hf_token=None)
    done_task = asyncio.get_running_loop().create_future()
    done_task.set_result(None)
    existing.session.sandbox_preload_task = done_task
    existing.session.sandbox_preload_error = (
        "No HF token available. Cannot create sandbox."
    )
    manager.sessions["s1"] = existing
    started: list[str] = []

    def fake_start_cpu_sandbox_preload(agent_session):
        started.append(agent_session.session_id)

    manager._start_cpu_sandbox_preload = fake_start_cpu_sandbox_preload  # type: ignore[method-assign]

    result = await manager.ensure_session_loaded(
        "s1",
        user_id="owner",
        hf_token="new-token",
    )

    assert result is existing
    assert existing.hf_token == "new-token"
    assert existing.session.hf_token == "new-token"
    assert existing.session.sandbox_preload_error is None
    assert existing.session.sandbox_preload_task is None
    assert started == ["s1"]


@pytest.mark.asyncio
async def test_existing_session_does_not_retry_preload_when_disabled():
    manager = _manager_with_store(NoopSessionStore())
    existing = _runtime_agent_session("s1", user_id="owner", hf_token=None)
    done_task = asyncio.get_running_loop().create_future()
    done_task.set_result(None)
    existing.session.sandbox_preload_task = done_task
    existing.session.sandbox_preload_error = (
        "No HF token available. Cannot create sandbox."
    )
    manager.sessions["s1"] = existing
    started: list[str] = []

    def fake_start_cpu_sandbox_preload(agent_session):
        started.append(agent_session.session_id)

    manager._start_cpu_sandbox_preload = fake_start_cpu_sandbox_preload  # type: ignore[method-assign]

    result = await manager.ensure_session_loaded(
        "s1",
        user_id="owner",
        hf_token="new-token",
        preload_sandbox=False,
    )

    assert result is existing
    assert existing.hf_token == "new-token"
    assert existing.session.hf_token == "new-token"
    assert existing.session.sandbox_preload_error == (
        "No HF token available. Cannot create sandbox."
    )
    assert started == []


@pytest.mark.asyncio
async def test_concurrent_lazy_restore_starts_only_one_agent_task():
    store = RestoreStore(delay=0.01)
    manager = _manager_with_store(store)
    stop = _install_fake_runtime(manager)
    scheduled: list[str] = []

    def fake_start_cpu_sandbox_preload(agent_session: AgentSession) -> None:
        scheduled.append(agent_session.session_id)

    manager._start_cpu_sandbox_preload = fake_start_cpu_sandbox_preload  # type: ignore[method-assign]

    try:
        first, second = await asyncio.gather(
            manager.ensure_session_loaded("persisted-session", user_id="owner"),
            manager.ensure_session_loaded("persisted-session", user_id="owner"),
        )
        await asyncio.sleep(0)

        assert first is second
        assert list(manager.sessions) == ["persisted-session"]
        assert manager.run_calls == 1  # type: ignore[attr-defined]
        assert scheduled == ["persisted-session"]
        assert not stop.is_set()
    finally:
        stop.set()
        await _cancel_runtime_tasks(manager)


@pytest.mark.asyncio
async def test_create_session_schedules_cpu_sandbox_preload():
    manager = _manager_with_store(NoopSessionStore())
    stop = _install_fake_runtime(manager)
    scheduled: list[str] = []

    def fake_start_cpu_sandbox_preload(agent_session: AgentSession) -> None:
        scheduled.append(agent_session.session_id)

    manager._start_cpu_sandbox_preload = fake_start_cpu_sandbox_preload  # type: ignore[method-assign]

    try:
        session_id = await manager.create_session(user_id="owner", hf_token="token")

        assert scheduled == [session_id]
        assert session_id in manager.sessions
    finally:
        stop.set()
        await _cancel_runtime_tasks(manager)


@pytest.mark.asyncio
async def test_lazy_restore_schedules_cpu_sandbox_preload():
    manager = _manager_with_store(RestoreStore())
    stop = _install_fake_runtime(manager)
    scheduled: list[str] = []

    def fake_start_cpu_sandbox_preload(agent_session: AgentSession) -> None:
        scheduled.append(agent_session.session_id)

    manager._start_cpu_sandbox_preload = fake_start_cpu_sandbox_preload  # type: ignore[method-assign]

    try:
        restored = await manager.ensure_session_loaded("persisted-session", user_id="owner")

        assert restored is not None
        assert scheduled == ["persisted-session"]
        assert "persisted-session" in manager.sessions
    finally:
        stop.set()
        await _cancel_runtime_tasks(manager)


@pytest.mark.asyncio
async def test_lazy_restore_deletes_persisted_sandbox_before_preload(monkeypatch):
    deleted: list[tuple[str, str, str]] = []

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def delete_repo(self, repo_id, repo_type):
            deleted.append((self.token, repo_id, repo_type))

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    store = RestoreStore(
        metadata={
            "session_id": "persisted-session",
            "user_id": "owner",
            "model": "test-model",
            "created_at": datetime.now(UTC),
            "sandbox_space_id": "owner/sandbox-12345678",
            "sandbox_hardware": "cpu-basic",
            "sandbox_owner": "owner",
            "sandbox_created_at": datetime.now(UTC),
            "sandbox_status": "active",
        }
    )
    manager = _manager_with_store(store)
    stop = _install_fake_runtime(manager)
    scheduled: list[str] = []

    def fake_start_cpu_sandbox_preload(agent_session: AgentSession) -> None:
        scheduled.append(agent_session.session_id)

    manager._start_cpu_sandbox_preload = fake_start_cpu_sandbox_preload  # type: ignore[method-assign]

    try:
        restored = await manager.ensure_session_loaded(
            "persisted-session",
            user_id="owner",
            hf_token="user-token",
        )

        assert restored is not None
        assert deleted == [("user-token", "owner/sandbox-12345678", "space")]
        assert scheduled == ["persisted-session"]
        assert store.metadata["sandbox_space_id"] is None
        assert store.metadata["sandbox_status"] == "destroyed"
    finally:
        stop.set()
        await _cancel_runtime_tasks(manager)


@pytest.mark.asyncio
async def test_lazy_restore_can_skip_cpu_sandbox_preload_after_cleanup(monkeypatch):
    deleted: list[str] = []

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def delete_repo(self, repo_id, repo_type):
            deleted.append(repo_id)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    store = RestoreStore(
        metadata={
            "session_id": "persisted-session",
            "user_id": "owner",
            "model": "test-model",
            "created_at": datetime.now(UTC),
            "sandbox_space_id": "owner/sandbox-87654321",
            "sandbox_status": "active",
        }
    )
    manager = _manager_with_store(store)
    stop = _install_fake_runtime(manager)
    scheduled: list[str] = []

    def fake_start_cpu_sandbox_preload(agent_session: AgentSession) -> None:
        scheduled.append(agent_session.session_id)

    manager._start_cpu_sandbox_preload = fake_start_cpu_sandbox_preload  # type: ignore[method-assign]

    try:
        restored = await manager.ensure_session_loaded(
            "persisted-session",
            user_id="owner",
            hf_token="user-token",
            preload_sandbox=False,
        )

        assert restored is not None
        assert deleted == ["owner/sandbox-87654321"]
        assert scheduled == []
        assert store.metadata["sandbox_space_id"] is None
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
async def test_lazy_restore_preserves_auto_approval_policy():
    store = RestoreStore(
        metadata={
            "session_id": "yolo-session",
            "user_id": "owner",
            "model": "test-model",
            "auto_approval_enabled": True,
            "auto_approval_cost_cap_usd": 5.0,
            "auto_approval_estimated_spend_usd": 1.25,
        }
    )
    manager = _manager_with_store(store)
    stop = _install_fake_runtime(manager)

    try:
        restored = await manager.ensure_session_loaded("yolo-session", user_id="owner")

        assert restored is not None
        assert restored.session.auto_approval_enabled is True
        assert restored.session.auto_approval_cost_cap_usd == 5.0
        assert restored.session.auto_approval_estimated_spend_usd == 1.25
        assert restored.session.auto_approval_policy_summary()["remaining_usd"] == 3.75
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
                        "auto_approval_enabled": True,
                        "auto_approval_cost_cap_usd": 5.0,
                        "auto_approval_estimated_spend_usd": 2.0,
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
    yolo = next(session for session in sessions if session["session_id"] == "s1")
    assert yolo["auto_approval"] == {
        "enabled": True,
        "cost_cap_usd": 5.0,
        "estimated_spend_usd": 2.0,
        "remaining_usd": 3.0,
    }
