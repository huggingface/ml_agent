"""Concurrency tests for backend/session_manager.py capacity enforcement."""

import asyncio
import dataclasses
import enum
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _install_agent_stubs() -> None:
    agent_pkg = ModuleType("agent")
    agent_pkg.__path__ = []

    core_pkg = ModuleType("agent.core")
    core_pkg.__path__ = []

    config_mod = ModuleType("agent.config")

    def load_config(path):
        class _StubConfig:
            def __init__(self) -> None:
                self.model_name = "test-model"
                self.mcpServers = {}

            def model_copy(self, deep: bool = False):
                return self

        return _StubConfig()

    config_mod.load_config = load_config

    agent_loop_mod = ModuleType("agent.core.agent_loop")

    async def process_submission(session, submission):
        return False

    agent_loop_mod.process_submission = process_submission

    session_mod = ModuleType("agent.core.session")

    class OpType(enum.Enum):
        USER_INPUT = "user_input"
        EXEC_APPROVAL = "exec_approval"
        UNDO = "undo"
        COMPACT = "compact"
        SHUTDOWN = "shutdown"

    @dataclasses.dataclass
    class Event:
        event_type: str
        data: object = None

    class Session:
        def __init__(self, event_queue, config, tool_router, hf_token=None):
            self.config = SimpleNamespace(model_name=config.model_name)
            self.context_manager = SimpleNamespace(items=[])
            self.pending_approval = None
            self.is_running = False
            self.hf_token = hf_token

        async def send_event(self, event):
            return None

        def cancel(self):
            self.is_running = False

    session_mod.Event = Event
    session_mod.OpType = OpType
    session_mod.Session = Session

    tools_mod = ModuleType("agent.core.tools")

    class ToolRouter:
        def __init__(self, mcp_servers, hf_token=None):
            self.mcp_servers = mcp_servers
            self.hf_token = hf_token

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get_tool_specs_for_llm(self):
            return []

    tools_mod.ToolRouter = ToolRouter

    sys.modules.setdefault("agent", agent_pkg)
    sys.modules.setdefault("agent.core", core_pkg)
    sys.modules.setdefault("agent.config", config_mod)
    sys.modules.setdefault("agent.core.agent_loop", agent_loop_mod)
    sys.modules.setdefault("agent.core.session", session_mod)
    sys.modules.setdefault("agent.core.tools", tools_mod)


_install_agent_stubs()

import session_manager as session_manager_module  # noqa: E402


class _SlowToolRouter(session_manager_module.ToolRouter):
    def __init__(self, mcp_servers, hf_token=None):
        time.sleep(0.2)
        super().__init__(mcp_servers, hf_token=hf_token)


class _DummySession(session_manager_module.Session):
    pass


@pytest.fixture
def manager(monkeypatch):
    class _FixtureConfig:
        def __init__(self) -> None:
            self.model_name = "test-model"
            self.mcpServers = {}

        def model_copy(self, deep: bool = False):
            return self

    monkeypatch.setattr(session_manager_module, "load_config", lambda path: _FixtureConfig())
    monkeypatch.setattr(session_manager_module, "ToolRouter", _SlowToolRouter)
    monkeypatch.setattr(session_manager_module, "Session", _DummySession)
    monkeypatch.setattr(session_manager_module, "MAX_SESSIONS", 200)
    monkeypatch.setattr(session_manager_module, "MAX_SESSIONS_PER_USER", 10)
    return session_manager_module.SessionManager(config_path="ignored")


def test_create_session_respects_global_capacity_under_concurrency(monkeypatch, manager):
    monkeypatch.setattr(session_manager_module, "MAX_SESSIONS", 1)
    monkeypatch.setattr(session_manager_module, "MAX_SESSIONS_PER_USER", 10)

    async def _run():
        return await asyncio.gather(
            manager.create_session(user_id="alice"),
            manager.create_session(user_id="bob"),
            return_exceptions=True,
        )

    results = asyncio.run(_run())

    successes = [result for result in results if isinstance(result, str)]
    failures = [result for result in results if isinstance(result, Exception)]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], session_manager_module.SessionCapacityError)
    assert failures[0].error_type == "global"


def test_create_session_respects_per_user_capacity_under_concurrency(monkeypatch, manager):
    monkeypatch.setattr(session_manager_module, "MAX_SESSIONS", 10)
    monkeypatch.setattr(session_manager_module, "MAX_SESSIONS_PER_USER", 1)

    async def _run():
        return await asyncio.gather(
            manager.create_session(user_id="alice"),
            manager.create_session(user_id="alice"),
            return_exceptions=True,
        )

    results = asyncio.run(_run())

    successes = [result for result in results if isinstance(result, str)]
    failures = [result for result in results if isinstance(result, Exception)]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], session_manager_module.SessionCapacityError)
    assert failures[0].error_type == "per_user"


def test_session_slot_is_released_if_construction_fails(monkeypatch, manager):
    monkeypatch.setattr(session_manager_module, "MAX_SESSIONS", 1)
    monkeypatch.setattr(session_manager_module, "MAX_SESSIONS_PER_USER", 1)

    calls = {"count": 0}

    class _FailingToolRouter(_SlowToolRouter):
        def __init__(self, mcp_servers, hf_token=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("boom")
            super().__init__(mcp_servers, hf_token=hf_token)

    monkeypatch.setattr(session_manager_module, "ToolRouter", _FailingToolRouter)

    async def _first_run():
        return await asyncio.gather(manager.create_session(user_id="alice"), return_exceptions=True)

    first = asyncio.run(_first_run())
    assert isinstance(first[0], RuntimeError)

    second = asyncio.run(manager.create_session(user_id="alice"))
    assert isinstance(second, str)