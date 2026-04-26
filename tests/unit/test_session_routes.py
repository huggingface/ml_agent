"""API-level tests for backend/routes/agent.py."""

import dataclasses
import enum
import sys
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _install_agent_stubs() -> None:
    agent_pkg = ModuleType("agent")
    agent_pkg.__path__ = []

    core_pkg = ModuleType("agent.core")
    core_pkg.__path__ = []

    config_mod = ModuleType("agent.config")

    class _StubConfig:
        def __init__(self) -> None:
            self.model_name = "test-model"
            self.mcpServers = {}

        def model_copy(self, deep: bool = False):
            return self

    def load_config(path):
        return _StubConfig()

    config_mod.load_config = load_config

    agent_loop_mod = ModuleType("agent.core.agent_loop")

    async def process_submission(session, submission):
        return False

    agent_loop_mod.process_submission = process_submission

    session_mod = ModuleType("agent.core.session")

    class OpType(str, enum.Enum):
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
            self.config = _StubConfig()
            self.context_manager = type("Context", (), {"items": []})()
            self.pending_approval = None
            self.is_running = False

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

    llm_params_mod = ModuleType("agent.core.llm_params")

    def _resolve_llm_params(model, reasoning_effort=None):
        return {"model": model}

    llm_params_mod._resolve_llm_params = _resolve_llm_params

    litellm_mod = ModuleType("litellm")

    async def acompletion(*args, **kwargs):
        raise RuntimeError("litellm should not be called in this test")

    litellm_mod.acompletion = acompletion

    sys.modules.setdefault("agent", agent_pkg)
    sys.modules.setdefault("agent.core", core_pkg)
    sys.modules.setdefault("agent.config", config_mod)
    sys.modules.setdefault("agent.core.agent_loop", agent_loop_mod)
    sys.modules.setdefault("agent.core.session", session_mod)
    sys.modules.setdefault("agent.core.tools", tools_mod)
    sys.modules.setdefault("agent.core.llm_params", llm_params_mod)
    sys.modules.setdefault("litellm", litellm_mod)


_install_agent_stubs()

import routes.agent as agent_routes  # noqa: E402


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(agent_routes.router)
    app.dependency_overrides[agent_routes.get_current_user] = lambda: {
        "user_id": "alice",
        "username": "alice",
        "authenticated": True,
        "plan": "pro",
    }
    return TestClient(app)


def test_create_session_returns_503_when_capacity_is_exhausted(monkeypatch):
    async def _raise_capacity(*args, **kwargs):
        raise agent_routes.SessionCapacityError(
            "Server is at capacity (200/200 sessions). Please try again later.",
            error_type="global",
        )

    monkeypatch.setattr(agent_routes.session_manager, "create_session", _raise_capacity)

    client = _build_client()
    response = client.post("/api/session", json={})

    assert response.status_code == 503
    assert "at capacity" in response.json()["detail"]