import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

from agent.config import Config
from agent.core import agent_loop
from agent.core.agent_loop import _needs_approval
from agent.core.session import OpType
from agent.core.tools import create_builtin_tools
from agent.tools.sandbox_tool import get_sandbox_tools


def test_default_cpu_sandbox_create_does_not_require_approval():
    config = SimpleNamespace(yolo_mode=False)

    assert _needs_approval("sandbox_create", {}, config) is False
    assert _needs_approval("sandbox_create", {"hardware": "cpu-basic"}, config) is False


def test_non_default_sandbox_create_still_requires_approval():
    config = SimpleNamespace(yolo_mode=False)

    assert (
        _needs_approval("sandbox_create", {"hardware": "cpu-upgrade"}, config) is True
    )
    assert _needs_approval("sandbox_create", {"hardware": "t4-small"}, config) is True


def test_prompt_and_tool_specs_do_not_require_cpu_sandbox_create():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()
    tool_specs = {tool.name: tool.description for tool in get_sandbox_tools()}

    assert "sandbox_create → install deps" not in prompt
    assert "Do NOT call sandbox_create before normal CPU work" in prompt
    assert "cpu-basic sandbox is already available" in prompt

    assert (
        "cpu-basic sandbox is already started automatically"
        in tool_specs["sandbox_create"]
    )
    assert "started automatically for normal CPU work" in tool_specs["bash"]


def test_prompt_rejects_local_machine_paths_for_hf_jobs_scripts():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "Never pass a local machine path to hf_jobs.script" in prompt
    assert "/fsx/..." in prompt
    assert "inline Python source code" in prompt
    assert "a file already written in the session sandbox" in prompt


def test_local_tool_runtime_excludes_sandbox_create():
    tool_names = {tool.name for tool in create_builtin_tools(local_mode=True)}

    assert {"bash", "read", "write", "edit"} <= tool_names
    assert "sandbox_create" not in tool_names


def test_sandbox_tool_runtime_includes_sandbox_create():
    tool_names = {tool.name for tool in create_builtin_tools(local_mode=False)}

    assert {"sandbox_create", "bash", "read", "write", "edit"} <= tool_names


@pytest.mark.asyncio
async def test_cli_sandbox_runtime_preloads_and_tears_down_sandbox(monkeypatch):
    started = []
    torn_down = []

    class FakeToolRouter:
        tools = {}

        def get_tool_specs_for_llm(self):
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    def fake_start_cpu_sandbox_preload(session):
        started.append(session)
        return None

    async def fake_teardown_session_sandbox(session):
        torn_down.append(session)

    monkeypatch.setattr(
        agent_loop, "start_cpu_sandbox_preload", fake_start_cpu_sandbox_preload
    )
    monkeypatch.setattr(
        agent_loop, "teardown_session_sandbox", fake_teardown_session_sandbox
    )

    submission_queue = asyncio.Queue()
    event_queue = asyncio.Queue()
    session_holder = [None]
    config = Config.model_validate(
        {"model_name": "openai/gpt-5.5", "save_sessions": False}
    )

    task = asyncio.create_task(
        agent_loop.submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=FakeToolRouter(),
            session_holder=session_holder,
            hf_token="hf-token",
            user_id="tester",
            local_mode=False,
        )
    )

    ready = await asyncio.wait_for(event_queue.get(), timeout=1)
    assert ready.event_type == "ready"
    assert started == [session_holder[0]]
    assert session_holder[0].local_mode is False

    await submission_queue.put(
        SimpleNamespace(
            operation=SimpleNamespace(op_type=OpType.SHUTDOWN, data=None),
        )
    )
    await asyncio.wait_for(task, timeout=1)

    assert torn_down == [session_holder[0]]
