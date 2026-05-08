from types import SimpleNamespace
from pathlib import Path

from agent.core.agent_loop import _needs_approval
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
