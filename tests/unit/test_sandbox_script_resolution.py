from types import SimpleNamespace

import pytest

from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC
from agent.tools.sandbox_tool import resolve_sandbox_script


class FakeSandbox:
    def __init__(self):
        self.read_paths = []

    def read(self, path, *, limit):
        self.read_paths.append((path, limit))
        return SimpleNamespace(
            success=True,
            output="1\tprint('training')\n2\tprint('done')",
            error="",
        )


@pytest.mark.asyncio
async def test_resolve_sandbox_script_accepts_bare_python_filename():
    sandbox = FakeSandbox()

    content, error = await resolve_sandbox_script(sandbox, "train_smollm2.py")

    assert error is None
    assert content == "print('training')\nprint('done')"
    assert sandbox.read_paths == [("train_smollm2.py", 100_000)]


@pytest.mark.asyncio
async def test_resolve_sandbox_script_accepts_relative_python_path():
    sandbox = FakeSandbox()

    content, error = await resolve_sandbox_script(sandbox, "scripts/train.py")

    assert error is None
    assert content == "print('training')\nprint('done')"
    assert sandbox.read_paths == [("scripts/train.py", 100_000)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "script",
    [
        "https://example.com/train.py",
        "http://example.com/train.py",
        "train_smollm2.py --epochs 1",
        "print('hello')",
    ],
)
async def test_resolve_sandbox_script_ignores_non_path_scripts(script):
    sandbox = FakeSandbox()

    content, error = await resolve_sandbox_script(sandbox, script)

    assert content is None
    assert error is None
    assert sandbox.read_paths == []


def test_hf_jobs_script_description_mentions_bare_python_filenames():
    script_description = HF_JOBS_TOOL_SPEC["parameters"]["properties"]["script"][
        "description"
    ]

    assert "bare 'train.py'" in script_description
    assert "smoke-test in a GPU sandbox before submission" in script_description
