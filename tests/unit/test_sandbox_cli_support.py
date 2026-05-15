import asyncio
from types import SimpleNamespace

import agent.tools.sandbox_tool as sandbox_tool
from agent.tools.sandbox_client import _DOCKERFILE, Sandbox


def test_sandbox_image_installs_gh_and_hf_clis():
    assert "git-lfs gh wget" in _DOCKERFILE
    assert '"huggingface_hub[cli]"' in _DOCKERFILE
    assert "gh and hf CLIs are preinstalled" in Sandbox.TOOLS["bash"]["description"]


def test_sandbox_create_forwards_user_github_token(monkeypatch):
    captured = {}

    async def fake_ensure_sandbox(session, **kwargs):
        captured.update(kwargs)
        return (
            SimpleNamespace(
                space_id="user/sandbox-abc123",
                url="https://huggingface.co/spaces/user/sandbox-abc123",
            ),
            None,
        )

    monkeypatch.setattr(sandbox_tool, "_ensure_sandbox", fake_ensure_sandbox)

    session = SimpleNamespace(sandbox=None, hf_token="hf-token")
    out, ok = asyncio.run(
        sandbox_tool.sandbox_create_handler(
            {"github_token": "github_pat_user_owned"}, session=session
        )
    )

    assert ok is True
    assert "github_pat_user_owned" not in out
    assert captured["extra_secrets"]["GH_TOKEN"] == "github_pat_user_owned"
    assert captured["extra_secrets"]["GITHUB_TOKEN"] == "github_pat_user_owned"
