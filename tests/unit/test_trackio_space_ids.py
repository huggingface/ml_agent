import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from agent.tools import sandbox_tool
from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, HfJobsTool
from agent.tools.sandbox_tool import SANDBOX_CREATE_TOOL_SPEC, sandbox_create_handler
from agent.tools.trackio_seed import normalize_trackio_space_id


def _legacy_space_id(suffix: str = "abcd1234") -> str:
    return "alice/" + "ml" + "intern" + f"-{suffix}"


def test_trackio_space_examples_use_hyphenated_ml_intern_prefix():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()
    tool_specs = json.dumps([HF_JOBS_TOOL_SPEC, SANDBOX_CREATE_TOOL_SPEC])
    legacy_prefix = "ml" + "intern"

    assert "<username>/ml-intern-<8-char-id>" in prompt
    assert "<username>/ml-intern-<8char>" in tool_specs
    assert legacy_prefix not in prompt
    assert legacy_prefix not in tool_specs


def test_normalize_trackio_space_id_rewrites_legacy_prefix():
    assert normalize_trackio_space_id(_legacy_space_id()) == "alice/ml-intern-abcd1234"
    assert (
        normalize_trackio_space_id("alice/custom-dashboard") == "alice/custom-dashboard"
    )
    assert normalize_trackio_space_id(None) is None


def test_sandbox_create_normalizes_trackio_space_id(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_seed_trackio_dashboard(session, space_id):
        captured["seeded_space_id"] = space_id

    async def fake_ensure_sandbox(
        session,
        hardware="cpu-basic",
        extra_secrets=None,
        **create_kwargs,
    ):
        captured["extra_secrets"] = extra_secrets
        return (
            SimpleNamespace(
                space_id="alice/sandbox-12345678",
                url="https://huggingface.co/spaces/alice/sandbox-12345678",
            ),
            None,
        )

    monkeypatch.setattr(
        sandbox_tool, "_seed_trackio_dashboard_safe", fake_seed_trackio_dashboard
    )
    monkeypatch.setattr(sandbox_tool, "_ensure_sandbox", fake_ensure_sandbox)

    out, ok = asyncio.run(
        sandbox_create_handler(
            {"trackio_space_id": _legacy_space_id()},
            session=SimpleNamespace(sandbox=None),
        )
    )

    assert ok is True
    assert "Visibility: private" in out
    assert captured["seeded_space_id"] == "alice/ml-intern-abcd1234"
    assert captured["extra_secrets"] == {"TRACKIO_SPACE_ID": "alice/ml-intern-abcd1234"}


def test_hf_jobs_normalizes_trackio_space_id(monkeypatch):
    class FakeApi:
        def __init__(self):
            self.run_kwargs = None

        def run_job(self, **kwargs):
            self.run_kwargs = kwargs
            return SimpleNamespace(
                id="job-123",
                url="https://huggingface.co/jobs/job-123",
            )

    api = FakeApi()
    tool = HfJobsTool(hf_token="hf-token", namespace="alice")
    tool.api = api
    seeded_space_ids: list[str] = []

    async def fake_seed_trackio_dashboard(space_id):
        seeded_space_ids.append(space_id)

    async def fake_wait_for_job_completion(job_id, namespace=None):
        return "COMPLETED", ["done"]

    monkeypatch.setattr(tool, "_seed_trackio_dashboard", fake_seed_trackio_dashboard)
    monkeypatch.setattr(tool, "_wait_for_job_completion", fake_wait_for_job_completion)

    result = asyncio.run(
        tool.execute(
            {
                "operation": "run",
                "command": ["python", "-c", "print('ok')"],
                "trackio_space_id": _legacy_space_id(),
                "trackio_project": "demo",
            }
        )
    )

    assert result["totalResults"] == 1
    assert seeded_space_ids == ["alice/ml-intern-abcd1234"]
    assert api.run_kwargs["env"]["TRACKIO_SPACE_ID"] == "alice/ml-intern-abcd1234"
    assert api.run_kwargs["env"]["TRACKIO_PROJECT"] == "demo"
