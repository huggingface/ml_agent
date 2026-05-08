import logging
from types import SimpleNamespace

import pytest

from agent.core import hub_artifacts
from agent.core.hub_artifacts import (
    ML_INTERN_TAG,
    PROVENANCE_MARKER,
    artifact_collection_title,
    augment_repo_card_content,
    build_hub_artifact_sitecustomize,
    is_known_hub_artifact,
    is_sandbox_hub_repo,
    register_hub_artifact,
    remember_hub_artifact,
    wrap_shell_command_with_hub_artifact_bootstrap,
)
from agent.tools import local_tools, sandbox_tool
from agent.tools.hf_repo_files_tool import HfRepoFilesTool
from agent.tools.hf_repo_git_tool import HfRepoGitTool
from agent.tools.jobs_tool import _wrap_command_with_artifact_bootstrap


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-123",
        session_start_time="2026-05-05T10:20:30",
    )


def test_artifact_collection_title_uses_session_date_and_id():
    assert (
        artifact_collection_title(_session())
        == "ml-intern-artifacts-2026-05-05-session-123"
    )


def test_artifact_collection_title_uses_short_uuid_fragment():
    session = SimpleNamespace(
        session_id="fadcbc77-3439-4c2b-bc52-50d7f6353af3",
        session_start_time="2026-05-05T10:20:30",
    )

    title = artifact_collection_title(session)

    assert title == "ml-intern-artifacts-2026-05-05-fadcbc77"
    assert len(title) < 60


def test_artifact_collection_title_still_truncates_long_non_uuid_ids():
    session = SimpleNamespace(
        session_id="custom-session-id-that-is-longer-than-the-hub-title-limit",
        session_start_time="2026-05-05T10:20:30",
    )

    title = artifact_collection_title(session)

    assert title.startswith("ml-intern-artifacts-2026-05-05-custom-session-id")
    assert len(title) < 60


def test_model_card_merges_tags_and_appends_provenance_and_usage():
    content = """---
license: apache-2.0
tags:
- text-generation
---
# Existing Model

Existing details stay here.
"""

    updated = augment_repo_card_content(content, "alice/model", "model")
    second_pass = augment_repo_card_content(updated, "alice/model", "model")

    assert "license: apache-2.0" in updated
    assert "- text-generation" in updated
    assert f"- {ML_INTERN_TAG}" in updated
    assert "# Existing Model" in updated
    assert "Existing details stay here." in updated
    assert PROVENANCE_MARKER in updated
    assert "AutoModelForCausalLM" in updated
    assert second_pass.count(PROVENANCE_MARKER) == 1
    assert second_pass.count("AutoModelForCausalLM") == updated.count(
        "AutoModelForCausalLM"
    )


def test_dataset_card_adds_load_dataset_usage():
    updated = augment_repo_card_content("", "alice/dataset", "dataset")

    assert f"- {ML_INTERN_TAG}" in updated
    assert "# alice/dataset" in updated
    assert "from datasets import load_dataset" in updated
    assert 'load_dataset("alice/dataset")' in updated


def test_existing_usage_section_is_preserved_without_duplicate_usage():
    content = """# Existing Dataset

## Usage

Use the custom loader in this repository.
"""

    updated = augment_repo_card_content(content, "alice/dataset", "dataset")

    assert "Use the custom loader in this repository." in updated
    assert "from datasets import load_dataset" not in updated
    assert PROVENANCE_MARKER in updated


def test_space_card_gets_metadata_without_provenance_body():
    updated = augment_repo_card_content("# Existing Space\n", "alice/space", "space")

    assert f"- {ML_INTERN_TAG}" in updated
    assert "# Existing Space" in updated
    assert PROVENANCE_MARKER not in updated


def test_register_hub_artifact_creates_private_collection_and_adds_item_once(
    monkeypatch,
):
    session = _session()

    class FakeApi:
        token = "hf-token"

        def __init__(self):
            self.created_collections = []
            self.collection_items = []
            self.uploads = []

        def create_collection(self, **kwargs):
            self.created_collections.append(kwargs)
            return SimpleNamespace(slug="alice/ml-intern-artifacts")

        def add_collection_item(self, **kwargs):
            self.collection_items.append(kwargs)

        def upload_file(self, **kwargs):
            self.uploads.append(kwargs)

    api = FakeApi()
    monkeypatch.setattr(hub_artifacts, "_read_remote_readme", lambda *_, **__: "")

    assert register_hub_artifact(api, "alice/model", "model", session=session)
    assert register_hub_artifact(api, "alice/model", "model", session=session)

    assert is_known_hub_artifact(session, "alice/model", "model")
    assert len(api.created_collections) == 1
    assert api.created_collections[0]["title"] == artifact_collection_title(session)
    assert api.created_collections[0]["private"] is True
    assert len(api.collection_items) == 1
    assert api.collection_items[0]["item_id"] == "alice/model"
    assert api.collection_items[0]["item_type"] == "model"
    assert api.collection_items[0]["exists_ok"] is True
    assert len(api.uploads) == 1
    assert b"ml-intern" in api.uploads[0]["path_or_fileobj"]


def test_register_hub_artifact_skips_sandbox_spaces(monkeypatch):
    session = _session()
    api = SimpleNamespace(token="hf-token")
    calls = []

    monkeypatch.setattr(
        hub_artifacts,
        "_update_repo_card",
        lambda *args, **kwargs: calls.append(("card", args, kwargs)),
    )
    monkeypatch.setattr(
        hub_artifacts,
        "_add_to_collection",
        lambda *args, **kwargs: calls.append(("collection", args, kwargs)),
    )

    assert is_sandbox_hub_repo("alice/sandbox-1234abcd", "space")
    assert not is_sandbox_hub_repo("alice/sandbox-1234abcd", "model")
    assert not is_sandbox_hub_repo("alice/demo-space", "space")
    assert not register_hub_artifact(
        api,
        "alice/sandbox-1234abcd",
        "space",
        session=session,
    )
    assert not is_known_hub_artifact(session, "alice/sandbox-1234abcd", "space")
    assert calls == []


def test_register_hub_artifact_retries_after_partial_failure(monkeypatch):
    session = _session()
    api = SimpleNamespace(token="hf-token")
    card_attempts = 0
    collection_attempts = 0

    def flaky_update_repo_card(*args, **kwargs):
        nonlocal card_attempts
        card_attempts += 1
        if card_attempts == 1:
            raise RuntimeError("temporary card failure")

    def add_to_collection(*args, **kwargs):
        nonlocal collection_attempts
        collection_attempts += 1
        return True

    monkeypatch.setattr(
        hub_artifacts,
        "_update_repo_card",
        flaky_update_repo_card,
    )
    monkeypatch.setattr(hub_artifacts, "_add_to_collection", add_to_collection)

    assert not register_hub_artifact(api, "alice/model", "model", session=session)
    assert register_hub_artifact(api, "alice/model", "model", session=session)
    assert register_hub_artifact(api, "alice/model", "model", session=session)

    assert card_attempts == 2
    assert collection_attempts == 2


def test_register_hub_artifact_retries_after_collection_failure(monkeypatch):
    session = _session()
    api = SimpleNamespace(token="hf-token")
    card_attempts = 0
    collection_attempts = 0

    def update_repo_card(*args, **kwargs):
        nonlocal card_attempts
        card_attempts += 1

    def flaky_add_to_collection(*args, **kwargs):
        nonlocal collection_attempts
        collection_attempts += 1
        if collection_attempts == 1:
            raise RuntimeError("temporary collection failure")
        return True

    monkeypatch.setattr(hub_artifacts, "_update_repo_card", update_repo_card)
    monkeypatch.setattr(
        hub_artifacts,
        "_add_to_collection",
        flaky_add_to_collection,
    )

    assert not register_hub_artifact(api, "alice/model", "model", session=session)
    assert register_hub_artifact(api, "alice/model", "model", session=session)
    assert register_hub_artifact(api, "alice/model", "model", session=session)

    assert card_attempts == 2
    assert collection_attempts == 2


def test_session_artifact_set_falls_back_when_session_rejects_attrs(caplog):
    class SlottedSession:
        __slots__ = ("session_id", "session_start_time")

        def __init__(self):
            self.session_id = "session-123"
            self.session_start_time = "2026-05-05T10:20:30"

    session = SlottedSession()

    with caplog.at_level(logging.WARNING):
        remember_hub_artifact(session, "alice/model", "model")

    assert is_known_hub_artifact(session, "alice/model", "model")
    assert "using process-local fallback state" in caplog.text


@pytest.mark.asyncio
async def test_hf_repo_git_create_repo_registers_artifact(monkeypatch):
    session = _session()
    calls = []

    class FakeApi:
        token = "hf-token"

        def create_repo(self, **kwargs):
            self.create_kwargs = kwargs
            return "https://huggingface.co/spaces/alice/demo"

    def fake_register(api, repo_id, repo_type, **kwargs):
        calls.append((api, repo_id, repo_type, kwargs))
        return True

    monkeypatch.setattr(
        "agent.tools.hf_repo_git_tool.register_hub_artifact",
        fake_register,
    )
    tool = HfRepoGitTool(hf_token="hf-token", session=session)
    tool.api = FakeApi()

    result = await tool._create_repo(
        {
            "repo_id": "alice/demo",
            "repo_type": "space",
            "space_sdk": "gradio",
            "private": True,
        }
    )

    assert result["totalResults"] == 1
    assert calls == [
        (
            tool.api,
            "alice/demo",
            "space",
            {"session": session, "extra_metadata": {"sdk": "gradio"}},
        )
    ]


@pytest.mark.asyncio
async def test_hf_repo_files_upload_registers_known_artifact_with_force(monkeypatch):
    session = _session()
    calls = []
    uploads = []

    class FakeApi:
        token = "hf-token"

        def upload_file(self, **kwargs):
            uploads.append(kwargs)
            return SimpleNamespace()

    def fake_register(api, repo_id, repo_type, **kwargs):
        calls.append((api, repo_id, repo_type, kwargs))
        return True

    monkeypatch.setattr(
        "agent.tools.hf_repo_files_tool.register_hub_artifact",
        fake_register,
    )
    remember_hub_artifact(session, "alice/model", "model")

    tool = HfRepoFilesTool(hf_token="hf-token", session=session)
    tool.api = FakeApi()

    result = await tool._upload(
        {
            "repo_id": "alice/model",
            "repo_type": "model",
            "path": "weights.bin",
            "content": b"weights",
        }
    )
    readme_result = await tool._upload(
        {
            "repo_id": "alice/model",
            "repo_type": "model",
            "path": "README.md",
            "content": "# Model",
        }
    )

    assert result["totalResults"] == 1
    assert readme_result["totalResults"] == 1
    assert [upload["path_in_repo"] for upload in uploads] == [
        "weights.bin",
        "README.md",
    ]
    assert calls == [
        (
            tool.api,
            "alice/model",
            "model",
            {"session": session, "force": False},
        ),
        (
            tool.api,
            "alice/model",
            "model",
            {"session": session, "force": True},
        ),
    ]


def test_hf_jobs_artifact_bootstrap_wraps_command_without_changing_exec_target():
    command = ["uv", "run", "train.py"]
    wrapped = _wrap_command_with_artifact_bootstrap(command, _session())

    assert wrapped[0:2] == ["/bin/sh", "-lc"]
    assert "sitecustomize.py" in wrapped[2]
    assert "PYTHONPATH" in wrapped[2]
    assert "exec uv run train.py" in wrapped[2]
    assert _wrap_command_with_artifact_bootstrap(command, None) == command


def test_shell_bootstrap_wraps_capybara_push_to_hub_pattern():
    command = (
        "pip install -q datasets huggingface_hub && python -c "
        "\"subset.push_to_hub('lewtun/Capybara-100', private=False)\""
    )

    wrapped = wrap_shell_command_with_hub_artifact_bootstrap(command, _session())

    assert "sitecustomize.py" in wrapped
    assert "PYTHONPATH" in wrapped
    assert command in wrapped
    assert wrap_shell_command_with_hub_artifact_bootstrap(command, None) == command
    assert (
        wrap_shell_command_with_hub_artifact_bootstrap(
            command,
            SimpleNamespace(session_start_time="2026-05-05T10:20:30"),
        )
        == command
    )


@pytest.mark.asyncio
async def test_sandbox_bash_wraps_command_for_session_artifact_hooks():
    calls = []

    class FakeSandbox:
        def call_tool(self, name, args):
            calls.append((name, args))
            return SimpleNamespace(success=True, output="ok", error="")

    session = _session()
    session.sandbox = FakeSandbox()

    handler = sandbox_tool._make_tool_handler("bash")
    output, ok = await handler({"command": "python make_dataset.py"}, session=session)

    assert ok is True
    assert output == "ok"
    assert calls[0][0] == "bash"
    assert "sitecustomize.py" in calls[0][1]["command"]
    assert "python make_dataset.py" in calls[0][1]["command"]


@pytest.mark.asyncio
async def test_local_bash_wraps_command_for_session_artifact_hooks(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(local_tools.subprocess, "run", fake_run)

    output, ok = await local_tools._bash_handler(
        {"command": "python make_dataset.py"},
        session=_session(),
    )

    assert ok is True
    assert output == "ok"
    assert "sitecustomize.py" in seen["command"]
    assert "python make_dataset.py" in seen["command"]


def test_sitecustomize_bootstrap_is_valid_python():
    code = build_hub_artifact_sitecustomize(_session())

    compile(code, "sitecustomize.py", "exec")
    assert "ml-intern-artifacts-2026-05-05-session-123" in code


def test_sitecustomize_bootstrap_reuses_existing_collection_slug():
    session = _session()
    setattr(
        session,
        hub_artifacts._COLLECTION_SLUG_ATTR,
        "alice/ml-intern-artifacts-2026-05-05-session-123",
    )

    code = build_hub_artifact_sitecustomize(session)

    compile(code, "sitecustomize.py", "exec")
    assert (
        "collection_slug = 'alice/ml-intern-artifacts-2026-05-05-session-123'" in code
    )


def test_sitecustomize_skips_sandbox_space_registration(monkeypatch):
    import huggingface_hub as hub
    from huggingface_hub import HfApi

    uploads = []
    downloads = []
    collection_creates = []
    collection_items = []

    for name in ("create_repo", "upload_folder", "create_commit"):
        if hasattr(HfApi, name):
            monkeypatch.setattr(HfApi, name, getattr(HfApi, name))
        if hasattr(hub, name):
            monkeypatch.setattr(hub, name, getattr(hub, name))

    def fake_upload_file(self, **kwargs):
        uploads.append(kwargs)
        return SimpleNamespace()

    def fake_hf_hub_download(*args, **kwargs):
        downloads.append((args, kwargs))
        raise RuntimeError("sandbox metadata update should be skipped")

    def fake_create_collection(self, **kwargs):
        collection_creates.append(kwargs)
        return SimpleNamespace(slug="alice/ml-intern-artifacts")

    def fake_add_collection_item(self, **kwargs):
        collection_items.append(kwargs)

    monkeypatch.setattr(HfApi, "upload_file", fake_upload_file)
    monkeypatch.setattr(HfApi, "create_collection", fake_create_collection)
    monkeypatch.setattr(HfApi, "add_collection_item", fake_add_collection_item)
    monkeypatch.setattr(hub, "upload_file", getattr(hub, "upload_file"))
    monkeypatch.setattr(hub, "hf_hub_download", fake_hf_hub_download)

    exec(build_hub_artifact_sitecustomize(_session()), {})
    assert HfApi.upload_file is not fake_upload_file

    HfApi(token="hf-token").upload_file(
        path_or_fileobj=b"app",
        path_in_repo="app.py",
        repo_id="alice/normal-space",
        repo_type="space",
        token="hf-token",
    )

    assert downloads[0][1]["repo_id"] == "alice/normal-space"
    assert len(collection_creates) == 1
    assert collection_items[0]["item_id"] == "alice/normal-space"

    uploads.clear()
    downloads.clear()
    collection_creates.clear()
    collection_items.clear()

    HfApi(token="hf-token").upload_file(
        path_or_fileobj=b"app",
        path_in_repo="app.py",
        repo_id="alice/sandbox-1234abcd",
        repo_type="space",
        token="hf-token",
    )

    assert [upload["repo_id"] for upload in uploads] == ["alice/sandbox-1234abcd"]
    assert downloads == []
    assert collection_creates == []
    assert collection_items == []
