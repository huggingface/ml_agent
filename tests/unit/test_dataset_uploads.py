import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import dataset_uploads  # noqa: E402
from routes import agent  # noqa: E402


def _upload(filename: str, content: bytes = b"a,b\n1,2\n") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


def _track_close(upload: UploadFile):
    state = {"closed": False}
    original_close = upload.close

    async def close():
        state["closed"] = True
        await original_close()

    upload.close = close
    return state


def test_sanitize_dataset_filename_strips_paths_and_unsafe_chars():
    assert (
        dataset_uploads.sanitize_dataset_filename("../../bad file (final).CSV")
        == "bad-file-final.csv"
    )
    assert dataset_uploads.sanitize_dataset_filename("") == "dataset.csv"


def test_dataset_format_rejects_unsupported_extension():
    with pytest.raises(HTTPException) as exc_info:
        dataset_uploads.dataset_format_from_filename("notes.txt")

    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException):
        dataset_uploads.dataset_format_from_filename("notes")


@pytest.mark.asyncio
async def test_validate_dataset_upload_rejects_size_over_limit(monkeypatch):
    monkeypatch.setattr(dataset_uploads, "MAX_DATASET_UPLOAD_BYTES", 3)
    upload = _upload("rows.csv", b"abcd")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await dataset_uploads.validate_dataset_upload(upload)
    finally:
        await upload.close()

    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_push_dataset_upload_creates_private_repo_and_uploads_file(monkeypatch):
    instances = []

    class FakeApi:
        def __init__(self, token):
            self.token = token
            self.create_calls = []
            self.upload_calls = []
            instances.append(self)

        def create_repo(self, **kwargs):
            self.create_calls.append(kwargs)

        def upload_file(self, **kwargs):
            if kwargs["path_in_repo"] != "README.md":
                assert kwargs["path_or_fileobj"].tell() == 0
                kwargs["path_or_fileobj"].read()
            self.upload_calls.append(kwargs)

    monkeypatch.setattr(dataset_uploads, "HfApi", FakeApi)
    monkeypatch.setattr(
        dataset_uploads.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="feedfacecafebeef"),
    )

    upload = _upload("../Data Set.CSV")
    try:
        result = await dataset_uploads.push_dataset_upload_to_hub(
            upload=upload,
            session_id="12345678-90ab-cdef-1234-567890abcdef",
            hf_username="alice",
            hf_token="hf-token",
        )
    finally:
        await upload.close()

    api = instances[0]
    assert api.token == "hf-token"
    assert api.create_calls == [
        {
            "repo_id": "alice/ml-intern-12345678-datasets",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert [call["path_in_repo"] for call in api.upload_calls] == [
        "README.md",
        "uploads/feedfacecafe/Data-Set.csv",
    ]
    assert result.repo_id == "alice/ml-intern-12345678-datasets"
    assert result.format == "csv"
    assert result.load_dataset_snippet.startswith("from datasets import load_dataset")


@pytest.mark.asyncio
async def test_upload_route_requires_hf_token_and_closes_upload(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    upload = _upload("rows.csv")
    close_state = _track_close(upload)

    async def fake_check_session_access(*_args, **_kwargs):
        return SimpleNamespace(
            is_active=True,
            is_processing=False,
            session=SimpleNamespace(pending_approval=None),
            hf_username="alice",
        )

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)

    with pytest.raises(HTTPException) as exc_info:
        await agent.upload_session_dataset(
            "s1",
            SimpleNamespace(headers={}, cookies={}),
            upload,
            {"user_id": "u1", "username": "alice"},
        )

    assert exc_info.value.status_code == 401
    assert close_state["closed"] is True


@pytest.mark.asyncio
async def test_upload_route_rejects_busy_session_and_closes_upload(monkeypatch):
    upload = _upload("rows.csv")
    close_state = _track_close(upload)

    async def fake_check_session_access(*_args, **_kwargs):
        return SimpleNamespace(
            is_active=True,
            is_processing=True,
            session=SimpleNamespace(pending_approval=None),
            hf_username="alice",
        )

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)

    with pytest.raises(HTTPException) as exc_info:
        await agent.upload_session_dataset(
            "s1",
            SimpleNamespace(headers={}, cookies={}),
            upload,
            {
                "user_id": "u1",
                "username": "alice",
                agent.INTERNAL_HF_TOKEN_KEY: "hf-token",
            },
        )

    assert exc_info.value.status_code == 409
    assert close_state["closed"] is True


@pytest.mark.asyncio
async def test_upload_route_appends_context_note_and_persists(monkeypatch):
    upload = _upload("rows.jsonl", b'{"text":"hi"}\n')
    close_state = _track_close(upload)
    messages = []
    persisted = []
    agent_session = SimpleNamespace(
        is_active=True,
        is_processing=False,
        session=SimpleNamespace(
            pending_approval=None,
            context_manager=SimpleNamespace(add_message=messages.append),
        ),
        hf_username="alice",
    )
    uploaded = dataset_uploads.DatasetUpload(
        session_id="s1",
        repo_id="alice/ml-intern-s1-datasets",
        repo_type="dataset",
        private=True,
        upload_id="abc123",
        filename="rows.jsonl",
        original_filename="rows.jsonl",
        path_in_repo="uploads/abc123/rows.jsonl",
        size_bytes=14,
        format="jsonl",
        hub_url="https://huggingface.co/datasets/alice/ml-intern-s1-datasets/blob/main/uploads/abc123/rows.jsonl",
        load_dataset_snippet='dataset = load_dataset("json")',
    )

    async def fake_check_session_access(*_args, **_kwargs):
        return agent_session

    async def fake_push_dataset_upload_to_hub(**kwargs):
        assert kwargs["upload"] is upload
        assert kwargs["hf_token"] == "hf-token"
        return uploaded

    async def fake_persist_session_snapshot(value):
        persisted.append(value)

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent, "push_dataset_upload_to_hub", fake_push_dataset_upload_to_hub
    )
    monkeypatch.setattr(
        agent.session_manager,
        "persist_session_snapshot",
        fake_persist_session_snapshot,
    )

    response = await agent.upload_session_dataset(
        "s1",
        SimpleNamespace(headers={}, cookies={}),
        upload,
        {
            "user_id": "u1",
            "username": "alice",
            agent.INTERNAL_HF_TOKEN_KEY: "hf-token",
        },
    )

    assert response.repo_id == uploaded.repo_id
    assert response.path_in_repo == uploaded.path_in_repo
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content.startswith("[SYSTEM:")
    assert uploaded.path_in_repo in messages[0].content
    assert persisted == [agent_session]
    assert close_state["closed"] is True
