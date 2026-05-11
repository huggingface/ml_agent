import io
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from huggingface_hub.errors import HfHubHTTPError
from starlette.datastructures import FormData

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


def _request(
    upload: UploadFile | None = None,
    headers: dict[str, str] | None = None,
):
    state = {"form_called": False}

    class FakeRequest:
        def __init__(self):
            self.headers = headers or {}
            self.cookies = {}

        async def form(self, **_kwargs):
            state["form_called"] = True
            if upload is None:
                raise AssertionError("request.form() should not be called")
            return FormData([("file", upload)])

    return FakeRequest(), state


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


def test_dataset_repo_card_exposes_each_upload_as_config():
    card = dataset_uploads.dataset_repo_card(
        "alice/ml-intern-s1-datasets",
        [
            "README.md",
            "uploads/oldabc/rows.jsonl",
            "uploads/oldabc/rows.jsonl",
            "uploads/newdef/table.csv",
        ],
    ).decode("utf-8")

    assert "configs:" in card
    assert "- config_name: upload_oldabc" in card
    assert '    path: "uploads/oldabc/rows.jsonl"' in card
    assert "- config_name: upload_newdef" in card
    assert '    path: "uploads/newdef/table.csv"' in card
    assert card.count("- config_name: upload_oldabc") == 1


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
            self.settings_calls = []
            self.list_calls = []
            self.upload_calls = []
            instances.append(self)

        def create_repo(self, **kwargs):
            self.create_calls.append(kwargs)

        def update_repo_settings(self, **kwargs):
            self.settings_calls.append(kwargs)

        def list_repo_files(self, **kwargs):
            self.list_calls.append(kwargs)
            return [
                "README.md",
                "uploads/oldupload/old.jsonl",
                "uploads/notes.txt",
            ]

        def upload_file(self, **kwargs):
            if kwargs["path_in_repo"] != "README.md":
                assert kwargs["path_or_fileobj"] == b"a,b\n1,2\n"
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
    assert api.settings_calls == [
        {
            "repo_id": "alice/ml-intern-12345678-datasets",
            "repo_type": "dataset",
            "private": True,
        }
    ]
    assert api.list_calls == [
        {
            "repo_id": "alice/ml-intern-12345678-datasets",
            "repo_type": "dataset",
        }
    ]
    assert [call["path_in_repo"] for call in api.upload_calls] == [
        "uploads/feedfacecafe/Data-Set.csv",
        "README.md",
    ]
    readme = api.upload_calls[1]["path_or_fileobj"].decode("utf-8")
    assert "- config_name: upload_oldupload" in readme
    assert '    path: "uploads/oldupload/old.jsonl"' in readme
    assert "- config_name: upload_feedfacecafe" in readme
    assert '    path: "uploads/feedfacecafe/Data-Set.csv"' in readme
    assert result.repo_id == "alice/ml-intern-12345678-datasets"
    assert result.config_name == "upload_feedfacecafe"
    assert result.format == "csv"
    assert result.load_dataset_snippet == (
        "from datasets import load_dataset\n\n"
        'dataset = load_dataset("alice/ml-intern-12345678-datasets", '
        '"upload_feedfacecafe", split="train", token=True)'
    )


@pytest.mark.asyncio
async def test_upload_route_requires_hf_token_without_parsing_upload(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    upload = _upload("rows.csv")
    close_state = _track_close(upload)
    request, request_state = _request(upload)

    async def fake_check_session_access(*_args, **_kwargs):
        return SimpleNamespace(
            is_active=True,
            is_processing=False,
            session=SimpleNamespace(pending_approval=None),
            hf_username="alice",
        )

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)

    try:
        with pytest.raises(HTTPException) as exc_info:
            await agent.upload_session_dataset(
                "s1",
                request,
                {"user_id": "u1", "username": "alice"},
            )

        assert exc_info.value.status_code == 401
        assert request_state["form_called"] is False
        assert close_state["closed"] is False
    finally:
        await upload.close()


@pytest.mark.asyncio
async def test_upload_route_rejects_content_length_before_parsing(monkeypatch):
    upload = _upload("rows.csv")
    close_state = _track_close(upload)
    request, request_state = _request(
        upload,
        headers={
            "content-length": str(
                dataset_uploads.MAX_DATASET_UPLOAD_BYTES
                + agent.DATASET_UPLOAD_MULTIPART_SLACK_BYTES
                + 1
            )
        },
    )

    async def fake_check_session_access(*_args, **_kwargs):
        raise AssertionError("session access should not run for oversized uploads")

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)

    try:
        with pytest.raises(HTTPException) as exc_info:
            await agent.upload_session_dataset(
                "s1",
                request,
                {
                    "user_id": "u1",
                    "username": "alice",
                    agent.INTERNAL_HF_TOKEN_KEY: "hf-token",
                },
            )

        assert exc_info.value.status_code == 413
        assert request_state["form_called"] is False
        assert close_state["closed"] is False
    finally:
        await upload.close()


@pytest.mark.asyncio
async def test_upload_route_rejects_busy_session_without_parsing_upload(monkeypatch):
    upload = _upload("rows.csv")
    close_state = _track_close(upload)
    request, request_state = _request(upload)

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
            request,
            {
                "user_id": "u1",
                "username": "alice",
                agent.INTERNAL_HF_TOKEN_KEY: "hf-token",
            },
        )

    assert exc_info.value.status_code == 409
    assert request_state["form_called"] is False
    assert close_state["closed"] is False
    await upload.close()


@pytest.mark.asyncio
async def test_upload_route_appends_context_note_and_persists(monkeypatch):
    upload = _upload("rows.jsonl", b'{"text":"hi"}\n')
    close_state = _track_close(upload)
    request, request_state = _request(upload)
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
        config_name="upload_abc123",
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
        request,
        {
            "user_id": "u1",
            "username": "alice",
            agent.INTERNAL_HF_TOKEN_KEY: "hf-token",
        },
    )

    assert response.repo_id == uploaded.repo_id
    assert response.config_name == uploaded.config_name
    assert response.path_in_repo == uploaded.path_in_repo
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content.startswith("[SYSTEM:")
    assert uploaded.config_name in messages[0].content
    assert uploaded.path_in_repo in messages[0].content
    assert persisted == [agent_session]
    assert request_state["form_called"] is True
    assert close_state["closed"] is True


@pytest.mark.asyncio
async def test_upload_route_closes_upload_when_hub_upload_fails(monkeypatch):
    upload = _upload("rows.csv")
    close_state = _track_close(upload)
    request, request_state = _request(upload)

    async def fake_check_session_access(*_args, **_kwargs):
        return SimpleNamespace(
            is_active=True,
            is_processing=False,
            session=SimpleNamespace(pending_approval=None),
            hf_username="alice",
        )

    async def fake_push_dataset_upload_to_hub(**_kwargs):
        raise RuntimeError("hub unavailable")

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent, "push_dataset_upload_to_hub", fake_push_dataset_upload_to_hub
    )

    with pytest.raises(HTTPException) as exc_info:
        await agent.upload_session_dataset(
            "s1",
            request,
            {
                "user_id": "u1",
                "username": "alice",
                agent.INTERNAL_HF_TOKEN_KEY: "hf-token",
            },
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Dataset upload failed. Please try again."
    assert request_state["form_called"] is True
    assert close_state["closed"] is True


@pytest.mark.asyncio
async def test_upload_route_maps_hub_permission_error_safely(monkeypatch):
    upload = _upload("rows.csv")
    close_state = _track_close(upload)
    request, request_state = _request(upload)

    async def fake_check_session_access(*_args, **_kwargs):
        return SimpleNamespace(
            is_active=True,
            is_processing=False,
            session=SimpleNamespace(pending_approval=None),
            hf_username="alice",
        )

    async def fake_push_dataset_upload_to_hub(**_kwargs):
        response = httpx.Response(
            403,
            request=httpx.Request("POST", "https://huggingface.co/api/datasets"),
            headers={"x-request-id": "req-123"},
        )
        raise HfHubHTTPError(
            "403 Forbidden: token hf_secret cannot write",
            response=response,
            server_message="token hf_secret cannot write",
        )

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)
    monkeypatch.setattr(
        agent, "push_dataset_upload_to_hub", fake_push_dataset_upload_to_hub
    )

    with pytest.raises(HTTPException) as exc_info:
        await agent.upload_session_dataset(
            "s1",
            request,
            {
                "user_id": "u1",
                "username": "alice",
                agent.INTERNAL_HF_TOKEN_KEY: "hf-token",
            },
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == (
        "Hugging Face denied permission to create or write to the dataset repo."
    )
    assert "hf_secret" not in exc_info.value.detail
    assert request_state["form_called"] is True
    assert close_state["closed"] is True
