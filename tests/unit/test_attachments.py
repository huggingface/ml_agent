from pathlib import Path

from agent.core.attachments import (
    AttachmentSource,
    build_user_content,
    create_context_manifest,
    import_dataset_batch,
    sanitize_filename,
)
from agent.main import _split_path_args


class FakeHfApi:
    def __init__(self):
        self.created = []
        self.uploads = []

    def whoami(self, token=None):
        return {"name": "alice"}

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)


def test_sanitize_filename_keeps_safe_basename():
    assert sanitize_filename("../../my data?.csv") == "my_data_.csv"
    assert sanitize_filename("   ") == "attachment"


def test_context_manifest_includes_metadata_and_text_preview(tmp_path: Path):
    path = tmp_path / "rows.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    manifest = create_context_manifest(
        [AttachmentSource(path=path)],
        scope_id="session-1",
    )

    assert manifest["type"] == "context_upload"
    item = manifest["items"][0]
    assert item["filename"] == "rows.csv"
    assert item["mime_type"] in {"text/csv", "application/vnd.ms-excel"}
    assert item["text_preview"] == "a,b\n1,2\n"
    assert item["placeholder"] == "[File #1]"


def test_build_user_content_adds_transient_image_part(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest = create_context_manifest(
        [AttachmentSource(path=image, kind="image")],
        scope_id="session-1",
    )

    content = build_user_content("describe this", [manifest])

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_import_dataset_batch_uploads_files_and_manifest(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hi"}\n', encoding="utf-8")
    api = FakeHfApi()

    manifest = import_dataset_batch(
        [AttachmentSource(path=data)],
        token="hf_test",
        scope_id="run-1",
        upload_id="upload-1",
        api=api,
    )

    assert api.created[0]["repo_id"] == "alice/ml-intern-user-datasets"
    assert api.created[0]["private"] is True
    assert manifest["repo_id"] == "alice/ml-intern-user-datasets"
    assert manifest["path_prefix"] == "sessions/run-1/upload-1"
    uploaded_paths = {upload["path_in_repo"] for upload in api.uploads}
    assert "sessions/run-1/upload-1/files/001-train.jsonl" in uploaded_paths
    assert "sessions/run-1/upload-1/manifest.json" in uploaded_paths


def test_cli_path_args_accept_repeated_and_comma_delimited():
    assert _split_path_args(["a.csv,b.csv", "c.png"]) == ["a.csv", "b.csv", "c.png"]

