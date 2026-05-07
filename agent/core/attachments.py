"""User-selected file and dataset attachment helpers.

The agent never gets ambient access to a user's laptop. Files enter the
conversation only through this deliberate attachment layer:

* context uploads are staged locally and summarized into the next turn;
* dataset imports are uploaded to a private HF dataset repo and represented by
  a manifest the agent can use for training/jobs.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
TEXT_PREVIEW_BYTES = 16 * 1024
STAGING_ROOT = Path(os.environ.get("ML_INTERN_UPLOAD_DIR", "/tmp/ml-intern-uploads"))


class AttachmentError(ValueError):
    """Readable validation/import error safe to surface to users."""


@dataclass(frozen=True)
class AttachmentSource:
    """A deliberate local file selected by CLI or web upload staging."""

    path: Path
    original_name: str | None = None
    kind: str = "file"


def sanitize_filename(name: str) -> str:
    """Return a conservative filename safe for HF repo paths."""
    base = Path(name).name.strip().replace("\x00", "")
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base)
    base = re.sub(r"\s+", "_", base).strip("._- ")
    return base[:180] or "attachment"


def _kind_for(path: Path, declared_kind: str = "file") -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    if declared_kind == "image" or mime.startswith("image/"):
        return "image", mime
    return "file", mime


def _validate_path(path: Path, *, max_bytes: int) -> int:
    if not path.exists():
        raise AttachmentError(f"Attachment not found: {path}")
    if not path.is_file():
        raise AttachmentError(f"Attachment is not a file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise AttachmentError(f"Attachment is empty: {path}")
    if size > max_bytes:
        mb = max_bytes // (1024 * 1024)
        raise AttachmentError(f"Attachment is too large: {path} ({size} bytes, max {mb} MB)")
    return size


def _text_preview(path: Path, mime_type: str) -> str | None:
    text_like = (
        mime_type.startswith("text/")
        or path.suffix.lower()
        in {
            ".csv",
            ".json",
            ".jsonl",
            ".md",
            ".py",
            ".txt",
            ".tsv",
            ".yaml",
            ".yml",
        }
    )
    if not text_like:
        return None
    raw = path.read_bytes()[:TEXT_PREVIEW_BYTES]
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    return text


def _file_item(path: Path, *, original_name: str | None, declared_kind: str) -> dict[str, Any]:
    size = _validate_path(path, max_bytes=DEFAULT_MAX_ATTACHMENT_BYTES)
    filename = sanitize_filename(original_name or path.name)
    kind, mime_type = _kind_for(path, declared_kind)
    item: dict[str, Any] = {
        "kind": kind,
        "filename": filename,
        "original_name": original_name or path.name,
        "size_bytes": size,
        "mime_type": mime_type,
    }
    preview = _text_preview(path, mime_type)
    if preview:
        item["text_preview"] = preview
        item["preview_truncated"] = size > TEXT_PREVIEW_BYTES
    return item


def create_context_manifest(
    sources: Iterable[AttachmentSource],
    *,
    scope_id: str,
    upload_id: str | None = None,
    copy_to_staging: bool = False,
) -> dict[str, Any]:
    """Create a local-only attachment manifest for one agent turn."""
    upload_id = upload_id or uuid.uuid4().hex
    prefix = STAGING_ROOT / scope_id / upload_id
    if copy_to_staging:
        prefix.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for idx, source in enumerate(sources, start=1):
        path = Path(source.path).expanduser()
        item = _file_item(path, original_name=source.original_name, declared_kind=source.kind)
        staged_path = path
        if copy_to_staging:
            staged_path = prefix / f"{idx:03d}-{item['filename']}"
            shutil.copyfile(path, staged_path)
        item["path"] = str(staged_path)
        item["placeholder"] = f"[{'Image' if item['kind'] == 'image' else 'File'} #{idx}]"
        items.append(item)

    if not items:
        raise AttachmentError("No attachments were provided.")

    manifest = {
        "type": "context_upload",
        "upload_id": upload_id,
        "scope_id": scope_id,
        "items": items,
    }
    if copy_to_staging:
        manifest_path = prefix / "manifest.json"
        manifest_path.write_text(json.dumps(_manifest_without_previews(manifest), indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_context_manifest(scope_id: str, upload_id: str) -> dict[str, Any]:
    manifest_path = STAGING_ROOT / scope_id / upload_id / "manifest.json"
    if not manifest_path.exists():
        raise AttachmentError(f"Upload not found: {upload_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Rehydrate previews from staged files; they are intentionally not persisted
    # to the JSON manifest to keep local history compact.
    for item in manifest.get("items", []):
        path = Path(item.get("path", ""))
        if path.exists():
            preview = _text_preview(path, item.get("mime_type") or "")
            if preview:
                item["text_preview"] = preview
                item["preview_truncated"] = path.stat().st_size > TEXT_PREVIEW_BYTES
    return manifest


def _manifest_without_previews(manifest: dict[str, Any]) -> dict[str, Any]:
    clean = dict(manifest)
    clean["items"] = [
        {k: v for k, v in item.items() if k not in {"text_preview"}}
        for item in manifest.get("items", [])
    ]
    return clean


def default_dataset_repo_id(username: str) -> str:
    return f"{username}/ml-intern-user-datasets"


def _repo_username(api: Any, token: str | None) -> str:
    whoami = api.whoami(token=token) if token else api.whoami()
    username = whoami.get("name") or whoami.get("fullname")
    if not username:
        raise AttachmentError("Could not resolve Hugging Face username for dataset import.")
    return username


def import_dataset_batch(
    sources: Iterable[AttachmentSource],
    *,
    token: str | None,
    scope_id: str,
    upload_id: str | None = None,
    repo_id: str | None = None,
    api: Any | None = None,
) -> dict[str, Any]:
    """Upload files plus manifest.json to a private HF dataset repo."""
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - import guard
        raise AttachmentError("huggingface_hub is required for dataset imports.") from exc

    api = api or HfApi(token=token)
    if repo_id is None:
        repo_id = default_dataset_repo_id(_repo_username(api, token))
    upload_id = upload_id or uuid.uuid4().hex
    path_prefix = f"sessions/{sanitize_filename(scope_id)}/{upload_id}"

    staged = create_context_manifest(
        sources,
        scope_id=scope_id,
        upload_id=upload_id,
        copy_to_staging=False,
    )
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
    items: list[dict[str, Any]] = []
    for idx, item in enumerate(staged["items"], start=1):
        source_path = item["path"]
        filename = f"{idx:03d}-{item['filename']}"
        path_in_repo = f"{path_prefix}/files/{filename}"
        api.upload_file(
            path_or_fileobj=source_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Add ML Intern dataset file {filename}",
        )
        clean_item = {k: v for k, v in item.items() if k not in {"path", "text_preview"}}
        clean_item["path_in_repo"] = path_in_repo
        items.append(clean_item)

    manifest = {
        "type": "dataset_import",
        "upload_id": upload_id,
        "scope_id": scope_id,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "path_prefix": path_prefix,
        "manifest_path": f"{path_prefix}/manifest.json",
        "items": items,
    }
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(manifest, tmp, indent=2)
        tmp_path = tmp.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=manifest["manifest_path"],
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add ML Intern dataset manifest",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return manifest


def attachment_note(manifests: Iterable[dict[str, Any]]) -> str:
    """Build a text note injected into the model turn."""
    lines = ["\n\n[Attached context]"]
    for manifest in manifests:
        if manifest.get("type") == "dataset_import":
            lines.append(
                "- Imported dataset batch: "
                f"repo={manifest.get('repo_id')} "
                f"path_prefix={manifest.get('path_prefix')} "
                f"manifest={manifest.get('manifest_path')}. "
                "Use this HF dataset path for training, jobs, and durable reuse."
            )
        else:
            lines.append(f"- Local per-turn attachment batch: upload_id={manifest.get('upload_id')}.")
        for item in manifest.get("items", []):
            lines.append(
                f"  {item.get('placeholder', '')} {item.get('filename')} "
                f"({item.get('mime_type')}, {item.get('size_bytes')} bytes)"
            )
            preview = item.get("text_preview")
            if preview:
                suffix = "\n  [preview truncated]" if item.get("preview_truncated") else ""
                lines.append(f"  Preview:\n```text\n{preview}\n```{suffix}")
    lines.append(
        "Only use files explicitly listed above. Local per-turn files are not durable; "
        "ask the user to import them as a dataset if an HF Job or later turn needs full access."
    )
    return "\n".join(lines)


def build_user_content(text: str, manifests: Iterable[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Return LiteLLM-compatible user content with transient image parts."""
    manifest_list = list(manifests)
    note = attachment_note(manifest_list) if manifest_list else ""
    text_part = (text or "").rstrip() + note
    parts: list[dict[str, Any]] = [{"type": "text", "text": text_part}]
    for manifest in manifest_list:
        if manifest.get("type") == "dataset_import":
            continue
        for item in manifest.get("items", []):
            if item.get("kind") != "image":
                continue
            path = Path(item.get("path", ""))
            if not path.exists():
                continue
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            mime_type = item.get("mime_type") or "image/png"
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}})
    return parts if len(parts) > 1 else text_part
