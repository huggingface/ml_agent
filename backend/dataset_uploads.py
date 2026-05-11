"""Helpers for session-scoped dataset uploads to the Hugging Face Hub."""

import asyncio
import os
import re
import uuid
from dataclasses import dataclass
from urllib.parse import quote

from fastapi import HTTPException, UploadFile
from huggingface_hub import HfApi

MAX_DATASET_UPLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_DATASET_EXTENSIONS = {"csv", "json", "jsonl"}
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


@dataclass(frozen=True)
class DatasetUpload:
    session_id: str
    repo_id: str
    repo_type: str
    private: bool
    upload_id: str
    filename: str
    original_filename: str
    path_in_repo: str
    size_bytes: int
    format: str
    hub_url: str
    load_dataset_snippet: str

    def response_payload(self) -> dict[str, str | int | bool]:
        return {
            "session_id": self.session_id,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "private": self.private,
            "upload_id": self.upload_id,
            "filename": self.filename,
            "path_in_repo": self.path_in_repo,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "hub_url": self.hub_url,
            "load_dataset_snippet": self.load_dataset_snippet,
        }


def sanitize_dataset_filename(filename: str | None) -> str:
    """Return a Hub-safe basename while preserving the extension."""
    raw = os.path.basename(filename or "").strip()
    if not raw:
        raw = "dataset.csv"

    safe = _SAFE_FILENAME_RE.sub("-", raw).strip(".-_")
    if not safe:
        safe = "dataset.csv"

    stem, ext = os.path.splitext(safe)
    if not stem:
        stem = "dataset"
    if not ext:
        ext = ".csv"

    max_stem_len = 96 - len(ext)
    stem = stem[:max_stem_len].strip(".-_") or "dataset"
    return f"{stem}{ext.lower()}"


def display_filename(filename: str | None, fallback: str) -> str:
    raw = os.path.basename(filename or "").strip()
    if not raw:
        return fallback
    cleaned = "".join(char for char in raw if ord(char) >= 32)
    return cleaned[:160] or fallback


def dataset_format_from_filename(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in ALLOWED_DATASET_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .csv, .json, and .jsonl dataset files are supported.",
        )
    return ext


def session_dataset_repo_id(hf_username: str | None, session_id: str) -> str:
    namespace = (hf_username or "").strip()
    if not namespace or not _SAFE_NAMESPACE_RE.fullmatch(namespace):
        raise HTTPException(
            status_code=400,
            detail="Could not determine a valid Hugging Face namespace.",
        )

    safe_session_id = re.sub(r"[^A-Za-z0-9]+", "-", session_id).strip("-")
    if not safe_session_id:
        safe_session_id = uuid.uuid4().hex[:8]
    return f"{namespace}/ml-intern-{safe_session_id[:8]}-datasets"


async def upload_size_bytes(upload: UploadFile) -> int:
    await asyncio.to_thread(upload.file.seek, 0, os.SEEK_END)
    size = await asyncio.to_thread(upload.file.tell)
    await asyncio.to_thread(upload.file.seek, 0)
    return int(size)


async def validate_dataset_upload(upload: UploadFile) -> tuple[str, str, int]:
    dataset_format = dataset_format_from_filename(upload.filename or "")
    safe_filename = sanitize_dataset_filename(upload.filename)
    size = await upload_size_bytes(upload)
    if size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded dataset file is empty.")
    if size > MAX_DATASET_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Dataset upload exceeds the 100 MB limit.",
        )
    return safe_filename, dataset_format, size


def dataset_hub_url(repo_id: str, path_in_repo: str) -> str:
    quoted_path = quote(path_in_repo, safe="/")
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{quoted_path}"


def load_dataset_snippet(repo_id: str, path_in_repo: str, dataset_format: str) -> str:
    builder = "csv" if dataset_format == "csv" else "json"
    data_file = f"hf://datasets/{repo_id}/{path_in_repo}"
    return (
        "from datasets import load_dataset\n\n"
        f'dataset = load_dataset("{builder}", data_files="{data_file}", '
        'split="train", token=True)'
    )


def dataset_repo_card(repo_id: str) -> bytes:
    content = f"""---
tags:
- ml-intern
- uploaded-dataset
---

# {repo_id}

Private dataset files uploaded through ML Intern.

Files are stored under `uploads/<upload_id>/` and are attached to the
corresponding ML Intern session context by Hub reference, not by copying file
contents into the chat.
"""
    return content.encode("utf-8")


def dataset_context_note(upload: DatasetUpload) -> str:
    return f"""[SYSTEM: The user uploaded a dataset file for this session.

Use this Hugging Face Hub dataset reference when the task needs the uploaded data.
Do not look for the uploaded file on local disk and do not ask the user to
upload it again unless this Hub reference fails.

- Repo ID: {upload.repo_id}
- Repo type: dataset
- File in repo: {upload.path_in_repo}
- Original filename: {upload.original_filename}
- Stored filename: {upload.filename}
- Format: {upload.format}
- Size: {upload.size_bytes} bytes
- Hub URL: {upload.hub_url}

Load it with:
```python
{upload.load_dataset_snippet}
```
]"""


async def push_dataset_upload_to_hub(
    *,
    upload: UploadFile,
    session_id: str,
    hf_username: str,
    hf_token: str,
) -> DatasetUpload:
    safe_filename, dataset_format, size = await validate_dataset_upload(upload)
    original_filename = display_filename(upload.filename, safe_filename)
    upload_id = uuid.uuid4().hex[:12]
    repo_id = session_dataset_repo_id(hf_username, session_id)
    path_in_repo = f"uploads/{upload_id}/{safe_filename}"
    hub_url = dataset_hub_url(repo_id, path_in_repo)
    snippet = load_dataset_snippet(repo_id, path_in_repo, dataset_format)
    api = HfApi(token=hf_token)

    await asyncio.to_thread(
        api.create_repo,
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    await asyncio.to_thread(
        api.update_repo_settings,
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
    )
    await asyncio.to_thread(
        api.upload_file,
        path_or_fileobj=dataset_repo_card(repo_id),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update ML Intern dataset upload README",
    )
    await asyncio.to_thread(upload.file.seek, 0)
    await asyncio.to_thread(
        api.upload_file,
        path_or_fileobj=upload.file,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload dataset file {safe_filename}",
    )

    return DatasetUpload(
        session_id=session_id,
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        upload_id=upload_id,
        filename=safe_filename,
        original_filename=original_filename,
        path_in_repo=path_in_repo,
        size_bytes=size,
        format=dataset_format,
        hub_url=hub_url,
        load_dataset_snippet=snippet,
    )
