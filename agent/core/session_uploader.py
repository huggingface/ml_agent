#!/usr/bin/env python3
"""
Standalone script for uploading session trajectories to HuggingFace.
This runs as a separate process to avoid blocking the main agent.
Uses individual file uploads to avoid race conditions.

Two formats are supported:

* ``row`` — single-line JSONL row used by the existing org telemetry/KPI
  pipeline (``smolagents/ml-intern-sessions``). Compatible with
  ``backend/kpis_scheduler.py``.
* ``claude_code`` — one event per line in the Claude Code JSONL schema,
  auto-detected by the HF Agent Trace Viewer
  (https://huggingface.co/changelog/agent-trace-viewer). Used for the
  per-user private dataset (default ``{hf_user}/ml-intern-sessions``).
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Token resolution for the org KPI dataset. Fallback chain (least-privilege
# first) — matches backend/kpis_scheduler.py so one write-scoped token on the
# Space covers every telemetry dataset. Never hardcode tokens in source.
_ORG_TOKEN_FALLBACK_CHAIN = (
    "HF_SESSION_UPLOAD_TOKEN",
    "HF_TOKEN",
    "HF_ADMIN_TOKEN",
)
_PERSONAL_TOKEN_ENV = "_ML_INTERN_PERSONAL_TOKEN"


def _resolve_token(token_env: str | None) -> str:
    """Resolve an HF token from env. ``token_env`` overrides the fallback chain."""
    if token_env == "HF_TOKEN":
        try:
            from agent.core.hf_tokens import resolve_hf_token

            return (
                resolve_hf_token(
                    os.environ.get(_PERSONAL_TOKEN_ENV),
                    os.environ.get("HF_TOKEN"),
                )
                or ""
            )
        except Exception:
            token = os.environ.get(_PERSONAL_TOKEN_ENV) or os.environ.get("HF_TOKEN")
            return token or ""

    if token_env:
        return os.environ.get(token_env, "") or ""
    for var in _ORG_TOKEN_FALLBACK_CHAIN:
        val = os.environ.get(var)
        if val:
            return val
    return ""


def _scrub(obj: Any) -> Any:
    """Best-effort regex scrub for HF tokens / API keys before upload."""
    try:
        from agent.core.redact import scrub  # type: ignore
    except Exception:
        # Fallback for environments where the agent package isn't importable
        # (shouldn't happen in our subprocess, but be defensive).
        import importlib.util

        _spec = importlib.util.spec_from_file_location(
            "_redact",
            Path(__file__).parent / "redact.py",
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)  # type: ignore
        scrub = _mod.scrub
    return scrub(obj)


def _msg_uuid(session_id: str, role: str, idx: int) -> str:
    """Deterministic UUID-shaped id for a Claude Code message.

    Uses sha1 of ``session_id::role::idx`` so re-uploads/heartbeats keep the
    parent/child chain stable. Same convention as the example dataset
    https://huggingface.co/datasets/clem/hf-coding-tools-traces.
    """
    digest = hashlib.sha1(f"{session_id}::{role}::{idx}".encode("utf-8")).hexdigest()
    # Format like a UUID for visual familiarity (32 hex chars w/ dashes).
    return (
        f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-"
        f"{digest[16:20]}-{digest[20:32]}"
    )


def _content_to_text(content: Any) -> str:
    """Best-effort flatten of a litellm/openai content field to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    # Unknown content block — keep round-trippable representation.
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _parse_tool_args(raw: Any) -> Any:
    """Tool call arguments arrive as a JSON-encoded string from LLMs."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": raw}
    return raw


def to_claude_code_jsonl(trajectory: dict) -> list[dict]:
    """Convert an internal trajectory dict to Claude Code JSONL events.

    Schema reference (per the HF Agent Trace Viewer auto-detector):

        {"type":"user","message":{"role":"user","content":"..."},
         "uuid":"...","parentUuid":null,"sessionId":"...","timestamp":"..."}
        {"type":"assistant",
         "message":{"role":"assistant","model":"...",
                     "content":[{"type":"text","text":"..."},
                                {"type":"tool_use","id":"...","name":"...","input":{...}}]},
         "uuid":"...","parentUuid":"<prev>","sessionId":"...","timestamp":"..."}
        {"type":"user","message":{"role":"user",
                                  "content":[{"type":"tool_result",
                                              "tool_use_id":"...","content":"..."}]},
         "uuid":"...","parentUuid":"<prev>","sessionId":"...","timestamp":"..."}

    System messages are skipped (they're not part of the viewer schema and
    contain large prompts that pollute the trace viewer UI).
    """
    session_id = trajectory["session_id"]
    model_name = trajectory.get("model_name") or ""
    fallback_timestamp = (
        trajectory.get("session_start_time") or datetime.now().isoformat()
    )
    messages: list[dict] = trajectory.get("messages") or []

    out: list[dict] = []
    parent_uuid: str | None = None

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue
        timestamp = msg.get("timestamp") or fallback_timestamp

        if role == "user":
            content = _content_to_text(msg.get("content"))
            event_uuid = _msg_uuid(session_id, "user", idx)
            out.append(
                {
                    "type": "user",
                    "message": {"role": "user", "content": content},
                    "uuid": event_uuid,
                    "parentUuid": parent_uuid,
                    "sessionId": session_id,
                    "timestamp": timestamp,
                }
            )
            parent_uuid = event_uuid

        elif role == "assistant":
            content_text = _content_to_text(msg.get("content"))
            content_blocks: list[dict] = []
            if content_text:
                content_blocks.append({"type": "text", "text": content_text})
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": _parse_tool_args(fn.get("arguments")),
                    }
                )
            if not content_blocks:
                # Edge case: empty assistant turn (shouldn't normally happen,
                # but skip rather than emit an empty content array which
                # confuses the viewer).
                continue
            event_uuid = _msg_uuid(session_id, "assistant", idx)
            out.append(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": model_name,
                        "content": content_blocks,
                    },
                    "uuid": event_uuid,
                    "parentUuid": parent_uuid,
                    "sessionId": session_id,
                    "timestamp": timestamp,
                }
            )
            parent_uuid = event_uuid

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id") or ""
            content_text = _content_to_text(msg.get("content"))
            event_uuid = _msg_uuid(session_id, "tool", idx)
            out.append(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": content_text,
                            }
                        ],
                    },
                    "uuid": event_uuid,
                    "parentUuid": parent_uuid,
                    "sessionId": session_id,
                    "timestamp": timestamp,
                }
            )
            parent_uuid = event_uuid

    return out


def _scrub_session_for_upload(data: dict) -> dict:
    """Best-effort scrub of transcript fields before any upload temp file."""
    scrubbed = dict(data)
    scrubbed["messages"] = _scrub(data.get("messages") or [])
    scrubbed["events"] = _scrub(data.get("events") or [])
    scrubbed["tools"] = _scrub(data.get("tools") or [])
    return scrubbed


def _write_row_payload(data: dict, tmp_path: str) -> None:
    """Single-row JSONL (existing format) — used by KPI scheduler."""
    scrubbed = _scrub_session_for_upload(data)
    session_row = {
        "session_id": data["session_id"],
        "user_id": data.get("user_id"),
        "session_start_time": data["session_start_time"],
        "session_end_time": data["session_end_time"],
        "model_name": data["model_name"],
        "total_cost_usd": data.get("total_cost_usd"),
        "messages": json.dumps(scrubbed["messages"]),
        "events": json.dumps(scrubbed["events"]),
        "tools": json.dumps(scrubbed["tools"]),
    }

    with open(tmp_path, "w") as tmp:
        json.dump(session_row, tmp)


def _write_claude_code_payload(data: dict, tmp_path: str) -> None:
    """Multi-line JSONL in Claude Code schema for the HF trace viewer."""
    # Scrub before conversion so secrets never reach the upload temp file.
    scrubbed = _scrub_session_for_upload(data)
    events = to_claude_code_jsonl(scrubbed)
    with open(tmp_path, "w") as tmp:
        for event in events:
            tmp.write(json.dumps(event))
            tmp.write("\n")


def _status_field(format: str) -> str:
    """Per-format upload status field on the local trajectory file."""
    return "personal_upload_status" if format == "claude_code" else "upload_status"


def _url_field(format: str) -> str:
    return "personal_upload_url" if format == "claude_code" else "upload_url"


def _read_session_file(session_file: str) -> dict:
    """Read a local session file while respecting uploader file locks."""
    import fcntl

    with open(session_file, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _update_upload_status(
    session_file: str,
    status_key: str,
    url_key: str,
    status: str,
    dataset_url: str | None = None,
) -> None:
    """Atomically update only this uploader's status fields.

    The org and personal uploaders run as separate processes against the same
    local session JSON file. Re-read under an exclusive lock so one uploader
    cannot clobber fields written by the other.
    """
    import fcntl

    with open(session_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
            data[status_key] = status
            if dataset_url is not None:
                data[url_key] = dataset_url
            data["last_save_time"] = datetime.now().isoformat()
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def dataset_card_readme(repo_id: str) -> str:
    """Dataset card for personal ML Intern session trace repos."""
    return f"""---
pretty_name: "ML Intern Session Traces"
language:
- en
license: other
task_categories:
- text-generation
tags:
- agent-traces
- coding-agent
- ml-intern
- session-traces
- claude-code
- hf-agent-trace-viewer
configs:
- config_name: default
  data_files:
  - split: train
    path: "sessions/**/*.jsonl"
---

# ML Intern session traces

This dataset contains ML Intern coding agent session traces uploaded from local
ML Intern runs. The traces are stored as JSON Lines files under `sessions/`,
with one file per session.

Repository: https://huggingface.co/datasets/{repo_id}

## Data description

Each `*.jsonl` file contains a single ML Intern session converted to a
Claude-Code-style event stream for the Hugging Face Agent Trace Viewer. Entries
can include user messages, assistant messages, tool calls, tool results, model
metadata, and timestamps.

Session files are written to paths of the form:

```text
sessions/YYYY-MM-DD/<session_id>.jsonl
```

## Redaction and review

**WARNING: no comprehensive redaction or human review has been performed for this dataset.**

ML Intern applies automated best-effort scrubbing for common secret patterns
such as Hugging Face, Anthropic, OpenAI, GitHub, and AWS tokens before upload.
This is not a privacy guarantee.

These traces may contain sensitive information, including prompts, code,
terminal output, file paths, repository names, private task context, tool
outputs, or other data from the local development environment. Treat every
session as potentially sensitive.

Do not make this dataset public unless you have manually inspected the uploaded
sessions and are comfortable sharing their full contents.

## Limitations

Coding agent transcripts can include private or off-topic content, failed
experiments, credentials accidentally pasted by a user, and outputs copied from
local files or services. Use with appropriate caution, especially before
changing repository visibility.
"""


def _upload_dataset_card(api: Any, repo_id: str, token: str, format: str) -> None:
    """Create/update a README for personal trace datasets."""
    if format != "claude_code":
        return

    api.upload_file(
        path_or_fileobj=dataset_card_readme(repo_id).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Update dataset card",
    )


def upload_session_as_file(
    session_file: str,
    repo_id: str,
    max_retries: int = 3,
    format: str = "row",
    token_env: str | None = None,
    private: bool = False,
) -> bool:
    """Upload a single session as an individual JSONL file (no race conditions).

    Args:
        session_file: Path to local session JSON file
        repo_id: HuggingFace dataset repo ID
        max_retries: Number of retry attempts
        format: ``row`` (default, KPI-compatible) or ``claude_code`` (HF
            Agent Trace Viewer compatible).
        token_env: Name of the env var holding the HF token. ``None`` falls
            back to the org-token chain (``HF_SESSION_UPLOAD_TOKEN`` →
            ``HF_TOKEN`` → ``HF_ADMIN_TOKEN``).
        private: When creating the repo for the first time, mark it private.

    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub library not available", file=sys.stderr)
        return False

    status_key = _status_field(format)
    url_key = _url_field(format)

    try:
        data = _read_session_file(session_file)

        # Skip if already uploaded for this format.
        if data.get(status_key) == "success":
            return True

        hf_token = _resolve_token(token_env)
        if not hf_token:
            _update_upload_status(session_file, status_key, url_key, "failed")
            return False

        # Build temp upload payload in the requested format.
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            if format == "claude_code":
                _write_claude_code_payload(data, tmp_path)
            else:
                _write_row_payload(data, tmp_path)

            session_id = data["session_id"]
            date_str = datetime.fromisoformat(data["session_start_time"]).strftime(
                "%Y-%m-%d"
            )
            repo_path = f"sessions/{date_str}/{session_id}.jsonl"

            api = HfApi()
            for attempt in range(max_retries):
                try:
                    # Idempotent create — visibility is set on first creation
                    # only. Existing repos keep whatever the user picked via
                    # /share-traces.
                    try:
                        api.create_repo(
                            repo_id=repo_id,
                            repo_type="dataset",
                            private=private,
                            token=hf_token,
                            exist_ok=True,
                        )
                    except Exception:
                        pass

                    _upload_dataset_card(api, repo_id, hf_token, format)

                    api.upload_file(
                        path_or_fileobj=tmp_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token,
                        commit_message=f"Add session {session_id}",
                    )

                    _update_upload_status(
                        session_file,
                        status_key,
                        url_key,
                        "success",
                        f"https://huggingface.co/datasets/{repo_id}",
                    )
                    return True

                except Exception:
                    if attempt < max_retries - 1:
                        import time

                        wait_time = 2**attempt
                        time.sleep(wait_time)
                    else:
                        _update_upload_status(
                            session_file, status_key, url_key, "failed"
                        )
                        return False

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        print(f"Error uploading session: {e}", file=sys.stderr)
        return False


def retry_failed_uploads(
    directory: str,
    repo_id: str,
    format: str = "row",
    token_env: str | None = None,
    private: bool = False,
):
    """Retry all failed/pending uploads in a directory for the given format."""
    log_dir = Path(directory)
    if not log_dir.exists():
        return

    status_key = _status_field(format)
    session_files = list(log_dir.glob("session_*.json"))

    for filepath in session_files:
        try:
            data = _read_session_file(str(filepath))

            # Only retry pending or failed uploads. Files predating this
            # field don't have it; treat unknown as "not yet attempted" for
            # the row format (legacy behavior) and "skip" for claude_code
            # so we don't suddenly re-upload pre-existing sessions to a
            # newly-introduced personal repo.
            status = data.get(status_key, "unknown")
            if format == "claude_code" and status_key not in data:
                continue

            if status in ("pending", "failed", "unknown"):
                upload_session_as_file(
                    str(filepath),
                    repo_id,
                    format=format,
                    token_env=token_env,
                    private=private,
                )

        except Exception:
            pass


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="session_uploader.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser("upload")
    p_upload.add_argument("session_file")
    p_upload.add_argument("repo_id")
    p_upload.add_argument(
        "--format",
        choices=["row", "claude_code"],
        default="row",
    )
    p_upload.add_argument(
        "--token-env",
        default=None,
        help="Env var name holding the HF token (default: org fallback chain).",
    )
    p_upload.add_argument("--private", default="false")

    p_retry = sub.add_parser("retry")
    p_retry.add_argument("directory")
    p_retry.add_argument("repo_id")
    p_retry.add_argument(
        "--format",
        choices=["row", "claude_code"],
        default="row",
    )
    p_retry.add_argument("--token-env", default=None)
    p_retry.add_argument("--private", default="false")

    args = parser.parse_args()

    if args.command == "upload":
        ok = upload_session_as_file(
            args.session_file,
            args.repo_id,
            format=args.format,
            token_env=args.token_env,
            private=_str2bool(args.private),
        )
        sys.exit(0 if ok else 1)

    if args.command == "retry":
        retry_failed_uploads(
            args.directory,
            args.repo_id,
            format=args.format,
            token_env=args.token_env,
            private=_str2bool(args.private),
        )
        sys.exit(0)

    parser.print_help()
    sys.exit(1)
