"""Reload a previously saved session log into the active CLI session."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from litellm import Message

from agent.core.session import DEFAULT_SESSION_LOG_DIR

logger = logging.getLogger(__name__)

_REDACTED_MARKER = re.compile(r"\[REDACTED_[A-Z_]+\]")


@dataclass
class SessionLogEntry:
    """Metadata for a locally saved session log."""

    path: Path
    session_id: str
    session_start_time: str | None
    session_end_time: str | None
    model_name: str | None
    message_count: int
    preview: str
    mtime: float


def _message_preview(content: Any, max_chars: int = 72) -> str:
    """Return a one-line preview for string or OpenAI-style block content."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(block, str):
                parts.append(block)
        text = " ".join(parts)
    else:
        text = ""
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _first_user_preview(messages: list[Any]) -> str:
    for raw in messages:
        if isinstance(raw, dict) and raw.get("role") == "user":
            preview = _message_preview(raw.get("content"))
            if preview:
                return preview
    return "(no user prompt preview)"


def list_session_logs(
    directory: Path = DEFAULT_SESSION_LOG_DIR,
) -> list[SessionLogEntry]:
    """Return readable session logs under ``directory``, newest first."""
    if not directory.exists():
        return []

    entries: list[SessionLogEntry] = []
    for path in directory.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        messages = data.get("messages") or []
        if not isinstance(messages, list):
            continue

        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            session_id = path.stem

        stat = path.stat()
        entries.append(
            SessionLogEntry(
                path=path,
                session_id=session_id,
                session_start_time=data.get("session_start_time"),
                session_end_time=data.get("session_end_time"),
                model_name=data.get("model_name"),
                message_count=len(messages),
                preview=_first_user_preview(messages),
                mtime=stat.st_mtime,
            )
        )

    entries.sort(key=lambda item: item.mtime, reverse=True)
    return entries


def format_session_log_entry(index: int, entry: SessionLogEntry) -> str:
    timestamp = entry.session_end_time or entry.session_start_time
    label = "unknown time"
    if isinstance(timestamp, str) and timestamp:
        try:
            label = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            label = timestamp[:16]
    short_id = entry.session_id[:8]
    model = entry.model_name or "unknown model"
    return (
        f"{index:>2}. {label}  {short_id}  "
        f"{entry.message_count} msgs  {model}\n"
        f"    {entry.preview}"
    )


def resolve_session_log_arg(
    arg: str,
    entries: list[SessionLogEntry],
    directory: Path = DEFAULT_SESSION_LOG_DIR,
) -> Path | None:
    """Resolve ``/resume <arg>`` as index, path, filename, or session id prefix."""
    value = arg.strip()
    if not value:
        return None

    if value.isdigit():
        idx = int(value)
        if 1 <= idx <= len(entries):
            return entries[idx - 1].path

    candidate = Path(value).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(directory / candidate)
        if candidate.suffix != ".json":
            candidates.append(directory / f"{value}.json")

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    matches = [
        entry.path
        for entry in entries
        if entry.session_id.startswith(value) or entry.path.name.startswith(value)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _turn_count_from_messages(messages: list[Any]) -> int:
    return sum(
        1 for raw in messages if isinstance(raw, dict) and raw.get("role") == "user"
    )


def _has_redacted_content(messages: list[Any]) -> bool:
    """Whether any message body contains a ``[REDACTED_*]`` marker."""
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content")
        if isinstance(content, str) and _REDACTED_MARKER.search(content):
            return True
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str) and _REDACTED_MARKER.search(text):
                        return True
    return False


def restore_session_from_log(session: Any, path: Path) -> dict[str, Any]:
    """Replace the active session context with messages from ``path``.

    Continues the saved session (reusing its id and on-disk save path) when
    the log's ``user_id`` matches the current session, and forks otherwise:
    the caller's session id stays put and future heartbeat saves go to a
    fresh file rather than overwriting the source log.

    Returns metadata for the ``resume_complete`` event.
    """
    with open(path) as f:
        data = json.load(f)

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Selected log does not contain a messages array")

    restored_messages: list[Message] = []
    for raw in raw_messages:
        if not isinstance(raw, dict) or raw.get("role") == "system":
            continue
        try:
            restored_messages.append(Message.model_validate(raw))
        except Exception as e:
            logger.warning("Dropping malformed message from %s: %s", path, e)

    if not restored_messages:
        raise ValueError("Selected log has no restorable non-system messages")

    cm = session.context_manager
    system_msg = cm.items[0] if cm.items and cm.items[0].role == "system" else None
    cm.items = ([system_msg] if system_msg else []) + restored_messages

    saved_model = data.get("model_name")
    if isinstance(saved_model, str) and saved_model:
        session.update_model(saved_model)

    cm._recompute_usage(session.config.model_name)

    saved_session_id = data.get("session_id")
    saved_user_id = data.get("user_id")
    is_continuation = saved_user_id == session.user_id

    if is_continuation:
        if isinstance(saved_session_id, str) and saved_session_id:
            session.session_id = saved_session_id
        session.session_start_time = (
            data.get("session_start_time") or session.session_start_time
        )
        session._local_save_path = str(path)
    else:
        # Different user (or one side anonymous) — keep current identity so
        # uploads land under the right account, and let the next save pick a
        # fresh filename instead of overwriting the source log.
        session._local_save_path = None

    saved_event_count = (
        len(data.get("events", [])) if isinstance(data.get("events"), list) else 0
    )
    session.logged_events = [
        {
            "timestamp": datetime.now().isoformat(),
            "event_type": "resumed_from",
            "data": {
                "path": str(path),
                "original_session_id": (
                    saved_session_id if isinstance(saved_session_id, str) else None
                ),
                "original_event_count": saved_event_count,
                "forked": not is_continuation,
            },
        }
    ]
    session.turn_count = _turn_count_from_messages(raw_messages)
    session.last_auto_save_turn = session.turn_count
    session.pending_approval = None

    return {
        "path": str(path),
        "restored_count": len(restored_messages),
        "model_name": session.config.model_name,
        "forked": not is_continuation,
        "had_redacted_content": _has_redacted_content(raw_messages),
    }
