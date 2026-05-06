"""Tests for ``agent.core.session_resume``."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from litellm import Message

from agent.core import session_resume


def _write_session_log(
    directory: Path,
    name: str,
    *,
    session_id: str,
    content: str,
    mtime: float,
    user_id: str | None = "user-a",
    extra_messages: list[dict] | None = None,
    events: list[dict] | None = None,
) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / name
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "session_start_time": "2026-01-01T00:00:00",
        "session_end_time": "2026-01-01T00:05:00",
        "model_name": "openai/gpt-5.5",
        "messages": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": content},
            *(extra_messages or []),
        ],
        "events": events
        if events is not None
        else [{"event_type": "turn_complete", "data": {}}],
    }
    path.write_text(json.dumps(payload))
    os.utime(path, (mtime, mtime))
    return path


class _FakeContext:
    def __init__(self) -> None:
        self.items = [Message(role="system", content="current system")]
        self.running_context_usage = 0
        self.recompute_calls: list[str] = []

    def _recompute_usage(self, model_name: str) -> None:
        self.recompute_calls.append(model_name)
        self.running_context_usage = 123


class _FakeSession:
    def __init__(self, *, user_id: str | None = "user-a") -> None:
        self.context_manager = _FakeContext()
        self.config = SimpleNamespace(model_name="moonshotai/Kimi-K2.6")
        self.session_id = "current-session"
        self.session_start_time = "2026-01-02T00:00:00"
        self.user_id = user_id
        self.logged_events: list[dict] = []
        self._local_save_path: str | None = None
        self.turn_count = 0
        self.last_auto_save_turn = 0
        self.pending_approval: dict | None = {"tool_calls": ["pending"]}

    def update_model(self, model_name: str) -> None:
        self.config.model_name = model_name


def test_session_log_listing_newest_first(tmp_path):
    log_dir = tmp_path / "session_logs"
    older = _write_session_log(
        log_dir,
        "older.json",
        session_id="older-session",
        content="older prompt",
        mtime=time.time() - 10,
    )
    newer = _write_session_log(
        log_dir,
        "newer.json",
        session_id="newer-session",
        content="newer prompt",
        mtime=time.time(),
    )

    entries = session_resume.list_session_logs(log_dir)

    assert [entry.path for entry in entries] == [newer, older]
    assert entries[0].session_id == "newer-session"
    assert entries[0].preview == "newer prompt"


def test_restore_continues_when_user_id_matches(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="continue this work",
        mtime=time.time(),
        user_id="user-a",
    )

    session = _FakeSession(user_id="user-a")

    result = session_resume.restore_session_from_log(session, path)

    assert result["restored_count"] == 1
    assert result["forked"] is False
    assert result["model_name"] == "openai/gpt-5.5"
    assert result["had_redacted_content"] is False
    assert session.config.model_name == "openai/gpt-5.5"
    assert session.session_id == "saved-session"
    assert session._local_save_path == str(path)
    assert session.turn_count == 1
    assert session.last_auto_save_turn == 1
    assert session.pending_approval is None
    assert [msg.role for msg in session.context_manager.items] == ["system", "user"]
    assert session.context_manager.items[0].content == "current system"
    assert session.context_manager.items[1].content == "continue this work"
    assert session.context_manager.running_context_usage == 123
    assert session.context_manager.recompute_calls == ["openai/gpt-5.5"]
    assert len(session.logged_events) == 1
    marker = session.logged_events[0]
    assert marker["event_type"] == "resumed_from"
    assert marker["data"]["forked"] is False
    assert marker["data"]["original_session_id"] == "saved-session"
    assert marker["data"]["original_event_count"] == 1


def test_restore_forks_when_user_id_differs(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="someone else's chat",
        mtime=time.time(),
        user_id="user-a",
    )

    session = _FakeSession(user_id="user-b")
    original_session_id = session.session_id
    original_start_time = session.session_start_time

    result = session_resume.restore_session_from_log(session, path)

    assert result["forked"] is True
    assert session.session_id == original_session_id
    assert session.session_start_time == original_start_time
    assert session._local_save_path is None
    marker = session.logged_events[0]
    assert marker["event_type"] == "resumed_from"
    assert marker["data"]["forked"] is True
    assert marker["data"]["original_session_id"] == "saved-session"


def test_restore_forks_when_one_side_is_anonymous(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="anonymous save",
        mtime=time.time(),
        user_id=None,
    )

    session = _FakeSession(user_id="user-a")

    result = session_resume.restore_session_from_log(session, path)

    assert result["forked"] is True
    assert session._local_save_path is None


def test_restore_continues_when_both_sides_anonymous(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="local-only chat",
        mtime=time.time(),
        user_id=None,
    )

    session = _FakeSession(user_id=None)

    result = session_resume.restore_session_from_log(session, path)

    assert result["forked"] is False
    assert session.session_id == "saved-session"
    assert session._local_save_path == str(path)


def test_restore_flags_redacted_messages(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="my token is [REDACTED_HF_TOKEN]",
        mtime=time.time(),
        user_id="user-a",
    )

    session = _FakeSession(user_id="user-a")

    result = session_resume.restore_session_from_log(session, path)

    assert result["had_redacted_content"] is True


def test_resolve_session_log_arg_accepts_index_and_id_prefix(tmp_path):
    log_dir = tmp_path / "session_logs"
    older = _write_session_log(
        log_dir,
        "older.json",
        session_id="abcdef-older",
        content="x",
        mtime=time.time() - 10,
    )
    newer = _write_session_log(
        log_dir,
        "newer.json",
        session_id="123456-newer",
        content="y",
        mtime=time.time(),
    )
    entries = session_resume.list_session_logs(log_dir)

    assert session_resume.resolve_session_log_arg("1", entries, log_dir) == newer
    assert session_resume.resolve_session_log_arg("abc", entries, log_dir) == older
    assert session_resume.resolve_session_log_arg("nope", entries, log_dir) is None
