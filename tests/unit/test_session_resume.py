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
    assert result["dropped_count"] == 0
    assert result["forked"] is False
    assert result["model_name"] == "openai/gpt-5.5"
    assert result["had_redacted_content"] is False
    assert result["invalid_saved_model"] is None
    assert session.config.model_name == "openai/gpt-5.5"
    assert session.session_id == "saved-session"
    # Source log path is never reused: future heartbeat saves write to a
    # fresh file so the snapshot stays intact (regression: see source-log
    # round-trip test below).
    assert session._local_save_path is None
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
    assert session._local_save_path is None


def test_restore_rejects_invalid_saved_model(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = log_dir / "session.json"
    log_dir.mkdir()
    path.write_text(
        json.dumps(
            {
                "session_id": "saved",
                "user_id": "user-a",
                "model_name": "not a real id with spaces",
                "messages": [{"role": "user", "content": "hello"}],
                "events": [],
            }
        )
    )

    session = _FakeSession(user_id="user-a")
    original_model = session.config.model_name

    result = session_resume.restore_session_from_log(session, path)

    assert result["invalid_saved_model"] == "not a real id with spaces"
    assert result["model_name"] == original_model
    assert session.config.model_name == original_model


def test_restore_counts_dropped_messages(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = log_dir / "session.json"
    log_dir.mkdir()
    path.write_text(
        json.dumps(
            {
                "session_id": "saved",
                "user_id": "user-a",
                "model_name": "openai/gpt-5.5",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": 12345},  # invalid content type
                ],
                "events": [],
            }
        )
    )

    session = _FakeSession(user_id="user-a")

    result = session_resume.restore_session_from_log(session, path)

    assert result["restored_count"] == 1
    assert result["dropped_count"] == 1


def test_restore_does_not_overwrite_source_log_on_save(tmp_path, monkeypatch):
    """Regression: resuming + saving must not destroy the source log on disk.

    Without the always-fork ``_local_save_path`` reset, the next heartbeat
    save would rewrite the source file with ``events=[resumed_from]`` and
    ``total_cost_usd=0``, wiping the original audit trail. This builds a
    real ``Session`` and exercises the round-trip.
    """
    monkeypatch.chdir(tmp_path)

    from agent.context_manager.manager import ContextManager
    from agent.core.session import Session

    log_dir = tmp_path / "session_logs"
    log_dir.mkdir()
    src_path = log_dir / "src.json"
    src_payload = {
        "session_id": "saved-session",
        "user_id": "user-a",
        "session_start_time": "2026-01-01T00:00:00",
        "session_end_time": "2026-01-01T00:05:00",
        "model_name": "openai/gpt-5.5",
        "messages": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "earlier work"},
        ],
        "events": [
            {"event_type": "llm_call", "data": {"cost_usd": 0.42}},
            {"event_type": "turn_complete", "data": {}},
        ],
    }
    src_path.write_text(json.dumps(src_payload, indent=2))
    src_bytes_before = src_path.read_bytes()

    class _Cfg:
        model_name = "openai/gpt-5.5"
        save_sessions = True
        session_dataset_repo = None
        auto_save_interval = 1
        heartbeat_interval_s = 60
        max_iterations = 10
        yolo_mode = False
        confirm_cpu_jobs = False
        auto_file_upload = False
        reasoning_effort = None
        share_traces = False
        personal_trace_repo_template = None
        mcpServers: dict = {}

    cm = ContextManager.__new__(ContextManager)
    cm.items = [Message(role="system", content="current system")]
    cm.tool_specs = []
    cm.model_max_tokens = 200_000
    cm.running_context_usage = 0
    cm.compact_size = 0.1
    cm.untouched_messages = 5
    cm.hf_token = None
    cm.local_mode = True
    cm.system_prompt = "current system"
    cm.on_message_added = None

    import asyncio as _asyncio

    session = Session(
        event_queue=_asyncio.Queue(),
        config=_Cfg(),
        tool_router=None,
        context_manager=cm,
        hf_token=None,
        user_id="user-a",
        local_mode=True,
    )

    session_resume.restore_session_from_log(session, src_path)
    assert session._local_save_path is None

    saved_path = session.save_trajectory_local(directory=str(log_dir))

    assert saved_path is not None
    assert Path(saved_path) != src_path
    assert src_path.read_bytes() == src_bytes_before


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
