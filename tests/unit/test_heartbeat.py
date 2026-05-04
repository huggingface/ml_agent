"""Heartbeat + stable-local-path tests for Session.

We don't spin up the real agent loop — we build a minimal Session with a
stubbed config and an in-memory queue, then call send_event repeatedly while
monkeypatching time.monotonic to simulate seconds passing.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from agent.core.session import Event, Session


class _FakeConfig:
    model_name = "claude-opus-4-6"
    save_sessions = True
    session_dataset_repo = "fake/repo"
    auto_save_interval = 1
    heartbeat_interval_s = 60
    max_iterations = 10
    yolo_mode = False
    confirm_cpu_jobs = False
    auto_file_upload = False
    reasoning_effort = None
    mcpServers: dict = {}


def _mk_session(tmp_path: Path, monkeypatch) -> Session:
    monkeypatch.chdir(tmp_path)  # so session_logs/ lands under tmp_path
    # Stub out the context manager to avoid litellm lookups.
    from agent.context_manager.manager import ContextManager
    cm = ContextManager.__new__(ContextManager)
    cm.items = []
    cm.tool_specs = []
    cm.model_max_tokens = 200_000
    cm.running_context_usage = 0
    cm.compact_size = 0.1
    cm.untouched_messages = 5
    cm.hf_token = None
    cm.local_mode = True
    s = Session(
        event_queue=asyncio.Queue(),
        config=_FakeConfig(),
        tool_router=None,
        context_manager=cm,
        hf_token=None,
        local_mode=True,
    )
    return s


def test_heartbeat_fires_after_interval(tmp_path, monkeypatch):
    # Use asyncio.run rather than pytest-asyncio so the test works without the
    # plugin installed (same pattern elsewhere in this repo).
    async def body():
        s = _mk_session(tmp_path, monkeypatch)
        calls = []

        def fake_upload(repo_id):
            calls.append(repo_id)
            return "fake/path.json"

        monkeypatch.setattr(s, "save_and_upload_detached", fake_upload)

        # t=0: first event, should NOT trigger (initial _last_heartbeat_ts = now)
        with patch("agent.core.telemetry.time.monotonic", return_value=100.0):
            s._last_heartbeat_ts = 100.0
            await s.send_event(Event(event_type="x"))
        assert calls == []

        # t=+30s: still under interval → no save
        with patch("agent.core.telemetry.time.monotonic", return_value=130.0):
            await s.send_event(Event(event_type="y"))
        assert calls == []

        # t=+61s: over 60s → save fires once
        with patch("agent.core.telemetry.time.monotonic", return_value=161.0):
            await s.send_event(Event(event_type="z"))
        # create_task runs on the event loop; wait for the to_thread to complete
        await asyncio.sleep(0.05)
        assert calls == ["fake/repo"]

        # Next event shortly after → no second save (interval resets to 161)
        with patch("agent.core.telemetry.time.monotonic", return_value=170.0):
            await s.send_event(Event(event_type="w"))
        await asyncio.sleep(0.05)
        assert len(calls) == 1

    asyncio.run(body())


def test_stable_local_path_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.context_manager.manager import ContextManager
    cm = ContextManager.__new__(ContextManager)
    cm.items = []
    cm.tool_specs = []
    cm.model_max_tokens = 200_000
    cm.running_context_usage = 0
    cm.compact_size = 0.1
    cm.untouched_messages = 5
    cm.hf_token = None
    cm.local_mode = True

    s = Session(
        event_queue=asyncio.Queue(),
        config=_FakeConfig(),
        tool_router=None,
        context_manager=cm,
        hf_token=None,
        local_mode=True,
    )

    p1 = s.save_trajectory_local(directory="session_logs")
    assert p1 is not None
    p2 = s.save_trajectory_local(directory="session_logs")
    p3 = s.save_trajectory_local(directory="session_logs")
    # All three saves land on the same file — heartbeat should not spam files.
    assert p1 == p2 == p3
    files = list(Path("session_logs").glob("session_*.json"))
    # Exactly one final file; the .tmp should be renamed away.
    assert len(files) == 1

    # File is valid JSON (atomic write → no torn content).
    with open(p1) as f:
        data = json.load(f)
    assert data["session_id"] == s.session_id
    assert data["upload_status"] == "pending"
