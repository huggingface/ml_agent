"""Regression tests for interactive CLI rendering and research model routing."""

import json
import os
import sys
import time
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from litellm import Message

import agent.main as main_mod
from agent.tools.research_tool import _get_research_model
from agent.utils import terminal_display


def test_direct_anthropic_research_model_stays_off_bedrock():
    assert (
        _get_research_model("anthropic/claude-opus-4-6")
        == "anthropic/claude-sonnet-4-6"
    )


def test_bedrock_anthropic_research_model_stays_on_bedrock():
    assert (
        _get_research_model("bedrock/us.anthropic.claude-opus-4-6-v1")
        == "bedrock/us.anthropic.claude-sonnet-4-6"
    )


def test_non_anthropic_research_model_is_unchanged():
    assert _get_research_model("openai/gpt-5.4") == "openai/gpt-5.4"


def test_subagent_display_does_not_spawn_background_redraw(monkeypatch):
    calls: list[object] = []

    def _unexpected_future(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("background redraw task should not be created")

    monkeypatch.setattr("asyncio.ensure_future", _unexpected_future)
    monkeypatch.setattr(
        terminal_display,
        "_console",
        SimpleNamespace(file=StringIO(), width=100),
    )

    mgr = terminal_display.SubAgentDisplayManager()
    mgr.start("agent-1", "research")
    mgr.add_call("agent-1", '▸ hf_papers  {"operation": "search"}')
    mgr.clear("agent-1")

    assert calls == []


def test_cli_forwards_model_flag_to_interactive_main(monkeypatch):
    seen: dict[str, str | None] = {}

    async def fake_main(*, model=None):
        seen["model"] = model

    monkeypatch.setattr(sys, "argv", ["ml-intern", "--model", "openai/gpt-5.5"])
    monkeypatch.setattr(main_mod, "main", fake_main)

    main_mod.cli()

    assert seen["model"] == "openai/gpt-5.5"


def _write_session_log(
    directory: Path,
    name: str,
    *,
    session_id: str,
    content: str,
    mtime: float,
) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "session_start_time": "2026-01-01T00:00:00",
                "session_end_time": "2026-01-01T00:05:00",
                "model_name": "openai/gpt-5.5",
                "messages": [
                    {"role": "system", "content": "old system"},
                    {"role": "user", "content": content},
                ],
                "events": [{"event_type": "turn_complete", "data": {}}],
            }
        )
    )
    os.utime(path, (mtime, mtime))
    return path


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

    entries = main_mod._list_session_logs(log_dir)

    assert [entry.path for entry in entries] == [newer, older]
    assert entries[0].session_id == "newer-session"
    assert entries[0].preview == "newer prompt"


def test_restore_session_from_log_replaces_context_and_reuses_log_path(tmp_path):
    log_dir = tmp_path / "session_logs"
    path = _write_session_log(
        log_dir,
        "session.json",
        session_id="saved-session",
        content="continue this work",
        mtime=time.time(),
    )

    class FakeContext:
        def __init__(self):
            self.items = [Message(role="system", content="current system")]
            self.running_context_usage = 0

        def _recompute_usage(self, model_name):
            assert model_name == "openai/gpt-5.5"
            self.running_context_usage = 123

    class FakeSession:
        def __init__(self):
            self.context_manager = FakeContext()
            self.config = SimpleNamespace(model_name="moonshotai/Kimi-K2.6")
            self.session_id = "current-session"
            self.session_start_time = "2026-01-02T00:00:00"
            self.logged_events = []
            self._local_save_path = None
            self.turn_count = 0
            self.last_auto_save_turn = 0
            self.pending_approval = {"tool_calls": ["pending"]}

        def update_model(self, model_name):
            self.config.model_name = model_name

    session = FakeSession()

    restored = main_mod._restore_session_from_log(session, path)

    assert restored == 1
    assert session.session_id == "saved-session"
    assert session._local_save_path == str(path)
    assert session.turn_count == 1
    assert session.last_auto_save_turn == 1
    assert session.pending_approval is None
    assert session.logged_events == [{"event_type": "turn_complete", "data": {}}]
    assert [msg.role for msg in session.context_manager.items] == ["system", "user"]
    assert session.context_manager.items[0].content == "current system"
    assert session.context_manager.items[1].content == "continue this work"
    assert session.context_manager.running_context_usage == 123


@pytest.mark.asyncio
async def test_interactive_main_applies_model_override_before_banner(monkeypatch):
    class StopAfterBanner(Exception):
        pass

    def fake_banner(*, model=None, hf_user=None):
        assert model == "openai/gpt-5.5"
        assert hf_user == "tester"
        raise StopAfterBanner

    monkeypatch.setattr(main_mod.os, "system", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(main_mod, "PromptSession", lambda: object())
    monkeypatch.setattr(main_mod, "resolve_hf_token", lambda: "hf-token")
    monkeypatch.setattr(main_mod, "_get_hf_user", lambda _token: "tester")
    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            model_name="moonshotai/Kimi-K2.6",
            mcpServers={},
        ),
    )
    monkeypatch.setattr(main_mod, "print_banner", fake_banner)

    with pytest.raises(StopAfterBanner):
        await main_mod.main(model="openai/gpt-5.5")
