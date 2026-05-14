import asyncio
from types import SimpleNamespace

import pytest
from litellm import Message

import agent.main as main_mod
from agent.core.agent_loop import process_submission
from agent.core.session import Event, OpType, Session
from agent.tools import plan_tool


class _FakeConfig:
    model_name = "openai/gpt-5.5"
    save_sessions = False
    session_dataset_repo = "fake/repo"
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


class _FakeContext:
    def __init__(self) -> None:
        self.items = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="old task"),
            Message(role="assistant", content="old answer"),
        ]
        self.model_max_tokens = 200_000
        self.running_context_usage = 123
        self.on_message_added = None


def _make_session() -> Session:
    return Session(
        event_queue=asyncio.Queue(),
        config=_FakeConfig(),
        tool_router=None,
        context_manager=_FakeContext(),
        hf_token=None,
        user_id="user-a",
        local_mode=True,
    )


def test_start_new_conversation_rotates_session_state(monkeypatch):
    plan_tool.reset_current_plan()
    session = _make_session()
    session.config.save_sessions = True
    session.turn_count = 2
    session.last_auto_save_turn = 2
    session.logged_events = [{"event_type": "turn_complete", "data": {}}]
    session._local_save_path = "session_logs/old.json"
    session.pending_approval = {"tool_calls": ["pending"]}
    session.auto_approval_estimated_spend_usd = 1.25
    session.current_plan = [
        {"id": "old-session", "content": "old session item", "status": "pending"}
    ]
    monkeypatch.setattr(
        plan_tool,
        "_current_plan",
        [{"id": "global", "content": "global item", "status": "in_progress"}],
    )
    old_session_id = session.session_id
    uploads: list[str] = []

    def fake_upload(repo_id: str) -> str:
        uploads.append(repo_id)
        return "session_logs/saved.json"

    monkeypatch.setattr(session, "save_and_upload_detached", fake_upload)

    result = session.start_new_conversation()

    assert uploads == ["fake/repo"]
    assert result["previous_session_id"] == old_session_id
    assert result["previous_turn_count"] == 2
    assert result["saved_path"] == "session_logs/saved.json"
    assert session.session_id != old_session_id
    assert [msg.role for msg in session.context_manager.items] == ["system"]
    assert session.context_manager.items[0].content == "system prompt"
    assert session.context_manager.running_context_usage == 0
    assert session.turn_count == 0
    assert session.last_auto_save_turn == 0
    assert session.logged_events == []
    assert session._local_save_path is None
    assert session.pending_approval is None
    assert session.auto_approval_estimated_spend_usd == 0.0
    assert session.current_plan == []
    assert plan_tool.get_current_plan() == []


@pytest.mark.asyncio
async def test_new_submission_resets_context_and_reports_clear_flag():
    session = _make_session()
    submission = SimpleNamespace(
        operation=SimpleNamespace(
            op_type=OpType.NEW,
            data={"clear_screen": True},
        )
    )

    should_continue = await process_submission(session, submission)

    event: Event = await session.event_queue.get()
    assert should_continue is True
    assert event.event_type == "new_complete"
    assert event.data["clear_screen"] is True
    assert event.data["session_id"] == session.session_id
    assert [msg.role for msg in session.context_manager.items] == ["system"]
    assert [e["event_type"] for e in session.logged_events] == ["new_complete"]


@pytest.mark.asyncio
async def test_new_and_clear_slash_commands_share_operation_with_distinct_clear_flags():
    submission_id = [0]
    holder = [object()]

    new_submission = await main_mod._handle_slash_command(
        "/new",
        _FakeConfig(),
        holder,
        asyncio.Queue(),
        submission_id,
    )
    clear_submission = await main_mod._handle_slash_command(
        "/clear",
        _FakeConfig(),
        holder,
        asyncio.Queue(),
        submission_id,
    )

    assert new_submission is not None
    assert new_submission.operation.op_type == OpType.NEW
    assert new_submission.operation.data == {"clear_screen": False}
    assert clear_submission is not None
    assert clear_submission.operation.op_type == OpType.NEW
    assert clear_submission.operation.data == {"clear_screen": True}
    assert submission_id == [2]
