import asyncio
import json

import pytest

from agent.config import Config
from agent.core import agent_loop
from agent.core.agent_loop import Handlers, LLMResult
from agent.core.session import Session
from agent.tools.plan_tool import PlanTool


class FakeToolRouter:
    def __init__(self):
        self.calls = []

    def get_tool_specs_for_llm(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "plan_tool",
                    "description": "Update plan",
                    "parameters": {"type": "object"},
                },
            }
        ]

    async def call_tool(self, name, arguments, session=None, tool_call_id=None):
        self.calls.append((name, arguments, tool_call_id))
        if name == "plan_tool" and session is not None:
            session.current_plan = [dict(todo) for todo in arguments["todos"]]
        return "plan updated", True


@pytest.mark.asyncio
async def test_plan_tool_stores_session_scoped_plan():
    events = []

    class FakeSession:
        current_plan = []

        async def send_event(self, event):
            events.append(event)

    session = FakeSession()
    todos = [{"id": "1", "content": "Smoke test", "status": "in_progress"}]

    result = await PlanTool(session=session).execute({"todos": todos})

    assert result["isError"] is False
    assert session.current_plan == todos
    assert events[0].event_type == "plan_update"
    assert events[0].data == {"plan": todos}


@pytest.mark.asyncio
async def test_no_tool_response_retries_when_plan_is_incomplete(monkeypatch):
    config = Config.model_validate(
        {"model_name": "openai/test", "save_sessions": False}
    )
    event_queue = asyncio.Queue()
    router = FakeToolRouter()
    session = Session(
        event_queue,
        config,
        tool_router=router,
        stream=False,
    )
    session.current_plan = [
        {
            "id": "1",
            "content": "Write and smoke-test training script",
            "status": "in_progress",
        },
        {"id": "2", "content": "Launch full training job", "status": "pending"},
    ]
    calls = []

    async def fake_call_llm_non_streaming(session, messages, tools, llm_params):
        calls.append(messages)
        if len(calls) == 1:
            return LLMResult(
                content="I should keep going, but I forgot to call a tool.",
                tool_calls_acc={},
                token_count=10,
                finish_reason="stop",
            )
        if len(calls) == 2:
            assert "CONTINUATION GUARD" in messages[-1].content
            return LLMResult(
                content=None,
                tool_calls_acc={
                    0: {
                        "id": "call_1",
                        "function": {
                            "name": "plan_tool",
                            "arguments": json.dumps(
                                {
                                    "todos": [
                                        {
                                            "id": "1",
                                            "content": "Write and smoke-test training script",
                                            "status": "completed",
                                        },
                                        {
                                            "id": "2",
                                            "content": "Launch full training job",
                                            "status": "completed",
                                        },
                                    ]
                                }
                            ),
                        },
                    }
                },
                token_count=20,
                finish_reason="tool_calls",
            )
        return LLMResult(
            content="Done.",
            tool_calls_acc={},
            token_count=30,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        agent_loop, "_resolve_llm_params", lambda *_, **__: {"model": "openai/test"}
    )
    monkeypatch.setattr(
        agent_loop, "_call_llm_non_streaming", fake_call_llm_non_streaming
    )

    final = await Handlers.run_agent(session, "continue")

    assert final == "Done."
    assert len(calls) == 3
    assert router.calls[0][0] == "plan_tool"
    assert all(todo["status"] == "completed" for todo in session.current_plan)
    events = []
    while not event_queue.empty():
        events.append(await event_queue.get())
    assert any(
        event.event_type == "tool_log"
        and "text-only response" in (event.data or {}).get("log", "")
        for event in events
    )
