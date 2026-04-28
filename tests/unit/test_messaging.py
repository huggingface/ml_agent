import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from agent.config import Config
from agent.core.session import Event, Session
from agent.messaging.gateway import NotificationGateway
from agent.messaging.models import NotificationRequest, NotificationResult
from agent.messaging.slack import SlackProvider, _format_slack_mrkdwn
from agent.tools.notify_tool import notify_handler
from backend.session_manager import AgentSession, SessionManager


class DummyToolRouter:
    def get_tool_specs_for_llm(self) -> list[dict]:
        return []


class RecordingGateway:
    def __init__(self):
        self.enqueued: list[NotificationRequest] = []
        self.sent: list[NotificationRequest] = []

    async def enqueue(self, request: NotificationRequest) -> bool:
        self.enqueued.append(request)
        return True

    async def send_many(
        self, requests: list[NotificationRequest]
    ) -> list[NotificationResult]:
        self.sent.extend(requests)
        return [
            NotificationResult(
                destination=request.destination,
                ok=True,
                provider="test",
            )
            for request in requests
        ]


def _config_with_messaging(**destination_overrides) -> Config:
    destination = {
        "provider": "slack",
        "token": "xoxb-test",
        "channel": "C123",
        **destination_overrides,
    }
    return Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "messaging": {
                "enabled": True,
                "destinations": {
                    "slack.ops": destination,
                },
            },
        }
    )


def _test_session(
    config: Config, gateway, session_id: str = "session-test"
) -> Session:
    return Session(
        asyncio.Queue(),
        config=config,
        tool_router=DummyToolRouter(),
        context_manager=SimpleNamespace(items=[]),
        notification_gateway=gateway,
        session_id=session_id,
    )


def test_messaging_config_validates_destination_names():
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "model_name": "moonshotai/Kimi-K2.6",
                "messaging": {
                    "enabled": True,
                    "destinations": {
                        "Slack Ops": {
                            "provider": "slack",
                            "token": "x",
                            "channel": "C123",
                        }
                    },
                },
            }
        )

    config = _config_with_messaging(allow_agent_tool=True, allow_auto_events=True)
    assert config.messaging.can_agent_tool_send("slack.ops")
    assert config.messaging.can_auto_send("slack.ops")


def test_messaging_config_default_auto_destinations_only_returns_auto_enabled():
    config = Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "messaging": {
                "enabled": True,
                "destinations": {
                    "slack.ops": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C123",
                        "allow_auto_events": True,
                    },
                    "slack.tool": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C999",
                        "allow_agent_tool": True,
                    },
                },
            },
        }
    )

    assert config.messaging.default_auto_destinations() == ["slack.ops"]


def test_messaging_config_default_auto_destinations_empty_when_disabled():
    config = Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "messaging": {
                "enabled": False,
                "destinations": {
                    "slack.ops": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C123",
                        "allow_auto_events": True,
                    },
                },
            },
        }
    )

    assert config.messaging.default_auto_destinations() == []


def test_slack_mrkdwn_formatter_converts_common_markdown():
    formatted = _format_slack_mrkdwn(
        "# Result\n"
        "**Done** with *details* and ~~old text~~.\n"
        "See [PR](https://github.com/huggingface/ml-intern/pull/116).\n"
        "Keep `**literal**` and ```python\nx < 3\n``` untouched.\n"
        "Escape <raw> & text."
    )

    assert "*Result*" in formatted
    assert "*Done*" in formatted
    assert "_details_" in formatted
    assert "~old text~" in formatted
    assert "<https://github.com/huggingface/ml-intern/pull/116|PR>" in formatted
    assert "`**literal**`" in formatted
    assert "```python\nx < 3\n```" in formatted
    assert "Escape &lt;raw&gt; &amp; text." in formatted


@pytest.mark.asyncio
async def test_slack_provider_formats_and_sends_payload():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["content_type"] = request.headers["Content-Type"]
        seen["json"] = request.read().decode("utf-8")
        return httpx.Response(200, json={"ok": True, "ts": "123.456"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SlackProvider()
        result = await provider.send(
            client,
            "slack.ops",
            _config_with_messaging().messaging.destinations["slack.ops"],
            NotificationRequest(
                destination="slack.ops",
                title="Approval required",
                message="A **run** is waiting. See [details](https://example.com).",
                severity="warning",
                metadata={"session_id": "sess-1"},
            ),
        )

    assert result.ok
    assert result.external_id == "123.456"
    assert seen["auth"] == "Bearer xoxb-test"
    assert seen["content_type"].startswith("application/json")
    payload = json.loads(str(seen["json"]))
    assert payload["channel"] == "C123"
    assert payload["mrkdwn"] is True
    assert payload["text"] == (
        "[WARNING] Approval required\n"
        "A *run* is waiting. See <https://example.com|details>.\n"
        "session_id: sess-1"
    )


@pytest.mark.asyncio
async def test_notification_gateway_retries_transient_failures(monkeypatch):
    attempts = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, json={"ok": False})
        return httpx.Response(200, json={"ok": True, "ts": "999.1"})

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("agent.messaging.gateway.asyncio.sleep", fake_sleep)

    config = _config_with_messaging(allow_agent_tool=True)
    gateway = NotificationGateway(config.messaging)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway._client = client
        result = await gateway.send(
            NotificationRequest(
                destination="slack.ops",
                message="hello",
            )
        )
        gateway._client = None

    assert attempts["count"] == 2
    assert result.ok


@pytest.mark.asyncio
async def test_notify_tool_rejects_non_allowlisted_destinations():
    config = _config_with_messaging(allow_agent_tool=False)
    gateway = RecordingGateway()
    session = _test_session(config, gateway)

    output, ok = await notify_handler(
        {"destinations": ["slack.ops"], "message": "done"},
        session=session,
    )

    assert not ok
    assert "unavailable for the notify tool" in output
    assert gateway.sent == []


@pytest.mark.asyncio
async def test_notify_tool_sends_to_allowlisted_destinations():
    config = _config_with_messaging(allow_agent_tool=True)
    gateway = RecordingGateway()
    session = _test_session(config, gateway, session_id="sess-42")

    output, ok = await notify_handler(
        {
            "destinations": ["slack.ops"],
            "title": "Training complete",
            "message": "The run finished successfully.",
            "severity": "success",
        },
        session=session,
    )

    assert ok
    assert output == "slack.ops: sent"
    assert len(gateway.sent) == 1
    sent = gateway.sent[0]
    assert sent.metadata["session_id"] == "sess-42"
    assert sent.metadata["model"] == "moonshotai/Kimi-K2.6"


@pytest.mark.asyncio
async def test_session_auto_notifications_only_send_opted_in_auto_destinations():
    config = Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "messaging": {
                "enabled": True,
                "destinations": {
                    "slack.ops": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C123",
                        "allow_auto_events": True,
                    },
                    "slack.tool": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C999",
                        "allow_agent_tool": True,
                    },
                },
            },
        }
    )
    gateway = RecordingGateway()
    session = _test_session(config, gateway, session_id="sess-auto")
    session.set_notification_destinations(["slack.ops", "slack.tool"])

    await session.send_event(
        Event(
            event_type="approval_required",
            data={"tools": [{"tool": "hf_jobs", "tool_call_id": "tc-1"}]},
        )
    )
    await session.send_event(
        Event(event_type="assistant_message", data={"content": "normal message"})
    )

    assert len(gateway.enqueued) == 1
    request = gateway.enqueued[0]
    assert request.destination == "slack.ops"
    assert request.severity == "warning"
    assert request.event_type == "approval_required"
    assert "hf_jobs" in request.message


@pytest.mark.asyncio
async def test_turn_complete_auto_notification_includes_final_response_summary():
    config = Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "messaging": {
                "enabled": True,
                "destinations": {
                    "slack.ops": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C123",
                        "allow_auto_events": True,
                    }
                },
            },
        }
    )
    gateway = RecordingGateway()
    session = _test_session(config, gateway, session_id="sess-done")
    session.set_notification_destinations(["slack.ops"])

    await session.send_event(
        Event(
            event_type="turn_complete",
            data={
                "history_size": 12,
                "final_response": "Evaluation finished. Accuracy: 84.2% on the validation split.",
            },
        )
    )

    assert len(gateway.enqueued) == 1
    request = gateway.enqueued[0]
    assert request.destination == "slack.ops"
    assert request.severity == "success"
    assert request.event_type == "turn_complete"
    assert "completed successfully" in request.message
    assert "Accuracy: 84.2%" in request.message


@pytest.mark.asyncio
async def test_turn_complete_auto_notification_supports_longer_summary():
    config = Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "messaging": {
                "enabled": True,
                "destinations": {
                    "slack.ops": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C123",
                        "allow_auto_events": True,
                    }
                },
            },
        }
    )
    gateway = RecordingGateway()
    session = _test_session(config, gateway, session_id="sess-long")
    session.set_notification_destinations(["slack.ops"])

    long_summary = "A" * 1200 + " END"
    await session.send_event(
        Event(
            event_type="turn_complete",
            data={
                "history_size": 12,
                "final_response": long_summary,
            },
        )
    )

    assert len(gateway.enqueued) == 1
    request = gateway.enqueued[0]
    assert request.event_type == "turn_complete"
    assert "A" * 1200 in request.message
    assert request.message.endswith("END")


@pytest.mark.asyncio
async def test_turn_complete_auto_notification_can_be_deferred():
    config = Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "messaging": {
                "enabled": True,
                "destinations": {
                    "slack.ops": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C123",
                        "allow_auto_events": True,
                    }
                },
            },
        }
    )
    gateway = RecordingGateway()
    session = Session(
        asyncio.Queue(),
        config=config,
        tool_router=DummyToolRouter(),
        context_manager=SimpleNamespace(items=[]),
        notification_gateway=gateway,
        notification_destinations=["slack.ops"],
        defer_turn_complete_notification=True,
        session_id="sess-deferred",
    )
    event = Event(
        event_type="turn_complete",
        data={"final_response": "Finished after the CLI drained the stream."},
    )

    await session.send_event(event)
    assert gateway.enqueued == []

    await session.send_deferred_turn_complete_notification(event)

    assert len(gateway.enqueued) == 1
    request = gateway.enqueued[0]
    assert request.destination == "slack.ops"
    assert request.event_type == "turn_complete"
    assert "Finished after the CLI drained the stream." in request.message


@pytest.mark.asyncio
async def test_turn_complete_can_be_disabled_by_custom_auto_event_config():
    config = Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "messaging": {
                "enabled": True,
                "auto_event_types": ["error"],
                "destinations": {
                    "slack.ops": {
                        "provider": "slack",
                        "token": "xoxb-test",
                        "channel": "C123",
                        "allow_auto_events": True,
                    }
                },
            },
        }
    )
    gateway = RecordingGateway()
    session = _test_session(config, gateway, session_id="sess-optout")
    session.set_notification_destinations(["slack.ops"])

    await session.send_event(
        Event(
            event_type="turn_complete",
            data={"final_response": "This should not notify."},
        )
    )

    assert gateway.enqueued == []


def test_session_manager_updates_notification_destinations_in_session_info():
    config = _config_with_messaging(allow_auto_events=True)
    manager = SessionManager(str(Path(__file__).resolve().parents[2] / "configs" / "cli_agent_config.json"))
    manager.config = config
    manager.sessions = {}

    session = _test_session(config, RecordingGateway(), session_id="sess-manager")
    manager.sessions["sess-manager"] = AgentSession(
        session_id="sess-manager",
        session=session,
        tool_router=DummyToolRouter(),
        submission_queue=asyncio.Queue(),
    )

    updated = manager.set_notification_destinations(
        "sess-manager",
        ["slack.ops", "slack.ops"],
    )

    assert updated == ["slack.ops"]
    info = manager.get_session_info("sess-manager")
    assert info is not None
    assert info["notification_destinations"] == ["slack.ops"]

    with pytest.raises(ValueError):
        manager.set_notification_destinations("sess-manager", ["slack.unknown"])
