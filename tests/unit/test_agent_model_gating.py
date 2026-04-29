"""Tests for premium model gating in backend/routes/agent.py."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from routes import agent  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_quota_store():
    agent.user_quotas._reset_for_tests()
    yield
    agent.user_quotas._reset_for_tests()


def test_premium_model_predicate_includes_bedrock_claude_and_gpt55():
    assert agent._is_premium_model("bedrock/us.anthropic.claude-opus-4-6-v1")
    assert agent._is_premium_model("anthropic/claude-opus-4-6")
    assert agent._is_premium_model("openai/gpt-5.5")
    assert not agent._is_premium_model("moonshotai/Kimi-K2.6")


@pytest.mark.asyncio
async def test_premium_model_gate_rejects_gpt55_for_non_hf_user(monkeypatch):
    async def fake_require_hf_org_member(_request):
        return False

    monkeypatch.setattr(agent, "require_huggingface_org_member", fake_require_hf_org_member)

    with pytest.raises(HTTPException) as exc_info:
        await agent._require_hf_for_premium_model(None, "openai/gpt-5.5")

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "premium_model_restricted"


@pytest.mark.asyncio
async def test_free_model_gate_skips_hf_membership_check(monkeypatch):
    async def fail_if_called(_request):
        raise AssertionError("free models must not require HF org membership")

    monkeypatch.setattr(agent, "require_huggingface_org_member", fail_if_called)

    await agent._require_hf_for_premium_model(None, "moonshotai/Kimi-K2.6")


@pytest.mark.asyncio
async def test_premium_quota_charges_gpt55(monkeypatch):
    persisted = []

    async def fake_persist_session_snapshot(agent_session):
        persisted.append(agent_session)

    monkeypatch.setattr(
        agent.session_manager,
        "persist_session_snapshot",
        fake_persist_session_snapshot,
    )

    agent_session = SimpleNamespace(
        claude_counted=False,
        session=SimpleNamespace(
            config=SimpleNamespace(model_name="openai/gpt-5.5"),
        ),
    )

    await agent._enforce_premium_model_quota(
        {"user_id": "u1", "plan": "free"},
        agent_session,
    )

    assert agent_session.claude_counted is True
    assert persisted == [agent_session]
    assert await agent.user_quotas.get_claude_used_today("u1") == 1


@pytest.mark.asyncio
async def test_user_quota_response_uses_premium_fields_only(monkeypatch):
    async def fake_get_used_today(user_id):
        assert user_id == "u1"
        return 2

    monkeypatch.setattr(agent.user_quotas, "get_claude_used_today", fake_get_used_today)
    monkeypatch.setattr(agent.user_quotas, "daily_cap_for", lambda plan: 5)

    response = await agent.get_user_quota({"user_id": "u1", "plan": "pro"})

    assert response == {
        "plan": "pro",
        "premium_used_today": 2,
        "premium_daily_cap": 5,
        "premium_remaining": 3,
    }
