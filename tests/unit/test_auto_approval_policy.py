from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.core import agent_loop
from agent.core.cost_estimation import CostEstimate


def _config(**overrides):
    data = {
        "model_name": "moonshotai/Kimi-K2.6",
        "confirm_cpu_jobs": True,
        "auto_file_upload": False,
        "yolo_mode": False,
        **overrides,
    }
    return Config.model_validate(data)


def _session(*, cap=5.0, spent=0.0, enabled=True):
    return SimpleNamespace(
        config=_config(),
        auto_approval_enabled=enabled,
        auto_approval_cost_cap_usd=cap,
        auto_approval_estimated_spend_usd=spent,
        sandbox=None,
    )


@pytest.mark.asyncio
async def test_session_yolo_auto_approves_non_costed_approval_tool():
    decision = await agent_loop._approval_decision(
        "hf_repo_files",
        {"operation": "upload", "path": "README.md"},
        _session(),
    )

    assert decision.requires_approval is False
    assert decision.auto_approved is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["scheduled run", "scheduled uv", "scheduled  run"],
)
async def test_scheduled_hf_jobs_always_require_manual_approval(operation):
    session = _session()
    session.config.yolo_mode = True

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": operation, "script": "print(1)"},
        session,
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert "Scheduled HF jobs" in decision.block_reason
    assert agent_loop._needs_approval(
        "hf_jobs", {"operation": operation}, session.config
    )


@pytest.mark.asyncio
async def test_immediate_hf_job_under_cap_auto_runs(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=2.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large", "timeout": "1h"},
        _session(cap=5.0, spent=1.0),
    )

    assert decision.requires_approval is False
    assert decision.auto_approved is True
    assert decision.estimated_cost_usd == 2.0


@pytest.mark.asyncio
async def test_immediate_hf_job_over_cap_falls_back_to_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=2.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large", "timeout": "1h"},
        _session(cap=5.0, spent=4.0),
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert "exceeds" in decision.block_reason
    assert decision.remaining_cap_usd == 1.0


@pytest.mark.asyncio
async def test_unknown_cost_falls_back_to_approval(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason="No price is available.",
        )

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)

    decision = await agent_loop._approval_decision(
        "sandbox_create",
        {"hardware": "mystery-gpu"},
        _session(),
    )

    assert decision.requires_approval is True
    assert decision.auto_approval_blocked is True
    assert decision.estimated_cost_usd is None


@pytest.mark.asyncio
async def test_batch_reservation_blocks_second_over_budget_job(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=3.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(cap=5.0, spent=0.0)

    first = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large"},
        session,
        reserved_spend_usd=0.0,
    )
    second = await agent_loop._approval_decision(
        "hf_jobs",
        {"operation": "run", "hardware_flavor": "a10g-large"},
        session,
        reserved_spend_usd=first.estimated_cost_usd or 0.0,
    )

    assert first.requires_approval is False
    assert second.requires_approval is True
    assert second.remaining_cap_usd == 2.0


@pytest.mark.asyncio
async def test_manual_approval_does_not_record_spend_when_session_yolo_disabled(
    monkeypatch,
):
    called = False

    async def fake_estimate(*args, **kwargs):
        nonlocal called
        called = True
        return CostEstimate(estimated_cost_usd=2.0, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(enabled=False, cap=5.0, spent=0.0)

    await agent_loop._record_manual_approved_spend_if_needed(
        session,
        "sandbox_create",
        {"hardware": "a10g-large"},
    )

    assert called is False
    assert session.auto_approval_estimated_spend_usd == 0.0


@pytest.mark.asyncio
async def test_manual_approval_records_spend_when_session_yolo_enabled(monkeypatch):
    async def fake_estimate(*args, **kwargs):
        return CostEstimate(estimated_cost_usd=1.25, billable=True)

    monkeypatch.setattr(agent_loop, "estimate_tool_cost", fake_estimate)
    session = _session(enabled=True, cap=5.0, spent=0.5)

    await agent_loop._record_manual_approved_spend_if_needed(
        session,
        "sandbox_create",
        {"hardware": "a10g-large"},
    )

    assert session.auto_approval_estimated_spend_usd == 1.75
