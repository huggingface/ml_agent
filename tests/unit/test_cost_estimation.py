from types import SimpleNamespace

import pytest

from agent.core import cost_estimation


def test_parse_timeout_hours_common_units():
    assert cost_estimation.parse_timeout_hours(None) == 0.5
    assert cost_estimation.parse_timeout_hours("30m") == 0.5
    assert cost_estimation.parse_timeout_hours("3h") == 3
    assert cost_estimation.parse_timeout_hours(3600) == 1
    assert cost_estimation.parse_timeout_hours("not-a-duration") is None


@pytest.mark.asyncio
async def test_estimate_hf_job_cost_uses_catalog_price(monkeypatch):
    async def fake_catalog():
        return {"a100-large": 4.0}

    monkeypatch.setattr(cost_estimation, "hf_jobs_price_catalog", fake_catalog)

    estimate = await cost_estimation.estimate_hf_job_cost(
        {"hardware_flavor": "a100-large", "timeout": "8h"}
    )

    assert estimate.estimated_cost_usd == 32.0
    assert estimate.billable is True


@pytest.mark.asyncio
async def test_estimate_hf_job_cost_blocks_unknown_price(monkeypatch):
    async def fake_catalog():
        return {}

    monkeypatch.setattr(cost_estimation, "hf_jobs_price_catalog", fake_catalog)

    estimate = await cost_estimation.estimate_hf_job_cost(
        {"hardware_flavor": "mystery-gpu", "timeout": "30m"}
    )

    assert estimate.estimated_cost_usd is None
    assert estimate.billable is True
    assert "No price" in estimate.block_reason


@pytest.mark.asyncio
async def test_estimate_sandbox_cost_is_zero_for_existing_or_cpu_basic():
    existing = await cost_estimation.estimate_sandbox_cost(
        {"hardware": "a100-large"},
        session=SimpleNamespace(sandbox=object()),
    )
    cpu = await cost_estimation.estimate_sandbox_cost({"hardware": "cpu-basic"})

    assert existing.estimated_cost_usd == 0.0
    assert existing.billable is False
    assert cpu.estimated_cost_usd == 0.0
    assert cpu.billable is False
