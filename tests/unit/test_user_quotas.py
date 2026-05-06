"""Tests for backend/user_quotas.py — the in-memory Claude daily-quota store."""

import asyncio
import sys
from pathlib import Path

import pytest

# The backend package isn't on sys.path by default; add it so we can import
# the module under test without pulling in the whole FastAPI app.
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import user_quotas  # noqa: E402
from agent.core.session_persistence import NoopSessionStore, _reset_store_for_tests  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_store():
    """Fresh in-memory store per test."""
    user_quotas._reset_for_tests()
    yield
    user_quotas._reset_for_tests()


def test_daily_cap_for_known_plans():
    assert user_quotas.daily_cap_for("free") == user_quotas.CLAUDE_FREE_DAILY
    assert user_quotas.daily_cap_for("pro") == user_quotas.CLAUDE_PRO_DAILY
    assert user_quotas.daily_cap_for("org") == user_quotas.CLAUDE_PRO_DAILY


def test_daily_cap_for_unknown_or_missing_defaults_to_free():
    assert user_quotas.daily_cap_for(None) == user_quotas.CLAUDE_FREE_DAILY
    assert user_quotas.daily_cap_for("") == user_quotas.CLAUDE_FREE_DAILY
    # Anything we don't recognize as the Pro/Org tier gets the Pro cap because
    # the function's contract is "free" is the only downgraded tier. If that
    # ever flips, this test will flip too — adjust consciously.
    assert user_quotas.daily_cap_for("mystery") == user_quotas.CLAUDE_PRO_DAILY


@pytest.mark.asyncio
async def test_increment_and_read_back_same_day():
    assert await user_quotas.get_claude_used_today("u1") == 0
    assert await user_quotas.increment_claude("u1") == 1
    assert await user_quotas.increment_claude("u1") == 2
    assert await user_quotas.get_claude_used_today("u1") == 2


@pytest.mark.asyncio
async def test_independent_users_do_not_share_counts():
    await user_quotas.increment_claude("alice")
    await user_quotas.increment_claude("alice")
    await user_quotas.increment_claude("bob")
    assert await user_quotas.get_claude_used_today("alice") == 2
    assert await user_quotas.get_claude_used_today("bob") == 1


@pytest.mark.asyncio
async def test_stale_day_resets_before_next_read():
    await user_quotas.increment_claude("u1")
    # Simulate yesterday's entry still in the store.
    user_quotas._claude_counts["u1"] = ("2000-01-01", 99)
    assert await user_quotas.get_claude_used_today("u1") == 0
    # And a fresh increment starts from 0.
    assert await user_quotas.increment_claude("u1") == 1


@pytest.mark.asyncio
async def test_concurrent_increments_under_lock_do_not_lose_writes():
    """50 coroutines bumping the same user must land at exactly 50."""
    await asyncio.gather(*[user_quotas.increment_claude("race") for _ in range(50)])
    assert await user_quotas.get_claude_used_today("race") == 50


@pytest.mark.asyncio
async def test_try_increment_returns_none_at_cap():
    assert await user_quotas.try_increment_claude("freebie", 1) == 1
    assert await user_quotas.try_increment_claude("freebie", 1) is None
    assert await user_quotas.get_claude_used_today("freebie") == 1


@pytest.mark.asyncio
async def test_try_increment_delegates_cap_to_enabled_store():
    class StoreAtCap(NoopSessionStore):
        enabled = True

        async def try_increment_quota(self, user_id: str, day: str, cap: int):
            assert user_id == "mongo-user"
            assert cap == 1
            return None

        async def get_quota(self, user_id: str, day: str):
            return 1

    _reset_store_for_tests(StoreAtCap())

    assert await user_quotas.try_increment_claude("mongo-user", 1) is None
    assert await user_quotas.get_claude_used_today("mongo-user") == 1
    assert "mongo-user" not in user_quotas._claude_counts


@pytest.mark.asyncio
async def test_refund_decrements_and_drops_entry_at_zero():
    await user_quotas.increment_claude("u1")
    assert await user_quotas.get_claude_used_today("u1") == 1
    await user_quotas.refund_claude("u1")
    assert await user_quotas.get_claude_used_today("u1") == 0
    assert "u1" not in user_quotas._claude_counts


@pytest.mark.asyncio
async def test_refund_on_nonexistent_user_is_noop():
    await user_quotas.refund_claude("ghost")  # should not raise
    assert await user_quotas.get_claude_used_today("ghost") == 0


@pytest.mark.asyncio
async def test_refund_on_stale_day_resets_rather_than_underflow():
    user_quotas._claude_counts["u1"] = ("2000-01-01", 5)
    await user_quotas.refund_claude("u1")
    # Stale entry dropped; today's count stays 0.
    assert await user_quotas.get_claude_used_today("u1") == 0


@pytest.mark.asyncio
async def test_free_user_cap_reached_at_one():
    cap = user_quotas.daily_cap_for("free")
    used = await user_quotas.increment_claude("freebie")
    assert used == 1
    assert used >= cap  # first bump exhausts the free tier (cap=1)


@pytest.mark.asyncio
async def test_pro_user_cap_reached_at_twenty():
    cap = user_quotas.daily_cap_for("pro")
    assert cap == 20
    for i in range(1, 21):
        assert await user_quotas.increment_claude("pro_user") == i
    # 21st would exceed — the gate in routes/agent.py enforces this; here
    # we just confirm the counter tracks past the cap so that check works.
    assert await user_quotas.increment_claude("pro_user") == 21
