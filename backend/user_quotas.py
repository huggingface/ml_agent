"""Daily quota for premium model session creations.

Tracks per-user premium model session starts against a daily cap derived from
the user's HF plan. MongoDB is the source of truth when configured; the
in-process dict remains the fallback for local/dev/test runs.

The public names still say ``claude`` because this quota bucket originally
only covered Claude and the persisted session field uses that name.

Unit: session *creations*, not messages. A user who sends with a premium model
in a new session consumes one quota point; switching an already-counted session
back to a premium model doesn't (`AgentSession.claude_counted` guards that).

Cap tiers:
  free user   → CLAUDE_FREE_DAILY (1)
  pro user    → CLAUDE_PRO_DAILY  (20)
"""

import asyncio
import os
from datetime import UTC, datetime

from agent.core.session_persistence import (
    NoopSessionStore,
    get_session_store,
    _reset_store_for_tests,
)

CLAUDE_FREE_DAILY: int = int(os.environ.get("CLAUDE_FREE_DAILY", "1"))
CLAUDE_PRO_DAILY: int = int(os.environ.get("CLAUDE_PRO_DAILY", "20"))

# user_id -> (day_utc_iso, count_for_that_day)
_claude_counts: dict[str, tuple[str, int]] = {}
_lock = asyncio.Lock()


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def daily_cap_for(plan: str | None) -> int:
    """Return the daily Claude-session cap for the given plan."""
    return CLAUDE_PRO_DAILY if plan == "pro" else CLAUDE_FREE_DAILY


async def get_claude_used_today(user_id: str) -> int:
    """Return today's Claude session count for the user (0 if none / stale day)."""
    store = get_session_store()
    if getattr(store, "enabled", False):
        db_count = await store.get_quota(user_id, _today())
        return db_count or 0

    async with _lock:
        entry = _claude_counts.get(user_id)
        if entry is None:
            return 0
        day, count = entry
        if day != _today():
            # Stale day — drop the entry so the first increment starts fresh.
            _claude_counts.pop(user_id, None)
            return 0
        return count


async def increment_claude(user_id: str) -> int:
    """Bump today's Claude session count for the user. Returns the new value."""
    store = get_session_store()
    if getattr(store, "enabled", False):
        db_count = await store.try_increment_quota(user_id, _today(), cap=10**9)
        return db_count or 0

    async with _lock:
        today = _today()
        day, count = _claude_counts.get(user_id, (today, 0))
        if day != today:
            count = 0
        count += 1
        _claude_counts[user_id] = (today, count)
        return count


async def try_increment_claude(user_id: str, cap: int) -> int | None:
    """Atomically bump today's count if below *cap*.

    Returns the new count, or None when the user is already at the cap.
    """
    store = get_session_store()
    if getattr(store, "enabled", False):
        return await store.try_increment_quota(user_id, _today(), cap)

    async with _lock:
        today = _today()
        day, count = _claude_counts.get(user_id, (today, 0))
        if day != today:
            count = 0
        if count >= cap:
            return None
        count += 1
        _claude_counts[user_id] = (today, count)
        return count


async def refund_claude(user_id: str) -> None:
    """Decrement today's count — used when session creation fails after a successful gate."""
    store = get_session_store()
    if getattr(store, "enabled", False):
        await store.refund_quota(user_id, _today())
        return

    async with _lock:
        entry = _claude_counts.get(user_id)
        if entry is None:
            return
        day, count = entry
        if day != _today():
            _claude_counts.pop(user_id, None)
            return
        new_count = max(0, count - 1)
        if new_count == 0:
            _claude_counts.pop(user_id, None)
        else:
            _claude_counts[user_id] = (day, new_count)


def _reset_for_tests() -> None:
    """Test-only: clear the in-memory store."""
    _claude_counts.clear()
    _reset_store_for_tests(NoopSessionStore())
