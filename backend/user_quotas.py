"""In-memory daily quota for Claude session creations.

Tracks per-user Claude session starts against a daily cap derived from the
user's HF plan. Caps reset at UTC midnight; the store itself is in-process
and wipes on restart (deliberate — the cost of occasional over-subsidy at
restart is much lower than running a DB).

Unit: session *creations*, not messages. A user who selects Claude in a new
session consumes one quota point; switching an existing Claude session to
Claude again doesn't (`AgentSession.claude_counted` guards that).

Cap tiers:
  free user   → CLAUDE_FREE_DAILY (1)
  pro / org   → CLAUDE_PRO_DAILY  (20)
"""

import asyncio
import os
from datetime import UTC, datetime

CLAUDE_FREE_DAILY: int = int(os.environ.get("CLAUDE_FREE_DAILY", "1"))
CLAUDE_PRO_DAILY: int = int(os.environ.get("CLAUDE_PRO_DAILY", "20"))

# user_id -> (day_utc_iso, count_for_that_day)
_claude_counts: dict[str, tuple[str, int]] = {}
_lock = asyncio.Lock()


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def daily_cap_for(plan: str | None) -> int:
    """Return the daily Claude-session cap for the given plan."""
    return CLAUDE_FREE_DAILY if (plan or "free") == "free" else CLAUDE_PRO_DAILY


async def get_claude_used_today(user_id: str) -> int:
    """Return today's Claude session count for the user (0 if none / stale day)."""
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
    async with _lock:
        today = _today()
        day, count = _claude_counts.get(user_id, (today, 0))
        if day != today:
            count = 0
        count += 1
        _claude_counts[user_id] = (today, count)
        return count


async def refund_claude(user_id: str) -> None:
    """Decrement today's count — used when session creation fails after a successful gate."""
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
