"""Per-provider async rate limiting for LLM calls.

Some providers (notably the OpenCode Zen free tier) advertise hard
requests-per-minute caps and respond with 429s once the budget is
exhausted. We respect those caps client-side with a small token-bucket
limiter so the agent doesn't get tarpitted on every turn.

Design:
* One :class:`_TokenBucket` per provider key.
* :func:`acquire` is the public entrypoint — it picks the right bucket
  from the model id and ``await``s a token. Models without a configured
  cap (Anthropic, Copilot, HF router, …) hit the no-op fast path.
* The bucket is leaky: tokens regenerate continuously at
  ``capacity / window`` per second, so a burst of ``capacity`` requests
  is allowed but the rolling rate stays at the configured cap.

Caps live in ``_PROVIDER_LIMITS`` keyed by the model-id prefix. To add a
new cap, drop in another entry — no agent_loop changes required.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class _Limit:
    """Static config for one provider's rate cap."""
    capacity: float       # max tokens in the bucket (== max burst size)
    window_s: float       # seconds over which ``capacity`` tokens regenerate


# OpenCode Zen free tier is documented at 50 requests / minute per API
# key. We stay one request below the cap to leave headroom for the
# probe ping and any retry storms LiteLLM does internally.
_PROVIDER_LIMITS: dict[str, _Limit] = {
    "opencode": _Limit(capacity=49, window_s=60.0),
}


class _TokenBucket:
    """Async token-bucket. Refills continuously at ``capacity/window``/s."""

    def __init__(self, limit: _Limit) -> None:
        self._capacity = limit.capacity
        self._refill_per_s = limit.capacity / limit.window_s
        self._tokens = limit.capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until at least one token is available, then consume it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity,
                        self._tokens + elapsed * self._refill_per_s,
                    )
                    self._last_refill = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                # Time until the next whole token is available.
                wait_s = (1 - self._tokens) / self._refill_per_s
            # Sleep outside the lock so other coroutines can still
            # update the bucket (e.g. tests injecting tokens).
            await asyncio.sleep(wait_s)


_BUCKETS: dict[str, _TokenBucket] = {}


def _provider_key(model_name: str) -> str | None:
    """Return the limiter key for ``model_name`` or ``None`` if uncapped.

    Match is by prefix on the model id (``opencode/...``), which mirrors
    how :func:`agent.core.llm_params._resolve_llm_params` dispatches.
    """
    if not model_name:
        return None
    head = model_name.split("/", 1)[0]
    return head if head in _PROVIDER_LIMITS else None


async def acquire(model_name: str) -> None:
    """Wait for a slot in the rate limiter for ``model_name``.

    No-op for providers without a configured cap, so callers can invoke
    this unconditionally before every ``acompletion`` call.
    """
    key = _provider_key(model_name)
    if key is None:
        return
    bucket = _BUCKETS.get(key)
    if bucket is None:
        bucket = _TokenBucket(_PROVIDER_LIMITS[key])
        _BUCKETS[key] = bucket
    await bucket.acquire()


def reset() -> None:
    """Drop all buckets. Test-only; not used at runtime."""
    _BUCKETS.clear()
