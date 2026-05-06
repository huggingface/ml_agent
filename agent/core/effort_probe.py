"""Probe-and-cascade for reasoning effort on /model switch.

We don't maintain a per-model capability table. Instead, the first time a
user picks a model we fire a 1-token ping with the same params we'd use
for real and walk down a cascade (``max`` → ``xhigh`` → ``high`` → …)
until the provider stops rejecting us. The result is cached per-model on
the session, so real messages don't pay the probe cost again.

Three outcomes, classified from the 400 error text:

* success → cache the effort that worked
* ``"thinking ... not supported"`` → model doesn't do thinking at all;
  cache ``None`` so we stop sending thinking params
* ``"effort ... invalid"`` / synonyms → cascade walks down and retries

Transient errors (5xx, timeout, connection reset) bubble out as
``ProbeInconclusive`` so the caller can complete the switch with a
warning instead of blocking on a flaky provider.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from litellm import acompletion

from agent.core.llm_params import UnsupportedEffortError, _resolve_llm_params

logger = logging.getLogger(__name__)


# Cascade: for each user-stated preference, the ordered list of levels to
# try. First success wins. ``max`` is Anthropic-only; ``xhigh`` is also
# supported on current OpenAI GPT-5 models. Providers that don't accept a
# requested level raise ``UnsupportedEffortError`` synchronously (no wasted
# network round-trip) and we advance to the next level.
_EFFORT_CASCADE: dict[str, list[str]] = {
    "max": ["max", "xhigh", "high", "medium", "low"],
    "xhigh": ["xhigh", "high", "medium", "low"],
    "high": ["high", "medium", "low"],
    "medium": ["medium", "low"],
    "minimal": ["minimal", "low"],
    "low": ["low"],
}

_PROBE_TIMEOUT = 15.0
# Keep the probe cheap, but high enough that frontier reasoning models can
# finish a trivial reply instead of tripping a false "output limit reached"
# error during capability detection.
_PROBE_MAX_TOKENS = 64


class ProbeInconclusive(Exception):
    """The probe couldn't reach a verdict (transient network / provider error).

    Caller should complete the switch with a warning — the next real call
    will re-surface the error if it's persistent.
    """


@dataclass
class ProbeOutcome:
    """What the probe learned. ``effective_effort`` semantics match the cache:

    * str → send this level
    * None → model doesn't support thinking; strip it
    """

    effective_effort: str | None
    attempts: int
    elapsed_ms: int
    note: str | None = None  # e.g. "max not supported, falling back"


def _is_thinking_unsupported(e: Exception) -> bool:
    """Model rejected any thinking config.

    Matches Anthropic's 'thinking.type.enabled is not supported for this
    model' as well as the adaptive variant. Substring-match because the
    exact wording shifts across API versions.
    """
    s = str(e).lower()
    return "thinking" in s and "not supported" in s


def _is_invalid_effort(e: Exception) -> bool:
    """The requested effort level isn't accepted for this model.

    Covers both API responses (Anthropic/OpenAI 400 with "invalid", "must
    be one of", etc.) and LiteLLM's local validation that fires *before*
    the request (e.g. "effort='max' is only supported by Claude Opus 4.6"
    — LiteLLM knows max is Opus-4.6-only and raises synchronously). The
    cascade walks down on either.

    Explicitly returns False when the message is really about thinking
    itself (e.g. Anthropic's 4.7 error mentions ``output_config.effort``
    in its fix hint, but the actual failure is ``thinking.type.enabled``
    being unsupported). That case is caught by ``_is_thinking_unsupported``.
    """
    if _is_thinking_unsupported(e):
        return False
    s = str(e).lower()
    if "effort" not in s and "output_config" not in s:
        return False
    return any(
        phrase in s
        for phrase in (
            "invalid",
            "not supported",
            "must be one of",
            "not a valid",
            "unrecognized",
            "unknown",
            # LiteLLM's own pre-flight validation phrasing.
            "only supported by",
            "is only supported",
        )
    )


def _is_transient(e: Exception) -> bool:
    """Network / provider-side flake. Keep in sync with agent_loop's list.

    Also matches by type for ``asyncio.TimeoutError`` — its ``str(e)`` is
    empty, so substring matching alone misses it.
    """
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return True
    s = str(e).lower()
    return any(
        p in s
        for p in (
            "timeout",
            "timed out",
            "429",
            "rate limit",
            "503",
            "service unavailable",
            "502",
            "bad gateway",
            "500",
            "internal server error",
            "overloaded",
            "capacity",
            "connection reset",
            "connection refused",
            "connection error",
            "eof",
            "broken pipe",
        )
    )


async def probe_effort(
    model_name: str,
    preference: str | None,
    hf_token: str | None,
    session: Any = None,
) -> ProbeOutcome:
    """Walk the cascade for ``preference`` on ``model_name``.

    Returns the first effort the provider accepts, or ``None`` if it
    rejects thinking altogether. Raises ``ProbeInconclusive`` only for
    transient errors (5xx, timeout) — persistent 4xx that aren't thinking/
    effort related bubble as the original exception so callers can surface
    them (auth, model-not-found, quota, etc.).

    ``session`` is optional; when provided, each successful probe attempt
    is recorded via ``telemetry.record_llm_call(kind="effort_probe")`` so
    the cost shows up in the session's ``total_cost_usd``. Failed probes
    (rejected by the provider) typically aren't billed, so we only record
    on success.
    """
    loop = asyncio.get_event_loop()
    start = loop.time()
    attempts = 0

    if not preference:
        # User explicitly turned effort off — nothing to probe. A bare
        # ping with no thinking params is pointless; just report "off".
        return ProbeOutcome(effective_effort=None, attempts=0, elapsed_ms=0)

    cascade = _EFFORT_CASCADE.get(preference, [preference])
    skipped: list[str] = []  # levels the provider rejected synchronously

    last_error: Exception | None = None
    for effort in cascade:
        try:
            params = _resolve_llm_params(
                model_name,
                hf_token,
                reasoning_effort=effort,
                strict=True,
            )
        except UnsupportedEffortError:
            # Provider can't even accept this effort name (e.g. "max" on
            # HF router). Skip without a network call.
            skipped.append(effort)
            continue

        attempts += 1
        try:
            _t0 = time.monotonic()
            response = await asyncio.wait_for(
                acompletion(
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=_PROBE_MAX_TOKENS,
                    stream=False,
                    **params,
                ),
                timeout=_PROBE_TIMEOUT,
            )
            if session is not None:
                # Best-effort telemetry — never let a logging blip propagate
                # out of the probe and break model switching.
                try:
                    from agent.core import telemetry

                    await telemetry.record_llm_call(
                        session,
                        model=model_name,
                        response=response,
                        latency_ms=int((time.monotonic() - _t0) * 1000),
                        finish_reason=response.choices[0].finish_reason
                        if response.choices
                        else None,
                        kind="effort_probe",
                    )
                except Exception as _telem_err:
                    logger.debug("effort_probe telemetry failed: %s", _telem_err)
        except Exception as e:
            last_error = e
            if _is_thinking_unsupported(e):
                elapsed = int((loop.time() - start) * 1000)
                return ProbeOutcome(
                    effective_effort=None,
                    attempts=attempts,
                    elapsed_ms=elapsed,
                    note="model doesn't support reasoning, dropped",
                )
            if _is_invalid_effort(e):
                logger.debug(
                    "probe: %s rejected effort=%s, trying next", model_name, effort
                )
                continue
            if _is_transient(e):
                raise ProbeInconclusive(str(e)) from e
            # Persistent non-thinking 4xx (auth, quota, model-not-found) —
            # let the caller classify & surface.
            raise
        else:
            elapsed = int((loop.time() - start) * 1000)
            note = None
            if effort != preference:
                note = f"{preference} not supported, using {effort}"
            return ProbeOutcome(
                effective_effort=effort,
                attempts=attempts,
                elapsed_ms=elapsed,
                note=note,
            )

    # Cascade exhausted without a success. This only happens when every
    # level was either rejected synchronously (``UnsupportedEffortError``,
    # e.g. preference=max on HF and we also somehow filtered all others)
    # or the provider 400'd ``invalid effort`` on every level.
    elapsed = int((loop.time() - start) * 1000)
    if last_error is not None and not _is_invalid_effort(last_error):
        raise last_error
    note = (
        "no effort level accepted — proceeding without thinking"
        if not skipped
        else f"provider rejected all efforts ({', '.join(skipped)})"
    )
    return ProbeOutcome(
        effective_effort=None,
        attempts=attempts,
        elapsed_ms=elapsed,
        note=note,
    )
