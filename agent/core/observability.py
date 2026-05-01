"""Opt-in third-party observability hooks for litellm calls.

Today ml-intern's primary telemetry pipeline writes ``Event``s to a Hugging
Face Dataset (see ``agent/core/telemetry.py``). This module is a small,
opt-in side channel that lets operators also stream LLM traces to a
LangFuse instance (self-hosted or SaaS) via litellm's OTEL callback.

Activation requires the LangFuse host plus both keys: either
``LANGFUSE_HOST`` or ``LANGFUSE_BASE_URL`` (the SDK v4 docs use the latter),
together with ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``. With any
of them missing, this module is a no-op and behavior is identical to today.
The host is mandatory by design — see issue #196 for the privacy rationale.
"""

from __future__ import annotations

import logging
import os

import litellm

logger = logging.getLogger(__name__)


def setup_langfuse() -> None:
    """Register litellm's LangFuse OTEL callback if host + keys are set.

    Accepts either ``LANGFUSE_HOST`` or ``LANGFUSE_BASE_URL`` for the host:
    Langfuse SDK v4's docs issue credentials as ``LANGFUSE_BASE_URL``, but
    litellm's callback only reads ``LANGFUSE_HOST`` — so we mirror the value
    into ``LANGFUSE_HOST`` when only ``BASE_URL`` was set.

    Uses the ``langfuse_otel`` callback rather than the legacy ``langfuse``
    one because the legacy integration in current litellm releases breaks
    against Langfuse SDK v4 (``module 'langfuse' has no attribute 'version'``).
    The OTEL path works against both v3 and v4.

    Idempotent: ``load_config`` runs multiple times per process (CLI start
    plus backend module-init), so guard against double-registration.
    """
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
    if not (
        host
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        return
    # litellm only reads LANGFUSE_HOST, so propagate the value if the
    # operator set only LANGFUSE_BASE_URL.
    os.environ.setdefault("LANGFUSE_HOST", host)
    if "langfuse_otel" not in litellm.success_callback:
        litellm.success_callback.append("langfuse_otel")
    if "langfuse_otel" not in litellm.failure_callback:
        litellm.failure_callback.append("langfuse_otel")
    logger.info("LangFuse observability enabled (host=%s)", host)
