"""LiteLLM kwargs resolution for the model ids this agent accepts.

Kept separate from ``agent_loop`` so tools (research, context compaction, etc.)
can import it without pulling in the whole agent loop / tool router and
creating circular imports.
"""

from agent.core.hf_tokens import resolve_hf_router_token
from agent.core.provider_adapters import (
    UnsupportedEffortError,
    resolve_adapter,
)


def _resolve_hf_router_token(session_hf_token: str | None = None) -> str | None:
    """Backward-compatible private wrapper used by tests and older imports."""
    return resolve_hf_router_token(session_hf_token)


__all__ = [
    "UnsupportedEffortError",
    "_resolve_hf_router_token",
    "_resolve_llm_params",
]


def _patch_litellm_effort_validation() -> None:
    """Patch LiteLLM's Anthropic effort validation for Claude Opus 4.7."""
    try:
        from litellm.llms.anthropic.chat import transformation as _t
    except Exception:
        return

    cfg = getattr(_t, "AnthropicConfig", None)
    if cfg is None:
        return

    original = getattr(cfg, "_is_opus_4_6_model", None)
    if original is None or getattr(original, "_hf_agent_patched", False):
        return

    def _widened(model: str) -> bool:
        m = model.lower()
        # Original 4.6 match plus any future Opus >= 4.6. We only need this
        # to return True for families where "max" / "xhigh" are acceptable
        # at the API; the cascade handles the case when they're not.
        return any(
            v in m
            for v in (
                "opus-4-6",
                "opus_4_6",
                "opus-4.6",
                "opus_4.6",
                "opus-4-7",
                "opus_4_7",
                "opus-4.7",
                "opus_4.7",
            )
        )

    _widened._hf_agent_patched = True  # type: ignore[attr-defined]
    cfg._is_opus_4_6_model = staticmethod(_widened)


_patch_litellm_effort_validation()


def _resolve_llm_params(
    model_name: str,
    session_hf_token: str | None = None,
    reasoning_effort: str | None = None,
    strict: bool = False,
) -> dict:
    """Build LiteLLM kwargs for a given model id.

    Delegates to the matching provider adapter.

    ``strict=True`` raises ``UnsupportedEffortError`` when the requested
    effort isn't in the provider's accepted set, instead of silently
    dropping it. The probe cascade uses strict mode so it can walk down
    (``max`` → ``xhigh`` → ``high`` …) without making an API call. Regular
    runtime callers leave ``strict=False``, so a stale cached effort
    can't crash a turn — it just doesn't get sent.

    Token precedence (first non-empty wins):
      1. INFERENCE_TOKEN env — shared key on the hosted Space (inference is
         free for users, billed to the Space owner via ``X-HF-Bill-To``).
      2. session.hf_token — the user's own token (CLI / OAuth / cache file).
      3. huggingface_hub cache — ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` /
         local ``hf auth login`` cache.
    """
    adapter = resolve_adapter(model_name)
    if adapter is None:
        raise ValueError(f"No provider adapter for model: {model_name}")
    return adapter.build_params(
        model_name,
        session_hf_token=session_hf_token,
        reasoning_effort=reasoning_effort,
        strict=strict,
    )
