"""LiteLLM kwargs resolution for the model ids this agent accepts.

Kept separate from ``agent_loop`` so tools (research, context compaction, etc.)
can import it without pulling in the whole agent loop / tool router and
creating circular imports.
"""

from agent.core.hf_tokens import get_hf_bill_to, resolve_hf_router_token


def _resolve_hf_router_token(session_hf_token: str | None = None) -> str | None:
    """Backward-compatible private wrapper used by tests and older imports."""
    return resolve_hf_router_token(session_hf_token)


def _patch_litellm_effort_validation() -> None:
    """Neuter LiteLLM 1.83's hardcoded effort-level validation.

    Context: at ``litellm/llms/anthropic/chat/transformation.py:~1443`` the
    Anthropic adapter validates ``output_config.effort ∈ {high, medium,
    low, max}`` and gates ``max`` behind an ``_is_opus_4_6_model`` check
    that only matches the substring ``opus-4-6`` / ``opus_4_6``. Result:

    * ``xhigh`` — valid on Anthropic's real API for Claude 4.7 — is
      rejected pre-flight with "Invalid effort value: xhigh".
    * ``max`` on Opus 4.7 is rejected with "effort='max' is only supported
      by Claude Opus 4.6", even though Opus 4.7 accepts it in practice.

    We don't want to maintain a parallel model table, so we let the
    Anthropic API itself be the validator: widen ``_is_opus_4_6_model``
    to also match ``opus-4-7``+ families, and drop the valid-effort-set
    check entirely. If Anthropic rejects an effort level, we see a 400
    and the cascade walks down — exactly the behavior we want for any
    future model family.

    Removable once litellm ships 1.83.8-stable (which merges PR #25867,
    "Litellm day 0 opus 4.7 support") — see commit 0868a82 on their main
    branch. Until then, this one-time patch is the escape hatch.
    """
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


# Effort levels accepted on the wire.
#   Anthropic (4.6+):  low | medium | high | xhigh | max   (output_config.effort)
#   OpenAI direct:     minimal | low | medium | high | xhigh (reasoning_effort top-level)
#   HF router:         low | medium | high                 (extra_body.reasoning_effort)
#
# We validate *shape* here and let the probe cascade walk down on rejection;
# we deliberately do NOT maintain a per-model capability table.
_ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_OPENAI_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
_HF_EFFORTS = {"low", "medium", "high"}


class UnsupportedEffortError(ValueError):
    """The requested effort isn't valid for this provider's API surface.

    Raised synchronously before any network call so the probe cascade can
    skip levels the provider can't accept (e.g. ``max`` on HF router).
    """


def _resolve_llm_params(
    model_name: str,
    session_hf_token: str | None = None,
    reasoning_effort: str | None = None,
    strict: bool = False,
) -> dict:
    """
    Build LiteLLM kwargs for a given model id.

    • ``anthropic/<model>`` — native thinking config. We bypass LiteLLM's
      ``reasoning_effort`` → ``thinking`` mapping (which lags new Claude
      releases like 4.7 and sends the wrong API shape). Instead we pass
      both ``thinking={"type": "adaptive"}`` and ``output_config=
      {"effort": <level>}`` as top-level kwargs — LiteLLM's Anthropic
      adapter forwards unknown top-level kwargs into the request body
      verbatim (confirmed by live probe; ``extra_body`` does NOT work
      here because Anthropic's API rejects it as "Extra inputs are not
      permitted"). This is the stable API for 4.6 and 4.7. Older
      extended-thinking models that only accept ``thinking.type.enabled``
      will reject this; the probe's cascade catches that and falls back
      to no thinking.

    • ``openai/<model>`` or ``azure/<model>`` — ``reasoning_effort`` forwarded
      as a top-level kwarg. LiteLLM uses the user's API key or Azure credentials.

    • Anything else is treated as a HuggingFace router id. We hit the
      auto-routing OpenAI-compatible endpoint at
      ``https://router.huggingface.co/v1``. The id can be bare or carry an
      HF routing suffix (``:fastest`` / ``:cheapest`` / ``:<provider>``).
      A leading ``huggingface/`` is stripped. ``reasoning_effort`` is
      forwarded via ``extra_body`` (LiteLLM's OpenAI adapter refuses it as
      a top-level kwarg for non-OpenAI models). "minimal" normalizes to
      "low".

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
    if model_name.startswith("anthropic/"):
        params: dict = {"model": model_name}
        if reasoning_effort:
            level = reasoning_effort
            if level == "minimal":
                level = "low"
            if level not in _ANTHROPIC_EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"Anthropic doesn't accept effort={level!r}"
                    )
            else:
                # Adaptive thinking + output_config.effort is the stable
                # Anthropic API for Claude 4.6 / 4.7. Both kwargs are
                # passed top-level: LiteLLM forwards unknown params into
                # the request body for Anthropic, so ``output_config``
                # reaches the API. ``extra_body`` does NOT work here —
                # Anthropic rejects it as "Extra inputs are not
                # permitted".
                params["thinking"] = {"type": "adaptive"}
                params["output_config"] = {"effort": level}
        return params

    if model_name.startswith("bedrock/"):
        # LiteLLM routes ``bedrock/...`` through the Converse adapter, which
        # picks up AWS credentials from the standard env vars
        # (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_REGION``).
        # The Anthropic thinking/effort shape is not forwarded through Converse
        # the same way, so we leave it off for now.
        return {"model": model_name}

    if model_name.startswith(("openai/", "azure/")):
        params = {"model": model_name}
        if reasoning_effort:
            if reasoning_effort not in _OPENAI_EFFORTS:
                if strict:
                    provider = "Azure" if model_name.startswith("azure/") else "OpenAI"
                    raise UnsupportedEffortError(
                        f"{provider} doesn't accept effort={reasoning_effort!r}"
                    )
            else:
                params["reasoning_effort"] = reasoning_effort
        return params

    hf_model = model_name.removeprefix("huggingface/")
    api_key = _resolve_hf_router_token(session_hf_token)
    params = {
        "model": f"openai/{hf_model}",
        "api_base": "https://router.huggingface.co/v1",
        "api_key": api_key,
    }
    if bill_to := get_hf_bill_to():
        params["extra_headers"] = {"X-HF-Bill-To": bill_to}
    if reasoning_effort:
        hf_level = "low" if reasoning_effort == "minimal" else reasoning_effort
        if hf_level not in _HF_EFFORTS:
            if strict:
                raise UnsupportedEffortError(
                    f"HF router doesn't accept effort={hf_level!r}"
                )
        else:
            params["extra_body"] = {"reasoning_effort": hf_level}
    return params
