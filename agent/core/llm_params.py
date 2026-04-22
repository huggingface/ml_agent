"""LiteLLM kwargs resolution for the model ids this agent accepts.

Kept separate from ``agent_loop`` so tools (research, context compaction, etc.)
can import it without pulling in the whole agent loop / tool router and
creating circular imports.
"""

import os


# HF router reasoning models only accept "low" | "medium" | "high" (e.g.
# MiniMax M2 actually *requires* reasoning to be enabled). OpenAI's GPT-5
# also accepts "minimal" for near-zero thinking. We map "minimal" to "low"
# for HF so the user doesn't get a 400.
_HF_ALLOWED_EFFORTS = {"low", "medium", "high"}


def _resolve_llm_params(
    model_name: str,
    session_hf_token: str | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """
    Build LiteLLM kwargs for a given model id.

    • ``anthropic/<model>`` / ``openai/<model>`` — passed straight through; the
      user's own ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` env vars are picked
      up by LiteLLM. ``reasoning_effort`` is forwarded as a top-level param
      (GPT-5 / o-series accept "minimal" | "low" | "medium" | "high"; Claude
      extended-thinking models accept "low" | "medium" | "high" and LiteLLM
      translates to the thinking config).

    • Anything else is treated as a HuggingFace router id. We hit the
      auto-routing OpenAI-compatible endpoint at
      ``https://router.huggingface.co/v1``, which bypasses LiteLLM's stale
      per-provider HF adapter entirely. The id can be bare or carry an HF
      routing suffix:

          MiniMaxAI/MiniMax-M2.7              # auto = fastest + failover
          MiniMaxAI/MiniMax-M2.7:cheapest
          moonshotai/Kimi-K2.6:novita         # pin a specific provider

      A leading ``huggingface/`` is stripped for convenience. ``reasoning_effort``
      is forwarded via ``extra_body`` (LiteLLM's OpenAI adapter refuses it as a
      top-level kwarg for non-OpenAI models). "minimal" is normalized to "low".

    Token precedence (first non-empty wins):
      1. INFERENCE_TOKEN env — shared key on the hosted Space (inference is
         free for users, billed to the Space owner via ``X-HF-Bill-To``).
      2. session.hf_token — the user's own token (CLI / OAuth / cache file).
      3. HF_TOKEN env — belt-and-suspenders fallback for CLI users.
    """
    if model_name.startswith(("anthropic/", "openai/")):
        params: dict = {"model": model_name}
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        return params

    hf_model = model_name.removeprefix("huggingface/")
    api_key = (
        os.environ.get("INFERENCE_TOKEN")
        or session_hf_token
        or os.environ.get("HF_TOKEN")
    )
    params = {
        "model": f"openai/{hf_model}",
        "api_base": "https://router.huggingface.co/v1",
        "api_key": api_key,
    }
    if os.environ.get("INFERENCE_TOKEN"):
        params["extra_headers"] = {"X-HF-Bill-To": "huggingface"}
    if reasoning_effort:
        hf_level = "low" if reasoning_effort == "minimal" else reasoning_effort
        if hf_level in _HF_ALLOWED_EFFORTS:
            params["extra_body"] = {"reasoning_effort": hf_level}
    return params
