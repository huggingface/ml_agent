"""Shared LLM error classification and user-facing messages."""

from __future__ import annotations

from typing import Literal

LlmErrorType = Literal[
    "auth",
    "credits",
    "model",
    "provider",
    "rate_limit",
    "network",
    "unknown",
]

_AUTH_MARKERS = (
    "authentication failed",
    "authentication_error",
    "authentication error",
    "unauthorized",
    "invalid x-api-key",
    "invalid api key",
    "incorrect api key",
    "didn't provide an api key",
    "did not provide an api key",
    "no api key provided",
    "provide your api key",
    "x-api-key header is required",
    "api key header is required",
    "api key required",
    "api key is missing or invalid",
    "api_key_invalid",
    "401",
)
_CREDITS_MARKERS = (
    "insufficient credit",
    "insufficient credits",
    "out of credits",
    "insufficient_quota",
    "credit balance is too low",
    "balance is too low",
    "purchase credits",
    "plans & billing",
    "quota",
    "billing",
    "payment required",
    "402",
)
_RATE_LIMIT_MARKERS = ("429", "rate limit", "too many requests")
_NETWORK_MARKERS = (
    "timeout",
    "timed out",
    "connect",
    "connection error",
    "connection refused",
    "connection reset",
    "network",
    "service unavailable",
    "bad gateway",
    "overloaded",
    "capacity",
)


def _has_any(err_str: str, markers: tuple[str, ...]) -> bool:
    return any(marker in err_str for marker in markers)


def classify_llm_error(error: Exception) -> LlmErrorType:
    """Classify common provider/API failures from the exception text."""
    err_str = str(error).lower()

    if _has_any(err_str, _AUTH_MARKERS):
        return "auth"
    if _has_any(err_str, _CREDITS_MARKERS):
        return "credits"
    if "not supported by provider" in err_str or "no provider supports" in err_str:
        return "provider"
    if "model_not_found" in err_str or "unknown model" in err_str:
        return "model"
    if "model" in err_str and (
        "not found" in err_str
        or "does not exist" in err_str
        or "not available" in err_str
    ):
        return "model"
    if _has_any(err_str, _RATE_LIMIT_MARKERS):
        return "rate_limit"
    if _has_any(err_str, _NETWORK_MARKERS):
        return "network"
    return "unknown"


def friendly_llm_error_message(error: Exception) -> str | None:
    """Return a clean user-facing message for common LLM failures."""
    error_type = classify_llm_error(error)

    if error_type == "auth":
        return (
            "Authentication failed — your API key is missing or invalid.\n\n"
            "To fix this, set the API key for your model provider:\n"
            "  • Anthropic:   export ANTHROPIC_API_KEY=sk-...\n"
            "  • OpenAI:      export OPENAI_API_KEY=sk-...\n"
            "  • HF Router:   export HF_TOKEN=hf_...\n\n"
            "You can also add it to a .env file in the project root.\n"
            "To switch models, use the /model command."
        )
    if error_type == "credits":
        return (
            "Insufficient API credits or quota for this model/provider.\n\n"
            "Check billing for the current provider, or switch models with /model."
        )
    if error_type == "provider":
        return (
            "The model isn't served by the provider you pinned.\n\n"
            "Drop the ':<provider>' suffix to let the HF router auto-pick a "
            "provider, or use '/model' (no arg) to see which providers host "
            "which models."
        )
    if error_type == "model":
        return (
            "Model not found. Use '/model' to list suggestions, or paste an "
            "HF model id like 'MiniMaxAI/MiniMax-M2.7'. Availability is shown "
            "when you switch."
        )
    if error_type == "rate_limit":
        return (
            "Rate limit reached. Wait a moment and retry, or switch models/providers "
            "with /model."
        )
    if error_type == "network":
        return "The model provider is unavailable or timed out. Retry in a moment."
    return None


def render_llm_error_message(error: Exception) -> str:
    """Return the message safe to show to users."""
    return friendly_llm_error_message(error) or str(error)


def health_error_type(error: Exception) -> str:
    """Map LLM failures to the backend health endpoint error_type values."""
    error_type = classify_llm_error(error)
    if error_type in {"auth", "credits", "rate_limit", "network"}:
        return error_type
    return "unknown"
