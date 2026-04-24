import pytest

from agent.core.llm_params import _resolve_llm_params
from agent.core.model_switcher import is_valid_model_id
from agent.core.provider_adapters import (
    UnsupportedEffortError,
    is_valid_model_name,
)


# -- Anthropic adapter -------------------------------------------------------


def test_anthropic_adapter_builds_thinking_config():
    params = _resolve_llm_params("anthropic/claude-opus-4-6", reasoning_effort="high")

    assert params == {
        "model": "anthropic/claude-opus-4-6",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


def test_anthropic_adapter_normalizes_minimal_to_low():
    params = _resolve_llm_params(
        "anthropic/claude-opus-4-7", reasoning_effort="minimal"
    )

    assert params["output_config"] == {"effort": "low"}


def test_anthropic_adapter_no_effort():
    params = _resolve_llm_params("anthropic/claude-opus-4-6")

    assert params == {"model": "anthropic/claude-opus-4-6"}


def test_anthropic_adapter_strict_rejects_invalid():
    with pytest.raises(UnsupportedEffortError):
        _resolve_llm_params(
            "anthropic/claude-opus-4-6", reasoning_effort="turbo", strict=True
        )


def test_anthropic_adapter_nonstrict_drops_invalid():
    params = _resolve_llm_params(
        "anthropic/claude-opus-4-6", reasoning_effort="turbo", strict=False
    )
    assert "thinking" not in params
    assert "output_config" not in params


# -- OpenAI adapter -----------------------------------------------------------


def test_openai_adapter_passes_reasoning_effort():
    params = _resolve_llm_params("openai/gpt-5", reasoning_effort="medium")

    assert params == {"model": "openai/gpt-5", "reasoning_effort": "medium"}


def test_openai_adapter_strict_rejects_max():
    with pytest.raises(UnsupportedEffortError):
        _resolve_llm_params("openai/gpt-5", reasoning_effort="max", strict=True)


# -- Bedrock adapter ----------------------------------------------------------


def test_bedrock_adapter_returns_model_only():
    params = _resolve_llm_params("bedrock/us.anthropic.claude-opus-4-7")
    assert params == {"model": "bedrock/us.anthropic.claude-opus-4-7"}


def test_bedrock_adapter_ignores_effort():
    params = _resolve_llm_params(
        "bedrock/us.anthropic.claude-opus-4-6-v1", reasoning_effort="high"
    )
    assert params == {"model": "bedrock/us.anthropic.claude-opus-4-6-v1"}


def test_bedrock_validation():
    assert is_valid_model_name("bedrock/us.anthropic.claude-opus-4-7") is True
    assert is_valid_model_name("bedrock/") is False


# -- HF Router adapter --------------------------------------------------------


def test_hf_adapter_builds_router_params(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")

    params = _resolve_llm_params(
        "moonshotai/Kimi-K2.6:novita", reasoning_effort="minimal"
    )

    assert params == {
        "model": "openai/moonshotai/Kimi-K2.6:novita",
        "api_base": "https://router.huggingface.co/v1",
        "api_key": "hf-test",
        "extra_body": {"reasoning_effort": "low"},
    }


def test_hf_adapter_adds_bill_to_header(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "hf-space-token")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    params = _resolve_llm_params("MiniMaxAI/MiniMax-M2.7")

    assert params["extra_headers"] == {"X-HF-Bill-To": "smolagents"}
    assert params["api_key"] == "hf-space-token"


def test_hf_adapter_strict_rejects_max():
    with pytest.raises(UnsupportedEffortError):
        _resolve_llm_params(
            "MiniMaxAI/MiniMax-M2.7", reasoning_effort="max", strict=True
        )


# -- Validation ---------------------------------------------------------------


def test_model_validation_accepts_free_form_hf_ids():
    assert is_valid_model_name("moonshotai/Kimi-K2.6:fastest") is True
    assert is_valid_model_name("huggingface/moonshotai/Kimi-K2.6:novita") is True


def test_model_validation_accepts_direct_provider_ids():
    assert is_valid_model_name("anthropic/claude-opus-4-7") is True
    assert is_valid_model_name("openai/gpt-5") is True


def test_model_validation_rejects_garbage():
    assert is_valid_model_name("") is False
    assert is_valid_model_name("no-slash") is False
    assert is_valid_model_name("anthropic/") is False
    assert is_valid_model_name("openai/") is False
    assert is_valid_model_name("huggingface/nope") is False
    assert is_valid_model_name("moonshotai/") is False


def test_cli_validation_matches_provider_validation():
    assert is_valid_model_id("openai/gpt-5") is True
    assert is_valid_model_id("moonshotai/Kimi-K2.6:fastest") is True
    assert is_valid_model_id("openai/") is False
    assert is_valid_model_id("anthropic/") is False


def test_unsupported_effort_reexport():
    """UnsupportedEffortError must be importable from llm_params (backward compat)."""
    from agent.core.llm_params import UnsupportedEffortError as FromLlm
    from agent.core.provider_adapters import UnsupportedEffortError as FromAdapters

    assert FromLlm is FromAdapters
