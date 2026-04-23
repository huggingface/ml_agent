import pytest

from agent.core.llm_params import _resolve_llm_params
from agent.core.provider_adapters import (
    UnsupportedEffortError,
    build_model_catalog,
    is_suggested_model_name,
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


# -- Catalog & validation -----------------------------------------------------


def test_model_catalog_comes_from_adapters():
    catalog = build_model_catalog("anthropic/claude-opus-4-6")

    assert catalog["current"] == "anthropic/claude-opus-4-6"
    assert any(model["provider"] == "anthropic" for model in catalog["available"])
    assert any(model["provider"] == "huggingface" for model in catalog["available"])
    assert set(catalog) == {"current", "available"}


def test_suggested_model_validation_is_strict():
    assert is_suggested_model_name("anthropic/claude-opus-4-6") is True
    assert is_suggested_model_name("moonshotai/Kimi-K2.6") is True
    assert is_suggested_model_name("moonshotai/Kimi-K2.6:fastest") is False


def test_model_validation_accepts_free_form_hf_ids():
    assert is_valid_model_name("moonshotai/Kimi-K2.6:fastest") is True
    assert is_valid_model_name("huggingface/moonshotai/Kimi-K2.6:novita") is True


def test_model_validation_rejects_garbage():
    assert is_valid_model_name("") is False
    assert is_valid_model_name("no-slash") is False


def test_unsupported_effort_reexport():
    """UnsupportedEffortError must be importable from llm_params (backward compat)."""
    from agent.core.llm_params import UnsupportedEffortError as FromLlm
    from agent.core.provider_adapters import UnsupportedEffortError as FromAdapters

    assert FromLlm is FromAdapters
