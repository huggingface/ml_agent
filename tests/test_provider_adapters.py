import pytest

import agent.core.provider_adapters as providers
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


# -- Gemini adapter -----------------------------------------------------------


def test_gemini_adapter_passes_reasoning_effort():
    params = _resolve_llm_params("gemini/gemini-2.5-pro", reasoning_effort="medium")
    assert params == {"model": "gemini/gemini-2.5-pro", "reasoning_effort": "medium"}


def test_gemini_adapter_normalizes_minimal():
    params = _resolve_llm_params("gemini/gemini-2.5-flash", reasoning_effort="minimal")
    assert params == {"model": "gemini/gemini-2.5-flash", "reasoning_effort": "low"}


def test_gemini_adapter_no_effort():
    params = _resolve_llm_params("gemini/gemini-2.5-pro")
    assert params == {"model": "gemini/gemini-2.5-pro"}


def test_gemini_adapter_strict_rejects_invalid():
    with pytest.raises(UnsupportedEffortError):
        _resolve_llm_params("gemini/gemini-2.5-pro", reasoning_effort="max", strict=True)
    with pytest.raises(UnsupportedEffortError):
        _resolve_llm_params("gemini/gemini-2.5-pro", reasoning_effort="xhigh", strict=True)


def test_gemini_validation():
    assert is_valid_model_name("gemini/gemini-2.5-pro") is True
    assert is_valid_model_name("gemini/") is False


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


# -- OpenAI-compatible adapters ------------------------------------------------


def test_ollama_adapter_builds_params(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    params = _resolve_llm_params("ollama/llama3.1")

    assert params == {
        "model": "openai/llama3.1",
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama",
    }


def test_ollama_adapter_normalizes_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")

    params = _resolve_llm_params("ollama/llama3.1")

    assert params["api_base"] == "http://localhost:11434/v1"


def test_ollama_adapter_strict_rejects_effort():
    with pytest.raises(UnsupportedEffortError):
        _resolve_llm_params("ollama/llama3.1", reasoning_effort="high", strict=True)


def test_lm_studio_adapter_uses_raw_model_name(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)

    params = _resolve_llm_params("lm_studio/google/gemma-3-12b")

    assert params == {
        "model": "lm_studio/google/gemma-3-12b",
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
    }


def test_vllm_adapter_uses_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("VLLM_API_KEY", "secret")

    params = _resolve_llm_params("vllm/Qwen3-32B")

    assert params == {
        "model": "openai/Qwen3-32B",
        "api_base": "http://127.0.0.1:8000/v1",
        "api_key": "secret",
    }


def test_openrouter_adapter_uses_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    params = _resolve_llm_params(
        "openrouter/anthropic/claude-sonnet-4.5", reasoning_effort="medium"
    )

    assert params == {
        "model": "openai/anthropic/claude-sonnet-4.5",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": "router-key",
        "reasoning_effort": "medium",
    }


def test_opencode_zen_adapter_uses_api_key(monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")

    params = _resolve_llm_params("opencode/kimi-k2.6")

    assert params == {
        "model": "openai/kimi-k2.6",
        "api_base": "https://opencode.ai/zen/v1",
        "api_key": "zen-key",
    }


def test_opencode_go_adapter_uses_api_key(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")

    params = _resolve_llm_params("opencode-go/kimi-k2.6")

    assert params == {
        "model": "openai/kimi-k2.6",
        "api_base": "https://opencode.ai/zen/go/v1",
        "api_key": "go-key",
    }


def test_openai_compat_requires_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_COMPAT_BASE_URL"):
        _resolve_llm_params("openai-compat/my-model")


def test_openai_compat_uses_required_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-key")

    params = _resolve_llm_params("openai-compat/my-model")

    assert params == {
        "model": "openai/my-model",
        "api_base": "http://localhost:8080/v1",
        "api_key": "compat-key",
    }


# -- Validation ---------------------------------------------------------------


def test_model_validation_accepts_free_form_hf_ids():
    assert is_valid_model_name("moonshotai/Kimi-K2.6:fastest") is True
    assert is_valid_model_name("huggingface/moonshotai/Kimi-K2.6:novita") is True


def test_model_validation_accepts_direct_provider_ids():
    assert is_valid_model_name("anthropic/claude-opus-4-7") is True
    assert is_valid_model_name("openai/gpt-5") is True
    assert is_valid_model_name("bedrock/us.anthropic.claude-opus-4-7") is True
    assert is_valid_model_name("gemini/gemini-2.5-pro") is True
    assert is_valid_model_name("ollama/llama3.1") is True
    assert is_valid_model_name("lm_studio/google/gemma-3-12b") is True
    assert is_valid_model_name("vllm/Qwen3-32B") is True
    assert is_valid_model_name("openrouter/anthropic/claude-sonnet-4.5") is True
    assert is_valid_model_name("opencode/kimi-k2.6") is True
    assert is_valid_model_name("opencode-go/kimi-k2.6") is True
    assert is_valid_model_name("openai-compat/my-model") is True


def test_model_validation_rejects_garbage():
    assert is_valid_model_name("") is False
    assert is_valid_model_name("no-slash") is False
    assert is_valid_model_name("anthropic/") is False
    assert is_valid_model_name("openai/") is False
    assert is_valid_model_name("gemini/") is False
    assert is_valid_model_name("ollama/") is False
    assert is_valid_model_name("lm_studio/") is False
    assert is_valid_model_name("vllm/") is False
    assert is_valid_model_name("openrouter/") is False
    assert is_valid_model_name("opencode/") is False
    assert is_valid_model_name("opencode-go/") is False
    assert is_valid_model_name("openai-compat/") is False
    assert is_valid_model_name("huggingface/nope") is False
    assert is_valid_model_name("moonshotai/") is False


def test_hf_validation_excludes_new_provider_prefixes():
    hf = providers.resolve_adapter("openrouter/google/gemini-2.5-pro")

    assert hf is not None
    assert hf.provider_id == "openrouter"


def test_model_catalog_hides_local_providers_when_unreachable(monkeypatch):
    monkeypatch.setattr(providers.OllamaAdapter, "available_models", lambda self: ())
    monkeypatch.setattr(providers.LmStudioAdapter, "available_models", lambda self: ())
    monkeypatch.setattr(providers.VllmAdapter, "available_models", lambda self: ())

    catalog = providers.build_model_catalog("anthropic/claude-opus-4-6")
    provider_ids = {p["id"] for p in catalog["providers"]}

    assert "ollama" not in provider_ids
    assert "lm_studio" not in provider_ids
    assert "vllm" not in provider_ids


def test_model_catalog_includes_local_providers_when_discovered(monkeypatch):
    dynamic = (
        providers.SuggestedModel(
            id="ollama/llama3.1",
            label="llama3.1",
            description="Ollama",
            provider="ollama",
            provider_label="Ollama",
            avatar_url="avatar",
            source="dynamic",
        ),
    )
    monkeypatch.setattr(
        providers.OllamaAdapter, "available_models", lambda self: dynamic
    )

    catalog = providers.build_model_catalog("ollama/llama3.1")

    assert any(m["id"] == "ollama/llama3.1" for m in catalog["available"])
    assert any(p["id"] == "ollama" for p in catalog["providers"])


def test_model_catalog_openai_compat_visibility_depends_on_env(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    hidden = providers.build_model_catalog("anthropic/claude-opus-4-6")
    assert not any(p["id"] == "openai_compat" for p in hidden["providers"])

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8080/v1")
    shown = providers.build_model_catalog("anthropic/claude-opus-4-6")
    assert any(p["id"] == "openai_compat" for p in shown["providers"])


def test_model_catalog_includes_current_info_for_custom_model(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8080/v1")

    catalog = providers.build_model_catalog("openai-compat/my-model")

    assert catalog["currentInfo"] is not None
    assert catalog["currentInfo"]["id"] == "openai-compat/my-model"
    assert catalog["currentInfo"]["source"] == "custom"


def test_cli_validation_matches_provider_validation():
    assert is_valid_model_id("openai/gpt-5") is True
    assert is_valid_model_id("moonshotai/Kimi-K2.6:fastest") is True
    assert is_valid_model_id("ollama/llama3.1") is True
    assert is_valid_model_id("openrouter/anthropic/claude-sonnet-4.5") is True
    assert is_valid_model_id("openai-compat/my-model") is True
    assert is_valid_model_id("openai/") is False
    assert is_valid_model_id("anthropic/") is False


def test_resolve_raises_on_no_adapter(monkeypatch):
    from agent.core import llm_params

    monkeypatch.setattr(llm_params, "resolve_adapter", lambda _: None)
    with pytest.raises(ValueError, match="No provider adapter"):
        _resolve_llm_params("anything")


def test_unsupported_effort_reexport():
    """UnsupportedEffortError must be importable from llm_params (backward compat)."""
    from agent.core.llm_params import UnsupportedEffortError as FromLlm
    from agent.core.provider_adapters import UnsupportedEffortError as FromAdapters

    assert FromLlm is FromAdapters
