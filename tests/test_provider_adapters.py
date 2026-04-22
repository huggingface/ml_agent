from agent.core.llm_params import _resolve_llm_params
from agent.core.provider_adapters import build_model_catalog, is_valid_model_name


def test_native_adapter_keeps_model_name():
    params = _resolve_llm_params("anthropic/claude-opus-4-6", reasoning_effort="high")

    assert params == {
        "model": "anthropic/claude-opus-4-6",
        "reasoning_effort": "high",
    }


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


def test_opencode_go_adapter_uses_api_key(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-test-key")

    params = _resolve_llm_params("opencode-go/kimi-k2.6")

    assert params == {
        "model": "openai/kimi-k2.6",
        "api_base": "https://opencode.ai/zen/go/v1",
        "api_key": "go-test-key",
    }


def test_model_catalog_comes_from_adapters():
    catalog = build_model_catalog("anthropic/claude-opus-4-6")

    assert catalog["current"] == "anthropic/claude-opus-4-6"
    assert any(model["provider"] == "anthropic" for model in catalog["available"])
    assert any(model["provider"] == "huggingface" for model in catalog["available"])
    assert any(model["provider"] == "opencode_go" for model in catalog["available"])
    assert any(provider["id"] == "opencode_go" for provider in catalog["providers"])
    assert any(provider["id"] == "huggingface" for provider in catalog["providers"])


def test_model_validation_accepts_free_form_hf_ids():
    assert is_valid_model_name("moonshotai/Kimi-K2.6:fastest") is True
    assert is_valid_model_name("huggingface/moonshotai/Kimi-K2.6:novita") is True


def test_model_validation_accepts_free_form_opencode_go_ids():
    assert is_valid_model_name("opencode-go/glm-5.1") is True
