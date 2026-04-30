from agent.core.hf_tokens import resolve_hf_request_token
from agent.core.llm_params import (
    UnsupportedEffortError,
    _resolve_hf_router_token,
    _resolve_llm_params,
)


def test_openai_xhigh_effort_is_forwarded():
    params = _resolve_llm_params(
        "openai/gpt-5.5",
        reasoning_effort="xhigh",
        strict=True,
    )

    assert params["model"] == "openai/gpt-5.5"
    assert params["reasoning_effort"] == "xhigh"


def test_openai_max_effort_is_still_rejected():
    try:
        _resolve_llm_params(
            "openai/gpt-5.4",
            reasoning_effort="max",
            strict=True,
        )
    except UnsupportedEffortError as exc:
        assert "OpenAI doesn't accept effort='max'" in str(exc)
    else:
        raise AssertionError("Expected UnsupportedEffortError for max effort")


def test_openrouter_params_use_openrouter_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", " openrouter-token ")
    monkeypatch.setenv("OR_SITE_URL", " https://example.com/ml-intern ")
    monkeypatch.setenv("OR_APP_NAME", " ML Intern ")

    params = _resolve_llm_params(
        "openrouter/anthropic/claude-sonnet-4-5",
        reasoning_effort="high",
        strict=True,
    )

    assert params["model"] == "openrouter/anthropic/claude-sonnet-4-5"
    assert params["api_key"] == "openrouter-token"
    assert params["reasoning_effort"] == "high"
    assert params["extra_headers"] == {
        "HTTP-Referer": "https://example.com/ml-intern",
        "X-Title": "ML Intern",
    }


def test_openrouter_api_base_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_BASE", " https://openrouter.example/v1 ")

    params = _resolve_llm_params("openrouter/openai/gpt-5.5")

    assert params["api_base"] == "https://openrouter.example/v1"


def test_openrouter_rejects_max_effort_in_strict_mode():
    try:
        _resolve_llm_params(
            "openrouter/openai/gpt-5.5",
            reasoning_effort="max",
            strict=True,
        )
    except UnsupportedEffortError as exc:
        assert "OpenRouter doesn't accept effort='max'" in str(exc)
    else:
        raise AssertionError("Expected UnsupportedEffortError for max effort")


def test_hf_router_token_prefers_inference_token(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", " inference-token ")
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert _resolve_hf_router_token("session-token") == "inference-token"


def test_hf_router_token_prefers_session_over_hf_cache(monkeypatch):
    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert _resolve_hf_router_token(" session-token ") == "session-token"


def test_hf_router_token_uses_hf_token_env_via_huggingface_hub(monkeypatch):
    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", " hf-token ")

    assert _resolve_hf_router_token(None) == "hf-token"


def test_hf_router_token_uses_huggingface_hub_cache(monkeypatch):
    import huggingface_hub

    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    assert _resolve_hf_router_token(None) == "cached-token"


def test_hf_router_token_swallows_huggingface_hub_errors(monkeypatch):
    import huggingface_hub

    def fail():
        raise RuntimeError("cache unavailable")

    monkeypatch.delenv("INFERENCE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", fail)

    assert _resolve_hf_router_token(None) is None


def test_hf_router_params_set_bill_to_only_for_inference_token(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "inference-token")
    monkeypatch.setenv("HF_BILL_TO", "test-org")

    params = _resolve_llm_params("moonshotai/Kimi-K2.6")

    assert params["api_key"] == "inference-token"
    assert params["extra_headers"] == {"X-HF-Bill-To": "test-org"}


def test_hf_request_token_keeps_browser_user_precedence(monkeypatch):
    class Request:
        headers = {"Authorization": "Bearer browser-token"}
        cookies = {"hf_access_token": "cookie-token"}

    monkeypatch.setenv("HF_TOKEN", "server-token")

    assert resolve_hf_request_token(Request()) == "browser-token"


def test_hf_request_token_does_not_use_cached_login(monkeypatch):
    import huggingface_hub

    class Request:
        headers = {}
        cookies = {}

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "cached-token")

    assert resolve_hf_request_token(Request()) is None
