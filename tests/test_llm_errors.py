import asyncio
from types import SimpleNamespace

from rich.console import Console

import agent.core.model_switcher as model_switcher
from agent.core.effort_probe import ProbeInconclusive
from agent.core.llm_errors import (
    classify_llm_error,
    friendly_llm_error_message,
    health_error_type,
    render_llm_error_message,
)


def test_auth_errors_get_clean_message() -> None:
    error = Exception("401 unauthorized: invalid api key")

    assert classify_llm_error(error) == "auth"
    assert "Authentication failed" in friendly_llm_error_message(error)


def test_missing_api_key_header_gets_clean_message() -> None:
    error = Exception("authentication_error: x-api-key header is required")

    assert classify_llm_error(error) == "auth"
    assert render_llm_error_message(error).startswith("Authentication failed")


def test_openai_missing_api_key_gets_clean_message() -> None:
    error = Exception(
        "You didn't provide an API key. You need to provide your API key in an Authorization header."
    )

    assert classify_llm_error(error) == "auth"
    assert render_llm_error_message(error).startswith("Authentication failed")


def test_anthropic_low_credit_error_gets_clean_message() -> None:
    error = Exception(
        "Your credit balance is too low to access the Anthropic API. "
        "Please go to Plans & Billing to upgrade or purchase credits."
    )

    assert classify_llm_error(error) == "credits"
    assert render_llm_error_message(error).startswith(
        "Insufficient API credits or quota"
    )


def test_model_not_found_error_gets_clean_message() -> None:
    error = Exception("model_not_found: requested model does not exist")

    assert classify_llm_error(error) == "model"
    assert render_llm_error_message(error).startswith("Model not found")


def test_unknown_errors_fall_back_to_plain_exception_text() -> None:
    error = RuntimeError("boom")

    assert classify_llm_error(error) == "unknown"
    assert render_llm_error_message(error) == "boom"


def test_health_error_type_keeps_public_categories_stable() -> None:
    assert health_error_type(Exception("invalid api key")) == "auth"
    assert health_error_type(Exception("credit balance is too low")) == "credits"
    assert health_error_type(Exception("rate limit exceeded")) == "rate_limit"
    assert health_error_type(Exception("model_not_found")) == "unknown"


def test_model_switcher_shows_clean_hard_failure(monkeypatch) -> None:
    async def fake_probe_effort(*args, **kwargs):
        raise Exception(
            "Your credit balance is too low to access the Anthropic API. "
            "Please go to Plans & Billing to upgrade or purchase credits."
        )

    monkeypatch.setattr(model_switcher, "probe_effort", fake_probe_effort)
    console = Console(record=True, width=120)
    config = SimpleNamespace(
        reasoning_effort="high",
        model_name="anthropic/claude-opus-4-6",
    )

    asyncio.run(
        model_switcher.probe_and_switch_model(
            "anthropic/claude-opus-4-7",
            config,
            None,
            console,
            None,
        )
    )

    output = console.export_text()
    assert "Insufficient API credits or quota" in output
    assert "credit balance is too low" not in output.lower()


def test_model_switcher_shows_clean_inconclusive_warning(monkeypatch) -> None:
    async def fake_probe_effort(*args, **kwargs):
        raise ProbeInconclusive("timeout talking to provider")

    monkeypatch.setattr(model_switcher, "probe_effort", fake_probe_effort)
    console = Console(record=True, width=120)
    config = SimpleNamespace(
        reasoning_effort="high",
        model_name="anthropic/claude-opus-4-6",
    )

    asyncio.run(
        model_switcher.probe_and_switch_model(
            "anthropic/claude-opus-4-7",
            config,
            None,
            console,
            None,
        )
    )

    output = console.export_text()
    assert "The model provider is unavailable or timed out" in output
    assert "timeout talking to provider" not in output.lower()
