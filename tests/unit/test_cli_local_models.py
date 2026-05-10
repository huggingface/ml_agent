import pytest

from agent.core import model_switcher
from agent.core.local_models import is_local_model_id


def test_local_model_helper_accepts_supported_prefixes():
    assert is_local_model_id("ollama/llama3.1:8b")
    assert is_local_model_id("vllm/meta-llama/Llama-3.1-8B-Instruct")
    assert is_local_model_id("lm_studio/google/gemma-3-4b")
    assert is_local_model_id("llamacpp/unsloth/Qwen3.5-2B")


def test_model_switcher_accepts_supported_local_prefixes():
    assert model_switcher.is_valid_model_id("ollama/llama3.1:8b")
    assert model_switcher.is_valid_model_id("vllm/meta-llama/Llama-3.1-8B")
    assert model_switcher.is_valid_model_id("lm_studio/google/gemma-3-4b")
    assert model_switcher.is_valid_model_id("llamacpp/llama-3.1-8b")
    assert model_switcher.is_valid_model_id("openai-codex/gpt-5.5")


def test_model_switcher_rejects_empty_or_whitespace_local_ids():
    assert not model_switcher.is_valid_model_id("ollama/")
    assert not model_switcher.is_valid_model_id("vllm/")
    assert not model_switcher.is_valid_model_id("lm_studio/")
    assert not model_switcher.is_valid_model_id("llamacpp/")
    assert not model_switcher.is_valid_model_id("ollama/llama 3.1")


def test_openai_compat_prefix_is_not_supported():
    assert not model_switcher.is_valid_model_id("openai-compat/custom-model")


def test_local_models_skip_hf_router_catalog_output():
    class NoPrintConsole:
        def print(self, *args, **kwargs):
            raise AssertionError("local models should not print HF catalog info")

    assert model_switcher._print_hf_routing_info(
        "ollama/llama3.1:8b",
        NoPrintConsole(),
    )


def test_codex_models_skip_hf_router_catalog_output():
    class NoPrintConsole:
        def print(self, *args, **kwargs):
            raise AssertionError("Codex models should not print HF catalog info")

    assert model_switcher._print_hf_routing_info(
        "openai-codex/gpt-5.5",
        NoPrintConsole(),
    )


@pytest.mark.asyncio
async def test_probe_and_switch_local_model_uses_no_effort(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(model_switcher, "acompletion", fake_acompletion)

    class Config:
        model_name = "openai/gpt-5.5"
        reasoning_effort = "max"

    class Session:
        def __init__(self):
            self.model_id = None
            self.model_effective_effort = {}

        def update_model(self, model_id):
            self.model_id = model_id

    class Console:
        def print(self, *args, **kwargs):
            pass

    session = Session()
    await model_switcher.probe_and_switch_model(
        "ollama/llama3.1:8b",
        Config(),
        session,
        Console(),
        hf_token=None,
    )

    assert session.model_id == "ollama/llama3.1:8b"
    assert session.model_effective_effort["ollama/llama3.1:8b"] is None
    assert calls[0]["model"] == "openai/llama3.1:8b"
    assert "reasoning_effort" not in calls[0]
    assert "extra_body" not in calls[0]


@pytest.mark.asyncio
async def test_probe_and_switch_local_model_rejects_probe_errors(monkeypatch):
    async def failing_acompletion(**kwargs):
        raise ConnectionRefusedError("no server")

    monkeypatch.setattr(model_switcher, "acompletion", failing_acompletion)

    class Config:
        model_name = "openai/gpt-5.5"
        reasoning_effort = None

    class Session:
        def __init__(self):
            self.model_id = None
            self.model_effective_effort = {}

        def update_model(self, model_id):
            self.model_id = model_id

    class Console:
        def print(self, *args, **kwargs):
            pass

    config = Config()
    session = Session()
    await model_switcher.probe_and_switch_model(
        "ollama/llama3.1:8b",
        config,
        session,
        Console(),
        hf_token=None,
    )

    assert config.model_name == "openai/gpt-5.5"
    assert session.model_id is None
    assert "ollama/llama3.1:8b" not in session.model_effective_effort


@pytest.mark.asyncio
async def test_probe_and_switch_codex_model_validates_oauth_without_litellm(
    monkeypatch,
):
    calls = []

    async def fake_resolve(model_id, *, user_id, reasoning_effort):
        calls.append((model_id, user_id, reasoning_effort))
        return {"_ml_intern_provider": "openai-codex"}

    async def fail_acompletion(**kwargs):
        raise AssertionError("Codex switch should not use LiteLLM")

    monkeypatch.setattr(model_switcher, "resolve_codex_llm_params", fake_resolve)
    monkeypatch.setattr(model_switcher, "acompletion", fail_acompletion)

    class Config:
        model_name = "openai/gpt-5.5"
        reasoning_effort = "high"

    class Session:
        user_id = "user-1"

        def __init__(self):
            self.model_id = None
            self.model_effective_effort = {}

        def update_model(self, model_id):
            self.model_id = model_id

    class Console:
        def print(self, *args, **kwargs):
            pass

    session = Session()
    await model_switcher.probe_and_switch_model(
        "openai-codex/gpt-5.5",
        Config(),
        session,
        Console(),
        hf_token=None,
    )

    assert calls == [("openai-codex/gpt-5.5", "user-1", "high")]
    assert session.model_id == "openai-codex/gpt-5.5"
    assert "openai-codex/gpt-5.5" not in session.model_effective_effort
