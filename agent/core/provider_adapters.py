"""Provider adapters for runtime params and model catalog metadata."""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.error import URLError
from urllib.request import urlopen

_DISCOVERY_TIMEOUT_SECONDS = 2.0
_DISCOVERY_CACHE_TTL_SECONDS = 30.0
_discovery_cache: dict[str, tuple[float, tuple["SuggestedModel", ...]]] = {}


class UnsupportedEffortError(ValueError):
    """The requested effort isn't valid for this provider's API surface.

    Raised synchronously before any network call so the probe cascade can
    skip levels the provider can't accept (e.g. ``max`` on HF router).
    """


@dataclass(frozen=True)
class SuggestedModel:
    id: str
    label: str
    description: str
    provider: str
    provider_label: str
    avatar_url: str
    recommended: bool = False
    source: str = "static"


def _has_model_suffix(model_name: str, prefix: str) -> bool:
    if not model_name.startswith(prefix):
        return False
    tail = model_name[len(prefix) :].split(":", 1)[0]
    return bool(tail) and all(tail.split("/"))


def _normalize_openai_api_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _provider_avatar_url(provider_id: str) -> str:
    avatars = {
        "anthropic": "https://huggingface.co/api/avatars/Anthropic",
        "openai": "https://openai.com/favicon.ico",
        "ollama": "https://ollama.com/public/ollama.png",
        "lm_studio": "https://avatars.githubusercontent.com/u/16906759?s=200&v=4",
        "vllm": "https://avatars.githubusercontent.com/u/132129714?s=200&v=4",
        "openrouter": "https://openrouter.ai/favicon.ico",
        "opencode_zen": "https://huggingface.co/api/avatars/opencode-ai",
        "opencode_go": "https://huggingface.co/api/avatars/opencode-ai",
        "openai_compat": "https://openai.com/favicon.ico",
        "huggingface": "https://huggingface.co/api/avatars/huggingface",
    }
    return avatars.get(provider_id, "https://huggingface.co/api/avatars/huggingface")


def _all_adapter_prefixes() -> tuple[str, ...]:
    prefixes: list[str] = []
    for adapter in ADAPTERS:
        prefixes.extend(adapter.prefixes)
    return tuple(dict.fromkeys(prefixes))


def _is_hf_model_name(model_name: str) -> bool:
    if model_name.startswith(_all_adapter_prefixes()):
        return False
    bare = model_name.removeprefix("huggingface/").split(":", 1)[0]
    parts = bare.split("/")
    return len(parts) >= 2 and all(parts)


def _discover_models(
    *,
    provider_id: str,
    provider_label: str,
    prefix: str,
    api_base: str,
) -> tuple[SuggestedModel, ...]:
    now = time.monotonic()
    cached = _discovery_cache.get(api_base)
    if cached and cached[0] > now:
        return cached[1]

    models: list[SuggestedModel] = []
    try:
        with urlopen(
            f"{api_base}/models", timeout=_DISCOVERY_TIMEOUT_SECONDS
        ) as response:
            payload = json.load(response)
    except (OSError, URLError, TimeoutError, ValueError):
        payload = {"data": []}

    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        models.append(
            SuggestedModel(
                id=f"{prefix}{model_id}",
                label=model_id,
                description=provider_label,
                provider=provider_id,
                provider_label=provider_label,
                avatar_url=_provider_avatar_url(provider_id),
                source="dynamic",
            )
        )

    resolved = tuple(sorted(models, key=lambda m: m.label.lower()))
    _discovery_cache[api_base] = (now + _DISCOVERY_CACHE_TTL_SECONDS, resolved)
    return resolved


@dataclass(frozen=True)
class ProviderAdapter:
    provider_id: str
    provider_label: str
    prefixes: tuple[str, ...] = ()
    supports_custom_model: bool = False
    custom_model_hint: str | None = None
    custom_model_mode: str | None = None

    def matches(self, model_name: str) -> bool:
        return bool(self.prefixes) and model_name.startswith(self.prefixes)

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return ()

    def available_models(self) -> tuple[SuggestedModel, ...]:
        return self.suggested_models()

    def should_show(self) -> bool:
        return True

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        raise NotImplementedError

    def allows_model_name(self, model_name: str) -> bool:
        return self.matches(model_name)

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.provider_id,
            "label": self.provider_label,
            "avatarUrl": _provider_avatar_url(self.provider_id),
            "supportsCustomModel": self.supports_custom_model,
            "customModelHint": self.custom_model_hint,
            "customModelMode": self.custom_model_mode,
            "prefix": self.prefixes[0] if self.prefixes else "",
        }


@dataclass(frozen=True)
class AnthropicAdapter(ProviderAdapter):
    """Anthropic models via native API (thinking + output_config.effort)."""

    prefixes: tuple[str, ...] = ("anthropic/",)
    _EFFORTS: ClassVar[frozenset[str]] = frozenset(
        {"low", "medium", "high", "xhigh", "max"}
    )

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return (
            SuggestedModel(
                id="anthropic/claude-opus-4-7",
                label="Claude Opus 4.7",
                description="Anthropic",
                provider="anthropic",
                provider_label="Anthropic",
                avatar_url=_provider_avatar_url("anthropic"),
                recommended=True,
            ),
            SuggestedModel(
                id="anthropic/claude-opus-4-6",
                label="Claude Opus 4.6",
                description="Anthropic",
                provider="anthropic",
                provider_label="Anthropic",
                avatar_url=_provider_avatar_url("anthropic"),
            ),
        )

    def allows_model_name(self, model_name: str) -> bool:
        return _has_model_suffix(model_name, "anthropic/")

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        params: dict[str, Any] = {"model": model_name}
        if reasoning_effort:
            level = "low" if reasoning_effort == "minimal" else reasoning_effort
            if level not in self._EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"Anthropic doesn't accept effort={level!r}"
                    )
            else:
                params["thinking"] = {"type": "adaptive"}
                params["output_config"] = {"effort": level}
        return params


@dataclass(frozen=True)
class OpenAIAdapter(ProviderAdapter):
    """OpenAI models via native API (reasoning_effort top-level kwarg)."""

    prefixes: tuple[str, ...] = ("openai/",)
    _EFFORTS: ClassVar[frozenset[str]] = frozenset({"minimal", "low", "medium", "high"})

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return (
            SuggestedModel(
                id="openai/gpt-5",
                label="GPT-5",
                description="OpenAI",
                provider="openai",
                provider_label="OpenAI",
                avatar_url=_provider_avatar_url("openai"),
                recommended=True,
            ),
        )

    def allows_model_name(self, model_name: str) -> bool:
        return _has_model_suffix(model_name, "openai/")

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        params: dict[str, Any] = {"model": model_name}
        if reasoning_effort:
            if reasoning_effort not in self._EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"OpenAI doesn't accept effort={reasoning_effort!r}"
                    )
            else:
                params["reasoning_effort"] = reasoning_effort
        return params


@dataclass(frozen=True)
class OpenAICompatAdapter(ProviderAdapter):
    api_base_url: str = ""
    api_key_env: str = ""
    default_api_key: str = ""
    supports_reasoning_effort: bool = True
    use_raw_model_name: bool = False

    def resolved_api_base(self) -> str:
        return _normalize_openai_api_base(self.api_base_url)

    def resolved_api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env, self.default_api_key)
        return self.default_api_key or None

    def suggested_model_defs(self) -> tuple[tuple[str, str, bool], ...]:
        return ()

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        prefix = self.prefixes[0]
        return tuple(
            SuggestedModel(
                id=f"{prefix}{model_id}",
                label=label,
                description=self.provider_label,
                provider=self.provider_id,
                provider_label=self.provider_label,
                avatar_url=_provider_avatar_url(self.provider_id),
                recommended=recommended,
            )
            for model_id, label, recommended in self.suggested_model_defs()
        )

    def allows_model_name(self, model_name: str) -> bool:
        return bool(self.prefixes) and _has_model_suffix(model_name, self.prefixes[0])

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        del session_hf_token

        model_id = model_name.removeprefix(self.prefixes[0])
        params: dict[str, Any] = {
            "model": model_name if self.use_raw_model_name else f"openai/{model_id}",
            "api_base": self.resolved_api_base(),
            "api_key": self.resolved_api_key(),
        }

        if reasoning_effort:
            if not self.supports_reasoning_effort:
                if strict:
                    raise UnsupportedEffortError(
                        f"{self.provider_id} doesn't accept effort={reasoning_effort!r}"
                    )
            else:
                params["reasoning_effort"] = reasoning_effort

        return params


@dataclass(frozen=True)
class OllamaAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("ollama/",)
    api_key_env: str = "OLLAMA_API_KEY"
    default_api_key: str = "ollama"
    supports_reasoning_effort: bool = False

    def resolved_api_base(self) -> str:
        return _normalize_openai_api_base(
            os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
        )

    def available_models(self) -> tuple[SuggestedModel, ...]:
        return _discover_models(
            provider_id=self.provider_id,
            provider_label=self.provider_label,
            prefix=self.prefixes[0],
            api_base=self.resolved_api_base(),
        )

    def should_show(self) -> bool:
        return bool(self.available_models())


@dataclass(frozen=True)
class LmStudioAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("lm_studio/",)
    api_key_env: str = "LMSTUDIO_API_KEY"
    default_api_key: str = "lm-studio"
    supports_reasoning_effort: bool = False
    use_raw_model_name: bool = True

    def resolved_api_base(self) -> str:
        return _normalize_openai_api_base(
            os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        )

    def available_models(self) -> tuple[SuggestedModel, ...]:
        return _discover_models(
            provider_id=self.provider_id,
            provider_label=self.provider_label,
            prefix=self.prefixes[0],
            api_base=self.resolved_api_base(),
        )

    def should_show(self) -> bool:
        return bool(self.available_models())


@dataclass(frozen=True)
class VllmAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("vllm/",)
    api_key_env: str = "VLLM_API_KEY"
    default_api_key: str = "vllm"
    supports_reasoning_effort: bool = False

    def resolved_api_base(self) -> str:
        return _normalize_openai_api_base(
            os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        )

    def available_models(self) -> tuple[SuggestedModel, ...]:
        return _discover_models(
            provider_id=self.provider_id,
            provider_label=self.provider_label,
            prefix=self.prefixes[0],
            api_base=self.resolved_api_base(),
        )

    def should_show(self) -> bool:
        return bool(self.available_models())


@dataclass(frozen=True)
class OpenRouterAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("openrouter/",)
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    def suggested_model_defs(self) -> tuple[tuple[str, str, bool], ...]:
        return (("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5", True),)


@dataclass(frozen=True)
class OpenCodeZenAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("opencode/",)
    api_base_url: str = "https://opencode.ai/zen/v1"
    api_key_env: str = "OPENCODE_ZEN_API_KEY"

    def suggested_model_defs(self) -> tuple[tuple[str, str, bool], ...]:
        return (("kimi-k2.6", "Kimi K2.6", True),)


@dataclass(frozen=True)
class OpenCodeGoAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("opencode-go/",)
    api_base_url: str = "https://opencode.ai/zen/go/v1"
    api_key_env: str = "OPENCODE_GO_API_KEY"

    def suggested_model_defs(self) -> tuple[tuple[str, str, bool], ...]:
        return (("kimi-k2.6", "Kimi K2.6", True),)


@dataclass(frozen=True)
class GenericOpenAICompatAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("openai-compat/",)
    api_key_env: str = "OPENAI_COMPAT_API_KEY"
    supports_custom_model: bool = True
    custom_model_hint: str | None = (
        "Use openai-compat/<model-id>. Configure OPENAI_COMPAT_BASE_URL on server."
    )
    custom_model_mode: str | None = "suffix"

    def resolved_api_base(self) -> str:
        api_base = os.environ.get("OPENAI_COMPAT_BASE_URL", "")
        if not api_base:
            raise ValueError("OPENAI_COMPAT_BASE_URL is required for openai-compat/")
        return _normalize_openai_api_base(api_base)

    def should_show(self) -> bool:
        return bool(os.environ.get("OPENAI_COMPAT_BASE_URL"))


@dataclass(frozen=True)
class HfRouterAdapter(ProviderAdapter):
    """HuggingFace router — OpenAI-compat endpoint with HF token chain."""

    _EFFORTS: ClassVar[frozenset[str]] = frozenset({"low", "medium", "high"})
    supports_custom_model: bool = True
    custom_model_hint: str | None = (
        "Paste any Hugging Face model id, optionally with :fastest/:cheapest/:preferred"
    )
    custom_model_mode: str | None = "raw"

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return (
            SuggestedModel(
                id="moonshotai/Kimi-K2.6",
                label="Kimi K2.6",
                description="HF Router",
                provider="huggingface",
                provider_label="Hugging Face Router",
                avatar_url="https://huggingface.co/api/avatars/moonshotai",
                recommended=True,
            ),
            SuggestedModel(
                id="MiniMaxAI/MiniMax-M2.7",
                label="MiniMax M2.7",
                description="HF Router",
                provider="huggingface",
                provider_label="Hugging Face Router",
                avatar_url="https://huggingface.co/api/avatars/MiniMaxAI",
            ),
            SuggestedModel(
                id="zai-org/GLM-5.1",
                label="GLM 5.1",
                description="HF Router",
                provider="huggingface",
                provider_label="Hugging Face Router",
                avatar_url="https://huggingface.co/api/avatars/zai-org",
            ),
        )

    def matches(self, model_name: str) -> bool:
        return _is_hf_model_name(model_name)

    def allows_model_name(self, model_name: str) -> bool:
        return _is_hf_model_name(model_name)

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        hf_model = model_name.removeprefix("huggingface/")
        inference_token = os.environ.get("INFERENCE_TOKEN")
        api_key = inference_token or session_hf_token or os.environ.get("HF_TOKEN")

        params: dict[str, Any] = {
            "model": f"openai/{hf_model}",
            "api_base": "https://router.huggingface.co/v1",
            "api_key": api_key,
        }

        if inference_token:
            bill_to = os.environ.get("HF_BILL_TO", "smolagents")
            params["extra_headers"] = {"X-HF-Bill-To": bill_to}

        if reasoning_effort:
            hf_level = "low" if reasoning_effort == "minimal" else reasoning_effort
            if hf_level not in self._EFFORTS:
                if strict:
                    raise UnsupportedEffortError(
                        f"HF router doesn't accept effort={hf_level!r}"
                    )
            else:
                params["extra_body"] = {"reasoning_effort": hf_level}

        return params


ADAPTERS: tuple[ProviderAdapter, ...] = (
    AnthropicAdapter(provider_id="anthropic", provider_label="Anthropic"),
    OpenAIAdapter(provider_id="openai", provider_label="OpenAI"),
    OllamaAdapter(provider_id="ollama", provider_label="Ollama"),
    LmStudioAdapter(provider_id="lm_studio", provider_label="LM Studio"),
    VllmAdapter(provider_id="vllm", provider_label="vLLM"),
    OpenRouterAdapter(provider_id="openrouter", provider_label="OpenRouter"),
    OpenCodeZenAdapter(provider_id="opencode_zen", provider_label="OpenCode Zen"),
    OpenCodeGoAdapter(provider_id="opencode_go", provider_label="OpenCode Go"),
    GenericOpenAICompatAdapter(
        provider_id="openai_compat",
        provider_label="OpenAI-Compatible",
    ),
    HfRouterAdapter(provider_id="huggingface", provider_label="Hugging Face Router"),
)


def resolve_adapter(model_name: str) -> ProviderAdapter | None:
    for adapter in ADAPTERS:
        if adapter.matches(model_name):
            return adapter
    return None


def is_valid_model_name(model_name: str) -> bool:
    adapter = resolve_adapter(model_name)
    return adapter is not None and adapter.allows_model_name(model_name)


def _serialized_model(model: SuggestedModel) -> dict[str, Any]:
    return {
        "id": model.id,
        "label": model.label,
        "description": model.description,
        "provider": model.provider,
        "providerLabel": model.provider_label,
        "avatarUrl": model.avatar_url,
        "recommended": model.recommended,
        "source": model.source,
    }


def get_available_models() -> list[dict[str, Any]]:
    available: list[dict[str, Any]] = []
    for adapter in ADAPTERS:
        if not adapter.should_show():
            continue
        for model in adapter.available_models():
            available.append(_serialized_model(model))
    return available


def get_provider_summaries() -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    for adapter in ADAPTERS:
        if not adapter.should_show():
            continue
        providers.append(adapter.to_summary())
    return providers


def find_model_option(model_name: str) -> dict[str, Any] | None:
    for model in get_available_models():
        if model["id"] == model_name:
            return model

    adapter = resolve_adapter(model_name)
    if not adapter or not adapter.allows_model_name(model_name):
        return None

    label = model_name
    if adapter.provider_id == "huggingface":
        label = model_name.removeprefix("huggingface/")
    elif adapter.prefixes:
        label = model_name.removeprefix(adapter.prefixes[0])

    return {
        "id": model_name,
        "label": label,
        "description": f"Custom {adapter.provider_label} model",
        "provider": adapter.provider_id,
        "providerLabel": adapter.provider_label,
        "avatarUrl": _provider_avatar_url(adapter.provider_id),
        "recommended": False,
        "source": "custom",
    }


def build_model_catalog(current_model: str) -> dict[str, Any]:
    return {
        "current": current_model,
        "available": get_available_models(),
        "providers": get_provider_summaries(),
        "currentInfo": find_model_option(current_model),
    }
