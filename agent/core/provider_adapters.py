"""Provider adapters for runtime params and model metadata."""

import os
from dataclasses import dataclass
from typing import Any, ClassVar


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


@dataclass(frozen=True)
class ProviderAdapter:
    provider_id: str
    provider_label: str
    prefixes: tuple[str, ...] = ()
    supports_custom_model: bool = False
    custom_model_hint: str | None = None

    def matches(self, model_name: str) -> bool:
        return bool(self.prefixes) and model_name.startswith(self.prefixes)

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return ()

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
        if any(model.id == model_name for model in self.suggested_models()):
            return True
        return self.supports_custom_model and self.matches(model_name)


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
                avatar_url="https://huggingface.co/api/avatars/Anthropic",
                recommended=True,
            ),
            SuggestedModel(
                id="anthropic/claude-opus-4-6",
                label="Claude Opus 4.6",
                description="Anthropic",
                provider="anthropic",
                provider_label="Anthropic",
                avatar_url="https://huggingface.co/api/avatars/Anthropic",
            ),
        )

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
class HfRouterAdapter(ProviderAdapter):
    """HuggingFace router — OpenAI-compat endpoint with HF token chain."""

    _EFFORTS: ClassVar[frozenset[str]] = frozenset({"low", "medium", "high"})

    def _is_hf_model_name(self, model_name: str) -> bool:
        if model_name.startswith(("anthropic/", "openai/")):
            return False
        bare = model_name.removeprefix("huggingface/").split(":", 1)[0]
        parts = bare.split("/")
        return len(parts) >= 2 and all(parts)

    def matches(self, model_name: str) -> bool:
        return self._is_hf_model_name(model_name)

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return (
            SuggestedModel(
                id="MiniMaxAI/MiniMax-M2.7",
                label="MiniMax M2.7",
                description="HF Router",
                provider="huggingface",
                provider_label="Hugging Face Router",
                avatar_url="https://huggingface.co/api/avatars/MiniMaxAI",
                recommended=True,
            ),
            SuggestedModel(
                id="moonshotai/Kimi-K2.6",
                label="Kimi K2.6",
                description="HF Router",
                provider="huggingface",
                provider_label="Hugging Face Router",
                avatar_url="https://huggingface.co/api/avatars/moonshotai",
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

    def allows_model_name(self, model_name: str) -> bool:
        return self._is_hf_model_name(model_name)

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
    OpenAIAdapter(
        provider_id="openai",
        provider_label="OpenAI",
        supports_custom_model=True,
        custom_model_hint="Use openai/<model>, for example openai/gpt-5",
    ),
    HfRouterAdapter(
        provider_id="huggingface",
        provider_label="Hugging Face Router",
        supports_custom_model=True,
        custom_model_hint=(
            "Paste any Hugging Face model id, optionally with "
            ":fastest, :cheapest, :preferred, or :<provider>"
        ),
    ),
)


def resolve_adapter(model_name: str) -> ProviderAdapter | None:
    for adapter in ADAPTERS:
        if adapter.matches(model_name):
            return adapter
    return None


def is_valid_model_name(model_name: str) -> bool:
    adapter = resolve_adapter(model_name)
    if not adapter:
        return False
    return adapter.allows_model_name(model_name)


def get_available_models() -> list[dict[str, Any]]:
    available: list[dict[str, Any]] = []
    for adapter in ADAPTERS:
        for model in adapter.suggested_models():
            available.append(
                {
                    "id": model.id,
                    "label": model.label,
                    "description": model.description,
                    "provider": model.provider,
                    "providerLabel": model.provider_label,
                    "avatarUrl": model.avatar_url,
                    "recommended": model.recommended,
                }
            )
    return available


def is_suggested_model_name(model_name: str) -> bool:
    return any(model["id"] == model_name for model in get_available_models())


def build_model_catalog(current_model: str) -> dict[str, Any]:
    return {
        "current": current_model,
        "available": get_available_models(),
    }
