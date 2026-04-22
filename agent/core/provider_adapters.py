"""Provider adapters for runtime params and model catalog metadata."""

import os
from dataclasses import dataclass
from typing import Any


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
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        raise NotImplementedError

    def allows_model_name(self, model_name: str) -> bool:
        if any(model.id == model_name for model in self.suggested_models()):
            return True
        return self.supports_custom_model and self.matches(model_name)

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.provider_id,
            "label": self.provider_label,
            "supportsCustomModel": self.supports_custom_model,
            "customModelHint": self.custom_model_hint,
        }


@dataclass(frozen=True)
class NativeAdapter(ProviderAdapter):
    prefixes: tuple[str, ...] = ("anthropic/", "openai/")

    def matches(self, model_name: str) -> bool:
        return model_name.startswith(self.prefixes)

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return (
            SuggestedModel(
                id="anthropic/claude-opus-4-6",
                label="Claude Opus 4.6",
                description="Anthropic",
                provider="anthropic",
                provider_label="Anthropic",
                avatar_url="https://huggingface.co/api/avatars/Anthropic",
                recommended=True,
            ),
        )

    def build_params(
        self,
        model_name: str,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"model": model_name}
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        return params


@dataclass(frozen=True)
class HfRouterAdapter(ProviderAdapter):
    allowed_efforts: tuple[str, ...] = ("low", "medium", "high")

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
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
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
            if hf_level in self.allowed_efforts:
                params["extra_body"] = {"reasoning_effort": hf_level}

        return params


@dataclass(frozen=True)
class OpenCodeGoAdapter(ProviderAdapter):
    prefixes: tuple[str, ...] = ("opencode-go/",)

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return (
            SuggestedModel(
                id="opencode-go/kimi-k2.6",
                label="Kimi K2.6",
                description="OpenCode Go",
                provider="opencode_go",
                provider_label="OpenCode Go",
                avatar_url="https://huggingface.co/api/avatars/opencode-ai",
                recommended=True,
            ),
        )

    def allows_model_name(self, model_name: str) -> bool:
        if not self.matches(model_name):
            return False
        return bool(model_name.removeprefix("opencode-go/"))

    def build_params(
        self,
        model_name: str,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        model_id = model_name.removeprefix("opencode-go/")
        api_key = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get(
            "OPENCODE_API_KEY"
        )
        return {
            "model": f"openai/{model_id}",
            "api_base": "https://opencode.ai/zen/go/v1",
            "api_key": api_key,
        }


ADAPTERS: tuple[ProviderAdapter, ...] = (
    NativeAdapter(provider_id="native", provider_label="Native"),
    OpenCodeGoAdapter(
        provider_id="opencode_go",
        provider_label="OpenCode Go",
        supports_custom_model=True,
        custom_model_hint="Use opencode-go/<model-id>, for example opencode-go/kimi-k2.6",
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


def get_provider_summaries() -> list[dict[str, Any]]:
    return [adapter.to_summary() for adapter in ADAPTERS]


def find_model_option(model_name: str) -> dict[str, Any] | None:
    for model in get_available_models():
        if model["id"] == model_name:
            return model

    adapter = resolve_adapter(model_name)
    if not adapter or not adapter.supports_custom_model:
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
        "avatarUrl": "https://huggingface.co/api/avatars/huggingface",
        "recommended": False,
    }


def build_model_catalog(current_model: str) -> dict[str, Any]:
    return {
        "current": current_model,
        "available": get_available_models(),
        "providers": get_provider_summaries(),
        "currentInfo": find_model_option(current_model),
    }
