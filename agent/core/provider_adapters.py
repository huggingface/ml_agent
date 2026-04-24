"""Provider adapters for runtime params and model-name validation."""

import os
from dataclasses import dataclass
from typing import Any, ClassVar


class UnsupportedEffortError(ValueError):
    """The requested effort isn't valid for this provider's API surface.

    Raised synchronously before any network call so the probe cascade can
    skip levels the provider can't accept (e.g. ``max`` on HF router).
    """


def _has_model_suffix(model_name: str, prefix: str) -> bool:
    if not model_name.startswith(prefix):
        return False
    tail = model_name[len(prefix) :].split(":", 1)[0]
    return bool(tail) and all(tail.split("/"))


def _is_hf_model_name(model_name: str) -> bool:
    if model_name.startswith(("anthropic/", "openai/", "bedrock/")):
        return False
    bare = model_name.removeprefix("huggingface/").split(":", 1)[0]
    parts = bare.split("/")
    return len(parts) >= 2 and all(parts)


@dataclass(frozen=True)
class ProviderAdapter:
    provider_id: str
    prefixes: tuple[str, ...] = ()

    def matches(self, model_name: str) -> bool:
        return bool(self.prefixes) and model_name.startswith(self.prefixes)

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


@dataclass(frozen=True)
class AnthropicAdapter(ProviderAdapter):
    """Anthropic models via native API (thinking + output_config.effort)."""

    prefixes: tuple[str, ...] = ("anthropic/",)
    _EFFORTS: ClassVar[frozenset[str]] = frozenset(
        {"low", "medium", "high", "xhigh", "max"}
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
class BedrockAdapter(ProviderAdapter):
    """AWS Bedrock models via LiteLLM Converse adapter.

    Picks up AWS credentials from standard env vars.
    Thinking/effort not forwarded through Converse for now.
    """

    prefixes: tuple[str, ...] = ("bedrock/",)

    def allows_model_name(self, model_name: str) -> bool:
        return _has_model_suffix(model_name, "bedrock/")

    def build_params(
        self,
        model_name: str,
        *,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
        strict: bool = False,
    ) -> dict:
        return {"model": model_name}


@dataclass(frozen=True)
class HfRouterAdapter(ProviderAdapter):
    """HuggingFace router — OpenAI-compat endpoint with HF token chain."""

    _EFFORTS: ClassVar[frozenset[str]] = frozenset({"low", "medium", "high"})

    def matches(self, model_name: str) -> bool:
        return not model_name.startswith(("anthropic/", "openai/", "bedrock/"))

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
    AnthropicAdapter(provider_id="anthropic"),
    BedrockAdapter(provider_id="bedrock"),
    OpenAIAdapter(provider_id="openai"),
    HfRouterAdapter(provider_id="huggingface"),
)


def resolve_adapter(model_name: str) -> ProviderAdapter | None:
    for adapter in ADAPTERS:
        if adapter.matches(model_name):
            return adapter
    return None


def is_valid_model_name(model_name: str) -> bool:
    adapter = resolve_adapter(model_name)
    return adapter is not None and adapter.allows_model_name(model_name)
