"""Helpers for CLI local OpenAI-compatible model ids."""

LOCAL_MODEL_PROVIDERS: dict[str, dict[str, str]] = {
    "ollama/": {
        "base_url_env": "OLLAMA_BASE_URL",
        "base_url_default": "http://localhost:11434",
        "api_key_env": "OLLAMA_API_KEY",
    },
    "vllm/": {
        "base_url_env": "VLLM_BASE_URL",
        "base_url_default": "http://localhost:8000",
        "api_key_env": "VLLM_API_KEY",
    },
    "lm_studio/": {
        "base_url_env": "LMSTUDIO_BASE_URL",
        "base_url_default": "http://127.0.0.1:1234",
        "api_key_env": "LMSTUDIO_API_KEY",
    },
    "llamacpp/": {
        "base_url_env": "LLAMACPP_BASE_URL",
        "base_url_default": "http://localhost:8080",
        "api_key_env": "LLAMACPP_API_KEY",
    },
}

LOCAL_MODEL_PREFIXES = tuple(LOCAL_MODEL_PROVIDERS)
RESERVED_LOCAL_MODEL_PREFIXES = ("openai-compat/",)
LOCAL_MODEL_BASE_URL_ENV = "LOCAL_LLM_BASE_URL"
LOCAL_MODEL_API_KEY_ENV = "LOCAL_LLM_API_KEY"
LOCAL_MODEL_API_KEY_DEFAULT = "sk-local-no-key-required"


def local_model_provider(model_id: str) -> dict[str, str] | None:
    """Return provider config for a local model id, if it uses a local prefix."""
    for prefix, config in LOCAL_MODEL_PROVIDERS.items():
        if model_id.startswith(prefix):
            return config
    return None


def local_model_name(model_id: str) -> str | None:
    """Return the backend model name with the local provider prefix removed."""
    for prefix in LOCAL_MODEL_PREFIXES:
        if model_id.startswith(prefix):
            name = model_id[len(prefix) :]
            return name or None
    return None


def is_local_model_id(model_id: str) -> bool:
    """Return True for non-empty, whitespace-free local model ids."""
    if not model_id or any(char.isspace() for char in model_id):
        return False
    return local_model_name(model_id) is not None


def is_reserved_local_model_id(model_id: str) -> bool:
    """Return True for local-style prefixes intentionally not supported."""
    return model_id.startswith(RESERVED_LOCAL_MODEL_PREFIXES)
