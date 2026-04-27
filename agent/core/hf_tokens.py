"""Hugging Face token resolution helpers."""

from __future__ import annotations

import os
from typing import Any


def clean_hf_token(token: str | None) -> str | None:
    """Normalize token strings the same way huggingface_hub does."""
    if token is None:
        return None
    return token.replace("\r", "").replace("\n", "").strip() or None


def get_cached_hf_token() -> str | None:
    """Return the token from huggingface_hub's normal env/cache lookup."""
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def resolve_hf_token(
    *candidates: str | None,
    include_cached: bool = True,
) -> str | None:
    """Return the first non-empty explicit token, then optionally HF cache."""
    for token in candidates:
        cleaned = clean_hf_token(token)
        if cleaned:
            return cleaned
    if include_cached:
        return get_cached_hf_token()
    return None


def resolve_hf_router_token(session_hf_token: str | None = None) -> str | None:
    """Resolve the token used for Hugging Face Router LLM calls.

    App-specific precedence:
    1. INFERENCE_TOKEN: shared hosted-Space inference token.
    2. session_hf_token: the active user/session token.
    3. huggingface_hub.get_token(): HF_TOKEN/HUGGING_FACE_HUB_TOKEN or
       local ``hf auth login`` cache.
    """
    return resolve_hf_token(os.environ.get("INFERENCE_TOKEN"), session_hf_token)


def get_hf_bill_to() -> str | None:
    """Return X-HF-Bill-To only when a shared inference token is active."""
    if clean_hf_token(os.environ.get("INFERENCE_TOKEN")):
        return os.environ.get("HF_BILL_TO", "smolagents")
    return None


def bearer_token_from_header(auth_header: str | None) -> str | None:
    """Extract a cleaned bearer token from an Authorization header."""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    return clean_hf_token(auth_header[7:])


def resolve_hf_request_token(
    request: Any,
    *,
    include_env_fallback: bool = True,
) -> str | None:
    """Resolve a user token from a FastAPI request.

    This intentionally does not use the local ``hf auth login`` cache. Backend
    request paths should act as the browser user from Authorization/cookie, or
    fall back only to an explicit server ``HF_TOKEN`` in dev/server contexts.
    """
    token = bearer_token_from_header(request.headers.get("Authorization", ""))
    if token:
        return token
    token = clean_hf_token(request.cookies.get("hf_access_token"))
    if token:
        return token
    if include_env_fallback:
        return clean_hf_token(os.environ.get("HF_TOKEN"))
    return None
