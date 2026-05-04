"""Authentication dependencies for FastAPI routes.

- In dev mode (OAUTH_CLIENT_ID not set): auth is bypassed, returns a default "dev" user.
- In production: validates Bearer tokens or cookies against HF OAuth.
"""

import logging
import os
import time
from typing import Any

import httpx
from fastapi import HTTPException, Request, status

from agent.core.hf_tokens import bearer_token_from_header, clean_hf_token

from agent.core.hf_access import fetch_whoami_v2

logger = logging.getLogger(__name__)

OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")
AUTH_ENABLED = bool(os.environ.get("OAUTH_CLIENT_ID", ""))
HF_EMPLOYEE_ORG = os.environ.get("HF_EMPLOYEE_ORG", "huggingface")

# Simple in-memory token cache: token -> (user_info, expiry_time)
_token_cache: dict[str, tuple[dict[str, Any], float]] = {}
TOKEN_CACHE_TTL = 300  # 5 minutes

# Org membership cache: key -> expiry_time (only caches positive results)
_org_member_cache: dict[str, float] = {}

DEV_USER: dict[str, Any] = {
    "user_id": "dev",
    "username": "dev",
    "authenticated": True,
    "plan": "org",  # Dev runs at the Pro/Org quota tier so local testing isn't capped.
}

INTERNAL_HF_TOKEN_KEY = "_hf_token"

# Plan field discovery — log the whoami-v2 shape once at DEBUG so we can
# confirm the actual key in production without hammering the HF API.
_WHOAMI_SHAPE_LOGGED = False


async def _validate_token(token: str) -> dict[str, Any] | None:
    """Validate a token against HF OAuth userinfo endpoint.

    Results are cached for TOKEN_CACHE_TTL seconds to avoid excessive API calls.
    """
    now = time.time()

    # Check cache
    if token in _token_cache:
        user_info, expiry = _token_cache[token]
        if now < expiry:
            return user_info
        del _token_cache[token]

    # Validate against HF
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"{OPENID_PROVIDER_URL}/oauth/userinfo",
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code != 200:
                logger.debug("Token validation failed: status %d", response.status_code)
                return None
            user_info = response.json()
            _token_cache[token] = (user_info, now + TOKEN_CACHE_TTL)
            return user_info
        except httpx.HTTPError as e:
            logger.warning("Token validation error: %s", e)
            return None


def _user_from_info(user_info: dict[str, Any]) -> dict[str, Any]:
    """Build a normalized user dict from HF userinfo response."""
    return {
        "user_id": user_info.get("sub", user_info.get("preferred_username", "unknown")),
        "username": user_info.get("preferred_username", "unknown"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
        "authenticated": True,
    }


async def _fetch_user_plan(token: str) -> str:
    """Look up the user's HF plan via /api/whoami-v2.

    Returns 'free' | 'pro' | 'org'. Non-200, network errors, or an unknown
    payload shape all collapse to 'free' — safe default; we'd rather under-
    grant the Pro cap than over-grant it on bad data.
    """
    global _WHOAMI_SHAPE_LOGGED
    whoami = await fetch_whoami_v2(token)
    if whoami is None:
        return "free"

    if not _WHOAMI_SHAPE_LOGGED:
        _WHOAMI_SHAPE_LOGGED = True
        logger.debug(
            "whoami-v2 payload keys: %s (sample values: plan=%r type=%r isPro=%r)",
            sorted(whoami.keys()) if isinstance(whoami, dict) else type(whoami).__name__,
            whoami.get("plan") if isinstance(whoami, dict) else None,
            whoami.get("type") if isinstance(whoami, dict) else None,
            whoami.get("isPro") if isinstance(whoami, dict) else None,
        )

    if not isinstance(whoami, dict):
        return "free"

    # OAuth whoami sets `type: "user"` and surfaces Pro via the `isPro` boolean
    # — see Space discussion #21. HF-Jobs eligibility (PR #172) ignores plan
    # entirely; the premium-model daily-cap tier is still a free vs pro/org split.
    if whoami.get("isPro") is True or whoami.get("is_pro") is True:
        return "pro"
    plan_str = ""
    for key in ("plan", "type", "accountType"):
        value = whoami.get(key)
        if isinstance(value, str) and value:
            plan_str = value.lower()
            break
    if any(tag in plan_str for tag in ("pro", "enterprise", "team")):
        return "pro"
    orgs = whoami.get("orgs") or []
    if isinstance(orgs, list) and orgs:
        return "org"
    return "free"


async def _extract_user_from_token(token: str) -> dict[str, Any] | None:
    """Validate a token and return a user dict, or None."""
    user_info = await _validate_token(token)
    if user_info is None:
        return None
    user = _user_from_info(user_info)
    user["plan"] = await _fetch_user_plan(token)
    user[INTERNAL_HF_TOKEN_KEY] = clean_hf_token(token)
    return user


async def _dev_user_from_env() -> dict[str, Any]:
    """Use HF_TOKEN as the dev identity when available.

    Local dev often runs without OAuth, but session trace uploads still need a
    real HF namespace. Deriving the dev user from HF_TOKEN keeps local uploads
    pointed at the token owner's dataset instead of dev/ml-intern-sessions.
    """
    token = clean_hf_token(os.environ.get("HF_TOKEN", ""))
    if not token:
        return dict(DEV_USER)

    whoami = await fetch_whoami_v2(token)
    if not isinstance(whoami, dict):
        return dict(DEV_USER)

    username = None
    for key in ("name", "user", "preferred_username"):
        value = whoami.get(key)
        if isinstance(value, str) and value:
            username = value
            break
    if not username:
        return dict(DEV_USER)

    return {
        "user_id": username,
        "username": username,
        "authenticated": True,
        "plan": await _fetch_user_plan(token),
        INTERNAL_HF_TOKEN_KEY: token,
    }


async def check_org_membership(token: str, org_name: str) -> bool:
    """Check if the token owner belongs to an HF org. Only caches positive results."""
    now = time.time()
    key = token + org_name
    cached = _org_member_cache.get(key)
    if cached and cached > now:
        return True

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                f"{OPENID_PROVIDER_URL}/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                return False
            orgs = {o.get("name") for o in resp.json().get("orgs", [])}
            if org_name in orgs:
                _org_member_cache[key] = now + TOKEN_CACHE_TTL
                return True
            return False
        except httpx.HTTPError:
            return False


async def get_current_user(request: Request) -> dict[str, Any]:
    """FastAPI dependency: extract and validate the current user.

    Checks (in order):
    1. Authorization: Bearer <token> header
    2. hf_access_token cookie

    In dev mode (AUTH_ENABLED=False), uses HF_TOKEN as the user when possible.
    """
    if not AUTH_ENABLED:
        return await _dev_user_from_env()

    # Try Authorization header
    token = bearer_token_from_header(request.headers.get("Authorization", ""))
    if token:
        user = await _extract_user_from_token(token)
        if user:
            return user

    # Try cookie
    token = request.cookies.get("hf_access_token")
    if token:
        user = await _extract_user_from_token(token)
        if user:
            return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Please log in via /auth/login.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_token(request: Request) -> str | None:
    """Pull the HF access token from the Authorization header or cookie.

    Mirrors the lookup order used by ``get_current_user``.
    """
    token = bearer_token_from_header(request.headers.get("Authorization", ""))
    if token:
        return token
    return request.cookies.get("hf_access_token")


async def require_huggingface_org_member(request: Request) -> bool:
    """Return True if the caller is a member of the ``huggingface`` org.

    Used to gate endpoints that can push a session onto an Anthropic model
    billed to the Space's ``ANTHROPIC_API_KEY``. Returns True unconditionally
    in dev mode so local testing isn't blocked.
    """
    if not AUTH_ENABLED:
        return True
    token = _extract_token(request)
    if not token:
        return False
    return await check_org_membership(token, HF_EMPLOYEE_ORG)
