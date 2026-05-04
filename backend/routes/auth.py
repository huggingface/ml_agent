"""Authentication routes for HF OAuth.

Handles the OAuth 2.0 authorization code flow with HF as provider.
After successful auth, sets an HttpOnly cookie with the access token.
"""

import os
import secrets
import time
from urllib.parse import urlencode

import httpx
from dependencies import AUTH_ENABLED, get_current_user
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/auth", tags=["auth"])

# OAuth configuration from environment
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")

# In-memory OAuth state store with expiry (5 min TTL)
_OAUTH_STATE_TTL = 300
oauth_states: dict[str, dict] = {}


def _cleanup_expired_states() -> None:
    """Remove expired OAuth states to prevent memory growth."""
    now = time.time()
    expired = [k for k, v in oauth_states.items() if now > v.get("expires_at", 0)]
    for k in expired:
        del oauth_states[k]


def get_redirect_uri(request: Request) -> str:
    """Get the OAuth callback redirect URI."""
    # In HF Spaces, use the SPACE_HOST if available
    space_host = os.environ.get("SPACE_HOST")
    if space_host:
        return f"https://{space_host}/auth/callback"
    # Otherwise construct from request
    return str(request.url_for("oauth_callback"))


@router.get("/login")
async def oauth_login(request: Request) -> RedirectResponse:
    """Initiate OAuth login flow."""
    if not OAUTH_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="OAuth not configured. Set OAUTH_CLIENT_ID environment variable.",
        )

    # Clean up expired states to prevent memory growth
    _cleanup_expired_states()

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {
        "redirect_uri": get_redirect_uri(request),
        "expires_at": time.time() + _OAUTH_STATE_TTL,
    }

    # Build authorization URL. We no longer suggest a default `orgIds` —
    # users no longer need to join the ML Agent Explorers org to use the
    # app, and HF Jobs are billed per-namespace via credits.
    params = {
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": get_redirect_uri(request),
        "scope": "openid profile read-repos write-repos contribute-repos manage-repos inference-api jobs write-discussions",
        "response_type": "code",
        "state": state,
    }
    auth_url = f"{OPENID_PROVIDER_URL}/oauth/authorize?{urlencode(params)}"

    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def oauth_callback(
    request: Request, code: str = "", state: str = ""
) -> RedirectResponse:
    """Handle OAuth callback."""
    # Verify state
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    stored_state = oauth_states.pop(state)
    redirect_uri = stored_state["redirect_uri"]

    if not code:
        raise HTTPException(status_code=400, detail="No authorization code provided")

    # Exchange code for token
    token_url = f"{OPENID_PROVIDER_URL}/oauth/token"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": OAUTH_CLIENT_ID,
                    "client_secret": OAUTH_CLIENT_SECRET,
                },
            )
            response.raise_for_status()
            token_data = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Token exchange failed: {e}")

    # Get user info
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=500,
            detail="Token exchange succeeded but no access_token was returned.",
        )

    # Fetch user info (optional — failure is not fatal)
    async with httpx.AsyncClient() as client:
        try:
            userinfo_response = await client.get(
                f"{OPENID_PROVIDER_URL}/oauth/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            userinfo_response.raise_for_status()
        except httpx.HTTPError:
            pass  # user_info not required for auth flow

    # Set access token as HttpOnly cookie (not in URL — avoids leaks via
    # Referrer headers, browser history, and server logs)
    is_production = bool(os.environ.get("SPACE_HOST"))
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="hf_access_token",
        value=access_token,
        httponly=True,
        secure=is_production,  # Secure flag only in production (HTTPS)
        samesite="lax",
        max_age=3600 * 24 * 7,  # 7 days
        path="/",
    )
    return response


@router.get("/logout")
async def logout() -> RedirectResponse:
    """Log out the user by clearing the auth cookie."""
    response = RedirectResponse(url="/")
    response.delete_cookie(key="hf_access_token", path="/")
    return response


@router.get("/status")
async def auth_status() -> dict:
    """Check if OAuth is enabled on this instance."""
    return {"auth_enabled": AUTH_ENABLED}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)) -> dict:
    """Get current user info. Returns the authenticated user or dev user.

    Uses the shared auth dependency which handles cookie + Bearer token.
    """
    return {key: value for key, value in user.items() if not key.startswith("_")}

