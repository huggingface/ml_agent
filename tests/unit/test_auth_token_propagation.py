"""Tests for authenticated HF token propagation through backend dependencies."""

import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import dependencies  # noqa: E402
from routes import auth  # noqa: E402


@pytest.mark.asyncio
async def test_current_user_carries_internal_hf_token(monkeypatch):
    monkeypatch.setattr(dependencies, "AUTH_ENABLED", True)
    dependencies._token_cache.clear()

    async def fake_validate_token(token):
        assert token == "hf-user-token"
        return {"sub": "user-id", "preferred_username": "alice"}

    async def fake_fetch_user_plan(token):
        assert token == "hf-user-token"
        return "pro"

    monkeypatch.setattr(dependencies, "_validate_token", fake_validate_token)
    monkeypatch.setattr(dependencies, "_fetch_user_plan", fake_fetch_user_plan)

    request = SimpleNamespace(
        headers={"Authorization": "Bearer hf-user-token"},
        cookies={},
    )

    user = await dependencies.get_current_user(request)

    assert user["user_id"] == "user-id"
    assert user["username"] == "alice"
    assert user["plan"] == "pro"
    assert user[dependencies.INTERNAL_HF_TOKEN_KEY] == "hf-user-token"


@pytest.mark.asyncio
async def test_cookie_auth_requires_current_oauth_scope_marker(monkeypatch):
    monkeypatch.setattr(dependencies, "AUTH_ENABLED", True)

    request = SimpleNamespace(
        headers={},
        cookies={"hf_access_token": "hf-user-token"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await dependencies.get_current_user(request)

    assert exc_info.value.status_code == 401
    assert "scopes changed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_cookie_auth_accepts_current_oauth_scope_marker(monkeypatch):
    monkeypatch.setattr(dependencies, "AUTH_ENABLED", True)
    dependencies._token_cache.clear()

    async def fake_validate_token(token):
        assert token == "hf-user-token"
        return {"sub": "user-id", "preferred_username": "alice"}

    async def fake_fetch_user_plan(token):
        assert token == "hf-user-token"
        return "pro"

    monkeypatch.setattr(dependencies, "_validate_token", fake_validate_token)
    monkeypatch.setattr(dependencies, "_fetch_user_plan", fake_fetch_user_plan)

    request = SimpleNamespace(
        headers={},
        cookies={
            "hf_access_token": "hf-user-token",
            dependencies.OAUTH_SCOPE_COOKIE: dependencies.oauth_scope_fingerprint(),
        },
    )

    user = await dependencies.get_current_user(request)

    assert user["user_id"] == "user-id"
    assert user[dependencies.INTERNAL_HF_TOKEN_KEY] == "hf-user-token"


@pytest.mark.asyncio
async def test_auth_me_does_not_expose_internal_hf_token():
    user = {
        "user_id": "user-id",
        "username": "alice",
        "authenticated": True,
        dependencies.INTERNAL_HF_TOKEN_KEY: "hf-user-token",
    }

    response = await auth.get_me(user)

    assert response == {
        "user_id": "user-id",
        "username": "alice",
        "authenticated": True,
    }


@pytest.mark.asyncio
async def test_oauth_login_requests_collection_write_scope(monkeypatch):
    monkeypatch.setattr(auth, "OAUTH_CLIENT_ID", "oauth-client")
    monkeypatch.setenv("SPACE_HOST", "example.hf.space")
    auth.oauth_states.clear()

    response = await auth.oauth_login(SimpleNamespace())
    params = parse_qs(urlparse(response.headers["location"]).query)
    scopes = set(params["scope"][0].split())

    assert "write-collections" in scopes


def test_oauth_callback_detects_missing_required_collection_scope():
    granted = [scope for scope in auth.OAUTH_SCOPES if scope != "write-collections"]

    assert auth._missing_required_scopes({"scope": " ".join(granted)}) == {
        "write-collections"
    }


@pytest.mark.asyncio
async def test_oauth_callback_sets_scope_marker_cookie(monkeypatch):
    monkeypatch.setenv("SPACE_HOST", "example.hf.space")
    auth.oauth_states.clear()
    auth.oauth_states["state"] = {
        "redirect_uri": "https://example.hf.space/auth/callback",
        "expires_at": 9999999999,
    }

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse(
                {
                    "access_token": "hf-user-token",
                    "scope": " ".join(auth.OAUTH_SCOPES),
                }
            )

        async def get(self, *args, **kwargs):
            return FakeResponse({})

    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeAsyncClient)

    response = await auth.oauth_callback(SimpleNamespace(), code="code", state="state")
    set_cookies = [
        value.decode("latin-1")
        for key, value in response.raw_headers
        if key == b"set-cookie"
    ]

    expected = (
        f"{dependencies.OAUTH_SCOPE_COOKIE}="
        f"{dependencies.oauth_scope_fingerprint(auth.OAUTH_SCOPES)}"
    )
    assert any(cookie.startswith(expected) for cookie in set_cookies)
