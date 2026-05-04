"""Tests for authenticated HF token propagation through backend dependencies."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
