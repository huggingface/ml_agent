import base64
import json
import time

import httpx
import pytest

from agent.core import codex_oauth
from agent.core.codex_oauth import (
    CodexCredentials,
    _credentials_from_token_response,
    _store_refreshed_codex_cli_credentials,
    get_codex_credentials_for_user,
    load_codex_cli_credentials,
    store_codex_credentials,
)


def _jwt(payload: dict) -> str:
    def encode(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


def test_load_codex_cli_credentials_reads_auth_json(monkeypatch, tmp_path):
    auth_path = tmp_path / "auth.json"
    access_token = _jwt(
        {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_test",
            },
        }
    )
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh_test",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_path))

    credentials = load_codex_cli_credentials()

    assert credentials is not None
    assert credentials.source == "codex"
    assert credentials.path == auth_path
    assert credentials.access == access_token
    assert credentials.refresh == "refresh_test"
    assert credentials.account_id == "acct_test"


def test_refreshed_codex_cli_credentials_preserve_codex_file_shape(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "old_access",
                    "refresh_token": "old_refresh",
                    "id_token": "keep_id_token",
                    "account_id": "old_account",
                },
            }
        ),
        encoding="utf-8",
    )
    refreshed = CodexCredentials(
        access="new_access",
        refresh="new_refresh",
        expires=123,
        account_id="new_account",
        source="codex",
        path=auth_path,
    )

    assert _store_refreshed_codex_cli_credentials(refreshed)

    data = json.loads(auth_path.read_text(encoding="utf-8"))
    assert "users" not in data
    assert data["tokens"]["access_token"] == "new_access"
    assert data["tokens"]["refresh_token"] == "new_refresh"
    assert data["tokens"]["account_id"] == "new_account"
    assert data["tokens"]["id_token"] == "keep_id_token"
    assert data["last_refresh"]


def test_refresh_token_response_can_preserve_existing_refresh_token():
    access_token = _jwt(
        {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_test",
            },
        }
    )
    response = httpx.Response(
        200,
        json={
            "access_token": access_token,
            "expires_in": 3600,
        },
    )

    credentials = _credentials_from_token_response(
        response,
        action="refresh",
        fallback_refresh="old_refresh",
    )

    assert credentials.access == access_token
    assert credentials.refresh == "old_refresh"
    assert credentials.account_id == "acct_test"


@pytest.mark.asyncio
async def test_get_codex_credentials_refreshes_codex_file_under_lock(
    monkeypatch,
    tmp_path,
):
    auth_path = tmp_path / "auth.json"
    old_access = _jwt(
        {
            "exp": int(time.time()) - 10,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "old_account",
            },
        }
    )
    new_access = _jwt(
        {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "new_account",
            },
        }
    )
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": old_access,
                    "refresh_token": "old_refresh",
                    "id_token": "keep_id_token",
                    "account_id": "old_account",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_path))

    async def fake_refresh(credentials):
        assert credentials.refresh == "old_refresh"
        return CodexCredentials(
            access=new_access,
            refresh="new_refresh",
            expires=int(time.time() * 1000) + 3_600_000,
            account_id="new_account",
        )

    monkeypatch.setattr(codex_oauth, "refresh_codex_credentials", fake_refresh)

    credentials = await get_codex_credentials_for_user("user-1")

    assert credentials is not None
    assert credentials.source == "codex"
    assert credentials.access == new_access
    assert credentials.refresh == "new_refresh"

    data = json.loads(auth_path.read_text(encoding="utf-8"))
    assert data["tokens"]["access_token"] == new_access
    assert data["tokens"]["refresh_token"] == "new_refresh"
    assert data["tokens"]["account_id"] == "new_account"
    assert data["tokens"]["id_token"] == "keep_id_token"


@pytest.mark.asyncio
async def test_get_codex_credentials_refreshes_internal_store_under_lock(
    monkeypatch,
    tmp_path,
):
    auth_path = tmp_path / "ml-intern-auth.json"
    old_access = _jwt(
        {
            "exp": int(time.time()) - 10,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "old_account",
            },
        }
    )
    new_access = _jwt(
        {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "new_account",
            },
        }
    )
    monkeypatch.setenv("ML_INTERN_CODEX_AUTH_PATH", str(auth_path))
    monkeypatch.setenv("ML_INTERN_CODEX_USE_CODEX_AUTH", "0")
    monkeypatch.setenv("ML_INTERN_CODEX_USE_PI_AUTH", "0")
    store_codex_credentials(
        "user-1",
        CodexCredentials(
            access=old_access,
            refresh="old_refresh",
            expires=int(time.time() * 1000) - 10_000,
            account_id="old_account",
        ),
    )

    async def fake_refresh(credentials):
        assert credentials.refresh == "old_refresh"
        return CodexCredentials(
            access=new_access,
            refresh="new_refresh",
            expires=int(time.time() * 1000) + 3_600_000,
            account_id="new_account",
        )

    monkeypatch.setattr(codex_oauth, "refresh_codex_credentials", fake_refresh)

    credentials = await get_codex_credentials_for_user("user-1")

    assert credentials is not None
    assert credentials.access == new_access
    assert credentials.refresh == "new_refresh"

    data = json.loads(auth_path.read_text(encoding="utf-8"))
    stored = data["users"][codex_oauth._user_key("user-1")]
    assert stored["access"] == new_access
    assert stored["refresh"] == "new_refresh"
    assert stored["accountId"] == "new_account"


@pytest.mark.asyncio
async def test_expired_pi_credentials_are_not_refreshed_or_copied(
    monkeypatch,
    tmp_path,
):
    pi_auth_path = tmp_path / "pi-auth.json"
    ml_intern_auth_path = tmp_path / "ml-intern-auth.json"
    expired_access = _jwt(
        {
            "exp": int(time.time()) - 10,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "pi_account",
            },
        }
    )
    pi_auth_path.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": expired_access,
                    "refresh": "pi_refresh",
                    "expires": int(time.time() * 1000) - 10_000,
                    "accountId": "pi_account",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ML_INTERN_CODEX_USE_CODEX_AUTH", "0")
    monkeypatch.setenv("ML_INTERN_CODEX_AUTH_PATH", str(ml_intern_auth_path))
    monkeypatch.setenv("PI_CODEX_AUTH_PATH", str(pi_auth_path))

    async def fail_refresh(credentials):
        raise AssertionError("Pi-owned refresh tokens must not be rotated by ml-intern")

    monkeypatch.setattr(codex_oauth, "refresh_codex_credentials", fail_refresh)

    credentials = await get_codex_credentials_for_user("user-1")

    assert credentials is None
    assert not ml_intern_auth_path.exists()
    data = json.loads(pi_auth_path.read_text(encoding="utf-8"))
    assert data["openai-codex"]["refresh"] == "pi_refresh"
