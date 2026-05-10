"""OpenAI Codex OAuth helpers.

This mirrors Codex/Pi's ``openai-codex`` provider shape: ChatGPT OAuth
credentials are refreshable bearer tokens for ChatGPT's Codex backend, not
normal OpenAI API keys.
"""

from __future__ import annotations

import base64
import asyncio
import hashlib
import json
import os
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

try:
    import fcntl
except ImportError:  # pragma: no cover - this app is deployed on Linux.
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows-only fallback.
    msvcrt = None  # type: ignore[assignment]


CODEX_PROVIDER_ID = "openai-codex"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
CODEX_SCOPE = "openid profile email offline_access"
CODEX_JWT_CLAIM_PATH = "https://api.openai.com/auth"
CODEX_ORIGINATOR = os.environ.get("ML_INTERN_CODEX_ORIGINATOR", "pi")

_REFRESH_SKEW_MS = 60_000
_REFRESH_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class CodexCredentials:
    access: str
    refresh: str
    expires: int
    account_id: str
    source: str = "ml-intern"
    path: Path | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "oauth",
            "access": self.access,
            "refresh": self.refresh,
            "expires": self.expires,
            "accountId": self.account_id,
        }


@dataclass(frozen=True)
class CodexAuthorizationFlow:
    verifier: str
    state: str
    url: str


def _auth_path() -> Path:
    configured = os.environ.get("ML_INTERN_CODEX_AUTH_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "ml-intern" / "codex-auth.json"


def _codex_auth_path() -> Path:
    configured = os.environ.get("CODEX_AUTH_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex" / "auth.json"


def _pi_auth_path() -> Path:
    configured = os.environ.get("PI_CODEX_AUTH_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".pi" / "agent" / "auth.json"


def _user_key(user_id: str | None) -> str:
    normalized = (user_id or "dev").strip() or "dev"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def generate_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def create_codex_authorization_flow() -> CodexAuthorizationFlow:
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(24)
    params = {
        "response_type": "code",
        "client_id": CODEX_CLIENT_ID,
        "redirect_uri": CODEX_REDIRECT_URI,
        "scope": CODEX_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": CODEX_ORIGINATOR,
    }
    return CodexAuthorizationFlow(
        verifier=verifier,
        state=state,
        url=f"{CODEX_AUTHORIZE_URL}?{urlencode(params)}",
    )


def parse_authorization_response(value: str) -> tuple[str | None, str | None]:
    raw = (value or "").strip()
    if not raw:
        return None, None

    try:
        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            params = parse_qs(parsed.query)
            return _first(params.get("code")), _first(params.get("state"))
    except Exception:
        pass

    if "#" in raw:
        code, state = raw.split("#", 1)
        return code or None, state or None

    if "code=" in raw or "state=" in raw:
        params = parse_qs(raw)
        return _first(params.get("code")), _first(params.get("state"))

    return raw, None


def _first(values: list[str] | None) -> str | None:
    if not values:
        return None
    value = values[0].strip()
    return value or None


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        parsed = json.loads(decoded)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def extract_account_id(access_token: str) -> str | None:
    payload = _decode_jwt_payload(access_token)
    auth = payload.get(CODEX_JWT_CLAIM_PATH) if payload else None
    if not isinstance(auth, dict):
        return None
    account_id = auth.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


def _credentials_from_raw(raw: Any, *, source: str) -> CodexCredentials | None:
    if not isinstance(raw, dict):
        return None
    access = raw.get("access")
    refresh = raw.get("refresh")
    expires = raw.get("expires")
    account_id = raw.get("accountId") or raw.get("account_id")
    if not isinstance(access, str) or not isinstance(refresh, str):
        return None
    if not isinstance(expires, (int, float)):
        return None
    if not isinstance(account_id, str) or not account_id:
        account_id = extract_account_id(access) or ""
    if not account_id:
        return None
    return CodexCredentials(
        access=access,
        refresh=refresh,
        expires=int(expires),
        account_id=account_id,
        source=source,
    )


def _credentials_from_codex_data(
    data: dict[str, Any], path: Path
) -> CodexCredentials | None:
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        return None
    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    account_id = tokens.get("account_id")
    if not isinstance(access, str) or not isinstance(refresh, str):
        return None
    expires = jwt_expires_at_ms(access)
    if expires is None:
        return None
    if not isinstance(account_id, str) or not account_id:
        account_id = extract_account_id(access) or ""
    if not account_id:
        return None
    return CodexCredentials(
        access=access,
        refresh=refresh,
        expires=expires,
        account_id=account_id,
        source="codex",
        path=path,
    )


def _credentials_from_codex_file(path: Path) -> CodexCredentials | None:
    return _credentials_from_codex_data(_read_json(path), path)


def _credentials_from_stored_data(
    data: dict[str, Any],
    user_id: str | None,
) -> CodexCredentials | None:
    users = data.get("users")
    if not isinstance(users, dict):
        return None
    return _credentials_from_raw(users.get(_user_key(user_id)), source="ml-intern")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+b") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows-only fallback.
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows-only fallback.
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _locked_json(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_name(f"{path.name}.lock")
    with _file_lock(lock_path):
        data = _read_json(path)
        yield data

        tmp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, path)
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def store_codex_credentials(user_id: str | None, credentials: CodexCredentials) -> None:
    with _locked_json(_auth_path()) as data:
        users = data.setdefault("users", {})
        users[_user_key(user_id)] = credentials.to_json()


def delete_codex_credentials(user_id: str | None) -> None:
    with _locked_json(_auth_path()) as data:
        users = data.setdefault("users", {})
        users.pop(_user_key(user_id), None)


def load_stored_codex_credentials(user_id: str | None) -> CodexCredentials | None:
    return _credentials_from_stored_data(_read_json(_auth_path()), user_id)


def load_pi_codex_credentials() -> CodexCredentials | None:
    if os.environ.get("ML_INTERN_CODEX_USE_PI_AUTH", "1") == "0":
        return None
    data = _read_json(_pi_auth_path())
    return _credentials_from_raw(data.get(CODEX_PROVIDER_ID), source="pi")


def load_codex_cli_credentials() -> CodexCredentials | None:
    if os.environ.get("ML_INTERN_CODEX_USE_CODEX_AUTH", "1") == "0":
        return None
    return _credentials_from_codex_file(_codex_auth_path())


def load_codex_credentials_for_user(user_id: str | None) -> CodexCredentials | None:
    return (
        load_codex_cli_credentials()
        or load_stored_codex_credentials(user_id)
        or load_pi_codex_credentials()
    )


def has_codex_credentials(user_id: str | None) -> bool:
    return load_codex_credentials_for_user(user_id) is not None


def _expires_soon(credentials: CodexCredentials) -> bool:
    return int(time.time() * 1000) + _REFRESH_SKEW_MS >= credentials.expires


async def exchange_codex_authorization_code(
    code: str,
    verifier: str,
) -> CodexCredentials:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CODEX_CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": CODEX_REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    return _credentials_from_token_response(response, action="exchange")


async def refresh_codex_credentials(credentials: CodexCredentials) -> CodexCredentials:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": credentials.refresh,
                "client_id": CODEX_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    return _credentials_from_token_response(
        response,
        action="refresh",
        fallback_refresh=credentials.refresh,
    )


def jwt_expires_at_ms(access_token: str) -> int | None:
    payload = _decode_jwt_payload(access_token)
    exp = payload.get("exp") if payload else None
    if not isinstance(exp, (int, float)):
        return None
    return int(float(exp) * 1000)


def _credentials_from_token_response(
    response: httpx.Response,
    *,
    action: str,
    fallback_refresh: str | None = None,
) -> CodexCredentials:
    if not response.is_success:
        text = response.text[:500] if response.text else response.reason_phrase
        raise ValueError(
            f"OpenAI Codex token {action} failed ({response.status_code}): {text}"
        )
    payload = response.json()
    access = payload.get("access_token")
    refresh = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access, str) or not access:
        raise ValueError(f"OpenAI Codex token {action} response missing access token")
    if not isinstance(refresh, str) or not refresh:
        refresh = fallback_refresh
    if not isinstance(refresh, str) or not refresh:
        raise ValueError(f"OpenAI Codex token {action} response missing refresh token")
    if not isinstance(expires_in, (int, float)):
        raise ValueError(f"OpenAI Codex token {action} response missing expires_in")
    account_id = extract_account_id(access)
    if not account_id:
        raise ValueError("OpenAI Codex token did not include a ChatGPT account id")
    jwt_expires = jwt_expires_at_ms(access)
    return CodexCredentials(
        access=access,
        refresh=refresh,
        expires=jwt_expires or int(time.time() * 1000 + float(expires_in) * 1000),
        account_id=account_id,
    )


def _write_codex_cli_credentials(
    data: dict[str, Any],
    credentials: CodexCredentials,
) -> None:
    tokens = data.setdefault("tokens", {})
    tokens["access_token"] = credentials.access
    tokens["refresh_token"] = credentials.refresh
    tokens["account_id"] = credentials.account_id
    # Codex stores id_token too, but ChatGPT Codex backend calls only need
    # access_token + refresh_token + account_id. Leave an existing id_token
    # untouched because refresh_token responses for this client do not need
    # to return a new one.
    data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _store_refreshed_codex_cli_credentials(credentials: CodexCredentials) -> bool:
    if credentials.path is None or credentials.source != "codex":
        return False
    with _locked_json(credentials.path) as data:
        _write_codex_cli_credentials(data, credentials)
    return True


async def _refresh_codex_cli_credentials_from_file(
    path: Path,
) -> CodexCredentials | None:
    try:
        with _locked_json(path) as data:
            current = _credentials_from_codex_data(data, path)
            if current is None:
                return None
            if not _expires_soon(current):
                return current

            refreshed = await refresh_codex_credentials(current)
            refreshed = CodexCredentials(
                access=refreshed.access,
                refresh=refreshed.refresh,
                expires=refreshed.expires,
                account_id=refreshed.account_id,
                source=current.source,
                path=current.path,
            )
            _write_codex_cli_credentials(data, refreshed)
            return refreshed
    except Exception:
        latest = _credentials_from_codex_file(path)
        if latest is not None and not _expires_soon(latest):
            return latest
        raise


async def _refresh_stored_codex_credentials(
    user_id: str | None,
) -> CodexCredentials | None:
    try:
        with _locked_json(_auth_path()) as data:
            current = _credentials_from_stored_data(data, user_id)
            if current is None:
                return None
            if not _expires_soon(current):
                return current

            refreshed = await refresh_codex_credentials(current)
            users = data.setdefault("users", {})
            users[_user_key(user_id)] = refreshed.to_json()
            return refreshed
    except Exception:
        latest = load_stored_codex_credentials(user_id)
        if latest is not None and not _expires_soon(latest):
            return latest
        raise


async def get_codex_credentials_for_user(
    user_id: str | None,
) -> CodexCredentials | None:
    credentials = load_codex_credentials_for_user(user_id)
    if credentials is None:
        return None
    if not _expires_soon(credentials):
        return credentials

    async with _REFRESH_LOCK:
        current = load_codex_credentials_for_user(user_id)
        if current is None:
            return None
        if not _expires_soon(current):
            return current

        if current.source == "codex" and current.path is not None:
            return await _refresh_codex_cli_credentials_from_file(current.path)
        if current.source == "ml-intern":
            return await _refresh_stored_codex_credentials(user_id)
        if current.source == "pi":
            # Pi owns ~/.pi/agent/auth.json and uses its own lock format. Do not
            # rotate Pi's refresh token from here; otherwise Pi can be left with
            # a stale refresh token. Valid Pi access tokens are still usable.
            return None

        refreshed = await refresh_codex_credentials(current)
        store_codex_credentials(user_id, refreshed)
        return refreshed


def codex_request_headers(
    credentials: CodexCredentials,
    *,
    session_id: str | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {credentials.access}",
        "chatgpt-account-id": credentials.account_id,
        "originator": CODEX_ORIGINATOR,
        "User-Agent": "pi (ml-intern)",
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    if session_id:
        headers["session_id"] = session_id
        headers["x-client-request-id"] = session_id
    return headers
