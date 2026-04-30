import hashlib

from fastapi.testclient import TestClient

from agent.tools.sandbox_client import (
    _SANDBOX_SERVER,
    _validate_sandbox_variables,
    SANDBOX_API_TOKEN_HEADER,
    Sandbox,
)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _sandbox_app(
    monkeypatch,
    token: str | None = "sandbox-secret",
    *,
    hf_token: str | None = None,
):
    monkeypatch.delenv("SANDBOX_API_TOKEN", raising=False)
    monkeypatch.delenv("SANDBOX_API_TOKEN_SHA256", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    if token is not None:
        monkeypatch.setenv("SANDBOX_API_TOKEN_SHA256", _hash_token(token))
    if hf_token is not None:
        monkeypatch.setenv("HF_TOKEN", hf_token)
    namespace = {}
    exec(_SANDBOX_SERVER, namespace)
    return namespace["app"]


def test_health_is_public(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_file_and_command_routes_require_bearer_token(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, "sandbox-secret"))

    for path, payload in [
        ("/api/bash", {"command": "true"}),
        ("/api/kill", {}),
        ("/api/read", {"path": "/tmp/x"}),
        ("/api/write", {"path": "/tmp/x", "content": "x"}),
        ("/api/edit", {"path": "/tmp/x", "old_str": "x", "new_str": "y"}),
        ("/api/exists", {"path": "/tmp"}),
    ]:
        response = client.post(path, json=payload)
        assert response.status_code == 401


def test_file_and_command_routes_accept_valid_bearer_token(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, "sandbox-secret"))

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={"Authorization": "Bearer sandbox-secret"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_file_and_command_routes_accept_valid_sandbox_header(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, "sandbox-secret"))

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={SANDBOX_API_TOKEN_HEADER: "sandbox-secret"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_sandbox_header_takes_precedence_over_authorization(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, "sandbox-secret"))

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={
            SANDBOX_API_TOKEN_HEADER: "wrong-secret",
            "Authorization": "Bearer sandbox-secret",
        },
    )

    assert response.status_code == 401


def test_file_and_command_routes_reject_wrong_bearer_token(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, "sandbox-secret"))

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={"Authorization": "Bearer wrong-secret"},
    )

    assert response.status_code == 401


def test_legacy_hf_token_fallback_is_not_accepted(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, token=None, hf_token="hf-secret"))

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={"Authorization": "Bearer hf-secret"},
    )

    assert response.status_code == 503


def test_legacy_raw_sandbox_token_env_is_not_accepted(monkeypatch):
    monkeypatch.delenv("SANDBOX_API_TOKEN_SHA256", raising=False)
    monkeypatch.setenv("SANDBOX_API_TOKEN", "sandbox-secret")
    namespace = {}
    exec(_SANDBOX_SERVER, namespace)
    client = TestClient(namespace["app"])

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={"Authorization": "Bearer sandbox-secret"},
    )

    assert response.status_code == 503


def test_protected_routes_fail_closed_without_configured_token(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, None))

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={"Authorization": "Bearer anything"},
    )

    assert response.status_code == 503


def test_private_sandbox_uses_hf_auth_and_control_plane_header():
    sandbox = Sandbox("owner/name", token="hf-token", api_token="sandbox-secret")

    assert sandbox._client.headers["authorization"] == "Bearer hf-token"
    assert sandbox._client.headers["x-sandbox-api-token"] == "sandbox-secret"


def test_public_sandbox_omits_hf_auth_header():
    sandbox = Sandbox(
        "owner/name",
        token="hf-token",
        api_token="sandbox-secret",
        private=False,
    )

    assert "authorization" not in sandbox._client.headers
    assert sandbox._client.headers["x-sandbox-api-token"] == "sandbox-secret"


def test_sandbox_does_not_fallback_to_hf_token_for_api_headers():
    sandbox = Sandbox("owner/name", token="hf-token")

    assert sandbox._client.headers["authorization"] == "Bearer hf-token"
    assert "x-sandbox-api-token" not in sandbox._client.headers
    result = sandbox._call("exists", {"path": "/tmp"})
    assert result.success is False
    assert "Sandbox API token is required" in result.error


def test_tokens_are_hidden_from_repr():
    sandbox = Sandbox("owner/name", token="hf-token", api_token="sandbox-secret")

    assert "sandbox-secret" not in repr(sandbox)
    assert "hf-token" not in repr(sandbox)


def test_sandbox_variables_allow_trackio_only():
    assert _validate_sandbox_variables(
        {"TRACKIO_SPACE_ID": "user/dashboard", "TRACKIO_PROJECT": "run"}
    ) == {"TRACKIO_SPACE_ID": "user/dashboard", "TRACKIO_PROJECT": "run"}


def test_sandbox_variables_reject_hf_token():
    try:
        _validate_sandbox_variables({"HF_TOKEN": "hf_secret"})
    except ValueError as e:
        assert "HF_TOKEN cannot be injected" in str(e)
    else:
        raise AssertionError("HF_TOKEN was accepted as a sandbox variable")


def test_sandbox_variables_reject_arbitrary_names():
    try:
        _validate_sandbox_variables({"SAFE_LOOKING": "value"})
    except ValueError as e:
        assert "not an allowed sandbox variable" in str(e)
    else:
        raise AssertionError("Unexpected sandbox variable was accepted")
