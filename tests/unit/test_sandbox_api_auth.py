from fastapi.testclient import TestClient

from agent.tools.sandbox_client import _SANDBOX_SERVER, Sandbox


def _sandbox_app(
    monkeypatch,
    token: str | None = "sandbox-secret",
    *,
    hf_token: str | None = None,
):
    monkeypatch.delenv("SANDBOX_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    if token is not None:
        monkeypatch.setenv("SANDBOX_API_TOKEN", token)
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

    response = client.post("/api/exists", json={"path": "/tmp"})

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


def test_legacy_hf_token_fallback_is_accepted(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, token=None, hf_token="hf-secret"))

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={"Authorization": "Bearer hf-secret"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_protected_routes_fail_closed_without_configured_token(monkeypatch):
    client = TestClient(_sandbox_app(monkeypatch, None))

    response = client.post(
        "/api/exists",
        json={"path": "/tmp"},
        headers={"Authorization": "Bearer anything"},
    )

    assert response.status_code == 503


def test_sandbox_prefers_control_plane_token_for_api_headers():
    sandbox = Sandbox("owner/name", token="hf-token", api_token="sandbox-secret")

    assert sandbox._client.headers["authorization"] == "Bearer sandbox-secret"


def test_sandbox_api_token_is_hidden_from_repr():
    sandbox = Sandbox("owner/name", token="hf-token", api_token="sandbox-secret")

    assert "sandbox-secret" not in repr(sandbox)
