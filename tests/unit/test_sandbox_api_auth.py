from fastapi.testclient import TestClient

from agent.tools.sandbox_client import _SANDBOX_SERVER, Sandbox


def _sandbox_app(monkeypatch, token: str | None = "sandbox-secret"):
    if token is None:
        monkeypatch.delenv("SANDBOX_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SANDBOX_API_TOKEN", token)
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
