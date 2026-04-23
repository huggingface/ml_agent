from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from main import app  # noqa: E402
from routes import agent as agent_routes  # noqa: E402


client = TestClient(app)


class DummySession:
    def __init__(self, model_name: str) -> None:
        self.config = SimpleNamespace(model_name=model_name)
        self.context_manager = SimpleNamespace(items=[])
        self.pending_approval = None

    def update_model(self, model_name: str) -> None:
        self.config.model_name = model_name


def test_get_model_returns_current_and_available_only() -> None:
    original = agent_routes.session_manager.config.model_name
    agent_routes.session_manager.config.model_name = "anthropic/claude-opus-4-6"
    try:
        response = client.get("/api/config/model")
    finally:
        agent_routes.session_manager.config.model_name = original

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"current", "available"}
    assert data["current"] == "anthropic/claude-opus-4-6"
    assert any(model["id"] == "moonshotai/Kimi-K2.6" for model in data["available"])


def test_set_model_rejects_custom_hf_id() -> None:
    response = client.post(
        "/api/config/model", json={"model": "moonshotai/Kimi-K2.6:fastest"}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown model: moonshotai/Kimi-K2.6:fastest"


def test_set_session_model_rejects_custom_hf_id() -> None:
    session_id = "test-session"
    sessions = agent_routes.session_manager.sessions
    sessions[session_id] = SimpleNamespace(
        user_id="dev",
        is_active=True,
        is_processing=False,
        created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
        session=DummySession("moonshotai/Kimi-K2.6"),
    )
    try:
        response = client.post(
            f"/api/session/{session_id}/model",
            json={"model": "moonshotai/Kimi-K2.6:fastest"},
        )
    finally:
        sessions.pop(session_id, None)

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown model: moonshotai/Kimi-K2.6:fastest"
