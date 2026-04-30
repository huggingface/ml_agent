import asyncio
from types import SimpleNamespace

from agent.core import telemetry
from agent.tools import sandbox_client, sandbox_tool
from agent.tools.sandbox_client import Sandbox
from agent.tools.sandbox_tool import sandbox_create_handler


def test_sandbox_client_defaults_to_private_spaces(monkeypatch):
    duplicate_kwargs = {}

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def duplicate_space(self, **kwargs):
            duplicate_kwargs.update(kwargs)

        def add_space_secret(self, *args, **kwargs):
            pass

        def get_space_runtime(self, space_id):
            return SimpleNamespace(stage="RUNNING", hardware="cpu-basic")

    monkeypatch.setattr(sandbox_client, "HfApi", FakeApi)
    monkeypatch.setattr(
        Sandbox,
        "_setup_server",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(Sandbox, "_wait_for_api", lambda self, *args, **kwargs: None)

    Sandbox.create(owner="alice", token="hf-token", log=lambda msg: None)

    assert duplicate_kwargs["private"] is True


def test_sandbox_tool_forces_private_spaces(monkeypatch):
    captured_kwargs = {}

    async def fake_ensure_sandbox(
        session,
        hardware="cpu-basic",
        extra_secrets=None,
        **create_kwargs,
    ):
        captured_kwargs.update(create_kwargs)
        return (
            SimpleNamespace(
                space_id="alice/sandbox-12345678",
                url="https://huggingface.co/spaces/alice/sandbox-12345678",
            ),
            None,
        )

    monkeypatch.setattr(sandbox_tool, "_ensure_sandbox", fake_ensure_sandbox)

    out, ok = asyncio.run(
        sandbox_create_handler(
            {"private": False},
            session=SimpleNamespace(sandbox=None),
        )
    )

    assert ok is True
    assert "private" not in captured_kwargs
    assert "Visibility: private" in out


def test_ensure_sandbox_overrides_private_argument(monkeypatch):
    captured_kwargs = {}

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "alice"}

    class FakeSession:
        def __init__(self):
            self.hf_token = "hf-token"
            self.sandbox = None
            self.event_queue = SimpleNamespace(put_nowait=lambda event: None)
            self._cancelled = asyncio.Event()

        async def send_event(self, event):
            pass

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            space_id="alice/sandbox-12345678",
            url="https://huggingface.co/spaces/alice/sandbox-12345678",
        )

    async def fake_record_sandbox_create(*args, **kwargs):
        pass

    monkeypatch.setattr(sandbox_tool, "HfApi", FakeApi)
    monkeypatch.setattr(sandbox_tool, "_cleanup_user_orphan_sandboxes", lambda *args: 0)
    monkeypatch.setattr(Sandbox, "create", staticmethod(fake_create))
    monkeypatch.setattr(telemetry, "record_sandbox_create", fake_record_sandbox_create)
    monkeypatch.setattr("huggingface_hub.metadata_update", lambda *args, **kwargs: None)

    async def run():
        session = FakeSession()
        sb, error = await sandbox_tool._ensure_sandbox(session, private=False)
        return sb, error

    sb, error = asyncio.run(run())

    assert error is None
    assert sb is not None
    assert captured_kwargs["private"] is True
