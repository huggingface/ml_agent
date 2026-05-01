import asyncio
import threading
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


def test_sandbox_client_retries_transient_runtime_404(monkeypatch):
    runtime_calls = 0

    class FakeResponse:
        status_code = 404

    class FakeRuntime404(Exception):
        response = FakeResponse()

        def __str__(self):
            return "404 Client Error: Repository Not Found"

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def duplicate_space(self, **kwargs):
            pass

        def add_space_secret(self, *args, **kwargs):
            pass

        def get_space_runtime(self, space_id):
            nonlocal runtime_calls
            runtime_calls += 1
            if runtime_calls == 1:
                raise FakeRuntime404()
            return SimpleNamespace(stage="RUNNING", hardware="cpu-basic")

    monkeypatch.setattr(sandbox_client, "HfApi", FakeApi)
    monkeypatch.setattr(sandbox_client.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        Sandbox,
        "_setup_server",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(Sandbox, "_wait_for_api", lambda self, *args, **kwargs: None)

    sandbox = Sandbox.create(owner="alice", token="hf-token", log=lambda msg: None)

    assert sandbox.space_id.startswith("alice/sandbox-")
    assert runtime_calls == 2


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


def test_sandbox_operation_waits_for_cpu_preload():
    calls: list[tuple[str, dict]] = []

    class FakeSandbox:
        def call_tool(self, name, args):
            calls.append((name, args))
            return SimpleNamespace(success=True, output="preloaded-ok", error="")

    async def run():
        session = SimpleNamespace(
            sandbox=None,
            sandbox_preload_error=None,
        )

        async def preload():
            await asyncio.sleep(0)
            session.sandbox = FakeSandbox()

        session.sandbox_preload_task = asyncio.create_task(preload())
        handler = sandbox_tool._make_tool_handler("bash")
        return await handler({"command": "echo ok"}, session=session)

    out, ok = asyncio.run(run())

    assert ok is True
    assert out == "preloaded-ok"
    assert calls == [("bash", {"command": "echo ok"})]


def test_default_sandbox_create_waits_for_cpu_preload():
    class FakeSandbox:
        space_id = "alice/sandbox-cpu"
        url = "https://huggingface.co/spaces/alice/sandbox-cpu"

    async def run():
        session = SimpleNamespace(
            sandbox=None,
            sandbox_preload_error=None,
        )

        async def preload():
            await asyncio.sleep(0)
            session.sandbox = FakeSandbox()
            session.sandbox_hardware = "cpu-basic"

        session.sandbox_preload_task = asyncio.create_task(preload())
        return await sandbox_tool.sandbox_create_handler({}, session=session)

    out, ok = asyncio.run(run())

    assert ok is True
    assert "Sandbox already active: alice/sandbox-cpu" in out
    assert "Hardware: cpu-basic" in out


def test_sandbox_create_replaces_auto_cpu_sandbox(monkeypatch):
    deleted: list[str] = []

    class FakeSession:
        def __init__(self):
            self.sandbox = SimpleNamespace(
                space_id="alice/sandbox-cpu",
                url="https://huggingface.co/spaces/alice/sandbox-cpu",
                _owns_space=True,
                delete=lambda: deleted.append("alice/sandbox-cpu"),
            )
            self.sandbox_hardware = "cpu-basic"
            self.sandbox_preload_task = None
            self.sandbox_preload_cancel_event = None

        async def send_event(self, event):
            pass

    gpu_sandbox = SimpleNamespace(
        space_id="alice/sandbox-gpu",
        url="https://huggingface.co/spaces/alice/sandbox-gpu",
        _owns_space=True,
    )

    async def fake_ensure_sandbox(session, hardware="cpu-basic", **kwargs):
        session.sandbox = gpu_sandbox
        session.sandbox_hardware = hardware
        return gpu_sandbox, None

    async def fake_record_sandbox_destroy(*args, **kwargs):
        pass

    monkeypatch.setattr(sandbox_tool, "_ensure_sandbox", fake_ensure_sandbox)
    monkeypatch.setattr(telemetry, "record_sandbox_destroy", fake_record_sandbox_destroy)

    session = FakeSession()
    out, ok = asyncio.run(
        sandbox_tool.sandbox_create_handler(
            {"hardware": "a100-large"},
            session=session,
        )
    )

    assert ok is True
    assert deleted == ["alice/sandbox-cpu"]
    assert session.sandbox is gpu_sandbox
    assert session.sandbox_hardware == "a100-large"
    assert "Hardware: a100-large" in out


def test_teardown_cancels_preload_and_deletes_owned_sandbox(monkeypatch):
    deleted: list[str] = []

    async def fake_record_sandbox_destroy(*args, **kwargs):
        pass

    monkeypatch.setattr(telemetry, "record_sandbox_destroy", fake_record_sandbox_destroy)

    async def run():
        cancel_event = threading.Event()

        async def preload():
            await asyncio.sleep(0)

        session = SimpleNamespace(
            sandbox=SimpleNamespace(
                space_id="alice/sandbox-12345678",
                _owns_space=True,
                delete=lambda: deleted.append("alice/sandbox-12345678"),
            ),
            sandbox_hardware="cpu-basic",
            sandbox_preload_task=asyncio.create_task(preload()),
            sandbox_preload_cancel_event=cancel_event,
        )

        await sandbox_tool.teardown_session_sandbox(session)
        return session, cancel_event

    session, cancel_event = asyncio.run(run())

    assert cancel_event.is_set()
    assert deleted == ["alice/sandbox-12345678"]
    assert session.sandbox is None
    assert session.sandbox_hardware is None
