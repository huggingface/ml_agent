import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from agent.core import telemetry
from agent.tools import sandbox_client, sandbox_tool
from agent.tools.sandbox_client import Sandbox
from agent.tools.sandbox_tool import sandbox_create_handler


def test_sandbox_client_defaults_to_private_spaces(monkeypatch):
    duplicate_kwargs = {}
    requested_hardware = []

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def duplicate_space(self, **kwargs):
            duplicate_kwargs.update(kwargs)

        def request_space_hardware(self, space_id, hardware, sleep_time=None):
            requested_hardware.append((space_id, hardware, sleep_time))
            return SimpleNamespace(stage="BUILDING", hardware=None)

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
    assert duplicate_kwargs["hardware"] == "cpu-basic"
    assert requested_hardware == []


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

        def request_space_hardware(self, space_id, hardware, sleep_time=None):
            return SimpleNamespace(stage="BUILDING", hardware=None)

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


def test_sandbox_client_retries_transient_hardware_401(monkeypatch):
    hardware_calls = 0
    logs: list[str] = []

    class FakeResponse:
        status_code = 401

    class FakeHardware401(Exception):
        response = FakeResponse()

        def __str__(self):
            return "401 Client Error: Repository Not Found"

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def duplicate_space(self, **kwargs):
            pass

        def request_space_hardware(self, space_id, hardware, sleep_time=None):
            nonlocal hardware_calls
            hardware_calls += 1
            if hardware_calls == 1:
                raise FakeHardware401()
            return SimpleNamespace(stage="BUILDING", hardware=None)

        def add_space_secret(self, *args, **kwargs):
            pass

        def get_space_runtime(self, space_id):
            return SimpleNamespace(stage="RUNNING", hardware="t4-small")

    monkeypatch.setattr(sandbox_client, "HfApi", FakeApi)
    monkeypatch.setattr(sandbox_client.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        Sandbox,
        "_setup_server",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(Sandbox, "_wait_for_api", lambda self, *args, **kwargs: None)

    sandbox = Sandbox.create(
        owner="alice",
        token="hf-token",
        hardware="t4-small",
        log=logs.append,
    )

    assert sandbox.space_id.startswith("alice/sandbox-")
    assert hardware_calls == 2
    assert any("Hardware request not accepted yet (HTTP 401)" in log for log in logs)


def test_sandbox_hardware_retry_reraises_after_timeout(monkeypatch):
    calls = 0
    logs: list[str] = []
    sleeps: list[float] = []

    class FakeResponse:
        status_code = 401

    class FakeHardware401(Exception):
        response = FakeResponse()

        def __str__(self):
            return "401 Client Error: Repository Not Found"

    first_error = FakeHardware401("first")
    timeout_error = FakeHardware401("timeout")

    class FakeApi:
        def request_space_hardware(self, space_id, hardware, sleep_time=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise first_error
            raise timeout_error

    timestamps = iter([100.0, 100.0, 161.0])

    monkeypatch.setattr(sandbox_client.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(sandbox_client.time, "sleep", sleeps.append)

    with pytest.raises(FakeHardware401) as excinfo:
        sandbox_client._request_space_hardware_with_retry(
            FakeApi(),
            "alice/sandbox-12345678",
            hardware="cpu-basic",
            sleep_time=None,
            log=logs.append,
            check_cancel=lambda: None,
        )

    assert excinfo.value is timeout_error
    assert calls == 2
    assert sleeps == [sandbox_client.WAIT_INTERVAL]
    assert len(logs) == 1


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


def test_orphan_sweep_preserves_spaces_without_last_modified():
    deleted: list[str] = []
    logs: list[str] = []

    class FakeApi:
        def list_spaces(self, **kwargs):
            assert kwargs["full"] is True
            return [SimpleNamespace(id="alice/sandbox-12345678")]

        def delete_repo(self, repo_id, repo_type):
            deleted.append(repo_id)

    count = sandbox_tool._cleanup_user_orphan_sandboxes(
        FakeApi(),
        "alice",
        logs.append,
    )

    assert count == 0
    assert deleted == []
    assert logs == ["orphan sweep: skipping alice/sandbox-12345678; missing lastModified"]


def test_ensure_sandbox_overrides_private_argument(monkeypatch):
    captured_kwargs = {}
    persisted: list[dict] = []

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "alice"}

    class FakeSession:
        def __init__(self):
            self.session_id = "s1"
            self.hf_token = "hf-token"
            self.sandbox = None
            self.event_queue = SimpleNamespace(put_nowait=lambda event: None)
            self._cancelled = asyncio.Event()
            self.persistence_store = SimpleNamespace(
                update_session_fields=lambda session_id, **fields: _record_metadata(
                    session_id, fields
                )
            )

        async def send_event(self, event):
            pass

    async def _record_metadata(session_id, fields):
        persisted.append({"session_id": session_id, **fields})

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
    assert persisted[-1]["session_id"] == "s1"
    assert persisted[-1]["sandbox_space_id"] == "alice/sandbox-12345678"
    assert persisted[-1]["sandbox_hardware"] == "cpu-basic"
    assert persisted[-1]["sandbox_owner"] == "alice"
    assert persisted[-1]["sandbox_status"] == "active"


def test_sandbox_creation_is_serialized_per_owner(monkeypatch):
    active_creates = 0
    max_active_creates = 0
    active_lock = threading.Lock()

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
        nonlocal active_creates, max_active_creates
        with active_lock:
            active_creates += 1
            max_active_creates = max(max_active_creates, active_creates)
        time.sleep(0.02)
        with active_lock:
            active_creates -= 1
        return SimpleNamespace(
            space_id=f"alice/sandbox-{kwargs['hardware']}",
            url="https://huggingface.co/spaces/alice/sandbox",
        )

    async def fake_record_sandbox_create(*args, **kwargs):
        pass

    monkeypatch.setattr(sandbox_tool, "HfApi", FakeApi)
    monkeypatch.setattr(sandbox_tool, "_cleanup_user_orphan_sandboxes", lambda *args: 0)
    monkeypatch.setattr(Sandbox, "create", staticmethod(fake_create))
    monkeypatch.setattr(telemetry, "record_sandbox_create", fake_record_sandbox_create)
    monkeypatch.setattr("huggingface_hub.metadata_update", lambda *args, **kwargs: None)

    async def run():
        await asyncio.gather(
            sandbox_tool._ensure_sandbox(FakeSession()),
            sandbox_tool._ensure_sandbox(FakeSession()),
        )

    asyncio.run(run())

    assert max_active_creates == 1


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
    persisted: list[dict] = []

    async def fake_record_sandbox_destroy(*args, **kwargs):
        pass

    monkeypatch.setattr(telemetry, "record_sandbox_destroy", fake_record_sandbox_destroy)

    async def run():
        cancel_event = threading.Event()

        async def preload():
            await asyncio.sleep(0)

        session = SimpleNamespace(
            session_id="s1",
            sandbox=SimpleNamespace(
                space_id="alice/sandbox-12345678",
                _owns_space=True,
                delete=lambda: deleted.append("alice/sandbox-12345678"),
            ),
            sandbox_hardware="cpu-basic",
            sandbox_preload_task=asyncio.create_task(preload()),
            sandbox_preload_cancel_event=cancel_event,
            persistence_store=SimpleNamespace(
                update_session_fields=lambda session_id, **fields: _record_metadata(
                    session_id, fields
                )
            ),
        )

        await sandbox_tool.teardown_session_sandbox(session)
        return session, cancel_event

    async def _record_metadata(session_id, fields):
        persisted.append({"session_id": session_id, **fields})

    session, cancel_event = asyncio.run(run())

    assert cancel_event.is_set()
    assert deleted == ["alice/sandbox-12345678"]
    assert session.sandbox is None
    assert session.sandbox_hardware is None
    assert persisted[-1]["session_id"] == "s1"
    assert persisted[-1]["sandbox_space_id"] is None
    assert persisted[-1]["sandbox_status"] == "destroyed"


def test_cancel_sandbox_preload_cancels_task_after_timeout(monkeypatch):
    async def run():
        async def fake_wait_for(awaitable, timeout):
            await asyncio.sleep(0)
            raise asyncio.TimeoutError

        monkeypatch.setattr(sandbox_tool.asyncio, "wait_for", fake_wait_for)

        cancel_event = threading.Event()
        blocker = asyncio.Event()

        async def preload():
            await blocker.wait()

        task = asyncio.create_task(preload())
        session = SimpleNamespace(
            sandbox_preload_task=task,
            sandbox_preload_cancel_event=cancel_event,
        )

        await sandbox_tool.cancel_sandbox_preload(session)
        await asyncio.sleep(0)

        return task.cancelled(), cancel_event.is_set()

    task_cancelled, cancel_event_set = asyncio.run(run())

    assert task_cancelled is True
    assert cancel_event_set is True
