"""
Sandbox tools — expose the Sandbox client as agent tools.

5 tools total:
  sandbox_create — create/replace sandbox for non-default hardware
  bash, read, write, edit — operations on the active sandbox

A cpu-basic sandbox is preloaded for each session. Operation tools wait for it
if startup is still in progress.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import weakref
from datetime import datetime, timedelta, timezone
from typing import Any

from huggingface_hub import HfApi, SpaceHardware

from agent.core.session import Event
from agent.tools.sandbox_client import Sandbox
from agent.tools.trackio_seed import ensure_trackio_dashboard

logger = logging.getLogger(__name__)

DEFAULT_CPU_SANDBOX_HARDWARE = "cpu-basic"

# Match the exact suffix pattern Sandbox.create produces: "sandbox-<8 hex>".
# Used to identify orphan sandboxes from prior sessions safely (won't match
# user-renamed lookalikes).
_SANDBOX_NAME_RE = re.compile(r"^sandbox-[a-f0-9]{8}$")

# How stale a sandbox must be before we treat it as definitely orphan.
# Anything more recent could be tied to a still-live session in another tab,
# so we leave it alone.
_ORPHAN_STALE_AFTER = timedelta(hours=1)

# HF Space duplication/build APIs can behave poorly when multiple private
# sandboxes are created concurrently for the same namespace. Keep session
# creation non-blocking, but serialize the actual Hub create path per owner.
_SANDBOX_CREATE_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _get_sandbox_create_lock(owner: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _SANDBOX_CREATE_LOCKS.setdefault(loop, {})
    lock = locks.get(owner)
    if lock is None:
        lock = asyncio.Lock()
        locks[owner] = lock
    return lock


def _looks_like_path(script: str) -> bool:
    """Return True if the script string looks like a file path (not inline code)."""
    return (
        isinstance(script, str)
        and script.strip() == script
        and not any(c in script for c in "\r\n\0")
        and (
            script.startswith("/")
            or script.startswith("./")
            or script.startswith("../")
        )
    )


async def resolve_sandbox_script(
    sandbox: Any, script: str
) -> tuple[str | None, str | None]:
    """Read a file from the sandbox if *script* looks like a path.

    Returns:
        (content, error) — content is the file text on success,
        error is a message on failure.  Both None means *script*
        is not a path (caller should use it as-is).
    """
    if not sandbox or not _looks_like_path(script):
        return None, None
    try:
        # Use the read endpoint instead of bash("cat ...") which truncates at 25KB.
        result = await asyncio.to_thread(sandbox.read, script, limit=100_000)
        if result.success and result.output:
            # Strip line number prefixes (read returns "N\tcontent" format)
            lines = []
            for line in result.output.split("\n"):
                parts = line.split("\t", 1)
                lines.append(parts[1] if len(parts) == 2 else line)
            return "\n".join(lines), None
        return None, f"Failed to read {script} from sandbox: {result.error}"
    except Exception as e:
        return None, f"Failed to read {script} from sandbox: {e}"


async def _seed_trackio_dashboard_safe(session: Any, space_id: str) -> None:
    """Idempotently seed *space_id* with trackio dashboard files using the
    session's HF token. Logs progress, swallows errors — a failed seed should
    not block sandbox creation."""
    if not session or not getattr(session, "hf_token", None):
        return
    loop = asyncio.get_running_loop()

    def _log(msg: str) -> None:
        loop.call_soon_threadsafe(
            session.event_queue.put_nowait,
            Event(event_type="tool_log", data={"tool": "sandbox_create", "log": msg}),
        )

    try:
        await asyncio.to_thread(
            ensure_trackio_dashboard, space_id, session.hf_token, _log
        )
    except Exception as e:
        _log(f"trackio dashboard seed failed: {e}")


async def _update_persisted_sandbox_fields(session: Any, **fields: Any) -> None:
    """Best-effort update of sandbox metadata on the durable session record."""
    store = getattr(session, "persistence_store", None)
    session_id = getattr(session, "session_id", None)
    if not (store and session_id and hasattr(store, "update_session_fields")):
        return
    try:
        await store.update_session_fields(session_id, **fields)
    except Exception as e:
        logger.warning("Failed to persist sandbox metadata for %s: %s", session_id, e)


async def _persist_active_sandbox(
    session: Any,
    sandbox: Sandbox,
    *,
    hardware: str,
) -> None:
    space_id = getattr(sandbox, "space_id", None)
    if not space_id:
        return
    owner = space_id.split("/", 1)[0] if "/" in space_id else None
    await _update_persisted_sandbox_fields(
        session,
        sandbox_space_id=space_id,
        sandbox_hardware=hardware,
        sandbox_owner=owner,
        sandbox_created_at=datetime.now(timezone.utc),
        sandbox_status="active",
    )


async def _clear_persisted_sandbox(session: Any) -> None:
    await _update_persisted_sandbox_fields(
        session,
        sandbox_space_id=None,
        sandbox_hardware=None,
        sandbox_owner=None,
        sandbox_created_at=None,
        sandbox_status="destroyed",
    )


# ── Tool name mapping (short agent names → Sandbox client names) ──────


def _cleanup_user_orphan_sandboxes(
    api: HfApi,
    owner: str,
    log: Any,
) -> int:
    """Delete stale ``sandbox-<8hex>`` Spaces in ``owner``'s account.

    "Stale" = not modified in the last hour. The naming pattern + staleness
    filter together make this safe:

    * Naming: only matches ``sandbox-<exactly 8 lowercase hex>``, the
      pattern Sandbox.create produces. Won't touch user-renamed Spaces.
    * Staleness: anything modified in the last hour might still be tied
      to a live session in another tab/replica, so we leave it alone.

    Runs blocking — call via ``asyncio.to_thread``. Best-effort: failures
    are logged but never raised, so a flaky HF API never blocks creation.
    """
    cutoff = datetime.now(timezone.utc) - _ORPHAN_STALE_AFTER
    deleted = 0
    try:
        spaces = list(api.list_spaces(author=owner, limit=200, full=True))
    except Exception as e:
        log(f"orphan sweep: list_spaces failed: {e}")
        return 0

    for space in spaces:
        space_name = space.id.rsplit("/", 1)[-1]
        if not _SANDBOX_NAME_RE.match(space_name):
            continue

        last_mod = getattr(space, "lastModified", None) or getattr(
            space, "last_modified", None
        )
        if isinstance(last_mod, str):
            try:
                last_mod = datetime.fromisoformat(last_mod.replace("Z", "+00:00"))
            except ValueError:
                last_mod = None
        if last_mod is None:
            log(f"orphan sweep: skipping {space.id}; missing lastModified")
            continue
        if last_mod and last_mod > cutoff:
            # Recent — could be a concurrent live session. Skip.
            continue

        try:
            api.delete_repo(repo_id=space.id, repo_type="space")
            deleted += 1
            log(f"orphan sweep: deleted {space.id}")
        except Exception as e:
            log(f"orphan sweep: failed to delete {space.id}: {e}")

    if deleted:
        log(f"orphan sweep: cleaned up {deleted} stale sandbox(es) before create")
    return deleted


async def _ensure_sandbox(
    session: Any,
    hardware: str = DEFAULT_CPU_SANDBOX_HARDWARE,
    extra_secrets: dict[str, str] | None = None,
    cancel_event: threading.Event | None = None,
    **create_kwargs,
) -> tuple[Sandbox | None, str | None]:
    """
    Ensure a sandbox exists on the session. Auto-creates with given hardware if needed.

    Returns:
        (sandbox, error_message) — one will be None.
    """
    if session and getattr(session, "sandbox", None):
        return session.sandbox, None

    if not session:
        return None, "No session available."

    token = session.hf_token
    if not token:
        return None, "No HF token available. Cannot create sandbox."

    api = HfApi(token=token)
    user_info = api.whoami()
    owner = user_info.get("name", user_info.get("user", ""))
    if not owner:
        return None, "Could not determine HF username from token."

    create_lock = _get_sandbox_create_lock(owner)
    if create_lock.locked():
        await session.send_event(
            Event(
                event_type="tool_log",
                data={
                    "tool": "sandbox",
                    "log": "Waiting for sandbox creation slot...",
                },
            )
        )

    async with create_lock:
        if getattr(session, "sandbox", None):
            return session.sandbox, None

        return await _create_sandbox_locked(
            session,
            api=api,
            owner=owner,
            hardware=hardware,
            extra_secrets=extra_secrets,
            cancel_event=cancel_event,
            **create_kwargs,
        )


async def _create_sandbox_locked(
    session: Any,
    *,
    api: HfApi,
    owner: str,
    hardware: str,
    extra_secrets: dict[str, str] | None = None,
    cancel_event: threading.Event | None = None,
    **create_kwargs,
) -> tuple[Sandbox | None, str | None]:
    """Create the Space while the per-owner sandbox creation lock is held."""
    token = session.hf_token
    await session.send_event(
        Event(
            event_type="tool_log",
            data={
                "tool": "sandbox",
                "log": f"Auto-creating sandbox for {owner} ({hardware})...",
            },
        )
    )

    # Thread-safe log callback: posts tool_log events from the worker thread
    loop = asyncio.get_running_loop()

    def _log(msg: str) -> None:
        loop.call_soon_threadsafe(
            session.event_queue.put_nowait,
            Event(event_type="tool_log", data={"tool": "sandbox", "log": msg}),
        )

    # Bridge asyncio cancel event to a threading.Event for the blocking create call.
    # We poll session._cancelled from the main loop in a background task and set
    # a threading.Event that Sandbox.create checks during its polling loops.
    cancel_flag = cancel_event or threading.Event()

    async def _watch_cancel():
        await session._cancelled.wait()
        cancel_flag.set()

    watcher_task = asyncio.create_task(_watch_cancel())

    secrets: dict[str, str] = {"HF_TOKEN": token}
    if extra_secrets:
        secrets.update({k: v for k, v in extra_secrets.items() if v})

    create_kwargs["private"] = True  # enforce: overrides any caller-supplied value
    kwargs = {
        "owner": owner,
        "hardware": hardware,
        "token": token,
        "secrets": secrets,
        "log": _log,
        "cancel_event": cancel_flag,
        **create_kwargs,
    }
    if hardware != DEFAULT_CPU_SANDBOX_HARDWARE:
        kwargs["sleep_time"] = 2700
    import time as _t

    _t_start = _t.monotonic()
    try:
        sb = await asyncio.to_thread(Sandbox.create, **kwargs)
    except Sandbox.Cancelled:
        return None, "Sandbox creation cancelled by user."
    finally:
        watcher_task.cancel()

    if cancel_flag.is_set():
        if getattr(sb, "_owns_space", False):
            try:
                await asyncio.to_thread(sb.delete)
            except Exception as e:
                logger.warning(
                    "Failed to delete cancelled sandbox %s: %s", sb.space_id, e
                )
        return None, "Sandbox creation cancelled by user."

    session.sandbox = sb
    session.sandbox_hardware = hardware
    session.sandbox_preload_error = None
    await _persist_active_sandbox(session, sb, hardware=hardware)

    # Telemetry: sandbox creation (infra consumption signal)
    from agent.core import telemetry

    await telemetry.record_sandbox_create(
        session,
        sb,
        hardware=hardware,
        create_latency_s=int(_t.monotonic() - _t_start),
    )

    # Set a descriptive title (template title is inherited on duplicate)
    from huggingface_hub import metadata_update

    await asyncio.to_thread(
        metadata_update,
        sb.space_id,
        {"title": "ml-intern sandbox"},
        repo_type="space",
        overwrite=True,
        token=token,
    )

    await session.send_event(
        Event(
            event_type="tool_log",
            data={"tool": "sandbox", "log": f"Sandbox ready: {sb.space_id} ({sb.url})"},
        )
    )

    return sb, None


def start_cpu_sandbox_preload(session: Any) -> asyncio.Task | None:
    """Start a background ``cpu-basic`` sandbox for this session."""
    if not session or getattr(session, "sandbox", None):
        return None

    existing_task = getattr(session, "sandbox_preload_task", None)
    if existing_task and not existing_task.done():
        return existing_task

    cancel_event = threading.Event()
    session.sandbox_preload_cancel_event = cancel_event
    session.sandbox_preload_error = None

    async def _preload() -> Sandbox | None:
        try:
            sb, error = await _ensure_sandbox(
                session,
                hardware=DEFAULT_CPU_SANDBOX_HARDWARE,
                cancel_event=cancel_event,
            )
            if error:
                session.sandbox_preload_error = error
                return None
            return sb
        except asyncio.CancelledError:
            cancel_event.set()
            session.sandbox_preload_error = "Sandbox creation cancelled by user."
            raise
        except Exception as e:
            session.sandbox_preload_error = f"Failed to create sandbox: {e}"
            logger.warning("CPU sandbox preload failed: %s", e)
            return None

    task = asyncio.create_task(_preload())
    session.sandbox_preload_task = task
    return task


async def cancel_sandbox_preload(session: Any) -> None:
    """Best-effort cancellation for an in-flight CPU sandbox preload."""
    cancel_event = getattr(session, "sandbox_preload_cancel_event", None)
    if cancel_event is not None:
        cancel_event.set()

    task = getattr(session, "sandbox_preload_task", None)
    if not task or task.done():
        return

    current_task = asyncio.current_task()
    if task is current_task:
        return

    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=30)
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out waiting for CPU sandbox preload cancellation; "
            "task is still live, cancelling asyncio wrapper"
        )
        task.cancel()
    except asyncio.CancelledError:
        raise
    except Exception:
        pass


async def get_active_or_preloaded_sandbox(
    session: Any,
) -> tuple[Sandbox | None, str | None]:
    """Return the active sandbox, waiting for the startup preload if needed."""
    if not session:
        return None, "No session available."
    if getattr(session, "sandbox", None):
        return session.sandbox, None

    task = getattr(session, "sandbox_preload_task", None)
    if task:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            session.sandbox_preload_error = f"Failed to create sandbox: {e}"

    if getattr(session, "sandbox", None):
        return session.sandbox, None

    preload_error = getattr(session, "sandbox_preload_error", None)
    if preload_error:
        return None, preload_error

    return None, "Sandbox is still starting. Please retry shortly."


async def teardown_session_sandbox(session: Any) -> None:
    """Cancel sandbox preload and delete the active owned sandbox, if present."""
    if not session:
        return

    await cancel_sandbox_preload(session)

    sandbox = getattr(session, "sandbox", None)
    session.sandbox = None
    session.sandbox_hardware = None

    if not sandbox:
        return

    try:
        if not getattr(sandbox, "_owns_space", False):
            return

        space_id = getattr(sandbox, "space_id", None)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                logger.info(
                    "Deleting sandbox %s (attempt %s/3)...",
                    space_id,
                    attempt + 1,
                )
                await asyncio.to_thread(sandbox.delete)
                from agent.core import telemetry

                await telemetry.record_sandbox_destroy(session, sandbox)
                return
            except Exception as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        logger.error(
            "Failed to delete sandbox %s after 3 attempts: %s. "
            "Orphan — sweep script will pick it up.",
            space_id,
            last_err,
        )
    finally:
        await _clear_persisted_sandbox(session)


# ── sandbox_create tool ──────────────────────────────────────────────

SANDBOX_CREATE_TOOL_SPEC = {
    "name": "sandbox_create",
    "description": (
        "Create or replace the session sandbox when non-default hardware is needed.\n\n"
        "A private cpu-basic sandbox is already started automatically for each session. "
        "For normal CPU code execution, call bash/read/write/edit directly; do NOT call sandbox_create first.\n\n"
        "Use sandbox_create when: you need GPU hardware, cpu-upgrade, or Trackio secrets before running code. "
        "The active sandbox persists across tool calls within the session. pip install works out of the box. "
        "Sandboxes are always created as private HF Spaces.\n\n"
        "For ML code that uses CUDA, bf16, or model loading: use GPU hardware (t4-small minimum). "
        "CPU sandboxes cannot run GPU code paths — your test will not catch GPU-related errors.\n\n"
        "Before choosing hardware, estimate your VRAM needs (models you run, training data size). Rule of thumb: bf16/fp16 ≈ 2 bytes/param, "
        "fp32 ≈ 4 bytes/param, plus ~20% overhead for optimizer states during training.\n"
        "Common picks: t4-small (16GB VRAM, fits ≤1-3B), a10g-small (24GB, ≤7B), a100-large (80GB, ≤30B). "
        "If the model won't fit, pick larger hardware upfront — OOM on a sandbox wastes time.\n\n"
        "If you intend to run a training script in this sandbox that uses report_to='trackio', "
        "pass `trackio_space_id` (e.g. '<username>/mlintern-<8char>') and `trackio_project` so they "
        "are set as TRACKIO_SPACE_ID/TRACKIO_PROJECT secrets in the sandbox and the UI can embed the live dashboard.\n\n"
        "Hardware: " + ", ".join([e.value for e in SpaceHardware]) + ".\n"
    ),
    "parameters": {
        "type": "object",
        "required": [],
        "additionalProperties": False,
        "properties": {
            "hardware": {
                "type": "string",
                "enum": [e.value for e in SpaceHardware],
                "description": (
                    "Hardware tier for the sandbox. Omit for the existing auto-started "
                    "cpu-basic sandbox; choose GPU/cpu-upgrade only when needed."
                ),
            },
            "trackio_space_id": {
                "type": "string",
                "description": (
                    "Optional. The HF Space hosting the trackio dashboard for runs in this sandbox "
                    "(e.g. '<username>/mlintern-<8char>', under YOUR HF namespace). Injected as "
                    "TRACKIO_SPACE_ID secret and surfaced to the UI. The Space is auto-created and "
                    "seeded with the trackio dashboard — DO NOT pre-create it via hf_repo_git, "
                    "that produces an empty Space that breaks the embed."
                ),
            },
            "trackio_project": {
                "type": "string",
                "description": (
                    "Optional. The trackio project name. Injected as TRACKIO_PROJECT secret and "
                    "used by the UI to filter the embedded dashboard to this project."
                ),
            },
        },
    },
}


async def sandbox_create_handler(
    args: dict[str, Any], session: Any = None, tool_call_id: str | None = None
) -> tuple[str, bool]:
    """Handle sandbox_create tool calls."""
    hardware = args.get("hardware", DEFAULT_CPU_SANDBOX_HARDWARE)
    trackio_space_id = args.get("trackio_space_id") or None
    trackio_project = args.get("trackio_project") or None

    async def _emit_trackio_state(sb: Sandbox) -> None:
        """Tell the frontend which trackio dashboard to embed for this sandbox."""
        if not (session and tool_call_id and trackio_space_id):
            return
        data: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "tool": "sandbox_create",
            "state": "running",
            "trackioSpaceId": trackio_space_id,
        }
        if trackio_project:
            data["trackioProject"] = trackio_project
        await session.send_event(Event(event_type="tool_state_change", data=data))

    preload_task = getattr(session, "sandbox_preload_task", None)
    if (
        session
        and not getattr(session, "sandbox", None)
        and preload_task
        and not preload_task.done()
        and hardware == DEFAULT_CPU_SANDBOX_HARDWARE
    ):
        sb, error = await get_active_or_preloaded_sandbox(session)
        if error:
            return error, False
        if sb:
            await _emit_trackio_state(sb)
            return (
                f"Sandbox already active: {sb.space_id}\n"
                f"URL: {sb.url}\n"
                f"Hardware: {DEFAULT_CPU_SANDBOX_HARDWARE}\n"
                f"Use bash/read/write/edit to interact with it."
            ), True

    if (
        session
        and not getattr(session, "sandbox", None)
        and preload_task
        and not preload_task.done()
        and hardware != DEFAULT_CPU_SANDBOX_HARDWARE
    ):
        await cancel_sandbox_preload(session)

    # If sandbox already exists, return its info or replace the auto CPU sandbox
    if session and getattr(session, "sandbox", None):
        sb = session.sandbox
        active_hardware = getattr(session, "sandbox_hardware", None)
        if active_hardware == hardware:
            await _emit_trackio_state(sb)
            return (
                f"Sandbox already active: {sb.space_id}\n"
                f"URL: {sb.url}\n"
                f"Hardware: {active_hardware}\n"
                f"Use bash/read/write/edit to interact with it."
            ), True

        requested_hardware = args.get("hardware")
        lockout_note = ""
        if (
            active_hardware == DEFAULT_CPU_SANDBOX_HARDWARE
            and hardware != DEFAULT_CPU_SANDBOX_HARDWARE
        ):
            await teardown_session_sandbox(session)
        elif requested_hardware:
            lockout_note = (
                f"\nRequested hardware: {requested_hardware}\n"
                "Hardware cannot be changed by calling sandbox_create again. "
                "Delete the existing sandbox first if you need a different tier."
            )
            await _emit_trackio_state(sb)
            return (
                f"Sandbox already active: {sb.space_id}\n"
                f"URL: {sb.url}\n"
                f"{lockout_note}\n"
                f"Use bash/read/write/edit to interact with it."
            ), True
        else:
            await _emit_trackio_state(sb)
            return (
                f"Sandbox already active: {sb.space_id}\n"
                f"URL: {sb.url}\n"
                f"Hardware: {active_hardware or 'unknown'}\n"
                f"Use bash/read/write/edit to interact with it."
            ), True

    create_kwargs: dict[str, Any] = {}

    extra_secrets: dict[str, str] = {}
    if trackio_space_id:
        extra_secrets["TRACKIO_SPACE_ID"] = trackio_space_id
        await _seed_trackio_dashboard_safe(session, trackio_space_id)
    if trackio_project:
        extra_secrets["TRACKIO_PROJECT"] = trackio_project

    try:
        sb, error = await _ensure_sandbox(
            session,
            hardware=hardware,
            extra_secrets=extra_secrets or None,
            **create_kwargs,
        )
    except Exception as e:
        return f"Failed to create sandbox: {e}", False

    if error:
        return error, False

    await _emit_trackio_state(sb)

    return (
        f"Sandbox created: {sb.space_id}\n"
        f"URL: {sb.url}\n"
        f"Hardware: {hardware}\n"
        "Visibility: private\n"
        f"Use bash/read/write/edit to interact with it."
    ), True


def _make_tool_handler(sandbox_tool_name: str):
    """Factory: create a handler for a sandbox operation tool."""

    async def handler(args: dict[str, Any], session: Any = None) -> tuple[str, bool]:
        sb, error = await get_active_or_preloaded_sandbox(session)
        if error:
            return error, False
        if not sb:
            return "Sandbox is still starting. Please retry shortly.", False

        try:
            result = await asyncio.to_thread(sb.call_tool, sandbox_tool_name, args)
            if result.success:
                output = result.output or "(no output)"
                return output, True
            else:
                error_msg = result.error or "Unknown error"
                output = result.output
                if output:
                    return f"{output}\n\nERROR: {error_msg}", False
                return f"ERROR: {error_msg}", False
        except Exception as e:
            return f"Sandbox operation failed: {e}", False

    return handler


def get_sandbox_tools():
    """Return all 5 sandbox ToolSpecs (sandbox_create + 4 operation tools)."""
    from agent.core.tools import ToolSpec

    tools = []

    # sandbox_create (for GPU or other non-default hardware)
    tools.append(
        ToolSpec(
            name=SANDBOX_CREATE_TOOL_SPEC["name"],
            description=SANDBOX_CREATE_TOOL_SPEC["description"],
            parameters=SANDBOX_CREATE_TOOL_SPEC["parameters"],
            handler=sandbox_create_handler,
        )
    )

    # Operation tools (auto-execute, no approval needed)
    for name in Sandbox.TOOLS.keys():
        spec = Sandbox.TOOLS[name]
        description = (
            "Uses the session's active sandbox. A private cpu-basic sandbox is "
            "started automatically for normal CPU work; call sandbox_create only "
            "for GPU or other non-default hardware.\n\n" + spec["description"]
        )
        tools.append(
            ToolSpec(
                name=name,
                description=description,
                parameters=spec["parameters"],
                handler=_make_tool_handler(name),
            )
        )

    return tools
