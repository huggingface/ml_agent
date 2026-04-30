"""
Sandbox tools — expose the Sandbox client as agent tools.

5 tools total:
  sandbox_create — explicit sandbox creation (requires approval)
  bash, read, write, edit — operations on the sandbox

If any operation tool is called without an active sandbox,
a cpu-basic sandbox is auto-created (no approval needed).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from huggingface_hub import HfApi, SpaceHardware

from agent.core.session import Event
from agent.tools.sandbox_client import Sandbox
from agent.tools.trackio_seed import ensure_trackio_dashboard

logger = logging.getLogger(__name__)

# Match the exact suffix pattern Sandbox.create produces: "sandbox-<8 hex>".
# Used to identify orphan sandboxes from prior sessions safely (won't match
# user-renamed lookalikes).
_SANDBOX_NAME_RE = re.compile(r"^sandbox-[a-f0-9]{8}$")

# How stale a sandbox must be before we treat it as definitely orphan.
# Anything more recent could be tied to a still-live session in another tab,
# so we leave it alone.
_ORPHAN_STALE_AFTER = timedelta(hours=1)


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
        spaces = list(api.list_spaces(author=owner, limit=200))
    except Exception as e:
        log(f"orphan sweep: list_spaces failed: {e}")
        return 0

    for space in spaces:
        space_name = space.id.rsplit("/", 1)[-1]
        if not _SANDBOX_NAME_RE.match(space_name):
            continue

        last_mod = getattr(space, "lastModified", None) or getattr(space, "last_modified", None)
        if isinstance(last_mod, str):
            try:
                last_mod = datetime.fromisoformat(last_mod.replace("Z", "+00:00"))
            except ValueError:
                last_mod = None
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
    hardware: str = "cpu-basic",
    extra_variables: dict[str, str] | None = None,
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

    # Before we create a new sandbox, sweep this user's stale sandboxes from
    # prior sessions. ``_cleanup_sandbox`` in session_manager fires only on
    # clean session exit; pod kills, WebSocket drops, etc. leave orphans
    # behind, and they accumulate on every new session forever (observed
    # 2310 leaked across the Hub on 2026-04-27). Doing the cleanup here at
    # session start = self-healing, no separate cron needed.
    #
    # The 1h staleness filter is the safety: a sandbox modified in the last
    # hour might still be tied to a live session in another tab, so we skip.
    # Anything older has no realistic chance of being active given typical
    # session lengths.
    try:
        await asyncio.to_thread(_cleanup_user_orphan_sandboxes, api, owner, _log)
    except Exception as e:
        # Cleanup is best-effort — never block sandbox_create on it.
        _log(f"orphan sandbox sweep failed (non-fatal): {e}")

    # Bridge asyncio cancel event to a threading.Event for the blocking create call.
    # We poll session._cancelled from the main loop in a background task and set
    # a threading.Event that Sandbox.create checks during its polling loops.
    cancel_flag = threading.Event()

    async def _watch_cancel():
        await session._cancelled.wait()
        cancel_flag.set()

    watcher_task = asyncio.create_task(_watch_cancel())

    kwargs = {
        "owner": owner,
        "hardware": hardware,
        "token": token,
        "variables": {k: v for k, v in (extra_variables or {}).items() if v},
        "log": _log,
        "cancel_event": cancel_flag,
        **create_kwargs,
    }
    if hardware != "cpu-basic":
        kwargs["sleep_time"] = 2700
    import time as _t
    _t_start = _t.monotonic()
    try:
        sb = await asyncio.to_thread(Sandbox.create, **kwargs)
    except Sandbox.Cancelled:
        return None, "Sandbox creation cancelled by user."
    finally:
        watcher_task.cancel()
    session.sandbox = sb

    # Telemetry: sandbox creation (infra consumption signal)
    from agent.core import telemetry
    await telemetry.record_sandbox_create(
        session, sb, hardware=hardware,
        create_latency_s=int(_t.monotonic() - _t_start),
    )

    # Set a descriptive title for the freshly created Space.
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


# ── sandbox_create tool ──────────────────────────────────────────────

SANDBOX_CREATE_TOOL_SPEC = {
    "name": "sandbox_create",
    "description": (
        "Create a persistent remote Linux environment for developing and testing scripts.\n\n"
        "Workflow: sandbox_create → write script → pip install → test with small run → fix errors → hf_jobs at scale.\n"
        "The sandbox persists across tool calls within the session. pip install works out of the box.\n\n"
        "Use this when: you need to develop, test, and iterate on scripts before launching via hf_jobs. "
        "Especially for training scripts where you need to verify imports, test on a small subset, and fix errors interactively.\n\n"
        "Skip this when: the task is a simple one-shot operation (status check, resource search, quick data query), "
        "or the script is copied from a verified working example with minimal changes.\n\n"
        "For ML code that uses CUDA, bf16, or model loading: use GPU hardware (t4-small minimum). "
        "CPU sandboxes cannot run GPU code paths — your test will not catch GPU-related errors.\n\n"
        "Before choosing hardware, estimate your VRAM needs (models you run, training data size). Rule of thumb: bf16/fp16 ≈ 2 bytes/param, "
        "fp32 ≈ 4 bytes/param, plus ~20% overhead for optimizer states during training.\n"
        "Common picks: t4-small (16GB VRAM, fits ≤1-3B), a10g-small (24GB, ≤7B), a100-large (80GB, ≤30B). "
        "If the model won't fit, pick larger hardware upfront — OOM on a sandbox wastes time.\n\n"
        "If you intend to run a training script in this sandbox that uses report_to='trackio', "
        "pass `trackio_space_id` (e.g. '<username>/mlintern-<8char>') and `trackio_project` so they "
        "are set as TRACKIO_SPACE_ID/TRACKIO_PROJECT variables in the sandbox and the UI can embed the live dashboard.\n\n"
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
                "description": "Hardware tier for the sandbox (default: cpu-basic)",
            },
            "trackio_space_id": {
                "type": "string",
                "description": (
                    "Optional. The HF Space hosting the trackio dashboard for runs in this sandbox "
                    "(e.g. '<username>/mlintern-<8char>', under YOUR HF namespace). Injected as the "
                    "TRACKIO_SPACE_ID variable and surfaced to the UI. The Space is auto-created and "
                    "seeded with the trackio dashboard — DO NOT pre-create it via hf_repo_git, "
                    "that produces an empty Space that breaks the embed."
                ),
            },
            "trackio_project": {
                "type": "string",
                "description": (
                    "Optional. The trackio project name. Injected as the TRACKIO_PROJECT variable and "
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
    hardware = args.get("hardware", "cpu-basic")
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

    # If sandbox already exists, return its info
    if session and getattr(session, "sandbox", None):
        sb = session.sandbox
        requested_hardware = args.get("hardware")
        lockout_note = ""
        if requested_hardware:
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

    create_kwargs: dict[str, Any] = {"private": True}

    extra_variables: dict[str, str] = {}
    if trackio_space_id:
        extra_variables["TRACKIO_SPACE_ID"] = trackio_space_id
        await _seed_trackio_dashboard_safe(session, trackio_space_id)
    if trackio_project:
        extra_variables["TRACKIO_PROJECT"] = trackio_project

    try:
        sb, error = await _ensure_sandbox(
            session,
            hardware=hardware,
            extra_variables=extra_variables or None,
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
        f"Use bash/read/write/edit to interact with it."
    ), True


def _make_tool_handler(sandbox_tool_name: str):
    """Factory: create a handler for a sandbox operation tool."""

    async def handler(args: dict[str, Any], session: Any = None) -> tuple[str, bool]:
        # Require sandbox to exist — user must approve sandbox_create first
        if not session or not getattr(session, "sandbox", None):
            return "No sandbox running. Call sandbox_create first to start one.", False

        sb = session.sandbox

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

    # sandbox_create (explicit creation, requires approval)
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
        tools.append(
            ToolSpec(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
                handler=_make_tool_handler(name),
            )
        )

    return tools
