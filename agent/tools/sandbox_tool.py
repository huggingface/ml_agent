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
import threading
from typing import Any

from huggingface_hub import HfApi, SpaceHardware

from agent.core.session import Event
from agent.tools.sandbox_client import Sandbox


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


# ── Tool name mapping (short agent names → Sandbox client names) ──────


async def _ensure_sandbox(
    session: Any, hardware: str = "cpu-basic", **create_kwargs
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
        "log": _log,
        "cancel_event": cancel_flag,
        **create_kwargs,
    }
    if hardware != "cpu-basic":
        kwargs["sleep_time"] = 2700
    try:
        sb = await asyncio.to_thread(Sandbox.create, **kwargs)
    except Sandbox.Cancelled:
        return None, "Sandbox creation cancelled by user."
    finally:
        watcher_task.cancel()
    session.sandbox = sb

    # Set a descriptive title (template title is inherited on duplicate)
    from huggingface_hub import metadata_update

    await asyncio.to_thread(
        metadata_update,
        sb.space_id,
        {"title": "ml-agent sandbox"},
        repo_type="space",
        overwrite=True,
        token=token,
    )

    # Inject the OAuth token into the sandbox so Hub operations work inside it
    await asyncio.to_thread(api.add_space_secret, sb.space_id, "HF_TOKEN", token)

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
            "private": {
                "type": "boolean",
                "description": "If true, create a private Space",
            },
        },
    },
}


async def sandbox_create_handler(
    args: dict[str, Any], session: Any = None
) -> tuple[str, bool]:
    """Handle sandbox_create tool calls."""
    # If sandbox already exists, return its info
    if session and getattr(session, "sandbox", None):
        sb = session.sandbox
        return (
            f"Sandbox already active: {sb.space_id}\n"
            f"URL: {sb.url}\n"
            f"Use bash/read/write/edit to interact with it."
        ), True

    hardware = args.get("hardware", "cpu-basic")
    create_kwargs = {}
    if "private" in args:
        create_kwargs["private"] = args["private"]

    try:
        sb, error = await _ensure_sandbox(session, hardware=hardware, **create_kwargs)
    except Exception as e:
        return f"Failed to create sandbox: {e}", False

    if error:
        return error, False

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
