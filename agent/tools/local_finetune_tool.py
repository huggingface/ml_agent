"""
Local fine-tuning tool — runs MLX training scripts on Mac via uv subprocess.

Pattern mirrors hf_jobs: agent writes a Python script, this tool executes it
locally with streaming output and user approval before running.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import uuid
from typing import Any

from agent.core.session import Event

LOCAL_FINETUNE_TOOL_SPEC = {
    "name": "local_finetune",
    "description": (
        "Run a fine-tuning script locally on Mac using MLX (Apple Silicon). "
        "Pass a Python training script — it will be executed via `uv run` with the "
        "specified dependencies. Streams stdout/stderr in real-time. "
        "Always requires user approval before running. "
        "Use `operation='ps'` to check status, `operation='cancel'` to stop a running job. "
        "Prefer mlx-community/ model variants (4-bit quantized) for memory efficiency. "
        "Always set `adapter_path` in train() or call `save_adapters()` — adapters are lost otherwise."
    ),
    "parameters": {
        "type": "object",
        "required": ["operation"],
        "additionalProperties": False,
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["run", "cancel", "ps"],
                "description": "run: execute the script; cancel: stop a running job; ps: show status",
            },
            "script": {
                "type": "string",
                "description": "Python training script (mlx-lm). Required for operation=run.",
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra pip packages beyond mlx-lm (e.g. ['datasets', 'huggingface_hub']).",
            },
            "output_dir": {
                "type": "string",
                "description": "Local directory to save adapters/fused model. Shown in approval UI.",
            },
            "timeout": {
                "type": "string",
                "description": "Max runtime before the process is killed. Examples: '2h', '30m', '1h30m'. Default: '2h'.",
            },
        },
    },
}


def _parse_timeout(timeout_str: str) -> float:
    """Convert '2h', '30m', '1h30m' to seconds. Returns 7200.0 on parse failure."""
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hHmMsS])", timeout_str):
        v = float(value)
        if unit.lower() == "h":
            total += v * 3600
        elif unit.lower() == "m":
            total += v * 60
        else:
            total += v
    return total if total > 0 else 7200.0


def _build_uv_command(script_path: str, deps: list[str]) -> list[str]:
    """Build: uv run [--with dep]... python <script_path>"""
    cmd = ["uv", "run"]
    for dep in deps:
        cmd += ["--with", dep]
    cmd += ["python", script_path]
    return cmd


async def _run_local(args: dict[str, Any], session: Any) -> tuple[str, bool]:
    script = args.get("script", "").strip()
    if not script:
        return "ERROR: 'script' is required for operation='run'.", False

    # Check if a job is already running
    existing = getattr(session, "_local_finetune_proc", None)
    if existing is not None and existing.returncode is None:
        return (
            "ERROR: A local fine-tune job is already running. "
            "Use operation='cancel' to stop it first.",
            False,
        )

    deps = ["mlx-lm"] + [d for d in (args.get("dependencies") or []) if d != "mlx-lm"]
    timeout_secs = _parse_timeout(args.get("timeout") or "2h")
    output_dir = args.get("output_dir", "")

    # Write script to a temp file
    tmp_path = os.path.join(tempfile.gettempdir(), f"mlx_finetune_{uuid.uuid4().hex[:8]}.py")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(script)

        cmd = _build_uv_command(tmp_path, deps)

        if session:
            await session.send_event(Event(
                event_type="tool_log",
                data={"tool": "local_finetune", "log": f"Starting: {' '.join(cmd)}"},
            ))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ},
        )

        if session:
            session._local_finetune_proc = proc

        # Stream output lines until done or timeout
        collected: list[str] = []
        try:
            async def _stream():
                assert proc.stdout is not None
                async for raw in proc.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    collected.append(line)
                    if session:
                        await session.send_event(Event(
                            event_type="tool_log",
                            data={"tool": "local_finetune", "log": line},
                        ))

            await asyncio.wait_for(_stream(), timeout=timeout_secs)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            timeout_msg = f"Training timed out after {args.get('timeout', '2h')}."
            if session:
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "local_finetune", "log": timeout_msg},
                ))
            return timeout_msg, False

        await proc.wait()
        rc = proc.returncode

        if rc == 0:
            summary = "Training completed successfully (exit 0)."
            if output_dir:
                summary += f" Adapters saved to: {output_dir}"
            if session:
                await session.send_event(Event(
                    event_type="tool_log",
                    data={"tool": "local_finetune", "log": summary},
                ))
            return summary, True
        else:
            tail = "\n".join(collected[-20:]) if collected else "(no output)"
            msg = f"Training failed (exit {rc}).\nLast output:\n{tail}"
            return msg, False

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _cancel_local(session: Any) -> tuple[str, bool]:
    proc = getattr(session, "_local_finetune_proc", None) if session else None
    if proc is None:
        return "No local fine-tune job has been started.", False
    if proc.returncode is not None:
        return f"Job already finished (exit {proc.returncode}).", False

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()

    return f"Local fine-tune job cancelled (pid {proc.pid}).", True


def _ps_local(session: Any) -> tuple[str, bool]:
    proc = getattr(session, "_local_finetune_proc", None) if session else None
    if proc is None:
        return "No local fine-tune job has been started this session.", True
    if proc.returncode is None:
        return f"Local fine-tune job running (pid {proc.pid}).", True
    return f"Last local fine-tune job finished (pid {proc.pid}, exit {proc.returncode}).", True


async def local_finetune_handler(
    args: dict[str, Any],
    session: Any = None,
    tool_call_id: str | None = None,
) -> tuple[str, bool]:
    operation = args.get("operation", "")
    if operation == "run":
        return await _run_local(args, session)
    if operation == "cancel":
        return await _cancel_local(session)
    if operation == "ps":
        return _ps_local(session)
    return f"Unknown operation: {operation!r}. Use 'run', 'cancel', or 'ps'.", False
