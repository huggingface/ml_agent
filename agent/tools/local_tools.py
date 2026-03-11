"""
Local tool implementations — bash/read/write/edit running on the user's machine.

Drop-in replacement for sandbox tools when running in CLI (local) mode.
Same tool specs (names, parameters) but handlers execute locally via
subprocess/pathlib instead of going through a remote sandbox.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent.tools.sandbox_client import Sandbox

MAX_OUTPUT_CHARS = 30_000
MAX_LINE_LENGTH = 2000
DEFAULT_READ_LINES = 2000
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600


# ── Handlers ────────────────────────────────────────────────────────────

async def _bash_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    command = args.get("command", "")
    if not command:
        return "No command provided.", False
    work_dir = args.get("work_dir", ".")
    timeout = min(args.get("timeout") or DEFAULT_TIMEOUT, MAX_TIMEOUT)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"
        if not output.strip():
            output = "(no output)"
        return output, result.returncode == 0
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s.", False
    except Exception as e:
        return f"bash error: {e}", False


async def _read_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    file_path = args.get("path", "")
    if not file_path:
        return "No path provided.", False
    p = Path(file_path)
    if not p.exists():
        return f"File not found: {file_path}", False
    if p.is_dir():
        return "Cannot read a directory. Use bash with 'ls' instead.", False
    try:
        lines = p.read_text().splitlines()
    except Exception as e:
        return f"read error: {e}", False

    offset = max((args.get("offset") or 1), 1)
    limit = args.get("limit") or DEFAULT_READ_LINES

    selected = lines[offset - 1 : offset - 1 + limit]
    numbered = []
    for i, line in enumerate(selected, start=offset):
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "..."
        numbered.append(f"{i:>6}\t{line}")
    return "\n".join(numbered), True


async def _write_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    file_path = args.get("path", "")
    content = args.get("content", "")
    if not file_path:
        return "No path provided.", False
    p = Path(file_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {file_path}", True
    except Exception as e:
        return f"write error: {e}", False


async def _edit_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    file_path = args.get("path", "")
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")
    replace_all = args.get("replace_all", False)

    if not file_path:
        return "No path provided.", False
    if old_str == new_str:
        return "old_str and new_str must differ.", False

    p = Path(file_path)
    if not p.exists():
        return f"File not found: {file_path}", False

    try:
        text = p.read_text()
    except Exception as e:
        return f"edit read error: {e}", False

    count = text.count(old_str)
    if count == 0:
        return "old_str not found in file.", False
    if count > 1 and not replace_all:
        return (
            f"old_str appears {count} times. Use replace_all=true to replace all, "
            "or provide a more specific old_str."
        ), False

    new_text = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
    try:
        p.write_text(new_text)
    except Exception as e:
        return f"edit write error: {e}", False

    replacements = count if replace_all else 1
    return f"Edited {file_path} ({replacements} replacement{'s' if replacements > 1 else ''})", True


# ── Public API ──────────────────────────────────────────────────────────

_HANDLERS = {
    "bash": _bash_handler,
    "read": _read_handler,
    "write": _write_handler,
    "edit": _edit_handler,
}


def get_local_tools():
    """Return local ToolSpecs for bash/read/write/edit (no sandbox_create)."""
    from agent.core.tools import ToolSpec

    tools = []
    for name, spec in Sandbox.TOOLS.items():
        handler = _HANDLERS.get(name)
        if handler is None:
            continue
        tools.append(
            ToolSpec(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
                handler=handler,
            )
        )
    return tools
