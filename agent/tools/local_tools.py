"""
Local tool implementations — bash/read/write/edit running on the user's machine.

Drop-in replacement for sandbox tools when running in CLI (local) mode.
Same tool specs (names, parameters) but handlers execute locally via
subprocess/pathlib instead of going through a remote sandbox.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent.tools.sandbox_client import Sandbox

MAX_OUTPUT_CHARS = 25_000
MAX_LINE_LENGTH = 2000
DEFAULT_READ_LINES = 2000
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 36000  # 10 hours — needed for long training runs (e.g. PostTrainBench)

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')


def _atomic_write(path: Path, content: str) -> None:
    """Write file atomically via temp file + os.replace().

    Ensures the file is never left in a partial/corrupted state — it's either
    the old content or the new content, never half-written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(tmp_path, str(path))
        tmp_path = None  # successfully replaced, nothing to clean up
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)


def _truncate_output(output: str, max_chars: int = MAX_OUTPUT_CHARS, head_ratio: float = 0.25) -> str:
    """Tail-biased truncation with temp file spillover for full output access."""
    if len(output) <= max_chars:
        return output
    # Write full output to temp file so LLM can read specific sections
    spill_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', prefix='bash_output_', delete=False) as f:
            f.write(output)
            spill_path = f.name
    except Exception:
        pass
    head_budget = int(max_chars * head_ratio)
    tail_budget = max_chars - head_budget
    head = output[:head_budget]
    tail = output[-tail_budget:]
    total = len(output)
    omitted = total - max_chars
    meta = f"\n\n... ({omitted:,} of {total:,} chars omitted, showing first {head_budget:,} + last {tail_budget:,}) ...\n"
    if spill_path:
        meta += f"Full output saved to {spill_path} — use the read tool with offset/limit to inspect specific sections.\n"
    return head + meta + tail


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
        output = _strip_ansi(result.stdout + result.stderr)
        output = _truncate_output(output)
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
        raw_content = p.read_text()
    except Exception as e:
        return f"read error: {e}", False

    # Check if file is unchanged since last read
    session = _kw.get("session")
    if session is not None:
        is_unchanged, last_turn = session.file_content_cache.check_unchanged(
            file_path, raw_content
        )
        if is_unchanged:
            return (
                f"[File unchanged since turn {last_turn}, "
                f"content already in context.]"
            ), True

    lines = raw_content.splitlines()
    offset = max((args.get("offset") or 1), 1)
    limit = args.get("limit") or DEFAULT_READ_LINES

    selected = lines[offset - 1 : offset - 1 + limit]
    numbered = []
    for i, line in enumerate(selected, start=offset):
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "..."
        numbered.append(f"{i:>6}\t{line}")

    if session is not None:
        session.file_content_cache.record_read(
            file_path, raw_content, session.turn_count
        )

    return "\n".join(numbered), True


async def _write_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    file_path = args.get("path", "")
    content = args.get("content", "")
    if not file_path:
        return "No path provided.", False
    p = Path(file_path)
    try:
        _atomic_write(p, content)
        session = _kw.get("session")
        if session is not None:
            session.file_content_cache.clear_path(file_path)
        msg = f"Wrote {len(content)} bytes to {file_path}"
        # Syntax validation for Python files
        if p.suffix == ".py":
            from agent.tools.edit_utils import validate_python
            warnings = validate_python(content, file_path)
            if warnings:
                msg += "\n\nValidation warnings:\n" + "\n".join(f"  ⚠ {w}" for w in warnings)
        return msg, True
    except Exception as e:
        return f"write error: {e}", False


async def _edit_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    from agent.tools.edit_utils import apply_edit, validate_python

    file_path = args.get("path", "")
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")
    replace_all = args.get("replace_all", False)
    mode = args.get("mode", "replace")

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

    try:
        new_text, replacements, fuzzy_note = apply_edit(
            text, old_str, new_str, mode=mode, replace_all=replace_all
        )
    except ValueError as e:
        return str(e), False

    try:
        _atomic_write(p, new_text)
    except Exception as e:
        return f"edit write error: {e}", False

    session = _kw.get("session")
    if session is not None:
        session.file_content_cache.clear_path(file_path)

    msg = f"Edited {file_path} ({replacements} replacement{'s' if replacements > 1 else ''})"
    if fuzzy_note:
        msg += f" {fuzzy_note}"
    # Syntax validation for Python files
    if p.suffix == ".py":
        warnings = validate_python(new_text, file_path)
        if warnings:
            msg += "\n\nValidation warnings:\n" + "\n".join(f"  ⚠ {w}" for w in warnings)
    return msg, True


# ── Local tool specs (override sandbox /app references) ────────────────

_LOCAL_TOOL_SPECS = {
    "bash": {
        "description": (
            "Run a shell command on the local machine and return stdout/stderr.\n"
            "\n"
            "Commands run in a shell at the working directory (default: current directory). "
            "Each invocation is independent.\n"
            "\n"
            "AVOID using bash for operations covered by specialized tools:\n"
            "- File reading: use read (not cat/head/tail)\n"
            "- File editing: use edit (not sed/awk)\n"
            "- File writing: use write (not echo/cat <<EOF)\n"
            "\n"
            "Chain dependent commands with &&. Independent commands should be "
            "separate bash calls (they can run in parallel).\n"
            "\n"
            "Timeout default 120s, max 600s."
        ),
        "parameters": {
            "type": "object",
            "required": ["command"],
            "additionalProperties": False,
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "description": {
                    "type": "string",
                    "description": "Short description (5-10 words, active voice).",
                },
                "work_dir": {
                    "type": "string",
                    "description": "Working directory (default: current directory).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120, max: 600).",
                },
            },
        },
    },
    "read": {
        "description": Sandbox.TOOLS["read"]["description"],
        "parameters": Sandbox.TOOLS["read"]["parameters"],
    },
    "write": {
        "description": Sandbox.TOOLS["write"]["description"],
        "parameters": Sandbox.TOOLS["write"]["parameters"],
    },
    "edit": {
        "description": Sandbox.TOOLS["edit"]["description"],
        "parameters": Sandbox.TOOLS["edit"]["parameters"],
    },
}

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
    for name, spec in _LOCAL_TOOL_SPECS.items():
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
