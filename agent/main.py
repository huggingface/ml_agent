"""
Interactive CLI chat with the agent

Supports two modes:
  Interactive:  python -m agent.main
  Headless:     python -m agent.main "find me bird datasets"
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import litellm
from prompt_toolkit import PromptSession

from agent.config import load_config
from agent.core.approval_policy import is_scheduled_operation
from agent.core.agent_loop import submission_loop
from agent.core import model_switcher
from agent.core.hf_tokens import resolve_hf_token
from agent.core.local_models import is_local_model_id
from agent.core.session import OpType
from agent.core.tools import ToolRouter
from agent.messaging.gateway import NotificationGateway
from agent.utils.reliability_checks import check_training_script_save_pattern
from agent.utils.terminal_display import (
    get_console,
    print_approval_header,
    print_approval_item,
    print_banner,
    print_compacted,
    print_error,
    print_help,
    print_init_done,
    print_interrupted,
    print_markdown,
    print_plan,
    print_tool_call,
    print_tool_log,
    print_tool_output,
    print_turn_complete,
    print_yolo_approve,
)

litellm.drop_params = True
# Suppress the "Give Feedback / Get Help" banner LiteLLM prints to stderr
# on every error — users don't need it, and our friendly errors cover the case.
litellm.suppress_debug_info = True

CLI_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "cli_agent_config.json"
logger = logging.getLogger(__name__)


def _apply_tool_runtime_override(config: Any, *, sandbox_tools: bool) -> str:
    if sandbox_tools:
        config.tool_runtime = "sandbox"
    return getattr(config, "tool_runtime", "local")


def _is_local_tool_runtime(config: Any) -> bool:
    return getattr(config, "tool_runtime", "local") == "local"


def _tool_runtime_label(local_mode: bool) -> str:
    return "local filesystem" if local_mode else "HF sandbox"


async def _wait_for_initial_sandbox_preload(session_holder: list | None) -> None:
    session = session_holder[0] if session_holder else None
    task = getattr(session, "sandbox_preload_task", None)
    if not task:
        return
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception:
        # The sandbox tool will surface the stored preload error on first use.
        return


def _is_scheduled_hf_job_tool(tool_info: dict[str, Any]) -> bool:
    if tool_info.get("tool") != "hf_jobs":
        return False
    arguments = tool_info.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False
    if not isinstance(arguments, dict):
        return False
    return is_scheduled_operation(arguments.get("operation"))


def _configure_runtime_logging() -> None:
    """Keep third-party warning spam from punching through the interactive UI."""
    import logging

    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)


def _safe_get_args(arguments: dict) -> dict:
    """Safely extract args dict from arguments, handling cases where LLM passes string."""
    args = arguments.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return {}
    return args if isinstance(args, dict) else {}


def _get_hf_user(token: str | None) -> str | None:
    """Resolve the HF username for a token, if available."""
    if not token:
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi(token=token).whoami().get("name")
    except Exception:
        return None


async def _prompt_and_save_hf_token(prompt_session: PromptSession) -> str:
    """Prompt user for HF token, validate it, save via huggingface_hub.login(). Loops until valid."""
    from prompt_toolkit.formatted_text import HTML
    from huggingface_hub import HfApi, login

    print("\nA Hugging Face token is required.")
    print("Get one at: https://huggingface.co/settings/tokens\n")

    while True:
        try:
            token = await prompt_session.prompt_async(
                HTML("<b>Paste your HF token: </b>")
            )
        except (EOFError, KeyboardInterrupt):
            print("\nToken is required to continue.")
            continue

        token = token.strip()
        if not token:
            print("Token cannot be empty.")
            continue

        # Validate token against the API
        try:
            api = HfApi(token=token)
            user_info = api.whoami()
            username = user_info.get("name", "unknown")
            print(f"Token valid (user: {username})")
        except Exception:
            print("Invalid token. Please try again.")
            continue

        # Save for future sessions
        try:
            login(token=token, add_to_git_credential=False)
            print("Token saved to ~/.cache/huggingface/token")
        except Exception as e:
            print(
                f"Warning: could not persist token ({e}), using for this session only."
            )

        return token


@dataclass
class Operation:
    """Operation to be executed by the agent"""

    op_type: OpType
    data: Optional[dict[str, Any]] = None


@dataclass
class Submission:
    """Submission to the agent loop"""

    id: str
    operation: Operation


def _create_rich_console():
    """Get the shared rich Console."""
    return get_console()


class _ThinkingShimmer:
    """Animated shiny/shimmer thinking indicator — a bright gradient sweeps across the text."""

    _BASE = (90, 90, 110)  # dim base color
    _HIGHLIGHT = (255, 200, 80)  # bright shimmer highlight (warm gold)
    _WIDTH = 5  # shimmer width in characters
    _FPS = 24

    def __init__(self, console):
        self._console = console
        self._task = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._animate())

    def stop(self):
        if not self._running:
            return  # no-op when never started (e.g. headless mode)
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        # Clear the shimmer line
        self._console.file.write("\r\033[K")
        self._console.file.flush()

    def _render_frame(self, text: str, offset: float) -> str:
        """Render one frame: a bright spot sweeps left-to-right across `text`."""
        out = []
        n = len(text)
        for i, ch in enumerate(text):
            # Distance from the shimmer center (wraps around)
            dist = abs(i - offset)
            wrap_dist = abs(i - offset + n + self._WIDTH)
            dist = min(dist, wrap_dist, abs(i - offset - n - self._WIDTH))
            # Blend factor: 1.0 at center, 0.0 beyond _WIDTH
            t = max(0.0, 1.0 - dist / self._WIDTH)
            t = t * t * (3 - 2 * t)  # smoothstep
            r = int(self._BASE[0] + (self._HIGHLIGHT[0] - self._BASE[0]) * t)
            g = int(self._BASE[1] + (self._HIGHLIGHT[1] - self._BASE[1]) * t)
            b = int(self._BASE[2] + (self._HIGHLIGHT[2] - self._BASE[2]) * t)
            out.append(f"\033[38;2;{r};{g};{b}m{ch}")
        out.append("\033[0m")
        return "".join(out)

    async def _animate(self):
        text = "Thinking..."
        n = len(text)
        speed = 0.45  # characters per frame
        pos = 0.0
        try:
            while self._running:
                frame = self._render_frame(text, pos)
                self._console.file.write(f"\r  {frame}")
                self._console.file.flush()
                pos = (pos + speed) % (n + self._WIDTH)
                await asyncio.sleep(1.0 / self._FPS)
        except asyncio.CancelledError:
            pass


class _StreamBuffer:
    """Accumulates streamed tokens, renders markdown block-by-block as complete
    blocks appear. A "block" is everything up to a paragraph break (\\n\\n).
    Unclosed code fences (odd count of ```) hold back flushing until closed so
    a code block is always rendered as one unit."""

    def __init__(self, console):
        self._console = console
        self._buffer = ""

    def add_chunk(self, text: str):
        self._buffer += text

    def _pop_block(self) -> str | None:
        """Extract the next complete block, or return None if nothing complete."""
        if self._buffer.count("```") % 2 == 1:
            return None  # inside an open code fence — wait for close
        idx = self._buffer.find("\n\n")
        if idx == -1:
            return None
        block = self._buffer[:idx]
        self._buffer = self._buffer[idx + 2 :]
        return block

    async def flush_ready(
        self,
        cancel_event: "asyncio.Event | None" = None,
        instant: bool = False,
    ):
        """Render any complete blocks that have accumulated; leave the tail."""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return
            block = self._pop_block()
            if block is None:
                return
            if block.strip():
                await print_markdown(block, cancel_event=cancel_event, instant=instant)

    async def finish(
        self,
        cancel_event: "asyncio.Event | None" = None,
        instant: bool = False,
    ):
        """Flush complete blocks, then render whatever incomplete tail remains."""
        await self.flush_ready(cancel_event=cancel_event, instant=instant)
        if self._buffer.strip():
            await print_markdown(
                self._buffer, cancel_event=cancel_event, instant=instant
            )
        self._buffer = ""

    def discard(self):
        self._buffer = ""


async def event_listener(
    event_queue: asyncio.Queue,
    submission_queue: asyncio.Queue,
    turn_complete_event: asyncio.Event,
    ready_event: asyncio.Event,
    prompt_session: PromptSession,
    config=None,
    session_holder=None,
) -> None:
    """Background task that listens for events and displays them"""
    submission_id = [1000]
    last_tool_name = [None]
    console = _create_rich_console()
    shimmer = _ThinkingShimmer(console)
    stream_buf = _StreamBuffer(console)

    def _cancel_event():
        """Return the session's cancellation Event so print_markdown can abort
        its typewriter loop mid-stream when Ctrl+C fires."""
        s = session_holder[0] if session_holder else None
        return s._cancelled if s is not None else None

    while True:
        try:
            event = await event_queue.get()

            if event.event_type == "ready":
                tool_count = event.data.get("tool_count", 0) if event.data else 0
                print_init_done(tool_count=tool_count)
                ready_event.set()
            elif event.event_type == "assistant_message":
                shimmer.stop()
                content = event.data.get("content", "") if event.data else ""
                if content:
                    await print_markdown(content, cancel_event=_cancel_event())
            elif event.event_type == "assistant_chunk":
                content = event.data.get("content", "") if event.data else ""
                if content:
                    stream_buf.add_chunk(content)
                    # Flush any complete markdown blocks progressively so the
                    # user sees paragraphs appear as they're produced, not just
                    # at the end of the whole response.
                    shimmer.stop()
                    await stream_buf.flush_ready(cancel_event=_cancel_event())
            elif event.event_type == "assistant_stream_end":
                shimmer.stop()
                await stream_buf.finish(cancel_event=_cancel_event())
            elif event.event_type == "tool_call":
                shimmer.stop()
                stream_buf.discard()
                tool_name = event.data.get("tool", "") if event.data else ""
                arguments = event.data.get("arguments", {}) if event.data else {}
                if tool_name:
                    last_tool_name[0] = tool_name
                    # Skip printing research tool_call — the tool_log handler shows it
                    if tool_name != "research":
                        args_str = json.dumps(arguments)[:80]
                        print_tool_call(tool_name, args_str)
            elif event.event_type == "tool_output":
                output = event.data.get("output", "") if event.data else ""
                success = event.data.get("success", False) if event.data else False
                # Only show output for plan_tool — everything else is noise
                if last_tool_name[0] == "plan_tool" and output:
                    print_tool_output(output, success, truncate=False)
                shimmer.start()
            elif event.event_type == "turn_complete":
                shimmer.stop()
                stream_buf.discard()
                print_turn_complete()
                print_plan()
                session = session_holder[0] if session_holder else None
                if session is not None:
                    await session.send_deferred_turn_complete_notification(event)
                turn_complete_event.set()
            elif event.event_type == "interrupted":
                shimmer.stop()
                stream_buf.discard()
                print_interrupted()
                turn_complete_event.set()
            elif event.event_type == "undo_complete":
                console.print("[dim]Undone.[/dim]")
                turn_complete_event.set()
            elif event.event_type == "resume_complete":
                data = event.data or {}
                path = data.get("path", "?")
                count = data.get("restored_count", 0)
                dropped = int(data.get("dropped_count", 0) or 0)
                model = data.get("model_name", "?")
                invalid_model = data.get("invalid_saved_model")
                forked = bool(data.get("forked", False))
                redacted = bool(data.get("had_redacted_content", False))
                verb = "Forked from" if forked else "Resumed"
                console.print(
                    f"[green]{verb}[/green] {path} "
                    f"([cyan]{count}[/cyan] messages, "
                    f"model [cyan]{model}[/cyan])."
                )
                if dropped:
                    console.print(
                        f"[yellow]Warning:[/yellow] dropped {dropped} "
                        "malformed message(s) while restoring — surrounding "
                        "tool-call alignment may be off."
                    )
                if invalid_model:
                    console.print(
                        f"[yellow]Warning:[/yellow] saved model id "
                        f"[cyan]{invalid_model}[/cyan] failed validation; "
                        f"kept current model [cyan]{model}[/cyan]."
                    )
                if forked:
                    console.print(
                        "[dim]Saved log belongs to a different user — kept "
                        "current session id; future saves go to a fresh file.[/dim]"
                    )
                if redacted:
                    console.print(
                        "[yellow]Note:[/yellow] tokens/secrets in restored "
                        "messages were scrubbed at save time. Your live tokens "
                        "are used for this session; [REDACTED_*] markers in "
                        "past messages are not re-injected."
                    )
                turn_complete_event.set()
            elif event.event_type == "tool_log":
                tool = event.data.get("tool", "") if event.data else ""
                log = event.data.get("log", "") if event.data else ""
                if log:
                    agent_id = event.data.get("agent_id", "") if event.data else ""
                    label = event.data.get("label", "") if event.data else ""
                    print_tool_log(tool, log, agent_id=agent_id, label=label)
            elif event.event_type == "tool_state_change":
                pass  # visual noise — approval flow handles this
            elif event.event_type == "error":
                shimmer.stop()
                stream_buf.discard()
                error = (
                    event.data.get("error", "Unknown error")
                    if event.data
                    else "Unknown error"
                )
                print_error(error)
                turn_complete_event.set()
            elif event.event_type == "shutdown":
                shimmer.stop()
                stream_buf.discard()
                break
            elif event.event_type == "processing":
                shimmer.start()
            elif event.event_type == "compacted":
                old_tokens = event.data.get("old_tokens", 0) if event.data else 0
                new_tokens = event.data.get("new_tokens", 0) if event.data else 0
                print_compacted(old_tokens, new_tokens)
            elif event.event_type == "approval_required":
                # Handle batch approval format
                tools_data = event.data.get("tools", []) if event.data else []
                count = event.data.get("count", 0) if event.data else 0

                # If yolo mode is active, auto-approve everything except
                # scheduled HF jobs, whose recurring cost stays manual.
                if (
                    config
                    and config.yolo_mode
                    and not any(_is_scheduled_hf_job_tool(t) for t in tools_data)
                ):
                    approvals = [
                        {
                            "tool_call_id": t.get("tool_call_id", ""),
                            "approved": True,
                            "feedback": None,
                        }
                        for t in tools_data
                    ]
                    print_yolo_approve(count)
                    submission_id[0] += 1
                    approval_submission = Submission(
                        id=f"approval_{submission_id[0]}",
                        operation=Operation(
                            op_type=OpType.EXEC_APPROVAL,
                            data={"approvals": approvals},
                        ),
                    )
                    await submission_queue.put(approval_submission)
                    continue

                print_approval_header(count)
                approvals = []

                # Ask for approval for each tool
                for i, tool_info in enumerate(tools_data, 1):
                    tool_name = tool_info.get("tool", "")
                    arguments = tool_info.get("arguments", {})
                    tool_call_id = tool_info.get("tool_call_id", "")

                    # Handle case where arguments might be a JSON string
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            print(f"Warning: Failed to parse arguments for {tool_name}")
                            arguments = {}

                    operation = arguments.get("operation", "")

                    print_approval_item(i, count, tool_name, operation)

                    # Handle different tool types
                    if tool_name == "hf_jobs":
                        # Check if this is Python mode (script) or Docker mode (command)
                        script = arguments.get("script")
                        command = arguments.get("command")

                        if script:
                            # Python mode
                            dependencies = arguments.get("dependencies", [])
                            python_version = arguments.get("python")
                            script_args = arguments.get("script_args", [])

                            # Show full script
                            print(f"Script:\n{script}")
                            if dependencies:
                                print(f"Dependencies: {', '.join(dependencies)}")
                            if python_version:
                                print(f"Python version: {python_version}")
                            if script_args:
                                print(f"Script args: {' '.join(script_args)}")

                            # Run reliability checks on the full script (not truncated)
                            check_message = check_training_script_save_pattern(script)
                            if check_message:
                                print(check_message)
                        elif command:
                            # Docker mode
                            image = arguments.get("image", "python:3.12")
                            command_str = (
                                " ".join(command)
                                if isinstance(command, list)
                                else str(command)
                            )
                            print(f"Docker image: {image}")
                            print(f"Command: {command_str}")

                        # Common parameters for jobs
                        hardware_flavor = arguments.get("hardware_flavor", "cpu-basic")
                        timeout = arguments.get("timeout", "30m")
                        env = arguments.get("env", {})
                        schedule = arguments.get("schedule")

                        print(f"Hardware: {hardware_flavor}")
                        print(f"Timeout: {timeout}")

                        if env:
                            env_keys = ", ".join(env.keys())
                            print(f"Environment variables: {env_keys}")

                        if schedule:
                            print(f"Schedule: {schedule}")

                    elif tool_name == "hf_private_repos":
                        # Handle private repo operations
                        args = _safe_get_args(arguments)

                        if operation in ["create_repo", "upload_file"]:
                            repo_id = args.get("repo_id", "")
                            repo_type = args.get("repo_type", "dataset")

                            # Build repo URL
                            type_path = "" if repo_type == "model" else f"{repo_type}s"
                            repo_url = (
                                f"https://huggingface.co/{type_path}/{repo_id}".replace(
                                    "//", "/"
                                )
                            )

                            print(f"Repository: {repo_id}")
                            print(f"Type: {repo_type}")
                            print("Private: Yes")
                            print(f"URL: {repo_url}")

                            # Show file preview for upload_file operation
                            if operation == "upload_file":
                                path_in_repo = args.get("path_in_repo", "")
                                file_content = args.get("file_content", "")
                                print(f"File: {path_in_repo}")

                                if isinstance(file_content, str):
                                    # Calculate metrics
                                    all_lines = file_content.split("\n")
                                    line_count = len(all_lines)
                                    size_bytes = len(file_content.encode("utf-8"))
                                    size_kb = size_bytes / 1024
                                    size_mb = size_kb / 1024

                                    print(f"Line count: {line_count}")
                                    if size_kb < 1024:
                                        print(f"Size: {size_kb:.2f} KB")
                                    else:
                                        print(f"Size: {size_mb:.2f} MB")

                                    # Show preview
                                    preview_lines = all_lines[:5]
                                    preview = "\n".join(preview_lines)
                                    print(
                                        f"Content preview (first 5 lines):\n{preview}"
                                    )
                                    if len(all_lines) > 5:
                                        print("...")

                    elif tool_name == "hf_repo_files":
                        # Handle repo files operations (upload, delete)
                        repo_id = arguments.get("repo_id", "")
                        repo_type = arguments.get("repo_type", "model")
                        revision = arguments.get("revision", "main")

                        # Build repo URL
                        if repo_type == "model":
                            repo_url = f"https://huggingface.co/{repo_id}"
                        else:
                            repo_url = f"https://huggingface.co/{repo_type}s/{repo_id}"

                        print(f"Repository: {repo_id}")
                        print(f"Type: {repo_type}")
                        print(f"Branch: {revision}")
                        print(f"URL: {repo_url}")

                        if operation == "upload":
                            path = arguments.get("path", "")
                            content = arguments.get("content", "")
                            create_pr = arguments.get("create_pr", False)

                            print(f"File: {path}")
                            if create_pr:
                                print("Mode: Create PR")

                            if isinstance(content, str):
                                all_lines = content.split("\n")
                                line_count = len(all_lines)
                                size_bytes = len(content.encode("utf-8"))
                                size_kb = size_bytes / 1024

                                print(f"Lines: {line_count}")
                                if size_kb < 1024:
                                    print(f"Size: {size_kb:.2f} KB")
                                else:
                                    print(f"Size: {size_kb / 1024:.2f} MB")

                                # Show full content
                                print(f"Content:\n{content}")

                        elif operation == "delete":
                            patterns = arguments.get("patterns", [])
                            if isinstance(patterns, str):
                                patterns = [patterns]
                            print(f"Patterns to delete: {', '.join(patterns)}")

                    elif tool_name == "hf_repo_git":
                        # Handle git operations (branches, tags, PRs, repo management)
                        repo_id = arguments.get("repo_id", "")
                        repo_type = arguments.get("repo_type", "model")

                        # Build repo URL
                        if repo_type == "model":
                            repo_url = f"https://huggingface.co/{repo_id}"
                        else:
                            repo_url = f"https://huggingface.co/{repo_type}s/{repo_id}"

                        print(f"Repository: {repo_id}")
                        print(f"Type: {repo_type}")
                        print(f"URL: {repo_url}")

                        if operation == "delete_branch":
                            branch = arguments.get("branch", "")
                            print(f"Branch to delete: {branch}")

                        elif operation == "delete_tag":
                            tag = arguments.get("tag", "")
                            print(f"Tag to delete: {tag}")

                        elif operation == "merge_pr":
                            pr_num = arguments.get("pr_num", "")
                            print(f"PR to merge: #{pr_num}")

                        elif operation == "create_repo":
                            private = arguments.get("private", False)
                            space_sdk = arguments.get("space_sdk")
                            print(f"Private: {private}")
                            if space_sdk:
                                print(f"Space SDK: {space_sdk}")

                        elif operation == "update_repo":
                            private = arguments.get("private")
                            gated = arguments.get("gated")
                            if private is not None:
                                print(f"Private: {private}")
                            if gated is not None:
                                print(f"Gated: {gated}")

                    # Get user decision for this item. Ctrl+C / EOF here is
                    # treated as "reject remaining" (matches Codex's modal
                    # priority and Forgecode's approval-cancel path). Without
                    # this, KeyboardInterrupt kills the event listener and
                    # the main loop deadlocks waiting for turn_complete.
                    try:
                        response = await prompt_session.prompt_async(
                            f"Approve item {i}? (y=yes, yolo=approve all, n=no, or provide feedback): "
                        )
                    except (KeyboardInterrupt, EOFError):
                        get_console().print(
                            "[dim]Approval cancelled — rejecting remaining items[/dim]"
                        )
                        approvals.append(
                            {
                                "tool_call_id": tool_call_id,
                                "approved": False,
                                "feedback": "User cancelled approval",
                            }
                        )
                        for remaining in tools_data[i:]:
                            approvals.append(
                                {
                                    "tool_call_id": remaining.get("tool_call_id", ""),
                                    "approved": False,
                                    "feedback": None,
                                }
                            )
                        break

                    response = response.strip().lower()

                    # Handle yolo mode activation
                    if response == "yolo":
                        config.yolo_mode = True
                        print(
                            "YOLO MODE ACTIVATED - Auto-approving all future tool calls"
                        )
                        # Auto-approve this item and all remaining
                        approvals.append(
                            {
                                "tool_call_id": tool_call_id,
                                "approved": True,
                                "feedback": None,
                            }
                        )
                        for remaining in tools_data[i:]:
                            approvals.append(
                                {
                                    "tool_call_id": remaining.get("tool_call_id", ""),
                                    "approved": True,
                                    "feedback": None,
                                }
                            )
                        break

                    approved = response in ["y", "yes"]
                    feedback = None if approved or response in ["n", "no"] else response

                    approvals.append(
                        {
                            "tool_call_id": tool_call_id,
                            "approved": approved,
                            "feedback": feedback,
                        }
                    )

                # Submit batch approval
                submission_id[0] += 1
                approval_submission = Submission(
                    id=f"approval_{submission_id[0]}",
                    operation=Operation(
                        op_type=OpType.EXEC_APPROVAL,
                        data={"approvals": approvals},
                    ),
                )
                await submission_queue.put(approval_submission)
                console.print()  # spacing after approval
            # Silently ignore other events

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Event listener error: {e}")


async def get_user_input(prompt_session: PromptSession) -> str:
    """Get user input asynchronously"""
    from prompt_toolkit.formatted_text import HTML

    return await prompt_session.prompt_async(HTML("\n<b><cyan>></cyan></b> "))


# ── Slash command helpers ────────────────────────────────────────────────

# Slash commands are defined in terminal_display


async def _resume_picker(
    arg: str,
    prompt_session: PromptSession | None,
) -> Path | None:
    """Resolve a session log path via ``arg`` or interactive selection.

    Returns ``None`` if the user cancels, no logs exist, or the argument
    matches nothing — already prints the explanation in those cases.
    """
    from agent.core.session_resume import (
        format_session_log_entry,
        list_session_logs,
        resolve_session_log_arg,
    )
    from agent.core.session import DEFAULT_SESSION_LOG_DIR

    console = get_console()
    directory = DEFAULT_SESSION_LOG_DIR
    entries = list_session_logs(directory)
    if not entries:
        console.print(f"[yellow]No session logs found in ./{directory}.[/yellow]")
        return None

    if arg:
        selected = resolve_session_log_arg(arg, entries, directory)
        if selected is None:
            console.print(f"[bold red]No matching session log:[/bold red] {arg}")
        return selected

    console.print()
    console.print("[bold]Saved sessions[/bold]")
    for index, entry in enumerate(entries, start=1):
        console.print(format_session_log_entry(index, entry))
    console.print()

    if prompt_session is None:
        console.print("[yellow]Cannot prompt for a selection here.[/yellow]")
        return None

    try:
        choice = await prompt_session.prompt_async(
            "Select session number (blank to cancel): "
        )
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Resume cancelled.[/dim]")
        return None
    choice = choice.strip()
    if not choice:
        console.print("[dim]Resume cancelled.[/dim]")
        return None
    selected = resolve_session_log_arg(choice, entries, directory)
    if selected is None:
        console.print(f"[bold red]Invalid selection:[/bold red] {choice}")
    return selected


async def _handle_slash_command(
    cmd: str,
    config,
    session_holder: list,
    submission_queue: asyncio.Queue,
    submission_id: list[int],
    prompt_session: PromptSession | None = None,
) -> Submission | None:
    """
    Handle a slash command. Returns a Submission to enqueue, or None if
    the command was handled locally (caller should set turn_complete_event).

    Async because ``/model`` fires a probe ping to validate the model+effort
    combo before committing the switch.
    """
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command == "/help":
        print_help()
        return None

    if command == "/undo":
        submission_id[0] += 1
        return Submission(
            id=f"sub_{submission_id[0]}",
            operation=Operation(op_type=OpType.UNDO),
        )

    if command == "/compact":
        submission_id[0] += 1
        return Submission(
            id=f"sub_{submission_id[0]}",
            operation=Operation(op_type=OpType.COMPACT),
        )

    if command == "/resume":
        session = session_holder[0] if session_holder else None
        if session is None:
            get_console().print(
                "[bold red]No active session to restore into.[/bold red]"
            )
            return None
        selected_path = await _resume_picker(arg, prompt_session)
        if selected_path is None:
            return None
        submission_id[0] += 1
        return Submission(
            id=f"sub_{submission_id[0]}",
            operation=Operation(
                op_type=OpType.RESUME, data={"path": str(selected_path)}
            ),
        )

    if command == "/model":
        console = get_console()
        if not arg:
            model_switcher.print_model_listing(config, console)
            return None
        if not model_switcher.is_valid_model_id(arg):
            model_switcher.print_invalid_id(arg, console)
            return None
        normalized = arg.removeprefix("huggingface/")
        session = session_holder[0] if session_holder else None
        await model_switcher.probe_and_switch_model(
            normalized,
            config,
            session,
            console,
            resolve_hf_token(),
        )
        return None

    if command == "/yolo":
        config.yolo_mode = not config.yolo_mode
        state = "ON" if config.yolo_mode else "OFF"
        print(f"YOLO mode: {state}")
        return None

    if command == "/effort":
        console = get_console()
        valid = {"minimal", "low", "medium", "high", "xhigh", "max", "off"}
        session = session_holder[0] if session_holder else None
        if not arg:
            current = config.reasoning_effort or "off"
            console.print(f"[bold]Reasoning effort preference:[/bold] {current}")
            if session and session.model_effective_effort:
                console.print("[dim]Probed per model:[/dim]")
                for m, eff in session.model_effective_effort.items():
                    console.print(f"  [dim]{m}: {eff or 'off'}[/dim]")
            console.print(
                "[dim]Set with '/effort minimal|low|medium|high|xhigh|max|off'. "
                "'max' is Anthropic-only; 'xhigh' is also supported by current "
                "OpenAI GPT-5 models. The cascade falls back to whatever the "
                "model actually accepts.[/dim]"
            )
            return None
        level = arg.lower()
        if level not in valid:
            console.print(f"[bold red]Invalid level:[/bold red] {arg}")
            console.print(f"[dim]Expected one of: {', '.join(sorted(valid))}[/dim]")
            return None
        config.reasoning_effort = None if level == "off" else level
        # Drop the per-model probe cache — the new preference may resolve
        # differently. Next ``/model`` (or the retry safety net) reprobes.
        if session is not None:
            session.model_effective_effort.clear()
        console.print(f"[green]Reasoning effort: {level}[/green]")
        if session is not None:
            console.print(
                "[dim]run /model <current> to re-probe, or send a message — "
                "the agent adjusts automatically if the new level isn't supported.[/dim]"
            )
        return None

    if command == "/status":
        session = session_holder[0] if session_holder else None
        print(f"Model: {config.model_name}")
        print(f"Reasoning effort: {config.reasoning_effort or 'off'}")
        print(f"Tool runtime: {_tool_runtime_label(_is_local_tool_runtime(config))}")
        if session:
            print(f"Turns: {session.turn_count}")
            print(f"Context items: {len(session.context_manager.items)}")
        return None

    if command == "/share-traces":
        session = session_holder[0] if session_holder else None
        await _handle_share_traces_command(arg, config, session)
        return None

    print(f"Unknown command: {command}. Type /help for available commands.")
    return None


async def _handle_share_traces_command(arg: str, config, session) -> None:
    """Show or flip visibility of the user's personal trace dataset.

    Uses the user's own HF_TOKEN (write-scoped to their namespace). Only
    operates on the personal trace repo configured via
    ``personal_trace_repo_template`` — never touches the shared org dataset.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    console = get_console()
    if session is None:
        console.print("[bold red]No active session.[/bold red]")
        return

    repo_id = session._personal_trace_repo_id() if session is not None else None
    if not repo_id:
        if not getattr(config, "share_traces", False):
            console.print(
                "[yellow]share_traces is disabled in config. "
                "Set it to true to publish per-session traces to your HF dataset."
                "[/yellow]"
            )
            return
        if not session.user_id:
            console.print(
                "[yellow]No HF username resolved \u2014 cannot pick a personal "
                "trace repo. Set HF_TOKEN to a token tied to your account.[/yellow]"
            )
            return
        console.print(
            "[yellow]personal_trace_repo_template is unset \u2014 nothing to do.[/yellow]"
        )
        return

    token = session.hf_token or resolve_hf_token()
    if not token:
        console.print(
            "[bold red]No HF_TOKEN available.[/bold red] Cannot read or change "
            "dataset visibility."
        )
        return

    api = HfApi(token=token)
    url = f"https://huggingface.co/datasets/{repo_id}"
    target = arg.strip().lower()

    if not target:
        try:
            info = await asyncio.to_thread(
                api.repo_info, repo_id=repo_id, repo_type="dataset"
            )
            visibility = "private" if getattr(info, "private", False) else "public"
            console.print(f"[bold]Trace dataset:[/bold] {url}")
            console.print(f"[bold]Visibility:[/bold] {visibility}")
            console.print(
                "[dim]Use '/share-traces public' to publish, "
                "'/share-traces private' to lock it back down.[/dim]"
            )
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                console.print(
                    f"[dim]Dataset {repo_id} doesn't exist yet \u2014 it'll be "
                    "created (private) on the next session save.[/dim]"
                )
            else:
                console.print(f"[bold red]Hub error:[/bold red] {e}")
        except Exception as e:
            console.print(f"[bold red]Could not fetch dataset info:[/bold red] {e}")
        return

    if target not in {"public", "private"}:
        console.print(
            f"[bold red]Unknown argument:[/bold red] {target}. "
            "Expected 'public' or 'private'."
        )
        return

    private = target == "private"
    try:
        # Idempotent — create if missing so first-flip works even before any
        # session has been saved yet.
        await asyncio.to_thread(
            api.create_repo,
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True,
        )
        await asyncio.to_thread(
            api.update_repo_settings,
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to update visibility:[/bold red] {e}")
        return

    label = "PUBLIC" if not private else "private"
    console.print(f"[green]Dataset is now {label}.[/green] {url}")


async def main(model: str | None = None, sandbox_tools: bool = False):
    """Interactive chat with the agent"""

    # Clear screen
    os.system("clear" if os.name != "nt" else "cls")

    # Create prompt session for input (needed early for token prompt)
    prompt_session = PromptSession()

    config = load_config(CLI_CONFIG_PATH, include_user_defaults=True)
    if model:
        config.model_name = model
    _apply_tool_runtime_override(config, sandbox_tools=sandbox_tools)
    local_mode = _is_local_tool_runtime(config)

    # HF token — required for Hub-backed models/tools and sandbox tools, but
    # not for local LLMs using only local filesystem tools.
    hf_token = resolve_hf_token()
    if not hf_token and (not is_local_model_id(config.model_name) or not local_mode):
        hf_token = await _prompt_and_save_hf_token(prompt_session)

    # Resolve username for banner
    hf_user = _get_hf_user(hf_token)

    print_banner(
        model=config.model_name,
        hf_user=hf_user,
        tool_runtime=_tool_runtime_label(local_mode),
    )

    # Pre-warm the HF router catalog in the background so /model switches
    # don't block on a network fetch.
    from agent.core import hf_router_catalog

    asyncio.create_task(asyncio.to_thread(hf_router_catalog.prewarm))

    # Create queues for communication
    submission_queue = asyncio.Queue()
    event_queue = asyncio.Queue()

    # Events to signal agent state
    turn_complete_event = asyncio.Event()
    turn_complete_event.set()
    ready_event = asyncio.Event()

    notification_gateway = NotificationGateway(config.messaging)
    await notification_gateway.start()
    # Create tool router with the selected CLI tool runtime.
    tool_router = ToolRouter(
        config.mcpServers, hf_token=hf_token, local_mode=local_mode
    )

    # Session holder for interrupt/model/status access
    session_holder = [None]

    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
            session_holder=session_holder,
            hf_token=hf_token,
            user_id=hf_user,
            local_mode=local_mode,
            stream=True,
            notification_gateway=notification_gateway,
            notification_destinations=config.messaging.default_auto_destinations(),
            defer_turn_complete_notification=True,
        )
    )

    # Start event listener in background
    listener_task = asyncio.create_task(
        event_listener(
            event_queue,
            submission_queue,
            turn_complete_event,
            ready_event,
            prompt_session,
            config,
            session_holder=session_holder,
        )
    )

    await ready_event.wait()
    if not local_mode:
        await _wait_for_initial_sandbox_preload(session_holder)

    submission_id = [0]
    # Mirrors codex-rs/tui/src/bottom_pane/mod.rs:137
    # (`QUIT_SHORTCUT_TIMEOUT = Duration::from_secs(1)`). Two Ctrl+C presses
    # within this window quit; a single press cancels the in-flight turn.
    CTRL_C_QUIT_WINDOW = 1.0
    # Hint string matches codex-rs/tui/src/bottom_pane/footer.rs:746
    # (`" again to quit"` prefixed with the key binding, rendered dim).
    CTRL_C_HINT = "[dim]ctrl + c again to quit[/dim]"
    interrupt_state = {"last": 0.0, "exit": False}

    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        """SIGINT handler — fires while the agent is generating (terminal is
        in cooked mode between prompts). Mirrors Codex's `on_ctrl_c` in
        codex-rs/tui/src/chatwidget.rs: first press cancels active work and
        arms the quit hint; second press within the window quits."""
        now = time.monotonic()
        session = session_holder[0]

        if now - interrupt_state["last"] < CTRL_C_QUIT_WINDOW:
            interrupt_state["exit"] = True
            if session:
                session.cancel()
            # Wake the main loop out of turn_complete_event.wait()
            turn_complete_event.set()
            return

        interrupt_state["last"] = now
        if session and not session.is_cancelled:
            session.cancel()
        get_console().print(f"\n{CTRL_C_HINT}")

    def _install_sigint() -> bool:
        try:
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
            return True
        except (NotImplementedError, RuntimeError):
            return False  # Windows or non-main thread

    # prompt_toolkit's prompt_async installs its own SIGINT handler and, on
    # exit, calls loop.remove_signal_handler(SIGINT) — which wipes ours too.
    # So we re-arm at the top of every loop iteration, right before the busy
    # wait. Without this, Ctrl+C during agent streaming after the first turn
    # falls through to the default handler and the terminal just echoes ^C.
    sigint_available = _install_sigint()

    try:
        while True:
            if sigint_available:
                _install_sigint()

            try:
                await turn_complete_event.wait()
            except asyncio.CancelledError:
                break
            turn_complete_event.clear()

            if interrupt_state["exit"]:
                break

            # Get user input. prompt_toolkit puts the terminal in raw mode and
            # installs its own SIGINT handling; ^C arrives as \x03 and surfaces
            # as KeyboardInterrupt here. On return, prompt_toolkit removes the
            # loop's SIGINT handler — we re-arm at the top of the next iter.
            try:
                user_input = await get_user_input(prompt_session)
            except EOFError:
                break
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - interrupt_state["last"] < CTRL_C_QUIT_WINDOW:
                    break
                interrupt_state["last"] = now
                get_console().print(CTRL_C_HINT)
                turn_complete_event.set()
                continue

            # A successful read ends the double-press window — an unrelated
            # Ctrl+C during the next turn should start a fresh arming.
            interrupt_state["last"] = 0.0

            # Check for exit commands
            if user_input.strip().lower() in ["exit", "quit", "/quit", "/exit"]:
                break

            # Skip empty input
            if not user_input.strip():
                turn_complete_event.set()
                continue

            # Handle slash commands
            if user_input.strip().startswith("/"):
                sub = await _handle_slash_command(
                    user_input.strip(),
                    config,
                    session_holder,
                    submission_queue,
                    submission_id,
                    prompt_session,
                )
                if sub is None:
                    # Command handled locally, loop back for input
                    turn_complete_event.set()
                    continue
                else:
                    await submission_queue.put(sub)
                    continue

            # Submit to agent
            submission_id[0] += 1
            submission = Submission(
                id=f"sub_{submission_id[0]}",
                operation=Operation(
                    op_type=OpType.USER_INPUT, data={"text": user_input}
                ),
            )
            await submission_queue.put(submission)

    except KeyboardInterrupt:
        pass
    finally:
        if sigint_available:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass

    # Shutdown
    shutdown_submission = Submission(
        id="sub_shutdown", operation=Operation(op_type=OpType.SHUTDOWN)
    )
    await submission_queue.put(shutdown_submission)

    # Wait for agent to finish (the listener must keep draining events
    # or the agent will block on event_queue.put)
    try:
        await asyncio.wait_for(agent_task, timeout=10.0)
    except asyncio.TimeoutError:
        agent_task.cancel()
        # Agent didn't shut down cleanly — close MCP explicitly
        await tool_router.__aexit__(None, None, None)
    finally:
        await notification_gateway.close()

    # Now safe to cancel the listener (agent is done emitting events)
    listener_task.cancel()

    get_console().print("\n[dim]Bye.[/dim]\n")


async def headless_main(
    prompt: str,
    model: str | None = None,
    max_iterations: int | None = None,
    stream: bool = True,
    sandbox_tools: bool = False,
) -> None:
    """Run a single prompt headlessly and exit."""
    import logging

    logging.basicConfig(level=logging.WARNING)
    _configure_runtime_logging()

    config = load_config(CLI_CONFIG_PATH, include_user_defaults=True)
    config.yolo_mode = True  # Auto-approve everything in headless mode

    if model:
        config.model_name = model
    _apply_tool_runtime_override(config, sandbox_tools=sandbox_tools)
    local_mode = _is_local_tool_runtime(config)

    hf_token = resolve_hf_token()
    if not hf_token and (not is_local_model_id(config.model_name) or not local_mode):
        print(
            "ERROR: No HF token found. Set HF_TOKEN or run `hf auth login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    if hf_token:
        print("HF token loaded", file=sys.stderr)

    notification_gateway = NotificationGateway(config.messaging)
    await notification_gateway.start()
    hf_user = _get_hf_user(hf_token)

    if max_iterations is not None:
        config.max_iterations = max_iterations

    print(f"Model: {config.model_name}", file=sys.stderr)
    print(f"Tool runtime: {_tool_runtime_label(local_mode)}", file=sys.stderr)
    print(f"Max iterations: {config.max_iterations}", file=sys.stderr)
    print(f"Prompt: {prompt}", file=sys.stderr)
    print("---", file=sys.stderr)

    submission_queue: asyncio.Queue = asyncio.Queue()
    event_queue: asyncio.Queue = asyncio.Queue()

    tool_router = ToolRouter(
        config.mcpServers, hf_token=hf_token, local_mode=local_mode
    )
    session_holder: list = [None]

    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
            session_holder=session_holder,
            hf_token=hf_token,
            user_id=hf_user,
            local_mode=local_mode,
            stream=stream,
            notification_gateway=notification_gateway,
            notification_destinations=config.messaging.default_auto_destinations(),
            defer_turn_complete_notification=True,
        )
    )

    # Wait for ready
    while True:
        event = await event_queue.get()
        if event.event_type == "ready":
            break

    # Submit the prompt
    submission = Submission(
        id="sub_1",
        operation=Operation(op_type=OpType.USER_INPUT, data={"text": prompt}),
    )
    await submission_queue.put(submission)

    # Process events until turn completes. Headless mode is for scripts /
    # log capture: no shimmer animation, no typewriter, no live-redrawing
    # research overlay. Output is plain, append-only text.
    console = _create_rich_console()
    stream_buf = _StreamBuffer(console)
    _hl_last_tool = [None]
    _hl_sub_id = [1]
    # Research sub-agent tool calls are buffered per agent_id and dumped as
    # a static block once each sub-agent finishes, instead of streaming via
    # the live redrawing SubAgentDisplayManager (which is TTY-only).
    _hl_research_buffers: dict[str, dict] = {}

    while True:
        event = await event_queue.get()

        if event.event_type == "assistant_chunk":
            content = event.data.get("content", "") if event.data else ""
            if content:
                stream_buf.add_chunk(content)
                await stream_buf.flush_ready(instant=True)
        elif event.event_type == "assistant_stream_end":
            await stream_buf.finish(instant=True)
        elif event.event_type == "assistant_message":
            content = event.data.get("content", "") if event.data else ""
            if content:
                await print_markdown(content, instant=True)
        elif event.event_type == "tool_call":
            stream_buf.discard()
            tool_name = event.data.get("tool", "") if event.data else ""
            arguments = event.data.get("arguments", {}) if event.data else {}
            if tool_name:
                _hl_last_tool[0] = tool_name
                if tool_name != "research":
                    args_str = json.dumps(arguments)[:80]
                    print_tool_call(tool_name, args_str)
        elif event.event_type == "tool_output":
            output = event.data.get("output", "") if event.data else ""
            success = event.data.get("success", False) if event.data else False
            if _hl_last_tool[0] == "plan_tool" and output:
                print_tool_output(output, success, truncate=False)
        elif event.event_type == "tool_log":
            tool = event.data.get("tool", "") if event.data else ""
            log = event.data.get("log", "") if event.data else ""
            if not log:
                pass
            elif tool == "research":
                # Headless mode: buffer research sub-agent activity per-agent,
                # then dump each as a static block on completion. The live
                # SubAgentDisplayManager uses terminal cursor tricks that are
                # unfit for non-TTY output, but parallel agents still need
                # distinct output so we key buffers by agent_id.
                agent_id = event.data.get("agent_id", "") if event.data else ""
                label = event.data.get("label", "") if event.data else ""
                aid = agent_id or "research"
                if log == "Starting research sub-agent...":
                    _hl_research_buffers[aid] = {
                        "label": label or "research",
                        "calls": [],
                    }
                elif log == "Research complete.":
                    buf = _hl_research_buffers.pop(aid, None)
                    if buf is not None:
                        f = get_console().file
                        f.write(f"  \033[38;2;255;200;80m▸ {buf['label']}\033[0m\n")
                        for call in buf["calls"]:
                            f.write(f"    \033[2m{call}\033[0m\n")
                        f.flush()
                elif log.startswith("tokens:") or log.startswith("tools:"):
                    pass  # stats updates — only useful for the live display
                elif aid in _hl_research_buffers:
                    _hl_research_buffers[aid]["calls"].append(log)
                else:
                    # Orphan event (Start was missed) — fall back to raw print
                    print_tool_log(tool, log, agent_id=agent_id, label=label)
            else:
                print_tool_log(tool, log)
        elif event.event_type == "approval_required":
            # Auto-approve in headless mode, except scheduled HF jobs. Those
            # are rejected because their recurring cost needs manual approval.
            tools_data = event.data.get("tools", []) if event.data else []
            approvals = [
                {
                    "tool_call_id": t.get("tool_call_id", ""),
                    "approved": not _is_scheduled_hf_job_tool(t),
                    "feedback": (
                        "Scheduled HF jobs require manual approval."
                        if _is_scheduled_hf_job_tool(t)
                        else None
                    ),
                }
                for t in tools_data
            ]
            _hl_sub_id[0] += 1
            await submission_queue.put(
                Submission(
                    id=f"hl_approval_{_hl_sub_id[0]}",
                    operation=Operation(
                        op_type=OpType.EXEC_APPROVAL,
                        data={"approvals": approvals},
                    ),
                )
            )
        elif event.event_type == "compacted":
            old_tokens = event.data.get("old_tokens", 0) if event.data else 0
            new_tokens = event.data.get("new_tokens", 0) if event.data else 0
            print_compacted(old_tokens, new_tokens)
        elif event.event_type == "error":
            stream_buf.discard()
            error = (
                event.data.get("error", "Unknown error")
                if event.data
                else "Unknown error"
            )
            print_error(error)
            break
        elif event.event_type in ("turn_complete", "interrupted"):
            stream_buf.discard()
            history_size = event.data.get("history_size", "?") if event.data else "?"
            print(
                f"\n--- Agent {event.event_type} (history_size={history_size}) ---",
                file=sys.stderr,
            )
            if event.event_type == "turn_complete":
                session = session_holder[0] if session_holder else None
                if session is not None:
                    await session.send_deferred_turn_complete_notification(event)
            break

    # Shutdown
    shutdown_submission = Submission(
        id="sub_shutdown", operation=Operation(op_type=OpType.SHUTDOWN)
    )
    await submission_queue.put(shutdown_submission)

    try:
        await asyncio.wait_for(agent_task, timeout=10.0)
    except asyncio.TimeoutError:
        agent_task.cancel()
        await tool_router.__aexit__(None, None, None)
    finally:
        await notification_gateway.close()


def cli():
    """Entry point for the ml-intern CLI command."""
    import logging as _logging
    import warnings

    # Suppress aiohttp "Unclosed client session" noise during event loop teardown
    _logging.getLogger("asyncio").setLevel(_logging.CRITICAL)
    _configure_runtime_logging()
    # Suppress litellm pydantic deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="litellm")
    # Suppress whoosh invalid escape sequence warnings (third-party, unfixed upstream)
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="whoosh")

    parser = argparse.ArgumentParser(description="Hugging Face Agent CLI")
    parser.add_argument(
        "prompt", nargs="?", default=None, help="Run headlessly with this prompt"
    )
    parser.add_argument(
        "--model", "-m", default=None, help="Model to use (default: from config)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Max LLM requests per turn (default: 50, use -1 for unlimited)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming (use non-streaming LLM calls)",
    )
    parser.add_argument(
        "--sandbox-tools",
        action="store_true",
        help="Use HF Space sandbox tools instead of local filesystem tools",
    )
    args = parser.parse_args()

    try:
        if args.prompt:
            max_iter = args.max_iterations
            if max_iter is not None and max_iter < 0:
                max_iter = 10_000  # effectively unlimited
            asyncio.run(
                headless_main(
                    args.prompt,
                    model=args.model,
                    max_iterations=max_iter,
                    stream=not args.no_stream,
                    sandbox_tools=args.sandbox_tools,
                )
            )
        else:
            asyncio.run(main(model=args.model, sandbox_tools=args.sandbox_tools))
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    cli()
