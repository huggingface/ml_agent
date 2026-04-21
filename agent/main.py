"""
Interactive CLI chat with the agent

Supports two modes:
  Interactive:  python -m agent.main
  Headless:     python -m agent.main "find me bird datasets"
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import litellm
from prompt_toolkit import PromptSession

from agent.config import load_config
from agent.core.agent_loop import submission_loop
from agent.core.session import OpType
from agent.core.tools import ToolRouter
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

# ── Suggested models shown by `/model` (not a gate) ──────────────────────
# Any model id accepted by litellm is usable; for the HF router the form is
# `huggingface/<inference_provider>/<org>/<model>` and users can pick any
# model supported by any HF inference provider.
SUGGESTED_MODELS = [
    {"id": "anthropic/claude-opus-4-6", "label": "Claude Opus 4.6"},
    {"id": "huggingface/fireworks-ai/MiniMaxAI/MiniMax-M2.5", "label": "MiniMax M2.5"},
    {"id": "huggingface/novita/moonshotai/kimi-k2.5", "label": "Kimi K2.5"},
    {"id": "huggingface/novita/zai-org/glm-5", "label": "GLM 5"},
]


def _is_valid_model_id(model_id: str) -> bool:
    """Loose format check — lets users pick any inference-provider model.

    Accepts:
      • huggingface/<provider>/<org>/<model>  (HF router)
      • anthropic/<model>
      • openai/<model>
    Actual availability is verified by the provider when the first call
    is made; we don't want to maintain a hardcoded allowlist.
    """
    if not model_id or "/" not in model_id:
        return False
    if model_id.startswith("huggingface/"):
        # needs provider + org + model → at least 3 slashes after the prefix
        parts = model_id.split("/")
        return len(parts) >= 4 and all(parts)
    if model_id.startswith(("anthropic/", "openai/")):
        parts = model_id.split("/", 1)
        return len(parts) == 2 and bool(parts[1])
    return False


def _safe_get_args(arguments: dict) -> dict:
    """Safely extract args dict from arguments, handling cases where LLM passes string."""
    args = arguments.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return {}
    return args if isinstance(args, dict) else {}


def _get_hf_token() -> str | None:
    """Get HF token from environment, huggingface_hub API, or cached token file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        token = api.token
        if token:
            return token
    except Exception:
        pass
    # Fallback: read the cached token file directly
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token
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
            print(f"Warning: could not persist token ({e}), using for this session only.")

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

    _BASE = (90, 90, 110)       # dim base color
    _HIGHLIGHT = (255, 200, 80) # bright shimmer highlight (warm gold)
    _WIDTH = 5                  # shimmer width in characters
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
    """Accumulates streamed tokens, renders full markdown on finish."""

    def __init__(self, console):
        self._console = console
        self._buffer = ""

    def add_chunk(self, text: str):
        self._buffer += text

    def finish(self):
        """Render the accumulated text as markdown, then reset."""
        if self._buffer.strip():
            print_markdown(self._buffer)
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
) -> None:
    """Background task that listens for events and displays them"""
    submission_id = [1000]
    last_tool_name = [None]
    console = _create_rich_console()
    shimmer = _ThinkingShimmer(console)
    stream_buf = _StreamBuffer(console)

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
                    print_markdown(content)
            elif event.event_type == "assistant_chunk":
                content = event.data.get("content", "") if event.data else ""
                if content:
                    stream_buf.add_chunk(content)
            elif event.event_type == "assistant_stream_end":
                shimmer.stop()
                stream_buf.finish()
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
                turn_complete_event.set()
            elif event.event_type == "interrupted":
                shimmer.stop()
                stream_buf.discard()
                print_interrupted()
                turn_complete_event.set()
            elif event.event_type == "undo_complete":
                console.print("[dim]Undone.[/dim]")
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
                error = event.data.get("error", "Unknown error") if event.data else "Unknown error"
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

                # If yolo mode is active, auto-approve everything
                if config and config.yolo_mode:
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

                    # Get user decision for this item
                    response = await prompt_session.prompt_async(
                        f"Approve item {i}? (y=yes, yolo=approve all, n=no, or provide feedback): "
                    )

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


def _handle_slash_command(
    cmd: str,
    config,
    session_holder: list,
    submission_queue: asyncio.Queue,
    submission_id: list[int],
) -> Submission | None:
    """
    Handle a slash command. Returns a Submission to enqueue, or None if
    the command was handled locally (caller should set turn_complete_event).
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

    if command == "/model":
        if not arg:
            current = config.model_name if config else ""
            print("Current model:")
            print(f"  {current}")
            print("\nSuggested models (any HF inference-provider model works):")
            for m in SUGGESTED_MODELS:
                marker = " <-- current" if m["id"] == current else ""
                print(f"  {m['id']}  ({m['label']}){marker}")
            print(
                "\nPass any id, e.g. huggingface/<provider>/<org>/<model>.\n"
                "Availability is verified on first use."
            )
            return None
        if not _is_valid_model_id(arg):
            print(f"Invalid model id format: {arg}")
            print(
                "Expected one of:\n"
                "  • huggingface/<provider>/<org>/<model>\n"
                "  • anthropic/<model>\n"
                "  • openai/<model>"
            )
            return None
        session = session_holder[0] if session_holder else None
        if session:
            session.update_model(arg)
            print(f"Model switched to {arg}")
        else:
            config.model_name = arg
            print(f"Model set to {arg} (session not started yet)")
        return None

    if command == "/yolo":
        config.yolo_mode = not config.yolo_mode
        state = "ON" if config.yolo_mode else "OFF"
        print(f"YOLO mode: {state}")
        return None

    if command == "/status":
        session = session_holder[0] if session_holder else None
        print(f"Model: {config.model_name}")
        if session:
            print(f"Turns: {session.turn_count}")
            print(f"Context items: {len(session.context_manager.items)}")
        return None

    print(f"Unknown command: {command}. Type /help for available commands.")
    return None


async def main():
    """Interactive chat with the agent"""

    # Clear screen
    os.system("clear" if os.name != "nt" else "cls")

    # Create prompt session for input (needed early for token prompt)
    prompt_session = PromptSession()

    # HF token — required, prompt if missing
    hf_token = _get_hf_token()
    if not hf_token:
        hf_token = await _prompt_and_save_hf_token(prompt_session)

    # Resolve username for banner
    hf_user = None
    try:
        from huggingface_hub import HfApi
        hf_user = HfApi(token=hf_token).whoami().get("name")
    except Exception:
        pass

    print_banner(hf_user=hf_user)

    # Create queues for communication
    submission_queue = asyncio.Queue()
    event_queue = asyncio.Queue()

    # Events to signal agent state
    turn_complete_event = asyncio.Event()
    turn_complete_event.set()
    ready_event = asyncio.Event()

    # Start agent loop in background
    config_path = Path(__file__).parent.parent / "configs" / "main_agent_config.json"
    config = load_config(config_path)

    # Create tool router with local mode
    tool_router = ToolRouter(config.mcpServers, hf_token=hf_token, local_mode=True)

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
            local_mode=True,
            stream=True,
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
        )
    )

    await ready_event.wait()

    submission_id = [0]
    last_interrupt_time = 0.0
    agent_busy = False  # True only while the agent is processing a submission

    try:
        while True:
            # Wait for previous turn to complete, with interrupt support
            try:
                await turn_complete_event.wait()
            except asyncio.CancelledError:
                break
            turn_complete_event.clear()
            agent_busy = False

            # Get user input
            try:
                user_input = await get_user_input(prompt_session)
            except EOFError:
                break
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - last_interrupt_time < 3.0:
                    break
                last_interrupt_time = now
                # If agent is actually working, cancel it
                session = session_holder[0]
                if agent_busy and session:
                    session.cancel()
                else:
                    get_console().print("[dim]Ctrl+C again to exit[/dim]")
                    turn_complete_event.set()
                continue

            # Check for exit commands
            if user_input.strip().lower() in ["exit", "quit", "/quit", "/exit"]:
                break

            # Skip empty input
            if not user_input.strip():
                turn_complete_event.set()
                continue

            # Handle slash commands
            if user_input.strip().startswith("/"):
                sub = _handle_slash_command(
                    user_input.strip(), config, session_holder, submission_queue, submission_id
                )
                if sub is None:
                    # Command handled locally, loop back for input
                    turn_complete_event.set()
                    continue
                else:
                    agent_busy = True
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
            agent_busy = True
            await submission_queue.put(submission)

    except KeyboardInterrupt:
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

    # Now safe to cancel the listener (agent is done emitting events)
    listener_task.cancel()

    get_console().print("\n[dim]Bye.[/dim]\n")


async def headless_main(
    prompt: str,
    model: str | None = None,
    max_iterations: int | None = None,
    stream: bool = True,
) -> None:
    """Run a single prompt headlessly and exit."""
    import logging

    logging.basicConfig(level=logging.WARNING)

    hf_token = _get_hf_token()
    if not hf_token:
        print("ERROR: No HF token found. Set HF_TOKEN or run `huggingface-cli login`.", file=sys.stderr)
        sys.exit(1)

    print(f"HF token loaded", file=sys.stderr)

    config_path = Path(__file__).parent.parent / "configs" / "main_agent_config.json"
    config = load_config(config_path)
    config.yolo_mode = True  # Auto-approve everything in headless mode

    if model:
        config.model_name = model

    if max_iterations is not None:
        config.max_iterations = max_iterations

    print(f"Model: {config.model_name}", file=sys.stderr)
    print(f"Max iterations: {config.max_iterations}", file=sys.stderr)
    print(f"Prompt: {prompt}", file=sys.stderr)
    print("---", file=sys.stderr)

    submission_queue: asyncio.Queue = asyncio.Queue()
    event_queue: asyncio.Queue = asyncio.Queue()

    tool_router = ToolRouter(config.mcpServers, hf_token=hf_token, local_mode=True)
    session_holder: list = [None]

    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
            session_holder=session_holder,
            hf_token=hf_token,
            local_mode=True,
            stream=stream,
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

    # Process events until turn completes
    console = _create_rich_console()
    shimmer = _ThinkingShimmer(console)
    stream_buf = _StreamBuffer(console)
    _hl_last_tool = [None]
    _hl_sub_id = [1]
    shimmer.start()

    while True:
        event = await event_queue.get()

        if event.event_type == "assistant_chunk":
            content = event.data.get("content", "") if event.data else ""
            if content:
                stream_buf.add_chunk(content)
        elif event.event_type == "assistant_stream_end":
            shimmer.stop()
            stream_buf.finish()
        elif event.event_type == "assistant_message":
            shimmer.stop()
            content = event.data.get("content", "") if event.data else ""
            if content:
                print_markdown(content)
        elif event.event_type == "tool_call":
            shimmer.stop()
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
            shimmer.start()
        elif event.event_type == "tool_log":
            tool = event.data.get("tool", "") if event.data else ""
            log = event.data.get("log", "") if event.data else ""
            if log:
                print_tool_log(tool, log)
        elif event.event_type == "approval_required":
            # Auto-approve everything in headless mode (safety net if yolo_mode
            # didn't prevent the approval event for some reason)
            tools_data = event.data.get("tools", []) if event.data else []
            approvals = [
                {
                    "tool_call_id": t.get("tool_call_id", ""),
                    "approved": True,
                    "feedback": None,
                }
                for t in tools_data
            ]
            _hl_sub_id[0] += 1
            await submission_queue.put(Submission(
                id=f"hl_approval_{_hl_sub_id[0]}",
                operation=Operation(
                    op_type=OpType.EXEC_APPROVAL,
                    data={"approvals": approvals},
                ),
            ))
        elif event.event_type == "compacted":
            old_tokens = event.data.get("old_tokens", 0) if event.data else 0
            new_tokens = event.data.get("new_tokens", 0) if event.data else 0
            print_compacted(old_tokens, new_tokens)
        elif event.event_type == "error":
            shimmer.stop()
            stream_buf.discard()
            error = event.data.get("error", "Unknown error") if event.data else "Unknown error"
            print_error(error)
            break
        elif event.event_type in ("turn_complete", "interrupted"):
            shimmer.stop()
            stream_buf.discard()
            history_size = event.data.get("history_size", "?") if event.data else "?"
            print(f"\n--- Agent {event.event_type} (history_size={history_size}) ---", file=sys.stderr)
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


def cli():
    """Entry point for the ml-intern CLI command."""
    import logging as _logging
    import warnings
    # Suppress aiohttp "Unclosed client session" noise during event loop teardown
    _logging.getLogger("asyncio").setLevel(_logging.CRITICAL)
    # Suppress litellm pydantic deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="litellm")
    # Suppress whoosh invalid escape sequence warnings (third-party, unfixed upstream)
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="whoosh")

    parser = argparse.ArgumentParser(description="Hugging Face Agent CLI")
    parser.add_argument("prompt", nargs="?", default=None, help="Run headlessly with this prompt")
    parser.add_argument("--model", "-m", default=None, help=f"Model to use (default: from config)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Max LLM requests per turn (default: 50, use -1 for unlimited)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable token streaming (use non-streaming LLM calls)")
    args = parser.parse_args()

    try:
        if args.prompt:
            max_iter = args.max_iterations
            if max_iter is not None and max_iter < 0:
                max_iter = 10_000  # effectively unlimited
            asyncio.run(headless_main(args.prompt, model=args.model, max_iterations=max_iter, stream=not args.no_stream))
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    cli()
