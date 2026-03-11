"""
Interactive CLI chat with the agent
"""

import asyncio
import json
import os
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
    format_error,
    format_header,
    format_plan_display,
    format_separator,
    format_success,
    format_tool_call,
    format_tool_output,
    format_turn_complete,
)

litellm.drop_params = True

# ── Available models (mirrors backend/routes/agent.py) ──────────────────
AVAILABLE_MODELS = [
    {"id": "anthropic/claude-opus-4-6", "label": "Claude Opus 4.6"},
    {"id": "huggingface/fireworks-ai/MiniMaxAI/MiniMax-M2.5", "label": "MiniMax M2.5"},
    {"id": "huggingface/novita/moonshotai/kimi-k2.5", "label": "Kimi K2.5"},
    {"id": "huggingface/novita/zai-org/glm-5", "label": "GLM 5"},
]
VALID_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}


def _safe_get_args(arguments: dict) -> dict:
    """Safely extract args dict from arguments, handling cases where LLM passes string."""
    args = arguments.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return {}
    return args if isinstance(args, dict) else {}


def _get_hf_token() -> str | None:
    """Get HF token from environment or huggingface_hub login."""
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
    return None


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


async def event_listener(
    event_queue: asyncio.Queue,
    submission_queue: asyncio.Queue,
    turn_complete_event: asyncio.Event,
    ready_event: asyncio.Event,
    prompt_session: PromptSession,
    config=None,
) -> None:
    """Background task that listens for events and displays them"""
    submission_id = [1000]  # Use list to make it mutable in closure
    last_tool_name = [None]  # Track last tool called

    while True:
        try:
            event = await event_queue.get()

            # Display event
            if event.event_type == "ready":
                print(format_success("\U0001f917 Agent ready"))
                ready_event.set()
            elif event.event_type == "assistant_message":
                content = event.data.get("content", "") if event.data else ""
                if content:
                    print(f"\nAssistant: {content}")
            elif event.event_type == "tool_call":
                tool_name = event.data.get("tool", "") if event.data else ""
                arguments = event.data.get("arguments", {}) if event.data else {}
                if tool_name:
                    last_tool_name[0] = tool_name  # Store for tool_output event
                    args_str = json.dumps(arguments)[:100] + "..."
                    print(format_tool_call(tool_name, args_str))
            elif event.event_type == "tool_output":
                output = event.data.get("output", "") if event.data else ""
                success = event.data.get("success", False) if event.data else False
                if output:
                    # Don't truncate plan_tool output, truncate everything else
                    should_truncate = last_tool_name[0] != "plan_tool"
                    print(format_tool_output(output, success, truncate=should_truncate))
            elif event.event_type == "turn_complete":
                print(format_turn_complete())
                # Display plan after turn complete
                plan_display = format_plan_display()
                if plan_display:
                    print(plan_display)
                turn_complete_event.set()
            elif event.event_type == "interrupted":
                print("\n(interrupted)")
                turn_complete_event.set()
            elif event.event_type == "undo_complete":
                print("Undo complete.")
                turn_complete_event.set()
            elif event.event_type == "tool_log":
                tool = event.data.get("tool", "") if event.data else ""
                log = event.data.get("log", "") if event.data else ""
                if log:
                    print(f"  [{tool}] {log}")
            elif event.event_type == "tool_state_change":
                tool = event.data.get("tool", "") if event.data else ""
                state = event.data.get("state", "") if event.data else ""
                if state in ("approved", "rejected", "running"):
                    print(f"  {tool}: {state}")
            elif event.event_type == "error":
                error = (
                    event.data.get("error", "Unknown error")
                    if event.data
                    else "Unknown error"
                )
                print(format_error(error))
                turn_complete_event.set()
            elif event.event_type == "shutdown":
                break
            elif event.event_type == "processing":
                pass  # print("Processing...", flush=True)
            elif event.event_type == "compacted":
                old_tokens = event.data.get("old_tokens", 0) if event.data else 0
                new_tokens = event.data.get("new_tokens", 0) if event.data else 0
                print(f"Compacted context: {old_tokens} -> {new_tokens} tokens")
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
                    print(f"\n YOLO MODE: Auto-approving {count} item(s)")
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

                print("\n" + format_separator())
                print(
                    format_header(
                        f"APPROVAL REQUIRED ({count} item{'s' if count != 1 else ''})"
                    )
                )
                print(format_separator())

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

                    print(f"\n[Item {i}/{count}]")
                    print(f"Tool: {tool_name}")
                    print(f"Operation: {operation}")

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
                print(format_separator() + "\n")
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

HELP_TEXT = """\
Commands:
  /help            Show this help
  /undo            Undo last turn
  /compact         Compact context window
  /model [id]      Show available models or switch model
  /yolo            Toggle auto-approve mode
  /status          Show current model, turn count
  /quit, /exit     Exit the CLI
"""


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
        print(HELP_TEXT)
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
            print("Available models:")
            session = session_holder[0] if session_holder else None
            current = config.model_name if config else ""
            for m in AVAILABLE_MODELS:
                marker = " <-- current" if m["id"] == current else ""
                print(f"  {m['id']}  ({m['label']}){marker}")
            return None
        if arg not in VALID_MODEL_IDS:
            print(f"Unknown model: {arg}")
            print(f"Valid: {', '.join(VALID_MODEL_IDS)}")
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
    from agent.utils.terminal_display import Colors

    # Clear screen
    os.system("clear" if os.name != "nt" else "cls")

    banner = r"""
  _   _                   _               _____                   _                    _
 | | | |_   _  __ _  __ _(_)_ __   __ _  |  ___|_ _  ___ ___     / \   __ _  ___ _ __ | |_
 | |_| | | | |/ _` |/ _` | | '_ \ / _` | | |_ / _` |/ __/ _ \   / _ \ / _` |/ _ \ '_ \| __|
 |  _  | |_| | (_| | (_| | | | | | (_| | |  _| (_| | (_|  __/  / ___ \ (_| |  __/ | | | |_
 |_| |_|\__,_|\__, |\__, |_|_| |_|\__, | |_|  \__,_|\___\___| /_/   \_\__, |\___|_| |_|\__|
              |___/ |___/         |___/                               |___/
    """

    print(format_separator())
    print(f"{Colors.YELLOW} {banner}{Colors.RESET}")
    print("Type your messages below. Type /help for commands, /quit to exit.\n")
    print(format_separator())
    # Wait for agent to initialize
    print("Initializing agent...")

    # HF token
    hf_token = _get_hf_token()
    if hf_token:
        print("HF token loaded")
    else:
        print("Warning: No HF token found. Set HF_TOKEN or run `huggingface-cli login`.")

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
    print(f"Loading MCP servers: {', '.join(config.mcpServers.keys())}")
    tool_router = ToolRouter(config.mcpServers, hf_token=hf_token, local_mode=True)

    # Create prompt session for input
    prompt_session = PromptSession()

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

    try:
        while True:
            # Wait for previous turn to complete, with interrupt support
            try:
                await turn_complete_event.wait()
            except asyncio.CancelledError:
                break
            turn_complete_event.clear()

            # Get user input
            try:
                user_input = await get_user_input(prompt_session)
            except EOFError:
                break
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - last_interrupt_time < 3.0:
                    print("\nDouble Ctrl+C, exiting...")
                    break
                last_interrupt_time = now
                # If agent is busy, cancel it
                session = session_holder[0]
                if session and not turn_complete_event.is_set():
                    session.cancel()
                    print("\nInterrupting agent...")
                else:
                    print("\n(Ctrl+C again within 3s to exit)")
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
        print("\n\nInterrupted by user")

    # Shutdown
    print("\nShutting down agent...")
    shutdown_submission = Submission(
        id="sub_shutdown", operation=Operation(op_type=OpType.SHUTDOWN)
    )
    await submission_queue.put(shutdown_submission)

    try:
        await asyncio.wait_for(agent_task, timeout=5.0)
    except asyncio.TimeoutError:
        agent_task.cancel()
    listener_task.cancel()

    print("Goodbye!\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
