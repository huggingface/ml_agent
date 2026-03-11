"""loop
Main agent implementation with integrated tool system and MCP support
"""

import asyncio
import json
import logging
import os

from litellm import ChatCompletionMessageToolCall, Message, acompletion
from litellm.exceptions import ContextWindowExceededError

from agent.config import Config
from agent.core.session import Event, OpType, Session
from agent.core.tools import ToolRouter
from agent.tools.jobs_tool import CPU_FLAVORS

logger = logging.getLogger(__name__)

ToolCall = ChatCompletionMessageToolCall
# Explicit inference token for LLM API calls (separate from user OAuth tokens).
_INFERENCE_API_KEY = os.environ.get("INFERENCE_TOKEN")


def _resolve_hf_router_params(model_name: str) -> dict:
    """
    Build LiteLLM kwargs for HuggingFace Router models.

    api-inference.huggingface.co is deprecated; the new router lives at
    router.huggingface.co/<provider>/v3/openai.  LiteLLM's built-in
    ``huggingface/`` provider still targets the old endpoint, so we
    rewrite model names to ``openai/`` and supply the correct api_base.

    Input format:  huggingface/<router_provider>/<org>/<model>
    Example:       huggingface/novita/moonshotai/kimi-k2.5
    """
    if not model_name.startswith("huggingface/"):
        return {"model": model_name}

    parts = model_name.split("/", 2)  # ['huggingface', 'novita', 'moonshotai/kimi-k2.5']
    if len(parts) < 3:
        return {"model": model_name}

    router_provider = parts[1]
    actual_model = parts[2]
    api_key = _INFERENCE_API_KEY

    return {
        "model": f"openai/{actual_model}",
        "api_base": f"https://router.huggingface.co/{router_provider}/v3/openai",
        "api_key": api_key,
    }


def _validate_tool_args(tool_args: dict) -> tuple[bool, str | None]:
    """
    Validate tool arguments structure.

    Returns:
        (is_valid, error_message)
    """
    args = tool_args.get("args", {})
    # Sometimes LLM passes args as string instead of dict
    if isinstance(args, str):
        return (
            False,
            f"Tool call error: 'args' must be a JSON object, not a string. You passed: {repr(args)}",
        )
    if not isinstance(args, dict) and args is not None:
        return (
            False,
            f"Tool call error: 'args' must be a JSON object. You passed type: {type(args).__name__}",
        )
    return True, None


def _needs_approval(
    tool_name: str, tool_args: dict, config: Config | None = None
) -> bool:
    """Check if a tool call requires user approval before execution."""
    # Yolo mode: skip all approvals
    if config and config.yolo_mode:
        return False

    # If args are malformed, skip approval (validation error will be shown later)
    args_valid, _ = _validate_tool_args(tool_args)
    if not args_valid:
        return False

    if tool_name == "sandbox_create":
        return True

    if tool_name == "hf_jobs":
        operation = tool_args.get("operation", "")
        if operation not in ["run", "uv", "scheduled run", "scheduled uv"]:
            return False

        # Check if this is a CPU-only job
        # hardware_flavor is at top level of tool_args, not nested in args
        hardware_flavor = (
            tool_args.get("hardware_flavor")
            or tool_args.get("flavor")
            or tool_args.get("hardware")
            or "cpu-basic"
        )
        is_cpu_job = hardware_flavor in CPU_FLAVORS

        if is_cpu_job:
            if config and not config.confirm_cpu_jobs:
                return False
            return True

        return True

    # Check for file upload operations (hf_private_repos or other tools)
    if tool_name == "hf_private_repos":
        operation = tool_args.get("operation", "")
        if operation == "upload_file":
            if config and config.auto_file_upload:
                return False
            return True
        # Other operations (create_repo, etc.) always require approval
        if operation in ["create_repo"]:
            return True

    # hf_repo_files: upload (can overwrite) and delete require approval
    if tool_name == "hf_repo_files":
        operation = tool_args.get("operation", "")
        if operation in ["upload", "delete"]:
            return True

    # hf_repo_git: destructive operations require approval
    if tool_name == "hf_repo_git":
        operation = tool_args.get("operation", "")
        if operation in [
            "delete_branch",
            "delete_tag",
            "merge_pr",
            "create_repo",
            "update_repo",
        ]:
            return True

    return False


async def _compact_and_notify(session: Session) -> None:
    """Run compaction and send event if context was reduced."""
    old_length = session.context_manager.context_length
    tool_specs = session.tool_router.get_tool_specs_for_llm()
    await session.context_manager.compact(
        model_name=session.config.model_name,
        tool_specs=tool_specs,
    )
    new_length = session.context_manager.context_length
    if new_length != old_length:
        await session.send_event(
            Event(
                event_type="compacted",
                data={"old_tokens": old_length, "new_tokens": new_length},
            )
        )




class Handlers:
    """Handler functions for each operation type"""

    @staticmethod
    async def _abandon_pending_approval(session: Session) -> None:
        """Cancel pending approval tools when the user continues the conversation.

        Injects rejection tool-result messages into the LLM context (so the
        history stays valid) and notifies the frontend that those tools were
        abandoned.
        """
        tool_calls = session.pending_approval.get("tool_calls", [])
        for tc in tool_calls:
            tool_name = tc.function.name
            abandon_msg = "Task abandoned — user continued the conversation without approving."

            # Keep LLM context valid: every tool_call needs a tool result
            tool_msg = Message(
                role="tool",
                content=abandon_msg,
                tool_call_id=tc.id,
                name=tool_name,
            )
            session.context_manager.add_message(tool_msg)

            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "abandoned",
                    },
                )
            )

        session.pending_approval = None
        logger.info("Abandoned %d pending approval tool(s)", len(tool_calls))

    @staticmethod
    async def run_agent(
        session: Session, text: str, max_iterations: int = 300
    ) -> str | None:
        """
        Handle user input (like user_input_or_turn in codex.rs:1291)
        Returns the final assistant response content, if any.
        """
        # Clear any stale cancellation flag from a previous run
        session.reset_cancel()

        # If there's a pending approval and the user sent a new message,
        # abandon the pending tools so the LLM context stays valid.
        if text and session.pending_approval:
            await Handlers._abandon_pending_approval(session)

        # Add user message to history only if there's actual content
        if text:
            user_msg = Message(role="user", content=text)
            session.context_manager.add_message(user_msg)

        # Send event that we're processing
        await session.send_event(
            Event(event_type="processing", data={"message": "Processing user input"})
        )

        # Agentic loop - continue until model doesn't call tools or max iterations is reached
        iteration = 0
        final_response = None
        errored = False

        while iteration < max_iterations:
            # ── Cancellation check: before LLM call ──
            if session.is_cancelled:
                break

            # Compact before calling the LLM if context is near the limit
            await _compact_and_notify(session)

            messages = session.context_manager.get_messages()
            tools = session.tool_router.get_tool_specs_for_llm()
            try:
                # ── Stream the LLM response ──────────────────────────
                llm_params = _resolve_hf_router_params(session.config.model_name)
                response = await acompletion(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                    stream_options={"include_usage": True},
                    max_completion_tokens=16384,
                    **llm_params,
                )

                full_content = ""
                tool_calls_acc: dict[int, dict] = {}
                token_count = 0
                finish_reason = None

                async for chunk in response:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        # Last chunk may carry only usage info
                        if hasattr(chunk, "usage") and chunk.usage:
                            token_count = chunk.usage.total_tokens
                        continue

                    delta = choice.delta
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    # Stream text deltas to the frontend
                    if delta.content:
                        full_content += delta.content
                        await session.send_event(
                            Event(
                                event_type="assistant_chunk",
                                data={"content": delta.content},
                            )
                        )

                    # Accumulate tool-call deltas (name + args arrive in pieces)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[idx]["function"]["name"] += (
                                        tc_delta.function.name
                                    )
                                if tc_delta.function.arguments:
                                    tool_calls_acc[idx]["function"]["arguments"] += (
                                        tc_delta.function.arguments
                                    )

                    # Capture usage from the final chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        token_count = chunk.usage.total_tokens

                # ── Stream finished — reconstruct full message ───────
                content = full_content or None

                # Detect output truncation — tool call args will be
                # incomplete JSON, which would poison the context.
                if finish_reason == "length" and tool_calls_acc:
                    logger.warning(
                        "Output truncated (finish_reason=length) with pending "
                        "tool calls — dropping truncated tool calls"
                    )
                    truncation_notice = (
                        "\n\n[Output was truncated due to length. "
                        "Tool calls were dropped. Break large writes "
                        "into smaller pieces or write to sandbox in parts.]"
                    )
                    full_content += truncation_notice
                    content = full_content
                    await session.send_event(
                        Event(
                            event_type="assistant_chunk",
                            data={"content": truncation_notice},
                        )
                    )
                    tool_calls_acc.clear()

                # Build tool_calls list from accumulated deltas
                tool_calls: list[ToolCall] = []
                for idx in sorted(tool_calls_acc.keys()):
                    tc_data = tool_calls_acc[idx]
                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            type="function",
                            function={
                                "name": tc_data["function"]["name"],
                                "arguments": tc_data["function"]["arguments"],
                            },
                        )
                    )

                # Signal end of streaming to the frontend
                await session.send_event(
                    Event(event_type="assistant_stream_end", data={})
                )

                # If no tool calls, add assistant message and we're done
                if not tool_calls:
                    if content:
                        assistant_msg = Message(role="assistant", content=content)
                        session.context_manager.add_message(assistant_msg, token_count)
                        final_response = content
                    break

                # Add assistant message with tool calls to history
                assistant_msg = Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
                session.context_manager.add_message(assistant_msg, token_count)

                # ── Cancellation check: before tool execution ──
                if session.is_cancelled:
                    break

                # Separate tools into those requiring approval and those that don't
                approval_required_tools = []
                non_approval_tools = []

                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Malformed tool arguments for {tool_name}: {e}")
                        tool_args = {}

                    if _needs_approval(tool_name, tool_args, session.config):
                        approval_required_tools.append(tc)
                    else:
                        non_approval_tools.append(tc)
                # Execute non-approval tools (in parallel when possible)
                if non_approval_tools:
                    # 1. Parse args and validate upfront
                    parsed_tools: list[
                        tuple[ChatCompletionMessageToolCall, str, dict, bool, str]
                    ] = []
                    for tc in non_approval_tools:
                        tool_name = tc.function.name
                        try:
                            tool_args = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            tool_args = {}

                        args_valid, error_msg = _validate_tool_args(tool_args)
                        parsed_tools.append(
                            (tc, tool_name, tool_args, args_valid, error_msg)
                        )

                    # 2. Send all tool_call events upfront (so frontend shows them all)
                    for tc, tool_name, tool_args, args_valid, _ in parsed_tools:
                        if args_valid:
                            await session.send_event(
                                Event(
                                    event_type="tool_call",
                                    data={
                                        "tool": tool_name,
                                        "arguments": tool_args,
                                        "tool_call_id": tc.id,
                                    },
                                )
                            )

                    # 3. Execute all valid tools in parallel
                    async def _exec_tool(
                        tc: ChatCompletionMessageToolCall,
                        name: str,
                        args: dict,
                        valid: bool,
                        err: str,
                    ) -> tuple[ChatCompletionMessageToolCall, str, dict, str, bool]:
                        if not valid:
                            return (tc, name, args, err, False)
                        out, ok = await session.tool_router.call_tool(
                            name, args, session=session
                        )
                        return (tc, name, args, out, ok)

                    results = await asyncio.gather(
                        *[
                            _exec_tool(tc, name, args, valid, err)
                            for tc, name, args, valid, err in parsed_tools
                        ]
                    )

                    # 4. Record results and send outputs (order preserved)
                    for tc, tool_name, tool_args, output, success in results:
                        tool_msg = Message(
                            role="tool",
                            content=output,
                            tool_call_id=tc.id,
                            name=tool_name,
                        )
                        session.context_manager.add_message(tool_msg)

                        await session.send_event(
                            Event(
                                event_type="tool_output",
                                data={
                                    "tool": tool_name,
                                    "tool_call_id": tc.id,
                                    "output": output,
                                    "success": success,
                                },
                            )
                        )

                # ── Cancellation check: after tool execution ──
                if session.is_cancelled:
                    break

                # If there are tools requiring approval, ask for batch approval
                if approval_required_tools:
                    # Prepare batch approval data
                    tools_data = []
                    for tc in approval_required_tools:
                        tool_name = tc.function.name
                        try:
                            tool_args = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            tool_args = {}

                        # Resolve sandbox file paths for hf_jobs scripts so the
                        # frontend can display & edit the actual file content.
                        if tool_name == "hf_jobs" and isinstance(tool_args.get("script"), str):
                            from agent.tools.sandbox_tool import resolve_sandbox_script
                            sandbox = getattr(session, "sandbox", None)
                            content, _ = await resolve_sandbox_script(sandbox, tool_args["script"])
                            if content:
                                tool_args = {**tool_args, "script": content}

                        tools_data.append(
                            {
                                "tool": tool_name,
                                "arguments": tool_args,
                                "tool_call_id": tc.id,
                            }
                        )

                    await session.send_event(
                        Event(
                            event_type="approval_required",
                            data={
                                "tools": tools_data,  # Batch of tools
                                "count": len(tools_data),
                            },
                        )
                    )

                    # Store all approval-requiring tools
                    session.pending_approval = {
                        "tool_calls": approval_required_tools,
                    }

                    # Return early - wait for EXEC_APPROVAL operation
                    return None

                iteration += 1

            except ContextWindowExceededError:
                # Force compact and retry this iteration
                session.context_manager.context_length = (
                    session.context_manager.max_context + 1
                )
                await _compact_and_notify(session)
                continue

            except Exception as e:
                import traceback

                await session.send_event(
                    Event(
                        event_type="error",
                        data={"error": str(e) + "\n" + traceback.format_exc()},
                    )
                )
                errored = True
                break

        if session.is_cancelled:
            await session.send_event(Event(event_type="interrupted"))
        elif not errored:
            await session.send_event(
                Event(
                    event_type="turn_complete",
                    data={"history_size": len(session.context_manager.items)},
                )
            )

        # Increment turn counter and check for auto-save
        session.increment_turn()
        await session.auto_save_if_needed()

        return final_response

    @staticmethod
    async def undo(session: Session) -> None:
        """Remove the last complete turn and notify the frontend."""
        removed = session.context_manager.undo_last_turn()
        if not removed:
            logger.warning("Undo: no user message found to remove")
        await session.send_event(Event(event_type="undo_complete"))

    @staticmethod
    async def exec_approval(session: Session, approvals: list[dict]) -> None:
        """Handle batch job execution approval"""
        if not session.pending_approval:
            await session.send_event(
                Event(
                    event_type="error",
                    data={"error": "No pending approval to process"},
                )
            )
            return

        tool_calls = session.pending_approval.get("tool_calls", [])
        if not tool_calls:
            await session.send_event(
                Event(
                    event_type="error",
                    data={"error": "No pending tool calls found"},
                )
            )
            return

        # Create a map of tool_call_id -> approval decision
        approval_map = {a["tool_call_id"]: a for a in approvals}
        for a in approvals:
            if a.get("edited_script"):
                logger.info(f"Received edited script for tool_call {a['tool_call_id']} ({len(a['edited_script'])} chars)")

        # Separate approved and rejected tool calls
        approved_tasks = []
        rejected_tasks = []

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError) as e:
                # Malformed arguments — treat as failed, notify agent
                logger.warning(f"Malformed tool arguments for {tool_name}: {e}")
                tool_msg = Message(
                    role="tool",
                    content=f"Malformed arguments: {e}",
                    tool_call_id=tc.id,
                    name=tool_name,
                )
                session.context_manager.add_message(tool_msg)
                await session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool": tool_name,
                            "tool_call_id": tc.id,
                            "output": f"Malformed arguments: {e}",
                            "success": False,
                        },
                    )
                )
                continue

            approval_decision = approval_map.get(tc.id, {"approved": False})

            if approval_decision.get("approved", False):
                edited_script = approval_decision.get("edited_script")
                was_edited = False
                if edited_script and "script" in tool_args:
                    tool_args["script"] = edited_script
                    was_edited = True
                    logger.info(f"Using user-edited script for {tool_name} ({tc.id})")
                approved_tasks.append((tc, tool_name, tool_args, was_edited))
            else:
                rejected_tasks.append((tc, tool_name, approval_decision))

        # Notify frontend of approval decisions immediately (before execution)
        for tc, tool_name, tool_args, _was_edited in approved_tasks:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "approved",
                    },
                )
            )
        for tc, tool_name, approval_decision in rejected_tasks:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "rejected",
                    },
                )
            )

        # Execute all approved tools concurrently
        async def execute_tool(tc, tool_name, tool_args, was_edited):
            """Execute a single tool and return its result.

            The TraceLog already exists on the frontend (created by
            approval_required), so we send tool_state_change instead of
            tool_call to avoid creating a duplicate.
            """
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tool_name,
                        "state": "running",
                    },
                )
            )

            output, success = await session.tool_router.call_tool(
                tool_name, tool_args, session=session, tool_call_id=tc.id
            )

            return (tc, tool_name, output, success, was_edited)

        # Execute all approved tools concurrently and wait for ALL to complete
        if approved_tasks:
            results = await asyncio.gather(
                *[
                    execute_tool(tc, tool_name, tool_args, was_edited)
                    for tc, tool_name, tool_args, was_edited in approved_tasks
                ],
                return_exceptions=True,
            )

            # Process results and add to context
            for result in results:
                if isinstance(result, Exception):
                    # Handle execution error
                    logger.error(f"Tool execution error: {result}")
                    continue

                tc, tool_name, output, success, was_edited = result

                if was_edited:
                    output = f"[Note: The user edited the script before execution. The output below reflects the user-modified version, not your original script.]\n\n{output}"

                # Add tool result to context
                tool_msg = Message(
                    role="tool",
                    content=output,
                    tool_call_id=tc.id,
                    name=tool_name,
                )
                session.context_manager.add_message(tool_msg)

                await session.send_event(
                    Event(
                        event_type="tool_output",
                        data={
                            "tool": tool_name,
                            "tool_call_id": tc.id,
                            "output": output,
                            "success": success,
                        },
                    )
                )

        # Process rejected tools
        for tc, tool_name, approval_decision in rejected_tasks:
            rejection_msg = "Job execution cancelled by user"
            user_feedback = approval_decision.get("feedback")
            if user_feedback:
                # Ensure feedback is a string and sanitize any problematic characters
                feedback_str = str(user_feedback).strip()
                # Remove any control characters that might break JSON parsing
                feedback_str = "".join(char for char in feedback_str if ord(char) >= 32 or char in "\n\t")
                rejection_msg += f". User feedback: {feedback_str}"

            # Ensure rejection_msg is a clean string
            rejection_msg = str(rejection_msg).strip()

            tool_msg = Message(
                role="tool",
                content=rejection_msg,
                tool_call_id=tc.id,
                name=tool_name,
            )
            session.context_manager.add_message(tool_msg)

            await session.send_event(
                Event(
                    event_type="tool_output",
                    data={
                        "tool": tool_name,
                        "tool_call_id": tc.id,
                        "output": rejection_msg,
                        "success": False,
                    },
                )
            )

        # Clear pending approval
        session.pending_approval = None

        # Continue agent loop with empty input to process the tool results
        await Handlers.run_agent(session, "")

    @staticmethod
    async def shutdown(session: Session) -> bool:
        """Handle shutdown (like shutdown in codex.rs:1329)"""
        # Save session trajectory if enabled (fire-and-forget, returns immediately)
        if session.config.save_sessions:
            logger.info("Saving session...")
            repo_id = session.config.session_dataset_repo
            _ = session.save_and_upload_detached(repo_id)

        session.is_running = False
        await session.send_event(Event(event_type="shutdown"))
        return True


async def process_submission(session: Session, submission) -> bool:
    """
    Process a single submission and return whether to continue running.

    Returns:
        bool: True to continue, False to shutdown
    """
    op = submission.operation
    logger.debug("Received operation: %s", op.op_type.value)

    if op.op_type == OpType.USER_INPUT:
        text = op.data.get("text", "") if op.data else ""
        await Handlers.run_agent(session, text)
        return True

    if op.op_type == OpType.COMPACT:
        await _compact_and_notify(session)
        return True

    if op.op_type == OpType.UNDO:
        await Handlers.undo(session)
        return True

    if op.op_type == OpType.EXEC_APPROVAL:
        approvals = op.data.get("approvals", []) if op.data else []
        await Handlers.exec_approval(session, approvals)
        return True

    if op.op_type == OpType.SHUTDOWN:
        return not await Handlers.shutdown(session)

    logger.warning(f"Unknown operation: {op.op_type}")
    return True


async def submission_loop(
    submission_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: Config | None = None,
    tool_router: ToolRouter | None = None,
    session_holder: list | None = None,
    hf_token: str | None = None,
) -> None:
    """
    Main agent loop - processes submissions and dispatches to handlers.
    This is the core of the agent (like submission_loop in codex.rs:1259-1340)
    """

    # Create session with tool router
    session = Session(event_queue, config=config, tool_router=tool_router, hf_token=hf_token)
    if session_holder is not None:
        session_holder[0] = session
    logger.info("Agent loop started")

    # Retry any failed uploads from previous sessions (fire-and-forget)
    if config and config.save_sessions:
        Session.retry_failed_uploads_detached(
            directory="session_logs", repo_id=config.session_dataset_repo
        )

    try:
        # Main processing loop
        async with tool_router:
            # Emit ready event after initialization
            await session.send_event(
                Event(event_type="ready", data={"message": "Agent initialized"})
            )

            while session.is_running:
                submission = await submission_queue.get()

                try:
                    should_continue = await process_submission(session, submission)
                    if not should_continue:
                        break
                except asyncio.CancelledError:
                    logger.warning("Agent loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in agent loop: {e}")
                    await session.send_event(
                        Event(event_type="error", data={"error": str(e)})
                    )

        logger.info("Agent loop exited")

    finally:
        # Emergency save if session saving is enabled and shutdown wasn't called properly
        if session.config.save_sessions and session.is_running:
            logger.info("Emergency save: preserving session before exit...")
            try:
                local_path = session.save_and_upload_detached(
                    session.config.session_dataset_repo
                )
                if local_path:
                    logger.info("Emergency save successful, upload in progress")
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")
