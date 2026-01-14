"""loop
Main agent implementation with integrated tool system and MCP support
"""

import asyncio
import json

from litellm import ChatCompletionMessageToolCall, Message, ModelResponse, acompletion
from lmnr import observe

from agent.config import Config
from agent.core.session import Event, OpType, Session
from agent.core.tools import ToolRouter
from agent.tools.jobs_tool import CPU_FLAVORS

ToolCall = ChatCompletionMessageToolCall


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


def _needs_approval(tool_name: str, tool_args: dict, config: Config | None = None) -> bool:
    """Check if a tool call requires user approval before execution."""
    # Yolo mode: skip all approvals
    if config and config.yolo_mode:
        return False

    # If args are malformed, skip approval (validation error will be shown later)
    args_valid, _ = _validate_tool_args(tool_args)
    if not args_valid:
        return False

    if tool_name == "hf_jobs":
        operation = tool_args.get("operation", "")
        if operation not in ["run", "uv", "scheduled run", "scheduled uv"]:
            return False
        
        # Check if this is a CPU-only job
        # hardware_flavor is at top level of tool_args, not nested in args
        hardware_flavor = tool_args.get("hardware_flavor") or tool_args.get("flavor") or tool_args.get("hardware") or "cpu-basic"
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
        if operation in ["delete_branch", "delete_tag", "merge_pr", "create_repo", "update_repo"]:
            return True

    return False


class Handlers:
    """Handler functions for each operation type"""

    @staticmethod
    @observe(name="run_agent")
    async def run_agent(
        session: Session, text: str, max_iterations: int = 10
    ) -> str | None:
        """
        Handle user input (like user_input_or_turn in codex.rs:1291)
        Returns the final assistant response content, if any.
        """
        # Set session ID for this trace
        if hasattr(session, "session_id"):
            from lmnr import Laminar

            Laminar.set_trace_session_id(session_id=session.session_id)

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

        while iteration < max_iterations:
            messages = session.context_manager.get_messages()
            tools = session.tool_router.get_tool_specs_for_llm()

            try:
                response: ModelResponse = await acompletion(
                    model=session.config.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )

                # Extract text response, token usage, and tool calls
                message = response.choices[0].message
                content = message.content
                token_count = response.usage.total_tokens
                tool_calls: list[ToolCall] = message.get("tool_calls", [])

                # If no tool calls, add assistant message and we're done
                if not tool_calls:
                    if content:
                        assistant_msg = Message(role="assistant", content=content)
                        session.context_manager.add_message(assistant_msg, token_count)
                        await session.send_event(
                            Event(
                                event_type="assistant_message",
                                data={"content": content},
                            )
                        )
                        final_response = content
                    break

                # Add assistant message with tool calls to history
                # LiteLLM will format this correctly for the provider
                assistant_msg = Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
                session.context_manager.add_message(assistant_msg, token_count)

                if content:
                    await session.send_event(
                        Event(event_type="assistant_message", data={"content": content})
                    )

                # Separate tools into those requiring approval and those that don't
                approval_required_tools = []
                non_approval_tools = []

                for tc in tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    if _needs_approval(tool_name, tool_args, session.config):
                        approval_required_tools.append(tc)
                    else:
                        non_approval_tools.append(tc)

                # Execute non-approval tools first
                for tc in non_approval_tools:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    # Validate tool arguments before calling
                    args_valid, error_msg = _validate_tool_args(tool_args)
                    if not args_valid:
                        # Return error to agent instead of calling tool
                        output = error_msg
                        success = False
                    else:
                        await session.send_event(
                            Event(
                                event_type="tool_call",
                                data={"tool": tool_name, "arguments": tool_args},
                            )
                        )

                        output, success = await session.tool_router.call_tool(
                            tool_name, tool_args
                        )

                    # Add tool result to history
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
                                "output": output,
                                "success": success,
                            },
                        )
                    )

                # If there are tools requiring approval, ask for batch approval
                if approval_required_tools:
                    # Prepare batch approval data
                    tools_data = []
                    for tc in approval_required_tools:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)
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

            except Exception as e:
                import traceback

                await session.send_event(
                    Event(
                        event_type="error",
                        data={"error": str(e) + "\n" + traceback.format_exc()},
                    )
                )
                break

        old_length = session.context_manager.context_length
        await session.context_manager.compact(model_name=session.config.model_name)
        new_length = session.context_manager.context_length

        if new_length != old_length:
            await session.send_event(
                Event(
                    event_type="compacted",
                    data={"old_tokens": old_length, "new_tokens": new_length},
                )
            )

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
    async def interrupt(session: Session) -> None:
        """Handle interrupt (like interrupt in codex.rs:1266)"""
        session.interrupt()
        await session.send_event(Event(event_type="interrupted"))

    @staticmethod
    async def compact(session: Session) -> None:
        """Handle compact (like compact in codex.rs:1317)"""
        old_length = session.context_manager.context_length
        await session.context_manager.compact(model_name=session.config.model_name)
        new_length = session.context_manager.context_length

        await session.send_event(
            Event(
                event_type="compacted",
                data={"removed": old_length, "remaining": new_length},
            )
        )

    @staticmethod
    async def undo(session: Session) -> None:
        """Handle undo (like undo in codex.rs:1314)"""
        # Remove last user turn and all following items
        # Simplified: just remove last 2 items
        for _ in range(min(2, len(session.context_manager.items))):
            session.context_manager.items.pop()

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

        # Separate approved and rejected tool calls
        approved_tasks = []
        rejected_tasks = []

        for tc in tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            approval_decision = approval_map.get(tc.id, {"approved": False})

            if approval_decision.get("approved", False):
                approved_tasks.append((tc, tool_name, tool_args))
            else:
                rejected_tasks.append((tc, tool_name, approval_decision))

        # Execute all approved tools concurrently
        async def execute_tool(tc, tool_name, tool_args):
            """Execute a single tool and return its result"""
            await session.send_event(
                Event(
                    event_type="tool_call",
                    data={"tool": tool_name, "arguments": tool_args},
                )
            )

            output, success = await session.tool_router.call_tool(tool_name, tool_args)

            return (tc, tool_name, output, success)

        # Execute all approved tools concurrently and wait for ALL to complete
        if approved_tasks:
            results = await asyncio.gather(
                *[
                    execute_tool(tc, tool_name, tool_args)
                    for tc, tool_name, tool_args in approved_tasks
                ],
                return_exceptions=True,
            )

            # Process results and add to context
            for result in results:
                if isinstance(result, Exception):
                    # Handle execution error
                    print(f"Tool execution error: {result}")
                    continue

                tc, tool_name, output, success = result

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
                rejection_msg += f". User feedback: {user_feedback}"

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
            print("💾 Saving session...")
            repo_id = session.config.session_dataset_repo
            _ = session.save_and_upload_detached(repo_id)
            # if local_path:
            # print("✅ Session saved locally, upload in progress")

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
    # print(f"📨 Received: {op.op_type.value}")

    if op.op_type == OpType.USER_INPUT:
        text = op.data.get("text", "") if op.data else ""
        await Handlers.run_agent(session, text)
        return True

    if op.op_type == OpType.INTERRUPT:
        await Handlers.interrupt(session)
        return True

    if op.op_type == OpType.COMPACT:
        await Handlers.compact(session)
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

    print(f"⚠️  Unknown operation: {op.op_type}")
    return True


@observe(name="submission_loop")
async def submission_loop(
    submission_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: Config | None = None,
    tool_router: ToolRouter | None = None,
) -> None:
    """
    Main agent loop - processes submissions and dispatches to handlers.
    This is the core of the agent (like submission_loop in codex.rs:1259-1340)
    """

    # Create session with tool router
    session = Session(event_queue, config=config, tool_router=tool_router)
    print("Agent loop started")

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
                    print("\n⚠️  Agent loop cancelled")
                    break
                except Exception as e:
                    print(f"❌ Error in agent loop: {e}")
                    await session.send_event(
                        Event(event_type="error", data={"error": str(e)})
                    )

        print("🛑 Agent loop exited")

    finally:
        # Emergency save if session saving is enabled and shutdown wasn't called properly
        if session.config.save_sessions and session.is_running:
            print("\n💾 Emergency save: preserving session before exit...")
            try:
                local_path = session.save_and_upload_detached(
                    session.config.session_dataset_repo
                )
                if local_path:
                    print("✅ Emergency save successful, upload in progress")
            except Exception as e:
                print(f"❌ Emergency save failed: {e}")
