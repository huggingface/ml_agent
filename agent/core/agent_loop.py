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

ToolCall = ChatCompletionMessageToolCall


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

        # Add user message to history
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

                # Execute tools
                for tc in tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

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
    async def shutdown(session: Session) -> bool:
        """Handle shutdown (like shutdown in codex.rs:1329)"""
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
    print(f"📨 Received: {op.op_type.value}")

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
    print("🤖 Agent loop started")

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
                break
            except Exception as e:
                print(f"❌ Error in agent loop: {e}")
                await session.send_event(
                    Event(event_type="error", data={"error": str(e)})
                )

    print("🛑 Agent loop exited")
