"""
Interactive CLI chat with the agent
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import litellm
from lmnr import Laminar, LaminarLiteLLMCallback

from agent.config import load_config
from agent.core.agent_loop import submission_loop
from agent.core.session import OpType
from agent.core.tools import ToolRouter

litellm.drop_params = True

lmnr_api_key = os.environ.get("LMNR_API_KEY")
if lmnr_api_key:
    try:
        Laminar.initialize(project_api_key=lmnr_api_key)
        litellm.callbacks = [LaminarLiteLLMCallback()]
        print("✅ Laminar initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Laminar: {e}")


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
    turn_complete_event: asyncio.Event,
    ready_event: asyncio.Event,
) -> None:
    """Background task that listens for events and displays them"""
    while True:
        try:
            event = await event_queue.get()

            # Display event
            if event.event_type == "ready":
                print("✅ Agent ready")
                ready_event.set()
            elif event.event_type == "assistant_message":
                content = event.data.get("content", "") if event.data else ""
                if content:
                    print(f"\n🤖 Assistant: {content}")
            elif event.event_type == "tool_call":
                tool_name = event.data.get("tool", "") if event.data else ""
                arguments = event.data.get("arguments", {}) if event.data else {}
                if tool_name:
                    print(
                        f"🔧 Calling tool: {tool_name} with arguments: {json.dumps(arguments)[:100]}..."
                    )
            elif event.event_type == "tool_output":
                output = event.data.get("output", "") if event.data else ""
                success = event.data.get("success", False) if event.data else False
                status = "✅" if success else "❌"
                if output:
                    print(f"{status} Tool output: {output}")
            elif event.event_type == "turn_complete":
                print("✅ Turn complete\n")
                turn_complete_event.set()
            elif event.event_type == "error":
                error = (
                    event.data.get("error", "Unknown error")
                    if event.data
                    else "Unknown error"
                )
                print(f"❌ Error: {error}")
                turn_complete_event.set()
            elif event.event_type == "shutdown":
                print("🛑 Agent shutdown")
                break
            elif event.event_type == "processing":
                print("⏳ Processing...", flush=True)
            elif event.event_type == "compacted":
                old_tokens = event.data.get("old_tokens", 0) if event.data else 0
                new_tokens = event.data.get("new_tokens", 0) if event.data else 0
                print(f"📦 Compacted context: {old_tokens} → {new_tokens} tokens")
            # Silently ignore other events

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ Event listener error: {e}")


async def get_user_input() -> str:
    """Get user input asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, "You: ")


async def main():
    """Interactive chat with the agent"""
    print("=" * 60)
    print("🤖 Interactive Agent Chat")
    print("=" * 60)
    print("Type your messages below. Type 'exit', 'quit', or '/quit' to end.\n")

    # Create queues for communication
    submission_queue = asyncio.Queue()
    event_queue = asyncio.Queue()

    # Events to signal agent state
    turn_complete_event = asyncio.Event()
    turn_complete_event.set()
    ready_event = asyncio.Event()

    # Start agent loop in background
    config_path = Path(__file__).parent / "config_mcp_example.json"
    config = load_config(config_path)

    # Create tool router
    print(f"Config: {config.mcpServers}")
    tool_router = ToolRouter(config.mcpServers)

    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
        )
    )

    # Start event listener in background
    listener_task = asyncio.create_task(
        event_listener(event_queue, turn_complete_event, ready_event)
    )

    # Wait for agent to initialize
    print("⏳ Initializing agent...")
    await ready_event.wait()

    submission_id = 0

    try:
        while True:
            # Wait for previous turn to complete
            await turn_complete_event.wait()
            turn_complete_event.clear()

            # Get user input
            try:
                user_input = await get_user_input()
            except EOFError:
                break

            # Check for exit commands
            if user_input.strip().lower() in ["exit", "quit", "/quit", "/exit"]:
                break

            # Skip empty input
            if not user_input.strip():
                turn_complete_event.set()
                continue

            # Submit to agent
            submission_id += 1
            submission = Submission(
                id=f"sub_{submission_id}",
                operation=Operation(
                    op_type=OpType.USER_INPUT, data={"text": user_input}
                ),
            )
            print(f"Main submitting: {submission.operation.op_type}")
            await submission_queue.put(submission)

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")

    # Shutdown
    print("\n🛑 Shutting down agent...")
    shutdown_submission = Submission(
        id="sub_shutdown", operation=Operation(op_type=OpType.SHUTDOWN)
    )
    await submission_queue.put(shutdown_submission)

    # Wait for tasks to complete
    await asyncio.wait_for(agent_task, timeout=2.0)
    listener_task.cancel()

    print("✨ Goodbye!\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✨ Goodbye!")
