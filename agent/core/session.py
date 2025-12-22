import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from litellm import get_max_tokens

from agent.config import Config
from agent.context_manager.manager import ContextManager


class OpType(Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    UNDO = "undo"
    COMPACT = "compact"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    event_type: str
    data: Optional[dict[str, Any]] = None


class Session:
    """
    Maintains agent session state
    Similar to Session in codex-rs/core/src/codex.rs
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        config: Config | None = None,
        tool_router=None,
    ):
        self.tool_router = tool_router
        tool_specs = tool_router.get_tool_specs_for_llm() if tool_router else []
        self.context_manager = ContextManager(
            max_context=get_max_tokens(config.model_name),
            compact_size=0.1,
            untouched_messages=5,
            tool_specs=tool_specs,
        )
        self.event_queue = event_queue
        self.session_id = str(uuid.uuid4())
        self.config = config or Config(
            model_name="anthropic/claude-sonnet-4-5-20250929",
            tools=[],
        )
        self.is_running = True
        self.current_task: asyncio.Task | None = None

    async def send_event(self, event: Event) -> None:
        """Send event back to client"""
        await self.event_queue.put(event)

    def interrupt(self) -> None:
        """Interrupt current running task"""
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
