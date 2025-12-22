"""
Minimum Viable Implementation of Codex Agent Loop in Python

This demonstrates the core architecture patterns from codex-rs:
- Async submission loop (like submission_loop in codex.rs)
- Context manager for conversation history
- Channel-based communication (submissions in, events out)
- Handler pattern for operations
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# ============================================================================
# PROTOCOL TYPES (ResponseItem equivalents)
# ============================================================================


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolOutput:
    call_id: str
    content: str
    success: bool = True


# ============================================================================
# CONTEXT MANAGER (like context_manager/history.rs)
# ============================================================================


class ContextManager:
    """
    Manages conversation history with normalization and truncation.
    Based on codex-rs/core/src/context_manager/history.rs
    """

    def __init__(self, max_history_length: int = 1000):
        self.items: List[Any] = []  # Oldest → Newest
        self.token_count: int = 0
        self.max_history_length = max_history_length

    def record_items(self, items: List[Any]) -> None:
        """Record new items to history (like record_items in history.rs:41)"""
        for item in items:
            # Filter and process items
            if self._is_api_message(item):
                processed = self._process_item(item)
                self.items.append(processed)

    def _is_api_message(self, item: Any) -> bool:
        """Filter out system messages (like is_api_message in history.rs:157)"""
        if isinstance(item, Message):
            return item.role != MessageRole.SYSTEM
        return isinstance(item, (ToolCall, ToolOutput))

    def _process_item(self, item: Any) -> Any:
        """Process item before adding (like process_item in history.rs:119)"""
        # Truncate long outputs
        if isinstance(item, ToolOutput):
            if len(item.content) > 2000:
                item.content = item.content[:2000] + "...[truncated]"
        return item

    def get_history_for_prompt(self) -> List[Any]:
        """
        Get normalized history ready for model
        (like get_history_for_prompt in history.rs:65)
        """
        self._normalize_history()
        return self.items.copy()

    def _normalize_history(self) -> None:
        """
        Enforce invariants (like normalize_history in history.rs:102):
        1. Every tool call has corresponding output
        2. Every output has corresponding call
        """
        # Build mapping of call_id → call
        calls = {}
        outputs = {}

        for item in self.items:
            if isinstance(item, ToolCall):
                calls[item.call_id] = item
            elif isinstance(item, ToolOutput):
                outputs[item.call_id] = item

        # Remove orphan outputs (no matching call)
        self.items = [
            item
            for item in self.items
            if not isinstance(item, ToolOutput) or item.call_id in calls
        ]

        # Add missing outputs for calls (create synthetic outputs)
        for call_id, call in calls.items():
            if call_id not in outputs:
                self.items.append(
                    ToolOutput(
                        call_id=call_id, content="[No output recorded]", success=False
                    )
                )

    def remove_first_item(self) -> None:
        """Remove oldest item for compaction (like remove_first_item in history.rs:71)"""
        if self.items:
            removed = self.items.pop(0)
            # Also remove corresponding pair if needed
            if isinstance(removed, ToolCall):
                self.items = [
                    item
                    for item in self.items
                    if not (
                        isinstance(item, ToolOutput) and item.call_id == removed.call_id
                    )
                ]
            elif isinstance(removed, ToolOutput):
                self.items = [
                    item
                    for item in self.items
                    if not (
                        isinstance(item, ToolCall) and item.call_id == removed.call_id
                    )
                ]

    def compact(self, target_size: int) -> None:
        """Remove old items until we're under target size"""
        while len(self.items) > target_size:
            self.remove_first_item()


# ============================================================================
# OPERATIONS (like Op enum in codex.rs)
# ============================================================================


class OpType(Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    UNDO = "undo"
    COMPACT = "compact"
    SHUTDOWN = "shutdown"


@dataclass
class Operation:
    op_type: OpType
    data: Optional[Dict[str, Any]] = None


@dataclass
class Submission:
    id: str
    operation: Operation


# ============================================================================
# EVENTS (like Event in codex-rs)
# ============================================================================


@dataclass
class Event:
    event_type: str
    data: Optional[Dict[str, Any]] = None


# ============================================================================
# SESSION STATE (like Session in codex.rs)
# ============================================================================


class Session:
    """
    Maintains agent session state
    Similar to Session in codex-rs/core/src/codex.rs
    """

    def __init__(self, event_queue: asyncio.Queue):
        self.context_manager = ContextManager(tool_specs=[])
        self.event_queue = event_queue
        self.is_running = True
        self.current_task: Optional[asyncio.Task] = None

    async def send_event(self, event: Event) -> None:
        """Send event back to client"""
        await self.event_queue.put(event)

    def interrupt(self) -> None:
        """Interrupt current running task"""
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()


# ============================================================================
# OPERATION HANDLERS (like handlers module in codex.rs:1343)
# ============================================================================


class Handlers:
    """Handler functions for each operation type"""

    @staticmethod
    async def user_input(session: Session, text: str) -> None:
        """Handle user input (like user_input_or_turn in codex.rs:1291)"""
        # Add user message to history
        user_msg = Message(role=MessageRole.USER, content=text)
        session.context_manager.record_items([user_msg])

        # Send event that we're processing
        await session.send_event(
            Event(event_type="processing", data={"message": "Processing user input"})
        )

        # Simulate agent processing
        await asyncio.sleep(0.1)

        # Generate mock assistant response
        assistant_msg = Message(
            role=MessageRole.ASSISTANT, content=f"I received: {text}"
        )
        session.context_manager.record_items([assistant_msg])

        # Simulate tool call
        tool_call = ToolCall(
            call_id="call_123", tool_name="bash", arguments={"command": "echo 'hello'"}
        )
        session.context_manager.record_items([tool_call])

        # Simulate tool execution
        await asyncio.sleep(0.1)

        tool_output = ToolOutput(call_id="call_123", content="hello\n", success=True)
        session.context_manager.record_items([tool_output])

        # Send completion event
        await session.send_event(
            Event(
                event_type="turn_complete",
                data={"history_size": len(session.context_manager.items)},
            )
        )

    @staticmethod
    async def interrupt(session: Session) -> None:
        """Handle interrupt (like interrupt in codex.rs:1266)"""
        session.interrupt()
        await session.send_event(Event(event_type="interrupted"))

    @staticmethod
    async def compact(session: Session) -> None:
        """Handle compact (like compact in codex.rs:1317)"""
        old_size = len(session.context_manager.items)
        session.context_manager.compact(target_size=10)
        new_size = len(session.context_manager.items)

        await session.send_event(
            Event(
                event_type="compacted",
                data={"removed": old_size - new_size, "remaining": new_size},
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


# ============================================================================
# MAIN AGENT LOOP (like submission_loop in codex.rs:1259)
# ============================================================================


async def submission_loop(
    submission_queue: asyncio.Queue, event_queue: asyncio.Queue
) -> None:
    """
    Main agent loop - processes submissions and dispatches to handlers.
    This is the core of the agent (like submission_loop in codex.rs:1259-1340)
    """
    session = Session(event_queue)

    print("🤖 Agent loop started")

    # Main processing loop
    while session.is_running:
        try:
            # Wait for next submission (like rx_sub.recv() in codex.rs:1262)
            submission = await submission_queue.get()

            print(f"📨 Received: {submission.operation.op_type.value}")

            # Dispatch to handler based on operation type
            # (like match in codex.rs:1264-1337)
            op = submission.operation

            if op.op_type == OpType.USER_INPUT:
                text = op.data.get("text", "") if op.data else ""
                await Handlers.user_input(session, text)

            elif op.op_type == OpType.INTERRUPT:
                await Handlers.interrupt(session)

            elif op.op_type == OpType.COMPACT:
                await Handlers.compact(session)

            elif op.op_type == OpType.UNDO:
                await Handlers.undo(session)

            elif op.op_type == OpType.SHUTDOWN:
                if await Handlers.shutdown(session):
                    break

            else:
                print(f"⚠️  Unknown operation: {op.op_type}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Error in agent loop: {e}")
            await session.send_event(Event(event_type="error", data={"error": str(e)}))

    print("🛑 Agent loop exited")


# ============================================================================
# CODEX INTERFACE (like Codex struct in codex.rs:154)
# ============================================================================


class Codex:
    """
    Main interface to the agent (like Codex in codex.rs:154-246)
    Provides submit() and next_event() methods
    """

    def __init__(self):
        self.submission_queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        self.agent_task: Optional[asyncio.Task] = None
        self.submission_counter = 0

    async def spawn(self) -> None:
        """Spawn the agent loop (like Codex::spawn in codex.rs:156)"""
        self.agent_task = asyncio.create_task(
            submission_loop(self.submission_queue, self.event_queue)
        )

    async def submit(self, operation: Operation) -> str:
        """Submit operation to agent (like Codex::submit in codex.rs:218)"""
        self.submission_counter += 1
        submission = Submission(
            id=f"sub_{self.submission_counter}", operation=operation
        )
        await self.submission_queue.put(submission)
        return submission.id

    async def next_event(self) -> Optional[Event]:
        """Get next event from agent (like Codex::next_event in codex.rs:238)"""
        try:
            return await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def shutdown(self) -> None:
        """Shutdown the agent"""
        await self.submit(Operation(op_type=OpType.SHUTDOWN))
        if self.agent_task:
            await self.agent_task


# ============================================================================
# DEMO / EXAMPLE USAGE
# ============================================================================


async def main():
    """Demo of the agent system"""
    print("=" * 60)
    print("Codex Agent Loop Demo (Python MVP)")
    print("=" * 60)

    # Create and spawn agent
    codex = Codex()
    await codex.spawn()

    # Submit some operations
    print("\n1️⃣  Submitting user input...")
    await codex.submit(
        Operation(op_type=OpType.USER_INPUT, data={"text": "Hello, agent!"})
    )

    # Receive events
    for _ in range(3):
        event = await codex.next_event()
        if event:
            print(f"   ✅ Event: {event.event_type} - {event.data}")

    print("\n2️⃣  Submitting another input...")
    await codex.submit(
        Operation(op_type=OpType.USER_INPUT, data={"text": "What's the weather?"})
    )

    for _ in range(3):
        event = await codex.next_event()
        if event:
            print(f"   ✅ Event: {event.event_type} - {event.data}")

    print("\n3️⃣  Compacting history...")
    await codex.submit(Operation(op_type=OpType.COMPACT))

    event = await codex.next_event()
    if event:
        print(f"   ✅ Event: {event.event_type} - {event.data}")

    print("\n4️⃣  Undoing last turn...")
    await codex.submit(Operation(op_type=OpType.UNDO))

    event = await codex.next_event()
    if event:
        print(f"   ✅ Event: {event.event_type}")

    # Shutdown
    print("\n5️⃣  Shutting down...")
    await codex.shutdown()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
