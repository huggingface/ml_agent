"""Session manager for handling multiple concurrent agent sessions."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent.config import load_config
from agent.core.agent_loop import process_submission
from agent.core.session import Event, OpType, Session
from agent.core.tools import ToolRouter

# Get project root (parent of backend directory)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "frontend_agent_config.json")


# These dataclasses match agent/main.py structure
@dataclass
class Operation:
    """Operation to be executed by the agent."""

    op_type: OpType
    data: Optional[dict[str, Any]] = None


@dataclass
class Submission:
    """Submission to the agent loop."""

    id: str
    operation: Operation


logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Reads from the agent's event queue and fans out to SSE subscribers.

    Events that arrive when no subscribers are listening are discarded.
    With SSE each turn is a separate request, so there is no reconnect
    scenario that would need buffered replay.
    """

    def __init__(self, event_queue: asyncio.Queue):
        self._source = event_queue
        self._subscribers: dict[int, asyncio.Queue] = {}
        self._counter = 0

    def subscribe(self) -> tuple[int, asyncio.Queue]:
        """Create a new subscriber. Returns (id, queue)."""
        self._counter += 1
        sub_id = self._counter
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[sub_id] = q
        return sub_id, q

    def unsubscribe(self, sub_id: int) -> None:
        self._subscribers.pop(sub_id, None)

    async def run(self) -> None:
        """Main loop — reads from source queue and broadcasts."""
        while True:
            try:
                event: Event = await self._source.get()
                msg = {"event_type": event.event_type, "data": event.data}
                for q in self._subscribers.values():
                    await q.put(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventBroadcaster error: {e}")


@dataclass
class AgentSession:
    """Wrapper for an agent session with its associated resources."""

    session_id: str
    session: Session
    tool_router: ToolRouter
    submission_queue: asyncio.Queue
    user_id: str = "dev"  # Owner of this session
    hf_token: str | None = None  # User's HF OAuth token for tool execution
    task: asyncio.Task | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    is_processing: bool = False  # True while a submission is being executed
    broadcaster: Any = None
    # True once this session has been counted against the user's daily
    # Claude quota. Guards double-counting when the user re-selects an
    # Anthropic model mid-session.
    claude_counted: bool = False


class SessionCapacityError(Exception):
    """Raised when no more sessions can be created."""

    def __init__(self, message: str, error_type: str = "global") -> None:
        super().__init__(message)
        self.error_type = error_type  # "global" or "per_user"


# ── Capacity limits ─────────────────────────────────────────────────
# Sized for HF Spaces 8 vCPU / 32 GB RAM.
# Each session uses ~10-20 MB (context, tools, queues, task); 200 × 20 MB
# = 4 GB worst case, leaving plenty of headroom for the Python runtime
# and per-request overhead.
MAX_SESSIONS: int = 200
MAX_SESSIONS_PER_USER: int = 10


class SessionManager:
    """Manages multiple concurrent agent sessions."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path or DEFAULT_CONFIG_PATH)
        self.sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    def _count_user_sessions(self, user_id: str) -> int:
        """Count active sessions owned by a specific user."""
        return sum(
            1
            for s in self.sessions.values()
            if s.user_id == user_id and s.is_active
        )

    async def create_session(
        self,
        user_id: str = "dev",
        hf_token: str | None = None,
        model: str | None = None,
    ) -> str:
        """Create a new agent session and return its ID.

        Session() and ToolRouter() constructors contain blocking I/O
        (e.g. HfApi().whoami(), litellm.get_max_tokens()) so they are
        executed in a thread pool to avoid freezing the async event loop.

        Args:
            user_id: The ID of the user who owns this session.
            hf_token: The user's HF OAuth token, stored for tool execution.
            model: Optional model override. When set, replaces ``model_name``
                on the per-session config clone. None falls back to the
                config default.

        Raises:
            SessionCapacityError: If the server or user has reached the
                maximum number of concurrent sessions.
        """
        # ── Capacity checks ──────────────────────────────────────────
        async with self._lock:
            active_count = self.active_session_count
            if active_count >= MAX_SESSIONS:
                raise SessionCapacityError(
                    f"Server is at capacity ({active_count}/{MAX_SESSIONS} sessions). "
                    "Please try again later.",
                    error_type="global",
                )
            if user_id != "dev":
                user_count = self._count_user_sessions(user_id)
                if user_count >= MAX_SESSIONS_PER_USER:
                    raise SessionCapacityError(
                        f"You have reached the maximum of {MAX_SESSIONS_PER_USER} "
                        "concurrent sessions. Please close an existing session first.",
                        error_type="per_user",
                    )

        session_id = str(uuid.uuid4())

        # Create queues for this session
        submission_queue: asyncio.Queue = asyncio.Queue()
        event_queue: asyncio.Queue = asyncio.Queue()

        # Run blocking constructors in a thread to keep the event loop responsive.
        # Without this, Session.__init__ → ContextManager → litellm.get_max_tokens()
        # blocks all HTTP/SSE handling.
        import time as _time

        def _create_session_sync():
            t0 = _time.monotonic()
            tool_router = ToolRouter(self.config.mcpServers, hf_token=hf_token)
            # Deep-copy config so each session's model switches independently —
            # tab A picking GLM doesn't flip tab B off Claude.
            session_config = self.config.model_copy(deep=True)
            if model:
                session_config.model_name = model
            session = Session(
                event_queue, config=session_config, tool_router=tool_router,
                hf_token=hf_token, user_id=user_id,
            )
            t1 = _time.monotonic()
            logger.info(f"Session initialized in {t1 - t0:.2f}s")
            return tool_router, session

        tool_router, session = await asyncio.to_thread(_create_session_sync)

        # Create wrapper
        agent_session = AgentSession(
            session_id=session_id,
            session=session,
            tool_router=tool_router,
            submission_queue=submission_queue,
            user_id=user_id,
            hf_token=hf_token,
        )

        async with self._lock:
            self.sessions[session_id] = agent_session

        # Start the agent loop task
        task = asyncio.create_task(
            self._run_session(session_id, submission_queue, event_queue, tool_router)
        )
        agent_session.task = task

        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def seed_from_summary(self, session_id: str, messages: list[dict]) -> int:
        """Rehydrate a session from cached prior messages via summarization.

        Runs the standard summarization prompt (same one compaction uses)
        over the provided messages, then seeds the new session's context
        with that summary. Tool-call pairing concerns disappear because the
        output is plain text. Returns the number of messages summarized.
        """
        from litellm import Message

        from agent.context_manager.manager import _RESTORE_PROMPT, summarize_messages

        agent_session = self.sessions.get(session_id)
        if not agent_session:
            raise ValueError(f"Session {session_id} not found")

        # Parse into Message objects, tolerating malformed entries.
        parsed: list[Message] = []
        for raw in messages:
            if raw.get("role") == "system":
                continue  # the new session has its own system prompt
            try:
                parsed.append(Message.model_validate(raw))
            except Exception as e:
                logger.warning("Dropping malformed message during seed: %s", e)

        if not parsed:
            return 0

        session = agent_session.session
        # Pass the real tool specs so the summarizer sees what the agent
        # actually has — otherwise Anthropic's modify_params injects a
        # dummy tool and the summarizer editorializes that the original
        # tool calls were fabricated.
        tool_specs = None
        try:
            tool_specs = agent_session.tool_router.get_tool_specs_for_llm()
        except Exception:
            pass
        try:
            summary, _ = await summarize_messages(
                parsed,
                model_name=session.config.model_name,
                hf_token=session.hf_token,
                max_tokens=4000,
                prompt=_RESTORE_PROMPT,
                tool_specs=tool_specs,
            )
        except Exception as e:
            logger.error("Summary call failed during seed: %s", e)
            raise

        seed = Message(
            role="user",
            content=(
                "[SYSTEM: Your prior memory of this conversation — written "
                "in your own voice right before restart. Continue from here.]\n\n"
                + (summary or "(no summary returned)")
            ),
        )
        session.context_manager.items.append(seed)
        return len(parsed)

    @staticmethod
    async def _cleanup_sandbox(session: Session) -> None:
        """Delete the sandbox Space if one was created for this session."""
        sandbox = getattr(session, "sandbox", None)
        if sandbox and getattr(sandbox, "_owns_space", False):
            space_id = getattr(sandbox, "space_id", None)
            try:
                logger.info(f"Deleting sandbox {space_id}...")
                await asyncio.to_thread(sandbox.delete)
                from agent.core import telemetry
                await telemetry.record_sandbox_destroy(session, sandbox)
            except Exception as e:
                logger.warning(f"Failed to delete sandbox {space_id}: {e}")

    async def _run_session(
        self,
        session_id: str,
        submission_queue: asyncio.Queue,
        event_queue: asyncio.Queue,
        tool_router: ToolRouter,
    ) -> None:
        """Run the agent loop for a session and broadcast events via EventBroadcaster."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            logger.error(f"Session {session_id} not found")
            return

        session = agent_session.session

        # Start event broadcaster task
        broadcaster = EventBroadcaster(event_queue)
        agent_session.broadcaster = broadcaster
        broadcast_task = asyncio.create_task(broadcaster.run())

        try:
            async with tool_router:
                # Send ready event
                await session.send_event(
                    Event(event_type="ready", data={"message": "Agent initialized"})
                )

                while session.is_running:
                    try:
                        # Wait for submission with timeout to allow checking is_running
                        submission = await asyncio.wait_for(
                            submission_queue.get(), timeout=1.0
                        )
                        agent_session.is_processing = True
                        try:
                            should_continue = await process_submission(session, submission)
                        finally:
                            agent_session.is_processing = False
                        if not should_continue:
                            break
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        logger.info(f"Session {session_id} cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Error in session {session_id}: {e}")
                        await session.send_event(
                            Event(event_type="error", data={"error": str(e)})
                        )

        finally:
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass

            await self._cleanup_sandbox(session)

            # Final-flush: always save on session death so we capture ended
            # sessions even if the client disconnects without /shutdown.
            # Idempotent via session_id key; detached subprocess.
            if session.config.save_sessions:
                try:
                    session.save_and_upload_detached(session.config.session_dataset_repo)
                except Exception as e:
                    logger.warning(f"Final-flush failed for {session_id}: {e}")

            async with self._lock:
                if session_id in self.sessions:
                    self.sessions[session_id].is_active = False

            logger.info(f"Session {session_id} ended")

    async def submit(self, session_id: str, operation: Operation) -> bool:
        """Submit an operation to a session."""
        async with self._lock:
            agent_session = self.sessions.get(session_id)

        if not agent_session or not agent_session.is_active:
            logger.warning(f"Session {session_id} not found or inactive")
            return False

        submission = Submission(id=f"sub_{uuid.uuid4().hex[:8]}", operation=operation)
        await agent_session.submission_queue.put(submission)
        return True

    async def submit_user_input(self, session_id: str, text: str) -> bool:
        """Submit user input to a session."""
        operation = Operation(op_type=OpType.USER_INPUT, data={"text": text})
        return await self.submit(session_id, operation)

    async def submit_approval(
        self, session_id: str, approvals: list[dict[str, Any]]
    ) -> bool:
        """Submit tool approvals to a session."""
        operation = Operation(
            op_type=OpType.EXEC_APPROVAL, data={"approvals": approvals}
        )
        return await self.submit(session_id, operation)

    async def interrupt(self, session_id: str) -> bool:
        """Interrupt a session by signalling cancellation directly (bypasses queue)."""
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False
        agent_session.session.cancel()
        return True

    async def undo(self, session_id: str) -> bool:
        """Undo last turn in a session."""
        operation = Operation(op_type=OpType.UNDO)
        return await self.submit(session_id, operation)

    async def truncate(self, session_id: str, user_message_index: int) -> bool:
        """Truncate conversation to before a specific user message (direct, no queue)."""
        async with self._lock:
            agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False
        return agent_session.session.context_manager.truncate_to_user_message(user_message_index)

    async def compact(self, session_id: str) -> bool:
        """Compact context in a session."""
        operation = Operation(op_type=OpType.COMPACT)
        return await self.submit(session_id, operation)

    async def shutdown_session(self, session_id: str) -> bool:
        """Shutdown a specific session."""
        operation = Operation(op_type=OpType.SHUTDOWN)
        success = await self.submit(session_id, operation)

        if success:
            async with self._lock:
                agent_session = self.sessions.get(session_id)
                if agent_session and agent_session.task:
                    # Wait for task to complete
                    try:
                        await asyncio.wait_for(agent_session.task, timeout=5.0)
                    except asyncio.TimeoutError:
                        agent_session.task.cancel()

        return success

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely."""
        async with self._lock:
            agent_session = self.sessions.pop(session_id, None)

        if not agent_session:
            return False

        # Clean up sandbox Space before cancelling the task
        await self._cleanup_sandbox(agent_session.session)

        # Cancel the task if running
        if agent_session.task and not agent_session.task.done():
            agent_session.task.cancel()
            try:
                await agent_session.task
            except asyncio.CancelledError:
                pass

        return True

    def get_session_owner(self, session_id: str) -> str | None:
        """Get the user_id that owns a session, or None if session doesn't exist."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            return None
        return agent_session.user_id

    def verify_session_access(self, session_id: str, user_id: str) -> bool:
        """Check if a user has access to a session.

        Returns True if:
        - The session exists AND the user owns it
        - The user_id is "dev" (dev mode bypass)
        """
        owner = self.get_session_owner(session_id)
        if owner is None:
            return False
        if user_id == "dev" or owner == "dev":
            return True
        return owner == user_id

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get information about a session."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            return None

        # Extract pending approval tools if any
        pending_approval = None
        pa = agent_session.session.pending_approval
        if pa and pa.get("tool_calls"):
            pending_approval = []
            for tc in pa["tool_calls"]:
                import json
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                pending_approval.append({
                    "tool": tc.function.name,
                    "tool_call_id": tc.id,
                    "arguments": args,
                })

        return {
            "session_id": session_id,
            "created_at": agent_session.created_at.isoformat(),
            "is_active": agent_session.is_active,
            "is_processing": agent_session.is_processing,
            "message_count": len(agent_session.session.context_manager.items),
            "user_id": agent_session.user_id,
            "pending_approval": pending_approval,
            "model": agent_session.session.config.model_name,
        }

    def list_sessions(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by user.

        Args:
            user_id: If provided, only return sessions owned by this user.
                     If "dev", return all sessions (dev mode).
        """
        results = []
        for sid in self.sessions:
            info = self.get_session_info(sid)
            if not info:
                continue
            if user_id and user_id != "dev" and info.get("user_id") != user_id:
                continue
            results.append(info)
        return results

    @property
    def active_session_count(self) -> int:
        """Get count of active sessions."""
        return sum(1 for s in self.sessions.values() if s.is_active)


# Global session manager instance
session_manager = SessionManager()
