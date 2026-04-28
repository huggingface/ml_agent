"""Session manager for handling multiple concurrent agent sessions."""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent.config import load_config
from agent.core.agent_loop import process_submission
from agent.messaging.gateway import NotificationGateway
from agent.core.session import Event, OpType, Session
from agent.core.session_persistence import get_session_store
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

    Events that arrive when no subscribers are listening are discarded by
    this in-memory fanout. Durable replay is handled by session_persistence.
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
                msg = {"event_type": event.event_type, "data": event.data, "seq": event.seq}
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
    title: str | None = None
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
        self.messaging_gateway = NotificationGateway(self.config.messaging)
        self.sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()
        self.persistence_store = None

    async def start(self) -> None:
        """Start shared background resources."""
        self.persistence_store = get_session_store()
        await self.persistence_store.init()
        await self.messaging_gateway.start()

    async def close(self) -> None:
        """Flush and close shared background resources."""
        await self.messaging_gateway.close()
        if self.persistence_store is not None:
            await self.persistence_store.close()

    def _store(self):
        if self.persistence_store is None:
            self.persistence_store = get_session_store()
        return self.persistence_store

    def _count_user_sessions(self, user_id: str) -> int:
        """Count active sessions owned by a specific user."""
        return sum(
            1
            for s in self.sessions.values()
            if s.user_id == user_id and s.is_active
        )

    def _create_session_sync(
        self,
        *,
        session_id: str,
        user_id: str,
        hf_token: str | None,
        model: str | None,
        event_queue: asyncio.Queue,
        notification_destinations: list[str] | None = None,
    ) -> tuple[ToolRouter, Session]:
        """Build blocking per-session resources in a worker thread."""
        import time as _time

        t0 = _time.monotonic()
        tool_router = ToolRouter(self.config.mcpServers, hf_token=hf_token)
        # Deep-copy config so each session's model switches independently —
        # tab A picking GLM doesn't flip tab B off Claude.
        session_config = self.config.model_copy(deep=True)
        if model:
            session_config.model_name = model
        session = Session(
            event_queue=event_queue,
            config=session_config,
            tool_router=tool_router,
            hf_token=hf_token,
            user_id=user_id,
            notification_gateway=self.messaging_gateway,
            notification_destinations=notification_destinations or [],
            session_id=session_id,
            persistence_store=self._store(),
        )
        t1 = _time.monotonic()
        logger.info("Session initialized in %.2fs", t1 - t0)
        return tool_router, session

    def _serialize_messages(self, session: Session) -> list[dict[str, Any]]:
        return [
            msg.model_dump(mode="json")
            for msg in session.context_manager.items
        ]

    def _serialize_pending_approval(self, session: Session) -> list[dict[str, Any]]:
        pending = session.pending_approval or {}
        tool_calls = pending.get("tool_calls") or []
        serialized: list[dict[str, Any]] = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                serialized.append(tc.model_dump(mode="json"))
            elif isinstance(tc, dict):
                serialized.append(tc)
        return serialized

    @staticmethod
    def _pending_tools_for_api(session: Session) -> list[dict[str, Any]] | None:
        pending = session.pending_approval or {}
        tool_calls = pending.get("tool_calls") or []
        if not tool_calls:
            return None
        result: list[dict[str, Any]] = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError, TypeError):
                args = {}
            result.append(
                {
                    "tool": getattr(tc.function, "name", None),
                    "tool_call_id": getattr(tc, "id", None),
                    "arguments": args,
                }
            )
        return result

    def _restore_pending_approval(
        self, session: Session, pending_approval: list[dict[str, Any]] | None
    ) -> None:
        if not pending_approval:
            session.pending_approval = None
            return
        from litellm import ChatCompletionMessageToolCall as ToolCall

        restored = []
        for raw in pending_approval:
            try:
                if "function" in raw:
                    restored.append(ToolCall(**raw))
                else:
                    restored.append(
                        ToolCall(
                            id=raw["tool_call_id"],
                            type="function",
                            function={
                                "name": raw["tool"],
                                "arguments": json.dumps(raw.get("arguments") or {}),
                            },
                        )
                    )
            except Exception as e:
                logger.warning("Dropping malformed pending approval: %s", e)
        session.pending_approval = {"tool_calls": restored} if restored else None

    @staticmethod
    def _pending_docs_for_api(
        pending_approval: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if not pending_approval:
            return None
        result: list[dict[str, Any]] = []
        for raw in pending_approval:
            if "function" in raw:
                function = raw.get("function") or {}
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result.append(
                    {
                        "tool": function.get("name"),
                        "tool_call_id": raw.get("id"),
                        "arguments": args,
                    }
                )
            elif {"tool", "tool_call_id"}.issubset(raw):
                result.append(
                    {
                        "tool": raw.get("tool"),
                        "tool_call_id": raw.get("tool_call_id"),
                        "arguments": raw.get("arguments") or {},
                    }
                )
        return result or None

    @staticmethod
    def _runtime_state(agent_session: AgentSession) -> str:
        if agent_session.session.pending_approval:
            return "waiting_approval"
        if agent_session.is_processing:
            return "processing"
        if not agent_session.is_active:
            return "ended"
        return "idle"

    async def _start_agent_session(
        self,
        *,
        agent_session: AgentSession,
        event_queue: asyncio.Queue,
        tool_router: ToolRouter,
    ) -> None:
        async with self._lock:
            self.sessions[agent_session.session_id] = agent_session

        task = asyncio.create_task(
            self._run_session(
                agent_session.session_id,
                agent_session.submission_queue,
                event_queue,
                tool_router,
            )
        )
        agent_session.task = task

    async def persist_session_snapshot(
        self,
        agent_session: AgentSession,
        *,
        runtime_state: str | None = None,
        status: str = "active",
    ) -> None:
        """Persist the current runtime context snapshot."""
        store = self._store()
        if not getattr(store, "enabled", False):
            return
        try:
            await store.save_snapshot(
                session_id=agent_session.session_id,
                user_id=agent_session.user_id,
                model=agent_session.session.config.model_name,
                title=agent_session.title,
                messages=self._serialize_messages(agent_session.session),
                runtime_state=runtime_state or self._runtime_state(agent_session),
                status=status,
                turn_count=agent_session.session.turn_count,
                pending_approval=self._serialize_pending_approval(agent_session.session),
                claude_counted=agent_session.claude_counted,
                created_at=agent_session.created_at,
                notification_destinations=list(
                    agent_session.session.notification_destinations
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to persist snapshot for %s: %s",
                agent_session.session_id,
                e,
            )

    async def ensure_session_loaded(
        self,
        session_id: str,
        user_id: str,
        hf_token: str | None = None,
    ) -> AgentSession | None:
        """Return a live runtime session, lazily restoring it from Mongo."""
        async with self._lock:
            existing = self.sessions.get(session_id)
        if existing:
            if hf_token:
                existing.hf_token = hf_token
                existing.session.hf_token = hf_token
            if user_id == "dev" or existing.user_id == "dev" or existing.user_id == user_id:
                return existing
            return None

        store = self._store()
        loaded = await store.load_session(session_id)
        if not loaded:
            return None

        meta = loaded.get("metadata") or {}
        owner = str(meta.get("user_id") or "")
        if user_id != "dev" and owner != "dev" and owner != user_id:
            return None

        from litellm import Message

        model = meta.get("model") or self.config.model_name
        event_queue: asyncio.Queue = asyncio.Queue()
        submission_queue: asyncio.Queue = asyncio.Queue()
        tool_router, session = await asyncio.to_thread(
            self._create_session_sync,
            session_id=session_id,
            user_id=owner or user_id,
            hf_token=hf_token,
            model=model,
            event_queue=event_queue,
            notification_destinations=meta.get("notification_destinations") or [],
        )

        restored_messages: list[Message] = []
        for raw in loaded.get("messages") or []:
            if not isinstance(raw, dict) or raw.get("role") == "system":
                continue
            try:
                restored_messages.append(Message.model_validate(raw))
            except Exception as e:
                logger.warning("Dropping malformed restored message: %s", e)
        if restored_messages:
            # Keep the freshly-rendered system prompt, then attach the durable
            # non-system context so tools/date/user context stay current.
            session.context_manager.items = [session.context_manager.items[0], *restored_messages]

        self._restore_pending_approval(session, meta.get("pending_approval") or [])
        session.turn_count = int(meta.get("turn_count") or 0)

        created_at = meta.get("created_at")
        if not isinstance(created_at, datetime):
            created_at = datetime.utcnow()

        agent_session = AgentSession(
            session_id=session_id,
            session=session,
            tool_router=tool_router,
            submission_queue=submission_queue,
            user_id=owner or user_id,
            hf_token=hf_token,
            created_at=created_at,
            is_active=True,
            is_processing=False,
            claude_counted=bool(meta.get("claude_counted")),
            title=meta.get("title"),
        )
        await self._start_agent_session(
            agent_session=agent_session,
            event_queue=event_queue,
            tool_router=tool_router,
        )
        logger.info("Restored session %s for user %s", session_id, owner or user_id)
        return agent_session

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
        tool_router, session = await asyncio.to_thread(
            self._create_session_sync,
            session_id=session_id,
            user_id=user_id,
            hf_token=hf_token,
            model=model,
            event_queue=event_queue,
        )

        # Create wrapper
        agent_session = AgentSession(
            session_id=session_id,
            session=session,
            tool_router=tool_router,
            submission_queue=submission_queue,
            user_id=user_id,
            hf_token=hf_token,
        )

        await self._start_agent_session(
            agent_session=agent_session,
            event_queue=event_queue,
            tool_router=tool_router,
        )
        await self.persist_session_snapshot(agent_session, runtime_state="idle")

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
        await self.persist_session_snapshot(agent_session, runtime_state="idle")
        return len(parsed)

    @staticmethod
    async def _cleanup_sandbox(session: Session) -> None:
        """Delete the sandbox Space if one was created for this session.

        Retries on transient failures (HF API 5xx, rate-limit, network blips)
        with exponential backoff. A single missed delete = a permanently
        orphaned Space, so the cost of an extra retry beats the alternative.
        """
        sandbox = getattr(session, "sandbox", None)
        if not (sandbox and getattr(sandbox, "_owns_space", False)):
            return

        space_id = getattr(sandbox, "space_id", None)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                logger.info(f"Deleting sandbox {space_id} (attempt {attempt + 1}/3)...")
                await asyncio.to_thread(sandbox.delete)
                from agent.core import telemetry
                await telemetry.record_sandbox_destroy(session, sandbox)
                return
            except Exception as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        logger.error(
            f"Failed to delete sandbox {space_id} after 3 attempts: {last_err}. "
            f"Orphan — sweep script will pick it up."
        )

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
                            await self.persist_session_snapshot(agent_session)
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
                    await self.persist_session_snapshot(
                        self.sessions[session_id],
                        runtime_state="ended",
                        status="ended",
                    )

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
        success = agent_session.session.context_manager.truncate_to_user_message(user_message_index)
        if success:
            await self.persist_session_snapshot(agent_session, runtime_state="idle")
        return success

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
        """Soft-delete a session and stop its runtime resources."""
        async with self._lock:
            agent_session = self.sessions.pop(session_id, None)

        if not agent_session:
            await self._store().soft_delete_session(session_id)
            return True

        await self._store().soft_delete_session(session_id)

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

    async def update_session_title(self, session_id: str, title: str | None) -> None:
        """Persist a user-visible title for sidebar rehydration."""
        agent_session = self.sessions.get(session_id)
        if agent_session:
            agent_session.title = title
        await self._store().update_session_fields(session_id, title=title)

    async def update_session_model(self, session_id: str, model_id: str) -> bool:
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            return False
        agent_session.session.update_model(model_id)
        await self.persist_session_snapshot(agent_session, runtime_state="idle")
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

        pending_approval = self._pending_tools_for_api(agent_session.session)

        return {
            "session_id": session_id,
            "created_at": agent_session.created_at.isoformat(),
            "is_active": agent_session.is_active,
            "is_processing": agent_session.is_processing,
            "message_count": len(agent_session.session.context_manager.items),
            "user_id": agent_session.user_id,
            "pending_approval": pending_approval,
            "model": agent_session.session.config.model_name,
            "title": agent_session.title,
            "notification_destinations": list(
                agent_session.session.notification_destinations
            ),
        }

    def set_notification_destinations(
        self, session_id: str, destinations: list[str]
    ) -> list[str]:
        """Replace the session's opted-in auto-notification destinations."""
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            raise ValueError("Session not found or inactive")

        normalized: list[str] = []
        seen: set[str] = set()
        for raw_name in destinations:
            name = raw_name.strip()
            if not name:
                raise ValueError("Destination names must not be empty")
            destination = self.config.messaging.get_destination(name)
            if destination is None:
                raise ValueError(f"Unknown destination '{name}'")
            if not destination.allow_auto_events:
                raise ValueError(
                    f"Destination '{name}' is not enabled for auto events"
                )
            if name not in seen:
                normalized.append(name)
                seen.add(name)

        agent_session.session.set_notification_destinations(normalized)
        return normalized

    async def list_sessions(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by user.

        Args:
            user_id: If provided, only return sessions owned by this user.
                     If "dev", return all sessions (dev mode).
        """
        results: list[dict[str, Any]] = []
        store = self._store()
        if getattr(store, "enabled", False):
            for row in await store.list_sessions(user_id or "dev"):
                sid = row.get("session_id") or row.get("_id")
                if not sid:
                    continue
                runtime_info = self.get_session_info(str(sid))
                if runtime_info:
                    results.append(runtime_info)
                    continue
                created_at = row.get("created_at")
                if isinstance(created_at, datetime):
                    created_at_str = created_at.isoformat()
                else:
                    created_at_str = str(created_at or datetime.utcnow().isoformat())
                pending = self._pending_docs_for_api(row.get("pending_approval") or [])
                results.append(
                    {
                        "session_id": str(sid),
                        "created_at": created_at_str,
                        "is_active": row.get("status") != "ended",
                        "is_processing": row.get("runtime_state") == "processing",
                        "message_count": int(row.get("message_count") or 0),
                        "user_id": row.get("user_id") or "dev",
                        "pending_approval": pending or None,
                        "model": row.get("model"),
                        "title": row.get("title"),
                        "notification_destinations": row.get("notification_destinations") or [],
                    }
                )
            return results

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
