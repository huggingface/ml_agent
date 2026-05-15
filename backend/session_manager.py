"""Session manager for handling multiple concurrent agent sessions."""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from pymongo.errors import PyMongoError

from agent.config import load_config
from agent.core.agent_loop import process_submission
from agent.messaging.gateway import NotificationGateway
from agent.core.session import Event, OpType, Session
from agent.core.session_persistence import get_session_store, make_holder_id
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
    """Wrapper for an agent session with its associated resources.

    ``session`` and ``tool_router`` are ``Optional`` to support cross-process
    *stubs* — lightweight placeholders for sessions held by another holder
    (Worker). A stub carries enough identity (``session_id``, ``user_id``,
    ``holder_id``) to satisfy access checks and the SSE non-holder slow
    path, but no live runtime resources. Only the actual lease holder ever
    constructs a fully populated ``AgentSession``.
    """

    session_id: str
    session: Session | None
    tool_router: ToolRouter | None
    user_id: str = "dev"  # Owner of this session
    hf_username: str | None = None  # HF namespace used for personal trace uploads
    hf_token: str | None = None  # User's HF OAuth token for tool execution
    task: asyncio.Task | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    is_processing: bool = False  # True while a submission is being executed
    broadcaster: Any = None
    title: str | None = None
    # True once this session has been counted against the user's daily
    # Claude quota. Guards double-counting when the user re-selects an
    # Anthropic model mid-session.
    claude_counted: bool = False
    # Wall-clock timestamp of the last submission processed for this
    # session — used by US-007's idle eviction. Updated in one place
    # inside ``_drain_and_process``.
    last_submission_at: float = field(default_factory=lambda: 0.0)
    # Holder identity that owns this session's lease. Set when the lease
    # is claimed in ``create_session`` / ``ensure_session_loaded`` to
    # ``SessionManager._holder_id``. The SSE fast path branches on
    # ``holder_id == session_manager._holder_id`` to decide whether to
    # subscribe to the in-process broadcaster (this process holds it) or
    # tail the Mongo change stream (a different process holds it).
    holder_id: str | None = None


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
DEFAULT_YOLO_COST_CAP_USD: float = 5.0


class SessionManager:
    """Manages multiple concurrent agent sessions."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path or DEFAULT_CONFIG_PATH)
        self.messaging_gateway = NotificationGateway(self.config.messaging)
        self.sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()
        self.persistence_store = None

        # Holder identity — pick once at process start, never recompute.
        # MODE controls which lane this process owns ("main" = synchronous
        # frontend handler, "worker" = background submission consumer).
        raw_mode = os.environ.get("MODE", "main").lower().strip()
        if raw_mode not in {"main", "worker"}:
            logger.warning(
                f"Unknown MODE={raw_mode!r}; falling back to 'main'"
            )
            raw_mode = "main"
        self.mode: str = raw_mode
        self._holder_id: str = make_holder_id(self.mode)
        self._heartbeat_task: asyncio.Task | None = None
        self._grace_sweep_task: asyncio.Task | None = None
        self._idle_eviction_task: asyncio.Task | None = None
        # SSE subscriber bookkeeping — used by the US-006 grace-period
        # sweeper to decide when a session has had no readers for long
        # enough to be evicted. Populated by ``_attach_subscriber`` /
        # ``_detach_subscriber`` from both SSE transport branches.
        self._subscriber_counts: dict[str, int] = {}
        # Wall-clock timestamp (``time.time()``) of when ``session_id``'s
        # subscriber count last hit zero. Cleared the moment a new
        # subscriber attaches.
        self._no_subscriber_since: dict[str, float] = {}
        logger.info(
            "SessionManager init: mode=%s holder_id=%s",
            self.mode,
            self._holder_id,
        )

    def _attach_subscriber(self, session_id: str) -> None:
        """Increment the subscriber count for ``session_id``.

        Called from both SSE transport branches (holder fast-path and
        non-holder slow-path) when a stream attaches.
        """
        self._subscriber_counts[session_id] = (
            self._subscriber_counts.get(session_id, 0) + 1
        )
        self._no_subscriber_since.pop(session_id, None)

    def _detach_subscriber(self, session_id: str) -> None:
        """Decrement the subscriber count; record the zero-point on transitions."""
        n = self._subscriber_counts.get(session_id, 0)
        n = max(0, n - 1)
        if n == 0:
            self._subscriber_counts.pop(session_id, None)
            self._no_subscriber_since[session_id] = time.time()
        else:
            self._subscriber_counts[session_id] = n

    async def start(self) -> None:
        """Start shared background resources."""
        self.persistence_store = get_session_store()
        await self.persistence_store.init()
        await self.messaging_gateway.start()
        self._heartbeat_task = asyncio.create_task(self._lease_heartbeat_loop())
        self._grace_sweep_task = asyncio.create_task(self._grace_period_sweep_loop())
        self._idle_eviction_task = asyncio.create_task(self._idle_eviction_loop())

    async def close(self) -> None:
        """Flush and close shared background resources."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._grace_sweep_task is not None:
            self._grace_sweep_task.cancel()
            try:
                await self._grace_sweep_task
            except asyncio.CancelledError:
                pass
            self._grace_sweep_task = None
        if self._idle_eviction_task is not None:
            self._idle_eviction_task.cancel()
            try:
                await self._idle_eviction_task
            except asyncio.CancelledError:
                pass
            self._idle_eviction_task = None
        await self.messaging_gateway.close()
        if self.persistence_store is not None:
            await self.persistence_store.close()

    async def _lease_heartbeat_loop(self) -> None:
        """Renew leases every TTL/3 seconds for sessions held by this process.

        On CAS-mismatch for a session (``renew_lease`` returns ``None``):
        requeue that session's claimed submissions, drop the session, log
        WARN. On a transient ``PyMongoError`` from ``renew_lease``: log a
        warning and skip this tick for the affected session — do NOT treat
        a Mongo flap as lease theft. The loop must never crash; any other
        unexpected exception is logged and the loop sleeps before retrying.
        """
        HEARTBEAT_INTERVAL_S = 10  # TTL=30s, renew at TTL/3
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
                store = self._store()
                if not getattr(store, "enabled", False):
                    continue  # NoopSessionStore — nothing to renew
                # Snapshot session_ids under lock to avoid mutation during iteration.
                async with self._lock:
                    session_ids = list(self.sessions.keys())
                for session_id in session_ids:
                    try:
                        renewed = await store.renew_lease(
                            session_id, self._holder_id, ttl_s=30
                        )
                    except PyMongoError as e:
                        logger.warning(
                            f"renew_lease transient error for {session_id} "
                            f"({self._holder_id}); skipping tick: {e}"
                        )
                        continue
                    if renewed is None:
                        # Lease lost — someone else holds it now.
                        await self._on_lease_lost(session_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                # Don't crash the loop; sleep briefly and retry.
                await asyncio.sleep(1)

    async def _on_lease_lost(self, session_id: str) -> None:
        """Called when our lease for ``session_id`` has been taken by another holder.

        Requeue claimed submissions for THIS session only, drop the session,
        log WARN. We must NOT requeue all claimed submissions for this
        holder — losing one session's lease shouldn't cause double-execution
        on every other session this Main still holds. The heartbeat loop
        must keep going, so we don't await the cancelled task here.
        """
        store = self._store()
        try:
            requeued = await store.requeue_claimed_for(
                self._holder_id, session_id=session_id
            )
            logger.warning(
                "Lease lost for session %s (held_by=%s); requeued %d claimed submissions",
                session_id, self._holder_id, requeued,
            )
        except Exception as e:
            logger.error(
                f"requeue_claimed_for failed during lease-loss for {session_id}: {e}"
            )
        async with self._lock:
            agent_session = self.sessions.pop(session_id, None)
        if agent_session and agent_session.task and not agent_session.task.done():
            agent_session.task.cancel()
            # Don't await — heartbeat loop must keep going.

    async def release_session_to_background(
        self, session_id: str, reason: str = "manual"
    ) -> bool:
        """Emit migrating event, requeue claimed submissions, release lease.

        Used by the lifespan shutdown sweep, the grace-period sweeper, and
        the manual ``/background`` route. Idempotent on already-released or
        unknown session IDs (returns False in that case).
        """
        async with self._lock:
            agent_session = self.sessions.get(session_id)
        if agent_session is None:
            return False
        # 1) Emit migrating event so the frontend can render a "reconnecting"
        # state. send_event also durably appends via append_event, so a
        # non-holder reader will see it on the next change-stream tick.
        try:
            await agent_session.session.send_event(
                Event(event_type="migrating", data={"reason": reason})
            )
            logger.info(f"migrating_emitted session_id={session_id} reason={reason}")
        except Exception as e:
            logger.warning(
                f"Failed to emit migrating event for {session_id}: {e}"
            )
        # 2) Requeue any submissions we have in-flight back to pending so a
        # Worker can pick them up.
        store = self._store()
        if getattr(store, "enabled", False):
            try:
                n = await store.requeue_claimed_for(self._holder_id)
                if n > 0:
                    logger.info(f"requeue_claimed holder_id={self._holder_id} count={n}")
            except Exception as e:
                logger.warning(
                    f"requeue_claimed_for failed during release of "
                    f"{session_id}: {e}"
                )
        # 3) Release the lease.
        try:
            await store.release_lease(session_id, self._holder_id)
            logger.info(f"lease_release session_id={session_id} holder_id={self._holder_id} reason={reason}")
        except Exception as e:
            logger.warning(f"release_lease failed for {session_id}: {e}")
        # 4) Drop from in-memory and cancel the agent task. Don't await the
        # cancel — heartbeat / sweep loops must keep going.
        async with self._lock:
            popped = self.sessions.pop(session_id, None)
        if popped and popped.task and not popped.task.done():
            popped.task.cancel()
        return True

    async def _grace_period_sweep_loop(self) -> None:
        """Every 30s, scan sessions held by this process. If a session has
        had zero subscribers for longer than ``GRACE_PERIOD_SECONDS`` AND has
        either in-flight work or pending submissions, release it to
        background. Idle-with-no-work sessions are NOT auto-backgrounded —
        they wait for idle eviction (US-007) or shutdown.
        """
        SWEEP_INTERVAL_S = 30
        GRACE_PERIOD_S = float(os.environ.get("GRACE_PERIOD_SECONDS", "180"))
        while True:
            try:
                await asyncio.sleep(SWEEP_INTERVAL_S)
                now = time.time()
                async with self._lock:
                    session_ids = list(self.sessions.keys())
                store = self._store()
                for session_id in session_ids:
                    agent_session = self.sessions.get(session_id)
                    if agent_session is None:
                        continue
                    if agent_session.holder_id != self._holder_id:
                        continue
                    no_sub_since = self._no_subscriber_since.get(session_id)
                    if no_sub_since is None:
                        # Either someone is connected now, or no one has
                        # ever connected — neither case is an eviction.
                        continue
                    if now - no_sub_since < GRACE_PERIOD_S:
                        continue
                    has_pending = False
                    if getattr(store, "enabled", False):
                        try:
                            pending_docs = await store.poll_pending_submissions_after(
                                session_id, None
                            )
                            has_pending = len(pending_docs) > 0
                        except Exception:
                            has_pending = False
                    has_work = (
                        agent_session.is_processing
                        or has_pending
                        or getattr(agent_session.session, "is_in_tool_call", False)
                    )
                    if not has_work:
                        continue
                    logger.info(
                        f"Grace period elapsed for {session_id} "
                        f"(no subs for {now - no_sub_since:.0f}s); "
                        "releasing to background"
                    )
                    await self.release_session_to_background(
                        session_id, reason="grace_period_elapsed"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Grace sweep loop error: {e}")
                await asyncio.sleep(1)

    async def _idle_eviction_loop(self) -> None:
        """Every 60s, drop sessions held by this process that are fully idle
        past ``IDLE_EVICTION_SECONDS`` (default 1800s = 30min).

        "Idle" predicate (US-007 spec):
            * not ``is_in_tool_call`` (tool may still be executing)
            * not ``is_processing`` (agent loop currently busy)
            * no pending submissions in Mongo
            * ``now - last_submission_at > IDLE_TTL_S``

        On eviction, the lease is released and the session is dropped from
        the in-memory map. No ``migrating`` event is emitted — by definition
        nobody is watching.
        """
        SWEEP_INTERVAL_S = 60
        IDLE_TTL_S = float(os.environ.get("IDLE_EVICTION_SECONDS", "1800"))
        while True:
            try:
                await asyncio.sleep(SWEEP_INTERVAL_S)
                now = time.time()
                store = self._store()
                async with self._lock:
                    session_ids = list(self.sessions.keys())
                for sid in session_ids:
                    agent_session = self.sessions.get(sid)
                    if agent_session is None:
                        continue
                    if agent_session.holder_id != self._holder_id:
                        continue
                    if agent_session.is_processing:
                        continue
                    if getattr(agent_session.session, "is_in_tool_call", False):
                        continue
                    if agent_session.last_submission_at == 0.0:
                        # Never had a submission yet (just-created); allow
                        # IDLE_TTL grace measured from creation.
                        last = (
                            agent_session.created_at.timestamp()
                            if hasattr(agent_session.created_at, "timestamp")
                            else 0.0
                        )
                    else:
                        last = agent_session.last_submission_at
                    if now - last < IDLE_TTL_S:
                        continue
                    # Pending submissions in Mongo? If so, skip — a worker
                    # tick will pick them up.
                    if getattr(store, "enabled", False):
                        try:
                            pending_docs = await store.poll_pending_submissions_after(
                                sid, None
                            )
                            if pending_docs:
                                continue
                        except Exception:
                            # If Mongo flapped, err on the side of NOT
                            # evicting — heartbeat will renew the lease and
                            # we'll try again on the next sweep.
                            continue
                    logger.info(
                        f"Idle-evicting {sid} (idle for {now - last:.0f}s)"
                    )
                    try:
                        await store.release_lease(sid, self._holder_id)
                        logger.info(f"lease_release session_id={sid} holder_id={self._holder_id} reason=idle_eviction")
                    except Exception as e:
                        logger.warning(
                            f"release_lease failed during idle evict for {sid}: {e}"
                        )
                    async with self._lock:
                        popped = self.sessions.pop(sid, None)
                    if popped and popped.task and not popped.task.done():
                        popped.task.cancel()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Idle eviction loop error: {e}")
                await asyncio.sleep(2)

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
        hf_username: str | None,
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
            hf_username=hf_username,
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

    @staticmethod
    def _auto_approval_summary(session: Session) -> dict[str, Any]:
        if hasattr(session, "auto_approval_policy_summary"):
            return session.auto_approval_policy_summary()
        cap = getattr(session, "auto_approval_cost_cap_usd", None)
        estimated = float(getattr(session, "auto_approval_estimated_spend_usd", 0.0) or 0.0)
        remaining = None if cap is None else round(max(0.0, float(cap) - estimated), 4)
        return {
            "enabled": bool(getattr(session, "auto_approval_enabled", False)),
            "cost_cap_usd": cap,
            "estimated_spend_usd": round(estimated, 4),
            "remaining_usd": remaining,
        }

    async def _start_agent_session(
        self,
        *,
        agent_session: AgentSession,
        event_queue: asyncio.Queue,
        tool_router: ToolRouter,
    ) -> AgentSession:
        async with self._lock:
            existing = self.sessions.get(agent_session.session_id)
            if existing:
                return existing
            self.sessions[agent_session.session_id] = agent_session

        task = asyncio.create_task(
            self._run_session(
                agent_session.session_id,
                event_queue,
                tool_router,
            )
        )
        agent_session.task = task
        return agent_session

    @staticmethod
    def _can_access_session(agent_session: AgentSession, user_id: str) -> bool:
        return (
            user_id == "dev"
            or agent_session.user_id == "dev"
            or agent_session.user_id == user_id
        )

    @staticmethod
    def _update_hf_identity(
        agent_session: AgentSession,
        *,
        hf_token: str | None,
        hf_username: str | None,
    ) -> None:
        if hf_token:
            agent_session.hf_token = hf_token
            agent_session.session.hf_token = hf_token
        if hf_username:
            agent_session.hf_username = hf_username
            agent_session.session.hf_username = hf_username

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
                auto_approval_enabled=bool(
                    getattr(agent_session.session, "auto_approval_enabled", False)
                ),
                auto_approval_cost_cap_usd=getattr(
                    agent_session.session, "auto_approval_cost_cap_usd", None
                ),
                auto_approval_estimated_spend_usd=float(
                    getattr(
                        agent_session.session,
                        "auto_approval_estimated_spend_usd",
                        0.0,
                    )
                    or 0.0
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to persist snapshot for %s: %s",
                agent_session.session_id,
                e,
            )

    async def _rebuild_agent_session_from_store(
        self,
        loaded: dict[str, Any],
        session_id: str,
        hf_token: str | None,
        hf_username: str | None,
        owner: str,
    ) -> AgentSession:
        """Reconstruct an ``AgentSession`` from a Mongo ``load_session`` result.

        Caller is expected to have already claimed the lease (or persistence
        is disabled). Shared by ``ensure_session_loaded`` and
        ``claim_dormant_session``.
        """
        from litellm import Message

        meta = loaded.get("metadata") or {}
        model = meta.get("model") or self.config.model_name
        event_queue: asyncio.Queue = asyncio.Queue()
        tool_router, session = await asyncio.to_thread(
            self._create_session_sync,
            session_id=session_id,
            user_id=owner,
            hf_username=hf_username,
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
        session.auto_approval_enabled = bool(meta.get("auto_approval_enabled", False))
        raw_cap = meta.get("auto_approval_cost_cap_usd")
        session.auto_approval_cost_cap_usd = (
            float(raw_cap) if isinstance(raw_cap, int | float) else None
        )
        session.auto_approval_estimated_spend_usd = float(
            meta.get("auto_approval_estimated_spend_usd") or 0.0
        )

        created_at = meta.get("created_at")
        if not isinstance(created_at, datetime):
            created_at = datetime.utcnow()

        agent_session = AgentSession(
            session_id=session_id,
            session=session,
            tool_router=tool_router,
            user_id=owner,
            hf_username=hf_username,
            hf_token=hf_token,
            created_at=created_at,
            is_active=True,
            is_processing=False,
            claude_counted=bool(meta.get("claude_counted")),
            title=meta.get("title"),
            holder_id=self._holder_id,
        )
        started = await self._start_agent_session(
            agent_session=agent_session,
            event_queue=event_queue,
            tool_router=tool_router,
        )
        if started is not agent_session:
            self._update_hf_identity(
                started,
                hf_token=hf_token,
                hf_username=hf_username,
            )
            return started
        return agent_session

    async def ensure_session_loaded(
        self,
        session_id: str,
        user_id: str,
        hf_token: str | None = None,
        hf_username: str | None = None,
    ) -> AgentSession | None:
        """Return a live runtime session, lazily restoring it from Mongo."""
        async with self._lock:
            existing = self.sessions.get(session_id)
        if existing:
            if self._can_access_session(existing, user_id):
                self._update_hf_identity(
                    existing,
                    hf_token=hf_token,
                    hf_username=hf_username,
                )
                return existing
            return None

        store = self._store()
        loaded = await store.load_session(session_id)
        if not loaded:
            return None

        async with self._lock:
            existing = self.sessions.get(session_id)
        if existing:
            if self._can_access_session(existing, user_id):
                self._update_hf_identity(
                    existing,
                    hf_token=hf_token,
                    hf_username=hf_username,
                )
                return existing
            return None

        meta = loaded.get("metadata") or {}
        owner = str(meta.get("user_id") or "")
        if user_id != "dev" and owner != "dev" and owner != user_id:
            return None

        if getattr(store, "enabled", False):
            claimed = await store.claim_lease(
                session_id, self._holder_id, ttl_s=30
            )
            if claimed is None:
                # Another holder owns an unexpired lease. Return a stub so
                # the route layer's access check passes and the SSE slow
                # path / submission-enqueue paths can deliver to the actual
                # holder. The stub is NOT inserted into ``self.sessions`` —
                # only the real holder owns that map entry.
                foreign_lease = (meta.get("lease") or {})
                foreign_holder = foreign_lease.get("holder_id")
                if not foreign_holder:
                    # Defensive: post-backfill every active session has a
                    # lease subdoc; if it's missing we can't safely build a
                    # stub. Preserve the legacy behaviour and return None.
                    logger.info(
                        f"Refusing restore of {session_id}: lease "
                        "held by another process (no holder_id on doc)"
                    )
                    return None
                created_at = meta.get("created_at")
                if not isinstance(created_at, datetime):
                    created_at = datetime.utcnow()
                logger.info(
                    f"ensure_session_loaded stub session_id={session_id} "
                    f"foreign_holder={foreign_holder} (lease held elsewhere)"
                )
                return AgentSession(
                    session_id=session_id,
                    session=None,
                    tool_router=None,
                    user_id=owner or user_id,
                    hf_username=hf_username,
                    hf_token=hf_token,
                    task=None,
                    created_at=created_at,
                    is_active=True,
                    is_processing=False,
                    broadcaster=None,
                    holder_id=foreign_holder,
                )
            logger.info(f"lease_claim session_id={session_id} holder_id={self._holder_id}")

        agent_session = await self._rebuild_agent_session_from_store(
            loaded=loaded,
            session_id=session_id,
            hf_token=hf_token,
            hf_username=hf_username,
            owner=owner or user_id,
        )
        logger.info("Restored session %s for user %s", session_id, owner or user_id)
        return agent_session

    async def claim_dormant_session(
        self, session_id: str
    ) -> AgentSession | None:
        """Internal: claim and load a dormant session without a user-ownership
        check. Used by ``worker_loop``'s claim tick — the worker process is
        process-level trusted, and the lease CAS still enforces the
        "one holder at a time" invariant.

        Returns the live ``AgentSession`` on success; ``None`` if the session
        doesn't exist, the lease is already held, or persistence is disabled.
        """
        async with self._lock:
            existing = self.sessions.get(session_id)
        if existing:
            return existing

        store = self._store()
        if not getattr(store, "enabled", False):
            return None

        loaded = await store.load_session(session_id)
        if not loaded:
            return None

        # Claim the lease BEFORE building the session — fast bail out if
        # someone else holds it.
        claimed = await store.claim_lease(
            session_id, self._holder_id, ttl_s=30
        )
        if claimed is None:
            logger.debug(
                f"Worker refusing to claim {session_id}: lease held by another process"
            )
            return None
        logger.info(f"lease_claim session_id={session_id} holder_id={self._holder_id}")

        meta = loaded.get("metadata") or {}
        owner = str(meta.get("user_id") or "") or "dev"
        agent_session = await self._rebuild_agent_session_from_store(
            loaded=loaded,
            session_id=session_id,
            hf_token=None,
            hf_username=None,
            owner=owner,
        )
        logger.info(
            "Worker claimed dormant session %s (owner=%s)", session_id, owner
        )
        return agent_session

    async def create_session(
        self,
        user_id: str = "dev",
        hf_username: str | None = None,
        hf_token: str | None = None,
        model: str | None = None,
        is_pro: bool | None = None,
    ) -> str:
        """Create a new agent session and return its ID.

        Session() and ToolRouter() constructors contain blocking I/O
        (e.g. HfApi().whoami(), litellm.get_max_tokens()) so they are
        executed in a thread pool to avoid freezing the async event loop.

        Args:
            user_id: The ID of the user who owns this session.
            hf_username: The HF username/namespace used for personal trace uploads.
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

        # Create queue for this session (events still flow through an
        # in-process queue; submissions now live in Mongo).
        event_queue: asyncio.Queue = asyncio.Queue()

        # Run blocking constructors in a thread to keep the event loop responsive.
        tool_router, session = await asyncio.to_thread(
            self._create_session_sync,
            session_id=session_id,
            user_id=user_id,
            hf_username=hf_username,
            hf_token=hf_token,
            model=model,
            event_queue=event_queue,
        )

        # Create wrapper
        agent_session = AgentSession(
            session_id=session_id,
            session=session,
            tool_router=tool_router,
            user_id=user_id,
            hf_username=hf_username,
            hf_token=hf_token,
        )

        # Persist the session doc BEFORE the lease CAS so claim_lease has
        # a row to update. upsert_session writes user_id / surface /
        # schema_version via $setOnInsert; running it first guarantees
        # those fields land in Mongo before any other writer touches the
        # doc.
        await self.persist_session_snapshot(agent_session, runtime_state="idle")

        # Claim the lease before starting the runtime task — the doc was
        # just persisted, so this should always succeed; failure is
        # treated as an internal error.
        claimed = await self._store().claim_lease(
            session_id, self._holder_id, ttl_s=30
        )
        if (
            claimed is None
            and getattr(self._store(), "enabled", False)
        ):
            logger.warning(
                f"Failed to claim lease for new session {session_id} "
                f"(holder={self._holder_id})"
            )
            raise RuntimeError(
                f"Failed to claim lease for new session {session_id}"
            )
        # Tag the session with our holder identity so the SSE fast-path
        # branch knows we own it. Always safe to set: either we just
        # claimed (Mongo path) or persistence is disabled (single-process
        # local dev — we are trivially the only holder).
        agent_session.holder_id = self._holder_id
        logger.info(f"lease_claim session_id={session_id} holder_id={self._holder_id}")

        await self._start_agent_session(
            agent_session=agent_session,
            event_queue=event_queue,
            tool_router=tool_router,
        )

        if is_pro is not None and user_id and user_id != "dev":
            await self._track_pro_status(agent_session, is_pro=is_pro)

        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def _track_pro_status(self, agent_session: AgentSession, *, is_pro: bool) -> None:
        """Update Mongo per-user Pro state and emit a one-shot conversion
        event if the store reports a free→Pro transition. Best-effort: any
        Mongo failure is swallowed so we never fail session creation on
        telemetry."""
        store = self._store()
        if not getattr(store, "enabled", False):
            return
        try:
            result = await store.mark_pro_seen(agent_session.user_id, is_pro=is_pro)
        except Exception as e:
            logger.debug("mark_pro_seen failed: %s", e)
            return
        if not result or not result.get("converted"):
            return
        try:
            from agent.core import telemetry
            await telemetry.record_pro_conversion(
                agent_session.session,
                first_seen_at=result.get("first_seen_at"),
            )
        except Exception as e:
            logger.debug("record_pro_conversion failed: %s", e)

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
                session=session,
                kind="restore",
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

    async def _consume_submissions(self, agent_session: AgentSession) -> None:
        """Consume pending submissions for this session.

        Tries the Mongo change stream first (push-based, low-latency); on
        replica-set unavailability or any ``PyMongoError`` from ``watch()``
        falls back to a 500 ms polling loop. Either path drains all
        currently-pending submissions through ``_drain_and_process``.
        """
        session = agent_session.session
        session_id = agent_session.session_id
        store = self._store()
        use_change_stream = bool(getattr(store, "enabled", False))

        # Drain anything that arrived before the consumer started — covers
        # the race where ``enqueue`` happened during runtime startup.
        await self._drain_and_process(agent_session)

        while session.is_running:
            try:
                if use_change_stream:
                    try:
                        async for _change_doc in store.change_stream_pending_submissions(
                            session_id
                        ):
                            await self._drain_and_process(agent_session)
                            if not session.is_running:
                                break
                        # Stream exited without error (e.g. shutdown). Break.
                        if not session.is_running:
                            break
                    except PyMongoError as e:
                        logger.warning(
                            f"Change stream failed for {session_id}, "
                            f"falling back to polling: {e}"
                        )
                        use_change_stream = False
                    except NotImplementedError:
                        # NoopSessionStore (or any store without watch())
                        use_change_stream = False
                else:
                    await asyncio.sleep(0.5)
                    await self._drain_and_process(agent_session)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Submission consume error for {session_id}: {e}"
                )
                await asyncio.sleep(1)

    async def _drain_and_process(self, agent_session: AgentSession) -> None:
        """Claim and process all pending submissions for ``agent_session``, FIFO.

        Handles ``interrupt`` and ``shutdown`` ops inline (they don't go
        through the agent loop). All other ops are reconstructed into a
        ``Submission(Operation(...))`` and dispatched to ``process_submission``.
        Marks each submission ``done`` in a finally so a poison submission
        never gets redelivered.
        """
        session = agent_session.session
        session_id = agent_session.session_id
        store = self._store()
        if not getattr(store, "enabled", False):
            return
        while session.is_running:
            claimed = await store.claim_pending_submission(session_id, self._holder_id)
            if claimed is None:
                return
            submission_id = claimed.get("_id")
            op_type = claimed.get("op_type")
            payload = claimed.get("payload") or {}
            try:
                # Wall-clock (time.time()) so it composes with the same clock
                # used by ``_no_subscriber_since`` and the idle-eviction loop.
                agent_session.last_submission_at = time.time()
                created_at = claimed.get("created_at")
                if isinstance(created_at, datetime):
                    _ca = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
                    lag = (datetime.now(UTC) - _ca).total_seconds()
                    if lag > 0.1:
                        logger.debug(
                            f"pending_submission_lag session_id={session_id} "
                            f"op_type={op_type} lag_ms={int(lag * 1000)}"
                        )
                # Inline ops: interrupt + shutdown bypass the agent loop.
                if op_type == "interrupt":
                    session.cancel()
                    continue
                if op_type == "shutdown":
                    session.is_running = False
                    return
                agent_session.is_processing = True
                try:
                    operation = self._build_operation(op_type, payload)
                    submission = Submission(
                        id=f"sub_{uuid.uuid4().hex[:8]}",
                        operation=operation,
                    )
                    should_continue = await process_submission(session, submission)
                finally:
                    agent_session.is_processing = False
                    await self.persist_session_snapshot(agent_session)
                if not should_continue:
                    session.is_running = False
                    return
            except Exception as e:
                logger.error(
                    f"Error processing submission {submission_id} "
                    f"for {session_id}: {e}"
                )
            finally:
                # Always mark done so a poison row is not redelivered.
                try:
                    await store.mark_submission_done(submission_id)
                except Exception as e:
                    logger.debug(
                        f"mark_submission_done failed for {submission_id}: {e}"
                    )

    def _build_operation(self, op_type: Any, payload: dict) -> Operation:
        """Reconstruct an ``Operation`` from a ``pending_submissions`` row."""
        if isinstance(op_type, OpType):
            enum_op = op_type
        else:
            enum_op = OpType(op_type)
        return Operation(op_type=enum_op, data=payload or None)

    async def _run_session(
        self,
        session_id: str,
        event_queue: asyncio.Queue,
        tool_router: ToolRouter,
    ) -> None:
        """Run the agent loop for a session and broadcast events via EventBroadcaster."""
        agent_session = self.sessions[session_id]
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

                try:
                    await self._consume_submissions(agent_session)
                except asyncio.CancelledError:
                    logger.info(f"Session {session_id} cancelled")
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

            try:
                await self._store().release_lease(session_id, self._holder_id)
                logger.info(f"lease_release session_id={session_id} holder_id={self._holder_id} reason=session_end")
            except Exception as e:
                logger.debug(
                    f"release_lease failed for {session_id} on session end: {e}"
                )

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

    async def _enqueue_or_false(
        self, session_id: str, op_type: str, payload: dict[str, Any]
    ) -> bool:
        """Enqueue a pending submission, returning False when no session
        exists in either runtime memory or the durable store.

        The route layer's ``_check_session_access`` already gates by user;
        this method only verifies the session exists somewhere we can
        deliver to. When the store is the no-op (Mongo disabled), require
        the session to be in our in-memory map and refuse if not — there
        is no other holder to forward to.
        """
        store = self._store()
        in_memory = self.sessions.get(session_id)
        if not getattr(store, "enabled", False):
            if in_memory is None or not in_memory.is_active:
                logger.warning(f"Session {session_id} not found or inactive")
                return False
            # No durable queue — without Mongo we cannot enqueue. Drop and
            # warn; this path is exercised in CLI/local-dev only and the
            # legacy in-memory flow has been removed.
            logger.warning(
                f"Cannot enqueue submission for {session_id}: "
                "Mongo persistence disabled"
            )
            return False
        if in_memory is None:
            doc = await store.load_session(session_id)
            if doc is None:
                logger.warning(f"Session {session_id} not found")
                return False
        await store.enqueue_pending_submission(
            session_id, op_type=op_type, payload=payload
        )
        return True

    async def submit(self, session_id: str, operation: Operation) -> bool:
        """Submit an operation to a session via the durable pending queue."""
        return await self._enqueue_or_false(
            session_id,
            op_type=operation.op_type.value,
            payload=operation.data or {},
        )

    async def submit_user_input(self, session_id: str, text: str) -> bool:
        """Submit user input to a session."""
        return await self._enqueue_or_false(
            session_id, op_type="user_input", payload={"text": text}
        )

    async def submit_approval(
        self, session_id: str, approvals: list[dict[str, Any]]
    ) -> bool:
        """Submit tool approvals to a session."""
        return await self._enqueue_or_false(
            session_id, op_type="exec_approval", payload={"approvals": approvals}
        )

    async def interrupt(self, session_id: str) -> bool:
        """Interrupt by signalling cancellation. Holder fast-path; non-holder
        enqueues an interrupt op for the actual lease holder to consume.
        """
        async with self._lock:
            agent_session = self.sessions.get(session_id)
        if agent_session and agent_session.is_active:
            # We are the holder — fast path, cancel directly.
            agent_session.session.cancel()
            return True
        store = self._store()
        if not getattr(store, "enabled", False):
            return False
        if await store.load_session(session_id) is None:
            return False
        await store.enqueue_pending_submission(
            session_id, op_type="interrupt", payload={}
        )
        return True

    async def undo(self, session_id: str) -> bool:
        """Undo last turn in a session."""
        return await self._enqueue_or_false(
            session_id, op_type="undo", payload={}
        )

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
        return await self._enqueue_or_false(
            session_id, op_type="compact", payload={}
        )

    async def shutdown_session(self, session_id: str) -> bool:
        """Shutdown a specific session.

        Enqueues a ``shutdown`` op (the consumer drains it inline by setting
        ``session.is_running = False``), then releases the lease and awaits
        the task locally so ``DELETE`` callers see a clean stop.

        We only acquire ``self._lock`` for the dict lookup — external I/O
        (``release_lease`` Mongo round-trip, ``wait_for(task)`` agent loop
        drain) runs without the lock so heartbeat snapshots, grace sweeps,
        idle eviction, and other shutdowns aren't serialized behind us.
        """
        success = await self._enqueue_or_false(
            session_id, op_type="shutdown", payload={}
        )

        if success:
            async with self._lock:
                agent_session = self.sessions.get(session_id)
            if agent_session and agent_session.task:
                try:
                    await self._store().release_lease(
                        session_id, self._holder_id
                    )
                except Exception as e:
                    logger.debug(
                        f"release_lease failed during shutdown of {session_id}: {e}"
                    )
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

        try:
            await self._store().release_lease(session_id, self._holder_id)
        except Exception as e:
            logger.debug(
                f"release_lease failed during delete of {session_id}: {e}"
            )

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

    async def update_session_auto_approval(
        self,
        session_id: str,
        *,
        enabled: bool,
        cost_cap_usd: float | None,
        cap_provided: bool = False,
    ) -> dict[str, Any]:
        agent_session = self.sessions.get(session_id)
        if not agent_session or not agent_session.is_active:
            raise ValueError("Session not found or inactive")

        session = agent_session.session
        if enabled:
            if not cap_provided and cost_cap_usd is None:
                cost_cap_usd = getattr(
                    session, "auto_approval_cost_cap_usd", None
                )
                if cost_cap_usd is None:
                    cost_cap_usd = DEFAULT_YOLO_COST_CAP_USD
            elif cost_cap_usd is None:
                cost_cap_usd = DEFAULT_YOLO_COST_CAP_USD
        else:
            if not cap_provided:
                cost_cap_usd = getattr(session, "auto_approval_cost_cap_usd", None)

        if hasattr(session, "set_auto_approval_policy"):
            session.set_auto_approval_policy(
                enabled=enabled,
                cost_cap_usd=cost_cap_usd,
            )
        else:
            session.auto_approval_enabled = bool(enabled)
            session.auto_approval_cost_cap_usd = cost_cap_usd
        await self.persist_session_snapshot(agent_session)
        return self._auto_approval_summary(session)

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
            "auto_approval": self._auto_approval_summary(agent_session.session),
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
                        "auto_approval": {
                            "enabled": bool(row.get("auto_approval_enabled", False)),
                            "cost_cap_usd": row.get("auto_approval_cost_cap_usd"),
                            "estimated_spend_usd": float(
                                row.get("auto_approval_estimated_spend_usd") or 0.0
                            ),
                            "remaining_usd": (
                                None
                                if row.get("auto_approval_cost_cap_usd") is None
                                else round(
                                    max(
                                        0.0,
                                        float(row.get("auto_approval_cost_cap_usd") or 0.0)
                                        - float(row.get("auto_approval_estimated_spend_usd") or 0.0),
                                    ),
                                    4,
                                )
                            ),
                        },
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
