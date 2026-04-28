"""Durable session-run worker for Space background execution.

This worker consumes Mongo-backed ``session_runs``. It intentionally reuses the
existing ``SessionManager`` and agent loop instead of introducing a second agent
execution path.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import socket
import uuid
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from agent.core.session import OpType
from session_manager import AgentSession, Operation, SessionManager

logger = logging.getLogger(__name__)

TERMINAL_EVENTS = {"turn_complete", "approval_required", "error", "interrupted", "shutdown"}


def background_workers_enabled() -> bool:
    return os.environ.get("ML_INTERN_BACKGROUND_WORKERS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def in_process_worker_enabled() -> bool:
    return os.environ.get("ML_INTERN_RUN_WORKER_IN_PROCESS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def default_worker_id() -> str:
    return os.environ.get(
        "ML_INTERN_WORKER_ID",
        f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}",
    )


def operation_from_run(run: dict[str, Any]) -> Operation:
    operation = run.get("operation") or {}
    op_type = operation.get("type")
    payload = operation.get("payload") or {}

    if op_type == OpType.USER_INPUT.value:
        return Operation(op_type=OpType.USER_INPUT, data={"text": payload.get("text", "")})
    if op_type == OpType.EXEC_APPROVAL.value:
        return Operation(
            op_type=OpType.EXEC_APPROVAL,
            data={"approvals": payload.get("approvals") or []},
        )
    if op_type == OpType.UNDO.value:
        return Operation(op_type=OpType.UNDO)
    if op_type == OpType.COMPACT.value:
        return Operation(op_type=OpType.COMPACT)
    if op_type == OpType.SHUTDOWN.value:
        return Operation(op_type=OpType.SHUTDOWN)

    raise ValueError(f"Unsupported background run operation: {op_type!r}")


async def _wait_for_broadcaster(agent_session: AgentSession, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while agent_session.broadcaster is None:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("session broadcaster was not initialized")
        await asyncio.sleep(0.05)
    return agent_session.broadcaster


def _run_status_from_event(event_type: str) -> str:
    if event_type == "approval_required":
        return "waiting_approval"
    if event_type == "error":
        return "failed"
    if event_type == "interrupted":
        return "interrupted"
    return "completed"


async def _heartbeat_loop(
    store,
    *,
    run_id: str,
    worker_id: str,
    lease_seconds: int,
    interval_seconds: int,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        ok = await store.heartbeat_run(
            run_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        if not ok:
            logger.warning("Worker %s lost lease for run %s", worker_id, run_id)
            return


async def process_run(
    manager: SessionManager,
    run: dict[str, Any],
    *,
    worker_id: str,
    lease_seconds: int = 120,
    heartbeat_interval_seconds: int = 30,
) -> None:
    """Execute one claimed run and update its durable status."""
    store = manager._store()
    run_id = str(run["_id"])
    session_id = str(run["session_id"])
    user_id = str(run.get("user_id") or "dev")
    heartbeat_task: asyncio.Task | None = None
    sub_id: int | None = None
    broadcaster = None

    try:
        agent_session = await manager.ensure_session_loaded(session_id, user_id)
        if not agent_session or not agent_session.is_active:
            raise RuntimeError("session not found or inactive")

        broadcaster = await _wait_for_broadcaster(agent_session)
        sub_id, event_queue = broadcaster.subscribe()
        operation = operation_from_run(run)

        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(
                store,
                run_id=run_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                interval_seconds=heartbeat_interval_seconds,
            )
        )

        success = await manager.submit(session_id, operation)
        if not success:
            raise RuntimeError("session rejected background run submission")

        while True:
            event = await event_queue.get()
            event_type = str(event.get("event_type") or "")
            if event_type in TERMINAL_EVENTS:
                await store.finish_run(
                    run_id,
                    status=_run_status_from_event(event_type),
                    error=(event.get("data") or {}).get("error"),
                )
                logger.info("Worker %s finished run %s as %s", worker_id, run_id, event_type)
                return
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("Worker %s failed run %s", worker_id, run_id)
        await store.finish_run(run_id, status="failed", error=str(e))
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        if broadcaster is not None and sub_id is not None:
            broadcaster.unsubscribe(sub_id)


async def run_worker_loop(
    manager: SessionManager,
    *,
    worker_id: str | None = None,
    poll_interval_seconds: float = 0.25,
    idle_interval_seconds: float = 2.0,
    lease_seconds: int = 120,
    heartbeat_interval_seconds: int = 30,
) -> None:
    """Continuously claim and process queued durable runs."""
    worker_id = worker_id or default_worker_id()
    store = manager._store()
    logger.info("Background worker %s starting", worker_id)

    while True:
        await store.interrupt_expired_runs()
        run = await store.claim_next_run(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        if not run:
            await asyncio.sleep(idle_interval_seconds)
            continue

        await process_run(
            manager,
            run,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        await asyncio.sleep(poll_interval_seconds)


async def main() -> None:
    from session_manager import session_manager

    await session_manager.start()
    try:
        await run_worker_loop(session_manager)
    finally:
        await session_manager.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
