"""HTTP health wrapper for the background worker Space."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from background_worker import default_worker_id, run_worker_loop
from session_manager import session_manager

logger = logging.getLogger(__name__)

_worker_task: asyncio.Task | None = None
_worker_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_task, _worker_id

    _worker_id = default_worker_id()
    logger.info("Starting worker Space app (%s)", _worker_id)
    await session_manager.start()
    _worker_task = asyncio.create_task(
        run_worker_loop(session_manager, worker_id=_worker_id),
        name="ml-intern-worker",
    )
    try:
        yield
    finally:
        logger.info("Stopping worker Space app (%s)", _worker_id)
        if _worker_task is not None:
            _worker_task.cancel()
            try:
                await _worker_task
            except asyncio.CancelledError:
                pass
            _worker_task = None
        await session_manager.close()


app = FastAPI(
    title="ML Intern Worker",
    description="Background worker for durable ML Intern session runs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "worker_id": _worker_id,
        "worker_running": bool(_worker_task and not _worker_task.done()),
    }
