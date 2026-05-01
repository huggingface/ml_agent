"""FastAPI application for HF Agent web interface."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing routes/session_manager so persistence and quota
# modules see local Mongo settings during startup.
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.agent import router as agent_router
from routes.auth import router as auth_router
from session_manager import session_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting HF Agent backend...")
    await session_manager.start()
    # Start in-process hourly KPI rollup. Replaces an external cron so the
    # rollup lives next to the data and reuses the Space's HF token.
    try:
        import kpis_scheduler
        kpis_scheduler.start()
    except Exception as e:
        logger.warning("KPI scheduler failed to start: %s", e)
    yield

    logger.info("Shutting down HF Agent backend...")
    try:
        import kpis_scheduler
        await kpis_scheduler.shutdown()
    except Exception as e:
        logger.warning("KPI scheduler shutdown failed: %s", e)

    # Final-flush: save every still-active session so we don't lose traces on
    # server restart. Uploads are detached subprocesses — this is fast.
    try:
        for sid, agent_session in list(session_manager.sessions.items()):
            sess = agent_session.session
            if sess.config.save_sessions:
                try:
                    sess.save_and_upload_detached(sess.config.session_dataset_repo)
                    logger.info("Flushed session %s on shutdown", sid)
                except Exception as e:
                    logger.warning("Failed to flush session %s: %s", sid, e)
    except Exception as e:
        logger.warning("Lifespan final-flush skipped: %s", e)

    # Lease handover sweep — for sessions still mid-turn when Main shuts
    # down, emit a migrating event then release the lease so a Worker can
    # pick them up. Idle sessions just rehydrate normally on next request
    # and don't need this dance.
    try:
        for sid, agent_session in list(session_manager.sessions.items()):
            runtime_state = session_manager._runtime_state(agent_session)
            if runtime_state == "processing" or agent_session.is_processing:
                try:
                    await session_manager.release_session_to_background(
                        sid,
                        reason="main_shutdown",
                    )
                except Exception as e:
                    logger.warning(
                        "Lease handover sweep failed for %s: %s", sid, e
                    )
    except Exception as e:
        logger.warning("Lifespan lease sweep skipped: %s", e)
    await session_manager.close()


app = FastAPI(
    title="HF Agent",
    description="ML Engineering Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent_router)
app.include_router(auth_router)

# Serve static files (frontend build) in production
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
    logger.info(f"Serving static files from {static_path}")
else:
    logger.info("No static directory found, running in API-only mode")


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "name": "HF Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


# ── Worker mode entrypoint ───────────────────────────────────────────────


async def _worker_claim_tick() -> None:
    """One pass: find sessions with pending submissions and no live lease,
    claim each via ``claim_dormant_session`` so the existing
    ``_consume_submissions`` loop in ``_run_session`` will pick up their
    pending docs.

    Sessions already held by this process (in ``session_manager.sessions``)
    are skipped — heartbeat keeps their leases alive.
    """
    store = session_manager._store()
    if not getattr(store, "enabled", False):
        return
    db = getattr(store, "db", None)
    if db is None:
        return

    held = set(session_manager.sessions.keys())
    cursor = db.pending_submissions.find(
        {"status": "pending"}, {"session_id": 1}
    ).limit(200)
    candidate_session_ids: set[str] = set()
    async for doc in cursor:
        sid = doc.get("session_id")
        if sid and sid not in held:
            candidate_session_ids.add(sid)

    for sid in candidate_session_ids:
        try:
            # claim_dormant_session does claim_lease internally; if claim
            # fails (another process holds it) we'll just try the next
            # session in the next tick.
            await session_manager.claim_dormant_session(sid)
        except Exception as e:
            logger.warning(f"Worker failed to claim {sid}: {e}")


async def worker_loop() -> None:
    """Worker mode entrypoint. Initializes a SessionManager in worker mode,
    polls Mongo for pending submissions across all sessions, claims their
    leases, and runs their agent loops.

    Heartbeat (US-002) renews held leases on a 10s cadence; the grace
    sweeper (US-006) auto-backgrounds inactive held sessions; idle
    eviction (US-007) drops fully idle ones.
    """
    logger.info("Starting ml-intern worker mode...")
    # Use the global session_manager (already imported); it has read MODE
    # at construction time.
    await session_manager.start()
    try:
        # Single coordinator task: scan ``pending_submissions`` across
        # all sessions and claim those with no current holder. Polling
        # cadence is 1s; that's enough for v1.
        while True:
            try:
                await _worker_claim_tick()
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker claim tick error: {e}")
                await asyncio.sleep(2.0)
    finally:
        await session_manager.close()
        logger.info("Worker shut down cleanly.")
