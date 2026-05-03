"""FastAPI application for HF Agent web interface."""

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
    await session_manager.close()


app = FastAPI(
    title="HF Agent",
    description="ML Engineering Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
allow_origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# Add production HF Spaces URL when deployed
space_host = os.environ.get("SPACE_HOST")
if space_host:
    allow_origins.append(f"https://{space_host}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
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
