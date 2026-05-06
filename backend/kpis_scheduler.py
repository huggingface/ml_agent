"""In-process hourly KPI rollup, owned by the backend Space lifespan.

Replaces an external GitHub Actions cron so the rollup lives next to the data
and reuses the Space's existing HF token — no production secrets on the
public source repo. See ``scripts/build_kpis.py`` for the data-flow diagram
and metric definitions.

Behaviour::

    lifespan startup → start APScheduler with cron("5 * * * *", UTC)
                     → fire a best-effort 6-hour backfill (fire-and-forget)
    each :05         → run ``build_kpis.run_for_hour`` for the just-completed hour
    lifespan shutdown → scheduler.shutdown(wait=False)

Environment::

    HF_KPI_WRITE_TOKEN | HF_SESSION_UPLOAD_TOKEN | HF_TOKEN | HF_ADMIN_TOKEN
        First one found is used. Least-privilege first.
    KPI_SOURCE_REPO     default smolagents/ml-intern-sessions
    KPI_TARGET_REPO     default smolagents/ml-intern-kpis
    ML_INTERN_KPIS_DISABLED  if truthy, the scheduler is not started
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Hold strong refs to backfill tasks so asyncio doesn't GC them mid-run.
_background_tasks: set[asyncio.Task] = set()

_scheduler = None  # AsyncIOScheduler instance (lazy import)


def _resolve_token() -> Optional[str]:
    """Pick the first available HF token. Least-privilege first."""
    for var in (
        "HF_KPI_WRITE_TOKEN",
        "HF_SESSION_UPLOAD_TOKEN",
        "HF_TOKEN",
        "HF_ADMIN_TOKEN",
    ):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _load_build_kpis():
    """Import ``scripts/build_kpis.py`` without putting ``scripts/`` on sys.path."""
    spec = importlib.util.spec_from_file_location(
        "build_kpis",
        _PROJECT_ROOT / "scripts" / "build_kpis.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


async def _run_hour(hour_dt: datetime) -> None:
    """Run one hourly rollup off the event loop. Best-effort, never raises."""
    token = _resolve_token()
    if not token:
        logger.warning("kpis_scheduler: no HF token available, skipping %s", hour_dt)
        return
    try:
        mod = _load_build_kpis()
        from huggingface_hub import HfApi

        api = HfApi()
        source = os.environ.get("KPI_SOURCE_REPO", "smolagents/ml-intern-sessions")
        target = os.environ.get("KPI_TARGET_REPO", "smolagents/ml-intern-kpis")
        await asyncio.to_thread(mod.run_for_hour, api, source, target, hour_dt, token)
    except Exception as e:
        logger.warning("kpis_scheduler: rollup for %s failed: %s", hour_dt, e)


async def run_last_completed_hour() -> None:
    """The scheduled-at-:05 job. Rolls up the previous whole hour."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    await _run_hour(now - timedelta(hours=1))


async def backfill(hours: int = 6) -> None:
    """Catch-up pass for hours the Space was down. Idempotent (overwrites)."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for i in range(1, hours + 1):
        await _run_hour(now - timedelta(hours=i))


def start(backfill_hours: int = 6) -> None:
    """Called from FastAPI lifespan startup."""
    global _scheduler
    if os.environ.get("ML_INTERN_KPIS_DISABLED"):
        logger.info("kpis_scheduler: disabled via ML_INTERN_KPIS_DISABLED")
        return
    if _scheduler is not None:
        return

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("kpis_scheduler: apscheduler not installed, skipping")
        return

    _scheduler = AsyncIOScheduler(timezone="UTC")
    _scheduler.add_job(
        run_last_completed_hour,
        CronTrigger(minute=5),
        id="kpis_hourly",
        misfire_grace_time=600,  # tolerate a 10-min misfire window
        coalesce=True,  # collapse multiple missed fires into one
        max_instances=1,
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("kpis_scheduler: started (cron '5 * * * *' UTC)")

    # Non-blocking backfill. Hold a strong ref until done so asyncio doesn't
    # GC the task before it finishes.
    try:
        task = asyncio.get_running_loop().create_task(backfill(backfill_hours))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except RuntimeError:
        # Not in an event loop (tests); skip backfill.
        pass


async def shutdown() -> None:
    """Called from FastAPI lifespan shutdown."""
    global _scheduler
    if _scheduler is None:
        return
    _scheduler.shutdown(wait=False)
    _scheduler = None
    logger.info("kpis_scheduler: stopped")
