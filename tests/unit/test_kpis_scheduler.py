"""Smoke tests for backend/kpis_scheduler.py.

Exercise the pure / fast paths only:
    * token resolution order
    * build_kpis import path
    * start()/shutdown() lifecycle without APScheduler actually running a job
    * backfill() passes the right hour values through to _run_hour
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load():
    path = Path(__file__).parent.parent.parent / "backend" / "kpis_scheduler.py"
    spec = importlib.util.spec_from_file_location("kpis_scheduler", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kpis_scheduler"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_token_resolution_order(monkeypatch):
    mod = _load()
    for var in ("HF_KPI_WRITE_TOKEN", "HF_SESSION_UPLOAD_TOKEN", "HF_TOKEN", "HF_ADMIN_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert mod._resolve_token() is None

    monkeypatch.setenv("HF_ADMIN_TOKEN", "admin")
    assert mod._resolve_token() == "admin"

    monkeypatch.setenv("HF_TOKEN", "generic")
    assert mod._resolve_token() == "generic"

    monkeypatch.setenv("HF_SESSION_UPLOAD_TOKEN", "sessions")
    assert mod._resolve_token() == "sessions"

    monkeypatch.setenv("HF_KPI_WRITE_TOKEN", "kpis")
    assert mod._resolve_token() == "kpis"


def test_load_build_kpis_exposes_run_for_hour():
    mod = _load()
    bk = mod._load_build_kpis()
    assert hasattr(bk, "run_for_hour")
    assert callable(bk.run_for_hour)


def test_backfill_calls_run_hour_for_each_hour(monkeypatch):
    mod = _load()
    monkeypatch.setenv("HF_KPI_WRITE_TOKEN", "x")
    calls: list[datetime] = []

    async def fake_run_hour(hour_dt):
        calls.append(hour_dt)

    monkeypatch.setattr(mod, "_run_hour", fake_run_hour)
    asyncio.run(mod.backfill(hours=3))
    assert len(calls) == 3
    # Hours are returned most-recent-first
    assert calls[0] > calls[1] > calls[2]
    # All aligned to the top of the hour
    for c in calls:
        assert c.minute == 0 and c.second == 0 and c.microsecond == 0
        assert c.tzinfo == timezone.utc


def test_start_is_no_op_when_disabled(monkeypatch):
    mod = _load()
    # Ensure clean state — _scheduler is module-global
    mod._scheduler = None
    monkeypatch.setenv("ML_INTERN_KPIS_DISABLED", "1")
    mod.start()
    assert mod._scheduler is None  # never instantiated


def test_start_skips_cleanly_without_apscheduler(monkeypatch):
    mod = _load()
    mod._scheduler = None
    monkeypatch.delenv("ML_INTERN_KPIS_DISABLED", raising=False)

    # Force the apscheduler import to fail — start() should log and return.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("apscheduler"):
            raise ImportError("apscheduler unavailable in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "builtins.__import__",
        fake_import,
    )
    mod.start()  # should not raise
    assert mod._scheduler is None


def test_shutdown_is_no_op_when_not_started():
    mod = _load()
    mod._scheduler = None
    asyncio.run(mod.shutdown())  # must not raise
