#!/usr/bin/env python3
"""Hourly KPI rollup for the session-trajectory dataset.

================================================================================
 Data flow
================================================================================

    ┌────────────────────┐   heartbeat      ┌────────────────────────────────┐
    │  agent (CLI/web)   │ ───────────────▶ │  hf-agent-sessions  (dataset)  │
    │  Session.send_event│                  │  sessions/YYYY-MM-DD/<id>.jsonl│
    └────────────────────┘                  └───────────────┬────────────────┘
                                                            │ cron @:05 each hour
                                                            ▼
                                         ┌──────────────────────────────────┐
                                         │   scripts/build_kpis.py          │
                                         │   (GitHub Actions)               │
                                         └───────────────┬──────────────────┘
                                                         │ upload CSV
                                                         ▼
                                         ┌──────────────────────────────────┐
                                         │  hf-agent-kpis  (dataset)        │
                                         │  hourly/YYYY-MM-DD/HH.csv        │
                                         └──────────────────────────────────┘

Each hourly run reads today's + yesterday's session folders (to cover sessions
that crossed midnight), filters events into the target hour window
``[hour, hour+1h)``, computes aggregates, and writes one CSV at
``hourly/<date>/<HH>.csv`` in the target dataset. Uploads are idempotent —
re-running the same hour overwrites.

================================================================================
 Metrics (one row per hour)
================================================================================

    sessions            — distinct session_ids with ≥1 event in window
    users               — distinct user ids (when present on session rows)
    turns               — sum of user-message counts across active sessions
    llm_calls           — count of llm_call events
    tokens_prompt / _completion / _cache_read / _cache_creation
    cost_usd            — sum of llm_call.cost_usd
    cost_per_session_mean / _p50 / _p95  — per-session cost distribution
    cache_hit_ratio     — cache_read / (cache_read + prompt)
    tool_calls_total / _succeeded / _failed  — per-tool_output reliability counts
    tool_success_rate   — succeeded / total (kept for back-compat)
    successful_sessions / errored_sessions / regenerated_sessions  — outcome counts
    failure_rate / regenerate_rate  — kept for back-compat
    time_to_first_action_s_p50 / _p95  — from session_start to first tool_call
    thumbs_up / thumbs_down
    hf_jobs_submitted / _succeeded / _blocked
    sandboxes_created / _cpu / _gpu  — sandbox_create events bucketed by hardware
    pro_cta_clicks
    gpu_hours_by_flavor_json   — JSON-serialised {flavor: gpu-hours}
    research_calls             — total `research` tool_call events
    sessions_with_research     — sessions that called `research` ≥1
    research_calls_per_session_p50 / _p95 — among sessions that did any (zero-only sessions excluded)
    distinct_tools_per_session_p50 / _p95 — among sessions with ≥1 named tool_call
    tool_calls_per_session_p50 / _p95     — among sessions with ≥1 named tool_call
    tool_calls_per_turn_p50 / _p95        — calls / turns, among sessions with turns>0
    tool_calls_by_name_json    — JSON {tool: total_calls} (all tools seen)
    sessions_using_tool_json   — JSON {tool: distinct_sessions_using}
    sessions_by_model_json     — JSON {model_name: count} (CLI vs Bedrock split)

================================================================================
 Usage
================================================================================

    # Run for the most recently completed hour (default — the cron path):
    python scripts/build_kpis.py

    # Backfill last 24 hours:
    python scripts/build_kpis.py --hours 24

    # Explicit hour (UTC):
    python scripts/build_kpis.py --datetime 2026-04-24T14

Env:
    HF_TOKEN (or HF_KPI_WRITE_TOKEN) — write access to the target dataset.

================================================================================
 Deploy
================================================================================

See ``.github/workflows/build-kpis.yml`` — runs every hour at :05. To provision:

    1. Create the target dataset (once):
         huggingface-cli repo create hf-agent-kpis --type dataset
    2. Put ``HF_KPI_WRITE_TOKEN`` (or ``HF_TOKEN``) into repo Actions secrets.
    3. Merge this file; the first scheduled run fires within the hour.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger("build_kpis")

# Rough gpu-hour pricing for hf_jobs flavor strings. Keep conservative; used
# only to compute gpu-hours (not dollars) — wall_time_s * flavor_gpu_count.
_FLAVOR_GPU_COUNT = {
    "cpu-basic": 0,
    "cpu-upgrade": 0,
    "t4-small": 1,
    "t4-medium": 1,
    "l4x1": 1,
    "l4x4": 4,
    "l40sx1": 1,
    "l40sx4": 4,
    "l40sx8": 8,
    "a10g-small": 1,
    "a10g-large": 1,
    "a10g-largex2": 2,
    "a10g-largex4": 4,
    "a100-large": 1,
    "a100x2": 2,
    "a100x4": 4,
    "a100x8": 8,
    "h100": 1,
    "h100x8": 8,
}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[f])
    return float(values[f] + (values[c] - values[f]) * (k - f))


def _parse_ts(s: Any) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    # Normalise to aware UTC so comparisons work against window bounds.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iter_session_files(api, repo_id: str, day: date, token: str) -> Iterable[str]:
    """Yield repo-relative paths for all sessions under ``sessions/YYYY-MM-DD/``."""
    prefix = f"sessions/{day.isoformat()}/"
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception as e:
        logger.warning("list_repo_files(%s) failed: %s", repo_id, e)
        return []
    return [f for f in files if f.startswith(prefix) and f.endswith(".jsonl")]


def _download_session(repo_id: str, path: str, token: str) -> dict | None:
    """Fetch one session JSONL and decode its single row.

    ``hf_hub_download`` caches; second run within the same process / runner
    directory is near-free.
    """
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=path,
            repo_type="dataset",
            token=token,
        )
    except Exception as e:
        logger.warning("hf_hub_download(%s) failed: %s", path, e)
        return None
    try:
        with open(local, "r") as f:
            line = f.readline().strip()
        if not line:
            return None
        row = json.loads(line)
        # Session uploader stores messages/events as JSON strings — unpack.
        for key in ("messages", "events", "tools"):
            v = row.get(key)
            if isinstance(v, str):
                try:
                    row[key] = json.loads(v)
                except Exception:
                    row[key] = []
        return row
    except Exception as e:
        logger.warning("parse(%s) failed: %s", path, e)
        return None


def _filter_session_to_window(
    session: dict,
    start: datetime,
    end: datetime,
) -> dict | None:
    """Return a copy of ``session`` whose events are only those in ``[start, end)``.

    ``None`` if no event falls in the window — the caller drops the session
    from this hour's aggregate.
    """
    events = session.get("events") or []
    in_window = []
    for ev in events:
        ts = _parse_ts(ev.get("timestamp"))
        if ts is None:
            continue
        if start <= ts < end:
            in_window.append(ev)
    if not in_window:
        return None
    return {**session, "events": in_window}


def _session_metrics(session: dict) -> dict:
    """Reduce a single session trajectory to its KPI contributions.

    Assumes ``events`` are already filtered to the target window by the caller.
    """
    # Pre-seed every numeric key so downstream aggregation can sum without
    # having to special-case empty sessions.
    out: dict = {
        "sessions": 0,
        "turns": 0,
        "llm_calls": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_cache_read": 0,
        "tokens_cache_creation": 0,
        "cost_usd": 0.0,
        "tool_calls_total": 0,
        "tool_calls_success": 0,
        "failures": 0,
        "regenerate_sessions": 0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        "hf_jobs_submitted": 0,
        "hf_jobs_succeeded": 0,
        "hf_jobs_blocked": 0,
        "pro_cta_clicks": 0,
        "pro_conversions": 0,
        "credits_topped_up": 0,
        "sandboxes_created": 0,
        "sandboxes_cpu": 0,
        "sandboxes_gpu": 0,
        "first_tool_s": -1,
    }
    events = session.get("events") or []
    messages = session.get("messages") or []

    turn_count = sum(1 for m in messages if m.get("role") == "user")
    out["turns"] = turn_count
    out["sessions"] = 1

    tool_success = 0
    tool_total = 0
    had_error = False
    had_undo = False
    first_tool_ts = None
    session_start = session.get("session_start_time")
    gpu_hours_by_flavor: dict[str, float] = defaultdict(float)
    jobs_submitted = 0
    jobs_succeeded = 0
    thumbs_up = 0
    thumbs_down = 0
    sandboxes_created = 0
    sandboxes_cpu = 0
    sandboxes_gpu = 0
    jobs_blocked = 0
    pro_cta_clicks = 0
    pro_conversions = 0
    credits_topped_up = 0
    pro_cta_by_source: dict[str, int] = defaultdict(int)
    # Per-tool counters from tool_call events. Counted off tool_call (which
    # carries data["tool"]) rather than tool_output (which only carries
    # success/output) so we can attribute calls to specific tools.
    tool_calls_by_name: dict[str, int] = defaultdict(int)
    total_named_tool_calls = 0

    start_dt = _parse_ts(session_start)

    for ev in events:
        et = ev.get("event_type")
        data = ev.get("data") or {}
        ts = _parse_ts(ev.get("timestamp"))

        if et == "llm_call":
            out["llm_calls"] += 1
            out["tokens_prompt"] += int(data.get("prompt_tokens") or 0)
            out["tokens_completion"] += int(data.get("completion_tokens") or 0)
            out["tokens_cache_read"] += int(data.get("cache_read_tokens") or 0)
            out["tokens_cache_creation"] += int(data.get("cache_creation_tokens") or 0)
            out["cost_usd"] += float(data.get("cost_usd") or 0.0)

        elif et == "tool_output":
            tool_total += 1
            if data.get("success"):
                tool_success += 1
            if first_tool_ts is None and ts is not None and start_dt is not None:
                first_tool_ts = (ts - start_dt).total_seconds()

        elif et == "tool_call":
            name = data.get("tool")
            if name:
                tool_calls_by_name[name] += 1
                total_named_tool_calls += 1
            if first_tool_ts is None and ts is not None and start_dt is not None:
                first_tool_ts = (ts - start_dt).total_seconds()

        elif et == "error":
            had_error = True

        elif et == "undo_complete":
            had_undo = True

        elif et == "feedback":
            rating = data.get("rating")
            if rating == "up":
                thumbs_up += 1
            elif rating == "down":
                thumbs_down += 1

        elif et == "hf_job_submit":
            jobs_submitted += 1

        elif et == "hf_job_complete":
            flavor = data.get("flavor") or "unknown"
            status = (data.get("final_status") or "").lower()
            wall = float(data.get("wall_time_s") or 0.0)
            gpus = _FLAVOR_GPU_COUNT.get(flavor, 0)
            gpu_hours_by_flavor[flavor] += wall * gpus / 3600.0
            if status in ("completed", "succeeded", "success"):
                jobs_succeeded += 1

        elif et == "jobs_access_blocked":
            jobs_blocked += 1

        elif et == "pro_cta_click":
            pro_cta_clicks += 1
            source = str(data.get("source") or "unknown")
            pro_cta_by_source[source] += 1

        elif et == "pro_conversion":
            pro_conversions += 1

        elif et == "credits_topped_up":
            credits_topped_up += 1

        elif et == "sandbox_create":
            sandboxes_created += 1
            hardware = (data.get("hardware") or "").lower()
            # CPU flavors are explicitly named "cpu-*". Everything else
            # (including unknown/missing hardware strings) lands in the GPU
            # bucket, since the auto-create default is "cpu-basic" which is
            # matched here — anything that isn't is almost always an explicit
            # GPU choice.
            if hardware.startswith("cpu-"):
                sandboxes_cpu += 1
            else:
                sandboxes_gpu += 1

    out["tool_calls_total"] = tool_total
    out["tool_calls_success"] = tool_success
    out["failures"] = 1 if had_error else 0
    out["regenerate_sessions"] = 1 if had_undo else 0
    out["thumbs_up"] = thumbs_up
    out["thumbs_down"] = thumbs_down
    out["hf_jobs_submitted"] = jobs_submitted
    out["hf_jobs_succeeded"] = jobs_succeeded
    out["sandboxes_created"] = sandboxes_created
    out["sandboxes_cpu"] = sandboxes_cpu
    out["sandboxes_gpu"] = sandboxes_gpu
    out["hf_jobs_blocked"] = jobs_blocked
    out["pro_cta_clicks"] = pro_cta_clicks
    out["pro_conversions"] = pro_conversions
    out["credits_topped_up"] = credits_topped_up
    out["first_tool_s"] = first_tool_ts if first_tool_ts is not None else -1
    out["_gpu_hours_by_flavor"] = dict(gpu_hours_by_flavor)
    out["_pro_cta_by_source"] = dict(pro_cta_by_source)
    out["_user"] = session.get("user_id") or session.get("session_id")
    # Intra-session tool fields. Underscore-prefixed = consumed by _aggregate
    # only, never written to CSV directly.
    out["_tool_calls_by_name"] = dict(tool_calls_by_name)
    out["_research_calls"] = tool_calls_by_name.get("research", 0)
    out["_distinct_tools_used"] = len(tool_calls_by_name)
    out["_total_named_tool_calls"] = total_named_tool_calls
    out["_model_name"] = session.get("model_name") or "unknown"
    return dict(out)


def _aggregate(per_session: list[dict]) -> dict:
    """Collapse a bucket's worth of session rollups into the final KPI row."""
    ttfa_values = [
        s["first_tool_s"] for s in per_session if s.get("first_tool_s", -1) >= 0
    ]
    gpu_hours: dict[str, float] = defaultdict(float)
    for s in per_session:
        for f, h in (s.get("_gpu_hours_by_flavor") or {}).items():
            gpu_hours[f] += h

    # Per-tool aggregates. ``sessions_using_tool`` counts each session at most
    # once per tool, so the dashboard can show "how many sessions reached for
    # research" alongside "how many research calls overall".
    tool_calls_by_name: dict[str, int] = defaultdict(int)
    sessions_using_tool: dict[str, int] = defaultdict(int)
    sessions_by_model: dict[str, int] = defaultdict(int)
    for s in per_session:
        for name, count in (s.get("_tool_calls_by_name") or {}).items():
            tool_calls_by_name[name] += int(count)
            sessions_using_tool[name] += 1
        sessions_by_model[s.get("_model_name") or "unknown"] += 1

    # Percentile inputs. All "per session" percentiles exclude sessions that
    # never reached for the relevant signal — otherwise quiet hours
    # (status-check sessions, abandoned new conversations) drag every median
    # to 0 and the chart tells you nothing.
    research_calls_nz = [
        s.get("_research_calls", 0)
        for s in per_session
        if s.get("_research_calls", 0) > 0
    ]
    distinct_tools_values = [
        s.get("_distinct_tools_used", 0)
        for s in per_session
        if s.get("_distinct_tools_used", 0) > 0
    ]
    total_calls_values = [
        s.get("_total_named_tool_calls", 0)
        for s in per_session
        if s.get("_total_named_tool_calls", 0) > 0
    ]
    # Per-turn intensity: turns>0 is the natural filter here (a session with
    # 5 turns and 0 tools is a meaningful 0). Don't strip those.
    calls_per_turn_values = [
        s.get("_total_named_tool_calls", 0) / s["turns"]
        for s in per_session
        if s.get("turns", 0) > 0
    ]

    total_sessions = sum(s["sessions"] for s in per_session)
    total_turns = sum(s["turns"] for s in per_session)
    tokens_prompt = sum(s["tokens_prompt"] for s in per_session)
    tokens_cache_read = sum(s["tokens_cache_read"] for s in per_session)
    tool_total = sum(s["tool_calls_total"] for s in per_session)
    tool_success = sum(s["tool_calls_success"] for s in per_session)
    failures = int(sum(s["failures"] for s in per_session))
    regenerates = int(sum(s["regenerate_sessions"] for s in per_session))
    research_calls_total = int(sum(s.get("_research_calls", 0) for s in per_session))
    sessions_with_research = sum(
        1 for s in per_session if s.get("_research_calls", 0) > 0
    )

    # Per-session cost percentiles — chart "median session cost" alongside the
    # mean so a few $700 outliers don't make you think every session is pricey.
    session_costs = [float(s.get("cost_usd") or 0.0) for s in per_session]
    cost_p50 = _percentile(session_costs, 0.5)
    cost_p95 = _percentile(session_costs, 0.95)

    unique_users = {s.get("_user") for s in per_session if s.get("_user")}

    return {
        "sessions": total_sessions,
        "users": len(unique_users),
        "turns": total_turns,
        "llm_calls": int(sum(s["llm_calls"] for s in per_session)),
        "tokens_prompt": int(tokens_prompt),
        "tokens_completion": int(sum(s["tokens_completion"] for s in per_session)),
        "tokens_cache_read": int(tokens_cache_read),
        "tokens_cache_creation": int(
            sum(s["tokens_cache_creation"] for s in per_session)
        ),
        "cost_usd": round(sum(s["cost_usd"] for s in per_session), 4),
        # Per-session cost summaries.
        "cost_per_session_mean": round(
            sum(s["cost_usd"] for s in per_session) / total_sessions, 6
        )
        if total_sessions > 0
        else 0.0,
        "cost_per_session_p50": round(cost_p50, 6),
        "cost_per_session_p95": round(cost_p95, 6),
        "cache_hit_ratio": round(
            tokens_cache_read / (tokens_cache_read + tokens_prompt), 4
        )
        if (tokens_cache_read + tokens_prompt) > 0
        else 0.0,
        # Raw reliability COUNTS (these are what the dashboard shows directly).
        "tool_calls_total": int(tool_total),
        "tool_calls_succeeded": int(tool_success),
        "tool_calls_failed": int(tool_total - tool_success),
        "errored_sessions": failures,
        # Successful = "did not raise an error event". Mutually exclusive
        # with errored_sessions; sums with errored_sessions to total sessions.
        "successful_sessions": int(total_sessions - failures),
        # Regenerated is an orthogonal dimension (the user retried) — a
        # session can be both successful and regenerated, or both errored
        # and regenerated.
        "regenerated_sessions": regenerates,
        # Rates kept for backwards compatibility with anything reading the
        # KPI dataset directly.
        "tool_success_rate": round(tool_success / tool_total, 4)
        if tool_total > 0
        else 0.0,
        "failure_rate": round(failures / total_sessions, 4)
        if total_sessions > 0
        else 0.0,
        "regenerate_rate": round(regenerates / total_sessions, 4)
        if total_sessions > 0
        else 0.0,
        "time_to_first_action_s_p50": round(_percentile(ttfa_values, 0.5), 2),
        "time_to_first_action_s_p95": round(_percentile(ttfa_values, 0.95), 2),
        "thumbs_up": int(sum(s["thumbs_up"] for s in per_session)),
        "thumbs_down": int(sum(s["thumbs_down"] for s in per_session)),
        "hf_jobs_submitted": int(sum(s["hf_jobs_submitted"] for s in per_session)),
        "hf_jobs_succeeded": int(sum(s["hf_jobs_succeeded"] for s in per_session)),
        "sandboxes_created": int(
            sum(s.get("sandboxes_created", 0) for s in per_session)
        ),
        "sandboxes_cpu": int(sum(s.get("sandboxes_cpu", 0) for s in per_session)),
        "sandboxes_gpu": int(sum(s.get("sandboxes_gpu", 0) for s in per_session)),
        "hf_jobs_blocked": int(sum(s.get("hf_jobs_blocked", 0) for s in per_session)),
        "pro_cta_clicks": int(sum(s.get("pro_cta_clicks", 0) for s in per_session)),
        "pro_conversions": int(sum(s.get("pro_conversions", 0) for s in per_session)),
        "credits_topped_up": int(
            sum(s.get("credits_topped_up", 0) for s in per_session)
        ),
        "gpu_hours_by_flavor_json": json.dumps(dict(gpu_hours), sort_keys=True),
        # Research KPIs — answer "is the agent reaching for research?".
        "research_calls": research_calls_total,
        "sessions_with_research": int(sessions_with_research),
        "research_calls_per_session_p50": round(_percentile(research_calls_nz, 0.5), 2),
        "research_calls_per_session_p95": round(
            _percentile(research_calls_nz, 0.95), 2
        ),
        # Intra-session breadth + intensity. p50 + p95 over per-session values.
        "distinct_tools_per_session_p50": round(
            _percentile(distinct_tools_values, 0.5), 2
        ),
        "distinct_tools_per_session_p95": round(
            _percentile(distinct_tools_values, 0.95), 2
        ),
        "tool_calls_per_session_p50": round(_percentile(total_calls_values, 0.5), 2),
        "tool_calls_per_session_p95": round(_percentile(total_calls_values, 0.95), 2),
        "tool_calls_per_turn_p50": round(_percentile(calls_per_turn_values, 0.5), 2),
        "tool_calls_per_turn_p95": round(_percentile(calls_per_turn_values, 0.95), 2),
        # JSON columns let the dashboard add/remove tools without schema churn.
        "tool_calls_by_name_json": json.dumps(dict(tool_calls_by_name), sort_keys=True),
        "sessions_using_tool_json": json.dumps(
            dict(sessions_using_tool), sort_keys=True
        ),
        # Surface split — answers "is research dropping on Bedrock specifically?".
        "sessions_by_model_json": json.dumps(dict(sessions_by_model), sort_keys=True),
    }


# Back-compat alias: older tests call _aggregate_day.
_aggregate_day = _aggregate


def _csv_cell(v: Any) -> str:
    s = str(v)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _write_csv(
    api,
    row: dict,
    bucket_key: str,
    path_in_repo: str,
    target_repo: str,
    token: str,
) -> None:
    """Render ``row`` to CSV with a leading ``bucket`` column and upload.

    ``bucket_key`` is the hour string (ISO ``YYYY-MM-DDTHH``) or date string;
    written as the ``bucket`` column so downstream consumers can union all
    CSVs without date-parsing paths. ``api`` is the caller's ``HfApi``
    instance — reused so we don't spin up a fresh one per CSV.
    """
    columns = list(row.keys())
    buf = io.StringIO()
    buf.write(",".join(["bucket", *columns]) + "\n")
    buf.write(",".join([bucket_key, *[_csv_cell(row[c]) for c in columns]]) + "\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write(buf.getvalue())
        tmp_path = tmp.name

    try:
        api.create_repo(
            repo_id=target_repo,
            repo_type="dataset",
            exist_ok=True,
            token=token,
        )
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            repo_id=target_repo,
            repo_type="dataset",
            token=token,
            commit_message=f"KPIs for {bucket_key}",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def run_for_hour(
    api,
    source_repo: str,
    target_repo: str,
    hour_dt: datetime,
    token: str,
) -> dict:
    """Roll up one UTC hour [hour_dt, hour_dt+1h).

    Reads today's + yesterday's session folders so sessions that crossed
    midnight land in the right hourly bucket.
    """
    if hour_dt.tzinfo is None:
        hour_dt = hour_dt.replace(tzinfo=timezone.utc)
    window_start = hour_dt.replace(minute=0, second=0, microsecond=0)
    window_end = window_start + timedelta(hours=1)

    # Sessions partition by session_start_time date. A session that started
    # at 23:50 yesterday can still emit events in today's first hours, so we
    # look at both folders.
    candidate_dates = {window_start.date(), (window_start - timedelta(days=1)).date()}

    per_session: list[dict] = []
    for d in sorted(candidate_dates):
        for path in _iter_session_files(api, source_repo, d, token):
            sess = _download_session(source_repo, path, token)
            if not sess:
                continue
            windowed = _filter_session_to_window(sess, window_start, window_end)
            if windowed is None:
                continue
            per_session.append(_session_metrics(windowed))

    if not per_session:
        logger.info("No sessions in window %s — skipping", window_start.isoformat())
        return {}

    row = _aggregate(per_session)
    bucket_key = window_start.strftime("%Y-%m-%dT%H")
    path_in_repo = (
        f"hourly/{window_start.strftime('%Y-%m-%d')}/{window_start.strftime('%H')}.csv"
    )
    _write_csv(api, row, bucket_key, path_in_repo, target_repo, token)
    logger.info(
        "Wrote KPIs for %s (%d sessions): %s",
        bucket_key,
        per_session and len(per_session),
        row,
    )
    return row


# Back-compat for daily backfills — unchanged behaviour.
def run_for_day(api, source_repo: str, target_repo: str, day: date, token: str) -> dict:
    paths = _iter_session_files(api, source_repo, day, token)
    per_session: list[dict] = []
    for path in paths:
        sess = _download_session(source_repo, path, token)
        if not sess:
            continue
        per_session.append(_session_metrics(sess))
    if not per_session:
        logger.info("No sessions found for %s — skipping", day)
        return {}
    row = _aggregate(per_session)
    path_in_repo = f"daily/{day.isoformat()}.csv"
    _write_csv(api, row, day.isoformat(), path_in_repo, target_repo, token)
    return row


def _parse_hour_arg(s: str) -> datetime:
    """Accept ``YYYY-MM-DDTHH`` or full ISO — always pinned to the start of the hour, UTC."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="smolagents/ml-intern-sessions")
    ap.add_argument("--target", default="smolagents/ml-intern-kpis")
    ap.add_argument(
        "--hours",
        type=int,
        default=1,
        help="Number of trailing hours to roll up (default: 1 = last completed hour).",
    )
    ap.add_argument(
        "--datetime",
        type=str,
        default=None,
        help="Single hour, ISO ``YYYY-MM-DDTHH`` (UTC); overrides --hours.",
    )
    ap.add_argument(
        "--daily-backfill",
        type=str,
        default=None,
        help="Escape hatch: aggregate a whole day at once (YYYY-MM-DD). "
        "Writes to daily/<date>.csv. Use for historical backfill only.",
    )
    args = ap.parse_args(argv)

    token = (
        os.environ.get("HF_KPI_WRITE_TOKEN")
        or os.environ.get("HF_SESSION_UPLOAD_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_ADMIN_TOKEN")
    )
    if not token:
        logger.error(
            "No HF token found. Set one of: HF_KPI_WRITE_TOKEN, "
            "HF_SESSION_UPLOAD_TOKEN, HF_TOKEN, HF_ADMIN_TOKEN."
        )
        return 1

    from huggingface_hub import HfApi

    api = HfApi()

    if args.daily_backfill:
        run_for_day(
            api,
            args.source,
            args.target,
            date.fromisoformat(args.daily_backfill),
            token,
        )
        return 0

    if args.datetime:
        target_hours = [_parse_hour_arg(args.datetime)]
    else:
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        # Roll up *completed* hours: start from the hour before ``now``.
        target_hours = [now - timedelta(hours=i) for i in range(1, args.hours + 1)]

    for hour in target_hours:
        run_for_hour(api, args.source, args.target, hour, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
