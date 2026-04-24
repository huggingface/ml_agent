"""Unit tests for the KPI rollup math.

We exercise the pure functions (``_session_metrics`` and ``_aggregate_day``)
on hand-crafted session trajectories — no network, no HF Hub.
"""

import importlib.util
import sys
from pathlib import Path


def _load():
    """Load ``scripts/build_kpis.py`` without treating ``scripts`` as a package."""
    path = Path(__file__).parent.parent.parent / "scripts" / "build_kpis.py"
    spec = importlib.util.spec_from_file_location("build_kpis", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_kpis"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _ev(event_type, data=None, ts="2026-04-24T10:00:00"):
    return {"timestamp": ts, "event_type": event_type, "data": data or {}}


def _session(events, user_id="u1", start="2026-04-24T09:59:00"):
    return {
        "session_id": "sess-" + user_id,
        "session_start_time": start,
        "session_end_time": "2026-04-24T10:05:00",
        "model_name": "claude-opus-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "events": events,
        "user_id": user_id,
    }


def test_llm_call_accumulates_tokens_and_cost():
    mod = _load()
    events = [
        _ev("llm_call", {
            "prompt_tokens": 100, "completion_tokens": 50,
            "cache_read_tokens": 40, "cache_creation_tokens": 10,
            "cost_usd": 0.01,
        }),
        _ev("llm_call", {
            "prompt_tokens": 200, "completion_tokens": 100,
            "cache_read_tokens": 80, "cost_usd": 0.02,
        }),
    ]
    m = mod._session_metrics(_session(events))
    assert m["llm_calls"] == 2
    assert m["tokens_prompt"] == 300
    assert m["tokens_completion"] == 150
    assert m["tokens_cache_read"] == 120
    assert m["tokens_cache_creation"] == 10
    assert abs(m["cost_usd"] - 0.03) < 1e-9


def test_tool_success_rate_and_first_action():
    mod = _load()
    events = [
        _ev("tool_call", {"tool": "bash"}, ts="2026-04-24T10:00:05"),
        _ev("tool_output", {"success": True}),
        _ev("tool_output", {"success": False}),
    ]
    m = mod._session_metrics(_session(events))
    assert m["tool_calls_total"] == 2
    assert m["tool_calls_success"] == 1
    # 65s from start to first action
    assert m["first_tool_s"] == 65


def test_hf_job_gpu_hours():
    mod = _load()
    events = [
        _ev("hf_job_submit", {"flavor": "a100-large", "job_id": "j1"}),
        _ev("hf_job_complete", {
            "flavor": "a100-large",
            "final_status": "COMPLETED",
            "wall_time_s": 3600,
        }),
    ]
    m = mod._session_metrics(_session(events))
    assert m["hf_jobs_submitted"] == 1
    assert m["hf_jobs_succeeded"] == 1
    # a100-large = 1 gpu * 1 hour = 1 gpu-hour
    assert abs(m["_gpu_hours_by_flavor"]["a100-large"] - 1.0) < 1e-6


def test_feedback_counts():
    mod = _load()
    events = [
        _ev("feedback", {"rating": "up"}),
        _ev("feedback", {"rating": "up"}),
        _ev("feedback", {"rating": "down"}),
    ]
    m = mod._session_metrics(_session(events))
    assert m["thumbs_up"] == 2
    assert m["thumbs_down"] == 1


def test_aggregate_day_cache_hit_and_users():
    mod = _load()
    s1 = mod._session_metrics(_session(
        [_ev("llm_call", {"prompt_tokens": 100, "cache_read_tokens": 400, "cost_usd": 0.5})],
        user_id="u1",
    ))
    s2 = mod._session_metrics(_session(
        [_ev("llm_call", {"prompt_tokens": 200, "cache_read_tokens": 100, "cost_usd": 1.0})],
        user_id="u2",
    ))
    row = mod._aggregate_day([s1, s2])
    assert row["sessions"] == 2
    assert row["users"] == 2
    assert row["tokens_prompt"] == 300
    assert row["tokens_cache_read"] == 500
    # 500 / (500 + 300) = 0.625
    assert abs(row["cache_hit_ratio"] - 0.625) < 1e-9
    assert abs(row["cost_usd"] - 1.5) < 1e-9


def test_failure_and_regenerate_rates():
    mod = _load()
    s1 = mod._session_metrics(_session([_ev("error", {"error": "boom"})], user_id="a"))
    s2 = mod._session_metrics(_session([_ev("undo_complete")], user_id="b"))
    s3 = mod._session_metrics(_session([], user_id="c"))
    row = mod._aggregate_day([s1, s2, s3])
    assert row["failure_rate"] == round(1 / 3, 4)
    assert row["regenerate_rate"] == round(1 / 3, 4)


def test_window_filter_keeps_only_events_in_range():
    from datetime import datetime, timezone
    mod = _load()
    events = [
        _ev("llm_call", {"prompt_tokens": 100}, ts="2026-04-24T09:45:00"),
        _ev("llm_call", {"prompt_tokens": 200}, ts="2026-04-24T10:05:00"),
        _ev("tool_call", {"tool": "bash"}, ts="2026-04-24T10:30:00"),
        _ev("llm_call", {"prompt_tokens": 400}, ts="2026-04-24T11:10:00"),
    ]
    session = _session(events, start="2026-04-24T09:44:00")
    # Only events in [10:00, 11:00) should remain.
    window_start = datetime(2026, 4, 24, 10, 0, 0, tzinfo=timezone.utc)
    window_end = datetime(2026, 4, 24, 11, 0, 0, tzinfo=timezone.utc)
    windowed = mod._filter_session_to_window(session, window_start, window_end)
    assert windowed is not None
    types = [e["event_type"] for e in windowed["events"]]
    assert types == ["llm_call", "tool_call"]
    # Metrics only reflect in-window events.
    m = mod._session_metrics(windowed)
    assert m["tokens_prompt"] == 200
    assert m["llm_calls"] == 1
    assert m["tool_calls_total"] == 0  # tool_call not tool_output


def test_window_filter_returns_none_when_nothing_in_range():
    from datetime import datetime, timezone
    mod = _load()
    events = [_ev("llm_call", {"prompt_tokens": 100}, ts="2026-04-24T09:45:00")]
    session = _session(events)
    window_start = datetime(2026, 4, 24, 10, 0, 0, tzinfo=timezone.utc)
    window_end = datetime(2026, 4, 24, 11, 0, 0, tzinfo=timezone.utc)
    assert mod._filter_session_to_window(session, window_start, window_end) is None
