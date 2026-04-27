"""Tests for agent.sft.tagger — one test per tag namespace."""

from agent.sft.tagger import tag_session


def _ev(event_type, data=None, ts="2026-04-24T10:00:00"):
    return {"timestamp": ts, "event_type": event_type, "data": data or {}}


def _traj(events=None, messages=None, model="claude-opus-4-6"):
    return {
        "session_id": "sess-1",
        "model_name": model,
        "session_start_time": "2026-04-24T09:59:00",
        "session_end_time": "2026-04-24T10:05:00",
        "messages": messages
        or [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        "events": events or [],
    }


def test_model_family():
    assert "model:opus" in tag_session(_traj(model="claude-opus-4-6"))
    assert "model:sonnet" in tag_session(_traj(model="bedrock/claude-sonnet-4-5"))
    assert "model:kimi" in tag_session(_traj(model="moonshotai/Kimi-K2.6"))
    assert "model:other" in tag_session(_traj(model="unknown-model-xyz"))


def test_turns_buckets():
    short = _traj(messages=[{"role": "user", "content": "hi"}])
    medium = _traj(messages=[{"role": "user", "content": "q"} for _ in range(10)])
    long = _traj(messages=[{"role": "user", "content": "q"} for _ in range(25)])
    assert "turns:short" in tag_session(short)
    assert "turns:medium" in tag_session(medium)
    assert "turns:long" in tag_session(long)


def test_cost_buckets():
    cheap = _traj(events=[_ev("llm_call", {"cost_usd": 0.05})])
    med = _traj(events=[_ev("llm_call", {"cost_usd": 0.5})])
    expensive = _traj(events=[_ev("llm_call", {"cost_usd": 5.0})])
    assert "cost:low" in tag_session(cheap)
    assert "cost:med" in tag_session(med)
    assert "cost:high" in tag_session(expensive)


def test_tool_tags():
    events = [
        _ev("tool_call", {"tool": "hf_jobs", "arguments": {}}),
        _ev("tool_call", {"tool": "research"}),
        _ev("tool_call", {"tool": "bash"}),
    ]
    tags = tag_session(_traj(events))
    assert "tool:hf_jobs" in tags
    assert "tool:research" in tags
    assert "tool:bash" in tags


def test_outcome_completed():
    events = [_ev("turn_complete", {"history_size": 10})]
    assert "outcome:completed" in tag_session(_traj(events))


def test_outcome_errored():
    events = [_ev("error", {"error": "boom"})]
    assert "outcome:errored" in tag_session(_traj(events))


def test_outcome_interrupted():
    events = [_ev("interrupted")]
    assert "outcome:interrupted" in tag_session(_traj(events))


def test_outcome_ongoing():
    # No terminal events → session was still running at save time
    events = [_ev("llm_call", {"cost_usd": 0.01})]
    assert "outcome:ongoing" in tag_session(_traj(events))


def test_outcome_doom_loop_and_context():
    events = [
        _ev("tool_log", {"tool": "system", "log": "Doom loop detected — injecting corrective prompt"}),
        _ev("compacted", {"old_tokens": 100, "new_tokens": 50}),
        _ev("turn_complete", {"history_size": 10}),
    ]
    tags = tag_session(_traj(events))
    assert "outcome:doom_loop" in tags
    assert "outcome:context_exceeded" in tags


def test_hf_job_tags():
    events = [
        _ev("tool_call", {"tool": "hf_jobs", "arguments": {"script": "from trl import SFTTrainer"}}),
        _ev("hf_job_submit", {
            "flavor": "a100-large", "push_to_hub": True, "job_id": "j1",
        }),
        _ev("hf_job_complete", {"flavor": "a100-large", "final_status": "COMPLETED", "wall_time_s": 3600}),
        _ev("hf_job_submit", {"flavor": "a100x4", "push_to_hub": False}),
        _ev("hf_job_complete", {"flavor": "a100x4", "final_status": "FAILED"}),
    ]
    tags = tag_session(_traj(events))
    assert "hf_job:submitted" in tags
    assert "hf_job:multi" in tags
    assert "hf_job:succeeded" in tags
    assert "hf_job:failed" in tags
    assert "hf_job:push_to_hub" in tags
    assert "gpu:a100" in tags
    assert "gpu:multi" in tags


def test_hf_job_oom():
    events = [
        _ev("tool_call", {"tool": "hf_jobs", "arguments": {}}),
        _ev("hf_job_submit", {"flavor": "a100-large"}),
        _ev("tool_output", {"success": False, "output": "RuntimeError: CUDA out of memory. Tried to allocate..."}),
    ]
    tags = tag_session(_traj(events))
    assert "hf_job:oom" in tags


def test_sandbox_tags():
    events = [
        _ev("sandbox_create", {"hardware": "t4-small", "sandbox_id": "s1", "create_latency_s": 5}),
        _ev("sandbox_destroy", {"sandbox_id": "s1", "lifetime_s": 3600}),
    ]
    tags = tag_session(_traj(events))
    assert "sandbox:created" in tags
    assert "sandbox:gpu" in tags
    assert "sandbox:long_lived" in tags


def test_sandbox_cpu_short():
    events = [
        _ev("sandbox_create", {"hardware": "cpu-basic"}),
        _ev("sandbox_destroy", {"lifetime_s": 120}),
    ]
    tags = tag_session(_traj(events))
    assert "sandbox:cpu" in tags
    assert "sandbox:long_lived" not in tags


def test_feedback_tags():
    up_only = _traj(events=[_ev("feedback", {"rating": "up"})])
    down_only = _traj(events=[_ev("feedback", {"rating": "down"})])
    mixed = _traj(events=[_ev("feedback", {"rating": "up"}), _ev("feedback", {"rating": "down"})])
    none = _traj()
    assert "feedback:up" in tag_session(up_only)
    assert "feedback:down" in tag_session(down_only)
    assert "feedback:mixed" in tag_session(mixed)
    assert "feedback:none" in tag_session(none)


def test_task_training():
    events = [
        _ev("tool_call", {"tool": "hf_jobs", "arguments": {
            "script": "from trl import SFTTrainer\ntrainer = SFTTrainer(...)"
        }}),
        _ev("hf_job_submit", {"flavor": "a100-large"}),
    ]
    assert "task:training" in tag_session(_traj(events))


def test_task_research_only():
    events = [
        _ev("tool_call", {"tool": "research"}),
        _ev("tool_call", {"tool": "explore_hf_docs"}),
    ]
    assert "task:research_only" in tag_session(_traj(events))


def test_task_data_prep():
    events = [
        _ev("tool_call", {"tool": "hf_inspect_dataset", "arguments": {}}),
        _ev("tool_call", {"tool": "hub_repo_details"}),
    ]
    tags = tag_session(_traj(events))
    assert "task:data_prep" in tags


def test_no_duplicates_and_sorted():
    events = [
        _ev("tool_call", {"tool": "hf_jobs"}),
        _ev("tool_call", {"tool": "hf_jobs"}),  # duplicate
        _ev("hf_job_submit", {"flavor": "a10g-small"}),
        _ev("hf_job_submit", {"flavor": "a10g-small"}),
    ]
    tags = tag_session(_traj(events))
    assert tags == sorted(tags)
    assert len(tags) == len(set(tags))


def test_empty_trajectory_has_required_tags():
    tags = tag_session(_traj())
    namespaces = {t.split(":", 1)[0] for t in tags}
    # Every session must have at least model/turns/cost/outcome/feedback.
    for required in ("model", "turns", "cost", "outcome", "feedback"):
        assert required in namespaces, f"missing {required} — got {tags}"
