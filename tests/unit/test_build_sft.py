"""Smoke test for the SFT reshape — raw passthrough with tags attached."""

import importlib.util
import sys
from pathlib import Path


def _load():
    path = Path(__file__).parent.parent.parent / "scripts" / "build_sft.py"
    spec = importlib.util.spec_from_file_location("build_sft", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_sft"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _session_row():
    return {
        "session_id": "abc",
        "session_start_time": "2026-04-24T10:00:00",
        "session_end_time": "2026-04-24T10:05:00",
        "model_name": "claude-opus-4-6",
        "messages": [
            {"role": "system", "content": "You are an agent"},
            {"role": "user", "content": "fine-tune llama"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "hf_jobs",
                            "arguments": '{"script":"from trl import SFTTrainer"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
        "events": [
            {
                "timestamp": "2026-04-24T10:00:05",
                "event_type": "tool_call",
                "data": {
                    "tool": "hf_jobs",
                    "arguments": {"script": "from trl import SFTTrainer"},
                },
            },
            {
                "timestamp": "2026-04-24T10:00:06",
                "event_type": "hf_job_submit",
                "data": {"flavor": "a100-large", "push_to_hub": True},
            },
            {
                "timestamp": "2026-04-24T10:45:00",
                "event_type": "hf_job_complete",
                "data": {
                    "flavor": "a100-large",
                    "final_status": "COMPLETED",
                    "wall_time_s": 2700,
                },
            },
            {
                "timestamp": "2026-04-24T10:45:05",
                "event_type": "turn_complete",
                "data": {},
            },
        ],
        "tools": [{"type": "function", "function": {"name": "hf_jobs"}}],
    }


def test_reshape_preserves_messages_and_tools_and_adds_tags():
    mod = _load()
    row = mod._reshape_to_sft(_session_row())
    assert row["session_id"] == "abc"
    assert row["model"] == "claude-opus-4-6"
    assert row["timestamp"] == "2026-04-24T10:00:00"
    # Messages preserved verbatim, in order, with tool_calls + tool role rows.
    assert len(row["messages"]) == 5
    assert row["messages"][2]["tool_calls"][0]["function"]["name"] == "hf_jobs"
    assert row["messages"][3]["role"] == "tool"
    # Tools preserved verbatim.
    assert row["tools"] == [{"type": "function", "function": {"name": "hf_jobs"}}]
    # Tags include the expected signals.
    tags = set(row["tags"])
    assert "tool:hf_jobs" in tags
    assert "hf_job:succeeded" in tags
    assert "hf_job:push_to_hub" in tags
    assert "gpu:a100" in tags
    assert "outcome:completed" in tags
    assert "task:training" in tags
    assert "model:opus" in tags


def test_reshape_handles_missing_tools_field():
    mod = _load()
    row = _session_row()
    del row["tools"]
    out = mod._reshape_to_sft(row)
    assert out["tools"] == []
    assert isinstance(out["tags"], list)  # still computes tags
