import json

from agent.core.session_uploader import (
    _PERSONAL_TOKEN_ENV,
    _resolve_token,
    _update_upload_status,
    _upload_dataset_card,
    _write_claude_code_payload,
    _write_row_payload,
    dataset_card_readme,
    to_claude_code_jsonl,
)

HF_SECRET = "hf_" + "a" * 30
ANTHROPIC_SECRET = "sk-ant-" + "b" * 24
GITHUB_SECRET = "ghp_" + "c" * 36


def test_dataset_card_readme_has_metadata_and_public_warning():
    readme = dataset_card_readme("lewtun/ml-intern-sessions")

    assert readme.startswith("---\n")
    assert 'pretty_name: "ML Intern Session Traces"' in readme
    assert "task_categories:\n- text-generation" in readme
    assert "- agent-traces" in readme
    assert "- coding-agent" in readme
    assert "- ml-intern" in readme
    assert 'path: "sessions/**/*.jsonl"' in readme
    assert "ML Intern demo: https://smolagents-ml-intern.hf.space" in readme
    assert "ML Intern CLI: https://github.com/huggingface/ml-intern" in readme
    assert "Repository: https://huggingface.co/datasets/" not in readme
    assert (
        "**WARNING: no comprehensive redaction or human review has been performed for this dataset.**"
        in readme
    )
    assert "automated best-effort scrubbing" in readme
    assert "Do not make this dataset public" in readme


def test_upload_dataset_card_only_for_claude_code_format():
    class FakeApi:
        def __init__(self):
            self.calls = []

        def upload_file(self, **kwargs):
            self.calls.append(kwargs)

    api = FakeApi()

    _upload_dataset_card(api, "lewtun/ml-intern-sessions", "hf_token", "row")
    assert api.calls == []

    _upload_dataset_card(api, "lewtun/ml-intern-sessions", "hf_token", "claude_code")
    assert len(api.calls) == 1
    assert api.calls[0]["path_in_repo"] == "README.md"
    assert api.calls[0]["repo_id"] == "lewtun/ml-intern-sessions"
    assert api.calls[0]["repo_type"] == "dataset"
    assert api.calls[0]["token"] == "hf_token"
    assert b"no comprehensive redaction or human review" in api.calls[0]["path_or_fileobj"]


def test_personal_token_env_takes_precedence_for_hf_token(monkeypatch):
    monkeypatch.setenv(_PERSONAL_TOKEN_ENV, "personal-token")
    monkeypatch.setenv("HF_TOKEN", "env-token")

    assert _resolve_token("HF_TOKEN") == "personal-token"


def test_update_upload_status_preserves_other_uploader_fields(tmp_path):
    session_file = tmp_path / "session_123.json"
    session_file.write_text(
        json.dumps(
            {
                "session_id": "123",
                "upload_status": "success",
                "upload_url": "https://huggingface.co/datasets/org/sessions",
                "personal_upload_status": "pending",
            }
        )
    )

    _update_upload_status(
        str(session_file),
        "personal_upload_status",
        "personal_upload_url",
        "success",
        "https://huggingface.co/datasets/user/ml-intern-sessions",
    )

    data = json.loads(session_file.read_text())
    assert data["upload_status"] == "success"
    assert data["upload_url"] == "https://huggingface.co/datasets/org/sessions"
    assert data["personal_upload_status"] == "success"
    assert (
        data["personal_upload_url"]
        == "https://huggingface.co/datasets/user/ml-intern-sessions"
    )


def test_claude_code_jsonl_uses_message_timestamps():
    events = to_claude_code_jsonl(
        {
            "session_id": "session-123",
            "model_name": "anthropic/claude-opus-4-6",
            "session_start_time": "2026-01-01T00:00:00",
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "timestamp": "2026-01-01T00:00:01",
                },
                {
                    "role": "assistant",
                    "content": "hi",
                    "timestamp": "2026-01-01T00:00:02",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "ok",
                    "timestamp": "2026-01-01T00:00:03",
                },
            ],
        }
    )

    assert [event["timestamp"] for event in events] == [
        "2026-01-01T00:00:01",
        "2026-01-01T00:00:02",
        "2026-01-01T00:00:03",
    ]


def test_row_payload_scrubs_messages_events_and_tools(tmp_path):
    tmp_file = tmp_path / "row.jsonl"
    data = {
        "session_id": "session-123",
        "user_id": "lewtun",
        "session_start_time": "2026-01-01T00:00:00",
        "session_end_time": "2026-01-01T00:00:03",
        "model_name": "anthropic/claude-opus-4-6",
        "total_cost_usd": 0.01,
        "messages": [{"role": "user", "content": f"token {HF_SECRET}"}],
        "events": [{"type": "debug", "content": f"key {ANTHROPIC_SECRET}"}],
        "tools": [{"name": "bash", "env": f"GITHUB_TOKEN={GITHUB_SECRET}"}],
    }

    _write_row_payload(data, str(tmp_file))

    payload = tmp_file.read_text()
    assert HF_SECRET not in payload
    assert ANTHROPIC_SECRET not in payload
    assert GITHUB_SECRET not in payload
    assert "[REDACTED_HF_TOKEN]" in payload
    assert "[REDACTED_ANTHROPIC_KEY]" in payload
    assert "GITHUB_TOKEN=[REDACTED]" in payload


def test_claude_code_payload_scrubs_messages_before_conversion(tmp_path):
    tmp_file = tmp_path / "claude_code.jsonl"
    data = {
        "session_id": "session-123",
        "model_name": "anthropic/claude-opus-4-6",
        "session_start_time": "2026-01-01T00:00:00",
        "messages": [
            {
                "role": "user",
                "content": f"token {HF_SECRET}",
                "timestamp": "2026-01-01T00:00:01",
            },
            {
                "role": "assistant",
                "content": "running tool",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"key": ANTHROPIC_SECRET}),
                        },
                    }
                ],
                "timestamp": "2026-01-01T00:00:02",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": f"GITHUB_TOKEN={GITHUB_SECRET}",
                "timestamp": "2026-01-01T00:00:03",
            },
        ],
    }

    _write_claude_code_payload(data, str(tmp_file))

    payload = tmp_file.read_text()
    assert HF_SECRET not in payload
    assert ANTHROPIC_SECRET not in payload
    assert GITHUB_SECRET not in payload
    assert "[REDACTED_HF_TOKEN]" in payload
    assert "[REDACTED_ANTHROPIC_KEY]" in payload
    assert "GITHUB_TOKEN=[REDACTED]" in payload
