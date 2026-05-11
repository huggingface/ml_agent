import asyncio
from types import SimpleNamespace

from agent.core.session import Session


class DummyToolRouter:
    def get_tool_specs_for_llm(self) -> list[dict]:
        return []


def _session(*, user_id: str | None, hf_username: str | None) -> Session:
    config = SimpleNamespace(
        model_name="moonshotai/Kimi-K2.6",
        save_sessions=True,
        share_traces=True,
        personal_trace_repo_template="{hf_user}/ml-intern-sessions",
        session_dataset_repo="smolagents/ml-intern-sessions",
        auto_save_interval=1,
        heartbeat_interval_s=0,
        reasoning_effort=None,
    )
    context_manager = SimpleNamespace(items=[], on_message_added=None)
    return Session(
        event_queue=asyncio.Queue(),
        config=config,
        tool_router=DummyToolRouter(),
        context_manager=context_manager,
        user_id=user_id,
        hf_username=hf_username,
    )


def test_personal_trace_repo_uses_hf_username_before_oauth_subject():
    session = _session(user_id="oauth-subject", hf_username="lewtun")

    assert session._personal_trace_repo_id() == "lewtun/ml-intern-sessions"


def test_personal_trace_repo_falls_back_to_user_id_for_cli():
    session = _session(user_id="lewtun", hf_username=None)

    assert session._personal_trace_repo_id() == "lewtun/ml-intern-sessions"
