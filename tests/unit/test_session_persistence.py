"""Unit tests for the optional durable session store abstraction."""

import pytest

from agent.core.session_persistence import NoopSessionStore, _safe_message_doc


@pytest.mark.asyncio
async def test_noop_store_keeps_local_cli_and_tests_db_free():
    store = NoopSessionStore()

    await store.init()
    await store.upsert_session(session_id="s1", user_id="u1", model="m")
    await store.save_snapshot(
        session_id="s1",
        user_id="u1",
        model="m",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert await store.load_session("s1") is None
    assert await store.list_sessions("u1") == []
    assert await store.append_event("s1", "processing", {}) is None
    assert await store.try_increment_quota("u1", "2099-01-01", 1) is None
    assert await store.latest_event_seq("s1") == 0
    assert await store.enqueue_run(session_id="s1", user_id="u1", operation={}) is None
    assert await store.claim_next_run(worker_id="w1") is None
    assert await store.heartbeat_run("r1", worker_id="w1") is False
    assert await store.interrupt_expired_runs() == 0
    await store.finish_run("r1", status="completed")


def test_unsafe_message_payload_is_replaced_with_marker():
    marker = _safe_message_doc({"role": "assistant", "content": object()})

    assert marker["role"] == "tool"
    assert marker["ml_intern_persistence_error"] == "message_too_large_or_invalid"
