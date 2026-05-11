"""Unit tests for the optional durable session store abstraction."""

import pytest

from agent.core.session_persistence import (
    MongoSessionStore,
    NoopSessionStore,
    _safe_message_doc,
)


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


def test_unsafe_message_payload_is_replaced_with_marker():
    marker = _safe_message_doc({"role": "assistant", "content": object()})

    assert marker["role"] == "tool"
    assert marker["ml_intern_persistence_error"] == "message_too_large_or_invalid"


# ── mark_pro_seen ─────────────────────────────────────────────────────────


class _FakeProUsers:
    """In-memory stand-in for the ``pro_users`` collection.

    Supports just enough of the Motor API to exercise ``mark_pro_seen``:
    ``update_one`` with ``$setOnInsert`` + ``$set`` + ``upsert=True``, and
    ``find_one_and_update`` with the guarded filter the conversion check uses.
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    async def update_one(self, filt, update, upsert=False):
        _id = filt["_id"]
        doc = self.docs.get(_id)
        if doc is None and upsert:
            doc = dict(update.get("$setOnInsert") or {})
            self.docs[_id] = doc
        if doc is None:
            return
        for k, v in (update.get("$set") or {}).items():
            doc[k] = v

    async def find_one_and_update(self, filt, update, return_document=None):
        _id = filt["_id"]
        doc = self.docs.get(_id)
        if doc is None:
            return None
        # Guard checks the conversion test uses: ever_non_pro=True AND
        # first_seen_pro_at missing.
        for k, v in filt.items():
            if k == "_id":
                continue
            if isinstance(v, dict) and "$exists" in v:
                if v["$exists"] and k not in doc:
                    return None
                if not v["$exists"] and k in doc:
                    return None
            elif doc.get(k) != v:
                return None
        for k, v in (update.get("$set") or {}).items():
            doc[k] = v
        return dict(doc)


class _FakeDB:
    def __init__(self) -> None:
        self.pro_users = _FakeProUsers()


def _store_with_fake_db() -> MongoSessionStore:
    s = MongoSessionStore.__new__(MongoSessionStore)
    s.enabled = True
    s.db = _FakeDB()
    return s


@pytest.mark.asyncio
async def test_mark_pro_seen_returns_none_when_unknown_user_starts_pro():
    """Joining as Pro shouldn't count as a conversion."""
    store = _store_with_fake_db()
    assert await store.mark_pro_seen("u-new-pro", is_pro=True) is None


@pytest.mark.asyncio
async def test_mark_pro_seen_emits_conversion_after_seeing_user_as_free():
    store = _store_with_fake_db()
    assert await store.mark_pro_seen("u1", is_pro=False) is None
    result = await store.mark_pro_seen("u1", is_pro=True)
    assert result is not None
    assert result["converted"] is True
    assert isinstance(result["first_seen_at"], str)


@pytest.mark.asyncio
async def test_mark_pro_seen_only_fires_conversion_once():
    """Re-checking a converted user must not re-emit the event."""
    store = _store_with_fake_db()
    await store.mark_pro_seen("u1", is_pro=False)
    first = await store.mark_pro_seen("u1", is_pro=True)
    assert first is not None and first["converted"] is True
    second = await store.mark_pro_seen("u1", is_pro=True)
    assert second is None


@pytest.mark.asyncio
async def test_noop_store_mark_pro_seen_returns_none():
    store = NoopSessionStore()
    assert await store.mark_pro_seen("u1", is_pro=True) is None
    assert await store.mark_pro_seen("u1", is_pro=False) is None
