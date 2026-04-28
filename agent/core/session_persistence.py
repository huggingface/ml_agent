"""Optional durable session persistence for the hosted backend.

The public CLI must keep working without MongoDB.  This module therefore
exposes one small async store interface and returns a no-op implementation
unless ``MONGODB_URI`` is configured and reachable.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from bson import BSON
from pymongo import AsyncMongoClient, DeleteMany, ReturnDocument, UpdateOne
from pymongo.errors import DuplicateKeyError, InvalidDocument, PyMongoError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_BSON_BYTES = 15 * 1024 * 1024


def _now() -> datetime:
    return datetime.now(UTC)


def _doc_id(session_id: str, idx: int) -> str:
    return f"{session_id}:{idx}"


def _safe_message_doc(message: dict[str, Any]) -> dict[str, Any]:
    """Return a Mongo-safe message document payload.

    Mongo's hard document limit is 16 MB.  We stay below that and store an
    explicit marker rather than failing the whole snapshot for one huge tool log.
    """
    try:
        if len(BSON.encode({"message": message})) <= MAX_BSON_BYTES:
            return message
    except (InvalidDocument, OverflowError):
        pass
    return {
        "role": "tool",
        "content": (
            "[SYSTEM: A single persisted message exceeded MongoDB's document "
            "size/encoding limit and was replaced by this marker.]"
        ),
        "ml_intern_persistence_error": "message_too_large_or_invalid",
    }


class NoopSessionStore:
    """Async no-op store used when Mongo is not configured."""

    enabled = False

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_session(self, **_: Any) -> None:
        return None

    async def save_snapshot(self, **_: Any) -> None:
        return None

    async def load_session(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def list_sessions(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def soft_delete_session(self, *_: Any, **__: Any) -> None:
        return None

    async def update_session_fields(self, *_: Any, **__: Any) -> None:
        return None

    async def append_event(self, *_: Any, **__: Any) -> int | None:
        return None

    async def load_events_after(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def latest_event_seq(self, *_: Any, **__: Any) -> int:
        return 0

    async def append_trace_message(self, *_: Any, **__: Any) -> int | None:
        return None

    async def get_quota(self, *_: Any, **__: Any) -> int | None:
        return None

    async def try_increment_quota(self, *_: Any, **__: Any) -> int | None:
        return None

    async def refund_quota(self, *_: Any, **__: Any) -> None:
        return None

    async def enqueue_run(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def claim_next_run(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def heartbeat_run(self, *_: Any, **__: Any) -> bool:
        return False

    async def finish_run(self, *_: Any, **__: Any) -> None:
        return None

    async def interrupt_expired_runs(self, *_: Any, **__: Any) -> int:
        return 0


class MongoSessionStore(NoopSessionStore):
    """MongoDB-backed session store."""

    enabled = True

    def __init__(self, uri: str, db_name: str) -> None:
        self.uri = uri
        self.db_name = db_name
        self.enabled = False
        self.client: AsyncMongoClient | None = None
        self.db = None

    async def init(self) -> None:
        try:
            self.client = AsyncMongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self.db = self.client[self.db_name]
            await self.client.admin.command("ping")
            await self._create_indexes()
            self.enabled = True
            logger.info("Mongo session persistence enabled (db=%s)", self.db_name)
        except Exception as e:
            logger.warning("Mongo session persistence disabled: %s", e)
            self.enabled = False
            if self.client is not None:
                await self.client.close()
            self.client = None
            self.db = None

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
        self.client = None
        self.db = None

    async def _create_indexes(self) -> None:
        if self.db is None:
            return
        await self.db.sessions.create_index(
            [("user_id", 1), ("visibility", 1), ("updated_at", -1)]
        )
        await self.db.sessions.create_index(
            [("visibility", 1), ("status", 1), ("last_active_at", -1)]
        )
        await self.db.session_messages.create_index(
            [("session_id", 1), ("idx", 1)], unique=True
        )
        await self.db.session_events.create_index(
            [("session_id", 1), ("seq", 1)], unique=True
        )
        await self.db.session_trace_messages.create_index(
            [("session_id", 1), ("seq", 1)], unique=True
        )
        await self.db.session_trace_messages.create_index([("created_at", -1)])
        await self.db.session_runs.create_index(
            [("status", 1), ("lease_until", 1), ("created_at", 1)]
        )
        await self.db.session_runs.create_index([("session_id", 1), ("created_at", -1)])
        await self.db.session_runs.create_index([("user_id", 1), ("created_at", -1)])
        await self.db.session_runs.create_index(
            [("idempotency_key", 1)], unique=True, sparse=True
        )

    def _ready(self) -> bool:
        return bool(self.enabled and self.db is not None)

    async def upsert_session(
        self,
        *,
        session_id: str,
        user_id: str,
        model: str,
        title: str | None = None,
        surface: str = "frontend",
        created_at: datetime | None = None,
        runtime_state: str = "idle",
        status: str = "active",
        message_count: int = 0,
        turn_count: int = 0,
        pending_approval: list[dict[str, Any]] | None = None,
        claude_counted: bool = False,
        notification_destinations: list[str] | None = None,
    ) -> None:
        if not self._ready():
            return
        now = _now()
        await self.db.sessions.update_one(
            {"_id": session_id},
            {
                "$setOnInsert": {
                    "_id": session_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "surface": surface,
                    "created_at": created_at or now,
                    "schema_version": SCHEMA_VERSION,
                    "visibility": "live",
                },
                "$set": {
                    "title": title,
                    "model": model,
                    "status": status,
                    "runtime_state": runtime_state,
                    "updated_at": now,
                    "last_active_at": now,
                    "message_count": message_count,
                    "turn_count": turn_count,
                    "pending_approval": pending_approval or [],
                    "claude_counted": claude_counted,
                    "notification_destinations": notification_destinations or [],
                },
            },
            upsert=True,
        )

    async def save_snapshot(
        self,
        *,
        session_id: str,
        user_id: str,
        model: str,
        messages: list[dict[str, Any]],
        title: str | None = None,
        runtime_state: str = "idle",
        status: str = "active",
        turn_count: int = 0,
        pending_approval: list[dict[str, Any]] | None = None,
        claude_counted: bool = False,
        created_at: datetime | None = None,
        notification_destinations: list[str] | None = None,
    ) -> None:
        if not self._ready():
            return
        now = _now()
        await self.upsert_session(
            session_id=session_id,
            user_id=user_id,
            model=model,
            title=title,
            created_at=created_at,
            runtime_state=runtime_state,
            status=status,
            message_count=len(messages),
            turn_count=turn_count,
            pending_approval=pending_approval,
            claude_counted=claude_counted,
            notification_destinations=notification_destinations,
        )
        ops: list[Any] = []
        for idx, raw in enumerate(messages):
            ops.append(
                UpdateOne(
                    {"_id": _doc_id(session_id, idx)},
                    {
                        "$set": {
                            "session_id": session_id,
                            "idx": idx,
                            "message": _safe_message_doc(raw),
                            "updated_at": now,
                        },
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            )
        ops.append(DeleteMany({"session_id": session_id, "idx": {"$gte": len(messages)}}))
        try:
            if ops:
                await self.db.session_messages.bulk_write(ops, ordered=False)
        except PyMongoError as e:
            logger.warning("Failed to persist session %s snapshot: %s", session_id, e)

    async def load_session(
        self, session_id: str, *, include_deleted: bool = False
    ) -> dict[str, Any] | None:
        if not self._ready():
            return None
        meta = await self.db.sessions.find_one({"_id": session_id})
        if not meta:
            return None
        if meta.get("visibility") == "deleted" and not include_deleted:
            return None
        cursor = self.db.session_messages.find({"session_id": session_id}).sort("idx", 1)
        messages = [row.get("message") async for row in cursor]
        return {"metadata": meta, "messages": messages}

    async def list_sessions(
        self, user_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        if not self._ready():
            return []
        query: dict[str, Any] = {"user_id": user_id}
        if user_id == "dev":
            query = {}
        if not include_deleted:
            query["visibility"] = {"$ne": "deleted"}
        cursor = self.db.sessions.find(query).sort("updated_at", -1)
        return [row async for row in cursor]

    async def soft_delete_session(self, session_id: str) -> None:
        if not self._ready():
            return
        await self.db.sessions.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "visibility": "deleted",
                    "runtime_state": "idle",
                    "updated_at": _now(),
                }
            },
        )

    async def update_session_fields(self, session_id: str, **fields: Any) -> None:
        if not self._ready() or not fields:
            return
        fields["updated_at"] = _now()
        await self.db.sessions.update_one({"_id": session_id}, {"$set": fields})

    async def _next_seq(self, counter_id: str) -> int:
        doc = await self.db.counters.find_one_and_update(
            {"_id": counter_id},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["seq"])

    async def append_event(
        self, session_id: str, event_type: str, data: dict[str, Any] | None
    ) -> int | None:
        if not self._ready():
            return None
        try:
            seq = await self._next_seq(f"event:{session_id}")
            await self.db.session_events.insert_one(
                {
                    "_id": _doc_id(session_id, seq),
                    "session_id": session_id,
                    "seq": seq,
                    "event_type": event_type,
                    "data": data or {},
                    "created_at": _now(),
                }
            )
            return seq
        except PyMongoError as e:
            logger.debug("Failed to append event for %s: %s", session_id, e)
            return None

    async def load_events_after(self, session_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        if not self._ready():
            return []
        cursor = self.db.session_events.find(
            {"session_id": session_id, "seq": {"$gt": int(after_seq or 0)}}
        ).sort("seq", 1)
        return [row async for row in cursor]

    async def latest_event_seq(self, session_id: str) -> int:
        if not self._ready():
            return 0
        doc = await self.db.session_events.find_one(
            {"session_id": session_id},
            sort=[("seq", -1)],
        )
        return int(doc.get("seq", 0)) if doc else 0

    async def append_trace_message(
        self, session_id: str, message: dict[str, Any], source: str = "message"
    ) -> int | None:
        if not self._ready():
            return None
        try:
            seq = await self._next_seq(f"trace:{session_id}")
            await self.db.session_trace_messages.insert_one(
                {
                    "_id": _doc_id(session_id, seq),
                    "session_id": session_id,
                    "seq": seq,
                    "role": message.get("role"),
                    "message": _safe_message_doc(message),
                    "source": source,
                    "created_at": _now(),
                }
            )
            return seq
        except PyMongoError as e:
            logger.debug("Failed to append trace message for %s: %s", session_id, e)
            return None

    async def get_quota(self, user_id: str, day: str) -> int | None:
        if not self._ready():
            return None
        doc = await self.db.claude_quotas.find_one({"_id": f"{user_id}:{day}"})
        return int(doc.get("count", 0)) if doc else 0

    async def try_increment_quota(self, user_id: str, day: str, cap: int) -> int | None:
        if not self._ready():
            return None
        key = f"{user_id}:{day}"
        now = _now()
        try:
            await self.db.claude_quotas.insert_one(
                {
                    "_id": key,
                    "user_id": user_id,
                    "day": day,
                    "count": 1,
                    "updated_at": now,
                }
            )
            return 1
        except DuplicateKeyError:
            pass
        doc = await self.db.claude_quotas.find_one_and_update(
            {"_id": key, "count": {"$lt": cap}},
            {"$inc": {"count": 1}, "$set": {"updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["count"]) if doc else None

    async def refund_quota(self, user_id: str, day: str) -> None:
        if not self._ready():
            return
        await self.db.claude_quotas.update_one(
            {"_id": f"{user_id}:{day}", "count": {"$gt": 0}},
            {"$inc": {"count": -1}, "$set": {"updated_at": _now()}},
        )

    async def enqueue_run(
        self,
        *,
        session_id: str,
        user_id: str,
        operation: dict[str, Any],
        idempotency_key: str | None = None,
        surface: str = "space",
    ) -> dict[str, Any] | None:
        """Create a durable queued run and attach it to the session.

        Returns None when the session already has an active run. The caller can
        surface that as a 409 rather than starting two concurrent turns against
        the same context.
        """
        if not self._ready():
            return None
        if idempotency_key:
            existing = await self.db.session_runs.find_one(
                {"idempotency_key": idempotency_key}
            )
            if existing:
                return existing

        now = _now()
        run_id = str(uuid.uuid4())
        run = {
            "_id": run_id,
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "user_id": user_id,
            "surface": surface,
            "operation": operation,
            "status": "queued",
            "idempotency_key": idempotency_key,
            "lease_owner": None,
            "lease_until": None,
            "retry_count": 0,
            "max_retries": 1,
            "created_at": now,
            "started_at": None,
            "updated_at": now,
            "finished_at": None,
            "error": None,
        }
        try:
            await self.db.session_runs.insert_one(run)
        except DuplicateKeyError:
            if idempotency_key:
                return await self.db.session_runs.find_one(
                    {"idempotency_key": idempotency_key}
                )
            raise

        attached = await self.db.sessions.update_one(
            {
                "_id": session_id,
                "$or": [
                    {"active_run_id": {"$exists": False}},
                    {"active_run_id": None},
                ],
            },
            {
                "$set": {
                    "active_run_id": run_id,
                    "runtime_state": "queued",
                    "updated_at": now,
                }
            },
        )
        if attached.matched_count == 0:
            await self.db.session_runs.update_one(
                {"_id": run_id},
                {
                    "$set": {
                        "status": "cancelled",
                        "error": "session already has an active run",
                        "updated_at": _now(),
                        "finished_at": _now(),
                    }
                },
            )
            return None
        return run

    async def claim_next_run(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 120,
    ) -> dict[str, Any] | None:
        """Atomically claim the oldest queued run for a worker."""
        if not self._ready():
            return None
        now = _now()
        lease_until = now + timedelta(seconds=lease_seconds)
        run = await self.db.session_runs.find_one_and_update(
            {"status": "queued"},
            {
                "$set": {
                    "status": "running",
                    "lease_owner": worker_id,
                    "lease_until": lease_until,
                    "started_at": now,
                    "updated_at": now,
                },
                "$inc": {"retry_count": 1},
            },
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )
        if not run:
            return None

        guarded = await self.db.sessions.update_one(
            {"_id": run["session_id"], "active_run_id": run["_id"]},
            {
                "$set": {
                    "runtime_state": "processing",
                    "worker_owner": worker_id,
                    "worker_lease_until": lease_until,
                    "updated_at": now,
                }
            },
        )
        if guarded.matched_count == 0:
            await self.finish_run(
                run["_id"],
                status="interrupted",
                error="session guard failed while claiming run",
            )
            return None
        return run

    async def heartbeat_run(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 120,
    ) -> bool:
        if not self._ready():
            return False
        now = _now()
        lease_until = now + timedelta(seconds=lease_seconds)
        result = await self.db.session_runs.update_one(
            {"_id": run_id, "status": "running", "lease_owner": worker_id},
            {"$set": {"lease_until": lease_until, "updated_at": now}},
        )
        if result.matched_count:
            run = await self.db.session_runs.find_one({"_id": run_id})
            if run:
                await self.db.sessions.update_one(
                    {"_id": run["session_id"], "active_run_id": run_id},
                    {
                        "$set": {
                            "worker_lease_until": lease_until,
                            "updated_at": now,
                        }
                    },
                )
        return bool(result.matched_count)

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        if not self._ready():
            return
        now = _now()
        run = await self.db.session_runs.find_one_and_update(
            {"_id": run_id},
            {
                "$set": {
                    "status": status,
                    "error": error,
                    "updated_at": now,
                    "finished_at": now,
                    "lease_until": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not run:
            return

        runtime_state = {
            "completed": "idle",
            "waiting_approval": "waiting_approval",
            "failed": "idle",
            "cancelled": "idle",
            "interrupted": "interrupted",
        }.get(status, "idle")
        await self.db.sessions.update_one(
            {"_id": run["session_id"], "active_run_id": run_id},
            {
                "$set": {
                    "runtime_state": runtime_state,
                    "active_run_id": None,
                    "worker_owner": None,
                    "worker_lease_until": None,
                    "updated_at": now,
                }
            },
        )

    async def interrupt_expired_runs(self, *, before: datetime | None = None) -> int:
        """Mark expired running runs interrupted instead of replaying tool calls."""
        if not self._ready():
            return 0
        now = _now()
        cutoff = before or now
        cursor = self.db.session_runs.find(
            {"status": "running", "lease_until": {"$lt": cutoff}}
        )
        count = 0
        async for run in cursor:
            await self.finish_run(
                run["_id"],
                status="interrupted",
                error="worker lease expired",
            )
            count += 1
        return count


_store: NoopSessionStore | MongoSessionStore | None = None


def get_session_store() -> NoopSessionStore | MongoSessionStore:
    global _store
    if _store is None:
        uri = os.environ.get("MONGODB_URI")
        db_name = os.environ.get("MONGODB_DB", "ml-intern")
        _store = MongoSessionStore(uri, db_name) if uri else NoopSessionStore()
    return _store


def _reset_store_for_tests(store: NoopSessionStore | MongoSessionStore | None = None) -> None:
    global _store
    _store = store
