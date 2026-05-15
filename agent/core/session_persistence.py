"""Optional durable session persistence for the hosted backend.

The public CLI must keep working without MongoDB.  This module therefore
exposes one small async store interface and returns a no-op implementation
unless ``MONGODB_URI`` is configured and reachable.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from bson import BSON, ObjectId
from pymongo import AsyncMongoClient, DeleteMany, ReturnDocument, UpdateOne
from pymongo.errors import DuplicateKeyError, InvalidDocument, PyMongoError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_BSON_BYTES = 15 * 1024 * 1024


def _now() -> datetime:
    return datetime.now(UTC)


def _doc_id(session_id: str, idx: int) -> str:
    return f"{session_id}:{idx}"


def make_holder_id(mode: str) -> str:
    """Build a process holder id ``f"{mode}:{hostname}:{8-hex-suffix}"``.

    Uses ``uuid7`` if available (Python ≥ 3.13) for chronological ordering;
    falls back to ``uuid4`` otherwise. Pick once at process start; do not
    change mid-run.
    """
    hostname = socket.gethostname()
    if hasattr(uuid, "uuid7"):
        suffix = uuid.uuid7().hex[:8]  # type: ignore[attr-defined]
    else:
        suffix = uuid.uuid4().hex[:8]
    return f"{mode}:{hostname}:{suffix}"


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

    async def append_trace_message(self, *_: Any, **__: Any) -> int | None:
        return None

    async def get_quota(self, *_: Any, **__: Any) -> int | None:
        return None

    async def try_increment_quota(self, *_: Any, **__: Any) -> int | None:
        return None

    async def refund_quota(self, *_: Any, **__: Any) -> None:
        return None

    async def mark_pro_seen(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    # ── Lease + pending-submission control plane (no-op) ──────────────────

    async def claim_lease(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def renew_lease(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def release_lease(self, *_: Any, **__: Any) -> None:
        return None

    async def enqueue_pending_submission(self, *_: Any, **__: Any) -> str:
        return ""

    async def claim_pending_submission(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return None

    async def mark_submission_done(self, *_: Any, **__: Any) -> None:
        return None

    async def requeue_claimed_for(self, *_: Any, **__: Any) -> int:
        return 0

    async def change_stream_pending_submissions(self, *_: Any, **__: Any):
        raise NotImplementedError("change streams require Mongo persistence")
        yield  # pragma: no cover - makes this an async generator

    async def change_stream_events(self, *_: Any, **__: Any):
        raise NotImplementedError("change streams require Mongo persistence")
        yield  # pragma: no cover - makes this an async generator

    async def poll_pending_submissions_after(
        self, *_: Any, **__: Any
    ) -> list[dict[str, Any]]:
        return []


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
            await self._backfill_lease_state()
            logger.info("Mongo session persistence enabled (db=%s)", self.db_name)
        except Exception as e:
            logger.warning("Mongo session persistence disabled: %s", e)
            self.enabled = False
            if self.client is not None:
                await self.client.close()
            self.client = None
            self.db = None

    async def _backfill_lease_state(self) -> None:
        """One-shot migration for sessions predating the lease control plane.

        Idempotent: the ``lease: {$exists: false}`` filter excludes already
        migrated rows, so re-running ``init()`` is a no-op.
        """
        if self.db is None:
            return
        try:
            cutoff = _now() - timedelta(hours=1)
            recent = await self.db.sessions.update_many(
                {
                    "lease": {"$exists": False},
                    "status": "active",
                    "last_active_at": {"$gt": cutoff},
                },
                {"$set": {"lease": {"holder_id": None, "expires_at": _now()}}},
            )
            old = await self.db.sessions.update_many(
                {
                    "lease": {"$exists": False},
                    "status": "active",
                    "last_active_at": {"$lte": cutoff},
                },
                {"$set": {"runtime_state": "idle"}},
            )
            logger.info(
                f"Backfilled empty lease on {recent.modified_count} sessions; "
                f"flipped {old.modified_count} old sessions to idle."
            )
        except PyMongoError as e:
            logger.warning(f"Lease backfill skipped due to Mongo error: {e}")

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
        await self.db.pro_users.create_index([("first_seen_pro_at", -1)])
        await self.db.pending_submissions.create_index(
            [("session_id", 1), ("status", 1), ("created_at", 1)]
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
        auto_approval_enabled: bool = False,
        auto_approval_cost_cap_usd: float | None = None,
        auto_approval_estimated_spend_usd: float = 0.0,
        encrypted_credential: str | None = None,
        credential_set_at: datetime | None = None,
    ) -> None:
        if not self._ready():
            return
        now = _now()
        set_fields: dict[str, Any] = {
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
            "auto_approval_enabled": auto_approval_enabled,
            "auto_approval_cost_cap_usd": auto_approval_cost_cap_usd,
            "auto_approval_estimated_spend_usd": auto_approval_estimated_spend_usd,
        }
        # Only overwrite the encrypted credential when the caller supplies
        # a fresh ciphertext. Snapshot saves (e.g. mid-turn) don't carry
        # the token and must not blank the field that Worker depends on.
        # ``credential_set_at`` is only written when explicitly given —
        # this preserves the original issue time across snapshot saves
        # so the Worker's TTL check is against the token's real age,
        # not against the most recent state-persistence write.
        if encrypted_credential is not None:
            set_fields["encrypted_credential"] = encrypted_credential
            if credential_set_at is not None:
                set_fields["credential_set_at"] = credential_set_at
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
                "$set": set_fields,
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
        auto_approval_enabled: bool = False,
        auto_approval_cost_cap_usd: float | None = None,
        auto_approval_estimated_spend_usd: float = 0.0,
        encrypted_credential: str | None = None,
        credential_set_at: datetime | None = None,
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
            auto_approval_enabled=auto_approval_enabled,
            auto_approval_cost_cap_usd=auto_approval_cost_cap_usd,
            auto_approval_estimated_spend_usd=auto_approval_estimated_spend_usd,
            encrypted_credential=encrypted_credential,
            credential_set_at=credential_set_at,
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

    async def mark_pro_seen(
        self, user_id: str, *, is_pro: bool
    ) -> dict[str, Any] | None:
        """Track per-user Pro state and detect free→Pro conversions.

        Returns ``{"converted": True, "first_seen_at": ..."}`` exactly once
        per user — the first time we see them as Pro after having recorded
        them as non-Pro at least once. Otherwise returns ``None``.

        Storing ``ever_non_pro`` lets us distinguish "user joined as Pro"
        (no conversion) from "user upgraded" (conversion). The atomic
        ``find_one_and_update`` on a guarded filter makes the conversion
        emit at-most-once even under concurrent requests.
        """
        if not self._ready() or not user_id:
            return None
        now = _now()
        set_fields: dict[str, Any] = {"last_seen_at": now, "is_pro": bool(is_pro)}
        if not is_pro:
            set_fields["ever_non_pro"] = True
        try:
            await self.db.pro_users.update_one(
                {"_id": user_id},
                {
                    "$setOnInsert": {"_id": user_id, "first_seen_at": now},
                    "$set": set_fields,
                },
                upsert=True,
            )
        except PyMongoError as e:
            logger.debug("mark_pro_seen upsert failed for %s: %s", user_id, e)
            return None

        if not is_pro:
            return None

        try:
            doc = await self.db.pro_users.find_one_and_update(
                {
                    "_id": user_id,
                    "ever_non_pro": True,
                    "first_seen_pro_at": {"$exists": False},
                },
                {"$set": {"first_seen_pro_at": now}},
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as e:
            logger.debug("mark_pro_seen conversion check failed for %s: %s", user_id, e)
            return None

        if not doc:
            return None
        return {
            "converted": True,
            "first_seen_at": (doc.get("first_seen_at") or now).isoformat(),
        }

    # ── Lease control plane ───────────────────────────────────────────────

    async def claim_lease(
        self, session_id: str, holder_id: str, ttl_s: int = 30
    ) -> dict[str, Any] | None:
        """Atomic CAS claim. Succeeds iff lease missing or expired.

        Returns the updated session doc, or ``None`` if another holder
        currently owns an unexpired lease. Caller is responsible for
        ensuring the session document exists (``upsert_session`` /
        ``persist_session_snapshot`` writes it on first save) — this CAS
        does not upsert because doing so would consume the ``$setOnInsert``
        slot needed by ``upsert_session`` for ``user_id`` / ``surface``.
        """
        if not self._ready():
            return None
        now = _now()
        try:
            return await self.db.sessions.find_one_and_update(
                {
                    "_id": session_id,
                    "$or": [
                        {"lease.expires_at": {"$lt": now}},
                        {"lease": {"$exists": False}},
                        {"lease.holder_id": None},
                    ],
                },
                {
                    "$set": {
                        "lease": {
                            "holder_id": holder_id,
                            "expires_at": now + timedelta(seconds=ttl_s),
                            "claimed_at": now,
                        },
                    },
                    "$inc": {"lease_generation": 1},
                },
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as e:
            logger.warning(f"claim_lease failed for {session_id} ({holder_id}): {e}")
            return None

    async def renew_lease(
        self, session_id: str, holder_id: str, ttl_s: int = 30
    ) -> dict[str, Any] | None:
        """Atomic renew. Returns updated doc, or ``None`` if we lost it.

        Raises ``PyMongoError`` on transient Mongo failures so callers can
        distinguish "we lost the lease" (return value ``None``) from "Mongo
        flapped" (exception). The heartbeat loop catches the exception and
        skips this tick for the affected session; only ``None`` triggers
        ``_on_lease_lost``.
        """
        if not self._ready():
            return None
        now = _now()
        return await self.db.sessions.find_one_and_update(
            {"_id": session_id, "lease.holder_id": holder_id},
            {"$set": {"lease.expires_at": now + timedelta(seconds=ttl_s)}},
            return_document=ReturnDocument.AFTER,
        )

    async def release_lease(self, session_id: str, holder_id: str) -> None:
        """Atomic release. No-op if we no longer hold the lease.

        Clears ``lease.holder_id`` in addition to expiring the lease so the
        renew CAS filter (``{"lease.holder_id": holder_id}``) no longer
        matches — preventing a heartbeat tick that snapshotted the session
        id pre-release from re-extending the lease 30 s into the future.
        """
        if not self._ready():
            return
        now = _now()
        try:
            await self.db.sessions.update_one(
                {"_id": session_id, "lease.holder_id": holder_id},
                {"$set": {"lease.expires_at": now, "lease.holder_id": None}},
            )
        except PyMongoError as e:
            logger.warning(f"release_lease failed for {session_id} ({holder_id}): {e}")

    # ── Pending submissions ───────────────────────────────────────────────

    async def enqueue_pending_submission(
        self, session_id: str, op_type: str, payload: dict[str, Any]
    ) -> str:
        """Insert a pending submission and return its inserted ``_id`` (str)."""
        if not self._ready():
            return ""
        doc = {
            "_id": ObjectId(),
            "session_id": session_id,
            "op_type": op_type,
            "payload": payload or {},
            "status": "pending",
            "claimed_by": None,
            "created_at": _now(),
        }
        try:
            await self.db.pending_submissions.insert_one(doc)
            return str(doc["_id"])
        except PyMongoError as e:
            logger.warning(
                f"enqueue_pending_submission failed for {session_id}: {e}"
            )
            return ""

    async def claim_pending_submission(
        self, session_id: str, holder_id: str
    ) -> dict[str, Any] | None:
        """Atomic FIFO claim of the oldest pending submission for a session."""
        if not self._ready():
            return None
        now = _now()
        try:
            return await self.db.pending_submissions.find_one_and_update(
                {"session_id": session_id, "status": "pending"},
                {
                    "$set": {
                        "status": "claimed",
                        "claimed_by": holder_id,
                        "claimed_at": now,
                    }
                },
                sort=[("created_at", 1)],
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as e:
            logger.warning(
                f"claim_pending_submission failed for {session_id} ({holder_id}): {e}"
            )
            return None

    async def mark_submission_done(self, submission_id: str | ObjectId) -> None:
        """Mark a previously claimed submission as completed."""
        if not self._ready():
            return None
        _id = submission_id if isinstance(submission_id, ObjectId) else ObjectId(submission_id)
        try:
            await self.db.pending_submissions.update_one(
                {"_id": _id},
                {"$set": {"status": "done", "completed_at": _now()}},
            )
        except PyMongoError as e:
            logger.warning(f"mark_submission_done failed for {submission_id}: {e}")

    async def requeue_claimed_for(
        self, holder_id: str, session_id: str | None = None
    ) -> int:
        """Flip ``claimed`` submissions for ``holder_id`` back to ``pending``.

        When ``session_id`` is provided, only submissions for that session
        are flipped — used by ``_on_lease_lost`` so losing one session's
        lease doesn't disturb the holder's other sessions. When ``None``
        (default), every claimed submission for this holder is flipped —
        the correct behaviour for ``release_session_to_background`` and the
        lifespan shutdown sweep.

        Must NOT modify ``created_at`` — FIFO ordering is preserved across
        handovers.
        """
        if not self._ready():
            return 0
        query: dict[str, Any] = {"status": "claimed", "claimed_by": holder_id}
        if session_id is not None:
            query["session_id"] = session_id
        try:
            result = await self.db.pending_submissions.update_many(
                query,
                {
                    "$set": {"status": "pending", "claimed_by": None},
                    "$unset": {"claimed_at": ""},
                },
            )
            return int(result.modified_count or 0)
        except PyMongoError as e:
            logger.warning(f"requeue_claimed_for failed for {holder_id}: {e}")
            return 0

    # ── Change-stream tails (replica-set required) ────────────────────────

    async def change_stream_pending_submissions(self, session_id: str):
        """Yield newly inserted pending submissions for ``session_id``.

        Raises ``PyMongoError`` (the standard pymongo behaviour) if the
        deployment isn't a replica set; callers fall back to polling.
        """
        if not self._ready():
            return
        pipeline = [
            {
                "$match": {
                    "operationType": "insert",
                    "fullDocument.session_id": session_id,
                    "fullDocument.status": "pending",
                }
            }
        ]
        async with await self.db.pending_submissions.watch(pipeline=pipeline) as stream:
            async for change in stream:
                full = change.get("fullDocument")
                if full is not None:
                    yield full

    async def change_stream_events(self, session_id: str, after_seq: int = 0):
        """Yield session_events documents with seq > ``after_seq``."""
        if not self._ready():
            return
        pipeline = [
            {
                "$match": {
                    "operationType": "insert",
                    "fullDocument.session_id": session_id,
                    "fullDocument.seq": {"$gt": int(after_seq or 0)},
                }
            }
        ]
        async with await self.db.session_events.watch(pipeline=pipeline) as stream:
            async for change in stream:
                full = change.get("fullDocument")
                if full is not None:
                    yield full

    # ── Polling fallback ──────────────────────────────────────────────────

    async def poll_pending_submissions_after(
        self, session_id: str, after_id: str | None
    ) -> list[dict[str, Any]]:
        """Return all pending submissions for ``session_id`` newer than ``after_id``.

        Used when change streams are unavailable. Sorted by ``created_at``.
        """
        if not self._ready():
            return []
        query: dict[str, Any] = {"session_id": session_id, "status": "pending"}
        if after_id:
            try:
                query["_id"] = {"$gt": ObjectId(after_id)}
            except Exception:  # noqa: BLE001 - bad id ⇒ start from beginning
                pass
        try:
            cursor = self.db.pending_submissions.find(query).sort("created_at", 1)
            return [row async for row in cursor]
        except PyMongoError as e:
            logger.warning(
                f"poll_pending_submissions_after failed for {session_id}: {e}"
            )
            return []


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
