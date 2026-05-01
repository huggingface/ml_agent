# RALPLAN: enable-ml-intern-to-run (v2 — post-Architect/Critic revision)

> Iteration 2 of `/plan --consensus`. Architect APPROVED WITH SUGGESTIONS; Critic returned APPROVE WITH REQUIRED REVISIONS. Revisions applied in place; see CHANGELOG at the end.

---

## RALPLAN-DR Summary

### Principles
1. **One code path on writes; durable substrate is canonical.** All inter-process *durability* and *coordination* goes through MongoDB. `submission_queue` (`backend/session_manager.py:88, 535, 723`) is deleted; `EventBroadcaster` (`backend/session_manager.py:44-78, 709-711`) is **kept as an opt-in read-side cache attached only on the holder process** (per Architect's "fifth option" synthesis — see Option A.5). Writes have a single path; reads have two transports for the same data.
2. **Mode is a lease holder, not a code path.** Main and Worker run the same Docker image; "background" is the side-effect of a lease changing hands, not a separate state machine. (Spec: Architecture > Foreground vs background mode.)
3. **Atomic CAS over coordination.** Lease ownership is decided by a single `findOneAndUpdate` against `sessions.lease.expires_at < now`. No locks, no zk, no leader election. (Spec: Lease + heartbeat ownership.)
4. **Reuse the durable substrate that already works.** `append_event` + monotonic `_next_seq` + `/api/events/{id}?after=<seq>` replay is "structurally sound" per the trace (`agent/core/session_persistence.py:316-353`, `backend/routes/agent.py:756-781`).
5. **No new infrastructure.** No Redis, no Nginx gateway, no message broker, no Cloudflare Worker — explicit spec constraint.

### Decision Drivers (top 3)
1. **Survives Main restart with active turn.** A deploy mid-turn does not interrupt the agent — ownership transfers to a Worker within ~30 s without the user seeing a banner.
2. **Horizontal scale is a deploy operation, not code work.** Adding a Worker = "spawn another Space with `MODE=worker`" — no routing changes, no sticky-session config.
3. **Frontend stays untouched.** The `/api/events/{id}?after=<seq>` reconnect path already works (`frontend/src/lib/sse-chat-transport.ts:426-453`).

### Viable Options

**Option A — Mongo-as-universal-control-plane, change-stream-only reads (variant of chosen).**
- *Pros*: Single substrate for events, submissions, leases. Replica-set + change streams give 50–200 ms p50 with no extra infra. Deletes `EventBroadcaster` outright — minimum surface area.
- *Cons*: 50–200 ms latency tax on the foreground path even when reader and writer are in the same process. Spec says this is acceptable; Architect flagged it as a regression worth quantifying or avoiding.

**Option A.5 — Mongo-as-universal-control-plane WITH holder-local fast-path overlay (CHOSEN — Architect synthesis).**
- *Pros*: All durability through Mongo (writes have one code path). On the holder process, an in-process `EventBroadcaster` acts as a read-side cache so SSE consumers attached to the holder skip the change-stream round-trip — preserves today's foreground latency profile. Non-holder readers (cross-process) fall back to the change stream automatically.
- *Cons*: Two read transports to reason about (in-process broadcaster vs change stream). The branch logic (`if session in self.sessions and session.holder_id == self._holder_id`) lives in one place and is testable. Net code size is roughly the same as Option A because the broadcaster code already exists.

**Option B — Redis pub/sub for control plane, Mongo for durability (REJECTED).**
- *Pros*: Sub-10 ms event fanout. Mature client libraries.
- *Cons*: New infrastructure (Redis cluster) — explicit spec constraint violation. The latency win is below the "50–200 ms acceptable" threshold, so it buys nothing the user values.

**Option C — Hybrid in-memory + Mongo (REJECTED).**
- *Pros*: Foreground keeps current latency.
- *Cons*: Two **write** code paths with different ordering/durability semantics. Spec Non-Goal explicitly forbids this. Note: Option A.5 differs — A.5 has one *write* path (Mongo) and two *read* transports (in-process cache + change stream), which the spec permits because the broadcaster is downstream of `append_event`.

**Option D — Always-Worker, Main is a relay only (REJECTED for v1).**
- *Pros*: Symmetric topology.
- *Cons*: Adds a network hop on the synchronous path. User explicitly said "you are overengineering this" in interview round 3.

### Mode
**SHORT consensus mode.** v1 of a closed spec; the deliberate-mode pre-mortem and expanded test plan are not required.

---

## 1. Requirements Summary

v1 of "background-running ml-intern": agent sessions survive SSE drops, browser close, and Main Space restarts. Per Spec > Goal:
1. Session keeps making progress when the user disconnects.
2. On return, frontend reconnects, replays missed events, surfaces pending approvals.
3. Architecture supports horizontal scale via more Worker Spaces.

Topology: 1 Main Space + 2 Worker Spaces, same Docker image, `MODE` env var differentiates. MongoDB Atlas (replica set) is the universal control plane. Frontend is unchanged.

## 2. Acceptance Criteria

### Drill 1 — "Close laptop, come back"

| Step | Action | Concrete check |
|------|--------|---------------|
| 1 | `POST /api/session` (logged in) | Response 200 + `session_id`. `db.sessions.findOne({_id: <id>})` returns a doc with `lease.holder_id` matching `main:*` and `lease.expires_at > now`. |
| 2 | `POST /api/chat/{session_id}` body `"fine-tune llama on dataset X"` | SSE stream begins emitting `assistant_start`, `tool_call_start` events. `db.session_events.find({session_id})` shows monotonically increasing `seq`. `db.pending_submissions.findOne({session_id, op_type: "user_input"})` exists with `status: "done"`. |
| 3 | Close browser tab. Wait 3 minutes 10 seconds. | After grace period: `db.sessions.findOne({_id})` shows `lease.holder_id` matching `worker:*`. A `migrating` session_event was emitted at handover. `db.session_events` continues to grow. No `error` events from the handover itself. |
| 4 | Reopen browser, `GET /api/sessions` then `GET /api/events/{session_id}?after=<lastSeq>`. | `/api/events/...` SSE replays every event with `seq > lastSeq` from Mongo (via change stream tail on Main), then continues live. `approval_required` (if pending) is in the replay. No duplicate or skipped seqs. |
| 5 | `POST /api/approve` for the pending tool call | Response 200. `db.pending_submissions.findOne({session_id, op_type: "approval"})` flips `pending → claimed → done` within ≤2 s (change-stream) or ≤500 ms × 1 (polling fallback). Worker resumes the turn. |

**Pass criterion**: All 5 steps succeed end-to-end, **including across a deliberate `docker restart` of Main between steps 2 and 3**.

### Drill 2 — Main restart with active turn

| Step | Action | Concrete check |
|------|--------|---------------|
| 1 | Start a session on Main, send a message that triggers a long-running tool call. | `db.sessions.findOne({_id})` shows `runtime_state: "processing"` and `lease.holder_id` starts with `main:`. |
| 2 | Force-restart Main mid-turn. | Main's `lifespan` shutdown sweep emits a `migrating` session_event for each in-flight session, then calls `release_lease`. `lease.expires_at <= now`. |
| 3 | Within 30 s of Main shutdown: a Worker claims the lease. | `lease.holder_id` flips to `worker:*` within ≤30 s. The agent loop resumes from the last persisted snapshot. |
| 4 | Fresh Main comes back; user opens new tab; hits `GET /api/events/{id}?after=<seq>`. | SSE streams ongoing events. No `interrupted` event from the restart itself. |

**Pass criterion**: Single uninterrupted user-visible turn across a deliberate Main restart.

### Out-of-scope (per spec)
- 10k concurrent load test (deferred; optional 100-session soak recommended pre-launch).
- Tool-call double-execution test.
- Push-when-away (NotificationGateway stays opt-in).

## 3. Implementation Steps

> Order is dependency-driven. Each step lists files, the change in one sentence, the test, and any dependency.

### Step 1 — Add `pending_submissions` collection + `sessions.lease` sub-doc to the persistence layer

- **Files**: `agent/core/session_persistence.py`
- **Change**: Add `pending_submissions` collection with index `(session_id, status, created_at)`; add `claim_lease`, `renew_lease`, `release_lease`, `claim_pending_submission`, `enqueue_pending_submission`, `mark_submission_done`, `requeue_claimed_for(holder_id)`, `change_stream_pending_submissions(session_id)`, `change_stream_events(session_id, after_seq)` methods on `MongoSessionStore` (mirror as no-ops on `NoopSessionStore` at `agent/core/session_persistence.py:54-103`).
- **Anchor points**:
  - Add new index in `_create_indexes` (`agent/core/session_persistence.py:139-158`).
  - Add new methods after `mark_pro_seen` — confirmed by Read: `mark_pro_seen` ends at **line 472**; insert new methods between line 472 and the module-level `_store: NoopSessionStore | MongoSessionStore | None = None` at line 475.
  - Use the same `_next_seq` pattern (`agent/core/session_persistence.py:316-323`) for `pending_submissions` IDs if needed.
- **Lease semantics** (Spec: Lease + heartbeat ownership):
  - `claim_lease(session_id, holder_id, ttl_s=30)` = atomic `findOneAndUpdate({_id: session_id, $or: [{lease.expires_at: {$lt: now}}, {lease: {$exists: false}}]}, {$set: {lease: {holder_id, expires_at: now + ttl}}})`. Returns the updated doc or `None`.
  - `renew_lease(session_id, holder_id, ttl_s=30)` = atomic `findOneAndUpdate({_id, "lease.holder_id": holder_id}, {$set: {"lease.expires_at": now + ttl}})`.
  - `release_lease(session_id, holder_id)` = atomic `findOneAndUpdate({_id, "lease.holder_id": holder_id}, {$set: {"lease.expires_at": now}})`.
- **Pending submission semantics**:
  - `enqueue_pending_submission(session_id, op_type, payload)` inserts `{_id, session_id, op_type, payload, status: "pending", claimed_by: null, created_at: now}`.
  - `claim_pending_submission(session_id, holder_id)` atomic `findOneAndUpdate({session_id, status: "pending"}, {$set: {status: "claimed", claimed_by: holder_id, claimed_at: now}}, sort=[("created_at", 1)])`.
  - `mark_submission_done(submission_id)` sets `status: "done", completed_at: now`.
  - **`requeue_claimed_for(holder_id)`** (new, per Critic MAJOR #3): atomic `update_many({status: "claimed", claimed_by: holder_id}, {$set: {status: "pending", claimed_by: null}})`. **Must NOT touch `created_at`** — preserves FIFO across handovers. Called from (a) lifespan shutdown sweep (Step 5) and (b) lease-renewal-failure path (Step 2).
- **Backfill on init** (per Critic CRITICAL #1, OQ #3 locked option (a)): `MongoSessionStore.init()` runs a one-shot migration:
  - `update_many({lease: {$exists: false}, status: "active", last_active_at: {$gt: now - 1h}}, {$set: {lease: {holder_id: null, expires_at: 0}}})` — recoverable in-flight sessions get an empty lease so the next CAS succeeds.
  - `update_many({lease: {$exists: false}, status: "active", last_active_at: {$lte: now - 1h}}, {$set: {runtime_state: "idle"}})` — older sessions flip to `idle` (NOT `ended`; they remain recoverable).
- **Holder ID format** (per Critic CRITICAL OQ #4 locked): `f"{mode}:{hostname}:{uuid7().hex[:8]}"` if uuid7 available; else `f"{mode}:{hostname}:{uuid4().hex[:8]}"`. Rationale: uuid7 sorts chronologically — debugging win when scanning leases over time. Fallback to uuid4 for environments without uuid7 (Python ≥3.13 has it; we pin uuid4 if pre-3.13). Pick one at process start; do not change mid-run.
- **Test**: New unit tests in `tests/test_session_persistence.py`: assert `claim_lease` is exactly-once under concurrent calls (`asyncio.gather` with N tasks, exactly 1 returns the doc); assert renew with wrong `holder_id` returns `None`; assert pending submission FIFO claim order; assert `requeue_claimed_for` preserves `created_at` and FIFO order on re-claim.
- **Depends on**: nothing.

### Step 1.5 — Re-queue claimed submissions on lease loss (NEW, per Critic MAJOR #3)

- **Files**: `backend/session_manager.py`
- **Change**: Wire `requeue_claimed_for(self._holder_id)` into TWO places:
  1. **Lifespan shutdown sweep** (Step 5): before/after `release_lease`, call `requeue_claimed_for` so any submissions this Main was mid-processing flip back to `pending` and a Worker can pick them up.
  2. **Lease-renewal-failure path** (in the heartbeat task added in Step 2): if `renew_lease` returns `None` (someone else now holds the lease, e.g. clock skew or a bug), the local process must (a) stop processing, (b) call `requeue_claimed_for(self._holder_id)`, (c) drop the session from `self.sessions`. Log a `WARN` with the session_id and holder_id transition.
- **Ordering contract**: `claimed → pending` must NOT modify `created_at`. FIFO is preserved across handovers — this is asserted by the unit test in Step 1.
- **Test**: Unit: claim 3 submissions, kill the holder, verify all 3 flip back to `pending` with original `created_at`; another holder picks them up in the original order.
- **Depends on**: Step 1.

### Step 2 — Refactor `SessionManager` to be lease-holder aware

- **Files**: `backend/session_manager.py`
- **Change**: Add `holder_id`, `mode` (`"main"|"worker"`) to `SessionManager.__init__`. Compute `self._holder_id` from `MODE` + hostname + uuid suffix per Step 1 format. Expose `claim_lease/renew_lease/release_lease` wrappers.
- **Anchor points**:
  - `SessionManager.__init__` (`backend/session_manager.py:124-129`) reads `MODE` env, hostname, and computes `self._holder_id`.
  - `SessionManager.start` (`backend/session_manager.py:131-135`) starts a background lease-renewal task (10 s cadence) that renews leases for every entry in `self.sessions`. **On renewal failure** → trigger Step 1.5 path (requeue + drop).
- **Lease TTL is the PRIMARY mechanism for stale-lease recovery** (per Critic MAJOR #4, see Risk #2 below). The lifespan-hook `release_lease` is a best-effort optimization; the SIGKILL path relies on TTL=30 s expiring naturally.
- **Test**: Boot Main locally with `MODE=main`, watch lease renewals every 10 s in Mongo; SIGKILL the process and watch `lease.expires_at` go stale within 30 s; another instance reclaims via CAS.
- **Depends on**: Step 1.

### Step 3 — Replace `submission_queue` with Mongo-backed reader

- **Files**: `backend/session_manager.py`, `backend/routes/agent.py`, `agent/core/session_persistence.py`
- **Change**:
  - Delete `submission_queue: asyncio.Queue` from `AgentSession` (`backend/session_manager.py:88`) and `_create_session_sync` (`backend/session_manager.py:535`).
  - Rewrite `_run_session` (`backend/session_manager.py:693-772`) so the loop body waits on `pending_submissions` change stream (or 500 ms poll fallback) filtered by `session_id` instead of `submission_queue.get()` (`backend/session_manager.py:723-725`). Each consumed doc → `Submission(operation=Operation(...))` → `process_submission`.
  - On consume, atomically flip `pending → claimed`; on completion (`finally` block), flip `claimed → done`. Worker crash between `claimed` and `done` → re-claim by sweep (per Step 1.5 / per spec, Workers are stable; idempotency is v2).
  - Rewrite `submit`, `submit_user_input`, `submit_approval`, `interrupt`, `undo`, `compact`, `shutdown_session` (`backend/session_manager.py:774-845`) so they call `enqueue_pending_submission(session_id, op_type, payload)` instead of `agent_session.submission_queue.put(...)`.
  - **`interrupt` non-holder fallback** (per Critic MAJOR #10): in `SessionManager.interrupt(session_id)`, branch on holder:
    - If `session_id in self.sessions` AND that session's lease is held by `self._holder_id` → call `session.cancel()` directly (today's fast path, preserves the in-process bypass at `backend/session_manager.py:801-807`).
    - Else → `enqueue_pending_submission(session_id, op_type="interrupt", payload={})`. The actual holder will consume the doc via its change stream and call `session.cancel()`.
  - **`is_in_tool_call` flag on `Session`** (per Critic MAJOR #5, supporting Step 6.3): add `is_in_tool_call: bool = False` field on `Session` (or `AgentSession` if `Session` is owned by the agent module — verify). Set `True` in the tool dispatch wrapper before calling the tool and `False` in its `finally`. Need to verify exact dispatch site with Read of `agent/core/session.py` during Step 3 implementation; the spec assumption is "wherever tools are dispatched."
- **Test**:
  - Unit: enqueue 3 submissions across 2 processes claiming the same session — exactly one processes each submission, in FIFO order.
  - Integration: 1 Main + 1 Worker locally with shared Mongo; `POST /api/submit` while Main holds the lease — Main consumes; release the lease, Worker takes it, next `POST /api/submit` is consumed by Worker.
  - Interrupt: `POST /api/interrupt` against a session whose lease lives on Worker → submission row appears with `op_type: "interrupt"`; Worker consumes it and cancels.
- **Depends on**: Steps 1, 2.

### Step 4 — Holder-local fast-path overlay: keep `EventBroadcaster` as opt-in read-side cache (REVISED per Critic MAJOR #2)

- **Files**: `backend/session_manager.py`, `backend/routes/agent.py`
- **Change**: **Keep** the `EventBroadcaster` class (`backend/session_manager.py:44-78`) and `agent_session.broadcaster` field (`backend/session_manager.py:96`). The class becomes an *opt-in read-side cache* attached only when this process is the lease holder.
  - **Writes are unchanged** on the durability side — `Session.send_event` (`agent/core/session.py:146-164`) still calls `append_event` first (the trace confirms this). After durable write, the holder process additionally pushes to its in-process broadcaster. This is the "one code path on writes" principle preserved on the durability side; the in-process push is a downstream side-effect, not a separate substrate.
  - **Read branch logic in the SSE generator** (replaces the `_sse_response` body at `backend/routes/agent.py:707-753`):
    ```python
    # pseudocode
    after_seq = _last_event_seq(request)
    replay_events = await store.load_events_after(session_id, after_seq)
    yield from format_replay(replay_events)

    sess = session_manager.sessions.get(session_id)
    if sess is not None and sess.holder_id == session_manager._holder_id:
        # Fast path: subscribe to in-process broadcaster
        sub_id, queue = sess.broadcaster.subscribe()
        try:
            async for msg in drain_with_keepalive(queue):
                yield format_sse(msg)
                if terminal: return
        finally:
            sess.broadcaster.unsubscribe(sub_id)
    else:
        # Slow path: change stream (or 500 ms poll fallback)
        async for doc in change_stream_events(session_id, after_seq=replay_max_seq):
            yield format_sse(_event_doc_to_msg(doc))
            if terminal: return
    ```
  - **Call-site enumeration** (per Critic MAJOR #6) — three sites use `broadcaster` today:
    | Site | Before | After |
    |------|--------|-------|
    | `_run_session` line ~709-711 (creates broadcaster) | `agent_session.broadcaster = EventBroadcaster()` always | Same line, but only effective on holder; `Session.send_event` still calls into it via `agent_session.broadcaster.publish(msg)` if attached. Workers also attach it for symmetry — the cache simply sees no readers when no SSE is open against that Worker. |
    | `chat_sse` route at `backend/routes/agent.py:604` (`broadcaster = agent_session.broadcaster`; `sub_id, event_queue = broadcaster.subscribe()`) | Subscribes unconditionally | Branch on `agent_session.holder_id == self._holder_id`. If true: subscribe (in-process transport). If false: open change stream filtered by `session_id` and `seq > replay_max_seq` (cross-process transport). |
    | `subscribe_events` route at `backend/routes/agent.py:773-774` (same broadcaster.subscribe) | Subscribes unconditionally | Same branching as `chat_sse`. |
  - **Polling fallback**: if change-stream open fails (no replica set), fall back to a 500 ms poll loop calling `load_events_after(session_id, last_seen_seq)`.
  - The `event_queue: asyncio.Queue` on `Session` (constructed at `backend/session_manager.py:178`, used at `agent/core/session.py:146-164`) is kept (it's the per-session in-memory queue separate from broadcaster fanout). No changes to `agent/core/session.py` send/recv paths.
- **Test**:
  - Unit: open two SSE streams against the same `session_id` — one against the holder, one against a non-holder Main — verify both see identical event sequences (broadcaster fanout matches change-stream tail for the same Mongo events).
  - Integration: Drill 1 step 4 — close + reopen browser, verify replay + live tail.
  - Latency: instrument both transports; confirm holder-local p50 stays at today's foreground latency, non-holder p50 sits in 50–200 ms band.
- **Depends on**: Steps 1, 2, 3.

### Step 5 — Main lifespan shutdown sweep + grace-period auto-release

- **Files**: `backend/main.py`, `backend/session_manager.py`, `backend/routes/agent.py`
- **Change**:
  - **In `lifespan` shutdown branch**, after the existing trace flush loop and **before `await session_manager.close()`**, insert the lease sweep. Confirmed by Read: insertion point is **between line 62 (close of `try/except` for the final-flush block) and line 63 (`await session_manager.close()`)**. For each session in `self.sessions` whose `runtime_state == "processing"` (Spec migration trigger 3):
    1. Emit a `migrating` session_event via `append_event` (per Critic MAJOR #8) — frontend handles unknown event types gracefully and can render a "reconnecting…" label.
    2. Call `requeue_claimed_for(self._holder_id)` (per Step 1.5).
    3. Call `release_lease(session_id, self._holder_id)` atomically.
    4. Log the count.
  - Add a new background task in `SessionManager.start` (`backend/session_manager.py:131-135`): **grace-period sweeper**. Every 30 s, for each `agent_session` in `self.sessions` owned by Main: check the session's subscriber-attached counter; if zero AND the no-subscriber window exceeds `BACKGROUND_GRACE_S` (env, default 180), emit `migrating` session_event then call `release_lease`.
  - **Subscriber-count storage** (per Critic MAJOR #7): add `_subscriber_counts: dict[str, int]` on `SessionManager` (keyed by `session_id`) and `_no_subscriber_since: dict[str, float]` (timestamp when count dropped to 0). Add two `SessionManager` methods:
    - `_attach_subscriber(session_id)`: increments `_subscriber_counts[session_id]`; clears `_no_subscriber_since[session_id]`.
    - `_detach_subscriber(session_id)`: decrements; if now 0, sets `_no_subscriber_since[session_id] = time.time()`.
    - `_sse_response` (in `backend/routes/agent.py`, both `chat_sse` and `subscribe_events` paths) calls `_attach_subscriber(session_id)` before the generator's first yield and `_detach_subscriber(session_id)` in its `finally`. This applies to BOTH transports (broadcaster and change-stream paths).
  - Add new POST `/api/session/{id}/background` route in `backend/routes/agent.py` that emits `migrating` session_event + calls `release_lease` immediately (Spec migration trigger 2 — manual button). Frontend wiring is OQ "during execution."
- **Test**:
  - Drill 2: `docker restart` mid-turn, verify `migrating` event appears in `db.session_events`, `pending_submissions` flips back to `pending` for this holder, `lease.expires_at` is zeroed before exit, Worker picks up within 30 s.
  - Drill 1 step 3: simulate SSE drop, wait 3 min, verify lease releases.
  - Manual button: `curl -X POST /api/session/<id>/background` — `migrating` event + lease releases immediately.
- **Depends on**: Steps 2, 3, 4.

### Step 6 — Worker entrypoint + `MODE` flag in `start.sh`

- **Files**: `backend/start.sh`, `backend/main.py` (extend, not replace)
- **Change**:
  - Modify `backend/start.sh`. Confirmed by `wc -l`: file is **15 lines**. Preserve the existing port-conflict graceful-exit hack for Main mode (per Critic MAJOR #9):
    ```bash
    #!/bin/bash
    # Entrypoint for HF Spaces dev mode compatibility.
    # ... (header comments preserved) ...

    if [ "$MODE" = "worker" ]; then
        exec python -m backend.worker
    fi

    # Main path: existing port-conflict graceful exit preserved.
    uvicorn main:app --host 0.0.0.0 --port 7860
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "uvicorn exited with code $EXIT_CODE, exiting gracefully."
        exit 0
    fi
    ```
  - **Worker entrypoint shape**: keep `python -m backend.worker` as the deploy-side command. Inside `backend/worker.py`, the body is a one-liner: `import asyncio; from backend.main import worker_loop; asyncio.run(worker_loop())`. This keeps logic centralized in `main.py` (no drift) while exposing a clean module entrypoint for the deploy command. (Listed as "during execution" OQ; this default is locked unless it breaks something.)
  - **The worker loop** (`worker_loop` in `backend/main.py`, new function added after the existing app definition near line 114):
    1. `await session_manager.start()` (initializes Mongo, starts heartbeat task with `mode="worker"`).
    2. Forever: open a change stream on `pending_submissions` filtered by `status: "pending"`. For each new doc:
       - Try `claim_lease(doc.session_id, self._holder_id)`.
       - If lease claimed: call `ensure_session_loaded(doc.session_id, owner_id)` (`backend/session_manager.py:381-486`) — already exists, already rehydrates from Mongo. The agent loop runs via `_run_session` (now Mongo-backed per Step 3).
       - If lease not claimed: skip; another process will get it.
    3. **Idle eviction**, every 60 s, per session in `self.sessions`. The eviction predicate is:
       ```python
       idle = (
           not session.is_in_tool_call          # MAJOR #5
           and not agent_session.is_processing
           and not pending_count_for(session_id)
           and (now - agent_session.last_submission_at) > WORKER_IDLE_EVICT_S
       )
       ```
       Where `last_submission_at: float` (per Critic OPTIONAL #14) is set on `AgentSession` whenever `enqueue_pending_submission` returns or whenever a submission is consumed — avoids `count_documents` polling on the hot path. `pending_count_for` is a cheap `count_documents({session_id, status: "pending"})` only run during the eviction sweep (60 s cadence, low overhead). Default `WORKER_IDLE_EVICT_S=1800` (30 min, per OQ #5 locked).
    4. SIGTERM handler: `await session_manager.close()` cleanly. Workers ALSO release leases they hold on graceful shutdown (so Main can reclaim). SIGKILL path falls back to TTL=30 s.
- **Test**:
  - `MODE=worker docker run ...` boots without binding port 7860; logs show heartbeat and "watching pending_submissions".
  - With Main + Worker against shared Mongo, write a `pending_submission` for an existing session — Worker claims and processes.
  - Idle eviction: spin up a session on Worker, leave it idle 31 min, verify lease releases.
- **Depends on**: Steps 2, 3.

### Step 7 — Observability + deployment artifacts

- **Files**: `backend/main.py`, `backend/routes/agent.py`, `Dockerfile` (no structural change), `README.md`/HF Space config.
- **Change**:
  - Add metrics endpoint or log lines for:
    - Lease state — count by holder, count expiring within 5 s.
    - `pending_submissions` lag — `now - created_at` p50/p95.
    - Change-stream connectivity — boolean health flag on Main.
    - **Replay event count** (per Critic OPTIONAL #17) — log `replay_event_count` when SSE generator emits replay batch; surfaces sessions crossing the long-replay pain threshold.
    - Per-session subscriber-count gauge (debugging aid for grace-period sweeper).
  - Document deployment requirement: `MONGODB_URI` must point at a replica set (Atlas or self-hosted). Add a README section. Document new env vars: `MODE`, `BACKGROUND_GRACE_S` (default 180), `WORKER_IDLE_EVICT_S` (default 1800), `LEASE_TTL_S` (default 30), `LEASE_RENEW_S` (default 10).
  - HF Spaces deployment: create 2 new Worker Spaces from the same repo with `MODE=worker` set in env. Same `MONGODB_URI`. Same Docker image.
  - **Migration ordering for the deploy that ships this change** (per Critic CRITICAL #1, see Risk #5):
    1. Workers deployed first with new code (they sit idle waiting for `pending_submissions`).
    2. Main rolls last (the `MongoSessionStore.init()` backfill from Step 1 runs on Main's startup).
    3. Backfill writes empty `lease={holder_id: null, expires_at: 0}` for active sessions with `last_active_at > now - 1h`.
    4. **User-visible blast radius**: TBD — collect baseline pre-launch by counting `db.sessions.countDocuments({status: "active", last_active_at: {$gt: now - 1h}})` over a 24 h sample. The number is the count of users who could see a brief reconnect during the deploy window. Today the existing `_run_session:731` snapshot-in-`finally` does NOT cover SIGKILL, so submissions in flight at deploy are *already* lost — this plan does not regress that, but it should be documented in the deploy runbook.
- **Test**: Deploy Main + 2 Workers to HF; run Drill 1 in production. Confirm Mongo metrics endpoint returns sane values.
- **Depends on**: Steps 1–6.

## 4. Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **Mongo deployment is not a replica set** in some environment. Change streams unavailable, polling-only path adds 500 ms per submission. | Medium | Medium | Polling fallback built into Step 1 (`change_stream_*` methods detect support and fall back). Document the requirement in Step 7. Existing prod Atlas confirmed healthy. |
| 2 | **Lease starvation / stale-lease window during deploys.** Main is SIGKILLed; lifespan hooks don't run; Workers don't see freed lease. | Low | Medium | **PRIMARY: TTL=30 s upper-bounds the window** — SIGKILL is the assumed failure mode (per Critic MAJOR #4). Workers reclaim via CAS once `expires_at < now`. **SECONDARY (best-effort optimization): lifespan-hook `release_lease` + `requeue_claimed_for` (Step 5)** drops the lease atomically before exit when the shutdown is graceful (SIGTERM, not SIGKILL). The lifespan hook is icing; the TTL is the cake. |
| 3 | **Change-stream connection drops** silently. Main stops streaming SSE events even though events are landing in Mongo. | Medium | High | Wrap the change stream in a reconnect loop with exponential backoff (1s, 2s, 4s, max 30s); on reconnect, replay from `last_seen_seq` via `load_events_after` so no events are skipped. Add the connectivity health flag (Step 7) so we alert on prolonged disconnects. **Chaos test** (per Critic OPTIONAL #15): kill the Mongo proxy briefly mid-stream and verify resume via resume token. |
| 4 | **`_safe_message_doc` 15 MB cap** interaction. Marker replacement breaks tool-call ordering on rehydrate. | Low | Medium | Spec accepts as v1 tech debt (Spec: Assumptions Exposed #8). For v1, log a `WARN` and emit a session_event so the user sees "your last tool output was truncated." Track for v1.1. |
| 5 | **Backward compatibility for in-flight sessions during the deploy that ships this change.** Sessions on the old code path have no `lease` and no `pending_submissions`. | High | High | **Locked option (a) per OQ #3**: `MongoSessionStore.init()` backfills `lease: {holder_id: null, expires_at: 0}` for active sessions with `last_active_at > now - 1h` (auto-resume). Older active sessions flip `runtime_state` to `idle` (recoverable, NOT `ended`). **Migration ordering**: (a) Workers deployed first with new code; (b) Main rolls last — `init()` runs the backfill on Main's startup; (c) backfill happens during Main boot before `lifespan` finishes; (d) **user-visible blast radius**: TBD — collect pre-launch baseline by counting active sessions with `last_active_at > now - 1h` over a 24 h sample. Note: today's `_run_session:731` `finally`-snapshot path does not cover SIGKILL, so this plan does not regress losses on hard restart. |
| 6 | **Unbounded `load_events_after` replay scan** for long sessions. | Medium | Medium | **Spec calls this accepted v1 tech debt** (Spec: Assumptions Exposed #8). Log a `replay_event_count` line (Step 7) so we see when sessions cross the pain threshold. Mitigations for v1.1: paginate the cursor or roll up token-stream events. |
| 7 | **Worker boot race**: two Workers boot simultaneously and both try to claim the same session's lease. | Low | Low | Atomic CAS in Step 1 — exactly one Worker wins. Tested by Step 1 unit test. |
| 8 | **Idempotency hole** if a Worker crashes after `claimed` but before `done` on a `pending_submission`. | Low | High | **Step 1.5's `requeue_claimed_for` will re-pending the doc** on lease loss/handover; that's the correct path during graceful Main rollover. For SIGKILL'd Workers, the next sweep re-pendings stuck `claimed` rows; a v2 idempotency layer (op_id deduping) addresses true double-execution. Note in code with `TODO: v2`. |

## 5. Verification Steps

### Local development verification

```bash
# 1. Start Mongo replica set locally (or use Atlas test cluster)
docker run -d --name mongo-rs -p 27017:27017 mongo:7 mongod --replSet rs0 --bind_ip_all
docker exec -it mongo-rs mongosh --eval 'rs.initiate()'
# Wait ~5s for the replica set to elect a primary
docker exec -it mongo-rs mongosh --eval 'rs.status()' | grep -E '"name"|"stateStr"'

# 2. Start Main locally (terminal A)
export MODE=main
export MONGODB_URI="mongodb://localhost:27017/?replicaSet=rs0"
export MONGODB_DB=ml-intern-dev
uv run uvicorn backend.main:app --host 0.0.0.0 --port 7860

# 3. Start Worker locally (terminal B)
export MODE=worker
export MONGODB_URI="mongodb://localhost:27017/?replicaSet=rs0"
export MONGODB_DB=ml-intern-dev
uv run python -m backend.worker

# 4. Run Drill 1 against local stack
./scripts/drill1.sh   # to be written; see below

# 5. Run Drill 2: kill Main mid-turn
pkill -SIGTERM uvicorn   # graceful, triggers lifespan shutdown
# Watch Mongo: lease should flip to worker:* within 30s
mongosh --eval 'db.sessions.findOne({_id: "<session_id>"}).lease'

# 6. Chaos test (per OPTIONAL #15)
docker pause mongo-rs && sleep 5 && docker unpause mongo-rs
# Verify the change stream resumes via resume token; no events skipped
```

### Unit test commands

```bash
# Step 1 — persistence layer
uv run pytest tests/test_session_persistence.py -k "lease or pending_submissions or requeue" -v

# Step 1.5 — re-queue on lease loss
uv run pytest tests/test_session_manager.py -k "requeue_on_renewal_failure" -v

# Step 3 — submission flow
uv run pytest tests/test_session_manager.py -k "submission_flow or interrupt_non_holder" -v

# Step 4 — change-stream SSE + holder-local fast path
uv run pytest tests/test_routes_agent.py -k "events_replay or chat_sse or holder_fast_path" -v
```

### Production verification (HF Spaces)

```bash
# Deploy order matters (per CRITICAL #1):
# 1. Deploy Workers FIRST
hf space upload <worker-1>  # MODE=worker set in Space settings
hf space upload <worker-2>

# 2. Deploy Main LAST (init() backfill runs on its startup)
hf space upload <main-space>

# 3. Verify backfill ran
mongosh --eval 'db.sessions.countDocuments({lease: {$exists: true}, status: "active"})'

# 4. Manual Drill 1 against production URL.
```

### Rollback procedure

If Drill 1 step 3 fails in prod (lease doesn't transfer), redeploy the previous commit to Main. Workers can stay (they'll see no `pending_submissions` on the old code path). Sessions in flight may be lost (acknowledged in Risk #5).

## 6. Open Questions / TBD — during execution

The 5 below stay open as "during execution." OQs #3 (backward-compat option), #4 (holder_id format), and #5 (idle eviction default) were resolved in this revision and removed from the open-questions file.

1. **Worker entrypoint shape** — locked default: `backend/worker.py` is a 3-line shim that calls `worker_loop()` from `backend/main.py`. Reversible.
2. **Manual "background" button frontend wiring** — backend route ships in Step 5; frontend button is post-v1.
3. **Lease TTL/renew defaults** — TTL=30s, renew=10s per spec. Long tool calls hold the lease for hours; renewal must keep pace. Confirm under load test.
4. **Polling fallback cadence = 500 ms** — confirm acceptable when Mongo is single-node (dev/local).
5. **`_safe_message_doc` 15 MB truncation visibility** — text of the user-visible session_event. Confirm with copywriter / design pass.

## 7. Files Touched (summary)

| File | Lines today | Change type |
|------|-------------|-------------|
| `agent/core/session_persistence.py` | 1-489 | **Add** — pending_submissions ops, lease CAS, `requeue_claimed_for`, change-stream wrappers, init() backfill. New methods inserted between line 472 and line 475. No removals. |
| `backend/session_manager.py` | 1-1014 | **Modify** — `EventBroadcaster` (44-78) **kept** as opt-in cache; `submission_queue` field (88) **removed**; `_run_session` body (693-772) rewritten; `submit*` methods (774-845) rewritten; `interrupt` (801-807) branches on holder. **Add** — `holder_id`, lease renew task with renewal-failure path (Step 1.5), grace-period sweeper, subscriber counts, `is_in_tool_call`, `last_submission_at`. |
| `backend/main.py` | 29-63, 100-114 | **Add** — lifespan shutdown lease sweep (insertion **between line 62 and line 63**, before `await session_manager.close()`); add `worker_loop()` function after line 114 / app definition. |
| `backend/start.sh` | 1-15 | **Modify** — branch on `$MODE` for worker; preserve existing port-conflict graceful-exit for Main. |
| `backend/routes/agent.py` | 588-650, 707-753, 756-781 | **Modify** — `/api/chat`, `/api/events`, `_sse_response` to use holder-fast-path overlay (broadcaster vs change-stream branch); attach/detach subscriber counts; `interrupt` non-holder fallback. **Add** — `/api/session/{id}/background` route. |
| `backend/worker.py` | new | **Add** — 3-line shim: `import asyncio; from backend.main import worker_loop; asyncio.run(worker_loop())`. |
| `agent/core/session.py` | 146-164 (send_event) | **Modify** — add `is_in_tool_call: bool` flag on `Session`; `send_event → append_event` durability path is unchanged. |
| `agent/core/agent_loop.py` | ~1111, ~1387 (tool dispatch via `tool_router.call_tool`) | **Modify** — set `session.is_in_tool_call = True` before each dispatch, clear in `finally`. Also covers the third dispatch site at `agent/tools/research_tool.py:~460`. |
| `Dockerfile` | 1-50 | **No change** — same image, mode flag at runtime. |
| `frontend/**` | — | **No change** — reconnect path already works (`frontend/src/lib/sse-chat-transport.ts:426-453`). |

## 8. ADR — final consensus

- **Decision**: Mongo-as-universal-control-plane with Main+Workers and lease+heartbeat ownership, **with holder-local fast-path overlay** for in-process SSE consumers (Architect's "fifth option" synthesis). All durability writes go through Mongo (one code path on writes); reads have two transports — in-process broadcaster on the lease holder, change-stream tail elsewhere.
- **Drivers**: (1) Survives Main restart with active turn (~30 s handover), (2) horizontal scale is operational, (3) frontend untouched.
- **Alternatives considered**:
  - **Option A** (Mongo-only, no broadcaster) — rejected in favor of A.5 because deleting the broadcaster outright imposes a 50–200 ms latency tax on the foreground path that A.5 avoids without touching the write path.
  - **Option A.5** (this design) — chosen.
  - **Option B** (Redis pub/sub) — rejected: violates "no new infra" spec constraint; latency win is below the 50–200 ms acceptable threshold.
  - **Option C** (hybrid in-memory + Mongo) — rejected: spec Non-Goal forbids two write paths. (Option A.5 is distinct: one write path, two read transports — spec permits this.)
  - **Option D** (always-Worker, Main is relay) — rejected for v1: adds a network hop on synchronous foreground; user explicitly closed this in interview round 3.
- **Why chosen**:
  - Spec §Architecture > Foreground vs background mode demands a single image with mode-as-lease-holder; A.5 satisfies this exactly.
  - Spec §Lease + heartbeat ownership demands atomic CAS over coordination; A.5 uses `findOneAndUpdate` with no new substrate.
  - Spec §Non-Goals forbids two **write** paths; A.5 has one. The read-side overlay is a downstream side-effect of `Session.send_event → append_event`, not a parallel write.
  - Architect synthesis: keeping `EventBroadcaster` as an opt-in read-side cache attached only on the holder process preserves today's foreground latency while gaining the durability and multi-process semantics the spec requires.
  - Reuses the durable substrate the trace validated.
- **Consequences**:
  - *Positive*: Durable session resume across browser-close, SSE drop, and Main restart. Horizontally scalable via more Worker Spaces. No new infra (no Redis, no broker, no gateway). One code path on writes. Today's foreground latency preserved on the holder fast path.
  - *Negative*: MongoDB **replica set** becomes a hard deployment requirement (change streams). 50–200 ms latency tax for non-holder cross-process reads (acceptable per spec). Deploy migration is non-trivial — Workers first, Main last, with backfill on Main's `init()`. Two read transports to reason about (mitigated: branch logic lives in one place).
- **Follow-ups (v1.1 / v2)**:
  - Idempotency layer for Worker crash mid-`claimed` (op_id dedup; today re-claim re-runs the op).
  - Push-when-away (NotificationGateway active wiring).
  - Async-dashboard for long-running background sessions.
  - `_safe_message_doc` 15 MB cap fix (chunked tool output storage).
  - Unbounded `load_events_after` replay scan — paginate cursor or roll up token-stream events.
  - UI button for manual backgrounding (frontend POST to `/api/session/{id}/background`).
  - Frontend rendering of the `migrating` session_event ("reconnecting…" banner).

---

## CHANGELOG (revisions applied in this iteration)

**CRITICAL (all applied)**
- **#1 Risk #5 migration ordering** — rewrote Risk #5 with explicit (a/b/c/d) ordering (Workers first, Main last, init() backfill, blast-radius TBD). Backfill targets `last_active_at > now - 1h` for empty-lease writeback; older sessions flip to `idle` (NOT `ended`). Added migration ordering to Step 7 verification.

**MAJOR (all applied)**
- **#2 Holder-local fast-path overlay** — added Option A.5 to RALPLAN-DR alternatives; rewrote Step 4 from "delete EventBroadcaster" to "keep as opt-in read-side cache attached only on the holder process." Added pseudocode for the SSE branch logic.
- **#3 Step 1.5 re-queue on lease loss** — added new step. Added `requeue_claimed_for(holder_id)` to Step 1 persistence layer. Wired into both lifespan shutdown sweep and lease-renewal-failure path. FIFO ordering preserved (does not modify `created_at`).
- **#4 Risk #2 re-ordering** — TTL=30 s is now PRIMARY (handles SIGKILL); lifespan-hook `release_lease` demoted to "best-effort optimization."
- **#5 Define "idle"** — eviction predicate in Step 6.3 now `not is_in_tool_call AND not is_processing AND no pending_submissions AND idle_window > threshold`. Added `is_in_tool_call: bool` flag to be set by tool dispatch wrapper (Step 3, exact site verified during execution).
- **#6 Step 4 call-site enumeration** — added a 3-row table covering `_run_session` line ~709-711 (broadcaster creation), `chat_sse` route at `agent.py:604`, and `subscribe_events` route at `agent.py:773-774`. Each row has before/after/transport.
- **#7 Subscriber-count storage** — locked: `_subscriber_counts: dict[str, int]` and `_no_subscriber_since: dict[str, float]` on `SessionManager`. Added `_attach_subscriber` / `_detach_subscriber` methods called from both SSE transport paths.
- **#8 `migrating` session_event** — emitted before `release_lease` in lifespan shutdown sweep, in grace-period sweeper, and in the `/api/session/{id}/background` route.
- **#9 `start.sh` port-exit-0 hack preserved** — Step 6 now shows the full `start.sh` rewrite with the `worker` branch first and the existing Main port-conflict graceful exit preserved verbatim.
- **#10 `interrupt` non-holder fallback** — Step 3 specifies the holder branch: holder calls `session.cancel()` directly; non-holder writes `op_type: "interrupt"` to `pending_submissions`.

**MINOR (all applied)**
- **#11 Off-by-one citations** — `start.sh` is now cited as 15 lines (verified by `wc -l`); lifespan shutdown insertion is between line 62 and line 63 (verified). Tool-dispatch sites for `is_in_tool_call` corrected to `agent/core/agent_loop.py:~1111` and `:~1387` (plus `agent/tools/research_tool.py:~460`), not `agent/core/session.py`.
- **#12 `mark_pro_seen` line range** — verified by Read: ends at **line 472**. New persistence methods inserted between line 472 and line 475 (`_store` global declaration).
- **#13 Local-dev verification** — §5 expanded with `mongo:7 --replSet rs0` setup, `rs.initiate()`, `rs.status()` verification, two-process startup (`MODE=main` and `MODE=worker` in separate terminals).

**OPTIONAL (all applied)**
- **#14 `last_submission_at` timestamp** — added on `AgentSession`; replaces `count_documents` polling on the eviction hot path (Step 6.3).
- **#15 Chaos-test note** — added to §5 Verification (`docker pause` Mongo mid-stream, verify resume token).
- **#16 uuid7 vs uuid4** — locked uuid7 with uuid4 fallback for pre-3.13 Python; rationale documented in Step 1 (chronological sort = debugging win).
- **#17 `replay_event_count` log line** — added to Step 7 observability list.

**ADR section** — fully filled in (Decision, Drivers, Alternatives considered, Why chosen, Consequences, Follow-ups).

**Open questions** — OQs #3, #4, #5 resolved and removed from open-questions.md. Remaining 5 stay annotated as "during execution."
