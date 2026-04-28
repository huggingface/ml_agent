# Phase 3 Plan: Space Background Workers

## Summary

Phase 3 decouples agent execution from the browser's SSE connection. The
frontend/backend Space will accept user submissions and persist them to MongoDB.
A long-running worker, preferably a separate Hugging Face Space, will claim those
submissions, run the agent loop, and write durable events/messages/snapshots back
to MongoDB.

This lets a user close their laptop or lose the SSE connection while the agent
continues running in the background. When the user returns, the frontend
rehydrates the session from MongoDB and replays events after the last seen event
sequence.

## Prerequisite

Mongo-backed session persistence must be working in production.

The backend must log:

```text
Mongo session persistence enabled (db=...)
```

If Mongo persistence is disabled, Phase 3 must not be enabled because queued
runs, event replay, session restore, and quota state all depend on durable
storage.

## Current State

The current Space implementation persists the right durable primitives:

- `sessions`: session metadata, runtime state, title, pending approvals, quota
  marker, soft-delete visibility.
- `session_messages`: latest restorable runtime context, stored per message.
- `session_events`: append-only event log with per-session sequence numbers.
- `session_trace_messages`: raw trace/SFT-ready message stream.
- `claude_quotas`: Mongo-backed quota counters.

However, active work is still driven by the API process:

1. The browser sends `POST /api/chat/{session_id}`.
2. The backend queues the operation in an in-memory `asyncio.Queue`.
3. The in-memory agent task runs `process_submission()`.
4. SSE streams events while the browser connection is alive.

That is good enough for restart recovery between turns, but it is not enough for
true background execution. If the API process restarts mid-turn, the in-flight
turn is lost.

## Target Architecture

```text
Browser
  POST /api/chat/{session_id}
  GET  /api/events/{session_id}?after=<last_seq>

Frontend/backend Space
  authenticates user
  validates session ownership
  creates durable session_runs document
  serves session metadata/messages/events
  does not own long-running agent execution

MongoDB
  sessions
  session_messages
  session_events
  session_trace_messages
  session_runs
  claude_quotas

Worker Space
  claims queued session_runs
  restores sessions from MongoDB
  runs process_submission()
  writes events/messages/snapshots
  renews leases while running
```

The worker should be a shared pool, not one spawned worker Space per user
session. Start with one worker Space and scale to multiple workers once Mongo
claim/lease behavior is proven.

## New Collection: `session_runs`

`session_runs` is the durable queue for work that must survive browser
disconnects and backend restarts.

Suggested document shape:

```json
{
  "_id": "run_uuid",
  "schema_version": 1,
  "session_id": "session_uuid",
  "user_id": "hf_user_id",
  "surface": "space",
  "operation": {
    "type": "user_input",
    "payload": {
      "text": "build a demo",
      "attachments": []
    }
  },
  "status": "queued",
  "idempotency_key": "client_generated_or_server_generated_key",
  "lease_owner": null,
  "lease_until": null,
  "retry_count": 0,
  "max_retries": 1,
  "created_at": "2026-04-28T00:00:00Z",
  "started_at": null,
  "updated_at": "2026-04-28T00:00:00Z",
  "finished_at": null,
  "error": null
}
```

Allowed `operation.type` values:

- `user_input`
- `exec_approval`
- `interrupt`
- `compact`
- `undo`
- `truncate`
- `shutdown`

Allowed `status` values:

- `queued`
- `running`
- `waiting_approval`
- `completed`
- `failed`
- `cancelled`
- `interrupted`

Indexes:

```text
{ status: 1, lease_until: 1, created_at: 1 }
{ session_id: 1, created_at: -1 }
{ user_id: 1, created_at: -1 }
{ idempotency_key: 1 } unique sparse
```

Optional later index for a worker pool:

```text
{ lease_owner: 1, lease_until: 1 }
```

## Session Metadata Additions

The current `sessions` collection already has most of what Phase 3 needs. Add or
standardize:

```text
runtime_state: idle | queued | processing | waiting_approval | ended | interrupted
active_run_id: string | null
worker_owner: string | null
worker_lease_until: datetime | null
last_event_seq: int
```

Only one active run should exist per session in v1. This avoids concurrent turns
modifying the same context.

## API Changes

### `POST /api/chat/{session_id}`

Current behavior: enqueue into an in-memory queue and stream events for the
current turn.

Phase 3 behavior:

1. Authenticate the user.
2. Verify session ownership via `ensure_session_loaded()` or a metadata-only
   ownership path.
3. Create a `session_runs` document with `status="queued"`.
4. Set `sessions.runtime_state="queued"` and `sessions.active_run_id=<run_id>`.
5. Return either:
   - `202 Accepted` with `{ run_id, session_id }`, or
   - the existing SSE response shape while internally streaming from the event
     log.

For lowest frontend disruption, v1 can keep the existing `POST /api/chat`
streaming contract. The important change is that the POST only creates durable
work; the worker owns execution.

### `GET /api/events/{session_id}?after=<seq>`

Keep this as the reconnect/replay endpoint.

Behavior:

1. Load persisted events after `after`.
2. Stream replayed events with their durable `seq`.
3. Subscribe to live event fanout if the current API process has one.
4. Keep the connection alive with comments.

Longer term, the API process should also tail Mongo events for sessions whose
worker is in another process. For v1, polling Mongo every 1-2 seconds is
acceptable and simpler than change streams.

### `GET /api/session/{session_id}` and `/messages`

Keep these as session rehydration endpoints. They should not require the worker
to be in the same process.

## Worker Space

Create a worker entrypoint, for example:

```text
ML_INTERN_PROCESS_ROLE=worker
```

The worker Space should use the same codebase and these secrets:

- `MONGODB_URI`
- `MONGODB_DB`
- model provider secrets
- any HF/tool credentials needed for agent execution

The worker does not need a public UI. In this repo, `ML_INTERN_PROCESS_ROLE=worker`
starts `worker_app:app`, which exposes `/health` for the Space while the worker
loop runs from the app lifespan.

The current implementation also supports an in-process worker for the API Space:

```text
ML_INTERN_BACKGROUND_WORKERS=true
ML_INTERN_RUN_WORKER_IN_PROCESS=true
```

That mode is the safe first rollout because the API process already has the
user's HF token in memory after request authentication. A separate worker Space
can claim the same durable runs, but user-scoped HF tool execution still needs an
explicit token handoff/token-broker design before it should be enabled for
production user traffic. Do not persist raw user OAuth tokens to Mongo as the
default path.

Worker loop:

1. Initialize Mongo session store.
2. Claim one queued or expired run atomically.
3. Restore the session context from Mongo.
4. Recreate runtime `Session`, `ToolRouter`, queues, and event persistence.
5. Mark run `running`; mark session `processing`.
6. Start a heartbeat task that renews `lease_until`.
7. Execute `process_submission()`.
8. Persist final message snapshot and runtime state.
9. Mark run `completed`, `waiting_approval`, `failed`, `cancelled`, or
   `interrupted`.

Atomic claim query:

```js
db.session_runs.findOneAndUpdate(
  {
    status: { $in: ["queued", "running"] },
    $or: [
      { status: "queued" },
      { lease_until: { $lt: now } }
    ]
  },
  {
    $set: {
      status: "running",
      lease_owner: worker_id,
      lease_until: now + lease_duration,
      started_at: now,
      updated_at: now
    },
    $inc: { retry_count: 1 }
  },
  { sort: { created_at: 1 }, returnDocument: "after" }
)
```

Use a session-level guard so two workers cannot run two turns for the same
session:

```js
db.sessions.updateOne(
  {
    _id: session_id,
    $or: [
      { active_run_id: null },
      { active_run_id: run_id },
      { worker_lease_until: { $lt: now } }
    ]
  },
  {
    $set: {
      active_run_id: run_id,
      runtime_state: "processing",
      worker_owner: worker_id,
      worker_lease_until: now + lease_duration
    }
  }
)
```

If the session guard fails, release or requeue the run.

## Handling Browser Close

With Phase 3:

1. User submits a request.
2. Backend writes `session_runs(status="queued")`.
3. Worker claims and runs it.
4. User closes the browser.
5. Nothing important happens to the run. The worker keeps executing.
6. Worker keeps appending `session_events` and saving snapshots.
7. User returns later.
8. Frontend loads `/api/sessions`, `/api/session/{id}/messages`, and
   `/api/events/{id}?after=<last_seq>`.
9. UI shows the completed turn or current progress.

The browser is an observer, not the owner of execution.

## Restart Semantics

### API Space Restart

No active run should be lost. The worker Space continues running. Reopened
browsers reconnect through the API and replay persisted events.

### Worker Space Restart Between Runs

No issue. Another worker or restarted worker claims the next queued run.

### Worker Space Restart Mid-Turn

The worker lease expires. V1 should not blindly resume arbitrary in-flight tool
calls. Instead:

1. Mark the run `interrupted`.
2. Mark session `runtime_state="interrupted"`.
3. Append an event telling the frontend the turn was interrupted.
4. Let the user continue/retry from the latest saved snapshot.

Pending approvals are the exception: if the session was waiting for user
approval, restore pending approvals exactly and keep the session in
`waiting_approval`.

## Tool-Call Idempotency Policy

Do not assume tools are safe to replay.

For v1:

- Completed turns are durable.
- Pending approvals are durable and exactly restorable.
- In-flight non-approval tool calls are interrupted on worker crash/restart.
- Long-running external HF Jobs should persist job IDs as soon as they are
  created so the restored agent can inspect status later.

Later, individual tools can opt into idempotent resume behavior.

## Frontend Changes

Frontend should make SSE reconnect normal:

- Keep `lastEventSeq` per session.
- On reconnect, call `/api/events/{session_id}?after=<lastEventSeq>`.
- Treat `POST /api/chat` response as submission acknowledgement plus optional
  live stream.
- Continue to merge server-side sessions into sidebar metadata.
- Hydrate titles from `sessions.title`.
- Hide `visibility="deleted"` sessions.

The UI should not show an expired-session recovery banner just because the
browser slept while a worker continued running.

## Implementation Phases

### Phase 3.1: Durable Run Store

- Add `session_runs` methods to `SessionStore`.
- Add Mongo indexes.
- Add typed run payload helpers.
- Add tests for enqueue, claim, lease renewal, completion, and expired lease
  recovery.

### Phase 3.2: API Enqueue Path

- Change `/api/chat/{session_id}` to create a durable run.
- Preserve current SSE response shape where possible.
- Add idempotency keys to prevent duplicate submissions on browser retry.
- Add session-level single-active-run guard.

### Phase 3.3: In-Process Worker First

- Add a worker loop inside the existing backend process behind a feature flag.
- Use it to validate queue semantics without deploying a second Space yet.
- Keep old direct execution path available behind a rollback flag.

### Phase 3.4: Separate Worker Space

- Add `backend/worker.py`.
- Add a Docker/entrypoint option for worker mode.
- Deploy `ml-intern-worker` Space with the same Mongo/model secrets.
- Disable in-process worker on the API Space once the worker Space is healthy.

### Phase 3.5: Reconnect Polish

- Ensure event replay is complete enough to rebuild visible progress.
- Add clear interrupted/retry UI states.
- Add observability for queued/running/failed/interrupted run counts.

## Testing Plan

Unit tests:

- run enqueue idempotency
- atomic claim only returns one worker winner
- lease renewal extends `lease_until`
- expired lease becomes retryable/interrupted
- one active run per session
- pending approvals survive restore
- event replay after sequence

Integration tests with local Mongo:

1. Submit a turn.
2. Kill the browser/SSE client.
3. Confirm worker completes the run.
4. Reconnect and replay events.
5. Restart API process mid-run; worker continues.
6. Restart worker mid-run; run becomes interrupted after lease expiry.
7. Submit approval after restore; worker continues from pending approval.

Production smoke:

1. Deploy API Space with Mongo enabled.
2. Deploy worker Space with same Mongo/model secrets.
3. Submit a long-ish turn.
4. Close the browser.
5. Reopen after completion and confirm the response is visible.
6. Restart API Space during a worker run and confirm no run loss.

## Observability

Add logs/metrics for:

- worker startup and worker ID
- run claimed
- run completed
- run failed
- run interrupted
- lease renewed
- lease expired
- queue depth
- oldest queued run age

Mongoku queries should make it easy to inspect:

```js
db.session_runs.find({ status: { $in: ["queued", "running"] } }).sort({ created_at: 1 })
db.sessions.find({ runtime_state: { $in: ["queued", "processing", "waiting_approval"] } })
db.session_events.find({ session_id }).sort({ seq: 1 })
```

## Rollback Plan

Keep a feature flag while migrating:

```text
ML_INTERN_BACKGROUND_WORKERS=false
```

When disabled:

- `/api/chat` uses the current direct in-process execution path.
- Mongo session persistence remains enabled.
- Event replay remains enabled.

This lets us deploy Phase 3 code safely before routing production traffic through
the worker queue.

Worker/process flags:

```text
ML_INTERN_BACKGROUND_WORKERS=true          # /api/chat enqueues durable runs
ML_INTERN_RUN_WORKER_IN_PROCESS=true      # API Space also runs a local worker
ML_INTERN_PROCESS_ROLE=worker             # run this container as worker-only
ML_INTERN_WORKER_ID=ml-intern-worker-1    # optional stable worker name
```

## Open Decisions

- Exact worker deployment name and ownership: `ml-intern-worker` is the suggested
  first name.
- Whether API Space should run an in-process worker as fallback when the external
  worker Space is unhealthy.
- Whether to use polling or Mongo change streams for live event fanout from
  external workers. Polling is simpler for v1.
- Lease duration and heartbeat interval. Suggested initial values:
  `lease_duration=120s`, `heartbeat_interval=30s`.
- How many retries before marking a run permanently `interrupted` or `failed`.
  Suggested v1: one retry for queued-before-start failures, no automatic replay
  for mid-tool-call failures.

## Non-Goals

- No per-user worker Space spawning in v1.
- No replay of arbitrary non-idempotent in-flight tool calls.
- No CLI Mongo persistence.
- No replacement of the existing sandbox Spaces; sandbox Spaces remain for user
  code execution, while the worker Space runs agent orchestration.

## Acceptance Criteria

- Closing the browser does not stop an active agent turn.
- Reopening the browser restores the session and displays completed or current
  progress.
- API Space restart does not stop worker-owned runs.
- Worker restart mid-turn produces a clear interrupted state, not silent loss.
- Pending approvals restore exactly.
- Only one worker can run a turn for a session at a time.
- Feature flag can roll back to the current direct execution path.
