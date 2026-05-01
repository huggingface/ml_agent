# Deep Dive Spec: enable-ml-intern-to-run

> v1 of "background-running ml-intern": agent sessions survive SSE drops, browser close, and Main Space restarts; users return to a session that's been running and acted on while they were gone, including pending tool approvals.

## Goal

Re-architect ml-intern's session execution so that:
1. An agent session that's mid-conversation **keeps making progress** even when the user disconnects (closes laptop, network drop, tab close).
2. When the user returns — minutes or hours later — the frontend **reconnects** to the same session, **replays everything that happened** while they were gone, and surfaces any pending tool approvals so the user can act.
3. The architecture **doesn't paint us into a corner** for ~10k concurrent active users; v1 isn't sized for 10k, but the ownership/leasing model must be horizontally scalable.

## Architecture (v1)

### Topology
- **1 Main Space** (existing FastAPI + React) — hosts HTTP routes, SSE endpoints, static frontend assets, and **runs agent loops for sessions whose lease it currently holds**.
- **2 Worker Spaces** — same Docker image as Main, started with `MODE=worker` env var. Run agent loops for **leased sessions only**; expose no public HTTP routes (or only an internal health endpoint).
- **MongoDB Atlas** (or self-hosted replica set) — universal control plane. Holds session state, submissions, events, leases.

### Universal Mongo control plane
**All inter-process comms go through Mongo. There is no in-memory `submission_queue` or `EventBroadcaster` for new code paths.**
- `pending_submissions` collection: every user message / approval / interrupt is an inserted doc. Whichever process holds the session's lease consumes it.
- `session_events` collection (already exists): every event is appended; SSE streams to the frontend by tailing a change stream filtered by `session_id`.
- `sessions` collection (already exists): per-session metadata, including the new `lease` sub-document `{ holder_id, expires_at }`.
- Reads use **Mongo change streams** when available (Atlas / replica set); fall back to **500 ms polling** otherwise. The `MONGODB_URI` connection string must point at a replica-set deployment for change-stream support, documented as a deployment requirement.

### Lease + heartbeat ownership
A session's agent loop runs in exactly one process at a time, determined by an atomic CAS on the `sessions.lease` field.
- A process claims a session by atomic-`findOneAndUpdate({_id, lease.expires_at: { $lt: now }}, { $set: { lease: { holder_id: self, expires_at: now + TTL } } })`.
- The holder renews the lease every `TTL/3` seconds (e.g. TTL=30 s, renew every 10 s).
- If the holder's heartbeat lapses (process crash, deploy), `expires_at` passes; another process claims on its next sweep.
- `holder_id` is `main:{instance_id}` or `worker:{instance_id}`.

### Foreground vs background mode
**Mode is just "who holds the lease," not a separate code path.**
- New sessions are created on Main and Main claims the lease immediately (same UX as today, only the I/O layer changes).
- Sessions migrate to a Worker by Main releasing its lease and a Worker picking it up — see triggers below.
- "Re-attaching" the user to an in-Worker session means the frontend opens an SSE stream to Main, and Main tails the Mongo change stream for that session's events. The Worker keeps running.

### Migration triggers
A session leaves Main and goes to a Worker when **any** of:
1. **SSE drop + grace period** (configurable, default **3 minutes**). Main detects no active SSE subscriber for a session that's mid-turn or has unprocessed submissions; after grace period elapses with no reconnect, Main drops the lease.
2. **Manual "run in background" button** in the UI. Frontend POSTs an explicit migration request; Main drops the lease immediately.
3. **Main Space shutdown (deploy)** — only for sessions that are mid-turn (`runtime_state == "processing"`). Lifespan shutdown hook sweeps active turns and drops their leases atomically before exit. Idle sessions just naturally rehydrate later.

A free Worker claims via the same lease CAS. The agent loop resumes from the last persisted snapshot.

### Worker idle eviction
**A Worker drops a session's lease after N minutes of inactivity** (configurable, **default 30 min**). "Inactivity" = no in-flight turn AND no `pending_submissions`. Released sessions become dormant in Mongo. The next user input wakes the session: Main writes to `pending_submissions`, a Worker (or Main itself) claims the lease, rehydrates from Mongo, and processes.

### v1 launch sizing
- Main: 1 × HF Space (CPU Upgrade tier) running `MODE=main`.
- Worker: 2 × HF Spaces (same image, `MODE=worker`) — running 2 from day one validates the lease/heartbeat under contention.
- Mongo: Atlas tier sized for replica-set + change streams. (Existing tier confirmed healthy.)

## Constraints

- **No new infrastructure** beyond what already exists (FastAPI, React, MongoDB). No Redis, no Nginx gateway, no message broker, no Cloudflare Worker.
- **Timeline**: weeks, 1-2 engineers. Externalized worker pools / queue infra are explicitly out of scope.
- **Same repo, one Docker image**, mode flag at startup. Shared `agent/*` code; `backend/session_manager.py` refactored to be agnostic of mode.
- **Worker Spaces are assumed stable** — they don't restart in v1. We do not engineer for Worker process death.
- **Main Space restarts only on deploy**; the lifespan shutdown hook handles graceful migration of active turns.
- **MongoDB must be a replica set** (Atlas, or self-hosted with replica set configured) to enable change streams. Without it, polling-only fallback works but adds 500 ms latency per submission.
- **The frontend stays as-is** — no UI redesign, no async-dashboard pivot. The reconnect path (`/api/events/{id}?after=<seq>`) already exists and is the only frontend touchpoint.
- **Latency regression of 50–200 ms per user message is acceptable** as the cost of unifying on Mongo. UX target is "same minus disconnect risk," not "same latency."

## Non-Goals

- **Push notifications when away** (email / Slack / Discord) — the existing `NotificationGateway` stays opt-in. Default-on push is out of v1.
- **Async-dashboard UX** (a "your running jobs" page) — explicitly deferred.
- **10k concurrent simultaneous active load** — the design must not preclude it, but v1 ships at 1 Main + 2 Workers (~400-session ceiling with eviction in play). Scaling out is operational, not architectural, work for v2.
- **Mid-turn LLM-stream / mid-tool-call checkpoint + replay** — Worker stability assumption removes the need. If a Worker did die, sessions would rehydrate to last completed turn (existing behavior). Not engineered for v1.
- **Resuming in-flight tool calls across process restarts** — Worker assumed stable; Main restart only matters for active turns, which auto-migrate.
- **Sticky-session HTTP routing across replicas** (Nginx, Cloudflare Worker, custom gateway). Not needed because Mongo + leases handle routing; Main is single-instance in v1.
- **Refactoring the existing in-memory `submission_queue` / `EventBroadcaster` to coexist** — they get replaced wholesale by the Mongo path. One code path.

## Acceptance Criteria

### Primary drill — 5-step "close laptop, come back" (must pass)
1. POST `/api/session` → receive `session_id`. Main claims lease.
2. POST `/api/chat/{session_id}` with body `"fine-tune llama on dataset X"`. Agent starts a turn that launches an HF Job (long-running tool). SSE streams initial events.
3. Close the browser tab. SSE drops. Wait 3+ minutes.
   - **Verify**: Main drops the lease; a Worker claims it via CAS; Worker resumes the agent loop and continues making progress (turns advance, events flow into Mongo).
4. Reopen the browser 30 minutes later in the same logged-in account.
   - **Verify**: `GET /api/sessions` lists the session. Frontend opens SSE on `/api/events/{session_id}?after=<lastSeq>`; Main tails the Mongo change stream and pipes everything that happened (assistant chunks, tool calls, tool outputs, approval prompts) to the frontend. The user sees a faithful catch-up.
5. POST `/api/approve` for a pending tool approval.
   - **Verify**: Main writes to `pending_submissions`; Worker (still holding the lease) consumes it; agent resumes the turn; SSE streams new output to the frontend.

**Pass criterion**: All 5 steps succeed end-to-end, including across a deliberate Main restart between steps 2 and 3.

### Secondary drill — Main restart with active turn (must pass)
1. Start a session on Main, send a message that triggers a long-running tool call.
2. While the turn is in flight, force-restart Main (`docker restart` or Space rebuild).
3. **Verify**: Main's `lifespan` shutdown hook flips the session's lease to expired. A Worker claims within ~30 s. The agent loop resumes from the last persisted snapshot. The user reconnects to a freshly-restarted Main, opens SSE, and sees the session continuing without a "previous response interrupted" banner.

### Out-of-scope for v1 acceptance
- No 10k concurrent load test. (Optional 100-session soak test recommended pre-launch but not blocking.)
- No tool-call double-execution test. Worker restart is assumed not to happen; if it does, idempotency is a v2 problem.

## Assumptions Exposed

1. **MongoDB is configured and healthy** in the production HF Space environment today (confirmed). Without `MONGODB_URI`, the entire architecture collapses to `NoopSessionStore` and v1 cannot function.
2. **Worker Spaces don't restart** during v1 operation. If HF Spaces silently rebuilds a Worker (e.g., for security patches), in-flight turns die. We're betting this is rare enough not to engineer around in v1.
3. **HF Spaces allows two same-image Spaces in the same workspace** (Main + Workers) and they can all reach the same Mongo cluster without networking gymnastics. Standard outbound HTTPS is sufficient.
4. **Mongo change streams** are available — i.e. Mongo deployment is a replica set. Atlas is. Self-hosted single-node Mongo is not, and would force the polling fallback.
5. **Frontend tolerates the existing reconnect contract**. The `/api/events/{id}?after=<seq>` path with seq-based replay already works in `frontend/src/lib/sse-chat-transport.ts:426-453`. New work is backend-only.
6. **Sessions backgrounded once stay backgrounded.** Once a session migrates to Worker, it's never explicitly migrated back to Main — Main only relays via SSE. (Lease can move back implicitly when Worker evicts on idle, but the user-visible behavior doesn't depend on it.)
7. **2 Workers × 200 sessions/Worker = 400 concurrent backgrounded sessions** is enough headroom for v1 launch traffic. Beyond that, scale by deploying more Worker Spaces (operational, not code change — the lease design accommodates N workers).
8. **The existing `_safe_message_doc` 15 MB Mongo cap and the unbounded replay scan for long sessions** are accepted v1 tech debt. Track but don't fix in this scope.

## Technical Context

### Files to refactor
- `backend/session_manager.py` — replace `submission_queue: asyncio.Queue` with Mongo-backed reader; replace `EventBroadcaster` with change-stream tailer. Add lease claim/renew/release logic. Add `MODE=main|worker` branching in `start()` (Worker doesn't expose HTTP routes).
- `backend/main.py` — `lifespan` shutdown hook gains the "drop leases on active turns" sweep.
- `backend/start.sh` — read `MODE` env var; if `worker`, run a different uvicorn entrypoint (or no uvicorn at all — just an event-loop process).
- `backend/routes/agent.py` — `/api/chat/{id}` and `/api/approve` write to `pending_submissions`; `/api/events/{id}?after=<seq>` opens a change stream against Mongo (replacing in-memory broadcaster subscribe).
- `agent/core/session_persistence.py` — add `pending_submissions` collection with index on `(session_id, status, created_at)`; add `lease` sub-doc operations on `sessions`.
- `Dockerfile` — unchanged structurally; same image used by Main and Worker.

### New files / collections
- `backend/worker.py` (or extend `main.py` with mode flag) — Worker entrypoint: loop that polls/streams `pending_submissions`, claims leases, runs agent loops via existing `_run_session`.
- Mongo collection: `pending_submissions` — `{ _id, session_id, op_type, payload, status: pending|claimed|done, claimed_by, created_at }`.
- Mongo `sessions.lease` sub-doc.

### Files unchanged (frontend)
- `frontend/src/lib/sse-chat-transport.ts` — the reconnect logic already does what we need.
- `frontend/src/hooks/useAgentChat.ts` — no changes; the protocol is the same.

### Observability gaps to add
- Lease-state metrics (count by holder, count expiring).
- `pending_submissions` lag (claimed - created).
- Change-stream connectivity health on Main.

## Ontology

| Term | Definition |
|------|------------|
| **Main Space** | Single HF Space hosting FastAPI HTTP routes, SSE endpoints, frontend static files. Also runs agent loops for sessions whose lease it holds. |
| **Worker Space** | HF Space running same Docker image with `MODE=worker`. Runs agent loops for leased sessions only; no public HTTP. |
| **Lease** | Atomic `{ holder_id, expires_at }` on `sessions` collection. Determines which process owns a session's agent loop. Held via CAS, renewed via heartbeat, expired by clock. |
| **Migration** | The lease moving from one holder to another. Triggered by SSE-drop grace-period elapse, manual button, or Main shutdown. No data is "moved" — the session lives in Mongo throughout. |
| **Pending submission** | A user-originated operation (message, approval, interrupt) waiting for the lease holder to consume. Replaces today's in-memory `submission_queue`. |
| **Backgrounded session** | A session whose lease is currently held by a Worker. From the user's POV, a session "in the background." |
| **Re-attach** | The frontend opens SSE on Main; Main tails Mongo change stream for that session's events. The Worker (if it holds the lease) continues to run unaware. |
| **Grace period** | Time after SSE drop before Main releases the lease. Configurable env var, default 3 min. |
| **Idle eviction** | Worker drops a session's lease after N min of no in-flight turn and no pending submissions. Default 30 min. |

## Ontology Convergence

- **"Background mode" was reframed mid-interview** from a separate state-machine to "just a different lease holder." This collapsed the design surface dramatically.
- **"Migration" started as data movement** between Main and Worker; converged to "a lease changing hands" with the data already in Mongo. No copy step.
- **"Worker"** is unambiguous — same image, different startup flag, no public routes.
- **"10k users"** was reframed from a hard v1 capacity target to an aspirational design constraint after the user's "you're overengineering" pushback. Agreed: the architecture must accommodate scale-out via more Workers, but v1 isn't a capacity-driven release.

## Trace Findings

The 3-lane causal trace (saved at `.omc/specs/deep-dive-trace-enable-ml-intern-to-run.md`) found:

- **Data plane is structurally sound**: Mongo seq monotonicity, `send_event → append_event` durability, `_restore_pending_approval` reconstruction, `/api/events/{id}?after=<seq>` replay are all wired correctly. The spec leans heavily on this — we're not reinventing it, only redirecting the in-memory paths through the same Mongo collections.
- **Control plane has 4 gaps**: no mid-turn checkpoint and no auto-resume (this spec addresses via lease + Mongo-backed submissions), no horizontal scale path (this spec addresses via lease + multiple Workers), no automatic push-when-away (explicit non-goal), approval-flow crash-window fragility (mitigated by Worker stability assumption + lease handover on Main shutdown).
- **Critical unknown #1 — Mongo configured in prod** — confirmed by user. v1 can proceed.
- **Critical unknown #2 — HF Spaces sticky-session routing** — sidestepped: Mongo is the routing mechanism, not HTTP load balancing. Multi-replica HTTP is no longer required for v1.
- **Critical unknown #3 — HeartbeatSaver interval** — sidestepped: lease-based heartbeats replace it. The lease mechanism's TTL/3 renewal cadence is the new failure-detection clock.

## Interview Transcript

**Round 1 — environment baseline:**
- Mongo healthy in prod? → **Yes.**
- v1 scope? → **Same UX, just durable** (no async-dashboard rebuild, no push).
- Timeline / headcount? → **Weeks, 1-2 engineers.**

**Round 2 — Goal precision on "10k":**
- Definition? → **10k actively chatting at once** (most demanding interpretation).

**Round 3 — first architecture pass (later reframed):**
- Sticky routing? → User pushed back: **"You are overengineering this. Focus on after-people-make-a-session-and-close-it."**
- Restart contract? → **Full mid-turn checkpoint + replay** (later softened by Worker-stability assumption).
- Success criterion? → **Behavioral drills, not load.**

**Round 4 — reframe to Main + Worker architecture:**
- v1 reframe? → User volunteered the architecture: **"Detect SSE drop or background button, move session to a Worker Space, rehydrate Main from Worker on return."**
- Mid-turn replay? → **Worker spaces won't restart; Main only restarts on deploy.**
- Crash drivers? → **Occasional, deploys only.**

**Round 5 — Main↔Worker boundary:**
- Migration trigger? → **Both** (auto on SSE drop + manual button).
- Comms? → **Mongo as the only channel.**
- Re-attach? → **Main subscribes, Worker keeps running** (no handoff dance).

**Round 6 — control flow:**
- Submission flow? → **Mongo-backed `pending_submissions` collection.**
- Worker assignment? → **Lease + heartbeat.**
- Acceptance drill? → **5-step manual drill** (with preview adopted).

**Round 7 — deployment shape and edges:**
- Codebase? → **Same repo, MODE flag at startup.**
- Launch shape? → **1 Main + 2 Workers** (validates lease under contention).
- Grace period? → **Configurable, default 3 min.**
- Main restart contract? → **Auto-migrate active turns only.**

**Round 8 — final mechanics:**
- Worker idle eviction? → **Evict after N min idle, default 30 min** (rehydrate on input).
- Mongo read mode? → **Change streams when available, polling fallback.**
- Foreground path? → **Unify on Mongo for everything** (one code path).
