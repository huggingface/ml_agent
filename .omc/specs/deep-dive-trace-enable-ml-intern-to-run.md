# Deep Dive Trace: enable-ml-intern-to-run

## Observed Result / Problem
Enable ml-intern (deployed as a single-container HF Space) to run agent work in the background — so users can close their laptop (SSE drops), the agent keeps working, and on return they can see what happened and act on pending approvals. Target: 10,000 concurrent users.

## Ranked Hypotheses

| Rank | Hypothesis | Confidence | Evidence Strength | Why it leads |
|------|------------|------------|-------------------|--------------|
| 1 | The control plane (resumption + horizontal scale + push) has 4 specific gaps; the data plane (durability/replay) is structurally sound | High | Strong — direct code reading + HF Spaces docs | Every claim grounded in file:line evidence |
| 2 | Vertical scale + minor ergonomic fixes can carry the system to 10k | Low | Weak — math fails: 32 GB tier caps at ~1,600 sessions; asyncio scheduler degrades >2k coroutines | Requires ignoring documented HF tier ceiling |
| 3 | A full rewrite onto serverless/per-turn workers is required | Medium | Moderate — the cleanest architecture, but 2–3 months of work and breaks SSE streaming relay | Solves statelessness but introduces new latency + relay complexity |

## Evidence Summary by Hypothesis

### Lane 1 — Execution Lifecycle (Confidence: High)
- **Survives single SSE disconnect**: `_run_session` runs as detached `asyncio.Task` (session_manager.py:313, 693). Broadcaster fans regardless of subscriber count (session_manager.py:67-78). `send_event` persists to Mongo via `append_event` BEFORE in-memory queue (session.py:146-164).
- **Survives browser close + reopen with process alive**: Same path; reconnect uses `/api/events/{id}?after=<seq>` (routes/agent.py:756-781) for replay.
- **Does NOT survive process restart cleanly**: No mid-turn checkpoint. Snapshot only at turn-end via `finally` in `_run_session:731`. SIGKILL during LLM stream or tool gather drops in-flight state. `ensure_session_loaded` (session_manager.py:381-486) restores to *idle* — there is no code that detects `runtime_state == "processing"` and re-enqueues a replay submission.
- **Does NOT survive HF Space sleep**: No wake hooks; in-memory `sessions` dict wiped; lazy rehydration only on next API call.
- **Approval continuity**: Works for happy path. Fragile if process crashes between approval gate and snapshot write.

### Lane 2 — Infrastructure Topology at 10k (Confidence: High)
Sources: HF Spaces docs (spaces-overview, spaces-gpus, spaces-storage, spaces-sdks-docker).

- **Vertical scale alone fails**: 10k × 20 MB = 200 GB. Largest CPU tier = CPU Upgrade (8 vCPU / 32 GB) → ceiling at ~1,600 sessions. GPU tiers cost-prohibitive ($20-23.50/hr). Plus: 10k concurrent `asyncio.Task` coroutines + 1-second poll loops (session_manager.py:723) saturate the event loop; `AsyncMongoClient` default pool = 100 connections.
- **HF Spaces native replicas exist** via POST `/api/spaces/{ns}/{repo}/replicas` but **session-affinity / sticky-session routing is undocumented** — load balancer treats replicas as stateless. Critical gap for in-process `asyncio.Task` sessions.
- **Networking**: Only ports 80/443/8080 exposed; no documented SSE proxy timeout.
- **Storage**: ephemeral by default; S3-compatible Storage Buckets available for persistence.
- **Architecture matrix**: Hybrid eviction (E) is highest-leverage near-term move (extends per-replica capacity by evicting idle sessions to Mongo); Multi-Space sharding (B) requires custom sticky-session gateway; Externalized workers (C) is the correct long-term path but 2-3 months work.

### Lane 3 — State / Replay / Approval Reconnect (Confidence: High)
- **Event durability**: `send_event` (session.py:146-164) unconditionally calls `append_event` for every event type; persistence failure logged at DEBUG, doesn't block in-memory queue.
- **Seq monotonicity unified**: Mongo `counters` collection is the source of truth via `_next_seq` (session_persistence.py:316-323). Restart-safe.
- **Approval replay structurally sound**: `_restore_pending_approval` (session_manager.py:231-257) reconstructs `ChatCompletionMessageToolCall` with same `tool_call_id`. `submit_approval` enqueues `EXEC_APPROVAL` carrying matching ids.
- **Push notifications gated by 4 stacked flags**, all default off: `MessagingConfig.enabled`, destination in `notification_destinations`, destination's `allow_auto_events=True` (default False), event type in `auto_event_types`. Only Slack provider wired (gateway.py:27-29). Absent user receives **zero push** without pre-configuration.
- **Replay scan is unbounded** for long sessions: every streaming token = one Mongo doc; `load_events_after` does full `$gt` cursor sort with no pagination cap.
- **Large outputs**: `_safe_message_doc` (15 MB cap) replaces over-cap snapshot messages with a marker; the events collection insert path bypasses this guard and can silently fail at Mongo's 16 MB doc limit.

## Evidence Against / Missing Evidence
- **Lane 1**: No empirical confirmation that `MONGODB_URI` is set in the live production HF Space environment. Without it, `NoopSessionStore` short-circuits all durability — even SSE reconnect replay returns empty.
- **Lane 2**: HF documentation does not address replica sticky-session behavior. No published max replica count. No empirical SSE-under-replica test.
- **Lane 3**: HeartbeatSaver interval not read — determines whether `pending_approval` reaches Mongo before a process crash. Default value of `auto_event_types` not confirmed (does it include `approval_required`?).

## Per-Lane Critical Unknowns

- **Lane 1 (Execution Lifecycle)**: Is `MONGODB_URI` actually configured in the live HF Space environment? Without it, the entire durability story collapses to zero — even the existing SSE-drop reconnect path breaks.
- **Lane 2 (Infrastructure Topology)**: Do HF Spaces native replicas provide any session-affinity routing (cookie-based, header-based), or is the load balancer strictly round-robin? This single fact determines whether multi-Space sharding is deployable in 2 weeks (with sticky routing) or requires building a custom gateway Space (3+ weeks).
- **Lane 3 (State / Replay / Approval)**: HeartbeatSaver interval and reliability — does `pending_approval` get persisted to Mongo while the agent loop is frozen at the approval gate, or only at turn boundaries? Determines whether approval-flow survives a crash-while-waiting.

## Rebuttal Round
- **Best rebuttal to leader**: "The `_run_session:731` snapshot lives in `finally`, so even a thrown `process_submission` persists state — restarts may be safe." **Why leader holds**: `finally` only runs after `await process_submission(...)` returns or raises. SIGKILL never runs `finally`. SIGTERM might, but the cancelled `acompletion` stream loses partial state regardless.
- **Best rebuttal to scale**: "Hybrid eviction extends per-replica capacity 5-10×; combined with replicas, 10k might fit on a few replicas." **Why this stands**: this is exactly Lane 2's recommendation — Hybrid (E) + Sharding (B) is the realistic path; vertical-only is not.

## Convergence / Separation Notes
- Lane 1 and Lane 3 converge on a single discriminating probe: a `kill -9` mid-turn (or mid-approval-wait) followed by a restart, then a Mongo + API check, simultaneously answers whether the lifecycle resumes correctly AND whether approval state survives.
- Lane 2 stands separate — its question is about HF Spaces platform behavior, not about ml-intern code, and needs its own probe.
- Common thread: **all three lanes' critical unknowns hinge on real-world production behavior**, not code reading. The next step is empirical, not analytical.

## Most Likely Explanation
The current architecture is "background-capable" only for **client-side disconnects with the process alive** (cases A and B). It is NOT background-capable for process restarts (C), HF Space sleep (D), or 10k-user scale.

The data plane (durability, replay, seq) is structurally sound; the control plane has **four specific gaps**:
1. **No mid-turn checkpoint** + no auto-resume on rehydrate (Lane 1)
2. **No horizontal scale path**: single-replica ceiling ~1,600 sessions; HF Spaces sticky-session routing undocumented (Lane 2)
3. **No automatic push** for absent users — gateway opt-in, most flags default off, only Slack provider (Lane 3)
4. **Approval-flow fragility window** if process crashes between approval gate and next heartbeat (Lane 1 + 3)

## Critical Unknown (synthesized)
**Two unknowns gate the design**: (1) Is MongoDB actively configured and healthy in production today? (2) Do HF Spaces native replicas support sticky-session routing? The first is more load-bearing — if Mongo is off, even the existing reconnect path is broken and none of the architectural work below matters until persistence is real.

## Recommended Discriminating Probe
**Single experiment, two outcomes:**
1. Trigger a session in production → drive it to `approval_required` → `kill -9` the uvicorn process → restart → query `db.sessions.findOne({_id: <id>})` (does `pending_approval` exist with the original `tool_call_id`?) and `GET /api/events/{id}?after=0` (is the `approval_required` event replayable?). Outcome resolves Lane 1 + Lane 3 simultaneously.
2. **Independent**: deploy 2 replicas of the Space, hit `/api/session/{id}` round-robin without sticky cookie, watch whether HF's load balancer maintains affinity or routes round-robin. ~2 hours of effort. Resolves Lane 2.
