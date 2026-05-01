# Open Questions

## ralplan-enable-ml-intern-to-run - 2026-05-01 (revised post-iteration-2)

> Iteration 2 of `/plan --consensus` resolved 3 of the 8 questions. Remaining 5 are annotated as "during execution" — they don't block the plan landing.

**Resolved in iteration 2 (removed):**
- ~~Backward-compat for in-flight sessions~~ — locked option (a) backfill empty `lease={holder_id:null, expires_at:0}` for sessions with `last_active_at > now-1h`; older active sessions flip `runtime_state` to `idle`.
- ~~Holder ID format~~ — locked `f"{mode}:{hostname}:{uuid7().hex[:8]}"` (uuid4 fallback for pre-3.13 Python). Rationale: uuid7 sorts chronologically.
- ~~Idle eviction default~~ — locked 30 min, per-session, with `not is_in_tool_call AND not is_processing AND no pending_submissions` predicate.

**Remaining (during execution):**
- [ ] **Worker entrypoint shape** [during execution] — Plan default: `backend/worker.py` is a 3-line shim calling `worker_loop()` from `backend/main.py`. Reversible. Confirm during Step 6 implementation.
- [ ] **Manual "background" button frontend wiring** [during execution] — Backend route ships in Step 5. Frontend POST hook is post-v1; confirm acceptable.
- [ ] **Lease TTL/renew defaults** [during execution] — TTL=30s, renew=10s per spec. Long tool calls hold the lease for hours; renewal must keep pace under load. Confirm under integration test.
- [ ] **Polling fallback cadence = 500 ms** [during execution] — Spec default. Confirm acceptable when Mongo is single-node (dev/local) and change streams unavailable.
- [ ] **`_safe_message_doc` 15 MB truncation visibility** [during execution] — Plan adds a `WARN` log + session_event when the marker is written. Confirm whether the user-visible event text is acceptable, or whether v1 should silently truncate.
