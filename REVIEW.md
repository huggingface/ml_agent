# Review instructions

These rules override the default review guidance. Treat them as the highest-priority
instruction block for any review of this repo. If something here contradicts a more
generic review habit, follow these.

## Severity levels

Every finding carries one of three priority labels:

- **P0** — blocks merge.
- **P1** — worth fixing, not blocking.
- **P2** — informational.

Write labels as plain text (`P0`, `P1`, `P2`) in finding headers. Do not use
emoji or colored markers. Use judgment on what belongs at which level — this
repo does not enumerate P0 cases; read the code and decide.

## Default bias: rigor

Reviews gate merges. This is an open-source repo that takes PRs from anyone; the
maintainer team is small and relies on the review to catch what they don't have
time to verify themselves. **Default bias is rigor, not speed.** When in doubt
on a P0-class concern, investigate further before deciding whether to flag — a
false negative ships a bug to production, a false positive costs the contributor
one round trip.

Rigor is not nitpicking. The P1 cap, "do not report" skip list, and verification
bar all still apply. Rigor means going deep on a small number of real concerns,
not surfacing a large number of shallow ones. Prefer one well-investigated P0
over three speculative P1s.

**Hold the line on P0.** If the author pushes back on a P0 finding without a fix
that actually addresses the root cause, re-state the concern with added
citations. Only accept the pushback if the author points to code or behavior you
missed. Do not soften a P0 because the contributor is polite or new to the repo.

For P1 and P2: if the author defers or pushes back without fixing, accept it
silently — do not re-flag on subsequent commits. P1/P2 are informational; the
author may defer to a follow-up issue at their discretion.

If Claude and the author repeatedly disagree on the same class of finding, the
signal is that REVIEW.md is missing a rule; note it once in the PR summary as
`suggest-rule: <short description>` and stop.

## Investigate before posting

The depth of your analysis determines the strength of your finding. For any
P0-class concern, before writing it up:

- Read the relevant callers and callees, not just the diff. Use Read and Grep
  to open files the diff doesn't touch but the changed code interacts with.
- Trace the full chain end-to-end for routing, auth, and agent-loop findings.
  Cite each hop by `file:line`, not just the suspicious line.
- Check whether the codebase already has an established pattern for this kind
  of change (`grep` for similar call sites, similar tool definitions, similar
  route guards). If the PR introduces a new approach where an established
  pattern exists, flag that — divergence from the existing pattern is usually a
  regression vector even when the new code "works."
- Confirm the specific behavior you're claiming. "This breaks X" must be
  grounded in either the code handling X or a test exercising X, not in
  inference from naming or structure.

A finding you "spotted" by scanning the diff is more likely to be a false
positive than a finding you verified by reading the code around it.

## P1 cap

Report at most **3** P1 findings per review. If you found more, say "plus N
similar items" in the summary. If everything you found is P1 or below, open the
summary with "No blocking issues."

## Re-review convergence

If this PR has already received a Claude review (there is a prior review comment
by the `claude` bot), suppress new P1 findings and post only P0 ones. Do not
re-post P1s that were already flagged on earlier commits. If the author pushed a
fix for a previously flagged issue, acknowledge it in one line rather than
re-flagging.

## Do not report

Anything in these paths — skip entirely:

- `frontend/node_modules/**`, `**/*.lock`, `uv.lock`, `package-lock.json`
- `hf_agent.egg-info/**`, `.ruff_cache/**`, `.pytest_cache/**`, `.venv/**`
- `session_logs/**`, `reports/**`
- Anything under a `gen/` or `generated/` path

Anything speculative — do not post:

- "This might be slow" without a concrete complexity claim tied to a specific
  input size
- Hypothetical race conditions without a concrete interleaving

## Dependency PRs

For PRs whose diff is only a lockfile bump, a `pyproject.toml` change, or a
new dependency, the code rules above don't apply — risks shift to provenance
and framing. Every claim in the title or body (CVE IDs, version numbers,
behavior fixes) must match what the diff actually does, and any new
transitive dep needs justification. A PR that lies in its framing is P0
regardless of whether the code change is safe in isolation.

## Verification bar

Every behavior claim in a finding must cite `file:line`. "This breaks X" is not
actionable without a line reference. If you cannot cite a line, do not post
the finding.

## Summary shape

Open the review body with a single-line tally and an explicit merge verdict, on
two lines:

```
2 P0, 3 P1
Verdict: changes requested
```

Valid verdicts:

- **Verdict: ready to merge** — no P0 findings, contributor can merge as-is
  once any CI passes
- **Verdict: changes requested** — at least one P0 that must be addressed
  before merging
- **Verdict: needs discussion** — a design-level concern the maintainer should
  weigh in on before the contributor iterates (use sparingly)

If it's a clean review, write `LGTM` followed by `Verdict: ready to merge`.

Then a **What I checked** bullet list — one line per major area you examined,
regardless of whether you found anything. This gives the maintainer visible
coverage at a glance and lets them decide whether to spot-check areas you
didn't touch.
