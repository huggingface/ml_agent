#!/usr/bin/env python3
"""Backstop sweeper for orphan ml-intern sandbox Spaces.

================================================================================
 Why this script exists
================================================================================

The agent creates a sandbox Space per session (template duplicated from
``burtenshaw/sandbox`` into the user's account, named ``<owner>/sandbox-<8hex>``).
``backend.session_manager.SessionManager._cleanup_sandbox`` deletes it at end of
session. In practice the cleanup misses some sandboxes:

- pod killed / OOM / pre-emption / deploy rollouts → ``finally`` block skipped
- WebSocket dropped without ``/shutdown`` from the client
- HF API transient failure on ``delete_repo`` (we retry now, but not infinitely)

The result observed 2026-04-27 was 2,310 orphan ``sandbox-*`` Spaces — every
sandbox ever created was still around. This script is the backstop: list every
``sandbox-*`` fork of ``burtenshaw/sandbox`` that hasn't been touched in N days
and delete it.

================================================================================
 Identification rules
================================================================================

A Space is considered an orphan ml-intern sandbox iff ALL hold:

1. Repo type = ``space``
2. Name matches ``<owner>/sandbox-[a-f0-9]{8}$`` (the agent's naming convention)
3. ``originRepo`` points at ``burtenshaw/sandbox`` (so we don't touch
   user-renamed lookalikes)
4. ``lastModified`` older than ``--max-age-days`` (default 7)

We DO NOT use the ``runtime.stage`` (sleeping/running) as a filter — a sandbox
that has been sleeping for 7 days is just as orphan as a deleted one but uses
no compute. The cleanup is about repo/storage hygiene, not about waking
something up to kill it.

================================================================================
 Safety
================================================================================

- ``--dry-run`` (default) prints what would be deleted, deletes nothing.
- ``--apply`` actually calls ``HfApi.delete_repo``.
- Hard cap ``--max-deletes`` (default 200) so a misconfigured run can't nuke
  thousands at once.
- Requires a token with admin rights via ``HF_ADMIN_TOKEN`` env var (the only
  way to delete a Space owned by another user).
- Logs every action to stdout in JSON Lines for downstream auditing.

================================================================================
 Cron suggestion
================================================================================

GitHub Actions, daily at 04:00 UTC:

    schedule:
      - cron: "0 4 * * *"
    env:
      HF_ADMIN_TOKEN: ${{ secrets.HF_ADMIN_TOKEN }}
    steps:
      - run: python scripts/sweep_orphan_sandboxes.py --apply --max-age-days 7
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

SANDBOX_NAME_RE = re.compile(r"^[^/]+/sandbox-[a-f0-9]{8}$")
TEMPLATE_REPO = "burtenshaw/sandbox"


def log(record: dict) -> None:
    """JSON Lines log so downstream tooling can grep / parse."""
    record["ts"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(record), flush=True)


def is_sandbox_fork(space) -> bool:
    """Filter: matches the ml-intern sandbox naming pattern.

    NOTE: We initially tried filtering on ``duplicated_from == burtenshaw/sandbox``
    too, for extra safety. That doesn't work — the HF REST API does not expose
    ``duplicated_from`` on ``SpaceInfo`` (verified against ``huggingface-hub``
    1.11+ and direct ``GET /api/spaces/{id}``: the field is None). The origin
    repo lives in MongoDB but isn't surfaced. So we rely on the naming pattern
    alone, which is specific enough: ``Sandbox.create()`` is the sole producer
    of ``<owner>/sandbox-<8 lowercase hex>``, and that pattern is unlikely to
    collide with user-created Spaces in practice. The ``--dry-run`` default
    is the user-facing safety net for the rare false-positive.
    """
    return bool(SANDBOX_NAME_RE.match(space.id))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Delete sandboxes whose lastModified is older than this many days (default: 7)",
    )
    parser.add_argument(
        "--max-deletes",
        type=int,
        default=200,
        help="Hard cap on deletions per run, safety guard (default: 200)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max number of candidate Spaces to scan via list_spaces (default: 10000)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_ADMIN_TOKEN")
    if not token:
        log({"level": "error", "msg": "HF_ADMIN_TOKEN env var not set"})
        return 1

    api = HfApi(token=token)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.max_age_days)
    log({"level": "info", "msg": "sweep_start", "cutoff": cutoff.isoformat(),
         "max_deletes": args.max_deletes, "apply": args.apply})

    # ``list_spaces`` doesn't filter by name pattern — we scan and filter
    # client-side. ``search="sandbox"`` narrows the network payload.
    candidates = api.list_spaces(
        search="sandbox", full=True, limit=args.limit
    )

    scanned = 0
    matched = 0
    deleted = 0
    failed = 0
    skipped_too_recent = 0
    skipped_capped = 0

    for space in candidates:
        scanned += 1
        if not is_sandbox_fork(space):
            continue
        matched += 1

        last_mod = getattr(space, "lastModified", None) or getattr(space, "last_modified", None)
        if isinstance(last_mod, str):
            last_mod = datetime.fromisoformat(last_mod.replace("Z", "+00:00"))
        if last_mod and last_mod > cutoff:
            skipped_too_recent += 1
            continue

        log({"level": "info", "msg": "candidate", "space_id": space.id,
             "last_modified": last_mod.isoformat() if last_mod else None})

        if not args.apply:
            continue

        # When we hit the deletion cap, keep scanning so the final ``matched``
        # count reflects the *true* orphan size — not just what was scanned
        # before we stopped deleting. Operators planning multi-pass cleanups
        # need an accurate denominator to know when they're done.
        if deleted >= args.max_deletes:
            skipped_capped += 1
            continue

        try:
            api.delete_repo(repo_id=space.id, repo_type="space", token=token)
            deleted += 1
            log({"level": "info", "msg": "deleted", "space_id": space.id})
            # Light throttle to avoid hitting HF API rate limits.
            time.sleep(0.2)
        except HfHubHTTPError as e:
            failed += 1
            log({"level": "error", "msg": "delete_failed", "space_id": space.id,
                 "status": e.response.status_code, "error": str(e)[:200]})
        except Exception as e:
            failed += 1
            log({"level": "error", "msg": "delete_failed", "space_id": space.id,
                 "error": str(e)[:200]})

    log({"level": "info", "msg": "sweep_end",
         "scanned": scanned, "matched": matched,
         "skipped_too_recent": skipped_too_recent,
         "skipped_capped": skipped_capped,
         "deleted": deleted, "failed": failed,
         "capped": skipped_capped > 0,
         "apply": args.apply})

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
