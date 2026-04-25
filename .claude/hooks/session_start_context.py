#!/usr/bin/env python3
"""
SessionStart hook — inject the dynamic session context that the standalone
CLI builds in agent/context_manager/manager.py:

  - HF username (so the agent uses the right namespace for hub_model_id)
  - Local-mode banner (only when ML_INTERN_LOCAL_MODE=1, mirrors the
    "CLI / Local mode" block injected into the system prompt)

Output is JSON `additionalContext` per Claude Code's SessionStart hook
contract — Claude Code surfaces it to the model as a system reminder.
"""

from __future__ import annotations

import json
import os
import sys


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _hf_username(token: str | None) -> tuple[str | None, str | None]:
    """Return (username, error_reason). Exactly one is non-None.

    The standalone CLI uses curl with `-4` to dodge IPv6 Happy-Eyeballs
    hangs (see agent/context_manager/manager.py:27-30). `huggingface_hub`
    is already a dep here and uses `requests`/`urllib3` which doesn't
    have the same pathology in normal setups; we use it for KISS reasons
    and accept that very-broken IPv6 environments will time out instead
    of falling back instantly.
    """
    if not token:
        return None, "no HF_TOKEN in environment"
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError

        info = HfApi(token=token).whoami()
    except HfHubHTTPError as e:
        return None, f"whoami HTTP error: {e}"
    except Exception as e:
        return None, f"whoami failed: {type(e).__name__}: {e}"

    name = info.get("name") if isinstance(info, dict) else None
    if isinstance(name, str) and name:
        return name, None
    return None, "whoami returned no name"


def main() -> int:
    try:
        sys.stdin.read()
    except Exception:
        pass

    parts: list[str] = []

    user, err = _hf_username(os.environ.get("HF_TOKEN"))
    if user:
        parts.append(
            f"HF user: **{user}** — use `{user}/<name>` as the namespace when "
            f"constructing `hub_model_id` for training jobs unless the user "
            f"specifies otherwise."
        )
    else:
        # Distinguish "no token" from "request failed" — the second case is
        # fixable (rotate token, check network), the first is configuration.
        parts.append(
            f"HF user: unknown ({err}). Ask the user for their HF org before "
            f"constructing `hub_model_id`."
        )

    if _env_flag("ML_INTERN_LOCAL_MODE", False):
        parts.append(
            "**CLI / Local mode is ON.** There is NO sandbox — `bash`, `read`, `write`, "
            "and `edit` (the `mcp__ml-intern-tools__*` versions) operate directly on the "
            "local filesystem. The `sandbox_create` tool is NOT available. Use absolute "
            "paths or paths relative to the working directory. Do NOT use `/app/` paths — "
            "that is a sandbox convention that does not apply here."
        )

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n\n".join(parts),
        }
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
