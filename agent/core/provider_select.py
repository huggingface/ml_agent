"""Auto-select the model provider from available credentials.

The agent supports several providers; the canonical fallback order, per
the project README, is:

    1. Anthropic API (``ANTHROPIC_API_KEY``)            — preferred
    2. GitHub Copilot (``GITHUB_COPILOT_TOKEN`` /
       ``GH_COPILOT_TOKEN`` / pre-existing LiteLLM OAuth cache)
    3. OpenCode Zen free tier (``OPENCODE_API_KEY``)    — last resort

The model id baked into ``configs/main_agent_config.json`` is used only
when none of the above credentials are present (for example, on a
machine that already has AWS / HF router auth wired up).

This module is consumed by both the interactive REPL and the headless
entrypoint so the cascade is consistent across launch modes. It does
not perform any network I/O.
"""

from __future__ import annotations

import os
from pathlib import Path


# Default model ids for each provider. We pick conservative, currently-
# available ids so first-run UX is "it just works". Users can override
# with ``--model``, ``ML_INTERN_MODEL``, or by editing the config file.
DEFAULTS: dict[str, str] = {
    "anthropic": "anthropic/claude-opus-4-7",
    "copilot":   "copilot/gpt-5",
    "opencode":  "opencode/minimax-m2.5-free",
}


def _has_copilot_credentials() -> bool:
    """Truthy if Copilot can authenticate without prompting the user.

    Either a static token is in env, or LiteLLM has previously completed
    its device-flow OAuth dance and cached state on disk.
    """
    if os.environ.get("GITHUB_COPILOT_TOKEN") or os.environ.get("GH_COPILOT_TOKEN"):
        return True
    cache_dir = Path.home() / ".config" / "litellm" / "github_copilot"
    return cache_dir.is_dir() and any(cache_dir.iterdir())


def auto_select_model(explicit: str | None = None) -> tuple[str, str]:
    """Return ``(model_id, source)`` for the best available provider.

    ``explicit`` is the value of ``--model`` (or any other caller-side
    override); when set, it wins unconditionally and ``source`` is
    ``"explicit"``.

    Otherwise the cascade walks Anthropic → Copilot → OpenCode and
    returns the first match. ``source`` is one of ``"explicit"``,
    ``"env:ML_INTERN_MODEL"``, ``"anthropic"``, ``"copilot"``,
    ``"opencode"`` (when ``OPENCODE_API_KEY`` is set), or
    ``"opencode-anonymous"`` (when no credentials at all are present —
    OpenCode Zen serves ``-free`` model ids without auth so the agent
    can always boot).
    """
    if explicit:
        return explicit, "explicit"

    env_pin = os.environ.get("ML_INTERN_MODEL")
    if env_pin:
        return env_pin, "env:ML_INTERN_MODEL"

    if os.environ.get("ANTHROPIC_API_KEY"):
        return DEFAULTS["anthropic"], "anthropic"

    if _has_copilot_credentials():
        return DEFAULTS["copilot"], "copilot"

    # OpenCode Zen's ``-free`` model ids are served anonymously, so we can
    # always fall back to them even without ``OPENCODE_API_KEY``. A key
    # only matters if the user has a paid OpenCode account; for the free
    # tier the upstream blanks the Authorization header for us (see
    # ``agent.core.llm_params._resolve_llm_params``).
    if os.environ.get("OPENCODE_API_KEY"):
        return DEFAULTS["opencode"], "opencode"
    return DEFAULTS["opencode"], "opencode-anonymous"


def apply_provider_cascade(config, explicit: str | None = None) -> str | None:
    """Mutate ``config.model_name`` to the auto-selected model when one
    is available.

    Returns the ``source`` tag (see :func:`auto_select_model`) so the
    caller can log which provider was picked, or ``None`` when no
    cascade was applied (we kept whatever was in the config).
    """
    model_id, source = auto_select_model(explicit)
    if source == "none":
        return None
    config.model_name = model_id
    return source
