"""
HF Agent - Main agent module
"""

import sys
from pathlib import Path

import litellm


def resource_root() -> Path:
    """Return the directory that contains ``agent/``, ``configs/``, etc.

    Normally this is the repository root (``Path(agent.__file__).parent.parent``).
    Inside a PyInstaller frozen binary (``--onefile``) the source's
    ``__file__`` is reported as a relative path like ``agent/main.py``,
    so ``parent.parent`` collapses to the process CWD instead of the
    extracted bundle. We detect that case via ``sys._MEIPASS`` (set by
    PyInstaller's bootstrapper) and use the bundle root, where the
    ``--add-data agent/prompts:agent/prompts`` and ``configs:configs``
    payloads are extracted at startup.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent.parent


# Global LiteLLM behavior — set once at package import so both CLI and
# backend entries share the same config.
#   drop_params: quietly drop unsupported params rather than raising
#   suppress_debug_info: hide the noisy "Give Feedback" banner on errors
#   modify_params: let LiteLLM patch Anthropic's tool-call requirements
#     (synthesize a dummy tool spec when we call completion on a history
#     that contains tool_calls but aren't passing `tools=` — happens
#     during summarization / session seeding).
litellm.drop_params = True
litellm.suppress_debug_info = True
litellm.modify_params = True

from agent.core.agent_loop import submission_loop  # noqa: E402

__all__ = ["submission_loop"]
