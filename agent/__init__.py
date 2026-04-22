"""
HF Agent - Main agent module
"""

import litellm

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
