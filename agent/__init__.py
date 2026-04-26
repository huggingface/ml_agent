"""
HF Agent - Main agent module
"""

import warnings

import litellm

# Suppress harmless Pydantic serialization warnings from litellm internals.
# litellm's Message model has validate_assignment=False, so its streaming
# handler can store tool_calls as raw dicts.  When those Messages are later
# serialised via model_dump(), Pydantic emits a noisy warning:
#   "Expected `ChatCompletionMessageToolCall` … input_type=dict"
# The serialization still succeeds — this just silences the cosmetic noise.
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:",
    category=UserWarning,
    module=r"pydantic\.main",
)

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
