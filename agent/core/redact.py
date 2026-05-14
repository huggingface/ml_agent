"""Secret scrubbing for session trajectories before upload.

Users frequently paste HF / API / GitHub tokens into the chat, or scripts echo
them via env dumps. This module applies regex-based redaction to any string
value found recursively in a trajectory payload. The goal is best-effort —
strict formats are matched; we won't catch free-form leaks like "my password
is hunter2".
"""

from __future__ import annotations

import re
from typing import Any

# Each entry: (compiled regex, replacement placeholder).
# Patterns are conservative: they only match tokens with the canonical prefix
# and a minimum body length so we don't paint over normal text.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Hugging Face tokens: hf_[A-Za-z0-9]{30,}
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "[REDACTED_HF_TOKEN]"),
    # Anthropic: sk-ant-[A-Za-z0-9_\-]{20,}
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "[REDACTED_ANTHROPIC_KEY]"),
    # OpenAI: sk-[A-Za-z0-9]{40,}  (legacy + proj keys)
    (re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{40,}"), "[REDACTED_OPENAI_KEY]"),
    # GitHub classic PATs: ghp_, gho_, ghu_, ghs_, ghr_ followed by 36+ chars
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    # GitHub fine-grained PATs: github_pat_<alphanumeric_underscore>
    (re.compile(r"github_pat_[A-Za-z0-9_]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    # AWS access key IDs: AKIA / ASIA + 16 uppercase alnum
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[REDACTED_AWS_KEY_ID]"),
    # Generic 'Bearer <token>' header values
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}"), "Bearer [REDACTED]"),
]

# Env-var-like exports: we scrub the value but keep the name so callers can
# still see which secret was referenced. Covers `KEY=value` and `KEY: value`
# when the key looks secret-y.
_SECRETY_NAMES = re.compile(
    r"(?i)\b(HF_TOKEN|HUGGINGFACEHUB_API_TOKEN|ANTHROPIC_API_KEY|OPENAI_API_KEY|"
    r"GITHUB_TOKEN|AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|PASSWORD|SECRET|API_KEY)"
    r"\s*[:=]\s*([^\s\"']+)"
)


def scrub_string(s: str) -> str:
    """Apply all redaction patterns to a single string. Safe on non-strings."""
    if not isinstance(s, str) or not s:
        return s
    out = s
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    out = _SECRETY_NAMES.sub(lambda m: f"{m.group(1)}=[REDACTED]", out)
    return out


def scrub(obj: Any) -> Any:
    """Recursively scrub every string value in a nested dict/list structure.

    Returns a new object — inputs are not mutated."""
    if isinstance(obj, str):
        return scrub_string(obj)
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub(v) for v in obj)
    return obj
