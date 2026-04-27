"""
Doom-loop detection for repeated tool call patterns.

Detects when the agent is stuck calling the same tools repeatedly
and injects a corrective prompt to break the cycle.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

from litellm import Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallSignature:
    """Hashable signature for a single tool call plus its observed result."""

    name: str
    args_hash: str
    result_hash: str | None = None


def _hash_args(args_str: str) -> str:
    """Return a short hash of the JSON arguments string."""
    return hashlib.md5(args_str.encode()).hexdigest()[:12]


def extract_recent_tool_signatures(
    messages: list[Message], lookback: int = 30
) -> list[ToolCallSignature]:
    """Extract tool call signatures from recent assistant messages.

    Includes the immediate tool result hash when present. This prevents
    legitimate polling from being classified as a doom loop when the poll
    arguments stay constant but the observed result keeps changing.
    """
    signatures: list[ToolCallSignature] = []
    recent = messages[-lookback:] if len(messages) > lookback else messages

    for idx, msg in enumerate(recent):
        if getattr(msg, "role", None) != "assistant":
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if not fn:
                continue
            name = getattr(fn, "name", "") or ""
            args_str = getattr(fn, "arguments", "") or ""
            result_hash = None
            for follow in recent[idx + 1:]:
                role = getattr(follow, "role", None)
                if role == "tool" and getattr(follow, "tool_call_id", None) == getattr(tc, "id", None):
                    result_hash = _hash_args(str(getattr(follow, "content", "") or ""))
                    break
                if role in {"assistant", "user"}:
                    break
            signatures.append(
                ToolCallSignature(
                    name=name,
                    args_hash=_hash_args(args_str),
                    result_hash=result_hash,
                )
            )

    return signatures


def detect_identical_consecutive(
    signatures: list[ToolCallSignature], threshold: int = 3
) -> str | None:
    """Return the tool name if threshold+ identical consecutive calls are found."""
    if len(signatures) < threshold:
        return None

    count = 1
    for i in range(1, len(signatures)):
        if signatures[i] == signatures[i - 1]:
            count += 1
            if count >= threshold:
                return signatures[i].name
        else:
            count = 1

    return None


def detect_repeating_sequence(
    signatures: list[ToolCallSignature],
) -> list[ToolCallSignature] | None:
    """Detect repeating patterns like [A,B,A,B] for sequences of length 2-5 with 2+ reps."""
    n = len(signatures)
    for seq_len in range(2, 6):
        min_required = seq_len * 2
        if n < min_required:
            continue

        # Check the tail of the signatures list
        tail = signatures[-min_required:]
        pattern = tail[:seq_len]

        # Count how many full repetitions from the end
        reps = 0
        for start in range(n - seq_len, -1, -seq_len):
            chunk = signatures[start : start + seq_len]
            if chunk == pattern:
                reps += 1
            else:
                break

        if reps >= 2:
            return pattern

    return None


def check_for_doom_loop(messages: list[Message]) -> str | None:
    """Check for doom loop patterns. Returns a corrective prompt or None."""
    signatures = extract_recent_tool_signatures(messages, lookback=30)
    if len(signatures) < 3:
        return None

    # Check for identical consecutive calls
    tool_name = detect_identical_consecutive(signatures, threshold=3)
    if tool_name:
        logger.warning(
            "Repetition guard activated: %d+ identical consecutive calls to '%s'",
            3,
            tool_name,
        )
        return (
            f"[SYSTEM: REPETITION GUARD] You have called '{tool_name}' with the same "
            f"arguments multiple times in a row, getting the same result each time. "
            f"STOP repeating this approach — it is not working. "
            f"Step back and try a fundamentally different strategy. "
            f"Consider: using a different tool, changing your arguments significantly, "
            f"or explaining to the user what you're stuck on and asking for guidance."
        )

    # Check for repeating sequences
    pattern = detect_repeating_sequence(signatures)
    if pattern:
        pattern_desc = " → ".join(s.name for s in pattern)
        logger.warning("Repetition guard activated: repeating sequence [%s]", pattern_desc)
        return (
            f"[SYSTEM: REPETITION GUARD] You are stuck in a repeating cycle of tool calls: "
            f"[{pattern_desc}]. This pattern has repeated multiple times without progress. "
            f"STOP this cycle and try a fundamentally different approach. "
            f"Consider: breaking down the problem differently, using alternative tools, "
            f"or explaining to the user what you're stuck on and asking for guidance."
        )

    return None
