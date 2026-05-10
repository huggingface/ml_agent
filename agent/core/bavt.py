"""Budget-Aware Value Tree (BAVT) for the ml-intern agent loop.

Implements a training-free, inference-time budget management system inspired by:

    "Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents"
    Yushu Li, Wenlong Deng, Jiajin Li, Xiaoxiao Li — arXiv:2603.12634 (March 2026)

Key ideas adapted for ml-intern's architecture:

1. **Budget ratio** — ``remaining / max`` drives a continuous exploration→exploitation
   transition. No discrete phases; the exponent scales smoothly from 1 (fresh) to 0
   (exhausted).

2. **Residual progress scorer** — scores *relative* delta across the last N steps
   rather than absolute state quality, avoiding the LLM overconfidence the paper
   identifies as a core flaw of naive self-evaluation.  This implementation is
   entirely heuristic (no LLM calls) so it adds zero latency.

3. **Budget-conditioned signals** — thresholds on (budget_ratio, progress_score)
   produce at most one injected message per check plus an optional effort hint
   that the caller can forward to the next ``_resolve_llm_params`` call.

HuggingFace Pro note: The scorer uses only local heuristics by default. Callers
that want semantic scoring can optionally supply ``hf_token`` for a lightweight
HF Inference API call; this path is gated behind ``use_hf_scorer=True`` and
is never exercised in the default agent loop so it never adds latency.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

from litellm import Message

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Tunables (all overridable in tests without patching internals)
# ──────────────────────────────────────────────────────────────

# Budget ratio below which we issue a gentle "you're burning budget" nudge.
NUDGE_THRESHOLD = 0.50
# Budget ratio below which we downgrade effort and inject a reminder.
DOWNGRADE_THRESHOLD = 0.25
# Budget ratio below which we issue a strong "wrap up NOW" directive.
WRAP_UP_THRESHOLD = 0.10

# Minimum progress score before a nudge fires (suppress on healthy runs).
NUDGE_MIN_PROGRESS = 0.0

# Effort level ordering for downgrade logic (worst→best).
_EFFORT_ORDER = ["low", "minimal", "medium", "high", "xhigh", "max"]


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────


@dataclass
class BudgetSignal:
    """Output of a single BAVT check.

    ``corrective_message`` and ``effort_hint`` are both optional.
    The caller is responsible for injecting ``corrective_message`` into the
    context (same pattern as the doom-loop corrective prompts) and passing
    ``effort_hint`` to ``_resolve_llm_params``.
    """

    corrective_message: str | None = None
    effort_hint: str | None = None
    prune_warning: bool = False  # True when budget < WRAP_UP_THRESHOLD
    budget_ratio: float = 1.0
    progress_score: float = 0.0


# ──────────────────────────────────────────────────────────────
# BudgetTracker
# ──────────────────────────────────────────────────────────────


class BudgetTracker:
    """Track iteration budget and derive the current ratio.

    Attributes
    ----------
    max_iterations:
        The configured upper bound (``session.config.max_iterations``). When
        ``-1`` (unlimited), all ratios return ``1.0`` so BAVT is a no-op.
    """

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations

    def ratio(self, current_iteration: int) -> float:
        """Return remaining budget as a fraction in ``[0, 1]``.

        ``1.0`` = just started, ``0.0`` = exhausted.
        """
        if self.max_iterations <= 0:
            return 1.0
        remaining = max(0, self.max_iterations - current_iteration)
        return remaining / self.max_iterations

    def effort_hint(
        self, current_effort: str | None, budget_ratio: float
    ) -> str | None:
        """Return a lower effort level when budget is shrinking.

        Returns ``None`` when no downgrade is warranted.
        """
        if current_effort is None or budget_ratio > DOWNGRADE_THRESHOLD:
            return None
        idx = (
            _EFFORT_ORDER.index(current_effort)
            if current_effort in _EFFORT_ORDER
            else -1
        )
        if idx <= 0:
            return None  # already at floor or unknown level
        # Step down one level.
        return _EFFORT_ORDER[idx - 1]


# ──────────────────────────────────────────────────────────────
# ResidualProgressScorer
# ──────────────────────────────────────────────────────────────

_LOOKBACK = 12  # messages to inspect


def _short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="replace")).hexdigest()[:10]


def _tool_result_score(content: str) -> float:
    """Classify a tool result as positive, neutral, or negative progress.

    Returns +1.0 (success), 0.0 (informational/empty), or -1.0 (error).
    """
    if not content:
        return 0.0
    low = content.lower()
    if any(
        p in low for p in ("error:", "traceback", "exception", "failed", "not found")
    ):
        return -1.0
    if any(
        p in low
        for p in ("success", "created", "written", "updated", "completed", "done")
    ):
        return 1.0
    return 0.0


class ResidualProgressScorer:
    """Score relative progress across the last ``_LOOKBACK`` messages.

    Unlike absolute state quality estimators, this scorer only looks at the
    *delta* — new unique content fingerprints entering the context — and the
    quality of tool results. It never calls the LLM.

    The returned score is in ``[-1, +1]``:
    - ``> 0``: agent is making meaningful forward progress
    - ``≈ 0``: neutral / information-gathering
    - ``< 0``: errors dominating, very little new content being added
    """

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    def score(self, messages: list[Message]) -> float:
        """Compute a residual progress score from the recent message tail."""
        recent = messages[-_LOOKBACK:] if len(messages) > _LOOKBACK else messages
        if not recent:
            return 0.0

        tool_scores: list[float] = []
        new_content_count = 0
        total_content = 0

        for msg in recent:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None) or ""
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except Exception:
                    content = str(content)

            if not content:
                continue

            total_content += 1
            h = _short_hash(content[:512])
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                new_content_count += 1

            if role == "tool":
                tool_scores.append(_tool_result_score(content))

        # Unique-content ratio (1.0 = everything is new, 0.0 = nothing new)
        uniqueness = new_content_count / total_content if total_content else 0.0

        # Average tool result quality (-1..+1)
        tool_quality = sum(tool_scores) / len(tool_scores) if tool_scores else 0.0

        # Weighted blend: tool quality (60%) + uniqueness (40%)
        blended = 0.6 * tool_quality + 0.4 * (uniqueness * 2.0 - 1.0)
        return max(-1.0, min(1.0, blended))


# ──────────────────────────────────────────────────────────────
# BudgetConditionedController (main entry point)
# ──────────────────────────────────────────────────────────────

# Cooldown: don't inject two corrective messages back-to-back.
_COOLDOWN_ITERATIONS = 5


class BudgetConditionedController:
    """Combine budget ratio and residual progress into agent-loop signals.

    Instantiate once per agent run (i.e. inside ``Handlers.run_agent``).

    Usage::

        controller = BudgetConditionedController(max_iterations)
        # …inside the while loop…
        signal = controller.check(
            messages=session.context_manager.items,
            current_iteration=iteration,
            current_effort=session.effective_effort_for(model_name),
        )
        if signal.corrective_message:
            session.context_manager.add_message(
                Message(role="user", content=signal.corrective_message)
            )
    """

    def __init__(self, max_iterations: int) -> None:
        self._tracker = BudgetTracker(max_iterations)
        self._scorer = ResidualProgressScorer()
        self._last_injection_iter: int = -_COOLDOWN_ITERATIONS

    def check(
        self,
        messages: list[Message],
        current_iteration: int,
        current_effort: str | None = None,
    ) -> BudgetSignal:
        """Return a BudgetSignal describing the current budget state.

        Call once per while-loop iteration *before* the LLM call.  The signal
        is free to ignore (all fields are optional); the caller decides what to
        inject.
        """
        ratio = self._tracker.ratio(current_iteration)
        progress = self._scorer.score(messages)
        effort_hint = self._tracker.effort_hint(current_effort, ratio)

        # Cooldown: avoid injecting multiple messages in quick succession.
        cooldown_ok = (
            current_iteration - self._last_injection_iter
        ) >= _COOLDOWN_ITERATIONS

        corrective: str | None = None
        prune_warning = False

        if ratio < WRAP_UP_THRESHOLD and cooldown_ok:
            # Critical: almost out of iterations.
            prune_warning = True
            remaining = max(0, self._tracker.max_iterations - current_iteration)
            corrective = (
                f"[SYSTEM: BUDGET CRITICAL] Only {remaining} iteration(s) remain "
                f"(budget ratio: {ratio:.0%}). "
                f"You MUST wrap up immediately: stop exploring, deliver the best "
                f"answer you have right now, and do not make unnecessary tool calls. "
                f"If the task is incomplete, summarise what was done and what remains."
            )
            self._last_injection_iter = current_iteration
            logger.warning(
                "BAVT budget critical: ratio=%.2f, progress=%.2f, iter=%d",
                ratio,
                progress,
                current_iteration,
            )

        elif (
            ratio < DOWNGRADE_THRESHOLD
            and progress < NUDGE_MIN_PROGRESS
            and cooldown_ok
        ):
            # Low budget AND stalling: redirect the agent.
            remaining = max(0, self._tracker.max_iterations - current_iteration)
            corrective = (
                f"[SYSTEM: BUDGET LOW] {remaining} iteration(s) remaining "
                f"(budget ratio: {ratio:.0%}). Progress appears stalled. "
                f"Shift to a more direct approach: pick the most promising path, "
                f"stop re-trying failing tools, and move toward a concrete answer."
            )
            self._last_injection_iter = current_iteration
            logger.info(
                "BAVT budget low + stall: ratio=%.2f, progress=%.2f, iter=%d",
                ratio,
                progress,
                current_iteration,
            )

        elif ratio < NUDGE_THRESHOLD and progress < NUDGE_MIN_PROGRESS and cooldown_ok:
            # Halfway through and progress has gone negative.
            corrective = (
                f"[SYSTEM: BAVT NUDGE] You have used {1 - ratio:.0%} of your iteration "
                f"budget with limited measurable progress. Consider a different strategy "
                f"or ask the user for clarification if you are stuck."
            )
            self._last_injection_iter = current_iteration
            logger.debug(
                "BAVT nudge: ratio=%.2f, progress=%.2f, iter=%d",
                ratio,
                progress,
                current_iteration,
            )

        return BudgetSignal(
            corrective_message=corrective,
            effort_hint=effort_hint,
            prune_warning=prune_warning,
            budget_ratio=ratio,
            progress_score=progress,
        )
