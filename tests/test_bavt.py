"""Unit tests for the BAVT (Budget-Aware Value Tree) module.

Covers BudgetTracker, ResidualProgressScorer, and BudgetConditionedController.
"""

from __future__ import annotations

from litellm import Message

from agent.core.bavt import (
    BudgetConditionedController,
    BudgetTracker,
    ResidualProgressScorer,
)


# ──────────────────────────────────────────────────────────────
# BudgetTracker
# ──────────────────────────────────────────────────────────────


class TestBudgetTracker:
    def test_ratio_full_budget(self) -> None:
        t = BudgetTracker(max_iterations=100)
        assert t.ratio(0) == 1.0

    def test_ratio_half_budget(self) -> None:
        t = BudgetTracker(max_iterations=100)
        assert t.ratio(50) == 0.5

    def test_ratio_exhausted(self) -> None:
        t = BudgetTracker(max_iterations=100)
        assert t.ratio(100) == 0.0

    def test_ratio_over_budget_clamps(self) -> None:
        t = BudgetTracker(max_iterations=100)
        assert t.ratio(200) == 0.0

    def test_ratio_unlimited(self) -> None:
        """When max_iterations == -1 the ratio is always 1.0 (BAVT is a no-op)."""
        t = BudgetTracker(max_iterations=-1)
        assert t.ratio(999) == 1.0

    def test_effort_hint_above_threshold_is_none(self) -> None:
        t = BudgetTracker(max_iterations=100)
        assert t.effort_hint("high", budget_ratio=0.9) is None

    def test_effort_hint_downgrade_one_level(self) -> None:
        t = BudgetTracker(max_iterations=100)
        # budget_ratio = 0.2 → below DOWNGRADE_THRESHOLD
        hint = t.effort_hint("high", budget_ratio=0.2)
        assert hint == "medium"

    def test_effort_hint_at_floor_is_none(self) -> None:
        t = BudgetTracker(max_iterations=100)
        hint = t.effort_hint("low", budget_ratio=0.05)
        assert hint is None  # already at lowest level

    def test_effort_hint_unknown_level_is_none(self) -> None:
        t = BudgetTracker(max_iterations=100)
        hint = t.effort_hint("turbo", budget_ratio=0.05)
        assert hint is None

    def test_effort_hint_none_effort_is_none(self) -> None:
        t = BudgetTracker(max_iterations=100)
        assert t.effort_hint(None, budget_ratio=0.05) is None


# ──────────────────────────────────────────────────────────────
# ResidualProgressScorer
# ──────────────────────────────────────────────────────────────


def _make_tool_msg(content: str) -> Message:
    return Message(role="tool", content=content)


def _make_assistant_msg(content: str) -> Message:
    return Message(role="assistant", content=content)


class TestResidualProgressScorer:
    def test_empty_messages_returns_zero(self) -> None:
        scorer = ResidualProgressScorer()
        assert scorer.score([]) == 0.0

    def test_all_success_messages_positive(self) -> None:
        scorer = ResidualProgressScorer()
        msgs = [_make_tool_msg("success: file written") for _ in range(5)]
        score = scorer.score(msgs)
        assert score > 0.0

    def test_all_error_messages_negative(self) -> None:
        scorer = ResidualProgressScorer()
        msgs = [_make_tool_msg("ERROR: command not found") for _ in range(5)]
        score = scorer.score(msgs)
        assert score < 0.0

    def test_mixed_content_neutral(self) -> None:
        scorer = ResidualProgressScorer()
        msgs = [
            _make_tool_msg("success: done"),
            _make_tool_msg("error: failed"),
        ]
        score = scorer.score(msgs)
        # Mixed — should be near zero
        assert -0.5 < score < 0.5

    def test_repeated_identical_content_lower_score(self) -> None:
        """Repeated identical content should yield lower score than fresh content."""
        scorer_fresh = ResidualProgressScorer()
        scorer_stale = ResidualProgressScorer()

        fresh_msgs = [_make_tool_msg(f"unique output {i}") for i in range(5)]
        stale_msgs = [_make_tool_msg("same output every time") for _ in range(5)]

        score_fresh = scorer_fresh.score(fresh_msgs)
        score_stale = scorer_stale.score(stale_msgs)
        assert score_fresh > score_stale


# ──────────────────────────────────────────────────────────────
# BudgetConditionedController
# ──────────────────────────────────────────────────────────────


class TestBudgetConditionedController:
    def _msgs(self, n: int = 5) -> list[Message]:
        """Generate distinct neutral messages."""
        return [_make_tool_msg(f"tool output step {i}") for i in range(n)]

    def test_high_budget_no_action(self) -> None:
        ctrl = BudgetConditionedController(max_iterations=100)
        signal = ctrl.check(self._msgs(), current_iteration=0, current_effort="high")
        assert signal.corrective_message is None
        assert signal.effort_hint is None
        assert not signal.prune_warning

    def test_wrap_up_fires_below_threshold(self) -> None:
        ctrl = BudgetConditionedController(max_iterations=100)
        # iteration 92 → ratio = 8/100 = 0.08 < WRAP_UP_THRESHOLD
        signal = ctrl.check(self._msgs(), current_iteration=92, current_effort="high")
        assert signal.corrective_message is not None
        assert "BUDGET CRITICAL" in signal.corrective_message
        assert signal.prune_warning is True

    def test_budget_ratio_returned_correctly(self) -> None:
        ctrl = BudgetConditionedController(max_iterations=100)
        signal = ctrl.check(self._msgs(), current_iteration=50)
        assert signal.budget_ratio == 0.5

    def test_unlimited_max_iterations_no_signal(self) -> None:
        ctrl = BudgetConditionedController(max_iterations=-1)
        signal = ctrl.check(self._msgs(), current_iteration=999)
        assert signal.corrective_message is None
        assert signal.effort_hint is None
        assert signal.budget_ratio == 1.0

    def test_cooldown_prevents_back_to_back_injection(self) -> None:
        ctrl = BudgetConditionedController(max_iterations=100)
        # First check at iter 92 → fires
        s1 = ctrl.check(self._msgs(), current_iteration=92)
        assert s1.corrective_message is not None
        # Immediately after → cooldown suppresses
        s2 = ctrl.check(self._msgs(), current_iteration=93)
        assert s2.corrective_message is None

    def test_effort_hint_applied_at_low_budget(self) -> None:
        ctrl = BudgetConditionedController(max_iterations=100)
        # iter=80 → ratio=0.2, below DOWNGRADE_THRESHOLD=0.25
        # Use error messages to drive progress < 0 so the downgrade path fires
        error_msgs = [_make_tool_msg("ERROR: not found") for _ in range(10)]
        signal = ctrl.check(error_msgs, current_iteration=80, current_effort="high")
        assert signal.effort_hint == "medium"
