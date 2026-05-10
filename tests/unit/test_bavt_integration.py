"""Integration tests for BAVT — simulating the full agent_loop.py mechanics.

These tests exercise the BAVT system at the integration boundary, replicating
exactly what agent_loop.py does each iteration, without needing a real Session
or LLM call. This lets us prove the wiring is correct before the live API tests.

Run with: uv run pytest tests/unit/test_bavt_integration.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litellm import Message

from agent.core.bavt import (
    NUDGE_THRESHOLD,
    BudgetConditionedController,
    BudgetTracker,
    ResidualProgressScorer,
)
from agent.core.doom_loop import check_for_doom_loop
from agent.core.llm_params import _resolve_llm_params


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — replicate the exact message shapes agent_loop.py produces
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _TC:
    id: str
    function: _Fn


@dataclass
class _Msg:
    role: str
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None


def _assistant_tool_call(
    tool_name: str, args: str = "{}", call_id: str = "tc1"
) -> _Msg:
    return _Msg(role="assistant", tool_calls=[_TC(call_id, _Fn(tool_name, args))])


def _tool_result(content: str, call_id: str = "tc1", tool_name: str = "bash") -> _Msg:
    return _Msg(role="tool", content=content, tool_call_id=call_id, name=tool_name)


def _user(content: str) -> _Msg:
    return _Msg(role="user", content=content)


# ──────────────────────────────────────────────────────────────────────────────
# Agent-loop simulation — mirrors the critical path in agent_loop.py
# ──────────────────────────────────────────────────────────────────────────────


class SimulatedAgentLoop:
    """Miniature agent loop that exercises the BAVT integration wiring.

    Mirrors exactly what agent_loop.py does:
      1. Compute budget_ratio
      2. check_for_doom_loop(messages, budget_ratio=budget_ratio)
      3. budget_controller.check(messages, iteration, current_effort)
      4. If corrective_message → append to messages (as "user" role)
      5. If effort_hint → update bavt_effort_hint
    Records every event for assertion.
    """

    def __init__(self, max_iterations: int, initial_effort: str = "high") -> None:
        self.max_iterations = max_iterations
        self.messages: list[_Msg] = [_user("Initial task")]
        self.budget_controller = BudgetConditionedController(max_iterations)
        self.bavt_effort_hint: str | None = None
        self.initial_effort = initial_effort
        self.events: list[dict[str, Any]] = []

    def _current_effort(self) -> str:
        return self.bavt_effort_hint or self.initial_effort

    def step(self, tool_name: str, tool_result_content: str, args: str = "{}") -> None:
        """Simulate one agent iteration with the given tool call + result."""
        iteration = sum(1 for e in self.events if e["type"] == "step")

        # ── Mirror agent_loop.py: doom + BAVT check ──
        budget_ratio = BudgetTracker(self.max_iterations).ratio(iteration)

        doom = check_for_doom_loop(self.messages, budget_ratio=budget_ratio)
        if doom:
            self.messages.append(_user(doom))
            self.events.append({"type": "doom", "iteration": iteration, "msg": doom})

        signal = self.budget_controller.check(
            messages=self.messages,  # type: ignore[arg-type]
            current_iteration=iteration,
            current_effort=self._current_effort(),
        )
        if signal.corrective_message:
            self.messages.append(_user(signal.corrective_message))
            self.events.append(
                {
                    "type": "bavt",
                    "iteration": iteration,
                    "budget_ratio": signal.budget_ratio,
                    "msg": signal.corrective_message,
                    "effort_hint": signal.effort_hint,
                }
            )
        if signal.effort_hint:
            self.bavt_effort_hint = signal.effort_hint

        # ── Simulate LLM call + tool execution ──
        call_id = f"tc_{iteration}"
        self.messages.append(_assistant_tool_call(tool_name, args, call_id))
        self.messages.append(_tool_result(tool_result_content, call_id, tool_name))
        self.events.append({"type": "step", "iteration": iteration, "tool": tool_name})

    def bavt_events(self) -> list[dict]:
        return [e for e in self.events if e["type"] == "bavt"]

    def doom_events(self) -> list[dict]:
        return [e for e in self.events if e["type"] == "doom"]


# ──────────────────────────────────────────────────────────────────────────────
# Test: healthy run — no BAVT signals when budget is ample and progress is good
# ──────────────────────────────────────────────────────────────────────────────


class TestBAVTHealthyRun:
    def test_no_signals_on_healthy_run(self) -> None:
        """When budget is fresh and tool calls succeed, BAVT stays silent."""
        loop = SimulatedAgentLoop(max_iterations=100)
        for i in range(10):
            loop.step("bash", f"success: step {i} complete")
        assert loop.bavt_events() == [], "BAVT should not fire on a healthy run"

    def test_no_signals_unlimited_budget(self) -> None:
        """max_iterations=-1 → BAVT is a strict no-op."""
        loop = SimulatedAgentLoop(max_iterations=-1)
        # Run 200 identical (stalling) iterations — should never fire
        for _ in range(50):
            loop.step("bash", "ERROR: same error")
        assert loop.bavt_events() == []


# ──────────────────────────────────────────────────────────────────────────────
# Test: budget thresholds fire at the right iteration
# ──────────────────────────────────────────────────────────────────────────────


class TestBAVTThresholds:
    def _stalling_loop(self, max_iter: int, n_steps: int) -> SimulatedAgentLoop:
        """Run n_steps of stalling (error) tool calls."""
        loop = SimulatedAgentLoop(max_iterations=max_iter)
        for i in range(n_steps):
            loop.step("bash", "ERROR: command not found")
        return loop

    def test_nudge_fires_past_50pct(self) -> None:
        """BAVT nudge fires when >50% budget is consumed and progress is bad."""
        max_iter = 20
        # 11 steps → iteration 10 → ratio = (20-10)/20 = 0.50, boundary
        # 12 steps → iteration 11 → ratio = 9/20 = 0.45 < NUDGE_THRESHOLD
        loop = self._stalling_loop(max_iter, 13)
        bavt = loop.bavt_events()
        assert any(e["budget_ratio"] <= NUDGE_THRESHOLD for e in bavt), (
            f"Expected a BAVT nudge below ratio {NUDGE_THRESHOLD}, got: {bavt}"
        )

    def test_downgrade_fires_past_75pct(self) -> None:
        """Effort downgrade fires when >75% budget is consumed."""
        max_iter = 20
        # budget_ratio < 0.25 → iter > 15 → step 16+
        loop = self._stalling_loop(max_iter, 18)
        bavt = loop.bavt_events()
        downgrade_events = [e for e in bavt if e.get("effort_hint") is not None]
        assert len(downgrade_events) > 0, (
            "Expected at least one effort downgrade event after 75% budget consumed"
        )

    def test_wrap_up_fires_at_90pct(self) -> None:
        """WRAP_UP message fires when <10% budget remains.

        Uses max_iter=50 so nudge (~iter 26) and downgrade (~iter 38) are
        far enough apart that the 5-iteration cooldown doesn't block the
        wrap-up at iter 46 (ratio = 4/50 = 0.08 < WRAP_UP_THRESHOLD).
        """
        max_iter = 50
        loop = self._stalling_loop(max_iter, 48)
        bavt = loop.bavt_events()
        wrap_up = [e for e in bavt if "CRITICAL" in e["msg"]]
        assert wrap_up, (
            f"Expected a BUDGET CRITICAL message at <10% remaining budget. "
            f"All BAVT events: {[(e['iteration'], e['budget_ratio']) for e in bavt]}"
        )

    def test_effort_hint_accumulates(self) -> None:
        """After a downgrade event, bavt_effort_hint is set on the loop."""
        loop = self._stalling_loop(max_iter=20, n_steps=18)
        assert loop.bavt_effort_hint is not None, (
            "Expected bavt_effort_hint to be set after budget depletion"
        )
        # Should have stepped down from 'high'
        assert loop.bavt_effort_hint in ("medium", "low", "minimal"), (
            f"Unexpected effort hint: {loop.bavt_effort_hint}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test: doom loop integrates with budget_ratio
# ──────────────────────────────────────────────────────────────────────────────


class TestBAVTDoomLoopIntegration:
    def _identical_calls(self, n: int) -> list[_Msg]:
        msgs: list[_Msg] = [_user("task")]
        for i in range(n):
            msgs.append(_assistant_tool_call("bash", '{"cmd":"ls"}', f"tc_{i}"))
            msgs.append(_tool_result("same output", f"tc_{i}", "bash"))
        return msgs

    def test_doom_fires_at_3_with_full_budget(self) -> None:
        msgs = self._identical_calls(3)
        result = check_for_doom_loop(msgs, budget_ratio=1.0)
        assert result is not None, "Doom loop should fire at 3 identical calls"

    def test_doom_fires_at_2_with_low_budget(self) -> None:
        msgs = self._identical_calls(2)
        # At full budget: 2 identical calls should NOT fire
        assert check_for_doom_loop(msgs, budget_ratio=1.0) is None
        # At 20% budget: 2 identical calls SHOULD fire (threshold tightens to 2)
        assert check_for_doom_loop(msgs, budget_ratio=0.20) is not None

    def test_doom_and_bavt_both_inject_when_stalling(self) -> None:
        """Both doom and BAVT signals inject at extreme budget depletion."""
        loop = SimulatedAgentLoop(max_iterations=20, initial_effort="high")
        # Run enough identical calls to trigger both doom detection and BAVT
        for _ in range(20):
            loop.step("bash", "ERROR: same error", args='{"cmd":"ls"}')

        assert loop.doom_events(), "Doom loop should have fired"
        assert loop.bavt_events(), "BAVT should have fired"


# ──────────────────────────────────────────────────────────────────────────────
# Test: effort hint flows through to _resolve_llm_params
# ──────────────────────────────────────────────────────────────────────────────


class TestEffortHintFlowThrough:
    """Prove that a BAVT effort_hint correctly modifies the LLM params dict."""

    def test_no_hint_uses_original_effort(self) -> None:
        params = _resolve_llm_params(
            "huggingface/Qwen/Qwen2.5-72B-Instruct",
            session_hf_token="test-token",
            reasoning_effort="high",
        )
        assert params.get("extra_body", {}).get("reasoning_effort") == "high"

    def test_bavt_hint_medium_overrides_high(self) -> None:
        """Simulate what agent_loop.py does: use effort_hint OR original."""
        original_effort = "high"
        bavt_hint = "medium"  # BAVT says to downgrade

        effective = bavt_hint or original_effort  # this is the agent_loop.py pattern
        params = _resolve_llm_params(
            "huggingface/Qwen/Qwen2.5-72B-Instruct",
            session_hf_token="test-token",
            reasoning_effort=effective,
        )
        assert params["extra_body"]["reasoning_effort"] == "medium"

    def test_bavt_hint_low_overrides_high(self) -> None:
        effective = "low"
        params = _resolve_llm_params(
            "huggingface/Qwen/Qwen2.5-72B-Instruct",
            session_hf_token="test-token",
            reasoning_effort=effective,
        )
        assert params["extra_body"]["reasoning_effort"] == "low"

    def test_effort_downgrade_chain(self) -> None:
        """Simulate multiple BAVT downgrade events walking down the effort ladder."""
        tracker = BudgetTracker(max_iterations=100)
        effort = "high"

        # Each call simulates a BAVT check at decreasing budget
        for ratio in [0.22, 0.15, 0.08]:
            hint = tracker.effort_hint(effort, budget_ratio=ratio)
            if hint:
                effort = hint

        # After 3 downgrade opportunities starting from 'high', we should
        # have walked down at least one level
        assert effort in ("medium", "low", "minimal"), (
            f"Expected effort to have been downgraded from 'high', got: {effort}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test: ResidualProgressScorer — realistic message sequences
# ──────────────────────────────────────────────────────────────────────────────


class TestResidualScorerRealism:
    def _make_msgs(self, tool_outputs: list[str]) -> list[Message]:
        msgs = [Message(role="user", content="task")]
        for i, out in enumerate(tool_outputs):
            msgs.append(Message(role="assistant", content=f"step {i}"))
            msgs.append(Message(role="tool", content=out))
        return msgs

    def test_successful_trajectory_scores_positive(self) -> None:
        scorer = ResidualProgressScorer()
        msgs = self._make_msgs(
            [
                "success: file created",
                "success: tests pass",
                "updated README",
                "completed: all done",
            ]
        )
        assert scorer.score(msgs) > 0.0

    def test_error_spiral_scores_negative(self) -> None:
        scorer = ResidualProgressScorer()
        msgs = self._make_msgs(
            [
                "ERROR: command not found",
                "ERROR: no such file",
                "Traceback: AttributeError",
                "ERROR: permission denied",
            ]
        )
        assert scorer.score(msgs) < 0.0

    def test_fresh_diverse_content_outscores_stale(self) -> None:
        scorer_a = ResidualProgressScorer()
        scorer_b = ResidualProgressScorer()

        fresh = self._make_msgs([f"unique result {i} with new data" for i in range(6)])
        stale = self._make_msgs(["same cached response"] * 6)

        score_fresh = scorer_a.score(fresh)
        score_stale = scorer_b.score(stale)
        assert score_fresh > score_stale, (
            f"Fresh content (score={score_fresh:.2f}) should outscore "
            f"stale content (score={score_stale:.2f})"
        )

    def test_cooldown_prevents_injection_spam(self) -> None:
        """5-iteration cooldown must suppress back-to-back corrections."""
        ctrl = BudgetConditionedController(max_iterations=20)
        error_msgs = [Message(role="tool", content="ERROR: fail")] * 10

        injections = 0
        for i in range(16, 20):  # All in wrap-up zone
            s = ctrl.check(error_msgs, current_iteration=i, current_effort="high")
            if s.corrective_message:
                injections += 1

        assert injections <= 1, (
            f"Cooldown failed: {injections} injections in 4 consecutive iterations"
        )
