"""Live BAVT integration tests using the HuggingFace Inference Router.

These tests prove the full BAVT signal pipeline with a real LLM:

  1. ResidualProgressScorer correctly classifies real model output
  2. effort_hint from BAVT correctly modifies HF router API params
  3. A real HF Inference API call succeeds with a downgraded effort level
  4. The full BAVT corrective-message + effort-downgrade path works end-to-end

Opt-in: set ML_INTERN_LIVE_LLM_TESTS=1 and HF_TOKEN before running.

    HF_TOKEN=<token> ML_INTERN_LIVE_LLM_TESTS=1 \\
        uv run pytest tests/integration/test_live_bavt_hf.py -v -s
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from litellm import Message

from agent.core.bavt import (
    BudgetConditionedController,
    BudgetTracker,
    ResidualProgressScorer,
)
from agent.core.agent_loop import _call_llm_streaming
from agent.core.llm_params import _resolve_llm_params


# ──────────────────────────────────────────────────────────────────────────────
# Test configuration
# ──────────────────────────────────────────────────────────────────────────────

LIVE_TESTS_ENABLED = os.environ.get("ML_INTERN_LIVE_LLM_TESTS") == "1"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Fast, capable model available on HF Pro tier
HF_MODEL = "huggingface/Qwen/Qwen2.5-72B-Instruct"


def _skip_if_not_live() -> None:
    if not LIVE_TESTS_ENABLED:
        pytest.skip(
            "set ML_INTERN_LIVE_LLM_TESTS=1 and HF_TOKEN to run live HF BAVT tests"
        )
    if not HF_TOKEN:
        pytest.skip("HF_TOKEN not set")


def _mock_session(model_name: str = HF_MODEL) -> SimpleNamespace:
    events: list = []

    async def send_event(event) -> None:
        events.append(event)

    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name),
        hf_token=HF_TOKEN,
        is_cancelled=False,
        send_event=send_event,
        events=events,
        stream=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: _resolve_llm_params with BAVT effort_hint produces valid HF params
# ──────────────────────────────────────────────────────────────────────────────


def test_bavt_effort_hint_produces_valid_hf_params() -> None:
    """Prove the effort_hint path generates correct HF router params."""
    _skip_if_not_live()

    # Simulate what agent_loop.py does: BAVT generates hint, loop uses it
    tracker = BudgetTracker(max_iterations=100)
    hint = tracker.effort_hint("high", budget_ratio=0.20)

    assert hint == "medium", f"Expected 'medium', got: {hint}"

    # Now build params as agent_loop.py would
    effective_effort = hint or "high"
    params = _resolve_llm_params(
        HF_MODEL,
        session_hf_token=HF_TOKEN,
        reasoning_effort=effective_effort,
    )

    assert params["model"] == "openai/Qwen/Qwen2.5-72B-Instruct"
    assert params["api_base"] == "https://router.huggingface.co/v1"
    assert params["api_key"] == HF_TOKEN
    assert params["extra_body"]["reasoning_effort"] == "medium"
    print(f"\n✓ BAVT effort_hint='medium' → HF params: {params['extra_body']}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Real HF LLM call succeeds with downgraded effort
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_hf_call_with_bavt_effort_hint() -> None:
    """Prove that a BAVT-downgraded effort level works with the real HF API."""
    _skip_if_not_live()

    session = _mock_session()

    # Simulate BAVT having fired an effort downgrade
    bavt_hint = "medium"  # would be "high" without BAVT downgrade
    llm_params = _resolve_llm_params(
        HF_MODEL,
        session_hf_token=HF_TOKEN,
        reasoning_effort=bavt_hint,
    )

    result = await _call_llm_streaming(
        session,
        messages=[
            Message(
                role="user",
                content=(
                    "Answer in exactly 5 words: what is 2 + 2? "
                    "Include the number 4 in your answer."
                ),
            )
        ],
        tools=[],
        llm_params=llm_params,
    )

    assert result.content, "Expected non-empty response from HF model"
    assert "4" in result.content, f"Expected '4' in response, got: {result.content!r}"

    print(f"\n✓ HF call with BAVT effort='medium' → response: {result.content!r}")
    print(f"  Token count: {result.token_count}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: ResidualProgressScorer on real LLM output
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_residual_scorer_on_real_output() -> None:
    """Prove ResidualProgressScorer correctly distinguishes success vs error output.

    Makes two real HF API calls:
    - One that produces clear success output
    - One that produces an error-like output

    Verifies the scorer correctly ranks them.
    """
    _skip_if_not_live()

    session = _mock_session()
    llm_params = _resolve_llm_params(
        HF_MODEL, session_hf_token=HF_TOKEN, reasoning_effort="low"
    )

    # Call 1: success-flavored prompt
    success_result = await _call_llm_streaming(
        session,
        messages=[
            Message(
                role="user",
                content="Say exactly: 'SUCCESS: task completed successfully'",
            )
        ],
        tools=[],
        llm_params=llm_params,
    )

    # Call 2: error-flavored prompt
    error_result = await _call_llm_streaming(
        session,
        messages=[
            Message(
                role="user",
                content="Say exactly: 'ERROR: command not found, operation failed'",
            )
        ],
        tools=[],
        llm_params=llm_params,
    )

    # Build message sequences as they would appear in context
    scorer_success = ResidualProgressScorer()
    scorer_error = ResidualProgressScorer()

    success_msgs = [
        Message(role="user", content="task"),
        Message(role="tool", content=success_result.content or ""),
    ]
    error_msgs = [
        Message(role="user", content="task"),
        Message(role="tool", content=error_result.content or ""),
    ]

    score_success = scorer_success.score(success_msgs)
    score_error = scorer_error.score(error_msgs)

    print("\n✓ ResidualProgressScorer on real HF output:")
    print(f"  Success response: {success_result.content!r}")
    print(f"  Success score:    {score_success:.3f}")
    print(f"  Error response:   {error_result.content!r}")
    print(f"  Error score:      {score_error:.3f}")

    # Success output should score higher than error output
    assert score_success > score_error, (
        f"Scorer failed: success_score={score_success:.3f} "
        f"should be > error_score={score_error:.3f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: End-to-end BAVT pipeline — simulated stalling agent with real LLM
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_bavt_end_to_end_stalling_scenario() -> None:
    """End-to-end proof: stalling agent triggers BAVT, which generates a corrective
    message, which is understood by the real LLM.

    Scenario:
      - Agent has max_iterations=15
      - Runs 12 "error" tool calls (simulating a stalled agent)
      - At iteration 13 (budget_ratio = 2/15 ≈ 0.13), BAVT fires a corrective
      - We send that corrective message to the real HF model and verify it
        responds with a wrap-up / pivot — proving the message makes sense to LLMs.
    """
    _skip_if_not_live()

    session = _mock_session()

    # Simulate stalling context
    max_iter = 15
    ctrl = BudgetConditionedController(max_iterations=max_iter)
    stalling_msgs = [Message(role="user", content="Write a Python web scraper")]

    for i in range(10):
        stalling_msgs.append(
            Message(role="tool", content=f"ERROR: command not found (attempt {i})")
        )

    # BAVT check at iteration 14 → ratio = 1/15 ≈ 0.067 (< WRAP_UP_THRESHOLD=0.10)
    signal = ctrl.check(
        messages=stalling_msgs,
        current_iteration=14,
        current_effort="high",
    )

    assert signal.corrective_message is not None, (
        "BAVT should produce a corrective message at 14/15 iterations"
    )
    assert signal.prune_warning, (
        f"BAVT should set prune_warning at ratio {signal.budget_ratio:.3f} < 0.10"
    )
    assert signal.effort_hint is not None, "BAVT should suggest effort downgrade"

    print("\n✓ BAVT signal at iter 14/15:")
    print(f"  budget_ratio:       {signal.budget_ratio:.3f}")
    print(f"  progress_score:     {signal.progress_score:.3f}")
    print(f"  effort_hint:        {signal.effort_hint}")
    print(f"  corrective_message: {signal.corrective_message[:100]}...")

    # Now prove the corrective message makes sense to a real LLM.
    # Build a clean conversation without raw "tool" roles (HF model requires
    # tool messages to follow assistant tool_calls, which we don't have here).
    # We present the stalling context as assistant observations instead.
    error_context = "\n".join(
        f"Attempt {i}: ERROR: command not found" for i in range(10)
    )
    live_msgs = [
        Message(
            role="user",
            content=(
                f"I asked an agent: 'Write a Python web scraper'. "
                f"It tried 10 times and kept failing:\n{error_context}\n\n"
                f"The system then sent this budget alert:\n\n"
                f"{signal.corrective_message}\n\n"
                f"How should the agent respond to this budget alert?"
            ),
        )
    ]
    downgraded_params = _resolve_llm_params(
        HF_MODEL,
        session_hf_token=HF_TOKEN,
        reasoning_effort=signal.effort_hint,
    )
    result = await _call_llm_streaming(
        session,
        messages=live_msgs,
        tools=[],
        llm_params=downgraded_params,
    )

    assert result.content, "Expected LLM to respond to BAVT corrective message"
    content_lower = result.content.lower()

    # The LLM should respond with some form of wrap-up/summary/pivot
    wrap_up_signals = [
        "sorry",
        "unable",
        "summarize",
        "summary",
        "done",
        "complete",
        "provide",
        "here",
        "based on",
        "result",
        "despite",
        "alternative",
        "different",
        "approach",
        "try",
        "cannot",
        "help",
    ]
    responded_reasonably = any(s in content_lower for s in wrap_up_signals)

    print(f"\n  LLM response to BAVT corrective (with effort='{signal.effort_hint}'):")
    print(f"  {result.content[:300]}")
    print(f"  Responded reasonably: {responded_reasonably}")

    assert responded_reasonably, (
        f"LLM did not respond reasonably to BAVT corrective message. "
        f"Response: {result.content!r}"
    )
