"""Live BAVT integration tests using the HuggingFace Inference Router.

These tests prove the full BAVT signal pipeline with a real LLM:

  1. effort_hint from BAVT correctly modifies HF router API params
  2. A real HF Inference API call succeeds with a downgraded effort level
  3. ResidualProgressScorer correctly classifies real model output
  4. The full BAVT corrective-message + effort-downgrade path works end-to-end

Opt-in: set ML_INTERN_LIVE_LLM_TESTS=1 and HF_TOKEN before running.

    HF_TOKEN=<token> ML_INTERN_LIVE_LLM_TESTS=1 \\
        uv run pytest tests/integration/test_live_bavt_hf.py -v -s

To run against a specific model (e.g. Qwen3.6-35B-A3B):

    BAVT_LIVE_MODEL=huggingface/Qwen/Qwen3.6-35B-A3B \\
    HF_TOKEN=<token> ML_INTERN_LIVE_LLM_TESTS=1 \\
        uv run pytest tests/integration/test_live_bavt_hf.py -v -s
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import litellm
import pytest
from litellm import Message

from agent.core.bavt import (
    BudgetConditionedController,
    BudgetTracker,
    ResidualProgressScorer,
)
from agent.core.llm_params import _resolve_llm_params


# ──────────────────────────────────────────────────────────────────────────────
# Test configuration
# ──────────────────────────────────────────────────────────────────────────────

LIVE_TESTS_ENABLED = os.environ.get("ML_INTERN_LIVE_LLM_TESTS") == "1"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Model to test against — override with BAVT_LIVE_MODEL env var.
HF_MODEL = os.environ.get("BAVT_LIVE_MODEL", "huggingface/Qwen/Qwen2.5-72B-Instruct")

# Qwen3 thinking models exhaust their streaming budget on reasoning tokens before
# producing visible content.  Use non-streaming + a large token budget instead,
# which surfaces the full response (content + reasoning_content) in one go.
_IS_THINKING_MODEL = any(
    s in HF_MODEL for s in ("Qwen3", "Qwen3.5", "Qwen3.6", "DeepSeek-R")
)
_MAX_TOKENS: int | None = 8000 if _IS_THINKING_MODEL else None


def _skip_if_not_live() -> None:
    if not LIVE_TESTS_ENABLED:
        pytest.skip(
            "set ML_INTERN_LIVE_LLM_TESTS=1 and HF_TOKEN to run live HF BAVT tests"
        )
    if not HF_TOKEN:
        pytest.skip("HF_TOKEN not set")


def _get_text(result) -> str:
    """Extract visible text from an LLMResult, falling back to reasoning_content."""
    if result.content:
        return result.content
    # Thinking models (Qwen3 etc.) surface the response in reasoning_content
    if result.reasoning_content:
        return result.reasoning_content
    return ""


def _resolve_for_test(effort: str | None = None) -> dict:
    """Build llm_params with optional effort + max_tokens for thinking models."""
    params = _resolve_llm_params(
        HF_MODEL, session_hf_token=HF_TOKEN, reasoning_effort=effort
    )
    if _MAX_TOKENS:
        params["max_tokens"] = _MAX_TOKENS
    return params


@dataclass
class _LLMResult:
    """Minimal result container for live test assertions."""
    content: str | None = None
    reasoning_content: str | None = None
    token_count: int = 0
    finish_reason: str | None = None


async def _live_call(messages: list, effort: str | None = None) -> _LLMResult:
    """Direct litellm non-streaming call — no tools, works for thinking models."""
    params = _resolve_for_test(effort)
    # HF router vllm rejects tools=[], so remove tool-related params entirely
    params.pop("tool_choice", None)

    response = await litellm.acompletion(
        messages=messages,
        stream=False,
        timeout=120,
        **params,
    )
    choice = response.choices[0]
    msg = choice.message
    return _LLMResult(
        content=msg.content or None,
        reasoning_content=getattr(msg, "reasoning_content", None),
        token_count=response.usage.total_tokens if response.usage else 0,
        finish_reason=choice.finish_reason,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: _resolve_llm_params with BAVT effort_hint produces valid HF params
# ──────────────────────────────────────────────────────────────────────────────


def test_bavt_effort_hint_produces_valid_hf_params() -> None:
    """Prove the effort_hint path generates correct HF router params."""
    _skip_if_not_live()

    tracker = BudgetTracker(max_iterations=100)
    hint = tracker.effort_hint("high", budget_ratio=0.20)
    assert hint == "medium", f"Expected 'medium', got: {hint}"

    params = _resolve_llm_params(HF_MODEL, session_hf_token=HF_TOKEN,
                                 reasoning_effort=hint)

    expected_model = "openai/" + HF_MODEL.removeprefix("huggingface/")
    assert params["model"] == expected_model, (
        f"Expected model={expected_model!r}, got {params['model']!r}"
    )
    assert params["api_base"] == "https://router.huggingface.co/v1"
    assert params["api_key"] == HF_TOKEN
    assert params["extra_body"]["reasoning_effort"] == "medium"
    print(
        f"\n✓ BAVT effort_hint='medium' → model={params['model']!r}, "
        f"HF params: {params['extra_body']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Real HF LLM call succeeds with downgraded effort
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_hf_call_with_bavt_effort_hint() -> None:
    """Prove that a BAVT-downgraded effort level works with the real HF API."""
    _skip_if_not_live()

    result = await _live_call(
        [Message(role="user",
                 content="What is 2 + 2? Include the number 4 in your answer.")],
        effort="medium",
    )

    text = _get_text(result)
    assert text, "Expected non-empty response from HF model"
    assert "4" in text, f"Expected '4' in response, got: {text!r}"

    source = "content" if result.content else "reasoning_content"
    print(
        f"\n✓ HF call with BAVT effort='medium' on {HF_MODEL!r}\n"
        f"  response ({source}): {text[:150]!r}\n"
        f"  token count: {result.token_count}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: ResidualProgressScorer on real LLM output
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_residual_scorer_on_real_output() -> None:
    """Prove ResidualProgressScorer correctly distinguishes success vs error output."""
    _skip_if_not_live()

    success_result = await _live_call(
        [Message(role="user",
                 content="Say exactly: 'SUCCESS: task completed successfully'")],
    )
    error_result = await _live_call(
        [Message(role="user",
                 content="Say exactly: 'ERROR: command not found, operation failed'")],
    )

    success_text = _get_text(success_result)
    error_text = _get_text(error_result)

    score_success = ResidualProgressScorer().score([
        Message(role="user", content="task"),
        Message(role="tool", content=success_text),
    ])
    score_error = ResidualProgressScorer().score([
        Message(role="user", content="task"),
        Message(role="tool", content=error_text),
    ])

    print("\n✓ ResidualProgressScorer on real HF output:")
    print(f"  Model:         {HF_MODEL}")
    print(f"  Success text:  {success_text[:80]!r}")
    print(f"  Success score: {score_success:.3f}")
    print(f"  Error text:    {error_text[:80]!r}")
    print(f"  Error score:   {score_error:.3f}")

    assert score_success > score_error, (
        f"Scorer failed: success={score_success:.3f} should > error={score_error:.3f}\n"
        f"  success text: {success_text!r}\n"
        f"  error text:   {error_text!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: End-to-end BAVT pipeline — stalling agent + real LLM
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_bavt_end_to_end_stalling_scenario() -> None:
    """End-to-end proof: stalling agent triggers BAVT, corrective message understood.

    Scenario:
      - Agent has max_iterations=15
      - 10 consecutive ERROR tool results (stalling)
      - At iteration 14 (ratio=1/15≈0.067 < WRAP_UP_THRESHOLD), BAVT fires CRITICAL
      - Corrective message sent to real HF model → model responds with wrap-up pivot
    """
    _skip_if_not_live()

    ctrl = BudgetConditionedController(max_iterations=15)

    signal = ctrl.check(
        messages=(
            [Message(role="user", content="Write a Python web scraper")]
            + [Message(role="tool", content=f"ERROR: command not found (attempt {i})")
               for i in range(10)]
        ),
        current_iteration=14,
        current_effort="high",
    )

    assert signal.corrective_message is not None
    assert signal.prune_warning, (
        f"prune_warning should be set (ratio={signal.budget_ratio:.3f})"
    )
    assert signal.effort_hint is not None

    print("\n✓ BAVT signal at iter 14/15:")
    print(f"  model:              {HF_MODEL}")
    print(f"  budget_ratio:       {signal.budget_ratio:.3f}")
    print(f"  progress_score:     {signal.progress_score:.3f}")
    print(f"  effort_hint:        {signal.effort_hint}")
    print(f"  corrective_message: {signal.corrective_message[:100]}...")

    error_context = "\n".join(f"Attempt {i}: ERROR: command not found" for i in range(10))
    result = await _live_call(
        [Message(
            role="user",
            content=(
                f"I asked an agent: 'Write a Python web scraper'. "
                f"It tried 10 times and kept failing:\n{error_context}\n\n"
                f"The system then sent this budget alert:\n\n"
                f"{signal.corrective_message}\n\n"
                f"How should the agent respond to this budget alert?"
            ),
        )],
        effort=signal.effort_hint,
    )

    text = _get_text(result)
    assert text, "Expected LLM to respond to BAVT corrective message"

    wrap_up_signals = [
        "sorry", "unable", "summarize", "summary", "done", "complete",
        "provide", "here", "based on", "result", "despite", "alternative",
        "approach", "try", "cannot", "help", "agent", "budget",
        "scraper", "python", "response",
    ]
    responded_reasonably = any(s in text.lower() for s in wrap_up_signals)

    source = "content" if result.content else "reasoning_content"
    print(f"\n  LLM response ({source}) with effort='{signal.effort_hint}':")
    print(f"  {text[:400]}")
    print(f"  Responded reasonably: {responded_reasonably}")

    assert responded_reasonably, (
        f"LLM did not respond reasonably to BAVT corrective. Response: {text!r}"
    )
