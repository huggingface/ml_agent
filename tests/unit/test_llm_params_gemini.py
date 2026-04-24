"""Unit tests for Gemini provider support in _resolve_llm_params.

Run with:
    uv run pytest tests/unit/test_llm_params_gemini.py -v

For a live smoke-test against the real Gemini API, set GEMINI_API_KEY in
your environment and run:
    uv run python tests/unit/test_llm_params_gemini.py --live
"""

import sys
import pytest
from agent.core.llm_params import (
    _resolve_llm_params,
    UnsupportedEffortError,
    _GEMINI_THINKING_BUDGETS,
)


# ---------------------------------------------------------------------------
# Parameter resolution (no network)
# ---------------------------------------------------------------------------

class TestGeminiParamResolution:
    def test_bare_model_no_effort(self):
        params = _resolve_llm_params("gemini/gemini-2.5-pro")
        assert params == {"model": "gemini/gemini-2.5-pro"}

    def test_effort_low_sets_reasoning_effort(self):
        params = _resolve_llm_params("gemini/gemini-2.5-pro", reasoning_effort="low")
        assert params["reasoning_effort"] == "low"
        assert "thinking" not in params

    def test_effort_medium_sets_reasoning_effort(self):
        params = _resolve_llm_params("gemini/gemini-2.5-pro", reasoning_effort="medium")
        assert params["reasoning_effort"] == "medium"

    def test_effort_high_sets_reasoning_effort(self):
        params = _resolve_llm_params("gemini/gemini-2.5-pro", reasoning_effort="high")
        assert params["reasoning_effort"] == "high"

    def test_effort_minimal_normalises_to_low(self):
        params = _resolve_llm_params("gemini/gemini-2.5-pro", reasoning_effort="minimal")
        assert params["reasoning_effort"] == "low"

    def test_effort_max_strict_raises(self):
        with pytest.raises(UnsupportedEffortError):
            _resolve_llm_params(
                "gemini/gemini-2.5-pro", reasoning_effort="max", strict=True
            )

    def test_effort_xhigh_strict_raises(self):
        with pytest.raises(UnsupportedEffortError):
            _resolve_llm_params(
                "gemini/gemini-2.5-pro", reasoning_effort="xhigh", strict=True
            )

    def test_effort_max_non_strict_omits_reasoning_effort(self):
        # Non-strict: invalid effort is silently dropped, no reasoning_effort key.
        params = _resolve_llm_params(
            "gemini/gemini-2.5-pro", reasoning_effort="max", strict=False
        )
        assert "reasoning_effort" not in params
        assert "thinking" not in params

    def test_no_hf_keys_in_params(self):
        params = _resolve_llm_params("gemini/gemini-2.5-pro", reasoning_effort="high")
        assert "api_base" not in params
        assert "extra_headers" not in params
        assert "extra_body" not in params

    def test_gemini_flash_model(self):
        params = _resolve_llm_params("gemini/gemini-2.5-flash", reasoning_effort="medium")
        assert params["model"] == "gemini/gemini-2.5-flash"
        assert params["reasoning_effort"] == "medium"

    def test_thinking_budgets_are_ordered(self):
        assert (
            _GEMINI_THINKING_BUDGETS["low"]
            < _GEMINI_THINKING_BUDGETS["medium"]
            < _GEMINI_THINKING_BUDGETS["high"]
        )


# ---------------------------------------------------------------------------
# Sanity: other providers are not affected
# ---------------------------------------------------------------------------

class TestOtherProvidersUnchanged:
    def test_anthropic_unaffected(self):
        params = _resolve_llm_params("anthropic/claude-opus-4-6", reasoning_effort="high")
        assert "thinking" in params
        assert params["thinking"] == {"type": "adaptive"}
        assert "output_config" in params

    def test_openai_unaffected(self):
        params = _resolve_llm_params("openai/gpt-4o", reasoning_effort="high")
        assert params.get("reasoning_effort") == "high"
        assert "thinking" not in params

    def test_hf_router_unaffected(self):
        params = _resolve_llm_params("moonshotai/Kimi-K2.6", reasoning_effort="high")
        assert "api_base" in params
        assert params["extra_body"] == {"reasoning_effort": "high"}


# ---------------------------------------------------------------------------
# Live smoke-test (opt-in, requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------

async def _live_smoke_test():
    import os
    import asyncio
    from litellm import acompletion

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY not set — skipping live test")
        return

    print("Running live smoke-test against gemini/gemini-2.5-flash ...")
    params = _resolve_llm_params("gemini/gemini-2.5-flash", reasoning_effort="low")
    response = await asyncio.wait_for(
        acompletion(
            messages=[{"role": "user", "content": "Reply with the single word: hello"}],
            max_tokens=16,
            **params,
        ),
        timeout=30,
    )
    text = response.choices[0].message.content.strip()
    print(f"Response: {text!r}")
    assert text, "Expected a non-empty response"
    print("Live smoke-test passed.")


if __name__ == "__main__":
    if "--live" in sys.argv:
        import asyncio
        asyncio.run(_live_smoke_test())
    else:
        # Run unit tests via pytest when executed directly
        sys.exit(pytest.main([__file__, "-v"]))
