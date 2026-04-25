"""Tests for the OpenCode Zen free-provider integration.

Covers three modules added / modified in the multi-provider PR:

* ``agent.core.llm_params._resolve_llm_params`` — OpenCode branch
* ``agent.core.rate_limiter``                   — token-bucket rate cap
* ``agent.core.provider_select``                — credential cascade
* ``agent.core.model_switcher.is_valid_model_id`` — accepts new prefixes

All async tests use ``asyncio.run`` rather than pytest-asyncio so they
work with ``pytest`` alone (matching the rest of the test suite).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_env(monkeypatch, *keys: str) -> None:
    """Remove the given env-var keys if present."""
    for k in keys:
        monkeypatch.delenv(k, raising=False)


_PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY",
    "GITHUB_COPILOT_TOKEN",
    "GH_COPILOT_TOKEN",
    "OPENCODE_API_KEY",
    "ML_INTERN_MODEL",
)


# ---------------------------------------------------------------------------
# llm_params — OpenCode branch
# ---------------------------------------------------------------------------

class TestResolveOpenCode:
    """_resolve_llm_params dispatches ``opencode/`` models correctly."""

    def test_api_base_and_model_rewrite(self, monkeypatch):
        """Bare ``opencode/<model>`` → OpenAI-compatible Zen endpoint."""
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/qwen3.6-plus-free")
        assert params["api_base"] == "https://opencode.ai/zen/v1"
        assert params["model"] == "openai/qwen3.6-plus-free"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-testkey")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/qwen3.6-plus-free")
        assert params["api_key"] == "sk-testkey"
        # When a real key is provided, we must NOT blank the Authorization
        # header — the upstream needs the Bearer token to validate the key.
        assert "extra_headers" not in params or \
            params["extra_headers"].get("Authorization") != ""

    def test_api_key_absent_when_env_unset(self, monkeypatch):
        """When ``OPENCODE_API_KEY`` isn't set we use the anonymous free tier:
        a placeholder ``api_key`` (LiteLLM's OpenAI adapter requires one) plus
        a blanked Authorization header so the upstream sees no credentials."""
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/minimax-m2.5-free")
        # api_key must be present (LiteLLM/openai SDK requires it) but a
        # non-real placeholder. The real auth bypass is via extra_headers.
        assert params.get("api_key") == "anonymous"
        assert params.get("extra_headers", {}).get("Authorization") == ""

    def test_effort_medium_forwarded_via_extra_body(self, monkeypatch):
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/minimax-m2.5-free", reasoning_effort="medium")
        assert params.get("extra_body") == {"reasoning_effort": "medium"}

    def test_effort_high_forwarded(self, monkeypatch):
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/minimax-m2.5-free", reasoning_effort="high")
        assert params.get("extra_body") == {"reasoning_effort": "high"}

    def test_effort_low_forwarded(self, monkeypatch):
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/minimax-m2.5-free", reasoning_effort="low")
        assert params.get("extra_body") == {"reasoning_effort": "low"}

    def test_minimal_normalizes_to_low(self, monkeypatch):
        """``minimal`` effort should be sent as ``low`` (OpenCode doesn't have minimal)."""
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/qwen3.6-plus-free", reasoning_effort="minimal")
        assert params.get("extra_body") == {"reasoning_effort": "low"}

    def test_max_effort_dropped_silently_in_lenient_mode(self, monkeypatch):
        """``max`` isn't in OpenCode's effort set; silently omit extra_body."""
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/qwen3.6-plus-free", reasoning_effort="max")
        assert "extra_body" not in params

    def test_max_effort_raises_in_strict_mode(self, monkeypatch):
        """Strict mode raises ``UnsupportedEffortError`` so the probe cascade can skip."""
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params, UnsupportedEffortError
        try:
            _resolve_llm_params("opencode/qwen3.6-plus-free", reasoning_effort="max", strict=True)
            assert False, "expected UnsupportedEffortError"
        except UnsupportedEffortError as exc:
            assert "OpenCode" in str(exc)

    def test_xhigh_effort_raises_in_strict_mode(self, monkeypatch):
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params, UnsupportedEffortError
        try:
            _resolve_llm_params("opencode/qwen3.6-plus-free", reasoning_effort="xhigh", strict=True)
            assert False, "expected UnsupportedEffortError"
        except UnsupportedEffortError as exc:
            assert "OpenCode" in str(exc)

    def test_no_effort_produces_no_extra_body(self, monkeypatch):
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        params = _resolve_llm_params("opencode/qwen3.6-plus-free")
        assert "extra_body" not in params

    def test_different_models_same_endpoint(self, monkeypatch):
        """Verify the model suffix is stripped and remapped regardless of the id."""
        _clear_env(monkeypatch, "OPENCODE_API_KEY")
        from agent.core.llm_params import _resolve_llm_params
        for model_suffix in ("qwen3.6-plus-free", "minimax-m2.5-free", "nemotron-3-super-free"):
            params = _resolve_llm_params(f"opencode/{model_suffix}")
            assert params["model"] == f"openai/{model_suffix}"
            assert params["api_base"] == "https://opencode.ai/zen/v1"


# ---------------------------------------------------------------------------
# rate_limiter
# ---------------------------------------------------------------------------

class TestProviderKey:
    """_provider_key correctly identifies capped vs. uncapped providers."""

    def test_opencode_model_returns_key(self):
        from agent.core.rate_limiter import _provider_key
        assert _provider_key("opencode/qwen3.6-plus-free") == "opencode"
        assert _provider_key("opencode/minimax-m2.5-free") == "opencode"

    def test_other_providers_return_none(self):
        from agent.core.rate_limiter import _provider_key
        for model in (
            "anthropic/claude-opus-4-7",
            "bedrock/us.anthropic.claude-opus-4-7",
            "openai/gpt-5",
            "copilot/gpt-5",
            "github_copilot/claude-sonnet-4.5",
            "MiniMaxAI/MiniMax-M2.7",
            "huggingface/meta-llama/llama-3",
        ):
            assert _provider_key(model) is None, f"Expected None for {model}"

    def test_empty_string_returns_none(self):
        from agent.core.rate_limiter import _provider_key
        assert _provider_key("") is None

    def test_unknown_prefix_returns_none(self):
        from agent.core.rate_limiter import _provider_key
        assert _provider_key("newprovider/some-model") is None


class TestTokenBucket:
    """Token bucket refills and throttles at the configured rate."""

    def test_burst_up_to_capacity_is_instant(self):
        """All tokens within the initial capacity should be granted without sleeping."""
        from agent.core.rate_limiter import _Limit, _TokenBucket

        async def body():
            bucket = _TokenBucket(_Limit(capacity=3, window_s=60.0))
            t0 = time.monotonic()
            await bucket.acquire()
            await bucket.acquire()
            await bucket.acquire()
            elapsed = time.monotonic() - t0
            # Three tokens were pre-filled; should complete near-instantly.
            assert elapsed < 0.1, f"Burst took too long: {elapsed:.3f}s"

        asyncio.run(body())

    def test_fourth_request_waits(self):
        """After the initial burst, the next token must wait for a refill."""
        from agent.core.rate_limiter import _Limit, _TokenBucket

        async def body():
            # 2 tokens, refill 1 every 0.1 s
            bucket = _TokenBucket(_Limit(capacity=2, window_s=0.2))
            await bucket.acquire()
            await bucket.acquire()
            t0 = time.monotonic()
            await bucket.acquire()  # must wait ≈ 0.1 s for a new token
            elapsed = time.monotonic() - t0
            assert 0.05 <= elapsed <= 0.5, f"Unexpected wait time: {elapsed:.3f}s"

        asyncio.run(body())

    def test_tokens_do_not_exceed_capacity(self):
        """After a long idle, the bucket should not accumulate beyond capacity."""
        from agent.core.rate_limiter import _Limit, _TokenBucket

        async def body():
            bucket = _TokenBucket(_Limit(capacity=2, window_s=0.1))
            # Drain the bucket completely
            await bucket.acquire()
            await bucket.acquire()
            # Wait long enough that re-fill would exceed capacity if uncapped.
            await asyncio.sleep(0.5)
            # Should still only allow 2 instant grants (capacity = 2).
            t0 = time.monotonic()
            await bucket.acquire()
            await bucket.acquire()
            elapsed_2 = time.monotonic() - t0
            assert elapsed_2 < 0.15, f"Re-burst too slow: {elapsed_2:.3f}s"
            # Third one requires another partial refill.
            t1 = time.monotonic()
            await bucket.acquire()
            elapsed_3 = time.monotonic() - t1
            assert elapsed_3 >= 0.03, f"Third grant too fast: {elapsed_3:.3f}s"

        asyncio.run(body())


class TestAcquirePublicApi:
    """Public ``acquire()`` function: no-op for uncapped, throttles for opencode."""

    def setup_method(self):
        from agent.core import rate_limiter
        rate_limiter.reset()

    def test_non_opencode_acquire_is_immediate(self):
        """Uncapped providers should return instantly for any number of calls."""
        from agent.core import rate_limiter

        async def body():
            t0 = time.monotonic()
            for _ in range(200):
                await rate_limiter.acquire("anthropic/claude-opus-4-7")
            elapsed = time.monotonic() - t0
            assert elapsed < 0.05, f"No-op path too slow: {elapsed:.3f}s"

        asyncio.run(body())

    def test_opencode_acquire_creates_bucket_on_first_call(self):
        from agent.core import rate_limiter

        async def body():
            assert "opencode" not in rate_limiter._BUCKETS
            await rate_limiter.acquire("opencode/qwen3.6-plus-free")
            assert "opencode" in rate_limiter._BUCKETS

        asyncio.run(body())

    def test_opencode_acquire_reuses_same_bucket(self):
        """Two different opencode model ids share one bucket keyed by prefix."""
        from agent.core import rate_limiter

        async def body():
            await rate_limiter.acquire("opencode/qwen3.6-plus-free")
            bucket_a = rate_limiter._BUCKETS["opencode"]
            await rate_limiter.acquire("opencode/minimax-m2.5-free")
            bucket_b = rate_limiter._BUCKETS["opencode"]
            assert bucket_a is bucket_b

        asyncio.run(body())

    def test_reset_clears_all_buckets(self):
        from agent.core import rate_limiter

        async def body():
            await rate_limiter.acquire("opencode/qwen3.6-plus-free")
            assert len(rate_limiter._BUCKETS) > 0
            rate_limiter.reset()
            assert len(rate_limiter._BUCKETS) == 0

        asyncio.run(body())

    def test_opencode_burst_then_throttle(self):
        """Initial burst fills quickly; the next batch takes longer."""
        from agent.core import rate_limiter
        # Override to small capacity (3 tokens, refill 1 every 0.1 s) for speed.
        from agent.core.rate_limiter import _Limit, _TokenBucket
        rate_limiter._PROVIDER_LIMITS["opencode"] = _Limit(capacity=3, window_s=0.3)

        async def body():
            t0 = time.monotonic()
            # Burst: 3 instant grants
            for _ in range(3):
                await rate_limiter.acquire("opencode/qwen3.6-plus-free")
            burst_time = time.monotonic() - t0
            assert burst_time < 0.1, f"Burst too slow: {burst_time:.3f}s"

            # 4th request should block for ≈ 0.1 s
            t1 = time.monotonic()
            await rate_limiter.acquire("opencode/qwen3.6-plus-free")
            throttle_time = time.monotonic() - t1
            assert throttle_time >= 0.05, f"Throttle too fast: {throttle_time:.3f}s"

        asyncio.run(body())

    def teardown_method(self):
        # Restore real OpenCode limit so other tests aren't affected.
        from agent.core.rate_limiter import _Limit, _PROVIDER_LIMITS
        _PROVIDER_LIMITS["opencode"] = _Limit(capacity=49, window_s=60.0)
        from agent.core import rate_limiter
        rate_limiter.reset()


# ---------------------------------------------------------------------------
# provider_select — credential cascade
# ---------------------------------------------------------------------------

class TestAutoSelectModel:
    """auto_select_model returns the right (model_id, source) pair."""

    def _isolate(self, monkeypatch):
        """Strip all provider-related env vars and the Copilot cache path."""
        _clear_env(monkeypatch, *_PROVIDER_KEYS)
        # Ensure no cached Copilot OAuth state is detected.
        monkeypatch.setattr(
            "agent.core.provider_select._has_copilot_credentials",
            lambda: False,
        )

    def test_explicit_arg_always_wins(self, monkeypatch):
        self._isolate(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from agent.core.provider_select import auto_select_model
        mid, src = auto_select_model("custom/my-model")
        assert mid == "custom/my-model"
        assert src == "explicit"

    def test_ml_intern_model_env_wins_over_cascade(self, monkeypatch):
        self._isolate(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("ML_INTERN_MODEL", "hf-model/foo")
        from agent.core.provider_select import auto_select_model
        mid, src = auto_select_model()
        assert mid == "hf-model/foo"
        assert src == "env:ML_INTERN_MODEL"

    def test_anthropic_key_selects_anthropic_model(self, monkeypatch):
        self._isolate(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from agent.core.provider_select import auto_select_model, DEFAULTS
        mid, src = auto_select_model()
        assert mid == DEFAULTS["anthropic"]
        assert src == "anthropic"

    def test_copilot_token_selects_copilot_model(self, monkeypatch):
        self._isolate(monkeypatch)
        monkeypatch.setattr(
            "agent.core.provider_select._has_copilot_credentials",
            lambda: True,
        )
        from agent.core.provider_select import auto_select_model, DEFAULTS
        mid, src = auto_select_model()
        assert mid == DEFAULTS["copilot"]
        assert src == "copilot"

    def test_opencode_key_selects_opencode_model(self, monkeypatch):
        """When only OPENCODE_API_KEY is set, we pick the free Zen model."""
        self._isolate(monkeypatch)
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-oc-test")
        from agent.core.provider_select import auto_select_model, DEFAULTS
        mid, src = auto_select_model()
        assert mid == DEFAULTS["opencode"]
        assert src == "opencode"
        assert mid.startswith("opencode/")

    def test_opencode_free_model_id_is_sensible(self):
        """The hardcoded default for opencode should look like a free-tier id."""
        from agent.core.provider_select import DEFAULTS
        assert DEFAULTS["opencode"].startswith("opencode/")
        assert "free" in DEFAULTS["opencode"]

    def test_no_credentials_falls_back_to_anonymous_opencode(self, monkeypatch):
        """With no credentials at all we still pick OpenCode (anonymous free
        tier). The agent must be able to boot for first-run UX."""
        self._isolate(monkeypatch)
        from agent.core.provider_select import auto_select_model, DEFAULTS
        mid, src = auto_select_model()
        assert src == "opencode-anonymous"
        assert mid == DEFAULTS["opencode"]

    def test_anthropic_beats_opencode(self, monkeypatch):
        """Anthropic has higher priority than OpenCode in the cascade."""
        self._isolate(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-oc-x")
        from agent.core.provider_select import auto_select_model
        _, src = auto_select_model()
        assert src == "anthropic"

    def test_copilot_beats_opencode(self, monkeypatch):
        """Copilot has higher priority than OpenCode in the cascade."""
        self._isolate(monkeypatch)
        monkeypatch.setattr(
            "agent.core.provider_select._has_copilot_credentials",
            lambda: True,
        )
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-oc-x")
        from agent.core.provider_select import auto_select_model
        _, src = auto_select_model()
        assert src == "copilot"

    def test_anthropic_beats_copilot(self, monkeypatch):
        """Anthropic has higher priority than Copilot."""
        self._isolate(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setattr(
            "agent.core.provider_select._has_copilot_credentials",
            lambda: True,
        )
        from agent.core.provider_select import auto_select_model
        _, src = auto_select_model()
        assert src == "anthropic"


class TestApplyProviderCascade:
    """apply_provider_cascade mutates config.model_name."""

    def _isolate(self, monkeypatch):
        _clear_env(monkeypatch, *_PROVIDER_KEYS)
        monkeypatch.setattr(
            "agent.core.provider_select._has_copilot_credentials",
            lambda: False,
        )

    def test_mutates_config_when_provider_found(self, monkeypatch):
        self._isolate(monkeypatch)
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-oc")

        class _FakeConfig:
            model_name = "bedrock/original-model"

        cfg = _FakeConfig()
        from agent.core.provider_select import apply_provider_cascade, DEFAULTS
        src = apply_provider_cascade(cfg)
        assert src == "opencode"
        assert cfg.model_name == DEFAULTS["opencode"]

    def test_returns_anonymous_opencode_when_no_credentials(self, monkeypatch):
        """With no credentials, cascade falls back to anonymous OpenCode."""
        self._isolate(monkeypatch)

        class _FakeConfig:
            model_name = "bedrock/original-model"

        cfg = _FakeConfig()
        from agent.core.provider_select import apply_provider_cascade, DEFAULTS
        src = apply_provider_cascade(cfg)
        assert src == "opencode-anonymous"
        assert cfg.model_name == DEFAULTS["opencode"]

    def test_explicit_model_wins(self, monkeypatch):
        self._isolate(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        class _FakeConfig:
            model_name = "bedrock/original"

        cfg = _FakeConfig()
        from agent.core.provider_select import apply_provider_cascade
        src = apply_provider_cascade(cfg, explicit="opencode/custom-model")
        assert src == "explicit"
        assert cfg.model_name == "opencode/custom-model"


# ---------------------------------------------------------------------------
# model_switcher — is_valid_model_id accepts new prefixes
# ---------------------------------------------------------------------------

class TestIsValidModelId:
    """is_valid_model_id accepts opencode/ and copilot/ prefixes."""

    def test_opencode_model_is_valid(self):
        from agent.core.model_switcher import is_valid_model_id
        assert is_valid_model_id("opencode/qwen3.6-plus-free")
        assert is_valid_model_id("opencode/minimax-m2.5-free")
        assert is_valid_model_id("opencode/nemotron-3-super-free")

    def test_copilot_model_is_valid(self):
        from agent.core.model_switcher import is_valid_model_id
        assert is_valid_model_id("copilot/gpt-5")
        assert is_valid_model_id("copilot/claude-sonnet-4.5")

    def test_github_copilot_prefix_is_valid(self):
        from agent.core.model_switcher import is_valid_model_id
        assert is_valid_model_id("github_copilot/gpt-5")

    def test_new_providers_in_suggested_models(self):
        from agent.core.model_switcher import SUGGESTED_MODELS
        ids = {m["id"] for m in SUGGESTED_MODELS}
        # At least one opencode and one copilot entry must appear.
        assert any(mid.startswith("opencode/") for mid in ids), \
            "No opencode model in SUGGESTED_MODELS"
        assert any(mid.startswith("copilot/") for mid in ids), \
            "No copilot model in SUGGESTED_MODELS"

    def test_direct_provider_skip_catalog(self):
        """opencode/ and copilot/ prefixes trigger the direct-provider short-circuit."""
        from agent.core.model_switcher import _DIRECT_PROVIDER_PREFIXES
        assert "opencode/" in _DIRECT_PROVIDER_PREFIXES
        assert "copilot/" in _DIRECT_PROVIDER_PREFIXES
        assert "github_copilot/" in _DIRECT_PROVIDER_PREFIXES
