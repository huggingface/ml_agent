import os

import litellm
import pytest

from agent.core.observability import setup_langfuse


@pytest.fixture(autouse=True)
def _reset_litellm_callbacks():
    """Restore litellm callback lists around each test so they don't leak."""
    success_before = list(litellm.success_callback)
    failure_before = list(litellm.failure_callback)
    try:
        yield
    finally:
        litellm.success_callback[:] = success_before
        litellm.failure_callback[:] = failure_before


def _set_all_vars(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example.com")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)


def test_setup_langfuse_registers_callbacks_when_all_vars_set(monkeypatch):
    _set_all_vars(monkeypatch)

    setup_langfuse()

    assert "langfuse_otel" in litellm.success_callback
    assert "langfuse_otel" in litellm.failure_callback


def test_setup_langfuse_accepts_base_url_alias(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.com")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    setup_langfuse()

    assert "langfuse_otel" in litellm.success_callback
    # litellm only reads LANGFUSE_HOST; the alias must be mirrored into it.
    assert os.environ["LANGFUSE_HOST"] == "https://langfuse.example.com"


def test_setup_langfuse_host_takes_precedence_over_base_url(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "https://from-host.example.com")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://from-base-url.example.com")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    setup_langfuse()

    assert os.environ["LANGFUSE_HOST"] == "https://from-host.example.com"


@pytest.mark.parametrize(
    "missing", ["LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
)
def test_setup_langfuse_is_noop_when_any_var_missing(monkeypatch, missing):
    _set_all_vars(monkeypatch)
    monkeypatch.delenv(missing, raising=False)
    success_before = list(litellm.success_callback)
    failure_before = list(litellm.failure_callback)

    setup_langfuse()

    assert litellm.success_callback == success_before
    assert litellm.failure_callback == failure_before


def test_setup_langfuse_is_idempotent(monkeypatch):
    _set_all_vars(monkeypatch)

    setup_langfuse()
    setup_langfuse()

    assert litellm.success_callback.count("langfuse_otel") == 1
    assert litellm.failure_callback.count("langfuse_otel") == 1
