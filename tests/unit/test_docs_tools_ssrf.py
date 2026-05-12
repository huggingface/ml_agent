"""Tests for _is_allowed_doc_url SSRF guard in agent/tools/docs_tools.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Stub heavy dependencies BEFORE any import chain triggers
from unittest.mock import MagicMock

_STUBS = [
    "litellm", "datasets", "fastmcp", "huggingface_hub",
    "sentence_transformers", "nbconvert", "torch",
    "agent", "agent.tools", "agent.core",
]
for mod in _STUBS:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Import just the source file directly, bypassing __init__.py chains
_spec = importlib.util.spec_from_file_location(
    "docs_tools",
    Path(__file__).resolve().parent.parent.parent / "agent" / "tools" / "docs_tools.py",
)
_docs_tools = importlib.util.module_from_spec(_spec)
# Provide the deps that docs_tools actually uses at module level
_deps = {
    "httpx": __import__("httpx"),
    "bs4": __import__("bs4"),
    "whoosh": __import__("whoosh"),
}
for name, mod in _deps.items():
    sys.modules[name] = mod
_spec.loader.exec_module(_docs_tools)

_is_allowed_doc_url = _docs_tools._is_allowed_doc_url


# ── Allowed origins ──────────────────────────────────────────────────────

class TestAllowedOrigins:

    @pytest.mark.parametrize(
        "url",
        [
            "https://huggingface.co/docs/transformers",
            "https://hf.co/docs/trl",
            "https://gradio.app/docs",
            "https://huggingface.co/docs/trl/dpo_trainer",
            "https://hf.co/docs/some-deep/path/page.md",
        ],
    )
    def test_exact_allowed_hosts(self, url: str):
        assert _is_allowed_doc_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://sub.huggingface.co/anything",
            "https://cdn.gradio.app/assets/foo",
            "https://mirror.hf.co/docs/x",
        ],
    )
    def test_subdomain_allowed(self, url: str):
        assert _is_allowed_doc_url(url) is True


# ── Blocked: wrong scheme ────────────────────────────────────────────────

class TestBlockedScheme:

    @pytest.mark.parametrize(
        "url",
        [
            "http://huggingface.co/docs/transformers",
            "http://hf.co/docs/x",
            "ftp://huggingface.co/etc/passwd",
        ],
    )
    def test_non_https_rejected(self, url: str):
        assert _is_allowed_doc_url(url) is False


# ── Blocked: disallowed hosts ────────────────────────────────────────────

class TestBlockedHosts:

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.com/docs",
            "https://169.254.169.254/latest/meta-data/",
            "https://evil-huggingface.co/docs",
            "https://huggingface.co.evil.com/docs",
        ],
    )
    def test_disallowed_hosts_rejected(self, url: str):
        assert _is_allowed_doc_url(url) is False


# ── Blocked: SSRF payloads ───────────────────────────────────────────────

class TestSSRFPayloads:

    @pytest.mark.parametrize(
        "url",
        [
            "https://127.0.0.1/api/internal",
            "https://0.0.0.0/",
            "https://[::1]/admin",
            "https://localhost/etc/passwd",
        ],
    )
    def test_internal_addresses_rejected(self, url: str):
        assert _is_allowed_doc_url(url) is False


# ── Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_string(self):
        assert _is_allowed_doc_url("") is False

    def test_bare_host_no_path(self):
        assert _is_allowed_doc_url("https://huggingface.co") is True

    def test_garbage_input(self):
        assert _is_allowed_doc_url("not-a-url") is False

    def test_port_number(self):
        assert _is_allowed_doc_url("https://huggingface.co:443/docs/x") is True
