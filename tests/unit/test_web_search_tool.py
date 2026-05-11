import json

import pytest

from agent.core.tools import create_builtin_tools
from agent.tools import web_search_tool


class _FakeResponse:
    def __init__(self, text: str, url: str = "https://html.duckduckgo.com/html/?q=x"):
        self.text = text
        self.url = url


def _content_block(output: dict):
    return next(item for item in output["results"] if isinstance(item, dict))["content"]


def test_web_search_extracts_duckduckgo_results_and_filters_domains(monkeypatch):
    seen = {}

    def fake_get(url, headers, timeout, allow_redirects):
        seen.update(
            {
                "url": url,
                "user_agent": headers["User-Agent"],
                "timeout": timeout,
                "allow_redirects": allow_redirects,
            }
        )
        return _FakeResponse(
            """
            <html><body>
              <a class="result__a" href="https://docs.rs/reqwest">Reqwest docs</a>
              <a class="result__a" href="https://example.com/blocked">Blocked result</a>
            </body></html>
            """,
            url,
        )

    monkeypatch.setenv(
        web_search_tool.WEB_SEARCH_BASE_URL_ENV, "http://search.test/search"
    )
    monkeypatch.setattr(web_search_tool.requests, "get", fake_get)

    output = web_search_tool.execute_web_search(
        "rust web search",
        allowed_domains=["https://DOCS.rs/"],
        blocked_domains=["HTTPS://EXAMPLE.COM"],
    )

    assert seen == {
        "url": "http://search.test/search?q=rust+web+search",
        "user_agent": "clawd-rust-tools/0.1",
        "timeout": 20,
        "allow_redirects": True,
    }
    assert output["query"] == "rust web search"
    assert _content_block(output) == [
        {"title": "Reqwest docs", "url": "https://docs.rs/reqwest"}
    ]
    assert "Include a Sources section" in output["results"][0]


def test_web_search_decodes_duckduckgo_redirects():
    hits = web_search_tool.extract_search_hits(
        """
        <a class="result__a"
           href="/l/?uddg=https%3A%2F%2Fexample.org%2Fpaper%3Fx%3D1&amp;rut=abc">
          Example Paper
        </a>
        """
    )

    assert hits == [
        web_search_tool.SearchHit(
            title="Example Paper",
            url="https://example.org/paper?x=1",
        )
    ]


def test_web_search_generic_fallback_dedupes_and_rejects_bad_base_url(monkeypatch):
    def fake_get(url, headers, timeout, allow_redirects):
        return _FakeResponse(
            """
            <html><body>
              <a href="https://example.com/one">Example One</a>
              <a href="https://example.com/one">Duplicate Example One</a>
              <a href="https://docs.rs/tokio">Tokio Docs</a>
            </body></html>
            """,
            url,
        )

    monkeypatch.setenv(
        web_search_tool.WEB_SEARCH_BASE_URL_ENV, "http://search.test/fallback"
    )
    monkeypatch.setattr(web_search_tool.requests, "get", fake_get)

    output = web_search_tool.execute_web_search("generic links")

    assert _content_block(output) == [
        {"title": "Example One", "url": "https://example.com/one"},
        {"title": "Tokio Docs", "url": "https://docs.rs/tokio"},
    ]

    monkeypatch.setenv(web_search_tool.WEB_SEARCH_BASE_URL_ENV, "://bad-base-url")
    with pytest.raises(ValueError):
        web_search_tool.execute_web_search("generic links")


@pytest.mark.asyncio
async def test_web_search_handler_returns_pretty_json(monkeypatch):
    to_thread_calls = []

    async def fake_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(
        web_search_tool,
        "execute_web_search",
        lambda **kwargs: {
            "query": kwargs["query"],
            "results": [
                "No web search results matched the query 'x'.",
                {"content": []},
            ],
            "durationSeconds": 0.1,
        },
    )
    monkeypatch.setattr(web_search_tool.asyncio, "to_thread", fake_to_thread)

    text, ok = await web_search_tool.web_search_handler({"query": "x"})

    assert ok is False
    assert "at least 2 characters" in text

    text, ok = await web_search_tool.web_search_handler(
        {"query": "valid query"}, tool_call_id="call_123"
    )

    assert ok is True
    parsed = json.loads(text)
    assert parsed["query"] == "valid query"
    assert to_thread_calls[0][0] is web_search_tool.execute_web_search
    assert to_thread_calls[0][2]["tool_use_id"] == "call_123"

    text, ok = await web_search_tool.web_search_handler(
        {"query": "valid query", "allowed_domains": "docs.rs"}
    )

    assert ok is False
    assert "allowed_domains must be an array of strings" in text

    text, ok = await web_search_tool.web_search_handler({"query": None})

    assert ok is False
    assert "query string" in text


def test_web_search_is_registered_for_llm():
    tools = create_builtin_tools(local_mode=True)
    specs = {tool.name: tool for tool in tools}

    assert "web_search" in specs
    assert specs["web_search"].parameters["required"] == ["query"]


def test_disabled_tools_are_not_registered_for_llm():
    tools = create_builtin_tools(
        local_mode=True,
        disabled_tools={"hf_jobs", "notify", "hf_repo_files", "hf_repo_git"},
    )
    specs = {tool.name: tool for tool in tools}

    assert "bash" in specs
    assert "web_search" in specs
    assert "hf_jobs" not in specs
    assert "notify" not in specs
    assert "hf_repo_files" not in specs
    assert "hf_repo_git" not in specs
