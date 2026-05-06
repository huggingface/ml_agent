"""DuckDuckGo HTML web search tool.

This mirrors Claw Code's Rust WebSearch behavior: fetch DuckDuckGo's HTML
endpoint, extract result links, optionally filter domains, and return a
JSON payload the model can cite.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, parse_qs, urlencode, urlparse, urlunparse

import requests

DEFAULT_SEARCH_URL = "https://html.duckduckgo.com/html/"
WEB_SEARCH_BASE_URL_ENV = "CLAWD_WEB_SEARCH_BASE_URL"
USER_AGENT = "clawd-rust-tools/0.1"
REQUEST_TIMEOUT_SECONDS = 20
MAX_RESULTS = 8


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str

    def as_json(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url}


class _AnchorParser(HTMLParser):
    def __init__(self, *, require_result_class: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.require_result_class = require_result_class
        self.hits: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {key.lower(): value or "" for key, value in attrs}
        href = attr_map.get("href")
        if not href:
            return
        if self.require_result_class and "result__a" not in attr_map.get("class", ""):
            return
        self._active_href = href
        self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._active_href is not None:
            self._active_text.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._active_href is not None:
            self._active_text.append(f"&#{name};")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._active_href is None:
            return
        title = collapse_whitespace(html.unescape("".join(self._active_text))).strip()
        self.hits.append((self._active_href, title))
        self._active_href = None
        self._active_text = []


def build_search_url(query: str) -> str:
    base = os.environ.get(WEB_SEARCH_BASE_URL_ENV, DEFAULT_SEARCH_URL)
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid search base URL: {base}")

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_pairs.append(("q", query))
    return urlunparse(parsed._replace(query=urlencode(query_pairs)))


def collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def decode_duckduckgo_redirect(url: str) -> str | None:
    if url.startswith("http://") or url.startswith("https://"):
        return html.unescape(url)
    if url.startswith("//"):
        joined = f"https:{url}"
    elif url.startswith("/"):
        joined = f"https://duckduckgo.com{url}"
    else:
        return None

    parsed = urlparse(joined)
    if parsed.path in {"/l", "/l/"}:
        uddg = parse_qs(parsed.query).get("uddg", [])
        if uddg:
            return html.unescape(uddg[0])
    return joined


def _extract_links(search_html: str, *, require_result_class: bool) -> list[SearchHit]:
    parser = _AnchorParser(require_result_class=require_result_class)
    parser.feed(search_html)

    hits: list[SearchHit] = []
    for raw_url, title in parser.hits:
        if not title:
            continue
        decoded_url = decode_duckduckgo_redirect(raw_url)
        if decoded_url and (
            decoded_url.startswith("http://") or decoded_url.startswith("https://")
        ):
            hits.append(SearchHit(title=title, url=decoded_url))
    return hits


def extract_search_hits(search_html: str) -> list[SearchHit]:
    return _extract_links(search_html, require_result_class=True)


def extract_search_hits_from_generic_links(search_html: str) -> list[SearchHit]:
    return _extract_links(search_html, require_result_class=False)


def normalize_domain_filter(domain: str) -> str:
    trimmed = domain.strip()
    parsed = urlparse(trimmed)
    candidate = parsed.hostname if parsed.scheme and parsed.hostname else trimmed
    return candidate.strip().lstrip(".").rstrip("/").lower()


def host_matches_list(url: str, domains: list[str]) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    normalized_host = host.lower()
    for domain in domains:
        normalized = normalize_domain_filter(domain)
        if normalized and (
            normalized_host == normalized or normalized_host.endswith(f".{normalized}")
        ):
            return True
    return False


def dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    deduped: list[SearchHit] = []
    for hit in hits:
        if hit.url in seen:
            continue
        seen.add(hit.url)
        deduped.append(hit)
    return deduped


def execute_web_search(
    query: str,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    tool_use_id: str = "web_search_1",
) -> dict[str, Any]:
    started = time.monotonic()
    search_url = build_search_url(query)
    response = requests.get(
        search_url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )

    hits = extract_search_hits(response.text)
    if not hits and urlparse(response.url or search_url).hostname:
        hits = extract_search_hits_from_generic_links(response.text)

    if allowed_domains is not None:
        hits = [hit for hit in hits if host_matches_list(hit.url, allowed_domains)]
    if blocked_domains is not None:
        hits = [hit for hit in hits if not host_matches_list(hit.url, blocked_domains)]

    hits = dedupe_hits(hits)[:MAX_RESULTS]
    rendered_hits = "\n".join(f"- [{hit.title}]({hit.url})" for hit in hits)
    if hits:
        summary = (
            f"Search results for {query!r}. Include a Sources section in the final answer.\n"
            f"{rendered_hits}"
        )
    else:
        summary = f"No web search results matched the query {query!r}."

    return {
        "query": query,
        "results": [
            summary,
            {
                "tool_use_id": tool_use_id,
                "content": [hit.as_json() for hit in hits],
            },
        ],
        "durationSeconds": time.monotonic() - started,
    }


WEB_SEARCH_TOOL_SPEC = {
    "name": "web_search",
    "description": "Search the web for current information and return cited results.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional allowlist of domains or URLs. Subdomains match.",
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional blocklist of domains or URLs. Subdomains match.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def _optional_string_list(arguments: dict[str, Any], key: str) -> list[str] | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return value


async def web_search_handler(
    arguments: dict[str, Any],
    session: Any = None,
    tool_call_id: str | None = None,
    **_kw: Any,
) -> tuple[str, bool]:
    query_value = arguments.get("query", "")
    if not isinstance(query_value, str):
        return (
            "Error: web_search requires a query string with at least 2 characters.",
            False,
        )

    query = query_value.strip()
    if len(query) < 2:
        return "Error: web_search requires a query with at least 2 characters.", False

    try:
        output = await asyncio.to_thread(
            execute_web_search,
            query=query,
            allowed_domains=_optional_string_list(arguments, "allowed_domains"),
            blocked_domains=_optional_string_list(arguments, "blocked_domains"),
            tool_use_id=tool_call_id or "web_search_1",
        )
    except Exception as exc:
        return f"Error executing web search: {exc}", False

    return json.dumps(output, indent=2), True
