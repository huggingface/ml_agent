"""Fetch and cache the HF Inference Router model catalog.

The router exposes an OpenAI-compatible listing at
``https://router.huggingface.co/v1/models`` with per-provider availability,
pricing, context length, and tool-use support. We use it to:

  • Validate ``/model`` switches with live data instead of a hard-coded allowlist.
  • Show the user which providers serve a model, at what price, and whether they
    support tool calls.
  • Derive a reasonable context-window limit for any routed model.

The listing is cached in-memory for a few minutes so repeated lookups during a
session are free. On fetch failure we return stale data if we have it, or an
empty catalog otherwise.
"""

import logging
import time
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_CATALOG_URL = "https://router.huggingface.co/v1/models"
_CACHE_TTL_SECONDS = 300
_HTTP_TIMEOUT_SECONDS = 5.0

_cache: Optional[dict] = None
_cache_time: float = 0.0


@dataclass
class ProviderInfo:
    provider: str
    status: str
    context_length: Optional[int]
    input_price: Optional[float]
    output_price: Optional[float]
    supports_tools: bool
    supports_structured_output: bool


@dataclass
class ModelInfo:
    id: str
    providers: list[ProviderInfo]

    @property
    def live_providers(self) -> list[ProviderInfo]:
        return [p for p in self.providers if p.status == "live"]

    @property
    def max_context_length(self) -> Optional[int]:
        lengths = [p.context_length for p in self.live_providers if p.context_length]
        return max(lengths) if lengths else None

    @property
    def any_supports_tools(self) -> bool:
        return any(p.supports_tools for p in self.live_providers)


def _fetch_catalog(force: bool = False) -> dict:
    global _cache, _cache_time
    now = time.time()
    if not force and _cache is not None and now - _cache_time < _CACHE_TTL_SECONDS:
        return _cache
    try:
        resp = httpx.get(_CATALOG_URL, timeout=_HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        _cache = resp.json()
        _cache_time = now
    except Exception as e:
        logger.warning("Failed to fetch HF router catalog: %s", e)
        if _cache is None:
            _cache = {"data": []}
            _cache_time = now
    return _cache


def _parse_entry(entry: dict) -> ModelInfo:
    providers = []
    for p in entry.get("providers", []) or []:
        pricing = p.get("pricing") or {}
        providers.append(
            ProviderInfo(
                provider=p.get("provider", ""),
                status=p.get("status", ""),
                context_length=p.get("context_length"),
                input_price=pricing.get("input"),
                output_price=pricing.get("output"),
                supports_tools=bool(p.get("supports_tools", False)),
                supports_structured_output=bool(p.get("supports_structured_output", False)),
            )
        )
    return ModelInfo(id=entry.get("id", ""), providers=providers)


def lookup(model_id: str) -> Optional[ModelInfo]:
    """Find a model in the router catalog.

    Accepts ``<org>/<model>`` or ``<org>/<model>:<tag>`` — the tag is stripped
    for lookup. Returns ``None`` if the model isn't listed.
    """
    bare = model_id.split(":", 1)[0]
    catalog = _fetch_catalog()
    for entry in catalog.get("data", []):
        if entry.get("id") == bare:
            return _parse_entry(entry)
    return None


def fuzzy_suggest(model_id: str, limit: int = 3) -> list[str]:
    """Return the closest model ids from the catalog."""
    bare = model_id.split(":", 1)[0]
    catalog = _fetch_catalog()
    ids = [e.get("id", "") for e in catalog.get("data", []) if e.get("id")]
    return get_close_matches(bare, ids, n=limit, cutoff=0.4)


def prewarm() -> None:
    """Fetch the catalog so subsequent lookups are instant. Safe to call from
    a background task — swallows failures."""
    try:
        _fetch_catalog(force=False)
    except Exception:
        pass
