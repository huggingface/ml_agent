"""Conservative cost estimates for auto-approved infrastructure actions."""

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")
JOBS_HARDWARE_URL = f"{OPENID_PROVIDER_URL}/api/jobs/hardware"
JOBS_PRICE_CACHE_TTL_S = 6 * 60 * 60

DEFAULT_JOB_TIMEOUT_HOURS = 0.5
DEFAULT_SANDBOX_RESERVATION_HOURS = 1.0

# Static fallback prices are intentionally conservative enough for a budget
# guard. The live /api/jobs/hardware catalog wins whenever it is reachable.
HF_JOBS_PRICE_USD_PER_HOUR: dict[str, float] = {
    "cpu-basic": 0.05,
    "cpu-upgrade": 0.25,
    "cpu-performance": 0.50,
    "cpu-xl": 1.00,
    "t4-small": 0.60,
    "t4-medium": 0.90,
    "l4x1": 1.00,
    "l4x4": 4.00,
    "l40sx1": 2.00,
    "l40sx4": 8.00,
    "l40sx8": 16.00,
    "a10g-small": 1.00,
    "a10g-large": 2.00,
    "a10g-largex2": 4.00,
    "a10g-largex4": 8.00,
    "a100-large": 4.00,
    "a100x4": 16.00,
    "a100x8": 32.00,
    "h200": 10.00,
    "h200x2": 20.00,
    "h200x4": 40.00,
    "h200x8": 80.00,
    "inf2x6": 6.00,
}

SPACE_PRICE_USD_PER_HOUR: dict[str, float] = {
    "cpu-basic": 0.0,
    "cpu-upgrade": 0.05,
    "cpu-performance": 0.50,
    "cpu-xl": 1.00,
    "t4-small": 0.60,
    "t4-medium": 0.90,
    "l4x1": 1.00,
    "l4x4": 4.00,
    "l40sx1": 2.00,
    "l40sx4": 8.00,
    "l40sx8": 16.00,
    "a10g-small": 1.00,
    "a10g-large": 2.00,
    "a10g-largex2": 4.00,
    "a10g-largex4": 8.00,
    "a100-large": 4.00,
    "a100x4": 16.00,
    "a100x8": 32.00,
    "h200": 10.00,
    "h200x2": 20.00,
    "h200x4": 40.00,
    "h200x8": 80.00,
    "inf2x6": 6.00,
}

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)
_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")
_jobs_price_cache: tuple[float, dict[str, float]] | None = None


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost for a tool call.

    ``estimated_cost_usd=None`` means the call may be billable but we could not
    estimate it safely, so auto-approval should fall back to a human decision.
    """

    estimated_cost_usd: float | None
    billable: bool
    block_reason: str | None = None
    label: str | None = None


def parse_timeout_hours(value: Any, *, default_hours: float = DEFAULT_JOB_TIMEOUT_HOURS) -> float | None:
    """Parse HF timeout values into hours.

    Strings accept ``s``, ``m``, ``h``, or ``d`` suffixes. Numeric values are
    treated as seconds, matching the Hub client's typed timeout parameter.
    """
    if value is None or value == "":
        return default_hours
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        seconds = float(value)
        return seconds / 3600 if seconds > 0 else None
    if not isinstance(value, str):
        return None

    match = _DURATION_RE.match(value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower() or "s"
    if amount <= 0:
        return None
    if unit == "s":
        return amount / 3600
    if unit == "m":
        return amount / 60
    if unit == "h":
        return amount
    if unit == "d":
        return amount * 24
    return None


def _extract_flavor(item: dict[str, Any]) -> str | None:
    for key in ("flavor", "name", "id", "value", "hardware", "hardware_flavor"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _coerce_price(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value) if value >= 0 else None
    if isinstance(value, str):
        match = _PRICE_RE.search(value.replace(",", ""))
        if match:
            return float(match.group(1))
    return None


def _extract_hourly_price(item: dict[str, Any]) -> float | None:
    for key in (
        "price",
        "price_usd",
        "priceUsd",
        "price_per_hour",
        "pricePerHour",
        "hourly_price",
        "hourlyPrice",
        "usd_per_hour",
        "usdPerHour",
    ):
        price = _coerce_price(item.get(key))
        if price is not None:
            return price
    for key in ("pricing", "billing", "cost"):
        nested = item.get(key)
        if isinstance(nested, dict):
            price = _extract_hourly_price(nested)
            if price is not None:
                return price
    return None


def _iter_hardware_items(payload: Any):
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_hardware_items(item)
    elif isinstance(payload, dict):
        if _extract_flavor(payload):
            yield payload
        for key in ("hardware", "flavors", "items", "data", "jobs"):
            child = payload.get(key)
            if child is not None:
                yield from _iter_hardware_items(child)


def _parse_jobs_price_catalog(payload: Any) -> dict[str, float]:
    prices: dict[str, float] = {}
    for item in _iter_hardware_items(payload):
        flavor = _extract_flavor(item)
        price = _extract_hourly_price(item)
        if flavor and price is not None:
            prices[flavor] = price
    return prices


async def hf_jobs_price_catalog() -> dict[str, float]:
    """Return live HF Jobs hourly prices, falling back to static prices."""
    global _jobs_price_cache
    now = time.monotonic()
    if _jobs_price_cache and now - _jobs_price_cache[0] < JOBS_PRICE_CACHE_TTL_S:
        return dict(_jobs_price_cache[1])

    prices: dict[str, float] = {}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(JOBS_HARDWARE_URL)
            if response.status_code == 200:
                prices = _parse_jobs_price_catalog(response.json())
    except (httpx.HTTPError, ValueError):
        prices = {}

    if not prices:
        prices = dict(HF_JOBS_PRICE_USD_PER_HOUR)
    else:
        prices = {**HF_JOBS_PRICE_USD_PER_HOUR, **prices}

    _jobs_price_cache = (now, prices)
    return dict(prices)


async def estimate_hf_job_cost(args: dict[str, Any]) -> CostEstimate:
    flavor = str(
        args.get("hardware_flavor")
        or args.get("flavor")
        or args.get("hardware")
        or "cpu-basic"
    )
    timeout_hours = parse_timeout_hours(args.get("timeout"))
    if timeout_hours is None:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=f"Could not parse HF job timeout: {args.get('timeout')!r}.",
            label=flavor,
        )

    prices = await hf_jobs_price_catalog()
    price = prices.get(flavor)
    if price is None:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=f"No price is available for HF job hardware '{flavor}'.",
            label=flavor,
        )

    return CostEstimate(
        estimated_cost_usd=round(price * timeout_hours, 4),
        billable=price > 0,
        label=flavor,
    )


async def estimate_sandbox_cost(args: dict[str, Any], *, session: Any = None) -> CostEstimate:
    if session is not None and getattr(session, "sandbox", None):
        return CostEstimate(estimated_cost_usd=0.0, billable=False, label="existing")

    hardware = str(args.get("hardware") or "cpu-basic")
    price = SPACE_PRICE_USD_PER_HOUR.get(hardware)
    if price is None:
        return CostEstimate(
            estimated_cost_usd=None,
            billable=True,
            block_reason=f"No price is available for sandbox hardware '{hardware}'.",
            label=hardware,
        )

    return CostEstimate(
        estimated_cost_usd=round(price * DEFAULT_SANDBOX_RESERVATION_HOURS, 4),
        billable=price > 0,
        label=hardware,
    )


async def estimate_tool_cost(
    tool_name: str, args: dict[str, Any], *, session: Any = None
) -> CostEstimate:
    if tool_name == "sandbox_create":
        return await estimate_sandbox_cost(args, session=session)
    if tool_name == "hf_jobs":
        return await estimate_hf_job_cost(args)
    return CostEstimate(estimated_cost_usd=0.0, billable=False)
