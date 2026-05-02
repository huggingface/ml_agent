"""Helpers for Hugging Face account / org access decisions.

HF Jobs are gated by *credits*, not by HF Pro subscriptions. Any user who
has credits — on their personal account or on an org they belong to — can
launch jobs under that namespace. The picker UI lets the caller choose
which wallet to bill.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")


@dataclass(frozen=True)
class JobsAccess:
    """Namespaces the caller may bill HF Jobs to."""

    username: str | None
    org_names: list[str]
    eligible_namespaces: list[str]
    default_namespace: str | None
    access_known: bool = True


class JobsAccessError(Exception):
    """Structured jobs-namespace error.

    ``namespace_required`` fires when the caller belongs to more than one
    eligible namespace and the UI must prompt them to pick one. There is no
    longer an ``upgrade_required`` state — Pro is irrelevant; HF Jobs are
    gated on per-wallet credits, surfaced separately when the API returns
    a billing error at job-creation time.
    """

    def __init__(
        self,
        message: str,
        *,
        access: JobsAccess | None = None,
        namespace_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.access = access
        self.namespace_required = namespace_required


def _extract_username(whoami: dict[str, Any]) -> str | None:
    for key in ("name", "user", "preferred_username"):
        value = whoami.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _org_names(whoami: dict[str, Any]) -> list[str]:
    """All orgs the caller belongs to.

    Plan/tier is ignored — credits live on the namespace itself, so any
    org the user belongs to can host a job as long as it has credits.
    """
    names: list[str] = []
    orgs = whoami.get("orgs") or []
    if not isinstance(orgs, list):
        return names
    for org in orgs:
        if not isinstance(org, dict):
            continue
        name = org.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(set(names))


def jobs_access_from_whoami(whoami: dict[str, Any]) -> JobsAccess:
    username = _extract_username(whoami)
    org_names = _org_names(whoami)
    eligible: list[str] = []
    if username:
        eligible.append(username)
    eligible.extend(org_names)
    default = username if username else (org_names[0] if org_names else None)
    return JobsAccess(
        username=username,
        org_names=org_names,
        eligible_namespaces=eligible,
        default_namespace=default,
    )


async def fetch_whoami_v2(token: str, timeout: float = 5.0) -> dict[str, Any] | None:
    if not token:
        return None
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(
                f"{OPENID_PROVIDER_URL}/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except (httpx.HTTPError, ValueError):
            return None


async def get_jobs_access(token: str) -> JobsAccess | None:
    whoami = await fetch_whoami_v2(token)
    if whoami is None:
        return None
    return jobs_access_from_whoami(whoami)


async def resolve_jobs_namespace(
    token: str,
    requested_namespace: str | None = None,
) -> tuple[str, JobsAccess | None]:
    """Return the namespace to use for jobs.

    If whoami-v2 is unavailable, fall back to the token owner's username.
    """
    access = await get_jobs_access(token)
    if access:
        if requested_namespace:
            if requested_namespace in access.eligible_namespaces:
                return requested_namespace, access
            raise JobsAccessError(
                f"You can only run jobs under your own account or an org you belong to. "
                f"Allowed namespaces: {', '.join(access.eligible_namespaces) or '(none)'}",
                access=access,
            )
        if access.default_namespace:
            return access.default_namespace, access
        raise JobsAccessError(
            "Couldn't resolve a Hugging Face namespace for this token.",
            access=access,
        )

    # Fallback: whoami-v2 unavailable. Don't block the call pre-emptively.
    from huggingface_hub import HfApi

    username = None
    if token:
        whoami = await asyncio.to_thread(HfApi(token=token).whoami)
        username = whoami.get("name")
    if not username:
        raise JobsAccessError("No HF token available to resolve a jobs namespace.")
    return requested_namespace or username, None


_BILLING_PATTERNS = re.compile(
    r"\b(insufficient[_\s-]?credits?|out\s+of\s+credits?|payment\s+required|"
    r"billing|no\s+credits?|add\s+credits?|requires?\s+credits?)\b",
    re.IGNORECASE,
)


def is_billing_error(message: str) -> bool:
    """True if an HF API error message looks like an out-of-credits / billing error."""
    if not message:
        return False
    if "402" in message:
        return True
    return bool(_BILLING_PATTERNS.search(message))
