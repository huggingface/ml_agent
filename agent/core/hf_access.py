"""Helpers for Hugging Face account / org access decisions."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx

OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")


@dataclass(frozen=True)
class JobsAccess:
    """Jobs entitlement derived from whoami-v2."""

    username: str | None
    plan: str
    personal_can_run_jobs: bool
    paid_org_names: list[str]
    eligible_namespaces: list[str]
    default_namespace: str | None
    access_known: bool = True

    @property
    def can_run_jobs(self) -> bool:
        return bool(self.default_namespace)


class JobsAccessError(Exception):
    """Structured jobs access error for upgrade / namespace gating."""

    def __init__(
        self,
        message: str,
        *,
        access: JobsAccess | None = None,
        upgrade_required: bool = False,
        namespace_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.access = access
        self.upgrade_required = upgrade_required
        self.namespace_required = namespace_required


def _extract_username(whoami: dict[str, Any]) -> str | None:
    for key in ("name", "user", "preferred_username"):
        value = whoami.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _normalize_personal_plan(whoami: dict[str, Any]) -> str:
    plan_str = ""
    for key in ("plan", "type", "accountType"):
        value = whoami.get(key)
        if isinstance(value, str) and value:
            plan_str = value.lower()
            break

    if not plan_str and (whoami.get("isPro") is True or whoami.get("is_pro") is True):
        return "pro"

    if any(tag in plan_str for tag in ("pro", "enterprise", "team")):
        return "pro"
    return "free"


def _paid_org_names(whoami: dict[str, Any]) -> list[str]:
    names: list[str] = []
    orgs = whoami.get("orgs") or []
    if not isinstance(orgs, list):
        return names

    for org in orgs:
        if not isinstance(org, dict):
            continue
        name = org.get("name")
        if not isinstance(name, str) or not name:
            continue
        org_plan = str(org.get("plan") or org.get("type") or "").lower()
        if any(tag in org_plan for tag in ("pro", "enterprise", "team")):
            names.append(name)
    return sorted(set(names))


def jobs_access_from_whoami(whoami: dict[str, Any]) -> JobsAccess:
    username = _extract_username(whoami)
    personal_plan = _normalize_personal_plan(whoami)
    paid_orgs = _paid_org_names(whoami)
    personal_can_run = personal_plan == "pro"

    eligible_namespaces: list[str] = []
    if personal_can_run and username:
        eligible_namespaces.append(username)
    eligible_namespaces.extend(paid_orgs)

    plan = "pro" if personal_can_run else ("org" if paid_orgs else "free")
    default_namespace = username if personal_can_run and username else None

    return JobsAccess(
        username=username,
        plan=plan,
        personal_can_run_jobs=personal_can_run,
        paid_org_names=paid_orgs,
        eligible_namespaces=eligible_namespaces,
        default_namespace=default_namespace,
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
                f"You can only run jobs under your own Pro account or a paid org you belong to. "
                f"Allowed namespaces: {', '.join(access.eligible_namespaces) or '(none)'}",
                access=access,
            )
        if access.default_namespace:
            return access.default_namespace, access
        if access.paid_org_names:
            raise JobsAccessError(
                "Choose which paid organization should own this job run.",
                access=access,
                namespace_required=True,
            )
        raise JobsAccessError(
            "Hugging Face Jobs are available only to Pro users and Team or Enterprise organizations. "
            "Upgrade to Pro, or run the job under a paid org you belong to.",
            access=access,
            upgrade_required=True,
        )

    # Fallback: whoami-v2 unavailable. Do not block the call pre-emptively.
    from huggingface_hub import HfApi

    username = None
    if token:
        whoami = await asyncio.to_thread(HfApi(token=token).whoami)
        username = whoami.get("name")
    if not username:
        raise JobsAccessError("No HF token available to resolve a jobs namespace.")
    return requested_namespace or username, None
