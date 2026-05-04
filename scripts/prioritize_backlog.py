#!/usr/bin/env python3
"""Prioritize the open ML Intern backlog with a product-manager prompt.

Collects open GitHub issues, open GitHub pull requests, and open Hugging Face
Space discussions, then asks an LLM to classify, cluster, and rank them by
likely product impact.

Usage:
    uv run python scripts/prioritize_backlog.py
    uv run python scripts/prioritize_backlog.py --model openai/gpt-5.5

Outputs:
    scratch/backlog-prioritization/<timestamp>/sources.json
    scratch/backlog-prioritization/<timestamp>/ranking.json
    scratch/backlog-prioritization/<timestamp>/report.md
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GITHUB_API = "https://api.github.com"
DEFAULT_GITHUB_REPO = "huggingface/ml-intern"
DEFAULT_HF_SPACE = "smolagents/ml-intern"
DEFAULT_CONFIG = "configs/cli_agent_config.json"
DEFAULT_BATCH_SIZE = 12
DEFAULT_MAX_COMMENTS = 8
DEFAULT_MAX_REVIEW_COMMENTS = 8
DEFAULT_MAX_BODY_CHARS = 6000
DEFAULT_MAX_COMMENT_CHARS = 1500
DEFAULT_MAX_OUTPUT_TOKENS = 4096

logger = logging.getLogger("prioritize_backlog")

PM_SYSTEM_PROMPT = """You are a senior product manager for ML Intern.

Your job is to turn messy public feedback into a pragmatic implementation
priority list. Optimize for:
- user impact and blocked workflows
- evidence of repeated demand or engagement
- recency and severity
- PR readiness and whether an open PR should be reviewed/merged/fixed forward
- implementation effort, risk, and strategic fit for ML Intern

Separate user-facing features from bug fixes. Treat open PRs as possible
ready-made implementations rather than duplicate feature requests. Every
recommendation must cite source ids and/or source URLs from the input.

Return valid JSON only. Do not use Markdown fences.
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def default_output_dir(now: datetime | None = None) -> Path:
    now = now or utc_now()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "scratch" / "backlog-prioritization" / stamp


def resolve_output_dir(value: str | None, now: datetime | None = None) -> Path:
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return default_output_dir(now)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prioritize GitHub and HF Space backlog items with an LLM."
    )
    ap.add_argument("--github-repo", default=DEFAULT_GITHUB_REPO)
    ap.add_argument("--hf-space", default=DEFAULT_HF_SPACE)
    ap.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Config file used to resolve the default model.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Override the model from configs/cli_agent_config.json.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to scratch/backlog-prioritization/<UTC timestamp>.",
    )
    ap.add_argument("--github-token", default=None, help="Defaults to GITHUB_TOKEN.")
    ap.add_argument(
        "--hf-token",
        default=None,
        help="Defaults to HF_TOKEN or the local huggingface_hub token cache.",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-comments", type=int, default=DEFAULT_MAX_COMMENTS)
    ap.add_argument(
        "--max-review-comments", type=int, default=DEFAULT_MAX_REVIEW_COMMENTS
    )
    ap.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS)
    ap.add_argument(
        "--max-comment-chars", type=int, default=DEFAULT_MAX_COMMENT_CHARS
    )
    ap.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    ap.add_argument(
        "--reasoning-effort",
        default="high",
        help="Reasoning effort preference passed through the repo LLM resolver.",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return ap.parse_args(argv)


def resolve_model(model: str | None, config_path: str) -> str:
    if model:
        return model

    from agent.config import load_config

    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return load_config(str(path), include_user_defaults=True).model_name


def resolve_hf_token(cli_token: str | None) -> str | None:
    from agent.core.hf_tokens import resolve_hf_token as _resolve_hf_token

    return _resolve_hf_token(cli_token, os.environ.get("HF_TOKEN"))


def _truncate_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    text = str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = "\n... [truncated]"
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _github_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ml-intern-backlog-prioritizer",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _raise_for_status(response: Any) -> None:
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()


def _get_json(client: Any, url: str, headers: dict[str, str]) -> Any:
    response = client.get(url, headers=headers)
    _raise_for_status(response)
    return response.json()


def _paginated_json(
    client: Any,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[Any]:
    params = dict(params or {})
    page = 1
    out: list[Any] = []
    while True:
        page_params = {**params, "per_page": 100, "page": page}
        response = client.get(url, headers=headers, params=page_params)
        _raise_for_status(response)
        data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"Expected list response from {url}, got {type(data)}")

        for item in data:
            out.append(item)
            if limit is not None and len(out) >= limit:
                return out

        link = getattr(response, "headers", {}).get("link", "")
        if not data or 'rel="next"' not in link:
            return out
        page += 1


def _labels(raw_labels: list[Any]) -> list[str]:
    labels: list[str] = []
    for label in raw_labels or []:
        if isinstance(label, dict):
            name = label.get("name")
        else:
            name = str(label)
        if name:
            labels.append(str(name))
    return labels


def _user_login(raw: dict[str, Any] | None) -> str | None:
    if not raw:
        return None
    return raw.get("login") or raw.get("name")


def _reactions(raw: dict[str, Any] | None) -> dict[str, int]:
    if not raw:
        return {}
    keep = (
        "total_count",
        "+1",
        "-1",
        "laugh",
        "hooray",
        "confused",
        "heart",
        "rocket",
        "eyes",
    )
    return {key: int(raw.get(key) or 0) for key in keep if raw.get(key) is not None}


def _normalize_github_comment(
    raw: dict[str, Any],
    *,
    max_comment_chars: int,
    kind: str = "comment",
) -> dict[str, Any]:
    return {
        "kind": kind,
        "author": _user_login(raw.get("user")),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "url": raw.get("html_url") or raw.get("url"),
        "state": raw.get("state"),
        "body": _truncate_text(raw.get("body"), max_comment_chars),
        "reactions": _reactions(raw.get("reactions")),
    }


def _fetch_github_comments(
    client: Any,
    url: str | None,
    headers: dict[str, str],
    *,
    max_comments: int,
    max_comment_chars: int,
    kind: str = "comment",
) -> list[dict[str, Any]]:
    if not url or max_comments <= 0:
        return []
    raw_comments = _paginated_json(client, url, headers, limit=max_comments)
    return [
        _normalize_github_comment(
            comment, max_comment_chars=max_comment_chars, kind=kind
        )
        for comment in raw_comments
    ]


def _normalize_github_issue(
    item: dict[str, Any],
    comments: list[dict[str, Any]],
    *,
    max_body_chars: int,
) -> dict[str, Any]:
    number = int(item["number"])
    return {
        "id": f"github_issue#{number}",
        "source": "github_issue",
        "number": number,
        "url": item.get("html_url"),
        "title": item.get("title") or "",
        "body": _truncate_text(item.get("body"), max_body_chars),
        "labels": _labels(item.get("labels") or []),
        "author": _user_login(item.get("user")),
        "state": item.get("state"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "closed_at": item.get("closed_at"),
        "engagement": {
            "comments_count": item.get("comments") or len(comments),
            "reactions": _reactions(item.get("reactions")),
        },
        "comments": comments,
        "metadata": {
            "state_reason": item.get("state_reason"),
        },
    }


def _normalize_github_pr(
    item: dict[str, Any],
    pr_details: dict[str, Any],
    comments: list[dict[str, Any]],
    review_comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    *,
    max_body_chars: int,
) -> dict[str, Any]:
    number = int(item["number"])
    combined_comments = [*comments, *reviews, *review_comments]
    base = pr_details.get("base") or {}
    head = pr_details.get("head") or {}
    return {
        "id": f"github_pr#{number}",
        "source": "github_pr",
        "number": number,
        "url": pr_details.get("html_url") or item.get("html_url"),
        "title": pr_details.get("title") or item.get("title") or "",
        "body": _truncate_text(pr_details.get("body") or item.get("body"), max_body_chars),
        "labels": _labels(item.get("labels") or []),
        "author": _user_login(pr_details.get("user") or item.get("user")),
        "state": pr_details.get("state") or item.get("state"),
        "created_at": pr_details.get("created_at") or item.get("created_at"),
        "updated_at": pr_details.get("updated_at") or item.get("updated_at"),
        "closed_at": pr_details.get("closed_at") or item.get("closed_at"),
        "engagement": {
            "comments_count": item.get("comments") or len(comments),
            "review_comments_count": pr_details.get("review_comments"),
            "reactions": _reactions(item.get("reactions")),
        },
        "comments": combined_comments,
        "metadata": {
            "draft": pr_details.get("draft"),
            "mergeable_state": pr_details.get("mergeable_state"),
            "base": base.get("ref"),
            "head": head.get("ref"),
            "commits": pr_details.get("commits"),
            "additions": pr_details.get("additions"),
            "deletions": pr_details.get("deletions"),
            "changed_files": pr_details.get("changed_files"),
        },
    }


def collect_github_sources(
    repo: str,
    *,
    token: str | None = None,
    max_comments: int = DEFAULT_MAX_COMMENTS,
    max_review_comments: int = DEFAULT_MAX_REVIEW_COMMENTS,
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
    max_comment_chars: int = DEFAULT_MAX_COMMENT_CHARS,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    headers = _github_headers(token)
    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=30.0, follow_redirects=True)

    try:
        issues_url = f"{GITHUB_API}/repos/{repo}/issues"
        raw_items = _paginated_json(
            client,
            issues_url,
            headers,
            params={"state": "open", "sort": "updated", "direction": "desc"},
        )

        records: list[dict[str, Any]] = []
        for item in raw_items:
            issue_comments = _fetch_github_comments(
                client,
                item.get("comments_url"),
                headers,
                max_comments=max_comments,
                max_comment_chars=max_comment_chars,
            )

            if "pull_request" not in item:
                records.append(
                    _normalize_github_issue(
                        item, issue_comments, max_body_chars=max_body_chars
                    )
                )
                continue

            number = item["number"]
            pr_url = f"{GITHUB_API}/repos/{repo}/pulls/{number}"
            pr_details = _get_json(client, pr_url, headers)
            review_comments = _fetch_github_comments(
                client,
                f"{pr_url}/comments",
                headers,
                max_comments=max_review_comments,
                max_comment_chars=max_comment_chars,
                kind="review_comment",
            )
            raw_reviews = _paginated_json(
                client,
                f"{pr_url}/reviews",
                headers,
                limit=max_review_comments,
            )
            reviews = [
                _normalize_github_comment(
                    review, max_comment_chars=max_comment_chars, kind="review"
                )
                for review in raw_reviews
                if review.get("body")
            ]
            records.append(
                _normalize_github_pr(
                    item,
                    pr_details,
                    issue_comments,
                    review_comments,
                    reviews,
                    max_body_chars=max_body_chars,
                )
            )
        return records
    finally:
        if close_client and hasattr(client, "close"):
            client.close()


def _hf_comment_event(event: Any, max_comment_chars: int) -> dict[str, Any] | None:
    content = getattr(event, "content", None)
    if content is None:
        return None
    if getattr(event, "hidden", False):
        return None
    return {
        "kind": getattr(event, "type", "comment") or "comment",
        "author": getattr(event, "author", None),
        "created_at": _iso(getattr(event, "created_at", None)),
        "updated_at": None,
        "url": None,
        "state": None,
        "body": _truncate_text(content, max_comment_chars),
        "reactions": {},
    }


def normalize_hf_discussion(
    discussion: Any,
    details: Any,
    *,
    max_comments: int = DEFAULT_MAX_COMMENTS,
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
    max_comment_chars: int = DEFAULT_MAX_COMMENT_CHARS,
) -> dict[str, Any]:
    events = list(getattr(details, "events", []) or [])
    visible_comment_events = [
        event
        for event in events
        if getattr(event, "content", None) is not None
        and not getattr(event, "hidden", False)
    ]
    first_comment = visible_comment_events[0] if visible_comment_events else None
    comments = [
        comment
        for comment in (
            _hf_comment_event(event, max_comment_chars=max_comment_chars)
            for event in visible_comment_events[1 : max_comments + 1]
        )
        if comment is not None
    ]
    number = int(getattr(discussion, "num", getattr(details, "num", 0)))
    repo_id = getattr(
        discussion, "repo_id", getattr(details, "repo_id", DEFAULT_HF_SPACE)
    )
    url = f"https://huggingface.co/spaces/{repo_id}/discussions/{number}"

    return {
        "id": f"hf_discussion#{number}",
        "source": "hf_discussion",
        "number": number,
        "url": url,
        "title": getattr(details, "title", getattr(discussion, "title", "")) or "",
        "body": _truncate_text(
            getattr(first_comment, "content", "") if first_comment else "",
            max_body_chars,
        ),
        "labels": [],
        "author": getattr(discussion, "author", getattr(details, "author", None)),
        "state": getattr(details, "status", getattr(discussion, "status", None)),
        "created_at": _iso(getattr(discussion, "created_at", None)),
        "updated_at": None,
        "closed_at": None,
        "engagement": {
            "comments_count": len(visible_comment_events),
            "reactions": {},
        },
        "comments": comments,
        "metadata": {
            "repo_id": repo_id,
            "repo_type": getattr(discussion, "repo_type", "space"),
            "events_count": len(events),
        },
    }


def collect_hf_discussions(
    space_id: str,
    *,
    token: str | None = None,
    max_comments: int = DEFAULT_MAX_COMMENTS,
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
    max_comment_chars: int = DEFAULT_MAX_COMMENT_CHARS,
    api: Any | None = None,
) -> list[dict[str, Any]]:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    records: list[dict[str, Any]] = []
    discussions = api.get_repo_discussions(
        repo_id=space_id,
        repo_type="space",
        discussion_type="discussion",
        discussion_status="open",
        token=token,
    )
    for discussion in discussions:
        details = api.get_discussion_details(
            repo_id=space_id,
            repo_type="space",
            discussion_num=discussion.num,
            token=token,
        )
        records.append(
            normalize_hf_discussion(
                discussion,
                details,
                max_comments=max_comments,
                max_body_chars=max_body_chars,
                max_comment_chars=max_comment_chars,
            )
        )
    return records


def collect_sources(
    github_repo: str,
    hf_space: str,
    *,
    github_token: str | None = None,
    hf_token: str | None = None,
    max_comments: int = DEFAULT_MAX_COMMENTS,
    max_review_comments: int = DEFAULT_MAX_REVIEW_COMMENTS,
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
    max_comment_chars: int = DEFAULT_MAX_COMMENT_CHARS,
) -> list[dict[str, Any]]:
    github_records = collect_github_sources(
        github_repo,
        token=github_token,
        max_comments=max_comments,
        max_review_comments=max_review_comments,
        max_body_chars=max_body_chars,
        max_comment_chars=max_comment_chars,
    )
    hf_records = collect_hf_discussions(
        hf_space,
        token=hf_token,
        max_comments=max_comments,
        max_body_chars=max_body_chars,
        max_comment_chars=max_comment_chars,
    )
    return [*github_records, *hf_records]


def _record_for_llm(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "number": record.get("number"),
        "url": record.get("url"),
        "title": record.get("title"),
        "body": record.get("body"),
        "labels": record.get("labels") or [],
        "author": record.get("author"),
        "state": record.get("state"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "engagement": record.get("engagement") or {},
        "metadata": record.get("metadata") or {},
        "comments": record.get("comments") or [],
    }


def _classification_messages(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    schema = {
        "items": [
            {
                "id": "source id from input",
                "category": "feature | fix | other",
                "impact_score": "integer 1-5",
                "effort_score": "integer 1-5, where 1 is easiest",
                "confidence": "number 0-1",
                "user_problem": "one sentence",
                "recommended_action": "one sentence",
                "evidence": ["short evidence strings tied to source content"],
                "related_source_ids": ["optional related source ids"],
            }
        ]
    }
    return [
        {"role": "system", "content": PM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify each backlog item. Use only the provided evidence. "
                "Return JSON matching this schema:\n"
                f"{json.dumps(schema, indent=2)}\n\n"
                "Backlog items:\n"
                f"{json.dumps(batch, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _synthesis_messages(
    records: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> list[dict[str, str]]:
    source_index = [
        {
            "id": record.get("id"),
            "source": record.get("source"),
            "url": record.get("url"),
            "title": record.get("title"),
            "labels": record.get("labels") or [],
            "metadata": record.get("metadata") or {},
        }
        for record in records
    ]
    schema = {
        "summary": "short executive summary",
        "highest_impact_next": [
            {
                "rank": 1,
                "title": "recommendation title",
                "category": "feature | fix",
                "recommendation": "what to implement/review next",
                "impact_score": "integer 1-5",
                "effort_score": "integer 1-5, where 1 is easiest",
                "confidence": "number 0-1",
                "source_ids": ["source ids"],
                "source_urls": ["source URLs"],
                "rationale": "why this is high impact",
                "next_action": "concrete next action",
            }
        ],
        "features": [],
        "fixes": [],
        "other": [],
        "clusters": [
            {
                "title": "cluster title",
                "category": "feature | fix | other",
                "source_ids": ["source ids"],
                "summary": "shared user problem",
            }
        ],
    }
    return [
        {"role": "system", "content": PM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Synthesize the item-level classifications into a ranked PM "
                "implementation plan. Cluster duplicates and related requests. "
                "Keep features and fixes separate. If an open PR addresses a "
                "high-impact item, recommend review/merge/fix-forward instead "
                "of reimplementation. Return JSON matching this schema:\n"
                f"{json.dumps(schema, indent=2)}\n\n"
                "Source index:\n"
                f"{json.dumps(source_index, ensure_ascii=False, indent=2)}\n\n"
                "Item classifications:\n"
                f"{json.dumps(classifications, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _extract_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("LLM response did not contain valid JSON")


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        choice = response["choices"][0]
        message = choice.get("message") or {}
        return message.get("content") or ""
    choice = response.choices[0]
    return choice.message.content or ""


async def _call_json_llm(
    messages: list[dict[str, str]],
    llm_params: dict[str, Any],
    *,
    completion_func: Callable[..., Any] | None = None,
    max_completion_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    retries: int = 1,
) -> Any:
    if completion_func is None:
        from litellm import acompletion

        completion_func = acompletion

    attempt_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        response = await completion_func(
            messages=attempt_messages,
            max_completion_tokens=max_completion_tokens,
            temperature=0.2,
            **llm_params,
        )
        content = _response_content(response)
        try:
            return _extract_json_object(content)
        except ValueError as exc:
            last_error = exc
            if attempt >= retries:
                break
            attempt_messages = [
                *messages,
                {"role": "assistant", "content": _truncate_text(content, 2000)},
                {
                    "role": "user",
                    "content": (
                        "The previous response was not valid JSON. Return the "
                        "same answer again as a single valid JSON object only."
                    ),
                },
            ]
    raise ValueError("LLM failed to return valid JSON after retry") from last_error


def _default_classification(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "category": "other",
        "impact_score": 1,
        "effort_score": 3,
        "confidence": 0,
        "user_problem": "No model classification returned.",
        "recommended_action": "Triage manually.",
        "evidence": [],
        "related_source_ids": [],
    }


def _normalize_classifications(
    payload: Any, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        items = []
    by_id = {
        str(item.get("id")): item
        for item in items
        if isinstance(item, dict) and item.get("id") is not None
    }
    normalized: list[dict[str, Any]] = []
    for record in batch:
        item = dict(by_id.get(str(record.get("id"))) or _default_classification(record))
        item["id"] = record.get("id")
        item.setdefault("category", "other")
        item.setdefault("impact_score", 1)
        item.setdefault("effort_score", 3)
        item.setdefault("confidence", 0)
        item.setdefault("evidence", [])
        item.setdefault("related_source_ids", [])
        item.setdefault("source_url", record.get("url"))
        item.setdefault("source_title", record.get("title"))
        normalized.append(item)
    return normalized


async def classify_records(
    records: list[dict[str, Any]],
    llm_params: dict[str, Any],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_completion_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    completion_func: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    classifications: list[dict[str, Any]] = []
    compact_records = [_record_for_llm(record) for record in records]
    for start in range(0, len(compact_records), max(1, batch_size)):
        batch = compact_records[start : start + max(1, batch_size)]
        logger.info(
            "Classifying backlog batch %d-%d of %d",
            start + 1,
            start + len(batch),
            len(compact_records),
        )
        payload = await _call_json_llm(
            _classification_messages(batch),
            llm_params,
            completion_func=completion_func,
            max_completion_tokens=max_completion_tokens,
            retries=1,
        )
        classifications.extend(_normalize_classifications(payload, batch))
    return classifications


def _empty_ranking() -> dict[str, Any]:
    return {
        "summary": "No open backlog items were found.",
        "highest_impact_next": [],
        "features": [],
        "fixes": [],
        "other": [],
        "clusters": [],
        "classifications": [],
    }


def _normalize_ranking(payload: Any) -> dict[str, Any]:
    ranking = dict(payload) if isinstance(payload, dict) else {}
    ranking.setdefault("summary", "")
    for key in ("highest_impact_next", "features", "fixes", "other", "clusters"):
        if not isinstance(ranking.get(key), list):
            ranking[key] = []
    return ranking


async def synthesize_ranking(
    records: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    llm_params: dict[str, Any],
    *,
    max_completion_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    completion_func: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if not records:
        return _empty_ranking()

    payload = await _call_json_llm(
        _synthesis_messages(records, classifications),
        llm_params,
        completion_func=completion_func,
        max_completion_tokens=max_completion_tokens,
        retries=1,
    )
    ranking = _normalize_ranking(payload)
    ranking["classifications"] = classifications
    return ranking


async def prioritize_records(
    records: list[dict[str, Any]],
    model: str,
    *,
    reasoning_effort: str | None = "high",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_completion_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    completion_func: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if not records:
        return _empty_ranking()

    from agent.core.llm_params import _resolve_llm_params

    llm_params = _resolve_llm_params(model, reasoning_effort=reasoning_effort)
    classifications = await classify_records(
        records,
        llm_params,
        batch_size=batch_size,
        max_completion_tokens=max_completion_tokens,
        completion_func=completion_func,
    )
    return await synthesize_ranking(
        records,
        classifications,
        llm_params,
        max_completion_tokens=max_completion_tokens,
        completion_func=completion_func,
    )


def _source_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("id")): record for record in records if record.get("id")}


def _source_links(item: dict[str, Any], records_by_id: dict[str, dict[str, Any]]) -> str:
    ids = item.get("source_ids") or item.get("related_source_ids") or []
    links: list[str] = []
    known_urls = {record.get("url") for record in records_by_id.values()}
    for source_id in ids:
        record = records_by_id.get(str(source_id))
        url = record.get("url") if record else None
        if url:
            links.append(f"[{source_id}]({url})")
        else:
            links.append(str(source_id))
    for url in item.get("source_urls") or []:
        if url and url not in known_urls:
            links.append(f"[source]({url})")
    return ", ".join(links) if links else "No source cited"


def _score_text(item: dict[str, Any]) -> str:
    bits = []
    if item.get("impact_score") is not None:
        bits.append(f"impact {item.get('impact_score')}/5")
    if item.get("effort_score") is not None:
        bits.append(f"effort {item.get('effort_score')}/5")
    if item.get("confidence") is not None:
        bits.append(f"confidence {item.get('confidence')}")
    return ", ".join(bits)


def _render_recommendations(
    title: str,
    items: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    lines = [f"## {title}"]
    if not items:
        lines.append("")
        lines.append("No items.")
        return lines

    for index, item in enumerate(items, start=1):
        heading = item.get("title") or item.get("recommendation") or "Untitled"
        score = _score_text(item)
        suffix = f" ({score})" if score else ""
        lines.append("")
        lines.append(f"{index}. **{heading}**{suffix}")
        if item.get("recommendation"):
            lines.append(f"   - Recommendation: {item['recommendation']}")
        if item.get("rationale"):
            lines.append(f"   - Rationale: {item['rationale']}")
        if item.get("next_action"):
            lines.append(f"   - Next action: {item['next_action']}")
        lines.append(f"   - Sources: {_source_links(item, records_by_id)}")
    return lines


def render_markdown_report(
    ranking: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
    model: str | None = None,
) -> str:
    records_by_id = _source_lookup(records)
    source_counts: dict[str, int] = {}
    for record in records:
        source = str(record.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    lines = ["# ML Intern Backlog Prioritization", ""]
    if generated_at:
        lines.append(f"Generated: {generated_at}")
    if model:
        lines.append(f"Model: `{model}`")
    if generated_at or model:
        lines.append("")
    lines.append(
        "Sources: "
        + ", ".join(f"{name}={count}" for name, count in sorted(source_counts.items()))
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(ranking.get("summary") or "No summary returned.")
    lines.append("")

    lines.extend(
        _render_recommendations(
            "Highest Impact Next",
            ranking.get("highest_impact_next") or [],
            records_by_id,
        )
    )
    lines.append("")
    lines.extend(
        _render_recommendations("Features", ranking.get("features") or [], records_by_id)
    )
    lines.append("")
    lines.extend(
        _render_recommendations("Fixes", ranking.get("fixes") or [], records_by_id)
    )

    other = ranking.get("other") or []
    if other:
        lines.append("")
        lines.extend(_render_recommendations("Other / Watchlist", other, records_by_id))

    clusters = ranking.get("clusters") or []
    if clusters:
        lines.append("")
        lines.append("## Clusters")
        for cluster in clusters:
            lines.append("")
            lines.append(f"- **{cluster.get('title', 'Untitled')}**")
            if cluster.get("summary"):
                lines.append(f"  - Summary: {cluster['summary']}")
            lines.append(f"  - Sources: {_source_links(cluster, records_by_id)}")

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    output_dir: Path,
    *,
    sources: list[dict[str, Any]],
    ranking: dict[str, Any],
    report: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ranking.json").write_text(
        json.dumps(ranking, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    model = resolve_model(args.model, args.config)
    output_dir = resolve_output_dir(args.output_dir)
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")
    hf_token = resolve_hf_token(args.hf_token)

    logger.info("Collecting GitHub and Hugging Face backlog sources")
    sources = collect_sources(
        args.github_repo,
        args.hf_space,
        github_token=github_token,
        hf_token=hf_token,
        max_comments=args.max_comments,
        max_review_comments=args.max_review_comments,
        max_body_chars=args.max_body_chars,
        max_comment_chars=args.max_comment_chars,
    )
    logger.info("Collected %d backlog items", len(sources))

    generated_at = utc_now().isoformat()
    ranking = await prioritize_records(
        sources,
        model,
        reasoning_effort=args.reasoning_effort,
        batch_size=args.batch_size,
        max_completion_tokens=args.max_output_tokens,
    )
    ranking["generated_at"] = generated_at
    ranking["model"] = model
    ranking["source_counts"] = {
        source: sum(
            1 for record in sources if str(record.get("source") or "unknown") == source
        )
        for source in sorted(
            {str(record.get("source") or "unknown") for record in sources}
        )
    }

    report = render_markdown_report(
        ranking,
        sources,
        generated_at=generated_at,
        model=model,
    )
    write_outputs(output_dir, sources=sources, ranking=ranking, report=report)
    print(f"Wrote backlog prioritization to {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
