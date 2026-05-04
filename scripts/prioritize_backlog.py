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
import subprocess
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
DEFAULT_MAX_OUTPUT_TOKENS = 12000
DEFAULT_RESOLUTION_REF = "main"
DEFAULT_RESOLUTION_LOG_COMMITS = 500
DEFAULT_GITHUB_ISSUE_BODY_CHARS = 60000

logger = logging.getLogger("prioritize_backlog")

PM_SYSTEM_PROMPT = """You are a senior product manager for ML Intern.

Your job is to turn messy public feedback into a pragmatic implementation
priority list. Optimize for:
- user impact and blocked workflows
- evidence of repeated demand or engagement
- recency and severity
- PR readiness and whether an open PR should be reviewed/merged/fixed forward
- resolved-in-main signals from the local codebase check
- implementation effort, risk, and strategic fit for ML Intern

Separate user-facing features from bug fixes. Treat open PRs as possible
ready-made implementations rather than duplicate feature requests. Every
recommendation must cite source ids and/or source URLs from the input.
If an item has a high-confidence resolved-in-main signal, recommend closure
instead of implementation.

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
        "--resolution-ref",
        default=DEFAULT_RESOLUTION_REF,
        help="Git ref used to check whether open items are already resolved.",
    )
    ap.add_argument(
        "--resolution-log-commits",
        type=int,
        default=DEFAULT_RESOLUTION_LOG_COMMITS,
        help="Number of commits on --resolution-ref to scan for closure signals.",
    )
    ap.add_argument(
        "--skip-resolution-check",
        action="store_true",
        help="Skip local resolved-in-main checks before the LLM pass.",
    )
    ap.add_argument(
        "--skip-pr-patch-check",
        action="store_true",
        help="Skip PR patch-id comparison against --resolution-ref history.",
    )
    ap.add_argument(
        "--create-github-issue",
        action="store_true",
        help="Post the generated Markdown report as a new GitHub issue.",
    )
    ap.add_argument(
        "--github-issue-title",
        default=None,
        help="Title for --create-github-issue. Defaults to a dated report title.",
    )
    ap.add_argument(
        "--github-issue-label",
        action="append",
        default=[],
        help="Label to add to the created issue. Repeat or pass comma-separated labels.",
    )
    ap.add_argument(
        "--github-issue-body-chars",
        type=int,
        default=DEFAULT_GITHUB_ISSUE_BODY_CHARS,
        help="Maximum report body characters to send to GitHub.",
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
        "Content-Type": "application/json",
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
            "base_sha": base.get("sha"),
            "head": head.get("ref"),
            "head_sha": head.get("sha"),
            "patch_url": pr_details.get("patch_url"),
            "diff_url": pr_details.get("diff_url"),
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


def _git(
    args: list[str],
    *,
    repo_root: Path = PROJECT_ROOT,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=check,
    )


def _git_ref_sha(ref: str, *, repo_root: Path = PROJECT_ROOT) -> str:
    return _git(["rev-parse", "--verify", ref], repo_root=repo_root).stdout.strip()


def _git_log_entries(
    ref: str,
    *,
    repo_root: Path = PROJECT_ROOT,
    max_commits: int = DEFAULT_RESOLUTION_LOG_COMMITS,
) -> list[dict[str, str]]:
    fmt = "%H%x1f%s%x1f%b%x1e"
    output = _git(
        ["log", f"--max-count={max_commits}", f"--format={fmt}", ref],
        repo_root=repo_root,
    ).stdout
    entries: list[dict[str, str]] = []
    for raw in output.strip("\x1e\n").split("\x1e"):
        if not raw.strip():
            continue
        parts = raw.strip("\n").split("\x1f", 2)
        if len(parts) != 3:
            continue
        commit, subject, body = parts
        entries.append({"commit": commit.strip(), "subject": subject, "body": body})
    return entries


def _git_patch_ids_for_ref(
    ref: str,
    *,
    repo_root: Path = PROJECT_ROOT,
    max_commits: int = DEFAULT_RESOLUTION_LOG_COMMITS,
) -> dict[str, str]:
    log = _git(
        ["log", "--patch", f"--max-count={max_commits}", "--format=medium", ref],
        repo_root=repo_root,
    )
    patch_ids = _git(
        ["patch-id", "--stable"],
        repo_root=repo_root,
        input_text=log.stdout,
        check=False,
    )
    out: dict[str, str] = {}
    for line in patch_ids.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def _patch_id_for_text(
    patch_text: str,
    *,
    repo_root: Path = PROJECT_ROOT,
) -> str | None:
    result = _git(
        ["patch-id", "--stable"],
        repo_root=repo_root,
        input_text=patch_text,
        check=False,
    )
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            return parts[0]
    return None


def _record_text_for_refs(record: dict[str, Any]) -> str:
    pieces = [
        str(record.get("id") or ""),
        str(record.get("url") or ""),
        str(record.get("title") or ""),
        str(record.get("body") or ""),
    ]
    for comment in record.get("comments") or []:
        pieces.append(str(comment.get("url") or ""))
        pieces.append(str(comment.get("body") or ""))
    return "\n".join(pieces)


def _repo_regex(repo: str) -> str:
    return re.escape(repo).replace("/", r"[/]")


def _commit_text(commit: dict[str, str]) -> str:
    return f"{commit.get('subject', '')}\n{commit.get('body', '')}"


def _commit_evidence(
    commit: dict[str, str],
    detail: str,
) -> dict[str, str]:
    return {
        "kind": "commit",
        "commit": commit.get("commit", "")[:12],
        "subject": commit.get("subject", ""),
        "detail": detail,
    }


def _record_evidence(record: dict[str, Any], detail: str) -> dict[str, str]:
    return {
        "kind": "source_link",
        "source_id": str(record.get("id") or ""),
        "title": str(record.get("title") or ""),
        "detail": detail,
    }


def _commit_mentions_pr(
    text: str,
    pr_number: int,
    *,
    github_repo: str,
) -> bool:
    repo = _repo_regex(github_repo)
    patterns = [
        rf"\(#{pr_number}\)",
        rf"\bPR\s*#{pr_number}\b",
        rf"\bpull\s+request\s*#{pr_number}\b",
        rf"\bpull\s*/\s*{pr_number}\b",
        rf"github\.com[:/]{repo}/pull/{pr_number}\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _commit_closes_record(
    text: str,
    record: dict[str, Any],
    *,
    github_repo: str,
) -> bool:
    source = record.get("source")
    number = record.get("number")
    if not isinstance(number, int):
        return False
    close = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
    repo = _repo_regex(github_repo)
    if source == "github_issue":
        patterns = [
            rf"\b{close}\s+(?:{repo})?#\s*{number}\b",
            rf"\b{close}\s+https://github\.com[:/]{repo}/issues/{number}\b",
        ]
        return any(
            re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns
        )
    if source == "hf_discussion":
        url = re.escape(str(record.get("url") or ""))
        return bool(url and re.search(rf"\b{close}\b.*{url}", text, re.IGNORECASE))
    return False


def _linked_pr_numbers(text: str, *, github_repo: str) -> set[int]:
    repo = _repo_regex(github_repo)
    patterns = [
        rf"github\.com[:/]{repo}/pull/(\d+)\b",
        r"\bPR\s*#(\d+)\b",
        r"\bpull\s+request\s*#(\d+)\b",
    ]
    numbers: set[int] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            numbers.add(int(match.group(1)))
    return numbers


def _new_resolution(checked_ref: str, checked_sha: str) -> dict[str, Any]:
    return {
        "checked_ref": checked_ref,
        "checked_sha": checked_sha,
        "status": "unresolved",
        "can_close": False,
        "confidence": 0.0,
        "reasons": [],
        "evidence": [],
    }


def _mark_resolution(
    resolution: dict[str, Any],
    *,
    status: str,
    confidence: float,
    reason: str,
    evidence: list[dict[str, Any]],
) -> None:
    if confidence < float(resolution.get("confidence") or 0):
        return
    resolution["status"] = status
    resolution["can_close"] = status in {"resolved", "likely_resolved"}
    resolution["confidence"] = confidence
    resolution["reasons"] = [reason]
    resolution["evidence"] = evidence


def apply_resolution_checks(
    records: list[dict[str, Any]],
    *,
    checked_ref: str,
    checked_sha: str,
    commits: list[dict[str, str]],
    github_repo: str,
    pr_patch_matches: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pr_patch_matches = pr_patch_matches or {}
    resolved_prs: dict[int, list[dict[str, Any]]] = {}
    direct_closures: dict[str, list[dict[str, Any]]] = {}

    for commit in commits:
        text = _commit_text(commit)
        for record in records:
            source_id = str(record.get("id") or "")
            number = record.get("number")
            if record.get("source") == "github_pr" and isinstance(number, int):
                if _commit_mentions_pr(text, number, github_repo=github_repo):
                    resolved_prs.setdefault(number, []).append(
                        _commit_evidence(commit, f"main history references PR #{number}")
                    )
            elif _commit_closes_record(text, record, github_repo=github_repo):
                direct_closures.setdefault(source_id, []).append(
                    _commit_evidence(commit, "main history contains a closing reference")
                )

    for pr_number, evidence in pr_patch_matches.items():
        resolved_prs.setdefault(pr_number, []).append(evidence)

    checked: list[dict[str, Any]] = []
    for record in records:
        out = dict(record)
        resolution = _new_resolution(checked_ref, checked_sha)
        source_id = str(record.get("id") or "")
        number = record.get("number")

        if record.get("source") == "github_pr" and isinstance(number, int):
            if evidences := resolved_prs.get(number):
                has_patch = any(ev.get("kind") == "patch_id" for ev in evidences)
                _mark_resolution(
                    resolution,
                    status="resolved",
                    confidence=0.98 if has_patch else 0.95,
                    reason=f"PR #{number} appears to already be present on {checked_ref}.",
                    evidence=evidences,
                )
        elif evidences := direct_closures.get(source_id):
            _mark_resolution(
                resolution,
                status="likely_resolved",
                confidence=0.9,
                reason=f"{source_id} has a closing reference in {checked_ref} history.",
                evidence=evidences,
            )
        else:
            linked = sorted(
                _linked_pr_numbers(_record_text_for_refs(record), github_repo=github_repo)
                & set(resolved_prs)
            )
            if linked:
                evidences = [
                    _record_evidence(
                        record,
                        "source text links to PR(s) already present on main: "
                        + ", ".join(f"#{num}" for num in linked),
                    )
                ]
                for pr_number in linked:
                    evidences.extend(resolved_prs[pr_number])
                _mark_resolution(
                    resolution,
                    status="likely_resolved",
                    confidence=0.85,
                    reason=(
                        f"{source_id} links to PR(s) already present on {checked_ref}: "
                        + ", ".join(f"#{num}" for num in linked)
                    ),
                    evidence=evidences,
                )

        out["resolution"] = resolution
        checked.append(out)
    return checked


def _fetch_pr_patch_matches(
    records: list[dict[str, Any]],
    *,
    github_token: str | None,
    main_patch_ids: dict[str, str],
    client: Any | None = None,
) -> dict[int, dict[str, Any]]:
    if not main_patch_ids:
        return {}

    headers = _github_headers(github_token)
    headers["Accept"] = "application/vnd.github.patch"
    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=30.0, follow_redirects=True)

    matches: dict[int, dict[str, Any]] = {}
    try:
        for record in records:
            if record.get("source") != "github_pr":
                continue
            number = record.get("number")
            patch_url = (record.get("metadata") or {}).get("patch_url")
            if not isinstance(number, int) or not patch_url:
                continue
            try:
                response = client.get(patch_url, headers=headers)
                _raise_for_status(response)
                patch_id = _patch_id_for_text(response.text)
            except Exception as exc:
                logger.debug("patch-id check failed for PR #%s: %s", number, exc)
                continue
            if patch_id and patch_id in main_patch_ids:
                matches[number] = {
                    "kind": "patch_id",
                    "patch_id": patch_id,
                    "commit": main_patch_ids[patch_id][:12],
                    "detail": "PR patch-id matches a commit already in main history",
                }
    finally:
        if close_client and hasattr(client, "close"):
            client.close()
    return matches


def add_resolution_checks(
    records: list[dict[str, Any]],
    *,
    checked_ref: str = DEFAULT_RESOLUTION_REF,
    github_repo: str = DEFAULT_GITHUB_REPO,
    github_token: str | None = None,
    max_commits: int = DEFAULT_RESOLUTION_LOG_COMMITS,
    include_patch_check: bool = True,
) -> list[dict[str, Any]]:
    checked_sha = _git_ref_sha(checked_ref)
    commits = _git_log_entries(checked_ref, max_commits=max_commits)
    pr_patch_matches: dict[int, dict[str, Any]] = {}
    if include_patch_check:
        main_patch_ids = _git_patch_ids_for_ref(checked_ref, max_commits=max_commits)
        pr_patch_matches = _fetch_pr_patch_matches(
            records,
            github_token=github_token,
            main_patch_ids=main_patch_ids,
        )
    return apply_resolution_checks(
        records,
        checked_ref=checked_ref,
        checked_sha=checked_sha,
        commits=commits,
        github_repo=github_repo,
        pr_patch_matches=pr_patch_matches,
    )


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
        "resolution": record.get("resolution") or {},
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
                "resolved_in_main": "yes | no | uncertain",
                "close_recommendation": "if resolved, why it can be closed",
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
                "Pay special attention to each item's resolution field, which "
                "contains deterministic checks against the local main commit. "
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
            "resolution": record.get("resolution") or {},
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
        "can_be_closed": [
            {
                "title": "item title",
                "source_ids": ["source ids"],
                "source_urls": ["source URLs"],
                "reason": "why main already resolves it",
                "confidence": "number 0-1",
                "close_action": "specific closure action",
            }
        ],
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
                "of reimplementation unless its resolution field says it is "
                "already present on main. Create can_be_closed entries only "
                "for items with strong resolved-in-main evidence. "
                "Keep the output concise: at most 8 highest_impact_next "
                "items, 12 features, 12 fixes, 12 can_be_closed items, "
                "6 other items, and 12 clusters. Keep strings short enough "
                "for a PM scan. If the output budget is tight, omit "
                "lower-priority entries but return a complete JSON object. "
                "Return JSON matching this schema:\n"
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


def _temperature_for_params(llm_params: dict[str, Any]) -> float:
    # Anthropic requires temperature=1 when adaptive/extended thinking is active.
    if llm_params.get("thinking") or llm_params.get("output_config"):
        return 1.0
    return 0.2


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
            temperature=_temperature_for_params(llm_params),
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
        "resolved_in_main": "uncertain",
        "close_recommendation": "",
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
        item.setdefault("resolved_in_main", "uncertain")
        item.setdefault("close_recommendation", "")
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
        "can_be_closed": [],
        "other": [],
        "clusters": [],
        "classifications": [],
    }


def _normalize_ranking(payload: Any) -> dict[str, Any]:
    ranking = dict(payload) if isinstance(payload, dict) else {}
    ranking.setdefault("summary", "")
    for key in (
        "highest_impact_next",
        "features",
        "fixes",
        "can_be_closed",
        "other",
        "clusters",
    ):
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
        retries=2,
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


def _local_can_be_closed(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        resolution = record.get("resolution") or {}
        if not resolution.get("can_close"):
            continue
        source_id = record.get("id")
        if not source_id:
            continue
        checked_ref = resolution.get("checked_ref") or DEFAULT_RESOLUTION_REF
        checked_sha = str(resolution.get("checked_sha") or "")[:12]
        source = str(record.get("source") or "item").replace("_", " ")
        if record.get("source") == "github_pr":
            action = (
                f"Close the PR as already present on {checked_ref}"
                + (f" ({checked_sha})" if checked_sha else "")
                + " after maintainer confirmation."
            )
        else:
            action = (
                f"Close the {source} as resolved on {checked_ref}"
                + (f" ({checked_sha})" if checked_sha else "")
                + " after maintainer confirmation."
            )
        items.append(
            {
                "title": record.get("title") or str(source_id),
                "source_ids": [source_id],
                "source_urls": [record.get("url")] if record.get("url") else [],
                "reason": "; ".join(resolution.get("reasons") or [])
                or "Local main contains a high-confidence resolution signal.",
                "confidence": resolution.get("confidence", 0),
                "close_action": action,
            }
        )
    return items


def merge_can_be_closed(
    ranking: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    merged = dict(ranking)
    existing = [
        item for item in merged.get("can_be_closed") or [] if isinstance(item, dict)
    ]
    seen = {
        tuple(sorted(str(source_id) for source_id in item.get("source_ids") or []))
        for item in existing
    }
    for item in _local_can_be_closed(records):
        key = tuple(sorted(str(source_id) for source_id in item.get("source_ids") or []))
        if key in seen:
            continue
        existing.append(item)
        seen.add(key)
    existing.sort(key=lambda item: float(item.get("confidence") or 0), reverse=True)
    merged["can_be_closed"] = existing
    return merged


def _render_can_be_closed(
    items: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    lines = ["## Can Be Closed"]
    if not items:
        lines.append("")
        lines.append("No high-confidence resolved-in-main candidates found.")
        return lines

    for index, item in enumerate(items, start=1):
        title = item.get("title") or "Untitled"
        confidence = item.get("confidence")
        suffix = f" (confidence {confidence})" if confidence is not None else ""
        lines.append("")
        lines.append(f"{index}. **{title}**{suffix}")
        if item.get("reason"):
            lines.append(f"   - Reason: {item['reason']}")
        if item.get("close_action"):
            lines.append(f"   - Close action: {item['close_action']}")
        lines.append(f"   - Sources: {_source_links(item, records_by_id)}")
    return lines


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
        _render_can_be_closed(ranking.get("can_be_closed") or [], records_by_id)
    )
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


def default_github_issue_title(generated_at: str) -> str:
    try:
        date_text = datetime.fromisoformat(generated_at).date().isoformat()
    except ValueError:
        date_text = generated_at[:10] or "latest"
    return f"ML Intern backlog prioritization report - {date_text}"


def _github_issue_labels(raw_labels: list[str]) -> list[str]:
    labels: list[str] = []
    for raw in raw_labels:
        for label in raw.split(","):
            cleaned = label.strip()
            if cleaned and cleaned not in labels:
                labels.append(cleaned)
    return labels


def _github_issue_body(report: str, *, max_chars: int) -> str:
    footer = (
        "\n\n---\n"
        "_Generated by `uv run python scripts/prioritize_backlog.py`._\n"
    )
    body = report.rstrip() + footer
    if max_chars <= 0 or len(body) <= max_chars:
        return body

    truncation = (
        "\n\n---\n"
        "_Report truncated to fit the configured GitHub issue body limit. "
        "See the local `report.md` output for the complete version._\n"
    )
    if len(truncation) >= max_chars:
        return truncation[:max_chars]
    return body[: max(0, max_chars - len(truncation))].rstrip() + truncation


def create_github_report_issue(
    repo: str,
    *,
    title: str,
    report: str,
    token: str | None,
    labels: list[str] | None = None,
    max_body_chars: int = DEFAULT_GITHUB_ISSUE_BODY_CHARS,
    client: Any | None = None,
) -> dict[str, Any]:
    if not token:
        raise ValueError(
            "Creating a GitHub issue requires --github-token or GITHUB_TOKEN."
        )

    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=30.0, follow_redirects=True)

    payload: dict[str, Any] = {
        "title": title,
        "body": _github_issue_body(report, max_chars=max_body_chars),
    }
    cleaned_labels = _github_issue_labels(labels or [])
    if cleaned_labels:
        payload["labels"] = cleaned_labels

    try:
        response = client.post(
            f"{GITHUB_API}/repos/{repo}/issues",
            headers=_github_headers(token),
            json=payload,
        )
        _raise_for_status(response)
        data = response.json()
    finally:
        if close_client and hasattr(client, "close"):
            client.close()

    return {
        "number": data.get("number"),
        "url": data.get("html_url"),
        "api_url": data.get("url"),
        "title": data.get("title") or title,
    }


def append_published_issue_section(report: str, issue: dict[str, Any]) -> str:
    number = issue.get("number")
    title = f"#{number}" if number else "GitHub issue"
    url = issue.get("url") or issue.get("api_url") or ""
    if not url:
        return report
    return report.rstrip() + f"\n\n## Published GitHub Issue\n\n- [{title}]({url})\n"


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
    if not args.skip_resolution_check:
        logger.info(
            "Checking whether open items are already resolved on %s",
            args.resolution_ref,
        )
        sources = add_resolution_checks(
            sources,
            checked_ref=args.resolution_ref,
            github_repo=args.github_repo,
            github_token=github_token,
            max_commits=args.resolution_log_commits,
            include_patch_check=not args.skip_pr_patch_check,
        )
        can_close = sum(
            1 for record in sources if (record.get("resolution") or {}).get("can_close")
        )
        logger.info("Found %d resolved-in-main closure candidates", can_close)

    generated_at = utc_now().isoformat()
    ranking = await prioritize_records(
        sources,
        model,
        reasoning_effort=args.reasoning_effort,
        batch_size=args.batch_size,
        max_completion_tokens=args.max_output_tokens,
    )
    ranking = merge_can_be_closed(ranking, sources)
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
    if args.create_github_issue:
        title = args.github_issue_title or default_github_issue_title(generated_at)
        issue = create_github_report_issue(
            args.github_repo,
            title=title,
            report=report,
            token=github_token,
            labels=args.github_issue_label,
            max_body_chars=args.github_issue_body_chars,
        )
        ranking["github_issue"] = issue
        report = append_published_issue_section(report, issue)
        write_outputs(output_dir, sources=sources, ranking=ranking, report=report)
        print(f"Created GitHub issue #{issue.get('number')}: {issue.get('url')}")
    print(f"Wrote backlog prioritization to {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
