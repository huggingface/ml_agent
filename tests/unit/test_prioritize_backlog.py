import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load():
    path = Path(__file__).parent.parent.parent / "scripts" / "prioritize_backlog.py"
    spec = importlib.util.spec_from_file_location("prioritize_backlog", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prioritize_backlog"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class FakeResponse:
    def __init__(self, data, headers=None, text=None):
        self._data = data
        self.headers = headers or {}
        self.text = text if text is not None else ""

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class FakeIssueClient:
    def __init__(self):
        self.posts = []
        self.closed = False

    def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "headers": headers or {}, "json": json or {}})
        return FakeResponse(
            {
                "number": 42,
                "html_url": "https://github.com/owner/repo/issues/42",
                "url": "https://api.github.com/repos/owner/repo/issues/42",
                "title": json["title"],
            }
        )

    def close(self):
        self.closed = True


class FakeGitHubClient:
    def __init__(self):
        self.requests = []

    def get(self, url, headers=None, params=None):
        self.requests.append((url, params or {}))
        page = (params or {}).get("page")

        if url == "https://api.github.com/repos/owner/repo/issues":
            if page == 1:
                return FakeResponse(
                    [
                        {
                            "number": 1,
                            "html_url": "https://github.com/owner/repo/issues/1",
                            "title": "Issue one",
                            "body": "broken",
                            "labels": [{"name": "bug"}],
                            "user": {"login": "alice"},
                            "state": "open",
                            "created_at": "2026-05-01T00:00:00Z",
                            "updated_at": "2026-05-02T00:00:00Z",
                            "comments": 1,
                            "comments_url": "https://api.github.test/issues/1/comments",
                        },
                        {
                            "number": 2,
                            "html_url": "https://github.com/owner/repo/pull/2",
                            "title": "PR two",
                            "body": "adds feature",
                            "labels": [{"name": "enhancement"}],
                            "user": {"login": "bob"},
                            "state": "open",
                            "created_at": "2026-05-01T00:00:00Z",
                            "updated_at": "2026-05-02T00:00:00Z",
                            "comments": 0,
                            "comments_url": "https://api.github.test/issues/2/comments",
                            "pull_request": {"url": "https://api.github.test/pulls/2"},
                        },
                    ],
                    headers={"link": '<https://api.github.test?page=2>; rel="next"'},
                )
            return FakeResponse(
                [
                    {
                        "number": 3,
                        "html_url": "https://github.com/owner/repo/issues/3",
                        "title": "Issue three",
                        "body": "request",
                        "labels": [],
                        "user": {"login": "carol"},
                        "state": "open",
                        "created_at": "2026-05-03T00:00:00Z",
                        "updated_at": "2026-05-03T00:00:00Z",
                        "comments": 0,
                        "comments_url": "https://api.github.test/issues/3/comments",
                    }
                ]
            )

        if url.endswith("/comments") and "/pulls/" not in url:
            return FakeResponse(
                [
                    {
                        "body": "comment",
                        "user": {"login": "dana"},
                        "created_at": "2026-05-02T00:00:00Z",
                        "html_url": "https://github.com/comment",
                    }
                ]
            )

        if url == "https://api.github.com/repos/owner/repo/pulls/2":
            return FakeResponse(
                {
                    "number": 2,
                    "html_url": "https://github.com/owner/repo/pull/2",
                    "title": "PR two",
                    "body": "adds feature",
                    "user": {"login": "bob"},
                    "state": "open",
                    "draft": False,
                    "base": {"ref": "main"},
                    "head": {"ref": "feature"},
                    "commits": 2,
                    "additions": 10,
                    "deletions": 3,
                    "changed_files": 2,
                    "review_comments": 0,
                }
            )

        if url in {
            "https://api.github.com/repos/owner/repo/pulls/2/comments",
            "https://api.github.com/repos/owner/repo/pulls/2/reviews",
        }:
            return FakeResponse([])

        raise AssertionError(f"unexpected URL: {url}")


def test_github_pagination_and_issue_pr_splitting():
    mod = _load()
    records = mod.collect_github_sources("owner/repo", client=FakeGitHubClient())

    assert [record["id"] for record in records] == [
        "github_issue#1",
        "github_pr#2",
        "github_issue#3",
    ]
    assert records[0]["source"] == "github_issue"
    assert records[1]["source"] == "github_pr"
    assert records[1]["metadata"]["base"] == "main"


def test_github_comment_cap_and_truncation():
    mod = _load()

    class CommentClient:
        def get(self, url, headers=None, params=None):
            assert url == "https://api.github.test/comments"
            return FakeResponse(
                [
                    {"body": "abcdef", "user": {"login": "one"}},
                    {"body": "second", "user": {"login": "two"}},
                ],
                headers={"link": '<https://api.github.test/comments?page=2>; rel="next"'},
            )

    comments = mod._fetch_github_comments(
        CommentClient(),
        "https://api.github.test/comments",
        {},
        max_comments=1,
        max_comment_chars=5,
    )

    assert len(comments) == 1
    assert comments[0]["author"] == "one"
    assert comments[0]["body"].endswith("[truncated]")


def test_hf_discussion_event_normalization():
    mod = _load()
    discussion = SimpleNamespace(
        num=7,
        repo_id="smolagents/ml-intern",
        repo_type="space",
        title="Space fails",
        status="open",
        author="alice",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    details = SimpleNamespace(
        title="Space fails",
        status="open",
        events=[
            SimpleNamespace(
                type="comment",
                content="Initial report",
                hidden=False,
                author="alice",
                created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                type="comment",
                content="Hidden moderation",
                hidden=True,
                author="mod",
                created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                type="comment",
                content="Maintainer reply",
                hidden=False,
                author="bob",
                created_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
            ),
            SimpleNamespace(type="status-change", new_status="open"),
        ],
    )

    record = mod.normalize_hf_discussion(discussion, details)

    assert record["id"] == "hf_discussion#7"
    assert record["url"] == (
        "https://huggingface.co/spaces/smolagents/ml-intern/discussions/7"
    )
    assert record["body"] == "Initial report"
    assert len(record["comments"]) == 1
    assert record["comments"][0]["body"] == "Maintainer reply"
    assert record["engagement"]["comments_count"] == 2


def test_resolution_check_marks_pr_and_linked_issue_as_closable():
    mod = _load()
    records = [
        {
            "id": "github_pr#2",
            "source": "github_pr",
            "number": 2,
            "url": "https://github.com/owner/repo/pull/2",
            "title": "Fix login",
            "body": "Fixes the login flow.",
            "comments": [],
        },
        {
            "id": "github_issue#1",
            "source": "github_issue",
            "number": 1,
            "url": "https://github.com/owner/repo/issues/1",
            "title": "Login broken",
            "body": "Fixed by PR #2.",
            "comments": [],
        },
        {
            "id": "github_issue#3",
            "source": "github_issue",
            "number": 3,
            "url": "https://github.com/owner/repo/issues/3",
            "title": "Direct issue",
            "body": "",
            "comments": [],
        },
    ]
    commits = [
        {
            "commit": "abcdef1234567890",
            "subject": "Fix login flow (#2)",
            "body": "Also fixes #3",
        }
    ]

    checked = mod.apply_resolution_checks(
        records,
        checked_ref="main",
        checked_sha="abcdef1234567890",
        commits=commits,
        github_repo="owner/repo",
    )

    by_id = {record["id"]: record for record in checked}
    assert by_id["github_pr#2"]["resolution"]["can_close"] is True
    assert by_id["github_pr#2"]["resolution"]["status"] == "resolved"
    assert by_id["github_issue#1"]["resolution"]["can_close"] is True
    assert by_id["github_issue#1"]["resolution"]["status"] == "likely_resolved"
    assert by_id["github_issue#3"]["resolution"]["can_close"] is True


def test_merge_can_be_closed_adds_local_resolution_candidates():
    mod = _load()
    records = [
        {
            "id": "github_pr#2",
            "source": "github_pr",
            "url": "https://github.com/owner/repo/pull/2",
            "title": "Fix login",
            "resolution": {
                "checked_ref": "main",
                "checked_sha": "abcdef1234567890",
                "status": "resolved",
                "can_close": True,
                "confidence": 0.95,
                "reasons": ["PR #2 appears to already be present on main."],
                "evidence": [],
            },
        }
    ]

    ranking = mod.merge_can_be_closed({"summary": "x"}, records)

    assert ranking["can_be_closed"][0]["source_ids"] == ["github_pr#2"]
    assert "already be present" in ranking["can_be_closed"][0]["reason"]


def test_fetch_pr_patch_matches_uses_patch_id(monkeypatch):
    mod = _load()
    records = [
        {
            "id": "github_pr#2",
            "source": "github_pr",
            "number": 2,
            "metadata": {"patch_url": "https://api.github.test/pr/2.patch"},
        }
    ]

    class PatchClient:
        def close(self):
            return None

        def get(self, url, headers=None):
            assert url == "https://api.github.test/pr/2.patch"
            assert headers["Accept"] == "application/vnd.github.patch"
            return FakeResponse({}, text="diff --git a/a b/a")

    monkeypatch.setattr(mod, "_patch_id_for_text", lambda _text: "patch-id")

    matches = mod._fetch_pr_patch_matches(
        records,
        github_token=None,
        main_patch_ids={"patch-id": "abcdef1234567890"},
        client=PatchClient(),
    )

    assert matches[2]["kind"] == "patch_id"
    assert matches[2]["commit"] == "abcdef123456"


def test_create_github_report_issue_posts_markdown_report():
    mod = _load()
    client = FakeIssueClient()

    issue = mod.create_github_report_issue(
        "owner/repo",
        title="Backlog report",
        report="# Report\n\nBody",
        token="gh-token",
        labels=["pm-report, backlog", "triage"],
        client=client,
    )

    assert issue["number"] == 42
    assert issue["url"] == "https://github.com/owner/repo/issues/42"
    assert client.closed is False
    post = client.posts[0]
    assert post["url"] == "https://api.github.com/repos/owner/repo/issues"
    assert post["headers"]["Authorization"] == "Bearer gh-token"
    assert post["json"]["title"] == "Backlog report"
    assert post["json"]["body"].startswith("# Report")
    assert "Generated by" in post["json"]["body"]
    assert post["json"]["labels"] == ["pm-report", "backlog", "triage"]


def test_create_github_report_issue_requires_token():
    mod = _load()

    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        mod.create_github_report_issue(
            "owner/repo",
            title="Backlog report",
            report="# Report",
            token=None,
            client=FakeIssueClient(),
        )


def test_github_issue_body_truncates_with_footer():
    mod = _load()
    body = mod._github_issue_body("abcdef" * 100, max_chars=120)

    assert len(body) <= 120
    assert "Report truncated" in body


def test_append_published_issue_section_adds_local_link():
    mod = _load()
    report = mod.append_published_issue_section(
        "# Report\n",
        {"number": 42, "url": "https://github.com/owner/repo/issues/42"},
    )

    assert "## Published GitHub Issue" in report
    assert "[#42](https://github.com/owner/repo/issues/42)" in report


@pytest.mark.asyncio
async def test_call_json_llm_retries_after_invalid_json():
    mod = _load()
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        content = "not json" if len(calls) == 1 else '{"ok": true}'
        return {"choices": [{"message": {"content": content}}]}

    result = await mod._call_json_llm(
        [{"role": "user", "content": "return json"}],
        {},
        completion_func=fake_completion,
        retries=1,
    )

    assert result == {"ok": True}
    assert len(calls) == 2
    assert "previous response was not valid JSON" in calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_call_json_llm_uses_temperature_one_for_thinking_params():
    mod = _load()
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    result = await mod._call_json_llm(
        [{"role": "user", "content": "return json"}],
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        completion_func=fake_completion,
        retries=0,
    )

    assert result == {"ok": True}
    assert calls[0]["temperature"] == 1.0


def test_render_markdown_report_from_sample_ranking():
    mod = _load()
    records = [
        {
            "id": "github_issue#1",
            "source": "github_issue",
            "url": "https://github.com/owner/repo/issues/1",
            "title": "Broken login",
        },
        {
            "id": "github_pr#2",
            "source": "github_pr",
            "url": "https://github.com/owner/repo/pull/2",
            "title": "Fix login",
        },
    ]
    ranking = {
        "summary": "Fix login first.",
        "can_be_closed": [
            {
                "title": "Fix login",
                "source_ids": ["github_pr#2"],
                "reason": "PR already landed on main.",
                "confidence": 0.95,
                "close_action": "Close duplicate PR.",
            }
        ],
        "highest_impact_next": [
            {
                "title": "Unblock login",
                "category": "fix",
                "recommendation": "Review and merge the existing PR.",
                "impact_score": 5,
                "effort_score": 1,
                "confidence": 0.9,
                "source_ids": ["github_issue#1", "github_pr#2"],
                "rationale": "It blocks onboarding.",
                "next_action": "Review PR #2.",
            }
        ],
        "features": [],
        "fixes": [],
    }

    report = mod.render_markdown_report(
        ranking,
        records,
        generated_at="2026-05-04T10:00:00+00:00",
        model="openai/gpt-5.5",
    )

    assert "# ML Intern Backlog Prioritization" in report
    assert "## Can Be Closed" in report
    assert "PR already landed on main." in report
    assert "## Highest Impact Next" in report
    assert "[github_issue#1](https://github.com/owner/repo/issues/1)" in report
    assert "Review and merge the existing PR." in report


def test_cli_defaults_without_live_network_or_llm():
    mod = _load()
    args = mod.parse_args([])
    out = mod.resolve_output_dir(
        None, now=datetime(2026, 5, 4, 12, 30, tzinfo=timezone.utc)
    )

    assert args.github_repo == "huggingface/ml-intern"
    assert args.hf_space == "smolagents/ml-intern"
    assert args.config == "configs/cli_agent_config.json"
    assert args.resolution_ref == "main"
    assert args.create_github_issue is False
    assert args.github_issue_label == []
    assert args.output_dir is None
    assert out.name == "20260504T123000Z"
    assert "scratch/backlog-prioritization" in str(out)
