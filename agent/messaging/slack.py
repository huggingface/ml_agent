import json
import re

import httpx

from agent.messaging.base import (
    NotificationError,
    NotificationProvider,
    RetryableNotificationError,
)
from agent.messaging.models import (
    NotificationRequest,
    NotificationResult,
    SlackDestinationConfig,
)

_SEVERITY_PREFIX = {
    "info": "[INFO]",
    "success": "[SUCCESS]",
    "warning": "[WARNING]",
    "error": "[ERROR]",
}


def _format_slack_mrkdwn(content: str) -> str:
    """Convert common Markdown constructs to Slack's mrkdwn syntax."""
    if not content:
        return content

    placeholders: dict[str, str] = {}
    placeholder_index = 0

    def placeholder(value: str) -> str:
        nonlocal placeholder_index
        key = f"\x00SLACK{placeholder_index}\x00"
        placeholder_index += 1
        placeholders[key] = value
        return key

    text = content

    # Protect code before any formatting conversion. Slack's mrkdwn ignores
    # formatting inside backticks, so these regions should stay byte-for-byte.
    text = re.sub(
        r"(```(?:[^\n]*\n)?[\s\S]*?```)",
        lambda match: placeholder(match.group(0)),
        text,
    )
    text = re.sub(r"(`[^`\n]+`)", lambda match: placeholder(match.group(0)), text)

    def convert_markdown_link(match: re.Match[str]) -> str:
        label = match.group(1)
        url = match.group(2).strip()
        if url.startswith("<") and url.endswith(">"):
            url = url[1:-1].strip()
        return placeholder(f"<{url}|{label}>")

    text = re.sub(
        r"\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
        convert_markdown_link,
        text,
    )

    # Preserve existing Slack entities and manual mrkdwn links before escaping.
    text = re.sub(
        r"(<(?:[@#!]|(?:https?|mailto|tel):)[^>\n]+>)",
        lambda match: placeholder(match.group(1)),
        text,
    )
    text = re.sub(
        r"^(>+\s)",
        lambda match: placeholder(match.group(0)),
        text,
        flags=re.MULTILINE,
    )

    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def convert_header(match: re.Match[str]) -> str:
        header = match.group(1).strip()
        header = re.sub(r"\*\*(.+?)\*\*", r"\1", header)
        return placeholder(f"*{header}*")

    text = re.sub(r"^#{1,6}\s+(.+)$", convert_header, text, flags=re.MULTILINE)
    text = re.sub(
        r"\*\*\*(.+?)\*\*\*",
        lambda match: placeholder(f"*_{match.group(1)}_*"),
        text,
    )
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda match: placeholder(f"*{match.group(1)}*"),
        text,
    )
    text = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        lambda match: placeholder(f"_{match.group(1)}_"),
        text,
    )
    text = re.sub(
        r"~~(.+?)~~",
        lambda match: placeholder(f"~{match.group(1)}~"),
        text,
    )

    for key in reversed(placeholders):
        text = text.replace(key, placeholders[key])

    return text


def _format_text(request: NotificationRequest) -> str:
    lines: list[str] = []
    prefix = _SEVERITY_PREFIX[request.severity]
    if request.title:
        lines.append(f"{prefix} {request.title}")
    else:
        lines.append(prefix)
    lines.append(request.message)
    for key, value in request.metadata.items():
        lines.append(f"{key}: {value}")
    return _format_slack_mrkdwn("\n".join(lines))


class SlackProvider(NotificationProvider):
    provider_name = "slack"

    async def send(
        self,
        client: httpx.AsyncClient,
        destination_name: str,
        destination: SlackDestinationConfig,
        request: NotificationRequest,
    ) -> NotificationResult:
        payload = {
            "channel": destination.channel,
            "text": _format_text(request),
            "mrkdwn": True,
            "unfurl_links": False,
            "unfurl_media": False,
        }
        if destination.username:
            payload["username"] = destination.username
        if destination.icon_emoji:
            payload["icon_emoji"] = destination.icon_emoji

        try:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {destination.token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                content=json.dumps(payload),
            )
        except httpx.TimeoutException as exc:
            raise RetryableNotificationError("Slack request timed out") from exc
        except httpx.TransportError as exc:
            raise RetryableNotificationError("Slack transport error") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableNotificationError(f"Slack HTTP {response.status_code}")
        if response.status_code >= 400:
            raise NotificationError(f"Slack HTTP {response.status_code}")

        try:
            data = response.json()
        except ValueError as exc:
            raise RetryableNotificationError("Slack returned invalid JSON") from exc

        if not data.get("ok"):
            error = str(data.get("error") or "unknown_error")
            if error == "ratelimited":
                raise RetryableNotificationError(error)
            raise NotificationError(error)

        return NotificationResult(
            destination=destination_name,
            ok=True,
            provider=self.provider_name,
            external_id=str(data.get("ts") or ""),
            error=None,
        )
