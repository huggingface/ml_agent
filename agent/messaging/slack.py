import json

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
    return "\n".join(lines)


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
            raise RetryableNotificationError(
                f"Slack HTTP {response.status_code}"
            )
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
