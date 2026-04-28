from typing import Any

from agent.messaging.models import NotificationRequest

NOTIFY_TOOL_SPEC = {
    "name": "notify",
    "description": (
        "Send an out-of-band notification to configured messaging destinations. "
        "Use this only when the user explicitly asked for proactive notifications "
        "or when the task requires reporting progress outside the chat. "
        "Destinations must be named server-side configs such as 'slack.ops'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "destinations": {
                "type": "array",
                "description": "Named messaging destinations to notify.",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "message": {
                "type": "string",
                "description": "Main notification body.",
            },
            "title": {
                "type": "string",
                "description": "Optional short title line.",
            },
            "severity": {
                "type": "string",
                "enum": ["info", "success", "warning", "error"],
                "description": "Notification severity label.",
            },
        },
        "required": ["destinations", "message"],
    },
}


async def notify_handler(
    arguments: dict[str, Any], session=None, **_kwargs
) -> tuple[str, bool]:
    if session is None or session.notification_gateway is None:
        return "Messaging is not configured for this session.", False

    raw_destinations = arguments.get("destinations", [])
    if not isinstance(raw_destinations, list) or not raw_destinations:
        return "destinations must be a non-empty array of destination names.", False

    destinations: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_destinations:
        if not isinstance(raw_name, str):
            return "Each destination must be a string.", False
        name = raw_name.strip()
        if not name:
            return "Destination names must not be empty.", False
        if name not in seen:
            destinations.append(name)
            seen.add(name)

    disallowed = [
        name
        for name in destinations
        if not session.config.messaging.can_agent_tool_send(name)
    ]
    if disallowed:
        return (
            "These destinations are unavailable for the notify tool: "
            + ", ".join(disallowed)
        ), False

    message = arguments.get("message", "")
    if not isinstance(message, str) or not message.strip():
        return "message must be a non-empty string.", False

    title = arguments.get("title")
    severity = arguments.get("severity", "info")
    if title is not None and not isinstance(title, str):
        return "title must be a string when provided.", False
    if severity not in {"info", "success", "warning", "error"}:
        return "severity must be one of: info, success, warning, error.", False

    requests = [
        NotificationRequest(
            destination=name,
            title=title,
            message=message,
            severity=severity,
            metadata={
                "session_id": session.session_id,
                "model": session.config.model_name,
            },
        )
        for name in destinations
    ]
    results = await session.notification_gateway.send_many(requests)

    lines = []
    all_ok = True
    for result in results:
        if result.ok:
            lines.append(f"{result.destination}: sent")
        else:
            all_ok = False
            lines.append(f"{result.destination}: failed ({result.error})")
    return "\n".join(lines), all_ok
