from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_DESTINATION_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
SUPPORTED_AUTO_EVENT_TYPES = {"approval_required", "error"}


class SlackDestinationConfig(BaseModel):
    provider: Literal["slack"] = "slack"
    token: str
    channel: str
    allow_agent_tool: bool = False
    allow_auto_events: bool = False
    username: str | None = None
    icon_emoji: str | None = None

    @field_validator("token", "channel")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


DestinationConfig = Annotated[SlackDestinationConfig, Field(discriminator="provider")]


class MessagingConfig(BaseModel):
    enabled: bool = False
    auto_event_types: list[str] = Field(
        default_factory=lambda: ["approval_required", "error"]
    )
    destinations: dict[str, DestinationConfig] = Field(default_factory=dict)

    @field_validator("destinations")
    @classmethod
    def _validate_destination_names(
        cls, destinations: dict[str, DestinationConfig]
    ) -> dict[str, DestinationConfig]:
        for name in destinations:
            if not name or any(char not in _DESTINATION_NAME_CHARS for char in name):
                raise ValueError(
                    "destination names must use lowercase letters, digits, '.', '_' or '-'"
                )
        return destinations

    @field_validator("auto_event_types")
    @classmethod
    def _validate_auto_event_types(cls, event_types: list[str]) -> list[str]:
        if not event_types:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for event_type in event_types:
            if event_type not in SUPPORTED_AUTO_EVENT_TYPES:
                raise ValueError(
                    f"unsupported auto event type '{event_type}'"
                )
            if event_type not in seen:
                normalized.append(event_type)
                seen.add(event_type)
        return normalized

    @model_validator(mode="after")
    def _require_destinations_when_enabled(self) -> "MessagingConfig":
        if self.enabled and not self.destinations:
            raise ValueError("messaging.enabled requires at least one destination")
        return self

    def get_destination(self, name: str) -> DestinationConfig | None:
        return self.destinations.get(name)

    def can_agent_tool_send(self, name: str) -> bool:
        destination = self.get_destination(name)
        return bool(destination and destination.allow_agent_tool)

    def can_auto_send(self, name: str) -> bool:
        destination = self.get_destination(name)
        return bool(destination and destination.allow_auto_events)


class NotificationRequest(BaseModel):
    destination: str
    title: str | None = None
    message: str
    severity: Literal["info", "success", "warning", "error"] = "info"
    metadata: dict[str, str] = Field(default_factory=dict)
    event_type: str | None = None

    @field_validator("destination", "message")
    @classmethod
    def _require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("title")
    @classmethod
    def _normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class NotificationResult(BaseModel):
    destination: str
    ok: bool
    provider: str
    error: str | None = None
    external_id: str | None = None
