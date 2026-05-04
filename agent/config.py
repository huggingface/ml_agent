import json
import os
import re
from pathlib import Path
from typing import Any, Union

from dotenv import load_dotenv
from fastmcp.mcp_config import (
    RemoteMCPServer,
    StdioMCPServer,
)
from pydantic import BaseModel

from agent.messaging.models import MessagingConfig

# These two are the canonical server config types for MCP servers.
MCPServerConfig = Union[StdioMCPServer, RemoteMCPServer]

# Project root: two levels up from this file (agent/config.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(BaseModel):
    """Configuration manager"""

    model_name: str
    mcpServers: dict[str, MCPServerConfig] = {}
    save_sessions: bool = True
    session_dataset_repo: str = "smolagents/ml-intern-sessions"
    # Per-user private dataset that mirrors each session in Claude Code JSONL
    # format so the HF Agent Trace Viewer auto-renders it
    # (https://huggingface.co/changelog/agent-trace-viewer). Created private
    # on first use; user flips it public via /share-traces. ``{hf_user}`` is
    # substituted at upload time from the authenticated HF username.
    share_traces: bool = True
    personal_trace_repo_template: str = "{hf_user}/ml-intern-sessions"
    auto_save_interval: int = 1  # Save every N user turns (0 = disabled)
    # Mid-turn heartbeat: save + upload every N seconds while events are being
    # emitted. Guards against losing trace data on long-running turns that
    # crash before turn_complete (e.g. a multi-hour hf_jobs wait that OOMs).
    # 0 = disabled. Consumed by agent.core.telemetry.HeartbeatSaver.
    heartbeat_interval_s: int = 60
    yolo_mode: bool = False  # Auto-approve all tool calls without confirmation
    max_iterations: int = 300  # Max LLM calls per agent turn (-1 = unlimited)

    # Permission control parameters
    confirm_cpu_jobs: bool = True
    auto_file_upload: bool = False

    # Reasoning effort *preference* — the ceiling the user wants. The probe
    # on `/model` walks a cascade down from here (``max`` → ``xhigh`` → ``high``
    # → …) and caches per-model what the provider actually accepted in
    # ``Session.model_effective_effort``. Default ``max`` because we'd rather
    # burn tokens thinking than ship a wrong ML recipe; the cascade lands on
    # whichever level the model supports (``high`` for GPT-5 / HF router,
    # ``xhigh`` or ``max`` for Anthropic 4.6 / 4.7). ``None`` = thinking off.
    # Valid values: None | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
    reasoning_effort: str | None = "max"
    messaging: MessagingConfig = MessagingConfig()


USER_CONFIG_ENV_VAR = "ML_INTERN_CLI_CONFIG"
DEFAULT_USER_CONFIG_PATH = (
    Path.home() / ".config" / "ml-intern" / "cli_agent_config.json"
)
SLACK_DEFAULT_DESTINATION = "slack.default"
SLACK_DEFAULT_AUTO_EVENT_TYPES = ["approval_required", "error", "turn_complete"]


def _deep_merge_config(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config(current, value)
        else:
            merged[key] = value
    return merged


def _load_json_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a JSON object")
    return data


def _load_user_config() -> dict[str, Any]:
    raw_path = os.environ.get(USER_CONFIG_ENV_VAR)
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"{USER_CONFIG_ENV_VAR} points to missing config file: {path}"
            )
        return _load_json_config(path)

    if DEFAULT_USER_CONFIG_PATH.exists():
        return _load_json_config(DEFAULT_USER_CONFIG_PATH)
    return {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_list(name: str) -> list[str] | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def apply_slack_user_defaults(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Enable a default Slack destination from user env vars, when present."""
    if not _env_bool("ML_INTERN_SLACK_NOTIFICATIONS", True):
        return raw_config

    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID") or os.environ.get("SLACK_CHANNEL")
    if not token or not channel:
        return raw_config

    config = dict(raw_config)
    messaging = dict(config.get("messaging") or {})
    destinations = dict(messaging.get("destinations") or {})
    destination_name = (
        os.environ.get("ML_INTERN_SLACK_DESTINATION") or SLACK_DEFAULT_DESTINATION
    ).strip()

    if destination_name not in destinations:
        destinations[destination_name] = {
            "provider": "slack",
            "token": token,
            "channel": channel,
            "allow_agent_tool": _env_bool("ML_INTERN_SLACK_ALLOW_AGENT_TOOL", True),
            "allow_auto_events": _env_bool("ML_INTERN_SLACK_ALLOW_AUTO_EVENTS", True),
        }

    auto_events = _env_list("ML_INTERN_SLACK_AUTO_EVENTS")
    if auto_events is not None:
        messaging["auto_event_types"] = auto_events
    elif "auto_event_types" not in messaging:
        messaging["auto_event_types"] = SLACK_DEFAULT_AUTO_EVENT_TYPES

    messaging["enabled"] = True
    messaging["destinations"] = destinations
    config["messaging"] = messaging
    return config


def substitute_env_vars(obj: Any) -> Any:
    """
    Recursively substitute environment variables in any data structure.

    Supports ${VAR_NAME} syntax for required variables and ${VAR_NAME:-default} for optional.
    """
    if isinstance(obj, str):
        pattern = r"\$\{([^}:]+)(?::(-)?([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) if has_default else None

            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif has_default:
                return default_value or ""
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set. "
                    f"Add it to your .env file."
                )

        return re.sub(pattern, replacer, obj)

    elif isinstance(obj, dict):
        return {key: substitute_env_vars(value) for key, value in obj.items()}

    elif isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]

    return obj


def load_config(
    config_path: str = "config.json",
    include_user_defaults: bool = False,
) -> Config:
    """
    Load configuration with environment variable substitution.

    Use ${VAR_NAME} in your JSON for any secret.
    Automatically loads from .env file.
    """
    # Load .env from project root first (so it works from any directory),
    # then CWD .env can override if present
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(override=False)

    raw_config = _load_json_config(Path(config_path))
    if include_user_defaults:
        raw_config = _deep_merge_config(raw_config, _load_user_config())
        raw_config = apply_slack_user_defaults(raw_config)

    config_with_env = substitute_env_vars(raw_config)
    return Config.model_validate(config_with_env)
