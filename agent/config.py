import json
import os
import re
from pathlib import Path
from typing import Any, Union

from dotenv import load_dotenv

# Project root: two levels up from this file (agent/config.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from fastmcp.mcp_config import (
    RemoteMCPServer,
    StdioMCPServer,
)
from pydantic import BaseModel

# These two are the canonical server config types for MCP servers.
MCPServerConfig = Union[StdioMCPServer, RemoteMCPServer]


class Config(BaseModel):
    """Configuration manager"""

    model_name: str
    mcpServers: dict[str, MCPServerConfig] = {}
    save_sessions: bool = True
    session_dataset_repo: str = "akseljoonas/hf-agent-sessions"
    auto_save_interval: int = 3  # Save every N user turns (0 = disabled)
    yolo_mode: bool = False  # Auto-approve all tool calls without confirmation
    max_iterations: int = 300  # Max LLM calls per agent turn (-1 = unlimited)

    # Permission control parameters
    confirm_cpu_jobs: bool = True
    auto_file_upload: bool = False

    # Reasoning effort *preference* — the ceiling the user wants. The probe
    # on `/model` walks a cascade down from here (``xhigh`` → ``high`` → …)
    # and caches per-model what the provider actually accepted in
    # ``Session.model_effective_effort``. Default ``xhigh`` on posttrain-bench:
    # max would burn our rate-limit budget for negligible quality gain over
    # xhigh, and for non-Anthropic models the cascade falls to ``high``.
    # Valid values: None | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
    reasoning_effort: str | None = "xhigh"


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


def load_config(config_path: str = "config.json") -> Config:
    """
    Load configuration with environment variable substitution.

    Use ${VAR_NAME} in your JSON for any secret.
    Automatically loads from .env file.
    """
    # Load .env from project root first (so it works from any directory),
    # then CWD .env can override if present
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(override=False)

    with open(config_path, "r") as f:
        raw_config = json.load(f)

    config_with_env = substitute_env_vars(raw_config)
    return Config.model_validate(config_with_env)
