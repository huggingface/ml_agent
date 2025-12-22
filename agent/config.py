import json
import os
import re
from typing import Any, Union

from dotenv import load_dotenv
from fastmcp.mcp_config import (
    RemoteMCPServer,
    StdioMCPServer,
)
from litellm import Tool
from pydantic import BaseModel

# These two are the canonical server config types for MCP servers.
MCPServerConfig = Union[StdioMCPServer, RemoteMCPServer]


class Config(BaseModel):
    """Configuration manager"""

    model_name: str
    tools: list[Tool] = []
    mcpServers: dict[str, MCPServerConfig] = {}


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
    # Load environment variables from .env file
    load_dotenv()

    with open(config_path, "r") as f:
        raw_config = json.load(f)

    config_with_env = substitute_env_vars(raw_config)
    return Config.model_validate(config_with_env)
