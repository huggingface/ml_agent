"""Pytest configuration and fixtures for unit tests.

This module sets up comprehensive stubs for agent dependencies to allow
isolated unit testing without requiring the full runtime stack.
"""

import sys
from unittest.mock import MagicMock
from types import ModuleType


def create_fake_package(name):
    """Create a fake package and all its submodules."""
    module = ModuleType(name)
    sys.modules[name] = module
    return module


def _install_stubs():
    """Install comprehensive stubs for all external dependencies.

    This must run at import time because some tests import modules that pull
    in agent dependencies at module import, before pytest hooks execute.
    """
    # Web frameworks
    fastapi_module = create_fake_package('fastapi')
    fastapi_module.FastAPI = MagicMock
    fastapi_module.testclient = MagicMock()
    fastapi_module.testclient.TestClient = MagicMock
    sys.modules['fastapi'] = fastapi_module
    sys.modules['fastapi.testclient'] = fastapi_module.testclient
    sys.modules['starlette'] = MagicMock()
    
    # HTTP and networking
    sys.modules['httpx'] = MagicMock()
    sys.modules['aiohttp'] = MagicMock()
    sys.modules['requests'] = MagicMock()

    # Data and models
    sys.modules['pydantic'] = MagicMock()
    sys.modules['pydantic.types'] = MagicMock()
    sys.modules['numpy'] = MagicMock()
    sys.modules['pandas'] = MagicMock()

    # Hugging Face
    sys.modules['huggingface_hub'] = MagicMock()
    sys.modules['datasets'] = MagicMock()
    sys.modules['transformers'] = MagicMock()

    # Notebook support
    sys.modules['nbformat'] = MagicMock()
    sys.modules['nbconvert'] = MagicMock()
    sys.modules['jupyter'] = MagicMock()

    # Text processing
    sys.modules['thefuzz'] = MagicMock()
    sys.modules['thefuzz.fuzz'] = MagicMock()

    # MCP stubs
    mcp = create_fake_package('mcp')
    mcp.types = MagicMock()
    sys.modules['mcp.types'] = mcp.types

    # FastMCP stubs - needs to be a real package, not just MagicMock
    fastmcp = create_fake_package('fastmcp')
    fastmcp.Client = MagicMock
    fastmcp.exceptions = create_fake_package('fastmcp.exceptions')
    fastmcp.exceptions.ToolError = Exception
    fastmcp.types = create_fake_package('fastmcp.types')
    fastmcp.mcp_config = MagicMock()  # Import location for MCPServerConfig loading
    sys.modules['fastmcp.client'] = fastmcp
    sys.modules['fastmcp.exceptions'] = fastmcp.exceptions
    sys.modules['fastmcp.types'] = fastmcp.types
    sys.modules['fastmcp.mcp_config'] = fastmcp.mcp_config

    # LLM and AI stubs
    sys.modules['litellm'] = MagicMock()
    sys.modules['anthropic'] = MagicMock()
    sys.modules['openai'] = MagicMock()

    # Sandbox/container stubs
    sys.modules['docker'] = MagicMock()
    sys.modules['docker.client'] = MagicMock()

    # Other common imports
    sys.modules['dotenv'] = MagicMock()
    sys.modules['yaml'] = MagicMock()
    sys.modules['toml'] = MagicMock()


# Install stubs IMMEDIATELY on module load (before pytest collection)
# Keep this at module scope: importing test modules can trigger dependency
# imports before pytest hooks run, so the stubs must already be present.
_install_stubs()


def pytest_configure(config):
    """Pytest hook called before test collection starts.
    
    We ensure stubs are already installed above, but this hook
    can be used for any other early setup if needed.
    """
    pass







