"""
Tool system for the agent
Provides ToolSpec and ToolRouter for managing both built-in and MCP tools
"""

import warnings
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from fastmcp import Client
from fastmcp.exceptions import ToolError
from lmnr import observe
from mcp.types import EmbeddedResource, ImageContent, TextContent

from agent.config import MCPServerConfig
from agent.tools.docs_tools import (
    EXPLORE_HF_DOCS_TOOL_SPEC,
    HF_DOCS_FETCH_TOOL_SPEC,
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
)
from agent.tools.github_find_examples import (
    GITHUB_FIND_EXAMPLES_TOOL_SPEC,
    github_find_examples_handler,
)
from agent.tools.github_list_repos import (
    GITHUB_LIST_REPOS_TOOL_SPEC,
    github_list_repos_handler,
)
from agent.tools.github_read_file import (
    GITHUB_READ_FILE_TOOL_SPEC,
    github_read_file_handler,
)
from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, hf_jobs_handler
from agent.tools.plan_tool import PLAN_TOOL_SPEC, plan_tool_handler
from agent.tools.private_hf_repo_tools import (
    PRIVATE_HF_REPO_TOOL_SPEC,
    private_hf_repo_handler,
)

# NOTE: Utils tool disabled - date/time now loaded into system prompt at initialization
# from agent.tools.utils_tools import UTILS_TOOL_SPEC, utils_handler

# Suppress aiohttp deprecation warning
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="aiohttp.connector"
)

NOT_ALLOWED_TOOL_NAMES = ["hf_jobs", "hf_doc_search", "hf_doc_fetch"]


def convert_mcp_content_to_string(content: list) -> str:
    """
    Convert MCP content blocks to a string format compatible with LLM messages.

    Based on FastMCP documentation, content can be:
    - TextContent: has .text field
    - ImageContent: has .data and .mimeType fields
    - EmbeddedResource: has .resource field with .text or .blob

    Args:
        content: List of MCP content blocks

    Returns:
        String representation of the content suitable for LLM consumption
    """
    if not content:
        return ""

    parts = []
    for item in content:
        if isinstance(item, TextContent):
            # Extract text from TextContent blocks
            parts.append(item.text)
        elif isinstance(item, ImageContent):
            # TODO: Handle images
            # For images, include a description with MIME type
            parts.append(f"[Image: {item.mimeType}]")
        elif isinstance(item, EmbeddedResource):
            # TODO: Handle embedded resources
            # For embedded resources, try to extract text
            resource = item.resource
            if hasattr(resource, "text") and resource.text:
                parts.append(resource.text)
            elif hasattr(resource, "blob") and resource.blob:
                parts.append(
                    f"[Binary data: {resource.mimeType if hasattr(resource, 'mimeType') else 'unknown'}]"
                )
            else:
                parts.append(
                    f"[Resource: {resource.uri if hasattr(resource, 'uri') else 'unknown'}]"
                )
        else:
            # Fallback: try to convert to string
            parts.append(str(item))

    return "\n".join(parts)


@dataclass
class ToolSpec:
    """Tool specification for LLM"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Optional[Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]] = None


class ToolRouter:
    """
    Routes tool calls to appropriate handlers.
    Based on codex-rs/core/src/tools/router.rs
    """

    def __init__(self, mcp_servers: dict[str, MCPServerConfig]):
        self.tools: dict[str, ToolSpec] = {}
        self.mcp_servers: dict[str, dict[str, Any]] = {}

        for tool in create_builtin_tools():
            self.register_tool(tool)

        if mcp_servers:
            mcp_servers_payload = {}
            for name, server in mcp_servers.items():
                mcp_servers_payload[name] = server.model_dump()
            self.mcp_client = Client({"mcpServers": mcp_servers_payload})
        self._mcp_initialized = False

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    async def register_mcp_tools(self) -> None:
        tools = await self.mcp_client.list_tools()
        for tool in tools:
            if tool.name in NOT_ALLOWED_TOOL_NAMES:
                print(f"Skipping not MCP allowed tool: {tool.name}")
                continue
            print(f"MCP Tool: {tool.name}")
            self.register_tool(
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.inputSchema,
                    handler=None,
                )
            )

    async def register_openapi_tool(self) -> None:
        """Register the OpenAPI search tool (requires async initialization)"""
        from agent.tools.docs_tools import (
            _get_api_search_tool_spec,
            search_openapi_handler,
        )

        print("Registering OpenAPI search tool...")

        # Register search_hf_api_endpoints with dynamic spec
        openapi_spec = await _get_api_search_tool_spec()
        self.register_tool(
            ToolSpec(
                name=openapi_spec["name"],
                description=openapi_spec["description"],
                parameters=openapi_spec["parameters"],
                handler=search_openapi_handler,
            )
        )
        print(f"Registered: {openapi_spec['name']}")

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        """Get tool specifications in OpenAI format"""
        specs = []
        for tool in self.tools.values():
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return specs

    async def __aenter__(self) -> "ToolRouter":
        if self.mcp_client is not None:
            await self.mcp_client.__aenter__()
            await self.mcp_client.initialize()
            await self.register_mcp_tools()
            self._mcp_initialized = True
        print(f"MCP initialized: {self._mcp_initialized}")

        # Register OpenAPI tool (requires async initialization)
        await self.register_openapi_tool()

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.mcp_client is not None:
            await self.mcp_client.__aexit__(exc_type, exc, tb)
            self._mcp_initialized = False

    @observe(name="call_tool")
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        """
        Call a tool and return (output_string, success_bool).

        For MCP tools, converts the CallToolResult content blocks to a string.
        For built-in tools, calls their handler directly.
        """
        # Check if this is a built-in tool with a handler
        tool = self.tools.get(tool_name)
        if tool and tool.handler:
            return await tool.handler(arguments)

        # Otherwise, use MCP client
        if self._mcp_initialized:
            try:
                result = await self.mcp_client.call_tool(tool_name, arguments)
                output = convert_mcp_content_to_string(result.content)
                return output, not result.is_error
            except ToolError as e:
                # Catch MCP tool errors and return them to the agent
                error_msg = f"Tool error: {str(e)}"
                return error_msg, False

        return "MCP client not initialized", False


# ============================================================================
# BUILT-IN TOOL HANDLERS
# ============================================================================


def create_builtin_tools() -> list[ToolSpec]:
    """Create built-in tool specifications"""
    print(
        f"Creating built-in tools: {EXPLORE_HF_DOCS_TOOL_SPEC['name']}, {HF_DOCS_FETCH_TOOL_SPEC['name']}, {PLAN_TOOL_SPEC['name']}, {HF_JOBS_TOOL_SPEC['name']}, {PRIVATE_HF_REPO_TOOL_SPEC['name']}, {GITHUB_FIND_EXAMPLES_TOOL_SPEC['name']}, {GITHUB_LIST_REPOS_TOOL_SPEC['name']}, {GITHUB_READ_FILE_TOOL_SPEC['name']}"
    )
    # in order of importance
    return [
        # Documentation search tools
        ToolSpec(
            name=EXPLORE_HF_DOCS_TOOL_SPEC["name"],
            description=EXPLORE_HF_DOCS_TOOL_SPEC["description"],
            parameters=EXPLORE_HF_DOCS_TOOL_SPEC["parameters"],
            handler=explore_hf_docs_handler,
        ),
        ToolSpec(
            name=HF_DOCS_FETCH_TOOL_SPEC["name"],
            description=HF_DOCS_FETCH_TOOL_SPEC["description"],
            parameters=HF_DOCS_FETCH_TOOL_SPEC["parameters"],
            handler=hf_docs_fetch_handler,
        ),
        # Planning and job management tools
        ToolSpec(
            name=PLAN_TOOL_SPEC["name"],
            description=PLAN_TOOL_SPEC["description"],
            parameters=PLAN_TOOL_SPEC["parameters"],
            handler=plan_tool_handler,
        ),
        ToolSpec(
            name=HF_JOBS_TOOL_SPEC["name"],
            description=HF_JOBS_TOOL_SPEC["description"],
            parameters=HF_JOBS_TOOL_SPEC["parameters"],
            handler=hf_jobs_handler,
        ),
        ToolSpec(
            name=PRIVATE_HF_REPO_TOOL_SPEC["name"],
            description=PRIVATE_HF_REPO_TOOL_SPEC["description"],
            parameters=PRIVATE_HF_REPO_TOOL_SPEC["parameters"],
            handler=private_hf_repo_handler,
        ),
        # NOTE: Utils tool disabled - date/time now loaded into system prompt at initialization (less tool calls=more reliablity)
        # ToolSpec(
        #     name=UTILS_TOOL_SPEC["name"],
        #     description=UTILS_TOOL_SPEC["description"],
        #     parameters=UTILS_TOOL_SPEC["parameters"],
        #     handler=utils_handler,
        # ),
        # GitHub tools
        # NOTE: Github search code tool disabled - a bit buggy
        # ToolSpec(
        #     name=GITHUB_SEARCH_CODE_TOOL_SPEC["name"],
        #     description=GITHUB_SEARCH_CODE_TOOL_SPEC["description"],
        #     parameters=GITHUB_SEARCH_CODE_TOOL_SPEC["parameters"],
        #     handler=github_search_code_handler,
        # ),
        ToolSpec(
            name=GITHUB_FIND_EXAMPLES_TOOL_SPEC["name"],
            description=GITHUB_FIND_EXAMPLES_TOOL_SPEC["description"],
            parameters=GITHUB_FIND_EXAMPLES_TOOL_SPEC["parameters"],
            handler=github_find_examples_handler,
        ),
        ToolSpec(
            name=GITHUB_LIST_REPOS_TOOL_SPEC["name"],
            description=GITHUB_LIST_REPOS_TOOL_SPEC["description"],
            parameters=GITHUB_LIST_REPOS_TOOL_SPEC["parameters"],
            handler=github_list_repos_handler,
        ),
        ToolSpec(
            name=GITHUB_READ_FILE_TOOL_SPEC["name"],
            description=GITHUB_READ_FILE_TOOL_SPEC["description"],
            parameters=GITHUB_READ_FILE_TOOL_SPEC["parameters"],
            handler=github_read_file_handler,
        ),
    ]
