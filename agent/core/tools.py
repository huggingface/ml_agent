"""
Tool system for the agent
Provides ToolSpec and ToolRouter for managing both built-in and MCP tools
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import EmbeddedResource, ImageContent, TextContent

from agent.config import MCPServerConfig
from agent.tools.dataset_tools import (
    HF_INSPECT_DATASET_TOOL_SPEC,
    hf_inspect_dataset_handler,
)
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
from agent.tools.hf_repo_files_tool import (
    HF_REPO_FILES_TOOL_SPEC,
    hf_repo_files_handler,
)
from agent.tools.hf_repo_git_tool import (
    HF_REPO_GIT_TOOL_SPEC,
    hf_repo_git_handler,
)
from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, hf_jobs_handler
from agent.tools.notify_tool import NOTIFY_TOOL_SPEC, notify_handler
from agent.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler
from agent.tools.plan_tool import PLAN_TOOL_SPEC, plan_tool_handler
from agent.tools.research_tool import RESEARCH_TOOL_SPEC, research_handler
from agent.tools.sandbox_tool import get_sandbox_tools
from agent.tools.web_search_tool import WEB_SEARCH_TOOL_SPEC, web_search_handler

# NOTE: Private HF repo tool disabled - replaced by hf_repo_files and hf_repo_git
# from agent.tools.private_hf_repo_tools import (
#     PRIVATE_HF_REPO_TOOL_SPEC,
#     private_hf_repo_handler,
# )

# Suppress aiohttp deprecation warning
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="aiohttp.connector"
)

logger = logging.getLogger(__name__)

NOT_ALLOWED_TOOL_NAMES = ["hf_jobs", "hf_doc_search", "hf_doc_fetch", "hf_whoami"]


def convert_mcp_content_to_llm_content(content: list) -> str | list[dict]:
    """
    Convert MCP content blocks to a format compatible with LiteLLM messages.

    Returns a plain string when content contains only text (backward-compatible
    with text-only tool results). Returns a list of OpenAI-style content blocks
    when any ImageContent is present, so vision-capable models receive the actual
    image data instead of a placeholder.

    Based on FastMCP documentation, content can be:
    - TextContent: has .text field
    - ImageContent: has .data (base64) and .mimeType fields
    - EmbeddedResource: has .resource field with .text or .blob
    """
    if not content:
        return ""

    blocks: list[dict] = []
    has_image = False

    for item in content:
        if isinstance(item, TextContent):
            blocks.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            has_image = True
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{item.mimeType};base64,{item.data}"
                    },
                }
            )
        elif isinstance(item, EmbeddedResource):
            resource = item.resource
            if hasattr(resource, "text") and resource.text:
                blocks.append({"type": "text", "text": resource.text})
            elif hasattr(resource, "blob") and resource.blob:
                mime = (
                    resource.mimeType
                    if hasattr(resource, "mimeType")
                    else "unknown"
                )
                blocks.append({"type": "text", "text": f"[Binary data: {mime}]"})
            else:
                uri = resource.uri if hasattr(resource, "uri") else "unknown"
                blocks.append({"type": "text", "text": f"[Resource: {uri}]"})
        else:
            blocks.append({"type": "text", "text": str(item)})

    if not has_image:
        return "\n".join(b["text"] for b in blocks)
    return blocks


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

    def __init__(
        self,
        mcp_servers: dict[str, MCPServerConfig],
        hf_token: str | None = None,
        local_mode: bool = False,
    ):
        self.tools: dict[str, ToolSpec] = {}
        self.mcp_servers: dict[str, dict[str, Any]] = {}

        for tool in create_builtin_tools(local_mode=local_mode):
            self.register_tool(tool)

        self.mcp_client: Client | None = None
        if mcp_servers:
            mcp_servers_payload = {}
            for name, server in mcp_servers.items():
                data = server.model_dump()
                if hf_token:
                    data.setdefault("headers", {})["Authorization"] = (
                        f"Bearer {hf_token}"
                    )
                mcp_servers_payload[name] = data
            self.mcp_client = Client({"mcpServers": mcp_servers_payload})
        self._mcp_initialized = False

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    async def register_mcp_tools(self) -> None:
        tools = await self.mcp_client.list_tools()
        registered_names = []
        skipped_count = 0
        for tool in tools:
            if tool.name in NOT_ALLOWED_TOOL_NAMES:
                skipped_count += 1
                continue
            registered_names.append(tool.name)
            self.register_tool(
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.inputSchema,
                    handler=None,
                )
            )
        logger.info(
            f"Loaded {len(registered_names)} MCP tools: {', '.join(registered_names)} ({skipped_count} disabled)"
        )

    async def register_openapi_tool(self) -> None:
        """Register the OpenAPI search tool (requires async initialization)"""
        from agent.tools.docs_tools import (
            _get_api_search_tool_spec,
            search_openapi_handler,
        )

        try:
            openapi_spec = await _get_api_search_tool_spec()
            self.register_tool(
                ToolSpec(
                    name=openapi_spec["name"],
                    description=openapi_spec["description"],
                    parameters=openapi_spec["parameters"],
                    handler=search_openapi_handler,
                )
            )
            logger.info(f"Loaded OpenAPI search tool: {openapi_spec['name']}")
        except Exception as e:
            logger.warning("Failed to load OpenAPI search tool: %s", e)

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
            try:
                await self.mcp_client.__aenter__()
                await self.mcp_client.initialize()
                await self.register_mcp_tools()
                self._mcp_initialized = True
            except Exception as e:
                logger.warning(
                    "MCP connection failed, continuing without MCP tools: %s", e
                )
                self.mcp_client = None

        await self.register_openapi_tool()

        total_tools = len(self.tools)
        logger.info(f"Agent ready with {total_tools} tools total")

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.mcp_client is not None:
            await self.mcp_client.__aexit__(exc_type, exc, tb)
            self._mcp_initialized = False

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: Any = None,
        tool_call_id: str | None = None,
    ) -> tuple[str | list[dict], bool]:
        """
        Call a tool and return (output, success_bool).

        For MCP tools, converts the CallToolResult content blocks via
        convert_mcp_content_to_llm_content — plain str for text-only results,
        list of OpenAI content blocks when images are present.
        For built-in tools, calls their handler directly (always returns str).
        """
        # Check if this is a built-in tool with a handler
        tool = self.tools.get(tool_name)
        if tool and tool.handler:
            import inspect

            # Check if handler accepts session argument
            sig = inspect.signature(tool.handler)
            if "session" in sig.parameters:
                # Check if handler also accepts tool_call_id parameter
                if "tool_call_id" in sig.parameters:
                    return await tool.handler(
                        arguments, session=session, tool_call_id=tool_call_id
                    )
                return await tool.handler(arguments, session=session)
            return await tool.handler(arguments)

        # Otherwise, use MCP client
        if self._mcp_initialized:
            try:
                result = await self.mcp_client.call_tool(tool_name, arguments)
                output = convert_mcp_content_to_llm_content(result.content)
                return output, not result.is_error
            except ToolError as e:
                # Catch MCP tool errors and return them to the agent
                error_msg = f"Tool error: {str(e)}"
                return error_msg, False

        return "MCP client not initialized", False


# ============================================================================
# BUILT-IN TOOL HANDLERS
# ============================================================================


def create_builtin_tools(local_mode: bool = False) -> list[ToolSpec]:
    """Create built-in tool specifications"""
    # in order of importance
    tools = [
        # Research sub-agent (delegates to read-only tools in independent context)
        ToolSpec(
            name=RESEARCH_TOOL_SPEC["name"],
            description=RESEARCH_TOOL_SPEC["description"],
            parameters=RESEARCH_TOOL_SPEC["parameters"],
            handler=research_handler,
        ),
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
        # Paper discovery and reading
        ToolSpec(
            name=HF_PAPERS_TOOL_SPEC["name"],
            description=HF_PAPERS_TOOL_SPEC["description"],
            parameters=HF_PAPERS_TOOL_SPEC["parameters"],
            handler=hf_papers_handler,
        ),
        ToolSpec(
            name=WEB_SEARCH_TOOL_SPEC["name"],
            description=WEB_SEARCH_TOOL_SPEC["description"],
            parameters=WEB_SEARCH_TOOL_SPEC["parameters"],
            handler=web_search_handler,
        ),
        # Dataset inspection tool (unified)
        ToolSpec(
            name=HF_INSPECT_DATASET_TOOL_SPEC["name"],
            description=HF_INSPECT_DATASET_TOOL_SPEC["description"],
            parameters=HF_INSPECT_DATASET_TOOL_SPEC["parameters"],
            handler=hf_inspect_dataset_handler,
        ),
        # Planning and job management tools
        ToolSpec(
            name=PLAN_TOOL_SPEC["name"],
            description=PLAN_TOOL_SPEC["description"],
            parameters=PLAN_TOOL_SPEC["parameters"],
            handler=plan_tool_handler,
        ),
        ToolSpec(
            name=NOTIFY_TOOL_SPEC["name"],
            description=NOTIFY_TOOL_SPEC["description"],
            parameters=NOTIFY_TOOL_SPEC["parameters"],
            handler=notify_handler,
        ),
        ToolSpec(
            name=HF_JOBS_TOOL_SPEC["name"],
            description=HF_JOBS_TOOL_SPEC["description"],
            parameters=HF_JOBS_TOOL_SPEC["parameters"],
            handler=hf_jobs_handler,
        ),
        # HF Repo management tools
        ToolSpec(
            name=HF_REPO_FILES_TOOL_SPEC["name"],
            description=HF_REPO_FILES_TOOL_SPEC["description"],
            parameters=HF_REPO_FILES_TOOL_SPEC["parameters"],
            handler=hf_repo_files_handler,
        ),
        ToolSpec(
            name=HF_REPO_GIT_TOOL_SPEC["name"],
            description=HF_REPO_GIT_TOOL_SPEC["description"],
            parameters=HF_REPO_GIT_TOOL_SPEC["parameters"],
            handler=hf_repo_git_handler,
        ),
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

    # Sandbox or local tools (highest priority)
    if local_mode:
        from agent.tools.local_tools import get_local_tools

        tools = get_local_tools() + tools
    else:
        tools = get_sandbox_tools() + tools

    tool_names = ", ".join([t.name for t in tools])
    logger.info(f"Loaded {len(tools)} built-in tools: {tool_names}")

    return tools
