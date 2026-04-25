"""
ML Intern tools, exposed as an MCP server for Claude Code.

Thin shim over agent/tools/*: same handlers, same JSON schemas, same
behavior — only the transport changes from "litellm tool calls inside
agent_loop.py" to "MCP stdio for Claude Code".

Uses the low-level `mcp.server.lowlevel.Server` API so we can register
tools with the original JSON schemas verbatim. FastMCP's high-level
`@mcp.tool` would re-derive schemas from Python type hints, which would
lose nullable/oneOf/operation-discriminated structures the existing
ml-intern specs encode.

Run via the `.mcp.json` at the repo root. Not intended to be invoked manually.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

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
from agent.tools.hf_repo_git_tool import HF_REPO_GIT_TOOL_SPEC, hf_repo_git_handler
from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, hf_jobs_handler
from agent.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler
from agent.tools.sandbox_tool import get_sandbox_tools

logger = logging.getLogger(__name__)

# `research` and `plan_tool` are intentionally NOT exposed:
#   research → replaced by .claude/agents/research.md (Claude Code subagent)
#   plan_tool → replaced by Claude Code's built-in TodoWrite
_TOOL_SPECS: list[tuple[dict[str, Any], Callable[..., Awaitable[tuple[str, bool]]]]] = [
    (EXPLORE_HF_DOCS_TOOL_SPEC, explore_hf_docs_handler),
    (HF_DOCS_FETCH_TOOL_SPEC, hf_docs_fetch_handler),
    (HF_PAPERS_TOOL_SPEC, hf_papers_handler),
    (HF_INSPECT_DATASET_TOOL_SPEC, hf_inspect_dataset_handler),
    (HF_JOBS_TOOL_SPEC, hf_jobs_handler),
    (HF_REPO_FILES_TOOL_SPEC, hf_repo_files_handler),
    (HF_REPO_GIT_TOOL_SPEC, hf_repo_git_handler),
    (GITHUB_FIND_EXAMPLES_TOOL_SPEC, github_find_examples_handler),
    (GITHUB_LIST_REPOS_TOOL_SPEC, github_list_repos_handler),
    (GITHUB_READ_FILE_TOOL_SPEC, github_read_file_handler),
]

# Discovered async at startup. Populated below in build_registry().
_REGISTRY: dict[str, tuple[types.Tool, Callable[..., Awaitable[tuple[str, bool]]]]] = {}


def _build_registry() -> None:
    """Populate the {name: (Tool, handler)} registry."""
    for spec, handler in _TOOL_SPECS:
        tool = types.Tool(
            name=spec["name"],
            description=spec["description"],
            inputSchema=spec["parameters"],
        )
        _REGISTRY[spec["name"]] = (tool, handler)

    # Sandbox tools come from a factory because they depend on local_mode.
    # Mirrors agent/main.py: ML_INTERN_LOCAL_MODE=1 routes shell/file ops to
    # the local machine instead of HF Sandboxes.
    local_mode = os.environ.get("ML_INTERN_LOCAL_MODE", "").lower() in ("1", "true", "yes")
    if local_mode:
        from agent.tools.local_tools import get_local_tools

        sandbox_specs = get_local_tools()
    else:
        sandbox_specs = get_sandbox_tools()

    for tool_spec in sandbox_specs:
        tool = types.Tool(
            name=tool_spec.name,
            description=tool_spec.description,
            inputSchema=tool_spec.parameters,
        )
        _REGISTRY[tool_spec.name] = (tool, tool_spec.handler)


server: Server = Server("ml-intern-tools")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [tool for tool, _ in _REGISTRY.values()]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    entry = _REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown tool: {name}")
    _tool, handler = entry

    output, ok = await handler(arguments or {})
    if not ok:
        # MCP convention: raise so the client sees isError=true with the message.
        raise RuntimeError(output)
    return [types.TextContent(type="text", text=output)]


async def _amain() -> None:
    _build_registry()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_amain())
