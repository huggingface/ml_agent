"""
Hugging Face tools for the agent
"""

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
from agent.tools.github_search_code import (
    GITHUB_SEARCH_CODE_TOOL_SPEC,
    github_search_code_handler,
)
from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, HfJobsTool, hf_jobs_handler
from agent.tools.types import ToolResult

__all__ = [
    "ToolResult",
    "HF_JOBS_TOOL_SPEC",
    "hf_jobs_handler",
    "HfJobsTool",
    "GITHUB_FIND_EXAMPLES_TOOL_SPEC",
    "github_find_examples_handler",
    "GITHUB_LIST_REPOS_TOOL_SPEC",
    "github_list_repos_handler",
    "GITHUB_READ_FILE_TOOL_SPEC",
    "github_read_file_handler",
    "GITHUB_SEARCH_CODE_TOOL_SPEC",
    "github_search_code_handler",
]
