"""
GitHub Read File Tool - Read file contents from any GitHub repository with line range support

Fetch exact file contents with metadata, supporting line ranges for efficient reading.
"""

import base64
import json
import os
from typing import Any, Dict, Optional

import nbformat
import requests
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ClearOutputPreprocessor, TagRemovePreprocessor

from agent.tools.types import ToolResult


def _convert_ipynb_to_markdown(content: str) -> str:
    """
    Convert Jupyter notebook JSON to LLM-friendly Markdown.

    Args:
        content: Raw notebook JSON string

    Returns:
        Converted Markdown string
    """
    try:
        # Parse notebook JSON
        nb_dict = json.loads(content)

        # Normalize cell sources (can be string or list of strings)
        if "cells" in nb_dict:
            for cell in nb_dict["cells"]:
                if "source" in cell and isinstance(cell["source"], list):
                    cell["source"] = "".join(cell["source"])

        # Read notebook with explicit version
        nb = nbformat.reads(json.dumps(nb_dict), as_version=4)

        # Strip outputs for LLM readability (outputs can be noisy/large)
        clear = ClearOutputPreprocessor()
        nb, _ = clear.preprocess(nb, {})

        # Optionally remove cells tagged with "hide" or similar
        remove = TagRemovePreprocessor(
            remove_cell_tags={"hide", "hidden", "remove"},
            remove_input_tags=set(),
            remove_all_outputs_tags=set(),
        )
        nb, _ = remove.preprocess(nb, {})

        # Convert to markdown
        exporter = MarkdownExporter()
        markdown, _ = exporter.from_notebook_node(nb)

        return markdown

    except json.JSONDecodeError:
        return content
    except Exception:
        return content


def read_file(
    repo: str,
    path: str,
    ref: str = "HEAD",
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
) -> ToolResult:
    """
    Read file contents from a GitHub repository with line range support.

    Args:
        repo: Repository in format "owner/repo" (e.g., "github/github-mcp-server")
        path: Path to file in repository (e.g., "pkg/github/search.go")
        ref: Git reference - branch name, tag, or commit SHA (default: "HEAD")
        line_start: Starting line number (1-indexed, inclusive)
        line_end: Ending line number (1-indexed, inclusive)

    Returns:
        ToolResult with file contents and metadata
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {
            "formatted": "Error: GITHUB_TOKEN environment variable is required",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    # Parse repo
    if "/" not in repo:
        return {
            "formatted": "Error: repo must be in format 'owner/repo'",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    owner, repo_name = repo.split("/", 1)

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }

    # Fetch file contents
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
    params = {}
    if ref and ref != "HEAD":
        params["ref"] = ref

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 404:
            return {
                "formatted": f"File not found: {path} in {repo} (ref: {ref})",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        if response.status_code != 200:
            error_msg = f"GitHub API error (status {response.status_code})"
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except Exception:
                pass
            return {
                "formatted": error_msg,
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        data = response.json()

        # Check if it's a file
        if data.get("type") != "file":
            return {
                "formatted": f"Path {path} is not a file (type: {data.get('type')})",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        # Decode content
        content_b64 = data.get("content", "")
        if content_b64:
            content_b64 = content_b64.replace("\n", "").replace(" ", "")
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        else:
            # For large files, fetch raw content
            raw_headers = {
                "Accept": "application/vnd.github.raw",
                "X-GitHub-Api-Version": "2022-11-28",
                "Authorization": f"Bearer {token}",
            }
            raw_response = requests.get(
                url, headers=raw_headers, params=params, timeout=30
            )
            if raw_response.status_code != 200:
                return {
                    "formatted": "Failed to fetch file content",
                    "totalResults": 0,
                    "resultsShared": 0,
                    "isError": True,
                }
            content = raw_response.text

        if path.lower().endswith(".ipynb"):
            content = _convert_ipynb_to_markdown(content)

        # Process line ranges
        lines = content.split("\n")
        total_lines = len(lines)

        truncated = False

        if line_start is None and line_end is None:
            # No range specified
            if total_lines > 300:
                line_start = 1
                line_end = 300
                truncated = True
            else:
                line_start = 1
                line_end = total_lines
        else:
            # Range specified
            if line_start is None:
                line_start = 1
            if line_end is None:
                line_end = total_lines

            # Validate range
            line_start = max(1, line_start)
            line_end = min(total_lines, line_end)
            if line_start > line_end:
                return {
                    "formatted": f"Invalid range: line_start ({line_start}) > line_end ({line_end})",
                    "totalResults": 0,
                    "resultsShared": 0,
                    "isError": True,
                }

        # Extract lines
        selected_lines = lines[line_start - 1 : line_end]
        selected_content = "\n".join(selected_lines)

        # Format output
        lines_output = [f"**Reading file from repo: {repo}, path: {path}**"]

        if ref and ref != "HEAD":
            lines_output.append(f"Ref: {ref}")

        lines_output.append("\n**File content:")
        lines_output.append("```")
        lines_output.append(selected_content)
        lines_output.append("```")
        if truncated:
            lines_output.append(
                f"Currently showing lines {line_start}-{line_end} out of {total_lines} total lines. Use line_start and line_end to view more lines."
            )
        return {
            "formatted": "\n".join(lines_output),
            "totalResults": 1,
            "resultsShared": 1,
        }

    except requests.exceptions.RequestException as e:
        return {
            "formatted": f"Failed to connect to GitHub API: {str(e)}",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }


# Tool specification
GITHUB_READ_FILE_TOOL_SPEC = {
    "name": "github_read_file",
    "description": (
        "Read file contents from any GitHub repository with line range support.\n\n"
        "Fetches exact file contents in the given line range (default 300 lines, use line_start/line_end adjust). \n\n"
        "## When to use this tool\n\n"
        "- When reading example code, implementations, or documentation on a specific github file\n"
        "- When you found a file via github_list_repos, or github_find_examples and need its contents\n"
        "- When investigating specific code sections with line ranges\n"
        "- When reading from specific branches, tags, or commits\n"
        "## When NOT to use this tool\n\n"
        "- When you don't know the exact file path beforehand (use github_search_code or github_find_examples first)\n\n"
        "## Examples\n\n"
        "<example>\n"
        "// ML Workflow Step: Reading example code from for GRPO training with TRL\n"
        "// Use case: Read trainer class to understand API and methods\n"
        "{\n"
        "  repo: 'huggingface/trl',\n"
        "  path: 'trl/trainer/grpo_trainer.py',\n"
        "  line_start: 1,\n"
        "  line_end: 200\n"
        "}\n"
        "// Read class definition and constructor to understand parameters\n"
        "</example>\n\n"
        "<example>\n"
        "// ML Workflow Step: Study complete training script\n"
        "// Use case: Learn end-to-end VLM fine-tuning with GRPO\n"
        "{\n"
        "  repo: 'huggingface/trl',\n"
        "  path: 'examples/scripts/grpo_vlm.py'\n"
        "}\n"
        "// Returns first 300 lines of the file\n"
        "</example>\n\n"
        "<example>\n"
        "// ML Workflow Step: Check configuration patterns\n"
        "// Use case: Learn how to structure training configs\n"
        "{\n"
        "  repo: 'huggingface/transformers',\n"
        "  path: 'examples/pytorch/language-modeling/run_clm.py',\n"
        "  line_start: 50,\n"
        "  line_end: 150\n"
        "}\n"
        "// Read argument parsing and config setup section\n"
        "</example>"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository in format 'owner/repo' (e.g., 'github/github-mcp-server'). Required.",
            },
            "path": {
                "type": "string",
                "description": "Path to file in repository (e.g., 'src/index.js'). Required.",
            },
            "ref": {
                "type": "string",
                "description": "Git reference - branch name, tag, or commit SHA. Default: 'HEAD'.",
            },
            "line_start": {
                "type": "integer",
                "description": "Starting line number (1-indexed, inclusive). Optional.",
            },
            "line_end": {
                "type": "integer",
                "description": "Ending line number (1-indexed, inclusive). Optional.",
            },
        },
        "required": ["repo", "path"],
    },
}


async def github_read_file_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        result = read_file(
            repo=arguments["repo"],
            path=arguments["path"],
            ref=arguments.get("ref", "HEAD"),
            line_start=arguments.get("line_start"),
            line_end=arguments.get("line_end"),
        )
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error reading file: {str(e)}", False
