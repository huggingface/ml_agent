"""
GitHub Code Search Tool - Search code across GitHub with intelligent filtering

Maps user-friendly patterns to GitHub's Code Search API capabilities.
"""

import fnmatch
import os
import re
from typing import Any, Dict, Optional

import requests

from agent.tools.types import ToolResult


def _glob_match(text: str, pattern: str) -> bool:
    """Check if text matches glob pattern, supporting ** for multi-level paths"""
    if "**" in pattern:
        regex_pattern = pattern.replace("**", "<<<DOUBLESTAR>>>")
        regex_pattern = fnmatch.translate(regex_pattern)
        regex_pattern = regex_pattern.replace("<<<DOUBLESTAR>>>", ".*")
        return re.match(regex_pattern, text) is not None
    return fnmatch.fnmatch(text, pattern)


def _parse_repo_filter(repo_pattern: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse repository pattern into GitHub API filter and client-side glob pattern.

    Returns: (api_filter, client_glob)
    - api_filter: GitHub API filter string (e.g., "org:huggingface")
    - client_glob: Pattern for client-side filtering (e.g., "huggingface/trl*")

    Examples:
        "huggingface/trl" → ("repo:huggingface/trl", None)
        "huggingface/*" → ("org:huggingface", "huggingface/*")
        "huggingface/trl*" → ("org:huggingface", "huggingface/trl*")
        "huggingface" → ("org:huggingface", None)
        "*/*" → (None, "*/*")
    """
    if not repo_pattern:
        return None, None

    # Pattern: owner/repo (exact match)
    if "/" in repo_pattern and "*" not in repo_pattern and "?" not in repo_pattern:
        return f"repo:{repo_pattern}", None

    # Pattern: owner/* or owner/prefix* (org + client filter)
    if "/" in repo_pattern and ("*" in repo_pattern or "?" in repo_pattern):
        org_name = repo_pattern.split("/")[0]
        if "*" not in org_name and "?" not in org_name:
            return f"org:{org_name}", repo_pattern
        # Org name has wildcards - can't filter server-side
        return None, repo_pattern

    # Pattern: owner (just org name, no wildcards)
    if "*" not in repo_pattern and "?" not in repo_pattern:
        return f"org:{repo_pattern}", None

    # Pattern: */* or other complex patterns (client-side only)
    return None, repo_pattern


def _parse_path_filter(path_pattern: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse path pattern into GitHub API filter and client-side glob pattern.

    Returns: (api_filter, client_glob)

    Examples:
        "*.py" → ("extension:py", None)
        "**/*.py" → ("extension:py", None)
        "src/**/*.py" → ("extension:py", "src/**/*.py")
        "test_*.py" → ("extension:py", "test_*.py")
        "src/main.py" → ("path:src/main.py", None)
    """
    if not path_pattern:
        return None, None

    # Exact path (no wildcards)
    if "*" not in path_pattern and "?" not in path_pattern:
        return f"path:{path_pattern}", None

    # Extract extension if present
    ext_match = re.search(r"\*\.(\w+)$", path_pattern)
    if ext_match:
        extension = ext_match.group(1)
        api_filter = f"extension:{extension}"

        # Check if there's a directory prefix that needs client-side filtering
        # e.g., "src/**/*.py" needs client filter, "**/*.py" doesn't
        if path_pattern in [f"*.{extension}", f"**/*.{extension}"]:
            # Simple patterns - API filter is enough
            return api_filter, None
        else:
            # Complex pattern - need client-side filter too
            return api_filter, path_pattern

    # Pattern like "test_*.py" or "README*" - use filename with client filter
    # GitHub's filename: doesn't support wildcards, so we rely on client-side
    if "/" not in path_pattern:
        # Try to extract extension for API filtering
        if "." in path_pattern:
            parts = path_pattern.rsplit(".", 1)
            if "*" not in parts[-1] and "?" not in parts[-1]:
                # Extension is clean
                return f"extension:{parts[-1]}", path_pattern
        # No extension or complex - client-side only
        return None, path_pattern

    # Complex path pattern - client-side only
    return None, path_pattern


def search_code(
    query: str,
    repo_pattern: Optional[str] = None,
    path_pattern: Optional[str] = None,
    regex: bool = False,
    max_results: int = 20,
) -> ToolResult:
    """
    Search for code across GitHub with intelligent pattern matching.

    This tool intelligently maps user patterns to GitHub's Code Search API capabilities:

    Repository Patterns:
        - "owner/repo" → Searches exact repository
        - "owner/*" or "owner" → Searches all repos in organization
        - "*/*" → Searches all GitHub (no repo filter)
        - Wildcards trigger client-side filtering when needed

    Path Patterns:
        - "*.py" → Searches all Python files
        - "**/*.js" → Searches all JavaScript files (any directory)
        - "src/**/*.py" → Python files in src/ (uses client-side filtering)
        - "test_*.py" → Files matching pattern (client-side filtering)
        - "path/to/file.py" → Exact file path

    Args:
        query: Search term or pattern to find in code
        repo_pattern: Repository pattern (e.g., "huggingface/trl", "huggingface/*", "huggingface")
        path_pattern: File path pattern (e.g., "*.py", "src/**/*.js")
        regex: If True, treat query as regular expression
        max_results: Maximum number of results to return (default 20)

    Returns:
        ToolResult with code matches and snippets
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {
            "formatted": "Error: GITHUB_TOKEN environment variable is required",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    # Build GitHub API query
    query_parts = []

    # Add search term
    if regex:
        query_parts.append(f"/{query}/")
    else:
        query_parts.append(f'"{query}"' if " " in query else query)

    # Parse repository filter
    repo_api_filter, repo_client_glob = _parse_repo_filter(repo_pattern)
    if repo_api_filter:
        query_parts.append(repo_api_filter)

    # Parse path filter
    path_api_filter, path_client_glob = _parse_path_filter(path_pattern)
    if path_api_filter:
        query_parts.append(path_api_filter)

    github_query = " ".join(query_parts)

    headers = {
        "Accept": "application/vnd.github.text-match+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }

    all_matches = []
    page = 1
    per_page = min(100, max_results)

    try:
        while len(all_matches) < max_results:
            params = {
                "q": github_query,
                "page": page,
                "per_page": per_page,
            }

            response = requests.get(
                "https://api.github.com/search/code",
                headers=headers,
                params=params,
                timeout=30,
            )

            if response.status_code == 403:
                error_data = response.json()
                return {
                    "formatted": f"GitHub API rate limit or permission error: {error_data.get('message', 'Unknown error')}",
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
            items = data.get("items", [])

            if not items:
                break

            for item in items:
                repo_name = item.get("repository", {}).get("full_name", "unknown")
                file_path = item.get("path", "")
                sha = item.get("sha", "")

                # Apply client-side filtering
                if repo_client_glob and not _glob_match(repo_name, repo_client_glob):
                    continue
                if path_client_glob and not _glob_match(file_path, path_client_glob):
                    continue

                # Extract text matches
                text_matches = item.get("text_matches", [])
                if text_matches:
                    for text_match in text_matches:
                        fragment = text_match.get("fragment", "")
                        lines = fragment.split("\n")
                        line_count = len([line for line in lines if line.strip()])

                        all_matches.append(
                            {
                                "repo": repo_name,
                                "path": file_path,
                                "ref": sha,
                                "line_start": 1,
                                "line_end": line_count,
                                "snippet": fragment.strip(),
                                "url": item.get("html_url", ""),
                            }
                        )
                else:
                    all_matches.append(
                        {
                            "repo": repo_name,
                            "path": file_path,
                            "ref": sha,
                            "line_start": 1,
                            "line_end": 1,
                            "snippet": "(snippet not available)",
                            "url": item.get("html_url", ""),
                        }
                    )

            if len(all_matches) >= data.get("total_count", 0):
                break

            page += 1

    except requests.exceptions.RequestException as e:
        return {
            "formatted": f"Failed to connect to GitHub API: {str(e)}",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    results = all_matches[:max_results]

    if not results:
        return {
            "formatted": f"No code matches found for query: {query}",
            "totalResults": 0,
            "resultsShared": 0,
        }

    # Format output
    lines_output = [f"**Found {len(results)} code matches:**\n"]

    for i, match in enumerate(results, 1):
        lines_output.append(f"{i}. **{match['repo']}:{match['path']}**")
        lines_output.append(
            f"   Lines: {match['line_start']}-{match['line_end']} | Ref: {match['ref'][:7]}"
        )
        lines_output.append(f"   URL: {match['url']}")

        # Copyable parameters for read_file tool
        read_params = f"{{'repo': '{match['repo']}', 'path': '{match['path']}', 'ref': '{match['ref'][:7]}'}}"
        lines_output.append(f"   To read, use: {read_params}")

        # Show snippet (first 5 lines)
        snippet_lines = match["snippet"].split("\n")[:5]
        if snippet_lines:
            lines_output.append("   ```")
            for line in snippet_lines:
                lines_output.append(f"   {line}")
            if len(match["snippet"].split("\n")) > 5:
                lines_output.append("   ...")
            lines_output.append("   ```")
        lines_output.append("")

    return {
        "formatted": "\n".join(lines_output),
        "totalResults": len(results),
        "resultsShared": len(results),
    }


# Tool specification
GITHUB_SEARCH_CODE_TOOL_SPEC = {
    "name": "github_search_code",
    "description": (
        "Search for code patterns across GitHub repositories with intelligent pattern matching.\n\n"
        "Searches for specific code patterns, functions, classes, or implementations across GitHub. "
        "Intelligently maps patterns to GitHub's Code Search API for efficient server-side filtering, "
        "with automatic client-side filtering for complex patterns. Returns code snippets with context.\n\n"
        "## When to use this tool\n\n"
        "- When searching for specific code patterns, functions, or classes across repositories\n"
        "- When looking for implementation examples of specific methods or APIs\n"
        "- When you need to find where specific code exists across multiple files or repos\n"
        "- When investigating how a feature is implemented in different repositories\n"
        "- When searching for TODO comments, specific patterns, or code structures\n"
        "- Use this for searching actual implementation code (not examples - use github_find_examples for those)\n\n"
        "## When NOT to use this tool\n\n"
        "- When looking for example files or tutorials (use github_find_examples instead)\n"
        "- When you already know the exact file path (use github_read_file directly)\n"
        "- When you need to list repositories (use github_list_repos instead)\n\n"
        "## Repository Patterns\n\n"
        "- **Exact repo**: `'huggingface/trl'` → Searches only that repository\n"
        "- **Organization**: `'huggingface'` or `'huggingface/*'` → All repos in organization\n"
        "- **All GitHub**: `'*/*'` or omit repo_pattern → Searches across all GitHub\n"
        "- **Wildcards**: `'huggingface/trl*'` → Automatic client-side filtering for complex patterns\n\n"
        "## Path Patterns\n\n"
        "- **Extension**: `'*.py'` or `'**/*.py'` → All Python files\n"
        "- **Directory**: `'src/**/*.js'` → JavaScript files in src/ directory (client-filtered)\n"
        "- **Pattern**: `'test_*.py'` → Files matching pattern (client-filtered)\n"
        "- **Exact path**: `'README.md'` → Specific file\n\n"
        "## How it works\n\n"
        "1. Parses repository and path patterns\n"
        "2. Converts to GitHub API filters when possible (server-side, fast)\n"
        "3. Falls back to client-side filtering for complex patterns\n"
        "4. Returns code snippets with line numbers, URLs, and file refs\n"
        "5. Results can be used directly with github_read_file tool\n\n"
        "## Examples\n\n"
        "<example>\n"
        "// ML Workflow Step: Find how AutoModelForCausalLM is used\n"
        "// Use case: Learning best practices for loading LLMs in TRL\n"
        "{\n"
        "  query: 'AutoModelForCausalLM.from_pretrained',\n"
        "  repo_pattern: 'huggingface/trl',\n"
        "  path_pattern: '*.py'\n"
        "}\n"
        "// Finds all model loading patterns with quantization, device_map, etc.\n"
        "</example>\n\n"
        "<example>\n"
        "// ML Workflow Step: Discover TrainingArguments configurations\n"
        "// Use case: Setting up training hyperparameters correctly\n"
        "{\n"
        "  query: 'TrainingArguments',\n"
        "  repo_pattern: 'huggingface/transformers',\n"
        "  path_pattern: 'examples/**/*.py',\n"
        "  max_results: 10\n"
        "}\n"
        "// Shows various TrainingArguments setups across different tasks\n"
        "</example>\n\n"
        "<example>\n"
        "// ML Workflow Step: Find dataset preprocessing patterns\n"
        "// Use case: Learning how to prepare data for instruction tuning\n"
        "{\n"
        "  query: 'map(tokenize',\n"
        "  repo_pattern: 'huggingface',\n"
        "  path_pattern: '*.py'\n"
        "}\n"
        "// Discovers tokenization and dataset mapping patterns\n"
        "</example>\n\n"
        "<example>\n"
        "// ML Workflow Step: Find all Trainer class implementations\n"
        "// Use case: Understanding available trainer variants for different tasks\n"
        "{\n"
        "  query: 'class \\\\w+Trainer\\\\(',\n"
        "  repo_pattern: 'huggingface/trl',\n"
        "  path_pattern: 'trl/trainer/**/*.py',\n"
        "  regex: true\n"
        "}\n"
        "// Lists: GRPOTrainer, DPOTrainer, PPOTrainer, RewardTrainer, etc.\n"
        "</example>"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term or pattern to find in code. Required.",
            },
            "repo_pattern": {
                "type": "string",
                "description": "Repository pattern: 'owner/repo' (exact), 'owner' (org), 'owner/*' (org with filter), '*/*' (all). Optional.",
            },
            "path_pattern": {
                "type": "string",
                "description": "File path pattern: '*.ext' (extension), 'dir/**/*.ext' (directory), 'pattern*.ext' (name pattern). Optional.",
            },
            "regex": {
                "type": "boolean",
                "description": "If true, treat query as regular expression. Default: false.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Default: 20.",
            },
        },
        "required": ["query"],
    },
}


async def github_search_code_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        result = search_code(
            query=arguments["query"],
            repo_pattern=arguments.get("repo_pattern"),
            path_pattern=arguments.get("path_pattern"),
            regex=arguments.get("regex", False),
            max_results=arguments.get("max_results", 20),
        )
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error searching code: {str(e)}", False
