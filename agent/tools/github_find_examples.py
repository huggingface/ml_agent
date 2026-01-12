"""
GitHub Find Examples Tool - Discover examples, tutorials, and guides for any library

Lists all files in a repository and performs deterministic keyword search.
"""

import os
from typing import Any, Dict, List

import requests
from thefuzz import fuzz

from agent.tools.types import ToolResult

# In order of priority (lower index = higher priority for sorting)
EXAMPLE_PATTERNS = [
    "scripts",
    # General example patterns (catch-all, lower priority)
    "examples",
    "example",
    # Notebook patterns
    "notebooks",
    "notebook",
    # Tutorial/learning patterns
    "tutorials",
    "tutorial",
    "quickstart",
    "walkthroughs",
    "walkthrough",
    # Cookbook/recipe patterns
    "cookbook",
    "cookbooks",
    "recipes",
    "recipe",
    # Demo/sample patterns
    "demos",
    "demo",
    "samples",
    "sample",
    # Other patterns
    "guides",
    "guide",
    "getting-started",
    "getting_started",
    "playground",
    "howto",
    "how-to",
    "use-cases",
    "usecases",
    "use_cases",
    "sandbox",
    "showcase",
]


def _get_repo_tree(org: str, repo: str, token: str) -> tuple[List[Dict[str, Any]], str]:
    """Get all files in a repository recursively. Returns (files, error_message)"""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }

    full_repo = f"{org}/{repo}"

    # Get default branch
    try:
        response = requests.get(
            f"https://api.github.com/repos/{full_repo}", headers=headers, timeout=10
        )
        if response.status_code == 404:
            return [], "not_found"
        if response.status_code != 200:
            return [], f"API error: {response.status_code}"

        repo_data = response.json()
        default_branch = repo_data.get("default_branch", "main")
    except Exception as e:
        return [], f"Error fetching repo: {str(e)}"

    # Get repository tree recursively
    try:
        response = requests.get(
            f"https://api.github.com/repos/{full_repo}/git/trees/{default_branch}",
            headers=headers,
            params={"recursive": "1"},
            timeout=30,
        )
        if response.status_code != 200:
            return [], f"Error fetching tree: {response.status_code}"

        data = response.json()
        tree = data.get("tree", [])

        # Filter to only include files (not directories)
        files = [
            {
                "path": item["path"],
                "ref": item["sha"],
                "size": item.get("size", 0),
                "url": f"https://github.com/{full_repo}/blob/{default_branch}/{item['path']}",
            }
            for item in tree
            if item["type"] == "blob"
        ]

        return files, ""
    except Exception as e:
        return [], f"Error processing tree: {str(e)}"


def _search_similar_repos(org: str, repo: str, token: str) -> List[Dict[str, Any]]:
    """Search for similar repository names in the organization"""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
    }

    # Search for repos in the org with similar name
    query = f"org:{org} {repo}"

    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            headers=headers,
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 10},
            timeout=30,
        )

        if response.status_code != 200:
            return []

        data = response.json()
        items = data.get("items", [])

        return [
            {
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "description": item.get("description"),
                "stars": item.get("stargazers_count", 0),
                "url": item.get("html_url"),
            }
            for item in items
        ]
    except Exception:
        return []


def _score_against_example_patterns(file_path: str) -> int:
    """Score file against example patterns using token_set_ratio"""
    scores = []
    for pattern in EXAMPLE_PATTERNS:
        score = fuzz.token_set_ratio(pattern.lower(), file_path.lower())
        scores.append(score)
    return max(scores) if scores else 0


def _score_against_keyword(file_path: str, keyword: str) -> int:
    """Calculate fuzzy match score for a file path against a keyword"""
    # Use partial_ratio for substring matching (good for paths)
    # Also check token_set_ratio for word-level matching
    partial_score = fuzz.partial_ratio(keyword.lower(), file_path.lower())
    token_score = fuzz.token_set_ratio(keyword.lower(), file_path.lower())

    # Return the higher of the two
    return max(partial_score, token_score)


def _get_pattern_priority(file_path: str) -> tuple[int, int, int]:
    """
    Get priority of a file path based on which example pattern directory it's in.

    Returns: (in_examples_dir, pattern_priority, path_depth)
    - in_examples_dir: 0 if in examples/ directory, 1 otherwise (lower is better)
    - pattern_priority: Index in EXAMPLE_PATTERNS (lower is better), or 999 if no match
    - path_depth: Number of path segments (lower is better)

    Note: Prioritizes files in "examples/" directory first, then by most specific pattern match.
    E.g., "examples/scripts/train.py" is better than "scripts/util.py"
    """
    path_lower = file_path.lower()
    path_parts = path_lower.split("/")

    # Check if file is in examples/ directory (highest priority)
    in_examples_dir = 0 if (path_parts[0] in ["examples", "example"]) else 1

    # Find ALL matching patterns and use the best (lowest index) one
    # But prefer deeper matches (more specific) over shallow ones
    best_priority = 999
    best_depth_at_match = -1

    for i, pattern in enumerate(EXAMPLE_PATTERNS):
        # Check if pattern appears as a directory component in the path
        if pattern in path_parts:
            # Find the depth where this pattern appears (rightmost occurrence)
            depth = len(path_parts) - 1 - path_parts[::-1].index(pattern)

            # Prefer deeper matches, or better priority if at same depth
            if depth > best_depth_at_match or (
                depth == best_depth_at_match and i < best_priority
            ):
                best_priority = i
                best_depth_at_match = depth

    return (in_examples_dir, best_priority, len(path_parts))


def _handle_repo_tree_errors(
    all_files: List[Dict[str, Any]],
    error: str,
    org: str,
    repo: str,
    token: str,
) -> ToolResult | None:
    """Handle errors from repo tree fetch. Returns ToolResult if error, None if OK."""
    if error == "not_found":
        similar_repos = _search_similar_repos(org, repo, token)

        if not similar_repos:
            return {
                "formatted": f"Repository '{org}/{repo}' not found and no similar repositories found.",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        # Format similar repos
        lines = [f"**Repository '{org}/{repo}' not found. Similar repositories:**\n"]
        for i, r in enumerate(similar_repos, 1):
            lines.append(f"{i}. **{r['full_name']}** (⭐ {r['stars']:,} stars)")
            if r["description"]:
                desc = (
                    r["description"][:100] + "..."
                    if len(r["description"]) > 100
                    else r["description"]
                )
                lines.append(f"   {desc}")
            lines.append(f"   {r['url']}\n")

        return {
            "formatted": "\n".join(lines),
            "totalResults": len(similar_repos),
            "resultsShared": len(similar_repos),
            "isError": True,
        }

    if error:
        return {
            "formatted": f"Error accessing repository '{org}/{repo}': {error}",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    if not all_files:
        return {
            "formatted": f"No files found in repository '{org}/{repo}'",
            "totalResults": 0,
            "resultsShared": 0,
        }

    return None


def find_examples(
    keyword: str = "",
    repo: str = "",
    org: str = "huggingface",
    max_results: int = 10,
    min_score: int = 80,
) -> ToolResult:
    """
    Find example files in a repository using fuzzy matching.

    Args:
        keyword: Keyword to fuzzy match against file paths (e.g., "grpo")
        repo: Repository name (e.g., "trl")
        org: GitHub organization (default: "huggingface")
        max_results: Maximum number of results (default 50)
        min_score: Minimum fuzzy match score (0-100, default 60)

    Returns:
        ToolResult with matching files, or similar repos if repo not found
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {
            "formatted": "Error: GITHUB_TOKEN environment variable is required",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    if not repo:
        return {
            "formatted": "Error: repo parameter is required",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    # Get all files in the repository
    all_files, error = _get_repo_tree(org, repo, token)

    # Handle errors (not found, API errors, empty repo)
    if error_result := _handle_repo_tree_errors(all_files, error, org, repo, token):
        return error_result

    # Step 1: Filter files by example patterns (score >= 60)
    example_threshold = 60
    example_files = []
    for file in all_files:
        example_score = _score_against_example_patterns(file["path"])
        if example_score >= example_threshold:
            example_files.append({**file, "example_score": example_score})

    if not example_files:
        return {
            "formatted": f"No example files found in {org}/{repo} (no files match example patterns with score >= {example_threshold}).",
            "totalResults": 0,
            "resultsShared": 0,
        }

    # Step 2: If keyword provided, score and filter by keyword
    if keyword:
        scored_files = []
        for file in example_files:
            keyword_score = _score_against_keyword(file["path"], keyword)
            if keyword_score >= min_score:
                scored_files.append({**file, "score": keyword_score})

        if not scored_files:
            return {
                "formatted": f"No files found in {org}/{repo} matching keyword '{keyword}' (min score: {min_score}) among {len(example_files)} example files.",
                "totalResults": 0,
                "resultsShared": 0,
            }

        # Sort by keyword score (descending) for best matches first
        scored_files.sort(key=lambda x: x["score"], reverse=True)
    else:
        # No keyword: prioritize by pattern directory, then path depth
        scored_files = []
        for file in example_files:
            in_examples_dir, pattern_priority, path_depth = _get_pattern_priority(
                file["path"]
            )
            scored_files.append(
                {
                    **file,
                    "score": file["example_score"],
                    "in_examples_dir": in_examples_dir,
                    "pattern_priority": pattern_priority,
                    "path_depth": path_depth,
                }
            )

        if not scored_files:
            return {
                "formatted": f"No example files found in {org}/{repo}.",
                "totalResults": 0,
                "resultsShared": 0,
            }

        # Sort by: 1) files in examples/ dir first, 2) pattern priority (scripts > datasets > etc), 3) path depth, 4) path name
        scored_files.sort(
            key=lambda x: (
                x["in_examples_dir"],
                x["pattern_priority"],
                x["path_depth"],
                x["path"],
            )
        )

    # Limit results
    results = scored_files[:max_results]

    # Format output
    keyword_desc = f" matching '{keyword}'" if keyword else ""
    lines = [f"**Found {len(results)} example files in {org}/{repo}{keyword_desc}:**"]
    if len(scored_files) > max_results:
        lines[0] += f" (showing {max_results} of {len(scored_files)})"
    lines.append("")

    for i, file in enumerate(results, 1):
        lines.append(f"{i}. **{file['path']}**")
        lines.append(f"   Size: {file['size']:,} bytes | Ref: {file['ref'][:7]}")
        lines.append(f"   URL: {file['url']}")

        # Copyable parameters for read_file tool
        read_params = f"{{'repo': '{org}/{repo}', 'path': '{file['path']}'}}"
        lines.append(f"   To read, use: {read_params}")
        lines.append("")

    return {
        "formatted": "\n".join(lines),
        "totalResults": len(results),
        "resultsShared": len(results),
    }


# Tool specification
GITHUB_FIND_EXAMPLES_TOOL_SPEC = {
    "name": "github_find_examples",
    "description": (
        "Discover working code examples, tutorials, scripts, and demos in GitHub repositories. "
        "⚠️ CRITICAL: ALWAYS use this BEFORE implementing ML tasks - find working reference code first. "
        "Your training data may be outdated; real repository examples show current best practices. "
        "**Use when:** (1) Starting any ML implementation (training, inference, evaluation), "
        "(2) User asks 'how to' questions about libraries, (3) Need reference implementations, "
        "(4) Exploring library capabilities, (5) Before writing training/processing scripts. "
        "**Pattern:** github_find_examples (discover) → github_read_file (study code) → implement with researched approach. "
        "Returns: List of example files (scripts/notebooks/tutorials) with paths and URLs, sorted by relevance. "
        "**Then:** Use github_read_file to read the actual implementation code. "
        "**Critical for reliability:** Real examples prevent outdated API usage and show proven patterns. "
        "## How it works\n\n"
        "1. Fetches all example files (examples/, scripts/, tutorials/, demos/, notebooks/, etc.) from repository\n"
        "2. If keyword provided, scores files against keyword using fuzzy matching\n"
        "3. Returns best matches sorted by relevance and pattern priority\n"
        "4. Provides copyable parameters for github_read_file tool\n\n"
        "## Examples\n\n"
        "<example>\n"
        "// ML Workflow Step: Find GRPO training examples before implementation\n"
        "// Task: Starting GRPO fine-tuning project, need reference implementation\n"
        "{\n"
        "  keyword: 'grpo',\n"
        "  repo: 'trl',\n"
        "  org: 'huggingface'\n"
        "}\n"
        "// Returns: examples/scripts/grpo_agent.py, examples/scripts/grpo_vlm.py\n"
        "// Next step: github_read_file to study working implementation\n"
        "</example>\n\n"
        "<example>\n"
        "// ML Workflow Step: Discover all available training methods\n"
        "// Task: Exploring TRL training options before choosing approach\n"
        "{\n"
        "  repo: 'trl',\n"
        "  org: 'huggingface',\n"
        "  max_results: 20\n"
        "}\n"
        "// Lists: SFT, DPO, GRPO, PPO, reward modeling examples\n"
        "// Helps user choose appropriate method\n"
        "</example>\n\n"
        "<example>\n"
        "// ML Workflow Step: Find LoRA fine-tuning examples\n"
        "// Task: Learning parameter-efficient fine-tuning patterns\n"
        "{\n"
        "  keyword: 'lora',\n"
        "  repo: 'peft',\n"
        "  org: 'huggingface'\n"
        "}\n"
        "// Discovers LoRA configuration and training examples\n"
        "// Shows current PEFT API usage patterns\n"
        "</example>"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Keyword to fuzzy match against file paths (e.g., 'grpo', 'sft').",
            },
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'trl', 'transformers'). Required.",
            },
            "org": {
                "type": "string",
                "description": "GitHub organization or username. Default: 'huggingface'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Default: 50.",
            },
            "min_score": {
                "type": "integer",
                "description": "Minimum fuzzy match score (0-100). Default: 60.",
            },
        },
        "required": ["repo"],
    },
}


async def github_find_examples_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        result = find_examples(
            keyword=arguments.get("keyword", ""),
            repo=arguments["repo"],
            org=arguments.get("org", "huggingface"),
            max_results=arguments.get("max_results", 50),
            min_score=arguments.get("min_score", 60),
        )
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error finding examples: {str(e)}", False
