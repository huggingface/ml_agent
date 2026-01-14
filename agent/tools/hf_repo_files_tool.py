"""
HF Repo Files Tool - File operations on Hugging Face repositories

Operations: list, read, upload, delete
"""

import asyncio
from typing import Any, Dict, Literal, Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from agent.tools.types import ToolResult

OperationType = Literal["list", "read", "upload", "delete"]


async def _async_call(func, *args, **kwargs):
    """Wrap synchronous HfApi calls for async context."""
    return await asyncio.to_thread(func, *args, **kwargs)


def _build_repo_url(repo_id: str, repo_type: str = "model") -> str:
    """Build the Hub URL for a repository."""
    if repo_type == "model":
        return f"https://huggingface.co/{repo_id}"
    return f"https://huggingface.co/{repo_type}s/{repo_id}"


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


class HfRepoFilesTool:
    """Tool for file operations on HF repos."""

    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi(token=hf_token)

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute the specified operation."""
        operation = args.get("operation")

        if not operation:
            return self._help()

        try:
            handlers = {
                "list": self._list,
                "read": self._read,
                "upload": self._upload,
                "delete": self._delete,
            }

            handler = handlers.get(operation)
            if handler:
                return await handler(args)
            else:
                return self._error(f"Unknown operation: {operation}. Valid: list, read, upload, delete")

        except RepositoryNotFoundError:
            return self._error(f"Repository not found: {args.get('repo_id')}")
        except EntryNotFoundError:
            return self._error(f"File not found: {args.get('path')}")
        except Exception as e:
            return self._error(f"Error: {str(e)}")

    def _help(self) -> ToolResult:
        """Show usage instructions."""
        return {
            "formatted": """**hf_repo_files** - File operations on HF repos

**Operations:**
- `list` - List files: `{"operation": "list", "repo_id": "gpt2"}`
- `read` - Read file: `{"operation": "read", "repo_id": "gpt2", "path": "config.json"}`
- `upload` - Upload: `{"operation": "upload", "repo_id": "my-model", "path": "README.md", "content": "..."}`
- `delete` - Delete: `{"operation": "delete", "repo_id": "my-model", "patterns": ["*.tmp"]}`

**Common params:** repo_id (required), repo_type (model/dataset/space), revision (default: main)""",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _list(self, args: Dict[str, Any]) -> ToolResult:
        """List files in a repository."""
        repo_id = args.get("repo_id")
        if not repo_id:
            return self._error("repo_id is required")

        repo_type = args.get("repo_type", "model")
        revision = args.get("revision", "main")
        path = args.get("path", "")

        items = list(await _async_call(
            self.api.list_repo_tree,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=path,
            recursive=True,
        ))

        if not items:
            return {"formatted": f"No files in {repo_id}", "totalResults": 0, "resultsShared": 0}

        lines = []
        total_size = 0
        for item in sorted(items, key=lambda x: x.path):
            if hasattr(item, "size") and item.size:
                total_size += item.size
                lines.append(f"{item.path} ({_format_size(item.size)})")
            else:
                lines.append(f"{item.path}/")

        url = _build_repo_url(repo_id, repo_type)
        response = f"**{repo_id}** ({len(items)} files, {_format_size(total_size)})\n{url}/tree/{revision}\n\n" + "\n".join(lines)

        return {"formatted": response, "totalResults": len(items), "resultsShared": len(items)}

    async def _read(self, args: Dict[str, Any]) -> ToolResult:
        """Read file content from a repository."""
        repo_id = args.get("repo_id")
        path = args.get("path")

        if not repo_id:
            return self._error("repo_id is required")
        if not path:
            return self._error("path is required")

        repo_type = args.get("repo_type", "model")
        revision = args.get("revision", "main")
        max_chars = args.get("max_chars", 50000)

        file_path = await _async_call(
            hf_hub_download,
            repo_id=repo_id,
            filename=path,
            repo_type=repo_type,
            revision=revision,
            token=self.api.token,
        )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            truncated = len(content) > max_chars
            if truncated:
                content = content[:max_chars]

            url = f"{_build_repo_url(repo_id, repo_type)}/blob/{revision}/{path}"
            response = f"**{path}**{' (truncated)' if truncated else ''}\n{url}\n\n```\n{content}\n```"

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except UnicodeDecodeError:
            import os
            size = os.path.getsize(file_path)
            return {"formatted": f"Binary file ({_format_size(size)})", "totalResults": 1, "resultsShared": 1}

    async def _upload(self, args: Dict[str, Any]) -> ToolResult:
        """Upload content to a repository."""
        repo_id = args.get("repo_id")
        path = args.get("path")
        content = args.get("content")

        if not repo_id:
            return self._error("repo_id is required")
        if not path:
            return self._error("path is required")
        if content is None:
            return self._error("content is required")

        repo_type = args.get("repo_type", "model")
        revision = args.get("revision", "main")
        create_pr = args.get("create_pr", False)
        commit_message = args.get("commit_message", f"Upload {path}")

        file_bytes = content.encode("utf-8") if isinstance(content, str) else content

        result = await _async_call(
            self.api.upload_file,
            path_or_fileobj=file_bytes,
            path_in_repo=path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            create_pr=create_pr,
        )

        url = _build_repo_url(repo_id, repo_type)
        if create_pr and hasattr(result, "pr_url"):
            response = f"**Uploaded as PR**\n{result.pr_url}"
        else:
            response = f"**Uploaded:** {path}\n{url}/blob/{revision}/{path}"

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    async def _delete(self, args: Dict[str, Any]) -> ToolResult:
        """Delete files from a repository."""
        repo_id = args.get("repo_id")
        patterns = args.get("patterns")

        if not repo_id:
            return self._error("repo_id is required")
        if not patterns:
            return self._error("patterns is required (list of paths/wildcards)")

        if isinstance(patterns, str):
            patterns = [patterns]

        repo_type = args.get("repo_type", "model")
        revision = args.get("revision", "main")
        create_pr = args.get("create_pr", False)
        commit_message = args.get("commit_message", f"Delete {', '.join(patterns)}")

        await _async_call(
            self.api.delete_files,
            repo_id=repo_id,
            delete_patterns=patterns,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            create_pr=create_pr,
        )

        response = f"**Deleted:** {', '.join(patterns)} from {repo_id}"
        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    def _error(self, message: str) -> ToolResult:
        """Return an error result."""
        return {"formatted": message, "totalResults": 0, "resultsShared": 0, "isError": True}


# Tool specification
HF_REPO_FILES_TOOL_SPEC = {
    "name": "hf_repo_files",
    "description": (
        "Read and write files in HF repos (models/datasets/spaces).\n\n"
        "## Operations\n"
        "- **list**: List files with sizes and structure\n"
        "- **read**: Read file content (text files only)\n"
        "- **upload**: Upload content to repo (can create PR)\n"
        "- **delete**: Delete files/folders (supports wildcards like *.tmp)\n\n"
        "## Use when\n"
        "- Need to see what files exist in a repo\n"
        "- Want to read config.json, README.md, or other text files\n"
        "- Uploading training scripts, configs, or results to a repo\n"
        "- Cleaning up temporary files from a repo\n\n"
        "## Examples\n"
        '{"operation": "list", "repo_id": "meta-llama/Llama-2-7b"}\n'
        '{"operation": "read", "repo_id": "gpt2", "path": "config.json"}\n'
        '{"operation": "upload", "repo_id": "my-model", "path": "README.md", "content": "# My Model"}\n'
        '{"operation": "upload", "repo_id": "org/model", "path": "fix.py", "content": "...", "create_pr": true}\n'
        '{"operation": "delete", "repo_id": "my-model", "patterns": ["*.tmp", "logs/"]}\n\n'
        "## Notes\n"
        "- For binary files (safetensors, bin), use list to see them but can't read content\n"
        "- upload/delete require approval (can overwrite/destroy data)\n"
        "- Use create_pr=true to propose changes instead of direct commit\n"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "read", "upload", "delete"],
                "description": "Operation: list, read, upload, delete",
            },
            "repo_id": {
                "type": "string",
                "description": "Repository ID (e.g., 'username/repo-name')",
            },
            "repo_type": {
                "type": "string",
                "enum": ["model", "dataset", "space"],
                "description": "Repository type (default: model)",
            },
            "revision": {
                "type": "string",
                "description": "Branch/tag/commit (default: main)",
            },
            "path": {
                "type": "string",
                "description": "File path for read/upload",
            },
            "content": {
                "type": "string",
                "description": "File content for upload",
            },
            "patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Patterns to delete (e.g., ['*.tmp', 'logs/'])",
            },
            "create_pr": {
                "type": "boolean",
                "description": "Create PR instead of direct commit",
            },
            "commit_message": {
                "type": "string",
                "description": "Custom commit message",
            },
        },
        "required": ["operation"],
    },
}


async def hf_repo_files_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router."""
    try:
        tool = HfRepoFilesTool()
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error: {str(e)}", False
