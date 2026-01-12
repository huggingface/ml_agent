"""
Private HF Repos Tool - Manage private Hugging Face repositories

PRIMARY USE: Store job outputs, training scripts, and logs from HF Jobs.
Since job results are ephemeral, this tool provides persistent storage in private repos.

SECONDARY USE: Read back stored files and list repo contents.
"""

import asyncio
from typing import Any, Dict, Literal, Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from agent.tools.types import ToolResult

# Operation names
OperationType = Literal[
    "upload_file", "create_repo", "check_repo", "list_files", "read_file"
]


async def _async_call(func, *args, **kwargs):
    """Wrap synchronous HfApi calls for async context."""
    return await asyncio.to_thread(func, *args, **kwargs)


def _build_repo_url(repo_id: str, repo_type: str = "dataset") -> str:
    """Build the Hub URL for a repository."""
    type_path = "" if repo_type == "model" else f"{repo_type}s"
    return f"https://huggingface.co/{type_path}/{repo_id}".replace("//", "/")


def _content_to_bytes(content: str | bytes) -> bytes:
    """Convert string or bytes content to bytes."""
    if isinstance(content, str):
        return content.encode("utf-8")
    return content


class PrivateHfRepoTool:
    """Tool for managing private Hugging Face repositories."""

    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi(token=hf_token)

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the specified upload operation."""
        operation = params.get("operation")
        args = params.get("args", {})

        # If no operation provided, return usage instructions
        if not operation:
            return self._show_help()

        # Normalize operation name
        operation = operation.lower()

        # Check if help is requested
        if args.get("help"):
            return self._show_operation_help(operation)

        try:
            # Route to appropriate handler
            if operation == "upload_file":
                return await self._upload_file(args)
            elif operation == "create_repo":
                return await self._create_repo(args)
            elif operation == "check_repo":
                return await self._check_repo(args)
            elif operation == "list_files":
                return await self._list_files(args)
            elif operation == "read_file":
                return await self._read_file(args)
            else:
                return {
                    "formatted": f'Unknown operation: "{operation}"\n\n'
                    "Available operations: upload_file, create_repo, check_repo, list_files, read_file\n\n"
                    "Call this tool with no operation for full usage instructions.",
                    "totalResults": 0,
                    "resultsShared": 0,
                    "isError": True,
                }

        except HfHubHTTPError as e:
            return {
                "formatted": f"API Error: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }
        except Exception as e:
            return {
                "formatted": f"Error executing {operation}: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

    def _show_help(self) -> ToolResult:
        """Show usage instructions when tool is called with no arguments."""
        usage_text = """# Private HF Repos Tool

**PRIMARY USE:** Store job outputs, scripts, and logs from HF Jobs to private repos.
Since job results are ephemeral, use this tool for persistent storage.

**SECONDARY USE:** Read back stored files and list repo contents.

## Available Commands

### Write Operations
- **upload_file** - Upload file content to a repository
- **create_repo** - Create a new private repository

### Read Operations
- **list_files** - List all files in a repository
- **read_file** - Read content of a specific file from a repository
- **check_repo** - Check if a repository exists

## Examples

### Upload a script to a dataset repo
Call this tool with:
```json
{
  "operation": "upload_file",
  "args": {
    "file_content": "import pandas as pd\\nprint('Hello from HF!')",
    "path_in_repo": "scripts/hello.py",
    "repo_id": "my-dataset",
    "repo_type": "dataset",
    "create_if_missing": true,
    "commit_message": "Add hello script"
  }
}
```

### Upload logs from a job
Call this tool with:
```json
{
  "operation": "upload_file",
  "args": {
    "file_content": "Job started...\\nJob completed successfully!",
    "path_in_repo": "jobs/job-abc123/logs.txt",
    "repo_id": "job-results",
    "create_if_missing": true
  }
}
```

### Create a repository
Call this tool with:
```json
{
  "operation": "create_repo",
  "args": {
    "repo_id": "my-results",
    "repo_type": "dataset"
  }
}
```

### Create a Space
Call this tool with:
```json
{
  "operation": "create_repo",
  "args": {
    "repo_id": "my-gradio-app",
    "repo_type": "space",
    "space_sdk": "gradio"
  }
}
```
Note: Repositories are always created as private. For spaces, `space_sdk` is required (gradio, streamlit, static, or docker).

### Check if a repository exists
Call this tool with:
```json
{
  "operation": "check_repo",
  "args": {
    "repo_id": "my-dataset",
    "repo_type": "dataset"
  }
}
```

### List files in a repository
Call this tool with:
```json
{
  "operation": "list_files",
  "args": {
    "repo_id": "job-results",
    "repo_type": "dataset"
  }
}
```

### Read a file from a repository
Call this tool with:
```json
{
  "operation": "read_file",
  "args": {
    "repo_id": "job-results",
    "path_in_repo": "jobs/job-abc123/script.py",
    "repo_type": "dataset"
  }
}
```

## Repository Types

- **dataset** (default) - For storing data, results, logs, scripts
- **model** - For ML models and related artifacts
- **space** - For Spaces and applications

## Tips

- **Content-based**: Pass file content directly as strings or bytes, not file paths
- **Repo ID format**: Use just the repo name (e.g., "my-dataset"). Username is automatically inferred from HF_TOKEN
- **Automatic repo creation**: Set `create_if_missing: true` to auto-create repos (requires user approval)
- **Organization**: Use path_in_repo to organize files (e.g., "jobs/job-123/script.py")
- **After jobs**: Upload job scripts and logs after compute jobs complete for reproducibility
"""
        return {"formatted": usage_text, "totalResults": 1, "resultsShared": 1}

    def _show_operation_help(self, operation: str) -> ToolResult:
        """Show help for a specific operation."""
        help_text = f"Help for operation: {operation}\n\nCall with appropriate arguments. Use the main help for examples."
        return {"formatted": help_text, "totalResults": 1, "resultsShared": 1}

    async def _upload_file(self, args: Dict[str, Any]) -> ToolResult:
        """Upload file content to a Hub repository."""
        # Validate required arguments
        file_content = args.get("file_content")
        path_in_repo = args.get("path_in_repo")
        repo_id = args.get("repo_id")

        if not file_content:
            return {
                "formatted": "file_content is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        if not path_in_repo:
            return {
                "formatted": "path_in_repo is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        if not repo_id:
            return {
                "formatted": "repo_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        repo_type = args.get("repo_type", "dataset")
        create_if_missing = args.get("create_if_missing", False)

        # Check if repo exists
        try:
            repo_exists = await _async_call(
                self.api.repo_exists, repo_id=repo_id, repo_type=repo_type
            )

            # Create repo if needed
            if not repo_exists and create_if_missing:
                create_args = {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "private": True,
                }
                # Pass through space_sdk if provided (required for spaces)
                if "space_sdk" in args:
                    create_args["space_sdk"] = args["space_sdk"]
                await self._create_repo(create_args)
            elif not repo_exists:
                return {
                    "formatted": f"Repository {repo_id} does not exist. Set create_if_missing: true to create it.",
                    "totalResults": 0,
                    "resultsShared": 0,
                    "isError": True,
                }

        except Exception as e:
            return {
                "formatted": f"Failed to check repository: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        # Convert content to bytes
        file_bytes = _content_to_bytes(file_content)

        # Upload file
        try:
            await _async_call(
                self.api.upload_file,
                path_or_fileobj=file_bytes,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=args.get("commit_message", f"Upload {path_in_repo}"),
            )

            repo_url = _build_repo_url(repo_id, repo_type)
            file_url = f"{repo_url}/blob/main/{path_in_repo}"

            response = f"""✓ File uploaded successfully!

**Repository:** {repo_id}
**File:** {path_in_repo}
**View at:** {file_url}
**Browse repo:** {repo_url}"""

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            return {
                "formatted": f"Failed to upload file: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

    async def _create_repo(self, args: Dict[str, Any]) -> ToolResult:
        """Create a new Hub repository."""
        repo_id = args.get("repo_id")

        if not repo_id:
            return {
                "formatted": "repo_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        repo_type = args.get("repo_type", "dataset")
        private = True  # Always create private repos
        space_sdk = args.get("space_sdk")  # Required if repo_type is "space"

        try:
            # Check if repo already exists
            repo_exists = await _async_call(
                self.api.repo_exists, repo_id=repo_id, repo_type=repo_type
            )

            if repo_exists:
                repo_url = _build_repo_url(repo_id, repo_type)
                return {
                    "formatted": f"Repository {repo_id} already exists.\n**View at:** {repo_url}",
                    "totalResults": 1,
                    "resultsShared": 1,
                }

            # Validate space_sdk for spaces
            if repo_type == "space" and not space_sdk:
                return {
                    "formatted": "space_sdk is required when creating a space. Valid values: gradio, streamlit, static, docker",
                    "totalResults": 0,
                    "resultsShared": 0,
                    "isError": True,
                }

            # Create repository
            create_kwargs = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "exist_ok": True,
            }
            # Add space_sdk only for spaces
            if repo_type == "space" and space_sdk:
                create_kwargs["space_sdk"] = space_sdk

            repo_url = await _async_call(self.api.create_repo, **create_kwargs)

            response = f"""✓ Repository created successfully!

**Repository:** {repo_id}
**Type:** {repo_type}
**Private:** Yes
**View at:** {repo_url}"""

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            return {
                "formatted": f"Failed to create repository: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

    async def _check_repo(self, args: Dict[str, Any]) -> ToolResult:
        """Check if a Hub repository exists."""
        repo_id = args.get("repo_id")

        if not repo_id:
            return {
                "formatted": "repo_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        repo_type = args.get("repo_type", "dataset")

        try:
            repo_exists = await _async_call(
                self.api.repo_exists, repo_id=repo_id, repo_type=repo_type
            )

            if repo_exists:
                repo_url = _build_repo_url(repo_id, repo_type)
                response = f"""✓ Repository exists!

**Repository:** {repo_id}
**Type:** {repo_type}
**View at:** {repo_url}"""
            else:
                response = f"""Repository does not exist: {repo_id}

To create it, call this tool with:
```json
{{
  "operation": "create_repo",
  "args": {{
    "repo_id": "{repo_id}",
    "repo_type": "{repo_type}"
  }}
}}
```"""

            return {
                "formatted": response,
                "totalResults": 1 if repo_exists else 0,
                "resultsShared": 1 if repo_exists else 0,
            }

        except Exception as e:
            return {
                "formatted": f"Failed to check repository: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

    async def _list_files(self, args: Dict[str, Any]) -> ToolResult:
        """List all files in a Hub repository."""
        repo_id = args.get("repo_id")

        if not repo_id:
            return {
                "formatted": "repo_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        repo_type = args.get("repo_type", "dataset")

        try:
            # List all files in the repository
            files = await _async_call(
                self.api.list_repo_files, repo_id=repo_id, repo_type=repo_type
            )

            if not files:
                return {
                    "formatted": f"No files found in repository: {repo_id}",
                    "totalResults": 0,
                    "resultsShared": 0,
                }

            # Format file list
            file_list = "\n".join(f"- {f}" for f in sorted(files))
            repo_url = _build_repo_url(repo_id, repo_type)

            response = f"""✓ Files in repository: {repo_id}

**Total files:** {len(files)}
**Repository URL:** {repo_url}

**Files:**
{file_list}"""

            return {
                "formatted": response,
                "totalResults": len(files),
                "resultsShared": len(files),
            }

        except Exception as e:
            return {
                "formatted": f"Failed to list files: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

    async def _read_file(self, args: Dict[str, Any]) -> ToolResult:
        """Read content of a specific file from a Hub repository."""
        repo_id = args.get("repo_id")
        path_in_repo = args.get("path_in_repo")

        if not repo_id:
            return {
                "formatted": "repo_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        if not path_in_repo:
            return {
                "formatted": "path_in_repo is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        repo_type = args.get("repo_type", "dataset")

        try:
            # Download file to cache and read it
            file_path = await _async_call(
                hf_hub_download,
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type=repo_type,
                token=self.api.token,
            )

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            repo_url = _build_repo_url(repo_id, repo_type)
            file_url = f"{repo_url}/blob/main/{path_in_repo}"

            response = f"""✓ File read successfully!

**Repository:** {repo_id}
**File:** {path_in_repo}
**Size:** {len(content)} characters
**View at:** {file_url}

**Content:**
```
{content}
```"""

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except UnicodeDecodeError:
            # If file is binary, return size info instead
            try:
                with open(file_path, "rb") as f:
                    binary_content = f.read()

                return {
                    "formatted": f"File is binary ({len(binary_content)} bytes). Cannot display as text.",
                    "totalResults": 1,
                    "resultsShared": 1,
                }
            except Exception as e:
                return {
                    "formatted": f"Failed to read binary file: {str(e)}",
                    "totalResults": 0,
                    "resultsShared": 0,
                    "isError": True,
                }
        except Exception as e:
            return {
                "formatted": f"Failed to read file: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }


# Tool specification for agent registration
PRIVATE_HF_REPO_TOOL_SPEC = {
    "name": "hf_private_repos",
    "description": (
        "Manage private HF repositories - create, upload, read, list files in models/datasets/spaces. "
        "⚠️ PRIMARY USE: Store job outputs persistently (job storage is EPHEMERAL - everything deleted after completion). "
        "**Use when:** (1) Job completes and need to store logs/scripts/results, (2) Creating repos for training outputs, "
        "(3) Reading back stored files, (4) Managing Space files, (5) Organizing job artifacts by path. "
        "**Pattern:** hf_jobs (ephemeral) → hf_private_repos upload_file (persistent) → can read_file later. "
        "ALWAYS pass file_content as string/bytes (✓), never file paths (✗) - this is content-based, no filesystem access. "
        "**Operations:** create_repo (new private repo), upload_file (store content), read_file (retrieve content), list_files (browse), check_repo (verify exists). "
        "**Critical for reliability:** Jobs lose all files after completion - use this tool to preserve important outputs. "
        "Repositories created are ALWAYS private by default (good for sensitive training data/models). "
        "For Spaces: must provide space_sdk ('gradio', 'streamlit', 'static', 'docker') when creating. "
        "**Then:** After uploading, provide user with repository URL for viewing/sharing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "upload_file",
                    "create_repo",
                    "check_repo",
                    "list_files",
                    "read_file",
                ],
                "description": (
                    "Operation to execute. Valid values: [upload_file, create_repo, check_repo, list_files, read_file]"
                ),
            },
            "args": {
                "type": "object",
                "description": (
                    "Operation-specific arguments as a JSON object. "
                    "Write ops: file_content (string/bytes), path_in_repo (string), repo_id (string), "
                    "repo_type (dataset/model/space), create_if_missing (boolean), commit_message (string), "
                    "space_sdk (gradio/streamlit/static/docker - required when repo_type=space). "
                    "Read ops: repo_id (string), path_in_repo (for read_file), repo_type (optional)."
                ),
                "additionalProperties": True,
            },
        },
    },
}


async def private_hf_repo_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router."""
    try:
        tool = PrivateHfRepoTool()
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error executing Private HF Repo tool: {str(e)}", False
