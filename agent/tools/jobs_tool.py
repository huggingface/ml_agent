"""
Hugging Face Jobs Tool - Using huggingface-hub library

Refactored to use official huggingface-hub library instead of custom HTTP client
"""

import asyncio
import base64
from typing import Any, Dict, Literal, Optional

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from agent.tools.types import ToolResult
from agent.tools.utilities import (
    format_job_details,
    format_jobs_table,
    format_scheduled_job_details,
    format_scheduled_jobs_table,
)

# Hardware flavors
CPU_FLAVORS = ["cpu-basic", "cpu-upgrade", "cpu-performance", "cpu-xl"]
GPU_FLAVORS = [
    "sprx8",
    "zero-a10g",
    "t4-small",
    "t4-medium",
    "l4x1",
    "l4x4",
    "l40sx1",
    "l40sx4",
    "l40sx8",
    "a10g-small",
    "a10g-large",
    "a10g-largex2",
    "a10g-largex4",
    "a100-large",
    "h100",
    "h100x8",
]
SPECIALIZED_FLAVORS = ["inf2x6"]
ALL_FLAVORS = CPU_FLAVORS + GPU_FLAVORS + SPECIALIZED_FLAVORS

# Operation names
OperationType = Literal[
    "run",
    "uv",
    "ps",
    "logs",
    "inspect",
    "cancel",
    "scheduled run",
    "scheduled uv",
    "scheduled ps",
    "scheduled inspect",
    "scheduled delete",
    "scheduled suspend",
    "scheduled resume",
]

# Constants
DEFAULT_LOG_WAIT_SECONDS = 10
UV_DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"


def _build_uv_command(
    script: str,
    with_deps: list[str] | None = None,
    python: str | None = None,
    script_args: list[str] | None = None,
) -> list[str]:
    """Build UV run command"""
    parts = ["uv", "run"]

    if with_deps:
        for dep in with_deps:
            parts.extend(["--with", dep])

    if python:
        parts.extend(["-p", python])

    parts.append(script)

    if script_args:
        parts.extend(script_args)

    return parts


def _wrap_inline_script(
    script: str,
    with_deps: list[str] | None = None,
    python: str | None = None,
    script_args: list[str] | None = None,
) -> str:
    """Wrap inline script with base64 encoding to avoid file creation"""
    encoded = base64.b64encode(script.encode("utf-8")).decode("utf-8")
    # Build the uv command with stdin (-)
    uv_command = _build_uv_command("-", with_deps, python, script_args)
    # Join command parts with proper spacing
    uv_command_str = " ".join(uv_command)
    return f'echo "{encoded}" | base64 -d | {uv_command_str}'


def _resolve_uv_command(
    script: str,
    with_deps: list[str] | None = None,
    python: str | None = None,
    script_args: list[str] | None = None,
) -> list[str]:
    """Resolve UV command based on script source (URL, inline, or file path)"""
    # If URL, use directly
    if script.startswith("http://") or script.startswith("https://"):
        return _build_uv_command(script, with_deps, python, script_args)

    # If contains newline, treat as inline script
    if "\n" in script:
        wrapped = _wrap_inline_script(script, with_deps, python, script_args)
        return ["/bin/sh", "-lc", wrapped]

    # Otherwise, treat as file path
    return _build_uv_command(script, with_deps, python, script_args)


async def _async_call(func, *args, **kwargs):
    """Wrap synchronous HfApi calls for async context"""
    return await asyncio.to_thread(func, *args, **kwargs)


def _job_info_to_dict(job_info) -> Dict[str, Any]:
    """Convert JobInfo object to dictionary for formatting functions"""
    return {
        "id": job_info.id,
        "status": {"stage": job_info.status.stage, "message": job_info.status.message},
        "command": job_info.command,
        "createdAt": job_info.created_at.isoformat(),
        "dockerImage": job_info.docker_image,
        "spaceId": job_info.space_id,
        "flavor": job_info.flavor,
        "owner": {"name": job_info.owner.name},
    }


def _scheduled_job_info_to_dict(scheduled_job_info) -> Dict[str, Any]:
    """Convert ScheduledJobInfo object to dictionary for formatting functions"""
    job_spec = scheduled_job_info.job_spec

    # Extract last run and next run from status
    last_run = None
    next_run = None
    if scheduled_job_info.status:
        if scheduled_job_info.status.last_job:
            last_run = scheduled_job_info.status.last_job.created_at
            if last_run:
                last_run = (
                    last_run.isoformat()
                    if hasattr(last_run, "isoformat")
                    else str(last_run)
                )
        if scheduled_job_info.status.next_job_run_at:
            next_run = scheduled_job_info.status.next_job_run_at
            next_run = (
                next_run.isoformat()
                if hasattr(next_run, "isoformat")
                else str(next_run)
            )

    return {
        "id": scheduled_job_info.id,
        "schedule": scheduled_job_info.schedule,
        "suspend": scheduled_job_info.suspend,
        "lastRun": last_run,
        "nextRun": next_run,
        "jobSpec": {
            "dockerImage": job_spec.docker_image,
            "spaceId": job_spec.space_id,
            "command": job_spec.command or [],
            "flavor": job_spec.flavor or "cpu-basic",
        },
    }


class HfJobsTool:
    """Tool for managing Hugging Face compute jobs using huggingface-hub library"""

    def __init__(self, hf_token: Optional[str] = None, namespace: Optional[str] = None):
        self.api = HfApi(token=hf_token)
        self.namespace = namespace

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the specified operation"""
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
            if operation == "run":
                return await self._run_job(args)
            elif operation == "uv":
                return await self._run_uv_job(args)
            elif operation == "ps":
                return await self._list_jobs(args)
            elif operation == "logs":
                return await self._get_logs(args)
            elif operation == "inspect":
                return await self._inspect_job(args)
            elif operation == "cancel":
                return await self._cancel_job(args)
            elif operation == "scheduled run":
                return await self._scheduled_run(args)
            elif operation == "scheduled uv":
                return await self._scheduled_uv(args)
            elif operation == "scheduled ps":
                return await self._list_scheduled_jobs(args)
            elif operation == "scheduled inspect":
                return await self._inspect_scheduled_job(args)
            elif operation == "scheduled delete":
                return await self._delete_scheduled_job(args)
            elif operation == "scheduled suspend":
                return await self._suspend_scheduled_job(args)
            elif operation == "scheduled resume":
                return await self._resume_scheduled_job(args)
            else:
                return {
                    "formatted": f'Unknown operation: "{operation}"\n\n'
                    "Available operations:\n"
                    "- run, uv, ps, logs, inspect, cancel\n"
                    "- scheduled run, scheduled uv, scheduled ps, scheduled inspect, "
                    "scheduled delete, scheduled suspend, scheduled resume\n\n"
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
        """Show usage instructions when tool is called with no arguments"""
        cpu_flavors_list = ", ".join(CPU_FLAVORS)
        gpu_flavors_list = ", ".join(GPU_FLAVORS)
        specialized_flavors_list = ", ".join(SPECIALIZED_FLAVORS)

        hardware_section = f"**CPU:** {cpu_flavors_list}\n"
        if GPU_FLAVORS:
            hardware_section += f"**GPU:** {gpu_flavors_list}\n"
        if SPECIALIZED_FLAVORS:
            hardware_section += f"**Specialized:** {specialized_flavors_list}"

        usage_text = f"""# HuggingFace Jobs API

Manage compute jobs on Hugging Face infrastructure.

## Available Commands

### Job Management
- **run** - Run a job with a Docker image
- **uv** - Run a Python script with UV (inline dependencies)
- **ps** - List jobs
- **logs** - Fetch job logs
- **inspect** - Get detailed job information
- **cancel** - Cancel a running job

### Scheduled Jobs
- **scheduled run** - Create a scheduled job
- **scheduled uv** - Create a scheduled UV job
- **scheduled ps** - List scheduled jobs
- **scheduled inspect** - Get scheduled job details
- **scheduled delete** - Delete a scheduled job
- **scheduled suspend** - Pause a scheduled job
- **scheduled resume** - Resume a suspended job

## Examples

### Run a simple job
Call this tool with:
```json
{{
  "operation": "run",
  "args": {{
    "image": "python:3.12",
    "command": ["python", "-c", "print('Hello from HF Jobs!')"],
    "flavor": "cpu-basic"
  }}
}}
```

### Run a Python script with UV
Call this tool with:
```json
{{
  "operation": "uv",
  "args": {{
    "script": "import random\\nprint(42 + random.randint(1, 5))"
  }}
}}
```

## Hardware Flavors

{hardware_section}

## Command Format Guidelines

**Array format (default):**
- Recommended for every command—JSON keeps arguments intact (URLs with `&`, spaces, etc.)
- Use `["/bin/sh", "-lc", "..."]` when you need shell operators like `&&`, `|`, or redirections
- Works with any language: Python, bash, node, npm, uv, etc.

**String format (simple cases only):**
- Still accepted for backwards compatibility, parsed with POSIX shell semantics
- Rejects shell operators and can mis-handle characters such as `&`; switch to arrays when things turn complex
- `$HF_TOKEN` stays literal—forward it via `secrets: {{ "HF_TOKEN": "$HF_TOKEN" }}`

### Show command-specific help
Call this tool with:
```json
{{"operation": "<operation>", "args": {{"help": true}}}}
```

## Tips

- Jobs default to non-detached mode (tail logs for up to {DEFAULT_LOG_WAIT_SECONDS}s or until completion). Set `detach: true` to return immediately.
- Prefer array commands to avoid shell parsing surprises
- To access private Hub assets, include `secrets: {{ "HF_TOKEN": "$HF_TOKEN" }}` to inject your auth token.
"""
        return {"formatted": usage_text, "totalResults": 1, "resultsShared": 1}

    def _show_operation_help(self, operation: str) -> ToolResult:
        """Show help for a specific operation"""
        help_text = f"Help for operation: {operation}\n\nCall with appropriate arguments. Use the main help for examples."
        return {"formatted": help_text, "totalResults": 1, "resultsShared": 1}

    async def _run_job(self, args: Dict[str, Any]) -> ToolResult:
        """Run a job using HfApi.run_job()"""
        try:
            job = await _async_call(
                self.api.run_job,
                image=args.get("image", "python:3.12"),
                command=args.get("command"),
                env=args.get("env"),
                secrets=args.get("secrets"),
                flavor=args.get("flavor", "cpu-basic"),
                timeout=args.get("timeout", "30m"),
                namespace=args.get("namespace") or self.namespace,
            )

            # If detached, return immediately
            if args.get("detach", False):
                response = f"""Job started successfully!

**Job ID:** {job.id}
**Status:** {job.status.stage}
**View at:** {job.url}

To check logs, call this tool with `{{"operation": "logs", "args": {{"job_id": "{job.id}"}}}}`
To inspect, call this tool with `{{"operation": "inspect", "args": {{"job_id": "{job.id}"}}}}`"""
                return {"formatted": response, "totalResults": 1, "resultsShared": 1}

            # Not detached - return job info
            response = f"""Job started: {job.id}

**Status:** {job.status.stage}
**View logs at:** {job.url}

Note: Logs are being collected. Check the job page for real-time logs.
"""
            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            raise Exception(f"Failed to run job: {str(e)}")

    async def _run_uv_job(self, args: Dict[str, Any]) -> ToolResult:
        """Run UV job with inline script support (no local files needed)"""
        try:
            script = args.get("script")
            if not script:
                raise ValueError("script is required")

            # Resolve the command based on script type (URL, inline, or file)
            command = _resolve_uv_command(
                script=script,
                with_deps=args.get("with_deps") or args.get("dependencies"),
                python=args.get("python"),
                script_args=args.get("script_args"),
            )

            # Use run_job with UV image instead of run_uv_job
            job = await _async_call(
                self.api.run_job,
                image=UV_DEFAULT_IMAGE,
                command=command,
                env=args.get("env"),
                secrets=args.get("secrets"),
                flavor=args.get("flavor", "cpu-basic"),
                timeout=args.get("timeout", "30m"),
                namespace=args.get("namespace") or self.namespace,
            )

            response = f"""UV Job started: {job.id}

**Status:** {job.status.stage}
**View at:** {job.url}

To check logs, call this tool with `{{"operation": "logs", "args": {{"job_id": "{job.id}"}}}}`
"""
            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            raise Exception(f"Failed to run UV job: {str(e)}")

    async def _list_jobs(self, args: Dict[str, Any]) -> ToolResult:
        """List jobs using HfApi.list_jobs()"""
        jobs_list = await _async_call(
            self.api.list_jobs, namespace=args.get("namespace") or self.namespace
        )

        # Filter jobs
        if not args.get("all", False):
            jobs_list = [j for j in jobs_list if j.status.stage == "RUNNING"]

        if args.get("status"):
            status_filter = args["status"].upper()
            jobs_list = [j for j in jobs_list if status_filter in j.status.stage]

        # Convert JobInfo objects to dicts for formatting
        jobs_dicts = [_job_info_to_dict(j) for j in jobs_list]

        table = format_jobs_table(jobs_dicts)

        if len(jobs_list) == 0:
            if args.get("all", False):
                return {
                    "formatted": "No jobs found.",
                    "totalResults": 0,
                    "resultsShared": 0,
                }
            return {
                "formatted": 'No running jobs found. Use `{"args": {"all": true}}` to show all jobs.',
                "totalResults": 0,
                "resultsShared": 0,
            }

        response = f"**Jobs ({len(jobs_list)} total):**\n\n{table}"
        return {
            "formatted": response,
            "totalResults": len(jobs_list),
            "resultsShared": len(jobs_list),
        }

    async def _get_logs(self, args: Dict[str, Any]) -> ToolResult:
        """Fetch logs using HfApi.fetch_job_logs()"""
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }

        try:
            # Fetch logs (returns generator, convert to list)
            logs_gen = self.api.fetch_job_logs(
                job_id=job_id, namespace=args.get("namespace") or self.namespace
            )
            logs = await _async_call(list, logs_gen)

            if not logs:
                return {
                    "formatted": f"No logs available for job {job_id}",
                    "totalResults": 0,
                    "resultsShared": 0,
                }

            log_text = "\n".join(logs)
            return {
                "formatted": f"**Logs for {job_id}:**\n\n```\n{log_text}\n```",
                "totalResults": 1,
                "resultsShared": 1,
            }

        except Exception as e:
            return {
                "formatted": f"Failed to fetch logs: {str(e)}",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }

    async def _inspect_job(self, args: Dict[str, Any]) -> ToolResult:
        """Inspect job using HfApi.inspect_job()"""
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        job_ids = job_id if isinstance(job_id, list) else [job_id]

        jobs = []
        for jid in job_ids:
            try:
                job = await _async_call(
                    self.api.inspect_job,
                    job_id=jid,
                    namespace=args.get("namespace") or self.namespace,
                )
                jobs.append(_job_info_to_dict(job))
            except Exception as e:
                raise Exception(f"Failed to inspect job {jid}: {str(e)}")

        formatted_details = format_job_details(jobs)
        response = f"**Job Details** ({len(jobs)} job{'s' if len(jobs) > 1 else ''}):\n\n{formatted_details}"

        return {
            "formatted": response,
            "totalResults": len(jobs),
            "resultsShared": len(jobs),
        }

    async def _cancel_job(self, args: Dict[str, Any]) -> ToolResult:
        """Cancel job using HfApi.cancel_job()"""
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        await _async_call(
            self.api.cancel_job,
            job_id=job_id,
            namespace=args.get("namespace") or self.namespace,
        )

        response = f"""✓ Job {job_id} has been cancelled.

To verify, call this tool with `{{"operation": "inspect", "args": {{"job_id": "{job_id}"}}}}`"""

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    async def _scheduled_run(self, args: Dict[str, Any]) -> ToolResult:
        """Create scheduled job using HfApi.create_scheduled_job()"""
        try:
            scheduled_job = await _async_call(
                self.api.create_scheduled_job,
                image=args.get("image", "python:3.12"),
                command=args.get("command"),
                schedule=args.get("schedule"),
                env=args.get("env"),
                secrets=args.get("secrets"),
                flavor=args.get("flavor", "cpu-basic"),
                timeout=args.get("timeout", "30m"),
                namespace=args.get("namespace") or self.namespace,
            )

            scheduled_dict = _scheduled_job_info_to_dict(scheduled_job)

            response = f"""✓ Scheduled job created successfully!

**Scheduled Job ID:** {scheduled_dict["id"]}
**Schedule:** {scheduled_dict["schedule"]}
**Suspended:** {"Yes" if scheduled_dict.get("suspend") else "No"}
**Next Run:** {scheduled_dict.get("nextRun", "N/A")}

To inspect, call this tool with `{{"operation": "scheduled inspect", "args": {{"scheduled_job_id": "{scheduled_dict["id"]}"}}}}`
To list all, call this tool with `{{"operation": "scheduled ps"}}`"""

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            raise Exception(f"Failed to create scheduled job: {str(e)}")

    async def _scheduled_uv(self, args: Dict[str, Any]) -> ToolResult:
        """Create scheduled UV job with inline script support"""
        try:
            script = args.get("script")
            if not script:
                raise ValueError("script is required")

            schedule = args.get("schedule")
            if not schedule:
                raise ValueError("schedule is required")

            # Resolve the command based on script type
            command = _resolve_uv_command(
                script=script,
                with_deps=args.get("with_deps") or args.get("dependencies"),
                python=args.get("python"),
                script_args=args.get("script_args"),
            )

            # Use create_scheduled_job with UV image
            scheduled_job = await _async_call(
                self.api.create_scheduled_job,
                image=UV_DEFAULT_IMAGE,
                command=command,
                schedule=schedule,
                env=args.get("env"),
                secrets=args.get("secrets"),
                flavor=args.get("flavor", "cpu-basic"),
                timeout=args.get("timeout", "30m"),
                namespace=args.get("namespace") or self.namespace,
            )

            scheduled_dict = _scheduled_job_info_to_dict(scheduled_job)

            response = f"""✓ Scheduled UV job created successfully!

**Scheduled Job ID:** {scheduled_dict["id"]}
**Schedule:** {scheduled_dict["schedule"]}
**Suspended:** {"Yes" if scheduled_dict.get("suspend") else "No"}
**Next Run:** {scheduled_dict.get("nextRun", "N/A")}

To inspect, call this tool with `{{"operation": "scheduled inspect", "args": {{"scheduled_job_id": "{scheduled_dict["id"]}"}}}}`"""

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            raise Exception(f"Failed to create scheduled UV job: {str(e)}")

    async def _list_scheduled_jobs(self, args: Dict[str, Any]) -> ToolResult:
        """List scheduled jobs using HfApi.list_scheduled_jobs()"""
        scheduled_jobs_list = await _async_call(
            self.api.list_scheduled_jobs,
            namespace=args.get("namespace") or self.namespace,
        )

        # Filter jobs - default: hide suspended jobs unless --all is specified
        if not args.get("all", False):
            scheduled_jobs_list = [j for j in scheduled_jobs_list if not j.suspend]

        # Convert to dicts for formatting
        scheduled_dicts = [_scheduled_job_info_to_dict(j) for j in scheduled_jobs_list]

        table = format_scheduled_jobs_table(scheduled_dicts)

        if len(scheduled_jobs_list) == 0:
            if args.get("all", False):
                return {
                    "formatted": "No scheduled jobs found.",
                    "totalResults": 0,
                    "resultsShared": 0,
                }
            return {
                "formatted": 'No active scheduled jobs found. Use `{"args": {"all": true}}` to show suspended jobs.',
                "totalResults": 0,
                "resultsShared": 0,
            }

        response = f"**Scheduled Jobs ({len(scheduled_jobs_list)} total):**\n\n{table}"
        return {
            "formatted": response,
            "totalResults": len(scheduled_jobs_list),
            "resultsShared": len(scheduled_jobs_list),
        }

    async def _inspect_scheduled_job(self, args: Dict[str, Any]) -> ToolResult:
        """Inspect scheduled job using HfApi.inspect_scheduled_job()"""
        scheduled_job_id = args.get("scheduled_job_id")
        if not scheduled_job_id:
            return {
                "formatted": "scheduled_job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        scheduled_job = await _async_call(
            self.api.inspect_scheduled_job,
            scheduled_job_id=scheduled_job_id,
            namespace=args.get("namespace") or self.namespace,
        )

        scheduled_dict = _scheduled_job_info_to_dict(scheduled_job)
        formatted_details = format_scheduled_job_details(scheduled_dict)

        return {
            "formatted": f"**Scheduled Job Details:**\n\n{formatted_details}",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _delete_scheduled_job(self, args: Dict[str, Any]) -> ToolResult:
        """Delete scheduled job using HfApi.delete_scheduled_job()"""
        scheduled_job_id = args.get("scheduled_job_id")
        if not scheduled_job_id:
            return {
                "formatted": "scheduled_job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        await _async_call(
            self.api.delete_scheduled_job,
            scheduled_job_id=scheduled_job_id,
            namespace=args.get("namespace") or self.namespace,
        )

        return {
            "formatted": f"✓ Scheduled job {scheduled_job_id} has been deleted.",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _suspend_scheduled_job(self, args: Dict[str, Any]) -> ToolResult:
        """Suspend scheduled job using HfApi.suspend_scheduled_job()"""
        scheduled_job_id = args.get("scheduled_job_id")
        if not scheduled_job_id:
            return {
                "formatted": "scheduled_job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        await _async_call(
            self.api.suspend_scheduled_job,
            scheduled_job_id=scheduled_job_id,
            namespace=args.get("namespace") or self.namespace,
        )

        response = f"""✓ Scheduled job {scheduled_job_id} has been suspended.

To resume, call this tool with `{{"operation": "scheduled resume", "args": {{"scheduled_job_id": "{scheduled_job_id}"}}}}`"""

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    async def _resume_scheduled_job(self, args: Dict[str, Any]) -> ToolResult:
        """Resume scheduled job using HfApi.resume_scheduled_job()"""
        scheduled_job_id = args.get("scheduled_job_id")
        if not scheduled_job_id:
            return {
                "formatted": "scheduled_job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        await _async_call(
            self.api.resume_scheduled_job,
            scheduled_job_id=scheduled_job_id,
            namespace=args.get("namespace") or self.namespace,
        )

        response = f"""✓ Scheduled job {scheduled_job_id} has been resumed.

To inspect, call this tool with `{{"operation": "scheduled inspect", "args": {{"scheduled_job_id": "{scheduled_job_id}"}}}}`"""

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}


# Tool specification for agent registration
HF_JOBS_TOOL_SPEC = {
    "name": "hf_jobs",
    "description": (
        "Manage Hugging Face CPU/GPU compute jobs. Run commands in Docker containers, "
        "execute Python scripts with UV. List, schedule and monitor jobs/logs. "
        "Call this tool with no operation for full usage instructions and examples."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "run",
                    "uv",
                    "ps",
                    "logs",
                    "inspect",
                    "cancel",
                    "scheduled run",
                    "scheduled uv",
                    "scheduled ps",
                    "scheduled inspect",
                    "scheduled delete",
                    "scheduled suspend",
                    "scheduled resume",
                ],
                "description": (
                    "Operation to execute. Valid values: [run, uv, ps, logs, inspect, cancel, "
                    "scheduled run, scheduled uv, scheduled ps, scheduled inspect, scheduled delete, "
                    "scheduled suspend, scheduled resume]"
                ),
            },
            "args": {
                "type": "object",
                "description": "Operation-specific arguments as a JSON object",
                "additionalProperties": True,
            },
        },
    },
}


async def hf_jobs_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        tool = HfJobsTool()
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error executing HF Jobs tool: {str(e)}", False
