"""
Hugging Face Jobs Tool - Using huggingface-hub library

Refactored to use official huggingface-hub library instead of custom HTTP client
"""

import asyncio
import base64
import http.client
import os
import re
from typing import Any, Dict, Literal, Optional, Callable, Awaitable

import logging

import httpx
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from agent.core.hf_access import JobsAccessError, is_billing_error, resolve_jobs_namespace
from agent.core.session import Event
from agent.tools.trackio_seed import ensure_trackio_dashboard
from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)
from agent.tools.utilities import (
    format_job_details,
    format_jobs_table,
    format_scheduled_job_details,
    format_scheduled_jobs_table,
)

# Hardware flavors
CPU_FLAVORS = ["cpu-basic", "cpu-upgrade"]
GPU_FLAVORS = [
    "t4-small",
    "t4-medium",
    "a10g-small",
    "a10g-large",
    "a10g-largex2",
    "a10g-largex4",
    "a100-large",
    "a100x4",
    "a100x8",
    "l4x1",
    "l4x4",
    "l40sx1",
    "l40sx4",
    "l40sx8",
]

# Detailed specs for display (vCPU/RAM/GPU VRAM)
CPU_FLAVORS_DESC = "cpu-basic(2vCPU/16GB), cpu-upgrade(8vCPU/32GB)"
GPU_FLAVORS_DESC = (
    "t4-small(4vCPU/15GB/GPU 16GB), t4-medium(8vCPU/30GB/GPU 16GB), "
    "a10g-small(4vCPU/15GB/GPU 24GB), a10g-large(12vCPU/46GB/GPU 24GB), "
    "a10g-largex2(24vCPU/92GB/GPU 48GB), a10g-largex4(48vCPU/184GB/GPU 96GB), "
    "a100-large(12vCPU/142GB/GPU 80GB), a100x4(48vCPU/568GB/GPU 320GB), a100x8(96vCPU/1136GB/GPU 640GB), "
    "l4x1(8vCPU/30GB/GPU 24GB), l4x4(48vCPU/186GB/GPU 96GB), "
    "l40sx1(8vCPU/62GB/GPU 48GB), l40sx4(48vCPU/382GB/GPU 192GB), l40sx8(192vCPU/1534GB/GPU 384GB)"
)
SPECIALIZED_FLAVORS = ["inf2x6"]
ALL_FLAVORS = CPU_FLAVORS + GPU_FLAVORS + SPECIALIZED_FLAVORS

# Operation names
OperationType = Literal[
    "run",
    "ps",
    "logs",
    "inspect",
    "cancel",
    "scheduled run",
    "scheduled ps",
    "scheduled inspect",
    "scheduled delete",
    "scheduled suspend",
    "scheduled resume",
]

# Constants
UV_DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"


def _filter_uv_install_output(logs: list[str]) -> list[str]:
    """
    Filter out UV package installation output from logs.

    Replaces installation details with "[installs truncated]" and keeps
    the "Installed X packages in Y ms/s" summary line.

    Args:
        logs: List of log lines

    Returns:
        Filtered list of log lines
    """
    if not logs:
        return logs

    # Regex pattern to match: "Installed X packages in Y ms" or "Installed X package in Y s"
    install_pattern = re.compile(
        r"^Installed\s+\d+\s+packages?\s+in\s+\d+(?:\.\d+)?\s*(?:ms|s)$"
    )

    # Find the index of the "Installed X packages" line
    install_line_idx = None
    for idx, line in enumerate(logs):
        if install_pattern.match(line.strip()):
            install_line_idx = idx
            break

    # If pattern found, replace installation details with truncation message
    if install_line_idx is not None and install_line_idx > 0:
        # Keep logs from the "Installed X packages" line onward
        # Add truncation message before the "Installed" line
        return ["[installs truncated]"] + logs[install_line_idx:]

    # If pattern not found, return original logs
    return logs


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)


_DEFAULT_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "warning",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "UV_NO_PROGRESS": "1",
}


def _add_default_env(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Inject default env vars for clean, agent-friendly output."""
    result = dict(_DEFAULT_ENV)
    result.update(params or {})  # user-provided values override defaults
    return result


def _add_environment_variables(
    params: Dict[str, Any] | None, user_token: str | None = None
) -> Dict[str, Any]:
    token = user_token or ""

    # Start with user-provided env vars, then force-set token last
    result = dict(params or {})

    # If the caller passed HF_TOKEN="$HF_TOKEN", ignore it.
    if result.get("HF_TOKEN", "").strip().startswith("$"):
        result.pop("HF_TOKEN", None)

    # Set both names to be safe (different libs check different vars)
    if token:
        result["HF_TOKEN"] = token
        result["HUGGINGFACE_HUB_TOKEN"] = token

    return result


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

    # add defaults
    # parts.extend(["--push_to_hub"])
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


def _ensure_hf_transfer_dependency(deps: list[str] | None) -> list[str]:
    """Ensure hf-transfer is included in the dependencies list"""

    if isinstance(deps, list):
        deps_copy = deps.copy()  # Don't modify the original
        if "hf-transfer" not in deps_copy:
            deps_copy.append("hf-transfer")
        return deps_copy

    return ["hf-transfer"]


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
        "hardware_flavor": job_info.flavor,
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
            "hardware_flavor": job_spec.flavor or "cpu-basic",
        },
    }


class HfJobsTool:
    """Tool for managing Hugging Face compute jobs using huggingface-hub library"""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        namespace: Optional[str] = None,
        jobs_access: Any = None,
        log_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        session: Any = None,
        tool_call_id: Optional[str] = None,
    ):
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)
        self.namespace = namespace
        self.jobs_access = jobs_access
        self.log_callback = log_callback
        self.session = session
        self.tool_call_id = tool_call_id

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the specified operation"""
        operation = params.get("operation")

        args = params

        # If no operation provided, return error
        if not operation:
            return {
                "formatted": "Error: 'operation' parameter is required. See tool description for available operations and usage examples.",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        # Normalize operation name
        operation = operation.lower()

        try:
            # Route to appropriate handler
            if operation == "run":
                return await self._run_job(args)
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
                    "- run, ps, logs, inspect, cancel\n"
                    "- scheduled run, scheduled ps, scheduled inspect, "
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

    async def _seed_trackio_dashboard(self, space_id: str) -> None:
        """Idempotently install trackio dashboard files into *space_id* before
        the job runs. Surfaces seed progress as tool_log events but never
        raises — a seed failure should not block job submission, since trackio
        often still works when the Space already has dashboard code from a
        previous run.
        """
        loop = asyncio.get_running_loop()

        def _log(msg: str) -> None:
            if self.session is None:
                return
            loop.call_soon_threadsafe(
                self.session.event_queue.put_nowait,
                Event(event_type="tool_log", data={"tool": "hf_jobs", "log": msg}),
            )

        try:
            await asyncio.to_thread(
                ensure_trackio_dashboard, space_id, self.hf_token, _log
            )
        except Exception as e:
            logger.warning(f"trackio dashboard seed failed for {space_id}: {e}")
            _log(f"trackio dashboard seed failed: {e}")

    async def _wait_for_job_completion(
        self, job_id: str, namespace: Optional[str] = None
    ) -> tuple[str, list[str]]:
        """
        Stream job logs until completion, printing them in real-time.
        Implements retry logic to handle connection drops during long-running jobs.

        Returns:
            tuple: (final_status, all_logs)
        """
        all_logs = []
        terminal_states = {"COMPLETED", "FAILED", "CANCELED", "ERROR"}
        max_retries = 100  # Allow many retries for 8h+ jobs
        retry_delay = 5  # Seconds between retries

        for _ in range(max_retries):
            try:
                # Use a queue to bridge sync generator to async consumer
                queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def log_producer():
                    try:
                        # fetch_job_logs is a blocking sync generator
                        logs_gen = self.api.fetch_job_logs(job_id=job_id, namespace=namespace)
                        for line in logs_gen:
                            # Push line to queue thread-safely
                            loop.call_soon_threadsafe(queue.put_nowait, line)
                        # Signal EOF
                        loop.call_soon_threadsafe(queue.put_nowait, None)
                    except Exception as e:
                        # Signal error
                        loop.call_soon_threadsafe(queue.put_nowait, e)

                # Start producer in a background thread so it doesn't block the event loop
                producer_future = loop.run_in_executor(None, log_producer)

                # Consume logs from the queue as they arrive
                while True:
                    item = await queue.get()

                    # EOF sentinel
                    if item is None:
                        break

                    # Error occurred in producer
                    if isinstance(item, Exception):
                        raise item

                    # Process log line
                    log_line = item
                    logger.debug(log_line)
                    if self.log_callback:
                        await self.log_callback(log_line)
                    all_logs.append(log_line)

                # If we get here, streaming completed normally (EOF received)
                # Wait for thread to cleanup (should be done)
                await producer_future
                break

            except (
                ConnectionError,
                TimeoutError,
                OSError,
                http.client.IncompleteRead,
                httpx.RemoteProtocolError,
                httpx.ReadError,
                HfHubHTTPError,
            ) as e:
                # Connection dropped - check if job is still running
                try:
                    job_info = await _async_call(
                        self.api.inspect_job, job_id=job_id, namespace=namespace
                    )
                    current_status = job_info.status.stage

                    if current_status in terminal_states:
                        # Job finished, no need to retry
                        logger.info(f"Job reached terminal state: {current_status}")
                        break

                    # Job still running, retry connection
                    logger.warning(
                        f"Connection interrupted ({str(e)[:50]}...), reconnecting in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                except (ConnectionError, TimeoutError, OSError):
                    # Can't even check job status, wait and retry
                    logger.warning(f"Connection error, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue

        # Fetch final job status — retry briefly if still RUNNING
        # (the API may lag a few seconds behind the log stream ending)
        final_status = "UNKNOWN"
        for _ in range(6):
            job_info = await _async_call(
                self.api.inspect_job, job_id=job_id, namespace=namespace
            )
            final_status = job_info.status.stage
            if final_status in terminal_states:
                break
            await asyncio.sleep(2.5)

        return final_status, all_logs

    async def _run_job(self, args: Dict[str, Any]) -> ToolResult:
        """Run a job using HfApi.run_job() - smart detection of Python vs Docker mode"""
        try:
            script = args.get("script")
            command = args.get("command")

            # Validate mutually exclusive parameters
            if script and command:
                raise ValueError(
                    "'script' and 'command' are mutually exclusive. Provide one or the other, not both."
                )

            if not script and not command:
                raise ValueError(
                    "Either 'script' (for Python) or 'command' (for Docker) must be provided."
                )

            # Python mode: script provided
            if script:
                # Get dependencies and ensure hf-transfer is included
                deps = _ensure_hf_transfer_dependency(args.get("dependencies"))

                # Resolve the command based on script type (URL, inline, or file)
                command = _resolve_uv_command(
                    script=script,
                    with_deps=deps,
                    python=args.get("python"),
                    script_args=args.get("script_args"),
                )

                # Use UV image unless overridden
                image = args.get("image", UV_DEFAULT_IMAGE)
                job_type = "Python"

            # Docker mode: command provided
            else:
                image = args.get("image", "python:3.12")
                job_type = "Docker"

            # Run the job
            flavor = args.get("hardware_flavor", "cpu-basic")
            timeout_str = args.get("timeout", "30m")

            # Trackio: agent-declared space + project become env vars on the job
            # so trackio.init() picks them up automatically. We also surface them
            # in tool_state_change so the frontend can embed the dashboard.
            env_dict = _add_default_env(args.get("env"))
            trackio_space_id = args.get("trackio_space_id")
            trackio_project = args.get("trackio_project")
            if trackio_space_id:
                env_dict["TRACKIO_SPACE_ID"] = trackio_space_id
                await self._seed_trackio_dashboard(trackio_space_id)
            if trackio_project:
                env_dict["TRACKIO_PROJECT"] = trackio_project

            try:
                job = await _async_call(
                    self.api.run_job,
                    image=image,
                    command=command,
                    env=env_dict,
                    secrets=_add_environment_variables(args.get("secrets"), self.hf_token),
                    flavor=flavor,
                    timeout=timeout_str,
                    namespace=self.namespace,
                )
            except HfHubHTTPError as e:
                if is_billing_error(str(e)):
                    if self.session and self.tool_call_id:
                        await self.session.send_event(
                            Event(
                                event_type="tool_state_change",
                                data={
                                    "tool_call_id": self.tool_call_id,
                                    "tool": "hf_jobs",
                                    "state": "billing_required",
                                    "namespace": self.namespace,
                                },
                            )
                        )
                    return {
                        "formatted": (
                            f"Hugging Face Jobs rejected this run because the "
                            f"namespace `{self.namespace}` has no available credits. "
                            "Tell the user to add credits at "
                            "https://huggingface.co/settings/billing — once topped up, "
                            "re-run this same job. (Switching namespaces is fine if "
                            "another wallet has credits.)"
                        ),
                        "totalResults": 0,
                        "resultsShared": 0,
                        "isError": True,
                    }
                raise

            # Track job ID for cancellation on interrupt
            if self.session:
                self.session._running_job_ids.add(job.id)

            # Send job URL immediately after job creation (before waiting for completion)
            if self.session and self.tool_call_id:
                state_data: Dict[str, Any] = {
                    "tool_call_id": self.tool_call_id,
                    "tool": "hf_jobs",
                    "state": "running",
                    "jobUrl": job.url,
                }
                if trackio_space_id:
                    state_data["trackioSpaceId"] = trackio_space_id
                if trackio_project:
                    state_data["trackioProject"] = trackio_project
                await self.session.send_event(
                    Event(event_type="tool_state_change", data=state_data)
                )

            # Telemetry: job submission + completion (infra consumption signal).
            submit_ts = None
            if self.session:
                from agent.core import telemetry
                submit_ts = await telemetry.record_hf_job_submit(
                    self.session, job,
                    {**args, "hardware_flavor": flavor, "timeout": timeout_str, "namespace": self.namespace},
                    image=image, job_type=job_type,
                )
                # Top-up signal: this submit succeeded after a prior billing
                # block in the same session, and we haven't fired the event
                # yet — the user came back from the HF billing flow.
                events = self.session.logged_events
                already_fired = any(
                    e.get("event_type") == "credits_topped_up" for e in events
                )
                if not already_fired:
                    blocked = any(
                        e.get("event_type") == "tool_state_change"
                        and (e.get("data") or {}).get("state") == "billing_required"
                        for e in events
                    )
                    if blocked:
                        await telemetry.record_credits_topped_up(
                            self.session, namespace=self.namespace,
                        )

            # Wait for completion and stream logs
            logger.info(f"{job_type} job started: {job.url}")
            logger.info("Streaming logs...")

            final_status, all_logs = await self._wait_for_job_completion(
                job_id=job.id,
                namespace=self.namespace,
            )

            if self.session and submit_ts is not None:
                from agent.core import telemetry
                await telemetry.record_hf_job_complete(
                    self.session, job,
                    flavor=flavor, final_status=final_status, submit_ts=submit_ts,
                )

            # Untrack job ID (completed or failed, no longer needs cancellation)
            if self.session:
                self.session._running_job_ids.discard(job.id)

            # Notify frontend of final status
            if self.session and self.tool_call_id:
                final_data: Dict[str, Any] = {
                    "tool_call_id": self.tool_call_id,
                    "tool": "hf_jobs",
                    "state": final_status.lower(),
                    "jobUrl": job.url,
                }
                if trackio_space_id:
                    final_data["trackioSpaceId"] = trackio_space_id
                if trackio_project:
                    final_data["trackioProject"] = trackio_project
                await self.session.send_event(
                    Event(event_type="tool_state_change", data=final_data)
                )

            # Filter out UV package installation output
            filtered_logs = _filter_uv_install_output(all_logs)

            # Format all logs for the agent
            log_text = _strip_ansi("\n".join(filtered_logs)) if filtered_logs else "(no logs)"

            response = f"""{job_type} job completed!

**Job ID:** {job.id}
**Final Status:** {final_status}
**View at:** {job.url}

**Logs:**
```
{log_text}
```"""
            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            raise Exception(f"Failed to run job: {str(e)}")

    async def _list_jobs(self, args: Dict[str, Any]) -> ToolResult:
        """List jobs using HfApi.list_jobs()"""
        jobs_list = await _async_call(self.api.list_jobs, namespace=self.namespace)

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
                "formatted": 'No running jobs found. Use `{"operation": "ps", "all": true}` to show all jobs.',
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
            logs_gen = self.api.fetch_job_logs(job_id=job_id, namespace=self.namespace)
            logs = await _async_call(list, logs_gen)

            if not logs:
                return {
                    "formatted": f"No logs available for job {job_id}",
                    "totalResults": 0,
                    "resultsShared": 0,
                }

            log_text = _strip_ansi("\n".join(logs))
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
                    namespace=self.namespace,
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
            namespace=self.namespace,
        )

        response = f"""✓ Job {job_id} has been cancelled.

To verify, call this tool with `{{"operation": "inspect", "job_id": "{job_id}"}}`"""

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    async def _scheduled_run(self, args: Dict[str, Any]) -> ToolResult:
        """Create scheduled job using HfApi.create_scheduled_job() - smart detection of Python vs Docker mode"""
        try:
            script = args.get("script")
            command = args.get("command")
            schedule = args.get("schedule")

            if not schedule:
                raise ValueError("schedule is required for scheduled jobs")

            # Validate mutually exclusive parameters
            if script and command:
                raise ValueError(
                    "'script' and 'command' are mutually exclusive. Provide one or the other, not both."
                )

            if not script and not command:
                raise ValueError(
                    "Either 'script' (for Python) or 'command' (for Docker) must be provided."
                )

            # Python mode: script provided
            if script:
                # Get dependencies and ensure hf-transfer is included
                deps = _ensure_hf_transfer_dependency(args.get("dependencies"))

                # Resolve the command based on script type
                command = _resolve_uv_command(
                    script=script,
                    with_deps=deps,
                    python=args.get("python"),
                    script_args=args.get("script_args"),
                )

                # Use UV image unless overridden
                image = args.get("image", UV_DEFAULT_IMAGE)
                job_type = "Python"

            # Docker mode: command provided
            else:
                image = args.get("image", "python:3.12")
                job_type = "Docker"

            # Create scheduled job
            scheduled_job = await _async_call(
                self.api.create_scheduled_job,
                image=image,
                command=command,
                schedule=schedule,
                env=_add_default_env(args.get("env")),
                secrets=_add_environment_variables(args.get("secrets"), self.hf_token),
                flavor=args.get("hardware_flavor", "cpu-basic"),
                timeout=args.get("timeout", "30m"),
                namespace=self.namespace,
            )

            scheduled_dict = _scheduled_job_info_to_dict(scheduled_job)

            response = f"""✓ Scheduled {job_type} job created successfully!

**Scheduled Job ID:** {scheduled_dict["id"]}
**Schedule:** {scheduled_dict["schedule"]}
**Suspended:** {"Yes" if scheduled_dict.get("suspend") else "No"}
**Next Run:** {scheduled_dict.get("nextRun", "N/A")}

To inspect, call this tool with `{{"operation": "scheduled inspect", "scheduled_job_id": "{scheduled_dict["id"]}"}}`
To list all, call this tool with `{{"operation": "scheduled ps"}}`"""

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            raise Exception(f"Failed to create scheduled job: {str(e)}")

    async def _list_scheduled_jobs(self, args: Dict[str, Any]) -> ToolResult:
        """List scheduled jobs using HfApi.list_scheduled_jobs()"""
        scheduled_jobs_list = await _async_call(
            self.api.list_scheduled_jobs,
            namespace=self.namespace,
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
                "formatted": 'No active scheduled jobs found. Use `{"operation": "scheduled ps", "all": true}` to show suspended jobs.',
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
            namespace=self.namespace,
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
            namespace=self.namespace,
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
            namespace=self.namespace,
        )

        response = f"""✓ Scheduled job {scheduled_job_id} has been suspended.

To resume, call this tool with `{{"operation": "scheduled resume", "scheduled_job_id": "{scheduled_job_id}"}}`"""

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
            namespace=self.namespace,
        )

        response = f"""✓ Scheduled job {scheduled_job_id} has been resumed.

To inspect, call this tool with `{{"operation": "scheduled inspect", "scheduled_job_id": "{scheduled_job_id}"}}`"""

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}


# Tool specification for agent registration
HF_JOBS_TOOL_SPEC = {
    "name": "hf_jobs",
    "description": (
        "Execute Python scripts or Docker containers on HF cloud infrastructure.\n\n"
        "Two modes (mutually exclusive): Python mode (script + dependencies) or Docker mode (command + image). "
        "Provide exactly ONE of 'script' or 'command'.\n\n"
        "BEFORE submitting training/fine-tuning jobs:\n"
        "- You MUST have called github_find_examples + github_read_file to find a working reference implementation. "
        "Scripts based on your internal knowledge WILL use outdated APIs and fail.\n"
        "- You MUST have validated dataset format via hf_inspect_dataset or hub_repo_details.\n"
        "- Training config MUST include push_to_hub=True and hub_model_id. "
        "Job storage is EPHEMERAL — all files are deleted when the job ends. Without push_to_hub, trained models are lost permanently.\n"
        "- Include trackio monitoring and provide the dashboard URL to the user. "
        "When the script uses report_to='trackio', also pass `trackio_space_id` "
        "(e.g. '<username>/mlintern-<8char>') and `trackio_project` as tool args — "
        "they are injected as TRACKIO_SPACE_ID/TRACKIO_PROJECT env vars and let the UI embed the live dashboard.\n\n"
        "BATCH/ABLATION JOBS: Submit ONE job first. Check logs to confirm it starts training successfully. "
        "Only then submit the remaining jobs. Never submit all at once — if there's a bug, all jobs fail.\n\n"
        "Operations: run, ps, logs, inspect, cancel, scheduled run/ps/inspect/delete/suspend/resume.\n\n"
        f"Hardware: CPU: {CPU_FLAVORS_DESC}. GPU: {GPU_FLAVORS_DESC}.\n"
        "Common picks: t4-small ($0.60/hr, 1-3B), a10g-large ($2/hr, 7-13B), a100-large ($4/hr, 30B+), h100 ($6/hr, 70B+). "
        "Note: a10g-small and a10g-large have the SAME 24GB GPU — the difference is CPU/RAM only.\n\n"
        "OOM RECOVERY: When a training job fails with CUDA OOM:\n"
        "1. Reduce per_device_train_batch_size and increase gradient_accumulation_steps proportionally (keep effective batch size identical)\n"
        "2. Enable gradient_checkpointing=True\n"
        "3. Upgrade to larger GPU (a10g→a100→h100)\n"
        "Do NOT switch training methods (e.g. full SFT to LoRA) or reduce max_length — those change what the user gets and require explicit approval.\n\n"
        "Examples:\n"
        "Training: {'operation': 'run', 'script': '/app/train.py', 'dependencies': ['transformers', 'trl', 'torch', 'datasets', 'trackio'], 'hardware_flavor': 'a100-large', 'timeout': '8h'}\n"
        "Monitor: {'operation': 'ps'}, {'operation': 'logs', 'job_id': 'xxx'}, {'operation': 'cancel', 'job_id': 'xxx'}"
        "Docker: {'operation': 'run', 'command': ['duckdb', '-c', 'select 1 + 2'], 'image': 'duckdb/duckdb', 'hardware_flavor': 'cpu-basic', 'timeout': '1h'}\n"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "run",
                    "ps",
                    "logs",
                    "inspect",
                    "cancel",
                    "scheduled run",
                    "scheduled ps",
                    "scheduled inspect",
                    "scheduled delete",
                    "scheduled suspend",
                    "scheduled resume",
                ],
                "description": "Operation to execute.",
            },
            "script": {
                "type": "string",
                "description": (
                    "Python code or sandbox file path (e.g. '/app/train.py') or URL. "
                    "Triggers Python mode. For ML training: base this on a working example found via github_find_examples, not on internal knowledge. "
                    "Mutually exclusive with 'command'."
                ),
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Pip packages to install. Include ALL required packages. "
                    "Common training set: ['transformers', 'trl', 'torch', 'datasets', 'trackio', 'accelerate']. "
                    "Only used with 'script'."
                ),
            },
            "image": {
                "type": "string",
                "description": "Docker image. Optional — auto-selected if not provided. Use with 'command'.",
            },
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Command to execute as list. Triggers Docker mode. Mutually exclusive with 'script'.",
            },
            "hardware_flavor": {
                "type": "string",
                "description": (
                    "Hardware type. Sizing guide: 1-3B params → t4-small/a10g-small, "
                    "7-13B → a10g-large, 30B+ → a100-large, 70B+ → h100/h100x8. "
                    f"All options: CPU: {CPU_FLAVORS}. GPU: {GPU_FLAVORS}."
                ),
            },
            "timeout": {
                "type": "string",
                "description": (
                    "Maximum job runtime. MUST be >2h for any training job — default 30m kills training mid-run. "
                    "Guidelines: 1-3B models: 3-4h, 7-13B: 6-8h, 30B+: 12-24h. "
                    "Use 30m-1h only for quick data processing or inference tasks. Default: '30m'."
                ),
            },
            "env": {
                "type": "object",
                "description": "Environment variables {'KEY': 'VALUE'}. HF_TOKEN is auto-included.",
            },
            "trackio_space_id": {
                "type": "string",
                "description": (
                    "Optional. The HF Space hosting the trackio dashboard for this run "
                    "(e.g. '<username>/mlintern-<8char>', under YOUR HF namespace). "
                    "Injected as TRACKIO_SPACE_ID env var and used by the UI to embed "
                    "the live dashboard. Set this whenever the script uses "
                    "report_to='trackio'. The Space is auto-created and seeded with the "
                    "trackio dashboard before the job starts — DO NOT pre-create it via "
                    "hf_repo_git, that produces an empty Space that breaks the embed."
                ),
            },
            "trackio_project": {
                "type": "string",
                "description": (
                    "Optional. The trackio project name to log this run under. "
                    "Injected as TRACKIO_PROJECT env var and used by the UI to filter "
                    "the embedded dashboard to this project."
                ),
            },
            "namespace": {
                "type": "string",
                "description": (
                    "Optional namespace to run the job under. Must be the caller's own "
                    "account or an org they belong to. If omitted, defaults to the "
                    "caller's personal account. Credits are billed against this namespace."
                ),
            },
            "job_id": {
                "type": "string",
                "description": "Job ID. Required for: logs, inspect, cancel.",
            },
            "scheduled_job_id": {
                "type": "string",
                "description": "Scheduled job ID. Required for: scheduled inspect/delete/suspend/resume.",
            },
            "schedule": {
                "type": "string",
                "description": "Cron schedule or preset (@hourly, @daily, @weekly, @monthly). Required for: scheduled run.",
            },
        },
        "required": ["operation"],
    },
}


async def hf_jobs_handler(
    arguments: Dict[str, Any], session: Any = None, tool_call_id: str | None = None
) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:

        async def log_callback(log: str):
            if session:
                await session.send_event(
                    Event(event_type="tool_log", data={"tool": "hf_jobs", "log": log})
                )

        # If script is a sandbox file path, read it from the sandbox
        script = arguments.get("script", "")
        sandbox = getattr(session, "sandbox", None) if session else None
        if sandbox and script:
            from agent.tools.sandbox_tool import resolve_sandbox_script
            content, error = await resolve_sandbox_script(sandbox, script)
            if error:
                return error, False
            if content:
                arguments = {**arguments, "script": content}

        hf_token = session.hf_token if session else None
        try:
            namespace, jobs_access = await resolve_jobs_namespace(
                hf_token or "",
                arguments.get("namespace"),
            )
        except JobsAccessError as e:
            return str(e), False

        tool = HfJobsTool(
            namespace=namespace,
            hf_token=hf_token,
            jobs_access=jobs_access,
            log_callback=log_callback if session else None,
            session=session,
            tool_call_id=tool_call_id,
        )
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error executing HF Jobs tool: {str(e)}", False
