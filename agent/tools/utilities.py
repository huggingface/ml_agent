"""
Utility functions for Hugging Face tools

Ported from: hf-mcp-server/packages/mcp/src/jobs/formatters.ts
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


def truncate(text: str, max_length: int) -> str:
    """Truncate a string to a maximum length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_date(date_str: Optional[str]) -> str:
    """Format a date string to a readable format"""
    if not date_str:
        return "N/A"
    try:
        date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return date.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return date_str


def format_command(command: Optional[List[str]]) -> str:
    """Format command array as a single string"""
    if not command or len(command) == 0:
        return "N/A"
    return " ".join(command)


def get_image_or_space(job: Dict[str, Any]) -> str:
    """Get image/space identifier from job"""
    if job.get("spaceId"):
        return job["spaceId"]
    if job.get("dockerImage"):
        return job["dockerImage"]
    return "N/A"


def format_jobs_table(jobs: List[Dict[str, Any]]) -> str:
    """Format jobs as a markdown table"""
    if len(jobs) == 0:
        return "No jobs found."

    # Calculate dynamic ID column width
    longest_id_length = max(len(job["id"]) for job in jobs)
    id_column_width = max(longest_id_length, len("JOB ID"))

    # Define column widths
    col_widths = {
        "id": id_column_width,
        "image": 20,
        "command": 30,
        "created": 19,
        "status": 12,
    }

    # Build header
    header = f"| {'JOB ID'.ljust(col_widths['id'])} | {'IMAGE/SPACE'.ljust(col_widths['image'])} | {'COMMAND'.ljust(col_widths['command'])} | {'CREATED'.ljust(col_widths['created'])} | {'STATUS'.ljust(col_widths['status'])} |"
    separator = f"|{'-' * (col_widths['id'] + 2)}|{'-' * (col_widths['image'] + 2)}|{'-' * (col_widths['command'] + 2)}|{'-' * (col_widths['created'] + 2)}|{'-' * (col_widths['status'] + 2)}|"

    # Build rows
    rows = []
    for job in jobs:
        job_id = job["id"]
        image = truncate(get_image_or_space(job), col_widths["image"])
        command = truncate(format_command(job.get("command")), col_widths["command"])
        created = truncate(format_date(job.get("createdAt")), col_widths["created"])
        status = truncate(job["status"]["stage"], col_widths["status"])

        rows.append(
            f"| {job_id.ljust(col_widths['id'])} | {image.ljust(col_widths['image'])} | {command.ljust(col_widths['command'])} | {created.ljust(col_widths['created'])} | {status.ljust(col_widths['status'])} |"
        )

    return "\n".join([header, separator] + rows)


def format_scheduled_jobs_table(jobs: List[Dict[str, Any]]) -> str:
    """Format scheduled jobs as a markdown table"""
    if len(jobs) == 0:
        return "No scheduled jobs found."

    # Calculate dynamic ID column width
    longest_id_length = max(len(job["id"]) for job in jobs)
    id_column_width = max(longest_id_length, len("ID"))

    # Define column widths
    col_widths = {
        "id": id_column_width,
        "schedule": 12,
        "image": 18,
        "command": 25,
        "lastRun": 19,
        "nextRun": 19,
        "suspend": 9,
    }

    # Build header
    header = f"| {'ID'.ljust(col_widths['id'])} | {'SCHEDULE'.ljust(col_widths['schedule'])} | {'IMAGE/SPACE'.ljust(col_widths['image'])} | {'COMMAND'.ljust(col_widths['command'])} | {'LAST RUN'.ljust(col_widths['lastRun'])} | {'NEXT RUN'.ljust(col_widths['nextRun'])} | {'SUSPENDED'.ljust(col_widths['suspend'])} |"
    separator = f"|{'-' * (col_widths['id'] + 2)}|{'-' * (col_widths['schedule'] + 2)}|{'-' * (col_widths['image'] + 2)}|{'-' * (col_widths['command'] + 2)}|{'-' * (col_widths['lastRun'] + 2)}|{'-' * (col_widths['nextRun'] + 2)}|{'-' * (col_widths['suspend'] + 2)}|"

    # Build rows
    rows = []
    for job in jobs:
        job_id = job["id"]
        schedule = truncate(job["schedule"], col_widths["schedule"])
        image = truncate(get_image_or_space(job["jobSpec"]), col_widths["image"])
        command = truncate(
            format_command(job["jobSpec"].get("command")), col_widths["command"]
        )
        last_run = truncate(format_date(job.get("lastRun")), col_widths["lastRun"])
        next_run = truncate(format_date(job.get("nextRun")), col_widths["nextRun"])
        suspend = "Yes" if job.get("suspend") else "No"

        rows.append(
            f"| {job_id.ljust(col_widths['id'])} | {schedule.ljust(col_widths['schedule'])} | {image.ljust(col_widths['image'])} | {command.ljust(col_widths['command'])} | {last_run.ljust(col_widths['lastRun'])} | {next_run.ljust(col_widths['nextRun'])} | {suspend.ljust(col_widths['suspend'])} |"
        )

    return "\n".join([header, separator] + rows)


def format_job_details(jobs: Any) -> str:
    """Format job details as JSON in a markdown code block"""
    import json

    job_array = jobs if isinstance(jobs, list) else [jobs]
    json_str = json.dumps(job_array, indent=2)
    return f"```json\n{json_str}\n```"


def format_scheduled_job_details(jobs: Any) -> str:
    """Format scheduled job details as JSON in a markdown code block"""
    import json

    job_array = jobs if isinstance(jobs, list) else [jobs]
    json_str = json.dumps(job_array, indent=2)
    return f"```json\n{json_str}\n```"
