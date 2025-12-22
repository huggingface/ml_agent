"""
Tests for HF Jobs Tool

Tests the refactored jobs tool implementation using huggingface-hub library
"""

from unittest.mock import AsyncMock, patch

import pytest

from agent.tools.jobs_tool import HfJobsTool, hf_jobs_handler


def create_mock_job_info(
    job_id="test-job-1",
    stage="RUNNING",
    command=None,
    docker_image="python:3.12",
):
    """Create a mock JobInfo object"""
    from huggingface_hub._jobs_api import JobInfo

    if command is None:
        command = ["echo", "test"]

    return JobInfo(
        id=job_id,
        created_at="2024-01-01T00:00:00.000000Z",
        docker_image=docker_image,
        space_id=None,
        command=command,
        arguments=[],
        environment={},
        secrets={},
        flavor="cpu-basic",
        status={"stage": stage, "message": None},
        owner={"id": "123", "name": "test-user", "type": "user"},
        endpoint="https://huggingface.co",
        url=f"https://huggingface.co/jobs/test-user/{job_id}",
    )


def create_mock_scheduled_job_info(
    job_id="sched-job-1",
    schedule="@daily",
    suspend=False,
):
    """Create a mock ScheduledJobInfo object"""
    from huggingface_hub._jobs_api import ScheduledJobInfo

    return ScheduledJobInfo(
        id=job_id,
        created_at="2024-01-01T00:00:00.000000Z",
        job_spec={
            "docker_image": "python:3.12",
            "space_id": None,
            "command": ["python", "backup.py"],
            "arguments": [],
            "environment": {},
            "secrets": {},
            "flavor": "cpu-basic",
            "timeout": 1800,
            "tags": None,
            "arch": None,
        },
        schedule=schedule,
        suspend=suspend,
        concurrency=False,
        status={
            "last_job": None,
            "next_job_run_at": "2024-01-02T00:00:00.000000Z",
        },
        owner={"id": "123", "name": "test-user", "type": "user"},
    )


@pytest.mark.asyncio
async def test_show_help():
    """Test that help message is shown when no operation specified"""
    tool = HfJobsTool()
    result = await tool.execute({})

    assert "HuggingFace Jobs API" in result["formatted"]
    assert "Available Commands" in result["formatted"]
    assert result["totalResults"] == 1
    assert not result.get("isError", False)


@pytest.mark.asyncio
async def test_show_operation_help():
    """Test operation-specific help"""
    tool = HfJobsTool()
    result = await tool.execute({"operation": "run", "args": {"help": True}})

    assert "Help for operation" in result["formatted"]
    assert result["totalResults"] == 1


@pytest.mark.asyncio
async def test_invalid_operation():
    """Test invalid operation handling"""
    tool = HfJobsTool()
    result = await tool.execute({"operation": "invalid_op"})

    assert result.get("isError") == True
    assert "Unknown operation" in result["formatted"]


@pytest.mark.asyncio
async def test_run_job_missing_command():
    """Test run job with missing required parameter"""
    tool = HfJobsTool()

    # Mock the HfApi.run_job to raise an error
    with patch.object(tool.api, "run_job") as mock_run:
        mock_run.side_effect = Exception("command parameter is required")

        result = await tool.execute(
            {"operation": "run", "args": {"image": "python:3.12"}}
        )

        assert result.get("isError") == True


@pytest.mark.asyncio
async def test_list_jobs_mock():
    """Test list jobs with mock API"""
    tool = HfJobsTool()

    # Create mock job objects
    running_job = create_mock_job_info("test-job-1", "RUNNING")
    completed_job = create_mock_job_info(
        "test-job-2", "COMPLETED", ["python", "script.py"]
    )

    # Mock the HfApi.list_jobs method
    with patch.object(tool.api, "list_jobs") as mock_list:
        mock_list.return_value = [running_job, completed_job]

        # Test listing only running jobs (default)
        result = await tool.execute({"operation": "ps"})

        assert not result.get("isError", False)
        assert "test-job-1" in result["formatted"]
        assert "test-job-2" not in result["formatted"]  # COMPLETED jobs filtered out
        assert result["totalResults"] == 1
        assert result["resultsShared"] == 1

        # Test listing all jobs
        result = await tool.execute({"operation": "ps", "args": {"all": True}})

        assert not result.get("isError", False)
        assert "test-job-1" in result["formatted"]
        assert "test-job-2" in result["formatted"]
        assert result["totalResults"] == 2
        assert result["resultsShared"] == 2


@pytest.mark.asyncio
async def test_inspect_job_mock():
    """Test inspect job with mock API"""
    tool = HfJobsTool()

    mock_job = create_mock_job_info("test-job-1", "RUNNING")

    with patch.object(tool.api, "inspect_job") as mock_inspect:
        mock_inspect.return_value = mock_job

        result = await tool.execute(
            {"operation": "inspect", "args": {"job_id": "test-job-1"}}
        )

        assert not result.get("isError", False)
        assert "test-job-1" in result["formatted"]
        assert "Job Details" in result["formatted"]
        mock_inspect.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_job_mock():
    """Test cancel job with mock API"""
    tool = HfJobsTool()

    with patch.object(tool.api, "cancel_job") as mock_cancel:
        mock_cancel.return_value = None

        result = await tool.execute(
            {"operation": "cancel", "args": {"job_id": "test-job-1"}}
        )

        assert not result.get("isError", False)
        assert "cancelled" in result["formatted"]
        assert "test-job-1" in result["formatted"]
        mock_cancel.assert_called_once()


@pytest.mark.asyncio
async def test_run_job_mock():
    """Test run job with mock API"""
    tool = HfJobsTool()

    mock_job = create_mock_job_info("new-job-123", "RUNNING")

    with patch.object(tool.api, "run_job") as mock_run:
        mock_run.return_value = mock_job

        result = await tool.execute(
            {
                "operation": "run",
                "args": {
                    "image": "python:3.12",
                    "command": ["python", "-c", "print('test')"],
                    "flavor": "cpu-basic",
                    "detach": True,
                },
            }
        )

        assert not result.get("isError", False)
        assert "new-job-123" in result["formatted"]
        assert "Job started" in result["formatted"]
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_run_uv_job_mock():
    """Test run UV job with mock API"""
    tool = HfJobsTool()

    mock_job = create_mock_job_info("uv-job-456", "RUNNING")

    with patch.object(tool.api, "run_uv_job") as mock_run:
        mock_run.return_value = mock_job

        result = await tool.execute(
            {
                "operation": "uv",
                "args": {
                    "script": "print('Hello UV')",
                    "flavor": "cpu-basic",
                },
            }
        )

        assert not result.get("isError", False)
        assert "uv-job-456" in result["formatted"]
        assert "UV Job started" in result["formatted"]
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_get_logs_mock():
    """Test get logs with mock API"""
    tool = HfJobsTool()

    # Mock fetch_job_logs to return a generator
    def log_generator():
        yield "Log line 1"
        yield "Log line 2"
        yield "Hello from HF Jobs!"

    with patch.object(tool.api, "fetch_job_logs") as mock_logs:
        mock_logs.return_value = log_generator()

        result = await tool.execute(
            {"operation": "logs", "args": {"job_id": "test-job-1"}}
        )

        assert not result.get("isError", False)
        assert "Log line 1" in result["formatted"]
        assert "Hello from HF Jobs!" in result["formatted"]


@pytest.mark.asyncio
async def test_handler():
    """Test the handler function"""
    with patch("agent.tools.jobs_tool.HfJobsTool") as MockTool:
        mock_tool_instance = MockTool.return_value
        mock_tool_instance.execute = AsyncMock(
            return_value={
                "formatted": "Test output",
                "totalResults": 1,
                "resultsShared": 1,
                "isError": False,
            }
        )

        output, success = await hf_jobs_handler({"operation": "ps"})

        assert success == True
        assert "Test output" in output


@pytest.mark.asyncio
async def test_handler_error():
    """Test handler with error"""
    with patch("agent.tools.jobs_tool.HfJobsTool") as MockTool:
        MockTool.side_effect = Exception("Test error")

        output, success = await hf_jobs_handler({})

        assert success == False
        assert "Error" in output


@pytest.mark.asyncio
async def test_scheduled_jobs_mock():
    """Test scheduled jobs operations with mock API"""
    tool = HfJobsTool()

    mock_scheduled_job = create_mock_scheduled_job_info()

    # Test list scheduled jobs
    with patch.object(tool.api, "list_scheduled_jobs") as mock_list:
        mock_list.return_value = [mock_scheduled_job]

        result = await tool.execute({"operation": "scheduled ps"})

        assert not result.get("isError", False)
        assert "sched-job-1" in result["formatted"]
        assert "Scheduled Jobs" in result["formatted"]


@pytest.mark.asyncio
async def test_create_scheduled_job_mock():
    """Test create scheduled job with mock API"""
    tool = HfJobsTool()

    mock_scheduled_job = create_mock_scheduled_job_info()

    with patch.object(tool.api, "create_scheduled_job") as mock_create:
        mock_create.return_value = mock_scheduled_job

        result = await tool.execute(
            {
                "operation": "scheduled run",
                "args": {
                    "image": "python:3.12",
                    "command": ["python", "backup.py"],
                    "schedule": "@daily",
                    "flavor": "cpu-basic",
                },
            }
        )

        assert not result.get("isError", False)
        assert "sched-job-1" in result["formatted"]
        assert "Scheduled job created" in result["formatted"]
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_inspect_scheduled_job_mock():
    """Test inspect scheduled job with mock API"""
    tool = HfJobsTool()

    mock_scheduled_job = create_mock_scheduled_job_info()

    with patch.object(tool.api, "inspect_scheduled_job") as mock_inspect:
        mock_inspect.return_value = mock_scheduled_job

        result = await tool.execute(
            {
                "operation": "scheduled inspect",
                "args": {"scheduled_job_id": "sched-job-1"},
            }
        )

        assert not result.get("isError", False)
        assert "sched-job-1" in result["formatted"]
        assert "Scheduled Job Details" in result["formatted"]


@pytest.mark.asyncio
async def test_suspend_scheduled_job_mock():
    """Test suspend scheduled job with mock API"""
    tool = HfJobsTool()

    with patch.object(tool.api, "suspend_scheduled_job") as mock_suspend:
        mock_suspend.return_value = None

        result = await tool.execute(
            {
                "operation": "scheduled suspend",
                "args": {"scheduled_job_id": "sched-job-1"},
            }
        )

        assert not result.get("isError", False)
        assert "suspended" in result["formatted"]
        assert "sched-job-1" in result["formatted"]


@pytest.mark.asyncio
async def test_resume_scheduled_job_mock():
    """Test resume scheduled job with mock API"""
    tool = HfJobsTool()

    with patch.object(tool.api, "resume_scheduled_job") as mock_resume:
        mock_resume.return_value = None

        result = await tool.execute(
            {
                "operation": "scheduled resume",
                "args": {"scheduled_job_id": "sched-job-1"},
            }
        )

        assert not result.get("isError", False)
        assert "resumed" in result["formatted"]
        assert "sched-job-1" in result["formatted"]


@pytest.mark.asyncio
async def test_delete_scheduled_job_mock():
    """Test delete scheduled job with mock API"""
    tool = HfJobsTool()

    with patch.object(tool.api, "delete_scheduled_job") as mock_delete:
        mock_delete.return_value = None

        result = await tool.execute(
            {
                "operation": "scheduled delete",
                "args": {"scheduled_job_id": "sched-job-1"},
            }
        )

        assert not result.get("isError", False)
        assert "deleted" in result["formatted"]
        assert "sched-job-1" in result["formatted"]


@pytest.mark.asyncio
async def test_list_jobs_with_status_filter():
    """Test list jobs with status filter"""
    tool = HfJobsTool()

    running_job = create_mock_job_info("job-1", "RUNNING")
    completed_job = create_mock_job_info("job-2", "COMPLETED")
    error_job = create_mock_job_info("job-3", "ERROR")

    with patch.object(tool.api, "list_jobs") as mock_list:
        mock_list.return_value = [running_job, completed_job, error_job]

        # Filter by status
        result = await tool.execute(
            {"operation": "ps", "args": {"all": True, "status": "ERROR"}}
        )

        assert not result.get("isError", False)
        assert "job-3" in result["formatted"]
        assert "job-1" not in result["formatted"]
        assert result["resultsShared"] == 1
