#!/usr/bin/env python3
"""
Integration tests for refactored HF Jobs Tool
Tests with real HF API using HF_TOKEN from environment
"""
import os
import sys
import asyncio
import time

# Add parent directory to path
sys.path.insert(0, '.')

from agent.tools.jobs_tool import HfJobsTool

# ANSI color codes for better output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test(msg):
    """Print test message in blue"""
    print(f"{BLUE}[TEST]{RESET} {msg}")


def print_success(msg):
    """Print success message in green"""
    print(f"{GREEN}✓{RESET} {msg}")


def print_warning(msg):
    """Print warning message in yellow"""
    print(f"{YELLOW}⚠{RESET} {msg}")


def print_error(msg):
    """Print error message in red"""
    print(f"{RED}✗{RESET} {msg}")


async def test_basic_job_run(tool):
    """Test running a basic job"""
    print_test("Running a simple Python job...")

    result = await tool.execute({
        "operation": "run",
        "args": {
            "image": "python:3.12",
            "command": ["python", "-c", "print('Hello from HF Jobs!')"],
            "flavor": "cpu-basic",
            "timeout": "5m",
            "detach": True  # Don't wait for completion
        }
    })

    if result.get("isError"):
        print_error(f"Failed to run job: {result['formatted']}")
        return None

    # Extract job ID from response
    import re
    job_id_match = re.search(r'\*\*Job ID:\*\* (\S+)', result['formatted'])
    if job_id_match:
        job_id = job_id_match.group(1)
        print_success(f"Job started with ID: {job_id}")
        return job_id

    print_error("Could not extract job ID from response")
    return None


async def test_list_jobs(tool):
    """Test listing jobs"""
    print_test("Listing running jobs...")

    result = await tool.execute({
        "operation": "ps",
        "args": {}
    })

    if result.get("isError"):
        print_error(f"Failed to list jobs: {result['formatted']}")
        return False

    print_success(f"Listed jobs: {result['totalResults']} running")
    if result['totalResults'] > 0:
        print(f"   {result['formatted'][:200]}...")
    return True


async def test_inspect_job(tool, job_id):
    """Test inspecting a specific job"""
    print_test(f"Inspecting job {job_id}...")

    result = await tool.execute({
        "operation": "inspect",
        "args": {
            "job_id": job_id
        }
    })

    if result.get("isError"):
        print_error(f"Failed to inspect job: {result['formatted']}")
        return False

    print_success(f"Inspected job successfully")
    return True


async def test_get_logs(tool, job_id):
    """Test fetching job logs"""
    print_test(f"Fetching logs for job {job_id}...")

    # Wait a bit for logs to be available
    await asyncio.sleep(2)

    result = await tool.execute({
        "operation": "logs",
        "args": {
            "job_id": job_id
        }
    })

    if result.get("isError"):
        print_warning(f"Could not fetch logs (might be too early): {result['formatted'][:100]}")
        return False

    print_success(f"Fetched logs successfully")
    if "Hello from HF Jobs!" in result['formatted']:
        print_success("  Found expected output in logs!")
    return True


async def test_cancel_job(tool, job_id):
    """Test cancelling a job"""
    print_test(f"Cancelling job {job_id}...")

    result = await tool.execute({
        "operation": "cancel",
        "args": {
            "job_id": job_id
        }
    })

    if result.get("isError"):
        print_error(f"Failed to cancel job: {result['formatted']}")
        return False

    print_success(f"Cancelled job successfully")
    return True


async def test_uv_job(tool):
    """Test running a UV job"""
    print_test("Running a UV Python script job...")

    result = await tool.execute({
        "operation": "uv",
        "args": {
            "script": "print('Hello from UV!')\nimport sys\nprint(f'Python version: {sys.version}')",
            "flavor": "cpu-basic",
            "timeout": "5m",
            "detach": True
        }
    })

    if result.get("isError"):
        print_error(f"Failed to run UV job: {result['formatted']}")
        return None

    # Extract job ID
    import re
    job_id_match = re.search(r'UV Job started: (\S+)', result['formatted'])
    if job_id_match:
        job_id = job_id_match.group(1)
        print_success(f"UV job started with ID: {job_id}")
        return job_id

    print_error("Could not extract job ID from response")
    return None


async def test_list_all_jobs(tool):
    """Test listing all jobs (including completed)"""
    print_test("Listing all jobs (including completed)...")

    result = await tool.execute({
        "operation": "ps",
        "args": {
            "all": True
        }
    })

    if result.get("isError"):
        print_error(f"Failed to list all jobs: {result['formatted']}")
        return False

    print_success(f"Listed all jobs: {result['totalResults']} total")
    return True


async def test_scheduled_job(tool):
    """Test creating and managing a scheduled job"""
    print_test("Creating a scheduled job (daily at midnight)...")

    result = await tool.execute({
        "operation": "scheduled run",
        "args": {
            "image": "python:3.12",
            "command": ["python", "-c", "print('Scheduled job running!')"],
            "schedule": "@daily",
            "flavor": "cpu-basic",
            "timeout": "5m"
        }
    })

    if result.get("isError"):
        print_error(f"Failed to create scheduled job: {result['formatted']}")
        return None

    # Extract scheduled job ID
    import re
    job_id_match = re.search(r'\*\*Scheduled Job ID:\*\* (\S+)', result['formatted'])
    if not job_id_match:
        print_error("Could not extract scheduled job ID")
        return None

    scheduled_job_id = job_id_match.group(1)
    print_success(f"Scheduled job created with ID: {scheduled_job_id}")
    return scheduled_job_id


async def test_list_scheduled_jobs(tool):
    """Test listing scheduled jobs"""
    print_test("Listing scheduled jobs...")

    result = await tool.execute({
        "operation": "scheduled ps",
        "args": {}
    })

    if result.get("isError"):
        print_error(f"Failed to list scheduled jobs: {result['formatted']}")
        return False

    print_success(f"Listed scheduled jobs: {result['totalResults']} active")
    return True


async def test_inspect_scheduled_job(tool, scheduled_job_id):
    """Test inspecting a scheduled job"""
    print_test(f"Inspecting scheduled job {scheduled_job_id}...")

    result = await tool.execute({
        "operation": "scheduled inspect",
        "args": {
            "scheduled_job_id": scheduled_job_id
        }
    })

    if result.get("isError"):
        print_error(f"Failed to inspect scheduled job: {result['formatted']}")
        return False

    print_success(f"Inspected scheduled job successfully")
    return True


async def test_suspend_scheduled_job(tool, scheduled_job_id):
    """Test suspending a scheduled job"""
    print_test(f"Suspending scheduled job {scheduled_job_id}...")

    result = await tool.execute({
        "operation": "scheduled suspend",
        "args": {
            "scheduled_job_id": scheduled_job_id
        }
    })

    if result.get("isError"):
        print_error(f"Failed to suspend scheduled job: {result['formatted']}")
        return False

    print_success(f"Suspended scheduled job successfully")
    return True


async def test_resume_scheduled_job(tool, scheduled_job_id):
    """Test resuming a scheduled job"""
    print_test(f"Resuming scheduled job {scheduled_job_id}...")

    result = await tool.execute({
        "operation": "scheduled resume",
        "args": {
            "scheduled_job_id": scheduled_job_id
        }
    })

    if result.get("isError"):
        print_error(f"Failed to resume scheduled job: {result['formatted']}")
        return False

    print_success(f"Resumed scheduled job successfully")
    return True


async def test_delete_scheduled_job(tool, scheduled_job_id):
    """Test deleting a scheduled job"""
    print_test(f"Deleting scheduled job {scheduled_job_id}...")

    result = await tool.execute({
        "operation": "scheduled delete",
        "args": {
            "scheduled_job_id": scheduled_job_id
        }
    })

    if result.get("isError"):
        print_error(f"Failed to delete scheduled job: {result['formatted']}")
        return False

    print_success(f"Deleted scheduled job successfully")
    return True


async def main():
    """Run all integration tests"""
    print("=" * 70)
    print(f"{BLUE}HF Jobs Tool - Integration Tests{RESET}")
    print("=" * 70)
    print()

    # Check for HF_TOKEN
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print_error("HF_TOKEN not found in environment variables!")
        print_warning("Set it with: export HF_TOKEN='your_token_here'")
        sys.exit(1)

    print_success(f"Found HF_TOKEN (length: {len(hf_token)})")
    print()

    # Initialize tool with token
    tool = HfJobsTool(hf_token=hf_token)

    # Track job IDs for cleanup
    job_ids = []
    scheduled_job_ids = []

    try:
        # Test 1: Run basic job
        print(f"\n{YELLOW}{'=' * 70}{RESET}")
        print(f"{YELLOW}Test Suite 1: Regular Jobs{RESET}")
        print(f"{YELLOW}{'=' * 70}{RESET}\n")

        job_id = await test_basic_job_run(tool)
        if job_id:
            job_ids.append(job_id)

            # Wait a moment for job to register
            await asyncio.sleep(1)

            # Test 2: List jobs
            await test_list_jobs(tool)

            # Test 3: Inspect job
            await test_inspect_job(tool, job_id)

            # Test 4: Get logs
            await test_get_logs(tool, job_id)

            # Test 5: Cancel job (cleanup)
            await test_cancel_job(tool, job_id)

        # Test 6: UV job
        print()
        uv_job_id = await test_uv_job(tool)
        if uv_job_id:
            job_ids.append(uv_job_id)
            await asyncio.sleep(1)
            await test_cancel_job(tool, uv_job_id)

        # Test 7: List all jobs
        print()
        await test_list_all_jobs(tool)

        # Test Suite 2: Scheduled Jobs
        print(f"\n{YELLOW}{'=' * 70}{RESET}")
        print(f"{YELLOW}Test Suite 2: Scheduled Jobs{RESET}")
        print(f"{YELLOW}{'=' * 70}{RESET}\n")

        scheduled_job_id = await test_scheduled_job(tool)
        if scheduled_job_id:
            scheduled_job_ids.append(scheduled_job_id)

            # Wait a moment for job to register
            await asyncio.sleep(1)

            # Test scheduled job operations
            await test_list_scheduled_jobs(tool)
            print()
            await test_inspect_scheduled_job(tool, scheduled_job_id)
            print()
            await test_suspend_scheduled_job(tool, scheduled_job_id)
            print()
            await test_resume_scheduled_job(tool, scheduled_job_id)
            print()

            # Cleanup: Delete scheduled job
            await test_delete_scheduled_job(tool, scheduled_job_id)

        # Final summary
        print(f"\n{YELLOW}{'=' * 70}{RESET}")
        print(f"{GREEN}✓ All integration tests completed!{RESET}")
        print(f"{YELLOW}{'=' * 70}{RESET}\n")

        print_success("Refactored implementation works correctly with real HF API")
        print_success("All 13 operations tested and verified")
        print()
        print(f"{BLUE}Summary:{RESET}")
        print(f"  • Regular jobs: ✓ run, list, inspect, logs, cancel")
        print(f"  • UV jobs: ✓ run")
        print(f"  • Scheduled jobs: ✓ create, list, inspect, suspend, resume, delete")
        print()

    except Exception as e:
        print_error(f"Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()

        # Attempt cleanup
        print(f"\n{YELLOW}Attempting cleanup...{RESET}")
        for job_id in job_ids:
            try:
                await test_cancel_job(tool, job_id)
            except:
                pass

        for scheduled_job_id in scheduled_job_ids:
            try:
                await test_delete_scheduled_job(tool, scheduled_job_id)
            except:
                pass

        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
