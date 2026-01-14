"""
Test script for hf_repo_files and hf_repo_git tools
"""

import asyncio
import sys
from typing import TypedDict
from unittest.mock import MagicMock


# Mock the types module before importing
class ToolResult(TypedDict, total=False):
    formatted: str
    totalResults: int
    resultsShared: int
    isError: bool


mock_types = MagicMock()
mock_types.ToolResult = ToolResult
sys.modules["agent.tools.types"] = mock_types

from agent.tools.hf_repo_files_tool import HfRepoFilesTool
from agent.tools.hf_repo_git_tool import HfRepoGitTool


async def test_hf_repo_files():
    """Test hf_repo_files tool"""
    print("=" * 60)
    print("Testing hf_repo_files")
    print("=" * 60)

    tool = HfRepoFilesTool()

    # Test list
    print("\n→ list files in gpt2:")
    result = await tool.execute(
        {"operation": "list", "repo_id": "openai-community/gpt2"}
    )
    print(f"   isError: {result.get('isError', False)}")
    print(f"   totalResults: {result['totalResults']}")
    # Just show first few lines
    lines = result["formatted"].split("\n")
    print("   Output (first 5 lines):\n" + "\n".join(f"   {line}" for line in lines))

    # Test read
    print("\n→ read config.json from gpt2:")
    result = await tool.execute(
        {"operation": "read", "repo_id": "openai-community/gpt2", "path": "config.json"}
    )
    print(f"   isError: {result.get('isError', False)}")
    lines = result["formatted"].split("\n")
    print("   Output (first 10 lines):\n" + "\n".join(f"   {line}" for line in lines))


async def test_hf_repo_git():
    """Test hf_repo_git tool"""
    print("\n" + "=" * 60)
    print("Testing hf_repo_git")
    print("=" * 60)

    tool = HfRepoGitTool()

    # Test list_refs
    print("\n→ list_refs for gpt2:")
    result = await tool.execute(
        {"operation": "list_refs", "repo_id": "openai-community/gpt2"}
    )
    print(f"   isError: {result.get('isError', False)}")
    print(
        "   Output:\n"
        + "\n".join(f"   {line}" for line in result["formatted"].split("\n"))
    )

    # Test help (no operation)
    print("\n→ help (no operation):")
    result = await tool.execute({})
    print(f"   isError: {result.get('isError', False)}")
    lines = result["formatted"].split("\n")[:6]
    print("   Output (first 6 lines):\n" + "\n".join(f"   {line}" for line in lines))


if __name__ == "__main__":
    print("\nHF Repo Tools Test\n")
    asyncio.run(test_hf_repo_files())
    asyncio.run(test_hf_repo_git())
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)
