"""
Types for Hugging Face tools

Ported from: hf-mcp-server/packages/mcp/src/types/
"""

from typing import TypedDict


class ToolResult(TypedDict, total=False):
    """Result returned by HF tool operations"""

    formatted: str
    totalResults: int
    resultsShared: int
    isError: bool
