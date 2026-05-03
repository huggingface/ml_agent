"""Shared predicates for approval-gated tool operations."""

from typing import Any


def normalize_tool_operation(operation: Any) -> str:
    return str(operation or "").strip().lower()


def is_scheduled_operation(operation: Any) -> bool:
    return normalize_tool_operation(operation).startswith("scheduled ")
