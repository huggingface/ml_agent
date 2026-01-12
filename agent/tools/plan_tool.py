from typing import Any, Dict, List

from agent.utils.terminal_display import format_plan_tool_output

from .types import ToolResult

# In-memory storage for the current plan (raw structure from agent)
_current_plan: List[Dict[str, str]] = []


class PlanTool:
    """Tool for managing a list of todos with status tracking."""

    def __init__(self):
        pass

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """
        Execute the WritePlan operation.

        Args:
            params: Dictionary containing:
                - todos: List of todo items, each with id, content, and status

        Returns:
            ToolResult with formatted output
        """
        global _current_plan

        todos = params.get("todos", [])

        # Validate todos structure
        for todo in todos:
            if not isinstance(todo, dict):
                return {
                    "formatted": "Error: Each todo must be an object. Re call the tool with correct format (mandatory).",
                    "isError": True,
                }

            required_fields = ["id", "content", "status"]
            for field in required_fields:
                if field not in todo:
                    return {
                        "formatted": f"Error: Todo missing required field '{field}'. Re call the tool with correct format (mandatory).",
                        "isError": True,
                    }

            # Validate status
            valid_statuses = ["pending", "in_progress", "completed"]
            if todo["status"] not in valid_statuses:
                return {
                    "formatted": f"Error: Invalid status '{todo['status']}'. Must be one of: {', '.join(valid_statuses)}. Re call the tool with correct format (mandatory).",
                    "isError": True,
                }

        # Store the raw todos structure in memory
        _current_plan = todos

        # Format only for display using terminal_display utility
        formatted_output = format_plan_tool_output(todos)

        return {
            "formatted": formatted_output,
            "totalResults": len(todos),
            "isError": False,
        }


def get_current_plan() -> List[Dict[str, str]]:
    """Get the current plan (raw structure)."""
    return _current_plan


# Tool specification
PLAN_TOOL_SPEC = {
    "name": "plan_tool",
    "description": (
        "Manage task planning and progress tracking with todo list (pending/in_progress/completed statuses). "
        "⚠️ CRITICAL: ALWAYS use for multi-step tasks (3+ steps) and MUST update frequently to show progress. "
        "**Use when:** (1) User provides multiple tasks, (2) Complex workflows (training, evaluation, data processing), "
        "(3) Tasks requiring multiple tool calls, (4) Need to communicate progress clearly to user, "
        "(5) Breaking down ambiguous requests into concrete steps. "
        "**Pattern:** Create plan at start → Mark in_progress when starting task → Mark completed immediately after finishing → User sees clear progress. "
        "Each call replaces entire plan (full list required). "
        "**Critical for reliability:** Exactly ONE task in_progress at a time (not zero, not multiple). "
        "Mark tasks completed IMMEDIATELY after finishing - don't batch completions. "
        "**For long-running tasks:** Update plan after each major step to keep user informed. "
        "**Only mark completed when:** Task fully accomplished, no errors, all requirements met. "
        "Keep tasks pending if blocked/errors occur - create new task to resolve blockers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of todo items",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for the todo",
                        },
                        "content": {
                            "type": "string",
                            "description": "Description of the todo task",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Current status of the todo",
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
}


async def plan_tool_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    tool = PlanTool()
    result = await tool.execute(arguments)
    return result["formatted"], not result.get("isError", False)
