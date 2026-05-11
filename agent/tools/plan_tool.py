from typing import Any, Dict, List

from agent.core.session import Event
from agent.utils.terminal_display import format_plan_tool_output

from .types import ToolResult

# In-memory storage for the current plan (raw structure from agent)
_current_plan: List[Dict[str, str]] = []


class PlanTool:
    """Tool for managing a list of todos with status tracking."""

    def __init__(self, session: Any = None):
        self.session = session

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

        # Store a session-scoped copy so the runtime can tell whether a
        # text-only model response is trying to stop while work remains.
        stored_todos = [dict(todo) for todo in todos]
        _current_plan = stored_todos
        if self.session is not None:
            self.session.current_plan = stored_todos

        # Emit plan update event if session is available
        if self.session:
            await self.session.send_event(
                Event(
                    event_type="plan_update",
                    data={"plan": stored_todos},
                )
            )

        # Format only for display using terminal_display utility
        formatted_output = format_plan_tool_output(stored_todos)

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
        "Track progress on multi-step tasks with a todo list (pending/in_progress/completed).\n\n"
        "Use for tasks with 3+ steps. Each call replaces the entire plan (send full list).\n\n"
        "Rules: exactly ONE task in_progress at a time. Mark completed immediately after finishing. "
        "Only mark completed when the task fully succeeded — keep in_progress if there are errors. "
        "Update frequently so the user sees progress."
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


async def plan_tool_handler(
    arguments: Dict[str, Any], session: Any = None
) -> tuple[str, bool]:
    tool = PlanTool(session=session)
    result = await tool.execute(arguments)
    return result["formatted"], not result.get("isError", False)
