from typing import Any, Dict, List

from agent.core.session import Event
from agent.utils.terminal_display import format_plan_tool_output

from .types import ToolResult

# Per-session plan storage to avoid cross-session corruption
_session_plans: Dict[str, List[Dict[str, str]]] = {}
_last_plan_session_id: str | None = None


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
        global _session_plans, _last_plan_session_id

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

        # Store per-session to prevent cross-session plan corruption
        session_id = self.session.session_id if self.session else "__no_session__"
        _session_plans[session_id] = todos
        _last_plan_session_id = session_id

        # Emit plan update event if session is available
        if self.session:
            await self.session.send_event(
                Event(
                    event_type="plan_update",
                    data={"plan": todos},
                )
            )

        # Format only for display using terminal_display utility
        formatted_output = format_plan_tool_output(todos)

        return {
            "formatted": formatted_output,
            "totalResults": len(todos),
            "isError": False,
        }


def get_current_plan(session_id: str | None = None) -> List[Dict[str, str]]:
    """Get the current plan for a session (raw structure)."""
    if session_id:
        return _session_plans.get(session_id, [])
    if _last_plan_session_id:
        return _session_plans.get(_last_plan_session_id, [])
    return []


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
