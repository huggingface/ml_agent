"""
Utils Tools - General utility operations

Provides system information like current date/time with timezone support.
"""

import zoneinfo
from datetime import datetime
from typing import Any, Dict, Literal

from agent.tools.types import ToolResult

# Operation names
OperationType = Literal["get_datetime"]


class UtilsTool:
    """Tool for general utility operations."""

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the specified utility operation."""
        operation = params.get("operation")
        args = params.get("args", {})

        # If no operation provided, return usage instructions
        if not operation:
            return self._show_help()

        # Normalize operation name
        operation = operation.lower()

        # Check if help is requested
        if args.get("help"):
            return self._show_operation_help(operation)

        try:
            # Route to appropriate handler
            if operation == "get_datetime":
                return await self._get_datetime(args)
            else:
                return {
                    "formatted": f'Unknown operation: "{operation}"\n\n'
                    "Available operations: get_datetime\n\n"
                    "Call this tool with no operation for full usage instructions.",
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

    def _show_help(self) -> ToolResult:
        """Show usage instructions when tool is called with no arguments."""
        usage_text = """# Utils Tool

Utility operations for system information.

## Available Commands

- **get_datetime** - Get current date and time with timezone support

## Examples

### Get current date and time (Paris timezone by default)
Call this tool with:
```json
{
  "operation": "get_datetime",
  "args": {}
}
```

### Get current date and time in a specific timezone
Call this tool with:
```json
{
  "operation": "get_datetime",
  "args": {
    "timezone": "America/New_York"
  }
}
```

Common timezones: Europe/Paris, America/New_York, America/Los_Angeles, Asia/Tokyo, UTC

## Tips

- **Default timezone**: Paris (Europe/Paris)
- **Date format**: dd-mm-yyyy
- **Time format**: HH:MM:SS.mmm (24-hour format with milliseconds)
- **Timezone names**: Use IANA timezone database names (e.g., "Europe/Paris", "UTC")
"""
        return {"formatted": usage_text, "totalResults": 1, "resultsShared": 1}

    def _show_operation_help(self, operation: str) -> ToolResult:
        """Show help for a specific operation."""
        help_text = f"Help for operation: {operation}\n\nCall with appropriate arguments. Use the main help for examples."
        return {"formatted": help_text, "totalResults": 1, "resultsShared": 1}

    async def _get_datetime(self, args: Dict[str, Any]) -> ToolResult:
        """Get current date and time with timezone support."""
        timezone_name = args.get("timezone", "Europe/Paris")

        try:
            # Get timezone object
            tz = zoneinfo.ZoneInfo(timezone_name)

            # Get current datetime in specified timezone
            now = datetime.now(tz)

            # Format date as dd-mm-yyyy
            date_str = now.strftime("%d-%m-%Y")

            # Format time as HH:MM:SS.mmm
            time_str = now.strftime("%H:%M:%S.%f")[
                :-3
            ]  # Remove last 3 digits to keep only milliseconds

            # Get timezone abbreviation/offset
            tz_offset = now.strftime("%z")
            tz_name = now.strftime("%Z")

            response = f"""✓ Current date and time

**Date:** {date_str}
**Time:** {time_str}
**Timezone:** {timezone_name} ({tz_name}, UTC{tz_offset[:3]}:{tz_offset[3:]})

**ISO Format:** {now.isoformat()}
**Unix Timestamp:** {int(now.timestamp())}"""

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except zoneinfo.ZoneInfoNotFoundError:
            return {
                "formatted": f"Invalid timezone: {timezone_name}\n\n"
                "Use IANA timezone database names like:\n"
                "- Europe/Paris\n"
                "- America/New_York\n"
                "- Asia/Tokyo\n"
                "- UTC\n\n"
                "See: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }
        except Exception as e:
            return {
                "formatted": f"Failed to get date/time: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }


# Tool specification for agent registration
UTILS_TOOL_SPEC = {
    "name": "utils",
    "description": (
        "Utility operations for system information. "
        "Get current date (dd-mm-yyyy) and time (HH:MM:SS.mmm) with timezone support. "
        "Default timezone: Paris (Europe/Paris). "
        "Call with no operation for full usage instructions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["get_datetime"],
                "description": "Operation to execute. Valid values: [get_datetime]",
            },
            "args": {
                "type": "object",
                "description": (
                    "Operation-specific arguments as a JSON object. "
                    "For get_datetime: timezone (string, optional, default: Europe/Paris). "
                    "Use IANA timezone names like 'America/New_York', 'Asia/Tokyo', 'UTC'."
                ),
                "additionalProperties": True,
            },
        },
    },
}


async def utils_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router."""
    try:
        tool = UtilsTool()
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error executing Utils tool: {str(e)}", False
