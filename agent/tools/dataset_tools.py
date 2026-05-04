"""
Dataset Inspection Tool - Comprehensive dataset analysis in one call

Combines /is-valid, /splits, /info, /first-rows, and /parquet endpoints
to provide everything needed for ML tasks in a single tool call.
"""

import asyncio
from typing import Any, TypedDict

import httpx

from agent.tools.types import ToolResult

BASE_URL = "https://datasets-server.huggingface.co"

# Truncation limit for long sample values in the output
MAX_SAMPLE_VALUE_LEN = 150


class SplitConfig(TypedDict):
    """Typed representation of a dataset config and its splits."""

    name: str
    splits: list[str]


def _get_headers(token: str | None = None) -> dict:
    """Get auth headers for private/gated datasets"""
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


async def inspect_dataset(
    dataset: str,
    config: str | None = None,
    split: str | None = None,
    sample_rows: int = 3,
    hf_token: str | None = None,
) -> ToolResult:
    """
    Get comprehensive dataset info in one call.
    All API calls made in parallel for speed.
    """
    headers = _get_headers(hf_token)
    output_parts = []
    errors = []

    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        # Phase 1: Parallel calls for structure info (no dependencies)
        is_valid_task = client.get(f"{BASE_URL}/is-valid", params={"dataset": dataset})
        splits_task = client.get(f"{BASE_URL}/splits", params={"dataset": dataset})
        parquet_task = client.get(f"{BASE_URL}/parquet", params={"dataset": dataset})

        results = await asyncio.gather(
            is_valid_task,
            splits_task,
            parquet_task,
            return_exceptions=True,
        )

        # Process is-valid
        if not isinstance(results[0], Exception):
            try:
                output_parts.append(_format_status(results[0].json()))
            except Exception as e:
                errors.append(f"is-valid: {e}")

        # Process splits and auto-detect config/split
        configs = []
        if not isinstance(results[1], Exception):
            try:
                splits_data = results[1].json()
                configs = _extract_configs(splits_data)
                if not config:
                    config = configs[0]["name"] if configs else "default"
                if not split:
                    split = configs[0]["splits"][0] if configs else "train"
                output_parts.append(_format_structure(configs))
            except Exception as e:
                errors.append(f"splits: {e}")

        if not config:
            config = "default"
        if not split:
            split = "train"

        # Process parquet (will be added at the end)
        parquet_section = None
        if not isinstance(results[2], Exception):
            try:
                parquet_section = _format_parquet_files(results[2].json())
            except Exception:
                pass  # Silently skip if no parquet

        # Phase 2: Parallel calls for content (depend on config/split)
        info_task = client.get(
            f"{BASE_URL}/info", params={"dataset": dataset, "config": config}
        )
        rows_task = client.get(
            f"{BASE_URL}/first-rows",
            params={"dataset": dataset, "config": config, "split": split},
            timeout=30,
        )

        content_results = await asyncio.gather(
            info_task,
            rows_task,
            return_exceptions=True,
        )

        # Process info (schema)
        if not isinstance(content_results[0], Exception):
            try:
                output_parts.append(_format_schema(content_results[0].json(), config))
            except Exception as e:
                errors.append(f"info: {e}")

        # Process sample rows
        if not isinstance(content_results[1], Exception):
            try:
                output_parts.append(
                    _format_samples(
                        content_results[1].json(), config, split, sample_rows
                    )
                )
            except Exception as e:
                errors.append(f"rows: {e}")

        # Add parquet section at the end if available
        if parquet_section:
            output_parts.append(parquet_section)

    # Combine output
    formatted = f"# {dataset}\n\n" + "\n\n".join(output_parts)
    if errors:
        formatted += f"\n\n**Warnings:** {'; '.join(errors)}"

    return {
        "formatted": formatted,
        "totalResults": 1,
        "resultsShared": 1,
        "isError": len(output_parts) == 0,
    }


def _format_status(data: dict) -> str:
    """Format /is-valid response as status line"""
    available = [
        k
        for k in ["viewer", "preview", "search", "filter", "statistics"]
        if data.get(k)
    ]
    if available:
        return f"## Status\n✓ Valid ({', '.join(available)})"
    return "## Status\n✗ Dataset may have issues"


def _extract_configs(splits_data: dict) -> list[SplitConfig]:
    """Group splits by config"""
    configs: dict[str, SplitConfig] = {}
    for s in splits_data.get("splits", []):
        cfg = s.get("config", "default")
        if cfg not in configs:
            configs[cfg] = {"name": cfg, "splits": []}
        configs[cfg]["splits"].append(s.get("split"))
    return list(configs.values())


def _format_structure(configs: list[SplitConfig], max_rows: int = 10) -> str:
    """Format configs and splits as a markdown table."""
    lines = [
        "## Structure (configs & splits)",
        "| Config | Split |",
        "|--------|-------|",
    ]

    total_splits = sum(len(cfg["splits"]) for cfg in configs)
    added_rows = 0

    for cfg in configs:
        for split_name in cfg["splits"]:
            if added_rows >= max_rows:
                break
            lines.append(f"| {cfg['name']} | {split_name} |")
            added_rows += 1
        if added_rows >= max_rows:
            break

    if total_splits > added_rows:
        lines.append(
            f"| ... | ... |  (_showing {added_rows} of {total_splits} config/split rows_) |"
        )

    return "\n".join(lines)


def _format_schema(info: dict, config: str) -> str:
    """Extract features and format as table"""
    features = info.get("dataset_info", {}).get("features", {})
    lines = [f"## Schema ({config})", "| Column | Type |", "|--------|------|"]
    for col_name, col_info in features.items():
        col_type = _get_type_str(col_info)
        lines.append(f"| {col_name} | {col_type} |")
    return "\n".join(lines)


def _get_type_str(col_info: dict) -> str:
    """Convert feature info to readable type string"""
    dtype = col_info.get("dtype") or col_info.get("_type", "unknown")
    if col_info.get("_type") == "ClassLabel":
        names = col_info.get("names", [])
        if names and len(names) <= 5:
            return f"ClassLabel ({', '.join(f'{n}={i}' for i, n in enumerate(names))})"
        return f"ClassLabel ({len(names)} classes)"
    return str(dtype)


def _format_samples(rows_data: dict, config: str, split: str, limit: int) -> str:
    """Format sample rows, truncate long values"""
    rows = rows_data.get("rows", [])[:limit]
    lines = [f"## Sample Rows ({config}/{split})"]

    messages_col_data = None

    for i, row_wrapper in enumerate(rows, 1):
        row = row_wrapper.get("row", {})
        lines.append(f"**Row {i}:**")
        for key, val in row.items():
            # Check for messages column and capture first one for format analysis
            if key.lower() == "messages" and messages_col_data is None:
                messages_col_data = val

            val_str = str(val)
            if len(val_str) > MAX_SAMPLE_VALUE_LEN:
                val_str = val_str[:MAX_SAMPLE_VALUE_LEN] + "..."
            lines.append(f"- {key}: {val_str}")

    # If we found a messages column, add format analysis
    if messages_col_data is not None:
        messages_format = _format_messages_structure(messages_col_data)
        if messages_format:
            lines.append("")
            lines.append(messages_format)

    return "\n".join(lines)


def _format_messages_structure(messages_data: Any) -> str | None:
    """
    Analyze and format the structure of a messages column.
    Common in chat/instruction datasets.
    """
    import json

    # Parse if string
    if isinstance(messages_data, str):
        try:
            messages_data = json.loads(messages_data)
        except json.JSONDecodeError:
            return None

    if not isinstance(messages_data, list) or not messages_data:
        return None

    lines = ["## Messages Column Format"]

    # Analyze message structure
    roles_seen = set()
    has_tool_calls = False
    has_tool_results = False
    message_keys = set()

    for msg in messages_data:
        if not isinstance(msg, dict):
            continue

        message_keys.update(msg.keys())

        role = msg.get("role", "")
        if role:
            roles_seen.add(role)

        if "tool_calls" in msg or "function_call" in msg:
            has_tool_calls = True
        if role in ("tool", "function") or msg.get("tool_call_id"):
            has_tool_results = True

    # Format the analysis
    lines.append(
        f"**Roles:** {', '.join(sorted(roles_seen)) if roles_seen else 'unknown'}"
    )

    # Show common message keys with presence indicators
    common_keys = [
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "function_call",
    ]
    key_status = []
    for key in common_keys:
        if key in message_keys:
            key_status.append(f"{key} ✓")
        else:
            key_status.append(f"{key} ✗")
    lines.append(f"**Message keys:** {', '.join(key_status)}")

    if has_tool_calls:
        lines.append("**Tool calls:** ✓ Present")
    if has_tool_results:
        lines.append("**Tool results:** ✓ Present")

    # Show example message structure
    # Priority: 1) message with tool_calls, 2) first assistant message, 3) first non-system message
    example = None
    fallback = None
    for msg in messages_data:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        # Check for actual tool_calls/function_call values (not None)
        if msg.get("tool_calls") or msg.get("function_call"):
            example = msg
            break
        if role == "assistant" and example is None:
            example = msg
        elif role != "system" and fallback is None:
            fallback = msg
    if example is None:
        example = fallback

    if example:
        lines.append("")
        lines.append("**Example message structure:**")
        # Build a copy with truncated content but keep all keys
        example_clean = {}
        for key, val in example.items():
            if key == "content" and isinstance(val, str) and len(val) > 100:
                example_clean[key] = val[:100] + "..."
            else:
                example_clean[key] = val
        lines.append("```json")
        lines.append(json.dumps(example_clean, indent=2, ensure_ascii=False))
        lines.append("```")

    return "\n".join(lines)


def _format_parquet_files(data: dict, max_rows: int = 10) -> str | None:
    """Format parquet file info, return None if no files."""
    files = data.get("parquet_files", [])
    if not files:
        return None

    # Group by config/split
    groups: dict[str, dict] = {}
    for f in files:
        key = f"{f.get('config', 'default')}/{f.get('split', 'train')}"
        if key not in groups:
            groups[key] = {"count": 0, "size": 0}
        size = f.get("size") or 0
        if not isinstance(size, (int, float)):
            size = 0
        groups[key]["count"] += 1
        groups[key]["size"] += int(size)

    lines = ["## Files (Parquet)"]
    items = list(groups.items())
    total_groups = len(items)

    shown = 0
    for key, info in items[:max_rows]:
        size_mb = info["size"] / (1024 * 1024)
        lines.append(f"- {key}: {info['count']} file(s) ({size_mb:.1f} MB)")
        shown += 1

    if total_groups > shown:
        lines.append(f"- ... (_showing {shown} of {total_groups} parquet groups_)")
    return "\n".join(lines)


# Tool specification
HF_INSPECT_DATASET_TOOL_SPEC = {
    "name": "hf_inspect_dataset",
    "description": (
        "Inspect a HF dataset in one call: status, configs/splits, schema, sample rows, parquet info.\n\n"
        "REQUIRED before any training job to verify dataset format matches training method:\n"
        "  SFT: needs 'messages', 'text', or 'prompt'/'completion'\n"
        "  DPO: needs 'prompt', 'chosen', 'rejected'\n"
        "  GRPO: needs 'prompt'\n"
        "All datasets used for training have to be in conversational ChatML format to be compatible with HF libraries.'\n"
        "Training will fail with KeyError if columns don't match.\n\n"
        "Also use to get example datapoints, understand column names, data types, and available splits before writing any data loading code. "
        "Supports private/gated datasets when HF_TOKEN is set."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "description": "Dataset ID in 'org/name' format (e.g., 'stanfordnlp/imdb')",
            },
            "config": {
                "type": "string",
                "description": "Config/subset name. Auto-detected if not specified.",
            },
            "split": {
                "type": "string",
                "description": "Split for sample rows. Auto-detected if not specified.",
            },
            "sample_rows": {
                "type": "integer",
                "description": "Number of sample rows to show (default: 3, max: 10)",
                "default": 3,
            },
        },
        "required": ["dataset"],
    },
}


async def hf_inspect_dataset_handler(
    arguments: dict[str, Any], session=None
) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        hf_token = session.hf_token if session else None
        result = await inspect_dataset(
            dataset=arguments["dataset"],
            config=arguments.get("config"),
            split=arguments.get("split"),
            sample_rows=min(arguments.get("sample_rows", 3), 10),
            hf_token=hf_token,
        )
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error inspecting dataset: {str(e)}", False
