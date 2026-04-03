"""
Research subagent tool — spawns a cheap LLM call with a focused
research task and returns a summary. The subagent gets its own
independent context (not the main conversation), so research
work doesn't pollute the main agent's context window.

Inspired by claude-code's code-explorer agent pattern.
"""

import json
import logging
import os
from typing import Any

from litellm import Message, acompletion

from agent.core.session import Event

logger = logging.getLogger(__name__)

# Tools the research agent can use (read-only subset)
RESEARCH_TOOL_NAMES = {
    "read",
    "bash",
    "explore_hf_docs",
    "fetch_hf_docs",
    "find_hf_api",
    "hf_papers",
    "github_find_examples",
    "github_list_repos",
    "github_read_file",
    "hf_inspect_dataset",
    "hf_repo_files",
}

RESEARCH_SYSTEM_PROMPT = """\
You are a research sub-agent for an ML engineering assistant.
Your job: explore documentation, code examples, APIs, and repos,
then return a concise, actionable summary. The main agent will use
your findings to implement the actual solution.

# Research methodology

1. **Discovery**: Find relevant entry points — example scripts, doc pages, API endpoints
2. **Tracing**: Follow the chain from entry point to implementation detail
3. **Analysis**: Identify patterns, current API usage, key dependencies
4. **Synthesis**: Summarize findings in a structured format

# How to use your tools

## GitHub code research (USE FIRST for any ML implementation task)
- `github_find_examples`: Find working example scripts in HF repos (trl, transformers, etc.)
  Example: `github_find_examples({"repo": "trl", "keyword": "sft"})`
  Returns: file paths in examples/, scripts/, notebooks/ directories
- `github_read_file`: Read the actual implementation code
  Example: `github_read_file({"repo": "huggingface/trl", "path": "examples/scripts/sft.py"})`
  Use line_start/line_end for large files

## Documentation
- `explore_hf_docs(endpoint)`: Search docs for a library. Endpoints: trl, transformers, datasets, peft, accelerate, trackio, vllm, inference-endpoints, etc.
- `fetch_hf_docs(url)`: Fetch full page content from explore results
- `find_hf_api(query=..., tag=...)`: Find REST API endpoints

## Dataset inspection
- `hf_inspect_dataset`: Check dataset schema, splits, sample rows
  CRITICAL for training: verify column format matches training method:
  - SFT: needs "messages", "text", or "prompt"/"completion"
  - DPO: needs "prompt", "chosen", "rejected"
  - GRPO: needs "prompt" only

## Papers
- `hf_papers`: Search papers, get details, find linked datasets/models

## Hub repo inspection
- `hf_repo_files`: List/read files in any HF repo (model, dataset, space)

# Correct research pattern for ML tasks

```
# 1. Find working example code FIRST
github_find_examples({"repo": "trl", "keyword": "sft"})

# 2. Read the implementation
github_read_file({"repo": "huggingface/trl", "path": "examples/scripts/sft.py"})

# 3. Check docs for parameters/config details
explore_hf_docs("trl")
fetch_hf_docs("https://huggingface.co/docs/trl/sft_trainer")

# 4. Validate dataset format if relevant
hf_inspect_dataset({"dataset": "org/name", "split": "train", "sample_rows": 3})
```

# Output format

Your output MUST include:
- **Key findings**: The most important things you discovered (current API usage, working patterns)
- **Essential references**: Specific file paths, URLs, function names, doc sections, code snippets
  that the main agent should use directly
- **Code patterns**: Key imports, configurations, and usage patterns from working examples
- **Recommendations**: What to do next based on your findings

Be concise. Your output goes into another agent's context — every token counts.
Aim for 500-1500 words max. Include actual code snippets from examples you read,
not paraphrased descriptions.
"""

RESEARCH_TOOL_SPEC = {
    "name": "research",
    "description": (
        "Spawn a research sub-agent to explore documentation, codebases, "
        "or repos WITHOUT polluting the main conversation context. "
        "The sub-agent gets its own independent context window with read-only "
        "research tools and returns a concise summary of findings.\n\n"
        "Use this for:\n"
        "- Researching current API usage before implementing ML tasks "
        "(find examples + read docs)\n"
        "- Exploring HF docs, reading papers, analyzing GitHub repos\n"
        "- Any research where raw tool outputs would be too verbose\n\n"
        "The sub-agent knows how to use github_find_examples, github_read_file, "
        "explore_hf_docs, fetch_hf_docs, hf_inspect_dataset, hf_papers, etc. "
        "Just describe what you need researched."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Detailed description of what to research. Be specific: "
                    "include library names, trainer types, dataset names, "
                    "repo names, or doc pages to explore. Example: "
                    "'Research current TRL SFTTrainer usage: find working "
                    "example scripts, read the SFT documentation, and check "
                    "SFTConfig parameters. Also validate that dataset "
                    "HuggingFaceH4/ultrachat_200k has the right format for SFT.'"
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional context from the current conversation that the "
                    "research agent needs (e.g., what the user wants to build, "
                    "constraints, what's been tried)."
                ),
            },
        },
        "required": ["task"],
    },
}


def _resolve_llm_params(model_name: str) -> dict:
    """Build LiteLLM kwargs, reusing the HF router logic from agent_loop."""
    if not model_name.startswith("huggingface/"):
        return {"model": model_name}

    parts = model_name.split("/", 2)  # ["huggingface", "<provider>", "<org>/<model>"]
    if len(parts) < 3:
        return {"model": model_name}

    provider = parts[1]
    model_id = parts[2]
    return {
        "model": f"openai/{model_id}",
        "api_base": f"https://router.huggingface.co/{provider}/v3/openai",
        "api_key": os.environ.get("INFERENCE_TOKEN", ""),
    }


def _get_research_model(main_model: str) -> str:
    """Pick a cheaper model for research based on the main model."""
    if "opus" in main_model:
        return "anthropic/claude-sonnet-4-5-20250929"
    if "sonnet" in main_model:
        return "anthropic/claude-haiku-3-5-20241022"
    # For HF router models, use the same model
    return main_model


async def research_handler(
    arguments: dict[str, Any], session=None, **_kw
) -> tuple[str, bool]:
    """Execute a research sub-agent with its own context."""
    task = arguments.get("task", "")
    context = arguments.get("context", "")
    if not task:
        return "No research task provided.", False

    if not session:
        return "No session available for research agent.", False

    # Build the sub-agent's messages (independent context)
    messages: list[Message] = [
        Message(role="system", content=RESEARCH_SYSTEM_PROMPT),
    ]

    user_content = f"Research task: {task}"
    if context:
        user_content = f"Context: {context}\n\n{user_content}"
    messages.append(Message(role="user", content=user_content))

    # Use a cheaper/faster model for research
    main_model = session.config.model_name
    research_model = _get_research_model(main_model)
    llm_params = _resolve_llm_params(research_model)

    # Get read-only tool specs from the session's tool router
    tool_specs = [
        spec
        for spec in session.tool_router.get_tool_specs_for_llm()
        if spec["function"]["name"] in RESEARCH_TOOL_NAMES
    ]

    async def _log(text: str) -> None:
        """Send a progress event to the UI so it doesn't look frozen."""
        try:
            await session.send_event(
                Event(event_type="tool_log", data={"tool": "research", "log": text})
            )
        except Exception:
            pass

    await _log("Starting research sub-agent...")

    # Run the research loop (max 20 iterations — research should be focused)
    max_iterations = 20
    for _iteration in range(max_iterations):
        try:
            response = await acompletion(
                messages=messages,
                tools=tool_specs if tool_specs else None,
                tool_choice="auto",
                stream=False,
                timeout=120,
                **llm_params,
            )
        except Exception as e:
            logger.error("Research sub-agent LLM error: %s", e)
            return f"Research agent LLM error: {e}", False

        choice = response.choices[0]
        msg = choice.message

        # If no tool calls, we have our final answer
        if not msg.tool_calls:
            await _log("Research complete.")
            content = msg.content or "Research completed but no summary generated."
            return content, True

        # Execute tool calls and add results
        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                messages.append(
                    Message(
                        role="tool",
                        content="Invalid tool arguments.",
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    )
                )
                continue

            tool_name = tc.function.name
            if tool_name not in RESEARCH_TOOL_NAMES:
                messages.append(
                    Message(
                        role="tool",
                        content=f"Tool '{tool_name}' not available for research.",
                        tool_call_id=tc.id,
                        name=tool_name,
                    )
                )
                continue

            try:
                # Brief description for the UI
                brief = tool_name
                if tool_name == "github_find_examples":
                    brief = f"github_find_examples({tool_args.get('repo', '')}/{tool_args.get('keyword', '')})"
                elif tool_name == "github_read_file":
                    brief = f"github_read_file({tool_args.get('path', '')})"
                elif tool_name == "explore_hf_docs":
                    brief = f"explore_hf_docs({tool_args.get('endpoint', '')})"
                elif tool_name == "fetch_hf_docs":
                    url = tool_args.get("url", "")
                    brief = f"fetch_hf_docs(...{url[-50:]})" if len(url) > 50 else f"fetch_hf_docs({url})"
                elif tool_name == "hf_inspect_dataset":
                    brief = f"hf_inspect_dataset({tool_args.get('dataset', '')})"
                await _log(brief)

                output, _success = await session.tool_router.call_tool(
                    tool_name, tool_args, session=session
                )
                # Log a short preview of the output
                preview = output.replace("\n", " ").strip()[:20]
                if len(output.strip()) > 20:
                    preview += "…"
                await _log(f"  → {preview}")
                # Truncate tool output for the research context
                if len(output) > 8000:
                    output = (
                        output[:4800]
                        + "\n...(truncated)...\n"
                        + output[-3200:]
                    )
            except Exception as e:
                output = f"Tool error: {e}"

            messages.append(
                Message(
                    role="tool",
                    content=output,
                    tool_call_id=tc.id,
                    name=tool_name,
                )
            )

    return (
        "Research agent hit iteration limit (20). "
        "Partial findings may be incomplete — try a more focused task.",
        False,
    )
