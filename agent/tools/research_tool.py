"""
Research subagent tool — spawns a cheap LLM call with a focused
research task and returns a summary. The subagent gets its own
independent context (not the main conversation), so research
work doesn't pollute the main agent's context window.

Inspired by claude-code's code-explorer agent pattern.
"""

import json
import logging
from typing import Any

from litellm import Message, acompletion

from agent.core import rate_limiter
from agent.core.doom_loop import check_for_doom_loop
from agent.core.llm_params import _resolve_llm_params
from agent.core.prompt_caching import with_prompt_caching
from agent.core.session import Event

logger = logging.getLogger(__name__)

# Context budget for the research subagent (tokens).
# When usage exceeds WARN threshold, the subagent is told to wrap up.
# At MAX, the loop is force-stopped and whatever content exists is returned.
_RESEARCH_CONTEXT_WARN = 170_000  # 85% of 200k
_RESEARCH_CONTEXT_MAX = 190_000

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
Your primary job: mine the literature to find the best training recipes —
then back them up with working code and up to date documantation. The main agent will use
your findings to implement the actual solution.

# Start from the literature

Your default approach is a deep literature crawl. Do not start from docs or
example scripts — start from papers. Papers contain the results, and results
tell you what actually works.

## The crawl

1. **Find anchor papers**: Search for the task/domain. Identify the landmark paper(s) — high citations, recent, or both.
2. **Crawl the citation graph**: Use `citation_graph` on the anchor paper(s). Look DOWNSTREAM (papers that cite it) — these are the ones that built on it, improved it, or applied it to new domains. Prioritize recent papers and papers with many citations.
3. **Read methodology sections**: For the most promising papers (strong results, recent, relevant), use `read_paper` with section parameter to read sections 3, 4, 5 (Methodology, Experiments, Results — not the abstract). Extract:
   - The exact dataset(s) used (name, source, size, any filtering/preprocessing)
   - The training method and configuration (optimizer, lr, schedule, epochs, batch size)
   - The results those choices produced (benchmark scores, metrics, comparisons)
4. **Attribute results to recipes**: This is the critical step. Every finding must link a RESULT to the RECIPE that produced it. "Dataset X + method Y + lr Z → score W on benchmark V" is useful. "They used SFT" is not.
5. **Validate datasets**: For the most promising datasets, check if they exist on HF Hub with `hf_inspect_dataset`. Verify format matches the training method. Report if doesnt.
6. **Find code**: Now find working implementation code via `github_find_examples` and `github_read_file`. Use docs (`explore_hf_docs`, `fetch_hf_docs`) to fill in API details.

## When to go deeper

- If the anchor paper is old (>1 year), its citation graph is your main source — the downstream papers will have better methods.
- If a downstream paper reports significantly better results, crawl ITS citation graph too.
- Use `snippet_search` to find specific claims across papers (e.g., "does dataset X consistently outperform Y for this task?").
- Use `recommend` to find related papers the citation graph might miss.

# How to use your tools

## Papers & citations (USE FIRST)
- `hf_papers(operation="search", query=...)`: Search papers (HF-tuned for ML)
- `hf_papers(operation="search", query=..., min_citations=50, sort_by="citationCount")`: Find highly-cited papers via Semantic Scholar
- `hf_papers(operation="search", query=..., date_from="2024-01-01")`: Search with date filter
- `hf_papers(operation="paper_details", arxiv_id=...)`: Metadata, citations, TL;DR
- `hf_papers(operation="citation_graph", arxiv_id=...)`: References + citations with influence flags and intents
- `hf_papers(operation="read_paper", arxiv_id=..., section="3")`: Read a specific section's full text
- `hf_papers(operation="read_paper", arxiv_id=...)`: Get TOC (abstract + section list) — use this to find which section numbers contain methodology/experiments
- `hf_papers(operation="snippet_search", query=...)`: Semantic search across 12M+ full-text paper passages
- `hf_papers(operation="recommend", arxiv_id=...)`: Find related papers
- `hf_papers(operation="find_datasets", arxiv_id=...)`: Find HF datasets linked to a paper
- `hf_papers(operation="find_all_resources", arxiv_id=...)`: Datasets + models + collections for a paper

## Dataset inspection
- `hf_inspect_dataset`: Check dataset schema, splits, sample rows
  CRITICAL for training: verify column format matches training method:
  - SFT: needs "messages", "text", or "prompt"/"completion"
  - DPO: needs "prompt", "chosen", "rejected"
  - GRPO: needs "prompt" only

## GitHub code research
- `github_find_examples`: Find working example scripts in HF repos (trl, transformers, etc.)
- `github_read_file`: Read the actual implementation code. Use line_start/line_end for large files.

## Documentation
- `explore_hf_docs(endpoint)`: Search docs for a library. Endpoints: trl, transformers, datasets, peft, accelerate, trackio, vllm, inference-endpoints, etc.
- `fetch_hf_docs(url)`: Fetch full page content from explore results
- `find_hf_api(query=..., tag=...)`: Find REST API endpoints

## Hub repo inspection
- `hf_repo_files`: List/read files in any HF repo (model, dataset, space)

# Correct research pattern

```
# 1. Find anchor paper(s) for the task
hf_papers({"operation": "search", "query": "GPQA graduate questions", "sort_by": "citationCount"})

# 2. Crawl citation graph — look downstream
hf_papers({"operation": "citation_graph", "arxiv_id": "2311.12022", "direction": "citations"})

# 3. Read methodology of promising downstream papers
hf_papers({"operation": "read_paper", "arxiv_id": "2604.01348"})  # TOC first
hf_papers({"operation": "read_paper", "arxiv_id": "2604.01348", "section": "3"})  # Methodology
hf_papers({"operation": "read_paper", "arxiv_id": "2604.01348", "section": "4"})  # Experiments

# 4. Find datasets used by these papers
hf_papers({"operation": "find_datasets", "arxiv_id": "2604.01348"})
hf_papers({"operation": "find_all_resources", "arxiv_id": "2604.01348"})

# 5. Validate datasets exist and have correct format
hf_inspect_dataset({"dataset": "org/dataset-name", "split": "train", "sample_rows": 3})

# 6. Now get working code for the training method
github_find_examples({"repo": "trl", "keyword": "sft"})
github_read_file({"repo": "huggingface/trl", "path": "examples/scripts/sft.py"})
explore_hf_docs("trl")
```

# Output format



Your output MUST be structured as a ranked list of training recipes, each attributed to published results:

## Recipe table (REQUIRED)
For each promising approach found, report:
- **Paper**: title, arxiv_id, date, venue
- **Result**: exact benchmark scores and what they were measured on
- **Dataset(s)**: name, size, source, HF Hub availability, format verified (yes/no)
- **Method**: training approach, key hyperparameters (lr, epochs, batch size, optimizer, schedule)
- **What made it work**: the specific insight or trick that drove the result (data curation, curriculum, loss function, etc.)

Rank recipes by result quality. The main agent will pick the best one that's feasible.

## Code patterns
- Key imports, configurations, and usage patterns from working examples
- Specific file paths, URLs, function names from docs

## Recommendations
- Which recipe to implement first and why
- What datasets to use (with HF Hub paths, verified)
- Any gaps: datasets that need preprocessing, methods that need adaptation

Additionally include:
- **SOTA landscape**: Current best models, datasets, and methods for the task (from recent papers). Flag anything outdated.
- **Essential references**: Specific file paths, URLs, function names, doc sections, code snippets
  that the main agent should use directly
- **Code patterns**: Key imports, configurations, and usage patterns from working examples

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


def _get_research_model(main_model: str) -> str:
    """Pick a cheaper model for research based on the main model."""
    if "anthropic" in main_model:
        return "bedrock/us.anthropic.claude-sonnet-4-6"
    # For non-Anthropic models (HF router etc.), use the same model
    return main_model


async def research_handler(
    arguments: dict[str, Any], session=None, tool_call_id: str | None = None, **_kw
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
    # Research is a cheap sub-call — cap the main session's effort at "high"
    # so a user preference of ``max``/``xhigh`` (valid for Opus 4.6/4.7) doesn't
    # propagate to a Sonnet research model that may not accept those levels.
    # We also haven't probed this sub-model so we don't know its ceiling.
    _pref = getattr(session.config, "reasoning_effort", None)
    _capped = "high" if _pref in ("max", "xhigh") else _pref
    llm_params = _resolve_llm_params(
        research_model,
        getattr(session, "hf_token", None),
        reasoning_effort=_capped,
    )

    # Get read-only tool specs from the session's tool router
    tool_specs = [
        spec
        for spec in session.tool_router.get_tool_specs_for_llm()
        if spec["function"]["name"] in RESEARCH_TOOL_NAMES
    ]

    # Unique ID + short label so parallel agents show separate status lines.
    # Use the tool_call_id when available — it's unique per invocation and lets
    # the frontend match a research tool card to its agent state. Fall back to
    # uuid for offline/test paths. Previously used md5(task), which collided
    # when the same task string was researched in parallel.
    if tool_call_id:
        _agent_id = tool_call_id
    else:
        import uuid
        _agent_id = uuid.uuid4().hex[:8]
    _agent_label = "research: " + (task[:50] + "…" if len(task) > 50 else task)

    async def _log(text: str) -> None:
        """Send a progress event to the UI so it doesn't look frozen."""
        try:
            await session.send_event(
                Event(event_type="tool_log", data={
                    "tool": "research",
                    "log": text,
                    "agent_id": _agent_id,
                    "label": _agent_label,
                })
            )
        except Exception:
            pass

    _tool_uses = 0
    _total_tokens = 0
    _warned_context = False

    await _log("Starting research sub-agent...")

    # Run the research loop — context budget is the real limiter
    max_iterations = 60
    for _iteration in range(max_iterations):
        # ── Doom-loop detection ──
        doom_prompt = check_for_doom_loop(messages)
        if doom_prompt:
            logger.warning("Research sub-agent doom loop detected at iteration %d", _iteration)
            await _log("Doom loop detected — injecting corrective prompt")
            messages.append(Message(role="user", content=doom_prompt))

        # ── Context budget: warn at 75%, hard-stop at 95% ──
        if _total_tokens >= _RESEARCH_CONTEXT_MAX:
            logger.warning(
                "Research sub-agent hit context max (%d tokens) — forcing summary",
                _total_tokens,
            )
            await _log(f"Context limit reached ({_total_tokens} tokens) — forcing wrap-up")
            # Ask for a final summary with no tools
            messages.append(Message(
                role="user",
                content=(
                    "[SYSTEM: CONTEXT LIMIT REACHED] You have used all available context. "
                    "Summarize your findings NOW. Do NOT call any more tools."
                ),
            ))
            try:
                _msgs, _ = with_prompt_caching(messages, None, llm_params.get("model"))
                await rate_limiter.acquire(research_model)
                response = await acompletion(
                    messages=_msgs,
                    tools=None,  # no tools — force text response
                    stream=False,
                    timeout=120,
                    **llm_params,
                )
                content = response.choices[0].message.content or ""
                return content or "Research context exhausted — no summary produced.", bool(content)
            except Exception:
                return "Research context exhausted and summary call failed.", False

        if not _warned_context and _total_tokens >= _RESEARCH_CONTEXT_WARN:
            _warned_context = True
            await _log(f"Context at {_total_tokens} tokens — nudging to wrap up")
            messages.append(Message(
                role="user",
                content=(
                    "[SYSTEM: You have used 75% of your context budget. "
                    "Start wrapping up: finish any critical lookups, then "
                    "produce your final summary within the next 1-2 iterations.]"
                ),
            ))

        try:
            _msgs, _tools = with_prompt_caching(
                messages, tool_specs if tool_specs else None, llm_params.get("model")
            )
            await rate_limiter.acquire(research_model)
            response = await acompletion(
                messages=_msgs,
                tools=_tools,
                tool_choice="auto",
                stream=False,
                timeout=120,
                **llm_params,
            )
        except Exception as e:
            logger.error("Research sub-agent LLM error: %s", e)
            return f"Research agent LLM error: {e}", False

        # Track tokens
        if response.usage:
            _total_tokens = response.usage.total_tokens
            await _log(f"tokens:{_total_tokens}")

        choice = response.choices[0]
        msg = choice.message

        # If no tool calls, we have our final answer
        if not msg.tool_calls:
            await _log("Research complete.")
            content = msg.content or "Research completed but no summary generated."
            return content, True

        # Execute tool calls and add results.
        # Rebuild the assistant message with only the wire-safe fields —
        # LiteLLM's raw Message carries `provider_specific_fields` and
        # `reasoning_content`, which the HF router's OpenAI schema rejects
        # if we echo them back in the next request.
        messages.append(Message(
            role="assistant",
            content=msg.content,
            tool_calls=msg.tool_calls,
        ))
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
                import json as _json

                args_str = _json.dumps(tool_args)[:80]
                await _log(f"▸ {tool_name}  {args_str}")

                output, _success = await session.tool_router.call_tool(
                    tool_name, tool_args, session=session
                )
                _tool_uses += 1
                await _log(f"tools:{_tool_uses}")
                # Truncate tool output for the research context
                if len(output) > 8000:
                    output = output[:4800] + "\n...(truncated)...\n" + output[-3200:]
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

    # ── Iteration limit: try to salvage findings ──
    await _log("Iteration limit reached — extracting summary")
    messages.append(Message(
        role="user",
        content=(
            "[SYSTEM: ITERATION LIMIT] You have reached the maximum number of research "
            "iterations. Summarize ALL findings so far. Do NOT call any more tools."
        ),
    ))
    try:
        _msgs, _ = with_prompt_caching(messages, None, llm_params.get("model"))
        await rate_limiter.acquire(research_model)
        response = await acompletion(
            messages=_msgs,
            tools=None,
            stream=False,
            timeout=120,
            **llm_params,
        )
        content = response.choices[0].message.content or ""
        if content:
            return content, True
    except Exception as e:
        logger.error("Research summary call failed: %s", e)

    return (
        "Research agent hit iteration limit (60). "
        "Partial findings may be incomplete — try a more focused task.",
        False,
    )
