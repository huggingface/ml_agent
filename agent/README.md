# Agent

Async agent loop with LiteLLM.

## Architecture

**Queue-based async system:**
- Submissions in (user input) → Agent Loop → Events output for possible UI updates
- Session maintains state (context + tools) for possible future Context Engineering
- Handlers operations like (USER_INPUT, INTERRUPT, COMPACT, UNDO, SHUTDOWN) for possible UI control

## Components

| Component | Purpose | Long Term Goal |
|-----------|---------|----------------|
| **`agent_loop.py`** | Core agentic loop: processes user input, calls LLM via LiteLLM, executes tool calls iteratively until completion, emits events | Support parallel tool execution, streaming responses, and advanced reasoning patterns |
| **`session.py`** | Maintains session state and interaction with potential UI (context, config, event queue), handles interrupts, assigns unique session IDs for tracing | Enable plugging in different UIs (CLI, web, API, programmatic etc.) |
| **`tools.py`** | `ToolRouter` manages potential built-in tools (e.g. bash, read_file, write_file which are dummy implementations rn) + MCP tools, converts specs to OpenAI format | Be the place for tools that can be used by the agent. All crazy tool design happens here. |
| **`context_manager/`** | Manages conversation history, very rudimentary context engineering support | Implement intelligent context engineering to keep the agent on track |
| **`config.py`** | Loads JSON config for the agent | Support different configs etc. |
| **`main.py`** | Interactive CLI with async queue architecture (submission→agent, agent→events) (simple way to interact with the agent now)| Serve as reference implementation for other UIs (web, API, programmatic) |

## Observability (optional)

LLM calls can additionally be streamed to a [LangFuse](https://langfuse.com)
instance — useful for local development and for self-hosted deployments
that already run LangFuse / Phoenix / Langsmith. The primary
HF-Dataset-based telemetry pipeline (`agent/core/telemetry.py`) is unchanged.

Set the LangFuse host plus both keys to opt in. Either env-var name for
the host works — Langfuse SDK v4 issues credentials as `LANGFUSE_BASE_URL`,
while litellm's callback reads `LANGFUSE_HOST`; this integration accepts
either and mirrors the value through to litellm:

```
LANGFUSE_BASE_URL=https://your-langfuse.example.com   # or LANGFUSE_HOST=...
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

Both self-hosted LangFuse and the SaaS endpoint
(`https://cloud.langfuse.com`) are supported. The host is mandatory so the
destination is always an explicit choice — there is no silent fallback.
With any of the three vars unset the integration is a no-op.

Install the optional dependency:

```
pip install ml-intern[observability]
```

**Privacy.** The callback ships the full prompt, tool calls, tool results,
and completions of every LLM turn to the configured host. Pick the
destination deliberately. See
https://github.com/huggingface/ml-intern/issues/196 for full details.
