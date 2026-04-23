<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

# ML Intern

An ML intern that autonomously researches, writes, and ships good quality ML releated code using the Hugging Face ecosystem — with deep access to docs, papers, datasets, and cloud compute.

## Quick Start

### Installation

```bash
git clone git@github.com:huggingface/ml-intern.git
cd ml-intern
uv sync
uv tool install -e .
```

#### That's it. Now `ml-intern` works from any directory:

```bash
ml-intern
```

Create a `.env` file in the project root (or export these in your shell):

```bash
ANTHROPIC_API_KEY=<your-anthropic-api-key> # if using anthropic models
HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token> 
```
If no `HF_TOKEN` is set, the CLI will prompt you to paste one on first launch. To get a GITHUB_TOKEN follow the tutorial [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token).

### Usage

**Interactive mode** (start a chat session):

```bash
ml-intern
```

**Headless mode** (single prompt, auto-approve):

```bash
ml-intern "fine-tune llama on my dataset"
```

**Options:**

```bash
ml-intern --model anthropic/claude-opus-4-6 "your prompt"
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User/CLI                            │
└────────────┬─────────────────────────────────────┬──────────┘
             │ Operations                          │ Events
             ↓ (user_input, exec_approval,         ↑
      submission_queue  interrupt, compact, ...)  event_queue
             │                                          │
             ↓                                          │
┌────────────────────────────────────────────────────┐  │
│            submission_loop (agent_loop.py)         │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │  1. Receive Operation from queue             │  │  │
│  │  2. Route to handler (run_agent/compact/...) │  │  │
│  └──────────────────────────────────────────────┘  │  │
│                      ↓                             │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │         Handlers.run_agent()                 │  ├──┤
│  │                                              │  │  │
│  │  ┌────────────────────────────────────────┐  │  │  │
│  │  │  Agentic Loop (max 300 iterations)     │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Session                          │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ContextManager             │  │  │  │  │  │
│  │  │  │  │ • Message history          │  │  │  │  │  │
│  │  │  │  │   (litellm.Message[])      │  │  │  │  │  │
│  │  │  │  │ • Auto-compaction (170k)   │  │  │  │  │  │
│  │  │  │  │ • Session upload to HF     │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  │                                  │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ToolRouter                 │  │  │  │  │  │
│  │  │  │  │  ├─ HF docs & research     │  │  │  │  │  │
│  │  │  │  │  ├─ HF repos, datasets,    │  │  │  │  │  │
│  │  │  │  │  │  jobs, papers           │  │  │  │  │  │
│  │  │  │  │  ├─ GitHub code search     │  │  │  │  │  │
│  │  │  │  │  ├─ Sandbox & local tools  │  │  │  │  │  │
│  │  │  │  │  ├─ Planning               │  │  │  │  │  │
│  │  │  │  │  └─ MCP server tools       │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Doom Loop Detector               │  │  │  │  │
│  │  │  │ • Detects repeated tool patterns │  │  │  │  │
│  │  │  │ • Injects corrective prompts     │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  Loop:                                 │  │  │  │
│  │  │    1. LLM call (litellm.acompletion)   │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    2. Parse tool_calls[]               │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    3. Approval check                   │  │  │  │
│  │  │       (jobs, sandbox, destructive ops) │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    4. Execute via ToolRouter           │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    5. Add results to ContextManager    │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    6. Repeat if tool_calls exist       │  │  │  │
│  │  └────────────────────────────────────────┘  │  │  │
│  └──────────────────────────────────────────────┘  │  │
└────────────────────────────────────────────────────┴──┘
```

### Agentic Loop Flow

```
User Message
     ↓
[Add to ContextManager]
     ↓
     ╔═══════════════════════════════════════════╗
     ║      Iteration Loop (max 300)             ║
     ║                                           ║
     ║  Get messages + tool specs                ║
     ║         ↓                                 ║
     ║  litellm.acompletion()                    ║
     ║         ↓                                 ║
     ║  Has tool_calls? ──No──> Done             ║
     ║         │                                 ║
     ║        Yes                                ║
     ║         ↓                                 ║
     ║  Add assistant msg (with tool_calls)      ║
     ║         ↓                                 ║
     ║  Doom loop check                          ║
     ║         ↓                                 ║
     ║  For each tool_call:                      ║
     ║    • Needs approval? ──Yes──> Wait for    ║
     ║    │                         user confirm ║
     ║    No                                     ║
     ║    ↓                                      ║
     ║    • ToolRouter.execute_tool()            ║
     ║    • Add result to ContextManager         ║
     ║         ↓                                 ║
     ║  Continue loop ─────────────────┐         ║
     ║         ↑                       │         ║
     ║         └───────────────────────┘         ║
     ╚═══════════════════════════════════════════╝
```

## Events

The agent emits the following events via `event_queue`:

- `processing` - Starting to process user input
- `ready` - Agent is ready for input
- `assistant_chunk` - Streaming token chunk
- `assistant_message` - Complete LLM response text
- `assistant_stream_end` - Token stream finished
- `tool_call` - Tool being called with arguments
- `tool_output` - Tool execution result
- `tool_log` - Informational tool log message
- `tool_state_change` - Tool execution state transition
- `approval_required` - Requesting user approval for sensitive operations
- `turn_complete` - Agent finished processing
- `error` - Error occurred during processing
- `interrupted` - Agent was interrupted
- `compacted` - Context was compacted
- `undo_complete` - Undo operation completed
- `shutdown` - Agent shutting down

## Session Limits

Session creation is concurrency-safe and enforced server-side. If the Space is at capacity, `/api/session` returns `503` instead of oversubscribing the limit. Per-user limits are enforced the same way, so bursts of simultaneous requests cannot create extra sessions past the configured cap.

## Development

### Adding Built-in Tools

Edit `agent/core/tools.py`:

```python
def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="your_tool",
            description="What your tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            },
            handler=your_async_handler
        ),
        # ... existing tools
    ]
```

### Adding MCP Servers

Edit `configs/main_agent_config.json`:

```json
{
  "model_name": "anthropic/claude-sonnet-4-5-20250929",
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

Note: Environment variables like `${YOUR_TOKEN}` are auto-substituted from `.env`.
