<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

# ML Intern

An ML intern that autonomously researches, writes, and ships good quality ML related code using the Hugging Face ecosystem — with deep access to docs, papers, datasets, and cloud compute.

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
OPENAI_API_KEY=<your-openai-api-key> # if using openai models
OPENROUTER_API_KEY=<your-openrouter-api-key> # if using openrouter models
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
ml-intern --model anthropic/claude-opus-4-7 "your prompt"   # requires ANTHROPIC_API_KEY
ml-intern --model openai/gpt-5.5 "your prompt"              # requires OPENAI_API_KEY
ml-intern --model openrouter/anthropic/claude-opus-4.7 "your prompt" # requires OPENROUTER_API_KEY
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
```

Run `ml-intern` then `/model` to see the full list of suggested model ids
(Claude, GPT, OpenRouter, and HF-router models like MiniMax, Kimi, GLM, DeepSeek).
OpenRouter models must use the explicit `openrouter/<provider>/<model>` prefix.
Optional OpenRouter env vars `OPENROUTER_API_BASE`, `OR_SITE_URL`, and
`OR_APP_NAME` are passed through by LiteLLM. `OPENAI_BASE_URL` is not used;
`openai/...` remains reserved for direct OpenAI models.

## Sharing Traces

Every session is auto-uploaded to your **own private Hugging Face dataset**
in [Claude Code JSONL format](https://huggingface.co/changelog/agent-trace-viewer),
which the HF Agent Trace Viewer auto-detects so you can browse turns, tool
calls, and model responses directly on the Hub.

By default the dataset is named `{your-hf-username}/ml-intern-sessions` and is
**created private**. You can flip it to public from inside the CLI:

```bash
/share-traces            # show current visibility + dataset URL
/share-traces public     # publish (anyone can view)
/share-traces private    # lock it back down
```

You can also flip visibility from the dataset page on huggingface.co — the
agent honours whatever you set there for subsequent uploads.

To opt out entirely, set in your CLI config (e.g. `configs/cli_agent_config.json`
or `~/.config/ml-intern/cli_agent_config.json`):

```json
{ "share_traces": false }
```

To override the destination repo, set:

```json
{ "personal_trace_repo_template": "{hf_user}/my-custom-traces" }
```

The shared `smolagents/ml-intern-sessions` dataset is unrelated and only
receives anonymized telemetry rows used by the backend KPI scheduler.

## Supported Gateways

ML Intern currently supports one-way notification gateways from CLI sessions.
These gateways send out-of-band status updates; they do not accept inbound chat
messages.

### Slack

Slack notifications use the Slack Web API to post messages when the agent needs
approval, hits an error, or completes a turn. Create a Slack app with a bot token
that has `chat:write`, invite the bot to the target channel, then set:

```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C...
```

The CLI automatically creates a `slack.default` destination when both variables
are present. Optional environment variables for the env-only default:

```bash
ML_INTERN_SLACK_NOTIFICATIONS=false
ML_INTERN_SLACK_DESTINATION=slack.ops
ML_INTERN_SLACK_AUTO_EVENTS=approval_required,error,turn_complete
ML_INTERN_SLACK_ALLOW_AGENT_TOOL=true
ML_INTERN_SLACK_ALLOW_AUTO_EVENTS=true
```

For a persistent user-level config, put overrides in
`~/.config/ml-intern/cli_agent_config.json` or point `ML_INTERN_CLI_CONFIG` at a
JSON file:

```json
{
  "messaging": {
    "enabled": true,
    "auto_event_types": ["approval_required", "error", "turn_complete"],
    "destinations": {
      "slack.ops": {
        "provider": "slack",
        "token": "${SLACK_BOT_TOKEN}",
        "channel": "${SLACK_CHANNEL_ID}",
        "allow_agent_tool": true,
        "allow_auto_events": true
      }
    }
  }
}
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

## Development

### Pre-commit Checks

Run Ruff before every commit:

```bash
uv run ruff check .
uv run ruff format --check .
```

If the format check fails, run `uv run ruff format .` and re-run the checks
before committing.

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

Edit `configs/cli_agent_config.json` for CLI defaults, or
`configs/frontend_agent_config.json` for web-session defaults:

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
