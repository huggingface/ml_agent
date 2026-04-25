# Using ml-intern with Claude Code

This repo can run two ways:

1. **Standalone CLI** — `ml-intern` (the original; see [README](README.md)).
2. **Inside Claude Code** — `claude` from the repo root, picks up `CLAUDE.md`, `.mcp.json`, `.claude/`.

This guide covers (2). Both share the same tools under `agent/tools/`, so behavior matches; only the harness changes.

---

## Prerequisites

- [Claude Code](https://docs.claude.com/en/docs/claude-code) installed and signed in.
- [`uv`](https://docs.astral.sh/uv/) on `$PATH` (used to launch the MCP server and hooks).
- A clone of this repo with deps synced:

  ```bash
  git clone git@github.com:huggingface/ml-intern.git
  cd ml-intern
  uv sync
  ```

- An `.env` (or exported shell vars) with at minimum:

  ```bash
  HF_TOKEN=hf_...        # required — HF MCP server, papers, datasets, jobs, sessions upload
  GITHUB_TOKEN=ghp_...   # required — github_find_examples, github_read_file, github_list_repos
  ```

  Without `HF_TOKEN`, the HF MCP server returns 401s and the SessionStart hook reports `HF user: unknown`. Without `GITHUB_TOKEN`, the GitHub tools error.

---

## First run

From the repo root:

```bash
claude
```

That's it. Claude Code reads:

- `CLAUDE.md` — persona and methodology (research-first, dataset audit, pre-flight checklist for jobs, error-recovery rules).
- `.mcp.json` — auto-starts two MCP servers:
  - `ml-intern-tools` (stdio, local) — exposes `hf_papers`, `hf_inspect_dataset`, `hf_jobs`, `hf_repo_files`, `hf_repo_git`, `explore_hf_docs`, `fetch_hf_docs`, `github_*`, sandbox `bash`/`read`/`write`/`edit`.
  - `hf-mcp-server` (HTTP, hosted at `huggingface.co/mcp`) — official HF tools.
- `.claude/agents/research.md` — the parallel research subagent (read-only HF tools).
- `.claude/commands/*.md` — the slash commands listed below.
- `.claude/hooks/*.py` — content-aware approval, session redaction+upload, dynamic context injection.

You should see (early in the first turn) a system reminder like:

> HF user: **your-org** — use `your-org/<name>` as the namespace when constructing `hub_model_id`...

That's the SessionStart hook injecting context. If it says `HF user: unknown (...)`, fix the cause (missing token, expired token, network) before continuing.

---

## Slash commands

All commands accept free-form arguments after the name. They're prompt templates that route the agent through the right ml-intern workflow.

### `/ml-intern <task>`

Default entrypoint. Equivalent to `ml-intern "<task>"` in the standalone CLI — runs the full research→validate→implement workflow per `CLAUDE.md`.

```
/ml-intern fine-tune llama-3-8b on HuggingFaceH4/ultrachat_200k for math reasoning
```

### `/research <topic>`

Forces a literature crawl via the `research` subagent. Use when you want recipes, citation graphs, or methodology comparison **without** the agent jumping straight to code.

```
/research diffusion model fine-tuning for medical imaging
/research best DPO recipe for instruction tuning, 7B-13B range
```

The subagent has its own context window and read-only tools (papers, docs, datasets, github, hf-repo). Returns a ranked recipe table.

### `/inspect-dataset <id>`

Audit a HF dataset before training: schema, splits, sample rows, red flags, training-method compatibility (SFT/DPO/GRPO).

```
/inspect-dataset HuggingFaceH4/ultrachat_200k
/inspect-dataset Anthropic/hh-rlhf
```

### `/finetune <task>`

Strict, opinionated end-to-end fine-tune. Forces:
1. Research subagent first.
2. `hf_inspect_dataset` to verify column format.
3. Sandbox smoke test before anything large.
4. Pre-flight check (reference impl, `push_to_hub`, hardware, timeout, Trackio).
5. **One** job submitted; logs watched; only then any sweep.

```
/finetune llama-3-8b on HuggingFaceH4/ultrachat_200k
/finetune mistral-7b DPO on Anthropic/hh-rlhf
```

### `/run-job <job description>`

Submit any HF Job (training, eval, batch inference, data prep). Refuses to call `hf_jobs` until the pre-flight checklist is filled, including a ≥2h timeout for training jobs.

```
/run-job batch eval gpt2 on lm-eval harness MMLU
/run-job convert webdataset shards on 32 vCPUs
```

---

## Approvals — what to expect

ml-intern's approval policy is enforced via a `PreToolUse` hook (`.claude/hooks/pre_tool_use_approval.py`). Claude Code will prompt you when:

| Tool / op | When you'll be asked |
|---|---|
| `hf_jobs` (run/uv) on **GPU hardware** | Always |
| `hf_jobs` on CPU hardware | When `ML_INTERN_CONFIRM_CPU_JOBS=1` (default) |
| `hf_jobs` with a script that has `from_pretrained` but no `push_to_hub` | Always (warning surfaces in the prompt) |
| `sandbox_create` | Always |
| `hf_repo_files` `upload` / `delete` | Always |
| `hf_repo_git` destructive ops (delete branch/tag, merge PR, create/update repo) | Always |
| Anything else | Auto-allowed by static permissions (see `.claude/settings.json`) |

To skip all approvals (e.g. unattended overnight runs): `ML_INTERN_YOLO=1 claude`. **Don't habit-form that.**

If the hook crashes or gets a malformed payload, it **fails safe** — forces a prompt rather than silently allowing.

---

## Environment knobs

Set in your shell, `.env`, or override in `.claude/settings.json` `env` block. All have ml-intern-CLI equivalents.

| Env var | Default | What it does | CLI equivalent |
|---|---|---|---|
| `HF_TOKEN` | — | HF auth (tools, MCP, sessions upload, whoami) | same |
| `GITHUB_TOKEN` | — | GitHub tools | same |
| `HF_SESSION_UPLOAD_TOKEN` | — | Preferred (write-only) token for sessions upload; falls back to `HF_TOKEN` then `HF_ADMIN_TOKEN` | same |
| `ML_INTERN_YOLO` | `0` | Skip all approvals | `Config.yolo_mode` |
| `ML_INTERN_CONFIRM_CPU_JOBS` | `1` | Prompt before CPU jobs | `Config.confirm_cpu_jobs` |
| `ML_INTERN_SAVE_SESSIONS` | `1` | Upload transcripts to HF dataset on session end | `Config.save_sessions` |
| `ML_INTERN_SESSION_REPO` | `smolagents/ml-intern-sessions` | Target dataset | `Config.session_dataset_repo` |
| `ML_INTERN_LOCAL_MODE` | `0` | Run sandbox-style tools (`bash`/`read`/`write`/`edit`) on local fs instead of remote sandbox | `--local` |

When `ML_INTERN_LOCAL_MODE=1`, the SessionStart hook injects an extra reminder telling the model "no sandbox — operate on local fs, no `/app/` paths."

---

## Headless / unattended

For one-shot runs from CI or a script:

```bash
claude -p "/ml-intern fine-tune gpt2-medium on tatsu-lab/alpaca, push to my-org/gpt2-alpaca-test"
```

Pair with `ML_INTERN_YOLO=1` if you genuinely have no human in the loop. Read [`CLAUDE.md`](CLAUDE.md)'s "Autonomous / headless mode" section first — the rules differ from interactive (no text-only responses, always be doing work, hyperparameter sweeps not manual tuning).

---

## Privacy: what gets uploaded

When `ML_INTERN_SAVE_SESSIONS=1` (default), at session end the transcript is uploaded to `ML_INTERN_SESSION_REPO` (default: `smolagents/ml-intern-sessions`) **after** running it through `agent/core/redact.py::scrub`, which strips:

- `hf_…` HF tokens
- `sk-ant-…` Anthropic keys
- `sk-…` OpenAI keys
- `ghp_/gho_/ghu_/ghs_/ghr_/github_pat_…` GitHub tokens
- `AKIA…/ASIA…` AWS access keys
- `Bearer …` Authorization headers
- `KEY=value` exports for any name matching `HF_TOKEN|API_KEY|SECRET|PASSWORD|...`

Redaction is regex-based and best-effort. If you paste an unusual secret format ("hunter2") it won't be caught — don't paste secrets into chat.

The hook also refuses to upload a transcript whose path is outside `~/.claude/` or `$CLAUDE_PROJECT_DIR`. To opt out entirely: `ML_INTERN_SAVE_SESSIONS=0`.

---

## Common workflows

### "What's the best recipe for X?"

```
/research X
```

Wait for the recipe table. Then either ask follow-ups in the same turn or invoke `/finetune` with the recipe in mind.

### "Train this model on this dataset"

```
/finetune <model> on <dataset>
```

Watch for:
1. The research subagent's findings (loss recipe, hyperparameters).
2. `hf_inspect_dataset` output (column format check).
3. The sandbox smoke-test logs.
4. The pre-flight checklist.
5. Approval prompt for the GPU job. **Read the warning text** if any.
6. The job URL + Trackio dashboard URL.

### "Just run this script as a job"

```
/run-job <description>
```

Provide the script body in the chat or as a file path. The model will fill the pre-flight checklist before submitting.

### "Audit this dataset"

```
/inspect-dataset <id>
```

Useful as a standalone read; also useful before kicking off `/finetune` to spot column-format issues early.

---

## Troubleshooting

**"Tool not found: `mcp__ml-intern-tools__...`"** — the MCP server isn't running. Check `claude mcp list`; if it errors, run `uv run python -m packages.mcp_server.server < /dev/null` to surface the import error.

**"401 Unauthorized" from `hf_papers` or `hf_jobs`** — `HF_TOKEN` not in env. The `.mcp.json` substitutes `${HF_TOKEN}` from the launching shell; if you `claude` from a shell where it's not exported, the MCP server inherits an empty token.

**SessionStart shows `HF user: unknown (whoami HTTP error: ...)`** — token rejected. Probably expired or scoped wrong. Generate a new one at <https://huggingface.co/settings/tokens>.

**Approval prompt every turn for `hf_papers`** — the static permissions list in `.claude/settings.json` doesn't include the tool name, or the MCP server didn't register it. Verify with `claude mcp list` and check the tool name format (`mcp__ml-intern-tools__<name>`).

**`from_pretrained` warning on a script that's fine** — substring match is conservative. If the script genuinely doesn't need `push_to_hub` (e.g. eval-only), approve and proceed.

**Session upload fails silently** — check stderr of the Claude Code process. Errors print there. Common causes: token doesn't have write access to `ML_INTERN_SESSION_REPO`, or the dataset doesn't exist.

**Hook crashes** — run the hook by hand to reproduce:

```bash
echo '{"tool_name":"mcp__ml-intern-tools__hf_jobs","tool_input":{"operation":"run","script":"x","hardware_flavor":"a100-large"}}' \
  | uv run python .claude/hooks/pre_tool_use_approval.py
```

---

## Adding your own tools

The standalone CLI exposes new tools via `agent/tools/*.py` + a `ToolSpec` registered in `agent/core/tools.py`. To make those tools available inside Claude Code:

1. Implement the handler in `agent/tools/your_tool.py` with a `YOUR_TOOL_SPEC` dict and an async handler.
2. Add the `(spec, handler)` tuple to `_TOOL_SPECS` in `packages/mcp_server/server.py`.
3. Add `mcp__ml-intern-tools__<name>` to `.claude/settings.json` `permissions.allow`.
4. (Optional) If destructive, extend `_needs_approval` in `.claude/hooks/pre_tool_use_approval.py`.
5. (Optional) If read-only, add it to `.claude/agents/research.md` `tools:` frontmatter so the research subagent can use it.

The standalone CLI continues to work — both frontends share the same handler.

---

## Adding your own slash commands

Drop a markdown file at `.claude/commands/<name>.md`:

```markdown
---
description: One-line description shown in `/` listing.
argument-hint: <hint shown after /yourname>
---

Your prompt template here. Use $ARGUMENTS for the user's input.
```

The commands in this repo are intentionally opinionated (forcing research, refusing to skip pre-flight) — match that posture if you want consistent behavior.

---

## When to use the standalone CLI instead

The Claude Code path is the recommended default. Reach for `ml-intern` directly when you need:

- The original CLI's `/effort`, `/model`, `/yolo` toggles mid-session.
- The session JSONL trajectory written locally (the standalone CLI writes one; Claude Code's transcript is its own format).
- The web UI under `backend/`+`frontend/` for browsing past sessions.

Otherwise, use Claude Code — you get plan mode, native subagent ergonomics, better context management, and the same tool surface.
