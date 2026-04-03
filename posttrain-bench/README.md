# PostTrainBench Integration

## Setup

1. Copy `solve.sh` to `PostTrainBench/agents/hf_agent/solve.sh`
2. Add `hf_agent` entries to `PostTrainBench/src/commit_utils/commit.sh`
3. Set `ANTHROPIC_API_KEY` and `HF_TOKEN` environment variables on the cluster

## How it works

PostTrainBench's `run_task.sh`:
1. Copies `agents/hf_agent/solve.sh` into the Apptainer container
2. Sets `$PROMPT` with the task description
3. Runs the script inside the container with H100 GPU access

The solve script:
1. Clones this repo (branch `posttrain-bench`) inside the container
2. Creates a Python 3.12 venv via `uv` (container ships 3.10)
3. Installs agent dependencies
4. Runs `python -m agent.main --max-iterations -1 "$PROMPT"` headlessly

## Running locally (for testing)

```bash
python -m agent.main --max-iterations -1 "Your prompt here"
```
