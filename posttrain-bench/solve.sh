#!/bin/bash
# PostTrainBench solve script for hf_agent
# Copied into the container as agent_solve.sh by run_task.sh.
# Environment: Apptainer container with CUDA, uv, Python 3.10 system.
# $PROMPT is set by run_task.sh. CWD is /home/ben/task.

set -euo pipefail

echo "=== hf_agent solve.sh ==="
echo "Working directory: $(pwd)"
echo "PROMPT length: ${#PROMPT}"

export PATH="/root/.local/bin:$PATH"

# Clone the agent source (container has git + internet)
AGENT_DIR="/home/ben/hf_agent"
git clone --depth 1 --branch posttrain-bench \
    https://github.com/huggingface/ml-intern.git "$AGENT_DIR"

cd "$AGENT_DIR"

# Install agent into the system Python (3.11) which already has
# torch, trl, vllm, flash-attn, etc. pre-installed in the Docker image.
# This avoids creating an isolated venv that can't see those packages.
uv pip install --system -e .

# Return to task directory (evaluate.py, timer.sh, templates/)
cd /home/ben/task

# Map PostTrainBench AGENT_CONFIG (e.g. "claude-opus-4-6") to litellm model ID
MODEL_FLAG=""
if [ -n "${AGENT_CONFIG:-}" ]; then
    MODEL_FLAG="--model anthropic/${AGENT_CONFIG}"
fi

# Run headlessly with unlimited iterations for the 10-hour budget
python -m agent.main --max-iterations -1 --no-stream $MODEL_FLAG "$PROMPT"
