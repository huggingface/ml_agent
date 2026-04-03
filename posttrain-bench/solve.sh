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
    https://github.com/huggingface/hf_agent.git "$AGENT_DIR"

cd "$AGENT_DIR"

# Container has Python 3.10 but agent needs 3.12+
# uv will auto-download Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install agent with its dependencies
uv pip install -e ".[agent]"

# Return to task directory (evaluate.py, timer.sh, templates/)
cd /home/ben/task

# Run headlessly with unlimited iterations for the 10-hour budget
python -m agent.main --max-iterations -1 "$PROMPT"
