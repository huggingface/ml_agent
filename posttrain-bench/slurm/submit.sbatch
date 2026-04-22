#!/bin/bash
#SBATCH --job-name=ptb
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=248000
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ptb_%j.out
#SBATCH --error=logs/ptb_%j.err
#SBATCH --export=ALL

# Usage: cd /fsx/aksel/hf_agent && sbatch posttrain-bench/slurm/submit.sbatch gsm8k hf_agent Qwen/Qwen3-1.7B-Base 1 claude-opus-4-6
# Must be submitted from the hf_agent repo root.

set -euo pipefail

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
NUM_HOURS="$4"
AGENT_CONFIG="$5"
REASONING_EFFORT="${6:-}"

# Paths
REPO_DIR="${SLURM_SUBMIT_DIR}"
POSTTRAINBENCH_DIR="${POSTTRAINBENCH_DIR:-/fsx/aksel/PostTrainBench}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REGISTRY="registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${REGISTRY}/posttrainbench:latest}"

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')
EXPERIMENT_NAME="${POST_TRAIN_BENCH_EXPERIMENT_NAME:-}"

EVAL_DIR="${POSTTRAINBENCH_DIR}/results/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${SLURM_JOB_ID}"
mkdir -p "${EVAL_DIR}"

echo "Task: $EVALUATION_TASK | Agent: $AGENT | Model: $MODEL_TO_TRAIN | Hours: $NUM_HOURS | Config: $AGENT_CONFIG"
echo "Results: $EVAL_DIR"
echo "Container: $CONTAINER_IMAGE"
echo "Node: $(hostname)"
nvidia-smi -L

# ── Prepare task directory ──
JOB_DIR="/tmp/posttrain_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}/task"

cp "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${JOB_DIR}/task/"
if [ -d "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${JOB_DIR}/task/"
fi
cp -r "${POSTTRAINBENCH_DIR}/src/eval/templates" "${JOB_DIR}/task/"
if [ -d "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/task_context"/* "${JOB_DIR}/task/"
fi

# Generate prompt (runs on host, before container)
# get_prompt.py uses relative paths, so must run from PostTrainBench root
BENCHMARK=$(cat "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")
PROMPT=$(cd "${POSTTRAINBENCH_DIR}" && python3 src/eval/general/get_prompt.py \
    --model-to-train "$MODEL_TO_TRAIN" \
    --benchmark-id "$EVALUATION_TASK" \
    --num-hours "$NUM_HOURS" \
    --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

# Generate timer
bash "${POSTTRAINBENCH_DIR}/src/utils/create_timer.sh" "$NUM_HOURS" "${JOB_DIR}/task/timer.sh"

# Write the prompt to a file so we don't have to escape it in env vars
echo "$PROMPT" > "${JOB_DIR}/prompt.txt"

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"
START_TIME=$(date +%s)

# Run agent inside container via Pyxis
# Mounts: task dir, HF cache, eval results dir, agent repo (for source)
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${JOB_DIR}:/home/ben,${HF_CACHE}:/root/.cache/huggingface,${EVAL_DIR}:/eval_dir" \
    --container-workdir="/home/ben/task" \
    --no-container-mount-home \
    --container-writable \
    bash -c '
        export PATH="/root/.local/bin:$PATH"
        export HF_HOME="/root/.cache/huggingface"
        export ANTHROPIC_API_KEY="'"${ANTHROPIC_API_KEY:-}"'"
        export OPENAI_API_KEY="'"${OPENAI_API_KEY:-}"'"
        export HF_TOKEN="'"${HF_TOKEN:-}"'"
        unset VIRTUAL_ENV

        # Clone agent and install into system Python (3.11) which already
        # has torch, trl, vllm, flash-attn etc. from the Docker image.
        git clone --depth 1 --branch posttrain-bench \
            https://github.com/huggingface/ml-intern.git /tmp/hf_agent
        cd /tmp/hf_agent
        uv pip install --system -e .

        # Run agent in the task directory
        cd /home/ben/task
        PROMPT=$(cat /home/ben/prompt.txt)

        MODEL_FLAG=""
        if [ -n "'"${AGENT_CONFIG}"'" ]; then
            MODEL_FLAG="--model '"${AGENT_CONFIG}"'"
        fi

        REASONING_FLAG=""
        if [ -n "'"${REASONING_EFFORT}"'" ]; then
            REASONING_FLAG="--reasoning-effort '"${REASONING_EFFORT}"'"
        fi

        python -m agent.main --max-iterations -1 --no-stream --resume timer.sh --resume-min-minutes 30 $MODEL_FLAG $REASONING_FLAG "$PROMPT"
    ' > "${SOLVE_OUT}" 2>&1 || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf '%02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) > "${EVAL_DIR}/time_taken.txt"

echo "============================================"
echo "=== TASK COMPLETE ($(cat ${EVAL_DIR}/time_taken.txt)) ==="
echo "============================================"

# Copy results out of container work dir
if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
    echo "Final model saved to $EVAL_DIR/final_model"
else
    echo "WARNING: No final_model directory found"
fi

cp -r "${JOB_DIR}/task" "$EVAL_DIR/task" 2>/dev/null || true

# Clean up container overlay from agent run to free /tmp for eval
rm -rf "${JOB_DIR}" 2>/dev/null || true

# ── Evaluation (also in container) ──
echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

if [ -d "${EVAL_DIR}/final_model" ] && [ -f "${EVAL_DIR}/final_model/config.json" ]; then
    srun \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${EVAL_DIR}:/eval_dir,${HF_CACHE}:/root/.cache/huggingface,${POSTTRAINBENCH_DIR}:/ptb" \
        --container-workdir="/ptb/src/eval/tasks/${EVALUATION_TASK}" \
        --no-container-mount-home \
        --container-writable \
        bash -c '
            export HF_HOME="/root/.cache/huggingface"
            export VLLM_API_KEY="inspectai"
            export OPENAI_API_KEY="'"${OPENAI_API_KEY:-}"'"
            unset HF_HUB_ENABLE_HF_TRANSFER
            python evaluate.py \
                --model-path /eval_dir/final_model \
                --templates-dir /ptb/src/eval/templates \
                --limit -1 \
                --json-output-file /eval_dir/metrics.json
        ' > "$EVAL_DIR/final_eval.txt" 2>&1 || true

    if [ -f "${EVAL_DIR}/metrics.json" ]; then
        echo "Metrics:"
        cat "${EVAL_DIR}/metrics.json"
    else
        echo "WARNING: Evaluation did not produce metrics.json"
    fi
else
    echo "WARNING: No final_model found, skipping evaluation"
fi

echo "================================"
echo "======= DONE ========"
echo "================================"
