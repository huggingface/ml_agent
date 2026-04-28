#!/bin/bash
#SBATCH --job-name=ml-intern-ptb
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper-prod
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "launch.slurm must be submitted as an array job" >&2
    exit 2
fi
if [ -z "${RUN_ROOT:-}" ]; then
    if [ -z "${RUN_PARENT:-}" ] || [ -z "${RUN_STAMP:-}" ]; then
        echo "RUN_ROOT or RUN_PARENT/RUN_STAMP is required" >&2
        exit 2
    fi
    RUN_ROOT="${RUN_PARENT}/${RUN_STAMP}_${SLURM_ARRAY_JOB_ID}"
fi
if [ -z "${MATRIX_FILE:-}" ]; then
    MATRIX_FILE="${RUN_ROOT}/matrix.jsonl"
fi
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="${RUN_ROOT}/source_snapshot"
fi
export RUN_ROOT
export REPO_ROOT
export RUN_ID="${RUN_ID:-$(basename "$RUN_ROOT")}"

mkdir -p "${RUN_ROOT}/slurm"
exec >"${RUN_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
exec 2>"${RUN_ROOT}/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

module load cuda/12.9 || true
set -x
cd "$REPO_ROOT"

readarray -t ROW < <(
    python - "$MATRIX_FILE" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
from pathlib import Path

matrix_path = Path(sys.argv[1])
task_id = int(sys.argv[2])
rows = [json.loads(line) for line in matrix_path.read_text().splitlines() if line.strip()]
row = rows[task_id]
num_hours = str(row["num_hours"])
duration_minutes = int(row.get("duration_minutes", round(float(num_hours) * 60)))
print(row["benchmark"])
print(row["model_to_train"])
print(num_hours)
print(duration_minutes)
PY
)

BENCHMARK="${ROW[0]}"
MODEL_TO_TRAIN="${ROW[1]}"
NUM_HOURS="${ROW[2]}"
DURATION_MINUTES="${ROW[3]}"

PTB_SLURM_NAME="$(
    python - "$ML_INTERN_AGENT_MODEL" "$MODEL_TO_TRAIN" "$BENCHMARK" <<'PY'
import re
import sys

parts = sys.argv[1:]
name = "ptb_" + "_".join(parts)
name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
print(name[:128])
PY
)"
scontrol update "JobId=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" "Name=${PTB_SLURM_NAME}" || true
echo "slurm_job_name=${PTB_SLURM_NAME}"

bash post_train_bench/run_task_docker.sh \
    "${BENCHMARK}" \
    "${MODEL_TO_TRAIN}" \
    "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    "${NUM_HOURS}" \
    "${DURATION_MINUTES}"
