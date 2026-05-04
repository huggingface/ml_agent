#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash post_train_bench/submit_eval_set.sh smoke

  bash post_train_bench/submit_eval_set.sh validation --dry-run

  bash post_train_bench/submit_eval_set.sh full --dry-run

Modes:
  smoke  Submit one 10-minute validation job.
  validation
         Submit a 4-job artifact-validity matrix with 2-hour solve budgets.
  full   Submit the full 4-model x 7-benchmark matrix. This is documented for manual use.

Options:
  --dry-run               Create metadata and matrix, print the sbatch command, do not submit.
  --allow-dirty           Allow full mode from a dirty worktree.
  --allow-mutable-images  Allow full mode with non-digest registry tags.

Environment:
  ML_INTERN_AGENT_MODEL        Intern model, used literally in runs/<model>/<run_id>.
                               Default: anthropic/claude-opus-4-6
  POST_TRAIN_BENCH_DIR         Default: scratch/PostTrainBench
  POST_TRAIN_BENCH_DOCKER_IMAGE
                               Default: registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest
  POST_TRAIN_BENCH_EVAL_DOCKER_IMAGE
                               Default: registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench-eval:latest
  POST_TRAIN_BENCH_SEED_HF_CACHE
                               Default: /fsx/lewis/post_train_bench/seed_hf_cache
  POST_TRAIN_BENCH_PROMPT_AGENT
                               Prompt rendering agent. Default: claude.
  POST_TRAIN_BENCH_SLURM_TIME  Slurm walltime. Default: 01:00:00 for smoke,
                               03:00:00 for validation,
                               14:00:00 for full.
  POST_TRAIN_BENCH_RUN_ID      Optional explicit run id. Overrides the default
                               YYYY-MM-DD_HH-MM-SS_{slurm_job_id} format.
  POST_TRAIN_BENCH_BASELINE_FINAL_MODEL
                               Smoke-only fallback. Default: 1 for smoke,
                               0 for validation/full.
  POST_TRAIN_BENCH_REPROMPT    Explicit reprompt method variant. Default: 0.
  POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES
                               Minimum minutes between headless continuation prompts.
                               Default: 30.
EOF
}

MODE="${1:-}"
if [ -z "$MODE" ] || [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    usage
    exit 0
fi
shift || true

DRY_RUN=0
ALLOW_DIRTY=0
ALLOW_MUTABLE_IMAGES=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --allow-dirty)
            ALLOW_DIRTY=1
            ;;
        --allow-mutable-images)
            ALLOW_MUTABLE_IMAGES=1
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

export ML_INTERN_AGENT_MODEL="${ML_INTERN_AGENT_MODEL:-anthropic/claude-opus-4-6}"

truthy_env() {
    case "${1,,}" in
        1|true|yes|on) echo 1 ;;
        *) echo 0 ;;
    esac
}

HOST_REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$HOST_REPO_ROOT"

if [ "$MODE" = "full" ] && [ "$DRY_RUN" -ne 1 ] && [ "$ALLOW_DIRTY" -ne 1 ] && [ -n "$(git status --short --untracked-files=no)" ]; then
    echo "Refusing full mode from a tracked-dirty worktree. Commit or stash changes, or pass --allow-dirty." >&2
    exit 2
fi

PTB_DIR="${POST_TRAIN_BENCH_DIR:-scratch/PostTrainBench}"
if [ ! -d "$PTB_DIR/src/eval/tasks" ]; then
    echo "PostTrainBench repo not found at $PTB_DIR" >&2
    exit 2
fi
PTB_DIR="$(cd "$PTB_DIR" && pwd)"

RUN_STAMP="${POST_TRAIN_BENCH_RUN_STAMP:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
RUN_PARENT="${HOST_REPO_ROOT}/post_train_bench/runs/${ML_INTERN_AGENT_MODEL}"
EXPLICIT_RUN_ID="${POST_TRAIN_BENCH_RUN_ID:-}"
DOCKER_IMAGE="${POST_TRAIN_BENCH_DOCKER_IMAGE:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest}"
EVAL_DOCKER_IMAGE="${POST_TRAIN_BENCH_EVAL_DOCKER_IMAGE:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench-eval:latest}"
SEED_HF_CACHE="${POST_TRAIN_BENCH_SEED_HF_CACHE:-/fsx/lewis/post_train_bench/seed_hf_cache}"
PROMPT_AGENT="${POST_TRAIN_BENCH_PROMPT_AGENT:-claude}"
BASELINE_FINAL_MODEL="${POST_TRAIN_BENCH_BASELINE_FINAL_MODEL:-0}"
REPROMPT="$(truthy_env "${POST_TRAIN_BENCH_REPROMPT:-0}")"
REPROMPT_MIN_MINUTES="${POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES:-30}"
METHOD_SUFFIX=""
if [ "$REPROMPT" = "1" ]; then
    METHOD_SUFFIX="_reprompt"
fi
export POST_TRAIN_BENCH_REPROMPT="$REPROMPT"
export POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES="$REPROMPT_MIN_MINUTES"
PTB_SLURM_JOB_ID=""

is_immutable_image() {
    local image="$1"
    if [ -f "$image" ]; then
        return 0
    fi
    case "$image" in
        *@sha256:*) return 0 ;;
        *) return 1 ;;
    esac
}

if [ "$MODE" = "full" ] && [ "$DRY_RUN" -ne 1 ] && [ "$ALLOW_MUTABLE_IMAGES" -ne 1 ]; then
    if ! is_immutable_image "$DOCKER_IMAGE"; then
        echo "Refusing full mode with mutable solve image: $DOCKER_IMAGE" >&2
        echo "Use a digest-pinned image or local .sqsh, or pass --allow-mutable-images." >&2
        exit 2
    fi
    if ! is_immutable_image "$EVAL_DOCKER_IMAGE"; then
        echo "Refusing full mode with mutable eval image: $EVAL_DOCKER_IMAGE" >&2
        echo "Use a digest-pinned image or local .sqsh, or pass --allow-mutable-images." >&2
        exit 2
    fi
fi

if [ -n "$EXPLICIT_RUN_ID" ] || [ "$DRY_RUN" -eq 1 ]; then
    RUN_ID="${EXPLICIT_RUN_ID:-${RUN_STAMP}_dryrun}"
    RUN_ROOT="${RUN_PARENT}/${RUN_ID}"
    if [ -e "$RUN_ROOT" ]; then
        echo "Run directory already exists: $RUN_ROOT" >&2
        exit 2
    fi
    mkdir -p "$RUN_ROOT"/{slurm,results,artifacts,env}
    MATRIX_FILE="$RUN_ROOT/matrix.jsonl"
else
    PENDING_ROOT="${RUN_PARENT}/.pending/${RUN_STAMP}_$$"
    mkdir -p "$PENDING_ROOT"
    MATRIX_FILE="$PENDING_ROOT/matrix.jsonl"
fi

case "$MODE" in
    smoke)
        BASELINE_FINAL_MODEL="${POST_TRAIN_BENCH_BASELINE_FINAL_MODEL:-1}"
        python3 - "$MATRIX_FILE" <<'PY'
import json
import sys
from pathlib import Path

rows = [{
    "benchmark": "gsm8k",
    "model_to_train": "Qwen/Qwen3-1.7B-Base",
    "num_hours": "0.167",
    "eval_limit": 8,
}]
Path(sys.argv[1]).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
PY
        ;;
    validation)
        python3 - "$MATRIX_FILE" <<'PY'
import json
import sys
from pathlib import Path

rows = [
    {
        "benchmark": "humaneval",
        "model_to_train": "Qwen/Qwen3-1.7B-Base",
        "num_hours": 2,
        "eval_limit": 8,
    },
    {
        "benchmark": "gsm8k",
        "model_to_train": "Qwen/Qwen3-1.7B-Base",
        "num_hours": 2,
        "eval_limit": 8,
    },
    {
        "benchmark": "bfcl",
        "model_to_train": "Qwen/Qwen3-1.7B-Base",
        "num_hours": 2,
        "eval_limit": 8,
    },
    {
        "benchmark": "gsm8k",
        "model_to_train": "google/gemma-3-4b-pt",
        "num_hours": 2,
        "eval_limit": 8,
    },
]
Path(sys.argv[1]).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
PY
        ;;
    full)
        python3 - "$MATRIX_FILE" <<'PY'
import json
import sys
from pathlib import Path

models = [
    "google/gemma-3-4b-pt",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
]
benchmarks = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "humaneval",
    "healthbench",
]
rows = [
    {"benchmark": benchmark, "model_to_train": model, "num_hours": 10}
    for model in models
    for benchmark in benchmarks
]
Path(sys.argv[1]).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
PY
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac

MATRIX_COUNT="$(wc -l < "$MATRIX_FILE" | tr -d ' ')"
case "$MODE" in
    smoke)
        DEFAULT_SLURM_TIME="01:00:00"
        ;;
    validation)
        DEFAULT_SLURM_TIME="03:00:00"
        ;;
    full)
        DEFAULT_SLURM_TIME="14:00:00"
        ;;
esac
SLURM_TIME="${POST_TRAIN_BENCH_SLURM_TIME:-$DEFAULT_SLURM_TIME}"

create_source_snapshot() {
    SOURCE_SNAPSHOT="${RUN_ROOT}/source_snapshot"
    rm -rf "$SOURCE_SNAPSHOT"
    mkdir -p "$SOURCE_SNAPSHOT"
    git archive --format=tar HEAD | tar -xf - -C "$SOURCE_SNAPSHOT"
    export SOURCE_SNAPSHOT
}

write_metadata() {
    export RUN_ID MODE DOCKER_IMAGE EVAL_DOCKER_IMAGE SEED_HF_CACHE PROMPT_AGENT PTB_DIR MATRIX_FILE MATRIX_COUNT RUN_STAMP PTB_SLURM_JOB_ID SOURCE_SNAPSHOT SLURM_TIME ALLOW_DIRTY ALLOW_MUTABLE_IMAGES BASELINE_FINAL_MODEL REPROMPT REPROMPT_MIN_MINUTES METHOD_SUFFIX
    python3 - "$RUN_ROOT/run_metadata.json" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, text=True, capture_output=True).stdout.strip()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def image_metadata(value: str) -> dict:
    path = Path(value)
    payload = {"value": value, "kind": "local_file" if path.is_file() else "registry"}
    if path.is_file():
        payload["sha256"] = sha256_file(path)
        payload["bytes"] = path.stat().st_size
    elif "@sha256:" in value:
        payload["digest"] = value.rsplit("@sha256:", 1)[1]
    else:
        payload["mutable"] = True
    return payload

status = git("status", "--short", "--untracked-files=no")
metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ["RUN_ID"],
    "run_stamp": os.environ["RUN_STAMP"],
    "slurm_job_id": os.environ.get("PTB_SLURM_JOB_ID") or None,
    "mode": os.environ["MODE"],
    "ml_intern_agent_model": os.environ["ML_INTERN_AGENT_MODEL"],
    "ml_intern_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "ml_intern_commit": git("rev-parse", "HEAD"),
    "ml_intern_short_commit": git("rev-parse", "--short=12", "HEAD"),
    "ml_intern_status_short": status,
    "dirty_worktree": bool(status),
    "docker_image": os.environ["DOCKER_IMAGE"],
    "solve_docker_image": os.environ["DOCKER_IMAGE"],
    "eval_docker_image": os.environ["EVAL_DOCKER_IMAGE"],
    "image_provenance": {
        "solve": image_metadata(os.environ["DOCKER_IMAGE"]),
        "eval": image_metadata(os.environ["EVAL_DOCKER_IMAGE"]),
    },
    "allow_dirty": os.environ["ALLOW_DIRTY"] == "1",
    "allow_mutable_images": os.environ["ALLOW_MUTABLE_IMAGES"] == "1",
    "baseline_final_model": os.environ["BASELINE_FINAL_MODEL"] == "1",
    "reprompt_enabled": os.environ["REPROMPT"] == "1",
    "reprompt_min_minutes": float(os.environ["REPROMPT_MIN_MINUTES"]),
    "method_variant": "reprompt" if os.environ["REPROMPT"] == "1" else "standard",
    "method_suffix": os.environ["METHOD_SUFFIX"],
    "seed_hf_cache": os.environ["SEED_HF_CACHE"],
    "prompt_agent": os.environ["PROMPT_AGENT"],
    "slurm_time": os.environ["SLURM_TIME"],
    "post_train_bench_dir": os.environ["PTB_DIR"],
    "matrix_file": os.environ["MATRIX_FILE"],
    "matrix_count": int(os.environ["MATRIX_COUNT"]),
    "source_snapshot": os.environ.get("SOURCE_SNAPSHOT") or None,
}
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + "\n")
PY
    python3 - "$RUN_ROOT/env/submit_env.txt" <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("ml_intern_redact", Path("agent/core/redact.py"))
assert spec is not None and spec.loader is not None
redact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(redact)

lines = [redact.scrub_string(f"{key}={value}") for key, value in sorted(os.environ.items())]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

if [ "$DRY_RUN" -eq 1 ]; then
    SOURCE_SNAPSHOT="${RUN_ROOT}/source_snapshot"
    SBATCH_CMD=(
        sbatch
        --parsable
        --hold
        "--array=0-$((MATRIX_COUNT - 1))"
        "--time=${SLURM_TIME}"
        "--export=ALL,RUN_PARENT=${RUN_PARENT},RUN_STAMP=${RUN_STAMP},PTB_DIR=${PTB_DIR},POST_TRAIN_BENCH_DOCKER_IMAGE=${DOCKER_IMAGE},POST_TRAIN_BENCH_EVAL_DOCKER_IMAGE=${EVAL_DOCKER_IMAGE},POST_TRAIN_BENCH_SEED_HF_CACHE=${SEED_HF_CACHE},POST_TRAIN_BENCH_PROMPT_AGENT=${PROMPT_AGENT},POST_TRAIN_BENCH_BASELINE_FINAL_MODEL=${BASELINE_FINAL_MODEL},POST_TRAIN_BENCH_REPROMPT=${REPROMPT},POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES=${REPROMPT_MIN_MINUTES}"
        post_train_bench/launch.slurm
    )
    write_metadata
    printf '%q ' "${SBATCH_CMD[@]}" > "$RUN_ROOT/sbatch_command.txt"
    printf '\n' >> "$RUN_ROOT/sbatch_command.txt"
    echo "Run root: $RUN_ROOT"
    echo "Matrix rows: $MATRIX_COUNT"
    echo "Command: $(cat "$RUN_ROOT/sbatch_command.txt")"
    echo "Dry run only; not submitting. The dry-run id uses a dryrun suffix because no Slurm job id exists."
    exit 0
fi

if [ -n "$EXPLICIT_RUN_ID" ]; then
    create_source_snapshot
    SBATCH_CMD=(
        sbatch
        --parsable
        "--array=0-$((MATRIX_COUNT - 1))"
        "--time=${SLURM_TIME}"
        "--export=ALL,RUN_ROOT=${RUN_ROOT},MATRIX_FILE=${MATRIX_FILE},PTB_DIR=${PTB_DIR},REPO_ROOT=${SOURCE_SNAPSHOT},POST_TRAIN_BENCH_DOCKER_IMAGE=${DOCKER_IMAGE},POST_TRAIN_BENCH_EVAL_DOCKER_IMAGE=${EVAL_DOCKER_IMAGE},POST_TRAIN_BENCH_SEED_HF_CACHE=${SEED_HF_CACHE},POST_TRAIN_BENCH_PROMPT_AGENT=${PROMPT_AGENT},POST_TRAIN_BENCH_BASELINE_FINAL_MODEL=${BASELINE_FINAL_MODEL},POST_TRAIN_BENCH_REPROMPT=${REPROMPT},POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES=${REPROMPT_MIN_MINUTES},RUN_ID=${RUN_ID}"
        post_train_bench/launch.slurm
    )
    write_metadata
    printf '%q ' "${SBATCH_CMD[@]}" > "$RUN_ROOT/sbatch_command.txt"
    printf '\n' >> "$RUN_ROOT/sbatch_command.txt"
    echo "Run root: $RUN_ROOT"
    echo "Matrix rows: $MATRIX_COUNT"
    echo "Command: $(cat "$RUN_ROOT/sbatch_command.txt")"
    SBATCH_RESULT="$("${SBATCH_CMD[@]}")"
    PTB_SLURM_JOB_ID="${SBATCH_RESULT%%;*}"
    write_metadata
    echo "Submitted batch job $PTB_SLURM_JOB_ID" | tee "$RUN_ROOT/sbatch_output.txt"
    exit 0
fi

SBATCH_CMD=(
    sbatch
    --parsable
    --hold
    "--array=0-$((MATRIX_COUNT - 1))"
    "--time=${SLURM_TIME}"
    "--export=ALL,RUN_PARENT=${RUN_PARENT},RUN_STAMP=${RUN_STAMP},PTB_DIR=${PTB_DIR},POST_TRAIN_BENCH_DOCKER_IMAGE=${DOCKER_IMAGE},POST_TRAIN_BENCH_EVAL_DOCKER_IMAGE=${EVAL_DOCKER_IMAGE},POST_TRAIN_BENCH_SEED_HF_CACHE=${SEED_HF_CACHE},POST_TRAIN_BENCH_PROMPT_AGENT=${PROMPT_AGENT},POST_TRAIN_BENCH_BASELINE_FINAL_MODEL=${BASELINE_FINAL_MODEL},POST_TRAIN_BENCH_REPROMPT=${REPROMPT},POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES=${REPROMPT_MIN_MINUTES}"
    post_train_bench/launch.slurm
)
SBATCH_RESULT="$("${SBATCH_CMD[@]}")"
PTB_SLURM_JOB_ID="${SBATCH_RESULT%%;*}"
RUN_ID="${RUN_STAMP}_${PTB_SLURM_JOB_ID}"
RUN_ROOT="${RUN_PARENT}/${RUN_ID}"

if [ -e "$RUN_ROOT" ]; then
    echo "Run directory already exists: $RUN_ROOT" >&2
    echo "Held Slurm job $PTB_SLURM_JOB_ID was not released." >&2
    exit 2
fi

mkdir -p "$RUN_ROOT"/{slurm,results,artifacts,env}
mv "$MATRIX_FILE" "$RUN_ROOT/matrix.jsonl"
rmdir "$PENDING_ROOT" 2>/dev/null || true
MATRIX_FILE="$RUN_ROOT/matrix.jsonl"
create_source_snapshot

write_metadata
printf '%q ' "${SBATCH_CMD[@]}" > "$RUN_ROOT/sbatch_command.txt"
printf '\n' >> "$RUN_ROOT/sbatch_command.txt"

echo "Run root: $RUN_ROOT"
echo "Matrix rows: $MATRIX_COUNT"
echo "Command: $(cat "$RUN_ROOT/sbatch_command.txt")"
{
    echo "Submitted batch job $PTB_SLURM_JOB_ID"
    echo "Slurm parsable output: $SBATCH_RESULT"
} > "$RUN_ROOT/sbatch_output.txt"
scontrol release "$PTB_SLURM_JOB_ID" | tee -a "$RUN_ROOT/sbatch_output.txt"
