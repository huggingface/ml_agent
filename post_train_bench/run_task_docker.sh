#!/bin/bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 BENCHMARK MODEL_TO_TRAIN TASK_RUN_ID NUM_HOURS EVAL_LIMIT" >&2
    exit 2
fi

BENCHMARK="$1"
MODEL_TO_TRAIN="$2"
TASK_RUN_ID="$3"
NUM_HOURS="$4"
EVAL_LIMIT="$5"

if [ -z "${RUN_ROOT:-}" ] || [ -z "${REPO_ROOT:-}" ] || [ -z "${PTB_DIR:-}" ]; then
    echo "RUN_ROOT, REPO_ROOT, and PTB_DIR must be exported" >&2
    exit 2
fi
if [ -z "${ML_INTERN_AGENT_MODEL:-}" ]; then
    echo "ML_INTERN_AGENT_MODEL must be exported" >&2
    exit 2
fi

SOLVE_DOCKER_IMAGE="${POST_TRAIN_BENCH_DOCKER_IMAGE:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest}"
EVAL_DOCKER_IMAGE="${POST_TRAIN_BENCH_EVAL_DOCKER_IMAGE:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench-eval:latest}"
SEED_HF_CACHE="${POST_TRAIN_BENCH_SEED_HF_CACHE:-/fsx/lewis/post_train_bench/seed_hf_cache}"
PROMPT_AGENT="${POST_TRAIN_BENCH_PROMPT_AGENT:-claude}"

truthy_env() {
    case "${1,,}" in
        1|true|yes|on) echo 1 ;;
        *) echo 0 ;;
    esac
}

REPROMPT="$(truthy_env "${POST_TRAIN_BENCH_REPROMPT:-0}")"
REPROMPT_MIN_MINUTES="${POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES:-30}"
METHOD_SUFFIX=""
if [ "$REPROMPT" = "1" ]; then
    METHOD_SUFFIX="_reprompt"
fi
export POST_TRAIN_BENCH_REPROMPT="$REPROMPT"
export POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES="$REPROMPT_MIN_MINUTES"

DURATION_MINUTES="$(python3 - "$NUM_HOURS" <<'PY'
import math
import sys
print(max(1, math.ceil(float(sys.argv[1]) * 60)))
PY
)"
DURATION_SECONDS="$((DURATION_MINUTES * 60))"
SOLVE_TIMEOUT_SECONDS="${POST_TRAIN_BENCH_FORCE_SOLVE_TIMEOUT_SECONDS:-$DURATION_SECONDS}"

safe_name() {
    python3 - "$1" <<'PY'
import sys
print(sys.argv[1].replace("/", "_").replace(":", "_").replace("[", "_").replace("]", "_"))
PY
}

MODEL_SAFE="$(safe_name "$MODEL_TO_TRAIN")"
AGENT_SAFE="$(safe_name "$ML_INTERN_AGENT_MODEL")"
METHOD_DIR="ml_intern_${AGENT_SAFE}_${NUM_HOURS}h${METHOD_SUFFIX}"
EVAL_DIR="${RUN_ROOT}/results/${METHOD_DIR}/${BENCHMARK}_${MODEL_SAFE}_${TASK_RUN_ID}"
TMP_BASE="${SLURM_TMPDIR:-/scratch/${USER:-user}}"
TMP_SUBDIR="${TMP_BASE}/ml_intern_ptb_${BENCHMARK}_${MODEL_SAFE}_${TASK_RUN_ID}_$$"
JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"
JOB_REPO="${TMP_SUBDIR}/ml-intern-src"
JOB_JUDGE="${TMP_SUBDIR}/judge"
TRUSTED_RUNNER_DIR="${TMP_SUBDIR}/trusted-runner"
TRUSTED_INTEGRITY="${TRUSTED_RUNNER_DIR}/post_train_bench/integrity.py"
TRUSTED_COLLECT="${TRUSTED_RUNNER_DIR}/post_train_bench/collect_artifacts.py"
JUDGE_EVIDENCE_DIR="${TMP_SUBDIR}/judge_evidence"
TASK_CACHE_ROOT="${TMP_BASE}/post_train_bench_hf_cache/${BENCHMARK}_${MODEL_SAFE}_${TASK_RUN_ID}_$$"
SOLVE_HF_CACHE="${TASK_CACHE_ROOT}/solve"
EVAL_HF_CACHE="${TASK_CACHE_ROOT}/eval"
MONITOR_PID=""

cleanup() {
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    rm -rf "$TMP_SUBDIR" "$TASK_CACHE_ROOT"
}
trap cleanup EXIT

seed_cache() {
    local dest="$1"
    mkdir -p "$dest"
    if [ -d "$SEED_HF_CACHE" ]; then
        cp -a "$SEED_HF_CACHE/." "$dest/"
    else
        echo "Seed HF cache not found, starting with an empty cache: $SEED_HF_CACHE"
    fi
}

start_system_monitor() {
    local interval="${POST_TRAIN_BENCH_MONITOR_INTERVAL_SECONDS:-30}"
    (
        while true; do
            echo "=== $(date -u --iso-8601=seconds) ==="
            uptime || true
            free -h || true
            df -h "$JOB_DIR" "$JOB_TMP" "$SOLVE_HF_CACHE" "$EVAL_HF_CACHE" 2>/dev/null || true
            if command -v nvidia-smi >/dev/null 2>&1; then
                nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv || true
            fi
            echo
            sleep "$interval"
        done
    ) >> "$EVAL_DIR/system_monitor.log" 2>&1 &
    MONITOR_PID="$!"
}

rm -rf "$TMP_SUBDIR" "$TASK_CACHE_ROOT"
mkdir -p "$EVAL_DIR" "$JOB_DIR/task" "$JOB_TMP" "$JOB_REPO" "$JOB_JUDGE" "$TRUSTED_RUNNER_DIR/post_train_bench" "$TASK_CACHE_ROOT"
rm -f "$EVAL_DIR/metrics.json"
cp -a "$REPO_ROOT/." "$JOB_REPO/"
rm -rf "$JOB_REPO/scratch/PostTrainBench" "$JOB_REPO/post_train_bench/runs"
cp "$REPO_ROOT/post_train_bench/integrity.py" "$TRUSTED_INTEGRITY"
cp "$REPO_ROOT/post_train_bench/collect_artifacts.py" "$TRUSTED_COLLECT"
cp "$REPO_ROOT/post_train_bench/run_judge.py" "$JOB_JUDGE/run_judge.py"
seed_cache "$SOLVE_HF_CACHE"
seed_cache "$EVAL_HF_CACHE"

exec > >(tee "$EVAL_DIR/output.log")
exec 2> >(tee "$EVAL_DIR/error.log" >&2)

echo "benchmark=$BENCHMARK"
echo "model_to_train=$MODEL_TO_TRAIN"
echo "agent_model=$ML_INTERN_AGENT_MODEL"
echo "task_run_id=$TASK_RUN_ID"
echo "num_hours=$NUM_HOURS"
echo "duration_minutes=$DURATION_MINUTES"
echo "duration_seconds=$DURATION_SECONDS"
echo "solve_timeout_seconds=$SOLVE_TIMEOUT_SECONDS"
echo "eval_limit=$EVAL_LIMIT"
echo "solve_docker_image=$SOLVE_DOCKER_IMAGE"
echo "eval_docker_image=$EVAL_DOCKER_IMAGE"
echo "baseline_final_model=${POST_TRAIN_BENCH_BASELINE_FINAL_MODEL:-0}"
echo "reprompt=$REPROMPT"
echo "reprompt_min_minutes=$REPROMPT_MIN_MINUTES"
echo "method_dir=$METHOD_DIR"
echo "seed_hf_cache=$SEED_HF_CACHE"
echo "solve_hf_cache=$SOLVE_HF_CACHE"
echo "eval_hf_cache=$EVAL_HF_CACHE"
echo "prompt_agent=$PROMPT_AGENT"

cp "$PTB_DIR/src/eval/tasks/${BENCHMARK}/evaluate.py" "$JOB_DIR/task/"
if [ -d "$PTB_DIR/src/eval/tasks/${BENCHMARK}/evaluation_code" ]; then
    cp -r "$PTB_DIR/src/eval/tasks/${BENCHMARK}/evaluation_code" "$JOB_DIR/task/"
fi
cp -r "$PTB_DIR/src/eval/templates" "$JOB_DIR/task/"
if [ -d "$PTB_DIR/src/eval/tasks/${BENCHMARK}/task_context" ]; then
    cp -r "$PTB_DIR/src/eval/tasks/${BENCHMARK}/task_context/." "$JOB_DIR/task/"
fi
find "$JOB_DIR/task" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$JOB_DIR/task" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
python3 "$TRUSTED_INTEGRITY" snapshot-protected-files \
    --task-dir "$JOB_DIR/task" \
    --output "$EVAL_DIR/protected_files_manifest.json"

BENCHMARK_NAME="$(cat "$PTB_DIR/src/eval/tasks/${BENCHMARK}/benchmark.txt")"
PROMPT="$(
    cd "$PTB_DIR"
    POST_TRAIN_BENCH_PROMPT="${POST_TRAIN_BENCH_PROMPT:-prompt}" \
        python3 src/eval/general/get_prompt.py \
            --model-to-train "$MODEL_TO_TRAIN" \
            --benchmark-id "$BENCHMARK" \
            --num-hours "$NUM_HOURS" \
            --num-gpus 1 \
            --agent "$PROMPT_AGENT"
)"
printf '%s\n' "$PROMPT" > "$EVAL_DIR/prompt.txt"
export PROMPT

CREATION_DATE="$(date +%s)"
cat > "$JOB_DIR/task/timer.sh" <<TIMER
#!/bin/bash

CREATION_DATE=${CREATION_DATE}
DURATION_SECONDS=${DURATION_SECONDS}

DEADLINE=\$((CREATION_DATE + DURATION_SECONDS))
NOW=\$(date +%s)
REMAINING=\$((DEADLINE - NOW))

if [ \$REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=\$((REMAINING / 3600))
    MINUTES=\$(((REMAINING % 3600) / 60))
    printf "%d:%02d\n" \$HOURS \$MINUTES
fi
TIMER
chmod +x "$JOB_DIR/task/timer.sh"

SOLVE_CONTAINER_MOUNTS="${JOB_REPO}:/ml-intern-src:ro,${JOB_DIR}:/workspace,${JOB_TMP}:/tmp,${SOLVE_HF_CACHE}:/hf-cache"
JUDGE_CONTAINER_MOUNTS="${JOB_JUDGE}:/judge:ro,${JUDGE_EVIDENCE_DIR}/task:/workspace/task:ro,${EVAL_DIR}:/result,${JOB_TMP}:/tmp"
EVAL_CONTAINER_MOUNTS="${PTB_DIR}:/posttrainbench,${EVAL_DIR}:/result,${JOB_TMP}:/tmp,${EVAL_HF_CACHE}:/hf-cache"
VALIDATION_CONTAINER_MOUNTS="${EVAL_DIR}/final_model:/final_model:ro,${JOB_TMP}:/tmp,${EVAL_HF_CACHE}:/hf-cache"
SOLVE_PROVIDER_ENV=""
case "$ML_INTERN_AGENT_MODEL" in
    anthropic/*|claude*) SOLVE_PROVIDER_ENV=",ANTHROPIC_API_KEY" ;;
    openai/*|gpt-*|o1*|o3*|o4*|o5*) SOLVE_PROVIDER_ENV=",OPENAI_API_KEY" ;;
    google/*|gemini*) SOLVE_PROVIDER_ENV=",GEMINI_API_KEY" ;;
esac
SOLVE_CONTAINER_ENV="POST_TRAIN_BENCH_SOLVE_HF_TOKEN,HUGGING_FACE_HUB_READ_TOKEN,POST_TRAIN_BENCH_TAMPER_EVALUATE,POST_TRAIN_BENCH_REPROMPT,POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES${SOLVE_PROVIDER_ENV},ML_INTERN_AGENT_MODEL,PROMPT,TRACKIO_PROJECT,TRACKIO_SPACE_ID"
JUDGE_CONTAINER_ENV="OPENAI_API_KEY,PTB_JUDGE_MODEL"
EVAL_CONTAINER_ENV="HF_TOKEN,HUGGING_FACE_HUB_TOKEN,OPENAI_API_KEY,INFERENCE_TOKEN,HF_BILL_TO"

echo "solve_container_mounts=$SOLVE_CONTAINER_MOUNTS"
echo "judge_container_mounts=$JUDGE_CONTAINER_MOUNTS"
echo "eval_container_mounts=$EVAL_CONTAINER_MOUNTS"
echo "validation_container_mounts=$VALIDATION_CONTAINER_MOUNTS"

run_judge_container() {
    srun \
        --no-container-mount-home \
        --container-image="$SOLVE_DOCKER_IMAGE" \
        --container-mounts="$JUDGE_CONTAINER_MOUNTS" \
        --container-workdir=/workspace/task \
        --container-env="$JUDGE_CONTAINER_ENV" \
        "$@"
}

run_eval_container() {
    srun \
        --no-container-mount-home \
        --container-image="$EVAL_DOCKER_IMAGE" \
        --container-mounts="$EVAL_CONTAINER_MOUNTS" \
        --container-workdir=/posttrainbench/src/eval/tasks/"$BENCHMARK" \
        --container-env="$EVAL_CONTAINER_ENV" \
        "$@"
}

run_validation_container() {
    srun \
        --no-container-mount-home \
        --container-image="$EVAL_DOCKER_IMAGE" \
        --container-mounts="$VALIDATION_CONTAINER_MOUNTS" \
        --container-workdir=/tmp \
        --container-env="HF_TOKEN,HUGGING_FACE_HUB_TOKEN" \
        "$@"
}

create_baseline_final_model() {
    if [ "${POST_TRAIN_BENCH_BASELINE_FINAL_MODEL:-0}" != "1" ]; then
        return
    fi
    if [ -d "$JOB_DIR/task/final_model" ]; then
        return
    fi
    echo "Creating smoke-only baseline final_model from $MODEL_TO_TRAIN"
    export MODEL_TO_TRAIN
    set +e
    srun \
        --no-container-mount-home \
        --container-image="$EVAL_DOCKER_IMAGE" \
        --container-mounts="${JOB_DIR}:/workspace,${JOB_TMP}:/tmp,${EVAL_HF_CACHE}:/hf-cache" \
        --container-workdir=/workspace/task \
        --container-env="MODEL_TO_TRAIN,POST_TRAIN_BENCH_SOLVE_HF_TOKEN,HUGGING_FACE_HUB_READ_TOKEN" \
        bash -lc '
            set -euo pipefail
            export HF_HOME=/hf-cache
            export PYTHONNOUSERSITE=1
            if [ -n "${POST_TRAIN_BENCH_SOLVE_HF_TOKEN:-}" ]; then
                export HF_TOKEN="$POST_TRAIN_BENCH_SOLVE_HF_TOKEN"
                export HUGGING_FACE_HUB_TOKEN="$POST_TRAIN_BENCH_SOLVE_HF_TOKEN"
            elif [ -n "${HUGGING_FACE_HUB_READ_TOKEN:-}" ]; then
                export HF_TOKEN="$HUGGING_FACE_HUB_READ_TOKEN"
                export HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_READ_TOKEN"
            fi
            python - <<'"'"'PY'"'"'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = os.environ["MODEL_TO_TRAIN"]
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained("final_model", safe_serialization=True)
tokenizer.save_pretrained("final_model")
print(f"saved baseline final_model from {model_id}")
PY
        ' > "$EVAL_DIR/baseline_final_model.txt" 2>&1
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        echo "Smoke baseline final_model creation failed; see $EVAL_DIR/baseline_final_model.txt" >&2
    fi
}

FINALIZED=0
SECRET_SCAN_FAILED=0

write_integrity_status() {
    python3 "$TRUSTED_INTEGRITY" write-status \
        --status "$1" \
        --reason "$2" \
        --output "$EVAL_DIR/integrity_status.json"
}

snapshot_evidence() {
    python3 "$TRUSTED_INTEGRITY" snapshot-evidence \
        --task-dir "$JOB_DIR/task" \
        --eval-dir "$EVAL_DIR" \
        --output "$EVAL_DIR/evidence_snapshot.json"
}

prepare_judge_evidence() {
    rm -rf "$JUDGE_EVIDENCE_DIR"
    mkdir -p "$JUDGE_EVIDENCE_DIR"
    snapshot_evidence
    cp -a "$EVAL_DIR/task" "$JUDGE_EVIDENCE_DIR/task"
    if [ -d "$EVAL_DIR/final_model" ]; then
        cp -a "$EVAL_DIR/final_model" "$JUDGE_EVIDENCE_DIR/task/final_model"
    fi
}

finalize_run() {
    if [ "$FINALIZED" -eq 1 ]; then
        return
    fi
    FINALIZED=1
    snapshot_evidence || true
    if ! python3 "$TRUSTED_INTEGRITY" scan-secrets \
        --path "$EVAL_DIR" \
        --output "$EVAL_DIR/secret_scan.json"; then
        SECRET_SCAN_FAILED=1
        if [ ! -s "$EVAL_DIR/integrity_status.json" ]; then
            write_integrity_status invalid "secret scan found unredacted secrets" || true
        fi
        echo "Secret scan found unredacted secrets; see $EVAL_DIR/secret_scan.json" >&2
    fi
    python3 "$TRUSTED_COLLECT" \
        --run-root "$RUN_ROOT" \
        --eval-dir "$EVAL_DIR" \
        --benchmark "$BENCHMARK" \
        --model-to-train "$MODEL_TO_TRAIN" \
        --task-run-id "$TASK_RUN_ID" \
        --method "$METHOD_DIR" || true
}

fail_run() {
    local code="$1"
    shift
    if [ "$#" -gt 0 ]; then
        echo "$*" >&2
    fi
    finalize_run
    exit "$code"
}

SOLVE_LOG_TS="$(date -u +%Y%m%dT%H%M%SZ)"
SOLVE_OUT="$EVAL_DIR/solve_out_${SOLVE_LOG_TS}.txt"

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

start_system_monitor
HOST_START_TS="$(date --iso-8601=seconds)"
export SOLVE_TIMEOUT_SECONDS
set +e
srun \
    --no-container-mount-home \
    --container-image="$SOLVE_DOCKER_IMAGE" \
    --container-mounts="$SOLVE_CONTAINER_MOUNTS" \
    --container-workdir=/workspace/task \
    --container-env="$SOLVE_CONTAINER_ENV,SOLVE_TIMEOUT_SECONDS" \
    bash -lc '
        set -euo pipefail
        export HF_HOME=/hf-cache
        export PYTHONNOUSERSITE=1
        export PYTHONPATH=/ml-intern-src:${PYTHONPATH:-}
        export PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
        if [ -n "${POST_TRAIN_BENCH_SOLVE_HF_TOKEN:-}" ]; then
            export HF_TOKEN="$POST_TRAIN_BENCH_SOLVE_HF_TOKEN"
            export HUGGING_FACE_HUB_TOKEN="$POST_TRAIN_BENCH_SOLVE_HF_TOKEN"
        elif [ -n "${HUGGING_FACE_HUB_READ_TOKEN:-}" ]; then
            export HF_TOKEN="$HUGGING_FACE_HUB_READ_TOKEN"
            export HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_READ_TOKEN"
        fi
        rm -rf /tmp/ml-intern-install-src
        cp -a /ml-intern-src /tmp/ml-intern-install-src
        cd /tmp/ml-intern-install-src
        uv pip install --system .
        cd /
        rm -rf /tmp/ml-intern-install-src
        cd /workspace/task
        date --iso-8601=seconds > /tmp/solve_start.txt
        set +e
        if [ "${POST_TRAIN_BENCH_TAMPER_EVALUATE:-0}" = "1" ]; then
            printf "\n# tampered by negative smoke\n" >> evaluate.py
            status=0
        else
            timeout --signal=TERM --kill-after=30s "${SOLVE_TIMEOUT_SECONDS}s" \
                python -m agent.main \
                    --config /ml-intern-src/post_train_bench/ml_intern_config.json \
                    --model "$ML_INTERN_AGENT_MODEL" \
                    --max-iterations -1 \
                    "$PROMPT"
            status=$?
        fi
        set -e
        printf "%s\n" "$status" > /tmp/solve_exit.txt
        date --iso-8601=seconds > /tmp/solve_end.txt
        exit "$status"
    ' > "$SOLVE_OUT" 2>&1
SRUN_EXIT=$?
set -e
HOST_END_TS="$(date --iso-8601=seconds)"
SOLVE_EXIT="$SRUN_EXIT"
if [ -s "$JOB_TMP/solve_exit.txt" ]; then
    SOLVE_EXIT="$(cat "$JOB_TMP/solve_exit.txt")"
fi
START_TS="$HOST_START_TS"
if [ -s "$JOB_TMP/solve_start.txt" ]; then
    START_TS="$(cat "$JOB_TMP/solve_start.txt")"
fi
END_TS="$HOST_END_TS"
if [ -s "$JOB_TMP/solve_end.txt" ]; then
    END_TS="$(cat "$JOB_TMP/solve_end.txt")"
fi
cp "$SOLVE_OUT" "$EVAL_DIR/solve_out.txt"
cp "$SOLVE_OUT" "$JOB_DIR/task/solve_out.txt"
printf '%s\n' "$SOLVE_EXIT" > "$EVAL_DIR/solve_exit.txt"
python3 - "$START_TS" "$END_TS" "$EVAL_DIR/time_taken.txt" <<'PY'
import datetime as dt
import sys

start = dt.datetime.fromisoformat(sys.argv[1])
end = dt.datetime.fromisoformat(sys.argv[2])
seconds = int((end - start).total_seconds())
with open(sys.argv[3], "w", encoding="utf-8") as f:
    f.write(f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}\n")
PY

echo "solve_exit=$SOLVE_EXIT"

snapshot_evidence || true

if ! python3 "$TRUSTED_INTEGRITY" verify-protected-files \
    --task-dir "$JOB_DIR/task" \
    --manifest "$EVAL_DIR/protected_files_manifest.json" \
    --output "$EVAL_DIR/protected_files_check.json"; then
    write_integrity_status invalid "protected benchmark files changed during solve"
    fail_run 1 "Protected benchmark files changed during solve; see $EVAL_DIR/protected_files_check.json"
fi
create_baseline_final_model
snapshot_evidence || true

echo "========================================="
echo "=== RUNNING CONTAMINATION JUDGE ========"
echo "========================================="

JUDGE_PROMPT="$(
    cd "$PTB_DIR"
    python3 src/disallowed_usage_judge/get_judge_prompt.py \
        --benchmark "$BENCHMARK_NAME" \
        --model "$MODEL_TO_TRAIN"
)"
printf '%s\n' "$JUDGE_PROMPT" > "$EVAL_DIR/judge_prompt.txt"
prepare_judge_evidence

set +e
run_judge_container python /judge/run_judge.py \
    --task-dir /workspace/task \
    --prompt-file /result/judge_prompt.txt \
    --output-dir /result > "$EVAL_DIR/judge_output.txt" 2>&1
JUDGE_EXIT=$?
set -e
echo "judge_exit=$JUDGE_EXIT"
if [ "$JUDGE_EXIT" -ne 0 ]; then
    write_integrity_status judge_failed "judge process exited with status $JUDGE_EXIT"
    fail_run "$JUDGE_EXIT"
fi
for required_judgement in contamination_judgement.txt disallowed_model_judgement.txt; do
    if [ ! -s "$EVAL_DIR/$required_judgement" ]; then
        echo "Missing required judge output: $required_judgement" >&2
        write_integrity_status judge_failed "missing required judge output: $required_judgement"
        fail_run 1
    fi
done
if ! python3 "$TRUSTED_INTEGRITY" judge-status \
    --eval-dir "$EVAL_DIR" \
    --output "$EVAL_DIR/integrity_status.json"; then
    fail_run 1 "Integrity judge did not return a clean verdict; see $EVAL_DIR/integrity_status.json"
fi

rm -rf "$JOB_DIR/task/final_model"
snapshot_evidence || true

validate_final_model() {
    echo "================================"
    echo "==== VALIDATING FINAL MODEL ===="
    echo "================================"
    set +e
    python3 "$TRUSTED_INTEGRITY" precheck-final-model \
        --model-path "$EVAL_DIR/final_model" \
        --base-model "$MODEL_TO_TRAIN" \
        --output "$EVAL_DIR/final_model_precheck.json"
    local precheck_status=$?
    set -e
    if [ "$precheck_status" -ne 0 ]; then
        write_integrity_status invalid "final model precheck failed"
        fail_run "$precheck_status" "Final model precheck failed; see $EVAL_DIR/final_model_precheck.json"
    fi
    set +e
    run_validation_container bash -lc '
        set -euo pipefail
        export HF_HOME=/hf-cache
        export PYTHONNOUSERSITE=1
        python - <<'"'"'PY'"'"'
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

model_path = Path("/final_model")
if not model_path.is_dir():
    raise SystemExit("final_model directory is missing")
if not (model_path / "config.json").is_file():
    raise SystemExit("final_model/config.json is missing")
AutoConfig.from_pretrained(model_path, local_files_only=True)
AutoTokenizer.from_pretrained(model_path, local_files_only=True)
print("final_model validation passed")
PY
    ' > "$EVAL_DIR/final_model_validation.txt" 2>&1
    local status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        fail_run "$status" "Final model validation failed; see $EVAL_DIR/final_model_validation.txt"
    fi
}

validate_final_model
rm -f "$EVAL_DIR/metrics.json"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"
    local metrics_candidate="/tmp/metrics_candidate_${eval_num}.json"
    local host_metrics_candidate="${JOB_TMP}/metrics_candidate_${eval_num}.json"
    rm -f "$host_metrics_candidate" "$EVAL_DIR/metrics.json"
    set +e
    run_eval_container bash -lc "
        set -euo pipefail
        export HF_HOME=/hf-cache
        export PYTHONNOUSERSITE=1
        export VLLM_API_KEY=inspectai
        python evaluate.py \
            --model-path /result/final_model \
            --templates-dir ../../../../src/eval/templates \
            --limit ${EVAL_LIMIT} \
            ${max_tokens_arg} \
            --json-output-file ${metrics_candidate}
    " > "$EVAL_DIR/final_eval_${eval_num}.txt" 2>&1
    local status=$?
    set -e
    if [ "$status" -eq 0 ] && [ -s "$host_metrics_candidate" ]; then
        mv "$host_metrics_candidate" "$EVAL_DIR/metrics.json"
        return 0
    fi
    rm -f "$host_metrics_candidate"
    if [ "$status" -eq 0 ]; then
        echo "Evaluation attempt $eval_num exited successfully but did not write metrics" >&2
        return 1
    fi
    return "$status"
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"
    local attempt
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"
        run_evaluation "$max_tokens_arg" "$EVAL_COUNTER" || true
        if [ -f "$EVAL_DIR/metrics.json" ]; then
            return 0
        fi
    done
    return 1
}

EVAL_COUNTER=0
run_evaluation_with_retry 4 "" || true

case "$BENCHMARK" in
    aime2025|bfcl|gpqamain) MAX_TOKENS_ARG="--max-tokens 12000" ;;
    gsm8k|humaneval) MAX_TOKENS_ARG="--max-tokens 3000" ;;
    arenahardwriting|healthbench) MAX_TOKENS_ARG="--max-new-tokens 12288" ;;
    *) MAX_TOKENS_ARG="" ;;
esac
run_evaluation_with_retry 3 "$MAX_TOKENS_ARG" || true

case "$BENCHMARK" in
    aime2025|bfcl|gpqamain) MAX_TOKENS_ARG="--max-tokens 8000" ;;
    gsm8k|humaneval) MAX_TOKENS_ARG="--max-tokens 2000" ;;
    arenahardwriting|healthbench) MAX_TOKENS_ARG="--max-new-tokens 8192" ;;
    *) MAX_TOKENS_ARG="" ;;
esac
run_evaluation_with_retry 2 "$MAX_TOKENS_ARG" || true

if [ ! -f "$EVAL_DIR/metrics.json" ]; then
    write_integrity_status invalid "evaluation failed after all retry phases"
    fail_run 1 "Evaluation failed after all retry phases"
fi

finalize_run
if [ "$SECRET_SCAN_FAILED" -ne 0 ]; then
    exit 1
fi

if [ "$SOLVE_EXIT" -ne 0 ] && [ "$SOLVE_EXIT" -ne 124 ]; then
    exit "$SOLVE_EXIT"
fi
