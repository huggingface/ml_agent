# PostTrainBench Evaluation

This directory contains the Slurm/Docker integration for evaluating `ml-intern`
on PostTrainBench with local H100 compute.

All run outputs are written under:

```bash
post_train_bench/runs/{ML_INTERN_AGENT_MODEL}/{RUN_ID}/
```

`ML_INTERN_AGENT_MODEL` is used literally as a path. For example,
`anthropic/claude-opus-4-6` writes under
`post_train_bench/runs/anthropic/claude-opus-4-6/...`.

`RUN_ID` is generated once per evaluation set as:

```text
YYYY-MM-DD_HH-MM-SS_{slurm_job_id}
```

The submitter gets the Slurm job id by submitting the array held, writes the
final run directory and metadata, then releases the job. Dry runs use a
`YYYY-MM-DD_HH-MM-SS_dryrun` suffix because no Slurm job id exists.

## Prerequisites

- A local PostTrainBench checkout is available. The default path is
  `scratch/PostTrainBench`; override it with `POST_TRAIN_BENCH_DIR`.
- Slurm with Pyxis container support is available.
- The current checkout contains the `ml-intern` commit you want to evaluate.
- Required tokens are exported. The solve phase receives only
  `POST_TRAIN_BENCH_SOLVE_HF_TOKEN` or `HUGGING_FACE_HUB_READ_TOKEN`; use a
  read-only token there. The eval phase can still use the normal evaluation
  tokens.

```bash
export POST_TRAIN_BENCH_SOLVE_HF_TOKEN=hf_...  # read-only
export HF_TOKEN=hf_...                         # eval-only
export ANTHROPIC_API_KEY=sk-ant-...   # or the provider key for ML_INTERN_AGENT_MODEL
export OPENAI_API_KEY=sk-...          # used by Arena/Health evals and required Codex judge
export ML_INTERN_AGENT_MODEL=anthropic/claude-opus-4-6  # optional; this is the default
```

The runner uses separate solve/judge and eval images. The default images are:

```bash
export POST_TRAIN_BENCH_DOCKER_IMAGE=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest
export POST_TRAIN_BENCH_EVAL_DOCKER_IMAGE=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench-eval:latest
```

The solve phase uses a fresh per-task HF cache seeded from:

```bash
export POST_TRAIN_BENCH_SEED_HF_CACHE=/fsx/lewis/post_train_bench/seed_hf_cache
```

Override the path if the cluster seed cache moves.

## Smoke Test

Submit one 10-minute GSM8K / Qwen3-1.7B job:

```bash
bash post_train_bench/submit_eval_set.sh smoke
```

The smoke mode is meant to validate the Slurm, Docker, agent launch, artifact
collection, judge, and evaluation plumbing quickly. It is not a faithful
quality estimate; use the full matrix for leaderboard runs.

Smoke uses a 10-minute solve budget, evaluates 8 GSM8K samples, and requests a
1-hour Slurm allocation by default so the judge, evaluation, and artifact
collection have room to finish. Override the scheduler allocation with:

```bash
export POST_TRAIN_BENCH_SLURM_TIME=00:30:00
```

Smoke mode defaults `POST_TRAIN_BENCH_BASELINE_FINAL_MODEL=1`. If the agent
does not leave a `final_model`, the runner creates a base-model `final_model`
after the protected-file check so the judge, validation, evaluation, artifact
collection, and hash reporting paths are still exercised. Validation and full
modes default this fallback off.

To check paths and metadata without submitting:

```bash
bash post_train_bench/submit_eval_set.sh smoke --dry-run
```

Monitor with:

```bash
squeue -u "$USER"
tail -f post_train_bench/runs/${ML_INTERN_AGENT_MODEL}/*/slurm/*.out
```

After completion, inspect:

```bash
find post_train_bench/runs/${ML_INTERN_AGENT_MODEL} -maxdepth 4 -type f | sort
```

## Artifact Validation Matrix

To check final-model artifact creation once per full-matrix base model, run:

```bash
bash post_train_bench/submit_eval_set.sh model-validation --dry-run
bash post_train_bench/submit_eval_set.sh model-validation
```

This submits one 2-hour GSM8K job with a small eval limit for each full-matrix
model: Gemma 3 4B, Qwen3 4B, Qwen3 1.7B, and SmolLM3 3B.

Before launching the full matrix, run the strict 4-job validation matrix:

```bash
bash post_train_bench/submit_eval_set.sh validation --dry-run
bash post_train_bench/submit_eval_set.sh validation
```

Validation uses 2-hour solve budgets with small eval limits for:

```text
humaneval + Qwen/Qwen3-1.7B-Base
gsm8k     + Qwen/Qwen3-1.7B-Base
bfcl      + Qwen/Qwen3-1.7B-Base
gsm8k     + google/gemma-3-4b-pt
```

`POST_TRAIN_BENCH_BASELINE_FINAL_MODEL` defaults to `0` in validation mode.
Treat the run as an artifact-validity gate: inspect `final_model_precheck.json`
and require at least 3 of 4 clean `final_model` prechecks before a full
non-reprompt Claude run.

Reprompting is an explicit method variant and is off by default:

```bash
export POST_TRAIN_BENCH_REPROMPT=1
export POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES=30
bash post_train_bench/submit_eval_set.sh validation
```

Reprompted runs write under method directories with a `_reprompt` suffix and
record `reprompt_enabled`, `reprompt_min_minutes`, and `method_variant` in
`run_metadata.json`. Compare them only against other reprompted-method runs.

## Run Layout

A completed run has this shape:

```text
post_train_bench/runs/{ML_INTERN_AGENT_MODEL}/{RUN_ID}
|-- artifacts
|   `-- {method}
|       `-- {benchmark}_{model_to_train}_{slurm_array_task}
|           |-- manifest.json           # checksums, copied artifact summary, final_model file references
|           |-- metrics.json            # copied per-run benchmark metrics
|           `-- session_logs/           # copied local ml-intern trajectories
|-- env
|   `-- submit_env.txt                  # redacted submission-time environment snapshot
|-- results
|   `-- {method}
|       `-- {benchmark}_{model_to_train}_{slurm_array_task}
|           |-- contamination_judgement.txt
|           |-- disallowed_model_judgement.txt
|           |-- evidence_snapshot.json   # task/final_model capture status
|           |-- final_eval_*.txt        # raw evaluation attempts
|           |-- baseline_final_model.txt # smoke fallback creation log, if used
|           |-- final_model_precheck.json
|           |-- final_model_validation.txt
|           |-- final_model/            # model selected by the agent
|           |-- integrity_status.json   # clean, cheating, judge_failed, or invalid
|           |-- judge_output.txt        # judge runner stdout/stderr
|           |-- judge_prompt.txt        # prompt sent to the contamination judge
|           |-- judge_raw_response.txt  # raw judge model response, if available
|           |-- metrics.json            # benchmark score for this task
|           |-- output.log              # runner stdout
|           |-- error.log               # runner stderr
|           |-- prompt.txt              # PostTrainBench prompt given to ml-intern
|           |-- protected_files_check.json
|           |-- protected_files_manifest.json
|           |-- solve_out.txt           # raw ml-intern agent trace
|           |-- solve_out_*.txt         # timestamped raw ml-intern agent trace
|           |-- solve_exit.txt          # solve command exit status
|           |-- secret_scan.json        # unredacted-secret scan result
|           |-- system_monitor.log      # host CPU/GPU/disk monitor samples
|           |-- task/                   # task workspace captured after solve
|           |`-- time_taken.txt         # wall time for the solve phase
|-- slurm
|   |-- {job_id}_{array_id}.err         # Slurm wrapper stderr
|   `-- {job_id}_{array_id}.out         # Slurm wrapper stdout
|-- matrix.jsonl                        # benchmark/model rows for the array
|-- run_metadata.json                   # commit, image provenance/hashes, run id, dirty flag
|-- sbatch_command.txt                  # exact submission command
`-- sbatch_output.txt                   # Slurm job id and release output
```

Use `tree -L 5` on a specific run directory when you need a quick sanity check:

```bash
tree -L 5 post_train_bench/runs/${ML_INTERN_AGENT_MODEL}/{RUN_ID}
```

## Full Matrix

Do not run this until smoke succeeds and the strict validation matrix has at
least 3 of 4 clean `final_model` prechecks. This command submits the full
4-model x 7-benchmark matrix with 10 agent hours per job:

```bash
bash post_train_bench/submit_eval_set.sh full
```

Full mode refuses dirty worktrees and mutable registry tags by default. Use
digest-pinned images or local `.sqsh` images. The escape hatches
`--allow-dirty` and `--allow-mutable-images` are for internal experiments only.

To inspect the generated full matrix without submitting:

```bash
bash post_train_bench/submit_eval_set.sh full --dry-run
```

Full mode requests a 14-hour Slurm allocation by default. Set
`POST_TRAIN_BENCH_SLURM_TIME` before submission if the cluster queue or a
specific benchmark needs a different ceiling.

Matrix rows support only these fields:

```json
{"benchmark": "gsm8k", "model_to_train": "Qwen/Qwen3-1.7B-Base", "num_hours": "0.083", "eval_limit": 8}
```

`eval_limit` is optional. `duration_minutes` is intentionally invalid; the
runner derives the solve budget from `num_hours`.

Aggregate completed runs with the checked-in factor-weighted reporter:

```bash
uv run python post_train_bench/aggregate_results.py \
  post_train_bench/runs/${ML_INTERN_AGENT_MODEL}/{RUN_ID} \
  --output-json post_train_bench/runs/${ML_INTERN_AGENT_MODEL}/{RUN_ID}/aggregate_report.json \
  --output-csv post_train_bench/runs/${ML_INTERN_AGENT_MODEL}/{RUN_ID}/aggregate_report.csv
```

Pass multiple run roots to report multi-run mean, standard deviation, standard
error, min, and max for each method. Non-clean integrity statuses are reported
explicitly and are not silently converted into benchmark scores.

## Rebuilding The Docker Image

The checked-in Dockerfiles build the solve/judge image and eval-only image.
The solve/judge image includes Codex CLI for the required contamination and
disallowed-model-use judge. The eval image installs the pinned benchmark stack,
`inspect_evals@06001a83`, and `inspect_ai_vllm_stdout`.

Build locally:

```bash
bash post_train_bench/build_container.sh \
  --sqsh-output /fsx/lewis/docker_images/posttrainbench.sqsh

bash post_train_bench/build_container_eval.sh \
  --sqsh-output /fsx/lewis/docker_images/posttrainbench-eval.sqsh
```

Push to the cluster registry:

```bash
docker push registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest
docker push registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench-eval:latest
```

Use a custom tag when testing dependency changes:

```bash
bash post_train_bench/build_container.sh \
  --image registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test
bash post_train_bench/build_container_eval.sh \
  --image registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench-eval:ptb-test
docker push registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test
docker push registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench-eval:ptb-test
export POST_TRAIN_BENCH_DOCKER_IMAGE=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test
export POST_TRAIN_BENCH_EVAL_DOCKER_IMAGE=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench-eval:ptb-test
```

You do not need to rebuild the image just to evaluate a different `ml-intern`
commit. The Slurm job copies the current checkout into a temporary solve
workspace, mounts it read-only, and installs it non-editably before the measured
solve timeout starts. The eval phase does not mount `/ml-intern-src` and does
not inherit solve-installed packages.

## Notes

- `post_train_bench/runs/` is ignored by Git.
- If `ML_INTERN_AGENT_MODEL` is unset, the runner uses
  `anthropic/claude-opus-4-6`.
- The run metadata records whether the source worktree was dirty at submission
  time. Commit intended changes before running official evaluations.
- The Codex judge is required. `contamination_judgement.txt` and
  `disallowed_model_judgement.txt` must both be present and nonempty before
  evaluation proceeds.
