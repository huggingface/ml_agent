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
- Required tokens are exported:

```bash
export HF_TOKEN=hf_...
export ANTHROPIC_API_KEY=sk-ant-...   # or the provider key for ML_INTERN_AGENT_MODEL
export OPENAI_API_KEY=sk-...          # used by Arena/Health evals and optional judge
export ML_INTERN_AGENT_MODEL=anthropic/claude-opus-4-6  # optional; this is the default
```

The default Docker image is:

```bash
registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest
```

Override it with:

```bash
export POST_TRAIN_BENCH_DOCKER_IMAGE=registry.../posttrainbench:your-tag
```

## Smoke Test

Submit one 5-minute GSM8K / Qwen3-1.7B job:

```bash
bash post_train_bench/submit_eval_set.sh smoke
```

The smoke mode is meant to validate the Slurm, Docker, agent launch, artifact
collection, judge, and evaluation plumbing quickly. It is not a faithful
quality estimate; use the full matrix for leaderboard runs.

Smoke uses a 5-minute solve budget, evaluates 8 GSM8K samples, and requests a
1-hour Slurm allocation by default so the judge, evaluation, and artifact
collection have room to finish. Override the scheduler allocation with:

```bash
export POST_TRAIN_BENCH_SLURM_TIME=00:30:00
```

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
|   `-- submit_env.txt                  # submission-time environment snapshot
|-- results
|   `-- {method}
|       `-- {benchmark}_{model_to_train}_{slurm_array_task}
|           |-- contamination_judgement.txt
|           |-- disallowed_model_judgement.txt
|           |-- final_eval_*.txt        # raw evaluation attempts
|           |-- final_model/            # model selected by the agent
|           |-- judge_output.txt        # judge runner stdout/stderr
|           |-- judge_prompt.txt        # prompt sent to the contamination judge
|           |-- judge_raw_response.txt  # raw judge model response, if available
|           |-- metrics.json            # benchmark score for this task
|           |-- output.log              # runner stdout
|           |-- error.log               # runner stderr
|           |-- prompt.txt              # PostTrainBench prompt given to ml-intern
|           |-- solve_out.txt           # raw ml-intern agent trace
|           |-- task/                   # task workspace captured after solve
|           |`-- time_taken.txt         # wall time for the solve phase
|-- slurm
|   |-- {job_id}_{array_id}.err         # Slurm wrapper stderr
|   `-- {job_id}_{array_id}.out         # Slurm wrapper stdout
|-- matrix.jsonl                        # benchmark/model rows for the array
|-- run_metadata.json                   # commit, Docker image, run id, dirty flag
|-- sbatch_command.txt                  # exact submission command
`-- sbatch_output.txt                   # Slurm job id and release output
```

Use `tree -L 5` on a specific run directory when you need a quick sanity check:

```bash
tree -L 5 post_train_bench/runs/${ML_INTERN_AGENT_MODEL}/{RUN_ID}
```

## Full Matrix

Do not run this until the smoke test succeeds. This command submits the full
4-model x 7-benchmark matrix with 10 agent hours per job:

```bash
bash post_train_bench/submit_eval_set.sh full
```

To inspect the generated full matrix without submitting:

```bash
bash post_train_bench/submit_eval_set.sh full --dry-run
```

Full mode requests a 14-hour Slurm allocation by default. Set
`POST_TRAIN_BENCH_SLURM_TIME` before submission if the cluster queue or a
specific benchmark needs a different ceiling.

## Rebuilding The Docker Image

The checked-in `post_train_bench/Dockerfile` mirrors the Dockerfile from the
`posttrain-bench` integration branch and pins the PostTrainBench-compatible ML
stack.

Build locally:

```bash
docker build -t registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest \
  -f post_train_bench/Dockerfile .
```

Push to the cluster registry:

```bash
docker push registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest
```

Use a custom tag when testing dependency changes:

```bash
docker build -t registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test \
  -f post_train_bench/Dockerfile .
docker push registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test
export POST_TRAIN_BENCH_DOCKER_IMAGE=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:ptb-test
```

You do not need to rebuild the image just to evaluate a different `ml-intern`
commit. The Slurm job mounts the current checkout into the container and
installs it at runtime.

## Notes

- `post_train_bench/runs/` is ignored by Git.
- If `ML_INTERN_AGENT_MODEL` is unset, the runner uses
  `anthropic/claude-opus-4-6`.
- The run metadata records whether the source worktree was dirty at submission
  time. Commit intended changes before running official evaluations.
- The optional judge writes `judge not run: ...` if `OPENAI_API_KEY` is not set
  or the judge API call fails.
