---
description: Fine-tune a model on a dataset, end-to-end (research → validate → train → push).
argument-hint: <natural language task, e.g. "llama-3-8b on HuggingFaceH4/ultrachat_200k">
---

Fine-tune the model described in: $ARGUMENTS

Fine-tuning is never trivial. Follow this sequence in order. Do **not** skip steps even if the request looks simple — `CLAUDE.md` lists the specific failures that happen when you do.

**1. Research first (mandatory).** Delegate to the `research` subagent via the Task tool with `subagent_type: "research"`. Brief it:

> Find the best fine-tuning recipe for: $ARGUMENTS.
> Identify the model architecture and intended task. Crawl the citation graph for recent papers that fine-tuned this (or a comparable) model on this (or a comparable) dataset. Read methodology sections (3, 4, 5) of the top 3 candidates. Extract: training method (SFT/DPO/GRPO/...), exact hyperparameters (lr, schedule, epochs, batch size, optimizer, max_length), and any data preprocessing. Verify the dataset's HF Hub format with `hf_inspect_dataset`. Return a ranked recipe table per CLAUDE.md.

Do not start writing code until the subagent returns.

**2. Validate dataset and model.** Independently of the research output, run:
- `mcp__ml-intern-tools__hf_inspect_dataset` on the target dataset — confirm columns match the chosen training method (SFT: `messages`/`text`/`prompt`+`completion`; DPO: `prompt`+`chosen`+`rejected`; GRPO: `prompt`).
- `mcp__ml-intern-tools__hf_repo_files` on the target model — confirm it exists and note tokenizer/architecture.

**3. Develop in a sandbox.** For non-trivial scripts, call `mcp__ml-intern-tools__sandbox_create` with a GPU flavor (`t4-small` minimum if the code touches CUDA/bf16/model loading). Write the script, install deps, run a tiny smoke test (1–2 steps), fix errors. Do not skip the smoke test.

**4. Pre-flight check (mandatory output before `hf_jobs`).** Print this checklist and verify every line is filled:

```
Reference implementation: <path or arxiv ID from research>
Dataset format verified:  <columns confirmed via hf_inspect_dataset>
Training method:          <SFT | DPO | GRPO | ...>
Hyperparameters:          <lr, schedule, epochs, batch size, max_length>
push_to_hub:              True
hub_model_id:             <org/name>
hardware_flavor:          <from sizing table in CLAUDE.md>
timeout:                  <≥ 2h for any training>
Trackio monitoring:       <project name + dashboard URL>
disable_tqdm=True, logging_strategy="steps", logging_first_step=True: yes
```

If any line is missing, **stop and complete it** before submitting.

**5. Submit ONE job.** Call `mcp__ml-intern-tools__hf_jobs` (operation `run` or `uv`) with the verified config. Watch the first 60s of logs to confirm training started (loss values printing as plain text, not stuck on tokenizer/model load). Only then submit any sweep/ablation runs.

**6. Report.** Provide:
- Direct Hub URL of the job (`https://huggingface.co/jobs/...`)
- Trackio dashboard URL
- Hub URL of the model that will appear on completion (`https://huggingface.co/<hub_model_id>`)

If anything fails, do not silently switch training methods, reduce `max_length`, or substitute datasets. Diagnose, fix the minimal thing, or ask the user.
