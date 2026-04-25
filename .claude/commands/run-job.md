---
description: Submit an HF Job (training, eval, batch inference) with the ml-intern pre-flight checklist.
argument-hint: <description of the job to run>
---

Submit an HF Job for: $ARGUMENTS

Before calling `mcp__ml-intern-tools__hf_jobs`, produce the pre-flight check below. **Do not call `hf_jobs` until every line is filled in.** If you cannot fill a line, complete the missing step (research, dataset inspection, sandbox test) first.

```
Job purpose:              <training | eval | batch inference | data prep | other>
Reference implementation: <example file or arxiv ID this is based on>
Dataset format verified:  <columns confirmed via hf_inspect_dataset, or N/A>
Model verified:           <hub repo confirmed, or N/A>
push_to_hub:              <True + hub_model_id, or N/A for non-training jobs>
hardware_flavor:          <from sizing table below>
timeout:                  <value>
Trackio monitoring:       <project + dashboard URL, or N/A>
Packages to install:      <flash-attn, bitsandbytes, etc. — anything not preinstalled>
```

**Hardware sizing** (from `CLAUDE.md`):
- 1–3B params → `a10g-largex2`
- 7–13B params → `a100-large`
- 30B+ params → `l40sx4` or `a100x4`
- 70B+ params → `a100x8`
- CPU-only data prep → `cpu-basic` or `cpu-upgrade`

Note: `a10g-small` and `a10g-large` have the SAME 24GB GPU memory — the difference is CPU/RAM only.

**Timeout floor:** for any training job, set timeout ≥ `2h`. The default 30m kills training. If your timeout is < 2h and the job is training, **stop and revise** unless the user explicitly justified a shorter run (e.g. a smoke test).

**Hooks will gate this call:** GPU jobs always prompt for confirmation. CPU jobs prompt by default (override with `ML_INTERN_CONFIRM_CPU_JOBS=0`). That is expected — present the pre-flight check clearly so the user can approve in one read.

**For batch / ablation work:** submit ONE job first. Watch the first ~60 seconds of logs (look for plain-text loss lines — `disable_tqdm=True, logging_strategy="steps", logging_first_step=True` should be set). Only after that one starts training successfully, submit the rest. Never submit all at once.

**After submission, report:**
- Job URL (`https://huggingface.co/jobs/...`)
- Trackio dashboard URL
- Expected output (model repo, dataset repo, eval scores file path) and where to find it after completion
