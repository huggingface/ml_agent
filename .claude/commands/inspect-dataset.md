---
description: Audit a HF dataset — schema, splits, sample rows, and red flags. Direct port of `hf_inspect_dataset`.
argument-hint: <dataset id, e.g. HuggingFaceH4/ultrachat_200k>
---

Inspect the dataset `$ARGUMENTS` using `mcp__ml-intern-tools__hf_inspect_dataset`.

Report back with:
- schema and column types
- number of rows per split
- 3 sample rows
- red flags: class imbalance, missing values, unexpected formats, duplicates
- training-method compatibility:
  - SFT-ready? (has `messages` / `text` / `prompt`+`completion`)
  - DPO-ready? (has `prompt` + `chosen` + `rejected`)
  - GRPO-ready? (has `prompt`)

Include the direct Hub URL: `https://huggingface.co/datasets/$ARGUMENTS`
