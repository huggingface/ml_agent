---
description: Force a literature-first research crawl — delegates immediately to the `research` subagent without doing anything else.
argument-hint: <topic, paper, or task to research>
---

Delegate this research task to the `research` subagent **immediately**. Do not
attempt the research yourself — the subagent has its own context window and
returns a structured recipe table.

Use the Task tool with `subagent_type: "research"`. Brief:

> Literature crawl for: $ARGUMENTS
>
> Start from anchor paper(s). Crawl citation graph for recent downstream
> papers. Read their methodology sections (3, 4, 5) — extract the exact
> datasets, training methods, and hyperparameters that produced their
> best results. Attribute every finding to a specific result. Also find
> working code examples using current TRL/Transformers APIs. Validate
> any datasets via `hf_inspect_dataset`.

When the subagent returns, summarize the top recipe to the user with direct
HF Hub URLs and the arxiv ID of the source paper.
