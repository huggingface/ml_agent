#!/usr/bin/env python3
"""Collect per-task PostTrainBench artifacts under a run-level artifacts dir."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HASHED_MODEL_SUFFIXES = {
    ".json",
    ".safetensors",
}
HASHED_MODEL_NAMES = {
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "adapter_config.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_hash_model_file(path: Path) -> bool:
    name = path.name
    if name in HASHED_MODEL_NAMES:
        return True
    if path.suffix.lower() in HASHED_MODEL_SUFFIXES:
        return True
    return name.startswith("tokenizer") or name.startswith("adapter_")


def copy_optional(src: Path, dst: Path, manifest: dict) -> None:
    if not src.exists():
        manifest["missing"].append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        ignore = shutil.ignore_patterns(
            "final_model",
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
            ".cache",
            "__pycache__",
        )
        shutil.copytree(src, dst, ignore=ignore)
        return
    shutil.copy2(src, dst)
    manifest["files"].append(
        {
            "path": str(dst),
            "bytes": dst.stat().st_size,
            "sha256": sha256(dst),
        }
    )


def record_optional_tree(src: Path, manifest: dict, key: str) -> None:
    if not src.exists():
        manifest["missing"].append(str(src))
        return
    for path in sorted(src.rglob("*")):
        if path.is_file():
            entry = {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
            if should_hash_model_file(path):
                entry["sha256"] = sha256(path)
            manifest[key].append(entry)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-to-train", required=True)
    parser.add_argument("--task-run-id", required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    eval_dir = Path(args.eval_dir)
    model_safe = args.model_to_train.replace("/", "_").replace(":", "_")
    dest = (
        run_root
        / "artifacts"
        / args.method
        / f"{args.benchmark}_{model_safe}_{args.task_run_id}"
    )
    dest.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "model_to_train": args.model_to_train,
        "task_run_id": args.task_run_id,
        "method": args.method,
        "eval_dir": str(eval_dir),
        "files": [],
        "referenced_files": [],
        "missing": [],
    }

    for name in [
        "prompt.txt",
        "solve_out.txt",
        "solve_exit.txt",
        "system_monitor.log",
        "output.log",
        "error.log",
        "time_taken.txt",
        "final_model_validation.txt",
        "baseline_final_model.txt",
        "final_model_precheck.json",
        "integrity_status.json",
        "protected_files_check.json",
        "protected_files_manifest.json",
        "evidence_snapshot.json",
        "metrics.json",
        "contamination_judgement.txt",
        "disallowed_model_judgement.txt",
        "judge_output.txt",
        "judge_prompt.txt",
        "codex_judge_prompt.txt",
        "judge_raw_response.txt",
    ]:
        copy_optional(eval_dir / name, dest / name, manifest)

    for path in sorted(eval_dir.glob("solve_out_*.txt")):
        copy_optional(path, dest / path.name, manifest)

    for path in sorted(eval_dir.glob("final_eval_*.txt")):
        copy_optional(path, dest / path.name, manifest)

    copy_optional(eval_dir / "task" / "session_logs", dest / "session_logs", manifest)
    copy_optional(eval_dir / "task", dest / "task_snapshot", manifest)
    record_optional_tree(eval_dir / "final_model", manifest, "referenced_files")

    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
