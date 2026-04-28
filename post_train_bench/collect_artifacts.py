#!/usr/bin/env python3
"""Collect per-task PostTrainBench artifacts under a run-level artifacts dir."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
            manifest[key].append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                }
            )


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
    dest = run_root / "artifacts" / args.method / f"{args.benchmark}_{model_safe}_{args.task_run_id}"
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
        "output.log",
        "error.log",
        "time_taken.txt",
        "metrics.json",
        "contamination_judgement.txt",
        "disallowed_model_judgement.txt",
        "judge_output.txt",
        "judge_raw_response.txt",
    ]:
        copy_optional(eval_dir / name, dest / name, manifest)

    for path in sorted(eval_dir.glob("final_eval_*.txt")):
        copy_optional(path, dest / path.name, manifest)

    copy_optional(eval_dir / "task" / "session_logs", dest / "session_logs", manifest)
    copy_optional(eval_dir / "task", dest / "task_snapshot", manifest)
    record_optional_tree(eval_dir / "final_model", manifest, "referenced_files")

    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
