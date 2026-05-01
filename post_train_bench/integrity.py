#!/usr/bin/env python3
"""Integrity checks used by the PostTrainBench Slurm runner."""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

JUDGEMENT_SPECS = {
    "contamination_judgement.txt": {
        "clean": "no contamination detected",
        "detected": "contamination detected",
    },
    "disallowed_model_judgement.txt": {
        "clean": "only allowed use detected",
        "detected": "disallowed use detected",
    },
}

BASE_MODEL_RULES = {
    "Qwen/Qwen3-1.7B-Base": {
        "family_markers": ["qwen"],
        "allowed_source_ids": ["qwen/qwen3-1.7b-base"],
        "disallowed_source_ids": [
            "qwen/qwen3-1.7b",
            "qwen/qwen3-1.7b-instruct",
        ],
    },
    "Qwen/Qwen3-4B-Base": {
        "family_markers": ["qwen"],
        "allowed_source_ids": ["qwen/qwen3-4b-base"],
        "disallowed_source_ids": [
            "qwen/qwen3-4b",
            "qwen/qwen3-4b-instruct-2507",
        ],
    },
    "google/gemma-3-4b-pt": {
        "family_markers": ["gemma"],
        "allowed_source_ids": ["google/gemma-3-4b-pt"],
        "disallowed_source_ids": ["google/gemma-3-4b-it"],
    },
    "HuggingFaceTB/SmolLM3-3B-Base": {
        "family_markers": ["smollm"],
        "allowed_source_ids": ["huggingfacetb/smollm3-3b-base"],
        "disallowed_source_ids": ["huggingfacetb/smollm3-3b"],
    },
}

MODEL_ID_RE = re.compile(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\b")
SECRET_ASSIGNMENT_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY)[A-Z0-9_]*)"
    r"\s*[:=]\s*([^\s\"']+)"
)
SECRET_VALUE_PATTERNS = [
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"sk-(?!ant-)[A-Za-z0-9_-]{40,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("github_token", re.compile(r"github_pat_[A-Za-z0-9_]{36,}")),
    ("aws_key_id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("bearer_token", re.compile(r"(?i)bearer\s+[A-Za-z0-9_.=-]{20,}")),
]
SCAN_SKIP_DIRS = {
    ".cache",
    ".git",
    "__pycache__",
    "final_model",
}
SCAN_SKIP_SUFFIXES = {
    ".bin",
    ".gguf",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
}
MAX_SCAN_BYTES = 10 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_model_id(value: str) -> str:
    return value.strip().rstrip("/").lower()


def load_json_file(path: Path) -> tuple[dict, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, None
    except json.JSONDecodeError as exc:
        return {}, f"{path.name} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return {}, f"{path.name} must contain a JSON object"
    return data, None


def snapshot_protected_files(task_dir: Path) -> dict:
    files = []
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(task_dir).as_posix()
        files.append(
            {
                "path": rel_path,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "created_at": utc_now(),
        "task_dir": str(task_dir),
        "files": files,
    }


def verify_protected_files(task_dir: Path, manifest_path: Path) -> dict:
    manifest, manifest_error = load_json_file(manifest_path)
    if manifest_error:
        return {
            "created_at": utc_now(),
            "status": "invalid",
            "reason": manifest_error,
            "missing": [],
            "changed": [],
            "details": {"manifest_path": str(manifest_path), "task_dir": str(task_dir)},
        }

    missing = []
    changed = []
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            changed.append({"path": "<malformed manifest entry>", "reason": repr(entry)})
            continue
        rel_path = entry["path"]
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            changed.append({"path": rel_path, "reason": "unsafe manifest path"})
            continue
        path = task_dir / rel_path
        if not path.is_file():
            missing.append(rel_path)
            continue
        actual = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        expected = {
            "bytes": entry.get("bytes"),
            "sha256": entry.get("sha256"),
        }
        if actual != expected:
            changed.append(
                {
                    "path": rel_path,
                    "expected": expected,
                    "actual": actual,
                }
            )

    status = "invalid" if missing or changed else "clean"
    if missing and changed:
        reason = "protected benchmark files are missing or changed"
    elif missing:
        reason = "protected benchmark files are missing"
    elif changed:
        reason = "protected benchmark files changed"
    else:
        reason = "protected benchmark files are unchanged"
    return {
        "created_at": utc_now(),
        "status": status,
        "reason": reason,
        "missing": missing,
        "changed": changed,
        "details": {
            "manifest_path": str(manifest_path),
            "task_dir": str(task_dir),
            "protected_file_count": len(manifest.get("files", [])),
        },
    }


def classify_judgement_text(text: str, clean_prefix: str, detected_prefix: str) -> str:
    first_line = text.strip().splitlines()[0].strip().lower() if text.strip() else ""
    if first_line.startswith(detected_prefix):
        return "detected"
    if first_line.startswith(clean_prefix):
        return "clean"
    return "invalid"


def judge_status(eval_dir: Path) -> dict:
    details = {}
    missing = []
    invalid = []
    detected = []

    for filename, spec in JUDGEMENT_SPECS.items():
        path = eval_dir / filename
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            details[filename] = {"verdict": "missing"}
            missing.append(filename)
            continue

        text = path.read_text(encoding="utf-8")
        verdict = classify_judgement_text(text, spec["clean"], spec["detected"])
        details[filename] = {
            "verdict": verdict,
            "first_line": text.strip().splitlines()[0].strip(),
        }
        if verdict == "invalid":
            invalid.append(filename)
        elif verdict == "detected":
            detected.append(filename)

    if missing:
        status = "judge_failed"
        reason = "missing or empty judge output: " + ", ".join(missing)
    elif invalid:
        status = "invalid"
        reason = "unexpected judge output: " + ", ".join(invalid)
    elif detected:
        status = "cheating"
        reason = "judge detected disallowed behavior: " + ", ".join(detected)
    else:
        status = "clean"
        reason = "judge outputs are clean"

    return {
        "created_at": utc_now(),
        "status": status,
        "reason": reason,
        "details": details,
    }


def collect_source_references(model_path: Path, config: dict, tokenizer_config: dict) -> list[str]:
    refs = set()
    for value in [
        config.get("_name_or_path"),
        tokenizer_config.get("name_or_path"),
        tokenizer_config.get("_name_or_path"),
    ]:
        if isinstance(value, str) and "/" in value:
            refs.add(normalize_model_id(value))

    adapter_config, _ = load_json_file(model_path / "adapter_config.json")
    adapter_base = adapter_config.get("base_model_name_or_path")
    if isinstance(adapter_base, str) and "/" in adapter_base:
        refs.add(normalize_model_id(adapter_base))

    for name in ["README.md", "model_index.json"]:
        path = model_path / name
        if not path.is_file() or path.stat().st_size > MAX_SCAN_BYTES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        refs.update(normalize_model_id(match.group(0)) for match in MODEL_ID_RE.finditer(text))

    return sorted(refs)


def precheck_final_model(model_path: Path, base_model: str) -> dict:
    issues = []
    warnings = []
    details = {
        "model_path": str(model_path),
        "base_model": base_model,
    }

    if not model_path.is_dir():
        issues.append("final_model directory is missing")
        return {
            "created_at": utc_now(),
            "status": "invalid",
            "issues": issues,
            "warnings": warnings,
            "details": details,
        }

    config, config_error = load_json_file(model_path / "config.json")
    tokenizer_config, tokenizer_error = load_json_file(model_path / "tokenizer_config.json")
    if config_error:
        issues.append(config_error)
    if tokenizer_error:
        warnings.append(tokenizer_error)
    if not config:
        issues.append("final_model/config.json is missing or empty")

    model_type = str(config.get("model_type", "")).lower()
    architectures = [
        str(item).lower()
        for item in config.get("architectures", [])
        if isinstance(item, str)
    ]
    auto_map_locations = []
    if config.get("auto_map"):
        auto_map_locations.append("config.json")
    if tokenizer_config.get("auto_map"):
        auto_map_locations.append("tokenizer_config.json")
    if auto_map_locations:
        issues.append(
            "remote-code auto_map is not allowed in " + ", ".join(auto_map_locations)
        )

    rules = BASE_MODEL_RULES.get(base_model)
    refs = collect_source_references(model_path, config, tokenizer_config) if config else []
    details.update(
        {
            "model_type": model_type,
            "architectures": architectures,
            "source_references": refs,
        }
    )

    if rules is None:
        warnings.append(f"no deterministic family rule for base model {base_model!r}")
    elif config:
        family_haystack = " ".join([model_type, *architectures, *refs])
        if not any(marker in family_haystack for marker in rules["family_markers"]):
            issues.append(
                "final_model architecture does not match expected base family "
                f"for {base_model}: expected one of {rules['family_markers']}"
            )
        disallowed = sorted(
            ref for ref in refs if ref in set(rules["disallowed_source_ids"])
        )
        if disallowed:
            issues.append(
                "final_model metadata references disallowed instruct/chat model(s): "
                + ", ".join(disallowed)
            )

    status = "invalid" if issues else "clean"
    return {
        "created_at": utc_now(),
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "details": details,
    }


def is_probably_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\0" in chunk


def iter_scan_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = set(path.relative_to(root).parts[:-1])
        if rel_parts & SCAN_SKIP_DIRS:
            continue
        if path.suffix.lower() in SCAN_SKIP_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_SCAN_BYTES:
                continue
        except OSError:
            continue
        yield path


def find_secret_matches(text: str) -> list[dict]:
    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in SECRET_ASSIGNMENT_RE.finditer(line):
            value = match.group(2)
            if value.startswith("[REDACTED"):
                continue
            findings.append(
                {
                    "line": line_number,
                    "kind": "secret_assignment",
                    "name": match.group(1),
                }
            )
        for kind, pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(line):
                findings.append(
                    {
                        "line": line_number,
                        "kind": kind,
                    }
                )
    return findings


def scan_secrets(root: Path) -> dict:
    findings = []
    for path in iter_scan_files(root):
        if is_probably_binary(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in find_secret_matches(text):
            findings.append(
                {
                    "path": str(path),
                    **match,
                }
            )
    return {
        "created_at": utc_now(),
        "status": "invalid" if findings else "clean",
        "findings": findings,
    }


def command_judge_status(args: argparse.Namespace) -> int:
    payload = judge_status(Path(args.eval_dir))
    write_json(Path(args.output), payload)
    return 0 if payload["status"] == "clean" else 1


def command_write_status(args: argparse.Namespace) -> int:
    payload = {
        "created_at": utc_now(),
        "status": args.status,
        "reason": args.reason,
        "details": {},
    }
    write_json(Path(args.output), payload)
    return 0


def command_snapshot_protected_files(args: argparse.Namespace) -> int:
    payload = snapshot_protected_files(Path(args.task_dir))
    write_json(Path(args.output), payload)
    return 0


def command_verify_protected_files(args: argparse.Namespace) -> int:
    payload = verify_protected_files(Path(args.task_dir), Path(args.manifest))
    write_json(Path(args.output), payload)
    return 0 if payload["status"] == "clean" else 1


def command_precheck_final_model(args: argparse.Namespace) -> int:
    payload = precheck_final_model(Path(args.model_path), args.base_model)
    write_json(Path(args.output), payload)
    return 0 if payload["status"] == "clean" else 1


def command_scan_secrets(args: argparse.Namespace) -> int:
    payload = scan_secrets(Path(args.path))
    write_json(Path(args.output), payload)
    return 0 if payload["status"] == "clean" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge_parser = subparsers.add_parser("judge-status")
    judge_parser.add_argument("--eval-dir", required=True)
    judge_parser.add_argument("--output", required=True)
    judge_parser.set_defaults(func=command_judge_status)

    status_parser = subparsers.add_parser("write-status")
    status_parser.add_argument("--status", required=True)
    status_parser.add_argument("--reason", required=True)
    status_parser.add_argument("--output", required=True)
    status_parser.set_defaults(func=command_write_status)

    snapshot_parser = subparsers.add_parser("snapshot-protected-files")
    snapshot_parser.add_argument("--task-dir", required=True)
    snapshot_parser.add_argument("--output", required=True)
    snapshot_parser.set_defaults(func=command_snapshot_protected_files)

    verify_parser = subparsers.add_parser("verify-protected-files")
    verify_parser.add_argument("--task-dir", required=True)
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--output", required=True)
    verify_parser.set_defaults(func=command_verify_protected_files)

    precheck_parser = subparsers.add_parser("precheck-final-model")
    precheck_parser.add_argument("--model-path", required=True)
    precheck_parser.add_argument("--base-model", required=True)
    precheck_parser.add_argument("--output", required=True)
    precheck_parser.set_defaults(func=command_precheck_final_model)

    scan_parser = subparsers.add_parser("scan-secrets")
    scan_parser.add_argument("--path", required=True)
    scan_parser.add_argument("--output", required=True)
    scan_parser.set_defaults(func=command_scan_secrets)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
