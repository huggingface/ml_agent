#!/usr/bin/env python3
"""Aggregate PostTrainBench per-task metrics into weighted run reports."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASELINE_AGENT_KEY = "base-model"
DEFAULT_BASELINE_CSV = "scratch/PostTrainBench/results/aggregated_baseline_zeroshot.csv"
MODEL_NAME_ALIASES = {
    "Qwen/Qwen3-1.7B-Base": "Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base": "Qwen3-4B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base": "SmolLM3-3B-Base",
    "google/gemma-3-4b-pt": "gemma-3-4b-pt",
}


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def metric_value(metrics: dict, preferred_key: str) -> float | None:
    value = metrics.get(preferred_key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    for key, value in sorted(metrics.items()):
        if key == "stderr":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def safe_model_name(model_name: str) -> str:
    safe = model_name
    for char in "/:[]":
        safe = safe.replace(char, "_")
    return safe


def official_model_name(model_name: str) -> str:
    if model_name in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[model_name]
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    for prefix in ("Qwen_", "HuggingFaceTB_", "google_"):
        if model_name.startswith(prefix):
            return model_name[len(prefix) :]
    return model_name


def normalize_score_table(
    scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    values = [
        value
        for benchmark_scores in scores.values()
        for value in benchmark_scores.values()
    ]
    if values and max(values) > 1.0:
        return {
            model: {
                benchmark: value / 100.0
                for benchmark, value in benchmark_scores.items()
            }
            for model, benchmark_scores in scores.items()
        }
    return scores


def load_baseline_csv(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return {}
        benchmarks = header[1:]
        scores = {}
        for row in reader:
            if not row:
                continue
            model = official_model_name(row[0])
            scores[model] = {}
            for index, benchmark in enumerate(benchmarks, start=1):
                if index >= len(row) or not row[index]:
                    continue
                scores[model][benchmark] = float(row[index])
    return normalize_score_table(scores)


def load_baseline_scores_json(path: Path) -> dict[str, dict[str, float]]:
    data = load_json(path)
    model_data = data.get("modelBenchmarkData", {}).get(BASELINE_AGENT_KEY, {})
    scores = {}
    for model, benchmark_entries in model_data.items():
        official_model = official_model_name(model)
        scores[official_model] = {}
        for benchmark, entry in benchmark_entries.items():
            if isinstance(entry, dict):
                value = entry.get("value")
            else:
                value = entry
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[official_model][benchmark] = float(value)
    return normalize_score_table(scores)


def merge_score_tables(
    primary: dict[str, dict[str, float]],
    secondary: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged = {
        model: dict(benchmark_scores) for model, benchmark_scores in primary.items()
    }
    for model, benchmark_scores in secondary.items():
        merged.setdefault(model, {}).update(benchmark_scores)
    return merged


def parse_task_name(name: str, benchmarks: set[str]) -> str | None:
    matches = [
        benchmark for benchmark in benchmarks if name.startswith(f"{benchmark}_")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def load_expected_cells(
    run_root: Path, benchmarks: set[str]
) -> tuple[set[tuple[str, str]], dict[str, str]]:
    matrix_path = run_root / "matrix.jsonl"
    expected = set()
    model_by_safe_name = {}
    if not matrix_path.exists():
        return expected, model_by_safe_name

    with matrix_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            benchmark = row.get("benchmark")
            model_to_train = row.get("model_to_train")
            if benchmark not in benchmarks or not isinstance(model_to_train, str):
                continue
            model = official_model_name(model_to_train)
            expected.add((benchmark, model))
            model_by_safe_name[safe_model_name(model_to_train)] = model
    return expected, model_by_safe_name


def parse_task_dir(
    name: str,
    benchmarks: set[str],
    model_by_safe_name: dict[str, str],
) -> tuple[str, str] | None:
    benchmark = parse_task_name(name, benchmarks)
    if benchmark is None:
        return None

    remainder = name[len(benchmark) + 1 :]
    for safe_name, model in sorted(
        model_by_safe_name.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if remainder == safe_name or remainder.startswith(f"{safe_name}_"):
            return benchmark, model

    parts = remainder.rsplit("_", 2)
    model_part = parts[0] if len(parts) == 3 else remainder
    return benchmark, official_model_name(model_part)


def benchmark_average(cell_scores: dict[tuple[str, str], float]) -> dict[str, float]:
    by_benchmark = defaultdict(list)
    for (benchmark, _model), value in cell_scores.items():
        by_benchmark[benchmark].append(value)
    return {
        benchmark: statistics.fmean(values)
        for benchmark, values in sorted(by_benchmark.items())
        if values
    }


def baseline_value(
    baseline_scores: dict[str, dict[str, float]],
    model: str,
    benchmark: str,
) -> float:
    try:
        return baseline_scores[model][benchmark]
    except KeyError as exc:
        raise ValueError(
            f"Missing baseline fallback for {model} x {benchmark}"
        ) from exc


def summarize_run(
    run_root: Path,
    factors: dict[str, float],
    metric_key: str,
    baseline_scores: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    results_dir = run_root / "results"
    cells_by_method = defaultdict(dict)
    status_counts = defaultdict(Counter)
    task_counts = defaultdict(int)
    benchmark_names = set(factors)
    expected_cells, model_by_safe_name = load_expected_cells(run_root, benchmark_names)

    for task_dir in sorted(results_dir.glob("*/*")):
        if not task_dir.is_dir():
            continue
        method = task_dir.parent.name
        parsed = parse_task_dir(task_dir.name, benchmark_names, model_by_safe_name)
        if parsed is None:
            continue
        benchmark, model = parsed

        task_counts[method] += 1
        status = load_json(task_dir / "integrity_status.json").get("status", "missing")
        status_counts[method][status] += 1
        value = None
        fallback_reason = None

        if status == "clean":
            value = metric_value(load_json(task_dir / "metrics.json"), metric_key)
            if value is None:
                fallback_reason = "missing_metric"
        else:
            fallback_reason = f"status:{status}"

        cells_by_method[method][(benchmark, model)] = {
            "task_dir": str(task_dir),
            "value": value,
            "fallback_reason": fallback_reason,
            "status": status,
        }

    summaries = []
    metadata = load_json(run_root / "run_metadata.json")
    for method in sorted(set(cells_by_method) | set(status_counts) | set(task_counts)):
        method_expected_cells = expected_cells or set(cells_by_method[method])
        cell_scores = {}
        fallback_cells = []
        for benchmark, model in sorted(method_expected_cells):
            cell = cells_by_method[method].get((benchmark, model))
            value = cell.get("value") if cell else None
            if value is None:
                reason = cell.get("fallback_reason") if cell else "missing_run"
                if baseline_scores is None:
                    raise ValueError(
                        "Baseline scores are required for PTB-compatible "
                        f"fallback on {model} x {benchmark} ({reason})"
                    )
                value = baseline_value(baseline_scores, model, benchmark)
                fallback_cells.append(
                    {
                        "benchmark": benchmark,
                        "model": model,
                        "reason": reason,
                        "baseline_value": value,
                        "task_dir": cell.get("task_dir") if cell else None,
                    }
                )
            cell_scores[(benchmark, model)] = float(value)

        benchmark_scores = benchmark_average(cell_scores)
        weighted_score = sum(
            factors[benchmark] * benchmark_scores[benchmark]
            for benchmark in benchmark_scores
        )
        present_weight = sum(factors[benchmark] for benchmark in benchmark_scores)
        missing_benchmarks = sorted(set(factors) - set(benchmark_scores))
        summaries.append(
            {
                "run_root": str(run_root),
                "run_id": metadata.get("run_id", run_root.name),
                "method": method,
                "weighted_score": weighted_score,
                "present_weight": present_weight,
                "coverage": present_weight / sum(factors.values()),
                "benchmark_scores": benchmark_scores,
                "missing_benchmarks": missing_benchmarks,
                "status_counts": dict(status_counts[method]),
                "fallback_count": len(fallback_cells),
                "fallback_cells": fallback_cells,
                "expected_cell_count": len(method_expected_cells),
                "scored_cell_count": len(cell_scores),
                "cell_scores": {
                    f"{benchmark}/{model}": value
                    for (benchmark, model), value in sorted(cell_scores.items())
                },
                "task_count": task_counts[method],
                "image_provenance": metadata.get("image_provenance", {}),
            }
        )
    return summaries


def summarize_variance(run_summaries: list[dict]) -> dict:
    grouped = defaultdict(list)
    for summary in run_summaries:
        grouped[summary["method"]].append(summary["weighted_score"])

    variance = {}
    for method, values in sorted(grouped.items()):
        variance[method] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "stderr": statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return variance


def write_csv(path: Path, run_summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "method",
                "weighted_score",
                "present_weight",
                "coverage",
                "task_count",
                "fallback_count",
                "expected_cell_count",
                "scored_cell_count",
                "status_counts",
                "missing_benchmarks",
            ],
        )
        writer.writeheader()
        for summary in run_summaries:
            writer.writerow(
                {
                    "run_id": summary["run_id"],
                    "method": summary["method"],
                    "weighted_score": summary["weighted_score"],
                    "present_weight": summary["present_weight"],
                    "coverage": summary["coverage"],
                    "task_count": summary["task_count"],
                    "fallback_count": summary["fallback_count"],
                    "expected_cell_count": summary["expected_cell_count"],
                    "scored_cell_count": summary["scored_cell_count"],
                    "status_counts": json.dumps(
                        summary["status_counts"], sort_keys=True
                    ),
                    "missing_benchmarks": ",".join(summary["missing_benchmarks"]),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_roots", nargs="+", help="One or more post_train_bench/runs/... run roots"
    )
    parser.add_argument(
        "--factors",
        default="scratch/PostTrainBench/scripts/factors.json",
        help="PostTrainBench benchmark weighting JSON",
    )
    parser.add_argument("--metric-key", default="accuracy")
    parser.add_argument(
        "--baseline-csv",
        default=DEFAULT_BASELINE_CSV,
        help=(
            "PTB zero-shot baseline CSV used for failed-cell fallback. "
            "Defaults to the upstream results path if available."
        ),
    )
    parser.add_argument(
        "--baseline-scores-json",
        help=(
            "Official posttrainbench.com scores.json. If supplied, the "
            "base-model table is used as the fallback source."
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    factors = {
        key: float(value) for key, value in load_json(Path(args.factors)).items()
    }
    if not factors:
        raise SystemExit(f"No benchmark factors found in {args.factors}")

    baseline_scores = {}
    baseline_sources = []
    baseline_csv = Path(args.baseline_csv) if args.baseline_csv else None
    if baseline_csv and baseline_csv.exists():
        baseline_scores = merge_score_tables(
            baseline_scores, load_baseline_csv(baseline_csv)
        )
        baseline_sources.append(str(baseline_csv))
    if args.baseline_scores_json:
        baseline_json = Path(args.baseline_scores_json)
        baseline_scores = merge_score_tables(
            baseline_scores, load_baseline_scores_json(baseline_json)
        )
        baseline_sources.append(str(baseline_json))
    if not baseline_scores:
        raise SystemExit(
            "No PTB baseline fallback scores loaded. Provide "
            "--baseline-csv path/to/aggregated_baseline_zeroshot.csv or "
            "--baseline-scores-json path/to/posttrainbench_scores.json."
        )

    run_summaries = []
    for run_root in args.run_roots:
        run_summaries.extend(
            summarize_run(Path(run_root), factors, args.metric_key, baseline_scores)
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "factors_path": args.factors,
        "metric_key": args.metric_key,
        "baseline_sources": baseline_sources,
        "run_summaries": run_summaries,
        "multi_run_variance": summarize_variance(run_summaries),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_csv:
        write_csv(Path(args.output_csv), run_summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
