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


def parse_task_name(name: str, benchmarks: set[str]) -> str | None:
    matches = [
        benchmark for benchmark in benchmarks if name.startswith(f"{benchmark}_")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def summarize_run(
    run_root: Path, factors: dict[str, float], metric_key: str
) -> list[dict]:
    results_dir = run_root / "results"
    by_method = defaultdict(lambda: defaultdict(list))
    status_counts = defaultdict(Counter)
    task_counts = defaultdict(int)
    benchmark_names = set(factors)

    for task_dir in sorted(results_dir.glob("*/*")):
        if not task_dir.is_dir():
            continue
        method = task_dir.parent.name
        benchmark = parse_task_name(task_dir.name, benchmark_names)
        if benchmark is None:
            continue

        task_counts[method] += 1
        status = load_json(task_dir / "integrity_status.json").get("status", "missing")
        status_counts[method][status] += 1
        if status != "clean":
            continue

        value = metric_value(load_json(task_dir / "metrics.json"), metric_key)
        if value is not None:
            by_method[method][benchmark].append(value)

    summaries = []
    metadata = load_json(run_root / "run_metadata.json")
    for method in sorted(set(by_method) | set(status_counts) | set(task_counts)):
        benchmark_scores = {
            benchmark: statistics.fmean(values)
            for benchmark, values in sorted(by_method[method].items())
            if values
        }
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
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    factors = {
        key: float(value) for key, value in load_json(Path(args.factors)).items()
    }
    if not factors:
        raise SystemExit(f"No benchmark factors found in {args.factors}")

    run_summaries = []
    for run_root in args.run_roots:
        run_summaries.extend(summarize_run(Path(run_root), factors, args.metric_key))

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "factors_path": args.factors,
        "metric_key": args.metric_key,
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
