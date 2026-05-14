import importlib.util
import json
from pathlib import Path


AGGREGATE_PATH = Path(__file__).parents[2] / "post_train_bench" / "aggregate_results.py"
spec = importlib.util.spec_from_file_location("aggregate_results", AGGREGATE_PATH)
assert spec is not None
aggregate_results = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(aggregate_results)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_task(
    run_root: Path, method: str, task_name: str, status: str, accuracy: float | None
):
    task_dir = run_root / "results" / method / task_name
    write_json(task_dir / "integrity_status.json", {"status": status})
    if accuracy is not None:
        write_json(task_dir / "metrics.json", {"accuracy": accuracy})


def test_aggregate_applies_reference_weights_and_excludes_nonclean_tasks(tmp_path):
    factors = {"gsm8k": 0.25, "humaneval": 0.75}
    run_root = tmp_path / "run1"
    write_json(run_root / "run_metadata.json", {"run_id": "run1"})
    make_task(run_root, "method", "gsm8k_Qwen_Qwen3-1.7B-Base_0", "clean", 0.8)
    make_task(run_root, "method", "humaneval_Qwen_Qwen3-1.7B-Base_0", "cheating", 1.0)

    [summary] = aggregate_results.summarize_run(run_root, factors, "accuracy")

    assert summary["weighted_score"] == 0.2
    assert summary["present_weight"] == 0.25
    assert summary["status_counts"] == {"clean": 1, "cheating": 1}
    assert summary["missing_benchmarks"] == ["humaneval"]


def test_aggregate_reports_multi_run_variance(tmp_path):
    summaries = [
        {"method": "method", "weighted_score": 0.2},
        {"method": "method", "weighted_score": 0.6},
    ]

    variance = aggregate_results.summarize_variance(summaries)

    assert variance["method"]["n"] == 2
    assert variance["method"]["mean"] == 0.4
    assert variance["method"]["stddev"] > 0
