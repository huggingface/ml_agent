import importlib.util
import json
import pytest
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


def write_matrix(run_root: Path, rows: list[dict]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    (run_root / "matrix.jsonl").write_text(payload, encoding="utf-8")


def test_aggregate_uses_ptb_baseline_fallback_for_failed_cells(tmp_path):
    factors = {"gsm8k": 1.0}
    run_root = tmp_path / "run1"
    write_json(run_root / "run_metadata.json", {"run_id": "run1"})
    write_matrix(
        run_root,
        [
            {"benchmark": "gsm8k", "model_to_train": "Qwen/Qwen3-1.7B-Base"},
            {"benchmark": "gsm8k", "model_to_train": "Qwen/Qwen3-4B-Base"},
        ],
    )
    make_task(run_root, "method", "gsm8k_Qwen_Qwen3-1.7B-Base_0", "clean", 0.8)
    make_task(run_root, "method", "gsm8k_Qwen_Qwen3-4B-Base_0", "cheating", 1.0)
    baseline_scores = {
        "Qwen3-1.7B-Base": {"gsm8k": 0.1},
        "Qwen3-4B-Base": {"gsm8k": 0.2},
    }

    [summary] = aggregate_results.summarize_run(
        run_root, factors, "accuracy", baseline_scores
    )

    assert summary["weighted_score"] == 0.5
    assert summary["present_weight"] == 1.0
    assert summary["status_counts"] == {"clean": 1, "cheating": 1}
    assert summary["missing_benchmarks"] == []
    assert summary["fallback_count"] == 1
    assert summary["fallback_cells"] == [
        {
            "benchmark": "gsm8k",
            "model": "Qwen3-4B-Base",
            "reason": "status:cheating",
            "baseline_value": 0.2,
            "task_dir": str(
                run_root / "results" / "method" / "gsm8k_Qwen_Qwen3-4B-Base_0"
            ),
        }
    ]


def test_aggregate_fills_missing_expected_cells_from_baseline(tmp_path):
    factors = {"humaneval": 1.0}
    run_root = tmp_path / "run1"
    write_json(run_root / "run_metadata.json", {"run_id": "run1"})
    write_matrix(
        run_root,
        [
            {"benchmark": "humaneval", "model_to_train": "Qwen/Qwen3-1.7B-Base"},
            {"benchmark": "humaneval", "model_to_train": "Qwen/Qwen3-4B-Base"},
        ],
    )
    make_task(
        run_root,
        "method",
        "humaneval_Qwen_Qwen3-1.7B-Base_0",
        "clean",
        0.7,
    )

    [summary] = aggregate_results.summarize_run(
        run_root,
        factors,
        "accuracy",
        {
            "Qwen3-1.7B-Base": {"humaneval": 0.3},
            "Qwen3-4B-Base": {"humaneval": 0.1},
        },
    )

    assert summary["weighted_score"] == pytest.approx(0.4)
    assert summary["task_count"] == 1
    assert summary["expected_cell_count"] == 2
    assert summary["scored_cell_count"] == 2
    assert summary["fallback_cells"][0]["reason"] == "missing_run"


def test_aggregate_requires_matrix_jsonl(tmp_path):
    factors = {"gsm8k": 1.0}
    run_root = tmp_path / "run1"
    write_json(run_root / "run_metadata.json", {"run_id": "run1"})
    make_task(run_root, "method", "gsm8k_Qwen_Qwen3-1.7B-Base_0", "clean", 0.8)

    with pytest.raises(FileNotFoundError, match="matrix.jsonl"):
        aggregate_results.summarize_run(
            run_root,
            factors,
            "accuracy",
            {"Qwen3-1.7B-Base": {"gsm8k": 0.1}},
        )


def test_aggregate_reports_multi_run_variance(tmp_path):
    summaries = [
        {"method": "method", "weighted_score": 0.2},
        {"method": "method", "weighted_score": 0.6},
    ]

    variance = aggregate_results.summarize_variance(summaries)

    assert variance["method"]["n"] == 2
    assert variance["method"]["mean"] == 0.4
    assert variance["method"]["stddev"] > 0
