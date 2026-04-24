import json

import pytest

from agent.eval.artifacts import (
    append_leaderboard_row,
    build_leaderboard_row,
    build_run_record,
    write_run_artifact,
)
from agent.eval.compare import ModelResult, compare_results


def test_write_run_artifact_creates_json_file(tmp_path):
    record = {
        "run_id": "run-123",
        "task": "glue_sst2",
        "primary_metric": "accuracy",
        "primary_delta": 0.05,
    }

    path = write_run_artifact(tmp_path, record)

    assert path.name == "run-123.json"
    assert json.loads(path.read_text()) == record


def test_write_run_artifact_rejects_duplicate_run_id(tmp_path):
    record = {
        "run_id": "run-123",
        "task": "glue_sst2",
        "primary_metric": "accuracy",
        "primary_delta": 0.05,
    }

    write_run_artifact(tmp_path, record)

    with pytest.raises(FileExistsError, match="Run artifact already exists:"):
        write_run_artifact(tmp_path, record)


def test_append_leaderboard_row_appends_jsonl_line(tmp_path):
    first_row = {
        "run_id": "run-123",
        "task": "glue_sst2",
        "delta": 0.05,
    }
    second_row = {
        "run_id": "run-456",
        "task": "glue_sst2",
        "delta": -0.01,
    }

    path = append_leaderboard_row(tmp_path, first_row)
    append_leaderboard_row(tmp_path, second_row)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == first_row
    assert json.loads(lines[1]) == second_row


def test_build_record_and_leaderboard_row_include_cost_metadata():
    comparison = compare_results(
        task_id="glue_sst2",
        primary_metric="accuracy",
        baseline=ModelResult("baseline", {"accuracy": 0.84}),
        candidate=ModelResult("candidate", {"accuracy": 0.89}),
    )

    record = build_run_record(
        run_id="run-123",
        comparison=comparison,
        dataset="glue/sst2",
        split="validation",
        parameters={"limit": 100},
        training_cost=12.5,
        eval_cost=0.25,
        notes="first pass",
    )
    row = build_leaderboard_row(record)

    assert record["run_id"] == "run-123"
    assert record["training_cost"] == 12.5
    assert record["eval_cost"] == 0.25
    assert record["parameters"] == {"limit": 100}
    assert row["baseline_score"] == 0.84
    assert row["candidate_score"] == 0.89
    assert row["delta"] == pytest.approx(0.05)
    assert row["training_cost"] == 12.5
    assert row["eval_cost"] == 0.25


def test_build_run_record_snapshots_optional_metadata_and_metrics():
    baseline_metrics = {"accuracy": 0.5}
    candidate_metrics = {"accuracy": 0.6}
    parameters = {"limit": 10}
    comparison = compare_results(
        task_id="glue_sst2",
        primary_metric="accuracy",
        baseline=ModelResult("baseline", baseline_metrics),
        candidate=ModelResult("candidate", candidate_metrics),
    )

    record = build_run_record(
        run_id="run-789",
        comparison=comparison,
        dataset="glue/sst2",
        split="validation",
        parameters=parameters,
        training_cost=None,
        eval_cost=None,
        notes=None,
    )

    baseline_metrics["accuracy"] = 0.0
    candidate_metrics["accuracy"] = 0.0
    parameters["limit"] = 1

    assert record["baseline_metrics"] == {"accuracy": 0.5}
    assert record["candidate_metrics"] == {"accuracy": 0.6}
    assert record["parameters"] == {"limit": 10}
    assert record["training_cost"] is None
    assert record["eval_cost"] is None
    assert record["notes"] is None
