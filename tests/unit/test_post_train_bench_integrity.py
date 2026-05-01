import importlib.util
import json
from pathlib import Path


INTEGRITY_PATH = Path(__file__).parents[2] / "post_train_bench" / "integrity.py"
spec = importlib.util.spec_from_file_location("post_train_bench_integrity", INTEGRITY_PATH)
assert spec is not None
integrity = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(integrity)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_judge_status_requires_clean_prefixes(tmp_path):
    (tmp_path / "contamination_judgement.txt").write_text(
        "no contamination detected\ntraining used only public data\n",
        encoding="utf-8",
    )
    (tmp_path / "disallowed_model_judgement.txt").write_text(
        "only allowed use detected\nfinal_model derives from the base\n",
        encoding="utf-8",
    )

    payload = integrity.judge_status(tmp_path)

    assert payload["status"] == "clean"


def test_judge_status_marks_detected_verdict_as_cheating(tmp_path):
    (tmp_path / "contamination_judgement.txt").write_text(
        "contamination detected\nused benchmark test labels\n",
        encoding="utf-8",
    )
    (tmp_path / "disallowed_model_judgement.txt").write_text(
        "only allowed use detected\n",
        encoding="utf-8",
    )

    payload = integrity.judge_status(tmp_path)

    assert payload["status"] == "cheating"
    assert "contamination_judgement.txt" in payload["reason"]


def test_judge_status_marks_malformed_output_invalid(tmp_path):
    (tmp_path / "contamination_judgement.txt").write_text(
        "probably fine\n",
        encoding="utf-8",
    )
    (tmp_path / "disallowed_model_judgement.txt").write_text(
        "only allowed use detected\n",
        encoding="utf-8",
    )

    payload = integrity.judge_status(tmp_path)

    assert payload["status"] == "invalid"


def test_precheck_accepts_matching_base_family(tmp_path):
    model_path = tmp_path / "final_model"
    model_path.mkdir()
    write_json(
        model_path / "config.json",
        {
            "_name_or_path": "Qwen/Qwen3-1.7B-Base",
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        },
    )

    payload = integrity.precheck_final_model(model_path, "Qwen/Qwen3-1.7B-Base")

    assert payload["status"] == "clean"


def test_precheck_rejects_remote_code_auto_map(tmp_path):
    model_path = tmp_path / "final_model"
    model_path.mkdir()
    write_json(
        model_path / "config.json",
        {
            "_name_or_path": "Qwen/Qwen3-1.7B-Base",
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "auto_map": {"AutoModelForCausalLM": "modeling.CustomModel"},
        },
    )

    payload = integrity.precheck_final_model(model_path, "Qwen/Qwen3-1.7B-Base")

    assert payload["status"] == "invalid"
    assert any("auto_map" in issue for issue in payload["issues"])


def test_precheck_rejects_known_instruct_substitution(tmp_path):
    model_path = tmp_path / "final_model"
    model_path.mkdir()
    write_json(
        model_path / "config.json",
        {
            "_name_or_path": "Qwen/Qwen3-1.7B",
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        },
    )

    payload = integrity.precheck_final_model(model_path, "Qwen/Qwen3-1.7B-Base")

    assert payload["status"] == "invalid"
    assert any("disallowed" in issue for issue in payload["issues"])


def test_secret_scan_skips_final_model_and_flags_text_artifacts(tmp_path):
    (tmp_path / "solve_out.txt").write_text(
        "OPENAI_API_KEY=sk-" + "A" * 45 + "\n",
        encoding="utf-8",
    )
    final_model = tmp_path / "final_model"
    final_model.mkdir()
    (final_model / "config.json").write_text(
        "OPENAI_API_KEY=sk-" + "B" * 45 + "\n",
        encoding="utf-8",
    )

    payload = integrity.scan_secrets(tmp_path)

    assert payload["status"] == "invalid"
    assert len(payload["findings"]) == 2
    assert all("final_model" not in finding["path"] for finding in payload["findings"])


def test_secret_scan_ignores_lowercase_token_parameter(tmp_path):
    (tmp_path / "evaluate.py").write_text(
        "max_tokens=args.max_tokens\n",
        encoding="utf-8",
    )

    payload = integrity.scan_secrets(tmp_path)

    assert payload["status"] == "clean"


def test_protected_files_snapshot_and_verify_clean_with_extra_files(tmp_path):
    task_dir = tmp_path / "task"
    (task_dir / "templates").mkdir(parents=True)
    (task_dir / "evaluate.py").write_text("print('eval')\n", encoding="utf-8")
    (task_dir / "templates" / "qwen3.jinja").write_text("template\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    integrity.write_json(manifest_path, integrity.snapshot_protected_files(task_dir))
    (task_dir / "train.py").write_text("print('allowed new file')\n", encoding="utf-8")

    payload = integrity.verify_protected_files(task_dir, manifest_path)

    assert payload["status"] == "clean"
    assert payload["missing"] == []
    assert payload["changed"] == []


def test_protected_files_verify_rejects_changed_file(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    protected = task_dir / "evaluate.py"
    protected.write_text("original\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    integrity.write_json(manifest_path, integrity.snapshot_protected_files(task_dir))
    protected.write_text("tampered\n", encoding="utf-8")

    payload = integrity.verify_protected_files(task_dir, manifest_path)

    assert payload["status"] == "invalid"
    assert payload["changed"][0]["path"] == "evaluate.py"


def test_protected_files_verify_rejects_missing_file(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    protected = task_dir / "evaluate.py"
    protected.write_text("original\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    integrity.write_json(manifest_path, integrity.snapshot_protected_files(task_dir))
    protected.unlink()

    payload = integrity.verify_protected_files(task_dir, manifest_path)

    assert payload["status"] == "invalid"
    assert payload["missing"] == ["evaluate.py"]


def test_runner_does_not_mount_result_into_solve_or_trust_remote_code():
    runner = (Path(__file__).parents[2] / "post_train_bench" / "run_task_docker.sh").read_text(
        encoding="utf-8"
    )

    solve_mount_line = next(
        line for line in runner.splitlines() if line.startswith("SOLVE_CONTAINER_MOUNTS=")
    )
    assert "${EVAL_DIR}:/result" not in solve_mount_line
    assert "trust_remote_code=True" not in runner
    assert "snapshot-protected-files" in runner
    assert "verify-protected-files" in runner
