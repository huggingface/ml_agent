import importlib.util
import json
from pathlib import Path


INTEGRITY_PATH = Path(__file__).parents[2] / "post_train_bench" / "integrity.py"
spec = importlib.util.spec_from_file_location(
    "post_train_bench_integrity", INTEGRITY_PATH
)
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


def test_protected_files_snapshot_ignores_python_bytecode_cache(tmp_path):
    task_dir = tmp_path / "task"
    cache_dir = task_dir / "evaluation_code" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (task_dir / "evaluate.py").write_text("print('eval')\n", encoding="utf-8")
    (task_dir / "evaluation_code" / "helper.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (cache_dir / "helper.cpython-311.pyc").write_bytes(b"old bytecode")
    (task_dir / "evaluation_code" / "legacy.pyo").write_bytes(b"old optimized bytecode")
    manifest = integrity.snapshot_protected_files(task_dir)
    manifest_paths = {entry["path"] for entry in manifest["files"]}

    assert "evaluate.py" in manifest_paths
    assert "evaluation_code/helper.py" in manifest_paths
    assert "evaluation_code/__pycache__/helper.cpython-311.pyc" not in manifest_paths
    assert "evaluation_code/legacy.pyo" not in manifest_paths

    manifest_path = tmp_path / "manifest.json"
    integrity.write_json(manifest_path, manifest)
    (cache_dir / "helper.cpython-311.pyc").write_bytes(b"new bytecode")
    (task_dir / "evaluation_code" / "legacy.pyo").write_bytes(b"new optimized bytecode")

    payload = integrity.verify_protected_files(task_dir, manifest_path)

    assert payload["status"] == "clean"
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


def test_snapshot_evidence_splits_task_snapshot_and_final_model(tmp_path):
    task_dir = tmp_path / "job" / "task"
    final_model = task_dir / "final_model"
    final_model.mkdir(parents=True)
    (task_dir / "solve_out.txt").write_text("log\n", encoding="utf-8")
    (final_model / "config.json").write_text("{}", encoding="utf-8")
    eval_dir = tmp_path / "result"

    payload = integrity.snapshot_evidence(task_dir, eval_dir)

    assert payload["status"] == "clean"
    assert (eval_dir / "task" / "solve_out.txt").is_file()
    assert not (eval_dir / "task" / "final_model").exists()
    assert (eval_dir / "final_model" / "config.json").is_file()


def test_runner_does_not_mount_result_into_solve_or_trust_remote_code():
    runner = (
        Path(__file__).parents[2] / "post_train_bench" / "run_task_docker.sh"
    ).read_text(encoding="utf-8")

    solve_mount_line = next(
        line
        for line in runner.splitlines()
        if line.startswith("SOLVE_CONTAINER_MOUNTS=")
    )
    assert "${EVAL_DIR}:/result" not in solve_mount_line
    assert "${JOB_REPO}:/ml-intern-src:ro" in solve_mount_line
    assert "trust_remote_code=True" not in runner
    assert "snapshot-protected-files" in runner
    assert "verify-protected-files" in runner
    assert "scan-secrets" not in runner
    assert "secret_scan" not in runner
    assert "TRUSTED_INTEGRITY" in runner
    assert (
        '"$JOB_REPO/post_train_bench/integrity.py" verify-protected-files' not in runner
    )
    assert "uv pip install --system -e ." not in runner
    assert "uv pip install --system ." in runner
    assert "create_baseline_final_model" in runner
    solve_env_line = next(
        line for line in runner.splitlines() if line.startswith("SOLVE_CONTAINER_ENV=")
    )
    assert "HF_TOKEN,HUGGING_FACE_HUB_TOKEN" not in solve_env_line
    assert "POST_TRAIN_BENCH_SOLVE_HF_TOKEN" in solve_env_line


def test_runner_labels_reprompt_method_variant():
    runner = (
        Path(__file__).parents[2] / "post_train_bench" / "run_task_docker.sh"
    ).read_text(encoding="utf-8")

    assert 'METHOD_SUFFIX="_reprompt"' in runner
    assert (
        'METHOD_DIR="ml_intern_${AGENT_SAFE}_${NUM_HOURS}h${METHOD_SUFFIX}"' in runner
    )
    assert 'echo "reprompt=$REPROMPT"' in runner
    solve_env_line = next(
        line for line in runner.splitlines() if line.startswith("SOLVE_CONTAINER_ENV=")
    )
    assert "POST_TRAIN_BENCH_REPROMPT" in solve_env_line
    assert "POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES" in solve_env_line


def test_agent_config_disables_hub_write_tools():
    config = json.loads(
        (
            Path(__file__).parents[2] / "post_train_bench" / "ml_intern_config.json"
        ).read_text(encoding="utf-8")
    )

    assert {"hf_repo_files", "hf_repo_git"} <= set(config["disabled_tools"])


def test_submit_full_mode_requires_clean_provenance():
    submit = (
        Path(__file__).parents[2] / "post_train_bench" / "submit_eval_set.sh"
    ).read_text(encoding="utf-8")

    assert "--allow-dirty" in submit
    assert "--allow-mutable-images" in submit
    assert "Refusing full mode from a tracked-dirty worktree" in submit
    assert "Refusing full mode with mutable solve image" in submit
    assert "image_provenance" in submit
    assert "sha256_file" in submit
    assert "POST_TRAIN_BENCH_BASELINE_FINAL_MODEL" in submit


def test_submit_supports_validation_and_reprompt_metadata():
    submit = (
        Path(__file__).parents[2] / "post_train_bench" / "submit_eval_set.sh"
    ).read_text(encoding="utf-8")

    assert "model-validation)" in submit
    assert "validation)" in submit
    assert '"benchmark": "humaneval"' in submit
    assert '"benchmark": "bfcl"' in submit
    assert '"model_to_train": "google/gemma-3-4b-pt"' in submit
    assert '"Qwen/Qwen3-4B-Base"' in submit
    assert '"HuggingFaceTB/SmolLM3-3B-Base"' in submit
    assert "POST_TRAIN_BENCH_REPROMPT" in submit
    assert "POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES" in submit
    assert '"reprompt_enabled"' in submit
    assert '"method_variant"' in submit
    assert '"method_suffix"' in submit
    assert "sha256_skipped" in submit


def test_headless_reprompt_is_explicit_opt_in():
    main_py = (Path(__file__).parents[2] / "agent" / "main.py").read_text(
        encoding="utf-8"
    )

    assert 'POST_TRAIN_BENCH_REPROMPT", False' in main_py
    assert "POST_TRAIN_BENCH_REPROMPT_MIN_MINUTES" in main_py
    assert "process_headless_turn" in main_py
    assert "_post_train_bench_reprompt_text" in main_py


def test_bash_guidance_does_not_default_to_nohup():
    local_tools = (
        Path(__file__).parents[2] / "agent" / "tools" / "local_tools.py"
    ).read_text(encoding="utf-8")
    sandbox_client = (
        Path(__file__).parents[2] / "agent" / "tools" / "sandbox_client.py"
    ).read_text(encoding="utf-8")

    assert "nohup <command>" not in local_tools
    assert "nohup <command>" not in sandbox_client
    assert "wait <PID>; echo $?" in local_tools
    assert "wait <PID>; echo $?" in sandbox_client
