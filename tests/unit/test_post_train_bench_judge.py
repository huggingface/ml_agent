import importlib.util
import json
from pathlib import Path


RUN_JUDGE_PATH = Path(__file__).parents[2] / "post_train_bench" / "run_judge.py"
spec = importlib.util.spec_from_file_location("run_judge", RUN_JUDGE_PATH)
assert spec is not None
run_judge = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_judge)
ensure_codex_auth = run_judge.ensure_codex_auth


def test_ensure_codex_auth_writes_api_key_auth_file(tmp_path):
    env = {
        "CODEX_HOME": str(tmp_path / "codex"),
        "OPENAI_API_KEY": "test-key",
    }

    ensure_codex_auth(env)

    auth_file = tmp_path / "codex" / "auth.json"
    assert json.loads(auth_file.read_text(encoding="utf-8")) == {
        "OPENAI_API_KEY": "test-key",
        "auth_mode": "apikey",
    }
    assert auth_file.stat().st_mode & 0o777 == 0o600


def test_ensure_codex_auth_preserves_existing_auth_file(tmp_path):
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    auth_file = codex_home / "auth.json"
    auth_file.write_text(
        json.dumps({"OPENAI_API_KEY": "existing", "auth_mode": "apikey"}),
        encoding="utf-8",
    )

    ensure_codex_auth(
        {"CODEX_HOME": str(codex_home), "OPENAI_API_KEY": "replacement"}
    )

    assert json.loads(auth_file.read_text(encoding="utf-8"))["OPENAI_API_KEY"] == "existing"
