import importlib.util
from pathlib import Path


COLLECT_PATH = Path(__file__).parents[2] / "post_train_bench" / "collect_artifacts.py"
spec = importlib.util.spec_from_file_location("collect_artifacts", COLLECT_PATH)
assert spec is not None
collect_artifacts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(collect_artifacts)


def test_record_final_model_tree_hashes_reproducibility_files(tmp_path):
    final_model = tmp_path / "final_model"
    final_model.mkdir()
    (final_model / "config.json").write_text("{}", encoding="utf-8")
    (final_model / "tokenizer.model").write_text("tok", encoding="utf-8")
    (final_model / "adapter_config.json").write_text("{}", encoding="utf-8")
    (final_model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (final_model / "training.log").write_text("not hashed", encoding="utf-8")
    manifest = {"referenced_files": [], "missing": []}

    collect_artifacts.record_optional_tree(final_model, manifest, "referenced_files")

    entries = {
        Path(entry["path"]).name: entry for entry in manifest["referenced_files"]
    }
    assert "sha256" in entries["config.json"]
    assert "sha256" in entries["tokenizer.model"]
    assert "sha256" in entries["adapter_config.json"]
    assert "sha256" in entries["model-00001-of-00001.safetensors"]
    assert "sha256" not in entries["training.log"]
