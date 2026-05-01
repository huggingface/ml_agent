from agent.core.session_uploader import _upload_dataset_card, dataset_card_readme


def test_dataset_card_readme_has_metadata_and_public_warning():
    readme = dataset_card_readme("lewtun/ml-intern-sessions")

    assert readme.startswith("---\n")
    assert 'pretty_name: "ML Intern Session Traces"' in readme
    assert "task_categories:\n- text-generation" in readme
    assert "- agent-traces" in readme
    assert "- coding-agent" in readme
    assert "- ml-intern" in readme
    assert 'path: "sessions/**/*.jsonl"' in readme
    assert "**WARNING: no redaction or human review has been performed for this dataset.**" in readme
    assert "Do not make this dataset public" in readme


def test_upload_dataset_card_only_for_claude_code_format():
    class FakeApi:
        def __init__(self):
            self.calls = []

        def upload_file(self, **kwargs):
            self.calls.append(kwargs)

    api = FakeApi()

    _upload_dataset_card(api, "lewtun/ml-intern-sessions", "hf_token", "row")
    assert api.calls == []

    _upload_dataset_card(api, "lewtun/ml-intern-sessions", "hf_token", "claude_code")
    assert len(api.calls) == 1
    assert api.calls[0]["path_in_repo"] == "README.md"
    assert api.calls[0]["repo_id"] == "lewtun/ml-intern-sessions"
    assert api.calls[0]["repo_type"] == "dataset"
    assert api.calls[0]["token"] == "hf_token"
    assert b"no redaction or human review" in api.calls[0]["path_or_fileobj"]
