"""Tests for the secret scrubber used before session upload."""

from agent.core.redact import scrub, scrub_string


def test_hf_token():
    s = "here is a token hf_" + "A" * 35 + " ok"
    out = scrub_string(s)
    assert "hf_" not in out
    assert "[REDACTED_HF_TOKEN]" in out


def test_anthropic_key():
    s = "key=sk-ant-api03_" + "a" * 40
    out = scrub_string(s)
    # The env-var name prefix matches too; just verify we don't leave the body.
    assert "sk-ant-api03_" not in out


def test_github_token():
    s = "ghp_" + "a" * 40
    out = scrub_string(s)
    assert out == "[REDACTED_GITHUB_TOKEN]"


def test_github_fine_grained_pat():
    # Fine-grained PATs: github_pat_<alphanumeric + underscore>, 36+ chars
    s = "github_pat_" + "A1B2_" * 10
    out = scrub_string(s)
    assert "github_pat_" not in out
    assert "[REDACTED_GITHUB_TOKEN]" in out


def test_aws_key_id():
    s = "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
    out = scrub_string(s)
    assert "AKIAABCDEFGHIJKLMNOP" not in out


def test_bearer_header():
    s = "Authorization: Bearer abcdef0123456789abcdef0123456789"
    out = scrub_string(s)
    assert "abcdef0123456789abcdef0123456789" not in out
    assert "Bearer [REDACTED]" in out


def test_env_var_style():
    s = "HF_TOKEN=hf_" + "x" * 40 + " run"
    out = scrub_string(s)
    # Either the value-scrubber or the HF-token regex should fire.
    assert "hf_xxxx" not in out


def test_scrub_nested_dict_and_list():
    payload = {
        "msg": "token hf_" + "Z" * 35,
        "tools": [
            {"args": {"secret": "ghp_" + "Q" * 40}},
            "no secrets here",
        ],
        "n": 42,
    }
    out = scrub(payload)
    # Original not mutated
    assert "hf_" in payload["msg"]
    # Redacted copy
    assert "[REDACTED_HF_TOKEN]" in out["msg"]
    assert out["tools"][0]["args"]["secret"] == "[REDACTED_GITHUB_TOKEN]"
    assert out["tools"][1] == "no secrets here"
    assert out["n"] == 42


def test_scrub_preserves_non_strings():
    assert scrub(None) is None
    assert scrub(123) == 123
    assert scrub(True) is True
