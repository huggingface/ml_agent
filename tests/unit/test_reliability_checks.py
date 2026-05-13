"""Tests for the static pre-flight checks run at the hf_jobs approval prompt."""

import pytest

from agent.utils.reliability_checks import (
    Finding,
    check_training_script_save_pattern,
    format_finding,
    run_preflight_checks,
)


# ── Finding / format_finding ────────────────────────────────────────────


def test_finding_is_frozen():
    f = Finding("warn", "msg")
    with pytest.raises(Exception):
        f.message = "other"  # type: ignore[misc]


def test_format_finding_uses_red_for_warn():
    out = format_finding(Finding("warn", "boom"))
    assert "\033[91m" in out and "boom" in out and out.endswith("\033[0m")


def test_format_finding_uses_green_for_info():
    out = format_finding(Finding("info", "ok"))
    assert "\033[92m" in out and "ok" in out and out.endswith("\033[0m")


# ── save-pattern check (system_prompt_v3.yaml:39) ──────────────────────


def test_save_pattern_warns_when_from_pretrained_without_push():
    findings = run_preflight_checks({"script": "model = AutoModel.from_pretrained('x')"})
    assert any(f.severity == "warn" and "No model save" in f.message for f in findings)


def test_save_pattern_info_when_push_to_hub_present():
    script = "AutoModel.from_pretrained('x'); trainer.push_to_hub()"
    findings = run_preflight_checks({"script": script})
    assert any(f.severity == "info" and "pushed to hub" in f.message for f in findings)


def test_save_pattern_warns_on_local_save_without_push():
    script = "AutoModel.from_pretrained('x'); trainer.save_model('out')"
    findings = run_preflight_checks({"script": script})
    assert any(
        f.severity == "warn" and "ephemeral" in f.message for f in findings
    )


def test_save_pattern_silent_when_no_from_pretrained():
    findings = run_preflight_checks({"script": "print('hello')"})
    assert all("save" not in f.message.lower() for f in findings)


# ── timeout check (system_prompt_v3.yaml:37) ───────────────────────────


@pytest.mark.parametrize("trainer_pattern", [
    "Trainer(model=m)",
    "SFTTrainer(model=m)",
    "GRPOTrainer(model=m)",
    "DPOTrainer(model=m)",
    "trainer.train(",
])
def test_timeout_warns_on_default_with_training_call(trainer_pattern):
    findings = run_preflight_checks({"script": trainer_pattern, "timeout": "30m"})
    assert any(
        f.severity == "warn" and "30m timeout" in f.message for f in findings
    )


def test_timeout_warns_when_timeout_missing_entirely():
    # Missing timeout is treated as the default 30m.
    findings = run_preflight_checks({"script": "Trainer(model=m)"})
    assert any("30m timeout" in f.message for f in findings)


def test_timeout_silent_when_explicitly_set_long():
    findings = run_preflight_checks({"script": "Trainer(model=m)", "timeout": "6h"})
    assert all("30m timeout" not in f.message for f in findings)


def test_timeout_silent_when_no_training_call():
    findings = run_preflight_checks({"script": "model.generate(x)", "timeout": "30m"})
    assert all("30m timeout" not in f.message for f in findings)


def test_timeout_check_runs_for_docker_mode():
    findings = run_preflight_checks({
        "command": ["python", "-c", "from trl import SFTTrainer; SFTTrainer(...)"],
        "timeout": "30m",
    })
    assert any("30m timeout" in f.message for f in findings)


# ── hub_model_id check (system_prompt_v3.yaml:39) ──────────────────────


def test_hub_model_id_warns_when_pushing_without_id():
    script = (
        "AutoModel.from_pretrained('x')\n"
        "args = TrainingArguments(push_to_hub=True)"
    )
    findings = run_preflight_checks({"script": script})
    assert any(
        f.severity == "warn" and "hub_model_id" in f.message for f in findings
    )


def test_hub_model_id_silent_for_method_call_with_inline_repo():
    # ``trainer.push_to_hub("me/foo")`` carries the destination inline; the
    # check must not fire on this form.
    script = "AutoModel.from_pretrained('x'); trainer.push_to_hub('me/foo')"
    findings = run_preflight_checks({"script": script})
    assert all("hub_model_id" not in f.message for f in findings)


def test_hub_model_id_silent_when_id_present():
    script = (
        "AutoModel.from_pretrained('x')\n"
        "args = TrainingArguments(push_to_hub=True, hub_model_id='me/foo')"
    )
    findings = run_preflight_checks({"script": script})
    assert all("hub_model_id" not in f.message for f in findings)


def test_hub_model_id_silent_when_push_explicitly_disabled():
    script = "AutoModel.from_pretrained('x')\nargs = TrainingArguments(push_to_hub=False)"
    findings = run_preflight_checks({"script": script})
    assert all("hub_model_id" not in f.message for f in findings)


# ── flash-attn check (system_prompt_v3.yaml:45) ────────────────────────


def test_flash_attn_warns_on_legacy_literal_even_with_dep():
    # Per system_prompt_v3.yaml:45 the guidance is to avoid building
    # flash-attn from source entirely. The check fires on the legacy
    # ``attn_implementation="flash_attention_2"`` literal regardless of
    # whether flash-attn is in deps.
    script = 'AutoModel.from_pretrained("x", attn_implementation="flash_attention_2")'
    findings = run_preflight_checks({
        "script": script,
        "dependencies": ["transformers", "flash-attn"],
    })
    assert any(
        f.severity == "warn" and "kernels-community/flash-attn2" in f.message
        for f in findings
    )


def test_flash_attn_warns_when_dep_missing():
    script = 'model = AutoModel.from_pretrained("x", attn_implementation="flash_attention_2")'
    findings = run_preflight_checks({"script": script, "dependencies": ["transformers"]})
    assert any(
        f.severity == "warn" and "kernels-community/flash-attn2" in f.message
        for f in findings
    )


def test_flash_attn_silent_for_kernels_community_form():
    # The recommended form must not trip the warning. Note the dash in
    # "flash-attn2" vs the underscore in the legacy "flash_attention_2".
    script = (
        'AutoModel.from_pretrained("x", '
        'attn_implementation="kernels-community/flash-attn2")'
    )
    findings = run_preflight_checks({"script": script, "dependencies": []})
    assert all("flash_attention_2" not in f.message for f in findings)


def test_flash_attn_silent_when_not_used():
    findings = run_preflight_checks({
        "script": "AutoModel.from_pretrained('x')",
        "dependencies": [],
    })
    assert all("flash_attention_2" not in f.message for f in findings)


# ── trackio check (system_prompt_v3.yaml:65-70) ────────────────────────


def test_trackio_info_when_training_without_trackio():
    findings = run_preflight_checks({"script": "Trainer(model=m).train()", "timeout": "6h"})
    assert any(
        f.severity == "info" and "trackio" in f.message for f in findings
    )


def test_trackio_silent_when_trackio_configured():
    script = 'args = TrainingArguments(report_to="trackio")\nTrainer(model=m).train()'
    findings = run_preflight_checks({"script": script, "timeout": "6h"})
    assert all("trackio" not in f.message for f in findings)


def test_trackio_silent_for_inference_only():
    findings = run_preflight_checks({"script": "model.generate(x)", "timeout": "6h"})
    assert all("trackio" not in f.message for f in findings)


# ── Docker mode / overall integration ──────────────────────────────────


def test_docker_mode_skips_script_parsing_checks():
    # No `script` key. Only the timeout check applies; the others must self-skip.
    findings = run_preflight_checks({"command": ["python", "infer.py"], "timeout": "6h"})
    assert findings == []


def test_empty_arguments_returns_no_findings():
    assert run_preflight_checks({}) == []


def test_findings_are_emitted_in_documented_order():
    # When several checks fire on one script, the save-pattern finding comes
    # before the timeout finding. The CLI relies on this for stable output.
    script = "AutoModel.from_pretrained('x')\nTrainer(model=m)"
    findings = run_preflight_checks({"script": script, "timeout": "30m"})
    severities = [f.message for f in findings]
    save_idx = next(i for i, m in enumerate(severities) if "save" in m.lower())
    timeout_idx = next(i for i, m in enumerate(severities) if "30m timeout" in m)
    assert save_idx < timeout_idx


# ── Legacy wrapper (back-compat) ───────────────────────────────────────


def test_legacy_wrapper_returns_warning_string_when_no_save():
    out = check_training_script_save_pattern("AutoModel.from_pretrained('x')")
    assert out is not None
    assert "\033[91m" in out and "No model save" in out


def test_legacy_wrapper_returns_info_string_when_pushing():
    out = check_training_script_save_pattern(
        "AutoModel.from_pretrained('x'); push_to_hub()"
    )
    assert out is not None
    assert "\033[92m" in out and "pushed to hub" in out


def test_legacy_wrapper_returns_none_for_plain_script():
    assert check_training_script_save_pattern("print('hi')") is None
