"""Static pre-flight checks for hf_jobs submissions.

Each check is pure substring inspection on the arguments dict the agent is
about to send — no network calls, no imports of training libraries. Findings
are advisory: the CLI prints them at the approval prompt; nothing is blocked.

The five failure modes covered are documented in
``agent/prompts/system_prompt_v3.yaml`` (see lines 29-47, 65-70).
"""

from dataclasses import dataclass
from typing import Literal

_RED = "\033[91m"
_GREEN = "\033[92m"
_RESET = "\033[0m"

# Substrings that strongly indicate a training entry point. Conservative on
# purpose: a script that only does ``model.generate(...)`` should not trip
# the timeout or trackio checks.
_TRAINER_PATTERNS = (
    "Trainer(",
    "SFTTrainer(",
    "GRPOTrainer(",
    "DPOTrainer(",
    "trainer.train(",
)


@dataclass(frozen=True)
class Finding:
    severity: Literal["warn", "info"]
    message: str


def format_finding(finding: Finding) -> str:
    """Render a finding as a single colored line for terminal output."""
    color = _RED if finding.severity == "warn" else _GREEN
    return f"\n{color}{finding.message}{_RESET}"


def run_preflight_checks(arguments: dict) -> list[Finding]:
    """Run every static check against an hf_jobs ``arguments`` dict.

    ``arguments`` is the same dict already in scope at the CLI approval
    prompt: keys include ``script``, ``command``, ``dependencies``,
    ``hardware_flavor``, ``timeout``, ``env``, ``schedule``. Script-parsing
    checks self-skip when the job is in Docker mode (no ``script``).
    """
    findings: list[Finding] = []
    script = arguments.get("script") or ""

    if script:
        if (f := _check_save_pattern(script)) is not None:
            findings.append(f)
        if (f := _check_hub_model_id(script)) is not None:
            findings.append(f)
        if (f := _check_flash_attn(arguments)) is not None:
            findings.append(f)
        if (f := _check_trackio(arguments)) is not None:
            findings.append(f)

    if (f := _check_timeout(arguments)) is not None:
        findings.append(f)

    return findings


def _check_save_pattern(script: str) -> Finding | None:
    has_from_pretrained = "from_pretrained" in script
    has_push_to_hub = "push_to_hub" in script
    has_local_save = "trainer.save_model" in script or "save_pretrained" in script

    if not has_from_pretrained:
        return None
    if has_push_to_hub:
        return Finding("info", "Model will be pushed to hub after training.")
    if has_local_save:
        return Finding(
            "warn",
            "Model is saved locally but not pushed to hub. hf_jobs storage is "
            "ephemeral — add push_to_hub=True to keep the model.",
        )
    return Finding(
        "warn",
        "No model save detected in this script. Ensure this is intentional.",
    )


def _check_timeout(arguments: dict) -> Finding | None:
    # The hf_jobs default is 30m (see agent/main.py: arguments.get("timeout", "30m")).
    # Treat both an explicit "30m" and a missing timeout the same way.
    timeout = arguments.get("timeout") or "30m"
    if timeout != "30m":
        return None

    script = arguments.get("script", "") or ""
    command = arguments.get("command") or ""
    command_text = " ".join(command) if isinstance(command, list) else str(command)
    text = f"{script}\n{command_text}"
    if not any(pat in text for pat in _TRAINER_PATTERNS):
        return None

    return Finding(
        "warn",
        "Default 30m timeout with a training call — training takes hours and "
        "the job will be killed mid-run. Set timeout explicitly (e.g. '6h').",
    )


def _check_hub_model_id(script: str) -> Finding | None:
    # Only the TrainingArguments config form (push_to_hub=True) requires a
    # matching hub_model_id keyword. The method-call form
    # ``trainer.push_to_hub("me/foo")`` carries the destination inline and
    # must not trip this warning.
    if "push_to_hub=True" not in script.replace(" ", ""):
        return None
    if "hub_model_id" in script:
        return None
    return Finding(
        "warn",
        "push_to_hub=True is set without hub_model_id — the model will land "
        "at a default repo path. Set hub_model_id explicitly.",
    )


def _check_flash_attn(arguments: dict) -> Finding | None:
    # system_prompt_v3.yaml:45 now steers users away from compiling
    # flash-attn from source: "Do NOT pip install 'flash-attn'… Instead,
    # use the HF kernels library and attn_implementation=
    # 'kernels-community/flash-attn2'". Fire on the legacy literal
    # regardless of deps — building from source is slow and fragile.
    script = arguments.get("script", "") or ""
    if "flash_attention_2" not in script:
        return None
    return Finding(
        "warn",
        'Script uses attn_implementation="flash_attention_2" — building '
        "flash-attn from source is slow and often fails on the job's CUDA "
        "build. Prefer attn_implementation=\"kernels-community/flash-attn2\" "
        "which loads a prebuilt kernel from the Hub.",
    )


def _check_trackio(arguments: dict) -> Finding | None:
    script = arguments.get("script", "") or ""
    if not any(pat in script for pat in _TRAINER_PATTERNS):
        return None
    if "trackio" in script.lower():
        return None
    return Finding(
        "info",
        'Training script does not configure report_to="trackio" — '
        "you will have no live training metrics.",
    )


# ---------------------------------------------------------------------------
# Backward-compatible legacy entry point.
# ---------------------------------------------------------------------------

def check_training_script_save_pattern(script: str) -> str | None:
    """Legacy single-string API. Kept so older imports keep working.

    Prefer ``run_preflight_checks(arguments)`` in new code — it returns
    structured findings for every check, not just the save-pattern one.
    """
    f = _check_save_pattern(script)
    return format_finding(f) if f is not None else None
