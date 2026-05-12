import json
from pathlib import Path

from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC
from agent.tools.sandbox_tool import SANDBOX_CREATE_TOOL_SPEC


def test_trackio_space_examples_use_hyphenated_ml_intern_prefix():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()
    tool_specs = json.dumps([HF_JOBS_TOOL_SPEC, SANDBOX_CREATE_TOOL_SPEC])
    legacy_prefix = "ml" + "intern"

    assert "<username>/ml-intern-<8-char-id>" in prompt
    assert "<username>/ml-intern-<8char>" in tool_specs
    assert legacy_prefix not in prompt
    assert legacy_prefix not in tool_specs
