from agent.core.llm_params import UnsupportedEffortError, _resolve_llm_params


def test_openai_xhigh_effort_is_forwarded():
    params = _resolve_llm_params(
        "openai/gpt-5.5",
        reasoning_effort="xhigh",
        strict=True,
    )

    assert params["model"] == "openai/gpt-5.5"
    assert params["reasoning_effort"] == "xhigh"


def test_openai_max_effort_is_still_rejected():
    try:
        _resolve_llm_params(
            "openai/gpt-5.4",
            reasoning_effort="max",
            strict=True,
        )
    except UnsupportedEffortError as exc:
        assert "OpenAI doesn't accept effort='max'" in str(exc)
    else:
        raise AssertionError("Expected UnsupportedEffortError for max effort")
