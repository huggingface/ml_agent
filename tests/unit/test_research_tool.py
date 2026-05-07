from agent.tools.research_tool import _get_research_model


def test_get_research_model_anthropic():
    assert (
        _get_research_model("anthropic/claude-3-opus-20240229")
        == "anthropic/claude-sonnet-4-6"
    )
    assert (
        _get_research_model("anthropic/claude-3-5-sonnet-20240620")
        == "anthropic/claude-sonnet-4-6"
    )


def test_get_research_model_bedrock_with_prefix():
    # US prefix
    assert (
        _get_research_model("bedrock/us.anthropic.claude-v3-opus:1")
        == "bedrock/us.anthropic.claude-sonnet-4-6"
    )
    # EU prefix
    assert (
        _get_research_model("bedrock/eu.anthropic.claude-v3-sonnet:1")
        == "bedrock/eu.anthropic.claude-sonnet-4-6"
    )
    # AP prefix
    assert (
        _get_research_model("bedrock/ap.anthropic.claude-v3-5-sonnet:1")
        == "bedrock/ap.anthropic.claude-sonnet-4-6"
    )


def test_get_research_model_bedrock_no_prefix():
    assert (
        _get_research_model("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
        == "bedrock/anthropic.claude-sonnet-4-6"
    )


def test_get_research_model_non_anthropic():
    # HF router models should remain unchanged
    assert (
        _get_research_model("meta-llama/Llama-3.1-8B-Instruct")
        == "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert (
        _get_research_model("huggingface/deepseek-ai/DeepSeek-V3")
        == "huggingface/deepseek-ai/DeepSeek-V3"
    )


def test_get_research_model_openai():
    # OpenAI models should remain unchanged
    assert _get_research_model("openai/gpt-4o") == "openai/gpt-4o"
