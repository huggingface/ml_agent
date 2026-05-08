"""Regression tests for interactive CLI rendering and research model routing."""

import sys
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

import agent.main as main_mod
from agent.tools.research_tool import _get_research_model
from agent.utils import terminal_display


def test_direct_anthropic_research_model_stays_off_bedrock():
    assert (
        _get_research_model("anthropic/claude-opus-4-6")
        == "anthropic/claude-sonnet-4-6"
    )


def test_bedrock_anthropic_research_model_stays_on_bedrock():
    assert (
        _get_research_model("bedrock/us.anthropic.claude-opus-4-6-v1")
        == "bedrock/us.anthropic.claude-sonnet-4-6"
    )


def test_non_anthropic_research_model_is_unchanged():
    assert _get_research_model("openai/gpt-5.4") == "openai/gpt-5.4"


def test_help_output_keeps_descriptions_aligned(monkeypatch):
    output = StringIO()
    console = Console(
        file=output,
        color_system=None,
        theme=terminal_display._THEME,
        width=120,
    )
    monkeypatch.setattr(terminal_display, "_console", console)

    terminal_display.print_help()

    lines = [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]
    description_columns = []
    for command, args, description in terminal_display.HELP_ROWS:
        line = next(line for line in lines if command in line)
        if args:
            assert args in line
        description_columns.append(line.index(description))

    assert len(set(description_columns)) == 1


def test_help_output_recomputes_widths_from_rows():
    rows = terminal_display.HELP_ROWS + (
        ("/longer-command", "[longer-args]", "Synthetic help row"),
    )
    output = StringIO()
    Console(
        file=output,
        color_system=None,
        theme=terminal_display._THEME,
        width=140,
    ).print(terminal_display.format_help_text(rows))

    lines = [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]
    description_columns = [
        next(line for line in lines if command in line).index(description)
        for command, _args, description in rows
    ]

    assert len(set(description_columns)) == 1


def test_subagent_display_does_not_spawn_background_redraw(monkeypatch):
    calls: list[object] = []

    def _unexpected_future(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("background redraw task should not be created")

    monkeypatch.setattr("asyncio.ensure_future", _unexpected_future)
    monkeypatch.setattr(
        terminal_display,
        "_console",
        SimpleNamespace(file=StringIO(), width=100),
    )

    mgr = terminal_display.SubAgentDisplayManager()
    mgr.start("agent-1", "research")
    mgr.add_call("agent-1", '▸ hf_papers  {"operation": "search"}')
    mgr.clear("agent-1")

    assert calls == []


def test_cli_forwards_model_flag_to_interactive_main(monkeypatch):
    seen: dict[str, str | None] = {}

    async def fake_main(*, model=None):
        seen["model"] = model

    monkeypatch.setattr(sys, "argv", ["ml-intern", "--model", "openai/gpt-5.5"])
    monkeypatch.setattr(main_mod, "main", fake_main)

    main_mod.cli()

    assert seen["model"] == "openai/gpt-5.5"


@pytest.mark.asyncio
async def test_interactive_main_applies_model_override_before_banner(monkeypatch):
    class StopAfterBanner(Exception):
        pass

    def fake_banner(*, model=None, hf_user=None):
        assert model == "openai/gpt-5.5"
        assert hf_user == "tester"
        raise StopAfterBanner

    monkeypatch.setattr(main_mod.os, "system", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(main_mod, "PromptSession", lambda: object())
    monkeypatch.setattr(main_mod, "resolve_hf_token", lambda: "hf-token")
    monkeypatch.setattr(main_mod, "_get_hf_user", lambda _token: "tester")
    monkeypatch.setattr(
        main_mod,
        "load_config",
        lambda _path, **_kwargs: SimpleNamespace(
            model_name="moonshotai/Kimi-K2.6",
            mcpServers={},
        ),
    )
    monkeypatch.setattr(main_mod, "print_banner", fake_banner)

    with pytest.raises(StopAfterBanner):
        await main_mod.main(model="openai/gpt-5.5")
