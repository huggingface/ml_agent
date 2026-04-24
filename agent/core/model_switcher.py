"""Model-switching logic for the interactive CLI's ``/model`` command.

Split out of ``agent.main`` so the REPL dispatcher stays focused on input
parsing. Exposes:

* Adapter-driven model catalog rendered by ``/model`` with no arg.
* ``is_valid_model_id`` — loose format check on user input.
* ``probe_and_switch_model`` — async: checks routing, fires a 1-token
  probe to resolve the effort cascade, then commits the switch (or
  rejects it on hard error).

The probe's cascade lives in ``agent.core.effort_probe``; this module
glues it to CLI output + session state.
"""

from __future__ import annotations

from agent.core.effort_probe import ProbeInconclusive, probe_effort
from agent.core.llm_errors import render_llm_error_message
from agent.core.provider_adapters import (
    ADAPTERS,
    find_model_option,
    get_available_models,
    is_valid_model_name,
    resolve_adapter,
)


_ROUTING_POLICIES = {"fastest", "cheapest", "preferred"}


def is_valid_model_id(model_id: str) -> bool:
    """Loose format check — lets users pick any model id.

    Accepts:
      • anthropic/<model>
      • openai/<model>
      • <org>/<model>[:<tag>]            (HF router; tag = provider or policy)
      • huggingface/<org>/<model>[:<tag>] (same, accepts legacy prefix)

    Actual availability is verified against the HF router catalog on
    switch, and by the provider on the probe's ping call.
    """
    return is_valid_model_name(model_id)


def _print_hf_routing_info(model_id: str, console) -> bool:
    """Show HF router catalog info (providers, price, context, tool support)
    for an HF-router model id. Returns ``True`` to signal the caller can
    proceed with the switch, ``False`` to indicate a hard problem the user
    should notice before we fire the effort probe.

    Anthropic / OpenAI ids return ``True`` without printing anything —
    the probe below covers "does this model exist".
    """
    adapter = resolve_adapter(model_id)
    if adapter and adapter.provider_id != "huggingface":
        return True

    from agent.core import hf_router_catalog as cat

    bare, _, tag = model_id.partition(":")
    info = cat.lookup(bare)
    if info is None:
        console.print(
            f"[bold red]Warning:[/bold red] '{bare}' isn't in the HF router "
            "catalog. Checking anyway — first call may fail."
        )
        suggestions = cat.fuzzy_suggest(bare)
        if suggestions:
            console.print(f"[dim]Did you mean: {', '.join(suggestions)}[/dim]")
        return True

    live = info.live_providers
    if not live:
        console.print(
            f"[bold red]Warning:[/bold red] '{bare}' has no live providers "
            "right now. First call will likely fail."
        )
        return True

    if tag and tag not in _ROUTING_POLICIES:
        matched = [p for p in live if p.provider == tag]
        if not matched:
            names = ", ".join(p.provider for p in live)
            console.print(
                f"[bold red]Warning:[/bold red] provider '{tag}' doesn't serve "
                f"'{bare}'. Live providers: {names}. Checking anyway."
            )

    if not info.any_supports_tools:
        console.print(
            f"[bold red]Warning:[/bold red] no provider for '{bare}' advertises "
            "tool-call support. This agent relies on tool calls — expect errors."
        )

    if tag in _ROUTING_POLICIES:
        policy = tag
    elif tag:
        policy = f"pinned to {tag}"
    else:
        policy = "auto (fastest)"
    console.print(f"  [dim]routing: {policy}[/dim]")
    for p in live:
        price = (
            f"${p.input_price:g}/${p.output_price:g} per M tok"
            if p.input_price is not None and p.output_price is not None
            else "price n/a"
        )
        ctx = f"{p.context_length:,} ctx" if p.context_length else "ctx n/a"
        tools = "tools" if p.supports_tools else "no tools"
        console.print(
            f"  [dim]{p.provider}: {price}, {ctx}, {tools}[/dim]"
        )
    return True


def print_model_listing(config, console) -> None:
    """Render the default ``/model`` (no-arg) view: current + available."""
    current = config.model_name if config else ""
    current_info = find_model_option(current)
    available = get_available_models()

    console.print("[bold]Current model:[/bold]")
    if current_info:
        console.print(f"  {current}  [dim]({current_info['label']})[/dim]")
    else:
        console.print(f"  {current}")

    console.print("\n[bold]Available:[/bold]")
    for adapter in ADAPTERS:
        section = [m for m in available if m.get("provider") == adapter.provider_id]
        if not section:
            if adapter.provider_id == "openai_compat" and adapter.should_show():
                console.print(f"\n[bold]{adapter.provider_label}[/bold]")
                console.print("  [dim]Use openai-compat/<model-id>[/dim]")
            continue

        console.print(f"\n[bold]{adapter.provider_label}[/bold]")
        section_sorted = sorted(
            section,
            key=lambda m: (
                not bool(m.get("recommended")),
                str(m.get("label", "")).lower(),
            ),
        )
        for m in section_sorted:
            marker = " [dim]<-- current[/dim]" if m["id"] == current else ""
            source = m.get("source")
            source_tag = " [dim](dynamic)[/dim]" if source == "dynamic" else ""
            console.print(f"  {m['id']}  [dim]({m['label']})[/dim]{source_tag}{marker}")

    console.print(
        "\n[dim]Paste any HF model id (e.g. 'MiniMaxAI/MiniMax-M2.7').\n"
        "Add ':fastest', ':cheapest', ':preferred', or ':<provider>' to override routing.\n"
        "Direct prefixes: 'anthropic/', 'openai/', 'openrouter/', 'opencode/',\n"
        "'opencode-go/', 'ollama/', 'lm_studio/', 'vllm/', 'openai-compat/'.[/dim]"
    )


def print_invalid_id(arg: str, console) -> None:
    console.print(f"[bold red]Invalid model id format:[/bold red] {arg}")
    console.print(
        "[dim]Expected:\n"
        "  • <org>/<model>[:tag]    (HF router — paste from huggingface.co)\n"
        "  • anthropic/<model>\n"
        "  • openai/<model>\n"
        "  • bedrock/<model>\n"
        "  • gemini/<model>\n"
        "  • openrouter/<model>\n"
        "  • opencode/<model>\n"
        "  • opencode-go/<model>\n"
        "  • ollama/<model>\n"
        "  • lm_studio/<model>\n"
        "  • vllm/<model>\n"
        "  • openai-compat/<model>[/dim]"
    )


async def probe_and_switch_model(
    model_id: str,
    config,
    session,
    console,
    hf_token: str | None,
) -> None:
    """Validate model+effort with a 1-token ping, cache the effective effort,
    then commit the switch.

    Three visible outcomes:

    * ✓ ``effort: <level>`` — model accepted the preferred effort (or a
      fallback from the cascade; the note explains if so)
    * ✓ ``effort: off`` — model doesn't support thinking; we'll strip it
    * ✗ hard error (auth, model-not-found, quota) — we reject the switch
      and keep the current model so the user isn't stranded

    Transient errors (5xx, timeout) complete the switch with a yellow
    warning; the next real call re-surfaces the error if it's persistent.
    """
    preference = config.reasoning_effort
    if not _print_hf_routing_info(model_id, console):
        return

    if not preference:
        # Nothing to validate with a ping that we couldn't validate on the
        # first real call just as cheaply. Skip the probe entirely.
        _commit_switch(model_id, config, session, effective=None, cache=False)
        console.print(f"[green]Model switched to {model_id}[/green] [dim](effort: off)[/dim]")
        return

    console.print(f"[dim]checking {model_id} (effort: {preference})...[/dim]")
    try:
        outcome = await probe_effort(model_id, preference, hf_token)
    except ProbeInconclusive as e:
        _commit_switch(model_id, config, session, effective=None, cache=False)
        warning = render_llm_error_message(e)
        console.print(
            f"[yellow]Model switched to {model_id}[/yellow] "
            f"[dim](couldn't validate: {warning}; will verify on first message)[/dim]"
        )
        return
    except Exception as e:
        # Hard persistent error — auth, unknown model, quota. Don't switch.
        console.print(f"[bold red]Switch failed:[/bold red] {render_llm_error_message(e)}")
        console.print(f"[dim]Keeping current model: {config.model_name}[/dim]")
        return

    _commit_switch(
        model_id, config, session,
        effective=outcome.effective_effort, cache=True,
    )
    effort_label = outcome.effective_effort or "off"
    suffix = f" — {outcome.note}" if outcome.note else ""
    console.print(
        f"[green]Model switched to {model_id}[/green] "
        f"[dim](effort: {effort_label}{suffix}, {outcome.elapsed_ms}ms)[/dim]"
    )


def _commit_switch(model_id, config, session, effective, cache: bool) -> None:
    """Apply the switch to the session (or bare config if no session yet).

    ``effective`` is the probe's resolved effort; ``cache=True`` stores it
    in the session's per-model cache so real calls use the resolved level
    instead of re-probing. ``cache=False`` (inconclusive probe / effort
    off) leaves the cache untouched — next call falls back to preference.
    """
    if session is not None:
        session.update_model(model_id)
        if cache:
            session.model_effective_effort[model_id] = effective
        else:
            session.model_effective_effort.pop(model_id, None)
    else:
        config.model_name = model_id
