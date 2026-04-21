"""
Terminal display utilities — rich-powered CLI formatting.
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

_THEME = Theme({
    "tool.name": "bold rgb(255,200,80)",
    "tool.args": "dim",
    "tool.ok": "dim green",
    "tool.fail": "dim red",
    "info": "dim",
    "muted": "dim",
    # Markdown emphasis colors
    "markdown.strong": "bold rgb(255,200,80)",
    "markdown.emphasis": "italic rgb(180,140,40)",
    "markdown.code": "rgb(120,220,255)",
    "markdown.code_block": "rgb(120,220,255)",
    "markdown.link": "underline rgb(90,180,255)",
    "markdown.h1": "bold rgb(255,200,80)",
    "markdown.h2": "bold rgb(240,180,95)",
    "markdown.h3": "bold rgb(220,165,100)",
})

_console = Console(theme=_THEME, highlight=False)

# Indent prefix for all agent output (aligns under the `>` prompt)
_I = "  "


def get_console() -> Console:
    return _console


# ── Banner ─────────────────────────────────────────────────────────────

def print_banner(model: str | None = None, hf_user: str | None = None) -> None:
    """Print particle logo then CRT boot sequence with system info."""
    from agent.utils.particle_logo import run_particle_logo
    from agent.utils.crt_boot import run_boot_sequence

    # Particle coalesce logo — 1.5s converge, 2s hold
    run_particle_logo(_console, hold_seconds=2.0)

    # Clear screen for CRT boot — starts from top
    _console.file.write("\033[2J\033[H")
    _console.file.flush()

    model_label = model or "anthropic/claude-opus-4-6"
    user_label = hf_user or "not logged in"

    # Warm gold palette matching the shimmer highlight (255, 200, 80)
    gold = "rgb(255,200,80)"
    dim_gold = "rgb(180,140,40)"

    boot_lines = [
        (f"{_I}Initializing agent runtime...", gold),
        (f"{_I}  User: {user_label}", dim_gold),
        (f"{_I}  Model: {model_label}", dim_gold),
        (f"{_I}  Tools: loading...", dim_gold),
        ("", ""),
        (f"{_I}/help for commands · /model to switch · /quit to exit", gold),
    ]

    run_boot_sequence(_console, boot_lines)


# ── Init progress ──────────────────────────────────────────────────────

def print_init_done(tool_count: int = 0) -> None:
    import time
    f = _console.file
    # Overwrite the "Tools: loading..." line with actual count
    f.write(f"\033[A\033[A\033[A\033[K")  # Move up 3 lines (blank + help + blank) then up to tools line
    f.write(f"\033[A\033[K")
    gold = "\033[38;2;180;140;40m"
    reset = "\033[0m"
    tool_text = f"{_I}  Tools: {tool_count} loaded"
    for ch in tool_text:
        f.write(f"{gold}{ch}{reset}")
        f.flush()
        time.sleep(0.012)
    f.write("\n\n")
    # Reprint the help line
    f.write(f"{_I}\033[38;2;255;200;80m/help for commands · /model to switch · /quit to exit{reset}\n\n")
    # Ready message — minimal padding
    f.write(f"{_I}\033[38;2;255;200;80mReady. Let's build something impressive.{reset}\n")
    f.flush()


# ── Tool calls ─────────────────────────────────────────────────────────

def print_tool_call(tool_name: str, args_preview: str) -> None:
    import time
    f = _console.file
    # CRT-style: type out tool name in HF yellow
    gold = "\033[38;2;255;200;80m"
    reset = "\033[0m"
    f.write(f"{_I}{gold}▸ ")
    for ch in tool_name:
        f.write(ch)
        f.flush()
        time.sleep(0.015)
    f.write(f"{reset}  \033[2m{args_preview}{reset}\n")
    f.flush()


def print_tool_output(output: str, success: bool, truncate: bool = True) -> None:
    if truncate:
        output = _truncate(output, max_lines=10)
    style = "tool.ok" if success else "tool.fail"
    # Indent each line of tool output
    indented = "\n".join(f"{_I}  {line}" for line in output.split("\n"))
    _console.print(f"[{style}]{indented}[/{style}]")


class SubAgentDisplayManager:
    """Manages multiple concurrent sub-agent displays.

    Each agent gets its own stats and rolling tool-call log.
    All agents are rendered together so terminal escape-code
    erase/redraw stays consistent.
    """

    _MAX_VISIBLE = 2  # tool-call lines shown per agent

    def __init__(self):
        self._agents: dict[str, dict] = {}  # agent_id -> state dict
        self._lines_on_screen = 0
        self._ticker_task = None

    def start(self, agent_id: str, label: str = "research") -> None:
        import asyncio
        import time
        self._agents[agent_id] = {
            "label": label,
            "calls": [],
            "tool_count": 0,
            "token_count": 0,
            "start_time": time.monotonic(),
        }
        if not self._ticker_task:
            self._ticker_task = asyncio.ensure_future(self._tick())
        self._redraw()

    def set_tokens(self, agent_id: str, tokens: int) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["token_count"] = tokens

    def set_tool_count(self, agent_id: str, count: int) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["tool_count"] = count

    def add_call(self, agent_id: str, tool_desc: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["calls"].append(tool_desc)
            self._redraw()

    def clear(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)
        if not self._agents:
            if self._ticker_task:
                self._ticker_task.cancel()
                self._ticker_task = None
            self._erase()
            self._lines_on_screen = 0
        else:
            self._redraw()

    async def _tick(self) -> None:
        import asyncio
        try:
            while True:
                await asyncio.sleep(1.0)
                if self._agents:
                    self._redraw()
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _format_stats(agent: dict) -> str:
        import time
        start = agent["start_time"]
        if start is None:
            return ""
        elapsed = time.monotonic() - start
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        else:
            time_str = f"{elapsed / 60:.0f}m {elapsed % 60:.0f}s"
        tok = agent["token_count"]
        tok_str = f"{tok / 1000:.1f}k" if tok >= 1000 else str(tok)
        return f"{agent['tool_count']} tool uses · {tok_str} tokens · {time_str}"

    def _erase(self) -> None:
        if self._lines_on_screen > 0:
            f = _console.file
            for _ in range(self._lines_on_screen):
                f.write("\033[A\033[K")
            f.flush()

    def _redraw(self) -> None:
        f = _console.file
        self._erase()
        lines: list[str] = []
        for agent in self._agents.values():
            stats = self._format_stats(agent)
            label = agent["label"]
            header = f"{_I}\033[38;2;255;200;80m▸ {label}\033[0m"
            if stats:
                header += f"  \033[2m({stats})\033[0m"
            lines.append(header)
            visible = agent["calls"][-self._MAX_VISIBLE:]
            for desc in visible:
                lines.append(f"{_I}  \033[2m{desc}\033[0m")
        for line in lines:
            f.write(line + "\n")
        f.flush()
        self._lines_on_screen = len(lines)


_subagent_display = SubAgentDisplayManager()


def print_tool_log(tool: str, log: str, agent_id: str = "", label: str = "") -> None:
    """Handle tool log events — sub-agent calls get the rolling display."""
    if tool == "research":
        aid = agent_id or "research"
        if log == "Starting research sub-agent...":
            _subagent_display.start(aid, label or "research")
        elif log == "Research complete.":
            _subagent_display.clear(aid)
        elif log.startswith("tokens:"):
            _subagent_display.set_tokens(aid, int(log[7:]))
        elif log.startswith("tools:"):
            _subagent_display.set_tool_count(aid, int(log[6:]))
        else:
            _subagent_display.add_call(aid, log)
    else:
        _console.print(f"{_I}[dim]{tool}: {log}[/dim]")


# ── Messages ───────────────────────────────────────────────────────────

def print_markdown(text: str) -> None:
    import io, time, random
    from rich.padding import Padding

    _console.print()

    # Render markdown to a string buffer so we can type it out
    buf = io.StringIO()
    # Important: StringIO is not a TTY, so Rich would normally strip styles.
    # Force terminal rendering so ANSI style codes are preserved for typewriter output.
    buf_console = Console(
        file=buf,
        width=_console.width,
        highlight=False,
        theme=_THEME,
        force_terminal=True,
        color_system=_console.color_system or "truecolor",
    )
    buf_console.print(Padding(Markdown(text), (0, 0, 0, 2)))
    rendered = buf.getvalue()

    # Strip trailing whitespace from each line so we don't type across the full width
    lines = rendered.split("\n")
    rendered = "\n".join(line.rstrip() for line in lines)

    # CRT typewriter effect — fast, with occasional glitch
    rng = random.Random(42)
    f = _console.file
    for ch in rendered:
        f.write(ch)
        f.flush()
        if ch == "\n":
            time.sleep(0.002)
        elif ch == " ":
            time.sleep(0.002)
        elif rng.random() < 0.03:
            time.sleep(0.015)
        else:
            time.sleep(0.004)
    f.write("\n")
    f.flush()


def print_error(message: str) -> None:
    _console.print(f"\n{_I}[bold red]Error:[/bold red] {message}")


def print_turn_complete() -> None:
    pass  # no separator — clean output


def print_interrupted() -> None:
    _console.print(f"\n{_I}[dim italic]interrupted[/dim italic]")


def print_compacted(old_tokens: int, new_tokens: int) -> None:
    _console.print(f"{_I}[dim]context compacted: {old_tokens:,} → {new_tokens:,} tokens[/dim]")


# ── Approval ───────────────────────────────────────────────────────────

def print_approval_header(count: int) -> None:
    label = f"Approval required — {count} item{'s' if count != 1 else ''}"
    _console.print()
    _console.print(f"{_I}", Panel(f"[bold yellow]{label}[/bold yellow]", border_style="yellow", expand=False))


def print_approval_item(index: int, total: int, tool_name: str, operation: str) -> None:
    _console.print(f"\n{_I}[bold]\\[{index}/{total}][/bold]  [tool.name]{tool_name}[/tool.name]  {operation}")


def print_yolo_approve(count: int) -> None:
    _console.print(f"{_I}[bold yellow]yolo →[/bold yellow] auto-approved {count} item(s)")


# ── Help ───────────────────────────────────────────────────────────────

HELP_TEXT = f"""\
{_I}[bold]Commands[/bold]
{_I}  [cyan]/help[/cyan]            Show this help
{_I}  [cyan]/undo[/cyan]            Undo last turn
{_I}  [cyan]/compact[/cyan]         Compact context window
{_I}  [cyan]/model[/cyan] [id]      Show available models or switch
{_I}  [cyan]/yolo[/cyan]            Toggle auto-approve mode
{_I}  [cyan]/status[/cyan]          Current model & turn count
{_I}  [cyan]/quit[/cyan]            Exit"""


def print_help() -> None:
    _console.print()
    _console.print(HELP_TEXT)
    _console.print()


# ── Plan display ───────────────────────────────────────────────────────

def format_plan_display() -> str:
    """Format the current plan for display."""
    from agent.tools.plan_tool import get_current_plan

    plan = get_current_plan()
    if not plan:
        return ""

    completed = [t for t in plan if t["status"] == "completed"]
    in_progress = [t for t in plan if t["status"] == "in_progress"]
    pending = [t for t in plan if t["status"] == "pending"]

    lines = []
    for t in completed:
        lines.append(f"{_I}[green]✓[/green] [dim]{t['content']}[/dim]")
    for t in in_progress:
        lines.append(f"{_I}[yellow]▸[/yellow] {t['content']}")
    for t in pending:
        lines.append(f"{_I}[dim]○ {t['content']}[/dim]")

    summary = f"[dim]{len(completed)}/{len(plan)} done[/dim]"
    lines.append(f"{_I}{summary}")
    return "\n".join(lines)


def print_plan() -> None:
    plan_str = format_plan_display()
    if plan_str:
        _console.print(plan_str)


# ── Formatting for plan_tool output (used by plan_tool handler) ────────

def format_plan_tool_output(todos: list) -> str:
    if not todos:
        return "Plan is empty."

    lines = ["Plan updated:", ""]
    completed = [t for t in todos if t["status"] == "completed"]
    in_progress = [t for t in todos if t["status"] == "in_progress"]
    pending = [t for t in todos if t["status"] == "pending"]

    for t in completed:
        lines.append(f"  [x] {t['id']}. {t['content']}")
    for t in in_progress:
        lines.append(f"  [~] {t['id']}. {t['content']}")
    for t in pending:
        lines.append(f"  [ ] {t['id']}. {t['content']}")

    lines.append(f"\n{len(completed)}/{len(todos)} done")
    return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────

def _truncate(text: str, max_lines: int = 6) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
