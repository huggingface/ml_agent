"""
Terminal display utilities — rich-powered CLI formatting.
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

_THEME = Theme({
    "tool.name": "bold cyan",
    "tool.args": "dim",
    "tool.ok": "dim green",
    "tool.fail": "dim red",
    "info": "dim",
    "muted": "dim",
})

_console = Console(theme=_THEME, highlight=False)


def get_console() -> Console:
    return _console


# ── Banner ─────────────────────────────────────────────────────────────

_BANNER = r"""
 [yellow] _  _ ___     _                _   [/yellow]
 [yellow]| || | __|   /_\  __ _ ___ _ _| |_ [/yellow]
 [yellow]| __ | _|   / _ \/ _` / -_) ' \  _|[/yellow]
 [yellow]|_||_|_|   /_/ \_\__, \___|_||_\__|[/yellow]
 [yellow]                 |___/             [/yellow]"""


def print_banner() -> None:
    _console.print()
    _console.print(
        Panel(
            _BANNER,
            subtitle="[dim]/help · /quit[/dim]",
            border_style="dim yellow",
            expand=False,
            padding=(0, 1),
        )
    )
    _console.print()


# ── Init progress ──────────────────────────────────────────────────────

def print_init_done() -> None:
    _console.print("[dim]Ready.[/dim]\n")


# ── Tool calls ─────────────────────────────────────────────────────────

def print_tool_call(tool_name: str, args_preview: str) -> None:
    _console.print(f"  [tool.name]▸ {tool_name}[/tool.name]  [tool.args]{args_preview}[/tool.args]")


def print_tool_output(output: str, success: bool, truncate: bool = True) -> None:
    if truncate:
        output = _truncate(output, max_lines=6)
    style = "tool.ok" if success else "tool.fail"
    _console.print(f"  [{style}]{output}[/{style}]")


def print_tool_log(tool: str, log: str) -> None:
    _console.print(f"  [dim]{tool}:[/dim] [dim]{log}[/dim]")


# ── Messages ───────────────────────────────────────────────────────────

def print_markdown(text: str) -> None:
    _console.print()
    _console.print(Markdown(text))


def print_error(message: str) -> None:
    _console.print(f"\n[bold red]Error:[/bold red] {message}")


def print_turn_complete() -> None:
    # Subtle separator — no noisy "turn complete" banner
    _console.print("[dim]─[/dim]")


def print_interrupted() -> None:
    _console.print("\n[dim italic]interrupted[/dim italic]")


def print_compacted(old_tokens: int, new_tokens: int) -> None:
    _console.print(f"  [dim]context compacted: {old_tokens:,} → {new_tokens:,} tokens[/dim]")


# ── Approval ───────────────────────────────────────────────────────────

def print_approval_header(count: int) -> None:
    label = f"Approval required — {count} item{'s' if count != 1 else ''}"
    _console.print()
    _console.print(Panel(f"[bold yellow]{label}[/bold yellow]", border_style="yellow", expand=False))


def print_approval_item(index: int, total: int, tool_name: str, operation: str) -> None:
    _console.print(f"\n  [bold]\\[{index}/{total}][/bold]  [tool.name]{tool_name}[/tool.name]  {operation}")


def print_yolo_approve(count: int) -> None:
    _console.print(f"  [bold yellow]yolo →[/bold yellow] auto-approved {count} item(s)")


# ── Help ───────────────────────────────────────────────────────────────

HELP_TEXT = """\
[bold]Commands[/bold]
  [cyan]/help[/cyan]            Show this help
  [cyan]/undo[/cyan]            Undo last turn
  [cyan]/compact[/cyan]         Compact context window
  [cyan]/model[/cyan] [id]      Show available models or switch
  [cyan]/yolo[/cyan]            Toggle auto-approve mode
  [cyan]/status[/cyan]          Current model & turn count
  [cyan]/quit[/cyan]            Exit"""


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
        lines.append(f"  [green]✓[/green] [dim]{t['content']}[/dim]")
    for t in in_progress:
        lines.append(f"  [yellow]▸[/yellow] {t['content']}")
    for t in pending:
        lines.append(f"  [dim]○ {t['content']}[/dim]")

    summary = f"[dim]{len(completed)}/{len(plan)} done[/dim]"
    lines.append(f"  {summary}")
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
