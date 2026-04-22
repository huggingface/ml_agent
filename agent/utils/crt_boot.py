"""CRT / glitch boot sequence effect for CLI startup.

Simulates an old CRT terminal booting up: text appearing character by character
with noise artifacts, then settling into a clean display.
"""

import random
import time

from rich.console import Console
from rich.text import Text
from rich.live import Live

from agent.utils.boot_timing import settle_curve


def _glitch_text(text: str, intensity: float, rng: random.Random) -> str:
    """Add random glitch characters to text."""
    glitch_chars = "█▓▒░┃┫┣╋╏╎─━┅┄"
    result = list(text)
    for i in range(len(result)):
        if rng.random() < intensity:
            result[i] = rng.choice(glitch_chars)
    return "".join(result)


def run_boot_sequence(console: Console, boot_lines: list[tuple[str, str]]) -> None:
    """Run the CRT boot sequence effect.

    Args:
        console: Rich console instance.
        boot_lines: List of (text, rich_style) tuples to display.
    """
    term_height = min(console.height - 2, 40)
    rng = random.Random(42)

    with Live(console=console, refresh_per_second=30, transient=True) as live:
        displayed_lines: list[tuple[str, str]] = []

        for line_text, line_style in boot_lines:
            if not line_text:
                displayed_lines.append(("", ""))
                continue

            line_len = max(1, len(line_text))
            # Type out each character
            for char_idx in range(len(line_text) + 1):
                result = Text()
                progress = char_idx / line_len
                noise = settle_curve(progress)
                prev_glitch_chance = 0.01 + 0.06 * noise
                prev_glitch_intensity = 0.02 + 0.12 * noise
                scanline_chance = 0.005 + 0.03 * noise

                # Render previously completed lines
                for prev_text, prev_style in displayed_lines:
                    if rng.random() < prev_glitch_chance:
                        result.append(_glitch_text(prev_text, prev_glitch_intensity, rng), style=prev_style)
                    else:
                        result.append(prev_text, style=prev_style)
                    result.append("\n")

                # Current line being typed
                typed = line_text[:char_idx]
                cursor = "█" if char_idx < len(line_text) else ""

                # Noise after cursor
                noise_tail = ""
                if char_idx < len(line_text):
                    noise_len = rng.randint(0, int(1 + 5 * noise))
                    noise_tail = "".join(rng.choice("░▒▓") for _ in range(noise_len))

                result.append(typed, style=line_style)
                result.append(cursor, style="bold rgb(255,200,80)")
                result.append(noise_tail, style="dim rgb(180,140,40)")
                result.append("\n")

                # Faint scanlines in remaining space
                remaining = term_height - len(displayed_lines) - 2
                for _ in range(max(0, remaining)):
                    if rng.random() < scanline_chance:
                        scan_len = rng.randint(5, 30)
                        result.append("─" * scan_len, style="dim rgb(180,140,40)")
                    result.append("\n")

                live.update(result)

                # Variable typing speed
                if line_text[char_idx - 1:char_idx] in " .":
                    time.sleep(0.025)
                else:
                    time.sleep(0.010)

            displayed_lines.append((line_text, line_style))
            time.sleep(0.06)

        # Hold with blinking cursor
        for frame in range(20):
            result = Text()
            for prev_text, prev_style in displayed_lines:
                result.append(prev_text, style=prev_style)
                result.append("\n")
            if frame % 8 < 4:
                result.append("█", style="rgb(255,200,80)")
            live.update(result)
            time.sleep(0.05)

    # Print final clean frame
    final = Text()
    for prev_text, prev_style in displayed_lines:
        final.append(prev_text, style=prev_style)
        final.append("\n")
    console.print(final)
