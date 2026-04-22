"""Particle coalesce effect for the HUGGING FACE ML INTERN logo.

Random particles swirl in from the edges, converge to form the text
"HUGGING FACE / ML INTERN", hold briefly, then the final frame is printed.
Rendered with braille characters for high detail.

Based on Leandro's particle_coalesce.py demo.
"""

import math
import random
import time

from rich.console import Console
from rich.text import Text
from rich.align import Align
from rich.live import Live

from agent.utils.braille import BrailleCanvas, text_to_pixels
from agent.utils.boot_timing import settle_curve, warm_gold_from_white


class Particle:
    __slots__ = ("x", "y", "target_x", "target_y", "vx", "vy", "phase", "delay")

    def __init__(self, x: float, y: float, target_x: float, target_y: float, delay: float = 0):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.vx = 0.0
        self.vy = 0.0
        self.phase = random.uniform(0, math.pi * 2)
        self.delay = delay

    def update_converge(self, t: float, strength: float = 0.08, damping: float = 0.92):
        """Move toward target with spring-like physics."""
        if t < self.delay:
            # Still in swirl phase
            self.x += self.vx
            self.y += self.vy
            self.vx *= 0.99
            self.vy *= 0.99
            # Gentle spiral
            angle = self.phase + t * 2
            self.vx += math.cos(angle) * 0.3
            self.vy += math.sin(angle) * 0.3
            return

        # Spring toward target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        self.vx += dx * strength
        self.vy += dy * strength
        self.vx *= damping
        self.vy *= damping
        self.x += self.vx
        self.y += self.vy

    @property
    def at_target(self) -> bool:
        return abs(self.x - self.target_x) < 1.5 and abs(self.y - self.target_y) < 1.5


def run_particle_logo(console: Console, hold_seconds: float = 1.5) -> None:
    """Run the particle coalesce effect."""
    term_width = min(console.width, 120)
    term_height = min(console.height - 4, 35)

    canvas = BrailleCanvas(term_width, term_height)

    # Get target positions from text
    text_pixels_line1 = text_to_pixels("HUGGING FACE", scale=2)
    text_pixels_line2 = text_to_pixels("ML INTERN", scale=2)

    # Calculate dimensions for centering
    def get_bounds(pixels):
        if not pixels:
            return 0, 0, 0, 0
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        return min(xs), max(xs), min(ys), max(ys)

    min_x1, max_x1, min_y1, max_y1 = get_bounds(text_pixels_line1)
    min_x2, max_x2, min_y2, max_y2 = get_bounds(text_pixels_line2)

    w1, h1 = max_x1 - min_x1 + 1, max_y1 - min_y1 + 1
    w2, h2 = max_x2 - min_x2 + 1, max_y2 - min_y2 + 1

    total_h = h1 + 6 + h2  # gap between lines
    start_y = (canvas.pixel_height - total_h) // 2

    # Center line 1
    offset_x1 = (canvas.pixel_width - w1) // 2 - min_x1
    offset_y1 = start_y - min_y1
    targets_1 = [(p[0] + offset_x1, p[1] + offset_y1) for p in text_pixels_line1]

    # Center line 2
    offset_x2 = (canvas.pixel_width - w2) // 2 - min_x2
    offset_y2 = start_y + h1 + 6 - min_y2
    targets_2 = [(p[0] + offset_x2, p[1] + offset_y2) for p in text_pixels_line2]

    all_targets = targets_1 + targets_2

    # Subsample for performance — take every Nth pixel
    step = max(1, len(all_targets) // 1500)
    sampled_targets = all_targets[::step]

    # Create particles at random edge positions
    rng = random.Random(42)
    particles = []
    pw, ph = canvas.pixel_width, canvas.pixel_height

    for i, (tx, ty) in enumerate(sampled_targets):
        # Spawn from random edge
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            sx, sy = rng.uniform(0, pw), rng.uniform(-20, -5)
        elif side == "bottom":
            sx, sy = rng.uniform(0, pw), rng.uniform(ph + 5, ph + 20)
        elif side == "left":
            sx, sy = rng.uniform(-20, -5), rng.uniform(0, ph)
        else:
            sx, sy = rng.uniform(pw + 5, pw + 20), rng.uniform(0, ph)

        delay = rng.uniform(0, 0.4)  # staggered start
        p = Particle(sx, sy, tx, ty, delay=delay)
        # Initial velocity — gentle swirl
        angle = math.atan2(ph / 2 - sy, pw / 2 - sx) + rng.gauss(0, 0.8)
        speed = rng.uniform(1.0, 2.5)
        p.vx = math.cos(angle) * speed
        p.vy = math.sin(angle) * speed
        particles.append(p)

    # Also add some extra ambient particles that never converge
    ambient = []
    for _ in range(200):
        ax = rng.uniform(0, pw)
        ay = rng.uniform(0, ph)
        ap = Particle(ax, ay, ax, ay)
        ap.vx = rng.gauss(0, 1)
        ap.vy = rng.gauss(0, 1)
        ambient.append(ap)

    # Timing: 1s converge + 2s hold = 3s total
    fps = 24
    converge_frames = int(fps * 0.9)
    hold_frames = int(fps * hold_seconds)
    total_frames = converge_frames + hold_frames

    with Live(console=console, refresh_per_second=fps, transient=True) as live:
        for frame in range(total_frames):
            canvas.clear()
            t = frame * 0.03

            # Update ambient particles (always drifting)
            for ap in ambient:
                ap.x += ap.vx + math.sin(t + ap.phase) * 0.5
                ap.y += ap.vy + math.cos(t + ap.phase * 1.3) * 0.5
                # Wrap around
                ap.x = ap.x % pw
                ap.y = ap.y % ph

                # Fade out ambient during hold phase
                if frame < converge_frames:
                    alpha = 0.3 + 0.2 * math.sin(t * 2 + ap.phase)
                else:
                    fade = (frame - converge_frames) / hold_frames
                    alpha = (0.3 + 0.2 * math.sin(t * 2 + ap.phase)) * (1 - fade)
                if alpha > 0.25:
                    canvas.set_pixel(int(ap.x), int(ap.y))

            if frame < converge_frames:
                # Converge phase
                progress = frame / converge_frames
                noise = settle_curve(progress)
                for p in particles:
                    p.update_converge(t, strength=0.06, damping=0.90)
                    canvas.set_pixel(int(p.x), int(p.y))

                    # Trail effect
                    trail_scale = 0.2 + 0.5 * noise
                    trail_x = int(p.x - p.vx * trail_scale)
                    trail_y = int(p.y - p.vy * trail_scale)
                    canvas.set_pixel(trail_x, trail_y)

                # Color transitions from white to warm gold
                r, g, b = warm_gold_from_white(progress)
            else:
                # Hold phase — settle into solid logo
                settle_t = (frame - converge_frames) / hold_frames
                for p in particles:
                    # Jitter decays to zero
                    jitter = (1 - settle_t) * 0.7
                    jx = p.target_x + math.sin(t * 3 + p.phase) * jitter
                    jy = p.target_y + math.cos(t * 3 + p.phase * 1.5) * jitter
                    canvas.set_pixel(int(jx), int(jy))
                    canvas.set_pixel(int(p.target_x), int(p.target_y))

                r, g, b = 255, 200, 80

            # Render with color
            lines = canvas.render()
            result = Text()
            for line in lines:
                for ch in line:
                    if ch == chr(0x2800):
                        result.append(ch)
                    else:
                        result.append(ch, style=f"rgb({r},{g},{b})")
                result.append("\n")

            live.update(Align.center(result))
            time.sleep(1.0 / fps)

    # Print final settled frame
    canvas.clear()
    for p in particles:
        canvas.set_pixel(int(p.target_x), int(p.target_y))
    final = Text()
    for line in canvas.render():
        for ch in line:
            if ch == chr(0x2800):
                final.append(ch)
            else:
                final.append(ch, style="rgb(255,200,80)")
        final.append("\n")
    console.print(Align.center(final))
