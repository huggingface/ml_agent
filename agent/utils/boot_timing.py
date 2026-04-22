"""Shared timing and color helpers for startup visual effects."""

import math


def settle_curve(progress: float, sharpness: float = 3.0) -> float:
    """Return noise amount in range 1..0 for normalized progress 0..1."""
    t = max(0.0, min(1.0, progress))
    return math.exp(-sharpness * t)


def warm_gold_from_white(progress: float) -> tuple[int, int, int]:
    """Interpolate from white to warm gold for progress 0..1."""
    t = max(0.0, min(1.0, progress))
    return 255, int(255 - 55 * t), int(255 - 175 * t)
