"""Device selection helpers.

Auto-picks the best available device: CUDA > MPS (Apple Silicon) > CPU.
"""

from __future__ import annotations

import torch


def best_device(prefer: str | None = None) -> str:
    """Return the best available torch device as a string.

    Args:
        prefer: optional override. If a string and the device is available,
            it's returned as-is. "auto" or None triggers the priority
            cascade.

    Returns:
        One of "cuda", "mps", "cpu".
    """
    if prefer and prefer not in ("auto", None):
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def empty_cache(device: str) -> None:
    """Free cached memory for the given device, if applicable."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
