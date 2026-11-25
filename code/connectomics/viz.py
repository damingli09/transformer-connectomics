"""
Connectome visualizations.

Three plot helpers:

- ``plot_matrix``: heatmap of an (N, N) connectivity matrix with heavy
  layer-boundary lines and "L0..L{n-1}" tick labels at the midpoint of
  each layer's block.
- ``plot_communities``: FC heatmap with nodes reordered by community.
- ``plot_modularity_sweep``: modularity Q (and community count K) versus γ.

Conventions
-----------
- Heads are ordered so within-layer heads are contiguous and layers are
  in depth order: flat index = layer * n_heads + head_in_layer.
- For non-negative matrices (SC) we use sequential colormaps and
  percentile-clipped value ranges. For signed matrices (FC, ΔFC) we use
  divergent colormaps symmetric around zero.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_np(m: torch.Tensor | np.ndarray) -> np.ndarray:
    return m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)


def _percentile_range(
    m: np.ndarray,
    signed: bool,
    clip_pct: float = 99.0,
) -> tuple[float, float]:
    """Robust color limits.

    For signed matrices, returns symmetric (-v, +v) where v is the clip_pct
    percentile of |m|. For non-negative matrices, returns (0, v) where v is
    the clip_pct percentile of m.
    """
    if signed:
        v = float(np.percentile(np.abs(m), clip_pct))
        if v <= 0:
            v = float(np.abs(m).max() or 1.0)
        return -v, v
    v = float(np.percentile(m[m > 0] if (m > 0).any() else m, clip_pct))
    if v <= 0:
        v = float(m.max() or 1.0)
    return 0.0, v


def _draw_layer_boundaries(
    ax: plt.Axes,
    n_layers: int,
    n_heads: int,
    color: str = "black",
    lw: float = 0.8,
    alpha: float = 0.9,
):
    """Heavy lines between layer blocks (every n_heads positions)."""
    for l in range(1, n_layers):
        x = l * n_heads - 0.5
        ax.axhline(x, color=color, lw=lw, alpha=alpha)
        ax.axvline(x, color=color, lw=lw, alpha=alpha)


def _layer_tick_setup(ax: plt.Axes, n_layers: int, n_heads: int):
    """Replace head-index ticks with 'L0', 'L1', ... at the midpoint of each block."""
    midpoints = [l * n_heads + n_heads / 2 - 0.5 for l in range(n_layers)]
    labels = [f"L{l}" for l in range(n_layers)]
    ax.set_xticks(midpoints)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_yticks(midpoints)
    ax.set_yticklabels(labels, fontsize=8)


def plot_matrix(
    matrix: torch.Tensor | np.ndarray,
    n_layers: int,
    n_heads: int,
    title: str = "",
    signed: bool | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    clip_pct: float = 99.0,
    ax: plt.Axes | None = None,
    cbar_label: str = "",
):
    """Heatmap with layer-grouped depth ordering and layer-boundary lines.

    Args:
        matrix: (N, N) connectivity matrix; N = n_layers * n_heads.
        signed: if True, treat as a signed matrix (divergent colormap, symmetric
            limits). If None, auto-detected from whether the matrix has any
            entries < -1e-6.
        cmap: matplotlib colormap. Defaults: ``"RdBu_r"`` if signed,
            ``"magma"`` otherwise.
        vmin, vmax: explicit color limits. If None, derived from
            ``clip_pct``-percentile of the data.
        clip_pct: percentile used to set color limits when vmin/vmax not given.
        cbar_label: label for the colorbar.
    """
    m = _to_np(matrix)
    if signed is None:
        signed = bool(np.any(m < -1e-6))
    if cmap is None:
        cmap = "RdBu_r" if signed else "magma"

    if vmin is None or vmax is None:
        v0, v1 = _percentile_range(m, signed=signed, clip_pct=clip_pct)
        vmin = v0 if vmin is None else vmin
        vmax = v1 if vmax is None else vmax

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    _draw_layer_boundaries(ax, n_layers, n_heads)
    _layer_tick_setup(ax, n_layers, n_heads)
    ax.set_xlabel("target layer")
    ax.set_ylabel("source layer")
    if title:
        ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9)
    return ax, im


def plot_communities(
    matrix: torch.Tensor | np.ndarray,
    partition: np.ndarray,
    n_layers: int,
    n_heads: int,
    title: str = "",
    ax: plt.Axes | None = None,
    clip_pct: float = 99.0,
):
    """FC heatmap with nodes reordered by community.

    Within-community blocks appear as dense diagonal patches. Community
    boundaries are drawn in black; for reference, layer-of-origin tick
    labels are written along the top axis.
    """
    m = _to_np(matrix)
    order = np.argsort(partition, kind="stable")
    m_sorted = m[np.ix_(order, order)]

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    vmin, vmax = _percentile_range(m_sorted, signed=True, clip_pct=clip_pct)
    im = ax.imshow(m_sorted, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")

    # Community boundaries.
    sorted_labels = partition[order]
    boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", lw=1.0)
        ax.axvline(b - 0.5, color="black", lw=1.0)

    # Mark each node's originating layer along the top axis with a thin tick.
    layer_of_each = order // n_heads  # in sorted order
    ax.set_xticks(np.arange(len(order)), minor=True)
    ax.set_xticklabels([str(int(layer_of_each[i])) if i % n_heads == 0 else "" for i in range(len(order))],
                       minor=True, fontsize=5)

    ax.set_xlabel("nodes (sorted by community)")
    ax.set_ylabel("nodes (sorted by community)")
    if title:
        ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    return ax


def plot_modularity_sweep(
    resolutions: Sequence[float],
    modularities: Sequence[float],
    n_communities: Sequence[int],
    ax: plt.Axes | None = None,
):
    """Two-axis plot: Q (left) and K (right) vs. resolution γ."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(resolutions, modularities, marker="o", color="C0", label="Modularity Q")
    ax.set_xlabel("Louvain resolution γ")
    ax.set_ylabel("Modularity Q", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    ax2 = ax.twinx()
    ax2.plot(resolutions, n_communities, marker="s", color="C3", label="# communities K")
    ax2.set_ylabel("# communities K", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    return ax, ax2
