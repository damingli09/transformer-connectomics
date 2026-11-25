"""
Community detection over FC matrices.

Wraps python-louvain (the ``community`` package) and sklearn NMI for
adjusted/normalized mutual information between partitions.

Conventions
-----------
- FC matrices are symmetric N x N with entries in [-1, 1]. Louvain requires
  non-negative weights; we offer three preprocessing modes:
    "positive" (default): zero out negative entries
    "abs":                use |FC|
    "shift":              (FC - min) so all non-negative

A "partition" is a 1D integer array of length N giving the community id of
each node. Community ids are normalized to be 0..K-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import community as community_louvain  # python-louvain
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _to_numpy(m: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(m, torch.Tensor):
        return m.detach().cpu().numpy()
    return np.asarray(m)


def prepare_weights(
    matrix: torch.Tensor | np.ndarray,
    mode: str = "positive",
    symmetrize: bool = True,
) -> np.ndarray:
    """Convert a connectivity matrix into Louvain-ready non-negative weights.

    Args:
        matrix: (N, N) FC matrix.
        mode: "positive" | "abs" | "shift". See module docstring.
        symmetrize: if True, takes 0.5 * (M + M.T).

    Returns:
        (N, N) non-negative numpy float64 array with zero diagonal.
    """
    m = _to_numpy(matrix).astype(np.float64).copy()
    if symmetrize:
        m = 0.5 * (m + m.T)

    if mode == "positive":
        m = np.where(m > 0, m, 0.0)
    elif mode == "abs":
        m = np.abs(m)
    elif mode == "shift":
        m = m - m.min()
    else:
        raise ValueError(f"Unknown weight mode: {mode!r}")

    np.fill_diagonal(m, 0.0)
    return m


def _to_graph(weights: np.ndarray) -> nx.Graph:
    """(N, N) non-negative weights -> undirected weighted graph."""
    G = nx.from_numpy_array(weights)
    return G


# ---------------------------------------------------------------------------
# Single Louvain run
# ---------------------------------------------------------------------------


@dataclass
class LouvainResult:
    partition: np.ndarray   # (N,) community ids
    modularity: float
    n_communities: int
    resolution: float


def _normalize_partition(part_dict: dict[int, int]) -> np.ndarray:
    """Convert {node_id: cid} to dense (N,) array with ids renumbered 0..K-1."""
    N = len(part_dict)
    raw = np.array([part_dict[i] for i in range(N)], dtype=np.int64)
    _, inv = np.unique(raw, return_inverse=True)
    return inv.astype(np.int64)


def louvain(
    matrix: torch.Tensor | np.ndarray,
    resolution: float = 1.0,
    weight_mode: str = "positive",
    seed: int | None = 0,
) -> LouvainResult:
    """Single Louvain run at a fixed resolution.

    Args:
        matrix: (N, N) FC matrix.
        resolution: Louvain γ. Higher = more, smaller communities.
        weight_mode: see prepare_weights.
        seed: RNG seed for reproducibility.

    Returns:
        LouvainResult.
    """
    W = prepare_weights(matrix, mode=weight_mode, symmetrize=True)
    G = _to_graph(W)
    part = community_louvain.best_partition(G, resolution=resolution, random_state=seed)
    arr = _normalize_partition(part)
    Q = community_louvain.modularity(part, G)
    return LouvainResult(
        partition=arr,
        modularity=float(Q),
        n_communities=int(arr.max() + 1),
        resolution=float(resolution),
    )


# ---------------------------------------------------------------------------
# Resolution sweep
# ---------------------------------------------------------------------------


@dataclass
class GammaSweepResult:
    resolutions: np.ndarray         # (G,)
    modularities: np.ndarray        # (G,)
    n_communities: np.ndarray       # (G,)
    partitions: list[np.ndarray]    # G entries of shape (N,)
    best_idx: int

    @property
    def best(self) -> LouvainResult:
        i = self.best_idx
        return LouvainResult(
            partition=self.partitions[i],
            modularity=float(self.modularities[i]),
            n_communities=int(self.n_communities[i]),
            resolution=float(self.resolutions[i]),
        )


def sweep_resolution(
    matrix: torch.Tensor | np.ndarray,
    resolutions: np.ndarray | list[float] | None = None,
    weight_mode: str = "positive",
    seed: int | None = 0,
) -> GammaSweepResult:
    """Run Louvain across a range of γ values; report best by modularity."""
    if resolutions is None:
        resolutions = np.linspace(0.5, 2.0, 16)
    resolutions = np.asarray(resolutions, dtype=np.float64)

    Qs = np.zeros_like(resolutions)
    Ks = np.zeros_like(resolutions, dtype=np.int64)
    parts: list[np.ndarray] = []
    for i, gamma in enumerate(resolutions):
        r = louvain(matrix, resolution=float(gamma), weight_mode=weight_mode, seed=seed)
        Qs[i] = r.modularity
        Ks[i] = r.n_communities
        parts.append(r.partition)

    best_idx = int(np.argmax(Qs))
    return GammaSweepResult(
        resolutions=resolutions,
        modularities=Qs,
        n_communities=Ks,
        partitions=parts,
        best_idx=best_idx,
    )


# ---------------------------------------------------------------------------
# Partition similarity
# ---------------------------------------------------------------------------


def nmi(p1: np.ndarray, p2: np.ndarray, adjusted: bool = False) -> float:
    """Normalized mutual information between two partitions.

    Args:
        p1, p2: (N,) integer arrays.
        adjusted: if True, returns the adjusted MI (corrects for chance).

    Returns:
        NMI in [0, 1] (or AMI in [~-0.5, 1] if adjusted).
    """
    if adjusted:
        return float(adjusted_mutual_info_score(p1, p2))
    return float(normalized_mutual_info_score(p1, p2))


def labels_to_partition(labels: dict[int, int], n_nodes: int) -> np.ndarray:
    """Build a partition where unlabeled nodes get id -1 (singleton-like).

    Useful for comparing Louvain output to a sparse circuit-label set
    (e.g., the 26 IOI heads): assign each labeled head its circuit id
    and every other head a unique "background" id so NMI is meaningful.
    """
    part = np.full(n_nodes, -1, dtype=np.int64)
    for node, cid in labels.items():
        part[node] = cid

    # Background nodes get unique ids beyond max(cid).
    base = int(part.max()) + 1 if (part >= 0).any() else 0
    bg_mask = part < 0
    n_bg = int(bg_mask.sum())
    part[bg_mask] = np.arange(base, base + n_bg)
    return part


def stability(
    partitions: list[np.ndarray],
    adjusted: bool = False,
) -> float:
    """Mean pairwise NMI across a list of partitions.

    Use to assess Louvain stability under perturbations (corpus splits,
    γ jitter, RNG reseeds).
    """
    if len(partitions) < 2:
        raise ValueError("Need at least two partitions.")
    scores = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            scores.append(nmi(partitions[i], partitions[j], adjusted=adjusted))
    return float(np.mean(scores))
