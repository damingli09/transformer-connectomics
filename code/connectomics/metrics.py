"""
Statistics linking SC and FC to external ground truth.

Three families:

- ``sc_fc_coupling`` (kappa): Pearson correlation between SC and FC over the
  directed-forward (i.e., strictly upper-triangular) entries.
- ``layer_depth_kappa``: kappa split by depth band (early / middle / late
  layers) to characterize the depth gradient.
- ``circuit_nmi`` / ``circuit_purity``: agreement between a discovered Louvain
  partition and a hand-specified circuit-label set (e.g., the IOI heads).
"""

from __future__ import annotations

import numpy as np
import torch

from .community import labels_to_partition, nmi


def _upper_triangular_pair(
    sc: torch.Tensor, fc: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired SC and FC entries over the strictly upper-triangular indices.

    Only the SC entries (i, j) with layer(i) < layer(j), i.e. structurally
    non-zero, are used. FC is symmetric, so FC[i, j] is the same as FC[j, i].
    """
    N = sc.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    sc_vec = sc[iu[0], iu[1]].detach().cpu().numpy()
    fc_vec = fc[iu[0], iu[1]].detach().cpu().numpy()
    # Restrict to entries where SC > 0 to avoid driving coupling with zeros.
    mask = sc_vec > 0
    return sc_vec[mask], fc_vec[mask]


def sc_fc_coupling(sc: torch.Tensor, fc: torch.Tensor) -> float:
    """Pearson correlation between SC and FC over forward directed entries (kappa)."""
    sc_v, fc_v = _upper_triangular_pair(sc, fc)
    if sc_v.size < 2:
        return float("nan")
    return float(np.corrcoef(sc_v, fc_v)[0, 1])


def layer_depth_kappa(
    sc: torch.Tensor,
    fc: torch.Tensor,
    n_layers: int,
    n_heads: int,
    bands: tuple[float, float] = (1 / 3, 2 / 3),
) -> dict[str, float]:
    """Compute kappa within depth bands.

    Bands split layers into early / middle / late by the two cutoffs.
    For each band, kappa is the Pearson correlation between SC and FC over
    edges with both endpoints in that band.

    Args:
        sc: (N, N) SC matrix.
        fc: (N, N) FC matrix.
        n_layers: model layer count (for band boundaries).
        n_heads: heads per layer (for index->layer mapping).
        bands: two fractions in (0, 1), strictly increasing, that split layers.

    Returns:
        dict with keys "early", "middle", "late", "global".
    """
    N = n_layers * n_heads
    iu = torch.triu_indices(N, N, offset=1)
    sc_vec = sc[iu[0], iu[1]].cpu().numpy()
    fc_vec = fc[iu[0], iu[1]].cpu().numpy()
    layer_of_node = (np.arange(N) // n_heads)
    layer_i = layer_of_node[iu[0].cpu().numpy()]
    layer_j = layer_of_node[iu[1].cpu().numpy()]

    b1 = int(np.floor(bands[0] * n_layers))
    b2 = int(np.floor(bands[1] * n_layers))

    def band_of(l):
        return np.where(l < b1, 0, np.where(l < b2, 1, 2))

    bi = band_of(layer_i)
    bj = band_of(layer_j)

    def kappa_for_band(b):
        mask = (bi == b) & (bj == b) & (sc_vec > 0)
        if mask.sum() < 2:
            return float("nan")
        return float(np.corrcoef(sc_vec[mask], fc_vec[mask])[0, 1])

    return {
        "early": kappa_for_band(0),
        "middle": kappa_for_band(1),
        "late": kappa_for_band(2),
        "global": sc_fc_coupling(sc, fc),
    }


def circuit_nmi(
    discovered_partition: np.ndarray,
    circuit_labels: dict[int, int],
    n_nodes: int,
    adjusted: bool = False,
    restricted: bool = True,
) -> float:
    """NMI between a Louvain partition and a sparse circuit-label set.

    Args:
        discovered_partition: (N,) array of community ids.
        circuit_labels: {node_idx: class_id} for labeled nodes only.
        n_nodes: total number of nodes (for the unrestricted path).
        adjusted: use AMI rather than NMI.
        restricted: if True (default), compute NMI only on the labeled
            subset. This is the methodologically correct version: it asks
            "do labeled heads in the same circuit class end up in the same
            Louvain community?" without the entropy of the background.

            If False, assign each unlabeled head a unique singleton id and
            compute NMI on the full set of nodes. This severely
            under-estimates NMI when the background is large relative to the
            labeled set.

    Returns:
        NMI in [0, 1]. For a single-class label set, NMI is degenerate and
        returns 0.0 (use ``circuit_purity`` instead).
    """
    if not circuit_labels:
        return float("nan")

    if restricted:
        labeled_nodes = sorted(circuit_labels.keys())
        target = np.array([circuit_labels[n] for n in labeled_nodes], dtype=np.int64)
        discovered = np.array([discovered_partition[n] for n in labeled_nodes], dtype=np.int64)
        if len(np.unique(target)) < 2:
            # NMI is undefined for a single-class target; use circuit_purity.
            return 0.0
        return nmi(discovered, target, adjusted=adjusted)

    target = labels_to_partition(circuit_labels, n_nodes)
    return nmi(discovered_partition, target, adjusted=adjusted)


def circuit_purity(
    discovered_partition: np.ndarray,
    circuit_labels: dict[int, int],
) -> float:
    """Fraction of labeled heads in their class's majority community.

    For each circuit class c, find the Louvain community that contains
    the most heads of class c, count those heads, sum over classes, and
    divide by the total number of labeled heads.

    Returns purity in [0, 1].
    """
    if not circuit_labels:
        return float("nan")
    labeled_nodes = sorted(circuit_labels.keys())
    class_of = {n: circuit_labels[n] for n in labeled_nodes}
    by_class: dict[int, list[int]] = {}
    for n in labeled_nodes:
        by_class.setdefault(class_of[n], []).append(int(discovered_partition[n]))

    total_purity_num = 0
    total = 0
    for cls, cids in by_class.items():
        counts = np.bincount(cids)
        total_purity_num += int(counts.max())
        total += len(cids)
    return total_purity_num / max(total, 1)
