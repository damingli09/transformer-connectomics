"""
Null distributions for hypothesis testing.

Three generators, one per quantity:

- ``fc_null_from_phase_randomization``: shuffle the temporal phase of every
  per-head activity timeseries (FFT phase scramble). Preserves each head's
  power spectrum and marginal distribution but destroys cross-head temporal
  coupling. Use as a null for FC entries and SC-FC coupling.

- ``configuration_model_modularity``: rewire a non-negative weighted graph
  preserving its (weighted) degree sequence, then Louvain. Use as a null
  for the modularity statistic Q.

- ``permute_within_layer_sc``: permute attention-head indices within each
  layer, recompute SC. Preserves the within-layer marginal distribution of
  W_Q/W_K/W_V/W_O but breaks specific head identities. Use as a null for
  SC entries.

All three return numpy arrays of statistics; the caller computes empirical
p-values.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .community import louvain, prepare_weights
from .fc import fc_l2
from .sc import flatten_upper_triangular


# ---------------------------------------------------------------------------
# Phase randomization for FC nulls
# ---------------------------------------------------------------------------


def _phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-scramble a 1D real signal preserving its power spectrum.

    Preserves DC (mean) and Nyquist (for even T) by keeping their original
    real-valued amplitudes — these bins must remain real for the inverse
    transform to give a real signal.
    """
    T = x.shape[0]
    X = np.fft.rfft(x)
    n_freqs = X.shape[0]
    # Original phase: keep for DC (and Nyquist, even T).
    original_phase = np.angle(X)
    new_phase = rng.uniform(0, 2 * np.pi, size=n_freqs)
    new_phase[0] = original_phase[0]
    if T % 2 == 0:
        new_phase[-1] = original_phase[-1]
    X_perm = np.abs(X) * np.exp(1j * new_phase)
    return np.fft.irfft(X_perm, n=T)


def phase_randomize_signals(
    signals: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """Phase-scramble each column of (T, N) timeseries.

    Args:
        signals: (T, N) tensor of per-head activity signals.
        seed: RNG seed.

    Returns:
        (T, N) tensor with each column independently phase-randomized.
    """
    rng = np.random.default_rng(seed)
    x = signals.detach().cpu().numpy().astype(np.float64)
    out = np.empty_like(x)
    for n in range(x.shape[1]):
        out[:, n] = _phase_randomize(x[:, n], rng)
    return torch.from_numpy(out).float()


def fc_null_from_phase_randomization(
    activity_l2: torch.Tensor,
    n_permutations: int = 1000,
    seed: int = 0,
    statistic_fn=None,
) -> np.ndarray:
    """Build a null distribution over an FC-derived statistic via phase scrambling.

    Args:
        activity_l2: (T, L, H) L2-norm activity tensor (i.e. ``Activity.l2``).
        n_permutations: number of null draws.
        seed: base RNG seed; each permutation uses seed + i.
        statistic_fn: callable mapping (N, N) FC tensor -> scalar statistic.
            Default: mean of the off-diagonal entries.

    Returns:
        (n_permutations,) numpy array of statistics under the null.
    """
    from .activity import Activity

    T, L, H = activity_l2.shape
    flat = activity_l2.reshape(T, L * H)

    if statistic_fn is None:
        def statistic_fn(fc):  # type: ignore[no-redef]
            iu = torch.triu_indices(fc.shape[0], fc.shape[0], offset=1)
            return float(fc[iu[0], iu[1]].mean().item())

    nulls = np.empty(n_permutations, dtype=np.float64)
    for i in tqdm(range(n_permutations), desc="phase-rand FC null"):
        permuted = phase_randomize_signals(flat, seed=seed + i)
        # Wrap in a faux Activity to reuse fc_l2 if desired; we go raw here.
        fake = Activity(
            l2=permuted.reshape(T, L, H),
            residual_norm=torch.zeros(T),
            n_layers=L,
            n_heads=H,
            d_head=0,
            d_model=0,
        )
        fc = fc_l2(fake)
        nulls[i] = statistic_fn(fc)
    return nulls


# ---------------------------------------------------------------------------
# Configuration-model nulls for modularity
# ---------------------------------------------------------------------------


def _double_edge_swap_weighted(
    W: np.ndarray, n_swaps: int, rng: np.random.Generator
) -> np.ndarray:
    """In-place weighted double-edge swap.

    Repeatedly pick two edges (a, b) and (c, d) with distinct endpoints,
    swap to (a, d) and (c, b) preserving the (weighted) degree sequence.
    Crude but standard for null modularity.
    """
    N = W.shape[0]
    W = W.copy()
    iu = np.triu_indices(N, k=1)
    edges_i, edges_j = iu
    nz = W[edges_i, edges_j] > 0
    pool_i = edges_i[nz]
    pool_j = edges_j[nz]
    n_edges = pool_i.shape[0]
    if n_edges < 2:
        return W

    for _ in range(n_swaps):
        a_idx, c_idx = rng.integers(0, n_edges, size=2)
        a, b = pool_i[a_idx], pool_j[a_idx]
        c, d = pool_i[c_idx], pool_j[c_idx]
        if len({a, b, c, d}) < 4:
            continue
        w1, w2 = W[a, b], W[c, d]
        W[a, b] = 0.0
        W[b, a] = 0.0
        W[c, d] = 0.0
        W[d, c] = 0.0
        W[a, d] += w1
        W[d, a] += w1
        W[c, b] += w2
        W[b, c] += w2
    return W


def modularity_null_configuration(
    fc_matrix: torch.Tensor,
    n_permutations: int = 1000,
    n_swaps_per: int | None = None,
    resolution: float = 1.0,
    weight_mode: str = "positive",
    seed: int = 0,
) -> np.ndarray:
    """Null distribution over Louvain modularity Q under degree-preserving rewiring.

    Args:
        fc_matrix: (N, N) FC matrix.
        n_permutations: number of null draws.
        n_swaps_per: number of double-edge swaps per draw. Defaults to
            10 * (number of edges).
        resolution: Louvain γ to apply on each draw.
        weight_mode: passed to prepare_weights.
        seed: base RNG seed.

    Returns:
        (n_permutations,) numpy array of modularity values under the null.
    """
    W = prepare_weights(fc_matrix, mode=weight_mode, symmetrize=True)
    n_edges = int((W > 0).sum() // 2)
    if n_swaps_per is None:
        n_swaps_per = max(10 * n_edges, 100)

    Qs = np.empty(n_permutations, dtype=np.float64)
    base_rng = np.random.default_rng(seed)
    for i in tqdm(range(n_permutations), desc="config-model Q null"):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        W_null = _double_edge_swap_weighted(W, n_swaps_per, rng)
        # Reuse louvain() by passing W_null as a torch tensor.
        res = louvain(torch.from_numpy(W_null), resolution=resolution, weight_mode="positive", seed=i)
        Qs[i] = res.modularity
    return Qs


# ---------------------------------------------------------------------------
# Weight-permutation nulls for SC
# ---------------------------------------------------------------------------


def permute_within_layer(W_Q, W_K, W_V, W_O, rng: np.random.Generator):
    """Permute head indices within each layer.

    Inputs are torch tensors with shapes:
        W_Q, W_K, W_V: (L, H, d_model, d_head)
        W_O:           (L, H, d_head, d_model)
    Returns permuted copies of the same shapes.
    """
    L, H = W_Q.shape[0], W_Q.shape[1]
    W_Q = W_Q.clone()
    W_K = W_K.clone()
    W_V = W_V.clone()
    W_O = W_O.clone()
    for l in range(L):
        perm = rng.permutation(H)
        idx = torch.from_numpy(perm).long()
        W_Q[l] = W_Q[l][idx]
        W_K[l] = W_K[l][idx]
        W_V[l] = W_V[l][idx]
        W_O[l] = W_O[l][idx]
    return W_Q, W_K, W_V, W_O


def sc_null_via_head_permutation(
    model,
    n_permutations: int = 1000,
    seed: int = 0,
    statistic_fn=None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Null distribution over an SC-derived statistic via within-layer head permutation.

    Args:
        model: TransformerLens HookedTransformer.
        n_permutations: number of null draws.
        seed: RNG seed.
        statistic_fn: callable mapping a flat upper-triangular SC vector to a scalar.
            Default: mean of SC_total upper-triangular entries.
        device: device for SC computation.

    Returns:
        (n_permutations,) array of null statistics.
    """
    from .sc import _stack_head_weights, compute_ov_qk, _frob_norm_of_product

    if device is None:
        device = next(model.parameters()).device
    W_Q, W_K, W_V, W_O = _stack_head_weights(model, device=device)

    if statistic_fn is None:
        def statistic_fn(sc_upper):  # type: ignore[no-redef]
            return float(sc_upper.mean().item())

    base_rng = np.random.default_rng(seed)
    out = np.empty(n_permutations, dtype=np.float64)
    for i in tqdm(range(n_permutations), desc="head-permutation SC null"):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        WQp, WKp, WVp, WOp = permute_within_layer(W_Q, W_K, W_V, W_O, rng)
        # Compute SC_total from permuted weights.
        W_OV, W_QK = compute_ov_qk(WQp, WKp, WVp, WOp)
        L, H, d_model, _ = WQp.shape
        N = L * H
        W_OV_flat = W_OV.reshape(N, d_model, d_model)
        W_QK_flat = W_QK.reshape(N, d_model, d_model)
        sc_total = torch.zeros((N, N), dtype=torch.float32, device=W_OV.device)
        layer_of = torch.arange(N, device=W_OV.device) // H
        for s_idx in range(N):
            mask = layer_of > layer_of[s_idx]
            if not mask.any():
                continue
            tgt = torch.arange(N, device=W_OV.device)[mask]
            A_ov = W_OV_flat[s_idx]
            B_ov = W_OV_flat[tgt]
            B_qk = W_QK_flat[tgt]
            q = _frob_norm_of_product(A_ov.unsqueeze(0), B_qk)
            k = _frob_norm_of_product(A_ov.unsqueeze(0), B_qk.transpose(-2, -1))
            v = _frob_norm_of_product(A_ov.unsqueeze(0), B_ov)
            sc_total[s_idx, tgt] = q + k + v
        out[i] = statistic_fn(flatten_upper_triangular(sc_total.cpu()))
    return out


# ---------------------------------------------------------------------------
# p-value helpers
# ---------------------------------------------------------------------------


def empirical_p_value(
    observed: float,
    null: np.ndarray,
    alternative: str = "two-sided",
) -> float:
    """Empirical p-value comparing an observed statistic to a null sample.

    Uses the standard add-one correction so p > 0 even for extreme values.
    """
    null = np.asarray(null)
    n = null.size
    if alternative == "greater":
        return (np.sum(null >= observed) + 1) / (n + 1)
    if alternative == "less":
        return (np.sum(null <= observed) + 1) / (n + 1)
    if alternative == "two-sided":
        center = np.median(null)
        extreme = np.sum(np.abs(null - center) >= abs(observed - center))
        return (extreme + 1) / (n + 1)
    raise ValueError(f"Unknown alternative: {alternative!r}")
