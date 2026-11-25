"""
Functional connectivity (FC) for transformer LLMs.

Two FC variants, each producing an (N, N) symmetric matrix over the
N = L * H attention heads:

- ``fc_l2``      -- z-scored Pearson correlation of per-head L2-norm
                    timeseries (the scalar "BOLD analog").
- ``fc_partial`` -- Pearson on z-scored L2 norms after regressing out the
                    residual-stream-norm timeseries. Addresses the
                    common-input confound.

Both return an (N, N) tensor with entries in [-1, 1] and 0 on the diagonal.

Convention: node index = layer * n_heads + head_within_layer.
"""

from __future__ import annotations

import torch

from .activity import Activity


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _flatten_lh(x: torch.Tensor) -> torch.Tensor:
    """(T, L, H[, ...]) -> (T, N[, ...]) where N = L * H."""
    if x.dim() < 3:
        raise ValueError(f"expected at least 3 dims (T, L, H), got {x.shape}")
    T, L, H, *rest = x.shape
    return x.reshape(T, L * H, *rest)


def _zscore(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """Z-score along `dim`. Constant-along-dim entries return 0."""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True, unbiased=False)
    return (x - mean) / std.clamp_min(eps)


def _zero_diagonal(m: torch.Tensor) -> torch.Tensor:
    m = m.clone()
    idx = torch.arange(m.shape[0])
    m[idx, idx] = 0.0
    return m


# ---------------------------------------------------------------------------
# FC 1: scalar L2-norm Pearson
# ---------------------------------------------------------------------------


def fc_l2(activity: Activity | torch.Tensor) -> torch.Tensor:
    """Pearson FC over per-head L2-norm timeseries.

    Args:
        activity: Activity object or a (T, L, H) tensor of L2 norms.

    Returns:
        (N, N) Pearson correlation matrix, symmetric, zero diagonal.
    """
    l2 = activity.l2 if isinstance(activity, Activity) else activity
    if l2.dim() == 3:
        l2 = _flatten_lh(l2)  # (T, N)
    T = l2.shape[0]
    z = _zscore(l2.float(), dim=0)
    fc = (z.t() @ z) / max(T - 1, 1)
    return _zero_diagonal(fc.clamp(-1.0, 1.0))


# ---------------------------------------------------------------------------
# FC 2: partial correlation controlling for residual-stream norm
# ---------------------------------------------------------------------------


def _regress_out(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return y after OLS-regressing out x.

    y: (T, K), x: (T,). Returns (T, K) residuals.
    """
    x = x.float()
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum().clamp_min(1e-12)
    beta = (x_centered.unsqueeze(0) @ (y - y.mean(0, keepdim=True))).squeeze(0) / denom  # (K,)
    return y - (x.unsqueeze(1) * beta.unsqueeze(0)) - (y.mean(0, keepdim=True) - x.mean() * beta)


def fc_partial(activity: Activity) -> torch.Tensor:
    """Partial-correlation FC controlling for the residual-stream norm.

    Equivalent to fc_l2 after regressing the residual-stream L2-norm
    timeseries out of every head's L2-norm signal. Addresses the
    common-input confound: heads at later layers share input from the
    residual stream, which can inflate apparent FC.

    Args:
        activity: Activity with both .l2 and .residual_norm populated.

    Returns:
        (N, N) symmetric Pearson matrix on residualized signals.
    """
    l2 = _flatten_lh(activity.l2.float())  # (T, N)
    rsn = activity.residual_norm.float()   # (T,)

    l2_resid = _regress_out(l2, rsn)
    T = l2.shape[0]
    z = _zscore(l2_resid, dim=0)
    fc = (z.t() @ z) / max(T - 1, 1)
    return _zero_diagonal(fc.clamp(-1.0, 1.0))


# ---------------------------------------------------------------------------
# Convenience: run both variants
# ---------------------------------------------------------------------------


def all_fc_measures(activity: Activity) -> dict[str, torch.Tensor]:
    """Compute both FC variants on a single Activity capture.

    Returns a dict with keys {"l2", "partial"}.
    """
    return {
        "l2": fc_l2(activity),
        "partial": fc_partial(activity),
    }
