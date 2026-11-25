"""
Activity extraction: per-head output timeseries over a token corpus.

Uses TransformerLens hooks to capture each attention head's contribution to
the residual stream at every token position. Streams batch-by-batch to keep
memory bounded.

The "activity" of a head at a token is the d_model-dimensional vector it
writes to the residual stream (after the W_O projection). From this we derive:

- ``l2`` (T, L, H): scalar L2 norm per (token, layer, head). The "BOLD analog."
- ``residual_norm`` (T,): L2 norm of the residual stream pre-block-0, used
  as a covariate for partial-correlation FC.

For GPT-2 Small at 100k tokens the l2 buffer is
    100k * 12 * 12 * 4 bytes  ~  56 MB.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm.auto import tqdm


@dataclass
class Activity:
    """Captured activity over a corpus."""

    l2: torch.Tensor                  # (T, L, H) — L2 norms of per-head residual-stream contributions
    residual_norm: torch.Tensor       # (T,) — L2 norm of residual stream entering layer 0
    n_layers: int
    n_heads: int
    d_head: int
    d_model: int

    @property
    def n_tokens(self) -> int:
        return self.l2.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.n_layers * self.n_heads


def _stack_W_O(model, device) -> torch.Tensor:
    """Return W_O as a (L, H, d_head, d_model) tensor on the requested device."""
    return model.W_O.to(device)


def extract_activity(
    model,
    tokens: torch.Tensor,
    batch_size: int = 8,
    seq_len: int | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> Activity:
    """Run the model over a tokenized corpus and capture per-head activity.

    Args:
        model: a TransformerLens HookedTransformer.
        tokens: 1D tensor of token ids, or 2D (n_sequences, seq_len).
            If 1D, it is reshaped into batches of `seq_len` tokens (default:
            ``model.cfg.n_ctx``). Tokens past the last full sequence are
            dropped.
        batch_size: number of sequences per forward pass.
        seq_len: context length for reshaping. Defaults to ``model.cfg.n_ctx``.
        device: model device; defaults to the model's current device.
        show_progress: whether to display a tqdm bar.

    Returns:
        Activity dataclass.
    """
    if device is None:
        device = next(model.parameters()).device

    if seq_len is None:
        seq_len = model.cfg.n_ctx

    if tokens.dim() == 1:
        n_full = (tokens.numel() // seq_len) * seq_len
        if n_full == 0:
            raise ValueError(
                f"Corpus has {tokens.numel()} tokens, fewer than one sequence "
                f"of length {seq_len}."
            )
        tokens = tokens[:n_full].reshape(-1, seq_len)
    elif tokens.dim() != 2:
        raise ValueError(f"tokens must be 1D or 2D, got {tokens.dim()}D.")

    n_seqs, seq_len_actual = tokens.shape
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_model = model.cfg.d_model
    T = n_seqs * seq_len_actual

    # Pre-fetch W_O for fast per-token L2-of-W_O@z computation.
    W_O = _stack_W_O(model, device)  # (L, H, d_head, d_model)

    # Accumulators on CPU to avoid OOM. Cast to float32 explicitly because
    # mixed-precision runs would otherwise produce fp16 norms.
    l2_buf = torch.empty((T, n_layers, n_heads), dtype=torch.float32)
    res_buf = torch.empty((T,), dtype=torch.float32)

    # Hook names: we capture per-layer hook_z (per-head pre-W_O output) and
    # hook_resid_pre at layer 0 (the residual stream entering the model).
    z_names = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]
    resid_name = "blocks.0.hook_resid_pre"

    captured: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(tensor, hook):  # noqa: ARG001
            captured[name] = tensor.detach()
        return hook

    fwd_hooks = [(name, make_hook(name)) for name in z_names + [resid_name]]

    it = range(0, n_seqs, batch_size)
    if show_progress:
        it = tqdm(it, desc="extract_activity", total=(n_seqs + batch_size - 1) // batch_size)

    model.eval()
    write_idx = 0
    with torch.no_grad():
        for start in it:
            end = min(start + batch_size, n_seqs)
            batch = tokens[start:end].to(device)
            captured.clear()
            model.run_with_hooks(batch, fwd_hooks=fwd_hooks, return_type=None)

            B, S = batch.shape
            flat_T = B * S
            slice_ = slice(write_idx, write_idx + flat_T)

            # Residual stream norms: (B, S, d_model) -> (B*S,)
            resid = captured[resid_name].reshape(B * S, d_model)
            res_buf[slice_] = resid.norm(dim=-1).float().cpu()

            # Per-layer per-head activity.
            for l, name in enumerate(z_names):
                # z: (B, S, H, d_head)
                z = captured[name].reshape(B * S, n_heads, d_head)
                # Per-head residual-stream contribution: z @ W_O[l]
                # W_O[l]: (H, d_head, d_model). Result: (B*S, H, d_model)
                contrib = torch.einsum("thd,hdm->thm", z, W_O[l])
                l2_buf[slice_, l, :] = contrib.norm(dim=-1).float().cpu()

            write_idx += flat_T

    assert write_idx == T

    return Activity(
        l2=l2_buf,
        residual_norm=res_buf,
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        d_model=d_model,
    )
