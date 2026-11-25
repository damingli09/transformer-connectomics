"""
Structural connectivity (SC) for transformer LLMs.

Defines edge weights between attention heads based on the weight-matrix
composition framework from Elhage et al. (2021), "A Mathematical Framework
for Transformer Circuits."

For two heads A (layer l_A, head h_A) and B (layer l_B, head h_B) with
l_B > l_A, three structural pathways exist:

    SC_Q(A -> B) = || W_OV^A @ W_QK^B   ||_F     # A writes -> B reads Q
    SC_K(A -> B) = || W_OV^A @ (W_QK^B)^T ||_F   # A writes -> B reads K
    SC_V(A -> B) = || W_OV^A @ W_OV^B   ||_F     # A writes -> B passes through

where W_OV = W_V @ W_O   (d_model x d_model)
      W_QK = W_Q @ W_K^T (d_model x d_model)

The total SC edge is the sum of the three pathway norms.

Nodes are indexed by a flat integer id = layer * n_heads + head_within_layer.
The resulting SC matrices are strictly upper-triangular in layer order (a DAG):
SC[i, j] = 0 whenever layer(i) >= layer(j).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SCMatrices:
    """Container for the four SC matrices (Q, K, V, total) plus metadata."""

    sc_q: torch.Tensor    # (N, N) upper-triangular, Q-composition Frobenius norms
    sc_k: torch.Tensor    # (N, N) upper-triangular, K-composition Frobenius norms
    sc_v: torch.Tensor    # (N, N) upper-triangular, V-composition Frobenius norms
    sc_total: torch.Tensor  # (N, N) sum of the three above
    n_layers: int
    n_heads: int

    @property
    def n_nodes(self) -> int:
        return self.n_layers * self.n_heads

    def head_index(self, layer: int, head: int) -> int:
        """Flat index for (layer, head)."""
        return layer * self.n_heads + head

    def normalized(self, key: str = "total") -> torch.Tensor:
        """Return the named matrix scaled so its max entry is 1.0.

        For cross-model comparison or visualization. Raises if the matrix is
        all zero.
        """
        m = getattr(self, f"sc_{key}")
        mx = m.max()
        if mx <= 0:
            raise ValueError(f"SC matrix '{key}' has max <= 0; cannot normalize.")
        return m / mx


def _stack_head_weights(
    model,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack per-head W_Q, W_K, W_V, W_O across all layers.

    Works with TransformerLens HookedTransformer. Returns tensors of shape:
        W_Q, W_K, W_V: (n_layers, n_heads, d_model, d_head)
        W_O:           (n_layers, n_heads, d_head, d_model)

    GQA models (e.g. Llama) are handled by TransformerLens by repeating
    the KV heads up to the query-head count, so the same shapes apply.
    """
    if device is None:
        device = next(model.parameters()).device
    W_Q = model.W_Q.to(device)  # (n_layers, n_heads, d_model, d_head)
    W_K = model.W_K.to(device)
    W_V = model.W_V.to(device)
    W_O = model.W_O.to(device)  # (n_layers, n_heads, d_head, d_model)
    return W_Q, W_K, W_V, W_O


def compute_ov_qk(
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    W_O: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute W_OV and W_QK for every head.

    W_OV[l, h] = W_V[l, h] @ W_O[l, h]  -> (d_model, d_model)
    W_QK[l, h] = W_Q[l, h] @ W_K[l, h]^T -> (d_model, d_model)

    Returns stacked tensors of shape (n_layers, n_heads, d_model, d_model).
    """
    W_OV = torch.einsum("lhmd,lhdn->lhmn", W_V, W_O)
    W_QK = torch.einsum("lhmd,lhnd->lhmn", W_Q, W_K)
    return W_OV, W_QK


def _frob_norm_of_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """||A @ B||_F as a scalar.

    A: (..., m, k), B: (..., k, n). Returns shape (...).
    Uses the actual product (not a trace trick) because the matrices are
    only d_model x d_model = 768 x 768 for GPT-2; cheap enough.
    """
    AB = A @ B
    return torch.linalg.norm(AB, ord="fro", dim=(-2, -1))


def compute_sc(
    model,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 16,
) -> SCMatrices:
    """Compute the structural connectivity matrices for a transformer.

    Args:
        model: a TransformerLens HookedTransformer (must expose W_Q, W_K,
            W_V, W_O with shapes (n_layers, n_heads, d_model, d_head) and
            W_O = (n_layers, n_heads, d_head, d_model)).
        device: torch device; defaults to model's device.
        dtype: precision for the Frobenius-norm computation. fp32 is
            recommended; fp16/bf16 can underflow for small composition norms.
        chunk_size: number of target heads processed per source. Keeps peak
            memory bounded — each chunk allocates ~3*chunk_size*d_model^2
            float32 elements in intermediate tensors. Reduce on memory-tight
            devices (MPS / small GPUs).

    Returns:
        SCMatrices with sc_q, sc_k, sc_v, sc_total tensors of shape (N, N)
        where N = n_layers * n_heads.

    Notes:
        SC[i, j] = 0 if layer(i) >= layer(j) (DAG constraint).
        Self-connections (i == j) are 0.
    """
    import gc

    # `torch.no_grad()` is essential: model parameters have requires_grad=True
    # by default, and the inner matmul loop would otherwise build a huge
    # autograd graph that accumulates intermediate tensors until the process
    # is OOM-killed.
    with torch.no_grad():
        W_Q_src, W_K_src, W_V_src, W_O_src = _stack_head_weights(model, device=device)
        W_Q = W_Q_src.detach().to(dtype).contiguous()
        W_K = W_K_src.detach().to(dtype).contiguous()
        W_V = W_V_src.detach().to(dtype).contiguous()
        W_O = W_O_src.detach().to(dtype).contiguous()
        del W_Q_src, W_K_src, W_V_src, W_O_src

        n_layers, n_heads, d_model, _ = W_Q.shape
        N = n_layers * n_heads

        W_OV, W_QK = compute_ov_qk(W_Q, W_K, W_V, W_O)
        del W_Q, W_K, W_V, W_O
        gc.collect()

        W_OV_flat = W_OV.reshape(N, d_model, d_model)
        W_QK_flat = W_QK.reshape(N, d_model, d_model)

        sc_q = torch.zeros((N, N), dtype=dtype)
        sc_k = torch.zeros((N, N), dtype=dtype)
        sc_v = torch.zeros((N, N), dtype=dtype)

        # Iterate over source heads. For each, batch over downstream
        # targets in chunks of `chunk_size` to keep peak memory bounded.
        for i in range(N):
            l_i = i // n_heads
            first_j = (l_i + 1) * n_heads
            if first_j >= N:
                continue
            A_ov = W_OV_flat[i]  # (d_model, d_model)
            for start in range(first_j, N, chunk_size):
                stop = min(start + chunk_size, N)
                B_ov = W_OV_flat[start:stop]   # (chunk, d_model, d_model)
                B_qk = W_QK_flat[start:stop]
                sc_q[i, start:stop] = _frob_norm_of_product(A_ov.unsqueeze(0), B_qk)
                sc_k[i, start:stop] = _frob_norm_of_product(
                    A_ov.unsqueeze(0), B_qk.transpose(-2, -1)
                )
                sc_v[i, start:stop] = _frob_norm_of_product(A_ov.unsqueeze(0), B_ov)

        sc_total = sc_q + sc_k + sc_v

    return SCMatrices(
        sc_q=sc_q.cpu(),
        sc_k=sc_k.cpu(),
        sc_v=sc_v.cpu(),
        sc_total=sc_total.cpu(),
        n_layers=n_layers,
        n_heads=n_heads,
    )


def compute_sc_lowrank(
    model,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> SCMatrices:
    """Memory-efficient SC via the rank-d_h trace trick.

    Mathematically equivalent to ``compute_sc`` but never materializes the
    full ``(L, H, d_model, d_model)`` composition tensors. Required for
    models with large d_model (Llama-3-8B has d_model=4096, the naive
    formulation would allocate >60 GB).

    Derivation
    ----------
    Both W_OV = W_V W_O and W_QK = W_Q W_K^T are rank ≤ d_h. Using the
    cyclic property of trace:

        ||W_V^A W_O^A W_Q^B (W_K^B)^T||_F^2
            = tr(M_Q^T (W_V^A^T W_V^A) M_Q (W_K^B^T W_K^B))
            where M_Q := W_O^A W_Q^B  ∈ R^{d_h × d_h}

    Analogous identities hold for the K- and V-composition pathways:

        ||W_OV^A (W_QK^B)^T||_F^2 = tr(M_K^T N_V[A] M_K N_Q[B])
                                    where M_K := W_O^A W_K^B
        ||W_OV^A W_OV^B||_F^2     = tr(M_V^T N_V[A] M_V N_O[B])
                                    where M_V := W_O^A W_V^B

    with N_V[A] = W_V^A^T W_V^A, N_Q[B] = W_Q^B^T W_Q^B,
    N_K[B] = W_K^B^T W_K^B, N_O[B] = W_O^B W_O^B^T — all d_h × d_h.

    Memory: O(N d_h²) for the precomputed Gram tensors instead of
    O(N d_model²) for compute_sc's intermediate. For Llama-3-8B this is
    256 MB vs 67 GB.

    GQA: TransformerLens broadcasts shared KV heads up to query-head count,
    so this function works on GQA models without special-casing. Heads in
    the same GQA group automatically share N_K and N_V (since their W_K,
    W_V are identical post-broadcast).
    """
    import gc

    with torch.no_grad():
        W_Q_src, W_K_src, W_V_src, W_O_src = _stack_head_weights(model, device=device)
        W_Q = W_Q_src.detach().to(dtype).contiguous()
        W_K = W_K_src.detach().to(dtype).contiguous()
        W_V = W_V_src.detach().to(dtype).contiguous()
        W_O = W_O_src.detach().to(dtype).contiguous()
        del W_Q_src, W_K_src, W_V_src, W_O_src

        L, H, d_model, d_head = W_Q.shape
        N = L * H

        # Per-head Gram tensors, all (L, H, d_h, d_h).
        # N_V[A] = W_V^A^T @ W_V^A  — used on the source side (A's writing geometry)
        # N_Q[B] = W_Q^B^T @ W_Q^B  — used on the target reading side (K-composition's right factor)
        # N_K[B] = W_K^B^T @ W_K^B  — used on the target reading side (Q-composition's right factor)
        # N_O[B] = W_O^B @ W_O^B^T  — used on the target side (V-composition's right factor)
        N_V = torch.einsum("lhmd,lhme->lhde", W_V, W_V)
        N_Q = torch.einsum("lhmd,lhme->lhde", W_Q, W_Q)
        N_K = torch.einsum("lhmd,lhme->lhde", W_K, W_K)
        N_O = torch.einsum("lhdm,lhem->lhde", W_O, W_O)
        gc.collect()

        sc_q_sq = torch.zeros((N, N), dtype=dtype)
        sc_k_sq = torch.zeros((N, N), dtype=dtype)
        sc_v_sq = torch.zeros((N, N), dtype=dtype)

        # Iterate over source heads; for each, batch over target heads
        # within each future layer.
        for src in range(N):
            l_A, h_A = src // H, src % H
            W_O_A = W_O[l_A, h_A]      # (d_head, d_model)
            N_V_A = N_V[l_A, h_A]      # (d_head, d_head)

            for l_B in range(l_A + 1, L):
                # Batched over the H query heads in target layer l_B.
                W_Q_B = W_Q[l_B]       # (H, d_model, d_head)
                W_K_B = W_K[l_B]
                W_V_B = W_V[l_B]
                N_Q_B = N_Q[l_B]       # (H, d_head, d_head)
                N_K_B = N_K[l_B]
                N_O_B = N_O[l_B]

                # M = W_O^A @ W_*^B for each target head:
                # W_O_A: (d_h, d_model), W_*_B: (H, d_model, d_h) → (H, d_h, d_h)
                M_Q = torch.einsum("dm,hme->hde", W_O_A, W_Q_B)
                M_K = torch.einsum("dm,hme->hde", W_O_A, W_K_B)
                M_V = torch.einsum("dm,hme->hde", W_O_A, W_V_B)

                # tr(M^T N_V_A M N_*) for each target head.
                # Compute via successive einsums; final reduction is a trace.
                # T_Q = M_Q^T @ N_V_A @ M_Q @ N_K_B; trace over the d_h × d_h result.
                #
                # Step 1: A := N_V_A @ M_Q   (broadcast N_V_A over H)
                # Step 2: B := M_Q^T @ A
                # Step 3: C := B @ N_K_B
                # Step 4: tr(C) per head
                A_q = torch.einsum("de,hef->hdf", N_V_A, M_Q)
                B_q = torch.einsum("hed,hef->hdf", M_Q, A_q)  # M_Q^T @ A_q
                C_q = torch.einsum("hde,hef->hdf", B_q, N_K_B)
                tr_q = torch.diagonal(C_q, dim1=-2, dim2=-1).sum(dim=-1)  # (H,)

                A_k = torch.einsum("de,hef->hdf", N_V_A, M_K)
                B_k = torch.einsum("hed,hef->hdf", M_K, A_k)
                C_k = torch.einsum("hde,hef->hdf", B_k, N_Q_B)
                tr_k = torch.diagonal(C_k, dim1=-2, dim2=-1).sum(dim=-1)

                A_v = torch.einsum("de,hef->hdf", N_V_A, M_V)
                B_v = torch.einsum("hed,hef->hdf", M_V, A_v)
                C_v = torch.einsum("hde,hef->hdf", B_v, N_O_B)
                tr_v = torch.diagonal(C_v, dim1=-2, dim2=-1).sum(dim=-1)

                tgt_start = l_B * H
                tgt_end = (l_B + 1) * H
                # Tiny numerical drift can produce negatives — clamp before sqrt.
                sc_q_sq[src, tgt_start:tgt_end] = tr_q.clamp_min(0.0).cpu()
                sc_k_sq[src, tgt_start:tgt_end] = tr_k.clamp_min(0.0).cpu()
                sc_v_sq[src, tgt_start:tgt_end] = tr_v.clamp_min(0.0).cpu()

        sc_q = sc_q_sq.sqrt()
        sc_k = sc_k_sq.sqrt()
        sc_v = sc_v_sq.sqrt()
        sc_total = sc_q + sc_k + sc_v

    return SCMatrices(
        sc_q=sc_q.cpu(),
        sc_k=sc_k.cpu(),
        sc_v=sc_v.cpu(),
        sc_total=sc_total.cpu(),
        n_layers=L,
        n_heads=H,
    )


def head_label(idx: int, n_heads: int) -> str:
    """Human-readable label 'L{layer}H{head}' for a flat head index."""
    return f"L{idx // n_heads}H{idx % n_heads}"


def flatten_upper_triangular(matrix: torch.Tensor) -> torch.Tensor:
    """Return the strictly upper-triangular entries (i < j) as a 1D vector.

    Used for SC-FC coupling: vectorize the directed-forward entries of SC
    and the matching off-diagonal entries of FC to compute Pearson kappa.
    """
    N = matrix.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    return matrix[iu[0], iu[1]]
