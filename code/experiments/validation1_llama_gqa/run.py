"""
Validation I: grouped-query attention (GQA) topology in Llama-3.1-8B.

Builds the connectome for a GQA model and runs three GQA-aware tests:

  H1 — structural similarity: incoming-SC vectors are more similar within a
       GQA group than between groups, per composition pathway (Q/K/V).
  H2 — functional similarity: partial-FC magnitude is larger within a GQA
       group than between groups (same layer).
  H3 — community structure: unsupervised Louvain partitioning recovers GQA
       group boundaries above a within-layer shuffle null.

Each test uses a permutation null (default 10,000 resamples) and the set is
evaluated at Bonferroni alpha = 0.05 / 5 (H1x3 + H2 + H3).

Produces, under --output:
    config.json            — exact args
    sc.pt                  — Q/K/V/total SC (low-rank implementation)
    fc.pt                  — partial FC
    communities.pt         — Louvain gamma-sweep on partial FC
    hypothesis_tests.json  — H1/H2/H3 statistics + p-values, plus kappa

Usage:
    python run.py \
        --model meta-llama/Llama-3.1-8B \
        --corpus-size 50000 \
        --device cuda --batch-size 2 --seq-len 2048 \
        --output out/llama31_8b

This model is large; the pipeline is intended for a high-memory GPU
(e.g. a single 80GB accelerator).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectomics import community as comm  # noqa: E402
from connectomics import metrics as met  # noqa: E402
from connectomics.activity import extract_activity  # noqa: E402
from connectomics.corpus import load_corpus  # noqa: E402
from connectomics.device import best_device, empty_cache  # noqa: E402
from connectomics.fc import fc_partial  # noqa: E402
from connectomics.sc import compute_sc_lowrank  # noqa: E402

# Resolve the shipped corpus relative to this file so the default works from
# any working directory (parents[2] = code/, its parent = artifact root).
DEFAULT_CORPUS = str(Path(__file__).resolve().parents[2].parent / "data" / "corpus_100k.pt")


def load_gqa_model(name: str, device: str, dtype: torch.dtype = torch.float16):
    """Load a HookedTransformer GQA model (Llama-3 family) in fp16."""
    from connectomics.models import _workaround_hf_empty_bearer
    _workaround_hf_empty_bearer()
    from transformer_lens import HookedTransformer

    print(f"[val1] loading {name} (dtype={dtype})")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(name, device=device, dtype=dtype)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model


# A large per-layer offset keeps groups from different layers from colliding
# when group ids are combined into a single integer.
_LAYER_OFFSET = 1000


def _same_group_mask(n_layers: int, n_heads: int, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (same_group_mask, same_layer_mask) over the (N, N) head pairs."""
    N = n_layers * n_heads
    layer_of = np.arange(N) // n_heads
    head_of = np.arange(N) % n_heads
    group_of = layer_of * _LAYER_OFFSET + head_of // group_size

    same_layer = (layer_of[:, None] == layer_of[None, :])
    same_group = (group_of[:, None] == group_of[None, :])
    return same_group, same_layer


# ---------------------------------------------------------------------------
# H1 — GQA structural cosine
# ---------------------------------------------------------------------------

def score_h1(
    sc_matrices: dict[str, torch.Tensor],
    n_layers: int,
    n_heads: int,
    group_size: int,
    n_perm: int,
    seed: int,
) -> dict:
    """Within-group vs between-group cosine of incoming-SC vectors."""
    rng = np.random.default_rng(seed)
    results = {}

    for sc_name in ("sc_q", "sc_k", "sc_v"):
        sc = sc_matrices[sc_name].numpy()  # (N, N), strictly upper-triangular
        N = sc.shape[0]

        # Normalize incoming SC columns to unit norm where non-zero.
        col_norms = np.linalg.norm(sc, axis=0)
        valid_targets = col_norms > 1e-10
        sc_unit = np.zeros_like(sc)
        sc_unit[:, valid_targets] = sc[:, valid_targets] / col_norms[valid_targets]

        cos_mat = sc_unit.T @ sc_unit  # (N, N)

        # Within- and between-group pairs are both restricted to the same layer.
        same_group, same_layer = _same_group_mask(n_layers, n_heads, group_size)
        iu = np.triu_indices(N, k=1)
        sl_mask = same_layer[iu]
        sg_mask = same_group[iu]
        cos_pairs = cos_mat[iu]

        within = cos_pairs[sl_mask & sg_mask & (cos_pairs != 0)]
        between = cos_pairs[sl_mask & (~sg_mask) & (cos_pairs != 0)]
        delta_obs = float(within.mean() - between.mean()) if len(within) and len(between) else float("nan")

        # Permutation null — relabel within-layer head->group.
        null_deltas = np.empty(n_perm)
        for i in range(n_perm):
            new_group = np.empty(N, dtype=np.int64)
            for l in range(n_layers):
                perm = rng.permutation(n_heads)
                new_group[l * n_heads:(l + 1) * n_heads] = l * _LAYER_OFFSET + (perm // group_size)
            sg_null = (new_group[:, None] == new_group[None, :])
            sg_pair = sg_null[iu]
            w = cos_pairs[sl_mask & sg_pair & (cos_pairs != 0)]
            b = cos_pairs[sl_mask & (~sg_pair) & (cos_pairs != 0)]
            null_deltas[i] = (w.mean() - b.mean()) if len(w) and len(b) else 0.0

        p_one_sided = float((np.sum(null_deltas >= delta_obs) + 1) / (n_perm + 1))
        results[sc_name] = {
            "within_group_mean_cosine": float(within.mean()) if len(within) else None,
            "between_group_mean_cosine": float(between.mean()) if len(between) else None,
            "delta_obs": delta_obs,
            "null_mean_delta": float(null_deltas.mean()),
            "null_std_delta": float(null_deltas.std(ddof=1)),
            "p_one_sided": p_one_sided,
        }
    return results


# ---------------------------------------------------------------------------
# H2 — Within-group vs between-group FC magnitude (same layer)
# ---------------------------------------------------------------------------

def score_h2(
    fc: torch.Tensor,
    n_layers: int,
    n_heads: int,
    group_size: int,
    n_perm: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    fc_np = fc.numpy()
    N = fc_np.shape[0]
    same_group, same_layer = _same_group_mask(n_layers, n_heads, group_size)
    iu = np.triu_indices(N, k=1)
    sl = same_layer[iu]
    sg = same_group[iu]
    fc_pairs = np.abs(fc_np[iu])

    within = fc_pairs[sl & sg]
    between = fc_pairs[sl & ~sg]
    delta_obs = float(within.mean() - between.mean())

    null_deltas = np.empty(n_perm)
    for i in range(n_perm):
        new_group = np.empty(N, dtype=np.int64)
        for l in range(n_layers):
            perm = rng.permutation(n_heads)
            new_group[l * n_heads:(l + 1) * n_heads] = l * _LAYER_OFFSET + (perm // group_size)
        sg_null = (new_group[:, None] == new_group[None, :])
        sg_pair = sg_null[iu]
        w = fc_pairs[sl & sg_pair]
        b = fc_pairs[sl & ~sg_pair]
        null_deltas[i] = w.mean() - b.mean()

    p = float((np.sum(null_deltas >= delta_obs) + 1) / (n_perm + 1))
    return {
        "within_group_mean_abs_fc": float(within.mean()),
        "between_group_mean_abs_fc": float(between.mean()),
        "delta_obs": delta_obs,
        "null_mean_delta": float(null_deltas.mean()),
        "null_std_delta": float(null_deltas.std(ddof=1)),
        "p_one_sided": p,
    }


# ---------------------------------------------------------------------------
# H3 — Louvain communities respect GQA groups
# ---------------------------------------------------------------------------

def score_h3(
    partition: np.ndarray,
    n_layers: int,
    n_heads: int,
    group_size: int,
    n_perm: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    def group_purity(part: np.ndarray) -> float:
        purities = []
        for l in range(n_layers):
            for g in range(n_heads // group_size):
                idx = [l * n_heads + g * group_size + k for k in range(group_size)]
                cids = part[idx]
                counts = np.bincount(cids)
                purities.append(counts.max() / group_size)
        return float(np.mean(purities))

    obs = group_purity(partition)

    null = np.empty(n_perm)
    for i in range(n_perm):
        # Permute community assignments within each layer (preserves per-layer
        # community sizes; breaks GQA grouping).
        permuted = partition.copy()
        for l in range(n_layers):
            sl = slice(l * n_heads, (l + 1) * n_heads)
            permuted[sl] = rng.permutation(partition[sl])
        null[i] = group_purity(permuted)

    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {
        "observed_group_purity": obs,
        "null_mean_purity": float(null.mean()),
        "null_std_purity": float(null.std(ddof=1)),
        "p_one_sided": p,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--corpus-size", type=int, default=50_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--device", default=best_device())
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--n-perm", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    corpus = load_corpus(args.corpus)
    tokens = corpus.tokens[: args.corpus_size]
    print(f"[val1] corpus: {len(tokens)} tokens, sha={corpus.sha256[:8]}")

    model = load_gqa_model(args.model, device=args.device)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads  # query heads after TransformerLens broadcast
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_heads
    group_size = n_heads // n_kv_heads
    N = n_layers * n_heads
    print(f"[val1] {args.model}: L={n_layers}, H_query={n_heads}, "
          f"H_kv={n_kv_heads}, GQA G={group_size}, N={N}")

    if group_size == 1:
        print("[val1] NOTE: this model has G=1 (no GQA); H1/H2/H3 are trivial.")

    print("\n[val1] SC (low-rank)")
    t0 = time.time()
    sc = compute_sc_lowrank(model, device=args.device, dtype=torch.float32)
    torch.save({
        "sc_q": sc.sc_q, "sc_k": sc.sc_k, "sc_v": sc.sc_v,
        "sc_total": sc.sc_total,
        "n_layers": n_layers, "n_heads": n_heads,
        "n_kv_heads": n_kv_heads, "group_size": group_size,
    }, out / "sc.pt")
    print(f"  SC: {time.time() - t0:.1f}s")

    print("[val1] activity extraction")
    t0 = time.time()
    n_seqs = max(1, tokens.numel() // args.seq_len)
    n_full = n_seqs * args.seq_len
    seqs = tokens[:n_full].reshape(n_seqs, args.seq_len)
    print(f"  using {n_seqs} sequences x {args.seq_len} tokens = {n_full} total tokens")
    activity = extract_activity(model, seqs, batch_size=args.batch_size, device=args.device)
    print(f"  activity: {time.time() - t0:.1f}s")

    print("[val1] partial FC")
    t0 = time.time()
    fc = fc_partial(activity)
    torch.save({"partial": fc, "n_layers": n_layers, "n_heads": n_heads}, out / "fc.pt")
    print(f"  FC: {time.time() - t0:.1f}s")

    del activity, model
    empty_cache(args.device)

    print("[val1] Louvain gamma-sweep on partial FC")
    t0 = time.time()
    sweep = comm.sweep_resolution(fc, resolutions=np.linspace(0.5, 2.0, 16), seed=args.seed)
    best_part = sweep.partitions[sweep.best_idx]
    torch.save({
        "resolutions": sweep.resolutions,
        "modularities": sweep.modularities,
        "n_communities": sweep.n_communities,
        "partitions": sweep.partitions,
        "best_idx": sweep.best_idx,
    }, out / "communities.pt")
    print(f"  gamma-sweep: {time.time() - t0:.1f}s "
          f"(best gamma={sweep.resolutions[sweep.best_idx]:.2f}, "
          f"Q={sweep.modularities[sweep.best_idx]:.3f}, "
          f"K={sweep.n_communities[sweep.best_idx]})")

    print(f"\n[val1] hypothesis tests ({args.n_perm} permutations each)")
    sc_dict = {"sc_q": sc.sc_q, "sc_k": sc.sc_k, "sc_v": sc.sc_v}

    print("  H1 — GQA structural cosine (per SC pathway)")
    h1 = score_h1(sc_dict, n_layers, n_heads, group_size, args.n_perm, args.seed)
    for k, v in h1.items():
        print(f"    {k}: dcos = {v['delta_obs']:+.4f}, p = {v['p_one_sided']:.4f}")

    print("  H2 — within-group vs between-group FC magnitude")
    h2 = score_h2(fc, n_layers, n_heads, group_size, args.n_perm, args.seed)
    print(f"    d|FC| = {h2['delta_obs']:+.4f}, p = {h2['p_one_sided']:.4f}")

    print("  H3 — Louvain group purity")
    h3 = score_h3(best_part, n_layers, n_heads, group_size, args.n_perm, args.seed)
    print(f"    observed = {h3['observed_group_purity']:.3f}, "
          f"null mean = {h3['null_mean_purity']:.3f}, p = {h3['p_one_sided']:.4f}")

    kappa_global = met.sc_fc_coupling(sc.sc_total, fc)
    kappa_depth = met.layer_depth_kappa(sc.sc_total, fc, n_layers, n_heads)

    alpha_bonf = 0.05 / 5  # H1x3 + H2 + H3

    results = {
        "model": args.model,
        "n_layers": n_layers,
        "n_heads_query": n_heads,
        "n_heads_kv": n_kv_heads,
        "gqa_group_size": group_size,
        "n_perm": args.n_perm,
        "alpha_bonferroni": alpha_bonf,
        "H1_gqa_structural_cosine": h1,
        "H2_within_group_fc_magnitude": h2,
        "H3_louvain_group_purity": h3,
        "sc_fc_coupling_partial": {"global": kappa_global, **kappa_depth},
        "verdicts_bonferroni": {
            "H1_sc_q": h1["sc_q"]["p_one_sided"] < alpha_bonf,
            "H1_sc_k": h1["sc_k"]["p_one_sided"] < alpha_bonf,
            "H1_sc_v": h1["sc_v"]["p_one_sided"] < alpha_bonf,
            "H2": h2["p_one_sided"] < alpha_bonf,
            "H3": h3["p_one_sided"] < alpha_bonf,
        },
    }
    (out / "hypothesis_tests.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"\n[done] {out}/hypothesis_tests.json")
    print(f"  Bonferroni alpha = {alpha_bonf:.4f}")
    for k, v in results["verdicts_bonferroni"].items():
        print(f"    {k}: {'PASS' if v else 'fail'}")


if __name__ == "__main__":
    main()
