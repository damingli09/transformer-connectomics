"""
GPT-2 Small validation experiment (Validation II: IOI circuit recovery).

Builds the connectome for GPT-2 Small (or a random-weight control) and scores
how well the discovered communities recover the published IOI circuit.

Pipeline (each stage is skip-on-existing-output by default):

    1. Sample / load the fixed 100k-token Pile corpus.
    2. Load model (trained or random-weight) and compute SC.
    3. Extract per-head activity over the corpus.
    4. Compute both FC variants (L2 and partial).
    5. Run a Louvain gamma-sweep on each FC variant.
    6. Compute circuit recovery (IOI NMI + purity) and SC-FC coupling (kappa).
    7. Save matrices, partitions, statistics; emit figures.

Run as:
    python run.py --variant trained --corpus-size 100000 --output out/trained

For the random-weight control, repeat with --variant random.

The script is idempotent: each artifact is saved once and reloaded on
subsequent runs unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make the connectomics package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectomics import community as comm  # noqa: E402
from connectomics import fc as fc_mod  # noqa: E402
from connectomics import metrics as met  # noqa: E402
from connectomics import viz  # noqa: E402
from connectomics.activity import extract_activity  # noqa: E402
from connectomics.corpus import sample_pile_tokens, save_corpus, load_corpus  # noqa: E402
from connectomics.device import best_device, empty_cache  # noqa: E402
from connectomics.models import load_gpt2_small  # noqa: E402
from connectomics.sc import compute_sc  # noqa: E402

from circuit_labels import ioi_label_dict  # noqa: E402


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_corpus(args, out_dir: Path) -> torch.Tensor:
    path = out_dir / "corpus.pt"
    if path.exists() and not args.force:
        corp = load_corpus(str(path))
        print(f"[corpus] loaded {corp.n_tokens} tokens, hash={corp.sha256[:8]}")
        return corp

    print(f"[corpus] sampling {args.corpus_size} tokens from {args.dataset}...")
    from transformer_lens import HookedTransformer

    # We only need the tokenizer here; loading on CPU is fine.
    tmp = HookedTransformer.from_pretrained("gpt2", device="cpu")
    corp = sample_pile_tokens(
        tmp.tokenizer,
        n_tokens=args.corpus_size,
        seed=args.corpus_seed,
        dataset_name=args.dataset,
    )
    save_corpus(corp, str(path))
    del tmp
    print(f"[corpus] saved {corp.n_tokens} tokens, hash={corp.sha256[:8]}")
    return corp


def stage_model(args):
    print(f"[model] loading GPT-2 Small (variant={args.variant})")
    return load_gpt2_small(
        random_weights=(args.variant == "random"),
        device=args.device,
        seed=args.random_init_seed,
    )


def stage_sc(model, out_dir: Path, args) -> dict:
    path = out_dir / "sc.pt"
    if path.exists() and not args.force:
        return torch.load(path, weights_only=False)
    print("[sc] computing SC matrices from weight composition")
    sc = compute_sc(model, device=args.device)
    payload = {
        "sc_q": sc.sc_q,
        "sc_k": sc.sc_k,
        "sc_v": sc.sc_v,
        "sc_total": sc.sc_total,
        "n_layers": sc.n_layers,
        "n_heads": sc.n_heads,
    }
    _save(path, payload)
    return payload


def stage_activity(model, corpus, out_dir: Path, args):
    path = out_dir / "activity.pt"
    if path.exists() and not args.force:
        return torch.load(path, weights_only=False)
    print("[activity] extracting per-head activity over corpus")
    activity = extract_activity(
        model,
        corpus.tokens,
        batch_size=args.batch_size,
        device=args.device,
    )
    payload = {
        "l2": activity.l2,
        "residual_norm": activity.residual_norm,
        "n_layers": activity.n_layers,
        "n_heads": activity.n_heads,
        "d_head": activity.d_head,
        "d_model": activity.d_model,
    }
    _save(path, payload)
    return payload


def _activity_from_payload(p):
    from connectomics.activity import Activity
    return Activity(
        l2=p["l2"], residual_norm=p["residual_norm"],
        n_layers=p["n_layers"], n_heads=p["n_heads"],
        d_head=p["d_head"], d_model=p["d_model"],
    )


def stage_fc(activity_payload, out_dir: Path, args) -> dict:
    path = out_dir / "fc.pt"
    if path.exists() and not args.force:
        return torch.load(path, weights_only=False)
    print("[fc] computing both FC variants (L2, partial)")
    activity = _activity_from_payload(activity_payload)
    fcs = fc_mod.all_fc_measures(activity)
    _save(path, fcs)
    return fcs


def stage_communities(fcs: dict, out_dir: Path, args) -> dict:
    path = out_dir / "communities.pt"
    if path.exists() and not args.force:
        return torch.load(path, weights_only=False)
    print("[communities] running Louvain gamma-sweep on each FC variant")
    resolutions = np.linspace(0.5, 2.0, 16)
    payload: dict = {}
    for name, mat in fcs.items():
        sweep = comm.sweep_resolution(mat, resolutions=resolutions, seed=args.seed)
        payload[name] = {
            "resolutions": sweep.resolutions,
            "modularities": sweep.modularities,
            "n_communities": sweep.n_communities,
            "partitions": sweep.partitions,
            "best_idx": sweep.best_idx,
        }
    _save(path, payload)
    return payload


def stage_statistics(
    sc_payload: dict,
    fcs: dict,
    communities: dict,
    out_dir: Path,
) -> dict:
    print("[stats] computing IOI circuit recovery and SC-FC coupling")
    n_layers = sc_payload["n_layers"]
    n_heads = sc_payload["n_heads"]
    N = n_layers * n_heads
    sc_total = sc_payload["sc_total"]

    results: dict = {}

    # Circuit recovery against the published IOI circuit. NMI is restricted to
    # labeled heads (the version that does not dilute by background entropy);
    # purity reports the fraction of labeled heads in their class's majority
    # community.
    ioi_labels = ioi_label_dict(n_heads)
    recovery: dict[str, dict[str, float]] = {}
    for name in fcs:
        best = communities[name]
        part = best["partitions"][best["best_idx"]]
        recovery[name] = {
            "ioi_nmi": met.circuit_nmi(part, ioi_labels, N),
            "ioi_purity": met.circuit_purity(part, ioi_labels),
        }
    results["circuit_recovery"] = recovery

    # SC-FC coupling (kappa), global and by depth band, for each FC variant.
    kappas: dict[str, dict] = {}
    for name, fc in fcs.items():
        kappas[name] = {
            "global": met.sc_fc_coupling(sc_total, fc),
            **met.layer_depth_kappa(sc_total, fc, n_layers, n_heads),
        }
    results["sc_fc_coupling"] = kappas

    # Modularity of the best partition per FC variant.
    modularity: dict[str, dict] = {}
    for name in fcs:
        best = communities[name]
        idx = best["best_idx"]
        modularity[name] = {
            "Q": float(best["modularities"][idx]),
            "K": int(best["n_communities"][idx]),
            "gamma": float(best["resolutions"][idx]),
        }
    results["modularity"] = modularity

    _save_json(out_dir / "hypothesis_tests.json", results)
    return results


def stage_figures(sc_payload: dict, fcs: dict, communities: dict, out_dir: Path):
    """Emit the standard figure set as PNGs."""
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_layers = sc_payload["n_layers"]
    n_heads = sc_payload["n_heads"]

    fig, ax = plt.subplots(figsize=(6, 6))
    viz.plot_matrix(
        sc_payload["sc_total"], n_layers, n_heads,
        title="SC (total, GPT-2 Small)", cmap="viridis", vmin=0, ax=ax,
    )
    fig.savefig(fig_dir / "sc_total.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for name, mat in fcs.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        viz.plot_matrix(mat, n_layers, n_heads, title=f"FC [{name}]", ax=ax)
        fig.savefig(fig_dir / f"fc_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Communities on the primary (partial) FC measure.
    primary = "partial" if "partial" in fcs else next(iter(fcs))
    best = communities[primary]
    part = best["partitions"][best["best_idx"]]
    fig, ax = plt.subplots(figsize=(6, 6))
    viz.plot_communities(
        fcs[primary], part, n_layers, n_heads,
        title=f"Communities ({primary}, gamma={best['resolutions'][best['best_idx']]:.2f})",
        ax=ax,
    )
    fig.savefig(fig_dir / "communities.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_modularity_sweep(
        best["resolutions"], best["modularities"], best["n_communities"], ax=ax,
    )
    ax.set_title(f"Modularity vs gamma ({primary})")
    fig.savefig(fig_dir / "modularity_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="GPT-2 Small IOI validation")
    p.add_argument("--variant", choices=["trained", "random"], default="trained")
    p.add_argument("--device", default=best_device(),
                   help="cuda / mps / cpu (default: best available)")
    p.add_argument("--corpus-size", type=int, default=100_000)
    p.add_argument("--corpus-seed", type=int, default=0)
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-init-seed", type=int, default=0)
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--force", action="store_true", help="overwrite cached stages")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "config.json", vars(args))

    corpus = stage_corpus(args, out_dir)
    model = stage_model(args)
    sc_payload = stage_sc(model, out_dir, args)
    activity_payload = stage_activity(model, corpus, out_dir, args)
    fcs = stage_fc(activity_payload, out_dir, args)

    # Free GPU memory before community detection (CPU-only step).
    del model
    empty_cache(args.device)

    communities = stage_communities(fcs, out_dir, args)
    results = stage_statistics(sc_payload, fcs, communities, out_dir)
    stage_figures(sc_payload, fcs, communities, out_dir)

    print(f"\n[done] {args.variant} artifacts in {out_dir}")
    for measure in fcs:
        rec = results["circuit_recovery"][measure]
        kap = results["sc_fc_coupling"][measure]
        print(f"  {measure:8s} IOI NMI={rec['ioi_nmi']:.3f} "
              f"purity={rec['ioi_purity']:.3f}  global kappa={kap['global']:.3f}")


if __name__ == "__main__":
    main()
