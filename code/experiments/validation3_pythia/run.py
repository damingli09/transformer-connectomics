"""
Validation III (data generation): connectome for a Pythia checkpoint.

Computes the same core artifacts as the GPT-2 pipeline so the two models can
be compared under an identical analysis:

  - sc.pt                  (Q/K/V/total SC matrices)
  - fc.pt                  (both FC variants: l2, partial)
  - communities.pt         (Louvain gamma-sweep per FC variant)
  - hypothesis_tests.json  (SC-FC coupling kappa table + modularity)
  - config.json

Pull the results next to the GPT-2 results and run ``compare.py`` for the
GPT-2 vs. Pythia comparison.

Usage:
    python run.py \
        --size 160m --step 143000 \
        --device cuda --batch-size 8 \
        --output out/pythia160m
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
from connectomics import fc as fc_mod  # noqa: E402
from connectomics import metrics as met  # noqa: E402
from connectomics.activity import extract_activity  # noqa: E402
from connectomics.corpus import load_corpus  # noqa: E402
from connectomics.device import best_device, empty_cache  # noqa: E402
from connectomics.models import load_pythia  # noqa: E402
from connectomics.sc import compute_sc  # noqa: E402

# Resolve the shipped corpus relative to this file so the default works from
# any working directory (parents[2] = code/, its parent = artifact root).
DEFAULT_CORPUS = str(Path(__file__).resolve().parents[2].parent / "data" / "corpus_100k.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="160m")
    p.add_argument("--step", type=int, default=143000)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--device", default=best_device())
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    corp = load_corpus(args.corpus)
    print(f"[val3] corpus: {corp.n_tokens} tokens, sha={corp.sha256[:16]}")

    print(f"\n[val3] loading Pythia-{args.size} @ step{args.step}")
    t0 = time.time()
    model = load_pythia(args.size, args.step, device=args.device)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    N = n_layers * n_heads
    print(f"  model loaded: {time.time() - t0:.1f}s (L={n_layers}, H={n_heads}, N={N})")

    print("[val3] computing SC")
    t0 = time.time()
    sc = compute_sc(model, device=args.device)
    torch.save({
        "sc_q": sc.sc_q, "sc_k": sc.sc_k, "sc_v": sc.sc_v,
        "sc_total": sc.sc_total,
        "n_layers": n_layers, "n_heads": n_heads,
    }, out / "sc.pt")
    print(f"  SC: {time.time() - t0:.1f}s")

    print("[val3] extracting activity")
    t0 = time.time()
    activity = extract_activity(
        model, corp.tokens, batch_size=args.batch_size, device=args.device,
    )
    print(f"  activity: {time.time() - t0:.1f}s")

    print("[val3] computing both FC variants")
    t0 = time.time()
    fcs = fc_mod.all_fc_measures(activity)
    torch.save(fcs, out / "fc.pt")
    print(f"  FC: {time.time() - t0:.1f}s")

    del activity, model
    empty_cache(args.device)

    print("[val3] Louvain gamma-sweep on each FC variant")
    t0 = time.time()
    resolutions = np.linspace(0.5, 2.0, 16)
    communities: dict = {}
    for name, mat in fcs.items():
        sweep = comm.sweep_resolution(mat, resolutions=resolutions, seed=args.seed)
        communities[name] = {
            "resolutions": sweep.resolutions,
            "modularities": sweep.modularities,
            "n_communities": sweep.n_communities,
            "partitions": sweep.partitions,
            "best_idx": sweep.best_idx,
        }
    torch.save(communities, out / "communities.pt")
    print(f"  gamma-sweep: {time.time() - t0:.1f}s")

    print("[val3] SC-FC coupling (kappa) + modularity")
    kappas: dict = {}
    modularity: dict = {}
    for name, mat in fcs.items():
        kappas[name] = {
            "global": met.sc_fc_coupling(sc.sc_total, mat),
            **met.layer_depth_kappa(sc.sc_total, mat, n_layers, n_heads),
        }
        best = communities[name]
        idx = best["best_idx"]
        modularity[name] = {
            "Q": float(best["modularities"][idx]),
            "K": int(best["n_communities"][idx]),
            "gamma": float(best["resolutions"][idx]),
        }

    results = {
        "model": f"pythia-{args.size}",
        "step": args.step,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "sc_fc_coupling": kappas,
        "modularity": modularity,
    }
    (out / "hypothesis_tests.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"\n[done] artifacts in {out}/")
    for name in fcs:
        k = kappas[name]
        print(f"  {name:8s} kappa: global={k['global']:+.3f} early={k['early']:+.3f} "
              f"middle={k['middle']:+.3f} late={k['late']:+.3f}")


if __name__ == "__main__":
    main()
