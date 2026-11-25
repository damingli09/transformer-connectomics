"""Within-label permutation significance test for IOI-circuit recovery.

Holds the discovered Louvain partition fixed and permutes the IOI class
labels among the labeled heads. For each of ``n_perm`` resamples it
recomputes restricted NMI and purity, yielding a null distribution that
answers: do the discovered communities align with the true 7-class IOI
structure better than a random relabeling of the same class sizes?

p-values use the add-one (Phipson & Smyth, 2010) correction, matching the
GQA permutation tests. Consumes the already-computed trained partition in
``communities.pt`` -- no model re-run required.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectomics import metrics as met  # noqa: E402
from connectomics.nulls import empirical_p_value  # noqa: E402
from circuit_labels import ioi_label_dict  # noqa: E402

N_HEADS = 12
N_NODES = 144  # GPT-2 Small: 12 layers x 12 heads
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2].parent / "data"


def run_null(
    partition: np.ndarray,
    ioi_labels: dict[int, int],
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    labeled_nodes = sorted(ioi_labels.keys())
    classes = np.array([ioi_labels[n] for n in labeled_nodes], dtype=np.int64)

    obs_nmi = met.circuit_nmi(partition, ioi_labels, N_NODES)
    obs_purity = met.circuit_purity(partition, ioi_labels)

    null_nmi = np.empty(n_perm)
    null_purity = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(classes)
        perm_labels = {n: int(c) for n, c in zip(labeled_nodes, perm)}
        null_nmi[k] = met.circuit_nmi(partition, perm_labels, N_NODES)
        null_purity[k] = met.circuit_purity(partition, perm_labels)

    return {
        "ioi_nmi": {
            "observed": float(obs_nmi),
            "null_mean": float(null_nmi.mean()),
            "null_std": float(null_nmi.std()),
            "p_one_sided": float(empirical_p_value(obs_nmi, null_nmi, "greater")),
        },
        "ioi_purity": {
            "observed": float(obs_purity),
            "null_mean": float(null_purity.mean()),
            "null_std": float(null_purity.std()),
            "p_one_sided": float(empirical_p_value(obs_purity, null_purity, "greater")),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--communities",
        default=str(DEFAULT_DATA_ROOT / "gpt2_trained" / "communities.pt"),
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_DATA_ROOT / "gpt2_trained" / "ioi_significance.json"),
    )
    ap.add_argument("--measures", nargs="+", default=["l2", "partial"])
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    communities = torch.load(args.communities, weights_only=False)
    ioi_labels = ioi_label_dict(N_HEADS)

    out: dict = {
        "test": "within-label permutation null (IOI class labels shuffled among labeled heads, partition fixed)",
        "n_perm": args.n_perm,
        "seed": args.seed,
        "n_labeled_heads": len(ioi_labels),
        "n_classes": len(set(ioi_labels.values())),
        "measures": {},
    }
    for measure in args.measures:
        rng = np.random.default_rng(args.seed)
        best = communities[measure]
        partition = np.asarray(best["partitions"][best["best_idx"]])
        res = run_null(partition, ioi_labels, args.n_perm, rng)
        out["measures"][measure] = res
        print(f"[{measure}]")
        for metric, d in res.items():
            print(
                f"  {metric:11s} obs={d['observed']:.3f} "
                f"null={d['null_mean']:.3f}+/-{d['null_std']:.3f} "
                f"p={d['p_one_sided']:.5f}"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
