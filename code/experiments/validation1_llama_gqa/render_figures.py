"""
Render Validation I (Llama-3.1-8B GQA) figures from precomputed results.

Produces:
  - <out-dir>/llama_gqa_verdicts.png — three-panel bar chart of the GQA tests
    (H1 structural cosine, H2 FC magnitude, H3 Louvain group purity).
  - <out-dir>/llama_gqa_grid.png — SC (total) and partial FC heatmaps on the
    full head-node grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2].parent / "data"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figures"


def plot_verdicts(hyp: dict, out_path: Path):
    """Three-panel bar chart: H1 (3 SC pathways), H2, H3."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # H1: within vs between per SC pathway.
    ax = axes[0]
    pathways = ["SC_Q", "SC_K", "SC_V"]
    within = [hyp["H1_gqa_structural_cosine"][p.lower()]["within_group_mean_cosine"]
              for p in pathways]
    between = [hyp["H1_gqa_structural_cosine"][p.lower()]["between_group_mean_cosine"]
               for p in pathways]
    x = np.arange(len(pathways))
    w = 0.35
    ax.bar(x - w / 2, within, w, label="within GQA group", color="C0")
    ax.bar(x + w / 2, between, w, label="between groups", color="C3", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(pathways)
    ax.set_ylabel("cosine of incoming-SC vectors")
    ax.set_title("H1: structural similarity\nwithin GQA groups")
    ax.set_ylim(0.9, 1.005)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # H2: within vs between FC magnitude.
    ax = axes[1]
    h2 = hyp["H2_within_group_fc_magnitude"]
    labels = ["within group", "between groups"]
    vals = [h2["within_group_mean_abs_fc"], h2["between_group_mean_abs_fc"]]
    bars = ax.bar(labels, vals, color=["C0", "C3"], alpha=0.85)
    ax.set_ylabel(r"mean $|\mathrm{FC}|$ (same-layer pairs)")
    ax.set_title(f"H2: functional similarity\n$\\Delta$ = +{h2['delta_obs']:.3f}")
    ax.grid(True, alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)

    # H3: group purity vs null.
    ax = axes[2]
    h3 = hyp["H3_louvain_group_purity"]
    obs = h3["observed_group_purity"]
    mu = h3["null_mean_purity"]
    sd = h3["null_std_purity"]
    ax.bar(["observed", "shuffle\nnull"], [obs, mu], yerr=[0, sd],
           color=["C0", "gray"], alpha=0.85, capsize=8)
    ax.set_ylabel("GQA group purity")
    ax.set_title("H3: Louvain communities\nrespect GQA groups")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.text(0, obs + 0.02, f"{obs:.3f}", ha="center", fontsize=10)
    ax.text(1, mu + sd + 0.02, f"{mu:.3f} +/- {sd:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        "Llama-3.1-8B: grouped-query attention imposes a detectable connectome "
        "signature\n(all three tests pass at Bonferroni alpha = 0.01)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_grid(sc_total, fc, n_layers, n_heads, out_path: Path):
    """Side-by-side SC (total) and partial FC heatmaps."""
    matrices = [("SC (total)", sc_total), ("Partial FC", fc)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (title, mat) in zip(axes, matrices):
        arr = mat.cpu().numpy()
        if title.startswith("SC"):
            vmax = float(np.percentile(arr[arr > 0], 99)) if (arr > 0).any() else 1.0
            im = ax.imshow(arr, cmap="magma", vmin=0, vmax=vmax,
                           aspect="equal", interpolation="nearest")
        else:
            vmax = float(np.percentile(np.abs(arr), 99))
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="equal", interpolation="nearest")
        for l in range(1, n_layers):
            ax.axhline(l * n_heads - 0.5, color="white", lw=0.2, alpha=0.5)
            ax.axvline(l * n_heads - 0.5, color="white", lw=0.2, alpha=0.5)
        layer_ticks = [l * n_heads + n_heads / 2 - 0.5 for l in range(0, n_layers, 8)]
        layer_labels = [f"L{l}" for l in range(0, n_layers, 8)]
        ax.set_xticks(layer_ticks); ax.set_xticklabels(layer_labels, fontsize=7)
        ax.set_yticks(layer_ticks); ax.set_yticklabels(layer_labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.suptitle("Llama-3.1-8B connectome", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_ROOT / "llama31_8b"),
                   help="Directory containing sc.pt, fc.pt, hypothesis_tests.json.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hyp = json.loads((data_dir / "hypothesis_tests.json").read_text())

    print("[render] verdict panels")
    plot_verdicts(hyp, out_dir / "llama_gqa_verdicts.png")
    print(f"  -> {out_dir / 'llama_gqa_verdicts.png'}")

    sc_payload = torch.load(data_dir / "sc.pt", weights_only=False)
    fc_payload = torch.load(data_dir / "fc.pt", weights_only=False)

    print("[render] SC / FC connectivity grid")
    plot_grid(
        sc_payload["sc_total"], fc_payload["partial"],
        n_layers=sc_payload["n_layers"], n_heads=sc_payload["n_heads"],
        out_path=out_dir / "llama_gqa_grid.png",
    )
    print(f"  -> {out_dir / 'llama_gqa_grid.png'}")


if __name__ == "__main__":
    main()
