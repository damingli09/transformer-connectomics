"""
Validation III: compare GPT-2 Small and Pythia-160M connectomes.

Both models share the same shape (12 layers x 12 heads, d_model=768) but were
trained on different corpora. Holding architecture, corpus, and analysis
fixed, this compares:

  1. The SC and partial-FC matrices (side-by-side heatmap grid).
  2. The SC-FC coupling (kappa) profile by depth band.

Outputs:
  gpt2_vs_pythia_comparison.json — numeric comparison.
  gpt2_vs_pythia_grid.png        — 2x2 grid: GPT-2 vs Pythia, SC | partial FC.
  gpt2_vs_pythia_kappa.png       — global-kappa side-by-side per FC variant.
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

from connectomics import viz  # noqa: E402

MEASURES = ("l2", "partial")
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2].parent / "data"


def _load(model_dir: Path) -> dict:
    return {
        "sc": torch.load(model_dir / "sc.pt", weights_only=False),
        "fc": torch.load(model_dir / "fc.pt", weights_only=False),
        "communities": torch.load(model_dir / "communities.pt", weights_only=False),
        "tests": json.loads((model_dir / "hypothesis_tests.json").read_text()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                   help="Parent dir containing gpt2_trained/ and pythia_160m/.")
    p.add_argument("--gpt2-subdir", default="gpt2_trained")
    p.add_argument("--pythia-subdir", default="pythia_160m")
    p.add_argument("--output", default=None,
                   help="Defaults to <data-root>/gpt2_vs_pythia_comparison.json")
    args = p.parse_args()

    root = Path(args.data_root)
    gpt2 = _load(root / args.gpt2_subdir)
    pyth = _load(root / args.pythia_subdir)
    n_layers, n_heads = gpt2["sc"]["n_layers"], gpt2["sc"]["n_heads"]
    assert (pyth["sc"]["n_layers"], pyth["sc"]["n_heads"]) == (n_layers, n_heads), \
        "shape mismatch between gpt2 and pythia"

    # kappa profile side-by-side.
    kappa = {}
    for name in MEASURES:
        g = gpt2["tests"]["sc_fc_coupling"][name]
        py = pyth["tests"]["sc_fc_coupling"][name]
        kappa[name] = {
            "gpt2_global": g["global"], "pyth_global": py["global"],
            "gpt2_early": g["early"], "pyth_early": py["early"],
            "gpt2_middle": g["middle"], "pyth_middle": py["middle"],
            "gpt2_late": g["late"], "pyth_late": py["late"],
        }

    comparison = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "kappa_side_by_side": kappa,
    }
    out_path = Path(args.output) if args.output else root / "gpt2_vs_pythia_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2, default=str))

    print("\n## SC-FC coupling kappa — global / early / middle / late")
    print(f"{'measure':>8} | {'band':>6} | {'GPT-2':>9} | {'Pythia':>9} | delta (Pyth-GPT2)")
    print("-" * 60)
    for name, r in kappa.items():
        for b in ("global", "early", "middle", "late"):
            g = r[f"gpt2_{b}"]; py = r[f"pyth_{b}"]
            print(f"{name:>8} | {b:>6} | {g:>+9.3f} | {py:>+9.3f} | {py - g:>+12.3f}")
    print(f"\n[done] comparison saved -> {out_path}")

    # 2x2 grid: SC | partial FC, top GPT-2, bottom Pythia.
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    rows = [
        ("GPT-2 Small", gpt2["sc"]["sc_total"], gpt2["fc"]["partial"]),
        ("Pythia-160M", pyth["sc"]["sc_total"], pyth["fc"]["partial"]),
    ]
    for r, (label, sc, fc) in enumerate(rows):
        viz.plot_matrix(sc, n_layers, n_heads, title=f"SC — {label}",
                        signed=False, cmap="magma", ax=axes[r, 0],
                        cbar_label=r"$\|\,W_{OV}^A \, W_{*}^B\,\|_F$")
        viz.plot_matrix(fc, n_layers, n_heads, title=f"FC [partial] — {label}",
                        signed=True, cmap="RdBu_r", ax=axes[r, 1],
                        cbar_label="correlation")
    fig.suptitle(
        "GPT-2 Small vs. Pythia-160M\n(same architecture: 12 layers x 12 heads x d_model=768)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    grid_path = root / "gpt2_vs_pythia_grid.png"
    fig.savefig(grid_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] 2x2 grid -> {grid_path}")

    # kappa global bar chart.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    measures = list(kappa.keys())
    x = np.arange(len(measures))
    w = 0.35
    gpt2_global = [kappa[m]["gpt2_global"] for m in measures]
    pyth_global = [kappa[m]["pyth_global"] for m in measures]
    ax.bar(x - w / 2, gpt2_global, w, label="GPT-2 Small", color="C0")
    ax.bar(x + w / 2, pyth_global, w, label="Pythia-160M", color="C3", alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(measures)
    ax.set_ylabel(r"SC-FC coupling $\kappa_\mathrm{global}$")
    ax.set_title("Global kappa by FC variant — GPT-2 vs Pythia-160M")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    kappa_path = root / "gpt2_vs_pythia_kappa.png"
    fig.savefig(kappa_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] kappa bar chart -> {kappa_path}")


if __name__ == "__main__":
    main()
