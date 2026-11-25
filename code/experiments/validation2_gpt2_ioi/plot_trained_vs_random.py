"""
2x2 grid figure comparing the SC and FC connectomes of trained vs.
random-weight GPT-2 Small.

Top row: trained.  Bottom row: random-weight control.  Columns: SC | FC.

A single FC variant (default: partial, the primary measure) is used for both
rows so the side-by-side comparison is meaningful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connectomics import viz  # noqa: E402

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2].parent / "data"


def _load_payloads(variant_dir: Path) -> tuple[dict, dict]:
    sc = torch.load(variant_dir / "sc.pt", weights_only=False)
    fc = torch.load(variant_dir / "fc.pt", weights_only=False)
    return sc, fc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                   help="Parent dir containing gpt2_trained/ and gpt2_random/.")
    p.add_argument("--trained-subdir", default="gpt2_trained")
    p.add_argument("--random-subdir", default="gpt2_random")
    p.add_argument("--fc-variant", default="partial",
                   help="FC variant to show in the FC column (default: partial).")
    p.add_argument("--clip-pct", type=float, default=99.0)
    args = p.parse_args()

    root = Path(args.data_root)
    sc_t, fc_t = _load_payloads(root / args.trained_subdir)
    sc_r, fc_r = _load_payloads(root / args.random_subdir)
    n_layers = sc_t["n_layers"]
    n_heads = sc_t["n_heads"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    rows = [
        ("trained", sc_t["sc_total"], fc_t[args.fc_variant]),
        ("random-weight", sc_r["sc_total"], fc_r[args.fc_variant]),
    ]

    for row_idx, (label, sc, fc) in enumerate(rows):
        viz.plot_matrix(sc, n_layers, n_heads,
                        title=f"SC — {label}",
                        signed=False, cmap="magma", clip_pct=args.clip_pct,
                        ax=axes[row_idx, 0],
                        cbar_label=r"$\|\,W_{OV}^A \, W_{*}^B\,\|_F$")
        viz.plot_matrix(fc, n_layers, n_heads,
                        title=f"FC [{args.fc_variant}] — {label}",
                        signed=True, cmap="RdBu_r", clip_pct=args.clip_pct,
                        ax=axes[row_idx, 1], cbar_label="correlation")

    fig.suptitle(
        f"GPT-2 Small connectomics: trained vs. random-weight control\n"
        f"(heads ordered by layer; FC shown using the {args.fc_variant} variant)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    out = root / "gpt2_trained_vs_random_grid.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
