"""
Analysis of trained vs. random-weight GPT-2 Small results.

Loads ``hypothesis_tests.json`` from both variants and produces a clean
comparison covering IOI circuit recovery, SC-FC coupling (kappa), and
modularity. If ``ioi_significance.json`` is present for the trained variant,
its permutation p-values are included.

Outputs:
- ``comparison.json`` in the data root (full numeric comparison).
- A printed table summarizing the headline numbers.
- ``comparison_kappa.png`` — global-kappa bar chart, trained vs. random.
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

MEASURES = ("l2", "partial")
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2].parent / "data"


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="Compare trained vs. random GPT-2 results.")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                   help="Parent dir containing gpt2_trained/ and gpt2_random/.")
    p.add_argument("--trained-subdir", default="gpt2_trained")
    p.add_argument("--random-subdir", default="gpt2_random")
    p.add_argument("--output", default=None,
                   help="Defaults to <data-root>/gpt2_comparison.json")
    args = p.parse_args()

    root = Path(args.data_root)
    trained_dir = root / args.trained_subdir
    random_dir = root / args.random_subdir
    trained_path = trained_dir / "hypothesis_tests.json"
    random_path = random_dir / "hypothesis_tests.json"
    if not trained_path.exists() or not random_path.exists():
        raise SystemExit(f"Missing inputs. Need both:\n  {trained_path}\n  {random_path}")

    trained = _load_json(trained_path)
    random = _load_json(random_path)

    sig_path = trained_dir / "ioi_significance.json"
    sig = _load_json(sig_path) if sig_path.exists() else None

    # --- IOI circuit recovery ---
    recovery_table = []
    for m in MEASURES:
        t = trained["circuit_recovery"].get(m, {})
        r = random["circuit_recovery"].get(m, {})
        row = {
            "measure": m,
            "trained_ioi_nmi": t.get("ioi_nmi"),
            "random_ioi_nmi": r.get("ioi_nmi"),
            "trained_ioi_purity": t.get("ioi_purity"),
            "random_ioi_purity": r.get("ioi_purity"),
        }
        if sig is not None and m in sig.get("measures", {}):
            row["nmi_p"] = sig["measures"][m]["ioi_nmi"]["p_one_sided"]
            row["purity_p"] = sig["measures"][m]["ioi_purity"]["p_one_sided"]
        recovery_table.append(row)

    # --- SC-FC coupling kappa ---
    kappa_table = []
    for m in MEASURES:
        kt = trained["sc_fc_coupling"].get(m, {})
        kr = random["sc_fc_coupling"].get(m, {})
        kappa_table.append({
            "measure": m,
            "trained_global": kt.get("global"), "random_global": kr.get("global"),
            "trained_early": kt.get("early"), "random_early": kr.get("early"),
            "trained_middle": kt.get("middle"), "random_middle": kr.get("middle"),
            "trained_late": kt.get("late"), "random_late": kr.get("late"),
        })

    # --- Modularity ---
    modularity_table = []
    for m in MEASURES:
        mt = trained["modularity"].get(m, {})
        mr = random["modularity"].get(m, {})
        modularity_table.append({
            "measure": m,
            "trained_Q": mt.get("Q"), "random_Q": mr.get("Q"),
            "trained_K": mt.get("K"), "random_K": mr.get("K"),
        })

    comparison = {
        "ioi_circuit_recovery": recovery_table,
        "sc_fc_coupling_kappa": kappa_table,
        "modularity": modularity_table,
    }

    out_path = Path(args.output) if args.output else root / "gpt2_comparison.json"
    with out_path.open("w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # --- printed summary ---
    print("\n## IOI circuit recovery (higher = better)")
    header = f"\n{'measure':>8} | {'NMI T':>7} | {'NMI R':>7} | {'purity T':>8} | {'purity R':>8}"
    if sig is not None:
        header += f" | {'NMI p':>8} | {'pur p':>8}"
    print(header)
    print("-" * (len(header) + 4))
    for r in recovery_table:
        line = (f"{r['measure']:>8} | {r['trained_ioi_nmi']:>7.3f} | {r['random_ioi_nmi']:>7.3f} | "
                f"{r['trained_ioi_purity']:>8.3f} | {r['random_ioi_purity']:>8.3f}")
        if "nmi_p" in r:
            line += f" | {r['nmi_p']:>8.4f} | {r['purity_p']:>8.4f}"
        print(line)

    print("\n## SC-FC coupling kappa — global / early / middle / late")
    print(f"\n{'measure':>8} | {'global T':>8} | {'global R':>8} | {'early T':>8} | "
          f"{'mid T':>7} | {'late T':>7}")
    print("-" * 70)
    for r in kappa_table:
        print(f"{r['measure']:>8} | {r['trained_global']:>8.3f} | {r['random_global']:>8.3f} | "
              f"{r['trained_early']:>8.3f} | {r['trained_middle']:>7.3f} | {r['trained_late']:>7.3f}")

    print("\n## Modularity Q — trained vs. random")
    for r in modularity_table:
        ratio = (r["trained_Q"] / r["random_Q"]) if r["random_Q"] else float("nan")
        print(f"  {r['measure']:>8}: Q_trained={r['trained_Q']:.4f} (K={r['trained_K']})  "
              f"Q_random={r['random_Q']:.4f} (K={r['random_K']})  ratio={ratio:.1f}x")

    print(f"\n[done] full comparison saved to {out_path}")

    # --- global-kappa bar chart ---
    measures = [r["measure"] for r in kappa_table]
    trained_k = [r["trained_global"] for r in kappa_table]
    random_k = [r["random_global"] for r in kappa_table]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(measures))
    w = 0.35
    ax.bar(x - w / 2, trained_k, w, label="trained", color="C0")
    ax.bar(x + w / 2, random_k, w, label="random-weight", color="C3", alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(measures)
    ax.set_ylabel("SC-FC coupling kappa (global)")
    ax.set_title("SC-FC coupling: trained vs. random weights")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = root / "gpt2_comparison_kappa.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] kappa comparison figure -> {fig_path}")


if __name__ == "__main__":
    main()
