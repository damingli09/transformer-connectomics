# Transformer Connectomics — Review Artifact

This artifact contains the code and precomputed data needed to reproduce the
results reported in the paper. The framework adapts structural connectivity
(SC), functional connectivity (FC), community detection, and
structure–function coupling from network neuroscience to the attention heads
of transformer language models.

Everything here corresponds to results shown in the paper; exploratory
analyses that do not appear in the paper have been removed.

## Layout

```
code/
  connectomics/                 # core library
    sc.py                       # structural connectivity (weight composition)
    activity.py                 # per-head activity extraction over a corpus
    fc.py                       # functional connectivity (L2 and partial)
    community.py                # Louvain community detection + NMI
    metrics.py                  # SC–FC coupling (kappa), circuit NMI / purity
    nulls.py                    # permutation-null utilities
    corpus.py                   # fixed, SHA-pinned corpus sampling
    models.py                   # model loaders (GPT-2, Pythia; random-weight control)
    device.py                   # device selection helpers
    viz.py                      # heatmap / community / modularity plotting
  experiments/
    validation1_llama_gqa/      # Validation I  — GQA topology in Llama-3.1-8B
    validation2_gpt2_ioi/       # Validation II — IOI circuit recovery in GPT-2 Small
    validation3_pythia/         # Validation III — GPT-2 vs. Pythia-160M
  requirements.txt
data/
  corpus_100k.pt                # the fixed 100k-token corpus (SHA-pinned)
  gpt2_trained/                 # GPT-2 Small connectome + statistics
  gpt2_random/                  # random-weight control
  pythia_160m/                  # Pythia-160M connectome + statistics
  llama31_8b/                   # Llama-3.1-8B connectome + statistics
```

Each model directory under `data/` contains:
- `sc.pt` — Q/K/V/total SC matrices.
- `fc.pt` — FC matrices (L2 and partial; Llama ships partial only).
- `communities.pt` — Louvain gamma-sweep results.
- `hypothesis_tests.json` — the statistics reported in the paper.
- `ioi_significance.json` (GPT-2 trained only) — permutation p-values for Table 3.

## Setup

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

The analysis and figure scripts read the precomputed `data/` directory, so the
tables and figures can be reproduced without any model download or GPU. The
`run.py` pipelines regenerate the raw connectomes from scratch and require the
corresponding models (and a GPU for the larger ones).

### Validation I — Llama-3.1-8B GQA (Section: Validation I)
```bash
python experiments/validation1_llama_gqa/render_figures.py
```
Reproduces the GQA verdict panels (H1/H2/H3) and the SC / partial-FC grid.
To regenerate from the model (large GPU required):
```bash
python experiments/validation1_llama_gqa/run.py \
    --model meta-llama/Llama-3.1-8B --output out/llama31_8b
```

### Validation II — GPT-2 Small IOI circuit (Section: Validation II)
```bash
python experiments/validation2_gpt2_ioi/analyze.py            # Table 3 + kappa
python experiments/validation2_gpt2_ioi/plot_trained_vs_random.py   # appendix grid
```
`analyze.py` prints the IOI recovery table (NMI + purity, trained vs. random)
with the permutation p-values from `ioi_significance.json`. To recompute the
permutation null:
```bash
python experiments/validation2_gpt2_ioi/ioi_significance.py --n-perm 10000
```
To regenerate the connectomes (a single mid-range GPU is sufficient):
```bash
python experiments/validation2_gpt2_ioi/run.py --variant trained --output out/trained
python experiments/validation2_gpt2_ioi/run.py --variant random  --output out/random
```

### Validation III — GPT-2 vs. Pythia-160M (Section: Validation III)
```bash
python experiments/validation3_pythia/compare.py
```
Prints the depth-resolved kappa comparison and renders the SC / partial-FC
grid. To regenerate the Pythia connectome:
```bash
python experiments/validation3_pythia/run.py --size 160m --step 143000 \
    --output out/pythia160m
```

## Notes

- All experiments use the same SHA-pinned 100k-token corpus (`data/corpus_100k.pt`),
  so every run sees byte-identical input.
- The random-weight control is an identically shaped GPT-2 Small with freshly
  reinitialized weights, pushed through the identical pipeline.
- IOI circuit labels follow Wang et al. (2022); GQA group structure follows the
  Llama-3.1 architecture.
