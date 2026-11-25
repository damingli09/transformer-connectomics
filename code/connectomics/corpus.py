"""
Corpus sampling for FC extraction.

The protocol fixes one 100k-token Pile sample at a published seed
and hash. The same sample is used at every Pythia checkpoint and for every
trained-vs-random comparison.

This module:

- Streams a HuggingFace dataset (default: NeelNanda/pile-10k, a convenient
  Pile snapshot mirrored for TransformerLens; switch to The Pile proper if
  bandwidth allows).
- Tokenizes with the model's tokenizer.
- Truncates / packs into a 1D token tensor of the requested size.
- Returns the SHA256 hash so the corpus can be referenced unambiguously in
  the paper and pre-registration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass
class TokenCorpus:
    tokens: torch.Tensor   # 1D int64 tensor
    n_tokens: int
    sha256: str
    source: str

    def reshape_seqs(self, seq_len: int) -> torch.Tensor:
        n_full = (self.n_tokens // seq_len) * seq_len
        return self.tokens[:n_full].reshape(-1, seq_len)


def _hash_tokens(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest()


def sample_pile_tokens(
    tokenizer,
    n_tokens: int = 100_000,
    seed: int = 0,
    dataset_name: str = "NeelNanda/pile-10k",
    split: str = "train",
    shuffle: bool = True,
) -> TokenCorpus:
    """Sample a fixed-seed token corpus from a Pile snapshot.

    Args:
        tokenizer: a HuggingFace tokenizer (use ``model.tokenizer`` from
            TransformerLens).
        n_tokens: target total tokens.
        seed: dataset shuffle seed (also part of the hash).
        dataset_name: HF dataset name; the default is a small-enough mirror
            for quick downloads. Use ``"monology/pile-uncopyrighted"`` for
            the full Pile.
        split: dataset split.
        shuffle: if True, shuffle the dataset before tokenizing.

    Returns:
        TokenCorpus with .tokens, .n_tokens, .sha256, .source.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    pieces: list[torch.Tensor] = []
    total = 0
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        pieces.append(ids)
        total += ids.numel()
        if total >= n_tokens:
            break
    if total < n_tokens:
        raise RuntimeError(
            f"Dataset {dataset_name} yielded only {total} tokens; requested {n_tokens}."
        )

    tokens = torch.cat(pieces, dim=0)[:n_tokens]
    return TokenCorpus(
        tokens=tokens.long(),
        n_tokens=int(tokens.numel()),
        sha256=_hash_tokens(tokens),
        source=f"{dataset_name}:{split}:seed={seed}",
    )


def save_corpus(corpus: TokenCorpus, path: str) -> None:
    """Persist a TokenCorpus to a single .pt file."""
    torch.save(
        {
            "tokens": corpus.tokens,
            "n_tokens": corpus.n_tokens,
            "sha256": corpus.sha256,
            "source": corpus.source,
        },
        path,
    )


def load_corpus(path: str) -> TokenCorpus:
    data = torch.load(path, weights_only=False)
    return TokenCorpus(
        tokens=data["tokens"],
        n_tokens=data["n_tokens"],
        sha256=data["sha256"],
        source=data["source"],
    )
