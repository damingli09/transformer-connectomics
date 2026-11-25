"""
Model loaders for the transformer-connectomics framework.

Two model families are supported via TransformerLens:

- GPT-2 Small.
- Pythia.

Random-weight variants reuse the trained architecture's config and
reinitialize all parameters from a fixed-seed initializer.
"""

from __future__ import annotations

import os

import torch


def _workaround_hf_empty_bearer():
    """Avoid an httpx 'Illegal header value' error for anonymous HF Hub calls.

    huggingface_hub + httpx send a literal 'Bearer ' header (no token) when
    neither HF_TOKEN nor HUGGING_FACE_HUB_TOKEN is set, which httpx rejects.
    Setting HF_TOKEN to a dummy non-empty value triggers the anonymous
    request path and works for public models.
    """
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HF_TOKEN"] = "hf_anon"


def _reinit_(model) -> None:
    """In-place reinitialization of all parameters (random-weight control).

    Weight matrices get a Kaiming-normal init; biases and other 1D
    parameters are zeroed.
    """
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.kaiming_normal_(p, nonlinearity="linear")
        else:
            torch.nn.init.zeros_(p)


def load_gpt2_small(
    random_weights: bool = False,
    device: str | torch.device | None = None,
    seed: int = 0,
):
    """Load GPT-2 Small via TransformerLens.

    Args:
        random_weights: if True, reinitialize all parameters after loading.
            Used for the random-weight architectural control.
        device: torch device. If None, auto-picks cuda > mps > cpu.
        seed: RNG seed for the random-weight reinit.
    """
    _workaround_hf_empty_bearer()
    from transformer_lens import HookedTransformer

    from .device import best_device

    if device is None:
        device = best_device()
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    # We extract via hook_z and apply W_O manually (see activity.py), so the
    # heavier hook_result path is not needed.
    if random_weights:
        torch.manual_seed(seed)
        _reinit_(model)
    model.eval()
    return model


def load_pythia(
    size: str,
    step: int,
    device: str | torch.device | None = None,
):
    """Load a Pythia checkpoint via TransformerLens.

    Args:
        size: model size, e.g. "160m".
        step: training step. Must match a published EleutherAI revision
            (e.g. step143000).
        device: torch device. If None, auto-picks cuda > mps > cpu.
    """
    _workaround_hf_empty_bearer()
    from transformer_lens import HookedTransformer

    from .device import best_device

    if device is None:
        device = best_device()
    name = f"EleutherAI/pythia-{size}-deduped"
    model = HookedTransformer.from_pretrained(
        name, checkpoint_value=step, device=device,
    )
    model.eval()
    return model
