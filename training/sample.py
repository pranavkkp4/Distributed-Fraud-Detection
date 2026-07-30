"""Deterministic synthetic data for smoke tests and export calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import TransactionTransformerConfig


@dataclass(frozen=True)
class SyntheticTransactionBatch:
    """A reproducible feature batch and its rule-derived binary labels."""

    features: Tensor
    labels: Tensor


def make_synthetic_transactions(
    config: TransactionTransformerConfig,
    batch_size: int = 8,
    seed: int = 7,
    device: str | torch.device = "cpu",
) -> SyntheticTransactionBatch:
    """Make fixed-width histories with deterministic, learnable fraud signals.

    Features 0--2 model amount, velocity, and device novelty respectively.
    The labels are intentionally simple, making the data appropriate for
    integration checks rather than for measuring real fraud performance.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(
        batch_size, config.context_length, config.feature_width, generator=generator
    )
    recent = features[:, -1]
    risk_signal = recent[:, 0] + 0.7 * recent[:, 1] + 0.5 * recent[:, 2]
    labels = (risk_signal > 0.65).to(dtype=torch.float32).unsqueeze(1)
    return SyntheticTransactionBatch(
        features=features.to(device=device, dtype=torch.float32), labels=labels.to(device),
    )
