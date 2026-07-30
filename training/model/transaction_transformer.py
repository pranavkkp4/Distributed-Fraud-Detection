"""Small, fixed-shape encoder model for transaction-history fraud scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TransactionTransformerConfig:
    """Static dimensions for the latency-oriented transaction transformer.

    Keeping these dimensions static makes the exported model suitable for
    CUDA-graph replay and bounded dynamic batching in the serving plane.
    """

    context_length: int = 16
    feature_width: int = 32
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    feedforward_size: int = 128
    explanation_width: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.context_length,
            self.feature_width,
            self.hidden_size,
            self.num_heads,
            self.num_layers,
            self.feedforward_size,
            self.explanation_width,
        ) < 1:
            raise ValueError("all model dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class FraudModelOutput:
    """Model outputs consumed by fraud policy and explanation components."""

    fraud_probability: Tensor
    calibrated_confidence: Tensor
    explanation: Tensor


class FixedShapeTransactionTransformer(nn.Module):
    """An encoder-only transformer over a bounded recent transaction history.

    Input is a normalized floating-point tensor with exact shape
    ``[batch, context_length, feature_width]``.  The model returns a fraud
    probability, a confidence score, and a fixed-width explanation vector.
    """

    def __init__(self, config: TransactionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.feature_width, config.hidden_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.context_length + 1, config.hidden_size)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Histories are always dense and fixed-width; nested tensor conversion
        # adds no benefit and produces a misleading runtime warning here.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.fraud_head = nn.Linear(config.hidden_size, 1)
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        self.explanation_head = nn.Linear(config.hidden_size, config.explanation_width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use stable initialization without relying on global module defaults."""
        nn.init.normal_(self.class_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, transaction_history: Tensor) -> FraudModelOutput:
        """Score a fixed-shape batch of normalized transaction histories."""
        expected = (self.config.context_length, self.config.feature_width)
        if transaction_history.ndim != 3 or tuple(transaction_history.shape[1:]) != expected:
            raise ValueError(
                "transaction_history must have shape "
                f"[batch, {expected[0]}, {expected[1]}], got {tuple(transaction_history.shape)}"
            )
        if not transaction_history.is_floating_point():
            raise TypeError("transaction_history must use a floating-point dtype")

        batch_size = transaction_history.shape[0]
        encoded = self.input_projection(transaction_history)
        class_token = self.class_token.expand(batch_size, -1, -1)
        encoded = torch.cat((class_token, encoded), dim=1)
        encoded = encoded + self.position_embedding
        pooled = self.final_norm(self.encoder(encoded)[:, 0])
        return FraudModelOutput(
            fraud_probability=torch.sigmoid(self.fraud_head(pooled)),
            calibrated_confidence=torch.sigmoid(self.confidence_head(pooled)),
            explanation=torch.tanh(self.explanation_head(pooled)),
        )
