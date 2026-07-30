"""Offline activation calibration for E4M3 and E5M2 inference paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import torch
from torch import Tensor, nn


class FP8Format(str, Enum):
    """Supported FP8 encodings and their finite maximum magnitudes."""

    E4M3 = "E4M3"
    E5M2 = "E5M2"

    @property
    def max_finite(self) -> float:
        return 448.0 if self is FP8Format.E4M3 else 57344.0


@dataclass(frozen=True)
class TensorScaleMetadata:
    """Scale and outlier policy for a single observed activation tensor."""

    format: str
    amax: float
    clip_threshold: float
    scale: float
    outlier_count: int
    observation_count: int
    outlier_ratio: float
    outlier_policy: str


@dataclass(frozen=True)
class FP8CalibrationResult:
    """Serializable scale metadata keyed by PyTorch module name."""

    format: str
    percentile: float
    outlier_policy: str
    tensors: dict[str, TensorScaleMetadata]

    def to_dict(self) -> dict[str, object]:
        """Convert metadata to JSON-compatible primitives."""
        return {
            "format": self.format,
            "percentile": self.percentile,
            "outlier_policy": self.outlier_policy,
            "tensors": {name: asdict(scale) for name, scale in self.tensors.items()},
        }


def _tensor_from_output(output: object) -> Tensor | None:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return next((item for item in output if isinstance(item, Tensor)), None)
    return None


def calibrate_fp8(
    model: nn.Module,
    batches: Iterable[Tensor],
    fp8_format: FP8Format = FP8Format.E4M3,
    percentile: float = 0.999,
    outlier_policy: str = "clip",
) -> FP8CalibrationResult:
    """Collect module activation ranges and calculate inference dequant scales.

    ``scale`` follows ``dequantized = fp8_value * scale``. Values exceeding
    the percentile range are counted and are either clipped (the serving
    default) or mark the tensor as unsuitable for FP8 (``fallback``).
    """
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    if outlier_policy not in {"clip", "fallback"}:
        raise ValueError("outlier_policy must be 'clip' or 'fallback'")

    observations: dict[str, list[Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, (nn.Linear, nn.LayerNorm)):
            continue

        def capture(_: nn.Module, __: tuple[object, ...], output: object, *, key: str = name) -> None:
            tensor = _tensor_from_output(output)
            if tensor is not None:
                observations.setdefault(key, []).append(tensor.detach().abs().flatten().cpu())

        handles.append(module.register_forward_hook(capture))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                model(batch)
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()

    if not observations:
        raise ValueError("no activations collected; provide at least one non-empty batch")

    scales: dict[str, TensorScaleMetadata] = {}
    for name, values in observations.items():
        all_values = torch.cat(values)
        amax = float(all_values.max().item())
        threshold = float(torch.quantile(all_values, percentile).item())
        outlier_count = int((all_values > threshold).sum().item())
        if threshold == 0.0:
            scale = 1.0
        elif outlier_policy == "fallback" and outlier_count:
            scale = 0.0
        else:
            scale = threshold / fp8_format.max_finite
        scales[name] = TensorScaleMetadata(
            format=fp8_format.value,
            amax=amax,
            clip_threshold=threshold,
            scale=scale,
            outlier_count=outlier_count,
            observation_count=int(all_values.numel()),
            outlier_ratio=outlier_count / max(1, int(all_values.numel())),
            outlier_policy=outlier_policy,
        )
    return FP8CalibrationResult(fp8_format.value, percentile, outlier_policy, scales)
