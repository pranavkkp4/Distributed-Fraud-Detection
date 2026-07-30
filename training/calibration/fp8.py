"""Offline activation calibration for E4M3 and E5M2 inference paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import torch
from torch import Tensor, nn


FP8_CALIBRATION_SCHEMA_VERSION = 1


class FP8Format(str, Enum):
    """Supported FP8 encodings and their finite maximum magnitudes."""

    E4M3 = "E4M3"
    E5M2 = "E5M2"

    @property
    def max_finite(self) -> float:
        return 448.0 if self is FP8Format.E4M3 else 57344.0


@dataclass(frozen=True)
class TensorScaleMetadata:
    """Runtime Q/DQ contract for one observed activation tensor.

    The field names intentionally match the native E4M3 metadata concepts.
    Module-name mapping into a complete native model remains a serving adapter
    responsibility; this analysis does not claim full-model adapter parity.
    """

    fp8_format: str
    fp8_max: float
    amax: float
    clip_threshold: float
    quant_scale: float
    dequant_scale: float
    outlier_count: int
    observation_count: int
    outlier_ratio: float
    outlier_policy: str
    eligible_for_fp8: bool
    recommended_action: str


@dataclass(frozen=True)
class FP8CalibrationResult:
    """Serializable scale metadata keyed by PyTorch module name."""

    schema_version: int
    format: str
    percentile: float
    outlier_policy: str
    tensors: dict[str, TensorScaleMetadata]

    def to_dict(self) -> dict[str, object]:
        """Convert metadata to JSON-compatible primitives."""
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "percentile": self.percentile,
            "outlier_policy": self.outlier_policy,
            "analysis_behavior": "observes_activations_without_mutation",
            "runtime_clip_contract": (
                "the runtime Q/DQ adapter must clip to clip_threshold before quantization"
            ),
            "native_contract_scope": (
                "E4M3 quant_scale/dequant_scale/clip_threshold semantics align conceptually; "
                "E5M2 support and module-to-native-tensor mapping are not provided"
            ),
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

    Analysis only observes tensors; it never mutates model activations. The
    emitted clip threshold is a runtime/QDQ contract. ``quant_scale`` follows
    ``quantized = clip(value * quant_scale, +/- fp8_max)`` and
    ``dequant_scale`` follows ``value = quantized * dequant_scale``.

    Values exceeding the percentile range are counted. ``clip`` marks the
    tensor eligible with a clip-then-quantize recommendation; ``fallback``
    marks it ineligible and recommends a higher-precision runtime path.
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

        def capture(
            _: nn.Module, __: tuple[object, ...], output: object, *, key: str = name
        ) -> None:
            tensor = _tensor_from_output(output)
            if tensor is not None:
                if tensor.numel() == 0:
                    raise ValueError(f"module {key!r} produced an empty activation tensor")
                # Quantiles are intentionally accumulated as CPU float32 so
                # FP16/BF16 models use a supported, stable analysis dtype.
                observed = tensor.detach().to(device="cpu", dtype=torch.float32)
                if not bool(torch.isfinite(observed).all()):
                    raise ValueError(f"module {key!r} produced non-finite activations")
                observations.setdefault(key, []).append(observed.abs().flatten())

        handles.append(module.register_forward_hook(capture))

    was_training = model.training
    model.eval()
    batch_count = 0
    try:
        with torch.inference_mode():
            for batch in batches:
                batch_count += 1
                if not isinstance(batch, Tensor):
                    raise TypeError("each calibration batch must be a torch.Tensor")
                if batch.numel() == 0:
                    raise ValueError("calibration batches must not be empty")
                if not batch.is_floating_point():
                    raise TypeError("calibration batches must use a floating-point dtype")
                if not bool(torch.isfinite(batch).all()):
                    raise ValueError("calibration batches must contain only finite values")
                model(batch)
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()

    if batch_count == 0:
        raise ValueError("at least one calibration batch is required")
    if not observations:
        raise ValueError("no activations collected; provide at least one non-empty batch")

    scales: dict[str, TensorScaleMetadata] = {}
    for name, values in observations.items():
        all_values = torch.cat(values)
        amax = float(all_values.max().item())
        threshold = float(torch.quantile(all_values, percentile).item())
        outlier_count = int((all_values > threshold).sum().item())
        if threshold == 0.0:
            # The native contract requires finite, positive mutually inverse
            # scales. Unit scales represent an all-zero tensor safely.
            clip_threshold = fp8_format.max_finite
            quant_scale = 1.0
            dequant_scale = 1.0
        else:
            # Prevent quant_scale from overflowing the float32 field consumed
            # by the native metadata contract for extremely small activations.
            minimum_threshold = fp8_format.max_finite / torch.finfo(torch.float32).max
            clip_threshold = max(threshold, minimum_threshold)
            quant_scale = fp8_format.max_finite / clip_threshold
            dequant_scale = clip_threshold / fp8_format.max_finite
        eligible_for_fp8 = not (outlier_policy == "fallback" and outlier_count > 0)
        if not eligible_for_fp8:
            recommended_action = "use_fp16_or_fp32"
        elif outlier_count:
            recommended_action = "clip_to_threshold_then_quantize"
        else:
            recommended_action = "quantize_with_scale"
        scales[name] = TensorScaleMetadata(
            fp8_format=fp8_format.value,
            fp8_max=fp8_format.max_finite,
            amax=amax,
            clip_threshold=clip_threshold,
            quant_scale=quant_scale,
            dequant_scale=dequant_scale,
            outlier_count=outlier_count,
            observation_count=int(all_values.numel()),
            outlier_ratio=outlier_count / max(1, int(all_values.numel())),
            outlier_policy=outlier_policy,
            eligible_for_fp8=eligible_for_fp8,
            recommended_action=recommended_action,
        )
    return FP8CalibrationResult(
        FP8_CALIBRATION_SCHEMA_VERSION, fp8_format.value, percentile, outlier_policy, scales
    )
