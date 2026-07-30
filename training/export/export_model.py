"""Export fixed-shape PyTorch checkpoints as immutable serving bundles."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from ..bundle import ModelBundleMetadata, sha256_file
from ..calibration import FP8CalibrationResult
from ..model import FixedShapeTransactionTransformer, TransactionTransformerConfig
from ..sample import make_synthetic_transactions


class _OnnxOutputAdapter(nn.Module):
    """Make the structured PyTorch output explicit to ONNX exporters."""

    def __init__(self, model: FixedShapeTransactionTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, transaction_history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output = self.model(transaction_history)
        return output.fraud_probability, output.calibrated_confidence, output.explanation


@dataclass(frozen=True)
class TensorRTBuildResult:
    """The result of a best-effort TensorRT engine build."""

    available: bool
    engine_path: Path | None
    reason: str | None = None


def export_model_bundle(
    model: FixedShapeTransactionTransformer,
    output_dir: str | Path,
    *,
    model_name: str = "fixed-shape-fraud-transformer",
    model_version: str = "0.1.0",
    calibration: FP8CalibrationResult | None = None,
    opset_version: int = 18,
) -> Path:
    """Publish a fixed-shape ONNX checkpoint plus immutable metadata.

    ``output_dir`` must not already exist. This avoids silently replacing a
    model artifact that may already be referenced by deployed workers.
    """
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing bundle: {destination}")
    missing_export_packages = [
        package
        for package in ("onnx", "onnxscript")
        if importlib.util.find_spec(package) is None
    ]
    if missing_export_packages:
        raise RuntimeError(
            "ONNX export requires the optional export packages "
            f"{', '.join(repr(package) for package in missing_export_packages)}; "
            "install training/requirements.txt"
        )
    destination.mkdir(parents=True)
    try:
        model.eval()
        config = model.config
        sample = make_synthetic_transactions(config, batch_size=1).features
        torch.save(model.state_dict(), destination / "model.pt")
        adapter = _OnnxOutputAdapter(model).eval()
        torch.onnx.export(
            adapter,
            sample,
            destination / "model.onnx",
            input_names=["transaction_history"],
            output_names=["fraud_probability", "calibrated_confidence", "explanation"],
            dynamic_axes=None,
            opset_version=opset_version,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
        artifact_sha256 = {
            "model.onnx": sha256_file(destination / "model.onnx"),
            "model.pt": sha256_file(destination / "model.pt"),
        }
        metadata = ModelBundleMetadata(
            model_name=model_name,
            model_version=model_version,
            model_config=config,
            artifact_sha256=artifact_sha256,
            fp8_calibration=calibration.to_dict() if calibration else None,
        )
        metadata.write(destination / "metadata.json")
    except Exception as export_error:
        # Recursive cleanup also handles temporary subdirectories created by exporters.
        try:
            shutil.rmtree(destination)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            export_error.add_note(f"incomplete bundle cleanup failed: {cleanup_error}")
        raise
    return destination


def _discard_partial_engine(target: Path) -> str | None:
    """Best-effort removal that preserves the original TensorRT failure reason."""
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        return f"partial engine cleanup failed: {exc}"
    return None


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp8: bool = False,
    trtexec: str = "trtexec",
) -> TensorRTBuildResult:
    """Build a TensorRT engine when ``trtexec`` is installed; otherwise report why.

    The function never claims an FP8 engine when the installed TensorRT binary
    rejects FP8, and it does not assume a Hopper/Ada GPU is present.
    """
    source, target = Path(onnx_path), Path(engine_path)
    if not source.is_file():
        return TensorRTBuildResult(False, None, f"ONNX model does not exist: {source}")
    executable = shutil.which(trtexec)
    if executable is None:
        return TensorRTBuildResult(False, None, f"TensorRT executable not found: {trtexec}")
    if target.exists():
        return TensorRTBuildResult(False, None, f"refusing to replace existing engine: {target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return TensorRTBuildResult(False, None, f"cannot create engine directory: {exc}")
    command = [executable, f"--onnx={source}", f"--saveEngine={target}", "--explicitBatch"]
    if fp8:
        command.append("--fp8")
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        cleanup_error = _discard_partial_engine(target)
        reason = f"TensorRT process failed to start: {exc}"
        if cleanup_error:
            reason = f"{reason}; {cleanup_error}"
        return TensorRTBuildResult(False, None, reason)
    if completed.returncode != 0 or not target.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        cleanup_error = _discard_partial_engine(target)
        reason = detail or "TensorRT engine build failed"
        if cleanup_error:
            reason = f"{reason}; {cleanup_error}"
        return TensorRTBuildResult(False, None, reason)
    return TensorRTBuildResult(True, target)


def main(argv: Sequence[str] | None = None) -> int:
    """Provide a small, deterministic bundle-export command line."""
    parser = argparse.ArgumentParser(description="Export a fixed-shape fraud model bundle.")
    parser.add_argument("output_dir", type=Path, help="new destination directory for the bundle")
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--feature-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    config = TransactionTransformerConfig(
        context_length=args.context_length, feature_width=args.feature_width
    )
    output = export_model_bundle(
        FixedShapeTransactionTransformer(config), args.output_dir, model_version=args.version
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module command
    raise SystemExit(main())
