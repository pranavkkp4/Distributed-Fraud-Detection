"""Export fixed-shape PyTorch checkpoints as immutable serving bundles."""

from __future__ import annotations

import argparse
import ctypes
import errno
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from ..bundle import ModelBundleMetadata, sha256_file, verify_model_bundle
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
        return output.fraud_probability, output.confidence_score, output.explanation


@dataclass(frozen=True)
class TensorRTBuildResult:
    """The result of a best-effort TensorRT engine build."""

    available: bool
    engine_path: Path | None
    reason: str | None = None


def _require_cpu_float32(model: nn.Module) -> None:
    """Reject ambiguous exports rather than silently moving or casting a model."""
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.is_floating_point() and (
            tensor.device.type != "cpu" or tensor.dtype != torch.float32
        ):
            raise ValueError(
                "model export requires every floating tensor to be CPU float32; "
                f"{name!r} is {tensor.device}/{tensor.dtype}"
            )


def _onnx_tensor_contract(value_info: object) -> tuple[str, int, tuple[int, ...]]:
    """Extract a fully static ONNX tensor signature."""
    # Kept duck-typed so importing this module does not require ONNX.
    name = value_info.name  # type: ignore[attr-defined]
    tensor_type = value_info.type.tensor_type  # type: ignore[attr-defined]
    dimensions: list[int] = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise RuntimeError(f"ONNX value {name!r} has a dynamic dimension")
        dimensions.append(int(dimension.dim_value))
    return name, int(tensor_type.elem_type), tuple(dimensions)


def _validate_onnx_contract(
    onnx_path: Path, config: TransactionTransformerConfig
) -> None:
    """Run ONNX checker and enforce the serving plane's exact static ABI."""
    import onnx

    model_proto = onnx.load_model(onnx_path, load_external_data=False)
    onnx.checker.check_model(model_proto, full_check=True)
    external_initializers = [
        initializer.name
        for initializer in model_proto.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL or initializer.external_data
    ]
    if external_initializers:
        raise RuntimeError(
            f"ONNX external tensor data is forbidden: {sorted(external_initializers)}"
        )

    expected_inputs = [
        (
            "transaction_history",
            onnx.TensorProto.FLOAT,
            (1, config.context_length, config.feature_width),
        )
    ]
    expected_outputs = [
        ("fraud_probability", onnx.TensorProto.FLOAT, (1, 1)),
        ("confidence_score", onnx.TensorProto.FLOAT, (1, 1)),
        ("explanation", onnx.TensorProto.FLOAT, (1, config.explanation_width)),
    ]
    actual_inputs = [_onnx_tensor_contract(value) for value in model_proto.graph.input]
    actual_outputs = [_onnx_tensor_contract(value) for value in model_proto.graph.output]
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"unexpected ONNX input contract: {actual_inputs!r}")
    if actual_outputs != expected_outputs:
        raise RuntimeError(f"unexpected ONNX output contract: {actual_outputs!r}")


def _rename_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path.

    Windows ``os.rename`` already has no-replace semantics. Linux requires
    ``renameat2(RENAME_NOREPLACE)`` because plain ``rename(2)`` may replace an
    existing empty directory. Unsupported platforms fail closed.
    """
    if os.name == "nt":
        try:
            os.rename(staging, destination)
        except OSError as exc:
            if os.path.lexists(destination):
                raise FileExistsError(
                    f"refusing to replace existing bundle: {destination}"
                ) from exc
            raise
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace directory publication requires renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            os.fsencode(staging),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number,
                f"refusing to replace existing bundle: {destination}",
                destination,
            )
        raise OSError(error_number, os.strerror(error_number), destination)
    raise RuntimeError(
        "atomic no-replace directory publication is supported on Windows and Linux"
    )


def _atomic_publish_directory(staging: Path, destination: Path) -> None:
    """Atomically publish a complete same-filesystem staging directory."""
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing bundle: {destination}")
    try:
        _rename_directory_noreplace(staging, destination)
    except OSError as exc:
        if destination.exists():
            raise FileExistsError(
                f"bundle destination won a concurrent publication race: {destination}"
            ) from exc
        raise


def export_model_bundle(
    model: FixedShapeTransactionTransformer,
    output_dir: str | Path,
    *,
    model_name: str = "fixed-shape-fraud-transformer",
    model_version: str = "0.1.0",
    calibration: FP8CalibrationResult | None = None,
    opset_version: int = 18,
) -> Path:
    """Stage, verify, and atomically publish an immutable model bundle.

    Export currently requires a CPU float32 model and records that contract in
    metadata. The caller's training/evaluation mode is restored on every path.
    ``output_dir`` must not already exist and is made visible only after every
    staged file passes integrity and ONNX ABI validation.
    """
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing bundle: {destination}")
    _require_cpu_float32(model)
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    was_training = model.training
    published = False
    try:
        model.eval()
        config = model.config
        sample = make_synthetic_transactions(config, batch_size=1).features
        torch.save(model.state_dict(), staging / "model.pt")
        adapter = _OnnxOutputAdapter(model).eval()
        torch.onnx.export(
            adapter,
            sample,
            staging / "model.onnx",
            input_names=["transaction_history"],
            output_names=["fraud_probability", "confidence_score", "explanation"],
            dynamic_axes=None,
            opset_version=opset_version,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
        _validate_onnx_contract(staging / "model.onnx", config)
        artifact_sha256 = {
            "model.onnx": sha256_file(staging / "model.onnx"),
            "model.pt": sha256_file(staging / "model.pt"),
        }
        metadata = ModelBundleMetadata(
            model_name=model_name,
            model_version=model_version,
            model_config=config,
            artifact_sha256=artifact_sha256,
            fp8_calibration=calibration.to_dict() if calibration else None,
        )
        metadata.write(staging / "metadata.json")
        verify_model_bundle(staging)
        _atomic_publish_directory(staging, destination)
        published = True
    except BaseException as export_error:
        if not published:
            # Recursive cleanup also handles exporter-created temporary trees.
            try:
                shutil.rmtree(staging)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                export_error.add_note(f"incomplete staging cleanup failed: {cleanup_error}")
        raise
    finally:
        model.train(was_training)
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
    timeout_seconds: float = 120.0,
) -> TensorRTBuildResult:
    """Build a TensorRT engine when ``trtexec`` is installed; otherwise report why.

    The function never claims an FP8 engine when the installed TensorRT binary
    rejects FP8, and it does not assume a Hopper/Ada GPU is present. ``fp8``
    only requests TensorRT builder precision via ``--fp8``; this function does
    not consume or apply the calibration metadata emitted by ``calibrate_fp8``.
    """
    source, target = Path(onnx_path), Path(engine_path)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        return TensorRTBuildResult(False, None, "timeout_seconds must be finite and positive")
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
    command = [executable, f"--onnx={source}", f"--saveEngine={target}"]
    if fp8:
        command.append("--fp8")
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        cleanup_error = _discard_partial_engine(target)
        reason = f"TensorRT build timed out after {timeout_seconds:g} seconds"
        if exc.stderr or exc.stdout:
            output = str(exc.stderr or exc.stdout).strip()
            reason = f"{reason}: {output[-2000:]}"
        if cleanup_error:
            reason = f"{reason}; {cleanup_error}"
        return TensorRTBuildResult(False, None, reason)
    except OSError as exc:
        cleanup_error = _discard_partial_engine(target)
        reason = f"TensorRT process failed to start: {exc}"
        if cleanup_error:
            reason = f"{reason}; {cleanup_error}"
        return TensorRTBuildResult(False, None, reason)
    try:
        engine_size = target.stat().st_size if target.is_file() else 0
    except OSError as exc:
        engine_size = 0
        stat_error = str(exc)
    else:
        stat_error = ""
    if completed.returncode != 0 or engine_size == 0:
        process_output = (completed.stderr or completed.stdout).strip()
        detail_parts = [f"trtexec exited with code {completed.returncode}"]
        if engine_size == 0:
            detail_parts.append("engine file is missing or empty")
        if stat_error:
            detail_parts.append(f"engine inspection failed: {stat_error}")
        if process_output:
            detail_parts.append(process_output[-4000:])
        cleanup_error = _discard_partial_engine(target)
        reason = "; ".join(detail_parts)
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
