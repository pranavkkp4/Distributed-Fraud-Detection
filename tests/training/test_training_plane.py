"""Unit coverage for offline model preparation contracts."""

from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - exercises clean optional-dependency skip
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional training dependency")
class TrainingPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        from training.model import TransactionTransformerConfig

        self.config = TransactionTransformerConfig(
            context_length=4,
            feature_width=6,
            hidden_size=12,
            num_heads=3,
            num_layers=1,
            feedforward_size=24,
            explanation_width=3,
        )

    def test_synthetic_data_is_deterministic(self) -> None:
        from training.sample import make_synthetic_transactions

        first = make_synthetic_transactions(self.config, batch_size=3, seed=123)
        second = make_synthetic_transactions(self.config, batch_size=3, seed=123)
        self.assertTrue(torch.equal(first.features, second.features))
        self.assertTrue(torch.equal(first.labels, second.labels))
        self.assertEqual(tuple(first.features.shape), (3, 4, 6))

    def test_model_returns_fixed_contract(self) -> None:
        from training.model import FixedShapeTransactionTransformer
        from training.sample import make_synthetic_transactions

        model = FixedShapeTransactionTransformer(self.config).eval()
        output = model(make_synthetic_transactions(self.config, batch_size=2).features)
        self.assertEqual(tuple(output.fraud_probability.shape), (2, 1))
        self.assertEqual(tuple(output.calibrated_confidence.shape), (2, 1))
        self.assertEqual(tuple(output.explanation.shape), (2, 3))
        self.assertTrue(torch.all((output.fraud_probability >= 0) & (output.fraud_probability <= 1)))

    def test_fp8_calibration_records_outliers(self) -> None:
        from training.calibration import FP8Format, calibrate_fp8
        from training.model import FixedShapeTransactionTransformer
        from training.sample import make_synthetic_transactions

        model = FixedShapeTransactionTransformer(self.config)
        batch = make_synthetic_transactions(self.config, batch_size=3).features
        batch[0, -1, 0] = 1_000.0
        result = calibrate_fp8(model, [batch], FP8Format.E4M3, percentile=0.95)
        self.assertEqual(result.format, "E4M3")
        self.assertTrue(result.tensors)
        self.assertTrue(any(item.outlier_count > 0 for item in result.tensors.values()))
        self.assertTrue(all(item.scale > 0 for item in result.tensors.values()))

    def test_metadata_is_fingerprinted_and_write_once(self) -> None:
        from training.bundle import ModelBundleMetadata, sha256_file

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            onnx = bundle / "model.onnx"
            state = bundle / "model.pt"
            onnx.write_bytes(b"onnx artifact")
            state.write_bytes(b"state artifact")
            hashes = {"model.onnx": sha256_file(onnx), "model.pt": sha256_file(state)}
            metadata = ModelBundleMetadata(
                "fraud",
                "test",
                self.config,
                artifact_sha256=hashes,
                fp8_calibration={"tensor": {"scale": 0.25}},
            )
            with self.assertRaises(TypeError):
                metadata.fp8_calibration["tensor"] = {}  # type: ignore[index]
            with self.assertRaises(TypeError):
                metadata.artifact_sha256["model.pt"] = "0" * 64  # type: ignore[index]

            path = metadata.write(bundle / "metadata.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["bundle_fingerprint"]), 64)
            self.assertEqual(payload["artifact_sha256"], hashes)
            self.assertEqual(hashes["model.onnx"], hashlib.sha256(b"onnx artifact").hexdigest())
            # Even a stale userspace existence check cannot bypass the OS-level
            # exclusive create used by ModelBundleMetadata.write().
            with patch.object(Path, "exists", return_value=False):
                with self.assertRaises(FileExistsError):
                    metadata.write(path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_bundle_fingerprint_commits_to_artifact_hashes(self) -> None:
        from training.bundle import ModelBundleMetadata

        first = ModelBundleMetadata(
            "fraud", "test", self.config, {"model.onnx": "1" * 64, "model.pt": "2" * 64}
        )
        changed = ModelBundleMetadata(
            "fraud", "test", self.config, {"model.onnx": "3" * 64, "model.pt": "2" * 64}
        )
        self.assertNotEqual(
            first.to_dict()["bundle_fingerprint"], changed.to_dict()["bundle_fingerprint"]
        )

    def test_failed_tensorrt_build_removes_partial_engine(self) -> None:
        from training.export import build_tensorrt_engine

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / "model.onnx"
            engine = root / "model.engine"
            onnx.write_bytes(b"model")

            def failed_build(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
                engine.write_bytes(b"partial")
                return subprocess.CompletedProcess([], returncode=1, stderr="unsupported GPU")

            with patch("training.export.export_model.shutil.which", return_value="trtexec"), patch(
                "training.export.export_model.subprocess.run", side_effect=failed_build
            ):
                result = build_tensorrt_engine(onnx, engine, fp8=True)
            self.assertFalse(result.available)
            self.assertIn("unsupported GPU", result.reason or "")
            self.assertFalse(engine.exists())

    def test_onnx_dependency_failure_is_clean(self) -> None:
        """An unavailable ONNX dependency must not leave a partial bundle."""
        import importlib.util

        if importlib.util.find_spec("onnx") is not None:
            self.skipTest("ONNX is installed; the success-path test covers export")
        from training.export import export_model_bundle
        from training.model import FixedShapeTransactionTransformer

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            with self.assertRaisesRegex(RuntimeError, "optional 'onnx' package"):
                export_model_bundle(FixedShapeTransactionTransformer(self.config), output)
            self.assertFalse(output.exists())

    def test_export_failure_recursively_removes_partial_bundle(self) -> None:
        from training.export import export_model_bundle
        from training.model import FixedShapeTransactionTransformer

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"

            def failed_export(*_: object, **__: object) -> None:
                nested = output / "exporter-temporary" / "partial.data"
                nested.parent.mkdir(parents=True)
                nested.write_bytes(b"partial")
                raise RuntimeError("simulated exporter failure")

            with patch(
                "training.export.export_model.importlib.util.find_spec", return_value=object()
            ), patch("training.export.export_model.torch.onnx.export", side_effect=failed_export):
                with self.assertRaisesRegex(RuntimeError, "simulated exporter failure"):
                    export_model_bundle(FixedShapeTransactionTransformer(self.config), output)
            self.assertFalse(output.exists())

    @unittest.skipUnless(__import__("importlib").util.find_spec("onnx"), "onnx is not installed")
    def test_onnx_bundle_export(self) -> None:
        from training.export import export_model_bundle
        from training.model import FixedShapeTransactionTransformer

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            export_model_bundle(FixedShapeTransactionTransformer(self.config), output)
            self.assertTrue((output / "model.onnx").is_file())
            self.assertTrue((output / "model.pt").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            from training.bundle import sha256_file

            self.assertEqual(metadata["artifact_sha256"]["model.onnx"], sha256_file(output / "model.onnx"))
            self.assertEqual(metadata["artifact_sha256"]["model.pt"], sha256_file(output / "model.pt"))


if __name__ == "__main__":
    unittest.main()
