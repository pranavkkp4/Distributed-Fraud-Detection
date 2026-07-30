"""Unit coverage for offline model preparation contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
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

    def test_model_returns_honest_fixed_contract(self) -> None:
        from training.model import FixedShapeTransactionTransformer
        from training.sample import make_synthetic_transactions

        model = FixedShapeTransactionTransformer(self.config).eval()
        output = model(make_synthetic_transactions(self.config, batch_size=2).features)
        self.assertEqual(tuple(output.fraud_probability.shape), (2, 1))
        self.assertEqual(tuple(output.confidence_score.shape), (2, 1))
        self.assertEqual(tuple(output.explanation.shape), (2, 3))
        self.assertTrue(
            torch.all((output.fraud_probability >= 0) & (output.fraud_probability <= 1))
        )

    def test_fp8_calibration_emits_explicit_qdq_and_outlier_policy(self) -> None:
        from training.calibration import FP8Format, calibrate_fp8
        from training.model import FixedShapeTransactionTransformer
        from training.sample import make_synthetic_transactions

        model = FixedShapeTransactionTransformer(self.config).train()
        batch = make_synthetic_transactions(self.config, batch_size=3).features
        batch[0, -1, 0] = 1_000.0
        original = batch.clone()
        clipped = calibrate_fp8(model, [batch], FP8Format.E4M3, percentile=0.95)
        self.assertEqual(clipped.schema_version, 1)
        self.assertEqual(clipped.format, "E4M3")
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(batch, original), "calibration analysis must not mutate inputs")
        self.assertTrue(clipped.tensors)
        self.assertTrue(any(item.outlier_count > 0 for item in clipped.tensors.values()))
        for item in clipped.tensors.values():
            self.assertGreater(item.quant_scale, 0.0)
            self.assertGreater(item.dequant_scale, 0.0)
            self.assertAlmostEqual(item.quant_scale * item.dequant_scale, 1.0, places=5)
            self.assertTrue(item.eligible_for_fp8)

        fallback = calibrate_fp8(
            model, [batch], FP8Format.E4M3, percentile=0.95, outlier_policy="fallback"
        )
        ineligible = [item for item in fallback.tensors.values() if item.outlier_count]
        self.assertTrue(ineligible)
        self.assertTrue(all(not item.eligible_for_fp8 for item in ineligible))
        self.assertTrue(all(item.dequant_scale > 0.0 for item in ineligible))
        self.assertTrue(all(item.recommended_action == "use_fp16_or_fp32" for item in ineligible))
        serialized = fallback.to_dict()
        self.assertEqual(serialized["analysis_behavior"], "observes_activations_without_mutation")
        self.assertIn("runtime Q/DQ", str(serialized["runtime_clip_contract"]))

    def test_fp8_calibration_accepts_half_precision_activations(self) -> None:
        from training.calibration import calibrate_fp8

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = torch.nn.Sequential(torch.nn.Linear(4, 4)).to(dtype=dtype)
                batch = torch.randn(3, 4, dtype=dtype)
                result = calibrate_fp8(model, [batch])
                self.assertIn("0", result.tensors)
                self.assertIsInstance(result.tensors["0"].amax, float)

    def test_fp8_calibration_rejects_empty_and_nonfinite_data(self) -> None:
        from training.calibration import calibrate_fp8

        model = torch.nn.Sequential(torch.nn.Linear(4, 4)).train()
        with self.assertRaisesRegex(ValueError, "at least one calibration batch"):
            calibrate_fp8(model, [])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            calibrate_fp8(model, [torch.empty(0, 4)])
        nonfinite = torch.zeros(2, 4)
        nonfinite[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "only finite"):
            calibrate_fp8(model, [nonfinite])
        self.assertTrue(model.training)

        activation_model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with torch.no_grad():
            activation_model[0].weight[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite activations"):
            calibrate_fp8(activation_model, [torch.ones(2, 4)])

    def _write_test_bundle(self, root: Path) -> tuple[object, dict[str, str]]:
        from training.bundle import ModelBundleMetadata, sha256_file

        (root / "model.onnx").write_bytes(b"onnx artifact")
        (root / "model.pt").write_bytes(b"state artifact")
        hashes = {
            "model.onnx": sha256_file(root / "model.onnx"),
            "model.pt": sha256_file(root / "model.pt"),
        }
        metadata = ModelBundleMetadata(
            "fraud",
            "test",
            self.config,
            artifact_sha256=hashes,
            fp8_calibration={"schema_version": 1, "tensor": {"dequant_scale": 0.25}},
        )
        metadata.write(root / "metadata.json")
        return metadata, hashes

    def test_metadata_is_fingerprinted_immutable_and_exclusive(self) -> None:
        from training.bundle import verify_model_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, hashes = self._write_test_bundle(root)
            with self.assertRaises(TypeError):
                metadata.fp8_calibration["tensor"] = {}  # type: ignore[index]
            with self.assertRaises(TypeError):
                metadata.artifact_sha256["model.pt"] = "0" * 64  # type: ignore[index]

            path = root / "metadata.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["bundle_fingerprint"]), 64)
            self.assertEqual(payload["artifact_sha256"], hashes)
            self.assertEqual(hashes["model.onnx"], hashlib.sha256(b"onnx artifact").hexdigest())
            with patch.object(Path, "exists", return_value=False):
                with self.assertRaises(FileExistsError):
                    metadata.write(path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            self.assertEqual(verify_model_bundle(root), payload)

    def test_bundle_verifier_rejects_tampering_missing_and_extra_files(self) -> None:
        from training.bundle import BundleVerificationError, verify_model_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_test_bundle(root)
            model_path = root / "model.onnx"
            model_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(BundleVerificationError, "SHA-256 mismatch"):
                verify_model_bundle(root)
            model_path.write_bytes(b"onnx artifact")

            extra = root / "unlisted.bin"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(BundleVerificationError, "unlisted"):
                verify_model_bundle(root)
            extra.unlink()

            (root / "model.pt").unlink()
            with self.assertRaisesRegex(BundleVerificationError, "missing"):
                verify_model_bundle(root)

    def test_bundle_verifier_rejects_metadata_fingerprint_tampering(self) -> None:
        from training.bundle import BundleVerificationError, verify_model_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_test_bundle(root)
            metadata_path = root / "metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["model_version"] = "tampered"
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(BundleVerificationError, "fingerprint mismatch"):
                verify_model_bundle(root)

    @unittest.skipUnless(os.name == "nt", "NTFS junction regression is Windows-specific")
    def test_bundle_verifier_rejects_root_and_nested_junctions(self) -> None:
        from training.bundle import (
            BundleVerificationError,
            ModelBundleMetadata,
            sha256_file,
            verify_model_bundle,
        )

        def junction(link: Path, target: Path) -> None:
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                self.skipTest(f"cannot create an NTFS junction: {completed.stderr}")

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            external_root = parent / "external-root"
            external_root.mkdir()
            self._write_test_bundle(external_root)
            root_junction = parent / "root-junction"
            junction(root_junction, external_root)
            try:
                with self.assertRaisesRegex(BundleVerificationError, "regular directory"):
                    verify_model_bundle(root_junction)
            finally:
                root_junction.rmdir()

            bundle = parent / "bundle"
            outside = parent / "outside"
            bundle.mkdir()
            outside.mkdir()
            (outside / "model.onnx").write_bytes(b"onnx artifact")
            (outside / "model.pt").write_bytes(b"state artifact")
            hashes = {
                "nested/model.onnx": sha256_file(outside / "model.onnx"),
                "nested/model.pt": sha256_file(outside / "model.pt"),
            }
            metadata = ModelBundleMetadata(
                "fraud",
                "test",
                self.config,
                artifact_sha256=hashes,
                onnx_file="nested/model.onnx",
                state_dict_file="nested/model.pt",
            )
            metadata.write(bundle / "metadata.json")
            nested_junction = bundle / "nested"
            junction(nested_junction, outside)
            try:
                with self.assertRaisesRegex(BundleVerificationError, "reparse point"):
                    verify_model_bundle(bundle)
            finally:
                nested_junction.rmdir()

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

    def test_atomic_directory_publish_has_one_concurrent_winner(self) -> None:
        from training.export.export_model import _atomic_publish_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destinations = [root / "stage-one", root / "stage-two"]
            for index, staging in enumerate(destinations):
                staging.mkdir()
                (staging / "winner.txt").write_text(str(index), encoding="utf-8")
            published = root / "bundle"
            barrier = threading.Barrier(2)

            def attempt(staging: Path) -> bool:
                barrier.wait()
                try:
                    _atomic_publish_directory(staging, published)
                except FileExistsError:
                    return False
                return True

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(attempt, destinations))
            self.assertEqual(sum(results), 1)
            self.assertIn((published / "winner.txt").read_text(encoding="utf-8"), {"0", "1"})

    @unittest.skipUnless(
        os.name == "nt" or sys.platform.startswith("linux"),
        "atomic no-replace publication is supported on Windows and Linux",
    )
    def test_atomic_publish_race_cannot_replace_empty_destination(self) -> None:
        from training.export.export_model import _atomic_publish_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            destination = root / "bundle"
            staging.mkdir()
            destination.mkdir()
            (staging / "candidate.txt").write_text("candidate", encoding="utf-8")
            # Simulate the destination appearing after the optimistic preflight
            # check but before the atomic publication primitive executes.
            with patch.object(Path, "exists", return_value=False):
                with self.assertRaises(FileExistsError):
                    _atomic_publish_directory(staging, destination)
            self.assertTrue(staging.is_dir())
            self.assertTrue(destination.is_dir())
            self.assertFalse((destination / "candidate.txt").exists())

    def test_failed_tensorrt_build_is_bounded_and_removes_partial_engine(self) -> None:
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
            ) as run:
                result = build_tensorrt_engine(onnx, engine, fp8=True, timeout_seconds=0.25)
            command = run.call_args.args[0]
            self.assertIn("--fp8", command)
            self.assertNotIn("--explicitBatch", command)
            self.assertEqual(run.call_args.kwargs["timeout"], 0.25)
            self.assertFalse(result.available)
            self.assertIn("exited with code 1", result.reason or "")
            self.assertIn("unsupported GPU", result.reason or "")
            self.assertFalse(engine.exists())

    def test_tensorrt_timeout_and_empty_engine_are_failures(self) -> None:
        from training.export import build_tensorrt_engine

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / "model.onnx"
            engine = root / "model.engine"
            onnx.write_bytes(b"model")

            def timeout(*_: object, **__: object) -> None:
                engine.write_bytes(b"partial")
                raise subprocess.TimeoutExpired("trtexec", timeout=0.1, stderr="still building")

            with patch("training.export.export_model.shutil.which", return_value="trtexec"), patch(
                "training.export.export_model.subprocess.run", side_effect=timeout
            ):
                result = build_tensorrt_engine(onnx, engine, timeout_seconds=0.1)
            self.assertIn("timed out", result.reason or "")
            self.assertFalse(engine.exists())

            def empty_build(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
                engine.touch()
                return subprocess.CompletedProcess([], returncode=0, stdout="done")

            with patch("training.export.export_model.shutil.which", return_value="trtexec"), patch(
                "training.export.export_model.subprocess.run", side_effect=empty_build
            ):
                result = build_tensorrt_engine(onnx, engine)
            self.assertIn("missing or empty", result.reason or "")
            self.assertFalse(engine.exists())

    def test_export_requires_cpu_float32(self) -> None:
        from training.export import export_model_bundle
        from training.model import FixedShapeTransactionTransformer

        with tempfile.TemporaryDirectory() as directory:
            model = FixedShapeTransactionTransformer(self.config).to(dtype=torch.float64)
            with self.assertRaisesRegex(ValueError, "CPU float32"):
                export_model_bundle(model, Path(directory) / "bundle")

    def test_onnx_dependency_failure_is_clean(self) -> None:
        from training.export import export_model_bundle
        from training.model import FixedShapeTransactionTransformer

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            with patch(
                "training.export.export_model.importlib.util.find_spec",
                side_effect=lambda package: None if package == "onnxscript" else object(),
            ):
                with self.assertRaisesRegex(RuntimeError, "'onnxscript'"):
                    export_model_bundle(FixedShapeTransactionTransformer(self.config), output)
            self.assertFalse(output.exists())

    def test_export_failure_cleans_staging_and_restores_training_mode(self) -> None:
        from training.export import export_model_bundle
        from training.model import FixedShapeTransactionTransformer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            model = FixedShapeTransactionTransformer(self.config).train()

            def failed_export(_: object, __: object, onnx_path: object, **___: object) -> None:
                staging = Path(onnx_path).parent
                nested = staging / "exporter-temporary" / "partial.data"
                nested.parent.mkdir(parents=True)
                nested.write_bytes(b"partial")
                raise RuntimeError("simulated exporter failure")

            with patch(
                "training.export.export_model.importlib.util.find_spec", return_value=object()
            ), patch("training.export.export_model.torch.onnx.export", side_effect=failed_export):
                with self.assertRaisesRegex(RuntimeError, "simulated exporter failure"):
                    export_model_bundle(model, output)
            self.assertTrue(model.training)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".bundle.staging-*")), [])

    def test_final_destination_race_preserves_winner_and_cleans_staging(self) -> None:
        from training.export import export_model_bundle
        from training.model import FixedShapeTransactionTransformer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            model = FixedShapeTransactionTransformer(self.config).train()

            def fake_export(_: object, __: object, onnx_path: object, **___: object) -> None:
                Path(onnx_path).write_bytes(b"mock onnx")

            def lose_race(_: Path, destination: Path) -> None:
                destination.mkdir()
                (destination / "winner.txt").write_text("winner", encoding="utf-8")
                raise FileExistsError("simulated final-destination race")

            with patch(
                "training.export.export_model.importlib.util.find_spec", return_value=object()
            ), patch(
                "training.export.export_model.torch.onnx.export", side_effect=fake_export
            ), patch(
                "training.export.export_model._validate_onnx_contract"
            ), patch(
                "training.export.export_model._atomic_publish_directory", side_effect=lose_race
            ):
                with self.assertRaisesRegex(FileExistsError, "final-destination race"):
                    export_model_bundle(model, output)
            self.assertTrue(model.training)
            self.assertEqual((output / "winner.txt").read_text(encoding="utf-8"), "winner")
            self.assertEqual(list(root.glob(".bundle.staging-*")), [])

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(package)
            for package in ("onnx", "onnxruntime", "onnxscript")
        ),
        "ONNX export and runtime packages are not installed",
    )
    def test_onnx_bundle_export_has_checked_static_contract(self) -> None:
        import onnx

        from training.bundle import sha256_file, verify_model_bundle
        from training.export import export_model_bundle
        from training.model import FixedShapeTransactionTransformer

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            model = FixedShapeTransactionTransformer(self.config).train()
            export_model_bundle(model, output)
            self.assertTrue(model.training)
            self.assertEqual(
                {"metadata.json", "model.onnx", "model.pt"},
                {artifact.name for artifact in output.iterdir()},
            )
            self.assertEqual(list(Path(directory).glob(".bundle.staging-*")), [])
            metadata = verify_model_bundle(output)
            self.assertEqual(metadata["export_device"], "cpu")
            self.assertEqual(metadata["export_dtype"], "float32")
            self.assertEqual(
                metadata["artifact_sha256"]["model.onnx"], sha256_file(output / "model.onnx")
            )

            graph = onnx.load_model(output / "model.onnx", load_external_data=False).graph
            def signature(value: object) -> tuple[str, int, tuple[int, ...]]:
                tensor_type = value.type.tensor_type  # type: ignore[attr-defined]
                return (
                    value.name,  # type: ignore[attr-defined]
                    int(tensor_type.elem_type),
                    tuple(int(dimension.dim_value) for dimension in tensor_type.shape.dim),
                )

            self.assertEqual(
                [signature(value) for value in graph.input],
                [("transaction_history", onnx.TensorProto.FLOAT, (1, 4, 6))],
            )
            self.assertEqual(
                [signature(value) for value in graph.output],
                [
                    ("fraud_probability", onnx.TensorProto.FLOAT, (1, 1)),
                    ("confidence_score", onnx.TensorProto.FLOAT, (1, 1)),
                    ("explanation", onnx.TensorProto.FLOAT, (1, 3)),
                ],
            )
            self.assertTrue(all(not item.external_data for item in graph.initializer))

            import numpy as np
            import onnxruntime

            from training.sample import make_synthetic_transactions

            sample = make_synthetic_transactions(self.config, batch_size=1, seed=99).features
            model.eval()
            with torch.inference_mode():
                expected = model(sample)
            session = onnxruntime.InferenceSession(
                str(output / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            actual = session.run(None, {"transaction_history": sample.numpy()})
            for observed, reference in zip(
                actual,
                (
                    expected.fraud_probability.numpy(),
                    expected.confidence_score.numpy(),
                    expected.explanation.numpy(),
                ),
                strict=True,
            ):
                np.testing.assert_allclose(observed, reference, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
