"""Immutable metadata contract for published offline model bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .model import TransactionTransformerConfig


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading an artifact into memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_json(value: Any) -> Any:
    """Recursively prevent mutation of JSON-compatible bundle provenance."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Convert frozen provenance back to standard JSON-compatible objects."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelBundleMetadata:
    """Provenance and fixed-shape contract accompanying a model artifact."""

    model_name: str
    model_version: str
    model_config: TransactionTransformerConfig
    artifact_sha256: Mapping[str, str]
    input_name: str = "transaction_history"
    input_dtype: str = "float32"
    fp8_calibration: Mapping[str, Any] | None = None
    onnx_file: str = "model.onnx"
    state_dict_file: str = "model.pt"

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_version:
            raise ValueError("model_name and model_version must be non-empty")
        required_artifacts = {self.onnx_file, self.state_dict_file}
        missing = required_artifacts.difference(self.artifact_sha256)
        if missing:
            raise ValueError(f"missing SHA-256 digest for artifacts: {sorted(missing)}")
        normalized_hashes: dict[str, str] = {}
        for name, digest in self.artifact_sha256.items():
            normalized = digest.lower()
            if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
                raise ValueError(f"invalid SHA-256 digest for artifact: {name}")
            normalized_hashes[str(name)] = normalized
        object.__setattr__(self, "artifact_sha256", MappingProxyType(normalized_hashes))
        if self.fp8_calibration is not None:
            object.__setattr__(self, "fp8_calibration", _freeze_json(self.fp8_calibration))

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible metadata including a fingerprint."""
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_config": asdict(self.model_config),
            "artifact_sha256": dict(sorted(self.artifact_sha256.items())),
            "input_name": self.input_name,
            "input_dtype": self.input_dtype,
            "fp8_calibration": _thaw_json(self.fp8_calibration),
            "onnx_file": self.onnx_file,
            "state_dict_file": self.state_dict_file,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["bundle_fingerprint"] = hashlib.sha256(encoded).hexdigest()
        return payload

    def write(self, path: str | Path) -> Path:
        """Create metadata exclusively so concurrent publishers cannot replace it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as metadata_file:
            json.dump(self.to_dict(), metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        return target
