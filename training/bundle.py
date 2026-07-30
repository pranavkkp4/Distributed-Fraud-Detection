"""Immutable metadata contract for published offline model bundles."""

from __future__ import annotations

import hashlib
import hmac
import json
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from .model import TransactionTransformerConfig


BUNDLE_SCHEMA_VERSION = 1


class BundleVerificationError(ValueError):
    """Raised when an immutable model bundle fails integrity validation."""


def _is_link_or_reparse_point(path: Path) -> bool:
    """Reject links plus Windows junctions and other filesystem reparse points."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(reparse_flag and attributes & reparse_flag)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading an artifact into memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(name: str, digest: object) -> str:
    if not isinstance(digest, str):
        raise ValueError(f"SHA-256 digest for artifact {name!r} must be a string")
    normalized = digest.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"invalid SHA-256 digest for artifact: {name}")
    return normalized


def _validate_artifact_name(name: object) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"invalid bundle artifact path: {name!r}")
    artifact = PurePosixPath(name)
    if artifact.is_absolute() or artifact.as_posix() != name or any(
        part in {"", ".", ".."} for part in artifact.parts
    ):
        raise ValueError(f"invalid bundle artifact path: {name!r}")
    if name == "metadata.json":
        raise ValueError("metadata.json cannot hash itself as a bundle artifact")
    return name


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    schema_version: int = BUNDLE_SCHEMA_VERSION
    input_name: str = "transaction_history"
    input_dtype: str = "float32"
    export_device: str = "cpu"
    export_dtype: str = "float32"
    fp8_calibration: Mapping[str, Any] | None = None
    onnx_file: str = "model.onnx"
    state_dict_file: str = "model.pt"

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_version:
            raise ValueError("model_name and model_version must be non-empty")
        if self.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported bundle schema version: {self.schema_version}")
        if (
            self.input_dtype != "float32"
            or self.export_device != "cpu"
            or self.export_dtype != "float32"
        ):
            raise ValueError("this bundle schema currently supports CPU float32 export only")
        required_artifacts = {self.onnx_file, self.state_dict_file}
        missing = required_artifacts.difference(self.artifact_sha256)
        if missing:
            raise ValueError(f"missing SHA-256 digest for artifacts: {sorted(missing)}")
        normalized_hashes: dict[str, str] = {}
        for name, digest in self.artifact_sha256.items():
            normalized_name = _validate_artifact_name(name)
            normalized_hashes[normalized_name] = _validate_sha256(normalized_name, digest)
        object.__setattr__(self, "artifact_sha256", MappingProxyType(normalized_hashes))
        if self.fp8_calibration is not None:
            object.__setattr__(self, "fp8_calibration", _freeze_json(self.fp8_calibration))

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible metadata including a fingerprint."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_config": asdict(self.model_config),
            "artifact_sha256": dict(sorted(self.artifact_sha256.items())),
            "input_name": self.input_name,
            "input_dtype": self.input_dtype,
            "export_device": self.export_device,
            "export_dtype": self.export_dtype,
            "fp8_calibration": _thaw_json(self.fp8_calibration),
            "onnx_file": self.onnx_file,
            "state_dict_file": self.state_dict_file,
        }
        payload["bundle_fingerprint"] = _fingerprint(payload)
        return payload

    def write(self, path: str | Path) -> Path:
        """Create metadata exclusively so concurrent publishers cannot replace it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as metadata_file:
            json.dump(self.to_dict(), metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        return target


def verify_model_bundle(path: str | Path) -> dict[str, Any]:
    """Verify metadata fingerprint, artifact hashes, and the exact file set.

    Symlinks and unlisted files are rejected so verification cannot be made to
    depend on mutable data outside the published directory.
    """
    root = Path(path)
    if not root.is_dir() or _is_link_or_reparse_point(root):
        raise BundleVerificationError(f"bundle is not a regular directory: {root}")
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file() or _is_link_or_reparse_point(metadata_path):
        raise BundleVerificationError("bundle metadata.json is missing or is not a regular file")
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"cannot read bundle metadata: {exc}") from exc
    if not isinstance(loaded, dict):
        raise BundleVerificationError("bundle metadata must be a JSON object")

    claimed_fingerprint = loaded.get("bundle_fingerprint")
    if not isinstance(claimed_fingerprint, str):
        raise BundleVerificationError("bundle fingerprint is missing")
    fingerprint_payload = dict(loaded)
    del fingerprint_payload["bundle_fingerprint"]
    actual_fingerprint = _fingerprint(fingerprint_payload)
    if not hmac.compare_digest(claimed_fingerprint, actual_fingerprint):
        raise BundleVerificationError("bundle metadata fingerprint mismatch")

    if loaded.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleVerificationError(
            f"unsupported bundle schema version: {loaded.get('schema_version')!r}"
        )
    if (
        loaded.get("input_dtype") != "float32"
        or loaded.get("export_device") != "cpu"
        or loaded.get("export_dtype") != "float32"
    ):
        raise BundleVerificationError("bundle export must declare CPU float32 artifacts")
    artifact_hashes = loaded.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise BundleVerificationError("artifact_sha256 must be a non-empty object")
    try:
        normalized_hashes = {
            _validate_artifact_name(name): _validate_sha256(str(name), digest)
            for name, digest in artifact_hashes.items()
        }
    except ValueError as exc:
        raise BundleVerificationError(str(exc)) from exc
    for metadata_key in ("onnx_file", "state_dict_file"):
        artifact_name = loaded.get(metadata_key)
        if artifact_name not in normalized_hashes:
            raise BundleVerificationError(f"{metadata_key} is not covered by artifact_sha256")

    expected_files = {"metadata.json", *normalized_hashes}
    expected_directories: set[str] = set()
    for name in normalized_hashes:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if _is_link_or_reparse_point(entry):
            raise BundleVerificationError(
                f"bundle contains a symlink or filesystem reparse point: {relative}"
            )
        if entry.is_file():
            actual_files.add(relative)
        elif entry.is_dir():
            actual_directories.add(relative)
        else:
            raise BundleVerificationError(f"bundle contains an unsupported entry: {relative}")
    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files
    extra_directories = actual_directories - expected_directories
    if missing_files:
        raise BundleVerificationError(f"bundle artifacts are missing: {sorted(missing_files)}")
    if extra_files or extra_directories:
        extras = sorted(extra_files | extra_directories)
        raise BundleVerificationError(f"bundle contains unlisted entries: {extras}")

    for name, expected_digest in normalized_hashes.items():
        artifact_path = root.joinpath(*PurePosixPath(name).parts)
        try:
            actual_digest = sha256_file(artifact_path)
        except OSError as exc:
            raise BundleVerificationError(f"cannot hash bundle artifact {name!r}: {exc}") from exc
        if not hmac.compare_digest(expected_digest, actual_digest):
            raise BundleVerificationError(f"artifact SHA-256 mismatch: {name}")
    return loaded
