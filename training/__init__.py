"""Offline preparation tools for the fixed-shape fraud scoring model.

The serving plane consumes the immutable bundles emitted by this package; it
does not import this package at request time.
"""

from .bundle import BundleVerificationError, ModelBundleMetadata, verify_model_bundle

__all__ = ["BundleVerificationError", "ModelBundleMetadata", "verify_model_bundle"]
