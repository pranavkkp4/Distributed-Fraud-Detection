"""ONNX and optional TensorRT bundle export utilities."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .export_model import TensorRTBuildResult, build_tensorrt_engine, export_model_bundle

__all__ = ["TensorRTBuildResult", "export_model_bundle", "build_tensorrt_engine"]


def __getattr__(name: str) -> Any:
    """Defer the heavyweight exporter import, including for ``python -m``."""
    if name in __all__:
        from . import export_model

        return getattr(export_model, name)
    raise AttributeError(name)
