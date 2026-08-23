"""Optional host-only frontend for strict fully-quantized TFLite models."""

from .importer import import_tflite

__all__ = ["import_tflite"]
