class BakeNNError(Exception):
    """Base class for deterministic user-facing compiler failures."""


class GraphValidationError(BakeNNError):
    """The quantized graph violates the BakeNN IR contract."""


class CompileError(BakeNNError):
    """The graph is valid but cannot be lowered safely for a backend."""
