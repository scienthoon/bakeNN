"""Framework-neutral immutable graph legalization and fusion passes."""

from .constants import ConstantChannel, analyze_constant_channels, deduplicate_constants
from .fuse import fuse_clamps
from .legalize import legalize_graph

__all__ = [
    "ConstantChannel",
    "analyze_constant_channels",
    "deduplicate_constants",
    "fuse_clamps",
    "legalize_graph",
]
