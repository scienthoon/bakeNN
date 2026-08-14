from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .op import LinearOp, Op
from .types import TensorType


@dataclass(frozen=True)
class QuantizedGraph:
    name: str
    values: Mapping[str, TensorType]
    constants: Mapping[str, np.ndarray]
    ops: tuple[Op, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    arithmetic_profile: str = "bakenn.int8.v1"

    def __post_init__(self) -> None:
        values = MappingProxyType(dict(self.values))
        constants: dict[str, np.ndarray] = {}
        for name, value in self.constants.items():
            frozen = np.array(value, copy=True, order="C")
            frozen.setflags(write=False)
            constants[name] = frozen
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "constants", MappingProxyType(constants))
        object.__setattr__(self, "ops", tuple(self.ops))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))


def verify_graph(graph: QuantizedGraph) -> None:
    """Compatibility import; the implementation lives in :mod:`bakenn.ir.verify`."""

    from .verify import verify_graph as _verify_graph

    _verify_graph(graph)


__all__ = ["LinearOp", "Op", "QuantizedGraph", "verify_graph"]
