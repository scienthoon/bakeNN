from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class SoftmaxOp:
    name: str
    input: str
    output: str
    beta: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.beta, bool) or not isinstance(self.beta, Real):
            raise ValueError("Softmax beta must be a real number")
        beta = float(self.beta)
        if not math.isfinite(beta) or beta != 1.0:
            raise ValueError("P0 Softmax supports beta=1.0 exactly")
        object.__setattr__(self, "beta", beta)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


__all__ = ["SoftmaxOp"]
