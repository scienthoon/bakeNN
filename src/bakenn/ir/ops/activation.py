from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SigmoidOp:
    name: str
    input: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class HardSwishOp:
    name: str
    input: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class HardSigmoidOp:
    name: str
    input: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class SiLUOp:
    name: str
    input: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


__all__ = ["HardSigmoidOp", "HardSwishOp", "SiLUOp", "SigmoidOp"]
