"""Immutable, machine-readable PTQ calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bakenn.ir.graph import QuantizedGraph


CALIBRATION_REPORT_SCHEMA = "bakenn.calibration-report.v1"
OBSERVER_PROFILE = "bakenn.minmax.fp32.v1"


@dataclass(frozen=True)
class CalibrationEdgeReport:
    """Observed FP32 range and the INT8 affine domain chosen for one edge."""

    name: str
    element_count: int
    minimum: float
    maximum: float
    scale: float
    zero_point: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("calibration edge name must be non-empty")
        if not isinstance(self.element_count, int) or self.element_count <= 0:
            raise ValueError("calibration edge element_count must be positive")
        if not all(
            math.isfinite(value)
            for value in (self.minimum, self.maximum, self.scale)
        ):
            raise ValueError("calibration edge values must be finite")
        if self.minimum > self.maximum:
            raise ValueError("calibration edge minimum exceeds maximum")
        if self.scale <= 0.0:
            raise ValueError("calibration edge scale must be positive")
        if not isinstance(self.zero_point, int) or isinstance(self.zero_point, bool):
            raise ValueError("calibration edge zero_point must be an integer")
        if not -128 <= self.zero_point <= 127:
            raise ValueError("calibration edge zero_point must fit INT8")

    @property
    def representable_minimum(self) -> float:
        return self.scale * (-128 - self.zero_point)

    @property
    def representable_maximum(self) -> float:
        return self.scale * (127 - self.zero_point)

    @property
    def is_degenerate(self) -> bool:
        return self.minimum == self.maximum

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "element_count": self.element_count,
            "observed": {"minimum": self.minimum, "maximum": self.maximum},
            "qparams": {
                "dtype": "int8",
                "scale": self.scale,
                "zero_point": self.zero_point,
            },
            "representable": {
                "minimum": self.representable_minimum,
                "maximum": self.representable_maximum,
            },
            "degenerate": self.is_degenerate,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Deterministic summary of the representative data consumed by PTQ."""

    graph_name: str
    sample_count: int
    input_shape: tuple[int, ...]
    edges: tuple[CalibrationEdgeReport, ...]
    schema: str = CALIBRATION_REPORT_SCHEMA
    observer_profile: str = OBSERVER_PROFILE

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "edges", tuple(self.edges))
        if not isinstance(self.graph_name, str) or not self.graph_name:
            raise ValueError("calibration graph_name must be non-empty")
        if not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ValueError("calibration sample_count must be positive")
        if not self.input_shape or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in self.input_shape
        ):
            raise ValueError("calibration input_shape must be positive and static")
        if not self.edges:
            raise ValueError("calibration report must contain at least one edge")
        if len({edge.name for edge in self.edges}) != len(self.edges):
            raise ValueError("calibration report edge names must be unique")
        if self.schema != CALIBRATION_REPORT_SCHEMA:
            raise ValueError("unsupported calibration report schema")
        if self.observer_profile != OBSERVER_PROFILE:
            raise ValueError("unsupported calibration observer profile")

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            f"{edge.name}: observed range is degenerate"
            for edge in self.edges
            if edge.is_degenerate
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "observer_profile": self.observer_profile,
            "graph_name": self.graph_name,
            "sample_count": self.sample_count,
            "input_shape": list(self.input_shape),
            "edge_count": len(self.edges),
            "edges": [edge.to_dict() for edge in self.edges],
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False
        ) + "\n"

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination


@dataclass(frozen=True)
class PTQResult:
    """Quantized graph paired with the calibration evidence that created it."""

    graph: "QuantizedGraph"
    report: CalibrationReport

    def __post_init__(self) -> None:
        from bakenn.ir.graph import QuantizedGraph

        if not isinstance(self.graph, QuantizedGraph):
            raise TypeError("PTQResult graph must be a QuantizedGraph")
        if not isinstance(self.report, CalibrationReport):
            raise TypeError("PTQResult report must be a CalibrationReport")
        if self.graph.name != self.report.graph_name:
            raise ValueError("PTQResult graph and report names must match")


__all__ = [
    "CALIBRATION_REPORT_SCHEMA",
    "OBSERVER_PROFILE",
    "CalibrationEdgeReport",
    "CalibrationReport",
    "PTQResult",
]
