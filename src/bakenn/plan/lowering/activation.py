from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Callable

from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.activation import HardSigmoidOp, HardSwishOp, SiLUOp, SigmoidOp
from bakenn.ir.types import PerTensorQParams
from bakenn.plan.lower import lower_op
from bakenn.plan.steps.activation import LUTActivationStep


def _sigmoid(value: Decimal) -> Decimal:
    if value >= 0:
        return Decimal(1) / (Decimal(1) + (-value).exp())
    exponential = value.exp()
    return exponential / (Decimal(1) + exponential)


def _hardswish(value: Decimal) -> Decimal:
    gate = min(Decimal(6), max(Decimal(0), value + Decimal(3)))
    return value * gate / Decimal(6)


def _hardsigmoid(value: Decimal) -> Decimal:
    return min(Decimal(6), max(Decimal(0), value + Decimal(3))) / Decimal(6)


def _silu(value: Decimal) -> Decimal:
    return value * _sigmoid(value)


def _lut(
    input_qparams: PerTensorQParams,
    output_qparams: PerTensorQParams,
    function: Callable[[Decimal], Decimal],
) -> tuple[int, ...]:
    with localcontext() as context:
        context.prec = 80
        input_scale = Decimal.from_float(input_qparams.scale)
        output_scale = Decimal.from_float(output_qparams.scale)
        values: list[int] = []
        for code in range(-128, 128):
            real = input_scale * Decimal(code - input_qparams.zero_point)
            transformed = function(real)
            quantized = int(
                (transformed / output_scale).to_integral_value(rounding=ROUND_HALF_UP)
            ) + output_qparams.zero_point
            values.append(min(127, max(-128, quantized)))
        return tuple(values)


def _lower(
    op: SigmoidOp | HardSigmoidOp | HardSwishOp | SiLUOp,
    graph: QuantizedGraph,
    operation: str,
    function: Callable[[Decimal], Decimal],
) -> LUTActivationStep:
    input_qparams = graph.values[op.input].qparams
    output_qparams = graph.values[op.output].qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    return LUTActivationStep(
        name=op.name,
        input=op.input,
        output=op.output,
        operation=operation,
        lut=_lut(input_qparams, output_qparams, function),
    )


@lower_op.register
def _lower_sigmoid(op: SigmoidOp, graph: QuantizedGraph) -> LUTActivationStep:
    return _lower(op, graph, "sigmoid", _sigmoid)


@lower_op.register
def _lower_hardswish(op: HardSwishOp, graph: QuantizedGraph) -> LUTActivationStep:
    return _lower(op, graph, "hardswish", _hardswish)


@lower_op.register
def _lower_hardsigmoid(op: HardSigmoidOp, graph: QuantizedGraph) -> LUTActivationStep:
    return _lower(op, graph, "hardsigmoid", _hardsigmoid)


@lower_op.register
def _lower_silu(op: SiLUOp, graph: QuantizedGraph) -> LUTActivationStep:
    return _lower(op, graph, "silu", _silu)


__all__: list[str] = []
