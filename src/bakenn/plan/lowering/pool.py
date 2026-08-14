from __future__ import annotations

from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.pool import AveragePool2DOp, MaxPool2DOp
from bakenn.ir.types import PerTensorQParams
from bakenn.plan.lower import lower_op
from bakenn.plan.steps.pool import AveragePool2DStep, MaxPool2DStep


@lower_op.register
def _lower_average_pool(op: AveragePool2DOp, graph: QuantizedGraph) -> AveragePool2DStep:
    input_type = graph.values[op.input]
    assert isinstance(input_type.qparams, PerTensorQParams)
    max_abs_centered = max(
        abs(-128 - input_type.qparams.zero_point),
        abs(127 - input_type.qparams.zero_point),
    )
    return AveragePool2DStep(
        name=op.name,
        input=op.input,
        output=op.output,
        kernel=op.kernel,
        stride=op.stride,
        padding=op.padding,
        activation_min=op.activation_min,
        activation_max=op.activation_max,
        accumulator_bound=max_abs_centered * op.kernel[0] * op.kernel[1],
    )


@lower_op.register
def _lower_max_pool(op: MaxPool2DOp, graph: QuantizedGraph) -> MaxPool2DStep:
    return MaxPool2DStep(
        name=op.name,
        input=op.input,
        output=op.output,
        kernel=op.kernel,
        stride=op.stride,
        padding=op.padding,
        activation_min=op.activation_min,
        activation_max=op.activation_max,
    )


__all__: list[str] = []
