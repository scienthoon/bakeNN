from __future__ import annotations

from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.sequence import AveragePool1DOp, Conv1DOp, MaxPool1DOp
from bakenn.ir.types import PerAxisQParams, PerTensorQParams
from bakenn.plan.lower import lower_op
from bakenn.plan.lowering.conv import _accumulator_bounds, _requantization
from bakenn.plan.steps.sequence import AveragePool1DStep, Conv1DStep, MaxPool1DStep


@lower_op.register
def _lower_conv1d(op: Conv1DOp, graph: QuantizedGraph) -> Conv1DStep:
    input_type, weight_type, output_type = (graph.values[name] for name in (op.input, op.weight, op.output))
    assert isinstance(input_type.qparams, PerTensorQParams)
    assert isinstance(weight_type.qparams, PerAxisQParams)
    assert isinstance(output_type.qparams, PerTensorQParams)
    weight, bias = graph.constants[op.weight], graph.constants[op.bias]
    bounds = _accumulator_bounds(input_type.qparams.zero_point, weight, bias, weight.shape[0], 0, op.name)
    multipliers, shifts = _requantization(op.name, input_type.qparams.scale, weight_type.qparams.scales, output_type.qparams.scale, bounds)
    return Conv1DStep(op.name, op.input, op.weight, op.bias, op.output, op.stride, op.dilation, op.padding, op.groups, multipliers, shifts, op.activation_min, op.activation_max, bounds)


@lower_op.register
def _lower_average_pool1d(op: AveragePool1DOp, graph: QuantizedGraph) -> AveragePool1DStep:
    qparams = graph.values[op.input].qparams
    assert isinstance(qparams, PerTensorQParams)
    max_abs = max(abs(-128 - qparams.zero_point), abs(127 - qparams.zero_point))
    return AveragePool1DStep(op.name, op.input, op.output, op.kernel, op.stride, op.padding, op.activation_min, op.activation_max, max_abs * op.kernel)


@lower_op.register
def _lower_max_pool1d(op: MaxPool1DOp, graph: QuantizedGraph) -> MaxPool1DStep:
    del graph
    return MaxPool1DStep(op.name, op.input, op.output, op.kernel, op.stride, op.padding, op.activation_min, op.activation_max, 0)


__all__: list[str] = []
