from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.tensor import Pad2DStep, ReduceMeanStep
from bakenn.plan.types import ExecutionPlan
from bakenn.quantization.fixedpoint import multiply_by_quantized_multiplier
from bakenn.reference.executor import execute_step


@execute_step.register
def _execute_pad2d(step: Pad2DStep, plan: ExecutionPlan, values: Mapping[str, np.ndarray]):  # type: ignore[no-untyped-def]
    input_qparams = plan.tensors[step.input].tensor_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    top, bottom, left, right = step.padding
    result = np.pad(
        values[step.input],
        ((0, 0), (top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=input_qparams.zero_point,
    )
    return {step.output: np.ascontiguousarray(result, dtype=np.int8)}


@execute_step.register
def _execute_reduce_mean(step: ReduceMeanStep, plan: ExecutionPlan, values: Mapping[str, np.ndarray]):  # type: ignore[no-untyped-def]
    input_qparams = plan.tensors[step.input].tensor_type.qparams
    output_qparams = plan.tensors[step.output].tensor_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    flat = values[step.input].reshape(step.position_count, step.channels)
    result = np.empty((step.channels,), dtype=np.int8)
    for channel in range(step.channels):
        centered_sum = sum(int(value) - input_qparams.zero_point for value in flat[:, channel])
        scaled = multiply_by_quantized_multiplier(centered_sum, step.multiplier, step.shift)
        result[channel] = np.int8(min(127, max(-128, scaled + output_qparams.zero_point)))
    return {step.output: result.reshape(plan.tensors[step.output].tensor_type.shape)}


__all__: list[str] = []
