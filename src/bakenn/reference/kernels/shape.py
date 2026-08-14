from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.plan.types import ExecutionPlan
from bakenn.plan.steps.shape import ConcatenateStep, FlattenStep, ReshapeStep, SliceStep
from bakenn.reference.executor import execute_step


def _execute_view(
    step: ReshapeStep | FlattenStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    output_shape = plan.tensors[step.output].tensor_type.shape
    return {step.output: values[step.input].reshape(output_shape)}


@execute_step.register
def _execute_reshape(
    step: ReshapeStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    return _execute_view(step, plan, values)


@execute_step.register
def _execute_flatten(
    step: FlattenStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    return _execute_view(step, plan, values)


@execute_step.register
def _execute_slice(
    step: SliceStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    selection = [slice(None)] * values[step.input].ndim
    stop = step.start + step.step * step.output_axis_size
    selection[step.axis] = slice(step.start, stop, step.step)
    result = values[step.input][tuple(selection)]
    return {step.output: np.ascontiguousarray(result, dtype=np.int8)}


@execute_step.register
def _execute_concatenate(
    step: ConcatenateStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    del plan
    result = np.concatenate(tuple(values[name] for name in step.input_names), axis=step.axis)
    return {step.output: np.ascontiguousarray(result, dtype=np.int8)}


__all__: list[str] = []
