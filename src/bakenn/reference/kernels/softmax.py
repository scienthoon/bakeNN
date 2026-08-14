from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.plan.types import ExecutionPlan
from bakenn.plan.steps.softmax import SoftmaxStep
from bakenn.reference.executor import execute_step


@execute_step.register
def _execute_softmax(
    step: SoftmaxStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    del plan
    rows = values[step.input].reshape(step.row_count, step.class_count)
    result = np.empty(rows.shape, dtype=np.int8)
    for row_index, row in enumerate(rows):
        maximum = max(int(value) for value in row)
        weights = [step.lut[maximum - int(value)] for value in row]
        weight_sum = sum(weights)
        if not 0 < weight_sum <= step.sum_bound:
            raise AssertionError("compile-time Softmax LUT-sum proof was violated")
        for class_index, weight in enumerate(weights):
            probability_code = (weight * 256 + weight_sum // 2) // weight_sum
            probability_code = min(255, probability_code)
            result[row_index, class_index] = probability_code - 128
    return {step.output: result.reshape(values[step.input].shape)}


__all__: list[str] = []
