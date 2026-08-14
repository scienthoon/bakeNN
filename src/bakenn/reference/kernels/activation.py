from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.plan.steps.activation import LUTActivationStep
from bakenn.plan.types import ExecutionPlan
from bakenn.reference.executor import execute_step


@execute_step.register
def _execute_lut(
    step: LUTActivationStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    del plan
    source = values[step.input].astype(np.int16, copy=False)
    lut = np.asarray(step.lut, dtype=np.int8)
    return {step.output: np.ascontiguousarray(lut[source + 128])}


__all__: list[str] = []
