from __future__ import annotations

import numpy as np

from bakenn.ir import (
    DType,
    Layout,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)


def cmsis_mlp_graph(widths: tuple[int, ...] = (32, 16, 4)) -> QuantizedGraph:
    """Build a deterministic per-tensor-weight MLP accepted by CMSIS-NN FC."""

    if len(widths) < 2:
        raise ValueError("MLP needs an input and output width")
    values: dict[str, TensorType] = {}
    constants: dict[str, np.ndarray] = {}
    ops: list[LinearOp] = []
    input_qparams = PerTensorQParams(0.03125, -3)
    values["input"] = TensorType(
        (1, widths[0]), DType.INT8, Layout.NC, input_qparams
    )
    current = "input"
    current_qparams = input_qparams
    for index, (input_count, output_count) in enumerate(zip(widths, widths[1:])):
        output = f"activation_{index}"
        weight_name = f"weight_{index}"
        bias_name = f"bias_{index}"
        weight_scale = float(np.float32(0.0078125 * (index + 1)))
        output_qparams = PerTensorQParams(
            float(np.float32(0.046875 * (index + 1))),
            -5 + index,
        )
        weight_qparams = PerAxisQParams(
            (weight_scale,) * output_count,
            (0,) * output_count,
            0,
        )
        bias_scale = float(np.float32(current_qparams.scale * weight_scale))
        bias_qparams = PerAxisQParams(
            (bias_scale,) * output_count,
            (0,) * output_count,
            0,
        )
        weight = (
            (np.arange(input_count * output_count, dtype=np.int32) * (7 + index) + 3)
            % 41
            - 20
        ).reshape(output_count, input_count).astype(np.int8)
        bias = (np.arange(output_count, dtype=np.int32) * 31 - 79).astype(np.int32)
        values[weight_name] = TensorType(
            weight.shape, DType.INT8, Layout.OI, weight_qparams
        )
        values[bias_name] = TensorType(
            bias.shape, DType.INT32, Layout.C, bias_qparams
        )
        values[output] = TensorType(
            (1, output_count), DType.INT8, Layout.NC, output_qparams
        )
        constants[weight_name] = weight
        constants[bias_name] = bias
        hidden = index + 1 < len(widths) - 1
        ops.append(
            LinearOp(
                f"linear_{index}",
                current,
                weight_name,
                bias_name,
                output,
                activation_min=output_qparams.zero_point if hidden else -128,
                activation_max=127,
            )
        )
        current = output
        current_qparams = output_qparams
    return QuantizedGraph(
        name="cmsis_mlp_" + "x".join(str(width) for width in widths),
        values=values,
        constants=constants,
        ops=tuple(ops),
        inputs=("input",),
        outputs=(current,),
    )


__all__ = ["cmsis_mlp_graph"]
