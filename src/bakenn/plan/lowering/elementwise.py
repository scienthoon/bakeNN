from __future__ import annotations

from bakenn.errors import CompileError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.elementwise import AddOp, ClampOp, MulOp, RequantizeOp
from bakenn.ir.types import PerTensorQParams
from bakenn.plan.lower import lower_op
from bakenn.plan.steps.elementwise import AddStep, ClampStep, MulStep, RequantizeStep
from bakenn.quantization.fixedpoint import (
    INT32_MAX,
    multiply_by_quantized_multiplier,
    quantize_multiplier,
)


def _qparams(graph: QuantizedGraph, value: str) -> PerTensorQParams:
    qparams = graph.values[value].qparams
    assert isinstance(qparams, PerTensorQParams)
    return qparams


def _centered_range(zero_point: int) -> tuple[int, int]:
    return -128 - zero_point, 127 - zero_point


def _abs_bound(bounds: tuple[int, int]) -> int:
    return max(abs(bounds[0]), abs(bounds[1]))


def _checked_scale_bound(value_bound: int, shift: int, description: str) -> int:
    shifted_bound = value_bound * (1 << max(shift, 0))
    if shifted_bound > INT32_MAX:
        raise CompileError(
            f"{description}: positive Q31 shift requires magnitude {shifted_bound}, exceeding int32"
        )
    return shifted_bound


def _scaled_code_bound(
    zero_point: int,
    outer_left_shift: int,
    multiplier: int,
    shift: int,
) -> int:
    largest = 0
    for code in range(-128, 128):
        centered = code - zero_point
        value = centered * (1 << outer_left_shift)
        scaled = multiply_by_quantized_multiplier(value, multiplier, shift)
        largest = max(largest, abs(scaled))
    return largest


@lower_op.register
def _lower_add(op: AddOp, graph: QuantizedGraph) -> AddStep:
    input_a = _qparams(graph, op.input_a)
    input_b = _qparams(graph, op.input_b)
    output = _qparams(graph, op.output)

    twice_max_scale = 2.0 * max(input_a.scale, input_b.scale)
    input_a_multiplier, input_a_shift = quantize_multiplier(input_a.scale / twice_max_scale)
    input_b_multiplier, input_b_shift = quantize_multiplier(input_b.scale / twice_max_scale)
    output_multiplier, output_shift = quantize_multiplier(
        twice_max_scale / ((1 << AddStep.left_shift) * output.scale)
    )

    input_a_centered_bound = _abs_bound(_centered_range(input_a.zero_point))
    input_b_centered_bound = _abs_bound(_centered_range(input_b.zero_point))
    input_a_shifted_bound = input_a_centered_bound * (1 << AddStep.left_shift)
    input_b_shifted_bound = input_b_centered_bound * (1 << AddStep.left_shift)
    if input_a_shifted_bound > INT32_MAX or input_b_shifted_bound > INT32_MAX:
        raise CompileError(f"{op.name}: Add left_shift={AddStep.left_shift} exceeds int32")
    input_a_pre_high_mul_bound = _checked_scale_bound(
        input_a_shifted_bound, input_a_shift, f"{op.name} input_a"
    )
    input_b_pre_high_mul_bound = _checked_scale_bound(
        input_b_shifted_bound, input_b_shift, f"{op.name} input_b"
    )
    input_a_scaled_bound = _scaled_code_bound(
        input_a.zero_point,
        AddStep.left_shift,
        input_a_multiplier,
        input_a_shift,
    )
    input_b_scaled_bound = _scaled_code_bound(
        input_b.zero_point,
        AddStep.left_shift,
        input_b_multiplier,
        input_b_shift,
    )
    sum_bound = input_a_scaled_bound + input_b_scaled_bound
    if sum_bound > INT32_MAX:
        raise CompileError(f"{op.name}: scaled Add inputs may overflow int32 when summed")
    output_pre_high_mul_bound = _checked_scale_bound(
        sum_bound, output_shift, f"{op.name} output"
    )

    return AddStep(
        name=op.name,
        input_a=op.input_a,
        input_b=op.input_b,
        output=op.output,
        input_a_multiplier=input_a_multiplier,
        input_a_shift=input_a_shift,
        input_b_multiplier=input_b_multiplier,
        input_b_shift=input_b_shift,
        output_multiplier=output_multiplier,
        output_shift=output_shift,
        activation_min=op.activation_min,
        activation_max=op.activation_max,
        input_a_centered_bound=input_a_centered_bound,
        input_b_centered_bound=input_b_centered_bound,
        input_a_shifted_bound=input_a_shifted_bound,
        input_b_shifted_bound=input_b_shifted_bound,
        input_a_pre_high_mul_bound=input_a_pre_high_mul_bound,
        input_b_pre_high_mul_bound=input_b_pre_high_mul_bound,
        input_a_scaled_bound=input_a_scaled_bound,
        input_b_scaled_bound=input_b_scaled_bound,
        sum_bound=sum_bound,
        output_pre_high_mul_bound=output_pre_high_mul_bound,
    )


@lower_op.register
def _lower_mul(op: MulOp, graph: QuantizedGraph) -> MulStep:
    input_a = _qparams(graph, op.input_a)
    input_b = _qparams(graph, op.input_b)
    output = _qparams(graph, op.output)
    output_multiplier, output_shift = quantize_multiplier(
        input_a.scale * input_b.scale / output.scale
    )
    input_a_centered_bound = _abs_bound(_centered_range(input_a.zero_point))
    input_b_centered_bound = _abs_bound(_centered_range(input_b.zero_point))
    product_bound = input_a_centered_bound * input_b_centered_bound
    if product_bound > INT32_MAX:
        raise CompileError(f"{op.name}: centered Mul product may overflow int32")
    requantize_pre_high_mul_bound = _checked_scale_bound(
        product_bound, output_shift, f"{op.name} product"
    )
    return MulStep(
        name=op.name,
        input_a=op.input_a,
        input_b=op.input_b,
        output=op.output,
        output_multiplier=output_multiplier,
        output_shift=output_shift,
        activation_min=op.activation_min,
        activation_max=op.activation_max,
        input_a_centered_bound=input_a_centered_bound,
        input_b_centered_bound=input_b_centered_bound,
        product_bound=product_bound,
        requantize_pre_high_mul_bound=requantize_pre_high_mul_bound,
    )


@lower_op.register
def _lower_clamp(op: ClampOp, graph: QuantizedGraph) -> ClampStep:
    del graph
    return ClampStep(
        name=op.name,
        input=op.input,
        output=op.output,
        activation_min=op.activation_min,
        activation_max=op.activation_max,
        inplace=op.inplace,
    )


@lower_op.register
def _lower_requantize(op: RequantizeOp, graph: QuantizedGraph) -> RequantizeStep:
    input_qparams = _qparams(graph, op.input)
    output_qparams = _qparams(graph, op.output)
    multiplier, shift = quantize_multiplier(input_qparams.scale / output_qparams.scale)
    centered_bound = _abs_bound(_centered_range(input_qparams.zero_point))
    requantize_pre_high_mul_bound = _checked_scale_bound(
        centered_bound, shift, f"{op.name} input"
    )
    return RequantizeStep(
        name=op.name,
        input=op.input,
        output=op.output,
        multiplier=multiplier,
        shift=shift,
        centered_bound=centered_bound,
        requantize_pre_high_mul_bound=requantize_pre_high_mul_bound,
        inplace=op.inplace,
        activation_min=op.activation_min,
        activation_max=op.activation_max,
    )


__all__: list[str] = []
