from __future__ import annotations

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir import PerTensorQParams
from bakenn.plan import ExecutionPlan, LinearStep

from ..contracts import (
    ConstantEmission,
    KernelEmission,
    StepEmitContext,
    StepEmission,
    emit_step,
)
from ..fixedpoint import clamp_s8_name, q31_kernel, q31_requantize_name
from ..formatting import format_values
from ..selection import (
    CBackendOptions,
    KernelCapability,
    PackedConstant,
    kernel_capabilities,
)


_PORTABLE_ID = "portable.linear_s8.v1"
_OPTIMIZED_ID = "optimized.linear_oi2.v1"
_TAIL_ID = "optimized.linear_oi2_tail.v1"
_CORTEX_M4_ID = "cortex_m4.linear_smlad.v1"
_MIN_OPTIMIZED_MACS = 48


def _int32_constant(symbol: str, values: tuple[int, ...]) -> ConstantEmission:
    array = np.asarray(values, dtype=np.int32)
    return ConstantEmission(
        symbol=symbol,
        declaration=f"extern const int32_t {symbol}[{len(values)}];",
        definition=f"const int32_t {symbol}[{len(values)}] = {{\n{format_values(array)}\n}};",
        size_bytes=int(array.nbytes),
    )


def _signature(kernel_fn: str) -> str:
    return f"""void {kernel_fn}(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_count,
    size_t output_count,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max)"""


def _portable_kernel(context: StepEmitContext) -> KernelEmission:
    kernel_fn = f"{context.symbol}_linear_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = _signature(kernel_fn)
    return KernelEmission(
        key="linear_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
        definition=f"""{signature} {{
    for (size_t channel = 0; channel < output_count; ++channel) {{
        int32_t accumulator = bias[channel];
        const int8_t *channel_weight = weight + channel * input_count;
        for (size_t index = 0; index < input_count; ++index) {{
            accumulator +=
                ((int32_t)input[index] - input_zero_point)
                * (int32_t)channel_weight[index];
        }}
        const int32_t scaled =
            {requantize}(accumulator, multiplier[channel], shift[channel]);
        output[channel] = {clamp}(
            (int64_t)scaled + output_zero_point, activation_min, activation_max);
    }}
}}""",
    )


def _optimized_kernel(context: StepEmitContext) -> KernelEmission:
    """Accumulate two outputs at once from input-major packed OI2 weights."""

    kernel_fn = f"{context.symbol}_linear_oi2_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = _signature(kernel_fn)
    return KernelEmission(
        key="linear_oi2_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
        definition=f"""{signature} {{
    for (size_t channel = 0; channel < output_count; channel += 2u) {{
        int32_t accumulator_0 = bias[channel];
        int32_t accumulator_1 = bias[channel + 1u];
        const int8_t *pair_weight = weight + channel * input_count;
        for (size_t index = 0; index < input_count; ++index) {{
            const int32_t input_value = (int32_t)input[index] - input_zero_point;
            const int8_t *pair = pair_weight + index * 2u;
            accumulator_0 += input_value * (int32_t)pair[0];
            accumulator_1 += input_value * (int32_t)pair[1];
        }}
        const int32_t scaled_0 = {requantize}(
            accumulator_0, multiplier[channel], shift[channel]);
        const int32_t scaled_1 = {requantize}(
            accumulator_1, multiplier[channel + 1u], shift[channel + 1u]);
        output[channel] = {clamp}(
            (int64_t)scaled_0 + output_zero_point, activation_min, activation_max);
        output[channel + 1u] = {clamp}(
            (int64_t)scaled_1 + output_zero_point, activation_min, activation_max);
    }}
}}""",
    )


def _optimized_tail_kernel(context: StepEmitContext) -> KernelEmission:
    """Pair output rows and process one final output row from a packed tail."""

    kernel_fn = f"{context.symbol}_linear_oi2_tail_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = _signature(kernel_fn)
    return KernelEmission(
        key="linear_oi2_tail_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
        definition=f"""{signature} {{
    const size_t pair_count = output_count - (output_count % 2u);
    for (size_t channel = 0; channel < pair_count; channel += 2u) {{
        int32_t accumulator_0 = bias[channel];
        int32_t accumulator_1 = bias[channel + 1u];
        const int8_t *pair_weight = weight + (channel / 2u) * input_count * 2u;
        for (size_t index = 0; index < input_count; ++index) {{
            const int32_t input_value = (int32_t)input[index] - input_zero_point;
            const int8_t *pair = pair_weight + index * 2u;
            accumulator_0 += input_value * (int32_t)pair[0];
            accumulator_1 += input_value * (int32_t)pair[1];
        }}
        const int32_t scaled_0 = {requantize}(
            accumulator_0, multiplier[channel], shift[channel]);
        const int32_t scaled_1 = {requantize}(
            accumulator_1, multiplier[channel + 1u], shift[channel + 1u]);
        output[channel] = {clamp}(
            (int64_t)scaled_0 + output_zero_point, activation_min, activation_max);
        output[channel + 1u] = {clamp}(
            (int64_t)scaled_1 + output_zero_point, activation_min, activation_max);
    }}
    if (pair_count != output_count) {{
        int32_t accumulator = bias[pair_count];
        const int8_t *tail_weight = weight + (pair_count / 2u) * input_count * 2u;
        for (size_t index = 0; index < input_count; ++index) {{
            const int32_t input_value = (int32_t)input[index] - input_zero_point;
            accumulator += input_value * (int32_t)tail_weight[index];
        }}
        const int32_t scaled = {requantize}(
            accumulator, multiplier[pair_count], shift[pair_count]);
        output[pair_count] = {clamp}(
            (int64_t)scaled + output_zero_point, activation_min, activation_max);
    }}
}}""",
    )


def _cortex_m4_kernel(context: StepEmitContext) -> KernelEmission:
    """Use the ARMv7E-M dual 16-bit multiply-accumulate instruction."""

    kernel_fn = f"{context.symbol}_linear_cortex_m4_smlad_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = f"""void {kernel_fn}(
    const int8_t *input,
    const int32_t *packed_weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_count,
    size_t output_count,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max)"""
    return KernelEmission(
        key="cortex_m4_linear_smlad_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
        definition=f"""{signature} {{
    const size_t pair_count = (input_count + 1u) / 2u;
    for (size_t channel = 0; channel < output_count; ++channel) {{
        int32_t accumulator = bias[channel];
        const int32_t *channel_weight = packed_weight + channel * pair_count;
        for (size_t pair = 0; pair < pair_count; ++pair) {{
            const size_t index = pair * 2u;
            const int32_t low = (int32_t)input[index] - input_zero_point;
            const int32_t high = index + 1u < input_count
                ? (int32_t)input[index + 1u] - input_zero_point
                : 0;
            const uint32_t input_pair =
                ((uint32_t)low & UINT32_C(0xffff))
                | (((uint32_t)high & UINT32_C(0xffff)) << 16u);
            accumulator = __builtin_arm_smlad(
                (int32_t)input_pair, channel_weight[pair], accumulator);
        }}
        const int32_t scaled = {requantize}(
            accumulator, multiplier[channel], shift[channel]);
        output[channel] = {clamp}(
            (int64_t)scaled + output_zero_point, activation_min, activation_max);
    }}
}}""",
    )


@kernel_capabilities.register
def _linear_capabilities(
    step: LinearStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> tuple[KernelCapability, ...]:
    portable = KernelCapability(
        kernel_id=_PORTABLE_ID,
        priority=0,
        optimized=False,
        supported=True,
        reason="generic OI Linear kernel supports every verified LinearStep",
    )
    input_count = plan.tensors[step.input].tensor_type.shape[1]
    output_count = plan.tensors[step.output].tensor_type.shape[1]
    weight = plan.constants[step.weight]
    shape_valid = weight.dtype == np.int8 and weight.shape == (output_count, input_count)

    def cortex_m4_capability() -> KernelCapability:
        failure: str | None = None
        if "dsp" not in options.target.features or "armv7e-m" not in options.target.features:
            failure = "target does not provide ARMv7E-M DSP instructions"
        elif not options.enable_weight_packing:
            failure = "weight packing is disabled"
        elif input_count * output_count < _MIN_OPTIMIZED_MACS:
            failure = (
                f"Linear MAC count {input_count * output_count} is below the "
                f"optimized threshold {_MIN_OPTIMIZED_MACS}"
            )
        elif not shape_valid:
            failure = "semantic weight must be an OI int8 matrix matching the Linear shape"
        if failure is not None:
            return KernelCapability(
                kernel_id=_CORTEX_M4_ID,
                priority=300,
                optimized=True,
                supported=False,
                reason=failure,
            )
        pair_count = (input_count + 1) // 2
        packed_value = np.zeros((output_count, pair_count), dtype=np.int32)
        for channel in range(output_count):
            for pair in range(pair_count):
                low = int(weight[channel, pair * 2]) & 0xFFFF
                high_index = pair * 2 + 1
                high = int(weight[channel, high_index]) & 0xFFFF if high_index < input_count else 0
                word = np.uint32(low | (high << 16))
                packed_value[channel, pair] = word.view(np.int32)
        packed_name = f"{step.weight}.cortex_m4_smlad"
        packed = PackedConstant(
            name=packed_name,
            source=step.weight,
            layout="linear_arm_smlad_i16x2_v1",
            value=packed_value,
            alignment=4,
        )
        return KernelCapability(
            kernel_id=_CORTEX_M4_ID,
            priority=300,
            optimized=True,
            supported=True,
            reason=(
                "ARMv7E-M SMLAD consumes two centered activation codes and two "
                "sign-extended weights per instruction"
            ),
            packed_constants=(packed,),
            constant_overrides={step.weight: packed.name},
        )

    def pair_capability() -> KernelCapability:
        failure: str | None = None
        if not options.enable_weight_packing:
            failure = "weight packing is disabled"
        elif output_count < 2 or output_count % 2:
            failure = "output feature count is not an even output-pair shape"
        elif input_count * output_count < _MIN_OPTIMIZED_MACS:
            failure = (
                f"Linear MAC count {input_count * output_count} is below the "
                f"optimized threshold {_MIN_OPTIMIZED_MACS}"
            )
        elif not shape_valid:
            failure = "semantic weight must be an OI int8 matrix matching the Linear shape"
        if failure is not None:
            return KernelCapability(
                kernel_id=_OPTIMIZED_ID,
                priority=100,
                optimized=True,
                supported=False,
                reason=failure,
            )
        packed_name = f"{step.weight}.linear_oi2"
        packed_value = np.ascontiguousarray(
            weight.reshape(output_count // 2, 2, input_count).transpose(0, 2, 1)
        )
        packed = PackedConstant(
            name=packed_name,
            source=step.weight,
            layout="linear_oi2_interleaved_v1",
            value=packed_value,
        )
        return KernelCapability(
            kernel_id=_OPTIMIZED_ID,
            priority=100,
            optimized=True,
            supported=True,
            reason=(
                "OI int8 weights packed as output pairs; two accumulators reuse each "
                "centered input code"
            ),
            packed_constants=(packed,),
            constant_overrides={step.weight: packed.name},
        )

    def tail_capability() -> KernelCapability:
        failure: str | None = None
        if not options.enable_weight_packing:
            failure = "weight packing is disabled"
        elif output_count < 3 or output_count % 2 == 0:
            failure = "output feature count has no odd tail"
        elif input_count * output_count < _MIN_OPTIMIZED_MACS:
            failure = (
                f"Linear MAC count {input_count * output_count} is below the "
                f"optimized threshold {_MIN_OPTIMIZED_MACS}"
            )
        elif not shape_valid:
            failure = "semantic weight must be an OI int8 matrix matching the Linear shape"
        if failure is not None:
            return KernelCapability(
                kernel_id=_TAIL_ID,
                priority=99,
                optimized=True,
                supported=False,
                reason=failure,
            )
        pair_count = output_count - 1
        packed_name = f"{step.weight}.linear_oi2_tail"
        pair_values = weight[:pair_count].reshape(pair_count // 2, 2, input_count).transpose(
            0, 2, 1
        )
        packed_value = np.ascontiguousarray(
            np.concatenate((pair_values.reshape(-1), weight[pair_count].reshape(-1)))
        )
        packed = PackedConstant(
            name=packed_name,
            source=step.weight,
            layout="linear_oi2_tail_interleaved_v1",
            value=packed_value,
        )
        return KernelCapability(
            kernel_id=_TAIL_ID,
            priority=99,
            optimized=True,
            supported=True,
            reason=(
                "OI int8 weights pack output pairs and append one scalar tail row"
            ),
            packed_constants=(packed,),
            constant_overrides={step.weight: packed.name},
        )

    return (cortex_m4_capability(), pair_capability(), tail_capability(), portable)


@emit_step.register
def _emit_linear(step: LinearStep, context: StepEmitContext) -> StepEmission:
    input_tensor = context.plan.tensors[step.input].tensor_type
    output_tensor = context.plan.tensors[step.output].tensor_type
    input_qparams = input_tensor.qparams
    output_qparams = output_tensor.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)

    implementation = _PORTABLE_ID if context.selection is None else context.selection.kernel_id
    if implementation == _PORTABLE_ID:
        kernel_fn = f"{context.symbol}_linear_s8"
        selected_kernel = _portable_kernel(context)
    elif implementation == _OPTIMIZED_ID:
        kernel_fn = f"{context.symbol}_linear_oi2_s8"
        selected_kernel = _optimized_kernel(context)
    elif implementation == _TAIL_ID:
        kernel_fn = f"{context.symbol}_linear_oi2_tail_s8"
        selected_kernel = _optimized_tail_kernel(context)
    elif implementation == _CORTEX_M4_ID:
        kernel_fn = f"{context.symbol}_linear_cortex_m4_smlad_s8"
        selected_kernel = _cortex_m4_kernel(context)
    else:
        raise CompileError(f"unsupported Linear C implementation {implementation}")

    multiplier_symbol = f"{context.symbol}_op{context.step_index}_multiplier"
    shift_symbol = f"{context.symbol}_op{context.step_index}_shift"
    call = (
        f"    {kernel_fn}(\n"
        f"        {context.pointer(step.input, mutable=False)},\n"
        f"        {context.pointer(step.weight, mutable=False)},\n"
        f"        {context.pointer(step.bias, mutable=False)},\n"
        f"        {multiplier_symbol},\n"
        f"        {shift_symbol},\n"
        f"        {context.pointer(step.output, mutable=True)},\n"
        f"        {input_tensor.shape[1]}u, {output_tensor.shape[1]}u,\n"
        f"        {input_qparams.zero_point}, {output_qparams.zero_point},\n"
        f"        {step.activation_min}, {step.activation_max});"
    )
    return StepEmission(
        constants=(
            _int32_constant(multiplier_symbol, step.multipliers),
            _int32_constant(shift_symbol, step.shifts),
        ),
        kernels=(q31_kernel(context), selected_kernel),
        call=call,
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "input": step.input,
            "output": step.output,
            "accumulator_bound_max": max(step.accumulator_bounds),
        },
    )


__all__: list[str] = []
