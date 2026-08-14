from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, localcontext

from bakenn.errors import CompileError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.softmax import SoftmaxOp
from bakenn.ir.types import PerTensorQParams
from bakenn.ir.verifiers.softmax import Q15_ONE, UINT32_MAX
from bakenn.plan.lower import lower_op
from bakenn.plan.steps.softmax import SoftmaxStep


def build_softmax_lut(input_scale: float) -> tuple[int, ...]:
    """Create the exact model-specific ``bakenn.softmax_lut.q15.v1`` table."""

    # Decimal avoids delegating bit-defining constants to the host libm.  The
    # quantized graph already stores ``input_scale`` as a binary64 value, so its
    # exact Decimal expansion is the normative LUT input.
    with localcontext() as context:
        context.prec = 80
        scale = Decimal.from_float(input_scale)
        q15_one = Decimal(Q15_ONE)
        values = tuple(
            max(
                0,
                min(
                    Q15_ONE,
                    int(
                        ((-Decimal(index) * scale).exp() * q15_one).to_integral_value(
                            rounding=ROUND_HALF_UP
                        )
                    ),
                ),
            )
            for index in range(256)
        )
    if len(values) != 256 or values[0] != Q15_ONE:
        raise CompileError("failed to construct a valid Softmax Q15 LUT")
    if any(left < right for left, right in zip(values, values[1:])):
        raise CompileError("Softmax Q15 LUT must be monotonically non-increasing")
    return values


@lower_op.register
def _lower_softmax(op: SoftmaxOp, graph: QuantizedGraph) -> SoftmaxStep:
    input_type = graph.values[op.input]
    assert isinstance(input_type.qparams, PerTensorQParams)
    row_count, class_count = input_type.shape
    sum_bound = class_count * Q15_ONE
    if sum_bound > UINT32_MAX:
        raise CompileError(f"{op.name}: Q15 LUT sum bound exceeds uint32")
    lut = build_softmax_lut(input_type.qparams.scale)
    if lut[0] == 0:
        raise CompileError(f"{op.name}: Softmax LUT cannot prove a non-zero row sum")
    return SoftmaxStep(
        name=op.name,
        input=op.input,
        output=op.output,
        lut=lut,
        row_count=row_count,
        class_count=class_count,
        sum_bound=sum_bound,
    )


__all__ = ["build_softmax_lut"]
