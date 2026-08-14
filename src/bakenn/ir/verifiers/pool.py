from __future__ import annotations

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.pool import AveragePool2DOp, MaxPool2DOp
from bakenn.ir.types import DType, Layout, PerTensorQParams, TARGET_SIZE_MAX
from bakenn.ir.verify import verify_op
from bakenn.quantization.fixedpoint import INT32_MAX


def _fail(op: AveragePool2DOp | MaxPool2DOp, message: str) -> None:
    raise GraphValidationError(f"{op.name}: {message}")


def _valid_extent(position: int, stride: int, padding_before: int, kernel: int, size: int) -> int:
    start = position * stride - padding_before
    return max(0, min(start + kernel, size) - max(start, 0))


def _verify_pool(op: AveragePool2DOp | MaxPool2DOp, graph: QuantizedGraph) -> None:
    input_type = graph.values[op.input]
    output_type = graph.values[op.output]
    if input_type.dtype is not DType.INT8 or output_type.dtype is not DType.INT8:
        _fail(op, "pool activations must be int8")
    if input_type.layout is not Layout.NHWC or output_type.layout is not Layout.NHWC:
        _fail(op, "pool activations must use NHWC layout")
    if len(input_type.shape) != 4 or len(output_type.shape) != 4:
        _fail(op, "pool activations must be rank-four")
    if input_type.shape[0] != 1 or output_type.shape[0] != 1:
        _fail(op, "P0 pool requires static batch size one")
    if not isinstance(input_type.qparams, PerTensorQParams) or not isinstance(
        output_type.qparams, PerTensorQParams
    ):
        _fail(op, "pool activations require per-tensor qparams")
    if input_type.qparams != output_type.qparams:
        _fail(op, "input and output qparams must be identical")

    _, input_h, input_w, input_c = input_type.shape
    _, output_h, output_w, output_c = output_type.shape
    kernel_h, kernel_w = op.kernel
    stride_h, stride_w = op.stride
    pad_top, pad_bottom, pad_left, pad_right = op.padding
    if any(
        value > TARGET_SIZE_MAX
        for value in (*op.kernel, *op.stride, *op.padding)
    ):
        _fail(op, "kernel, stride, and padding must fit the 32-bit target ABI")
    expected_h = (input_h + pad_top + pad_bottom - kernel_h) // stride_h + 1
    expected_w = (input_w + pad_left + pad_right - kernel_w) // stride_w + 1
    if expected_h <= 0 or expected_w <= 0:
        _fail(op, "kernel, stride, and padding produce an empty output")
    if output_type.shape != (1, expected_h, expected_w, input_c) or output_c != input_c:
        _fail(
            op,
            f"output shape must be (1, {expected_h}, {expected_w}, {input_c})",
        )

    max_signed_coordinate = (1 << 63) - 1
    max_runtime_y = (output_h - 1) * stride_h + (kernel_h - 1)
    max_runtime_x = (output_w - 1) * stride_w + (kernel_w - 1)
    if max_runtime_y > max_signed_coordinate or max_runtime_x > max_signed_coordinate:
        _fail(op, "runtime pooling coordinates exceed int64")

    # The overlap of a monotonically sliding window with the input is positive
    # on one contiguous interval. Checking both ends therefore proves that no
    # output window has a zero valid-element count.
    if min(
        _valid_extent(0, stride_h, pad_top, kernel_h, input_h),
        _valid_extent(output_h - 1, stride_h, pad_top, kernel_h, input_h),
        _valid_extent(0, stride_w, pad_left, kernel_w, input_w),
        _valid_extent(output_w - 1, stride_w, pad_left, kernel_w, input_w),
    ) <= 0:
        _fail(op, "every pooling window must contain at least one input element")

    if isinstance(op, AveragePool2DOp):
        max_abs_centered = max(
            abs(-128 - input_type.qparams.zero_point),
            abs(127 - input_type.qparams.zero_point),
        )
        bound = max_abs_centered * kernel_h * kernel_w
        if bound > INT32_MAX:
            _fail(op, f"centered average accumulator bound {bound} exceeds int32")


@verify_op.register
def _verify_average_pool(op: AveragePool2DOp, graph: QuantizedGraph) -> None:
    _verify_pool(op, graph)


@verify_op.register
def _verify_max_pool(op: MaxPool2DOp, graph: QuantizedGraph) -> None:
    _verify_pool(op, graph)


__all__: list[str] = []
