from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

# Family modules are intentionally explicit until the central P0 aggregators
# are wired by the integration owner.
import bakenn.ir.verifiers.elementwise  # noqa: F401
import bakenn.plan.lowering.elementwise  # noqa: F401
import bakenn.reference.kernels.elementwise  # noqa: F401
from bakenn.errors import CompileError, GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.elementwise import AddOp, ClampOp, MulOp, RequantizeOp
from bakenn.ir.types import DType, Layout, PerTensorQParams, TensorType
from bakenn.ir.verify import verify_graph
from bakenn.plan.lower import lower_to_plan
from bakenn.plan.steps.elementwise import AddStep, ClampStep, MulStep, RequantizeStep
from bakenn.quantization.fixedpoint import INT32_MAX
from bakenn.reference.executor import run_reference


def _type(
    qparams: PerTensorQParams,
    shape: tuple[int, ...] = (1, 4),
    *,
    layout: Layout = Layout.NC,
    dtype: DType = DType.INT8,
) -> TensorType:
    return TensorType(shape, dtype, layout, qparams)


def _unary_graph(op: ClampOp | RequantizeOp, input_type: TensorType, output_type: TensorType) -> QuantizedGraph:
    return QuantizedGraph(
        name=op.name,
        values={"input": input_type, "output": output_type},
        constants={},
        ops=(op,),
        inputs=("input",),
        outputs=("output",),
    )


def _binary_graph(
    op: AddOp | MulOp,
    input_type: TensorType,
    rhs_type: TensorType,
    output_type: TensorType,
    rhs: np.ndarray,
) -> QuantizedGraph:
    return QuantizedGraph(
        name=op.name,
        values={"input": input_type, "rhs": rhs_type, "output": output_type},
        constants={"rhs": np.asarray(rhs, dtype=np.int8).reshape(rhs_type.shape)},
        ops=(op,),
        inputs=("input",),
        outputs=("output",),
    )


def test_add_hand_golden_uses_declared_left_shift_20_profile() -> None:
    graph = _binary_graph(
        AddOp("add", "input", "rhs", "output"),
        _type(PerTensorQParams(0.25, -3)),
        _type(PerTensorQParams(0.5, 5)),
        _type(PerTensorQParams(0.25, -1)),
        np.asarray([[-7, -1, 3, 120]], dtype=np.int8),
    )
    plan = lower_to_plan(graph)
    step = plan.steps[0]
    assert isinstance(step, AddStep)
    assert step.left_shift == 20
    assert step.input_a_shifted_bound == step.input_a_centered_bound * (1 << 20)
    assert step.input_b_shifted_bound == step.input_b_centered_bound * (1 << 20)
    for bound in (
        step.input_a_shifted_bound,
        step.input_b_shifted_bound,
        step.input_a_pre_high_mul_bound,
        step.input_b_pre_high_mul_bound,
        step.sum_bound,
        step.output_pre_high_mul_bound,
    ):
        assert 0 <= bound <= INT32_MAX
    actual = run_reference(plan, np.asarray([[-5, -1, 4, 20]], dtype=np.int8))
    # For these power-of-two scales the exact affine result is a + 2*b - 8.
    np.testing.assert_array_equal(actual, np.asarray([[-27, -11, 2, 127]], dtype=np.int8))


def test_mul_hand_golden_negative_and_saturation() -> None:
    graph = _binary_graph(
        MulOp("mul", "input", "rhs", "output"),
        _type(PerTensorQParams(0.5, 0)),
        _type(PerTensorQParams(0.25, 0)),
        _type(PerTensorQParams(0.125, 0)),
        np.asarray([[-2, -1, 2, 30]], dtype=np.int8),
    )
    plan = lower_to_plan(graph)
    step = plan.steps[0]
    assert isinstance(step, MulStep)
    assert step.requantize_pre_high_mul_bound <= INT32_MAX
    actual = run_reference(plan, np.asarray([[-3, 4, -5, 6]], dtype=np.int8))
    np.testing.assert_array_equal(actual, np.asarray([[6, -4, -10, 127]], dtype=np.int8))


def test_clamp_and_requantize_hand_goldens() -> None:
    clamp_qparams = PerTensorQParams(0.25, -4)
    clamp_plan = lower_to_plan(
        _unary_graph(
            ClampOp("relu6", "input", "output", -4, 20),
            _type(clamp_qparams),
            _type(clamp_qparams),
        )
    )
    clamp_step = clamp_plan.steps[0]
    assert isinstance(clamp_step, ClampStep)
    assert clamp_step.aliases == ()
    np.testing.assert_array_equal(
        run_reference(clamp_plan, np.asarray([[-128, -4, 7, 127]], dtype=np.int8)),
        np.asarray([[-4, -4, 7, 20]], dtype=np.int8),
    )

    requantize_plan = lower_to_plan(
        _unary_graph(
            RequantizeOp("requantize", "input", "output"),
            _type(PerTensorQParams(0.5, -3), shape=(1, 5)),
            _type(PerTensorQParams(0.25, 1), shape=(1, 5)),
        )
    )
    requantize_step = requantize_plan.steps[0]
    assert isinstance(requantize_step, RequantizeStep)
    assert requantize_step.aliases == ()
    np.testing.assert_array_equal(
        run_reference(
            requantize_plan, np.asarray([[-128, -3, -2, 0, 127]], dtype=np.int8)
        ),
        np.asarray([[-128, 1, 3, 7, 127]], dtype=np.int8),
    )


def test_elementwise_rejects_broadcast_layout_dtype_and_clamp_qparam_mismatch() -> None:
    qparams = PerTensorQParams(0.25, 0)
    rhs = np.zeros((1, 3), dtype=np.int8)
    broadcast = _binary_graph(
        AddOp("bad_add", "input", "rhs", "output"),
        _type(qparams),
        _type(qparams, shape=(1, 3)),
        _type(qparams),
        rhs,
    )
    with pytest.raises(GraphValidationError, match="not statically broadcast-compatible"):
        verify_graph(broadcast)

    layout_mismatch = replace(
        broadcast,
        values={
            "input": _type(qparams, shape=(1, 2, 2, 1), layout=Layout.NHWC),
            "rhs": _type(qparams, shape=(1, 2, 2, 1), layout=Layout.NHWC),
            "output": _type(qparams, shape=(1, 2, 2, 1), layout=Layout.NC),
        },
        constants={"rhs": np.zeros((1, 2, 2, 1), dtype=np.int8)},
    )
    with pytest.raises(GraphValidationError, match="layouts must match"):
        verify_graph(layout_mismatch)

    dtype_mismatch = _binary_graph(
        MulOp("bad_mul", "input", "rhs", "output"),
        _type(qparams),
        _type(qparams),
        _type(qparams, dtype=DType.INT32),
        np.zeros((1, 4), dtype=np.int8),
    )
    with pytest.raises(GraphValidationError, match="must be int8"):
        verify_graph(dtype_mismatch)

    clamp = _unary_graph(
        ClampOp("bad_clamp", "input", "output", 0, 127),
        _type(PerTensorQParams(0.25, 0)),
        _type(PerTensorQParams(0.5, 0)),
    )
    with pytest.raises(GraphValidationError, match="Clamp preserves qparams"):
        verify_graph(clamp)


def test_aliasing_is_explicit_and_memory_planner_rejects_unsafe_caller_input_alias() -> None:
    qparams = PerTensorQParams(0.25, 0)
    ordinary = lower_to_plan(
        _unary_graph(
            ClampOp("ordinary", "input", "output", 0, 127),
            _type(qparams),
            _type(qparams),
        )
    )
    assert ordinary.steps[0].aliases == ()

    unsafe = _unary_graph(
        ClampOp("unsafe", "input", "output", 0, 127, inplace=True),
        _type(qparams),
        _type(qparams),
    )
    with pytest.raises(CompileError, match="caller/constant storage is read-only"):
        lower_to_plan(unsafe)

    safe = QuantizedGraph(
        name="safe_inplace",
        values={
            "input": _type(PerTensorQParams(0.5, 0)),
            "intermediate": _type(qparams),
            "output": _type(qparams),
        },
        constants={},
        ops=(
            RequantizeOp("prepare", "input", "intermediate"),
            ClampOp("explicit_inplace", "intermediate", "output", 0, 127, inplace=True),
        ),
        inputs=("input",),
        outputs=("output",),
    )
    safe_plan = lower_to_plan(safe)
    assert safe_plan.steps[0].aliases == ()
    assert len(safe_plan.steps[1].aliases) == 1
    assert safe_plan.tensors["output"].alias_of == "intermediate"


def test_mul_and_requantize_positive_q31_shift_proofs_fail_closed() -> None:
    mul = _binary_graph(
        MulOp("unsafe_mul", "input", "rhs", "output"),
        _type(PerTensorQParams(256.0, -128)),
        _type(PerTensorQParams(256.0, -128)),
        _type(PerTensorQParams(1.0, 0)),
        np.full((1, 4), 127, dtype=np.int8),
    )
    with pytest.raises(CompileError, match="positive Q31 shift"):
        lower_to_plan(mul)

    requantize = _unary_graph(
        RequantizeOp("unsafe_requantize", "input", "output"),
        _type(PerTensorQParams(float(1 << 23), -128)),
        _type(PerTensorQParams(1.0, 0)),
    )
    with pytest.raises(CompileError, match="positive Q31 shift"):
        lower_to_plan(requantize)


def test_add_output_q31_proof_fails_closed_for_unrepresentable_scale() -> None:
    graph = _binary_graph(
        AddOp("unsafe_add", "input", "rhs", "output"),
        _type(PerTensorQParams(1.0, -128)),
        _type(PerTensorQParams(1.0, -128)),
        _type(PerTensorQParams(1.0e-9, 0)),
        np.full((1, 4), 127, dtype=np.int8),
    )
    with pytest.raises(CompileError, match="positive Q31 shift"):
        lower_to_plan(graph)
