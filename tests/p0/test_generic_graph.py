from dataclasses import dataclass, replace

import numpy as np
import pytest

from bakenn.errors import CompileError, GraphValidationError
from bakenn.ir import (
    DType,
    Layout,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
    verify_graph,
)
from bakenn.plan import ExecutionPlan, PlanTensor, Storage, lower_to_plan
from bakenn.reference import execute_step, run_reference


def _linear_graph() -> QuantizedGraph:
    activation = PerTensorQParams(0.25, -3)
    weight_qparams = PerAxisQParams((0.5, 0.25), (0, 0), 0)
    bias_qparams = PerAxisQParams((0.125, 0.0625), (0, 0), 0)
    return QuantizedGraph(
        name="generic_linear",
        values={
            "input": TensorType((1, 3), DType.INT8, Layout.NC, activation),
            "weight": TensorType((2, 3), DType.INT8, Layout.OI, weight_qparams),
            "bias": TensorType((2,), DType.INT32, Layout.C, bias_qparams),
            "output": TensorType((1, 2), DType.INT8, Layout.NC, PerTensorQParams(0.5, 1)),
        },
        constants={
            "weight": np.asarray([[1, 2, 3], [-1, -2, -3]], dtype=np.int8),
            "bias": np.asarray([0, 0], dtype=np.int32),
        },
        ops=(LinearOp("linear", "input", "weight", "bias", "output"),),
        inputs=("input",),
        outputs=("output",),
    )


def test_generic_op_edges_and_snapshots_are_immutable() -> None:
    source_values = dict(_linear_graph().values)
    source_weight = np.asarray([[1, 2, 3], [-1, -2, -3]], dtype=np.int8)
    graph = replace(
        _linear_graph(),
        values=source_values,
        constants={"weight": source_weight, "bias": np.asarray([0, 0], dtype=np.int32)},
    )
    source_values.clear()
    source_weight[0, 0] = 99

    op = graph.ops[0]
    assert op.inputs == ("input", "weight", "bias")
    assert op.outputs == ("output",)
    assert graph.constants["weight"][0, 0] == 1
    with pytest.raises(TypeError):
        graph.values["new"] = graph.values["input"]  # type: ignore[index]
    with pytest.raises(ValueError):
        graph.constants["weight"][0, 0] = 7

    plan = lower_to_plan(graph)
    with pytest.raises(TypeError):
        plan.tensors["new"] = plan.tensors["input"]  # type: ignore[index]
    with pytest.raises(ValueError):
        plan.constants["weight"][0, 0] = 7


def test_generic_verifier_rejects_multiple_producers_cycles_dead_ops_and_missing_constants() -> None:
    graph = _linear_graph()
    duplicate = LinearOp("duplicate", "output", "weight", "bias", "output")
    with pytest.raises(GraphValidationError, match="multiple definitions"):
        verify_graph(replace(graph, ops=graph.ops + (duplicate,)))

    later_type = graph.values["output"]
    cyclic_values = dict(graph.values)
    cyclic_values["later"] = graph.values["input"]
    cyclic_values["first"] = later_type
    cyclic = replace(
        graph,
        values=cyclic_values,
        ops=(
            LinearOp("first", "later", "weight", "bias", "first"),
            LinearOp("later", "input", "weight", "bias", "later"),
        ),
        outputs=("first",),
    )
    with pytest.raises(GraphValidationError, match="cyclic or not topological"):
        verify_graph(cyclic)

    dead_values = dict(graph.values)
    dead_values["dead"] = graph.values["output"]
    dead = replace(
        graph,
        values=dead_values,
        ops=graph.ops + (LinearOp("dead", "input", "weight", "bias", "dead"),),
    )
    with pytest.raises(GraphValidationError, match="dead or disconnected"):
        verify_graph(dead)

    with pytest.raises(GraphValidationError, match="before it is defined|compile-time constants"):
        verify_graph(replace(graph, constants={"bias": graph.constants["bias"]}))


@dataclass(frozen=True)
class _CopyStep:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    constants: tuple[str, ...] = ()
    aliases: tuple[object, ...] = ()
    scratch_size: int = 0
    scratch_alignment: int = 1
    kernel_kind: str = "test_copy"
    arithmetic_profile: str = "test.copy.v1"


def test_reference_step_dispatch_is_extensible_and_fail_closed() -> None:
    tensor_type = TensorType((1, 2), DType.INT8, Layout.NC, PerTensorQParams(1.0, 0))
    step = _CopyStep("copy", ("input",), ("output",))
    plan = ExecutionPlan(
        name="copy",
        tensors={
            "input": PlanTensor("input", tensor_type, Storage.INPUT),
            "output": PlanTensor("output", tensor_type, Storage.OUTPUT),
        },
        constants={},
        steps=(step,),
        inputs=("input",),
        outputs=("output",),
        arena_size=0,
        arena_alignment=16,
        arithmetic_profile="bakenn.int8.v1",
    )
    with pytest.raises(CompileError, match="no reference executor"):
        run_reference(plan, np.asarray([[1, -2]], dtype=np.int8))

    @execute_step.register(_CopyStep)
    def _execute_copy(step, plan, values):  # type: ignore[no-untyped-def]
        del plan
        return {step.outputs[0]: np.array(values[step.inputs[0]], copy=True)}

    np.testing.assert_array_equal(
        run_reference(plan, np.asarray([[1, -2]], dtype=np.int8)),
        np.asarray([[1, -2]], dtype=np.int8),
    )


def test_execution_plan_rejects_unknown_or_misclassified_public_abi_tensors() -> None:
    tensor_type = TensorType((1, 2), DType.INT8, Layout.NC, PerTensorQParams(1.0, 0))
    with pytest.raises(CompileError, match="plan input.*input storage"):
        ExecutionPlan(
            name="bad_input_storage",
            tensors={
                "input": PlanTensor("input", tensor_type, Storage.OUTPUT),
                "output": PlanTensor("output", tensor_type, Storage.OUTPUT),
            },
            constants={},
            steps=(_CopyStep("copy", ("input",), ("output",)),),
            inputs=("input",),
            outputs=("output",),
            arena_size=0,
            arena_alignment=16,
            arithmetic_profile="bakenn.int8.v1",
        )
    with pytest.raises(CompileError, match="unknown tensors"):
        ExecutionPlan(
            name="unknown_step_tensor",
            tensors={
                "input": PlanTensor("input", tensor_type, Storage.INPUT),
                "output": PlanTensor("output", tensor_type, Storage.OUTPUT),
            },
            constants={},
            steps=(_CopyStep("copy", ("missing",), ("output",)),),
            inputs=("input",),
            outputs=("output",),
            arena_size=0,
            arena_alignment=16,
            arithmetic_profile="bakenn.int8.v1",
        )
