from __future__ import annotations

from dataclasses import replace

from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.op import LinearOp
from bakenn.ir.ops.conv import Conv2DOp, DepthwiseConv2DOp
from bakenn.ir.ops.elementwise import AddOp, ClampOp, MulOp, RequantizeOp
from bakenn.ir.ops.pool import AveragePool2DOp, MaxPool2DOp
from bakenn.ir.ops.sequence import AveragePool1DOp, Conv1DOp, MaxPool1DOp
from bakenn.ir.verify import verify_graph

# Install only framework-neutral verifier registrations. The pass itself never
# imports or stores a PyTorch/TFLite type.
import bakenn.ir.verifiers.conv  # noqa: F401
import bakenn.ir.verifiers.elementwise  # noqa: F401
import bakenn.ir.verifiers.pool  # noqa: F401
import bakenn.ir.verifiers.shape  # noqa: F401
import bakenn.ir.verifiers.softmax  # noqa: F401
import bakenn.ir.verifiers.sequence  # noqa: F401
import bakenn.ir.verifiers.tensor  # noqa: F401


_FUSIBLE_PRODUCERS = (
    Conv2DOp,
    Conv1DOp,
    DepthwiseConv2DOp,
    LinearOp,
    AddOp,
    MulOp,
    AveragePool2DOp,
    MaxPool2DOp,
    AveragePool1DOp,
    MaxPool1DOp,
    RequantizeOp,
)


def _composed_bounds(producer: object, clamp: ClampOp) -> tuple[int, int]:
    """Compose two monotone saturating clamps, including disjoint ranges."""

    producer_min = int(getattr(producer, "activation_min"))
    producer_max = int(getattr(producer, "activation_max"))

    def downstream(value: int) -> int:
        return min(clamp.activation_max, max(clamp.activation_min, value))

    return downstream(producer_min), downstream(producer_max)


def _snapshot(graph: QuantizedGraph) -> QuantizedGraph:
    return QuantizedGraph(
        name=graph.name,
        values=graph.values,
        constants=graph.constants,
        ops=graph.ops,
        inputs=graph.inputs,
        outputs=graph.outputs,
        arithmetic_profile=graph.arithmetic_profile,
    )


def _fuse_once(graph: QuantizedGraph) -> tuple[QuantizedGraph, bool]:
    producers: dict[str, tuple[int, object]] = {}
    consumers: dict[str, list[tuple[int, object, int]]] = {}
    for op_index, op in enumerate(graph.ops):
        for output in op.outputs:
            producers[output] = (op_index, op)
        for edge_index, input_name in enumerate(op.inputs):
            consumers.setdefault(input_name, []).append((op_index, op, edge_index))

    for clamp_index, candidate in enumerate(graph.ops):
        if not isinstance(candidate, ClampOp):
            continue
        producer_entry = producers.get(candidate.input)
        if producer_entry is None:
            continue
        producer_index, producer = producer_entry
        if not isinstance(producer, _FUSIBLE_PRODUCERS):
            continue
        if candidate.input in graph.outputs:
            continue
        uses = consumers.get(candidate.input, [])
        if len(uses) != 1 or uses[0][0] != clamp_index:
            continue
        if graph.values[candidate.input] != graph.values[candidate.output]:
            # Shape/layout/dtype/qparams equality proves that removing the
            # explicit Clamp does not cross a requantization/rounding point.
            continue

        activation_min, activation_max = _composed_bounds(producer, candidate)
        fused_producer = replace(
            producer,
            output=candidate.output,
            activation_min=activation_min,
            activation_max=activation_max,
        )
        operations = [
            fused_producer if index == producer_index else op
            for index, op in enumerate(graph.ops)
            if index != clamp_index
        ]
        values = dict(graph.values)
        del values[candidate.input]
        return (
            QuantizedGraph(
                name=graph.name,
                values=values,
                constants=graph.constants,
                ops=tuple(operations),
                inputs=graph.inputs,
                outputs=graph.outputs,
                arithmetic_profile=graph.arithmetic_profile,
            ),
            True,
        )
    return graph, False


def fuse_clamps(graph: QuantizedGraph) -> QuantizedGraph:
    """Fuse legal single-consumer Clamp chains into arithmetic producers."""

    verify_graph(graph)
    current = _snapshot(graph)
    while True:
        current, changed = _fuse_once(current)
        if not changed:
            break
    verify_graph(current)
    return current


__all__ = ["fuse_clamps"]
