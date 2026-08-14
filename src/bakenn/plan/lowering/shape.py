from __future__ import annotations

from math import prod

from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.shape import ConcatenateOp, FlattenOp, ReshapeOp, SliceOp
from bakenn.plan.lower import lower_op
from bakenn.plan.steps.shape import ConcatenateStep, FlattenStep, ReshapeStep, SliceStep


def _materialize_caller_view(op: ReshapeOp | FlattenOp, graph: QuantizedGraph) -> bool:
    if op.output not in graph.outputs:
        return False
    producers = {output: candidate for candidate in graph.ops for output in candidate.outputs}
    source = op.input
    seen: set[str] = set()
    while source not in seen:
        seen.add(source)
        producer = producers.get(source)
        if not isinstance(producer, (ReshapeOp, FlattenOp)):
            break
        source = producer.input
    return source in graph.inputs


@lower_op.register
def _lower_reshape(op: ReshapeOp, graph: QuantizedGraph) -> ReshapeStep:
    return ReshapeStep(op.name, op.input, op.output, _materialize_caller_view(op, graph))


@lower_op.register
def _lower_flatten(op: FlattenOp, graph: QuantizedGraph) -> FlattenStep:
    return FlattenStep(op.name, op.input, op.output, _materialize_caller_view(op, graph))


@lower_op.register
def _lower_slice(op: SliceOp, graph: QuantizedGraph) -> SliceStep:
    input_shape = graph.values[op.input].shape
    output_shape = graph.values[op.output].shape
    return SliceStep(
        op.name,
        op.input,
        op.output,
        op.axis,
        op.start,
        op.step,
        prod(input_shape[:op.axis]),
        input_shape[op.axis],
        output_shape[op.axis],
        prod(input_shape[op.axis + 1 :]),
    )


@lower_op.register
def _lower_concatenate(op: ConcatenateOp, graph: QuantizedGraph) -> ConcatenateStep:
    output_shape = graph.values[op.output].shape
    axis = op.axis % len(output_shape)
    return ConcatenateStep(
        name=op.name,
        input_names=op.input_names,
        output=op.output,
        axis=axis,
        outer_size=prod(output_shape[:axis]),
        inner_size=prod(output_shape[axis + 1 :]),
        axis_sizes=tuple(graph.values[name].shape[axis] for name in op.input_names),
    )


__all__: list[str] = []
