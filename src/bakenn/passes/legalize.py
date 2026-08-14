from __future__ import annotations

from dataclasses import replace

from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.elementwise import RequantizeOp
from bakenn.ir.ops.shape import ConcatenateOp
from bakenn.ir.types import TensorType
from bakenn.ir.verify import verify_graph

# These are registration modules, not framework frontends. Importing them here
# makes a standalone legalization result verifiable without relying on an
# external import order.
import bakenn.ir.verifiers.conv  # noqa: F401
import bakenn.ir.verifiers.elementwise  # noqa: F401
import bakenn.ir.verifiers.pool  # noqa: F401
import bakenn.ir.verifiers.shape  # noqa: F401
import bakenn.ir.verifiers.softmax  # noqa: F401


def _unique(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    suffix = 1
    while f"{base}.{suffix}" in used:
        suffix += 1
    result = f"{base}.{suffix}"
    used.add(result)
    return result


def legalize_graph(graph: QuantizedGraph) -> QuantizedGraph:
    """Return a graph whose Concatenate edges all use its output qparams.

    One non-inplace :class:`RequantizeOp` is inserted per mismatched input edge.
    Values and operation names are collision-free and deterministic with
    respect to the immutable input graph. Already-legal edges are untouched.
    """

    values = dict(graph.values)
    operations: list[object] = []
    used_values = set(values)
    used_op_names = {op.name for op in graph.ops}

    for op in graph.ops:
        if not isinstance(op, ConcatenateOp):
            operations.append(op)
            continue

        output_type = values[op.output]
        legalized_inputs: list[str] = []
        for edge_index, input_name in enumerate(op.input_names):
            input_type = values[input_name]
            if input_type.qparams == output_type.qparams:
                legalized_inputs.append(input_name)
                continue

            value_name = _unique(
                f"{input_name}.requantized_for.{op.name}.edge{edge_index}",
                used_values,
            )
            op_name = _unique(f"{op.name}.requantize.edge{edge_index}", used_op_names)
            values[value_name] = TensorType(
                shape=input_type.shape,
                dtype=input_type.dtype,
                layout=input_type.layout,
                qparams=output_type.qparams,
            )
            operations.append(
                RequantizeOp(
                    name=op_name,
                    input=input_name,
                    output=value_name,
                    inplace=False,
                )
            )
            legalized_inputs.append(value_name)

        operations.append(replace(op, input_names=tuple(legalized_inputs)))

    result = QuantizedGraph(
        name=graph.name,
        values=values,
        constants=graph.constants,
        ops=tuple(operations),
        inputs=graph.inputs,
        outputs=graph.outputs,
        arithmetic_profile=graph.arithmetic_profile,
    )
    verify_graph(result)
    return result


__all__ = ["legalize_graph"]
