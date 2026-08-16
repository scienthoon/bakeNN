#!/usr/bin/env python3
"""Regenerate the exact MobileNetV2 TFLM comparison artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

import bakenn  # noqa: E402
from bakenn.ir import Conv2DOp, DepthwiseConv2DOp, QuantizedGraph  # noqa: E402
from benchmarks.tflm_compare.quantized_graph_to_tflite import (  # noqa: E402
    export_quantized_graph,
)
from examples.mobilenet_v2_cifar10.run import (  # noqa: E402
    _balanced_calibration,
    _datasets,
    mobilenet_v2_quarter,
)


def _c_rows(data: bytes, width: int = 12) -> str:
    return ",\n".join(
        "    " + ", ".join(f"0x{value:02x}" for value in data[start : start + width])
        for start in range(0, len(data), width)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix-ops",
        type=int,
        default=0,
        help="diagnostic: export only the first N quantized operations",
    )
    parser.add_argument(
        "--normalize-same-padding",
        action="store_true",
        help=(
            "diagnostic: replace stride-2 symmetric 3x3 padding with the "
            "equivalent TFLite SAME split while preserving static shapes"
        ),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    model = mobilenet_v2_quarter()
    model.load_state_dict(
        torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    _, calibration_data, _ = _datasets(arguments.data_dir)
    calibration = _balanced_calibration(calibration_data, 2)
    compiled = bakenn.compile_torch_ptq(
        model,
        calibration[:1],
        calibration,
        arguments.output_dir / "bakenn_intermediate",
        name="mobilenet_v2_025_cifar10_tflm",
    )
    graph = compiled.graph
    if arguments.normalize_same_padding:
        normalized_ops = []
        for op in graph.ops:
            if (
                isinstance(op, (Conv2DOp, DepthwiseConv2DOp))
                and op.stride == (2, 2)
                and op.dilation == (1, 1)
                and op.padding == (1, 1, 1, 1)
            ):
                input_shape = graph.values[op.input].shape
                output_shape = graph.values[op.output].shape
                total_h = max(0, (output_shape[1] - 1) * 2 + 3 - input_shape[1])
                total_w = max(0, (output_shape[2] - 1) * 2 + 3 - input_shape[2])
                op = replace(
                    op,
                    padding=(
                        total_h // 2,
                        total_h - total_h // 2,
                        total_w // 2,
                        total_w - total_w // 2,
                    ),
                )
            normalized_ops.append(op)
        graph = replace(graph, ops=tuple(normalized_ops))
        compiled = bakenn.compile(
            graph,
            arguments.output_dir / "bakenn_same_padding_intermediate",
            model_name=graph.name,
        )
    if arguments.prefix_ops:
        if not 1 <= arguments.prefix_ops <= len(graph.ops):
            raise ValueError("--prefix-ops is outside the graph")
        ops = graph.ops[: arguments.prefix_ops]
        required = set(graph.inputs)
        for op in ops:
            required.update(op.inputs)
            required.update(op.outputs)
        graph = QuantizedGraph(
            name=f"{graph.name}_prefix_{arguments.prefix_ops}",
            values={name: value for name, value in graph.values.items() if name in required},
            constants={
                name: value for name, value in graph.constants.items() if name in required
            },
            ops=ops,
            inputs=graph.inputs,
            outputs=ops[-1].outputs,
            arithmetic_profile=graph.arithmetic_profile,
        )
        compiled = bakenn.compile(
            graph,
            arguments.output_dir / "bakenn_prefix_intermediate",
            model_name=graph.name,
        )
    exported = export_quantized_graph(graph)
    model_path = arguments.output_dir / "model.tflite"
    model_path.write_bytes(exported.data)

    input_type = graph.values[graph.inputs[0]]
    input_codes = np.full(input_type.shape, input_type.qparams.zero_point, dtype=np.int8)
    expected = bakenn.run_reference(compiled.plan, input_codes).reshape(-1)
    header_path = arguments.output_dir / "model_data.h"
    header_path.write_text(
        "#pragma once\n\n"
        "#include <stdint.h>\n\n"
        "extern const unsigned char tflm_model_data[];\n"
        "extern const unsigned int tflm_model_data_len;\n"
        "extern const int8_t tflm_expected_output[];\n"
        "extern const unsigned int tflm_expected_output_len;\n",
        encoding="utf-8",
    )
    source_path = arguments.output_dir / "model_data.cc"
    expected_values = ", ".join(str(int(value)) for value in expected)
    source_path.write_text(
        '#include "model_data.h"\n\n'
        "alignas(16) const unsigned char tflm_model_data[] = {\n"
        + _c_rows(exported.data)
        + "\n};\n"
        + f"const unsigned int tflm_model_data_len = {len(exported.data)}u;\n"
        + f"const int8_t tflm_expected_output[] = {{{expected_values}}};\n"
        + f"const unsigned int tflm_expected_output_len = {len(expected)}u;\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(exported.data).hexdigest()
    print(f"model={model_path} bytes={len(exported.data)} sha256={digest}")
    print(f"operators={exported.operator_counts}")
    print(f"input_zero_point={input_type.qparams.zero_point}")
    print("expected_output=" + " ".join(str(int(value)) for value in expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
