"""Host probe for the vendored ESP-NN ESP32-S3 scratch getters.

BakeNN plans ESP-NN scratch on the host before any target code runs, so it
mirrors the pinned getter arithmetic in Python.  This helper compiles the real
vendored getters for the host and queries them with exactly the dimension and
parameter structs that the generated wrappers pass on the target.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Iterable, Sequence

import bakenn.backend.esp_nn as esp_nn_backend
from bakenn.plan import ExecutionPlan
from bakenn.plan.steps import Conv2DStep, DepthwiseConv2DStep

from .support import require_compiler


VENDOR_ROOT = Path(esp_nn_backend.__file__).resolve().parent / "vendor/esp_nn"

_PROBE_SOURCE = r"""#include "esp_nn_esp32s3.h"
#include <stdio.h>

/* One case per line: kind iw ih ic kw kh oc_or_mult ow oh sw sh pad_w pad_h.
 * The structs match the generated BakeNN ESP-NN wrappers field for field. */
int main(void) {
    char kind;
    int iw, ih, ic, kw, kh, count, ow, oh, sw, sh, pad_w, pad_h;
    while (scanf(" %c %d %d %d %d %d %d %d %d %d %d %d %d", &kind, &iw, &ih,
                 &ic, &kw, &kh, &count, &ow, &oh, &sw, &sh, &pad_w,
                 &pad_h) == 13) {
        const data_dims_t input = { iw, ih, ic, 1 };
        if (kind == 'c') {
            const data_dims_t filter = { kw, kh, ic, count };
            const data_dims_t output = { ow, oh, count, 1 };
            const conv_params_t params = {
                .stride = { sw, sh }, .padding = { pad_w, pad_h },
                .dilation = { 1, 1 }
            };
            printf("%d\n", esp_nn_get_conv_scratch_size_esp32s3(
                &input, &filter, &output, &params));
        } else if (kind == 'd') {
            const int channels = ic * count;
            const data_dims_t filter = { kw, kh, channels, 1 };
            const data_dims_t output = { ow, oh, channels, 1 };
            const dw_conv_params_t params = {
                .ch_mult = count, .stride = { sw, sh },
                .padding = { pad_w, pad_h }, .dilation = { 1, 1 }
            };
            printf("%d\n", esp_nn_get_depthwise_conv_scratch_size_esp32s3(
                &input, &filter, &output, &params));
        } else {
            return 2;
        }
    }
    return ferror(stdin) ? 3 : 0;
}
"""


@dataclass(frozen=True)
class ScratchCase:
    """One ESP-NN S3 scratch query in BakeNN's NHWC terms."""

    kind: str  # "conv" or "depthwise"
    input_height: int
    input_width: int
    input_channels: int
    kernel_height: int
    kernel_width: int
    # Output channels for Conv2D, channel multiplier for DepthwiseConv2D.
    count: int
    output_height: int
    output_width: int
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]  # top, bottom, left, right

    def probe_line(self) -> str:
        pad_top, _, pad_left, _ = self.padding
        stride_height, stride_width = self.stride
        values = (
            self.input_width,
            self.input_height,
            self.input_channels,
            self.kernel_width,
            self.kernel_height,
            self.count,
            self.output_width,
            self.output_height,
            stride_width,
            stride_height,
            pad_left,
            pad_top,
        )
        prefix = {"conv": "c", "depthwise": "d"}[self.kind]
        return prefix + " " + " ".join(str(value) for value in values)


def case_for_step(plan: ExecutionPlan, step: object) -> ScratchCase:
    """Describe one planned Conv2D or DepthwiseConv2D step as a probe case."""

    input_shape = plan.tensors[step.input].tensor_type.shape  # type: ignore[attr-defined]
    output_shape = plan.tensors[step.output].tensor_type.shape  # type: ignore[attr-defined]
    weight_shape = plan.constants[step.weight].shape  # type: ignore[attr-defined]
    _, input_height, input_width, input_channels = input_shape
    _, output_height, output_width, output_channels = output_shape
    if isinstance(step, DepthwiseConv2DStep):
        kernel_height, kernel_width, _ = weight_shape
        kind, count = "depthwise", step.depth_multiplier
    elif isinstance(step, Conv2DStep):
        _, kernel_height, kernel_width, _ = weight_shape
        kind, count = "conv", output_channels
    else:
        raise TypeError(f"not an ESP-NN scratch step: {type(step).__name__}")
    return ScratchCase(
        kind,
        input_height,
        input_width,
        input_channels,
        kernel_height,
        kernel_width,
        count,
        output_height,
        output_width,
        tuple(step.stride),  # type: ignore[attr-defined]
        tuple(step.padding),  # type: ignore[attr-defined]
    )


def build_s3_scratch_probe(output_dir: Path) -> Path:
    """Compile the vendored S3 getters into a host executable."""

    compiler = require_compiler("clang")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "esp_nn_s3_scratch_probe.c"
    source.write_text(_PROBE_SOURCE, encoding="utf-8")
    executable = output_dir / "esp_nn_s3_scratch_probe"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wno-implicit-function-declaration",
            "-Wno-pointer-to-int-cast",
            str(source),
            str(VENDOR_ROOT / "src/convolution/esp_nn_conv_esp32s3.c"),
            str(VENDOR_ROOT / "src/convolution/esp_nn_depthwise_conv_s8_esp32s3.c"),
            "-I",
            str(VENDOR_ROOT / "include"),
            "-I",
            str(VENDOR_ROOT / "src/common"),
            "-Wl,-dead_strip" if os.uname().sysname == "Darwin" else "-Wl,--gc-sections",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable


def query_s3_scratch(probe: Path, cases: Iterable[ScratchCase]) -> list[int]:
    """Return the vendored getter result for every case, in order."""

    lines = [case.probe_line() for case in cases]
    result = subprocess.run(
        [probe],
        input="\n".join(lines) + "\n",
        check=True,
        capture_output=True,
        text=True,
    )
    sizes = [int(value) for value in result.stdout.split()]
    if len(sizes) != len(lines):
        raise AssertionError(
            f"probe answered {len(sizes)} of {len(lines)} scratch queries"
        )
    return sizes


def mismatches(
    cases: Sequence[ScratchCase],
    mirrored: Sequence[int],
    official: Sequence[int],
) -> list[str]:
    """Human-readable disagreements between the mirror and the getter."""

    return [
        f"{case}: mirror={mine} getter={theirs}"
        for case, mine, theirs in zip(cases, mirrored, official)
        if mine != theirs
    ]


__all__ = [
    "ScratchCase",
    "VENDOR_ROOT",
    "build_s3_scratch_probe",
    "case_for_step",
    "mismatches",
    "query_s3_scratch",
]
