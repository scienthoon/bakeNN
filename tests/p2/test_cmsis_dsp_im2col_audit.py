"""Exercise the pinned DSP buffer-fill code, which host CMSIS builds skip.

This isolates the actual source block before its ARM intrinsics. It does not
emulate the whole ARM kernel or claim physical Cortex-M4 execution.
"""

from importlib.resources import files
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tests.p2.support import require_compiler


@pytest.mark.parametrize("origin", [-2, -1, 0, 1, 2])
def test_pinned_dsp_im2col_write_extent(tmp_path: Path, origin: int) -> None:
    source = files("bakenn.backend.cmsis_nn.vendor").joinpath(
        "cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c"
    ).read_text(encoding="utf-8")
    dsp = source.split("#else // ARM_MATH_DSP", 1)[1]
    start = dsp.index("            /* Out of bounds is only considered for the y axis")
    end = dsp.index("            row_count = output_ch / 4;", start)
    # Compile the vendored statements unchanged. Only the independent q7->q15
    # conversion is replaced with its scalar definition; no DSP arithmetic is
    # involved in this scratch-write reproduction.
    block = dsp[start:end]
    harness = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
static void arm_q7_to_q15_with_offset(const int8_t *src, int16_t *dst,
                                    int32_t count, int32_t offset) {
    for (int32_t i = 0; i < count; ++i) dst[i] = src[i] + offset;
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const int16_t base_idx_y = (int16_t)atoi(argv[1]);
    const int16_t base_idx_x = 0;
    const int kernel_x = 1, kernel_y = 1, input_ch = 1;
    const int input_x = 1, input_y = 1, input_offset = 0;
    const int8_t input[1] = {2};
    int16_t *col_buffer = malloc(sizeof(int16_t));
    if (col_buffer == NULL) return 3;
''' + block + r'''
    const size_t written = fwrite(col_buffer, sizeof(int16_t), 1, stdout);
    free(col_buffer);
    return written != 1;
}
'''
    runner = tmp_path / "dsp_fill"
    built = subprocess.run([
        require_compiler(os.environ.get("CC", "cc")), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
        "-pedantic", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        "-x", "c", "-", "-o", str(runner),
    ], input=harness, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(runner), str(origin)], capture_output=True,
                            env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"})
    if abs(origin) == 2:
        assert result.returncode != 0
        assert b"AddressSanitizer: heap-buffer-overflow" in result.stderr
    else:
        assert result.returncode == 0, result.stderr.decode()
        np.testing.assert_array_equal(np.frombuffer(result.stdout, dtype=np.int16),
                                      [2 if origin == 0 else 0])
