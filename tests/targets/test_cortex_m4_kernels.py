from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.backend.portable_c import select_backend_plan
from bakenn.targets import CORTEX_M4, build_freestanding_elf
from tests.p0.model_fixtures import mobilenet_v1_graph, tiny_cnn_graph
from tests.p2.test_backend_selection import linear_graph


def _options() -> bakenn.CBackendOptions:
    return bakenn.CBackendOptions(
        kernel_policy=bakenn.KernelPolicy.AUTO,
        target=CORTEX_M4,
    )


def _decode_lane(word: np.int32, lane: int) -> int:
    bits = (int(np.uint32(word)) >> (lane * 16)) & 0xFFFF
    return bits - 0x10000 if bits & 0x8000 else bits


def test_cortex_m4_capabilities_select_versioned_intrinsic_and_scratch_kernels(
    tmp_path: Path,
) -> None:
    options = _options()
    tiny = bakenn.compile(tiny_cnn_graph(), tmp_path / "tiny", backend_options=options, target=CORTEX_M4)
    tiny_backend = select_backend_plan(tiny.plan, options)
    assert [selection.kernel_id for selection in tiny_backend.selections] == [
        "cortex_m4.conv2d_3x3_im2col_smlad.v1",
        "cortex_m4.max_pool2d_2x2_s2.v1",
        "portable.flatten_view.v1",
        "portable.linear_s8.v1",
    ]
    assert tiny_backend.scratch_size == 20
    assert tiny_backend.scratch_alignment >= 4

    mobile = bakenn.compile(
        mobilenet_v1_graph(),
        tmp_path / "mobile",
        backend_options=options,
        target=CORTEX_M4,
    )
    mobile_backend = select_backend_plan(mobile.plan, options)
    assert [selection.kernel_id for selection in mobile_backend.selections[:3]] == [
        "cortex_m4.depthwise_3x3_smlad.v1",
        "cortex_m4.conv2d_1x1_smlad.v1",
        "cortex_m4.global_average_pool2d_s8.v1",
    ]

    linear = bakenn.compile(
        linear_graph(13, 5),
        tmp_path / "linear",
        backend_options=options,
        target=CORTEX_M4,
    )
    linear_backend = select_backend_plan(linear.plan, options)
    assert linear_backend.selections[0].kernel_id == "cortex_m4.linear_smlad.v1"


def test_cortex_m4_packed_i16_lanes_preserve_every_signed_weight(tmp_path: Path) -> None:
    graph = linear_graph(13, 5)
    options = _options()
    compiled = bakenn.compile(graph, tmp_path, backend_options=options, target=CORTEX_M4)
    backend = select_backend_plan(compiled.plan, options)
    packed = next(iter(backend.packed_constants.values())).value
    semantic = compiled.plan.constants["weight"]
    for channel in range(semantic.shape[0]):
        for index in range(semantic.shape[1]):
            assert _decode_lane(packed[channel, index // 2], index % 2) == int(
                semantic[channel, index]
            )
        assert _decode_lane(packed[channel, -1], 1) == 0


def test_cortex_m4_freestanding_elf_contains_real_smlad_instruction(tmp_path: Path) -> None:
    compiler = shutil.which("arm-none-eabi-gcc")
    objdump = shutil.which("arm-none-eabi-objdump")
    if compiler is None or objdump is None:
        if __import__("os").environ.get("BAKENN_REQUIRE_ARM_CC") == "1":
            pytest.fail("ARM cross compiler/objdump is required by CI")
        pytest.skip("ARM cross compiler/objdump is unavailable")
    options = _options()
    compiled = bakenn.compile(
        mobilenet_v1_graph(),
        tmp_path / "generated",
        backend_options=options,
        target=CORTEX_M4,
    )
    report = build_freestanding_elf(
        compiled.artifacts,
        CORTEX_M4,
        tmp_path / "elf",
        compiler=compiler,
    )
    disassembly = subprocess.run(
        [objdump, "-d", str(report.elf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "smlad" in disassembly.lower()
    assert report.undefined_symbols == ()
    assert report.forbidden_symbols == ()
