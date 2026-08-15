from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.ir import (
    DType,
    Layout,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)
from bakenn.targets import CORTEX_M4, PORTABLE_32, build_freestanding_elf
from benchmarks.tflm_compare.model_fixtures import cmsis_mlp_graph

from .support import require_compiler


def _options(*, target=CORTEX_M4, packing: bool = True) -> bakenn.CBackendOptions:  # type: ignore[no-untyped-def]
    return bakenn.CBackendOptions(
        kernel_policy=bakenn.KernelPolicy.AUTO,
        enable_weight_packing=packing,
        enable_cmsis_nn=True,
        target=target,
    )


def _runner_source(
    portable: bakenn.compiler.CompiledModel,
    cmsis: bakenn.compiler.CompiledModel,
) -> str:
    portable_manifest = json.loads(portable.artifacts.manifest.read_text(encoding="utf-8"))
    cmsis_manifest = json.loads(cmsis.artifacts.manifest.read_text(encoding="utf-8"))
    portable_symbol = str(portable_manifest["model"])
    cmsis_symbol = str(cmsis_manifest["model"])
    portable_macro = portable_symbol.upper()
    cmsis_macro = cmsis_symbol.upper()
    return f'''#include "{portable.artifacts.header.name}"
#include "{cmsis.artifacts.header.name}"
#include <stdio.h>

int main(void) {{
    int8_t input[{portable_macro}_INPUT_SIZE];
    int8_t portable_output[{portable_macro}_OUTPUT_SIZE];
    int8_t cmsis_output[{cmsis_macro}_OUTPUT_SIZE];
    uint8_t portable_arena[{portable_macro}_ARENA_SIZE > 0u ? {portable_macro}_ARENA_SIZE : 1u];
    uint8_t cmsis_arena[{cmsis_macro}_ARENA_SIZE > 0u ? {cmsis_macro}_ARENA_SIZE : 1u];
    while (fread(input, 1u, {portable_macro}_INPUT_SIZE, stdin) == {portable_macro}_INPUT_SIZE) {{
        {portable_symbol}_infer(portable_arena, input, portable_output);
        {cmsis_symbol}_infer(cmsis_arena, input, cmsis_output);
        if (fwrite(portable_output, 1u, {portable_macro}_OUTPUT_SIZE, stdout)
                != {portable_macro}_OUTPUT_SIZE
            || fwrite(cmsis_output, 1u, {cmsis_macro}_OUTPUT_SIZE, stdout)
                != {cmsis_macro}_OUTPUT_SIZE) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
'''


def _compile_host_pair(
    graph: QuantizedGraph,
    tmp_path: Path,
    *,
    optimization: str = "-O2",
) -> tuple[bakenn.compiler.CompiledModel, bakenn.compiler.CompiledModel, Path]:
    compiler = require_compiler(os.environ.get("CC", "cc"))
    portable = bakenn.compile(
        graph,
        tmp_path / "portable",
        model_name="cmsis_fc_portable",
    )
    cmsis = bakenn.compile(
        graph,
        tmp_path / "cmsis",
        model_name="cmsis_fc_direct",
        backend_options=_options(),
        target=CORTEX_M4,
    )
    runner = tmp_path / "runner.c"
    runner.write_text(_runner_source(portable, cmsis), encoding="utf-8")
    executable = tmp_path / "runner"
    command = [
        compiler,
        "-std=c11",
        optimization,
        "-D__GNUC_PYTHON__",
        "-D__RESTRICT=restrict",
        "-DBAKENN_CMSIS_NN_BUILTIN_MEMORY",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        *(str(path) for path in (
            portable.artifacts.model_source,
            portable.artifacts.weights_source,
            portable.artifacts.kernels_source,
            cmsis.artifacts.model_source,
            cmsis.artifacts.weights_source,
            cmsis.artifacts.kernels_source,
            *cmsis.artifacts.support_sources,
            runner,
        )),
        "-I",
        str(portable.artifacts.output_dir),
        "-I",
        str(cmsis.artifacts.output_dir),
        *(
            flag
            for include_dir in cmsis.artifacts.support_include_dirs
            for flag in ("-I", str(include_dir))
        ),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return portable, cmsis, executable


@pytest.mark.parametrize(
    ("widths", "random_count"),
    [
        ((32, 16, 4), 10_000),
        ((31, 15, 7), 2_048),
        ((784, 64, 10), 128),
    ],
)
def test_cmsis_nn_fc_multiple_mlp_shapes_are_byte_exact(
    tmp_path: Path,
    widths: tuple[int, ...],
    random_count: int,
) -> None:
    graph = cmsis_mlp_graph(widths)
    portable, cmsis, executable = _compile_host_pair(graph, tmp_path)
    assert all(
        selection.kernel_id == "cmsis_nn.linear_s8.v4.0.0"
        for selection in cmsis.artifacts.backend_plan.selections
    )
    rng = np.random.default_rng(sum(widths))
    inputs = rng.integers(
        -128, 128, size=(random_count, widths[0]), dtype=np.int16
    ).astype(np.int8)
    inputs[:4] = np.asarray(
        [
            [-128] * widths[0],
            [127] * widths[0],
            [-3] * widths[0],
            ((np.arange(widths[0]) * 37) % 256 - 128).astype(np.int8),
        ],
        dtype=np.int8,
    )
    result = subprocess.run(
        executable,
        input=inputs.tobytes(),
        capture_output=True,
        check=True,
    )
    output_count = widths[-1]
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(
        random_count, 2, output_count
    )
    np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
    reference_count = min(random_count, 512)
    expected = np.concatenate(
        [
            bakenn.run_reference(
                portable.plan,
                row.reshape(1, widths[0]),
            )
            for row in inputs[:reference_count]
        ],
        axis=0,
    )
    np.testing.assert_array_equal(actual[:reference_count, 0], expected)


def test_cmsis_nn_fc_positive_requant_shift_is_byte_exact(tmp_path: Path) -> None:
    input_qparams = PerTensorQParams(0.5, -3)
    weight_qparams = PerAxisQParams((0.5,) * 6, (0,) * 6, 0)
    bias_qparams = PerAxisQParams((0.25,) * 6, (0,) * 6, 0)
    output_qparams = PerTensorQParams(0.125, 4)
    weight = ((np.arange(48, dtype=np.int32) * 5) % 9 - 4).reshape(6, 8).astype(
        np.int8
    )
    bias = np.asarray((0, 1, -1, 3, -3, 5), dtype=np.int32)
    graph = QuantizedGraph(
        name="cmsis_positive_shift",
        values={
            "input": TensorType((1, 8), DType.INT8, Layout.NC, input_qparams),
            "weight": TensorType((6, 8), DType.INT8, Layout.OI, weight_qparams),
            "bias": TensorType((6,), DType.INT32, Layout.C, bias_qparams),
            "output": TensorType((1, 6), DType.INT8, Layout.NC, output_qparams),
        },
        constants={"weight": weight, "bias": bias},
        ops=(LinearOp("linear", "input", "weight", "bias", "output"),),
        inputs=("input",),
        outputs=("output",),
    )
    portable, cmsis, executable = _compile_host_pair(graph, tmp_path)
    step = cmsis.plan.steps[0]
    assert all(shift > 0 for shift in step.shifts)  # type: ignore[attr-defined]
    values = np.arange(256 * 8, dtype=np.int32).reshape(256, 8)
    inputs = ((values * 73 + 19) % 256 - 128).astype(np.int8)
    result = subprocess.run(
        executable,
        input=inputs.tobytes(),
        capture_output=True,
        check=True,
    )
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(256, 2, 6)
    np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
    expected = np.concatenate(
        [bakenn.run_reference(portable.plan, row.reshape(1, 8)) for row in inputs],
        axis=0,
    )
    np.testing.assert_array_equal(actual[:, 0], expected)


def test_cmsis_nn_fc_bundle_is_pinned_self_contained_and_deterministic(
    tmp_path: Path,
) -> None:
    graph = cmsis_mlp_graph((32, 16, 4))
    first = bakenn.compile(
        graph,
        tmp_path / "first",
        backend_options=_options(),
        target=CORTEX_M4,
    )
    second = bakenn.compile(
        graph,
        tmp_path / "second",
        backend_options=_options(),
        target=CORTEX_M4,
    )
    assert len(first.artifacts.support_sources) == 3
    assert len(first.artifacts.support_include_dirs) == 2
    assert len(first.artifacts.third_party_licenses) == 2
    assert all(path.is_file() for path in first.artifacts.support_sources)
    assert all(path.is_file() for path in first.artifacts.third_party_licenses)
    build_fragment = first.artifacts.build_fragment.read_text(encoding="utf-8")
    assert "bakenn_cmsis_memory.c" in build_fragment
    assert "arm_fully_connected_s8.c" in build_fragment
    assert "BAKENN_MODEL_COMPILE_DEFINITIONS" in build_fragment
    assert "BAKENN_CMSIS_NN_BUILTIN_MEMORY" in build_fragment
    manifest = json.loads(first.artifacts.manifest.read_text(encoding="utf-8"))
    dependency = manifest["bundled_dependencies"][0]
    assert dependency["name"] == "CMSIS-NN"
    assert dependency["version"] == "4.0.0"
    assert dependency["revision"] == "ca5dc34313be2ee5c46652917c30baac96c52621"
    relative_files = (
        first.artifacts.model_source.name,
        first.artifacts.weights_source.name,
        first.artifacts.kernels_source.name,
        "bakenn_sources.cmake",
        *(
            path.relative_to(first.artifacts.output_dir).as_posix()
            for path in first.artifacts.support_sources
        ),
    )
    for relative in relative_files:
        first_bytes = (first.artifacts.output_dir / relative).read_bytes()
        second_bytes = (second.artifacts.output_dir / relative).read_bytes()
        assert first_bytes == second_bytes


def test_cmsis_nn_fc_falls_back_for_incompatible_qparams_and_target(
    tmp_path: Path,
) -> None:
    from .test_backend_selection import linear_graph

    graph = linear_graph(12, 6)
    m4 = bakenn.compile(
        graph,
        tmp_path / "m4",
        backend_options=_options(packing=False),
        target=CORTEX_M4,
    )
    selection = m4.artifacts.backend_plan.selections[0]
    assert selection.kernel_id == "portable.linear_s8.v1"
    assert "per-output-channel requantization" in selection.rejected[
        "cmsis_nn.linear_s8.v4.0.0"
    ]
    portable = bakenn.compile(
        cmsis_mlp_graph((12, 6)),
        tmp_path / "portable_target",
        backend_options=_options(target=PORTABLE_32, packing=False),
        target=PORTABLE_32,
    )
    portable_selection = portable.artifacts.backend_plan.selections[0]
    assert portable_selection.kernel_id == "portable.linear_s8.v1"
    assert "ARMv7E-M DSP target" in portable_selection.rejected[
        "cmsis_nn.linear_s8.v4.0.0"
    ]
    assert not portable.artifacts.support_sources


def test_cmsis_nn_fc_cross_links_without_tflm_or_unresolved_symbols(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("arm-none-eabi-gcc")
    objdump = shutil.which("arm-none-eabi-objdump")
    if compiler is None or objdump is None:
        if os.environ.get("BAKENN_REQUIRE_ARM_CC") == "1":
            pytest.fail("ARM cross compiler and objdump are required by CI")
        pytest.skip("ARM cross compiler or objdump is unavailable")
    compiled = bakenn.compile(
        cmsis_mlp_graph((32, 16, 4)),
        tmp_path / "generated",
        backend_options=_options(),
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
    ).stdout.lower()
    assert "arm_fully_connected_s8" in disassembly
    assert "smlad" in disassembly
    assert report.undefined_symbols == ()
    assert report.forbidden_symbols == ()


def test_torch_ptq_selects_cmsis_compatible_linear_quantization(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class MnistMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(784, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )

        def forward(self, value):  # type: ignore[no-untyped-def]
            return self.layers(value)

    torch.manual_seed(447626)
    model = MnistMLP().eval()
    example = torch.zeros(1, 784)
    calibration = torch.randn(8, 784)
    compiled = bakenn.compile_torch_ptq(
        model,
        example,
        calibration,
        tmp_path,
        name="mnist_mlp_cmsis",
        backend_options=_options(),
        target=CORTEX_M4,
    )
    linear_selections = [
        selection
        for selection in compiled.artifacts.backend_plan.selections
        if selection.step_name.startswith("linear")
    ]
    assert len(linear_selections) == 2
    assert all(
        selection.kernel_id == "cmsis_nn.linear_s8.v4.0.0"
        for selection in linear_selections
    )
    for op in compiled.graph.ops:
        if isinstance(op, LinearOp):
            qparams = compiled.graph.values[op.weight].qparams
            assert isinstance(qparams, PerAxisQParams)
            assert len(set(qparams.scales)) == 1


def test_cmsis_opt_in_does_not_change_ptq_when_policy_forces_portable(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    model = torch.nn.Linear(16, 8).eval()
    with torch.no_grad():
        for channel in range(8):
            model.weight[channel].mul_(float(channel + 1))
    compiled = bakenn.compile_torch_ptq(
        model,
        torch.zeros(1, 16),
        torch.randn(4, 16),
        tmp_path,
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.PORTABLE,
            enable_cmsis_nn=True,
            target=CORTEX_M4,
        ),
        target=CORTEX_M4,
    )
    linear = next(op for op in compiled.graph.ops if isinstance(op, LinearOp))
    qparams = compiled.graph.values[linear.weight].qparams
    assert isinstance(qparams, PerAxisQParams)
    assert len(set(qparams.scales)) > 1
    assert compiled.artifacts.backend_plan.selections[0].kernel_id == (
        "portable.linear_s8.v1"
    )
    assert not compiled.artifacts.support_sources
