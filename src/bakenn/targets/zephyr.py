from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from bakenn.errors import CompileError

from .model import TargetDescriptor
from .profiles import resolve_target

if TYPE_CHECKING:
    from bakenn.backend.portable_c import CompilationArtifacts


_SUPPORTED_BOARDS = {
    "nrf52840dk_nrf52840": "nrf52840dk",
    "nrf52dk_nrf52832": "nrf52dk",
    "disco_l475_iot1": "st-iotnode",
}


@dataclass(frozen=True)
class ZephyrProject:
    root: Path
    source: Path
    generated: Path
    target: TargetDescriptor
    board: str
    iotlab_architecture: str
    model_symbol: str


def _load_manifest(artifacts: "CompilationArtifacts") -> dict[str, object]:
    return json.loads(artifacts.manifest.read_text(encoding="utf-8"))


def _require_cortex_m4(
    artifacts: "CompilationArtifacts", target: str | TargetDescriptor, board: str
) -> tuple[TargetDescriptor, dict[str, object]]:
    descriptor = resolve_target(target)
    if descriptor.target_id != "cortex-m4" or "dsp" not in descriptor.features:
        raise CompileError(
            f"Zephyr IoT-LAB runner requires the cortex-m4 DSP target, got "
            f"{descriptor.target_id}"
        )
    if board not in _SUPPORTED_BOARDS:
        choices = ", ".join(sorted(_SUPPORTED_BOARDS))
        raise CompileError(f"unsupported Zephyr IoT-LAB board {board!r}; expected: {choices}")
    manifest = _load_manifest(artifacts)
    artifact_target = manifest["backend"]["target"]["id"]  # type: ignore[index]
    if artifact_target != descriptor.target_id:
        raise CompileError(
            f"artifact target {artifact_target} does not match Zephyr target "
            f"{descriptor.target_id}; recompile with target={descriptor.target_id!r}"
        )
    return descriptor, manifest


def _main_source(symbol: str, board: str) -> str:
    macro = symbol.upper()
    return f'''#include "{symbol}.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__has_include)
#if __has_include(<zephyr/kernel.h>)
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/timing/timing.h>
#else
#include <kernel.h>
#include <sys/printk.h>
#include <timing/timing.h>
#endif
#else
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/timing/timing.h>
#endif

#define BAKENN_BENCHMARK_RUNS 101u
#define BAKENN_WARMUP_RUNS 8u

_Alignas({macro}_ARENA_ALIGNMENT)
static uint8_t model_arena[{macro}_ARENA_SIZE > 0u ? {macro}_ARENA_SIZE : 1u];
static int8_t model_input[{macro}_INPUT_SIZE];
static int8_t model_output[{macro}_OUTPUT_SIZE];
static uint64_t measured_cycles[BAKENN_BENCHMARK_RUNS];

static void sort_cycles(uint64_t *values, size_t count) {{
    for (size_t index = 1u; index < count; ++index) {{
        const uint64_t value = values[index];
        size_t position = index;
        while (position > 0u && values[position - 1u] > value) {{
            values[position] = values[position - 1u];
            --position;
        }}
        values[position] = value;
    }}
}}

int main(void) {{
    uint8_t *arena = {macro}_ARENA_SIZE > 0u ? model_arena : NULL;
    for (size_t index = 0u; index < {macro}_INPUT_SIZE; ++index) {{
        model_input[index] = (int8_t){macro}_INPUT_ZERO_POINT;
    }}

    timing_init();
    timing_start();
    timing_t start = timing_counter_get();
    {symbol}_infer(arena, model_input, model_output);
    timing_t end = timing_counter_get();
    const uint64_t first_cycles = timing_cycles_get(&start, &end);

    for (size_t run = 0u; run < BAKENN_WARMUP_RUNS; ++run) {{
        {symbol}_infer(arena, model_input, model_output);
    }}
    for (size_t run = 0u; run < BAKENN_BENCHMARK_RUNS; ++run) {{
        start = timing_counter_get();
        {symbol}_infer(arena, model_input, model_output);
        end = timing_counter_get();
        measured_cycles[run] = timing_cycles_get(&start, &end);
    }}
    timing_stop();
    sort_cycles(measured_cycles, BAKENN_BENCHMARK_RUNS);

    size_t stack_unused = 0u;
    const int stack_status = k_thread_stack_space_get(k_current_get(), &stack_unused);
    printk("BAKENN target=nrf52840dk board={board} runs=%u first_cycles=%" PRIu64
           " median_cycles=%" PRIu64 " p95_cycles=%" PRIu64
           " arena_bytes=%u stack_unused_bytes=%zu stack_status=%d\\n",
           BAKENN_BENCHMARK_RUNS, first_cycles, measured_cycles[50],
           measured_cycles[95], (unsigned){macro}_ARENA_SIZE, stack_unused,
           stack_status);
    uint32_t output_checksum = 2166136261u;
    for (size_t index = 0u; index < {macro}_OUTPUT_SIZE; ++index) {{
        output_checksum ^= (uint8_t)model_output[index];
        output_checksum *= 16777619u;
    }}
    printk("BAKENN_OUTPUT_FNV1A=0x%08x first", output_checksum);
    const size_t preview = {macro}_OUTPUT_SIZE < 8u ? {macro}_OUTPUT_SIZE : 8u;
    for (size_t index = 0u; index < preview; ++index) {{
        printk(" %d", (int)model_output[index]);
    }}
    printk("\\n");
    return 0;
}}
'''


def export_zephyr_project(
    artifacts: "CompilationArtifacts",
    target: str | TargetDescriptor,
    output_dir: str | Path,
    *,
    board: str = "nrf52840dk_nrf52840",
) -> ZephyrProject:
    """Create a self-contained Zephyr benchmark project for an IoT-LAB M4 board."""

    descriptor, manifest = _require_cortex_m4(artifacts, target, board)
    symbol = str(manifest["model"])
    root = Path(output_dir)
    source = root / "src"
    generated = source / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for artifact in (
        artifacts.header,
        artifacts.model_source,
        artifacts.weights_header,
        artifacts.weights_source,
        artifacts.kernels_header,
        artifacts.kernels_source,
        artifacts.manifest,
        artifacts.build_fragment,
    ):
        shutil.copy2(artifact, generated / artifact.name)
    if artifacts.support_sources:
        shutil.copytree(
            artifacts.output_dir / "third_party",
            generated / "third_party",
            dirs_exist_ok=True,
        )

    support_sources = "".join(
        f"  src/generated/{path.relative_to(artifacts.output_dir).as_posix()}\n"
        for path in artifacts.support_sources
    )
    support_includes = "".join(
        f"  src/generated/{path.relative_to(artifacts.output_dir).as_posix()}\n"
        for path in artifacts.support_include_dirs
    )

    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20.0)\n"
        + "find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})\n"
        + "project(bakenn_iotlab_benchmark C)\n"
        + "target_sources(app PRIVATE\n"
        + "  src/main.c\n"
        + f"  src/generated/{artifacts.model_source.name}\n"
        + f"  src/generated/{artifacts.weights_source.name}\n"
        + f"  src/generated/{artifacts.kernels_source.name}\n"
        + support_sources
        + ")\n"
        + "target_include_directories(app PRIVATE\n"
        + "  src/generated\n"
        + support_includes
        + ")\n"
        + (
            "target_compile_definitions(app PRIVATE "
            "BAKENN_CMSIS_NN_BUILTIN_MEMORY)\n"
            if artifacts.support_sources
            else ""
        )
        + "target_compile_options(app PRIVATE -Wall -Wextra -Werror)\n",
        encoding="utf-8",
    )
    (root / "prj.conf").write_text(
        "CONFIG_CONSOLE=y\n"
        "CONFIG_UART_CONSOLE=y\n"
        "CONFIG_SERIAL=y\n"
        "CONFIG_PRINTK=y\n"
        "CONFIG_TIMING_FUNCTIONS=y\n"
        "CONFIG_THREAD_STACK_INFO=y\n"
        "CONFIG_INIT_STACKS=y\n"
        "CONFIG_MAIN_STACK_SIZE=4096\n"
        "CONFIG_SPEED_OPTIMIZATIONS=y\n",
        encoding="utf-8",
    )
    (source / "main.c").write_text(_main_source(symbol, board), encoding="utf-8")
    (root / "bakenn_target.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "zephyr_board": board,
                "iotlab_architecture": _SUPPORTED_BOARDS[board],
                "target": descriptor.manifest(),
                "model_manifest": f"src/generated/{artifacts.manifest.name}",
                "benchmark": {
                    "input": "all elements set to the model input zero-point",
                    "warmup_runs": 8,
                    "measured_runs": 101,
                    "percentiles": "sorted exact samples; median index 50, p95 index 95",
                    "cycle_source": "Zephyr timing API on the physical board",
                    "stack": "current-thread unused bytes reported by Zephyr",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ZephyrProject(
        root,
        source,
        generated,
        descriptor,
        board,
        _SUPPORTED_BOARDS[board],
        symbol,
    )


__all__ = ["ZephyrProject", "export_zephyr_project"]
