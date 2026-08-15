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


@dataclass(frozen=True)
class ESPIDFProject:
    root: Path
    component: Path
    main: Path
    target: TargetDescriptor
    model_symbol: str


def _load_manifest(artifacts: "CompilationArtifacts") -> dict[str, object]:
    return json.loads(artifacts.manifest.read_text(encoding="utf-8"))


def _require_esp_target(
    artifacts: "CompilationArtifacts", target: str | TargetDescriptor
) -> tuple[TargetDescriptor, dict[str, object]]:
    descriptor = resolve_target(target)
    if descriptor.toolchain != "esp-idf" or "idf_target" not in descriptor.metadata:
        raise CompileError(f"target {descriptor.target_id} is not an ESP-IDF target")
    manifest = _load_manifest(artifacts)
    artifact_target = manifest["backend"]["target"]["id"]  # type: ignore[index]
    if artifact_target != descriptor.target_id:
        raise CompileError(
            f"artifact target {artifact_target} does not match ESP-IDF target "
            f"{descriptor.target_id}; recompile with target={descriptor.target_id!r}"
        )
    return descriptor, manifest


def export_esp_idf_component(
    artifacts: "CompilationArtifacts",
    target: str | TargetDescriptor,
    output_dir: str | Path,
) -> Path:
    """Package generated model and pinned support sources as an ESP-IDF component."""

    descriptor, _ = _require_esp_target(artifacts, target)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = (
        artifacts.header,
        artifacts.model_source,
        artifacts.weights_header,
        artifacts.weights_source,
        artifacts.kernels_header,
        artifacts.kernels_source,
        artifacts.manifest,
    )
    for source in files:
        shutil.copy2(source, output / source.name)
    generated_source_names = (
        artifacts.model_source.name,
        artifacts.weights_source.name,
        artifacts.kernels_source.name,
    )
    support_source_names: list[str] = []
    for source in (*artifacts.support_sources, *artifacts.third_party_licenses):
        try:
            relative = source.relative_to(artifacts.output_dir)
        except ValueError as error:
            raise CompileError(
                f"support file is outside the generated artifact: {source}"
            ) from error
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if source in artifacts.support_sources:
            support_source_names.append(relative.as_posix())
    include_names = ["."]
    for include_dir in artifacts.support_include_dirs:
        try:
            relative = include_dir.relative_to(artifacts.output_dir)
        except ValueError as error:
            raise CompileError(
                f"support include directory is outside the generated artifact: {include_dir}"
            ) from error
        include_names.append(relative.as_posix())
        shutil.copytree(include_dir, output / relative, dirs_exist_ok=True)

    cmake_sources = " ".join(
        f'"{name}"' for name in (*generated_source_names, *support_source_names)
    )
    cmake_includes = " ".join(f'"{name}"' for name in include_names)
    strict_generated = " ".join(f'"{name}"' for name in generated_source_names)
    manifest = _load_manifest(artifacts)
    dependencies = manifest.get("bundled_dependencies", [])
    has_esp_nn = any(
        isinstance(item, dict) and item.get("name") == "ESP-NN"
        for item in dependencies
    )
    idf_target_macro = {
        "esp32": "CONFIG_IDF_TARGET_ESP32",
        "esp32s3": "CONFIG_IDF_TARGET_ESP32S3",
        "esp32c3": "CONFIG_IDF_TARGET_ESP32C3",
    }[str(descriptor.metadata["idf_target"])]
    definitions = (
        "target_compile_definitions(${COMPONENT_LIB} PRIVATE "
        f"CONFIG_NN_OPTIMIZED=1 {idf_target_macro}=1)\n"
        if has_esp_nn
        else ""
    )
    target_options = (
        "if(CONFIG_IDF_TARGET_ESP32S3)\n"
        "  target_compile_options(${COMPONENT_LIB} PRIVATE -mlongcalls "
        "-fno-unroll-loops -O2 -Wno-unused-function)\n"
        "else()\n"
        "  target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-function)\n"
        "endif()\n"
        if has_esp_nn
        else ""
    )
    (output / "CMakeLists.txt").write_text(
        f"idf_component_register(SRCS {cmake_sources} INCLUDE_DIRS {cmake_includes})\n"
        "target_compile_options(${COMPONENT_LIB} PRIVATE "
        "-std=c11 -ffunction-sections -fdata-sections)\n"
        f"set_source_files_properties({strict_generated} PROPERTIES "
        'COMPILE_OPTIONS "-Wall;-Wextra;-Werror;-pedantic")\n'
        + definitions
        + target_options,
        encoding="utf-8",
    )
    return output


def _main_source(symbol: str) -> str:
    macro = symbol.upper()
    return f'''#include "{symbol}.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "sdkconfig.h"
#include "esp_cpu.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

_Alignas({macro}_ARENA_ALIGNMENT)
static uint8_t model_arena[{macro}_ARENA_SIZE > 0u ? {macro}_ARENA_SIZE : 1u];
static int8_t model_input[{macro}_INPUT_SIZE];
static int8_t model_output[{macro}_OUTPUT_SIZE];

void app_main(void) {{
    uint8_t *arena = {macro}_ARENA_SIZE > 0u ? model_arena : NULL;
    const uint32_t start = esp_cpu_get_cycle_count();
    {symbol}_infer(arena, model_input, model_output);
    const uint32_t cycles = esp_cpu_get_cycle_count() - start;
    const UBaseType_t stack_words_free = uxTaskGetStackHighWaterMark(NULL);

    printf("BAKENN target=%s cycles=%" PRIu32
           " stack_high_water_words=%" PRIu32 " arena=%u\\n",
           CONFIG_IDF_TARGET, cycles, (uint32_t)stack_words_free,
           (unsigned){macro}_ARENA_SIZE);
    printf("BAKENN_OUTPUT");
    for (uint32_t index = 0; index < {macro}_OUTPUT_SIZE; ++index) {{
        printf(" %d", (int)model_output[index]);
    }}
    printf("\\n");
}}
'''


def export_esp_idf_project(
    artifacts: "CompilationArtifacts",
    target: str | TargetDescriptor,
    output_dir: str | Path,
) -> ESPIDFProject:
    """Create a buildable ESP-IDF smoke project; no physical board is required."""

    descriptor, manifest = _require_esp_target(artifacts, target)
    symbol = str(manifest["model"])
    root = Path(output_dir)
    component = root / "components" / "bakenn_model"
    main = root / "main"
    component.mkdir(parents=True, exist_ok=True)
    main.mkdir(parents=True, exist_ok=True)
    export_esp_idf_component(artifacts, descriptor, component)
    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "include($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
        "project(bakenn_target_smoke)\n",
        encoding="utf-8",
    )
    (root / "sdkconfig.defaults").write_text(
        "CONFIG_COMPILER_OPTIMIZATION_SIZE=y\n"
        "CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192\n",
        encoding="utf-8",
    )
    (main / "CMakeLists.txt").write_text(
        'idf_component_register(SRCS "main.c" INCLUDE_DIRS "." REQUIRES bakenn_model)\n',
        encoding="utf-8",
    )
    (main / "main.c").write_text(_main_source(symbol), encoding="utf-8")
    (root / "bakenn_target.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "idf_target": descriptor.metadata["idf_target"],
                "target": descriptor.manifest(),
                "model_manifest": f"components/bakenn_model/{artifacts.manifest.name}",
                "execution_metrics": {
                    "cycles": "printed by the physical-board runner; unmeasured at package time",
                    "stack": "FreeRTOS high-water mark printed by the physical-board runner",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ESPIDFProject(root, component, main, descriptor, symbol)


__all__ = ["ESPIDFProject", "export_esp_idf_component", "export_esp_idf_project"]
