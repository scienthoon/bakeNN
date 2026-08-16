from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from bakenn._version import VERSION
from bakenn.backend.cmsis_nn import bundle_kernels
from bakenn.backend.cmsis_nn.bundle import (
    CMSIS_CORE_VERSION,
    CMSIS_NN_REVISION,
    CMSIS_NN_VERSION,
)
from bakenn.backend.esp_nn import bundle_kernels as bundle_esp_nn_kernels
from bakenn.backend.esp_nn.bundle import ESP_NN_REVISION, ESP_NN_VERSION
from bakenn.errors import CompileError
from bakenn.ir import PerTensorQParams
from bakenn.ir.types import TARGET_SIZE_MAX
from bakenn.plan import ExecutionPlan
from bakenn.reporting import MemoryReport, build_memory_report

# Importing the central family aggregator installs built-in singledispatch
# registrations.  Adding an op family does not require editing this generator.
from . import families as _families
from .contracts import ConstantEmission, KernelEmission, StepEmitContext, checked_emit_step
from .formatting import c_float, format_values, guard, identifier, qparams_dict
from .selection import CBackendOptions, CBackendPlan, select_backend_plan


@dataclass(frozen=True)
class CompilationArtifacts:
    output_dir: Path
    header: Path
    model_source: Path
    weights_header: Path
    weights_source: Path
    kernels_header: Path
    kernels_source: Path
    manifest: Path
    memory_report: MemoryReport
    memory_report_json: Path
    memory_report_text: Path
    backend_plan: CBackendPlan
    build_fragment: Path
    support_sources: tuple[Path, ...] = ()
    support_include_dirs: tuple[Path, ...] = ()
    third_party_licenses: tuple[Path, ...] = ()


def _plan_constant(symbol: str, array: np.ndarray, *, alignment: int = 1) -> ConstantEmission:
    if array.dtype == np.int8:
        c_type = "int8_t"
    elif array.dtype == np.int32:
        c_type = "int32_t"
    else:
        raise CompileError(f"portable C does not support constant dtype {array.dtype}")
    alignment_prefix = "" if alignment == 1 else f"_Alignas({alignment}) "
    return ConstantEmission(
        symbol=symbol,
        declaration=f"extern const {c_type} {symbol}[{array.size}];",
        definition=(
            f"{alignment_prefix}const {c_type} {symbol}[{array.size}] = "
            f"{{\n{format_values(array)}\n}};"
        ),
        size_bytes=int(array.nbytes),
        alignment=alignment,
    )


def _include_lines(includes: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"#include {value}" for value in dict.fromkeys(includes))


def generate_portable_c(
    plan: ExecutionPlan,
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    options: CBackendOptions | None = None,
) -> CompilationArtifacts:
    backend_plan = select_backend_plan(plan, options)
    symbol = "bknn_" + identifier(model_name or plan.name)
    if not symbol:
        raise CompileError("model name cannot be converted into a C identifier")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cmsis_kernel_ids = tuple(
        selection.kernel_id
        for selection in backend_plan.selections
        if selection.kernel_id.startswith("cmsis_nn.")
    )
    cmsis_bundle = (
        bundle_kernels(output, cmsis_kernel_ids) if cmsis_kernel_ids else None
    )
    esp_nn_kernel_ids = tuple(
        selection.kernel_id
        for selection in backend_plan.selections
        if selection.kernel_id.startswith("esp_nn.")
    )
    esp_nn_bundle = (
        bundle_esp_nn_kernels(
            output,
            esp_nn_kernel_ids,
            backend_plan.options.target.target_id,
        )
        if esp_nn_kernel_ids
        else None
    )

    input_type = plan.tensors[plan.inputs[0]].tensor_type
    output_type = plan.tensors[plan.outputs[0]].tensor_type
    input_qparams = input_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    header_name = f"{symbol}.h"
    weights_header_name = f"{symbol}_weights.h"
    kernels_header_name = f"{symbol}_kernels.h"
    macro = symbol.upper()

    input_dimensions = "\n".join(
        f"#define {macro}_INPUT_DIM_{index} {dimension}u"
        for index, dimension in enumerate(input_type.shape)
    )
    output_dimensions = "\n".join(
        f"#define {macro}_OUTPUT_DIM_{index} {dimension}u"
        for index, dimension in enumerate(output_type.shape)
    )

    header = output / header_name
    model_source = output / f"{symbol}.c"
    weights_header = output / weights_header_name
    weights_source = output / f"{symbol}_weights.c"
    kernels_header = output / kernels_header_name
    kernels_source = output / f"{symbol}_kernels.c"
    manifest = output / f"{symbol}_manifest.json"
    memory_report_json = output / f"{symbol}_memory.json"
    memory_report_text = output / f"{symbol}_memory.txt"
    build_fragment = output / "bakenn_sources.cmake"

    header.write_text(
        f"""#ifndef {guard(symbol)}
#define {guard(symbol)}

#include <stdint.h>

#ifndef BKNN_LAYOUT_NHWC
#define BKNN_LAYOUT_NHWC 1u
#endif
#ifndef BKNN_LAYOUT_NC
#define BKNN_LAYOUT_NC 2u
#endif

#define {macro}_INPUT_SIZE {input_type.numel}u
#define {macro}_INPUT_BYTES {input_type.nbytes}u
#define {macro}_INPUT_RANK {len(input_type.shape)}u
{input_dimensions}
#define {macro}_INPUT_LAYOUT BKNN_LAYOUT_{input_type.layout.value}
#define {macro}_OUTPUT_SIZE {output_type.numel}u
#define {macro}_OUTPUT_BYTES {output_type.nbytes}u
#define {macro}_OUTPUT_RANK {len(output_type.shape)}u
{output_dimensions}
#define {macro}_OUTPUT_LAYOUT BKNN_LAYOUT_{output_type.layout.value}
#define {macro}_ARENA_SIZE {backend_plan.arena_size}u
#define {macro}_ARENA_ALIGNMENT {backend_plan.arena_alignment}u
#define {macro}_INPUT_SCALE {c_float(input_qparams.scale)}
#define {macro}_INPUT_ZERO_POINT {input_qparams.zero_point}
#define {macro}_OUTPUT_SCALE {c_float(output_qparams.scale)}
#define {macro}_OUTPUT_ZERO_POINT {output_qparams.zero_point}

/* input, output, and arena must not overlap. arena may be NULL when ARENA_SIZE is zero. */
void {symbol}_infer(
    uint8_t *restrict arena,
    const int8_t *restrict input,
    int8_t *restrict output);

#endif
""",
        encoding="utf-8",
    )

    constant_symbols = {
        name: f"{symbol}_constant_{index}" for index, name in enumerate(plan.constants)
    }
    packed_symbols = {
        name: f"{symbol}_packed_{index}"
        for index, name in enumerate(sorted(backend_plan.packed_constants))
    }
    required_constants = {
        name
        for selection, step in zip(backend_plan.selections, plan.steps)
        for name in (*step.inputs, *step.constants)
        if name in plan.constants and name not in selection.constant_overrides
    }
    target_constant_alignment = backend_plan.options.target.constant_alignment
    constant_emissions = [
        _plan_constant(
            constant_symbols[name],
            array,
            alignment=target_constant_alignment,
        )
        for name, array in plan.constants.items()
        if name in required_constants
    ]
    constant_emissions.extend(
        _plan_constant(
            packed_symbols[name],
            backend_plan.packed_constants[name].value,
            alignment=max(
                target_constant_alignment,
                backend_plan.packed_constants[name].alignment,
            ),
        )
        for name in sorted(backend_plan.packed_constants)
    )
    step_emissions = []
    for index, (step, selection) in enumerate(zip(plan.steps, backend_plan.selections)):
        override_symbols = {
            source: packed_symbols[packed_name]
            for source, packed_name in selection.constant_overrides.items()
        }
        context = StepEmitContext(
            plan,
            symbol,
            index,
            constant_symbols,
            selection,
            override_symbols,
            backend_plan.scratch_offset,
        )
        step_emissions.append(checked_emit_step(step, context))
    for emission in step_emissions:
        constant_emissions.extend(emission.constants)

    constant_by_symbol: dict[str, ConstantEmission] = {}
    for emission in constant_emissions:
        previous = constant_by_symbol.setdefault(emission.symbol, emission)
        if previous != emission:
            raise CompileError(f"conflicting portable C constant symbol {emission.symbol}")
    constants = tuple(constant_by_symbol.values())
    constant_bytes = sum(item.size_bytes for item in constants)
    if constant_bytes > TARGET_SIZE_MAX:
        raise CompileError(
            "generated constant payload exceeds the 32-bit target byte limit"
        )
    target = backend_plan.options.target
    if target.flash_bytes is not None and constant_bytes > target.flash_bytes:
        raise CompileError(
            f"target {target.target_id} constant payload {constant_bytes} exceeds Flash budget "
            f"{target.flash_bytes}; generated code and initialized data are not included"
        )
    if target.sram_bytes is not None and backend_plan.arena_size > target.sram_bytes:
        raise CompileError(
            f"target {target.target_id} arena {backend_plan.arena_size} exceeds SRAM budget "
            f"{target.sram_bytes}; application globals and stack are not included"
        )
    constant_max_alignment = max((item.alignment for item in constants), default=1)
    memory_report = build_memory_report(
        plan,
        backend_plan,
        emitted_constant_payload_bytes=constant_bytes,
    )
    memory_report.write_json(memory_report_json)
    memory_report.write_text(memory_report_text)

    weights_header.write_text(
        f"#ifndef {guard(symbol + '_weights')}\n#define {guard(symbol + '_weights')}\n\n"
        "#include <stdint.h>\n\n"
        + "\n".join(item.declaration for item in constants)
        + "\n\n#endif\n",
        encoding="utf-8",
    )
    weights_source.write_text(
        f'#include "{weights_header_name}"\n\n'
        + "\n\n".join(item.definition for item in constants)
        + "\n",
        encoding="utf-8",
    )

    kernel_by_key: dict[str, KernelEmission] = {}
    for emission in step_emissions:
        for kernel in emission.kernels:
            previous = kernel_by_key.setdefault(kernel.key, kernel)
            if previous != kernel:
                raise CompileError(f"conflicting portable C kernel emission {kernel.key}")
    kernels = tuple(kernel_by_key.values())
    header_includes = tuple(value for kernel in kernels for value in kernel.header_includes)
    source_includes = tuple(value for kernel in kernels for value in kernel.source_includes)
    header_include_text = _include_lines(header_includes)
    source_include_text = _include_lines(source_includes)

    kernels_header.write_text(
        f"#ifndef {guard(symbol + '_kernels')}\n#define {guard(symbol + '_kernels')}\n\n"
        + (header_include_text + "\n\n" if header_include_text else "")
        + "\n\n".join(kernel.declaration for kernel in kernels)
        + "\n\n#endif\n",
        encoding="utf-8",
    )
    kernels_source.write_text(
        f'#include "{kernels_header_name}"\n\n'
        + (source_include_text + "\n\n" if source_include_text else "")
        + "\n\n".join(kernel.definition for kernel in kernels)
        + "\n",
        encoding="utf-8",
    )

    model_source.write_text(
        f'#include "{header_name}"\n#include "{kernels_header_name}"\n#include "{weights_header_name}"\n\n'
        f"void {symbol}_infer(\n"
        f"    uint8_t *restrict arena,\n"
        f"    const int8_t *restrict input,\n"
        f"    int8_t *restrict output) {{\n"
        + "    (void)arena;\n"
        + "\n\n".join(emission.call for emission in step_emissions)
        + "\n}\n",
        encoding="utf-8",
    )

    operations = [dict(emission.manifest) for emission in step_emissions]
    kernel_selections = []
    for selection in backend_plan.selections:
        kernel_selections.append(
            {
                "step_index": selection.step_index,
                "step_name": selection.step_name,
                "implementation": selection.kernel_id,
                "optimized": selection.optimized,
                "selection_reason": selection.reason,
                "packed_constants": [
                    {
                        "name": item.name,
                        "source": item.source,
                        "symbol": packed_symbols[item.name],
                        "layout": item.layout,
                        "alignment": item.alignment,
                        "bytes": int(item.value.nbytes),
                    }
                    for item in selection.packed_constants
                ],
                "rejected_implementations": dict(selection.rejected),
                "scratch_bytes": selection.scratch_size,
                "scratch_alignment": selection.scratch_alignment,
            }
        )

    manifest_data = {
        "schema_version": 3,
        "compiler_version": VERSION,
        "model": symbol,
        "arithmetic_profile": plan.arithmetic_profile,
        "backend": {
            "name": "c11",
            "target": backend_plan.options.target.manifest(),
            "kernel_policy": backend_plan.options.kernel_policy.value,
            "weight_packing": backend_plan.options.enable_weight_packing,
            "cmsis_nn_enabled": backend_plan.options.enable_cmsis_nn,
            "esp_nn_enabled": backend_plan.options.enable_esp_nn,
            "optimized_steps": sum(item.optimized for item in backend_plan.selections),
            "selections": kernel_selections,
        },
        "arena_bytes": backend_plan.arena_size,
        "activation_arena_bytes": backend_plan.activation_arena_size,
        "scratch_bytes": backend_plan.scratch_size,
        "scratch_offset": backend_plan.scratch_offset,
        "scratch_alignment": backend_plan.scratch_alignment,
        "arena_alignment": backend_plan.arena_alignment,
        "constant_bytes": constant_bytes,
        "constant_payload_bytes": constant_bytes,
        "constant_max_alignment": constant_max_alignment,
        "memory_report": {
            "schema_version": 1,
            "json": memory_report_json.name,
            "text": memory_report_text.name,
        },
        "input": {
            "shape": list(input_type.shape),
            "dtype": input_type.dtype.value,
            "layout": input_type.layout.value,
            "qparams": qparams_dict(input_type.qparams),
        },
        "output": {
            "shape": list(output_type.shape),
            "dtype": output_type.dtype.value,
            "layout": output_type.layout.value,
            "qparams": qparams_dict(output_type.qparams),
        },
        "operations": operations,
    }
    bundled_dependencies: list[dict[str, object]] = []
    if cmsis_bundle is not None:
        bundled_dependencies.append(
            {
                "name": "CMSIS-NN",
                "version": CMSIS_NN_VERSION,
                "revision": CMSIS_NN_REVISION,
                "cmsis_core_version": CMSIS_CORE_VERSION,
                "compatibility_patches": [
                    "BAKENN_CMSIS_NN_FREESTANDING suppresses unused hosted headers",
                    "BAKENN_CMSIS_NN_BUILTIN_MEMORY preserves fixed-width DSP loads",
                    "freestanding memory calls use bundled namespaced byte-loop shims",
                    "Clang excludes the GCC-only no-unroll optimize attribute",
                    "unused hosted stdio includes are removed from 1x1 Conv sources",
                ],
                "sources": [
                    path.relative_to(output).as_posix()
                    for path in cmsis_bundle.sources
                ],
                "include_dirs": [
                    path.relative_to(output).as_posix()
                    for path in cmsis_bundle.include_dirs
                ],
                "licenses": [
                    path.relative_to(output).as_posix()
                    for path in cmsis_bundle.license_files
                ],
            }
        )
    if esp_nn_bundle is not None:
        bundled_dependencies.append(
            {
                "name": "ESP-NN",
                "version": ESP_NN_VERSION,
                "revision": ESP_NN_REVISION,
                "target": esp_nn_bundle.target_id,
                "requantization": {
                    "profile": "TFLM-compatible double rounding",
                    "CONFIG_NN_SKIP_NUDGE": False,
                },
                "sources": [
                    path.relative_to(output).as_posix()
                    for path in esp_nn_bundle.sources
                ],
                "include_dirs": [
                    path.relative_to(output).as_posix()
                    for path in esp_nn_bundle.include_dirs
                ],
                "licenses": [
                    path.relative_to(output).as_posix()
                    for path in esp_nn_bundle.license_files
                ],
            }
        )
    if bundled_dependencies:
        manifest_data["bundled_dependencies"] = bundled_dependencies
    manifest.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    support_sources = (
        (() if cmsis_bundle is None else cmsis_bundle.sources)
        + (() if esp_nn_bundle is None else esp_nn_bundle.sources)
    )
    support_include_dirs = (
        (() if cmsis_bundle is None else cmsis_bundle.include_dirs)
        + (() if esp_nn_bundle is None else esp_nn_bundle.include_dirs)
    )
    third_party_licenses = (
        (() if cmsis_bundle is None else cmsis_bundle.license_files)
        + (() if esp_nn_bundle is None else esp_nn_bundle.license_files)
    )
    cmake_sources = (
        model_source,
        weights_source,
        kernels_source,
        *support_sources,
    )
    cmake_include_dirs = (output, *support_include_dirs)
    build_fragment.write_text(
        "# Generated by BakeNN; paths are relative to this file.\n"
        "set(BAKENN_MODEL_SOURCES\n"
        + "".join(
            f'  "${{CMAKE_CURRENT_LIST_DIR}}/{path.relative_to(output).as_posix()}"\n'
            for path in cmake_sources
        )
        + ")\n"
        "set(BAKENN_MODEL_INCLUDE_DIRS\n"
        + "".join(
            f'  "${{CMAKE_CURRENT_LIST_DIR}}/{path.relative_to(output).as_posix()}"\n'
            for path in cmake_include_dirs
        )
        + ")\n"
        + "set(BAKENN_MODEL_COMPILE_DEFINITIONS\n"
        + (
            "  BAKENN_CMSIS_NN_BUILTIN_MEMORY\n"
            if cmsis_bundle is not None
            else ""
        )
        + (
            "  CONFIG_NN_OPTIMIZED=1\n"
            if esp_nn_bundle is not None
            else ""
        )
        + ")\n",
        encoding="utf-8",
    )

    return CompilationArtifacts(
        output_dir=output,
        header=header,
        model_source=model_source,
        weights_header=weights_header,
        weights_source=weights_source,
        kernels_header=kernels_header,
        kernels_source=kernels_source,
        manifest=manifest,
        memory_report=memory_report,
        memory_report_json=memory_report_json,
        memory_report_text=memory_report_text,
        backend_plan=backend_plan,
        build_fragment=build_fragment,
        support_sources=support_sources,
        support_include_dirs=support_include_dirs,
        third_party_licenses=third_party_licenses,
    )


__all__ = ["CompilationArtifacts", "generate_portable_c"]
