"""Public API for the independent BakeNN INT8 AOT compiler core."""

from ._version import VERSION as __version__
from .compiler import PTQCompiledModel, compile, compile_torch_ptq
from .backend.portable_c import CBackendOptions, KernelPolicy
from .errors import CompileError, GraphValidationError, BakeNNError
from .quantization.ptq import FloatLinear, FloatMLP, quantize_ptq
from .quantization.ptq_graph import (
    LinearWeightGranularity,
    PTQOptions,
    quantize_float_graph,
)
from .reference import dequantize_output, quantize_input, run_reference
from .targets import (
    KernelCostMeasurement,
    TARGET_PROFILES,
    TargetArchitecture,
    TargetDescriptor,
    TargetBuildReport,
    build_freestanding_elf,
    ESPIDFProject,
    export_esp_idf_component,
    export_esp_idf_project,
    export_zephyr_project,
    resolve_target,
    ZephyrProject,
)

__all__ = [
    "CompileError",
    "CBackendOptions",
    "FloatLinear",
    "FloatMLP",
    "GraphValidationError",
    "ESPIDFProject",
    "KernelPolicy",
    "KernelCostMeasurement",
    "LinearWeightGranularity",
    "BakeNNError",
    "PTQCompiledModel",
    "PTQOptions",
    "TARGET_PROFILES",
    "TargetArchitecture",
    "TargetBuildReport",
    "TargetDescriptor",
    "ZephyrProject",
    "build_freestanding_elf",
    "compile",
    "compile_torch_ptq",
    "dequantize_output",
    "export_esp_idf_component",
    "export_esp_idf_project",
    "export_zephyr_project",
    "quantize_input",
    "quantize_float_graph",
    "quantize_ptq",
    "run_reference",
    "resolve_target",
]
