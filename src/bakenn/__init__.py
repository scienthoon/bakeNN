"""Public API for the independent BakeNN INT8 AOT compiler core."""

from ._version import VERSION as __version__
from .compiler import (
    CompiledModel,
    PTQCompiledModel,
    compile,
    compile_tflite,
    compile_torch_ptq,
)
from .artifacts import (
    ARITHMETIC_PROFILE_VERSION,
    GENERATED_C_ABI_VERSION,
    MANIFEST_SCHEMA_VERSION,
    load_manifest,
)
from .backend.portable_c import (
    CBackendOptions,
    KernelPolicy,
    canonical_workload_key,
)
from .errors import BakeNNError, CompileError, Diagnostic, GraphValidationError
from .frontends.tflite import import_tflite
from .quantization.ptq import FloatLinear, FloatMLP, quantize_ptq
from .quantization.ptq_graph import (
    LinearWeightGranularity,
    PTQOptions,
    quantize_float_graph,
    quantize_float_graph_with_report,
)
from .quantization.report import CalibrationEdgeReport, CalibrationReport, PTQResult
from .quantization.verification import (
    LayerErrorReport,
    PTQVerificationReport,
    verify_ptq_accuracy,
)
from .reference import (
    dequantize_output,
    quantize_input,
    run_reference,
    run_reference_trace,
)
from .reporting import MemoryReport, build_memory_report
from .targets import (
    CORTEX_M0PLUS,
    CORTEX_M4,
    ESP32,
    ESP32_C3,
    ESP32_S3,
    KernelCostMeasurement,
    PORTABLE_32,
    RV32IMC,
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
    "CompiledModel",
    "CORTEX_M0PLUS",
    "CORTEX_M4",
    "CBackendOptions",
    "ARITHMETIC_PROFILE_VERSION",
    "CalibrationEdgeReport",
    "CalibrationReport",
    "Diagnostic",
    "FloatLinear",
    "FloatMLP",
    "GraphValidationError",
    "GENERATED_C_ABI_VERSION",
    "ESPIDFProject",
    "ESP32",
    "ESP32_C3",
    "ESP32_S3",
    "KernelPolicy",
    "KernelCostMeasurement",
    "LayerErrorReport",
    "LinearWeightGranularity",
    "MemoryReport",
    "MANIFEST_SCHEMA_VERSION",
    "BakeNNError",
    "PTQCompiledModel",
    "PTQOptions",
    "PTQResult",
    "PTQVerificationReport",
    "PORTABLE_32",
    "RV32IMC",
    "TARGET_PROFILES",
    "TargetArchitecture",
    "TargetBuildReport",
    "TargetDescriptor",
    "ZephyrProject",
    "build_freestanding_elf",
    "build_memory_report",
    "canonical_workload_key",
    "compile",
    "compile_tflite",
    "compile_torch_ptq",
    "dequantize_output",
    "export_esp_idf_component",
    "export_esp_idf_project",
    "export_zephyr_project",
    "load_manifest",
    "import_tflite",
    "quantize_input",
    "quantize_float_graph",
    "quantize_float_graph_with_report",
    "quantize_ptq",
    "run_reference",
    "run_reference_trace",
    "resolve_target",
    "verify_ptq_accuracy",
]
