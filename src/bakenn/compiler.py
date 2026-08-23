from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from bakenn.backend.portable_c import (
    CBackendOptions,
    CompilationArtifacts,
    generate_portable_c,
)
from bakenn.ir import QuantizedGraph
from bakenn.passes import deduplicate_constants, fuse_clamps, legalize_graph
from bakenn.plan import ExecutionPlan, lower_to_plan
from bakenn.targets import PORTABLE_32, TargetDescriptor, resolve_target
from bakenn.reporting import MemoryReport

if TYPE_CHECKING:
    from bakenn.frontends.torch_export import FloatGraph
    from bakenn.quantization.ptq_graph import PTQOptions
    from bakenn.quantization.report import CalibrationReport
    from bakenn.quantization.verification import PTQVerificationReport


@dataclass(frozen=True)
class CompiledModel:
    plan: ExecutionPlan
    artifacts: CompilationArtifacts

    @property
    def memory_report(self) -> MemoryReport:
        return self.artifacts.memory_report


@dataclass(frozen=True)
class PTQCompiledModel:
    """Artifacts and immutable intermediate graphs from FP32 PTQ compilation."""

    float_graph: "FloatGraph"
    graph: QuantizedGraph
    plan: ExecutionPlan
    artifacts: CompilationArtifacts
    calibration_report: "CalibrationReport"

    @property
    def memory_report(self) -> MemoryReport:
        return self.artifacts.memory_report

    def verify_accuracy(
        self, samples: object, *, max_samples: int | None = None
    ) -> "PTQVerificationReport":
        """Measure FP32-to-deployed-INT8 error on a validation corpus."""

        from bakenn.quantization.verification import verify_ptq_accuracy

        return verify_ptq_accuracy(
            self.float_graph,
            self.plan,
            samples,
            max_samples=max_samples,
        )


def compile(
    graph: QuantizedGraph,
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    backend_options: CBackendOptions | None = None,
    target: str | TargetDescriptor | None = None,
) -> CompiledModel:
    """Legalize, fuse, verify, statically plan, and emit portable C11."""
    legalized = legalize_graph(graph)
    optimized = deduplicate_constants(fuse_clamps(legalized))
    plan = lower_to_plan(optimized)
    options = CBackendOptions() if backend_options is None else backend_options
    if target is not None:
        resolved_target = resolve_target(target)
        if backend_options is not None and options.target not in (PORTABLE_32, resolved_target):
            raise ValueError(
                "target argument conflicts with backend_options.target; specify the target once"
            )
        options = replace(options, target=resolved_target)
    artifacts = generate_portable_c(
        plan,
        output_dir,
        model_name=model_name,
        options=options,
    )
    return CompiledModel(plan=plan, artifacts=artifacts)


def compile_torch_ptq(
    model: object,
    example_input: object,
    calibration_data: object,
    output_dir: str | Path,
    *,
    name: str | None = None,
    backend_options: CBackendOptions | None = None,
    ptq_options: "PTQOptions | None" = None,
    target: str | TargetDescriptor | None = None,
) -> PTQCompiledModel:
    """Capture, PTQ, legalize, statically plan, and emit one PyTorch model.

    PyTorch is imported only inside the frontend call.  The returned
    QuantizedGraph and all generated artifacts remain framework independent.
    """

    from bakenn.frontends.torch_export import capture_torch_export
    from bakenn.quantization.ptq_graph import quantize_float_graph_with_report

    float_graph = capture_torch_export(model, example_input, name=name)
    quantized = quantize_float_graph_with_report(
        float_graph,
        calibration_data,
        name=name,
        options=ptq_options,
    )
    compiled = compile(
        quantized.graph,
        output_dir,
        model_name=name,
        backend_options=backend_options,
        target=target,
    )
    return PTQCompiledModel(
        float_graph,
        quantized.graph,
        compiled.plan,
        compiled.artifacts,
        quantized.report,
    )


def compile_tflite(
    source: object,
    output_dir: str | Path,
    *,
    name: str | None = None,
    backend_options: CBackendOptions | None = None,
    target: str | TargetDescriptor | None = None,
) -> CompiledModel:
    """Import a strict fully-quantized TFLite model and emit standalone C.

    FlatBuffers and the TFLite schema are optional host-only dependencies.
    The generated target artifact never links either dependency or a TFLite
    interpreter.
    """

    from bakenn.frontends.tflite import import_tflite

    graph = import_tflite(source, name=name)
    return compile(
        graph,
        output_dir,
        model_name=name,
        backend_options=backend_options,
        target=target,
    )


__all__ = [
    "CompiledModel",
    "PTQCompiledModel",
    "compile",
    "compile_tflite",
    "compile_torch_ptq",
]
