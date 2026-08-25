"""Portable C11 code generation backend."""

from .contracts import (
    ConstantEmission,
    KernelEmission,
    StepEmitContext,
    StepEmission,
    emit_step,
)
from .generator import CompilationArtifacts, generate_portable_c
from .selection import (
    CBackendOptions,
    CBackendPlan,
    KernelCapability,
    KernelPolicy,
    KernelSelection,
    PackedConstant,
    canonical_workload_key,
    kernel_capabilities,
    select_backend_plan,
)

__all__ = [
    "CompilationArtifacts",
    "CBackendOptions",
    "CBackendPlan",
    "ConstantEmission",
    "KernelEmission",
    "KernelCapability",
    "KernelPolicy",
    "KernelSelection",
    "PackedConstant",
    "StepEmitContext",
    "StepEmission",
    "emit_step",
    "generate_portable_c",
    "canonical_workload_key",
    "kernel_capabilities",
    "select_backend_plan",
]
