from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from bakenn.errors import CompileError
from bakenn.ir import DType, Layout, PerTensorQParams, TensorType
from bakenn.plan import ExecutionPlan, PlanTensor, Storage
from bakenn.backend.portable_c import (
    KernelEmission,
    StepEmitContext,
    StepEmission,
    emit_step,
    generate_portable_c,
)


@dataclass(frozen=True)
class _UnsupportedStep:
    name: str = "unsupported"
    kernel_kind: str = "unsupported"
    arithmetic_profile: str = "test.v1"
    inputs: tuple[str, ...] = ("input",)
    outputs: tuple[str, ...] = ("output",)
    constants: tuple[str, ...] = ()
    aliases: tuple[object, ...] = ()
    scratch_size: int = 0
    scratch_alignment: int = 1


@dataclass(frozen=True)
class _CopyStep:
    name: str = "copy"
    kernel_kind: str = "copy_s8"
    arithmetic_profile: str = "test.copy.v1"
    inputs: tuple[str, ...] = ("input",)
    outputs: tuple[str, ...] = ("output",)
    constants: tuple[str, ...] = ()
    aliases: tuple[object, ...] = ()
    scratch_size: int = 0
    scratch_alignment: int = 1


def _plan(step: object) -> ExecutionPlan:
    tensor_type = TensorType((1, 2), DType.INT8, Layout.NC, PerTensorQParams(0.25, -3))
    return ExecutionPlan(
        name="extension",
        tensors={
            "input": PlanTensor("input", tensor_type, Storage.INPUT),
            "output": PlanTensor("output", tensor_type, Storage.OUTPUT),
        },
        constants={},
        steps=(step,),
        inputs=("input",),
        outputs=("output",),
        arena_size=0,
        arena_alignment=16,
        arithmetic_profile="test.v1",
    )


def test_portable_c_step_dispatch_is_extensible_immutable_and_fail_closed(tmp_path) -> None:
    unsupported_plan = _plan(_UnsupportedStep())
    with pytest.raises(CompileError, match="no portable C emitter"):
        generate_portable_c(unsupported_plan, tmp_path / "unsupported")

    @emit_step.register(_CopyStep)
    def _emit_copy(step, context):  # type: ignore[no-untyped-def]
        kernel_name = f"{context.symbol}_copy_s8"
        return StepEmission(
            constants=(),
            kernels=(
                KernelEmission(
                    key="copy_s8",
                    header_includes=("<stdint.h>",),
                    declaration=(
                        f"void {kernel_name}(const int8_t *input, int8_t *output);"
                    ),
                    definition=(
                        f"void {kernel_name}(const int8_t *input, int8_t *output) {{\n"
                        "    output[0] = input[0];\n"
                        "    output[1] = input[1];\n"
                        "}"
                    ),
                ),
            ),
            call=(
                f"    {kernel_name}({context.pointer(step.inputs[0], mutable=False)}, "
                f"{context.pointer(step.outputs[0], mutable=True)});"
            ),
            manifest={"name": step.name, "kind": step.kernel_kind},
        )

    plan = _plan(_CopyStep())
    context = StepEmitContext(plan, "bknn_extension", 0, {})
    emission = emit_step(plan.steps[0], context)
    with pytest.raises(TypeError):
        emission.manifest["kind"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        context.constant_symbols["new"] = "symbol"  # type: ignore[index]

    artifacts = generate_portable_c(plan, tmp_path / "copy")
    metadata = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert metadata["operations"] == [{"kind": "copy_s8", "name": "copy"}]
    assert "bknn_extension_copy_s8" in artifacts.kernels_source.read_text(encoding="utf-8")
    assert "bknn_extension_copy_s8" in artifacts.model_source.read_text(encoding="utf-8")
