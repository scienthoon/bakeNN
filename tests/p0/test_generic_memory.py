from dataclasses import dataclass, replace

import pytest

from bakenn.errors import CompileError
from bakenn.ir import DType, Layout, PerTensorQParams, TensorType
from bakenn.plan import AliasKind, AliasSpec, Storage, plan_memory


@dataclass(frozen=True)
class _SyntheticStep:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    constants: tuple[str, ...] = ()
    aliases: tuple[AliasSpec, ...] = ()
    scratch_size: int = 0
    scratch_alignment: int = 1
    kernel_kind: str = "test"
    arithmetic_profile: str = "test.v1"


def _tensor(features: int = 16) -> TensorType:
    return TensorType((1, features), DType.INT8, Layout.NC, PerTensorQParams(0.25, 0))


def test_diamond_keeps_both_inputs_live_through_join_and_rejects_overlap() -> None:
    values = {name: _tensor() for name in ("input", "left", "right", "joined")}
    steps = (
        _SyntheticStep("left", ("input",), ("left",)),
        _SyntheticStep("right", ("input",), ("right",)),
        _SyntheticStep("join", ("left", "right"), ("joined",)),
    )
    layout = plan_memory(
        values=values,
        constants=(),
        inputs=("input",),
        outputs=("joined",),
        steps=steps,
    )

    left = layout.tensors["left"]
    right = layout.tensors["right"]
    assert left.offset != right.offset
    assert layout.lifetimes["left"].overlaps(layout.lifetimes["right"])

    overlapping = dict(layout.tensors)
    overlapping["right"] = replace(right, offset=left.offset)
    with pytest.raises(CompileError, match="live arena buffers overlap"):
        replace(layout, tensors=overlapping)


def test_four_stage_chain_reuses_buffer_only_after_last_use() -> None:
    values = {
        "input": _tensor(8),
        "a": _tensor(9),
        "b": _tensor(17),
        "c": _tensor(9),
        "output": _tensor(3),
    }
    steps = (
        _SyntheticStep("one", ("input",), ("a",)),
        _SyntheticStep("two", ("a",), ("b",)),
        _SyntheticStep("three", ("b",), ("c",)),
        _SyntheticStep("four", ("c",), ("output",)),
    )
    layout = plan_memory(
        values=values,
        constants=(),
        inputs=("input",),
        outputs=("output",),
        steps=steps,
    )

    assert layout.tensors["a"].offset == layout.tensors["c"].offset
    assert not layout.lifetimes["a"].overlaps(layout.lifetimes["c"])
    assert layout.tensors["b"].offset != layout.tensors["a"].offset


def test_alias_groups_route_view_outputs_and_reject_unsafe_inplace() -> None:
    values = {name: _tensor() for name in ("input", "a", "view")}
    view_steps = (
        _SyntheticStep("produce", ("input",), ("a",)),
        _SyntheticStep(
            "view",
            ("a",),
            ("view",),
            aliases=(AliasSpec("view", "a", AliasKind.VIEW),),
        ),
    )
    layout = plan_memory(
        values=values,
        constants=(),
        inputs=("input",),
        outputs=("view",),
        steps=view_steps,
    )
    assert layout.tensors["a"].storage is Storage.OUTPUT
    assert layout.tensors["view"].storage is Storage.ALIAS
    assert layout.tensors["view"].alias_of == "a"
    assert layout.alias_groups["a"] == ("a", "view")
    with pytest.raises(TypeError):
        layout.alias_groups["a"] = ("a",)  # type: ignore[index]

    unsafe_values = {name: _tensor() for name in ("input", "a", "b", "output")}
    unsafe_steps = (
        _SyntheticStep("produce", ("input",), ("a",)),
        _SyntheticStep(
            "inplace",
            ("a",),
            ("b",),
            aliases=(AliasSpec("b", "a", AliasKind.INPLACE),),
        ),
        _SyntheticStep("later_consumer", ("a",), ("output",)),
    )
    with pytest.raises(CompileError, match="later consumer"):
        plan_memory(
            values=unsafe_values,
            constants=(),
            inputs=("input",),
            outputs=("output",),
            steps=unsafe_steps,
        )

    alias_chain_values = {
        name: _tensor() for name in ("input", "a", "view", "clamped", "output")
    }
    alias_chain_steps = (
        _SyntheticStep("produce", ("input",), ("a",)),
        _SyntheticStep(
            "view",
            ("a",),
            ("view",),
            aliases=(AliasSpec("view", "a", AliasKind.VIEW),),
        ),
        _SyntheticStep(
            "clamp_inplace",
            ("view",),
            ("clamped",),
            aliases=(AliasSpec("clamped", "view", AliasKind.INPLACE),),
        ),
        _SyntheticStep("join", ("clamped", "a"), ("output",)),
    )
    with pytest.raises(CompileError, match="alias group has a later consumer"):
        plan_memory(
            values=alias_chain_values,
            constants=(),
            inputs=("input",),
            outputs=("output",),
            steps=alias_chain_steps,
        )


def test_scratch_uses_max_step_requirement_after_activation_arena() -> None:
    values = {name: _tensor() for name in ("input", "middle", "output")}
    steps = (
        _SyntheticStep("one", ("input",), ("middle",), scratch_size=7, scratch_alignment=4),
        _SyntheticStep("two", ("middle",), ("output",), scratch_size=19, scratch_alignment=32),
    )
    layout = plan_memory(
        values=values,
        constants=(),
        inputs=("input",),
        outputs=("output",),
        steps=steps,
        arena_alignment=16,
    )

    assert layout.scratch_size == 19
    assert layout.scratch_alignment == 32
    assert layout.arena_alignment == 32
    assert layout.scratch_offset is not None
    assert layout.scratch_offset >= layout.activation_arena_size
    assert layout.scratch_offset % 32 == 0
    assert layout.arena_size >= layout.scratch_offset + layout.scratch_size
