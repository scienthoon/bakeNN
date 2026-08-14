from __future__ import annotations

from types import MappingProxyType

from .model import TargetArchitecture, TargetDescriptor


PORTABLE_32 = TargetDescriptor(
    target_id="portable32",
    architecture=TargetArchitecture.PORTABLE,
    cpu="generic-32-bit",
    abi="c11-ilp32",
    toolchain=None,
    features=frozenset({"scalar-int8"}),
)

CORTEX_M0PLUS = TargetDescriptor(
    target_id="cortex-m0plus",
    architecture=TargetArchitecture.ARM,
    cpu="cortex-m0plus",
    abi="aapcs32-soft",
    toolchain="arm-none-eabi",
    features=frozenset({"scalar-int8", "thumb", "armv6-m"}),
    arena_alignment=4,
    constant_alignment=4,
    compiler_flags=("-mcpu=cortex-m0plus", "-mthumb", "-mfloat-abi=soft"),
)

CORTEX_M4 = TargetDescriptor(
    target_id="cortex-m4",
    architecture=TargetArchitecture.ARM,
    cpu="cortex-m4",
    abi="aapcs32-soft",
    toolchain="arm-none-eabi",
    features=frozenset({"scalar-int8", "thumb", "armv7e-m", "dsp"}),
    arena_alignment=8,
    constant_alignment=4,
    compiler_flags=("-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=soft"),
)

RV32IMC = TargetDescriptor(
    target_id="rv32imc",
    architecture=TargetArchitecture.RISCV,
    cpu="rv32imc",
    abi="ilp32",
    toolchain="riscv-unknown-elf",
    features=frozenset({"scalar-int8", "riscv-i", "riscv-m", "riscv-c"}),
    arena_alignment=4,
    constant_alignment=4,
    compiler_flags=("-march=rv32imc", "-mabi=ilp32"),
)

ESP32 = TargetDescriptor(
    target_id="esp32",
    architecture=TargetArchitecture.XTENSA,
    cpu="xtensa-lx6",
    abi="esp-idf",
    toolchain="esp-idf",
    features=frozenset({"scalar-int8", "xtensa", "external-flash"}),
    arena_alignment=4,
    constant_alignment=4,
    metadata={"idf_target": "esp32"},
)

ESP32_S3 = TargetDescriptor(
    target_id="esp32s3",
    architecture=TargetArchitecture.XTENSA,
    cpu="xtensa-lx7",
    abi="esp-idf",
    toolchain="esp-idf",
    features=frozenset(
        {"scalar-int8", "xtensa", "esp-vector-int8", "external-flash", "optional-psram"}
    ),
    arena_alignment=16,
    constant_alignment=16,
    metadata={"idf_target": "esp32s3"},
)

ESP32_C3 = TargetDescriptor(
    target_id="esp32c3",
    architecture=TargetArchitecture.RISCV,
    cpu="esp-rv32imc",
    abi="esp-idf",
    toolchain="esp-idf",
    features=frozenset({"scalar-int8", "riscv-i", "riscv-m", "riscv-c", "external-flash"}),
    arena_alignment=4,
    constant_alignment=4,
    metadata={"idf_target": "esp32c3"},
)

_PROFILES = {
    item.target_id: item
    for item in (PORTABLE_32, CORTEX_M0PLUS, CORTEX_M4, RV32IMC, ESP32, ESP32_S3, ESP32_C3)
}
TARGET_PROFILES = MappingProxyType(_PROFILES)


def resolve_target(target: str | TargetDescriptor | None) -> TargetDescriptor:
    if target is None:
        return PORTABLE_32
    if isinstance(target, TargetDescriptor):
        return target
    if not isinstance(target, str) or not target:
        raise TypeError("target must be a profile id, TargetDescriptor, or None")
    try:
        return TARGET_PROFILES[target]
    except KeyError as error:
        choices = ", ".join(sorted(TARGET_PROFILES))
        raise ValueError(f"unknown BakeNN target {target!r}; expected one of: {choices}") from error


__all__ = [
    "CORTEX_M0PLUS",
    "CORTEX_M4",
    "ESP32",
    "ESP32_C3",
    "ESP32_S3",
    "PORTABLE_32",
    "RV32IMC",
    "TARGET_PROFILES",
    "resolve_target",
]
