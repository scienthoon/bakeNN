from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Iterable

from bakenn.errors import CompileError

from .model import TargetArchitecture, TargetDescriptor
from .profiles import resolve_target

if TYPE_CHECKING:
    from bakenn.backend.portable_c import CompilationArtifacts


_COMPILER_CANDIDATES = {
    "arm-none-eabi": ("arm-none-eabi-gcc",),
    "riscv-unknown-elf": (
        "riscv-none-elf-gcc",
        "riscv64-elf-gcc",
        "riscv64-unknown-elf-gcc",
        "riscv32-unknown-elf-gcc",
    ),
}

_FORBIDDEN_SYMBOLS = frozenset(
    {
        "malloc",
        "calloc",
        "realloc",
        "free",
        "aligned_alloc",
        "memalign",
        "_malloc_r",
        "_calloc_r",
        "_realloc_r",
        "_free_r",
    }
)
_FLOAT_HELPER = re.compile(
    r"^(?:__aeabi_[fd]|__(?:add|sub|mul|div|neg|extendsfdf|truncdfsf|fix|float).*[sd]f)"
)


@dataclass(frozen=True)
class GNUEmbeddedToolchain:
    compiler: Path
    nm: Path
    size: Path

    @property
    def version(self) -> str:
        result = subprocess.run(
            [str(self.compiler), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()[0].strip()


@dataclass(frozen=True)
class TargetBuildReport:
    target_id: str
    toolchain: str
    compiler: str
    compiler_version: str
    compiler_flags: tuple[str, ...]
    elf: Path
    map_file: Path
    text_bytes: int
    data_bytes: int
    bss_bytes: int
    flash_load_bytes: int
    static_sram_bytes: int
    model_arena_bytes: int
    undefined_symbols: tuple[str, ...]
    forbidden_symbols: tuple[str, ...]

    def json_data(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "target": self.target_id,
            "toolchain": self.toolchain,
            "compiler": self.compiler,
            "compiler_version": self.compiler_version,
            "compiler_flags": list(self.compiler_flags),
            "elf": self.elf.name,
            "map": self.map_file.name,
            "sections": {
                "text_bytes": self.text_bytes,
                "data_bytes": self.data_bytes,
                "bss_bytes": self.bss_bytes,
                "flash_load_bytes": self.flash_load_bytes,
                "static_sram_bytes": self.static_sram_bytes,
            },
            "model_arena_bytes": self.model_arena_bytes,
            "stack_bytes": None,
            "stack_reason": "requires execution on a physical target or cycle-accurate system",
            "cycles": None,
            "cycles_reason": "requires execution on the selected physical target",
            "undefined_symbols": list(self.undefined_symbols),
            "forbidden_symbols": list(self.forbidden_symbols),
        }

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(
            json.dumps(self.json_data(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


def _related_program(compiler: Path, name: str) -> Path:
    result = subprocess.run(
        [str(compiler), f"-print-prog-name={name}"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if value and value != name:
        candidate = Path(value)
        if candidate.is_file():
            return candidate
    compiler_name = compiler.name
    suffix = "gcc"
    candidate_name = compiler_name[: -len(suffix)] + name if compiler_name.endswith(suffix) else name
    candidate = shutil.which(candidate_name)
    if candidate is None:
        raise CompileError(
            f"embedded compiler {compiler} was found but companion {name} was not"
        )
    return Path(candidate)


def discover_gnu_toolchain(
    target: str | TargetDescriptor,
    *,
    compiler: str | Path | None = None,
) -> GNUEmbeddedToolchain:
    descriptor = resolve_target(target)
    if descriptor.toolchain not in _COMPILER_CANDIDATES:
        raise CompileError(
            f"target {descriptor.target_id} does not use a supported freestanding GNU toolchain"
        )
    if compiler is None:
        found = next(
            (
                shutil.which(candidate)
                for candidate in _COMPILER_CANDIDATES[descriptor.toolchain]
                if shutil.which(candidate) is not None
            ),
            None,
        )
        if found is None:
            candidates = ", ".join(_COMPILER_CANDIDATES[descriptor.toolchain])
            raise CompileError(
                f"target {descriptor.target_id} requires one of these compilers on PATH: {candidates}"
            )
        compiler_path = Path(found)
    else:
        requested = Path(compiler)
        found = str(requested) if requested.is_file() else shutil.which(str(compiler))
        if found is None:
            raise CompileError(f"embedded compiler was not found: {compiler}")
        compiler_path = Path(found)
    return GNUEmbeddedToolchain(
        compiler=compiler_path,
        nm=_related_program(compiler_path, "nm"),
        size=_related_program(compiler_path, "size"),
    )


def _linker_script() -> str:
    return """ENTRY(bknn_target_entry)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 16M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 16M
}

SECTIONS
{
    .text :
    {
        KEEP(*(.text.bknn_target_entry))
        *(.text*)
        *(.rodata*)
        *(.srodata*)
    } > FLASH

    .ARM.extab : { *(.ARM.extab*) } > FLASH
    .ARM.exidx : { *(.ARM.exidx*) } > FLASH

    .data :
    {
        *(.data*)
        *(.sdata*)
    } > RAM AT > FLASH

    .bss (NOLOAD) :
    {
        *(.bss*)
        *(.sbss*)
        *(COMMON)
    } > RAM

    /DISCARD/ : { *(.comment*) *(.note*) }
}
"""


def _runner_source(symbol: str) -> str:
    macro = symbol.upper()
    return f'''#include "{symbol}.h"
#include <stdint.h>

_Alignas({macro}_ARENA_ALIGNMENT)
static uint8_t bknn_arena[{macro}_ARENA_SIZE > 0u ? {macro}_ARENA_SIZE : 1u];
static int8_t bknn_input[{macro}_INPUT_SIZE];
static int8_t bknn_output[{macro}_OUTPUT_SIZE];
volatile int32_t bknn_output_checksum;

__attribute__((used, section(".text.bknn_target_entry")))
void bknn_target_entry(void) {{
    uint8_t *arena = {macro}_ARENA_SIZE > 0u ? bknn_arena : (uint8_t *)0;
    {symbol}_infer(arena, bknn_input, bknn_output);
    int32_t checksum = 0;
    for (uint32_t index = 0; index < {macro}_OUTPUT_SIZE; ++index) {{
        checksum += bknn_output[index];
    }}
    bknn_output_checksum = checksum;
    for (;;) {{
    }}
}}
'''


def _symbols(toolchain: GNUEmbeddedToolchain, elf: Path, *options: str) -> tuple[str, ...]:
    result = subprocess.run(
        [str(toolchain.nm), *options, str(elf)],
        check=True,
        capture_output=True,
        text=True,
    )
    names: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields:
            names.append(fields[-1])
    return tuple(sorted(set(names)))


def _parse_size(toolchain: GNUEmbeddedToolchain, elf: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        [str(toolchain.size), str(elf)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    for fields in reversed(lines):
        if len(fields) >= 3:
            try:
                return int(fields[0]), int(fields[1]), int(fields[2])
            except ValueError:
                continue
    raise CompileError(f"could not parse GNU size output for {elf}")


def _artifact_sources(artifacts: "CompilationArtifacts") -> tuple[Path, ...]:
    return (
        artifacts.model_source,
        artifacts.weights_source,
        artifacts.kernels_source,
        *artifacts.support_sources,
    )


def build_freestanding_elf(
    artifacts: "CompilationArtifacts",
    target: str | TargetDescriptor,
    output_dir: str | Path,
    *,
    compiler: str | Path | None = None,
    optimization: str = "-Os",
    extra_flags: Iterable[str] = (),
) -> TargetBuildReport:
    """Cross-link generated C into a freestanding ELF and audit its symbols.

    This is a build/link compatibility check, not execution or cycle evidence.
    The generated runner statically owns the arena, so ``data+bss`` already
    includes the arena and must not be added to it a second time.
    """

    descriptor = resolve_target(target)
    if descriptor.architecture not in (TargetArchitecture.ARM, TargetArchitecture.RISCV):
        raise CompileError(
            f"freestanding ELF verification supports ARM/RISC-V, not {descriptor.target_id}"
        )
    if optimization not in ("-O0", "-O1", "-O2", "-O3", "-Os"):
        raise ValueError("optimization must be one of -O0/-O1/-O2/-O3/-Os")
    metadata = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    artifact_target = metadata["backend"]["target"]["id"]
    if artifact_target != descriptor.target_id:
        raise CompileError(
            f"artifact target {artifact_target} does not match build target {descriptor.target_id}; "
            "recompile the QuantizedGraph with the requested target"
        )
    symbol = metadata["model"]
    build_dir = Path(output_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    runner = build_dir / "bknn_target_runner.c"
    linker = build_dir / "bknn_freestanding.ld"
    elf = build_dir / f"{symbol}_{descriptor.target_id}.elf"
    map_file = build_dir / f"{symbol}_{descriptor.target_id}.map"
    runner.write_text(_runner_source(symbol), encoding="utf-8")
    linker.write_text(_linker_script(), encoding="utf-8")

    toolchain = discover_gnu_toolchain(descriptor, compiler=compiler)
    flags = (
        *descriptor.compiler_flags,
        optimization,
        "-std=c11",
        "-ffreestanding",
        "-fno-builtin",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        *(
            (
                "-DBAKENN_CMSIS_NN_FREESTANDING",
                "-DBAKENN_CMSIS_NN_BUILTIN_MEMORY",
            )
            if artifacts.support_sources
            else ()
        ),
        *(str(value) for value in extra_flags),
    )
    command = [
        str(toolchain.compiler),
        *flags,
        *(str(path) for path in _artifact_sources(artifacts)),
        str(runner),
        "-I",
        str(artifacts.output_dir),
        *(
            flag
            for include_dir in artifacts.support_include_dirs
            for flag in ("-I", str(include_dir))
        ),
        "-nostdlib",
        "-Wl,--gc-sections",
        f"-Wl,-Map={map_file}",
        f"-Wl,-T,{linker}",
        "-Wl,--entry=bknn_target_entry",
        "-lgcc",
        "-o",
        str(elf),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        details = (completed.stderr or completed.stdout).strip()
        raise CompileError(
            f"{descriptor.target_id} freestanding link failed with {toolchain.compiler.name}: {details}"
        )

    undefined = _symbols(toolchain, elf, "--undefined-only")
    all_symbols = _symbols(toolchain, elf)
    forbidden = tuple(
        name
        for name in all_symbols
        if name in _FORBIDDEN_SYMBOLS or _FLOAT_HELPER.match(name)
    )
    if undefined:
        raise CompileError(
            f"{descriptor.target_id} ELF contains unresolved symbols: {', '.join(undefined)}"
        )
    if forbidden:
        raise CompileError(
            f"{descriptor.target_id} ELF contains forbidden heap/float symbols: "
            + ", ".join(forbidden)
        )
    text_bytes, data_bytes, bss_bytes = _parse_size(toolchain, elf)
    flash_load_bytes = text_bytes + data_bytes
    static_sram_bytes = data_bytes + bss_bytes
    if descriptor.flash_bytes is not None and flash_load_bytes > descriptor.flash_bytes:
        raise CompileError(
            f"{descriptor.target_id} ELF Flash load {flash_load_bytes} exceeds budget "
            f"{descriptor.flash_bytes}"
        )
    if descriptor.sram_bytes is not None and static_sram_bytes > descriptor.sram_bytes:
        raise CompileError(
            f"{descriptor.target_id} ELF static SRAM {static_sram_bytes} exceeds budget "
            f"{descriptor.sram_bytes}; stack is not included"
        )
    report = TargetBuildReport(
        target_id=descriptor.target_id,
        toolchain=descriptor.toolchain or "",
        compiler=str(toolchain.compiler),
        compiler_version=toolchain.version,
        compiler_flags=flags,
        elf=elf,
        map_file=map_file,
        text_bytes=text_bytes,
        data_bytes=data_bytes,
        bss_bytes=bss_bytes,
        flash_load_bytes=flash_load_bytes,
        static_sram_bytes=static_sram_bytes,
        model_arena_bytes=int(metadata["arena_bytes"]),
        undefined_symbols=undefined,
        forbidden_symbols=forbidden,
    )
    report.write_json(build_dir / f"{symbol}_{descriptor.target_id}_report.json")
    return report


__all__ = [
    "GNUEmbeddedToolchain",
    "TargetBuildReport",
    "build_freestanding_elf",
    "discover_gnu_toolchain",
]
