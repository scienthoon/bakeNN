#!/usr/bin/env python3
"""Build the trained MNIST comparison with microTVM AOT+USMP+CMSIS-NN.

This is deliberately a boardless build/correctness protocol.  It compiles the
same frozen FP32 checkpoint and calibration corpus through BakeNN, serializes
that exact quantized graph as TFLite for TVM, validates both generated C paths
on the host, and cross-links both paths for Cortex-M4.  It does not report MCU
cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples" / "mnist"
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(REPOSITORY / "benchmarks" / "tflm_compare"))

import bakenn  # noqa: E402
from bakenn.backend.cmsis_nn.bundle import bundle_kernels  # noqa: E402
from bakenn.targets import (  # noqa: E402
    CORTEX_M4,
    build_freestanding_elf,
    discover_gnu_toolchain,
)
from evidence_utils import (  # noqa: E402
    artifact_set_sha256,
    corpus_sha256,
    logical_checkpoint_sha256,
    sha256_file,
)
from quantized_graph_to_tflite import export_quantized_graph  # noqa: E402
from run_mnist import MNISTNet, quantize_mnist_corpus  # noqa: E402


TVM_SOURCE_SHA512 = (
    "e2d7f81ed87d184fdd20b7e1f2fd16bf7e15a52aea9c52fde95cb1444101e645"
    "88a8a1d0d360f3cd60d72cad01d619195ce45eefec7d07b6544888da8252609b"
)
CMSIS_KERNEL_IDS = (
    "cmsis_nn.conv2d_s8.v4.0.0",
    "cmsis_nn.depthwise_conv2d_s8.v4.0.0",
    "cmsis_nn.linear_s8.v4.0.0",
    "cmsis_nn.max_pool2d_s8.v4.0.0",
)
EXPECTED_OPERATOR_COUNTS = {
    "CONV_2D": 2,
    "MAX_POOL_2D": 2,
    "RESHAPE": 1,
    "FULLY_CONNECTED": 1,
}
EXPECTED_CMSIS_CALLS = (
    "arm_depthwise_conv_wrapper_s8",
    "arm_max_pool_s8",
    "arm_convolve_wrapper_s8",
    "arm_fully_connected_s8",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tvm-source",
        type=Path,
        required=True,
        help="Apache TVM v0.16.0 source tree",
    )
    parser.add_argument(
        "--tvm-build",
        type=Path,
        required=True,
        help="TVM v0.16.0 build with USE_MICRO=ON and USE_CMSISNN=ON",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=EXAMPLE / "evidence",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "build" / "microtvm_compare",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY / "benchmarks" / "microtvm_compare" / "results",
    )
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--arm-gcc", default="arm-none-eabi-gcc")
    return parser.parse_args()


def _verify_tvm_tree(source: Path, build: Path) -> None:
    version = (source / "version.py").read_text(encoding="utf-8")
    if '"0.16.0"' not in version and "'0.16.0'" not in version:
        raise RuntimeError("--tvm-source is not Apache TVM v0.16.0")
    library = next(
        (
            candidate
            for candidate in (
                build / "libtvm.dylib",
                build / "libtvm.so",
                build / "Release" / "tvm.dll",
            )
            if candidate.is_file()
        ),
        None,
    )
    if library is None:
        raise RuntimeError("--tvm-build does not contain a built libtvm")


def _load_frozen_inputs(
    evidence_dir: Path,
) -> tuple[dict[str, object], MNISTNet, torch.Tensor, np.ndarray, np.ndarray]:
    evidence = json.loads((evidence_dir / "mnist_evidence.json").read_text())
    checkpoint = evidence_dir / "mnist_fp32.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if logical_checkpoint_sha256(state) != evidence["checkpoint"]["logical_tensor_sha256"]:
        raise RuntimeError("checkpoint logical hash does not match MNIST evidence")
    if sha256_file(checkpoint) != evidence["checkpoint"]["file_sha256"]:
        raise RuntimeError("checkpoint file hash does not match MNIST evidence")

    calibration_raw = np.fromfile(
        evidence_dir / "calibration_images_u8.bin", dtype=np.uint8
    ).reshape(tuple(evidence["calibration"]["shape_nhw"]))
    calibration_labels = np.fromfile(
        evidence_dir / "calibration_labels_u8.bin", dtype=np.uint8
    )
    if (
        corpus_sha256(calibration_raw, calibration_labels, domain="calibration-u8")
        != evidence["calibration"]["corpus_sha256"]
    ):
        raise RuntimeError("calibration corpus hash does not match MNIST evidence")
    calibration = (
        torch.from_numpy(calibration_raw.copy()).unsqueeze(1).to(torch.float32) / 255.0
    )

    raw_images = np.fromfile(
        evidence_dir / "physical_test_images_u8.bin", dtype=np.uint8
    ).reshape((-1, 28, 28))
    labels = np.fromfile(evidence_dir / "physical_test_labels_u8.bin", dtype=np.uint8)
    if (
        corpus_sha256(raw_images, labels, domain="physical-raw-u8")
        != evidence["physical_test_corpus"]["raw_corpus_sha256"]
    ):
        raise RuntimeError("physical corpus hash does not match MNIST evidence")

    model = MNISTNet().eval()
    model.load_state_dict(state)
    return evidence, model, calibration, raw_images, labels


def _host_runner_source(header: str, symbol: str, input_size: int, output_size: int) -> str:
    return f'''#include "{header}"
#include <stdio.h>
#include <stdint.h>

int main(void) {{
    static int8_t input[{input_size}];
    static int8_t output[{output_size}];
    struct tvmgen_mnist_inputs inputs = {{input}};
    struct tvmgen_mnist_outputs outputs = {{output}};
    while (fread(input, 1u, sizeof(input), stdin) == sizeof(input)) {{
        if ({symbol}(&inputs, &outputs) != 0) return 4;
        if (fwrite(output, 1u, sizeof(output), stdout) != sizeof(output)) return 2;
    }}
    return ferror(stdin) ? 3 : 0;
}}
'''


def _compile_microtvm_host(
    *,
    cc: str,
    tvm_source: Path,
    mlf: Path,
    bundle: object,
    output: Path,
) -> Path:
    runner = output / "microtvm_host_runner.c"
    runner.write_text(
        _host_runner_source("tvmgen_mnist.h", "tvmgen_mnist_run", 784, 10),
        encoding="utf-8",
    )
    executable = output / "microtvm_host_runner"
    source_dir = output / "microtvm_host_codegen"
    source_dir.mkdir(exist_ok=True)
    sources: list[Path] = []
    for source in sorted((mlf / "codegen" / "host" / "src").glob("*.c")):
        text = source.read_text(encoding="utf-8")
        # Mach-O requires a two-part section name.  Section placement is not
        # part of this host numerical check, so preserve alignment and remove
        # only the ELF section annotation in the host-only copy.
        text = text.replace('__attribute__((section(".rodata.tvm"), ))', "")
        text = text.replace(
            '__attribute__((section(".bss.noinit.tvm"), aligned(16)))',
            "__attribute__((aligned(16)))",
        )
        destination = source_dir / source.name
        destination.write_text(text, encoding="utf-8")
        sources.append(destination)
    command = [
        cc,
        "-std=c11",
        "-O2",
        "-D__GNUC_PYTHON__",
        "-D__RESTRICT=restrict",
        "-DBAKENN_CMSIS_NN_BUILTIN_MEMORY",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-missing-field-initializers",
        "-Wno-switch-bool",
        *(str(path) for path in sources),
        *(str(path) for path in bundle.sources),
        str(runner),
        "-I",
        str(mlf / "codegen" / "host" / "include"),
        "-I",
        str(tvm_source / "include"),
        "-I",
        str(tvm_source / "3rdparty" / "dlpack" / "include"),
        "-I",
        str(tvm_source / "3rdparty" / "dmlc-core" / "include"),
        *(flag for path in bundle.include_dirs for flag in ("-I", str(path))),
        "-o",
        str(executable),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "microTVM host runner compilation failed:\n"
            + (completed.stderr or completed.stdout)
        )
    return executable


def _compile_bakenn_host(compiled: object, cc: str, output: Path) -> Path:
    artifacts = compiled.artifacts
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = str(manifest["model"])
    macro = symbol.upper()
    runner = output / "bakenn_host_runner.c"
    runner.write_text(
        f'''#include "{artifacts.header.name}"
#include <stdio.h>
int main(void) {{
  static uint8_t arena[{macro}_ARENA_SIZE > 0u ? {macro}_ARENA_SIZE : 1u];
  static int8_t input[{macro}_INPUT_SIZE];
  static int8_t output[{macro}_OUTPUT_SIZE];
  while (fread(input, 1u, sizeof(input), stdin) == sizeof(input)) {{
    {symbol}_infer({macro}_ARENA_SIZE > 0u ? arena : NULL, input, output);
    if (fwrite(output, 1u, sizeof(output), stdout) != sizeof(output)) return 2;
  }}
  return ferror(stdin) ? 3 : 0;
}}
''',
        encoding="utf-8",
    )
    executable = output / "bakenn_host_runner"
    command = [
        cc,
        "-std=c11",
        "-O2",
        "-D__GNUC_PYTHON__",
        "-D__RESTRICT=restrict",
        "-DBAKENN_CMSIS_NN_BUILTIN_MEMORY",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(artifacts.model_source),
        str(artifacts.weights_source),
        str(artifacts.kernels_source),
        *(str(path) for path in artifacts.support_sources),
        str(runner),
        "-I",
        str(artifacts.output_dir),
        *(flag for path in artifacts.support_include_dirs for flag in ("-I", str(path))),
        "-o",
        str(executable),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "BakeNN host runner compilation failed:\n"
            + (completed.stderr or completed.stdout)
        )
    return executable


def _run_host(
    executable: Path,
    input_codes: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
) -> dict[str, int]:
    result = subprocess.run(
        [str(executable)], input=input_codes.tobytes(), capture_output=True, check=True
    )
    actual = np.frombuffer(result.stdout, dtype=np.int8)
    if actual.size != expected.size:
        raise RuntimeError(
            f"host generated C returned {actual.size} bytes, expected {expected.size}"
        )
    actual = actual.reshape(expected.shape)
    mismatches = int(np.count_nonzero(actual != expected))
    max_error = int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))))
    if mismatches:
        raise RuntimeError(
            f"{label} differs from the integer reference in {mismatches} bytes "
            f"(max error {max_error} LSB)"
        )
    return {
        "compared_bytes": int(expected.size),
        "mismatched_bytes": mismatches,
        "max_abs_lsb_error": max_error,
    }


def _host_constant_evaluator(
    tvm: object, cxx: str, output: Path, tvm_source: Path
) -> str:
    """Replace TVM's LLVM-only constant executor with its C backend.

    TVM 0.16's CMSIS-NN weight-layout pass evaluates a compile-time transpose
    via the Relay interpreter and defaults that tiny host function to LLVM.
    The target model remains C/AOT/Cortex-M4; this hook only evaluates that
    host-side constant transform on installations built without LLVM.
    """

    original = tvm.get_global_func("relay.backend.build")
    header = output / "tvm_host_vector_compat.hpp"
    header.write_text(
        "#include <stdint.h>\n"
        "struct int32_t4 {\n"
        "  int32_t s0, s1, s2, s3;\n"
        "  int32_t4(int32_t a, int32_t b, int32_t c, int32_t d)\n"
        "      : s0(a), s1(b), s2(c), s3(d) {}\n"
        "};\n"
        "typedef int32_t int8_t4;\n",
        encoding="utf-8",
    )

    @tvm._ffi.register_func("relay.backend.build", override=True)
    def build_host(mod: object, target: object, target_host: object = None) -> object:
        if getattr(target.kind, "name", "") != "llvm":
            return original(mod, target, target_host)
        c_module = tvm.driver.build(mod, target=tvm.target.Target("c"))
        library = output / "tvm_host_constant_eval.dylib"

        def compile_shared(destination: str, objects: list[str], options: object = None) -> None:
            command = [
                cxx,
                "-shared",
                "-fPIC",
                *(("-undefined", "dynamic_lookup") if sys.platform == "darwin" else ()),
                "-std=c++17",
                "-include",
                str(header),
                "-I",
                str(tvm_source / "include"),
                "-I",
                str(tvm_source / "3rdparty" / "dlpack" / "include"),
                "-I",
                str(tvm_source / "3rdparty" / "dmlc-core" / "include"),
                "-x",
                "c++",
                *objects,
                "-o",
                destination,
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(
                    "TVM C host-constant evaluator compilation failed:\n"
                    + (completed.stderr or completed.stdout)
                )

        compile_shared.output_format = "dylib" if sys.platform == "darwin" else "so"
        c_module.export_library(str(library), fcompile=compile_shared)
        return tvm.runtime.load_module(str(library))

    return "TVM C backend + host C++17 compiler (compile-time constants only)"


def _build_microtvm(
    *,
    tvm_source: Path,
    tvm_build: Path,
    cxx: str,
    tflite_data: bytes,
    output: Path,
) -> tuple[Path, Path, dict[str, object], str]:
    sys.path.insert(0, str(tvm_source / "python"))
    os.environ["TVM_LIBRARY_PATH"] = str(tvm_build)
    os.environ.setdefault("TOPHUB_LOCATION", "NONE")
    os.environ.setdefault("TEST_DATA_ROOT_PATH", str(output / "tvm_test_data"))

    import tflite  # noqa: PLC0415
    import tvm  # noqa: PLC0415
    from tvm import relay  # noqa: PLC0415
    from tvm.micro import export_model_library_format  # noqa: PLC0415
    from tvm.relay.backend import Executor, Runtime  # noqa: PLC0415
    from tvm.relay.op.contrib import cmsisnn  # noqa: PLC0415

    host_evaluator = _host_constant_evaluator(tvm, cxx, output, tvm_source)
    parsed = tflite.Model.GetRootAsModel(tflite_data, 0)
    subgraph = parsed.Subgraphs(0)
    input_tensor = subgraph.Tensors(subgraph.Inputs(0))
    input_name = input_tensor.Name().decode("utf-8")
    relay_module, params = relay.frontend.from_tflite(
        parsed,
        shape_dict={input_name: (1, 28, 28, 1)},
        dtype_dict={input_name: "int8"},
    )
    target = tvm.target.Target("c -keys=arm_cpu,cpu -mcpu=cortex-m4")
    runtime = Runtime("crt")
    executor = Executor(
        "aot",
        {
            "unpacked-api": True,
            "interface-api": "c",
            "workspace-byte-alignment": 16,
            "constant-byte-alignment": 16,
        },
    )
    config = {
        "tir.disable_vectorize": True,
        "tir.usmp.enable": True,
        "tir.usmp.algorithm": "greedy_by_size",
        "relay.ext.cmsisnn.options": {"mcpu": "cortex-m4"},
    }
    with tvm.transform.PassContext(opt_level=3, config=config):
        partitioned = cmsisnn.partition_for_cmsisnn(
            relay_module, params, mcpu="cortex-m4", mod_name="default"
        )
        cmsis_functions = sorted(
            name.name_hint
            for name, function in partitioned.functions.items()
            if function.attrs is not None
            and function.attrs.get("Compiler") is not None
            and str(function.attrs.get("Compiler")) == "cmsis-nn"
        )
        if len(cmsis_functions) != 5:
            raise RuntimeError(
                f"expected 5 CMSIS-NN partitions, found {len(cmsis_functions)}"
            )
        lowered = relay.build(
            partitioned,
            target=target,
            params=None,
            runtime=runtime,
            executor=executor,
            mod_name="mnist",
        )

    tar_path = output / "mnist_microtvm_mlf.tar"
    export_model_library_format(lowered, tar_path)
    extracted = output / "mnist_microtvm_mlf"
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    with tarfile.open(tar_path) as archive:
        root = extracted.resolve()
        for member in archive.getmembers():
            destination = (extracted / member.name).resolve()
            if root != destination and root not in destination.parents:
                raise RuntimeError(f"unsafe path in generated MLF: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"unexpected link in generated MLF: {member.name}")
        try:
            archive.extractall(extracted, filter="fully_trusted")
        except TypeError:  # Python 3.10/3.11
            archive.extractall(extracted)
    metadata = json.loads((extracted / "metadata.json").read_text())
    generated_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((extracted / "codegen" / "host" / "src").glob("*.c"))
    )
    missing = [name for name in EXPECTED_CMSIS_CALLS if name not in generated_source]
    if missing:
        raise RuntimeError(f"microTVM C is missing CMSIS-NN calls: {missing}")
    return tar_path, extracted, metadata, host_evaluator


def _linker_script() -> str:
    return """ENTRY(microtvm_target_entry)
MEMORY {
  FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 16M
  RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 16M
}
SECTIONS {
  .text : {
    KEEP(*(.text.microtvm_target_entry))
    *(.text*) *(.rodata*) *(.srodata*)
  } > FLASH
  .ARM.extab : { *(.ARM.extab*) } > FLASH
  .ARM.exidx : { *(.ARM.exidx*) } > FLASH
  .data : { *(.data*) *(.sdata*) } > RAM AT > FLASH
  .bss (NOLOAD) : { *(.bss*) *(.sbss*) *(COMMON) } > RAM
  /DISCARD/ : { *(.comment*) *(.note*) }
}
"""


def _cross_link_microtvm(
    *,
    compiler: str,
    tvm_source: Path,
    mlf: Path,
    bundle: object,
    output: Path,
) -> dict[str, object]:
    toolchain = discover_gnu_toolchain(CORTEX_M4, compiler=compiler)
    runner = output / "microtvm_target_runner.c"
    runner.write_text(
        '''#include "tvmgen_mnist.h"
#include <stdint.h>
static int8_t model_input[TVMGEN_MNIST_VALUE_SIZE];
static int8_t model_output[TVMGEN_MNIST_LINEAR_SIZE];
volatile int32_t microtvm_output_checksum;
__attribute__((used, section(".text.microtvm_target_entry")))
void microtvm_target_entry(void) {
  struct tvmgen_mnist_inputs inputs = {model_input};
  struct tvmgen_mnist_outputs outputs = {model_output};
  int32_t status = tvmgen_mnist_run(&inputs, &outputs);
  int32_t sum = status;
  for (uint32_t i = 0; i < TVMGEN_MNIST_LINEAR_SIZE; ++i) sum += model_output[i];
  microtvm_output_checksum = sum;
  for (;;) { }
}
''',
        encoding="utf-8",
    )
    linker = output / "microtvm_freestanding.ld"
    linker.write_text(_linker_script(), encoding="utf-8")
    elf = output / "microtvm_mnist_cortex_m4.elf"
    map_file = output / "microtvm_mnist_cortex_m4.map"
    flags = (
        *CORTEX_M4.compiler_flags,
        "-Os",
        "-std=c11",
        "-ffreestanding",
        "-fno-builtin",
        "-ffunction-sections",
        "-fdata-sections",
        "-D__RESTRICT=restrict",
        "-DBAKENN_CMSIS_NN_FREESTANDING",
        "-DBAKENN_CMSIS_NN_BUILTIN_MEMORY",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-missing-field-initializers",
        "-Wno-switch-bool",
        "-Wno-error=attributes",
    )
    source_dir = output / "microtvm_cortex_m4_codegen"
    source_dir.mkdir(exist_ok=True)
    sources: list[Path] = []
    for source in sorted((mlf / "codegen" / "host" / "src").glob("*.c")):
        text = source.read_text(encoding="utf-8")
        # TVM emits these hosted includes but the generated functions use no
        # declarations from them.  Remove only the dead includes in this
        # freestanding cross-link copy; the MLF evidence remains untouched.
        for include in (
            "#include <math.h>\n",
            "#include <stdio.h>\n",
            "#include <stdlib.h>\n",
        ):
            text = text.replace(include, "")
        destination = source_dir / source.name
        destination.write_text(text, encoding="utf-8")
        sources.append(destination)
    command = [
        str(toolchain.compiler),
        *flags,
        *(str(path) for path in sources),
        *(str(path) for path in bundle.sources),
        str(runner),
        "-I",
        str(mlf / "codegen" / "host" / "include"),
        "-I",
        str(tvm_source / "include"),
        "-I",
        str(tvm_source / "3rdparty" / "dlpack" / "include"),
        "-I",
        str(tvm_source / "3rdparty" / "dmlc-core" / "include"),
        *(flag for path in bundle.include_dirs for flag in ("-I", str(path))),
        "-nostdlib",
        "-Wl,--gc-sections",
        f"-Wl,-Map={map_file}",
        f"-Wl,-T,{linker}",
        "-Wl,--entry=microtvm_target_entry",
        "-lgcc",
        "-o",
        str(elf),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"microTVM Cortex-M4 link failed:\n{completed.stderr}")
    undefined = subprocess.run(
        [str(toolchain.nm), "--undefined-only", str(elf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if undefined:
        raise RuntimeError(f"microTVM ELF has undefined symbols: {undefined}")
    size_output = subprocess.run(
        [str(toolchain.size), str(elf)], check=True, capture_output=True, text=True
    ).stdout
    fields = size_output.strip().splitlines()[-1].split()
    text_bytes, data_bytes, bss_bytes = map(int, fields[:3])
    return {
        "compiler": str(toolchain.compiler),
        "compiler_version": toolchain.version,
        "compiler_flags": list(flags),
        "elf": elf,
        "map": map_file,
        "text_bytes": text_bytes,
        "data_bytes": data_bytes,
        "bss_bytes": bss_bytes,
        "flash_load_bytes": text_bytes + data_bytes,
        "static_sram_bytes": data_bytes + bss_bytes,
        "undefined_symbols": [],
    }


def _copy_evidence(
    results: Path,
    *,
    tflite_path: Path,
    mlf: Path,
    expected: Path,
    bakenn_manifest: Path,
    bakenn_memory: Path,
) -> list[Path]:
    if results.exists():
        shutil.rmtree(results)
    (results / "microtvm_codegen").mkdir(parents=True)
    copied: list[Path] = []
    for source, relative in (
        (tflite_path, Path("mnist_common_int8.tflite")),
        (expected, Path("mnist_common_expected_int8.bin")),
        (mlf / "metadata.json", Path("microtvm_codegen/metadata.json")),
        (mlf / "src" / "mnist.relay", Path("microtvm_codegen/mnist.relay")),
        (bakenn_manifest, Path("bakenn_manifest.json")),
        (bakenn_memory, Path("bakenn_memory.json")),
    ):
        destination = results / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    for source in sorted((mlf / "codegen" / "host" / "src").glob("*.c")):
        destination = results / "microtvm_codegen" / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    header = mlf / "codegen" / "host" / "include" / "tvmgen_mnist.h"
    destination = results / "microtvm_codegen" / header.name
    shutil.copy2(header, destination)
    copied.append(destination)
    return copied


def _hash_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    args = _arguments()
    _verify_tvm_tree(args.tvm_source, args.tvm_build)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence, model, calibration, raw_images, labels = _load_frozen_inputs(args.evidence_dir)

    options = bakenn.CBackendOptions(
        kernel_policy=bakenn.KernelPolicy.AUTO,
        enable_cmsis_nn=True,
        target=CORTEX_M4,
    )
    ptq_options = bakenn.PTQOptions(
        linear_weight_granularity=bakenn.LinearWeightGranularity.PER_TENSOR
    )
    compiled = bakenn.compile_torch_ptq(
        model,
        calibration[:1],
        calibration,
        args.output_dir / "bakenn",
        name="mnist_common",
        backend_options=options,
        ptq_options=ptq_options,
        target=CORTEX_M4,
    )
    tflite_export = export_quantized_graph(compiled.graph)
    if tflite_export.operator_counts != EXPECTED_OPERATOR_COUNTS:
        raise RuntimeError(f"unexpected TFLite operators: {tflite_export.operator_counts}")
    tflite_path = args.output_dir / "mnist_common_int8.tflite"
    tflite_path.write_bytes(tflite_export.data)

    input_codes = quantize_mnist_corpus(compiled.plan, raw_images)
    expected_outputs = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, input_codes[index : index + 1])
            for index in range(input_codes.shape[0])
        ],
        axis=0,
    ).reshape((input_codes.shape[0], -1))
    expected_path = args.output_dir / "mnist_common_expected_int8.bin"
    expected_path.write_bytes(expected_outputs.tobytes())
    accuracy = float(np.mean(np.argmax(expected_outputs, axis=1) == labels))

    tar_path, mlf, metadata, host_evaluator = _build_microtvm(
        tvm_source=args.tvm_source,
        tvm_build=args.tvm_build,
        cxx=args.cxx,
        tflite_data=tflite_export.data,
        output=args.output_dir,
    )
    bundle = bundle_kernels(args.output_dir / "microtvm_cmsis", CMSIS_KERNEL_IDS)
    bakenn_host_runner = _compile_bakenn_host(compiled, args.cc, args.output_dir)
    bakenn_differential = _run_host(
        bakenn_host_runner,
        input_codes,
        expected_outputs,
        label="BakeNN generated CMSIS-NN C",
    )
    microtvm_host_runner = _compile_microtvm_host(
        cc=args.cc,
        tvm_source=args.tvm_source,
        mlf=mlf,
        bundle=bundle,
        output=args.output_dir,
    )
    microtvm_differential = _run_host(
        microtvm_host_runner,
        input_codes,
        expected_outputs,
        label="microTVM generated CMSIS-NN C",
    )

    bakenn_link = build_freestanding_elf(
        compiled.artifacts,
        CORTEX_M4,
        args.output_dir / "bakenn_cross",
        compiler=args.arm_gcc,
    )
    microtvm_link = _cross_link_microtvm(
        compiler=args.arm_gcc,
        tvm_source=args.tvm_source,
        mlf=mlf,
        bundle=bundle,
        output=args.output_dir,
    )

    copied = _copy_evidence(
        args.results_dir,
        tflite_path=tflite_path,
        mlf=mlf,
        expected=expected_path,
        bakenn_manifest=compiled.artifacts.manifest,
        bakenn_memory=compiled.artifacts.memory_report_json,
    )
    semantic_hash = artifact_set_sha256(args.results_dir, copied)
    memory = metadata["modules"]["mnist"]["memory"]["functions"]["main"][0]
    result = {
        "schema_version": 1,
        "evidence_class": "boardless_cross_build",
        "performance_claim_permitted": False,
        "comparison": "BakeNN direct CMSIS-NN vs microTVM AOT+USMP+CMSIS-NN",
        "model": "trained MNISTNet full model",
        "contract": {
            "batch": 1,
            "input_shape": [1, 28, 28, 1],
            "output_shape": [1, 10],
            "samples": int(input_codes.shape[0]),
            "operator_counts": tflite_export.operator_counts,
            "linear_weight_quantization": "per-tensor symmetric int8",
            "checkpoint_logical_sha256": evidence["checkpoint"]["logical_tensor_sha256"],
            "calibration_corpus_sha256": evidence["calibration"]["corpus_sha256"],
            "input_sha256": hashlib.sha256(input_codes.tobytes()).hexdigest(),
            "expected_output_sha256": sha256_file(expected_path),
            "python_int8_accuracy": accuracy,
        },
        "versions": {
            "bakenn": bakenn.__version__,
            "apache_tvm": "0.16.0",
            "apache_tvm_source_archive_sha512": TVM_SOURCE_SHA512,
            "tflite_schema_package": "2.18.0",
            "cmsis_nn": "4.0.0",
            "host_constant_evaluator": host_evaluator,
        },
        "microtvm": {
            "target": "c -keys=arm_cpu,cpu -mcpu=cortex-m4",
            "executor": "AOT unpacked C interface",
            "runtime": "CRT",
            "usmp": {"enabled": True, "algorithm": "greedy_by_size"},
            "cmsis_nn_calls": list(EXPECTED_CMSIS_CALLS),
            "workspace_size_bytes": int(memory["workspace_size_bytes"]),
            "constants_size_bytes": int(memory["constants_size_bytes"]),
            "metadata_io_size_bytes": int(memory["io_size_bytes"]),
            "mlf": _hash_record(tar_path),
        },
        "host_differential": {
            "bakenn_vs_reference": bakenn_differential,
            "microtvm_vs_reference": microtvm_differential,
            "microtvm_vs_bakenn_mismatched_bytes": 0,
        },
        "cortex_m4_cross_link": {
            "toolchain": microtvm_link["compiler_version"],
            "bakenn": {
                "flash_load_bytes": bakenn_link.flash_load_bytes,
                "static_sram_bytes": bakenn_link.static_sram_bytes,
                "text_bytes": bakenn_link.text_bytes,
                "data_bytes": bakenn_link.data_bytes,
                "bss_bytes": bakenn_link.bss_bytes,
                "arena_bytes": bakenn_link.model_arena_bytes,
                "elf_sha256": sha256_file(bakenn_link.elf),
                "map_sha256": sha256_file(bakenn_link.map_file),
            },
            "microtvm": {
                "flash_load_bytes": microtvm_link["flash_load_bytes"],
                "static_sram_bytes": microtvm_link["static_sram_bytes"],
                "text_bytes": microtvm_link["text_bytes"],
                "data_bytes": microtvm_link["data_bytes"],
                "bss_bytes": microtvm_link["bss_bytes"],
                "usmp_workspace_bytes": int(memory["workspace_size_bytes"]),
                "elf_sha256": sha256_file(microtvm_link["elf"]),
                "map_sha256": sha256_file(microtvm_link["map"]),
            },
        },
        "artifacts": {
            "evidence_set_sha256": semantic_hash,
            "tflite_sha256": sha256_file(args.results_dir / "mnist_common_int8.tflite"),
            "files": [
                {
                    "path": path.relative_to(args.results_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(copied)
            ],
        },
        "runtime": {
            "physical_cycles": {
                "status": "unmeasured",
                "reason": "This record is a cross-build and host correctness comparison.",
            },
            "physical_stack": {
                "status": "unmeasured",
                "reason": "Requires execution on the same physical Cortex-M4 board.",
            },
        },
        "notes": [
            "Both paths start from the same BakeNN QuantizedGraph and TFLite merely transports that fixed contract into TVM.",
            "The common comparison uses per-tensor FC weights because TVM 0.16 CMSIS-NN FullyConnected requires a scalar weight scale.",
            "Flash and static SRAM are linker evidence, not latency measurements.",
            "TVM metadata io_size_bytes is reported verbatim and is not treated as the 794-byte public input/output ABI alone.",
        ],
    }
    result_path = args.results_dir / "mnist_cortex_m4_cross_build.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
