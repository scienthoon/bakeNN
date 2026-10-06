#!/usr/bin/env python3
"""Run BakeNN's functional checks offline and print one result line per check.

Every check uses only files in this repository: the frozen MNIST checkpoint,
calibration images and test images under ``examples/mnist/evidence``. Nothing
is downloaded and nothing is trained.

    python scripts/functional_checks.py            # all checks
    python scripts/functional_checks.py F03 F06    # selected checks
    python scripts/functional_checks.py --list     # ids and titles
    python scripts/functional_checks.py --strict   # a missing tool is a failure

A check passes only when its stated condition holds; ``SKIP`` means an
optional host tool (a cross compiler, a C++ compiler) is not installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

REPOSITORY = Path(__file__).resolve().parents[1]
for entry in ("src", "examples/mnist", "examples/targets", "scripts"):
    sys.path.insert(0, str(REPOSITORY / entry))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bakenn  # noqa: E402
from bakenn.errors import CompileError  # noqa: E402
from run_mnist import MNISTNet, quantize_mnist_corpus  # noqa: E402

EVIDENCE = REPOSITORY / "examples/mnist/evidence"
STRICT_C = ("-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic")


class Skip(Exception):
    """An optional host tool needed by this check is not installed."""


@dataclass(frozen=True)
class Check:
    identifier: str
    title: str
    run: Callable[["Session"], str]


class Session:
    """Shared inputs and the default compilation, built once per run."""

    def __init__(self, output: Path, compiler: str) -> None:
        self.output = output
        self.compiler = compiler
        self._compiled: object | None = None
        self.evidence = json.loads((EVIDENCE / "mnist_evidence.json").read_text())
        calibration_shape = tuple(self.evidence["calibration"]["shape_nhw"])
        raw = np.fromfile(EVIDENCE / "calibration_images_u8.bin", dtype=np.uint8)
        self.calibration = (
            torch.from_numpy(raw.reshape(calibration_shape).copy()).unsqueeze(1).float() / 255.0
        )

    def model(self) -> torch.nn.Module:
        model = MNISTNet().eval()
        model.load_state_dict(
            torch.load(EVIDENCE / "mnist_fp32.pt", map_location="cpu", weights_only=True)
        )
        return model

    def compile(self, name: str, **options: object):  # type: ignore[no-untyped-def]
        return bakenn.compile_torch_ptq(
            self.model(),
            self.calibration[:1],
            self.calibration,
            self.output / name,
            name="mnist",
            **options,
        )

    @property
    def compiled(self):  # type: ignore[no-untyped-def]
        if self._compiled is None:
            self._compiled = self.compile("portable")
        return self._compiled

    def test_corpus(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = tuple(self.evidence["physical_test_corpus"]["input_shape_nhwc"])
        pixels = np.fromfile(EVIDENCE / "physical_test_images_u8.bin", dtype=np.uint8)
        inputs = quantize_mnist_corpus(self.compiled.plan, pixels.reshape(shape[:-1]))
        expected = np.fromfile(EVIDENCE / "physical_expected_outputs_int8.bin", dtype=np.int8)
        labels = np.fromfile(EVIDENCE / "physical_test_labels_u8.bin", dtype=np.uint8)
        return inputs, expected.reshape(shape[0], -1), labels


def _require(tool: str) -> str:
    resolved = shutil.which(tool)
    if resolved is None:
        raise Skip(f"{tool} is not installed")
    return resolved


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rejection(action: Callable[[], object]) -> str:
    try:
        action()
    except CompileError as error:
        return str(error)
    raise AssertionError("the compiler accepted an input it must reject")


def _host_runner(session: Session) -> Path:
    artifacts = session.compiled.artifacts
    symbol = artifacts.model_source.stem
    macro = symbol.upper()
    source = session.output / "host_runner.c"
    source.write_text(
        f"""#include "{artifacts.header.name}"
#include <stddef.h>
#include <stdio.h>

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT) static uint8_t arena[{macro}_ARENA_SIZE > 0u ? {macro}_ARENA_SIZE : 1u];
    static int8_t input[{macro}_INPUT_SIZE];
    static int8_t output[{macro}_OUTPUT_SIZE];
    while (fread(input, 1u, {macro}_INPUT_BYTES, stdin) == {macro}_INPUT_BYTES) {{
        {symbol}_infer({macro}_ARENA_SIZE > 0u ? arena : NULL, input, output);
        if (fwrite(output, 1u, {macro}_OUTPUT_BYTES, stdout) != {macro}_OUTPUT_BYTES) {{
            return 1;
        }}
    }}
    return 0;
}}
"""
    )
    executable = session.output / "host_runner"
    subprocess.run(
        [
            session.compiler, *STRICT_C, "-I", str(artifacts.output_dir),
            str(artifacts.model_source), str(artifacts.weights_source),
            str(artifacts.kernels_source), str(source), "-o", str(executable),
        ],
        check=True,
    )
    return executable


def check_install(session: Session) -> str:
    del session
    return (
        f"bakenn={bakenn.__version__} python={sys.version.split()[0]} "
        f"torch={torch.__version__.split('+')[0]}"
    )


def check_compile(session: Session) -> str:
    artifacts = session.compiled.artifacts
    files = sorted(path.name for path in artifacts.output_dir.iterdir() if path.is_file())
    for required in (
        artifacts.header, artifacts.model_source, artifacts.weights_source,
        artifacts.kernels_source, artifacts.manifest, artifacts.memory_report_text,
    ):
        _expect(required.is_file(), f"missing generated file {required.name}")
    generated = "\n".join(
        path.read_text() for path in (artifacts.model_source, artifacts.kernels_source)
    )
    for forbidden in ("malloc(", "calloc(", "free(", "float ", "double "):
        _expect(forbidden not in generated, f"generated C contains {forbidden!r}")
    return f"files={len(files)} heap_calls=0 float_types=0 output={artifacts.output_dir.relative_to(session.output)}"


def check_c_matches_reference(session: Session) -> str:
    inputs, expected, labels = session.test_corpus()
    runner = _host_runner(session)
    completed = subprocess.run([str(runner)], input=inputs.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(completed.stdout, dtype=np.int8).reshape(expected.shape)
    mismatched = int(np.count_nonzero(actual != expected))
    reference = np.concatenate(
        [bakenn.run_reference(session.compiled.plan, inputs[index : index + 1]) for index in range(10)]
    ).reshape(10, -1)
    reference_mismatched = int(np.count_nonzero(actual[:10] != reference))
    correct = int(np.count_nonzero(np.argmax(actual, axis=1) == labels))
    _expect(mismatched == 0, f"{mismatched} bytes differ from the frozen expected outputs")
    _expect(reference_mismatched == 0, f"{reference_mismatched} bytes differ from the Python reference")
    return (
        f"mismatched_bytes={mismatched}/{actual.size} "
        f"python_reference_mismatched_bytes={reference_mismatched}/{reference.size} "
        f"correct={correct}/{len(labels)}"
    )


def check_accuracy_report(session: Session) -> str:
    report = session.compiled.verify_accuracy(session.calibration[:20])
    output = report.output
    _expect(report.sample_count == 20 and len(report.layers) > 0, "accuracy report is empty")
    return (
        f"samples={report.sample_count} layers={len(report.layers)} "
        f"output_max_abs_error={output.maximum_absolute_error:.4f} "
        f"output_rms_error={output.root_mean_square_error:.4f}"
    )


def check_memory_report(session: Session) -> str:
    artifacts = session.compiled.artifacts
    report = json.loads(artifacts.memory_report_json.read_text())
    manifest = bakenn.load_manifest(artifacts.manifest)
    macro = f"#define {artifacts.model_source.stem.upper()}_ARENA_SIZE {manifest['arena_bytes']}u"
    _expect(macro in artifacts.header.read_text(), "header arena size differs from the manifest")
    _expect(artifacts.memory_report_text.read_text().strip() != "", "memory report text is empty")
    _expect(isinstance(report, dict) and bool(report), "memory report JSON is empty")
    return f"arena_bytes={manifest['arena_bytes']} constant_bytes={manifest['constant_bytes']} header_macro_matches=1"


def check_budget(session: Session) -> str:
    arena = session.compiled.memory_report.arena_bytes
    budget = arena // 2
    target = replace(bakenn.PORTABLE_32, sram_bytes=budget)
    message = _rejection(lambda: session.compile("over_budget", target=target))
    _expect("SRAM" in message, f"unexpected rejection: {message}")
    _expect(not (session.output / "over_budget").exists(), "a rejected compile left artifacts behind")
    return f"arena_bytes={arena} sram_budget={budget} rejected=1 artifacts_written=0"


def check_unsupported(session: Session) -> str:
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Tanh()).eval()
    sample = torch.zeros(1, 8)
    message = _rejection(
        lambda: bakenn.compile_torch_ptq(model, sample, sample, session.output / "unsupported")
    )
    _expect("tanh" in message.lower(), f"unexpected rejection: {message}")
    _expect(not (session.output / "unsupported").exists(), "a rejected compile left artifacts behind")
    return "operator=tanh rejected=1 artifacts_written=0"


def check_cortex_m4(session: Session) -> str:
    target = bakenn.CORTEX_M4
    compiled = session.compile(
        "cortex_m4",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY, enable_cmsis_nn=True, target=target
        ),
        target=target,
    )
    kernels = [item.kernel_id for item in compiled.artifacts.backend_plan.selections]
    cmsis = sum(1 for kernel in kernels if kernel.startswith("cmsis_nn."))
    _expect(cmsis > 0, f"no CMSIS-NN kernel was selected: {kernels}")
    _require("arm-none-eabi-gcc")
    report = bakenn.build_freestanding_elf(compiled.artifacts, target, session.output / "cortex_m4_elf")
    _expect(not report.undefined_symbols, f"undefined symbols: {report.undefined_symbols}")
    _expect(not report.forbidden_symbols, f"heap or float symbols: {report.forbidden_symbols}")
    return (
        f"cmsis_nn_kernels={cmsis}/{len(kernels)} flash_bytes={report.flash_load_bytes} "
        f"static_sram_bytes={report.static_sram_bytes} undefined_symbols=0 heap_or_float_symbols=0"
    )


def check_esp_idf(session: Session) -> str:
    from generate_smoke import esp_nn_smoke_graph

    target = bakenn.ESP32_S3
    compiled = bakenn.compile(
        esp_nn_smoke_graph(),
        session.output / "esp32s3",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY, enable_esp_nn=True, target=target
        ),
        target=target,
    )
    kernels = [item.kernel_id for item in compiled.artifacts.backend_plan.selections]
    for operation in ("conv2d_s8", "depthwise_conv2d_s8", "add_s8"):
        _expect(
            any(kernel.startswith(f"esp_nn.esp32s3.{operation}") for kernel in kernels),
            f"ESP-NN {operation} was not selected: {kernels}",
        )
    project = bakenn.export_esp_idf_project(compiled.artifacts, target, session.output / "esp32s3_project")
    component = (project.component / "CMakeLists.txt").read_text()
    for path in (project.root / "CMakeLists.txt", project.main / "main.c"):
        _expect(path.is_file(), f"missing project file {path.name}")
    _expect("esp_nn_add_s8_esp32s3.S" in component, "the ESP-NN Add assembly is not in the component")
    esp_nn = sum(1 for kernel in kernels if kernel.startswith("esp_nn."))
    return f"esp_nn_kernels={esp_nn}/{len(kernels)} project={project.root.relative_to(session.output)}"


def check_arduino(session: Session) -> str:
    compiled = session.compiled
    library = bakenn.export_arduino_library(compiled.artifacts, session.output / "arduino_library")
    for path in (library.root / "library.properties", library.example):
        _expect(path.is_file(), f"missing library file {path.name}")
    compiler = _require("c++")
    build = session.output / "arduino_build"
    build.mkdir()
    (build / "Arduino.h").write_text(
        "#include <cstddef>\n#include <cstdint>\n#include <cstdio>\n"
        "struct SerialStub {\n"
        "  void begin(unsigned long) {}\n"
        "  void print(const char *text) { std::fputs(text, stdout); }\n"
        "  void print(char value) { std::fputc(value, stdout); }\n"
        "  void print(int value) { std::printf(\"%d\", value); }\n"
        "  void print(unsigned long value) { std::printf(\"%lu\", value); }\n"
        "  void println() { std::fputc('\\n', stdout); }\n"
        "};\n"
        "static SerialStub Serial;\n"
        "inline unsigned long micros() { return 0; }\n"
        "inline void delay(unsigned long) {}\n"
        "void setup();\nvoid loop();\n"
    )
    (build / "main.cpp").write_text("void setup();\nvoid loop();\nint main() { setup(); loop(); }\n")
    objects = []
    for source in sorted(library.source.glob("*.c")):
        objects.append(build / f"{source.stem}.o")
        subprocess.run(
            [session.compiler, *STRICT_C, "-I", str(library.source), "-c", str(source), "-o", str(objects[-1])],
            check=True,
        )
    # Arduino builds a sketch as C++ with Arduino.h included first.
    subprocess.run(
        [
            compiler, "-std=gnu++11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I", str(library.source), "-I", str(build), "-include", "Arduino.h",
            "-x", "c++", "-c", str(library.example), "-o", str(build / "sketch.o"),
        ],
        check=True,
    )
    subprocess.run(
        [compiler, str(build / "main.cpp"), str(build / "sketch.o"), *map(str, objects), "-o", str(build / "sketch")],
        check=True,
    )
    printed = subprocess.run([str(build / "sketch")], check=True, capture_output=True, text=True).stdout
    input_type = compiled.plan.tensors[compiled.plan.inputs[0]].tensor_type
    zero_input = np.full(input_type.shape, input_type.qparams.zero_point, dtype=np.int8)
    expected = bakenn.run_reference(compiled.plan, zero_input).reshape(-1).tolist()
    actual = [int(value) for value in printed.split("output=")[1].split()]
    _expect(actual == expected, f"sketch printed {actual}, reference is {expected}")
    return f"library={library.root.relative_to(session.output)} sketch_builds_as_cpp=1 sketch_output_matches_reference=1"


def check_manifest(session: Session) -> str:
    artifacts = session.compiled.artifacts
    bakenn.load_manifest(artifacts.manifest)
    tampered = session.output / "tampered"
    shutil.copytree(artifacts.output_dir, tampered)
    weights = tampered / artifacts.weights_source.name
    data = bytearray(weights.read_bytes())
    data[len(data) // 2] ^= 0x01
    weights.write_bytes(bytes(data))
    message = _rejection(lambda: bakenn.load_manifest(tampered / artifacts.manifest.name))
    _expect("manifest" in message, f"unexpected rejection: {message}")
    return "intact_artifacts=verified one_flipped_bit=rejected"


def check_deterministic(session: Session) -> str:
    def digests(directory: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir())
            if path.is_file()
        }

    first = digests(session.compiled.artifacts.output_dir)
    second = digests(session.compile("repeat").artifacts.output_dir)
    different = sorted(name for name in first if first[name] != second.get(name))
    _expect(first.keys() == second.keys() and not different, f"files differ between runs: {different}")
    return f"identical_files={len(first)}/{len(first)}"


def check_frozen_evidence(session: Session) -> str:
    from verify_mnist_evidence import verify

    result = verify(EVIDENCE, session.compiler)
    text = json.dumps(result, sort_keys=True)
    _expect('"mismatched_output_bytes": 0' in text, f"frozen evidence did not verify: {text[:200]}")
    return "frozen_generated_c=verified mismatched_output_bytes=0"


CHECKS = (
    Check("F01", "import the package and report versions", check_install),
    Check("F02", "compile a PyTorch FP32 model to heap-free INT8 C", check_compile),
    Check("F03", "generated C matches the integer reference byte for byte", check_c_matches_reference),
    Check("F04", "report FP32-versus-INT8 error per layer", check_accuracy_report),
    Check("F05", "report model memory and expose it in the C header", check_memory_report),
    Check("F06", "reject a model that exceeds the SRAM budget", check_budget),
    Check("F07", "reject an unsupported operator", check_unsupported),
    Check("F08", "select CMSIS-NN kernels and cross-link for Cortex-M4", check_cortex_m4),
    Check("F09", "export an ESP-IDF project with ESP-NN kernels", check_esp_idf),
    Check("F10", "export an Arduino library whose sketch runs the model", check_arduino),
    Check("F11", "detect a modified artifact through the manifest", check_manifest),
    Check("F12", "produce identical artifacts for identical inputs", check_deterministic),
    Check("F13", "re-verify the frozen MNIST evidence", check_frozen_evidence),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checks", nargs="*", help="check ids to run; default is all")
    parser.add_argument("--list", action="store_true", help="print ids and titles, then exit")
    parser.add_argument("--strict", action="store_true", help="treat a skipped check as a failure")
    parser.add_argument("--cc", default="cc", help="host C compiler")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "build/functional_checks",
        help="scratch directory; an earlier one from this script is replaced",
    )
    arguments = parser.parse_args()
    if arguments.list:
        for check in CHECKS:
            print(f"{check.identifier}  {check.title}")
        return 0
    known = {check.identifier for check in CHECKS}
    unknown = sorted(set(arguments.checks) - known)
    if unknown:
        parser.error(f"unknown check id: {', '.join(unknown)}")
    selected = [check for check in CHECKS if not arguments.checks or check.identifier in arguments.checks]

    # Only a directory this script created is ever deleted.
    output = arguments.output.resolve()
    marker = output / ".bakenn_functional_checks"
    if output.exists() and any(output.iterdir()) and not marker.is_file():
        parser.error(f"--output is not an earlier functional-checks directory: {output}")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    marker.write_text("scratch directory of scripts/functional_checks.py\n")
    session = Session(output, arguments.cc)

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for check in selected:
        try:
            status, detail = "PASS", check.run(session)
        except Skip as skipped:
            status, detail = "SKIP", str(skipped)
        except Exception as error:  # noqa: BLE001 - every failure becomes one result line
            status, detail = "FAIL", f"{type(error).__name__}: {error}"
        counts[status] += 1
        print(f"{check.identifier} {status}  {check.title}\n         {detail}", flush=True)
    print(f"\nRESULT pass={counts['PASS']} fail={counts['FAIL']} skip={counts['SKIP']} of {len(selected)}")
    return 1 if counts["FAIL"] or (arguments.strict and counts["SKIP"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
