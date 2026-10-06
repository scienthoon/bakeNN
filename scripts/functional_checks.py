#!/usr/bin/env python3
"""Run BakeNN's numbered functional tests offline; print one result line per test.

Every test uses only files in this repository: the frozen MNIST checkpoint,
calibration images and test images under ``examples/mnist/evidence``. Nothing
is downloaded and nothing is trained.

    python scripts/functional_checks.py            # all tests
    python scripts/functional_checks.py T07 T12    # selected tests
    python scripts/functional_checks.py --list     # ids and titles
    python scripts/functional_checks.py --strict   # a missing tool is a failure

A test passes only when its stated condition holds; ``SKIP`` means an optional
host tool (a cross compiler, a C++ compiler) is not installed. The same tests
run under pytest through ``tests/test_functional_checks.py``.
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
# Unrelated Arm cores, from ARMv6-M without hardware divide to ARMv8.1-M.
ARM_CORES = (
    "cortex-m0plus", "cortex-m3", "cortex-m4", "cortex-m7", "cortex-m33", "cortex-m55", "cortex-m85",
)


class Skip(Exception):
    """An optional host tool needed by this test is not installed."""


@dataclass(frozen=True)
class Check:
    identifier: str
    title: str
    run: Callable[["Session"], str]


class Session:
    """Shared inputs and intermediate results, each built once per run."""

    def __init__(self, output: Path, compiler: str) -> None:
        self.output = output
        self.compiler = compiler
        self._cache: dict[str, object] = {}
        self.evidence = json.loads((EVIDENCE / "mnist_evidence.json").read_text())
        calibration_shape = tuple(self.evidence["calibration"]["shape_nhw"])
        raw = np.fromfile(EVIDENCE / "calibration_images_u8.bin", dtype=np.uint8)
        self.calibration = (
            torch.from_numpy(raw.reshape(calibration_shape).copy()).unsqueeze(1).float() / 255.0
        )

    def once(self, key: str, build: Callable[[], object]):  # type: ignore[no-untyped-def]
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

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
        return self.once("portable", lambda: self.compile("portable"))


def run_check(check: Check, session: Session) -> str:
    """Run one test and return its evidence line."""

    # Finder drops .DS_Store into a folder that is open on screen, and the
    # manifest check rejects any file it did not hash.
    for stray in session.output.rglob(".DS_Store"):
        stray.unlink(missing_ok=True)
    return check.run(session)


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


def _host_run(session: Session) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the generated C for the host and run the 100 frozen test images once."""

    def build() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        shape = tuple(session.evidence["physical_test_corpus"]["input_shape_nhwc"])
        pixels = np.fromfile(EVIDENCE / "physical_test_images_u8.bin", dtype=np.uint8)
        inputs = quantize_mnist_corpus(session.compiled.plan, pixels.reshape(shape[:-1]))
        expected = np.fromfile(EVIDENCE / "physical_expected_outputs_int8.bin", dtype=np.int8)
        expected = expected.reshape(shape[0], -1)
        labels = np.fromfile(EVIDENCE / "physical_test_labels_u8.bin", dtype=np.uint8)
        completed = subprocess.run(
            [str(executable)], input=inputs.tobytes(), capture_output=True, check=True
        )
        actual = np.frombuffer(completed.stdout, dtype=np.int8).reshape(expected.shape)
        return inputs, actual, expected, labels

    return session.once("host_run", build)


def _esp32s3(session: Session):  # type: ignore[no-untyped-def]
    def build():  # type: ignore[no-untyped-def]
        from generate_smoke import esp_nn_smoke_graph

        target = bakenn.ESP32_S3
        return bakenn.compile(
            esp_nn_smoke_graph(),
            session.output / "esp32s3",
            backend_options=bakenn.CBackendOptions(
                kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY, enable_esp_nn=True, target=target
            ),
            target=target,
        )

    return session.once("esp32s3", build)


def _arduino(session: Session):  # type: ignore[no-untyped-def]
    return session.once(
        "arduino",
        lambda: bakenn.export_arduino_library(
            session.compiled.artifacts, session.output / "arduino_library"
        ),
    )


def check_install(session: Session) -> str:
    del session
    return (
        f"bakenn={bakenn.__version__} python={sys.version.split()[0]} "
        f"torch={torch.__version__.split('+')[0]}"
    )


def check_compile(session: Session) -> str:
    artifacts = session.compiled.artifacts
    bakenn.load_manifest(artifacts.manifest)
    generated = "\n".join(
        path.read_text() for path in (artifacts.model_source, artifacts.kernels_source)
    )
    for forbidden in ("malloc(", "calloc(", "free(", "float ", "double "):
        _expect(forbidden not in generated, f"generated C contains {forbidden!r}")
    steps = len(artifacts.backend_plan.selections)
    return f"steps={steps} heap_calls=0 float_types=0 manifest=verified"


def check_generated_files(session: Session) -> str:
    artifacts = session.compiled.artifacts
    expected = (
        artifacts.header, artifacts.model_source, artifacts.weights_header,
        artifacts.weights_source, artifacts.kernels_header, artifacts.kernels_source,
        artifacts.manifest, artifacts.memory_report_json, artifacts.memory_report_text,
        artifacts.build_fragment,
    )
    present = [path for path in expected if path.is_file() and path.stat().st_size > 0]
    names = {path.name for path in expected}
    unexpected = sorted(
        path.name
        for path in artifacts.output_dir.iterdir()
        if path.name not in names and not path.name.startswith(".")
    )
    _expect(len(present) == len(expected), "a generated file is missing or empty")
    _expect(not unexpected, f"unexpected generated files: {unexpected}")
    return f"expected_files_present={len(present)}/{len(expected)} unexpected_files=0"


def check_deterministic(session: Session) -> str:
    def digests(directory: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir())
            if path.is_file() and not path.name.startswith(".")
        }

    first = digests(session.compiled.artifacts.output_dir)
    second = digests(session.compile("repeat").artifacts.output_dir)
    different = sorted(name for name in first if first[name] != second.get(name))
    _expect(first.keys() == second.keys() and not different, f"files differ between runs: {different}")
    return f"identical_files={len(first)}/{len(first)}"


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


def check_c_matches_frozen_outputs(session: Session) -> str:
    _, actual, expected, labels = _host_run(session)
    mismatched = int(np.count_nonzero(actual != expected))
    correct = int(np.count_nonzero(np.argmax(actual, axis=1) == labels))
    _expect(mismatched == 0, f"{mismatched} bytes differ from the frozen expected outputs")
    return f"mismatched_bytes={mismatched}/{actual.size} correct={correct}/{len(labels)}"


def check_reference_matches_c(session: Session) -> str:
    inputs, actual, _, _ = _host_run(session)
    reference = np.concatenate(
        [bakenn.run_reference(session.compiled.plan, inputs[index : index + 1]) for index in range(10)]
    ).reshape(10, -1)
    mismatched = int(np.count_nonzero(actual[:10] != reference))
    _expect(mismatched == 0, f"{mismatched} bytes differ from the Python reference")
    return f"python_reference_mismatched_bytes={mismatched}/{reference.size}"


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
    manifest = bakenn.load_manifest(artifacts.manifest)
    arena = session.compiled.memory_report.arena_bytes
    macro = f"#define {artifacts.model_source.stem.upper()}_ARENA_SIZE {arena}u"
    _expect(manifest["arena_bytes"] == arena, "manifest arena size differs from the memory report")
    _expect(macro in artifacts.header.read_text(), "header arena size differs from the memory report")
    return f"arena_bytes={arena} constant_bytes={manifest['constant_bytes']} header_macro_matches=1"


def check_memory_report_file(session: Session) -> str:
    artifacts = session.compiled.artifacts
    arena = session.compiled.memory_report.arena_bytes
    text = artifacts.memory_report_text.read_text()
    report = json.loads(artifacts.memory_report_json.read_text())
    _expect(f"({arena} B)" in text, "the text report does not list the arena size")
    _expect(isinstance(report, dict) and bool(report), "the JSON report is empty")
    return f"report_file={artifacts.memory_report_text.name} lists_arena_bytes=1 json_report=1"


def check_manifest(session: Session) -> str:
    artifacts = session.compiled.artifacts
    bakenn.load_manifest(artifacts.manifest)
    tampered = session.output / "tampered"
    shutil.rmtree(tampered, ignore_errors=True)
    shutil.copytree(artifacts.output_dir, tampered, ignore=shutil.ignore_patterns(".*"))
    weights = tampered / artifacts.weights_source.name
    data = bytearray(weights.read_bytes())
    data[len(data) // 2] ^= 0x01
    weights.write_bytes(bytes(data))
    message = _rejection(lambda: bakenn.load_manifest(tampered / artifacts.manifest.name))
    _expect("manifest" in message, f"unexpected rejection: {message}")
    return "intact_artifacts=verified one_flipped_bit=rejected"


def check_frozen_evidence(session: Session) -> str:
    from verify_mnist_evidence import verify

    result = verify(EVIDENCE, session.compiler)
    text = json.dumps(result, sort_keys=True)
    _expect('"mismatched_output_bytes": 0' in text, f"frozen evidence did not verify: {text[:200]}")
    return "frozen_generated_c=verified mismatched_output_bytes=0"


def check_any_cpu(session: Session) -> str:
    """The default output is plain C, not code for one chip family."""

    artifacts = session.compiled.artifacts
    kernels = {item.kernel_id for item in artifacts.backend_plan.selections}
    _expect(all(kernel.startswith("portable.") for kernel in kernels), f"non-portable kernels: {kernels}")
    for source in (artifacts.model_source, artifacts.weights_source, artifacts.kernels_source):
        subprocess.run(
            [
                session.compiler, "-std=c99", "-pedantic-errors", "-Wall", "-Wextra", "-Werror",
                "-fsyntax-only", "-I", str(artifacts.output_dir), str(source),
            ],
            check=True,
        )
    _require("arm-none-eabi-gcc")
    sources: set[str] = set()
    for cpu in ARM_CORES:
        target = bakenn.TargetDescriptor(
            target_id=f"any-{cpu}",
            architecture=bakenn.TargetArchitecture.ARM,
            cpu=cpu,
            abi="aapcs32-soft",
            toolchain="arm-none-eabi",
            features=frozenset({"scalar-int8", "thumb"}),
            arena_alignment=8,
            constant_alignment=4,
            compiler_flags=(f"-mcpu={cpu}", "-mthumb", "-mfloat-abi=soft"),
        )
        # Only the backend depends on the target, so reuse the quantized graph.
        built = bakenn.compile(
            session.compiled.graph, session.output / f"any_cpu/{cpu}", target=target
        ).artifacts
        report = bakenn.build_freestanding_elf(built, target, session.output / f"any_cpu/{cpu}_elf")
        _expect(not report.undefined_symbols, f"{cpu}: undefined symbols {report.undefined_symbols}")
        _expect(not report.forbidden_symbols, f"{cpu}: heap or float symbols {report.forbidden_symbols}")
        sources.add(
            hashlib.sha256(
                b"".join(
                    path.read_bytes()
                    for path in (built.header, built.model_source, built.weights_source, built.kernels_source)
                )
            ).hexdigest()
        )
    _expect(len(sources) == 1, "the generated C differs between cores")
    return f"strict_c99=ok arm_cores_linked={len(ARM_CORES)}/{len(ARM_CORES)} identical_c_for_all_cores=1"


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


def check_esp_nn_selection(session: Session) -> str:
    kernels = [item.kernel_id for item in _esp32s3(session).artifacts.backend_plan.selections]
    for operation in ("conv2d_s8", "depthwise_conv2d_s8", "add_s8"):
        _expect(
            any(kernel.startswith(f"esp_nn.esp32s3.{operation}") for kernel in kernels),
            f"ESP-NN {operation} was not selected: {kernels}",
        )
    esp_nn = sum(1 for kernel in kernels if kernel.startswith("esp_nn."))
    return f"esp_nn_kernels={esp_nn}/{len(kernels)} conv=1 depthwise=1 add=1"


def check_esp_idf_export(session: Session) -> str:
    project = session.once(
        "esp_project",
        lambda: bakenn.export_esp_idf_project(
            _esp32s3(session).artifacts, bakenn.ESP32_S3, session.output / "esp32s3_project"
        ),
    )
    files = (
        project.root / "CMakeLists.txt",
        project.main / "main.c",
        project.component / "CMakeLists.txt",
    )
    for path in files:
        _expect(path.is_file(), f"missing project file {path.name}")
    _expect(
        "esp_nn_add_s8_esp32s3.S" in files[2].read_text(),
        "the ESP-NN Add assembly is not in the component",
    )
    return f"project_files_present={len(files)}/{len(files)} esp_nn_add_assembly_listed=1"


def check_arduino_export(session: Session) -> str:
    library = _arduino(session)
    header = library.source / f"{library.model_symbol}.h"
    files = (library.root / "library.properties", header, library.example)
    for path in files:
        _expect(path.is_file(), f"missing library file {path.name}")
    properties = files[0].read_text()
    _expect(f"includes={header.name}\n" in properties, "library.properties does not name the header")
    return f"library_files_present={len(files)}/{len(files)} properties_name_header=1"


def check_arduino_sketch(session: Session) -> str:
    compiled = session.compiled
    library = _arduino(session)
    compiler = _require("c++")
    build = session.output / "arduino_build"
    build.mkdir(exist_ok=True)
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
    return "sketch_builds_as_cpp=1 sketch_output_matches_reference=1"


CHECKS = (
    Check("T01", "import the package and report versions", check_install),
    Check("T02", "compile a PyTorch FP32 model to heap-free INT8 C", check_compile),
    Check("T03", "write the ten generated files", check_generated_files),
    Check("T04", "produce identical artifacts for identical inputs", check_deterministic),
    Check("T05", "reject a model that exceeds the SRAM budget", check_budget),
    Check("T06", "reject an unsupported operator", check_unsupported),
    Check("T07", "generated C reproduces the frozen expected outputs", check_c_matches_frozen_outputs),
    Check("T08", "Python integer reference matches the generated C", check_reference_matches_c),
    Check("T09", "report FP32-versus-INT8 error per layer", check_accuracy_report),
    Check("T10", "report model memory and expose it in the C header", check_memory_report),
    Check("T11", "write the memory report files", check_memory_report_file),
    Check("T12", "detect a modified artifact through the manifest", check_manifest),
    Check("T13", "re-verify the frozen MNIST evidence", check_frozen_evidence),
    Check("T14", "build the same generated C for unrelated CPU cores", check_any_cpu),
    Check("T15", "select CMSIS-NN kernels and cross-link for Cortex-M4", check_cortex_m4),
    Check("T16", "select ESP-NN kernels for ESP32-S3", check_esp_nn_selection),
    Check("T17", "export an ESP-IDF project", check_esp_idf_export),
    Check("T18", "export an Arduino library", check_arduino_export),
    Check("T19", "the Arduino sketch builds as C++ and runs the model", check_arduino_sketch),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checks", nargs="*", help="test ids to run; default is all")
    parser.add_argument("--list", action="store_true", help="print ids and titles, then exit")
    parser.add_argument("--strict", action="store_true", help="treat a skipped test as a failure")
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
        parser.error(f"unknown test id: {', '.join(unknown)}")
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
            status, detail = "PASS", run_check(check, session)
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
