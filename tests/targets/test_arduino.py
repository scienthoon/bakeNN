from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.artifacts import load_manifest
from bakenn.errors import CompileError
from bakenn.targets import CORTEX_M4, ESP32, ESP32_S3
from tests.p0.model_fixtures import mobilenet_v1_graph, residual_ds_cnn_graph
from tests.p2.support import require_compiler
from tests.p2.test_backend_selection import linear_graph


# The sketch only needs the Serial, micros and delay surface of an Arduino core.
_ARDUINO_STUB = """#include <cstddef>
#include <cstdint>
#include <cstdio>
struct SerialStub {
  void begin(unsigned long) {}
  void print(const char *text) { std::fputs(text, stdout); }
  void print(char value) { std::fputc(value, stdout); }
  void print(int value) { std::printf("%d", value); }
  void print(unsigned long value) { std::printf("%lu", value); }
  void println() { std::fputc('\\n', stdout); }
};
static SerialStub Serial;
inline unsigned long micros() { return 0; }
inline void delay(unsigned long) {}
void setup();
void loop();
"""


def test_arduino_library_has_the_specified_layout(tmp_path: Path) -> None:
    compiled = bakenn.compile(mobilenet_v1_graph(), tmp_path / "model")
    library = bakenn.export_arduino_library(compiled.artifacts, tmp_path / "library")
    symbol = library.model_symbol

    assert library.name == symbol
    assert library.example == tmp_path / "library/examples/Infer/Infer.ino"
    properties = dict(
        line.split("=", 1)
        for line in (library.root / "library.properties").read_text().splitlines()
    )
    assert properties["name"] == symbol
    assert properties["version"] == "1.0.0"
    assert properties["architectures"] == "*"
    assert properties["includes"] == f"{symbol}.h"
    assert {path.name for path in library.source.glob("*.c")} == {
        compiled.artifacts.model_source.name,
        compiled.artifacts.weights_source.name,
        compiled.artifacts.kernels_source.name,
    }
    # The copied artifact set stays complete, so its manifest still verifies.
    load_manifest(library.source / compiled.artifacts.manifest.name)
    sketch = library.example.read_text()
    assert f"#include <{symbol}.h>" in sketch
    assert f"{symbol}_infer(model_arena, model_input, model_output);" in sketch

    named = bakenn.export_arduino_library(
        compiled.artifacts, tmp_path / "named", name="Wake Word", version="2.3.4"
    )
    assert named.name == "Wake Word"
    assert "version=2.3.4\n" in (named.root / "library.properties").read_text()


def test_arduino_sketch_builds_as_cpp_and_matches_the_reference(tmp_path: Path) -> None:
    cc = require_compiler("cc")
    cxx = require_compiler("c++")
    compiled = bakenn.compile(mobilenet_v1_graph(), tmp_path / "model")
    library = bakenn.export_arduino_library(compiled.artifacts, tmp_path / "library")
    build = tmp_path / "build"
    build.mkdir()
    (build / "Arduino.h").write_text(_ARDUINO_STUB)
    (build / "main.cpp").write_text("void setup();\nvoid loop();\nint main() { setup(); loop(); }\n")
    objects = []
    for source in sorted(library.source.glob("*.c")):
        objects.append(build / f"{source.stem}.o")
        subprocess.run(
            [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
             "-I", str(library.source), "-c", str(source), "-o", str(objects[-1])],
            check=True, capture_output=True,
        )
    # Arduino builds a sketch as C++ with Arduino.h included first.
    subprocess.run(
        [cxx, "-std=gnu++11", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(library.source), "-I", str(build), "-include", "Arduino.h",
         "-x", "c++", "-c", str(library.example), "-o", str(build / "sketch.o")],
        check=True, capture_output=True,
    )
    runner = build / "runner"
    subprocess.run(
        [cxx, str(build / "main.cpp"), str(build / "sketch.o"),
         *(str(item) for item in objects), "-o", str(runner)],
        check=True, capture_output=True,
    )
    printed = subprocess.run([runner], check=True, capture_output=True, text=True).stdout
    assert printed.startswith("BAKENN inference_us=0 output= ")

    input_type = compiled.plan.tensors[compiled.plan.inputs[0]].tensor_type
    zero_input = np.full(input_type.shape, input_type.qparams.zero_point, dtype=np.int8)
    expected = bakenn.run_reference(compiled.plan, zero_input).reshape(-1)
    actual = [int(value) for value in printed.split("output=")[1].split()]
    assert actual == expected.tolist()


def test_arduino_export_fails_closed(tmp_path: Path) -> None:
    def artifacts(name, graph, **options):  # type: ignore[no-untyped-def]
        target = options["target"]
        return bakenn.compile(
            graph,
            tmp_path / name,
            backend_options=bakenn.CBackendOptions(
                kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY, **options
            ),
            target=target,
        ).artifacts

    rejected = (
        (
            artifacts("esp_nn", residual_ds_cnn_graph(), target=ESP32_S3, enable_esp_nn=True),
            "bundled CMSIS-NN or ESP-NN",
        ),
        (
            artifacts("dram", linear_graph(), target=ESP32, requantization_in_dram=True),
            "requantization_in_dram",
        ),
        (
            artifacts("smlad", linear_graph(32, 16), target=CORTEX_M4),
            "portable or generic optimized kernels",
        ),
    )
    for index, (rejected_artifacts, reason) in enumerate(rejected):
        output = tmp_path / f"rejected_{index}"
        with pytest.raises(CompileError, match=reason):
            bakenn.export_arduino_library(rejected_artifacts, output)
        assert not output.exists()

    portable = bakenn.compile(linear_graph(), tmp_path / "portable").artifacts
    for keyword, value in (("name", "ab"), ("name", "-model"), ("version", "1.0")):
        with pytest.raises(CompileError, match=f"invalid Arduino library {keyword}"):
            bakenn.export_arduino_library(portable, tmp_path / "invalid", **{keyword: value})
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "sketch.ino").write_text("// user sketch\n")
    with pytest.raises(CompileError, match="non-empty"):
        bakenn.export_arduino_library(portable, occupied)
    assert [path.name for path in occupied.iterdir()] == ["sketch.ino"]
