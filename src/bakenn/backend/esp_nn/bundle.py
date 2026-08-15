from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import shutil

from bakenn.errors import CompileError


ESP_NN_REVISION = "c0876179f1cf4b4b9073b4f81cb65c8051ccb476"
ESP_NN_VERSION = "1.2.6"

_BASE_SOURCES_BY_FAMILY = {
    "conv": (
        "src/convolution/esp_nn_conv_ansi.c",
        "src/convolution/esp_nn_conv_opt.c",
    ),
    "depthwise": (
        "src/convolution/esp_nn_depthwise_conv_ansi.c",
        "src/convolution/esp_nn_depthwise_conv_opt.c",
    ),
    "linear": ("src/fully_connected/esp_nn_fully_connected_ansi.c",),
    "average_pool": ("src/pooling/esp_nn_avg_pool_ansi.c",),
    "max_pool": ("src/pooling/esp_nn_max_pool_ansi.c",),
}

_ESP32S3_COMMON_SOURCES = (
    "src/common/esp_nn_common_functions_esp32s3.S",
    "src/common/esp_nn_dot_s8_esp32s3.S",
    "src/common/esp_nn_multiply_by_quantized_mult_esp32s3.S",
    "src/common/esp_nn_multiply_by_quantized_mult_ver1_esp32s3.S",
)

_ESP32S3_SOURCES_BY_FAMILY = {
    "conv": (
        "src/convolution/esp_nn_conv_esp32s3.c",
        "src/convolution/esp_nn_conv_s8_1x1_esp32s3.c",
        "src/convolution/esp_nn_conv_s8_3x3_opt_esp32s3.c",
        "src/convolution/esp_nn_conv_s16_mult8_esp32s3.S",
        "src/convolution/esp_nn_conv_s8_mult8_1x1_esp32s3.S",
        "src/convolution/esp_nn_conv_s16_mult4_1x1_esp32s3.S",
        "src/convolution/esp_nn_conv_s8_filter_aligned_input_padded_esp32s3.S",
    ),
    "depthwise": (
        # The S3 kernel falls back to this implementation for unsupported
        # channel/window combinations, so it is part of the target closure.
        "src/convolution/esp_nn_depthwise_conv_opt.c",
        "src/convolution/esp_nn_depthwise_conv_s8_esp32s3.c",
        "src/convolution/esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3.S",
        "src/convolution/esp_nn_depthwise_conv_s16_mult1_esp32s3.S",
        "src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_esp32s3.S",
        "src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_no_pad_esp32s3.S",
        "src/convolution/esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3.S",
        "src/convolution/esp_nn_depthwise_conv_s16_mult4_esp32s3.S",
        "src/convolution/esp_nn_depthwise_conv_s16_mult8_esp32s3.S",
    ),
    "linear": (
        "src/fully_connected/esp_nn_fully_connected_esp32s3.c",
        "src/fully_connected/esp_nn_fully_connected_s8_esp32s3.S",
        "src/fully_connected/esp_nn_fully_connected_per_ch_s8_esp32s3.S",
    ),
    "average_pool": (
        "src/pooling/esp_nn_avg_pool_s8_esp32s3.c",
        "src/pooling/esp_nn_avg_pool_s8_esp32s3.S",
    ),
    "max_pool": ("src/pooling/esp_nn_max_pool_s8_esp32s3.S",),
}

_SUPPORTED_PREFIXES = (
    "esp_nn.esp32.",
    "esp_nn.esp32s3.",
)


def _kernel_family(kernel_id: str) -> str:
    if ".depthwise_conv2d_" in kernel_id:
        return "depthwise"
    if ".conv2d_" in kernel_id:
        return "conv"
    if ".linear_" in kernel_id:
        return "linear"
    if ".average_pool2d_" in kernel_id:
        return "average_pool"
    if ".max_pool2d_" in kernel_id:
        return "max_pool"
    raise CompileError(f"unknown ESP-NN kernel family: {kernel_id}")


@dataclass(frozen=True)
class BundledESPNN:
    root: Path
    sources: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    license_files: tuple[Path, ...]
    target_id: str


def bundle_kernels(
    output_dir: str | Path,
    kernel_ids: tuple[str, ...],
    target_id: str,
) -> BundledESPNN:
    """Copy the exact pinned ESP-NN source closure for one ESP-IDF target."""

    selected_ids = tuple(sorted(set(kernel_ids)))
    if not selected_ids:
        raise CompileError("ESP-NN bundling requires at least one selected kernel")
    if target_id not in {"esp32", "esp32s3"}:
        raise CompileError(f"ESP-NN does not support BakeNN target {target_id}")
    expected_prefix = f"esp_nn.{target_id}."
    invalid = [
        value
        for value in selected_ids
        if not value.startswith(expected_prefix)
        or not value.endswith(f".v{ESP_NN_VERSION}")
    ]
    if invalid or any(
        value.startswith(_SUPPORTED_PREFIXES) and not value.startswith(expected_prefix)
        for value in selected_ids
    ):
        raise CompileError(
            f"ESP-NN kernel ids do not match target {target_id}: {invalid or selected_ids}"
        )

    output = Path(output_dir) / "third_party" / "esp_nn"
    vendor = files("bakenn.backend.esp_nn.vendor").joinpath("esp_nn")
    families = tuple(dict.fromkeys(_kernel_family(value) for value in selected_ids))
    selected_sources: list[str] = []
    for family in families:
        base_sources = _BASE_SOURCES_BY_FAMILY[family]
        if target_id == "esp32s3":
            # S3 dispatches directly to its target implementation. Keep the
            # ANSI source as the host differential oracle and let the exact
            # family closure below add any target-internal fallback source.
            base_sources = tuple(
                value for value in base_sources if not value.endswith("_opt.c")
            )
        selected_sources.extend(base_sources)
    if target_id == "esp32s3":
        selected_sources.extend(_ESP32S3_COMMON_SOURCES)
        for family in families:
            selected_sources.extend(_ESP32S3_SOURCES_BY_FAMILY[family])
    source_files = tuple(dict.fromkeys(selected_sources))
    header_files = tuple(
        f"include/{name}"
        for name in (
            "esp_nn.h",
            "esp_nn_ansi_c.h",
            "esp_nn_ansi_headers.h",
            "esp_nn_defs.h",
            "esp_nn_esp32p4.h",
            "esp_nn_esp32s3.h",
            "esp_nn_generic_opt.h",
        )
    )
    copied: dict[str, Path] = {}
    for relative in (
        *header_files,
        "src/common/common_functions.h",
        "src/softmax/softmax_common.h",
        "LICENSE",
        *source_files,
    ):
        source = vendor.joinpath(relative)
        if not source.is_file():
            raise CompileError(f"packaged ESP-NN resource is missing: {relative}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_file, destination.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
        copied[relative] = destination

    return BundledESPNN(
        root=output,
        sources=tuple(copied[value] for value in source_files),
        include_dirs=(output / "include", output / "src/common"),
        license_files=(copied["LICENSE"],),
        target_id=target_id,
    )


__all__ = [
    "ESP_NN_REVISION",
    "ESP_NN_VERSION",
    "BundledESPNN",
    "bundle_kernels",
]
