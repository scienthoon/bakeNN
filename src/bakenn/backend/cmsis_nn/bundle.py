from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import shutil

from bakenn.errors import CompileError


CMSIS_NN_REVISION = "ca5dc34313be2ee5c46652917c30baac96c52621"
CMSIS_NN_VERSION = "4.0.0"
CMSIS_CORE_VERSION = "5.9.0"

_COMMON_FILES = (
    "cmsis_nn/Include/arm_nn_math_types.h",
    "cmsis_nn/Include/arm_nn_types.h",
    "cmsis_nn/Include/arm_nnfunctions.h",
    "cmsis_nn/Include/arm_nnsupportfunctions.h",
    "cmsis_nn/LICENSE.txt",
    "cmsis_core/Include/cmsis_compiler.h",
    "cmsis_core/Include/cmsis_gcc.h",
    "cmsis_core/LICENSE.txt",
)

_KERNEL_SOURCES = {
    "cmsis_nn.linear_s8.v4.0.0": (
        "cmsis_nn/Source/NNSupportFunctions/bakenn_cmsis_memory.c",
        "cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_s8.c",
        "cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c",
    ),
    "cmsis_nn.conv2d_s8.v4.0.0": (
        "cmsis_nn/Source/NNSupportFunctions/bakenn_cmsis_memory.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_convolve_s8.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c",
        "cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c",
        "cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c",
        "cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
        "cmsis_nn/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
    ),
    "cmsis_nn.depthwise_conv2d_s8.v4.0.0": (
        "cmsis_nn/Source/NNSupportFunctions/bakenn_cmsis_memory.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c",
        "cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c",
        "cmsis_nn/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c",
        "cmsis_nn/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c",
        "cmsis_nn/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
    ),
    "cmsis_nn.average_pool2d_s8.v4.0.0": (
        "cmsis_nn/Source/NNSupportFunctions/bakenn_cmsis_memory.c",
        "cmsis_nn/Source/PoolingFunctions/arm_avgpool_s8.c",
    ),
    "cmsis_nn.max_pool2d_s8.v4.0.0": (
        "cmsis_nn/Source/NNSupportFunctions/bakenn_cmsis_memory.c",
        "cmsis_nn/Source/PoolingFunctions/arm_max_pool_s8.c",
    ),
}


@dataclass(frozen=True)
class BundledCMSISNN:
    root: Path
    sources: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    license_files: tuple[Path, ...]


def bundle_kernels(
    output_dir: str | Path,
    kernel_ids: tuple[str, ...],
) -> BundledCMSISNN:
    """Copy the exact pinned source closure for selected CMSIS-NN kernels."""

    output = Path(output_dir) / "third_party"
    vendor = files("bakenn.backend.cmsis_nn.vendor")
    selected_ids = tuple(sorted(set(kernel_ids)))
    if not selected_ids:
        raise CompileError("CMSIS-NN bundling requires at least one selected kernel")
    unknown = set(selected_ids) - set(_KERNEL_SOURCES)
    if unknown:
        raise CompileError(f"no pinned CMSIS-NN source closure for {sorted(unknown)}")
    source_files = tuple(
        sorted(
            {
                relative
                for kernel_id in selected_ids
                for relative in _KERNEL_SOURCES[kernel_id]
            }
        )
    )
    copied: dict[str, Path] = {}
    for relative in (*_COMMON_FILES, *source_files):
        source = vendor.joinpath(relative)
        if not source.is_file():
            raise CompileError(f"packaged CMSIS-NN resource is missing: {relative}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_file, destination.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
        copied[relative] = destination

    return BundledCMSISNN(
        root=output,
        sources=tuple(copied[relative] for relative in source_files),
        include_dirs=(
            output / "cmsis_nn/Include",
            output / "cmsis_core/Include",
        ),
        license_files=(
            copied["cmsis_nn/LICENSE.txt"],
            copied["cmsis_core/LICENSE.txt"],
        ),
    )


def bundle_fully_connected(output_dir: str | Path) -> BundledCMSISNN:
    """Backward-compatible wrapper for the pinned FC-only source closure."""

    return bundle_kernels(output_dir, ("cmsis_nn.linear_s8.v4.0.0",))


__all__ = [
    "CMSIS_CORE_VERSION",
    "CMSIS_NN_REVISION",
    "CMSIS_NN_VERSION",
    "BundledCMSISNN",
    "bundle_fully_connected",
    "bundle_kernels",
]
