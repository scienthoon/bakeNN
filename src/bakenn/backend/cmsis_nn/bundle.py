from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import shutil

from bakenn.errors import CompileError


CMSIS_NN_REVISION = "ca5dc34313be2ee5c46652917c30baac96c52621"
CMSIS_NN_VERSION = "4.0.0"
CMSIS_CORE_VERSION = "5.9.0"

_FC_FILES = (
    "cmsis_nn/Include/arm_nn_math_types.h",
    "cmsis_nn/Include/arm_nn_types.h",
    "cmsis_nn/Include/arm_nnfunctions.h",
    "cmsis_nn/Include/arm_nnsupportfunctions.h",
    "cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_s8.c",
    "cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c",
    "cmsis_nn/LICENSE.txt",
    "cmsis_core/Include/cmsis_compiler.h",
    "cmsis_core/Include/cmsis_gcc.h",
    "cmsis_core/LICENSE.txt",
)


@dataclass(frozen=True)
class BundledCMSISNN:
    root: Path
    sources: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    license_files: tuple[Path, ...]


def bundle_fully_connected(output_dir: str | Path) -> BundledCMSISNN:
    """Copy the pinned FC source closure into one generated model directory."""

    output = Path(output_dir) / "third_party"
    vendor = files("bakenn.backend.cmsis_nn.vendor")
    copied: dict[str, Path] = {}
    for relative in _FC_FILES:
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
        sources=(
            copied["cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_s8.c"],
            copied["cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c"],
        ),
        include_dirs=(
            output / "cmsis_nn/Include",
            output / "cmsis_core/Include",
        ),
        license_files=(
            copied["cmsis_nn/LICENSE.txt"],
            copied["cmsis_core/LICENSE.txt"],
        ),
    )


__all__ = [
    "CMSIS_CORE_VERSION",
    "CMSIS_NN_REVISION",
    "CMSIS_NN_VERSION",
    "BundledCMSISNN",
    "bundle_fully_connected",
]
