"""Pinned CMSIS-NN source bundles used by target-specific BakeNN kernels."""

from .bundle import CMSIS_NN_REVISION, BundledCMSISNN, bundle_fully_connected

__all__ = ["CMSIS_NN_REVISION", "BundledCMSISNN", "bundle_fully_connected"]
