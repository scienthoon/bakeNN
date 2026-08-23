"""Versioned, verifiable BakeNN generated-artifact metadata."""

from .manifest import (
    ARITHMETIC_PROFILE_VERSION,
    GENERATED_C_ABI_VERSION,
    MANIFEST_SCHEMA_VERSION,
    build_artifact_inventory,
    canonical_plan_fingerprints,
    compiler_source_identity,
    load_manifest,
    manifest_payload_sha256,
    validate_manifest,
)

__all__ = [
    "ARITHMETIC_PROFILE_VERSION",
    "GENERATED_C_ABI_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "build_artifact_inventory",
    "canonical_plan_fingerprints",
    "compiler_source_identity",
    "load_manifest",
    "manifest_payload_sha256",
    "validate_manifest",
]
