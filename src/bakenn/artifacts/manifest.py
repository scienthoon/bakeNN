"""Strict schema and deterministic fingerprints for generated C artifacts.

The manifest deliberately excludes itself from the generated-file inventory:
including its own digest would create a recursive value.  The inventory records
that exclusion and binds every other emitted byte to one canonical set hash.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from bakenn.errors import CompileError
from bakenn.plan import ExecutionPlan


GENERATED_C_ABI_VERSION = 1
MANIFEST_SCHEMA_VERSION = 4
ARITHMETIC_PROFILE_VERSION = 1

_GRAPH_HASH_DOMAIN = "bakenn.canonical-lowered-graph.v1"
_PLAN_HASH_DOMAIN = "bakenn.execution-plan.v1"
_CONSTANT_HASH_DOMAIN = "bakenn.semantic-constants.v1"
_ARTIFACT_SET_HASH_DOMAIN = "bakenn.generated-artifact-set.v1"
_MANIFEST_PAYLOAD_HASH_DOMAIN = "bakenn.manifest-payload.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _fail(message: str) -> CompileError:
    return CompileError(f"invalid BakeNN manifest: {message}")


def _canonical_array(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    dtype = array.dtype
    if dtype.byteorder in ("=", ">") and dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    payload = array.tobytes(order="C")
    return {
        "$ndarray": {
            "dtype": dtype.str,
            "shape": [int(dimension) for dimension in array.shape],
            "data_sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    }


def _canonical_value(value: object) -> object:
    """Convert compiler values into an unambiguous JSON value."""

    if isinstance(value, Enum):
        return {
            "$enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical_value(value.value),
        }
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise CompileError("canonical compiler metadata cannot contain non-finite floats")
        return {"$float64_hex": value.hex()}
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, np.ndarray):
        return _canonical_array(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _canonical_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CompileError("canonical compiler metadata requires string mapping keys")
        return {
            key: _canonical_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    raise CompileError(
        f"canonical compiler metadata does not support {type(value).__name__}"
    )


def _canonical_sha256(value: object, domain: str) -> str:
    document = {
        "domain": domain,
        "value": _canonical_value(value),
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_payload_sha256(document: Mapping[str, object]) -> str:
    """Hash every manifest field except the digest that stores this result."""

    payload = dict(document)
    payload.pop("manifest_payload_sha256", None)
    return _canonical_sha256(payload, _MANIFEST_PAYLOAD_HASH_DOMAIN)


def canonical_plan_fingerprints(plan: ExecutionPlan) -> dict[str, object]:
    """Return canonical semantic and physical-plan fingerprints."""

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    semantic_graph = {
        "name": plan.name,
        "tensors": {
            name: tensor.tensor_type for name, tensor in plan.tensors.items()
        },
        "constants": dict(plan.constants),
        "steps": plan.steps,
        "inputs": plan.inputs,
        "outputs": plan.outputs,
        "arithmetic_profile": plan.arithmetic_profile,
    }
    return {
        "canonical_graph_hash_domain": _GRAPH_HASH_DOMAIN,
        "canonical_graph_sha256": _canonical_sha256(
            semantic_graph, _GRAPH_HASH_DOMAIN
        ),
        "execution_plan_hash_domain": _PLAN_HASH_DOMAIN,
        "execution_plan_sha256": _canonical_sha256(plan, _PLAN_HASH_DOMAIN),
        "semantic_constants_hash_domain": _CONSTANT_HASH_DOMAIN,
        "semantic_constants_sha256": _canonical_sha256(
            dict(plan.constants), _CONSTANT_HASH_DOMAIN
        ),
    }


def _git_source_identity() -> tuple[str | None, bool | None, str]:
    explicit_revision = os.environ.get("BAKENN_SOURCE_REVISION")
    explicit_dirty = os.environ.get("BAKENN_SOURCE_DIRTY")
    if explicit_revision is not None:
        revision = explicit_revision.strip()
        if not revision:
            raise CompileError("BAKENN_SOURCE_REVISION must not be empty")
        if explicit_dirty is None:
            dirty: bool | None = None
        elif explicit_dirty in ("0", "false", "False"):
            dirty = False
        elif explicit_dirty in ("1", "true", "True"):
            dirty = True
        else:
            raise CompileError("BAKENN_SOURCE_DIRTY must be 0/1 or false/true")
        return revision, dirty, "environment"

    start = Path(__file__).resolve()
    try:
        root_result = subprocess.run(
            ["git", "-C", str(start.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        root = Path(root_result.stdout.strip())
        # A wheel is commonly installed into ``.venv`` inside an unrelated
        # application repository.  Walking to that outer repository must not
        # mislabel the application's commit as the BakeNN compiler source.
        try:
            start.relative_to(root / "src" / "bakenn")
        except ValueError:
            return None, None, "unavailable"
        revision_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None, None, "unavailable"
    revision = revision_result.stdout.strip()
    if not revision:
        return None, None, "unavailable"
    return revision, bool(status_result.stdout), "git"


def compiler_source_identity(package_version: str) -> dict[str, object]:
    """Describe the package build and source revision when it is knowable."""

    if not isinstance(package_version, str) or not package_version:
        raise ValueError("package_version must be a non-empty string")
    revision, dirty, origin = _git_source_identity()
    build_identity = package_version
    if revision is not None:
        build_identity += f"+source.{revision[:12]}"
        if dirty:
            build_identity += ".dirty"
    return {
        "package_version": package_version,
        "build_identity": build_identity,
        "source_revision": revision,
        "source_dirty": dirty,
        "source_origin": origin,
    }


def _safe_relative_path(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{field_name} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise _fail(f"{field_name} must be a normalized relative POSIX path")
    return value


def build_artifact_inventory(
    root: str | Path,
    *,
    manifest_path: str,
) -> dict[str, object]:
    """Hash every regular emitted file other than the manifest itself."""

    directory = Path(root)
    manifest_relative = _safe_relative_path(
        manifest_path, field_name="artifact_inventory.manifest_path"
    )
    records: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise CompileError(f"generated artifacts must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CompileError(f"generated artifact is not a regular file: {path}")
        relative = path.relative_to(directory).as_posix()
        if relative == manifest_relative:
            continue
        payload = path.read_bytes()
        records.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "hash_domain": _ARTIFACT_SET_HASH_DOMAIN,
        "manifest_path": manifest_relative,
        "manifest_included": False,
        "files": records,
        "set_sha256": _canonical_sha256(records, _ARTIFACT_SET_HASH_DOMAIN),
    }


def _object_no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    name: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise _fail(f"{name} is missing keys {sorted(missing)}")
    if unknown:
        raise _fail(f"{name} has unknown keys {sorted(unknown)}")


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{name} must be an integer >= {minimum}")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{name} must be a non-empty string")
    return value


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_manifest(
    document: object,
    *,
    root: str | Path | None = None,
    verify_files: bool = False,
) -> dict[str, object]:
    """Validate schema v4 and optionally every generated file digest."""

    manifest = _require_mapping(document, "root")
    required = {
        "schema_version",
        "c_abi_version",
        "compiler_version",
        "compiler",
        "model",
        "arithmetic_profile",
        "arithmetic_profile_version",
        "graph_fingerprints",
        "backend",
        "arena_bytes",
        "activation_arena_bytes",
        "scratch_bytes",
        "scratch_offset",
        "scratch_alignment",
        "arena_alignment",
        "constant_bytes",
        "constant_payload_bytes",
        "constant_max_alignment",
        "memory_report",
        "input",
        "output",
        "operations",
        "artifact_inventory",
        "manifest_payload_hash_domain",
        "manifest_payload_sha256",
    }
    _require_exact_keys(
        manifest,
        "root",
        required=required,
        optional={"bundled_dependencies"},
    )
    schema_version = _require_int(manifest["schema_version"], "schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise _fail(
            f"schema_version must be {MANIFEST_SCHEMA_VERSION}, got "
            f"{manifest['schema_version']!r}"
        )
    c_abi_version = _require_int(manifest["c_abi_version"], "c_abi_version")
    if c_abi_version != GENERATED_C_ABI_VERSION:
        raise _fail(f"c_abi_version must be {GENERATED_C_ABI_VERSION}")
    profile_version = _require_int(
        manifest["arithmetic_profile_version"], "arithmetic_profile_version"
    )
    if profile_version != ARITHMETIC_PROFILE_VERSION:
        raise _fail(
            f"arithmetic_profile_version must be {ARITHMETIC_PROFILE_VERSION}"
        )
    for name in ("compiler_version", "model", "arithmetic_profile"):
        _require_string(manifest[name], name)
    if manifest["manifest_payload_hash_domain"] != _MANIFEST_PAYLOAD_HASH_DOMAIN:
        raise _fail("manifest payload hash domain is unsupported")
    payload_digest = _validate_digest(
        manifest["manifest_payload_sha256"], "manifest_payload_sha256"
    )
    if payload_digest != manifest_payload_sha256(manifest):
        raise _fail("manifest_payload_sha256 does not match the manifest payload")

    compiler = _require_mapping(manifest["compiler"], "compiler")
    _require_exact_keys(
        compiler,
        "compiler",
        required={
            "package_version",
            "build_identity",
            "source_revision",
            "source_dirty",
            "source_origin",
        },
    )
    if compiler["package_version"] != manifest["compiler_version"]:
        raise _fail("compiler.package_version must equal compiler_version")
    _require_string(compiler["build_identity"], "compiler.build_identity")
    if compiler["source_revision"] is not None:
        _require_string(compiler["source_revision"], "compiler.source_revision")
    if compiler["source_dirty"] is not None and not isinstance(
        compiler["source_dirty"], bool
    ):
        raise _fail("compiler.source_dirty must be boolean or null")
    source_origin = _require_string(
        compiler["source_origin"], "compiler.source_origin"
    )
    if source_origin not in {"git", "environment", "unavailable"}:
        raise _fail("compiler.source_origin is invalid")

    fingerprints = _require_mapping(
        manifest["graph_fingerprints"], "graph_fingerprints"
    )
    fingerprint_keys = {
        "canonical_graph_hash_domain",
        "canonical_graph_sha256",
        "execution_plan_hash_domain",
        "execution_plan_sha256",
        "semantic_constants_hash_domain",
        "semantic_constants_sha256",
    }
    _require_exact_keys(
        fingerprints, "graph_fingerprints", required=fingerprint_keys
    )
    if fingerprints["canonical_graph_hash_domain"] != _GRAPH_HASH_DOMAIN:
        raise _fail("canonical graph hash domain is unsupported")
    if fingerprints["execution_plan_hash_domain"] != _PLAN_HASH_DOMAIN:
        raise _fail("execution plan hash domain is unsupported")
    if fingerprints["semantic_constants_hash_domain"] != _CONSTANT_HASH_DOMAIN:
        raise _fail("semantic constants hash domain is unsupported")
    for name in (
        "canonical_graph_sha256",
        "execution_plan_sha256",
        "semantic_constants_sha256",
    ):
        _validate_digest(fingerprints[name], f"graph_fingerprints.{name}")

    backend = _require_mapping(manifest["backend"], "backend")
    _require_exact_keys(
        backend,
        "backend",
        required={
            "name",
            "target",
            "kernel_policy",
            "weight_packing",
            "cmsis_nn_enabled",
            "esp_nn_enabled",
            "optimized_steps",
            "selections",
        },
    )
    _require_string(backend["name"], "backend.name")
    _require_mapping(backend["target"], "backend.target")
    _require_string(backend["kernel_policy"], "backend.kernel_policy")
    for name in ("weight_packing", "cmsis_nn_enabled", "esp_nn_enabled"):
        if not isinstance(backend[name], bool):
            raise _fail(f"backend.{name} must be boolean")
    _require_int(backend["optimized_steps"], "backend.optimized_steps")
    selections = backend["selections"]
    if not isinstance(selections, list):
        raise _fail("backend.selections must be an array")
    selection_required = {
        "step_index",
        "step_name",
        "implementation",
        "optimized",
        "selection_reason",
        "packed_constants",
        "rejected_implementations",
        "scratch_bytes",
        "scratch_alignment",
        "selection_basis",
        "workload_key",
    }
    for index, raw_selection in enumerate(selections):
        selection = _require_mapping(raw_selection, f"backend.selections[{index}]")
        _require_exact_keys(
            selection,
            f"backend.selections[{index}]",
            required=selection_required,
            optional={"matched_measurement"},
        )
        step_index = _require_int(
            selection["step_index"], f"backend.selections[{index}].step_index"
        )
        if step_index != index:
            raise _fail("backend selection indexes must be contiguous execution order")
        for name in ("step_name", "implementation", "selection_reason"):
            _require_string(selection[name], f"backend.selections[{index}].{name}")
        if not isinstance(selection["optimized"], bool):
            raise _fail(f"backend.selections[{index}].optimized must be boolean")
        _require_int(selection["scratch_bytes"], "selection.scratch_bytes")
        _require_int(
            selection["scratch_alignment"],
            "selection.scratch_alignment",
            minimum=1,
        )
        if not isinstance(selection["packed_constants"], list):
            raise _fail("selection.packed_constants must be an array")
        _require_mapping(
            selection["rejected_implementations"],
            "selection.rejected_implementations",
        )
        selection_basis = _require_string(
            selection["selection_basis"],
            f"backend.selections[{index}].selection_basis",
        )
        workload_key = _require_string(
            selection["workload_key"],
            f"backend.selections[{index}].workload_key",
        )
        if "matched_measurement" in selection:
            measurement = _require_mapping(
                selection["matched_measurement"],
                f"backend.selections[{index}].matched_measurement",
            )
            _require_exact_keys(
                measurement,
                f"backend.selections[{index}].matched_measurement",
                required={
                    "kernel_id",
                    "workload",
                    "cycles",
                    "toolchain",
                    "compiler_flags",
                    "evidence",
                },
            )
            if measurement["kernel_id"] != selection["implementation"]:
                raise _fail("matched measurement kernel_id must equal implementation")
            if measurement["workload"] != workload_key:
                raise _fail("matched measurement workload must equal workload_key")
            _require_int(measurement["cycles"], "matched_measurement.cycles", minimum=1)
            for name in ("toolchain", "evidence"):
                _require_string(measurement[name], f"matched_measurement.{name}")
            flags = measurement["compiler_flags"]
            if not isinstance(flags, list) or any(
                not isinstance(flag, str) or not flag for flag in flags
            ):
                raise _fail("matched_measurement.compiler_flags must be strings")
            if not selection_basis.startswith("measured"):
                raise _fail("matched measurement requires a measured selection basis")

    for name in (
        "arena_bytes",
        "activation_arena_bytes",
        "scratch_bytes",
        "constant_bytes",
        "constant_payload_bytes",
    ):
        _require_int(manifest[name], name)
    for name in ("scratch_alignment", "arena_alignment", "constant_max_alignment"):
        _require_int(manifest[name], name, minimum=1)
    if manifest["scratch_offset"] is not None:
        _require_int(manifest["scratch_offset"], "scratch_offset")
    if manifest["constant_bytes"] != manifest["constant_payload_bytes"]:
        raise _fail("constant_bytes and constant_payload_bytes must match")
    for io_name in ("input", "output"):
        io = _require_mapping(manifest[io_name], io_name)
        _require_exact_keys(
            io,
            io_name,
            required={"shape", "dtype", "layout", "qparams"},
        )
        shape = io["shape"]
        if not isinstance(shape, list) or not shape:
            raise _fail(f"{io_name}.shape must be a non-empty array")
        for index, dimension in enumerate(shape):
            _require_int(dimension, f"{io_name}.shape[{index}]", minimum=1)
        _require_string(io["dtype"], f"{io_name}.dtype")
        _require_string(io["layout"], f"{io_name}.layout")
        _require_mapping(io["qparams"], f"{io_name}.qparams")
    if not isinstance(manifest["operations"], list):
        raise _fail("operations must be an array")
    if len(manifest["operations"]) != len(selections):
        raise _fail("operations and backend selections must have equal length")
    _require_mapping(manifest["memory_report"], "memory_report")
    if "bundled_dependencies" in manifest and not isinstance(
        manifest["bundled_dependencies"], list
    ):
        raise _fail("bundled_dependencies must be an array")

    inventory = _require_mapping(manifest["artifact_inventory"], "artifact_inventory")
    _require_exact_keys(
        inventory,
        "artifact_inventory",
        required={
            "schema_version",
            "hash_algorithm",
            "hash_domain",
            "manifest_path",
            "manifest_included",
            "files",
            "set_sha256",
        },
    )
    inventory_schema = _require_int(
        inventory["schema_version"], "artifact_inventory.schema_version"
    )
    if inventory_schema != 1:
        raise _fail("artifact_inventory.schema_version must be 1")
    if inventory["hash_algorithm"] != "sha256":
        raise _fail("artifact_inventory.hash_algorithm must be sha256")
    if inventory["hash_domain"] != _ARTIFACT_SET_HASH_DOMAIN:
        raise _fail("artifact inventory hash domain is unsupported")
    manifest_relative = _safe_relative_path(
        inventory["manifest_path"], field_name="artifact_inventory.manifest_path"
    )
    if inventory["manifest_included"] is not False:
        raise _fail("artifact manifest must be explicitly excluded from its own inventory")
    files = inventory["files"]
    if not isinstance(files, list):
        raise _fail("artifact_inventory.files must be an array")
    paths: list[str] = []
    for index, raw_record in enumerate(files):
        record = _require_mapping(raw_record, f"artifact_inventory.files[{index}]")
        _require_exact_keys(
            record,
            f"artifact_inventory.files[{index}]",
            required={"path", "bytes", "sha256"},
        )
        relative = _safe_relative_path(
            record["path"], field_name=f"artifact_inventory.files[{index}].path"
        )
        if relative == manifest_relative:
            raise _fail("artifact inventory cannot contain the manifest itself")
        paths.append(relative)
        _require_int(record["bytes"], f"artifact_inventory.files[{index}].bytes")
        _validate_digest(
            record["sha256"], f"artifact_inventory.files[{index}].sha256"
        )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise _fail("artifact inventory file paths must be unique and sorted")
    expected_set_hash = _canonical_sha256(files, _ARTIFACT_SET_HASH_DOMAIN)
    if inventory["set_sha256"] != expected_set_hash:
        raise _fail("artifact inventory set_sha256 does not match its file records")

    if verify_files:
        if root is None:
            raise _fail("root is required when verify_files=True")
        directory = Path(root)
        if not directory.is_dir() or directory.is_symlink():
            raise _fail("artifact root must be a real directory")
        actual_files: set[str] = set()
        actual_dirs: set[str] = set()
        for path in directory.rglob("*"):
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise _fail(f"artifact tree contains symlink {relative}")
            if path.is_dir():
                actual_dirs.add(relative)
            elif path.is_file():
                actual_files.add(relative)
            else:
                raise _fail(f"artifact tree contains non-regular entry {relative}")
        expected_files = set(paths) | {manifest_relative}
        if actual_files != expected_files:
            raise _fail(
                "artifact tree file set differs from the hashed inventory "
                f"(missing={sorted(expected_files - actual_files)}, "
                f"unexpected={sorted(actual_files - expected_files)})"
            )
        expected_dirs = {
            PurePosixPath(relative).parent.as_posix()
            for relative in expected_files
            if PurePosixPath(relative).parent.as_posix() != "."
        }
        expected_dirs |= {
            parent.as_posix()
            for relative in expected_files
            for parent in PurePosixPath(relative).parents
            if parent.as_posix() != "."
        }
        if actual_dirs != expected_dirs:
            raise _fail(
                "artifact tree contains missing or untracked directories "
                f"(missing={sorted(expected_dirs - actual_dirs)}, "
                f"unexpected={sorted(actual_dirs - expected_dirs)})"
            )
        for record in files:
            relative = str(record["path"])
            payload = (directory / relative).read_bytes()
            if len(payload) != record["bytes"]:
                raise _fail(f"artifact {relative} byte count does not match inventory")
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise _fail(f"artifact {relative} SHA-256 does not match inventory")

    return dict(manifest)


def load_manifest(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> dict[str, object]:
    """Load one strict schema-v4 manifest, optionally verifying its closure."""

    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _fail(f"cannot read {manifest_path}: {error}") from error
    try:
        document = json.loads(text, object_pairs_hook=_object_no_duplicate_keys)
    except CompileError:
        raise
    except json.JSONDecodeError as error:
        raise _fail(
            f"malformed JSON at line {error.lineno}, column {error.colno}"
        ) from error
    result = validate_manifest(
        document,
        root=manifest_path.parent if verify_files else None,
        verify_files=verify_files,
    )
    inventory = _require_mapping(result["artifact_inventory"], "artifact_inventory")
    if inventory["manifest_path"] != manifest_path.name:
        raise _fail("artifact_inventory.manifest_path does not name the loaded manifest")
    return result


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
