from __future__ import annotations

import re

import numpy as np

from bakenn.ir import PerAxisQParams, PerTensorQParams


def identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not result or result[0].isdigit():
        result = f"model_{result}"
    return result.lower()


def c_float(value: float) -> str:
    # A hexadecimal floating literal is exact and locale independent.  The
    # explicit suffix keeps public macros float-typed on embedded C targets.
    return float(value).hex() + "f"


def guard(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", value).upper() + "_H"


def format_values(array: np.ndarray, *, per_line: int = 12) -> str:
    flat = array.reshape(-1)
    rows = []
    for start in range(0, flat.size, per_line):
        rows.append("    " + ", ".join(str(int(value)) for value in flat[start : start + per_line]))
    return ",\n".join(rows)


def qparams_dict(qparams: PerTensorQParams | PerAxisQParams) -> dict[str, object]:
    if isinstance(qparams, PerTensorQParams):
        return {"kind": "per_tensor", "scale": qparams.scale, "zero_point": qparams.zero_point}
    return {
        "kind": "per_axis",
        "scales": list(qparams.scales),
        "zero_points": list(qparams.zero_points),
        "axis": qparams.axis,
    }


__all__ = ["c_float", "format_values", "guard", "identifier", "qparams_dict"]
