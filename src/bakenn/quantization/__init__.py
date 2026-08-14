from .fixedpoint import (
    ARITHMETIC_PROFILE,
    multiply_by_quantized_multiplier,
    quantize_multiplier,
    round_half_away_from_zero,
)


def __getattr__(name: str):
    """Load the legacy MLP PTQ frontend without creating IR import cycles."""

    if name == "quantize_float_graph":
        from .ptq_graph import quantize_float_graph

        return quantize_float_graph
    if name in {"FloatLinear", "FloatMLP", "quantize_ptq"}:
        from .ptq import FloatLinear, FloatMLP, quantize_ptq

        exports = {
            "FloatLinear": FloatLinear,
            "FloatMLP": FloatMLP,
            "quantize_ptq": quantize_ptq,
        }
        return exports[name]
    raise AttributeError(name)

__all__ = [
    "ARITHMETIC_PROFILE",
    "FloatLinear",
    "FloatMLP",
    "multiply_by_quantized_multiplier",
    "quantize_multiplier",
    "quantize_float_graph",
    "quantize_ptq",
    "round_half_away_from_zero",
]
