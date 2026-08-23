from .fixedpoint import (
    ARITHMETIC_PROFILE,
    multiply_by_quantized_multiplier,
    quantize_multiplier,
    round_half_away_from_zero,
)


def __getattr__(name: str):
    """Load the legacy MLP PTQ frontend without creating IR import cycles."""

    if name in {"quantize_float_graph", "quantize_float_graph_with_report"}:
        from .ptq_graph import quantize_float_graph, quantize_float_graph_with_report

        return {
            "quantize_float_graph": quantize_float_graph,
            "quantize_float_graph_with_report": quantize_float_graph_with_report,
        }[name]
    if name in {"CalibrationEdgeReport", "CalibrationReport", "PTQResult"}:
        from .report import CalibrationEdgeReport, CalibrationReport, PTQResult

        return {
            "CalibrationEdgeReport": CalibrationEdgeReport,
            "CalibrationReport": CalibrationReport,
            "PTQResult": PTQResult,
        }[name]
    if name in {"LayerErrorReport", "PTQVerificationReport", "verify_ptq_accuracy"}:
        from .verification import (
            LayerErrorReport,
            PTQVerificationReport,
            verify_ptq_accuracy,
        )

        return {
            "LayerErrorReport": LayerErrorReport,
            "PTQVerificationReport": PTQVerificationReport,
            "verify_ptq_accuracy": verify_ptq_accuracy,
        }[name]
    if name in {"LinearWeightGranularity", "PTQOptions"}:
        from .ptq_graph import LinearWeightGranularity, PTQOptions

        exports = {
            "LinearWeightGranularity": LinearWeightGranularity,
            "PTQOptions": PTQOptions,
        }
        return exports[name]
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
    "CalibrationEdgeReport",
    "CalibrationReport",
    "FloatLinear",
    "FloatMLP",
    "LayerErrorReport",
    "LinearWeightGranularity",
    "PTQOptions",
    "PTQResult",
    "PTQVerificationReport",
    "multiply_by_quantized_multiplier",
    "quantize_multiplier",
    "quantize_float_graph",
    "quantize_float_graph_with_report",
    "quantize_ptq",
    "round_half_away_from_zero",
    "verify_ptq_accuracy",
]
