from __future__ import annotations

import pytest

from bakenn.errors import CompileError, Diagnostic, GraphValidationError


def test_plain_errors_keep_human_message_and_gain_stable_metadata() -> None:
    error = CompileError("unsupported operator")
    assert str(error) == "unsupported operator"
    assert error.code == "BAKENN_COMPILE_FAILED"
    assert error.stage == "compile"
    assert error.location is None

    graph_error = GraphValidationError("bad graph")
    assert graph_error.code == "BAKENN_GRAPH_INVALID"
    assert graph_error.stage == "verify"


def test_structured_diagnostic_exposes_location_and_suggestions() -> None:
    diagnostic = Diagnostic(
        code="BAKENN_TFLITE_OPERATOR_UNSUPPORTED",
        stage="tflite_import",
        location="subgraph[0]/operator[3]",
        reason="CUSTOM operators are unsupported",
        suggestions=("Convert the operator to a supported builtin",),
    )
    error = CompileError(diagnostic)
    assert str(error) == "subgraph[0]/operator[3]: CUSTOM operators are unsupported"
    assert error.diagnostic == diagnostic
    assert error.suggestions == diagnostic.suggestions


@pytest.mark.parametrize(
    "code", ["bad", "BAKENN-bad", "OTHER_ERROR", "BAKENN_lowercase"]
)
def test_diagnostic_rejects_unstable_code_shapes(code: str) -> None:
    with pytest.raises(ValueError, match="BAKENN"):
        Diagnostic(code, "compile", "reason")
