from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnostic:
    """Stable machine-readable compiler diagnostic carried by BakeNN errors."""

    code: str
    stage: str
    reason: str
    location: str | None = None
    suggestions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.code.startswith("BAKENN_")
            or self.code != self.code.upper()
            or not self.code.replace("_", "").isalnum()
        ):
            raise ValueError("diagnostic code must be an uppercase BAKENN_* identifier")
        if not self.stage or not self.reason:
            raise ValueError("diagnostic stage and reason must be non-empty")
        if self.location is not None and not self.location:
            raise ValueError("diagnostic location must be None or non-empty")
        if not isinstance(self.suggestions, tuple) or any(
            not isinstance(item, str) or not item for item in self.suggestions
        ):
            raise ValueError("diagnostic suggestions must be non-empty strings")

    def format(self) -> str:
        prefix = f"{self.location}: " if self.location else ""
        return prefix + self.reason


class BakeNNError(Exception):
    """Base class for deterministic user-facing compiler failures.

    Existing callers may continue raising or matching a plain message. New
    frontends and validators can attach a :class:`Diagnostic` without changing
    the human-readable ``str(error)`` contract.
    """

    default_code = "BAKENN_ERROR"
    default_stage = "compiler"

    def __init__(
        self,
        message: str | Diagnostic,
        *,
        code: str | None = None,
        stage: str | None = None,
        location: str | None = None,
        suggestions: tuple[str, ...] = (),
    ) -> None:
        if isinstance(message, Diagnostic):
            if any(value is not None for value in (code, stage, location)) or suggestions:
                raise TypeError("Diagnostic cannot be combined with diagnostic keyword fields")
            diagnostic = message
        elif isinstance(message, str) and message:
            diagnostic = Diagnostic(
                code=code or self.default_code,
                stage=stage or self.default_stage,
                location=location,
                reason=message,
                suggestions=tuple(suggestions),
            )
        else:
            raise TypeError("BakeNN errors require a non-empty message or Diagnostic")
        self.diagnostic = diagnostic
        self.code = diagnostic.code
        self.stage = diagnostic.stage
        self.location = diagnostic.location
        self.suggestions = diagnostic.suggestions
        super().__init__(diagnostic.format())


class GraphValidationError(BakeNNError):
    """The quantized graph violates the BakeNN IR contract."""

    default_code = "BAKENN_GRAPH_INVALID"
    default_stage = "verify"


class CompileError(BakeNNError):
    """The graph is valid but cannot be lowered safely for a backend."""

    default_code = "BAKENN_COMPILE_FAILED"
    default_stage = "compile"


__all__ = ["BakeNNError", "CompileError", "Diagnostic", "GraphValidationError"]
