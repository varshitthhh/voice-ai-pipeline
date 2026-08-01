"""Generic reject-and-report wrapper around Pydantic validation.

Stage-boundary payloads (stages.py) funnel through this: bad input never
raises past this module's boundary, it comes back as a structured report
the caller can log and skip past instead of crashing the pipeline.
"""

from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class FieldError(BaseModel):
    field: str
    message: str
    input_value: Any = None


class ValidationReport(BaseModel):
    valid: bool
    model_name: str
    errors: list[FieldError] = []

    @property
    def summary(self) -> str:
        if self.valid:
            return f"{self.model_name}: valid"
        reasons = "; ".join(f"{e.field}: {e.message}" for e in self.errors)
        return f"{self.model_name}: REJECTED ({reasons})"


def validate_payload(model_cls: Type[ModelT], data: Any) -> tuple[Optional[ModelT], ValidationReport]:
    """Validate `data` against `model_cls`. Never raises.

    Returns (instance, report): instance is None and report.valid is False
    on any validation failure, including malformed `data` that isn't even a
    dict (e.g. None, a string, a list) — those don't raise ValidationError
    from pydantic, so they're caught separately rather than letting a
    stray TypeError crash the caller.
    """
    try:
        instance = model_cls.model_validate(data)
        return instance, ValidationReport(valid=True, model_name=model_cls.__name__)
    except ValidationError as exc:
        errors = [
            FieldError(
                field=".".join(str(p) for p in e["loc"]) or "<root>",
                message=e["msg"],
                input_value=e.get("input"),
            )
            for e in exc.errors()
        ]
        return None, ValidationReport(valid=False, model_name=model_cls.__name__, errors=errors)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: malformed input must never crash the caller
        errors = [FieldError(field="<root>", message=f"{type(exc).__name__}: {exc}", input_value=data)]
        return None, ValidationReport(valid=False, model_name=model_cls.__name__, errors=errors)
