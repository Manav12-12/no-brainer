from __future__ import annotations

import math
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
DecisionSpec: TypeAlias = dict[str, dict[str, JsonValue]]


class QuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["choice", "score", "noul"]
    probabilities: dict[str, float]
    selected: str | float
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_distribution(self) -> QuestionResult:
        if not self.probabilities:
            raise ValueError("probability distribution cannot be empty")
        if any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in self.probabilities.values()
        ):
            raise ValueError("probabilities must be finite and between zero and one")
        if not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        return self


class DecisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answers: dict[str, QuestionResult]
    model_version: str
    latency_ms: float = Field(ge=0.0)
    usage_input_tokens: int | None = Field(default=None, ge=0)


class DecisionModel(Protocol):
    def decide(
        self, state: dict[str, JsonValue], spec: DecisionSpec
    ) -> DecisionResult: ...


class DecisionUnavailable(RuntimeError):
    """Raised when a decision backend cannot return a valid result."""


class VersionMismatch(DecisionUnavailable):
    """Raised when the API returns a model other than the pinned model."""


def validate_spec(spec: DecisionSpec) -> None:
    if not spec:
        raise ValueError("at least one question is required")
    for name, question in spec.items():
        if not name or not isinstance(question, dict):
            raise ValueError("question names and definitions must be mappings")
        question_type = question.get("type")
        if question_type not in {"choice", "score", "noul"}:
            raise ValueError("only documented typed questions are supported")
        if "instructions" not in question:
            raise ValueError("question instructions are required")
        criteria = question.get("criteria")
        if question_type == "choice" and (
            not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255
        ):
            raise ValueError("choice criteria must contain 1 through 255 options")
        if question_type == "score" and (
            not isinstance(criteria, list) or not 2 <= len(criteria) <= 10
        ):
            raise ValueError("score criteria must contain 2 through 10 levels")
