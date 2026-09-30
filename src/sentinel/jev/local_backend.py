from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from sentinel.jev.interface import (
    DecisionResult,
    DecisionSpec,
    JsonValue,
    QuestionResult,
    validate_spec,
)
from sentinel.jev.payload_guard import guard_payload


class LocalBackend:
    """Calibratable local binary baseline. It is not a Jev replacement."""

    def __init__(self) -> None:
        estimator = HistGradientBoostingClassifier(random_state=0)
        self.classifier = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
        self.fitted = False

    def fit(
        self, features: NDArray[np.number], labels: NDArray[np.number]
    ) -> LocalBackend:
        self.classifier.fit(features, labels)
        self.fitted = True
        return self

    def decide(self, state: dict[str, JsonValue], spec: DecisionSpec) -> DecisionResult:
        guard_payload(state)
        validate_spec(spec)
        if not self.fitted:
            raise RuntimeError("local backend must be fitted")
        started = time.perf_counter()
        features = np.asarray(state["features"], dtype=float).reshape(1, -1)
        probability = float(self.classifier.predict_proba(features)[0, 1])
        answers: dict[str, QuestionResult] = {}
        for name, question in spec.items():
            question_type = question["type"]
            if question_type == "noul":
                answers[name] = QuestionResult(
                    type="noul",
                    selected=probability,
                    probabilities={"false": 1 - probability, "true": probability},
                )
            elif question_type == "choice":
                criteria = question["criteria"]
                if not isinstance(criteria, dict):
                    raise ValueError("choice criteria must be a mapping")
                options = list(criteria)
                threat_option = (
                    "isolate_host" if "isolate_host" in options else options[-1]
                )
                safe_option = "no_op" if "no_op" in options else options[0]
                probabilities = dict.fromkeys(options, 0.0)
                probabilities[safe_option] += 1 - probability
                probabilities[threat_option] += probability
                selected = max(probabilities, key=probabilities.__getitem__)
                answers[name] = QuestionResult(
                    type="choice",
                    selected=selected,
                    probabilities=probabilities,
                    confidence=abs(2 * probability - 1),
                )
            else:
                raise ValueError("local baseline does not implement score questions")
        return DecisionResult(
            answers=answers,
            model_version="local-logistic-v1",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
