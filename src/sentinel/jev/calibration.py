from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def brier_score(y_true: NDArray[np.float64], probability: NDArray[np.float64]) -> float:
    return float(np.mean((probability - y_true) ** 2))


def expected_calibration_error(
    y_true: NDArray[np.float64], probability: NDArray[np.float64], bins: int = 10
) -> float:
    if bins < 1:
        raise ValueError("bins must be positive")
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_true)
    if total == 0:
        raise ValueError("calibration arrays cannot be empty")
    error = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if upper_inclusive
            else probability < edges[index + 1]
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(y_true[mask].mean())
            )
    return error
