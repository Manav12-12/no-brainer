from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Estimate:
    mean: float
    ci95_low: float
    ci95_high: float
    samples: int


def mean_ci(values: NDArray[np.float64]) -> Estimate:
    if len(values) == 0:
        raise ValueError("cannot summarize an empty metric")
    mean = float(np.mean(values))
    if len(values) == 1:
        return Estimate(mean, mean, mean, 1)
    margin = 1.96 * float(np.std(values, ddof=1)) / sqrt(len(values))
    return Estimate(mean, mean - margin, mean + margin, len(values))


def detection_rate(labels: NDArray[np.int64], actions: NDArray[np.int64]) -> float:
    attacked = labels == 1
    if not attacked.any():
        return float("nan")
    return float(actions[attacked].mean())


def false_action_rate(labels: NDArray[np.int64], actions: NDArray[np.int64]) -> float:
    benign = labels == 0
    if not benign.any():
        return float("nan")
    return float(actions[benign].mean())
