from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def offline_kc_mbon_update(
    weights: NDArray[np.float64],
    kc_activity: NDArray[np.float64],
    error: NDArray[np.float64],
    learning_rate: float,
) -> NDArray[np.float64]:
    if learning_rate < 0 or weights.shape != (len(kc_activity), len(error)):
        raise ValueError("plasticity dimensions or learning rate are invalid")
    return weights + learning_rate * np.outer(kc_activity, error)
