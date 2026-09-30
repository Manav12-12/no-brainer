from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SensoryEncoder:
    def __init__(self, sensory_neurons: int, max_rate_hz: float = 150.0) -> None:
        if sensory_neurons < 1 or max_rate_hz <= 0:
            raise ValueError("encoder dimensions and rate must be positive")
        self.sensory_neurons = sensory_neurons
        self.max_rate_hz = max_rate_hz

    def encode(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if features.shape != (40,):
            raise ValueError("encoder requires 40 features")
        if not np.isfinite(features).all():
            raise ValueError("encoder features must be finite")
        groups = np.array_split(features, self.sensory_neurons)
        pooled = np.asarray([float(np.mean(group)) for group in groups])
        rates = self.max_rate_hz / (1.0 + np.exp(-np.clip(pooled, -20, 20)))
        return np.asarray(np.clip(rates, 0.0, self.max_rate_hz), dtype=np.float64)
