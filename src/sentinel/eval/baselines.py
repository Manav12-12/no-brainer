from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from torch import nn


class IsolationBaseline:
    def __init__(self, seed: int) -> None:
        self.model = IsolationForest(contamination=0.2, random_state=seed)

    def fit(self, benign: NDArray[np.float64]) -> IsolationBaseline:
        self.model.fit(benign)
        return self

    def detect(self, features: NDArray[np.float64]) -> bool:
        return bool(self.model.predict(features.reshape(1, -1))[0] == -1)


class _Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(40, 12), nn.ReLU(), nn.Linear(12, 40))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(value))


class AutoencoderBaseline:
    def __init__(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.model = _Autoencoder()
        self.threshold = float("inf")

    def fit(self, benign: NDArray[np.float64], epochs: int = 10) -> AutoencoderBaseline:
        values = torch.as_tensor(benign, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean((self.model(values) - values) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        with torch.no_grad():
            errors = torch.mean((self.model(values) - values) ** 2, dim=1)
        self.threshold = float(torch.quantile(errors, 0.95))
        return self

    def detect(self, features: NDArray[np.float64]) -> bool:
        value = torch.as_tensor(features.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            error = float(torch.mean((self.model(value) - value) ** 2))
        return error > self.threshold
