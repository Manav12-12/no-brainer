from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

FEATURE_VERSION: Final = "sentinel-features-v1"
FEATURE_NAMES: Final = tuple(f"feature_{index:02d}" for index in range(40))
CONTEXT_COLUMNS: Final = ("host_role", "segment", "recent_event_count")
LABEL_COLUMNS: Final = ("label", "attack_family", "timestamp")


@dataclass
class FeaturePipeline:
    scaler: StandardScaler
    fitted: bool = False

    @classmethod
    def create(cls) -> FeaturePipeline:
        return cls(scaler=StandardScaler())

    def fit(self, train: pd.DataFrame) -> FeaturePipeline:
        self.scaler.fit(train.loc[:, FEATURE_NAMES])
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("feature pipeline must be fitted on training data")
        output = frame.copy()
        values = self.scaler.transform(frame.loc[:, FEATURE_NAMES])
        output.loc[:, FEATURE_NAMES] = values
        return output


def serialize_features(row: pd.Series) -> dict[str, object]:
    values = np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float64)
    return serialize_vector(
        values,
        host_role=str(row["host_role"]),
        segment=str(row["segment"]),
        recent_event_count=int(row["recent_event_count"]),
    )


def serialize_vector(
    features: NDArray[np.float64],
    *,
    host_role: str,
    segment: str,
    recent_event_count: int,
) -> dict[str, object]:
    values = [float(value) for value in features]
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    return {
        "serialization_version": FEATURE_VERSION,
        "features": values,
        "context": {
            "host_role": host_role,
            "segment": segment,
            "recent_event_count": recent_event_count,
        },
    }
