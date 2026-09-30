from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentinel.data.features import CONTEXT_COLUMNS, FEATURE_NAMES, LABEL_COLUMNS


def load_processed(path: Path, *, synthetic_expected: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(FEATURE_NAMES + CONTEXT_COLUMNS + LABEL_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing processed columns: {sorted(missing)}")
    actual = bool(frame.attrs.get("synthetic_mode", False))
    if actual != synthetic_expected:
        raise ValueError("synthetic and public-dataset modes cannot be mixed")
    return frame
