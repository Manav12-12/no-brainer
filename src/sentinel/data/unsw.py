from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sentinel.data.features import FEATURE_NAMES

UNSW_ARCHIVE_MEMBER = "UNSW-NB15-V3.csv"
UNSW_EXCLUDED_COLUMNS = {
    "srcip",
    "dstip",
    "stcpb",
    "dtcpb",
    "Stime",
    "Ltime",
}


def sample_unsw_archive(
    archive: Path, *, per_family: int = 500, chunk_size: int = 100_000
) -> tuple[pd.DataFrame, dict[str, int]]:
    if per_family < 1 or chunk_size < 1:
        raise ValueError("sampling sizes must be positive")
    with ZipFile(archive) as bundle, bundle.open(UNSW_ARCHIVE_MEMBER) as handle:
        header = pd.read_csv(handle, nrows=0)
    numeric_columns = [
        name
        for name in header.columns
        if name != "label" and name not in UNSW_EXCLUDED_COLUMNS
    ]
    if len(numeric_columns) != len(FEATURE_NAMES):
        raise ValueError(
            f"expected 40 selected UNSW features, got {len(numeric_columns)}"
        )

    samples: dict[str, list[pd.DataFrame]] = {}
    observed: dict[str, int] = {}
    offset = 0
    usecols = [*numeric_columns, "label"]
    with ZipFile(archive) as bundle, bundle.open(UNSW_ARCHIVE_MEMBER) as handle:
        reader = pd.read_csv(handle, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            chunk.insert(0, "source_row", np.arange(offset, offset + len(chunk)))
            offset += len(chunk)
            chunk["label"] = chunk["label"].astype(str).str.strip().str.lower()
            for family, group in chunk.groupby("label", sort=True):
                name = str(family)
                observed[name] = observed.get(name, 0) + len(group)
                kept = sum(len(part) for part in samples.get(name, []))
                remaining = per_family - kept
                if remaining > 0:
                    samples.setdefault(name, []).append(group.head(remaining).copy())
    if not samples:
        raise ValueError("UNSW archive yielded no records")
    sampled = pd.concat(
        [part for family in sorted(samples) for part in samples[family]],
        ignore_index=True,
    )
    values = sampled.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("UNSW selected features contain non-finite values")
    output = pd.DataFrame(values, columns=FEATURE_NAMES)
    output["host_role"] = np.resize(
        np.asarray(["workstation", "server", "gateway"]), len(output)
    )
    output["segment"] = np.resize(np.asarray(["user", "server", "dmz"]), len(output))
    output["recent_event_count"] = sampled["source_row"].to_numpy() % 10_001
    output["label"] = (sampled["label"] != "benign").astype(np.int64).to_numpy()
    output["attack_family"] = sampled["label"].to_numpy()
    output["timestamp"] = sampled["source_row"].to_numpy(dtype=np.int64)
    output.attrs["synthetic_mode"] = False
    output.attrs["source_feature_columns"] = numeric_columns
    return output.sort_values("timestamp").reset_index(drop=True), observed
