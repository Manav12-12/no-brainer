from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from sentinel.data.features import FEATURE_NAMES, FeaturePipeline, serialize_features
from sentinel.data.splits import family_holdout_split, time_split
from sentinel.data.synthetic import generate_synthetic_events
from sentinel.data.unsw import sample_unsw_archive


@pytest.mark.unit
def test_generator_is_seeded_and_has_fixed_schema() -> None:
    first = generate_synthetic_events(100, 7)
    second = generate_synthetic_events(100, 7)
    assert first.equals(second)
    assert len(FEATURE_NAMES) == 40


@pytest.mark.unit
def test_scaler_fits_train_only_and_labels_are_not_serialized() -> None:
    frame = generate_synthetic_events(100, 8)
    split = time_split(frame)
    pipeline = FeaturePipeline.create().fit(split.train)
    transformed = pipeline.transform(split.test)
    state = serialize_features(transformed.iloc[0])
    assert set(state) == {"serialization_version", "features", "context"}
    assert "label" not in str(state)
    assert np.asarray(state["features"]).shape == (40,)


@pytest.mark.unit
def test_pipeline_rejects_transform_before_fit() -> None:
    with pytest.raises(RuntimeError):
        FeaturePipeline.create().transform(generate_synthetic_events(2, 1))


@pytest.mark.unit
def test_family_holdout_has_no_leakage() -> None:
    split = family_holdout_split(generate_synthetic_events(1000, 9), "exfiltration")
    assert "exfiltration" not in set(split.train["attack_family"])
    assert set(split.test["attack_family"]) == {"exfiltration"}


@pytest.mark.unit
def test_invalid_splits_and_features_are_rejected() -> None:
    frame = generate_synthetic_events(20, 10)
    with pytest.raises(ValueError, match="sorted"):
        time_split(frame.sort_values("label"))
    with pytest.raises(ValueError, match="absent"):
        family_holdout_split(frame, "missing")
    frame.loc[0, FEATURE_NAMES[0]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        serialize_features(frame.iloc[0])


@pytest.mark.unit
def test_unsw_sampler_selects_real_schema(tmp_path: Path) -> None:
    columns = [
        "srcip",
        "dstip",
        "stcpb",
        "dtcpb",
        "Stime",
        "Ltime",
        *[f"value_{index}" for index in range(40)],
        "label",
    ]
    csv = tmp_path / "UNSW-NB15-V3.csv"
    pd.DataFrame(
        [
            [0.0] * 46 + ["benign"],
            [1.0] * 46 + ["worms"],
        ],
        columns=columns,
    ).to_csv(csv, index=False)
    archive = tmp_path / "dataset.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.write(csv, "UNSW-NB15-V3.csv")
    frame, observed = sample_unsw_archive(archive, per_family=2, chunk_size=1)
    assert tuple(frame.columns[:40]) == FEATURE_NAMES
    assert observed == {"benign": 1, "worms": 1}
    assert frame.attrs["synthetic_mode"] is False
