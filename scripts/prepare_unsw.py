from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sentinel.data.unsw import sample_unsw_archive


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/public/unsw_nb15_v3"
    archive = directory / "UNSW-NB15-V3.zip"
    frame, observed = sample_unsw_archive(archive)
    processed = directory / "processed.parquet"
    frame.to_parquet(processed, index=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    metadata = {
        "archive": archive.name,
        "archive_sha256": digest,
        "dataset": "UNSW-NB15 V3",
        "doi": "10.5281/zenodo.10141617",
        "license": "CC-BY-4.0",
        "observed_rows": sum(observed.values()),
        "observed_family_counts": observed,
        "processed_rows": len(frame),
        "sampled_family_counts": {
            str(name): int(count)
            for name, count in frame["attack_family"]
            .value_counts()
            .sort_index()
            .items()
        },
        "selected_source_features": frame.attrs["source_feature_columns"],
        "selection": "first 500 source-ordered records per family; all if fewer",
        "synthetic_mode": False,
    }
    destination = root / "artifacts/unsw-preprocessing.json"
    destination.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
