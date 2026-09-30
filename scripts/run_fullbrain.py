from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sentinel.brain.full_flywire import load_flywire_arrays, run_full_flywire


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    asset_root = root / "data/connectome/full"
    completeness = asset_root / "Completeness_783.csv"
    connectivity = asset_root / "Connectivity_783.parquet"
    arrays = load_flywire_arrays(completeness, connectivity)
    result = run_full_flywire(
        arrays,
        np.linspace(-2.0, 2.0, 40, dtype=np.float64),
        duration_ms=5.0,
        random_seed=1729,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = {
        **asdict(result),
        "asset_release": "FlyWire v783",
        "asset_sha256": {
            completeness.name: sha256(completeness),
            connectivity.name: sha256(connectivity),
        },
        "git_commit": commit,
        "python": platform.python_version(),
        "seed": 1729,
        "simulation_scope": "all neurons and all directed connections",
    }
    if result.total_spikes <= 0 or result.active_neurons <= 0:
        raise SystemExit("full-brain run was degenerate: no spikes")
    destination = root / "artifacts/fullbrain-run.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
