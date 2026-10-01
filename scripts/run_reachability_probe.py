from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.brain.lif_model import LIFParameters, simulate_lif
from sentinel.connectome.annotations import population_ids
from sentinel.connectome.loader import PROXY_EDGE_CONNECTIVITY, synthetic_connectome


def summary(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values)
    return {
        "active_runs": int(np.sum(array > 0)),
        "mean_spikes": float(array.mean()),
        "p50_spikes": float(np.percentile(array, 50)),
        "max_spikes": int(array.max()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    connectivity = pd.read_parquet(
        root / "data/connectome/full/Connectivity_783.parquet",
        columns=["Postsynaptic_Index", "Connectivity", "Excitatory"],
    )
    excitatory = connectivity[connectivity["Excitatory"] > 0]
    indegree = excitatory.groupby("Postsynaptic_Index").size()
    graph = synthetic_connectome(1729)
    populations = ("ORN", "PN", "KC", "MBON", "descending")
    sensory = population_ids(graph, "ORN")
    durations: dict[str, dict[str, dict[str, float | int]]] = {}
    for duration in (15.0, 20.0, 50.0, 100.0):
        runs: dict[str, list[int]] = {name: [] for name in populations}
        for random_seed in range(1000, 1100):
            counts = simulate_lif(
                graph,
                sensory,
                np.full(len(sensory), 150.0),
                duration,
                LIFParameters(),
                random_seed,
            )
            for name in populations:
                runs[name].append(
                    sum(counts[node] for node in population_ids(graph, name))
                )
        durations[f"{duration:g}ms"] = {
            name: summary(values) for name, values in runs.items()
        }
    result = {
        "runs_per_duration": 100,
        "input_rate_hz": 150.0,
        "proxy_edge_connectivity": PROXY_EDGE_CONNECTIVITY,
        "flywire_v783_measurements": {
            "excitatory_edge_connectivity_p99": float(
                np.percentile(excitatory["Connectivity"], 99)
            ),
            "excitatory_indegree_median": float(np.percentile(indegree, 50)),
        },
        "durations": durations,
    }
    destination = root / "artifacts/brain-reachability.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
