from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from sentinel.connectome.annotations import population_ids


@dataclass(frozen=True)
class BrainOutput:
    novelty_score: float
    sensory_stimulation: float
    drive_stimulation: float
    threat_class: str
    action: str
    population_activity: dict[str, float]


def read_population_activity(
    graph: nx.DiGraph, spike_counts: dict[int, int]
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in ("KC", "MBON", "descending"):
        nodes = population_ids(graph, name)
        output[name] = float(np.mean([spike_counts.get(node, 0) for node in nodes]))
    return output


def brain_readout(
    graph: nx.DiGraph,
    spike_counts: dict[int, int],
    sensory_rates_hz: NDArray[np.float64] | None = None,
    max_sensory_rate_hz: float = 150.0,
) -> BrainOutput:
    activity = read_population_activity(graph, spike_counts)
    kc = activity["KC"]
    descending = activity["descending"]
    novelty = float(np.clip(kc / 10.0, 0.0, 1.0))
    sensory = 0.0
    if sensory_rates_hz is not None:
        normalized_rate = float(np.mean(sensory_rates_hz) / max_sensory_rate_hz)
        sensory = float(np.clip((normalized_rate - 0.5) / 0.15, 0.0, 1.0))
    # Pressure combines encoded telemetry with propagated KC activity. Either
    # component alone was non-discriminative on validation sequences.
    drive_stimulation = float(np.clip((novelty + sensory) / 2.0, 0.0, 1.0))
    threat_class = "novel" if novelty >= 0.5 else "benign_or_known"
    action = "isolate_host" if descending > 0 else "no_op"
    return BrainOutput(
        novelty, sensory, drive_stimulation, threat_class, action, activity
    )
