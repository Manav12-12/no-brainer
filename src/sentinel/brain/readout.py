from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from sentinel.connectome.annotations import population_ids


@dataclass(frozen=True)
class BrainOutput:
    novelty_score: float
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


def brain_readout(graph: nx.DiGraph, spike_counts: dict[int, int]) -> BrainOutput:
    activity = read_population_activity(graph, spike_counts)
    kc = activity["KC"]
    descending = activity["descending"]
    novelty = float(np.clip(kc / 10.0, 0.0, 1.0))
    threat_class = "novel" if novelty >= 0.5 else "benign_or_known"
    action = "isolate_host" if descending > 0 else "no_op"
    return BrainOutput(novelty, threat_class, action, activity)
