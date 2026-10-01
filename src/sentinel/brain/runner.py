from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from sentinel.brain.encoder import SensoryEncoder
from sentinel.brain.lif_model import LIFParameters, simulate_lif
from sentinel.brain.readout import BrainOutput, brain_readout
from sentinel.connectome.annotations import population_ids, require_populations


def run_brain(
    graph: nx.DiGraph,
    features: NDArray[np.float64],
    seed: int,
    duration_ms: float = 100.0,
    params: LIFParameters | None = None,
) -> BrainOutput:
    require_populations(graph)
    sensory = population_ids(graph, "ORN")
    rates = SensoryEncoder(len(sensory)).encode(features)
    counts = simulate_lif(
        graph,
        sensory,
        rates,
        duration_ms,
        params or LIFParameters(),
        seed,
    )
    return brain_readout(graph, counts, rates)
