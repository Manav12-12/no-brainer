from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from sentinel.brain.encoder import SensoryEncoder
from sentinel.brain.lif_model import LIFParameters, simulate_lif
from sentinel.connectome.annotations import population_ids


def offline_kc_mbon_update(
    weights: NDArray[np.float64],
    kc_activity: NDArray[np.float64],
    error: NDArray[np.float64],
    learning_rate: float,
) -> NDArray[np.float64]:
    if learning_rate < 0 or weights.shape != (len(kc_activity), len(error)):
        raise ValueError("plasticity dimensions or learning rate are invalid")
    return weights + learning_rate * np.outer(kc_activity, error)


@dataclass(frozen=True)
class PlasticityReport:
    examples: int
    updated_edges: int
    l1_weight_delta: float


def train_kc_mbon_weights(
    graph: nx.DiGraph,
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    duration_ms: float,
    random_seed: int,
    learning_rate: float,
) -> PlasticityReport:
    """Apply an offline reward-prediction update to graph weights in place."""
    if features.ndim != 2 or features.shape[1] != 40 or len(features) != len(labels):
        raise ValueError("plasticity training data must be N x 40 with N labels")
    kc_nodes = population_ids(graph, "KC")
    mbon_nodes = population_ids(graph, "MBON")
    sensory_nodes = population_ids(graph, "ORN")
    weights = np.zeros((len(kc_nodes), len(mbon_nodes)), dtype=np.float64)
    edge_mask = np.zeros_like(weights, dtype=np.bool_)
    for kc_index, kc in enumerate(kc_nodes):
        for mbon_index, mbon in enumerate(mbon_nodes):
            if graph.has_edge(kc, mbon):
                weights[kc_index, mbon_index] = float(graph.edges[kc, mbon]["weight"])
                edge_mask[kc_index, mbon_index] = True
    initial = weights.copy()
    encoder = SensoryEncoder(len(sensory_nodes))
    for index in range(len(features)):
        vector = np.asarray(features[index], dtype=np.float64)
        label = int(labels[index])
        counts = simulate_lif(
            graph,
            sensory_nodes,
            encoder.encode(vector),
            duration_ms,
            LIFParameters(),
            random_seed + index,
        )
        kc_activity = np.asarray([counts[node] for node in kc_nodes], dtype=np.float64)
        observed = np.clip(
            np.asarray([counts[node] for node in mbon_nodes], dtype=np.float64) / 10.0,
            0.0,
            1.0,
        )
        target = np.full(len(mbon_nodes), float(label), dtype=np.float64)
        reward_prediction_error = target - observed
        proposed = offline_kc_mbon_update(
            weights, kc_activity, reward_prediction_error, learning_rate
        )
        weights[:, :] = np.where(edge_mask, np.clip(proposed, 0.1, 100.0), 0.0)
        for kc_index, kc in enumerate(kc_nodes):
            for mbon_index, mbon in enumerate(mbon_nodes):
                if edge_mask[kc_index, mbon_index]:
                    graph.edges[kc, mbon]["weight"] = float(
                        weights[kc_index, mbon_index]
                    )
    delta = float(np.abs(weights - initial).sum())
    return PlasticityReport(len(features), int(edge_mask.sum()), delta)
