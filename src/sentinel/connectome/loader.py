from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"source", "target", "weight", "excitatory"}
PROXY_EDGE_CONNECTIVITY = 31.0


def load_connectome(path: Path) -> nx.DiGraph:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("connectome must be CSV or parquet")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing connectome columns: {sorted(missing)}")
    graph = nx.DiGraph()
    for row in frame.itertuples(index=False):
        weight = float(row.weight)
        if weight <= 0:
            raise ValueError("connectome weights must be positive")
        graph.add_edge(
            int(row.source),
            int(row.target),
            weight=weight,
            excitatory=bool(row.excitatory),
        )
    return graph


def synthetic_connectome(seed: int = 0) -> nx.DiGraph:
    graph = nx.DiGraph(
        synthetic_mode=True,
        seed=seed,
        proxy_edge_connectivity=PROXY_EDGE_CONNECTIVITY,
    )
    populations = {
        "ORN": range(0, 8),
        "PN": range(8, 16),
        "KC": range(16, 40),
        "MBON": range(40, 44),
        "APL": range(44, 45),
        "descending": range(45, 48),
    }
    for population, nodes in populations.items():
        for node in nodes:
            graph.add_node(node, population=population)
    # A proxy edge represents an aggregate bundle of biological contacts. The
    # value 31 is the measured 99th percentile edge connectivity in the local
    # FlyWire v783 asset. Complete feed-forward fan-in compensates explicitly
    # for reducing a graph with median excitatory in-degree 37 to 48 neurons.
    feed_forward = (("ORN", "PN"), ("PN", "KC"), ("KC", "MBON"))
    for source_name, target_name in feed_forward:
        for source in populations[source_name]:
            for target in populations[target_name]:
                graph.add_edge(
                    source,
                    target,
                    weight=PROXY_EDGE_CONNECTIVITY,
                    excitatory=True,
                )
    for kc in populations["KC"]:
        graph.add_edge(44, kc, weight=1.0, excitatory=False)
    for mbon in populations["MBON"]:
        for descending in populations["descending"]:
            graph.add_edge(
                mbon,
                descending,
                weight=PROXY_EDGE_CONNECTIVITY,
                excitatory=True,
            )
        graph.add_edge(mbon, 44, weight=1.0, excitatory=True)
    return graph


def random_sparse_mushroom_body(seed: int = 0) -> nx.DiGraph:
    template = synthetic_connectome(seed)
    graph = nx.DiGraph(synthetic_mode=True, random_sparse_control=True, seed=seed)
    graph.add_nodes_from(template.nodes(data=True))
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes)
    target_edges = template.number_of_edges()
    while graph.number_of_edges() < target_edges:
        source = int(rng.choice(nodes))
        target = int(rng.choice(nodes))
        if source != target:
            graph.add_edge(
                source,
                target,
                weight=float(rng.integers(1, 5)),
                excitatory=bool(rng.integers(0, 2)),
            )
    return graph
