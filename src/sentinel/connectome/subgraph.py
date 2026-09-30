from __future__ import annotations

import networkx as nx

from sentinel.connectome.annotations import population_ids


def mushroom_body_subgraph(graph: nx.DiGraph) -> nx.DiGraph:
    names = ("ORN", "PN", "KC", "MBON", "APL", "descending")
    nodes = [node for name in names for node in population_ids(graph, name)]
    if not nodes:
        raise ValueError("no mushroom-body populations found")
    return graph.subgraph(nodes).copy()
