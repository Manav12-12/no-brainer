from __future__ import annotations

import networkx as nx
import numpy as np


def degree_preserving_shuffle(
    graph: nx.DiGraph, seed: int, swaps: int | None = None
) -> nx.DiGraph:
    shuffled = graph.copy()
    target_swaps = swaps if swaps is not None else max(1, graph.number_of_edges() * 2)
    rng = np.random.default_rng(seed)
    completed = 0
    attempts = 0
    while completed < target_swaps and attempts < target_swaps * 100:
        attempts += 1
        edges = list(shuffled.edges())
        if len(edges) < 2:
            break
        selected = rng.choice(len(edges), size=2, replace=False)
        first, second = edges[int(selected[0])], edges[int(selected[1])]
        a, b = first
        c, d = second
        if len({a, b, c, d}) < 4 or shuffled.has_edge(a, d) or shuffled.has_edge(c, b):
            continue
        first_data = shuffled.edges[a, b].copy()
        second_data = shuffled.edges[c, d].copy()
        shuffled.remove_edge(a, b)
        shuffled.remove_edge(c, d)
        shuffled.add_edge(a, d, **first_data)
        shuffled.add_edge(c, b, **second_data)
        completed += 1
    shuffled.graph["shuffle_swaps"] = completed
    return shuffled
