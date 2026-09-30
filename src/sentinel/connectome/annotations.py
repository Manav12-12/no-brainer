from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

POPULATIONS = ("ORN", "PN", "KC", "MBON", "APL", "descending")


def population_ids(graph: nx.DiGraph, population: str) -> list[int]:
    if population not in POPULATIONS:
        raise ValueError(f"unknown population: {population}")
    return sorted(
        int(node)
        for node, data in graph.nodes(data=True)
        if data.get("population") == population
    )


def require_populations(
    graph: nx.DiGraph, populations: Iterable[str] = POPULATIONS
) -> None:
    missing = [name for name in populations if not population_ids(graph, name)]
    if missing:
        raise ValueError(f"connectome lacks annotated populations: {missing}")
