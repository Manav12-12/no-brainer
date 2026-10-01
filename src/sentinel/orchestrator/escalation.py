from __future__ import annotations

import networkx as nx

from sentinel.brain.readout import BrainOutput
from sentinel.brain.runner import run_brain_from_pain


def escalate_to_brain(
    graph: nx.DiGraph, pain_signal: float, seed: int, duration_ms: float
) -> tuple[BrainOutput, dict[int, int]]:
    """Convert the reflex's pain report into neural activity, never an action."""
    return run_brain_from_pain(graph, pain_signal, seed, duration_ms)
