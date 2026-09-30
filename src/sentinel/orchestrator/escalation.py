from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from sentinel.brain.readout import BrainOutput
from sentinel.brain.runner import run_brain
from sentinel.cyberbody.actions import DefensiveAction


def escalate_to_brain(
    graph: nx.DiGraph, features: NDArray[np.float64], seed: int, duration_ms: float
) -> tuple[BrainOutput, DefensiveAction]:
    output = run_brain(graph, features, seed, duration_ms)
    try:
        action = DefensiveAction(output.action)
    except ValueError:
        action = DefensiveAction.NO_OP
    return output, action
