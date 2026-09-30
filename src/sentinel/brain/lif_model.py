from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from brian2 import (
    Hz,
    Network,
    NeuronGroup,
    PoissonGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    ms,
    mV,
    prefs,
    seed,
    start_scope,
)
from numpy.typing import NDArray


@dataclass(frozen=True)
class LIFParameters:
    resting_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    synaptic_tau_ms: float = 5.0
    refractory_ms: float = 2.2
    synaptic_delay_ms: float = 1.8
    weight_per_synapse_mv: float = 0.275
    stimulation_weight_scale: float = 250.0
    timestep_ms: float = 0.1
    codegen_target: str = "numpy"


def simulate_lif(
    graph: nx.DiGraph,
    sensory_nodes: list[int],
    rates_hz: NDArray[np.float64],
    duration_ms: float,
    params: LIFParameters,
    random_seed: int,
) -> dict[int, int]:
    if len(sensory_nodes) != len(rates_hz) or duration_ms <= 0:
        raise ValueError("stimulation dimensions and duration must be valid")
    nodes = sorted(int(node) for node in graph.nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    if any(node not in node_index for node in sensory_nodes):
        raise ValueError("sensory node is absent from graph")
    start_scope()
    seed(random_seed)
    prefs.codegen.target = params.codegen_target
    defaultclock.dt = params.timestep_ms * ms
    equations = """
    dv/dt = (v_rest - v + g) / tau_m : volt (unless refractory)
    dg/dt = -g / tau_syn : volt (unless refractory)
    """
    neurons = NeuronGroup(
        len(nodes),
        equations,
        threshold="v > v_threshold",
        reset="v = v_reset; g = 0*mV",
        refractory=params.refractory_ms * ms,
        method="euler",
        namespace={
            "v_rest": params.resting_mv * mV,
            "v_threshold": params.threshold_mv * mV,
            "v_reset": params.reset_mv * mV,
            "tau_m": params.membrane_tau_ms * ms,
            "tau_syn": params.synaptic_tau_ms * ms,
        },
    )
    neurons.v = params.resting_mv * mV
    recurrent = Synapses(neurons, neurons, model="w : volt", on_pre="g += w")
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for source, target, data in graph.edges(data=True):
        sources.append(node_index[int(source)])
        targets.append(node_index[int(target)])
        sign = 1.0 if bool(data.get("excitatory", True)) else -1.0
        weights.append(
            sign * float(data.get("weight", 1.0)) * params.weight_per_synapse_mv
        )
    recurrent.connect(i=sources, j=targets)
    recurrent.w = np.asarray(weights) * mV
    recurrent.delay = params.synaptic_delay_ms * ms
    inputs = PoissonGroup(len(sensory_nodes), rates=rates_hz * Hz)
    stimulation = Synapses(
        inputs,
        neurons,
        on_pre="g += stimulation_weight",
        namespace={
            "stimulation_weight": params.weight_per_synapse_mv
            * params.stimulation_weight_scale
            * mV
        },
    )
    stimulation.connect(
        i=np.arange(len(sensory_nodes)),
        j=[node_index[node] for node in sensory_nodes],
    )
    monitor = SpikeMonitor(neurons)
    network = Network(neurons, recurrent, inputs, stimulation, monitor)
    network.run(duration_ms * ms)
    counts = np.asarray(monitor.count)
    return {node: int(counts[index]) for node, index in node_index.items()}
