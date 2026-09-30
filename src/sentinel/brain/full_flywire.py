from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
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

from sentinel.brain.encoder import SensoryEncoder
from sentinel.brain.lif_model import LIFParameters


@dataclass(frozen=True)
class FlyWireArrays:
    neuron_ids: NDArray[np.int64]
    presynaptic: NDArray[np.int64]
    postsynaptic: NDArray[np.int64]
    signed_connectivity: NDArray[np.float64]


@dataclass(frozen=True)
class FullBrainResult:
    neurons: int
    connections: int
    duration_ms: float
    total_spikes: int
    active_neurons: int
    max_spikes_per_neuron: int
    elapsed_seconds: float
    active_flywire_ids: list[int]


def load_flywire_arrays(
    completeness_path: Path, connectivity_path: Path
) -> FlyWireArrays:
    completeness = pd.read_csv(completeness_path, usecols=[0])
    connectivity = pd.read_parquet(
        connectivity_path,
        columns=[
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Excitatory x Connectivity",
        ],
    )
    neuron_ids = completeness.iloc[:, 0].to_numpy(dtype=np.int64, copy=True)
    presynaptic = connectivity["Presynaptic_Index"].to_numpy(dtype=np.int64, copy=True)
    postsynaptic = connectivity["Postsynaptic_Index"].to_numpy(
        dtype=np.int64, copy=True
    )
    signed = connectivity["Excitatory x Connectivity"].to_numpy(
        dtype=np.float64, copy=True
    )
    if len(neuron_ids) == 0 or len(presynaptic) == 0:
        raise ValueError("FlyWire assets must not be empty")
    if presynaptic.min() < 0 or postsynaptic.min() < 0:
        raise ValueError("FlyWire indices must be non-negative")
    if presynaptic.max() >= len(neuron_ids) or postsynaptic.max() >= len(neuron_ids):
        raise ValueError("FlyWire connection index exceeds the neuron table")
    return FlyWireArrays(neuron_ids, presynaptic, postsynaptic, signed)


def run_full_flywire(
    arrays: FlyWireArrays,
    features: NDArray[np.float64],
    *,
    duration_ms: float,
    random_seed: int,
    sensory_neurons: int = 40,
    params: LIFParameters | None = None,
) -> FullBrainResult:
    if duration_ms <= 0 or not 1 <= sensory_neurons <= len(arrays.neuron_ids):
        raise ValueError("full-brain stimulation settings are invalid")
    selected = params or LIFParameters()
    rates = SensoryEncoder(sensory_neurons).encode(features)
    start_scope()
    seed(random_seed)
    prefs.codegen.target = selected.codegen_target
    defaultclock.dt = selected.timestep_ms * ms
    equations = """
    dv/dt = (v_rest - v + g) / tau_m : volt (unless refractory)
    dg/dt = -g / tau_syn : volt (unless refractory)
    """
    neurons = NeuronGroup(
        len(arrays.neuron_ids),
        equations,
        threshold="v > v_threshold",
        reset="v = v_reset; g = 0*mV",
        refractory=selected.refractory_ms * ms,
        method="linear",
        namespace={
            "v_rest": selected.resting_mv * mV,
            "v_threshold": selected.threshold_mv * mV,
            "v_reset": selected.reset_mv * mV,
            "tau_m": selected.membrane_tau_ms * ms,
            "tau_syn": selected.synaptic_tau_ms * ms,
        },
    )
    neurons.v = selected.resting_mv * mV
    recurrent = Synapses(neurons, neurons, model="w : volt", on_pre="g += w")
    recurrent.connect(i=arrays.presynaptic, j=arrays.postsynaptic)
    recurrent.w = arrays.signed_connectivity * selected.weight_per_synapse_mv * mV
    recurrent.delay = selected.synaptic_delay_ms * ms
    inputs = PoissonGroup(sensory_neurons, rates=rates * Hz)
    stimulation = Synapses(
        inputs,
        neurons,
        on_pre="g += stimulation_weight",
        namespace={
            "stimulation_weight": selected.weight_per_synapse_mv
            * selected.stimulation_weight_scale
            * mV
        },
    )
    stimulation.connect(i=np.arange(sensory_neurons), j=np.arange(sensory_neurons))
    monitor = SpikeMonitor(neurons)
    network = Network(neurons, recurrent, inputs, stimulation, monitor)
    started = perf_counter()
    network.run(duration_ms * ms)
    elapsed = perf_counter() - started
    counts = np.asarray(monitor.count, dtype=np.int64)
    active = np.flatnonzero(counts)
    return FullBrainResult(
        neurons=len(arrays.neuron_ids),
        connections=len(arrays.presynaptic),
        duration_ms=duration_ms,
        total_spikes=int(counts.sum()),
        active_neurons=len(active),
        max_spikes_per_neuron=int(counts.max(initial=0)),
        elapsed_seconds=elapsed,
        active_flywire_ids=[int(arrays.neuron_ids[index]) for index in active[:20]],
    )
