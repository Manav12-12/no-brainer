from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from sentinel.brain.encoder import SensoryEncoder
from sentinel.brain.full_flywire import load_flywire_arrays
from sentinel.brain.lif_model import LIFParameters, simulate_lif
from sentinel.brain.plasticity import (
    offline_kc_mbon_update,
    train_kc_mbon_weights,
)
from sentinel.brain.readout import brain_readout, read_population_activity
from sentinel.brain.runner import run_brain
from sentinel.connectome.annotations import population_ids, require_populations
from sentinel.connectome.loader import load_connectome, synthetic_connectome
from sentinel.connectome.subgraph import mushroom_body_subgraph


@pytest.mark.unit
def test_connectome_loader_and_subgraph(tmp_path: Path) -> None:
    source = tmp_path / "graph.csv"
    source.write_text(
        "source,target,weight,excitatory\n1,2,3,true\n2,3,1,false\n",
        encoding="utf-8",
    )
    graph = load_connectome(source)
    assert graph.edges[1, 2]["weight"] == 3.0
    synthetic = synthetic_connectome(1)
    require_populations(synthetic)
    assert mushroom_body_subgraph(synthetic).number_of_nodes() == 48
    assert len(population_ids(synthetic, "KC")) == 24


@pytest.mark.unit
def test_connectome_loader_rejects_bad_input(tmp_path: Path) -> None:
    text = tmp_path / "graph.txt"
    text.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV"):
        load_connectome(text)
    with pytest.raises(ValueError, match="unknown population"):
        population_ids(nx.DiGraph(), "bad")
    with pytest.raises(ValueError, match="lacks"):
        require_populations(nx.DiGraph())


@pytest.mark.unit
def test_encoder_is_finite_and_bounded() -> None:
    encoder = SensoryEncoder(8, 150)
    rates = encoder.encode(np.linspace(-100, 100, 40))
    assert rates.shape == (8,)
    assert np.isfinite(rates).all()
    assert (rates >= 0).all() and (rates <= 150).all()
    with pytest.raises(ValueError, match="40"):
        encoder.encode(np.zeros(39))


@pytest.mark.unit
def test_readout_and_plasticity() -> None:
    graph = synthetic_connectome()
    counts = {node: 2 for node in graph.nodes}
    activity = read_population_activity(graph, counts)
    assert activity == {"KC": 2.0, "MBON": 2.0, "descending": 2.0}
    output = brain_readout(graph, counts)
    assert output.action == "isolate_host"
    weights = np.zeros((2, 1))
    updated = offline_kc_mbon_update(weights, np.ones(2), np.ones(1), 0.1)
    assert np.array_equal(updated, np.full((2, 1), 0.1))
    with pytest.raises(ValueError, match="dimensions"):
        offline_kc_mbon_update(weights, np.ones(3), np.ones(1), 0.1)


@pytest.mark.integration
def test_small_lif_is_reproducible_and_non_silent() -> None:
    graph = synthetic_connectome()
    sensory = population_ids(graph, "ORN")
    rates = np.full(len(sensory), 150.0)
    params = LIFParameters(codegen_target="numpy")
    first = simulate_lif(graph, sensory, rates, 50.0, params, 77)
    second = simulate_lif(graph, sensory, rates, 50.0, params, 77)
    assert first == second
    for population in ("ORN", "PN", "KC", "MBON", "descending"):
        assert sum(first[node] for node in population_ids(graph, population)) > 0


@pytest.mark.integration
def test_brain_runner_synthetic_mode() -> None:
    output = run_brain(synthetic_connectome(), np.ones(40), 12, duration_ms=20)
    assert 0 <= output.novelty_score <= 1
    assert output.action in {"no_op", "isolate_host"}


@pytest.mark.integration
def test_plasticity_changes_weights_used_by_brain() -> None:
    graph = synthetic_connectome()
    edge = (population_ids(graph, "KC")[0], population_ids(graph, "MBON")[0])
    initial = float(graph.edges[edge]["weight"])
    report = train_kc_mbon_weights(
        graph,
        np.full((1, 40), 20.0),
        np.ones(1, dtype=np.int64),
        duration_ms=50.0,
        random_seed=77,
        learning_rate=0.05,
    )
    assert report.updated_edges == 96
    assert report.l1_weight_delta > 0
    assert float(graph.edges[edge]["weight"]) != initial
    output = run_brain(graph, np.full(40, 20.0), 78, duration_ms=50.0)
    assert output.population_activity["MBON"] > 0


@pytest.mark.unit
def test_full_flywire_array_loader(tmp_path: Path) -> None:
    completeness = tmp_path / "completeness.csv"
    connectivity = tmp_path / "connectivity.parquet"
    pd.DataFrame({"flywire_id": [10, 20, 30]}).to_csv(completeness, index=False)
    pd.DataFrame(
        {
            "Presynaptic_Index": [0, 1],
            "Postsynaptic_Index": [1, 2],
            "Excitatory x Connectivity": [3, -2],
        }
    ).to_parquet(connectivity, index=False)
    arrays = load_flywire_arrays(completeness, connectivity)
    assert arrays.neuron_ids.tolist() == [10, 20, 30]
    assert arrays.signed_connectivity.tolist() == [3.0, -2.0]
