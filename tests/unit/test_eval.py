from pathlib import Path

import numpy as np
import pytest

from sentinel.connectome.loader import random_sparse_mushroom_body
from sentinel.eval.ablations import default_ablations
from sentinel.eval.baselines import AutoencoderBaseline, IsolationBaseline
from sentinel.eval.metrics import detection_rate, false_action_rate, mean_ci
from sentinel.eval.report import write_results


@pytest.mark.unit
def test_metric_summaries() -> None:
    estimate = mean_ci(np.array([1.0, 2.0, 3.0]))
    assert estimate.mean == 2.0
    assert estimate.ci95_low < estimate.mean < estimate.ci95_high
    single = mean_ci(np.array([2.0]))
    assert single.ci95_low == single.ci95_high == 2.0
    with pytest.raises(ValueError, match="empty"):
        mean_ci(np.array([]))
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    actions = np.array([0, 1, 1, 1], dtype=np.int64)
    assert detection_rate(labels, actions) == 1.0
    assert false_action_rate(labels, actions) == 0.5


@pytest.mark.unit
def test_local_anomaly_baselines_train() -> None:
    rng = np.random.default_rng(3)
    benign = rng.normal(0, 1, (80, 40))
    sample = np.full(40, 10.0)
    isolation = IsolationBaseline(3).fit(benign)
    autoencoder = AutoencoderBaseline(3).fit(benign, epochs=1)
    assert isinstance(isolation.detect(sample), bool)
    assert isinstance(autoencoder.detect(sample), bool)


@pytest.mark.unit
def test_controls_ablation_grid_and_report(tmp_path: Path) -> None:
    graph = random_sparse_mushroom_body(4)
    assert graph.number_of_nodes() == 48
    assert graph.number_of_edges() > 0
    assert len(default_ablations()) == 36
    rows = [
        {
            "arm": arm,
            "detection_rate": 0.5,
            "false_action_rate": 0.1,
            "injuries": 1,
            "reflex_fraction": 0.5,
            "ascend_fraction": 0.5,
            "jev_failures": 0,
            "mean_layer_latency_ms": 1.0,
        }
        for arm in ("B0", "B1", "B1L")
    ]
    write_results(rows, {"synthetic_mode": True}, tmp_path)
    assert (tmp_path / "episodes.json").exists()
    assert (tmp_path / "detection_rate.png").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "reflex_vs_ascend.png").exists()
