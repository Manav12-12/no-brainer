from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel.brain.plasticity import train_kc_mbon_nociception
from sentinel.brain.runner import run_brain_from_nociception
from sentinel.config import JevConfig, load_yaml
from sentinel.connectome.loader import synthetic_connectome
from sentinel.cyberbody.actions import DefensiveAction, apply_action
from sentinel.cyberbody.topology import build_topology
from sentinel.data.features import FEATURE_NAMES, FeaturePipeline, serialize_features
from sentinel.data.loaders import load_processed
from sentinel.data.splits import family_holdout_split
from sentinel.eval.baselines import (
    AnomalyEnsemble,
    AutoencoderBaseline,
    IsolationBaseline,
)
from sentinel.jev.local_backend import LocalBackend
from sentinel.orchestrator.drive import DriveParameters, HomeostaticDrive
from sentinel.orchestrator.reflex_arc import evaluate_reflex
from sentinel.orchestrator.strategy import select_response_strategy
from sentinel.seed import set_global_seed


def transform(pipeline: FeaturePipeline, frame: pd.DataFrame) -> pd.DataFrame:
    return pipeline.transform(frame)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    jev_config = JevConfig.model_validate(load_yaml(root / "configs/jev.yaml"))
    brain_config = load_yaml(root / "configs/brain_mb_subnet.yaml")
    source = root / "data/public/unsw_nb15_v3/processed.parquet"
    frame = load_processed(source, synthetic_expected=False)
    split = family_holdout_split(frame, "worms")
    pipeline = FeaturePipeline.create().fit(split.train)
    train = transform(pipeline, split.train)
    validation = transform(pipeline, split.validation)
    test = transform(pipeline, split.test)
    x_train = train.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_train = train["label"].to_numpy(dtype=np.int64)
    local = LocalBackend().fit(x_train, y_train)
    benign = x_train[y_train == 0]
    isolation = IsolationBaseline(1729).fit(benign)
    autoencoder = AutoencoderBaseline(1729).fit(benign, epochs=5)
    anomaly_ensemble = AnomalyEnsemble(1729).fit(benign, epochs=5)
    anomaly_ensemble.calibrate(
        validation.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    )
    graph = synthetic_connectome(1729)
    plasticity_examples = int(brain_config["plasticity_examples"])
    benign_indices = np.flatnonzero(y_train == 0)[: plasticity_examples // 2]
    attack_indices = np.flatnonzero(y_train == 1)[: plasticity_examples // 2]
    plasticity_indices = np.concatenate([benign_indices, attack_indices])
    training_pain = np.asarray(
        [
            evaluate_reflex(
                local,
                serialize_features(train.iloc[index]),
                jev_config.reflex_confidence_threshold,
                jev_config.known_pattern_threshold,
            ).pain_signal
            for index in plasticity_indices
        ],
        dtype=np.float64,
    )
    training_anomaly = np.asarray(
        [anomaly_ensemble.score(x_train[index]) for index in plasticity_indices],
        dtype=np.float64,
    )
    plasticity_report = train_kc_mbon_nociception(
        graph,
        training_pain,
        training_anomaly,
        y_train[plasticity_indices],
        duration_ms=float(brain_config["simulation_ms"]),
        random_seed=20_000,
        learning_rate=float(brain_config["plasticity_learning_rate"]),
    )
    set_global_seed(1729)
    hosts = build_topology(12)
    drive = HomeostaticDrive(
        DriveParameters(
            window=int(brain_config["drive_window"]),
            decay=float(brain_config["drive_decay"]),
            gain=float(brain_config["drive_gain"]),
            stimulation_floor=float(brain_config["drive_stimulation_floor"]),
            threshold=float(brain_config["drive_threshold"]),
            relief_fraction=float(brain_config["drive_relief_fraction"]),
        )
    )
    rows: list[dict[str, Any]] = []
    first_action: int | None = None
    for event_index, (_, event) in enumerate(test.iterrows()):
        state = serialize_features(event)
        started = time.perf_counter()
        reflex = evaluate_reflex(
            local,
            state,
            jev_config.reflex_confidence_threshold,
            jev_config.known_pattern_threshold,
        )
        reflex_action = reflex.action or DefensiveAction.NO_OP
        final_action = reflex_action
        values = event.loc[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        host_id = f"host-{event_index % len(hosts):02d}"
        if reflex.action is not None:
            apply_action(reflex.action, hosts[host_id])
        anomaly_signal = anomaly_ensemble.score(values)
        brain, _ = run_brain_from_nociception(
            graph,
            reflex.pain_signal,
            anomaly_signal,
            1729 + event_index,
            float(brain_config["simulation_ms"]),
        )
        drive_observation = drive.observe(host_id, brain.drive_stimulation)
        strategy_name = None
        if drive_observation.current >= drive_observation.threshold:
            strategy = select_response_strategy(
                host_id,
                hosts,
                drive.values(),
                drive_observation.threshold,
                reflex.action,
            )
            strategy_name = strategy.name
            for directive in strategy.directives:
                outcome = apply_action(directive.action, hosts[directive.host_id])
                if outcome.changed:
                    final_action = directive.action
            drive.relieve(host_id)
        latency_ms = (time.perf_counter() - started) * 1000
        baseline_action = bool(isolation.detect(values) or autoencoder.detect(values))
        acted = final_action != DefensiveAction.NO_OP
        if acted and first_action is None:
            first_action = event_index
        rows.append(
            {
                "source_row": int(event["timestamp"]),
                "family": str(event["attack_family"]),
                "label": int(event["label"]),
                "reflex_action": reflex_action.value,
                "final_action": final_action.value,
                "pain_signal": reflex.pain_signal,
                "brain_stimulation": brain.drive_stimulation,
                "strategy": strategy_name,
                "confidence": reflex.confidence,
                "pipeline_latency_ms": latency_ms,
                "baseline_action": baseline_action,
            }
        )
    actions = np.asarray([row["final_action"] != "no_op" for row in rows])
    baseline_actions = np.asarray([row["baseline_action"] for row in rows])
    latencies = np.asarray([row["pipeline_latency_ms"] for row in rows])
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "dataset": "UNSW-NB15 V3",
        "held_out_family": "worms",
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "held_out_family_absent_from_train": "worms" not in set(train["attack_family"]),
        "pipeline_detection_rate": float(actions.mean()),
        "baseline_detection_rate": float(baseline_actions.mean()),
        "brain_fraction": 1.0,
        "mean_pipeline_latency_ms": float(latencies.mean()),
        "p95_pipeline_latency_ms": float(np.percentile(latencies, 95)),
        "containment_event_index": first_action,
        "containment_definition": (
            "zero-based held-out event index of first simulated non-no-op action"
        ),
        "events": len(rows),
        "git_commit": commit,
        "synthetic_mode": False,
        "event_records": rows,
        "plasticity": {
            "examples": plasticity_report.examples,
            "updated_edges": plasticity_report.updated_edges,
            "l1_weight_delta": plasticity_report.l1_weight_delta,
            "evaluation_weights": "trained",
        },
    }
    destination = root / "artifacts/public-dataset-eval.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {key: value for key, value in result.items() if key != "event_records"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
