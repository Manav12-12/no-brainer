from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.brain.plasticity import train_kc_mbon_nociception
from sentinel.brain.runner import run_brain_from_nociception
from sentinel.config import JevConfig, load_yaml
from sentinel.connectome.loader import synthetic_connectome
from sentinel.cyberbody.attacker import KillChainStage
from sentinel.cyberbody.env import CyberRange
from sentinel.data.features import (
    FEATURE_NAMES,
    FeaturePipeline,
    serialize_features,
    serialize_vector,
)
from sentinel.data.synthetic import generate_synthetic_events
from sentinel.eval.baselines import AnomalyEnsemble
from sentinel.jev.local_backend import LocalBackend
from sentinel.orchestrator.drive import DriveParameters, HomeostaticDrive
from sentinel.orchestrator.reflex_arc import evaluate_reflex


def validation_traces(root: Path) -> list[dict[str, Any]]:
    training = generate_synthetic_events(800, 1729)
    pipeline = FeaturePipeline.create().fit(training)
    transformed = pipeline.transform(training)
    values = transformed.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    labels = transformed["label"].to_numpy(dtype=np.int64)
    reflex_model = LocalBackend().fit(values, labels)
    anomaly_ensemble = AnomalyEnsemble(1729).fit(values[labels == 0], epochs=5)
    validation_features = []
    for benign_only in (False, True):
        for episode_index in range(10):
            seed = 30_000 + int(benign_only) * 1_000 + episode_index
            environment = CyberRange(12, 16, seed, benign_only=benign_only)
            validation_features.extend(
                environment.observe().features.copy() for _ in range(16)
            )
    anomaly_ensemble.calibrate(np.asarray(validation_features, dtype=np.float64))
    jev_config = JevConfig.model_validate(load_yaml(root / "configs/jev.yaml"))
    selected = np.concatenate(
        [np.flatnonzero(labels == 0)[:16], np.flatnonzero(labels == 1)[:16]]
    )
    graph = synthetic_connectome(1729)
    training_pain = np.asarray(
        [
            evaluate_reflex(
                reflex_model,
                serialize_features(transformed.iloc[index]),
                jev_config.reflex_confidence_threshold,
                jev_config.known_pattern_threshold,
            ).pain_signal
            for index in selected
        ],
        dtype=np.float64,
    )
    training_anomaly = np.asarray(
        [anomaly_ensemble.score(values[index]) for index in selected],
        dtype=np.float64,
    )
    train_kc_mbon_nociception(
        graph,
        training_pain,
        training_anomaly,
        labels[selected],
        duration_ms=50.0,
        random_seed=21_729,
        learning_rate=0.05,
    )
    traces: list[dict[str, Any]] = []
    for benign_only in (False, True):
        for episode_index in range(10):
            seed = 30_000 + int(benign_only) * 1_000 + episode_index
            environment = CyberRange(12, 16, seed, benign_only=benign_only)
            events = []
            for _ in range(16):
                observation = environment.observe()
                state = serialize_vector(
                    observation.features,
                    host_role=observation.host_role,
                    segment=observation.segment,
                    recent_event_count=observation.step,
                )
                reflex = evaluate_reflex(
                    reflex_model,
                    state,
                    jev_config.reflex_confidence_threshold,
                    jev_config.known_pattern_threshold,
                )
                anomaly_signal = anomaly_ensemble.score(observation.features)
                brain, _ = run_brain_from_nociception(
                    graph,
                    reflex.pain_signal,
                    anomaly_signal,
                    seed + observation.step,
                    duration_ms=50.0,
                )
                attacked = observation.attacker.stage not in {
                    KillChainStage.DORMANT,
                    KillChainStage.CONTAINED,
                }
                events.append(
                    {
                        "host_id": observation.host_id,
                        "pain_signal": reflex.pain_signal,
                        "novelty": brain.novelty_score,
                        "drive_stimulation": brain.drive_stimulation,
                        "attacked": attacked,
                    }
                )
            traces.append({"benign_only": benign_only, "events": events})
    return traces


def score(
    traces: list[dict[str, Any]], parameters: DriveParameters
) -> dict[str, float]:
    detections = 0
    false_actions = 0
    containment_steps: list[int] = []
    attack_episodes = 0
    benign_episodes = 0
    for trace in traces:
        drive = HomeostaticDrive(parameters)
        acted = False
        contained = False
        if trace["benign_only"]:
            benign_episodes += 1
        else:
            attack_episodes += 1
        for step, event in enumerate(trace["events"]):
            stimulation = 0.0 if contained else float(event["drive_stimulation"])
            observation = drive.observe(str(event["host_id"]), stimulation)
            if observation.current >= observation.threshold and not acted:
                acted = True
                drive.relieve(str(event["host_id"]))
                if bool(event["attacked"]):
                    detections += 1
                    containment_steps.append(step)
                    contained = True
                else:
                    false_actions += 1
    return {
        "detection_rate": detections / attack_episodes,
        "false_action_rate": false_actions / benign_episodes,
        "mean_time_to_contain": float(np.mean(containment_steps))
        if containment_steps
        else float("inf"),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    traces = validation_traces(root)
    candidates: list[tuple[float, float, float, DriveParameters]] = []
    attempted: list[tuple[dict[str, float], DriveParameters]] = []
    for window in (3, 4, 5):
        for decay in (0.75, 0.82, 0.9):
            for floor in (0.10, 0.15, 0.20, 0.25):
                for threshold in (0.4, 0.6, 0.8, 1.0, 1.2):
                    parameters = DriveParameters(
                        window=window,
                        decay=decay,
                        gain=1.0,
                        stimulation_floor=floor,
                        threshold=threshold,
                        relief_fraction=0.15,
                    )
                    metrics = score(traces, parameters)
                    attempted.append((metrics, parameters))
                    if metrics["false_action_rate"] <= 0.05:
                        candidates.append(
                            (
                                metrics["detection_rate"],
                                -metrics["mean_time_to_contain"],
                                -metrics["false_action_rate"],
                                parameters,
                            )
                        )
    parameters = max(candidates, key=lambda item: item[:3])[3] if candidates else None
    lowest_false_rate = min(
        attempted,
        key=lambda item: (
            item[0]["false_action_rate"],
            -item[0]["detection_rate"],
        ),
    )
    result = {
        "status": "selected" if parameters is not None else "no_candidate",
        "validation": score(traces, parameters) if parameters is not None else None,
        "parameters": parameters.__dict__ if parameters is not None else None,
        "lowest_false_action_candidate": {
            "metrics": lowest_false_rate[0],
            "parameters": lowest_false_rate[1].__dict__,
        },
        "validation_episodes": {
            "attack": 10,
            "benign": 10,
            "steps_each": 16,
            "seeds": "30000-30009 and 31000-31009",
        },
        "selection": (
            "maximize attack-episode detection, then minimize containment time, "
            "subject to benign-episode false-action rate <= 0.05"
        ),
        "held_out_used": False,
        "failure_reason": None
        if parameters is not None
        else "No candidate met the precommitted false-action rate <= 0.05 cap.",
    }
    destination = root / "artifacts/drive-validation.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
