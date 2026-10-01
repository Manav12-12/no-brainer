from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentinel.brain.plasticity import train_kc_mbon_weights
from sentinel.config import JevConfig, config_hash, load_yaml
from sentinel.connectome.loader import random_sparse_mushroom_body, synthetic_connectome
from sentinel.connectome.shuffle import degree_preserving_shuffle
from sentinel.cyberbody.actions import DefensiveAction
from sentinel.cyberbody.attacker import KillChainStage
from sentinel.cyberbody.env import CyberRange
from sentinel.data.features import FEATURE_NAMES, FeaturePipeline, serialize_vector
from sentinel.data.synthetic import generate_synthetic_events
from sentinel.eval.baselines import AutoencoderBaseline, IsolationBaseline
from sentinel.eval.report import write_results
from sentinel.jev.budget import BudgetGuard
from sentinel.jev.cache import DecisionCache
from sentinel.jev.interface import DecisionModel
from sentinel.jev.local_backend import LocalBackend
from sentinel.jev.replay_backend import ReplayBackend
from sentinel.jev.typesafe_backend import TypeSafeBackend
from sentinel.orchestrator.drive import DriveParameters, HomeostaticDrive
from sentinel.orchestrator.escalation import escalate_to_brain
from sentinel.orchestrator.reflex_arc import evaluate_reflex
from sentinel.seed import set_global_seed


def make_jev(root: Path, config: JevConfig, mode: str) -> DecisionModel:
    cache = DecisionCache(root / config.cache_path)
    if mode == "replay":
        return ReplayBackend(cache, config.model_id, config.serialization_version)
    if mode not in {"live", "record"}:
        raise ValueError(f"unsupported Jev mode: {mode}")
    return TypeSafeBackend(
        model_id=config.model_id,
        base_url=config.base_url,
        endpoint=config.endpoint,
        cache=cache,
        budget=BudgetGuard(
            config.max_calls,
            config.estimated_cost_usd_ceiling,
            config.input_usd_per_million_tokens,
        ),
        serialization_version=config.serialization_version,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        backoff_initial_seconds=config.backoff_initial_seconds,
        backoff_max_seconds=config.backoff_max_seconds,
        jitter_fraction=config.backoff_jitter_fraction,
    )


def train_local(
    seed: int,
) -> tuple[
    LocalBackend,
    IsolationBaseline,
    AutoencoderBaseline,
    NDArray[np.float64],
    NDArray[np.int64],
]:
    training = generate_synthetic_events(800, seed)
    pipeline = FeaturePipeline.create().fit(training)
    transformed = pipeline.transform(training)
    values = transformed.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    labels = transformed["label"].to_numpy(dtype=np.int64)
    local = LocalBackend().fit(values, labels)
    benign = values[labels == 0]
    return (
        local,
        IsolationBaseline(seed).fit(benign),
        AutoencoderBaseline(seed).fit(benign, epochs=5),
        values,
        labels,
    )


def run_episode(
    arm: str,
    seed: int,
    steps: int,
    jev: DecisionModel,
    local: LocalBackend,
    isolation: IsolationBaseline,
    autoencoder: AutoencoderBaseline,
    known_pattern_threshold: float = 0.27,
    reflex_confidence_threshold: float = 0.28,
    brain_duration_ms: float = 50.0,
    plasticity_features: NDArray[np.float64] | None = None,
    plasticity_labels: NDArray[np.int64] | None = None,
    plasticity_examples: int = 32,
    plasticity_learning_rate: float = 0.05,
    drive_parameters: DriveParameters | None = None,
) -> dict[str, Any]:
    environment = CyberRange(12, steps, seed)
    base_graph = synthetic_connectome(seed)
    graphs = {
        "B2": base_graph,
        "B3": base_graph,
        "B4": degree_preserving_shuffle(base_graph, seed),
        "B5": random_sparse_mushroom_body(seed),
    }
    training_delta = 0.0
    training_edges = 0
    drive = HomeostaticDrive(drive_parameters or DriveParameters())
    drive_actions = 0
    peak_drive = 0.0
    if (
        arm in graphs
        and plasticity_features is not None
        and plasticity_labels is not None
    ):
        benign = np.flatnonzero(plasticity_labels == 0)[: plasticity_examples // 2]
        attack = np.flatnonzero(plasticity_labels == 1)[: plasticity_examples // 2]
        selected = np.concatenate([benign, attack])
        report = train_kc_mbon_weights(
            graphs[arm],
            plasticity_features[selected],
            plasticity_labels[selected],
            duration_ms=brain_duration_ms,
            random_seed=seed + 20_000,
            learning_rate=plasticity_learning_rate,
        )
        training_delta = report.l1_weight_delta
        training_edges = report.updated_edges
    actions = 0
    attack_events = 0
    detected_attacks = 0
    reflex_handled = 0
    ascended = 0
    jev_failures = 0
    layer_latency_ms = 0.0
    contained_at: int | None = None
    for _ in range(steps):
        observation = environment.observe()
        attacked = observation.attacker.stage not in {
            KillChainStage.DORMANT,
            KillChainStage.CONTAINED,
        }
        attack_events += int(attacked)
        state = serialize_vector(
            observation.features,
            host_role=observation.host_role,
            segment=observation.segment,
            recent_event_count=observation.step,
        )
        action = DefensiveAction.NO_OP
        started = time.perf_counter()
        if arm == "B1":
            reflex = evaluate_reflex(
                jev,
                state,
                reflex_confidence_threshold,
                known_pattern_threshold,
            )
            jev_failures += int(not reflex.available)
            reflex_handled += int(not reflex.ascend)
            ascended += int(reflex.ascend)
            action = reflex.action or DefensiveAction.NO_OP
        elif arm == "B1L":
            reflex = evaluate_reflex(
                local,
                state,
                reflex_confidence_threshold,
                known_pattern_threshold,
            )
            reflex_handled += int(not reflex.ascend)
            ascended += int(reflex.ascend)
            action = reflex.action or DefensiveAction.NO_OP
        elif arm in graphs:
            use_brain = arm != "B3"
            if arm == "B3":
                reflex = evaluate_reflex(
                    jev,
                    state,
                    reflex_confidence_threshold,
                    known_pattern_threshold,
                )
                jev_failures += int(not reflex.available)
                reflex_handled += int(not reflex.ascend)
                action = reflex.action or DefensiveAction.NO_OP
                use_brain = reflex.ascend
            if use_brain:
                ascended += 1
                brain, _ = escalate_to_brain(
                    graphs[arm],
                    observation.features,
                    seed + observation.step,
                    brain_duration_ms,
                )
                drive_observation = drive.observe(
                    observation.host_id, brain.drive_stimulation
                )
                peak_drive = max(peak_drive, drive_observation.current)
                if (
                    drive_observation.current >= drive_observation.threshold
                    and not environment.hosts[observation.host_id].isolated
                ):
                    action = DefensiveAction.ISOLATE_HOST
                    drive.relieve(observation.host_id)
                    drive_actions += 1
        elif arm == "B6":
            if isolation.detect(observation.features) or autoencoder.detect(
                observation.features
            ):
                action = DefensiveAction.ISOLATE_HOST
        layer_latency_ms += (time.perf_counter() - started) * 1000
        if action != DefensiveAction.NO_OP:
            actions += 1
            environment.act(action, observation.host_id)
            detected_attacks += int(attacked)
        if (
            observation.attacker.stage == KillChainStage.CONTAINED
            and contained_at is None
        ):
            contained_at = observation.step
    return {
        "arm": arm,
        "seed": seed,
        "synthetic_mode": True,
        "detection_rate": detected_attacks / attack_events if attack_events else 0.0,
        "false_action_rate": environment.injuries / max(1, steps - attack_events),
        "injuries": environment.injuries,
        "actions": actions,
        "time_to_contain": contained_at,
        "reflex_fraction": reflex_handled / steps,
        "ascend_fraction": ascended / steps,
        "jev_failures": jev_failures,
        "mean_layer_latency_ms": layer_latency_ms / steps,
        "plasticity_examples": plasticity_examples if arm in graphs else 0,
        "plasticity_updated_edges": training_edges,
        "plasticity_l1_weight_delta": training_delta,
        "drive_actions": drive_actions,
        "peak_drive": peak_drive,
    }


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("replay", "record", "live"), default="replay"
    )
    parser.add_argument("--run-fullbrain", action="store_true")
    arguments = parser.parse_args()
    if arguments.run_fullbrain:
        raise SystemExit("full-brain assets are absent; refusing to mix modes")
    root = Path(__file__).resolve().parents[1]
    experiment = load_yaml(root / "configs/experiments.yaml")
    cyber = load_yaml(root / "configs/cyberrange.yaml")
    brain_config = load_yaml(root / "configs/brain_mb_subnet.yaml")
    jev_config = JevConfig.model_validate(load_yaml(root / "configs/jev.yaml"))
    drive_parameters = DriveParameters(
        window=int(brain_config["drive_window"]),
        decay=float(brain_config["drive_decay"]),
        gain=float(brain_config["drive_gain"]),
        stimulation_floor=float(brain_config["drive_stimulation_floor"]),
        threshold=float(brain_config["drive_threshold"]),
        relief_fraction=float(brain_config["drive_relief_fraction"]),
    )
    jev = make_jev(root, jev_config, arguments.mode)
    rows: list[dict[str, Any]] = []
    for seed_value in experiment["seeds"]:
        seed = int(seed_value)
        set_global_seed(seed)
        local, isolation, autoencoder, training_values, training_labels = train_local(
            seed
        )
        for arm in experiment["arms"]:
            rows.append(
                run_episode(
                    str(arm),
                    seed,
                    int(cyber["episode_steps"]),
                    jev,
                    local,
                    isolation,
                    autoencoder,
                    jev_config.known_pattern_threshold,
                    jev_config.reflex_confidence_threshold,
                    float(brain_config["simulation_ms"]),
                    training_values,
                    training_labels,
                    int(brain_config["plasticity_examples"]),
                    float(brain_config["plasticity_learning_rate"]),
                    drive_parameters,
                )
            )
    metadata = {
        "seed_list": experiment["seeds"],
        "config_hash": config_hash(
            {
                "experiments": experiment,
                "cyberrange": cyber,
                "jev": jev_config.model_dump(),
                "brain": brain_config,
            }
        ),
        "git_commit": git_commit(root),
        "jev_pinned_model": jev_config.model_id,
        "mode": arguments.mode,
        "synthetic_mode": True,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("brian2", "numpy", "scikit-learn", "torch", "httpx")
        },
        "limitations": [
            "Synthetic traffic and synthetic connectome-like graph",
            "Replay cache misses count as Jev failures",
            "No full FlyWire simulation",
        ],
    }
    write_results(rows, metadata, root / "results")
    print(json.dumps({"episodes": len(rows), "output": "results"}))


if __name__ == "__main__":
    main()
