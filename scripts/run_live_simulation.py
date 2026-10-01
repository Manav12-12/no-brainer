from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentinel.brain.plasticity import train_kc_mbon_nociception
from sentinel.brain.runner import run_brain_from_nociception
from sentinel.config import JevConfig, load_yaml
from sentinel.connectome.loader import synthetic_connectome
from sentinel.cyberbody.actions import DefensiveAction
from sentinel.cyberbody.env import CyberRange
from sentinel.data.features import FEATURE_NAMES, FeaturePipeline, serialize_vector
from sentinel.data.synthetic import generate_synthetic_events
from sentinel.eval.baselines import AnomalyEnsemble
from sentinel.jev.local_backend import LocalBackend
from sentinel.orchestrator.drive import DriveParameters, HomeostaticDrive
from sentinel.orchestrator.reflex_arc import evaluate_reflex
from sentinel.orchestrator.strategy import select_response_strategy
from sentinel.seed import set_global_seed

SIMULATION_LOCK = threading.Lock()
EPISODE_SEED_OFFSETS = (0, 13, 29)
STEPS_PER_EPISODE = 8


def training_data(
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], LocalBackend, AnomalyEnsemble]:
    training = generate_synthetic_events(800, seed)
    pipeline = FeaturePipeline.create().fit(training)
    transformed = pipeline.transform(training)
    values = transformed.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    labels = transformed["label"].to_numpy(dtype=np.int64)
    benign = values[labels == 0]
    return (
        values,
        labels,
        LocalBackend().fit(values, labels),
        AnomalyEnsemble(seed).fit(benign, epochs=5).calibrate(values),
    )


def graph_payload(seed: int) -> tuple[Any, dict[str, Any]]:
    graph = synthetic_connectome(seed)
    nodes = [
        {"id": int(node), "population": str(data["population"])}
        for node, data in graph.nodes(data=True)
    ]
    edges = [
        {
            "source": int(source),
            "target": int(target),
            "weight": float(data["weight"]),
            "excitatory": bool(data["excitatory"]),
        }
        for source, target, data in graph.edges(data=True)
    ]
    return graph, {"nodes": nodes, "edges": edges}


def simulation_config(seed: int) -> dict[str, Any]:
    _, graph = graph_payload(seed)
    root = Path(__file__).resolve().parents[1]
    brain_config = load_yaml(root / "configs/brain_mb_subnet.yaml")
    return {
        "title": "Drosophila Sentinel",
        "seed": seed,
        "synthetic_mode": True,
        "anatomy": "stylized fly brain and ventral nerve cord",
        "model": "48-node synthetic connectome proxy",
        "episodes": len(EPISODE_SEED_OFFSETS),
        "steps_per_episode": STEPS_PER_EPISODE,
        "total_steps": len(EPISODE_SEED_OFFSETS) * STEPS_PER_EPISODE,
        "drive_threshold": float(brain_config["drive_threshold"]),
        "graph": graph,
    }


def event_payload(
    *,
    global_step: int,
    episode: int,
    observation: Any,
    reflex: Any,
    brain: Any,
    counts: dict[int, int],
    action: DefensiveAction,
    outcome: Any,
    environment: CyberRange,
    drive: HomeostaticDrive,
    drive_observation: Any,
    relief: tuple[float, float] | None,
    action_source: str,
    strategy_name: str | None,
    compute_ms: float,
) -> dict[str, Any]:
    return {
        "type": "simulation_event",
        "step": global_step,
        "episode": episode,
        "episode_step": observation.step,
        "stage": observation.attacker.stage.value,
        "target_host": observation.host_id,
        "segment": observation.segment,
        "role": observation.host_role,
        "compute_ms": compute_ms,
        "reflex": {
            "confidence": reflex.confidence,
            "pain_signal": reflex.pain_signal,
            "available": reflex.available,
            "reason": reflex.reason,
        },
        "brain": None
        if brain is None
        else {
            "novelty": brain.novelty_score,
            "sensory_stimulation": brain.sensory_stimulation,
            "drive_stimulation": brain.drive_stimulation,
            "threat_class": brain.threat_class,
            "population_activity": brain.population_activity,
            "active_nodes": [
                {"id": node, "spikes": count}
                for node, count in counts.items()
                if count > 0
            ],
            "total_spikes": sum(counts.values()),
        },
        "action": action.value,
        "action_source": action_source,
        "strategy": strategy_name,
        "drive": {
            "host_id": observation.host_id,
            "previous": drive_observation.previous,
            "before_action": drive_observation.current,
            "current": drive.value(observation.host_id),
            "threshold": drive_observation.threshold,
            "stimulation": drive_observation.stimulation,
            "windowed_stimulation": drive_observation.windowed_stimulation,
            "crossed": drive_observation.crossed,
            "relieved": relief is not None,
            "relief_before": relief[0] if relief is not None else None,
            "relief_after": relief[1] if relief is not None else None,
        },
        "outcome": None
        if outcome is None
        else {
            "changed": outcome.changed,
            "injury": outcome.injury,
            "cost": outcome.cost,
        },
        "hosts": [
            {
                "id": host.host_id,
                "segment": host.segment,
                "role": host.role,
                "compromised": host.compromised,
                "isolated": host.isolated,
                "rate_limited": host.rate_limited,
                "session_active": host.session_active,
                "drive": drive.value(host.host_id),
            }
            for host in environment.hosts.values()
        ],
    }


def simulation_events(seed: int = 1729) -> Iterator[dict[str, Any]]:
    set_global_seed(seed)
    root = Path(__file__).resolve().parents[1]
    jev_config = JevConfig.model_validate(load_yaml(root / "configs/jev.yaml"))
    brain_config = load_yaml(root / "configs/brain_mb_subnet.yaml")
    yield {
        "type": "status",
        "state": "training",
        "message": "Training KC→MBON weights",
    }
    values, labels, model, anomaly_ensemble = training_data(seed)
    benign = np.flatnonzero(labels == 0)[:16]
    attack = np.flatnonzero(labels == 1)[:16]
    selected = np.concatenate([benign, attack])
    graph, _ = graph_payload(seed)
    training_pain = np.asarray(
        [
            evaluate_reflex(
                model,
                serialize_vector(
                    values[index],
                    host_role="workstation",
                    segment="user",
                    recent_event_count=int(index),
                ),
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
        duration_ms=float(brain_config["simulation_ms"]),
        random_seed=seed + 20_000,
        learning_rate=float(brain_config["plasticity_learning_rate"]),
    )
    drive_parameters = DriveParameters(
        window=int(brain_config["drive_window"]),
        decay=float(brain_config["drive_decay"]),
        gain=float(brain_config["drive_gain"]),
        stimulation_floor=float(brain_config["drive_stimulation_floor"]),
        threshold=float(brain_config["drive_threshold"]),
        relief_fraction=float(brain_config["drive_relief_fraction"]),
    )
    global_step = 0
    yield {"type": "status", "state": "running", "message": "Simulation live"}

    for episode, offset in enumerate(EPISODE_SEED_OFFSETS, start=1):
        episode_seed = seed + offset
        environment = CyberRange(12, STEPS_PER_EPISODE, episode_seed)
        drive = HomeostaticDrive(drive_parameters)
        for _ in range(STEPS_PER_EPISODE):
            started = time.perf_counter()
            observation = environment.observe()
            state = serialize_vector(
                observation.features,
                host_role=observation.host_role,
                segment=observation.segment,
                recent_event_count=observation.step,
            )
            reflex = evaluate_reflex(
                model,
                state,
                jev_config.reflex_confidence_threshold,
                jev_config.known_pattern_threshold,
            )
            anomaly_signal = anomaly_ensemble.score(observation.features)
            brain, counts = run_brain_from_nociception(
                graph,
                reflex.pain_signal,
                anomaly_signal,
                episode_seed + observation.step,
                float(brain_config["simulation_ms"]),
            )
            action = reflex.action or DefensiveAction.NO_OP
            action_source = "reflex" if reflex.action is not None else "none"
            drive_observation = drive.observe(
                observation.host_id, brain.drive_stimulation
            )

            outcome = None
            relief = None
            strategy_name = None
            if reflex.action is not None:
                outcome = environment.act(action, observation.host_id)
            if drive_observation.current >= drive_observation.threshold:
                strategy = select_response_strategy(
                    observation.host_id,
                    environment.hosts,
                    drive.values(),
                    drive_observation.threshold,
                    reflex.action,
                )
                strategy_name = strategy.name
                strategy_changed = False
                for directive in strategy.directives:
                    strategy_outcome = environment.act(
                        directive.action, directive.host_id
                    )
                    if strategy_outcome.changed:
                        action = directive.action
                        outcome = strategy_outcome
                        action_source = "brain_strategy"
                        strategy_changed = True
                if strategy_changed:
                    relief = drive.relieve(observation.host_id)
            compute_ms = (time.perf_counter() - started) * 1000
            yield event_payload(
                global_step=global_step,
                episode=episode,
                observation=observation,
                reflex=reflex,
                brain=brain,
                counts=counts,
                action=action,
                outcome=outcome,
                environment=environment,
                drive=drive,
                drive_observation=drive_observation,
                relief=relief,
                action_source=action_source,
                strategy_name=strategy_name,
                compute_ms=compute_ms,
            )
            global_step += 1

    yield {"type": "complete", "state": "complete", "message": "Run complete"}


def handler_factory(directory: Path, seed: int) -> type[SimpleHTTPRequestHandler]:
    config = json.dumps(simulation_config(seed), separators=(",", ":")).encode()
    config_script = b"window.SENTINEL_LIVE_CONFIG=" + config + b";"

    class LiveHandler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self) -> None:
            if self.path == "/api/config":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(config)))
                self.end_headers()
                self.wfile.write(config)
                return
            if self.path == "/config.js":
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(config_script)))
                self.end_headers()
                self.wfile.write(config_script)
                return
            if self.path.startswith("/api/stream"):
                self.stream_simulation()
                return
            super().do_GET()

        def stream_simulation(self) -> None:
            if not SIMULATION_LOCK.acquire(blocking=False):
                self.send_error(409, "another live simulation is already running")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                for item in simulation_events(seed):
                    encoded = json.dumps(item, separators=(",", ":")).encode()
                    self.wfile.write(b"data:" + encoded + b"\n\n")
                    self.wfile.flush()
                    if item["type"] == "simulation_event":
                        time.sleep(1.15)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                SIMULATION_LOCK.release()

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:",
            )
            super().end_headers()

        def log_message(self, message: str, *args: Any) -> None:
            print(f"live: {message % args}")

    return LiveHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live neural simulation")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--no-open", action="store_true")
    arguments = parser.parse_args()
    if not 1024 <= arguments.port <= 65_535:
        raise SystemExit("port must be between 1024 and 65535")

    root = Path(__file__).resolve().parents[1]
    handler = handler_factory(root / "live", arguments.seed)
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), handler)
    url = f"http://127.0.0.1:{arguments.port}"
    print(f"Live simulation ready at {url}")
    print("Brian2 runs as events stream to the browser. No live Jev calls are used.")
    if not arguments.no_open:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live simulation")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
