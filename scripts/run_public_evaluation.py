from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel.connectome.loader import synthetic_connectome
from sentinel.cyberbody.actions import DefensiveAction
from sentinel.data.features import FEATURE_NAMES, FeaturePipeline, serialize_features
from sentinel.data.loaders import load_processed
from sentinel.data.splits import family_holdout_split
from sentinel.eval.baselines import AutoencoderBaseline, IsolationBaseline
from sentinel.jev.local_backend import LocalBackend
from sentinel.orchestrator.escalation import escalate_to_brain
from sentinel.orchestrator.reflex_arc import evaluate_reflex
from sentinel.seed import set_global_seed


def transform(pipeline: FeaturePipeline, frame: pd.DataFrame) -> pd.DataFrame:
    return pipeline.transform(frame)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
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
    graph = synthetic_connectome(1729)
    set_global_seed(1729)
    rows: list[dict[str, Any]] = []
    first_action: int | None = None
    for event_index, (_, event) in enumerate(test.iterrows()):
        state = serialize_features(event)
        started = time.perf_counter()
        reflex = evaluate_reflex(local, state, 0.8)
        reflex_action = reflex.action or DefensiveAction.NO_OP
        final_action = reflex_action
        values = event.loc[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        if reflex.ascend:
            _, final_action = escalate_to_brain(
                graph,
                values,
                1729 + event_index,
                15.0,
            )
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
                "ascended": reflex.ascend,
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
        "ascend_fraction": float(np.mean([row["ascended"] for row in rows])),
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
    }
    destination = root / "artifacts/public-dataset-eval.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {key: value for key, value in result.items() if key != "event_records"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
