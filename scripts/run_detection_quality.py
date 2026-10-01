from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from run_jev_calibration import calibration_frame
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from sentinel.brain.plasticity import train_kc_mbon_nociception
from sentinel.brain.runner import run_brain_from_nociception, run_brain_from_pain
from sentinel.connectome.loader import random_sparse_mushroom_body, synthetic_connectome
from sentinel.connectome.shuffle import degree_preserving_shuffle
from sentinel.data.features import FEATURE_NAMES, FeaturePipeline, serialize_features
from sentinel.data.loaders import load_processed
from sentinel.data.splits import family_holdout_split
from sentinel.data.unsw import UNSW_ARCHIVE_MEMBER, UNSW_EXCLUDED_COLUMNS
from sentinel.eval.baselines import AnomalyEnsemble
from sentinel.jev.cache import DecisionCache, cache_key
from sentinel.jev.replay_backend import ReplayBackend
from sentinel.orchestrator.reflex_arc import REFLEX_SPEC

THRESHOLDS = np.linspace(0.0, 1.0, 21)
CANDIDATE_SPEC = {
    "malicious_likelihood": {
        "type": "noul",
        "instructions": (
            "Estimate whether this normalized network flow is malicious. Use the "
            "named protocol, volume, timing, TCP, and connection-history signals."
        ),
        "criteria": {
            "true": "The flow is malicious or part of an attack campaign",
            "false": "The flow is benign operational traffic",
        },
    },
    "action": REFLEX_SPEC["action"],
}


def metric_summary(
    labels: NDArray[np.int64], scores: NDArray[np.float64]
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "benign_median": float(np.median(scores[labels == 0])),
        "attack_median": float(np.median(scores[labels == 1])),
    }


def select_threshold(labels: NDArray[np.int64], scores: NDArray[np.float64]) -> float:
    candidates: list[tuple[float, float, float]] = []
    for threshold in THRESHOLDS:
        predicted = scores >= threshold
        false_rate = float(predicted[labels == 0].mean())
        candidates.append(
            (
                float(f1_score(labels, predicted, zero_division=0)),
                -false_rate,
                float(threshold),
            )
        )
    return max(candidates)[2]


def evaluate(
    labels: NDArray[np.int64], scores: NDArray[np.float64], threshold: float
) -> dict[str, Any]:
    predicted = scores >= threshold
    curve = []
    for value in THRESHOLDS:
        selected = scores >= value
        curve.append(
            {
                "threshold": float(value),
                "detection_rate": float(selected[labels == 1].mean()),
                "false_positive_rate": float(selected[labels == 0].mean()),
            }
        )
    return {
        **metric_summary(labels, scores),
        "selected_threshold": threshold,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_positive_rate": float(predicted[labels == 0].mean()),
        "threshold_curve": curve,
    }


def semantic_feature_names(root: Path) -> list[str]:
    archive = root / "data/public/unsw_nb15_v3/UNSW-NB15-V3.zip"
    with ZipFile(archive) as bundle, bundle.open(UNSW_ARCHIVE_MEMBER) as handle:
        columns = pd.read_csv(handle, nrows=0).columns
    return [
        str(name)
        for name in columns
        if name != "label" and name not in UNSW_EXCLUDED_COLUMNS
    ]


def load_cached_partition(
    root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], NDArray[np.int64], NDArray[np.int64]]:
    frame = calibration_frame(root)
    backend = ReplayBackend(
        DecisionCache(root / "data/jev_cache/unsw_calibration.jsonl"),
        "jev-1.13.0",
        "sentinel-features-v1",
    )
    responses = []
    for _, row in frame.iterrows():
        result = backend.decide(serialize_features(row), REFLEX_SPEC)
        known = float(result.answers["known_pattern"].probabilities["true"])
        action = result.answers["action"]
        confidence = float(action.confidence or 0.0)
        old_pain = (known + 1.0 - action.probabilities.get("no_op", 0.0)) / 2.0
        improved_pain = (confidence + 1.0 - known) / 2.0
        responses.append(
            {
                "known": known,
                "action_confidence": confidence,
                "old_pain": old_pain,
                "pain": improved_pain,
            }
        )
    indices = np.arange(len(frame))
    validation, held_out = train_test_split(
        indices,
        test_size=0.5,
        random_state=1729,
        stratify=frame["attack_family"].to_numpy(),
    )
    return frame, responses, validation, held_out


def fit_anomaly_ensemble(
    root: Path, excluded_timestamps: set[int]
) -> tuple[AnomalyEnsemble, FeaturePipeline]:
    source = root / "data/public/unsw_nb15_v3/processed.parquet"
    frame = load_processed(source, synthetic_expected=False)
    split = family_holdout_split(frame, "worms")
    available = split.train[~split.train["timestamp"].isin(excluded_timestamps)]
    pipeline = FeaturePipeline.create().fit(available)
    transformed = pipeline.transform(available)
    values = transformed.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    labels = transformed["label"].to_numpy(dtype=np.int64)
    return AnomalyEnsemble(1729).fit(values[labels == 0], epochs=5), pipeline


def scale_from_validation(
    validation_scores: NDArray[np.float64], scores: NDArray[np.float64]
) -> NDArray[np.float64]:
    lower = float(validation_scores.min())
    upper = float(validation_scores.max())
    if upper <= lower:
        return np.zeros_like(scores)
    return np.asarray(np.clip((scores - lower) / (upper - lower), 0.0, 1.0))


def percentile_from_validation(
    validation_scores: NDArray[np.float64], scores: NDArray[np.float64]
) -> NDArray[np.float64]:
    reference = np.sort(validation_scores)
    return np.asarray(
        [
            np.searchsorted(reference, value, side="right") / len(reference)
            for value in scores
        ],
        dtype=np.float64,
    )


def neural_scores(
    graph: Any,
    pain: NDArray[np.float64],
    anomaly: NDArray[np.float64] | None,
    seeds: NDArray[np.int64],
) -> NDArray[np.float64]:
    output = []
    for index in range(len(pain)):
        if anomaly is None:
            result, _ = run_brain_from_pain(
                graph, float(pain[index]), int(seeds[index]), duration_ms=50.0
            )
        else:
            result, _ = run_brain_from_nociception(
                graph,
                float(pain[index]),
                float(anomaly[index]),
                int(seeds[index]),
                duration_ms=50.0,
            )
        output.append(result.drive_stimulation)
    return np.asarray(output, dtype=np.float64)


def train_graph(
    factory: Callable[[], Any],
    pain: NDArray[np.float64],
    anomaly: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[Any, dict[str, float | int]]:
    graph = factory()
    benign = np.flatnonzero(labels == 0)[:16]
    attacks = np.flatnonzero(labels == 1)[:16]
    selected = np.concatenate([benign, attacks])
    report = train_kc_mbon_nociception(
        graph,
        pain[selected],
        anomaly[selected],
        labels[selected],
        duration_ms=50.0,
        random_seed=50_000,
        learning_rate=0.05,
    )
    return graph, {
        "examples": report.examples,
        "updated_edges": report.updated_edges,
        "l1_weight_delta": report.l1_weight_delta,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    frame, responses, validation, held_out = load_cached_partition(root)
    labels = frame["label"].to_numpy(dtype=np.int64)
    pain = np.asarray([row["pain"] for row in responses], dtype=np.float64)
    old_pain = np.asarray([row["old_pain"] for row in responses], dtype=np.float64)

    names = semantic_feature_names(root)
    candidate_states = [
        {
            "serialization_version": "sentinel-semantic-features-v2-candidate",
            "signals": {
                name: float(frame.iloc[index][feature])
                for name, feature in zip(names, FEATURE_NAMES, strict=True)
            },
            "context": {
                "host_role": str(frame.iloc[index]["host_role"]),
                "segment": str(frame.iloc[index]["segment"]),
            },
        }
        for index in validation
    ]
    candidate_hits = sum(
        DecisionCache(root / "data/jev_cache/unsw_calibration.jsonl").get(
            cache_key(state, CANDIDATE_SPEC, "jev-1.13.0", "semantic-v2-candidate")
        )
        is not None
        for state in candidate_states
    )

    excluded = set(int(value) for value in frame["timestamp"])
    anomaly_model, pipeline = fit_anomaly_ensemble(root, excluded)
    transformed = pipeline.transform(frame)
    features = transformed.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    isolation_raw = np.asarray(
        [anomaly_model.isolation.score(row) for row in features], dtype=np.float64
    )
    autoencoder_raw = np.asarray(
        [anomaly_model.autoencoder.score(row) for row in features], dtype=np.float64
    )
    # Training-benign empirical CDFs saturated on the calibration cohort. Use
    # the unlabeled validation distribution as the fixed score scale instead;
    # labels remain reserved for threshold selection below.
    isolation_score = percentile_from_validation(
        isolation_raw[validation], isolation_raw
    )
    autoencoder_score = percentile_from_validation(
        autoencoder_raw[validation], autoencoder_raw
    )
    anomaly = np.maximum(isolation_score, autoencoder_score)
    y_validation = labels[validation]

    graph_factories: dict[str, Callable[[], Any]] = {
        "combined": lambda: synthetic_connectome(1729),
        "B4_shuffled": lambda: degree_preserving_shuffle(
            synthetic_connectome(1729), 1729
        ),
        "B5_random": lambda: random_sparse_mushroom_body(1729),
    }
    training: dict[str, dict[str, float | int]] = {}
    all_scores: dict[str, NDArray[np.float64]] = {
        "jev_reflex": pain,
        "B6_baseline": anomaly,
    }

    brain_graph, brain_training = train_graph(
        lambda: synthetic_connectome(1729),
        pain[validation],
        np.zeros(len(validation), dtype=np.float64),
        y_validation,
    )
    training["brain_drive"] = brain_training
    brain_raw = neural_scores(
        brain_graph,
        pain,
        None,
        np.arange(60_000, 60_000 + len(frame), dtype=np.int64),
    )
    all_scores["brain_drive"] = scale_from_validation(brain_raw[validation], brain_raw)

    for name, factory in graph_factories.items():
        graph, report = train_graph(
            factory, pain[validation], anomaly[validation], y_validation
        )
        training[name] = report
        raw = neural_scores(
            graph,
            pain,
            anomaly,
            np.arange(
                70_000,
                70_000 + len(frame),
                dtype=np.int64,
            ),
        )
        all_scores[name] = scale_from_validation(raw[validation], raw)

    thresholds = {
        name: select_threshold(y_validation, scores[validation])
        for name, scores in all_scores.items()
    }
    validation_metrics = {
        "old_pain": metric_summary(y_validation, old_pain[validation]),
        **{
            name: metric_summary(y_validation, scores[validation])
            for name, scores in all_scores.items()
        },
    }
    # This is the only point at which held-out labels are used.
    held_out_metrics = {
        name: evaluate(labels[held_out], scores[held_out], thresholds[name])
        for name, scores in all_scores.items()
    }
    result = {
        "step1": {
            "validation_metrics": validation_metrics,
            "decision": "improved_but_limited",
            "candidate_semantic_payload": {
                "feature_names": names,
                "validation_requests": len(candidate_states),
                "cache_hits": candidate_hits,
                "live_calls_required": len(candidate_states) - candidate_hits,
                "live_credentials_available": False,
                "measured_jev_metrics": None,
            },
            "payload_diagnosis": [
                "The v1 vector retains all 40 permitted numeric values.",
                "The v1 serializer removes all source feature names before Jev.",
                "host_role and segment are synthetic cycling metadata.",
                "recent_event_count is source_row modulo 10001, not an event window.",
            ],
        },
        "mixed_held_out": {
            "records": len(held_out),
            "benign": int(np.sum(labels[held_out] == 0)),
            "malicious": int(np.sum(labels[held_out] == 1)),
            "families": sorted(set(frame.iloc[held_out]["attack_family"])),
            "source_overlap_with_training": 0,
            "limitation": (
                "This is the pre-existing locked Jev cache partition; aggregate gate "
                "results were reported previously, so it is not a pristine unseen set."
            ),
        },
        "validation_selected_thresholds": thresholds,
        "plasticity": training,
        "held_out_metrics": held_out_metrics,
        "interpretation": {
            "record_level": True,
            "temporal_drive_limitation": (
                "UNSW IPs were intentionally excluded and role/segment are synthetic; "
                "scores test neural pressure per record, not genuine per-host "
                "sequences."
            ),
        },
    }
    destination = root / "artifacts/detection-quality.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
