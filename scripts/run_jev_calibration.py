from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel.config import JevConfig, load_yaml
from sentinel.data.features import FeaturePipeline, serialize_features
from sentinel.data.loaders import load_processed
from sentinel.data.splits import family_holdout_split
from sentinel.jev.budget import BudgetGuard
from sentinel.jev.cache import DecisionCache, cache_key
from sentinel.jev.calibration import brier_score, expected_calibration_error
from sentinel.jev.replay_backend import ReplayBackend
from sentinel.jev.typesafe_backend import TypeSafeBackend
from sentinel.orchestrator.reflex_arc import REFLEX_SPEC


def calibration_frame(root: Path) -> pd.DataFrame:
    source = root / "data/public/unsw_nb15_v3/processed.parquet"
    frame = load_processed(source, synthetic_expected=False)
    split = family_holdout_split(frame, "worms")
    pipeline = FeaturePipeline.create().fit(split.train)
    benign = frame[frame["attack_family"] == "benign"].head(200)
    attacks = pd.concat(
        [
            group.head(20)
            for family, group in frame[frame["attack_family"] != "benign"].groupby(
                "attack_family", sort=True
            )
        ],
        ignore_index=False,
    )
    selected = pd.concat([benign, attacks], ignore_index=True)
    if len(selected) != 400 or selected["label"].value_counts().to_dict() != {
        0: 200,
        1: 200,
    }:
        raise ValueError(
            "calibration selection must contain 200 benign and 200 attacks"
        )
    return pipeline.transform(selected)


def reliability_bins(
    labels: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        output.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_confidence": float(probabilities[mask].mean())
                if mask.any()
                else None,
                "observed_rate": float(labels[mask].mean()) if mask.any() else None,
            }
        )
    return output


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_manifest(cache_path: Path) -> str:
    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    manifest = cache_path.parent / "MANIFEST.sha256"
    manifest.write_text(f"{digest}  {cache_path.name}\n", encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "record", "replay"), required=True)
    parser.add_argument("--confirm-live", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = JevConfig.model_validate(load_yaml(root / "configs/jev.yaml"))
    cache_path = root / "data/jev_cache/unsw_calibration.jsonl"
    cache = DecisionCache(cache_path)
    frame = calibration_frame(root)
    states = [serialize_features(row) for _, row in frame.iterrows()]
    keys = [
        cache_key(state, REFLEX_SPEC, config.model_id, config.serialization_version)
        for state in states
    ]
    hits = sum(cache.get(key) is not None for key in keys)
    if arguments.mode == "plan":
        print(
            json.dumps(
                {
                    "unique_requests": len(set(keys)),
                    "cache_hits": hits,
                    "live_calls_required": len(keys) - hits,
                    "hard_cost_ceiling_usd": config.estimated_cost_usd_ceiling,
                    "estimated_cost_usd": None,
                    "dataset": "UNSW-NB15 V3",
                    "model": config.model_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.mode == "record":
        if not arguments.confirm_live:
            raise SystemExit("refusing live calibration without --confirm-live")
        budget = BudgetGuard(
            max_calls=400,
            cost_ceiling_usd=config.estimated_cost_usd_ceiling,
            input_usd_per_million_tokens=config.input_usd_per_million_tokens,
        )
        backend: TypeSafeBackend | ReplayBackend = TypeSafeBackend(
            model_id=config.model_id,
            base_url=config.base_url,
            endpoint=config.endpoint,
            cache=cache,
            budget=budget,
            serialization_version=config.serialization_version,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            backoff_initial_seconds=config.backoff_initial_seconds,
            backoff_max_seconds=config.backoff_max_seconds,
            jitter_fraction=config.backoff_jitter_fraction,
        )
    else:
        budget = None
        backend = ReplayBackend(cache, config.model_id, config.serialization_version)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, state in enumerate(states):
        result = backend.decide(state, REFLEX_SPEC)
        known = result.answers["known_pattern"].probabilities["true"]
        action = result.answers["action"]
        confidence = min(abs(2 * known - 1), action.confidence or 0.0)
        records.append(
            {
                "index": index,
                "label": int(frame.iloc[index]["label"]),
                "family": str(frame.iloc[index]["attack_family"]),
                "known_probability": known,
                "selected_action": action.selected,
                "action_confidence": action.confidence,
                "reflex_confidence": confidence,
                "reflex_acted": bool(
                    known >= 0.5 and confidence >= 0.8 and action.selected != "no_op"
                ),
                "latency_ms": result.latency_ms,
                "input_tokens": result.usage_input_tokens,
            }
        )
    wall_seconds = time.perf_counter() - started
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    probabilities = np.asarray(
        [record["known_probability"] for record in records], dtype=np.float64
    )
    latencies = np.asarray(
        [record["latency_ms"] for record in records], dtype=np.float64
    )
    token_count = sum(int(record["input_tokens"] or 0) for record in records)
    cache_sha256 = write_manifest(cache_path)
    report = {
        "dataset": "UNSW-NB15 V3",
        "records": len(records),
        "balanced_labels": {"benign": 200, "attack": 200},
        "model": config.model_id,
        "mode": arguments.mode,
        "logical_live_calls_this_run": budget.calls if budget is not None else 0,
        "cache_hits_before_run": hits,
        "input_tokens": token_count,
        "measured_cost_usd": token_count
        * config.input_usd_per_million_tokens
        / 1_000_000,
        "brier_score": brier_score(labels, probabilities),
        "ece_10_bins": expected_calibration_error(labels, probabilities, bins=10),
        "latency_ms": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "wall_seconds": wall_seconds,
        "reliability_bins": reliability_bins(labels, probabilities),
        "cache_sha256": cache_sha256,
        "git_commit": git_commit(root),
        "records_detail": records,
    }
    destination = root / "artifacts/jev-live-calibration.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {key: value for key, value in report.items() if key != "records_detail"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
