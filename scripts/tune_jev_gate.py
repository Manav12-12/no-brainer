from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


def gate_metrics(
    records: list[dict[str, Any]],
    indices: np.ndarray,
    known_threshold: float,
    confidence_threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray([int(records[index]["label"]) for index in indices])
    actions = np.asarray(
        [
            float(records[index]["known_probability"]) >= known_threshold
            and float(records[index]["reflex_confidence"]) >= confidence_threshold
            and records[index]["selected_action"] != "no_op"
            for index in indices
        ]
    )
    attacks = labels == 1
    benign = labels == 0
    return {
        "records": len(indices),
        "actions": int(actions.sum()),
        "detection_rate": float(actions[attacks].mean()),
        "false_action_rate": float(actions[benign].mean()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts/jev-live-calibration.json"
    records: list[dict[str, Any]] = json.loads(source.read_text(encoding="utf-8"))[
        "records_detail"
    ]
    indices = np.arange(len(records))
    strata = np.asarray([str(record["family"]) for record in records])
    validation, report = train_test_split(
        indices, test_size=0.5, random_state=1729, stratify=strata
    )

    candidates: list[tuple[float, float, float, float]] = []
    for known_percent in range(20, 39):
        for confidence_percent in range(12, 61):
            known_threshold = known_percent / 100
            confidence_threshold = confidence_percent / 100
            metrics = gate_metrics(
                records,
                validation,
                known_threshold,
                confidence_threshold,
            )
            false_rate = float(metrics["false_action_rate"])
            if false_rate <= 0.05:
                candidates.append(
                    (
                        float(metrics["detection_rate"]),
                        -false_rate,
                        known_threshold,
                        confidence_threshold,
                    )
                )
    if not candidates:
        raise RuntimeError("no Jev gate satisfies the validation false-action cap")
    _, _, known_threshold, confidence_threshold = max(candidates)
    result = {
        "source": str(source.relative_to(root)),
        "method": (
            "family-stratified 50/50 cached split; maximize validation detection "
            "subject to false-action rate <= 0.05"
        ),
        "seed": 1729,
        "known_pattern_threshold": known_threshold,
        "reflex_confidence_threshold": confidence_threshold,
        "validation": gate_metrics(
            records, validation, known_threshold, confidence_threshold
        ),
        "locked_report_partition": gate_metrics(
            records, report, known_threshold, confidence_threshold
        ),
        "limitations": [
            "No independent Jev cache exists; this partitions the historical "
            "cache without new network calls.",
            "The historical Phase 1 aggregate already described all 400 records.",
        ],
    }
    destination = root / "artifacts/jev-gate-validation.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
