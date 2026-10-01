from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sentinel.eval.metrics import mean_ci


def write_results(
    rows: list[dict[str, Any]], metadata: dict[str, Any], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    arms = sorted({str(row["arm"]) for row in rows})
    summary = {
        arm: {
            metric: mean_ci(
                np.asarray(
                    [float(row[metric]) for row in rows if row["arm"] == arm],
                    dtype=np.float64,
                )
            ).__dict__
            for metric in (
                "detection_rate",
                "false_action_rate",
                "injuries",
                "reflex_fraction",
                "brain_fraction",
                "jev_failures",
                "mean_layer_latency_ms",
            )
        }
        for arm in arms
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    detection = [
        sum(float(row["detection_rate"]) for row in rows if row["arm"] == arm)
        / sum(1 for row in rows if row["arm"] == arm)
        for arm in arms
    ]
    injuries = [
        sum(float(row["injuries"]) for row in rows if row["arm"] == arm)
        / sum(1 for row in rows if row["arm"] == arm)
        for arm in arms
    ]
    _bar_plot(arms, detection, "Detection rate", output / "detection_rate.png")
    _bar_plot(arms, injuries, "Mean injury count", output / "injury_count.png")
    reflex = [float(summary[arm]["reflex_fraction"]["mean"]) for arm in arms]
    brain = [float(summary[arm]["brain_fraction"]["mean"]) for arm in arms]
    _grouped_plot(
        arms,
        reflex,
        brain,
        "Immediate reflex and brain-processing fractions",
        output / "reflex_vs_brain.png",
    )
    comparison = [
        float(summary[arm]["detection_rate"]["mean"]) for arm in ("B1", "B1L")
    ]
    _bar_plot(
        ["Jev replay", "LocalBackend"],
        comparison,
        "Jev replay versus local baseline",
        output / "jev_vs_local.png",
    )


def _bar_plot(labels: list[str], values: list[float], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(labels, values)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _grouped_plot(
    labels: list[str], first: list[float], second: list[float], title: str, path: Path
) -> None:
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(positions - 0.2, first, 0.4, label="reflex")
    axis.bar(positions + 0.2, second, 0.4, label="brain processed")
    axis.set_xticks(positions, labels, rotation=45)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
