from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrainAblation:
    stimulation_rate_hz: float
    simulation_ms: float
    readout: str
    shuffled_labels: bool = False


def default_ablations() -> list[BrainAblation]:
    return [
        BrainAblation(rate, duration, readout, shuffled)
        for rate in (75.0, 150.0, 225.0)
        for duration in (50.0, 100.0)
        for readout in ("KC", "MBON", "descending")
        for shuffled in (False, True)
    ]
