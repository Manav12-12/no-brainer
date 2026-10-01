from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DriveParameters:
    window: int = 4
    decay: float = 0.82
    gain: float = 1.0
    stimulation_floor: float = 0.15
    threshold: float = 0.75
    relief_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("drive window must contain at least two observations")
        if not 0 <= self.decay < 1 or self.gain <= 0:
            raise ValueError("drive decay and gain are invalid")
        if not 0 <= self.stimulation_floor <= 1 or self.threshold <= 0:
            raise ValueError("drive floor and threshold are invalid")
        if not 0 <= self.relief_fraction < 1:
            raise ValueError("drive relief fraction must be in [0, 1)")


@dataclass(frozen=True)
class DriveObservation:
    host_id: str
    stimulation: float
    windowed_stimulation: float
    previous: float
    current: float
    threshold: float
    crossed: bool


class HomeostaticDrive:
    """Persistent per-host pressure built from a window of neural stimulation."""

    def __init__(self, parameters: DriveParameters) -> None:
        self.parameters = parameters
        self._values: dict[str, float] = defaultdict(float)
        self._windows: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=parameters.window)
        )

    def value(self, host_id: str) -> float:
        return self._values[host_id]

    def observe(self, host_id: str, neural_stimulation: float) -> DriveObservation:
        bounded = min(1.0, max(0.0, neural_stimulation))
        stimulation = max(0.0, bounded - self.parameters.stimulation_floor)
        window = self._windows[host_id]
        window.append(stimulation)
        windowed = sum(window) / self.parameters.window
        previous = self._values[host_id]
        current = max(
            0.0,
            previous * self.parameters.decay + self.parameters.gain * windowed,
        )
        self._values[host_id] = current
        return DriveObservation(
            host_id,
            stimulation,
            windowed,
            previous,
            current,
            self.parameters.threshold,
            previous < self.parameters.threshold <= current,
        )

    def relieve(self, host_id: str) -> tuple[float, float]:
        before = self._values[host_id]
        after = before * self.parameters.relief_fraction
        self._values[host_id] = after
        self._windows[host_id].clear()
        return before, after
