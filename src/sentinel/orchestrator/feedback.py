from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sentinel.brain.plasticity import offline_kc_mbon_update
from sentinel.cyberbody.actions import DefensiveAction


@dataclass(frozen=True)
class AnalystVerdict:
    correct: bool
    reward: float


def analyst_oracle(compromised: bool, action: DefensiveAction) -> AnalystVerdict:
    acted = action != DefensiveAction.NO_OP
    correct = acted == compromised
    return AnalystVerdict(correct=correct, reward=1.0 if correct else -1.0)


def apply_episode_feedback(
    weights: NDArray[np.float64],
    kc_activity: NDArray[np.float64],
    verdict: AnalystVerdict,
    learning_rate: float,
) -> NDArray[np.float64]:
    error = np.asarray([verdict.reward], dtype=np.float64)
    return offline_kc_mbon_update(weights, kc_activity, error, learning_rate)
