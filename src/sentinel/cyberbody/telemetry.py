from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sentinel.cyberbody.attacker import KillChainStage

_STAGE_OFFSET = {
    KillChainStage.DORMANT: 0.0,
    KillChainStage.RECON: 0.5,
    KillChainStage.INITIAL_ACCESS: 1.0,
    KillChainStage.LATERAL: 1.5,
    KillChainStage.EXFILTRATION: 2.0,
    KillChainStage.CONTAINED: 0.0,
}


def event_features(stage: KillChainStage, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, 40)
    values[:8] += _STAGE_OFFSET[stage]
    return values
