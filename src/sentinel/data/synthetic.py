from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.data.features import FEATURE_NAMES

ATTACK_FAMILIES = ("recon", "credential", "lateral", "exfiltration")


def generate_synthetic_events(count: int, seed: int) -> pd.DataFrame:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, 0.35, count)
    families = np.full(count, "benign", dtype=object)
    attacked = np.flatnonzero(labels)
    families[attacked] = rng.choice(ATTACK_FAMILIES, len(attacked))
    values = rng.normal(0.0, 1.0, (count, len(FEATURE_NAMES)))
    for row in attacked:
        family_index = ATTACK_FAMILIES.index(str(families[row]))
        start = family_index * 5
        values[row, start : start + 5] += 2.0
    frame = pd.DataFrame(values, columns=FEATURE_NAMES)
    frame["timestamp"] = np.arange(count, dtype=np.int64)
    frame["label"] = labels
    frame["attack_family"] = families
    frame["host_role"] = rng.choice(("workstation", "server", "gateway"), count)
    frame["segment"] = rng.choice(("user", "server", "dmz"), count)
    frame["recent_event_count"] = rng.integers(0, 20, count)
    frame.attrs["synthetic_mode"] = True
    return frame
