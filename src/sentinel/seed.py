from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
    try:
        from brian2 import seed as brian_seed

        brian_seed(seed)
    except ImportError:
        pass
