from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from sentinel.cyberbody.topology import Host


class KillChainStage(str, Enum):
    DORMANT = "dormant"
    RECON = "recon"
    INITIAL_ACCESS = "initial_access"
    LATERAL = "lateral"
    EXFILTRATION = "exfiltration"
    CONTAINED = "contained"


@dataclass
class AttackerState:
    stage: KillChainStage
    target_host: str | None = None


class AbstractAttacker:
    """A label-only state machine. It performs no real network operations."""

    def __init__(self, benign_only: bool, seed: int) -> None:
        self.state = AttackerState(KillChainStage.DORMANT)
        self.benign_only = benign_only
        self.rng = np.random.default_rng(seed)

    def step(self, hosts: dict[str, Host]) -> AttackerState:
        if self.benign_only:
            return self.state
        stage = self.state.stage
        if stage == KillChainStage.DORMANT:
            self.state = AttackerState(KillChainStage.RECON)
        elif stage == KillChainStage.RECON:
            candidates = [host for host in hosts.values() if not host.isolated]
            if candidates:
                chosen = candidates[int(self.rng.integers(0, len(candidates)))]
                chosen.compromised = True
                self.state = AttackerState(
                    KillChainStage.INITIAL_ACCESS, chosen.host_id
                )
        elif stage == KillChainStage.INITIAL_ACCESS:
            self.state.stage = KillChainStage.LATERAL
        elif stage == KillChainStage.LATERAL:
            self.state.stage = KillChainStage.EXFILTRATION
        target = hosts.get(self.state.target_host or "")
        if target is not None and target.isolated:
            self.state.stage = KillChainStage.CONTAINED
        return self.state
