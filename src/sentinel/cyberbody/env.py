from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sentinel.cyberbody.actions import ActionOutcome, DefensiveAction, apply_action
from sentinel.cyberbody.attacker import AbstractAttacker, AttackerState
from sentinel.cyberbody.telemetry import event_features
from sentinel.cyberbody.topology import Host, build_topology


@dataclass(frozen=True)
class Observation:
    step: int
    attacker: AttackerState
    host_id: str
    features: NDArray[np.float64]
    host_role: str
    segment: str


class CyberRange:
    def __init__(
        self, host_count: int, episode_steps: int, seed: int, benign_only: bool = False
    ) -> None:
        if episode_steps < 1:
            raise ValueError("episode must contain at least one step")
        self.hosts: dict[str, Host] = build_topology(host_count)
        self.episode_steps = episode_steps
        self.seed = seed
        self.attacker = AbstractAttacker(benign_only, seed)
        self.current_step = 0
        self.total_cost = 0.0
        self.injuries = 0

    def observe(self) -> Observation:
        if self.current_step >= self.episode_steps:
            raise StopIteration
        state = self.attacker.step(self.hosts)
        host_id = state.target_host or sorted(self.hosts)[0]
        host = self.hosts[host_id]
        observation = Observation(
            step=self.current_step,
            attacker=AttackerState(state.stage, state.target_host),
            host_id=host_id,
            features=event_features(state.stage, self.seed + self.current_step),
            host_role=host.role,
            segment=host.segment,
        )
        self.current_step += 1
        return observation

    def act(self, action: DefensiveAction, host_id: str) -> ActionOutcome:
        if host_id not in self.hosts:
            raise KeyError(f"unknown simulated host: {host_id}")
        outcome = apply_action(action, self.hosts[host_id])
        self.total_cost += outcome.cost
        self.injuries += outcome.injury
        return outcome
