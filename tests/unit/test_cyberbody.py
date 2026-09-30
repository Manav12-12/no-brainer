import numpy as np
import pytest

from sentinel.cyberbody.actions import DefensiveAction, apply_action
from sentinel.cyberbody.attacker import AbstractAttacker, KillChainStage
from sentinel.cyberbody.env import CyberRange
from sentinel.cyberbody.telemetry import event_features
from sentinel.cyberbody.topology import build_topology


@pytest.mark.unit
def test_topology_limits_and_fields() -> None:
    hosts = build_topology(12)
    assert len(hosts) == 12
    assert {host.segment for host in hosts.values()} == {"user", "server", "dmz"}
    with pytest.raises(ValueError):
        build_topology(9)


@pytest.mark.unit
def test_attacker_fsm_is_abstract_and_seeded() -> None:
    hosts = build_topology()
    attacker = AbstractAttacker(False, 5)
    stages = [attacker.step(hosts).stage for _ in range(4)]
    assert stages == [
        KillChainStage.RECON,
        KillChainStage.INITIAL_ACCESS,
        KillChainStage.LATERAL,
        KillChainStage.EXFILTRATION,
    ]
    benign = AbstractAttacker(True, 5)
    assert benign.step(build_topology()).stage == KillChainStage.DORMANT


@pytest.mark.unit
def test_action_effects_and_injury() -> None:
    healthy = build_topology()["host-00"]
    outcome = apply_action(DefensiveAction.ISOLATE_HOST, healthy)
    assert healthy.isolated and outcome.injury == 1 and outcome.cost == 5.0
    compromised = build_topology()["host-00"]
    compromised.compromised = True
    outcome = apply_action(DefensiveAction.REVOKE_SESSION, compromised)
    assert not compromised.session_active and outcome.injury == 0


@pytest.mark.unit
def test_telemetry_deterministic() -> None:
    first = event_features(KillChainStage.LATERAL, 2)
    second = event_features(KillChainStage.LATERAL, 2)
    assert np.array_equal(first, second)
    assert first.shape == (40,)


@pytest.mark.integration
def test_seeded_episode_contains_only_simulated_mutations() -> None:
    environment = CyberRange(12, 5, 99)
    observations = []
    for _ in range(5):
        observation = environment.observe()
        observations.append(observation)
        if observation.attacker.stage == KillChainStage.INITIAL_ACCESS:
            environment.act(DefensiveAction.ISOLATE_HOST, observation.host_id)
    assert len(observations) == 5
    assert environment.total_cost == 5.0
    with pytest.raises(StopIteration):
        environment.observe()
