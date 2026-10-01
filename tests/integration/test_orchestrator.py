from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sentinel.connectome.loader import synthetic_connectome
from sentinel.cyberbody.actions import DefensiveAction
from sentinel.cyberbody.topology import build_topology
from sentinel.data.features import FeaturePipeline, serialize_features
from sentinel.data.synthetic import generate_synthetic_events
from sentinel.jev.cache import DecisionCache, cache_key
from sentinel.jev.interface import DecisionResult, QuestionResult
from sentinel.jev.replay_backend import ReplayBackend
from sentinel.orchestrator.descending import adjust_reflex_threshold
from sentinel.orchestrator.drive import DriveParameters, HomeostaticDrive
from sentinel.orchestrator.escalation import escalate_to_brain
from sentinel.orchestrator.feedback import analyst_oracle, apply_episode_feedback
from sentinel.orchestrator.reflex_arc import REFLEX_SPEC, evaluate_reflex
from sentinel.orchestrator.strategy import select_response_strategy


def feature_state() -> dict[str, object]:
    frame = generate_synthetic_events(20, 41)
    pipeline = FeaturePipeline.create().fit(frame)
    return serialize_features(pipeline.transform(frame).iloc[0])


def cached_backend(tmp_path: Path, known: float, confidence: float) -> ReplayBackend:
    state = feature_state()
    cache = DecisionCache(tmp_path / "replay.jsonl")
    result = DecisionResult(
        answers={
            "known_pattern": QuestionResult(
                type="noul",
                selected=known,
                probabilities={"false": 1 - known, "true": known},
            ),
            "action": QuestionResult(
                type="choice",
                selected="isolate_host",
                probabilities={
                    "no_op": 0.02,
                    "isolate_host": 0.9,
                    "block_flow": 0.02,
                    "rate_limit": 0.03,
                    "revoke_session": 0.03,
                },
                confidence=confidence,
            ),
        },
        model_version="jev-1.13.0",
        latency_ms=10,
    )
    key = cache_key(state, REFLEX_SPEC, "jev-1.13.0", "sentinel-features-v1")
    cache.append(key, result, "2026-09-30T00:00:00Z")
    return ReplayBackend(cache, "jev-1.13.0", "sentinel-features-v1")


@pytest.mark.integration
def test_reflex_act_and_ascend_paths(tmp_path: Path) -> None:
    state = feature_state()
    immediate = evaluate_reflex(cached_backend(tmp_path, 0.95, 0.9), state, 0.8)
    assert immediate.action == DefensiveAction.ISOLATE_HOST
    assert immediate.pain_signal > 0.9

    other = tmp_path / "other"
    other.mkdir()
    uncertain = evaluate_reflex(cached_backend(other, 0.55, 0.6), state, 0.8)
    assert uncertain.action is None and uncertain.available
    assert uncertain.pain_signal > 0.7

    calibrated = tmp_path / "calibrated"
    calibrated.mkdir()
    empirical = evaluate_reflex(
        cached_backend(calibrated, 0.30, 0.30), state, 0.28, 0.27
    )
    assert empirical.action == DefensiveAction.ISOLATE_HOST
    assert empirical.pain_signal > 0.6


@pytest.mark.integration
def test_replay_miss_emits_explicit_zero_pain(tmp_path: Path) -> None:
    state = feature_state()
    replay = ReplayBackend(
        DecisionCache(tmp_path / "missing.jsonl"),
        "jev-1.13.0",
        "sentinel-features-v1",
    )
    reflex = evaluate_reflex(replay, state, 0.8)
    assert reflex.pain_signal == 0 and not reflex.available
    brain, _ = escalate_to_brain(synthetic_connectome(), reflex.pain_signal, 5, 20)
    assert 0 <= brain.novelty_score <= 1


@pytest.mark.integration
def test_brain_strategy_can_coordinate_correlated_hosts() -> None:
    hosts = build_topology(12)
    strategy = select_response_strategy(
        "host-00",
        hosts,
        {"host-00": 0.9, "host-03": 0.7},
        threshold=1.0,
        reflex_action=DefensiveAction.RATE_LIMIT,
    )
    assert strategy.name == "segment_coordination"
    assert {item.action for item in strategy.directives} == {
        DefensiveAction.ISOLATE_HOST,
        DefensiveAction.RATE_LIMIT,
    }
    escalation = select_response_strategy(
        "host-00",
        hosts,
        {"host-00": 1.1},
        threshold=1.0,
        reflex_action=DefensiveAction.RATE_LIMIT,
    )
    assert escalation.name == "escalate_insufficient_reflex"
    assert escalation.directives[0].action == DefensiveAction.ISOLATE_HOST


@pytest.mark.integration
def test_descending_feedback_loop_is_bounded() -> None:
    threshold = adjust_reflex_threshold(0.8, 0.9, 0.5)
    assert 0.5 <= threshold <= 0.99
    verdict = analyst_oracle(True, DefensiveAction.ISOLATE_HOST)
    updated = apply_episode_feedback(np.zeros((2, 1)), np.ones(2), verdict, 0.1)
    assert np.array_equal(updated, np.full((2, 1), 0.1))
    injury = analyst_oracle(False, DefensiveAction.ISOLATE_HOST)
    assert not injury.correct and injury.reward == -1


@pytest.mark.integration
def test_action_stops_attack_telemetry_and_relieves_drive() -> None:
    from sentinel.cyberbody.env import CyberRange

    environment = CyberRange(12, 8, 1729)
    drive = HomeostaticDrive(
        DriveParameters(
            window=3,
            decay=0.9,
            gain=1.0,
            stimulation_floor=0.0,
            threshold=0.8,
            relief_fraction=0.2,
        )
    )
    first = environment.observe()
    second = environment.observe()
    assert second.attacker.stage.value == "initial_access"
    host_id = second.host_id
    drive.observe(host_id, 1.0)
    crossing = drive.observe(host_id, 1.0)
    assert crossing.current > crossing.previous
    before, relieved = drive.relieve(host_id)
    environment.act(DefensiveAction.ISOLATE_HOST, host_id)
    contained = environment.observe()
    assert contained.attacker.stage.value == "contained"
    after = drive.observe(host_id, 0.0)
    assert relieved < before
    assert after.current < relieved
    assert first.attacker.stage.value == "recon"
