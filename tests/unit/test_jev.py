from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from sentinel.data.features import FeaturePipeline, serialize_features
from sentinel.data.synthetic import generate_synthetic_events
from sentinel.jev.budget import BudgetExceeded, BudgetGuard
from sentinel.jev.cache import DecisionCache, cache_key
from sentinel.jev.calibration import brier_score, expected_calibration_error
from sentinel.jev.interface import (
    DecisionResult,
    DecisionUnavailable,
    QuestionResult,
    VersionMismatch,
    validate_spec,
)
from sentinel.jev.local_backend import LocalBackend
from sentinel.jev.payload_guard import guard_payload
from sentinel.jev.replay_backend import ReplayBackend
from sentinel.jev.typesafe_backend import TypeSafeBackend, normalize_probabilities
from tests.fixtures.fake_jev_server import transport, valid_response

SPEC = {"known": {"type": "noul", "instructions": "Is this a known pattern?"}}


def state() -> dict[str, object]:
    frame = generate_synthetic_events(20, 11)
    pipeline = FeaturePipeline.create().fit(frame)
    return serialize_features(pipeline.transform(frame).iloc[0])


def backend(
    tmp_path: Path, mock: httpx.MockTransport, sleeps: list[float]
) -> TypeSafeBackend:
    return TypeSafeBackend(
        model_id="jev-1.13.0",
        base_url="https://api.typesafe.ai",
        endpoint="/v1/systemone",
        cache=DecisionCache(tmp_path / "cache.jsonl"),
        budget=BudgetGuard(5, 1.0, 0.042),
        serialization_version="sentinel-features-v1",
        transport=mock,
        sleep=sleeps.append,
    )


@pytest.mark.unit
def test_payload_guard_rejects_unproduced_content() -> None:
    good = state()
    guard_payload(good)
    for bad in (
        {**good, "path": "/etc/passwd"},
        {**good, "serialization_version": "unknown"},
        {**good, "features": [0.0]},
        {**good, "features": [float("nan")] * 40},
    ):
        with pytest.raises(ValueError):
            guard_payload(bad)  # type: ignore[arg-type]


@pytest.mark.unit
def test_cache_key_is_canonical_and_replay_miss_raises(tmp_path: Path) -> None:
    item = state()
    reversed_item = dict(reversed(list(item.items())))
    assert cache_key(item, SPEC, "jev-1.13.0", "v1") == cache_key(
        reversed_item, SPEC, "jev-1.13.0", "v1"
    )
    replay = ReplayBackend(DecisionCache(tmp_path / "missing"), "jev-1.13.0", "v1")
    with pytest.raises(KeyError, match="cache miss"):
        replay.decide(item, SPEC)


@pytest.mark.unit
def test_cache_round_trip(tmp_path: Path) -> None:
    item = state()
    cache = DecisionCache(tmp_path / "cache.jsonl")
    key = cache_key(item, SPEC, "jev-1.13.0", "v1")
    result = DecisionResult(
        answers={
            "known": QuestionResult(
                type="noul", selected=0.9, probabilities={"false": 0.1, "true": 0.9}
            )
        },
        model_version="jev-1.13.0",
        latency_ms=1,
    )
    cache.append(key, result, "2026-09-30T00:00:00Z")
    assert ReplayBackend(cache, "jev-1.13.0", "v1").decide(item, SPEC) == result
    stored = json.loads((tmp_path / "cache.jsonl").read_text().splitlines()[0])
    assert stored["returned_version"] == "jev-1.13.0"
    assert "raw_response" in stored


@pytest.mark.unit
def test_live_backend_validates_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-placeholder")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["Authorization"].startswith("Bearer ")
        assert request.url.host == "api.typesafe.ai"
        return httpx.Response(200, json=valid_response())

    client = backend(tmp_path, transport(handler), [])
    result = client.decide(state(), SPEC)
    assert result.model_version == "jev-1.13.0"
    assert result.answers["known"].probabilities["false"] == pytest.approx(0.1)
    assert result.answers["known"].probabilities["true"] == pytest.approx(0.9)
    assert calls == 1
    client.decide(state(), SPEC)
    assert calls == 1
    stored = json.loads((tmp_path / "cache.jsonl").read_text().splitlines()[0])
    assert stored["raw_response"]["usage"]["input_tokens"] == 100
    assert stored["normalized_response"] == stored["raw_response"]
    assert stored["normalization_events"] == []


@pytest.mark.unit
def test_probability_normalization_is_bounded_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO"):
        normalized, event = normalize_probabilities(
            {"no_op": 0.49, "isolate": 0.5}, "action"
        )
    assert sum(normalized.values()) == pytest.approx(1.0)
    assert event is not None
    assert event["question"] == "action"
    assert event["raw_sum"] == pytest.approx(0.99)
    assert "question=action" in caplog.text

    exact, exact_event = normalize_probabilities(
        {"no_op": 0.4, "isolate": 0.6}, "action"
    )
    assert exact == {"no_op": 0.4, "isolate": 0.6}
    assert exact_event is None
    for boundary in (0.98, 1.02):
        weight = 0.24595207670996116
        bounded, bounded_event = normalize_probabilities(
            {
                "no_op": boundary * weight,
                "isolate": boundary * (1.0 - weight),
            },
            "action",
        )
        assert sum(bounded.values()) == pytest.approx(1.0)
        assert bounded_event is not None
    with pytest.raises(DecisionUnavailable, match="outside normalization tolerance"):
        normalize_probabilities({"no_op": 0.4, "isolate": 0.5}, "action")


@pytest.mark.unit
def test_normalized_response_is_auditable_in_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-placeholder")
    spec = {
        "action": {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {"no_op": None, "isolate": None},
        }
    }
    response = {
        "model": "jev-1.13.0",
        "answers": {
            "action": {
                "type": "choice",
                "choice": "no_op",
                "probabilities": {"no_op": 0.49, "isolate": 0.5},
                "confidence": 0.5,
            }
        },
        "usage": {"input_tokens": 10},
    }
    client = backend(
        tmp_path, transport(lambda _: httpx.Response(200, json=response)), []
    )
    client.decide(state(), spec)
    stored = json.loads((tmp_path / "cache.jsonl").read_text().splitlines()[0])
    assert sum(
        stored["normalized_response"]["answers"]["action"]["probabilities"].values()
    ) == pytest.approx(1.0)
    assert stored["raw_response"]["answers"]["action"]["probabilities"] == {
        "no_op": 0.49,
        "isolate": 0.5,
    }
    assert stored["normalization_events"][0]["raw_sum"] == pytest.approx(0.99)


@pytest.mark.unit
def test_retries_429_and_honors_retry_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-placeholder")
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json=valid_response())

    sleeps: list[float] = []
    backend(tmp_path, transport(handler), sleeps).decide(state(), SPEC)
    assert attempts == 2
    assert sleeps == [0.0]


@pytest.mark.unit
def test_version_mismatch_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-placeholder")
    mock = transport(lambda _: httpx.Response(200, json=valid_response("jev-1.14.0")))
    with pytest.raises(VersionMismatch):
        backend(tmp_path, mock, []).decide(state(), SPEC)


@pytest.mark.unit
def test_malformed_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-placeholder")
    mock = transport(lambda _: httpx.Response(200, content=b"not-json"))
    with pytest.raises(RuntimeError, match="malformed"):
        backend(tmp_path, mock, []).decide(state(), SPEC)


@pytest.mark.unit
def test_timeout_degrades_without_leaking_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-placeholder")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = backend(tmp_path, transport(handler), [])
    with pytest.raises(DecisionUnavailable, match="failed after retries") as captured:
        client.decide(state(), SPEC)
    assert "placeholder" not in str(captured.value)


@pytest.mark.unit
def test_missing_key_and_open_circuit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = backend(tmp_path, transport(lambda _: httpx.Response(500)), [])
    with pytest.raises(DecisionUnavailable, match="not set"):
        client.decide(state(), SPEC)
    client.consecutive_failures = 3
    with pytest.raises(DecisionUnavailable, match="circuit"):
        client.decide(state(), SPEC)


@pytest.mark.unit
def test_choice_and_score_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-placeholder")
    spec = {
        "action": {
            "type": "choice",
            "instructions": "Choose an action",
            "criteria": {"noop": None, "isolate": None},
        },
        "severity": {
            "type": "score",
            "instructions": "Score severity",
            "criteria": ["low", "high"],
        },
    }
    response = {
        "model": "jev-1.13.0",
        "answers": {
            "action": {
                "type": "choice",
                "choice": "noop",
                "probabilities": {"noop": 0.8, "isolate": 0.2},
                "confidence": 0.6,
            },
            "severity": {
                "type": "score",
                "score": 0.25,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.75, "1": 0.25},
                "confidence": 0.5,
            },
        },
        "usage": {"input_tokens": 50, "output_tokens": 20},
    }
    mock = transport(lambda _: httpx.Response(200, json=response))
    result = backend(tmp_path, mock, []).decide(state(), spec)
    assert result.answers["action"].selected == "noop"
    assert result.answers["severity"].selected == 0.25


@pytest.mark.unit
def test_budget_and_question_guards() -> None:
    budget = BudgetGuard(1, 0.000001, 1.0)
    budget.reserve_call()
    with pytest.raises(BudgetExceeded):
        budget.reserve_call()
    with pytest.raises(BudgetExceeded):
        budget.record_usage(2)
    with pytest.raises(ValueError, match="typed"):
        validate_spec({"text": {"type": "string", "instructions": "write"}})
    with pytest.raises(ValueError, match="required"):
        validate_spec({"known": {"type": "noul"}})
    with pytest.raises(ValueError, match="choice criteria"):
        validate_spec({"choice": {"type": "choice", "instructions": "pick"}})


@pytest.mark.unit
def test_local_backend_and_calibration() -> None:
    frame = generate_synthetic_events(200, 12)
    pipeline = FeaturePipeline.create().fit(frame)
    transformed = pipeline.transform(frame)
    local = LocalBackend().fit(
        transformed.loc[:, [f"feature_{index:02d}" for index in range(40)]].to_numpy(),
        transformed["label"].to_numpy(),
    )
    result = local.decide(serialize_features(transformed.iloc[0]), SPEC)
    probability = result.answers["known"].probabilities["true"]
    assert 0 <= probability <= 1
    labels = np.array([0, 1])
    probabilities = np.array([0.1, 0.9])
    assert brier_score(labels, probabilities) == pytest.approx(0.01)
    assert expected_calibration_error(labels, probabilities, bins=2) == pytest.approx(
        0.1
    )
    with pytest.raises(ValueError, match="positive"):
        expected_calibration_error(labels, probabilities, bins=0)
    with pytest.raises(ValueError, match="empty"):
        expected_calibration_error(np.array([]), np.array([]))


@pytest.mark.unit
def test_unfitted_local_backend_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        LocalBackend().decide(state(), SPEC)
