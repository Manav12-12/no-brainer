from __future__ import annotations

import copy
import logging
import math
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast

import httpx

from sentinel.egress_allowlist import require_allowed_url
from sentinel.jev.budget import BudgetGuard
from sentinel.jev.cache import DecisionCache, cache_key
from sentinel.jev.interface import (
    DecisionResult,
    DecisionSpec,
    DecisionUnavailable,
    JsonValue,
    QuestionResult,
    VersionMismatch,
    validate_spec,
)
from sentinel.jev.payload_guard import guard_payload

RETRYABLE_STATUS = {408, 429, *range(500, 600)}
PROBABILITY_SCHEMA_TOLERANCE = 1e-6
PROBABILITY_NORMALIZATION_TOLERANCE = 0.02
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedResponse:
    result: DecisionResult
    normalized_response: dict[str, Any]
    normalization_events: list[dict[str, Any]]


def normalize_probabilities(
    probabilities: dict[str, Any], question_name: str
) -> tuple[dict[str, float], dict[str, Any] | None]:
    if not probabilities:
        raise DecisionUnavailable("Jev answer has an empty probability distribution")
    converted: dict[str, float] = {}
    for name, value in probabilities.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DecisionUnavailable("Jev answer has a non-numeric probability")
        converted[name] = float(value)
    raw_sum = sum(converted.values())
    if not math.isfinite(raw_sum):
        raise DecisionUnavailable("Jev probability sum is not finite")
    if math.isclose(raw_sum, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SCHEMA_TOLERANCE):
        return converted, None
    lower = 1.0 - PROBABILITY_NORMALIZATION_TOLERANCE
    upper = 1.0 + PROBABILITY_NORMALIZATION_TOLERANCE
    inclusive_lower = math.nextafter(lower, -math.inf)
    inclusive_upper = math.nextafter(upper, math.inf)
    if not inclusive_lower <= raw_sum <= inclusive_upper or raw_sum == 0:
        raise DecisionUnavailable(
            f"Jev probability sum {raw_sum:.12g} is outside normalization tolerance"
        )
    correction_factor = 1.0 / raw_sum
    normalized = {name: value * correction_factor for name, value in converted.items()}
    event: dict[str, Any] = {
        "question": question_name,
        "raw_sum": raw_sum,
        "correction_factor": correction_factor,
        "normalized_sum": sum(normalized.values()),
    }
    LOGGER.info(
        "normalized Jev probabilities question=%s raw_sum=%.12g "
        "correction_factor=%.12g",
        question_name,
        raw_sum,
        correction_factor,
    )
    return normalized, event


class TypeSafeBackend:
    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        endpoint: str,
        cache: DecisionCache,
        budget: BudgetGuard,
        serialization_version: str,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        backoff_initial_seconds: float = 0.5,
        backoff_max_seconds: float = 5.0,
        jitter_fraction: float = 0.25,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        if model_id in {"jev-latest", "jev-preview"}:
            raise ValueError("moving Jev aliases are forbidden")
        self.model_id = model_id
        self.url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        require_allowed_url(self.url)
        self.cache = cache
        self.budget = budget
        self.serialization_version = serialization_version
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_initial_seconds = backoff_initial_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.jitter_fraction = jitter_fraction
        self.transport = transport
        self.sleep = sleep
        self.consecutive_failures = 0
        self.circuit_open_after = 3

    def decide(self, state: dict[str, JsonValue], spec: DecisionSpec) -> DecisionResult:
        guard_payload(state)
        validate_spec(spec)
        key = cache_key(state, spec, self.model_id, self.serialization_version)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.consecutive_failures >= self.circuit_open_after:
            raise DecisionUnavailable("Jev circuit breaker is open")
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise DecisionUnavailable("TYPESAFE_API_KEY is not set")
        self.budget.reserve_call()
        request: dict[str, object] = {
            "state": state,
            "model": self.model_id,
            "questions": spec,
        }
        started = time.perf_counter()
        try:
            payload = self._post(request, api_key)
            latency_ms = (time.perf_counter() - started) * 1000
            parsed = self._parse(payload, spec, latency_ms)
            result = parsed.result
            usage = result.usage_input_tokens or 0
            self.budget.record_usage(usage)
            self.cache.append(
                key,
                result,
                datetime.now(UTC).isoformat(),
                raw_response=payload,
                normalized_response=parsed.normalized_response,
                normalization_events=parsed.normalization_events,
            )
            self.consecutive_failures = 0
            return result
        except (DecisionUnavailable, httpx.HTTPError):
            self.consecutive_failures += 1
            raise

    def _post(self, request: dict[str, object], api_key: str) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(
            transport=self.transport,
            timeout=self.timeout_seconds,
            follow_redirects=False,
        ) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(self.url, headers=headers, json=request)
                except (httpx.TimeoutException, httpx.NetworkError) as error:
                    if attempt >= self.max_retries:
                        raise DecisionUnavailable(
                            "Jev request failed after retries"
                        ) from error
                    self.sleep(self._backoff(attempt, None))
                    continue
                if (
                    response.status_code in RETRYABLE_STATUS
                    and attempt < self.max_retries
                ):
                    self.sleep(self._backoff(attempt, response.headers))
                    continue
                if response.status_code >= 400:
                    raise DecisionUnavailable(
                        f"Jev returned HTTP {response.status_code}"
                    )
                try:
                    value = response.json()
                except ValueError as error:
                    raise DecisionUnavailable("Jev returned malformed JSON") from error
                if not isinstance(value, dict):
                    raise DecisionUnavailable("Jev response must be an object")
                return value
        raise DecisionUnavailable("Jev request loop ended unexpectedly")

    def _backoff(self, attempt: int, headers: httpx.Headers | None) -> float:
        if headers is not None:
            milliseconds = headers.get("retry-after-ms")
            if milliseconds is not None:
                try:
                    return float(max(0.0, float(milliseconds) / 1000))
                except ValueError:
                    pass
            retry_after = headers.get("retry-after")
            if retry_after is not None:
                try:
                    return float(max(0.0, float(retry_after)))
                except ValueError:
                    try:
                        parsed = parsedate_to_datetime(retry_after)
                        return float(
                            max(
                                0.0,
                                (parsed - datetime.now(UTC)).total_seconds(),
                            )
                        )
                    except (TypeError, ValueError):
                        pass
        base = min(self.backoff_max_seconds, self.backoff_initial_seconds * 2**attempt)
        jitter = secrets.randbelow(1_000_001) / 1_000_000
        return float(base * (1 - jitter * self.jitter_fraction))

    def _parse(
        self, payload: dict[str, Any], spec: DecisionSpec, latency_ms: float
    ) -> ParsedResponse:
        returned_model = payload.get("model")
        if returned_model != self.model_id:
            message = (
                f"returned model {returned_model!r} differs from pinned model "
                f"{self.model_id!r}"
            )
            raise VersionMismatch(message)
        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, dict) or set(raw_answers) != set(spec):
            raise DecisionUnavailable("Jev answers do not match requested questions")
        normalized_response = copy.deepcopy(payload)
        normalized_answers = normalized_response["answers"]
        if not isinstance(normalized_answers, dict):
            raise DecisionUnavailable("Jev normalized answers must be a mapping")
        normalization_events: list[dict[str, Any]] = []
        answers: dict[str, QuestionResult] = {}
        for name, question in spec.items():
            raw = raw_answers[name]
            if not isinstance(raw, dict) or raw.get("type") != question["type"]:
                raise DecisionUnavailable("Jev answer type mismatch")
            normalized_raw = normalized_answers[name]
            if not isinstance(normalized_raw, dict):
                raise DecisionUnavailable("Jev normalized answer must be a mapping")
            if raw.get("type") != "noul":
                probabilities = raw.get("probabilities")
                if not isinstance(probabilities, dict):
                    raise DecisionUnavailable("Jev answer lacks probabilities")
                normalized, event = normalize_probabilities(probabilities, name)
                normalized_raw["probabilities"] = normalized
                if event is not None:
                    normalization_events.append(event)
            answers[name] = self._parse_answer(normalized_raw)
        usage = payload.get("usage")
        tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        return ParsedResponse(
            result=DecisionResult(
                answers=answers,
                model_version=cast(str, returned_model),
                latency_ms=latency_ms,
                usage_input_tokens=tokens,
            ),
            normalized_response=normalized_response,
            normalization_events=normalization_events,
        )

    @staticmethod
    def _parse_answer(raw: dict[str, Any]) -> QuestionResult:
        answer_type = raw["type"]
        if answer_type == "noul":
            value = float(raw["noul"])
            return QuestionResult(
                type="noul",
                selected=value,
                probabilities={"false": 1.0 - value, "true": value},
            )
        probabilities = raw.get("probabilities")
        if not isinstance(probabilities, dict):
            raise DecisionUnavailable("Jev answer lacks probabilities")
        selected = raw.get("choice") if answer_type == "choice" else raw.get("score")
        if not isinstance(selected, str | int | float) or isinstance(selected, bool):
            raise DecisionUnavailable("Jev answer lacks a typed selection")
        return QuestionResult(
            type=cast(Any, answer_type),
            selected=float(selected) if isinstance(selected, int | float) else selected,
            probabilities=probabilities,
            confidence=raw.get("confidence"),
        )
