from __future__ import annotations

from sentinel.jev.cache import DecisionCache, cache_key
from sentinel.jev.interface import (
    DecisionResult,
    DecisionSpec,
    JsonValue,
    validate_spec,
)
from sentinel.jev.payload_guard import guard_payload


class ReplayBackend:
    def __init__(
        self, cache: DecisionCache, model_id: str, serialization_version: str
    ) -> None:
        self.cache = cache
        self.model_id = model_id
        self.serialization_version = serialization_version

    def decide(self, state: dict[str, JsonValue], spec: DecisionSpec) -> DecisionResult:
        guard_payload(state)
        validate_spec(spec)
        key = cache_key(state, spec, self.model_id, self.serialization_version)
        result = self.cache.get(key)
        if result is None:
            raise KeyError(f"Jev replay cache miss: {key}")
        if result.model_version != self.model_id:
            raise ValueError("cached response model does not match pinned version")
        return result
