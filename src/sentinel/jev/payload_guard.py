from __future__ import annotations

import math

from sentinel.data.features import FEATURE_NAMES, FEATURE_VERSION
from sentinel.jev.interface import JsonValue

_ROLES = {"workstation", "server", "gateway"}
_SEGMENTS = {"user", "server", "dmz"}


def guard_payload(state: dict[str, JsonValue]) -> None:
    if set(state) != {"serialization_version", "features", "context"}:
        raise ValueError("state must come from the versioned feature serializer")
    if state["serialization_version"] != FEATURE_VERSION:
        raise ValueError("unsupported feature serialization version")
    features = state["features"]
    if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
        raise ValueError("state must contain exactly 40 features")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in features
    ):
        raise ValueError("features must be finite numbers")
    context_value = state["context"]
    if not isinstance(context_value, dict):
        raise ValueError("context must be a mapping")
    context = context_value
    if set(context) != {"host_role", "segment", "recent_event_count"}:
        raise ValueError("unexpected context fields")
    if context["host_role"] not in _ROLES or context["segment"] not in _SEGMENTS:
        raise ValueError("context contains an unknown categorical value")
    count = context["recent_event_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= 10_000
    ):
        raise ValueError("recent event count is invalid")
