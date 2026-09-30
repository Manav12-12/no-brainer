from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sentinel.jev.interface import DecisionResult, DecisionSpec, JsonValue


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_key(
    state: dict[str, JsonValue],
    spec: DecisionSpec,
    model_id: str,
    serialization_version: str,
) -> str:
    source = canonical_json(
        {
            "state": state,
            "questions": spec,
            "model": model_id,
            "serialization_version": serialization_version,
        }
    )
    return hashlib.sha256(source.encode()).hexdigest()


class DecisionCache:
    def __init__(self, path: Path, *, read_enabled: bool = True) -> None:
        self.path = path
        self.read_enabled = read_enabled

    def get(self, key: str) -> DecisionResult | None:
        if not self.read_enabled or not self.path.exists():
            return None
        found: DecisionResult | None = None
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("key") == key:
                    found = DecisionResult.model_validate(record["result"])
        return found

    def append(
        self,
        key: str,
        result: DecisionResult,
        timestamp: str,
        raw_response: dict[str, Any] | None = None,
        normalized_response: dict[str, Any] | None = None,
        normalization_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "key": key,
            "timestamp": timestamp,
            "returned_version": result.model_version,
            "latency_ms": result.latency_ms,
            "result": result.model_dump(mode="json"),
            "raw_response": raw_response,
            "normalized_response": normalized_response,
            "normalization_events": normalization_events or [],
        }
        line = canonical_json(record) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
