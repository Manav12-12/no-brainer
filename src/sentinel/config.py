from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class JevConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "replay"
    model_id: str = "jev-1.13.0"
    base_url: str = "https://api.typesafe.ai"
    endpoint: str = "/v1/systemone"
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    backoff_initial_seconds: float = Field(default=0.5, ge=0)
    backoff_max_seconds: float = Field(default=5.0, ge=0)
    backoff_jitter_fraction: float = Field(default=0.25, ge=0, le=1)
    max_calls: int = Field(default=1000, ge=0)
    estimated_cost_usd_ceiling: float = Field(default=1.0, ge=0)
    input_usd_per_million_tokens: float = Field(default=0.042, ge=0)
    cache_path: Path = Path("data/jev_cache/responses.jsonl")
    serialization_version: str = "sentinel-features-v1"

    def model_post_init(self, __context: Any) -> None:
        if self.model_id in {"jev-latest", "jev-preview"}:
            raise ValueError("Jev model must be an explicit versioned ID")


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a mapping in {path}")
    return loaded


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()
