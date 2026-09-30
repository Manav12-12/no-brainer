from __future__ import annotations

import json
from pathlib import Path

from sentinel.config import JevConfig, load_yaml
from sentinel.cyberbody.env import CyberRange
from sentinel.data.features import serialize_vector
from sentinel.jev.cache import DecisionCache, cache_key
from sentinel.orchestrator.reflex_arc import REFLEX_SPEC


def planned_keys(root: Path, config: JevConfig) -> set[str]:
    experiment = load_yaml(root / "configs/experiments.yaml")
    cyber = load_yaml(root / "configs/cyberrange.yaml")
    keys: set[str] = set()
    for seed in experiment["seeds"]:
        environment = CyberRange(
            int(cyber["hosts"]), int(cyber["episode_steps"]), int(seed)
        )
        for _ in range(environment.episode_steps):
            observation = environment.observe()
            state = serialize_vector(
                observation.features,
                host_role=observation.host_role,
                segment=observation.segment,
                recent_event_count=observation.step,
            )
            keys.add(
                cache_key(
                    state,
                    REFLEX_SPEC,
                    config.model_id,
                    config.serialization_version,
                )
            )
    return keys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = JevConfig.model_validate(load_yaml(root / "configs/jev.yaml"))
    keys = planned_keys(root, config)
    cache = DecisionCache(root / config.cache_path)
    hits = sum(cache.get(key) is not None for key in keys)
    misses = len(keys) - hits
    output = {
        "pinned_model": config.model_id,
        "unique_requests": len(keys),
        "cache_hits": hits,
        "live_calls_required": misses,
        "documented_input_usd_per_million_tokens": config.input_usd_per_million_tokens,
        "estimated_cost_usd": None,
        "cost_note": (
            "Unknown before recording because the vendor docs provide no local "
            "tokenizer "
            "or preflight token-count endpoint. The configured hard ceiling is shown."
        ),
        "hard_cost_ceiling_usd": config.estimated_cost_usd_ceiling,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
