from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sentinel.config import JevConfig, load_yaml
from sentinel.jev.budget import BudgetGuard
from sentinel.jev.cache import DecisionCache
from sentinel.jev.interface import DecisionSpec, JsonValue
from sentinel.jev.typesafe_backend import TypeSafeBackend
from sentinel.offline_guard import unauthorized_network_imports

app = typer.Typer(no_args_is_help=True)


@app.command()
def smoke() -> None:
    """Validate configuration and the static network boundary."""
    root = Path(__file__).resolve().parents[2]
    JevConfig.model_validate(load_yaml(root / "configs/jev.yaml"))
    findings = unauthorized_network_imports(root)
    if findings:
        typer.echo(f"unauthorized network imports: {findings}", err=True)
        raise typer.Exit(code=2)
    typer.echo("smoke test passed")


@app.command("jev-check")
def jev_check(
    confirm_live: Annotated[
        bool,
        typer.Option(
            "--confirm-live",
            help="Authorize one live request to api.typesafe.ai.",
        ),
    ] = False,
) -> None:
    """Run one explicitly authorized live contract call."""
    if not confirm_live:
        typer.echo("refusing live call without --confirm-live", err=True)
        raise typer.Exit(code=2)

    root = Path(__file__).resolve().parents[2]
    config = JevConfig.model_validate(load_yaml(root / "configs/jev.yaml"))
    state: dict[str, JsonValue] = {
        "serialization_version": config.serialization_version,
        "features": [0.0] * 40,
        "context": {
            "host_role": "workstation",
            "segment": "user",
            "recent_event_count": 0,
        },
    }
    spec: DecisionSpec = {
        "known_pattern": {
            "type": "noul",
            "instructions": (
                "Is this synthetic all-zero contract-check state a known pattern?"
            ),
        }
    }
    backend = TypeSafeBackend(
        model_id=config.model_id,
        base_url=config.base_url,
        endpoint=config.endpoint,
        cache=DecisionCache(root / config.cache_path, read_enabled=False),
        budget=BudgetGuard(
            max_calls=1,
            cost_ceiling_usd=config.estimated_cost_usd_ceiling,
            input_usd_per_million_tokens=config.input_usd_per_million_tokens,
        ),
        serialization_version=config.serialization_version,
        timeout_seconds=config.timeout_seconds,
        max_retries=0,
        backoff_initial_seconds=config.backoff_initial_seconds,
        backoff_max_seconds=config.backoff_max_seconds,
        jitter_fraction=config.backoff_jitter_fraction,
    )
    result = backend.decide(state, spec)
    typer.echo(
        f"Jev contract check passed: model={result.model_version} "
        f"input_tokens={result.usage_input_tokens} latency_ms={result.latency_ms:.1f}"
    )


if __name__ == "__main__":
    app()
