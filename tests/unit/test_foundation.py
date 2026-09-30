import logging
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sentinel.cli import app
from sentinel.config import JevConfig, config_hash, load_yaml
from sentinel.egress_allowlist import require_allowed_url
from sentinel.jev.interface import DecisionResult, QuestionResult
from sentinel.logging_utils import RedactingFilter, configure_logging, redact
from sentinel.offline_guard import unauthorized_network_imports
from sentinel.seed import set_global_seed


@pytest.mark.unit
def test_moving_model_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="explicit versioned"):
        JevConfig(model_id="jev-latest")


@pytest.mark.unit
def test_config_hash_ignores_dict_order() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


@pytest.mark.audit
def test_egress_allowlist() -> None:
    require_allowed_url("https://api.typesafe.ai/v1/systemone")
    with pytest.raises(PermissionError):
        require_allowed_url("https://example.com/v1/systemone")
    with pytest.raises(PermissionError):
        require_allowed_url("http://api.typesafe.ai/v1/systemone")


@pytest.mark.audit
def test_no_unauthorized_network_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    assert unauthorized_network_imports(root) == []


@pytest.mark.unit
def test_secret_redaction() -> None:
    assert "secret" not in redact("Authorization: Bearer secret")
    assert "value" not in redact("apikey_value")


@pytest.mark.unit
def test_yaml_loading_rejects_scalar(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text("a: 1\n", encoding="utf-8")
    assert load_yaml(valid) == {"a": 1}
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_yaml(invalid)


@pytest.mark.unit
def test_logging_filter_redacts_message() -> None:
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "Bearer secret", (), None
    )
    assert RedactingFilter().filter(record)
    assert "secret" not in record.msg
    configure_logging()


@pytest.mark.unit
def test_seed_is_repeatable() -> None:
    import random

    import numpy as np

    set_global_seed(42)
    first = (random.random(), np.random.random())  # noqa: S311
    set_global_seed(42)
    assert first == (random.random(), np.random.random())  # noqa: S311


@pytest.mark.unit
def test_cli_commands() -> None:
    runner = CliRunner()
    smoke = runner.invoke(app, ["smoke"])
    assert smoke.exit_code == 0
    assert "passed" in smoke.stdout
    check = runner.invoke(app, ["jev-check"])
    assert check.exit_code == 2
    assert "--confirm-live" in check.output


@pytest.mark.unit
def test_jev_check_is_one_uncached_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    captured: dict[str, Any] = {}

    class FakeBackend:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def decide(self, state: object, spec: object) -> DecisionResult:
            nonlocal calls
            calls += 1
            return DecisionResult(
                answers={
                    "known_pattern": QuestionResult(
                        type="noul",
                        selected=0.5,
                        probabilities={"false": 0.5, "true": 0.5},
                    )
                },
                model_version="jev-1.13.0",
                latency_ms=1.0,
                usage_input_tokens=10,
            )

    monkeypatch.setattr("sentinel.cli.TypeSafeBackend", FakeBackend)
    result = CliRunner().invoke(app, ["jev-check", "--confirm-live"])
    assert result.exit_code == 0
    assert calls == 1
    assert captured["max_retries"] == 0
    assert captured["budget"].max_calls == 1
    assert captured["cache"].read_enabled is False


@pytest.mark.audit
def test_network_import_finding(tmp_path: Path) -> None:
    source = tmp_path / "src/example"
    source.mkdir(parents=True)
    (source / "bad.py").write_text("import socket\n", encoding="utf-8")
    assert unauthorized_network_imports(tmp_path) == ["src/example/bad.py:1:socket"]
