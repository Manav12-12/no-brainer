import pytest

from scripts.run_live_simulation import simulation_config, simulation_events


@pytest.mark.integration
def test_live_stream_uses_simulated_spikes_and_actions() -> None:
    config = simulation_config(1729)
    timeline = [
        item for item in simulation_events() if item["type"] == "simulation_event"
    ]
    assert config["synthetic_mode"] is True
    assert len(config["graph"]["nodes"]) == 48
    assert len(timeline) == config["total_steps"] == 24
    assert any(event["action"] != "no_op" for event in timeline)
    assert sum((event["brain"] or {}).get("total_spikes", 0) for event in timeline) > 0
    assert all("pain_signal" in event["reflex"] for event in timeline)
    reflex_actions = [event for event in timeline if event["action_source"] == "reflex"]
    assert reflex_actions
    assert all(event["reflex"]["pain_signal"] > 0 for event in reflex_actions)
    assert all(event["brain"]["total_spikes"] > 0 for event in reflex_actions)
