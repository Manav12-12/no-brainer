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
    drive_actions = [event for event in timeline if event["action_source"] == "drive"]
    assert drive_actions
    action = drive_actions[0]
    assert action["drive"]["before_action"] >= action["drive"]["threshold"]
    assert action["drive"]["current"] < action["drive"]["before_action"]
    following = timeline[action["step"] + 1]
    assert following["stage"] == "contained"
    assert following["drive"]["current"] < action["drive"]["current"]
