from __future__ import annotations


def adjust_reflex_threshold(
    current: float,
    novelty_score: float,
    false_action_rate: float,
    step_size: float = 0.05,
) -> float:
    if not all(
        0 <= value <= 1 for value in (current, novelty_score, false_action_rate)
    ):
        raise ValueError("descending inputs must be probabilities")
    adjustment = step_size * (false_action_rate + novelty_score - 0.5)
    return min(0.99, max(0.5, current + adjustment))
