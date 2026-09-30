from __future__ import annotations

from dataclasses import dataclass

from sentinel.cyberbody.actions import DefensiveAction
from sentinel.jev.interface import (
    DecisionModel,
    DecisionSpec,
    DecisionUnavailable,
    JsonValue,
)

REFLEX_SPEC: DecisionSpec = {
    "known_pattern": {
        "type": "noul",
        "instructions": "Is this telemetry consistent with a known threat pattern?",
        "criteria": {
            "true": "Known threat pattern",
            "false": "Benign, uncertain, or novel pattern",
        },
    },
    "action": {
        "type": "choice",
        "instructions": "Which closed-set simulated defensive action is appropriate?",
        "criteria": {
            "no_op": "Take no simulated action",
            "isolate_host": "Isolate the simulated host",
            "block_flow": "Block the simulated flow",
            "rate_limit": "Rate limit the simulated host",
            "revoke_session": "Revoke the simulated user session",
        },
    },
}


@dataclass(frozen=True)
class ReflexDecision:
    action: DefensiveAction | None
    ascend: bool
    available: bool
    confidence: float
    reason: str


def evaluate_reflex(
    model: DecisionModel,
    state: dict[str, JsonValue],
    tau_reflex: float,
) -> ReflexDecision:
    if not 0 <= tau_reflex <= 1:
        raise ValueError("reflex threshold must be between zero and one")
    try:
        result = model.decide(state, REFLEX_SPEC)
    except (DecisionUnavailable, KeyError, ValueError) as error:
        return ReflexDecision(None, True, False, 0.0, type(error).__name__)
    known = result.answers["known_pattern"].probabilities["true"]
    known_confidence = abs(2 * known - 1)
    action_answer = result.answers["action"]
    action_confidence = action_answer.confidence
    if action_confidence is None:
        return ReflexDecision(None, True, True, 0.0, "missing_action_confidence")
    confidence = min(known_confidence, action_confidence)
    selected = action_answer.selected
    if known >= 0.5 and confidence >= tau_reflex and isinstance(selected, str):
        try:
            action = DefensiveAction(selected)
        except ValueError:
            return ReflexDecision(None, True, True, confidence, "unknown_action")
        return ReflexDecision(action, False, True, confidence, "known_high_confidence")
    return ReflexDecision(None, True, True, confidence, "uncertain_or_novel")
