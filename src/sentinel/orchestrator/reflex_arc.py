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
    pain_signal: float
    available: bool
    confidence: float
    reason: str


def evaluate_reflex(
    model: DecisionModel,
    state: dict[str, JsonValue],
    tau_reflex: float,
    known_pattern_threshold: float = 0.5,
) -> ReflexDecision:
    if not 0 <= tau_reflex <= 1 or not 0 <= known_pattern_threshold <= 1:
        raise ValueError("reflex thresholds must be between zero and one")
    try:
        result = model.decide(state, REFLEX_SPEC)
    except (DecisionUnavailable, KeyError, ValueError) as error:
        # The upstream channel remains explicit on backend failure. Zero is a
        # missing-safe signal, not a synthetic classification or network fallback.
        return ReflexDecision(None, 0.0, False, 0.0, type(error).__name__)
    known = result.answers["known_pattern"].probabilities["true"]
    known_confidence = abs(2 * known - 1)
    action_answer = result.answers["action"]
    action_confidence = action_answer.confidence
    if action_confidence is None:
        return ReflexDecision(None, 0.0, True, 0.0, "missing_action_confidence")
    # Validation showed that low known-pattern probability carries novelty,
    # while action confidence carries salience. The former average of `known`
    # and non-no-op mass inverted the observed attack ranking (ROC-AUC 0.373).
    pain_signal = min(1.0, max(0.0, (action_confidence + 1.0 - known) / 2.0))
    confidence = min(known_confidence, action_confidence)
    selected = action_answer.selected
    # Jev 1.13.0 emitted known-pattern probabilities in [0.20, 0.38] on the
    # cached UNSW sample. Keep this boundary explicit and externally tuned;
    # 0.5 is not a model-independent semantic cutoff for a `noul` response.
    if (
        known >= known_pattern_threshold
        and confidence >= tau_reflex
        and isinstance(selected, str)
        and selected != DefensiveAction.NO_OP.value
    ):
        try:
            action = DefensiveAction(selected)
        except ValueError:
            return ReflexDecision(None, pain_signal, True, confidence, "unknown_action")
        return ReflexDecision(
            action, pain_signal, True, confidence, "known_high_confidence"
        )
    return ReflexDecision(None, pain_signal, True, confidence, "no_immediate_reflex")
