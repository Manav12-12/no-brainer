from __future__ import annotations

from dataclasses import dataclass

from sentinel.cyberbody.actions import DefensiveAction
from sentinel.cyberbody.topology import Host


@dataclass(frozen=True)
class ActionDirective:
    host_id: str
    action: DefensiveAction


@dataclass(frozen=True)
class ResponseStrategy:
    name: str
    directives: tuple[ActionDirective, ...]
    reason: str


def select_response_strategy(
    trigger_host_id: str,
    hosts: dict[str, Host],
    drive_values: dict[str, float],
    threshold: float,
    reflex_action: DefensiveAction | None,
) -> ResponseStrategy:
    """Choose a response using sustained and cross-host pressure."""
    trigger = hosts[trigger_host_id]
    correlated = [
        host
        for host in hosts.values()
        if host.segment == trigger.segment
        and not host.isolated
        and drive_values.get(host.host_id, 0.0) >= 0.6 * threshold
    ]
    if len(correlated) >= 2:
        directives = tuple(
            ActionDirective(
                host.host_id,
                DefensiveAction.ISOLATE_HOST
                if host.host_id == trigger_host_id
                else DefensiveAction.RATE_LIMIT,
            )
            for host in correlated
        )
        return ResponseStrategy(
            "segment_coordination",
            directives,
            "correlated sustained pain in one segment",
        )
    if reflex_action in {
        DefensiveAction.RATE_LIMIT,
        DefensiveAction.BLOCK_FLOW,
        DefensiveAction.REVOKE_SESSION,
    }:
        return ResponseStrategy(
            "escalate_insufficient_reflex",
            (ActionDirective(trigger_host_id, DefensiveAction.ISOLATE_HOST),),
            "pain persisted after a narrower reflex",
        )
    return ResponseStrategy(
        "sustained_host_containment",
        (ActionDirective(trigger_host_id, DefensiveAction.ISOLATE_HOST),),
        "host drive crossed the sustained-pain threshold",
    )
