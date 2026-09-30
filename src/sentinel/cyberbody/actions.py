from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sentinel.cyberbody.topology import Host


class DefensiveAction(str, Enum):
    NO_OP = "no_op"
    ISOLATE_HOST = "isolate_host"
    BLOCK_FLOW = "block_flow"
    RATE_LIMIT = "rate_limit"
    REVOKE_SESSION = "revoke_session"


@dataclass(frozen=True)
class ActionOutcome:
    cost: float
    injury: int
    changed: bool


_COST = {
    DefensiveAction.NO_OP: 0.0,
    DefensiveAction.ISOLATE_HOST: 5.0,
    DefensiveAction.BLOCK_FLOW: 2.0,
    DefensiveAction.RATE_LIMIT: 1.0,
    DefensiveAction.REVOKE_SESSION: 2.0,
}


def apply_action(action: DefensiveAction, host: Host) -> ActionOutcome:
    injury = int(action != DefensiveAction.NO_OP and not host.compromised)
    changed = False
    if action in {DefensiveAction.ISOLATE_HOST, DefensiveAction.BLOCK_FLOW}:
        changed = not host.isolated
        host.isolated = True
    elif action == DefensiveAction.RATE_LIMIT:
        changed = not host.rate_limited
        host.rate_limited = True
    elif action == DefensiveAction.REVOKE_SESSION:
        changed = host.session_active
        host.session_active = False
    return ActionOutcome(_COST[action], injury, changed)
