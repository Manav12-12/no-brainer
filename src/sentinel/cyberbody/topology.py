from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Host:
    host_id: str
    segment: str
    role: str
    compromised: bool = False
    isolated: bool = False
    session_active: bool = True
    rate_limited: bool = False


def build_topology(host_count: int = 12) -> dict[str, Host]:
    if not 10 <= host_count <= 30:
        raise ValueError("cyber range supports 10 through 30 hosts")
    segments = ("user", "server", "dmz")
    roles = ("workstation", "server", "gateway")
    return {
        f"host-{index:02d}": Host(
            host_id=f"host-{index:02d}",
            segment=segments[index % len(segments)],
            role=roles[index % len(roles)],
        )
        for index in range(host_count)
    }
