from __future__ import annotations

from collections.abc import Callable

import httpx


def valid_response(model: str = "jev-1.13.0") -> dict[str, object]:
    return {
        "model": model,
        "answers": {"known": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)
