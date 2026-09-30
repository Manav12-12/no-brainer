from __future__ import annotations

from urllib.parse import urlparse

ALLOWED_ORIGIN = "https://api.typesafe.ai"


def require_allowed_url(url: str) -> None:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin != ALLOWED_ORIGIN or parsed.username or parsed.password:
        host = parsed.hostname or "<invalid>"
        raise PermissionError(f"egress refused for host: {host}")
