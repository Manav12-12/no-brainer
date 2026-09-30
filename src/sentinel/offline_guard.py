from __future__ import annotations

import ast
from pathlib import Path

NETWORK_MODULES = {"httpx", "requests", "urllib3", "socket", "aiohttp"}
ALLOWED_NETWORK_FILE = Path("src/sentinel/jev/typesafe_backend.py")


def unauthorized_network_imports(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted((root / "src").rglob("*.py")):
        relative = path.relative_to(root)
        if relative == ALLOWED_NETWORK_FILE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in NETWORK_MODULES:
                    findings.append(f"{relative}:{getattr(node, 'lineno', 0)}:{name}")
    return findings
