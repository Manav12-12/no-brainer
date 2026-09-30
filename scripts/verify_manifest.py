from __future__ import annotations

import hashlib
from pathlib import Path


def verify_manifest(manifest: Path) -> None:
    if not manifest.exists():
        return
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        target = manifest.parent / relative.lstrip("* ")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"hash mismatch: {target}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for item in (
        root / "data/connectome/MANIFEST.sha256",
        root / "data/connectome/full/MANIFEST.sha256",
        root / "data/public/unsw_nb15_v3/MANIFEST.sha256",
        root / "data/jev_cache/MANIFEST.sha256",
        root / "wheelhouse/MANIFEST.sha256",
    ):
        verify_manifest(item)
