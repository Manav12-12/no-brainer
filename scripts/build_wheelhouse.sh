#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cat >&2 <<'EOF'
Wheel downloads are restricted to scripts/fetch_offline_assets.sh.
Run that script with --runtime-only while network access is explicitly allowed.
EOF
exec "$ROOT/scripts/fetch_offline_assets.sh" --runtime-only
