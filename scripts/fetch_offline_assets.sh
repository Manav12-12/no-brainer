#!/usr/bin/env bash
set -euo pipefail

# This is the project's only unauthenticated download entry point. It is meant
# to be run once by a human before switching to the offline workflow.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

download() {
  local url="$1"
  local output="$2"
  curl --fail --location --proto '=https' --tlsv1.2 \
    --retry 3 --retry-all-errors --output "$output" "$url"
}

if [[ "${1:-}" == "--dependency-audit" ]]; then
  mkdir -p "$ROOT/artifacts"
  # 2026-09-30: PYSEC-2026-113 explicitly excludes Python bindings. The
  # affected C++ PreBufferMetadata API is not exposed to this project.
  "$ROOT/.venv/bin/python" -m pip_audit --local --format json \
    --ignore-vuln PYSEC-2026-113 \
    --output "$ROOT/artifacts/dependency-audit.json"
  exit 0
fi

if [[ "${1:-}" == "--engineering-assets" ]]; then
  SHIU_COMMIT="91bdd1e7dcf193f3e7ca5a8933497fcef63b7960" # pragma: allowlist secret
  CONNECTOME_DIR="$ROOT/data/connectome/full"
  PUBLIC_DIR="$ROOT/data/public/unsw_nb15_v3"
  mkdir -p "$CONNECTOME_DIR" "$PUBLIC_DIR"

  download \
    "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/$SHIU_COMMIT/Completeness_783.csv" \
    "$CONNECTOME_DIR/Completeness_783.csv"
  download \
    "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/$SHIU_COMMIT/Connectivity_783.parquet" \
    "$CONNECTOME_DIR/Connectivity_783.parquet"
  download \
    "https://zenodo.org/api/records/10141617/files/UNSW-NB15-V3.zip/content" \
    "$PUBLIC_DIR/UNSW-NB15-V3.zip"

  echo "227a16cb2fda2997a2cc04907828e5a5  $PUBLIC_DIR/UNSW-NB15-V3.zip" \
    | md5sum --check --status
  (
    cd "$CONNECTOME_DIR"
    sha256sum -- Completeness_783.csv Connectivity_783.parquet > MANIFEST.sha256
  )
  (
    cd "$PUBLIC_DIR"
    sha256sum -- UNSW-NB15-V3.zip > MANIFEST.sha256
  )
  exit 0
fi

SHIU_COMMIT="91bdd1e7dcf193f3e7ca5a8933497fcef63b7960" # pragma: allowlist secret
SHIU_DIR="$ROOT/data/connectome/vendor_shiu"
mkdir -p "$SHIU_DIR"
for file in model.py LICENSE; do
  download \
    "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/$SHIU_COMMIT/$file" \
    "$SHIU_DIR/$file"
done
(
  cd "$ROOT/data/connectome"
  sha256sum -- vendor_shiu/LICENSE vendor_shiu/model.py \
    > MANIFEST.sha256
)

if [[ "${1:-}" == "--model-source-only" ]]; then
  exit 0
fi

cd "$ROOT"
mkdir -p wheelhouse
uv lock --python 3.11
uv export --frozen --all-groups --no-emit-project --no-hashes \
  --output-file requirements.lock
find wheelhouse -maxdepth 1 -type f -name 'pytest-*.whl' -delete
python3.11 -m pip download --only-binary=:all: --dest wheelhouse \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements.lock
(
  cd wheelhouse
  sha256sum -- *.whl > MANIFEST.sha256
)

if [[ "${1:-}" == "--runtime-only" ]]; then
  cat >&2 <<'EOF'
  Dataset and FlyWire downloads are optional. The pinned model source and
  wheelhouse are ready for the synthetic offline workflow.
EOF
  exit 0
fi

cat >&2 <<'EOF'
Dataset and connectome downloads are not configured yet. Their exact release
URLs, checksums, and licenses must be selected and recorded before this script
may download them. The TypeSafe documentation and wheelhouse were downloaded
successfully.
EOF
exit 2
