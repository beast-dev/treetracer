#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# TreeTracer PyApp Builder
#
# Creates a single executable that bootstraps Python via uv on first run.
# Requires: Rust toolchain (cargo)
#
# Usage:
#   ./packaging/build.sh              # build for current platform
#   ./packaging/build.sh --embedded   # embed the wheel (fully offline)
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/packaging/dist"

# Read version from pyproject.toml
VERSION=$(grep '^version' "$REPO_ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
echo "Building TreeTracer v${VERSION}"

# ── Step 1: Build the wheel ──────────────────────────────────────────
echo "Building wheel..."
cd "$REPO_ROOT"
uv build --wheel --out-dir "$OUT_DIR"
WHEEL=$(ls "$OUT_DIR"/treetracer-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: Wheel not found in $OUT_DIR"
    exit 1
fi
echo "Wheel: $WHEEL"

# ── Step 2: Get or build PyApp ───────────────────────────────────────
PYAPP_DIR="$SCRIPT_DIR/.pyapp-source"
if [ ! -d "$PYAPP_DIR" ]; then
    echo "Downloading PyApp source..."
    PYAPP_VERSION="0.27.0"
    curl -sSL "https://github.com/ofek/pyapp/releases/download/v${PYAPP_VERSION}/source.tar.gz" \
        | tar xz -C "$SCRIPT_DIR"
    mv "$SCRIPT_DIR/pyapp-v${PYAPP_VERSION}" "$PYAPP_DIR"
fi

# ── Step 3: Configure and build ──────────────────────────────────────
echo "Building executable..."
cd "$PYAPP_DIR"

# Core config
export PYAPP_PROJECT_NAME="treetracer"
export PYAPP_PROJECT_VERSION="$VERSION"
export PYAPP_PYTHON_VERSION="3.13"
export PYAPP_EXEC_MODULE="treetracer"

# Use the local wheel if --embedded flag is set, otherwise install from wheel path
if [[ "${1:-}" == "--embedded" ]]; then
    export PYAPP_PROJECT_PATH="$WHEEL"
    echo "Mode: embedded wheel (offline-capable)"
else
    export PYAPP_PROJECT_PATH="$WHEEL"
    echo "Mode: local wheel"
fi

# Isolation & environment
export PYAPP_FULL_ISOLATION="true"
export PYAPP_PASS_LOCATION="true"

# Show a message during first-run setup
export PYAPP_SELF_COMMAND="none"
export PYAPP_METADATA_TEMPLATE="Setting up TreeTracer for the first time. This may take a minute..."

cargo build --release 2>&1 | tail -5

# ── Step 4: Copy output ─────────────────────────────────────────────
BINARY="$PYAPP_DIR/target/release/pyapp"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi

# Name it nicely
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
DEST="$OUT_DIR/TreeTracer-${VERSION}-${PLATFORM}-${ARCH}"
cp "$BINARY" "$DEST"
chmod +x "$DEST"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Built: $DEST"
echo "  Size:  $(du -h "$DEST" | cut -f1)"
echo ""
echo "  Users run this single file. On first launch it will"
echo "  download Python and install dependencies (~30s)."
echo "  Subsequent launches are instant."
echo "════════════════════════════════════════════════════════"
