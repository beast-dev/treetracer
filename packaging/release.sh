#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# TreeTracer Release Builder
#
# Clears cache, builds binary, creates .app bundle, optionally creates .dmg.
#
# Usage:
#   ./packaging/release.sh          # build .app
#   ./packaging/release.sh --dmg    # build .app + .dmg
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Clearing cached environment..."
rm -rf "$HOME/Library/Application Support/pyapp/treetracer" 2>/dev/null || true
rm -rf "$HOME/.local/share/pyapp/treetracer" 2>/dev/null || true

echo ""
"$SCRIPT_DIR/build.sh"

echo ""
"$SCRIPT_DIR/make_macos_app.sh"

if [[ "${1:-}" == "--dmg" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    VERSION=$(grep '^version' "$REPO_ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
    DMG="$SCRIPT_DIR/dist/TreeTracer-${VERSION}.dmg"
    echo ""
    echo "Creating DMG..."
    hdiutil create -volname TreeTracer \
        -srcfolder "$SCRIPT_DIR/dist/TreeTracer.app" \
        -ov -format UDZO \
        "$DMG"
    echo "DMG: $DMG"
fi

echo ""
echo "Done. Run:  open packaging/dist/TreeTracer.app"
