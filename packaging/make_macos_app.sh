#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Creates a macOS .app bundle from the PyApp binary
# Usage: ./packaging/make_macos_app.sh
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"

VERSION=$(grep '^version' "$REPO_ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
ARCH=$(uname -m)
BINARY="$DIST_DIR/TreeTracer-${VERSION}-darwin-${ARCH}"

if [ ! -f "$BINARY" ]; then
    echo "Binary not found. Run ./packaging/build.sh first."
    exit 1
fi

APP_DIR="$DIST_DIR/TreeTracer.app"
CONTENTS="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"

# Clean previous build
rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES"

# Copy binary
cp "$BINARY" "$MACOS_DIR/TreeTracer"
chmod +x "$MACOS_DIR/TreeTracer"

# Info.plist
cat > "$CONTENTS/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>TreeTracer</string>
    <key>CFBundleDisplayName</key>
    <string>TreeTracer</string>
    <key>CFBundleIdentifier</key>
    <string>be.kuleuven.treetracer</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>TreeTracer</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
</dict>
</plist>
EOF

# Copy icon if it exists
ICON_SRC="$REPO_ROOT/src/treetracer/assets/treetracer-icon.png"
if [ -f "$ICON_SRC" ]; then
    # Convert PNG to icns (macOS only)
    ICONSET="$RESOURCES/AppIcon.iconset"
    mkdir -p "$ICONSET"
    sips -z 16 16     "$ICON_SRC" --out "$ICONSET/icon_16x16.png"    2>/dev/null
    sips -z 32 32     "$ICON_SRC" --out "$ICONSET/icon_16x16@2x.png" 2>/dev/null
    sips -z 32 32     "$ICON_SRC" --out "$ICONSET/icon_32x32.png"    2>/dev/null
    sips -z 64 64     "$ICON_SRC" --out "$ICONSET/icon_32x32@2x.png" 2>/dev/null
    sips -z 128 128   "$ICON_SRC" --out "$ICONSET/icon_128x128.png"  2>/dev/null
    sips -z 256 256   "$ICON_SRC" --out "$ICONSET/icon_128x128@2x.png" 2>/dev/null
    sips -z 256 256   "$ICON_SRC" --out "$ICONSET/icon_256x256.png"  2>/dev/null
    sips -z 512 512   "$ICON_SRC" --out "$ICONSET/icon_256x256@2x.png" 2>/dev/null
    sips -z 512 512   "$ICON_SRC" --out "$ICONSET/icon_512x512.png"  2>/dev/null
    sips -z 1024 1024 "$ICON_SRC" --out "$ICONSET/icon_512x512@2x.png" 2>/dev/null
    iconutil -c icns "$ICONSET" -o "$RESOURCES/AppIcon.icns" 2>/dev/null || true
    rm -rf "$ICONSET"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Created: $APP_DIR"
echo ""
echo "  Double-click to launch, or:"
echo "    open $APP_DIR"
echo ""
echo "  To distribute, create a DMG:"
echo "    hdiutil create -volname TreeTracer \\"
echo "      -srcfolder $APP_DIR \\"
echo "      -ov -format UDZO \\"
echo "      $DIST_DIR/TreeTracer-${VERSION}.dmg"
echo "════════════════════════════════════════════════════════"
