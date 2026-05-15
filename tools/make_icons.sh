#!/usr/bin/env bash
# Regenerate packaging/icons/treetracer.{icns,ico,png} from the source PNG.
#
# Source of truth: src/treetracer/assets/treetracer-icon.png
# Outputs go to:   packaging/icons/
#
# The .icns step uses macOS's native ``sips`` + ``iconutil`` — pure-Python
# alternatives exist (icnsutil) but produce slightly inferior multi-resolution
# bundles. If you don't have a Mac and need to regen icons, ping someone
# who does, or extend this script with the icnsutil fallback.
#
# .ico and .png are generated via Pillow inside an ephemeral uv-run env so
# the script has no persistent dep beyond ``uv``.
set -euo pipefail

cd "$(dirname "$0")/.."

SRC="src/treetracer/assets/treetracer-icon.png"
OUT="packaging/icons"
mkdir -p "$OUT"

# ---- .icns (macOS) ----------------------------------------------------------
if ! command -v sips >/dev/null 2>&1; then
    echo "error: 'sips' not found. .icns regen requires macOS." >&2
    exit 1
fi
ICONSET="$OUT/treetracer.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
for sz in 16 32 64 128 256 512 1024; do
    sips -z "$sz" "$sz" "$SRC" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null
    if [ "$sz" -le 512 ]; then
        sz2=$((sz * 2))
        sips -z "$sz2" "$sz2" "$SRC" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
    fi
done
iconutil -c icns "$ICONSET" -o "$OUT/treetracer.icns"
rm -rf "$ICONSET"
echo "Wrote $OUT/treetracer.icns"

# ---- .ico + .png (cross-platform via Pillow) --------------------------------
uv run --with pillow python - <<'PY'
from PIL import Image
src = Image.open("src/treetracer/assets/treetracer-icon.png").convert("RGBA")
sizes = [(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)]
src.save("packaging/icons/treetracer.ico", format="ICO", sizes=sizes)
src.resize((512,512), Image.LANCZOS).save("packaging/icons/treetracer.png", format="PNG")
PY
echo "Wrote $OUT/treetracer.ico"
echo "Wrote $OUT/treetracer.png"
