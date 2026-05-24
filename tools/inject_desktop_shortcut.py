#!/usr/bin/env python3
"""Inject a desktop-shortcut Component into the Briefcase-generated WXS.

Briefcase's Windows template (as of 0.4.x) creates a Start Menu
shortcut but not a desktop one. This script post-processes the
``.wxs`` file that ``briefcase create windows`` generates, adding a
desktop ``<Component>`` + ``<ComponentRef>`` in the main feature, so
the resulting MSI also drops a TreeTracer shortcut on the user's
Desktop.

Designed for the GitHub Actions release workflow: run between
``briefcase create windows`` and ``briefcase build windows``. See
``.github/workflows/release.yml`` for the wiring.

Why pattern-based text editing rather than XML parsing: WXS files
have a mandatory namespace and ElementTree's namespace handling
serialises out as ``ns0:Element`` prefixes that WiX rejects. The
insertions are small enough that targeted string replacements are
easier to keep correct than wrestling lxml or namespace-preserving
ET serialisers.

Idempotency: the script checks for ``DESKTOP_SHORTCUT_GUID`` in the
WXS before patching; if present it no-ops. Safe to re-run.

Usage:

    python tools/inject_desktop_shortcut.py                  # autodetect WXS
    python tools/inject_desktop_shortcut.py <path/to/file.wxs>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Stable, hardcoded GUID for the desktop-shortcut Component.
#
# MUST NOT change across releases. Windows uses the Component GUID to
# track which files / registry entries / shortcuts belong to which
# install generation; changing this GUID would orphan the desktop
# shortcut on upgrade (old GUID's shortcut stays behind, new GUID's
# gets created, user sees two icons).
DESKTOP_SHORTCUT_GUID = "1B7AF935-2C8E-4D31-9F62-7E5C84A6F3B1"

# WiX schema namespace literals — v3 (older Briefcase) and v4 (current).
WIX_V3_NS = "http://schemas.microsoft.com/wix/2006/wi"
WIX_V4_NS = "http://wixtoolset.org/schemas/v4/wxs"


def find_wxs_file(search_root: Path) -> Path:
    """Locate the single WXS file Briefcase generated under
    ``search_root``. Raises ``SystemExit`` with a clear message if
    zero or more than one file is found (so a build-tree layout
    change surfaces immediately, not as a silent no-op)."""
    if not search_root.is_dir():
        raise SystemExit(
            f"expected briefcase Windows build dir {search_root!s} "
            f"not found; run 'briefcase create windows' first"
        )
    candidates = sorted(search_root.rglob("*.wxs"))
    if not candidates:
        raise SystemExit(f"no .wxs files found under {search_root!s}")
    if len(candidates) > 1:
        listing = "\n  ".join(str(p) for p in candidates)
        raise SystemExit(
            f"expected 1 .wxs file under {search_root!s}, "
            f"found {len(candidates)}:\n  {listing}"
        )
    return candidates[0]


def _extract_shortcut_attr(wxs_text: str, attr: str) -> str | None:
    """Return the value of ``attr=`` from the first existing
    ``<Shortcut ...>`` element in the WXS, or None if not found.

    Used to copy the Target / WorkingDirectory reference from the
    Start Menu shortcut so the desktop shortcut points at the same
    target — whether that's the v4 ``[#exe]`` file-id reference or a
    v3-style ``[INSTALLFOLDER]TreeTracer.exe`` path.
    """
    m = re.search(
        rf'<Shortcut[^>]*\b{re.escape(attr)}="([^"]+)"',
        wxs_text,
        re.DOTALL,
    )
    return m.group(1) if m else None


def patch_wxs(wxs_text: str, app_name: str = "TreeTracer") -> tuple[str, bool]:
    """Inject the desktop-shortcut Component + Feature ref into
    ``wxs_text``. Returns ``(new_text, modified)``: ``modified`` is
    ``False`` on a no-op (already patched).

    The function is intentionally chatty on ``stdout`` so the GitHub
    Actions log records what it did — useful when the patcher
    inevitably needs adjusting for some future Briefcase template
    change.
    """
    if DESKTOP_SHORTCUT_GUID.lower() in wxs_text.lower():
        print("  desktop shortcut already present; no changes")
        return wxs_text, False

    # Detect WiX version.
    if WIX_V4_NS in wxs_text:
        wix_version = 4
    elif WIX_V3_NS in wxs_text:
        wix_version = 3
    else:
        raise SystemExit(
            "could not detect WiX schema in WXS — looked for "
            f"{WIX_V3_NS!r} or {WIX_V4_NS!r}. Briefcase template may "
            "have moved to a newer schema; update this script."
        )
    print(f"  detected WiX v{wix_version}")

    # Copy Target / WorkingDirectory from the existing Start Menu
    # shortcut. ``[#exe]`` is the conventional fallback for WiX v4 and
    # references whichever <File> has ``Id="exe"``.
    target = _extract_shortcut_attr(wxs_text, "Target") or "[#exe]"
    working_dir = (
        _extract_shortcut_attr(wxs_text, "WorkingDirectory")
        or "INSTALLFOLDER"
    )
    print(f"  copying Target={target!r} WorkingDirectory={working_dir!r} "
          "from existing Start Menu shortcut")

    # ── Build the new desktop-shortcut XML ─────────────────────────
    #
    # The RegistryValue serves as the Component's KeyPath, which is
    # required for non-file Components in a per-user directory like
    # DesktopFolder. We put it under HKCU\Software\<app_name>\Shortcuts
    # to be a good citizen — clearly scoped to this app, easy to find
    # on uninstall.
    shortcut_body = (
        f'<Shortcut Id="ApplicationDesktopShortcut" '
        f'Name="{app_name}" '
        f'Description="{app_name}" '
        f'Target="{target}" '
        f'WorkingDirectory="{working_dir}" />'
    )
    registry_body = (
        f'<RegistryValue Root="HKCU" '
        f'Key="Software\\{app_name}\\Shortcuts" '
        f'Name="Desktop" '
        f'Type="integer" Value="1" KeyPath="yes" />'
    )
    if wix_version == 4:
        # WiX v4: <StandardDirectory Id="DesktopFolder"> wraps the
        # Component directly. ``DesktopFolder`` is a recognised
        # standard directory id; no explicit Directory declaration
        # needed elsewhere.
        new_dir_block = (
            f'\n    <StandardDirectory Id="DesktopFolder">\n'
            f'      <Component Id="DesktopShortcut" '
            f'Guid="{DESKTOP_SHORTCUT_GUID}">\n'
            f'        {shortcut_body}\n'
            f'        {registry_body}\n'
            f'      </Component>\n'
            f'    </StandardDirectory>\n'
        )
        close_tag = "</Package>"
    else:
        # WiX v3: <DirectoryRef Id="DesktopFolder"> wraps it. ``Desktop
        # Folder`` is built into WiX v3's standard directory set; no
        # explicit <Directory> declaration needed.
        new_dir_block = (
            f'\n    <DirectoryRef Id="DesktopFolder">\n'
            f'      <Component Id="DesktopShortcut" '
            f'Guid="{DESKTOP_SHORTCUT_GUID}">\n'
            f'        {shortcut_body}\n'
            f'        {registry_body}\n'
            f'      </Component>\n'
            f'    </DirectoryRef>\n'
        )
        close_tag = "</Product>"

    new_componentref = '      <ComponentRef Id="DesktopShortcut" />\n      '

    # ── Insert: Component just before the closing Package/Product tag.
    if close_tag not in wxs_text:
        raise SystemExit(
            f"could not find {close_tag!r} in WXS — expected exactly one "
            "to anchor the new Component before"
        )
    wxs_text = wxs_text.replace(
        close_tag,
        new_dir_block + "  " + close_tag,
        1,
    )

    # ── Insert: ComponentRef just before the closing Feature tag.
    if "</Feature>" not in wxs_text:
        raise SystemExit(
            "could not find </Feature> in WXS — expected the main feature "
            "to anchor the new ComponentRef before"
        )
    wxs_text = wxs_text.replace(
        "</Feature>", new_componentref + "</Feature>", 1,
    )

    print(f"  injected desktop shortcut Component "
          f"(GUID {DESKTOP_SHORTCUT_GUID}) + Feature ref")
    return wxs_text, True


def main(argv: list[str]) -> int:
    if argv:
        wxs_path = Path(argv[0])
        if not wxs_path.is_file():
            print(f"error: WXS file not found: {wxs_path}", file=sys.stderr)
            return 2
    else:
        wxs_path = find_wxs_file(Path("build/treetracer/windows"))
    print(f"patching {wxs_path}")
    text = wxs_path.read_text(encoding="utf-8")
    new_text, modified = patch_wxs(text)
    if modified:
        wxs_path.write_text(new_text, encoding="utf-8")
        print(f"  wrote {wxs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
