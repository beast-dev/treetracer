"""Tests for tools/inject_desktop_shortcut.py.

We can't run the real Briefcase Windows build on macOS / Linux CI, so
the only way to verify the patcher before shipping is to drive it
against synthetic v3 and v4 WXS fixtures that mirror the structure
Briefcase's templates actually produce. If the patcher would break on
a real Briefcase WXS, these fixtures should catch it.

The fixtures are minimal — just the elements the patcher inspects —
but they exercise both schema versions, both shortcut Target shapes
(``[#exe]`` for v4 / ``[INSTALLFOLDER]TreeTracer.exe`` for v3), and
the idempotency contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# Load the tools/ script as a module — it's not a package, just a
# top-level script, so importlib.util is the cleanest way.
_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "inject_desktop_shortcut.py"
_spec = importlib.util.spec_from_file_location("inject_desktop_shortcut", _TOOL_PATH)
inject_desktop_shortcut = importlib.util.module_from_spec(_spec)
sys.modules["inject_desktop_shortcut"] = inject_desktop_shortcut
_spec.loader.exec_module(inject_desktop_shortcut)


# ── Fixtures: minimal WXS for v3 and v4 ─────────────────────────────

WXS_V4_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">
  <Package Name="TreeTracer" Manufacturer="BEAST community"
           Version="0.9.0" UpgradeCode="ABCD1234-0000-0000-0000-000000000000">
    <MediaTemplate EmbedCab="yes" />

    <StandardDirectory Id="ProgramFiles64Folder">
      <Directory Id="INSTALLFOLDER" Name="TreeTracer">
        <Component Id="MainExeComponent" Guid="11111111-2222-3333-4444-555555555555">
          <File Id="exe" Source="TreeTracer.exe" KeyPath="yes" />
        </Component>
      </Directory>
    </StandardDirectory>

    <StandardDirectory Id="ProgramMenuFolder">
      <Component Id="ProgramMenuShortcut" Guid="99999999-8888-7777-6666-555555555555">
        <Shortcut Id="ApplicationStartMenuShortcut"
                  Name="TreeTracer"
                  Target="[#exe]"
                  WorkingDirectory="INSTALLFOLDER" />
        <RegistryValue Root="HKCU" Key="Software\\TreeTracer"
                       Name="installed" Type="integer" Value="1" KeyPath="yes" />
      </Component>
    </StandardDirectory>

    <Feature Id="ApplicationFeature" Title="TreeTracer" Level="1">
      <ComponentRef Id="MainExeComponent" />
      <ComponentRef Id="ProgramMenuShortcut" />
    </Feature>
  </Package>
</Wix>
"""

WXS_V3_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" Name="TreeTracer" Language="1033" Version="0.9.0"
           Manufacturer="BEAST community"
           UpgradeCode="ABCD1234-0000-0000-0000-000000000000">
    <Package InstallerVersion="500" Compressed="yes" InstallScope="perMachine" />
    <MediaTemplate EmbedCab="yes" />

    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFiles64Folder">
        <Directory Id="INSTALLFOLDER" Name="TreeTracer">
          <Component Id="MainExeComponent" Guid="11111111-2222-3333-4444-555555555555">
            <File Id="exe" Source="TreeTracer.exe" KeyPath="yes" />
          </Component>
        </Directory>
      </Directory>
      <Directory Id="ProgramMenuFolder">
        <Component Id="ProgramMenuShortcut" Guid="99999999-8888-7777-6666-555555555555">
          <Shortcut Id="ApplicationStartMenuShortcut"
                    Name="TreeTracer"
                    Target="[INSTALLFOLDER]TreeTracer.exe"
                    WorkingDirectory="INSTALLFOLDER" />
          <RegistryValue Root="HKCU" Key="Software\\TreeTracer"
                         Name="installed" Type="integer" Value="1" KeyPath="yes" />
        </Component>
      </Directory>
      <Directory Id="DesktopFolder" Name="Desktop" />
    </Directory>

    <Feature Id="ApplicationFeature" Title="TreeTracer" Level="1">
      <ComponentRef Id="MainExeComponent" />
      <ComponentRef Id="ProgramMenuShortcut" />
    </Feature>
  </Product>
</Wix>
"""


# ── Tests ───────────────────────────────────────────────────────────


def test_wix_v4_basic_injection():
    """WiX v4: inject and verify all the pieces land in the right
    structural positions."""
    out, modified = inject_desktop_shortcut.patch_wxs(WXS_V4_TEMPLATE)
    assert modified is True

    # Component lives in a StandardDirectory(DesktopFolder).
    assert '<StandardDirectory Id="DesktopFolder">' in out
    assert 'Id="DesktopShortcut"' in out
    assert inject_desktop_shortcut.DESKTOP_SHORTCUT_GUID in out

    # Component reference is inside the Feature.
    feature_block = out.split("<Feature")[1].split("</Feature>")[0]
    assert '<ComponentRef Id="DesktopShortcut" />' in feature_block

    # Component sits before the closing </Package>.
    assert out.index('Id="DesktopShortcut"') < out.index("</Package>")

    # Target was copied verbatim from the existing Start Menu shortcut.
    assert 'Target="[#exe]"' in out

    # Output is still well-formed XML (catches missing close-tags etc.).
    import xml.etree.ElementTree as ET
    ET.fromstring(out)


def test_wix_v3_basic_injection():
    """WiX v3: inject and verify the v3-style DirectoryRef wrapping."""
    out, modified = inject_desktop_shortcut.patch_wxs(WXS_V3_TEMPLATE)
    assert modified is True

    # v3 uses DirectoryRef rather than StandardDirectory.
    assert '<DirectoryRef Id="DesktopFolder">' in out
    assert '<StandardDirectory' not in out  # would be the v4 form
    assert 'Id="DesktopShortcut"' in out
    assert inject_desktop_shortcut.DESKTOP_SHORTCUT_GUID in out

    # Component reference is inside the Feature.
    feature_block = out.split("<Feature")[1].split("</Feature>")[0]
    assert '<ComponentRef Id="DesktopShortcut" />' in feature_block

    # Component sits before the closing </Product>.
    assert out.index('Id="DesktopShortcut"') < out.index("</Product>")

    # Target was copied verbatim — v3 templates use the explicit path form.
    assert 'Target="[INSTALLFOLDER]TreeTracer.exe"' in out

    # XML well-formed.
    import xml.etree.ElementTree as ET
    ET.fromstring(out)


def test_idempotency_v4():
    """Running the patcher twice on the same input must produce the
    same output and report no modification on the second run."""
    first, mod1 = inject_desktop_shortcut.patch_wxs(WXS_V4_TEMPLATE)
    second, mod2 = inject_desktop_shortcut.patch_wxs(first)
    assert mod1 is True
    assert mod2 is False
    assert first == second


def test_idempotency_v3():
    first, mod1 = inject_desktop_shortcut.patch_wxs(WXS_V3_TEMPLATE)
    second, mod2 = inject_desktop_shortcut.patch_wxs(first)
    assert mod1 is True
    assert mod2 is False
    assert first == second


def test_unknown_schema_errors_loudly():
    """A WXS with a schema we don't recognise should exit with a clear
    message so a future Briefcase template change surfaces immediately
    instead of silently producing a broken installer."""
    bogus = '<?xml version="1.0"?><Wix xmlns="http://example.com/wix/v9"/>'
    with pytest.raises(SystemExit, match="could not detect WiX schema"):
        inject_desktop_shortcut.patch_wxs(bogus)


def test_missing_shortcut_falls_back_to_default_target():
    """If the template has no <Shortcut> element to copy from (edge
    case), the patcher should still produce a valid output using
    [#exe] as the conventional default Target."""
    no_shortcut = WXS_V4_TEMPLATE.replace(
        '<Shortcut Id="ApplicationStartMenuShortcut"', '<!--')
    no_shortcut = no_shortcut.replace(
        'WorkingDirectory="INSTALLFOLDER" />', 'placeholder -->')
    out, modified = inject_desktop_shortcut.patch_wxs(no_shortcut)
    assert modified is True
    assert 'Target="[#exe]"' in out
    assert 'WorkingDirectory="INSTALLFOLDER"' in out


def test_missing_close_tag_errors():
    """Truncated input must error, not silently produce garbage."""
    truncated = WXS_V4_TEMPLATE.replace("</Package>", "")
    with pytest.raises(SystemExit, match=r"</Package>"):
        inject_desktop_shortcut.patch_wxs(truncated)


def test_guid_stability_is_documented():
    """The Component GUID MUST NOT change across releases — Windows
    uses it for upgrade tracking. Pin the literal here so a casual
    edit can't drift it."""
    assert inject_desktop_shortcut.DESKTOP_SHORTCUT_GUID == \
        "1B7AF935-2C8E-4D31-9F62-7E5C84A6F3B1"


def test_find_wxs_no_dir_errors():
    """Helpful error when the build dir doesn't exist (i.e. user ran
    the patcher before briefcase create)."""
    with pytest.raises(SystemExit, match="not found"):
        inject_desktop_shortcut.find_wxs_file(Path("/nonexistent/path/xyz"))


def test_find_wxs_no_files_errors(tmp_path):
    with pytest.raises(SystemExit, match="no .wxs files found"):
        inject_desktop_shortcut.find_wxs_file(tmp_path)


def test_find_wxs_multiple_files_errors(tmp_path):
    (tmp_path / "a.wxs").write_text("<?xml ?>")
    (tmp_path / "b.wxs").write_text("<?xml ?>")
    with pytest.raises(SystemExit, match="expected 1 .wxs file"):
        inject_desktop_shortcut.find_wxs_file(tmp_path)


def test_find_wxs_single_file_returns_it(tmp_path):
    nested = tmp_path / "app" / "wix"
    nested.mkdir(parents=True)
    target = nested / "TreeTracer.wxs"
    target.write_text("<?xml ?>")
    assert inject_desktop_shortcut.find_wxs_file(tmp_path) == target
