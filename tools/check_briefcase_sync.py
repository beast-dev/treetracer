#!/usr/bin/env python3
"""Verify Briefcase config mirrors ``[project]`` in pyproject.toml.

Two invariants:

1. ``[tool.briefcase.app.treetracer].requires`` == ``[project].dependencies``
2. ``[tool.briefcase].version`` == ``[project].version``

Failure prints a diff and exits 1, so CI catches drift on the PR that
introduced it. The fix is always "edit pyproject.toml manually" — we do
not auto-write, because Briefcase deps occasionally need version-pin
divergences (e.g. when a wheel only exists at a specific version for a
platform Briefcase targets) and a silent auto-sync would mask that.

Run after any change to ``[project].dependencies`` or
``[project].version``::

    uv run python tools/check_briefcase_sync.py

Stdlib-only on purpose — Python 3.11+'s ``tomllib`` is enough, no need
to pull tomlkit just for a CI check.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def main() -> int:
    data = tomllib.loads(PYPROJECT.read_text())

    project = data["project"]
    briefcase = data["tool"]["briefcase"]
    briefcase_app = briefcase["app"]["treetracer"]

    project_deps = list(project["dependencies"])
    # Filter out path-style entries (local wheelhouse references). These
    # exist in briefcase to work around sdist-only PyPI packages and have
    # no equivalent in ``[project].dependencies`` (uv installs from PyPI).
    briefcase_requires = [
        r for r in briefcase_app["requires"]
        if "/" not in r and not r.endswith(".whl")
    ]
    project_version = project["version"]
    briefcase_version = briefcase["version"]

    failures: list[str] = []

    if project_deps != briefcase_requires:
        only_project = [d for d in project_deps if d not in briefcase_requires]
        only_briefcase = [d for d in briefcase_requires if d not in project_deps]
        failures.append(
            "[project].dependencies and [tool.briefcase.app.treetracer].requires "
            "are out of sync (excluding local wheelhouse entries)."
        )
        if only_project:
            failures.append("  In [project] only:    " + ", ".join(only_project))
        if only_briefcase:
            failures.append("  In [briefcase] only:  " + ", ".join(only_briefcase))

    if project_version != briefcase_version:
        failures.append(
            f"Version drift: [project].version={project_version!r}, "
            f"[tool.briefcase].version={briefcase_version!r}"
        )

    if failures:
        sys.stderr.write("Briefcase config out of sync with [project]:\n\n")
        for f in failures:
            sys.stderr.write(f + "\n")
        sys.stderr.write(
            "\nFix pyproject.toml manually so the two lists match, then re-run.\n"
        )
        return 1

    print("Briefcase config in sync with [project]. OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
