"""Shared pytest fixtures for the TreeTracer CI test suite.

The 100-tree fixture at ``src/test/test.trees`` is parsed once per
session (raw NEXUS + rapidtrees presence matrix) and re-used by every
integration test. Session-scoped to keep CI runtime under control.

The minimal NEXUS parser handles BEAST 1.x quirks: the ``[&lnP=...]``
annotation that sits *inside* the ``= [&R]`` separator (so ``find('=')``
grabs the wrong equals), and the ``[&R]`` rooted flag that has to be
stripped from the newick body before handing it to rapidtrees.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
TREES_PATH = HERE / "test.trees"


def _parse_nexus_minimal(path: Path, n_trees: int | None = None):
    """Extract (translate_map, names, newicks) from a BEAST .trees file.

    * Uses ``find(' = ')`` (with spaces) to avoid the BEAST-1 inline
      ``[&lnP=...,joint=...]`` annotation that contains a literal ``=``
      before the actual separator.
    * Strips the leading ``[&R]`` rooted flag so newick bodies start
      with ``(`` — the format rapidtrees expects.
    """
    tmap: dict[str, str] = {}
    names: list[str] = []
    newicks: list[str] = []
    with open(path) as f:
        in_translate = False
        for line in f:
            s = line.strip()
            if not in_translate and re.match(r"^[Tt]ranslate$", s):
                in_translate = True
                continue
            if in_translate:
                if s == ";":
                    in_translate = False
                    continue
                parts = s.rstrip(",").split(None, 1)
                if len(parts) == 2:
                    tmap[parts[0]] = parts[1].strip("'")
                continue
            if s.lower().startswith("tree "):
                eq = line.find(" = ")
                if eq < 0:
                    continue
                name = line[5:eq].split("[")[0].strip()
                newick = line[eq + 3:].strip()
                if newick.startswith("[&R]"):
                    newick = newick[4:].lstrip()
                names.append(name)
                newicks.append(newick)
                if n_trees is not None and len(names) >= n_trees:
                    break
    return tmap, names, newicks


@pytest.fixture(scope="session")
def trees_path() -> Path:
    """Absolute path to the 100-tree CI fixture."""
    if not TREES_PATH.exists():
        pytest.skip(f"missing test fixture: {TREES_PATH}")
    return TREES_PATH


@pytest.fixture(scope="session")
def parsed_full(trees_path):
    """All 100 trees parsed via the minimal NEXUS reader.

    Returns ``(translate_map, names, newicks)``. ``newicks`` are raw
    newick bodies (no ``[&R]`` prefix), in source-file order.
    """
    return _parse_nexus_minimal(trees_path)


@pytest.fixture(scope="session")
def parsed_50(trees_path):
    """First 50 trees only — used by the DendroPy pairwise RF
    cross-check (O(n²), too slow on 100 in per-PR CI)."""
    return _parse_nexus_minimal(trees_path, n_trees=50)


@pytest.fixture(scope="session")
def rapidtrees_full(parsed_full):
    """``(names, rf_matrix, presence, leaf_names, n_bip)`` from
    rapidtrees on all 100 fixture trees, **rooted-clade mode**
    (matches the production worker — see ``rf/_worker.py``)."""
    from treetracer.rf import rf_distance_with_snapshots_from_newick_iter
    tmap, names, newicks = parsed_full
    result_names, rf_matrix, presence, leaf_names, n_bip, _bip_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            names, iter(newicks), [tmap], [0] * len(names), rooted=True,
        )
    )
    return list(result_names), rf_matrix, presence, leaf_names, n_bip


@pytest.fixture(scope="session")
def rapidtrees_50(parsed_50):
    """``(names, rf_matrix, presence, leaf_names, n_bip)`` from
    rapidtrees on the first 50 fixture trees, **rooted-clade mode**."""
    from treetracer.rf import rf_distance_with_snapshots_from_newick_iter
    tmap, names, newicks = parsed_50
    result_names, rf_matrix, presence, leaf_names, n_bip, _bip_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            names, iter(newicks), [tmap], [0] * len(names), rooted=True,
        )
    )
    return list(result_names), rf_matrix, presence, leaf_names, n_bip


@pytest.fixture(scope="session")
def dendropy_trees_50(trees_path):
    """First 50 fixture trees parsed by DendroPy, sharing one
    TaxonNamespace. Trees stay rooted (BEAST's ``[&R]`` flag is
    honoured) to match production rapidtrees ``rooted=True``: every
    rooted clade is a distinct entity and the root-induced split is
    real, not a ghost. RF cross-checks compute rooted-clade
    symmetric difference manually rather than calling
    ``dendropy.symmetric_difference`` (that one ignores rooting for
    historical reasons).
    """
    dendropy = pytest.importorskip("dendropy")
    taxa = dendropy.TaxonNamespace()
    trees = []
    for tree, _ in zip(
        dendropy.Tree.yield_from_files(
            files=[str(trees_path)], schema="nexus",
            taxon_namespace=taxa, preserve_underscores=True,
        ),
        range(50),
    ):
        trees.append(tree)
    return dendropy.TreeList(trees, taxon_namespace=taxa)


@pytest.fixture(scope="session")
def dendropy_trees_100(trees_path):
    """All 100 fixture trees parsed by DendroPy, rooted (matches
    rapidtrees' ``rooted=True`` production mode)."""
    dendropy = pytest.importorskip("dendropy")
    taxa = dendropy.TaxonNamespace()
    trees = []
    for tree, _ in zip(
        dendropy.Tree.yield_from_files(
            files=[str(trees_path)], schema="nexus",
            taxon_namespace=taxa, preserve_underscores=True,
        ),
        range(100),
    ):
        trees.append(tree)
    return dendropy.TreeList(trees, taxon_namespace=taxa)


def rooted_clade_set(tree):
    """Per-internal-node descendant tip-label set, matching
    rapidtrees' rooted-mode encoding (every subtree bitset except
    the all-leaves seed-node clade).

    Exposed as a module-level helper so tests can reach for it
    without re-implementing the symmetric-difference reference."""
    out = set()
    for node in tree.preorder_node_iter():
        if node is tree.seed_node:
            continue
        out.add(frozenset(l.taxon.label for l in node.leaf_iter()))
    return out
