"""Cross-source MCC remapping logic.

When the user selects trees from two or more ``.trees`` files for an
MCC, those files may have **different** translate blocks — the same
taxon ``"foo"`` could be integer 5 in file A and integer 12 in file B.
``_build_canonical_remaps`` figures out the int→int remap so the
combined NEXUS export carries consistent integer labels. Trees from
sources missing a canonical taxon get flagged via ``missing_taxa``.

Both halves matter — silently dropping taxa would produce a wrong
MCC; silently relabelling would produce a topologically-wrong export.
"""

from __future__ import annotations

from treetracer.callbacks.treespace import (
    _build_canonical_remaps,
    _substitute_newick_labels,
)


# Tiny in-memory translate tables.
A_TRANSLATE = {"1": "alpha", "2": "beta", "3": "gamma"}
B_SAME = {"1": "alpha", "2": "beta", "3": "gamma"}          # identical to A
B_PERMUTED = {"1": "beta", "2": "gamma", "3": "alpha"}      # same taxa, diff order
B_MISSING = {"1": "alpha", "2": "delta"}                    # 'delta' not in A


def _translates(mapping):
    """Return a get_translate_map-compatible lookup over ``mapping``."""
    return lambda source: mapping.get(source)


def test_canonical_source_gets_identity_remap():
    remaps, missing = _build_canonical_remaps(
        ["A.trees"], _translates({"A.trees": A_TRANSLATE}), "A.trees",
    )
    assert remaps["A.trees"] == {}
    assert missing == set()


def test_same_translate_yields_empty_remap():
    """Source whose translate matches the canonical 1:1 needs no
    relabelling."""
    remaps, missing = _build_canonical_remaps(
        ["A.trees", "B.trees"],
        _translates({"A.trees": A_TRANSLATE, "B.trees": B_SAME}),
        "A.trees",
    )
    assert remaps["B.trees"] == {}
    assert missing == set()


def test_permuted_translate_produces_int_remap():
    """Same taxa, different integer labels → non-trivial remap."""
    remaps, missing = _build_canonical_remaps(
        ["A.trees", "B.trees"],
        _translates({"A.trees": A_TRANSLATE, "B.trees": B_PERMUTED}),
        "A.trees",
    )
    # In B: 1=beta, 2=gamma, 3=alpha. In A: 1=alpha, 2=beta, 3=gamma.
    # So B's int 1 (beta) should map to A's int 2.
    assert remaps["B.trees"] == {"1": "2", "2": "3", "3": "1"}
    assert missing == set()


def test_missing_taxa_reported_not_silently_dropped():
    """B has 'delta' (not in A's canonical translate) — must surface
    in missing_taxa so callers can bail with a user-facing error."""
    remaps, missing = _build_canonical_remaps(
        ["A.trees", "B.trees"],
        _translates({"A.trees": A_TRANSLATE, "B.trees": B_MISSING}),
        "A.trees",
    )
    assert "delta" in missing
    # 'alpha' is in both → produces an identity remap entry (omitted
    # because canonical int already equals source int).
    assert "1" not in remaps["B.trees"]


def test_substitute_newick_labels_replaces_only_whole_integers():
    """The label substitution must not eat partial matches — e.g.
    when remapping ``1 → 10``, a subsequent ``10`` in the source must
    not become ``100``. Smoke test the common case."""
    newick = "(1:0.5,(2:0.3,3:0.7):0.2)"
    remap = {"1": "2", "2": "3", "3": "1"}
    out = _substitute_newick_labels(newick, remap)
    # Each integer token gets replaced exactly once.
    assert out == "(2:0.5,(3:0.3,1:0.7):0.2)"


def test_substitute_newick_labels_no_op_on_empty_map():
    newick = "(1:0.5,2:0.3)"
    assert _substitute_newick_labels(newick, {}) == newick
