"""Focused tests for the MrHIPSTR implementation."""

from __future__ import annotations

import numpy as np
import pytest

from treetracer.consensus_tree.mrhipstr import decode_rooted_clade_catalog
from treetracer.rf import rf_distance_with_snapshots_from_newick_iter


def _bits(n_taxa: int, *taxon_indices: int) -> np.ndarray:
    row = np.zeros(n_taxa, dtype=np.uint8)
    row[list(taxon_indices)] = 1
    return row


def test_decode_rooted_clade_catalog_maps_columns_and_adds_implicit_root():
    rows = np.stack(
        [
            _bits(4, 0),
            _bits(4, 1),
            _bits(4, 2),
            _bits(4, 3),
            _bits(4, 0, 1),
            _bits(4, 2, 3),
            _bits(4, 0, 1, 2),
            _bits(4, 0, 2),  # inactive in the selected posterior sample
        ]
    )

    catalog = decode_rooted_clade_catalog(
        bipartition_bits=rows,
        leaf_names=np.array(["'A'", '"B"', "C", "D"]),
        active_columns=[6, 4, 0, 5, 2, 3, 1],
    )

    assert catalog.leaf_names == ("A", "B", "C", "D")
    assert catalog.n_taxa == 4
    assert catalog.root_bits == 0b1111
    assert catalog.clade_by_column == {
        0: 0b0001,
        1: 0b0010,
        2: 0b0100,
        3: 0b1000,
        4: 0b0011,
        5: 0b1100,
        6: 0b0111,
    }
    assert catalog.column_by_clade[0b0011] == 4
    assert catalog.root_bits not in catalog.column_by_clade
    assert 7 not in catalog.clade_by_column
    assert catalog.clades == (
        0b0001,
        0b0010,
        0b0100,
        0b1000,
        0b0011,
        0b1100,
        0b0111,
        0b1111,
    )


def test_decode_rooted_clade_catalog_supports_more_than_64_taxa():
    n_taxa = 70
    rows = [_bits(n_taxa, index) for index in range(n_taxa)]
    rows.append(_bits(n_taxa, 0, 64, 69))

    catalog = decode_rooted_clade_catalog(
        bipartition_bits=np.stack(rows),
        leaf_names=[f"taxon-{index}" for index in range(n_taxa)],
        active_columns=np.arange(len(rows)),
    )

    wide_clade = (1 << 0) | (1 << 64) | (1 << 69)
    assert catalog.clade_by_column[70] == wide_clade
    assert catalog.column_by_clade[wide_clade] == 70
    assert catalog.root_bits == (1 << 70) - 1


def test_decode_rooted_clade_catalog_matches_rapidtrees_rooted_snapshot():
    newicks = [
        "(A:1,(B:1,(C:1,D:1):1):1):0;",
        "((A:1,B:1):1,(C:1,D:1):1):0;",
    ]
    _, _, presence, leaf_names, _, bipartition_bits = (
        rf_distance_with_snapshots_from_newick_iter(
            ["tree-1", "tree-2"],
            iter(newicks),
            [{}],
            [0, 0],
            rooted=True,
        )
    )
    active_columns = np.flatnonzero(presence.sum(axis=0))

    catalog = decode_rooted_clade_catalog(
        bipartition_bits=bipartition_bits,
        leaf_names=leaf_names,
        active_columns=active_columns,
    )

    assert catalog.leaf_names == ("A", "B", "C", "D")
    assert catalog.root_bits == 0b1111
    assert catalog.root_bits not in catalog.column_by_clade
    assert {1 << index for index in range(4)} <= set(catalog.clades)
    assert presence.sum(axis=1).tolist() == [6, 6]


def test_decode_rooted_clade_catalog_rejects_missing_singletons():
    rows = np.stack(
        [
            _bits(3, 0),
            _bits(3, 1),
            _bits(3, 0, 1),
        ]
    )

    with pytest.raises(ValueError, match="missing active singleton.*'C'"):
        decode_rooted_clade_catalog(
            bipartition_bits=rows,
            leaf_names=["A", "B", "C"],
            active_columns=[0, 1, 2],
        )


def test_decode_rooted_clade_catalog_rejects_explicit_root():
    rows = np.stack(
        [
            _bits(2, 0),
            _bits(2, 1),
            _bits(2, 0, 1),
        ]
    )

    with pytest.raises(ValueError, match="all-taxa root"):
        decode_rooted_clade_catalog(
            bipartition_bits=rows,
            leaf_names=["A", "B"],
            active_columns=[0, 1, 2],
        )


@pytest.mark.parametrize(
    ("rows", "leaf_names", "active_columns", "error", "message"),
    [
        (
            np.array([1, 0], dtype=np.uint8),
            ["A", "B"],
            [0],
            ValueError,
            "two-dimensional",
        ),
        (
            np.eye(2, dtype=np.uint8),
            ["A"],
            [0, 1],
            ValueError,
            "width does not match",
        ),
        (
            np.eye(2, dtype=np.float64),
            ["A", "B"],
            [0, 1],
            TypeError,
            "uint8 or bool",
        ),
        (
            np.eye(2, dtype=np.uint8),
            ["A", "B"],
            [0, 2],
            IndexError,
            "outside",
        ),
        (
            np.eye(2, dtype=np.uint8),
            ["A", "B"],
            [0, 0, 1],
            ValueError,
            "duplicate column",
        ),
        (
            np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8),
            ["A", "B"],
            [0, 1, 2],
            ValueError,
            "same clade",
        ),
        (
            np.eye(2, dtype=np.uint8),
            ["'A'", "A"],
            [0, 1],
            ValueError,
            "duplicate normalized",
        ),
    ],
)
def test_decode_rooted_clade_catalog_rejects_incompatible_snapshots(
    rows,
    leaf_names,
    active_columns,
    error,
    message,
):
    with pytest.raises(error, match=message):
        decode_rooted_clade_catalog(
            bipartition_bits=rows,
            leaf_names=leaf_names,
            active_columns=active_columns,
        )
