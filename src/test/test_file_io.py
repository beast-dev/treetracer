"""File ingest round-trip: parse → DB → byte-offset re-read.

``process_nexus_trees_streaming`` reads a NEXUS file, stores per-tree
byte offsets in a ``TreeManagerPandas``, and registers the
preamble + translate map for later export. The consensus tree export, the LnP
trace, and several other features depend on this lossless. Bytes-on-
disk semantics make this fragile around CRLF line endings (handled by
the ``.gitattributes *.trees binary`` declaration in this repo) and
text/binary mode mismatches — both worth pinning.
"""

from __future__ import annotations

import pytest

from treetracer.db.tree_manager import TreeManagerPandas
from treetracer.db.process_trees import process_nexus_trees_streaming


@pytest.fixture(scope="module")
def db_with_fixture(trees_path):
    """Fresh DB populated by streaming the 100-tree fixture."""
    db = TreeManagerPandas()
    process_nexus_trees_streaming(str(trees_path), db, file_source="test.trees")
    db.flush()
    return db


@pytest.mark.integration
def test_streaming_loads_expected_tree_count(db_with_fixture):
    assert len(db_with_fixture._trees) == 100


@pytest.mark.integration
def test_translate_map_present_and_well_formed(db_with_fixture):
    tmap = db_with_fixture.get_translate_map("test.trees")
    assert tmap is not None
    # Keys are integer-as-string, values are taxon labels. We don't
    # care which taxa exactly — just that the map round-tripped.
    assert len(tmap) > 0
    for k, v in tmap.items():
        assert k.isdigit(), f"non-integer key: {k!r}"
        assert isinstance(v, str) and v, f"empty value for {k!r}"


@pytest.mark.integration
def test_byte_offset_read_matches_source_line(db_with_fixture, trees_path):
    """For a random tree row, the bytes at (offset, offset+length) in
    the source file must match the newick that ``_read_newick``
    returns."""
    df = db_with_fixture._trees
    # Pick a couple of rows from different positions.
    for idx in (0, 50, 99):
        row = df.iloc[idx]
        # Read through the DB API.
        from_db = db_with_fixture._read_newick(
            row["file_source"], int(row["line_offset"]), int(row["line_length"]),
        )
        if isinstance(from_db, bytes):
            from_db = from_db.decode("utf-8")
        # Read the same byte range directly from disk.
        with open(trees_path, "rb") as f:
            f.seek(int(row["line_offset"]))
            from_disk = f.read(int(row["line_length"])).decode("utf-8")
        assert from_db == from_disk, (
            f"row {idx} '{row['name']}': DB read disagrees with on-disk bytes"
        )


@pytest.mark.integration
def test_metadata_traces_yield_lnP(db_with_fixture):
    """The Diagnostics LnP trace pulls ``lnP`` (or ``posterior`` /
    ``joint``) per tree from the metadata dict. The fixture has
    ``lnP=...`` on every tree, so the trace builder should return it
    as one of the available fields."""
    from treetracer.db.tree_service import TreeService
    svc = TreeService.__new__(TreeService)
    svc.db_manager = db_with_fixture
    traces = svc.get_metadata_traces(file_sources=["test.trees"])
    assert "lnP" in traces or "joint" in traces or "posterior" in traces, (
        f"expected an lnP-style field; got {list(traces.keys())}"
    )
    # Pick whichever exists and check it has the right length.
    for key in ("lnP", "joint", "posterior"):
        if key in traces:
            assert len(traces[key]) == 100, (
                f"{key} trace has {len(traces[key])} rows, expected 100"
            )
            assert traces[key]["value"].notna().all(), \
                f"{key} trace has NaN values"
            break


@pytest.mark.integration
def test_preamble_captured(db_with_fixture):
    """The byte-exact preamble (taxa block + translate) is stored on
    the manager so consensus tree export can reuse it verbatim."""
    preamble = db_with_fixture._source_preambles.get("test.trees")
    assert preamble is not None
    assert isinstance(preamble, (bytes, bytearray))
    # Preamble has the #NEXUS header + the translate block, but
    # stops before the first 'tree '.
    assert b"#NEXUS" in preamble
    assert b"Translate" in preamble or b"translate" in preamble
    assert b"tree " not in preamble
