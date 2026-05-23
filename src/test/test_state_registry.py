"""Server-side state module — distmat index, MCC cache, MCC registry.

These structures hold the live state of a TreeTracer session. They
have no external reference, so the tests are self-consistency:
operations should round-trip and bookkeeping (LRU bounds, counters,
cascade clears) should behave as documented.

The implementation lives in ``src/treetracer/state.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from treetracer import state


@pytest.fixture(autouse=True)
def _isolate_state():
    """Wipe global state before AND after each test so tests don't see
    each other's leftover registries."""
    state.clear_all_distmats()
    state.clear_all_mcc_trees()  # cascades into clear_all_mcc_registry
    yield
    state.clear_all_distmats()
    state.clear_all_mcc_trees()


# ---- distmat index ---------------------------------------------------------

def test_next_distmat_name_increments_zero_padded():
    a = state.next_distmat_name()
    b = state.next_distmat_name()
    assert a.startswith("RF_") and b.startswith("RF_")
    # Per-call increment + 3-digit zero-padded suffix.
    assert int(b.split("_")[1]) == int(a.split("_")[1]) + 1
    assert len(a.split("_")[1]) == 3


def test_distmat_register_and_lookup(tmp_path):
    name = "RF_TEST_001"
    names = ["g/STATE_1", "g/STATE_2", "g/STATE_3"]
    matrix_path = tmp_path / "fake_distmat.npy"
    np.save(matrix_path, np.zeros((3, 3), dtype=np.uint16))
    state.register_distmat(
        name, names, str(matrix_path),
        file_breakdown={"f": 3},
        groups_per_file={"f": ["g"]},
    )
    assert state.has_distmat(name)
    assert state.get_distmat_names(name) == names
    assert state.get_distmat_file_path(name) == str(matrix_path)
    # load_distmat returns ``(names, matrix)`` — same shape consumed by
    # ess.rf_trace.compute_rf_trace_data and related callers.
    loaded_names, M = state.load_distmat(name)
    assert loaded_names == names
    assert M.shape == (3, 3)


def test_clear_all_distmats():
    state.register_distmat("RF_X", ["a"], "/tmp/x.npy")
    assert state.has_distmat("RF_X")
    state.clear_all_distmats()
    assert not state.has_distmat("RF_X")


# ---- MCC cache + registry --------------------------------------------------

def _entry(name="RF_001", mode="Between", run=None, *, lnp=None):
    """Helper to register a synthetic MCC and return the entry dict."""
    uid = state.cache_mcc_tree(b"#NEXUS\nbegin trees;\nEnd;\n")
    return state.register_mcc(
        source_distmat=name, mode=mode, run=run, uuid=uid,
        mcc_tree={"group": run or "g", "treenum": 1, "tree_name": "g/STATE_1"},
        selection=[["g", 1]],
        log_clade_credibility=-3.0,
        mcc_log_posterior=lnp,
    )


def test_mcc_naming_per_distmat_mode_run():
    a = _entry("RF_001", "Between")
    b = _entry("RF_001", "Between")
    c = _entry("RF_001", "Within", "runA")
    d = _entry("RF_001", "Within", "runA")
    e = _entry("RF_001", "Within", "runB")
    f = _entry("RF_002", "Between")
    assert a["name"] == "RF_001_Between_MCC_1"
    assert b["name"] == "RF_001_Between_MCC_2"
    assert c["name"] == "RF_001_Within_runA_MCC_1"
    assert d["name"] == "RF_001_Within_runA_MCC_2"
    assert e["name"] == "RF_001_Within_runB_MCC_1"
    assert f["name"] == "RF_002_Between_MCC_1"


def test_mcc_get_filtered():
    a = _entry("RF_001", "Between")
    b = _entry("RF_001", "Within", "runA")
    c = _entry("RF_002", "Between")
    bw = state.get_mcc_registry_filtered(mode="Between")
    assert {e["name"] for e in bw} == {a["name"], c["name"]}
    ra = state.get_mcc_registry_filtered(run="runA")
    assert {e["name"] for e in ra} == {b["name"]}
    rf001 = state.get_mcc_registry_filtered(source_distmat="RF_001")
    assert {e["name"] for e in rf001} == {a["name"], b["name"]}


def test_mcc_delete_existing_and_missing():
    a = _entry()
    assert state.delete_mcc(a["name"]) is True
    assert state.delete_mcc(a["name"]) is False  # already gone
    assert state.delete_mcc("RF_999_Between_MCC_42") is False
    # Cache pop should run in lockstep with registry pop.
    assert not state.has_cached_mcc_tree(a["uuid"])


def test_mcc_lru_eviction_at_cap():
    """Push (cap + 1) entries and verify the oldest is evicted from
    BOTH the registry list AND the underlying MCC cache."""
    cap = state._MAX_MCC_REGISTRY
    entries = [_entry() for _ in range(cap)]
    assert len(state.get_mcc_registry()) == cap
    extra = _entry()
    assert len(state.get_mcc_registry()) == cap
    # Oldest entry gone from registry...
    names = {e["name"] for e in state.get_mcc_registry()}
    assert entries[0]["name"] not in names
    assert extra["name"] in names
    # ...and its NEXUS cache entry gone too (lockstep eviction).
    assert not state.has_cached_mcc_tree(entries[0]["uuid"])


def test_clear_all_mcc_trees_cascades_to_registry():
    _entry()
    _entry()
    assert len(state.get_mcc_registry()) == 2
    state.clear_all_mcc_trees()
    assert state.get_mcc_registry() == []
    # And counters are reset — next entry starts at _MCC_1 again.
    again = _entry()
    assert again["name"].endswith("_MCC_1"), again["name"]


def test_get_mcc_registry_entry_returns_none_on_unknown_uid():
    """Defensive lookup: callbacks pass a uuid received from the
    browser store, which may be stale (registry could have evicted
    or Cleared between the click and the callback). Must return
    ``None`` instead of raising."""
    assert state.get_mcc_registry_entry("nonexistent-uid-12345") is None
    _entry()
    assert state.get_mcc_registry_entry("still-not-there") is None


def test_registry_entry_stores_tree_names_as_defensive_copy():
    """The ``tree_names`` list passed to ``register_mcc`` shouldn't be
    aliased on the stored entry — caller mutations after registration
    must not leak into the registry."""
    names_in = ["selected/STATE_100", "selected/STATE_200"]
    uid = state.cache_mcc_tree(b"#NEXUS\nbegin trees;\nEnd;\n")
    e = state.register_mcc(
        source_distmat="RF_001", mode="Between", run=None,
        uuid=uid, mcc_tree={}, selection=[], log_clade_credibility=None,
        tree_names=names_in,
    )
    names_in.append("selected/STATE_DRIFT")
    assert "selected/STATE_DRIFT" not in e["tree_names"]
    assert len(e["tree_names"]) == 2


# ---- rename_mcc ------------------------------------------------------------

def test_register_mcc_starts_with_name_user_set_false():
    """The rename modal in callbacks/rename_mcc.py keys off this flag —
    new entries must default to "name still auto-generated"."""
    e = _entry()
    assert e["name_user_set"] is False


def test_rename_mcc_basic_by_name():
    a = _entry()
    entry, err = state.rename_mcc(a["name"], "custom_run3")
    assert err is None
    # In-place mutation — the entry's identity in the registry is stable.
    assert entry is a
    assert a["name"] == "custom_run3"
    assert a["name_user_set"] is True


def test_rename_mcc_by_uuid():
    a = _entry()
    entry, err = state.rename_mcc(a["uuid"], "via_uuid")
    assert err is None
    assert entry["name"] == "via_uuid"


def test_rename_mcc_strips_whitespace():
    a = _entry()
    entry, err = state.rename_mcc(a["name"], "  trimmed  ")
    assert err is None
    assert entry["name"] == "trimmed"


def test_rename_mcc_empty_rejected():
    a = _entry()
    for blank in ("", "   ", None):
        entry, err = state.rename_mcc(a["name"], blank)
        assert entry is None
        assert "empty" in err.lower()
    # Original name is untouched and flag stays False.
    assert a["name"] == "RF_001_Between_MCC_1"
    assert a["name_user_set"] is False


def test_rename_mcc_too_long_rejected():
    a = _entry()
    entry, err = state.rename_mcc(a["name"], "x" * 81)
    assert entry is None
    assert "long" in err.lower()


def test_rename_mcc_collision_rejected():
    a = _entry()
    b = _entry()
    entry, err = state.rename_mcc(a["name"], b["name"])
    assert entry is None
    assert "already" in err.lower()
    # Neither entry's name changed.
    assert a["name"] == "RF_001_Between_MCC_1"
    assert b["name"] == "RF_001_Between_MCC_2"


def test_rename_mcc_noop_still_sets_flag():
    """Even an unedited Save click should mark name_user_set so the
    View-flow modal stops prompting — the user explicitly confirmed."""
    a = _entry()
    entry, err = state.rename_mcc(a["name"], a["name"])
    assert err is None
    assert entry["name_user_set"] is True


def test_rename_mcc_unknown_id_returns_error():
    _entry()
    entry, err = state.rename_mcc("never_existed", "x")
    assert entry is None
    assert "not found" in err.lower()


def test_rename_mcc_persists_across_delete_lookup():
    """After rename, ``delete_mcc`` (which looks up by name) must still
    find the entry using the new name."""
    a = _entry()
    state.rename_mcc(a["name"], "renamed")
    assert state.delete_mcc("renamed") is True
    assert state.get_mcc_registry() == []
