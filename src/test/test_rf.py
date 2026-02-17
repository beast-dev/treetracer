#!/usr/bin/env python3
"""Self-contained tests for the Rust-backed RF distance module.

Uses test_BIG.trees (or test.trees as fallback) from the db/ directory.
Tests both the newick path and the file path, validates correctness,
and prints a timing summary.

Run:
    uv run python src/test/test_rf.py
"""

import os
import re
import sys
import tempfile
import time

# Ensure package imports work when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from treetracer.rf.rf import (
    rf_distance_from_newicks,
    rf_distance_from_file,
    matrix_to_numpy,
    matrix_to_dict,
)
from treetracer.db.tree_manager import TreeManagerPandas
from treetracer.db.process_trees import process_nexus_trees_streaming

# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------

passed = 0
failed = 0


def check(condition, description):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {description}")
    else:
        failed += 1
        print(f"  FAIL: {description}")


def section(title):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def extract_state_id(name):
    """Pull STATE_XXXX from a possibly-prefixed tree name."""
    m = re.search(r'(STATE_\d+)', name)
    return m.group(1) if m else name


# ------------------------------------------------------------------
# Locate test data
# ------------------------------------------------------------------

def find_test_file():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in ('test_BIG.trees', 'test.trees'):
        path = os.path.join(test_dir, candidate)
        if os.path.exists(path):
            return path
    return None


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_basic_file_path(test_file):
    """RF via file path — basic structural checks."""
    section("TEST: rf_distance_from_file — basic")
    names, matrix = rf_distance_from_file(test_file, burnin_trees=100)
    n = len(names)
    check(n > 0, f"Got {n} trees from file")
    check(len(matrix) == n, f"Matrix has {n} rows")
    check(all(len(row) == n for row in matrix), "All rows have correct length")
    check(all(matrix[i][i] == 0 for i in range(n)), "Diagonal is zero")
    check(
        all(matrix[i][j] == matrix[j][i] for i in range(n) for j in range(i + 1, n)),
        "Matrix is symmetric",
    )
    check(
        all(matrix[i][j] >= 0 for i in range(n) for j in range(n)),
        "All values non-negative",
    )
    return names, matrix


def test_helpers(names, matrix):
    """matrix_to_numpy and matrix_to_dict helpers."""
    section("TEST: helper functions")
    arr = matrix_to_numpy(matrix)
    check(arr.shape == (len(names), len(names)), f"numpy shape {arr.shape}")
    check(arr.dtype == 'int32', f"numpy dtype {arr.dtype}")

    d = matrix_to_dict(names, matrix)
    check(len(d) == len(names), f"dict has {len(d)} entries")
    # spot-check symmetry via dict
    n0, n1 = names[0], names[1]
    check(d[n0][n1] == d[n1][n0], "dict lookup symmetric")


def test_newick_vs_file(test_file, mgr, file_source, sample_sizes):
    """Cross-validate newick path against file path at varying sizes."""
    section("TEST: newick path vs file path cross-validation")

    translate_map = mgr.get_translate_map(file_source)
    check(translate_map is not None and len(translate_map) > 0,
          f"Translate map has {len(translate_map or {})} entries")

    timing_results = []
    temp_files = []

    for n in sample_sizes:
        print(f"\n  --- n = {n} ---")
        trees = mgr.get_trees_sample(
            filters={'file_source': file_source},
            limit=n,
            strategy='uniform',
        )
        actual = len(trees)
        check(actual == n, f"Sampled {actual} trees (expected {n})")
        if actual < 2:
            print("  SKIP: need >= 2 trees")
            continue

        names_in = [t['name'] for t in trees]
        newicks_in = [t['newick'] for t in trees]

        # Path 1: newick
        t0 = time.perf_counter()
        names1, mat1 = rf_distance_from_newicks(
            names_in, newicks_in, [translate_map],
        )
        newick_time = time.perf_counter() - t0
        check(len(names1) == actual, f"Newick path: {actual}x{actual} in {newick_time:.3f}s")

        # Path 2: export → file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.trees', prefix=f'rf_test_n{n}_')
        os.close(tmp_fd)
        temp_files.append(tmp_path)

        t0 = time.perf_counter()
        mgr.export_trees_nexus(tmp_path, trees)
        export_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        names2, mat2 = rf_distance_from_file(tmp_path)
        file_rf_time = time.perf_counter() - t0
        file_total = export_time + file_rf_time
        check(len(names2) == actual, f"File path:   {actual}x{actual} in {file_total:.3f}s")

        timing_results.append((n, newick_time, export_time, file_rf_time, file_total))

        # Structural checks on newick result
        check(all(mat1[i][i] == 0 for i in range(actual)), "Diagonal zero")
        check(
            all(mat1[i][j] == mat1[j][i]
                for i in range(actual) for j in range(i + 1, actual)),
            "Symmetric",
        )
        check(
            all(mat1[i][j] >= 0 and mat1[i][j] % 2 == 0
                for i in range(actual) for j in range(actual)),
            "Non-negative even integers",
        )

        # Element-wise comparison between paths
        idx1 = {extract_state_id(nm): i for i, nm in enumerate(names1)}
        idx2 = {extract_state_id(nm): i for i, nm in enumerate(names2)}
        common = sorted(set(idx1) & set(idx2))
        check(len(common) == actual, f"All {len(common)} STATE ids in both paths")

        mismatches = 0
        for sa in common:
            for sb in common:
                if mat1[idx1[sa]][idx1[sb]] != mat2[idx2[sa]][idx2[sb]]:
                    mismatches += 1
        check(mismatches == 0, f"All {actual*actual} pairwise values match")

    # Cleanup temp files
    for p in temp_files:
        try:
            os.unlink(p)
        except OSError:
            pass

    # Timing summary
    if timing_results:
        section("TIMING SUMMARY")
        print(f"  {'N':>5}  {'Newick (s)':>10}  {'File Total (s)':>14}  "
              f"{'Export (s)':>10}  {'File RF (s)':>11}")
        print(f"  {'-----':>5}  {'----------':>10}  {'--------------':>14}  "
              f"{'----------':>10}  {'-----------':>11}")
        for n, nwk_t, exp_t, frf_t, ftot_t in timing_results:
            print(f"  {n:>5}  {nwk_t:>10.3f}  {ftot_t:>14.3f}  "
                  f"{exp_t:>10.3f}  {frf_t:>11.3f}")


def test_rooted_vs_unrooted(mgr, file_source):
    """Rooted RF >= unrooted RF for every pair."""
    section("TEST: rooted vs unrooted")
    translate_map = mgr.get_translate_map(file_source)
    trees = mgr.get_trees_sample(
        filters={'file_source': file_source}, limit=50, strategy='uniform',
    )
    names_in = [t['name'] for t in trees]
    newicks_in = [t['newick'] for t in trees]

    _, mat_unrooted = rf_distance_from_newicks(names_in, newicks_in, [translate_map], rooted=False)
    _, mat_rooted = rf_distance_from_newicks(names_in, newicks_in, [translate_map], rooted=True)

    n = len(names_in)
    all_ge = all(
        mat_rooted[i][j] >= mat_unrooted[i][j]
        for i in range(n) for j in range(i + 1, n)
    )
    check(all_ge, f"Rooted RF >= unrooted RF for all {n*(n-1)//2} pairs")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    global passed, failed

    print("RF Module Tests (Rust-backed)")
    print("=" * 70)

    test_file = find_test_file()
    if not test_file:
        print("ERROR: No test .trees file found in db/ directory")
        return 1

    file_source = os.path.basename(test_file)
    print(f"Using: {file_source}")

    # Load trees into a fresh manager
    section("LOAD TREES")
    mgr = TreeManagerPandas()
    count = process_nexus_trees_streaming(test_file, mgr, file_source,
                                          batch_size=500, transaction_size=1000)
    mgr.flush()
    print(f"  Loaded {count} trees")

    # Run tests
    names, matrix = test_basic_file_path(test_file)
    test_helpers(names, matrix)

    sample_sizes = [100, 500, 1000]
    sample_sizes = [s for s in sample_sizes if s <= count]
    test_newick_vs_file(test_file, mgr, file_source, sample_sizes)

    test_rooted_vs_unrooted(mgr, file_source)

    # Cleanup
    mgr.cleanup()

    # Summary
    section("SUMMARY")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    if failed:
        print(f"\n  {failed} TEST(S) FAILED")
        return 1
    print(f"\n  ALL {passed} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
