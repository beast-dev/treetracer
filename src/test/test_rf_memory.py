#!/usr/bin/env python3
"""Test memory usage of Python RF distance computation at varying sample sizes.

Loads trees via the tree manager, samples at increasing sizes, and measures
peak memory consumed by rf_distance_from_newicks using tracemalloc.

Run:
    uv run python src/test/test_rf_memory.py
"""

import gc
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from treetracer.rf.rf import rf_distance_from_newicks, rf_distance_from_newick_iter
from treetracer.db.tree_manager import TreeManagerPandas
from treetracer.db.process_trees import process_nexus_trees_streaming


# ------------------------------------------------------------------
# Helpers
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


def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def find_test_file():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in ('test_BIG.trees', 'test.trees'):
        path = os.path.join(test_dir, candidate)
        if os.path.exists(path):
            return path
    return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    global passed, failed

    print("RF Distance Memory Usage Test")
    print("=" * 70)

    test_file = find_test_file()
    if not test_file:
        print("ERROR: No test .trees file found")
        return 1

    file_source = os.path.basename(test_file)

    # ------------------------------------------------------------------
    # Load trees
    # ------------------------------------------------------------------
    section("LOAD TREES")

    _stdout = sys.stdout
    sys.stdout = sys.stderr
    mgr = TreeManagerPandas()
    count = process_nexus_trees_streaming(test_file, mgr, file_source,
                                          batch_size=5000, transaction_size=20000)
    mgr.flush()
    sys.stdout = _stdout

    translate_map = mgr.get_translate_map(file_source)
    print(f"  File: {file_source}")
    print(f"  Trees: {count}")
    print(f"  Taxa: {len(translate_map)}")

    # ------------------------------------------------------------------
    # Measure memory at each sample size
    # ------------------------------------------------------------------
    sample_sizes = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]
    sample_sizes = [n for n in sample_sizes if n <= count]

    section("MEMORY MEASUREMENTS")

    results = []  # (n, pairs, elapsed, input_bytes, peak_bytes, matrix_bytes)

    for n in sample_sizes:
        trees = mgr.get_trees_sample(
            filters={'file_source': file_source}, limit=n, strategy='uniform',
        )
        names = [t['name'] for t in trees]
        newicks = [t['newick'] for t in trees]
        pairs = n * (n - 1) // 2

        # Measure input size (newick strings held in memory)
        input_bytes = sum(sys.getsizeof(nw) for nw in newicks)

        # Force GC before measurement for cleaner numbers
        gc.collect()

        tracemalloc.start()
        t0 = time.perf_counter()

        result_names, matrix = rf_distance_from_newicks(
            names, newicks, [translate_map],
        )

        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Estimate output matrix size (list of lists of ints)
        matrix_bytes = sys.getsizeof(matrix)
        for row in matrix:
            matrix_bytes += sys.getsizeof(row)
            # Each int in CPython is 28 bytes
            matrix_bytes += len(row) * 28

        results.append((n, pairs, elapsed, input_bytes, peak, matrix_bytes))

        print(f"  n={n:>5}  pairs={pairs:>10,}  time={elapsed:.3f}s  "
              f"peak={fmt_bytes(peak):>10}  "
              f"input={fmt_bytes(input_bytes):>10}  "
              f"matrix={fmt_bytes(matrix_bytes):>10}")

        # Checks
        check(len(result_names) == n, f"n={n}: got {len(result_names)} names")
        check(len(matrix) == n, f"n={n}: matrix is {n}x{n}")

        # Free results before next iteration
        del result_names, matrix
        gc.collect()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    section("SUMMARY TABLE")

    print(f"  {'N':>5}  {'Pairs':>10}  {'Time (s)':>9}  "
          f"{'Peak Mem':>10}  {'Input':>10}  {'Matrix':>10}  {'Overhead':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*9}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for n, pairs, elapsed, inp, peak, mat in results:
        overhead = peak - mat if peak > mat else 0
        print(f"  {n:>5}  {pairs:>10,}  {elapsed:>9.3f}  "
              f"{fmt_bytes(peak):>10}  {fmt_bytes(inp):>10}  "
              f"{fmt_bytes(mat):>10}  {fmt_bytes(overhead):>10}")

    # ------------------------------------------------------------------
    # Scaling checks
    # ------------------------------------------------------------------
    section("SCALING CHECKS")

    # Memory should scale roughly as O(n^2) for the distance matrix.
    # Verify peak memory doesn't blow up unexpectedly.
    if len(results) >= 2:
        first_n, _, _, _, first_peak, _ = results[0]
        last_n, _, _, _, last_peak, _ = results[-1]
        n_ratio = (last_n / first_n) ** 2
        mem_ratio = last_peak / first_peak if first_peak > 0 else 0
        check(mem_ratio < n_ratio * 3,
              f"Memory scales reasonably: n grew {last_n/first_n:.1f}x, "
              f"peak mem grew {mem_ratio:.1f}x "
              f"(expected ~{n_ratio:.0f}x for O(n^2))")

    # ------------------------------------------------------------------
    # Iterator vs list comparison
    # ------------------------------------------------------------------
    section("ITERATOR vs LIST COMPARISON")

    # Pick a few sample sizes for head-to-head comparison
    compare_sizes = [1000, 1500, 2000]
    compare_sizes = [n for n in compare_sizes if n <= count]

    for n in compare_sizes:
        # Get metadata only (no newicks yet) for fair measurement
        trees = mgr.get_trees_sample(
            filters={'file_source': file_source}, limit=n, strategy='uniform',
        )
        names = [t['name'] for t in trees]
        tree_ids = [t['id'] for t in trees]

        # --- List path: build newick list INSIDE tracemalloc window ---
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        newicks = list(mgr.iter_newicks(tree_ids))
        list_names, list_matrix = rf_distance_from_newicks(
            names, newicks, [translate_map],
        )
        list_elapsed = time.perf_counter() - t0
        _, list_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del newicks
        gc.collect()

        # --- Iterator path: newicks read lazily, never all in memory ---
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        iter_names, iter_matrix = rf_distance_from_newick_iter(
            names, mgr.iter_newicks(tree_ids), [translate_map],
        )
        iter_elapsed = time.perf_counter() - t0
        _, iter_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        saving = list_peak - iter_peak
        print(f"\n  n={n}:")
        print(f"    List path:     time={list_elapsed:.3f}s  peak={fmt_bytes(list_peak)}")
        print(f"    Iterator path: time={iter_elapsed:.3f}s  peak={fmt_bytes(iter_peak)}")
        print(f"    Saving:        {fmt_bytes(saving)}  ({saving / list_peak * 100:.1f}%)")

        # Verify identical results (iter_matrix is numpy uint32, list_matrix is list-of-lists)
        list_as_np = np.array(list_matrix, dtype=np.uint32)
        check(list_names == iter_names,
              f"n={n}: names match between list and iterator paths")
        check(np.array_equal(list_as_np, iter_matrix),
              f"n={n}: distance matrices match between list and iterator paths")
        check(iter_peak < list_peak,
              f"n={n}: iterator peak ({fmt_bytes(iter_peak)}) < list peak ({fmt_bytes(list_peak)})")

        del list_names, list_matrix, iter_names, iter_matrix
        gc.collect()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    section("RESULTS")

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
