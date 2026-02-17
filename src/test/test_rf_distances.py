#!/usr/bin/env python3
"""End-to-end test & benchmark: compute RF distances via newick strings and via
.trees file export, verify identical results, and compare timing.

Loads trees from a test .trees file using the tree_manager infrastructure,
subsamples at increasing sizes (100 → 3000), then for each size:
  Path 1 (newick):  RF.rf_distance_from_newicks(names, newicks, translate_map)
  Path 2 (file):    export to temp .trees file -> RF.rf_distance_from_file(path)
Compares both matrices element-wise and reports wall-clock timing.
"""

import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from treetracer.db.tree_manager import TreeManagerPandas
from treetracer.db.process_trees import process_nexus_trees_streaming
from treetracer.rf import rf as RF

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


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}")


def extract_state_id(name):
    """Extract STATE_XXXX from a possibly-prefixed tree name."""
    m = re.search(r'(STATE_\d+)', name)
    return m.group(1) if m else name


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    global passed, failed

    print("RF Distance Cross-Validation & Benchmark")
    print("=" * 70)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "test_BIG.trees")
    if not os.path.exists(test_file):
        test_file = os.path.join(current_dir, "test.trees")
        if not os.path.exists(test_file):
            print("No test .trees file found (need test_BIG.trees or test.trees)")
            return 1

    file_source = os.path.basename(test_file)

    # ------------------------------------------------------------------
    # 1. Load trees
    # ------------------------------------------------------------------
    print_section("1. LOAD TREES")

    mgr = TreeManagerPandas()
    count = process_nexus_trees_streaming(test_file, mgr, file_source,
                                          batch_size=500, transaction_size=1000)
    mgr.flush()
    print(f"  Loaded {count} trees from {file_source}")

    translate_map = mgr.get_translate_map(file_source)
    check(translate_map is not None and len(translate_map) > 0,
          f"Translate map has {len(translate_map or {})} entries")

    # ------------------------------------------------------------------
    # 2. Test & benchmark at increasing sample sizes
    # ------------------------------------------------------------------
    sample_sizes = [100, 250, 500, 750, 1000, 1500]#, 2000, 2500, 3000]
    # Cap to available trees
    sample_sizes = [n for n in sample_sizes if n <= count]

    temp_files = []
    timing_results = []  # (n, newick_time, export_time, file_rf_time, file_total_time)

    for n in sample_sizes:
        print_section(f"SAMPLE SIZE = {n}")

        # Sample trees
        trees = mgr.get_trees_sample(
            filters={'file_source': file_source},
            limit=n,
            strategy='uniform',
        )
        actual_n = len(trees)
        check(actual_n == n, f"Sampled {actual_n} trees (expected {n})")
        if actual_n < 2:
            print("  SKIP: need at least 2 trees")
            continue

        names = [t['name'] for t in trees]
        newicks = [t['newick'] for t in trees]

        # --- Path 1: newick-based RF (timed) ---
        t0 = time.perf_counter()
        names1, matrix1 = RF.rf_distance_from_newicks(names, newicks, [translate_map])
        newick_time = time.perf_counter() - t0

        check(len(names1) == actual_n and len(matrix1) == actual_n,
              f"Path 1 (newick): {actual_n}x{actual_n} matrix in {newick_time:.3f}s")

        # --- Path 2: export + file-based RF (timed separately) ---
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.trees', prefix=f'rf_test_n{n}_')
        os.close(tmp_fd)
        temp_files.append(tmp_path)

        t0 = time.perf_counter()
        written = mgr.export_trees_nexus(tmp_path, trees)
        export_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        names2, matrix2 = RF.rf_distance_from_file(tmp_path)
        file_rf_time = time.perf_counter() - t0

        file_total_time = export_time + file_rf_time

        check(len(names2) == actual_n and len(matrix2) == actual_n,
              f"Path 2 (file):   {actual_n}x{actual_n} matrix in {file_total_time:.3f}s "
              f"(export {export_time:.3f}s + RF {file_rf_time:.3f}s)")

        timing_results.append((n, newick_time, export_time, file_rf_time, file_total_time))

        # --- Structural checks ---
        diag_zero = all(matrix1[i][i] == 0 for i in range(actual_n))
        check(diag_zero, "Diagonal is all zeros")

        symmetric = all(
            matrix1[i][j] == matrix1[j][i]
            for i in range(actual_n)
            for j in range(i + 1, actual_n)
        )
        check(symmetric, "Matrix is symmetric")

        all_nonneg_even = all(
            matrix1[i][j] >= 0 and matrix1[i][j] % 2 == 0
            for i in range(actual_n)
            for j in range(actual_n)
        )
        check(all_nonneg_even, "All RF values are non-negative even integers")

        # --- Cross-path comparison ---
        idx1 = {extract_state_id(nm): i for i, nm in enumerate(names1)}
        idx2 = {extract_state_id(nm): i for i, nm in enumerate(names2)}

        common_states = sorted(set(idx1) & set(idx2))
        check(len(common_states) == actual_n,
              f"All {len(common_states)} STATE ids found in both paths")

        mismatches = 0
        compared = 0
        for sa in common_states:
            for sb in common_states:
                i1, j1 = idx1[sa], idx1[sb]
                i2, j2 = idx2[sa], idx2[sb]
                compared += 1
                if matrix1[i1][j1] != matrix2[i2][j2]:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"    MISMATCH: ({sa}, {sb}): "
                              f"newick={matrix1[i1][j1]} vs file={matrix2[i2][j2]}")

        check(mismatches == 0,
              f"All {compared} pairwise RF values match (0 mismatches)")

    # ------------------------------------------------------------------
    # Timing summary table
    # ------------------------------------------------------------------
    print_section("TIMING SUMMARY")
    print(f"  {'N':>5}  {'Newick (s)':>10}  {'File Total (s)':>14}  "
          f"{'Export (s)':>10}  {'File RF (s)':>11}  {'Speedup':>8}")
    print(f"  {'-----':>5}  {'----------':>10}  {'--------------':>14}  "
          f"{'----------':>10}  {'-----------':>11}  {'--------':>8}")
    for n, nwk_t, exp_t, frf_t, ftot_t in timing_results:
        if nwk_t > 0:
            speedup = ftot_t / nwk_t
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        print(f"  {n:>5}  {nwk_t:>10.3f}  {ftot_t:>14.3f}  "
              f"{exp_t:>10.3f}  {frf_t:>11.3f}  {speedup_str:>8}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print_section("CLEANUP")
    for tmp_path in temp_files:
        try:
            os.unlink(tmp_path)
            print(f"  Deleted {os.path.basename(tmp_path)}")
        except OSError as e:
            print(f"  Failed to delete {tmp_path}: {e}")

    mgr.cleanup()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_section("SUMMARY")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")

    if failed > 0:
        print(f"\n  {failed} TEST(S) FAILED")
        return 1
    else:
        print(f"\n  ALL {passed} TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
