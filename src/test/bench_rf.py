#!/usr/bin/env python3
"""Benchmark time and memory for RF distance computation at varying sample sizes.

Run:
    uv run python src/test/bench_rf.py
"""

import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from treetracer.rf.rf import rf_distance_from_newicks
from treetracer.db.tree_manager import TreeManagerPandas
from treetracer.db.process_trees import process_nexus_trees_streaming


def find_test_file():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in ('test_BIG.trees', 'test.trees'):
        path = os.path.join(test_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def main():
    sample_sizes = [500, 1000, 1500, 2000]

    test_file = find_test_file()
    if not test_file:
        print("ERROR: No test .trees file found in db/ directory")
        return 1

    file_source = os.path.basename(test_file)
    print(f"Loading trees from {file_source}...")

    mgr = TreeManagerPandas()
    count = process_nexus_trees_streaming(test_file, mgr, file_source,
                                          batch_size=500, transaction_size=1000)
    mgr.flush()
    print(f"Loaded {count} trees ({len(mgr.get_translate_map(file_source))} taxa)\n")

    sample_sizes = [s for s in sample_sizes if s <= count]
    translate_map = mgr.get_translate_map(file_source)

    results = []

    for n in sample_sizes:
        trees = mgr.get_trees_sample(
            filters={'file_source': file_source}, limit=n, strategy='uniform',
        )
        names = [t['name'] for t in trees]
        newicks = [t['newick'] for t in trees]
        pairs = n * (n - 1) // 2

        tracemalloc.start()
        t0 = time.perf_counter()

        rf_distance_from_newicks(names, newicks, [translate_map])

        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append((n, pairs, elapsed, current, peak))
        print(f"n={n:>5}  pairs={pairs:>10,}  time={elapsed:.3f}s  "
              f"mem_current={fmt_bytes(current)}  mem_peak={fmt_bytes(peak)}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"{'N':>6}  {'Pairs':>10}  {'Time (s)':>9}  {'Pairs/s':>10}  "
          f"{'Cur Mem':>10}  {'Peak Mem':>10}")
    print(f"{'-'*6}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*10}")
    for n, pairs, elapsed, cur, peak in results:
        throughput = pairs / elapsed if elapsed > 0 else 0
        print(f"{n:>6}  {pairs:>10,}  {elapsed:>9.3f}  {throughput:>10,.0f}  "
              f"{fmt_bytes(cur):>10}  {fmt_bytes(peak):>10}")
    print(f"{'=' * 80}")

    mgr.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
