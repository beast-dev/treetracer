#!/usr/bin/env python3
"""Measure total process memory (Python + Rust) for RF distance computation.

Uses subprocess isolation: each (mode, n) combination runs in a fresh process
so that resource.getrusage(RUSAGE_SELF).ru_maxrss gives an accurate peak RSS
for that test case alone.

Run:
    uv run python src/test/test_rf_process_memory.py
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc


def get_peak_rss_bytes():
    """Return peak RSS of this process in bytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return ru  # macOS reports bytes
    return ru * 1024  # Linux reports KB


def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


# ======================================================================
# Worker — runs in a subprocess
# ======================================================================

def worker(test_file, mode, n):
    """Run a single RF test case and print JSON results to stdout."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from treetracer.db.tree_manager import TreeManagerPandas
    from treetracer.db.process_trees import process_nexus_trees_streaming
    from treetracer.rf.rf import (
        rf_distance_from_newicks,
        rf_distance_from_newick_iter,
        rf_distance_from_newick_iter_chunked,
    )

    file_source = os.path.basename(test_file)

    # Load trees (suppress stdout noise)
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    mgr = TreeManagerPandas()
    total = process_nexus_trees_streaming(
        test_file, mgr, file_source, batch_size=5000, transaction_size=20000,
    )
    mgr.flush()
    sys.stdout = old_stdout

    translate_map = mgr.get_translate_map(file_source)

    trees = mgr.get_trees_sample(
        filters={"file_source": file_source}, limit=n, strategy="uniform",
    )
    names = [t["name"] for t in trees]
    tree_ids = [t["id"] for t in trees]

    # Baseline peak RSS after loading (before computation)
    rss_before = get_peak_rss_bytes()

    import gc
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    if mode == "list":
        newicks = list(mgr.iter_newicks(tree_ids))
        result_names, matrix = rf_distance_from_newicks(
            names, newicks, [translate_map],
        )
    elif mode == "iter":
        result_names, matrix = rf_distance_from_newick_iter(
            names, mgr.iter_newicks(tree_ids), [translate_map],
        )
    elif mode.startswith("chunked"):
        # e.g. "chunked500" → chunk_size=500
        chunk_size = int(mode.replace("chunked", "")) if mode != "chunked" else 1000
        result_names, matrix = rf_distance_from_newick_iter_chunked(
            names, mgr.iter_newicks, tree_ids, [translate_map],
            chunk_size=chunk_size,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    elapsed = time.perf_counter() - t0
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = get_peak_rss_bytes()

    # Checksum for correctness comparison
    import numpy as np
    if isinstance(matrix, np.ndarray):
        checksum = int(matrix.sum())
    else:
        checksum = sum(sum(row) for row in matrix)

    result = {
        "mode": mode,
        "n": n,
        "elapsed": elapsed,
        "tracemalloc_peak": tracemalloc_peak,
        "rss_before": rss_before,
        "rss_after": rss_after,
        "rss_delta": rss_after - rss_before,
        "checksum": checksum,
    }
    # Print JSON to stdout for parent to capture
    print(json.dumps(result), flush=True)


# ======================================================================
# Orchestrator — spawns subprocesses
# ======================================================================

def run_worker(test_file, mode, n):
    """Spawn a subprocess worker and return parsed JSON result."""
    cmd = [
        sys.executable, __file__,
        "--worker", test_file, mode, str(n),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"  ERROR (mode={mode}, n={n}): {proc.stderr.strip()}", file=sys.stderr)
        return None
    # Last non-empty line of stdout is our JSON
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line:
            return json.loads(line)
    return None


def find_test_file():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in ("test_BIG.trees", "test.trees"):
        path = os.path.join(test_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def main():
    print("RF Distance — Total Process Memory (Python + Rust)")
    print("=" * 80)

    # Handle --worker invocation
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        test_file, mode, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
        worker(test_file, mode, n)
        return 0

    test_file = find_test_file()
    if not test_file:
        print("ERROR: No test .trees file found")
        return 1

    print(f"  File: {os.path.basename(test_file)}")
    print(f"  Each (mode, n) runs in a fresh subprocess for accurate RSS.\n")

    sample_sizes = [1000, 1500, 2000]
    modes = ["list", "iter", "chunked500"]

    # Collect results: results[(mode, n)] = {...}
    results = {}

    for n in sample_sizes:
        for mode in modes:
            label = f"mode={mode:<12}  n={n}"
            print(f"  Running {label} ...", end="", flush=True)
            t0 = time.perf_counter()
            r = run_worker(test_file, mode, n)
            wall = time.perf_counter() - t0
            if r is None:
                print(f"  FAILED")
                continue
            results[(mode, n)] = r
            print(f"  done in {wall:.1f}s  "
                  f"(compute={r['elapsed']:.1f}s  "
                  f"peak_rss={fmt_bytes(r['rss_after'])}  "
                  f"tracemalloc={fmt_bytes(r['tracemalloc_peak'])})")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f" COMPARISON: List vs Iterator vs Chunked")
    print(f"{'=' * 80}\n")

    print(f"  {'N':>5}  {'Mode':<12}  {'Time (s)':>9}  "
          f"{'Peak RSS':>10}  {'RSS delta':>10}  "
          f"{'tracemalloc':>12}  {'Checksum':>12}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*9}  "
          f"{'-'*10}  {'-'*10}  "
          f"{'-'*12}  {'-'*12}")

    for n in sample_sizes:
        for mode in modes:
            r = results.get((mode, n))
            if r is None:
                continue
            print(f"  {n:>5}  {mode:<12}  {r['elapsed']:>9.3f}  "
                  f"{fmt_bytes(r['rss_after']):>10}  "
                  f"{fmt_bytes(r['rss_delta']):>10}  "
                  f"{fmt_bytes(r['tracemalloc_peak']):>12}  "
                  f"{r['checksum']:>12}")
        # Print savings vs list
        lr = results.get(("list", n))
        for alt in ["iter", "chunked500"]:
            ar = results.get((alt, n))
            if lr and ar:
                rss_saving = lr["rss_after"] - ar["rss_after"]
                print(f"  {'':>5}  {alt + ':':>12}  {'':>9}  "
                      f"{'saving':>10}  "
                      f"{fmt_bytes(rss_saving):>10}  "
                      f"({rss_saving / lr['rss_after'] * 100:.1f}% of list)")
        print()

    # ------------------------------------------------------------------
    # Correctness checks
    # ------------------------------------------------------------------
    print(f"{'=' * 80}")
    print(f" CORRECTNESS CHECKS")
    print(f"{'=' * 80}\n")

    checks_passed = 0
    checks_failed = 0
    for n in sample_sizes:
        lr = results.get(("list", n))
        for alt in ["iter", "chunked500"]:
            ar = results.get((alt, n))
            if not (lr and ar):
                continue

            if lr["checksum"] == ar["checksum"]:
                checks_passed += 1
                print(f"  PASS: n={n}: {alt} checksum matches list ({lr['checksum']})")
            else:
                checks_failed += 1
                print(f"  FAIL: n={n}: {alt} checksum {ar['checksum']} "
                      f"!= list {lr['checksum']}")

            if ar["rss_after"] < lr["rss_after"]:
                checks_passed += 1
                print(f"  PASS: n={n}: {alt} RSS ({fmt_bytes(ar['rss_after'])}) "
                      f"< list RSS ({fmt_bytes(lr['rss_after'])})")
            else:
                checks_failed += 1
                print(f"  FAIL: n={n}: {alt} RSS ({fmt_bytes(ar['rss_after'])}) "
                      f">= list RSS ({fmt_bytes(lr['rss_after'])})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_checks = checks_passed + checks_failed
    print(f"\n{'=' * 80}")
    print(f" RESULTS")
    print(f"{'=' * 80}\n")
    print(f"  Passed: {checks_passed}/{total_checks}")
    print(f"  Failed: {checks_failed}/{total_checks}")
    if checks_failed:
        print(f"\n  {checks_failed} CHECK(S) FAILED")
        return 1
    if total_checks > 0:
        print(f"\n  ALL {checks_passed} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
