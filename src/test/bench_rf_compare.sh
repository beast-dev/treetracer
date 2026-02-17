#!/usr/bin/env bash
# Compare RF distance performance AND correctness: Python vs R (phangorn).
#
# Both tools compute on the SAME exported .trees files so distances are
# directly comparable.  Produces a timing table and a scatter plot of
# R-distances vs Python-distances.
#
# Usage:
#   bash src/test/bench_rf_compare.sh [trees_file]
#
# Defaults to test_BIG.trees in the same directory as this script.
# Expects:
#   - Rscript from mamba env "h5n1" (override with RSCRIPT env var)
#   - uv-managed treetracer package with rust_python_tree_distances

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TREES_FILE="${1:-$SCRIPT_DIR/test_BIG.trees}"

if [[ ! -f "$TREES_FILE" ]]; then
    echo "ERROR: Trees file not found: $TREES_FILE" >&2
    exit 1
fi

RSCRIPT="${RSCRIPT:-/Users/samuelhong/mamba/envs/h5n1/bin/Rscript}"
if [[ ! -x "$RSCRIPT" ]]; then
    echo "ERROR: Rscript not found at $RSCRIPT" >&2
    echo "  Set RSCRIPT env var or activate the h5n1 mamba env" >&2
    exit 1
fi

SAMPLE_SIZES="100 250 500 1000 2000 3000"
TMPDIR=$(mktemp -d /tmp/bench_rf_XXXXXX)
PLOT_FILE="$SCRIPT_DIR/rf_R_vs_Python.png"

cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

echo "============================================================"
echo " RF Distance Benchmark: Python vs R (phangorn)"
echo "============================================================"
echo ""
echo "Trees file:    $TREES_FILE"
echo "Sample sizes:  $SAMPLE_SIZES"
echo "Temp dir:      $TMPDIR"
echo "Plot output:   $PLOT_FILE"
echo ""

# ------------------------------------------------------------------
# Phase 1: Export sampled trees to temp .trees files
# ------------------------------------------------------------------
echo "============================================================"
echo " Phase 1: Exporting sampled trees"
echo "============================================================"

cd "$PROJECT_DIR"

uv run python -c "
import sys, os
sys.path.insert(0, os.path.join('$PROJECT_DIR', 'src'))
from treetracer.db.tree_manager import TreeManagerPandas
from treetracer.db.process_trees import process_nexus_trees_streaming

trees_file = '$TREES_FILE'
sample_sizes = [int(x) for x in '$SAMPLE_SIZES'.split()]
tmpdir = '$TMPDIR'
file_source = os.path.basename(trees_file)

# Redirect process_trees prints to stderr
_real_stdout = sys.stdout
sys.stdout = sys.stderr
mgr = TreeManagerPandas()
count = process_nexus_trees_streaming(trees_file, mgr, file_source,
                                      batch_size=5000, transaction_size=20000)
sys.stdout = _real_stdout
mgr.flush()

for n in sample_sizes:
    if n > count:
        sys.stderr.write(f'Skipping n={n} (only {count} trees)\n')
        continue
    trees = mgr.get_trees_sample(
        filters={'file_source': file_source}, limit=n, strategy='uniform')
    out_path = os.path.join(tmpdir, f'sample_{n}.trees')
    written = mgr.export_trees_nexus(out_path, trees)
    sys.stderr.write(f'  Exported {written} trees -> sample_{n}.trees\n')

mgr.cleanup()
" 2>&1

echo ""

# ------------------------------------------------------------------
# Phase 2: R benchmark on each exported file
# ------------------------------------------------------------------
echo "============================================================"
echo " Phase 2: R / phangorn"
echo "============================================================"

for n in $SAMPLE_SIZES; do
    sample_file="$TMPDIR/sample_${n}.trees"
    if [[ ! -f "$sample_file" ]]; then
        echo "  Skipping n=$n (no exported file)"
        continue
    fi
    r_out="$TMPDIR/r_${n}.txt"
    echo "  n=$n ..."
    "$RSCRIPT" "$SCRIPT_DIR/bench_rf_R.R" "$sample_file" "$r_out"
done

echo ""

# ------------------------------------------------------------------
# Phase 3: Python benchmark on each exported file
# ------------------------------------------------------------------
echo "============================================================"
echo " Phase 3: Python (treetracer.rf)"
echo "============================================================"

cd "$PROJECT_DIR"

uv run python -c "
import sys, os, time
sys.path.insert(0, os.path.join('$PROJECT_DIR', 'src'))
from treetracer.rf.rf import rf_distance_from_file

tmpdir = '$TMPDIR'
sample_sizes = [int(x) for x in '$SAMPLE_SIZES'.split()]

for n in sample_sizes:
    sample_file = os.path.join(tmpdir, f'sample_{n}.trees')
    if not os.path.exists(sample_file):
        continue
    py_out = os.path.join(tmpdir, f'python_{n}.txt')

    sys.stderr.write(f'  n={n} ...\n')
    t0 = time.perf_counter()
    names, matrix = rf_distance_from_file(sample_file)
    elapsed = time.perf_counter() - t0
    sys.stderr.write(f'    {elapsed:.3f}s\n')

    # Write same format as R: line 1 = time, then upper triangle row-major
    nn = len(names)
    with open(py_out, 'w') as f:
        f.write(f'{elapsed:.6f}\n')
        for i in range(nn):
            for j in range(i + 1, nn):
                f.write(f'{matrix[i][j]}\n')
" 2>&1

echo ""

# ------------------------------------------------------------------
# Phase 4: Compare distances + timing table + scatter plot
# ------------------------------------------------------------------
echo "============================================================"
echo " Phase 4: Results"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"

uv run python -c "
import os, sys
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

tmpdir = '$TMPDIR'
plot_file = '$PLOT_FILE'
sample_sizes = [int(x) for x in '$SAMPLE_SIZES'.split()]

# Collect timing and distances
rows = []          # (n, r_time, py_time, pairs, exact, off_by_2)
all_r_dists = []   # flat list across sizes
all_py_dists = []
all_diffs = []     # python - r

for n in sample_sizes:
    r_file = os.path.join(tmpdir, f'r_{n}.txt')
    py_file = os.path.join(tmpdir, f'python_{n}.txt')
    if not os.path.exists(r_file) or not os.path.exists(py_file):
        continue

    with open(r_file) as f:
        lines = f.read().strip().split('\n')
    r_time = float(lines[0])
    r_dists = [int(x) for x in lines[1:]]

    with open(py_file) as f:
        lines = f.read().strip().split('\n')
    py_time = float(lines[0])
    py_dists = [int(x) for x in lines[1:]]

    expected_pairs = n * (n - 1) // 2
    if len(r_dists) != expected_pairs or len(py_dists) != expected_pairs:
        print(f'  WARNING n={n}: R has {len(r_dists)} pairs, '
              f'Python has {len(py_dists)} pairs, expected {expected_pairs}')
        continue

    diffs = [b - a for a, b in zip(r_dists, py_dists)]
    diff_counts = Counter(diffs)
    exact = diff_counts.get(0, 0)
    off_by_2 = diff_counts.get(2, 0)
    other = expected_pairs - exact - off_by_2

    rows.append((n, r_time, py_time, expected_pairs, exact, off_by_2, other))
    all_r_dists.extend(r_dists)
    all_py_dists.extend(py_dists)
    all_diffs.extend(diffs)

# --- Difference analysis ---
diff_counts = Counter(all_diffs)
print('Difference distribution (Python - R):')
for d in sorted(diff_counts):
    pct = 100 * diff_counts[d] / len(all_diffs)
    print(f'  {d:+d}: {diff_counts[d]:>8} ({pct:.1f}%)')

total_pairs = len(all_r_dists)
exact_total = diff_counts.get(0, 0)
off2_total = diff_counts.get(2, 0)
print(f'\nTotal: {total_pairs:,} pairs')
print(f'  Exact match (diff=0): {exact_total:,} ({100*exact_total/total_pairs:.1f}%)')
print(f'  Off by +2:            {off2_total:,} ({100*off2_total/total_pairs:.1f}%)')
if exact_total + off2_total == total_pairs:
    print('  All differences are 0 or +2 (root bipartition)')

# --- Timing table ---
print(f\"\n{'N':>5}  {'R (s)':>8}  {'Python (s)':>10}  {'Speedup':>8}  {'Pairs':>8}  {'Exact':>8}  {'Off +2':>8}  {'Other':>6}\")
print(f\"{'-----':>5}  {'--------':>8}  {'----------':>10}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'------':>6}\")
for n, rt, pyt, pairs, exact, off2, other in rows:
    sp = rt / pyt if pyt > 0 else 0
    print(f'{n:>5}  {rt:>8.3f}  {pyt:>10.3f}  {sp:>7.1f}x  {pairs:>8}  {exact:>8}  {off2:>8}  {other:>6}')

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: scatter of distances
ax = axes[0]
ax.scatter(all_r_dists, all_py_dists, s=0.5, alpha=0.15, c='steelblue', edgecolors='none')
lo = min(min(all_r_dists), min(all_py_dists))
hi = max(max(all_r_dists), max(all_py_dists))
ax.plot([lo, hi], [lo, hi], 'r-', linewidth=1, label='y = x')
ax.plot([lo, hi], [lo+2, hi+2], 'r--', linewidth=0.8, alpha=0.5, label='y = x + 2')
ax.set_xlabel('R phangorn RF distance')
ax.set_ylabel('Python RF distance')
ax.set_title(f'RF Distances ({total_pairs:,} pairs)')
ax.legend(loc='upper left', fontsize=9)
ax.set_aspect('equal')

# Panel 2: histogram of differences
ax2 = axes[1]
unique_diffs = sorted(diff_counts.keys())
ax2.bar(unique_diffs, [diff_counts[d] for d in unique_diffs],
        color=['tab:green' if d == 0 else 'tab:orange' for d in unique_diffs],
        edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Python - R difference')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of differences')
ax2.set_xticks(unique_diffs)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k' if x >= 1e3 else f'{x:.0f}'))
for d in unique_diffs:
    pct = 100 * diff_counts[d] / total_pairs
    ax2.annotate(f'{pct:.1f}%', (d, diff_counts[d]),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 3: timing comparison
ns     = [r[0] for r in rows]
r_ts   = [r[1] for r in rows]
py_ts  = [r[2] for r in rows]
ax3 = axes[2]
ax3.plot(ns, r_ts,  'o-', color='tab:orange', label='R phangorn', markersize=6)
ax3.plot(ns, py_ts, 's-', color='tab:blue',   label='Python',     markersize=6)
ax3.set_xlabel('Number of trees')
ax3.set_ylabel('Time (seconds)')
ax3.set_title('RF Computation Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_file, dpi=150)
print(f'\nPlot saved to {plot_file}')
"
