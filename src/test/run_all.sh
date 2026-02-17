#!/usr/bin/env bash
# Run all TreeTracer test scripts and report results.
#
# Usage:
#   bash src/test/run_all.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

tests=(
    src/test/test_trees_pandas.py
    src/test/test_export_trees.py
    src/test/test_rf.py
    src/test/test_rf_distances.py
)

passed=0
failed=0
failed_names=()

for test in "${tests[@]}"; do
    echo ""
    echo "============================================================"
    echo " Running: $test"
    echo "============================================================"
    if uv run python "$test"; then
        ((passed++))
    else
        ((failed++))
        failed_names+=("$test")
    fi
done

echo ""
echo "============================================================"
echo " ALL TESTS SUMMARY"
echo "============================================================"
echo "  Passed: $passed/${#tests[@]}"
echo "  Failed: $failed/${#tests[@]}"
if (( failed > 0 )); then
    for name in "${failed_names[@]}"; do
        echo "    - $name"
    done
    exit 1
else
    echo "  All tests passed."
    exit 0
fi
