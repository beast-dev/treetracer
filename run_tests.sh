#!/usr/bin/env bash
# Entrypoint for the TreeTracer CI test suite.
#
# Full run:       bash run_tests.sh
# Pick a subset:  bash run_tests.sh -k mcc    # any pytest flags pass through
set -euo pipefail
cd "$(dirname "$0")"

# Install the test dependency group only if it isn't already present.
if ! uv run --no-sync python -c "import pytest, arviz, dendropy" 2>/dev/null; then
    uv sync --group test
fi

uv run --no-sync pytest src/test/ "$@"
