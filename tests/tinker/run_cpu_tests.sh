#!/bin/bash
# Run the torch-backend contract tests. No GPU, no Ray, no live service: they
# execute the production function bodies lifted from their AST with small
# stubs, so they assert against the real code without the full stack.
set -u -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO/.venv-cputest"
[ -x "$VENV/bin/python" ] || { echo "no env at $VENV -- run tests/tinker/setup_cpu_env.sh"; exit 1; }
cd "$REPO"
"$VENV/bin/python" -m pytest -q --no-header \
  tests/tinker/test_by_codexqa_torch_contracts.py
rc=$?
echo "=== CPU SUITE EXIT $rc ==="
exit $rc
