#!/bin/bash
# Run every fork test that runs without a GPU or a live service.
#
# The JAX backend and the torch backends cannot share an interpreter in a
# normal skyrl install (mutually conflicting extras), but the torch-side
# contract tests only need torch itself, so this one env covers both.
set -u -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO/.venv-jaxtest"
[ -x "$VENV/bin/python" ] || { echo "no env at $VENV -- run tests/tinker/setup_cpu_jax_env.sh"; exit 1; }
cd "$REPO"
export JAX_PLATFORMS=cpu
"$VENV/bin/python" -m pytest -q --no-header \
  tests/tinker/test_loss_fns.py \
  tests/tinker/skyrl_train/test_loss_normalization.py \
  tests/tinker/test_by_codexqa_jax_contracts.py \
  tests/tinker/test_by_codexqa_torch_contracts.py
rc=$?
echo "=== CPU SUITE EXIT $rc ==="
exit $rc
