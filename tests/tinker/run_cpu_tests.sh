#!/bin/bash
# Run every JAX-backend test that runs without a GPU or a live service.
set -u -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO/.venv-jaxtest"
[ -x "$VENV/bin/python" ] || { echo "no env at $VENV -- run tests/tinker/setup_cpu_jax_env.sh"; exit 1; }
cd "$REPO"
export JAX_PLATFORMS=cpu
"$VENV/bin/python" -m pytest -q --no-header \
  tests/tinker/test_loss_fns.py \
  tests/tinker/skyrl_train/test_loss_normalization.py \
  tests/tinker/test_by_codexqa_jax_contracts.py
rc=$?
echo "=== CPU SUITE EXIT $rc ==="
exit $rc
