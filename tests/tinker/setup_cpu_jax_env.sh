#!/bin/bash
# Build the CPU-only JAX environment these tests need, INSIDE the repo.
#
# Why a separate venv at all: skyrl declares `jax`, `fsdp` and `megatron` as
# mutually conflicting extras, so the JAX backend cannot share an interpreter
# with the torch backends. This one holds only the JAX side.
#
# Why --no-config: run from inside the repo, uv picks up skyrl's own
# [tool.uv] block, whose extra-build-dependencies expect the full torch/vllm
# resolution -- it aborts with "torch ... not found in the resolution". It also
# picks up a global ~/.config/uv/uv.toml exclude-newer cutoff that hides every
# release these pins need. --no-config drops both; the cutoff is then restated
# explicitly below so the resolution stays reproducible.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO/.venv-jaxtest"
REQ="$REPO/tests/tinker/test_by_codexqa_jax_requirements.txt"

command -v uv >/dev/null || { echo "uv not on PATH"; exit 1; }

uv venv --no-config --python 3.12 "$VENV"
uv pip install --no-config --python "$VENV/bin/python" \
  --exclude-newer 2026-09-18T00:00:00Z \
  -r "$REQ"

echo
echo "--- verifying the versions that matter ---"
"$VENV/bin/python" - <<'PY'
import jax, flax, optax
print("jax  ", jax.__version__)
print("flax ", flax.__version__)
print("optax", optax.__version__)
assert jax.__version__ == "0.11.1", f"jax {jax.__version__}: flax breaks on 0.11.2"
PY

echo
echo "env ready: $VENV"
echo "run the tests with:"
echo "  cd $REPO && ./tests/tinker/run_cpu_tests.sh"
