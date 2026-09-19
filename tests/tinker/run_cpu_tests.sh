#!/bin/bash
# Every CPU test on this branch, across BOTH backends.
#
# The backend extras (jax, fsdp, megatron) are mutually exclusive -- uv
# refuses to resolve them together -- so "the whole suite" is necessarily two
# interpreters, not one. Everything else (ray, omegaconf, ...) is orthogonal
# and available to either.
#
#   .venv-cputest  torch contracts      tests/tinker/setup_cpu_env.sh
#   .venv-jaxtest  JAX backend          tests/tinker/setup_cpu_jax_env.sh
set -u -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

rc=0
run() {  # $1 venv, $2 setup script, $3.. test paths
  local venv="$REPO/$1" setup="$2"; shift 2
  if [ ! -x "$venv/bin/python" ]; then
    echo "!! no env at $venv -- run $setup"
    rc=1
    return
  fi
  echo "########## $(basename "$venv") ##########"
  JAX_PLATFORMS=cpu "$venv/bin/python" -m pytest -q --no-header -rs -p no:cacheprovider "$@"
  local this=$?
  [ $this -eq 0 ] || rc=1
}

run .venv-cputest tests/tinker/setup_cpu_env.sh \
  tests/tinker/test_by_codexqa_torch_contracts.py

run .venv-jaxtest tests/tinker/setup_cpu_jax_env.sh \
  tests/tinker/test_loss_fns.py \
  tests/tinker/test_by_codexqa_jax_contracts.py

echo "=== CPU SUITE (all backends) EXIT $rc ==="
exit $rc
