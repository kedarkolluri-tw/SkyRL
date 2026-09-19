"""Do the client's AdamParams actually reach Megatron's optimizer?

GPU-gated; skipped without CUDA. This is the gap the CPU contract tests in
tests/tinker/test_by_codexqa_torch_contracts.py cannot close: those lift the
production functions out by AST and run them against torch's own AdamW, which
proves the LOGIC but says nothing about Transformer Engine's FusedAdam,
Megatron's DistributedOptimizer, the ChainedOptimizer wrapper, AdapterStore
priming, or Ray dispatch. Every one of those sits between
``set_adam_hyperparams`` and the arithmetic.

Two things are checked, chosen because they fail for different reasons:

  1. The values come back from the LIVE optimizer. If TE FusedAdam's
     param_groups do not carry an ``eps`` key, or Megatron's several groups
     disagree, optim_step raises instead of reporting success.

  2. eps changes the update. A param_group write that lands in the dict but
     never reaches the kernel would pass (1) and fail (2). With an absurd eps
     the Adam denominator is dominated by eps, the update collapses toward
     ``lr * g / eps``, and training visibly stops moving.

NOT covered here: the bias-correction step counter. Proving the first update
is t=1 rather than t=2 needs the weight delta, which means downloading a
checkpoint; the CPU test
``test_priming_step_counter_is_reset_so_the_first_client_step_is_t1`` pins the
reset itself and ``test_the_priming_step_is_worth_resetting`` pins the 0.735x
consequence numerically.

THIS FILE HAS NEVER BEEN EXECUTED -- no Megatron hardware was available when
it was written. Treat a first run as debugging, not as a regression.

Run with:
  uv run --extra tinker --extra megatron --with pytest --with pytest-timeout \
    pytest -s tests/tinker/skyrl_train/test_by_codexqa_megatron_adam.py
"""

from __future__ import annotations

import pytest

# The CUDA gate must be OURS. Importing fixtures from the multi-LoRA module
# does not inherit that module's skipif -- module-level marks are not carried
# through fixture imports -- and `pytest.mark.gpu` only labels, it does not
# skip. Without this, the CPU workflow that runs all of tests/tinker/
# skyrl_train/ with tinker installed would try to boot a Megatron server.
_cuda_ok = False
try:  # pragma: no cover - import guard
    import torch

    _cuda_ok = bool(torch.cuda.is_available() and torch.cuda.device_count() >= 3)
except Exception:
    _cuda_ok = False

if not _cuda_ok:
    pytest.skip(
        "needs >= 3 CUDA GPUs (2 policy at DP=2 + 1 vLLM), matching the module "
        "these fixtures come from",
        allow_module_level=True,
    )

from tests.tinker.skyrl_train.test_multi_lora_megatron import (  # noqa: E402,F401
    BASE_MODEL,
    _make_datum,
    server,
    service_client,
)

tinker_types = pytest.importorskip("tinker.types")

pytestmark = [pytest.mark.codexqa, pytest.mark.megatron, pytest.mark.gpu]

REQUESTED = dict(learning_rate=1e-3, beta1=0.8, beta2=0.95, eps=1e-12, weight_decay=0.0)


def _loss(result) -> float:
    return sum(sum(o["elementwise_loss"].data) for o in result.loss_fn_outputs)


def test_requested_adam_params_are_read_back_off_the_live_megatron_optimizer(service_client):
    """Not 'were echoed' -- the metrics are read off the optimizer after the write."""
    client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    client.forward_backward(data, "cross_entropy").result()
    out = client.optim_step(tinker_types.AdamParams(**REQUESTED)).result()
    metrics = dict(out.metrics or {})

    for name, key in (
        ("beta1", "skyrl.ai/effective_beta1"),
        ("beta2", "skyrl.ai/effective_beta2"),
        ("eps", "skyrl.ai/effective_eps"),
        ("weight_decay", "skyrl.ai/effective_weight_decay"),
    ):
        assert key in metrics, (
            f"{key} missing. Either the optimizer's param_groups do not expose {name} "
            f"(TE FusedAdam may not), or the groups disagreed. Metrics: {sorted(metrics)}"
        )
        assert metrics[key] == pytest.approx(REQUESTED[name]), (
            f"{name}: asked {REQUESTED[name]!r}, optimizer holds {metrics[key]!r}"
        )


def test_eps_reaches_the_kernel_not_just_the_param_group_dict(service_client):
    """A write that lands in the dict but is ignored by the step would pass the
    read-back test and fail this one."""
    tok_client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = tok_client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    def loss_drop_over(eps: float, steps: int = 3) -> float:
        client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
        first = _loss(client.forward_backward(data, "cross_entropy").result())
        client.optim_step(tinker_types.AdamParams(**{**REQUESTED, "eps": eps})).result()
        for _ in range(steps - 1):
            client.forward_backward(data, "cross_entropy").result()
            client.optim_step(tinker_types.AdamParams(**{**REQUESTED, "eps": eps})).result()
        last = _loss(client.forward_backward(data, "cross_entropy").result())
        return first - last

    normal = loss_drop_over(1e-12)
    crippled = loss_drop_over(1e9)

    assert normal > 0, f"training did not reduce the loss at all (drop={normal})"
    assert crippled < normal * 0.1, (
        f"eps=1e9 should nearly stop the update (lr*g/eps -> 0) but the loss still "
        f"fell {crippled} against {normal} at eps=1e-12. eps is being written to the "
        "param_group and ignored by the optimizer step."
    )


# ----------------------------------------------------------------------
# The priming fix itself: is the client's FIRST update t=1 or t=2?
# ----------------------------------------------------------------------


def _lora_tensors(uri: str, tag: str) -> dict:
    """Every LoRA tensor from a save_state checkpoint, keyed by name.

    engine.py writes {checkpoints_base}/{model_id}/{checkpoint_id}.tar.gz and
    checkpoints_base defaults to /tmp/skyrl_checkpoints. Despite the name the
    archive is UNCOMPRESSED tar, so it opens with "r:*".
    """
    import glob
    import os
    import shutil
    import tarfile
    import tempfile

    from safetensors.numpy import load_file

    assert uri.startswith("tinker://"), uri
    model_id, _, tail = uri[len("tinker://") :].partition("/")
    base = os.environ.get("SKYRL_CHECKPOINTS_BASE", "/tmp/skyrl_checkpoints")
    path = os.path.join(base, model_id, f"{tail.split('/')[-1]}.tar.gz")
    assert os.path.exists(path), (
        f"no checkpoint at {path}. Present: {sorted(glob.glob(os.path.join(base, '*', '*')))[:10]}"
    )
    dest = os.path.join(tempfile.gettempdir(), f"codexqa_megatron_{tag}")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(path, "r:*") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # python < 3.12
            tf.extractall(dest)
    st = os.path.join(dest, "lora_adapter", "adapter_model.safetensors")
    assert os.path.exists(st), f"no adapter_model.safetensors under {dest}"
    return load_file(st)


def test_the_first_client_update_is_adam_t1_not_t2(service_client):
    """The priming fix, measured where it shows: the weight delta.

    Megatron primes Adam state with a zero-gradient optimizer.step(), which
    leaves the counter at 1 and is then cloned into every adapter. Without the
    reset, the client's first update runs bias correction at t=2.

    lora_B starts at zero, so dB IS B_after, and from zero moments

        t=1   |dB| = lr * |g| / (|g| + eps)          -> ~1.000 * lr
        t=2   |dB| = lr * (1/(1+b1)) * sqrt(1+b2)    -> ~0.735 * lr

    for eps << |g|. Those are 26.5% apart, far outside checkpoint dtype noise,
    so the median ratio separates them cleanly.
    """
    import numpy as np

    lr = 1e-3
    client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    tok = client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    before_uri = client.save_state("t1_before").result().path
    client.forward_backward(data, "cross_entropy").result()
    client.optim_step(
        tinker_types.AdamParams(
            learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0
        )
    ).result()
    after_uri = client.save_state("t1_after").result().path

    before, after = _lora_tensors(before_uri, "before"), _lora_tensors(after_uri, "after")
    b_keys = [k for k in before if "lora_B" in k]
    assert b_keys, f"no lora_B tensors in the checkpoint: {sorted(before)[:10]}"

    db = np.concatenate(
        [(after[k].astype(np.float64) - before[k].astype(np.float64)).ravel() for k in b_keys]
    )
    moved = np.abs(db[np.abs(db) > 0])
    assert moved.size > 0, "lora_B did not move at all"
    ratio = float(np.median(moved) / lr)

    print(f"[t1] median |dB|/lr = {ratio:.4f}   (t=1 -> ~1.000, t=2 -> ~0.735)")
    assert abs(ratio - 0.7350) > 0.05, (
        f"median |dB|/lr = {ratio:.4f}, which is Adam's t=2 magnitude. The priming "
        "dummy step's counter was not reset, so every adapter's first update is "
        "26.5% short."
    )
    assert abs(ratio - 1.0) < 0.05, (
        f"median |dB|/lr = {ratio:.4f}, expected ~1.0 for a t=1 Adam step. Either lr "
        "was not applied, eps is comparable to |g|, or more than one step was taken."
    )


def test_priming_leaves_the_adapter_undecayed(service_client):
    """Two adapters created back to back must be bit-identical.

    The priming dummy step is a real AdamW step; decoupled weight decay does
    not need a gradient, so without suppression each pristine snapshot is
    taken from already-decayed weights.
    """
    import numpy as np

    a = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    b = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    ta = _lora_tensors(a.save_state("undecayed_a").result().path, "ua")
    tb = _lora_tensors(b.save_state("undecayed_b").result().path, "ub")

    assert set(ta) == set(tb)
    for k in ta:
        assert np.array_equal(ta[k], tb[k]), (
            f"{k} differs between two freshly created adapters; priming is mutating "
            "the pristine template (decoupled weight decay on the dummy step)."
        )
