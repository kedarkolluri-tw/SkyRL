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

  3. The bias-correction counter. ``test_the_first_client_update_is_adam_t1_not_t2``
     reads lora_B out of save_state checkpoints either side of one step and
     separates t=1 (~1.000 * lr) from t=2 (~0.735 * lr).

NOT covered here: the priming WEIGHT-DECAY guard. It cannot be observed
through the client API at all. ``register_pristine`` runs once and every
adapter is cloned from that one slot (adapter_store.py), so two adapters
created after priming are bit-identical whether or not the guard is present
-- an earlier version of this file compared exactly that and proved nothing.
Seeing the mutation requires snapshotting the Megatron weights either side of
``prime_optimizer_state`` itself, which is worker-internal. The guard is
covered on CPU by
``test_priming_leaves_the_adapter_bit_identical_with_nonzero_weight_decay``,
which drives the production ``prime_optimizer_state`` end to end and fails
when the guard is deleted.

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

_SKIP_REASON = "needs >= 3 CUDA GPUs (2 policy at DP=2 + 1 vLLM), matching the module these fixtures come from"

# A module-level pytest.skip() aborts COLLECTION, and pytest then exits 5
# ("no tests collected") -- which CI and a bare `pytest <thisfile>` read as a
# failure even though the terminal says "skipped". So the heavy fixtures are
# imported conditionally and the skip is a MARK, which leaves the items
# collectable and the run green.
if _cuda_ok:
    from tests.tinker.skyrl_train.test_multi_lora_megatron import (  # noqa: E402,F401
        BASE_MODEL,
        _make_datum,
        server,
        service_client,
    )

    tinker_types = pytest.importorskip("tinker.types")
else:
    BASE_MODEL = "<unavailable without CUDA>"
    tinker_types = None

    def _make_datum(*_args, **_kwargs):  # pragma: no cover - never reached
        raise RuntimeError("CUDA-only helper")

    @pytest.fixture
    def service_client():  # pragma: no cover - the mark skips first
        pytest.skip(_SKIP_REASON)

pytestmark = [
    pytest.mark.codexqa,
    pytest.mark.megatron,
    pytest.mark.gpu,
    pytest.mark.skipif(not _cuda_ok, reason=_SKIP_REASON),
]

REQUESTED = dict(learning_rate=1e-3, beta1=0.8, beta2=0.95, eps=1e-12, weight_decay=0.0)

# Every differential below compares two clients. They must start from the SAME
# lora_A draw or the difference being measured is partly three random inits.
# lora_B is zero at init so it cannot carry the confound, but lora_A can, and
# the weight-decay closed form reads lora_A directly.
LORA_SEED = 7


def _loss(result) -> float:
    return sum(sum(o["elementwise_loss"].data) for o in result.loss_fn_outputs)


def test_requested_adam_params_are_read_back_off_the_live_megatron_optimizer(service_client):
    """Not 'were echoed' -- the metrics are read off the optimizer after the write."""
    client = service_client.create_lora_training_client(
        base_model=BASE_MODEL, rank=8, seed=LORA_SEED
    )
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
    tok_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL, rank=8, seed=LORA_SEED
    )
    tok = tok_client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    def loss_drop_over(eps: float, steps: int = 3) -> float:
        client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, seed=LORA_SEED
        )
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

    Two layouts, because the backends do not agree and the first H100 run
    found out the hard way:

      FSDP      lora_adapter/adapter_model.safetensors, keys containing
                lora_A / lora_B.
      Megatron  adapter_tp0_pp0_cp0_dp{N}_ep0_etp0.pt, a torch dict under
                "model_state_dict", keys containing adapter.linear_in
                (= lora_A) and adapter.linear_out (= lora_B). Measured: the
                DP ranks hold identical tensors, so rank 0 is read and the
                rest ignored.

    engine.py writes {checkpoints_base}/{model_id}/{checkpoint_id}.tar.gz,
    uncompressed tar despite the name, so it opens with "r:*".
    """
    import glob
    import os
    import shutil
    import tarfile
    import tempfile

    assert uri.startswith("tinker://"), uri
    model_id, _, tail = uri[len("tinker://") :].partition("/")
    name = f"{tail.split('/')[-1]}.tar.gz"

    # Search rather than trust SKYRL_CHECKPOINTS_BASE. That env var moves only
    # where the TEST looks; config.py declares checkpoints_base with no env_var
    # and its env loop is opt-in per field, so the SERVER keeps writing its
    # default. Setting it broke the first H100 run.
    candidates = [
        os.environ.get("SKYRL_CHECKPOINTS_BASE"),
        "/tmp/skyrl_checkpoints",
        os.path.join(tempfile.gettempdir(), "skyrl_checkpoints"),
    ]
    path = next(
        (os.path.join(b, model_id, name) for b in candidates
         if b and os.path.exists(os.path.join(b, model_id, name))),
        None,
    )
    assert path, (
        f"no checkpoint named {name} for {model_id} under any of "
        f"{[b for b in candidates if b]}. Present: "
        f"{sorted(glob.glob('/tmp/skyrl_checkpoints/*/*'))[:10]}"
    )

    dest = os.path.join(tempfile.gettempdir(), f"codexqa_megatron_{tag}")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(path, "r:*") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # python < 3.12
            tf.extractall(dest)

    st = glob.glob(os.path.join(dest, "**", "*.safetensors"), recursive=True)
    if st:
        from safetensors.numpy import load_file

        out = {}
        for f in sorted(st):
            out.update(load_file(f))
        return out

    # Megatron: one .pt per rank, all holding the same tensors. Take rank 0.
    import torch as _torch

    pts = sorted(glob.glob(os.path.join(dest, "adapter_*_dp0_*.pt"))) or sorted(
        glob.glob(os.path.join(dest, "adapter_*.pt"))
    )
    assert pts, f"no adapter weights (.safetensors or .pt) under {dest}: {os.listdir(dest)}"
    sd = _torch.load(pts[0], map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    return {k: v.float().numpy() for k, v in sd.items() if hasattr(v, "numpy")}


def _b_keys(tensors: dict) -> list:
    """The zero-initialised LoRA factor, under either backend's naming."""
    keys = [k for k in tensors if "lora_B" in k or "linear_out" in k]
    assert keys, f"no lora_B / linear_out tensors: {sorted(tensors)[:10]}"
    return keys


def test_a_fresh_adapters_b_factor_is_zero(service_client):
    """The premise the t=1 measurement rests on, asserted rather than assumed.

    If B were not zero at init, `dB = W_after` below would be wrong and the
    ratio meaningless. Kept as its own test with its own client so it needs
    only ONE save_state -- see the note on the next test.
    """
    import numpy as np

    client = service_client.create_lora_training_client(
        base_model=BASE_MODEL, rank=8, seed=LORA_SEED
    )
    fresh = _lora_tensors(client.save_state("zero_check").result().path, "zero")
    for k in _b_keys(fresh):
        assert not np.any(fresh[k]), f"{k} is not zero at init; dB = W_after is invalid"


def test_the_first_client_update_is_adam_t1_not_t2(service_client):
    """The priming fix, measured where it shows: the weight delta.

    Megatron primes Adam state with a zero-gradient optimizer.step(), which
    leaves the counter at 1 and is then cloned into every adapter. Without the
    reset, the client's first update runs bias correction at t=2.

    B starts at zero (asserted by the test above), so dB IS W_after, and from
    zero moments

        t=1   |dB| = lr * |g| / (|g| + eps)          -> ~1.000 * lr
        t=2   |dB| = lr * (1/(1+b1)) * sqrt(1+b2)    -> ~0.735 * lr

    for eps << |g|. Those are 26.5% apart, far outside checkpoint dtype noise
    (the adapter is bf16, ~0.4%), so the median ratio separates them cleanly.

    ONE save_state, because the "before" snapshot is not needed: B starts at
    zero, so dB IS W_after. That is the whole justification.

    It is NOT a fix for the NCCL failure seen once on this path:
        DistBackendError: NCCL ... Cuda failure 999 'unknown error'
        save_state_dict_async_finalize -> torch.distributed.broadcast
    An earlier version of this comment blamed two save_state calls racing.
    That is wrong on two counts: a prior run completed both saves and wrote
    both archives, and `async_save` was false anyway -- the traceback goes
    through `execute_sync()`, so there is no background writer. The failure
    was a 2-rank NCCL broadcast on a single node with the GPUs idle and clean
    afterwards, and its cause is UNIDENTIFIED. Fewer saves means less
    exposure, not a diagnosis.
    """
    import numpy as np

    lr = 1e-3
    client = service_client.create_lora_training_client(
        base_model=BASE_MODEL, rank=8, seed=LORA_SEED
    )
    tok = client.get_tokenizer()
    data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]

    client.forward_backward(data, "cross_entropy").result()
    client.optim_step(
        tinker_types.AdamParams(
            learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0
        )
    ).result()
    after = _lora_tensors(client.save_state("t1_after").result().path, "after")

    db = np.concatenate([after[k].astype(np.float64).ravel() for k in _b_keys(after)])
    moved = np.abs(db[np.abs(db) > 0])
    assert moved.size > 0, "the B factor did not move at all"
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


# ----------------------------------------------------------------------
# Beta and weight-decay sensitivity on the real TE kernel.
#
# The eps test above proves ONE hyperparameter reaches the optimizer. It
# does not prove the others do: a backend could honour eps and quietly drop
# beta1/beta2/weight_decay, and every test so far would still pass. These
# two close that, by making each one produce a visibly different update.
# ----------------------------------------------------------------------


def _b_norm_after_steps(service_client, tag, steps=2, **adam):
    """||B|| after `steps` optim_steps, with grad_clip_norm pinned off.

    Clipping OFF is not incidental. With it on, two different gradients can
    clip to the same norm and produce identical updates -- which would make
    a real difference between settings look like no difference, the same
    trap that invalidates the high-eps probe when clipping is left at 1.0.

    The LoRA seed is pinned for the same reason the probe pins it: two
    clients created without one draw different lora_A, and dL/dB carries a
    factor of lora_A, so an unpinned differential measures the init as much
    as the setting.
    """
    import numpy as np

    params = dict(
        learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-12,
        weight_decay=0.0, grad_clip_norm=0.0,
    )
    params.update(adam)
    client = service_client.create_lora_training_client(
        base_model=BASE_MODEL, rank=8, seed=LORA_SEED
    )
    tok = client.get_tokenizer()
    # Two DIFFERENT batches: with a constant gradient the bias-corrected first
    # moment is g at every t and the betas cancel out entirely.
    batches = [
        [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")],
        [_make_datum(tok, "Question: 2+3?\nAnswer:", " 5")],
    ]
    for i in range(steps):
        client.forward_backward(batches[i % len(batches)], "cross_entropy").result()
        client.optim_step(tinker_types.AdamParams(**params)).result()
    after = _lora_tensors(client.save_state(tag).result().path, tag)
    return float(
        np.sqrt(sum((after[k].astype(np.float64) ** 2).sum() for k in _b_keys(after)))
    )


def _rel(x: float, y: float) -> float:
    return abs(x - y) / max(abs(x), abs(y))


def test_the_sensitivity_harness_is_deterministic(service_client):
    """The control for every differential below.

    If two runs at IDENTICAL settings already differ by more than the effect
    being claimed, none of those tests measure anything. This pins the noise
    floor instead of assuming it, and it is the test that fails first if the
    seed pinning above ever stops working.
    """
    a = _b_norm_after_steps(service_client, "determinism_a")
    b = _b_norm_after_steps(service_client, "determinism_b")
    print(f"[control] ||B|| run1={a:.9e}  run2={b:.9e}  rel={_rel(a, b):.3e}")
    assert a > 0, "no movement at all; the harness cannot discriminate anything"
    assert _rel(a, b) < 1e-6, (
        f"two IDENTICAL runs differ by {_rel(a, b):.3e}. The differentials below "
        "cannot resolve a real effect smaller than that, and the beta2 test "
        "(~2e-3) is smaller than most plausible noise floors."
    )


def test_beta1_alone_changes_the_update_on_the_real_te_kernel(service_client):
    """beta1 varied with beta2 HELD FIXED.

    An earlier version moved beta1 and beta2 together, so it would have
    passed with beta2 entirely ignored -- beta1 alone is worth ~6.5% and
    would have carried the assertion by itself.
    """
    slow = _b_norm_after_steps(service_client, "beta1_slow", beta1=0.5, beta2=0.95)
    fast = _b_norm_after_steps(service_client, "beta1_fast", beta1=0.99, beta2=0.95)
    print(f"[beta1] ||B|| 0.5={slow:.6e}  0.99={fast:.6e}  rel={_rel(slow, fast):.4%}")
    assert slow > 0 and fast > 0, "no movement at all; the test cannot discriminate"
    assert _rel(slow, fast) > 1e-2, (
        f"beta1 made no difference ({slow:.6e} vs {fast:.6e}); it is being "
        "accepted and dropped somewhere below the param_group write"
    )


def test_beta2_alone_changes_the_update_on_the_real_te_kernel(service_client):
    """beta2 varied with beta1 HELD FIXED.

    beta2 is the weak one: it moves ||B|| by only ~0.2%, three decades above
    the determinism control's floor but far below beta1's 6.5%. That gap is
    exactly why the two had to be separated -- a combined test cannot fail
    on beta2.
    """
    slow = _b_norm_after_steps(service_client, "beta2_slow", beta1=0.9, beta2=0.9)
    fast = _b_norm_after_steps(service_client, "beta2_fast", beta1=0.9, beta2=0.999)
    print(f"[beta2] ||B|| 0.9={slow:.6e}  0.999={fast:.6e}  rel={_rel(slow, fast):.4%}")
    assert slow > 0 and fast > 0, "no movement at all; the test cannot discriminate"
    assert _rel(slow, fast) > 1e-4, (
        f"beta2 made no difference ({slow:.6e} vs {fast:.6e}); it is being "
        "accepted and dropped somewhere below the param_group write"
    )


def test_weight_decay_is_applied_with_the_exact_closed_form(service_client):
    """Measured on lora_A, and against a closed form, not a direction.

    A first attempt compared ||B|| at wd=0 vs wd=0.5 and "passed" with the
    two differing by 0.027% -- inside noise, and proving nothing. B starts at
    ZERO, so decoupled decay has almost nothing to act on.

    lora_A is the right probe: it starts nonzero, and on the FIRST step
    dL/dA is proportional to B, which is still zero, so A receives no
    gradient at all. The only term touching it is decay:

        A_after = A_init * (1 - lr * wd)

    Two clients with the same PINNED seed start from the same A_init, so the
    ratio of their norms is exactly (1 - lr*wd) with no before-snapshot
    needed -- which also avoids the two-saves-in-one-client path that once
    hit a NCCL CUDA 999.

    lr is deliberately large here (0.1, not 1e-3) so the predicted 5% shift
    sits far outside bf16 checkpoint noise.
    """
    import numpy as np

    lr, wd = 0.1, 0.5

    def a_norm(tag, weight_decay):
        client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, seed=LORA_SEED
        )
        tok = client.get_tokenizer()
        data = [_make_datum(tok, "Question: 1+1?\nAnswer:", " 2")]
        client.forward_backward(data, "cross_entropy").result()
        client.optim_step(
            tinker_types.AdamParams(
                learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-12,
                weight_decay=weight_decay, grad_clip_norm=0.0,
            )
        ).result()
        t = _lora_tensors(client.save_state(tag).result().path, tag)
        keys = [k for k in t if any(n in k for n in ("lora_A", "linear_in"))]
        assert keys, f"no A-factor tensors: {sorted(t)[:10]}"
        return float(np.sqrt(sum((t[k].astype(np.float64) ** 2).sum() for k in keys)))

    undecayed = a_norm("wd_zero", 0.0)
    decayed = a_norm("wd_half", wd)
    ratio = decayed / undecayed
    predicted = 1.0 - lr * wd

    print(f"[wd] ||A|| wd=0={undecayed:.6e}  wd={wd}={decayed:.6e}  "
          f"ratio={ratio:.6f}  predicted={predicted:.6f}")
    assert abs(ratio - predicted) < 0.005, (
        f"||A|| ratio {ratio:.6f} does not match the closed form "
        f"(1 - lr*wd) = {predicted:.6f}. Either decay was not applied, or A "
        "received a gradient it should not have on the first step."
    )


def test_grad_clip_norm_changes_the_update_on_the_real_te_kernel(service_client):
    """The parity bug itself, measured: 0.0 must mean NO clipping.

    Tinker defaults grad_clip_norm to 0.0; SkyRL used to drop the field and
    clip at OptimizerConfig.max_grad_norm = 1.0. If 0.0 and a tight
    threshold produce the same weights, the request is still being ignored.
    """
    unclipped = _b_norm_after_steps(service_client, "clip_off", grad_clip_norm=0.0)
    clipped = _b_norm_after_steps(service_client, "clip_tight", grad_clip_norm=1e-4)
    print(f"[clip] ||B|| off={unclipped:.6e}  tight={clipped:.6e}  "
          f"rel={_rel(unclipped, clipped):.4%}")
    assert unclipped > 0, "no movement at all; the test cannot discriminate"
    assert _rel(unclipped, clipped) > 1e-2, (
        f"grad_clip_norm made no difference ({unclipped:.6e} vs {clipped:.6e}); "
        "0.0 is supposed to mean no clipping and 1e-4 to clip hard"
    )
