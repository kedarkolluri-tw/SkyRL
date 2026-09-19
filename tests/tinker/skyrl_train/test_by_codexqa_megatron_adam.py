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

from tests.tinker.skyrl_train.test_multi_lora_megatron import (  # noqa: F401
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
