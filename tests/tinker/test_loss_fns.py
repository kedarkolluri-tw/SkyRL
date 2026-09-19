import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tinker.loss_fns import LossFnConfig, cispo_loss


def test_cispo_loss_clipping_and_masking():
    target_logprobs = jnp.array([0.0, 5.0, -5.0], dtype=jnp.float32)
    sampling_logprobs = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    advantages = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    loss_mask = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
    loss_fn_config = LossFnConfig(
        clip_low_threshold=jnp.array(0.8, dtype=jnp.float32),
        clip_high_threshold=jnp.array(1.2, dtype=jnp.float32),
    )

    actual = np.asarray(cispo_loss(target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config))

    ratios = np.exp(np.asarray(target_logprobs - sampling_logprobs))
    clipped_ratios = np.clip(ratios, 0.8, 1.2)
    expected = -(clipped_ratios * np.asarray(target_logprobs) * np.asarray(advantages))
    expected[2] = 0.0

    assert np.allclose(actual, expected)


def test_cispo_stops_gradient_through_clipped_ratio():
    sampling_logprobs = jnp.array([0.1, -0.2], dtype=jnp.float32)
    advantages = jnp.array([1.3, -0.7], dtype=jnp.float32)
    loss_mask = jnp.ones((2,), dtype=jnp.float32)
    loss_fn_config = LossFnConfig(
        clip_low_threshold=jnp.array(0.8, dtype=jnp.float32),
        clip_high_threshold=jnp.array(1.2, dtype=jnp.float32),
    )

    def total_loss(target_logprobs):
        return cispo_loss(target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config).sum()

    target_logprobs = jnp.array([0.0, 0.2], dtype=jnp.float32)
    grad = np.asarray(jax.grad(total_loss)(target_logprobs))

    clipped_ratio = np.clip(
        np.exp(np.asarray(target_logprobs - sampling_logprobs)),
        0.8,
        1.2,
    )
    expected_grad = -(clipped_ratio * np.asarray(advantages))

    assert np.allclose(grad, expected_grad, rtol=1e-5, atol=1e-6)


# --- per-loss default clip bounds ------------------------------------------
# A client sending loss_fn="cispo" with no loss_fn_config used to get the
# PPO-family bounds (0.8, 1.2) here while the torch backends fell back to
# CISPOConfig's (0.0, 5.0). Same request, a different objective per backend.


def test_cispo_default_bounds_are_not_ppos():
    from skyrl.backends.jax import (
        _DEFAULT_CISPO_CLIP_HIGH_THRESHOLD,
        _DEFAULT_CISPO_CLIP_LOW_THRESHOLD,
        _DEFAULT_PPO_CLIP_HIGH_THRESHOLD,
        _DEFAULT_PPO_CLIP_LOW_THRESHOLD,
        JaxBackendImpl,
    )
    from skyrl.tinker.types import LOSS_TYPES

    cfg = JaxBackendImpl._build_loss_fn_config([None], [LOSS_TYPES["cispo"]])
    assert float(cfg.clip_low_threshold[0]) == _DEFAULT_CISPO_CLIP_LOW_THRESHOLD == 0.0
    assert float(cfg.clip_high_threshold[0]) == _DEFAULT_CISPO_CLIP_HIGH_THRESHOLD == 5.0

    # ...and these match what the torch backends use, which is the point.
    # skyrl.train.config pulls in omegaconf, which the `jax` extra does not
    # install, so this half of the assertion is skipped rather than silently
    # dropped when only the jax deps are present.
    CISPOConfig = pytest.importorskip(
        "skyrl.train.config.config", reason="needs omegaconf (torch extras)"
    ).CISPOConfig

    c = CISPOConfig()
    assert (1 - c.cispo_eps_clip_low, 1 + c.cispo_eps_clip_high) == (
        _DEFAULT_CISPO_CLIP_LOW_THRESHOLD,
        _DEFAULT_CISPO_CLIP_HIGH_THRESHOLD,
    ), "JAX cispo defaults drifted from CISPOConfig -- the backends disagree again"

    ppo = JaxBackendImpl._build_loss_fn_config([None], [LOSS_TYPES["ppo"]])
    # 0.8 and 1.2 are not exactly representable in float32, which is the array
    # dtype here; 0.0 and 5.0 above are, hence the exact comparisons there.
    assert float(ppo.clip_low_threshold[0]) == pytest.approx(_DEFAULT_PPO_CLIP_LOW_THRESHOLD)
    assert float(ppo.clip_high_threshold[0]) == pytest.approx(_DEFAULT_PPO_CLIP_HIGH_THRESHOLD)


def test_mixed_batch_gets_per_example_defaults():
    """One batch, two loss fns: each example must get its own defaults.

    The JAX path vmaps a per-example lax.switch over loss types, so a single
    forward_backward can legitimately mix cispo and ppo examples.
    """
    from skyrl.backends.jax import JaxBackendImpl
    from skyrl.tinker.types import LOSS_TYPES

    cfg = JaxBackendImpl._build_loss_fn_config(
        [None, None, None],
        [LOSS_TYPES["ppo"], LOSS_TYPES["cispo"], LOSS_TYPES["ppo"]],
    )
    assert [float(x) for x in cfg.clip_low_threshold] == pytest.approx([0.8, 0.0, 0.8])
    assert [float(x) for x in cfg.clip_high_threshold] == pytest.approx([1.2, 5.0, 1.2])


def test_explicit_config_still_wins_over_defaults():
    from skyrl.backends.jax import JaxBackendImpl
    from skyrl.tinker.types import LOSS_TYPES

    cfg = JaxBackendImpl._build_loss_fn_config(
        [{"clip_low_threshold": 0.2, "clip_high_threshold": 2.0}], [LOSS_TYPES["cispo"]]
    )
    assert float(cfg.clip_low_threshold[0]) == pytest.approx(0.2)
    assert float(cfg.clip_high_threshold[0]) == pytest.approx(2.0)


def test_omitting_loss_types_preserves_old_behaviour():
    """Back-compat: the types argument is optional, and omitting it keeps the
    previous PPO-default-for-everything semantics rather than silently changing
    any other caller."""
    from skyrl.backends.jax import JaxBackendImpl

    cfg = JaxBackendImpl._build_loss_fn_config([None])
    assert float(cfg.clip_low_threshold[0]) == pytest.approx(0.8)
    assert float(cfg.clip_high_threshold[0]) == pytest.approx(1.2)


# --- loss reduction (task 1.5) ----------------------------------------------
# The torch backends plain-sum (ppo_utils.reduce_loss = (loss*mask).sum()) and
# leave the reduction to the client, which pre-scales advantages -- see
# examples/tinker/ppo/ppo_client.py calling
# apply_loss_reduction_to_advantages_minibatch before sending.
#
# The divisor lives in AccumulatedGradients.get_mean, NOT in loss_for_lora,
# because gradients are summed across micro-batches before being normalised.
# An earlier version of this patch chose the reduction in the loss function
# only, so get_mean still divided by the sequence count and "sum" was really
# sum/N_sequences. These tests therefore drive the REAL AccumulatedGradients
# end to end rather than a local mirror of the arithmetic -- the mirror is what
# let that bug look correct.


def _accumulate_and_normalise(per_seq_sums, loss_mask, reduction, max_adapters=2):
    """Push a known gradient through the real accumulate -> get_mean path."""
    from skyrl.backends.jax import AccumulatedGradients

    if reduction == "sequence_mean":
        total = (per_seq_sums / jnp.maximum(loss_mask.sum(axis=-1), 1e-9)).sum()
    else:
        total = per_seq_sums.sum()

    adapter_indices = jnp.zeros((per_seq_sums.shape[0],), dtype=jnp.int32)
    acc = AccumulatedGradients.create({"w": jnp.zeros((max_adapters,))}, max_adapters)
    acc = acc.add({"w": jnp.zeros((max_adapters,)).at[0].set(total)}, adapter_indices, loss_mask)
    return float(acc.get_mean(jnp.int32(0), reduction)["w"][0])


def test_loss_reduction_default_is_unchanged():
    from skyrl.backends.jax import JaxBackendConfig

    assert JaxBackendConfig().loss_reduction == "sequence_mean"


def test_loss_reduction_rejects_unknown_values():
    from skyrl.backends.jax import JaxBackendConfig

    with pytest.raises(ValueError, match="loss_reduction must be one of"):
        JaxBackendConfig(loss_reduction="mean")


def test_sum_reduction_matches_torch_reduce_loss_end_to_end():
    """'sum' must equal torch's (loss * mask).sum() AFTER normalisation.

    Two sequences with loss sums 2 and 3: torch gives 5. Before the get_mean
    fix this path gave 2.5, because the accumulated gradient was divided by the
    sequence count.
    """
    per_seq = jnp.array([2.0, 3.0])
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    assert _accumulate_and_normalise(per_seq, mask, "sum") == pytest.approx(5.0)


def test_token_mean_divides_by_total_tokens_across_the_batch():
    per_seq = jnp.array([2.0, 3.0])
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])   # 5 unmasked tokens
    assert _accumulate_and_normalise(per_seq, mask, "token_mean") == pytest.approx(1.0)


def test_sequence_mean_is_bitwise_unchanged():
    """The default must not move: per-sequence mean, then mean over sequences."""
    per_seq = jnp.array([2.0, 3.0])
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    expected = float((per_seq / mask.sum(axis=-1)).sum() / 2)
    assert _accumulate_and_normalise(per_seq, mask, "sequence_mean") == pytest.approx(expected)


def test_token_mean_normalises_over_accumulated_micro_batches():
    """The divisor must span gradient accumulation, not one micro-batch.

    Two micro-batches of one sequence each: token_mean must divide by the total
    token count (2 + 3 = 5), which a per-micro-batch divide cannot produce.
    """
    from skyrl.backends.jax import AccumulatedGradients

    acc = AccumulatedGradients.create({"w": jnp.zeros((2,))}, 2)
    for loss, mask in ((2.0, jnp.array([[1.0, 1.0, 0.0]])), (3.0, jnp.array([[1.0, 1.0, 1.0]]))):
        acc = acc.add({"w": jnp.zeros((2,)).at[0].set(loss)}, jnp.zeros((1,), jnp.int32), mask)
    assert float(acc.token_counts[0]) == pytest.approx(5.0)
    assert float(acc.get_mean(jnp.int32(0), "token_mean")["w"][0]) == pytest.approx(5.0 / 5.0)
    assert float(acc.get_mean(jnp.int32(0), "sum")["w"][0]) == pytest.approx(5.0)


def test_reset_adapter_clears_token_counts():
    from skyrl.backends.jax import AccumulatedGradients

    acc = AccumulatedGradients.create({"w": jnp.zeros((2,))}, 2)
    acc = acc.add({"w": jnp.ones((2,))}, jnp.zeros((1,), jnp.int32), jnp.array([[1.0, 1.0]]))
    assert float(acc.token_counts[0]) == pytest.approx(2.0)
    acc = acc.reset_adapter(jnp.int32(0))
    assert float(acc.token_counts[0]) == 0.0
    assert int(acc.counts[0]) == 0


def test_all_ones_mask_makes_the_denominator_sequence_length():
    """Why this matters for harness-1.

    tinker_cookbook's RL path strips its "mask" key before sending
    (rl/train.py _remove_mask), and the server defaults absent weights to
    all-ones (api.py Datum.to_types). So loss_mask.sum() becomes the FULL
    sequence length, not the count of tokens carrying gradient -- ~11,145
    against ~217 on harness-1's datums. 'sum' is invariant to that; the
    mean-style reductions are not.
    """
    n_total, n_scoring = 100, 4
    per_seq = jnp.array([float(-n_scoring)])
    all_ones = jnp.ones((1, n_total))
    true_mask = jnp.array([[1.0] * n_scoring + [0.0] * (n_total - n_scoring)])

    assert _accumulate_and_normalise(per_seq, all_ones, "sum") == pytest.approx(-4.0)
    assert _accumulate_and_normalise(per_seq, true_mask, "sum") == pytest.approx(-4.0)
    # ...while token_mean swings by the ratio of the two denominators.
    assert _accumulate_and_normalise(per_seq, all_ones, "token_mean") == pytest.approx(-0.04)
    assert _accumulate_and_normalise(per_seq, true_mask, "token_mean") == pytest.approx(-1.0)
