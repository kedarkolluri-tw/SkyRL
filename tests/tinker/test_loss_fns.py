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
# apply_loss_reduction_to_advantages_minibatch before sending. The JAX backend
# additionally divided each sequence by its own loss_mask sum, so the same
# client request was reduced twice on jax and once on torch.
#
# These assert the reduction arithmetic directly, so they need no GPU and no
# model. The default is unchanged, so no existing run moves.


def _reduce(per_token, loss_mask, reduction):
    """Mirror of the reduction block in JaxBackendImpl (jax.py)."""
    if reduction == "sum":
        return per_token.sum()
    if reduction == "token_mean":
        return per_token.sum() / jnp.maximum(loss_mask.sum(), 1e-9)
    return (per_token.sum(axis=-1) / jnp.maximum(loss_mask.sum(axis=-1), 1e-9)).sum()


def test_loss_reduction_default_is_unchanged():
    from skyrl.backends.jax import JaxBackendConfig

    assert JaxBackendConfig().loss_reduction == "sequence_mean"


def test_loss_reduction_rejects_unknown_values():
    from skyrl.backends.jax import JaxBackendConfig

    with pytest.raises(ValueError, match="loss_reduction must be one of"):
        JaxBackendConfig(loss_reduction="mean")


def test_sum_reduction_matches_torch_reduce_loss():
    """'sum' reproduces torch's reduce_loss exactly.

    torch: (loss * mask).sum(). JAX's cispo_loss already folds the mask into
    its per-token output via safe_loss_mask, so the plain sum of the JAX
    per-token array is the same quantity.
    """
    per_token = jnp.array([[-1.0, -2.0, 0.0], [-0.5, -0.25, -0.25]])
    loss_mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    torch_equivalent = float((per_token * loss_mask).sum())
    assert float(_reduce(per_token, loss_mask, "sum")) == pytest.approx(torch_equivalent)


def test_sequence_mean_and_sum_differ_on_unequal_lengths():
    """The divergence is a per-sequence reweighting, not a scalar factor.

    Two sequences with 2 and 3 unmasked tokens: sequence_mean weights them
    1/2 and 1/3, so no single constant relates it to the sum. This is why the
    two backends could not be reconciled by rescaling.
    """
    per_token = jnp.array([[-1.0, -1.0, 0.0], [-1.0, -1.0, -1.0]])
    loss_mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    s = float(_reduce(per_token, loss_mask, "sum"))
    sm = float(_reduce(per_token, loss_mask, "sequence_mean"))
    tm = float(_reduce(per_token, loss_mask, "token_mean"))

    assert s == pytest.approx(-5.0)
    assert sm == pytest.approx(-2.0)          # -1.0 + -1.0
    assert tm == pytest.approx(-1.0)          # -5/5
    # no scalar c makes sequence_mean == c * sum for both of these sequences
    assert sm / s != pytest.approx(1 / 2)
    assert sm / s != pytest.approx(1 / 3)


def test_token_mean_is_a_single_denominator():
    per_token = jnp.array([[-2.0, -2.0, 0.0], [-1.0, -1.0, -1.0]])
    loss_mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    assert float(_reduce(per_token, loss_mask, "token_mean")) == pytest.approx(-7.0 / 5.0)


def test_all_ones_mask_makes_the_denominator_sequence_length():
    """The case that matters for harness-1.

    tinker_cookbook's RL path strips its "mask" key before sending
    (rl/train.py _remove_mask), and the server defaults absent weights to
    all-ones (api.py Datum.to_types). So loss_mask.sum() becomes the FULL
    sequence length, not the number of tokens carrying gradient. On harness-1's
    datums that is ~11,145 against ~217 -- a ~51x denominator set by how long
    the environment's observations happen to be.
    """
    n_total, n_scoring = 100, 4
    per_token = jnp.array([[-1.0] * n_scoring + [0.0] * (n_total - n_scoring)])
    all_ones = jnp.ones((1, n_total))
    true_mask = jnp.array([[1.0] * n_scoring + [0.0] * (n_total - n_scoring)])

    assert float(_reduce(per_token, all_ones, "sequence_mean")) == pytest.approx(-n_scoring / n_total)
    assert float(_reduce(per_token, true_mask, "sequence_mean")) == pytest.approx(-1.0)
    assert float(_reduce(per_token, all_ones, "sum")) == pytest.approx(-float(n_scoring))
    # 'sum' is invariant to the bogus denominator; 'sequence_mean' is not.
    assert float(_reduce(per_token, all_ones, "sum")) == float(_reduce(per_token, true_mask, "sum"))
