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
    from skyrl.train.config.config import CISPOConfig

    c = CISPOConfig()
    assert (1 - c.cispo_eps_clip_low, 1 + c.cispo_eps_clip_high) == (
        _DEFAULT_CISPO_CLIP_LOW_THRESHOLD,
        _DEFAULT_CISPO_CLIP_HIGH_THRESHOLD,
    ), "JAX cispo defaults drifted from CISPOConfig -- the backends disagree again"

    ppo = JaxBackendImpl._build_loss_fn_config([None], [LOSS_TYPES["ppo"]])
    assert float(ppo.clip_low_threshold[0]) == _DEFAULT_PPO_CLIP_LOW_THRESHOLD
    assert float(ppo.clip_high_threshold[0]) == _DEFAULT_PPO_CLIP_HIGH_THRESHOLD


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
    assert [float(x) for x in cfg.clip_low_threshold] == [0.8, 0.0, 0.8]
    assert [float(x) for x in cfg.clip_high_threshold] == [1.2, 5.0, 1.2]


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
    assert float(cfg.clip_low_threshold[0]) == 0.8
    assert float(cfg.clip_high_threshold[0]) == 1.2
