"""Unit tests for SkyRLTrainBackend._normalize_policy_loss_request.

No Ray runtime or GPUs are needed — the method is pure. Requires the
SkyRL-Train backend deps (ray/vllm) to be importable. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_loss_normalization.py
"""

from __future__ import annotations

import pytest

# Skip if skyrl_train_backend.py cannot be imported
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

_normalize = skyrl_train_backend.SkyRLTrainBackend._normalize_policy_loss_request


def test_ppo_thresholds_map_to_eps_clip():
    loss_fn, config = _normalize(None, "policy", "ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28})
    assert loss_fn == "regular"
    assert config == pytest.approx({"eps_clip_low": 0.2, "eps_clip_high": 0.28})


def test_dppo_deltas_are_nested_under_dppo():
    loss_fn, config = _normalize(None, "policy", "dppo", {"delta_low": 0.2, "delta_high": 0.3})
    assert loss_fn == "dppo"
    assert config == {"dppo": {"delta_low": 0.2, "delta_high": 0.3}}


def test_dppo_partial_deltas():
    loss_fn, config = _normalize(None, "policy", "dppo", {"delta_high": 0.05})
    assert loss_fn == "dppo"
    assert config == {"dppo": {"delta_high": 0.05}}


def test_dppo_without_config_passes_through():
    loss_fn, config = _normalize(None, "policy", "dppo", None)
    assert loss_fn == "dppo"
    assert config is None


def test_critic_config_passes_through_unchanged():
    loss_fn, config = _normalize(None, "critic", "ppo", {"value_clip": 0.2})
    assert loss_fn == "ppo"
    assert config == {"value_clip": 0.2}


# --- cispo (Tinker absolute thresholds -> nested CISPOConfig offsets) ---------
#
# Pre-patch behaviour, verified against a running FSDP Tinker server rather
# than inferred: the flat keys reached
# validate_dict_keys_against_dataclass and raised
#   ValueError: Invalid fields {'clip_low_threshold','clip_high_threshold'}
#              for AlgorithmConfig
# so cispo + any loss_fn_config was a hard 400. These are exact-equality
# assertions: every expected value is exactly representable
# (1.0-0.0, 5.0-1.0, 1.0-0.2, 2.0-1.0 are all exact doubles), so pytest.approx
# would only weaken them.


def test_cispo_thresholds_are_nested_under_cispo():
    """The ScaleRL bounds (0, 5) become the offsets CISPOConfig expects."""
    loss_fn, config = _normalize(None, "policy", "cispo", {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0})
    assert loss_fn == "cispo"
    # compute_policy_loss_cispo clamps to (1 - low, 1 + high), so (1.0, 4.0) -> (0, 5).
    assert config == {"cispo": {"cispo_eps_clip_low": 1.0, "cispo_eps_clip_high": 4.0}}


def test_cispo_non_default_thresholds_are_translated():
    """(0, 5) coincides with CISPOConfig's own defaults, so only a non-default
    pair distinguishes 'translated' from 'would have been rejected'."""
    loss_fn, config = _normalize(None, "policy", "cispo", {"clip_low_threshold": 0.2, "clip_high_threshold": 2.0})
    assert loss_fn == "cispo"
    assert config == {"cispo": {"cispo_eps_clip_low": 0.8, "cispo_eps_clip_high": 1.0}}


def test_cispo_partial_thresholds():
    loss_fn, config = _normalize(None, "policy", "cispo", {"clip_high_threshold": 5.0})
    assert loss_fn == "cispo"
    assert config == {"cispo": {"cispo_eps_clip_high": 4.0}}


def test_cispo_without_config_passes_through():
    """No config -> nothing to translate.

    NOTE this leaves a live cross-backend divergence that normalization cannot
    fix: the torch path then uses CISPOConfig's defaults -> bounds (0, 5),
    while the JAX backend's _build_loss_fn_config falls back to
    _DEFAULT_PPO_CLIP_LOW/HIGH_THRESHOLD (0.8, 1.2) for *every* loss fn
    including cispo. Same client request, two different objectives depending on
    which backend serves it.
    """
    loss_fn, config = _normalize(None, "policy", "cispo", None)
    assert loss_fn == "cispo"
    assert config is None


@pytest.mark.parametrize(
    "cfg, why",
    [
        ({"clip_low_threshold": 5.0, "clip_high_threshold": 0.0}, "low > high, high <= 0"),
        ({"clip_low_threshold": 2.0, "clip_high_threshold": 1.0}, "low > high, high > 0"),
        ({"clip_high_threshold": 0.0}, "non-positive upper bound"),
        ({"clip_high_threshold": -1.0}, "negative upper bound"),
        ({"clip_low_threshold": -0.5, "clip_high_threshold": 5.0}, "negative lower bound"),
        ({"clip_low_threshold": 10.0}, "low-only, inverted against the DEFAULT high of 5"),
        ({"clip_low_threshold": float("nan")}, "non-finite"),
        ({"clip_high_threshold": float("inf")}, "non-finite"),
    ],
)
def test_cispo_unusable_bounds_rejected(cfg, why):
    """Bounds that cannot express a usable objective must raise.

    Note the reasoning, because an earlier version of this branch had it wrong:
    CISPO detaches the clipped ratio and multiplies it by log_prob, so a
    constant POSITIVE multiplier still gives a nonzero REINFORCE-style
    gradient. Only a multiplier of zero kills the gradient. These cases are
    rejected either because the multiplier is zero (high <= 0) or because the
    bounds are transposed, which is a mistake rather than an intent.

    The low-only case matters separately: an omitted bound falls back to
    CISPOConfig's default, so clip_low_threshold=10 is inverted against the
    default high of 5 even though only one value was supplied.
    """
    with pytest.raises(ValueError, match="cispo"):
        _normalize(None, "policy", "cispo", cfg)


def test_cispo_equal_bounds_are_allowed():
    """low == high is plain REINFORCE -- a constant, nonzero multiplier.

    An earlier version rejected this as "zero gradient", which is false: the
    clipped ratio is detached, so a constant c > 0 still yields -adv * c *
    log_prob. It is unusual, so it warns, but it is a legitimate request.
    """
    loss_fn, config = _normalize(None, "policy", "cispo",
                                 {"clip_low_threshold": 1.0, "clip_high_threshold": 1.0})
    assert loss_fn == "cispo"
    assert config == {"cispo": {"cispo_eps_clip_low": 0.0, "cispo_eps_clip_high": 0.0}}


def test_cispo_lower_bound_zero_is_not_treated_as_absent():
    """clip_low_threshold=0.0 is falsy but meaningful -- it is the ScaleRL
    lower bound. A truthiness check would drop it and leave
    cispo_eps_clip_low at its default."""
    _, config = _normalize(None, "policy", "cispo",
                           {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0})
    assert "cispo_eps_clip_low" in config["cispo"]
    assert config["cispo"]["cispo_eps_clip_low"] == 1.0


def test_cispo_high_only_against_default_low_is_fine():
    """The mirror of the low-only case: high=3 against the default low of 0 is
    a perfectly ordinary request and must NOT be rejected."""
    loss_fn, config = _normalize(None, "policy", "cispo", {"clip_high_threshold": 3.0})
    assert loss_fn == "cispo"
    assert config == {"cispo": {"cispo_eps_clip_high": 2.0}}
