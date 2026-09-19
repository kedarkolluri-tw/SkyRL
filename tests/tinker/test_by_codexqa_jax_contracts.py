"""Written by CodexQA: CPU-JAX contracts for CISPO defaults and reductions."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="install the SkyRL jax extra")
jnp = pytest.importorskip("jax.numpy", reason="install the SkyRL jax extra")
pytest.importorskip("flax", reason="install the SkyRL jax extra")
pytest.importorskip("optax", reason="install the SkyRL jax extra")

from skyrl.backends.jax import (  # noqa: E402
    _DEFAULT_CISPO_CLIP_HIGH_THRESHOLD,
    _DEFAULT_CISPO_CLIP_LOW_THRESHOLD,
    AccumulatedGradients,
    JaxBackendImpl,
)
from skyrl.tinker.loss_fns import LossFnConfig, cispo_loss  # noqa: E402
from skyrl.tinker.types import LOSS_TYPES  # noqa: E402

pytestmark = pytest.mark.codexqa

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_PATH = REPO_ROOT / "skyrl" / "train" / "config" / "config.py"


def _class_literal(class_name, field_name):
    tree = ast.parse(TRAIN_CONFIG_PATH.read_text(), filename=str(TRAIN_CONFIG_PATH))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    field = next(
        node
        for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == field_name
    )
    return ast.literal_eval(field.value)


def test_cispo_and_ppo_get_distinct_per_example_defaults():
    config = JaxBackendImpl._build_loss_fn_config(
        [None, None, None],
        [LOSS_TYPES["ppo"], LOSS_TYPES["cispo"], LOSS_TYPES["ppo"]],
    )
    assert np.asarray(config.clip_low_threshold).tolist() == pytest.approx([0.8, 0.0, 0.8])
    assert np.asarray(config.clip_high_threshold).tolist() == pytest.approx([1.2, 5.0, 1.2])
    assert _DEFAULT_CISPO_CLIP_LOW_THRESHOLD == 0.0
    assert _DEFAULT_CISPO_CLIP_HIGH_THRESHOLD == 5.0


def test_jax_cispo_defaults_equal_the_torch_backend_defaults():
    torch_eps_low = _class_literal("CISPOConfig", "cispo_eps_clip_low")
    torch_eps_high = _class_literal("CISPOConfig", "cispo_eps_clip_high")
    assert (
        _DEFAULT_CISPO_CLIP_LOW_THRESHOLD,
        _DEFAULT_CISPO_CLIP_HIGH_THRESHOLD,
    ) == (1.0 - torch_eps_low, 1.0 + torch_eps_high)


def test_jax_cispo_equal_positive_bounds_have_nonzero_reinforce_gradient():
    old_log_probs = jnp.zeros((2,), dtype=jnp.float32)
    advantages = jnp.array([2.0, -3.0], dtype=jnp.float32)
    mask = jnp.ones((2,), dtype=jnp.float32)
    config = LossFnConfig(
        clip_low_threshold=jnp.array(1.0, dtype=jnp.float32),
        clip_high_threshold=jnp.array(1.0, dtype=jnp.float32),
    )

    def total(log_probs):
        return cispo_loss(log_probs, mask, old_log_probs, advantages, config).sum()

    gradient = np.asarray(jax.grad(total)(jnp.array([-1.2, 0.3], dtype=jnp.float32)))
    np.testing.assert_array_equal(gradient, -np.asarray(advantages))
    assert np.count_nonzero(gradient) == 2


def _accumulated_value(chunks, reduction):
    accumulator = AccumulatedGradients.create({"w": jnp.zeros((2,), dtype=jnp.float32)}, 2)
    for per_sequence_values, token_counts in chunks:
        per_sequence_values = jnp.asarray(per_sequence_values, dtype=jnp.float32)
        mask = jnp.asarray(
            [[1.0] * count + [0.0] * (max(token_counts) - count) for count in token_counts],
            dtype=jnp.float32,
        )
        if reduction == "sequence_mean":
            total = (per_sequence_values / mask.sum(axis=-1)).sum()
        else:
            total = per_sequence_values.sum()
        gradients = {"w": jnp.array([total, 0.0], dtype=jnp.float32)}
        adapter_indices = jnp.zeros((len(per_sequence_values),), dtype=jnp.int32)
        accumulator = accumulator.add(gradients, adapter_indices, mask)
    return float(accumulator.get_mean(jnp.int32(0), reduction)["w"][0])


def test_sum_reduction_is_invariant_to_microbatch_partitioning():
    one_batch = [([2.0, 3.0, 4.0], [2, 3, 1])]
    three_microbatches = [([2.0], [2]), ([3.0], [3]), ([4.0], [1])]
    assert _accumulated_value(one_batch, "sum") == pytest.approx(9.0)
    assert _accumulated_value(three_microbatches, "sum") == pytest.approx(9.0)


def test_token_mean_uses_the_total_token_count_across_microbatches():
    chunks = [([2.0], [2]), ([3.0], [3]), ([4.0], [1])]
    assert _accumulated_value(chunks, "token_mean") == pytest.approx(9.0 / 6.0)


def test_sequence_mean_uses_the_total_sequence_count_across_microbatches():
    chunks = [([2.0], [2]), ([3.0], [3]), ([4.0], [1])]
    expected = ((2.0 / 2.0) + (3.0 / 3.0) + (4.0 / 1.0)) / 3.0
    assert _accumulated_value(chunks, "sequence_mean") == pytest.approx(expected)
