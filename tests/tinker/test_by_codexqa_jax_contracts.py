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
    _clip_by_global_norm,
)
from skyrl.tinker.loss_fns import LossFnConfig, cispo_loss  # noqa: E402
from skyrl.tinker.types import LOSS_TYPES  # noqa: E402

pytestmark = pytest.mark.codexqa

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_PATH = REPO_ROOT / "skyrl" / "train" / "config" / "config.py"
JAX_BACKEND_PATH = REPO_ROOT / "skyrl" / "backends" / "jax.py"


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


# ----------------------------------------------------------------------
# An all-empty batch must COMPLETE, not hang.
# ----------------------------------------------------------------------


def test_both_public_jax_entry_points_complete_an_all_empty_batch():
    """Round 3's fix landed only in SkyRLTrainBackend, so JAX still hung.

    The engine completes only the futures present in the returned dict, so
    `return {}` leaves every request of an all-empty batch pending forever.
    Both forward and forward_backward funnel through _model_pass, so this
    drives them through that shared helper rather than testing it directly.
    """
    import ast as _ast
    from types import SimpleNamespace as _NS

    captured = []

    def _fb_output(loss_fn_output_type, loss_fn_outputs, metrics):
        rec = _NS(
            loss_fn_output_type=loss_fn_output_type,
            loss_fn_outputs=loss_fn_outputs,
            metrics=metrics,
        )
        captured.append(rec)
        return rec

    src = JAX_BACKEND_PATH.read_text()
    tree = _ast.parse(src, filename=str(JAX_BACKEND_PATH))
    cls = next(n for n in tree.body if isinstance(n, _ast.ClassDef) and n.name == "JaxBackendImpl")

    ns = {
        "types": _NS(ForwardBackwardOutput=_fb_output),
        "LOSS_TYPES": {"cross_entropy": object()},
    }
    compiled = {}
    for name in ("_model_pass", "forward", "forward_backward"):
        node = next(
            n
            for n in cls.body
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and n.name == name
        )
        node.decorator_list = []
        mod = _ast.Module(body=[node], type_ignores=[])
        _ast.fix_missing_locations(mod)
        local: dict = dict(ns)
        exec(compile(mod, str(JAX_BACKEND_PATH), "exec"), local)
        compiled[name] = local[name]

    batch = _NS(
        all_model_inputs=[],
        all_loss_fns=[],
        request_batch_slices=[("r1", "m", 0, 0), ("r2", "m", 0, 0)],
    )

    for entry in ("forward", "forward_backward"):
        backend = _NS(
            _model_pass=lambda pb, fn, _mp=compiled["_model_pass"]: _mp(backend_self, pb, fn),
            _forward=object(),
            _forward_backward_and_accumulate=object(),
        )
        backend_self = backend
        results = compiled[entry](backend, batch)
        assert set(results) == {"r1", "r2"}, (
            f"JAX {entry} returned {results!r} for an all-empty batch; every request "
            "must be completed or the caller waits forever"
        )


# ----------------------------------------------------------------------
# grad_clip_norm on the JAX backend.
#
# Round 5 carried the field API -> internal type and wired it into the
# FSDP/Megatron path. JAX still dropped it: optim_step read learning_rate,
# betas, eps and weight_decay and never touched grad_clip_norm, so a client
# asking for clipping got none and never heard about it.
# ----------------------------------------------------------------------

_GRADS = {"a": None, "b": None}


def _tree(scale=1.0):
    return {"a": jnp.array([3.0, 0.0]) * scale, "b": jnp.array([[4.0]]) * scale}


def _gnorm(tree):
    return float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(tree))))


def test_clip_by_global_norm_treats_zero_as_disabled():
    """Tinker's contract: 0.0 means NO clipping, not "clip to zero".

    Reading 0.0 as a threshold is the whole bug -- it would scale every
    gradient by 0/||g|| and annihilate the update rather than leave it
    alone.
    """
    grads = _tree()          # global norm is exactly 5.0
    out = _clip_by_global_norm(grads, jnp.float32(0.0), jnp.float32(_gnorm(grads)))
    assert _gnorm(out) == pytest.approx(5.0), (
        f"grad_clip_norm=0.0 changed the gradient (norm {_gnorm(out)}); 0.0 is "
        "Tinker's default and means clipping is OFF"
    )
    for k in grads:
        np.testing.assert_allclose(np.asarray(out[k]), np.asarray(grads[k]))


def test_clip_by_global_norm_leaves_a_gradient_under_the_threshold_alone():
    grads = _tree()
    out = _clip_by_global_norm(grads, jnp.float32(100.0), jnp.float32(_gnorm(grads)))
    assert _gnorm(out) == pytest.approx(5.0)


def test_clip_by_global_norm_scales_a_gradient_over_the_threshold_to_it():
    grads = _tree()          # 5.0
    out = _clip_by_global_norm(grads, jnp.float32(1.0), jnp.float32(_gnorm(grads)))
    assert _gnorm(out) == pytest.approx(1.0, rel=1e-6), (
        "a gradient above the threshold must come out AT the threshold"
    )
    # direction preserved: every leaf scaled by the same factor
    for k in grads:
        np.testing.assert_allclose(
            np.asarray(out[k]), np.asarray(grads[k]) * 0.2, rtol=1e-6
        )


def test_clip_by_global_norm_does_not_nan_on_a_zero_gradient():
    """The guarded denominator, tested where it actually bites.

    A first version of this asserted "no NaN in the output" and could not
    fail: both arms of a jnp.where are evaluated, but the 0/0 lands in the
    UNSELECTED arm, so the returned value is clean either way. It passed
    against a deliberately unguarded implementation.

    The guard is still load-bearing, just not there. This backend supports
    enforce_eager, and in eager mode jax_debug_nans turns that discarded
    0/0 into a FloatingPointError that aborts the optim_step. Measured:
    unguarded raises, guarded does not.
    """
    zero = {"a": jnp.zeros(2), "b": jnp.zeros((1, 1))}
    out = _clip_by_global_norm(zero, jnp.float32(1.0), jnp.float32(0.0))
    for k in out:
        assert not bool(jnp.any(jnp.isnan(out[k]))), f"{k} came back NaN"
        np.testing.assert_allclose(np.asarray(out[k]), 0.0)

    # the eager + debug_nans path, which is the one that can actually crash
    jax.config.update("jax_debug_nans", True)
    try:
        out = _clip_by_global_norm(zero, jnp.float32(0.0), jnp.float32(0.0))
    except FloatingPointError as e:                      # pragma: no cover
        raise AssertionError(
            "a zero gradient with clipping disabled hit an unguarded 0/0: "
            f"{e}. Under enforce_eager this aborts the optimizer step."
        ) from e
    finally:
        jax.config.update("jax_debug_nans", False)
    np.testing.assert_allclose(np.asarray(out["a"]), 0.0)


def test_jax_optim_step_passes_grad_clip_norm_to_the_update():
    """The consumption half: the value must REACH the update, not just exist.

    Codex found this by supplying an AdamParams whose grad_clip_norm raises
    when read, and watching JAX complete the step without ever touching it.
    This does the stronger version -- record what the update actually
    received -- so it fails both if the field is dropped and if the wrong
    value is forwarded.

    optim_step is lifted out by AST and run against stubs, the same trick
    the all-empty-batch test above uses, because the real one needs a mesh,
    a compiled update and a loaded model.
    """
    import ast as _ast
    from contextlib import nullcontext
    from types import SimpleNamespace as _NS

    src = JAX_BACKEND_PATH.read_text()
    tree = _ast.parse(src, filename=str(JAX_BACKEND_PATH))
    cls = next(n for n in tree.body if isinstance(n, _ast.ClassDef) and n.name == "JaxBackendImpl")
    node = next(
        n for n in cls.body
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and n.name == "optim_step"
    )
    node.decorator_list = []
    mod = _ast.Module(body=[node], type_ignores=[])
    _ast.fix_missing_locations(mod)

    seen: dict = {}

    ns = {
        "jax": _NS(set_mesh=lambda _m: nullcontext(), device_get=lambda x: x),
        "jnp": _NS(int32=int, float32=float),
        "logger": _NS(info=lambda *a, **k: None, warning=lambda *a, **k: None),
        "types": _NS(OptimStepOutput=lambda metrics: _NS(metrics=metrics)),
    }
    local: dict = dict(ns)
    exec(compile(mod, str(JAX_BACKEND_PATH), "exec"), local)
    optim_step = local["optim_step"]

    def _update(grads, params, optimizer, adapter_index, grad_clip_norm):
        seen["grad_clip_norm"] = grad_clip_norm
        return grads, _NS(to_output_metrics=lambda: {})

    hp = {k: np.zeros(()) for k in ("learning_rate", "b1", "b2", "eps", "weight_decay")}
    backend = _NS(
        models={"m": _NS(adapter_index=1)},
        optimizers={"m": _NS(opt_state=_NS(hyperparams=hp))},
        accumulated_grads=_NS(counts=np.array([0, 1])),
        lora_params=object(),
        mesh=object(),
        _compute_grads_and_update=_update,
    )
    request = _NS(adam_params=_NS(
        learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
        weight_decay=0.0, grad_clip_norm=2.5,
    ))

    optim_step(backend, "m", request)

    assert "grad_clip_norm" in seen, (
        "JAX optim_step ran the update without ever passing grad_clip_norm; the "
        "client asked for clipping and silently got none"
    )
    assert seen["grad_clip_norm"] == pytest.approx(2.5), (
        f"the update received grad_clip_norm={seen['grad_clip_norm']!r}, not the "
        "2.5 the request carried"
    )
