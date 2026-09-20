"""Written by CodexQA: CPU contracts for CISPO and Tinker optimizer settings.

The two production modules normally import the complete Ray/FSDP stack.  These
tests extract the named production functions from their AST and execute those
exact function bodies with tiny dependency stubs.  This keeps pure math and
normalization regressions runnable on a Mac without weakening the assertions
or copying the implementation into the test.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.codexqa

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = REPO_ROOT / "skyrl" / "backends" / "skyrl_train_backend.py"
PPO_UTILS_PATH = REPO_ROOT / "skyrl" / "backends" / "skyrl_train" / "utils" / "ppo_utils.py"
WORKER_PATH = REPO_ROOT / "skyrl" / "backends" / "skyrl_train" / "workers" / "worker.py"
MEGATRON_WORKER_PATH = (
    REPO_ROOT / "skyrl" / "backends" / "skyrl_train" / "workers" / "megatron" / "megatron_worker.py"
)
TRAIN_CONFIG_PATH = REPO_ROOT / "skyrl" / "train" / "config" / "config.py"
API_PATH = REPO_ROOT / "skyrl" / "tinker" / "api.py"


class _RecordingLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, message):
        self.warnings.append(str(message))

    def info(self, message):
        self.infos.append(str(message))


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        candidate
        for candidate in tree.body
        if isinstance(candidate, (ast.Assign, ast.AnnAssign))
        and (
            (isinstance(candidate, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in candidate.targets))
            or (isinstance(candidate, ast.AnnAssign) and isinstance(candidate.target, ast.Name) and candidate.target.id == name)
        )
    )
    return ast.literal_eval(node.value)


def _compile_function(path: Path, function_name: str, namespace: dict):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        candidate
        for candidate in tree.body
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.name == function_name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def _compile_method(path: Path, class_name: str, method_name: str, namespace: dict):
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        candidate
        for candidate in tree.body
        if isinstance(candidate, ast.ClassDef) and candidate.name == class_name
    )
    node = next(
        candidate
        for candidate in cls.body
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.name == method_name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _class_field_defaults(path: Path, class_name: str) -> dict:
    """Literal field defaults off a class in the real source.

    Importing skyrl.train.config.config for real drags in omegaconf,
    skyrl_gym, pandas and the rest of the training stack, which is the whole
    point of this file being runnable without them. Reading the literals out of
    the source keeps the check honest anyway: if CISPOConfig's declared default
    moves, this moves with it, which is exactly the drift being guarded.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    out = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _cispo_config_stub():
    fields = _class_field_defaults(TRAIN_CONFIG_PATH, "CISPOConfig")
    module = ModuleType("skyrl.train.config.config")
    module.CISPOConfig = lambda: SimpleNamespace(**fields)
    return module


@pytest.fixture
def stub_train_config(monkeypatch):
    for name in ("skyrl", "skyrl.train", "skyrl.train.config"):
        pkg = ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, "skyrl.train.config.config", _cispo_config_stub())


def _default_thresholds():
    return _compile_function(BACKEND_PATH, "_cispo_default_thresholds", {})


def _normalizer(logger=None):
    return _compile_method(
        BACKEND_PATH,
        "SkyRLTrainBackend",
        "_normalize_policy_loss_request",
        {
            "math": math,
            "logger": logger or _RecordingLogger(),
            "_cispo_default_thresholds": _default_thresholds(),
        },
    )


def test_cispo_public_bounds_are_translated_to_the_nested_torch_config(stub_train_config):
    normalize = _normalizer()
    loss_name, config = normalize(
        None,
        "policy",
        "cispo",
        {"clip_low_threshold": 0.2, "clip_high_threshold": 2.0},
    )
    assert loss_name == "cispo"
    assert config == {
        "cispo": {
            "cispo_eps_clip_low": 0.8,
            "cispo_eps_clip_high": 1.0,
        }
    }


def test_cispo_scale_rl_bounds_preserve_zero_instead_of_treating_it_as_missing(stub_train_config):
    _, config = _normalizer()(
        None,
        "policy",
        "cispo",
        {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0},
    )
    assert config == {
        "cispo": {
            "cispo_eps_clip_low": 1.0,
            "cispo_eps_clip_high": 4.0,
        }
    }


@pytest.mark.parametrize(
    "config",
    [
        {"clip_low_threshold": 2.0, "clip_high_threshold": 1.0},
        # 10 alone is still inverted against the DEFAULT high, which is why the
        # omitted bound has to be resolved before the order check.
        {"clip_low_threshold": 10.0},
        {"clip_low_threshold": float("nan")},
        {"clip_high_threshold": float("inf")},
    ],
)
def test_cispo_rejects_only_non_finite_and_inverted_bounds(config, stub_train_config):
    with pytest.raises(ValueError, match="cispo"):
        _normalizer()(None, "policy", "cispo", config)


@pytest.mark.parametrize(
    "config, expected_warning",
    [
        # high == 0: the clipped ratio really is identically zero.
        ({"clip_high_threshold": 0.0}, "cannot learn"),
        # high < 0: NOT zero -- a negative constant multiplier, so the gradient
        # is reversed. The earlier error message claimed every high <= 0 gave a
        # zero gradient; that was wrong, and this pins the correction.
        ({"clip_low_threshold": -2.0, "clip_high_threshold": -1.0}, "reversed"),
        # low < 0 never binds against a strictly positive ratio: harmless.
        ({"clip_low_threshold": -0.1}, None),
    ],
)
def test_cispo_strange_but_expressible_bounds_are_allowed(config, expected_warning, stub_train_config):
    """The adapter implements the request contract; it does not invent policy."""
    logger = _RecordingLogger()
    _, normalized = _normalizer(logger)(None, "policy", "cispo", config)
    assert normalized is not None
    if expected_warning is None:
        assert not logger.warnings
    else:
        assert any(expected_warning in message for message in logger.warnings)


def test_cispo_defaults_come_from_cispo_config_not_a_second_copy(stub_train_config):
    """Guards the drift Codex flagged: duplicated (0, 5) constants in the backend."""
    declared = _class_field_defaults(TRAIN_CONFIG_PATH, "CISPOConfig")
    assert _default_thresholds()() == pytest.approx(
        (1.0 - declared["cispo_eps_clip_low"], 1.0 + declared["cispo_eps_clip_high"])
    )


def test_equal_positive_cispo_bounds_are_allowed_and_warned(stub_train_config):
    logger = _RecordingLogger()
    normalize = _normalizer(logger)
    _, config = normalize(
        None,
        "policy",
        "cispo",
        {"clip_low_threshold": 1.0, "clip_high_threshold": 1.0},
    )
    assert config == {
        "cispo": {
            "cispo_eps_clip_low": 0.0,
            "cispo_eps_clip_high": 0.0,
        }
    }
    assert any("plain REINFORCE" in message for message in logger.warnings)


def _masked_mean(values, mask, dim=None):
    if mask is None:
        return values.mean(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(1e-8)
    return (values * mask).sum(dim=dim) / denominator


def _cispo_loss_function():
    def safe_exp_delta(delta, clip, out_dtype):
        return torch.exp(torch.clamp(delta, min=-clip, max=clip)).to(out_dtype)

    def apply_off_policy_correction(loss, old_log_probs, rollout_logprobs, loss_mask, config):
        return loss, loss_mask, {}

    def reduce_loss(loss, loss_mask):
        return (loss * loss_mask).sum() if loss_mask is not None else loss.sum()

    return _compile_function(
        PPO_UTILS_PATH,
        "compute_policy_loss_cispo",
        {
            "torch": torch,
            "safe_exp_delta": safe_exp_delta,
            "masked_mean": _masked_mean,
            "apply_off_policy_correction": apply_off_policy_correction,
            "reduce_loss": reduce_loss,
        },
    )


def _algorithm_config(low_bound: float, high_bound: float):
    return SimpleNamespace(
        cispo=SimpleNamespace(
            cispo_anchor="old",
            cispo_eps_clip_low=1.0 - low_bound,
            cispo_eps_clip_high=high_bound - 1.0,
        ),
        off_policy_correction=SimpleNamespace(tis_ratio_type=None),
    )


def test_equal_positive_bounds_have_the_reinforce_gradient_not_zero_gradient():
    loss_fn = _cispo_loss_function()
    log_probs = torch.tensor([[-1.2, 0.3]], dtype=torch.float64, requires_grad=True)
    old_log_probs = torch.zeros_like(log_probs)
    advantages = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
    mask = torch.ones_like(log_probs)

    loss, _ = loss_fn(
        log_probs,
        old_log_probs,
        advantages,
        _algorithm_config(1.0, 1.0),
        mask,
        None,
    )
    loss.backward()

    assert torch.equal(log_probs.grad, -advantages)
    assert torch.count_nonzero(log_probs.grad).item() == 2


def test_only_a_zero_cispo_multiplier_zeroes_the_policy_gradient():
    loss_fn = _cispo_loss_function()
    log_probs = torch.tensor([[-1.2, 0.3]], dtype=torch.float64, requires_grad=True)
    old_log_probs = torch.zeros_like(log_probs)
    advantages = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
    mask = torch.ones_like(log_probs)

    loss, _ = loss_fn(
        log_probs,
        old_log_probs,
        advantages,
        _algorithm_config(0.0, 0.0),
        mask,
        None,
    )
    loss.backward()

    assert torch.equal(log_probs.grad, torch.zeros_like(log_probs))


def _optim_step(namespace_extra=None):
    ns = {
        "logger": _RecordingLogger(),
        "types": SimpleNamespace(OptimStepOutput=lambda metrics: SimpleNamespace(metrics=metrics)),
    }
    ns.update(namespace_extra or {})
    return _compile_method(BACKEND_PATH, "SkyRLTrainBackend", "optim_step", ns)


def _assert_honoured():
    return _compile_method(
        BACKEND_PATH, "SkyRLTrainBackend", "_assert_optimizer_honoured", {"logger": _RecordingLogger()}
    )


class _Dispatch:
    """Records the call ORDER, which is the thing under test."""

    def __init__(self, effective, effective_clip=0.0):
        self.calls = []
        self._effective = effective
        self._effective_clip = effective_clip

    def set_grad_clip_norm(self, role, grad_clip_norm, model_id=None):
        self.calls.append(("set_grad_clip_norm", role, grad_clip_norm, model_id))

    def get_grad_clip_norm(self, role, model_id=None):
        self.calls.append(("get_grad_clip_norm", role, model_id))
        return self._effective_clip

    def set_lr(self, role, learning_rate, model_id):
        self.calls.append(("set_lr", role, learning_rate, model_id))

    def set_adam_hyperparams(self, role, model_id=None, **kwargs):
        self.calls.append(("set_adam_hyperparams", role, model_id, dict(sorted(kwargs.items()))))

    def get_adam_hyperparams(self, role, model_id=None):
        self.calls.append(("get_adam_hyperparams", role, model_id))
        return dict(self._effective) if self._effective is not None else None

    def optim_step(self, role, model_id):
        self.calls.append(("optim_step", role, model_id))
        return 2.5


ADAM = SimpleNamespace(
    learning_rate=0.001, beta1=0.8, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0
)
HONOURED = {"beta1": 0.8, "beta2": 0.95, "eps": 1e-12, "weight_decay": 0.0, "lr": 0.001}


def _backend(dispatch):
    return SimpleNamespace(
        _dispatch=dispatch,
        _get_role=lambda model_id: "policy",
        _assert_optimizer_honoured=staticmethod(_assert_honoured()).__func__,
        _assert_grad_clip_honoured=staticmethod(
            _compile_method(
                BACKEND_PATH, "SkyRLTrainBackend", "_assert_grad_clip_honoured",
                {"logger": _RecordingLogger()},
            )
        ).__func__,
    )


def test_optim_step_applies_every_adam_setting_before_stepping():
    """The old code applied lr only, then warned AFTER the wrong step was taken."""
    dispatch = _Dispatch(HONOURED)
    output = _optim_step()(_backend(dispatch), "adapter-a", SimpleNamespace(adam_params=ADAM))

    assert dispatch.calls == [
        ("set_lr", "policy", 0.001, "adapter-a"),
        (
            "set_adam_hyperparams",
            "policy",
            "adapter-a",
            {"beta1": 0.8, "beta2": 0.95, "eps": 1e-12, "weight_decay": 0.0},
        ),
        ("set_grad_clip_norm", "policy", 0.0, "adapter-a"),
        ("get_adam_hyperparams", "policy", "adapter-a"),
        ("get_grad_clip_norm", "policy", "adapter-a"),
        ("optim_step", "policy", "adapter-a"),
    ]
    names = [c[0] for c in dispatch.calls]
    assert names.index("set_adam_hyperparams") < names.index("optim_step")

    assert output.metrics == pytest.approx(
        {
            "skyrl.ai/grad_norm": 2.5,
            "skyrl.ai/learning_rate": 0.001,
            "skyrl.ai/effective_beta1": 0.8,
            "skyrl.ai/effective_beta2": 0.95,
            "skyrl.ai/effective_eps": 1e-12,
            "skyrl.ai/effective_weight_decay": 0.0,
            "skyrl.ai/effective_grad_clip_norm": 0.0,
        }
    )


def test_optim_step_reports_the_read_back_values_not_the_requested_ones():
    """A backend that quietly keeps its own settings must not be reported as compliant."""
    dispatch = _Dispatch({**HONOURED, "weight_decay": 0.01})
    with pytest.raises(RuntimeError) as excinfo:
        _optim_step()(_backend(dispatch), "adapter-a", SimpleNamespace(adam_params=ADAM))
    assert "weight_decay" in str(excinfo.value)
    assert "asked 0.0" in str(excinfo.value)
    # and it failed BEFORE taking the step with the wrong settings
    assert "optim_step" not in [c[0] for c in dispatch.calls]


def test_optim_step_fails_when_the_optimizer_cannot_be_read_back():
    """None means no optimizer, or ranks that disagree. Either way there is no evidence."""
    dispatch = _Dispatch(None)
    with pytest.raises(RuntimeError, match="no evidence"):
        _optim_step()(_backend(dispatch), "adapter-a", SimpleNamespace(adam_params=ADAM))
    assert "optim_step" not in [c[0] for c in dispatch.calls]


def test_optim_step_fails_when_the_optimizer_does_not_expose_a_setting():
    """A missing key would otherwise read as 'applied' while doing nothing."""
    missing = {k: v for k, v in HONOURED.items() if k != "eps"}
    with pytest.raises(RuntimeError, match="eps"):
        _optim_step()(_backend(_Dispatch(missing)), "adapter-a", SimpleNamespace(adam_params=ADAM))


def test_optim_step_uses_the_resolved_role_not_a_hardcoded_policy():
    dispatch = _Dispatch(HONOURED)
    backend = _backend(dispatch)
    backend._get_role = lambda model_id: "critic"
    _optim_step()(backend, "adapter-c", SimpleNamespace(adam_params=ADAM))
    assert {c[1] for c in dispatch.calls} == {"critic"}


# ----------------------------------------------------------------------
# The worker-side setters, against a REAL torch optimizer.
# ----------------------------------------------------------------------


def _worker_methods():
    ns = {"Optional": object}
    return (
        _compile_method(WORKER_PATH, "Worker", "_optimizer_param_groups", ns),
        _compile_method(WORKER_PATH, "Worker", "set_adam_hyperparams", ns),
        _compile_method(WORKER_PATH, "Worker", "get_adam_hyperparams", ns),
        _compile_method(WORKER_PATH, "Worker", "_group_hyperparams", ns),
    )


def _worker_with(optimizer):
    groups, setter, getter, per_group = _worker_methods()
    worker = SimpleNamespace(optimizer=optimizer)
    worker._optimizer_param_groups = lambda: groups(worker)
    worker._group_hyperparams = per_group
    return worker, setter, getter


def test_worker_sets_adam_hyperparams_on_a_real_adamw():
    param = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    worker, setter, getter = _worker_with(opt)

    setter(worker, beta1=0.8, beta2=0.95, eps=1e-12, weight_decay=0.0)

    assert opt.param_groups[0]["betas"] == (0.8, 0.95)
    assert opt.param_groups[0]["eps"] == 1e-12
    assert opt.param_groups[0]["weight_decay"] == 0.0
    assert getter(worker) == pytest.approx(
        {"beta1": 0.8, "beta2": 0.95, "eps": 1e-12, "weight_decay": 0.0, "lr": 0.1}
    )


def test_worker_setters_touch_every_param_group_not_just_the_first():
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([{"params": [a]}, {"params": [b]}], lr=0.1)
    worker, setter, _ = _worker_with(opt)

    setter(worker, weight_decay=0.25)

    assert [g["weight_decay"] for g in opt.param_groups] == [0.25, 0.25]


def test_worker_setters_change_the_actual_update_not_just_the_bookkeeping():
    """weight_decay=0 vs 0.01 must produce different weights after one step."""

    def step_with(weight_decay):
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.full((2,), 3.0))
        opt = torch.optim.AdamW([param], lr=0.1, weight_decay=0.0)
        worker, setter, _ = _worker_with(opt)
        setter(worker, weight_decay=weight_decay)
        param.grad = torch.zeros_like(param)  # only decay can move it
        opt.step()
        return param.detach().clone()

    assert torch.equal(step_with(0.0), torch.full((2,), 3.0))
    assert not torch.equal(step_with(0.01), torch.full((2,), 3.0))


def test_worker_reports_none_without_an_optimizer():
    worker, _, getter = _worker_with(None)
    assert getter(worker) is None


def test_megatron_worker_flattens_chained_optimizer_param_groups():
    """The override is REDUNDANT, not load-bearing -- pin that, do not pretend otherwise.

    An earlier version of this test asserted ChainedOptimizer.param_groups
    omits its inner optimizers' groups. That premise is false: the pinned
    Megatron class aggregates them
    (megatron/core/optimizer/optimizer.py, ChainedOptimizer.param_groups).
    So the fake below aggregates too, and the test asserts only that the
    override returns every inner group -- which is what matters, and is true
    whether the base class would have managed it or not. SkyRL's own set_lr
    carries the same override, so removing it is a separate question.
    """
    inner_a = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    inner_b = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=0.1)

    class FakeChained:
        def __init__(self, opts):
            self.chained_optimizers = opts

        @property
        def param_groups(self):  # models the real class: aggregated
            return [g for o in self.chained_optimizers for g in o.param_groups]

    groups = _compile_method(
        MEGATRON_WORKER_PATH,
        "MegatronPolicyWorkerBase",
        "_optimizer_param_groups",
        {"ChainedOptimizer": FakeChained},
    )
    worker = SimpleNamespace(optimizer=FakeChained([inner_a, inner_b]))
    assert groups(worker) == inner_a.param_groups + inner_b.param_groups


def test_readback_fails_when_param_groups_disagree():
    """Megatron builds several groups. Reporting group 0 would hide the rest."""
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW(
        [
            {"params": [a], "betas": (0.8, 0.95), "eps": 1e-12, "weight_decay": 0.0},
            {"params": [b], "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        ],
        lr=0.1,
    )
    worker, _, getter = _worker_with(opt)
    assert getter(worker) is None, "divergent groups must not report as a single honoured value"


def test_setter_then_readback_agrees_across_all_groups():
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW(
        [
            {"params": [a], "betas": (0.8, 0.95), "eps": 1e-12, "weight_decay": 0.0},
            {"params": [b], "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        ],
        lr=0.1,
    )
    worker, setter, getter = _worker_with(opt)
    setter(worker, beta1=0.8, beta2=0.95, eps=1e-12, weight_decay=0.0)
    assert getter(worker) == pytest.approx(
        {"beta1": 0.8, "beta2": 0.95, "eps": 1e-12, "weight_decay": 0.0, "lr": 0.1}
    )


def test_two_steps_match_the_closed_form_for_the_REQUESTED_betas_and_eps():
    """Numerically pin beta1, beta2 and eps -- weight_decay alone did not.

    Two steps with DIFFERENT gradients are required. With a constant gradient
    the bias-corrected first moment is g at every t, so the betas cancel and
    the test would pass whatever they were set to.
    """
    lr, b1, b2, eps = 0.1, 0.8, 0.95, 1e-3
    g1, g2 = 0.5, -0.25

    param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.AdamW([param], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    worker, setter, _ = _worker_with(opt)
    setter(worker, beta1=b1, beta2=b2, eps=eps, weight_decay=0.0)

    for g in (g1, g2):
        param.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()

    # closed-form AdamW from zero state, weight_decay = 0
    m1, v1 = (1 - b1) * g1, (1 - b2) * g1**2
    d1 = lr * (m1 / (1 - b1**1)) / ((v1 / (1 - b2**1)) ** 0.5 + eps)
    m2 = b1 * m1 + (1 - b1) * g2
    v2 = b2 * v1 + (1 - b2) * g2**2
    d2 = lr * (m2 / (1 - b1**2)) / ((v2 / (1 - b2**2)) ** 0.5 + eps)
    expected = -(d1 + d2)

    assert param.item() == pytest.approx(expected, rel=1e-9)

    # and it is NOT what the optimizer's construction-time settings would give
    ref = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    ref_opt = torch.optim.AdamW([ref], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for g in (g1, g2):
        ref.grad = torch.tensor([g], dtype=torch.float64)
        ref_opt.step()
    assert abs(ref.item() - expected) > 1e-6, "the requested betas/eps made no difference"


def test_priming_step_counter_is_reset_so_the_first_client_step_is_t1():
    """Megatron primes Adam state with a dummy optimizer.step(), which lands
    the counter on 1. Left alone, every adapter's first real update runs bias
    correction at t=2 and is ~26.5% short."""

    class FakeInner:
        def __init__(self):
            self.param_groups = [{"step": 1, "lr": 0.1}, {"step": 1, "lr": 0.1}]
            self.state = {"p": {"step": torch.tensor(1.0)}, "q": {"step": 1}}

    class FakeOpt:
        def __init__(self):
            self.optimizer = FakeInner()

    reset = _compile_method(
        MEGATRON_WORKER_PATH,
        "MegatronPolicyWorkerBase",
        "_reset_optimizer_step_counters",
        {"iter_opts": lambda o: [o], "torch": torch},
    )
    opt = FakeOpt()
    reset(SimpleNamespace(optimizer=opt))

    assert [g["step"] for g in opt.optimizer.param_groups] == [0, 0]
    assert opt.optimizer.state["p"]["step"].item() == 0.0
    assert opt.optimizer.state["q"]["step"] == 0


def test_the_priming_step_is_worth_resetting():
    """Quantify it, so the fix is not taken on faith: t=2 really is ~0.735x."""
    # eps must be tiny-but-nonzero: the priming step has a ZERO gradient, so
    # eps=0 makes its update 0/0 = nan and the comparison is meaningless.
    lr, b1, b2, eps, g = 1.0, 0.9, 0.95, 1e-12, 3.0

    def move():
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        o = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0)
        p.grad = torch.tensor([g], dtype=torch.float64)
        o.step()
        return -p.item()

    def move_after_dummy():
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        o = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0)
        p.grad = torch.zeros(1, dtype=torch.float64)
        o.step()                       # the priming dummy step
        p.grad = torch.tensor([g], dtype=torch.float64)
        before = p.item()
        o.step()
        return -(p.item() - before)

    assert move() == pytest.approx(lr, rel=1e-9)
    assert move_after_dummy() == pytest.approx(0.7350, rel=1e-3)


# ----------------------------------------------------------------------
# Batch splitting. Two concurrently batched requests with different CISPO
# bounds used to train with whichever config was found first, after a warning.
# ----------------------------------------------------------------------


def _split():
    return _compile_method(
        BACKEND_PATH,
        "SkyRLTrainBackend",
        "_split_model_pass_batch",
        {"json": json, "types": SimpleNamespace(PreparedModelPassBatch=_FakeBatch)},
    )


class _FakeBatch(SimpleNamespace):
    pass


def _batch(rows):
    """rows: list of (request_id, model_id, loss_fn, config). One row each."""
    return _FakeBatch(
        all_model_inputs=[f"tok{i}" for i in range(len(rows))],
        all_targets=list(range(len(rows))),
        all_token_weights=list(range(len(rows))),
        all_sampling_logprobs=list(range(len(rows))),
        all_advantages=list(range(len(rows))),
        all_values=list(range(len(rows))),
        all_returns=list(range(len(rows))),
        all_model_ids=[r[1] for r in rows],
        all_loss_fns=[r[2] for r in rows],
        all_loss_fn_configs=[r[3] for r in rows],
        request_batch_slices=[(r[0], r[1], i, i + 1) for i, r in enumerate(rows)],
    )


def _backend_for_split():
    return SimpleNamespace(
        _get_role=lambda model_id: "policy",
        _loss_key=_compile_method(BACKEND_PATH, "SkyRLTrainBackend", "_loss_key", {"json": json}),
    )


def test_same_model_different_cispo_bounds_are_not_trained_together():
    tight = {"clip_low_threshold": 0.99, "clip_high_threshold": 1.01}
    wide = {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
    batch = _batch([("r1", "m", "cispo", tight), ("r2", "m", "cispo", wide)])

    subs = _split()(_backend_for_split(), batch)

    assert len(subs) == 2
    assert [s.all_loss_fn_configs for s in subs] == [[tight], [wide]]


def test_identical_requests_still_share_one_batch():
    cfg = {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
    batch = _batch([("r1", "m", "cispo", cfg), ("r2", "m", "cispo", dict(cfg))])
    assert _split()(_backend_for_split(), batch) == [batch]


def test_key_order_does_not_split_an_otherwise_identical_config():
    """Same computation written in a different key order must still batch."""
    a = {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
    b = {"clip_high_threshold": 5.0, "clip_low_threshold": 0.0}
    batch = _batch([("r1", "m", "cispo", a), ("r2", "m", "cispo", b)])
    assert _split()(_backend_for_split(), batch) == [batch]


def test_different_loss_functions_are_not_trained_together():
    batch = _batch([("r1", "m", "cispo", None), ("r2", "m", "ppo", None)])
    subs = _split()(_backend_for_split(), batch)
    assert [s.all_loss_fns for s in subs] == [["cispo"], ["ppo"]]


def test_model_id_still_splits():
    batch = _batch([("r1", "a", "cispo", None), ("r2", "b", "cispo", None)])
    assert len(_split()(_backend_for_split(), batch)) == 2


def test_one_request_carrying_two_objectives_is_rejected():
    batch = _batch([("r1", "m", "cispo", {"clip_high_threshold": 5.0}), ("r1", "m", "cispo", None)])
    batch.request_batch_slices = [("r1", "m", 0, 2)]
    with pytest.raises(ValueError, match="more than one"):
        _split()(_backend_for_split(), batch)


def test_forward_backward_refuses_a_sub_batch_that_was_not_split():
    """Belt and braces: if the split regresses, the backward pass must not proceed."""
    fb = _compile_method(
        BACKEND_PATH,
        "SkyRLTrainBackend",
        "_forward_backward_single_model_batch",
        {"json": json, "logger": _RecordingLogger(), "types": SimpleNamespace()},
    )
    backend = SimpleNamespace(
        _get_batch_role=lambda ids: "policy",
        _loss_key=_compile_method(BACKEND_PATH, "SkyRLTrainBackend", "_loss_key", {"json": json}),
    )
    batch = _batch(
        [
            ("r1", "m", "cispo", {"clip_high_threshold": 1.01}),
            ("r2", "m", "cispo", {"clip_high_threshold": 5.0}),
        ]
    )
    with pytest.raises(ValueError, match="distinct"):
        fb(backend, batch)


def test_an_empty_request_does_not_create_a_zero_row_sub_batch():
    """Empty requests are API-valid. Keying them on their own builds a
    zero-row sub-batch that all_loss_fns[0] then raises IndexError on."""
    batch = _batch(
        [
            ("r1", "m", "cispo", {"clip_high_threshold": 1.01}),
            ("r2", "m", "cispo", {"clip_high_threshold": 5.0}),
        ]
    )
    # r0 is an empty slice alongside two genuinely different objectives
    batch.request_batch_slices = [("r0", "m", 0, 0)] + list(batch.request_batch_slices)

    subs = _split()(_backend_for_split(), batch)

    assert all(len(sub.all_loss_fns) > 0 for sub in subs), "a sub-batch has no rows"
    assert len(subs) == 2
    assert {"r0"} <= {rid for sub in subs for rid, *_ in sub.request_batch_slices}


def test_a_batch_of_only_empty_requests_is_left_alone():
    batch = _batch([("r1", "m", "cispo", None)])
    batch.request_batch_slices = [("r1", "m", 0, 0), ("r2", "m", 0, 0)]
    batch.all_loss_fns = ["cispo", "ppo"]          # force the split path
    batch.all_loss_fn_configs = [None, None]
    batch.all_model_ids = ["m", "m"]
    assert _split()(_backend_for_split(), batch) == [batch]



# ----------------------------------------------------------------------
# Round 3: priming must leave the adapter pristine, wd_mult must survive,
# and an all-empty batch must still complete.
# ----------------------------------------------------------------------


def _megatron_method(name, ns=None):
    base = {"iter_opts": lambda o: [o], "torch": torch, "contextmanager": contextmanager}
    base.update(ns or {})
    return _compile_method(MEGATRON_WORKER_PATH, "MegatronPolicyWorkerBase", name, base)


class _PrimedOptimizer:
    """Stands in for Megatron's DistributedOptimizer.

    `_init_optimizer_states_with_dummy_values` is a REAL AdamW step with
    zeroed gradients, which is what MCore does -- so the fake does exactly
    that against a real torch optimizer rather than pretending.
    """

    def __init__(self, weight_decay=0.2, lr=0.1):
        self.param = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        self.optimizer = torch.optim.AdamW(
            [self.param], lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay
        )

    def _init_optimizer_states_with_dummy_values(self):
        self.param.grad = torch.zeros_like(self.param)
        self.optimizer.step()
        self.optimizer.zero_grad()


def _prime_worker(opt):
    worker = SimpleNamespace(optimizer=opt, _is_lora=True)
    # _compile_method strips decorators, so @contextmanager has to be
    # reapplied here. The production function keeps its decorator.
    _suppress = contextmanager(_megatron_method("_weight_decay_suppressed"))
    worker._weight_decay_suppressed = lambda: _suppress(worker)
    worker._reset_optimizer_step_counters = lambda: _megatron_method(
        "_reset_optimizer_step_counters"
    )(worker)
    return worker


def test_priming_leaves_the_adapter_bit_identical_with_nonzero_weight_decay():
    """Decoupled decay does not need a gradient: p <- p - lr*wd*p fires on the
    dummy step too, so the 'pristine' snapshot would be of decayed weights."""
    opt = _PrimedOptimizer(weight_decay=0.2, lr=0.1)
    before = opt.param.detach().clone()

    _megatron_method("prime_optimizer_state")(_prime_worker(opt))

    assert torch.equal(opt.param.detach(), before), (
        f"priming moved the weights: {before.tolist()} -> {opt.param.detach().tolist()}"
    )
    assert opt.optimizer.param_groups[0]["weight_decay"] == 0.2, "weight_decay was not restored"


def test_priming_without_the_guard_really_would_decay():
    """Quantify what the guard prevents, so it is not taken on faith."""
    opt = _PrimedOptimizer(weight_decay=0.2, lr=0.1)
    opt._init_optimizer_states_with_dummy_values()          # no suppression
    assert opt.param.detach().tolist() == pytest.approx([0.98, 0.98])


def test_prime_optimizer_state_resets_the_counter_end_to_end():
    """Drives the PRODUCTION prime_optimizer_state, not just the reset helper.

    The earlier test called _reset_optimizer_step_counters directly, so
    deleting the call from prime_optimizer_state left every test green.
    """
    opt = _PrimedOptimizer(weight_decay=0.0, lr=0.1)
    _megatron_method("prime_optimizer_state")(_prime_worker(opt))
    for state in opt.optimizer.state.values():
        step = state.get("step")
        step = step.item() if torch.is_tensor(step) else step
        assert step == 0, f"step counter left at {step}; the first client update would be t=2"


def test_weight_decay_is_scaled_by_each_groups_wd_mult():
    """Megatron gives biases/1-D params wd_mult=0. Stamping the raw value into
    every group starts decaying parameters that are meant to be exempt."""
    decay = torch.nn.Parameter(torch.zeros(2))
    no_decay = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW(
        [
            {"params": [decay], "weight_decay": 0.0},
            {"params": [no_decay], "weight_decay": 0.0},
        ],
        lr=0.1,
    )
    opt.param_groups[0]["wd_mult"] = 1.0
    opt.param_groups[1]["wd_mult"] = 0.0
    worker, setter, getter = _worker_with(opt)

    setter(worker, weight_decay=0.05)

    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.05)
    assert opt.param_groups[1]["weight_decay"] == 0.0, "a wd_mult=0 group was given decay"
    # ...and the legitimate difference is NOT reported as a disagreement
    assert getter(worker)["weight_decay"] == pytest.approx(0.05)


def test_a_wd_mult_zero_group_holding_decay_is_a_failure():
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW(
        [{"params": [p1], "weight_decay": 0.05}, {"params": [p2], "weight_decay": 0.05}], lr=0.1
    )
    opt.param_groups[0]["wd_mult"] = 1.0
    opt.param_groups[1]["wd_mult"] = 0.0      # holds 0.05 it should not have
    worker, _, getter = _worker_with(opt)
    assert getter(worker) is None


def test_an_all_empty_batch_still_completes_every_request():
    """forward/forward_backward returned {} for an all-empty batch, and the
    engine only completes the futures it finds there -- so the request hung."""
    outputs = []

    def _fb_output(loss_fn_output_type, loss_fn_outputs, metrics):
        rec = SimpleNamespace(
            loss_fn_output_type=loss_fn_output_type, loss_fn_outputs=loss_fn_outputs, metrics=metrics
        )
        outputs.append(rec)
        return rec

    ns = {"types": SimpleNamespace(ForwardBackwardOutput=_fb_output), "logger": _RecordingLogger()}
    empty_results = _compile_method(BACKEND_PATH, "SkyRLTrainBackend", "_empty_pass_results", ns)

    batch = _batch([("r1", "m", "cispo", None)])
    batch.all_model_inputs = []
    batch.request_batch_slices = [("r1", "m", 0, 0), ("r2", "m", 0, 0)]

    for method in ("forward_backward", "forward"):
        fn = _compile_method(BACKEND_PATH, "SkyRLTrainBackend", method, ns)
        backend = SimpleNamespace(_empty_pass_results=empty_results)
        results = fn(backend, batch)
        assert set(results) == {"r1", "r2"}, f"{method} did not complete every request: {results}"


def test_the_gpu_module_exits_zero_when_invoked_standalone_without_cuda():
    """A module-level pytest.skip() aborts collection, and pytest then exits 5
    ("no tests collected") -- which CI reads as a failure even though the
    terminal prints "skipped". Source-order inspection cannot catch that; only
    running the command can, so this runs it.
    """
    import os
    import subprocess

    gpu_module = REPO_ROOT / "tests" / "tinker" / "skyrl_train" / "test_by_codexqa_megatron_adam.py"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         str(gpu_module)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, (
        f"standalone run exited {result.returncode} (5 = no tests collected). "
        f"The module must stay COLLECTABLE and skip via a mark.\n{result.stdout[-1500:]}"
    )
    assert "skipped" in result.stdout, result.stdout[-1500:]



def test_grad_clip_norm_is_applied_and_defaults_to_disabled():
    """Tinker defaults grad_clip_norm to 0.0 == NO clipping. SkyRL dropped the
    field entirely, so the torch backends silently clipped at
    OptimizerConfig.max_grad_norm = 1.0 instead."""
    dispatch = _Dispatch(HONOURED, effective_clip=0.0)
    out = _optim_step()(_backend(dispatch), "adapter-a", SimpleNamespace(adam_params=ADAM))
    assert ("set_grad_clip_norm", "policy", 0.0, "adapter-a") in dispatch.calls
    assert out.metrics["skyrl.ai/effective_grad_clip_norm"] == 0.0
    names = [c[0] for c in dispatch.calls]
    assert names.index("set_grad_clip_norm") < names.index("optim_step")


def test_a_nonzero_grad_clip_norm_is_passed_through_verbatim():
    adam = SimpleNamespace(**{**vars(ADAM), "grad_clip_norm": 2.5})
    dispatch = _Dispatch(HONOURED, effective_clip=2.5)
    out = _optim_step()(_backend(dispatch), "adapter-a", SimpleNamespace(adam_params=adam))
    assert ("set_grad_clip_norm", "policy", 2.5, "adapter-a") in dispatch.calls
    assert out.metrics["skyrl.ai/effective_grad_clip_norm"] == 2.5


def test_a_backend_that_keeps_its_own_clipping_fails_the_step():
    """The exact parity bug: request 0.0 (disabled), server holds 1.0."""
    dispatch = _Dispatch(HONOURED, effective_clip=1.0)
    with pytest.raises(RuntimeError, match="grad_clip_norm"):
        _optim_step()(_backend(dispatch), "adapter-a", SimpleNamespace(adam_params=ADAM))
    assert "optim_step" not in [c[0] for c in dispatch.calls]


def test_grad_clip_norm_survives_the_api_to_types_conversion():
    """to_types copies field by field, so a field added to both models but not
    there is still dropped. That was half of how this went missing."""
    src = API_PATH.read_text()
    tree = ast.parse(src, filename=str(API_PATH))
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "AdamParams"
    )
    fn = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "to_types"
    )
    kwargs = {k.arg for call in ast.walk(fn) if isinstance(call, ast.Call) for k in call.keywords}
    declared = {
        n.target.id for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    assert declared <= kwargs, f"to_types drops {sorted(declared - kwargs)}"
