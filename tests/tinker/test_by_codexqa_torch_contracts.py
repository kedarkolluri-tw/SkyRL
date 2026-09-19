"""Written by CodexQA: CPU contracts for CISPO and Tinker optimizer settings.

The two production modules normally import the complete Ray/FSDP stack.  These
tests extract the named production functions from their AST and execute those
exact function bodies with tiny dependency stubs.  This keeps pure math and
normalization regressions runnable on a Mac without weakening the assertions
or copying the implementation into the test.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.codexqa

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = REPO_ROOT / "skyrl" / "backends" / "skyrl_train_backend.py"
PPO_UTILS_PATH = REPO_ROOT / "skyrl" / "backends" / "skyrl_train" / "utils" / "ppo_utils.py"


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


def _normalizer(logger=None):
    return _compile_method(
        BACKEND_PATH,
        "SkyRLTrainBackend",
        "_normalize_policy_loss_request",
        {
            "math": math,
            "logger": logger or _RecordingLogger(),
            "_CISPO_DEFAULT_CLIP_LOW": _literal_assignment(BACKEND_PATH, "_CISPO_DEFAULT_CLIP_LOW"),
            "_CISPO_DEFAULT_CLIP_HIGH": _literal_assignment(BACKEND_PATH, "_CISPO_DEFAULT_CLIP_HIGH"),
        },
    )


def test_cispo_public_bounds_are_translated_to_the_nested_torch_config():
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


def test_cispo_scale_rl_bounds_preserve_zero_instead_of_treating_it_as_missing():
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
        {"clip_high_threshold": 0.0},
        {"clip_low_threshold": 2.0, "clip_high_threshold": 1.0},
        {"clip_low_threshold": 10.0},
        {"clip_low_threshold": -0.1},
        {"clip_low_threshold": float("nan")},
        {"clip_high_threshold": float("inf")},
    ],
)
def test_cispo_invalid_or_degenerate_effective_bounds_are_rejected(config):
    with pytest.raises(ValueError, match="cispo"):
        _normalizer()(None, "policy", "cispo", config)


def test_equal_positive_cispo_bounds_are_allowed_and_warned():
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


def test_effective_optimizer_metrics_report_every_ignored_adam_setting():
    logger = _RecordingLogger()
    report = _compile_method(
        BACKEND_PATH,
        "SkyRLTrainBackend",
        "_effective_optimizer_metrics",
        {
            "logger": logger,
            "_FRAMEWORK_DEFAULT_ADAM_EPS": _literal_assignment(
                BACKEND_PATH, "_FRAMEWORK_DEFAULT_ADAM_EPS"
            ),
        },
    )
    backend = SimpleNamespace(
        _cfg=SimpleNamespace(
            trainer=SimpleNamespace(
                policy=SimpleNamespace(
                    optimizer_config=SimpleNamespace(
                        adam_betas=[0.9, 0.999],
                        weight_decay=0.01,
                    )
                )
            )
        )
    )
    requested = SimpleNamespace(beta1=0.8, beta2=0.95, eps=1e-12, weight_decay=0.0)

    metrics = report(backend, requested)

    assert metrics == pytest.approx(
        {
            "skyrl.ai/effective_beta1": 0.9,
            "skyrl.ai/effective_beta2": 0.999,
            "skyrl.ai/effective_weight_decay": 0.01,
            "skyrl.ai/effective_eps": 1e-8,
        }
    )
    warning_text = "\n".join(logger.warnings)
    for field in ("beta1", "beta2", "weight_decay", "eps"):
        assert f"AdamParams.{field}" in warning_text


def test_optim_step_returns_the_effective_settings_with_the_applied_learning_rate():
    class Dispatch:
        def __init__(self):
            self.calls = []

        def set_lr(self, role, learning_rate, model_id):
            self.calls.append(("set_lr", role, learning_rate, model_id))

        def optim_step(self, role, model_id):
            self.calls.append(("optim_step", role, model_id))
            return 2.5

    def output_type(metrics):
        return SimpleNamespace(metrics=metrics)

    optim_step = _compile_method(
        BACKEND_PATH,
        "SkyRLTrainBackend",
        "optim_step",
        {
            "logger": _RecordingLogger(),
            "types": SimpleNamespace(OptimStepOutput=output_type),
        },
    )
    dispatch = Dispatch()
    effective = {
        "skyrl.ai/effective_beta1": 0.9,
        "skyrl.ai/effective_beta2": 0.999,
        "skyrl.ai/effective_weight_decay": 0.01,
        "skyrl.ai/effective_eps": 1e-8,
    }
    backend = SimpleNamespace(
        _dispatch=dispatch,
        _get_role=lambda model_id: "policy",
        _effective_optimizer_metrics=lambda params: dict(effective),
    )
    adam = SimpleNamespace(
        learning_rate=0.001,
        beta1=0.8,
        beta2=0.95,
        eps=1e-12,
        weight_decay=0.0,
    )

    output = optim_step(backend, "adapter-a", SimpleNamespace(adam_params=adam))

    assert dispatch.calls == [
        ("set_lr", "policy", 0.001, "adapter-a"),
        ("optim_step", "policy", "adapter-a"),
    ]
    assert output.metrics == pytest.approx(
        {
            "skyrl.ai/grad_norm": 2.5,
            "skyrl.ai/learning_rate": 0.001,
            **effective,
        }
    )
