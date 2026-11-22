"""Integration tests: DARL inside a real training loop.

1. Optimizer-step integration: a small MLP with DARL layers is trained with
   real optimizers (SGD with momentum, Adam). Checks that weights change, D is
   updated exactly once per tracked step and matches an independent EMA
   recomputation, eval/untracked passes leave D untouched, and gradients reach
   every parameter but never D.
2. Complete checkpoint resume: model, optimizer, DARL state, global step and
   RNG state are saved to disk mid-run, loaded into freshly built objects, and
   training continues. The resumed run must be bitwise identical on CPU to an
   uninterrupted run (parameters, D, num_updates, optimizer state, losses).
"""

import random

import numpy as np
import pytest
import torch
from torch import nn

from darl_study.activations import DARL, darl_slope, darl_tracking, inactive_fraction

IN, HID, OUT = 6, 8, 3


def make_model(rho=0.9, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(IN, HID), DARL(HID, rho=rho),
        nn.Linear(HID, HID), DARL(HID, rho=rho),
        nn.Linear(HID, OUT),
    )


def darl_layers(model):
    return [m for m in model.modules() if isinstance(m, DARL)]


def make_optimizer(name, model):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    return torch.optim.Adam(model.parameters(), lr=1e-2)


def fixed_data(n=64, seed=123):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN, generator=g)
    y = torch.randint(0, OUT, (n,), generator=g)
    return x, y


def train_step(model, opt, x, y):
    model.train()
    opt.zero_grad(set_to_none=True)
    with darl_tracking(model):
        loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return loss.detach()


# ------------------------------------------------- 1. optimizer integration
@pytest.mark.parametrize("opt_name", ["sgd", "adam"])
class TestOptimizerStep:
    def test_weights_change_and_state_counts_steps(self, opt_name):
        model = make_model()
        opt = make_optimizer(opt_name, model)
        x, y = fixed_data()
        w0 = [p.detach().clone() for p in model.parameters()]
        steps = 7
        for _ in range(steps):
            loss = train_step(model, opt, x, y)
            assert torch.isfinite(loss)
        for p, p0 in zip(model.parameters(), w0):
            assert not torch.equal(p, p0)
        for m in darl_layers(model):
            assert int(m.num_updates) == steps
            assert torch.isfinite(m.D).all()
            assert ((m.D >= 0) & (m.D <= 1)).all()

    def test_D_matches_independent_ema_of_observed_preactivations(self, opt_name):
        # Record each DARL input with a forward pre-hook and recompute the EMA
        # by hand. D is updated before the slope is applied (decision D2).
        model = make_model(rho=0.8)
        opt = make_optimizer(opt_name, model)
        x, y = fixed_data()
        layers = darl_layers(model)
        seen = {id(m): [] for m in layers}
        hooks = [m.register_forward_pre_hook(
            lambda mod, inp: seen[id(mod)].append(inp[0].detach().clone()))
            for m in layers]
        for _ in range(5):
            train_step(model, opt, x, y)
        for h in hooks:
            h.remove()
        for m in layers:
            D = torch.zeros(HID)
            for z in seen[id(m)]:
                D = 0.8 * D + 0.2 * inactive_fraction(z)
            assert torch.allclose(m.D, D, atol=1e-6)
            assert torch.allclose(m.alpha, darl_slope(D, 0.01, 0.3, 1.0), atol=1e-6)

    def test_gradients_reach_parameters_not_state(self, opt_name):
        model = make_model()
        opt = make_optimizer(opt_name, model)
        x, y = fixed_data()
        model.train()
        opt.zero_grad(set_to_none=True)
        with darl_tracking(model):
            loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
        assert any(p.grad.abs().sum() > 0 for p in model.parameters())
        for m in darl_layers(model):
            assert not m.D.requires_grad
            assert m.D.grad is None
            assert "D" not in dict(m.named_parameters())

    def test_eval_and_untracked_passes_do_not_change_state(self, opt_name):
        model = make_model()
        opt = make_optimizer(opt_name, model)
        x, y = fixed_data()
        for _ in range(3):
            train_step(model, opt, x, y)
        snap = [(m.D.clone(), int(m.num_updates)) for m in darl_layers(model)]
        model.eval()
        with torch.no_grad():
            model(x)
            with darl_tracking(model):  # eval mode: still frozen
                model(x)
        model.train()
        with torch.no_grad():
            model(x)  # train mode but not tracked (metric pass)
        for m, (D0, n0) in zip(darl_layers(model), snap):
            assert torch.equal(m.D, D0)
            assert int(m.num_updates) == n0

    def test_loss_decreases_on_fixed_batch(self, opt_name):
        model = make_model()
        opt = make_optimizer(opt_name, model)
        x, y = fixed_data()
        losses = [float(train_step(model, opt, x, y)) for _ in range(60)]
        assert losses[-1] < losses[0]


# -------------------------------------------------- 2. checkpoint and resume
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_steps(model, opt, start, stop, losses):
    """Each step draws a fresh random batch from the global RNG, so resume
    is only exact if the RNG state is restored too."""
    for step in range(start, stop):
        x = torch.randn(32, IN)
        y = torch.randint(0, OUT, (32,))
        # Dropout-like randomness from numpy and python to cover every RNG.
        scale = 1.0 + 0.01 * float(np.random.rand()) + 0.01 * random.random()
        losses.append(train_step(model, opt, x * scale, y))


def save_checkpoint(path, model, opt, step):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, path)


def load_checkpoint(path, model, opt):
    # weights_only=False: the checkpoint holds numpy/python RNG tuples. Only
    # load checkpoints you wrote yourself.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["rng"]["torch"])
    np.random.set_state(ck["rng"]["numpy"])
    random.setstate(ck["rng"]["python"])
    return ck["step"]


def assert_optimizer_states_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    assert sa["state"].keys() == sb["state"].keys()
    for k in sa["state"]:
        for name, va in sa["state"][k].items():
            vb = sb["state"][k][name]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), (k, name)
            else:
                assert va == vb, (k, name)


@pytest.mark.parametrize("opt_name", ["sgd", "adam"])
@pytest.mark.parametrize("split", [1, 6])
def test_checkpoint_resume_is_bitwise_identical(tmp_path, opt_name, split):
    total = 12

    # Uninterrupted reference run.
    seed_everything(7)
    ref_model = make_model()
    ref_opt = make_optimizer(opt_name, ref_model)
    ref_losses = []
    run_steps(ref_model, ref_opt, 0, total, ref_losses)

    # Interrupted run: train `split` steps, save, discard everything.
    seed_everything(7)
    model = make_model()
    opt = make_optimizer(opt_name, model)
    losses = []
    run_steps(model, opt, 0, split, losses)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, split)
    del model, opt

    # Scramble every RNG to prove the checkpoint restores it.
    seed_everything(999)
    torch.randn(100)

    # Fresh objects (different init seed), load, continue.
    model = make_model(seed=42)
    opt = make_optimizer(opt_name, model)
    for m in darl_layers(model):
        assert int(m.num_updates) == 0
    start = load_checkpoint(path, model, opt)
    assert start == split
    for m in darl_layers(model):
        assert int(m.num_updates) == split
    run_steps(model, opt, start, total, losses)

    assert len(losses) == len(ref_losses) == total
    for i, (a, b) in enumerate(zip(losses, ref_losses)):
        assert torch.equal(a, b), f"loss differs at step {i}"
    for (n, p), (_, q) in zip(model.named_parameters(), ref_model.named_parameters()):
        assert torch.equal(p, q), n
    for m, r in zip(darl_layers(model), darl_layers(ref_model)):
        assert torch.equal(m.D, r.D)
        assert int(m.num_updates) == int(r.num_updates) == total
    assert_optimizer_states_equal(opt, ref_opt)


def test_resume_without_restoring_D_would_diverge(tmp_path):
    """Sanity check that the resume test is sensitive: dropping D from the
    checkpoint changes the continued run."""
    seed_everything(7)
    model = make_model()
    opt = make_optimizer("sgd", model)
    run_steps(model, opt, 0, 4, [])
    sd = {k: v for k, v in model.state_dict().items() if not k.endswith(".D")}
    fresh = make_model()
    missing, _ = fresh.load_state_dict(sd, strict=False)
    assert any(k.endswith(".D") for k in missing)
    for a, b in zip(darl_layers(fresh), darl_layers(model)):
        assert not torch.equal(a.D, b.D)


def test_checkpoint_with_mismatched_hparams_is_rejected(tmp_path):
    model = make_model(rho=0.9)
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    other = make_model(rho=0.5)
    with pytest.raises(ValueError, match="hyperparameters"):
        other.load_state_dict(torch.load(path, weights_only=True))
