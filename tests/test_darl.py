"""Unit tests for DARL (proposal sections 5.1 to 5.3 and roadmap section 2A).

Every expected value below is computed by hand from the proposal equations,
never by calling the code under test.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from darl_study.activations import DARL, darl_slope, darl_tracking, inactive_fraction

CUDA = torch.cuda.is_available()


def make(num_units=3, **kw):
    kw.setdefault("alpha_min", 0.01)
    kw.setdefault("alpha_max", 0.3)
    kw.setdefault("p", 1.0)
    kw.setdefault("rho", 0.9)
    return DARL(num_units, **kw)


def train_step(m, z):
    """One marked training forward (state update + activation)."""
    m.train()
    with darl_tracking(m):
        return m(z)


# --------------------------------------------------------------- 1. output
class TestOutput:
    def test_positive_inputs_pass_unchanged(self):
        m = make()
        z = torch.tensor([[0.5, 2.0, 3.0]])
        assert torch.equal(m.eval()(z), z)

    def test_negative_inputs_scaled_by_alpha_min_when_D_is_zero(self):
        m = make()  # D(0) = 0 -> alpha = alpha_min = 0.01
        z = torch.tensor([[-1.0, -2.0, -4.0]])
        expected = torch.tensor([[-0.01, -0.02, -0.04]])
        assert torch.allclose(m.eval()(z), expected)

    def test_negative_inputs_use_per_unit_alpha(self):
        m = make()
        m.reset_state(torch.tensor([0.0, 0.5, 1.0]))
        # alpha = 0.01 + 0.29 * D  ->  [0.01, 0.155, 0.30]
        z = torch.tensor([[-2.0, -2.0, -2.0]])
        expected = torch.tensor([[-0.02, -0.31, -0.60]])
        assert torch.allclose(m.eval()(z), expected, atol=1e-6)

    def test_zero_maps_to_zero(self):
        m = make()
        z = torch.zeros(4, 3)
        assert torch.equal(m.eval()(z), z)

    def test_conv_input_uses_one_alpha_per_channel(self):
        m = make(num_units=2)
        m.reset_state(torch.tensor([0.0, 1.0]))  # alpha = [0.01, 0.30]
        z = -torch.ones(1, 2, 2, 2)
        out = m.eval()(z)
        assert torch.allclose(out[:, 0], torch.full((1, 2, 2), -0.01))
        assert torch.allclose(out[:, 1], torch.full((1, 2, 2), -0.30))

    def test_wrong_unit_count_raises(self):
        with pytest.raises(ValueError):
            make(num_units=3)(torch.zeros(2, 4))

    def test_unbatched_input_raises(self):
        with pytest.raises(ValueError):
            make(num_units=3)(torch.zeros(3))

    def test_check_finite_flag(self):
        m = make(check_finite=True)
        with pytest.raises(FloatingPointError):
            m(torch.tensor([[float("nan"), 0.0, 0.0]]))

    @pytest.mark.parametrize(
        "kw",
        [
            {"alpha_min": 0.0},
            {"alpha_min": 0.4, "alpha_max": 0.3},
            {"alpha_max": 1.5},
            {"p": 0.0},
            {"rho": 1.0},
            {"rho": -0.1},
            {"d_init": 1.5},
        ],
    )
    def test_invalid_hyperparameters_raise(self, kw):
        with pytest.raises(ValueError):
            make(**kw)


# ----------------------------------------------------------- 2. EMA state
class TestDeadnessEMA:
    def test_inactive_fraction_counts_nonpositive_including_zero(self):
        z = torch.tensor(
            [
                [-1.0, 1.0, 0.0],
                [-2.0, 2.0, -0.0],
                [3.0, 3.0, 5.0],
                [-4.0, 4.0, 6.0],
            ]
        )
        # unit0: 3 of 4 <= 0; unit1: 0 of 4; unit2: 0.0 and -0.0 -> 2 of 4
        assert torch.allclose(inactive_fraction(z), torch.tensor([0.75, 0.0, 0.5]))

    def test_inactive_fraction_pools_batch_and_space_for_conv(self):
        z = torch.ones(2, 2, 2, 2)
        z[0, 0, 0, 0] = -1.0  # channel 0: 1 of 8 inactive
        z[:, 1] = -1.0  # channel 1: 8 of 8 inactive
        assert torch.allclose(inactive_fraction(z), torch.tensor([0.125, 1.0]))

    def test_single_update_matches_hand_computation(self):
        m = make(rho=0.9)
        z = torch.tensor([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0]])  # r = [0.5, 1, 0]
        train_step(m, z)
        # D = 0.9 * 0 + 0.1 * r
        assert torch.allclose(m.D, torch.tensor([0.05, 0.10, 0.0]))
        assert int(m.num_updates) == 1

    def test_sequence_matches_closed_form(self):
        rho, d0 = 0.8, 0.3
        m = make(num_units=1, rho=rho, d_init=d0)
        rs = [1.0, 0.0, 0.5, 1.0, 0.25]
        for r in rs:
            n_inactive = int(r * 4)
            z = torch.tensor([[-1.0]] * n_inactive + [[1.0]] * (4 - n_inactive))
            train_step(m, z)
        t = len(rs)
        expected = rho**t * d0 + (1 - rho) * sum(
            rho ** (t - 1 - k) * r for k, r in enumerate(rs)
        )
        assert math.isclose(m.D.item(), expected, rel_tol=1e-6)

    def test_rho_zero_sets_D_to_current_batch_fraction(self):
        m = make(rho=0.0, d_init=0.7)
        z = torch.tensor([[-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]])
        train_step(m, z)
        assert torch.allclose(m.D, torch.tensor([1.0, 0.0, 0.5]))

    def test_slope_uses_state_after_this_batch_update(self):
        """Decision D2: alpha(t) is computed from D(t), which includes batch t."""
        m = make(num_units=1, rho=0.5)
        z = torch.tensor([[-1.0], [-1.0]])  # r = 1 -> D(1) = 0.5
        out = train_step(m, z)
        alpha_t = 0.01 + 0.29 * 0.5  # from D(1), not D(0) = 0
        assert torch.allclose(out, torch.tensor([[-alpha_t], [-alpha_t]]))

    def test_no_update_without_tracking_marker(self):
        m = make()
        m.train()
        m(-torch.ones(4, 3))  # unmarked forward, e.g. a metric pass
        assert torch.equal(m.D, torch.zeros(3))
        assert int(m.num_updates) == 0

    def test_tracking_context_restores_flag(self):
        m = make()
        with darl_tracking(m):
            assert m.update_enabled
        assert not m.update_enabled

    def test_tracking_reaches_nested_modules(self):
        net = torch.nn.Sequential(torch.nn.Linear(2, 3), make(), torch.nn.Linear(3, 1))
        net.train()
        with darl_tracking(net):
            net(torch.randn(8, 2))
        assert int(net[1].num_updates) == 1

    def test_state_stays_in_unit_interval(self):
        torch.manual_seed(0)
        m = make(num_units=5, rho=0.3)
        for _ in range(200):
            train_step(m, torch.randn(16, 5) * 10)
            assert torch.all(m.D >= 0) and torch.all(m.D <= 1)

    def test_always_inactive_unit_converges_to_one(self):
        m = make(num_units=1, rho=0.9)
        for _ in range(300):
            train_step(m, -torch.ones(8, 1))
        # 1 - 0.9**300 is 1 to float32 precision
        assert math.isclose(m.D.item(), 1.0, abs_tol=1e-6)
        assert math.isclose(m.alpha.item(), 0.3, abs_tol=1e-6)


# ------------------------------------------------------ 3. slope behaviour
class TestSlope:
    @pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 4.0])
    def test_endpoints(self, p):
        D = torch.tensor([0.0, 1.0])
        assert torch.allclose(darl_slope(D, 0.01, 0.3, p), torch.tensor([0.01, 0.3]))

    @pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 4.0])
    def test_monotone_nondecreasing_and_strict_for_positive_D(self, p):
        D = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
        a = darl_slope(D, 0.01, 0.3, p)
        diffs = a[1:] - a[:-1]
        assert torch.all(diffs >= 0)
        assert torch.all(diffs[1:] > 0)  # strict once D > 0 and alpha_max > alpha_min

    def test_equal_bounds_give_constant_slope(self):
        D = torch.linspace(0, 1, 11)
        assert torch.allclose(darl_slope(D, 0.2, 0.2, 2.0), torch.full((11,), 0.2))

    def test_equal_bounds_reduce_to_leaky_relu(self):
        """Property P6: alpha_min = alpha_max gives LeakyReLU (away from x = 0)."""
        torch.manual_seed(0)
        m = make(num_units=4, alpha_min=0.05, alpha_max=0.05)
        z = torch.randn(32, 4)
        assert torch.allclose(train_step(m, z), F.leaky_relu(z, 0.05))

    def test_p_half_matches_formula(self):
        D = torch.tensor([0.25])
        expected = 0.01 + 0.29 * 0.5
        assert math.isclose(darl_slope(D, 0.01, 0.3, 0.5).item(), expected, rel_tol=1e-6)

    def test_slope_drift_per_step_is_bounded(self):
        """Property P5: |delta alpha| <= (alpha_max - alpha_min) * p * (1 - rho), p >= 1."""
        torch.manual_seed(1)
        rho, p = 0.9, 2.0
        m = make(num_units=6, rho=rho, p=p)
        bound = (0.3 - 0.01) * p * (1 - rho) + 1e-6
        prev = m.alpha.clone()
        for _ in range(100):
            train_step(m, torch.randn(4, 6))
            assert torch.all((m.alpha - prev).abs() <= bound)
            prev = m.alpha.clone()


# ------------------------------------------------------------ 4. autograd
class TestAutograd:
    def test_input_gradient_is_one_or_alpha(self):
        m = make()
        m.reset_state(torch.tensor([0.0, 0.5, 1.0]))
        m.eval()
        z = torch.tensor([[1.0, -1.0, -3.0], [-2.0, 2.0, 0.5]], requires_grad=True)
        m(z).sum().backward()
        expected = torch.tensor([[1.0, 0.155, 0.30], [0.01, 1.0, 1.0]])
        assert torch.allclose(z.grad, expected, atol=1e-6)

    def test_gradient_at_exactly_zero_is_one(self):
        m = make().eval()
        z = torch.zeros(1, 3, requires_grad=True)
        m(z).sum().backward()
        assert torch.equal(z.grad, torch.ones(1, 3))

    def test_negative_side_gradient_never_zero(self):
        """Property P1 / H1: derivative on x < 0 is >= alpha_min > 0."""
        torch.manual_seed(0)
        m = make(num_units=8)
        z = -torch.rand(64, 8) - 0.1
        z.requires_grad_(True)
        train_step(m, z).sum().backward()
        assert torch.all(z.grad >= 0.01 - 1e-7)
        assert torch.all(z.grad <= 1.0)

    def test_gradcheck_away_from_zero(self):
        m = make(num_units=4).double().eval()
        m.reset_state(torch.tensor([0.1, 0.4, 0.7, 1.0]))
        torch.manual_seed(0)
        z = torch.randn(5, 4, dtype=torch.float64)
        z = z + 0.2 * z.sign()  # keep every entry at least 0.2 away from the kink
        z.requires_grad_(True)
        assert torch.autograd.gradcheck(m, (z,), eps=1e-6, atol=1e-6)

    def test_no_trainable_parameters_and_no_grad_to_state(self):
        m = make()
        assert list(m.parameters()) == []
        assert not m.D.requires_grad
        z = torch.randn(4, 3, requires_grad=True)
        train_step(m, z).sum().backward()
        assert m.D.grad is None

    def test_gradient_flows_to_upstream_weights_of_inactive_units(self):
        """A unit that is negative on every input still passes alpha-scaled gradient."""
        torch.manual_seed(0)
        lin = torch.nn.Linear(2, 3)
        with torch.no_grad():
            lin.bias.fill_(-100.0)  # every preactivation strongly negative
        net = torch.nn.Sequential(lin, make())
        out = train_step(net[1], net[0](torch.randn(16, 2)))
        out.sum().backward()
        assert lin.weight.grad.abs().sum() > 0
        assert torch.all(lin.bias.grad != 0)


# ---------------------------------------------------------- 5. eval mode
class TestEvalMode:
    def test_eval_never_updates_even_when_marked(self):
        m = make()
        m.eval()
        with darl_tracking(m):
            m(-torch.ones(4, 3))
        assert torch.equal(m.D, torch.zeros(3))
        assert int(m.num_updates) == 0

    def test_eval_uses_frozen_training_state(self):
        m = make(num_units=1, rho=0.5)
        train_step(m, -torch.ones(2, 1))  # D = 0.5
        frozen = m.alpha.clone()
        m.eval()
        out = m(torch.tensor([[-1.0], [1.0]]))
        assert torch.allclose(out[0], -frozen)
        assert torch.allclose(m.alpha, frozen)

    def test_eval_output_of_a_sample_does_not_depend_on_batch(self):
        torch.manual_seed(0)
        m = make(num_units=4)
        m.reset_state(torch.tensor([0.2, 0.4, 0.6, 0.8]))
        m.eval()
        x = torch.randn(1, 4)
        alone = m(x)
        in_batch = m(torch.cat([x, -10 * torch.ones(31, 4)]))[:1]
        assert torch.equal(alone, in_batch)


# -------------------------------------------------- 6. state dict / resume
class TestStateDict:
    def test_state_dict_holds_state_and_hyperparameters(self):
        sd = make().state_dict()
        assert {"D", "num_updates", "_extra_state"} <= set(sd)
        assert sd["_extra_state"]["hparams"]["rho"] == 0.9

    def test_save_load_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        m = make(num_units=4)
        for _ in range(5):
            train_step(m, torch.randn(8, 4))
        path = tmp_path / "darl.pt"
        torch.save(m.state_dict(), path)
        m2 = make(num_units=4)
        m2.load_state_dict(torch.load(path, weights_only=True))
        assert torch.equal(m2.D, m.D)
        assert int(m2.num_updates) == 5
        z = torch.randn(6, 4)
        assert torch.equal(m2.eval()(z), m.eval()(z))

    def test_load_rejects_mismatched_hyperparameters(self):
        sd = make(rho=0.9).state_dict()
        with pytest.raises(ValueError):
            make(rho=0.99).load_state_dict(sd)

    def test_resume_matches_uninterrupted_run(self, tmp_path):
        torch.manual_seed(0)
        batches = [torch.randn(8, 3) for _ in range(10)]
        full = make()
        for b in batches:
            train_step(full, b)

        first = make()
        for b in batches[:4]:
            train_step(first, b)
        path = tmp_path / "mid.pt"
        torch.save(first.state_dict(), path)
        resumed = make()
        resumed.load_state_dict(torch.load(path, weights_only=True))
        for b in batches[4:]:
            train_step(resumed, b)

        assert torch.equal(resumed.D, full.D)
        assert int(resumed.num_updates) == int(full.num_updates) == 10

    def test_state_saved_inside_a_full_model_checkpoint(self, tmp_path):
        net = torch.nn.Sequential(torch.nn.Linear(2, 3), make())
        train_step(net[1], net[0](torch.randn(4, 2)))
        path = tmp_path / "net.pt"
        torch.save(net.state_dict(), path)
        net2 = torch.nn.Sequential(torch.nn.Linear(2, 3), make())
        net2.load_state_dict(torch.load(path, weights_only=True))
        assert torch.equal(net2[1].D, net[1].D)


# --------------------------------------------------- 7. dtype and device
class TestDtypeDevice:
    def test_half_conversion_keeps_state_float32(self):
        m = make().half()
        assert m.D.dtype == torch.float32

    def test_bfloat16_forward_keeps_input_dtype(self):
        m = make()
        z = torch.randn(4, 3).to(torch.bfloat16)
        out = train_step(m, z)
        assert out.dtype == torch.bfloat16
        assert m.D.dtype == torch.float32

    @pytest.mark.skipif(not CUDA, reason="CUDA not available on this machine")
    def test_cuda_matches_cpu(self):
        torch.manual_seed(0)
        batches = [torch.randn(16, 3) for _ in range(5)]
        cpu, gpu = make(), make().cuda()
        for b in batches:
            out_cpu = train_step(cpu, b)
            out_gpu = train_step(gpu, b.cuda())
        assert torch.allclose(gpu.D.cpu(), cpu.D, atol=1e-6)
        assert torch.allclose(out_gpu.cpu(), out_cpu, atol=1e-6)
