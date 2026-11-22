"""Regression tests for the three validation bugs found in the independent review.

1. An empty batch (0, num_units) was accepted and wrote NaN into D.
2. reset_state(float("nan")) was accepted.
3. unit_dim=0 (the batch dimension) was accepted.

Plus: finite-value checking is on by default and can be switched off.
"""

import math

import pytest
import torch

from darl_study.activations import DARL, darl_tracking, inactive_fraction


def train_step(m, z):
    m.train()
    with darl_tracking(m):
        return m(z)


# ------------------------------------------------------------ 1. empty batch
class TestEmptyBatch:
    @pytest.mark.parametrize("shape", [(0, 3), (0, 3, 4, 4), (2, 3, 0, 4)])
    def test_empty_batch_rejected_in_training_and_state_untouched(self, shape):
        m = DARL(3, rho=0.9)
        m.reset_state(torch.tensor([0.1, 0.2, 0.3]))
        before = m.D.clone()
        with pytest.raises(ValueError, match="empty"):
            train_step(m, torch.empty(shape))
        assert torch.equal(m.D, before)
        assert int(m.num_updates) == 0
        assert torch.isfinite(m.D).all()

    def test_empty_batch_rejected_in_eval(self):
        m = DARL(3).eval()
        with pytest.raises(ValueError, match="empty"):
            m(torch.empty(0, 3))

    def test_empty_batch_rejected_even_without_finite_check(self):
        m = DARL(3, check_finite=False)
        with pytest.raises(ValueError, match="empty"):
            train_step(m, torch.empty(0, 3))
        assert torch.equal(m.D, torch.zeros(3))

    def test_inactive_fraction_rejects_empty_tensor(self):
        with pytest.raises(ValueError, match="empty"):
            inactive_fraction(torch.empty(0, 3))


# --------------------------------------------------- 2. non-finite state values
class TestNonFiniteState:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_reset_state_rejects_nonfinite_scalar(self, bad):
        m = DARL(3)
        with pytest.raises(ValueError, match="finite"):
            m.reset_state(bad)
        assert torch.equal(m.D, torch.zeros(3))

    def test_reset_state_rejects_nonfinite_entry_in_vector(self):
        m = DARL(3)
        with pytest.raises(ValueError, match="finite"):
            m.reset_state(torch.tensor([0.1, float("nan"), 0.3]))
        assert torch.equal(m.D, torch.zeros(3))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_constructor_rejects_nonfinite_d_init(self, bad):
        with pytest.raises(ValueError):
            DARL(3, d_init=bad)

    @pytest.mark.parametrize(
        "kw",
        [
            {"alpha_min": float("nan")},
            {"alpha_max": float("nan")},
            {"p": float("nan")},
            {"p": float("inf")},
            {"rho": float("nan")},
        ],
    )
    def test_constructor_rejects_nonfinite_hyperparameters(self, kw):
        with pytest.raises(ValueError):
            DARL(3, **kw)

    def test_load_state_dict_rejects_nan_state(self):
        sd = DARL(3).state_dict()
        sd["D"] = torch.tensor([0.1, float("nan"), 0.2])
        m = DARL(3)
        with pytest.raises(ValueError, match="finite"):
            m.load_state_dict(sd)
        assert torch.equal(m.D, torch.zeros(3))

    def test_load_state_dict_rejects_out_of_range_state(self):
        sd = DARL(3).state_dict()
        sd["D"] = torch.tensor([0.1, 1.5, 0.2])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            DARL(3).load_state_dict(sd)

    def test_valid_reset_still_works(self):
        m = DARL(3)
        m.reset_state(torch.tensor([0.0, 0.5, 1.0]))
        assert torch.equal(m.D, torch.tensor([0.0, 0.5, 1.0]))


# ------------------------------------------------------------- 3. unit_dim
class TestUnitDim:
    @pytest.mark.parametrize("bad", [0, -1, -3])
    def test_constructor_rejects_batch_or_negative_unit_dim(self, bad):
        with pytest.raises(ValueError, match="unit_dim"):
            DARL(4, unit_dim=bad)

    @pytest.mark.parametrize("bad", [1.0, "1", True])
    def test_constructor_rejects_non_int_unit_dim(self, bad):
        with pytest.raises(ValueError, match="unit_dim"):
            DARL(4, unit_dim=bad)

    def test_unit_dim_beyond_input_rank_rejected(self):
        m = DARL(4, unit_dim=2)
        with pytest.raises(ValueError, match="out of range"):
            m(torch.randn(8, 4))  # a 2-D input has no dim 2

    def test_inactive_fraction_rejects_batch_dimension(self):
        with pytest.raises(ValueError, match="unit_dim"):
            inactive_fraction(torch.randn(4, 3), unit_dim=0)

    def test_unit_dim_two_is_valid_for_sequence_layout(self):
        # (B, T, F) with units on the last dim: statistics pool over B and T.
        m = DARL(2, unit_dim=2, rho=0.0)
        z = torch.tensor([[[-1.0, 1.0], [-1.0, -1.0]]])  # unit0: 2/2, unit1: 1/2
        train_step(m, z)
        assert torch.allclose(m.D, torch.tensor([1.0, 0.5]))


# ---------------------------------------------------- 4. finite-value checks
class TestFiniteCheckDefault:
    def test_finite_check_is_on_by_default(self):
        assert DARL(3).check_finite is True

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_default_rejects_nonfinite_input_and_keeps_state(self, bad):
        m = DARL(3, rho=0.5)
        with pytest.raises(FloatingPointError):
            train_step(m, torch.tensor([[bad, 1.0, -1.0]]))
        assert torch.equal(m.D, torch.zeros(3))
        assert int(m.num_updates) == 0

    def test_check_can_be_disabled_for_performance_runs(self):
        m = DARL(3, check_finite=False).eval()
        out = m(torch.tensor([[float("nan"), 1.0, -1.0]]))
        assert math.isnan(out[0, 0].item())

    def test_disabling_the_check_does_not_change_values(self):
        torch.manual_seed(0)
        z = torch.randn(16, 3)
        a, b = DARL(3, check_finite=True), DARL(3, check_finite=False)
        assert torch.equal(train_step(a, z), train_step(b, z))
        assert torch.equal(a.D, b.D)
