"""Stage 2 (A1), Step 5: neuron-level metrics."""

import json

import pytest
import torch

from darl_study.activations import darl_tracking
from darl_study.a1.data import build_a1_datasets
from darl_study.a1.init import initialize_
from darl_study.a1.metrics import a1_extra_metrics, function_shape, neuron_snapshot, recovery_summary
from darl_study.a1.model import DeepNarrowMLP
from darl_study.a1.train import TrainConfig, train

DATA = build_a1_datasets()
LEAKY = {"negative_slope": 0.01}


def hand_net(activation="relu", act_kwargs=None, dead_layer=0, kill_unit=None):
    """Alive network: layer 1 = (x + 2, -x + 2), identity layers, output x+2 + 2(-x+2).

    dead_layer=k makes every unit of hidden layer k inactive on all inputs.
    kill_unit=(k, j) makes only unit j of layer k inactive.
    """
    m = DeepNarrowMLP(activation=activation, act_kwargs=act_kwargs or {})
    with torch.no_grad():
        m.linears[0].weight.copy_(torch.tensor([[1.0], [-1.0]]))
        m.linears[0].bias.fill_(2.0)
        for l in m.linears[1:9]:
            l.weight.copy_(torch.eye(2))
            l.bias.zero_()
        m.linears[9].weight.copy_(torch.tensor([[1.0, 2.0]]))
        m.linears[9].bias.zero_()
        if dead_layer:
            m.linears[dead_layer - 1].bias.fill_(-100.0)
        if kill_unit:
            k, j = kill_unit
            m.linears[k - 1].bias[j] = -100.0
    return m


# ------------------------------------------------------------ inactivity
class TestInactivity:
    def test_alive_network_has_no_dead_units(self):
        s = neuron_snapshot(hand_net(), DATA)
        assert s["dead_count"] == 0 and s["dead_train_count"] == 0
        assert s["inactive_frac_mean"] == 0.0
        assert s["grad_receiving_count"] == 18 and s["zero_out_count"] == 0
        assert s["structural_dead_layer"] == 0

    def test_q_is_fraction_of_probe_points_with_z_le_0(self):
        # Layer 1 unit 0 has z = x + b; with b = 0 exactly half of the
        # probe points are < 0, plus x = 0 itself counts as inactive.
        m = hand_net()
        with torch.no_grad():
            m.linears[0].bias.zero_()
        s = neuron_snapshot(m, DATA)
        assert s["q"][0][0] == pytest.approx(1001 / 2001)
        assert s["q"][0][1] == pytest.approx(1001 / 2001)

    def test_z_equal_zero_counts_as_inactive(self):
        m = hand_net()
        with torch.no_grad():
            m.linears[0].weight.zero_()
            m.linears[0].bias.zero_()  # z = 0 everywhere
        s = neuron_snapshot(m, DATA)
        assert s["q"][0] == [1.0, 1.0] and s["dead"][0] == [1, 1]

    @pytest.mark.parametrize("act,kw", [("relu", {}), ("leaky_relu", LEAKY), ("darl", {})])
    def test_dead_layer_is_dead_for_every_activation(self, act, kw):
        s = neuron_snapshot(hand_net(act, kw, dead_layer=4), DATA)
        assert s["dead"][3] == [1, 1] and s["dead_count_layers"][3] == 2
        assert s["structural_dead_layer"] == 1

    def test_single_dead_unit_located(self):
        s = neuron_snapshot(hand_net(kill_unit=(3, 1)), DATA)
        assert s["dead"][2] == [0, 1]
        assert s["dead_count"] >= 1
        assert s["dead_count_layers"][:2] == [0, 0]

    def test_per_layer_and_mean_inactivity_agree(self):
        s = neuron_snapshot(hand_net(kill_unit=(5, 0)), DATA)
        assert s["inactive_frac_mean"] == pytest.approx(sum(s["inactive_frac_layers"]) / 9)


# ---------------------------------------------- zero output vs dead vs grad
class TestDistinctions:
    def test_relu_dead_unit_outputs_zero_and_gets_no_gradient(self):
        s = neuron_snapshot(hand_net("relu", kill_unit=(3, 1)), DATA)
        assert s["zero_out"][2][1] == 1
        assert s["grad_z"][2][1] == 0.0

    def test_leaky_dead_unit_is_dead_but_not_zero_and_gets_gradient(self):
        s = neuron_snapshot(hand_net("leaky_relu", LEAKY, kill_unit=(3, 1)), DATA)
        assert s["dead"][2][1] == 1            # preactivation test: dead
        assert s["zero_out"][2][1] == 0        # output alpha*z != 0
        assert s["grad_z"][2][1] > 0.0         # gradient still flows

    def test_darl_dead_unit_same_classification_as_leaky(self):
        a = neuron_snapshot(hand_net("darl", kill_unit=(3, 1)), DATA)
        b = neuron_snapshot(hand_net("leaky_relu", LEAKY, kill_unit=(3, 1)), DATA)
        for k in ("q", "dead", "dead_train", "zero_out"):
            assert a[k] == b[k]

    def test_units_below_dead_relu_layer_are_constant(self):
        s = neuron_snapshot(hand_net("relu", dead_layer=4), DATA)
        # Layers 5..9 get h = 0 from layer 4, so z = b is constant on P.
        assert all(r < 1e-6 for layer in s["z_range"][4:] for r in layer)
        assert s["constant_unit_count"] >= 10

    def test_darl_state_reported(self):
        m = hand_net("darl", kill_unit=(3, 1))  # one inactive unit, so D must move
        s0 = neuron_snapshot(m, DATA)
        assert s0["darl_D"] == [[0.0, 0.0]] * 9
        assert all(a == pytest.approx(0.01) for layer in s0["darl_alpha"] for a in layer)
        m.train()
        with darl_tracking(m):
            m(DATA["train"].x[:128])
        s1 = neuron_snapshot(m, DATA)
        assert s1["darl_D"][2][1] == pytest.approx(0.01)  # D = (1 - rho) * r, r = 1
        assert s1["darl_D"][0] == [0.0, 0.0]               # always-active units stay at 0
        assert all(0.01 - 1e-7 <= a <= 0.3 + 1e-7 for layer in s1["darl_alpha"] for a in layer)  # float32

    def test_relu_and_leaky_have_no_darl_fields(self):
        assert "darl_D" not in neuron_snapshot(hand_net(), DATA)


# --------------------------------------------------------------- side effects
class TestNoSideEffects:
    def test_snapshot_leaves_model_untouched(self):
        m = DeepNarrowMLP(activation="darl")
        initialize_(m, "he", 4)
        m.train()
        before = [p.detach().clone() for p in m.parameters()]
        with darl_tracking(m):  # even inside tracking, metrics must not update D
            a1_extra_metrics(m, DATA)
        assert m.training
        assert all(p.grad is None for p in m.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
        assert all(int(l.num_updates) == 0 for l in m.darl_layers())

    def test_snapshot_is_json_serialisable_and_deterministic(self):
        m = DeepNarrowMLP(activation="darl")
        initialize_(m, "rai", 2)
        a, b = a1_extra_metrics(m, DATA), a1_extra_metrics(m, DATA)
        assert json.loads(json.dumps(a)) == a == b


# -------------------------------------------------------------- function shape
class TestFunctionShape:
    def test_perfect_fit_of_abs(self):
        m = hand_net()
        with torch.no_grad():  # output = ReLU(x) + ReLU(-x) = |x| via layer 1 with b = 0
            m.linears[0].bias.zero_()
            m.linears[9].weight.copy_(torch.tensor([[1.0, 1.0]]))
        s = function_shape(m, DATA)
        assert s["probe_mse"] < 1e-12 and s["left_mse"] < 1e-12 and s["right_mse"] < 1e-12
        assert s["target_std"] == pytest.approx(0.5, abs=1e-3)
        assert s["left_target_std"] == pytest.approx(0.5, abs=1e-3)

    def test_constant_network(self):
        s = function_shape(hand_net(dead_layer=2), DATA)
        assert s["out_std"] == 0.0 and s["out_range"] == 0.0
        assert s["left_out_std"] == 0.0 and s["right_out_std"] == 0.0

    def test_half_fit(self):
        m = hand_net()
        with torch.no_grad():  # keep only ReLU(x): right half fitted, left flat at 0
            m.linears[0].bias.zero_()
            m.linears[9].weight.copy_(torch.tensor([[1.0, 0.0]]))
        s = function_shape(m, DATA)
        assert s["right_mse"] < 1e-12 and s["left_out_std"] == 0.0
        assert s["left_mse"] == pytest.approx(1.0, rel=2e-3)  # mean of x^2 on [-sqrt3, 0)


# ------------------------------------------------------------------ recovery
def q_hist(*frames):
    """frames: lists of 18 q values (9 layers x 2) -> nested history."""
    return [[list(f[2 * l:2 * l + 2]) for l in range(9)] for f in frames]


class TestRecovery:
    def test_no_dead_units_gives_none_rate(self):
        h = q_hist([0.5] * 18, [0.2] * 18)
        r = recovery_summary(h)
        assert r["dead_ever"] == 0 and r["recovery_eligible"] == 0
        assert r["recovery_rate"] is None

    def test_dead_then_alive_is_recovered(self):
        a = [0.5] * 18
        b = list(a)
        a[0] = 1.0
        r = recovery_summary(q_hist(a, b))
        assert (r["recovery_eligible"], r["recovered"], r["recovery_rate"]) == (1, 1, 1.0)
        assert r["recovered_alive_end"] == 1
        assert r["recovered_units"][0] == [1, 0]

    def test_staying_dead_is_not_recovered(self):
        a = [0.5] * 18
        a[3] = 1.0
        r = recovery_summary(q_hist(a, a, a))
        assert (r["recovery_eligible"], r["recovered"], r["recovery_rate"]) == (1, 0, 0.0)
        assert r["dead_at_last"] == 1

    def test_dead_only_at_last_checkpoint_is_not_eligible(self):
        a = [0.5] * 18
        b = list(a)
        b[5] = 1.0
        r = recovery_summary(q_hist(a, b))
        assert r["dead_ever"] == 1 and r["recovery_eligible"] == 0 and r["recovery_rate"] is None

    def test_recover_then_die_again(self):
        a, b = [0.5] * 18, [0.5] * 18
        a[0] = 1.0
        r = recovery_summary(q_hist(a, b, a))
        assert r["recovered"] == 1 and r["recovered_alive_end"] == 0

    def test_nearly_dead_is_not_dead(self):
        a = [0.5] * 18
        a[0] = 1.0 - 1 / 2001  # one probe point positive
        r = recovery_summary(q_hist(a, a))
        assert r["dead_ever"] == 0

    def test_nonconstant_recovery_needs_signal(self):
        a, b = [0.5] * 18, [0.0] * 18
        a[0] = 1.0
        zr_const = q_hist([1.0] * 18, [0.0] * 18)   # alive again but constant z
        zr_var = q_hist([1.0] * 18, [0.3] * 18)
        assert recovery_summary(q_hist(a, b), zr_const)["recovered_nonconstant"] == 0
        assert recovery_summary(q_hist(a, b), zr_var)["recovered_nonconstant"] == 1

    def test_single_checkpoint(self):
        a = [1.0] * 18
        r = recovery_summary(q_hist(a))
        assert r["dead_ever"] == 18 and r["recovery_eligible"] == 0 and r["recovery_rate"] is None

    @pytest.mark.parametrize("bad", [[], [[[1.5, 0.0]] * 9], [[[float("nan"), 0.0]] * 9]])
    def test_bad_input_rejected(self, bad):
        with pytest.raises(ValueError):
            recovery_summary(bad)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            recovery_summary(q_hist([0.5] * 18), q_hist([0.5] * 18, [0.5] * 18))


# ------------------------------------------------------------- with training
class TestWithTraining:
    @pytest.mark.parametrize("act,kw", [("relu", {}), ("leaky_relu", LEAKY), ("darl", {})])
    def test_hook_records_every_checkpoint(self, act, kw):
        r = train(TrainConfig(activation=act, act_kwargs=kw, init="rai", run_seed=2, epochs=4, log_every=2),
                  extra_metrics=a1_extra_metrics)
        assert r["status"] == "completed"
        for h in r["history"]:
            assert len(h["neurons"]["q"]) == 9 and len(h["neurons"]["q"][0]) == 2
            assert "out_std" in h["shape"]
        summ = recovery_summary([h["neurons"]["q"] for h in r["history"]],
                                [h["neurons"]["z_range"] for h in r["history"]])
        assert summ["n_checkpoints"] == len(r["history"])
        if act == "darl":
            assert r["darl_num_updates"] == [96] * 9  # metrics added no DARL updates

    def test_born_dead_relu_has_all_dead_layer_and_zero_gradients(self):
        r = train(TrainConfig(run_seed=1, epochs=1), extra_metrics=a1_extra_metrics)  # seed 1/He: born dead
        n0 = r["history"][0]["neurons"]
        assert n0["structural_dead_layer"] == 1
        assert n0["dead"][1] == [1, 1]                # first dead layer is 2 (Step 3)
        assert n0["grad_z"][0] == [0.0, 0.0]
