"""Stage 2 (A1), Step 3: architecture, activations, He/RAI init, seeds, born-dead."""

import csv
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from darl_study.activations import DARL, darl_tracking
from darl_study.a1.init import (
    RAI_SIGMA,
    RAI_SIGMA_FORMULA,
    SEED_STREAMS,
    derive_seeds,
    he_init_,
    initialize_,
    make_generator,
    rai_init_,
)
from darl_study.a1.model import DARL_A1_DEFAULTS, DeepNarrowMLP, make_activation

REPO = Path(__file__).resolve().parents[1]


def load_born_dead_script():
    spec = importlib.util.spec_from_file_location("a1_born_dead", REPO / "scripts" / "a1_born_dead.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = load_born_dead_script()


def param_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().numpy().astype("<f4").tobytes())
    return h.hexdigest()


# Recorded when init.py was written. Same seed must give the same tensors on
# every CPU; a mismatch means runs are not comparable across machines.
GOLDEN_PARAM_SHA256 = {
    "he": "90f26e98da594d0e351b14fff0be21ae984398b2f61b1c7d9e8afa476fce406d",
    "rai": "37bbedae8c336c6e0573b89cdb66d40a33851c24447fd7ec74c49ee8e24eb62f",
}
GOLDEN_DERIVED_SEED0 = {"init": 4334430513956379144, "order": 2440950710608614359}


# ------------------------------------------------------------ architecture
class TestArchitecture:
    def test_lu_layer_convention(self):
        m = DeepNarrowMLP()
        assert len(m.linears) == 10 and len(m.acts) == 9 and m.n_hidden == 9
        dims = [(l.in_features, l.out_features) for l in m.linears]
        assert dims == [(1, 2)] + [(2, 2)] * 8 + [(2, 1)]

    def test_parameter_count_is_55(self):
        m = DeepNarrowMLP()
        assert sum(p.numel() for p in m.parameters()) == 55
        assert m.describe()["n_params"] == 55

    @pytest.mark.parametrize("act", ["relu", "leaky_relu", "darl"])
    def test_forward_shapes(self, act):
        kw = {"negative_slope": 0.01} if act == "leaky_relu" else {}
        m = DeepNarrowMLP(activation=act, act_kwargs=kw)
        x = torch.randn(7, 1)
        out, pre = m.forward_with_preactivations(x)
        assert out.shape == (7, 1) and out.dtype == torch.float32
        assert len(pre) == 9 and all(z.shape == (7, 2) for z in pre)
        assert torch.equal(m(x), out)

    def test_forward_equals_manual_composition(self):
        m = DeepNarrowMLP()
        initialize_(m, "he", 3)
        x = torch.linspace(-1.7, 1.7, 11).unsqueeze(1)
        h = x
        for i in range(9):
            h = torch.relu(h @ m.linears[i].weight.T + m.linears[i].bias)
        ref = h @ m.linears[9].weight.T + m.linears[9].bias
        assert torch.allclose(m(x), ref, atol=1e-6)

    def test_preactivations_are_linear_outputs(self):
        m = DeepNarrowMLP(activation="leaky_relu", act_kwargs={"negative_slope": 0.1})
        x = torch.randn(5, 1)
        _, pre = m.forward_with_preactivations(x)
        assert torch.allclose(pre[0], m.linears[0](x))
        h1 = torch.where(pre[0] >= 0, pre[0], 0.1 * pre[0])
        assert torch.allclose(pre[1], m.linears[1](h1))

    def test_each_hidden_layer_has_its_own_darl_state(self):
        m = DeepNarrowMLP(activation="darl")
        layers = m.darl_layers()
        assert len(layers) == 9
        assert len({id(l) for l in layers}) == 9
        assert len({l.D.data_ptr() for l in layers}) == 9

    @pytest.mark.parametrize("kw", [{"n_layers": 1}, {"width": 0}, {"d_in": True}, {"n_layers": 3.0}])
    def test_bad_sizes_rejected(self, kw):
        with pytest.raises(ValueError):
            DeepNarrowMLP(**kw)

    def test_bad_input_shape_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            DeepNarrowMLP()(torch.randn(4))

    def test_other_sizes_work(self):
        m = DeepNarrowMLP(n_layers=20, width=4, d_in=2, d_out=2)
        assert m(torch.randn(3, 2)).shape == (3, 2)


# ------------------------------------------------------------- activations
class TestActivations:
    def test_relu(self):
        assert isinstance(make_activation("relu", 2), nn.ReLU)

    def test_leaky_relu_needs_explicit_slope(self):
        a = make_activation("leaky_relu", 2, negative_slope=0.01)
        assert isinstance(a, nn.LeakyReLU) and a.negative_slope == 0.01
        with pytest.raises(ValueError):
            make_activation("leaky_relu", 2)
        with pytest.raises(ValueError):
            make_activation("leaky_relu", 2, negative_slope=0.0)

    def test_darl_uses_approved_defaults(self):
        a = make_activation("darl", 2)
        assert isinstance(a, DARL)
        assert (a.alpha_min, a.alpha_max, a.p, a.rho) == (0.01, 0.3, 1.0, 0.99)
        assert torch.equal(a.D, torch.zeros(2)) and a.check_finite is True
        assert DARL_A1_DEFAULTS["d_init"] == 0.0

    def test_darl_describe_records_resolved_parameters(self):
        d = DeepNarrowMLP(activation="darl").describe()
        assert d["act_kwargs"] == DARL_A1_DEFAULTS

    def test_unknown_activation_or_argument_rejected(self):
        with pytest.raises(ValueError):
            make_activation("elu", 2)
        with pytest.raises(ValueError):
            make_activation("relu", 2, negative_slope=0.1)
        with pytest.raises(ValueError):
            make_activation("darl", 2, alpha=0.1)

    def test_darl_updates_exactly_once_per_tracked_forward(self):
        m = DeepNarrowMLP(activation="darl")
        initialize_(m, "he", 0)
        x = torch.randn(128, 1)
        m.train()
        m(x)  # untracked: no update
        assert all(int(l.num_updates) == 0 for l in m.darl_layers())
        with darl_tracking(m):
            m.forward_with_preactivations(x)
        assert all(int(l.num_updates) == 1 for l in m.darl_layers())
        m.eval()
        with darl_tracking(m):
            m(x)  # eval: frozen
        assert all(int(l.num_updates) == 1 for l in m.darl_layers())

    def test_darl_at_init_equals_leaky_001_in_eval(self):
        # D(0) = 0 gives alpha = alpha_min = 0.01 before any update.
        d = DeepNarrowMLP(activation="darl")
        l = DeepNarrowMLP(activation="leaky_relu", act_kwargs={"negative_slope": 0.01})
        initialize_(d, "he", 5)
        initialize_(l, "he", 5)
        x = torch.linspace(-1.7, 1.7, 101).unsqueeze(1)
        assert torch.equal(d.eval()(x), l.eval()(x))


# ------------------------------------------------------------------ seeds
class TestSeeds:
    def test_derived_seeds_are_deterministic_and_golden(self):
        assert derive_seeds(0) == derive_seeds(0) == GOLDEN_DERIVED_SEED0
        assert set(derive_seeds(7)) == set(SEED_STREAMS)

    def test_streams_are_distinct(self):
        seen = set()
        for s in range(200):
            d = derive_seeds(s)
            assert d["init"] != d["order"]
            seen.update(d.values())
        assert len(seen) == 400
        assert all(0 <= v < 2**63 for v in seen)

    @pytest.mark.parametrize("bad", [-1, 1.0, "0", True])
    def test_bad_run_seed_rejected(self, bad):
        with pytest.raises(ValueError):
            derive_seeds(bad)

    @pytest.mark.parametrize("scheme", ["he", "rai"])
    def test_golden_param_hash(self, scheme):
        m = DeepNarrowMLP()
        initialize_(m, scheme, 0)
        assert param_hash(m) == GOLDEN_PARAM_SHA256[scheme]

    @pytest.mark.parametrize("scheme", ["he", "rai"])
    def test_same_seed_same_tensors_different_seed_different(self, scheme):
        a, b, c = DeepNarrowMLP(), DeepNarrowMLP(), DeepNarrowMLP()
        initialize_(a, scheme, 11)
        initialize_(b, scheme, 11)
        initialize_(c, scheme, 12)
        assert param_hash(a) == param_hash(b) != param_hash(c)

    @pytest.mark.parametrize("scheme", ["he", "rai"])
    def test_identical_tensors_across_activations(self, scheme):
        # Fairness requirement: same seed -> same weights for every arm.
        hashes = set()
        for act, kw in (("relu", {}), ("leaky_relu", {"negative_slope": 0.01}), ("darl", {})):
            m = DeepNarrowMLP(activation=act, act_kwargs=kw)
            initialize_(m, scheme, 42)
            hashes.add(param_hash(m))
        assert len(hashes) == 1

    def test_init_independent_of_global_rng_and_leaves_it_untouched(self):
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        a = DeepNarrowMLP()  # nn.Linear's default init uses the global RNG; ours must not
        t0, p0 = torch.get_rng_state(), random.getstate()
        n0 = np.random.get_state()[1].copy()
        initialize_(a, "rai", 3)
        assert torch.equal(torch.get_rng_state(), t0)
        assert random.getstate() == p0
        assert np.array_equal(np.random.get_state()[1], n0)
        torch.manual_seed(999)
        b = DeepNarrowMLP()
        initialize_(b, "rai", 3)
        assert param_hash(a) == param_hash(b)

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError):
            initialize_(DeepNarrowMLP(), "xavier", 0)


# --------------------------------------------------------------------- He
def stacked(scheme, n_seeds, layer_idx):
    ws, bs = [], []
    for s in range(n_seeds):
        m = DeepNarrowMLP()
        initialize_(m, scheme, s)
        ws.append(m.linears[layer_idx].weight.detach().double().flatten())
        bs.append(m.linears[layer_idx].bias.detach().double().flatten())
    return torch.cat(ws), torch.cat(bs)


class TestHe:
    def test_biases_zero_everywhere(self):
        m = DeepNarrowMLP()
        initialize_(m, "he", 0)
        assert all(torch.equal(l.bias, torch.zeros_like(l.bias)) for l in m.linears)

    @pytest.mark.parametrize("layer,fan_in", [(0, 1), (4, 2), (9, 2)])
    def test_weight_variance_is_2_over_fan_in(self, layer, fan_in):
        w, _ = stacked("he", 1500, layer)
        n = w.numel()
        var = 2.0 / fan_in
        assert abs(w.mean().item()) < 4 * math.sqrt(var / n)
        # sample variance SE for a normal = var * sqrt(2 / n)
        assert abs(w.var().item() - var) < 4 * var * math.sqrt(2.0 / n)

    def test_he_init_matches_direct_generator_draws(self):
        m = DeepNarrowMLP(n_layers=2, width=2)
        he_init_(m, make_generator(5))
        g = make_generator(5)
        w0 = torch.randn((2, 1), generator=g, dtype=torch.float64) * math.sqrt(2.0)
        w1 = torch.randn((1, 2), generator=g, dtype=torch.float64) * 1.0
        assert torch.equal(m.linears[0].weight, w0.float())
        assert torch.equal(m.linears[1].weight, w1.float())


# -------------------------------------------------------------------- RAI
class TestRAI:
    def test_sigma_constant_matches_closed_form(self):
        assert RAI_SIGMA == 0.6007
        assert abs(RAI_SIGMA_FORMULA - 0.6007) < 5e-5

    def test_first_layer_is_he_with_zero_bias(self):
        m = DeepNarrowMLP()
        k = rai_init_(m, make_generator(0))
        assert k[0] is None
        assert torch.equal(m.linears[0].bias, torch.zeros(2))
        w, b = stacked("rai", 2000, 0)
        assert torch.equal(b, torch.zeros_like(b))
        assert abs(w.var().item() - 2.0) < 4 * 2.0 * math.sqrt(2.0 / w.numel())

    def test_one_beta_entry_per_row_in_every_later_layer(self):
        m = DeepNarrowMLP()
        ks = rai_init_(m, make_generator(1))
        assert len(ks) == 10
        for layer, k in zip(m.linears[1:], ks[1:]):
            v = torch.cat([layer.weight, layer.bias.unsqueeze(1)], dim=1)
            assert k.shape == (layer.out_features,)
            assert ((k >= 0) & (k <= layer.in_features)).all()
            picked = v[torch.arange(layer.out_features), k]
            assert ((picked >= 0) & (picked <= 1)).all()

    def test_output_layer_is_rai_too(self):
        m = DeepNarrowMLP()
        ks = rai_init_(m, make_generator(2))
        assert ks[-1] is not None and ks[-1].shape == (1,)

    def test_rai_matches_appendix_g_draw_by_draw(self):
        m = DeepNarrowMLP(n_layers=3, width=2)
        ks = rai_init_(m, make_generator(9))
        g = make_generator(9)
        torch.randn((2, 1), generator=g, dtype=torch.float64)  # layer 1, He
        for layer, k_got in zip(m.linears[1:], ks[1:]):
            fo, fi = layer.out_features, layer.in_features
            v = torch.randn((fo, fi + 1), generator=g, dtype=torch.float64) * (0.6007 / math.sqrt(fi))
            k = torch.randint(0, fi + 1, (fo,), generator=g)
            u = torch.rand((fo,), generator=g, dtype=torch.float64)
            v[torch.arange(fo), k] = u.sqrt()
            assert torch.equal(k, k_got)
            assert torch.equal(layer.weight, v[:, :fi].float())
            assert torch.equal(layer.bias, v[:, fi].float())

    def test_distributions_over_many_seeds(self):
        normals, betas, ks = [], [], []
        for s in range(1500):
            m = DeepNarrowMLP()
            k_all = rai_init_(m, make_generator(s))
            for layer, k in zip(m.linears[1:9], k_all[1:9]):  # hidden 2x2 layers
                v = torch.cat([layer.weight, layer.bias.unsqueeze(1)], 1).double()
                mask = torch.zeros_like(v, dtype=torch.bool)
                mask[torch.arange(2), k] = True
                betas.append(v[mask])
                normals.append(v[~mask])
                ks.append(k)
        nrm, bet, ks = torch.cat(normals), torch.cat(betas), torch.cat(ks)
        sig2 = (0.6007 / math.sqrt(2)) ** 2
        assert abs(nrm.mean().item()) < 4 * math.sqrt(sig2 / nrm.numel())
        assert abs(nrm.var().item() - sig2) < 4 * sig2 * math.sqrt(2 / nrm.numel())
        # Beta(2,1): mean 2/3, var 1/18, support [0, 1]
        assert abs(bet.mean().item() - 2 / 3) < 4 * math.sqrt((1 / 18) / bet.numel())
        assert bet.min() >= 0 and bet.max() <= 1
        # k uniform over {0, 1, 2}
        counts = torch.bincount(ks, minlength=3).double()
        expected = ks.numel() / 3
        chi2 = (((counts - expected) ** 2) / expected).sum().item()
        assert chi2 < 13.8  # chi-square, 2 dof, p = 0.001

    def test_rai_requires_bias(self):
        with pytest.raises(ValueError, match="bias"):
            rai_init_(nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 1, bias=False)), make_generator(0))


# --------------------------------------------------------------- born dead
def hand_net(dead_layer):
    """He-initialised net with one hidden layer forced dead (or none)."""
    m = DeepNarrowMLP()
    initialize_(m, "he", 0)
    with torch.no_grad():
        for l in m.linears:  # make every layer clearly alive: identity-ish, positive bias
            l.weight.zero_()
        m.linears[0].weight.copy_(torch.tensor([[1.0], [-1.0]]))
        m.linears[0].bias.fill_(2.0)
        for l in m.linears[1:9]:
            l.weight.copy_(torch.eye(2))
            l.bias.zero_()
        m.linears[9].weight.copy_(torch.tensor([[1.0, 2.0]]))  # not symmetric, so output varies
        if dead_layer:
            m.linears[dead_layer - 1].bias.fill_(-100.0)
    return m


class TestBornDead:
    x = torch.linspace(-1.7, 1.7, 301).unsqueeze(1)

    def test_alive_network_is_not_born_dead(self):
        r = bd.born_dead_record(hand_net(0), self.x)
        assert r["dead_layer"] == 0 and r["first_dead"] == 0 and r["const_output"] == 0
        assert r["dead_units"] == 0

    @pytest.mark.parametrize("layer", [1, 4, 9])
    def test_dead_layer_detected_and_output_constant(self, layer):
        r = bd.born_dead_record(hand_net(layer), self.x)
        assert r["dead_layer"] == 1 and r["first_dead"] == layer
        assert r["out_range"] == 0.0 and r["const_output"] == 1

    def test_leaky_dead_layer_is_structural_but_output_not_exactly_constant(self):
        relu = hand_net(4)
        leaky = DeepNarrowMLP(activation="leaky_relu", act_kwargs={"negative_slope": 0.1})
        leaky.load_state_dict(relu.state_dict())
        r = bd.born_dead_record(leaky, self.x)
        assert r["dead_layer"] == 1 and r["out_range"] > 0.0

    def test_check_runs_in_eval_and_does_not_update_darl(self):
        m = DeepNarrowMLP(activation="darl")
        initialize_(m, "he", 0)
        m.train()
        with darl_tracking(m):
            bd.born_dead_record(m, self.x)
        assert m.training
        assert all(int(l.num_updates) == 0 for l in m.darl_layers())

    def test_clopper_pearson_known_values(self):
        lo, hi = bd.clopper_pearson(936, 1000)
        assert round(100 * lo, 1) == 91.9 and round(100 * hi, 1) == 95.0
        assert bd.clopper_pearson(0, 10)[0] == 0.0
        assert bd.clopper_pearson(10, 10)[1] == 1.0
        lo, hi = bd.clopper_pearson(5, 10)
        assert abs(lo - 0.187086) < 1e-5 and abs(hi - 0.812914) < 1e-5

    def test_script_end_to_end(self, tmp_path):
        rc = bd.main(["--seeds", "6", "--start", "3", "--activations", "relu", "darl", "--out", str(tmp_path)])
        assert rc == 0
        rows = list(csv.DictReader(open(tmp_path / "per_seed.csv", encoding="utf-8")))
        assert len(rows) == 6 * 2 * 2
        assert {r["seed"] for r in rows} == {str(s) for s in range(3, 9)}
        summ = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summ["seeds"] == [3, 8] and summ["network"]["n_params"] == 55
        assert len(summ["summary"]) == 4
        # Rerunning gives identical per-seed results.
        first = (tmp_path / "per_seed.csv").read_text(encoding="utf-8")
        bd.main(["--seeds", "6", "--start", "3", "--activations", "relu", "darl", "--out", str(tmp_path)])
        assert (tmp_path / "per_seed.csv").read_text(encoding="utf-8") == first
