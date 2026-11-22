"""Stage 2 (A1), Step 4: training loop tests."""

import json
import math

import pytest
import torch

import darl_study.a1.train as train_mod
from darl_study.a1.data import DataConfig, build_a1_datasets
from darl_study.a1.init import derive_seeds, initialize_, make_generator
from darl_study.a1.model import DARL_A1_DEFAULTS, DeepNarrowMLP
from darl_study.a1.train import NonFiniteError, TrainConfig, evaluate, log_epochs, train

ARMS = [("relu", {}), ("leaky_relu", {"negative_slope": 0.01}), ("darl", {})]


def cfg(act="relu", kw=None, **over):
    base = dict(activation=act, act_kwargs=kw or {}, init="he", run_seed=1, epochs=6,
                log_every=2, checkpoint_every=2)
    base.update(over)
    return TrainConfig(**base)


def params_of(model):
    return [p.detach().clone() for p in model.parameters()]


# ------------------------------------------------------------------ config
class TestConfig:
    def test_defaults_are_the_approved_protocol(self):
        c = TrainConfig()
        assert (c.batch_size, c.lr, c.betas, c.eps, c.weight_decay) == (128, 1e-3, (0.9, 0.999), 1e-8, 0.0)
        assert (c.n_layers, c.width, c.epochs) == (10, 2, 2000)
        assert c.data == DataConfig()

    def test_darl_parameters_are_resolved_into_the_config(self):
        assert cfg("darl").act_kwargs == DARL_A1_DEFAULTS

    def test_dict_round_trip_and_stable_hash(self):
        c = cfg("darl", run_seed=7)
        d = json.loads(json.dumps(c.to_dict()))
        c2 = TrainConfig.from_dict(d)
        assert c2 == c and c2.hash() == c.hash()
        assert cfg("darl", run_seed=8).hash() != c.hash()
        assert cfg("relu", run_seed=7).hash() != c.hash()

    @pytest.mark.parametrize("over", [
        {"activation": "elu"}, {"init": "xavier"}, {"epochs": 0}, {"batch_size": 0},
        {"lr": 0.0}, {"lr": float("nan")}, {"run_seed": -1}, {"betas": (0.9, 1.0)},
        {"epochs": 2.0}, {"weight_decay": -1.0},
    ])
    def test_invalid_rejected(self, over):
        with pytest.raises(ValueError):
            cfg(**over)

    def test_leaky_requires_slope(self):
        with pytest.raises(ValueError):
            TrainConfig(activation="leaky_relu")

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            TrainConfig.from_dict({"epochs": 3, "epocs": 4})

    def test_log_epochs(self):
        assert log_epochs(6, 2) == [0, 1, 2, 4, 5, 6]
        marks = log_epochs(2000, 10)
        assert marks[:6] == [0, 1, 2, 5, 10, 20]
        assert 2000 in marks and 1990 in marks and len(marks) == 204


# --------------------------------------------------------------- batching
class TestBatching:
    def test_epoch_is_24_steps_with_final_batch_of_56(self):
        r = train(cfg(epochs=1))
        assert r["steps_completed"] == 24
        sizes = [min(128, 3000 - i) for i in range(0, 3000, 128)]
        assert len(sizes) == 24 and sizes[-1] == 56 and sum(sizes) == 3000

    def test_every_sample_once_per_epoch_and_reshuffled(self):
        g = make_generator(derive_seeds(1)["order"])
        p1, p2 = torch.randperm(3000, generator=g), torch.randperm(3000, generator=g)
        assert torch.equal(p1.sort().values, torch.arange(3000))
        assert not torch.equal(p1, p2)

    def test_batches_seen_match_the_order_seed(self, monkeypatch):
        seen = []
        real = train_mod._mse

        def spy(pred, target):
            if pred.shape[0] != 3000 and pred.shape[0] != 1000 and pred.shape[0] != 2001:
                seen.append(target.detach().clone())
            return real(pred, target)

        monkeypatch.setattr(train_mod, "_mse", spy)
        train(cfg(epochs=1))
        data = build_a1_datasets()
        g = make_generator(derive_seeds(1)["order"])
        perm = torch.randperm(3000, generator=g)
        assert len(seen) == 24
        assert torch.equal(seen[0], data["train"].y[perm[:128]])
        assert torch.equal(seen[-1], data["train"].y[perm[-56:]])


# ---------------------------------------------------------------- training
class TestTraining:
    @pytest.mark.parametrize("act,kw", ARMS)
    def test_runs_and_completes(self, act, kw):
        r = train(cfg(act, kw))
        assert r["status"] == "completed" and r["failure"] is None
        assert r["epochs_completed"] == 6 and r["steps_completed"] == 144
        assert [h["epoch"] for h in r["history"]] == [0, 1, 2, 4, 5, 6]
        json.dumps(r)  # fully serialisable

    def test_darl_updates_exactly_once_per_optimizer_step(self):
        r = train(cfg("darl"))
        assert r["darl_num_updates"] == [144] * 9
        # Measurement passes (eval mode) added no updates.
        assert all(h["darl_num_updates"] == h["step"] for h in r["history"])

    def test_darl_history_has_slope_statistics(self):
        r = train(cfg("darl"))
        h0, hN = r["history"][0], r["history"][-1]
        assert h0["darl_alpha_mean"] == pytest.approx(0.01) and h0["darl_alpha_time_mean"] is None
        assert 0.01 <= hN["darl_alpha_mean"] <= 0.3
        assert 0.01 <= r["darl_alpha_time_mean"] <= 0.3
        assert len(hN["darl_D_mean_layers"]) == 9

    def test_parameters_change_and_loss_is_mse(self):
        c = cfg("relu", init="rai", run_seed=3, epochs=3)
        m = DeepNarrowMLP()
        initialize_(m, "rai", 3)
        before = params_of(m)
        r = train(c)
        h0 = r["history"][0]
        data = build_a1_datasets()
        with torch.no_grad():
            assert h0["train_mse"] == pytest.approx(float(((m(data["train"].x) - data["train"].y) ** 2).mean()), rel=1e-6)
            assert h0["val_mse"] == pytest.approx(float(((m(data["val"].x) - data["val"].y) ** 2).mean()), rel=1e-6)
        assert r["history"][-1]["train_mse"] != h0["train_mse"]
        del before

    def test_training_reduces_loss_for_a_trainable_seed(self):
        r = train(cfg("darl", init="rai", run_seed=3, epochs=40, log_every=10))
        first, last = r["history"][0], r["history"][-1]
        assert last["train_mse"] < first["train_mse"]

    def test_bitwise_deterministic(self):
        a, b = train(cfg("darl")), train(cfg("darl"))
        assert a["history"] == b["history"]
        assert a["darl_alpha_time_mean"] == b["darl_alpha_time_mean"]

    def test_arms_share_initial_weights(self):
        h0 = [train(cfg(a, k, epochs=1))["history"][0] for a, k in ARMS]
        # DARL at D=0 equals LeakyReLU 0.01, so their initial losses agree exactly.
        assert h0[1]["train_mse"] == h0[2]["train_mse"]
        assert h0[1]["grad_norm_layers"] == h0[2]["grad_norm_layers"]

    def test_gradient_norms_are_recorded_per_layer(self):
        h = train(cfg(epochs=1))["history"][0]
        assert len(h["grad_norm_layers"]) == 10
        assert h["grad_norm_total"] == pytest.approx(math.sqrt(sum(g * g for g in h["grad_norm_layers"])))

    def test_dead_relu_net_has_zero_hidden_gradient(self):
        # Seed 1 / He / ReLU is born dead, first dead layer 2 (Step 3). Layers up to
        # the first dead layer get exactly zero gradient.
        h = train(cfg("relu", run_seed=1, epochs=1))["history"][0]
        assert h["probe_out_range"] == 0.0
        assert h["grad_norm_layers"][0] == 0.0 and h["grad_norm_layers"][1] == 0.0

    def test_extra_metrics_hook_is_merged(self):
        r = train(cfg(epochs=2), extra_metrics=lambda m, d: {"n_probe": len(d["probe"])})
        assert all(h["n_probe"] == 2001 for h in r["history"])

    def test_evaluate_does_not_change_state_or_grads(self):
        m = DeepNarrowMLP(activation="darl")
        initialize_(m, "he", 2)
        m.train()
        data = build_a1_datasets()
        before = params_of(m)
        evaluate(m, data, 0, 0)
        assert m.training
        assert all(p.grad is None for p in m.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, params_of(m)))
        assert all(int(l.num_updates) == 0 for l in m.darl_layers())


# --------------------------------------------------------------- fail-fast
class TestNonFinite:
    def _inject_nan_loss(self, monkeypatch, at_call):
        real = train_mod._mse
        calls = {"n": 0}

        def fake(pred, target):
            out = real(pred, target)
            if pred.shape[0] in (128, 56):  # training batches only
                calls["n"] += 1
                if calls["n"] == at_call:
                    return out * float("nan")
            return out

        monkeypatch.setattr(train_mod, "_mse", fake)

    def test_nan_loss_stops_the_run_and_is_recorded(self, monkeypatch, tmp_path):
        self._inject_nan_loss(monkeypatch, at_call=30)
        r = train(cfg("darl", epochs=6), run_dir=tmp_path)
        assert r["status"] == "nonfinite"
        assert r["failure"]["step"] == 30 and r["failure"]["epoch"] == 2
        assert r["failure"]["error"] == "NonFiniteError"
        assert r["steps_completed"] == 29  # the failing step was not applied
        saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
        assert saved["status"] == "nonfinite"

    def test_nonfinite_run_is_terminal_on_resume(self, monkeypatch, tmp_path):
        self._inject_nan_loss(monkeypatch, at_call=30)
        train(cfg("relu", epochs=6), run_dir=tmp_path)
        monkeypatch.undo()
        again = train(cfg("relu", epochs=6), run_dir=tmp_path)
        assert again["status"] == "nonfinite" and again["steps_completed"] == 29

    def test_nonfinite_weights_trip_darl_check(self, monkeypatch):
        real_init = train_mod.initialize_

        def bad_init(model, scheme, seed):
            out = real_init(model, scheme, seed)
            with torch.no_grad():
                model.linears[0].weight[0, 0] = float("inf")
            return out

        monkeypatch.setattr(train_mod, "initialize_", bad_init)
        # Step 5 review fix: the initial evaluation's NaN/Inf is now recorded
        # as a terminal nonfinite result instead of escaping as an exception.
        r = train(cfg("darl", epochs=2))
        assert r["status"] == "nonfinite"
        assert r["failure"]["phase"] == "initial_evaluation"
        assert r["failure"]["error"] == "FloatingPointError"  # DARL's own check

    def test_nonfinite_error_is_a_floating_point_error(self):
        assert issubclass(NonFiniteError, FloatingPointError)


# ------------------------------------------------------------ timing script
def test_timing_script_runs(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "a1_time.py"
    spec = importlib.util.spec_from_file_location("a1_time", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "timing.json"
    assert mod.main(["--epochs", "2", "--seeds", "1", "--activations", "relu", "darl",
                     "--inits", "he", "--workers", "2", "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert len(rec["measurements"]) == 2
    assert all(m["steps_per_epoch"] == 24 for m in rec["measurements"])
    assert all(m["projected_seconds_per_run"] > 0 for m in rec["measurements"])


# ------------------------------------- Step 5 review: evaluation-time NaN/Inf
def _inf_weight_init(monkeypatch):
    real_init = train_mod.initialize_

    def bad_init(model, scheme, seed):
        out = real_init(model, scheme, seed)
        with torch.no_grad():
            model.linears[0].weight[0, 0] = float("inf")
        return out

    monkeypatch.setattr(train_mod, "initialize_", bad_init)


class TestEvaluationNonFinite:
    @pytest.mark.parametrize("act,kw", ARMS)
    def test_initial_evaluation_failure_is_recorded(self, monkeypatch, tmp_path, act, kw):
        _inf_weight_init(monkeypatch)
        r = train(cfg(act, kw), run_dir=tmp_path)
        assert r["status"] == "nonfinite"
        assert r["failure"]["phase"] == "initial_evaluation"
        assert (r["failure"]["epoch"], r["failure"]["step"]) == (0, 0)
        assert r["steps_completed"] == 0 and r["history"] == [] and r["final"] is None
        saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
        assert saved["status"] == "nonfinite" and saved["failure"]["phase"] == "initial_evaluation"
        assert (tmp_path / "checkpoint.pt").exists()

    def test_initial_failure_is_terminal_on_resume(self, monkeypatch, tmp_path):
        _inf_weight_init(monkeypatch)
        first = train(cfg("darl"), run_dir=tmp_path)
        monkeypatch.undo()  # even with a healthy init, the saved run stays terminal
        again = train(cfg("darl"), run_dir=tmp_path)
        assert again["status"] == "nonfinite" and again["failure"] == first["failure"]
        assert again["steps_completed"] == 0

    def test_scheduled_evaluation_failure_is_recorded(self, tmp_path):
        def hook(model, data):
            if model.darl_layers() and int(model.darl_layers()[0].num_updates) >= 48:
                raise FloatingPointError("injected non-finite metric")
            return {}

        r = train(cfg("darl", epochs=6), run_dir=tmp_path, extra_metrics=hook)
        assert r["status"] == "nonfinite"
        f = r["failure"]
        assert f["phase"] == "scheduled_evaluation" and f["epoch"] == 2 and f["step"] == 48
        assert [h["epoch"] for h in r["history"]] == [0, 1]  # records before the failure kept
        assert r["epochs_completed"] == 2  # no training after the failed measurement
        saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
        assert saved["failure"]["phase"] == "scheduled_evaluation"
        again = train(cfg("darl", epochs=6), run_dir=tmp_path, extra_metrics=hook)
        assert again["status"] == "nonfinite" and again["epochs_completed"] == 2

    def test_nonfinite_error_raised_by_evaluate_itself_is_recorded(self, monkeypatch):
        real = train_mod._mse
        calls = {"n": 0}

        def nan_val(pred, target):
            out = real(pred, target)
            if pred.shape[0] == 1000:  # validation pass inside evaluate()
                calls["n"] += 1
                if calls["n"] == 3:
                    return out * float("nan")
            return out

        monkeypatch.setattr(train_mod, "_mse", nan_val)
        r = train(cfg("relu", epochs=6))
        assert r["status"] == "nonfinite"
        assert r["failure"]["error"] == "NonFiniteError"
        assert r["failure"]["phase"] == "scheduled_evaluation" and r["failure"]["epoch"] == 2

    def test_training_step_failures_name_their_phase(self, monkeypatch):
        real = train_mod._mse
        calls = {"n": 0}

        def fake(pred, target):
            out = real(pred, target)
            if pred.shape[0] in (128, 56):
                calls["n"] += 1
                if calls["n"] == 5:
                    return out * float("nan")
            return out

        monkeypatch.setattr(train_mod, "_mse", fake)
        assert train(cfg("relu"))["failure"]["phase"] == "training_step"

    def test_other_evaluation_errors_still_propagate(self):
        def broken(model, data):
            raise KeyError("a bug, not a NaN")

        with pytest.raises(KeyError):
            train(cfg("relu", epochs=2), extra_metrics=broken)
