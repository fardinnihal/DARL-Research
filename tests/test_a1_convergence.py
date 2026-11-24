"""Stage 2 (A1), Step 7b: convergence diagnostics (training MSE only; decide nothing)."""

import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

from darl_study.a1.convergence import _pass_rate_if_flat, convergence_diagnostics, run_diagnostics
from darl_study.a1.outcomes import OutcomeThresholds, load_thresholds, write_thresholds
from darl_study.a1.pilot import budget_check
from darl_study.a1.runner import Arm, ExperimentConfig, run_experiment

REPO = Path(__file__).resolve().parents[1]
DRAFT = OutcomeThresholds()


def hist(f, top=2000, step=10, out_std=0.5):
    return [{"epoch": e, "train_mse": f(e), "val_mse": f(e), "train_loss_epoch": f(e),
             "shape": {"out_std": out_std, "target_std": 0.5}} for e in range(0, top + 1, step)]


def noisy(level, rel_sd, seed=0):
    rng = random.Random(seed)
    return lambda e: level * 10 ** rng.gauss(0.0, math.log10(1 + rel_sd))


def run(seed, h, status="completed"):
    return {"arm": "A", "seed": seed, "status": status, "history": h, "epochs": h[-1]["epoch"]}


class TestRunDiagnostics:
    def test_falling_loss_is_still_improving(self):
        d = run_diagnostics(hist(lambda e: 1e-2 * 10 ** (-0.5 * e / 1000)), 2000)
        assert d["status"] == "still_improving"
        assert d["slope_dec_per_1000"] == pytest.approx(-0.5, rel=1e-6)
        assert d["best_gain_pct"] > 1 and not d["approved_pass"]
        assert d["projected_pct_next_window"] == pytest.approx(100 * (10 ** -0.1 - 1), rel=1e-6)

    def test_flat_noisy_tiny_loss_is_plateaued_and_can_fail_the_rule(self):
        d = run_diagnostics(hist(noisy(1e-7, 0.3)), 2000)
        assert d["status"] == "plateaued"
        assert 20 < d["noise_pct"] < 40 and d["pass_rate_if_flat"] < 0.1

    def test_flat_quiet_loss_is_plateaued_and_passes(self):
        d = run_diagnostics(hist(noisy(1e-3, 0.001)), 2000)
        assert d["status"] == "plateaued" and d["approved_pass"] and d["pass_rate_if_flat"] > 0.99

    def test_rising_loss(self):
        assert run_diagnostics(hist(lambda e: 1e-4 * 10 ** (e / 1000)), 2000)["status"] == "rising"

    def test_approved_change_matches_the_budget_rule(self):
        h = hist(noisy(1e-5, 0.05, seed=3))
        b = budget_check([run(0, h)], DRAFT, budgets=[2000], loss_floor=None)  # the 1% clause (rule v1)
        d = run_diagnostics(h, 2000)
        assert d["approved_pass"] == b["budgets"][0]["passes"]
        assert d["approved_change"] == pytest.approx(b["budgets"][0]["max_change"])

    def test_needs_records_and_positive_loss(self):
        with pytest.raises(ValueError, match="too few"):
            run_diagnostics(hist(lambda e: 1.0, top=2000, step=500), 2000)
        with pytest.raises(ValueError, match="positive"):
            run_diagnostics(hist(lambda e: 0.0), 2000)

    def test_pass_rate_if_flat_matches_simulation(self):
        rel = 0.01
        sd = math.log10(1 + rel)
        rng = random.Random(1)
        n = 20000
        hits = sum(abs(rng.gauss(0, sd) - rng.gauss(0, sd)) < math.log10(1.01) for _ in range(n))
        assert _pass_rate_if_flat(sd, 0.01) == pytest.approx(hits / n, abs=0.02)


class TestConvergenceDiagnostics:
    def _runs(self):
        return [run(0, hist(lambda e: 1e-2 * 10 ** (-0.5 * e / 1000))),
                run(1, hist(noisy(1e-7, 0.3))),
                run(2, hist(noisy(1e-3, 0.001))),
                run(3, hist(lambda e: 0.25, out_std=0.0)),                   # collapsed
                run(4, hist(lambda e: 0.1, top=500), status="nonfinite")]

    def test_counts_and_skips(self):
        d = convergence_diagnostics(self._runs(), DRAFT)
        assert d["n_non_collapsed"] == 3 and d["skipped"] == {"collapsed": 1, "nonfinite": 1}
        assert d["status_counts"] == {"still_improving": 1, "plateaued": 2}
        assert d["approved_pass"] == 1
        assert d["approved_fail_by_status"] == {"still_improving": 1, "plateaued": 1, "rising": 0}
        assert d["note"].startswith("DIAGNOSTIC ONLY")

    def test_never_reads_validation_mse(self):
        runs = self._runs()
        a = convergence_diagnostics(runs, DRAFT)
        for r in runs:
            for h in r["history"]:
                h["val_mse"] = 99.0
        assert convergence_diagnostics(runs, DRAFT) == a

    def test_earlier_budget(self):
        d = convergence_diagnostics(self._runs()[:3], DRAFT, budget=1000)
        assert {r["budget"] for r in d["rows"]} == {1000}


def test_cli_on_a_real_tiny_pilot(tmp_path, capsys):
    arms = (Arm("ReLU-He", "relu", "he"), Arm("ReLU-RAI", "relu", "rai"))
    train = {"epochs": 200, "log_every": 10, "checkpoint_every": 50}
    draft = tmp_path / "outcomes_draft.json"  # pilots run with draft thresholds (repo file is frozen)
    write_thresholds(draft, OutcomeThresholds())
    exp = ExperimentConfig(name="p", mode="pilot", seed_start=0, seed_count=1, arms=arms, train=train,
                           results_root=str(tmp_path / "results"), thresholds=str(draft))
    run_experiment(exp, report=lambda m: None)
    before = {p: p.read_bytes() for p in exp.root.rglob("*") if p.is_file()}
    cfg = tmp_path / "p.json"
    cfg.write_text(json.dumps({"name": "p", "mode": "pilot", "seeds": {"start": 0, "count": 1}, "train": train,
                               "results_root": str(tmp_path / "results"), "thresholds": str(draft),
                               "arms": [{"name": a.name, "activation": a.activation, "init": a.init} for a in arms]}))
    spec = importlib.util.spec_from_file_location("a1_convergence", REPO / "scripts" / "a1_convergence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--config", str(cfg)]) == 0
    assert "DIAGNOSTIC ONLY" in capsys.readouterr().out
    after = {p: p.read_bytes() for p in exp.root.rglob("*") if p.is_file() and "analysis" not in p.parts}
    assert after == before
    assert (exp.root / "analysis" / "convergence_diagnostics.md").exists()
    assert load_thresholds(REPO / "configs" / "a1" / "outcomes.json").frozen is True  # untouched by the CLI
