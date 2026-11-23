"""Stage 2 (A1), Step 7: ReLU pilot config, budget check, threshold calibration, summary."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from darl_study.a1.outcomes import OutcomeThresholds, load_thresholds, write_thresholds
from darl_study.a1.pilot import (
    analyze_pilot,
    budget_check,
    calibrate_thresholds,
    candidate_budgets,
    checkpoint_table,
    clopper_pearson,
    freeze_thresholds,
    gap_report,
    load_pilot_runs,
    outcome_summary,
    plateau_change,
    seed_table,
)
from darl_study.a1.runner import Arm, ExperimentConfig, load_experiment, run_dir_for, run_experiment

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "configs" / "a1" / "pilot.json"
DRAFT = OutcomeThresholds()


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------- synthetic runs
def rec(epoch, train_mse, out_std=0.5, val_mse=None, left=None, right=None, lstd=0.5, rstd=0.5):
    val = train_mse if val_mse is None else val_mse
    return {"epoch": epoch, "train_mse": train_mse, "train_loss_epoch": train_mse, "val_mse": val,
            "neurons": {"dead_count": 0},
            "shape": {"out_std": out_std, "target_std": 0.5, "probe_mse": train_mse,
                      "left_mse": train_mse if left is None else left,
                      "right_mse": train_mse if right is None else right,
                      "left_out_std": lstd, "right_out_std": rstd,
                      "left_target_std": 0.5, "right_target_std": 0.5}}


def run(arm, seed, curve, status="completed", born_dead=0, out_range=0.3, **final_kw):
    """curve: {epoch: train_mse}; the final record takes final_kw (shape overrides)."""
    eps = sorted(curve)
    hist = [rec(e, curve[e], **(final_kw if e == eps[-1] else {})) for e in eps]
    return {"arm": arm, "activation": "relu", "init": "he", "seed": seed, "status": status,
            "history": hist, "failure": None, "epochs": eps[-1], "config_hash": "x",
            "born": {"born_dead": born_dead, "first_dead_layer": 0, "out_range": out_range,
                     "structural_dead_layer": born_dead}}


def flat_curve(level, top=2000):
    return {e: level for e in range(0, top + 1, 100)}


def collapsed_run(arm, seed, top=2000):
    c = flat_curve(0.2535, top)
    r = run(arm, seed, c, born_dead=1, out_range=0.0)
    for h in r["history"]:
        h["shape"].update(out_std=0.0, left_out_std=0.0, right_out_std=0.0)
    return r


# ------------------------------------------------------------------ config
class TestPilotConfig:
    def test_pilot_file(self):
        exp = load_experiment(PILOT)
        assert exp.mode == "pilot" and exp.purpose == "pilot" and exp.name == "relu-pilot"
        assert [(a.name, a.activation, a.init) for a in exp.arms] == [
            ("ReLU-He", "relu", "he"), ("ReLU-RAI", "relu", "rai")]
        assert exp.seeds == list(range(20))
        t = exp.train
        assert (t["epochs"], t["batch_size"], t["lr"], t["n_layers"], t["width"]) == (2000, 128, 0.001, 10, 2)
        assert t["log_every"] == 10 and 2000 % t["checkpoint_every"] == 0

    def test_pilot_results_live_apart_from_main(self):
        exp = load_experiment(PILOT)
        assert exp.root.parts[-2:] == ("pilot", "relu-pilot")
        assert "main" not in exp.root.parts

    def test_every_candidate_budget_is_recorded(self):
        from darl_study.a1.train import log_epochs
        marks = set(log_epochs(2000, 10))
        assert all(b in marks and b - b // 10 in marks for b in range(100, 2001, 100))

    def test_pilot_runs_with_draft_thresholds(self):
        exp = load_experiment(PILOT)
        load_thresholds(exp.path(exp.thresholds), purpose=exp.purpose)  # must not raise


# ------------------------------------------------------------------ stats
class TestClopperPearson:
    @pytest.mark.parametrize("k,n", [(0, 20), (1, 20), (10, 20), (19, 20), (20, 20), (3, 40)])
    def test_matches_born_dead_script(self, k, n):
        ref = _script("a1_born_dead").clopper_pearson(k, n)
        assert clopper_pearson(k, n) == pytest.approx(ref, abs=1e-9)

    def test_known_value(self):
        lo, hi = clopper_pearson(0, 20)
        assert lo == 0.0 and hi == pytest.approx(0.1684, abs=1e-4)


# ----------------------------------------------------------------- budget
class TestBudget:
    def test_plateau_change_definition(self):
        h = [rec(1800, 0.100), rec(2000, 0.0995)]
        assert plateau_change(h, 2000) == pytest.approx(0.005)
        assert plateau_change(h, 1000) is None  # epochs not recorded
        assert plateau_change([rec(90, 0.0), rec(100, 0.0)], 100) == 0.0
        assert plateau_change([rec(90, 0.0), rec(100, 1.0)], 100) == math.inf

    def test_candidates_need_both_epochs(self):
        r = run("A", 0, {0: 1.0, 90: 1.0, 100: 1.0, 180: 1.0, 200: 1.0, 300: 1.0})
        assert candidate_budgets([r]) == [100, 200]

    def test_collapsed_runs_are_excluded_and_counted(self):
        runs = [run("A", s, flat_curve(1e-2)) for s in range(19)] + [collapsed_run("A", 19)]
        b = budget_check(runs, DRAFT, budgets=[1000, 2000])
        top = b["budgets"][-1]
        assert (top["eligible"], top["converged"], top["excluded_collapsed"]) == (19, 19, 1)
        assert b["largest_candidate_sufficient"] and b["recommended_budget"] == 1000

    def test_95_percent_rule(self):
        def moving(s):  # still dropping 5% per 100 epochs
            return run("A", s, {e: 0.1 * 0.95 ** (e / 100) for e in range(0, 2001, 100)})
        ok = [run("A", s, flat_curve(1e-2)) for s in range(19)]
        assert budget_check(ok + [moving(19)], DRAFT, budgets=[2000])["budgets"][0]["passes"]  # 19/20 = 95%
        two_bad = ok[:18] + [moving(18), moving(19)]
        b = budget_check(two_bad, DRAFT, budgets=[2000])
        assert not b["budgets"][0]["passes"] and b["recommended_budget"] is None
        assert not b["largest_candidate_sufficient"]
        assert {w["seed"] for w in b["budgets"][0]["not_converged"]} == {18, 19}

    def test_recommended_budget_ignores_an_isolated_early_pass(self):
        # Flat until 300, then drops at 500-1000, then flat: B=300 passes, 500-1000 fail.
        def curve():
            c = {}
            for e in range(0, 2001, 10):
                c[e] = 0.1 if e <= 400 else (0.1 * 0.5 ** ((e - 400) / 300) if e <= 1100 else 0.1 * 0.5 ** (700 / 300))
            return c
        runs = [run("A", s, curve()) for s in range(20)]
        b = budget_check(runs, DRAFT, budgets=[300, 500, 1000, 1500, 2000])
        passes = {r["budget"]: r["passes"] for r in b["budgets"]}
        assert passes == {300: True, 500: False, 1000: False, 1500: True, 2000: True}
        assert b["recommended_budget"] == 1500

    def test_nonfinite_runs_are_counted_not_used(self):
        runs = [run("A", s, flat_curve(1e-2)) for s in range(5)]
        bad = run("A", 9, {0: 1.0, 10: 1.0}, status="nonfinite")
        b = budget_check(runs + [bad], DRAFT, budgets=[2000])
        assert b["budgets"][0]["excluded_nonfinite"] == 1 and b["budgets"][0]["eligible"] == 5

    def test_tiny_noisy_losses_are_judged_by_the_approved_rule_only(self):
        def noisy(s):  # tiny loss, large relative noise
            c = flat_curve(1e-7)
            c[1800], c[2000] = 1e-7, 3e-7
            return run("A", s, c)
        b = budget_check([noisy(s) for s in range(20)], DRAFT, budgets=[2000])
        row = b["budgets"][0]
        assert row["fraction"] == 0.0 and not row["passes"] and b["recommended_budget"] is None
        assert not any("diagnostic" in k for k in row) and "diagnostic" not in b["criterion"]

    def test_budget_never_reads_validation_mse_or_success_threshold(self):
        runs = [run("A", s, {e: 0.1 * 0.95 ** (e / 100) for e in range(0, 2001, 100)}) for s in range(20)]
        a = budget_check(runs, DRAFT, budgets=[1000, 2000])
        for r in runs:
            for h in r["history"]:
                h["val_mse"] = 1e-9
        b = budget_check(runs, OutcomeThresholds(success_val_mse=0.5), budgets=[1000, 2000])
        assert a == b and a["criterion"]["loss"].startswith("train_mse")


# ------------------------------------------------------------ calibration
class TestGapReport:
    def test_keeps_a_well_separated_draft(self):
        r = gap_report([1e-6] * 3 + [0.0] * 2 + [0.5] * 5, 0.01)
        assert r["decision"] == "kept_draft" and r["proposed"] == 0.01 and not r["manual_review"]
        assert (r["n"], r["n_zero"], r["n_below"], r["n_above"]) == (10, 2, 5, 5)

    def test_moves_a_crowded_draft_to_the_widest_gap(self):
        vals = [1e-8, 2e-8, 3e-8, 0.012, 0.3, 0.5]
        r = gap_report(vals, 0.01)
        assert r["decision"] == "moved_to_gap_midpoint"
        assert r["proposed"] == pytest.approx(math.sqrt(3e-8 * 0.012))

    def test_zeros_are_not_a_gap(self):
        r = gap_report([0.0] * 5 + [1e-3, 0.4, 0.5], 1e-3 * 1.2)
        assert r["widest_gap"]["low"] == 1e-3  # 0 is never the low end of a gap

    def test_too_few_values_flags_review(self):
        r = gap_report([0.5, 0.6], 0.01)
        assert r["manual_review"] and r["proposed"] == 0.01

    def test_no_gap_flags_review(self):
        vals = [10 ** (-3 + 0.2 * i) for i in range(10)]
        r = gap_report(vals, 0.01)
        assert r["manual_review"] and r["decision"] == "kept_draft"


class TestCalibrate:
    def _runs(self):
        runs = [collapsed_run("ReLU-He", s) for s in range(8)]
        runs += [run("ReLU-RAI", s, flat_curve(1e-6)) for s in range(10)]
        runs += [run("ReLU-RAI", 10 + s, flat_curve(0.12), left=1e-6, right=0.25, rstd=1e-4)
                 for s in range(6)]
        return runs

    def test_validation_mse_is_never_used(self):
        runs = self._runs()
        a = calibrate_thresholds(runs, DRAFT)
        for r in runs:  # wildly different validation numbers, same calibration
            for h in r["history"]:
                h["val_mse"] = 123.0
        b = calibrate_thresholds(runs, DRAFT)
        assert a["proposed"] == b["proposed"]

    def test_proposal_is_draft_and_classifies_cleanly(self):
        c = calibrate_thresholds(self._runs(), DRAFT, "relu-pilot")
        p = OutcomeThresholds(**c["proposed"])
        assert p.frozen is False and "relu-pilot" in p.frozen_note
        s = outcome_summary(self._runs(), p)
        assert s["arms"]["ReLU-He"]["outcomes"]["collapsed"]["count"] == 8
        rai = s["arms"]["ReLU-RAI"]["outcomes"]
        assert (rai["success"]["count"], rai["half_trained"]["count"]) == (10, 6)
        assert c["manual_review"] == [] and c["changed"] == []

    def test_reports_name_their_quantity(self):
        c = calibrate_thresholds(self._runs(), DRAFT)
        assert set(c["reports"]) == {"collapse_std_frac", "success_val_mse", "half_fit_mse",
                                     "half_flat_std_frac", "const_range_tol"}
        assert "not calibrated" in c["reports"]["success_val_mse"]["quantity"]


class TestSuccessThresholdIsPredetermined:
    """Regression: success_val_mse is a validation threshold and is never calibrated."""

    def _crowded(self):
        # Training MSE crowds 1e-3 from both sides: the old code moved the threshold.
        runs = [run("A", s, flat_curve(v)) for s, v in enumerate([8e-4, 9e-4, 1.1e-3, 1.2e-3, 9.5e-4, 1.05e-3])]
        return runs + [run("A", 10 + s, flat_curve(1e-6)) for s in range(6)]

    def test_kept_at_draft_value_even_when_training_mse_crowds_it(self):
        c = calibrate_thresholds(self._crowded(), DRAFT)
        r = c["reports"]["success_val_mse"]
        assert r["decision"] == "fixed_predetermined" and r["proposed"] == DRAFT.success_val_mse == 1e-3
        assert c["proposed"]["success_val_mse"] == 1e-3 and "success_val_mse" not in c["changed"]
        assert r["n"] == 0 and r["decade_histogram"] == {}  # no pilot distribution used

    def test_follows_whatever_value_was_predetermined(self):
        th = OutcomeThresholds(success_val_mse=5e-4)
        assert calibrate_thresholds(self._crowded(), th)["proposed"]["success_val_mse"] == 5e-4

    def test_independent_of_training_and_validation_mse(self):
        runs = self._crowded()
        a = calibrate_thresholds(runs, DRAFT)
        for r in runs:
            for h in r["history"]:
                h["train_mse"] *= 100.0
                h["val_mse"] = 7.0
        b = calibrate_thresholds(runs, DRAFT)
        assert a["proposed"]["success_val_mse"] == b["proposed"]["success_val_mse"] == 1e-3

    def test_classifier_still_uses_validation_mse(self):
        # Training MSE says success, validation MSE says not: the label follows validation.
        r = run("A", 0, flat_curve(1e-6))
        r["history"][-1]["val_mse"] = 0.2
        r["history"][-1]["shape"].update(left_mse=0.2, right_mse=0.2)
        assert seed_table([r], DRAFT)[0]["outcome"] == "other"

    def test_report_says_it_is_fixed(self, finished_copy):
        exp = finished_copy
        analyze_pilot(exp, DRAFT)
        md = (exp.root / "analysis" / "pilot_report.md").read_text(encoding="utf-8")
        assert "| success_val_mse | 0.001 | 0.001 | fixed_predetermined |" in md


# ----------------------------------------------------------------- summary
class TestSummary:
    def test_counts_cis_and_born_dead(self):
        runs = [collapsed_run("A", s) for s in range(3)] + [run("A", 3 + s, flat_curve(1e-6)) for s in range(17)]
        s = outcome_summary(runs, DRAFT)["arms"]["A"]
        assert s["n"] == 20 and s["outcomes"]["collapsed"]["count"] == 3
        assert s["outcomes"]["collapsed"]["ci95"] == pytest.approx([100 * v for v in clopper_pearson(3, 20)])
        assert s["born_dead"]["count"] == 3 and s["collapsed_not_born_dead"] == 0
        assert sum(v["count"] for v in s["outcomes"].values()) == 20
        assert s["outcomes_at_epoch"]["1000"]["collapsed"] == 3

    def test_tables(self):
        runs = [collapsed_run("A", 0), run("A", 1, flat_curve(1e-6))]
        rows = seed_table(runs, DRAFT)
        assert [r["outcome"] for r in rows] == ["collapsed", "success"]
        assert rows[1]["final_out_var"] == pytest.approx(0.25)
        cp = checkpoint_table(runs, DRAFT, epochs=(0, 1000, 2000, 2500))
        assert sorted({r["epoch"] for r in cp}) == [0, 1000, 2000]
        assert {"train_mse", "val_mse", "out_std", "out_var", "outcome"} <= set(cp[0])


# ------------------------------------------------------- real tiny pilot
TINY = {"epochs": 100, "log_every": 10, "checkpoint_every": 50}
RELU = (Arm("ReLU-He", "relu", "he"), Arm("ReLU-RAI", "relu", "rai"))


def tiny(tmp_path, **over):
    kw = dict(name="p", mode="pilot", seed_start=0, seed_count=2, arms=RELU, train=dict(TINY),
              results_root=str(tmp_path / "results"))
    kw.update(over)
    return ExperimentConfig(**kw)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """One real 2-arm x 2-seed pilot of 100 epochs, shared by the analysis tests."""
    exp = tiny(tmp_path_factory.mktemp("pilot"))
    run_experiment(exp, report=lambda m: None)
    return exp


@pytest.fixture
def finished_copy(finished):
    """The shared pilot; every file under its root is restored after the test."""
    saved = {p: p.read_bytes() for p in finished.root.rglob("*") if p.is_file()}
    yield finished
    for p in list(finished.root.rglob("*")):
        if p.is_file() and p not in saved:
            p.unlink()
    for p, b in saved.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)


def _edit_json(path, **changes):
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.update(changes)
    path.write_text(json.dumps(rec), encoding="utf-8")


def _d(exp, arm=0, seed=0):
    return run_dir_for(exp, exp.arms[arm], seed)


RUN_TAMPERS = {
    "outcome_missing": lambda e: (_d(e) / "outcome.json").unlink(),
    "outcome_corrupt": lambda e: (_d(e) / "outcome.json").write_text("{", encoding="utf-8"),
    "outcome_wrong_config_hash": lambda e: _edit_json(_d(e) / "outcome.json", config_hash="0" * 64),
    "outcome_wrong_thresholds_hash": lambda e: _edit_json(_d(e) / "outcome.json", thresholds_hash="0" * 64),
    "outcome_unknown_label": lambda e: _edit_json(_d(e) / "outcome.json", outcome="bogus"),
    "outcome_disagrees_with_DONE": lambda e: _edit_json(_d(e) / "DONE", outcome="half_trained"),
    "outcome_status_mismatch": lambda e: _edit_json(_d(e) / "outcome.json", status="nonfinite"),
    "outcome_other_seed": lambda e: _edit_json(_d(e) / "outcome.json", seed=1),
    "outcome_other_arm": lambda e: _edit_json(_d(e) / "outcome.json", arm="ReLU-RAI"),
    "outcome_bad_born_record": lambda e: _edit_json(_d(e) / "outcome.json", born_dead_record=[1]),
    "DONE_missing": lambda e: (_d(e) / "DONE").unlink(),
    "DONE_wrong_thresholds_hash": lambda e: _edit_json(_d(e) / "DONE", thresholds_hash="0" * 64),
    "result_corrupt": lambda e: (_d(e) / "result.json").write_text("{", encoding="utf-8"),
    "result_wrong_config_hash": lambda e: _edit_json(_d(e) / "result.json", config_hash="0" * 64),
}


class TestValidatedArtifacts:
    """Regression: the analyzer uses only runs whose artifacts are valid and consistent."""

    def test_intact_pilot_loads_every_run(self, finished_copy):
        runs, problems = load_pilot_runs(finished_copy, DRAFT)
        assert len(runs) == 4 and problems == []

    @pytest.mark.parametrize("tamper", sorted(RUN_TAMPERS))
    def test_bad_run_artifacts_are_refused(self, finished_copy, tamper):
        exp = finished_copy
        RUN_TAMPERS[tamper](exp)
        with pytest.raises(ValueError, match="1 pilot run"):
            load_pilot_runs(exp, DRAFT)
        with pytest.raises(ValueError, match="1 pilot run"):
            analyze_pilot(exp, DRAFT)
        runs, problems = load_pilot_runs(exp, DRAFT, allow_incomplete=True)
        assert len(runs) == 3 and len(problems) == 1 and problems[0].startswith("ReLU-He seed 0")
        assert ("ReLU-He", 0) not in {(r["arm"], r["seed"]) for r in runs}

    def test_other_thresholds_than_the_run_used_are_refused(self, finished_copy):
        with pytest.raises(ValueError, match="thresholds differ"):
            load_pilot_runs(finished_copy, OutcomeThresholds(collapse_std_frac=0.02))

    def test_missing_or_foreign_experiment_record_is_refused(self, finished_copy):
        exp = finished_copy
        _edit_json(exp.root / "experiment.json", experiment_hash="0" * 64)
        with pytest.raises(ValueError, match="different experiment"):
            load_pilot_runs(exp, DRAFT)
        (exp.root / "experiment.json").unlink()
        with pytest.raises(ValueError, match="experiment.json"):
            load_pilot_runs(exp, DRAFT)

    def test_incomplete_report_lists_the_excluded_run(self, finished_copy):
        exp = finished_copy
        (_d(exp, 1, 1) / "outcome.json").write_text("{", encoding="utf-8")
        s = analyze_pilot(exp, DRAFT, allow_incomplete=True)
        assert s["n_runs"] == 3 and len(s["incomplete"]) == 1
        md = (exp.root / "analysis" / "pilot_report.md").read_text(encoding="utf-8")
        assert "Incomplete" in md and "ReLU-RAI seed 1" in md

    def test_rerunning_the_runner_repairs_what_the_analyzer_refused(self, finished_copy, monkeypatch):
        exp = finished_copy
        (_d(exp) / "outcome.json").unlink()

        def no_training(*a, **k):
            raise AssertionError("must not retrain")

        monkeypatch.setattr("darl_study.a1.runner.train", no_training)
        run_experiment(exp, report=lambda m: None)
        assert len(load_pilot_runs(exp, DRAFT)[0]) == 4


class TestAnalyzeRealRuns:
    def test_refuses_main_mode_and_non_relu_arms(self, tmp_path):
        with pytest.raises(ValueError, match="separate"):
            load_pilot_runs(tiny(tmp_path, mode="main"), DRAFT)
        with pytest.raises(ValueError, match="ReLU arms only"):
            load_pilot_runs(tiny(tmp_path, arms=(Arm("D", "darl", "he"),)), DRAFT)

    def test_refuses_incomplete_pilot_unless_asked(self, tmp_path):
        exp = tiny(tmp_path, train={**TINY, "epochs": 20})
        run_experiment(tiny(tmp_path, seed_count=1, train={**TINY, "epochs": 20}), report=lambda m: None)
        with pytest.raises(ValueError, match="2 pilot run"):
            load_pilot_runs(exp, DRAFT)
        runs, problems = load_pilot_runs(exp, DRAFT, allow_incomplete=True)
        assert len(runs) == 2 and len(problems) == 2

    def test_full_analysis_writes_only_the_analysis_folder(self, finished):
        exp = finished
        before = {p: p.read_bytes() for p in exp.root.rglob("*") if p.is_file() and "analysis" not in p.parts}
        s = analyze_pilot(exp, DRAFT)
        after = {p: p.read_bytes() for p in exp.root.rglob("*") if p.is_file() and "analysis" not in p.parts}
        assert after == before  # run folders untouched
        out = exp.root / "analysis"
        assert {p.name for p in out.iterdir()} == {
            "per_seed.csv", "checkpoints.csv", "budget.json", "calibration.json",
            "outcomes_proposed.json", "summary.json", "pilot_report.md"}
        assert s["n_runs"] == 4 and s["note"].startswith("PILOT")
        assert [b["budget"] for b in s["budget"]["budgets"]] == [100]
        md = (out / "pilot_report.md").read_text(encoding="utf-8")
        assert "Not research results" in md and "Numerical proxy" in md
        prop = load_thresholds(out / "outcomes_proposed.json")
        assert prop.frozen is False

    def test_cli_analyze_and_freeze(self, finished, tmp_path, capsys):
        exp = finished
        cfg = tmp_path / "pilot.json"
        cfg.write_text(json.dumps({"name": "p", "mode": "pilot", "seeds": {"start": 0, "count": 2},
                                   "train": TINY, "results_root": exp.results_root,
                                   "arms": [{"name": a.name, "activation": a.activation, "init": a.init}
                                            for a in RELU]}), encoding="utf-8")
        cli = _script("a1_pilot")
        assert cli.main(["analyze", "--config", str(cfg)]) == 0
        assert "not research results" in capsys.readouterr().out
        target = tmp_path / "outcomes.json"
        write_thresholds(target, DRAFT)
        prop = exp.root / "analysis" / "outcomes_proposed.json"
        assert cli.main(["freeze", "--proposed", str(prop), "--target", str(target), "--note", "UA reviewed"]) == 0
        frozen = load_thresholds(target, require_frozen=True)
        assert "UA reviewed" in frozen.frozen_note
        assert cli.main(["freeze", "--proposed", str(prop), "--target", str(target), "--note", "again"]) == 2


class TestFreeze:
    def test_requires_note_and_never_refreezes(self, tmp_path):
        p, t = tmp_path / "p.json", tmp_path / "t.json"
        write_thresholds(p, OutcomeThresholds(success_val_mse=2e-3))
        with pytest.raises(ValueError, match="note"):
            freeze_thresholds(p, t, "  ")
        th = freeze_thresholds(p, t, "reviewed")
        assert th.frozen and th.success_val_mse == 2e-3
        assert load_thresholds(t, require_frozen=True, purpose="main") == th
        with pytest.raises(ValueError, match="already frozen"):
            freeze_thresholds(p, t, "again")

    def test_repo_thresholds_are_still_draft(self):
        assert load_thresholds(REPO / "configs" / "a1" / "outcomes.json").frozen is False


def test_timing_script_measures_the_runner_metrics():
    src = (REPO / "scripts" / "a1_time.py").read_text(encoding="utf-8")
    assert src.count("extra_metrics=a1_extra_metrics") >= 4
