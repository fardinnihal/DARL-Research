"""Stage 2 (A1), Step 9: main comparison design, slope-matched calibration, seed guards."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from darl_study.a1.main_plan import (
    build_main_config,
    check_seeds_disjoint,
    fill_template,
    load_calibration_runs,
    slope_match,
)
from darl_study.a1.model import DARL_A1_DEFAULTS
from darl_study.a1.outcomes import OutcomeThresholds, frozen_copy, load_thresholds, write_thresholds
from darl_study.a1.runner import (
    experiment_from_dict,
    load_experiment,
    run_dir_for,
    run_experiment,
    used_seed_ranges,
)

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "a1"
TINY = {"epochs": 3, "batch_size": 128, "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-08,
        "weight_decay": 0.0, "n_layers": 10, "width": 2, "log_every": 2, "checkpoint_every": 1}
MAIN_ARMS = ["ReLU-He", "LReLU001-He", "DARL-He", "LReLUM-He",
             "ReLU-RAI", "LReLU001-RAI", "DARL-RAI", "LReLUM-RAI"]


def quiet(_m):
    pass


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, d):
    Path(p).write_text(json.dumps(d, indent=2), encoding="utf-8")
    return Path(p)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------- repo configs
class TestRepoConfigs:
    def test_calibration_config(self):
        cal = load_experiment(CFG / "calibration.json")
        pilot = read(CFG / "pilot.json")
        assert cal.mode == "calibration" and cal.purpose == "calibration"
        assert cal.seeds == list(range(20, 40))
        assert [(a.name, a.activation, a.init, a.act_kwargs) for a in cal.arms] == [
            ("DARL-He", "darl", "he", {}), ("DARL-RAI", "darl", "rai", {})]
        assert cal.train == pilot["train"] and cal.train["epochs"] == 2000
        assert cal.thresholds == "configs/a1/outcomes.json"

    def test_main_template_design(self):
        t = read(CFG / "main_template.json")
        pilot = read(CFG / "pilot.json")
        assert t["mode"] == "main" and t["seeds"] == {"start": 1000, "count": 1000}
        assert t["train"] == pilot["train"] and t["train"]["epochs"] == 2000
        assert t["data_config"] == pilot["data_config"] and t["thresholds"] == "configs/a1/outcomes.json"
        assert [a["name"] for a in t["arms"]] == MAIN_ARMS
        by = {a["name"]: a for a in t["arms"]}
        for init, tag in (("he", "He"), ("rai", "RAI")):
            assert {by[f"{n}-{tag}"]["init"] for n in ("ReLU", "LReLU001", "DARL", "LReLUM")} == {init}
            assert by[f"ReLU-{tag}"]["activation"] == "relu" and "act_kwargs" not in by[f"ReLU-{tag}"]
            assert by[f"LReLU001-{tag}"]["act_kwargs"] == {"negative_slope": 0.01}
            assert by[f"DARL-{tag}"]["activation"] == "darl" and "act_kwargs" not in by[f"DARL-{tag}"]
            assert by[f"LReLUM-{tag}"]["act_kwargs"] == {"negative_slope": f"<slope_match:{init}>"}
        assert t["slope_match"]["calibration_config"] == "configs/a1/calibration.json"

    def test_template_cannot_be_run(self):
        with pytest.raises(ValueError):
            load_experiment(CFG / "main_template.json")
        t = read(CFG / "main_template.json")
        t.pop("slope_match")
        with pytest.raises(ValueError):  # placeholder is not a number
            experiment_from_dict(t)

    def test_pilot_calibration_and_main_seeds_are_disjoint(self):
        t = read(CFG / "main_template.json")
        t.pop("slope_match")
        for a in t["arms"]:
            if a["name"].startswith("LReLUM"):
                a["act_kwargs"] = {"negative_slope": 0.1}
        main = experiment_from_dict(t)
        cal, pilot = load_experiment(CFG / "calibration.json"), load_experiment(CFG / "pilot.json")
        check_seeds_disjoint(main, [("pilot", pilot.seed_start, pilot.seed_count),
                                    ("calibration", cal.seed_start, cal.seed_count)])
        assert not set(pilot.seeds) & set(cal.seeds)

    def test_smoke_main_covers_every_arm_and_is_not_research(self):
        s = load_experiment(CFG / "smoke_main.json")
        assert s.mode == "smoke" and s.purpose == "pilot" and s.train["epochs"] == 20
        assert [a.name for a in s.arms] == MAIN_ARMS
        t = read(CFG / "main_template.json")
        assert {k: v for k, v in s.train.items() if k not in ("epochs", "checkpoint_every")} == \
               {k: v for k, v in t["train"].items() if k not in ("epochs", "checkpoint_every")}

    def test_existing_main_config_matches_template_and_record(self):
        """Once a1_slope_match.py --write has run, main.json must be the filled template."""
        main = CFG / "main.json"
        if not main.exists():  # calibration pending: no main config and no slope record yet
            assert not (CFG / "main_slope_match.json").exists()
            return
        m, t = read(main), read(CFG / "main_template.json")
        rec = read(CFG / "main_slope_match.json")
        t.pop("slope_match")
        for a in t["arms"]:
            if a["name"].startswith("LReLUM"):
                a["act_kwargs"] = {"negative_slope": rec["per_init"][a["init"]]["alpha_bar"]}
        assert {k: v for k, v in m.items() if k != "comment"} == {k: v for k, v in t.items() if k != "comment"}


# ------------------------------------------------------- tiny calibration
def _frozen(tmp):
    p = tmp / "frozen.json"
    write_thresholds(p, frozen_copy(OutcomeThresholds(), "test"))
    return p


def _cal_dict(tmp, th, **over):
    d = {"name": "cal", "mode": "calibration", "seeds": {"start": 20, "count": 2}, "train": dict(TINY),
         "arms": [{"name": "DARL-He", "activation": "darl", "init": "he"},
                  {"name": "DARL-RAI", "activation": "darl", "init": "rai"}],
         "thresholds": str(th), "results_root": str(tmp / "results")}
    d.update(over)
    return d


def _template(tmp, th, cal_path, **over):
    t = read(CFG / "main_template.json")
    t["slope_match"]["calibration_config"] = str(cal_path)
    t.update({"name": "main", "seeds": {"start": 1000, "count": 2}, "train": dict(TINY),
              "thresholds": str(th), "results_root": str(tmp / "results")})
    t.update(over)
    return write(tmp / "template.json", t)


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cal")
    th = _frozen(tmp)
    cal_path = write(tmp / "cal.json", _cal_dict(tmp, th))
    cal = load_experiment(cal_path)
    run_experiment(cal, report=quiet)
    return {"tmp": tmp, "th": th, "cal_path": cal_path, "cal": cal}


class TestSlopeMatch:
    def test_alpha_bar_is_the_mean_of_the_runs_time_means(self, calibrated):
        cal = calibrated["cal"]
        m = slope_match(cal)
        assert set(m["per_init"]) == {"he", "rai"}
        for arm in cal.arms:
            vals = [read(run_dir_for(cal, arm, s) / "result.json")["darl_alpha_time_mean"] for s in cal.seeds]
            v = m["per_init"][arm.init]
            assert v["alpha_bar"] == pytest.approx(sum(vals) / len(vals), rel=1e-12)
            assert DARL_A1_DEFAULTS["alpha_min"] <= v["alpha_bar"] <= DARL_A1_DEFAULTS["alpha_max"]
            assert [r["seed"] for r in v["per_seed"]] == cal.seeds
            assert v["n_completed"] == 2 and v["n_nonfinite_excluded"] == 0
        assert m["calibration_seeds"] == {"start": 20, "count": 2}
        assert m["darl_parameters"] == DARL_A1_DEFAULTS

    def test_alpha_time_mean_is_training_state_only(self, calibrated):
        """The value equals the mean of the per-step slopes recorded during training
        (alpha_sum / steps), not anything measured on validation or probe data."""
        cal = calibrated["cal"]
        r = read(run_dir_for(cal, cal.arms[0], 20) / "result.json")
        assert r["darl_num_updates"] == [r["steps_completed"]] * 9
        assert r["final"]["darl_alpha_time_mean"] == r["darl_alpha_time_mean"]

    def test_incomplete_calibration_is_refused(self, tmp_path):
        th = _frozen(tmp_path)
        cal = experiment_from_dict(_cal_dict(tmp_path, th))
        run_experiment(cal, arm_names=["DARL-He"], report=quiet)
        with pytest.raises(ValueError, match="not finished"):
            slope_match(cal)

    def test_missing_calibration_is_refused(self, tmp_path):
        cal = experiment_from_dict(_cal_dict(tmp_path, _frozen(tmp_path)))
        with pytest.raises(ValueError, match="missing"):
            slope_match(cal)

    @pytest.mark.parametrize("over,msg", [
        ({"mode": "pilot"}, "calibration experiment"),
        ({"arms": [{"name": "R", "activation": "relu", "init": "he"}]}, "must be DARL"),
        ({"arms": [{"name": "D", "activation": "darl", "init": "he", "act_kwargs": {"rho": 0.9}}]},
         "approved DARL defaults"),
        ({"arms": [{"name": "A", "activation": "darl", "init": "he"},
                   {"name": "B", "activation": "darl", "init": "he"}]}, "one DARL arm per"),
    ])
    def test_wrong_calibration_design_is_refused(self, tmp_path, over, msg):
        cal = experiment_from_dict(_cal_dict(tmp_path, _frozen(tmp_path), **over))
        with pytest.raises(ValueError, match=msg):
            load_calibration_runs(cal)

    def test_a_tampered_run_is_refused(self, tmp_path):
        cal = experiment_from_dict(_cal_dict(tmp_path, _frozen(tmp_path)))
        run_experiment(cal, report=quiet)
        (run_dir_for(cal, cal.arms[1], 21) / "DONE").write_text("{", encoding="utf-8")
        with pytest.raises(ValueError, match="DARL-RAI seed 21"):
            slope_match(cal)


# ------------------------------------------------------------ main config
class TestBuildMain:
    def test_filled_main_config(self, calibrated):
        c = calibrated
        tpl = _template(c["tmp"], c["th"], c["cal_path"])
        main_dict, match = build_main_config(c["cal_path"], tpl)
        main = experiment_from_dict(main_dict)
        assert main.mode == "main" and [a.name for a in main.arms] == MAIN_ARMS
        for a in main.arms:
            if a.name.startswith("LReLUM"):
                assert a.act_kwargs == {"negative_slope": match["per_init"][a.init]["alpha_bar"]}
            elif a.name.startswith("LReLU001"):
                assert a.act_kwargs == {"negative_slope": 0.01}
        assert "slope_match" not in main_dict
        assert c["cal"].name in main_dict["comment"]
        assert match["main_thresholds_hash"] == load_thresholds(c["th"], purpose="main").hash()

    def test_overlapping_seeds_are_refused(self, calibrated):
        c = calibrated
        tpl = _template(c["tmp"], c["th"], c["cal_path"], seeds={"start": 21, "count": 5})
        with pytest.raises(ValueError, match="overlap calibration"):
            build_main_config(c["cal_path"], tpl)

    def test_pilot_seeds_are_refused(self, calibrated):
        c = calibrated
        tpl = _template(c["tmp"], c["th"], c["cal_path"], seeds={"start": 0, "count": 5})
        with pytest.raises(ValueError, match="overlap pilot"):
            build_main_config(c["cal_path"], tpl, CFG / "pilot.json")

    def test_different_training_settings_are_refused(self, calibrated):
        c = calibrated
        tpl = _template(c["tmp"], c["th"], c["cal_path"], train={**TINY, "lr": 0.01})
        with pytest.raises(ValueError, match="differ in train"):
            build_main_config(c["cal_path"], tpl)

    def test_draft_thresholds_are_refused_for_main(self, calibrated):
        c = calibrated
        draft = c["tmp"] / "draft.json"
        write_thresholds(draft, OutcomeThresholds())
        tpl = _template(c["tmp"], draft, c["cal_path"])
        with pytest.raises(ValueError, match="DRAFT"):
            build_main_config(c["cal_path"], tpl)

    def test_template_bound_to_another_calibration_is_refused(self, calibrated, tmp_path):
        c = calibrated
        tpl = _template(c["tmp"], c["th"], tmp_path / "other.json")
        with pytest.raises(ValueError, match="expects calibration config"):
            build_main_config(c["cal_path"], tpl)

    @pytest.mark.parametrize("slope,msg", [("<slope_match:rai>", "uses the rai slope"),
                                           ("0.1", "not a number or a placeholder"),
                                           ("<slope_match:xavier>", "uses the xavier slope")])
    def test_bad_placeholders(self, slope, msg):
        t = {"mode": "main", "arms": [{"name": "L", "activation": "leaky_relu", "init": "he",
                                       "act_kwargs": {"negative_slope": slope}}]}
        match = {"per_init": {"he": {"alpha_bar": 0.05}}, "calibration_experiment": "c",
                 "calibration_experiment_hash": "0" * 64, "calibration_seeds": {"start": 0, "count": 1},
                 "created": "x"}
        with pytest.raises(ValueError, match=msg):
            fill_template(t, match)

    def test_template_without_placeholders_is_refused(self):
        with pytest.raises(ValueError, match="no slope placeholders"):
            fill_template({"mode": "main", "arms": [{"name": "R", "activation": "relu", "init": "he"}]},
                          {"per_init": {}})

    def test_tiny_main_runs_paired_and_records_cost(self, calibrated, monkeypatch):
        c = calibrated
        tpl = _template(c["tmp"], c["th"], c["cal_path"], seeds={"start": 1000, "count": 1})
        main = experiment_from_dict(build_main_config(c["cal_path"], tpl)[0])
        res = run_experiment(main, report=quiet)
        assert sorted(r["action"] for r in res) == ["trained"] * 8
        seeds = {read(run_dir_for(main, a, 1000) / "result.json")["seeds"]["init"] for a in main.arms}
        assert len(seeds) == 1  # same init stream in every arm: paired by seed
        for a in main.arms:
            out = read(run_dir_for(main, a, 1000) / "outcome.json")
            assert out["purpose"] == "main" and out["thresholds_frozen"] is True
            res_json = read(run_dir_for(main, a, 1000) / "result.json")
            assert res_json["wall_seconds"] > 0 and res_json["versions"]["platform"]  # cost + platform
        rows = list(csv.DictReader(open(main.root / "runs_index.csv", encoding="utf-8")))
        assert len(rows) == 8
        assert all(r["darl_alpha_time_mean"] for r in rows if r["activation"] == "darl")


# ----------------------------------------------------------- runner guard
class TestMainSeedGuard:
    def _main(self, tmp, th, start):
        return experiment_from_dict({"name": "m", "mode": "main", "seeds": {"start": start, "count": 2},
                                     "train": dict(TINY), "arms": [{"name": "R", "activation": "relu",
                                                                     "init": "he"}],
                                     "thresholds": str(th), "results_root": str(tmp / "results")})

    def test_main_refuses_seeds_of_a_recorded_calibration(self, calibrated):
        c = calibrated
        main = self._main(c["tmp"], c["th"], 21)
        with pytest.raises(ValueError, match="overlap calibration 'cal'"):
            run_experiment(main, report=quiet)
        assert not main.root.exists()
        with pytest.raises(ValueError, match="overlap"):
            run_experiment(main, report=quiet, dry_run=True)

    def test_main_refuses_seeds_of_a_recorded_pilot(self, tmp_path):
        th = _frozen(tmp_path)
        pilot = experiment_from_dict({"name": "p", "mode": "pilot", "seeds": {"start": 0, "count": 2},
                                      "train": dict(TINY), "arms": [{"name": "R", "activation": "relu",
                                                                      "init": "he"}],
                                      "thresholds": str(th), "results_root": str(tmp_path / "results")})
        run_experiment(pilot, report=quiet)
        assert [u["name"] for u in used_seed_ranges(tmp_path / "results")] == ["p"]
        with pytest.raises(ValueError, match="overlap pilot 'p'"):
            run_experiment(self._main(tmp_path, th, 1), report=quiet)
        run_experiment(self._main(tmp_path, th, 2), report=quiet)  # disjoint: allowed

    def test_smoke_is_not_restricted(self, calibrated):
        c = calibrated
        s = experiment_from_dict({"name": "s", "mode": "smoke", "seeds": {"start": 20, "count": 1},
                                  "train": dict(TINY), "arms": [{"name": "R", "activation": "relu",
                                                                  "init": "he"}],
                                  "thresholds": str(c["th"]), "results_root": str(c["tmp"] / "results")})
        run_experiment(s, report=quiet)


# ----------------------------------------------------------------- script
class TestScript:
    def test_preview_then_write_then_idempotent(self, calibrated, tmp_path, capsys):
        c = calibrated
        tpl = _template(c["tmp"], c["th"], c["cal_path"])
        out = tmp_path / "main.json"
        mod = _script("a1_slope_match")
        argv = ["--calibration", str(c["cal_path"]), "--template", str(tpl),
                "--pilot", str(CFG / "pilot.json"), "--out", str(out)]
        assert mod.main(argv) == 0
        assert "PREVIEW" in capsys.readouterr().out and not out.exists()
        assert mod.main(argv + ["--write"]) == 0
        written = read(out)
        record = read(tmp_path / "main_slope_match.json")
        assert experiment_from_dict(written).mode == "main"
        slopes = {a["init"]: a["act_kwargs"]["negative_slope"] for a in written["arms"]
                  if a["name"].startswith("LReLUM")}
        assert slopes == {i: v["alpha_bar"] for i, v in record["per_init"].items()}
        before = out.read_bytes()
        assert mod.main(argv + ["--write"]) == 0 and out.read_bytes() == before
        changed = dict(written, seeds={"start": 5000, "count": 2})
        write(out, changed)
        assert mod.main(argv + ["--write"]) == 2
        assert "never overwritten" in capsys.readouterr().err
        assert read(out) == changed

    def test_script_refuses_unfinished_calibration(self, tmp_path, capsys):
        th = _frozen(tmp_path)
        cal_path = write(tmp_path / "cal.json", _cal_dict(tmp_path, th))
        tpl = _template(tmp_path, th, cal_path)
        out = tmp_path / "main.json"
        assert _script("a1_slope_match").main(["--calibration", str(cal_path), "--template", str(tpl),
                                               "--out", str(out), "--write"]) == 2
        assert not out.exists()
