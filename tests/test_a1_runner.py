"""Stage 2 (A1), Step 6: experiment runner (paired seeds, skip, resume, nonfinite, modes)."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

import darl_study.a1.train as train_mod
from darl_study.a1.outcomes import OutcomeThresholds, frozen_copy, write_thresholds
from darl_study.a1.runner import (
    MODE_PURPOSE,
    Arm,
    ExperimentConfig,
    load_experiment,
    run_dir_for,
    run_experiment,
    run_one,
)

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "a1" / "smoke.json"
TINY_TRAIN = {"epochs": 3, "log_every": 2, "checkpoint_every": 1}
ARMS = (Arm("ReLU-He", "relu", "he"), Arm("DARL-He", "darl", "he"))


def tiny(tmp_path, **over):
    kw = dict(name="t", mode="smoke", seed_start=0, seed_count=2, arms=ARMS,
              train=dict(TINY_TRAIN), results_root=str(tmp_path / "results"))
    kw.update(over)
    return ExperimentConfig(**kw)


def draft_file(dirpath):
    """An isolated DRAFT thresholds file (the repo file is frozen since Step 8)."""
    p = Path(dirpath) / "outcomes_draft.json"
    write_thresholds(p, OutcomeThresholds())
    return str(p)


def quiet(_msg):
    pass


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def histories(exp):
    out = {}
    for arm in exp.arms:
        for s in exp.seeds:
            r = read(run_dir_for(exp, arm, s) / "result.json")
            out[(arm.name, s)] = (r["status"], r["history"])
    return out


# ------------------------------------------------------------------ config
class TestSmokeConfig:
    def test_smoke_file(self):
        exp = load_experiment(SMOKE)
        assert exp.mode == "smoke" and exp.purpose == "pilot"
        assert exp.seeds == [0, 1, 2]
        assert exp.train["epochs"] == 20
        assert [a.name for a in exp.arms] == ["ReLU-He", "ReLU-RAI", "LReLU001-He", "DARL-He"]
        lrelu = exp.arms[2]
        assert lrelu.activation == "leaky_relu" and lrelu.act_kwargs == {"negative_slope": 0.01}
        assert exp.arms[3].activation == "darl" and exp.arms[3].init == "he"

    def test_smoke_train_settings_match_the_protocol(self):
        exp = load_experiment(SMOKE)
        from darl_study.a1.data import load_data_config
        cfg = exp.train_config(exp.arms[0], 0, load_data_config(REPO / "configs" / "a1" / "data.json"))
        assert (cfg.batch_size, cfg.lr, cfg.betas, cfg.n_layers, cfg.width) == (128, 1e-3, (0.9, 0.999), 10, 2)

    def test_mode_purpose_mapping(self):
        assert MODE_PURPOSE == {"smoke": "pilot", "pilot": "pilot",
                                "calibration": "calibration", "main": "main"}

    @pytest.mark.parametrize("over", [
        {"mode": "final"}, {"name": "../escape"}, {"name": ""}, {"seed_count": 0},
        {"seed_start": -1}, {"arms": ()}, {"arms": (ARMS[0], ARMS[0])},
        {"train": {"activation": "relu"}}, {"train": {"run_seed": 3}}, {"train": {"epoch": 3}},
        {"arms": (Arm("x", "elu", "he"),)}, {"arms": (Arm("x", "leaky_relu", "he"),)},
        {"arms": (Arm("x", "relu", "xavier"),)},
    ])
    def test_invalid_experiments_rejected(self, tmp_path, over):
        with pytest.raises(ValueError):
            tiny(tmp_path, **over)

    def test_unsafe_arm_name_rejected(self):
        with pytest.raises(ValueError):
            Arm("a/b", "relu", "he")

    def test_unknown_file_keys_rejected(self, tmp_path):
        d = read(SMOKE)
        d["seedz"] = 1
        p = tmp_path / "x.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown"):
            load_experiment(p)


# ------------------------------------------------------------------ running
class TestRun:
    def test_folders_files_and_index(self, tmp_path):
        exp = tiny(tmp_path, thresholds=draft_file(tmp_path))
        res = run_experiment(exp, report=quiet)
        assert [r["action"] for r in res] == ["trained"] * 4
        for arm in exp.arms:
            for s in exp.seeds:
                d = run_dir_for(exp, arm, s)
                assert sorted(p.name for p in d.iterdir()) == [
                    "DONE", "checkpoint.pt", "config.json", "outcome.json", "result.json"]
                done, out = read(d / "DONE"), read(d / "outcome.json")
                assert done["config_hash"] == read(d / "config.json")["config_hash"]
                assert out["purpose"] == "pilot" and out["thresholds_frozen"] is False
                assert out["outcome"] in ("collapsed", "half_trained", "success", "other", "nonfinite")
                assert "recovery" in out and "born_dead_record" in out
        rec = read(exp.root / "experiment.json")
        assert rec["note"].startswith("SMOKE TEST")
        assert rec["dataset_sha256"]["train"].startswith("a29ad95b")
        rows = list(csv.DictReader(open(exp.root / "runs_index.csv", encoding="utf-8")))
        assert len(rows) == 4 and {r["arm"] for r in rows} == {"ReLU-He", "DARL-He"}

    def test_arms_are_paired_by_seed(self, tmp_path):
        exp = tiny(tmp_path, arms=(Arm("LReLU001-He", "leaky_relu", "he", {"negative_slope": 0.01}),
                                   Arm("DARL-He", "darl", "he")))
        run_experiment(exp, report=quiet)
        for s in exp.seeds:
            a = read(run_dir_for(exp, exp.arms[0], s) / "config.json")
            b = read(run_dir_for(exp, exp.arms[1], s) / "config.json")
            assert a["seeds"] == b["seeds"] and a["dataset_sha256"] == b["dataset_sha256"]
            ha = read(run_dir_for(exp, exp.arms[0], s) / "result.json")["history"][0]
            hb = read(run_dir_for(exp, exp.arms[1], s) / "result.json")["history"][0]
            # Same tensors: DARL at D(0) = 0 equals LeakyReLU 0.01 at epoch 0.
            assert ha["train_mse"] == hb["train_mse"] and ha["grad_norm_layers"] == hb["grad_norm_layers"]
        a0 = read(run_dir_for(exp, exp.arms[0], 0) / "config.json")["seeds"]
        a1 = read(run_dir_for(exp, exp.arms[0], 1) / "config.json")["seeds"]
        assert a0 != a1

    def test_darl_updates_once_per_step_through_the_runner(self, tmp_path):
        exp = tiny(tmp_path, arms=(ARMS[1],), seed_count=1)
        run_experiment(exp, report=quiet)
        r = read(run_dir_for(exp, ARMS[1], 0) / "result.json")
        assert r["darl_num_updates"] == [3 * 24] * 9

    def test_completed_runs_are_skipped_without_retraining(self, tmp_path, monkeypatch):
        exp = tiny(tmp_path)
        run_experiment(exp, report=quiet)
        before = histories(exp)

        def no_training(*a, **k):
            raise AssertionError("train() called for a completed run")

        monkeypatch.setattr("darl_study.a1.runner.train", no_training)
        res = run_experiment(exp, report=quiet)
        assert [r["action"] for r in res] == ["skipped"] * 4
        assert histories(exp) == before

    def test_changed_thresholds_reclassify_without_retraining(self, tmp_path, monkeypatch):
        th_path = tmp_path / "th.json"
        write_thresholds(th_path, OutcomeThresholds())
        exp = tiny(tmp_path, thresholds=str(th_path))
        run_experiment(exp, report=quiet)
        before = histories(exp)
        write_thresholds(th_path, OutcomeThresholds(success_val_mse=0.5))  # edited in place

        def no_training(*a, **k):
            raise AssertionError("train() called although only thresholds changed")

        monkeypatch.setattr("darl_study.a1.runner.train", no_training)
        res = run_experiment(exp, report=quiet)
        assert [r["action"] for r in res] == ["reclassified"] * 4
        out = read(run_dir_for(exp, ARMS[0], 0) / "outcome.json")
        assert out["thresholds_hash"] == OutcomeThresholds(success_val_mse=0.5).hash()
        assert read(run_dir_for(exp, ARMS[0], 0) / "DONE")["thresholds_hash"] == out["thresholds_hash"]
        assert histories(exp) == before
        res = run_experiment(exp, report=quiet)  # now up to date
        assert [r["action"] for r in res] == ["skipped"] * 4

    def test_pointing_an_experiment_at_another_thresholds_file_is_refused(self, tmp_path):
        run_experiment(tiny(tmp_path, seed_count=1, arms=(ARMS[0],)), report=quiet)
        other = tmp_path / "other.json"
        write_thresholds(other, OutcomeThresholds())
        with pytest.raises(ValueError, match="different experiment"):
            run_experiment(tiny(tmp_path, seed_count=1, arms=(ARMS[0],), thresholds=str(other)), report=quiet)

    def test_folder_of_a_different_config_is_never_overwritten(self, tmp_path):
        exp = tiny(tmp_path, seed_count=1, arms=(ARMS[0],))
        run_experiment(exp, report=quiet)
        d = run_dir_for(exp, ARMS[0], 0)
        before = (d / "result.json").read_text(encoding="utf-8")
        # Same folder, different training config (epochs) -> conflict, untouched.
        from darl_study.a1.data import DataConfig
        other = exp.train_config(ARMS[0], 0, DataConfig())
        job = {"arm": "ReLU-He", "train_config": {**other.to_dict(), "epochs": 4},
               "thresholds": OutcomeThresholds().to_dict(), "purpose": "pilot", "mode": "smoke",
               "threads": 1, "run_dir": str(d)}
        r = run_one(job)
        assert r["action"] == "conflict"
        assert (d / "result.json").read_text(encoding="utf-8") == before
        # And the experiment level refuses a changed experiment under the same name.
        with pytest.raises(ValueError, match="different experiment"):
            run_experiment(tiny(tmp_path, seed_count=1, arms=(ARMS[0],), train={**TINY_TRAIN, "epochs": 4}),
                           report=quiet)

    def test_more_seeds_can_be_added_later(self, tmp_path):
        run_experiment(tiny(tmp_path, seed_count=1), report=quiet)
        res = run_experiment(tiny(tmp_path, seed_count=2), report=quiet)
        assert sorted(r["action"] for r in res) == ["skipped", "skipped", "trained", "trained"]

    def test_arm_filter(self, tmp_path):
        exp = tiny(tmp_path)
        res = run_experiment(exp, arm_names=["DARL-He"], report=quiet)
        assert {r["arm"] for r in res} == {"DARL-He"}
        with pytest.raises(ValueError, match="unknown arms"):
            run_experiment(exp, arm_names=["Nope"], report=quiet)

    def test_dry_run_writes_nothing(self, tmp_path):
        exp = tiny(tmp_path)
        res = run_experiment(exp, report=quiet, dry_run=True)
        assert [r["action"] for r in res] == ["would:new"] * 4
        assert not exp.root.exists()


# ---------------------------------------------------------- resume / failure
class TestResumeAndFailure:
    def test_interrupted_experiment_resumes_to_identical_results(self, tmp_path, monkeypatch):
        ref = tiny(tmp_path / "ref")
        run_experiment(ref, report=quiet)

        exp = tiny(tmp_path / "int")
        real = train_mod._mse
        calls = {"n": 0}

        def crash(pred, target):
            if pred.shape[0] in (128, 56):
                calls["n"] += 1
                if calls["n"] == 3 * 24 + 2 * 24 + 5:  # 2nd run, epoch 3, step 5
                    raise KeyboardInterrupt
            return real(pred, target)

        monkeypatch.setattr(train_mod, "_mse", crash)
        with pytest.raises(KeyboardInterrupt):
            run_experiment(exp, report=quiet)
        monkeypatch.undo()
        res = run_experiment(exp, report=quiet)
        actions = [r["action"] for r in res]
        assert actions[0] == "skipped" and actions[1] == "resumed@2"
        assert actions[2:] == ["trained", "trained"]
        assert histories(exp) == histories(ref)

    def test_nonfinite_run_is_recorded_and_never_retried(self, tmp_path, monkeypatch):
        exp = tiny(tmp_path, arms=(ARMS[1],), seed_count=1)
        real = train_mod._mse
        calls = {"n": 0}

        def nan(pred, target):
            out = real(pred, target)
            if pred.shape[0] in (128, 56):
                calls["n"] += 1
                if calls["n"] == 10:
                    return out * float("nan")
            return out

        monkeypatch.setattr(train_mod, "_mse", nan)
        res = run_experiment(exp, report=quiet)
        assert res[0]["status"] == "nonfinite" and res[0]["outcome"] == "nonfinite"
        d = run_dir_for(exp, ARMS[1], 0)
        assert read(d / "DONE")["outcome"] == "nonfinite"
        assert read(d / "outcome.json")["failure"]["phase"] == "training_step"
        monkeypatch.undo()  # a healthy rerun must not retry it
        res = run_experiment(exp, report=quiet)
        assert res[0]["action"] == "skipped" and res[0]["outcome"] == "nonfinite"

    def test_nonfinite_initial_network_is_recorded(self, tmp_path, monkeypatch):
        import torch
        real_init = train_mod.initialize_

        def bad_init(model, scheme, seed):
            out = real_init(model, scheme, seed)
            with torch.no_grad():
                model.linears[0].weight[0, 0] = float("inf")
            return out

        monkeypatch.setattr(train_mod, "initialize_", bad_init)
        monkeypatch.setattr("darl_study.a1.runner.initialize_", bad_init)
        exp = tiny(tmp_path, arms=(ARMS[1],), seed_count=1)
        res = run_experiment(exp, report=quiet)
        assert res[0]["outcome"] == "nonfinite"
        out = read(run_dir_for(exp, ARMS[1], 0) / "outcome.json")
        assert out["born_dead_record"] is None and out["failure"]["phase"] == "initial_evaluation"


# ------------------------------------------------- artifact validation
def _no_training(*a, **k):
    raise AssertionError("train() called although result.json is valid")


def _edit(path, **changes):
    rec = read(path)
    rec.update(changes)
    Path(path).write_text(json.dumps(rec), encoding="utf-8")


def _other_label(path):
    return next(o for o in ("success", "collapsed", "other") if o != read(path)["outcome"])


# Each tamper(run_dir) damages outcome.json or DONE while result.json stays valid.
OUTCOME_TAMPERS = {
    "outcome_missing": lambda d: (d / "outcome.json").unlink(),
    "outcome_invalid_json": lambda d: (d / "outcome.json").write_text("{not json", encoding="utf-8"),
    "outcome_truncated": lambda d: (d / "outcome.json").write_text(
        (d / "outcome.json").read_text(encoding="utf-8")[:40], encoding="utf-8"),
    "outcome_empty": lambda d: (d / "outcome.json").write_text("", encoding="utf-8"),
    "outcome_not_an_object": lambda d: (d / "outcome.json").write_text("[]", encoding="utf-8"),
    "outcome_binary": lambda d: (d / "outcome.json").write_bytes(b"\xff\xfe\x00garbage"),
    "outcome_wrong_config_hash": lambda d: _edit(d / "outcome.json", config_hash="0" * 64),
    "outcome_wrong_thresholds_hash": lambda d: _edit(d / "outcome.json", thresholds_hash="0" * 64),
    "outcome_unknown_label": lambda d: _edit(d / "outcome.json", outcome="bogus"),
    "outcome_label_missing": lambda d: _edit(d / "outcome.json", outcome=None),
    "outcome_disagrees_with_DONE": lambda d: _edit(d / "outcome.json", outcome=_other_label(d / "DONE")),
    "outcome_status_mismatch": lambda d: _edit(d / "outcome.json", status="nonfinite"),
    "DONE_missing": lambda d: (d / "DONE").unlink(),
    "DONE_invalid_json": lambda d: (d / "DONE").write_text("{", encoding="utf-8"),
    "DONE_disagrees_with_outcome": lambda d: _edit(d / "DONE", outcome=_other_label(d / "outcome.json")),
    "DONE_status_mismatch": lambda d: _edit(d / "DONE", status="nonfinite"),
    "outcome_and_DONE_missing": lambda d: ((d / "outcome.json").unlink(), (d / "DONE").unlink()),
}

# Each tamper(run_dir) makes result.json unusable: the run must not be skipped.
RESULT_TAMPERS = {
    "result_missing": lambda d: (d / "result.json").unlink(),
    "result_invalid_json": lambda d: (d / "result.json").write_text("{", encoding="utf-8"),
    "result_truncated": lambda d: (d / "result.json").write_text(
        (d / "result.json").read_text(encoding="utf-8")[:200], encoding="utf-8"),
    "result_not_an_object": lambda d: (d / "result.json").write_text("null", encoding="utf-8"),
    "result_running": lambda d: _edit(d / "result.json", status="running"),
    "result_wrong_config_hash": lambda d: _edit(d / "result.json", config_hash="0" * 64),
    "result_short_of_epochs": lambda d: _edit(d / "result.json", epochs_completed=1),
    "result_history_missing": lambda d: _edit(d / "result.json", history=None),
}


class TestArtifactValidation:
    """A run is skipped only when result.json, outcome.json and DONE are all valid."""

    @pytest.fixture
    def finished(self, tmp_path):
        exp = tiny(tmp_path, seed_count=1)
        run_experiment(exp, report=quiet)
        snap = {}
        for arm in exp.arms:
            d = run_dir_for(exp, arm, 0)
            snap[arm.name] = {n: (d / n).read_text(encoding="utf-8")
                              for n in ("outcome.json", "DONE", "result.json")}
        return exp, snap

    def test_intact_run_is_skipped(self, finished, monkeypatch):
        exp, _ = finished
        monkeypatch.setattr("darl_study.a1.runner.train", _no_training)
        assert [r["action"] for r in run_experiment(exp, report=quiet)] == ["skipped"] * 2

    @pytest.mark.parametrize("tamper", sorted(OUTCOME_TAMPERS))
    def test_bad_outcome_is_regenerated_without_retraining(self, finished, monkeypatch, tamper):
        exp, snap = finished
        for arm in exp.arms:
            OUTCOME_TAMPERS[tamper](run_dir_for(exp, arm, 0))
        monkeypatch.setattr("darl_study.a1.runner.train", _no_training)
        res = run_experiment(exp, report=quiet)
        assert [r["action"] for r in res] == ["regenerated"] * 2
        for arm in exp.arms:
            d = run_dir_for(exp, arm, 0)
            for name in ("outcome.json", "DONE", "result.json"):  # identical to the original
                assert (d / name).read_text(encoding="utf-8") == snap[arm.name][name], name
        assert [r["action"] for r in run_experiment(exp, report=quiet)] == ["skipped"] * 2

    def test_regenerated_outcome_is_back_in_the_index(self, finished, monkeypatch):
        exp, _ = finished
        (run_dir_for(exp, ARMS[0], 0) / "outcome.json").unlink()
        monkeypatch.setattr("darl_study.a1.runner.train", _no_training)
        run_experiment(exp, report=quiet)
        with open(exp.root / "runs_index.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert sorted(r["arm"] for r in rows) == ["DARL-He", "ReLU-He"]

    def test_dry_run_reports_regenerate_and_writes_nothing(self, finished):
        exp, _ = finished
        d = run_dir_for(exp, ARMS[0], 0)
        (d / "outcome.json").unlink()
        res = run_experiment(exp, report=quiet, dry_run=True)
        assert [r["action"] for r in res] == ["would:regenerate", "would:done"]
        assert not (d / "outcome.json").exists()

    def test_skip_result_uses_the_validated_outcome(self, finished):
        exp, snap = finished
        res = run_experiment(exp, report=quiet)
        for r in res:
            out = json.loads(snap[r["arm"]]["outcome.json"])
            assert (r["action"], r["status"], r["outcome"]) == ("skipped", out["status"], out["outcome"])

    @pytest.mark.parametrize("tamper", sorted(RESULT_TAMPERS))
    def test_bad_result_is_never_skipped(self, finished, tamper):
        exp, snap = finished
        d = run_dir_for(exp, ARMS[0], 0)
        RESULT_TAMPERS[tamper](d)
        res = run_experiment(exp, report=quiet)
        assert res[0]["action"].startswith("resumed@") and res[1]["action"] == "skipped"
        # Resuming from the final checkpoint restores the same result and outcome.
        new = read(d / "result.json")
        old = json.loads(snap[ARMS[0].name]["result.json"])
        assert (new["status"], new["history"]) == (old["status"], old["history"])
        assert read(d / "outcome.json") == json.loads(snap[ARMS[0].name]["outcome.json"])

    def test_bad_result_with_good_outcome_does_not_skip(self, finished, monkeypatch):
        exp, _ = finished
        d = run_dir_for(exp, ARMS[0], 0)
        (d / "result.json").write_text("{", encoding="utf-8")
        calls = []
        monkeypatch.setattr("darl_study.a1.runner.train",
                            lambda *a, **k: calls.append(1) or train_mod.train(*a, **k))
        run_experiment(exp, report=quiet)
        assert calls == [1]  # only the damaged run trains

    def test_conflicting_DONE_is_still_a_conflict(self, finished):
        exp, snap = finished
        d = run_dir_for(exp, ARMS[0], 0)
        (d / "outcome.json").unlink()
        _edit(d / "DONE", config_hash="0" * 64)
        res = run_experiment(exp, report=quiet)
        assert res[0]["action"] == "conflict"
        assert not (d / "outcome.json").exists()

    def test_nonfinite_outcome_is_regenerated_without_retraining(self, tmp_path, monkeypatch):
        exp = tiny(tmp_path, arms=(ARMS[1],), seed_count=1)
        real = train_mod._mse
        calls = {"n": 0}

        def nan(pred, target):
            out = real(pred, target)
            if pred.shape[0] in (128, 56):
                calls["n"] += 1
                if calls["n"] == 10:
                    return out * float("nan")
            return out

        monkeypatch.setattr(train_mod, "_mse", nan)
        run_experiment(exp, report=quiet)
        d = run_dir_for(exp, ARMS[1], 0)
        before = (d / "outcome.json").read_text(encoding="utf-8")
        (d / "outcome.json").unlink()
        monkeypatch.undo()
        monkeypatch.setattr("darl_study.a1.runner.train", _no_training)
        res = run_experiment(exp, report=quiet)
        assert (res[0]["action"], res[0]["status"], res[0]["outcome"]) == ("regenerated", "nonfinite", "nonfinite")
        assert (d / "outcome.json").read_text(encoding="utf-8") == before


# ------------------------------------------------------------------- modes
class TestModes:
    def test_smoke_with_the_frozen_repo_thresholds_records_frozen(self, tmp_path):
        exp = tiny(tmp_path, seed_count=1, arms=ARMS[:1])  # default = configs/a1/outcomes.json
        run_experiment(exp, report=quiet)
        out = read(run_dir_for(exp, ARMS[0], 0) / "outcome.json")
        assert out["purpose"] == "pilot" and out["thresholds_frozen"] is True

    def test_main_mode_runs_with_the_frozen_repo_thresholds(self, tmp_path):
        exp = tiny(tmp_path, mode="main", seed_count=1, arms=ARMS[:1])
        res = run_experiment(exp, report=quiet)
        assert res[0]["action"] == "trained"
        out = read(run_dir_for(exp, ARMS[0], 0) / "outcome.json")
        assert out["purpose"] == "main" and out["thresholds_frozen"] is True

    def test_main_mode_refuses_draft_thresholds_before_training(self, tmp_path):
        exp = tiny(tmp_path, mode="main", thresholds=draft_file(tmp_path))
        with pytest.raises(ValueError, match="DRAFT"):
            run_experiment(exp, report=quiet)
        assert not exp.root.exists()

    def test_main_mode_runs_with_frozen_thresholds(self, tmp_path):
        p = tmp_path / "frozen.json"
        write_thresholds(p, frozen_copy(OutcomeThresholds(), "test"))
        exp = tiny(tmp_path, mode="main", thresholds=str(p), seed_count=1, arms=(ARMS[0],))
        run_experiment(exp, report=quiet)
        out = read(run_dir_for(exp, ARMS[0], 0) / "outcome.json")
        assert out["purpose"] == "main" and out["thresholds_frozen"] is True
        assert exp.root.parts[-2] == "main"

    @pytest.mark.parametrize("mode", ["smoke", "pilot", "calibration"])
    def test_draft_modes_are_kept_apart(self, tmp_path, mode):
        exp = tiny(tmp_path, mode=mode, seed_count=1, arms=(ARMS[0],))
        run_experiment(exp, report=quiet)
        assert exp.root == tmp_path / "results" / mode / "t"
        assert read(run_dir_for(exp, ARMS[0], 0) / "outcome.json")["purpose"] == MODE_PURPOSE[mode]

    def test_smoke_banner_says_no_conclusions(self, tmp_path):
        lines = []
        run_experiment(tiny(tmp_path, seed_count=1, arms=(ARMS[0],)), report=lines.append)
        assert any("Do not draw scientific conclusions" in l for l in lines)


# --------------------------------------------------------- workers / script
def test_parallel_workers_match_serial(tmp_path):
    serial = tiny(tmp_path / "s")
    run_experiment(serial, report=quiet)
    par = tiny(tmp_path / "p")
    res = run_experiment(par, workers=2, report=quiet)
    assert sorted(r["action"] for r in res) == ["trained"] * 4
    assert histories(par) == histories(serial)


def test_script_dry_run(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("a1_run", REPO / "scripts" / "a1_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = read(SMOKE)
    d["results_root"] = str(tmp_path / "results")
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert mod.main(["--config", str(p), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "12 runs" in out and "SMOKE TEST" in out
    assert not (tmp_path / "results").exists()
