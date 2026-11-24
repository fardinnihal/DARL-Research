"""Stage 2 (A1), Step 8: regression tests for the draft -> frozen transition.

Replays the Step 8 sequence on an isolated copy (never the repo file):
pilot with DRAFT thresholds -> analyse -> review hash -> freeze in place ->
what the pilot, the analyser and a main experiment do afterwards.
"""

import json
from pathlib import Path

import pytest

from darl_study.a1.outcomes import OutcomeThresholds, load_thresholds, write_thresholds
from darl_study.a1.pilot import ANALYSIS_DIRNAME, analyze_pilot, freeze_thresholds, preview_freeze
from darl_study.a1.runner import Arm, ExperimentConfig, run_dir_for, run_experiment

TINY = {"epochs": 100, "log_every": 10, "checkpoint_every": 50}
RELU = (Arm("ReLU-He", "relu", "he"), Arm("ReLU-RAI", "relu", "rai"))
NOTE = "UA reviewed test pilot"


def quiet(_msg):
    pass


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def snapshot(exp):
    """(outcome label, training history) of every run."""
    out = {}
    for arm in exp.arms:
        for s in exp.seeds:
            d = run_dir_for(exp, arm, s)
            out[(arm.name, s)] = (read(d / "outcome.json")["outcome"], read(d / "result.json")["history"])
    return out


@pytest.fixture(scope="module")
def frozen_pilot(tmp_path_factory):
    """A real tiny pilot run with draft thresholds, analysed, then frozen in place."""
    tmp = tmp_path_factory.mktemp("transition")
    th_path = tmp / "outcomes.json"
    write_thresholds(th_path, OutcomeThresholds())
    exp = ExperimentConfig(name="p", mode="pilot", seed_start=0, seed_count=2, arms=RELU,
                           train=dict(TINY), results_root=str(tmp / "results"), thresholds=str(th_path))
    run_experiment(exp, report=quiet)
    draft = load_thresholds(th_path, purpose="pilot")
    before = snapshot(exp)
    record_before = read(exp.root / "experiment.json")
    analyze_pilot(exp, draft)
    proposed = exp.root / ANALYSIS_DIRNAME / "outcomes_proposed.json"
    phash = load_thresholds(proposed).hash()
    frozen = freeze_thresholds(proposed, th_path, NOTE, phash[:12])
    return {"tmp": tmp, "exp": exp, "th_path": th_path, "draft": draft, "proposed": proposed,
            "phash": phash, "frozen": frozen, "before": before, "record_before": record_before}


class TestFreezeStep:
    def test_freeze_marks_frozen_and_keeps_every_value(self, frozen_pilot):
        f = frozen_pilot
        on_disk = load_thresholds(f["th_path"])
        prop = load_thresholds(f["proposed"])
        assert on_disk.frozen is True and on_disk == f["frozen"]
        for k, v in prop.to_dict().items():
            if k not in ("frozen", "frozen_note"):
                assert getattr(on_disk, k) == v, k
        assert on_disk.success_val_mse == 1e-3
        assert on_disk.frozen_note.startswith(NOTE + " | proposal " + f["phash"][:12])

    def test_freezing_does_not_touch_the_pilot(self, frozen_pilot):
        f = frozen_pilot
        assert snapshot(f["exp"]) == f["before"]
        assert read(f["exp"].root / "experiment.json") == f["record_before"]
        assert f["record_before"]["thresholds_hash"] == f["draft"].hash()

    def test_refreeze_is_refused_and_leaves_the_file(self, frozen_pilot):
        f = frozen_pilot
        before = f["th_path"].read_bytes()
        with pytest.raises(ValueError, match="already frozen"):
            freeze_thresholds(f["proposed"], f["th_path"], NOTE, f["phash"][:12])
        assert f["th_path"].read_bytes() == before

    def test_preview_after_freeze_reports_frozen_and_writes_nothing(self, frozen_pilot):
        f = frozen_pilot
        before = f["th_path"].read_bytes()
        pr = preview_freeze(f["proposed"], f["th_path"], f["phash"][:12])
        assert pr["target_frozen"] is True
        assert f["th_path"].read_bytes() == before

    def test_frozen_file_cannot_serve_as_a_proposal(self, frozen_pilot):
        f = frozen_pilot
        other = f["tmp"] / "other_target.json"
        with pytest.raises(ValueError, match="frozen"):
            freeze_thresholds(f["th_path"], other, NOTE, f["frozen"].hash()[:12])
        assert not other.exists()


class TestAfterFreeze:
    def test_analyser_refuses_the_frozen_file_but_accepts_the_draft(self, frozen_pilot):
        f = frozen_pilot
        with pytest.raises(ValueError, match="thresholds differ"):
            analyze_pilot(f["exp"], f["frozen"], out_dir=f["tmp"] / "never")
        assert not (f["tmp"] / "never").exists()
        analyze_pilot(f["exp"], f["draft"], out_dir=f["tmp"] / "again")  # pilot record unchanged
        assert (f["tmp"] / "again" / "pilot_report.md").exists()

    def test_main_refuses_a_draft_before_creating_anything(self, frozen_pilot, tmp_path):
        d = tmp_path / "draft.json"
        write_thresholds(d, OutcomeThresholds())
        exp = ExperimentConfig(name="m", mode="main", seed_start=100, seed_count=1, arms=RELU[:1],
                               train={"epochs": 3, "log_every": 2, "checkpoint_every": 1},
                               results_root=str(tmp_path / "results"), thresholds=str(d))
        with pytest.raises(ValueError, match="DRAFT"):
            run_experiment(exp, report=quiet)
        assert not (tmp_path / "results").exists()
        with pytest.raises(ValueError, match="DRAFT"):
            load_thresholds(d, purpose="main")

    def test_main_accepts_the_frozen_file(self, frozen_pilot, tmp_path):
        f = frozen_pilot
        exp = ExperimentConfig(name="m", mode="main", seed_start=100, seed_count=1, arms=RELU[:1],
                               train={"epochs": 3, "log_every": 2, "checkpoint_every": 1},
                               results_root=str(tmp_path / "results"), thresholds=str(f["th_path"]))
        res = run_experiment(exp, report=quiet)
        assert res[0]["action"] == "trained"
        out = read(run_dir_for(exp, RELU[0], 100) / "outcome.json")
        assert out["purpose"] == "main" and out["thresholds_frozen"] is True
        assert out["thresholds_hash"] == f["frozen"].hash()

    def test_rerunning_the_pilot_reclassifies_without_training(self, frozen_pilot, monkeypatch):
        """Last: it rewrites the pilot's outcome records (the other tests need them untouched)."""
        f = frozen_pilot

        def no_training(*a, **k):
            raise AssertionError("train() called although only the frozen flag changed")

        monkeypatch.setattr("darl_study.a1.runner.train", no_training)
        res = run_experiment(f["exp"], report=quiet)
        assert [r["action"] for r in res] == ["reclassified"] * 4
        after = snapshot(f["exp"])
        assert {k: h for k, (_, h) in after.items()} == {k: h for k, (_, h) in f["before"].items()}
        if f["frozen"].to_dict() | {"frozen": False, "frozen_note": ""} == \
                f["draft"].to_dict() | {"frozen": False, "frozen_note": ""}:
            assert after == f["before"]  # same values -> same labels
        for arm in f["exp"].arms:
            for s in f["exp"].seeds:
                out = read(run_dir_for(f["exp"], arm, s) / "outcome.json")
                assert out["thresholds_frozen"] is True and out["thresholds_hash"] == f["frozen"].hash()
        assert read(f["exp"].root / "experiment.json")["thresholds_hash"] == f["frozen"].hash()
        assert [r["action"] for r in run_experiment(f["exp"], report=quiet)] == ["skipped"] * 4
