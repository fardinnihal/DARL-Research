"""Stage 2 (A1), Step 5: outcome classification and threshold freezing."""

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from darl_study.a1.data import build_a1_datasets
from darl_study.a1.init import initialize_
from darl_study.a1.metrics import a1_extra_metrics, function_shape
from darl_study.a1.model import DeepNarrowMLP
from darl_study.a1.outcomes import (
    DRAFT_PURPOSES,
    OUTCOMES,
    RUN_PURPOSES,
    OutcomeThresholds,
    check_thresholds,
    born_dead,
    classify_result,
    classify_shape,
    collapse_timeline,
    frozen_copy,
    load_thresholds,
    write_thresholds,
)
from darl_study.a1.train import TrainConfig, train

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "a1" / "outcomes.json"
DATA = build_a1_datasets()
TH = OutcomeThresholds()
# Step 8: configs/a1/outcomes.json is frozen. Its five values are the approved
# ones (the draft defaults), frozen from the reviewed proposal with this hash.
STEP8_PROPOSAL_HASH = "f1e7f80f6c4e"
STEP8_PROPOSAL_NOTE = ("PROPOSED from ReLU pilot 'relu-pilot'; review, then freeze with "
                       "scripts/a1_pilot.py freeze")
APPROVED_VALUES = {"collapse_std_frac": 0.01, "success_val_mse": 1e-3, "half_fit_mse": 1e-3,
                   "half_flat_std_frac": 0.01, "const_range_tol": 1e-6}


def draft_file(dirpath):
    """An isolated DRAFT thresholds file, for tests of draft behaviour."""
    p = Path(dirpath) / "outcomes_draft.json"
    write_thresholds(p, OutcomeThresholds())
    return p


def shape(out_std=0.5, left_mse=1e-6, right_mse=1e-6, left_std=0.5, right_std=0.5):
    return {"out_std": out_std, "target_std": 0.5, "left_mse": left_mse, "right_mse": right_mse,
            "left_out_std": left_std, "right_out_std": right_std,
            "left_target_std": 0.5, "right_target_std": 0.5}


def net(kind):
    """Hand-built networks with known shapes on [-sqrt3, sqrt3]."""
    m = DeepNarrowMLP()
    with torch.no_grad():
        m.linears[0].weight.copy_(torch.tensor([[1.0], [-1.0]]))
        m.linears[0].bias.zero_()
        for l in m.linears[1:9]:
            l.weight.copy_(torch.eye(2))
            l.bias.zero_()
        w = {"abs": [1.0, 1.0], "right": [1.0, 0.0], "left": [0.0, 1.0], "skew": [1.0, 0.5]}.get(kind, [1.0, 1.0])
        m.linears[9].weight.copy_(torch.tensor([w]))
        m.linears[9].bias.zero_()
        if kind == "dead":
            m.linears[3].bias.fill_(-100.0)
            m.linears[9].bias.fill_(0.866)
    return m


def val_mse(m):
    with torch.no_grad():
        return float(((m(DATA["val"].x) - DATA["val"].y) ** 2).mean())


# --------------------------------------------------------------- thresholds
class TestThresholds:
    def test_draft_file_matches_draft_defaults(self, tmp_path):
        th = load_thresholds(draft_file(tmp_path))
        assert th == OutcomeThresholds()
        assert th.frozen is False

    def test_draft_thresholds_refused_for_research_runs(self, tmp_path):
        with pytest.raises(ValueError, match="DRAFT"):
            load_thresholds(draft_file(tmp_path), require_frozen=True)


class TestFrozenRepoConfig:
    """Step 8: the production thresholds file is frozen with the approved values."""

    def test_repo_config_is_frozen_with_the_approved_values(self):
        th = load_thresholds(CONFIG, require_frozen=True)
        assert th.frozen is True
        assert {k: v for k, v in th.to_dict().items() if k in APPROVED_VALUES} == APPROVED_VALUES
        assert dataclasses.replace(th, frozen=False, frozen_note=TH.frozen_note) == TH  # = draft defaults

    def test_repo_config_is_the_reviewed_proposal(self):
        th = load_thresholds(CONFIG)
        assert f"proposal {STEP8_PROPOSAL_HASH}" in th.frozen_note
        proposal = dataclasses.replace(th, frozen=False, frozen_note=STEP8_PROPOSAL_NOTE)
        assert proposal.hash().startswith(STEP8_PROPOSAL_HASH)

    @pytest.mark.parametrize("purpose", ["pilot", "calibration", "main"])
    def test_repo_config_accepted_for_every_purpose(self, purpose):
        assert load_thresholds(CONFIG, purpose=purpose).frozen

    def test_frozen_file_accepted(self, tmp_path):
        p = tmp_path / "o.json"
        write_thresholds(p, frozen_copy(TH, "calibrated on ReLU pilot"))
        th = load_thresholds(p, require_frozen=True)
        assert th.frozen and "frozen" in th.frozen_note
        assert th.collapse_std_frac == TH.collapse_std_frac

    def test_hash_changes_with_any_threshold(self):
        assert TH.hash() != dataclasses.replace(TH, success_val_mse=2e-3).hash()
        assert TH.hash() != frozen_copy(TH, "x").hash()

    @pytest.mark.parametrize("kw", [{"collapse_std_frac": 0.0}, {"success_val_mse": -1.0},
                                    {"half_fit_mse": float("nan")}, {"collapse_std_frac": 1.0},
                                    {"const_range_tol": True}, {"frozen": "yes"}])
    def test_invalid_thresholds_rejected(self, kw):
        with pytest.raises(ValueError):
            OutcomeThresholds(**kw)

    def test_unknown_or_missing_keys_rejected(self, tmp_path):
        p = tmp_path / "o.json"
        d = TH.to_dict()
        d["colapse"] = 1
        p.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown"):
            load_thresholds(p)
        d = TH.to_dict()
        del d["success_val_mse"]
        p.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            load_thresholds(p)


# ------------------------------------------------------------ classify_shape
class TestClassifyShape:
    def test_success(self):
        assert classify_shape(shape(), 5e-4, TH) == "success"

    def test_collapsed(self):
        assert classify_shape(shape(out_std=0.004), 0.25, TH) == "collapsed"

    def test_collapse_threshold_boundary(self):
        assert classify_shape(shape(out_std=0.00499), 0.25, TH) == "collapsed"
        assert classify_shape(shape(out_std=0.00501, left_mse=0.2, right_mse=0.2), 0.25, TH) == "other"

    def test_success_boundary(self):
        assert classify_shape(shape(), 0.000999, TH) == "success"
        assert classify_shape(shape(left_mse=0.1, right_mse=0.1), 0.001, TH) == "other"

    @pytest.mark.parametrize("fit,flat", [("left", "right"), ("right", "left")])
    def test_half_trained_either_side(self, fit, flat):
        kw = {f"{fit}_mse": 1e-5, f"{flat}_mse": 0.4, f"{flat}_std": 0.0, f"{fit}_std": 0.5}
        assert classify_shape(shape(out_std=0.3, **kw), 0.2, TH) == "half_trained"

    def test_half_fit_but_other_side_not_flat_is_other(self):
        assert classify_shape(shape(out_std=0.3, left_mse=1e-5, right_mse=0.1, right_std=0.2), 0.05, TH) == "other"

    def test_collapse_takes_precedence(self):
        assert classify_shape(shape(out_std=0.0, left_mse=1e-5, right_std=0.0), 1e-4, TH) == "collapsed"

    def test_nonfinite_values(self):
        assert classify_shape(shape(out_std=float("nan")), 0.1, TH) == "nonfinite"
        assert classify_shape(shape(), float("inf"), TH) == "nonfinite"

    def test_outcome_names(self):
        assert set(OUTCOMES) == {"collapsed", "half_trained", "success", "other", "nonfinite"}


# --------------------------------------------- shapes of real hand-built nets
class TestHandBuiltNetworks:
    def test_exact_abs_is_success(self):
        m = net("abs")
        assert classify_shape(function_shape(m, DATA), val_mse(m), TH) == "success"

    @pytest.mark.parametrize("kind", ["right", "left"])
    def test_one_sided_relu_is_half_trained(self, kind):
        m = net(kind)
        assert classify_shape(function_shape(m, DATA), val_mse(m), TH) == "half_trained"

    def test_dead_network_is_collapsed(self):
        m = net("dead")
        assert classify_shape(function_shape(m, DATA), val_mse(m), TH) == "collapsed"

    def test_wrong_slope_is_other(self):
        m = net("skew")
        assert classify_shape(function_shape(m, DATA), val_mse(m), TH) == "other"


# ------------------------------------------------------------------ born dead
class TestBornDead:
    def test_matches_step3_script_on_100_seeds(self):
        spec = importlib.util.spec_from_file_location("bd", REPO / "scripts" / "a1_born_dead.py")
        bd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bd)
        x = DATA["train"].x
        for scheme in ("he", "rai"):
            for s in range(50):
                m = DeepNarrowMLP()
                initialize_(m, scheme, s)
                a, b = born_dead(m, x), bd.born_dead_record(m, x)
                pairs = (("structural_dead_layer", "dead_layer"), ("first_dead_layer", "first_dead"),
                         ("out_range", "out_range"), ("near_constant_output", "const_output"))
                for ka, kb in pairs:
                    assert a[ka] == b[kb], (scheme, s, ka)

    def test_hand_built(self):
        d, a = born_dead(net("dead"), DATA["train"].x), born_dead(net("abs"), DATA["train"].x)
        assert (d["structural_dead_layer"], d["born_dead"], d["float_constant_output"]) == (1, 1, 1)
        assert (a["structural_dead_layer"], a["born_dead"], a["float_constant_output"]) == (0, 0, 0)


# --------------------------------------------------------- timeline / result
def hist(*stds):
    return [{"epoch": 10 * i, "probe_out_std": s, "shape": shape(out_std=s)} for i, s in enumerate(stds)]


class TestTimeline:
    def test_collapse_during_training(self):
        t = collapse_timeline(hist(0.3, 0.1, 0.0, 0.0), TH)
        assert t == {"collapsed_at_init": False, "collapsed_at_end": True,
                     "first_collapse_epoch": 20, "uncollapsed_after_collapse": False}

    def test_collapsed_from_init(self):
        t = collapse_timeline(hist(0.0, 0.0), TH)
        assert t["collapsed_at_init"] and t["first_collapse_epoch"] == 0

    def test_escape_from_collapse_is_flagged(self):
        t = collapse_timeline(hist(0.0, 0.2, 0.4), TH)
        assert t["uncollapsed_after_collapse"] is True and t["collapsed_at_end"] is False

    def test_never_collapsed(self):
        t = collapse_timeline(hist(0.3, 0.5), TH)
        assert t["first_collapse_epoch"] is None and t["uncollapsed_after_collapse"] is None

    def test_empty_history(self):
        assert collapse_timeline([], TH)["collapsed_at_end"] is None

    def test_falls_back_to_probe_out_std(self):
        h = [{"epoch": 0, "probe_out_std": 0.001}]
        assert collapse_timeline(h, TH)["collapsed_at_init"] is True


class TestClassifyResult:
    def test_nonfinite_run(self):
        r = {"status": "nonfinite", "history": hist(0.3)}
        assert classify_result(r, TH, purpose="pilot")["outcome"] == "nonfinite"

    def test_unfinished_run_refused(self):
        with pytest.raises(ValueError, match="not finished"):
            classify_result({"status": "running", "history": hist(0.3)}, TH, purpose="pilot")

    def test_needs_shape_metrics(self):
        r = {"status": "completed", "history": [{"epoch": 0, "probe_out_std": 0.3, "val_mse": 0.1}]}
        with pytest.raises(ValueError, match="shape"):
            classify_result(r, TH, purpose="pilot")

    def test_real_run_born_dead_relu_collapses(self):
        cfg = TrainConfig(run_seed=1, epochs=3)  # seed 1 / He / ReLU is born dead (Step 3)
        m = DeepNarrowMLP()
        initialize_(m, "he", 1)
        b = born_dead(m, DATA["train"].x)
        r = train(cfg, extra_metrics=a1_extra_metrics)
        o = classify_result(r, TH, b, purpose="pilot")
        assert o["outcome"] == "collapsed" and o["born_dead"] == 1
        assert o["structural_dead_layer_at_init"] == 1 and o["collapsed_during_training"] == 0
        assert o["collapsed_at_init"] is True
        assert o["collapsed_after_training_not_born_dead"] == 0
        assert o["thresholds_frozen"] is False and o["thresholds_hash"] == TH.hash()

    def test_result_round_trips_through_json(self, tmp_path):
        cfg = TrainConfig(activation="darl", init="rai", run_seed=0, epochs=3)
        train(cfg, run_dir=tmp_path, extra_metrics=a1_extra_metrics)
        saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
        o = classify_result(saved, TH, purpose="calibration")
        assert o["outcome"] in OUTCOMES


# ------------------------- Step 5 review: structural dead layer vs constant output
class TestStructuralVersusConstant:
    def _labels(self, act, scheme, seed):
        kw = {"negative_slope": 0.01} if act == "leaky_relu" else {}
        m = DeepNarrowMLP(activation=act, act_kwargs=kw)
        initialize_(m, scheme, seed)
        return born_dead(m, DATA["train"].x)

    @pytest.mark.parametrize("scheme", ["he", "rai"])
    def test_relu_structural_dead_layer_means_exactly_constant(self, scheme):
        for seed in range(100):
            b = self._labels("relu", scheme, seed)
            assert b["born_dead"] == b["structural_dead_layer"]
            if b["structural_dead_layer"]:
                assert b["float_constant_output"] == 1 and b["out_range"] == 0.0

    @pytest.mark.parametrize("act", ["leaky_relu", "darl"])
    def test_leaky_activations_never_get_born_dead_from_structure(self, act):
        structural_not_constant = 0
        for seed in range(100):
            b = self._labels(act, "he", seed)
            assert b["born_dead"] is None
            assert "not applicable" in b["born_dead_basis"]
            if b["structural_dead_layer"] and not b["float_constant_output"]:
                structural_not_constant += 1
        # Dead layers occur, yet the output still varies: the labels differ.
        assert structural_not_constant > 0

    def test_same_weights_different_labels_across_activations(self):
        # He seed 3 (Step 3 per-seed table): both ReLU and DARL have a structural
        # dead layer with the same tensors, but only ReLU is born dead.
        r = self._labels("relu", "he", 3)
        d = self._labels("darl", "he", 3)
        assert r["structural_dead_layer"] == d["structural_dead_layer"] == 1
        assert r["born_dead"] == 1 and d["born_dead"] is None
        assert r["out_range"] == 0.0 and d["out_range"] > 0.0

    def test_darl_result_has_no_lu_split_but_has_network_level_labels(self):
        m = DeepNarrowMLP(activation="darl")
        initialize_(m, "he", 3)
        b = born_dead(m, DATA["train"].x)
        r = train(TrainConfig(activation="darl", run_seed=3, epochs=2), extra_metrics=a1_extra_metrics)
        o = classify_result(r, TH, b, purpose="pilot")
        assert o["born_dead"] is None and o["structural_dead_layer_at_init"] == 1
        assert "collapsed_after_training_not_born_dead" not in o
        assert o["collapsed_at_init"] in (True, False)


# ------------------------------- Step 5 review: frozen thresholds for main runs
class TestRunPurpose:
    FROZEN = frozen_copy(OutcomeThresholds(), "test")

    def _completed(self):
        return {"status": "completed",
                "history": [{"epoch": 0, "probe_out_std": 0.5, "val_mse": 1e-4, "shape": shape()}]}

    def test_purposes(self):
        assert RUN_PURPOSES == ("pilot", "calibration", "main")
        assert DRAFT_PURPOSES == ("pilot", "calibration")

    @pytest.mark.parametrize("purpose", ["pilot", "calibration"])
    def test_draft_allowed_for_pilot_and_calibration(self, purpose):
        o = classify_result(self._completed(), TH, purpose=purpose)
        assert o["outcome"] == "success" and o["purpose"] == purpose and o["thresholds_frozen"] is False

    def test_draft_refused_for_main(self):
        with pytest.raises(ValueError, match="DRAFT"):
            classify_result(self._completed(), TH, purpose="main")
        with pytest.raises(ValueError, match="DRAFT"):
            classify_result({"status": "nonfinite", "history": []}, TH, purpose="main")

    def test_frozen_allowed_for_main(self):
        o = classify_result(self._completed(), self.FROZEN, purpose="main")
        assert o["thresholds_frozen"] is True and o["thresholds_hash"] == self.FROZEN.hash()

    @pytest.mark.parametrize("bad", ["smoke", "final", "", None])
    def test_unknown_purpose_refused(self, bad):
        with pytest.raises(ValueError, match="purpose"):
            check_thresholds(TH, bad)

    def test_purpose_is_mandatory(self):
        with pytest.raises(TypeError):
            classify_result(self._completed(), TH)

    def test_load_with_purpose(self, tmp_path):
        draft = draft_file(tmp_path)
        assert load_thresholds(draft, purpose="pilot") == TH
        assert load_thresholds(draft, purpose="calibration") == TH
        with pytest.raises(ValueError, match="DRAFT"):
            load_thresholds(draft, purpose="main")
        p = tmp_path / "frozen.json"
        write_thresholds(p, self.FROZEN)
        assert load_thresholds(p, purpose="main").frozen
