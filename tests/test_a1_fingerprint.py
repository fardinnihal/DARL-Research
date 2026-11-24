"""Stage 2 (A1), Step 8 prep: fingerprints locate cross-machine differences exactly."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from darl_study.a1.runner import Arm, ExperimentConfig, run_experiment

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a1_fingerprint", REPO / "scripts" / "a1_fingerprint.py")
fpmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fpmod)

TRAIN = {"epochs": 20, "log_every": 10, "checkpoint_every": 10}
ARMS = (Arm("ReLU-He", "relu", "he"), Arm("ReLU-RAI", "relu", "rai"))


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fp")
    cfg = tmp / "p.json"
    cfg.write_text(json.dumps({"name": "p", "mode": "pilot", "seeds": {"start": 0, "count": 2}, "train": TRAIN,
                               "results_root": str(tmp / "results"),
                               "arms": [{"name": a.name, "activation": a.activation, "init": a.init} for a in ARMS]}))
    exp = ExperimentConfig(name="p", mode="pilot", seed_start=0, seed_count=2, arms=ARMS, train=dict(TRAIN),
                           results_root=str(tmp / "results"))
    run_experiment(exp, report=lambda m: None)
    before = {p: p.read_bytes() for p in exp.root.rglob("*") if p.is_file()}
    fp = fpmod.make(cfg)
    assert {p: p.read_bytes() for p in exp.root.rglob("*") if p.is_file()} == before  # read only
    return cfg, fp


def test_fingerprint_contents(made):
    _, fp = made
    assert len(fp["runs"]) == 4 and all(v is not None for v in fp["runs"].values())
    r = fp["runs"]["ReLU-He/0"]
    assert r["status"] == "completed" and len(r["train_mse_hex"]) == len(r["epochs"])
    assert float.fromhex(r["train_mse_hex"][0]) > 0
    assert fp["experiment_record"]["dataset_sha256"]["train"].startswith("a29ad95b")
    assert "a1/train.py" in fp["source_sha256"]


def test_identical_fingerprints_compare_bit_identical(made):
    _, fp = made
    c = fpmod.compare(fp, copy.deepcopy(fp))
    assert c["summary"]["bit_identical_runs"] == 4 and c["summary"]["inputs_identical"]
    assert c["summary"]["earliest_divergence_epoch"] is None and c["source_files_differing"] == []


def test_first_differing_epoch_is_located(made):
    _, fp = made
    other = copy.deepcopy(fp)
    r = other["runs"]["ReLU-RAI/1"]
    i = r["epochs"].index(10)
    for j in range(i, len(r["train_mse_hex"])):  # perturb from epoch 10 on by one ulp
        v = float.fromhex(r["train_mse_hex"][j])
        r["train_mse_hex"][j] = (v * (1 + 2 ** -52)).hex()
    c = fpmod.compare(fp, other)
    assert c["runs"]["ReLU-RAI/1"]["first_differing_epoch"] == 10
    assert c["runs"]["ReLU-RAI/1"]["epoch0_bit_identical"]
    assert c["summary"]["bit_identical_runs"] == 3 and c["summary"]["earliest_divergence_epoch"] == 10


def test_input_and_code_differences_are_flagged(made):
    _, fp = made
    other = copy.deepcopy(fp)
    other["runs"]["ReLU-He/0"]["config_hash"] = "0" * 64
    other["source_sha256"]["a1/train.py"] = "0" * 64
    other["runs"]["ReLU-He/1"] = None
    c = fpmod.compare(fp, other)
    assert not c["summary"]["inputs_identical"]
    assert c["source_files_differing"] == ["a1/train.py"]
    assert c["runs"]["ReLU-He/1"] == {"missing": [False, True]}


def test_cli_make_and_compare(made, tmp_path, capsys):
    cfg, _ = made
    a, b = tmp_path / "sub" / "a.json", tmp_path / "b.json"
    assert fpmod.main(["make", "--config", str(cfg), "--out", str(a)]) == 0
    assert fpmod.main(["make", "--config", str(cfg), "--out", str(b)]) == 0
    assert fpmod.main(["compare", str(a), str(b), "--out", str(tmp_path / "c.json")]) == 0
    out = capsys.readouterr().out
    assert "bit-identical runs: 4/4" in out
