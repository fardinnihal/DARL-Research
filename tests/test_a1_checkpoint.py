"""Stage 2 (A1), Step 4: atomic checkpoints and exact CPU resume."""

import json
import os

import pytest
import torch

import darl_study.a1.checkpoint as ck_mod
import darl_study.a1.train as train_mod
from darl_study.a1.checkpoint import (
    CHECKPOINT_FORMAT,
    atomic_write_text,
    config_hash,
    load_checkpoint,
    save_checkpoint,
)
from darl_study.a1.train import CHECKPOINT_NAME, TrainConfig, train

ARMS = [
    ("relu", {}, "he"),
    ("leaky_relu", {"negative_slope": 0.01}, "rai"),
    ("darl", {}, "he"),
    ("darl", {}, "rai"),
]


def cfg(act="darl", kw=None, init="rai", **over):
    base = dict(activation=act, act_kwargs=kw or {}, init=init, run_seed=3, epochs=12,
                log_every=3, checkpoint_every=2)
    base.update(over)
    return TrainConfig(**base)


def comparable(result):
    """Everything except wall-clock time and platform strings."""
    return {k: v for k, v in result.items() if k not in ("wall_seconds", "versions")}


def load_state(run_dir):
    return load_checkpoint(run_dir / CHECKPOINT_NAME)


def assert_states_equal(a, b):
    assert a["model"].keys() == b["model"].keys()
    for k in a["model"]:
        va, vb = a["model"][k], b["model"][k]
        if torch.is_tensor(va):
            assert torch.equal(va, vb), k
        else:
            assert va == vb, k
    oa, ob = a["optimizer"], b["optimizer"]
    assert oa["param_groups"] == ob["param_groups"]
    for i in oa["state"]:
        for name, v in oa["state"][i].items():
            assert torch.equal(v, ob["state"][i][name]), (i, name)
    assert torch.equal(a["order_rng"], b["order_rng"])
    ra = {k: v for k, v in a["run"].items() if k != "wall_seconds"}
    rb = {k: v for k, v in b["run"].items() if k != "wall_seconds"}
    assert ra == rb


# ---------------------------------------------------------- atomic writes
class TestAtomicSave:
    def test_round_trip_with_weights_only(self, tmp_path):
        p = tmp_path / "c.pt"
        state = {"config_hash": "abc", "t": torch.arange(5), "run": {"epoch": 3, "history": [{"a": 1.5}]}}
        save_checkpoint(p, state)
        assert not (tmp_path / "c.pt.tmp").exists()
        back = load_checkpoint(p, expected_config_hash="abc")
        assert back["format"] == CHECKPOINT_FORMAT
        assert torch.equal(back["t"], state["t"]) and back["run"] == state["run"]
        torch.load(p, weights_only=True)  # loads without executing pickled code

    def test_failed_write_keeps_previous_checkpoint(self, tmp_path, monkeypatch):
        p = tmp_path / "c.pt"
        save_checkpoint(p, {"config_hash": "h", "v": torch.tensor(1)})

        def boom(obj, f, *a, **k):
            with open(f, "wb") as fh:
                fh.write(b"half-written")
            raise KeyboardInterrupt  # simulate Ctrl+C mid-write

        monkeypatch.setattr(ck_mod.torch, "save", boom)
        with pytest.raises(KeyboardInterrupt):
            save_checkpoint(p, {"config_hash": "h", "v": torch.tensor(2)})
        monkeypatch.undo()
        assert not (tmp_path / "c.pt.tmp").exists()
        assert int(load_checkpoint(p)["v"]) == 1

    def test_stale_tmp_file_is_ignored_and_replaced(self, tmp_path):
        p = tmp_path / "c.pt"
        (tmp_path / "c.pt.tmp").write_bytes(b"garbage from a crash")
        save_checkpoint(p, {"config_hash": "h", "v": torch.tensor(3)})
        assert int(load_checkpoint(p)["v"]) == 3
        assert not (tmp_path / "c.pt.tmp").exists()

    def test_wrong_config_hash_refused(self, tmp_path):
        p = tmp_path / "c.pt"
        save_checkpoint(p, {"config_hash": "aaa"})
        with pytest.raises(ValueError, match="different config"):
            load_checkpoint(p, expected_config_hash="bbb")

    def test_non_checkpoint_file_refused(self, tmp_path):
        p = tmp_path / "x.pt"
        torch.save({"hello": 1}, p)
        with pytest.raises(ValueError, match="not an A1 checkpoint"):
            load_checkpoint(p)

    def test_atomic_text(self, tmp_path):
        p = tmp_path / "r.json"
        atomic_write_text(p, '{"a": 1}')
        atomic_write_text(p, '{"a": 2}')
        assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2}
        assert not (tmp_path / "r.json.tmp").exists()

    def test_config_hash_is_order_independent(self):
        assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
        assert config_hash({"a": 1}) != config_hash({"a": 2})
        with pytest.raises(ValueError):
            config_hash({"a": float("nan")})


# ------------------------------------------------------------ exact resume
@pytest.mark.parametrize("act,kw,init", ARMS)
@pytest.mark.parametrize("split", [2, 5])
def test_resume_is_bitwise_identical(tmp_path, act, kw, init, split):
    c = cfg(act, kw, init)
    straight = train(c, run_dir=tmp_path / "straight")

    part = tmp_path / "split"
    first = train(c, run_dir=part, stop_after_epoch=split)
    assert first["status"] == "running" and first["epochs_completed"] == split
    resumed = train(c, run_dir=part)  # fresh objects, state read from disk

    assert resumed["status"] == "completed"
    assert comparable(resumed) == comparable(straight)
    assert_states_equal(load_state(part), load_state(tmp_path / "straight"))
    if act == "darl":
        assert resumed["darl_num_updates"] == [12 * 24] * 9


def test_resume_after_crash_between_checkpoints(tmp_path, monkeypatch):
    """Ctrl+C in the middle of epoch 6 (last checkpoint at epoch 4): the rerun
    restarts from epoch 4 and still matches an uninterrupted run exactly."""
    c = cfg("darl", init="he")
    straight = train(c, run_dir=tmp_path / "straight")

    real = train_mod._mse
    calls = {"n": 0}

    def crash(pred, target):
        if pred.shape[0] in (128, 56):
            calls["n"] += 1
            if calls["n"] == 5 * 24 + 7:  # step 7 of epoch 6
                raise KeyboardInterrupt
        return real(pred, target)

    run = tmp_path / "crashed"
    monkeypatch.setattr(train_mod, "_mse", crash)
    with pytest.raises(KeyboardInterrupt):
        train(c, run_dir=run)
    monkeypatch.undo()
    assert load_state(run)["run"]["epoch"] == 4

    resumed = train(c, run_dir=run)
    assert comparable(resumed) == comparable(straight)
    assert_states_equal(load_state(run), load_state(tmp_path / "straight"))


def test_completed_run_is_not_retrained(tmp_path, monkeypatch):
    c = cfg("relu", init="he", epochs=4)
    first = train(c, run_dir=tmp_path)

    def must_not_train(*a, **k):
        raise AssertionError("training step called on a completed run")

    monkeypatch.setattr(train_mod.torch.optim.Adam, "step", must_not_train)
    again = train(c, run_dir=tmp_path)
    assert comparable(again) == comparable(first)


def test_resume_with_changed_config_is_refused(tmp_path):
    train(cfg("darl", epochs=4), run_dir=tmp_path, stop_after_epoch=2)
    with pytest.raises(ValueError, match="different config"):
        train(cfg("darl", epochs=4, lr=2e-3), run_dir=tmp_path)


def test_run_folder_records_config_seeds_and_data(tmp_path):
    c = cfg("darl", epochs=2)
    r = train(c, run_dir=tmp_path)
    rec = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert rec["config_hash"] == c.hash() == r["config_hash"]
    assert rec["config"] == c.to_dict()
    assert rec["seeds"] == r["seeds"] and set(rec["seeds"]) == {"init", "order"}
    assert rec["dataset_sha256"]["train"].startswith("a29ad95b")
    assert rec["network"]["n_params"] == 55
    res = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert res["status"] == "completed" and len(res["history"]) == len(r["history"])
    assert sorted(os.listdir(tmp_path)) == ["checkpoint.pt", "config.json", "result.json"]
