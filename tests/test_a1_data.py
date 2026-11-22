"""Stage 2 (A1), Step 2: dataset integrity and reproducibility tests."""

import dataclasses
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from darl_study.a1.data import (
    SQRT3,
    A1Dataset,
    DataConfig,
    build_a1_datasets,
    dataset_manifest,
    load_data_config,
    make_probe_grid,
    make_uniform_set,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "a1" / "data.json"

# Recorded when data.py was written (numpy RandomState MT19937, float32 cast).
# The same numbers must come out on every machine; a mismatch means the data
# differ and no result would be comparable.
GOLDEN_SHA256 = {
    "train": "a29ad95bc4dc9b5a4fecdcd9d03d8c1a5ab3ce9a62778a2eb5c81b4947dc27f6",
    "val": "e3b8f03e66f14410fc772f1cbfe7d9bba6ba38ed1ac6e9bc8c8347a664a2b4c8",
    "probe": "13b6c4211ec04efc9bd34aaa30c797662dd3ed6f684c083a8292400febc0d884",
}
GOLDEN_FIRST_X = {
    "train": [0.16909493505954742, 0.7454378604888916, 0.35598278045654297],
    "val": [-0.28744420409202576, 0.7632264494895935, -1.7316546440124512],
}
F32_SQRT3 = float(np.float32(SQRT3))  # 1.7320507764816284, just below sqrt(3)


@pytest.fixture(scope="module")
def sets():
    return build_a1_datasets()


# --------------------------------------------------------------- reference
class TestReferenceConfig:
    def test_defaults_are_the_a1_reference(self):
        c = DataConfig()
        assert (c.n_train, c.n_val, c.n_probe) == (3000, 1000, 2001)
        assert c.low == -math.sqrt(3) and c.high == math.sqrt(3)
        assert c.target == "abs"
        assert (c.train_seed, c.val_seed) == (0, 1)

    def test_config_file_matches_defaults_exactly(self):
        assert load_data_config(CONFIG) == DataConfig()

    def test_config_file_is_plain_json(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        assert set(raw) == {f.name for f in dataclasses.fields(DataConfig)}


# ------------------------------------------------------------------ shapes
class TestShapesAndTypes:
    @pytest.mark.parametrize("name,n", [("train", 3000), ("val", 1000), ("probe", 2001)])
    def test_shape_dtype_device(self, sets, name, n):
        d = sets[name]
        assert len(d) == n
        for t in (d.x, d.y):
            assert t.shape == (n, 1)
            assert t.dtype == torch.float32
            assert t.device.type == "cpu"
            assert t.is_contiguous()
            assert not t.requires_grad
            assert torch.isfinite(t).all()

    def test_tensors_do_not_share_memory(self, sets):
        assert sets["train"].x.data_ptr() != sets["val"].x.data_ptr()
        assert sets["train"].x.data_ptr() != sets["train"].y.data_ptr()


# ------------------------------------------------------------------ target
class TestTarget:
    @pytest.mark.parametrize("name", ["train", "val", "probe"])
    def test_y_is_exactly_abs_x(self, sets, name):
        d = sets[name]
        assert torch.equal(d.y, d.x.abs())
        assert (d.y >= 0).all()

    def test_unknown_target_rejected(self):
        with pytest.raises(ValueError, match="target"):
            DataConfig(target="square")


# ------------------------------------------------------------ distribution
class TestUniformSets:
    @pytest.mark.parametrize("name", ["train", "val"])
    def test_inside_interval(self, sets, name):
        x = sets[name].x
        assert x.min() >= -F32_SQRT3
        assert x.max() <= F32_SQRT3

    @pytest.mark.parametrize("name", ["train", "val"])
    def test_moments_match_uniform(self, sets, name):
        # U[-sqrt3, sqrt3]: E x = 0, Var x = 1, E|x| = sqrt3/2, Var|x| = 1/4.
        x = sets[name].x.double().flatten()
        n = x.numel()
        se = 1 / math.sqrt(n)
        assert abs(x.mean().item()) < 4 * se
        assert abs(x.var(unbiased=False).item() - 1.0) < 4 * math.sqrt(0.8 / n)
        y = x.abs()
        assert abs(y.mean().item() - SQRT3 / 2) < 4 * 0.5 * se

    @pytest.mark.parametrize("name", ["train", "val"])
    def test_kolmogorov_smirnov_against_uniform(self, sets, name):
        x = np.sort(sets[name].x.double().numpy().ravel())
        n = x.size
        cdf = (x + SQRT3) / (2 * SQRT3)
        i = np.arange(1, n + 1)
        d = max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n))
        assert d < 1.63 / math.sqrt(n)  # alpha = 0.01 critical value

    def test_both_signs_well_represented(self, sets):
        x = sets["train"].x
        frac_neg = (x < 0).double().mean().item()
        assert 0.45 < frac_neg < 0.55

    def test_train_and_val_share_no_values(self, sets):
        tr = set(sets["train"].x.flatten().tolist())
        va = set(sets["val"].x.flatten().tolist())
        assert tr.isdisjoint(va)


# ------------------------------------------------------------------- probe
class TestProbeGrid:
    def test_endpoints_and_zero(self, sets):
        x = sets["probe"].x.flatten()
        assert x[0].item() == -F32_SQRT3
        assert x[-1].item() == F32_SQRT3
        assert x[1000].item() == 0.0
        assert (x == 0).sum().item() == 1

    def test_exactly_symmetric(self, sets):
        x = sets["probe"].x.flatten()
        assert torch.equal(x, -x.flip(0))

    def test_strictly_increasing_and_evenly_spaced(self, sets):
        x = sets["probe"].x.double().flatten()
        dx = x[1:] - x[:-1]
        assert (dx > 0).all()
        step = 2 * SQRT3 / 2000
        assert torch.allclose(dx, torch.full_like(dx, step), atol=1e-6)

    def test_has_no_seed(self, sets):
        assert sets["probe"].seed is None

    def test_asymmetric_interval_uses_linspace(self):
        cfg = DataConfig(low=-1.0, high=2.0, n_probe=4)
        x = make_probe_grid(cfg).x.flatten().tolist()
        assert x == [-1.0, 0.0, 1.0, 2.0]


# -------------------------------------------------------- reproducibility
class TestReproducibility:
    def test_two_builds_are_bitwise_identical(self):
        a, b = build_a1_datasets(), build_a1_datasets()
        for k in ("train", "val", "probe"):
            assert torch.equal(a[k].x, b[k].x)
            assert torch.equal(a[k].y, b[k].y)
            assert a[k].fingerprint() == b[k].fingerprint()

    @pytest.mark.parametrize("name", ["train", "val", "probe"])
    def test_golden_fingerprint(self, sets, name):
        assert sets[name].fingerprint() == GOLDEN_SHA256[name]

    @pytest.mark.parametrize("name", ["train", "val"])
    def test_golden_first_values(self, sets, name):
        assert sets[name].x[:3].flatten().tolist() == GOLDEN_FIRST_X[name]

    def test_seed_changes_data(self):
        cfg = DataConfig()
        a = make_uniform_set("a", 100, 0, cfg)
        b = make_uniform_set("b", 100, 2, cfg)
        assert not torch.equal(a.x, b.x)

    def test_train_set_does_not_depend_on_val_settings(self, sets):
        other = build_a1_datasets(DataConfig(n_val=17, val_seed=99))
        assert torch.equal(other["train"].x, sets["train"].x)

    def test_prefix_stability(self):
        # A smaller set with the same seed is a prefix of the larger one.
        cfg = DataConfig()
        small = make_uniform_set("s", 10, 0, cfg)
        big = make_uniform_set("b", 3000, 0, cfg)
        assert torch.equal(small.x, big.x[:10])

    def test_global_rng_state_is_untouched(self):
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        t0, n0, p0 = torch.get_rng_state(), np.random.get_state(), random.getstate()
        build_a1_datasets()
        assert torch.equal(torch.get_rng_state(), t0)
        n1 = np.random.get_state()
        assert n1[0] == n0[0] and np.array_equal(n1[1], n0[1]) and n1[2:] == n0[2:]
        assert random.getstate() == p0

    def test_fingerprint_detects_one_ulp_change(self, sets):
        d = sets["train"]
        x = d.x.clone()
        x[1234, 0] = torch.nextafter(x[1234, 0], torch.tensor(10.0))
        changed = A1Dataset(d.name, x, x.abs(), d.seed)
        assert changed.fingerprint() != d.fingerprint()


# --------------------------------------------------------------- manifest
class TestManifest:
    def test_manifest_is_json_and_complete(self):
        m = dataset_manifest()
        text = json.dumps(m)
        back = json.loads(text)
        assert back["config"] == DataConfig().to_dict()
        assert set(back["sets"]) == {"train", "val", "probe"}
        for k, s in back["sets"].items():
            assert s["sha256"] == GOLDEN_SHA256[k]
        assert back["sets"]["train"]["n"] == 3000
        assert back["sets"]["train"]["seed"] == 0
        assert back["sets"]["val"]["seed"] == 1


# ------------------------------------------------------------- validation
class TestConfigValidation:
    @pytest.mark.parametrize(
        "kw",
        [
            {"n_train": 0},
            {"n_val": -1},
            {"n_probe": 1},
            {"n_train": 3000.0},
            {"n_train": True},
            {"train_seed": -1},
            {"train_seed": 2**32},
            {"train_seed": "0"},
            {"low": 1.0, "high": 1.0},
            {"low": 2.0, "high": 1.0},
            {"low": float("nan")},
            {"high": float("inf")},
            {"low": True},
        ],
    )
    def test_invalid_values_rejected(self, kw):
        with pytest.raises(ValueError):
            DataConfig(**kw)

    def test_equal_train_and_val_seed_rejected(self):
        with pytest.raises(ValueError, match="differ"):
            DataConfig(train_seed=5, val_seed=5)

    def test_unknown_key_in_file_rejected(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"n_train": 3000, "n_trian": 10}), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown"):
            load_data_config(p)

    def test_partial_file_uses_defaults(self, tmp_path):
        p = tmp_path / "part.json"
        p.write_text(json.dumps({"n_val": 500}), encoding="utf-8")
        assert load_data_config(p) == DataConfig(n_val=500)

    def test_config_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DataConfig().n_train = 10
