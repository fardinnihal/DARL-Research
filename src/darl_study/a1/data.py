"""Synthetic datasets for experiment A1 (Lu et al., deep narrow ReLU networks).

Reference task (Lu et al., Section 5): approximate f(x) = |x| from 3,000
training inputs drawn uniformly from [-sqrt(3), sqrt(3)].

Three fixed, deterministic sets are produced:

* train  -- ``n_train`` = 3,000 inputs, x ~ Uniform[low, high), seed ``train_seed``.
            Used for optimisation and for the born-dead test (Lu eq. 3.1 is
            defined on the training inputs).
* val    -- ``n_val`` = 1,000 inputs from the same distribution, separate seed
            ``val_seed``. Used only to REPORT validation loss; never used to
            select models, thresholds or budgets (project addition, not in Lu).
* probe  -- ``n_probe`` = 2,001 evenly spaced points on [low, high], endpoints
            included and x = 0 included exactly. Used for function-shape and
            neuron-inactivity measurements.

Project decisions (approved): one fixed training set is shared by every run,
arm and seed; run seeds vary only initialisation and batch order.

Reproducibility
---------------
Inputs are drawn with ``numpy.random.RandomState`` (MT19937). NumPy freezes
the legacy RandomState stream (NEP 19), so a given seed gives the same
float64 values on every platform and NumPy version. Values are then cast
once to float32, the training dtype. Each set carries a SHA-256 fingerprint
of its little-endian float32 bytes, which the tests compare against values
recorded when this file was written, and which later steps record in every
run's config.

Generation never touches the global torch, NumPy or Python RNG state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

__all__ = [
    "SQRT3",
    "DataConfig",
    "A1Dataset",
    "target_fn",
    "make_uniform_set",
    "make_probe_grid",
    "build_a1_datasets",
    "load_data_config",
    "dataset_manifest",
]

SQRT3 = math.sqrt(3.0)

# Targets available by name. Only |x| is used in A1.
_TARGETS = {"abs": np.abs}


def target_fn(name: str, x: np.ndarray) -> np.ndarray:
    """Apply the named target function elementwise."""
    if name not in _TARGETS:
        raise ValueError(f"unknown target {name!r}; available: {sorted(_TARGETS)}")
    return _TARGETS[name](x)


# ------------------------------------------------------------------ config
@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Dataset settings. Defaults are the A1 reference values."""

    target: str = "abs"
    low: float = -SQRT3
    high: float = SQRT3
    n_train: int = 3000
    n_val: int = 1000
    n_probe: int = 2001
    train_seed: int = 0
    val_seed: int = 1

    def __post_init__(self) -> None:
        if self.target not in _TARGETS:
            raise ValueError(f"unknown target {self.target!r}")
        for name in ("low", "high"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
        if not self.low < self.high:
            raise ValueError(f"require low < high, got low={self.low}, high={self.high}")
        for name in ("n_train", "n_val", "n_probe", "train_seed", "val_seed"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{name} must be an int, got {v!r}")
        for name in ("n_train", "n_val"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.n_probe < 2:
            raise ValueError("n_probe must be >= 2 (both endpoints are included)")
        for name in ("train_seed", "val_seed"):
            if not 0 <= getattr(self, name) < 2**32:
                raise ValueError(f"{name} must be in [0, 2**32)")
        if self.train_seed == self.val_seed:
            raise ValueError("train_seed and val_seed must differ, or validation would copy training data")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_data_config(path: Union[str, Path]) -> DataConfig:
    """Read a JSON data config. Unknown keys are rejected, missing keys use defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("data config must be a JSON object")
    known = {f.name for f in dataclasses.fields(DataConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown data config keys: {sorted(unknown)}")
    return DataConfig(**raw)


# ----------------------------------------------------------------- dataset
@dataclasses.dataclass(frozen=True)
class A1Dataset:
    """A fixed set of (x, y) pairs, both float32 tensors of shape (n, 1)."""

    name: str
    x: torch.Tensor
    y: torch.Tensor
    seed: Union[int, None]  # None for the deterministic probe grid

    def __len__(self) -> int:
        return self.x.shape[0]

    def fingerprint(self) -> str:
        """SHA-256 of the little-endian float32 bytes of x, then y."""
        h = hashlib.sha256()
        for label, t in ((b"x", self.x), (b"y", self.y)):
            h.update(label)
            h.update(t.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
        return h.hexdigest()

    def summary(self) -> Dict[str, Any]:
        x = self.x.double()
        y = self.y.double()
        return {
            "name": self.name,
            "n": len(self),
            "seed": self.seed,
            "dtype": str(self.x.dtype).replace("torch.", ""),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "x_mean": float(x.mean()),
            "x_var": float(x.var(unbiased=False)),
            "y_mean": float(y.mean()),
            "y_var": float(y.var(unbiased=False)),
            "sha256": self.fingerprint(),
        }


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32).reshape(-1, 1).copy())


def make_uniform_set(name: str, n: int, seed: int, cfg: DataConfig) -> A1Dataset:
    """Draw n inputs from Uniform[low, high) with a private RNG, then y = f(x)."""
    rng = np.random.RandomState(seed)  # private stream; global state untouched
    u = rng.random_sample(n)  # float64 in [0, 1)
    x64 = cfg.low + (cfg.high - cfg.low) * u
    x = x64.astype(np.float32)
    y = target_fn(cfg.target, x).astype(np.float32)  # |x| is exact in float32
    return A1Dataset(name=name, x=_to_tensor(x), y=_to_tensor(y), seed=seed)


def make_probe_grid(cfg: DataConfig) -> A1Dataset:
    """Evenly spaced grid on [low, high] with both endpoints.

    For a symmetric interval and an odd point count the grid is built as
    high * k / m for k = -m..m, so it is exactly symmetric and contains 0.
    """
    n = cfg.n_probe
    if cfg.low == -cfg.high and n % 2 == 1:
        m = (n - 1) // 2
        x64 = cfg.high * (np.arange(-m, m + 1, dtype=np.float64) / m)
    else:
        x64 = np.linspace(cfg.low, cfg.high, n, dtype=np.float64)
    x = x64.astype(np.float32)
    y = target_fn(cfg.target, x).astype(np.float32)
    return A1Dataset(name="probe", x=_to_tensor(x), y=_to_tensor(y), seed=None)


def build_a1_datasets(cfg: DataConfig | None = None) -> Dict[str, A1Dataset]:
    """Return {"train", "val", "probe"} for the given config (default: A1 reference)."""
    cfg = DataConfig() if cfg is None else cfg
    return {
        "train": make_uniform_set("train", cfg.n_train, cfg.train_seed, cfg),
        "val": make_uniform_set("val", cfg.n_val, cfg.val_seed, cfg),
        "probe": make_probe_grid(cfg),
    }


def dataset_manifest(cfg: DataConfig | None = None) -> Dict[str, Any]:
    """JSON-serialisable record of the config and every set's summary and hash."""
    cfg = DataConfig() if cfg is None else cfg
    sets = build_a1_datasets(cfg)
    return {
        "config": cfg.to_dict(),
        "generator": f"numpy.random.RandomState (MT19937), numpy {np.__version__}",
        "sets": {k: v.summary() for k, v in sets.items()},
    }


if __name__ == "__main__":  # python -m darl_study.a1.data
    print(json.dumps(dataset_manifest(), indent=2))
