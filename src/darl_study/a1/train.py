"""Training loop for experiment A1 (y = |x|, 10-layer width-2 MLP).

Protocol (approved; Lu et al. settings where they are stated):

* Data: the fixed Step 2 sets. Training on 3,000 inputs; validation (1,000)
  and the probe grid (2,001) are only measured, never used for selection.
* Loss: MSE, mean over the mini-batch (Lu's L2 loss).
* Optimizer: Adam, PyTorch defaults (lr 1e-3, betas (0.9, 0.999), eps 1e-8),
  no weight decay.
* Batches: 128, reshuffled every epoch from the run's "order" seed; the last
  partial batch (56 samples) is kept, so one epoch = 24 optimizer steps.
* DARL: every training forward is wrapped in ``darl_tracking(model)`` exactly
  once per optimizer step, so each DARL layer's EMA updates once per step.
  Measurement passes run in ``model.eval()``, where D is frozen.
* Fail-fast: a non-finite preactivation (DARL ``check_finite``), loss,
  gradient or parameter stops the run with status "nonfinite". The event is
  recorded and kept as an outcome; it is never silently retried.
* Checkpoints at epoch boundaries (atomic). Resuming continues bitwise
  identically on CPU: model, Adam state, DARL state, batch-order RNG, step
  counters and history are all restored.

Measurements in this step: train/validation MSE, probe-output spread,
per-layer gradient norms (full training batch, eval mode) and DARL slope
statistics. Neuron-level inactivity, collapse and recovery metrics are added
in Step 5 through the ``extra_metrics`` hook.
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from darl_study import __version__ as PKG_VERSION
from darl_study.activations import darl_tracking
from darl_study.a1.checkpoint import atomic_write_text, config_hash, load_checkpoint, save_checkpoint
from darl_study.a1.data import DataConfig, build_a1_datasets
from darl_study.a1.init import INIT_SCHEMES, initialize_, make_generator
from darl_study.a1.model import ACTIVATIONS, DARL_A1_DEFAULTS, DeepNarrowMLP

__all__ = [
    "TrainConfig",
    "NonFiniteError",
    "log_epochs",
    "evaluate",
    "train",
    "CHECKPOINT_NAME",
    "RESULT_NAME",
    "CONFIG_NAME",
]

CHECKPOINT_NAME = "checkpoint.pt"
RESULT_NAME = "result.json"
CONFIG_NAME = "config.json"

_mse = nn.functional.mse_loss  # module attribute so tests can inject faults


class NonFiniteError(FloatingPointError):
    """A loss, gradient or parameter became NaN or Inf."""


# ------------------------------------------------------------------ config
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    activation: str = "relu"
    act_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    init: str = "he"
    run_seed: int = 0
    epochs: int = 2000
    batch_size: int = 128
    lr: float = 1e-3
    betas: Sequence[float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    n_layers: int = 10
    width: int = 2
    log_every: int = 10
    checkpoint_every: int = 50
    data: DataConfig = dataclasses.field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.init not in INIT_SCHEMES:
            raise ValueError(f"unknown init {self.init!r}")
        kw = dict(self.act_kwargs)
        if self.activation == "darl":
            kw = {**DARL_A1_DEFAULTS, **kw}  # record the resolved parameters
        object.__setattr__(self, "act_kwargs", kw)
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        for name in ("run_seed", "epochs", "batch_size", "n_layers", "width", "log_every", "checkpoint_every"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{name} must be an int, got {v!r}")
        if self.run_seed < 0:
            raise ValueError("run_seed must be >= 0")
        for name in ("epochs", "batch_size", "log_every", "checkpoint_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("lr", "eps"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a positive finite number")
        if not (len(self.betas) == 2 and all(0.0 <= b < 1.0 for b in self.betas)):
            raise ValueError("betas must be two numbers in [0, 1)")
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError("weight_decay must be >= 0")
        if not isinstance(self.data, DataConfig):
            raise ValueError("data must be a DataConfig")
        # Building the model validates activation arguments and sizes.
        DeepNarrowMLP(self.n_layers, self.width, 1, 1, self.activation, self.act_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["betas"] = list(self.betas)
        d["data"] = self.data.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        if "data" in d:
            d["data"] = DataConfig(**d["data"])
        return cls(**d)

    def hash(self) -> str:
        return config_hash(self.to_dict())


def log_epochs(epochs: int, log_every: int) -> List[int]:
    """Epochs at which history is recorded: 0 (initialization), early
    log-spaced epochs (1, 2, 5, 10, 20, 50, ...), every ``log_every``, and the last."""
    marks = {0, epochs}
    k = 1
    while k <= epochs:
        for m in (1, 2, 5):
            if m * k <= epochs:
                marks.add(m * k)
        k *= 10
    marks.update(range(log_every, epochs + 1, log_every))
    return sorted(marks)


# ------------------------------------------------------------- measuring
def _all_finite(tensors) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors if t is not None)


def evaluate(
    model: DeepNarrowMLP,
    data: Dict[str, Any],
    epoch: int,
    step: int,
    train_loss_epoch: Optional[float] = None,
    darl_alpha_time_mean: Optional[float] = None,
    extra_metrics: Optional[Callable[[DeepNarrowMLP, Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """One measurement record. Runs in eval mode: never updates DARL state."""
    was_training = model.training
    model.eval()
    try:
        train, val, probe = data["train"], data["val"], data["probe"]
        with torch.no_grad():
            val_mse = float(_mse(model(val.x), val.y))
            probe_out = model(probe.x).double()
        # Full-batch gradient on the training set, without touching .grad.
        with torch.enable_grad():
            loss = _mse(model(train.x), train.y)
            params = list(model.parameters())
            grads = torch.autograd.grad(loss, params)
        if not (math.isfinite(val_mse) and bool(torch.isfinite(loss)) and _all_finite(grads)):
            raise NonFiniteError(f"non-finite measurement at epoch {epoch}")
        grad_of = dict(zip((id(p) for p in params), grads))
        layer_norms = []
        for lin in model.linears:
            sq = sum(float(grad_of[id(p)].double().pow(2).sum()) for p in lin.parameters())
            layer_norms.append(math.sqrt(sq))
        rec: Dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "train_mse": float(loss.detach()),
            "train_loss_epoch": train_loss_epoch,
            "val_mse": val_mse,
            "probe_out_std": float(probe_out.std(unbiased=False)),
            "probe_out_range": float(probe_out.max() - probe_out.min()),
            "grad_norm_layers": layer_norms,
            "grad_norm_total": math.sqrt(sum(g * g for g in layer_norms)),
        }
        darl = model.darl_layers()
        if darl:
            rec["darl_D_mean_layers"] = [float(m.D.mean()) for m in darl]
            rec["darl_alpha_mean_layers"] = [float(m.alpha.mean()) for m in darl]
            rec["darl_alpha_mean"] = float(torch.cat([m.alpha for m in darl]).mean())
            rec["darl_alpha_time_mean"] = darl_alpha_time_mean
            rec["darl_num_updates"] = int(darl[0].num_updates)
        if extra_metrics is not None:
            rec.update(extra_metrics(model, data))
        return rec
    finally:
        model.train(was_training)


# ---------------------------------------------------------------- training
def _versions() -> Dict[str, str]:
    return {
        "darl_study": PKG_VERSION,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def train(
    cfg: TrainConfig,
    run_dir: Union[str, Path, None] = None,
    stop_after_epoch: Optional[int] = None,
    extra_metrics: Optional[Callable[[DeepNarrowMLP, Dict[str, Any]], Dict[str, Any]]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Train one (config, seed) run. Resumes from ``run_dir/checkpoint.pt`` if present.

    ``stop_after_epoch`` ends this call early at that epoch (status stays
    "running"), e.g. to split a long run; calling again resumes it.
    Returns the result dict that is also written to ``run_dir/result.json``.
    """
    run_dir = Path(run_dir) if run_dir is not None else None
    chash = cfg.hash()
    data = build_a1_datasets(cfg.data)
    model = DeepNarrowMLP(cfg.n_layers, cfg.width, 1, 1, cfg.activation, cfg.act_kwargs)
    seeds = initialize_(model, cfg.init, cfg.run_seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                           weight_decay=cfg.weight_decay)
    order = make_generator(seeds["order"])
    marks = set(log_epochs(cfg.epochs, cfg.log_every))
    darl = model.darl_layers()

    st: Dict[str, Any] = {
        "epoch": 0, "step": 0, "status": "running", "failure": None,
        "history": [], "alpha_sum": 0.0, "wall_seconds": 0.0,
    }
    ckpt_path = run_dir / CHECKPOINT_NAME if run_dir else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_record = {"config": cfg.to_dict(), "config_hash": chash, "seeds": seeds,
                      "dataset_sha256": {k: v.fingerprint() for k, v in data.items()},
                      "network": model.describe(), "versions": _versions()}
        cfg_file = run_dir / CONFIG_NAME
        if cfg_file.exists():
            if json.loads(cfg_file.read_text(encoding="utf-8")).get("config_hash") != chash:
                raise ValueError(f"{cfg_file} belongs to a different config; refusing to reuse {run_dir}")
        else:
            atomic_write_text(cfg_file, json.dumps(cfg_record, indent=2))
    if ckpt_path is not None and ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, expected_config_hash=chash)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        order.set_state(ck["order_rng"])
        for k in st:
            st[k] = ck["run"][k]

    def checkpoint() -> None:
        if ckpt_path is None:
            return
        save_checkpoint(ckpt_path, {
            "config_hash": chash,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "order_rng": order.get_state(),
            "run": dict(st),
        })

    def alpha_time_mean() -> Optional[float]:
        return st["alpha_sum"] / st["step"] if darl and st["step"] > 0 else None

    t_start = time.perf_counter()
    wall0 = st["wall_seconds"]
    x, y = data["train"].x, data["train"].y
    n, bs = x.shape[0], cfg.batch_size

    if st["status"] == "running" and st["epoch"] == 0 and not st["history"]:
        st["history"].append(evaluate(model, data, 0, 0, None, None, extra_metrics))
        if progress:
            progress(st["history"][-1])

    while st["status"] == "running" and st["epoch"] < cfg.epochs:
        if stop_after_epoch is not None and st["epoch"] >= stop_after_epoch:
            break
        perm = torch.randperm(n, generator=order)
        loss_sum = 0.0
        current_step = st["step"] + 1
        try:
            for i in range(0, n, bs):
                current_step = st["step"] + 1
                idx = perm[i:i + bs]
                xb, yb = x[idx], y[idx]
                model.train()
                opt.zero_grad(set_to_none=True)
                with darl_tracking(model):  # exactly one tracked forward per step
                    pred = model(xb)
                loss = _mse(pred, yb)
                if not torch.isfinite(loss):
                    raise NonFiniteError(f"non-finite loss {float(loss.detach())}")
                loss.backward()
                if not _all_finite(p.grad for p in model.parameters()):
                    raise NonFiniteError("non-finite gradient")
                opt.step()
                st["step"] += 1
                if not _all_finite(model.parameters()):
                    raise NonFiniteError("non-finite parameter after optimizer step")
                loss_sum += float(loss.detach()) * idx.numel()
                if darl:
                    st["alpha_sum"] += float(torch.cat([m.alpha for m in darl]).mean())
        except FloatingPointError as e:  # includes DARL's check and NonFiniteError
            st["status"] = "nonfinite"
            st["failure"] = {"epoch": st["epoch"] + 1, "step": current_step,
                             "error": type(e).__name__, "message": str(e)}
            break
        st["epoch"] += 1
        if st["epoch"] in marks:
            st["history"].append(evaluate(model, data, st["epoch"], st["step"], loss_sum / n,
                                          alpha_time_mean(), extra_metrics))
            if progress:
                progress(st["history"][-1])
        if st["epoch"] == cfg.epochs:
            st["status"] = "completed"
        st["wall_seconds"] = wall0 + time.perf_counter() - t_start
        if st["epoch"] % cfg.checkpoint_every == 0 or st["status"] != "running":
            checkpoint()

    st["wall_seconds"] = wall0 + time.perf_counter() - t_start
    if st["status"] == "nonfinite" or (stop_after_epoch is not None and st["status"] == "running"):
        # A non-finite run is terminal: its checkpoint records the failure and
        # is never trained further, so a resume just returns this result.
        checkpoint()

    result = {
        "config_hash": chash,
        "config": cfg.to_dict(),
        "seeds": seeds,
        "status": st["status"],
        "failure": st["failure"],
        "epochs_completed": st["epoch"],
        "steps_completed": st["step"],
        "darl_num_updates": [int(m.num_updates) for m in darl] if darl else None,
        "darl_alpha_time_mean": alpha_time_mean(),
        "final": st["history"][-1] if st["history"] else None,
        "history": st["history"],
        "wall_seconds": st["wall_seconds"],
        "versions": _versions(),
    }
    if run_dir is not None:
        atomic_write_text(run_dir / RESULT_NAME, json.dumps(result, indent=2, allow_nan=True))
    return result
