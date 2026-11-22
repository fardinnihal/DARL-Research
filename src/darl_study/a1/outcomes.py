"""Network-level outcome definitions for experiment A1.

Lu et al. classify trained networks for f(x) = |x| by eye into three shapes
(Fig. 6): A collapse (constant function), B half-trained (one side of the V
fitted, the other flat), C success (the full V). This module turns those
into explicit, labelled NUMERICAL PROXIES on the fixed probe grid P, plus two
bins Lu did not need: "other" and "nonfinite".

Threshold status
----------------
The thresholds below are DRAFTS. They are calibrated on ReLU pilot runs only
(output-spread and validation-MSE histograms), written to
``configs/a1/outcomes.json`` with ``"frozen": true``, and committed BEFORE any
main run or any DARL comparison.

Every classification names its run purpose (``RUN_PURPOSES``). Draft
thresholds are accepted only for the explicitly designated purposes "pilot"
and "calibration"; purpose "main" (every research result) refuses them. This
is enforced in ``check_thresholds``, ``load_thresholds(..., purpose=...)`` and
``classify_result(..., purpose=...)``.

Initialization labels (kept separate on purpose; see ``born_dead``)
---------------------
  structural_dead_layer     PREACTIVATION test, any activation: some hidden
                            layer has z <= 0 for every unit on every training
                            input. Comparable across activations as a
                            statement about preactivations only.
  born_dead                 Lu eq. 3.1: the network IS a constant function on
                            the training inputs. Decidable exactly only for
                            ReLU, where a structurally dead layer outputs 0 and
                            makes the network constant. For LeakyReLU and DARL
                            (alpha > 0) a dead layer still passes alpha * z, so
                            the structural test does not imply constancy;
                            born_dead is None ("not applicable"), never copied
                            from the structural test.
  float_constant_output     network output identical in float32 on every
                            training input (numerical; for leaky activations
                            this can be rounding of a tiny signal).
  near_constant_output      output range < const_range_tol (numerical proxy).
  collapsed_at_init         the collapse proxy below, applied at epoch 0;
                            the network-level label comparable across all
                            activations.

Outcome definitions (std = population standard deviation over the probe points)
-------------------
  collapsed (proxy, Lu A)   out_std < collapse_std_frac * target_std on P.
                            target_std = std(|x|) on P = 0.5.
  success (proxy, Lu C)     validation MSE < success_val_mse.
  half-trained (proxy, B)   not collapsed, not success; on one half of P the
                            MSE < half_fit_mse, and on the other half the
                            output std < half_flat_std_frac * that half's
                            target std (also 0.5 on each half).
  other                     finite, none of the above. Never merged.
  nonfinite                 the run stopped on a NaN/Inf. Counted, never dropped.

Precedence: nonfinite, collapsed, success, half-trained, other.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from darl_study.a1.checkpoint import config_hash
from darl_study.a1.model import DeepNarrowMLP

__all__ = [
    "OUTCOMES",
    "RUN_PURPOSES",
    "DRAFT_PURPOSES",
    "check_thresholds",
    "OutcomeThresholds",
    "load_thresholds",
    "born_dead",
    "classify_shape",
    "classify_result",
    "collapse_timeline",
]

OUTCOMES = ("collapsed", "half_trained", "success", "other", "nonfinite")

# Run purposes. Draft (unfrozen) thresholds are allowed only for DRAFT_PURPOSES.
RUN_PURPOSES = ("pilot", "calibration", "main")
DRAFT_PURPOSES = ("pilot", "calibration")

# Activations whose dead layer outputs exactly 0 (so structural => constant).
_ZEROING_ACTIVATIONS = ("relu",)


@dataclasses.dataclass(frozen=True)
class OutcomeThresholds:
    collapse_std_frac: float = 0.01   # collapse: out_std < 0.01 * 0.5 = 0.005
    success_val_mse: float = 1e-3     # 0.4% of Var(|x|) = 0.25
    half_fit_mse: float = 1e-3        # fitted half
    half_flat_std_frac: float = 0.01  # flat half: std < 0.005
    const_range_tol: float = 1e-6     # born-dead numeric proxy
    frozen: bool = False
    frozen_note: str = "DRAFT - to be calibrated on ReLU pilot runs, then frozen"

    def __post_init__(self) -> None:
        for f in ("collapse_std_frac", "success_val_mse", "half_fit_mse",
                  "half_flat_std_frac", "const_range_tol"):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{f} must be a positive finite number, got {v!r}")
        for f in ("collapse_std_frac", "half_flat_std_frac"):
            if getattr(self, f) >= 1:
                raise ValueError(f"{f} must be < 1")
        if not isinstance(self.frozen, bool):
            raise ValueError("frozen must be true or false")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        return config_hash(self.to_dict())


def check_thresholds(th: OutcomeThresholds, purpose: str) -> None:
    """Refuse draft thresholds unless the run purpose explicitly allows them."""
    if purpose not in RUN_PURPOSES:
        raise ValueError(f"unknown run purpose {purpose!r}; choose from {RUN_PURPOSES}")
    if not th.frozen and purpose not in DRAFT_PURPOSES:
        raise ValueError(
            f"outcome thresholds are still DRAFT (frozen=false); purpose {purpose!r} needs "
            f"frozen thresholds. Drafts are allowed only for {DRAFT_PURPOSES}."
        )


def load_thresholds(path: Union[str, Path], require_frozen: bool = False,
                    purpose: Optional[str] = None) -> OutcomeThresholds:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    known = {f.name for f in dataclasses.fields(OutcomeThresholds)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown outcome threshold keys: {sorted(unknown)}")
    missing = known - set(raw) - {"frozen_note"}
    if missing:
        raise ValueError(f"outcome threshold file is missing keys: {sorted(missing)}")
    th = OutcomeThresholds(**raw)
    if purpose is not None:
        check_thresholds(th, purpose)
    if require_frozen and not th.frozen:
        raise ValueError(
            f"{path}: outcome thresholds are still DRAFT (frozen=false). Freeze them "
            "from the ReLU pilot before any main or DARL comparison run."
        )
    return th


# --------------------------------------------------------------- born dead
@torch.no_grad()
def born_dead(model: DeepNarrowMLP, x: torch.Tensor, const_range_tol: float = 1e-6) -> Dict[str, Any]:
    """Initialization labels on inputs x (the training inputs), eval mode.

    Returns separate labels for the structural (preactivation) test and for
    network-level constancy; see the module docstring. ``structural_dead_layer``,
    ``first_dead_layer``, ``out_range`` and ``near_constant_output`` use the same
    definitions as ``scripts/a1_born_dead.py`` (its ``dead_layer``,
    ``first_dead``, ``out_range`` and ``const_output``).
    """
    was_training = model.training
    model.eval()
    try:
        out, pre = model.forward_with_preactivations(x)
    finally:
        model.train(was_training)
    layer_dead = [bool((z <= 0).all()) for z in pre]
    first = next((i + 1 for i, d in enumerate(layer_dead) if d), 0)
    out_range = float(out.max() - out.min())
    if not math.isfinite(out_range):
        raise FloatingPointError("non-finite network output")
    zeroing = model.activation_name in _ZEROING_ACTIVATIONS
    structural = int(first > 0)
    return {
        "activation": model.activation_name,
        "structural_dead_layer": structural,
        "first_dead_layer": first,
        # Lu eq. 3.1, exact for ReLU only; None = not decidable by this test.
        "born_dead": structural if zeroing else None,
        "born_dead_basis": ("relu: structural dead layer implies a constant network"
                            if zeroing else
                            "not applicable: a dead layer still passes alpha*z for this activation"),
        "out_range": out_range,
        "float_constant_output": int(out_range == 0.0),
        "near_constant_output": int(out_range < const_range_tol),
    }


# ----------------------------------------------------------- classification
def classify_shape(shape: Dict[str, float], val_mse: float, th: OutcomeThresholds) -> str:
    """Outcome of a finite trained network from its probe-grid shape statistics."""
    values = [val_mse] + [shape[k] for k in ("out_std", "target_std", "left_mse", "right_mse",
                                             "left_out_std", "right_out_std",
                                             "left_target_std", "right_target_std")]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return "nonfinite"
    if shape["out_std"] < th.collapse_std_frac * shape["target_std"]:
        return "collapsed"
    if val_mse < th.success_val_mse:
        return "success"
    for fit, flat in (("left", "right"), ("right", "left")):
        if (shape[f"{fit}_mse"] < th.half_fit_mse
                and shape[f"{flat}_out_std"] < th.half_flat_std_frac * shape[f"{flat}_target_std"]):
            return "half_trained"
    return "other"


def collapse_timeline(history: List[Dict[str, Any]], th: OutcomeThresholds) -> Dict[str, Any]:
    """When did the network collapse? Uses each record's probe output spread.

    collapsed(t) = shape.out_std < collapse_std_frac * shape.target_std, with
    shape from the Step 5 metrics hook (falls back to probe_out_std / 0.5).
    """
    flags = []
    for rec in history:
        shape = rec.get("shape")
        if shape is not None:
            flags.append(shape["out_std"] < th.collapse_std_frac * shape["target_std"])
        else:
            flags.append(rec["probe_out_std"] < th.collapse_std_frac * 0.5)
    if not flags:
        return {"collapsed_at_init": None, "collapsed_at_end": None,
                "first_collapse_epoch": None, "uncollapsed_after_collapse": None}
    first = next((history[i]["epoch"] for i, f in enumerate(flags) if f), None)
    i_first = flags.index(True) if True in flags else None
    return {
        "collapsed_at_init": bool(flags[0]),
        "collapsed_at_end": bool(flags[-1]),
        "first_collapse_epoch": first,
        "uncollapsed_after_collapse": (None if i_first is None else (not all(flags[i_first:]))),
    }


def classify_result(result: Dict[str, Any], th: OutcomeThresholds,
                    born: Optional[Dict[str, Any]] = None, *, purpose: str) -> Dict[str, Any]:
    """Outcome record for one finished run (a ``train()`` result dict).

    Works from the saved history only, so outcomes can be recomputed from
    result.json files without the model. ``born`` is the optional born-dead
    record of the initial network (``born_dead``). ``purpose`` is required:
    draft thresholds are refused unless it is "pilot" or "calibration".
    """
    check_thresholds(th, purpose)
    history = result.get("history") or []
    status = result.get("status")
    rec: Dict[str, Any] = {
        "status": status,
        "thresholds_hash": th.hash(),
        "thresholds_frozen": th.frozen,
        "purpose": purpose,
        "structural_dead_layer_at_init": None if born is None else born["structural_dead_layer"],
        "born_dead": None if born is None else born["born_dead"],
        "near_constant_output_at_init": None if born is None else born["near_constant_output"],
    }
    rec.update(collapse_timeline(history, th))
    if status == "nonfinite":
        rec["outcome"] = "nonfinite"
        return rec
    if status != "completed":
        raise ValueError(f"cannot classify a run with status {status!r} (not finished)")
    final = history[-1]
    if "shape" not in final:
        raise ValueError("final history record has no 'shape'; train with extra_metrics=a1_extra_metrics")
    rec["outcome"] = classify_shape(final["shape"], final["val_mse"], th)
    rec["final_val_mse"] = final["val_mse"]
    rec["final_out_std"] = final["shape"]["out_std"]
    if rec["outcome"] == "collapsed":
        # Network-level, comparable across activations: collapsed now, not at epoch 0.
        rec["collapsed_during_training"] = int(rec["collapsed_at_init"] is False)
        # Lu's split (5.6% He / 18.3% RAI), defined only where born_dead is decidable.
        if born is not None and born["born_dead"] is not None:
            rec["collapsed_after_training_not_born_dead"] = int(born["born_dead"] == 0)
    return rec


def write_thresholds(path: Union[str, Path], th: OutcomeThresholds) -> None:
    """Write a thresholds file (used when freezing after the pilot)."""
    d = th.to_dict()
    Path(path).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")


def frozen_copy(th: OutcomeThresholds, note: str) -> OutcomeThresholds:
    """Return the same thresholds marked frozen, with a dated note."""
    stamp = _dt.date.today().isoformat()
    return dataclasses.replace(th, frozen=True, frozen_note=f"{note} (frozen {stamp})")
