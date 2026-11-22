"""Neuron-level measurements for experiment A1.

Every definition here uses the hidden PREactivations z = W h + b, never the
activation outputs, so it means the same for ReLU, LeakyReLU and DARL. A DARL
or LeakyReLU unit with z < 0 still outputs alpha * z != 0; that output never
makes a unit count as active or recovered.

Measured per hidden unit j of layer l (18 units for the A1 network):

  q          inactive fraction on the probe grid P: share of points with
             z <= 0 (Stage 1 ``inactive_fraction``; z = 0 and -0.0 count as
             inactive, matching the DARL EMA).
  dead       consistently inactive on P: q == 1 (z <= 0 at every probe point).
  dead_train consistently inactive on the 3,000 training inputs (the set Lu's
             born-dead definition uses).
  zero_out   activation output exactly 0 at every probe point. Equals ``dead``
             for ReLU; for LeakyReLU/DARL it needs z == 0 exactly. Reported
             separately, never used as the dead test.
  grad_z     mean |dL/dz_j| over the training set (full batch, eval mode).
             grad_z > 0 means the unit receives gradient. A dead DARL unit
             still receives alpha-scaled gradient; that is gradient flow, not
             recovery.
  z_range    max - min of z_j over P. A unit downstream of a dead ReLU layer
             has constant z: "active" by sign but carries no signal.

Recovery (``recovery_summary``) is computed across checkpoints: a unit that
was dead at an earlier checkpoint and later has q < 1, i.e. a genuinely
positive preactivation at some probe point.

All passes run in eval mode with D frozen, and use autograd.grad, so the
model's parameters, .grad fields and DARL state are left untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from darl_study.activations import inactive_fraction
from darl_study.a1.model import DeepNarrowMLP

__all__ = [
    "CONSTANT_TOL",
    "neuron_snapshot",
    "function_shape",
    "a1_extra_metrics",
    "recovery_summary",
]

CONSTANT_TOL = 1e-6  # z_range below this = constant unit (numerical proxy)

_mse = torch.nn.functional.mse_loss


def _as_lists(t: torch.Tensor) -> List[List[Any]]:
    return t.tolist()


def neuron_snapshot(model: DeepNarrowMLP, data: Dict[str, Any]) -> Dict[str, Any]:
    """Per-unit inactivity, deadness, zero output, gradient and DARL state."""
    was_training = model.training
    model.eval()
    try:
        probe, train = data["probe"], data["train"]
        with torch.no_grad():
            _, pre_p = model.forward_with_preactivations(probe.x)
            _, pre_t = model.forward_with_preactivations(train.x)
            q = torch.stack([inactive_fraction(z) for z in pre_p])            # (L-1, N)
            dead = q == 1.0
            dead_train = torch.stack([(z <= 0).all(dim=0) for z in pre_t])
            zero_out = torch.stack([(act(z) == 0).all(dim=0) for act, z in zip(model.acts, pre_p)])
            z_range = torch.stack([z.max(dim=0).values - z.min(dim=0).values for z in pre_p])
        with torch.enable_grad():
            out, pre = model.forward_with_preactivations(train.x)
            loss = _mse(out, train.y)
            gz = torch.autograd.grad(loss, pre)
        grad_z = torch.stack([g.abs().mean(dim=0) for g in gz]).double()

        n_units = q.numel()
        snap: Dict[str, Any] = {
            "q": _as_lists(q.double()),
            "dead": _as_lists(dead.int()),
            "dead_train": _as_lists(dead_train.int()),
            "zero_out": _as_lists(zero_out.int()),
            "grad_z": _as_lists(grad_z),
            "z_range": _as_lists(z_range.double()),
            "inactive_frac_mean": float(q.double().mean()),
            "inactive_frac_layers": [float(v) for v in q.double().mean(dim=1)],
            "dead_count": int(dead.sum()),
            "dead_frac": float(dead.sum()) / n_units,
            "dead_count_layers": [int(v) for v in dead.sum(dim=1)],
            "dead_train_count": int(dead_train.sum()),
            "zero_out_count": int(zero_out.sum()),
            "grad_receiving_count": int((grad_z > 0).sum()),
            "constant_unit_count": int((z_range < CONSTANT_TOL).sum()),
            # Preactivation statement only; NOT network collapse (see outcomes.py).
            "structural_dead_layer": int(bool(dead_train.all(dim=1).any())),
        }
        darl = model.darl_layers()
        if darl:
            snap["darl_D"] = [m.D.double().tolist() for m in darl]
            snap["darl_alpha"] = [m.alpha.double().tolist() for m in darl]
        return snap
    finally:
        model.train(was_training)


def function_shape(model: DeepNarrowMLP, data: Dict[str, Any]) -> Dict[str, Any]:
    """Output statistics on the probe grid, overall and for each half (x < 0, x > 0).

    x = 0 belongs to neither half. Used by the outcome classifier.
    """
    was_training = model.training
    model.eval()
    try:
        x, y = data["probe"].x.double(), data["probe"].y.double()
        with torch.no_grad():
            out = model(data["probe"].x).double()
        rec: Dict[str, Any] = {
            "out_std": float(out.std(unbiased=False)),
            "out_range": float(out.max() - out.min()),
            "out_mean": float(out.mean()),
            "target_std": float(y.std(unbiased=False)),
            "probe_mse": float(((out - y) ** 2).mean()),
        }
        for side, mask in (("left", x < 0), ("right", x > 0)):
            o, t = out[mask], y[mask]
            rec[f"{side}_mse"] = float(((o - t) ** 2).mean())
            rec[f"{side}_out_std"] = float(o.std(unbiased=False))
            rec[f"{side}_target_std"] = float(t.std(unbiased=False))
        return rec
    finally:
        model.train(was_training)


def a1_extra_metrics(model: DeepNarrowMLP, data: Dict[str, Any]) -> Dict[str, Any]:
    """Hook for ``train(..., extra_metrics=a1_extra_metrics)``."""
    return {"neurons": neuron_snapshot(model, data), "shape": function_shape(model, data)}


def recovery_summary(
    q_history: Sequence[Sequence[Sequence[float]]],
    z_range_history: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    constant_tol: float = CONSTANT_TOL,
) -> Dict[str, Any]:
    """Dead-unit recovery across checkpoints.

    ``q_history[t][l][j]`` is the inactive fraction of unit (l, j) at checkpoint
    t (in time order). A unit is dead at t if q == 1.

    * dead_ever           dead at any checkpoint
    * recovery_eligible   dead at some checkpoint before the last one
    * recovered           eligible, and q < 1 at some LATER checkpoint
                          (a positive preactivation at some probe point)
    * recovered_alive_end recovered and not dead at the last checkpoint
    * recovered_nonconstant  recovered with z_range >= tol at the checkpoint
                          where it was first seen alive again (only if
                          ``z_range_history`` is given)
    * recovery_rate       recovered / recovery_eligible, or None if no unit
                          was ever eligible (reported as a count, never 0 or 1)
    """
    if len(q_history) == 0:
        raise ValueError("q_history is empty")
    q = torch.tensor([list(map(list, s)) for s in q_history], dtype=torch.float64)  # (T, L, N)
    if q.dim() != 3:
        raise ValueError("q_history must be a sequence of (layers x units) arrays")
    if ((q < 0) | (q > 1) | torch.isnan(q)).any():
        raise ValueError("q values must lie in [0, 1]")
    T = q.shape[0]
    dead = q == 1.0
    eligible = dead[:-1].any(dim=0) if T > 1 else torch.zeros_like(dead[0])
    recovered = torch.zeros_like(eligible)
    first_back = torch.full(eligible.shape, -1, dtype=torch.long)
    for l in range(q.shape[1]):
        for j in range(q.shape[2]):
            seen_dead = False
            for t in range(T):
                if dead[t, l, j]:
                    seen_dead = True
                elif seen_dead:
                    recovered[l, j] = True
                    first_back[l, j] = t
                    break
    alive_end = ~dead[-1]
    out: Dict[str, Any] = {
        "n_checkpoints": T,
        "dead_at_first": int(dead[0].sum()),
        "dead_at_last": int(dead[-1].sum()),
        "dead_ever": int(dead.any(dim=0).sum()),
        "recovery_eligible": int(eligible.sum()),
        "recovered": int(recovered.sum()),
        "recovered_alive_end": int((recovered & alive_end).sum()),
        "recovery_rate": (float(recovered.sum()) / float(eligible.sum())) if int(eligible.sum()) > 0 else None,
        "recovered_units": [[int(v) for v in row] for row in recovered.int().tolist()],
    }
    if z_range_history is not None:
        zr = torch.tensor([list(map(list, s)) for s in z_range_history], dtype=torch.float64)
        if zr.shape != q.shape:
            raise ValueError("z_range_history must have the same shape as q_history")
        nonconst = 0
        for l in range(q.shape[1]):
            for j in range(q.shape[2]):
                t = int(first_back[l, j])
                if t >= 0 and zr[t, l, j] >= constant_tol:
                    nonconst += 1
        out["recovered_nonconstant"] = nonconst
    return out
