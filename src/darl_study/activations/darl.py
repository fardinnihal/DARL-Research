"""DARL: Deadness-Adaptive Rectified Linear Unit.

Implements the formulation in the DARL research proposal, sections 5.1 to 5.3,
without modification:

    r_j(t)     = (1/B) * sum_i 1[z_ij(t) <= 0]                  (5.3, inactivity)
    D_j(t)     = rho * D_j(t-1) + (1 - rho) * r_j(t)            (5.3, EMA state)
    alpha_j(t) = alpha_min + (alpha_max - alpha_min) * D_j(t)^p (5.2, slope)
    DARL_j(x,t) = x            if x >= 0                        (5.1, activation)
                = alpha_j(t)*x if x <  0

Constraints (5.2, 5.3): 0 < alpha_min <= alpha_max <= 1, p > 0, 0 <= rho < 1.

Implementation decisions approved in the roadmap (section 3.4), none of which
change the equations above:

* D1  D(0) = 0 by default (primary). Other initial values can be set with
      ``reset_state`` for the separate calibrated-initialization experiment.
* D2  Literal index reading: the slope applied at step t uses D(t), i.e. the
      state AFTER this mini-batch's update.
* D3  A "neuron" is index ``j`` along ``unit_dim`` (dim 1 by default). For a
      (B, C, H, W) conv tensor this is a channel, and r_j pools over the batch
      and all spatial positions.
* D4  D is updated only in training mode AND only on forward passes that the
      training loop marks with ``darl_tracking(model)``. Metric passes,
      evaluation, and recomputed forwards therefore never change the state.
* D15 D is clamped to [0, 1] after each update (no effect in exact arithmetic)
      and is always stored in float32.

The state D is a buffer, never a trainable parameter, and receives no gradient
(proposal 5.3: "current backward passes treat its value as fixed").
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Sequence

import torch
from torch import nn

__all__ = [
    "DARL",
    "darl_slope",
    "inactive_fraction",
    "darl_tracking",
]


def inactive_fraction(z: torch.Tensor, unit_dim: int = 1) -> torch.Tensor:
    """Per-unit fraction of entries with z <= 0 (proposal eq. for r_j).

    Averages the indicator 1[z <= 0] over every dimension except ``unit_dim``.
    For a (B, n) tensor that is the mini-batch average of section 5.3; for a
    (B, C, H, W) tensor it also averages over spatial positions (decision D3).

    Returns a float32 tensor of shape (z.shape[unit_dim],).
    ``-0.0 <= 0`` is True, so negative zeros count as inactive. NaN compares
    False, so a NaN entry counts as active (see ``DARL(check_finite=True)``).
    """
    unit_dim = unit_dim % z.dim()
    reduce_dims = [d for d in range(z.dim()) if d != unit_dim]
    return (z <= 0).to(torch.float32).mean(dim=reduce_dims)


def darl_slope(
    D: torch.Tensor, alpha_min: float, alpha_max: float, p: float
) -> torch.Tensor:
    """Negative-side slope alpha = alpha_min + (alpha_max - alpha_min) * D**p."""
    return alpha_min + (alpha_max - alpha_min) * D.pow(p)


class DARL(nn.Module):
    """Deadness-Adaptive Rectified Linear Unit (stateful, per-unit slopes).

    Args:
        num_units: number of neurons (MLP) or channels (CNN) along ``unit_dim``.
        alpha_min, alpha_max: slope bounds, 0 < alpha_min <= alpha_max <= 1.
        p: exponent of the slope rule, p > 0.
        rho: EMA momentum, 0 <= rho < 1.
        unit_dim: tensor dimension that indexes units (1 for (B, n) and (B, C, H, W)).
        d_init: initial value of every D_j, in [0, 1]. Default 0 (decision D1).
        check_finite: if True, raise when the input contains NaN or Inf.
    """

    _EXTRA_STATE_VERSION = 1

    def __init__(
        self,
        num_units: int,
        alpha_min: float = 0.01,
        alpha_max: float = 0.3,
        p: float = 1.0,
        rho: float = 0.99,
        unit_dim: int = 1,
        d_init: float = 0.0,
        check_finite: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(num_units, int) or num_units < 1:
            raise ValueError(f"num_units must be a positive int, got {num_units!r}")
        if not (0.0 < alpha_min <= alpha_max <= 1.0):
            raise ValueError(
                "require 0 < alpha_min <= alpha_max <= 1, got "
                f"alpha_min={alpha_min}, alpha_max={alpha_max}"
            )
        if not p > 0.0:
            raise ValueError(f"require p > 0, got p={p}")
        if not (0.0 <= rho < 1.0):
            raise ValueError(f"require 0 <= rho < 1, got rho={rho}")
        if not (0.0 <= d_init <= 1.0):
            raise ValueError(f"require 0 <= d_init <= 1, got d_init={d_init}")

        self.num_units = num_units
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.p = float(p)
        self.rho = float(rho)
        self.unit_dim = int(unit_dim)
        self.check_finite = bool(check_finite)

        # Training loop sets this through ``darl_tracking`` (decision D4).
        self.update_enabled = False

        self.register_buffer(
            "D", torch.full((num_units,), float(d_init), dtype=torch.float32)
        )
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------ state
    @property
    def alpha(self) -> torch.Tensor:
        """Current per-unit negative slopes (float32, detached)."""
        return darl_slope(self.D, self.alpha_min, self.alpha_max, self.p).detach()

    @torch.no_grad()
    def reset_state(self, value: float | torch.Tensor = 0.0) -> None:
        """Set D to ``value`` (scalar or per-unit tensor) and the counter to 0."""
        value_t = torch.as_tensor(value, dtype=torch.float32, device=self.D.device)
        if value_t.numel() not in (1, self.num_units):
            raise ValueError("value must be a scalar or have num_units entries")
        if torch.any(value_t < 0) or torch.any(value_t > 1):
            raise ValueError("D values must lie in [0, 1]")
        self.D.copy_(value_t.expand_as(self.D))
        self.num_updates.zero_()

    @torch.no_grad()
    def _update_state(self, z: torch.Tensor) -> None:
        r = inactive_fraction(z, self.unit_dim)  # float32, shape (num_units,)
        self.D.mul_(self.rho).add_((1.0 - self.rho) * r)
        self.D.clamp_(0.0, 1.0)  # decision D15
        self.num_updates.add_(1)

    def _apply(self, fn, *args: Any, **kwargs: Any):  # keep D in float32
        module = super()._apply(fn, *args, **kwargs)
        if self.D.dtype != torch.float32:
            self.D = self.D.to(torch.float32)
        return module

    # ---------------------------------------------------------------- forward
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() < 2:
            raise ValueError("DARL expects a batched input with at least 2 dims")
        unit_dim = self.unit_dim % z.dim()
        if z.shape[unit_dim] != self.num_units:
            raise ValueError(
                f"input has {z.shape[unit_dim]} units along dim {unit_dim}, "
                f"module was built for {self.num_units}"
            )
        if self.check_finite and not torch.isfinite(z).all():
            raise FloatingPointError("DARL received non-finite preactivations")

        # Proposal 5.3: update the EMA with this mini-batch, in training only.
        if self.training and self.update_enabled:
            self._update_state(z.detach())

        # Proposal 5.2: slope from the (just updated) state D(t); no gradient.
        with torch.no_grad():
            alpha = darl_slope(self.D, self.alpha_min, self.alpha_max, self.p)
        shape = [1] * z.dim()
        shape[unit_dim] = self.num_units
        alpha = alpha.to(dtype=z.dtype).view(shape)

        # Proposal 5.1: x if x >= 0, alpha * x if x < 0.
        return torch.where(z >= 0, z, alpha * z)

    # ------------------------------------------------------------ checkpoint
    def _hparams(self) -> Dict[str, Any]:
        return {
            "num_units": self.num_units,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "p": self.p,
            "rho": self.rho,
            "unit_dim": self.unit_dim,
        }

    def get_extra_state(self) -> Dict[str, Any]:
        """Saved inside ``state_dict`` so a checkpoint records the hyperparameters."""
        return {"version": self._EXTRA_STATE_VERSION, "hparams": self._hparams()}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        saved = state.get("hparams", {})
        current = self._hparams()
        if saved != current:
            raise ValueError(
                "DARL checkpoint hyperparameters do not match this module: "
                f"saved={saved}, current={current}"
            )

    def extra_repr(self) -> str:
        return (
            f"num_units={self.num_units}, alpha_min={self.alpha_min}, "
            f"alpha_max={self.alpha_max}, p={self.p}, rho={self.rho}, "
            f"unit_dim={self.unit_dim}"
        )


def _darl_modules(models: Sequence[nn.Module] | nn.Module) -> Iterator[DARL]:
    if isinstance(models, nn.Module):
        models = [models]
    for model in models:
        for module in model.modules():
            if isinstance(module, DARL):
                yield module


@contextlib.contextmanager
def darl_tracking(models: Sequence[nn.Module] | nn.Module, enabled: bool = True):
    """Mark the enclosed forward passes as training steps that update D (D4).

    Usage in a training loop::

        model.train()
        with darl_tracking(model):
            loss = criterion(model(x), y)
        loss.backward()

    State is still never updated in eval mode, even inside this context.
    """
    modules = list(_darl_modules(models))
    previous = [m.update_enabled for m in modules]
    try:
        for m in modules:
            m.update_enabled = enabled
        yield
    finally:
        for m, prev in zip(modules, previous):
            m.update_enabled = prev
