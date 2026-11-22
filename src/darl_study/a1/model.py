"""Deep narrow MLP for experiment A1 (Lu et al., Section 5).

Lu et al. count layers as affine maps: an L-layer network has L-1 hidden
layers (their eq. 2.1). The A1 network is "10-layer, width 2":

    x (1) -> [Linear(1,2) -> act] -> 8 x [Linear(2,2) -> act] -> Linear(2,1) -> y

i.e. 10 Linear layers, 9 hidden activation layers of width 2, 55 parameters.
Width 2 = d_in + d_out, the width Lu et al. use for all layers.

Each hidden layer gets its OWN activation instance. For DARL this is required:
each layer keeps its own deadness state D.

Activations
-----------
* "relu"        nn.ReLU
* "leaky_relu"  nn.LeakyReLU(negative_slope)   (slope must be given explicitly)
* "darl"        the verified Stage 1 DARL module, unchanged. Defaults below are
                the approved A1 values; D(0) = 0; check_finite on.

The DARL state is updated only when the training loop wraps its forward in
``darl_tracking(model)`` (Stage 1 decision D4). Nothing in this file does that.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from darl_study.activations import DARL

__all__ = [
    "ACTIVATIONS",
    "DARL_A1_DEFAULTS",
    "make_activation",
    "DeepNarrowMLP",
]

ACTIVATIONS = ("relu", "leaky_relu", "darl")

# Approved A1 DARL parameters (Stage 2 brief): alpha_min 0.01, alpha_max 0.3,
# p 1.0, rho 0.99, D(0) = 0.
DARL_A1_DEFAULTS: Dict[str, Any] = {
    "alpha_min": 0.01,
    "alpha_max": 0.3,
    "p": 1.0,
    "rho": 0.99,
    "d_init": 0.0,
    "check_finite": True,
}


def make_activation(name: str, width: int, **kwargs: Any) -> nn.Module:
    """Build one hidden-layer activation of the given width."""
    if name == "relu":
        if kwargs:
            raise ValueError(f"relu takes no arguments, got {sorted(kwargs)}")
        return nn.ReLU()
    if name == "leaky_relu":
        if set(kwargs) != {"negative_slope"}:
            raise ValueError("leaky_relu needs exactly one argument: negative_slope")
        slope = float(kwargs["negative_slope"])
        if not 0.0 < slope <= 1.0:
            raise ValueError(f"negative_slope must be in (0, 1], got {slope}")
        return nn.LeakyReLU(negative_slope=slope)
    if name == "darl":
        unknown = set(kwargs) - set(DARL_A1_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown DARL arguments: {sorted(unknown)}")
        params = {**DARL_A1_DEFAULTS, **kwargs}
        return DARL(width, **params)
    raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")


class DeepNarrowMLP(nn.Module):
    """Fully connected network with ``n_layers`` affine maps of equal width.

    Args:
        n_layers: number of Linear layers L (Lu's convention). Hidden layers = L - 1.
        width: hidden width N.
        d_in, d_out: input and output dimension.
        activation: "relu", "leaky_relu" or "darl".
        act_kwargs: arguments for the activation (see ``make_activation``).
    """

    def __init__(
        self,
        n_layers: int = 10,
        width: int = 2,
        d_in: int = 1,
        d_out: int = 1,
        activation: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        for name, v in (("n_layers", n_layers), ("width", width), ("d_in", d_in), ("d_out", d_out)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2 (at least one hidden layer)")
        act_kwargs = dict(act_kwargs or {})

        self.n_layers = n_layers
        self.width = width
        self.d_in = d_in
        self.d_out = d_out
        self.activation_name = activation
        self.act_kwargs = act_kwargs

        dims = [d_in] + [width] * (n_layers - 1) + [d_out]
        self.linears = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))
        self.acts = nn.ModuleList(make_activation(activation, width, **act_kwargs) for _ in range(n_layers - 1))

    @property
    def n_hidden(self) -> int:
        return self.n_layers - 1

    def forward_with_preactivations(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return (output, [z_1, ..., z_{L-1}]) with z_l the hidden PREactivations.

        This is the same single forward pass as ``forward``: DARL sees each z_l
        exactly once, so a tracked call updates each DARL state exactly once.
        """
        if x.dim() != 2 or x.shape[1] != self.d_in:
            raise ValueError(f"expected input of shape (n, {self.d_in}), got {tuple(x.shape)}")
        h = x
        pre: List[torch.Tensor] = []
        for lin, act in zip(self.linears[:-1], self.acts):
            z = lin(h)
            pre.append(z)
            h = act(z)
        return self.linears[-1](h), pre

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_preactivations(x)[0]

    def darl_layers(self) -> List[DARL]:
        return [m for m in self.acts if isinstance(m, DARL)]

    def describe(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "n_hidden": self.n_hidden,
            "width": self.width,
            "d_in": self.d_in,
            "d_out": self.d_out,
            "activation": self.activation_name,
            "act_kwargs": (
                {**DARL_A1_DEFAULTS, **self.act_kwargs}
                if self.activation_name == "darl"
                else dict(self.act_kwargs)
            ),
            "n_params": sum(p.numel() for p in self.parameters()),
        }
